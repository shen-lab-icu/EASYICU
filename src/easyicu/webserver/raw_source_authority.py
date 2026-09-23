"""Host-only binding from a registered export to its verified raw source tables.

Modern EasyICU exports carry ``data_path`` in their manifest: the sealed raw
root the export was built from.  ``resolve_raw_hospital_source_binding`` binds
such an export to the raw tables its database needs for hospital-mortality
follow-up (MIMIC-IV ``icustays`` + ``admissions``; eICU ``patient``), digesting
and schema-checking them at bind time.  Older prepared exports did not record
``data_path``, so they cannot safely be paired with a nearby raw directory by
filename guessing.  For those, a private, digest-bound host configuration binds
one exact export manifest to one raw MIMIC-IV root (the legacy migration path,
``resolve_raw_mimic_iv_source_binding``).  Browser and provider projections
receive only digests and an authority reference, never filesystem paths or
identifiers.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq

from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    HospitalMortalityFollowup,
    derive_eicu_hospital_mortality_followup,
    derive_mimic_iv_hospital_mortality_followup,
)
from easyicu.research_agent.authority.filesystem import (
    AnchoredDirectory,
    AuthorityFilesystemError,
)
from easyicu.webserver import state_paths

if TYPE_CHECKING:  # pragma: no cover - typing only
    from easyicu.research_agent.acquisition.patient_grouping import (
        DerivedPatientGrouping,
    )


_PREFIX = "EASYICU_RAW_SOURCE_"
_DEFAULT_CONFIG_PATH = state_paths.state_root() / "raw-source-authority.env"
_FIELDS = (
    "EXPORT_ROOT",
    "EXPORT_MANIFEST",
    "EXPORT_MANIFEST_SHA256",
    "DATABASE",
    "SOURCE_ROOT",
    "ICUSTAYS_FILE",
    "ICUSTAYS_SHA256",
    "ADMISSIONS_FILE",
    "ADMISSIONS_SHA256",
    "AUTHORITY_REF",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RAW_TABLE_BYTES = 128 * 1024 * 1024
_MAX_EXPORT_MANIFEST_BYTES = 16 * 1024 * 1024
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXPORT_MANIFEST_NAMES = ("_manifest.json", "easyicu_export_manifest.json")
_DATABASE_ALIASES = {
    "miiv": "mimic_iv",
    "mimiciv": "mimic_iv",
    "mimic_iv": "mimic_iv",
    "mimic-iv": "mimic_iv",
    "eicu": "eicu",
    "eicu_demo": "eicu",
    "eicu-demo": "eicu",
}
_REQUIRED_TABLE_COLUMNS = {
    "icustays": frozenset({"stay_id", "hadm_id", "intime"}),
    "admissions": frozenset(
        {"hadm_id", "dischtime", "deathtime", "hospital_expire_flag"}
    ),
    "patient": frozenset(
        {"patientunitstayid", "hospitaldischargeoffset", "hospitaldischargestatus"}
    ),
}
# Columns read when present; they inform receipts but never gate a binding.
# ``uniquepid`` is eICU's official patient identifier.  It is optional here on
# purpose: hospital-mortality follow-up never needs it, so a source without it
# must still bind.  The patient-grouping resolver below fails closed instead.
_OPTIONAL_TABLE_COLUMNS = {
    "patient": frozenset({"unitdischargeoffset", "uniquepid"})
}
# Which official table carries each family's stay-to-patient identity, and the
# exact columns the bridge is derived from.  A family absent here has no
# official patient identifier reachable from its bound raw source.
_PATIENT_GROUPING_PROFILES = {
    "eicu": {
        "table": "patient",
        "stay_column": "patientunitstayid",
        "patient_column": "uniquepid",
    },
}
# Which raw tables each database family needs for hospital-mortality follow-up
# and the materializer name its public receipt advertises.
_HOSPITAL_SOURCE_PROFILES = {
    "mimic_iv": {
        "tables": ("icustays", "admissions"),
        "materializer": "mimic_iv_hospital_death_or_discharge_censor",
    },
    "eicu": {
        "tables": ("patient",),
        "materializer": "eicu_hospital_death_or_discharge_censor",
    },
}


class RawSourceAuthorityError(ValueError):
    """A private registered-export/raw-source binding is invalid or stale."""

    def __init__(self, code: str, message: str, *, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return _DATABASE_ALIASES.get(normalized, normalized)


def _read_private_config(path: Path) -> dict[str, str]:
    """Read a strict 0600 JSON-value environment file, if configured."""

    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RawSourceAuthorityError(
            "raw_source_authority_config_unreadable",
            "The private raw-source authority configuration cannot be read.",
        ) from exc
    if not path.is_file() or mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise RawSourceAuthorityError(
            "raw_source_authority_config_insecure",
            "The private raw-source authority configuration must be a 0600 file.",
        )
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RawSourceAuthorityError(
            "raw_source_authority_config_unreadable",
            "The private raw-source authority configuration cannot be read.",
        ) from exc
    allowed = {f"{_PREFIX}{name}" for name in _FIELDS}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, encoded = line.split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            values[key] = value
    return values


def _regular_directory(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise RawSourceAuthorityError(
            "raw_source_authority_directory_invalid",
            f"The configured {label} must be an absolute non-symlink directory.",
            details={"object": label},
        )
    return candidate.resolve(strict=True)


def _component(value: str, *, label: str) -> str:
    if _COMPONENT.fullmatch(value) is None:
        raise RawSourceAuthorityError(
            "raw_source_authority_component_invalid",
            f"The configured {label} must be one safe path component.",
            details={"object": label},
        )
    return value


def _regular_descendant(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts:
        raise RawSourceAuthorityError(
            "raw_source_authority_table_selector_invalid",
            f"The configured {label} must be a relative raw-table path.",
            details={"object": label},
        )
    current = root
    for part in candidate.parts:
        _component(part, label=label)
        current = current / part
        if current.is_symlink():
            raise RawSourceAuthorityError(
                "raw_source_authority_table_symlink_forbidden",
                f"The configured {label} must not traverse a symbolic link.",
                details={"object": label},
            )
    if not current.is_file():
        raise RawSourceAuthorityError(
            "raw_source_authority_table_unavailable",
            f"The configured {label} is unavailable.",
            details={"object": label},
        )
    return current.resolve(strict=True)


def _verify_table_schema(path: Path, *, table: str) -> set[str]:
    """Verify the minimal raw columns before issuing an authority."""

    try:
        columns = set(pq.read_schema(path).names)
    except (OSError, ValueError, ImportError) as exc:
        raise RawSourceAuthorityError(
            "raw_source_authority_table_schema_unreadable",
            "The raw table schema cannot be verified.",
            details={"object": table},
        ) from exc
    missing = sorted(_REQUIRED_TABLE_COLUMNS[table] - columns)
    if missing:
        raise RawSourceAuthorityError(
            "raw_source_authority_table_schema_mismatch",
            "The raw table lacks columns required for hospital-mortality follow-up.",
            details={"object": table, "missing_columns": missing},
        )
    return columns


def _table_read_columns(table: str, available: set[str]) -> list[str]:
    """Required columns plus any receipt-only optional columns the file has."""

    optional = _OPTIONAL_TABLE_COLUMNS.get(table, frozenset()) & available
    return sorted(_REQUIRED_TABLE_COLUMNS[table] | optional)


@dataclass(frozen=True)
class RawMimicIVSourceBinding:
    """One internal raw-source binding; its public view is path-free."""

    source_root: Path
    database: str
    icustays_path: Path
    icustays_sha256: str
    admissions_path: Path
    admissions_sha256: str
    authority_ref: str
    export_manifest_file: str
    export_manifest_sha256: str

    def public_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "easyicu.registered_export_raw_source_authority/1",
            "database": self.database,
            "authority_ref": self.authority_ref,
            "export_manifest_file": self.export_manifest_file,
            "export_manifest_sha256": self.export_manifest_sha256,
            "raw_table_sha256": {
                "icustays": self.icustays_sha256,
                "admissions": self.admissions_sha256,
            },
            "hospital_mortality_followup": {
                "outcome": "death",
                "event_time_column": "death_time_hours",
                "observation_duration_column": "hospital_followup_time_hours",
                "unit": "hours",
                "materializer": "mimic_iv_hospital_death_or_discharge_censor",
            },
            "source_paths_returned": False,
            "identifier_values_returned": False,
        }

    def materialize_hospital_mortality_followup(self) -> HospitalMortalityFollowup:
        """Parse the exact verified bytes, never reopen a validated pathname."""

        icustays = _read_verified_table(
            self.icustays_path, self.icustays_sha256, table="icustays"
        )
        admissions = _read_verified_table(
            self.admissions_path, self.admissions_sha256, table="admissions"
        )
        return derive_mimic_iv_hospital_mortality_followup(icustays, admissions)

    def materialize_hospital_mortality_status(self):
        from easyicu.hospital_mortality import derive_mimic_hospital_mortality_status

        return derive_mimic_hospital_mortality_status(
            _read_verified_table(self.icustays_path, self.icustays_sha256, table="icustays"),
            _read_verified_table(self.admissions_path, self.admissions_sha256, table="admissions"),
        )


def _read_verified_table(
    path: Path,
    expected_sha256: str,
    *,
    table: str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    try:
        with AnchoredDirectory.open(path.parent) as directory:
            payload = directory.read_bytes(path.name, max_bytes=_MAX_RAW_TABLE_BYTES)
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise RawSourceAuthorityError(
                "raw_source_authority_table_digest_mismatch",
                "A raw table changed after its source authority was resolved.",
                details={"object": table},
            )
        return pd.read_parquet(
            io.BytesIO(payload),
            columns=list(columns or sorted(_REQUIRED_TABLE_COLUMNS[table])),
        )
    except (AuthorityFilesystemError, OSError, ValueError, ImportError) as exc:
        if isinstance(exc, RawSourceAuthorityError):
            raise
        raise RawSourceAuthorityError(
            "raw_source_authority_tables_unreadable",
            "The raw table cannot be read through its verified source binding.",
            details={"object": table},
        ) from exc


@dataclass(frozen=True)
class RawHospitalSourceBinding:
    """A modern export bound to its manifest-declared raw tables.

    The public view is path-free and names the database family's materializer;
    the private view keeps the digests taken at bind time so every later read
    re-verifies the exact bytes.
    """

    source_root: Path
    database: str
    family: str
    table_paths: Mapping[str, Path]
    table_sha256: Mapping[str, str]
    table_columns: Mapping[str, tuple[str, ...]]
    export_manifest_file: str
    export_manifest_sha256: str

    @property
    def authority_ref(self) -> str:
        return f"export_manifest_data_path/{self.family}/1"

    def public_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "easyicu.registered_export_raw_source_authority/2",
            "database": self.database,
            "authority_kind": "export_manifest_data_path",
            "authority_ref": self.authority_ref,
            "export_manifest_file": self.export_manifest_file,
            "export_manifest_sha256": self.export_manifest_sha256,
            "raw_table_sha256": dict(self.table_sha256),
            "hospital_mortality_followup": {
                "outcome": "death",
                "event_time_column": "death_time_hours",
                "observation_duration_column": "hospital_followup_time_hours",
                "unit": "hours",
                "materializer": _HOSPITAL_SOURCE_PROFILES[self.family]["materializer"],
            },
            "source_paths_returned": False,
            "identifier_values_returned": False,
            **(
                {"patient_grouping": self.patient_grouping_receipt()}
                if self.patient_grouping_available
                else {}
            ),
        }

    @property
    def patient_grouping_profile(self) -> Optional[Mapping[str, str]]:
        return _PATIENT_GROUPING_PROFILES.get(self.family)

    @property
    def patient_grouping_available(self) -> bool:
        """Whether this bound source can derive a stay-to-patient bridge.

        The profile says which official column carries the identity; the bound
        table's own column list says whether this particular source actually
        has it.  Both must hold, and neither is assumed.
        """

        profile = self.patient_grouping_profile
        if profile is None:
            return False
        table = profile["table"]
        return profile["patient_column"] in set(self.table_columns.get(table, ()))

    def patient_grouping_receipt(self) -> dict[str, Any]:
        profile = self.patient_grouping_profile
        if profile is None:  # pragma: no cover - guarded by the caller
            raise RawSourceAuthorityError(
                "raw_source_authority_patient_grouping_unsupported",
                "This database has no official patient identity table.",
                details={"database": self.database},
            )
        return {
            "cluster_unit": "patient",
            "identity_table": profile["table"],
            "identity_table_sha256": self.table_sha256[profile["table"]],
            "patient_identifier_column": profile["patient_column"],
            "identifier_values_returned": False,
        }

    def materialize_patient_grouping(self) -> "DerivedPatientGrouping":
        """Derive the private stay-to-patient bridge from the bound table."""

        from easyicu.research_agent.acquisition.patient_grouping import (
            PatientGroupingError,
            derive_patient_grouping,
        )

        profile = self.patient_grouping_profile
        if profile is None or not self.patient_grouping_available:
            raise RawSourceAuthorityError(
                "raw_source_authority_patient_grouping_unavailable",
                "The bound raw source carries no official patient identifier.",
                details={"database": self.database},
            )
        table = profile["table"]
        try:
            return derive_patient_grouping(
                self._table(table),
                stay_column=profile["stay_column"],
                patient_column=profile["patient_column"],
                identity_table_name=table,
            )
        except PatientGroupingError as exc:
            raise RawSourceAuthorityError(
                "raw_source_authority_patient_grouping_invalid",
                "The official patient identity table cannot support a grouping.",
                details={"database": self.database, "cause": str(exc)},
            ) from exc

    def _table(self, name: str) -> pd.DataFrame:
        return _read_verified_table(
            self.table_paths[name],
            self.table_sha256[name],
            table=name,
            columns=self.table_columns[name],
        )

    def materialize_hospital_mortality_followup(self) -> HospitalMortalityFollowup:
        if self.family == "eicu":
            return derive_eicu_hospital_mortality_followup(
                self._table("patient"), database=self.database
            )
        return derive_mimic_iv_hospital_mortality_followup(
            self._table("icustays"), self._table("admissions")
        )

    def materialize_hospital_mortality_status(self):
        if self.family != "mimic_iv":
            raise RawSourceAuthorityError(
                "raw_source_authority_status_unsupported",
                "Clock-free hospital status replacement is defined for MIMIC-IV only.",
                details={"database": self.database},
            )
        from easyicu.hospital_mortality import derive_mimic_hospital_mortality_status

        return derive_mimic_hospital_mortality_status(
            self._table("icustays"), self._table("admissions")
        )


def _read_export_manifest(export_root: Path) -> Optional[tuple[str, Path, dict]]:
    """Return the first canonical manifest of a registered export, if any."""

    for name in _EXPORT_MANIFEST_NAMES:
        candidate = export_root / name
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            with AnchoredDirectory.open(export_root) as directory:
                payload = directory.read_bytes(name, max_bytes=_MAX_EXPORT_MANIFEST_BYTES)
            manifest = json.loads(payload.decode("utf-8"))
        except (AuthorityFilesystemError, OSError, ValueError) as exc:
            raise RawSourceAuthorityError(
                "raw_source_authority_export_manifest_unreadable",
                "The registered export manifest cannot be read.",
                details={"object": name},
            ) from exc
        if not isinstance(manifest, dict):
            raise RawSourceAuthorityError(
                "raw_source_authority_export_manifest_invalid",
                "The registered export manifest is not a JSON object.",
                details={"object": name},
            )
        return name, candidate, manifest
    return None


def resolve_manifest_raw_source_binding(
    *,
    export_path: str | Path,
    database: str,
) -> Optional[RawHospitalSourceBinding]:
    """Bind a modern export to the raw tables its manifest ``data_path`` names.

    ``None`` means the export does not carry a sealed ``data_path`` (a legacy
    export, or no export at that path), so the caller may consult the private
    legacy authority instead.  A present but broken coordinate fails closed.
    """

    export_root = Path(export_path).expanduser()
    if not export_root.is_absolute() or export_root.is_symlink() or not export_root.is_dir():
        return None
    export_root = export_root.resolve(strict=True)
    found = _read_export_manifest(export_root)
    if found is None:
        return None
    manifest_name, manifest_path, manifest = found
    data_path = str(manifest.get("data_path") or "").strip()
    if not data_path:
        return None
    selected_database = _database(database)
    manifest_database = _database(manifest.get("database"))
    if not selected_database or selected_database != manifest_database:
        raise RawSourceAuthorityError(
            "raw_source_authority_database_mismatch",
            "The registered export manifest names a different database.",
            details={"database": selected_database},
        )
    profile = _HOSPITAL_SOURCE_PROFILES.get(selected_database)
    if profile is None:
        raise RawSourceAuthorityError(
            "raw_source_authority_database_unsupported",
            "Hospital-mortality follow-up has no raw-table profile for this database.",
            details={"database": selected_database},
        )
    source_root = _regular_directory(Path(data_path), label="raw source root")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    columns: dict[str, tuple[str, ...]] = {}
    for table in profile["tables"]:
        path = _regular_descendant(source_root, f"{table}.parquet", label=f"{table} table")
        available = _verify_table_schema(path, table=table)
        paths[table] = path
        digests[table] = _sha256_file(path)
        columns[table] = tuple(_table_read_columns(table, available))
    return RawHospitalSourceBinding(
        source_root=source_root,
        database=str(database or "").strip().lower(),
        family=selected_database,
        table_paths=paths,
        table_sha256=digests,
        table_columns=columns,
        export_manifest_file=manifest_name,
        export_manifest_sha256=_sha256_file(manifest_path),
    )


def resolve_raw_hospital_source_binding(
    *,
    export_path: str | Path,
    database: str,
) -> Optional["RawHospitalSourceBinding | RawMimicIVSourceBinding"]:
    """Resolve the raw hospital-follow-up source for one registered export.

    A manifest-declared ``data_path`` is the sealed coordinate and takes
    precedence; only an export without one falls back to the private legacy
    MIMIC-IV authority.  Either binding exposes the same path-free receipt and
    materializers, so runtime owners stay database- and vintage-agnostic.
    """

    binding = resolve_manifest_raw_source_binding(
        export_path=export_path, database=database
    )
    if binding is not None:
        return binding
    return resolve_raw_mimic_iv_source_binding(
        export_path=export_path, database=database
    )


def resolve_raw_mimic_iv_source_binding(
    *,
    export_path: str | Path,
    database: str,
    environ: Optional[Mapping[str, str]] = None,
    config_path: Optional[Path] = None,
) -> Optional[RawMimicIVSourceBinding]:
    """Resolve the exact configured raw source for one registered export.

    An absent configuration is intentionally distinct from a broken one: it
    means no migration/source authority has been granted for this legacy
    export.  Partial, selector-mismatched, or digest-drifted authorities fail
    closed.
    """

    if environ is None:
        env = {
            **_read_private_config(Path(config_path or _DEFAULT_CONFIG_PATH)),
            **{
                key: value
                for key, value in os.environ.items()
                if key.startswith(_PREFIX)
            },
        }
    else:
        env = environ
    values = {name: str(env.get(f"{_PREFIX}{name}") or "").strip() for name in _FIELDS}
    present = {name for name, value in values.items() if value}
    if not present:
        return None
    missing = sorted(set(_FIELDS) - present)
    if missing:
        raise RawSourceAuthorityError(
            "raw_source_authority_incomplete",
            "The configured raw-source authority is incomplete.",
            details={"missing_fields": missing},
        )

    selected_selector = Path(export_path).expanduser().resolve(strict=False)
    configured_selector = Path(values["EXPORT_ROOT"]).expanduser().resolve(
        strict=False
    )
    if selected_selector != configured_selector:
        return None
    selected_database = _database(database)
    configured_database = _database(values["DATABASE"])
    if selected_database != configured_database:
        return None
    if configured_database != "mimic_iv":
        raise RawSourceAuthorityError(
            "raw_source_authority_database_unsupported",
            "The raw hospital-mortality source authority currently supports MIMIC-IV only.",
            details={"database": configured_database},
        )

    export_root = _regular_directory(Path(values["EXPORT_ROOT"]), label="export root")
    if export_root != selected_selector.resolve(strict=True):
        raise RawSourceAuthorityError(
            "raw_source_authority_export_selector_changed",
            "The selected export no longer resolves to the configured raw source.",
            details={"database": selected_database},
        )
    manifest_name = _component(values["EXPORT_MANIFEST"], label="export manifest")
    manifest_path = export_root / manifest_name
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RawSourceAuthorityError(
            "raw_source_authority_export_manifest_unavailable",
            "The configured export manifest is unavailable.",
        )
    expected_manifest_sha = values["EXPORT_MANIFEST_SHA256"]
    if _SHA256.fullmatch(expected_manifest_sha) is None:
        raise RawSourceAuthorityError(
            "raw_source_authority_digest_invalid",
            "The raw-source authority requires lowercase SHA-256 digests.",
        )
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise RawSourceAuthorityError(
            "raw_source_authority_export_manifest_mismatch",
            "The selected export no longer matches the raw-source authority.",
            details={"database": selected_database},
        )

    source_root = _regular_directory(Path(values["SOURCE_ROOT"]), label="raw source root")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name, key in (("icustays", "ICUSTAYS"), ("admissions", "ADMISSIONS")):
        expected = values[f"{key}_SHA256"]
        if _SHA256.fullmatch(expected) is None:
            raise RawSourceAuthorityError(
                "raw_source_authority_digest_invalid",
                "The raw-source authority requires lowercase SHA-256 digests.",
            )
        path = _regular_descendant(
            source_root,
            values[f"{key}_FILE"],
            label=f"{name} table",
        )
        if path.suffix.lower() not in {".parquet", ".pq"}:
            raise RawSourceAuthorityError(
                "raw_source_authority_table_format_unsupported",
                "The raw hospital-mortality source authority currently requires Parquet tables.",
                details={"object": name},
            )
        if _sha256_file(path) != expected:
            raise RawSourceAuthorityError(
                "raw_source_authority_table_digest_mismatch",
                "A raw table no longer matches its source authority digest.",
                details={"object": name, "database": selected_database},
            )
        _verify_table_schema(path, table=name)
        paths[name] = path
        digests[name] = expected

    return RawMimicIVSourceBinding(
        source_root=source_root,
        database="miiv",
        icustays_path=paths["icustays"],
        icustays_sha256=digests["icustays"],
        admissions_path=paths["admissions"],
        admissions_sha256=digests["admissions"],
        authority_ref=values["AUTHORITY_REF"],
        export_manifest_file=manifest_name,
        export_manifest_sha256=expected_manifest_sha,
    )


__all__ = [
    "RawHospitalSourceBinding",
    "RawMimicIVSourceBinding",
    "RawSourceAuthorityError",
    "resolve_manifest_raw_source_binding",
    "resolve_raw_hospital_source_binding",
    "resolve_raw_mimic_iv_source_binding",
]
