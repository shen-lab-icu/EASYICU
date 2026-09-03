"""Host-only binding from a registered export to its verified raw MIMIC-IV source.

Modern EasyICU exports carry ``data_path`` in their manifest.  Older prepared
exports did not, so they cannot safely be paired with a nearby raw directory
by filename guessing.  This owner provides the deliberate migration path: a
private, digest-bound host configuration binds one exact export manifest to one
raw MIMIC-IV root and to the two raw tables required for hospital-mortality
follow-up.  Browser and provider projections receive only digests and an
authority reference, never filesystem paths or identifiers.
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
from typing import Any, Mapping, Optional

import pandas as pd
import pyarrow.parquet as pq

from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    HospitalMortalityFollowup,
    derive_mimic_iv_hospital_mortality_followup,
)
from easyicu.research_agent.authority.filesystem import (
    AnchoredDirectory,
    AuthorityFilesystemError,
)
from easyicu.webserver import state_paths


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
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DATABASE_ALIASES = {
    "miiv": "mimic_iv",
    "mimiciv": "mimic_iv",
    "mimic_iv": "mimic_iv",
    "mimic-iv": "mimic_iv",
}
_REQUIRED_TABLE_COLUMNS = {
    "icustays": frozenset({"stay_id", "hadm_id", "intime"}),
    "admissions": frozenset(
        {"hadm_id", "dischtime", "deathtime", "hospital_expire_flag"}
    ),
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


def _verify_table_schema(path: Path, *, table: str) -> None:
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


def _read_verified_table(path: Path, expected_sha256: str, *, table: str) -> pd.DataFrame:
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
            io.BytesIO(payload), columns=sorted(_REQUIRED_TABLE_COLUMNS[table])
        )
    except (AuthorityFilesystemError, OSError, ValueError, ImportError) as exc:
        if isinstance(exc, RawSourceAuthorityError):
            raise
        raise RawSourceAuthorityError(
            "raw_source_authority_tables_unreadable",
            "The raw table cannot be read through its verified source binding.",
            details={"object": table},
        ) from exc


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
    "RawMimicIVSourceBinding",
    "RawSourceAuthorityError",
    "resolve_raw_mimic_iv_source_binding",
]
