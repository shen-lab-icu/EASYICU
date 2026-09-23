"""Host-only resolver for verified patient-grouping coordinates.

Prepared EasyICU exports intentionally do not expose direct patient identifiers.
Two authorities can supply the bridge, and both keep it host-private: Pi and the
Provider receive only the derived column name and digests, never the mapping
path or its values.

The first is an environment-configured bridge a data owner approved separately
(:func:`resolve_patient_grouping_authority`).  The second applies when the
export already names its sealed raw root and that source carries an official
patient identifier (:func:`resolve_derived_patient_grouping`): eICU's
``patient`` table is bound and digest-verified for hospital-mortality
follow-up, and the same verified bytes carry ``uniquepid``.  Requiring a second
manual approval for a column inside an already-bound table would be ceremony
without a privacy gain, so the host derives the bridge, materializes it into
private state, and binds it by digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Mapping, Optional

import pyarrow.parquet as pq

from easyicu.research_agent.acquisition.patient_grouping import (
    DERIVED_PATIENT_COLUMN,
    DERIVED_STAY_COLUMN,
    PatientGroupingBinding,
)
from easyicu.webserver import state_paths


_PREFIX = "EASYICU_PATIENT_GROUPING_"
_DEFAULT_CONFIG_PATH = state_paths.state_root() / "patient-grouping.env"
_FIELDS = (
    "EXPORT_ROOT",
    "EXPORT_MANIFEST",
    "EXPORT_MANIFEST_SHA256",
    "DATABASE",
    "MAPPING_PATH",
    "MAPPING_SHA256",
    "STAY_COLUMN",
    "PATIENT_COLUMN",
    "AUTHORITY_REF",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# A single path component, never a traversal or an option-like name.  The
# leading character excludes "." and "-"; "_" is allowed because the native
# export manifest is literally "_manifest.json", and an owner-approved
# bridge must be configurable against a modern export.
_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_DATABASE_ALIASES = {
    "miiv": "mimic_iv",
    "mimiciv": "mimic_iv",
    "mimic_iv": "mimic_iv",
    "mimic-iv": "mimic_iv",
}


class PatientGroupingAuthorityError(ValueError):
    """A configured private grouping authority is incomplete or mismatched."""

    def __init__(self, code: str, message: str, *, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _database(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return _DATABASE_ALIASES.get(normalized, normalized)


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_file_invalid",
            f"The configured {label} must be an absolute regular non-symlink file.",
            details={"object": label},
        )
    return candidate.resolve(strict=True)


def _read_private_config(path: Path) -> dict[str, str]:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_config_unreadable",
            "The private patient-grouping authority configuration cannot be read.",
        ) from exc
    if not path.is_file() or mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_config_insecure",
            "The private patient-grouping authority configuration must be a 0600 file.",
        )
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_config_unreadable",
            "The private patient-grouping authority configuration cannot be read.",
        ) from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, encoded = line.split("=", 1)
        key = key.strip()
        if key not in {f"{_PREFIX}{name}" for name in _FIELDS}:
            continue
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            values[key] = value
    return values


def resolve_patient_grouping_authority(
    *,
    export_path: str | Path,
    database: str,
    environ: Optional[Mapping[str, str]] = None,
    config_path: Optional[Path] = None,
) -> Optional[PatientGroupingBinding]:
    """Resolve one exact source-bound patient grouping authority.

    An entirely absent configuration means the source has no grouping authority.
    Any partial, stale, or digest-mismatched configuration fails closed.
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
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_incomplete",
            "The configured patient-grouping authority is incomplete.",
            details={"missing_fields": missing},
        )

    # First compare normalized selectors without requiring either source to
    # exist.  One machine-level authority may be configured while a caller is
    # merely validating a different (or not-yet-mounted) export.  Requiring
    # that unrelated selector to exist would make the private configuration
    # change otherwise deterministic validation results.  Once the selectors
    # match, both are resolved strictly before any authority is issued.
    selected_selector = Path(export_path).expanduser().resolve(strict=False)
    configured_selector = Path(values["EXPORT_ROOT"]).expanduser().resolve(strict=False)
    if selected_selector != configured_selector:
        return None
    selected_database = _database(database)
    configured_database = _database(values["DATABASE"])
    if selected_database != configured_database:
        return None
    try:
        selected_root = Path(export_path).expanduser().resolve(strict=True)
        configured_root = Path(values["EXPORT_ROOT"]).expanduser().resolve(strict=True)
    except OSError as exc:
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_export_unavailable",
            "The export selected by the patient-grouping authority is unavailable.",
            details={"database": selected_database},
        ) from exc
    if selected_root != configured_root:
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_export_selector_changed",
            "The selected export no longer resolves to the configured grouping source.",
            details={"database": selected_database},
        )

    manifest_name = values["EXPORT_MANIFEST"]
    if _COMPONENT.fullmatch(manifest_name) is None:
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_manifest_invalid",
            "The patient-grouping export manifest selector is invalid.",
        )
    manifest_sha256 = values["EXPORT_MANIFEST_SHA256"]
    mapping_sha256 = values["MAPPING_SHA256"]
    if _SHA256.fullmatch(manifest_sha256) is None or _SHA256.fullmatch(mapping_sha256) is None:
        raise PatientGroupingAuthorityError(
            "patient_grouping_authority_digest_invalid",
            "The patient-grouping authority requires lowercase SHA-256 digests.",
        )

    manifest_path = _regular_file(selected_root / manifest_name, label="export manifest")
    if _digest(manifest_path) != manifest_sha256:
        raise PatientGroupingAuthorityError(
            "patient_grouping_export_manifest_mismatch",
            "The selected export no longer matches the patient-grouping authority.",
            details={"database": selected_database},
        )
    mapping_path = _regular_file(Path(values["MAPPING_PATH"]), label="mapping")
    if _digest(mapping_path) != mapping_sha256:
        raise PatientGroupingAuthorityError(
            "patient_grouping_mapping_digest_mismatch",
            "The private patient-grouping mapping does not match its authority digest.",
            details={"database": selected_database},
        )
    try:
        columns = set(pq.read_schema(mapping_path).names)
    except (OSError, ValueError) as exc:
        raise PatientGroupingAuthorityError(
            "patient_grouping_mapping_schema_unreadable",
            "The private patient-grouping mapping schema cannot be verified.",
        ) from exc
    required_columns = {values["STAY_COLUMN"], values["PATIENT_COLUMN"]}
    if not required_columns <= columns:
        raise PatientGroupingAuthorityError(
            "patient_grouping_mapping_schema_mismatch",
            "The private patient-grouping mapping lacks its declared columns.",
            details={"missing_columns": sorted(required_columns - columns)},
        )

    return PatientGroupingBinding(
        mapping_path=mapping_path,
        mapping_sha256=mapping_sha256,
        mapping_stay_column=values["STAY_COLUMN"],
        mapping_patient_column=values["PATIENT_COLUMN"],
        output_identity_column="patient_stay_id",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": values["AUTHORITY_REF"],
            "database": selected_database,
            "export_manifest_file": manifest_name,
            "export_manifest_sha256": manifest_sha256,
            "mapping_sha256": mapping_sha256,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )


def _private_grouping_root() -> Path:
    root = state_paths.state_root() / "private" / "patient-grouping"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def _serialize_grouping(frame) -> bytes:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq_writer

    buffer = io.BytesIO()
    pq_writer.write_table(pa.Table.from_pandas(frame, preserve_index=False), buffer)
    return buffer.getvalue()


def _materialize_private_mapping(payload: bytes, *, target: Path) -> str:
    """Write a 0600 private mapping, reusing an identical existing file."""

    digest = hashlib.sha256(payload).hexdigest()
    if target.exists() and not target.is_symlink() and target.is_file():
        if _digest(target) == digest:
            os.chmod(target, 0o600)
            return digest
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, target)
    os.chmod(target, 0o600)
    return digest


def resolve_derived_patient_grouping(
    *,
    export_path: str | Path,
    database: str,
) -> Optional[PatientGroupingBinding]:
    """Bind a grouping derived from the export's own sealed raw source.

    ``None`` means this source has no official patient identity reachable
    through a verified binding -- not that the study may proceed unclustered.
    That judgement belongs to the caller.
    """

    from easyicu.webserver import raw_source_authority

    try:
        binding = raw_source_authority.resolve_raw_hospital_source_binding(
            export_path=export_path, database=database
        )
    except raw_source_authority.RawSourceAuthorityError as exc:
        raise PatientGroupingAuthorityError(
            exc.code, str(exc), details=exc.details
        ) from exc
    if binding is None or not getattr(binding, "patient_grouping_available", False):
        return None
    try:
        derived = binding.materialize_patient_grouping()
    except raw_source_authority.RawSourceAuthorityError as exc:
        raise PatientGroupingAuthorityError(
            exc.code, str(exc), details=exc.details
        ) from exc
    receipt = binding.patient_grouping_receipt()
    payload = _serialize_grouping(derived.frame)
    # Name the file after the verified source table, so a changed raw source
    # can never be served from a stale bridge.
    target = _private_grouping_root() / (
        f"{_database(database)}.{receipt['identity_table_sha256']}.parquet"
    )
    mapping_sha256 = _materialize_private_mapping(payload, target=target)
    return PatientGroupingBinding(
        mapping_path=target,
        mapping_sha256=mapping_sha256,
        mapping_stay_column=DERIVED_STAY_COLUMN,
        mapping_patient_column=DERIVED_PATIENT_COLUMN,
        output_identity_column="patient_stay_id",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": f"{binding.authority_ref}/patient_grouping",
            "database": _database(database),
            "export_manifest_file": binding.export_manifest_file,
            "export_manifest_sha256": binding.export_manifest_sha256,
            "mapping_sha256": mapping_sha256,
            "grouping_derivation": "prefix_before_:s",
            "identity_table": receipt["identity_table"],
            "identity_table_sha256": receipt["identity_table_sha256"],
            "patient_identifier_column": receipt["patient_identifier_column"],
            "patient_key_derivation": derived.receipt["patient_key_derivation"],
            "stays": derived.receipt["stays"],
            "patients": derived.receipt["patients"],
            "patients_with_repeated_stays": derived.receipt[
                "patients_with_repeated_stays"
            ],
            "provider_visible_values": False,
        },
    )


def resolve_study_patient_grouping(
    *,
    export_path: str | Path,
    database: str,
) -> Optional[PatientGroupingBinding]:
    """Resolve the patient grouping a study may use, by declared precedence.

    An owner-approved bridge wins; otherwise a source that names its sealed raw
    root may derive one from an official identity table the host already binds.
    Every consumer -- launcher, readiness review, plan compilation -- must ask
    here, or the UI would promise a clustering the runner cannot honour (or
    refuse one it could).
    """

    configured = resolve_patient_grouping_authority(
        export_path=export_path, database=database
    )
    if configured is not None:
        return configured
    return resolve_derived_patient_grouping(
        export_path=export_path, database=database
    )


__all__ = [
    "PatientGroupingAuthorityError",
    "resolve_derived_patient_grouping",
    "resolve_patient_grouping_authority",
    "resolve_study_patient_grouping",
]
