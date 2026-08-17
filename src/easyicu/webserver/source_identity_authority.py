"""Host-only resolver for verified patient-grouping coordinates.

Prepared EasyICU exports intentionally do not expose direct patient identifiers.
When a data owner has separately approved a private stay-to-patient bridge, the
Web host can bind that bridge through environment coordinates.  Pi and the
Provider receive only the derived column name and digests, never the mapping
path or its values.
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
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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


__all__ = [
    "PatientGroupingAuthorityError",
    "resolve_patient_grouping_authority",
]
