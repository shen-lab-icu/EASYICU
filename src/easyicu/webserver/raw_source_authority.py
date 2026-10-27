"""Host-only binding from a registered export to its verified raw source tables.

Modern EasyICU exports carry ``data_path`` in their manifest: the sealed raw
root the export was built from.  ``resolve_raw_hospital_source_binding`` binds
such an export to the raw tables its database needs for hospital-mortality
follow-up (MIMIC-IV ``icustays`` + ``admissions``; eICU ``patient``), digesting
and schema-checking them at bind time.  ``resolve_raw_source_identity_binding``
binds the same coordinate for a stay's official patient identity, which also
reaches databases without a hospital follow-up profile (MIMIC-III
``icustays``, SICdb ``cases``, AmsterdamUMCdb ``admissions``).  Older prepared exports did not record
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
    from easyicu.research_agent.acquisition.first_icu_stay import (
        DerivedFirstIcuStay,
    )
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
    "mimic": "mimic_iii",
    "mimic_demo": "mimic_iii",
    "mimiciii": "mimic_iii",
    "mimic_iii": "mimic_iii",
    "mimic-iii": "mimic_iii",
    "mimic3": "mimic_iii",
    "miii": "mimic_iii",
    "sic": "sicdb",
    "sicdb": "sicdb",
    "sic-db": "sicdb",
    "aumc": "aumc",
    "amsterdamumc": "aumc",
    "amsterdamumcdb": "aumc",
    "amsterdam": "aumc",
    "hirid": "hirid",
    "hi-rid": "hirid",
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
# ``uniquepid`` (eICU) and ``subject_id`` (MIMIC-IV) are the official patient
# identifiers.  They are optional here on purpose: hospital-mortality follow-up
# never needs them, so a source without one must still bind.  The
# patient-grouping resolver below fails closed instead.
_OPTIONAL_TABLE_COLUMNS = {
    "patient": frozenset({"unitdischargeoffset", "uniquepid", "patienthealthsystemstayid", "unitvisitnumber", "unittype"}),
    "icustays": frozenset({"subject_id"}),
}
# Which official table carries each family's stay-to-patient identity, and the
# columns the bridge is derived from.  The stay column is the identity the
# export keys its stays by.  A column matches exactly, or ignoring case when
# exactly one column does: MIMIC-III ships upper-case CSV headers
# (``ICUSTAY_ID``) and lower-case database columns.  A family absent here has no
# official patient identifier reachable from its bound raw source.
_PATIENT_GROUPING_PROFILES = {
    "eicu": {
        "table": "patient",
        "stay_column": "patientunitstayid",
        "patient_column": "uniquepid",
    },
    "mimic_iv": {
        "table": "icustays",
        "stay_column": "stay_id",
        "patient_column": "subject_id",
    },
    "mimic_iii": {
        "table": "icustays",
        "stay_column": "icustay_id",
        "patient_column": "subject_id",
    },
    "sicdb": {
        "table": "cases",
        "stay_column": "CaseID",
        "patient_column": "PatientID",
    },
    "aumc": {
        "table": "admissions",
        "stay_column": "admissionid",
        "patient_column": "patientid",
    },
}
# Families whose published source has no cross-admission patient identity, so
# their stays are never grouped and no first stay is derived.  This is a fact of
# the source, not a broken binding: HiRID assigns a new ``patientid`` to every
# ICU admission and publishes no linkage between them.
_NO_PATIENT_IDENTITY_FAMILIES = frozenset({"hirid"})
# Which official table orders each family's ICU stays within a patient, and by
# which column.
# - MIMIC-IV and MIMIC-III ``intime``: dates are shifted by one offset per
#   patient, so a patient's stays keep their order across hospitalizations.
#   The earliest stay is the first; a tie fails closed.
# - AmsterdamUMCdb ``admissioncount``: the patient's ICU/MCU admission number,
#   1 at the first (``admittedat`` is the same patient-relative clock, 0 there).
#   The declared first value, not the earliest present one, marks the first
#   stay, so a readmission is never promoted when the first admission is
#   missing from the table; two first values fail closed.
# SICdb is absent until an order is verified.  Its ``OffsetAfterFirstAdmission``
# is 0 on several cases of one patient for 2,964 of 21,583 patients (up to 12
# cases) in the registered source (10-27, aggregate counts only), so it marks a
# first admission, not a first ICU stay, and ``ICUOffset`` still leaves 854 of
# them tied.  Dropping those patients would remove readmitted patients
# selectively; SICdb stays are grouped by patient instead.
# eICU is absent on purpose: ``unitvisitnumber`` orders stays only within one
# hospitalization, and a discharge year cannot order two hospitalizations of
# the same year.  HiRID has no patient identity to order by.
_FIRST_ICU_STAY_PROFILES = {
    "mimic_iv": {
        "table": "icustays",
        "stay_column": "stay_id",
        "patient_column": "subject_id",
        "order_column": "intime",
    },
    "mimic_iii": {
        "table": "icustays",
        "stay_column": "icustay_id",
        "patient_column": "subject_id",
        "order_column": "intime",
    },
    "aumc": {
        "table": "admissions",
        "stay_column": "admissionid",
        "patient_column": "patientid",
        "order_column": "admissioncount",
        "first_order_value": 1,
    },
}
_PROFILE_COLUMN_KEYS = ("stay_column", "patient_column", "order_column")
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


def _table_schema(path: Path, *, table: str) -> set[str]:
    try:
        return set(pq.read_schema(path).names)
    except (OSError, ValueError, ImportError) as exc:
        raise RawSourceAuthorityError(
            "raw_source_authority_table_schema_unreadable",
            "The raw table schema cannot be verified.",
            details={"object": table},
        ) from exc


def _verify_table_schema(path: Path, *, table: str) -> set[str]:
    """Verify the minimal raw columns before issuing an authority."""

    columns = _table_schema(path, table=table)
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


def _physical_column(
    available: Sequence[str] | set[str], name: str, *, table: str
) -> Optional[str]:
    """The table's own name for one declared identity column, if it has one.

    An exact match wins.  Otherwise exactly one column may match ignoring
    case; two such columns could each be the identity, so they fail closed.
    """

    if name in available:
        return name
    folded = sorted(column for column in available if column.casefold() == name.casefold())
    if len(folded) > 1:
        raise RawSourceAuthorityError(
            "raw_source_authority_identity_column_ambiguous",
            "The raw table carries more than one column for a declared identity column.",
            details={"object": table, "column": name},
        )
    return folded[0] if folded else None


def _identity_profiles(family: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        profile
        for profile in (
            _PATIENT_GROUPING_PROFILES.get(family),
            _FIRST_ICU_STAY_PROFILES.get(family),
        )
        if profile is not None
    )


def _identity_read_columns(family: str, table: str, available: set[str]) -> set[str]:
    """The physical identity columns one bound table carries for its family.

    The stay column is the table's key and must exist; a missing patient or
    order column only leaves that profile unavailable.
    """

    selected: set[str] = set()
    for profile in _identity_profiles(family):
        if profile["table"] != table:
            continue
        for key in _PROFILE_COLUMN_KEYS:
            if key not in profile:
                continue
            physical = _physical_column(available, profile[key], table=table)
            if physical is not None:
                selected.add(physical)
            elif key == "stay_column":
                raise RawSourceAuthorityError(
                    "raw_source_authority_table_schema_mismatch",
                    "The raw table lacks the stay identity its database declares.",
                    details={"object": table, "missing_columns": [profile[key]]},
                )
    return selected


def _bound_profile(
    profile: Optional[Mapping[str, Any]], columns: Sequence[str]
) -> Optional[dict[str, Any]]:
    """A profile whose column names are the ones the bound table carries."""

    if profile is None:
        return None
    bound = dict(profile)
    for key in _PROFILE_COLUMN_KEYS:
        if key in profile:
            bound[key] = (
                _physical_column(columns, profile[key], table=profile["table"])
                or profile[key]
            )
    return bound


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
    icustays_columns: frozenset[str] = frozenset()

    # The public receipt keeps its version-1 shape.  A patient grouping this
    # binding derives reaches readers through the grouping authority's own
    # coordinates, so runs already bound to this receipt keep their identity.
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

    @property
    def patient_grouping_profile(self) -> Mapping[str, str]:
        return _PATIENT_GROUPING_PROFILES["mimic_iv"]

    @property
    def patient_grouping_available(self) -> bool:
        """Whether the bound ``icustays`` table carries ``subject_id``."""

        return self.patient_grouping_profile["patient_column"] in self.icustays_columns

    def patient_grouping_receipt(self) -> dict[str, Any]:
        return _patient_grouping_receipt(
            self.patient_grouping_profile, table_sha256=self.icustays_sha256
        )

    def materialize_patient_grouping(self) -> "DerivedPatientGrouping":
        """Derive the private stay-to-patient bridge from the bound table."""

        profile = self.patient_grouping_profile
        if not self.patient_grouping_available:
            raise _patient_grouping_unavailable(self.database)
        icustays = _read_verified_table(
            self.icustays_path,
            self.icustays_sha256,
            table="icustays",
            columns=[profile["stay_column"], profile["patient_column"]],
        )
        return _derive_patient_grouping(icustays, profile, database=self.database)

    @property
    def first_icu_stay_profile(self) -> Mapping[str, str]:
        return _FIRST_ICU_STAY_PROFILES["mimic_iv"]

    @property
    def first_icu_stay_available(self) -> bool:
        """Whether the bound ``icustays`` table can order a patient's stays."""

        profile = self.first_icu_stay_profile
        return {profile["patient_column"], profile["order_column"]} <= self.icustays_columns

    def first_icu_stay_receipt(self) -> dict[str, Any]:
        return _first_icu_stay_receipt(
            self.first_icu_stay_profile, table_sha256=self.icustays_sha256
        )

    def materialize_first_icu_stay(self) -> "DerivedFirstIcuStay":
        """Derive the private first-stay coordinate from the bound table."""

        profile = self.first_icu_stay_profile
        if not self.first_icu_stay_available:
            raise _first_icu_stay_unavailable(self.database)
        icustays = _read_verified_table(
            self.icustays_path,
            self.icustays_sha256,
            table="icustays",
            columns=_first_icu_stay_columns(profile),
        )
        return _derive_first_icu_stay(icustays, profile, database=self.database)


def _patient_grouping_receipt(
    profile: Mapping[str, str], *, table_sha256: str
) -> dict[str, Any]:
    return {
        "cluster_unit": "patient",
        "identity_table": profile["table"],
        "identity_table_sha256": table_sha256,
        "patient_identifier_column": profile["patient_column"],
        "identifier_values_returned": False,
    }


def _patient_grouping_unavailable(database: str) -> RawSourceAuthorityError:
    return RawSourceAuthorityError(
        "raw_source_authority_patient_grouping_unavailable",
        "The bound raw source carries no official patient identifier.",
        details={"database": database},
    )


def _derive_patient_grouping(
    identity_table: pd.DataFrame, profile: Mapping[str, str], *, database: str
) -> "DerivedPatientGrouping":
    from easyicu.research_agent.acquisition.patient_grouping import (
        PatientGroupingError,
        derive_patient_grouping,
    )

    try:
        return derive_patient_grouping(
            identity_table,
            stay_column=profile["stay_column"],
            patient_column=profile["patient_column"],
            identity_table_name=profile["table"],
        )
    except PatientGroupingError as exc:
        raise RawSourceAuthorityError(
            "raw_source_authority_patient_grouping_invalid",
            "The official patient identity table cannot support a grouping.",
            details={"database": database, "cause": str(exc)},
        ) from exc


def _first_icu_stay_columns(profile: Mapping[str, Any]) -> list[str]:
    return [profile["stay_column"], profile["patient_column"], profile["order_column"]]


def _first_icu_stay_receipt(
    profile: Mapping[str, Any], *, table_sha256: str
) -> dict[str, Any]:
    return {
        "scope": "source_global",
        "identity_table": profile["table"],
        "identity_table_sha256": table_sha256,
        "patient_identifier_column": profile["patient_column"],
        "order_column": profile["order_column"],
        "identifier_values_returned": False,
    }


def _first_icu_stay_unavailable(database: str) -> RawSourceAuthorityError:
    return RawSourceAuthorityError(
        "raw_source_authority_first_icu_stay_unavailable",
        "The bound raw source cannot order a patient's ICU stays.",
        details={"database": database},
    )


def _derive_first_icu_stay(
    identity_table: pd.DataFrame, profile: Mapping[str, Any], *, database: str
) -> "DerivedFirstIcuStay":
    from easyicu.research_agent.acquisition.first_icu_stay import (
        FirstIcuStayError,
        derive_first_icu_stay,
    )

    try:
        return derive_first_icu_stay(
            identity_table,
            stay_column=profile["stay_column"],
            patient_column=profile["patient_column"],
            order_column=profile["order_column"],
            identity_table_name=profile["table"],
            first_order_value=profile.get("first_order_value"),
        )
    except FirstIcuStayError as exc:
        raise RawSourceAuthorityError(
            "raw_source_authority_first_icu_stay_invalid",
            "The official stay table cannot prove each patient's first ICU stay.",
            details={"database": database, "cause_code": exc.code, "cause": str(exc)},
        ) from exc


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
    re-verifies the exact bytes.  A family without a hospital follow-up
    profile binds only its official stay-identity table, and its hospital
    materializers refuse.
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
            **(
                {
                    "hospital_mortality_followup": {
                        "outcome": "death",
                        "event_time_column": "death_time_hours",
                        "observation_duration_column": "hospital_followup_time_hours",
                        "unit": "hours",
                        "materializer": _HOSPITAL_SOURCE_PROFILES[self.family][
                            "materializer"
                        ],
                    }
                }
                if self.hospital_mortality_followup_available
                else {}
            ),
            "source_paths_returned": False,
            "identifier_values_returned": False,
            **(
                {"patient_grouping": self.patient_grouping_receipt()}
                if self.patient_grouping_available
                else {}
            ),
        }

    @property
    def hospital_mortality_followup_available(self) -> bool:
        return self.family in _HOSPITAL_SOURCE_PROFILES

    def _require_hospital_followup(self) -> None:
        if not self.hospital_mortality_followup_available:
            raise _hospital_followup_unsupported(self.family)

    def _profile_as_bound(
        self, profile: Optional[Mapping[str, Any]]
    ) -> Optional[dict[str, Any]]:
        if profile is None:
            return None
        return _bound_profile(profile, self.table_columns.get(profile["table"], ()))

    @property
    def patient_grouping_profile(self) -> Optional[Mapping[str, Any]]:
        return self._profile_as_bound(_PATIENT_GROUPING_PROFILES.get(self.family))

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
        return _patient_grouping_receipt(
            profile, table_sha256=self.table_sha256[profile["table"]]
        )

    def materialize_patient_grouping(self) -> "DerivedPatientGrouping":
        """Derive the private stay-to-patient bridge from the bound table."""

        profile = self.patient_grouping_profile
        if profile is None or not self.patient_grouping_available:
            raise _patient_grouping_unavailable(self.database)
        return _derive_patient_grouping(
            self._table(profile["table"]), profile, database=self.database
        )

    @property
    def first_icu_stay_profile(self) -> Optional[Mapping[str, Any]]:
        return self._profile_as_bound(_FIRST_ICU_STAY_PROFILES.get(self.family))

    @property
    def first_icu_stay_available(self) -> bool:
        """Whether this bound source can order each patient's ICU stays."""

        profile = self.first_icu_stay_profile
        if profile is None:
            return False
        columns = set(self.table_columns.get(profile["table"], ()))
        return {profile["patient_column"], profile["order_column"]} <= columns

    def first_icu_stay_receipt(self) -> dict[str, Any]:
        profile = self.first_icu_stay_profile
        if profile is None:  # pragma: no cover - guarded by the caller
            raise _first_icu_stay_unavailable(self.database)
        return _first_icu_stay_receipt(
            profile, table_sha256=self.table_sha256[profile["table"]]
        )

    def materialize_first_icu_stay(self) -> "DerivedFirstIcuStay":
        """Derive the private first-stay coordinate from the bound table."""

        profile = self.first_icu_stay_profile
        if profile is None or not self.first_icu_stay_available:
            raise _first_icu_stay_unavailable(self.database)
        table = profile["table"]
        return _derive_first_icu_stay(
            _read_verified_table(
                self.table_paths[table],
                self.table_sha256[table],
                table=table,
                columns=_first_icu_stay_columns(profile),
            ),
            profile,
            database=self.database,
        )

    @property
    def first_icu_stay_per_hospitalization_available(self) -> bool:
        return self.family == "eicu" and {
            "patienthealthsystemstayid", "unitvisitnumber", "unittype",
        } <= set(self.table_columns.get("patient", ()))

    def materialize_first_icu_stay_per_hospitalization(self):
        from easyicu.research_agent.acquisition.first_icu_stay import (
            FirstIcuStayPerHospitalizationError,
            derive_first_icu_stay_per_hospitalization,
        )

        if not self.first_icu_stay_per_hospitalization_available:
            raise RawSourceAuthorityError(
                "first_icu_stay_per_hospitalization_authority_unavailable",
                "This source cannot identify the first ICU stay within each hospitalization.",
                details={"database": self.database},
            )
        try:
            return derive_first_icu_stay_per_hospitalization(self._table("patient"))
        except FirstIcuStayPerHospitalizationError as exc:
            raise RawSourceAuthorityError(
                "first_icu_stay_per_hospitalization_coordinate_invalid", str(exc),
                details={"database": self.database},
            ) from exc

    def _table(self, name: str) -> pd.DataFrame:
        return _read_verified_table(
            self.table_paths[name],
            self.table_sha256[name],
            table=name,
            columns=self.table_columns[name],
        )

    def materialize_hospital_mortality_followup(self) -> HospitalMortalityFollowup:
        self._require_hospital_followup()
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


def _hospital_followup_unsupported(database: str) -> RawSourceAuthorityError:
    return RawSourceAuthorityError(
        "raw_source_authority_database_unsupported",
        "Hospital-mortality follow-up has no raw-table profile for this database.",
        details={"database": database},
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
    require_hospital_followup: bool = True,
) -> Optional[RawHospitalSourceBinding]:
    """Bind a modern export to the raw tables its manifest ``data_path`` names.

    ``None`` means the export does not carry a sealed ``data_path`` (a legacy
    export, or no export at that path), so the caller may consult the private
    legacy authority instead.  A present but broken coordinate fails closed.
    ``require_hospital_followup=False`` also binds a family whose only profiled
    table is its official stay identity.
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
    hospital_tables = tuple(profile["tables"]) if profile is not None else ()
    identity_tables = tuple(item["table"] for item in _identity_profiles(selected_database))
    if profile is None and (require_hospital_followup or not identity_tables):
        raise _hospital_followup_unsupported(selected_database)
    source_root = _regular_directory(Path(data_path), label="raw source root")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    columns: dict[str, tuple[str, ...]] = {}
    for table in dict.fromkeys((*hospital_tables, *identity_tables)):
        path = _regular_descendant(source_root, f"{table}.parquet", label=f"{table} table")
        if table in hospital_tables:
            available = _verify_table_schema(path, table=table)
            read = set(_table_read_columns(table, available))
        else:
            available = _table_schema(path, table=table)
            read = set()
        read |= _identity_read_columns(selected_database, table, available)
        paths[table] = path
        digests[table] = _sha256_file(path)
        columns[table] = tuple(sorted(read))
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


def resolve_raw_source_identity_binding(
    *,
    export_path: str | Path,
    database: str,
) -> Optional["RawHospitalSourceBinding | RawMimicIVSourceBinding"]:
    """Resolve the raw source that carries one export's official stay identity.

    A family with hospital follow-up keeps exactly that binding, legacy
    MIMIC-IV authority included, so grouping and follow-up read the same
    verified bytes.  A family with only an identity profile binds that table
    through the export's sealed ``data_path``.  ``None`` means no verified
    identity is reachable -- for HiRID, by the nature of the source -- never
    that stays may be treated as independent patients.
    """

    family = _database(database)
    if family in _NO_PATIENT_IDENTITY_FAMILIES:
        return None
    if family in _HOSPITAL_SOURCE_PROFILES or not _identity_profiles(family):
        return resolve_raw_hospital_source_binding(
            export_path=export_path, database=database
        )
    return resolve_manifest_raw_source_binding(
        export_path=export_path, database=database, require_hospital_followup=False
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
    columns: dict[str, frozenset[str]] = {}
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
        columns[name] = frozenset(_verify_table_schema(path, table=name))
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
        icustays_columns=columns["icustays"],
    )


__all__ = [
    "RawHospitalSourceBinding",
    "RawMimicIVSourceBinding",
    "RawSourceAuthorityError",
    "resolve_manifest_raw_source_binding",
    "resolve_raw_hospital_source_binding",
    "resolve_raw_mimic_iv_source_binding",
    "resolve_raw_source_identity_binding",
]
