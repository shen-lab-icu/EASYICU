"""Typed host authority for a first-ICU-stay-per-patient restriction.

A study may keep only each patient's first ICU stay when the host can prove,
from an official stay table it already binds by digest, which stay came first.
This module owns that proof: the fail-closed ordering rules, the private
coordinate the materializer reads, and the aggregate receipt a reader sees.
The coordinate carries stay identifiers and a flag, never a patient key, and
it covers every stay of the source, so a stay whose patient was first admitted
outside a study's export is never promoted to a first stay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import pandas as pd

from .patient_grouping import (
    PatientGroupingError,
    exact_integer_series,
    read_digest_bound_parquet,
)


FIRST_ICU_STAY_SCHEMA = "easyicu.derived_first_icu_stay/1"
COORDINATE_STAY_COLUMN = "stay_id"
COORDINATE_FLAG_COLUMN = "first_icu_stay"
ORDER_RULE = "earliest_order_time_per_patient"
DECLARED_FIRST_ORDER_RULE = "declared_first_admission_value_per_patient"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FirstIcuStayError(ValueError):
    """A first-ICU-stay coordinate cannot be derived or used safely."""

    def __init__(self, message: str, *, code: str = "first_icu_stay_coordinate_invalid"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DerivedFirstIcuStay:
    """Every official stay, flagged when it is its patient's first ICU stay.

    ``frame`` holds no patient key, but it is still private host state: it
    lists the source's stays.  ``receipt`` is aggregate-only.
    """

    frame: pd.DataFrame
    receipt: Mapping[str, object]


def _earliest_order_per_patient(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Flag the stay with the earliest admission time of each patient."""

    order = pd.Series(
        pd.to_datetime(values, errors="coerce").to_numpy(), index=frame.index
    )
    repeated = frame.groupby("patient", sort=False)["stay"].transform("size").gt(1)
    if bool(order.loc[repeated].isna().any()):
        raise FirstIcuStayError(
            "a patient with repeated stays has a stay without a usable admission time",
            code="first_icu_stay_order_time_missing",
        )
    first = pd.Series(True, index=frame.index)
    if bool(repeated.any()):
        several = frame.loc[repeated]
        earliest = order.loc[repeated].groupby(several["patient"], sort=False).transform("min")
        first.loc[repeated] = order.loc[repeated].eq(earliest)
        per_patient = first.loc[repeated].groupby(several["patient"], sort=False).sum()
        if bool(per_patient.gt(1).any()):
            raise FirstIcuStayError(
                "two stays of one patient share the earliest admission time",
                code="first_icu_stay_order_tied",
            )
    return first


def _declared_first_per_patient(
    frame: pd.DataFrame, values: pd.Series, first_order_value: int
) -> tuple[pd.Series, int]:
    """Flag the stay carrying the source's declared first-admission value.

    Returns the flags and the number of patients the table holds no first
    admission for: their stays are readmissions by the source's own column.
    """

    if bool(values.isna().any()):
        raise FirstIcuStayError(
            "a stay has no value in the declared first-admission order column",
            code="first_icu_stay_order_time_missing",
        )
    try:
        parsed = exact_integer_series(values, label="declared first-admission order")
    except PatientGroupingError as exc:
        raise FirstIcuStayError(
            str(exc), code="first_icu_stay_order_value_invalid"
        ) from exc
    order = pd.Series(parsed.to_numpy(dtype="int64"), index=frame.index)
    if bool(order.lt(first_order_value).any()):
        raise FirstIcuStayError(
            "a stay's order value precedes the declared first admission",
            code="first_icu_stay_order_value_invalid",
        )
    first = order.eq(first_order_value)
    per_patient = first.groupby(frame["patient"], sort=False).sum()
    if bool(per_patient.gt(1).any()):
        raise FirstIcuStayError(
            "two stays of one patient carry the declared first-admission value",
            code="first_icu_stay_order_tied",
        )
    return first, int(per_patient.eq(0).sum())


def derive_first_icu_stay(
    identity_table: pd.DataFrame,
    *,
    stay_column: str,
    patient_column: str,
    order_column: str,
    identity_table_name: str,
    first_order_value: Optional[int] = None,
) -> DerivedFirstIcuStay:
    """Mark each patient's first ICU stay in one verified official table.

    By default the order is the table's own admission time.  A patient with
    one stay needs no order.  For a patient with several, a missing or
    unparseable time, or two stays sharing the earliest time, makes the whole
    coordinate unusable: breaking the tie by stay number, or dropping that
    patient, would silently change the population.

    ``first_order_value`` instead declares a per-patient column whose value at
    the patient's first ICU admission the source itself fixes: an admission
    number of 1, or an offset of 0 from the patient's first admission.  That
    value, not the earliest one present, identifies the first stay, so a
    patient whose first admission the table lacks keeps no first stay instead
    of having a readmission promoted.  Every stay then needs an exact integer
    at or after the declared value, and two stays of one patient carrying it
    make the coordinate unusable.
    """

    if first_order_value is not None and (
        isinstance(first_order_value, bool) or not isinstance(first_order_value, int)
    ):
        raise FirstIcuStayError(
            "the declared first-admission value must be an integer",
            code="first_icu_stay_order_value_invalid",
        )
    missing = [
        column
        for column in (stay_column, patient_column, order_column)
        if column not in identity_table.columns
    ]
    if missing:
        raise FirstIcuStayError(
            "official stay table lacks its declared columns: " + ", ".join(missing),
            code="first_icu_stay_identity_columns_missing",
        )
    try:
        stays = exact_integer_series(
            identity_table[stay_column], label="official stay identity"
        )
    except PatientGroupingError as exc:
        raise FirstIcuStayError(
            str(exc), code="first_icu_stay_stay_identity_invalid"
        ) from exc
    if bool(stays.duplicated().any()):
        raise FirstIcuStayError(
            "official stay table repeats a stay identifier",
            code="first_icu_stay_stay_identity_duplicate",
        )
    patients = identity_table[patient_column].astype("string")
    if bool(patients.isna().any()) or bool(patients.str.strip().eq("").any()):
        raise FirstIcuStayError(
            "official patient identifiers must all be present and non-empty",
            code="first_icu_stay_patient_identifier_missing",
        )
    frame = pd.DataFrame(
        {"stay": stays.to_numpy(dtype="int64"), "patient": patients.to_numpy()}
    )
    values = identity_table[order_column]
    if first_order_value is None:
        first = _earliest_order_per_patient(frame, values)
        rule: dict[str, object] = {"order_rule": ORDER_RULE}
        missing_order_policy = "fail_closed_when_patient_has_repeated_stays"
        absent: dict[str, object] = {}
    else:
        first, without_first = _declared_first_per_patient(
            frame, values, first_order_value
        )
        rule = {
            "order_rule": DECLARED_FIRST_ORDER_RULE,
            "first_order_value": int(first_order_value),
        }
        missing_order_policy = "fail_closed"
        absent = {"patients_without_first_icu_stay_in_source": without_first}
    coordinate = (
        pd.DataFrame(
            {
                COORDINATE_STAY_COLUMN: frame["stay"].to_numpy(dtype="int64"),
                COORDINATE_FLAG_COLUMN: first.to_numpy(dtype=bool),
            }
        )
        .sort_values(COORDINATE_STAY_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    first_stays = int(coordinate[COORDINATE_FLAG_COLUMN].sum())
    receipt = {
        "schema_version": FIRST_ICU_STAY_SCHEMA,
        "scope": "source_global",
        "identity_table": str(identity_table_name),
        "stay_column": str(stay_column),
        "patient_identifier_column": str(patient_column),
        "order_column": str(order_column),
        **rule,
        "tie_policy": "fail_closed",
        "missing_order_policy": missing_order_policy,
        "stays": int(len(coordinate)),
        "patients": int(frame["patient"].nunique()),
        "first_icu_stays": first_stays,
        "non_first_icu_stays": int(len(coordinate)) - first_stays,
        **absent,
        "identifier_values_returned": False,
    }
    return DerivedFirstIcuStay(frame=coordinate, receipt=receipt)


@dataclass(frozen=True, slots=True)
class FirstIcuStayBinding:
    """One digest-bound private first-stay coordinate used by materialization."""

    coordinate_path: Path
    coordinate_sha256: str
    authority_coordinates: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.coordinate_path).expanduser()
        if not path.is_absolute():
            raise ValueError("first ICU stay coordinate path must be absolute")
        if _SHA256.fullmatch(str(self.coordinate_sha256 or "")) is None:
            raise ValueError("first ICU stay coordinate sha256 is invalid")
        object.__setattr__(self, "coordinate_path", path)
        object.__setattr__(
            self,
            "authority_coordinates",
            MappingProxyType(dict(self.authority_coordinates)),
        )

    def materializer_kwargs(self) -> dict[str, object]:
        """Return the exact materializer coordinates."""

        return {
            "first_icu_stay_path": self.coordinate_path,
            "first_icu_stay_sha256": self.coordinate_sha256,
            "first_icu_stay_authority_coordinates": dict(self.authority_coordinates),
        }


def load_verified_first_icu_stay(
    path: Path, *, expected_sha256: str
) -> pd.DataFrame:
    """Read one digest-bound coordinate as ``stay_id`` and its first-stay flag."""

    table, _size = read_digest_bound_parquet(
        Path(path),
        expected_sha256=expected_sha256,
        columns=[COORDINATE_STAY_COLUMN, COORDINATE_FLAG_COLUMN],
        label="first ICU stay coordinate",
        error=FirstIcuStayError,
    )
    try:
        stays = exact_integer_series(
            table[COORDINATE_STAY_COLUMN], label="first ICU stay coordinate stay"
        )
    except PatientGroupingError as exc:
        raise FirstIcuStayError(str(exc)) from exc
    if bool(stays.duplicated().any()):
        raise FirstIcuStayError("first ICU stay coordinate repeats a stay identifier")
    flags = table[COORDINATE_FLAG_COLUMN]
    if not pd.api.types.is_bool_dtype(flags) or bool(flags.isna().any()):
        raise FirstIcuStayError("first ICU stay flags must be boolean")
    return pd.DataFrame(
        {
            COORDINATE_STAY_COLUMN: stays.to_numpy(dtype="int64"),
            COORDINATE_FLAG_COLUMN: flags.to_numpy(dtype=bool),
        }
    )


__all__ = [
    "COORDINATE_FLAG_COLUMN",
    "COORDINATE_STAY_COLUMN",
    "DECLARED_FIRST_ORDER_RULE",
    "DerivedFirstIcuStay",
    "FIRST_ICU_STAY_SCHEMA",
    "FirstIcuStayBinding",
    "FirstIcuStayError",
    "derive_first_icu_stay",
    "load_verified_first_icu_stay",
]


SCHEMA = "easyicu.derived_first_icu_stay_per_hospitalization/1"
RESTRICTION_SCHEMA = "easyicu.first_icu_stay_per_hospitalization_restriction/1"
FLAG = "first_icu_stay_per_hospitalization"
REASON = "first_icu_stay_per_hospitalization_reason"
SELECTION_RULE = "minimum_declared_icu_unitvisitnumber_per_hospitalization"
ICU_UNIT_TYPES = (
    "MICU", "SICU", "Med-Surg ICU", "CCU-CTICU", "CTICU", "CSICU", "Cardiac ICU",
    "Cardiovascular ICU", "Neuro ICU", "CCU", "ICU", "VICU", "Virtual ICU",
    "Mobile ICU", "Vent ICU", "Burn-Trauma ICU",
)
NON_ICU_UNIT_TYPES = ("SDU/Step down",)
SELECTION_POLICY = {
    "selection_rule": SELECTION_RULE,
    "missing_hospitalization_policy": "exclude_rows_with_receipt",
    "invalid_or_duplicate_ordinal_policy": "exclude_hospitalization_with_receipt",
    "unknown_unit_type_policy": "exclude_hospitalization_with_receipt",
    "prior_unit_coverage_policy": "require_all_ordinals_one_through_selected_icu",
    "non_icu_unit_policy": "exclude_declared_non_icu_units",
    "icu_unit_types": list(ICU_UNIT_TYPES), "non_icu_unit_types": list(NON_ICU_UNIT_TYPES),
    "establishes_patient_independence": False,
}
UNKNOWN_REASONS = frozenset({
    "hospitalization_identifier_missing_or_invalid",
    "hospitalization_unit_ordinal_missing_or_invalid",
    "hospitalization_unit_ordinal_duplicate",
    "hospitalization_prior_unit_coverage_incomplete",
    "hospitalization_unit_type_missing_or_unknown",
})


class FirstIcuStayPerHospitalizationError(ValueError):
    """The declared hospital-level source coordinate cannot be verified."""


@dataclass(frozen=True, slots=True)
class DerivedFirstIcuStayPerHospitalization:
    frame: pd.DataFrame
    receipt: Mapping[str, object]


def _positive_integers(values: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce")
    valid = numbers.notna() & numbers.gt(0) & numbers.le(2**53 - 1) & numbers.mod(1).eq(0)
    valid &= ~values.map(lambda item: isinstance(item, bool))
    return numbers.where(valid).astype("Int64")


def derive_first_icu_stay_per_hospitalization(table: pd.DataFrame) -> DerivedFirstIcuStayPerHospitalization:
    """Keep the earliest declared ICU in each unambiguous hospitalization.

    Missing/invalid hospital identifiers exclude their rows. Any missing,
    invalid or duplicated ordinal excludes that entire hospitalization. An
    ICU candidate needs all prior ordinals present and classified: a later
    observed ICU is never promoted across unseen earlier units. Missing or
    unknown unit types exclude the hospital. Duplicate/missing stay identity
    refuses the whole coordinate. The unit taxonomy is explicit in the receipt.
    """
    columns = {"patientunitstayid", "patienthealthsystemstayid", "unitvisitnumber", "unittype"}
    if not columns <= set(table):
        raise FirstIcuStayPerHospitalizationError("hospital first-stay source columns are missing")
    try:
        stays = exact_integer_series(table["patientunitstayid"], label="official ICU stay identity")
    except PatientGroupingError as exc:
        raise FirstIcuStayPerHospitalizationError(str(exc)) from exc
    if stays.duplicated().any() or stays.le(0).any():
        raise FirstIcuStayPerHospitalizationError("official ICU stay identities must be unique and positive")
    frame = pd.DataFrame({
        "stay_id": stays.to_numpy(dtype="int64"),
        "hospital": _positive_integers(table["patienthealthsystemstayid"]).to_numpy(),
        "ordinal": _positive_integers(table["unitvisitnumber"]).to_numpy(),
        "unit_type": table["unittype"].astype("string").str.strip().str.casefold().to_numpy(),
    })
    reason = pd.Series("", index=frame.index, dtype="string")
    reason.loc[frame["hospital"].isna()] = "hospitalization_identifier_missing_or_invalid"
    groups = frame.loc[frame["hospital"].notna()]
    invalid = set(groups.loc[groups["ordinal"].isna(), "hospital"])
    reason.loc[frame["hospital"].isin(invalid)] = "hospitalization_unit_ordinal_missing_or_invalid"
    duplicate = groups.duplicated(["hospital", "ordinal"], keep=False)
    duplicate_groups = set(groups.loc[duplicate, "hospital"])
    reason.loc[reason.eq("") & frame["hospital"].isin(duplicate_groups)] = "hospitalization_unit_ordinal_duplicate"
    icu = frame["unit_type"].isin({value.casefold() for value in ICU_UNIT_TYPES})
    non_icu = frame["unit_type"].isin({value.casefold() for value in NON_ICU_UNIT_TYPES})
    unknown_type_groups = set(frame.loc[~(icu | non_icu) & frame["hospital"].notna(), "hospital"])
    reason.loc[reason.eq("") & frame["hospital"].isin(unknown_type_groups)] = "hospitalization_unit_type_missing_or_unknown"
    candidate = frame.loc[icu].groupby("hospital")["ordinal"].min()
    first_icu_ordinal = frame["hospital"].map(candidate)
    preceding = frame.loc[frame["ordinal"].lt(first_icu_ordinal)].groupby("hospital").size()
    prior_count = frame["hospital"].map(preceding).fillna(0)
    gap = first_icu_ordinal.notna() & prior_count.ne(first_icu_ordinal - 1)
    reason.loc[reason.eq("") & gap] = "hospitalization_prior_unit_coverage_incomplete"
    eligible = reason.eq("")
    reason.loc[eligible & non_icu] = "non_icu_unit"
    reason.loc[eligible & icu & frame["ordinal"].eq(first_icu_ordinal)] = "kept"
    reason.loc[eligible & icu & ~frame["ordinal"].eq(first_icu_ordinal)] = "non_first_icu_stay"
    coordinate = pd.DataFrame({
        "stay_id": frame["stay_id"], FLAG: reason.eq("kept"), REASON: reason,
    }).sort_values("stay_id", kind="mergesort").reset_index(drop=True)
    counts = {str(key): int(value) for key, value in reason.value_counts().items() if key != "kept"}
    receipt = {
        "schema_version": SCHEMA, "scope": "source_global", "identity_table": "patient",
        "hospitalization_identifier_column": "patienthealthsystemstayid",
        "order_column": "unitvisitnumber", "unit_type_column": "unittype",
        **SELECTION_POLICY,
        "stays": int(len(frame)), "hospitalizations": int(groups["hospital"].nunique()),
        "first_icu_stays": int(coordinate[FLAG].sum()),
        "non_first_icu_stays": counts.get("non_first_icu_stay", 0),
        "non_icu_stays": counts.get("non_icu_unit", 0),
        "unknown_eligibility_stays": sum(counts.get(key, 0) for key in UNKNOWN_REASONS),
        "excluded_reason_counts": counts, "identifier_values_returned": False,
    }
    return DerivedFirstIcuStayPerHospitalization(coordinate, receipt)


def load_verified_first_icu_stay_per_hospitalization(path: Path, *, expected_sha256: str) -> pd.DataFrame:
    frame, _ = read_digest_bound_parquet(
        path, expected_sha256=expected_sha256, columns=["stay_id", FLAG, REASON],
        label="first ICU stay per hospitalization", error=FirstIcuStayPerHospitalizationError,
    )
    try:
        frame["stay_id"] = exact_integer_series(frame["stay_id"], label="coordinate stay")
    except PatientGroupingError as exc:
        raise FirstIcuStayPerHospitalizationError(str(exc)) from exc
    if frame["stay_id"].duplicated().any() or frame["stay_id"].le(0).any():
        raise FirstIcuStayPerHospitalizationError("coordinate stay identities must be unique and positive")
    if not pd.api.types.is_bool_dtype(frame[FLAG]) or frame[FLAG].isna().any():
        raise FirstIcuStayPerHospitalizationError("coordinate first-stay flags must be boolean")
    if not frame[REASON].isin({"kept", "non_first_icu_stay", "non_icu_unit", *UNKNOWN_REASONS}).all():
        raise FirstIcuStayPerHospitalizationError("coordinate has an undeclared exclusion reason")
    if not frame[FLAG].eq(frame[REASON].eq("kept")).all():
        raise FirstIcuStayPerHospitalizationError("coordinate flags disagree with exclusion reasons")
    return frame


@dataclass(frozen=True, slots=True)
class FirstIcuStayPerHospitalizationBinding:
    coordinate_path: Path
    coordinate_sha256: str
    authority_coordinates: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.coordinate_path).expanduser()
        if not path.is_absolute() or re.fullmatch(r"[0-9a-f]{64}", self.coordinate_sha256) is None:
            raise ValueError("hospital first-stay coordinate requires an absolute path and SHA-256")
        object.__setattr__(self, "coordinate_path", path)
        object.__setattr__(self, "authority_coordinates", MappingProxyType(dict(self.authority_coordinates)))

    def materializer_kwargs(self) -> dict[str, object]:
        return {
            "first_icu_stay_per_hospitalization_path": self.coordinate_path,
            "first_icu_stay_per_hospitalization_sha256": self.coordinate_sha256,
            "first_icu_stay_per_hospitalization_authority_coordinates": dict(self.authority_coordinates),
        }


def restrict_to_first_icu_stay_per_hospitalization(
    cohort: pd.DataFrame, *, coordinate_path: Path, coordinate_sha256: str,
    authority_coordinates: Mapping[str, object] | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    coordinate = load_verified_first_icu_stay_per_hospitalization(
        coordinate_path, expected_sha256=coordinate_sha256,
    )
    if {FLAG, REASON} & set(cohort):
        raise FirstIcuStayPerHospitalizationError("hospital first-stay coordinate would overwrite cohort columns")
    joined = cohort.merge(coordinate, on="stay_id", how="left", validate="one_to_one", sort=False)
    if joined[FLAG].isna().any():
        raise FirstIcuStayPerHospitalizationError("hospital first-stay coordinate does not cover all cohort stays")
    keep = joined[FLAG].astype(bool).to_numpy()
    result = cohort.loc[keep].reset_index(drop=True)
    counts = {str(key): int(value) for key, value in joined.loc[~keep, REASON].value_counts().items()}
    return result, {
        "schema_version": RESTRICTION_SCHEMA, "coordinate_sha256": coordinate_sha256,
        "stays_before": int(len(cohort)), "stays_after": int(len(result)),
        "non_first_icu_stays_removed": counts.get("non_first_icu_stay", 0),
        "non_icu_stays_removed": counts.get("non_icu_unit", 0),
        "unknown_eligibility_stays_removed": sum(counts.get(key, 0) for key in UNKNOWN_REASONS),
        "excluded_reason_counts": counts, "authority_coordinates": dict(authority_coordinates or {}),
    }


# The eICU source-observed subset excludes ambiguous hospitalizations.
# Its coordinate is not a claim about a patient's lifetime first ICU.

REQUIRED_COLUMNS = (
    "patientunitstayid", "uniquepid", "patienthealthsystemstayid",
    "unitvisitnumber", "unittype", "age",
)
FLAG_COLUMN = "first_observed_icu_single_hospital_adult"
REASON_COLUMN = "first_observed_icu_single_hospital_reason"
EICU_FIRST_OBSERVED_CANDIDATE_SCHEMA = "easyicu.eicu_first_observed_single_hospital_candidate/1"


class FirstObservedCandidateError(ValueError):
    """The source cannot establish this conservative candidate."""


@dataclass(frozen=True)
class FirstObservedCandidate:
    # Host-private coordinate: stay IDs must not go to a model or public receipt.
    frame: pd.DataFrame
    receipt: Mapping[str, Any]


def _adult(value: object) -> bool | None:
    """Interpret only eICU's integer-year and censored >89 representations."""
    if pd.isna(value) or isinstance(value, bool):
        return None
    value = str(value).strip().replace(" ", "")
    if value == ">89":
        return True
    if not value.isascii() or not value.isdigit():
        return None
    age = int(value)
    return age >= 18 if 0 <= age <= 89 else None


def derive_first_observed_candidate(table: pd.DataFrame) -> FirstObservedCandidate:
    """Derive one first-observed ICU per adult with exactly one visible hospital.

    Input is the complete source-global patient table, not a prefiltered cohort.
    All patient keys must be present: otherwise an anonymous row could belong to
    any identified patient and invalidate the single-hospital assertion.  Patients
    with missing hospital IDs, reused hospital IDs, multiple hospitalizations,
    unprovable unit order/type, unknown age, or age below 18 are excluded.
    A known unit ordinal/type is checked by the existing hospital-level owner.
    """
    if not set(REQUIRED_COLUMNS) <= set(table):
        raise FirstObservedCandidateError("eICU source lacks required identity, age, or unit columns")
    patients = table["uniquepid"]
    if patients.isna().any() or any(
        not isinstance(v, str) or not v or v != v.strip() for v in patients
    ):
        raise FirstObservedCandidateError("all source patient identifiers must be present")
    try:
        per_hospital = derive_first_icu_stay_per_hospitalization(table).frame
    except FirstIcuStayPerHospitalizationError as exc:
        raise FirstObservedCandidateError("source stay identities cannot be verified") from exc

    # The source owner validates exact positive stay identifiers and their
    # uniqueness. Join its private coordinate back to original rows by stay ID.
    working = table.loc[:, list(REQUIRED_COLUMNS)].copy()
    working["_patient"] = patients.to_numpy()
    working["_hospital"] = pd.to_numeric(working["patienthealthsystemstayid"], errors="coerce")
    valid_hospital = (
        working["_hospital"].notna() & working["_hospital"].gt(0)
        & working["_hospital"].le(2**53 - 1) & working["_hospital"].mod(1).eq(0)
        & ~working["patienthealthsystemstayid"].map(lambda v: isinstance(v, bool))
    )
    working.loc[~valid_hospital, "_hospital"] = pd.NA
    working = working.merge(per_hospital, left_on="patientunitstayid", right_on="stay_id", validate="one_to_one")
    patient_reason: dict[str, str] = {}
    # A hospital identifier assigned to multiple patients makes each patient
    # attached to that identifier untrustworthy, not a joinable population.
    shared = working.dropna(subset=["_hospital"]).groupby("_hospital")["_patient"].nunique()
    shared_hospitals = set(shared[shared.gt(1)].index)
    for patient, rows in working.groupby("_patient", sort=False):
        if rows["_hospital"].isna().any():
            reason = "hospitalization_identifier_missing_or_invalid"
        elif rows["_hospital"].isin(shared_hospitals).any():
            reason = "hospitalization_identifier_shared_across_patients"
        elif rows["_hospital"].nunique() != 1:
            reason = "multiple_observed_hospitalizations"
        else:
            chosen = rows[rows[FLAG]]
            if len(chosen) != 1:
                reason = "first_icu_order_or_type_unverifiable"
            else:
                adult = _adult(chosen.iloc[0]["age"])
                reason = "age_unknown_or_invalid" if adult is None else "under_18" if not adult else "eligible"
        patient_reason[patient] = reason
    patient_status = working["_patient"].map(patient_reason)
    eligible = patient_status.eq("eligible") & working[FLAG]
    row_reason = patient_status.where(~patient_status.eq("eligible"), "non_selected_unit")
    row_reason = row_reason.where(~eligible, "kept")
    coordinate = pd.DataFrame({
        "stay_id": working["stay_id"].to_numpy(),
        FLAG_COLUMN: eligible.to_numpy(dtype=bool),
        REASON_COLUMN: row_reason.to_numpy(),
    }).sort_values("stay_id", kind="mergesort").reset_index(drop=True)
    counts = pd.Series(list(patient_reason.values()), dtype="string").value_counts()
    receipt = {
        "schema_version": EICU_FIRST_OBSERVED_CANDIDATE_SCHEMA,
        "scope": "first_observed_icu_in_single_observed_hospitalization_per_patient",
        "source_global_table_required": True,
        "not_lifetime_first_icu": True,
        "multiple_hospitalization_policy": "exclude_all_stays_of_patient",
        "unidentified_patient_policy": "refuse_source",
        "age_policy": "adult_at_selected_stay_unknown_excluded_eicu_over_89_adult",
        "unit_order_policy": "existing_eicu_first_icu_per_hospitalization_owner",
        "source_stays": len(working),
        "source_patients": len(patient_reason),
        "eligible_patients": int(counts.get("eligible", 0)),
        "excluded_patients_by_reason": {str(k): int(v) for k, v in counts.items() if k != "eligible"},
        "excluded_stays_by_reason": {str(k): int(v) for k, v in row_reason.value_counts().items() if k != "kept"},
        "receipt_contains_identifier_values": False,
        "private_coordinate_contains_stay_ids": True,
        "private_coordinate_contains_patient_ids": False,
        "outcome_columns_read": False,
        "research_execution_authorized": False,
    }
    return FirstObservedCandidate(coordinate, receipt)


class FirstObservedCoordinateError(ValueError):
    """An outcome-blind coordinate is malformed or no longer matches its bytes."""


_REASONS = frozenset({
    "kept", "non_selected_unit", "hospitalization_identifier_missing_or_invalid",
    "hospitalization_identifier_shared_across_patients", "multiple_observed_hospitalizations",
    "first_icu_order_or_type_unverifiable", "age_unknown_or_invalid", "under_18",
})


@dataclass(frozen=True, slots=True)
class FirstObservedIcuCoordinate:
    """Host-private bytes and exact source/review digests, not an approval."""

    path: Path
    sha256: str
    archive_sha256: str
    patient_table_sha256: str
    export_manifest_sha256: str
    plan_sha256: str
    preparation_checkpoint_sha256: str
    # Optional host origin coordinates make a real signed preparation
    # discoverable later.  They are not themselves a host grant.
    preparation_run_id: str | None = None
    preparation_binding_sha256: str | None = None

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("first-observed coordinate path must be absolute")
        for name in ("sha256", "archive_sha256", "patient_table_sha256", "export_manifest_sha256", "plan_sha256", "preparation_checkpoint_sha256"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"invalid first-observed {name}")
        if (self.preparation_run_id is None) != (self.preparation_binding_sha256 is None):
            raise ValueError("first-observed preparation origin is incomplete")
        if self.preparation_run_id is not None and (
                not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", self.preparation_run_id)
                or not re.fullmatch(r"[0-9a-f]{64}", self.preparation_binding_sha256 or "")):
            raise ValueError("first-observed preparation origin is invalid")
        object.__setattr__(self, "path", Path(self.path))


def restrict_to_first_observed_icu(cohort: pd.DataFrame, *, coordinate: FirstObservedIcuCoordinate) -> pd.DataFrame:
    """Return source-visible selected stays, or fail on any uncovered stay.

    The result and coordinate remain host-private. No exclusion counts or
    row-level data are emitted; a separate disclosure policy must prepare any
    cohort-flow report, including suppression of small/inferable cells.
    """
    if {FLAG_COLUMN, REASON_COLUMN} & set(cohort):
        raise FirstObservedCoordinateError("coordinate would overwrite cohort columns")
    try:
        identities = exact_integer_series(cohort["stay_id"], label="cohort stay")
        rows, _ = read_digest_bound_parquet(
            coordinate.path, expected_sha256=coordinate.sha256,
            columns=["stay_id", FLAG_COLUMN, REASON_COLUMN],
            label="first observed ICU coordinate", error=FirstObservedCoordinateError,
        )
        rows["stay_id"] = exact_integer_series(rows["stay_id"], label="coordinate stay")
    except (KeyError, PatientGroupingError) as exc:
        raise FirstObservedCoordinateError("stay identities are invalid") from exc
    if identities.isna().any() or identities.le(0).any() or identities.duplicated().any():
        raise FirstObservedCoordinateError("cohort stays must be unique and positive")
    if rows["stay_id"].isna().any() or rows["stay_id"].le(0).any() or rows["stay_id"].duplicated().any():
        raise FirstObservedCoordinateError("coordinate stays must be unique and positive")
    if not pd.api.types.is_bool_dtype(rows[FLAG_COLUMN]) or rows[FLAG_COLUMN].isna().any():
        raise FirstObservedCoordinateError("coordinate selection must be boolean")
    if not rows[REASON_COLUMN].isin(_REASONS).all() or not rows[FLAG_COLUMN].eq(rows[REASON_COLUMN].eq("kept")).all():
        raise FirstObservedCoordinateError("coordinate reasons disagree with selection")
    ordered = pd.DataFrame({"stay_id": identities.to_numpy()}).merge(rows, on="stay_id", how="left", validate="one_to_one", sort=False)
    if ordered[FLAG_COLUMN].isna().any():
        raise FirstObservedCoordinateError("coordinate does not cover every cohort stay")
    return cohort.loc[ordered[FLAG_COLUMN].to_numpy(dtype=bool)].reset_index(drop=True)
