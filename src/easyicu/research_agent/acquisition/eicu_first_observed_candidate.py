"""Conservative, outcome-blind eICU first-observed-ICU research candidate.

This is not wired into extraction, planning, or execution authority.  Its scope
is one complete, exact bound source table, never a patient's lifetime history.
It intentionally excludes everyone with more than one observed hospitalization
because eICU does not reliably order hospitalizations for the same person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .first_icu_stay import (
    FLAG,
    FirstIcuStayPerHospitalizationError,
    derive_first_icu_stay_per_hospitalization,
)


REQUIRED_COLUMNS = (
    "patientunitstayid", "uniquepid", "patienthealthsystemstayid",
    "unitvisitnumber", "unittype", "age",
)
FLAG_COLUMN = "first_observed_icu_single_hospital_adult"
REASON_COLUMN = "first_observed_icu_single_hospital_reason"
SCHEMA = "easyicu.eicu_first_observed_single_hospital_candidate/1"


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
        "schema_version": SCHEMA,
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
