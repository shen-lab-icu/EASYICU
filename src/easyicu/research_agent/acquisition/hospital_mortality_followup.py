"""Deterministic hospital-mortality follow-up construction from raw tables.

The prepared EasyICU ``death`` concept is an event-status flag.  It does not,
by itself, establish the time axis needed by a longitudinal survival analysis:
ICU length of stay ends at ICU discharge and cannot censor an in-hospital
mortality endpoint.  This owner derives that axis only from raw source tables
supplied by a separately verified source binding: MIMIC-IV ``icu.icustays``
plus ``hosp.admissions``, or the eICU ``patient`` table.  Both derivations
share one output contract and one exclusion/receipt vocabulary so a landmark
or time-varying runtime never learns which database produced its axis.

It deliberately has no path discovery, no export selection, and no modelling
logic.  Its small public contract makes the data-quality action explicit:
chronologically impossible or incomplete rows are returned as typed
exclusions with aggregate receipts, while an event exactly at ICU admission is
retained as a valid zero-time event.  A later survival runtime must declare
how it represents such an event computationally; this module never shifts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...hospital_mortality import (
    HospitalMortalityStatusError,
    hospital_mortality_status_from_flag,
    join_mimic_hospital_admissions,
)

HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS = (
    "stay_id",
    "hospital_death",
    "death_time_hours",
    "hospital_followup_time_hours",
)
MIMIC_IV_HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS = HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS
EICU_HOSPITAL_MORTALITY_DATABASES = ("eicu", "eicu_demo")


class HospitalMortalityFollowupError(ValueError):
    """The raw source cannot supply an unambiguous follow-up axis."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class HospitalMortalityFollowup:
    """One exact, path-free hospital-mortality follow-up materialization."""

    frame: pd.DataFrame
    exclusions: pd.DataFrame
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(self.frame.columns) != HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS:
            raise HospitalMortalityFollowupError(
                "hospital_followup_output_schema_invalid",
                "The hospital follow-up output does not use the canonical columns.",
            )
        if tuple(self.exclusions.columns) != ("stay_id", "reason_code"):
            raise HospitalMortalityFollowupError(
                "hospital_followup_exclusion_schema_invalid",
                "The hospital follow-up exclusions do not use the canonical columns.",
            )


def _require_columns(
    frame: pd.DataFrame, *, label: str, columns: tuple[str, ...]
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise HospitalMortalityFollowupError(
            f"hospital_followup_{label}_columns_missing",
            f"The raw {label} table lacks required columns: {', '.join(missing)}.",
        )


def _timestamps(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return parsed timestamps and an explicit non-null parse-failure mask."""

    parsed = pd.to_datetime(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    return parsed, invalid


def _first_reason(
    reason: pd.Series,
    mask: pd.Series,
    code: str,
) -> None:
    """Set a reason only while preserving a stable, declared priority order."""

    reason.loc[reason.eq("") & mask] = code


def derive_mimic_iv_hospital_mortality_followup(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
) -> HospitalMortalityFollowup:
    """Derive a stay-level hospital-mortality event/censoring pair.

    ``hospital_expire_flag`` is the event-status authority.  For an event,
    follow-up ends at ``deathtime``; otherwise it ends at ``dischtime``.  Both
    are measured from the ICU stay's ``intime`` in hours.  A row that cannot
    establish this exact pair is excluded with a typed reason rather than
    silently recoded or assigned ICU length of stay.
    """

    if not isinstance(icustays, pd.DataFrame) or not isinstance(
        admissions, pd.DataFrame
    ):
        raise TypeError("MIMIC-IV hospital follow-up requires pandas DataFrames")
    _require_columns(
        icustays,
        label="icustays",
        columns=("stay_id", "hadm_id", "intime"),
    )
    _require_columns(
        admissions,
        label="admissions",
        columns=(
            "hadm_id",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
        ),
    )
    try:
        joined = join_mimic_hospital_admissions(icustays, admissions)
    except HospitalMortalityStatusError as exc:
        raise HospitalMortalityFollowupError(
            exc.code.replace("hospital_status_", "hospital_followup_", 1),
            "The raw hospital-status linkage is ambiguous.",
        ) from exc

    intime, invalid_intime = _timestamps(joined["intime"])
    dischtime, invalid_dischtime = _timestamps(joined["dischtime"])
    deathtime, invalid_deathtime = _timestamps(joined["deathtime"])
    status = hospital_mortality_status_from_flag(joined["hospital_expire_flag"])
    event_flag_valid = status.notna()
    event = status.fillna(False).astype(bool)
    censor = (~status).fillna(False).astype(bool)

    reason = pd.Series("", index=joined.index, dtype="object")
    _first_reason(
        reason,
        joined["__admission_match"].ne("both"),
        "hospital_admission_missing",
    )
    _first_reason(reason, invalid_intime | intime.isna(), "icu_intime_unavailable")
    _first_reason(reason, ~event_flag_valid, "hospital_mortality_flag_invalid")
    _first_reason(
        reason,
        event & invalid_deathtime,
        "hospital_death_time_invalid",
    )
    _first_reason(
        reason,
        event & deathtime.isna(),
        "hospital_death_time_missing",
    )
    _first_reason(
        reason,
        censor & invalid_dischtime,
        "hospital_discharge_time_invalid",
    )
    _first_reason(
        reason,
        censor & dischtime.isna(),
        "hospital_discharge_time_missing",
    )
    _first_reason(
        reason,
        censor & (invalid_deathtime | deathtime.notna()),
        "hospital_survivor_death_time_inconsistent",
    )

    endpoint_time = deathtime.where(event, dischtime)
    followup_hours = (endpoint_time - intime).dt.total_seconds() / 3600.0
    nonfinite_followup = ~np.isfinite(followup_hours)
    _first_reason(
        reason,
        reason.eq("") & nonfinite_followup,
        "hospital_followup_time_nonfinite",
    )
    _first_reason(
        reason,
        reason.eq("") & event & followup_hours.lt(0),
        "hospital_death_before_icu_admission",
    )
    _first_reason(
        reason,
        reason.eq("") & censor & followup_hours.lt(0),
        "hospital_discharge_before_icu_admission",
    )

    receipt = {
        "schema_version": "easyicu.mimic_iv_hospital_mortality_followup/1",
        "database": "miiv",
        "analysis_unit": "icu_stay",
        "time_origin": "icu_admission",
        "time_unit": "hours",
        "event": {
            "column": "hospital_death",
            "definition": "admissions.hospital_expire_flag == 1",
            "event_time_column": "death_time_hours",
            "event_time_source": "admissions.deathtime - icustays.intime",
        },
        "censoring": {
            "followup_time_column": "hospital_followup_time_hours",
            "rule": ("event_at_deathtime_else_censor_at_hospital_discharge_time"),
            "source": "admissions.dischtime - icustays.intime",
        },
    }
    return _finalize_followup(
        stay_ids=joined["stay_id"],
        event=event,
        reason=reason,
        followup_hours=followup_hours,
        receipt=receipt,
    )


def derive_eicu_hospital_mortality_followup(
    patient: pd.DataFrame,
    *,
    database: str = "eicu",
) -> HospitalMortalityFollowup:
    """Derive the eICU stay-level hospital-mortality event/censoring pair.

    ``patient.hospitaldischargestatus`` is the event-status authority: an
    ``Expired`` stay is an event and an ``Alive`` stay is censored at hospital
    discharge.  Both endpoints occur at ``hospitaldischargeoffset``, which eICU
    records in minutes from ICU unit admission, so the axis keeps the same
    ICU-admission origin in hours as the MIMIC-IV derivation.  The prepared
    ``death`` concept already reads the same status column but stamps it at ICU
    discharge; only this raw offset supplies the hospital follow-up clock.  A
    stay whose status or offset cannot establish the exact pair is excluded
    with a typed reason rather than recoded.  eICU occasionally records a
    hospital discharge a few minutes before the unit discharge; that
    administrative ordering does not touch the hospital clock and is only
    counted in the receipt.
    """

    if not isinstance(patient, pd.DataFrame):
        raise TypeError("eICU hospital follow-up requires a pandas DataFrame")
    normalized_database = str(database or "").strip().lower()
    if normalized_database not in EICU_HOSPITAL_MORTALITY_DATABASES:
        raise HospitalMortalityFollowupError(
            "hospital_followup_database_unsupported",
            "The eICU hospital follow-up derivation applies to eICU releases only.",
        )
    _require_columns(
        patient,
        label="patient",
        columns=(
            "patientunitstayid",
            "hospitaldischargeoffset",
            "hospitaldischargestatus",
        ),
    )
    stay_ids = patient["patientunitstayid"]
    if stay_ids.isna().any() or stay_ids.duplicated().any():
        suffix = "missing" if stay_ids.isna().any() else "nonunique"
        raise HospitalMortalityFollowupError(
            f"hospital_followup_patient_key_{suffix}",
            "The raw patient identity is ambiguous.",
        )
    rows = patient.reset_index(drop=True)
    stay_ids = rows["patientunitstayid"]

    status_raw = rows["hospitaldischargestatus"]
    status = status_raw.astype("string").str.strip().str.lower()
    event = status.eq("expired").fillna(False).astype(bool)
    censor = status.eq("alive").fillna(False).astype(bool)
    offset = pd.to_numeric(rows["hospitaldischargeoffset"], errors="coerce")
    offset_invalid = rows["hospitaldischargeoffset"].notna() & offset.isna()
    followup_hours = offset.astype(float) / 60.0

    reason = pd.Series("", index=rows.index, dtype="object")
    _first_reason(reason, status_raw.isna(), "hospital_mortality_status_missing")
    _first_reason(reason, ~(event | censor), "hospital_mortality_status_invalid")
    _first_reason(reason, offset_invalid, "hospital_discharge_offset_invalid")
    _first_reason(
        reason,
        ~offset_invalid & offset.isna(),
        "hospital_discharge_offset_missing",
    )
    nonfinite_followup = ~np.isfinite(followup_hours.to_numpy(dtype=float))
    _first_reason(
        reason,
        reason.eq("") & nonfinite_followup,
        "hospital_followup_time_nonfinite",
    )
    _first_reason(
        reason,
        reason.eq("") & event & followup_hours.lt(0),
        "hospital_death_before_icu_admission",
    )
    _first_reason(
        reason,
        reason.eq("") & censor & followup_hours.lt(0),
        "hospital_discharge_before_icu_admission",
    )

    chronology_notes: dict[str, Any] = {}
    if "unitdischargeoffset" in rows.columns:
        unit_offset = pd.to_numeric(rows["unitdischargeoffset"], errors="coerce")
        chronology_notes["hospital_discharge_before_icu_discharge_stays"] = int(
            (reason.eq("") & offset.lt(unit_offset)).sum()
        )
    receipt = {
        "schema_version": "easyicu.eicu_hospital_mortality_followup/1",
        "database": normalized_database,
        "analysis_unit": "icu_stay",
        "time_origin": "icu_admission",
        "time_unit": "hours",
        "event": {
            "column": "hospital_death",
            "definition": "patient.hospitaldischargestatus == 'Expired'",
            "event_time_column": "death_time_hours",
            "event_time_source": "patient.hospitaldischargeoffset / 60",
        },
        "censoring": {
            "followup_time_column": "hospital_followup_time_hours",
            "rule": (
                "event_at_hospital_discharge_offset_else_censor_at_"
                "hospital_discharge_offset"
            ),
            "source": "patient.hospitaldischargeoffset / 60",
        },
        "chronology_notes": chronology_notes,
    }
    return _finalize_followup(
        stay_ids=stay_ids,
        event=event,
        reason=reason,
        followup_hours=followup_hours,
        receipt=receipt,
    )


def _finalize_followup(
    *,
    stay_ids: pd.Series,
    event: pd.Series,
    reason: pd.Series,
    followup_hours: pd.Series,
    receipt: dict[str, Any],
) -> HospitalMortalityFollowup:
    """Split valid rows from typed exclusions and count them for the receipt."""

    valid = reason.eq("")
    valid_rows = pd.DataFrame({"stay_id": stay_ids.loc[valid].to_numpy()})
    valid_rows["hospital_death"] = event.loc[valid].astype("int8").to_numpy()
    valid_rows["death_time_hours"] = np.where(
        event.loc[valid].to_numpy(),
        followup_hours.loc[valid].to_numpy(dtype=float),
        np.nan,
    )
    valid_rows["hospital_followup_time_hours"] = followup_hours.loc[valid].to_numpy(
        dtype=float
    )
    valid_rows = valid_rows.loc[:, HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS]

    exclusions = pd.DataFrame({"stay_id": stay_ids.loc[~valid].to_numpy()})
    exclusions["reason_code"] = reason.loc[~valid].astype(str).to_numpy()
    exclusions = exclusions.loc[:, ["stay_id", "reason_code"]]
    exclusion_counts = {
        str(code): int(count)
        for code, count in exclusions["reason_code"].value_counts(sort=False).items()
    }
    exclusion_counts = dict(sorted(exclusion_counts.items()))
    zero_time_events = int(
        (
            valid_rows["hospital_death"].eq(1) & valid_rows["death_time_hours"].eq(0.0)
        ).sum()
    )
    zero_time_censoring = int(
        (
            valid_rows["hospital_death"].eq(0)
            & valid_rows["hospital_followup_time_hours"].eq(0.0)
        ).sum()
    )
    receipt = {
        **receipt,
        "input_stays": int(len(stay_ids)),
        "valid_stays": int(len(valid_rows)),
        "excluded_stays": int(len(exclusions)),
        "event_stays": int(valid_rows["hospital_death"].sum()),
        "censored_stays": int((valid_rows["hospital_death"] == 0).sum()),
        "zero_time_event_stays": zero_time_events,
        "zero_time_censored_stays": zero_time_censoring,
        "exclusion_counts": exclusion_counts,
        "privacy": {
            "raw_rows_returned": False,
            "identifier_values_returned": False,
            "source_paths_returned": False,
        },
    }
    return HospitalMortalityFollowup(
        frame=valid_rows.reset_index(drop=True),
        exclusions=exclusions.reset_index(drop=True),
        receipt=receipt,
    )


__all__ = [
    "EICU_HOSPITAL_MORTALITY_DATABASES",
    "HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS",
    "HospitalMortalityFollowup",
    "HospitalMortalityFollowupError",
    "MIMIC_IV_HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS",
    "derive_eicu_hospital_mortality_followup",
    "derive_mimic_iv_hospital_mortality_followup",
]
