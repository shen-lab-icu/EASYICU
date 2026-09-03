"""Deterministic MIMIC-IV hospital-mortality follow-up construction.

The prepared EasyICU ``death`` concept is an event-status flag.  It does not,
by itself, establish the time axis needed by a longitudinal survival analysis:
ICU length of stay ends at ICU discharge and cannot censor an in-hospital
mortality endpoint.  This owner derives that axis only from the raw MIMIC-IV
``icu.icustays`` and ``hosp.admissions`` tables supplied by a separately
verified source binding.

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


MIMIC_IV_HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS = (
    "stay_id",
    "hospital_death",
    "death_time_hours",
    "hospital_followup_time_hours",
)


class HospitalMortalityFollowupError(ValueError):
    """The raw MIMIC-IV source cannot supply an unambiguous follow-up axis."""

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
        if tuple(self.frame.columns) != MIMIC_IV_HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS:
            raise HospitalMortalityFollowupError(
                "hospital_followup_output_schema_invalid",
                "The hospital follow-up output does not use the canonical columns.",
            )
        if tuple(self.exclusions.columns) != ("stay_id", "reason_code"):
            raise HospitalMortalityFollowupError(
                "hospital_followup_exclusion_schema_invalid",
                "The hospital follow-up exclusions do not use the canonical columns.",
            )


def _require_columns(frame: pd.DataFrame, *, label: str, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise HospitalMortalityFollowupError(
            f"hospital_followup_{label}_columns_missing",
            f"The raw {label} table lacks required columns: {', '.join(missing)}.",
        )


def _require_unique_nonmissing_key(
    frame: pd.DataFrame, *, label: str, key: str
) -> None:
    values = frame[key]
    if bool(values.isna().any()):
        raise HospitalMortalityFollowupError(
            f"hospital_followup_{label}_key_missing",
            f"The raw {label} table has missing {key} values.",
        )
    if bool(values.duplicated().any()):
        raise HospitalMortalityFollowupError(
            f"hospital_followup_{label}_key_nonunique",
            f"The raw {label} table has non-unique {key} values.",
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

    if not isinstance(icustays, pd.DataFrame) or not isinstance(admissions, pd.DataFrame):
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
    _require_unique_nonmissing_key(icustays, label="icustays", key="stay_id")
    _require_unique_nonmissing_key(admissions, label="admissions", key="hadm_id")

    stay_rows = icustays[["stay_id", "hadm_id", "intime"]].copy()
    stay_rows["__input_order"] = np.arange(len(stay_rows), dtype=np.int64)
    admission_rows = admissions[
        ["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"]
    ].copy()
    joined = stay_rows.merge(
        admission_rows,
        on="hadm_id",
        how="left",
        sort=False,
        validate="many_to_one",
        indicator="__admission_match",
    )
    joined = joined.sort_values("__input_order", kind="stable").reset_index(drop=True)

    intime, invalid_intime = _timestamps(joined["intime"])
    dischtime, invalid_dischtime = _timestamps(joined["dischtime"])
    deathtime, invalid_deathtime = _timestamps(joined["deathtime"])
    raw_event = pd.to_numeric(joined["hospital_expire_flag"], errors="coerce")
    event_flag_valid = raw_event.isin([0, 1]) & np.isfinite(raw_event)
    event = raw_event.eq(1)
    censor = raw_event.eq(0)

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

    valid = reason.eq("")
    valid_rows = joined.loc[valid, ["stay_id"]].copy()
    valid_rows["hospital_death"] = event.loc[valid].astype("int8").to_numpy()
    valid_rows["death_time_hours"] = np.where(
        event.loc[valid].to_numpy(),
        followup_hours.loc[valid].to_numpy(dtype=float),
        np.nan,
    )
    valid_rows["hospital_followup_time_hours"] = followup_hours.loc[valid].to_numpy(
        dtype=float
    )
    valid_rows = valid_rows.loc[:, MIMIC_IV_HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS]

    exclusions = joined.loc[~valid, ["stay_id"]].copy()
    exclusions["reason_code"] = reason.loc[~valid].astype(str).to_numpy()
    exclusions = exclusions.loc[:, ["stay_id", "reason_code"]]
    exclusion_counts = {
        str(code): int(count)
        for code, count in exclusions["reason_code"].value_counts(sort=False).items()
    }
    exclusion_counts = dict(sorted(exclusion_counts.items()))
    zero_time_events = int(
        (
            valid_rows["hospital_death"].eq(1)
            & valid_rows["death_time_hours"].eq(0.0)
        ).sum()
    )
    zero_time_censoring = int(
        (
            valid_rows["hospital_death"].eq(0)
            & valid_rows["hospital_followup_time_hours"].eq(0.0)
        ).sum()
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
            "rule": (
                "event_at_deathtime_else_censor_at_hospital_discharge_time"
            ),
            "source": "admissions.dischtime - icustays.intime",
        },
        "input_stays": int(len(stay_rows)),
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
    "HospitalMortalityFollowup",
    "HospitalMortalityFollowupError",
    "MIMIC_IV_HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS",
    "derive_mimic_iv_hospital_mortality_followup",
]
