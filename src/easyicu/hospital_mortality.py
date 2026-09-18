"""MIMIC hospital-discharge status, independent of an event/censor clock.

One admission status applies to every linked ICU stay. Missing timestamps do
not erase known status; missing/invalid flags never establish survival. This
leaf owns that meaning for extraction, patient filters and research adapters.
It does not discover files, select a cohort or authorize an analysis.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

MIMIC_HOSPITAL_STATUS_BINDING = "mimic_hospital_mortality_status_v1"


class HospitalMortalityStatusError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def hospital_mortality_status_from_flag(values: pd.Series) -> pd.Series:
    flag = pd.to_numeric(values, errors="coerce")
    status = pd.Series(pd.NA, index=values.index, dtype="boolean")
    status.loc[flag.eq(0)] = False
    status.loc[flag.eq(1)] = True
    return status


def join_mimic_hospital_admissions(
    icustays: pd.DataFrame, admissions: pd.DataFrame
) -> pd.DataFrame:
    """A cardinality-checked, order-preserving admission-to-stay projection."""
    for frame, label, key, required in (
        (icustays, "icustays", "stay_id", {"stay_id", "hadm_id"}),
        (admissions, "admissions", "hadm_id", {"hadm_id", "hospital_expire_flag"}),
    ):
        if not required.issubset(frame.columns):
            raise HospitalMortalityStatusError(
                f"hospital_status_{label}_columns_missing",
                "Required source columns are absent.",
            )
        if frame[key].isna().any() or frame[key].duplicated().any():
            reason = "missing" if frame[key].isna().any() else "nonunique"
            raise HospitalMortalityStatusError(
                f"hospital_status_{label}_key_{reason}", "Source identity is ambiguous."
            )
    stay_columns = [c for c in ("stay_id", "hadm_id", "intime") if c in icustays]
    admission_columns = [
        c
        for c in ("hadm_id", "hospital_expire_flag", "deathtime", "dischtime")
        if c in admissions
    ]
    return (
        icustays[stay_columns]
        .merge(
            admissions[admission_columns],
            on="hadm_id",
            how="left",
            sort=False,
            validate="many_to_one",
            indicator="__admission_match",
        )
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class HospitalMortalityStatus:
    frame: pd.DataFrame
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(self.frame.columns) != ("stay_id", "hospital_death"):
            raise HospitalMortalityStatusError(
                "hospital_status_output_schema_invalid",
                "Canonical status columns are required.",
            )
        values = self.frame.hospital_death
        if not values.dropna().isin([0, 1]).all():
            raise HospitalMortalityStatusError(
                "hospital_status_output_domain_invalid",
                "Status must be 0, 1 or unknown.",
            )


def derive_mimic_hospital_mortality_status(
    icustays: pd.DataFrame, admissions: pd.DataFrame
) -> HospitalMortalityStatus:
    joined = join_mimic_hospital_admissions(icustays, admissions)
    frame = joined[["stay_id"]].copy()
    frame["hospital_death"] = hospital_mortality_status_from_flag(
        joined.hospital_expire_flag
    )
    return HospitalMortalityStatus(
        frame=frame,
        receipt={
            "schema_version": "easyicu.hospital_mortality_status/1",
            "clinical_binding": MIMIC_HOSPITAL_STATUS_BINDING,
            "analysis_unit": "icu_stay",
            "status_source": "admissions.hospital_expire_flag",
            "linkage": "icustays.hadm_id -> admissions.hadm_id (many_to_one)",
            "clock_required": False,
            "unknown_policy": "retain_null",
            "source_stays": len(frame),
            "event_stays": int(frame.hospital_death.sum()),
            "unknown_status_stays": int(frame.hospital_death.isna().sum()),
            "unlinked_stays": int(joined.__admission_match.ne("both").sum()),
        },
    )
