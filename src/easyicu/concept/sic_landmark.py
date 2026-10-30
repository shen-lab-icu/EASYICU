"""SICdb case-interval fields used by an explicitly reviewed landmark plan.

All raw offsets are seconds from first admission. The care-unit origin can
include intermediate care; neither the interval nor status is ICU-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIC_CASE_LANDMARK_COLUMNS = frozenset({
    "care_unit_to_case_closure_days", "death_time_during_case_hours", "death_during_case",
})
SIC_CASE_SOURCE_COLUMNS = ("CaseID", "ICUOffset", "TimeOfStay", "OffsetOfDeath", "HospitalDischargeType")


def sic_case_landmark_fields(cases: pd.DataFrame) -> pd.DataFrame:
    """Derive three nullable fields without interpreting unknown as survival."""
    missing = set(SIC_CASE_SOURCE_COLUMNS) - set(cases.columns)
    if missing:
        raise ValueError(f"SIC case landmark source fields missing: {sorted(missing)}")
    if cases["CaseID"].isna().any() or cases["CaseID"].duplicated().any():
        raise ValueError("SIC case landmark source requires unique non-null CaseID")
    origin = pd.to_numeric(cases["ICUOffset"], errors="coerce")
    end = pd.to_numeric(cases["TimeOfStay"], errors="coerce")
    # An all-null Parquet column may arrive as datetime NaT; numeric
    # conversion alone turns NaT into an integer sentinel, not missing.
    event_absent = cases["OffsetOfDeath"].isna()
    event = pd.to_numeric(cases["OffsetOfDeath"], errors="coerce").where(~event_absent)
    disposition = pd.to_numeric(cases["HospitalDischargeType"], errors="coerce")
    valid_case = np.isfinite(origin) & np.isfinite(end) & (origin >= 0) & (end >= origin)
    duration = ((end - origin) / 86400).where(valid_case)
    valid_event = pd.Series(np.isfinite(event), index=cases.index)
    inside = valid_case & valid_event & (event >= origin) & (event <= end)
    contradiction = (valid_event & (event < origin)) | (inside & (disposition == 2026))
    known_status = valid_case & disposition.isin([2026, 2028]) & ~contradiction
    known_status &= valid_event | (event_absent & (disposition == 2026))
    status = inside.astype("boolean").where(known_status, pd.NA)
    death_time = ((event - origin) / 3600).where(status.fillna(False))
    return pd.DataFrame({
        "CaseID": cases["CaseID"], "care_unit_to_case_closure_days": duration,
        "death_time_during_case_hours": death_time, "death_during_case": status,
    })
