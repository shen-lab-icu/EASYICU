"""Explicit case-duration landmark risk-set mechanics, with nullable death data.

This is not an execution grant. Callers must bind source/time-axis authority,
the complete plan and the researcher's decision before reading patients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CareUnitLandmarkRiskSet:
    eligible: pd.Series
    unidentifiable: pd.Series
    closed_or_dead_at_landmark: pd.Series


def case_duration_landmark_risk_set(
    *, duration_hours: pd.Series, death_status: pd.Series, death_time_hours: pd.Series,
    landmark_hours: float,
) -> CareUnitLandmarkRiskSet:
    """Require a case to remain open and the patient alive *after* landmark.

    ``death_status`` must describe the same death event and follow-up horizon
    as ``death_time_hours``; a hospital-discharge status cannot be paired with
    a registry death time covering later follow-up. Both closure and death times are hours since the documented care-unit
    origin. Missing/nonfinite duration, missing death status, and deaths with
    missing death times remain unidentifiable. A contradictory non-death with
    a recorded death time also remains unidentifiable. This does not assert
    that case closure is ICU discharge or classify unknown status as survival.
    """
    if isinstance(landmark_hours, bool) or not np.isfinite(landmark_hours) or landmark_hours < 0:
        raise ValueError("landmark_hours_invalid")
    if (not duration_hours.index.equals(death_status.index)
            or not duration_hours.index.equals(death_time_hours.index)
            or not duration_hours.index.is_unique):
        raise ValueError("landmark_input_identity_mismatch")
    duration = pd.to_numeric(duration_hours, errors="coerce").astype(float)
    death_time = pd.to_numeric(death_time_hours, errors="coerce").astype(float)
    known_status = death_status.isin([True, False]) & death_status.notna()
    died = death_status.eq(True).fillna(False)
    alive = death_status.eq(False).fillna(False)
    valid_duration = pd.Series(np.isfinite(duration) & duration.ge(0), index=duration.index)
    valid_death_time = pd.Series(np.isfinite(death_time) & death_time.ge(0), index=death_time.index)
    identifiable = valid_duration & known_status & ((died & valid_death_time) | (alive & death_time_hours.isna()))
    after = duration.gt(landmark_hours) & (alive | (died & death_time.gt(landmark_hours)))
    eligible = (identifiable & after).astype(bool)
    return CareUnitLandmarkRiskSet(
        eligible=eligible,
        unidentifiable=(~identifiable).astype(bool),
        closed_or_dead_at_landmark=(identifiable & ~after).astype(bool),
    )
