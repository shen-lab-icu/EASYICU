"""Source rules that time a recorded death, shared by the death concept and outcomes.

Both the in-hospital ``death`` concept and the fixed-horizon mortality loader
read the same source fields; each rule here is applied by both so the two never
disagree about whether, or when, a stay's patient died.
"""

from __future__ import annotations

import pandas as pd

# SICdb codes (cases.HospitalDischargeType, cases.DischargeState).
SIC_HOSPITAL_DECEASED = 2028
SIC_HOSPITAL_SURVIVED = 2026
SIC_ICU_DECEASED = 2215


def aumc_registry_death_after_discharge(dod: pd.Series, dis: pd.Series, *, day: float) -> pd.Series:
    """Place an AmsterdamUMCdb registry death no earlier than ICU discharge.

    ``dateofdeath`` is the midnight of the calendar day of death.  A ward death
    on the day of ICU discharge therefore falls up to a day before it; it is
    timed at the discharge.  A registry date more than a day before the
    discharge of a stay that left the ICU is not a death of that stay and is
    treated as missing (2026-11-04 audit round 3, R3-13).  ``day`` is one day
    in the unit of ``dod`` and ``dis``.
    """
    lead = dis - dod
    same_day = dis.notna() & lead.gt(0) & lead.le(day)
    corrupt = dis.notna() & lead.gt(day)
    return dod.where(~same_day, dis).where(~corrupt)


def sic_recorded_death(hospital_discharge: pd.Series, discharge_state: pd.Series) -> pd.Series:
    """True where SICdb records a death in hospital (2028) or at ICU discharge (2215).

    34 in-hospital deaths carry no OffsetOfDeath; they are deaths all the same
    (2026-11-04 audit round 3, R3-15).
    """
    hospital = pd.to_numeric(hospital_discharge, errors="coerce")
    state = pd.to_numeric(discharge_state, errors="coerce")
    return (hospital.eq(SIC_HOSPITAL_DECEASED) | state.eq(SIC_ICU_DECEASED)).fillna(False)


__all__ = [
    "SIC_HOSPITAL_DECEASED",
    "SIC_HOSPITAL_SURVIVED",
    "SIC_ICU_DECEASED",
    "aumc_registry_death_after_discharge",
    "sic_recorded_death",
]
