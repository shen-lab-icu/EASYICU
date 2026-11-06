"""Source rules that time a recorded death, shared by the death concept and outcomes.

Both the in-hospital ``death`` concept and the fixed-horizon mortality loader
read the same source fields; each rule here is applied by both so the two never
disagree about whether, or when, a stay's patient died.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd

# SICdb codes are release-specific: 1.0.6 records a hospital death as 2028
# "Deceased", 1.0.8 as 3130 "Sterbefall" and splits survival into six
# German discharge types (3129, 3131-3134). They are therefore read from
# d_references by label, never assumed.
SIC_DECEASED_DISCHARGE_LABELS = frozenset({"deceased", "sterbefall"})
SIC_ICU_DECEASED_STATE_LABELS = frozenset({"verstorben"})
SIC_REFERENCE_UNAVAILABLE = "sic_discharge_reference_unavailable"


class SicReferenceUnavailable(ValueError):
    """d_references cannot establish SICdb discharge semantics."""

    reason_code = SIC_REFERENCE_UNAVAILABLE

    def __init__(self, detail: str) -> None:
        super().__init__(f"{SIC_REFERENCE_UNAVAILABLE}: {detail}")
        self.detail = detail


@dataclass(frozen=True)
class SicDischargeCodes:
    """SICdb disposition codes resolved from one release's d_references."""

    deceased: frozenset[int]
    survived: frozenset[int]
    icu_deceased: frozenset[int]
    followup_days: dict[int, float] = field(default_factory=dict)

    @property
    def known(self) -> frozenset[int]:
        return self.deceased | self.survived


def _label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def sic_discharge_codes(references: pd.DataFrame) -> SicDischargeCodes:
    """Resolve discharge, ICU-death and follow-up codes from d_references labels.

    ``DischargeType``: a "Deceased" (1.0.6) or "Sterbefall" (1.0.8) label is a
    hospital death; every other discharge type is alive at hospital discharge.
    ``DischargeState`` "verstorben" is a death at ICU discharge.
    ``EstimatedSurvivalObservationTime`` "6 Months" / "1 Year" are 180 / 365
    days. A table that cannot name a death code and a survival code fails
    closed with ``sic_discharge_reference_unavailable``.
    """
    columns = {str(c).casefold(): c for c in references.columns}
    try:
        ids = pd.to_numeric(references[columns["referenceglobalid"]], errors="coerce")
        names = references[columns["referencename"]].map(_label)
        labels = references[columns["referencevalue"]].map(_label)
    except KeyError as exc:
        raise SicReferenceUnavailable(f"d_references lacks {exc}") from exc
    frame = pd.DataFrame({"id": ids, "name": names, "label": labels}).dropna(subset=["id"])
    frame = frame.drop_duplicates()
    frame["id"] = frame["id"].astype(np.int64)

    def rows(name):
        return frame.loc[frame["name"].eq(name.casefold())]

    discharge = rows("DischargeType")
    if discharge["id"].duplicated().any():
        raise SicReferenceUnavailable("a DischargeType code carries two labels")
    dead = discharge["label"].isin(SIC_DECEASED_DISCHARGE_LABELS)
    deceased = frozenset(int(v) for v in discharge.loc[dead, "id"])
    survived = frozenset(int(v) for v in discharge.loc[~dead, "id"])
    if not deceased or not survived:
        raise SicReferenceUnavailable("DischargeType has no deceased or no surviving label")
    state = rows("DischargeState")
    icu_deceased = frozenset(
        int(v) for v in state.loc[state["label"].isin(SIC_ICU_DECEASED_STATE_LABELS), "id"]
    )
    if not icu_deceased:
        raise SicReferenceUnavailable("DischargeState has no 'verstorben' label")
    followup = {}
    for code, label in rows("EstimatedSurvivalObservationTime")[["id", "label"]].itertuples(index=False):
        if re.fullmatch(r"6 months?", label):
            followup[int(code)] = 180.0
        elif re.fullmatch(r"1 year", label):
            followup[int(code)] = 365.0
    return SicDischargeCodes(deceased, survived, icu_deceased, followup)


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


def sic_recorded_death(
    hospital_discharge: pd.Series, discharge_state: pd.Series, codes: SicDischargeCodes,
) -> pd.Series:
    """True where SICdb records a death in hospital or at ICU discharge.

    The codes come from :func:`sic_discharge_codes` for the release read.
    34 in-hospital deaths in 1.0.6 carry no OffsetOfDeath; they are deaths
    all the same (2026-11-04 audit round 3, R3-15).
    """
    hospital = pd.to_numeric(hospital_discharge, errors="coerce")
    state = pd.to_numeric(discharge_state, errors="coerce")
    return (hospital.isin(codes.deceased) | state.isin(codes.icu_deceased)).fillna(False)


__all__ = [
    "SIC_DECEASED_DISCHARGE_LABELS",
    "SIC_ICU_DECEASED_STATE_LABELS",
    "SIC_REFERENCE_UNAVAILABLE",
    "SicDischargeCodes",
    "SicReferenceUnavailable",
    "sic_discharge_codes",
    "aumc_registry_death_after_discharge",
    "sic_recorded_death",
]
