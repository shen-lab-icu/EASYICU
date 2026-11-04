"""eICU ventilation windows from charted ventilator settings and care intervals.

eICU has no ventilation table with reliable stop times.  vent_ind used to keep
each piece of evidence as an isolated hour, so 92.6% of the hours with charted
ventilator parameters carried no vent_ind (2026-11-04 audit round 3, R3-05).
Following the eICU-CRD ventilation logic, an episode is

* a respiratoryCare row whose ventEndOffset follows its ventStartOffset (rows
  with both offsets 0 are placeholders and carry no time), or
* a run of ventilator evidence hours (charted ventilator settings or measured
  ventilator data, an invasive airway, a ventilation treatment) whose
  successive hours are at most ``MAX_GAP_HOURS`` apart.

A window ends at its last evidence hour; nothing is extrapolated beyond it, and
care intervals are clipped to ICU admission and unit discharge.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

ID_COLUMN = "patientunitstayid"
CARE_TABLE = "respiratorycare"
DISCHARGE_COLUMN = "unitdischargeoffset"
MAX_GAP_HOURS = 8

# respiratoryCharting labels that only a ventilator reports or is set with.
INVASIVE_SETTING_LABELS = frozenset(
    {
        "Tidal Volume (set)",
        "TV/kg IBW",
        "Vent Rate",
        "PEEP",
        "Pressure Support",
        "Pressure Control",
        "Pressure to Trigger PS",
        "Flow Sensitivity",
        "Peak Flow",
        "Peak Insp. Pressure",
        "Plateau Pressure",
        "Mean Airway Pressure",
        "Exhaled MV",
        "Exhaled TV (machine)",
        "Compliance",
    }
)
NONINVASIVE_SETTING_LABELS = frozenset(
    {"CPAP", "PEEP/CPAP", "B1: IPAP", "B2: EPAP", "B3: Est Mask Leak"}
)
# The respiratory therapist's ventilator on/off log; Start and Continued mean on.
VENT_ON_OFF_LABEL = "RT Vent On/Off"
VENT_ON_VALUES = frozenset({"start", "continued"})
SETTING_LABELS = INVASIVE_SETTING_LABELS | NONINVASIVE_SETTING_LABELS | {VENT_ON_OFF_LABEL}


def setting_evidence(
    frame: pd.DataFrame, *, label_column: str, value_column: str
) -> pd.DataFrame:
    """Keep charted ventilator settings with a positive value, as a mech_vent level."""
    labels = frame[label_column].astype("string").str.strip()
    raw = frame[value_column].astype("string").str.strip()
    vent_on = labels.eq(VENT_ON_OFF_LABEL).fillna(False) & raw.str.lower().isin(VENT_ON_VALUES).fillna(False)
    level = pd.Series(pd.NA, index=frame.index, dtype="object")
    invasive = labels.isin(INVASIVE_SETTING_LABELS).fillna(False) | vent_on
    level[invasive.to_numpy()] = "invasive"
    level[labels.isin(NONINVASIVE_SETTING_LABELS).fillna(False).to_numpy()] = "noninvasive"
    values = pd.to_numeric(raw, errors="coerce")
    valid = level.notna() & (values.gt(0).fillna(False) | vent_on)
    out = frame.loc[valid].copy()
    out[value_column] = level.loc[valid]
    return out


def care_intervals(
    care: pd.DataFrame, discharge_offsets: Optional[dict] = None
) -> pd.DataFrame:
    """Return ``[id, start_h, end_h]`` from respiratoryCare current and prior vent offsets."""
    pieces = []
    for start_col, end_col in (
        ("ventstartoffset", "ventendoffset"),
        ("priorventstartoffset", "priorventendoffset"),
    ):
        if start_col not in care.columns or end_col not in care.columns:
            continue
        start = pd.to_numeric(care[start_col], errors="coerce")
        end = pd.to_numeric(care[end_col], errors="coerce")
        valid = start.notna() & end.notna() & end.gt(start)
        pieces.append(
            pd.DataFrame(
                {
                    ID_COLUMN: care.loc[valid, ID_COLUMN].to_numpy(),
                    "start_h": start[valid].clip(lower=0).to_numpy() / 60.0,
                    "end_h": end[valid].to_numpy() / 60.0,
                }
            )
        )
    if not pieces:
        return pd.DataFrame(columns=[ID_COLUMN, "start_h", "end_h"])
    out = pd.concat(pieces, ignore_index=True)
    if discharge_offsets:
        limit = out[ID_COLUMN].map(discharge_offsets).astype("float64") / 60.0
        out["end_h"] = np.where(limit.notna(), np.minimum(out["end_h"], limit), out["end_h"])
    out = out[out["end_h"] > out["start_h"]]
    return out.drop_duplicates().reset_index(drop=True)


def bridge_hours(
    point_hours: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    id_column: str = ID_COLUMN,
    max_gap_hours: int = MAX_GAP_HOURS,
) -> pd.DataFrame:
    """Return ``[id, hour]`` covering every evidence hour and the gaps of at most ``max_gap_hours``."""
    hours = [point_hours[[id_column, "hour"]]]
    if not intervals.empty:
        start = np.floor(intervals["start_h"].to_numpy()).astype(np.int64)
        end = np.floor(intervals["end_h"].to_numpy()).astype(np.int64)
        lengths = end - start + 1
        ids = np.repeat(intervals[id_column].to_numpy(), lengths)
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        hours.append(pd.DataFrame({id_column: ids, "hour": np.repeat(start, lengths) + offsets}))
    work = pd.concat(hours, ignore_index=True).dropna()
    if work.empty:
        return pd.DataFrame(columns=[id_column, "hour"])
    work["hour"] = work["hour"].astype(np.int64)
    work = work.drop_duplicates().sort_values([id_column, "hour"]).reset_index(drop=True)
    nxt = work.groupby(id_column)["hour"].shift(-1)
    gap = (nxt - work["hour"]).where(lambda g: g.le(max_gap_hours), 1).fillna(1).astype(np.int64)
    ids = np.repeat(work[id_column].to_numpy(), gap)
    offsets = np.arange(gap.sum()) - np.repeat(np.cumsum(gap) - gap, gap)
    return pd.DataFrame(
        {id_column: ids, "hour": np.repeat(work["hour"].to_numpy(), gap) + offsets}
    )


def load_care_intervals(data_source, stays: Iterable[object]) -> pd.DataFrame:
    """Load the respiratoryCare vent offsets and unit discharge of ``stays``."""
    from ..datasource import FilterOp, FilterSpec

    stays = list(stays)
    if not stays:
        return pd.DataFrame(columns=[ID_COLUMN, "start_h", "end_h"])
    columns = [
        ID_COLUMN,
        "ventstartoffset",
        "ventendoffset",
        "priorventstartoffset",
        "priorventendoffset",
    ]
    care = data_source.load_table(
        CARE_TABLE,
        columns=columns,
        filters=[FilterSpec(ID_COLUMN, FilterOp.IN, stays)],
        verbose=False,
    )
    care = care.data if hasattr(care, "data") else care
    patients = data_source.load_table(
        "patient",
        columns=[ID_COLUMN, DISCHARGE_COLUMN],
        filters=[FilterSpec(ID_COLUMN, FilterOp.IN, stays)],
        verbose=False,
    )
    patients = patients.data if hasattr(patients, "data") else patients
    discharge = dict(
        zip(patients[ID_COLUMN], pd.to_numeric(patients[DISCHARGE_COLUMN], errors="coerce"))
    )
    return care_intervals(care, discharge)


__all__ = [
    "MAX_GAP_HOURS",
    "SETTING_LABELS",
    "bridge_hours",
    "care_intervals",
    "load_care_intervals",
    "setting_evidence",
]
