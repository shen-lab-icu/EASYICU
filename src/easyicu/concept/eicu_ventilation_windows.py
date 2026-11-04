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

# respiratoryCharting labels that only an invasive ventilator is set with.
INVASIVE_SETTING_LABELS = frozenset(
    {
        "Tidal Volume (set)",
        "TV/kg IBW",
        "Vent Rate",
        "Pressure Control",
        "Plateau Pressure",
        "Exhaled TV (machine)",
    }
)
# Settings and measurements an NIV ventilator reports as well.  They take the
# level of the nearest invasive or NIV setting of the stay within
# MAX_GAP_HOURS, invasive first, and carry no level otherwise: as invasive
# evidence they made NIV stays invasive (2026-11-04 audit round 4, R4-024).
SHARED_SETTING_LABELS = frozenset(
    {
        "PEEP",
        "Pressure Support",
        "Pressure to Trigger PS",
        "Flow Sensitivity",
        "Peak Flow",
        "Peak Insp. Pressure",
        "Mean Airway Pressure",
        "Exhaled MV",
        "Compliance",
    }
)
NONINVASIVE_SETTING_LABELS = frozenset(
    {"CPAP", "PEEP/CPAP", "B1: IPAP", "B2: EPAP", "B3: Est Mask Leak"}
)
# The respiratory therapist's ventilator on/off log; Start and Continued mean on.
VENT_ON_OFF_LABEL = "RT Vent On/Off"
VENT_ON_VALUES = frozenset({"start", "continued"})
SETTING_LABELS = (
    INVASIVE_SETTING_LABELS | SHARED_SETTING_LABELS | NONINVASIVE_SETTING_LABELS | {VENT_ON_OFF_LABEL}
)


def setting_evidence(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    id_column: Optional[str] = None,
    time_column: Optional[str] = None,
    data_source=None,
) -> pd.DataFrame:
    """Keep charted ventilator settings with a positive value, as a mech_vent level.

    ``time_column`` is the respiratoryCharting offset in minutes; without it a
    shared setting cannot be placed and carries no level.
    """
    labels = frame[label_column].astype("string").str.strip()
    raw = frame[value_column].astype("string").str.strip()
    vent_on = labels.eq(VENT_ON_OFF_LABEL).fillna(False) & raw.str.lower().isin(VENT_ON_VALUES).fillna(False)
    values = pd.to_numeric(raw, errors="coerce")
    positive = values.gt(0).fillna(False) | vent_on
    level = pd.Series(pd.NA, index=frame.index, dtype="object")
    invasive = labels.isin(INVASIVE_SETTING_LABELS).fillna(False) | vent_on
    level[invasive.to_numpy()] = "invasive"
    level[labels.isin(NONINVASIVE_SETTING_LABELS).fillna(False).to_numpy()] = "noninvasive"
    shared = labels.isin(SHARED_SETTING_LABELS).fillna(False) & positive
    if shared.any() and time_column in frame.columns and (id_column or ID_COLUMN) in frame.columns:
        level.loc[shared] = _shared_setting_levels(
            frame, level.where(positive), shared, id_column or ID_COLUMN, time_column,
            data_source=data_source,
        )
    valid = level.notna() & positive
    out = frame.loc[valid].copy()
    out[value_column] = level.loc[valid]
    out.attrs["eicu_ventilation_attribution"] = {"shared_rows": int(shared.sum()), "unknown_rows": int((shared & level.isna()).sum()), "rule": "most recent cross-source anchor within 8h; NIV/stop terminate invasive attribution"}
    return out


def _shared_setting_levels(frame, level, shared, id_column, time_column, data_source=None) -> pd.Series:
    """Join shared settings to chronological airway/device/treatment evidence."""
    minutes = pd.to_numeric(frame[time_column], errors="coerce").astype("float64")
    tolerance = MAX_GAP_HOURS * 60.0
    rows = pd.DataFrame({"_id": frame[id_column], "_t": minutes, "_row": np.arange(len(frame))})
    target = rows.loc[shared.to_numpy() & minutes.notna().to_numpy()].sort_values("_t")
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    anchor = rows.loc[level.notna().to_numpy() & minutes.notna().to_numpy(), ["_id", "_t"]].copy()
    anchor["_level"] = level.loc[level.notna() & minutes.notna()].to_numpy()
    external = load_ventilation_anchors(data_source, frame[id_column].dropna().unique())
    if not external.empty:
        anchor = pd.concat([anchor, external], ignore_index=True)
    if anchor.empty or target.empty:
        return result.loc[shared]
    # A documented NIV/stop wins a same-time conflict. A later invasive anchor
    # can start a new episode; an older airway cannot override a recent NIV.
    anchor["_rank"] = anchor._level.map({"invasive": 0, "noninvasive": 1, "stopped": 2})
    anchor = anchor.sort_values(["_t", "_rank"]).drop_duplicates(["_id", "_t"], keep="last")
    near = pd.merge_asof(target, anchor[["_id", "_t", "_level"]], on="_t", by="_id", direction="backward", tolerance=tolerance)
    # Before the first anchor only: a near future airway may corroborate a
    # setting, but it cannot bridge backward over an explicit NIV/stop.
    missing = near._level.isna()
    future = pd.merge_asof(target, anchor[["_id", "_t", "_level"]], on="_t", by="_id", direction="forward", tolerance=tolerance)
    has_prior = pd.merge_asof(target, anchor[["_id", "_t", "_level"]], on="_t", by="_id", direction="backward")._level.notna()
    near.loc[missing & ~has_prior, "_level"] = future.loc[missing & ~has_prior, "_level"]
    hit = near._level.isin(["invasive", "noninvasive"])
    result.iloc[near.loc[hit, "_row"].to_numpy()] = near.loc[hit, "_level"].to_numpy()
    return result.loc[shared]


def load_ventilation_anchors(data_source, stays) -> pd.DataFrame:
    """Raw-minute anchors across sources; stop evidence remains a barrier."""
    from .eicu_source_qc import load_optional

    pieces = []
    care = load_optional(data_source, CARE_TABLE, stays, [ID_COLUMN, "airwaytype", "ventstartoffset", "ventendoffset"])
    if not care.empty:
        invasive = care.airwaytype.isin(["Oral ETT", "Nasal ETT", "Double-Lumen Tube", "Cricothyrotomy"])
        start = pd.to_numeric(care.ventstartoffset, errors="coerce").clip(lower=0)
        for mask, time, level in [(invasive & start.notna(), start, "invasive")]:
            pieces.append(pd.DataFrame({"_id": care.loc[mask, ID_COLUMN], "_t": time.loc[mask], "_level": level}))
        if "ventendoffset" in care:
            end = pd.to_numeric(care.ventendoffset, errors="coerce")
            mask = invasive & end.gt(start)
            pieces.append(pd.DataFrame({"_id": care.loc[mask, ID_COLUMN], "_t": end.loc[mask], "_level": "stopped"}))
    for table, time_col, value_col in [("respiratorycharting", "respchartoffset", "respchartvalue"), ("treatment", "treatmentoffset", "treatmentstring")]:
        columns = [ID_COLUMN, time_col, value_col]
        if table == "respiratorycharting":
            columns.append("respchartvaluelabel")
        raw = load_optional(data_source, table, stays, columns)
        if raw.empty:
            continue
        text = raw[value_col].astype("string").str.strip().str.casefold()
        level = pd.Series(pd.NA, index=raw.index, dtype="object")
        if table == "respiratorycharting":
            device = raw.respchartvaluelabel.isin(["O2 Device", "Oxygen Delivery Method"])
            numeric = pd.to_numeric(raw[value_col], errors="coerce").gt(0)
            level.loc[raw.respchartvaluelabel.isin(INVASIVE_SETTING_LABELS) & numeric] = "invasive"
            level.loc[raw.respchartvaluelabel.isin(NONINVASIVE_SETTING_LABELS) & numeric] = "noninvasive"
            level.loc[raw.respchartvaluelabel.eq(VENT_ON_OFF_LABEL) & text.isin(VENT_ON_VALUES)] = "invasive"
            level.loc[device & text.isin(["ventilator", "mechanical ventilator", "ett"])] = "invasive"
            level.loc[device & text.str.contains(r"\b(?:bi-?pap|cpap|niv)\b", na=False)] = "noninvasive"
            level.loc[device & text.str.contains(r"extubat\w*|\bhfnc\b|high[- ]?flow", na=False)] = "stopped"
            level.loc[raw.respchartvaluelabel.eq(VENT_ON_OFF_LABEL) & text.isin(["stop", "off", "discontinued"])] = "stopped"
        else:
            niv = text.str.contains(r"non[- ]?invasive ventilation", na=False)
            level.loc[text.str.contains("mechanical ventilation", regex=False, na=False) & ~niv] = "invasive"
            level.loc[niv] = "noninvasive"
            level.loc[text.str.contains(r"extubat\w*|ventilat\w*.*(?:discontinu\w*|stop\w*)", na=False)] = "stopped"
        valid = level.notna()
        pieces.append(pd.DataFrame({"_id": raw.loc[valid, ID_COLUMN], "_t": pd.to_numeric(raw.loc[valid, time_col], errors="coerce"), "_level": level.loc[valid]}))
    return pd.concat(pieces, ignore_index=True).dropna() if pieces else pd.DataFrame(columns=["_id", "_t", "_level"])


def cut_hours_at_barriers(hours, point_hours, intervals, data_source, stays, id_column):
    """Do not bridge across NIV/extubation until new invasive evidence arrives."""
    anchors = load_ventilation_anchors(data_source, stays)
    barrier = anchors.loc[anchors._level.ne("invasive")].copy()
    if hours.empty or barrier.empty:
        return hours
    starts = point_hours.rename(columns={id_column: "_id", "hour": "_t"})[["_id", "_t"]].copy()
    if not intervals.empty:
        starts = pd.concat([starts, intervals[[id_column, "start_h"]].rename(columns={id_column: "_id", "start_h": "_t"})], ignore_index=True)
    starts["_t"] *= 60
    starts["_active"] = True
    # Retain raw-minute ordering: NIV at minute 60 cannot cancel a new ETT
    # at minute 90. Exact simultaneous conflicts still favor the barrier.
    anchors["_active"] = anchors._level.eq("invasive")
    events = pd.concat([starts, anchors[["_id", "_t", "_active"]]], ignore_index=True)
    events = events.sort_values(["_t", "_active"], ascending=[True, False]).drop_duplicates(["_id", "_t"], keep="last")
    target = hours.rename(columns={id_column: "_id", "hour": "_h"}).sort_values("_h")
    target["_t"] = (target._h.astype(float) + 1) * 60
    events["_t"] = events._t.astype(float)
    joined = pd.merge_asof(target, events, by="_id", on="_t", direction="backward", allow_exact_matches=False)
    return joined.loc[joined._active.eq(True), ["_id", "_h"]].rename(columns={"_id": id_column, "_h": "hour"})  # noqa: E712


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
    from .eicu_source_qc import load_optional

    care = load_optional(data_source, CARE_TABLE, stays, columns)
    if care.empty:
        return pd.DataFrame(columns=[ID_COLUMN, "start_h", "end_h"])
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
    "SHARED_SETTING_LABELS",
    "bridge_hours",
    "care_intervals",
    "load_care_intervals",
    "setting_evidence",
]
