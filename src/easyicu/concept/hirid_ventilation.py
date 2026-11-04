"""HiRID airway intervals and evidence constrained by explicit stop states."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ventilator_evidence(ctx, id_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PEEP plus a measured companion; standby and NIV block invasive bridges."""
    names = ["peep", "etco2", "tidal_vol", "vent_mode"]
    loaded = ctx.resolver.load_concepts(
        names, ctx.data_source, merge=False, patient_ids=ctx.patient_ids,
        verbose=False, interval=pd.Timedelta(minutes=1),
    )

    def points(name: str, positive: bool = False) -> pd.DataFrame:
        table = loaded.get(name) if isinstance(loaded, dict) else None
        if table is None or table.data.empty or id_col not in table.data:
            return pd.DataFrame(columns=[id_col, "hour"])
        data = table.data
        value = pd.to_numeric(data[table.value_column or name], errors="coerce")
        keep = value.gt(0) if positive else value.notna()
        time = pd.to_numeric(data[table.index_column], errors="coerce")
        return pd.DataFrame({id_col: data[id_col], "hour": np.floor(time)}).loc[keep].dropna()

    measured = pd.concat([points("etco2"), points("tidal_vol")]).drop_duplicates()
    evidence = points("peep", True).merge(measured, on=[id_col, "hour"]).drop_duplicates()
    barriers = pd.DataFrame(columns=[id_col, "start", "end"])
    mode = loaded.get("vent_mode") if isinstance(loaded, dict) else None
    if mode is not None and not mode.data.empty:
        data = mode.data.sort_values([id_col, mode.index_column], kind="stable")
        start = pd.to_numeric(data[mode.index_column], errors="coerce")
        following = start.groupby(data[id_col]).shift(-1)
        end = following.where(following.sub(start).le(12), start + 4).fillna(start + 4)
        off = data[mode.value_column or "vent_mode"].astype("string").str.fullmatch(
            r"(?i)\s*(?:stand\s*by|off|stopped|no support)\s*", na=False,
        )
        if "vent_mode_interface" in data:
            off |= data.vent_mode_interface.eq("niv").fillna(False)
        barriers = pd.DataFrame({id_col: data[id_col], "start": start, "end": end}).loc[off]
    return evidence, barriers



def _finite(values: pd.Series) -> np.ndarray:
    """np.isfinite for nullable or object columns (pandas 3 keeps Float64/object here)."""
    return np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", na_value=np.nan))


def airway_hours(
    airway, evidence: pd.DataFrame, barriers: pd.DataFrame, *, id_col: str,
    time_col: str, max_gap_hours: float = 8,
) -> pd.DataFrame:
    """Expand reliable [start,end) windows and bridge only unopposed evidence.

    All barriers operate on exact intervals. An hour with an invasive-to-NIV
    transition retains its genuine invasive overlap, while the later NIV
    interval prevents a bridge from continuing through that transition.
    """
    data = airway.data
    index = airway.index_column or time_col
    start = pd.to_numeric(data[index], errors="coerce")
    duration_col = getattr(airway, "dur_var", None)
    if duration_col not in data.columns:
        duration_col = next((c for c in ("dur_var", "duration", "mech_vent_dur") if c in data), None)
    if duration_col:
        raw = data[duration_col]
        duration = raw.dt.total_seconds() / 3600 if pd.api.types.is_timedelta64_dtype(raw) else pd.to_numeric(raw, errors="coerce")
        end = start + duration
    else:
        # Already expanded points represent just their occupied hour.
        end = np.floor(start) + 1
    level_col = "mech_vent" if "mech_vent" in data else (airway.value_column or "vent_ind")
    levels = data[level_col].astype("string").str.strip().str.lower()
    invasive = levels.eq("invasive").fillna(False) if level_col == "mech_vent" else data[level_col].eq(True)
    intervals = pd.DataFrame({id_col: data[id_col], "start": start, "end": end, "seed": invasive})
    intervals = intervals.loc[_finite(start) & _finite(end) & end.gt(start)]
    blockers = pd.concat([barriers, intervals.loc[~intervals.seed, [id_col, "start", "end"]]], ignore_index=True)
    nodes = intervals.loc[intervals.seed, [id_col, "start", "end"]]
    stays = nodes[id_col].unique()
    measured = evidence.loc[evidence[id_col].isin(stays), [id_col, "hour"]].drop_duplicates()
    measured = measured.rename(columns={"hour": "start"})
    measured["end"] = measured.start + 1
    spans = pd.concat([
        nodes.assign(seed=1, measured=0, blocked=0),
        measured.assign(seed=0, measured=1, blocked=0),
        blockers.loc[blockers[id_col].isin(stays)].assign(seed=0, measured=0, blocked=1),
    ], ignore_index=True)
    spans = spans.loc[_finite(spans.start) & _finite(spans.end) & spans.end.gt(spans.start)]
    if spans.empty:
        return pd.DataFrame(columns=[id_col, time_col])

    # Sweep interval boundaries once, instead of scanning every patient's
    # full evidence/barrier table and subtracting each barrier from each hour.
    # Real invasive overlap survives an opposing state; inferred evidence does not.
    flags = ["seed", "measured", "blocked"]
    opening = spans[[id_col, "start", *flags]].rename(columns={"start": "time"})
    closing = spans[[id_col, "end", *flags]].rename(columns={"end": "time"})
    closing[flags] *= -1
    events = pd.concat([opening, closing], ignore_index=True).groupby(
        [id_col, "time"], as_index=False, sort=True,
    )[flags].sum()
    events[flags] = events.groupby(id_col, sort=False)[flags].cumsum()
    events["end"] = events.groupby(id_col, sort=False).time.shift(-1)
    blocked_length = (events.end - events.time).where(events.blocked.gt(0), 0).fillna(0)
    events["blocked_end"] = blocked_length.groupby(events[id_col], sort=False).cumsum()
    events["blocked_start"] = events.blocked_end - blocked_length
    active = events.loc[events.end.gt(events.time) & (
        events.seed.gt(0) | (events.measured.gt(0) & events.blocked.eq(0))
    )].copy()
    previous = active.groupby(id_col, sort=False)[["end", "blocked_end"]].shift()
    separated = (previous.end.isna() | active.time.sub(previous.end).gt(max_gap_hours)
                 | active.blocked_start.gt(previous.blocked_end + 1e-9))
    active["run"] = separated.cumsum()
    runs = active.groupby("run", sort=False).agg(
        **{id_col: (id_col, "first")}, start=("time", "first"), end=("end", "last"), seed=("seed", "max"),
    )
    runs = runs.loc[runs.seed.gt(0)]
    starts = np.floor(runs.start.to_numpy(dtype=float))
    counts = (np.ceil(runs.end.to_numpy(dtype=float) - 1e-9) - starts).astype(int)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    out = pd.DataFrame({id_col: np.repeat(runs[id_col].to_numpy(), counts),
                        time_col: np.repeat(starts, counts) + offsets})
    out[id_col] = out[id_col].astype(data[id_col].dtype)
    return out.drop_duplicates().sort_values([id_col, time_col]).reset_index(drop=True)


def advanced_airway_hours(frame: pd.DataFrame, *, concept_name: str, value_col: str, time_col: str) -> pd.DataFrame:
    """Reliable airway support intervals, preserving unknown as missing."""
    from ..utils.callback_utils import _expand_hirid_vent_to_hourly, hirid_vent

    windows = hirid_vent(frame, concept_name=concept_name, val_col=value_col,
                         index_col=time_col, expand_to_hourly=False)
    value_col = value_col if value_col in windows else concept_name
    supported = pd.to_numeric(windows[value_col], errors="coerce").isin([1, 2, 3, 4, 5])
    # Compute boundaries with every state, then expand supported intervals
    # only. An earlier unknown in the same hour must not mask real support
    # when the expansion deduplicates that hour. Absence stays missing.
    windows = windows.loc[supported].copy()
    windows[concept_name] = True
    return _expand_hirid_vent_to_hourly(
        windows, id_col="patientid", index_col=time_col, dur_col="dur_var",
        value_col=concept_name, concept_name=concept_name,
    )
