"""Vectorized expansion of WinTbl duration rows onto a regular time grid."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ..utils.thresholds import ge, le

WinTblEndMode = Literal["raw", "floored_clamped"]

#: Window concepts published on the hourly grid exactly as ricu ``expand(x,
#: aggregate = "any")`` builds them inside its ``gcs`` callback: the window's
#: start cell through ``re_time(start + dur, interval)`` inclusive, one row per
#: stay and cell, TRUE when any window covering the cell is TRUE. The GCS
#: callback and every published ``ett_gcs`` column read this one owner.
INCLUSIVE_ANY_WINDOW_CONCEPTS = frozenset({"ett_gcs"})

__all__ = [
    "INCLUSIVE_ANY_WINDOW_CONCEPTS",
    "WinTblEndMode",
    "expand_windows_inclusive_any",
    "expand_wintbl_vectorized",
    "publish_inclusive_any_window",
]


def clip_episode_windows(
    frame: pd.DataFrame, *, start: pd.Series, upper: pd.Series, lower: float,
    keep_post_discharge: bool = False,
) -> pd.DataFrame | None:
    """Select true windows by overlap, then clip endpoints before expansion.

    ``None`` delegates point-only frames to the ordinary point quarantine.
    Source duration declarations are respected and re-emitted in hours.
    """
    from ..table.duration import UNIT_HOURS, resolve_dur_var_hours, set_dur_var_unit

    # Raw point tables can carry an endtime too (for example vent_start).
    # Only a source-declared/constructed duration establishes window semantics.
    if "dur_var" not in frame:
        return None
    # The normalized duration is authoritative. Some callbacks have already
    # normalized the start while retaining an absolute datetime endtime.
    end = start + resolve_dur_var_hours(frame)
    window = end.notna()
    valid = np.isfinite(start) & np.isfinite(end) & end.gt(start) & end.gt(lower)
    if not keep_post_discharge:
        valid &= start.lt(upper)
    # Empty neighbouring sources can contribute an all-null endpoint column
    # during concat. Actual point rows retain the ordinary quarantine rule.
    # Point rows take the shared episode rule (inclusive, decimal helpers).
    points = ~window & np.isfinite(start) & ge(start, lower)
    if not keep_post_discharge:
        points &= le(start, upper)
    valid |= points
    out = frame.loc[valid].copy()
    clipped_start = start.loc[valid].clip(lower=lower)
    clipped_end = end.loc[valid]
    if not keep_post_discharge:
        clipped_end = clipped_end.clip(upper=upper.loc[valid])
    # The caller's start Series is the actual index column, including aliases.
    out[start.name] = clipped_start
    if "endtime" in out:
        out["endtime"] = clipped_end
    if "dur_var" in out:
        out["dur_var"] = clipped_end - clipped_start
        set_dur_var_unit(out, UNIT_HOURS)
    return out


def expand_wintbl_vectorized(
    data: pd.DataFrame,
    *,
    idx_col: str,
    dur_col: str,
    id_cols: List[str],
    value_columns: List[str],
    interval_hours: float,
    end_mode: WinTblEndMode = "raw",
    duration_zero_single: bool = False,
) -> pd.DataFrame:
    """Publish grid cells with positive overlap with ``[start, start + dur)``.

    The legacy mode/zero-duration arguments remain accepted for callers, but
    neither may extend a window or turn an empty interval into point evidence.
    """

    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    base_cols = [idx_col]
    base_cols += [
        column for column in id_cols if column in data and column not in base_cols
    ]
    base_cols += [
        column
        for column in value_columns
        if column in data and column != dur_col and column not in base_cols
    ]
    if data.empty:
        return pd.DataFrame(columns=base_cols)

    start = pd.to_numeric(data[idx_col], errors="coerce").to_numpy(dtype=np.float64)
    duration = pd.to_numeric(data[dur_col], errors="coerce").to_numpy(dtype=np.float64)
    if end_mode not in ("raw", "floored_clamped"):
        raise ValueError(f"Unknown end_mode: {end_mode!r}")
    end = start + duration
    valid = np.isfinite(start) & np.isfinite(end) & (duration > 0) & (end > start)
    start_aligned = np.floor(np.where(valid, start, 0) / interval_hours) * interval_hours
    point_counts = np.where(
        valid,
        np.ceil(np.where(valid, end, 0) / interval_hours)
        - np.floor(np.where(valid, start, 0) / interval_hours),
        0,
    ).astype(np.int64)
    total = int(point_counts.sum())
    if total == 0:
        return pd.DataFrame(columns=base_cols)

    row_indices = np.repeat(np.arange(len(start)), point_counts)
    offsets = np.arange(total) - np.repeat(
        np.concatenate(([0], np.cumsum(point_counts)[:-1])), point_counts
    )
    times = start_aligned[row_indices] + offsets.astype(np.float64) * interval_hours

    output: Dict[str, np.ndarray] = {idx_col: times}
    seen = {idx_col}
    for column in id_cols:
        if column in data and column not in seen:
            output[column] = data[column].to_numpy()[row_indices]
            seen.add(column)
    for column in value_columns:
        if column in seen or column == dur_col:
            continue
        if column in data:
            output[column] = data[column].to_numpy()[row_indices]
            seen.add(column)
    return pd.DataFrame(output)


def _clock_hours(series: pd.Series) -> tuple[np.ndarray, str]:
    """Return ``series`` as float hours plus the clock kind to restore later."""

    if pd.api.types.is_timedelta64_dtype(series):
        return (series.dt.total_seconds() / 3600.0).to_numpy(dtype=np.float64), "timedelta"
    if pd.api.types.is_datetime64_any_dtype(series):
        origin = pd.Timestamp("1970-01-01", tz=series.dt.tz)
        return ((series - origin) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float64), "datetime"
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64), "numeric"


def _restore_clock(hours: np.ndarray, kind: str, template: pd.Series) -> pd.Series:
    if kind == "timedelta":
        return pd.Series(pd.to_timedelta(hours, unit="h"))
    if kind == "datetime":
        origin = pd.Timestamp("1970-01-01", tz=template.dt.tz)
        return pd.Series(origin + pd.to_timedelta(hours, unit="h"))
    return pd.Series(hours, dtype=np.float64)


def expand_windows_inclusive_any(
    data: pd.DataFrame,
    *,
    idx_col: str,
    id_cols: List[str],
    value_col: str,
    dur_hours: Optional[pd.Series],
    interval_hours: float = 1.0,
) -> pd.DataFrame:
    """ricu ``expand(x, aggregate = "any")`` for a logical window table.

    ricu 0.6.3 (``expand``): ``end := re_time(start + dur, interval)`` where
    ``re_time`` is ``round_to`` = floor to the interval, rows with a missing
    start or end are dropped (``rm_na``), rows with ``start <= end`` become
    ``seq(start, end, step)`` (both ends included), and cells are aggregated
    with ``any`` so a TRUE window wins a cell it shares with a FALSE one. The
    start sits on the grid because ricu loads the index at the interval; it is
    floored here for the same reason. As in ricu (``x[end < 0, end := 0]``), a
    window that closes before admission on an admission-relative clock is
    extended to hour zero.

    ``dur_hours`` is ``None`` for a point table (no duration column; SICdb and
    AUMC publish hourly airway points): ricu expands only a ``win_tbl``, so
    each point stays in its own cell, merged with ``any``, never clamped or
    extended (delta4 D4G-POINT). A window whose declared duration is missing
    is dropped, as ``rm_na`` drops it; it is not a zero-length exposure
    (D4G-MISSING-DURATION). An explicit zero duration is a one-cell window.
    """

    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    columns = [*id_cols, idx_col, value_col]
    if data.empty:
        empty = pd.DataFrame({column: data[column].iloc[0:0] for column in columns[:-1]})
        empty[value_col] = pd.Series([], dtype=bool)
        return empty

    raw_start, kind = _clock_hours(data[idx_col])
    start = np.floor(raw_start / interval_hours) * interval_hours
    if dur_hours is None:
        end = start
    else:
        # A missing duration stays NaN, so the finite mask below drops it.
        duration = pd.to_numeric(dur_hours, errors="coerce").to_numpy(dtype=np.float64)
        end = np.floor((start + duration) / interval_hours) * interval_hours
        if kind != "datetime":
            end = np.where(end < 0, 0.0, end)
    valid = np.isfinite(start) & np.isfinite(end) & (end >= start)
    counts = np.where(valid, np.rint((np.where(valid, end, 0) - np.where(valid, start, 0)) / interval_hours) + 1, 0).astype(np.int64)
    total = int(counts.sum())
    values = data[value_col].astype("boolean").fillna(False).to_numpy(dtype=bool)
    if total == 0:
        empty = {column: data[column].iloc[0:0] for column in id_cols}
        empty[idx_col] = _restore_clock(np.empty(0), kind, data[idx_col])
        empty[value_col] = pd.Series([], dtype=bool)
        return pd.DataFrame(empty)[columns]

    rows = np.repeat(np.arange(len(start)), counts)
    offsets = np.arange(total) - np.repeat(np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    times = start[rows] + offsets.astype(np.float64) * interval_hours
    out = {column: data[column].to_numpy()[rows] for column in id_cols}
    out[idx_col] = times
    out[value_col] = values[rows]
    expanded = pd.DataFrame(out)
    expanded = (
        expanded.groupby([*id_cols, idx_col], as_index=False, sort=True, dropna=False)[value_col]
        .any()
    )
    expanded[value_col] = expanded[value_col].astype(bool)
    expanded[idx_col] = _restore_clock(expanded[idx_col].to_numpy(dtype=np.float64), kind, data[idx_col]).to_numpy()
    return expanded[columns].reset_index(drop=True)


def _window_parts(table) -> tuple[pd.DataFrame, List[str], Optional[str], Optional[str]]:
    frame = getattr(table, "data", table)
    id_cols = list(getattr(table, "id_vars", None) or getattr(table, "id_columns", None) or [])
    index = getattr(table, "index_var", None) or getattr(table, "index_column", None)
    dur = getattr(table, "dur_var", None)
    if dur is None and "dur_var" in frame.columns:
        dur = "dur_var"
    return frame, id_cols, index, dur


def window_duration_hours(frame: pd.DataFrame, dur_col: Optional[str], *, concept: str) -> pd.Series:
    """Read a window table's duration as hours from its declared unit."""

    from ..table.duration import get_dur_var_unit, resolve_dur_var_hours, set_dur_var_unit

    if not dur_col or dur_col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    probe = frame[[dur_col]].rename(columns={dur_col: "dur_var"})
    unit = get_dur_var_unit(frame)
    if unit:
        set_dur_var_unit(probe, unit)
    return resolve_dur_var_hours(probe, concept=concept)


def publish_inclusive_any_window(table, concept_name: str, interval_hours: float = 1.0):
    """Return ``table`` as the hourly ``ett_gcs`` column every output publishes.

    The window table becomes a time-series table (ids, index, value) built by
    :func:`expand_windows_inclusive_any`; it carries no duration afterwards.
    A table without a duration column is a point table: its rows keep their
    own cells.
    """

    from ..table import ICUTable

    frame, id_cols, index, dur = _window_parts(table)
    if index is None or index not in frame.columns or concept_name not in frame.columns:
        return table
    id_cols = [column for column in id_cols if column in frame.columns]
    expanded = expand_windows_inclusive_any(
        frame,
        idx_col=index,
        id_cols=id_cols,
        value_col=concept_name,
        dur_hours=(
            window_duration_hours(frame, dur, concept=concept_name)
            if dur and dur in frame.columns else None
        ),
        interval_hours=interval_hours,
    )
    return ICUTable(
        data=expanded,
        id_columns=id_cols,
        index_column=index,
        value_column=concept_name,
        time_basis=getattr(table, "time_basis", None),
        interval=getattr(table, "interval", None),
    )
