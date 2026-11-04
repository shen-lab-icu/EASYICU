"""Vectorized expansion of WinTbl duration rows onto a regular time grid."""

from __future__ import annotations

from typing import Dict, List, Literal

import numpy as np
import pandas as pd

WinTblEndMode = Literal["raw", "floored_clamped"]

__all__ = ["WinTblEndMode", "expand_wintbl_vectorized"]


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
