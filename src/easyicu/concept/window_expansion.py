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
    """Expand WinTbl rows with the resolver's two historical end semantics."""

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
    start = np.where(np.isnan(start), 0.0, start)
    duration = np.where(np.isnan(duration), 0.0, duration)
    start_aligned = np.floor(start / interval_hours) * interval_hours

    if end_mode == "raw":
        end_effective = start + duration
        point_counts = (
            np.floor((end_effective - start_aligned) / interval_hours).astype(np.int64)
            + 1
        )
        if duration_zero_single:
            point_counts = np.where(duration <= 0.0, 1, point_counts)
    elif end_mode == "floored_clamped":
        end_effective = (
            np.floor((start_aligned + duration) / interval_hours) * interval_hours
        )
        end_effective = np.maximum(end_effective, 0.0)
        point_counts = (
            np.floor((end_effective - start_aligned) / interval_hours).astype(np.int64)
            + 1
        )
    else:
        raise ValueError(f"Unknown end_mode: {end_mode!r}")

    point_counts = np.maximum(point_counts, 1)
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
