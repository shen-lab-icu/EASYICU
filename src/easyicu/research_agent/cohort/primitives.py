"""Case-neutral dataframe primitives shared by cohort producers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

ID_COL = "stay_id"
TIME_COL = "charttime"


def window(df: pd.DataFrame, start_hour: float, end_hour: float) -> pd.DataFrame:
    """Return rows inside the inclusive chart-time window."""
    if TIME_COL not in df.columns:
        return df.copy()
    out = df.copy()
    out[TIME_COL] = pd.to_numeric(out[TIME_COL], errors="coerce")
    return out[(out[TIME_COL] >= start_hour) & (out[TIME_COL] <= end_hour)].copy()


def first_nonnull(series: pd.Series):
    """Return the first non-null value, or ``pd.NA``."""
    nonnull = series.dropna()
    return nonnull.iloc[0] if len(nonnull) else pd.NA


def merge_left(base: pd.DataFrame, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Left-join non-empty stay-level frames onto ``base``."""
    out = base.copy()
    for frame in frames:
        if frame is None or frame.empty or ID_COL not in frame.columns:
            continue
        out = out.merge(frame, on=ID_COL, how="left")
    return out


__all__ = ["ID_COL", "TIME_COL", "first_nonnull", "merge_left", "window"]
