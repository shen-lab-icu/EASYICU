"""Assessment evidence for urine-duration criteria, separate from descriptive UO.

Recorded volumes are back-allocated over their collection interval using the
same overlap and twelve-hour gap kernel as descriptive UO and KDIGO. Coverage
is a collection-interval convention, not proof of continuous observation.

The >6 h assessment uses the shortest complete segment suffix ending now that
is strictly longer than 6 h. Its volume, duration and rate all refer to that
same continuously covered suffix; it never infers duration from two averages.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from easyicu.io.ts_utils import _grouped_row_batches

from .urine_output import collection_intervals, collection_windows, window_availability

from easyicu.urine_weight_linkage import (
    resolve_keyed_unique_weights,
    resolve_unkeyed_single_entity_weight,
)

WINDOWS = (6, 12, 24)


def urine_evidence_columns(name: str) -> list[str]:
    columns = [
        f"{name}_covered_h",
        f"{name}_assessment_rate",
        f"{name}_assessment_reason",
        f"{name}_assessment_available_at",
    ]
    if name == "uo_6h":
        columns += [
            "uo_6h_oliguria_gt6h",
            "uo_6h_oliguria_duration_h",
            "uo_6h_oliguria_rate",
            "uo_6h_oliguria_available_at",
        ]
    return columns


def _hours(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_timedelta64_dtype(values):
        return (values / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    if pd.api.types.is_datetime64_any_dtype(values):
        # An arbitrary common origin preserves all elapsed intervals and NaT.
        return ((values - values.min()) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def assess_urine_windows(
    urine: pd.DataFrame,
    weight: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    time_column: str,
    interval: pd.Timedelta,
    source_is_rate: bool = False,
) -> pd.DataFrame:
    """Return keyed window values, coverage and the shared >6 h RRT evidence.

    Input numeric times are relative hours (the concept-layer contract). Same
    timestamp volume channels are summed, rate channels averaged, and counted
    once for coverage. Invalid records remain explicit holes in the timeline.
    """
    ids = list(id_columns)
    width = pd.Timedelta(interval).total_seconds() / 3600
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Urine assessment requires a positive interval")
    if not ids or not set(ids + [time_column, "urine"]).issubset(urine.columns):
        raise ValueError(
            "Urine assessment requires explicit patient/time/value columns"
        )
    data = urine[ids + [time_column, "urine"]].copy()
    data["urine"] = pd.to_numeric(data["urine"], errors="coerce")
    data.loc[~np.isfinite(data.urine) | (data.urine < 0), "urine"] = np.nan
    data = data.dropna(subset=ids + [time_column])
    # A valid zero is retained. An explicitly invalid channel cannot be silently
    # removed and interpreted as a completely observed volume bin.
    groups = data.groupby(ids + [time_column], sort=False, dropna=False)["urine"]
    values = groups.agg(["sum", "mean", "count", "size"]).reset_index()
    values["urine"] = values["mean" if source_is_rate else "sum"].where(
        values["count"] == values["size"]
    )
    data = (
        values[ids + [time_column, "urine"]]
        .sort_values(ids + [time_column])
        .reset_index(drop=True)
    )
    if '_urine_available_at' in urine:
        receipt_cols = [c for c in ('_urine_available_at', '_urine_gap_h') if c in urine]
        receipts = urine.groupby(ids + [time_column], as_index=False)[receipt_cols].max()
        data = data.merge(receipts, on=ids + [time_column], how='left', validate='one_to_one')
    common_ids = [col for col in ids if col in weight.columns]
    if common_ids and "weight" in weight.columns:
        keyed = resolve_keyed_unique_weights(weight, id_columns=common_ids)
        data = data.merge(keyed, on=common_ids, how="left", validate="many_to_one")
    else:
        resolution = resolve_unkeyed_single_entity_weight(
            data, weight, urine_id_columns=ids
        )
        data["weight"] = resolution.weight if resolution.weight is not None else np.nan
    times = _hours(data[time_column])
    out = data[ids + [time_column]].copy()
    row_count = len(data)
    covered_by_window: dict[int, np.ndarray] = {}
    rate_by_window: dict[int, np.ndarray] = {}
    reason_by_window: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        covered_by_window[window] = np.zeros(row_count, dtype=float)
        rate_by_window[window] = np.full(row_count, np.nan, dtype=float)
        # Scalar assignment shares the immutable Python string. np.full with
        # a Unicode scalar converts/allocates a Python string for every cell.
        reason_by_window[window] = np.empty(row_count, dtype=object)
        reason_by_window[window].fill("insufficient_window")
    available_by_window = {w: np.full(row_count, np.nan) for w in WINDOWS}
    oliguria_available = np.full(row_count, np.nan)
    oliguria_gt6h = np.zeros(row_count, dtype=bool)
    oliguria_duration_h = np.full(row_count, np.nan, dtype=float)
    oliguria_rate = np.full(row_count, np.nan, dtype=float)

    amounts_all = data.urine.to_numpy(dtype=float)
    weights_all = data.weight.to_numpy(dtype=float)
    source_times_all = (
        data['_urine_available_at'].to_numpy(dtype=float)
        if '_urine_available_at' in data else times
    )
    record_gaps_all = data['_urine_gap_h'].to_numpy(dtype=float) if '_urine_gap_h' in data else None
    group_codes = data.groupby(ids, sort=False, observed=True).ngroup().to_numpy()
    for row_batch, edges in _grouped_row_batches(group_codes):
        for begin, end in zip(edges[:-1], edges[1:]):
            batch = slice(row_batch.start + begin, row_batch.start + end)
            ends = times[batch]
            amounts = amounts_all[batch] / width if source_is_rate else amounts_all[batch]
            weights = weights_all[batch]
            source_times = source_times_all[batch]
            kwargs = dict(source_is_rate=source_is_rate, first_hours=width,
                          source_times=source_times,
                          record_gaps=record_gaps_all[batch] if record_gaps_all is not None else None)
            integrated = collection_windows(ends, amounts, WINDOWS, **kwargs)
            starts, lengths, rates, valid, _ = collection_intervals(ends, amounts, **kwargs)
            observed_lengths = np.where(valid, lengths, 0.)
            volumes = rates * observed_lengths
            cumulative_h = np.r_[0., np.cumsum(observed_lengths)]
            cumulative_v = np.r_[0., np.cumsum(volumes)]
            offsets = np.arange(len(ends))
            weight_ok = np.isfinite(weights) & (weights > 0)
            for window in WINDOWS:
                volume, covered, available = integrated[window]
                assessable = (covered >= window - 1e-9) & weight_ok
                covered_by_window[window][batch] = covered
                rate_by_window[window][batch] = np.divide(
                    volume, window * weights, out=np.full(len(ends), np.nan), where=assessable,
                )
                available_by_window[window][batch] = available
                reason = np.empty(len(ends), dtype=object)
                reason.fill("insufficient_window")
                reason[~weight_ok] = "missing_weight"
                reason[assessable] = "complete_rate_intervals" if source_is_rate else "complete_collection_intervals"
                reason_by_window[window][batch] = reason

            # Find the closest complete segment start strictly before t-6. Reset
            # continuity at every invalid segment or genuine gap, per patient.
            candidate = np.searchsorted(starts, ends - 6.0 - 1e-9, side="left") - 1
            safe = np.maximum(candidate, 0)
            break_before = ~valid
            if len(ends) > 1:
                break_before[1:] |= (~valid[:-1]) | (starts[1:] > ends[:-1] + 1e-9)
            break_before[0] = True
            block_start = np.maximum.accumulate(
                np.where(break_before, offsets, 0),
            )
            duration = ends - starts[safe]
            covered = cumulative_h[1:] - cumulative_h[safe]
            volume = cumulative_v[1:] - cumulative_v[safe]
            assessable = (candidate >= block_start) & valid & (duration > 6.0 + 1e-9)
            assessable &= covered >= duration - 1e-9
            assessable &= np.isfinite(weights) & (weights > 0)
            rate = np.full(len(ends), np.nan)
            rate[assessable] = (
                volume[assessable] / duration[assessable] / weights[assessable]
            )
            oliguria_available[batch] = np.where(assessable, window_availability(source_times, safe, valid), np.nan)
            oliguria_duration_h[batch] = np.where(assessable, duration, np.nan)
            oliguria_rate[batch] = rate
            oliguria_gt6h[batch] = assessable & (rate < 0.3)

    # Assign each result column once. Repeated ``DataFrame.loc`` writes per
    # patient make pandas rebuild indexers hundreds of thousands of times on a
    # full ICU cohort. The NumPy buffers preserve the same row order/dtypes.
    for window in WINDOWS:
        name = f"uo_{window}h"
        out[f"{name}_covered_h"] = covered_by_window[window]
        out[f"{name}_assessment_rate"] = rate_by_window[window]
        out[f"{name}_assessment_reason"] = reason_by_window[window]
        out[f"{name}_assessment_available_at"] = available_by_window[window]
    out["uo_6h_oliguria_gt6h"] = oliguria_gt6h
    out["uo_6h_oliguria_duration_h"] = oliguria_duration_h
    out["uo_6h_oliguria_rate"] = oliguria_rate
    out["uo_6h_oliguria_available_at"] = oliguria_available
    return out
