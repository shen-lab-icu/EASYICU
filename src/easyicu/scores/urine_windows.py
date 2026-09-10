"""Assessment evidence for urine-duration criteria, separate from descriptive UO.

Event inputs use estimated requested-interval bins ending at their labelled time. This
is an explicit discretized assessment convention, not proof of uncharted
continuous observation. Missing bins get no coverage. A partially intersected
volume bin cannot be apportioned without a distribution assumption, so its
window rate stays unavailable. HiRID inputs instead carry a rate over the
preceding chart interval, with no interval invented before the first record.

The >6 h assessment uses the shortest complete segment suffix ending now that
is strictly longer than 6 h. Its volume, duration and rate all refer to that
same continuously covered suffix; it never infers duration from two averages.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

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
    ]
    if name == "uo_6h":
        columns += [
            "uo_6h_oliguria_gt6h",
            "uo_6h_oliguria_duration_h",
            "uo_6h_oliguria_rate",
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
    for window in WINDOWS:
        name = f"uo_{window}h"
        out[f"{name}_covered_h"] = 0.0
        out[f"{name}_assessment_rate"] = np.nan
        out[f"{name}_assessment_reason"] = "insufficient_window"
    out["uo_6h_oliguria_gt6h"] = False
    out["uo_6h_oliguria_duration_h"] = np.nan
    out["uo_6h_oliguria_rate"] = np.nan

    for positions in data.groupby(ids, sort=False, observed=True).indices.values():
        pos = np.asarray(positions)
        ends = times[pos]
        starts = np.r_[ends[0], ends[:-1]] if source_is_rate else ends - width
        lengths = ends - starts
        amounts = data.urine.to_numpy(dtype=float)[pos]
        weights = data.weight.to_numpy(dtype=float)[pos]
        valid = np.isfinite(amounts) & np.isfinite(ends) & (lengths > 0)
        if not source_is_rate and len(ends) > 1:
            # Overlapping nonidentical volume bins have no unambiguous volume
            # allocation. Do not round them into manufactured grid continuity.
            valid[1:] &= starts[1:] >= ends[:-1] - 1e-9
        rates = amounts if source_is_rate else amounts / width
        observed_lengths = np.where(valid, lengths, 0.0)
        volumes = np.where(valid, rates * lengths, 0.0)
        cumulative_h = np.r_[0.0, np.cumsum(observed_lengths)]
        cumulative_v = np.r_[0.0, np.cumsum(volumes)]

        for window in WINDOWS:
            left_edges = ends - window
            left = np.searchsorted(ends, left_edges, side="right")
            left = np.minimum(left, np.arange(len(ends)))
            partial = np.maximum(0.0, left_edges - starts[left])
            partial = np.minimum(partial, lengths[left])
            partial = np.where(valid[left], partial, 0.0)
            covered = cumulative_h[1:] - cumulative_h[left] - partial
            volume = (
                cumulative_v[1:]
                - cumulative_v[left]
                - partial * np.where(valid[left], rates[left], 0.0)
            )
            full = covered >= window - 1e-9
            whole_volume_bins = source_is_rate | (partial < 1e-9)
            weight_ok = np.isfinite(weights) & (weights > 0)
            assessable = full & whole_volume_bins & weight_ok
            name = f"uo_{window}h"
            out.loc[pos, f"{name}_covered_h"] = np.maximum(0.0, covered)
            out.loc[pos[assessable], f"{name}_assessment_rate"] = (
                volume[assessable] / window / weights[assessable]
            )
            reason = np.full(len(pos), "insufficient_window", dtype=object)
            reason[full & ~whole_volume_bins] = "partial_volume_bin"
            reason[~weight_ok] = "missing_weight"
            reason[assessable] = (
                "complete_rate_intervals"
                if source_is_rate
                else "complete_estimated_bins"
            )
            out.loc[pos, f"{name}_assessment_reason"] = reason

        # Find the closest complete segment start strictly before t-6. Reset
        # continuity at every invalid segment or genuine gap, per patient.
        candidate = np.searchsorted(starts, ends - 6.0 - 1e-9, side="left") - 1
        safe = np.maximum(candidate, 0)
        break_before = ~valid
        if len(pos) > 1:
            break_before[1:] |= (~valid[:-1]) | (starts[1:] > ends[:-1] + 1e-9)
        block_start = np.maximum.accumulate(
            np.where(break_before, np.arange(len(pos)), 0)
        )
        duration = ends - starts[safe]
        covered = cumulative_h[1:] - cumulative_h[safe]
        volume = cumulative_v[1:] - cumulative_v[safe]
        assessable = (candidate >= block_start) & valid & (duration > 6.0 + 1e-9)
        assessable &= covered >= duration - 1e-9
        assessable &= np.isfinite(weights) & (weights > 0)
        rate = np.full(len(pos), np.nan)
        rate[assessable] = (
            volume[assessable] / duration[assessable] / weights[assessable]
        )
        out.loc[pos, "uo_6h_oliguria_duration_h"] = np.where(
            assessable, duration, np.nan
        )
        out.loc[pos, "uo_6h_oliguria_rate"] = rate
        out.loc[pos, "uo_6h_oliguria_gt6h"] = assessable & (rate < 0.3)
    return out
