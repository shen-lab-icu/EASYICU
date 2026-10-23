"""Frozen pre-optimization urine kernels, for synthetic parity checks only."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from easyicu.urine_weight_linkage import (
    resolve_keyed_unique_weights, resolve_unkeyed_single_entity_weight,
)
from easyicu.callbacks import _urine_rate_window_avg_multi
from easyicu.scores.aki_profiles import _time_hours, AKIProfileError
from easyicu.scores.kdigo_aki import detect_id_col, detect_time_col, detect_value_col
from typing import Optional
from easyicu.io.ts_utils import _infer_numeric_time_unit

WINDOWS = (6, 12, 24)


def _concat_component_frames(frames, key_cols):
    """The original callback merge's deduplication and MultiIndex concat."""
    indexed = []
    for frame in frames:
        if frame.duplicated(subset=key_cols).any():
            frame = frame.drop_duplicates(subset=key_cols, keep="last")
        indexed.append(frame.set_index(key_cols, drop=True))
    return pd.concat(indexed, axis=1, join="outer", sort=False, copy=False).reset_index()

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
    row_count = len(data)
    covered_by_window: dict[int, np.ndarray] = {}
    rate_by_window: dict[int, np.ndarray] = {}
    reason_by_window: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        covered_by_window[window] = np.zeros(row_count, dtype=float)
        rate_by_window[window] = np.full(row_count, np.nan, dtype=float)
        reason_by_window[window] = np.full(
            row_count, "insufficient_window", dtype=object
        )
    oliguria_gt6h = np.zeros(row_count, dtype=bool)
    oliguria_duration_h = np.full(row_count, np.nan, dtype=float)
    oliguria_rate = np.full(row_count, np.nan, dtype=float)

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
            covered_by_window[window][pos] = np.maximum(0.0, covered)
            rate_by_window[window][pos[assessable]] = (
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
            reason_by_window[window][pos] = reason

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
        oliguria_duration_h[pos] = np.where(
            assessable, duration, np.nan
        )
        oliguria_rate[pos] = rate
        oliguria_gt6h[pos] = assessable & (rate < 0.3)

    # Assign each result column once. Repeated ``DataFrame.loc`` writes per
    # patient make pandas rebuild indexers hundreds of thousands of times on a
    # full ICU cohort even though ``pos`` is already a positional integer
    # array. The NumPy buffers above preserve the same row order and dtypes.
    for window in WINDOWS:
        name = f"uo_{window}h"
        out[f"{name}_covered_h"] = covered_by_window[window]
        out[f"{name}_assessment_rate"] = rate_by_window[window]
        out[f"{name}_assessment_reason"] = reason_by_window[window]
    out["uo_6h_oliguria_gt6h"] = oliguria_gt6h
    out["uo_6h_oliguria_duration_h"] = oliguria_duration_h
    out["uo_6h_oliguria_rate"] = oliguria_rate
    return out


def _urine_window_avg(
    urine: pd.DataFrame,
    weight: pd.DataFrame,
    window_hours: int,
    min_hours: int,
    interval: pd.Timedelta = None,
    source_is_rate: bool = False,
) -> pd.DataFrame:
    """Internal function to calculate windowed average urine output.

    Args:
        urine: DataFrame with urine output (mL)
        weight: DataFrame with patient weight (kg)
        window_hours: Window size in hours
        min_hours: Minimum hours of data required in window
        interval: Time interval for data

    Returns:
        DataFrame with averaged urine output (mL/kg/h)
    """
    if source_is_rate:
        return _urine_rate_window_avg_multi(
            urine,
            weight,
            windows=[(window_hours, min_hours)],
            interval=interval,
        )[f"uo_{window_hours}h"]

    # Determine ID and time columns
    # 🔧 FIX 2025-01-31: Include CaseID (uppercase) for SICdb support
    id_cols = [col for col in urine.columns if col.endswith('_id') or col in ['stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']]

    # 不同数据库使用不同的时间列名
    if 'charttime' in urine.columns:
        time_col = 'charttime'
    elif 'measuredat' in urine.columns:
        time_col = 'measuredat'
    elif 'measuredat_minutes' in urine.columns:
        time_col = 'measuredat_minutes'  # DuckDB aggregation output
    elif 'nursingchartoffset' in urine.columns:
        time_col = 'nursingchartoffset'
    elif 'observationoffset' in urine.columns:
        time_col = 'observationoffset'
    elif 'intakeoutputoffset' in urine.columns:
        time_col = 'intakeoutputoffset'  # eICU urine
    elif 'registeredat' in urine.columns:
        time_col = 'registeredat'  # AUMC
    else:
        # 尝试找到任何时间相关的列
        time_like_cols = [col for col in urine.columns if 'time' in col.lower() or 'offset' in col.lower() or 'minute' in col.lower()]
        if time_like_cols:
            time_col = time_like_cols[0]
        else:
            raise ValueError(f"Cannot find time column in urine data. Available columns: {urine.columns.tolist()}")

    val_col = 'urine'
    result_col = f'uo_{window_hours}h'

    # 如果没有找到ID列，尝试常见的ID列名
    if not id_cols:
        for potential_id in ['admissionid', 'stay_id', 'patientunitstayid', 'patientid', 'icustay_id', 'CaseID']:
            if potential_id in urine.columns:
                id_cols = [potential_id]
                break

    # Merge urine and weight data
    if id_cols:
        # 检查weight表是否有相同的ID列
        # 对于AUMC，urine有admissionid但weight只有patientid
        # 需要先join admissions表获取admissionid->patientid映射
        common_ids = [col for col in id_cols if col in weight.columns]
        if common_ids:
            merged = pd.merge(urine, weight, on=common_ids, how='left', suffixes=('', '_weight'))
        else:
            merged = urine.copy()
            from easyicu.urine_weight_linkage import resolve_unkeyed_single_entity_weight

            resolution = resolve_unkeyed_single_entity_weight(
                urine,
                weight,
                urine_id_columns=id_cols,
                weight_column="weight",
            )
            merged['weight'] = (
                resolution.weight if resolution.weight is not None else np.nan
            )
    else:
        merged = urine.copy()
        merged['weight'] = np.nan

    # Handle weight time column
    if 'charttime_weight' in merged.columns:
        # Use forward fill to propagate weight values
        merged = merged.sort_values(id_cols + [time_col])
        for id_col in id_cols:
            merged['weight'] = merged.groupby(id_col)['weight'].ffill()

    # Remove rows without weight
    merged = merged[merged['weight'].notna() & (merged['weight'] > 0)]

    if len(merged) == 0:
        # Return empty DataFrame with correct structure
        result = urine[id_cols + [time_col]].copy()
        result[result_col] = np.nan
        return result

    # Sort by ID and time
    merged = merged.sort_values(id_cols + [time_col])

    # Infer interval if not provided
    if interval is None:
        # Try to infer from data
        if len(merged) > 1:
            time_diffs = merged.groupby(id_cols)[time_col].diff()
            valid_diffs = time_diffs[time_diffs.notna() & (time_diffs > pd.Timedelta(0))]
            if len(valid_diffs) > 0:
                interval = valid_diffs.median()
            else:
                interval = pd.Timedelta(hours=1)
        else:
            interval = pd.Timedelta(hours=1)

    # Calculate window parameters (use lowercase 'h' to avoid deprecation warning)
    min_periods = max(1, int(min_hours / (interval.total_seconds() / 3600)))

    # Vectorized rolling — avoids per-group Python function call overhead
    # Data is already sorted by id_cols + [time_col], so sort=False is safe
    interval_hours = interval.total_seconds() / 3600
    n_measurements = max(1, int(window_hours / interval_hours))

    urine_sum = merged.groupby(id_cols, sort=False)[val_col].rolling(
        window=n_measurements,
        min_periods=max(1, min_periods),
        closed='right'
    ).sum()

    # .values preserves row order (sort=False + pre-sorted data)
    merged[result_col] = urine_sum.values / (merged['weight'].values * window_hours)

    # Keep only relevant columns
    keep_cols = id_cols + [time_col, result_col]
    result = merged[[col for col in keep_cols if col in merged.columns]]

    return result


def _urine_window_avg_multi(
    urine,
    weight,
    interval=None,
    source_is_rate: bool = False,
):
    """Compute uo_6h, uo_12h, uo_24h in a single merge+sort+groupby pass.

    Returns dict: {'uo_6h': DataFrame, 'uo_12h': DataFrame, 'uo_24h': DataFrame}
    """
    windows = [(6, 3), (12, 6), (24, 12)]  # (window_hours, min_hours)

    if source_is_rate:
        return _urine_rate_window_avg_multi(
            urine,
            weight,
            windows=windows,
            interval=interval,
        )

    # Reuse _urine_window_avg's column detection logic for the first call
    # Then share the merged+sorted data for subsequent windows
    id_cols = [col for col in urine.columns if col.endswith('_id') or col in [
        'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']]

    time_col = None
    for candidate in ['charttime', 'measuredat', 'measuredat_minutes', 'nursingchartoffset',
                       'observationoffset', 'intakeoutputoffset', 'registeredat']:
        if candidate in urine.columns:
            time_col = candidate
            break
    if time_col is None:
        time_like = [c for c in urine.columns if 'time' in c.lower() or 'offset' in c.lower() or 'minute' in c.lower()]
        if time_like:
            time_col = time_like[0]
        else:
            # Fallback to individual calls
            return {
                'uo_6h': _urine_window_avg(urine, weight, 6, 3, interval),
                'uo_12h': _urine_window_avg(urine, weight, 12, 6, interval),
                'uo_24h': _urine_window_avg(urine, weight, 24, 12, interval),
            }

    if not id_cols:
        for pid in ['admissionid', 'stay_id', 'patientunitstayid', 'patientid', 'icustay_id', 'CaseID']:
            if pid in urine.columns:
                id_cols = [pid]
                break

    # Single merge of urine + weight
    common_ids = [c for c in id_cols if c in weight.columns]
    if common_ids:
        merged = pd.merge(urine, weight, on=common_ids, how='left', suffixes=('', '_weight'))
    else:
        merged = urine.copy()
        from easyicu.urine_weight_linkage import resolve_unkeyed_single_entity_weight

        resolution = resolve_unkeyed_single_entity_weight(
            urine,
            weight,
            urine_id_columns=id_cols,
            weight_column="weight",
        )
        merged['weight'] = (
            resolution.weight if resolution.weight is not None else np.nan
        )

    if 'charttime_weight' in merged.columns:
        merged = merged.sort_values(id_cols + [time_col])
        for ic in id_cols:
            merged['weight'] = merged.groupby(ic)['weight'].ffill()

    merged = merged[merged['weight'].notna() & (merged['weight'] > 0)]

    if len(merged) == 0:
        empty = urine[id_cols + [time_col]].copy()
        results = {}
        for wh, _ in windows:
            r = empty.copy()
            r[f'uo_{wh}h'] = np.nan
            results[f'uo_{wh}h'] = r
        return results

    merged = merged.sort_values(id_cols + [time_col])

    if interval is None:
        if len(merged) > 1:
            td = merged.groupby(id_cols)[time_col].diff()
            valid = td[td.notna() & (td > pd.Timedelta(0))]
            interval = valid.median() if len(valid) > 0 else pd.Timedelta(hours=1)
        else:
            interval = pd.Timedelta(hours=1)

    val_col = 'urine'
    interval_hours = interval.total_seconds() / 3600

    # Vectorized rolling for all 3 windows — no per-group function calls
    for wh, mh in windows:
        n_meas = max(1, int(wh / interval_hours))
        min_p = max(1, int(mh / interval_hours))
        urine_sum = merged.groupby(id_cols, sort=False)[val_col].rolling(
            window=n_meas, min_periods=min_p, closed='right'
        ).sum()
        merged[f'uo_{wh}h'] = urine_sum.values / (merged['weight'].values * wh)

    results = {}
    for wh, _ in windows:
        col = f'uo_{wh}h'
        keep = id_cols + [time_col, col]
        results[col] = merged[[c for c in keep if c in merged.columns]].copy()

    return results


def _urine24_batch(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str,
    time_step: float,
    window_steps: int,
    min_steps: int,
    step_factor: float,
) -> pd.DataFrame:
    """Batch-vectorized urine24 computation for all patients at once.

    🚀 Performance: Replaces per-patient loop (4796 calls to process_patient = 13.3s)
    with batch operations:
    - 1 groupby().agg() for patient stats
    - 1 merge for urine values onto grid
    - 1 groupby().rolling() for rolling sum

    Expected speedup: 13.3s → ~2-3s (5-6x faster)

    Args:
        df: Input DataFrame with urine data
        id_col: Patient ID column name
        time_col: Time column name (numeric)
        urine_col: Urine value column name
        time_step: Time step size (in time column units)
        window_steps: Rolling window size in steps (typically 24)
        min_steps: Minimum window steps for non-NA output (typically 12)
        step_factor: Scaling factor (typically 24.0)
    """
    # Step 1: Compute per-patient stats
    stats = df.groupby(id_col)[time_col].agg(['min', 'max']).reset_index()
    stats['ricu_end'] = stats['max'] - stats['min']  # ricu buggy behavior: duration as end

    # Step 2: Generate time grids for all patients (vectorized with numpy)
    # Each patient needs: arange(start, ricu_end + step, step)
    grid_sizes = np.maximum(1, np.floor(
        (stats['ricu_end'].values - stats['min'].values) / time_step + 1
    ).astype(int))

    # Handle edge case: ricu_end < start (duration = 0)
    grid_sizes = np.where(stats['ricu_end'].values >= stats['min'].values, grid_sizes, 1)

    total_points = grid_sizes.sum()

    # Pre-allocate arrays for speed
    all_times = np.empty(total_points, dtype=np.float64)
    all_ids = np.empty(total_points, dtype=df[id_col].dtype)

    # 2026-05-20 perf fix: previous code called `stats.iloc[i]['col']`
    # three times per iteration of the per-patient loop. Each iloc call
    # is O(N) in pandas and the loop ran once per patient — on a
    # 60k-patient MIMIC-III run that was the dominant cost in urine24.
    # Pull the arrays out of pandas once and index with plain numpy.
    starts_arr = stats['min'].to_numpy()
    ends_arr = stats['ricu_end'].to_numpy()
    id_arr = stats[id_col].to_numpy()

    offset = 0
    for i in range(len(stats)):
        n = grid_sizes[i]
        start = starts_arr[i]
        end = ends_arr[i]
        if end >= start:
            times = np.arange(start, end + time_step * 0.5, time_step)[:n]
        else:
            times = np.array([start])
            n = 1
        actual_n = len(times)
        all_times[offset:offset + actual_n] = times
        all_ids[offset:offset + actual_n] = id_arr[i]
        offset += actual_n

    full_grid = pd.DataFrame({
        id_col: all_ids[:offset],
        time_col: all_times[:offset],
    })

    # Step 3: Merge original urine values onto grid (single merge!)
    orig = df[[id_col, time_col, urine_col]].drop_duplicates([id_col, time_col])
    full_grid = full_grid.merge(orig, on=[id_col, time_col], how='left')
    full_grid[urine_col] = full_grid[urine_col].fillna(0.0)
    full_grid = full_grid.sort_values([id_col, time_col]).reset_index(drop=True)

    # Step 4: Rolling sum using groupby().rolling() (single operation!)
    rolling_sum = (
        full_grid
        .groupby(id_col, sort=True)[urine_col]
        .rolling(window=window_steps, min_periods=min_steps, center=False)
        .sum()
    )
    # Drop group level from MultiIndex to align with full_grid
    if isinstance(rolling_sum.index, pd.MultiIndex):
        rolling_sum = rolling_sum.droplevel(0)
    rolling_sum_vals = rolling_sum.values

    # Step 5: Calculate urine24 = rolling_sum * step_factor / actual_window_length
    cumcount = full_grid.groupby(id_col).cumcount().values + 1
    actual_window_lens = np.minimum(cumcount, window_steps)

    full_grid['urine24'] = np.where(
        (actual_window_lens >= min_steps) & np.isfinite(rolling_sum_vals),
        rolling_sum_vals * step_factor / actual_window_lens,
        np.nan
    )

    return full_grid[[id_col, time_col, 'urine24']]


def _event_stage_at_exact_rows(
    result: pd.DataFrame,
    rrt_df: Optional[pd.DataFrame],
    id_col: str,
    time_col: str,
) -> pd.Series:
    stage = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if rrt_df is None or rrt_df.empty:
        return stage
    source_id = detect_id_col(rrt_df, id_col)
    source_time = detect_time_col(rrt_df, time_col)
    value_col = detect_value_col(rrt_df, "rrt")
    if source_id is None or source_time is None or value_col is None:
        raise AKIProfileError("RRT source has no ID/time/value contract")
    active = rrt_df.loc[
        rrt_df[value_col].fillna(False).astype(bool), [source_id, source_time]
    ]
    active_keys = set(map(tuple, active.to_numpy()))
    for index, row in result[[id_col, time_col]].iterrows():
        if (row[id_col], row[time_col]) in active_keys:
            stage.loc[index] = 3
    return stage


def _rolling_stage_max(
    frame: pd.DataFrame,
    id_col: str,
    time_col: str,
    *,
    stage_col: str,
    hours: float,
    time_unit: Optional[str],
) -> pd.Series:
    output = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, index in frame.groupby(id_col, sort=False).groups.items():
        group = frame.loc[index].sort_values(time_col)
        numeric_time = _time_hours(group[time_col], time_col, time_unit).to_numpy()
        stages = group[stage_col].fillna(0).to_numpy(int)
        maxima = [
            stages[(numeric_time >= current - hours) & (numeric_time <= current)].max()
            for current in numeric_time
        ]
        output.loc[group.index] = pd.array(maxima, dtype="Int64")
    return output


def _slide_vectorized_bulk(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    agg_map: dict,
    window_size_hours: float,
    full_window: bool,
    is_numeric_time: bool,
    _pre_sorted: bool = False,
) -> pd.DataFrame:
    """Bulk vectorized slide using pandas groupby().rolling().

    🚀 Performance: Eliminates per-patient Python loop entirely.
    All rolling computations happen in a single C-level call.

    cProfile benchmark: 5000 patients MIIV SOFA
    - Per-patient loop: 17.4s (sort + set_index + rolling + agg per patient)
    - Bulk groupby().rolling(): ~1-2s (single C-level operation)
    """
    # Sort globally once (the only sort we need!)
    if not _pre_sorted:
        data = data.sort_values(id_cols + [index_col]).reset_index(drop=True)
    elif not isinstance(data.index, pd.RangeIndex) or data.index[0] != 0:
        data = data.reset_index(drop=True)

    agg_cols = list(agg_map.keys())

    # Convert time to TimedeltaIndex for pandas rolling
    if is_numeric_time:
        time_unit = _infer_numeric_time_unit(data[index_col], index_col)
        td_index = pd.to_timedelta(data[index_col], unit=time_unit)
    else:
        td_index = pd.to_datetime(data[index_col])

    window_td = pd.Timedelta(hours=window_size_hours)

    # Create group key array
    if len(id_cols) == 1:
        group_key = data[id_cols[0]].values
    else:
        group_key = list(zip(*(data[c].values for c in id_cols)))

    # Create work DataFrame with only aggregation columns, indexed by time
    # 🚀 Avoid .copy() - construct from numpy arrays directly
    work = pd.DataFrame(
        {c: data[c].values for c in agg_cols},
        index=td_index,
    )

    # 🚀 核心: 单次 groupby().rolling().agg() 替代 N 次逐患者循环
    min_periods = None if full_window else 1
    grouped_rolling = (
        work
        .groupby(group_key, sort=False)[agg_cols]
        .rolling(window=window_td, closed='both', min_periods=min_periods)
    )

    # 🚀 PERF: 当所有聚合函数相同时，使用直接方法调用替代 .agg(dict)
    # .agg(dict) 对每列创建独立 Rolling 对象 (_gotitem)，开销大
    # 直接 .max() / .min() 走 Cython 快路径
    unique_funcs = set(agg_map.values())
    if len(unique_funcs) == 1:
        func_name = next(iter(unique_funcs))
        result = getattr(grouped_rolling, func_name)()
    else:
        result = grouped_rolling.agg({c: agg_map[c] for c in agg_cols})

    # 🚀 PERF: 用 .values 直接提取 numpy 数组，跳过昂贵的 droplevel + reset_index
    # droplevel(0) 对 6.8M 行 MultiIndex 花费 ~25s（Index.get_loc 51.8s）
    # rolling 保持行数和顺序与输入完全一致，可安全用 .values 提取
    result_df = pd.DataFrame(
        result.values,
        columns=agg_cols,
    )

    # Restore id and time columns from sorted data
    for col in id_cols:
        result_df[col] = data[col].values
    result_df[index_col] = data[index_col].values

    # Handle full_window post-filter
    if full_window:
        group_min = data.groupby(id_cols, sort=False)[index_col].transform('min')
        time_diff = data[index_col] - group_min
        if is_numeric_time:
            if time_unit == 's':
                threshold = window_size_hours * 3600
            elif time_unit == 'min':
                threshold = window_size_hours * 60
            else:
                threshold = window_size_hours
            mask = time_diff >= threshold
        else:
            mask = time_diff >= window_td
        result_df = result_df[mask.values].reset_index(drop=True)

    return result_df
