"""Collection-interval integration and provenance for the urine output chain.

Numeric axes at this boundary are ICU-relative hours. Event volumes cover the
preceding collection interval; rate records cover the same interval in mL/h.
Intervals longer than twelve hours contribute no volume or coverage and start
a new segment. Every window is open on the left and closed on the right.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from easyicu.urine_weight_linkage import (
    resolve_keyed_unique_weights,
    resolve_unkeyed_single_entity_weight,
)


DAILY_COLUMNS = [
    "urine24", "urine24_est", "urine24_observed_ml", "urine24_covered_h",
    "urine24_available_at", "urine24_est_available_at", "urine24_uses_future",
]
KDIGO_EVIDENCE_COLUMNS = [
    *[f"uo_rt_{w}hr_covered_h" for w in (6, 12, 24)],
    "aki_stage_uo_available_at",
]


def source_availability(frame, ids, time, interval):
    """Keep the latest contributing record time across hourly source pooling."""
    frame = frame.loc[frame.urine.notna()]
    receipts = frame[ids + [time]].copy()
    hours = time_hours(frame[time])
    receipts['_urine_available_at'] = hours + frame.get('_urine_source_lag_h', 0.)
    receipts['_urine_first_at'] = hours + frame.get('_urine_first_lag_h', frame.get('_urine_source_lag_h', 0.))
    width = pd.Timedelta(interval).total_seconds() / 3600
    receipts[time] = np.floor(hours / width) * width
    receipts = receipts.groupby(ids + [time], as_index=False).agg(
        {'_urine_available_at': 'max', '_urine_first_at': 'min'},
    )
    receipts['_urine_gap_h'] = receipts['_urine_first_at'] - receipts.groupby(ids)['_urine_available_at'].shift()
    return receipts.drop(columns='_urine_first_at')


def collection_windows(
    ends: np.ndarray,
    amounts: np.ndarray,
    windows: Sequence[int],
    *,
    source_is_rate: bool = False,
    first_hours: float = 1.,
    max_gap: float = 12.,
    source_times: np.ndarray | None = None,
    record_gaps: np.ndarray | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return volume, covered hours and latest source time for each window.

    Arrays describe one stay, sorted with one record per timestamp. Invalid
    records are explicit holes. A rate source has no coverage before its first
    record; an event source retains the existing first-bin convention.
    """
    ends = np.asarray(ends, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if not len(ends):
        return {w: (ends.copy(), ends.copy(), ends.copy()) for w in windows}
    lengths = np.r_[0. if source_is_rate else first_hours, np.diff(ends)]
    available_times = ends if source_times is None else np.asarray(source_times, dtype=float)
    source_gaps = np.r_[0., np.diff(available_times)]
    if record_gaps is not None:
        source_gaps = np.nan_to_num(record_gaps, nan=0.)
    valid = np.isfinite(amounts) & (amounts >= 0) & (lengths > 0) & (lengths <= max_gap)
    valid &= source_gaps <= max_gap
    breaks = (lengths > max_gap) | (source_gaps > max_gap) | ~np.isfinite(amounts) | (amounts < 0)
    segment = np.maximum.accumulate(np.where(breaks, np.arange(len(ends)), 0))
    starts = ends - lengths
    rates = amounts.copy() if source_is_rate else np.divide(
        amounts, lengths, out=np.zeros_like(amounts), where=lengths > 0,
    )
    rates = np.where(valid, rates, 0.)
    covered_lengths = np.where(valid, lengths, 0.)
    cumulative_h = np.r_[0., np.cumsum(covered_lengths)]
    cumulative_v = np.r_[0., np.cumsum(rates * covered_lengths)]
    right = np.arange(len(ends))
    result = {}
    for window in windows:
        left_edge = ends - window
        left = np.searchsorted(ends, left_edge, side="right")
        left = np.maximum(np.minimum(left, right), segment)
        trimmed = np.minimum(np.maximum(left_edge - starts[left], 0.), lengths[left])
        trimmed = np.where(valid[left], trimmed, 0.)
        covered = np.maximum(cumulative_h[right + 1] - cumulative_h[left] - trimmed, 0.)
        volume = np.maximum(cumulative_v[right + 1] - cumulative_v[left] - rates[left] * trimmed, 0.)
        available = np.where(covered > 1e-9, available_times, np.nan)
        result[window] = (volume, covered, available)
    return result


def time_hours(values: pd.Series) -> np.ndarray:
    """Normalize an explicit concept-layer hour axis, without spacing guesses."""
    if pd.api.types.is_timedelta64_dtype(values):
        return (values / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    if pd.api.types.is_datetime64_any_dtype(values):
        return ((values - values.min()) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def descriptive_windows(
    urine: pd.DataFrame,
    weight: pd.DataFrame,
    windows: Sequence[tuple[int, float]],
    *,
    interval: pd.Timedelta | None = None,
    source_is_rate: bool = False,
) -> dict[str, pd.DataFrame]:
    """Descriptive rates use apportioned volume / actual overlap / weight."""
    ids = [c for c in urine if c.endswith("_id") or c in {
        "icustay_id", "patientunitstayid", "admissionid", "patientid", "CaseID", "patient_SN", "INP_NO",
    }]
    time_col = next((c for c in (
        "charttime", "measuredat", "measuredat_minutes", "nursingchartoffset",
        "observationoffset", "intakeoutputoffset", "registeredat", "datetime", "Offset", "offset",
    ) if c in urine), None)
    if not ids or time_col is None:
        raise ValueError("Urine windows require patient and time columns")
    data = urine[ids + [time_col, "urine"]].copy().dropna(subset=ids + [time_col])
    data["urine"] = pd.to_numeric(data.urine, errors="coerce")
    data.loc[~np.isfinite(data.urine) | (data.urine < 0), "urine"] = np.nan
    groups = data.groupby(ids + [time_col], sort=True, dropna=False).urine
    grouped = groups.agg(["sum", "mean", "count", "size"]).reset_index()
    grouped["urine"] = grouped["mean" if source_is_rate else "sum"].where(grouped["count"] == grouped["size"])
    data = grouped[ids + [time_col, "urine"]]
    if '_urine_available_at' in urine:
        receipt_cols = [c for c in ('_urine_available_at', '_urine_gap_h') if c in urine]
        receipts = urine.groupby(ids + [time_col], as_index=False)[receipt_cols].max()
        data = data.merge(receipts, on=ids + [time_col], how='left', validate='one_to_one')
    common_ids = [c for c in ids if c in weight]
    if common_ids and "weight" in weight:
        keyed = resolve_keyed_unique_weights(weight, id_columns=common_ids)
        data = data.merge(keyed, on=common_ids, how="left", validate="many_to_one")
    else:
        resolution = resolve_unkeyed_single_entity_weight(data, weight, urine_id_columns=ids)
        data["weight"] = resolution.weight if resolution.weight is not None else np.nan
    # Preserve the descriptive API's existing omission of unweighted stays
    # whenever at least one stay has a usable weight.
    weight_ok = pd.to_numeric(data.weight, errors="coerce").gt(0)
    if weight_ok.any():
        data = data.loc[weight_ok].reset_index(drop=True)
    width = pd.Timedelta(interval or pd.Timedelta(hours=1)).total_seconds() / 3600
    times = time_hours(data[time_col])
    result = {}
    for window, minimum in windows:
        name = f"uo_{window}h"
        frame = data[ids + [time_col]].copy()
        frame[name] = np.nan
        frame[f"{name}_covered_h"] = 0.
        frame[f"{name}_partial_window"] = True
        result[name] = frame
    for positions in data.groupby(ids, sort=False).indices.values():
        amounts = data.urine.to_numpy(dtype=float)[positions]
        # HiRID source callbacks express a rate as volume per extraction bin.
        if source_is_rate:
            amounts = amounts / width
        integrated = collection_windows(times[positions], amounts, [w for w, _ in windows],
                                        source_is_rate=source_is_rate, first_hours=width,
                                        source_times=data['_urine_available_at'].to_numpy()[positions]
                                        if '_urine_available_at' in data else None,
                                        record_gaps=data['_urine_gap_h'].to_numpy()[positions]
                                        if '_urine_gap_h' in data else None)
        weights = pd.to_numeric(data.weight.iloc[positions], errors="coerce").to_numpy(dtype=float)
        for window, minimum in windows:
            volume, covered, _ = integrated[window]
            valid = (covered >= minimum - 1e-9) & np.isfinite(weights) & (weights > 0)
            values = np.divide(volume, covered * weights, out=np.full(len(volume), np.nan), where=valid)
            name = f"uo_{window}h"
            result[name].loc[positions, name] = values
            result[name].loc[positions, f"{name}_covered_h"] = covered
            result[name].loc[positions, f"{name}_partial_window"] = covered < window - 1e-9
    return result


def assign_daily(
    frame: pd.DataFrame,
    volume: np.ndarray,
    covered: np.ndarray,
    available: np.ndarray,
    *,
    minimum: float = 18.,
    window: float = 24.,
) -> pd.DataFrame:
    """Separate full measured volume from a partial-window daily equivalent."""
    full = covered >= window - 1e-9
    partial = (covered >= minimum - 1e-9) & ~full
    frame["urine24"] = np.where(full, volume, np.nan)
    frame["urine24_est"] = np.divide(volume * window, covered,
                                     out=np.full(len(frame), np.nan), where=partial)
    frame["urine24_observed_ml"] = np.where(covered > 0, volume, np.nan)
    frame["urine24_covered_h"] = covered
    frame["urine24_available_at"] = np.where(full, available, np.nan)
    frame["urine24_est_available_at"] = np.where(partial, available, np.nan)
    frame["urine24_uses_future"] = (full | partial) & (available > frame["_hours"].to_numpy() + 1e-9)
    return frame


def rate_daily(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str,
    *,
    minimum: float = 18.,
    max_gap: float = 12.,
    rate_width: float = 1.,
) -> pd.DataFrame:
    """Daily rate integration uses the same gap rejection as KDIGO."""
    data = data.copy()
    data['_urine_available_at'] = data.get('_urine_available_at', data[time_col])
    aggregation = {urine_col: 'mean', '_urine_available_at': 'max'}
    if '_urine_gap_h' in data:
        aggregation['_urine_gap_h'] = 'max'
    data = data.groupby([id_col, time_col], as_index=False, sort=True).agg(aggregation)
    output = []
    for _, group in data.groupby(id_col, sort=False):
        hours = group[time_col].to_numpy(dtype=float)
        volume, covered, available = collection_windows(
            hours, group[urine_col].to_numpy(dtype=float) / rate_width, [24],
            source_is_rate=True, max_gap=max_gap, source_times=group['_urine_available_at'].to_numpy(),
            record_gaps=group['_urine_gap_h'].to_numpy() if '_urine_gap_h' in group else None,
        )[24]
        output.append(assign_daily(group.assign(_hours=hours), volume, covered, available, minimum=minimum))
    columns = [id_col, time_col, *DAILY_COLUMNS]
    return pd.concat(output, ignore_index=True)[columns] if output else pd.DataFrame(columns=columns)


def daily_volume_batch(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str,
    time_step: float,
    window_steps: int,
    min_steps: int,
    cap_steps: int,
) -> pd.DataFrame:
    """Retain the event-grid apportioning and export coverage and source time."""
    out_cols = [id_col, time_col, *DAILY_COLUMNS]
    rec = df[[id_col, time_col, urine_col]].copy()
    rec['_urine_available_at'] = df.get('_urine_available_at', df[time_col])
    rec['_urine_gap_h'] = df.get('_urine_gap_h', np.nan)
    rec[urine_col] = pd.to_numeric(rec[urine_col], errors="coerce")
    rec = rec.dropna(subset=[time_col, urine_col])
    if rec.empty:
        return pd.DataFrame(columns=out_cols)
    rec["_k"] = np.floor(rec[time_col].to_numpy(dtype=np.float64) / time_step + 1e-9).astype(np.int64)
    rec["_source_time"] = rec['_urine_available_at']
    rec = rec.groupby([id_col, "_k"], sort=True, as_index=False).agg(
        **{urine_col: (urine_col, "sum"), "_source_time": ("_source_time", "max"),
           '_urine_gap_h': ('_urine_gap_h', 'max')}
    )

    gap = rec.groupby(id_col, sort=False)["_k"].diff().fillna(1).astype(np.int64).to_numpy()
    elapsed = rec.groupby(id_col, sort=False)["_source_time"].diff().fillna(time_step).to_numpy()
    if '_urine_gap_h' in df:
        elapsed = rec['_urine_gap_h'].fillna(time_step).to_numpy()
    rejected = (gap > cap_steps) | (elapsed > cap_steps * time_step)
    placed = ~rejected
    reps = gap[placed]
    within = np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps)
    cells = pd.DataFrame({
        id_col: np.repeat(rec[id_col].to_numpy()[placed], reps),
        "_k": np.repeat(rec["_k"].to_numpy()[placed] - reps + 1, reps) + within,
        "_available": np.repeat(rec["_source_time"].to_numpy()[placed], reps),
        "_vol": np.repeat(rec[urine_col].to_numpy(dtype=np.float64)[placed] / reps, reps),
    })
    cells["_cov"] = 1.0

    bounds = rec.groupby(id_col, sort=True)["_k"].agg(["min", "max"])
    sizes = (bounds["max"] - bounds["min"] + 1).to_numpy()
    grid = pd.DataFrame({
        id_col: np.repeat(bounds.index.to_numpy(), sizes),
        "_k": np.repeat(bounds["min"].to_numpy(), sizes)
        + (np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)),
    })
    # Long unplaceable intervals start a new evaluable segment, not a cap.
    rec["_segment"] = pd.Series(rejected).groupby(rec[id_col]).cumsum().to_numpy()
    grid = grid.merge(rec[[id_col, "_k", "_segment"]], on=[id_col, "_k"], how="left")
    grid["_segment"] = grid.groupby(id_col)["_segment"].ffill().fillna(0)
    grid = grid.merge(cells, on=[id_col, "_k"], how="left")
    grid[["_vol", "_cov"]] = grid[["_vol", "_cov"]].fillna(0.0)
    grid = grid.sort_values([id_col, "_k"]).reset_index(drop=True)

    rolled = (
        grid.groupby([id_col, "_segment"], sort=True)[["_vol", "_cov"]]
        .rolling(window=window_steps, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .sort_index()
    )
    covered = rolled["_cov"].to_numpy()
    available = (grid.groupby([id_col, "_segment"], sort=True)["_available"]
                 .rolling(window=window_steps, min_periods=1).max()
                 .reset_index(level=[0, 1], drop=True).sort_index().to_numpy())
    grid["_hours"] = grid["_k"].to_numpy(dtype=float) * time_step
    assign_daily(grid, rolled["_vol"].to_numpy(), covered * time_step, available,
                 minimum=min_steps * time_step, window=window_steps * time_step)
    grid[time_col] = grid["_k"].to_numpy(dtype=np.float64) * time_step
    return grid[out_cols]
