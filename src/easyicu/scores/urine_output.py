"""Collection-interval integration and provenance for the urine output chain.

Numeric axes at this boundary are ICU-relative hours. Event volumes cover the
preceding collection interval; rate records cover the same interval in mL/h.
Intervals longer than twelve hours contribute no volume or coverage and start
a new segment. Every window is open on the left and closed on the right.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import weakref

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer

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

#: Float tolerance for the 12 h record-gap rule (about 3.6 ms; sources are whole seconds).
GAP_TOLERANCE_H = 1e-6


def urine_availability_columns(name: str) -> list[str]:
    """Closed export contract for availability and scoring-window receipts."""
    if name in {"urine24", "urine24_est"}:
        return [f"{name}_available_at"]
    if name in {"kdigo_uo", "aki_stage_uo"}:
        return ["aki_stage_uo_available_at"]
    if name in {"kdigo_aki", "aki_stage_uo_reference"}:
        return ["aki_stage_uo_reference_available_at"]
    if name not in {"uo_6h", "uo_12h", "uo_24h"}:
        return []
    columns = [f"{name}_available_at", f"{name}_assessment_available_at",
               f"{name}_assessment_rate", f"{name}_assessment_covered_h"]
    if name == "uo_6h":
        columns.append("uo_6h_oliguria_available_at")
    return columns


def availability_fields(database: str, table: str | None = None) -> tuple[str, ...]:
    """Urine entry/store clocks declared by data-sources.json; SIC has none."""
    database = database.removesuffix("_demo")
    source_tables = {"eicu": "intakeoutput", "mimic": "outputevents", "miiv": "outputevents",
                     "hirid": "observations", "aumc": "numericitems"}
    if table is not None and table != source_tables.get(database):
        return ()
    return {
        "eicu": ("intakeoutputentryoffset",),
        "mimic": ("storetime",), "miiv": ("storetime",),
        "hirid": ("entertime",), "aumc": ("registeredat", "updatedat"),
    }.get(database, ())


def optional_availability_projection(path, columns, database, table):
    """Older converted files can omit entry clocks; inspect only their schema."""
    fields = set(availability_fields(database, table))
    if columns is None or not fields.intersection(str(c).lower() for c in columns):
        return columns
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        available = pq.read_schema(path).names
    elif path.name.lower().endswith((".csv", ".csv.gz")):
        available = pd.read_csv(path, nrows=0).columns
    else:
        return columns
    return available_source_columns(columns, available, database, table)


def available_source_columns(columns, available, database, table):
    """Only entry/store clocks are optional; required source fields stay strict."""
    if columns is None:
        return None
    fields = set(availability_fields(database, table))
    present = {str(c).lower() for c in available}
    return [c for c in columns if str(c).lower() not in fields or str(c).lower() in present]


def partitioned_source_columns(connection, relation, columns, database, table):
    """Inspect the already selected partitions, preserving SQL row pushdown."""
    if columns is None or not set(availability_fields(database, table)).intersection(str(c).lower() for c in columns):
        return columns
    available = [row[0] for row in connection.execute(f"SELECT * FROM {relation} LIMIT 0").description]
    return available_source_columns(columns, available, database, table)


def retain_availability_lag(frame, database, time):
    """Capture entry minus chart before admission alignment or rate pooling.

    DataSource has normalized AUMC to minutes; eICU offsets are also minutes.
    A lag survives all subsequent clock-origin changes. Missing source fields
    or per-record clocks fall back to chart time, never earlier than chart.
    """
    if "_urine_availability_lag_h" in frame or time not in frame:
        return
    lag = np.zeros(len(frame), dtype=float)
    for column in availability_fields(database):
        if column not in frame or frame[column].isna().all():
            continue
        entry = pd.to_datetime(frame[column], errors="coerce") if pd.api.types.is_datetime64_any_dtype(frame[time]) else frame[column]
        delta = entry - frame[time]
        if pd.api.types.is_timedelta64_dtype(delta):
            delta = delta / pd.Timedelta(hours=1)
        else:
            delta = pd.to_numeric(delta, errors="coerce") / 60.
        lag = np.fmax(lag, delta.to_numpy(dtype=float, na_value=np.nan))
    frame["_urine_availability_lag_h"] = lag


def source_time_select_sql(raw_hours, bucket_hours, valid, database, time, columns):
    """SQL receipts share the raw-path clocks without materializing source rows."""
    clock_lag = f"({raw_hours}) - ({bucket_hours})"
    prefix = "" if database.startswith("eicu") else "o."
    delays = ["0.0"]
    for column in availability_fields(database):
        if column not in columns:
            continue
        chart, entry = f'{prefix}"{time}"', f'{prefix}"{column}"'
        if database == "aumc" or database.startswith("eicu"):
            scale = 3600000. if database == "aumc" else 60.
            delays.append(f"(TRY_CAST({entry} AS DOUBLE) - TRY_CAST({chart} AS DOUBLE)) / {scale}")
        else:
            delays.append(f"EPOCH(TRY_CAST({entry} AS TIMESTAMP) - TRY_CAST({chart} AS TIMESTAMP)) / 3600.0")
    availability = f"({clock_lag}) + GREATEST({', '.join(delays)})"
    return (f", MAX(CASE WHEN {valid} THEN {clock_lag} END) AS _urine_source_lag_h"
            f", MIN(CASE WHEN {valid} THEN {clock_lag} END) AS _urine_first_lag_h"
            f", MAX(CASE WHEN {valid} THEN {availability} END) AS _urine_availability_lag_h")


def source_availability(frame, ids, time, interval):
    """Keep the latest contributing record time across hourly source pooling."""
    frame = frame.loc[frame.urine.notna()]
    receipts = frame[ids + [time]].copy()
    hours = time_hours(frame[time])
    receipts['_urine_chart_at'] = hours + frame.get('_urine_source_lag_h', 0.)
    receipts['_urine_available_at'] = hours + frame.get('_urine_availability_lag_h', frame.get('_urine_source_lag_h', 0.))
    receipts['_urine_first_at'] = hours + frame.get('_urine_first_lag_h', frame.get('_urine_source_lag_h', 0.))
    if interval is not None:
        width = pd.Timedelta(interval).total_seconds() / 3600
        receipts[time] = np.floor(hours / width) * width
    receipts = receipts.groupby(ids + [time], as_index=False).agg(
        {'_urine_available_at': 'max', '_urine_first_at': 'min', '_urine_chart_at': 'max'},
    )
    receipts['_urine_gap_h'] = receipts['_urine_first_at'] - receipts.groupby(ids)['_urine_chart_at'].shift()
    return receipts.drop(columns=['_urine_first_at', '_urine_chart_at'])


class _CollectionIndexer(BaseIndexer):
    """Monotone interval bounds; pandas computes maxima in compiled O(N) code."""

    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None, step=None):
        return self.index_array, np.arange(1, num_values + 1, dtype=np.int64)


def interval_availability(source_times):
    """Both endpoints determine an allocated interval, including its duration."""
    times = np.asarray(source_times, dtype=float)
    return np.fmax(times, np.r_[times[:1], times[:-1]])


def window_availability(source_times, left, valid):
    """Latest availability of every contributing interval, including zero urine."""
    values = np.where(valid, interval_availability(source_times), np.nan)
    return pd.Series(values).rolling(_CollectionIndexer(index_array=left.astype(np.int64)), min_periods=1).max().to_numpy()


def collection_intervals(
    ends: np.ndarray,
    amounts: np.ndarray,
    *,
    source_is_rate: bool = False,
    first_hours: float = 1.,
    max_gap: float = 12.,
    source_times: np.ndarray | None = None,
    record_gaps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared allocation and validity for one sorted stay, including suffixes."""
    ends = np.asarray(ends, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    lengths = np.r_[0. if source_is_rate else first_hours, np.diff(ends)]
    # Entry delays are provenance, not recording gaps. Raw chart gaps survive
    # hourly pooling separately; without that receipt use the chart axis.
    source_gaps = np.r_[0., np.diff(ends)]
    if record_gaps is not None:
        source_gaps = np.nan_to_num(record_gaps, nan=0.)
    # Gaps are differences of float hours on different clocks (minute clock,
    # sub-minute lags), so an exact 12 h gap can land 1e-15 h either side.
    # Source clocks are whole seconds: any real excess is >= 1 s (2.8e-4 h).
    limit = max_gap + GAP_TOLERANCE_H
    valid = np.isfinite(amounts) & (amounts >= 0) & (lengths > 0) & (lengths <= limit)
    valid &= source_gaps <= limit
    breaks = (lengths > limit) | (source_gaps > limit) | ~np.isfinite(amounts) | (amounts < 0)
    segment = np.maximum.accumulate(np.where(breaks, np.arange(len(ends)), 0))
    starts = ends - lengths
    rates = amounts.copy() if source_is_rate else np.divide(
        amounts, lengths, out=np.zeros_like(amounts), where=lengths > 0,
    )
    rates = np.where(valid, rates, 0.)
    return starts, lengths, rates, valid, segment


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
    if not len(ends):
        return {w: (ends.copy(), ends.copy(), ends.copy()) for w in windows}
    starts, lengths, rates, valid, segment = collection_intervals(
        ends, amounts, source_is_rate=source_is_rate, first_hours=first_hours,
        max_gap=max_gap, source_times=source_times, record_gaps=record_gaps,
    )
    available_times = ends if source_times is None else np.asarray(source_times, dtype=float)
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
        available = np.where(covered > 1e-9, window_availability(available_times, left, valid), np.nan)
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
    # Fill positional arrays per stay, then publish each column once. Keep
    # the kernel, arithmetic order, output dtypes and column order unchanged.
    arrays = {
        window: (np.full(len(data), np.nan), np.zeros(len(data)),
                 np.ones(len(data), dtype=bool), np.full(len(data), np.nan))
        for window, _ in windows
    }
    all_amounts = data.urine.to_numpy(dtype=float)
    all_weights = pd.to_numeric(data.weight, errors="coerce").to_numpy(dtype=float)
    source_times = data['_urine_available_at'].to_numpy() if '_urine_available_at' in data else None
    record_gaps = data['_urine_gap_h'].to_numpy() if '_urine_gap_h' in data else None
    for positions in data.groupby(ids, sort=False).indices.values():
        amounts = all_amounts[positions]
        # HiRID source callbacks express a rate as volume per extraction bin.
        if source_is_rate:
            amounts = amounts / width
        integrated = collection_windows(times[positions], amounts, [w for w, _ in windows],
                                        source_is_rate=source_is_rate, first_hours=width,
                                        source_times=source_times[positions] if source_times is not None else None,
                                        record_gaps=record_gaps[positions] if record_gaps is not None else None)
        weights = all_weights[positions]
        for window, minimum in windows:
            volume, covered, available = integrated[window]
            valid = (covered >= minimum - 1e-9) & np.isfinite(weights) & (weights > 0)
            values = np.divide(volume, covered * weights, out=np.full(len(volume), np.nan), where=valid)
            rates, coverage, partial, availability = arrays[window]
            rates[positions] = values
            coverage[positions] = covered
            partial[positions] = covered < window - 1e-9
            availability[positions] = available
    result = {}
    for window, _ in windows:
        name = f"uo_{window}h"
        frame = data[ids + [time_col]].copy()
        for column, values in zip(
            (name, f"{name}_covered_h", f"{name}_partial_window", f"{name}_available_at"),
            arrays[window],
        ):
            frame[column] = values
        result[name] = frame
    return result


@dataclass
class _DescriptiveWindowCacheEntry:
    urine: weakref.ReferenceType
    weight: weakref.ReferenceType
    data: dict[str, pd.DataFrame]


def shared_descriptive_windows(
    urine, weight, windows, *, requested_window, resolver, interval, source_is_rate,
):
    """Reuse identical source integration across windows in one bounded request."""
    budget = resolver._cache_budget_bytes
    # Conservatively include the source keys, float/bool outputs and index.
    # If the bundle is unlikely to fit, keep the old single-window work rather
    # than repeatedly computing three windows that cannot be retained.
    estimated = len(windows) * (int(urine.memory_usage(index=True, deep=False).sum()) + 33 * len(urine))
    if budget is not None and estimated > budget:
        return descriptive_windows(
            urine, weight, [spec for spec in windows if spec[0] == requested_window],
            interval=interval, source_is_rate=source_is_rate,
        )
    key = ("urine_descriptive_windows", id(urine), id(weight), tuple(windows),
           pd.Timedelta(interval), source_is_rate)
    with resolver._cache_lock:
        entry = resolver._concept_data_cache.get(key)
        if (isinstance(entry, _DescriptiveWindowCacheEntry)
                and entry.urine() is urine and entry.weight() is weight):
            resolver._cache_touch("data", key)
            return entry.data
        data = descriptive_windows(urine, weight, windows, interval=interval, source_is_rate=source_is_rate)
        # A bundle contains three frames, so charge their complete size rather
        # than the generic cache estimator's small non-frame fallback.
        size = sum(int(frame.memory_usage(index=True, deep=True).sum()) for frame in data.values())
        resolver._bounded_cache_store(
            "data", key, _DescriptiveWindowCacheEntry(weakref.ref(urine), weakref.ref(weight), data),
            charge_bytes=size,
        )
        return data


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
    rec["_chart_time"] = rec[time_col]
    rec = rec.groupby([id_col, "_k"], sort=True, as_index=False).agg(
        **{urine_col: (urine_col, "sum"), "_source_time": ("_source_time", "max"),
           '_urine_gap_h': ('_urine_gap_h', 'max'), '_chart_time': ('_chart_time', 'max')}
    )

    gap = rec.groupby(id_col, sort=False)["_k"].diff().fillna(1).astype(np.int64).to_numpy()
    elapsed = rec.groupby(id_col, sort=False)["_chart_time"].diff().fillna(time_step).to_numpy()
    if '_urine_gap_h' in df:
        elapsed = rec['_urine_gap_h'].fillna(time_step).to_numpy()
    rejected = (gap > cap_steps) | (elapsed > cap_steps * time_step)
    placed = ~rejected
    reps = gap[placed]
    within = np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps)
    previous = rec.groupby(id_col, sort=False)["_source_time"].shift().to_numpy()
    available = np.fmax(rec["_source_time"].to_numpy(), previous)
    cells = pd.DataFrame({
        id_col: np.repeat(rec[id_col].to_numpy()[placed], reps),
        "_k": np.repeat(rec["_k"].to_numpy()[placed] - reps + 1, reps) + within,
        "_available": np.repeat(available[placed], reps),
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
