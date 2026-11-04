"""HiRID daily fluid counters as admission-anchored hourly volumes.

HiRID ``observations`` carries two variables that the official variable
reference names "Fluid balance" (unit "cummulative"): 30005010 and 30005110.
Neither is a net balance. An aggregate audit of the full HiRID v1.1.1
observations (2026-11-03) established their semantics from the data:

* both are non-decreasing counters that restart once a day at 12:00: 98.3%
  (30005010) and 91.8% (30005110) of all decreases fall exactly on 12:00, and
  99.2% of consecutive readings that straddle 12:00 decrease;
* 30005010 is never negative and is at least the sum of the separately
  charted saline (30005075) and colloid (30005080) counters in 99.99% of
  12:00-anchored days, so it is the cumulative fluid INPUT of the day;
* 30005110 is at least 0.9x the time-weighted hourly urine (10020000) in 91%
  of well-covered days (median ratio 1.46) and tracks urine (r=0.36) while the
  input counter does not (r=0.07), so it is the cumulative fluid OUTPUT of the
  day (urine plus other recorded losses).

Hourly volumes are therefore derived from counter increments, never from the
counter level. Within one 12:00-anchored day the increment is the rise of the
running maximum, which is non-negative and sums to the day's highest reading;
an intra-day downward correction does not count twice. A reading below half
of the running maximum is an off-schedule restart, not a correction: the
running maximum restarts from it (round-2 audit F1: 43 reset-like drops of
530 intra-day decreases in a 10% sample; ignoring them lost 4.6% of output).
Each increment is allocated over the interval since the previous reading of
the same day (or since the 12:00 restart) with exact volume conservation.

A net balance needs both counters on the same 12:00-anchored day; a day with
only one counter has no balance (round-2 audit F1: 6.2% of patient-days were
input-only and read as +input).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.callback_utils import distribute_volume_hourly

INPUT_COUNTER_ID = 30005010
OUTPUT_COUNTER_ID = 30005110
COUNTER_RESET_HOUR = 12
# Readings above 100 L in one day are the int64/float overflow sentinels seen
# in the export (~9.2e18 and ~9.2e31), not volumes.
MAX_DAILY_COUNTER_ML = 100_000.0
#: A reading below this share of the running maximum restarts the counter.
RESTART_FRACTION = 0.5
#: ...unless the next reading of the same day returns to this share of it.
REBOUND_FRACTION = 0.8


def _restart_aware_increments(ids: np.ndarray, days: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Increment of each reading over its block's running maximum.

    Rows are sorted by stay and time.  A new stay or 12:00 day starts a block;
    so does a reading below ``RESTART_FRACTION`` of the running maximum, unless
    the next reading of the same day rebounds to ``REBOUND_FRACTION`` of it: a
    single low reading between two on-trajectory ones is a charting dip, and
    restarting at it counted the day's total twice (2026-11-04 audit round 3,
    M F1: 109 of 470 drops, 163,744 mL of output).
    """
    out = np.zeros(len(values), dtype=float)
    peak = 0.0
    n = len(values)
    for i in range(n):
        if i == 0 or ids[i] != ids[i - 1] or days[i] != days[i - 1]:
            peak = 0.0
        elif values[i] < RESTART_FRACTION * peak:
            rebounds = (
                i + 1 < n
                and ids[i + 1] == ids[i]
                and days[i + 1] == days[i]
                and values[i + 1] >= REBOUND_FRACTION * peak
            )
            if rebounds:
                continue
            peak = 0.0
        if values[i] > peak:
            out[i] = values[i] - peak
            peak = values[i]
    return out


def counter_increments(
    frame: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    value_col: str,
    reset_hour: int = COUNTER_RESET_HOUR,
) -> pd.DataFrame:
    """Return one ``(id, start, end, volume)`` row per counter reading.

    ``time_col`` must hold absolute clock times. Invalid readings (missing,
    negative, or above the overflow bound) are discarded before increments
    are formed so they cannot restart or inflate the running maximum.
    """

    columns = [id_col, "start", "end", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    data = pd.DataFrame(
        {
            id_col: frame[id_col].to_numpy(),
            "end": pd.to_datetime(frame[time_col], errors="coerce").to_numpy(),
            "value": pd.to_numeric(frame[value_col], errors="coerce").to_numpy(),
        }
    )
    valid = (
        data["end"].notna()
        & data["value"].notna()
        & data["value"].ge(0)
        & data["value"].le(MAX_DAILY_COUNTER_ML)
    )
    data = data.loc[valid].sort_values([id_col, "end"], kind="mergesort")
    if data.empty:
        return pd.DataFrame(columns=columns)

    offset = pd.Timedelta(hours=reset_hour)
    data["day_start"] = (data["end"] - offset).dt.normalize() + offset
    group = [id_col, "day_start"]
    data["volume"] = _restart_aware_increments(
        data[id_col].to_numpy(), data["day_start"].to_numpy(), data["value"].to_numpy(dtype=float)
    )
    previous_time = data.groupby(group, sort=False)["end"].shift(1)
    data["start"] = previous_time.fillna(data["day_start"])
    return data[columns].reset_index(drop=True)


def hourly_counter_volume(
    frame: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    value_col: str,
    origins: pd.Series,
) -> pd.DataFrame:
    """Allocate counter increments to ICU-relative hours ``[h, h + 1)``.

    ``origins`` maps the stay identifier to its absolute admission time.
    Returns ``id_col``, ``hour`` (float bin start) and ``volume`` (mL).
    """

    increments = counter_increments(
        frame, id_col=id_col, time_col=time_col, value_col=value_col
    )
    if increments.empty:
        return pd.DataFrame(columns=[id_col, "hour", "volume"])
    origin = pd.to_datetime(increments[id_col].map(origins), errors="coerce")
    increments = increments.loc[origin.notna()].copy()
    origin = origin.loc[increments.index]
    # A day's first reading may follow a 12:00 restart that preceded
    # admission; volume cannot accrue in the ICU before the stay began.
    start = increments["start"].where(increments["start"] >= origin, origin)
    start = start.where(start <= increments["end"], increments["end"])
    increments["start"] = (start - origin).dt.total_seconds() / 3600.0
    increments["end"] = (increments["end"] - origin).dt.total_seconds() / 3600.0
    allocated = distribute_volume_hourly(
        increments,
        val_col="volume",
        end_col="end",
        index_col="start",
        id_col=id_col,
    )
    return allocated.rename(columns={"start": "hour"})[[id_col, "hour", "volume"]]


def _absolute_times(
    frame: pd.DataFrame, id_col: str, time_col: str, origins: pd.Series
) -> pd.Series:
    raw = frame[time_col]
    if pd.api.types.is_numeric_dtype(raw):
        origin = pd.to_datetime(frame[id_col].map(origins), errors="coerce")
        return origin + pd.to_timedelta(pd.to_numeric(raw, errors="coerce"), unit="h")
    return pd.to_datetime(raw, errors="coerce")


def _restore_time_axis(
    hours: pd.Series, ids: pd.Series, origins: pd.Series, numeric: bool
) -> pd.Series:
    if numeric:
        return hours.astype(float)
    origin = pd.to_datetime(ids.map(origins), errors="coerce")
    return origin + pd.to_timedelta(hours, unit="h")


def hirid_counter_frame(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    origins: pd.Series,
    value_col: str,
    time_col: str,
    item_col: str = "variableid",
    net_balance: bool = False,
) -> pd.DataFrame:
    """Turn HiRID counter readings into one hourly row per patient-hour.

    ``net_balance=False`` returns the hourly volume of every counter present
    (``total_input_ml`` maps only 30005010). ``net_balance=True`` returns input
    minus output and is emitted only for patients with readings of BOTH
    counters; a missing side would otherwise fabricate +input or -output.
    The time column keeps its incoming representation (ICU-relative hours or
    absolute datetimes) so the generic alignment stage applies unchanged.
    """

    id_col = next(
        (c for c in ("patientid", "stay_id", "admissionid") if c in frame.columns),
        None,
    )
    if id_col is None:
        raise ValueError("HiRID fluid counters require a patientid column")
    if value_col not in frame.columns:
        value_col = concept_name
    output_columns = [id_col, time_col, concept_name]
    if frame.empty or value_col not in frame.columns:
        return pd.DataFrame(columns=output_columns)

    numeric_time = pd.api.types.is_numeric_dtype(frame[time_col])
    absolute = frame.assign(_abs_time=_absolute_times(frame, id_col, time_col, origins))
    items = pd.to_numeric(absolute[item_col], errors="coerce")

    def volume(counter_id: int) -> pd.DataFrame:
        part = absolute.loc[items.eq(counter_id)]
        return hourly_counter_volume(
            part,
            id_col=id_col,
            time_col="_abs_time",
            value_col=value_col,
            origins=origins,
        )

    if net_balance:
        inflow = volume(INPUT_COUNTER_ID)
        outflow = volume(OUTPUT_COUNTER_ID)
        both = set(inflow[id_col]) & set(outflow[id_col])
        if not both:
            return pd.DataFrame(columns=output_columns)
        offset = pd.Timedelta(hours=COUNTER_RESET_HOUR)
        reading_day = (pd.to_datetime(absolute["_abs_time"], errors="coerce") - offset).dt.normalize() + offset
        counted = pd.to_numeric(absolute[value_col], errors="coerce").between(0, MAX_DAILY_COUNTER_ML)
        days = pd.DataFrame({id_col: absolute[id_col], "day": reading_day, "item": items}).loc[counted]
        per_day = days.groupby([id_col, "day"])["item"].agg(
            lambda s: {INPUT_COUNTER_ID, OUTPUT_COUNTER_ID} <= set(s)
        )
        complete_days = set(per_day.index[per_day.to_numpy(dtype=bool)])
        merged = pd.merge(
            inflow.loc[inflow[id_col].isin(both)],
            outflow.loc[outflow[id_col].isin(both)],
            on=[id_col, "hour"],
            how="outer",
            suffixes=("_in", "_out"),
        )
        merged[concept_name] = merged["volume_in"].fillna(0.0) - merged[
            "volume_out"
        ].fillna(0.0)
        # The day of an hour is the 12:00 day of its clock start.
        clock = pd.to_datetime(merged[id_col].map(origins), errors="coerce") + pd.to_timedelta(
            merged["hour"], unit="h"
        )
        hour_day = (clock - offset).dt.normalize() + offset
        keep = [key in complete_days for key in zip(merged[id_col], hour_day)]
        hourly = merged.loc[keep, [id_col, "hour", concept_name]]
    else:
        parts = [volume(int(c)) for c in sorted(items.dropna().unique())]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(columns=output_columns)
        hourly = (
            pd.concat(parts, ignore_index=True)
            .groupby([id_col, "hour"], as_index=False, sort=True)["volume"]
            .sum()
            .rename(columns={"volume": concept_name})
        )

    hourly = hourly.sort_values([id_col, "hour"], kind="mergesort").reset_index(drop=True)
    hourly[time_col] = _restore_time_axis(
        hourly["hour"], hourly[id_col], origins, numeric_time
    )
    result = hourly[[id_col, time_col, concept_name]]
    if not np.isfinite(pd.to_numeric(result[concept_name], errors="coerce")).all():
        raise ValueError("HiRID fluid counter produced a non-finite hourly volume")
    return result


def load_hirid_admission_origins(data_source) -> pd.Series:
    """Return ``patientid -> admissiontime`` from HiRID ``general``."""

    general = data_source.load_table("general", verbose=False)
    table = general.data if hasattr(general, "data") else general
    if "patientid" not in table.columns or "admissiontime" not in table.columns:
        raise ValueError("HiRID general table lacks patientid/admissiontime")
    origins = table[["patientid", "admissiontime"]].dropna().drop_duplicates("patientid")
    return pd.Series(
        pd.to_datetime(origins["admissiontime"], errors="coerce").to_numpy(),
        index=origins["patientid"].to_numpy(),
    )
