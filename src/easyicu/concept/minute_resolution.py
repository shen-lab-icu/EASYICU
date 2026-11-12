"""Minute-resolution (sub-hour) loading contract.

Owner of the opt-in sub-hour mode used by the ``subhour_signals`` export
module. The default exports are hourly, and two of their behaviours are part
of the sealed output: the DuckDB fast path aggregates to the hour even when a
concept declares a one-minute grid, and SICdb ``data_float_h`` contributes only
its hourly ``Val``. Both change only when a caller asks for minute rows
explicitly with ``MINUTE_RESOLUTION_KWARG``; a concept's own ``interval`` never
turns this mode on.

SICdb stores minute samples in ``data_float_h.rawdata``: ``0x`` and 60
little-endian IEEE 754 float32 values, one per minute of the hour that starts
at ``Offset`` (SICdb Documentation, "Raw Data"/"Encoding"). The hourly ``Val``
is ``sum(minute values) / cnt``. The documentation defines an all-zero slot as
NULL, yet ``cnt`` also counts measured zeros, so a zero slot is a value only
when the row is complete (``cnt == 60``). Where ``cnt`` exceeds the non-zero
slots of an incomplete row, the measured zeros cannot be placed in time; they
are dropped and counted, never guessed.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

MINUTE_RESOLUTION_KWARG = "_minute_resolution"

# Opt-in export modules that load at a sub-hour grid: module -> grid.
SUBHOUR_MODULE_INTERVALS: Mapping[str, pd.Timedelta] = {
    "subhour_signals": pd.Timedelta(minutes=1),
}

SICDB_MINUTE_TABLE = "data_float_h"
SICDB_MINUTE_COLUMNS = ("cnt", "rawdata")
SICDB_RECEIPT_ATTR = "sicdb_minute_rows"
_SLOTS = 60
_HEX_LENGTH = 2 + 8 * _SLOTS


def minute_resolution_requested(options: Optional[Mapping], interval) -> bool:
    """True only for an explicit opt-in with a grid shorter than one hour."""
    if not options or options.get(MINUTE_RESOLUTION_KWARG) is not True or interval is None:
        return False
    return pd.to_timedelta(interval) < pd.Timedelta(hours=1)


def sicdb_minute_rows_apply(database: str, table: str, options: Optional[Mapping], interval) -> bool:
    """Whether a SICdb source read must expand ``rawdata`` into minute rows."""
    return (
        str(database) in {"sic", "sic_demo"}
        and table == SICDB_MINUTE_TABLE
        and minute_resolution_requested(options, interval)
    )


def expand_sicdb_minute_rows(
    frame: pd.DataFrame,
    *,
    value_column: str = "Val",
    time_column: str = "Offset",
) -> pd.DataFrame:
    """Replace each hourly row that has ``rawdata`` with its minute rows.

    Rows without a well-formed ``rawdata`` keep their hourly value at their
    own ``Offset`` (a single sample has no minute stream). The counts of each
    rule are attached as ``frame.attrs[SICDB_RECEIPT_ATTR]``.
    """
    missing = [c for c in (value_column, time_column, *SICDB_MINUTE_COLUMNS) if c not in frame]
    if missing:
        raise ValueError(f"sicdb_minute_rows_missing_columns: {missing}")
    raw = frame["rawdata"]
    if (pd.api.types.is_numeric_dtype(raw.dtype) or pd.api.types.is_bool_dtype(raw.dtype)) and raw.notna().any():
        # SICdb ships rawdata as hex text; numeric values mean a reader cast
        # it (data-sources.json declares col_double) and lost the stream. An
        # all-null column carries no stream, so its hourly rows stand.
        raise ValueError("sicdb_minute_rows_rawdata_not_text")
    text = raw.astype("string")
    present = text.notna()
    well_formed = present & text.str.len().eq(_HEX_LENGTH) & text.str.startswith("0x")
    receipt = {
        "hour_rows": int(len(frame)),
        "hour_rows_with_rawdata": int(well_formed.sum()),
        "hour_rows_malformed_rawdata": int((present & ~well_formed).sum()),
        "minute_values": 0,
        "measured_zero_values": 0,
        "unplaceable_zero_values": 0,
        "non_finite_values": 0,
    }
    keep = frame.loc[~well_formed].drop(columns=list(SICDB_MINUTE_COLUMNS))
    expanded = frame.loc[well_formed]
    if expanded.empty:
        keep.attrs[SICDB_RECEIPT_ATTR] = receipt
        return keep.reset_index(drop=True)

    try:
        payload = bytes.fromhex("".join(s[2:] for s in text[well_formed].tolist()))
    except ValueError as exc:
        raise ValueError("sicdb_minute_rows_rawdata_not_hex") from exc
    values = np.frombuffer(payload, dtype="<f4").reshape(len(expanded), _SLOTS).astype("float64")
    finite = np.isfinite(values)
    nonzero = finite & (values != 0.0)
    count = pd.to_numeric(expanded["cnt"], errors="coerce").to_numpy(dtype="float64")
    complete = (count == _SLOTS)[:, None]
    zero = finite & (values == 0.0)
    measured_zero = zero & complete
    valued = nonzero | measured_zero
    nonzero_per_row = nonzero.sum(axis=1)
    unplaceable = np.where(
        ~complete[:, 0] & np.isfinite(count),
        np.clip(count - nonzero_per_row, 0, None),
        0,
    )
    receipt.update(
        minute_values=int(valued.sum()),
        measured_zero_values=int(measured_zero.sum()),
        unplaceable_zero_values=int(unplaceable.sum()),
        non_finite_values=int((~finite).sum()),
    )

    row_index, slot = np.nonzero(valued)
    minutes = expanded.drop(columns=list(SICDB_MINUTE_COLUMNS) + [value_column]).iloc[row_index]
    minutes = minutes.reset_index(drop=True)
    offset = pd.to_numeric(expanded[time_column], errors="coerce").to_numpy(dtype="float64")[row_index]
    time_dtype = frame[time_column].dtype
    minutes[time_column] = pd.Series(offset + 60.0 * slot).astype(
        time_dtype if pd.api.types.is_numeric_dtype(time_dtype) else "float64"
    )
    minutes[value_column] = values[row_index, slot]
    out = pd.concat([keep, minutes[keep.columns]], ignore_index=True)
    out.attrs[SICDB_RECEIPT_ATTR] = receipt
    return out


def resolution_summary(parquet_path, *, id_column: str, concepts, time_column: str = "charttime",
                       memory_mb: int = 1024) -> dict:
    """Per-signal inter-sample gaps in minutes of a published module; aggregates only.

    The module's ``charttime`` is in ICU-relative hours. Counts are rounded
    down to tens and counts below ten are reported as ``"<10"``; the summary
    never carries a stay identifier or a value. Quantiles need ten gaps.
    """
    from ..runtime import duckdb_spill

    def quote_identifier(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def rounded(n):
        n = int(n or 0)
        return "<10" if n < 10 else n // 10 * 10

    summary = {}
    with duckdb_spill.connection(
        database=":memory:",
        config={"memory_limit": f"{int(memory_mb)}MB", "threads": "1"},
    ) as connection:
        columns = {row[0] for row in connection.execute(
            "SELECT name FROM parquet_schema(?)", [str(parquet_path)]
        ).fetchall()}
        ident, time = quote_identifier(id_column), quote_identifier(time_column)
        for concept in concepts:
            if concept not in columns:
                continue
            value = quote_identifier(concept)
            rows, stays, gaps, q25, q50, q75, le5, le15 = connection.execute(
                f"""
                WITH present AS (
                    SELECT {ident} AS id, {time} * 60.0 AS t
                    FROM read_parquet(?)
                    WHERE {value} IS NOT NULL AND {time} IS NOT NULL
                ), gaps AS (
                    SELECT t - lag(t) OVER (PARTITION BY id ORDER BY t) AS gap FROM present
                )
                SELECT
                    (SELECT count(*) FROM present),
                    (SELECT count(DISTINCT id) FROM present),
                    count(gap),
                    quantile_cont(gap, 0.25), quantile_cont(gap, 0.5), quantile_cont(gap, 0.75),
                    avg(CASE WHEN gap <= 5 THEN 1.0 ELSE 0.0 END) FILTER (WHERE gap IS NOT NULL),
                    avg(CASE WHEN gap <= 15 THEN 1.0 ELSE 0.0 END) FILTER (WHERE gap IS NOT NULL)
                FROM gaps
                """,
                [str(parquet_path)],
            ).fetchone()
            entry = {"rows": rounded(rows), "stays": rounded(stays), "gaps": rounded(gaps)}
            if int(gaps or 0) >= 10:
                entry.update(
                    gap_minutes_p25=round(float(q25), 2),
                    gap_minutes_median=round(float(q50), 2),
                    gap_minutes_p75=round(float(q75), 2),
                    share_gaps_le_5_min=round(float(le5), 4),
                    share_gaps_le_15_min=round(float(le15), 4),
                )
            summary[concept] = entry
    return summary
