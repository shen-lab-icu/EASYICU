"""eICU I/O identity and systemic-intake semantics (audit R5-006/015)."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_RAW = "_eicu_io_raw_rows"
_KEPT = "_eicu_io_kept_rows"


def io_relation_sql(connection, read_sql: str, where_sql: str = "") -> str:
    """Group by every raw non-key field inside DuckDB, before any projection.

    The hash aggregate can spill; Python sees only selected rows or hourly
    aggregates. Predicate pushdown is safe for raw-field selectors. The count
    travels with each group, so audits require no second scan of the table.
    Hive bucket IDs describe storage, not clinical record identity.
    """
    from ..datasource import _duckdb_identifier

    names = [row[0] for row in connection.execute(f"DESCRIBE SELECT * FROM {read_sql}").fetchall()]
    keys = [_duckdb_identifier(c) for c in names if c not in {"intakeoutputid", "bucket_id"}]
    identity = ", ".join(keys)
    primary = ', MIN("intakeoutputid") AS "intakeoutputid"' if "intakeoutputid" in names else ""
    return f"(SELECT {identity}{primary}, COUNT(*) AS {_RAW} FROM {read_sql} {where_sql} GROUP BY ALL)"


def io_audit_select() -> str:
    """Aggregate audit counts alongside hourly values without returning raw rows."""
    return f", SUM({_RAW}) AS {_RAW}, COUNT(*) AS {_KEPT}"


def finish_io_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if _RAW not in frame:
        return frame
    rows = int(frame.pop(_RAW).sum())
    kept = int(frame.pop(_KEPT).sum()) if _KEPT in frame else len(frame)
    audit = {"rows": rows, "duplicates_removed": rows - kept, "rows_kept": kept}
    frame.attrs["eicu_io_dedup_audit"] = audit
    logger.info("eICU raw I/O deduplication: %s", audit)
    return frame


def read_io_parquet(path, columns, patient_filter, item_filter, *, files=None):
    """Stay/selector scoped parquet scan with bounded DuckDB memory and spill."""
    from ..datasource import (
        FilterOp, _duckdb_identifier, _duckdb_read_parquet_clause,
        _duckdb_sql_literal, _get_duckdb_connection,
    )

    from .. import storage_layout

    connection = _get_duckdb_connection()
    # A disk reader supplies the owner's validated set. Do not discover a
    # second inventory with different suffix or partition rules.
    read = (storage_layout.parquet_relation(files) if files is not None else
            _duckdb_read_parquet_clause(path, hive_partitioning=False) if path.is_dir()
            else f"read_parquet({_duckdb_sql_literal(str(path))})")
    predicates = []
    if patient_filter is not None and patient_filter.op == FilterOp.IN:
        values = ", ".join(_duckdb_sql_literal(v) for v in patient_filter._value_set)
        predicates.append(f"{_duckdb_identifier(patient_filter.column)} IN ({values})" if values else "FALSE")
    if item_filter is not None:
        column, ids = item_filter
        values = ", ".join(_duckdb_sql_literal(v) for v in ids)
        predicates.append(f"{_duckdb_identifier(column)} IN ({values})" if values else "FALSE")
    where = "WHERE " + " AND ".join(predicates) if predicates else ""
    relation = io_relation_sql(connection, read, where)
    selected = ", ".join(_duckdb_identifier(c) for c in columns) + f", {_RAW}" if columns else "*"
    return finish_io_audit(connection.execute(f"SELECT {selected} FROM {relation}").fetchdf())


def deduplicate_io(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate raw records on every field except the physical row key.

    Must run before projection: equal charted values with different entry times,
    channels, totals or text are independent records. Audit counts contain no IDs.
    """
    if "eicu_io_dedup_audit" in frame.attrs:
        return frame
    keys = [c for c in frame.columns if c not in {"intakeoutputid", "bucket_id"}]
    duplicate = frame.duplicated(subset=keys) if keys else pd.Series(False, index=frame.index)
    out = frame.loc[~duplicate].copy()
    audit = {"rows": len(frame), "duplicates_removed": int(duplicate.sum()), "rows_kept": len(out)}
    out.attrs["eicu_io_dedup_audit"] = audit
    logger.info("eICU raw I/O deduplication: %s", audit)
    return out


_NONSYSTEMIC = re.compile(
    r"24\s*(?:h|hrs?|hours?)\b|\b(?:daily|cumulative|since admission)\b|"
    r"\b(?:cbi|irrig\w*|bladder)\b|amount\s*\(mc?g\)", re.I,
)
_RATE = re.compile(r"\bml\s*(?:/|per\s*)\s*(?:hr|hour|h)\b", re.I)


def systemic_intake(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Keep volumes; integrate rates only to a subsequent same-channel record.

    The next record must be within six hours. A final rate has no documented
    duration and contributes no guessed volume. Zero terminates the interval.
    Benign labels containing 'Total' remain eligible; total/component duplication
    cannot be inferred from that word (R5-016).
    """
    labels = frame["cellpath"].astype("string")
    bad = labels.str.contains(_NONSYSTEMIC, na=False)
    rates = labels.str.contains(_RATE, na=False) & ~bad
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.between(0, 10000)
    out = frame.loc[~bad & ~rates & valid].copy()
    out[column] = values.loc[out.index]
    work = frame.loc[rates].copy()
    work[column] = values.loc[work.index]
    pieces = work.iloc[:0].copy()
    if not work.empty:
        # Total order: same-time records of one channel resolve the same way
        # whatever order the batch or thread layout delivered them in.
        work = work.sort_values(
            ["patientunitstayid", "cellpath", "intakeoutputoffset", column], kind="mergesort"
        )
        work["_next"] = work.groupby(["patientunitstayid", "cellpath"])["intakeoutputoffset"].shift(-1)
        duration = work._next - work.intakeoutputoffset
        work = work.loc[duration.gt(0) & duration.le(360) & work[column].gt(0) & work[column].le(10000)]
        start = work.intakeoutputoffset.to_numpy(dtype=float)
        end = work._next.to_numpy(dtype=float)
        first = np.floor(start / 60).astype(np.int64)
        count = np.ceil(end / 60).astype(np.int64) - first
        positions = np.repeat(np.arange(len(work)), count)
        offsets = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        hour = first[positions] + offsets
        overlap = np.minimum(end[positions], (hour + 1) * 60) - np.maximum(start[positions], hour * 60)
        pieces = work.iloc[positions].drop(columns="_next").copy()
        pieces["intakeoutputoffset"] = hour * 60
        pieces[column] = work[column].to_numpy()[positions] * overlap / 60
    if not pieces.empty:
        out = pd.concat([out, pieces], ignore_index=True)
    out.attrs.update(frame.attrs)
    out.attrs["eicu_intake_audit"] = {
        "nonsystemic_or_cumulative_dropped": int(bad.sum()),
        "rate_rows": int(rates.sum()), "integrated_pieces": len(pieces),
        "volume_rows": int((~bad & ~rates & valid).sum()),
    }
    return out
