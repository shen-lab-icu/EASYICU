"""Counts of source values dropped by declared concept bounds.

A value outside a concept's declared ``[min, max]`` is dropped per row, as
ricu's ``filter_bounds`` does, whatever else shares its batch. Nothing is
refused and no unit is guessed from the values. What was dropped is recorded
instead, per concept and source table:

``dropped_values``
    source values outside the bounds;
``stays_all_out_of_bounds``
    stays with at least one value, none of them within the bounds;
``stays_read``
    stays with at least one value.

Each read records its own counts. A read is identified by its database,
concept, source table and the digest of the stays it read, so a read repeated
for another module or a retried batch is counted once. Batches partition the
stays, so the totals are the same for any batch size.

Readers in this process and in extraction workers append to the JSON-lines log
named by ``EASYICU_BOUNDS_DROP_LOG`` (set by ``extract_database``, like the
storage-fallback log). The module manifests and the release receipt publish the
totals; ``io.data_quality.source_bounds_warnings`` turns them into the release
QC warning table. Counts only: no identifier or value is recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Iterable, Iterator, Mapping, Optional

__all__ = [
    "BOUNDS_DROP_LOG_ENV",
    "stay_set_digest",
    "record_bounds_drops",
    "record_read_counts",
    "read_bounds_drop_log",
    "merge_bounds_drops",
    "collect_bounds_drops",
]

BOUNDS_DROP_LOG_ENV = "EASYICU_BOUNDS_DROP_LOG"


def stay_set_digest(stays: Optional[Iterable]) -> str:
    """Order-independent digest of the stays one read covered (``all`` if unfiltered)."""
    if stays is None:
        return "all"
    keys = sorted({str(stay) for stay in stays})
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:32]


def record_bounds_drops(
    *,
    database: str,
    concept: str,
    table: str,
    read: str,
    dropped_values: int,
    stays_all_out_of_bounds: int,
    stays_read: int,
) -> None:
    """Append one read's counts to the active log; a no-op when none is active."""
    path = os.environ.get(BOUNDS_DROP_LOG_ENV)
    if not path:
        return
    line = json.dumps({
        "database": str(database),
        "concept": str(concept),
        "table": str(table),
        "read": str(read),
        "dropped_values": int(dropped_values),
        "stays_all_out_of_bounds": int(stays_all_out_of_bounds),
        "stays_read": int(stays_read),
    }, sort_keys=True)
    # One short line per read, appended: concurrent workers do not interleave.
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def record_read_counts(
    *,
    database: str,
    concept: str,
    table: str,
    stays: Optional[Iterable],
    frame,
    id_column: str,
    raw_column: str,
    bounded_column: str,
) -> None:
    """Record one read from per-row counts of values and of values within bounds.

    ``frame`` has an id column and two count columns at any grain (per stay,
    or per stay and time bucket); they are summed per stay. ``stays`` is the
    stay filter of the read (``None`` for an unfiltered read).
    """
    if not os.environ.get(BOUNDS_DROP_LOG_ENV) or frame is None:
        return
    import pandas as pd

    if len(frame) == 0 or id_column not in frame.columns:
        per_stay = pd.DataFrame({"raw": [], "bounded": []})
    else:
        per_stay = (
            frame[[id_column, raw_column, bounded_column]]
            .assign(**{
                raw_column: pd.to_numeric(frame[raw_column], errors="coerce").fillna(0),
                bounded_column: pd.to_numeric(frame[bounded_column], errors="coerce").fillna(0),
            })
            .groupby(id_column, sort=False, dropna=False)[[raw_column, bounded_column]]
            .sum()
            .rename(columns={raw_column: "raw", bounded_column: "bounded"})
        )
    raw = per_stay["raw"].astype("int64")
    bounded = per_stay["bounded"].astype("int64")
    record_bounds_drops(
        database=database,
        concept=concept,
        table=table,
        read=stay_set_digest(stays),
        dropped_values=int((raw - bounded).clip(lower=0).sum()),
        stays_all_out_of_bounds=int(((raw > 0) & (bounded == 0)).sum()),
        stays_read=int((raw > 0).sum()),
    )


def read_bounds_drop_log(path) -> dict:
    """Totals ``{concept: {table: counts}}`` from a log, each read counted once."""
    totals: dict = {}
    seen: set = set()
    try:
        handle = open(path, encoding="utf-8")
    except FileNotFoundError:
        return totals
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry["database"], entry["concept"], entry["table"], entry["read"])
            if key in seen:
                continue
            seen.add(key)
            counts = totals.setdefault(entry["concept"], {}).setdefault(
                entry["table"],
                {"dropped_values": 0, "stays_all_out_of_bounds": 0, "stays_read": 0},
            )
            for field in counts:
                counts[field] += int(entry[field])
    return {
        concept: {table: dict(tables[table]) for table in sorted(tables)}
        for concept, tables in sorted(totals.items())
    }


def merge_bounds_drops(*parts: Mapping) -> dict:
    """Sum ``{concept: {table: counts}}`` mappings."""
    merged: dict = {}
    for part in parts:
        for concept, tables in (part or {}).items():
            for table, counts in tables.items():
                target = merged.setdefault(concept, {}).setdefault(
                    table, {"dropped_values": 0, "stays_all_out_of_bounds": 0, "stays_read": 0}
                )
                for field in target:
                    target[field] += int(counts.get(field, 0))
    return merged


@contextmanager
def collect_bounds_drops(directory=None) -> Iterator[dict]:
    """Activate a log for the block; the yielded dict holds the totals afterwards."""
    handle, path = tempfile.mkstemp(prefix="easyicu_bounds_drops_", suffix=".jsonl", dir=directory)
    os.close(handle)
    previous = os.environ.get(BOUNDS_DROP_LOG_ENV)
    os.environ[BOUNDS_DROP_LOG_ENV] = path
    totals: dict = {}
    try:
        yield totals
    finally:
        if previous is None:
            os.environ.pop(BOUNDS_DROP_LOG_ENV, None)
        else:
            os.environ[BOUNDS_DROP_LOG_ENV] = previous
        totals.update(read_bounds_drop_log(path))
        os.unlink(path)
