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

Every bounded reader calls :func:`record_read_counts`. The record goes to the
innermost active :func:`bounds_scope` of this process (a reader outside any
scope records nothing). Extraction opens one scope per committed unit of work
and stores its totals with that unit's existing record: a streamed batch in
its ledger entry (an isolated batch process hands its totals back beside its
part file), a whole-module load in the module manifest. A failed or killed
attempt commits nothing, and a resumed run sums the ledger's committed
batches, so the totals do not depend on batch size, retries or resume.

Within one scope a read is keyed by database, concept, source table and the
digest of the stays it read, so the same read repeated in a batch counts once.
Counts only: no identifier or value is recorded.
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Mapping, Optional

__all__ = [
    "stay_set_digest",
    "record_bounds_drops",
    "record_read_counts",
    "bounds_scope",
    "scope_active",
    "bounds_totals",
    "merge_bounds_drops",
    "collect_bounds_drops",
]

_FIELDS = ("dropped_values", "stays_all_out_of_bounds", "stays_read")
_SCOPES: List[list] = []
_LOCK = threading.Lock()


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
    """Add one read's counts to the innermost active scope; a no-op without one."""
    with _LOCK:
        if not _SCOPES:
            return
        _SCOPES[-1].append({
            "database": str(database),
            "concept": str(concept),
            "table": str(table),
            "read": str(read),
            "dropped_values": int(dropped_values),
            "stays_all_out_of_bounds": int(stays_all_out_of_bounds),
            "stays_read": int(stays_read),
        })


def scope_active() -> bool:
    """Whether a reader's counts would be collected (skip extra count queries otherwise)."""
    return bool(_SCOPES)


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
    if not scope_active() or frame is None:
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


@contextmanager
def bounds_scope() -> Iterator[list]:
    """Collect the records of the reads in this block (nested scopes take their own)."""
    records: list = []
    with _LOCK:
        _SCOPES.append(records)
    try:
        yield records
    finally:
        with _LOCK:
            for position in range(len(_SCOPES) - 1, -1, -1):
                if _SCOPES[position] is records:
                    del _SCOPES[position]
                    break


def bounds_totals(records: Iterable[Mapping]) -> dict:
    """Totals ``{concept: {table: counts}}`` of one scope's records, each read once."""
    totals: dict = {}
    seen: set = set()
    for entry in records:
        key = (entry["database"], entry["concept"], entry["table"], entry["read"])
        if key in seen:
            continue
        seen.add(key)
        counts = totals.setdefault(entry["concept"], {}).setdefault(
            entry["table"], dict.fromkeys(_FIELDS, 0)
        )
        for field in _FIELDS:
            counts[field] += int(entry[field])
    return merge_bounds_drops(totals)


def merge_bounds_drops(*parts: Optional[Mapping]) -> dict:
    """Sum ``{concept: {table: counts}}`` mappings (sorted keys)."""
    merged: dict = {}
    for part in parts:
        for concept, tables in (part or {}).items():
            for table, counts in (tables or {}).items():
                target = merged.setdefault(concept, {}).setdefault(table, dict.fromkeys(_FIELDS, 0))
                for field in _FIELDS:
                    target[field] += int(counts.get(field, 0))
    return {
        concept: {table: dict(merged[concept][table]) for table in sorted(merged[concept])}
        for concept in sorted(merged)
    }


@contextmanager
def collect_bounds_drops(directory=None) -> Iterator[dict]:
    """A scope whose totals fill the yielded dict when the block ends (for callers and tests)."""
    totals: dict = {}
    with bounds_scope() as records:
        try:
            yield totals
        finally:
            totals.update(bounds_totals(records))
