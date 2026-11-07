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

The same owner counts a declared numeric (or logical) concept's source values
that are not numbers, at every point that converts them (the raw resolver,
the single/multi/wide SQL aggregations, the module export). A non-null
original that becomes NaN/NULL is counted, never converted silently
(B-IMPLICIT-GATE):

``non_numeric_values``
    non-null source values that became NaN/NULL;
``stays_with_non_numeric``
    stays with at least one such value;
``stays_with_values``
    stays with at least one non-null source value at that conversion point.

A concept without a class and without bounds that has any such value is
refused at publication (``ConceptValueKindUndeclared``).

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
    "SourceBoundsAuditError",
    "accounting",
    "stay_set_digest",
    "record_bounds_drops",
    "record_read_counts",
    "record_coercion_counts",
    "record_series_coercion",
    "non_numeric_by_concept",
    "bounds_scope",
    "scope_active",
    "bounds_totals",
    "merge_bounds_drops",
    "collect_bounds_drops",
]

_BOUNDS_FIELDS = ("dropped_values", "stays_all_out_of_bounds", "stays_read")
_COERCION_FIELDS = ("non_numeric_values", "stays_with_non_numeric", "stays_with_values")
_FIELDS = _BOUNDS_FIELDS + _COERCION_FIELDS


class SourceBoundsAuditError(RuntimeError):
    """Counting or committing the bounds counts failed.

    Typed and in ``TYPED_VALUE_FAILURES``: no source-unavailable handler or
    fallback may absorb it, since a failure of the accounting must never
    read as an absent source (Codex delta7x B-AUDIT-CATCH). ``stage`` names
    where it failed; the cause is chained.
    """

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(f"source_bounds_audit_failed at {stage}: {detail}")


@contextmanager
def accounting(stage: str) -> Iterator[None]:
    """Turn any failure of the block into ``SourceBoundsAuditError`` (cause chained)."""
    try:
        yield
    except SourceBoundsAuditError:
        raise
    except Exception as exc:
        raise SourceBoundsAuditError(stage, f"{type(exc).__name__}: {exc}") from exc
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
    with accounting("record"), _LOCK:
        if not _SCOPES:
            return
        _SCOPES[-1].append({
            "kind": "bounds",
            "database": str(database),
            "concept": str(concept),
            "table": str(table),
            "read": str(read),
            "dropped_values": int(dropped_values),
            "stays_all_out_of_bounds": int(stays_all_out_of_bounds),
            "stays_read": int(stays_read),
        })


def record_coercion_counts(
    *,
    database: str,
    concept: str,
    table: str,
    stays: Optional[Iterable],
    frame,
    id_column: str,
    values_column: str,
    non_numeric_column: str,
) -> None:
    """Record one conversion point's non-numeric counts for one read.

    ``frame`` has an id column and two count columns at any grain: the
    non-null source values and those of them that did not convert to a
    number. They are summed per stay. A no-op outside any scope.
    """
    if not scope_active() or frame is None:
        return
    with accounting(f"coercion[{database}.{table}:{concept}]"):
        import pandas as pd

        if len(frame) == 0 or id_column not in frame.columns:
            values = non_numeric = pd.Series([], dtype="int64")
        else:
            per_stay = (
                frame[[id_column, values_column, non_numeric_column]]
                .assign(**{
                    values_column: pd.to_numeric(frame[values_column], errors="coerce").fillna(0),
                    non_numeric_column: pd.to_numeric(frame[non_numeric_column], errors="coerce").fillna(0),
                })
                .groupby(id_column, sort=False, dropna=False)[[values_column, non_numeric_column]]
                .sum()
            )
            values = per_stay[values_column].astype("int64")
            non_numeric = per_stay[non_numeric_column].astype("int64")
        with _LOCK:
            if not _SCOPES:
                return
            _SCOPES[-1].append({
                "kind": "coercion",
                "database": str(database),
                "concept": str(concept),
                "table": str(table),
                "read": stay_set_digest(stays),
                "non_numeric_values": int(non_numeric.sum()),
                "stays_with_non_numeric": int((non_numeric > 0).sum()),
                "stays_with_values": int((values > 0).sum()),
            })


def record_series_coercion(
    *, database: str, concept: str, table: str, stays: Optional[Iterable], original, converted, ids
) -> int:
    """Count the non-null ``original`` cells that are null in ``converted``; record them.

    ``ids`` gives each cell's stay (aligned with ``original``). Returns the
    number of non-numeric values (also outside any scope).
    """
    import pandas as pd

    present = original.notna().to_numpy()
    lost = present & converted.isna().to_numpy()
    count = int(lost.sum())
    if scope_active():
        record_coercion_counts(
            database=database, concept=concept, table=table, stays=stays,
            frame=pd.DataFrame({
                "stay": pd.Series(ids).to_numpy() if ids is not None else 0,
                "values": present.astype("int64"),
                "non_numeric": lost.astype("int64"),
            }),
            id_column="stay", values_column="values", non_numeric_column="non_numeric",
        )
    return count


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
    with accounting(f"count[{database}.{table}:{concept}]"):
        _record_read_counts(
            database=database, concept=concept, table=table, stays=stays, frame=frame,
            id_column=id_column, raw_column=raw_column, bounded_column=bounded_column,
        )


def _record_read_counts(*, database, concept, table, stays, frame, id_column, raw_column, bounded_column):
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
    with accounting("totals"):
        return _bounds_totals(records)


def _bounds_totals(records: Iterable[Mapping]) -> dict:
    totals: dict = {}
    seen: set = set()
    for entry in records:
        kind = entry.get("kind", "bounds")
        key = (kind, entry["database"], entry["concept"], entry["table"], entry["read"])
        if key in seen:
            continue
        seen.add(key)
        counts = totals.setdefault(entry["concept"], {}).setdefault(
            entry["table"], dict.fromkeys(_BOUNDS_FIELDS, 0)
        )
        # Coercion fields appear where a conversion point counted (sparse).
        for field in _BOUNDS_FIELDS if kind == "bounds" else _COERCION_FIELDS:
            counts[field] = counts.get(field, 0) + int(entry.get(field, 0))
    return _merge_bounds_drops(totals)


def merge_bounds_drops(*parts: Optional[Mapping]) -> dict:
    """Sum ``{concept: {table: counts}}`` mappings (sorted keys)."""
    with accounting("merge"):
        return _merge_bounds_drops(*parts)


def _merge_bounds_drops(*parts: Optional[Mapping]) -> dict:
    merged: dict = {}
    for part in parts:
        for concept, tables in (part or {}).items():
            for table, counts in (tables or {}).items():
                target = merged.setdefault(concept, {}).setdefault(table, dict.fromkeys(_BOUNDS_FIELDS, 0))
                for field in _FIELDS:
                    if field in _BOUNDS_FIELDS or field in (counts or {}):
                        target[field] = target.get(field, 0) + int(counts.get(field, 0))
    return {
        concept: {table: dict(merged[concept][table]) for table in sorted(merged[concept])}
        for concept in sorted(merged)
    }


def non_numeric_by_concept(*parts: Optional[Mapping]) -> dict:
    """``{concept: non_numeric_values}``: unique per concept and source, summed over sources.

    ``parts`` are ``{concept: {table: counts}}`` totals of the modules of one
    run. A source read by several modules is the same read of the same
    stays, so it is counted once: the module that read the most stays
    (as :func:`easyicu.io.data_quality.source_value_coercion_warnings`
    reports it), never the sum across the modules that read it.
    """
    best: dict = {}
    for part in parts:
        for concept, tables in (part or {}).items():
            for table, counts in (tables or {}).items():
                counts = counts or {}
                rank = (int(counts.get("stays_with_values", 0)), int(counts.get("non_numeric_values", 0)))
                if (concept, table) not in best or rank > best[(concept, table)]:
                    best[(concept, table)] = rank
    out: dict = {}
    for (concept, _table), (_stays, non_numeric) in best.items():
        out[concept] = out.get(concept, 0) + non_numeric
    return out


@contextmanager
def collect_bounds_drops(directory=None) -> Iterator[dict]:
    """A scope whose totals fill the yielded dict when the block ends (for callers and tests)."""
    totals: dict = {}
    with bounds_scope() as records:
        try:
            yield totals
        finally:
            totals.update(bounds_totals(records))
