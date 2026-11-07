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
refused at publication (``ConceptValueKindUndeclared``): the refusal checks
every module's counts first, before any aggregation.

A coercion record also carries a PRIVATE per-stay ledger (``_stays``: the
stays with values and each stay's non-numeric count). It travels with the
in-process totals and the temporary worker manifests only, so a source read
by several modules is merged as a union keyed by stay identity
(:func:`coercion_union`). Published manifests carry :func:`public_bounds`
only: counts, with 1-9 written ``"<10"`` (:func:`small_count`), never the
ledger. A published count read back (republication) has no ledger: a source
read by several modules then has a lower bound and ``audit_unknown``.

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
    "coercion_union",
    "public_bounds",
    "small_count",
    "count_value",
    "AUDIT_UNKNOWN",
    "bounds_scope",
    "scope_active",
    "bounds_totals",
    "merge_bounds_drops",
    "collect_bounds_drops",
]

_BOUNDS_FIELDS = ("dropped_values", "stays_all_out_of_bounds", "stays_read")
_COERCION_FIELDS = ("non_numeric_values", "stays_with_non_numeric", "stays_with_values")
_FIELDS = _BOUNDS_FIELDS + _COERCION_FIELDS
# Private per-stay ledger of a coercion record (never published).
_LEDGER = "_stays"
AUDIT_UNKNOWN = "audit_unknown"
_SMALL = "<10"


def small_count(value) -> object:
    """A user-visible count: 1-9 is written ``"<10"`` (0 and >= 10 as they are)."""
    value = count_value(value)
    return _SMALL if 0 < value < 10 else value


def count_value(value) -> int:
    """An int from a count that may be published (``"<10"`` reads as 1, a lower bound)."""
    if value == _SMALL:
        return 1
    return int(value or 0)


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
    identified: bool = True,
) -> None:
    """Record one conversion point's non-numeric counts for one read.

    ``frame`` has an id column and two count columns at any grain: the
    non-null source values and those of them that did not convert to a
    number. They are summed per stay; the per-stay counts form the record's
    private ledger (``identified=False``: the id column is not a stay
    identity, so the record has no ledger). A no-op outside any scope.
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
        ledger = None
        if identified and len(frame) and id_column in frame.columns:
            ledger = {
                "seen": sorted(str(stay) for stay in values.index[values.to_numpy() > 0]),
                "bad": {
                    str(stay): int(count)
                    for stay, count in non_numeric[non_numeric > 0].items()
                },
            }
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
                _LEDGER: ledger,
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
            identified=ids is not None,
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
        if kind != "bounds":
            had_coercion = any(field in counts for field in _COERCION_FIELDS)
            ledger = entry.get(_LEDGER)
            if ledger is None or (had_coercion and _LEDGER not in counts):
                counts.pop(_LEDGER, None)
                counts["_no_ledger"] = True
            elif not counts.get("_no_ledger"):
                counts[_LEDGER] = _merge_ledgers(counts.get(_LEDGER), ledger)
        for field in _BOUNDS_FIELDS if kind == "bounds" else _COERCION_FIELDS:
            counts[field] = counts.get(field, 0) + int(entry.get(field, 0))
    for tables in totals.values():
        for counts in tables.values():
            counts.pop("_no_ledger", None)
    return _merge_bounds_drops(totals)


def _merge_ledgers(*ledgers: Optional[Mapping]) -> dict:
    """Union of per-stay ledgers: stays with values, and each stay's largest count.

    Two reads of the same stay and source read the same source events, so a
    stay's count is the larger of its records, never their sum.
    """
    seen: set = set()
    bad: dict = {}
    for ledger in ledgers:
        if not ledger:
            continue
        seen.update(ledger.get("seen") or ())
        for stay, count in (ledger.get("bad") or {}).items():
            bad[stay] = max(int(count), bad.get(stay, 0))
    return {"seen": sorted(seen), "bad": {stay: bad[stay] for stay in sorted(bad)}}


def merge_bounds_drops(*parts: Optional[Mapping]) -> dict:
    """Sum ``{concept: {table: counts}}`` mappings (sorted keys)."""
    with accounting("merge"):
        return _merge_bounds_drops(*parts)


def _merge_bounds_drops(*parts: Optional[Mapping]) -> dict:
    merged: dict = {}
    unledgered: set = set()
    for part in parts:
        for concept, tables in (part or {}).items():
            for table, counts in (tables or {}).items():
                counts = counts or {}
                target = merged.setdefault(concept, {}).setdefault(table, dict.fromkeys(_BOUNDS_FIELDS, 0))
                if any(field in counts for field in _COERCION_FIELDS):
                    # A part without a ledger makes the merged ledger unknown.
                    if _LEDGER not in counts or (concept, table) in unledgered:
                        unledgered.add((concept, table))
                        target.pop(_LEDGER, None)
                    else:
                        target[_LEDGER] = _merge_ledgers(target.get(_LEDGER), counts[_LEDGER])
                for field in _FIELDS:
                    if field in _BOUNDS_FIELDS or field in counts:
                        target[field] = target.get(field, 0) + count_value(counts.get(field, 0))
    return {
        concept: {table: dict(merged[concept][table]) for table in sorted(merged[concept])}
        for concept in sorted(merged)
    }


def public_bounds(totals: Optional[Mapping]) -> dict:
    """The published form of ``{concept: {table: counts}}``: no ledger, 1-9 as ``"<10"``."""
    out: dict = {}
    for concept, tables in sorted((totals or {}).items()):
        for table, counts in sorted((tables or {}).items()):
            out.setdefault(concept, {})[table] = {
                field: (small_count(value) if field in _COERCION_FIELDS else value)
                for field, value in (counts or {}).items()
                if field != _LEDGER
            }
    return out


def coercion_union(drops_by_module: Optional[Mapping]) -> dict:
    """Run-level non-numeric counts ``{concept: {table: counts}}`` across modules.

    A source read by several modules is merged as the union of their ledgers,
    keyed by stay (a stay's count is the largest of its records): disjoint
    readers add up, a reader that saw more stays cannot hide one that saw
    the anomalies. ``union`` says how exact the row is: ``exact`` (every
    reader had a ledger), ``single_module`` (one reader's published counts)
    or ``audit_unknown`` (several readers without ledgers: each field is the
    largest reader's, a lower bound). Integers; :func:`public_bounds` and
    :func:`small_count` format them for publication.
    """
    readers: dict = {}
    for module in sorted(drops_by_module or {}):
        for concept, tables in ((drops_by_module or {})[module] or {}).items():
            for table, counts in (tables or {}).items():
                counts = counts or {}
                if any(field in counts for field in _COERCION_FIELDS):
                    readers.setdefault((concept, table), []).append((module, counts))
    out: dict = {}
    for (concept, table), entries in sorted(readers.items()):
        modules = [module for module, _counts in entries]
        if all(isinstance(counts.get(_LEDGER), Mapping) for _m, counts in entries):
            ledger = _merge_ledgers(*(counts[_LEDGER] for _m, counts in entries))
            row = {
                "non_numeric_values": sum(ledger["bad"].values()),
                "stays_with_non_numeric": len(ledger["bad"]),
                "stays_with_values": len(set(ledger["seen"]) | set(ledger["bad"])),
                "union": "exact",
            }
        elif len(entries) == 1:
            counts = entries[0][1]
            row = {field: count_value(counts.get(field, 0)) for field in _COERCION_FIELDS}
            row["union"] = "single_module"
        else:
            row = {
                field: max(count_value(counts.get(field, 0)) for _m, counts in entries)
                for field in _COERCION_FIELDS
            }
            row["union"] = AUDIT_UNKNOWN
        row["modules"] = modules
        out.setdefault(concept, {})[table] = row
    return out


def non_numeric_by_concept(*parts: Optional[Mapping]) -> dict:
    """``{concept: non_numeric_values}`` of the union of ``parts`` (see :func:`coercion_union`)."""
    union = coercion_union({f"part_{index:06d}": part for index, part in enumerate(parts)})
    out: dict = {}
    for concept, tables in union.items():
        out[concept] = sum(row["non_numeric_values"] for row in tables.values())
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
