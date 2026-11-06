"""Diagnostic switch that routes concept loads around DuckDB aggregation.

Every DuckDB aggregation entry point (single-concept bucketed SQL, the wide
single-column fast path, and the wide/long batch loaders) must publish the
same rows, values and dtypes as the pandas ``load_table`` path for the same
concept.  ``tools/duckdb_raw_parity_census.py`` and the parity tests verify
that by loading a concept once normally and once inside
:func:`force_raw_aggregation`.  The switch is process-wide (concept loads may
run on worker threads) and is not a production configuration.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, List

__all__ = [
    "duckdb_aggregation_enabled",
    "force_raw_aggregation",
    "note_duckdb_route",
    "record_duckdb_routes",
]

_lock = Lock()
_depth = 0
_recorders: List[Counter] = []


def duckdb_aggregation_enabled() -> bool:
    """False while a :func:`force_raw_aggregation` block is active."""

    return _depth == 0


@contextmanager
def force_raw_aggregation() -> Iterator[None]:
    """Load every concept through the pandas path inside this block."""

    global _depth
    with _lock:
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1


def note_duckdb_route(route: str) -> None:
    """Record that a concept load took the named DuckDB aggregation route."""

    with _lock:
        for recorder in _recorders:
            recorder[route] += 1


@contextmanager
def record_duckdb_routes() -> Iterator[Counter]:
    """Count the DuckDB aggregation routes taken inside this block."""

    recorder: Counter = Counter()
    with _lock:
        _recorders.append(recorder)
    try:
        yield recorder
    finally:
        with _lock:
            _recorders.remove(recorder)
