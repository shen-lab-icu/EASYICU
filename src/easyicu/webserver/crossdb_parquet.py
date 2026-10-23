"""Exact, resource-bounded feature aggregates; parquet rows stay in DuckDB.

The payload equals ``crossdb_review._summarize_feature_distribution`` on the
same column read by pandas, computed over every row. Boolean, integer and
double columns share a few whole-file scans: counts and range, exact 1st/99th
percentiles per column, then trimmed range and NumPy-equivalent histogram
bins. Other columns (text, float32, decimals, dates) keep the per-column
text-aware path. Nothing is sampled and nothing falls back to pandas.
"""

from __future__ import annotations

from easyicu.runtime import duckdb_spill
from collections import OrderedDict
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

import duckdb
import numpy as np

# Per job; a hosted server runs one summary per clinician at a time.
MEMORY_LIMIT = "1GiB"
THREADS = 4
TEMP_LIMIT = "4GiB"
POLL_SECONDS = 0.025
_CACHE_ENTRIES = 64

_FAST_TYPES = {"BOOLEAN", "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "DOUBLE"}
_INTEGER_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}

_cache: OrderedDict[tuple, list[dict[str, Any]]] = OrderedDict()
_cache_lock = threading.Lock()


class ParquetAggregationError(RuntimeError):
    """An exact aggregate failed, retaining the engine exception as its cause."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@contextmanager
def _connection(checkpoint: Callable[[], None] | None, deadline_seconds: float):
    deadline = time.monotonic() + deadline_seconds
    stopped = threading.Event()
    interrupted: list[Exception] = []

    def check() -> None:
        if interrupted:
            # The watcher saw it first; every later step stops as well.
            raise interrupted[0]
        if checkpoint:
            checkpoint()
        if time.monotonic() >= deadline:
            raise TimeoutError("exact parquet aggregation deadline exceeded")

    with duckdb_spill.connection(
        config={
            'threads': THREADS,
            'max_temp_directory_size': TEMP_LIMIT,
        },
    ) as conn:
        conn.execute("SET memory_limit = ?", [MEMORY_LIMIT])
        conn.execute("SET threads = ?", [THREADS])
        conn.execute("SET preserve_insertion_order = false")

        def watch() -> None:
            while not stopped.wait(POLL_SECONDS):
                if not interrupted:
                    try:
                        check()
                    except Exception as exc:
                        interrupted.append(exc)
                if interrupted:
                    # Repeat until closed: interruption can race a query start.
                    conn.interrupt()

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            check()
            yield conn, check
            check()
        except Exception as exc:
            cause = interrupted[0] if interrupted else exc
            if isinstance(cause, (TimeoutError, duckdb.OutOfMemoryException)):
                raise ParquetAggregationError(
                    "crossdb_feature_read_budget_exceeded"
                ) from cause
            if interrupted:
                raise cause from exc
            if isinstance(exc, duckdb.Error):
                raise ParquetAggregationError("crossdb_feature_read_failed") from exc
            raise
        finally:
            stopped.set()
            watcher.join()


def parquet_feature_distributions(
    path: Path,
    features: list[str],
    *,
    checkpoint: Callable[[], None] | None = None,
    deadline_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    """Return complete feature summaries, never a row sample or pandas fallback.

    A file's summaries are kept in memory, keyed by its path, size and
    modification time, so repeating a comparison does not rescan it.
    """
    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, tuple(features))
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return copy.deepcopy(_cache[key])
    payloads: dict[str, dict[str, Any]] = {}
    with _connection(checkpoint, deadline_seconds) as (conn, check):
        conn.read_parquet(str(path)).create_view("source")
        types = {row[0]: row[1] for row in conn.execute("DESCRIBE source").fetchall()}
        fast = [feature for feature in features if types[feature] in _FAST_TYPES]
        if fast:
            payloads.update(_summarize_typed(conn, fast, types, check))
        for feature in features:
            if feature not in payloads:
                check()
                payloads[feature] = _summarize(conn, feature, types[feature])
    result = [{"feature": feature, **payloads[feature]} for feature in features]
    with _cache_lock:
        _cache[key] = copy.deepcopy(result)
        while len(_cache) > _CACHE_ENTRIES:
            _cache.popitem(last=False)
    return result


def _quoted(feature: str) -> str:
    return '"' + feature.replace('"', '""') + '"'


def _typed_value(feature: str, dtype: str) -> str:
    """The value pandas keeps after ``dropna``: NaN is missing, booleans are 1/0."""
    column = _quoted(feature)
    if dtype == "DOUBLE":
        return f"CASE WHEN isnan({column}) THEN NULL ELSE {column} END"
    return f"CAST({column} AS DOUBLE)"


def _pandas_label(value: float, dtype: str, has_missing: bool) -> str:
    """``str`` of the single kept value as pandas reads the parquet column."""
    if dtype == "BOOLEAN":
        return "True" if value else "False"
    if dtype in _INTEGER_TYPES and not has_missing:
        return str(int(value))
    # Doubles, and integer columns with missing values (read as float64).
    return str(float(value))


def _summarize_typed(
    conn: Any, features: list[str], types: dict[str, str], check: Callable[[], None]
) -> dict[str, dict[str, Any]]:
    values = {feature: _typed_value(feature, types[feature]) for feature in features}
    selects = ["count(*)"]
    for feature in features:
        selects += [f"count({values[feature]})", f"min({values[feature]})", f"max({values[feature]})"]
    row = conn.execute(f"SELECT {', '.join(selects)} FROM source").fetchone()
    total = row[0]
    stats = {
        feature: tuple(row[1 + 3 * index:4 + 3 * index])
        for index, feature in enumerate(features)
    }
    out: dict[str, dict[str, Any]] = {}
    ranges: dict[str, tuple[float, float, int]] = {}
    for feature in features:
        n, low, high = stats[feature]
        if not n:
            out[feature] = {"kind": "empty", "n": total, "non_null": 0, "points": [], "categories": []}
        elif n == 1:
            # pandas: one kept value is under max(2, ...), so it is a category.
            label = _pandas_label(low, types[feature], total > n)
            out[feature] = {
                "kind": "categorical", "n": total, "non_null": 1, "points": [],
                "categories": [{"label": label, "count": 1, "pct": round(1 / total * 100, 1)}],
            }
        else:
            out[feature] = {
                "kind": "numeric", "n": total, "non_null": n,
                "min": round(low, 6), "max": round(high, 6), "categories": [],
            }
            if low == high:
                out[feature]["points"] = _triangle(low)
            else:
                ranges[feature] = (low, high, n)

    # Exact percentiles hold one column's values at a time.
    quantiles: dict[str, tuple[float, float]] = {}
    for feature, (_low, _high, n) in ranges.items():
        if n >= 10:
            check()
            q1, q99 = conn.execute(
                f"SELECT quantile_cont({values[feature]}, [0.01, 0.99]) FROM source"
            ).fetchone()[0]
            if not (math.isnan(q1) or math.isnan(q99)):
                # A NaN percentile keeps nothing in NumPy, so nothing is trimmed.
                quantiles[feature] = (q1, q99)
    if quantiles:
        check()
        selects, params = [], []
        for feature, (q1, q99) in quantiles.items():
            inside = f"{values[feature]} >= ? AND {values[feature]} <= ?"
            selects += [
                f"count(*) FILTER (WHERE {inside})",
                f"min({values[feature]}) FILTER (WHERE {inside})",
                f"max({values[feature]}) FILTER (WHERE {inside})",
            ]
            params += [q1, q99] * 3
        row = conn.execute(f"SELECT {', '.join(selects)} FROM source", params).fetchone()
        for index, feature in enumerate(quantiles):
            kept, trim_low, trim_high = row[3 * index:3 * index + 3]
            if kept >= 6 and trim_low != trim_high:
                low, high, n = ranges[feature]
                ranges[feature] = (trim_low, trim_high, n)

    histograms = _histograms(conn, values, ranges, check) if ranges else {}
    for feature in ranges:
        out[feature]["points"] = histograms[feature]
    return out


def _histograms(
    conn: Any,
    values: dict[str, str],
    ranges: dict[str, tuple[float, float, int]],
    check: Callable[[], None],
) -> dict[str, list[dict[str, float]]]:
    """NumPy's uniform-bin ``histogram`` (closed last bin) for each column, one scan."""
    selects, params, layout = [], [], {}
    for feature, (low, high, n) in ranges.items():
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f"autodetected range of [{low}, {high}] is not finite")
        bins = max(6, min(32, int(math.sqrt(n)) + 2))
        edges = np.linspace(low, high, bins + 1)
        layout[feature] = (bins, edges)
        v = values[feature]
        # numpy.histogram: index = (v - low) / (high - low) * bins, the last
        # edge joins the last bin, then one-step corrections against the edges.
        first = f"least(CAST(trunc(({v} - ?) / ? * {bins}) AS BIGINT), {bins - 1})"
        lowered = f"({first}) - CAST({v} < (?::DOUBLE[])[({first}) + 1] AS BIGINT)"
        index = (
            f"({lowered}) + CAST({v} >= (?::DOUBLE[])[({lowered}) + 2] "
            f"AND ({lowered}) != {bins - 1} AS BIGINT)"
        )
        # The CASE keeps out-of-range values (such as trimmed infinities) away
        # from the integer cast; a FILTER alone is applied after the argument.
        inside = f"{v} >= ? AND {v} <= ?"
        selects.append(f"histogram(CASE WHEN {inside} THEN {index} END) FILTER (WHERE {inside})")
        edge_list = edges.tolist()
        first_params = [low, high - low]
        lowered_params = first_params + [edge_list] + first_params
        params += (
            [low, high] + lowered_params + [edge_list] + lowered_params + lowered_params
            + [low, high]
        )
    check()
    row = conn.execute(f"SELECT {', '.join(selects)} FROM source", params).fetchone()
    out = {}
    for position, (feature, (bins, edges)) in enumerate(layout.items()):
        counts = np.zeros(bins, dtype=float)
        for index, count in (row[position] or {}).items():
            counts[index] = count
        out[feature] = _smoothed_points(counts, edges)
    return out


def _smoothed_points(counts: np.ndarray, edges: np.ndarray) -> list[dict[str, float]]:
    densities = counts / np.diff(edges) / counts.sum()
    densities = np.convolve(densities, np.array([0.25, 0.5, 0.25]), mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    return [
        {"x": round(float(x), 6), "density": round(float(y), 8)}
        for x, y in zip(centers, densities)
        if np.isfinite(x) and np.isfinite(y)
    ]


def _triangle(value: float) -> list[dict[str, float]]:
    spread = max(abs(value) * 0.05, 0.5)
    return [
        {"x": round(value - spread, 6), "density": 0.0},
        {"x": round(value, 6), "density": 1.0},
        {"x": round(value + spread, 6), "density": 0.0},
    ]


def _projection(feature: str, dtype: str) -> str:
    column = _quoted(feature)
    if dtype in {"FLOAT", "DOUBLE"}:
        column = f"CASE WHEN isnan({column}) THEN NULL ELSE {column} END"
    text = f"trim(CAST({column} AS VARCHAR), E' \\t\\n\\r\\v\\f')"
    if dtype == "BOOLEAN":
        text = f"CASE WHEN {column} THEN 'True' WHEN NOT {column} THEN 'False' END"
    numeric = f"TRY_CAST({column} AS DOUBLE)"
    if dtype == "VARCHAR" or dtype.startswith("ENUM"):
        # TRY_CAST also accepts hex and underscores, unlike pandas.to_numeric.
        # An overflowing decimal such as 1e999 is missing, as pandas 2 reads
        # it (pandas 3 reads infinity); only the words inf/infinity are.
        pattern = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
        numeric = (
            f"CASE WHEN regexp_full_match({text}, '{pattern}') "
            f"AND isfinite({numeric}) THEN {numeric} "
            f"WHEN regexp_full_match(lower({text}), '[+-]?inf(?:inity)?') "
            f"THEN {numeric} END"
        )
    return (
        f"SELECT {column} AS raw, {text} AS label, {numeric} AS numeric, "
        f"CAST(CASE lower({text}) WHEN 'true' THEN 1.0 WHEN 't' THEN 1.0 "
        "WHEN 'yes' THEN 1.0 WHEN 'y' THEN 1.0 WHEN '1' THEN 1.0 "
        "WHEN 'false' THEN 0.0 WHEN 'f' THEN 0.0 WHEN 'no' THEN 0.0 "
        "WHEN 'n' THEN 0.0 WHEN '0' THEN 0.0 END AS DOUBLE) AS boolean FROM source"
    )


def _summarize(conn: Any, feature: str, dtype: str) -> dict[str, Any]:
    conn.execute("CREATE OR REPLACE VIEW projected AS " + _projection(feature, dtype))
    total, clean, bool_count, invalid_bool = conn.execute(
        "SELECT count(*), count(raw), count(boolean), "
        "count(*) FILTER (WHERE raw IS NOT NULL AND label != '' AND boolean IS NULL) "
        "FROM projected"
    ).fetchone()
    empty = {"kind": "empty", "n": total, "non_null": 0, "points": [], "categories": []}
    if not clean:
        return empty
    value = "boolean" if bool_count and not invalid_bool else "numeric"
    conn.execute(
        f"CREATE OR REPLACE VIEW numbers AS SELECT {value} AS v FROM projected "
        f"WHERE raw IS NOT NULL AND {value} IS NOT NULL AND NOT isnan({value})"
    )
    n, low, high = conn.execute("SELECT count(*), min(v), max(v) FROM numbers").fetchone()
    if n >= max(2, int(clean * 0.65)):
        return {
            "kind": "numeric", "n": total, "non_null": n,
            "min": round(low, 6), "max": round(high, 6),
            "points": _density(conn, n, low, high), "categories": [],
        }
    counts = conn.execute(
        "SELECT CASE WHEN label = '' THEN 'missing' ELSE label END AS category, "
        "count(*) AS n FROM projected WHERE raw IS NOT NULL "
        "GROUP BY category ORDER BY n DESC, category LIMIT 8"
    ).fetchall()
    return {
        "kind": "categorical", "n": total, "non_null": clean, "points": [],
        "categories": [
            {"label": label, "count": count, "pct": round(count / total * 100, 1)}
            for label, count in counts
        ],
    }


def _density(conn: Any, n: int, low: float, high: float) -> list[dict[str, float]]:
    if n == 1 or low == high:
        return _triangle(low)
    selection = "SELECT v FROM numbers"
    if n >= 10:
        q1, q99 = conn.execute(
            "SELECT quantile_cont(v, [0.01, 0.99]) FROM numbers"
        ).fetchone()[0]
        if not (math.isnan(q1) or math.isnan(q99)):
            kept, trim_low, trim_high = conn.execute(
                "SELECT count(*), min(v), max(v) FROM numbers WHERE v >= ? AND v <= ?",
                [q1, q99],
            ).fetchone()
            if kept >= 6 and trim_low != trim_high:
                low, high = trim_low, trim_high
                # The retained range has the same rows as the percentile predicate.
                selection += f" WHERE v >= {low!r} AND v <= {high!r}"
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"autodetected range of [{low}, {high}] is not finite")
    bins = max(6, min(32, int(math.sqrt(n)) + 2))
    edges = np.linspace(low, high, bins + 1)
    # NumPy corrects the arithmetic index against the actual floating bin edges.
    # This preserves boundary values even when a division rounds one ULP away.
    counts = conn.execute(
        "WITH settings AS (SELECT ? AS edges), indexed AS ("
        f"SELECT v, least({bins - 1}, CAST(floor((v - ?) / (? - ?) * {bins}) AS INT)) AS i "
        f"FROM ({selection})), decremented AS ("
        "SELECT v, i - CAST(v < edges[i + 1] AS INT) AS i FROM indexed, settings) "
        f"SELECT i + CAST(v >= edges[i + 2] AND i != {bins - 1} AS INT) AS bin, count(*) "
        "FROM decremented, settings GROUP BY bin ORDER BY bin",
        [edges.tolist(), low, high, low],
    ).fetchall()
    histogram = np.zeros(bins, dtype=float)
    for index, count in counts:
        histogram[index] = count
    return _smoothed_points(histogram, edges)
