"""Bounded stay-level reductions for Cohort Statistics.

Parquet observation rows never enter pandas. DuckDB projects the requested
columns and groups by the canonical stay key; only stay-level values and the
existing kind-inference sample (at most 5,000 values) cross into Python.
"""

from __future__ import annotations

from easyicu.runtime import duckdb_spill
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.webserver import entity_ids as entity_id_contract

THREADS = 8
MEMORY_LIMIT = "2GiB"
# Exact medians retain observations in the engine. Bound concurrent states,
# including when the request selects the maximum 48 features.
_COLUMNS_PER_QUERY = 2
_INTEGER_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
}
_NUMERIC_TYPES = _INTEGER_TYPES | {"BOOLEAN", "FLOAT", "DOUBLE"}
# Python str.strip() includes Unicode spaces and four ASCII separators in
# addition to the usual six ASCII whitespace characters.
_STRIP_WHITESPACE = (
    " \t\n\r\v\f\x1c\x1d\x1e\x1f\x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


@dataclass
class StayReductions:
    """A module's projected schema and reductions, without observation rows."""

    columns: tuple[str, ...]
    entities: set[str]
    numeric: dict[tuple[str, str], dict[str, float]]
    flags: dict[str, dict[str, bool]]

    @property
    def empty(self) -> bool:
        return not self.entities

    def filter(self, ids: set[str]) -> StayReductions:
        return StayReductions(
            self.columns,
            self.entities & ids,
            {
                key: {sid: value for sid, value in mapping.items() if sid in ids}
                for key, mapping in self.numeric.items()
            },
            {
                key: {sid: value for sid, value in mapping.items() if sid in ids}
                for key, mapping in self.flags.items()
            },
        )


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


@contextmanager
def _connection():
    with duckdb_spill.connection(
        config={"threads": THREADS, "memory_limit": MEMORY_LIMIT}
    ) as conn:
        # Cap the dense integer hash table at 2^20 slots. Sparse identifiers
        # retain DuckDB's regular hash aggregate; both paths are exact.
        conn.execute("SET perfect_ht_threshold = 20")
        yield conn


def _safe_integer_keys(path: Path, column: str) -> bool:
    # Scalar identifier normalization historically converts through float.
    # Large or nullable integer keys therefore need the exact pandas boundary,
    # rather than an unconditional integer-to-text SQL cast.
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.get_field_index(column)
    for number in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(number)
        stats = group.column(index).statistics
        if stats is None or not stats.has_min_max:
            return False
        if min(stats.min, stats.max) < -(2**53) or max(stats.min, stats.max) > 2**53:
            return False
    return True


@contextmanager
def _source(
    path: Path,
    entity_column: str,
    *,
    normalize: Callable[[Any], Any] | None = None,
    ids: set[str] | None = None,
) -> Iterator[tuple[Any, dict[str, str]]]:
    # No persistent cache, worker pool, connection or spill file survives a
    # request. The summary owner already caches complete payloads by file identity.
    with _connection() as conn:
        conn.read_parquet(str(path), file_row_number=True).create_view("observations")
        types = {
            row[0]: row[1] for row in conn.execute("DESCRIBE observations").fetchall()
        }
        key = _quoted(entity_column)
        integer_key = types[entity_column] in _INTEGER_TYPES and _safe_integer_keys(
            path, entity_column
        )
        if integer_key:
            # Hash and join the native integer key. Formatting 108 M repeated
            # keys as VARCHAR would dominate an otherwise projected scan.
            canonical = key
            source = "observations"
        else:
            # Non-integer identifiers need Python's exact float/text spelling.
            # Normalize unique keys, never every repeated observation.
            keys = conn.execute(
                f"SELECT DISTINCT {key} FROM observations"
            ).fetch_arrow_table()
            # Retain the export's pandas extension dtypes, including nullable
            # integers, while the join key itself stays in Arrow's exact type.
            keys = keys.replace_schema_metadata(
                pq.ParquetFile(path).schema_arrow.metadata
            )
            values = keys.to_pandas()[entity_column]
            normalized = (
                normalize(values)
                if normalize
                else values.map(entity_id_contract.normalize_entity_id)
            )
            conn.register(
                "id_names",
                pa.table(
                    {"raw_key": keys.column(0), "canonical_key": pa.array(normalized)}
                ),
            )
            canonical = "canonical_key"
            source = f"observations JOIN id_names ON {key} IS NOT DISTINCT FROM raw_key"
        conn.execute(
            f"CREATE VIEW normalized AS SELECT observations.*, {canonical} AS cohort_stay_id FROM {source}"
        )
        if ids is not None:
            conn.register(
                "cohort_id_names",
                pa.table({"name": pa.array(sorted(ids), type=pa.string())}),
            )
            if integer_key:
                cast = f"TRY_CAST(name AS {types[entity_column]})"
                conn.execute(
                    f"CREATE VIEW cohort_ids AS SELECT {cast} AS cohort_stay_id FROM cohort_id_names WHERE CAST({cast} AS VARCHAR) = name"
                )
            else:
                conn.execute(
                    "CREATE VIEW cohort_ids AS SELECT name AS cohort_stay_id FROM cohort_id_names"
                )
            conn.execute(
                "CREATE VIEW selected AS SELECT normalized.* FROM normalized SEMI JOIN cohort_ids USING (cohort_stay_id)"
            )
        else:
            conn.execute("CREATE VIEW selected AS SELECT * FROM normalized")
        yield conn, types


def entity_frame(path: Path, entity_column: str) -> Any:
    """The original first-occurrence stay order, without repeated key rows."""
    with _source(path, entity_column) as (conn, _types):
        rows = conn.execute(
            "SELECT cohort_stay_id FROM selected GROUP BY cohort_stay_id ORDER BY min(file_row_number)"
        ).fetchall()
    return pd.DataFrame(
        {"stay_id": ["" if row[0] is None else str(row[0]) for row in rows]}
    )


def covered_entities(path: Path, entity_column: str, ids: set[str]) -> int:
    with _source(path, entity_column, ids=ids) as (conn, _types):
        return conn.execute(
            "SELECT count(*) FROM (SELECT DISTINCT cohort_stay_id FROM normalized) keys SEMI JOIN cohort_ids USING (cohort_stay_id)"
        ).fetchone()[0]


def _numeric(conn: Any, column: str, dtype: str) -> str:
    value = _quoted(column)
    if dtype in _NUMERIC_TYPES:
        number = f"CAST({value} AS DOUBLE)"
    else:
        # pandas accepts a different text grammar (and datetime units) from
        # TRY_CAST. Apply its conversion to bounded Arrow vectors in DuckDB.
        name = "cohort_numeric_" + column

        def convert(values):
            numeric = pd.to_numeric(values.to_pandas(), errors="coerce")
            return pa.array(numeric, type=pa.float64(), from_pandas=True)

        conn.create_function(
            name, convert, [dtype], "DOUBLE", type="arrow", null_handling="special"
        )
        number = f"{_quoted(name)}({value})"
    return f"CASE WHEN isnan({number}) THEN NULL ELSE {number} END"


def _flag(column: str, dtype: str) -> str:
    value = _quoted(column)
    if dtype in _NUMERIC_TYPES:
        return f"CASE WHEN {value} = 1 THEN true WHEN {value} = 0 THEN false END"
    text = f"lower(trim(CAST({value} AS VARCHAR), '{_STRIP_WHITESPACE}'))"
    return (
        f"CASE WHEN {text} IN ('true','1','yes','y') THEN true "
        f"WHEN {text} IN ('false','0','no','n') THEN false "
        f"WHEN TRY_CAST({text} AS DOUBLE) = 1 THEN true "
        f"WHEN TRY_CAST({text} AS DOUBLE) = 0 THEN false END"
    )


def _reduce(conn: Any, expressions: Sequence[str]) -> list[tuple]:
    rows = conn.execute(
        f"SELECT cohort_stay_id, {', '.join(expressions)} FROM selected GROUP BY cohort_stay_id ORDER BY cohort_stay_id"
    ).fetchall()
    return [("" if row[0] is None else str(row[0]), *row[1:]) for row in rows]


def module_reductions(
    path: Path,
    entity_column: str,
    columns: Sequence[str],
    *,
    numeric_modes: Sequence[str],
    flag_columns: set[str],
    normalize: Callable[[Any], Any],
) -> StayReductions:
    numeric_maps = {}
    flag_maps = {}
    with _source(path, entity_column, normalize=normalize) as (conn, types):
        entities = {
            "" if row[0] is None else str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT cohort_stay_id FROM selected"
            ).fetchall()
        }
        for column in columns:
            if column == entity_column:
                continue
            if column in flag_columns:
                rows = _reduce(conn, [f"bool_or({_flag(column, types[column])})"])
                flag_maps[column] = {
                    sid: value for sid, value in rows if value is not None
                }
            elif column not in {"stay_id", "charttime", "time", "timestamp"}:
                value = _numeric(conn, column, types[column])
                expressions = [f"{mode}({value})" for mode in numeric_modes]
                if types[column] == "FLOAT":
                    expressions = [f"CAST({expr} AS FLOAT)" for expr in expressions]
                rows = _reduce(conn, expressions)
                for index, mode in enumerate(numeric_modes):
                    numeric_maps[column, mode] = {
                        row[0]: float(row[index + 1])
                        for row in rows
                        if row[index + 1] is not None
                    }
    return StayReductions(
        tuple(["stay_id"] + [column for column in columns if column != entity_column]),
        entities,
        numeric_maps,
        flag_maps,
    )


def _feature_samples(path, entity_column, columns, ids, normalize):
    parquet = pq.ParquetFile(path)

    def all_missing(column):
        index = parquet.schema_arrow.get_field_index(column)
        return all(
            group.column(index).statistics is not None
            and group.column(index).statistics.null_count == group.num_rows
            for group in (
                parquet.metadata.row_group(number)
                for number in range(parquet.num_row_groups)
            )
        )

    active = [column for column in columns if not all_missing(column)]
    samples = {column: [] for column in columns}
    remaining = {column: 5000 for column in active}
    for batch in (
        ()
        if not active
        else parquet.iter_batches(columns=[entity_column, *active], batch_size=8192)
    ):
        frame = batch.to_pandas()
        frame = frame.loc[normalize(frame[entity_column]).isin(ids)]
        for column in active:
            if remaining[column]:
                values = frame[column].dropna().head(remaining[column])
                samples[column].append(values)
                remaining[column] -= len(values)
        if not any(remaining.values()):
            break
    return {
        column: pd.DataFrame({column: pd.concat(parts, ignore_index=True)})
        if parts
        else pd.DataFrame({column: []})
        for column, parts in samples.items()
    }


def feature_profiles(
    path: Path,
    entity_column: str,
    features: list[dict],
    ids: set[str],
    *,
    normalize: Callable[[Any], Any],
    infer_kind: Callable[[Any, str], str],
    death_columns: Sequence[str],
    coverage_counts: dict[str, int] | None = None,
) -> list[dict]:
    profiles = []
    samples = _feature_samples(
        path, entity_column, [feature["column"] for feature in features], ids, normalize
    )
    with _source(path, entity_column, normalize=normalize) as (conn, types):
        for offset in range(0, len(features), _COLUMNS_PER_QUERY):
            batch = features[offset : offset + _COLUMNS_PER_QUERY]
            expressions = []
            kinds = []
            for feature in batch:
                column = feature["column"]
                dtype = types[column]
                quoted = _quoted(column)
                present = f"{quoted} IS NOT NULL"
                if dtype in {"FLOAT", "DOUBLE"}:
                    present += f" AND NOT isnan({quoted})"
                kind = infer_kind(samples[column], column)
                kinds.append(kind)
                if kind == "numeric":
                    expression = f"median({_numeric(conn, column, dtype)})"
                    if dtype == "FLOAT":
                        expression = f"CAST({expression} AS FLOAT)"
                elif kind == "binary":
                    expression = f"bool_or({_flag(column, dtype)})"
                    if (
                        feature["module"].strip().lower() == "outcome"
                        and column.strip().lower() in death_columns
                    ):
                        expression = f"coalesce({expression}, false)"
                else:
                    expression = f"bool_or({present})"
                expressions.append(expression)
            rows = [row for row in _reduce(conn, expressions) if row[0] in ids]
            if (
                coverage_counts is not None
                and types[entity_column] in _INTEGER_TYPES
                and _safe_integer_keys(path, entity_column)
            ):
                coverage_counts[str(path)] = len(rows)
            for index, (feature, kind) in enumerate(zip(batch, kinds)):
                mapping = {
                    row[0]: row[index + 1]
                    for row in rows
                    if row[index + 1] is not None
                    and (kind != "presence" or row[index + 1])
                }
                profiles.append(
                    {
                        "id": feature.get("id"),
                        "module": feature.get("module"),
                        "column": feature["column"],
                        "label": feature.get("label") or feature["column"],
                        "kind": kind,
                        "aggregation": {
                            "numeric": "entity_median",
                            "binary": "entity_any_positive_pct",
                            "presence": "entity_nonmissing_pct",
                        }[kind],
                        "mapping": mapping,
                        "reason": None,
                    }
                )
    return profiles


def group_medians(
    groups: dict[str, list[tuple[str, list[str]]]],
    mappings: list[dict[str, float]],
) -> dict[str, dict[int, list[float | None]]]:
    """All Table 1 medians, after the historical three-decimal rounding.

    Build one stay-level Arrow table, join the existing group memberships,
    then compute the exact medians in one DuckDB GROUP BY. Rounding values
    with Python preserves tie behavior at decimal boundaries.
    """
    import math

    unique = {id(mapping): mapping for mapping in mappings}
    ids = sorted(
        {
            sid
            for members in groups.values()
            for _label, group in members
            for sid in group
        }
    )
    positions = {sid: index for index, sid in enumerate(ids)}
    arrays = {"sid": pa.array(range(len(ids)), type=pa.int32())}
    for index, mapping in enumerate(unique.values()):
        rounded = {
            value: round(float(value), 3)
            if value is not None and not math.isnan(float(value))
            else None
            for value in set(mapping.values())
        }
        arrays[f"v{index}"] = pa.array(
            [
                float(mapping[sid])
                if mapping.get(sid) == 0
                else rounded.get(mapping.get(sid))
                for sid in ids
            ],
            type=pa.float64(),
        )
    member_ids, group_numbers = [], []
    layout = []
    for name, members in groups.items():
        for label, group in members:
            member_ids.extend(positions[sid] for sid in group)
            group_numbers.extend([len(layout)] * len(group))
            layout.append((name, label))
    result = {
        name: {key: [None] * len(members) for key in unique}
        for name, members in groups.items()
    }
    if not ids or not unique:
        return result
    with _connection() as conn:
        conn.register("stay_values", pa.table(arrays))
        conn.register(
            "members",
            pa.table(
                {
                    "sid": pa.array(member_ids, type=pa.int32()),
                    "group_number": pa.array(group_numbers, type=pa.int32()),
                }
            ),
        )
        expressions = [f"median(v{index})" for index in range(len(unique))]
        rows = conn.execute(
            f"SELECT group_number, {', '.join(expressions)} FROM members JOIN stay_values USING (sid) GROUP BY group_number"
        ).fetchall()
    starts = {}
    offset = 0
    for name, members in groups.items():
        starts[name] = offset
        offset += len(members)
    for row in rows:
        name, _label = layout[row[0]]
        position = row[0] - starts[name]
        for index, key in enumerate(unique):
            value = row[index + 1]
            if value == 0:
                # SQL quantiles may choose either sign among tied zeroes.
                # Python's stable order determines the historical JSON sign.
                mapping = unique[key]
                values = sorted(
                    round(float(mapping[sid]), 3)
                    for sid in groups[name][position][1]
                    if mapping.get(sid) is not None
                    and not math.isnan(float(mapping[sid]))
                )
                mid = len(values) // 2
                value = (
                    values[mid]
                    if len(values) % 2
                    else (values[mid - 1] + values[mid]) / 2
                )
            result[name][key][position] = round(value, 2) if value is not None else None
    return result


@dataclass
class TimeCounts:
    """Sufficient statistics for exact KM, censoring marks and log-rank."""

    total: dict[float, int]
    events: dict[float, int]

    def __len__(self):
        return sum(self.total.values())


def survival_counts(events, times, groups):
    """Count the already windowed stay-level KM inputs for every group once."""
    ids = [sid for sid in times if sid in events]
    members, numbers, layout = [], [], []
    for group in groups:
        if group.get("status") != "ready":
            continue
        for label, group_ids in group.get("_members", []):
            members.extend(group_ids)
            numbers.extend([len(layout)] * len(group_ids))
            layout.append((group["id"], label))
    result = {key: TimeCounts({}, {}) for key in layout}
    if not ids or not members:
        return result
    with _connection() as conn:
        conn.register(
            "timed",
            pa.table(
                {
                    "sid": pa.array(ids, type=pa.string()),
                    "time": pa.array([times[sid] for sid in ids], type=pa.float64()),
                    "event": pa.array(
                        [events[sid] is True for sid in ids], type=pa.bool_()
                    ),
                }
            ),
        )
        conn.register(
            "members",
            pa.table(
                {
                    "sid": pa.array(members, type=pa.string()),
                    "number": pa.array(numbers, type=pa.int32()),
                }
            ),
        )
        rows = conn.execute(
            "SELECT number, time, count(*), count(*) FILTER (WHERE event) FROM timed JOIN members USING (sid) GROUP BY number, time"
        ).fetchall()
    for number, time, total, positive in rows:
        counts = result[layout[number]]
        counts.total[time] = total
        if positive:
            counts.events[time] = positive
    return result
