#!/usr/bin/env python3
"""Build or adopt a verified, atomically installed physical Parquet layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import easyicu
import easyicu.storage_layout as _storage_layout
from easyicu.runtime import duckdb_spill
from easyicu.storage_layout import (
    BUILDER_VERSION,
    MANIFEST,
    ORDINAL,
    PARTITION_PREFIX,
    SCHEMA,
    VARIABLE_SCHEME,
    arrow_key_type,
)

RECEIPT_SCHEMA = "easyicu_itemid_bucket_cache_v1"
ADOPTION_RECEIPT = "_BUCKET_ADOPTION_RECEIPT.json"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TABLE_CONFIGS = {
    "miiv": ("chartevents", "itemid", "stay_id", "charttime"),
    "eicu": (
        "nursecharting",
        "nursingchartcelltypevallabel",
        "patientunitstayid",
        "nursingchartoffset",
    ),
    "hirid": ("observations", "variableid", "patientid", "datetime"),
    "mimic": ("chartevents", "itemid", "icustay_id", "charttime"),
    "aumc": ("numericitems", "itemid", "admissionid", "measuredat"),
}

# Variable × stay layouts: (variable key, secondary selector columns). The
# stay key is TABLE_CONFIGS' third field when the table is listed there,
# otherwise --stay-key.
VARIABLE_PROFILES = {
    ("eicu", "nursecharting"): ("nursingchartcelltypevallabel", ["nursingchartcelltypevalname"]),
    ("eicu", "lab"): ("labname", []),
    ("eicu", "infusiondrug"): ("drugname", []),
    ("eicu", "intakeoutput"): ("celllabel", ["cellpath"]),
    ("eicu", "respiratorycharting"): ("respchartvaluelabel", []),
    ("miiv", "chartevents"): ("itemid", []),
    ("mimic", "chartevents"): ("itemid", []),
    ("aumc", "numericitems"): ("itemid", []),
    ("hirid", "observations"): ("variableid", []),
    ("sic", "medication"): ("DrugID", []),
}
STAY_KEYS = {"eicu": "patientunitstayid", "miiv": "stay_id", "mimic": "icustay_id",
             "aumc": "admissionid", "hirid": "patientid", "sic": "CaseID"}
# Distinct secondary values kept in the manifest, summed over partitions.
MAX_SECONDARY_VALUES = 200_000


class ImportOriginError(RuntimeError):
    """The easyicu package was imported from outside this script's checkout."""

    reason = "builder_import_origin_mismatch"

    def __init__(self, expected: str, modules: dict[str, str]):
        self.expected = expected
        self.modules = modules
        super().__init__(
            f"{self.reason}: easyicu must come from {expected}; imported {modules}"
        )


def builder_import_origin() -> dict[str, Any]:
    """Assert that easyicu resolves to this checkout's src and describe where from.

    An editable install elsewhere (for example a webapp runtime) would otherwise
    silently supply storage_layout and duckdb_spill from a different commit.
    """
    expected = (Path(__file__).resolve().parents[1] / "src").resolve()
    modules = {
        module.__name__: os.path.realpath(module.__file__)
        for module in (easyicu, _storage_layout, duckdb_spill)
    }
    if any(not Path(path).is_relative_to(expected) for path in modules.values()):
        raise ImportOriginError(str(expected), modules)
    return {"expected_src": str(expected), "modules": modules}


def builder_code() -> dict[str, Any]:
    """The commit this script was checked out at and whether that checkout is clean.

    Recorded identically in the manifest and the build receipt, so a seal can
    require builder_code.commit == the seal SHA and dirty == false. Outside a
    git checkout the commit is None, which a seal refuses (builder_code_missing).
    """
    repo = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    if not commit or status is None:
        return {"source": "unavailable", "commit": None, "dirty": None}
    return {"source": "git", "commit": commit, "dirty": bool(status)}


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


# DuckDB's parquet dictionary encoder keys floating-point values by equality, so
# -0.0 and +0.0 (and NaN payloads) share one dictionary entry and come back with
# whichever bit pattern was inserted first: the sign of zero would depend on row
# and thread order. A table with a floating-point column is therefore written
# without dictionary encoding (DICTIONARY_SIZE_LIMIT 0 keeps every bit pattern);
# other tables keep the default. Never canonicalise: the flat path reads the
# source bits, and a bucket must return the same bits.
_FLOAT_TYPES = frozenset({"FLOAT", "DOUBLE", "REAL", "FLOAT4", "FLOAT8"})
_FLOAT_ALIASES = {"REAL": "FLOAT", "FLOAT4": "FLOAT", "FLOAT8": "DOUBLE"}
_FLOAT_TOKEN = re.compile(r"\b(?:DOUBLE|FLOAT|REAL|FLOAT4|FLOAT8)\b")
_INTEGER_TYPES = frozenset({
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
})
# A source shard's float column may be read into the table's unified type only
# along these widenings: the flat reader (read_parquet union_by_name) applies
# the same CAST, so the bucket must hold exactly what the flat path returns
# (DECIMAL(p,s) also widens into DOUBLE, as DuckDB's union does).
_WIDENING = {
    "DOUBLE": frozenset({"DOUBLE", "FLOAT"}) | _INTEGER_TYPES,
    "FLOAT": frozenset({"FLOAT"}) | _INTEGER_TYPES,
}
# Rows per compared batch = this budget / bytes per row (ordinal + validity +
# every float column), so QC memory does not grow with table width.
QC_BATCH_BYTES = 64 << 20
FLOAT_FIDELITY_ALGORITHM = (
    "every row bound to its source ordinal: the source is read file by file in the "
    "builder's ordinal order (file name, file row number) and cast to the flat "
    "reader's unified type; the bucket is read ordered by the ordinal; per float "
    "column the null positions and IEEE bit patterns (sign of zero, NaN payload) "
    "are compared exactly, row by row. Recorded per column: non-null values, nulls, "
    "-0.0, +0.0, NaN, and sha256 over (ordinal int64, validity u8, bits u64) in "
    "ordinal order."
)


class FloatFidelityError(RuntimeError):
    """A bucket's floating-point bits differ from its source, or cannot be bound (typed build failure)."""

    def __init__(
        self,
        column: str,
        source: dict,
        target: dict,
        reason: str = "bucket_float_fidelity_mismatch",
    ):
        self.reason, self.column, self.source, self.target = reason, column, source, target
        super().__init__(f"{reason}: column {column!r} source {source} target {target}")


def _float_columns(schema: list[list[str]]) -> list[str]:
    """Scalar float columns; a float nested in a LIST/STRUCT/MAP/ARRAY fails closed."""
    columns = []
    for name, kind in schema:
        upper = kind.upper()
        if upper in _FLOAT_TYPES:
            columns.append(name)
        elif _FLOAT_TOKEN.search(upper):
            raise FloatFidelityError(name, {"type": kind}, {}, reason="unsupported_nested_float")
    return columns


def _parquet_options(schema: list[list[str]], *extra: str) -> str:
    """COPY options shared by staging and final writes (see ``_FLOAT_TYPES``)."""
    options = ["FORMAT PARQUET", *extra, "COMPRESSION ZSTD"]
    if _float_columns(schema):
        options.append("DICTIONARY_SIZE_LIMIT 0")
    return ", ".join(options)


def _unified(kind: str) -> str:
    upper = kind.upper()
    return _FLOAT_ALIASES.get(upper, upper)


def _check_source_float_types(con, source_files, schema, columns) -> None:
    """Each shard's float column must widen into the unified type the flat reader uses."""
    unified = {c: _unified(t) for c, t in schema if c in columns}
    for path in source_files:
        shard = {c.casefold(): t for c, t in _schema(con, _relation([path]))}
        for column in columns:
            kind = shard.get(column.casefold())
            if kind is None:
                continue  # absent in this shard: NULL, as union_by_name reads it
            if _unified(kind) not in _WIDENING[unified[column]] and not (
                unified[column] == "DOUBLE" and _unified(kind).startswith("DECIMAL(")
            ):
                raise FloatFidelityError(
                    column, {"shard_type": kind}, {"unified_type": unified[column]},
                    reason="bucket_float_source_type_unsupported",
                )


class _Rows:
    """Re-slices a stream of record batches into chunks of an exact row count."""

    def __init__(self, batches):
        self._batches, self._current, self._offset = iter(batches), None, 0

    def take(self, rows: int):
        parts, need = [], rows
        while need:
            if self._current is None or self._offset >= self._current.num_rows:
                self._current, self._offset = next(self._batches, None), 0
                if self._current is None:
                    break
                continue
            count = min(need, self._current.num_rows - self._offset)
            parts.append(self._current.slice(self._offset, count))
            self._offset += count
            need -= count
        return pa.Table.from_batches(parts) if parts else None


def _source_batches(con, source_files, columns, unified, rows):
    """The source in the builder's ordinal order, cast to the unified types."""
    for path in sorted(source_files, key=str):
        present = {c.casefold(): c for c, _ in _schema(con, _relation([path]))}
        projection = ", ".join(
            f"CAST({_identifier(present[c.casefold()])} AS {unified[c]}) AS {_identifier(c)}"
            if c.casefold() in present else f"CAST(NULL AS {unified[c]}) AS {_identifier(c)}"
            for c in columns
        )
        reader = con.execute(
            f"SELECT file_row_number AS __row, {projection} FROM read_parquet("
            f"{_sql_string(path)}, file_row_number=true, hive_partitioning=false)"
        ).fetch_record_batch(rows)
        expected = 0
        for batch in reader:
            row = batch.column(0).to_numpy(zero_copy_only=False)
            if not np.array_equal(row, np.arange(expected, expected + len(row))):
                raise RuntimeError("bucket_float_source_order_unverifiable")
            expected += len(row)
            yield batch.drop_columns(["__row"]) if hasattr(batch, "drop_columns") else pa.RecordBatch.from_arrays(
                batch.columns[1:], names=batch.schema.names[1:]
            )


class _ColumnProfile:
    def __init__(self):
        self.values = self.nulls = self.neg_zero = self.pos_zero = self.nan = 0
        self.digest = hashlib.sha256()

    def update(self, ordinals: np.ndarray, valid: np.ndarray, values: np.ndarray, bits: np.ndarray) -> None:
        zero = valid & (values == 0)
        negative = np.signbit(values)
        self.values += int(np.count_nonzero(valid))
        self.nulls += int(valid.size - np.count_nonzero(valid))
        self.neg_zero += int(np.count_nonzero(zero & negative))
        self.pos_zero += int(np.count_nonzero(zero & ~negative))
        self.nan += int(np.count_nonzero(valid & np.isnan(values)))
        record = np.empty(ordinals.size, dtype=[("o", "<i8"), ("v", "u1"), ("b", "<u8")])
        record["o"], record["v"], record["b"] = ordinals, valid, np.where(valid, bits, 0)
        self.digest.update(record.tobytes())

    def result(self) -> dict:
        return {"values": self.values, "nulls": self.nulls, "neg_zero": self.neg_zero,
                "pos_zero": self.pos_zero, "nan": self.nan, "sha256": self.digest.hexdigest()}


def _float_arrays(column):
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    valid = array.is_valid().to_numpy(zero_copy_only=False)
    values = array.fill_null(0).to_numpy(zero_copy_only=False)
    width = {2: np.uint16, 4: np.uint32, 8: np.uint64}[values.dtype.itemsize]
    return valid, values, values.view(width).astype(np.uint64)


def _peak_rss_kb() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak // 1024) if sys.platform == "darwin" else int(peak)


def _assert_float_fidelity(con, source_files, target_files, schema) -> tuple[dict, dict]:
    """Bind every float value to its source ordinal and compare bits exactly.

    Returns (fidelity, qc): ``fidelity`` is deterministic (manifest and receipt);
    ``qc`` holds this run's wall time, peak RSS and batch size (receipt only).
    Raises ``FloatFidelityError`` (typed reason) on any difference.
    """
    started, rss_before = time.monotonic(), _peak_rss_kb()
    columns = _float_columns(schema)
    fidelity = {
        "algorithm": FLOAT_FIDELITY_ALGORITHM,
        "dictionary_encoding": "disabled" if columns else "default",
        "columns": {},
    }
    if not columns:
        return fidelity, {"wall_seconds": 0.0, "batch_rows": 0}
    _check_source_float_types(con, source_files, schema, columns)
    unified = {c: _unified(t) for c, t in schema if c in columns}
    width = 9 + sum(8 if unified[c] == "DOUBLE" else 4 for c in columns)
    rows = max(4096, QC_BATCH_BYTES // width)
    source_cursor, target_cursor = con.cursor(), con.cursor()
    for cursor in (source_cursor, target_cursor):
        cursor.execute("SET preserve_insertion_order=true")
    target = target_cursor.execute(
        f"SELECT {_identifier(ORDINAL)}, {', '.join(_identifier(c) for c in columns)} "
        f"FROM {_relation(target_files)} ORDER BY {_identifier(ORDINAL)}"
    ).fetch_record_batch(rows)
    source_rows = _Rows(_source_batches(source_cursor, source_files, columns, unified, rows))
    target_rows = _Rows(target)
    source_profile = {c: _ColumnProfile() for c in columns}
    target_profile = {c: _ColumnProfile() for c in columns}
    mismatched = dict.fromkeys(columns, 0)
    first = dict.fromkeys(columns)
    ordinal = 0
    while True:
        s, t = source_rows.take(rows), target_rows.take(rows)
        if s is None and t is None:
            break
        if s is None or t is None or s.num_rows != t.num_rows:
            raise FloatFidelityError(
                ORDINAL, {"rows_from": ordinal}, {"rows_from": ordinal},
                reason="bucket_float_row_count_mismatch",
            )
        ordinals = np.arange(ordinal, ordinal + t.num_rows, dtype=np.int64)
        if not np.array_equal(t.column(0).to_numpy(), ordinals):
            raise FloatFidelityError(
                ORDINAL, {"ordinal_from": ordinal}, {"ordinal_from": ordinal},
                reason="bucket_ordinal_binding_mismatch",
            )
        for column in columns:
            sv, svalues, sbits = _float_arrays(s.column(column))
            tv, tvalues, tbits = _float_arrays(t.column(column))
            source_profile[column].update(ordinals, sv, svalues, sbits)
            target_profile[column].update(ordinals, tv, tvalues, tbits)
            bad = (sv != tv) | (sv & (sbits != tbits))
            if bad.any():
                mismatched[column] += int(np.count_nonzero(bad))
                if first[column] is None:
                    first[column] = int(ordinals[np.argmax(bad)])
        ordinal += t.num_rows
    for column in columns:
        src, tgt = source_profile[column].result(), target_profile[column].result()
        if mismatched[column] or src != tgt:
            raise FloatFidelityError(
                column,
                {**src, "mismatched_rows": mismatched[column], "first_mismatch_ordinal": first[column]},
                tgt,
            )
        fidelity["columns"][column] = src
    fidelity["rows"] = ordinal
    qc = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "batch_rows": rows,
        "batch_bytes_budget": QC_BATCH_BYTES,
        "process_peak_rss_kb_before": rss_before,
        "process_peak_rss_kb_after": _peak_rss_kb(),
    }
    return fidelity, qc


def _assert_float_fidelity_unordered(con, source_files, files, schema, key, count) -> tuple[dict, dict]:
    """Adoption of a legacy cache without an ordinal: bind floats to the whole record.

    Per bucket, the multiset of complete source rows routed there
    (``hash(key) % count``) must equal the bucket's rows exactly, with each float
    column's sign bit carried alongside its value (EXCEPT ALL both ways), so
    sign corruption cannot cancel across buckets or move between distinct
    records. Records equal apart from the sign of a zero have no identity
    without an ordinal, so such a source is refused
    (``adopt_sign_ambiguous_rebuild_required``). NaN payloads are not
    distinguished on this path.
    """
    started, rss_before = time.monotonic(), _peak_rss_kb()
    columns = _float_columns(schema)
    fidelity = {"algorithm": "adoption: per-bucket exact multiset of complete rows with float sign bits "
                "(EXCEPT ALL both ways), refused when source records equal apart from a zero's sign "
                "exist (no ordinal to bind them); NaN payloads not distinguished",
                "dictionary_encoding": "unverified_legacy", "columns": {}}
    if not columns:
        return fidelity, {"wall_seconds": 0.0}
    _check_source_float_types(con, source_files, schema, columns)
    signs = ", ".join(f"signbit({_identifier(c)}) AS {_identifier('__sign_' + c)}" for c in columns)
    source = f"SELECT hash({_identifier(key)}) % {count} AS __bucket, *, {signs} FROM {_relation(source_files)}"
    target = " UNION ALL ".join(
        f"SELECT {int(f.parent.name.split('=')[1])}::UBIGINT AS __bucket, *, {signs} FROM {_relation([f])}"
        for f in files
    )
    missing, extra = con.execute(
        f"SELECT (SELECT count(*) FROM (({source}) EXCEPT ALL ({target}))), "
        f"(SELECT count(*) FROM (({target}) EXCEPT ALL ({source})))"
    ).fetchone()
    if missing or extra:
        raise FloatFidelityError("*", {"rows_not_in_bucket": int(missing)}, {"rows_not_in_source": int(extra)})
    # Without a source ordinal, records that are equal apart from the sign of a
    # zero cannot be told apart: a sign swap between them leaves the signed
    # multiset unchanged, yet FIRST() and other order-dependent reads change.
    # Such a source can only be certified by an ordinal-bound rebuild.
    normalized = ", ".join(
        f"CASE WHEN {_identifier(c)} = 0 THEN abs({_identifier(c)}) ELSE {_identifier(c)} END"
        if c in columns else _identifier(c)
        for c, _ in schema
    )
    pattern = " || ".join(
        f"coalesce(signbit({_identifier(c)})::INTEGER::VARCHAR, 'n')" for c in columns
    )
    ambiguous = con.execute(
        f"SELECT count(*) FROM (SELECT count(DISTINCT {pattern}) AS patterns "
        f"FROM {_relation(source_files)} GROUP BY {normalized}) WHERE patterns > 1"
    ).fetchone()[0]
    if ambiguous:
        raise FloatFidelityError(
            "*",
            {"sign_ambiguous_groups": int(ambiguous)},
            {"required": "rebuild with build_bucket_cache (ordinal-bound fidelity check)"},
            reason="adopt_sign_ambiguous_rebuild_required",
        )
    for column in columns:
        c = _identifier(column)
        values, nulls, neg, pos, nan = con.execute(
            f"SELECT count({c}), count(*) - count({c}), count(*) FILTER (WHERE {c} = 0 AND signbit({c})), "
            f"count(*) FILTER (WHERE {c} = 0 AND NOT signbit({c})), count(*) FILTER (WHERE isnan({c})) "
            f"FROM {_relation(source_files)}"
        ).fetchone()
        fidelity["columns"][column] = {"values": values, "nulls": nulls, "neg_zero": neg,
                                       "pos_zero": pos, "nan": nan}
    qc = {"wall_seconds": round(time.monotonic() - started, 3),
          "process_peak_rss_kb_before": rss_before, "process_peak_rss_kb_after": _peak_rss_kb()}
    return fidelity, qc


def _relation(files) -> str:
    return (
        "read_parquet(["
        + ",".join(_sql_string(p) for p in files)
        + "], union_by_name=true, hive_partitioning=false)"
    )


def _schema(con, relation: str) -> list[list[str]]:
    return [
        [str(row[0]), str(row[1])]
        for row in con.execute(f"DESCRIBE SELECT * FROM {relation} LIMIT 0").fetchall()
    ]


def _row_fingerprint(con, relation: str, columns: list[str]) -> dict[str, str | int]:
    quoted = ", ".join(_identifier(c) for c in columns)
    rows, xor_hash, sum_hash = con.execute(
        f"SELECT count(*), bit_xor(hash({quoted})), sum(hash({quoted})) FROM {relation}"
    ).fetchone()
    return {"rows": int(rows), "bit_xor_hash": str(xor_hash), "sum_hash": str(sum_hash)}


def _verify_bucket(con, path: Path, sort_keys: list[str], expected_rows: int) -> None:
    """Compare physical positions to the declared total order, including ties."""
    order = ",".join(_identifier(c) + " ASC NULLS LAST" for c in sort_keys)
    rows, disorder = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE physical != expected) FROM (
            SELECT row_number() OVER () physical,
                   row_number() OVER (ORDER BY {order}) expected FROM {_relation([path])}
        )
    """).fetchone()
    if rows != expected_rows or disorder:
        raise RuntimeError(f"bucket_sort_verification_failed: {path.parent.name}")


def _inventory(root: Path) -> list[dict]:
    return [
        dict(
            relative_path=p.relative_to(root).as_posix(),
            size=p.stat().st_size,
            row_count=pq.ParquetFile(p).metadata.num_rows,
        )
        for p in sorted(root.glob("bucket_id=*/*.parquet"))
    ]


def _manifest(
    root, *, key, key_type, count, scheme, breaks, sort, row_group_target, source_order,
    derived_from=None,
):
    result = dict(
        schema=SCHEMA,
        scheme=scheme,
        key_column=key,
        key_type=key_type,
        sort_keys=sort,
        source_order=source_order,
        row_group_target=row_group_target,
        builder_version=BUILDER_VERSION,
        inventory=_inventory(root),
    )
    if derived_from is not None:
        result["derived_from"] = derived_from
    if scheme == "hash":
        result.update(
            modulus=count,
            hash={"name": "duckdb.hash", "duckdb_version": duckdb.__version__},
        )
    else:
        result["range_breakpoints"] = breaks
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_from(data_path: Path, table: str, source_files) -> dict:
    """Bind the installation to the exact source bytes it was built from.

    Paths are relative to the data path as given (a symlinked file in an
    assembly directory is hashed through its link, i.e. the data it names).
    """
    return dict(
        table=table,
        files=[
            dict(
                path=p.relative_to(data_path).as_posix(),
                size=p.stat().st_size,
                mtime_ns=p.stat().st_mtime_ns,
                sha256=_sha256_file(p),
            )
            for p in source_files
        ],
    )


def _source_files(data_path, table):
    directory = data_path / table
    files = sorted(p for p in directory.glob("*.parquet") if not p.name.startswith("."))
    if not files and (data_path / f"{table}.parquet").is_file():
        files = [data_path / f"{table}.parquet"]
    if not files:
        raise FileNotFoundError(f"No flat Parquet source for {table} under {data_path}")
    return files


def _assert_multiset(con, source_relation, target_relation, columns):
    source = _row_fingerprint(con, source_relation, columns)
    if source != _row_fingerprint(con, target_relation, columns):
        raise RuntimeError("Row-multiset fingerprint mismatch")
    return {
        "algorithm": "count+bit_xor(duckdb_hash(row))+sum(duckdb_hash(row))",
        **source,
    }


def build_bucket_cache(
    data_path: Path,
    *,
    table: str,
    key: str,
    bucket_count: int = 64,
    memory_limit: str = "2GB",
    threads: int = 2,
    verify_row_hash: bool = True,
    scheme: str = "hash",
    sort_keys: list[str] | None = None,
    row_group_target: int = 131072,
    output_root: Path | None = None,
    range_breakpoints: list | None = None,
) -> dict[str, Any]:
    """Stage once, explicitly sort and verify each bucket, then atomically install.

    The ordinal is assigned from the sorted source file list and physical row
    numbers. It both resolves exact sort ties and restores the original logical
    scan order in readers. Row-multiset validation is mandatory; the historical
    verify_row_hash argument remains accepted for callers of the old builder.
    Spill storage is owned by easyicu.runtime.duckdb_spill.
    """
    import_origin = builder_import_origin()
    code = builder_code()
    data_path = Path(data_path).resolve()
    if not _IDENTIFIER.fullmatch(table) or not _IDENTIFIER.fullmatch(key):
        raise ValueError("table and key must be simple SQL identifiers")
    if (
        any(
            type(value) is not int or value <= 0
            for value in (bucket_count, threads, row_group_target)
        )
        or row_group_target < 2048
    ):
        raise ValueError(
            "bucket_count/threads must be positive and row_group_target >= 2048"
        )
    breaks = list(range_breakpoints or [])
    if scheme not in {"hash", "range"} or (
        scheme == "range" and any(a >= b for a, b in zip(breaks, breaks[1:]))
    ):
        raise ValueError("invalid scheme or range breakpoints")
    count = bucket_count if scheme == "hash" else len(breaks) + 1
    source_files = _source_files(data_path, table)
    target = (
        Path(output_root).resolve() if output_root else data_path / f"{table}_bucket"
    )
    temporary = target.with_name(f".{target.name}_build")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {target}")
    if temporary.exists():
        raise FileExistsError(
            f"Inspect incomplete cache build before retrying: {temporary}"
        )
    if target.is_relative_to(data_path / table) or any(
        target == p for p in source_files
    ):
        raise ValueError("output must be a new root, separate from the source")
    temporary.mkdir(parents=True)
    started = time.monotonic()
    try:
        with duckdb_spill.connection(
            config={"threads": threads, "memory_limit": memory_limit}
        ) as con:
            con.execute("SET preserve_insertion_order=true")
            source_relation = _relation(source_files)
            schema = _schema(con, source_relation)
            _float_columns(schema)  # fails closed on a nested float leaf
            columns = [c for c, _ in schema]
            by_lower = {c.casefold(): c for c in columns}
            key = by_lower.get(key.casefold(), key)
            if key not in columns:
                raise KeyError(f"Bucket key {key!r} is absent")
            if {
                ORDINAL,
                "bucket_id",
                "_easyicu_bucket_id",
                "filename",
                "file_row_number",
            } & set(columns):
                raise ValueError("Source contains a reserved storage column")
            key_type = dict(schema)[key]
            if (
                arrow_key_type(pq.read_schema(source_files[0]).field(key).type)
                != key_type
            ):
                raise ValueError("Unsupported or inconsistent bucket key type")
            if sort_keys is None:
                stay = next(
                    (
                        by_lower[c]
                        for c in (
                            "stay_id",
                            "icustay_id",
                            "patientunitstayid",
                            "admissionid",
                            "patientid",
                            "caseid",
                        )
                        if c in by_lower
                    ),
                    None,
                )
                clock = next(
                    (
                        by_lower[c]
                        for c in (
                            "charttime",
                            "nursingchartoffset",
                            "datetime",
                            "measuredat",
                            "starttime",
                            "offset",
                        )
                        if c in by_lower
                    ),
                    None,
                )
                sort_keys = list(dict.fromkeys(c for c in (key, stay, clock) if c))
            else:
                sort_keys = [by_lower.get(c.casefold(), c) for c in sort_keys]
            if any(c not in columns for c in sort_keys):
                raise ValueError("sort_keys must name source columns")
            sort = list(dict.fromkeys([*sort_keys, ORDINAL]))
            source_rows = con.execute(
                f"SELECT count(*) FROM {source_relation}"
            ).fetchone()[0]
            if scheme == "hash":
                partition = f"hash({_identifier(key)}) % {count}"
            else:
                partition = (
                    "CASE "
                    + " ".join(
                        f"WHEN {_identifier(key)} <= {_sql_string(b) if isinstance(b, str) else b} THEN {i}"
                        for i, b in enumerate(breaks)
                    )
                    + f" ELSE {count - 1} END"
                    if breaks
                    else "0"
                )
            # Multi-threaded PARTITION_BY is only a staging operation. Final
            # writes below each have their own ORDER BY and verification.
            paths = "[" + ",".join(_sql_string(p) for p in source_files) + "]"
            staging = temporary / ".stage"
            con.execute(f"""COPY (
                SELECT {",".join(_identifier(c) for c in columns)},
                    CAST(row_number() OVER (ORDER BY filename, file_row_number) - 1 AS BIGINT) AS {_identifier(ORDINAL)},
                    {partition} AS _easyicu_bucket_id
                FROM read_parquet({paths}, filename=true, file_row_number=true, union_by_name=true, hive_partitioning=false)
            ) TO {_sql_string(staging)} ({_parquet_options(schema, "PARTITION_BY (_easyicu_bucket_id)")})""")
            for bid in range(count):
                directory = temporary / f"bucket_id={bid}"
                directory.mkdir()
                staged = sorted(
                    (staging / f"_easyicu_bucket_id={bid}").glob("*.parquet")
                )
                if not staged and (source_rows or bid != 0):
                    continue
                relation = (
                    _relation(staged)
                    if staged
                    else f"(SELECT *, 0::BIGINT AS {_identifier(ORDINAL)} FROM {source_relation} LIMIT 0)"
                )
                expected_rows = con.execute(
                    f"SELECT count(*) FROM {relation}"
                ).fetchone()[0]
                path = directory / "part.parquet"
                order = ",".join(_identifier(c) + " ASC NULLS LAST" for c in sort)
                con.execute(
                    f"COPY (SELECT * FROM {relation} ORDER BY {order}) TO {_sql_string(path)} "
                    f"({_parquet_options(schema, f'ROW_GROUP_SIZE {row_group_target}')})"
                )
                _verify_bucket(con, path, sort, expected_rows)
            shutil.rmtree(staging, ignore_errors=True)
            target_files = sorted(temporary.glob("bucket_id=*/*.parquet"))
            target_relation = _relation(target_files)
            if (
                _schema(con, f'(SELECT * EXCLUDE("{ORDINAL}") FROM {target_relation})')
                != schema
            ):
                raise RuntimeError("Schema mismatch")
            fingerprint = _assert_multiset(
                con, source_relation, target_relation, columns
            )
            float_fidelity, float_fidelity_qc = _assert_float_fidelity(
                con, source_files, target_files, schema
            )
            derived = _derived_from(data_path, table, source_files)
            receipt = dict(
                schema=RECEIPT_SCHEMA,
                builder_version=BUILDER_VERSION,
                derived_from=derived,
                source=str(data_path / table),
                source_files=len(source_files),
                source_inventory=[
                    dict(
                        path=str(p.relative_to(data_path)),
                        size=p.stat().st_size,
                        mtime_ns=p.stat().st_mtime_ns,
                    )
                    for p in source_files
                ],
                rows=source_rows,
                key=key,
                bucket_count=count,
                target_files=len(target_files),
                partition_expression=partition,
                source_schema=schema,
                memory_limit=memory_limit,
                threads=threads,
                duckdb_version=duckdb.__version__,
                sort_keys=sort,
                row_group_target=row_group_target,
                row_multiset_fingerprint=fingerprint,
                float_fidelity=float_fidelity,
                float_fidelity_qc=float_fidelity_qc,
                import_origin=import_origin,
                builder_code=code,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            manifest = _manifest(
                temporary,
                derived_from=derived,
                key=key,
                key_type=key_type,
                count=count,
                scheme=scheme,
                breaks=breaks,
                sort=sort,
                row_group_target=row_group_target,
                source_order=ORDINAL,
            )
            manifest["float_fidelity"] = float_fidelity
            manifest["builder_code"] = code
            (temporary / "_BUCKET_BUILD_RECEIPT.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            (temporary / MANIFEST).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            (temporary / "_COMPLETE").write_text("complete\n")
        os.replace(temporary, target)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _pow2_floor(value: int) -> int:
    return 1 << (max(1, int(value)).bit_length() - 1)


def build_variable_cache(
    data_path: Path,
    *,
    table: str,
    key: str,
    stay_key: str,
    bucket_count: int = 64,
    rows_per_bucket: int = 1_048_576,
    secondary_keys: list[str] | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
    sort_keys: list[str] | None = None,
    row_group_target: int = 131072,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Build a verified variable × stay-range layout (``variable_stay_range``).

    One partition per value of ``key`` (a null key is its own partition),
    then stay ranges inside it. ``bucket_count`` is the number M of stay
    ranges, a power of two, cut at row quantiles of ``stay_key`` and shared by
    every partition (``stay <= breakpoint`` belongs to the earlier range).
    Each partition merges M/m consecutive ranges, m the largest power of two
    <= M leaving about ``rows_per_bucket`` rows per bucket. Files are sorted by
    the stay, the clock and the source ordinal; staging, sort verification,
    the row-multiset check, receipts and the atomic install are those of
    :func:`build_bucket_cache`.
    """
    import_origin = builder_import_origin()
    code = builder_code()
    data_path = Path(data_path).resolve()
    secondary_keys = list(secondary_keys or [])
    for name in (table, key, stay_key, *secondary_keys):
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError("table and key columns must be simple SQL identifiers")
    if (
        any(
            type(value) is not int or value <= 0
            for value in (bucket_count, threads, row_group_target, rows_per_bucket)
        )
        or bucket_count & (bucket_count - 1)
        or row_group_target < 2048
    ):
        raise ValueError(
            "bucket_count must be a power of two; counts positive; row_group_target >= 2048"
        )
    source_files = _source_files(data_path, table)
    target = (
        Path(output_root).resolve() if output_root else data_path / f"{table}_bucket"
    )
    temporary = target.with_name(f".{target.name}_build")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {target}")
    if temporary.exists():
        raise FileExistsError(
            f"Inspect incomplete cache build before retrying: {temporary}"
        )
    if target.is_relative_to(data_path / table) or any(target == p for p in source_files):
        raise ValueError("output must be a new root, separate from the source")
    temporary.mkdir(parents=True)
    started = time.monotonic()
    try:
        with duckdb_spill.connection(
            config={"threads": threads, "memory_limit": memory_limit}
        ) as con:
            con.execute("SET preserve_insertion_order=true")
            source_relation = _relation(source_files)
            schema = _schema(con, source_relation)
            _float_columns(schema)  # fails closed on a nested float leaf
            columns = [c for c, _ in schema]
            by_lower = {c.casefold(): c for c in columns}
            key = by_lower.get(key.casefold(), key)
            stay_key = by_lower.get(stay_key.casefold(), stay_key)
            secondary_keys = [by_lower.get(c.casefold(), c) for c in secondary_keys]
            for column in (key, stay_key, *secondary_keys):
                if column not in columns:
                    raise KeyError(f"Layout column {column!r} is absent")
            if key == stay_key or key in secondary_keys or stay_key in secondary_keys:
                raise ValueError("key, stay key and secondary keys must differ")
            if {
                ORDINAL, "bucket_id", "_easyicu_bucket_id", "_easyicu_partition_id",
                "filename", "file_row_number",
            } & set(columns):
                raise ValueError("Source contains a reserved storage column")
            types = dict(schema)
            first = pq.read_schema(source_files[0])
            for column in (key, stay_key):
                if arrow_key_type(first.field(column).type) != types[column]:
                    raise ValueError("Unsupported or inconsistent layout key type")
            key_type, stay_type = types[key], types[stay_key]
            if sort_keys is None:
                clock = next(
                    (by_lower[c] for c in (
                        "charttime", "nursingchartoffset", "labresultoffset",
                        "infusionoffset", "intakeoutputoffset", "respchartoffset",
                        "datetime", "measuredat", "starttime", "offset",
                    ) if c in by_lower),
                    None,
                )
                sort_keys = [c for c in (stay_key, clock) if c]
            else:
                sort_keys = [by_lower.get(c.casefold(), c) for c in sort_keys]
            if any(c not in columns for c in sort_keys):
                raise ValueError("sort_keys must name source columns")
            sort = list(dict.fromkeys([*sort_keys, ORDINAL]))
            source_rows = con.execute(f"SELECT count(*) FROM {source_relation}").fetchone()[0]
            if not source_rows:
                raise ValueError("a variable layout needs at least one source row")
            counts = con.execute(
                f"SELECT {_identifier(key)}, count(*) FROM {source_relation} "
                f"GROUP BY 1 ORDER BY 1 ASC NULLS LAST"
            ).fetchall()
            partitions = []
            for pid, (value, rows) in enumerate(counts):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("variable key has a non-finite value")
                modulus = min(bucket_count, _pow2_floor(rows // rows_per_bucket))
                partitions.append({"id": pid, "value": value, "modulus": modulus, "rows": int(rows)})
            con.execute("CREATE OR REPLACE TEMP TABLE _easyicu_partitions (pid BIGINT, value "
                        f"{key_type}, span BIGINT)")
            con.executemany(
                "INSERT INTO _easyicu_partitions VALUES (?, ?, ?)",
                [(p["id"], p["value"], bucket_count // p["modulus"]) for p in partitions],
            )
            if bucket_count > 1:
                fractions = [i / bucket_count for i in range(1, bucket_count)]
                breakpoints = con.execute(
                    f"SELECT quantile_disc({_identifier(stay_key)}, {fractions}) "
                    f"FROM {source_relation} WHERE {_identifier(stay_key)} IS NOT NULL"
                ).fetchone()[0]
                if breakpoints is None:
                    raise ValueError("stay key has no values")
                breakpoints = list(breakpoints)
            else:
                breakpoints = []
            stay_range = (
                "CASE "
                + " ".join(
                    f"WHEN s.{_identifier(stay_key)} <= {_sql_string(b) if isinstance(b, str) else b} THEN {i}"
                    for i, b in enumerate(breakpoints)
                )
                + f" ELSE {bucket_count - 1} END"
                if breakpoints
                else "0"
            )
            paths = "[" + ",".join(_sql_string(p) for p in source_files) + "]"
            staging = temporary / ".stage"
            projected = ",".join("s." + _identifier(c) for c in columns)
            con.execute(f"""COPY (
                SELECT {projected}, s.{_identifier(ORDINAL)},
                    m.pid AS _easyicu_partition_id,
                    ({stay_range}) // m.span AS _easyicu_bucket_id
                FROM (
                    SELECT *, CAST(row_number() OVER (ORDER BY filename, file_row_number) - 1 AS BIGINT)
                        AS {_identifier(ORDINAL)}
                    FROM read_parquet({paths}, filename=true, file_row_number=true,
                                      union_by_name=true, hive_partitioning=false)
                ) s
                JOIN _easyicu_partitions m ON s.{_identifier(key)} IS NOT DISTINCT FROM m.value
            ) TO {_sql_string(staging)} ({_parquet_options(schema, "PARTITION_BY (_easyicu_partition_id)")})""")
            order = ",".join(_identifier(c) + " ASC NULLS LAST" for c in sort)
            secondary_total = 0
            for part in partitions:
                pid, modulus = part["id"], part["modulus"]
                partition_dir = temporary / f"{PARTITION_PREFIX}{pid}"
                staged = sorted((staging / f"_easyicu_partition_id={pid}").glob("*.parquet"))
                if not staged:
                    raise RuntimeError("variable_partition_staging_missing")
                relation = _relation(staged)
                per_bucket = dict(con.execute(
                    f"SELECT _easyicu_bucket_id, count(*) FROM {relation} GROUP BY 1"
                ).fetchall())
                for bid in range(modulus):
                    directory = partition_dir / f"bucket_id={bid}"
                    directory.mkdir(parents=True)
                    expected_rows = per_bucket.get(bid, 0)
                    if not expected_rows:
                        continue
                    path = directory / "part.parquet"
                    con.execute(
                        f"COPY (SELECT * EXCLUDE (_easyicu_bucket_id) FROM {relation} "
                        f"WHERE _easyicu_bucket_id = {bid} ORDER BY {order}) TO {_sql_string(path)} "
                        f"({_parquet_options(schema, f'ROW_GROUP_SIZE {row_group_target}')})"
                    )
                    _verify_bucket(con, path, sort, expected_rows)
                part["secondary"] = {}
                for column in secondary_keys:
                    values = [
                        row[0] for row in con.execute(
                            f"SELECT DISTINCT {_identifier(column)} FROM {relation} "
                            f"WHERE {_identifier(column)} IS NOT NULL ORDER BY 1"
                        ).fetchall()
                    ]
                    secondary_total += len(values)
                    part["secondary"][column] = values
                if secondary_total > MAX_SECONDARY_VALUES:
                    raise ValueError("too many secondary key values for a manifest")
                if sum(per_bucket.values()) != part["rows"]:
                    raise RuntimeError("variable_partition_row_mismatch")
            shutil.rmtree(staging, ignore_errors=True)
            target_files = sorted(temporary.glob(f"{PARTITION_PREFIX}*/bucket_id=*/*.parquet"))
            target_relation = _relation(target_files)
            if (
                _schema(con, f'(SELECT * EXCLUDE("{ORDINAL}") FROM {target_relation})')
                != schema
            ):
                raise RuntimeError("Schema mismatch")
            fingerprint = _assert_multiset(con, source_relation, target_relation, columns)
            float_fidelity, float_fidelity_qc = _assert_float_fidelity(
                con, source_files, target_files, schema
            )
            derived = _derived_from(data_path, table, source_files)
            receipt = dict(
                schema=RECEIPT_SCHEMA,
                builder_version=BUILDER_VERSION,
                derived_from=derived,
                layout_scheme=VARIABLE_SCHEME,
                source=str(data_path / table),
                source_files=len(source_files),
                source_inventory=[
                    dict(path=str(p.relative_to(data_path)), size=p.stat().st_size,
                         mtime_ns=p.stat().st_mtime_ns)
                    for p in source_files
                ],
                rows=source_rows,
                key=key,
                stay_key=stay_key,
                secondary_keys=secondary_keys,
                bucket_count=bucket_count,
                rows_per_bucket=rows_per_bucket,
                partitions=len(partitions),
                target_files=len(target_files),
                partition_expression=(
                    f"{key} value -> partition; stay range of {stay_key} // (M / partition modulus)"
                ),
                stay_breakpoints=breakpoints,
                source_schema=schema,
                memory_limit=memory_limit,
                threads=threads,
                duckdb_version=duckdb.__version__,
                sort_keys=sort,
                row_group_target=row_group_target,
                row_multiset_fingerprint=fingerprint,
                float_fidelity=float_fidelity,
                float_fidelity_qc=float_fidelity_qc,
                import_origin=import_origin,
                builder_code=code,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            manifest = dict(
                schema=SCHEMA,
                scheme=VARIABLE_SCHEME,
                key_column=key,
                key_type=key_type,
                stay_column=stay_key,
                stay_key_type=stay_type,
                modulus=bucket_count,
                stay_breakpoints=breakpoints,
                derived_from=derived,
                secondary_keys=secondary_keys,
                partitions=[
                    {k: part[k] for k in ("id", "value", "modulus", "secondary")}
                    for part in partitions
                ],
                sort_keys=sort,
                source_order=ORDINAL,
                row_group_target=row_group_target,
                builder_version=BUILDER_VERSION,
                builder_code=code,
                float_fidelity=float_fidelity,
                inventory=[
                    dict(
                        relative_path=p.relative_to(temporary).as_posix(),
                        size=p.stat().st_size,
                        row_count=pq.ParquetFile(p).metadata.num_rows,
                    )
                    for p in target_files
                ],
            )
            (temporary / "_BUCKET_BUILD_RECEIPT.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n"
            )
            (temporary / MANIFEST).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            (temporary / "_COMPLETE").write_text("complete\n")
        os.replace(temporary, target)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def adopt_bucket_cache(
    data_path: Path,
    *,
    table="numericitems",
    key="itemid",
    bucket_root: Path | None = None,
    memory_limit="2GB",
    threads=2,
) -> dict:
    """Verify an existing legacy cache against its source and receipted modulus.

    No Parquet bytes are rewritten. The old cache's scan order is retained.
    The manifest alone is installed atomically, after every check succeeds.
    """
    import_origin = builder_import_origin()
    code = builder_code()
    data_path = Path(data_path).resolve()
    root = Path(bucket_root) if bucket_root else data_path / f"{table}_bucket"
    receipt = json.loads((root / "_BUCKET_BUILD_RECEIPT.json").read_text())
    count = receipt.get("bucket_count")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("key") != key
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or receipt.get("duckdb_version") != duckdb.__version__
        or not (root / "_COMPLETE").is_file()
    ):
        raise ValueError(
            "bucket_legacy_unverifiable: build receipt/modulus/hash identity required"
        )
    if {d.name for d in root.glob("bucket_id=*") if d.is_dir()} - {
        f"bucket_id={i}" for i in range(count)
    }:
        raise ValueError("bucket_modulus_mismatch")
    source = _source_files(data_path, table)
    files = sorted(
        p for p in root.glob("bucket_id=*/*.parquet") if not p.name.startswith(".")
    )
    if not files:
        raise ValueError("bucket_legacy_unverifiable: no files")
    with duckdb_spill.connection(
        config={"threads": threads, "memory_limit": memory_limit}
    ) as con:
        schema = _schema(con, _relation(source))
        if (
            schema != receipt.get("source_schema")
            or _schema(con, _relation(files)) != schema
        ):
            raise ValueError("bucket_legacy_unverifiable: schema mismatch")
        fingerprint = _assert_multiset(
            con, _relation(source), _relation(files), [c for c, _ in schema]
        )
        if fingerprint["rows"] != receipt["rows"]:
            raise ValueError("bucket_legacy_unverifiable: receipt row count mismatch")
        for file in files:
            bid = int(file.parent.name.split("=")[1])
            wrong = con.execute(
                f"SELECT count(*) FROM {_relation([file])} WHERE hash({_identifier(key)}) % {count} != {bid}"
            ).fetchone()[0]
            if wrong:
                raise ValueError(
                    "bucket_legacy_unverifiable: partition membership mismatch"
                )
        # The same float-fidelity gate as a build; a mismatch refuses adoption.
        float_fidelity, float_fidelity_qc = _assert_float_fidelity_unordered(
            con, source, files, schema, key, count
        )
    # Only after verification may an empty, omitted legacy bucket be created.
    for bid in range(count):
        (root / f"bucket_id={bid}").mkdir(exist_ok=True)
    manifest = _manifest(
        root,
        key=key,
        key_type=dict(schema)[key],
        count=count,
        scheme="hash",
        breaks=[],
        sort=[],
        row_group_target=122880,
        source_order=None,
    )
    manifest["float_fidelity"] = float_fidelity
    manifest["builder_code"] = code
    adoption = dict(
        schema="easyicu_bucket_adoption_v1",
        builder_version=BUILDER_VERSION,
        duckdb_version=duckdb.__version__,
        rows=fingerprint["rows"],
        row_multiset_fingerprint=fingerprint,
        float_fidelity=float_fidelity,
        float_fidelity_qc=float_fidelity_qc,
        import_origin=import_origin,
        builder_code=code,
    )
    # The adoption receipt first, the manifest last: the manifest is what
    # makes the cache readable, so it appears only after every check passed.
    for name, content in ((ADOPTION_RECEIPT, adoption), (MANIFEST, manifest)):
        temporary = root / (name + ".tmp")
        try:
            temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, root / name)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database", choices=sorted(set(TABLE_CONFIGS) | set(STAY_KEYS)))
    parser.add_argument("--table")
    parser.add_argument("--key")
    parser.add_argument("--bucket-count", "--modulus", type=int, default=64)
    parser.add_argument(
        "--scheme", choices=("hash", "range", VARIABLE_SCHEME), default="hash",
        help=(
            f"{VARIABLE_SCHEME}: one partition per --key value, then stay ranges "
            "inside it (--stay-key; --bucket-count is the power-of-two number of "
            "stay ranges; --rows-per-bucket sizes each partition's buckets; "
            "--secondary-keys records per-partition values of other selector columns)"
        ),
    )
    parser.add_argument("--stay-key")
    parser.add_argument("--rows-per-bucket", type=int, default=1_048_576)
    parser.add_argument("--secondary-keys", help="Comma-separated selector columns")
    parser.add_argument(
        "--range-breakpoints",
        help="JSON array; equal values belong to the earlier range",
    )
    parser.add_argument(
        "--sort-keys",
        help="Comma-separated source columns; source ordinal is always appended",
    )
    parser.add_argument("--row-group-target", type=int, default=131072)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--verify-row-hash",
        action="store_true",
        help="Compatibility flag; verification is always enabled",
    )
    parser.add_argument("--adopt", action="store_true")
    args = parser.parse_args()
    # Databases without a default table (SIC) name --table and --key.
    profile = TABLE_CONFIGS.get(args.database or "aumc", (None, None, None, None))
    common = dict(
        table=args.table or profile[0],
        key=args.key or profile[1],
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    if not common["table"] or not (common["key"] or args.scheme == VARIABLE_SCHEME):
        parser.error("--table and --key are required for this database")
    if args.scheme == VARIABLE_SCHEME:
        database = args.database or ""
        table = args.table or profile[0]
        default_key, default_secondary = VARIABLE_PROFILES.get((database, table), (None, []))
        result = build_variable_cache(
            args.data_path,
            table=table,
            key=args.key or default_key or profile[1],
            stay_key=args.stay_key or STAY_KEYS.get(database) or profile[2],
            bucket_count=args.bucket_count,
            rows_per_bucket=args.rows_per_bucket,
            secondary_keys=(
                args.secondary_keys.split(",") if args.secondary_keys else default_secondary
            ),
            memory_limit=args.memory_limit,
            threads=args.threads,
            sort_keys=args.sort_keys.split(",") if args.sort_keys else None,
            row_group_target=args.row_group_target,
            output_root=args.output_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.adopt:
        result = adopt_bucket_cache(
            args.data_path, **common, bucket_root=args.output_root
        )
    else:
        result = build_bucket_cache(
            args.data_path,
            **common,
            bucket_count=args.bucket_count,
            scheme=args.scheme,
            sort_keys=args.sort_keys.split(",") if args.sort_keys else None,
            row_group_target=args.row_group_target,
            output_root=args.output_root,
            range_breakpoints=json.loads(args.range_breakpoints)
            if args.range_breakpoints
            else None,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
