#!/usr/bin/env python3
"""Build an atomic DuckDB-hash bucket cache for a flat Parquet table.

The EasyICU reader prefers ``<table>_bucket/bucket_id=N/*.parquet`` when it
exists.  This utility materialises that layout without changing the source
Parquet files.  It is intended for very large long tables that are repeatedly
filtered by an item identifier, such as AmsterdamUMCdb ``numericitems``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb


RECEIPT_SCHEMA = "easyicu_itemid_bucket_cache_v1"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _schema(con: duckdb.DuckDBPyConnection, relation: str) -> list[list[str]]:
    return [[str(row[0]), str(row[1])] for row in con.execute(
        f"DESCRIBE SELECT * FROM {relation} LIMIT 0"
    ).fetchall()]


def _row_fingerprint(
    con: duckdb.DuckDBPyConnection,
    relation: str,
    columns: list[str],
) -> dict[str, str | int]:
    quoted = ", ".join(
        '"' + column.replace('"', '""') + '"' for column in columns
    )
    rows, xor_hash, sum_hash = con.execute(
        f"SELECT count(*), bit_xor(hash({quoted})), sum(hash({quoted})) "
        f"FROM {relation}"
    ).fetchone()
    return {
        "rows": int(rows),
        "bit_xor_hash": str(xor_hash),
        "sum_hash": str(sum_hash),
    }


def build_bucket_cache(
    data_path: Path,
    *,
    table: str,
    key: str,
    bucket_count: int = 64,
    memory_limit: str = "2GB",
    threads: int = 2,
    verify_row_hash: bool = False,
) -> dict[str, Any]:
    """Build, validate, and atomically install one item-hash bucket cache."""

    data_path = Path(data_path).resolve()
    if not _IDENTIFIER.fullmatch(table) or not _IDENTIFIER.fullmatch(key):
        raise ValueError("table and key must be simple SQL identifiers")
    if bucket_count <= 0 or threads <= 0:
        raise ValueError("bucket_count and threads must be positive")

    source = data_path / table
    source_files = sorted(source.glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No flat Parquet shards found under {source}")

    target = data_path / f"{table}_bucket"
    temporary = data_path / f".{table}_bucket_build"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {target}")
    if temporary.exists():
        raise FileExistsError(
            f"Remove or inspect incomplete cache build before retrying: {temporary}"
        )

    temporary.mkdir(parents=True)
    spill = data_path / f".duckdb_bucket_spill_{table}"
    source_glob = source / "*.parquet"
    target_glob = temporary / "bucket_id=*" / "*.parquet"
    started = time.time()
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit={_sql_string(memory_limit)}")
        con.execute(f"SET threads={int(threads)}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET partitioned_write_max_open_files={int(bucket_count)}")
        con.execute(f"SET temp_directory={_sql_string(spill)}")

        source_relation = (
            f"read_parquet({_sql_string(source_glob)}, union_by_name=true)"
        )
        source_rows, source_file_count = con.execute(
            "SELECT sum(num_rows), count(*) FROM parquet_file_metadata(?)",
            [str(source_glob)],
        ).fetchone()
        source_schema = _schema(con, source_relation)
        if key not in {name for name, _ in source_schema}:
            raise KeyError(f"Bucket key {key!r} is absent from {source}")
        if "bucket_id" in {name for name, _ in source_schema}:
            raise ValueError("Source table already contains a bucket_id column")

        con.execute(
            f"""
            COPY (
              SELECT *, hash("{key}") % {int(bucket_count)} AS bucket_id
              FROM {source_relation}
            ) TO {_sql_string(temporary)} (
              FORMAT PARQUET,
              PARTITION_BY (bucket_id),
              COMPRESSION ZSTD,
              ROW_GROUP_SIZE 122880
            )
            """
        )

        target_files = sorted(temporary.glob("bucket_id=*/*.parquet"))
        if not target_files:
            raise RuntimeError("Bucket build produced no Parquet files")
        target_rows, target_file_count = con.execute(
            "SELECT sum(num_rows), count(*) FROM parquet_file_metadata(?)",
            [str(target_glob)],
        ).fetchone()
        target_relation = (
            f"read_parquet({_sql_string(target_glob)}, "
            "hive_partitioning=true, union_by_name=true)"
        )
        target_schema = _schema(
            con,
            f"(SELECT * EXCLUDE(bucket_id) FROM {target_relation})",
        )
        if int(source_rows) != int(target_rows):
            raise RuntimeError(
                f"Row-count mismatch: source={source_rows}, target={target_rows}"
            )
        if source_schema != target_schema:
            raise RuntimeError(
                f"Schema mismatch: source={source_schema}, target={target_schema}"
            )

        row_multiset_fingerprint = None
        if verify_row_hash:
            columns = [name for name, _ in source_schema]
            source_fingerprint = _row_fingerprint(con, source_relation, columns)
            target_fingerprint = _row_fingerprint(con, target_relation, columns)
            if source_fingerprint != target_fingerprint:
                raise RuntimeError(
                    "Row-multiset fingerprint mismatch: "
                    f"source={source_fingerprint}, target={target_fingerprint}"
                )
            row_multiset_fingerprint = {
                "algorithm": "count+bit_xor(duckdb_hash(row))+sum(duckdb_hash(row))",
                **source_fingerprint,
            }

        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "source": str(source),
            "source_files": int(source_file_count),
            "source_inventory": [
                {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in source_files
            ],
            "rows": int(source_rows),
            "key": key,
            "bucket_count": int(bucket_count),
            "target_files": int(target_file_count),
            "partition_expression": f'hash("{key}") % {int(bucket_count)}',
            "source_schema": source_schema,
            "memory_limit": memory_limit,
            "threads": int(threads),
            "duckdb_version": duckdb.__version__,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if row_multiset_fingerprint is not None:
            receipt["row_multiset_fingerprint"] = row_multiset_fingerprint
        (temporary / "_BUCKET_BUILD_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "_COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, target)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        con.close()
        shutil.rmtree(spill, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--table", default="numericitems")
    parser.add_argument("--key", default="itemid")
    parser.add_argument("--bucket-count", type=int, default=64)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--verify-row-hash",
        action="store_true",
        help="Scan source and cache once more and compare order-independent row hashes.",
    )
    args = parser.parse_args()
    receipt = build_bucket_cache(
        args.data_path,
        table=args.table,
        key=args.key,
        bucket_count=args.bucket_count,
        memory_limit=args.memory_limit,
        threads=args.threads,
        verify_row_hash=args.verify_row_hash,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
