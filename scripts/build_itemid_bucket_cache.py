#!/usr/bin/env python3
"""Build or adopt a verified, atomically installed physical Parquet layout."""

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
import pyarrow.parquet as pq

from easyicu.runtime import duckdb_spill
from easyicu.storage_layout import (
    BUILDER_VERSION,
    MANIFEST,
    ORDINAL,
    SCHEMA,
    arrow_key_type,
)

RECEIPT_SCHEMA = "easyicu_itemid_bucket_cache_v1"
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


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


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
    root, *, key, key_type, count, scheme, breaks, sort, row_group_target, source_order
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
    if scheme == "hash":
        result.update(
            modulus=count,
            hash={"name": "duckdb.hash", "duckdb_version": duckdb.__version__},
        )
    else:
        result["range_breakpoints"] = breaks
    return result


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
            ) TO {_sql_string(staging)} (FORMAT PARQUET, PARTITION_BY (_easyicu_bucket_id), COMPRESSION ZSTD)""")
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
                    f"COPY (SELECT * FROM {relation} ORDER BY {order}) TO {_sql_string(path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_target})"
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
            receipt = dict(
                schema=RECEIPT_SCHEMA,
                builder_version=BUILDER_VERSION,
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
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            manifest = _manifest(
                temporary,
                key=key,
                key_type=key_type,
                count=count,
                scheme=scheme,
                breaks=breaks,
                sort=sort,
                row_group_target=row_group_target,
                source_order=ORDINAL,
            )
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
    temporary = root / (MANIFEST + ".tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, root / MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database", choices=TABLE_CONFIGS)
    parser.add_argument("--table")
    parser.add_argument("--key")
    parser.add_argument("--bucket-count", "--modulus", type=int, default=64)
    parser.add_argument("--scheme", choices=("hash", "range"), default="hash")
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
    profile = TABLE_CONFIGS[args.database or "aumc"]
    common = dict(
        table=args.table or profile[0],
        key=args.key or profile[1],
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
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
