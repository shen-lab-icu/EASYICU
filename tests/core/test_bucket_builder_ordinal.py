"""Footer ordinals replace the seal builder's full-table ordering (synthetic only)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import types
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.build_itemid_bucket_cache as builder
from easyicu.storage_layout import MANIFEST, ORDINAL

SEAL_SHA = "70e5ad2c99ab8a8cc030bfe31063234d98599bd4"


def _source(root, shape="many"):
    directory = root / "events"
    directory.mkdir(parents=True)
    # Natural, case-folded and locale orders differ from DuckDB's binary order.
    # Valid Unicode scalar strings have the same lexicographic UTF-8 byte order
    # as Python code-point order; include composed/decomposed and astral names.
    names = ["2", "10", "a", "Z", "é", "e\u0301", "Ω", "😀", "quote'"]
    sizes = [5003, 0, 901, 4100, 17, 2301, 0, 89, 2100]
    if shape != "many":
        names, sizes = ["single"], [5003 if shape == "single" else 0]
    for name, size in zip(names, sizes):
        row = np.arange(size, dtype=np.int64)
        values = np.where(row % 7 == 0, -0.0, row / 11)
        table = pa.table(
            {
                "stay_id": row % 31,
                "itemid": row % 3,
                "clock": row % 13,
                "value": values,
            }
        )
        pq.write_table(table, directory / f"{name}.parquet", row_group_size=777)
    return list(reversed(sorted(directory.glob("*.parquet"))))


def _legacy_relation(files):
    paths = "[" + ",".join(builder._sql_string(p) for p in files) + "]"
    return (
        "(SELECT *, CAST(row_number() OVER (ORDER BY filename, file_row_number) - 1 "
        f"AS BIGINT) AS {builder._identifier(ORDINAL)} FROM read_parquet({paths}, "
        "filename=true, file_row_number=true, union_by_name=true, hive_partitioning=false))"
    )


@pytest.mark.parametrize("shape", ["many", "single", "empty"])
def test_footer_ordinal_equals_seal_window(tmp_path, shape):
    files = _source(tmp_path, shape)
    with duckdb.connect(config={"threads": 2}) as con:
        relation = builder._ordinal_relation(con, files)
        projection = f'filename, file_row_number, "{ORDINAL}"'
        expected = con.execute(
            f"SELECT {projection} FROM {_legacy_relation(files)} ORDER BY 1, 2"
        ).fetchall()
        actual = con.execute(
            f"SELECT {projection} FROM {relation} ORDER BY 1, 2"
        ).fetchall()
        assert actual == expected
        ordered = con.execute(
            "SELECT filename FROM _easyicu_source_offsets ORDER BY filename"
        ).fetchall()
        assert [name for (name,) in ordered] == sorted(
            (str(p) for p in files), key=lambda name: name.encode("utf-8")
        )


def test_no_global_filename_window():
    source = Path(builder.__file__).read_text()
    assert not re.search(
        r"row_number\(\)\s+OVER\s*\(\s*ORDER\s+BY\s+filename", source, re.I
    )


def _legacy_builder():
    repo = Path(builder.__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", f"{SEAL_SHA}:scripts/build_itemid_bucket_cache.py"],
        cwd=repo,
        text=True,
    )
    module = types.ModuleType("legacy_seal_builder")
    module.__file__ = builder.__file__
    exec(compile(source, builder.__file__, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("scheme", ["hash", "range", "variable_stay_range"])
def test_footer_builder_bytes_equal_seal(tmp_path, monkeypatch, scheme):
    _source(tmp_path)
    calls = []
    ordinal_relation = builder._ordinal_relation

    def tracked(con, files):
        calls.append(len(files))
        return ordinal_relation(con, files)

    monkeypatch.setattr(builder, "_ordinal_relation", tracked)
    kwargs = dict(
        table="events",
        key="itemid",
        bucket_count=4,
        threads=2,
        memory_limit="256MB",
        row_group_target=2048,
        sort_keys=["stay_id", "clock"],
    )
    legacy = _legacy_builder()
    for module, name in [(legacy, "old"), (builder, "new")]:
        if scheme == "variable_stay_range":
            module.build_variable_cache(
                tmp_path,
                **kwargs,
                stay_key="stay_id",
                rows_per_bucket=2048,
                output_root=tmp_path / name,
            )
        else:
            module.build_bucket_cache(
                tmp_path,
                **kwargs,
                scheme=scheme,
                range_breakpoints=[0, 1, 2] if scheme == "range" else None,
                output_root=tmp_path / name,
            )
    assert calls == [9]
    roots = [tmp_path / name for name in ("old", "new")]
    digests = [
        {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.parquet")
        }
        for root in roots
    ]
    assert digests[0] == digests[1]
    assert json.loads((roots[0] / MANIFEST).read_text()) == json.loads(
        (roots[1] / MANIFEST).read_text()
    )
