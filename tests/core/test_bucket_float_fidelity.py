"""Bucket builds keep every floating-point bit of the source (lead-r13 r13bitexact).

DuckDB's parquet dictionary encoder keys floats by equality, so -0.0 and +0.0
would share one entry and come back with one sign. The builder writes tables
with float columns without dictionary encoding and checks, per float column,
the -0.0/+0.0/NaN counts and a bit-pattern multiset hash of source vs bucket;
a mismatch fails the build with ``bucket_float_fidelity_mismatch``. Rows are
synthetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.build_itemid_bucket_cache as builder
from easyicu.storage_layout import MANIFEST, read_bucket_contract


def _source(root: Path, n: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(7)
    value = rng.choice([0.0, 1.5, 2.5, 7.0], n)
    value[rng.random(n) < 0.05] = -0.0
    value[0] = -0.0  # a negative zero reaches the dictionary first in some order
    value[rng.random(n) < 0.01] = np.nan
    table = pa.table(
        {
            "stay_id": pa.array(rng.integers(1, 50, n), pa.int64()),
            "itemid": pa.array(rng.integers(1, 6, n), pa.int64()),
            "charttime": pa.array(np.arange(n), pa.int64()),
            "valuenum": pa.array(value, pa.float64()),
            "weight": pa.array(value.astype(np.float32), pa.float32()),
        }
    )
    (root / "chartevents").mkdir(parents=True)
    pq.write_table(table, root / "chartevents" / "1.parquet", use_dictionary=False)
    return value


def _bits(paths, column):
    out = []
    for path in paths:
        array = pq.read_table(path, columns=[column]).column(column).combine_chunks()
        width = {4: np.uint32, 8: np.uint64}[array.type.bit_width // 8]
        out.append(array.to_numpy(zero_copy_only=False).view(width))
    return np.sort(np.concatenate(out))


def _expect_profile(value: np.ndarray) -> dict:
    zero = value == 0
    return {
        "neg_zero": int((zero & np.signbit(value)).sum()),
        "pos_zero": int((zero & ~np.signbit(value)).sum()),
        "nan": int(np.isnan(value).sum()),
        "values": int(value.size),
    }


@pytest.mark.parametrize("variable", [False, True], ids=["hash", "variable_stay_range"])
def test_bucket_keeps_signed_zero_and_nan_bits(tmp_path, variable):
    value = _source(tmp_path)
    if variable:
        builder.build_variable_cache(
            tmp_path,
            table="chartevents",
            key="itemid",
            stay_key="stay_id",
            bucket_count=4,
            rows_per_bucket=4096,
            threads=2,
        )
    else:
        builder.build_bucket_cache(
            tmp_path, table="chartevents", key="itemid", bucket_count=4, threads=2
        )
    root = tmp_path / "chartevents_bucket"
    parts = sorted(root.rglob("*.parquet"))
    source = [tmp_path / "chartevents" / "1.parquet"]
    for column in ("valuenum", "weight"):
        assert np.array_equal(_bits(parts, column), _bits(source, column)), column
    for part in parts:
        meta = pq.ParquetFile(part).metadata
        for i in range(meta.num_row_groups):
            for j in range(meta.num_columns):
                chunk = meta.row_group(i).column(j)
                if chunk.path_in_schema in ("valuenum", "weight"):
                    assert not chunk.has_dictionary_page
    manifest = json.loads((root / MANIFEST).read_text())
    fidelity = manifest["float_fidelity"]
    assert fidelity["dictionary_encoding"] == "disabled"
    expected = _expect_profile(value)
    for key, count in expected.items():
        assert fidelity["columns"]["valuenum"][key] == count
    assert (
        fidelity["columns"]["valuenum"]["neg_zero"] > 0
        and fidelity["columns"]["valuenum"]["nan"] > 0
    )
    receipt = json.loads((root / "_BUCKET_BUILD_RECEIPT.json").read_text())
    assert receipt["float_fidelity"] == fidelity
    read_bucket_contract(root)  # the reader still accepts the manifest


def test_float_profile_distinguishes_signed_zero_and_counts_nan(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    pq.write_table(pa.table({"v": pa.array([0.0, -0.0, math.nan, 1.0])}), a)
    pq.write_table(pa.table({"v": pa.array([0.0, 0.0, math.nan, 1.0])}), b)
    pa_, pb = builder._float_profile([a], ["v"]), builder._float_profile([b], ["v"])
    assert (
        pa_["v"]["neg_zero"] == 1 and pa_["v"]["pos_zero"] == 1 and pa_["v"]["nan"] == 1
    )
    assert pb["v"]["neg_zero"] == 0 and pb["v"]["pos_zero"] == 2
    assert pa_["v"]["sum_hash"] != pb["v"]["sum_hash"]
    shuffled = tmp_path / "c.parquet"
    pq.write_table(pa.table({"v": pa.array([1.0, math.nan, -0.0, 0.0])}), shuffled)
    assert builder._float_profile([shuffled], ["v"]) == pa_  # order-independent


def test_table_without_float_columns_keeps_dictionary_default(tmp_path):
    assert "DICTIONARY_SIZE_LIMIT" not in builder._parquet_options(
        [["itemid", "BIGINT"], ["label", "VARCHAR"]]
    )
    assert "DICTIONARY_SIZE_LIMIT 0" in builder._parquet_options(
        [["itemid", "BIGINT"], ["v", "DOUBLE"]]
    )


def test_float_bit_mismatch_fails_the_build_typed(tmp_path, monkeypatch):
    _source(tmp_path)
    # Reintroduce the dictionary encoder for float columns: signed zeros collapse.
    monkeypatch.setattr(
        builder,
        "_parquet_options",
        lambda schema, *extra: ", ".join(
            ["FORMAT PARQUET", *extra, "COMPRESSION ZSTD"]
        ),
    )
    with pytest.raises(builder.FloatFidelityError) as caught:
        builder.build_bucket_cache(
            tmp_path, table="chartevents", key="itemid", bucket_count=4, threads=2
        )
    assert caught.value.reason == "bucket_float_fidelity_mismatch"
    assert caught.value.column in ("valuenum", "weight")
    assert caught.value.source["neg_zero"] != caught.value.target["neg_zero"]
    assert not (tmp_path / "chartevents_bucket").exists()
    assert not (tmp_path / ".chartevents_bucket_build").exists()
