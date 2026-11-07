"""Bucket builds keep every floating-point bit of the source, bound to the record (lead-r13 r13bitexact).

DuckDB's parquet dictionary encoder keys floats by equality, so -0.0 and +0.0
would share one entry and come back with one sign. The builder writes tables
with float columns without dictionary encoding, and its QC binds every float
value to its source ordinal: the source is read in the builder's ordinal order
and cast to the flat reader's unified type, the bucket is read ordered by the
ordinal, and null positions and bit patterns are compared row by row. Any
difference (including corruption that cancels across buckets or swaps signs
between records) fails the build with a typed reason, and adoption of a legacy
cache runs the same gate. Rows are synthetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.build_itemid_bucket_cache as builder
from easyicu.storage_layout import MANIFEST, ORDINAL, read_bucket_contract


def _values(n: int = 20_000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.choice([0.0, 1.5, 2.5, 7.0], n)
    value[rng.random(n) < 0.05] = -0.0
    value[0] = -0.0  # a negative zero reaches the dictionary first in some order
    value[rng.random(n) < 0.01] = np.nan
    return value


def _table(value: np.ndarray, seed: int = 7) -> pa.Table:
    rng = np.random.default_rng(seed + 1)
    n = value.size
    return pa.table(
        {
            "stay_id": pa.array(rng.integers(1, 50, n), pa.int64()),
            "itemid": pa.array(rng.integers(1, 6, n), pa.int64()),
            "charttime": pa.array(np.arange(n), pa.int64()),
            "valuenum": pa.array(value, pa.float64()),
            "weight": pa.array(value.astype(np.float32), pa.float32()),
        }
    )


def _source(root: Path, n: int = 20_000) -> np.ndarray:
    value = _values(n)
    (root / "chartevents").mkdir(parents=True)
    pq.write_table(
        _table(value), root / "chartevents" / "1.parquet", use_dictionary=False
    )
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
        "nulls": 0,
    }


def _build(root: Path, variable: bool = False):
    if variable:
        return builder.build_variable_cache(
            root,
            table="chartevents",
            key="itemid",
            stay_key="stay_id",
            bucket_count=4,
            rows_per_bucket=4096,
            threads=2,
        )
    return builder.build_bucket_cache(
        root, table="chartevents", key="itemid", bucket_count=4, threads=2
    )


@pytest.mark.parametrize("variable", [False, True], ids=["hash", "variable_stay_range"])
def test_bucket_keeps_signed_zero_and_nan_bits(tmp_path, variable):
    value = _source(tmp_path)
    _build(tmp_path, variable)
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
    assert fidelity["rows"] == value.size
    for key, count in _expect_profile(value).items():
        assert fidelity["columns"]["valuenum"][key] == count
    assert len(fidelity["columns"]["valuenum"]["sha256"]) == 64
    receipt = json.loads((root / "_BUCKET_BUILD_RECEIPT.json").read_text())
    assert receipt["float_fidelity"] == fidelity
    qc = receipt["float_fidelity_qc"]
    assert qc["wall_seconds"] >= 0 and qc["batch_rows"] >= 4096
    assert qc["process_peak_rss_kb_after"] >= qc["process_peak_rss_kb_before"] > 0
    read_bucket_contract(root)  # the reader still accepts the manifest


def test_comparison_spans_batch_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(
        builder, "QC_BATCH_BYTES", 1
    )  # 4096-row batches over 20,000 rows
    _source(tmp_path)
    receipt = _build(tmp_path)
    assert receipt["float_fidelity_qc"]["batch_rows"] == 4096
    assert receipt["float_fidelity"]["rows"] == 20_000


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
        _build(tmp_path)
    assert caught.value.reason == "bucket_float_fidelity_mismatch"
    assert caught.value.column in ("valuenum", "weight")
    assert caught.value.source["neg_zero"] != caught.value.target["neg_zero"]
    assert caught.value.source["mismatched_rows"] > 0
    assert not (tmp_path / "chartevents_bucket").exists()
    assert not (tmp_path / ".chartevents_bucket_build").exists()


def _corrupting_verify(monkeypatch, corrupt):
    """After each final bucket write (and its own order check), let ``corrupt`` rewrite the part."""
    original = builder._verify_bucket

    def verify(con, path, sort_keys, expected_rows):
        original(con, path, sort_keys, expected_rows)
        table = pq.read_table(path)
        changed = corrupt(table)
        if changed is not None:
            pq.write_table(changed, path, use_dictionary=False)

    monkeypatch.setattr(builder, "_verify_bucket", verify)


def _with_values(table: pa.Table, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index("valuenum")
    return table.set_column(index, "valuenum", pa.array(values, pa.float64()))


@pytest.mark.parametrize("variable", [False, True], ids=["hash", "variable_stay_range"])
def test_sign_corruption_cancelling_across_buckets_fails(
    tmp_path, monkeypatch, variable
):
    _source(tmp_path)
    state = {"to_positive": False, "to_negative": False}

    def corrupt(table):
        v = table.column("valuenum").to_numpy(zero_copy_only=False).copy()
        neg = np.flatnonzero((v == 0) & np.signbit(v))
        pos = np.flatnonzero((v == 0) & ~np.signbit(v))
        if not state["to_positive"] and neg.size:
            v[neg[0]] = 0.0
            state["to_positive"] = True
            return _with_values(table, v)
        if state["to_positive"] and not state["to_negative"] and pos.size:
            v[pos[0]] = -0.0
            state["to_negative"] = True
            return _with_values(table, v)
        return None

    _corrupting_verify(monkeypatch, corrupt)
    with pytest.raises(builder.FloatFidelityError) as caught:
        _build(tmp_path, variable)
    assert state == {"to_positive": True, "to_negative": True}
    error = caught.value
    assert (
        error.reason == "bucket_float_fidelity_mismatch" and error.column == "valuenum"
    )
    # The per-column marginal counts are unchanged: only the record binding sees it.
    assert error.source["neg_zero"] == error.target["neg_zero"]
    assert error.source["pos_zero"] == error.target["pos_zero"]
    assert error.source["mismatched_rows"] == 2
    assert not (tmp_path / "chartevents_bucket").exists()


def test_sign_swap_between_records_in_one_bucket_fails(tmp_path, monkeypatch):
    _source(tmp_path)
    done = []

    def corrupt(table):
        if done:
            return None
        v = table.column("valuenum").to_numpy(zero_copy_only=False).copy()
        neg = np.flatnonzero((v == 0) & np.signbit(v))
        pos = np.flatnonzero((v == 0) & ~np.signbit(v))
        if not (neg.size and pos.size):
            return None
        v[neg[0]], v[pos[0]] = 0.0, -0.0
        done.append(True)
        return _with_values(table, v)

    _corrupting_verify(monkeypatch, corrupt)
    with pytest.raises(builder.FloatFidelityError) as caught:
        _build(tmp_path)
    assert done and caught.value.source["mismatched_rows"] == 2
    assert caught.value.source["neg_zero"] == caught.value.target["neg_zero"]


@pytest.mark.parametrize(
    "first,second",
    [(pa.float32(), pa.float64()), (pa.int64(), pa.float64())],
    ids=["float_and_double_shards", "int64_and_double_shards"],
)
def test_mixed_dtype_shards_build_as_the_flat_reader_reads_them(
    tmp_path, first, second
):
    value = _values(6000)
    table = _table(value)
    directory = tmp_path / "chartevents"
    directory.mkdir()
    halves = (table.slice(0, 3000), table.slice(3000))
    for i, (part, kind) in enumerate(zip(halves, (first, second))):
        column = part.column("valuenum")
        if pa.types.is_integer(kind):
            column = pa.array(
                np.round(column.to_numpy(zero_copy_only=False) * 0).astype(np.int64)
            )
        else:
            column = column.cast(kind)
        part = part.set_column(
            part.schema.get_field_index("valuenum"), "valuenum", column
        )
        pq.write_table(part, directory / f"{i}.parquet", use_dictionary=False)
    receipt = _build(tmp_path)
    flat = (
        duckdb.sql(
            f"SELECT valuenum FROM read_parquet('{directory}/*.parquet', union_by_name=true, "
            "filename=true, file_row_number=true) ORDER BY filename, file_row_number"
        )
        .fetch_arrow_table()
        .column(0)
        .combine_chunks()
    )
    bucket = (
        duckdb.sql(
            f"SELECT valuenum FROM read_parquet('{tmp_path}/chartevents_bucket/bucket_id=*/*.parquet') "
            f'ORDER BY "{ORDINAL}"'
        )
        .fetch_arrow_table()
        .column(0)
        .combine_chunks()
    )
    assert flat.type == bucket.type == pa.float64()
    as_bits = lambda a: a.fill_null(0).to_numpy(zero_copy_only=False).view(np.uint64)  # noqa: E731
    assert np.array_equal(as_bits(flat), as_bits(bucket))
    assert receipt["float_fidelity"]["columns"]["valuenum"]["values"] == 6000


def test_a_shard_type_outside_the_widening_rules_is_refused_typed(
    tmp_path, monkeypatch
):
    value = _values(4000)
    table = _table(value)
    directory = tmp_path / "chartevents"
    directory.mkdir()
    pq.write_table(table.slice(0, 2000), directory / "0.parquet", use_dictionary=False)
    narrow = table.slice(2000)
    narrow = narrow.set_column(
        narrow.schema.get_field_index("valuenum"),
        "valuenum",
        narrow.column("valuenum").cast(pa.float32()),
    )
    pq.write_table(narrow, directory / "1.parquet", use_dictionary=False)
    monkeypatch.setitem(builder._WIDENING, "DOUBLE", frozenset({"DOUBLE"}))
    with pytest.raises(builder.FloatFidelityError) as caught:
        _build(tmp_path)
    assert caught.value.reason == "bucket_float_source_type_unsupported"
    assert caught.value.source == {"shard_type": "FLOAT"}
    assert not (tmp_path / "chartevents_bucket").exists()


def test_a_nested_float_leaf_fails_closed(tmp_path):
    value = _values(2000)
    table = _table(value).append_column(
        "readings", pa.array([[float(v), 1.0] for v in value], pa.list_(pa.float64()))
    )
    (tmp_path / "chartevents").mkdir()
    pq.write_table(table, tmp_path / "chartevents" / "1.parquet", use_dictionary=False)
    with pytest.raises(builder.FloatFidelityError) as caught:
        _build(tmp_path)
    assert caught.value.reason == "unsupported_nested_float"
    assert caught.value.column == "readings"
    assert not (tmp_path / "chartevents_bucket").exists()
    assert not (tmp_path / ".chartevents_bucket_build").exists()


# ---- adoption of a legacy cache (no ordinal): the same gate, bound to the whole record


def _legacy_cache(root: Path, *, flip_sign: bool) -> Path:
    value = _values(8000)
    table = _table(value).drop_columns(["weight"])
    (root / "numericitems").mkdir(parents=True)
    source = root / "numericitems" / "1.parquet"
    pq.write_table(table, source, use_dictionary=False)
    cache = root / "numericitems_bucket"
    cache.mkdir()
    relation = builder._relation([source])
    with duckdb.connect() as con:
        flipped = False
        for bid in range(4):
            part = con.execute(
                f"SELECT * FROM {relation} WHERE hash(itemid) % 4 = {bid}"
            ).fetch_arrow_table()
            if not part.num_rows:
                continue
            v = part.column("valuenum").to_numpy(zero_copy_only=False).copy()
            neg = np.flatnonzero((v == 0) & np.signbit(v))
            if flip_sign and not flipped and neg.size:
                v[neg[0]] = 0.0  # a historical dictionary collapse of one -0.0
                flipped = True
                part = part.set_column(
                    part.schema.get_field_index("valuenum"),
                    "valuenum",
                    pa.array(v, pa.float64()),
                )
            (cache / f"bucket_id={bid}").mkdir()
            pq.write_table(
                part, cache / f"bucket_id={bid}" / "part.parquet", use_dictionary=False
            )
        assert flipped or not flip_sign
        schema = builder._schema(con, relation)
        fingerprint = builder._row_fingerprint(con, relation, [c for c, _ in schema])
    (cache / "_BUCKET_BUILD_RECEIPT.json").write_text(
        json.dumps(
            dict(
                schema=builder.RECEIPT_SCHEMA,
                key="itemid",
                bucket_count=4,
                duckdb_version=duckdb.__version__,
                source_schema=schema,
                rows=table.num_rows,
                row_multiset_fingerprint=fingerprint,
            )
        )
    )
    (cache / "_COMPLETE").write_text("complete\n")
    return cache


def test_adoption_runs_the_fidelity_gate_and_records_it(tmp_path):
    cache = _legacy_cache(tmp_path, flip_sign=False)
    manifest = builder.adopt_bucket_cache(tmp_path, threads=1, memory_limit="256MB")
    adoption = json.loads((cache / builder.ADOPTION_RECEIPT).read_text())
    assert manifest["float_fidelity"] == adoption["float_fidelity"]
    assert adoption["float_fidelity"]["columns"]["valuenum"]["neg_zero"] > 0
    assert adoption["float_fidelity_qc"]["wall_seconds"] >= 0
    assert read_bucket_contract(cache).modulus == 4


def test_adoption_of_a_sign_corrupted_legacy_cache_is_refused(tmp_path):
    cache = _legacy_cache(tmp_path, flip_sign=True)
    with pytest.raises(builder.FloatFidelityError) as caught:
        builder.adopt_bucket_cache(tmp_path, threads=1, memory_limit="256MB")
    assert caught.value.reason == "bucket_float_fidelity_mismatch"
    assert caught.value.source == {"rows_not_in_bucket": 1}
    assert caught.value.target == {"rows_not_in_source": 1}
    assert not (cache / MANIFEST).exists()
    assert not (cache / builder.ADOPTION_RECEIPT).exists()


def test_profile_counts_nan_and_signed_zero_per_record():
    profile = builder._ColumnProfile()
    values = np.array([0.0, -0.0, math.nan, 1.0, 0.0])
    valid = np.array([True, True, True, True, False])
    profile.update(np.arange(5), valid, values, values.view(np.uint64))
    result = profile.result()
    assert (
        result["values"],
        result["nulls"],
        result["neg_zero"],
        result["pos_zero"],
        result["nan"],
    ) == (4, 1, 1, 1, 1)
    swapped = builder._ColumnProfile()
    other = np.array([-0.0, 0.0, math.nan, 1.0, 0.0])
    swapped.update(np.arange(5), valid, other, other.view(np.uint64))
    assert (
        swapped.result()["sha256"] != result["sha256"]
    )  # same marginals, records differ


def _tied_legacy_cache(root: Path, *, flip: bool) -> tuple[Path, Path]:
    """The delta26b tie probe: records equal apart from the sign of a zero.

    Every non-float field ties; the bucket flips every zero's sign, which keeps
    the signed complete-row multiset but changes FIRST() over the bucket.
    """
    n = 12_000
    value = np.tile(np.array([-0.0, 0.0]), n // 2)
    table = pa.table(
        {
            "itemid": pa.array([1] * n, pa.int64()),
            "seq": pa.array([10] * n, pa.int64()),
            "v": pa.array(value, pa.float64()),
        }
    )
    (root / "chartevents").mkdir(parents=True)
    source = root / "chartevents" / "part.parquet"
    pq.write_table(table, source, use_dictionary=False)
    cache = root / "chartevents_bucket"
    (cache / "bucket_id=0").mkdir(parents=True)
    stored = -value if flip else value
    pq.write_table(
        table.set_column(2, "v", pa.array(stored, pa.float64())),
        cache / "bucket_id=0" / "part.parquet",
        use_dictionary=False,
    )
    with duckdb.connect() as con:
        schema = builder._schema(con, builder._relation([source]))
    (cache / "_BUCKET_BUILD_RECEIPT.json").write_text(
        json.dumps(
            {
                "schema": builder.RECEIPT_SCHEMA,
                "key": "itemid",
                "bucket_count": 1,
                "duckdb_version": duckdb.__version__,
                "source_schema": schema,
                "rows": n,
            }
        )
    )
    (cache / "_COMPLETE").write_text("complete\n")
    return source, cache


def test_adoption_refuses_records_tied_apart_from_a_zero_sign(tmp_path):
    _, cache = _tied_legacy_cache(tmp_path, flip=True)
    with pytest.raises(builder.FloatFidelityError) as caught:
        builder.adopt_bucket_cache(
            tmp_path, table="chartevents", key="itemid", threads=1, memory_limit="256MB"
        )
    assert caught.value.reason == "adopt_sign_ambiguous_rebuild_required"
    assert caught.value.source == {"sign_ambiguous_groups": 1}
    assert not (cache / MANIFEST).exists()
    assert not (cache / builder.ADOPTION_RECEIPT).exists()


def test_adoption_refuses_ambiguous_ties_even_when_the_bucket_is_unchanged(tmp_path):
    # Without an ordinal the tie cannot be certified, whatever the stored order.
    _, cache = _tied_legacy_cache(tmp_path, flip=False)
    with pytest.raises(builder.FloatFidelityError) as caught:
        builder.adopt_bucket_cache(
            tmp_path, table="chartevents", key="itemid", threads=1, memory_limit="256MB"
        )
    assert caught.value.reason == "adopt_sign_ambiguous_rebuild_required"
    assert not (cache / MANIFEST).exists()


def test_an_ordinal_rebuild_of_the_tied_source_passes(tmp_path):
    source, _ = _tied_legacy_cache(tmp_path / "legacy", flip=False)
    root = tmp_path / "rebuild"
    (root / "chartevents").mkdir(parents=True)
    (root / "chartevents" / "part.parquet").write_bytes(source.read_bytes())
    receipt = builder.build_bucket_cache(
        root, table="chartevents", key="itemid", bucket_count=1, threads=1
    )
    assert receipt["float_fidelity"]["columns"]["v"]["neg_zero"] == 6000


def _expected_src() -> Path:
    return (Path(builder.__file__).resolve().parents[1] / "src").resolve()


@pytest.mark.parametrize("variable", [False, True], ids=["hash", "variable_stay_range"])
def test_build_receipt_records_an_import_origin_inside_this_checkout(
    tmp_path, variable
):
    _source(tmp_path, n=2_000)
    _build(tmp_path, variable)
    receipt = json.loads(
        (tmp_path / "chartevents_bucket" / "_BUCKET_BUILD_RECEIPT.json").read_text()
    )
    origin = receipt["import_origin"]
    assert origin["expected_src"] == str(_expected_src())
    assert set(origin["modules"]) == {
        "easyicu",
        "easyicu.storage_layout",
        "easyicu.runtime.duckdb_spill",
    }
    for path in origin["modules"].values():
        assert Path(path).is_relative_to(_expected_src()), path


def test_adoption_receipt_records_the_import_origin(tmp_path):
    cache = _legacy_cache(tmp_path, flip_sign=False)
    builder.adopt_bucket_cache(tmp_path, threads=1, memory_limit="256MB")
    adoption = json.loads((cache / builder.ADOPTION_RECEIPT).read_text())
    assert adoption["import_origin"] == builder.builder_import_origin()


@pytest.mark.parametrize("entry", ["hash", "variable_stay_range", "adopt"])
def test_easyicu_from_another_checkout_is_refused_before_any_write(
    tmp_path, monkeypatch, entry
):
    """An editable install elsewhere (e.g. a webapp runtime) must not supply easyicu."""
    elsewhere = tmp_path / "webapp" / "src" / "easyicu" / "storage_layout.py"
    monkeypatch.setattr(builder._storage_layout, "__file__", str(elsewhere))
    if entry == "adopt":
        cache = _legacy_cache(tmp_path, flip_sign=False)
        before = sorted(p.name for p in cache.iterdir())
    else:
        _source(tmp_path, n=2_000)
    with pytest.raises(builder.ImportOriginError) as caught:
        if entry == "adopt":
            builder.adopt_bucket_cache(tmp_path, threads=1, memory_limit="256MB")
        else:
            _build(tmp_path, entry == "variable_stay_range")
    assert caught.value.reason == "builder_import_origin_mismatch"
    assert caught.value.expected == str(_expected_src())
    assert caught.value.modules["easyicu.storage_layout"] == str(elsewhere)
    if entry == "adopt":
        assert sorted(p.name for p in cache.iterdir()) == before
    else:
        assert not (tmp_path / "chartevents_bucket").exists()
        assert not list(tmp_path.glob(".*bucket*"))


def _git(*args: str) -> str:
    import subprocess

    repo = Path(builder.__file__).resolve().parents[1]
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.mark.parametrize("entry", ["hash", "variable_stay_range", "adopt"])
def test_manifest_and_receipt_carry_the_same_builder_code(tmp_path, entry):
    """The seal tool requires manifest builder_code == receipt builder_code, commit == SHA."""
    if entry == "adopt":
        root = _legacy_cache(tmp_path, flip_sign=False)
        builder.adopt_bucket_cache(tmp_path, threads=1, memory_limit="256MB")
        receipt_name = builder.ADOPTION_RECEIPT
    else:
        _source(tmp_path, n=2_000)
        _build(tmp_path, entry == "variable_stay_range")
        root = tmp_path / "chartevents_bucket"
        receipt_name = "_BUCKET_BUILD_RECEIPT.json"
    manifest = json.loads((root / MANIFEST).read_text())
    receipt = json.loads((root / receipt_name).read_text())
    expected = {
        "source": "git",
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(
            _git(
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--ignored",
                "--ignore-submodules=none",
            )
        ),
    }
    assert manifest["builder_code"] == receipt["builder_code"] == expected
    assert len(expected["commit"]) == 40
    read_bucket_contract(root)  # the reader still accepts the manifest


def test_builder_code_outside_a_git_checkout_has_no_commit(monkeypatch):
    import subprocess

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert builder.builder_code() == {
        "source": "unavailable",
        "commit": None,
        "dirty": None,
    }


def test_builder_source_carries_the_seal_tool_float_fidelity_mark():
    """The seal tool refuses a builder whose source lacks this reason (BUILDER_MARK)."""
    source = Path(builder.__file__).read_text()
    assert "bucket_float_fidelity_mismatch" in source
    assert (
        builder.FloatFidelityError("c", {}, {}).reason
        == "bucket_float_fidelity_mismatch"
    )


def test_an_ignored_file_in_the_checkout_makes_builder_code_dirty(
    tmp_path, monkeypatch
):
    """Matches the seal tool's clean-checkout rule (--ignored), N36-DIRTY."""
    import shutil
    import subprocess

    repo = tmp_path / "checkout"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".gitignore").write_text("*.log\n__pycache__/\n")
    (repo / "scripts" / "build_itemid_bucket_cache.py").write_text("# stand-in\n")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "stand-in")
    monkeypatch.setattr(
        builder, "__file__", str(repo / "scripts" / "build_itemid_bucket_cache.py")
    )
    head = git("rev-parse", "HEAD")
    assert builder.builder_code() == {"source": "git", "commit": head, "dirty": False}
    (repo / "scripts" / "__pycache__").mkdir()
    (repo / "scripts" / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"\0")
    assert builder.builder_code() == {"source": "git", "commit": head, "dirty": True}
    shutil.rmtree(repo / "scripts" / "__pycache__")
    (repo / "build.log").write_text("x\n")
    assert builder.builder_code()["dirty"] is True
