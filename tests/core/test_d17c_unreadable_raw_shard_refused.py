"""A corrupt or unreadable raw Parquet shard is a typed refusal, never excluded (delta17c).

The partitioned raw-source loader built its PyArrow dataset with
``exclude_invalid_files=True``: when the DuckDB read failed on a corrupt shard,
the fallback scan dropped that shard and returned the other shards' rows.
Now the scan reads every resolved file, and the per-file read names the bad
one: ``TableScanError("table_scan_file_unreadable")`` with the reader's error as
its cause. The benchmark preloader, the batch spill merge and the stay-table
column synthesis follow the same rule. A clean partitioned table reads as
before. Rows are synthetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.storage_layout import TableScanError
from tests.support.bucket_storage import b6_source

FILE_UNREADABLE = "table_scan_file_unreadable"  # the stable reason code
TABLE = "chartevents"
COLUMNS = ["stay_id", "itemid", "valuenum"]


def _corrupt(path) -> None:
    path.write_bytes(b"PAR1" + b"\x00not a parquet file\x00" * 8 + b"PAR1")


@pytest.fixture
def partitioned(tmp_path):
    root = tmp_path / "miiv"
    source = b6_source(root, "miiv")
    first = pd.read_parquet(root / TABLE / "1.parquet")
    first.assign(valuenum=first["valuenum"] + 1000).to_parquet(root / TABLE / "2.parquet", index=False)
    return source, root


def _reference(root) -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=COLUMNS) for p in sorted((root / TABLE).glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[COLUMNS].sort_values(COLUMNS, kind="mergesort").reset_index(drop=True)


def test_a_clean_partitioned_table_reads_every_shard(partitioned):
    source, root = partitioned
    direct = source._read_partitioned_data_optimized(root / TABLE, COLUMNS, None, table_name=TABLE)
    loaded = source.load_table(TABLE, columns=COLUMNS).data
    expected = _sorted(_reference(root))
    pd.testing.assert_frame_equal(_sorted(direct), expected, check_dtype=False)
    pd.testing.assert_frame_equal(_sorted(loaded), expected, check_dtype=False)


def test_a_corrupt_shard_in_the_pyarrow_scan_is_refused_typed(partitioned):
    source, root = partitioned
    _corrupt(root / TABLE / "3.parquet")
    with pytest.raises(TableScanError) as error:
        source._read_partitioned_data_optimized(root / TABLE, COLUMNS, None, table_name=TABLE)
    assert error.value.reason == FILE_UNREADABLE and "3.parquet" in str(error.value)
    assert error.value.__cause__ is not None


def test_a_corrupt_shard_is_refused_through_load_table(partitioned):
    """DuckDB fails on the shard; its PyArrow fallback must not drop it."""
    source, root = partitioned
    _corrupt(root / TABLE / "3.parquet")
    with pytest.raises(TableScanError) as error:
        source.load_table(TABLE, columns=COLUMNS)
    assert error.value.reason == FILE_UNREADABLE and "3.parquet" in str(error.value)


def test_the_benchmark_preloader_refuses_a_corrupt_shard(partitioned):
    from easyicu.runtime.performance import TablePreloader

    _source, root = partitioned
    preloader = TablePreloader(root, enable_preload=True)
    preloader.preload_for_patients([1, 2], {TABLE: COLUMNS})
    assert len(preloader.get_table(TABLE)) == len(_reference(root))
    (root / TABLE / "note.fst").write_bytes(b"not parquet")  # a sibling format is not a shard
    preloader.clear_cache()
    preloader.preload_for_patients([1, 2], {TABLE: COLUMNS})
    assert len(preloader.get_table(TABLE)) == len(_reference(root))
    _corrupt(root / TABLE / "3.parquet")
    preloader.clear_cache()
    with pytest.raises(TableScanError) as error:
        preloader.preload_for_patients([1, 2], {TABLE: COLUMNS})
    assert error.value.reason == FILE_UNREADABLE


def test_an_unreadable_spilled_batch_fails_the_merge(tmp_path):
    from easyicu.runtime.memory_manager import _merge_parquet_batches

    pd.DataFrame({"stay_id": [1], "value": [2.0]}).to_parquet(tmp_path / "batch_0000.parquet")
    _corrupt(tmp_path / "batch_0001.parquet")
    with pytest.raises(RuntimeError, match="batch_spill_unreadable: batch_0001.parquet"):
        _merge_parquet_batches(str(tmp_path))


@pytest.mark.parametrize("where", ["resolver_footer", "synthesis_read"])
def test_a_corrupt_stay_table_is_refused_by_column_synthesis(tmp_path, monkeypatch, where):
    """Its read used to log a warning and leave the column unsynthesized."""
    from easyicu.concept import ConceptResolver

    root = tmp_path / "miiv"
    source = b6_source(root, "miiv")
    stays = pd.read_parquet(root / "icustays.parquet", columns=["stay_id"])
    _corrupt(root / "icustays.parquet")
    if where == "synthesis_read":  # past the resolver: the synthesis read itself refuses
        monkeypatch.setattr(source, "resolve_target_files", lambda _table, loader: (loader,))
    with pytest.raises(TableScanError) as error:
        ConceptResolver._synthesise_los_column(SimpleNamespace(), "outtime", source, stays)
    assert error.value.reason == FILE_UNREADABLE


def test_a_corrupt_shard_in_the_flat_scan_footer_check_is_refused_typed(partitioned):
    from easyicu.storage_layout import verify_flat_scan

    source, root = partitioned
    _corrupt(root / TABLE / "3.parquet")
    with pytest.raises(TableScanError) as error:
        verify_flat_scan(source, TABLE, sorted((root / TABLE).glob("*.parquet")))
    assert error.value.reason == FILE_UNREADABLE and error.value.__cause__ is not None
