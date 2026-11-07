"""Stay discovery resolves every Parquet layout through the owner before any CSV (delta25d).

B-D25-COHORT-LAYOUT: discovery looked for a single ``<table>.parquet``, then
the CSV, and only then the stay shard directory. A shard directory, healthy
or corrupt, beside a CSV was never seen: the CSV's row set was used without
the owner checking anything. A dangling ``.parquet`` link counted as no
file. Now the storage owner resolves and validates every supported Parquet
layout (single file, shard directory, bucket) first. A dangling link is a
present, broken source and is refused. The CSV is read only when no Parquet
layout exists. Rows are synthetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.storage_layout import StorageIntegrityError

FILE_UNREADABLE = "table_scan_file_unreadable"


def _csv_beside(root):
    pd.DataFrame({"icustay_id": [900, 901]}).to_csv(root / "icustays.csv", index=False)


def _discover(root):
    from easyicu.api import get_all_patient_ids

    listing: dict = {}
    ids, column = get_all_patient_ids(root, database="mimic", listing=listing)
    return ids, column, listing


def test_a_healthy_shard_directory_beside_a_csv_is_the_stay_table(tmp_path):
    shards = tmp_path / "icustays"
    shards.mkdir()
    pd.DataFrame({"icustay_id": [101, 102]}).to_parquet(shards / "part-0.parquet", index=False)
    pd.DataFrame({"icustay_id": [103]}).to_parquet(shards / "part-1.parquet", index=False)
    _csv_beside(tmp_path)
    ids, column, listing = _discover(tmp_path)
    assert (ids, column) == ([101, 102, 103], "icustay_id")
    assert listing == {"order": "source_file_order", "source_format": "parquet"}


def test_a_corrupt_shard_directory_beside_a_csv_is_refused(tmp_path):
    shards = tmp_path / "icustays"
    shards.mkdir()
    (shards / "part-0.parquet").write_bytes(b"not parquet")
    _csv_beside(tmp_path)
    with pytest.raises(StorageIntegrityError) as error:
        _discover(tmp_path)
    assert getattr(error.value, "reason", "") == FILE_UNREADABLE


def test_a_dangling_parquet_link_beside_a_csv_is_refused(tmp_path):
    (tmp_path / "icustays.parquet").symlink_to(tmp_path / "missing_target.parquet")
    _csv_beside(tmp_path)
    with pytest.raises(StorageIntegrityError) as error:
        _discover(tmp_path)
    assert getattr(error.value, "reason", "") == FILE_UNREADABLE


def test_only_an_absent_parquet_layout_falls_back_to_the_csv(tmp_path):
    _csv_beside(tmp_path)
    ids, _column, listing = _discover(tmp_path)
    assert sorted(ids) == [900, 901] and listing["source_format"] == "csv_fallback"
