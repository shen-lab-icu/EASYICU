"""Outcome raw tables read the storage policy's validated file set.

``outcomes._raw_table`` listed ``<table>/**/*.parquet`` itself, so a nested
leftover set was concatenated into the stay table (rows counted twice) and a
missing chunk went unnoticed (final review U01).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.scores.outcomes import _raw_table
from easyicu.storage_layout import StorageIntegrityError


def _stays(ids):
    return pd.DataFrame({"icustay_id": ids, "hadm_id": [i * 10 for i in ids]})


def _install(root):
    (root / "icustays").mkdir()
    _stays([1, 2]).to_parquet(root / "icustays" / "1.parquet")
    _stays([3]).to_parquet(root / "icustays" / "2.parquet")
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": [{
        "table": "icustays", "row_count": 3,
        "outputs": [{"relative_path": "icustays/1.parquet"},
                    {"relative_path": "icustays/2.parquet"}],
    }]}))


def test_a_partitioned_table_is_read_once_in_file_order(tmp_path):
    _install(tmp_path)
    assert _raw_table("mimic", tmp_path, "icustays")["icustay_id"].tolist() == [1, 2, 3]


def test_a_nested_copy_is_not_concatenated_into_the_table(tmp_path):
    _install(tmp_path)
    nested = tmp_path / "icustays" / "rebuild"
    nested.mkdir()
    _stays([1, 2, 3]).to_parquet(nested / "1.parquet")
    assert _raw_table("mimic", tmp_path, "icustays")["icustay_id"].tolist() == [1, 2, 3]
    # Without a manifest the directory inventory cannot attribute the copy.
    (tmp_path / "conversion_manifest.json").unlink()
    with pytest.raises(StorageIntegrityError) as caught:
        _raw_table("mimic", tmp_path, "icustays")
    assert caught.value.reason == "table_scan_incomplete"


def test_a_missing_chunk_fails_the_outcome_read(tmp_path):
    _install(tmp_path)
    (tmp_path / "icustays" / "2.parquet").unlink()
    with pytest.raises(StorageIntegrityError) as caught:
        _raw_table("mimic", tmp_path, "icustays")
    assert caught.value.reason == "table_scan_manifest_mismatch"


def test_a_single_file_table_reads_as_before(tmp_path):
    _stays([4, 5]).to_parquet(tmp_path / "icustays.parquet")
    assert _raw_table("mimic", tmp_path, "icustays")["icustay_id"].tolist() == [4, 5]


def test_an_absent_table_is_still_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _raw_table("mimic", tmp_path, "icustays")
