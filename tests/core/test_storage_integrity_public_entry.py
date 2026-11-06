"""A storage-integrity failure reaches the caller, never "source unavailable".

``TableScanError`` once subclassed ``ValueError``; the concept loader treats a
``ValueError`` from a table read as "this source is not available here" and
moved on, so a converted table with a missing chunk produced an extract with
the concept silently absent (final review B-STORAGE-CATCH).  Every Parquet
backend of ``load_table`` (DuckDB, PyArrow, single file) also reads the
storage policy's validated set (B-STORAGE-INVENTORY).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import easyicu
from easyicu.datasource import FilterOp, FilterSpec, ICUDataSource
from easyicu.resources import load_data_sources
from easyicu.storage_layout import StorageIntegrityError
from tests.support.mimic_chartevents import chart

T = pd.Timestamp("2150-01-01 08:00")


def _icustays(root):
    pd.DataFrame(
        {
            "row_id": [1, 2], "subject_id": [10, 20], "hadm_id": [100, 200],
            "icustay_id": [1000, 2000], "dbsource": ["carevue"] * 2,
            "first_careunit": ["MICU"] * 2, "last_careunit": ["MICU"] * 2,
            "first_wardid": [1, 1], "last_wardid": [1, 1],
            "intime": [T, T], "outtime": [T + pd.Timedelta(days=2)] * 2, "los": [2.0, 2.0],
        }
    ).to_parquet(root / "icustays.parquet")


def _chart(rows):
    return chart(rows, T)


def _install(root, layout):
    """A converted MIMIC-III installation and its conversion manifest."""
    _icustays(root)
    first = _chart([(1, 10, 100, 1000, 80.0), (2, 10, 100, 1000, 82.0)])
    second = _chart([(3, 20, 200, 2000, 70.0)])
    if layout == "file":
        pd.concat([first, second], ignore_index=True).to_parquet(root / "chartevents.parquet")
        chart_outputs = ["chartevents.parquet"]
    else:
        (root / "chartevents").mkdir()
        first.to_parquet(root / "chartevents" / "1.parquet")
        second.to_parquet(root / "chartevents" / "2.parquet")
        chart_outputs = ["chartevents/1.parquet", "chartevents/2.parquet"]
    tables = [
        {"table": "icustays", "row_count": 2, "outputs": [{"relative_path": "icustays.parquet"}]},
        {"table": "chartevents", "row_count": 3,
         "outputs": [{"relative_path": path} for path in chart_outputs]},
    ]
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": tables}))


def _weight(root):
    out = easyicu.load_concepts(
        ["weight"], database="mimic", data_path=str(root),
        patient_ids=[1000, 2000], merge=False, verbose=False,
    )
    return out["weight"] if isinstance(out, dict) else out


@pytest.mark.parametrize("layout", ["file", "directory"])
def test_a_complete_manifested_installation_loads(tmp_path, layout):
    _install(tmp_path, layout)
    out = _weight(tmp_path)
    assert dict(zip(out["icustay_id"], out["weight"])) == {1000: 81.0, 2000: 70.0}


def test_a_missing_chunk_fails_the_public_concept_load(tmp_path):
    _install(tmp_path, "directory")
    (tmp_path / "chartevents" / "2.parquet").unlink()
    with pytest.raises(StorageIntegrityError) as caught:
        _weight(tmp_path)
    assert caught.value.reason == "table_scan_manifest_mismatch"


def test_a_truncated_single_file_fails_the_public_concept_load(tmp_path):
    _install(tmp_path, "file")
    _chart([(1, 10, 100, 1000, 80.0)]).to_parquet(tmp_path / "chartevents.parquet")
    with pytest.raises(StorageIntegrityError) as caught:
        _weight(tmp_path)
    assert caught.value.reason == "table_scan_manifest_mismatch"


@pytest.mark.parametrize("layout", ["file", "directory"])
def test_a_registered_table_that_vanished_fails_the_public_concept_load(tmp_path, layout):
    import shutil

    _install(tmp_path, layout)
    target = tmp_path / ("chartevents.parquet" if layout == "file" else "chartevents")
    target.unlink() if layout == "file" else shutil.rmtree(target)
    with pytest.raises(StorageIntegrityError) as caught:
        _weight(tmp_path)
    assert caught.value.reason == "table_scan_files_missing"


# --- every load_table backend consumes the validated file set ---------------


def _source(root):
    return ICUDataSource(config=load_data_sources().get("mimic"), base_path=root)


def _many_stays(extra=150):
    # More than 100 stays routes a non-bucket directory to the PyArrow reader.
    return FilterSpec(column="icustay_id", op=FilterOp.IN, value=[1000, 2000, *range(5000, 5000 + extra)])


@pytest.mark.parametrize("filters", [None, "many"])
def test_both_directory_readers_read_the_whole_validated_table(tmp_path, filters):
    _install(tmp_path, "directory")
    spec = [_many_stays()] if filters else None
    frame = _source(tmp_path).load_table("chartevents", filters=spec).data
    assert sorted(frame["valuenum"]) == [70.0, 80.0, 82.0]


@pytest.mark.parametrize("filters", [None, "many"])
def test_both_directory_readers_refuse_a_missing_chunk(tmp_path, filters):
    _install(tmp_path, "directory")
    (tmp_path / "chartevents" / "1.parquet").unlink()
    spec = [_many_stays()] if filters else None
    with pytest.raises(StorageIntegrityError):
        _source(tmp_path).load_table("chartevents", filters=spec)


def test_the_single_file_reader_refuses_a_hand_added_table(tmp_path, monkeypatch):
    monkeypatch.delenv("EASYICU_ALLOW_UNREGISTERED_SCAN", raising=False)
    _install(tmp_path, "file")
    pd.DataFrame({"subject_id": [10], "hadm_id": [100]}).to_parquet(tmp_path / "services.parquet")
    with pytest.raises(StorageIntegrityError) as caught:
        _source(tmp_path).load_table("services")
    assert caught.value.reason == "table_scan_unregistered"


def test_a_reader_and_the_policy_disagreeing_fails_closed(tmp_path):
    from easyicu.storage_layout import TableScanError, resolve_table_files

    _install(tmp_path, "directory")
    stray = tmp_path / "elsewhere"
    stray.mkdir()
    _chart([(9, 10, 100, 1000, 1.0)]).to_parquet(stray / "1.parquet")
    with pytest.raises(TableScanError) as caught:
        resolve_table_files(_source(tmp_path), "chartevents", loader=stray)
    assert caught.value.reason == "table_scan_loader_mismatch"
