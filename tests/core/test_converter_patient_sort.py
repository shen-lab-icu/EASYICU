"""Regression tests for the patient-id secondary sort (perf Z2).

Large wide tables (MIMIC-IV chartevents, AUMC numericitems, HiRID
observations) are sharded by itemid/variableid, and each chunk is sorted by
that partition column so parquet row-group zone-maps prune an itemid filter.
A *patient-only* cohort filter (``stay_id IN (...)`` with no itemid) got no
pruning, because the zone-maps carried the full stay_id range of every row
group. The fix adds the patient-id column as a SECONDARY sort key, so within
each itemid the rows are also ordered by patient id and the zone-maps narrow.

These tests convert tiny synthetic fixtures through the real ``DataConverter``
and assert the on-disk shards are ordered by ``(partition_col, patient_id)``
— the property that produces prunable zone-maps. Both the default pandas
id-partition path and the opt-in Arrow path are covered.

The picker helper is unit-tested directly for the case where the partition
column already IS the patient id (eICU vitalperiodic), where no secondary key
should be added.

Run without ``--run-real``.
"""
from __future__ import annotations

import glob

import duckdb
import pytest

from easyicu.io.data_converter import (
    ConversionStatus,
    DataConverter,
    _pick_secondary_sort_column,
)


def _make_chartevents_csv(path):
    """miiv chartevents: partitioned by itemid. Rows are written with
    DESCENDING stay_id within each itemid so an unsorted input can only pass
    the ordering assertion if the converter actually re-sorts.
    """
    # itemids 220045/220046/220047 all fall in partition 1 (< first break
    # 220048); 227240 lands in a later partition. Interleave them.
    lines = ["subject_id,hadm_id,stay_id,itemid,charttime,value"]
    rows = []
    for itemid in (220047, 220045, 220046, 227240):
        for stay_id in (900, 700, 500, 300, 100):  # descending on purpose
            rows.append(f"1,1,{stay_id},{itemid},2020-01-01 00:00:00,{stay_id / 10}")
    # shuffle-ish interleave: write itemids in mixed order
    path.write_text("\n".join(lines + rows) + "\n")


def _shard_paths(shard_dir):
    return sorted(glob.glob(str(shard_dir / "*.parquet")))


def _assert_shards_sorted(shard_dir, partition_col, patient_col):
    con = duckdb.connect()
    try:
        shards = _shard_paths(shard_dir)
        assert shards, f"no shards written under {shard_dir}"
        total = 0
        for shard in shards:
            rows = con.execute(
                f"SELECT {partition_col}, {patient_col} "
                f"FROM read_parquet('{shard}')"
            ).fetchall()
            total += len(rows)
            # rows must be non-decreasing on (partition_col, patient_col)
            assert rows == sorted(rows), (
                f"shard {shard} not ordered by ({partition_col}, {patient_col})"
            )
        return total
    finally:
        con.close()


def test_picker_skips_when_partition_is_patient_id():
    # eICU vitalperiodic is partitioned BY patientunitstayid — no secondary
    # key should be added (the primary sort already gives patient zone-maps).
    assert (
        _pick_secondary_sort_column(
            ["patientunitstayid", "heartrate", "observationoffset"],
            "patientunitstayid",
        )
        is None
    )
    # chartevents: itemid partition -> stay_id secondary
    assert (
        _pick_secondary_sort_column(
            ["subject_id", "hadm_id", "stay_id", "itemid", "value"], "itemid"
        )
        == "stay_id"
    )
    # labevents has no stay_id -> falls back to subject_id
    assert (
        _pick_secondary_sort_column(
            ["subject_id", "hadm_id", "itemid", "value"], "itemid"
        )
        == "subject_id"
    )


def test_id_partition_pandas_path_sorts_by_patient(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_CSV_READER", "pandas")  # default, made explicit
    src = tmp_path / "chartevents.csv"
    _make_chartevents_csv(src)

    converter = DataConverter(str(tmp_path), database="miiv", verbose=False)
    results = converter.convert_all()

    result = results["chartevents.csv"]
    assert result["status"] == ConversionStatus.COMPLETED, result.get("error")

    shard_dir = tmp_path / "chartevents"
    total = _assert_shards_sorted(shard_dir, "itemid", "stay_id")
    assert total == 20  # 4 itemids x 5 stays, nothing dropped


def test_id_partition_arrow_path_sorts_by_patient(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_CSV_READER", "pyarrow")  # opt-in Arrow path
    src = tmp_path / "chartevents.csv"
    _make_chartevents_csv(src)

    converter = DataConverter(str(tmp_path), database="miiv", verbose=False)
    results = converter.convert_all()

    result = results["chartevents.csv"]
    assert result["status"] == ConversionStatus.COMPLETED, result.get("error")

    shard_dir = tmp_path / "chartevents"
    total = _assert_shards_sorted(shard_dir, "itemid", "stay_id")
    assert total == 20
