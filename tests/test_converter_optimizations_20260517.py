"""Regression tests for the 2026-05-17 converter consolidation pass.

The webapp and the extraction API now share a single converter
(``DataConverter``); ``DuckDBConverter`` / ``bucket_converter`` were
removed. Three things are pinned here:

1. ``MIXED_TYPE_COLUMNS`` keeps eICU ``medication.dosage`` as a string so
   a later text dosage ("2 tablets PO") is not silently dropped when type
   inference picks a numeric type from an all-numeric sample.
2. ``MIXED_TYPE_COLUMNS`` now also pins ``chartevents.value`` — GCS
   components store text ("Spontaneously") alongside numeric vitals;
   without the override the text rows are lost.
3. ``DataConverter._save_status`` / ``_record_status`` are lock-guarded and
   write atomically. ``convert_all()`` runs ``_convert_file()`` on a
   ThreadPoolExecutor, so concurrent unlocked writes could corrupt
   ``.easyicu_conversion_status.json``.

All tests use tiny synthetic fixtures and run without ``--run-real``.
"""
from __future__ import annotations

import gzip
import json
import threading

import duckdb

from easyicu.data_converter import ConversionStatus, DataConverter


def _read_parquet_column_type(parquet_path, column: str) -> str:
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT {column} FROM read_parquet('{parquet_path}') LIMIT 0"
        ).fetchall()
        return {r[0]: r[1] for r in rows}[column].upper()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Fix 1 — eICU medication.dosage must be pinned to a string type
# ---------------------------------------------------------------------------
def test_eicu_medication_keeps_text_dosage_rows(tmp_path):
    """An all-numeric `dosage` sample makes type inference pick a numeric
    type; a later text dosage then fails to cast. The MIXED_TYPE_COLUMNS
    override pins `dosage` to string so every row survives.
    """
    csv_text = "medicationid,patientunitstayid,drugstartoffset,dosage\n"
    for i in range(15):
        csv_text += f"{i + 1},100,{i * 10},{2 + i}\n"
    csv_text += "16,100,200,2 tablets PO\n"

    src = tmp_path / "medication.csv"
    src.write_text(csv_text)

    converter = DataConverter(str(tmp_path), database="eicu", verbose=False)
    results = converter.convert_all()

    result = results["medication.csv"]
    assert result["status"] == ConversionStatus.COMPLETED, result.get("error")
    assert result["row_count"] == 16, "text-dosage row must not be dropped"

    out = tmp_path / "medication.parquet"
    assert _read_parquet_column_type(out, "dosage").startswith("VARCHAR")

    con = duckdb.connect()
    try:
        rows = con.execute(
            f"SELECT dosage FROM read_parquet('{out}') WHERE dosage = '2 tablets PO'"
        ).fetchall()
    finally:
        con.close()
    assert rows, "categorical dosage text must be preserved"


def test_override_tolerates_absent_columns(tmp_path):
    """The eICU `medication` override lists dosage/loadingdose/frequency.
    A CSV missing loadingdose+frequency (demo subset / schema drift) must
    still convert, with the present override column pinned to string.
    """
    csv_text = "medicationid,patientunitstayid,dosage\n"
    for i in range(10):
        csv_text += f"{i + 1},100,{1.0 + i}\n"

    src = tmp_path / "medication.csv.gz"
    with gzip.open(src, "wt") as f:
        f.write(csv_text)

    converter = DataConverter(str(tmp_path), database="eicu", verbose=False)
    results = converter.convert_all()

    result = results["medication.csv.gz"]
    assert result["status"] == ConversionStatus.COMPLETED, result.get("error")
    assert result["row_count"] == 10

    out = tmp_path / "medication.parquet"
    assert _read_parquet_column_type(out, "dosage").startswith("VARCHAR")


# ---------------------------------------------------------------------------
# Fix 2 — chartevents.value carries GCS-component text and must stay string
# ---------------------------------------------------------------------------
def test_chartevents_value_keeps_text_rows(tmp_path):
    """chartevents.value mixes numeric vitals with GCS-component text. With
    the value column pinned to string the text rows survive conversion.
    """
    csv_text = "subject_id,hadm_id,itemid,value\n"
    for i in range(15):
        csv_text += f"{i + 1},200,220045,{60 + i}\n"
    csv_text += "16,200,223900,Spontaneously\n"

    src = tmp_path / "chartevents.csv"
    src.write_text(csv_text)

    # 'mimic' has no chartevents partitioning config → flat parquet output.
    converter = DataConverter(str(tmp_path), database="mimic", verbose=False)
    results = converter.convert_all()

    result = results["chartevents.csv"]
    assert result["status"] == ConversionStatus.COMPLETED, result.get("error")
    assert result["row_count"] == 16, "GCS-component text row must not be dropped"

    out = tmp_path / "chartevents.parquet"
    assert _read_parquet_column_type(out, "value").startswith("VARCHAR")

    con = duckdb.connect()
    try:
        rows = con.execute(
            f"SELECT value FROM read_parquet('{out}') WHERE value = 'Spontaneously'"
        ).fetchall()
    finally:
        con.close()
    assert rows, "GCS-component text must be preserved"


# ---------------------------------------------------------------------------
# Fix 3 — concurrent status writes must not corrupt the JSON file.
# ---------------------------------------------------------------------------
def test_concurrent_record_status_is_consistent(tmp_path):
    """DataConverter.convert_all runs _convert_file on a ThreadPoolExecutor.
    _record_status / _save_status must serialise the dict mutation and the
    JSON write so the on-disk file is always parseable and complete.

    Stress: 16 threads x 40 distinct keys each. Without the lock json.dumps
    iterating self._status while another worker mutates it would raise, or
    leave a truncated file.
    """
    converter = DataConverter(str(tmp_path), verbose=False)

    n_threads = 16
    per_thread = 40
    errors: list = []

    def worker(tid: int) -> None:
        try:
            for i in range(per_thread):
                key = f"file_{tid}_{i}.csv"
                converter._record_status(
                    key,
                    {"status": "completed", "row_count": tid * 1000 + i},
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent _record_status raised: {errors}"

    expected_keys = {
        f"file_{t}_{i}.csv"
        for t in range(n_threads)
        for i in range(per_thread)
    }
    assert set(converter._status) == expected_keys

    # The on-disk file must be a complete, parseable JSON document with
    # every key present.
    status_file = tmp_path / DataConverter.STATUS_FILE
    on_disk = json.loads(status_file.read_text())
    assert set(on_disk) == expected_keys

    # No leftover temp files from the atomic-write path.
    assert not list(tmp_path.glob(f"{DataConverter.STATUS_FILE}.*.tmp"))
