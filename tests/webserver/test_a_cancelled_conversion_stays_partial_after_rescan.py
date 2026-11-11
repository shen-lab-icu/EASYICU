"""A cancelled conversion is still partial after a rescan or restart.

External review P2-2: the live job said ``partial`` after a cancel, but a later
scan read only the completed receipts and reported the folder clean and ready
for analysis with tables missing. A started conversion now persists the tables
it expects; the scan compares the receipts against that set.
"""

from __future__ import annotations

import json
import time

import pytest

from easyicu.io.data_converter import ConversionStopped, DataConverter
from easyicu.webserver import dataio
from easyicu.webserver.jobs import Job

TABLES = ["patients", "admissions", "icustays", "transfers", "diagnoses_icd", "labevents"]


@pytest.fixture(autouse=True)
def slow_files(monkeypatch):
    original = DataConverter._convert_file

    def slow(self, csv_path):
        time.sleep(0.1)
        return original(self, csv_path)

    monkeypatch.setattr(DataConverter, "_convert_file", slow)


_HEADERS = {
    "patients": "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod",
    "admissions": "subject_id,hadm_id,admittime,dischtime,admission_type",
    "icustays": "subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los",
}


def _folder(tmp_path):
    # Synthetic MIMIC-IV-shaped headers so schema-first detection names the database.
    for name in TABLES:
        header = _HEADERS.get(name, "subject_id,hadm_id,value")
        row = ",".join("1" for _ in header.split(","))
        (tmp_path / f"{name}.csv").write_text(f"{header}\n{row}\n", encoding="utf-8")
    return tmp_path


def test_cancel_then_restart_scan_is_not_ready_until_resume_completes(tmp_path):
    folder = _folder(tmp_path)
    job = Job("convert-cancel", "convert")
    original_emit = job.emit

    def emit(event):
        original_emit(event)
        if event.get("type") == "progress":
            job.cancel_requested = True

    job.emit = emit
    dataio.make_convert_runner(str(folder), "miiv")(job)
    assert (folder / ".easyicu_conversion_pending.json").exists()

    # "Restart": nothing is carried over except the files on disk.
    scan = dataio.scan_path(str(folder))
    quality = scan["conversion_quality"]
    assert scan["ok"] is True
    assert scan["ready"] is False
    assert quality["data_quality_status"] == "partial"
    assert quality["ready_for_analysis"] is False
    assert quality["pending_tables"]
    assert set(quality["pending_tables"]) <= {f"{name}.csv" for name in TABLES}

    # Resume to completion: the pending record is gone and the folder is ready.
    result = dataio.make_convert_runner(str(folder), "miiv")(Job("convert-resume", "convert"))
    assert result["ready_for_analysis"] is True
    assert not (folder / ".easyicu_conversion_pending.json").exists()
    scan = dataio.scan_path(str(folder))
    assert scan["conversion_quality"]["data_quality_status"] == "clean"
    assert "pending_tables" not in scan["conversion_quality"]
    assert scan["ready"] is True


def test_pending_names_match_the_status_keys_and_a_corrupt_record_is_not_ready(tmp_path):
    folder = _folder(tmp_path)
    converter = DataConverter(data_path=str(folder), database="miiv", verbose=False)
    converter.parallel_workers = 1
    with pytest.raises(ConversionStopped):
        converter.convert_all(should_stop=lambda: True)
    expected = json.loads((folder / ".easyicu_conversion_pending.json").read_text())["expected"]
    assert sorted(expected) == sorted(converter.get_conversion_status())
    (folder / ".easyicu_conversion_pending.json").write_text("{", encoding="utf-8")
    quality = dataio._conversion_quality_for_path(folder)
    assert quality["ready_for_analysis"] is False
    assert quality["data_quality_status"] == "unknown"


def test_an_install_without_a_pending_record_keeps_its_clean_status(tmp_path):
    folder = _folder(tmp_path)
    DataConverter(data_path=str(folder), database="miiv", verbose=False).convert_all()
    assert not (folder / ".easyicu_conversion_pending.json").exists()
    assert dataio._conversion_quality_for_path(folder)["ready_for_analysis"] is True
