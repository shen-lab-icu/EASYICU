"""A folder exactly as PhysioNet delivers it is recognized as raw data.

A fresh MIMIC-IV download keeps its tables in ``hosp/`` and ``icu/`` (and a
download of the whole project nests them under a version folder). The
converter walks that tree, so the scan that offers conversion must see it too;
it used to look only at the folder root and answered "no recognized ICU
tables" for every fresh MIMIC-IV download.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.webserver import dataio

_TABLES = {
    "hosp": {
        "patients": pd.DataFrame({"subject_id": [100], "anchor_age": [65]}),
        "admissions": pd.DataFrame({"hadm_id": [10], "subject_id": [100]}),
    },
    "icu": {
        "icustays": pd.DataFrame({"stay_id": [1], "hadm_id": [10], "subject_id": [100]}),
    },
}


def _download(root: Path) -> Path:
    for module, tables in _TABLES.items():
        (root / module).mkdir(parents=True)
        for name, frame in tables.items():
            frame.to_csv(root / module / f"{name}.csv.gz", index=False, compression="gzip")
    return root


def test_a_fresh_mimic_iv_download_is_raw_data_to_convert(tmp_path: Path) -> None:
    result = dataio.scan_path(str(_download(tmp_path / "mimic-iv-3.1")))
    assert result["ok"] is True
    assert result["db_key"] == "miiv"
    assert result["source"] == "raw"
    assert result["tables"] == 3
    assert result["ready"] is False
    assert result["size_hint"]
    assert result["privacy"] == {"raw_rows_read": False, "patient_identifiers_returned": False}


def test_a_download_under_its_version_folder_is_recognized(tmp_path: Path) -> None:
    root = tmp_path / "mimic-iv"
    _download(root / "3.1")
    result = dataio.scan_path(str(root))
    assert result["ok"] is True
    assert (result["source"], result["tables"]) == ("raw", 3)


def test_the_scan_stops_two_folders_down_and_skips_hidden_entries(tmp_path: Path) -> None:
    root = tmp_path / "deep"
    _download(root / "a" / "b")
    frame = pd.DataFrame({"x": [1]})
    (root / ".cache").mkdir()
    frame.to_csv(root / ".cache" / "icustays.csv.gz", index=False, compression="gzip")
    frame.to_csv(root / ".hidden.csv", index=False)
    # Tables three folders down are not part of this download's layout.
    assert dataio._raw_table_files(root) == []


def test_a_flat_raw_folder_is_still_recognized(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    root.mkdir()
    for tables in _TABLES.values():
        for name, frame in tables.items():
            frame.to_csv(root / f"{name}.csv.gz", index=False, compression="gzip")
    result = dataio.scan_path(str(root))
    assert (result["ok"], result["source"], result["tables"]) == (True, "raw", 3)
