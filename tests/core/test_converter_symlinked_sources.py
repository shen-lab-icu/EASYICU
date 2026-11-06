"""The converter follows symlinked source folders and never converts nothing silently.

Raw data often sits on another drive with its folders linked into the
database directory (``hosp -> /Volumes/drive/mimiciv/hosp``). The converter
used to walk without following links, found no CSV, wrote an empty manifest
and reported success.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest

from easyicu.datasource import ICUDataSource
from easyicu.io.data_converter import (
    MANIFEST_MISSING_REBUILD_REQUIRED,
    ConversionNoSourceFiles,
    DataConverter,
)
from easyicu.resources import load_data_sources
from easyicu.storage_layout import scan_fallbacks


def _csv(path: Path, rows: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["subject_id,value"] + [f"{i},{i * 1.5}" for i in range(1, rows + 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _converter(path: Path) -> DataConverter:
    return DataConverter(path, database="miiv", parallel_workers=1, verbose=False)


def test_csvs_under_a_symlinked_folder_are_found_and_converted(tmp_path):
    elsewhere = tmp_path / "drive" / "hosp"
    _csv(elsewhere / "admissions.csv")
    _csv(elsewhere / "patients.csv")
    root = tmp_path / "miiv"
    root.mkdir()
    os.symlink(elsewhere, root / "hosp")

    converter = _converter(root)
    names = sorted(p.name for p in converter._get_csv_files())
    assert names == ["admissions.csv", "patients.csv"]

    results = converter.convert_all()
    assert len(results) == 2
    assert (root / "admissions.parquet").exists()
    assert (root / "patients.parquet").exists()
    manifest = json.loads((root / "conversion_manifest.json").read_text())
    assert sorted(t["table"] for t in manifest["tables"]) == ["admissions", "patients"]


def test_a_symlink_cycle_terminates_and_lists_each_csv_once(tmp_path):
    root = tmp_path / "miiv"
    _csv(root / "a" / "admissions.csv")
    os.symlink(root, root / "a" / "loop")  # a/loop -> root -> a -> loop ...
    os.symlink(root / "a", root / "a" / "self")

    files = _converter(root)._get_csv_files()
    assert [p.name for p in files] == ["admissions.csv"]


def test_two_links_to_one_folder_do_not_duplicate_files(tmp_path):
    shared = tmp_path / "drive" / "icu"
    _csv(shared / "icustays.csv")
    root = tmp_path / "miiv"
    root.mkdir()
    os.symlink(shared, root / "icu")
    os.symlink(shared, root / "icu_again")

    files = _converter(root)._get_csv_files()
    assert [p.name for p in files] == ["icustays.csv"]


def test_an_empty_folder_refuses_conversion(tmp_path):
    root = tmp_path / "miiv"
    root.mkdir()
    with pytest.raises(ConversionNoSourceFiles) as excinfo:
        _converter(root).convert_all()
    assert excinfo.value.reason == "conversion_no_source_files"
    assert str(root) in str(excinfo.value)
    assert not (root / "conversion_manifest.json").exists()


def test_a_dangling_source_link_refuses_conversion(tmp_path):
    root = tmp_path / "miiv"
    root.mkdir()
    os.symlink(tmp_path / "unmounted" / "hosp", root / "hosp")
    with pytest.raises(ConversionNoSourceFiles):
        _converter(root).convert_all()


def _admissions(root: Path) -> Path:
    path = root / "hosp" / "admissions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "subject_id,hadm_id,admittime\n10,100,2150-01-01 08:00:00\n20,200,2150-02-01 08:00:00\n",
        encoding="utf-8",
    )
    return path


def _load_admissions(root: Path):
    source = ICUDataSource(config=load_data_sources().get("miiv"), base_path=root)
    return source.load_table("admissions").data


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_converted_folder_without_csvs_keeps_its_manifest_and_loads(tmp_path):
    root = tmp_path / "miiv"
    source = _admissions(root)
    _converter(root).convert_all()
    manifest = root / "conversion_manifest.json"
    before = _sha(manifest)
    source.unlink()

    converter = _converter(root)
    assert converter.convert_all() == {}
    assert converter.conversion_warnings == []
    # The conversion's inventory is kept; an empty manifest would disown it.
    assert _sha(manifest) == before
    assert sorted(_load_admissions(root)["hadm_id"]) == [100, 200]


def test_a_missing_manifest_without_sources_asks_for_a_rebuild(tmp_path, caplog):
    root = tmp_path / "miiv"
    source = _admissions(root)
    _converter(root).convert_all()
    (root / "conversion_manifest.json").unlink()
    source.unlink()

    converter = _converter(root)
    with caplog.at_level(logging.WARNING, logger="easyicu.io.data_converter"):
        assert converter.convert_all() == {}
    assert not (root / "conversion_manifest.json").exists()
    assert [w["code"] for w in converter.conversion_warnings] == [MANIFEST_MISSING_REBUILD_REQUIRED]
    assert MANIFEST_MISSING_REBUILD_REQUIRED in caplog.text
    assert "convert_all()" in converter.conversion_warnings[0]["message"]
    # The loader still reads the table, through its recorded no-manifest fallback.
    assert sorted(_load_admissions(root)["hadm_id"]) == [100, 200]
    assert {"reason": "table_scan_unmanifested", "root": str(root),
            "table": "admissions"} in scan_fallbacks()
