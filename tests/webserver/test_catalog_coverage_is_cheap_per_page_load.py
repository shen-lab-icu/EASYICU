"""The catalog is fetched on every page load, so its active-export coverage must
not re-read the export each time.

Hosted pilot 2026-10-22: a clinician's 2,000-stay Excel extraction became the
active source, and /api/catalog then took 96 s per page load; every page sat at
its pre-data render meanwhile (Patient Review showed the demo and synthetic
choices on a hospital instance).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import catalog as catalog_module
from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store


def _write_export(root: Path) -> Path:
    root.mkdir()
    tables = {
        "demographics": pd.DataFrame({"stay_id": [1, 2, 3], "age": [50, 70, 60]}),
        "vitals": pd.DataFrame({"stay_id": [1, 1, 2], "hr": [90, 95, 80]}),
    }
    files = []
    for module, frame in tables.items():
        frame.to_csv(root / f"{module}.csv", index=False)
        files.append({"file": f"{module}.csv", "module": module, "rows": len(frame)})
    (root / "_manifest.json").write_text(
        json.dumps({"database": "eicu", "format": "csv", "files": files}), encoding="utf-8"
    )
    return root


@pytest.fixture(autouse=True)
def _empty_cache():
    catalog_module._ROW_COVERAGE_CACHE.clear()
    yield
    catalog_module._ROW_COVERAGE_CACHE.clear()


def test_row_coverage_is_read_once_until_a_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_export(tmp_path / "export")
    monkeypatch.setattr(source_store, "load_registry", lambda: {"active_path": str(export)})
    reads: list[str] = []
    real = catalog_module._file_concept_coverage

    def counting(**kwargs):
        reads.append(kwargs["file_path"].name)
        return real(**kwargs)

    monkeypatch.setattr(catalog_module, "_file_concept_coverage", counting)

    first = catalog_module.build_catalog()["activeExportCoverage"]
    assert first["status"] == "ready"
    assert first["concepts"]["hr"]["coverage_pct"] == 66.7
    read_once = len(reads)
    assert read_once == 2

    again = catalog_module.build_catalog()["activeExportCoverage"]
    assert again == first
    assert len(reads) == read_once

    # A caller cannot change what the next request reads.
    again["concepts"]["hr"]["coverage_pct"] = 0
    assert catalog_module.build_catalog()["activeExportCoverage"]["concepts"]["hr"]["coverage_pct"] == 66.7

    vitals = export / "vitals.csv"
    pd.DataFrame({"stay_id": [1, 2, 3], "hr": [90, 80, 70]}).to_csv(vitals, index=False)
    stat = vitals.stat()
    os.utime(vitals, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    changed = catalog_module.build_catalog()["activeExportCoverage"]
    assert changed["concepts"]["hr"]["coverage_pct"] == 100.0
    assert len(reads) > read_once


def test_an_excel_export_is_covered_by_its_schema_without_reading_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_export(tmp_path / "export")
    desc = dataio.describe_export_source(str(export))
    for meta in desc["files"]:
        meta["file"] = meta["file"].replace(".csv", ".xlsx")

    def no_row_reads(*args, **kwargs):
        raise AssertionError("an Excel export's catalog coverage must not read rows")

    monkeypatch.setattr(source_store, "load_registry", lambda: {"active_path": str(export)})
    monkeypatch.setattr(dataio, "describe_export_source", lambda _path: desc)
    monkeypatch.setattr(dataio, "_fast_stay_ids", no_row_reads)
    monkeypatch.setattr(catalog_module, "_file_concept_coverage", no_row_reads)

    active = catalog_module.build_catalog()["activeExportCoverage"]

    assert active["status"] == "ready"
    assert active["mode"] == "schema_only"
    assert active["concepts"]["hr"]["basis"] == "column_present_in_export_schema"
    assert active["concepts"]["hr"]["coverage_pct"] is None
    assert "stay_id" not in json.dumps(active)
