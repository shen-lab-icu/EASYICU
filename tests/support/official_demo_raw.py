"""Synthetic official raw Demo receipts shared by web-source flow tests."""

from dataclasses import replace
import hashlib
import json

import pandas as pd
import pytest

from easyicu.webserver import demo_source_storage as storage


@pytest.fixture
def official_raw(tmp_path, monkeypatch):
    archive = b"engineering fixture, not patient data"
    source = replace(
        next(row for row in storage.SOURCES if row.database == "eicu_demo"),
        size_bytes=len(archive), archive_sha256=hashlib.sha256(archive).hexdigest(),
    )
    monkeypatch.setattr(storage, "SOURCES", (source,))
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    paths = storage.source_paths(source)
    paths.raw.mkdir(parents=True)
    paths.archive.write_bytes(archive)
    csv = paths.raw / "patient.csv"
    csv.write_text("patientunitstayid,uniquepid\n")
    parquet = paths.raw / "patient.parquet"
    pd.DataFrame({"patientunitstayid": pd.Series(dtype="int64"),
                  "uniquepid": pd.Series(dtype="str")}).to_parquet(parquet)
    for marker in (paths.extracted_marker, paths.converted_marker):
        storage.write_marker(marker, source, archive_sha256=source.archive_sha256)

    def member(path):
        info = path.stat()
        return {"path": str(path), "relative_path": path.name, "exists": True,
                "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}

    manifest = {
        "schema_version": "easyicu.conversion_manifest/1", "database": "eicu_demo",
        "data_path": str(paths.raw), "tables": [{
            "table": "patient", "status": "completed", "data_quality_status": "clean",
            "bad_rows_skipped": 0, "input": member(csv), "outputs": [member(parquet)],
        }],
    }
    (paths.raw / "conversion_manifest.json").write_text(json.dumps(manifest))
    return paths, source, manifest

