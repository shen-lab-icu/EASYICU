"""Optional offline release audit; emits aggregates only, never patient rows.

Set EASYICU_MIMIC_III_DEMO_ARCHIVE to an approved local public demo archive.
The archive and its sibling SHA256SUMS.txt are read only. All generated data
are deleted with the temporary directory, including on assertion failure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback
import zipfile

import pandas as pd
import pytest

import easyicu.concept as concept_module
from easyicu.base import BaseICULoader
from easyicu.datasource import _close_duckdb_connections
from easyicu.table.duration import get_dur_var_unit
from easyicu.webserver import demo_source_prepare, demo_source_storage
from easyicu.webserver.demo_source_contracts import get_source
from tests.support.windows_laptop_demo import isolated_demo_paths


ARCHIVE = Path(os.environ.get(
    "EASYICU_MIMIC_III_DEMO_ARCHIVE",
    "/home/zhuhb/easyicu_dev9/official_demos/mimiciii_demo_1_4/"
    "mimic-iii-clinical-database-demo-1.4.zip",
))


class _Job:
    def emit(self, _event):
        return True


@pytest.mark.slow
@pytest.mark.skipif(not ARCHIVE.is_file(), reason="Approved local MIMIC-III public demo archive is absent")
def test_local_official_archive_converts_and_extracts_all_mapped_concepts(monkeypatch):
    source = get_source("mimic_iii_demo_v1_4")
    monkeypatch.setenv("EASYICU_CONV_WORKERS", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_MEMORY_LIMIT", "1GB")
    monkeypatch.setattr(
        demo_source_storage, "official_response",
        lambda *_args, **_kwargs: pytest.fail("Offline audit attempted a download"),
    )
    report = {"concepts": [], "errors": []}
    report["durations"] = []
    normalize = concept_module._normalize_source_dur_var_hours

    def observe_duration(frame, **kwargs):
        frame = normalize(frame, **kwargs)
        if "dur_var" in frame:
            report["durations"].append({"concept": kwargs["concept_name"], "rows": len(frame),
                                        "dtype": str(frame.dur_var.dtype), "unit": get_dur_var_unit(frame)})
        return frame

    monkeypatch.setattr(concept_module, "_normalize_source_dur_var_hours", observe_duration)
    with tempfile.TemporaryDirectory(prefix="easyicu-mimiciii-audit-") as directory:
        paths = isolated_demo_paths(Path(directory), source)
        paths.root.mkdir(parents=True)
        shutil.copyfile(ARCHIVE, paths.archive)
        digest, reused = demo_source_storage.download_archive(source, paths, _Job())
        assert reused and digest == source.archive_sha256
        checksums = {}
        for line in ARCHIVE.with_name("SHA256SUMS.txt").read_text().splitlines():
            checksum, filename = line.split(maxsplit=1)
            checksums[Path(filename.lstrip("*")).name] = checksum
        with zipfile.ZipFile(paths.archive) as archive:
            files = [entry for entry in archive.infolist()
                     if not entry.is_dir() and Path(entry.filename).name != "SHA256SUMS.txt"]
            assert len(files) == 27
            for entry in files:
                with archive.open(entry) as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                assert actual == checksums[Path(entry.filename).name], entry.filename
        assert not demo_source_storage.extract_archive(source, paths, digest, _Job())
        conversion, reused = demo_source_prepare.convert_dataset(source, paths, digest, _Job())
        assert conversion == {"converted": 26, "failed": 0, "skipped": 0, "total_files": 26}
        assert not reused
        report["conversion"] = conversion
        loader = BaseICULoader(data_path=paths.raw, database=source.database)
        try:
            tables = {
                table: loader.datasource.load_table(table).data
                for table in ("patients", "admissions", "icustays")
            }
            patients, admissions, stays = (tables[table] for table in tables)
            assert patients.subject_id.nunique() == 100
            assert stays.icustay_id.nunique() == 136
            assert set(admissions.subject_id) <= set(patients.subject_id)
            assert set(stays.hadm_id) <= set(admissions.hadm_id)
            assert set(map(tuple, stays[["subject_id", "hadm_id"]].values)) <= set(
                map(tuple, admissions[["subject_id", "hadm_id"]].values)
            )
            for table, columns in {
                "patients": ("dob", "dod"),
                "admissions": ("admittime", "dischtime", "deathtime"),
                "icustays": ("intime", "outtime"),
            }.items():
                for column in columns:
                    assert pd.api.types.is_datetime64_any_dtype(tables[table][column]), (table, column)
            report["hierarchy"] = {"patients": 100, "admissions": len(admissions), "icu_stays": 136}
            for name, definition in loader.concept_dict.items():
                if not definition.for_data_source(loader.datasource.config):
                    continue
                try:
                    frame = loader.load_concepts(
                        name, concept_workers=1, parallel_workers=1, r_compatible=False,
                    )
                    column = frame[name]
                    classes = str(definition.class_name or "num_cncpt")
                    expected = "boolean" if "lgl_cncpt" in classes else (
                        "categorical" if "fct_cncpt" in classes else "numeric"
                    )
                    valid = {
                        "boolean": pd.api.types.is_bool_dtype(column),
                        "categorical": isinstance(column.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(column),
                        "numeric": pd.api.types.is_numeric_dtype(column),
                    }[expected]
                    row = {"concept": name, "non_null": int(column.notna().sum()),
                           "dtype": str(column.dtype), "expected": expected, "valid_dtype": valid}
                    report["concepts"].append(row)
                    row["icu_identifiers_valid"] = "icustay_id" in frame and bool(
                        frame.icustay_id.isin(stays.icustay_id).all()
                    )
                    if not row["icu_identifiers_valid"]:
                        report["errors"].append({"concept": name, "error": "icu_identifier_mismatch"})
                    if "charttime" in frame:
                        row["time_dtype"] = str(frame.charttime.dtype)
                        if not frame.empty and not pd.api.types.is_numeric_dtype(frame.charttime):
                            report["errors"].append({"concept": name, "error": "time_not_relative_hours"})
                    selected = loader.load_concepts(
                        name, patient_ids={"icustay_id": stays.icustay_id.tolist()},
                        concept_workers=1, parallel_workers=1, r_compatible=False,
                    )[name]
                    row["selected_non_null"] = int(selected.notna().sum())
                    row["selected_dtype"] = str(selected.dtype)
                    if row["non_null"] != row["selected_non_null"]:
                        report["errors"].append({"concept": name, "error": "selected_count_mismatch"})
                    if not valid:
                        report["errors"].append({"concept": name, "error": "dtype_mismatch"})
                except Exception as exc:  # Audit all concepts before reporting failure; no row-bearing messages.
                    cause = exc
                    while cause.__cause__ or cause.__context__:
                        cause = cause.__cause__ or cause.__context__
                    report["errors"].append({"concept": name, "error": type(cause).__name__,
                                             "frames": [f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                                                        for f in traceback.extract_tb(cause.__traceback__)]})
        finally:
            loader.clear_cache()
            _close_duckdb_connections()
    output = os.environ.get("EASYICU_DEMO_AUDIT_REPORT")
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    assert not report["errors"], report["errors"]
