"""Every module of every database publishes on the native-v2 path.

2026-11-04 extraction audit round 4 (R4-004/R4-005): one AUMC module crash
stopped native-v2 publication and left the other 18 module files raw. The
synthetic HiRID/AUMC/SICdb sources run in CI. The six real databases run when
``EASYICU_NATIVE_V2_REAL_ROOT`` names a directory holding them (the server
regression); outputs are aggregates checked in place, never copied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REAL_ROOT = os.environ.get("EASYICU_NATIVE_V2_REAL_ROOT")
REAL_PATHS = {
    "miiv": "mimiciv", "mimic": "mimiciii/1.4", "eicu": "eicu",
    "aumc": "aumc", "hirid": "hirid", "sic": "sic",
}
RAW_ID_COLUMNS = {"admissionid", "patientid", "CaseID", "patientunitstayid", "icustay_id"}


@pytest.fixture(autouse=True)
def _restore_process_temp_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)


def _assert_published(database: str, output: Path, result: dict) -> None:
    from easyicu.api.extraction import EXTRACT_MODULE_ORDER

    assert not {m: v["errors"] for m, v in result["modules"].items() if v.get("errors")}
    assert (output / "_manifest.json").exists()
    assert not (output / ".easyicu-unpublished").exists()
    manifest = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest  # sealed by the producer
    for module in EXTRACT_MODULE_ORDER:
        path = output / f"{module}.parquet"
        assert path.exists(), f"{database}: {module} not published"
        names = set(pq.read_schema(path).names)
        assert "stay_id" in names, (database, module)
        assert not names & RAW_ID_COLUMNS, (database, module, names & RAW_ID_COLUMNS)


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_synthetic_source_publishes_all_modules(tmp_path, database):
    from easyicu.api.extraction import EXTRACT_MODULE_ORDER, extract_database
    from easyicu.synthetic_sources import generate_source

    generate_source(tmp_path / "source", database, stays=12)
    output = tmp_path / "out"
    result = extract_database(
        database, data_path=str(tmp_path / "source"), output_dir=str(output),
        modules=list(EXTRACT_MODULE_ORDER), native_export_v2=True, verbose=False,
    )
    _assert_published(database, output, result)


@pytest.mark.skipif(not REAL_ROOT, reason="EASYICU_NATIVE_V2_REAL_ROOT is not set")
@pytest.mark.parametrize("database", sorted(REAL_PATHS))
def test_real_database_publishes_all_modules(tmp_path, database):
    from easyicu.api.extraction import EXTRACT_MODULE_ORDER, _get_all_patient_ids, extract_database

    data_path = str(Path(REAL_ROOT) / REAL_PATHS[database])
    ids, id_col = _get_all_patient_ids(data_path, database, None)
    sample = sorted(ids)[: int(os.environ.get("EASYICU_NATIVE_V2_REAL_STAYS", "200"))]
    output = tmp_path / "out"
    result = extract_database(
        database, data_path=data_path, output_dir=str(output),
        modules=list(EXTRACT_MODULE_ORDER), patient_ids={id_col: sample},
        native_export_v2=True, stream_output_batches=True, verbose=False,
    )
    _assert_published(database, output, result)
