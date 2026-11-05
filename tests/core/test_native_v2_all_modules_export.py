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
import pandas as pd
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
    assert "renal_urine_chem" in EXTRACT_MODULE_ORDER
    urine = next(row for row in manifest["files"] if row["module"] == "renal_urine_chem")
    assert urine["physical_concept_ids"] == [
        "urine_na", "urine_crea", "urine_osm", "urine_urea", "sic_precondition_renal_dysfunction",
    ]
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
    if database == "aumc":
        from easyicu.concept_output_sources import SOURCE_ITEM_OUTPUT_COLUMNS

        # R6-F019: source values travel through publication as companions of
        # fio2_chart, without adding independently selectable public concepts.
        respiratory = pq.read_table(output / "respiratory.parquet")
        assert set(SOURCE_ITEM_OUTPUT_COLUMNS["fio2_chart"]) <= set(respiratory.column_names)
    if database == "sic":
        # R5-SIC-001: raw CVP survives publication beside the exclusion reason.
        vitals = pq.read_table(output / "vitals.parquet").to_pandas()
        assert vitals.cvp_raw.max() == 60.
        assert "above_50" in set(vitals.cvp_qc.dropna())
        assert (vitals.cvp.dropna() <= 50).all()
        # R5-SIC-016: publication keeps the converted physical values, not
        # only a non-empty column or a relabelled unit. Pool the source rows
        # independently on the official ICU clock to check all six channels.
        source = tmp_path / "source"
        events = pd.read_parquet(source / "data_float_h.parquet")
        origins = pd.read_parquet(source / "cases.parquet").set_index("CaseID").ICUOffset
        events["charttime"] = ((events.Offset - events.CaseID.map(origins)) // 3600).astype(float)
        channels = {"peep": ([2278], 40), "peep_set": ([2278], 40),
                    "pip": ([715], 100), "ps": ([2284], 40),
                    "mean_airway_pres": ([2279, 712], 60), "compliance": ([3120], 200)}
        published = pd.read_parquet(output / "ventilator.parquet").set_index(["stay_id", "charttime"])
        for name, (items, upper) in channels.items():
            rows = events.loc[events.DataID.isin(items)].copy()
            rows["value"] = rows.Val * (98.0665/100 if name == "compliance" else 100/98.0665)
            expected = rows.loc[rows.value.between(0, upper)].groupby(["CaseID", "charttime"]).value.median()
            actual = published[name].dropna().sort_index()
            expected.index.names = actual.index.names
            pd.testing.assert_series_equal(actual, expected.sort_index(), check_names=False, check_dtype=False)


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
