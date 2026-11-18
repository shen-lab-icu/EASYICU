"""Synthetic stay/hour fixtures for the r15 support projections."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES, extract_database
from easyicu.concept import thesis_support
from easyicu.concept.thesis_support import load_thesis_support, thesis_support_manifest
from easyicu.config import DATABASE_ID_CONFIG
from tools.thesis_support import frozen_equivalence

_STAY = 411471
_OTHER = 411479
_ADMISSION = pd.Timestamp("2135-02-02 03:04:05")
_MODULES = ("imv_recorded_compat", "observed_peep_set", "rrt_time")


def _write(root, table, rows):
    pd.DataFrame(rows).to_parquet(root / f"{table}.parquet", index=False)


def _stays(root, database, **fields):
    config = DATABASE_ID_CONFIG[database]
    _write(root, config["table"], {config["id_col"]: [_STAY, _OTHER], **fields})


def _load(root, database, module):
    return load_thesis_support(
        module=module, database=database, data_path=root, patient_ids=[_STAY],
    )


def _peep_source(root, database):
    offsets_ms = [-1, 0, 3_599_999, 3_600_000, 3_600_001, 0, 0, 0, 7200000, 0]
    vals = [6., 0., 10., 40., 40.1, 18., 39., -1., 41., 39.]
    members = [_STAY] * 9 + [_OTHER]
    timestamps = _ADMISSION + pd.to_timedelta(offsets_ms, unit="ms")
    if database == "aumc":
        admitted = 172_800_000 + 789
        _stays(root, database, admittedat=admitted)
        _write(root, "numericitems", {
            "admissionid": members, "measuredat": [admitted + t for t in offsets_ms], "value": vals,
            "itemid": [12284, 12336, 8862, 12284, 12284, 16250, 6694, 12284, 12284, 12284],
            "unit": "mbar",  # Frozen sources consume the raw setting value.
        })
    elif database == "hirid":
        _stays(root, database, admissiontime=_ADMISSION)
        observations = root / "observations"
        observations.mkdir()
        _write(observations, "part", {
            "patientid": members, "datetime": timestamps, "value": vals,
            "variableid": [2600, 2600, 2600, 2600, 2600, 2610, 2010, 2600, 2600, 2600],
        })
    elif database == "sic":
        offsets = [-1, 0, 3599, 3600, 3601, 0, 0, 0, 7200, 0, None, 10800]
        icu_offset = 12_345
        _stays(root, database, ICUOffset=[icu_offset, 500])
        _write(root, "data_float_h", {
            "CaseID": members + [_STAY, _STAY],
            "Offset": pd.array(
                [icu_offset + t if t is not None else None for t in offsets], dtype="Int64",
            ),
            "Val": vals + [8., None],
            "DataID": [2278, 2278, 2278, 2278, 2278, 2283, 710, 2278, 2278, 2278, 2278, 2278],
        })
    else:
        _stays(root, database, intime=_ADMISSION, outtime=_ADMISSION, hadm_id=[71, 81])
        _write(root, "chartevents", {
            "stay_id": members, "charttime": timestamps, "valuenum": vals,
            "itemid": [220339, 220339, 220339, 220339, 220339, 224700, 220277, 220339, 220339, 220339],
        })


@pytest.mark.parametrize("database", ["aumc", "hirid", "miiv", "sic"])
def test_peep_uses_original_source_clock_range_channel_and_median(tmp_path, database):
    _peep_source(tmp_path, database)
    rows = _load(tmp_path, database, "observed_peep_set")
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    assert rows[stay].tolist() == [_STAY, _STAY, _STAY]
    assert rows.charttime.tolist() == [-1., 0., 1.]
    assert rows.observed_peep_set.tolist() == [6., 5., 40.]


@pytest.mark.parametrize("database,module,missing_input", [
    *((db, "imv_recorded_compat", f"{db}/respiratory.parquet:mech_vent")
      for db in ("aumc", "miiv", "hirid", "sic")),
    *((db, "rrt_time", f"{db}_hourly_core.parquet:rrt") for db in ("aumc", "eicu", "hirid", "sic")),
    ("miiv", "rrt_time", "raw_timestamp_actual_end.parquet"),
    ("mimic", "rrt_time", "mimiciii/chartevents/*.parquet"),
    *((db, module, None) for db in ("eicu", "mimic") for module in ("imv_recorded_compat", "observed_peep_set")),
])
def test_typed_absence_without_any_source_reads(tmp_path, monkeypatch, database, module, missing_input):
    def unexpected_source(*args, **kwargs):
        pytest.fail("an unavailable reducer opened a source")

    monkeypatch.setattr(thesis_support, "storage_source", unexpected_source)
    monkeypatch.setattr(frozen_equivalence, "storage_source", unexpected_source)
    rows = _load(tmp_path, database, module)
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    fields = {
        "imv_recorded_compat": {"imv": "boolean", "imv_recorded_end_hour": "float64"},
        "observed_peep_set": {"observed_peep_set": "float64"},
        "rrt_time": {"rrt_active": "boolean"},
    }
    assert rows.empty
    assert rows.dtypes.astype(str).to_dict() == {stay: "int64", "charttime": "float64", **fields[module]}
    assert rows.attrs["available_fields"] == []
    receipt = thesis_support_manifest(module, database, rows.attrs["available_fields"])
    if missing_input is None:
        assert receipt == {"source": None, "status": "no_frozen_reducer"}
    else:
        assert receipt["status"] == "UNAVAILABLE_IN_SCOPE"
        assert receipt["reason"] == "no_raw_producer"
        assert missing_input in receipt["missing_inputs"]
    if database == "mimic" and module == "rrt_time":
        assert "#641 CareVue view" in receipt["detail"]
        assert "mimiciii/procedureevents_mv.parquet" in receipt["missing_inputs"]
    report = frozen_equivalence.run_equivalence(
        database=database, db_root=tmp_path, stay_ids=[_STAY], module=module,
    )
    assert report == {"status": receipt["status"]}


def test_miiv_thesis_export_requires_explicit_stay_selection(tmp_path):
    with pytest.raises(ValueError, match="explicit allowed stay cohort"):
        extract_database("miiv", data_path=tmp_path, modules=["rrt_time"])


@pytest.mark.parametrize("stream", [False, True])
def test_native_export_is_opt_in_and_seals_typed_support_receipts(tmp_path, stream):
    source = tmp_path / "source"
    source.mkdir()
    _peep_source(source, "sic")
    # The old authority path would fail if it tried to read this file.
    (source / "sic_hourly_core.parquet").write_bytes(b"unused staged input")
    assert all(name in EXTRACT_MODULES and name not in EXTRACT_MODULE_ORDER for name in _MODULES)
    output = tmp_path / "out"
    extract_database(
        "sic", data_path=source, output_dir=output, modules=list(_MODULES),
        patient_ids=[_STAY], stream_output_batches=stream, batch_size=1,
        native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entries = {entry["module"]: entry for entry in manifest["files"]}
    for module in ("imv_recorded_compat", "rrt_time"):
        assert entries[module]["availability"] == "structurally_unavailable"
        assert entries[module]["thesis_support"]["status"] == "UNAVAILABLE_IN_SCOPE"
        assert entries[module]["thesis_support"]["reason"] == "no_raw_producer"
        assert entries[module]["time_axis_audit"]["policy"] == "frozen_source_icu_hours"
    assert "sic_hourly_core.parquet:rrt" in entries["rrt_time"]["thesis_support"]["missing_inputs"]
    imv = pd.read_parquet(output / "imv_recorded_compat.parquet")
    assert imv.empty and str(imv.imv.dtype) == "boolean"
    assert str(imv.imv_recorded_end_hour.dtype) == "float64"
    rrt = pd.read_parquet(output / "rrt_time.parquet")
    assert rrt.empty and str(rrt.rrt_active.dtype) == "boolean"
    peep = pd.read_parquet(output / "observed_peep_set.parquet")
    assert peep.charttime.tolist() == [-1., 0., 1.]
    assert peep.observed_peep_set.tolist() == [6., 5., 40.]
    assert entries["observed_peep_set"]["availability"] == "available"
    assert entries["observed_peep_set"]["thesis_support"]["input_tables"] == ["cases", "data_float_h"]


def test_native_export_describes_absent_hirid_spans_and_peep_medians(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _peep_source(source, "hirid")
    output = tmp_path / "out"
    extract_database(
        "hirid", data_path=source, output_dir=output, modules=list(_MODULES[:2]),
        patient_ids=[_STAY], native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entries = {entry["module"]: entry for entry in manifest["files"]}
    assert entries["imv_recorded_compat"]["thesis_support"]["status"] == "UNAVAILABLE_IN_SCOPE"
    assert entries["imv_recorded_compat"]["thesis_support"]["source_grain"] == "stay"
    assert entries["imv_recorded_compat"]["thesis_support"]["charttime_role"] == "first_recorded_hour"
    sidecar = json.loads((output / manifest["column_metadata"]["file"]).read_text())
    metadata = {entry["module"]: entry["columns"] for entry in sidecar["files"]}
    assert metadata["observed_peep_set"]["observed_peep_set"]["metadata"]["aggregation"] == "median"
    imv_entry = entries["imv_recorded_compat"]
    assert imv_entry["physical_schema"]["imv_recorded_end_hour"] == "double"
    assert imv_entry["concept_status"]["imv_recorded_end_hour"]["availability"] == "structurally_unavailable_placeholder"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("has_source", [False, True])
def test_available_peep_source_with_no_qualifying_rows(tmp_path, stream, has_source):
    source = tmp_path / "source"
    source.mkdir()
    _stays(source, "sic", ICUOffset=0)
    if has_source:
        _write(source, "data_float_h", {"CaseID": [_STAY], "Offset": [0], "DataID": [2278], "Val": [41.]})
    output = tmp_path / "out"
    extract_database(
        "sic", data_path=source, output_dir=output, modules=["observed_peep_set"],
        patient_ids=[_STAY], stream_output_batches=stream, batch_size=1,
        native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entry, = manifest["files"]
    assert entry["rows"] == 0
    assert entry["thesis_support"]["status"] == ("projected" if has_source else "authority_unavailable")
    assert entry["availability"] == ("available" if has_source else "structurally_unavailable")


@pytest.fixture
def frozen_src_dir():
    value = os.environ.get("THESIS_FROZEN_SOURCE_DIR")
    if not value:
        pytest.skip("frozen reference checkout is external")
    return Path(value)


@pytest.mark.parametrize("database", ["aumc", "hirid", "miiv", "sic"])
def test_harness_agrees_with_frozen_sql_for_identical_synthetic_inputs(tmp_path, frozen_src_dir, database):
    _peep_source(tmp_path, database)
    report = frozen_equivalence.run_equivalence(
        database=database, db_root=tmp_path, stay_ids=[_STAY],
        module="observed_peep_set", frozen_src_dir=frozen_src_dir,
    )
    assert report["status"] == "agree"
    assert report["module"] == report["frozen"] == {"rows": None, "stays": None}


def test_harness_detects_a_planted_difference_and_prints_aggregates_only(
    tmp_path, frozen_src_dir, monkeypatch, capsys,
):
    _peep_source(tmp_path, "sic")
    load = frozen_equivalence.load_thesis_support

    def planted(**kwargs):
        frame = load(**kwargs)
        frame.loc[0, "observed_peep_set"] = 332929.5
        return frame

    monkeypatch.setattr(frozen_equivalence, "load_thesis_support", planted)
    status = frozen_equivalence.main([
        "--database", "sic", "--db-root", str(tmp_path), "--stay-ids", str(_STAY),
        "--module", "observed_peep_set", "--frozen-src-dir", str(frozen_src_dir),
    ])
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert status == 1 and report["status"] == "different"
    assert report["mismatching_rows"] is None
    assert report["mismatching_cells"]["observed_peep_set"] is None
    assert report["module"] == report["frozen"] == {"rows": None, "stays": None}
    assert str(_STAY) not in out.out and "332929.5" not in out.out
    assert not out.err


def test_aggregate_comparison_counts_larger_differences_per_column():
    observed = pd.DataFrame({"stay_id": range(100, 114), "charttime": 0, "observed_peep_set": 12.})
    frozen = observed.assign(observed_peep_set=6.).iloc[::-1]
    result = frozen_equivalence.compare_rows(observed, frozen, key_columns=["stay_id", "charttime"])
    assert result["status"] == "different"
    assert result["mismatching_rows"] == 10
    assert result["mismatching_cells"] == {"stay_id": None, "charttime": None, "observed_peep_set": 10}
    assert result["module"]["stays"] == result["frozen"]["stays"] == 10
