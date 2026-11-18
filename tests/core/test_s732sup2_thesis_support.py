"""Synthetic stay/hour fixtures for the r15 support projections."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES, extract_database
from easyicu.concept.thesis_support import load_thesis_support
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
        _write(root, "observations", {
            "patientid": members, "datetime": timestamps, "value": vals,
            "variableid": [2600, 2600, 2600, 2600, 2600, 2610, 2010, 2600, 2600, 2600],
        })
    else:
        _stays(root, database, intime=_ADMISSION, outtime=_ADMISSION, hadm_id=[71, 81])
        _write(root, "chartevents", {
            "stay_id": members, "charttime": timestamps, "valuenum": vals,
            "itemid": [220339, 220339, 220339, 220339, 220339, 224700, 220277, 220339, 220339, 220339],
        })


@pytest.mark.parametrize("database", ["aumc", "hirid", "miiv"])
def test_peep_uses_original_source_clock_range_channel_and_median(tmp_path, database):
    _peep_source(tmp_path, database)
    rows = _load(tmp_path, database, "observed_peep_set")
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    assert rows[stay].tolist() == [_STAY, _STAY, _STAY]
    assert rows.charttime.tolist() == [-1., 0., 1.]
    assert rows.observed_peep_set.tolist() == [6., 5., 40.]


def _hourly_imv_source(root, database):
    _stays(root, database)
    _write(root, f"{database}_hourly_core", {
        "stay_id": [_STAY] * 5 + [_OTHER], "icu_hour": [-2, -1, 0, 1, 12, 5],
        "imv": pd.Series([True, False, True, None, True, True], dtype="boolean"),
    })


@pytest.mark.parametrize("database", ["aumc", "miiv", "sic"])
def test_imv_flags_preserve_source_clock_and_disconnected_hours(tmp_path, database):
    _hourly_imv_source(tmp_path, database)
    rows = _load(tmp_path, database, "imv_recorded_compat")
    assert rows.charttime.tolist() == [-2., 0., 12.]
    assert rows.imv.tolist() == [True] * 3
    assert rows.imv_recorded_end_hour.isna().all()


def _hirid_imv_source(root):
    _stays(root, "hirid", admissiontime=_ADMISSION)
    # The frozen consumer receives these first/last bounds. Separate
    # recording bursts for the same stay are already spanned by its anchor.
    _write(root, "hirid_vent_anchor", {
        "stay_id": [_STAY, _OTHER], "imv_t0_h": [-0.25, 4.9], "imv_end_h": [12.75, 4.3],
    })


def test_hirid_joins_disconnected_recordings_through_the_frozen_span(tmp_path):
    _hirid_imv_source(tmp_path)
    rows = load_thesis_support(
        module="imv_recorded_compat", database="hirid", data_path=tmp_path, patient_ids=[_STAY, _OTHER],
    )
    assert rows.charttime.tolist() == [0., 4.]
    assert rows.imv_recorded_end_hour.tolist() == [12., 4.]
    assert rows.imv.isna().all()


def _rrt_source(root, database):
    _stays(root, database)
    if database == "miiv":
        # An actual end before a scheduled end leaves no active rows at the
        # later hours, even when a generic procedure record would imply them.
        _write(root, "miiv_rrt_time", {
            "stay_id": [_STAY] * 5 + [_OTHER], "charttime": [-23.5, -1., 1., 2., 5., 50.],
            "rrt": [True, True, True, False, None, True],
        })
        _write(root, "procedureevents", {
            "stay_id": [_STAY], "itemid": [225802],
            "starttime": [_ADMISSION], "endtime": [_ADMISSION + pd.Timedelta(hours=6)],
        })
        return [-24., -1., 1.]
    if database == "mimic":
        _write(root, "mimic3cv_rrt_hours", {"stay_id": [_STAY, _STAY, _OTHER], "icu_hour": [-25, 3, 11]})
    else:
        _write(root, f"{database}_hourly_core", {
            "stay_id": [_STAY] * 4 + [_OTHER], "icu_hour": [-25, 0, 3, 11, 12],
            "rrt": pd.Series([True, False, True, None, True], dtype="boolean"),
        })
    return [-25., 3.]


@pytest.mark.parametrize("database", ["aumc", "eicu", "hirid", "miiv", "mimic", "sic"])
def test_rrt_projects_active_treatment_and_preserves_the_full_clock(tmp_path, database):
    expected = _rrt_source(tmp_path, database)
    rows = _load(tmp_path, database, "rrt_time")
    assert rows.charttime.tolist() == expected
    assert rows.rrt_active.tolist() == [True] * len(expected)


@pytest.mark.parametrize("database,module,value,dtype", [
    ("eicu", "imv_recorded_compat", "imv", "boolean"),
    ("mimic", "imv_recorded_compat", "imv", "boolean"),
    ("hirid", "imv_recorded_compat", "imv_recorded_end_hour", "float64"),
    ("sic", "imv_recorded_compat", "imv", "boolean"),
    ("eicu", "observed_peep_set", "observed_peep_set", "float64"),
    ("mimic", "observed_peep_set", "observed_peep_set", "float64"),
    ("sic", "observed_peep_set", "observed_peep_set", "float64"),
    ("miiv", "rrt_time", "rrt_active", "boolean"),
    ("sic", "rrt_time", "rrt_active", "boolean"),
])
def test_missing_authority_is_typed_and_distinguished_from_an_empty_source(tmp_path, database, module, value, dtype):
    _stays(tmp_path, database)
    rows = _load(tmp_path, database, module)
    assert rows.empty
    assert str(rows[value].dtype) == dtype
    assert rows.attrs["available_fields"] == []


def test_miiv_thesis_export_requires_explicit_stay_selection(tmp_path):
    with pytest.raises(ValueError, match="explicit allowed stay cohort"):
        extract_database("miiv", data_path=tmp_path, modules=["rrt_time"])


@pytest.mark.parametrize("stream", [False, True])
def test_native_export_is_opt_in_and_seals_typed_support_receipts(tmp_path, stream):
    source = tmp_path / "source"
    source.mkdir()
    _stays(source, "aumc", admittedat=0)
    _write(source, "aumc_hourly_core", {
        "stay_id": [_STAY, _STAY, _OTHER], "icu_hour": [-32, 14, 5],
        "imv": [True, True, True], "rrt": [True, False, True],
    })
    assert all(name in EXTRACT_MODULES and name not in EXTRACT_MODULE_ORDER for name in _MODULES)
    output = tmp_path / "out"
    extract_database(
        "aumc", data_path=source, output_dir=output, modules=list(_MODULES),
        patient_ids=[_STAY], stream_output_batches=stream, batch_size=1,
        native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entries = {entry["module"]: entry for entry in manifest["files"]}
    assert entries["rrt_time"]["thesis_support"]["source"] == "sealed_hourly_rrt"
    assert entries["rrt_time"]["thesis_support"]["status"] == "projected"
    assert entries["imv_recorded_compat"]["time_axis_audit"]["policy"] == "frozen_source_icu_hours"
    imv = pd.read_parquet(output / "imv_recorded_compat.parquet")
    assert imv.charttime.tolist() == [-32., 14.]
    assert imv.imv.tolist() == [True, True]
    assert imv.imv_recorded_end_hour.isna().all()
    rrt = pd.read_parquet(output / "rrt_time.parquet")
    assert rrt.charttime.tolist() == [-32.]
    assert rrt.rrt_active.tolist() == [True]
    peep = pd.read_parquet(output / "observed_peep_set.parquet")
    assert peep.empty and str(peep.observed_peep_set.dtype) == "float64"
    assert entries["observed_peep_set"]["availability"] == "structurally_unavailable"
    assert entries["observed_peep_set"]["thesis_support"]["status"] == "authority_unavailable"


def test_native_export_describes_hirid_spans_and_peep_medians(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _hirid_imv_source(source)
    _peep_source(source, "hirid")
    output = tmp_path / "out"
    extract_database(
        "hirid", data_path=source, output_dir=output, modules=list(_MODULES[:2]),
        patient_ids=[_STAY], native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entries = {entry["module"]: entry for entry in manifest["files"]}
    assert entries["imv_recorded_compat"]["thesis_support"]["source_grain"] == "stay"
    assert entries["imv_recorded_compat"]["thesis_support"]["charttime_role"] == "first_recorded_hour"
    sidecar = json.loads((output / manifest["column_metadata"]["file"]).read_text())
    metadata = {entry["module"]: entry["columns"] for entry in sidecar["files"]}
    assert metadata["observed_peep_set"]["observed_peep_set"]["metadata"]["aggregation"] == "median"
    assert metadata["imv_recorded_compat"]["imv_recorded_end_hour"]["metadata"]["time_unit"] == "h"


@pytest.mark.parametrize("stream", [False, True])
def test_no_treatment_preserves_rrt_source_availability(tmp_path, stream):
    source = tmp_path / "source"
    source.mkdir()
    _stays(source, "aumc")
    _write(source, "aumc_hourly_core", {"stay_id": [_STAY], "icu_hour": [0], "rrt": [False]})
    output = tmp_path / "out"
    extract_database(
        "aumc", data_path=source, output_dir=output, modules=["rrt_time"],
        patient_ids=[_STAY], stream_output_batches=stream, batch_size=1,
        native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entry, = manifest["files"]
    assert entry["rows"] == 0
    assert entry["thesis_support"]["status"] == "projected"
    assert entry["availability"] == "available"


@pytest.fixture
def frozen_src_dir():
    value = os.environ.get("THESIS_FROZEN_SOURCE_DIR")
    if not value:
        pytest.skip("frozen reference checkout is external")
    return Path(value)


@pytest.mark.parametrize("database,module", [
    *((db, "observed_peep_set") for db in ("aumc", "hirid", "miiv")),
    *((db, "imv_recorded_compat") for db in ("aumc", "hirid", "miiv", "sic")),
    *((db, "rrt_time") for db in ("aumc", "eicu", "hirid", "miiv", "mimic", "sic")),
])
def test_harness_agrees_with_frozen_sql_for_identical_synthetic_inputs(tmp_path, frozen_src_dir, database, module):
    if module == "observed_peep_set":
        _peep_source(tmp_path, database)
    elif module == "rrt_time":
        _rrt_source(tmp_path, database)
    elif database == "hirid":
        _hirid_imv_source(tmp_path)
    else:
        _hourly_imv_source(tmp_path, database)
    report = frozen_equivalence.run_equivalence(
        database=database, db_root=tmp_path, stay_ids=[_STAY], module=module, frozen_src_dir=frozen_src_dir,
    )
    assert report["status"] == "agree"
    assert report["module"] == report["frozen"]
    assert report["module"]["rows"] is None
    assert report["module"]["stays"] is None
    assert len(report["module"]["sha256"]) == 64


def test_harness_detects_a_planted_difference_and_prints_aggregates_only(
    tmp_path, frozen_src_dir, monkeypatch, capsys,
):
    _peep_source(tmp_path, "aumc")
    load = frozen_equivalence.load_thesis_support

    def planted(**kwargs):
        frame = load(**kwargs)
        frame.loc[0, "observed_peep_set"] = 332929.5
        return frame

    monkeypatch.setattr(frozen_equivalence, "load_thesis_support", planted)
    status = frozen_equivalence.main([
        "--database", "aumc", "--db-root", str(tmp_path), "--stay-ids", str(_STAY),
        "--module", "observed_peep_set", "--frozen-src-dir", str(frozen_src_dir),
    ])
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert status == 1 and report["status"] == "different"
    assert report["mismatching_rows"] is None
    assert report["mismatching_cells"]["observed_peep_set"] is None
    assert report["module"]["sha256"] != report["frozen"]["sha256"]
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
