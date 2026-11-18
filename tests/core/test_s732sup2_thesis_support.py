"""Synthetic stay/hour fixtures for the r15 support projections."""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from easyicu import datasource
from easyicu.api import extraction
from easyicu.api.extraction import (
    EXTRACT_MODULE_ORDER, EXTRACT_MODULES, _load_module_concepts, extract_database, publication_dictionary,
)
from easyicu.concept import thesis_support
from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL, FIRST_IN_HOSP_FROZEN
from easyicu.concept.export_metadata import build_export_file_metadata_binding
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
    *((db, "imv_recorded_compat", "release.params.cohort.episode_gap_tolerance_h")
      for db in ("aumc", "miiv", "sic")),
    ("hirid", "imv_recorded_compat", "hirid/respiratory.parquet:mech_vent"),
    *((db, "rrt_time", f"{db}/renal.parquet:rrt") for db in ("aumc", "eicu", "hirid", "sic")),
    ("miiv", "rrt_time", "miiv_rrt_time.parquet: treatment start and actual end through pre-ICU hour -24"),
    ("mimic", "rrt_time", None),
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
        assert receipt["reason"] == ("no_exact_producer" if (database, module) == ("miiv", "rrt_time")
                                     else "no_raw_producer")
        assert missing_input in receipt["missing_inputs"]
    report = frozen_equivalence.run_equivalence(
        database=database, db_root=tmp_path, stay_ids=[_STAY], module=module,
    )
    assert report == {key: receipt[key] for key in ("status", "reason", "missing_inputs") if key in receipt}


def test_unknown_support_module_fails_at_both_entry_points(tmp_path):
    name = "recorded_airway_events"
    with pytest.raises(KeyError, match=name):
        load_thesis_support(module=name, database="hirid", data_path=tmp_path)
    with pytest.raises(KeyError, match=name):
        thesis_support_manifest(name, "hirid", [])


def test_hirid_imv_only_request_reports_no_producer(tmp_path):
    assert _load_module_concepts(
        None, module_name="imv_recorded_compat",
        load_kwargs={
            "database": "hirid", "data_path": tmp_path, "patient_ids": [_OTHER],
            "concepts": ["imv"],
        },
    ) is None


def test_miiv_thesis_export_requires_explicit_stay_selection(tmp_path):
    with pytest.raises(ValueError, match="explicit allowed stay cohort"):
        extract_database("miiv", data_path=tmp_path, modules=["rrt_time"])


@pytest.mark.parametrize("stream", [False, True])
def test_native_export_is_opt_in_and_seals_typed_support_receipts(tmp_path, stream):
    source = tmp_path / "source"
    source.mkdir()
    _stays(source, "sic", ICUOffset=733)
    _write(source, "data_float_h", {
        "CaseID": [_OTHER] * 4, "Offset": [732, 733, 743, 4333],
        "Val": [6., 0., 10., 40.], "DataID": 2278,
    })
    # The old authority path would fail if it tried to read this file.
    (source / "sic_hourly_core.parquet").write_bytes(b"unused staged input")
    assert all(name in EXTRACT_MODULES and name not in EXTRACT_MODULE_ORDER for name in _MODULES)
    output = tmp_path / "out"
    extract_database(
        "sic", data_path=source, output_dir=output, modules=list(_MODULES),
        patient_ids=[_STAY, _OTHER], stream_output_batches=stream, batch_size=1,
        native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entries = {entry["module"]: entry for entry in manifest["files"]}
    for module in ("imv_recorded_compat", "rrt_time"):
        assert entries[module]["availability"] == "structurally_unavailable"
        assert entries[module]["thesis_support"]["status"] == "UNAVAILABLE_IN_SCOPE"
        assert entries[module]["thesis_support"]["reason"] == "no_raw_producer"
        assert entries[module]["time_axis_audit"]["policy"] == "frozen_source_icu_hours"
    assert "sic/renal.parquet:rrt" in entries["rrt_time"]["thesis_support"]["missing_inputs"]
    imv = pd.read_parquet(output / "imv_recorded_compat.parquet")
    assert imv.empty and str(imv.imv.dtype) == "boolean"
    assert str(imv.imv_recorded_end_hour.dtype) == "float64"
    rrt = pd.read_parquet(output / "rrt_time.parquet")
    assert rrt.empty and str(rrt.rrt_active.dtype) == "boolean"
    peep = pd.read_parquet(output / "observed_peep_set.parquet")
    if stream:
        batches = manifest["stream_batches"]["observed_peep_set"]
        assert [batch["output_rows"] for batch in batches] == [0, 3]
    assert peep.stay_id.tolist() == [_OTHER] * 3
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
        patient_ids=[_STAY, _OTHER], stream_output_batches=stream, batch_size=1,
        native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entry, = manifest["files"]
    assert entry["rows"] == 0
    if stream and has_source:
        batches = manifest["stream_batches"]["observed_peep_set"]
        assert [batch["output_rows"] for batch in batches] == [0, 0]
    assert entry["thesis_support"]["status"] == ("projected" if has_source else "authority_unavailable")
    assert entry["availability"] == ("available" if has_source else "structurally_unavailable")


@pytest.fixture
def frozen_src_dir():
    value = os.environ.get("THESIS_FROZEN_SOURCE_DIR")
    path = Path(value) if value else frozen_equivalence._FROZEN_SOURCE_DIR
    if not value and not path.is_dir():
        pytest.skip("frozen reference checkout is external")
    return path


@pytest.mark.parametrize("name", sorted(frozen_equivalence._FROZEN_DIGESTS))
def test_pinned_frozen_source_digest_is_complete_and_matches_local_copy(name):
    digest = frozen_equivalence._FROZEN_DIGESTS[name]
    assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    path = frozen_equivalence._FROZEN_SOURCE_DIR / f"{name}.py"
    if path.is_file():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_frozen_fixture_discovers_default_checkout(monkeypatch, request):
    path = frozen_equivalence._FROZEN_SOURCE_DIR
    if not path.is_dir():
        pytest.skip("frozen reference checkout is external")
    monkeypatch.delenv("THESIS_FROZEN_SOURCE_DIR", raising=False)
    try:
        discovered = request.getfixturevalue("frozen_src_dir")
    except pytest.skip.Exception:
        pytest.fail("fixture skipped despite an installed frozen checkout")
    assert discovered == path


@pytest.mark.parametrize("location", ["relative", "alternate"])
def test_cli_accepts_digest_verified_frozen_checkout(tmp_path, frozen_src_dir, capsys, location):
    _peep_source(tmp_path, "sic")
    if location == "relative":
        checkout = os.path.relpath(frozen_src_dir)
    else:
        checkout = tmp_path / "frozen"
        checkout.mkdir()
        (checkout / "new_sources.py").write_bytes((frozen_src_dir / "new_sources.py").read_bytes())
    status = frozen_equivalence.main([
        "--database", "sic", "--db-root", str(tmp_path), "--stay-ids", str(_STAY),
        "--module", "observed_peep_set", "--frozen-src-dir", str(checkout),
    ])
    assert json.loads(capsys.readouterr().out)["status"] == "agree"
    assert status == 0


def test_alternate_frozen_checkout_checks_digest_before_execution(tmp_path):
    path = tmp_path / "sources.py"
    path.write_text("raise AssertionError('unpinned source ran')", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected frozen source digest"):
        with frozen_equivalence._frozen_module(tmp_path, "sources"):
            pytest.fail("unpinned source loaded")


@pytest.mark.parametrize("database", ["aumc", "hirid", "miiv", "sic"])
def test_harness_agrees_with_frozen_sql_for_identical_synthetic_inputs(tmp_path, frozen_src_dir, database):
    _peep_source(tmp_path, database)
    report = frozen_equivalence.run_equivalence(
        database=database, db_root=tmp_path, stay_ids=[_STAY, _OTHER] if database == "hirid" else [_STAY],
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


@pytest.mark.parametrize(("database", "column"), [("miiv", None), ("miiv", "stay_id"), ("sic", "CaseID"), ("sic", "stay_id")])
def test_cli_loads_explicit_cohort_file(tmp_path, frozen_src_dir, capsys, monkeypatch, database, column):
    _peep_source(tmp_path, database)
    if column is None:
        cohort = tmp_path / "requested.txt"
        cohort.write_text(f"  {_OTHER}\n\n{_OTHER}\n", encoding="utf-8")
    else:
        cohort = tmp_path / "requested.parquet"
        _write(tmp_path, "requested", {column: [_OTHER], "unused": ["second"]})

    compare = frozen_equivalence.compare_rows

    def compare_cohort(actual, expected, **kwargs):
        assert actual.stay_id.unique().tolist() == [_OTHER]
        return compare(actual, expected, **kwargs)

    monkeypatch.setattr(frozen_equivalence, "compare_rows", compare_cohort)
    status = frozen_equivalence.main([
        "--database", database, "--db-root", str(tmp_path),
        "--stay-ids-file", str(cohort), "--module", "observed_peep_set",
        "--frozen-src-dir", str(frozen_src_dir),
    ])
    out = capsys.readouterr()
    assert json.loads(out.out)["status"] == "agree"
    assert status == 0
    assert str(_STAY) not in out.out and str(_OTHER) not in out.out and not out.err


@pytest.mark.parametrize("database", ["aumc", "hirid", "sic"])
def test_cli_defaults_to_all_stays_outside_miiv(tmp_path, frozen_src_dir, capsys, database):
    _peep_source(tmp_path, database)
    status = frozen_equivalence.main([
        "--database", database, "--db-root", str(tmp_path),
        "--module", "observed_peep_set", "--frozen-src-dir", str(frozen_src_dir),
    ])
    assert json.loads(capsys.readouterr().out)["status"] == "agree"
    assert status == 0


@pytest.mark.parametrize("case", ["miiv_without_cohort", "both_options", "malformed_cohort"])
def test_cli_cohort_errors_hide_input_values(tmp_path, capsys, case):
    cohort = tmp_path / "cohort.txt"
    token = "invalid-stay-720054"
    cohort.write_text(token, encoding="utf-8")
    args = [
        "--database", "miiv" if case == "miiv_without_cohort" else "sic",
        "--db-root", str(tmp_path), "--module", "observed_peep_set",
    ]
    if case != "miiv_without_cohort":
        args += ["--stay-ids-file", str(cohort)]
    if case == "both_options":
        args += ["--stay-ids", str(_STAY)]
    if case == "malformed_cohort":
        assert frozen_equivalence.main(args) == 2
        report = {"status": "comparison_failed", "error_type": "ValueError"}
    else:
        with pytest.raises(SystemExit) as exit_info:
            frozen_equivalence.main(args)
        assert exit_info.value.code == 2
        report = {"status": "invalid_arguments"}
    out = capsys.readouterr()
    assert json.loads(out.out) == report and not out.err


@pytest.mark.parametrize("shard_pruned", [False, True], ids=["empty_cohort", "empty_shard_selection"])
def test_harness_compares_empty_selections(tmp_path, monkeypatch, shard_pruned):
    _peep_source(tmp_path, "sic")
    if shard_pruned:
        resolve = thesis_support.resolve_table_files

        def resolve_without_matching_shards(source, table, **kwargs):
            if table == "data_float_h" and kwargs.get("stay_values") == [_OTHER]:
                return ()
            return resolve(source, table, **kwargs)

        monkeypatch.setattr(thesis_support, "resolve_table_files", resolve_without_matching_shards)
    report = frozen_equivalence.run_equivalence(
        database="sic", db_root=tmp_path, stay_ids=[_OTHER] if shard_pruned else [],
        module="observed_peep_set", frozen_src_dir=tmp_path / "unused_reference",
    )
    assert report["status"] == "agree"
    assert report["module"] == report["frozen"] == {"rows": None, "stays": None}


def test_aggregate_comparison_counts_larger_differences_per_column():
    observed = pd.DataFrame({"stay_id": range(100, 114), "charttime": 0, "observed_peep_set": 12.})
    frozen = observed.assign(observed_peep_set=6.).iloc[::-1]
    result = frozen_equivalence.compare_rows(observed, frozen, key_columns=["stay_id", "charttime"])
    assert result["status"] == "different"
    assert result["mismatching_rows"] == 10
    assert result["mismatching_cells"] == {"stay_id": None, "charttime": None, "observed_peep_set": 10}
    assert result["module"]["stays"] == result["frozen"]["stays"] == 10


_FIRST_IN_HOSP_FIXTURES = {
    "aumc": ("admissioncount", [None, 0, 1, 2, -1], [None, False, True, False, False],
             "admissions.admissioncount = 1"),
    "sic": ("OffsetAfterFirstAdmission", [None, 0, 3, -3], [True, True, False, False],
            "COALESCE(cases.OffsetAfterFirstAdmission, 0) = 0"),
}


def _first_in_hosp_source(root, database, *, extra_nulls=0):
    config = DATABASE_ID_CONFIG[database]
    field, values, expected, expression = _FIRST_IN_HOSP_FIXTURES[database]
    values = values + [None] * extra_nulls
    expected = expected + [None] * extra_nulls
    ids = list(range(_STAY, _STAY + len(values)))
    root.mkdir()
    raw = root / "raw"
    raw.mkdir()
    table = config["table"]
    _write(raw, table, {config["id_col"]: ids, field: pd.array(values, dtype="Int64")})
    path = raw / f"{table}.parquet"
    (root / path.name).symlink_to(path)
    rows = pd.DataFrame({
        config["id_col"]: ids,
        FIRST_IN_HOSP_FROZEN: pd.array(expected, dtype="boolean"),
    })
    return rows, path, field, expression


@pytest.mark.parametrize("database", ["aumc", "sic"])
def test_first_in_hosp_duckdb_predicates_and_selected_stays(tmp_path, database):
    source = tmp_path / "source"
    expected, _, _, _ = _first_in_hosp_source(source, database)
    stay = expected.columns[0]
    for ids, wanted in [
        (None, expected),
        ({stay: [expected[stay].iloc[2], _STAY, _STAY, _OTHER]}, expected.iloc[[0, 2]]),
        ([_OTHER], expected.iloc[:0]),
    ]:
        rows = load_thesis_support(
            module=FIRST_IN_HOSP_FROZEN, database=database, data_path=source, patient_ids=ids,
        )
        pd.testing.assert_frame_equal(rows, wanted.reset_index(drop=True))
        assert rows.attrs["thesis_support"]["status"] == "projected"
        assert frozen_equivalence.run_equivalence(
            database=database, db_root=source, stay_ids=ids, module=FIRST_IN_HOSP_FROZEN,
            frozen_src_dir=tmp_path / "unused_reference",
        )["status"] == "agree"


@pytest.mark.parametrize("database", ["eicu", "hirid", "miiv", "mimic"])
def test_first_in_hosp_absence_before_storage(tmp_path, monkeypatch, database):
    def unexpected_source(*args, **kwargs):
        pytest.fail("out-of-scope first-in-hospital request opened storage")

    monkeypatch.setattr(thesis_support, "storage_source", unexpected_source)
    monkeypatch.setattr(frozen_equivalence, "storage_source", unexpected_source)
    monkeypatch.setattr(datasource, "storage_source", unexpected_source)
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    rows = load_thesis_support(
        module=FIRST_IN_HOSP_FROZEN, database=database, data_path=tmp_path,
    )
    assert rows.empty
    assert rows.dtypes.astype(str).to_dict() == {stay: "int64", FIRST_IN_HOSP_FROZEN: "boolean"}
    receipt = rows.attrs["thesis_support"]
    assert receipt["status"] == "UNAVAILABLE_IN_SCOPE"
    assert receipt["reason"] == "no_frozen_predicate_in_scope"
    assert frozen_equivalence.run_equivalence(
        database=database, db_root=tmp_path, module=FIRST_IN_HOSP_FROZEN,
    ) == {"status": "UNAVAILABLE_IN_SCOPE"}
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "export"
    result = extract_database(
        database, data_path=source, output_dir=output, modules=[FIRST_IN_HOSP_FROZEN], max_workers=1,
    )
    assert result["storage_consumed"] == []
    manifest = json.loads((output / "_manifest.json").read_text())
    entry, = manifest["files"]
    assert entry["thesis_support"] == receipt
    assert entry["availability"] == "structurally_unavailable"
    assert entry["primary_key"] == [stay]
    pd.testing.assert_frame_equal(pd.read_parquet(output / entry["file"]), rows)


@pytest.mark.parametrize("database", ["aumc", "sic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("explicit_cohort", [False, True], ids=["implicit_stays", "explicit_stays"])
def test_native_first_in_hosp_export_has_producer_lineage(tmp_path, database, stream, explicit_cohort):
    source = tmp_path / "source"
    expected, realpath, field, expression = _first_in_hosp_source(source, database)
    stay = expected.columns[0]
    output = tmp_path / "export"
    extract_database(
        database, data_path=source, output_dir=output, modules=[FIRST_IN_HOSP_FROZEN],
        patient_ids=expected[stay].iloc[::-1].tolist() if explicit_cohort else None,
        stream_output_batches=stream, batch_size=1,
        native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entry, = manifest["files"]
    assert entry["primary_key"] == [stay]
    assert entry["row_grain"] == "one_row_per_icu_stay"
    assert entry["availability"] == "available"
    assert pq.read_schema(output / entry["file"]).field(FIRST_IN_HOSP_FROZEN).nullable
    pd.testing.assert_frame_equal(pd.read_parquet(output / entry["file"]), expected)
    if stream:
        assert len(manifest["stream_batches"][FIRST_IN_HOSP_FROZEN]) == len(expected)
    receipt = entry["thesis_support"]
    table = DATABASE_ID_CONFIG[database]["table"]
    assert receipt["status"] == "projected"
    assert receipt["source_table"] == table
    assert receipt["source_field"] == field
    assert receipt["availability_basis"] == "frozen_raw_predicate"
    parameters = {
        "expression": expression,
        "frozen_reference": "build_c565v6.py:99-107",
        "source_realpath": str(realpath),
        "source_sha256": hashlib.sha256(realpath.read_bytes()).hexdigest(),
    }
    assert {key: receipt[key] for key in parameters} == parameters
    producer = json.loads((output / f"{FIRST_IN_HOSP_FROZEN}.manifest.json").read_text())
    assert producer["thesis_support"] == receipt
    sidecar = json.loads((output / manifest["column_metadata"]["file"]).read_text())
    metadata = sidecar["files"][0]["columns"][FIRST_IN_HOSP_FROZEN]["metadata"]
    assert metadata["availability_basis"] == receipt["availability_basis"]
    lineage, = metadata["source_lineage"]
    assert lineage["table"] == table
    assert lineage["value_variable"] == field
    assert lineage["semantic_parameters"] == parameters


def test_first_in_hosp_remains_opt_in():
    assert EXTRACT_MODULES[FIRST_IN_HOSP_FROZEN] == [FIRST_IN_HOSP_FROZEN]
    assert set(EXTRACT_MODULE_ORDER) == set(CONCEPT_GROUPS_INTERNAL)
    assert FIRST_IN_HOSP_FROZEN not in EXTRACT_MODULE_ORDER
    assert all(FIRST_IN_HOSP_FROZEN not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


def test_first_in_hosp_exports_an_empty_selected_cohort(tmp_path):
    source = tmp_path / "source"
    expected, _, _, _ = _first_in_hosp_source(source, "sic")
    output = tmp_path / "export"
    result = extract_database(
        "sic", data_path=source, output_dir=output, modules=[FIRST_IN_HOSP_FROZEN],
        patient_ids=[_OTHER], stream_output_batches=True, max_workers=1,
    )
    assert result["num_patients"] == 0
    entry, = json.loads((output / "_manifest.json").read_text())["files"]
    assert entry["availability"] == "available"
    assert entry["thesis_support"]["status"] == "projected"
    pd.testing.assert_frame_equal(pd.read_parquet(output / entry["file"]), expected.iloc[:0])


def test_first_in_hosp_cli_compares_by_file_and_reports_only_aggregates(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    expected, _, _, _ = _first_in_hosp_source(source, "aumc", extra_nulls=9)
    cohort = tmp_path / "cohort.parquet"
    expected[[expected.columns[0]]].to_parquet(cohort)
    args = [
        "--database", "aumc", "--db-root", str(source), "--module", FIRST_IN_HOSP_FROZEN,
        "--stay-ids-file", str(cohort),
    ]
    assert frozen_equivalence.main(args) == 0
    agreeing = capsys.readouterr()
    assert json.loads(agreeing.out)["status"] == "agree"

    def fill_nulls(**kwargs):
        rows = load_thesis_support(**kwargs)
        rows[FIRST_IN_HOSP_FROZEN] = rows[FIRST_IN_HOSP_FROZEN].fillna(False)
        return rows

    monkeypatch.setattr(frozen_equivalence, "load_thesis_support", fill_nulls)
    assert frozen_equivalence.main(args) == 1
    mismatching = capsys.readouterr()
    report = json.loads(mismatching.out)
    assert report["status"] == "different"
    assert report["mismatching_rows"] == 10
    assert report["mismatching_cells"] == {"stay_id": None, FIRST_IN_HOSP_FROZEN: 10}
    for output in (agreeing, mismatching):
        assert not output.err
        assert all(str(value) not in output.out for value in expected.iloc[:, 0])


@pytest.mark.parametrize("database", ["eicu", "hirid"])
@pytest.mark.parametrize("missing_file", [False, True], ids=["typed_worker_file", "no_worker_file"])
def test_streaming_unavailable_first_in_hosp_seals_native_key_and_receipt(tmp_path, monkeypatch, database, missing_file):
    source = tmp_path / "source"
    source.mkdir()
    if database == "hirid":
        _peep_source(source, database)
    else:
        _stays(source, database)
    if missing_file:
        exists = extraction.os.path.exists

        def no_worker_file(path):
            return False if Path(path).parts[-2:] == (
                FIRST_IN_HOSP_FROZEN, f"{FIRST_IN_HOSP_FROZEN}.parquet",
            ) else exists(path)

        monkeypatch.setattr(extraction.os.path, "exists", no_worker_file)
    output = tmp_path / "export"
    extract_database(
        database, data_path=source, output_dir=output,
        modules=["observed_peep_set", FIRST_IN_HOSP_FROZEN], patient_ids=[_STAY],
        stream_output_batches=True, batch_size=10, native_export_v2=True, max_workers=1,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    entry = next(item for item in manifest["files"] if item["module"] == FIRST_IN_HOSP_FROZEN)
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    assert entry["primary_key"] == [stay]
    assert entry["availability"] == "structurally_unavailable"
    backend = "pandas_structural_placeholder" if missing_file else "pandas_bounded_row_grain_fallback"
    assert entry["row_grain_audit"]["publication_backend"] == backend
    rows = pd.read_parquet(output / entry["file"])
    assert rows.empty
    assert rows.dtypes.astype(str).to_dict() == {stay: "int64", FIRST_IN_HOSP_FROZEN: "boolean"}
    producer = json.loads((output / f"{FIRST_IN_HOSP_FROZEN}.manifest.json").read_text())
    assert producer["thesis_support"] == entry["thesis_support"]
    assert producer["thesis_support"]["status"] == "UNAVAILABLE_IN_SCOPE"


@pytest.mark.parametrize("kind", ["reloaded", "absent", "missing_authority"])
def test_first_in_hosp_metadata_accepts_frames_without_projected_lineage(tmp_path, kind):
    database = "eicu" if kind == "absent" else "aumc"
    if kind == "reloaded":
        source = tmp_path / "source"
        _first_in_hosp_source(source, database)
        output = tmp_path / "export"
        extract_database(database, data_path=source, output_dir=output, modules=[FIRST_IN_HOSP_FROZEN])
        rows = pd.read_parquet(output / f"{FIRST_IN_HOSP_FROZEN}.parquet")
        assert "thesis_support" not in rows.attrs
    else:
        rows = load_thesis_support(module=FIRST_IN_HOSP_FROZEN, database=database, data_path=tmp_path)
    binding = build_export_file_metadata_binding(
        relative_path=f"{FIRST_IN_HOSP_FROZEN}.parquet", module=FIRST_IN_HOSP_FROZEN,
        frame=rows, concept_ids=[FIRST_IN_HOSP_FROZEN], database=database,
        database_class_prefixes=(), dictionary=publication_dictionary(database),
    )
    metadata = binding.columns[FIRST_IN_HOSP_FROZEN].metadata
    assert metadata.available_databases == ("aumc", "sic")
    assert metadata.source_declared_for_database == (database == "aumc")
    assert metadata.source_lineage == ()
    if kind != "reloaded":
        assert metadata.availability_basis == rows.attrs["thesis_support"]["availability_basis"]


@pytest.mark.parametrize("relation", ["same", "below", "above", "symlink"])
def test_absence_only_export_rejects_overlapping_source_and_output(tmp_path, relation):
    source = tmp_path / "source"
    source.mkdir()
    output = {"same": source, "below": source / "export", "above": tmp_path, "symlink": tmp_path / "alias"}[relation]
    if relation == "symlink":
        output.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="overlap"):
        extract_database("miiv", data_path=source, output_dir=output, modules=[FIRST_IN_HOSP_FROZEN])
    assert list(source.iterdir()) == []
    assert not (output / "_manifest.json").exists()


@pytest.mark.parametrize("change", ["contents", "symlink"])
def test_first_in_hosp_cli_rejects_reference_source_changed_after_production(tmp_path, monkeypatch, capsys, change):
    source = tmp_path / "source"
    expected, original, field, _ = _first_in_hosp_source(source, "aumc")
    replacement = original if change == "contents" else original.with_name("replacement.parquet")
    resolve = frozen_equivalence.support_input_files

    def changed_source(*args):
        files = resolve(*args)
        expected.iloc[:, :1].assign(**{field: 1}).to_parquet(replacement, index=False)
        if change == "symlink":
            link = source / original.name
            link.unlink()
            link.symlink_to(replacement)
            return (link,)
        return files

    monkeypatch.setattr(frozen_equivalence, "support_input_files", changed_source)
    assert frozen_equivalence.main([
        "--database", "aumc", "--db-root", str(source), "--module", FIRST_IN_HOSP_FROZEN,
    ]) == 2
    out = capsys.readouterr()
    assert json.loads(out.out) == {"status": "comparison_failed", "error_type": "StorageSourceChanged"}
    assert not out.err
