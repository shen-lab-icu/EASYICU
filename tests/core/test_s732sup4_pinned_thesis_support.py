"""Synthetic pinned D0 inputs and independent frozen equivalence checks."""

from dataclasses import replace
from hashlib import sha256
from io import BufferedReader
import json
from math import inf, nextafter
from pathlib import Path
from sys import float_info
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu import datasource, extract_database
from easyicu.api import extraction
from easyicu.concept import thesis_support as support
from easyicu.config import DATABASE_ID_CONFIG
from easyicu.export_publication import IN_PROGRESS_MARKER
from tools.thesis_support import frozen_equivalence

_IMV = "imv_recorded_compat"
_RRT = "rrt_time"
_EPISODE_GAP = "episode_gap_tolerance_h"
_S1345 = "S1345 narrow config read, 6 files, sha-pinned"


def _pin(root, database, module, **columns):
    path = root / database / f"{module}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(columns).assign(extra_column="must remain unread")
    frame.to_parquet(path, index=False)
    return support.PinnedModuleInput(module, path, sha256(path.read_bytes()).hexdigest(), tuple(columns))


def _rrt_inputs(root, db):
    return (
        _pin(root, db, "outcome", stay_id=[101, 102, 103, 104], los_icu=[2.1 / 24, 1 / 24, 0, 1 / 24]),
        _pin(
            root, db, "renal",
            stay_id=[101] * 7 + [102, 103, 104, 105],
            charttime=[-1.0, 0.0, 1.0, 2.0, 3.0, None, 4.0, 1.0, 0.0, 0.0, 0.0],
            rrt=pd.array([True, True, False, True, True, None, None, True, True, True, True], dtype="boolean"),
        ),
    )


def _span_inputs(root, *, outcome=None):
    return (
        outcome if outcome is not None else _pin(
            root, "hirid", "outcome", stay_id=[101, 102, 103, 104], los_icu=[1 / 24, 4 / 24, 1 / 24, 1 / 24],
        ),
        _pin(
            root, "hirid", "respiratory",
            stay_id=[101, 101, 101, 101, 102, 103, 104, 105],
            charttime=[-3.1, 1.3, 8.9, None, -0.1, 1.3, 0.0, 0.0],
            mech_vent=["INVASIVE", "noninvasive", "invasive", "invasive", "invasive", "invasive", None, "invasive"],
        ),
    )


def _episode_inputs(root, database):
    return (
        _pin(
            root, database, "outcome", stay_id=[101, 102, 103, 104, 105],
            los_icu=[9.5 / 24, 6 / 24, 2 / 24, 8 / 24, 0],
        ),
        _pin(
            root, database, "respiratory",
            stay_id=[101] * 7 + [102, 102, 103, 103, 104, 105],
            charttime=[-0.5, 0.2, 0.5, 3.1, 3.8, 7.4, 10., 1., 5., 2., None, 2., 0.],
            mech_vent=[
                "invasive", "invasive", "invasive", "Invasive", "noninvasive", "invasive", "invasive",
                "invasive", "invasive", "invasive", None, "invasive", "invasive",
            ],
        ),
    )


def _episode_parameter(root, *, resolved=None, **options):
    path = root / "signed-episode-parameters.json"
    path.write_text(json.dumps({
        "rule": _S1345,
        "defined_at": ["config/common_v0.yaml:10"],
        "resolved": {"v0.2": {_EPISODE_GAP: 2}, "v0.4": {_EPISODE_GAP: 0}} if resolved is None else resolved,
    }), encoding="utf-8")
    return support.FrozenReleaseParameter(path, sha256(path.read_bytes()).hexdigest(), **options)


@pytest.fixture(autouse=True)
def _deny_raw_access(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("pinned support attempted raw storage resolution")

    monkeypatch.setattr(support, "storage_source", denied)
    monkeypatch.setattr(frozen_equivalence, "storage_source", denied)
    monkeypatch.setattr(datasource, "storage_source", denied)


@pytest.mark.parametrize("database", ["aumc", "eicu", "hirid", "sic"])
def test_rrt_projects_active_hours_on_the_frozen_spine(tmp_path, monkeypatch, database):
    inputs = _rrt_inputs(tmp_path, database)
    read = support.pq.read_table

    def checked_projection(handle, *, columns, filters):
        assert isinstance(handle, BufferedReader)
        assert set(columns) in [{"stay_id", "los_icu"}, {"stay_id", "charttime", "rrt"}]
        return read(handle, columns=columns, filters=filters)

    monkeypatch.setattr(support.pq, "read_table", checked_projection)
    args = dict(module=_RRT, database=database, pinned_inputs=inputs)
    rows = support.load_thesis_support(**args, patient_ids={DATABASE_ID_CONFIG[database]["id_col"]: [101, 102, 103]})
    assert rows.values.tolist() == [[101, 0., True], [101, 2., True]]
    assert rows.dtypes.astype(str).tolist() == ["int64", "float64", "boolean"]
    receipt = rows.attrs["thesis_support"]
    assert receipt["status"] == "projected"
    assert receipt["source"] == "20261117_d93795ce4_default_r14"
    assert receipt["source_manifest_sha16"] == "7f14448b9d9d1870"
    assert receipt["inputs"]["renal"]["sha256"] == inputs[1].expected_sha256
    report = frozen_equivalence.run_equivalence(**args, stay_ids=[101, 102, 103])
    assert report["status"] == "agree"
    assert report["module"] == report["frozen"] == {"rows": None, "stays": None}


def test_hirid_recorded_span_clamps_start_floors_ends_and_does_not_fill_gaps(tmp_path):
    inputs = _span_inputs(tmp_path)
    args = dict(database="hirid", module=_IMV, pinned_inputs=inputs)
    rows = support.load_thesis_support(**args, patient_ids=[101, 102, 103, 104])
    assert rows.values.tolist() == [[101, 0., True, 8.], [102, 0., True, 0.]]
    assert rows.attrs["thesis_support"]["source_grain"] == "stay"
    assert rows.attrs["thesis_support"]["charttime_role"] == "first_recorded_hour"
    assert frozen_equivalence.run_equivalence(**args, stay_ids=[101, 102, 103, 104])["status"] == "agree"


@pytest.mark.parametrize("database", ["aumc", "miiv", "sic"])
def test_hourly_imv_requires_the_missing_frozen_episode_gap(tmp_path, monkeypatch, database):
    pins = (
        support.PinnedModuleInput("outcome", tmp_path / database / "outcome.parquet", "0" * 64, ("stay_id", "los_icu")),
        support.PinnedModuleInput(
            "respiratory", tmp_path / database / "respiratory.parquet", "0" * 64,
            ("stay_id", "charttime", "mech_vent"),
        ),
    )
    monkeypatch.setattr(support, "open_pinned_modules", lambda *a: pytest.fail("opened blocked IMV inputs"))
    rows = support.load_thesis_support(database=database, module=_IMV, pinned_inputs=pins, patient_ids=[101])
    assert rows.empty
    assert rows.attrs["thesis_support"]["reason"] == "no_raw_producer"
    assert rows.attrs["thesis_support"]["missing_inputs"] == ["release.params.cohort.episode_gap_tolerance_h"]
    assert "episode_gap_tolerance_h" not in support.read_hourly_contract().get("cohort", {})


@pytest.mark.parametrize(("database", "module", "reason"), [
    ("eicu", _IMV, None), ("mimic", _IMV, None), ("mimic", _RRT, None),
    ("miiv", _RRT, "no_exact_producer"),
])
def test_final_absence_does_not_open_pinned_files(monkeypatch, database, module, reason):
    monkeypatch.setattr(support, "open_pinned_modules", lambda *a: pytest.fail("opened out-of-scope inputs"))
    rows = support.load_thesis_support(database=database, module=module, patient_ids=[101])
    assert rows.empty
    assert rows.attrs["available_fields"] == []
    receipt = rows.attrs["thesis_support"]
    if reason:
        assert receipt["status"] == "UNAVAILABLE_IN_SCOPE"
        assert receipt["reason"] == reason
        assert "actual end" in receipt["missing_inputs"][0]
    else:
        assert receipt["status"] == "no_frozen_reducer"


@pytest.mark.parametrize("module", [_IMV, _RRT])
def test_miiv_requires_explicit_allowed_members(module):
    with pytest.raises(ValueError, match="explicit allowed stay cohort"):
        support.load_thesis_support(database="miiv", module=module)
    with pytest.raises(support.PinnedInputError, match="pinned_cohort_requires_stay_ids"):
        support.load_thesis_support(database="miiv", module=module, patient_ids={"subject_id": [101]})
    assert support.load_thesis_support(database="miiv", module=module, patient_ids=[]).empty


@pytest.mark.parametrize(("failure", "reason"), [
    ("digest", "pinned_sha256_mismatch"), ("missing", "pinned_file_missing"),
])
def test_all_files_are_verified_before_the_first_parquet_decode(tmp_path, monkeypatch, failure, reason):
    outcome, renal = _rrt_inputs(tmp_path, "eicu")
    if failure == "digest":
        renal = replace(renal, expected_sha256="0" * 64)
    else:
        renal.path.unlink()
    monkeypatch.setattr(support.pq, "ParquetFile", lambda *a: pytest.fail("decoded before checking every hash"))
    monkeypatch.setattr(support.pq, "read_table", lambda *a, **kw: pytest.fail("decoded before checking every hash"))
    with pytest.raises(support.PinnedInputError) as error:
        support.load_thesis_support(database="eicu", module=_RRT, pinned_inputs=(outcome, renal))
    assert error.value.reason_code == reason


def test_pinned_scope_refuses_a_raw_table_forbidden_pair_and_extra_column_before_access(tmp_path, monkeypatch):
    outcome, renal = _rrt_inputs(tmp_path, "eicu")
    monkeypatch.setattr(support, "read_hourly_contract", lambda: pytest.fail("access before scope validation"))
    checks = [
        (_RRT, replace(renal, module="treatment"), "pinned_module_not_allowed"),
        (_RRT, replace(renal, path=renal.path.with_name("treatment.parquet")), "pinned_path_not_allowed"),
        (_RRT, replace(renal, allowed_columns=(*renal.allowed_columns, "extra_column")), "pinned_columns_not_allowed"),
        (_IMV, replace(renal, module="respiratory"), "pinned_module_not_allowed"),
        (_IMV, outcome, "pinned_module_not_allowed"),
    ]
    for module, pin, reason in checks:
        with pytest.raises(support.PinnedInputError) as error:
            support.load_thesis_support(database="eicu", module=module, pinned_inputs=[pin])
        assert error.value.reason_code == reason


def test_pinned_scope_rejects_symlinks_to_raw_files(tmp_path):
    outcome, renal = _rrt_inputs(tmp_path, "eicu")
    raw = renal.path.with_name("treatment.parquet")
    renal.path.rename(raw)
    renal.path.symlink_to(raw)
    with pytest.raises(support.PinnedInputError, match="pinned_path_not_allowed"):
        support.load_thesis_support(database="eicu", module=_RRT, pinned_inputs=(outcome, renal))


def test_incomplete_declaration_names_missing_input_without_hashing(tmp_path, monkeypatch):
    outcome, _ = _rrt_inputs(tmp_path, "sic")
    monkeypatch.setattr(support, "open_pinned_modules", lambda *a: pytest.fail("opened an incomplete binding"))
    rows = support.load_thesis_support(database="sic", module=_RRT, pinned_inputs=[outcome])
    assert rows.empty and rows.attrs["thesis_support"]["missing_inputs"] == ["sic/renal.parquet:rrt"]


@pytest.mark.parametrize("clock", [3.5, float("inf")])
def test_rrt_rejects_invalid_target_bearing_hours_even_outside_the_spine(tmp_path, clock):
    outcome, _ = _rrt_inputs(tmp_path, "aumc")
    renal = _pin(tmp_path, "aumc", "renal", stay_id=[101], charttime=[clock], rrt=[True])
    with pytest.raises(support.PinnedInputError, match="pinned_rrt_hour_invalid"):
        support.load_thesis_support(database="aumc", module=_RRT, pinned_inputs=[outcome, renal])


@pytest.mark.parametrize("module", [_RRT, _IMV])
def test_cli_detects_a_planted_mismatch_and_suppresses_small_counts(tmp_path, monkeypatch, capsys, module):
    database = "hirid"
    pins = _rrt_inputs(tmp_path, database) if module == _RRT else _span_inputs(tmp_path)
    args = ["--database", database, "--module", module, "--stay-ids", "101"]
    for pin in pins:
        args.extend(["--pinned-input", pin.module, str(pin.path), pin.expected_sha256, ",".join(pin.allowed_columns)])
    assert frozen_equivalence.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "agree"
    original = frozen_equivalence.load_thesis_support

    def planted(**kwargs):
        rows = original(**kwargs)
        rows.loc[0, "charttime"] += 1
        return rows

    monkeypatch.setattr(frozen_equivalence, "load_thesis_support", planted)
    assert frozen_equivalence.main(args) == 1
    captured = capsys.readouterr()
    assert not captured.err and "101" not in captured.out and str(tmp_path) not in captured.out
    report = json.loads(captured.out)
    assert report["status"] == "different"
    assert report["mismatching_rows"] is None
    assert all(count is None for count in report["mismatching_cells"].values())


def test_native_export_consumes_only_pinned_modules_and_seals_both_receipts(tmp_path):
    outcome, renal = _rrt_inputs(tmp_path / "rrt", "hirid")
    pins = {_RRT: (outcome, renal), _IMV: _span_inputs(tmp_path / "imv", outcome=outcome)}
    output = tmp_path / "output"
    result = extract_database(
        "hirid", pinned_inputs=pins, patient_ids=[101, 102], output_dir=output, native_export_v2=True,
    )
    assert result["num_patients"] == 2
    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["time_window_authority"] == {
        "required": True, "source": "pinned_outcome.los_icu", "bounded_stays": 2,
    }
    files = {item["module"]: item for item in manifest["files"]}
    assert files[_IMV]["thesis_support"]["source_grain"] == "stay"
    assert files[_RRT]["thesis_support"]["inputs"]["renal"]["sha256"] == pins[_RRT][1].expected_sha256
    assert all(entry["availability"] == "available" for entry in files.values())
    assert pd.read_parquet(output / f"{_IMV}.parquet").values.tolist() == [[101, 0., True, 8.], [102, 0., True, 0.]]
    assert pd.read_parquet(output / f"{_RRT}.parquet").values.tolist() == [[101, 0., True], [101, 2., True]]


def test_native_export_seals_typed_absence_without_pinned_miiv_columns(tmp_path):
    output = tmp_path / "output"
    extract_database(
        "miiv", modules=[_IMV, _RRT], pinned_inputs={}, patient_ids=[101], output_dir=output,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    assert [entry["thesis_support"]["reason"] for entry in manifest["files"]] == ["no_raw_producer", "no_exact_producer"]
    assert all(entry["availability"] == "structurally_unavailable" for entry in manifest["files"])
    for module in (_IMV, _RRT):
        assert pd.read_parquet(output / f"{module}.parquet").empty


def test_support_only_export_without_pins_never_discovers_raw_stays(tmp_path):
    result = extract_database("eicu", data_path=tmp_path / "raw", modules=[_IMV, _RRT])
    assert result["modules"][_IMV]["concepts"][_IMV].empty
    assert result["modules"][_RRT]["concepts"][_RRT].empty
    assert result["modules"][_RRT]["thesis_support"]["missing_inputs"] == [
        "eicu/renal.parquet:rrt", "eicu/outcome.parquet:los_icu",
    ]


@pytest.mark.parametrize("existing_marker", [False, True], ids=["unowned_file", "unfinished_writer"])
def test_pinned_export_checks_destination_before_starting_publication(tmp_path, monkeypatch, existing_marker):
    output = tmp_path / "busy"
    output.mkdir()
    existing = output / (IN_PROGRESS_MARKER if existing_marker else "unpublished.parquet")
    existing.write_bytes(b"existing writer evidence")
    entered = []
    begin = extraction.begin_native_export

    def track_begin(root):
        entered.append(root)
        return begin(root)

    monkeypatch.setattr(extraction, "begin_native_export", track_begin)
    with pytest.raises(ValueError):
        extract_database("sic", modules=[_RRT], output_dir=output)
    assert entered == [], "a nonempty destination must be rejected before creating a publication marker"
    assert list(output.iterdir()) == [existing]
    assert existing.read_bytes() == b"existing writer evidence"


@pytest.mark.parametrize("empty_selection", [False, True], ids=["typed_absence", "empty_explicit_cohort"])
def test_pinned_export_never_claims_unread_outcome(tmp_path, empty_selection):
    pins = {_RRT: _rrt_inputs(tmp_path, "sic")} if empty_selection else None
    output = tmp_path / "output"
    result = extract_database(
        "sic", modules=[_RRT], pinned_inputs=pins,
        patient_ids=[] if empty_selection else None, output_dir=output,
    )
    assert result["num_patients"] == 0
    assert result["pinned_outcome_stays"] is None
    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["time_window_authority"] == {
        "required": False, "source": "fallback_only", "bounded_stays": 0,
    }
    assert "pinned outcome.los_icu" not in manifest["canonical_physical_schema"]["time_window_policy"]


def test_pinned_export_rejects_conflicting_shared_outcome(tmp_path):
    pins = {_RRT: _rrt_inputs(tmp_path / "rrt", "hirid"), _IMV: _span_inputs(tmp_path / "imv")}
    output = tmp_path / "output"
    with pytest.raises(support.PinnedInputError, match="pinned_shared_input_conflict: outcome"):
        extract_database("hirid", pinned_inputs=pins, patient_ids=[101, 102, 103], output_dir=output)
    assert not output.exists()


def test_pinned_path_resolution_error_is_sanitized(tmp_path):
    outcome, renal = _rrt_inputs(tmp_path, "sic")
    renal.path.unlink()
    renal.path.symlink_to(renal.path.name)
    with pytest.raises(support.PinnedInputError) as error:
        support.load_thesis_support(database="sic", module=_RRT, pinned_inputs=[outcome, renal])
    assert error.value.reason_code == "pinned_file_unreadable"
    assert str(tmp_path) not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("database", ["aumc", "miiv", "sic"])
def test_first_hourly_imv_episode_obeys_frozen_gap_and_stay_membership(tmp_path, monkeypatch, database):
    parameter = _episode_parameter(tmp_path)
    inputs = _episode_inputs(tmp_path, database)
    read = support.pq.read_table

    def checked_projection(handle, *, columns, filters):
        assert isinstance(handle, BufferedReader)
        assert set(columns) in [{"stay_id", "los_icu"}, {"stay_id", "charttime", "mech_vent"}]
        assert set(filters[0][2]) <= {101, 102, 103, 105}
        return read(handle, columns=columns, filters=filters)

    monkeypatch.setattr(support.pq, "read_table", checked_projection)
    ids = [101, 102, 103, 105]
    args = dict(database=database, module=_IMV, pinned_inputs=inputs, frozen_release_parameter=parameter)
    rows = support.load_thesis_support(**args, patient_ids=ids)
    assert rows.values.tolist() == [[101, 0., True, 3.], [102, 1., True, 1.]]
    assert rows.attrs["pinned_cohort_size"] == 3
    assert rows.attrs["thesis_support"]["frozen_release_parameter"] == {
        "key_path": ["cohort", _EPISODE_GAP],
        "common_version": "v0.2",
        "value": 2,
        "receipt_sha256": parameter.expected_sha256,
        "defined_at": ["config/common_v0.yaml:10"],
        "rule": _S1345,
    }
    assert frozen_equivalence.run_equivalence(**args, stay_ids=ids)["status"] == "agree"

    out = tmp_path / "export"
    extract_database(
        database, pinned_inputs={_IMV: inputs}, frozen_release_parameter=parameter,
        patient_ids=ids, output_dir=out,
    )
    receipt = rows.attrs["thesis_support"]
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["time_window_authority"]["bounded_stays"] == 3
    assert manifest["files"][0]["thesis_support"] == receipt
    assert json.loads((out / f"{_IMV}.manifest.json").read_text())["pinned_thesis_support"] == receipt
    pd.testing.assert_frame_equal(
        support.pq.ParquetFile(out / f"{_IMV}.parquet").read().to_pandas(),
        rows.rename(columns={rows.columns[0]: "stay_id"}),
    )


@pytest.mark.parametrize("database", ["aumc", "miiv", "sic"])
def test_hourly_imv_case_and_unordered_singletons_are_cohort_invariant(tmp_path, database):
    inputs = (
        _pin(tmp_path, database, "outcome", stay_id=[101, 102], los_icu=[3 / 24, 3 / 24]),
        _pin(
            tmp_path, database, "respiratory",
            stay_id=[101, 101, 102, 102, 102],
            charttime=[0.2, 0.5, 0.0, 1.0, 9.0],
            mech_vent=["invasive", "NonInvasive", "Invasive", "standby confirmed", "source disconnected"],
        ),
    )
    args = dict(database=database, module=_IMV, pinned_inputs=inputs,
                frozen_release_parameter=_episode_parameter(tmp_path))
    only = support.load_thesis_support(**args, patient_ids=[102])
    both = support.load_thesis_support(**args, patient_ids=[101, 102])
    assert only.values.tolist() == both.iloc[1:].values.tolist() == [[102, 0., True, 0.]]
    for ids in ([102], [101, 102]):
        assert frozen_equivalence.run_equivalence(**args, stay_ids=ids)["status"] == "agree"


def test_hourly_imv_excludes_off_spine_and_other_stay_rows_before_cast_and_validation(tmp_path):
    inputs = (
        _pin(tmp_path, "sic", "outcome", stay_id=[101, 102], los_icu=[2 / 24, 2 / 24]),
        _pin(
            tmp_path, "sic", "respiratory",
            stay_id=[102, 102, 102, 102, 102, 101],
            charttime=[0.1, 1.7e12, -1.8e12, 3.1, 3.8, None],
            mech_vent=["invasive", "invasive", "invasive", "standby confirmed", "noninvasive", "invasive"],
        ),
    )
    args = dict(database="sic", module=_IMV, pinned_inputs=inputs,
                frozen_release_parameter=_episode_parameter(tmp_path))
    with np.errstate(invalid="raise"):
        rows = support.load_thesis_support(**args, patient_ids=[102])
        report = frozen_equivalence.run_equivalence(**args, stay_ids=[102])
    assert rows.values.tolist() == [[102, 0., True, 0.]]
    assert report["status"] == "agree"


def test_hourly_imv_refuses_an_unordered_mode_in_an_on_spine_collision(tmp_path):
    inputs = (
        _pin(tmp_path, "aumc", "outcome", stay_id=[101, 102], los_icu=[2 / 24, 2 / 24]),
        _pin(
            tmp_path, "aumc", "respiratory",
            stay_id=[101, 101, 102], charttime=[1.2, 1.7, 1.2],
            mech_vent=["invasive", "standby confirmed", "invasive"],
        ),
    )
    args = dict(database="aumc", module=_IMV, pinned_inputs=inputs,
                frozen_release_parameter=_episode_parameter(tmp_path))
    assert support.load_thesis_support(**args, patient_ids=[102]).values.tolist() == [[102, 1., True, 1.]]
    with pytest.raises(support.PinnedInputError) as error:
        support.load_thesis_support(**args, patient_ids=[101, 102])
    assert error.value.reason_code == "pinned_vent_type_invalid"


def test_miiv_demo_uses_the_release_family_with_explicit_stays(tmp_path, capsys):
    parameter = _episode_parameter(tmp_path, common_version="v0.4")
    inputs = _episode_inputs(tmp_path, "miiv")
    args = dict(database="miiv_demo", module=_IMV, pinned_inputs=inputs, frozen_release_parameter=parameter)
    rows = support.load_thesis_support(**args, patient_ids={"stay_id": [102]})
    assert rows.values.tolist() == [[102, 1., True, 1.]]
    assert rows.attrs["thesis_support"]["frozen_release_parameter"]["common_version"] == "v0.4"
    assert frozen_equivalence.run_equivalence(**args, stay_ids=[102])["status"] == "agree"
    cohort = tmp_path / "night-rounds-stays.parquet"
    pd.DataFrame({"stay_id": [102]}).to_parquet(cohort, index=False)
    cli = [
        "--database", "miiv_demo", "--module", _IMV, "--stay-ids-file", str(cohort),
        "--frozen-release-parameter", str(parameter.path), parameter.expected_sha256,
        "--common-version", parameter.common_version,
    ]
    for pin in inputs:
        cli.extend(["--pinned-input", pin.module, str(pin.path), pin.expected_sha256, ",".join(pin.allowed_columns)])
    assert frozen_equivalence.main(cli) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "agree"

    output = tmp_path / "night-rounds-export"
    extract_database(
        "miiv_demo", pinned_inputs={_IMV: inputs}, frozen_release_parameter=parameter,
        patient_ids=[102], output_dir=output,
    )
    assert pd.read_parquet(output / f"{_IMV}.parquet").values.tolist() == rows.values.tolist()


@pytest.mark.parametrize(("changes", "reason"), [
    ({"key_path": ()}, "frozen_release_parameter_not_allowed"),
    ({"key_path": ("quality", "overnight_margin")}, "frozen_release_parameter_not_allowed"),
    ({"key_path": ("cohort", False)}, "frozen_release_parameter_not_allowed"),
    ({"common_version": ["overnight"]}, "frozen_release_parameter_version_invalid"),
    ({"expected_sha256": 64}, "frozen_release_parameter_sha256_invalid"),
    ({"expected_sha256": "z" * 64}, "frozen_release_parameter_sha256_invalid"),
    ({"expected_sha256": "b" * 63}, "frozen_release_parameter_sha256_invalid"),
    ({"path": False}, "frozen_release_parameter_path_not_allowed"),
    ({"path": Path("night-rounds.yaml")}, "frozen_release_parameter_path_not_allowed"),
], ids=["empty-key", "unrelated-key", "non-string-key", "version-type", "digest-type", "non-hex",
        "short-digest", "path-type", "non-json"])
def test_public_release_parameter_reader_validates_fields_before_resolving(tmp_path, monkeypatch, changes, reason):
    parameter = replace(_episode_parameter(tmp_path), **changes)
    resolve = Path.resolve

    def guarded_resolve(path, *args, **kwargs):
        assert path != parameter.path, "malformed release receipt reached file resolution"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(support.PinnedInputError) as error:
        support.read_frozen_release_parameter(parameter)
    assert error.value.reason_code == reason


def test_public_release_parameter_reader_rejects_lookalike_before_resolving(tmp_path, monkeypatch):
    parameter = SimpleNamespace(**vars(_episode_parameter(tmp_path)))
    resolve = Path.resolve

    def guarded_resolve(path, *args, **kwargs):
        assert path != parameter.path, "an untyped receipt reached file resolution"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(support.PinnedInputError) as error:
        support.read_frozen_release_parameter(parameter)
    assert error.value.reason_code == "frozen_release_parameter_not_allowed"


def test_public_release_parameter_reader_canonicalizes_hex_digest(tmp_path):
    parameter = _episode_parameter(tmp_path)
    upper = replace(parameter, expected_sha256=parameter.expected_sha256.upper())
    assert support.read_frozen_release_parameter(upper) == support.read_frozen_release_parameter(parameter)


@pytest.mark.parametrize(("failure", "reason"), [
    ("sha256", "frozen_release_parameter_sha256_mismatch"),
    ("version", "frozen_release_parameter_version_missing"),
    ("key", "frozen_release_parameter_key_missing"),
])
def test_bad_frozen_release_parameter_fails_closed_before_parquet(tmp_path, monkeypatch, failure, reason):
    if failure == "key":
        parameter = _episode_parameter(tmp_path, resolved={"v0.2": {}})
    else:
        parameter = _episode_parameter(tmp_path)
        parameter = (
            replace(parameter, expected_sha256="f" * 64) if failure == "sha256"
            else replace(parameter, common_version="v9.4")
        )
    pins = _episode_inputs(tmp_path, "sic")
    monkeypatch.setattr(support, "open_pinned_modules", lambda *a: pytest.fail("opened data before resolving the gap"))
    with pytest.raises(support.PinnedInputError) as error:
        support.load_thesis_support(database="sic", module=_IMV, pinned_inputs=pins, frozen_release_parameter=parameter)
    assert error.value.reason_code == reason
    assert str(tmp_path) not in str(error.value)


def test_cli_uses_requested_gap_version_and_catches_planted_episode_end(tmp_path, monkeypatch, capsys):
    parameter = _episode_parameter(tmp_path, common_version="v0.4")
    pins = _episode_inputs(tmp_path, "miiv")
    args = [
        "--database", "miiv", "--module", _IMV, "--stay-ids", "101",
        "--frozen-release-parameter", str(parameter.path), parameter.expected_sha256,
        "--common-version", parameter.common_version,
    ]
    for pin in pins:
        args.extend(["--pinned-input", pin.module, str(pin.path), pin.expected_sha256, ",".join(pin.allowed_columns)])
    assert frozen_equivalence.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "agree"
    original = frozen_equivalence.load_thesis_support

    def planted(**kwargs):
        rows = original(**kwargs)
        assert rows.imv_recorded_end_hour.tolist() == [0.]
        rows.loc[0, "imv_recorded_end_hour"] = 3.
        return rows

    monkeypatch.setattr(frozen_equivalence, "load_thesis_support", planted)
    assert frozen_equivalence.main(args) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "different"
    assert not captured.err and str(tmp_path) not in captured.out and "101" not in captured.out


@pytest.mark.parametrize(("bad_hour", "los", "reason"), [
    (float("inf"), 3 / 24, "pinned_vent_hour_invalid"),
    (float("-inf"), 3 / 24, "pinned_vent_hour_invalid"),
    ("inf", 3 / 24, "pinned_vent_hour_invalid"),
    ("-inf", 3 / 24, "pinned_vent_hour_invalid"),
    (1.75, float("inf"), "pinned_icu_los_invalid"),
    (1.75, float("-inf"), "pinned_icu_los_invalid"),
    (1.75, 1e308, "pinned_icu_los_invalid"),
    (1.75, nextafter(float_info.max / 24.0, inf), "pinned_icu_los_invalid"),
], ids=["positive-float-hour", "negative-float-hour", "positive-text-hour", "negative-text-hour",
        "positive-infinite-los", "negative-infinite-los", "overnight-los-overflow", "los-just-over-hour-limit"])
@pytest.mark.parametrize("entry", ["producer", "harness", "reference"])
def test_overnight_imv_rejects_infinite_clock_inputs(tmp_path, monkeypatch, capsys, bad_hour, los, reason, entry):
    database = "sic"
    stays = [7301, 7302]
    parameter = _episode_parameter(tmp_path)
    good_hour = "0.25" if isinstance(bad_hour, str) else 0.25
    second_mode = "invasive" if reason == "pinned_icu_los_invalid" else "standby at handover"
    inputs = (
        _pin(tmp_path, database, "outcome", stay_id=stays, los_icu=[3 / 24, los]),
        _pin(
            tmp_path, database, "respiratory", stay_id=stays,
            charttime=[good_hour, bad_hour], mech_vent=["InVaSiVe", second_mode],
        ),
    )
    if entry == "producer":
        with np.errstate(invalid="raise"), pytest.raises(support.PinnedInputError) as error:
            support.load_thesis_support(
                module=_IMV, database=database, patient_ids=stays,
                pinned_inputs=inputs, frozen_release_parameter=parameter,
            )
        assert error.value.reason_code == reason
        return

    if entry == "reference":
        # Supply a projected producer result to exercise the independent frozen path.
        clean_root = tmp_path / "clean-shift"
        clean_pins = (
            _pin(clean_root, database, "outcome", stay_id=stays, los_icu=[3 / 24, 3 / 24]),
            _pin(clean_root, database, "respiratory", stay_id=[stays[0]], charttime=[0.25], mech_vent=["invasive"]),
        )
        projected = support.load_thesis_support(
            module=_IMV, database=database, patient_ids=stays,
            pinned_inputs=clean_pins, frozen_release_parameter=parameter,
        )
        monkeypatch.setattr(frozen_equivalence, "load_thesis_support", lambda **options: projected)

    args = [
        "--database", database, "--module", _IMV, "--stay-ids", *map(str, stays),
        "--frozen-release-parameter", str(parameter.path), parameter.expected_sha256,
    ]
    for pin in inputs:
        args.extend(["--pinned-input", pin.module, str(pin.path), pin.expected_sha256, ",".join(pin.allowed_columns)])
    with np.errstate(invalid="raise"):
        status = frozen_equivalence.main(args)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "comparison_failed", "reason": reason}
    assert status == 2
    assert not captured.err


def test_direct_pinned_stay_ids_requires_explicit_miiv_demo_cohort():
    with pytest.raises(ValueError, match="MIIV thesis support requires an explicit allowed stay cohort"):
        support.pinned_stay_ids("miiv_demo", None)


def test_direct_pinned_stay_ids_accepts_aumc_demo_mapping():
    assert support.pinned_stay_ids("aumc_demo", {"admissionid": [8462]}) == {8462}


def test_direct_pinned_stay_ids_accepts_database_keyword():
    assert support.pinned_stay_ids(database="miiv_demo", patient_ids={"stay_id": [8462]}) == {8462}


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "database-keyword"])
def test_direct_pin_validation_accepts_single_demo_alias(tmp_path, keyword):
    inputs = _episode_inputs(tmp_path / "relief-team", "miiv")
    if keyword:
        support.validate_pinned_inputs(module=_IMV, database="miiv_demo", inputs=inputs)
    else:
        support.validate_pinned_inputs(_IMV, "miiv_demo", inputs)


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "database-keyword"])
def test_direct_release_validation_accepts_single_demo_alias(tmp_path, keyword):
    parameter = _episode_parameter(tmp_path)
    if keyword:
        support.validate_frozen_release_parameter(module=_IMV, database="miiv_demo", parameter=parameter)
    else:
        support.validate_frozen_release_parameter(_IMV, "miiv_demo", parameter)


@pytest.mark.parametrize("database", ["miiv_overnight", "miiv_demo_demo"])
@pytest.mark.parametrize("helper", ["cohort", "pins", "release"])
def test_direct_pinned_helpers_reject_unknown_databases(tmp_path, database, helper):
    parameter = support.FrozenReleaseParameter(tmp_path / "relief-team.json", "0" * 64)
    with pytest.raises(support.PinnedInputError) as error:
        if helper == "cohort":
            support.pinned_stay_ids(database, [8462])
        elif helper == "pins":
            support.validate_pinned_inputs(_IMV, database, ())
        else:
            support.validate_frozen_release_parameter(_IMV, database, parameter)
    assert error.value.reason_code == "pinned_database_not_allowed"


@pytest.mark.parametrize("database", ["miiv_demo_demo", "aumc_demo_demo", "miiv_overnight"])
@pytest.mark.parametrize("entry", ["extract", "load", "compare", "manifest", "open", "cli"])
def test_pinned_entries_reject_unknown_database_names_before_access(tmp_path, monkeypatch, database, entry, capsys):
    family = "aumc" if database.startswith("aumc") else "miiv"
    pins = _episode_inputs(tmp_path, family)
    parameter = _episode_parameter(tmp_path)

    def deny_access(*args, **kwargs):
        pytest.fail("invalid overnight database name reached an input path")

    monkeypatch.setattr(support, "_resolve_pinned_module_path", deny_access)
    monkeypatch.setattr(support, "read_frozen_release_parameter", deny_access)
    monkeypatch.setattr(frozen_equivalence, "_read_cohort_file", deny_access)
    monkeypatch.setattr(frozen_equivalence, "read_frozen_release_parameter", deny_access)
    options = dict(database=database, module=_IMV, pinned_inputs=pins, frozen_release_parameter=parameter)
    if entry == "cli":
        status = frozen_equivalence.main([
            "--database", database, "--module", _IMV, "--stay-ids-file", str(tmp_path / "overnight-stays.txt"),
        ])
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {"status": "comparison_failed", "reason": "pinned_database_not_allowed"}
        assert status == 2
        assert not captured.err
        return

    with pytest.raises(support.PinnedInputError) as error:
        if entry == "extract":
            extract_database(database, pinned_inputs={_IMV: pins}, frozen_release_parameter=parameter, patient_ids=[102])
        elif entry == "load":
            support.load_thesis_support(**options, patient_ids=[102])
        elif entry == "compare":
            frozen_equivalence.run_equivalence(**options, stay_ids=[102])
        elif entry == "manifest":
            support.thesis_support_manifest(_IMV, database, ())
        else:
            with support.open_pinned_modules(_IMV, database, pins):
                pytest.fail("invalid database opened pinned modules")
    assert error.value.reason_code == "pinned_database_not_allowed"
    assert database not in str(error.value)
