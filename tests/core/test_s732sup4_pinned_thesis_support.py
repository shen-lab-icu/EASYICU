"""Synthetic pinned D0 inputs and independent frozen equivalence checks."""

from dataclasses import replace
from hashlib import sha256
from io import BufferedReader
import json

import pandas as pd
import pytest

from easyicu import datasource, extract_database
from easyicu.concept import thesis_support as support
from easyicu.config import DATABASE_ID_CONFIG
from tools.thesis_support import frozen_equivalence

_IMV = "imv_recorded_compat"
_RRT = "rrt_time"


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


def _span_inputs(root):
    return (
        _pin(root, "hirid", "outcome", stay_id=[101, 102, 103, 104], los_icu=[1 / 24, 4 / 24, 1 / 24, 1 / 24]),
        _pin(
            root, "hirid", "respiratory",
            stay_id=[101, 101, 101, 101, 102, 103, 104, 105],
            charttime=[-3.1, 1.3, 8.9, None, -0.1, 1.3, 0.0, 0.0],
            mech_vent=["INVASIVE", "noninvasive", "invasive", "invasive", "invasive", "invasive", None, "invasive"],
        ),
    )


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
    pins = {_RRT: _rrt_inputs(tmp_path / "rrt", "hirid"), _IMV: _span_inputs(tmp_path / "imv")}
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
