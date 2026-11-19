"""Native web publication and research intake share the ICU-episode contract."""

from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.intake import export_package as intake
from easyicu.webserver import dataio, stop_next_actions
from tests.support.typed_export import typed_export


class ExportJob:
    def emit(self, payload):
        pass


# A forked worker is how the exit is injected. macOS keeps these batches
# in-process (S1458: a forked worker spun in DuckDB); the macos case checks
# that the same export completes there without a worker.
_LINUX_FORK = pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "darwin",
                                 reason="isolated nested batches fork on Linux only")


@pytest.mark.parametrize("worker_exit,platform", [
    pytest.param(0, "linux", id="clean", marks=_LINUX_FORK), pytest.param(9, "linux", id="exit-9", marks=_LINUX_FORK),
    pytest.param(0, "darwin", id="macos"),
])
def test_hosted_web_export_requires_complete_nested_los_batches(tmp_path, monkeypatch, worker_exit, platform):
    import psutil

    from easyicu import api
    from easyicu.api import concepts as concept_api
    from easyicu.api import extraction
    from easyicu.concept.errors import ConceptExtractionUnavailable
    from easyicu.runtime import memory_manager
    from easyicu.webserver.jobs import JobManager
    from tests.support.bucket_storage import b6_source

    source = tmp_path / "input"
    b6_source(source, "miiv")
    stays_path = source / "icustays.parquet"
    cohort_size = 5_001
    ids = list(range(1, cohort_size + 1))
    stays = pd.read_parquet(stays_path).iloc[[0, 1] + [2] * (cohort_size - 2)].reset_index(drop=True)
    stays["stay_id"] = ids
    stays["subject_id"] = stays["stay_id"] + 10
    stays["hadm_id"] = stays["stay_id"] + 20
    stays["los"] = [1.0, 1.5] + [2.0] * (cohort_size - 2)
    stays["outtime"] = stays["intime"] + pd.to_timedelta(stays["los"], unit="D")
    stays.to_parquet(stays_path, index=False)
    pd.DataFrame({
        "subject_id": stays["subject_id"],
        "anchor_age": [42, 58] + [67] * (cohort_size - 2), "anchor_year": 2150, "gender": "F",
    }).to_parquet(source / "patients.parquet", index=False)
    chart_path = source / "chartevents" / "1.parquet"
    chart = pd.read_parquet(chart_path)
    late_batch_hr = chart.loc[chart["itemid"].eq(220045)].iloc[[0, -1]].copy()
    late_batch_hr = late_batch_hr.assign(
        stay_id=cohort_size, subject_id=cohort_size + 10, hadm_id=cohort_size + 20,
        valuenum=[86.0, 92.0],
    )
    late_batch_hr["charttime"] = stays["intime"].iloc[-1] + pd.to_timedelta([5, 4500], unit="m")
    late_batch_hr["storetime"] = late_batch_hr["charttime"]
    pd.concat([chart, late_batch_hr], ignore_index=True).to_parquet(chart_path, index=False)
    home = tmp_path / "instance"
    home.mkdir()
    out = home / "export"
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(source))
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.delenv("EASYICU_FORCE_INPROCESS_BATCH", raising=False)
    monkeypatch.delenv("EASYICU_FORCE_SUBPROCESS_BATCH", raising=False)
    memory = psutil.virtual_memory()._replace(total=8 * 1024**3, available=3 * 1024**3)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    if platform == "darwin":
        policy = concept_api._memory_policy_isolates_batches
        monkeypatch.setattr(concept_api, "_memory_policy_isolates_batches",
                            lambda total, **_: policy(total, platform="darwin"))
    assert api.plan_extraction_resources("miiv", ["demographics"], cohort_size).mode == "one_shot"
    plan = api.plan_extraction_resources("miiv", ["vitals"], cohort_size)
    assert plan.reason_code == "measured_profile_insufficient_memory"
    batch_size = plan.batch_size
    assert batch_size < cohort_size <= 2 * batch_size

    bounds = extraction._native_export_stay_time_upper_bounds
    expected = bounds(api.load_concepts(
        ["los_icu"], database="miiv", data_path=str(source),
        patient_ids={"stay_id": ids}, batch_size=cohort_size, verbose=False,
    ))
    assert expected == {**dict.fromkeys(ids, 72.0), 1: 48.0, 2: 60.0}
    bound_calls = []

    def observe_bounds(frame):
        result = bounds(frame)
        bound_calls.append(result)
        return result

    run_batch = memory_manager._run_batch_process
    load_worker = memory_manager._subprocess_load_worker
    attempts = []

    def worker(args):
        if worker_exit and args["concepts"] == ["los_icu"] and cohort_size in args["patient_ids"]["stay_id"]:
            raise SystemExit(worker_exit)
        load_worker(args)

    def observe_batch(args, attempt_dir, batch_num, mode):
        patients = args["patient_ids"]["stay_id"]
        exitcode = run_batch(args, attempt_dir, batch_num, mode)
        attempts.append((args["concepts"], patients[0], patients[-1], exitcode))
        return exitcode

    monkeypatch.setattr(extraction, "_native_export_stay_time_upper_bounds", observe_bounds)
    monkeypatch.setattr(memory_manager, "_subprocess_load_worker", worker)
    monkeypatch.setattr(memory_manager, "_run_batch_process", observe_batch)
    runner = dataio.make_export_runner(
        data_path=str(source), database="miiv", modules=["demographics", "vitals"],
        concepts={"demographics": ["age"], "vitals": ["hr"]},
        export_format="parquet", out_dir=str(out),
        include_feature_definitions=False,
    )
    failures = []
    written = []
    write_frame = dataio._write_frame

    def observe_write(frame, dest, export_format):
        rows = write_frame(frame, dest, export_format)
        written.append((dest.name, rows))
        return rows

    monkeypatch.setattr(dataio, "_write_frame", observe_write)

    def run_export(job):
        try:
            return runner(job)
        except dataio.ExportCohortError as exc:
            failures.append(exc)
            raise

    manager = JobManager()
    job = manager.reserve("extract")
    manager.run_reserved(job.id, run_export)
    if worker_exit:
        assert job.status == "failed"
        assert job.error == "stay_time_upper_bounds_unavailable"
        assert job.result is None
        failure, = failures
        assert failure.detail == {
            "error": "stay_time_upper_bounds_unavailable", "module": "vitals",
            "stage": "subprocess_batch_load", "reason": "batch 2: worker exited with code 9",
        }
        assert isinstance(failure.__cause__, ConceptExtractionUnavailable)
        assert bound_calls == []
        assert written == [("demographics.parquet", cohort_size)]
        assert not (out / "_manifest.json").exists()
        assert not list(out.glob("*.parquet"))
        repair_end = batch_size + (cohort_size - batch_size + 1) // 2
        assert attempts == [
            (["los_icu"], 1, batch_size, 0), (["los_icu"], batch_size + 1, cohort_size, 9),
            (["los_icu"], batch_size + 1, repair_end, 0), (["los_icu"], repair_end + 1, cohort_size, 9),
        ]
    else:
        assert job.status == "done", job.error
        assert attempts == ([] if platform == "darwin" else [
            (["los_icu"], 1, batch_size, 0), (["los_icu"], batch_size + 1, cohort_size, 0),
            (["hr"], 1, batch_size, 0), (["hr"], batch_size + 1, cohort_size, 0),
        ])
        assert bound_calls == [expected]
        demographics = pd.read_parquet(out / "demographics.parquet").set_index("stay_id")
        assert len(demographics) == cohort_size
        assert demographics.loc[[1, 2, cohort_size], "age"].tolist() == [42, 58, 67]
        vitals = pd.read_parquet(out / "vitals.parquet").sort_values(["stay_id", "charttime"])
        assert vitals[["stay_id", "charttime", "hr"]].values.tolist() == [
            [2, 0.0, 84.0], [cohort_size, 0.0, 86.0],
        ]
        assert job.result["module_resource_plans"]["vitals"]["batch_size"] == batch_size
        manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
        assert manifest["time_window_authority"] == {
            "required": True, "source": "outcome.los_icu", "bounded_stays": cohort_size,
        }
    assert "EASYICU_FORCE_INPROCESS_BATCH" not in os.environ


@pytest.fixture
def source(monkeypatch):
    import easyicu.api as api

    calls = []

    def load(concepts, **kwargs):
        calls.append((list(concepts), kwargs))
        if concepts == ["los_icu"]:
            return pd.DataFrame({"stay_id": [1, 2], "los_icu": [1.0, 2.0]})
        if concepts == ["death"]:
            return pd.DataFrame({"stay_id": [1, 2], "death": [False, True]})
        if concepts == ["age"]:
            return pd.DataFrame({"stay_id": [1, 2], "age": [40.0, 50.0]})
        assert concepts == ["hr"]
        return pd.DataFrame({
            "stay_id": [1] * 6 + [2] * 6,
            "charttime": [-25.0, -24.0, 0.0, 48.0, 48.1, 700.0,
                          -25.0, -24.0, 0.0, 72.0, 72.1, 700.0],
            "hr": [70.0] * 12,
        })

    monkeypatch.setattr(api, "load_concepts", load)
    monkeypatch.setattr(api, "keep_cache", lambda **kwargs: nullcontext())
    monkeypatch.setattr(api, "get_all_patient_ids", lambda *args, **kwargs: ([1, 2], "stay_id"))
    return calls


def publish(root: Path, modules, export_format="parquet"):
    concepts = {"vitals": ["hr"], "outcome": ["death"], "demographics": ["age"],
                "sofa1_score": ["sofa"]}
    dataio.make_export_runner(
        data_path=str(root.parent), database="miiv", modules=modules,
        concepts={module: concepts[module] for module in modules},
        export_format=export_format, out_dir=str(root),
        include_feature_definitions=False,
        cohort={"preset": "all_icu", "observation_window_hours": 720},
    )(ExportJob())
    return json.loads((root / "_manifest.json").read_text())


@pytest.mark.parametrize("export_format", ["parquet", "csv", "excel"])
def test_web_publication_bounds_every_stay_before_metadata(tmp_path, source, export_format):
    root = tmp_path / "export"
    manifest = publish(root, ["vitals"], export_format)
    with intake.open_export_package(root) as package:
        frame = intake.read_exported_concept(package, "hr")
    assert frame["charttime"].tolist() == [-24.0, 0.0, 48.0, -24.0, 0.0, 72.0]
    assert manifest["files"][0]["rows"] == 6
    assert manifest["export_kind"] == "web_module_extraction"
    assert manifest["time_window_authority"] == {
        "required": True, "source": "outcome.los_icu", "bounded_stays": 2,
    }
    assert manifest["cohort_contract"]["observation_window_hours"] == 720
    assert manifest["cohort_execution"]["concept_cohort_window"] is None
    assert [concepts for concepts, _ in source].count(["los_icu"]) == 1
    assert all(kwargs["patient_ids"] == {"stay_id": [1, 2]} for _, kwargs in source)


@pytest.mark.parametrize("modules", [
    ["outcome", "demographics", "vitals"], ["vitals", "outcome", "demographics"],
])
def test_web_module_is_independent_of_coexported_modules(tmp_path, source, modules):
    alone = tmp_path / "alone"
    together = tmp_path / "together"
    publish(alone, ["vitals"])
    manifest = publish(together, modules)
    pd.testing.assert_frame_equal(pd.read_parquet(alone / "vitals.parquet"),
                                  pd.read_parquet(together / "vitals.parquet"))
    assert len(pd.read_parquet(together / "vitals.parquet")) == 6
    assert manifest["time_window_authority"]["bounded_stays"] == 2


def test_web_score_keeps_in_window_values_using_full_precision_los(tmp_path, source, monkeypatch):
    import easyicu.api as api

    load = api.load_concepts
    los = pd.Series([0.3], dtype="float32")
    upper = float(los.iloc[0]) * 24.0 + 24.0
    rounded_upper = float((los * 24.0 + 24.0).iloc[0])
    assert rounded_upper > upper
    scores = pd.DataFrame({
        "stay_id": [1] * 5,
        "charttime": [-24.0, 0.0, upper, rounded_upper, 700.0],
        "sofa": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    def with_scores(concepts, **kwargs):
        if concepts == ["los_icu"]:
            return pd.DataFrame({"stay_id": [1], "los_icu": los})
        if concepts == ["sofa"]:
            return scores.copy()
        return load(concepts, **kwargs)

    monkeypatch.setattr(api, "load_concepts", with_scores)
    for name, modules in [("alone", ["sofa1_score"]),
                          ("together", ["outcome", "vitals", "sofa1_score"])]:
        root = tmp_path / name
        publish(root, modules)
        actual = pd.read_parquet(root / "sofa1_score.parquet")
        pd.testing.assert_frame_equal(actual, scores.iloc[:3])


def test_web_longitudinal_export_cannot_claim_missing_los_bounds(tmp_path, source, monkeypatch):
    import easyicu.api as api

    load = api.load_concepts
    monkeypatch.setattr(api, "load_concepts", lambda concepts, **kwargs:
                        pd.DataFrame({"stay_id": [1, 2], "los_icu": [None, None]})
                        if concepts == ["los_icu"] else load(concepts, **kwargs))
    with pytest.raises(ValueError, match="los_icu.*time bounds"):
        publish(tmp_path / "export", ["vitals"])
    assert not (tmp_path / "export" / "_manifest.json").exists()


def test_stay_level_export_needs_no_longitudinal_bounds(tmp_path, source):
    root = tmp_path / "export"
    manifest = publish(root, ["demographics", "outcome"])
    assert all(concepts != ["los_icu"] for concepts, _ in source)
    manifest.pop("time_window_authority")
    (root / "_manifest.json").write_text(json.dumps(manifest))
    with intake.open_export_package(root) as package:
        assert set(package.concept_index) == {"age", "death"}
    assert dataio.validate_research_pipeline_source(str(root), database="miiv")["ok"]


@pytest.mark.parametrize("authority", [None, {}, {"required": False},
                                      {"required": "true"}, {"required": 1}])
def test_intake_refuses_unbounded_longitudinal_native_v2(tmp_path, authority):
    root = typed_export(tmp_path / "export")
    marker = root / "_manifest.json"
    manifest = json.loads(marker.read_text())
    manifest.pop("time_window_authority", None)
    if authority is not None:
        manifest["time_window_authority"] = authority
    marker.write_text(json.dumps(manifest))
    with pytest.raises(intake.ExportPackageError) as caught:
        intake.open_export_package(root)
    assert caught.value.code == "export_time_window_unbounded"
    assert str(caught.value) == "This export was made by an earlier version of EasyICU that did not limit measurements to each ICU stay. Export the data again, then run the study."


def test_intake_accepts_required_time_window_authority(tmp_path):
    root = typed_export(tmp_path / "export")
    marker = root / "_manifest.json"
    manifest = json.loads(marker.read_text())
    manifest["time_window_authority"] = {
        "required": True, "source": "outcome.los_icu", "bounded_stays": 2,
    }
    marker.write_text(json.dumps(manifest))
    with intake.open_export_package(root) as package:
        assert "lact" in package.concept_index


def test_unbounded_export_stop_is_a_data_action():
    assert stop_next_actions.STOP_POLICIES["export_time_window_unbounded"] == "data"
    actions = stop_next_actions.next_actions_for_stop("export_time_window_unbounded")
    assert [action.kind for action in actions] == ["reexport_data"]
    assert actions[0].params.model_dump() == {
        "route": "data_extraction", "requires_explicit_choice": True,
    }


@pytest.mark.parametrize("field", ["cause_code", "reason_code", "intake_error_code"])
def test_wrapped_unbounded_export_stop_preserves_the_data_action(field):
    actions = stop_next_actions.next_actions_for_stop(
        "research_pipeline_manifest_invalid", {field: "export_time_window_unbounded"},
    )
    assert [action.kind for action in actions] == ["reexport_data"]


@pytest.mark.parametrize("legacy", [False, True])
def test_run_binding_refuses_old_export_before_planning(tmp_path, legacy):
    root = typed_export(tmp_path / "export")
    marker = root / "_manifest.json"
    manifest = json.loads(marker.read_text())
    manifest.pop("time_window_authority", None)
    if legacy:
        manifest.pop("schema_version")
        sidecar = manifest.pop("column_metadata")
        (root / sidecar["file"]).unlink()
        for entry in manifest["files"]:
            entry.pop("column_metadata_columns")
    marker.write_text(json.dumps(manifest))
    before = (root / "labs.parquet").read_bytes()
    with pytest.raises(dataio.ExportCohortError) as caught:
        dataio.validate_research_pipeline_source(str(root), database="miiv")
    assert caught.value.error == "export_time_window_unbounded"
    assert caught.value.detail["intake_error_message"] == (
        "This export was made by an earlier version of EasyICU that did not limit measurements to each ICU stay. Export the data again, then run the study."
    )
    assert (root / "labs.parquet").read_bytes() == before


def test_web_export_is_accepted_by_run_binding(tmp_path, source):
    root = tmp_path / "export"
    publish(root, ["vitals"])
    assert dataio.validate_research_pipeline_source(str(root), database="miiv")["ok"]


@pytest.mark.parametrize("already_bound", [False, True])
def test_run_binding_preserves_typed_refusal_for_unreadable_layout(tmp_path, already_bound):
    root = typed_export(tmp_path / "export")
    binding = (dataio.validate_research_pipeline_source(str(root), database="miiv")["binding"]
               if already_bound else None)
    (root / "labs.parquet").write_bytes(b"invalid parquet")
    with pytest.raises(dataio.ExportCohortError) as caught:
        dataio.validate_research_pipeline_source(
            str(root), database="miiv", expected_binding=binding,
        )
    assert caught.value.error == (
        "research_pipeline_package_binding_changed" if already_bound
        else "research_pipeline_manifest_invalid"
    )
    if not already_bound:
        # The rewritten member no longer matches its recorded digest; that typed
        # refusal precedes any parse of the unreadable bytes.
        assert caught.value.detail["intake_error_code"] == "export_member_integrity_failed"


def test_submission_refuses_an_old_package_before_provider_or_planner(tmp_path, monkeypatch):
    from easyicu.webserver import research_run_submission as submission

    root = typed_export(tmp_path / "export")
    marker = root / "_manifest.json"
    manifest = json.loads(marker.read_text())
    manifest.pop("time_window_authority")
    marker.write_text(json.dumps(manifest))
    study = {"id": "synthetic_study", "revision": 1, "question": "Describe the cohort.",
             "data_source": {"path": str(root), "database": "miiv"}}
    monkeypatch.setattr(submission.context_store, "get_context", lambda _: study)
    monkeypatch.setattr(submission, "build_research_workflow_snapshot",
                        lambda **kwargs: SimpleNamespace(planning_prerequisites_missing=[]))

    def forbidden(*args, **kwargs):
        pytest.fail("Unbounded data reached provider setup or job submission")

    monkeypatch.setattr(submission, "provider_environment_for_agent_run", forbidden)
    monkeypatch.setattr(submission, "_submit_job", forbidden)
    request = submission.ResearchRunSubmissionRequest(
        study_context_id=study["id"], provider="mock", credential_source="scientific_provider",
        external_llm_opt_in=False, intent="candidate_plan",
    )
    with pytest.raises(submission.ResearchRunSubmissionError) as caught:
        submission.submit_research_run(request)
    assert caught.value.code == "export_time_window_unbounded"
    assert caught.value.summary == "This export was made by an earlier version of EasyICU that did not limit measurements to each ICU stay. Export the data again, then run the study."



def test_layout_inventory_can_read_an_old_export_before_run_binding(tmp_path):
    root = typed_export(tmp_path / "export")
    marker = root / "_manifest.json"
    manifest = json.loads(marker.read_text())
    manifest.pop("time_window_authority")
    marker.write_text(json.dumps(manifest))
    assert intake.inspect_export_layout(root)["readiness_scope"] == "layout_only"
    with pytest.raises(dataio.ExportCohortError) as caught:
        dataio.validate_research_pipeline_source(str(root), database="miiv")
    assert caught.value.error == "export_time_window_unbounded"

def test_web_dictionary_output_uses_the_same_time_guard(tmp_path, source, monkeypatch):
    import easyicu.api as api

    load = api.load_concepts

    def split(concepts, **kwargs):
        frame = load(concepts, **kwargs)
        return {"measurements": frame} if concepts == ["hr"] else frame

    monkeypatch.setattr(api, "load_concepts", split)
    root = tmp_path / "export"
    manifest = publish(root, ["vitals"])
    assert manifest["files"][0]["rows"] == 6
    assert len(pd.read_parquet(root / "vitals__measurements.parquet")) == 6


@pytest.mark.parametrize("multi_index", [False, True], ids=["flat-index", "multi-index"])
@pytest.mark.parametrize("has_charttime", [True, False], ids=["partial", "no-time"])
def test_time_guard_normalises_zero_row_longitudinal_output(has_charttime, multi_index):
    from easyicu.api import extraction

    frame = pd.DataFrame({"hr": pd.Series(dtype="float64")})
    if has_charttime:
        frame["charttime"] = pd.Series(dtype="object")
    if multi_index:
        frame.index = pd.MultiIndex.from_arrays([[], []], names=["stay_id", "charttime"])

    prepared, audit = extraction._enforce_native_export_time_axis(
        frame, module="vitals", stay_time_upper_bounds={1: 48.0},
    )

    assert len(prepared) == 0
    assert list(prepared.columns) == ["stay_id", "charttime", "hr"]
    assert prepared.dtypes.astype(str).to_dict() == {
        "stay_id": "int64", "charttime": "float64", "hr": "float64",
    }
    assert audit["rows_with_los_bound"] == audit["excluded_rows"] == 0


@pytest.mark.parametrize("export_format", ["parquet", "csv"])
def test_web_writer_preserves_headers_across_longitudinal_batches(
    tmp_path, source, monkeypatch, export_format,
):
    from easyicu import api

    load = api.load_concepts

    def load_empty_batch(concepts, **kwargs):
        if concepts == ["hr"]:
            return {"measurements": pd.DataFrame({"hr": pd.Series(dtype="float64")})}
        return load(concepts, **kwargs)

    def load_populated_batch(concepts, **kwargs):
        frame = load(concepts, **kwargs)
        return {"measurements": frame} if concepts == ["hr"] else frame

    def export_batch(name, loader):
        monkeypatch.setattr(api, "load_concepts", loader)
        root = tmp_path / name
        manifest = publish(root, ["vitals"], export_format)
        file, = manifest["files"]
        read = pd.read_parquet if export_format == "parquet" else pd.read_csv
        return read(root / file["file"])

    empty = export_batch("no_observations", load_empty_batch)
    populated = export_batch("ward_transfer_followup", load_populated_batch)

    assert empty.empty
    assert len(populated) == 6
    assert list(empty.columns) == list(populated.columns) == ["stay_id", "charttime", "hr"]


def test_time_guard_accepts_registered_episode_support_with_native_stay_key(tmp_path):
    from easyicu.api import extraction

    pd.DataFrame({"patient_SN": [41]}).to_parquet(tmp_path / "stays.parquet", index=False)
    frame = extraction._load_module_concepts(
        None,
        module_name="imv_first_episode",
        load_kwargs={
            "concepts": ["episode_status"], "database": "zhejiang_eicu",
            "data_path": str(tmp_path), "patient_ids": {"patient_SN": [41]},
        },
    )
    assert frame["patient_SN"].tolist() == [41]
    assert frame["episode_status"].tolist() == ["source_unavailable"]

    prepared, audit = extraction._enforce_native_export_time_axis(
        frame, module="imv_first_episode", database="zhejiang_eicu",
        stay_time_upper_bounds={41: 48.0},
    )

    pd.testing.assert_frame_equal(prepared, frame)
    assert audit["policy"] == "not_applicable_stay_level_module"


@pytest.mark.parametrize("columns", [{"charttime": [0.0]}, {}], ids=["timed", "no-columns"])
def test_time_guard_refuses_nonempty_output_without_a_stay_key(columns):
    from easyicu.api import extraction

    with pytest.raises(ValueError, match="no ICU-stay identity column"):
        extraction._enforce_native_export_time_axis(
            pd.DataFrame(columns, index=[0]),
            module="vitals", stay_time_upper_bounds={1: 48.0},
        )


@pytest.mark.parametrize("identity", ["stay_id", "patientunitstayid", "admissionid"])
def test_time_guard_is_independent_of_patient_batch_and_source_identity(identity):
    from easyicu.api import extraction

    frame = pd.DataFrame({identity: [1, 1, 2, 2], "charttime": [48.0, 49.0, 72.0, 73.0],
                          "hr": [70.0] * 4})
    bounds = extraction._native_export_stay_time_upper_bounds(
        pd.DataFrame({identity: [1, 2], "los_icu": [1.0, 2.0]})
    )
    whole, _ = extraction._enforce_native_export_time_axis(
        frame, module="vitals", stay_time_upper_bounds=bounds,
    )
    batches = [extraction._enforce_native_export_time_axis(
        batch, module="vitals", stay_time_upper_bounds=bounds,
    )[0] for _, batch in frame.groupby(identity)]
    pd.testing.assert_frame_equal(whole, pd.concat(batches))
    assert whole["charttime"].tolist() == [48.0, 72.0]

    with pytest.raises(ValueError, match="without an event time"):
        extraction._enforce_native_export_time_axis(
            pd.DataFrame({identity: [1], "charttime": [None], "rrt_criteria": [True]}),
            module="renal", stay_time_upper_bounds=bounds,
        )
