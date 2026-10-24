"""Native web publication and research intake share the ICU-episode contract."""

from __future__ import annotations

import json
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
        assert caught.value.detail["intake_error_code"] == "export_package_unreadable"


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
