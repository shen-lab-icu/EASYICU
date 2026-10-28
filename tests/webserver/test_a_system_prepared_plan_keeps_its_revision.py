"""Synthetic launch preparation must not invalidate its own job handoff."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.webserver import agent_pipeline_runs, research_job_recovery, study_contexts
from easyicu.webserver import research_pipeline_run_preparation as preparation
from easyicu.webserver import research_run_submission as submission


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    study = study_contexts.upsert_context({
        "id": "synthetic-launch", "question": "Estimate the association for a synthetic assay.",
        "data_source": {"path": str(tmp_path / "source"), "database": "eicu"},
        "analysis_design": {"analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "model_based"},
        "time_window": {"anchor": "icu_admission", "hours": 24},
        "execution_concepts": {"primary_exposure": "synthetic_assay", "outcome": "death", "covariates": []},
        "modules": ["labevents"], "confirmations": {"feature_time_window": True},
    })
    monkeypatch.setattr(preparation, "_require_export_holds_study_cohort", lambda *_: None)
    monkeypatch.setattr(preparation, "_validate_primary_concept_selection", lambda *_: None)
    monkeypatch.setattr(preparation, "_patient_grouping_for_analysis_design", lambda *_: None)
    monkeypatch.setattr(preparation, "_metadata_only_planning_coordinates", lambda **_: {
        "primary_exposure": "synthetic_assay", "target_outcome": "death",
        "endpoint": EndpointSpec(name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]),
    })
    monkeypatch.setattr(submission.dataio, "describe_source_tables", lambda _: {"ok": True})
    monkeypatch.setattr(submission.dataio, "prepared_export_manifest_path", lambda _: None)
    monkeypatch.setattr(submission, "build_research_workflow_snapshot", lambda **_: SimpleNamespace(planning_prerequisites_missing=[]))
    monkeypatch.setattr(submission, "provider_environment_for_agent_run", lambda **_: {})
    monkeypatch.setattr(submission.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(submission.capabilities, "validate_compute_target", lambda _: {"ok": True})
    monkeypatch.setattr(submission.agent_runs, "resolve_agent_provider_config", lambda **_: {})
    monkeypatch.setattr(study_contexts, "build_agent_context_binding", lambda *_, **__: {})
    monkeypatch.setattr(submission, "research_pipeline_workspace", lambda: SimpleNamespace(project_root=lambda _: tmp_path))
    monkeypatch.setattr(submission.capabilities, "record_tool_event", lambda *_, **__: None)
    # Exercise real scientific preparation and runner construction, replacing
    # only provider/execution setup and the eventual research operation.
    captured = {}

    def prepare(request):
        scientific = preparation._prepare_scientific_launch(request)
        captured["scientific"] = scientific
        return SimpleNamespace(scientific=scientific, execution=SimpleNamespace(execution_resume_source_run_id=""))

    def wrap(_runner, request):
        captured["recovery_request"] = request
        return lambda _job: {"gate": {"status": "plan_review_required"}}

    monkeypatch.setattr(agent_pipeline_runs, "prepare_research_pipeline_run", prepare)
    monkeypatch.setattr(research_job_recovery, "wrap_pipeline_runner", wrap)
    job = SimpleNamespace(id="synthetic-job", kind="agent-run", status="queued", cancel_requested=False)

    def enqueue(_kind, runner):
        captured["runner"] = runner
        return job

    monkeypatch.setattr(submission, "_submit_job", enqueue)

    def submit():
        return submission.submit_research_run(submission.ResearchRunSubmissionRequest(
            study_context_id=study["id"], provider="mock", credential_source="pi_verified",
            external_llm_opt_in=False, intent="candidate_plan", planner_start_mode="fresh",
        ))

    return SimpleNamespace(study=study, submit=submit, captured=captured, job=job, wrap=wrap)


def test_system_landmark_revision_starts_on_the_first_click(launch):
    receipt = launch.submit()
    prepared = launch.captured["scientific"].study
    assert prepared["revision"] == launch.study["revision"] + 1
    assert prepared["confirmations"]["plan_timing_landmark_24h"] is True
    assert prepared["sensitivity_specs"][0]["spec_id"] == "agent_post_baseline_landmark_24h"
    assert receipt.study_context_revision == prepared["revision"] + 1
    current = study_contexts.get_context(launch.study["id"])
    assert current["active_job_id"] == receipt.job_id
    assert study_contexts.scientific_configuration_sha256(current) == study_contexts.scientific_configuration_sha256(prepared)
    assert launch.captured["runner"](launch.job)["gate"]["status"] == "plan_review_required"


def test_recovery_binds_the_system_prepared_scientific_state(launch):
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=launch.study["data_source"]["path"], study_context=launch.study, project_root=None,
        provider={}, provider_environment={}, credential_source="pi_verified", literature_search_authorized=False,
        plan_revision_source_run_id="", execution_resume_source_run_id="", development_resume_source_job_id="",
        budget_mode="planner_canary", runner_image=None,
    )
    agent_pipeline_runs.make_research_pipeline_run_runner(**request.__dict__)
    prepared = launch.captured["scientific"].study
    assert launch.captured["recovery_request"].study_context == prepared
    assert launch.study["sensitivity_specs"] == []


@pytest.mark.parametrize("edit", [{"question": "A different synthetic research question."}, {"title": "A concurrent edit"}])
def test_a_concurrent_edit_after_preparation_still_refuses_the_handoff(launch, monkeypatch, edit):
    def wrap(runner, request):
        result = launch.wrap(runner, request)
        current = study_contexts.get_context(launch.study["id"])
        study_contexts.upsert_context({"id": current["id"], **edit}, expected_revision=current["revision"])
        return result

    monkeypatch.setattr(research_job_recovery, "wrap_pipeline_runner", wrap)
    with pytest.raises(submission.ResearchRunSubmissionError) as refused:
        launch.submit()
    assert refused.value.code == "study_context_revision_conflict"
    with pytest.raises(RuntimeError, match="agent_run_start_blocked"):
        launch.captured["runner"](launch.job)
    assert study_contexts.get_context(launch.study["id"])["active_job_id"] is None


def test_a_concurrent_edit_during_scientific_preparation_is_not_adopted(launch, monkeypatch):
    original = agent_pipeline_runs.prepare_research_pipeline_run

    def prepare(request):
        study_contexts.upsert_context({"id": launch.study["id"], "question": "Changed before the timing write."},
                                     expected_revision=launch.study["revision"])
        return original(replace(request))

    monkeypatch.setattr(agent_pipeline_runs, "prepare_research_pipeline_run", prepare)
    with pytest.raises(submission.ResearchRunSubmissionError) as refused:
        launch.submit()
    assert refused.value.code == "study_context_revision_conflict"
    assert "runner" not in launch.captured
