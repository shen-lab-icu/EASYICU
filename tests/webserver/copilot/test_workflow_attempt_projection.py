"""A failed attempt and the plan it must not erase are separate UI facts."""

from copy import deepcopy

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from tests.webserver.copilot.research_workflow_fixtures import complete_study


def case():
    study = complete_study()
    digest = study_contexts.scientific_configuration_sha256(study)
    candidate = {
        "run_id": "run_candidate", "study_id": study["id"], "run_type": "full",
        "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
        "gate_reason": "human_plan_review_required", "run_status": "human_review_pending",
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "scientific_configuration_sha256": digest,
        "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
    }
    failed = {
        "run_id": "run_preparation", "study_id": study["id"], "run_type": "full",
        "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
        "gate_reason": "research_pipeline_planner_provider_unavailable", "run_status": "failed",
        "scientific_configuration_sha256": digest,
        "artifact_names": ["source_run_manifest.json"],
        "development_planner_checkpoint_available": True,
        "project_dir": "/private/not-for-browser",
    }
    review = {
        "run_id": candidate["run_id"], "resumable_here": True,
        "scientific_configuration_sha256": digest, "budget_mode": "planner_canary",
        "research_input_state": "metadata_only", "plan_approval_allowed": True,
    }
    return study, candidate, failed, review


def snapshot(study, candidate, failed, review, active_job=None):
    return build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=active_job,
        latest_run=candidate, latest_attempt=failed, plan_review_authority=review,
    )


def test_failed_preparation_is_visible_without_erasing_candidate_or_repeating_planner():
    args = case()
    original = deepcopy(args)
    result = snapshot(*args)
    assert result.next_action_code == "planner_checkpoint_resume_available"
    assert result.plan_execution_ready is False
    assert result.latest_attempt_failure.model_dump() == {
        "run_id": "run_preparation", "candidate_run_id": "run_candidate",
        "reason": "provider_unavailable", "checkpoint_resume_available": True,
    }
    assert next(row for row in result.stages if row.id == "analysis").status == "blocked"
    assert "/private/" not in result.latest_attempt_failure.model_dump_json()
    assert args == original


@pytest.mark.parametrize("change", [
    {"development_planner_checkpoint_available": False},
    {"gate_reason": "research_pipeline_progressive_compile_failed"},
])
def test_nonresumable_attempt_stays_visible_but_never_invents_resume(change):
    study, candidate, failed, review = case()
    failed.update(change)
    result = snapshot(study, candidate, failed, review)
    assert result.next_action_code == "plan_execution_upgrade_required"
    assert result.latest_attempt_failure is not None
    assert result.latest_attempt_failure.checkpoint_resume_available is False


@pytest.mark.parametrize("change", [
    {"scientific_configuration_sha256": "f" * 64},
    {"study_id": "another-study"}, {"run_status": "complete"},
    {"artifact_names": ["agent_plan.json"]}, {"run_id": "run_candidate"},
    {"run_id": "/private/not-an-id"}, {"engine": "other"},
])
def test_unbound_or_plan_owning_attempt_cannot_override_candidate(change):
    study, candidate, failed, review = case()
    failed.update(change)
    result = snapshot(study, candidate, failed, review)
    assert result.latest_attempt_failure is None
    assert result.next_action_code == "plan_execution_upgrade_required"


def test_reviewed_executable_plan_is_not_downgraded_to_checkpoint_planning():
    study, candidate, failed, review = case()
    review.update(budget_mode="full_reviewed", research_input_state="prepared")
    result = snapshot(study, candidate, failed, review)
    assert result.next_action_code == "operator_plan_approval_required"


@pytest.mark.parametrize("events", [[], [
    {"type": "progress", "step": "planning", "status": "running"},
], [
    {"type": "progress", "step": "planning", "label": "analysis is running"},
]])
def test_agent_job_kind_and_labels_are_not_evidence_that_analysis_started(events):
    study, candidate, failed, review = case()
    result = snapshot(study, candidate, failed, review, {
        "kind": "agent-run", "status": "running", "events": events,
    })
    by_id = {row.id: row for row in result.stages}
    assert result.next_action_code == "research_planning_running"
    assert by_id["plan"].status == "running"
    assert by_id["analysis"].status == "blocked"
    assert result.plan_execution_ready is False


def test_exact_execution_progress_advances_only_analysis_not_plan():
    study, candidate, failed, review = case()
    result = snapshot(study, candidate, failed, review, {
        "kind": "agent-run", "status": "running", "events": [
            {"type": "progress", "step": "planning", "status": "complete"},
            {"type": "progress", "step": "step", "status": "running"},
            {"type": "progress", "step": "audit", "status": "running"},
        ],
    })
    by_id = {row.id: row for row in result.stages}
    assert result.current_stage == "analysis"
    assert by_id["analysis"].status == "running"
    assert by_id["plan"].status != "running"
    assert result.plan_execution_ready is False


@pytest.mark.parametrize("event", [None, {"step": []},
    {"type": "progress", "step": "runner", "status": []},
    {"type": "progress", "step": "step", "status": "blocked"},
    {"type": "progress", "step": "step", "status": "skipped"},
    {"type": "other", "step": "runner", "status": "running"},
])
def test_malformed_and_blocked_events_are_not_execution_progress(event):
    from easyicu.webserver.pi_copilot.workflow_attempts import research_job_has_execution_progress

    assert research_job_has_execution_progress({"events": [event]}) is False


def test_project_adapter_passes_latest_attempt_separately_from_plan(monkeypatch):
    from easyicu.webserver.pi_copilot import workflow as owner

    study, candidate, failed, review = case()
    monkeypatch.setattr(owner.sources, "load_registry", lambda: {})
    monkeypatch.setattr(owner, "list_bound_run_history", lambda **k: [failed, candidate])
    monkeypatch.setattr(owner.agent_pipeline_runs, "pending_review", lambda _: review)
    monkeypatch.setattr(owner.agent_runs, "read_run_review", lambda _: {})
    result = owner.build_project_workflow_projection(
        study_context_id=study["id"], study_override=study,
    )
    assert result.workflow.latest_attempt_failure.run_id == failed["run_id"]
    assert result.workflow.latest_attempt_failure.candidate_run_id == candidate["run_id"]
    assert result.workflow.next_action_code == "planner_checkpoint_resume_available"
