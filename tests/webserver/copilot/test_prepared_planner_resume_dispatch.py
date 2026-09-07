"""Checkpoint continuation preserves prepared-input scope, never approves analysis."""

from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot import tools as owner


@pytest.fixture
def dispatch(monkeypatch):
    study = {"id": "study-recovery", "question": "A natural research question"}
    latest = {"run_id": "run_prepared", "study_id": study["id"], "run_status": "failed"}
    workflow = {
        "next_action_code": "planner_checkpoint_resume_available",
        "latest_attempt_failure": {
            "run_id": "run_prepared", "candidate_run_id": "run_candidate",
            "reason": "provider_unavailable", "checkpoint_resume_available": True,
        },
    }
    context = SimpleNamespace(
        session=SimpleNamespace(binding=SimpleNamespace()), user_message="Continue planning",
    )
    calls = []
    monkeypatch.setattr(owner, "_bound_context", lambda _: study)
    monkeypatch.setattr(owner, "_select_run", lambda _: latest)
    monkeypatch.setattr(owner, "_workflow_snapshot", lambda *a, **k: workflow)
    monkeypatch.setattr(owner, "_run", lambda *a, **k: calls.append(k) or {"code": "submitted"})
    return context, workflow, calls


def test_prepared_checkpoint_stays_package_bound_instead_of_reverting_to_metadata(dispatch):
    context, _, calls = dispatch
    result = owner._request_replan(context, {"strategy": "resume_checkpoint"})
    assert result["code"] == "submitted"
    assert calls[0]["planner_start_mode"] == "resume_checkpoint"
    assert calls[0]["run_intent"] == "reviewed_analysis"
    assert calls[0].get("plan_change_request") is None


@pytest.mark.parametrize("change", [None, {"run_id": "run_other"}, {"checkpoint_resume_available": False}])
def test_unbound_or_absent_prepared_failure_does_not_promote_metadata_scope(dispatch, change):
    context, workflow, calls = dispatch
    if change is None:
        workflow["latest_attempt_failure"] = None
    else:
        workflow["latest_attempt_failure"].update(change)
    owner._request_replan(context, {"strategy": "resume_checkpoint"})
    assert calls[0]["run_intent"] == "candidate_plan"


def test_explicit_fresh_planning_never_silently_reuses_checkpoint(dispatch):
    context, _, calls = dispatch
    owner._request_replan(context, {"strategy": "fresh"})
    assert calls[0]["planner_start_mode"] == "fresh"
    assert calls[0]["run_intent"] == "candidate_plan"


def test_approval_tool_at_prepared_checkpoint_routes_to_planning_not_analysis(dispatch):
    context, _, calls = dispatch
    result = owner._resume(context, {"decision": "approved"})
    assert result["code"] == "submitted"
    assert calls[0]["planner_start_mode"] == "resume_checkpoint"
    assert calls[0]["run_intent"] == "reviewed_analysis"
