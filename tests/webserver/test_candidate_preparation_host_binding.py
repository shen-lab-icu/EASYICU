"""The browser's data-preparation decision must name the submitted candidate."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from easyicu.webserver.routes import agent


def _body(*, submitted_source="run_candidate", offered_source="run_candidate",
          action_state="plan_execution_upgrade_required"):
    return {
        "engine": "research_agent_pipeline",
        "llm_provider": "mock",
        "study_context_id": "study",
        "plan_revision_source_run_id": submitted_source,
        "host_action": {
            "session_id": "session", "project_id": "project",
            "action_code": "prepare_analysis_data",
            "decision": {
                "family": "plan_transition", "next_action_code": action_state,
                "scientific_configuration_sha256": "a" * 64,
                "source_run_id": offered_source,
            },
        },
    }


def test_candidate_preparation_route_passes_the_matching_decision_to_host_owner(monkeypatch):
    submitted = []
    monkeypatch.setattr(agent, "submit_agent_run", lambda *args, **kwargs: submitted.append(True) or {"job_id": "job"})
    def host_owner(action, body, submit, **kwargs):
        assert action.decision.source_run_id == body["plan_revision_source_run_id"]
        return submit()
    monkeypatch.setattr(agent, "_submit_host_decision", host_owner)
    assert agent.jobs_agent_run(_body(), None) == {"job_id": "job"}
    assert submitted == [True]


@pytest.mark.parametrize("body", [
    _body(submitted_source="another_candidate"),
    _body(submitted_source=""),
    _body(offered_source=""),
    _body(action_state="provider_ready_to_generate_plan"),
])
def test_mismatched_candidate_is_rejected_before_job_or_data_access(monkeypatch, body):
    def unexpected(*args, **kwargs):
        pytest.fail("refused request crossed the job-creation boundary")
    monkeypatch.setattr(agent, "_submit_host_decision", unexpected)
    monkeypatch.setattr(agent, "submit_agent_run", unexpected)
    with pytest.raises(HTTPException) as error:
        agent.jobs_agent_run(body, None)
    assert error.value.status_code == 400
    assert error.value.detail["error"] == "host_action_request_mismatch"
