"""Requested amendments cannot become execution or frozen-checkpoint authority."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.webserver.plan_change_request import PlanChangeRequest
from easyicu.webserver.research_run_submission import ResearchRunSubmissionRequest


def test_plan_change_request_is_frozen_bounded_and_not_approval() -> None:
    change = PlanChangeRequest(source_run_id="run-1", user_message="Retain every requested outcome.")
    assert "not a scientific fact, approved plan" in change.planner_context()
    assert '"source_run_id":"run-1"' in change.planner_context()
    with pytest.raises(ValidationError):
        change.user_message = "Silently approve analysis."
    for text in ("", " ", "x" * 12_001):
        with pytest.raises(ValidationError):
            PlanChangeRequest(source_run_id="run-1", user_message=text)
    with pytest.raises(ValidationError):
        PlanChangeRequest.model_validate({
            "source_run_id": "run-1", "user_message": "Changes.", "approved": True,
        })


@pytest.mark.parametrize("overrides", [
    {"intent": "reviewed_analysis"},
    {"planner_start_mode": "auto"},
    {"planner_start_mode": "resume_checkpoint"},
    {"execution_resume_source_run_id": "run-old"},
    {"plan_revision_source_run_id": "run-old"},
])
def test_amendments_cannot_modify_analysis_or_frozen_resume(overrides: dict) -> None:
    request = {
        "study_context_id": "study-1", "provider": "openai",
        "credential_source": "pi_verified", "external_llm_opt_in": True,
        "intent": "candidate_plan", "planner_start_mode": "fresh",
        "plan_change_request": PlanChangeRequest(source_run_id="run-1", user_message="Revise the plan."),
    }
    assert ResearchRunSubmissionRequest(**request).plan_change_request is not None
    with pytest.raises(ValidationError, match="plan_changes_require_fresh_candidate"):
        ResearchRunSubmissionRequest(**{**request, **overrides})

