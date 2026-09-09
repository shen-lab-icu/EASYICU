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



def test_population_constraint_uses_host_bound_current_plan_not_another_reference():
    from easyicu.webserver.plan_change_request import ReferencedPlan

    def reference(run_id, scope, digest):
        return ReferencedPlan(run_id=run_id, artifact_sha256=digest, plan={"steps": [{
            "step_id": "risk", "population_scope": scope,
            "expected_outputs": ["table:absolute_risk_context"],
        }]})
    request = PlanChangeRequest(
        source_run_id="current", user_message="Revise figures and retain the scientific population.",
        reference_plans=(reference("older", "analysis_cohort", "a" * 64), reference("current", "primary_model", "b" * 64)),
    )
    requirement = request.population_requirements()
    assert requirement.source_plan_sha256 == "b" * 64
    assert requirement.source_digest_kind == "artifact_sha256"
    assert requirement.populations[0].population_scope == "primary_model"
    assert PlanChangeRequest(source_run_id="legacy", user_message="Revise the plan.").population_requirements() is None
    with pytest.raises(ValueError, match="current source plan is absent"):
        request.model_copy(update={"source_run_id": "missing"}).population_requirements()
