from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from easyicu.webserver import research_run_submission
from easyicu.webserver.routes import agent as agent_route


def test_host_candidate_plan_control_recovers_server_owned_planner_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="provider_ready_to_generate_plan"
            )
        ),
    )

    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "auto",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="planner_checkpoint_resume_available"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "resume_checkpoint",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="plan_scientific_changes_required"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "auto",
            "plan_revision_source_run_id": "run-review-required",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="failed_pipeline_requires_fresh_plan"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "auto",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(next_action_code="plan_configuration_superseded")
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )


def test_failed_plan_regeneration_stays_candidate_only_after_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="failed_pipeline_requires_fresh_plan"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )


def test_pipeline_route_delegates_to_submission_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def submit(request: Any, **kwargs: Any) -> Any:
        captured["request"] = request
        captured.update(kwargs)
        return research_run_submission.ResearchRunSubmissionReceipt(
            job_id="job-owned",
            kind="agent-run",
            status="running",
            study_context_id=request.study_context_id,
            study_context_revision=2,
            budget_mode="planner_canary",
            planner_start_mode=request.planner_start_mode,
        )

    monkeypatch.setattr(research_run_submission, "submit_research_run", submit)
    body = {
        "engine": "research_agent_pipeline",
        "study_context_id": "study-1",
        "llm_provider": "openai",
        "credential_source": "pi_verified",
        "external_llm_opt_in": True,
        "path": "/not-read-by-route",
    }

    result = agent_route.submit_agent_run(
        body,
        account_environment={"ACCOUNT": "private"},
        metadata_only_planning_authorized=True,
    )

    assert result["job_id"] == "job-owned"
    request = captured.pop("request")
    assert request.study_context_id == "study-1"
    assert request.intent == "candidate_plan"
    assert not hasattr(request, "path")
    assert captured == {"account_environment": {"ACCOUNT": "private"}}


def test_pipeline_route_maps_typed_submission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise research_run_submission.ResearchRunSubmissionError(
            {"error": "job_capacity_exceeded"},
            status_code=429,
        )

    monkeypatch.setattr(research_run_submission, "submit_research_run", reject)

    with pytest.raises(HTTPException) as raised:
        agent_route.submit_agent_run(
            {
                "engine": "research_agent_pipeline",
                "study_context_id": "study-1",
            }
        )

    assert raised.value.status_code == 429
    assert raised.value.detail == {"error": "job_capacity_exceeded"}


def test_submission_request_forbids_adapter_owned_coordinates() -> None:
    with pytest.raises(ValidationError):
        research_run_submission.ResearchRunSubmissionRequest.model_validate(
            {
                "study_context_id": "study-1",
                "provider": "openai",
                "credential_source": "pi_verified",
                "external_llm_opt_in": True,
                "path": "/adapter-must-not-pass-this",
                "budget_mode": "full_reviewed",
                "development_resume_source_job_id": "job-old",
            }
        )


def test_owner_checks_readiness_before_consuming_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {
        "id": "study-not-ready",
        "revision": 3,
        "question": "",
        "data_source": {"path": "/private/export", "database": "miiv"},
        "cohort": {"preset": "adult_icu"},
    }
    monkeypatch.setattr(
        research_run_submission.context_store,
        "get_context",
        lambda _study_id: study,
    )
    monkeypatch.setattr(
        research_run_submission.dataio,
        "describe_export_source",
        lambda _path: {"ok": True},
    )
    authorizations: list[str] = []
    request = research_run_submission.ResearchRunSubmissionRequest(
        study_context_id=study["id"],
        provider="openai",
        credential_source="pi_verified",
        external_llm_opt_in=True,
    )

    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as raised:
        research_run_submission.submit_research_run(
            request,
            authorize=lambda: authorizations.append("consumed"),
        )

    assert raised.value.code == "study_setup_incomplete"
    assert authorizations == []


def test_submission_receipt_is_frozen_and_path_free() -> None:
    receipt = research_run_submission.ResearchRunSubmissionReceipt(
        job_id="job-1",
        kind="agent-run",
        status="queued",
        study_context_id="study-1",
        study_context_revision=4,
        budget_mode="planner_canary",
        planner_start_mode="auto",
    )

    assert "path" not in receipt.model_dump(mode="json")
    with pytest.raises(ValidationError):
        receipt.job_id = "job-2"
