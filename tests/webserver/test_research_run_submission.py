from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from easyicu.webserver import research_run_submission
from easyicu.webserver.routes import agent as agent_route


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
