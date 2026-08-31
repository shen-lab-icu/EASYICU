from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from easyicu.webserver import research_run_submission
from easyicu.webserver.routes import agent as agent_route


def test_pipeline_route_delegates_to_submission_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def submit(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured["body"] = body
        captured.update(kwargs)
        return {"job_id": "job-owned", "engine": "research_agent_pipeline"}

    monkeypatch.setattr(research_run_submission, "submit_research_run", submit)
    body = {"engine": "research_agent_pipeline", "path": "/not-read-by-route"}

    result = agent_route.submit_agent_run(
        body,
        account_environment={"ACCOUNT": "private"},
        metadata_only_planning_authorized=True,
    )

    assert result == {"job_id": "job-owned", "engine": "research_agent_pipeline"}
    assert captured == {
        "body": body,
        "account_environment": {"ACCOUNT": "private"},
        "metadata_only_planning_authorized": True,
    }


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
        agent_route.submit_agent_run({"engine": "research_agent_pipeline"})

    assert raised.value.status_code == 429
    assert raised.value.detail == {"error": "job_capacity_exceeded"}


def test_submission_owner_rejects_non_pipeline_engine() -> None:
    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as raised:
        research_run_submission.submit_research_run({"engine": "native_summary"})

    assert raised.value.status_code == 400
    assert raised.value.detail == {"error": "research_run_submission_engine_required"}
