"""Focused owner and fail-closed tests for the Copilot research workflow."""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.reporting.result_card import (
    build_result_interpretation_card,
)
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.workflow import (
    build_research_workflow_snapshot,
)


def _complete_study() -> dict[str, Any]:
    return {
        "id": "study-workflow",
        "revision": 4,
        "question": "Does an aggregate ICU feature predict mortality?",
        "data_source": {
            "path": "/private/prepared/source",
            "database": "mimiciv",
        },
        "cohort": {"preset": "adult_icu", "max_patients": 2000},
        "modules": ["vitals", "outcome"],
        "outcome": "In-hospital mortality",
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "export_format": "parquet",
        "analysis_goal": "Descriptive prognostic association",
    }


def test_workflow_projection_advances_only_from_owner_receipts() -> None:
    empty = build_research_workflow_snapshot(
        study={"id": "study-empty"},
        active_export_present=False,
        active_job=None,
        latest_run=None,
    )
    assert empty.current_stage == "question"
    assert empty.missing_setup_fields == [
        "question",
        "data_source",
        "cohort",
        "modules",
        "outcome",
        "time_window",
        "export_format",
        "analysis_goal",
    ]
    assert next(row for row in empty.stages if row.id == "idea").status == "blocked"

    finished = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "gate_status": "analysis_only",
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "table1_summary.json",
                "manuscript_draft.json",
            ],
        },
    )
    by_id = {row.id: row for row in finished.stages}
    assert by_id["analysis"].status == "complete"
    assert by_id["interpretation"].status == "review_required"
    assert by_id["manuscript"].status == "review_required"
    assert finished.next_action_code == "human_review_and_reporting"


def test_result_interpretation_card_reuses_agent_claims_without_new_numbers() -> None:
    card = build_result_interpretation_card(
        run_id="run_safe",
        review={
            "gate": {
                "status": "analysis_only",
                "reason": "Human review remains required.",
                "checks": [{"name": "human_signoff", "passed": False}],
            },
            "readiness": {
                "status": "awaiting_human_signoff",
                "reportable": False,
            },
            "artifacts": [
                {"name": "table1_summary.json"},
                {"name": "manuscript_draft.json"},
            ],
        },
        manuscript={
            "claims": [
                {
                    "text": "The bounded Research Agent claim is analysis-only.",
                    "evidence_ids": ["ev_table1"],
                }
            ]
        },
    )
    assert card.status == "analysis_only"
    assert card.generated_numbers is False
    assert card.source == "research_agent_artifacts_only"
    assert card.claims[0].evidence_ids == ["ev_table1"]
    assert card.human_review_required is True


def test_idea_tool_never_accepts_a_host_path_from_the_model() -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-idea"),
        allowed_actions={"idea"},
    )
    with pytest.raises(PiCopilotError) as rejected:
        tool_module.execute_tool(
            "easyicu_mine_ideas",
            {"topic": "Aggregate ICU question", "path": "/private/source"},
            context,
        )
    assert rejected.value.code == "pi_tool_unknown_arguments"


def test_extraction_uses_bound_study_source_and_returns_no_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[dict[str, Any]] = []

    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: _complete_study(),
    )
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {"active_path": None, "sources": []},
    )
    from easyicu.webserver.routes import jobs as jobs_route

    def submit(body: dict[str, Any]) -> dict[str, Any]:
        submitted.append(dict(body))
        return {
            "job_id": "extract-job-1",
            "kind": "extract",
            "status": "running",
            "study_context_id": "study-workflow",
            "study_context_revision": 5,
        }

    monkeypatch.setattr(jobs_route, "jobs_extract", submit)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-extract",
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
            ),
        ),
        allowed_actions={"extract"},
    )
    result = tool_module.execute_tool("easyicu_start_extraction", {}, context)

    assert result["code"] == "easyicu_extraction_submitted"
    assert submitted[0]["path"] == "/private/prepared/source"
    assert submitted[0]["database"] == "mimiciv"
    assert "path" not in json.dumps(result)
    with pytest.raises(PiCopilotError) as stale:
        tool_module.execute_tool("easyicu_inspect_workflow", {}, context)
    assert stale.value.code == "pi_session_authority_stale"


def test_full_run_cannot_use_mock_as_scientific_output() -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-full",
            external_llm_opt_in=True,
        ),
        allowed_actions={"provider_run"},
    )
    result = tool_module.execute_tool(
        "easyicu_run",
        {"run_type": "full", "llm_provider": "mock"},
        context,
    )
    assert result["code"] == "pi_full_mock_not_scientific"


def test_full_run_delegates_to_research_agent_provider_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: {
            **_complete_study(),
            "question": "Bound aggregate scientific question",
        },
    )
    from easyicu.webserver.routes import agent as agent_route

    def submit(body: dict[str, Any]) -> dict[str, Any]:
        submitted.append(dict(body))
        return {
            "job_id": "agent-job-full",
            "kind": "agent-run",
            "status": "running",
            "study_context_id": "study-workflow",
            "study_context_revision": 5,
        }

    monkeypatch.setattr(agent_route, "jobs_agent_run", submit)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-full-owner",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
            ),
        ),
        allowed_actions={"provider_run"},
    )
    result = tool_module.execute_tool(
        "easyicu_run",
        {"run_type": "full", "llm_provider": "openai"},
        context,
    )

    assert result["code"] == "easyicu_full_run_submitted"
    assert submitted == [
        {
            "study_context_id": "study-workflow",
            "question": "Bound aggregate scientific question",
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": True,
        }
    ]
