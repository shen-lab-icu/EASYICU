"""Contracts for the sole explicit research-agent phase dispatcher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from easyicu.research_agent.orchestration.workflow import (
    HumanReviewDecision,
    HumanReviewRequest,
    WorkflowEngine,
    WorkflowCompleted,
    WorkflowPaused,
    build_pipeline_workflow,
    orchestration_runtime_receipt,
)


def _workflow(**overrides):
    calls: list[str] = []
    defaults = {
        "plan_invoker": lambda: (
            calls.append("plan") or SimpleNamespace(aborted_result=None)
        ),
        "execute_invoker": lambda _plan: calls.append("execute") or "executed",
        "write_invoker": (lambda _plan, _execute: calls.append("write") or "written"),
        "finalise_invoker": (
            lambda _plan, _execute, _write: calls.append("finalise") or "final"
        ),
    }
    defaults.update(overrides)
    return build_pipeline_workflow(**defaults), calls


def test_runtime_receipt_identifies_the_explicit_state_machine() -> None:
    receipt = orchestration_runtime_receipt()

    assert receipt.backend == "explicit_state_machine"
    assert receipt.phase_order == (
        "plan",
        "human_review",
        "execute",
        "write",
        "finalise",
    )


def test_builder_returns_the_framework_neutral_workflow_engine_contract() -> None:
    workflow, _calls = _workflow()

    assert isinstance(workflow, WorkflowEngine)


def test_workflow_runs_each_phase_once_in_order() -> None:
    workflow, calls = _workflow()

    outcome = workflow.start()

    assert isinstance(outcome, WorkflowCompleted)
    assert outcome.final_result == "final"
    assert calls == ["plan", "execute", "write", "finalise"]
    assert workflow.state == "completed"


def test_aborted_plan_skips_every_downstream_phase() -> None:
    plan_calls: list[str] = []
    workflow, downstream_calls = _workflow(
        plan_invoker=lambda: (
            plan_calls.append("plan") or SimpleNamespace(aborted_result="aborted")
        )
    )

    outcome = workflow.start()

    assert isinstance(outcome, WorkflowCompleted)
    assert outcome.final_result == "aborted"
    assert plan_calls == ["plan"]
    assert downstream_calls == []


def test_human_review_pause_requires_digest_bound_approval() -> None:
    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="Approve tested package in immutable image",
        authority_sha256="a" * 64,
        payload={"request_id": "cap-123"},
    )
    recorded: list[dict] = []
    workflow, calls = _workflow(
        human_review_invoker=lambda _plan: (request,),
        human_review_recorder=recorded.extend,
    )

    paused = workflow.start()

    assert isinstance(paused, WorkflowPaused)
    assert paused.requests == (request,)
    assert calls == ["plan"]
    decision = HumanReviewDecision(
        review_id=request.review_id,
        authority_sha256=request.authority_sha256,
        decision="approved",
        reviewer="maintainer",
        decided_at="2026-07-22T05:00:00Z",
    )
    completed = workflow.resume([decision])
    assert completed.final_result == "final"
    assert completed.human_review_decisions[0]["review_id"] == request.review_id
    assert recorded[0]["authority_sha256"] == request.authority_sha256
    assert calls == ["plan", "execute", "write", "finalise"]


def test_human_review_rejects_duplicate_request_ids_before_pause() -> None:
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Confirm an unresolved scientific stop",
        authority_sha256="b" * 64,
        payload={"finding": "positivity_not_established"},
    )
    workflow, _calls = _workflow(human_review_invoker=lambda _plan: (request, request))

    with pytest.raises(ValueError, match="requests must have unique review_id"):
        workflow.start()


def test_human_review_rejects_duplicate_decisions() -> None:
    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="Approve tested package in immutable image",
        authority_sha256="c" * 64,
        payload={"request_id": "cap-duplicate"},
    )
    workflow, _calls = _workflow(human_review_invoker=lambda _plan: (request,))
    assert isinstance(workflow.start(), WorkflowPaused)
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision=decision,
            reviewer="maintainer",
            decided_at="2026-07-22T07:30:00Z",
        )
        for decision in ("rejected", "approved")
    ]

    with pytest.raises(ValueError, match="decisions must have unique review_id"):
        workflow.resume(decisions)


def test_retired_graph_builder_cannot_recreate_a_shadow_dispatcher() -> None:
    from easyicu.research_agent.graph import build_pipeline_graph

    with pytest.raises(RuntimeError, match="retired"):
        build_pipeline_graph()


def test_production_modules_do_not_import_langgraph() -> None:
    root = Path(__file__).resolve().parents[2]
    production = root / "src" / "easyicu" / "research_agent"
    offenders = []
    for path in production.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import langgraph" in text or "from langgraph" in text:
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


def test_langgraph_is_not_a_core_dependency() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert not any(item.lower().startswith("langgraph") for item in dependencies)


def test_langgraph_is_not_in_the_runner_lock() -> None:
    root = Path(__file__).resolve().parents[2]
    lock_lines = (
        root / "src/easyicu/research_agent/runner_image/requirements.lock"
    ).read_text(encoding="utf-8").splitlines()

    assert not any(line.lower().startswith("langgraph==") for line in lock_lines)


def test_run_with_graph_is_only_a_deprecated_alias() -> None:
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    pipeline = ResearchAgentPipeline.__new__(ResearchAgentPipeline)
    pipeline.run = lambda **kwargs: kwargs

    with pytest.warns(DeprecationWarning, match="run_with_graph"):
        result = pipeline.run_with_graph(question="q")

    assert result == {"question": "q"}
