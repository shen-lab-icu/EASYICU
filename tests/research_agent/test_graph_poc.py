"""Tests for the sole LangGraph phase dispatcher.

``pipeline.run_with_graph(...)`` routes the existing
``plan → execute → write → finalise`` phases through a
``langgraph.graph.StateGraph``. The wrapper is intended to have
the default ``pipeline.run(...)`` path.

LangGraph is a core research-agent dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _run_args(ra, cohort):
    return dict(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=cohort,
        cohort_name="graph_poc",
        database="synthetic",
        target_outcome="death",
    )


def test_default_run_uses_langgraph(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "graph_run",
        llm=ra.MockLLMClient(),
        runner_kind="subprocess",
    )
    result = pipeline.run(**_run_args(ra, synthetic_cohort))
    assert result.run_id
    assert Path(result.manifest_path).exists()
    assert result.evidence_count >= 1
    receipt = Path(result.workdir) / "orchestration_runtime.json"
    assert '"backend": "langgraph"' in receipt.read_text(encoding="utf-8")


def test_build_pipeline_graph_is_compiled_runnable(ra):
    """The graph builder returns a compiled runnable so that callers
    can ``invoke({})`` it directly without re-running ``compile()``.
    """
    from easyicu.research_agent.graph import build_pipeline_graph

    def _noop_plan():
        class _R:
            aborted_result = "stub"
            findings = []
            evidence = None

        return _R()

    graph = build_pipeline_graph(
        plan_invoker=_noop_plan,
        execute_invoker=lambda p: None,
        write_invoker=lambda p, e: None,
        finalise_invoker=lambda p, e, w: None,
    )
    assert hasattr(
        graph, "invoke"
    ), "build_pipeline_graph must return a compiled runnable"

    final_state = graph.invoke({})
    assert (
        final_state["final_result"] == "stub"
    ), "abort route must surface the aborted_result"


def test_human_review_interrupt_requires_digest_bound_approval() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from easyicu.research_agent.graph import (
        HumanReviewDecision,
        HumanReviewRequest,
        build_pipeline_graph,
    )

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="Approve tested package in immutable image",
        authority_sha256="a" * 64,
        payload={"request_id": "cap-123"},
    )
    graph = build_pipeline_graph(
        plan_invoker=lambda: {"aborted_result": None},
        execute_invoker=lambda _plan: "executed",
        write_invoker=lambda _plan, _execute: "written",
        finalise_invoker=lambda _plan, _execute, _write: "final",
        human_review_invoker=lambda _plan: (request,),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-test"}}
    paused = graph.invoke({}, config=config)
    assert paused["__interrupt__"]
    decision = HumanReviewDecision(
        review_id=request.review_id,
        authority_sha256=request.authority_sha256,
        decision="approved",
        reviewer="maintainer",
        decided_at="2026-07-22T05:00:00Z",
    )
    resumed = graph.invoke(
        Command(resume={"decisions": [decision.model_dump(mode="json")]}),
        config=config,
    )
    assert resumed["final_result"] == "final"
    assert resumed["human_review_decisions"][0]["review_id"] == request.review_id


def test_human_review_rejects_duplicate_request_ids_before_interrupt() -> None:
    from easyicu.research_agent.graph import HumanReviewRequest, build_pipeline_graph

    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Confirm an unresolved scientific stop",
        authority_sha256="b" * 64,
        payload={"finding": "positivity_not_established"},
    )
    graph = build_pipeline_graph(
        plan_invoker=lambda: {"aborted_result": None},
        execute_invoker=lambda _plan: "executed",
        write_invoker=lambda _plan, _execute: "written",
        finalise_invoker=lambda _plan, _execute, _write: "final",
        human_review_invoker=lambda _plan: (request, request),
    )

    with pytest.raises(ValueError, match="requests must have unique review_id"):
        graph.invoke({})


def test_human_review_rejects_duplicate_decisions_including_reject_then_approve() -> (
    None
):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from easyicu.research_agent.graph import (
        HumanReviewDecision,
        HumanReviewRequest,
        build_pipeline_graph,
    )

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="Approve tested package in immutable image",
        authority_sha256="c" * 64,
        payload={"request_id": "cap-duplicate"},
    )
    graph = build_pipeline_graph(
        plan_invoker=lambda: {"aborted_result": None},
        execute_invoker=lambda _plan: "executed",
        write_invoker=lambda _plan, _execute: "written",
        finalise_invoker=lambda _plan, _execute, _write: "final",
        human_review_invoker=lambda _plan: (request,),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "duplicate-review-test"}}
    paused = graph.invoke({}, config=config)
    assert paused["__interrupt__"]
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision=decision,
            reviewer="maintainer",
            decided_at="2026-07-22T07:30:00Z",
        ).model_dump(mode="json")
        for decision in ("rejected", "approved")
    ]

    with pytest.raises(ValueError, match="decisions must have unique review_id"):
        graph.invoke(Command(resume={"decisions": decisions}), config=config)
