"""LangGraph orchestration for the research-agent pipeline.

This module is the default wrapper that orchestrates the existing
``plan → execute → write → finalise`` phases of
:class:`~easyicu.research_agent.pipeline.ResearchAgentPipeline` as a
``langgraph.graph.StateGraph``. LangGraph is the default phase dispatcher; the
existing EasyICU receipts, capsules, evidence and checkpoint remain the sole
scientific and replay authority.

Why this design:

* The graph receives **invoker callables** rather than raw kwargs. The
  pipeline closes over its prelude locals (audit logger, progress
  emitter, run dir, etc.) when constructing these callables, so the
  graph itself stays free of pipeline-specific argument plumbing.
* Phase outputs flow through a ``TypedDict`` graph state. The state
  uses ``Any`` for the phase result fields because the underlying
  ``_PlanPhaseResult`` / ``_ExecutePhaseResult`` / ``_WritePhaseResult``
  dataclasses are pipeline-internal and we do not want to import them
  at module top to avoid a circular dependency.
* Aborts during planning route directly to ``END`` without running
  execute/write/finalise. The pipeline's own ``_finalise_aborted`` has
  already been called inside ``_run_plan_phase`` in that branch, so the
  ``final_result`` is populated by the plan node.

``langgraph`` is a core dependency of the research-agent runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8 not supported anyway
    from typing_extensions import TypedDict  # type: ignore[no-redef]


__all__ = [
    "HumanReviewDecision",
    "HumanReviewRequest",
    "OrchestrationRuntimeReceipt",
    "PipelineGraphState",
    "build_pipeline_graph",
    "orchestration_runtime_receipt",
]


def _review_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HumanReviewRequest(BaseModel):
    """Digest-bound pause request; it carries no authority by itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str = Field(pattern=r"^review-[0-9a-f]{16}$")
    kind: Literal["protocol_claim", "capability_request", "scientific_stop"]
    summary: str = Field(min_length=1, max_length=1_000)
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, Any]

    @model_validator(mode="after")
    def _id_binds_payload(self) -> "HumanReviewRequest":
        expected = (
            "review-" + _review_digest(self.model_dump(exclude={"review_id"}))[:16]
        )
        if self.review_id != expected:
            raise ValueError("human-review id does not bind request contents")
        return self

    @classmethod
    def create(
        cls,
        *,
        kind: Literal["protocol_claim", "capability_request", "scientific_stop"],
        summary: str,
        authority_sha256: str,
        payload: Mapping[str, Any],
    ) -> "HumanReviewRequest":
        body = {
            "kind": kind,
            "summary": summary,
            "authority_sha256": authority_sha256,
            "payload": dict(payload),
        }
        return cls(review_id="review-" + _review_digest(body)[:16], **body)


class HumanReviewDecision(BaseModel):
    """Explicit operator decision bound to the paused request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=200)
    decided_at: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=1_000)


class OrchestrationRuntimeReceipt(BaseModel):
    """Non-scientific receipt identifying the phase dispatcher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.orchestration_runtime/1"] = (
        "easyicu.orchestration_runtime/1"
    )
    backend: Literal["langgraph"]
    backend_version: str
    phase_order: tuple[str, ...] = ("plan", "execute", "write", "finalise")
    checkpoint_authority: Literal["easyicu_receipt_capsule_checkpoint"] = (
        "easyicu_receipt_capsule_checkpoint"
    )
    scientific_authority: Literal["easyicu_host_control_plane"] = (
        "easyicu_host_control_plane"
    )


def orchestration_runtime_receipt() -> OrchestrationRuntimeReceipt:
    """Return the exact dispatcher identity without changing scientific state."""

    try:
        version = metadata.version("langgraph")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "LangGraph is required by the default research-agent runtime; "
            "reinstall EasyICU so its core dependencies are complete"
        ) from exc
    return OrchestrationRuntimeReceipt(backend="langgraph", backend_version=version)


class PipelineGraphState(TypedDict, total=False):
    """Mutable state passed between graph nodes.

    Fields are populated incrementally as each node runs. ``aborted``
    is set by the plan node when ``_PlanPhaseResult.aborted_result`` is
    not ``None``; the conditional edge after ``plan`` then routes
    directly to ``END`` and the value of ``final_result`` is already
    correct.
    """

    plan_result: Any  # _PlanPhaseResult
    execute_result: Any  # _ExecutePhaseResult
    write_result: Any  # _WritePhaseResult
    final_result: Any  # PipelineResult
    aborted: bool
    human_review_decisions: tuple[dict[str, Any], ...]


def build_pipeline_graph(
    *,
    plan_invoker: Callable[[], Any],
    execute_invoker: Callable[[Any], Any],
    write_invoker: Callable[[Any, Any], Any],
    finalise_invoker: Callable[[Any, Any, Any], Any],
    provenance_hook: Optional[Callable[[Any], None]] = None,
    human_review_invoker: Optional[
        Callable[[Any], Sequence[HumanReviewRequest]]
    ] = None,
    checkpointer: Any = None,
):
    """Build the compiled langgraph StateGraph for the pipeline.

    Parameters
    ----------
    plan_invoker:
        Zero-arg callable that runs the plan phase and returns a
        ``_PlanPhaseResult``. If its ``aborted_result`` field is not
        ``None`` the graph routes to ``END``.
    execute_invoker:
        Receives the plan result, returns an ``_ExecutePhaseResult``.
    write_invoker:
        Receives ``(plan_result, execute_result)``, returns a
        ``_WritePhaseResult``.
    finalise_invoker:
        Receives ``(plan_result, execute_result, write_result)``,
        returns the final ``PipelineResult``.
    provenance_hook:
        Optional callable invoked after a successful plan with the
        plan result, mirroring the O27 raw-EHR provenance step in
        :meth:`ResearchAgentPipeline.run`.

    Returns
    -------
    A compiled langgraph runnable. Invoke it with an empty initial
    state dict; the final state's ``final_result`` field is the
    ``PipelineResult``.
    """

    from langgraph.graph import StateGraph, END

    def plan_node(state: PipelineGraphState) -> dict[str, Any]:
        plan_result = plan_invoker()
        aborted_result = (
            plan_result.get("aborted_result")
            if isinstance(plan_result, Mapping)
            else plan_result.aborted_result
        )
        if aborted_result is not None:
            return {
                "plan_result": plan_result,
                "final_result": aborted_result,
                "aborted": True,
            }
        if provenance_hook is not None:
            provenance_hook(plan_result)
        return {"plan_result": plan_result, "aborted": False}

    def execute_node(state: PipelineGraphState) -> dict[str, Any]:
        execute_result = execute_invoker(state["plan_result"])
        return {"execute_result": execute_result}

    def human_review_node(state: PipelineGraphState) -> dict[str, Any]:
        if human_review_invoker is None:
            return {"human_review_decisions": ()}
        requests = tuple(human_review_invoker(state["plan_result"]))
        if not requests:
            return {"human_review_decisions": ()}
        request_ids = [item.review_id for item in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("human review requests must have unique review_id values")
        from langgraph.types import interrupt

        raw = interrupt(
            {
                "schema_version": "easyicu.human_review_interrupt/1",
                "requests": [item.model_dump(mode="json") for item in requests],
            }
        )
        if not isinstance(raw, Mapping) or not isinstance(raw.get("decisions"), list):
            raise ValueError("human review resume payload must contain decisions")
        decisions = tuple(
            HumanReviewDecision.model_validate(item) for item in raw["decisions"]
        )
        decision_ids = [item.review_id for item in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("human review decisions must have unique review_id values")
        expected = {item.review_id: item for item in requests}
        observed = {item.review_id: item for item in decisions}
        if set(observed) != set(expected):
            raise ValueError("human review decisions must cover exact paused requests")
        for review_id, decision in observed.items():
            request = expected[review_id]
            if decision.authority_sha256 != request.authority_sha256:
                raise ValueError("human review decision authority digest mismatch")
            if decision.decision != "approved":
                raise RuntimeError(f"human review rejected request {review_id}")
        return {
            "human_review_decisions": tuple(
                item.model_dump(mode="json") for item in decisions
            )
        }

    def write_node(state: PipelineGraphState) -> dict[str, Any]:
        write_result = write_invoker(state["plan_result"], state["execute_result"])
        return {"write_result": write_result}

    def finalise_node(state: PipelineGraphState) -> dict[str, Any]:
        final_result = finalise_invoker(
            state["plan_result"],
            state["execute_result"],
            state["write_result"],
        )
        return {"final_result": final_result}

    def route_after_plan(state: PipelineGraphState) -> str:
        return "abort" if state.get("aborted") else "continue"

    graph: StateGraph = StateGraph(PipelineGraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("execute", execute_node)
    graph.add_node("write", write_node)
    graph.add_node("finalise", finalise_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"abort": END, "continue": "human_review"},
    )
    graph.add_edge("human_review", "execute")
    graph.add_edge("execute", "write")
    graph.add_edge("write", "finalise")
    graph.add_edge("finalise", END)
    return graph.compile(checkpointer=checkpointer)
