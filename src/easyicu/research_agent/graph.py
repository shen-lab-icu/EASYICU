"""LangGraph orchestration for the research-agent pipeline.

This module is the default wrapper that orchestrates the existing
``plan → execute → write → finalise`` phases of
:class:`~easyicu.research_agent.pipeline.ResearchAgentPipeline` as a
``langgraph.graph.StateGraph``. LangGraph is the default phase dispatcher; the
existing EasyICU receipts, capsules, evidence and checkpoint remain the sole
scientific and replay authority. A private legacy-dispatch switch is retained
only for parity tests and bounded rollback.

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

from importlib import metadata
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8 not supported anyway
    from typing_extensions import TypedDict  # type: ignore[no-redef]


__all__ = [
    "OrchestrationRuntimeReceipt",
    "PipelineGraphState",
    "build_pipeline_graph",
    "orchestration_runtime_receipt",
]


class OrchestrationRuntimeReceipt(BaseModel):
    """Non-scientific receipt identifying the phase dispatcher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.orchestration_runtime/1"] = (
        "easyicu.orchestration_runtime/1"
    )
    backend: Literal["langgraph", "legacy_sequential"]
    backend_version: str
    phase_order: tuple[str, ...] = ("plan", "execute", "write", "finalise")
    checkpoint_authority: Literal["easyicu_receipt_capsule_checkpoint"] = (
        "easyicu_receipt_capsule_checkpoint"
    )
    scientific_authority: Literal["easyicu_host_control_plane"] = (
        "easyicu_host_control_plane"
    )


def orchestration_runtime_receipt(*, use_graph: bool) -> OrchestrationRuntimeReceipt:
    """Return the exact dispatcher identity without changing scientific state."""

    if use_graph:
        try:
            version = metadata.version("langgraph")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "LangGraph is required by the default research-agent runtime; "
                "reinstall EasyICU so its core dependencies are complete"
            ) from exc
        return OrchestrationRuntimeReceipt(backend="langgraph", backend_version=version)
    return OrchestrationRuntimeReceipt(
        backend="legacy_sequential", backend_version="easyicu-legacy-1"
    )


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


def build_pipeline_graph(
    *,
    plan_invoker: Callable[[], Any],
    execute_invoker: Callable[[Any], Any],
    write_invoker: Callable[[Any, Any], Any],
    finalise_invoker: Callable[[Any, Any, Any], Any],
    provenance_hook: Optional[Callable[[Any], None]] = None,
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
        if plan_result.aborted_result is not None:
            return {
                "plan_result": plan_result,
                "final_result": plan_result.aborted_result,
                "aborted": True,
            }
        if provenance_hook is not None:
            provenance_hook(plan_result)
        return {"plan_result": plan_result, "aborted": False}

    def execute_node(state: PipelineGraphState) -> dict[str, Any]:
        execute_result = execute_invoker(state["plan_result"])
        return {"execute_result": execute_result}

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
    graph.add_node("execute", execute_node)
    graph.add_node("write", write_node)
    graph.add_node("finalise", finalise_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"abort": END, "continue": "execute"},
    )
    graph.add_edge("execute", "write")
    graph.add_edge("write", "finalise")
    graph.add_edge("finalise", END)
    return graph.compile()
