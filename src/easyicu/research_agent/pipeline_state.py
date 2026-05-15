"""Public state contract for the pipeline's three phases.

:class:`ResearchAgentPipeline.run` already factors execution into three
private methods — ``_run_plan_phase`` → ``_run_execute_phase`` →
``_run_write_phase`` — that pass typed result dataclasses between
themselves. Those dataclasses (``_PlanPhaseResult`` / ``_ExecutePhaseResult``
/ ``_WritePhaseResult``) are the de-facto inter-phase contract.

This module promotes them to first-class public types so:

* future ``PlanPhaseRunner`` / ``ExecutePhaseRunner`` / ``WritePhaseRunner``
  extractions can publish a stable signature without touching internal
  names;
* tests and external orchestrators (LangGraph, benchmark harnesses,
  retry shims) can build / inspect / mutate phase state without
  importing leading-underscore names from ``pipeline``;
* the data flow between phases is documented in one place rather than
  scattered across a 12k-line module.

The classes are re-exported, not redefined. Pipeline keeps the original
``_PlanPhaseResult`` etc. as aliases, so all existing call sites in
``pipeline.py`` continue to work identically.

Phase data flow
---------------
::

    PlanPhaseState
        ├─ context (ResearchContext) ── grounding for every prompt
        ├─ evidence (EvidenceStore)  ── SHA256 store, mutated by execute
        ├─ plan (AnalysisPlan)       ── ordered analysis steps
        ├─ findings (list)           ── validator findings to date
        ├─ role_resolver / cost_meter / repro_envelope — runtime services
        └─ aborted_result (PipelineResult | None) — short-circuit signal
                ▼
    ExecutePhaseState
        ├─ plan (possibly replanned)
        ├─ per_step_records          ── what each step produced
        ├─ probe_summary             ── probe outputs feeding replanning
        ├─ runtime_state             ── typed shared state for agents
        └─ flush_partial_manifest    ── partial-result flusher for retries
                ▼
    WritePhaseState
        ├─ literature (LiteratureBundle | None)
        ├─ bound_path (manuscript_scaffold_bound.md)
        └─ manuscript_packet (ManuscriptDraftPacket | None)

Naming convention
-----------------
The public names use ``State`` rather than ``Result`` because they
function as carrier state (not just return values) — each phase reads
and writes the same evidence store, findings list, and runtime state.
The original ``Result`` names are kept as aliases for back-compat with
internal type hints in :mod:`pipeline`.
"""

from __future__ import annotations

from .pipeline import (
    _ExecutePhaseResult as ExecutePhaseState,
    _PlanPhaseResult as PlanPhaseState,
    _WritePhaseResult as WritePhaseState,
)

__all__ = [
    "PlanPhaseState",
    "ExecutePhaseState",
    "WritePhaseState",
]
