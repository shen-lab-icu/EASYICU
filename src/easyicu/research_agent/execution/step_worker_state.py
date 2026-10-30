"""Data-only progress state for one analysis-step worker.

This object consolidates transient counters and provenance labels that belong
to one worker attempt.  It deliberately owns no checkpoint, evidence, input,
or scientific authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..schema import AgentRuntimeState


@dataclass(slots=True)
class StepWorkerProgress:
    """Mutable scratch progress for exactly one step worker."""

    resumed_code_reuse_used: bool = False
    critic_resume_repair_used: bool = False
    deterministic_fallback_used: bool = False
    deterministic_standard_executor_used: bool = False
    preexecution_runner_repair_name: str | None = None
    runner_repair_name: str | None = None
    concept_repair_attempts: int = 0
    concept_audit_error_count: int = 0
    deterministic_concept_repairs: int = 0
    applied_concept_repair_names: list[str] = field(default_factory=list)
    llm_repair_used: bool = False
    # Output-mutation counters only. They are not provider-call receipts or
    # the logical LLM budget, and concept repairs remain separately counted.
    repair_attempts: int = 0
    contract_repair_attempts: int = 0
    # Automatic structural repairs remain visible in
    # ``contract_repair_attempts`` but must not consume the independent paid
    # contract-repair allowance.
    llm_contract_repair_attempts: int = 0
    visual_repair_attempts: int = 0
    runtime_repair_attempts: int = 0

    def generation_mode(self, *, llm_repair_used: bool | None = None) -> str:
        """Describe the code that executed, preserving legacy label priority."""

        used_llm_repair = (
            self.llm_repair_used if llm_repair_used is None else bool(llm_repair_used)
        )
        if self.deterministic_standard_executor_used:
            return "deterministic_standard"
        if used_llm_repair:
            return "repaired"
        if self.deterministic_fallback_used:
            return "fallback"
        if self.runner_repair_name:
            return "runner_repaired"
        if self.repair_attempts > 0 or (
            self.concept_repair_attempts > 0 or self.deterministic_concept_repairs > 0
        ):
            return "repaired"
        if self.resumed_code_reuse_used:
            return "resumed_code_reuse"
        return "llm"


def _parallel_evidence_isolation_available(*, evidence: Any, run_dir: Path) -> bool:
    """Whether the host has a verified per-worker evidence view for this run.

    The current execution owner passes a live shared EvidenceStore and the run
    root to every worker. A later worker can therefore observe another worker's
    result before that batch has joined. Neither disjoint plan inputs nor a
    conflict check after the batch proves isolation. No current execution owner
    supplies a sealed worker-local reader and write merge, so concurrent
    execution is not admitted here. This is not a user-configurable switch.
    """

    del evidence, run_dir
    return False


def parallel_isolation_blocked(*, pipeline, steps, typed_dependencies, cohort_producer,
                               stop_after, evidence, run_dir, isolation_available, findings):
    """Report the serial fallback only when a parallel wave was otherwise eligible."""
    blocked = (pipeline._max_concurrent_steps > 1 and len(steps) > 1
        and not pipeline._enable_replanning and not typed_dependencies and not cohort_producer
        and stop_after is None and pipeline._submission_profile_name is None
        and not isolation_available(evidence=evidence, run_dir=run_dir))
    if blocked:
        from ..schema import ValidationFinding
        findings.append(ValidationFinding(
            validator="parallel_evidence_isolation", severity="info",
            message="Concurrent steps were run in plan order because the host cannot yet isolate each worker's evidence reads.",
            detail={"reason": "parallel_evidence_isolation_unavailable"},
        ))
    return blocked


def prepare_parallel_wave(*, runtime_state, shared_lock, flush_manifest, per_step_records,
                          evidence, run_dir, run_id, plan_result, typed_binding_resolver,
                          execute_step):
    """Capture a common authority view before scheduling the wave."""
    import copy
    import functools
    from dataclasses import replace
    from .runner import capture_wave_artifact_authority

    with shared_lock:
        flush_manifest()
        step_records = copy.deepcopy(per_step_records)
    view = evidence.read_view()
    authority = capture_wave_artifact_authority(
        workdir=run_dir, run_id=run_id, step_records=step_records,
        evidence_records=view.records(), resumed=plan_result.resume_state is not None,
    )
    claims = view.authoritative_numeric_claims(step_records)
    resolver = replace(typed_binding_resolver, evidence_store=view,
                       per_step_records=step_records, unpublished_evidence_store=evidence)
    return functools.partial(execute_step, batch_base=runtime_state,
        batch_binding_resolver=resolver, batch_numeric_claims=claims,
        batch_artifact_authority=authority)


def merge_parallel_worker_results(*, runtime_state, shared_lock, steps,
                                  parallel_runtime_states, merge_states):
    """Merge completed worker states in plan order after the wave joins."""
    with shared_lock:
        completed = [(step.step_id, parallel_runtime_states[step.step_id])
                     for step in steps if step.step_id in parallel_runtime_states]
    return merge_states(runtime_state, completed)


class StepExecutionState:
    """Keep sequential state and batch-local worker states with their owner."""

    def __init__(self, current: Any, lock: Any) -> None:
        self.current = current
        self.lock = lock
        self.parallel_runtime_states: dict[str, Any] = {}

    def execute(self, executor: Any, step: Any, *, batch: dict[str, Any],
                typed_binding_resolver: Any, **kwargs: Any) -> dict[str, Any]:
        base = batch.get("batch_base")
        batch_resolver = batch.get("batch_binding_resolver")
        result, record = executor(
            step, runtime_state=self.current if base is None else base,
            typed_binding_resolver=typed_binding_resolver if batch_resolver is None else batch_resolver,
            batch_numeric_claims=batch.get("batch_numeric_claims"),
            batch_artifact_authority=batch.get("batch_artifact_authority"), **kwargs,
        )
        if base is None:
            self.current = result
        else:
            with self.lock:
                self.parallel_runtime_states[step.step_id] = result
        return record


def merge_parallel_runtime_states(
    base: AgentRuntimeState,
    states: Sequence[tuple[str, AgentRuntimeState]],
) -> AgentRuntimeState:
    """Merge independent supervisor reflections in the declared plan order.

    Each worker must descend from the same frozen batch state and may append
    reflections, but may not replace prior memory or the scientific scope.
    Step-local request, critique and result fields come from the last step
    that actually advanced its supervisor state in plan order.
    """
    reflections = list(base.reflection_memory)
    selected = base
    fixed_fields = (
        "run_id", "analysis_family", "semantics", "extraction_request",
        "extraction_result", "manuscript_packet",
    )
    for step_id, state in states:
        if any(getattr(base, name) != getattr(state, name) for name in fixed_fields):
            raise ValueError("parallel supervisor state changed scientific scope")
        count = len(base.reflection_memory)
        if state.reflection_memory[:count] != base.reflection_memory:
            raise ValueError("parallel supervisor state replaced prior reflections")
        reflections.extend(state.reflection_memory[count:])
        if state.current_step is not None and state.current_step.step_id == step_id:
            selected = state
    return selected.model_copy(update={"reflection_memory": reflections})


def guarded_parallel_execute(step: Any, *, stopped: Any, execute_step: Any) -> Any:
    """Stop queued work after a raised worker while in-flight workers settle."""
    if stopped.is_set():
        return None
    try:
        return execute_step(step)
    except BaseException:
        stopped.set()
        raise
