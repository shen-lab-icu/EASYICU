"""Science-neutral scheduling primitives for the execute phase.

The caller remains the sole plan authority and resolves every replan decision.
This module only advances an already-authorized queue and applies transitions
returned by that caller.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from threading import Event
from typing import Any, Callable, Iterable, Optional, Sequence

from ..schema import AgentRuntimeState, AnalysisStep
from ..authority.provider_hard_stop import ProviderHardStopError
from ..contracts.runtime import RunExecutionState, RunTransition
from .replan_review import (
    _successful_run_transition_requests_replan as _successful_run_transition_requests_replan,
    _successful_step_requests_replan as _successful_step_requests_replan,
)


class RunCoordinator:
    """Advance step queues without deciding why a transition is authorized."""

    def run_sequential(
        self,
        *,
        state: RunExecutionState,
        execute_step: Callable[[AnalysisStep], Any],
        resolve_transition: Callable[[AnalysisStep, Any, bool], RunTransition],
        apply_revised_plan: Callable[[Any, set[str]], Sequence[AnalysisStep]],
        on_step_exception: Optional[Callable[[AnalysisStep, BaseException], None]] = (
            None
        ),
        resolve_run_halt: Optional[
            Callable[[AnalysisStep, Any], Optional[RunTransition]]
        ] = None,
    ) -> RunExecutionState:
        while state.remaining_steps:
            step = state.remaining_steps.pop(0)
            try:
                record = execute_step(step)
            except BaseException as error:  # noqa: BLE001 - see below
                # An escaping exception used to end the whole run: nothing
                # wrapped this call, so it left run_sequential, the execute
                # phase and pipeline.run, and the run finished with no sealed
                # manifest and no diagnosis -- one repeated word in a declared
                # input list cost a 14-minute run exactly this way. There are
                # ~2,900 raise statements behind this call, so the class cannot
                # be closed by enumerating them; it is closed by honouring the
                # contract that a step returns a record.
                #
                # run_parallel already treats a raising step as a handled event
                # (on_worker_error); only the sequential path did not.
                #
                # This is deliberately NOT a repair path. An unexpected
                # exception means an unknown invariant broke, so the run stops
                # here fail-closed rather than replanning around it -- the step
                # is recorded, the reason names the exception, and the caller
                # still seals its manifest.
                state.executed_step_ids.add(step.step_id)
                state.stop_reason = f"step_raised:{step.step_id}:{type(error).__name__}"
                if on_step_exception is not None:
                    on_step_exception(step, error)
                if isinstance(error, (KeyboardInterrupt, SystemExit, ProviderHardStopError)):
                    raise
                break
            state.executed_step_ids.add(step.step_id)
            # The caller's run-level halt check (input-authority corruption,
            # requested stop, pending replan review) is host policy, not a
            # replan decision. It runs after every executed step -- including
            # a failed auxiliary one -- so a corrupt input authority can never
            # be skipped by a branch that keeps the independent tail.
            if resolve_run_halt is not None:
                halt = resolve_run_halt(step, record)
                if halt is not None and halt.kind in {"stop", "pause"}:
                    state.stop_reason = halt.reason
                    break
            record_failed = isinstance(record, dict) and (
                str(record.get("status") or "").strip().lower() != "ok"
            )
            record_role = (
                str(record.get("planned_analysis_role") or "").strip().lower()
                if isinstance(record, dict)
                else ""
            )
            if record_failed and record_role in state.stop_failure_roles:
                # A declared required role owns the reported result; its failure
                # ends the queue here. The caller still seals the manifest.
                record["remaining_steps_suppressed"] = True
                state.stop_reason = "required_step_failed:" + (
                    str(record.get("status") or "").strip().lower() or "missing"
                )
                break
            if record_failed and state.stop_on_failure:
                # A submission-profile run must not let one auxiliary failure
                # swallow the independent tail. The caller's execution-time
                # dependency gate (figure-parent and typed ``kind:product``
                # edges) marks real consumers as ``skipped_dependency_failed``;
                # everything else keeps its slot in the denominator. The failed
                # step is not a replan anchor, so the replan-producing
                # transition callback is never invited for it.
                if state.stop_after_step_id == step.step_id:
                    state.stop_reason = "requested_stop_after_step"
                    break
                if resolve_run_halt is None:
                    # Compatibility for callers that only wired the full
                    # transition callback: ask it once, but honour only a
                    # run-level halt. A replan/continue answer is discarded;
                    # this step is still a failed auxiliary step.
                    maybe_halt = resolve_transition(
                        step,
                        record,
                        bool(state.remaining_steps),
                    )
                    if (
                        maybe_halt is not None
                        and maybe_halt.kind in {"stop", "pause"}
                    ):
                        state.stop_reason = maybe_halt.reason
                        break
                state.failed_step_ids.append(step.step_id)
                continue
            transition = resolve_transition(
                step,
                record,
                bool(state.remaining_steps),
            )
            if transition.kind in {"stop", "pause"}:
                state.stop_reason = transition.reason
                break
            if transition.kind == "replan":
                if transition.rerun_current_step:
                    state.executed_step_ids.discard(step.step_id)
                state.remaining_steps = list(
                    apply_revised_plan(
                        transition.revised_plan,
                        set(state.executed_step_ids),
                    )
                )
        return state

    def run_parallel(
        self,
        *,
        steps: Iterable[AnalysisStep],
        max_workers: int,
        execute_step: Callable[[AnalysisStep], Any],
        submit_step: Callable[[Any, Callable[..., Any], AnalysisStep], Any],
        on_worker_error: Callable[[AnalysisStep, BaseException], None],
    ) -> None:
        step_list = list(steps)
        stopped = Event()

        guarded_execute = partial(
            guarded_parallel_execute, stopped=stopped, execute_step=execute_step
        )

        stop_error: Optional[BaseException] = None
        with ThreadPoolExecutor(
            max_workers=min(int(max_workers), len(step_list)),
            thread_name_prefix="ra_step",
        ) as executor:
            futures = {submit_step(executor, guarded_execute, step): step for step in step_list}
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                error = future.exception()
                if error is not None:
                    on_worker_error(futures[future], error)
                    if isinstance(error, (ProviderHardStopError, KeyboardInterrupt, SystemExit)):
                        stop_error = stop_error or error
                    for pending in futures:
                        pending.cancel()
        if stop_error is not None:
            raise stop_error


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


__all__ = [
    "RunCoordinator", "RunExecutionState", "RunTransition",
    "merge_parallel_runtime_states",
]
