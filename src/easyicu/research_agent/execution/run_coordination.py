"""Science-neutral scheduling primitives for the execute phase.

The caller remains the sole plan authority and resolves every replan decision.
This module only advances an already-authorized queue and applies transitions
returned by that caller.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

from ..schema import AnalysisStep

_SUCCESS_REPLAN_REQUEST_FIELDS = (
    "replan_requested",
    "plan_revision_requested",
)


def _successful_step_requests_replan(
    record: Mapping[str, Any],
    *,
    progressive_observation_loop: bool = False,
) -> bool:
    """Decide whether a successful observation authorizes suffix revision."""

    if str(record.get("status") or "") != "ok":
        return False
    containers = [record]
    summary = record.get("step_summary")
    if isinstance(summary, Mapping):
        containers.append(summary)
    explicit_request = any(
        container.get(field) is True
        for container in containers
        for field in _SUCCESS_REPLAN_REQUEST_FIELDS
    )
    generation_mode = str(record.get("generation_mode") or "").strip().lower()
    authority_kind = str(record.get("step_authority_kind") or "").strip().lower()
    host_deterministic = generation_mode.startswith(
        "deterministic_"
    ) or authority_kind.startswith("host_deterministic_")
    return explicit_request or (
        progressive_observation_loop and not host_deterministic
    )


def _successful_run_transition_requests_replan(
    pipeline: Any,
    record: Mapping[str, Any],
    has_remaining: bool,
) -> bool:
    """Apply host strategy and queue guards to one clean observation."""

    return bool(
        pipeline._enable_replanning
        and has_remaining
        and _successful_step_requests_replan(
            record,
            progressive_observation_loop=(
                getattr(pipeline, "_planner_strategy", None) == "progressive_v2"
            ),
        )
    )


@dataclass(slots=True)
class RunExecutionState:
    """Mutable queue state without a second copy of the current plan."""

    remaining_steps: list[AnalysisStep]
    executed_step_ids: set[str]
    stop_on_failure: bool = False
    stop_failure_roles: frozenset[str] = frozenset()
    stop_reason: Optional[str] = None


@dataclass(frozen=True)
class RunTransition:
    """A caller-authorized scheduling transition after one completed step."""

    kind: Literal["continue", "stop", "pause", "replan"]
    reason: Optional[str] = None
    revised_plan: Any = None
    rerun_current_step: bool = False

    @classmethod
    def continue_run(cls) -> "RunTransition":
        return cls(kind="continue")

    @classmethod
    def stop(cls, reason: str) -> "RunTransition":
        return cls(kind="stop", reason=str(reason))

    @classmethod
    def pause(cls, reason: str) -> "RunTransition":
        """Stop scheduling without converting a review wait into a failure."""

        return cls(kind="pause", reason=str(reason))

    @classmethod
    def replan(
        cls,
        revised_plan: Any,
        *,
        rerun_current_step: bool = False,
    ) -> "RunTransition":
        return cls(
            kind="replan",
            revised_plan=revised_plan,
            rerun_current_step=bool(rerun_current_step),
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
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                break
            state.executed_step_ids.add(step.step_id)
            record_failed = isinstance(record, dict) and (
                str(record.get("status") or "").strip().lower() != "ok"
            )
            record_role = (
                str(record.get("planned_analysis_role") or "").strip().lower()
                if isinstance(record, dict)
                else ""
            )
            if record_failed and (
                state.stop_on_failure or record_role in state.stop_failure_roles
            ):
                record["remaining_steps_suppressed"] = True
                state.stop_reason = "required_step_failed:" + (
                    str(record.get("status") or "").strip().lower() or "missing"
                )
                break
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
        on_worker_error: Callable[[BaseException], None],
    ) -> None:
        step_list = list(steps)
        with ThreadPoolExecutor(
            max_workers=min(int(max_workers), len(step_list)),
            thread_name_prefix="ra_step",
        ) as executor:
            futures = [submit_step(executor, execute_step, step) for step in step_list]
            for future in as_completed(futures):
                error = future.exception()
                if error is not None:
                    on_worker_error(error)


__all__ = ["RunCoordinator", "RunExecutionState", "RunTransition"]
