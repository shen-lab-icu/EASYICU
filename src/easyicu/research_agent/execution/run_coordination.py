"""Science-neutral scheduling primitives for the execute phase.

The caller remains the sole plan authority and resolves every replan decision.
This module only advances an already-authorized queue and applies transitions
returned by that caller.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

from ..schema import AnalysisStep


@dataclass(slots=True)
class RunExecutionState:
    """Mutable queue state without a second copy of the current plan."""

    remaining_steps: list[AnalysisStep]
    executed_step_ids: set[str]
    stop_on_failure: bool = False
    stop_reason: Optional[str] = None


@dataclass(frozen=True)
class RunTransition:
    """A caller-authorized scheduling transition after one completed step."""

    kind: Literal["continue", "stop", "replan"]
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
    ) -> RunExecutionState:
        while state.remaining_steps:
            step = state.remaining_steps.pop(0)
            record = execute_step(step)
            state.executed_step_ids.add(step.step_id)
            if (
                state.stop_on_failure
                and isinstance(record, dict)
                and str(record.get("status") or "").strip().lower() != "ok"
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
            if transition.kind == "stop":
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
