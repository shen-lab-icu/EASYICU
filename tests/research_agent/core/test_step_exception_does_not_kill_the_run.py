"""A step that raises must end the run, not escape it.

Nothing wrapped ``execute_step`` in ``run_sequential``, so any exception raised
anywhere behind it left the coordinator, ``run_execute_phase`` and
``pipeline.run``. fresh16 died that way (``BENCH_EXIT=5``) after ~14 minutes of
real provider spend: the run directory held no final manifest and nothing named
the failing step -- the operator got a bare traceback.

There are ~2,900 ``raise`` statements in ``research_agent``, so this class
cannot be closed by finding them. It is closed by honouring the contract that a
step returns a record: the coordinator catches, records, and stops.

``run_parallel`` already treated a raising step as a handled event
(``on_worker_error``); only the sequential path did not.

This is deliberately not a repair path -- an unexpected exception means an
unknown invariant broke, so the run stops fail-closed rather than replanning
around it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from easyicu.research_agent.execution.run_coordination import (
    RunCoordinator,
    RunExecutionState,
    RunTransition,
)
from easyicu.research_agent.schema import AnalysisStep

_PHASE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src/easyicu/research_agent/execution/phase.py"
)


def _step(step_id: str) -> AnalysisStep:
    return AnalysisStep(step_id=step_id, intent=f"execute {step_id}")


def _state(*step_ids: str) -> RunExecutionState:
    return RunExecutionState(
        remaining_steps=[_step(s) for s in step_ids],
        executed_step_ids=set(),
    )


def _never(*args, **kwargs):  # pragma: no cover - must not be reached
    raise AssertionError("the run continued past a raising step")


def test_the_exception_does_not_escape_the_coordinator() -> None:
    """The regression: this used to propagate out of pipeline.run."""

    state = _state("01", "02")

    def execute(step: AnalysisStep):
        raise ValueError("Planner-declared raw inputs must be unique")

    RunCoordinator().run_sequential(
        state=state,
        execute_step=execute,
        resolve_transition=_never,
        apply_revised_plan=_never,
    )

    assert state.stop_reason == "step_raised:01:ValueError"


def test_no_later_step_runs_after_a_raise() -> None:
    """Fail closed: an unknown broken invariant does not get replanned around."""

    state = _state("01", "02", "03")
    seen: list[str] = []

    def execute(step: AnalysisStep):
        seen.append(step.step_id)
        raise RuntimeError("boom")

    RunCoordinator().run_sequential(
        state=state,
        execute_step=execute,
        resolve_transition=_never,
        apply_revised_plan=_never,
    )

    assert seen == ["01"]
    # The unrun steps stay queued and unexecuted, exactly as on the existing
    # ``stop`` path -- they must not be silently recorded as attempted.
    assert [step.step_id for step in state.remaining_steps] == ["02", "03"]
    assert state.executed_step_ids == {"01"}


def test_the_failing_step_is_reported_to_the_caller() -> None:
    """The caller owns record shape, so it is the caller that gets told."""

    state = _state("06_missingness_event_timing_audit")
    reported: list[tuple[str, str]] = []

    def execute(step: AnalysisStep):
        raise ValueError("no context descriptor")

    RunCoordinator().run_sequential(
        state=state,
        execute_step=execute,
        resolve_transition=_never,
        apply_revised_plan=_never,
        on_step_exception=lambda step, error: reported.append(
            (step.step_id, type(error).__name__)
        ),
    )

    assert reported == [("06_missingness_event_timing_audit", "ValueError")]


def test_the_step_still_counts_as_attempted() -> None:
    """A resume must not silently re-run a step that already blew up."""

    state = _state("01")

    def execute(step: AnalysisStep):
        raise RuntimeError("boom")

    RunCoordinator().run_sequential(
        state=state,
        execute_step=execute,
        resolve_transition=_never,
        apply_revised_plan=_never,
    )

    assert state.executed_step_ids == {"01"}


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_operator_interrupts_are_still_propagated(interrupt) -> None:
    """Catching BaseException must not make the run unkillable."""

    state = _state("01")

    def execute(step: AnalysisStep):
        raise interrupt()

    with pytest.raises(interrupt):
        RunCoordinator().run_sequential(
            state=state,
            execute_step=execute,
            resolve_transition=_never,
            apply_revised_plan=_never,
        )


def test_a_normal_run_is_unaffected() -> None:
    """The happy path keeps resolving transitions exactly as before."""

    state = _state("01", "02")
    seen: list[str] = []

    def execute(step: AnalysisStep):
        seen.append(step.step_id)
        return {"step_id": step.step_id, "status": "ok"}

    RunCoordinator().run_sequential(
        state=state,
        execute_step=execute,
        resolve_transition=lambda step, record, more: RunTransition.continue_run(),
        apply_revised_plan=_never,
    )

    assert seen == ["01", "02"]
    assert state.stop_reason is None


# ---------------------------------------------------------------------------
# Reachability: the production caller must actually pass a recorder.
# ---------------------------------------------------------------------------


def test_the_recorder_persists_the_traceback() -> None:
    """Structural check -- the recorder is a closure inside run_execute_phase.

    Not propagating the exception is what keeps the run sealable, but with
    ~2,900 raise sites behind the call the frames are the only thing that says
    which one fired. Dropping them would trade one lost diagnosis for another.
    """

    tree = ast.parse(_PHASE.read_text(encoding="utf-8"))
    recorder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_step_record_step_exception"
    )
    body = ast.unparse(recorder)

    assert "traceback.format_exception" in body
    assert "'traceback'" in body or '"traceback"' in body


def test_the_execute_phase_passes_its_own_recorder() -> None:
    tree = ast.parse(_PHASE.read_text(encoding="utf-8"))
    passed = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "run_sequential"
        for keyword in node.keywords
        if keyword.arg == "on_step_exception"
    ]

    assert len(passed) == 1
    assert getattr(passed[0], "id", "") == "_record_step_exception"
