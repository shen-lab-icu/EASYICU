from __future__ import annotations

import inspect
from contextvars import ContextVar

import pytest

from easyicu.research_agent.schema import AnalysisStep


def _step(step_id: str) -> AnalysisStep:
    return AnalysisStep(step_id=step_id, intent=f"execute {step_id}")


def test_sequential_stop_is_resolved_after_step_execution() -> None:
    from easyicu.research_agent.run_coordination import (
        RunCoordinator,
        RunExecutionState,
        RunTransition,
    )

    events: list[str] = []
    state = RunExecutionState(
        remaining_steps=[_step("01"), _step("02")],
        executed_step_ids=set(),
    )

    def execute(step: AnalysisStep):
        events.append(f"execute:{step.step_id}")
        return {"step_id": step.step_id}

    def transition(step: AnalysisStep, record: dict, has_remaining: bool):
        events.append(f"transition:{step.step_id}:{has_remaining}")
        return RunTransition.stop("requested_stop")

    RunCoordinator().run_sequential(
        state=state,
        execute_step=execute,
        resolve_transition=transition,
        apply_revised_plan=lambda plan, executed: [],
    )

    assert events == ["execute:01", "transition:01:True"]
    assert state.executed_step_ids == {"01"}
    assert state.stop_reason == "requested_stop"


def test_directed_replan_retries_current_step_without_two_plan_authorities() -> None:
    from easyicu.research_agent.run_coordination import (
        RunCoordinator,
        RunExecutionState,
        RunTransition,
    )

    first = _step("01")
    retried = _step("01")
    tail = _step("02")
    revised_plan = object()
    calls: list[str] = []
    applied: list[tuple[object, set[str]]] = []
    state = RunExecutionState(
        remaining_steps=[first, tail],
        executed_step_ids=set(),
    )

    def transition(step: AnalysisStep, record: dict, has_remaining: bool):
        if calls == ["01"]:
            return RunTransition.replan(
                revised_plan,
                rerun_current_step=True,
            )
        return RunTransition.continue_run()

    def apply(plan: object, executed: set[str]):
        applied.append((plan, set(executed)))
        return [retried, tail]

    RunCoordinator().run_sequential(
        state=state,
        execute_step=lambda step: calls.append(step.step_id) or {},
        resolve_transition=transition,
        apply_revised_plan=apply,
    )

    assert calls == ["01", "01", "02"]
    assert applied == [(revised_plan, set())]
    assert state.executed_step_ids == {"01", "02"}


def test_success_replan_keeps_completed_step_out_of_rebuilt_queue() -> None:
    from easyicu.research_agent.run_coordination import (
        RunCoordinator,
        RunExecutionState,
        RunTransition,
    )

    revised_plan = object()
    calls: list[str] = []
    applied_executed: list[set[str]] = []
    state = RunExecutionState(
        remaining_steps=[_step("01"), _step("stale")],
        executed_step_ids={"00_probe"},
    )

    def transition(step: AnalysisStep, record: dict, has_remaining: bool):
        if step.step_id == "01":
            return RunTransition.replan(revised_plan)
        return RunTransition.continue_run()

    def apply(plan: object, executed: set[str]):
        applied_executed.append(set(executed))
        return [_step("02")]

    RunCoordinator().run_sequential(
        state=state,
        execute_step=lambda step: calls.append(step.step_id) or {},
        resolve_transition=transition,
        apply_revised_plan=apply,
    )

    assert calls == ["01", "02"]
    assert applied_executed == [{"00_probe", "01"}]


def test_empty_blocked_schedule_executes_nothing() -> None:
    from easyicu.research_agent.run_coordination import (
        RunCoordinator,
        RunExecutionState,
    )

    state = RunExecutionState(remaining_steps=[], executed_step_ids={"00_probe"})
    RunCoordinator().run_sequential(
        state=state,
        execute_step=lambda step: pytest.fail("blocked schedule was revived"),
        resolve_transition=lambda step, record, has_remaining: pytest.fail(
            "blocked schedule transitioned"
        ),
        apply_revised_plan=lambda plan, executed: pytest.fail(
            "blocked schedule replanned"
        ),
    )
    assert state.executed_step_ids == {"00_probe"}


def test_sequential_worker_exception_propagates_unchanged() -> None:
    from easyicu.research_agent.run_coordination import (
        RunCoordinator,
        RunExecutionState,
    )

    expected = RuntimeError("step failed")

    def fail(step: AnalysisStep):
        raise expected

    with pytest.raises(RuntimeError) as raised:
        RunCoordinator().run_sequential(
            state=RunExecutionState(
                remaining_steps=[_step("01")],
                executed_step_ids=set(),
            ),
            execute_step=fail,
            resolve_transition=lambda step, record, has_remaining: pytest.fail(
                "failed step must not transition"
            ),
            apply_revised_plan=lambda plan, executed: [],
        )
    assert raised.value is expected


def test_parallel_workers_use_supplied_context_submitter_and_report_errors() -> None:
    from easyicu.research_agent.pipeline_execute import _submit_in_current_context
    from easyicu.research_agent.run_coordination import RunCoordinator

    marker: ContextVar[str] = ContextVar("run_coordinator_marker", default="missing")
    marker.set("bound")
    observed: list[str] = []
    errors: list[BaseException] = []
    expected = ValueError("worker failed")

    def execute(step: AnalysisStep):
        observed.append(f"{step.step_id}:{marker.get()}")
        if step.step_id == "02":
            raise expected
        return {}

    RunCoordinator().run_parallel(
        steps=[_step("01"), _step("02")],
        max_workers=2,
        execute_step=execute,
        submit_step=_submit_in_current_context,
        on_worker_error=errors.append,
    )

    assert sorted(observed) == ["01:bound", "02:bound"]
    assert errors == [expected]


def test_run_coordinator_is_science_neutral_and_pipeline_owns_transitions() -> None:
    import easyicu.research_agent.pipeline_execute as pipeline_execute
    import easyicu.research_agent.run_coordination as run_coordination

    module_source = inspect.getsource(run_coordination)
    for forbidden in (
        "Validator",
        "Coder",
        "EvidenceStore",
        "ResearchContext",
        "estimand",
        "cohort_path",
        "target_outcome",
    ):
        assert forbidden not in module_source

    phase_source = inspect.getsource(pipeline_execute.run_execute_phase)
    assert "while remaining_steps:" not in phase_source
    assert phase_source.count("run_coordinator.run_sequential(") == 1
    assert phase_source.count("run_coordinator.run_parallel(") == 1
    corruption = phase_source.index('if run_input_authority_state["corrupted"]:')
    requested_stop = phase_source.index(
        "if step.step_id == requested_stop_after_step_id:", corruption
    )
    directed = phase_source.index("directed_plan = _maybe_directed_model_replan(")
    ordinary = phase_source.index(
        "and _successful_step_requests_replan(record)", directed
    )
    assert corruption < requested_stop < directed < ordinary
