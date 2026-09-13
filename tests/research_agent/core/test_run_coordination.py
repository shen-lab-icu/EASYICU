from __future__ import annotations

import inspect
from contextvars import ContextVar
from types import SimpleNamespace
from functools import partial
from threading import RLock

import pytest

from easyicu.research_agent.schema import AnalysisStep


def _step(step_id: str) -> AnalysisStep:
    return AnalysisStep(step_id=step_id, intent=f"execute {step_id}")


def test_parallel_exception_seals_identity_and_flushes_terminal_record():
    from easyicu.research_agent.execution.run_coordination import RunCoordinator
    from easyicu.research_agent.execution.phase import _step_record_step_exception
    records, findings, flushes = [], [], []

    def execute(step):
        if step.step_id == "bad":
            raise RuntimeError("synthetic crash")
        return {"step_id": step.step_id, "status": "ok"}

    RunCoordinator().run_parallel(
        steps=[_step("bad"), _step("good")], max_workers=2,
        execute_step=execute, submit_step=lambda pool, fn, step: pool.submit(fn, step),
        on_worker_error=partial(
            _step_record_step_exception, shared_lock=RLock(), findings=findings,
            per_step_records=records, _append_terminal_step_record=lambda rows, row: rows.append(row),
            _flush_partial_manifest=lambda payload: flushes.append((payload, list(records))),
            parallel=True,
        ),
    )
    assert len(records) == 1
    assert records[0]["step_id"] == "bad"
    assert records[0]["status"] == "execution_raised"
    assert "synthetic crash" in records[0]["traceback"]
    assert flushes[0][1] == records


def test_sequential_stop_is_resolved_after_step_execution() -> None:
    from easyicu.research_agent.execution.run_coordination import (
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


def test_sequential_auxiliary_failure_keeps_independent_tail() -> None:
    """A submission-profile auxiliary failure must not swallow the tail.

    Real consumers are marked ``skipped_dependency_failed`` by the
    execution-time dependency gate; steps with no declared edge to the failure
    keep their slot in the queue and the failure denominator. The failed step
    itself is never used as a replan anchor.
    """

    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
        RunTransition,
    )

    calls: list[str] = []
    transitions: list[str] = []
    failed = {
        "step_id": "04",
        "status": "coder_failed",
        "planned_analysis_role": "auxiliary",
        "expected_outputs": ["table:04_result"],
    }
    state = RunExecutionState(
        remaining_steps=[
            AnalysisStep(
                step_id="04",
                intent="auxiliary analysis",
                planned_analysis_role="auxiliary",
                expected_outputs=["table:04_result"],
            ),
            AnalysisStep(step_id="05", intent="independent tail renderer"),
        ],
        executed_step_ids=set(),
        stop_on_failure=True,
    )

    def execute(step: AnalysisStep):
        calls.append(step.step_id)
        if step.step_id == "04":
            return failed
        return {"step_id": step.step_id, "status": "ok"}

    result = RunCoordinator().run_sequential(
        state=state,
        execute_step=execute,
        resolve_transition=lambda step, record, has_remaining: (
            transitions.append(step.step_id) or RunTransition.continue_run()
        ),
        resolve_run_halt=lambda step, record: None,
        apply_revised_plan=lambda plan, executed: [],
    )

    assert calls == ["04", "05"]
    assert transitions == ["05"]
    assert result.failed_step_ids == ["04"]
    assert result.stop_reason is None
    assert "remaining_steps_suppressed" not in failed


def test_auxiliary_failure_never_skips_a_run_level_halt() -> None:
    """Input-authority corruption halts even when an auxiliary step failed.

    Codex counterexample (2026-09-12): the continuation branch must not skip
    the host halt check. The real halt resolver runs after every executed
    step, so the independent tail is never scheduled once input authority is
    reported corrupt.
    """

    from easyicu.research_agent.execution.phase import _step_resolve_run_halt
    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
    )

    corruption = SimpleNamespace(corrupted=False, step_id=None)
    calls: list[str] = []
    failed = {
        "step_id": "aux",
        "status": "blocked_input_authority_mutation",
        "planned_analysis_role": "auxiliary",
    }

    def execute(step: AnalysisStep):
        calls.append(step.step_id)
        if step.step_id == "aux":
            corruption.corrupted = True
            corruption.step_id = "aux"
            return failed
        return {"step_id": step.step_id, "status": "ok", "planned_analysis_role": "auxiliary"}

    halt = partial(
        _step_resolve_run_halt,
        run_input_authority_state=corruption,
        emit_progress=lambda *args, **kwargs: None,
        run_id="synthetic",
        requested_stop_after_step_id=None,
        _replan_state={},
    )
    state = RunCoordinator().run_sequential(
        state=RunExecutionState(
            remaining_steps=[_step("aux"), _step("tail")],
            executed_step_ids=set(),
            stop_on_failure=True,
            stop_failure_roles=frozenset({"primary"}),
        ),
        execute_step=execute,
        resolve_transition=lambda step, record, has_remaining: pytest.fail(
            "a failed auxiliary step must not reach the replan transition"
        ),
        resolve_run_halt=halt,
        apply_revised_plan=lambda plan, executed: [],
    )

    assert calls == ["aux"]
    assert state.stop_reason == "input_authority_corrupted"


def test_failed_auxiliary_step_without_halt_resolver_only_asks_for_a_halt() -> None:
    """Compatibility path: one transition call, halt answers honoured only."""

    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
        RunTransition,
    )

    calls: list[str] = []
    asked: list[str] = []
    failed = {
        "step_id": "aux",
        "status": "execution_failed",
        "planned_analysis_role": "auxiliary",
    }

    def transition(step: AnalysisStep, record: dict, has_remaining: bool):
        asked.append(step.step_id)
        if record.get("status") != "ok":
            return RunTransition.replan(object())
        return RunTransition.continue_run()

    state = RunCoordinator().run_sequential(
        state=RunExecutionState(
            remaining_steps=[_step("aux"), _step("tail")],
            executed_step_ids=set(),
            stop_on_failure=True,
        ),
        execute_step=lambda step: calls.append(step.step_id) or (
            failed if step.step_id == "aux" else {"step_id": step.step_id, "status": "ok"}
        ),
        resolve_transition=transition,
        apply_revised_plan=lambda plan, executed: pytest.fail(
            "a discarded replan answer must not rebuild the queue"
        ),
    )

    assert calls == ["aux", "tail"]
    assert asked == ["aux", "tail"]
    assert state.failed_step_ids == ["aux"]
    assert state.stop_reason is None


def test_failed_auxiliary_step_without_halt_resolver_honours_a_halt() -> None:
    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
        RunTransition,
    )

    calls: list[str] = []
    failed = {
        "step_id": "aux",
        "status": "execution_failed",
        "planned_analysis_role": "auxiliary",
    }
    state = RunCoordinator().run_sequential(
        state=RunExecutionState(
            remaining_steps=[_step("aux"), _step("tail")],
            executed_step_ids=set(),
            stop_on_failure=True,
        ),
        execute_step=lambda step: calls.append(step.step_id) or failed,
        resolve_transition=lambda step, record, has_remaining: RunTransition.stop(
            "input_authority_corrupted"
        ),
        apply_revised_plan=lambda plan, executed: [],
    )

    assert calls == ["aux"]
    assert state.stop_reason == "input_authority_corrupted"


def test_sequential_primary_failure_still_suppresses_tail() -> None:
    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
    )

    calls: list[str] = []
    failed = {
        "step_id": "03",
        "status": "execution_failed",
        "planned_analysis_role": "primary",
    }
    state = RunExecutionState(
        remaining_steps=[
            AnalysisStep(
                step_id="03",
                intent="primary model",
                planned_analysis_role="primary",
            ),
            AnalysisStep(step_id="04", intent="tail renderer"),
        ],
        executed_step_ids=set(),
        stop_on_failure=True,
        stop_failure_roles=frozenset({"primary"}),
    )

    result = RunCoordinator().run_sequential(
        state=state,
        execute_step=lambda step: calls.append(step.step_id) or failed,
        resolve_transition=lambda step, record, has_remaining: pytest.fail(
            "a failed required role must not transition"
        ),
        apply_revised_plan=lambda plan, executed: pytest.fail(
            "a failed required role must not replan"
        ),
    )

    assert calls == ["03"]
    assert result.stop_reason == "required_step_failed:execution_failed"
    assert failed["remaining_steps_suppressed"] is True


def test_requested_stop_still_holds_when_that_step_fails() -> None:
    """A user-requested stop point is a queue command, not a replan anchor."""

    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
    )

    calls: list[str] = []
    failed = {
        "step_id": "04",
        "status": "execution_failed",
        "planned_analysis_role": "auxiliary",
    }
    state = RunExecutionState(
        remaining_steps=[_step("04"), _step("05")],
        executed_step_ids=set(),
        stop_on_failure=True,
        stop_after_step_id="04",
    )

    result = RunCoordinator().run_sequential(
        state=state,
        execute_step=lambda step: calls.append(step.step_id) or failed,
        resolve_transition=lambda step, record, has_remaining: pytest.fail(
            "a failed requested stop must not transition"
        ),
        apply_revised_plan=lambda plan, executed: pytest.fail(
            "a failed requested stop must not replan"
        ),
    )

    assert calls == ["04"]
    assert result.stop_reason == "requested_stop_after_step"


def test_sequential_fail_stop_can_target_declared_step_roles() -> None:
    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
        RunTransition,
    )

    coordinator = RunCoordinator()
    steps = [_step("aux"), _step("primary"), _step("later")]
    executed: list[str] = []

    def execute(step: AnalysisStep) -> dict[str, str]:
        executed.append(step.step_id)
        return {
            "step_id": step.step_id,
            "status": "error" if step.step_id in {"aux", "primary"} else "ok",
            "planned_analysis_role": (
                "primary" if step.step_id == "primary" else "auxiliary"
            ),
        }

    state = coordinator.run_sequential(
        state=RunExecutionState(
            remaining_steps=list(steps),
            executed_step_ids=set(),
            stop_failure_roles=frozenset({"primary"}),
        ),
        execute_step=execute,
        resolve_transition=lambda *_: RunTransition.continue_run(),
        apply_revised_plan=lambda *_: [],
    )

    assert executed == ["aux", "primary"]
    assert state.stop_reason == "required_step_failed:error"


def test_directed_replan_retries_current_step_without_two_plan_authorities() -> None:
    from easyicu.research_agent.execution.run_coordination import (
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
    from easyicu.research_agent.execution.run_coordination import (
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
    from easyicu.research_agent.execution.run_coordination import (
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


def test_sequential_worker_exception_stops_the_run_without_escaping() -> None:
    """A raising step ends the run here instead of unwinding past pipeline.run.

    This replaces an earlier test that asserted the exception propagated
    unchanged. Propagation is what killed fresh16: nothing above this call
    sealed a manifest, so a real run ended with a bare traceback and a run
    directory that never named the failing step. See
    ``test_step_exception_does_not_kill_the_run.py``.

    Both properties the old test protected are kept: a raised step still must
    not transition, and the exception itself is still surfaced -- now handed to
    the caller, which owns record shape, rather than thrown past it.
    """

    from easyicu.research_agent.execution.run_coordination import (
        RunCoordinator,
        RunExecutionState,
    )

    expected = RuntimeError("step failed")
    surfaced: list[BaseException] = []

    def fail(step: AnalysisStep):
        raise expected

    state = RunExecutionState(
        remaining_steps=[_step("01")],
        executed_step_ids=set(),
    )
    RunCoordinator().run_sequential(
        state=state,
        execute_step=fail,
        resolve_transition=lambda step, record, has_remaining: pytest.fail(
            "failed step must not transition"
        ),
        apply_revised_plan=lambda plan, executed: [],
        on_step_exception=lambda step, error: surfaced.append(error),
    )

    assert surfaced == [expected]
    assert surfaced[0] is expected
    assert state.stop_reason == "step_raised:01:RuntimeError"


def test_parallel_workers_use_supplied_context_submitter_and_report_errors() -> None:
    from easyicu.research_agent.execution.phase import _submit_in_current_context
    from easyicu.research_agent.execution.run_coordination import RunCoordinator

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
        on_worker_error=lambda step, error: errors.append(error),
    )

    assert sorted(observed) == ["01:bound", "02:bound"]
    assert errors == [expected]


def test_run_coordinator_is_science_neutral_and_pipeline_owns_transitions() -> None:
    import easyicu.research_agent.execution.phase as pipeline_execute
    import easyicu.research_agent.execution.run_coordination as run_coordination

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

    phase_source = (
        inspect.getsource(pipeline_execute.run_execute_phase)
        + inspect.getsource(pipeline_execute._step_resolve_run_halt)
        + inspect.getsource(pipeline_execute._step_resolve_run_transition)
        + inspect.getsource(pipeline_execute._step_audit_final_figures)
    )
    assert "while remaining_steps:" not in phase_source
    assert phase_source.count("run_coordinator.run_sequential(") == 1
    assert phase_source.count("run_coordinator.run_parallel(") == 1
    assert "_resolve_run_halt = functools.partial(" in phase_source
    assert "resolve_run_halt=_resolve_run_halt," in phase_source
    corruption = phase_source.index("if run_input_authority_state.corrupted:")
    requested_stop = phase_source.index(
        "if step.step_id == requested_stop_after_step_id:", corruption
    )
    directed = phase_source.index("directed_plan = _maybe_directed_model_replan(")
    ordinary = phase_source.index(
        "if _successful_run_transition_requests_replan(", directed
    )
    assert corruption < requested_stop < directed < ordinary
    assert (
        "stop_on_failure=(pipeline._submission_profile_name is not None)"
        in phase_source
    )
    assert 'stop_failure_roles=frozenset({"primary"})' in phase_source
    assert "or pipeline._submission_profile_name is not None" in phase_source
    assert "def current_plan() -> AnalysisPlan:" in phase_source
    assert phase_source.count("plan_supplier=current_plan") == 2
    assert (
        "if pipeline._enable_visual_qa and requested_stop_after_step_id is None:"
        in phase_source
    )


def test_step_checkpoint_is_also_a_write_phase_boundary() -> None:
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    source = inspect.getsource(ResearchAgentPipeline.run)
    assert "stop_after_analysis = bool(stop_after_analysis or stop_after_step_id)" in source


def test_execute_transition_reads_the_live_replanned_plan() -> None:
    from easyicu.research_agent.execution.phase import _step_resolve_run_transition

    live = {"plan": "revision_3"}
    observed: list[str] = []

    def maybe_replan(**kwargs):
        observed.append(kwargs["current_plan"])
        return kwargs["current_plan"]

    transition = _step_resolve_run_transition(
        _step("01"),
        {"status": "ok", "generation_mode": "llm"},
        True,
        _maybe_directed_model_replan=lambda **kwargs: None,
        _replan_state={},
        pipeline=SimpleNamespace(
            _enable_replanning=True,
            _planner_strategy="progressive_v2",
        ),
        _maybe_replan=maybe_replan,
        plan_supplier=lambda: live["plan"],
        probe_summary=None,
        per_step_records=[],
    )

    assert observed == ["revision_3"]
    assert transition.kind == "replan"
    assert transition.revised_plan == "revision_3"
