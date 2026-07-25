from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import pytest


class _Runner:
    def __init__(self, events: list[object], *, manages_output_cleanup: bool) -> None:
        self.events = events
        self.manages_output_cleanup = manages_output_cleanup
        self.result = object()

    def run(self, **kwargs):
        self.events.append(("run", kwargs))
        return self.result


def _request(tmp_path: Path):
    from easyicu.research_agent.execution.step_execution import (
        LockedStepExecutionRequest,
    )

    return LockedStepExecutionRequest(
        step_id="02_model",
        code="print('locked')\n",
        resolved_inputs_path=tmp_path / "resolved_inputs.json",
        output_dir=tmp_path / "steps" / "02_model" / "outputs",
    )


def test_unmanaged_runner_is_cleared_before_exact_single_execution(tmp_path) -> None:
    from easyicu.research_agent.execution.step_execution import StepExecutor

    events: list[object] = []
    runner = _Runner(events, manages_output_cleanup=False)
    request = _request(tmp_path)
    executor = StepExecutor(
        clear_output_dir=lambda path: events.append(("clear", path))
    )

    result = executor.execute(runner=runner, request=request)

    assert result is runner.result
    assert events == [
        ("clear", request.output_dir),
        (
            "run",
            {
                "step_id": request.step_id,
                "code": request.code,
                "resolved_inputs_path": request.resolved_inputs_path,
            },
        ),
    ]


def test_runner_owned_cleanup_is_not_preempted(tmp_path) -> None:
    from easyicu.research_agent.execution.step_execution import StepExecutor

    events: list[object] = []
    runner = _Runner(events, manages_output_cleanup=True)
    executor = StepExecutor(
        clear_output_dir=lambda path: events.append(("clear", path))
    )

    result = executor.execute(runner=runner, request=_request(tmp_path))

    assert result is runner.result
    assert [event[0] for event in events] == ["run"]


def test_clear_failure_propagates_without_running(tmp_path) -> None:
    from easyicu.research_agent.execution.step_execution import StepExecutor

    expected = RuntimeError("clear failed")
    runner = _Runner([], manages_output_cleanup=False)

    def fail_clear(path: Path) -> None:
        raise expected

    with pytest.raises(RuntimeError) as raised:
        StepExecutor(clear_output_dir=fail_clear).execute(
            runner=runner,
            request=_request(tmp_path),
        )

    assert raised.value is expected
    assert runner.events == []


def test_runner_failure_propagates_without_wrapping(tmp_path) -> None:
    from easyicu.research_agent.execution.step_execution import StepExecutor

    expected = LookupError("runner failed")

    class FailingRunner:
        manages_output_cleanup = True

        def run(self, **kwargs):
            raise expected

    with pytest.raises(LookupError) as raised:
        StepExecutor(clear_output_dir=lambda path: None).execute(
            runner=FailingRunner(),
            request=_request(tmp_path),
        )

    assert raised.value is expected


def test_step_executor_is_a_mechanical_single_call_boundary() -> None:
    from easyicu.research_agent.execution import step_execution

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(step_execution.StepExecutor.execute))
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "runner"
        and node.func.attr == "run"
    ]
    assert len(calls) == 1
    assert [keyword.arg for keyword in calls[0].keywords] == [
        "step_id",
        "code",
        "resolved_inputs_path",
    ]
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))

    module_source = inspect.getsource(step_execution)
    for forbidden in (
        "AnalysisStep",
        "ResearchContext",
        "validators",
        "EvidenceStore",
        "RepairCoordinator",
        "StepAuthorityCapsule",
        "pipeline_execute",
    ):
        assert forbidden not in module_source


def test_pipeline_routes_only_fresh_sandbox_calls_through_step_executor() -> None:
    import easyicu.research_agent.execution.phase as pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    assert "execution_runner.run(" not in source
    assert source.count("step_executor.execute(") == 1
    assert source.index("if replay_execution is None:") < source.index(
        "step_executor.execute("
    )
    assert source.index("step_executor.execute(") < source.index(
        "_execution_input_authority_integrity_finding("
    )
