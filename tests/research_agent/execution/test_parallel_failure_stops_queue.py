"""Parallel work must stop launching after an unexpected worker failure."""

from threading import Barrier

import pytest

from easyicu.research_agent.execution.run_coordination import RunCoordinator
from easyicu.research_agent.schema import AnalysisStep


def _steps(count):
    return [AnalysisStep(step_id=str(i), intent="synthetic") for i in range(count)]


def _run(steps, workers, execute, errors):
    return RunCoordinator().run_parallel(
        steps=steps,
        max_workers=workers,
        execute_step=execute,
        submit_step=lambda pool, fn, step: pool.submit(fn, step),
        on_worker_error=lambda step, error: errors.append((step.step_id, error)),
    )


def test_unexpected_failure_prevents_queued_workers_from_starting():
    attempted, errors = [], []
    failure = ValueError("broken worker invariant")

    def execute(step):
        attempted.append(step.step_id)
        raise failure

    _run(_steps(8), 1, execute, errors)
    assert attempted == ["0"]
    assert errors == [("0", failure)]


def test_already_running_worker_finishes_and_failure_is_recorded():
    barrier = Barrier(2, timeout=5)
    completed, errors = [], []

    def execute(step):
        barrier.wait()
        if step.step_id == "0":
            raise ValueError("broken worker invariant")
        completed.append(step.step_id)

    _run(_steps(2), 2, execute, errors)
    assert completed == ["1"]
    assert len(errors) == 1 and errors[0][0] == "0"


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(2)])
def test_process_interrupt_is_recorded_and_propagated(error):
    attempted, errors = [], []

    def execute(step):
        attempted.append(step.step_id)
        raise error

    with pytest.raises(type(error)) as caught:
        _run(_steps(8), 1, execute, errors)
    assert caught.value is error
    assert attempted == ["0"]
    assert errors == [("0", error)]
