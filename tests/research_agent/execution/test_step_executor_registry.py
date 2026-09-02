from __future__ import annotations

import pytest

from easyicu.research_agent.execution.step_executor_registry import (
    StepExecutor,
    StepExecutorContext,
    StepExecutorRegistry,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _context() -> StepExecutorContext:
    step = AnalysisStep(
        step_id="01_table",
        intent="Build the declared table",
        method="descriptive",
        inputs=[],
        expected_outputs=["table:summary"],
    )
    return StepExecutorContext(
        step=step,
        plan=AnalysisPlan(research_question="Describe the cohort", steps=[step]),
    )


def _executor(key: str, owns: bool = True) -> StepExecutor:
    return StepExecutor(
        key=key,
        owns=lambda _context: owns,
        render=lambda _context: "print('owned')",
        analysis_kind=key,
        selection_reason=f"{key}_contract",
        progress_message=f"Using {key}",
        consumed_input_keys=lambda _context: (),
    )


def test_registry_calls_every_owner_through_one_context_to_selection_interface() -> (
    None
):
    registry = StepExecutorRegistry()
    registry.declare(_executor("first", owns=False))
    registry.declare(_executor("second"))
    trace = []

    selected = registry.select(_context(), trace=trace)

    assert selected is not None
    assert selected.analysis_kind == "second"
    assert selected.code == "print('owned')"
    assert [(row.analysis_kind, row.outcome) for row in trace] == [
        ("first", "contract_declined"),
        ("second", "selected"),
    ]


def test_registry_refuses_duplicate_executor_ownership() -> None:
    registry = StepExecutorRegistry()
    registry.declare(_executor("summary"))

    with pytest.raises(ValueError, match="already declared: summary"):
        registry.declare(_executor("summary"))


def test_registry_refuses_an_empty_executor_key() -> None:
    registry = StepExecutorRegistry()

    with pytest.raises(ValueError, match="key is required"):
        registry.declare(_executor("  "))
