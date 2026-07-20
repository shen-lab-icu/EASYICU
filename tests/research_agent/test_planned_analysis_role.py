"""Planner-owned analysis-role contract tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.core import (
    PlannerAgent,
    _build_planner_user_prompt,
    _normalise_plan_payload,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    PlannedAnalysisRole,
    ResearchContext,
    StepRecord,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate the prespecified study result.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )


def _raw_plan(*, include_role: bool, role: str = "primary") -> str:
    step = {
        "step_id": "01_model",
        "intent": "Estimate the prespecified result.",
        "inputs": [],
        "expected_outputs": ["table:estimate"],
        "method": "descriptive",
    }
    if include_role:
        step["planned_analysis_role"] = role
    return json.dumps(
        {
            "research_question": "Estimate the prespecified study result.",
            "steps": [step],
            "rationale": "Use the declared analysis plan.",
        }
    )


def test_host_constructed_step_defaults_to_auxiliary() -> None:
    step = AnalysisStep(step_id="01_prepare", intent="Prepare typed inputs.")
    assert step.planned_analysis_role == "auxiliary"


def test_planned_analysis_role_is_part_of_the_public_schema_api() -> None:
    from easyicu import research_agent

    assert research_agent.PlannedAnalysisRole is PlannedAnalysisRole


@pytest.mark.parametrize("role", ["primary", "secondary", "sensitivity", "auxiliary"])
def test_analysis_step_accepts_each_typed_role(role: str) -> None:
    step = AnalysisStep(
        step_id="01_step",
        intent="Run the planned step.",
        planned_analysis_role=role,
    )
    assert step.planned_analysis_role == role


def test_analysis_step_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError, match="planned_analysis_role"):
        AnalysisStep(
            step_id="01_step",
            intent="Run the planned step.",
            planned_analysis_role="headline",
        )


def test_analysis_plan_allows_zero_primary_steps() -> None:
    plan = AnalysisPlan(
        research_question="Prepare the research package.",
        steps=[AnalysisStep(step_id="01_prepare", intent="Prepare typed inputs.")],
    )
    assert not [step for step in plan.steps if step.planned_analysis_role == "primary"]


def test_analysis_plan_rejects_multiple_primary_steps() -> None:
    with pytest.raises(ValidationError, match="at most one step"):
        AnalysisPlan(
            research_question="Estimate one headline result.",
            steps=[
                AnalysisStep(
                    step_id="01_model",
                    intent="Estimate the headline result.",
                    planned_analysis_role="primary",
                ),
                AnalysisStep(
                    step_id="02_model",
                    intent="Estimate another headline result.",
                    planned_analysis_role="primary",
                ),
            ],
        )


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        ["forest_plot.svg"],
        ["figure:forest_plot"],
        ["log:audit"],
        ["report:manuscript"],
        ["code:analysis"],
        ["test:quality"],
    ],
)
def test_analysis_plan_rejects_primary_without_typed_scientific_result(
    outputs: list[str],
) -> None:
    with pytest.raises(ValidationError, match="typed, non-rendering"):
        AnalysisPlan(
            research_question="Estimate one headline result.",
            steps=[
                AnalysisStep(
                    step_id="01_primary",
                    intent="Produce only a renderer or support artifact.",
                    planned_analysis_role="primary",
                    expected_outputs=outputs,
                )
            ],
        )


def test_step_record_role_is_typed_but_historical_none_remains_readable() -> None:
    historical = StepRecord(step_id="01_model", intent="Run model.")
    current = StepRecord(
        step_id="01_model",
        intent="Run model.",
        planned_analysis_role="primary",
    )
    assert historical.planned_analysis_role is None
    assert current.planned_analysis_role == "primary"
    with pytest.raises(ValidationError, match="planned_analysis_role"):
        StepRecord(
            step_id="01_model",
            intent="Run model.",
            planned_analysis_role="headline",
        )


def test_plan_normalizer_preserves_planned_analysis_role() -> None:
    normalized, dropped = _normalise_plan_payload(
        json.loads(_raw_plan(include_role=True, role="secondary"))
    )
    assert normalized["steps"][0]["planned_analysis_role"] == "secondary"
    assert not dropped["steps"]


def test_planner_parse_requires_explicit_role_despite_schema_default() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}
    with pytest.raises(ValueError, match="must explicitly declare"):
        planner._parse(_raw_plan(include_role=False), _context())


def test_planner_parse_preserves_explicit_role() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}
    plan = planner._parse(_raw_plan(include_role=True, role="sensitivity"), _context())
    assert plan.steps[0].planned_analysis_role == "sensitivity"


def test_planner_run_retries_missing_role_and_feedback_names_contract() -> None:
    class _RetryingLLM:
        name = "retrying"

        def __init__(self) -> None:
            self.calls: list[list[object]] = []

        def complete(self, messages, **kwargs):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                return _raw_plan(include_role=False)
            return _raw_plan(include_role=True)

    llm = _RetryingLLM()
    plan = PlannerAgent(llm).run(_context())

    assert plan.steps[0].planned_analysis_role == "primary"
    assert len(llm.calls) == 2
    retry_feedback = llm.calls[1][-1].content
    assert "planned_analysis_role" in retry_feedback


def test_planner_prompt_defines_required_role_without_case_specific_terms() -> None:
    prompt = _build_planner_user_prompt(_context())
    assert "Every step MUST explicitly declare `planned_analysis_role`" in prompt
    assert "at most one step may be primary" in prompt
    assert '"planned_analysis_role": "auxiliary"' in prompt
    assert "exactly one materialised closed primary-cohort product" in prompt
    assert "`artifact:cohort_defined` is not a cohort dataset" in prompt
