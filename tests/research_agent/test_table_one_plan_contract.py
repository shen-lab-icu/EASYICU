from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.core import PlannerAgent, _normalise_plan_payload
from easyicu.research_agent.providers.prompts import load_prompt_pack
from easyicu.research_agent.research_context.prompt_scope import coder_guide_for_step
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Describe the cohort by the Planner-selected group.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )


def _step(*, include_spec: bool) -> dict:
    step = {
        "step_id": "02_table_one",
        "planned_analysis_role": "auxiliary",
        "intent": "Produce the grouped baseline table.",
        "inputs": ["arm", "age"],
        "expected_outputs": ["table:table_one"],
        "method": "table_one",
    }
    if include_spec:
        step["table_one_spec"] = {
            "group_by": "arm",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                }
            ],
        }
    return step


def _raw(*, include_spec: bool) -> str:
    return json.dumps(
        {
            "research_question": "Describe the cohort.",
            "steps": [_step(include_spec=include_spec)],
            "rationale": "Use the declared grouped table design.",
        }
    )


def test_fresh_planner_table_one_requires_typed_design() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    with pytest.raises(ValueError, match="must declare table_one_spec"):
        planner._parse(_raw(include_spec=False), _context())


def test_fresh_planner_table_one_preserves_typed_design() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    plan = planner._parse(_raw(include_spec=True), _context())
    assert plan.steps[0].table_one_spec is not None
    assert plan.steps[0].table_one_spec.group_by == "arm"
    assert plan.steps[0].table_one_spec.variables[0].test == ("mann_whitney_or_kruskal")


def test_archival_analysis_step_remains_readable_without_new_optional_spec() -> None:
    step = AnalysisStep.model_validate(_step(include_spec=False))
    assert step.table_one_spec is None


def test_table_one_spec_must_bind_only_explicit_step_inputs() -> None:
    payload = _step(include_spec=True)
    payload["inputs"] = ["age"]
    with pytest.raises(ValidationError, match="must be explicit step inputs"):
        AnalysisStep.model_validate(payload)


def test_plan_normalizer_keeps_only_closed_table_one_schema() -> None:
    payload = {
        "research_question": "Describe the cohort.",
        "steps": [_step(include_spec=True)],
    }
    payload["steps"][0]["table_one_spec"]["invented_policy"] = "ignore"
    payload["steps"][0]["table_one_spec"]["variables"][0]["invented"] = True
    normalized, dropped = _normalise_plan_payload(payload)
    spec = normalized["steps"][0]["table_one_spec"]
    assert "invented_policy" not in spec
    assert "invented" not in spec["variables"][0]
    assert dropped["table_one_spec"] == [
        "step[0]:invented_policy",
        "step[0].variables[0]:invented",
    ]


def test_table_one_sdk_guidance_is_only_added_for_typed_table_one() -> None:
    typed = AnalysisStep.model_validate(_step(include_spec=True))
    legacy = AnalysisStep.model_validate(_step(include_spec=False))
    full = load_prompt_pack()["coder"]

    assert "build_grouped_table_one" in coder_guide_for_step(full, typed)
    assert "build_grouped_table_one" not in coder_guide_for_step(full, legacy)
