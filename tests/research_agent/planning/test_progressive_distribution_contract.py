"""Distribution contract failures stay repairable at the current plan step."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_compiler import (
    _compile_distribution,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanSkeleton,
    ProgressiveSkeletonStep,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
    _payload,
)


@pytest.mark.parametrize("counts_only", [False, True])
@pytest.mark.parametrize(
    ("denominator", "missing_policy", "required_denominator"),
    [
        ("all_declared_rows", "exclude_from_denominator", "observed_outcome_rows"),
        (
            "observed_outcome_rows",
            "structural_absence_is_non_event",
            "all_declared_rows",
        ),
    ],
)
def test_invalid_distribution_policy_retains_owner_and_current_step(
    counts_only, denominator, missing_policy, required_denominator
):
    payload = _payload()["steps"][2]
    payload.update(
        denominator_policy=denominator, missing_outcome_policy=missing_policy
    )
    step = ProgressiveSkeletonStep.model_validate(payload)
    before = step.model_dump(mode="json")
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile_distribution(
            variables={item.name: item for item in _context().variables},
            step=step,
            step_index=2,
            counts_only=counts_only,
        )

    error = caught.value
    assert error.reason_code == "progressive_distribution_spec_invalid"
    assert error.step_id == "03_distribution"
    assert error.step_index == 2
    assert error.path == "exposure_outcome_distribution"
    assert required_denominator in error.details["message"]
    assert missing_policy in error.details["message"]
    assert isinstance(error.__cause__, ValidationError)
    assert step.model_dump(mode="json") == before


@pytest.mark.parametrize("counts_only", [False, True])
def test_distribution_contract_is_reported_through_full_compiler_preflight(counts_only):
    payload = _payload()
    payload["steps"][2]["missing_outcome_policy"] = "exclude_from_denominator"
    context = _context()
    if counts_only:
        payload.update(analysis_type="descriptive_epidemiology", robustness_intents=[])
        payload["steps"] = payload["steps"][:3]
        context = context.model_copy(update={
            "cohort": context.cohort.model_copy(update={
                "provenance": {"analysis_unit": "icu_stay"}, "n_patients": None,
            }),
        })
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context
        )
    assert caught.value.reason_code == "progressive_distribution_spec_invalid"
    assert caught.value.details["findings"][0]["step_id"] == "03_distribution"


def test_distribution_policy_repair_reuses_compiled_prefix_and_sealed_outline():
    materializations = _materialization_payloads()
    invalid = copy.deepcopy(materializations[2])
    invalid["step"]["missing_outcome_policy"] = "exclude_from_denominator"
    corrected = copy.deepcopy(invalid)
    corrected["step"]["denominator_policy"] = "observed_outcome_rows"
    responses = [
        _outline_payload(),
        _foundation_payload(),
        *materializations[:2],
        invalid,
        corrected,
        *materializations[3:],
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    attempt = ProgressivePlannerAgent(llm).run_attempt(_context())

    plan = attempt.output
    assert len(llm.calls) == 10
    assert len(plan.steps) == 7
    spec = plan.steps[2].exposure_outcome_distribution_spec
    assert spec.denominator_policy == "observed_outcome_rows"
    assert spec.missing_outcome_policy == "exclude_from_denominator"
    retry_prompt = llm.calls[5][0][-1].content
    assert "progressive_distribution_spec_invalid" in retry_prompt
    assert "observed_outcome_rows" in retry_prompt
    assert attempt.facts.prompt_metrics["compile_revision_count"] == 1
    assert [item.step.step_id for item in attempt.facts.materializations[:2]] == [
        "01_cohort", "02_table_one",
    ]
