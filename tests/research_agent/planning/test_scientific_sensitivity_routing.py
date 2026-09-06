"""Scientific action ownership must precede binary sensitivity inheritance."""

from copy import deepcopy

import pytest

from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanSkeleton,
)
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


def _survival_sensitivity_payload() -> dict:
    payload = _payload()
    payload["analysis_type"] = "survival"
    payload["robustness_intents"] = []
    primary = deepcopy(payload["steps"][5])
    primary.update(
        step_id="05_primary", planned_analysis_role="primary",
        scientific_action_id="time_to_event.cox_hr", custom_method="cox_ph",
        depends_on=["01_cohort"], product_inputs=[], sensitivity_spec_ids=[],
        outputs=[{"product_id": "table:cox_results", "semantic_role": "custom"}],
    )
    sensitivity = deepcopy(payload["steps"][5])
    sensitivity.update(
        scientific_action_id="time_to_event.rmst", custom_method="rmst",
        product_inputs=[{
            "producer_step_id": "05_primary", "product_id": "table:cox_results",
        }],
    )
    payload["steps"] = [payload["steps"][0], primary, sensitivity]
    return payload


def test_survival_sensitivity_keeps_its_action_without_binary_authority() -> None:
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_survival_sensitivity_payload()),
        context=_context(),
    )
    step = plan.steps[-1]

    assert step.scientific_action_id == "time_to_event.rmst"
    assert step.scientific_capability is None
    assert "table:cox_results" in step.inputs
    assert "table:adjusted_association_estimates" not in step.inputs


@pytest.mark.parametrize("action", [None, "time_to_event.invented"])
def test_unowned_sensitivity_does_not_bypass_action_or_parent_validation(action) -> None:
    payload = _survival_sensitivity_payload()
    payload["steps"][-1]["scientific_action_id"] = action

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context(),
        )

    assert caught.value.reason_code == (
        "progressive_association_sensitivity_parent_invalid"
        if action is None else "progressive_scientific_action_invalid"
    )


def test_binary_scientific_sensitivity_retains_its_closed_owner() -> None:
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_payload()), context=_context(),
    )
    step = next(item for item in plan.steps if item.step_id == "06_sensitivity")

    assert step.scientific_capability == "association_freeform_v1"


def test_survival_action_still_rejects_an_invented_binary_parent_product() -> None:
    payload = _survival_sensitivity_payload()
    payload["steps"][-1]["product_inputs"][0]["product_id"] = (
        "table:adjusted_association_estimates"
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context(),
        )

    assert caught.value.reason_code == "progressive_product_reference_mismatch"
