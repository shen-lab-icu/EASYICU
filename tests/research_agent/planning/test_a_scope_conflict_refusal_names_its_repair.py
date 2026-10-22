"""A broader-cohort risk table that binds the primary result is told what to drop.

The absolute-risk table's population scope is sealed by the outline; the step
can change only the products it binds.  Hosted pilot: a step sealed to
``analysis_cohort`` that followed the primary step kept binding the primary
result through three repairs, told only that the two "cannot" go together.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.planning.progressive_compiler import (
    ProgressivePlanCompileError,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


def _risk_after_the_primary(**changes) -> dict:
    payload = json.loads(json.dumps(_payload()))
    step = next(item for item in payload["steps"] if item["step_id"] == "03_distribution")
    payload["steps"].remove(step)
    step.update({
        "step_id": "03_absolute_risk_context",
        "module_id": "absolute_risk_context",
        "objective": "Describe exposure prevalence and absolute outcome risk.",
        "event_level_index": None,
        "reference_exposure_level_index": None,
        "comparison_exposure_level_index": None,
        "denominator_policy": None,
        "missing_exposure_policy": None,
        "missing_outcome_policy": None,
        "confidence_level": None,
        "population_scope": "analysis_cohort",
        "depends_on": ["05_primary"],
        "product_inputs": [
            {"producer_step_id": "05_primary", "product_id": "table:adjusted_association_estimates"}
        ],
        **changes,
    })
    index = next(i for i, item in enumerate(payload["steps"]) if item["step_id"] == "05_primary")
    payload["steps"].insert(index + 1, step)
    figure = next(item for item in payload["steps"] if item["step_id"] == "07_figure")
    figure["depends_on"][0] = "03_absolute_risk_context"
    figure["product_inputs"][0] = {
        "producer_step_id": "03_absolute_risk_context",
        "product_id": "table:absolute_risk_context",
    }
    return payload


def test_the_refusal_names_the_product_to_drop() -> None:
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(_risk_after_the_primary()),
            context=_context(),
        )

    assert caught.value.reason_code == "progressive_population_scope_conflict"
    assert "remove table:adjusted_association_estimates from product_inputs" in str(caught.value)


def test_following_the_primary_without_binding_it_compiles() -> None:
    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_risk_after_the_primary(product_inputs=[])),
        context=_context(),
    )

    risk = next(step for step in plan.steps if step.step_id == "03_absolute_risk_context")
    assert "table:adjusted_association_estimates" not in risk.inputs
