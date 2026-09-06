"""Fixed host estimators must not masquerade as another scientific action."""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _outline_payload,
    _payload,
)


def _family_payload(*, outline: bool, family: str = "survival") -> dict:
    payload = _outline_payload() if outline else _payload()
    payload["analysis_type"] = family
    if outline:
        for candidate in payload["design_selection"]["candidates"]:
            candidate["analysis_type"] = family
    next(
        step for step in payload["steps"] if step["module_id"] == "adjusted_association"
    )["scientific_action_id"] = None
    return payload


def _validate_outline(payload: dict) -> None:
    ProgressivePlannerAgent._validate_outline_authority(
        ProgressivePlanOutline.model_validate(payload),
        analysis_types=(payload["analysis_type"],),
        variable_names=("exposure_flag", "outcome_flag", "age_years", "sex_code"),
        allowed_literature_citation_keys=(),
    )


@pytest.mark.parametrize("family", ("survival", "prediction", "trajectory_clustering"))
def test_outline_null_action_cannot_change_the_primary_estimator_family(family) -> None:
    payload = _family_payload(outline=True, family=family)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate_outline(payload)

    assert caught.value.reason_code == "progressive_outline_primary_module_family_mismatch"
    assert caught.value.step_id == "05_primary"
    assert caught.value.path == "module_id"
    assert caught.value.details["findings"][0]["module_id"] == "adjusted_association"


def test_compiler_rejects_saved_null_action_primary_from_another_family() -> None:
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(_family_payload(outline=False)),
            context=_context(),
        )

    assert caught.value.reason_code == "progressive_primary_module_family_mismatch"
    assert caught.value.step_id == "05_primary"
    assert caught.value.path == "module_id"


def test_outline_counts_module_cannot_claim_kaplan_meier() -> None:
    payload = _family_payload(outline=True)
    primary = next(step for step in payload["steps"] if step["step_id"] == "05_primary")
    primary.update(module_id="custom_analysis", scientific_action_id="time_to_event.cox_hr")
    risk = next(step for step in payload["steps"] if step["step_id"] == "03_distribution")
    risk.update(module_id="absolute_risk_context", scientific_action_id="time_to_event.km_logrank")

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate_outline(payload)

    assert caught.value.reason_code == "progressive_outline_action_module_mismatch"
    assert caught.value.step_id == risk["step_id"]
    assert caught.value.details["findings"] == [{
        "module_id": "absolute_risk_context",
        "scientific_action_id": "time_to_event.km_logrank",
        "compatible_action_ids": [],
    }]


def test_compiler_counts_module_cannot_claim_kaplan_meier() -> None:
    payload = _family_payload(outline=False)
    risk = next(step for step in payload["steps"] if step["step_id"] == "03_distribution")
    risk.update(module_id="absolute_risk_context", scientific_action_id="time_to_event.km_logrank")
    figure = next(step for step in payload["steps"] if step["step_id"] == "07_figure")
    figure["product_inputs"][0]["product_id"] = "table:absolute_risk_context"

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context()
        )

    assert caught.value.reason_code == "progressive_compile_batch_invalid"
    assert caught.value.step_id == risk["step_id"]
    assert {
        (finding["step_id"], finding["reason_code"])
        for finding in caught.value.details["findings"]
    } == {
        (risk["step_id"], "progressive_action_module_mismatch"),
        ("05_primary", "progressive_primary_module_family_mismatch"),
    }


def test_compiler_binary_module_cannot_claim_cox() -> None:
    payload = _family_payload(outline=False)
    primary = next(step for step in payload["steps"] if step["step_id"] == "05_primary")
    primary["scientific_action_id"] = "time_to_event.cox_hr"

    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context()
        )

    assert caught.value.reason_code == "progressive_action_module_mismatch"
    assert caught.value.step_id == primary["step_id"]


def test_registered_custom_survival_actions_remain_available_in_the_outline() -> None:
    payload = _family_payload(outline=True)
    primary = next(step for step in payload["steps"] if step["step_id"] == "05_primary")
    primary.update(module_id="custom_analysis", scientific_action_id="time_to_event.cox_hr")
    risk = next(step for step in payload["steps"] if step["step_id"] == "03_distribution")
    risk.update(module_id="custom_analysis", scientific_action_id="time_to_event.km_logrank")
    payload["steps"].remove(risk)
    risk["depends_on"] = [primary["step_id"]]
    payload["steps"].insert(payload["steps"].index(primary) + 1, risk)

    _validate_outline(payload)


def test_legacy_null_action_primary_stays_valid_in_its_association_family() -> None:
    payload = _family_payload(outline=False, family="association_study")
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context()
    )

    assert next(step for step in plan.steps if step.step_id == "05_primary").method == (
        "adjusted_association_models"
    )
