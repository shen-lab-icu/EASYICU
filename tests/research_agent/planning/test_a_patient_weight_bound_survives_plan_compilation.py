"""A reviewed interval choice is preserved, not invented by compilation."""

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.progressive_compiler import (
    _compile_distribution, compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError, ProgressivePlanSkeleton, ProgressiveSkeletonStep,
)
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


def _payload_with_bound():
    payload = _payload()
    payload["steps"][2].update(
        repeated_unit_interval_method="patient_weighted_hoeffding",
        patient_weighted_interval_assumptions={
            "target": "conditional_expected_stay_proportion",
            "independence": "independent_patient_clusters_conditional_on_denominator",
            "weights": "fixed_observed_patient_stay_weights",
        },
    )
    return payload


def test_public_compiler_preserves_the_explicit_method_and_conditional_target():
    skeleton = ProgressivePlanSkeleton.model_validate(_payload_with_bound())
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=_context())
    spec = plan.steps[2].exposure_outcome_distribution_spec
    assert spec.repeated_unit_interval_method == "patient_weighted_hoeffding"
    assert spec.patient_weighted_interval_assumptions.target == "conditional_expected_stay_proportion"


def test_silence_retains_the_existing_wald_design():
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_payload()), context=_context(),
    )
    spec = plan.steps[2].exposure_outcome_distribution_spec
    assert spec.repeated_unit_interval_method == "patient_cluster_robust_wald"
    assert spec.patient_weighted_interval_assumptions is None


def test_counts_only_authority_does_not_silently_drop_a_requested_interval():
    step = ProgressiveSkeletonStep.model_validate(_payload_with_bound()["steps"][2])
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile_distribution(
            variables={item.name: item for item in _context().variables},
            step=step, step_index=2, counts_only=True,
        )
    assert caught.value.reason_code == "progressive_distribution_counts_only_interval_conflict"


def test_other_modules_cannot_accept_and_ignore_patient_interval_fields():
    payload = _payload_with_bound()["steps"][0]
    payload.update(repeated_unit_interval_method="patient_cluster_robust_wald")
    with pytest.raises(ValidationError, match="distribution modules"):
        ProgressiveSkeletonStep.model_validate(payload)


def test_bound_choices_are_offered_only_to_the_current_distribution_coordinate():
    from easyicu.research_agent.agents.progressive_payload import progressive_step_materialization_request
    from easyicu.research_agent.agents.progressive_prompt_contracts import step_materialization_shape_contract
    from easyicu.research_agent.planning.progressive_contract import ProgressiveOutlineStep

    for module in ("exposure_outcome_distribution", "table_one"):
        outline = ProgressiveOutlineStep(
            step_id="02_result", planned_analysis_role="auxiliary", module_id=module,
            objective="Report the prespecified distribution with its reviewed uncertainty.",
            depends_on=["01_cohort"], variable_names=["exposure_flag", "outcome_flag"],
        )
        request = progressive_step_materialization_request(
            outline_step=outline, outline_step_sha256="a"*64,
            variable_names=["exposure_flag", "outcome_flag"], scientific_action_ids=[],
            available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
        )
        schema = json.loads(request.schema_json)
        text = json.dumps(schema)
        assert ("patient_weighted_hoeffding" in text) == (module == "exposure_outcome_distribution")
        assert ("PatientWeightedIntervalAssumptions" in schema["$defs"]) == (module == "exposure_outcome_distribution")
        shape = step_materialization_shape_contract(outline_step=outline, outline_step_sha256="a"*64)
        assert ("patient_weighted_interval_assumptions" in shape) == (module == "exposure_outcome_distribution")
