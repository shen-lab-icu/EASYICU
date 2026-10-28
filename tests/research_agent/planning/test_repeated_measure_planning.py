"""Repeated outcomes retain their reviewed time and death-dropout decisions."""

from easyicu.research_agent.method_skills import method_skill_catalog
from easyicu.research_agent.planning.analysis_types import infer_analysis_type
import json

import pytest

from easyicu.research_agent.contracts.trajectory_design import REPEATED_OUTPUTS, repeated_measure_method
from easyicu.research_agent.planning.capability_registry import resolve_primary_capability
from easyicu.research_agent.contracts.trajectory_design import canonical_repeated_measure_spec
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review, render_plan_scientific_guardrails
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton, ProgressiveSkeletonStep
from easyicu.research_agent.planning.scientific_action_catalog import scientific_actions_for_analysis_type
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences


def test_change_over_time_routes_to_repeated_measures():
    context = ResearchContext(
        research_question="Estimate the change over time in daily synthetic_marker during ICU days 0 to 6.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
        variables=[],
    )
    assert infer_analysis_type(context).key == "repeated_measures"


def test_real_repeated_kernels_are_discoverable_and_lcga_stays_planned():
    catalog = method_skill_catalog(enabled=True)
    by_id = {row["id"]: row for row in catalog["components"]}
    assert by_id["association.mixed_effects"]["kernel_modules"] == ["mixed_effects"]
    assert by_id["association.gee"]["kernel_modules"] == ["gee"]
    assert by_id["phenotyping.mixed_effects_growth"]["kernel_modules"] == ["mixed_effects"]
    assert "phenotyping.lcga_gbtm" not in by_id


def reviewed_plan():
    spec = canonical_repeated_measure_spec()
    context = ResearchContext(
        research_question="Estimate change over time in a repeated synthetic measurement.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=30, n_patients=30, id_columns=["stay_id"]),
        variables=[ConceptDescriptor(name=name, role="other", dtype="float64") for name in spec.required_columns],
        user_preferences=UserPreferences(data_constraints=json.dumps({"data_preparation": {"aggregations": {"measurement": "mean"}}})),
    )
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="repeated_measures", steps=[
        AnalysisStep(step_id="growth", planned_analysis_role="primary", intent="Estimate growth and state informative death dropout",
                     inputs=[*spec.required_columns, "artifact:analysis_cohort"], expected_outputs=list(REPEATED_OUTPUTS),
                     method=repeated_measure_method(spec), scientific_action_id="association.mixed_effects"),
    ])
    return context, plan


def test_scientific_review_blocks_missing_death_design_and_accepts_complete_design():
    context, plan = reviewed_plan()
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "REPEATED_MEASURE_DESIGN_UNRESOLVED" not in {v.code for v in review.findings}
    assert resolve_primary_capability(analysis_type=plan.analysis_type, plan=plan).owner_claimed is True
    spec_payload = canonical_repeated_measure_spec().model_dump(mode="json")
    spec_payload.pop("dropout_limitation")
    plan.steps[0].method = "repeated_measures(" + json.dumps(spec_payload) + ")"
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "REPEATED_MEASURE_DESIGN_UNRESOLVED" in {v.code for v in review.findings}
    assert review.approval_allowed is False
    assert resolve_primary_capability(analysis_type=plan.analysis_type, plan=plan).owner_claimed is False


def test_plan_reuses_accepted_preparation_aggregation():
    context, plan = reviewed_plan()
    plan.steps[0].method = repeated_measure_method(canonical_repeated_measure_spec().model_copy(update={"aggregation": "max"}))
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "REPEATED_MEASURE_DESIGN_UNRESOLVED" in {v.code for v in review.findings}


def test_planner_receives_closed_time_and_dropout_coordinates():
    context, _plan = reviewed_plan()
    guide = render_plan_scientific_guardrails(context)
    assert all(name in guide for name in ("repeated_measure_spec", "time_origin", "bin_width", "dropout_limitation", "death_column"))


@pytest.mark.parametrize("question", [
    "用混合效应模型分析重复测量的生化指标随时间变化。",
    "Fit GEE to a binary repeated outcome on ICU days 0 through 5.",
])
def test_repeated_outcomes_select_their_family_in_both_languages(question):
    context, _plan = reviewed_plan()
    assert infer_analysis_type(context.model_copy(update={"research_question": question})).key == "repeated_measures"


def test_progressive_compilation_retains_the_reviewed_time_and_dropout_packet():
    context, _plan = reviewed_plan()
    spec = canonical_repeated_measure_spec()
    skeleton = ProgressivePlanSkeleton(
        analysis_type="repeated_measures", cohort={"name": "synthetic", "selection_mode": "all_input_rows"},
        rationale="Estimate the repeated outcome on its reviewed ICU time grid and retain the death-dropout limitation.",
        display_labels=[{"key": name, "value": name.replace("_", " ")} for name in spec.required_columns],
        steps=[
            ProgressiveSkeletonStep(step_id="cohort", module_id="cohort_definition", planned_analysis_role="auxiliary", objective="Account for the complete synthetic source population."),
            ProgressiveSkeletonStep(step_id="growth", module_id="custom_analysis", planned_analysis_role="primary",
                                    custom_method="repeated_measures",
                                    objective="Estimate repeated growth with an informative-death-dropout limitation.", depends_on=["cohort"],
                                    raw_inputs=list(spec.required_columns), product_inputs=[{"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}],
                                    scientific_action_id=spec.action_id, repeated_measure_spec=spec.model_dump_json(),
                                    outputs=[{"product_id": p, "semantic_role": "custom"} for p in REPEATED_OUTPUTS]),
        ],
    )
    plan, _receipt = compile_progressive_plan(skeleton=skeleton, context=context)
    assert plan.steps[1].method == repeated_measure_method(spec)


def test_repeated_actions_are_offered_to_their_own_planning_family():
    repeated = {row.action_id for row in scientific_actions_for_analysis_type("repeated_measures").actions}
    ordinary = {row.action_id for row in scientific_actions_for_analysis_type("association_study").actions}
    assert {"association.mixed_effects", "association.gee"} <= repeated
    assert not {"association.mixed_effects", "association.gee"} & ordinary
