"""Fresh Planner typed-output scientific authority contracts."""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.planner_output_contract import (
    PlannerOutputContractError,
    missing_post_baseline_descriptive_claims,
    validate_fresh_planner_typed_product_specs,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context(*, exposure_window: str = "icu_admission[0,24]h") -> ResearchContext:
    return ResearchContext(
        research_question="Describe exposure prevalence and hospital mortality.",
        cohort=CohortDescriptor(
            cohort_name="adult ICU stays",
            database="miiv",
            n_stays=100,
            inclusion_criteria=["adult ICU stays"],
            id_columns=["stay_id"],
        ),
        variables=[
            ConceptDescriptor(
                name="exposure",
                role=VariableRole.OTHER,
                dtype="int64",
                analysis_window=exposure_window,
                analysis_window_role="exposure_definition",
            ),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int64",
            ),
        ],
        target_outcome="death",
        endpoint=EndpointSpec(
            name="death",
            kind="binary",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
        ),
        primary_exposure="exposure",
    )


def _distribution_step(*, with_ceiling: bool) -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_distribution",
        planned_analysis_role="primary",
        intent="Report observed absolute risks without causal interpretation.",
        inputs=["cohort:analysis_set", "exposure", "death"],
        expected_outputs=["table:exposure_outcome_distribution"],
        method="descriptive",
        descriptive_claim=(
            {
                "claim_ceiling": "descriptive_only",
                "unresolved_limitations": [
                    "post_baseline_exposure_opportunity_unresolved"
                ],
            }
            if with_ceiling
            else None
        ),
        exposure_outcome_distribution_spec={
            "exposure": "exposure",
            "exposure_levels": [0, 1],
            "outcome": "death",
            "outcome_levels": [0, 1],
            "outcome_positive_value": 1,
            "level_match_policy": "exact_typed",
            "denominator_policy": "all_declared_rows",
            "missing_outcome_policy": "structural_absence_is_non_event",
            "confidence_level": 0.95,
        },
    )


def test_post_baseline_primary_distribution_requires_typed_claim_ceiling() -> None:
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="descriptive_study",
        steps=[_distribution_step(with_ceiling=False)],
    )

    assert missing_post_baseline_descriptive_claims(
        plan=plan,
        context=_context(),
    ) == ("primary_distribution",)
    with pytest.raises(
        PlannerOutputContractError,
        match="Descriptive prose does not create",
    ):
        validate_fresh_planner_typed_product_specs(plan, context=_context())


def test_exact_typed_ceiling_closes_fresh_planner_contract() -> None:
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="descriptive_study",
        steps=[_distribution_step(with_ceiling=True)],
    )

    validate_fresh_planner_typed_product_specs(plan, context=_context())
    assert missing_post_baseline_descriptive_claims(
        plan=plan,
        context=_context(),
    ) == ()


def test_baseline_distribution_does_not_invent_post_baseline_limitation() -> None:
    context = _context(exposure_window="icu_admission[-24,0]h")
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_distribution_step(with_ceiling=False)],
    )

    validate_fresh_planner_typed_product_specs(plan, context=context)
    assert missing_post_baseline_descriptive_claims(plan=plan, context=context) == ()


def test_association_step_is_not_relabelled_as_descriptive() -> None:
    step = _distribution_step(with_ceiling=False).model_copy(
        update={
            "method": "adjusted_association_models",
            "scientific_capability": "association_freeform_v1",
        }
    )
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="association_study",
        steps=[step],
    )

    validate_fresh_planner_typed_product_specs(plan, context=_context())
    assert missing_post_baseline_descriptive_claims(
        plan=plan,
        context=_context(),
    ) == ()
