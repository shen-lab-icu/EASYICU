"""The user-facing adjustment decision governs planning and execution."""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.adjustment_authority import (
    AdjustmentAuthorityError,
    validate_plan_against_adjustment_authority,
)
from easyicu.research_agent.research_context.outbound import (
    outbound_safe_context_payload,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    PlannedModelRequirement,
    ResearchContext,
    UserPreferences,
)


def _context(*, selection: str, covariates: list[str]) -> ResearchContext:
    decisions = (
        {
            name: f"{name} is a prespecified baseline confounder."
            for name in covariates
        }
        if selection == "exact"
        else {}
    )
    return ResearchContext(
        research_question="Is exposure associated with death?",
        cohort=CohortDescriptor(
            cohort_name="test",
            database="miiv",
            n_stays=10,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="exposure",
        user_preferences=UserPreferences(
            covariates=covariates,
            covariate_selection=selection,
            covariate_rationales=decisions,
            covariate_temporal_roles={
                name: "baseline_static" for name in decisions
            },
        ),
    )


def _plan(covariates: list[str] | None) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Is exposure associated with death?",
        analysis_type="association_study",
        steps=[
            AnalysisStep(
                step_id="primary_model",
                planned_analysis_role="primary",
                intent="Estimate the declared observational association.",
                inputs=["exposure", "death", "age", "sex"],
                expected_outputs=["table:adjusted_association_estimates"],
                method="adjusted_association_models",
                model_requirements=[
                    PlannedModelRequirement(
                        requirement_id="primary",
                        outcome="death",
                        outcome_type="binary",
                        method_family="statsmodels_logit_mle",
                        exposure_source="exposure",
                        analysis_role="primary",
                        analysis_set="complete_case",
                        covariates=covariates,
                    )
                ],
            )
        ],
        rationale="Use the declared model.",
    )


def test_exact_empty_adjustment_set_rejects_planner_added_covariates() -> None:
    with pytest.raises(
        AdjustmentAuthorityError,
        match="adjustment_set_authority_mismatch",
    ):
        validate_plan_against_adjustment_authority(
            plan=_plan(["age", "sex"]),
            context=_context(selection="exact", covariates=[]),
        )


def test_exact_adjustment_set_accepts_only_the_exact_ordered_roster() -> None:
    context = _context(selection="exact", covariates=["age", "sex"])
    validate_plan_against_adjustment_authority(
        plan=_plan(["age", "sex"]), context=context
    )
    with pytest.raises(AdjustmentAuthorityError):
        validate_plan_against_adjustment_authority(
            plan=_plan(["sex", "age"]), context=context
        )


def test_exact_adjustment_set_accepts_value_aggregation_of_same_concept() -> None:
    validate_plan_against_adjustment_authority(
        plan=_plan(["age", "sex", "charlson_first"]),
        context=_context(
            selection="exact",
            covariates=["age", "sex", "charlson"],
        ),
    )


def test_exact_operationalization_locks_one_materialized_value_column() -> None:
    context = _context(
        selection="exact",
        covariates=["age", "sex", "charlson"],
    )
    context.user_preferences.covariate_operationalizations = {
        "charlson": "charlson_first"
    }
    validate_plan_against_adjustment_authority(
        plan=_plan(["age", "sex", "charlson_first"]), context=context
    )
    with pytest.raises(AdjustmentAuthorityError):
        validate_plan_against_adjustment_authority(
            plan=_plan(["age", "sex", "charlson_max"]), context=context
        )


@pytest.mark.parametrize(
    "companion",
    ["charlson_n", "charlson_measured", "charlson_first_time"],
)
def test_exact_adjustment_set_rejects_non_value_materialized_companion(
    companion: str,
) -> None:
    with pytest.raises(AdjustmentAuthorityError):
        validate_plan_against_adjustment_authority(
            plan=_plan(["age", "sex", companion]),
            context=_context(
                selection="exact",
                covariates=["age", "sex", "charlson"],
            ),
        )


def test_planner_selectable_adjustment_set_preserves_existing_behavior() -> None:
    validate_plan_against_adjustment_authority(
        plan=_plan(["age", "sex"]),
        context=_context(selection="planner_selectable", covariates=[]),
    )


def test_outbound_context_preserves_exact_empty_as_positive_decision() -> None:
    payload = outbound_safe_context_payload(
        _context(selection="exact", covariates=[])
    )
    assert payload["explicit_user_choices"]["covariate_selection"] == "exact"
    assert payload["explicit_user_choices"]["covariates"] == []


def test_exact_adjustment_decision_projects_rationale_and_temporal_role() -> None:
    payload = outbound_safe_context_payload(
        _context(selection="exact", covariates=["age", "sex"])
    )

    choices = payload["explicit_user_choices"]
    assert choices["covariate_rationales"]["age"].startswith("age is")
    assert choices["covariate_temporal_roles"] == {
        "age": "baseline_static",
        "sex": "baseline_static",
    }
