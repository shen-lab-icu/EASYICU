"""Shared typed study and descriptive-product fixtures for scientific review."""

from __future__ import annotations

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.claim_ceiling import DescriptiveClaimContract
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question=(
            "Among adult ICU stays, is a first-24-hour exposure associated "
            "with in-hospital mortality?"
        ),
        cohort=CohortDescriptor(
            cohort_name="adult ICU stays",
            database="miiv",
            n_patients=None,
            n_stays=94_458,
            inclusion_criteria=["adult ICU stays; retain ICU readmissions"],
            id_columns=["stay_id"],
            provenance={"analysis_unit": "icu_stay"},
        ),
        variables=[
            ConceptDescriptor(
                name="exposure",
                role=VariableRole.OTHER,
                dtype="int64",
                analysis_window="icu_admission[0,24]h",
                analysis_window_role="exposure_definition",
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"
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
        user_preferences=UserPreferences(
            covariates=["age"],
            covariate_selection="planner_selectable",
            timing_and_design="Audit timing and readmissions.",
            must_have_outputs="Execute timing and readmission sensitivity analyses.",
        ),
    )


def _absolute_risk_distribution_step(*, descriptive: bool = True) -> AnalysisStep:
    return AnalysisStep(
        step_id="absolute_risk_distribution",
        planned_analysis_role="primary",
        intent="Report observed prevalence, absolute risks, and risk difference.",
        inputs=["cohort:analysis_set", "exposure", "death"],
        expected_outputs=["table:exposure_outcome_distribution"],
        method="descriptive",
        descriptive_claim=(
            DescriptiveClaimContract(
                unresolved_limitations=(
                    "post_baseline_exposure_opportunity_unresolved",
                )
            )
            if descriptive
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
            "risk_difference_contrast": {
                "reference_exposure_level": 0,
                "comparison_exposure_level": 1,
            },
            "confidence_level": 0.95,
        },
    )


def _traditional_table_one_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="table_one",
        planned_analysis_role="auxiliary",
        intent="Describe the cohort by exposure group.",
        inputs=["cohort:analysis_set", "exposure", "age"],
        expected_outputs=["table:table_one"],
        method="descriptive",
        table_one_spec={
            "group_by": "exposure",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                }
            ],
        },
    )
