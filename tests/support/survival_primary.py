"""Synthetic primary-survival declarations shared by owner regression tests."""

from easyicu.research_agent.schema import (
    AnalysisStep, CohortDescriptor, ConceptDescriptor, EndpointSpec,
    FamilyPrimaryResultRequirement, ResearchContext, VariableRole,
)
from easyicu.research_agent.contracts.survival import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT, SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
)

def _requirement() -> FamilyPrimaryResultRequirement:
    return FamilyPrimaryResultRequirement(
        analysis_family="survival",
        exposure_source="treatment",
        outcome="death",
        expected_result_product="table:survival_effect_estimates",
        input_product="table:analysis_cohort",
        estimator="cox_proportional_hazards",
        effect_scale="hazard_ratio",
        uncertainty_method="wald_95ci",
        population="eligible cohort",
        time_origin="ICU admission",
        time_column="follow_up_days",
        time_unit="days",
        event_column="death",
        event_value=1,
        event_definition="death",
        censoring_strategy="administrative at 28 days",
        competing_risk_strategy="none",
        time_horizon="28 days",
        time_horizon_value=28,
        effect_measure="hazard ratio",
        proportional_hazards_diagnostic="schoenfeld_global_test",
        proportional_hazards_alpha=0.05,
        proportional_hazards_policy="block_paper_authorization",
        covariates=["age"],
        model_terms=[
            {
                "name": "treatment",
                "role": "exposure",
                "coding": "binary",
                "levels": ["0", "1"],
                "reference_level": "0",
                "transform": "treatment_contrast",
            },
            {
                "name": "age",
                "role": "covariate",
                "coding": "continuous",
                "transform": "identity",
            },
        ],
        exposure_encoding="declared_model_terms",
        missing_data_policy="complete_case",
    )

def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="04_primary",
        intent="Estimate the declared survival contrast.",
        inputs=["table:analysis_cohort"],
        expected_outputs=[
            "table:survival_effect_estimates",
            SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
            SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
        ],
        method="cox_proportional_hazards",
        planned_analysis_role="primary",
        family_primary_result_requirement=_requirement(),
    )

def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Does treatment change 28-day mortality?",
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=8, n_stays=8
        ),
        variables=[
            ConceptDescriptor(
                name="treatment", role=VariableRole.INTERVENTION, dtype="int64"
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(
                name="follow_up_days",
                role=VariableRole.TIME,
                dtype="float64",
                unit="days",
            ),
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"
            ),
        ],
        primary_exposure="treatment",
        target_outcome="death",
        endpoint=EndpointSpec(
            name="death",
            kind="time_to_event",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
            event_column="death",
            time_column="follow_up_days",
            time_origin="ICU admission",
            censoring_rule="administrative at 28 days",
        ),
    )

def _runtime_kwargs(
    requirement: FamilyPrimaryResultRequirement,
    *,
    input_path,
    input_sha256: str,
) -> dict:
    return {
        "input_path": input_path,
        "input_product": str(requirement.input_product),
        "input_evidence_id": "evidence-analysis-cohort",
        "input_sha256": input_sha256,
        "result_product": requirement.expected_result_product,
        "exposure": requirement.exposure_source,
        "outcome": requirement.outcome,
        "effect_scale": requirement.effect_scale,
        "population": requirement.population,
        "time_origin": str(requirement.time_origin),
        "time_column": str(requirement.time_column),
        "time_unit": str(requirement.time_unit),
        "event_column": str(requirement.event_column),
        "event_value": int(requirement.event_value),
        "event_definition": str(requirement.event_definition),
        "censoring_strategy": str(requirement.censoring_strategy),
        "competing_risk_strategy": str(requirement.competing_risk_strategy),
        "time_horizon": str(requirement.time_horizon),
        "time_horizon_value": float(requirement.time_horizon_value),
        "estimator": requirement.estimator,
        "effect_measure": str(requirement.effect_measure),
        "covariates": list(requirement.covariates or ()),
        "model_terms": list(requirement.model_terms or ()),
        "ph_diagnostic": str(requirement.proportional_hazards_diagnostic),
        "ph_alpha": float(requirement.proportional_hazards_alpha),
        "ph_policy": str(requirement.proportional_hazards_policy),
    }
