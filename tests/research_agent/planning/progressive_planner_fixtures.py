"""Shared typed context and payload builders for progressive planner contracts."""

from __future__ import annotations
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanOutline,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate an exposure-outcome association with audit context.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=120,
            id_columns=["stay_id"],
            outcome_columns=["outcome_flag"],
        ),
        variables=[
            ConceptDescriptor(
                name="exposure_flag",
                role=VariableRole.INTERVENTION,
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            ),
            ConceptDescriptor(
                name="outcome_flag",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            ),
            ConceptDescriptor(
                name="age_years",
                role=VariableRole.DEMOGRAPHIC,
                dtype="float64",
                observed_domain={"n_unique": 83, "min": 18.0, "max": 100.0},
            ),
            ConceptDescriptor(
                name="sex_code",
                role=VariableRole.DEMOGRAPHIC,
                dtype="object",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": False,
                    "levels": ["A", "B"],
                },
            ),
        ],
        primary_exposure="exposure_flag",
        target_outcome="outcome_flag",
    )


def _payload() -> dict:
    return {
        "schema_version": "easyicu.progressive_plan_skeleton/1",
        "analysis_type": "association_study",
        "cohort": {
            "name": "primary",
            "selection_mode": "all_input_rows",
            "inclusion": [],
            "exclusion": [],
        },
        "display_labels": [
            {"key": "exposure_flag", "value": "Exposure status"},
            {"key": "outcome_flag", "value": "In-hospital outcome"},
            {"key": "age_years", "value": "Age in years"},
            {"key": "sex_code", "value": "Recorded sex"},
            {"key": "exposure_flag=0", "value": "Exposure absent"},
            {"key": "exposure_flag=1", "value": "Exposure present"},
        ],
        "robustness_intents": [
            {
                "spec_id": "complete_case",
                "axis": "missing",
                "description": "Refit the declared model on complete observations.",
                "missing_strategy": "complete_case",
                "complete_case_variables": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
            }
        ],
        "steps": [
            {
                "step_id": "01_cohort",
                "planned_analysis_role": "auxiliary",
                "module_id": "cohort_definition",
                "objective": "Bind and account for the prespecified analysis universe.",
                "depends_on": [],
                "raw_inputs": [],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "02_table_one",
                "planned_analysis_role": "auxiliary",
                "module_id": "table_one",
                "objective": "Describe baseline variables by the declared exposure groups.",
                "depends_on": ["01_cohort"],
                "raw_inputs": ["exposure_flag", "age_years", "sex_code"],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": "exposure_flag",
                "table_one_mode": "descriptive_smd_only",
                "table_one_variables": [
                    {"name": "age_years", "summary": "median_iqr"},
                    {"name": "sex_code", "summary": "count_percent"},
                ],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "03_distribution",
                "planned_analysis_role": "secondary",
                "module_id": "exposure_outcome_distribution",
                "objective": "Estimate prevalence and absolute outcome risk by exposure.",
                "depends_on": ["01_cohort"],
                "raw_inputs": ["exposure_flag", "outcome_flag"],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": "exposure_flag",
                "outcome": "outcome_flag",
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": 1,
                "reference_exposure_level_index": 0,
                "comparison_exposure_level_index": 1,
                "primary_contrast_level_index": None,
                "denominator_policy": "all_declared_rows",
                "missing_exposure_policy": "fail_closed",
                "missing_outcome_policy": "fail_closed",
                "confidence_level": 0.95,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "04_measurement",
                "planned_analysis_role": "auxiliary",
                "module_id": "measurement_audit",
                "objective": "Audit missingness and observation-process coverage.",
                "depends_on": ["01_cohort"],
                "raw_inputs": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
                "product_inputs": [],
                "outputs": [
                    {
                        "product_id": "table:measurement_missingness",
                        "semantic_role": "measurement_missingness",
                    },
                    {
                        "product_id": "table:measurement_process",
                        "semantic_role": "measurement_process",
                    },
                ],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "05_primary",
                "planned_analysis_role": "primary",
                "module_id": "adjusted_association",
                "objective": "Estimate the prespecified adjusted association.",
                "depends_on": ["01_cohort"],
                "raw_inputs": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
                "product_inputs": [],
                "outputs": [],
                "scientific_action_id": "association.adjusted_association",
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": "exposure_flag",
                "outcome": "outcome_flag",
                "outcome_type": "binary",
                "model_terms": [
                    {
                        "name": "exposure_flag",
                        "role": "exposure",
                        "coding": "binary",
                        "reference_level_index": 0,
                    },
                    {
                        "name": "age_years",
                        "role": "covariate",
                        "coding": "continuous",
                        "reference_level_index": None,
                        "clinical_rationale": (
                            "Age can confound the association because it precedes "
                            "exposure ascertainment and relates to outcome risk."
                        ),
                    },
                    {
                        "name": "sex_code",
                        "role": "covariate",
                        "coding": "binary",
                        "reference_level_index": 0,
                        "clinical_rationale": (
                            "Sex can confound the association because it is fixed "
                            "before exposure ascertainment and relates to risk."
                        ),
                    },
                ],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
            {
                "step_id": "06_sensitivity",
                "planned_analysis_role": "sensitivity",
                "module_id": "custom_analysis",
                "objective": "Run the explicitly prespecified scientific sensitivity grid.",
                "depends_on": ["05_primary"],
                "raw_inputs": ["exposure_flag", "outcome_flag", "age_years"],
                "product_inputs": [
                    {
                        "producer_step_id": "05_primary",
                        "product_id": "table:adjusted_association_estimates",
                    }
                ],
                "outputs": [
                    {
                        "product_id": "table:scientific_sensitivity",
                        "semantic_role": "scientific_sensitivity",
                    }
                ],
                "scientific_action_id": None,
                "custom_method": "prespecified_scientific_sensitivity",
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": ["flexible_form"],
                "literature_bindings": [],
            },
            {
                "step_id": "07_figure",
                "planned_analysis_role": "auxiliary",
                "module_id": "visualization",
                "objective": "Render the exact descriptive and adjusted result products.",
                "depends_on": ["03_distribution", "05_primary"],
                "raw_inputs": [],
                "product_inputs": [
                    {
                        "producer_step_id": "03_distribution",
                        "product_id": "table:exposure_outcome_distribution",
                    },
                    {
                        "producer_step_id": "05_primary",
                        "product_id": "table:adjusted_association_estimates",
                    },
                ],
                "outputs": [
                    {
                        "product_id": "figure:primary_results",
                        "semantic_role": "figure",
                    }
                ],
                "scientific_action_id": None,
                "custom_method": None,
                "table_one_group_by": None,
                "table_one_mode": None,
                "table_one_variables": [],
                "primary_exposure": None,
                "outcome": None,
                "outcome_type": None,
                "model_terms": [],
                "event_level_index": None,
                "reference_exposure_level_index": None,
                "comparison_exposure_level_index": None,
                "primary_contrast_level_index": None,
                "denominator_policy": None,
                "missing_exposure_policy": None,
                "missing_outcome_policy": None,
                "confidence_level": None,
                "sensitivity_spec_ids": [],
                "literature_bindings": [],
            },
        ],
        "rationale": "Separate descriptive denominators from the adjusted association.",
    }


def _outline_payload(payload: dict | None = None) -> dict:
    source = payload or _payload()
    analysis_type = source["analysis_type"]
    return {
        "schema_version": "easyicu.progressive_plan_outline/1",
        "analysis_type": analysis_type,
        "cohort_objective": "Use the sealed cohort and preserve its denominator.",
        "design_selection": {
            "schema_version": "easyicu.research_design_selection/1",
            "claim_ceiling": "analysis_only",
            "candidates": [
                {
                    "design_id": "selected_primary_design",
                    "analysis_type": analysis_type,
                    "estimand": "Adjusted exposure contrast for outcome_flag.",
                    "time_zero": "Start of the sealed synthetic cohort episode.",
                    "observation_window": "The prespecified episode observation window.",
                    "primary_method": "Host-owned primary analysis method",
                    "required_variables": ["exposure_flag", "outcome_flag"],
                    "assumptions": ["The declared adjustment set is adequate."],
                    "literature_citation_keys": [],
                    "novelty_positioning": "Tests the question in the sealed cohort context.",
                    "figure_role": "Show the primary estimate with its uncertainty.",
                    "supports": "The prespecified primary association estimate.",
                    "cannot_prove": "A causal effect without stronger identification.",
                    "reviewable_plan": [
                        "Use the sealed cohort with one row per declared analysis unit.",
                        "Use the declared exposure and its prespecified baseline timing and aggregation.",
                        "Use outcome_flag through the declared episode follow-up.",
                        "Use the host-owned adjusted association model and prespecified covariates.",
                        "Quantify missingness and apply the prespecified missing-data strategy.",
                        "Check denominator, events, coverage, missingness, and alternative specifications.",
                    ],
                    "disposition": "selected",
                    "decision_reason": (
                        "Directly binds exposure_flag and outcome_flag to the "
                        "prespecified primary question."
                    ),
                },
                {
                    "design_id": "rejected_alternative_design",
                    "analysis_type": analysis_type,
                    "estimand": "Unadjusted exposure contrast for outcome_flag.",
                    "time_zero": "Start of the sealed synthetic cohort episode.",
                    "observation_window": "The prespecified episode observation window.",
                    "primary_method": "Unadjusted descriptive contrast",
                    "required_variables": ["exposure_flag", "outcome_flag"],
                    "assumptions": ["Crude group differences are interpretable."],
                    "literature_citation_keys": [],
                    "novelty_positioning": "Provides a less adjusted comparator design.",
                    "figure_role": "Show only the crude group contrast.",
                    "supports": "A descriptive difference between exposure groups.",
                    "cannot_prove": "An adjusted or causal exposure effect.",
                    "disposition": "rejected",
                    "decision_reason": (
                        "Reject because exposure_flag confounding is not addressed "
                        "for the outcome_flag question."
                    ),
                },
            ],
        },
        "steps": [
            {
                "step_id": step["step_id"],
                "planned_analysis_role": step["planned_analysis_role"],
                "module_id": step["module_id"],
                "objective": step["objective"],
                "depends_on": list(step["depends_on"]),
                "variable_names": [
                    "exposure_flag",
                    "outcome_flag",
                    "age_years",
                    "sex_code",
                ],
                "literature_citation_keys": list(
                    dict.fromkeys(
                        binding["citation_key"]
                        for binding in step["literature_bindings"]
                    )
                ),
                "scientific_action_id": step["scientific_action_id"],
            }
            for step in source["steps"]
        ],
        "rationale": source["rationale"],
    }


def _materialization_payloads(payload: dict | None = None) -> list[dict]:
    source = payload or _payload()
    outline = ProgressivePlanOutline.model_validate(_outline_payload(source))
    responses = []
    for outline_step, step in zip(outline.steps, source["steps"], strict=True):
        responses.append(
            {
                "schema_version": "easyicu.progressive_step_materialization/1",
                "outline_step_sha256": canonical_sha256(
                    outline_step.model_dump(mode="json")
                ),
                "foundation": None,
                "step": step,
            }
        )
    return responses


def _foundation_payload(payload: dict | None = None) -> dict:
    source = payload or _payload()
    outline = ProgressivePlanOutline.model_validate(_outline_payload(source))
    return {
        "schema_version": "easyicu.progressive_plan_foundation/1",
        "outline_sha256": canonical_sha256(outline.model_dump(mode="json")),
        "foundation": {
            "cohort": source["cohort"],
            "display_labels": source["display_labels"],
            "robustness_intents": source["robustness_intents"],
            "know_how_decisions": source.get("know_how_decisions", []),
        },
    }
