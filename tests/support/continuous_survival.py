"""A synthetic continuous-exposure survival study, shared by its tests.

Test modules may not import one another (``tests/governance/test_test_organization.py``).
The continuous survival suite's authority body (a laboratory value's 24-hour
maximum and 28-day mortality) serves the engine tests; the planning, review
and Web compilation tests start from one study with no declared design: the
highest total bilirubin in the first 24 hours and 90-day mortality, which no
benchmark item asks.  Zero patient rows.
"""

from __future__ import annotations

import json

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    UserPreferences,
    VariableRole,
)

OUTPUTS = (
    "table:landmark_continuous_table_one",
    "table:landmark_continuous_risk_set_flow",
    "table:landmark_continuous_km_curve",
    "table:landmark_continuous_cox_summary",
    "table:landmark_continuous_ph_diagnostics",
    "table:landmark_continuous_time_varying_cox_summary",
    "table:landmark_continuous_spline_curve",
    "table:landmark_continuous_measurement_audit",
    "log:landmark_continuous_survival_receipt",
    "figure:landmark_continuous_survival_suite",
)


def continuous_authority_body(**overrides) -> dict:
    """A complete authority body: a laboratory value's 24-hour maximum and 28-day mortality."""

    body = {
        "schema_version": "easyicu.landmark_continuous_survival_runtime_authority/2",
        "authority_kind": "landmark_continuous_survival_suite",
        "protocol_content_sha256": "a" * 64,
        "plan_method": "signed_landmark_continuous_survival_suite",
        "plan_intent": "Execute the signed continuous-exposure landmark survival suite.",
        "plan_outputs": list(OUTPUTS),
        "development_execution_only_allowed": False,
        "exposure_column": "lab_max",
        "exposure_label": "Laboratory value, highest in hours 0 to 24",
        "exposure_unit": "mmol/L",
        "exposure_window_summary": "max",
        "exposure_window_hours": [0.0, 24.0],
        "exposure_increment_rule": "largest_round_step_within_interquartile_range",
        "event_column": "mort_28d",
        "followup_time_column": "followup_days_28d",
        "endpoint_time_origin": "ICU admission",
        "endpoint_censoring_rule": "Observed death time or administrative censoring at 28 days.",
        "landmark_hours": 24.0,
        "endpoint_horizon_days": 28.0,
        "analysis_unit_label": "ICU stays",
        "derived_event_column": "death_after_24h_by_day28",
        "derived_time_column": "followup_days_from_24h_landmark",
        "adjustment_columns": ["age", "sex"],
        "categorical_adjustment_columns": ["sex"],
        "table_one_columns": ["age", "sex"],
        "estimator": "cox_ph_lifelines_efron",
        "effect_measure": "hazard_ratio_per_exposure_step",
        "uncertainty_method": "wald_95_ci",
        "proportional_hazards_diagnostic": "schoenfeld_residual_test",
        "proportional_hazards_alpha": 0.05,
        "proportional_hazards_policy": "block_paper_authorization",
        "time_varying_effect_method": "piecewise_time_varying_cox",
        "time_varying_interval_cutpoints_days": [7.0, 14.0],
        "spline_knot_quantiles": [0.1, 0.5, 0.9],
        "spline_reference": "median_in_model_population",
        "curve_quantile_range": [0.1, 0.9],
        "curve_points": 41,
        "functional_form_alpha": 0.05,
        "functional_form_policy": "spline_contrasts_replace_linear_estimate",
        "descriptive_grouping": "value_tertiles",
        "interpretation": "descriptive_prognostic_association_not_causal",
        "table_one_product": OUTPUTS[0],
        "risk_set_product": OUTPUTS[1],
        "km_product": OUTPUTS[2],
        "cox_product": OUTPUTS[3],
        "ph_product": OUTPUTS[4],
        "time_varying_cox_product": OUTPUTS[5],
        "spline_product": OUTPUTS[6],
        "measurement_audit_product": OUTPUTS[7],
        "receipt_product": OUTPUTS[8],
        "figure_product": OUTPUTS[9],
    }
    body.update(overrides)
    return body




ALLOWED = ["strobe_2007", "record_2015"]
BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
QUESTION = (
    "Among adult ICU stays, is the highest total bilirubin in the first 24 hours "
    "associated with the time to death within 90 days?"
)
AGE = {
    "name": "age", "coding": "continuous", "reference_level_index": None,
    "clinical_rationale": "Age precedes admission and drives both liver dysfunction and death.",
}
SEX = {
    "name": "sex", "coding": "binary", "reference_level_index": 0,
    "clinical_rationale": "Sex is fixed at admission and associated with both bilirubin and death.",
}


def continuous_survival_context(**update) -> ResearchContext:
    base = ResearchContext(
        research_question=QUESTION,
        cohort=CohortDescriptor(
            cohort_name="synthetic_bilirubin", database="miiv", n_stays=0, id_columns=["stay_id"],
            outcome_columns=["mort_90d"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="bili_max", description="Highest total bilirubin in the first 24 h",
                              role=VariableRole.LAB, dtype="float64", unit="mg/dL",
                              source_concept="bili"),
            # Another summary of the same measurement: never a confounder.
            ConceptDescriptor(name="bili_mean", description="Mean total bilirubin in the first 24 h",
                              role=VariableRole.LAB, dtype="float64", unit="mg/dL",
                              source_concept="bili"),
            ConceptDescriptor(name="mort_90d", description="Death by day 90 from ICU admission",
                              role=VariableRole.OUTCOME, dtype="float64", observed_domain=BINARY),
            ConceptDescriptor(name="followup_days_90d", description="90-day event or censoring time",
                              role=VariableRole.OTHER, dtype="float64", unit="days"),
            ConceptDescriptor(name="age", description="age", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", unit="years"),
            ConceptDescriptor(name="sex", description="sex", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", observed_domain=BINARY),
        ],
        target_outcome="mort_90d",
        primary_exposure="bili_max",
        time_windows=[
            TimeWindow(name="icu_admission_0_24h", anchor="icu_admission", start_hours=0.0, end_hours=24.0)
        ],
        user_preferences=UserPreferences(inferred_analysis_family="survival"),
    )
    return base.model_copy(update=update)


def continuous_survival_request(context: ResearchContext, planning_contract_context: str = ""):
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=planning_contract_context,
    )


def continuous_survival_spec(request, roster: list[dict], *, labels: dict | None = None) -> dict:
    keys = [*request.required_reader_label_keys, *request.level_label_keys, *(item["name"] for item in roster)]
    labels = labels if labels is not None else {key: f"Reader label for {key}" for key in keys}
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": roster,
        "reader_display_labels": [{"key": key, "value": value} for key, value in labels.items()],
        "comparator_applications": [],
        "roster_decision_note": "Adjust for the baseline factors that drive both bilirubin and death.",
    }


def proposed_continuous_survival_plan(context: ResearchContext, spec: dict):
    """One scripted Planner answer compiled into the proposed continuous suite plan."""

    llm = ScriptedMockLLMClient([json.dumps(spec)])
    result = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context="", required_primary_cohort_selection_mode="all_input_rows",
    )
    return result.output, llm
