"""Synthetic descriptive outline, context, and skeleton fixtures."""

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline, ProgressivePlanSkeleton
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences, VariableRole


CASES = [
    ("eicu", "age", "描述一个合成 ICU 队列中年龄的分布。"),
    ("miiv", "lact", "描述一个合成 MIMIC-IV 队列中首日血乳酸的分布。"),
    ("eicu", "age", "Describe the age distribution in a synthetic ICU cohort."),
    ("miiv", "lact", "Describe first-day blood lactate in a synthetic MIMIC-IV cohort."),
]


def descriptive_context(database="eicu", variable="age", question=CASES[0][2]):
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="synthetic", database=database, n_stays=0,
            id_columns=["stay_id"],
            provenance={"evidence_stage": "metadata_only_planning", "patient_rows_read": False},
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(
                name=variable, dtype="float64",
                role=VariableRole.DEMOGRAPHIC if variable == "age" else VariableRole.LAB,
                description="Age at admission" if variable == "age" else "First-day measured lactate",
            ),
        ],
        time_windows=[{
            "name": "first_day", "anchor": "icu_admission",
            "start_hours": 0, "end_hours": 24,
        }],
        primary_exposure=None, target_outcome=None,
        user_preferences=UserPreferences(inferred_analysis_family="descriptive_epidemiology"),
    )


def descriptive_answer(variable="age"):
    selected = {
        "design_id": "empirical_distribution",
        "analysis_type": "descriptive_epidemiology",
        "estimand": "The empirical distribution of the recorded measurement among ICU stays.",
        "time_zero": "ICU admission defines the study time origin.",
        "observation_window": "Use the host-bound measurement window for each stay.",
        "primary_method": "Univariate empirical summaries",
        "required_variables": [variable],
        "assumptions": ["Each input row represents one eligible ICU stay."],
        "novelty_positioning": "Characterize the selected cohort before any modeling.",
        "figure_role": "Display the empirical distribution and observed denominator.",
        "supports": "The observed distribution of the measured variable.",
        "cannot_prove": "Causal effects, associations and population inference.",
        "reviewable_plan": [
            "Recommend all eligible ICU stays with an explicit denominator.",
            "Describe the measured variable in the host-bound window; no exposure contrast.",
            "No outcome or follow-up is needed for this distributional question.",
            "Use empirical quantiles, mean and SD; no adjusted model is applicable.",
            "Report missing and observed counts; summarize observed values without imputation.",
            "Check units, plausible ranges, duplicate stays and measurement availability.",
        ],
        "disposition": "selected",
        "decision_reason": "Retain the full empirical shape without imposing a distribution.",
    }
    alternative = {
        **selected,
        "design_id": "parametric_distribution",
        "primary_method": "Fit a normal reference distribution",
        "reviewable_plan": None,
        "disposition": "rejected",
        "decision_reason": "A parametric shape assumption is unnecessary for this question.",
    }
    return {
        "analysis_type": "descriptive_epidemiology",
        "cohort_objective": "Account for the selected ICU stays and measurement availability.",
        "design_selection": {"candidates": [selected, alternative]},
        "steps": [
            {
                "step_id": "cohort", "planned_analysis_role": "auxiliary",
                "module_id": "cohort_definition",
                "objective": "Account for the eligible ICU stay denominator.",
                "variable_names": [variable],
            },
            {
                "step_id": "summary", "planned_analysis_role": "primary",
                "module_id": "custom_analysis",
                "scientific_action_id": "descriptive.descriptive_summary",
                "objective": "Describe the empirical distribution, quantiles, mean and SD.",
                "variable_names": [variable], "depends_on": ["cohort"],
            },
            {
                "step_id": "audit", "planned_analysis_role": "auxiliary",
                "module_id": "measurement_audit",
                "objective": "Report missingness and measurement availability for the same variable.",
                "variable_names": [variable], "depends_on": ["cohort"],
            },
        ],
        "rationale": "Describe one measured variable without inventing an exposure or outcome.",
    }


def validate_answer(payload, context):
    outline = ProgressivePlanOutline.model_validate(payload)
    ProgressivePlannerAgent._validate_outline_authority(
        outline, analysis_types=["descriptive_epidemiology"],
        variable_names=[v.name for v in context.variables],
        allowed_literature_citation_keys=(), closed_domain_variables=(),
        continuous_domain_variables=[context.variables[-1].name],
        article_context=context, require_design_selection=True,
    )
    return outline


def descriptive_skeleton(outline, *, grouping=None):
    steps = []
    for step in outline.steps:
        values = step.model_dump(exclude={"variable_names", "literature_citation_keys"})
        values["raw_inputs"] = step.variable_names if step.step_id != "cohort" else []
        if step.module_id == "continuous_outcome_summary":
            values["primary_exposure"] = grouping
        if step.step_id != "cohort":
            values["product_inputs"] = [
                {"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"},
            ]
        if step.step_id == "summary":
            values["custom_method"] = "univariate_descriptive_summary"
            values["outputs"] = [
                {"product_id": "table:cohort_summary", "semantic_role": "custom"},
                {"product_id": "table:distribution_summary", "semantic_role": "custom"},
            ]
        if step.step_id == "audit":
            values["outputs"] = [
                {"product_id": "table:measurement_missingness", "semantic_role": "measurement_missingness"},
            ]
        steps.append(values)
    variable = outline.design_selection.selected.required_variables[0]
    return ProgressivePlanSkeleton(
        analysis_type=outline.analysis_type, design_selection=outline.design_selection,
        cohort={"name": "primary", "selection_mode": "all_input_rows"},
        display_labels=[{"key": variable, "value": "Measured variable"}],
        steps=steps, rationale=outline.rationale,
    )


