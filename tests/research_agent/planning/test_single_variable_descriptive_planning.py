"""A measured variable is sufficient for a descriptive design, without a model."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_payload import (
    progressive_outline_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.providers.structured_diagnostics import (
    safe_projected_validation_issues,
    safe_validation_issues,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


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


def descriptive_skeleton(outline):
    steps = []
    for step in outline.steps:
        values = step.model_dump(exclude={"variable_names", "literature_citation_keys"})
        values["raw_inputs"] = step.variable_names if step.step_id != "cohort" else []
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


@pytest.mark.parametrize("database,variable,question", CASES)
def test_single_variable_answer_validates_and_compiles(database, variable, question):
    context = descriptive_context(database, variable, question)
    outline = validate_answer(descriptive_answer(variable), context)
    # An inert client cannot call a Provider and does not bypass production
    # typed-product validation through the registered mock-client exemption.
    client = object()
    plan, receipt = ProgressivePlannerAgent(client)._compile_and_accept(
        descriptive_skeleton(outline), agent_context=context, article_context=context,
        allowed_literature_citation_keys=("strobe_2007",),
        direct_comparator_literature_keys=(), allowed_know_how_decisions=None,
        enforce_article_contract=True,
    )
    assert context.primary_exposure is None
    summary = next(step for step in plan.steps if step.step_id == "summary")
    assert summary.scientific_action_id == "descriptive.descriptive_summary"
    assert variable in summary.inputs
    assert "table:cohort_summary" in summary.expected_outputs
    assert receipt is not None


def test_single_variable_structure_contract_has_a_summary_owner():
    context = descriptive_context()
    families, variables, actions, rows = ProgressivePlannerAgent._request_authorities(context)
    prompt = ProgressivePlannerAgent._user_prompt(
        context, analysis_types=families, variables=variables, action_rows=rows,
    )
    request = progressive_outline_structured_output_request(
        analysis_types=families, variable_names=variables,
        scientific_action_ids=actions,
    )
    schema = json.loads(request.schema_json)
    assert schema["$defs"]["ResearchDesignCandidate"]["properties"]["required_variables"]["minItems"] == 1
    assert "measured variable" in prompt.lower()
    assert "exactly six" in prompt.lower()
    assert "at least 8 characters" in prompt
    assert "descriptive.descriptive_summary" in prompt
    assert "<outcome/follow-up or why these are inapplicable>" in prompt
    outline = ProgressivePlanOutline.model_validate(descriptive_answer())
    materialization = ProgressivePlannerAgent._materialization_prompt(
        context=context, outline=outline, outline_step=outline.steps[1],
        outline_step_sha256="a" * 64, variables=variables, action_rows=rows,
        allowed_literature_citation_keys=(), know_how_context="",
        planning_contract_context="", prefix_summary=[],
        available_product_refs=[("cohort", "artifact:analysis_cohort")],
    )
    assert "table:distribution_summary" in materialization
    assert "semantic_role=custom" in materialization


def test_incomplete_review_plan_keeps_its_typed_refusal():
    payload = descriptive_answer()
    payload["design_selection"]["candidates"][0]["reviewable_plan"][2] = "N/A"
    with pytest.raises(ProgressivePlanCompileError) as raised:
        validate_answer(payload, descriptive_context())
    assert raised.value.reason_code == "progressive_design_selection_reviewable_plan_incomplete"


def test_an_arbitrary_custom_descriptive_step_is_still_unavailable():
    payload = descriptive_answer()
    payload["steps"][1]["scientific_action_id"] = None
    outline = ProgressivePlanOutline.model_validate(payload)
    with pytest.raises(ProgressivePlanCompileError) as raised:
        ProgressivePlannerAgent._validate_outline_authority(
            outline, analysis_types=["descriptive_epidemiology"],
            variable_names=["age"], allowed_literature_citation_keys=(),
        )
    assert raised.value.reason_code == "progressive_outline_module_unavailable"


def test_schema_issue_paths_keep_owned_names_without_rejected_values():
    payload = descriptive_answer("private-measurement-value")
    for candidate in payload["design_selection"]["candidates"]:
        candidate["required_variables"] = []
    payload["steps"][1]["variable_names"] = ["private-column", "private-column"]
    payload["private-key"] = "private-value"
    with pytest.raises(ValidationError) as raised:
        ProgressivePlanOutline.model_validate(payload)
    issues = safe_validation_issues(raised.value)
    assert ["design_selection", "candidates", 0, "required_variables"] in [
        item["location"] for item in issues
    ]
    assert safe_projected_validation_issues(issues) == issues
    assert "private" not in json.dumps(issues)


@pytest.mark.parametrize("model", [ProgressivePlanOutline, ProgressivePlanSkeleton])
def test_owned_schema_names_survive_diagnostic_projection(model):
    fields = set()

    def walk(node):
        if isinstance(node, dict):
            fields.update(node.get("properties", {}))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(model.model_json_schema())
    for name in fields:
        projected = safe_projected_validation_issues([
            {"location": [name, "private-value"], "issue_type": "missing"},
        ])
        assert projected[0]["location"] == [name, "<other>"]


def test_non_descriptive_candidate_still_requires_two_variables():
    payload = descriptive_answer()
    for candidate in payload["design_selection"]["candidates"]:
        candidate["analysis_type"] = "association_study"
    with pytest.raises(ValidationError, match="at least two variables"):
        ProgressivePlanOutline.model_validate(payload)


def test_grouped_secondary_summary_cannot_impersonate_a_primary_distribution():
    payload = descriptive_answer()
    payload["steps"][1]["module_id"] = "continuous_outcome_summary"
    payload["steps"][1]["scientific_action_id"] = None
    with pytest.raises(ValidationError) as raised:
        ProgressivePlanOutline.model_validate(payload)
    assert safe_validation_issues(raised.value)[0]["location"] == ["steps", 1]
    assert "secondary host summary" in str(raised.value)


def test_compiler_does_not_accept_an_arbitrary_custom_primary():
    outline = ProgressivePlanOutline.model_validate(descriptive_answer())
    skeleton = descriptive_skeleton(outline)
    steps = [
        step.model_copy(update={"scientific_action_id": None})
        if step.step_id == "summary" else step
        for step in skeleton.steps
    ]
    with pytest.raises(ProgressivePlanCompileError) as raised:
        compile_progressive_plan(
            skeleton=skeleton.model_copy(update={"steps": steps}),
            context=descriptive_context(),
        )
    assert raised.value.reason_code == "progressive_analysis_module_unavailable"
