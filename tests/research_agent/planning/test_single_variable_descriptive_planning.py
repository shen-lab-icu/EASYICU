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
from tests.support.descriptive_plans import (
    CASES, descriptive_answer, descriptive_context, descriptive_skeleton, validate_answer,
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


@pytest.mark.parametrize("field", ["replicates", "seed"])
def test_bootstrap_diagnostic_coordinates_survive_without_rejected_values(field):
    from easyicu.research_agent.contracts.prediction_validation import BootstrapSpec

    with pytest.raises(ValidationError) as raised:
        BootstrapSpec.model_validate({"replicates": 20, "seed": 17, field: "private-value"})
    issues = safe_validation_issues(raised.value)
    assert issues[0]["location"] == [field]
    issues[0]["location"] = ["steps", 2, "bootstrap_spec", field, "private-key"]
    projected = safe_projected_validation_issues(issues)
    assert projected[0]["location"] == ["steps", 2, "bootstrap_spec", field, "<other>"]
    assert "private" not in json.dumps(projected)


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
