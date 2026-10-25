"""A baseline materialization sees the executable contract before guessing products."""

from copy import deepcopy
import json

import pytest

from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.cohort_summary_products import auxiliary_summary_product
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError, ProgressivePlanOutline, ProgressivePlanSkeleton,
    ProgressiveSkeletonStep,
)
from easyicu.research_agent.planning.progressive_module_ids import progressive_module_ids_for_analysis_types
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ConceptDescriptor
from tests.support.descriptive_plans import descriptive_answer, descriptive_context, descriptive_skeleton
from .progressive_planner_fixtures import (
    _context, _foundation_payload, _materialization_payloads, _outline_payload, _payload,
)


def _case(question="Describe serum concentration by admission category in a synthetic cohort."):
    context = descriptive_context(question=question)
    outline = ProgressivePlanOutline.model_validate(descriptive_answer())
    skeleton = descriptive_skeleton(outline)
    baseline = ProgressiveSkeletonStep(
        step_id="population_profile", module_id="custom_analysis",
        planned_analysis_role="auxiliary", objective="Describe the cohort's baseline age distribution.",
        depends_on=["cohort"], raw_inputs=["age"],
        product_inputs=[{"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}],
        outputs=[{"product_id": "table:invented_profile", "semantic_role": "custom"}],
        custom_method="prespecified_population_description",
    )
    skeleton = skeleton.model_copy(update={"steps": [skeleton.steps[0], baseline, *skeleton.steps[1:]]})
    outline = ProgressivePlanOutline.model_validate({
        **outline.model_dump(mode="json"),
        "steps": [
            {**step.model_dump(include={"step_id", "module_id", "planned_analysis_role", "objective",
                                       "depends_on", "scientific_action_id"}),
             "variable_names": step.raw_inputs or ["age"]}
            for step in skeleton.steps
        ],
    })
    return context, outline, skeleton


def _prompt(context, outline):
    return ProgressivePlannerAgent._materialization_prompt(
        context=context, outline=outline, outline_step=outline.steps[1],
        outline_step_sha256=canonical_sha256(outline.steps[1].model_dump(mode="json")),
        variables=[v.name for v in context.variables], action_rows=[],
        allowed_literature_citation_keys=(), know_how_context="", planning_contract_context="",
        prefix_summary=[], available_product_refs=[("cohort", "artifact:analysis_cohort")],
    )


@pytest.mark.parametrize("kind,question", [
    ("continuous", "Describe serum concentration by admission category in a synthetic cohort."),
    ("count", "Count synthetic ICU stays by unit type and describe their age distribution."),
])
def test_baseline_prompt_and_rejection_name_the_same_executable_repair(kind, question):
    context, outline, skeleton = _case(question)
    measure = "serum_value" if kind == "continuous" else "unit_type"
    context.variables.append(ConceptDescriptor(
        name=measure, role="outcome" if kind == "continuous" else "demographic",
        dtype="float64" if kind == "continuous" else "object",
    ))
    skeleton.steps[2] = skeleton.steps[2].model_copy(update={"raw_inputs": [measure]})
    skeleton.steps[3] = skeleton.steps[3].model_copy(update={"raw_inputs": [measure]})
    if kind == "continuous":
        context.variables.append(ConceptDescriptor(name="admission_category", role="demographic", dtype="object", observed_domain={"levels": ["planned", "unplanned"]}))
        context = context.model_copy(update={"primary_exposure": "admission_category", "target_outcome": measure})
        skeleton.steps.append(ProgressiveSkeletonStep(
            step_id="stratified_measure", planned_analysis_role="secondary", module_id="continuous_outcome_summary",
            objective="Summarize serum concentration within each admission category.", depends_on=["summary"],
            raw_inputs=["admission_category", measure], primary_exposure="admission_category",
        ))
    prompt = _prompt(context, outline)
    assert "Available modules for this analysis type" in prompt
    assert "table_one" in prompt and "closed grouping domain" in prompt
    product = auxiliary_summary_product(outline.steps[1].step_id)
    assert product in prompt and "semantic_role=custom" in prompt
    assert "identity" in prompt and "raw_inputs" in prompt
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=context)
    error = caught.value
    assert error.reason_code == "progressive_analysis_module_unavailable"
    assert "custom_analysis" in str(error) and product in str(error)
    assert error.details["available_modules"] == list(
        progressive_module_ids_for_analysis_types((outline.analysis_type,))
    )
    assert len(error.details["available_modules"]) <= 11
    assert error.details["repair_scope"] == "current_step_only"
    assert all(module in str(error) for module in error.details["available_modules"])
    repaired = skeleton.model_dump(mode="json")
    repaired["steps"][1]["outputs"][0]["product_id"] = product
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(repaired), context=context,
    )
    assert plan.steps[1].method == "descriptive_cohort_summary"
    assert plan.steps[1].expected_outputs == [product]
    assert skeleton.steps[1].custom_method == repaired["steps"][1]["custom_method"]
    assert plan.steps[2].inputs == [measure, "artifact:analysis_cohort"]
    if kind == "continuous":
        assert plan.steps[-1].expected_outputs == ["table:continuous_outcome_summary"]


def test_association_still_offers_and_compiles_table_one():
    context = _context()
    outline = ProgressivePlanOutline.model_validate(_outline_payload())
    prompt = _prompt(context, outline)
    assert "Available modules for this analysis type" in prompt
    assert "table_one" in prompt and "adjusted_association" in prompt
    plan, _ = compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate(_payload()), context=context)
    assert plan.steps[1].table_one_spec is not None


def test_a_module_without_a_compatible_shape_requires_outline_repair():
    context, _, skeleton = _case()
    steps = list(skeleton.steps)
    steps[1] = steps[1].model_copy(update={"planned_analysis_role": "sensitivity"})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton.model_copy(update={"steps": steps}), context=context)
    assert caught.value.reason_code == "progressive_analysis_module_unavailable"
    assert caught.value.details["repair_scope"] == "outline_required"
    assert "Do not substitute" in str(caught.value)


def test_outline_rejection_already_names_the_family_menu():
    context, outline, _ = _case()
    steps = list(outline.steps)
    steps[1] = steps[1].model_copy(update={"module_id": "report"})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline.model_copy(update={"steps": steps}), analysis_types=[outline.analysis_type],
            variable_names=[v.name for v in context.variables], allowed_literature_citation_keys=(),
        )
    assert caught.value.reason_code == "progressive_outline_module_unavailable"
    assert "report" in str(caught.value)
    assert "Available modules for this analysis type" in str(caught.value)
    assert "table_one" in str(caught.value)


@pytest.mark.parametrize("invalid_attempts", [1, 3])
def test_compile_retry_budget_and_history_preserve_the_rejections(invalid_attempts):
    context, outline, skeleton = _case()
    payload = skeleton.model_dump(mode="json")
    materializations = _materialization_payloads(payload)
    for response, step in zip(materializations, outline.steps, strict=True):
        response["outline_step_sha256"] = canonical_sha256(step.model_dump(mode="json"))
    foundation = _foundation_payload(payload)
    foundation["outline_sha256"] = canonical_sha256(outline.model_dump(mode="json"))
    repaired = deepcopy(materializations[1])
    repaired["step"]["outputs"][0]["product_id"] = auxiliary_summary_product(outline.steps[1].step_id)
    responses = [outline.model_dump(mode="json"), foundation, materializations[0],
                 *[materializations[1]] * invalid_attempts, repaired, *materializations[2:]]
    client = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    client.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(client)
    if invalid_attempts == 1:
        plan = agent.run(context)
        assert plan.steps[1].expected_outputs == [auxiliary_summary_product(outline.steps[1].step_id)]
        assert agent.last_result.facts.prompt_metrics["compile_revision_count"] == 1
        assert len(client.calls) == 7
    else:
        with pytest.raises(ProgressivePlanCompileError) as caught:
            agent.run(context)
        assert caught.value.reason_code == "progressive_analysis_module_unavailable"
        facts = progressive_planner_failure_facts(caught.value)
        assert facts.prompt_metrics["compile_revision_count"] == 2
        assert [a.revision for a in facts.compile_failure_attempts] == [0, 1, 2]
        assert len(client.calls) == 6
        history = caught.value.easyicu_compile_retry_metadata
        assert [row["attempt"] for row in history] == [1, 2, 3]
        assert {row["reason_code"] for row in history} == {"progressive_analysis_module_unavailable"}
        assert "prespecified_population_description" not in json.dumps(history)
        assert "invented_profile" not in json.dumps(history)
    repair_prompt = client.calls[4][0][-1].content
    assert "HOST COMPILER OBSERVATION" in repair_prompt
    assert "available_modules" in repair_prompt
    assert auxiliary_summary_product(outline.steps[1].step_id) in repair_prompt


def test_an_unavailable_locked_module_stops_without_spending_step_retries(monkeypatch):
    """A capability refusal after sealing cannot be fixed by rewriting a step."""
    from easyicu.research_agent.agents import progressive_planner as owner

    context, outline, skeleton = _case()
    responses = _materialization_payloads(skeleton.model_dump(mode="json"))
    for response, step in zip(responses, outline.steps, strict=True):
        response["outline_step_sha256"] = canonical_sha256(step.model_dump(mode="json"))
    foundation = _foundation_payload(skeleton.model_dump(mode="json"))
    foundation["outline_sha256"] = canonical_sha256(outline.model_dump(mode="json"))
    compile_prefix = owner.compile_progressive_prefix

    def unavailable(state, materialization, **kwargs):
        if materialization.step.step_id == outline.steps[1].step_id:
            steps = list(skeleton.steps)
            steps[1] = steps[1].model_copy(update={"planned_analysis_role": "sensitivity"})
            return compile_progressive_plan(skeleton=skeleton.model_copy(update={"steps": steps}), context=context)
        return compile_prefix(state, materialization, **kwargs)

    monkeypatch.setattr(owner, "compile_progressive_prefix", unavailable)
    client = ScriptedMockLLMClient([
        json.dumps(item) for item in [outline.model_dump(mode="json"), foundation, *responses]
    ])
    agent = ProgressivePlannerAgent(client)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        agent.run(context)
    assert caught.value.reason_code == "progressive_analysis_module_unavailable"
    assert len(client.calls) == 4
    assert progressive_planner_failure_facts(caught.value).prompt_metrics["compile_revision_count"] == 0
