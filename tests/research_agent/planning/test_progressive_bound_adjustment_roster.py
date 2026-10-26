"""A bound adjustment roster gives the Planner an actionable local repair."""

from copy import deepcopy
import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.adjustment_authority import (
    AdjustmentSetAuthority, adjusted_model_term_planning_authority,
)
from easyicu.research_agent.planning.progressive_artifacts import ProgressiveCompilerFinding
from easyicu.research_agent.planning.progressive_compiler import (
    _compile_adjusted_association, compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError, ProgressivePlanOutline, ProgressivePlanSkeleton,
    ProgressiveSkeletonStep,
)
from easyicu.research_agent.planning.requirement_coverage import stated_subgroup_coverage
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.outbound import outbound_safe_context_payload
from easyicu.research_agent.schema import VariableRole

from .progressive_planner_fixtures import (
    _context, _foundation_payload, _materialization_payloads, _outline_payload, _payload,
)


REASON = "progressive_adjustment_roster_mismatch"


def _case(*, subgroup=True, exact=None, selectable=False):
    context = _context()
    context.variable("age_years").source_concept = "age"
    context.variable("sex_code").source_concept = "sex"
    context = context.model_copy(update={
        "research_question": "Relate the synthetic exposure to the outcome within recorded sex strata."
        if subgroup else "Relate the synthetic exposure to the outcome.",
    })
    constraints = json.loads(context.user_preferences.data_constraints) if selectable else {}
    if subgroup:
        constraints["stated_subgroups"] = [{
            "variable": "sex", "concept": "sex", "words": "within recorded sex strata",
            "source_status": "available",
        }]
    context.user_preferences.data_constraints = json.dumps(constraints)
    if exact is not None:
        preferences = context.user_preferences
        preferences.covariate_selection = "exact"
        preferences.covariate_authority = "user"
        preferences.covariates = exact
        preferences.covariate_rationales = {name: "Prespecified baseline adjustment." for name in exact}
        preferences.covariate_temporal_roles = {name: "baseline_static" for name in exact}
    return context, _payload()


def _compile(context, step):
    return _compile_adjusted_association(
        context=context, variables={v.name: v for v in context.variables},
        step=ProgressiveSkeletonStep.model_validate(step), step_index=4,
    )[0]


def _prompt(context, payload, observation=None):
    outline = ProgressivePlanOutline.model_validate(_outline_payload(payload))
    step = outline.steps[4]
    return ProgressivePlannerAgent._materialization_prompt(
        context=context, outline=outline, outline_step=step,
        outline_step_sha256=canonical_sha256(step.model_dump(mode="json")),
        variables=[v.name for v in context.variables], action_rows=[],
        allowed_literature_citation_keys=(), know_how_context="", planning_contract_context="",
        prefix_summary=[], available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
        compiler_observation=observation,
    )


def test_continuous_landmark_replay_names_the_extra_stratifier_before_contract_validation():
    context, payload = _case()
    rename = {"exposure_flag": "hgb_min", "outcome_flag": "death", "sex_code": "sex", "age_years": "age"}
    for variable in context.variables:
        variable.name = rename[variable.name]
    exposure = context.variable("hgb_min")
    exposure.role = "lab"
    exposure.dtype = "float64"
    exposure.observed_domain = None
    exposure.analysis_window = "icu_admission[0,24]h"
    context = context.model_copy(update={"primary_exposure": "hgb_min", "target_outcome": "death"})
    context.cohort.outcome_columns = ["death"]
    context.user_preferences.landmark_hours = 24
    step = payload["steps"][4]
    step.update(primary_exposure="hgb_min", outcome="death", raw_inputs=["hgb_min", "death", "sex", "age"])
    step["model_terms"] = [
        {"name": "hgb_min", "role": "exposure", "coding": "continuous"},
        {"name": "sex", "role": "covariate", "coding": "binary", "reference_level_index": 0,
         "clinical_rationale": "Prespecified subgroup in the synthetic study."},
        {"name": "age", "role": "covariate", "coding": "continuous",
         "clinical_rationale": "Baseline demographic adjustment."},
    ]
    authority = AdjustmentSetAuthority.from_context(context)
    assert authority.operational_covariates == ("age",)
    assert authority.rule == "family_minimal_adjustment/1"
    before = deepcopy(step)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, step)
    error = caught.value
    assert error.reason_code == REASON
    assert error.details["bound_covariates"] == ["age"]
    assert error.details["planned_covariates"] == ["sex", "age"]
    assert error.details["missing_covariates"] == []
    assert error.details["extra_covariates"] == ["sex"]
    assert error.details["subgroup_covariates"] == ["sex"]
    assert error.details["repair_scope"] == "current_step_only"
    assert (error.step_id, error.step_index, error.path) == ("05_primary", 4, "model_terms")
    replay = ProgressiveCompilerFinding.from_details(error.details)
    assert "sex" in replay.message and "age" in replay.message
    assert "association.effect_modification" in replay.message
    assert step == before
    repaired = deepcopy(step)
    repaired["model_terms"].pop(1)
    requirement = _compile(context, repaired)
    assert requirement.covariates == ["age"]
    assert requirement.exposure_source == "hgb_min"


@pytest.mark.parametrize("roster,missing,extra", [
    ([], ["age_years", "sex_code"], []),
    (["sex_code"], ["age_years"], []),
    (["sex_code", "age_years"], [], []),
])
def test_missing_or_reordered_exact_terms_require_planner_repair(roster, missing, extra):
    context, payload = _case(subgroup=False)
    step = payload["steps"][4]
    terms = {term["name"]: term for term in step["model_terms"]}
    step["model_terms"] = [terms["exposure_flag"], *[terms[name] for name in roster]]
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, step)
    assert caught.value.reason_code == REASON
    assert caught.value.details["missing_covariates"] == missing
    assert caught.value.details["extra_covariates"] == extra
    assert caught.value.details["bound_covariates"] == ["age_years", "sex_code"]


@pytest.mark.parametrize("roster", [[], ["sex_code"], ["sex_code", "age_years"]])
def test_user_rosters_including_no_adjustment_keep_priority(roster):
    context, payload = _case(exact=roster)
    step = payload["steps"][4]
    terms = {term["name"]: term for term in step["model_terms"]}
    step["model_terms"] = [terms["exposure_flag"], *[terms[name] for name in roster]]
    assert _compile(context, step).covariates == roster


def test_an_extra_term_is_rejected_even_when_the_bound_roster_is_empty():
    context, payload = _case(subgroup=False, exact=[])
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, payload["steps"][4])
    assert caught.value.reason_code == REASON
    assert caught.value.details["bound_covariates"] == []
    assert caught.value.details["extra_covariates"] == ["age_years", "sex_code"]
    assert caught.value.details["subgroup_covariates"] == []


def test_a_large_named_mismatch_still_fits_the_replay_contract():
    context, payload = _case(subgroup=False, exact=[])
    step = payload["steps"][4]
    names = [f"baseline_{index}_" + "x" * 90 for index in range(20)]
    context.variables.extend(context.variable("age_years").model_copy(update={"name": name}) for name in names)
    step["raw_inputs"].extend(names)
    step["model_terms"] = [step["model_terms"][0], *[{
        "name": name, "role": "covariate", "coding": "continuous", "clinical_rationale": "Baseline descriptor.",
    } for name in names]]
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, step)
    assert caught.value.reason_code == REASON
    assert caught.value.details["extra_covariates"] == names
    assert ProgressiveCompilerFinding.from_details(caught.value.details).reason_code == REASON


def test_planner_selectable_terms_keep_their_rationale_and_timing_owner():
    context, payload = _case(subgroup=False, selectable=True)
    assert AdjustmentSetAuthority.from_context(context).selection == "planner_selectable"
    requirement = _compile(context, payload["steps"][4])
    assert requirement.covariates == ["age_years", "sex_code"]
    assert requirement.covariate_rationales == {
        term["name"]: term["clinical_rationale"]
        for term in payload["steps"][4]["model_terms"] if term["role"] == "covariate"
    }
    assert set(requirement.covariate_temporal_roles.values()) == {"baseline_static"}


def test_a_scientific_user_roster_is_compared_using_its_bound_physical_columns():
    context, payload = _case(exact=["age"])
    context.user_preferences.covariate_operationalizations = {"age": "age_years"}
    payload["steps"][4]["model_terms"] = payload["steps"][4]["model_terms"][:2]
    requirement = _compile(context, payload["steps"][4])
    assert requirement.covariates == ["age_years"]
    assert set(requirement.covariate_rationales) == {"age_years"}


def test_the_non_stratified_default_still_compiles_the_full_plan():
    context, payload = _case(subgroup=False)
    plan, _ = compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context)
    assert plan.steps[4].model_requirements[0].covariates == ["age_years", "sex_code"]
    AdjustmentSetAuthority.from_context(context).validate_plan(plan)


@pytest.mark.parametrize("exact", [None, []])
def test_first_attempt_sees_the_bound_roster_in_outbound_context_and_term_authority(exact):
    context, payload = _case(exact=exact)
    roster = ["age_years"] if exact is None else []
    outbound = outbound_safe_context_payload(context)
    assert outbound["adjustment_authority"]["operational_covariates"] == roster
    authority = adjusted_model_term_planning_authority(context, [v.name for v in context.variables])
    assert [row["name"] for row in authority["eligible_covariates"]] == roster
    assert {row["name"]: row["reason"] for row in authority["excluded_covariates"]}["sex_code"] == "outside_bound_adjustment_roster"
    prompt = _prompt(context, payload)
    assert '"operational_covariates":' + json.dumps(roster, separators=(",", ":")) in prompt
    assert "association.effect_modification" in prompt
    assert "custom_analysis" in prompt and "raw_inputs" in prompt
    assert "every bound covariate in order" in prompt
    assert "Only planner_selectable permits choosing a subset." in prompt


def test_repair_prompt_carries_named_differences_and_corrected_skeleton_compiles():
    context, payload = _case()
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, payload["steps"][4])
    prompt = _prompt(context, payload, caught.value.details)
    assert '"extra_covariates":["sex_code"]' in prompt
    assert '"bound_covariates":["age_years"]' in prompt
    assert '"subgroup_covariates":["sex_code"]' in prompt
    assert "HOST COMPILER OBSERVATION FOR THIS CURRENT STEP" in prompt
    payload["steps"][4]["model_terms"] = payload["steps"][4]["model_terms"][:2]
    payload["steps"].insert(6, _subgroup_step(payload))
    plan, _ = compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context)
    assert plan.steps[4].model_requirements[0].covariates == ["age_years"]
    AdjustmentSetAuthority.from_context(context).validate_plan(plan)
    assert stated_subgroup_coverage(context, plan).complete


def _subgroup_step(payload):
    step = deepcopy(payload["steps"][5])
    step.update(
        step_id="06_subgroup", planned_analysis_role="secondary",
        objective="Estimate the synthetic association within each recorded sex stratum.",
        scientific_action_id="association.effect_modification", custom_method="effect_modification",
        raw_inputs=["exposure_flag", "outcome_flag", "age_years", "sex_code"],
        outputs=[{"product_id": "table:subgroup_association", "semantic_role": "custom"}],
        sensitivity_spec_ids=[],
    )
    step.pop("functional_form_spec", None)
    return step


@pytest.mark.parametrize("needs_repair", [False, True])
@pytest.mark.parametrize("continuous", [False, True])
def test_stratified_plan_completes_the_progressive_loop_with_a_distinct_subgroup_owner(needs_repair, continuous):
    context, payload = _case()
    payload["steps"][4]["model_terms"] = payload["steps"][4]["model_terms"][:2]
    if continuous:
        exposure = context.variable("exposure_flag")
        exposure.role = VariableRole.LAB
        exposure.dtype = "float64"
        exposure.observed_domain = None
        exposure.analysis_window = "icu_admission[0,24]h"
        context.user_preferences.landmark_hours = 24
        payload["steps"][4]["model_terms"][0].update(coding="continuous", reference_level_index=None)
        payload["steps"][1].update(table_one_group_by="sex_code", table_one_variables=[
            {"name": "age_years", "summary": "median_iqr"},
            {"name": "exposure_flag", "summary": "median_iqr"},
        ])
        payload["steps"][2]["module_id"] = "absolute_risk_context"
        payload["steps"][2]["population_scope"] = "analysis_cohort"
        payload["steps"][6]["product_inputs"][0]["product_id"] = "table:absolute_risk_context"
    payload["steps"].insert(6, _subgroup_step(payload))
    # The existing article gate requires a functional-form check for continuous adjustment.
    sensitivity = payload["steps"][5]
    sensitivity.update(
        step_id="06_functional_form_sensitivity", custom_method="restricted_cubic_spline_sensitivity",
        functional_form_spec={"target_column": "age_years", "knot_quantiles": [0.1, 0.5, 0.9]},
        sensitivity_spec_ids=["age_restricted_cubic_spline_vs_linear"],
    )
    outline = _outline_payload(payload)
    if continuous:
        outline["steps"][2]["population_scope"] = "analysis_cohort"
    outline["design_selection"]["candidates"][0]["required_variables"].extend(["age_years", "sex_code"])
    outline = ProgressivePlanOutline.model_validate(outline)
    foundation = _foundation_payload(payload)
    foundation["outline_sha256"] = canonical_sha256(outline.model_dump(mode="json"))
    materializations = _materialization_payloads(payload)
    for response, step in zip(materializations, outline.steps, strict=True):
        response["outline_step_sha256"] = canonical_sha256(step.model_dump(mode="json"))
    responses = [outline.model_dump(mode="json"), foundation, *materializations[:4]]
    if needs_repair:
        invalid = deepcopy(materializations[4])
        invalid["step"]["model_terms"].insert(1, _payload()["steps"][4]["model_terms"][2])
        responses.append(invalid)
    responses.extend(materializations[4:])
    client = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    client.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(client)
    plan = agent.run(context)
    assert agent.last_result.facts.prompt_metrics["compile_revision_count"] == int(needs_repair)
    assert plan.steps[4].model_requirements[0].covariates == ["age_years"]
    assert stated_subgroup_coverage(context, plan).complete
    subgroup = next(step for step in plan.steps if step.step_id == "06_subgroup")
    assert subgroup.scientific_action_id == "association.effect_modification"
    assert "sex_code" in subgroup.inputs
    prompts = [messages[-1].content for messages, _ in client.calls]
    assert '"operational_covariates":["age_years"]' in prompts[6]
    assert "association.effect_modification" in prompts[6]
    if needs_repair:
        assert REASON in prompts[7]
        assert '"extra_covariates":["sex_code"]' in prompts[7]
        assert '"missing_covariates":[]' in prompts[7]
        assert '"subgroup_covariates":["sex_code"]' in prompts[7]
