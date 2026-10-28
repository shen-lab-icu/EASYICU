"""Default checks are visible, executable, and leave the primary model intact."""

import json

import pytest

from easyicu.research_agent.execution.runners.association_binary_sensitivity_executor import (
    resolve_binary_sensitivity_variant,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.baseline_association_template import build_baseline_association_skeleton
from easyicu.research_agent.planning.family_spec.contract import FamilyPlanSpec
from easyicu.research_agent.planning.family_spec.landmark_categorical_template import build_landmark_categorical_skeleton
from easyicu.research_agent.planning.scientific_review import _sensitivity_facts
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.schema import ConceptDescriptor
from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime

from .family_spec_fixtures import _context, _request, _run, _spec_payload


def _baseline_request(context):
    return build_family_spec_request(context, analysis_types=("association_study",),
                                    variable_roster=[v.name for v in context.variables],
                                    allowed_literature_citation_keys=())


def _draft(family, *, subgroup=False):
    if family == "baseline":
        context = baseline_context("chronic_flag", covariates=("age",), subgroup="sex" if subgroup else None)
        request = _baseline_request(context)
        spec = FamilyPlanSpec(
            family_id=request.family_id, request_sha256=request.request_sha256,
            adjustment_set=[], reader_display_labels=[], comparator_applications=[],
            roster_decision_note="Preserve the exact synthetic adjustment roster.",
        )
        return context, request, build_baseline_association_skeleton(request, spec)
    context = _context()
    if subgroup:
        constraints = json.loads(context.user_preferences.data_constraints)
        constraints["stated_subgroups"] = [dict(variable="sex", concept="sex", words="within strata", source_status="available")]
        context.user_preferences.data_constraints = json.dumps(constraints)
    request = _request(context)
    return context, request, build_landmark_categorical_skeleton(request, FamilyPlanSpec.model_validate(_spec_payload(request)))


def test_baseline_default_spline_is_counted_and_survives_signed_binding():
    context = baseline_context("chronic_flag", covariates=("age",))
    attempt, _ = baseline_plan_attempt(context)
    assert attempt.output is not None
    primary = attempt.output.steps[3].model_requirements[0].model_dump(mode="json")
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    plan, _ = runtime.bind_plan(attempt.output)
    child, = [step for step in plan.steps if step.functional_form_spec is not None]
    variant = resolve_binary_sensitivity_variant(child, plan=plan)
    assert variant is not None and variant.target_column == "age"
    assert plan.steps[3].model_requirements[0].model_dump(mode="json") == primary
    assert "functional_form" in _sensitivity_facts(context, plan, runtime_authority=runtime.current_case)["typed_executable"]


@pytest.mark.parametrize("family", ["baseline", "landmark"])
def test_checks_have_edges_from_the_primary_into_both_consumers(family):
    _, _, draft = _draft(family)
    steps = {step.step_id: step for step in draft.outline.steps}
    checks = [step for step in steps.values() if step.step_id.endswith("_functional_form")]
    assert checks
    for step in checks:
        assert "adjusted_association" in step.depends_on
        for consumer in ("visualization", "report"):
            assert step.step_id in steps[consumer].depends_on
            if consumer == "report":
                materialized = next(item.step for item in draft.materializations if item.step.step_id == consumer)
                assert step.step_id in {ref.producer_step_id for ref in materialized.product_inputs}


@pytest.mark.parametrize("family", ["baseline", "landmark"])
def test_only_typed_modifiers_get_a_visible_secondary_step(family):
    _, request, draft = _draft(family, subgroup=True)
    assert request.subgroup_columns == ["sex"]
    step, = [item.step for item in draft.materializations
             if item.step.scientific_action_id == "association.effect_modification"]
    assert step.planned_analysis_role == "secondary"
    assert "sex" in step.raw_inputs
    assert "adjusted_association" in step.depends_on
    for consumer in ("visualization", "report"):
        assert step.step_id in next(item for item in draft.outline.steps if item.step_id == consumer).depends_on
    _, _, without = _draft(family)
    assert not any(item.step.scientific_action_id == "association.effect_modification" for item in without.materializations)


def test_typed_first_stay_is_preserved_without_enlarging_adjustment():
    context = baseline_context("chronic_flag", covariates=("age",))
    context.variables.append(ConceptDescriptor(name="initial_episode", role="other", dtype="bool"))
    context.user_preferences.sensitivity_specs = [PrespecifiedSensitivitySpec(
        spec_id="initial_episode_only", axis="repeated_stays", strategy="first_stay",
        execution_variables=("initial_episode",),
    )]
    request = _baseline_request(context)
    assert request.first_stay is not None
    attempt, _ = baseline_plan_attempt(context)
    assert attempt.output is not None
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    plan, _ = runtime.bind_plan(attempt.output)
    child, = [step for step in plan.steps if step.method == "first_stay_association"]
    assert resolve_binary_sensitivity_variant(child, plan=plan).restriction_column == "initial_episode"
    assert "initial_episode" not in plan.steps[3].model_requirements[0].covariates


def test_missing_inputs_have_typed_omissions_and_never_invent_definitions():
    _, _, draft = _draft("baseline")
    reasons = {item.analysis: item.reason_code for item in draft.analysis_omissions}
    assert reasons["first_stay"] == "typed_restriction_unavailable"
    assert reasons["alternate_exposure"] == "typed_alternative_unavailable"
    assert reasons["alternate_definition"] == "typed_alternative_unavailable"
    assert reasons["alternative_adjustment"] == "strategy_not_supported"


def test_interaction_parent_does_not_get_an_incompatible_refit():
    _, _, draft = _draft("baseline", subgroup=True)
    assert not any(item.step.functional_form_spec is not None for item in draft.materializations)
    assert any(item.reason_code == "primary_model_not_supported" for item in draft.analysis_omissions)


def test_landmark_typed_modifiers_stay_in_the_family_route():
    context, request, _ = _draft("landmark", subgroup=True)
    _, attempt = _run(context, [json.dumps(_spec_payload(request))])
    assert attempt.output is not None
    assert any(step.scientific_action_id == "association.effect_modification" for step in attempt.output.steps)


def test_landmark_first_stay_has_one_exact_restriction_coordinate():
    context = _context()
    request = _request(context)
    _, attempt = _run(context, [json.dumps(_spec_payload(request))])
    step = next(step for step in attempt.output.steps if step.method == "first_stay_association")
    variant = resolve_binary_sensitivity_variant(step, plan=attempt.output)
    assert variant is not None
    assert variant.restriction_column == "first_stay_flag"
    assert "readmission" in _sensitivity_facts(context, attempt.output)["typed_executable"]


def test_landmark_replay_spec_ids_do_not_hide_unexecuted_alternatives():
    context = _context()
    request = _request(context)
    _, attempt = _run(context, [json.dumps(_spec_payload(request))])
    omissions = attempt.facts.prompt_metrics["family_spec_analysis_omissions"]
    for spec in request.alternate_exposures:
        assert any(item["spec_id"] == spec.spec_id and item["reason_code"] == "strategy_not_supported"
                   for item in omissions)
    assert "exposure_definition" not in _sensitivity_facts(context, attempt.output)["typed_executable"]


def test_landmark_without_continuous_adjustment_records_why_no_form_check_runs():
    request = _request(_context()).model_copy(update={"exact_roster": ["sex"]})
    draft = build_landmark_categorical_skeleton(request, FamilyPlanSpec.model_validate(_spec_payload(request)))
    assert not any(item.step.functional_form_spec for item in draft.materializations)
    assert any(item.analysis == "restricted_cubic_spline" and item.reason_code == "no_continuous_adjustment"
               for item in draft.analysis_omissions)


def _landmark_runtime(plan, request):
    from easyicu.research_agent.authority.current_case_scientific_runtime import build_current_case_scientific_runtime_authority

    requirement = plan.steps[3].model_requirements[0]
    return build_current_case_scientific_runtime_authority(dict(
        schema_version="easyicu.landmark_categorical_association_runtime_authority/1",
        authority_kind="landmark_categorical_association", protocol_content_sha256="b" * 64,
        cohort_method="signed_landmark_analysis_cohort", primary_method="signed_landmark_categorical_association",
        plan_intent="Estimate a synthetic categorical association.", landmark_spec_id=request.landmark_spec_id,
        cohort_product="artifact:analysis_cohort", cohort_flow_product="table:cohort_flow",
        primary_product="table:adjusted_association_estimates", exposure_column=request.primary_exposure,
        exposure_kind="ordinal", exposure_levels=request.exposure_levels, exposure_reference_level=requirement.exposure_reference_level,
        primary_contrast_level=requirement.primary_contrast_level, outcome_column=request.outcome,
        event_time_column=request.event_time_column, observation_duration_column=request.observation_duration_column,
        observation_duration_unit="hours", landmark_hours=24, exclude_negative_event_times=True, require_alive_at_landmark=True,
        required_adjustment_columns=requirement.covariates,
        categorical_adjustment_columns=[term.name for term in requirement.model_terms if term.role == "covariate" and term.coding != "continuous"],
        dependence=requirement.dependence.model_dump(mode="json") if requirement.dependence else None,
        interpretation="descriptive_prognostic_association_not_causal",
    ))


def test_landmark_modifiers_execute_on_the_signed_primary_population(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from easyicu.research_agent.execution.owner_declaration import resolve_standard_executor
    from easyicu.research_agent.execution.runners.association_effect_modification import fit_baseline_association, run_effect_modification

    context, request, _ = _draft("landmark", subgroup=True)
    _, attempt = _run(context, [json.dumps(_spec_payload(request))])
    authority = _landmark_runtime(attempt.output, request)
    plan = authority.bind_plan(attempt.output)
    step = next(step for step in plan.steps if step.scientific_action_id == "association.effect_modification")
    assert resolve_standard_executor(step, plan=plan, current_case_scientific_runtime_authority=authority).claimed_by == "prespecified_effect_modification"
    primary = authority.governed_primary_step(plan).model_requirements[0]
    before = plan.model_dump_json()
    rng = np.random.default_rng(719)
    n = 256
    frame = pd.DataFrame({
        request.primary_exposure: np.tile(np.arange(4), n // 4), "sex": rng.integers(0, 2, n),
        "age": rng.normal(60, 8, n), "comorbidity_index": rng.normal(2, 0.4, n),
        request.outcome: rng.binomial(1, 0.35, n),
        primary.dependence.group_source: [f"p{i}:s1" for i in range(n)],
    })
    model = SimpleNamespace(exposure_column=primary.exposure_source, outcome_column=primary.outcome,
                            exposure_encoding="treatment_contrast", exposure_reference_level=primary.exposure_reference_level,
                            primary_contrast_level=primary.primary_contrast_level, subgroup_terms=(), dependence=primary.dependence)
    _, _, _, rows, _ = fit_baseline_association(frame, requirement=primary, authority=model)
    primary_table = pd.DataFrame(rows)
    summary = run_effect_modification(step_id=step.step_id, plan=plan, authority=authority,
                                      frame=frame, primary_table=primary_table, output_dir=tmp_path)
    assert summary["n_total"] == n and set(summary["interaction_tests"]) == {"sex"}
    result = pd.read_csv(tmp_path / "prespecified_effect_modification.csv")
    assert result["analysis_role"].eq("secondary").all()
    assert set(result["stratum"]) == {0, 1}
    assert plan.model_dump_json() == before
    wrong = primary_table.assign(n=n + 1)
    with pytest.raises(ValueError, match="primary model population"):
        run_effect_modification(step_id=step.step_id, plan=plan, authority=authority,
                                frame=frame, primary_table=wrong, output_dir=tmp_path)


def test_modifier_binding_rejects_a_changed_primary_or_missing_terms():
    from easyicu.research_agent.contracts.model_terms import EffectModificationBinding

    context = baseline_context("chronic_flag", covariates=("age",), subgroup="sex")
    attempt, _ = baseline_plan_attempt(context)
    primary = attempt.output.steps[3].model_requirements[0]
    step = next(step for step in attempt.output.steps if step.scientific_action_id == "association.effect_modification")
    assert EffectModificationBinding.requirement_for(step, primary).effect_modifiers[0].name == "sex"
    with pytest.raises(ValueError, match="digest mismatch"):
        EffectModificationBinding.requirement_for(step, primary.model_copy(update={"outcome": "different_response"}))
    with pytest.raises(ValueError, match="one typed modifier"):
        EffectModificationBinding.requirement_for(step.model_copy(update={"icu_rule_refs": []}), primary)


def test_baseline_rejects_a_companion_with_a_different_spline_target():
    context = baseline_context("chronic_flag", covariates=("age",))
    attempt, _ = baseline_plan_attempt(context)
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    plan, _ = runtime.bind_plan(attempt.output)
    child = next(step for step in plan.steps if step.functional_form_spec)
    child.functional_form_spec = child.functional_form_spec.model_copy(update={"target_column": "chronic_flag"})
    with pytest.raises(ValueError, match="step roster"):
        runtime.validate_plan(plan)


def test_modifier_binding_preserves_the_signed_roster_order():
    context = baseline_context("chronic_flag", covariates=("age",), subgroup="sex")
    context.variables.append(ConceptDescriptor(name="region_group", role="other", dtype="int64",
                                              observed_domain={"levels": [0, 1], "n_unique": 2, "is_binary": True}))
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_subgroups"].append(dict(variable="region_group", concept="region_group",
                                               words="within strata", source_status="available"))
    context.user_preferences.data_constraints = json.dumps(constraints)
    attempt, _ = baseline_plan_attempt(context)
    assert attempt.output is not None
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    plan, _ = runtime.bind_plan(attempt.output)
    runtime.validate_plan(plan)


def test_unavailable_requested_alternative_has_its_own_typed_omission():
    context = baseline_context("chronic_flag", covariates=("age",))
    context.user_preferences.sensitivity_specs = [PrespecifiedSensitivitySpec(
        spec_id="alternate_endpoint", axis="outcome_definition", strategy="alternate_definition",
        execution_variables=("other_endpoint",),
    )]
    attempt, _ = baseline_plan_attempt(context)
    assert attempt.output is not None
    omissions = attempt.facts.prompt_metrics["family_spec_analysis_omissions"]
    assert any(item["spec_id"] == "alternate_endpoint" and item["reason_code"] == "typed_alternative_unavailable"
               for item in omissions)


def test_landmark_missing_category_policy_omits_incompatible_interactions():
    from easyicu.research_agent.schema import MissingnessProfile

    context, _, _ = _draft("landmark", subgroup=True)
    context = context.model_copy(update={"variables": [item.model_copy(update={"missingness": MissingnessProfile(
        fraction_missing=0.4, n_missing=40, n_total=100,
    )}) if item.name == "age" else item for item in context.variables]})
    request = _request(context)
    _, attempt = _run(context, [json.dumps(_spec_payload(request))])
    assert attempt.output is not None
    assert not any(step.method == "prespecified_effect_modification" for step in attempt.output.steps)
    omissions = attempt.facts.prompt_metrics["family_spec_analysis_omissions"]
    assert any(item["analysis"] == "effect_modification" and item["reason_code"] == "primary_model_not_supported"
               for item in omissions)
