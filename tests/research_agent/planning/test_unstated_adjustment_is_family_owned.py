"""Unstated adjustment is a reproducible family decision, before the LLM call."""

import json

import pytest

from easyicu.research_agent.planning.family_spec import (
    FamilyPlanSpec, FamilySpecError, build_family_spec_request, validate_family_plan_spec,
)
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.support.baseline_association import baseline_context

from .family_spec_fixtures import _context, _request as _full_request, _run, _spec_payload


def _study(family="baseline", *, modifier=None):
    if family == "baseline":
        context = baseline_context("chronic_flag", subgroup=modifier)
        context.user_preferences.covariate_selection = "planner_selectable"
        context = context.model_copy(update={"research_question": "Relate the chronic attribute to hospital death."})
    else:
        context = _context(exact=False)
        constraints = json.loads(context.user_preferences.data_constraints)
        constraints["stated_adjustments"] = []
        constraints["stated_subgroups"] = ([{
            "variable": modifier, "concept": modifier, "words": "within the stated strata",
            "source_status": "available",
        }] if modifier else [])
        context.user_preferences.data_constraints = json.dumps(constraints)
        if family == "spline":
            context.variables.append(ConceptDescriptor(
                name="assay_max", role="lab", dtype="float64", analysis_window="icu_admission[0,24]h",
            ))
            context = context.model_copy(update={"primary_exposure": "assay_max"})
        context = context.model_copy(update={"research_question": "Relate the first-day assay to hospital death after 24 hours."})
    return context


def _request(context, roster=None):
    return build_family_spec_request(
        context, analysis_types=("association_study",),
        variable_roster=roster if roster is not None else [v.name for v in context.variables],
        allowed_literature_citation_keys=(), required_primary_cohort_selection_mode="all_input_rows",
    )


@pytest.mark.parametrize("family", ["baseline", "categorical", "spline"])
def test_unstated_adjustment_is_bound_before_the_planner(family):
    context = _study(family)
    original = context.model_dump(mode="json")
    first, second = _request(context), _request(context.model_copy(deep=True))
    assert first.adjustment_selection == "exact"
    assert first.exact_roster == second.exact_roster == ["age", "sex"]
    assert first.request_sha256 == second.request_sha256
    assert set(first.exact_rationales) == {"age", "sex"}
    assert all("family_minimal_adjustment/1" in reason for reason in first.exact_rationales.values())
    assert first.exact_temporal_roles == {"age": "baseline_static", "sex": "baseline_static"}
    assert context.model_dump(mode="json") == original


@pytest.mark.parametrize("family", ["baseline", "categorical", "spline"])
def test_a_default_covariate_that_is_a_modifier_has_its_own_model_role(family):
    assert _request(_study(family, modifier="sex")).exact_roster == ["age"]


@pytest.mark.parametrize("exposure,expected", [("age", ["sex"]), ("sex", ["age"]), ("age_group", ["sex"])])
def test_exposure_and_group_sources_are_never_default_adjustments(exposure, expected):
    context = baseline_context(exposure)
    context.user_preferences.covariate_selection = "planner_selectable"
    assert _request(context).exact_roster == expected


def test_available_defaults_have_a_fixed_order_independent_of_retrieval_order():
    context = _study()
    context.variables.reverse()
    assert _request(context).exact_roster == ["age", "sex"]
    with pytest.raises(FamilySpecError, match="covariate_unavailable"):
        _request(context, [v.name for v in context.variables if v.name != "age"])


@pytest.mark.parametrize("roster", [[], ["sex"], ["sex", "age"]])
def test_an_explicit_exact_roster_keeps_priority_including_unadjusted(roster):
    context = baseline_context("chronic_flag", covariates=roster)
    context.user_preferences.covariate_authority = "user"
    assert _request(context).exact_roster == roster


def test_question_stated_adjustment_keeps_priority_over_defaults():
    context = _study()
    context.user_preferences.data_constraints = json.dumps({"stated_adjustments": [{
        "variable": "sex", "concept": "sex", "words": "adjust for sex", "source_status": "available",
    }]})
    assert _request(context).exact_roster == ["sex"]


@pytest.mark.parametrize("change", ["absent", "not_demographic", "wrong_source", "ambiguous"])
def test_an_unavailable_or_ambiguous_default_does_not_expand_the_model(change):
    context = _study()
    age = context.variable("age")
    if change == "absent":
        context.variables.remove(age)
    elif change == "not_demographic":
        age.role = VariableRole.LAB
    elif change == "wrong_source":
        age.source_concept = "unrelated_measurement"
    else:
        context.variables.remove(age)
        context.variables.extend(age.model_copy(update={"name": name}) for name in ("admission_age_a", "admission_age_b"))
    assert _request(context).exact_roster == ["sex"]


def test_a_unique_baseline_alias_is_used_without_inventing_a_column():
    context = _study()
    context.variable("age").name = "admission_age"
    assert _request(context).exact_roster == ["admission_age", "sex"]


def test_native_baseline_value_precedes_redundant_representations():
    context = _study()
    context.variables.insert(0, context.variable("age").model_copy(update={"name": "age_copy"}))
    assert _request(context).exact_roster == ["age", "sex"]


def test_a_derived_demographic_is_not_substituted_for_the_prespecified_value():
    context = _study()
    age = context.variable("age")
    context.variables.remove(age)
    context.variables.append(age.model_copy(update={"name": "age_band", "derived_from_concepts": ["age"]}))
    assert _request(context).exact_roster == ["sex"]


def test_the_ordinal_association_family_declares_the_same_minimal_defaults():
    from easyicu.research_agent.planning.adjustment_authority import AdjustmentSetAuthority

    context = _study()
    context.user_preferences.inferred_analysis_family = "ordinal_dose_response"
    assert AdjustmentSetAuthority.from_context(context).operational_covariates == ("age", "sex")


def test_source_unavailability_is_not_treated_as_a_prespecified_value(monkeypatch):
    from easyicu.research_agent.planning import adjustment_authority

    monkeypatch.setattr(adjustment_authority, "variable_source_unavailability", lambda v, database: v.name == "age")
    assert _request(_study()).exact_roster == ["sex"]


@pytest.mark.parametrize("family", ["prediction_model", "descriptive_epidemiology", "survival", "causal_inference"])
def test_other_study_families_keep_their_adjustment_owner(family):
    from easyicu.research_agent.planning.adjustment_authority import AdjustmentSetAuthority

    context = _study()
    context.user_preferences.inferred_analysis_family = family
    assert AdjustmentSetAuthority.from_context(context).selection == "planner_selectable"


def test_outcome_lineage_is_not_a_default_adjustment():
    context = _study()
    context.variable("death").derived_from_concepts = ["sex"]
    assert _request(context).exact_roster == ["age"]


def _plan(context, *, roster_note="Use the bound family adjustment without additions."):
    request = _full_request(context, cohort_mode="all_input_rows")
    payload = _spec_payload(request.model_copy(update={"required_reader_label_keys": []}), reader_display_labels=[
        {"key": key, "value": "Synthetic " + key.replace("_", " ")}
        for key in request.required_reader_label_keys
    ], roster_decision_note=roster_note, request_sha256=request.request_sha256)
    llm, attempt = _run(context, [json.dumps(payload)], required_primary_cohort_selection_mode="all_input_rows")
    assert len(llm.calls) == 1
    assert attempt.facts.prompt_metrics["family_spec_roster_decision_note"] == roster_note
    assert attempt.facts.prompt_metrics["family_spec_bound_adjustment_set"] == request.exact_roster
    return attempt.output


@pytest.mark.parametrize("family", ["baseline", "categorical", "spline"])
def test_two_plans_keep_the_same_adjustment_and_complete_case_population(family):
    import pandas as pd

    from easyicu.research_agent.execution.model_matrix import primary_model_rows
    from easyicu.research_agent.execution.runners.landmark_categorical_association_executor import landmark_eligibility_mask

    context = _study(family)
    counts, rosters = [], []
    frame = pd.DataFrame({
        "age": [30.0 + i for i in range(48)], "sex": [0, 1] * 24,
        context.primary_exposure: [0, 1, 2, 3] * 12 if family == "categorical" else [0, 1] * 24,
        "death": [0, 0, 1, 1] * 12, "death_time_hours": [None, None, 48, 48] * 12,
        "followup_time_hours": [96.0] * 48,
    })
    frame.loc[0, "age"] = float("nan")
    frame.loc[1, "sex"] = float("nan")
    frame.loc[2, context.primary_exposure] = float("nan")
    frame.loc[3, "followup_time_hours"] = 12
    frame.loc[6, "death_time_hours"] = 12
    for _ in range(2):
        plan = _plan(context.model_copy(deep=True))
        requirement = next(s for s in plan.steps if s.planned_analysis_role == "primary").model_requirements[0]
        assert "family_minimal_adjustment/1" in plan.rationale
        assert all("No adjustment was stated" in r for r in requirement.covariate_rationales.values())
        rosters.append(requirement.covariates)
        population = frame
        if family != "baseline":
            nonnegative, alive, observed = landmark_eligibility_mask(
                frame, outcome_column="death", event_time_column="death_time_hours",
                observation_duration_column="followup_time_hours", observation_duration_unit="hours", landmark_hours=24,
            )
            population = frame.loc[nonnegative & alive & observed]
        compiled = primary_model_rows(population, terms=requirement.model_terms,
                                      exposure=context.primary_exposure, outcome="death")
        counts.append(len(compiled.design))
    assert rosters == [["age", "sex"], ["age", "sex"]]
    assert counts == ([45, 45] if family == "baseline" else [43, 43])


def test_modifier_main_effect_and_interaction_are_fitted_once_with_stable_rows():
    import numpy as np
    import pandas as pd

    from easyicu.research_agent.execution.runners.association_effect_modification import fit_baseline_association
    from tests.support.baseline_association import baseline_runtime

    context = _study(modifier="sex")
    rng = np.random.default_rng(291)
    frame = pd.DataFrame({
        "chronic_flag": rng.integers(0, 2, 160), "age": rng.uniform(20, 80, 160),
        "sex": rng.integers(0, 2, 160), "death": rng.integers(0, 2, 160),
    })
    frame.loc[0, "age"] = float("nan")
    frame.loc[1, "sex"] = float("nan")
    observed = []
    for _ in range(2):
        plan = _plan(context.model_copy(deep=True))
        runtime = baseline_runtime(context, plan).seal_for_plan(plan)
        bound, _ = runtime.bind_plan(plan)
        requirement = bound.steps[3].model_requirements[0]
        assert requirement.covariates == ["age"]
        assert [term.name for term in requirement.effect_modifiers] == ["sex"]
        _, design, fitted, rows, interactions = fit_baseline_association(
            frame, requirement=requirement, authority=runtime.current_case,
        )
        assert sum(name.startswith("sex") for name in design.columns) == 1
        assert sum(":" in name for name in design.columns) == 1
        assert "sex" in interactions
        observed.append((fitted.n, next(row["estimate"] for row in rows if row["is_primary_contrast"])))
    assert observed[0] == observed[1]
    assert observed[0][0] == 158


def test_a_planner_addition_is_recorded_for_review_without_changing_the_fit():
    context = _study()
    reason = "Propose admission type in a study revision: it precedes the exposure and may confound its association."
    plan = _plan(context, roster_note=reason)
    assert plan.steps[3].model_requirements[0].covariates == ["age", "sex"]
    request = _request(context)
    spec = FamilyPlanSpec(
        family_id=request.family_id, request_sha256=request.request_sha256,
        adjustment_set=[{
            "name": "admission_type", "coding": "categorical", "reference_level_index": 0,
            "clinical_rationale": reason,
        }], reader_display_labels=[], comparator_applications=[], roster_decision_note=reason,
    )
    with pytest.raises(FamilySpecError, match="exact_roster_mismatch"):
        validate_family_plan_spec(spec, request)


def test_a_reviewed_agent_roster_retains_its_recorded_addition_reason():
    context = baseline_context("chronic_flag", covariates=("age", "sex", "admission_type"))
    context.user_preferences.covariate_authority = "agent_plan"
    reason = "Reviewed addition: admission type precedes exposure and may confound the association."
    context.user_preferences.covariate_rationales["admission_type"] = reason
    plan = _plan(context)
    assert plan.steps[3].model_requirements[0].covariates == ["age", "sex", "admission_type"]
    assert plan.steps[3].model_requirements[0].covariate_rationales["admission_type"] == reason


def test_no_eligible_defaults_records_the_policy_with_an_empty_roster():
    context = _study()
    context = context.model_copy(update={"variables": [v for v in context.variables if v.name not in {"age", "sex"}]})
    plan = _plan(context)
    assert plan.steps[3].model_requirements[0].covariates == []
    assert "family_minimal_adjustment/1" in plan.rationale
    assert "bound roster is []" in plan.rationale


def test_subgroup_fallback_gets_the_same_bound_authority_and_prompt():
    from easyicu.research_agent.agents.family_spec_planner import family_spec_fallback_reason
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent, select_progressive_variables
    from easyicu.research_agent.planning.adjustment_authority import AdjustmentSetAuthority

    context = _study("spline", modifier="sex")
    assert family_spec_fallback_reason(context, analysis_types=("association_study",),
                                       resume_checkpoint=None, stop_after_outline=False) == "family_template_stated_subgroup_requires_outline"
    authority = AdjustmentSetAuthority.from_context(context)
    assert authority.operational_covariates == ("age",)
    assert authority.rule == "family_minimal_adjustment/1"
    assert authority.prompt_projection()["selection_rule"] == authority.rule
    assert "Host family default" in authority.prompt_projection()["authorship_boundary"]
    variables = select_progressive_variables(context)
    assert "age" in variables
    prompt = ProgressivePlannerAgent._user_prompt(context, analysis_types=("association_study",),
                                                  variables=variables, action_rows=())
    assert "family_minimal_adjustment/1" in prompt


@pytest.mark.parametrize("roster", [["age"], ["sex", "age"], ["age_first", "sex"], ["age", "sex", "admission_type"]])
def test_general_planner_cannot_change_the_bound_default(roster):
    from easyicu.research_agent.planning.adjustment_authority import AdjustmentAuthorityError, validate_plan_against_adjustment_authority

    context = _study()
    plan = _plan(context)
    requirement = plan.steps[3].model_requirements[0]
    plan.steps[3].model_requirements = [requirement.model_copy(update={"covariates": roster})]
    with pytest.raises(AdjustmentAuthorityError, match="family_minimal_adjustment/1"):
        validate_plan_against_adjustment_authority(plan=plan, context=context)


def test_general_compiler_records_the_default_without_a_family_template():
    from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from .progressive_planner_fixtures import _context as general_context, _payload

    context = general_context()
    context.user_preferences.data_constraints = "{}"
    context = context.model_copy(update={"research_question": "Relate the synthetic exposure to the outcome."})
    context.variable("age_years").source_concept = "age"
    context.variable("sex_code").source_concept = "sex"
    plans = [compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate(_payload()), context=context)[0]
             for _ in range(2)]
    for plan in plans:
        primary = next(s for s in plan.steps if s.planned_analysis_role == "primary").model_requirements[0]
        assert primary.covariates == ["age_years", "sex_code"]
        assert all("family_minimal_adjustment/1" in reason for reason in primary.covariate_rationales.values())
        assert "family_minimal_adjustment/1" in plan.rationale
    assert plans[0].model_dump() == plans[1].model_dump()
