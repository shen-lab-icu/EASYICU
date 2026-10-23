"""The first landmark plan binds the question's roster through execution."""

import json

import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.planning.family_spec import FamilySpecError
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from easyicu.webserver import agent_pipeline_runs

from .family_spec_fixtures import LABELS, _context, _request, _run, _spec_payload


def _study(roster=("sex",)):
    context = _context(exact=False)
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_adjustments"] = [
        {"variable": name, "concept": name, "words": f"adjust for {name}",
         "source_status": "available"}
        for name in roster
    ]
    keep = {"patient_stay_id", "age", "sex", "death", "death_time_hours", "followup_time_hours"}
    return context.model_copy(update={
        "research_question": "Relate the first-day assay to hospital death after a 24-hour landmark.",
        "primary_exposure": "assay_max",
        "variables": [v for v in context.variables if v.name in keep] + [ConceptDescriptor(
            name="assay_max", role="lab", dtype="float64", analysis_window="icu_admission[0,24]h",
        )],
        "cohort": context.cohort.model_copy(update={"outcome_columns": ["death"], "requested_outcome_columns": ["death"]}),
        "user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps(constraints),
            "sensitivity_specs": [s for s in context.user_preferences.sensitivity_specs if s.axis == "timing"],
            "must_have_outputs": None,
        }),
    })


def _first_plan(context, roster):
    request = _request(context)
    labels = {**LABELS, "assay_max": "Peak first-day assay concentration"}
    spec = {
        "family_id": request.family_id, "request_sha256": request.request_sha256,
        "adjustment_set": [
            {"name": name, "coding": "binary" if name == "sex" else "continuous",
             "reference_level_index": 0 if name == "sex" else None,
             "clinical_rationale": "The question states this covariate, measured before the landmark."}
            for name in roster
        ],
        "reader_display_labels": [{"key": name, "value": labels[name]}
                                  for name in request.required_reader_label_keys],
        "comparator_applications": [{"citation_key": key,
                                     "application": "Compare population, exposure and landmark without adopting another estimand."}
                                    for key in request.direct_comparator_literature_keys],
        "roster_decision_note": "Keep exactly the adjustment variables stated by the question.",
    }
    llm, result = _run(context, [json.dumps(spec)])
    assert len(llm.calls) == 1
    return result.output


def _runtime(context, *, clustered=False):
    authority = build_current_case_scientific_runtime_authority({
        "schema_version": "easyicu.landmark_spline_runtime_authority/5",
        "authority_kind": "landmark_spline_association", "protocol_content_sha256": "a" * 64,
        "plan_method": "signed_landmark_restricted_cubic_spline",
        "plan_intent": "Fit the reviewed landmark spline association.",
        "plan_outputs": ["table:landmark_rcs_curve", "table:landmark_rcs_contrasts",
                         "table:landmark_linear_sensitivity", "table:landmark_adjusted_absolute_risk",
                         "table:landmark_population_flow", "table:landmark_variable_opportunity_sensitivity",
                         "log:landmark_scientific_runtime_receipt"],
        "exposure_column": context.primary_exposure, "outcome_column": "death",
        "outcome_time_column": "death_time_hours", "observation_duration_column": "followup_time_hours",
        "observation_duration_unit": "hours", "landmark_hours": 24,
        "required_adjustment_columns": [], "categorical_adjustment_columns": [],
        "alternative_exposure_columns": [],
        "dependence": (PlannedDependenceRequirement(
            group_source="patient_stay_id", group_derivation="prefix_before_delimiter", delimiter=":s",
        ).model_dump(mode="json") if clustered else None),
        "spline_knot_quantiles": [0.1, 0.5, 0.9], "spline_reference": "median_in_primary_population",
        "curve_quantile_range": [0.1, 0.9], "curve_points": 41, "linear_sensitivity_per_unit": 1.0,
        "interpretation": "descriptive_prognostic_association_not_causal",
        "adjusted_absolute_risk_product": "table:landmark_adjusted_absolute_risk",
        "population_flow_product": "table:landmark_population_flow",
        "variable_opportunity_sensitivity_product": "table:landmark_variable_opportunity_sensitivity",
        "plan_bound_adjustment_roster": {"authority": "plan_primary_model", "admissible_columns": ["age", "sex"],
                                         "admissible_categorical_columns": ["sex"], "sealed": False},
    })
    return ScientificRuntimeAuthorities(trajectory=None, current_case=authority)


@pytest.mark.parametrize("roster", [("sex",), ("age",), ("sex", "age"), ()])
def test_first_request_binds_exactly_the_stated_roster(roster):
    request = _request(_study(roster))
    assert request.adjustment_selection == "exact"
    assert request.exact_roster == list(roster)


@pytest.mark.parametrize("roster", [("sex",), ("age",)])
def test_signed_primary_still_covers_the_stated_adjustment(roster):
    context = _study(roster)
    draft = _first_plan(context, roster)
    runtime = _runtime(context).seal_for_plan(draft)
    bound, _ = runtime.bind_plan(draft)
    assert runtime.current_case.required_adjustment_columns == roster
    assert runtime.current_case.governed_step(bound).method == "signed_landmark_restricted_cubic_spline"
    review = build_plan_scientific_review(context=context, plan=bound, runtime_authority=runtime.current_case)
    assert review.facts["stated_adjustment_coverage"]["complete"]
    assert "STATED_ADJUSTMENT_MISSING" not in {finding.code for finding in review.findings}


def test_the_host_binds_the_stated_roster_even_if_the_spec_omits_it():
    context = _study(("sex", "age"))
    draft = _first_plan(context, ())
    primary = next(step for step in draft.steps if step.planned_analysis_role == "primary")
    assert primary.model_requirements[0].covariates == ["sex", "age"]
    assert _runtime(context).seal_for_plan(draft).current_case.required_adjustment_columns == ("sex", "age")


def test_unavailable_stated_covariate_stops_before_planning():
    with pytest.raises(FamilySpecError, match="family_spec_stated_adjustment_unavailable"):
        _request(_study(("missing_covariate",)))


@pytest.mark.parametrize("code,sentence,detail", [
    (
        "family_spec_stated_adjustment_unavailable",
        "The question asks to adjust for a variable that the extracted data does not provide, "
        "so planning stopped and no analysis was run. Choose data that includes it, or remove "
        "it from the adjustment variables, then generate the plan again.",
        "The question asks to adjust for 'missing_covariate', but the extracted data has no single column for it.",
    ),
    (
        "family_spec_exact_roster_mismatch",
        "The plan's adjustment variables differ from the ones named in the question, "
        "so planning stopped and no analysis was run. Generate the plan again.",
        "The plan's adjustment variables differ from the ones the question names.",
    ),
    (
        "family_spec_covariate_timing_unproven",
        "An adjustment variable is not known by the landmark time, so it cannot be used to "
        "adjust this analysis. Planning stopped and no analysis was run. Give that variable "
        "a time window that ends before the landmark, or remove it, then generate the plan again.",
        "The adjustment variable 'sex' is not known by the landmark time.",
    ),
], ids=["unavailable", "roster_mismatch", "timing_unproven"])
def test_adjustment_refusals_reach_their_own_planning_stop_sentence(code, sentence, detail):
    context = _study(("missing_covariate",) if code.endswith("unavailable") else ("sex",))
    if code.endswith("mismatch"):
        context.user_preferences.covariate_selection = "exact"
        context.user_preferences.covariates = ["age"]
    elif code.endswith("timing_unproven"):
        context = context.model_copy(update={"variables": [
            v.model_copy(update={"analysis_window": "icu_admission[0,48]h"})
            if v.name == "sex" else v for v in context.variables
        ]})
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        _run(context, [])
    stop = stopped.value
    assert stop.reason_code == f"progressive_{code}"
    assert isinstance(stop.__cause__, FamilySpecError)
    assert stop.__cause__.reason_code == code
    assert detail in str(stop.__cause__)
    assert stop.easyicu_safe_diagnostic["metrics"] == {"planner_provider_calls": 0}
    assert agent_pipeline_runs._pipeline_failure_code(stop) == "research_pipeline_progressive_compile_failed"
    assert agent_pipeline_runs._safe_pipeline_typed_failure(stop)["reason_code"] == stop.reason_code
    assert agent_pipeline_runs._progressive_compile_failure_message(stop) == sentence


@pytest.mark.parametrize("status", ["unavailable", "unbound", "unresolved"])
def test_a_name_in_the_roster_cannot_override_the_source_status(status):
    context = _study()
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_adjustments"][0]["source_status"] = status
    context.user_preferences.data_constraints = json.dumps(constraints)
    with pytest.raises(FamilySpecError, match="family_spec_stated_adjustment_unavailable"):
        _request(context)


def test_ambiguous_value_columns_do_not_silently_pick_an_adjustment():
    context = _study(("routine_assay",))
    context.variables.extend([
        ConceptDescriptor(name=f"routine_assay_{suffix}", role="lab", dtype="float64",
                          analysis_window="icu_admission[0,24]h")
        for suffix in ("min", "max")
    ])
    with pytest.raises(FamilySpecError, match="family_spec_stated_adjustment_unavailable"):
        _request(context)


def test_an_exact_roster_cannot_add_to_or_omit_the_question_roster():
    for covariates in ([], ["age"], ["sex", "age"]):
        context = _study()
        context.user_preferences.covariate_selection = "exact"
        context.user_preferences.covariates = covariates
        with pytest.raises(FamilySpecError, match="family_spec_exact_roster_mismatch"):
            _request(context)


def test_no_stated_field_still_binds_an_unadjusted_model():
    context = _study(())
    constraints = json.loads(context.user_preferences.data_constraints)
    del constraints["stated_adjustments"]
    context.user_preferences.data_constraints = json.dumps(constraints)
    draft = _first_plan(context, ())
    assert _runtime(context).seal_for_plan(draft).current_case.required_adjustment_columns == ()


@pytest.mark.parametrize("role", ["demographic", "lab"])
def test_a_declared_static_role_cannot_override_a_late_measurement(role):
    context = _study(("age",))
    context = context.model_copy(update={"variables": [
        v.model_copy(update={"role": VariableRole(role), "analysis_window": "icu_admission[0,48]h"})
        if v.name == "age" else v for v in context.variables
    ]})
    context.user_preferences.covariate_selection = "exact"
    context.user_preferences.covariates = ["age"]
    context.user_preferences.covariate_temporal_roles = {"age": "baseline_static"}
    with pytest.raises(FamilySpecError, match="family_spec_covariate_timing_unproven"):
        _request(context)


def test_a_host_window_can_time_an_untimed_measurement():
    context = _study(("routine_assay",))
    context.variables.append(ConceptDescriptor(name="routine_assay", role="lab", dtype="float64"))
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["materialization_window"] = {"anchor": "icu_admission", "hours": 24, "role": "outer_observation_window"}
    context.user_preferences.data_constraints = json.dumps(constraints)
    assert _request(context).exact_roster == ["routine_assay"]


def test_an_untimed_lab_is_not_baseline_even_if_an_exact_roster_calls_it_static():
    context = _study(("routine_assay",))
    context.variables.append(ConceptDescriptor(name="routine_assay", role="lab", dtype="float64"))
    context.user_preferences.covariate_selection = "exact"
    context.user_preferences.covariates = ["routine_assay"]
    context.user_preferences.covariate_temporal_roles = {"routine_assay": "baseline_static"}
    with pytest.raises(FamilySpecError, match="family_spec_covariate_timing_unproven"):
        _request(context)


def test_a_categorical_landmark_with_no_stated_roster_also_compiles_without_adjustment():
    context = _context(exact=False)
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_adjustments"] = []
    context.user_preferences.data_constraints = json.dumps(constraints)
    request = _request(context)
    _, result = _run(context, [json.dumps(_spec_payload(request))])
    primary = next(s for s in result.output.steps if s.planned_analysis_role == "primary")
    assert primary.model_requirements[0].covariates == []


@pytest.mark.parametrize("window,allowed", [
    ("icu_admission[0,12]h", True), ("icu_admission[0,24]h", True),
    ("icu_admission[24,48]h", False), ("icu_admission[-12,12]h", False),
    ("event_onset[0,12]h", False), (None, False),
])
def test_stated_measurement_must_be_known_at_the_landmark(window, allowed):
    context = _study(("routine_assay",))
    context.variables.append(ConceptDescriptor(
        name="routine_assay", role="lab", dtype="float64", analysis_window=window,
    ))
    context = context.model_copy(update={"time_windows": []})
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints.pop("materialization_window", None)
    context.user_preferences.data_constraints = json.dumps(constraints)
    if allowed:
        request = _request(context)
        assert request.exact_roster == ["routine_assay"]
        assert request.exact_temporal_roles == {"routine_assay": "at_or_before_time_zero"}
    else:
        with pytest.raises(FamilySpecError) as stopped:
            _request(context)
        assert stopped.value.reason_code == "family_spec_covariate_timing_unproven"


@pytest.mark.parametrize("drift", ["no_authority", "unsealed", "wrong_digest", "wrong_method", "missing_input"])
def test_an_unsigned_or_drifted_primary_cannot_claim_adjustment(drift):
    context = _study()
    draft = _first_plan(context, ("sex",))
    runtime = _runtime(context).seal_for_plan(draft)
    bound, _ = runtime.bind_plan(draft)
    authority = runtime.current_case
    primary = authority.governed_step(bound)
    if drift == "no_authority":
        authority = None
    elif drift == "unsealed":
        authority = _runtime(context).current_case
    else:
        updates = {"wrong_digest": {"icu_rule_refs": []}, "wrong_method": {"method": "descriptive_statistics"},
                   "missing_input": {"inputs": [name for name in primary.inputs if name != "sex"]}}
        bound = bound.model_copy(update={"steps": [
            step.model_copy(update=updates[drift]) if step.step_id == primary.step_id else step
            for step in bound.steps
        ]})
    review = build_plan_scientific_review(context=context, plan=bound, runtime_authority=authority)
    assert "STATED_ADJUSTMENT_MISSING" in {finding.code for finding in review.findings}


def test_an_input_column_outside_the_signed_adjustment_set_does_not_cover_adjustment():
    from easyicu.research_agent.planning.requirement_coverage import stated_adjustment_coverage

    context = _study(())
    draft = _first_plan(context, ())
    runtime = _runtime(context).seal_for_plan(draft)
    bound, _ = runtime.bind_plan(draft)
    bound = bound.model_copy(update={"steps": [
        step.model_copy(update={"inputs": [*step.inputs, "sex"]}) if step.planned_analysis_role == "primary" else step
        for step in bound.steps
    ]})
    coverage = stated_adjustment_coverage(_study(("sex",)), bound, runtime_authority=runtime.current_case)
    assert not coverage.complete


@pytest.mark.parametrize("coordinate", ["primary_exposure", "target_outcome"])
def test_a_runtime_for_another_question_coordinate_does_not_cover_adjustment(coordinate):
    from easyicu.research_agent.planning.requirement_coverage import stated_adjustment_coverage

    context = _study()
    draft = _first_plan(context, ("sex",))
    runtime = _runtime(context).seal_for_plan(draft)
    bound, _ = runtime.bind_plan(draft)
    changed = context.model_copy(update={coordinate: "another_variable"})
    assert not stated_adjustment_coverage(changed, bound, runtime_authority=runtime.current_case).complete


@pytest.mark.parametrize("roster", [("sex",), ("age",), ("sex", "age"), ()])
@pytest.mark.parametrize("clustered", [False, True])
def test_signed_model_fits_its_adjustment_and_publishes_a_valid_receipt(tmp_path, monkeypatch, roster, clustered):
    import numpy as np
    import pandas as pd

    from easyicu.research_agent.contracts.landmark_spline_validation import LandmarkSplineRuntimeReceipt
    from easyicu.research_agent.execution.runners import landmark_spline_executor as executor

    context = _study(roster)
    draft = _first_plan(context, roster)
    authority = _runtime(context, clustered=clustered).seal_for_plan(draft).current_case
    rng = np.random.default_rng(31)
    n = 180
    sex = np.arange(n) % 2
    age = rng.uniform(25, 85, n)
    assay = rng.normal(10, 2, n) + 2 * sex + 0.03 * age
    probability = 1 / (1 + np.exp(-(-1.5 + 0.2 * (assay - 10) + sex + 0.03 * (age - 50))))
    death = rng.binomial(1, probability)
    frame = pd.DataFrame({
        "patient_stay_id": [f"p{i // 2}:s{i % 2}" for i in range(n)],
        "assay_max": assay, "age": age, "sex": sex, "death": death,
        "death_time_hours": np.where(death, 60.0, np.nan), "followup_time_hours": 72.0,
    })
    fitted = []
    original = executor._fit_binomial_model

    def capture(**kwargs):
        fitted.append(list(kwargs["design"].columns))
        return original(**kwargs)

    monkeypatch.setattr(executor, "_fit_binomial_model", capture)
    summary = executor.run_landmark_spline_association(
        frame=frame, authority=authority, runtime_projection_sha256="b" * 64, out_dir=tmp_path,
    )
    expected_terms = ["sex_1" if name == "sex" else name for name in roster]
    assert fitted[0] == ["const", "exposure_rcs_1", "exposure_rcs_2", *expected_terms]
    assert fitted[1] == ["const", "assay_max", *expected_terms]
    receipt = LandmarkSplineRuntimeReceipt.model_validate(summary["scientific_runtime_receipt"])
    assert receipt.adjustment_columns == roster
    assert receipt.schema_version.endswith("/4" if clustered else "/2")
    assert (receipt.variance_estimator == "cluster_robust") is clustered
