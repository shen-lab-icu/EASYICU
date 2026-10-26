"""Baseline questions use a closed host template, without landmark eligibility."""

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent, FAMILY_SPEC_STRATEGY
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime
from easyicu.research_agent.planning.family_spec import build_family_spec_request

from easyicu.research_agent.planning.family_spec import family_template_id_for_context
from easyicu.research_agent.schema import (
    CohortDescriptor, ConceptDescriptor, EndpointSpec, ResearchContext, UserPreferences,
)


@pytest.mark.parametrize("exposure", ["sex", "admission_type", "age", "chronic_flag", "age_group"])
def test_baseline_exposures_route_without_a_landmark(exposure):
    assert family_template_id_for_context(
        baseline_context(exposure), analysis_types=("association_study",),
    ) == "baseline_adjusted_association"


@pytest.mark.parametrize("hours", [6, 12, 24])
def test_post_baseline_exposure_without_a_landmark_is_not_routed(hours):
    context = baseline_context("age")
    context.variables[1] = context.variables[1].model_copy(update={"analysis_window": f"icu_admission[0,{hours}]h"})
    assert family_template_id_for_context(context, analysis_types=("association_study",)) is None


@pytest.mark.parametrize("exposure,expected", [("sex", "landmark_categorical_association"), ("age", "landmark_spline_association")])
@pytest.mark.parametrize("hours", [6, 12, 24])
def test_a_declared_24_hour_landmark_keeps_the_existing_route(exposure, expected, hours):
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec

    context = baseline_context(exposure)
    if exposure == "age":
        context.variables[1] = context.variables[1].model_copy(update={"role": "lab"})
    context.user_preferences.sensitivity_specs = [PrespecifiedSensitivitySpec(
        spec_id="landmark_24h", axis="timing", strategy="landmark", landmark_hours=24,
        require_alive_at_landmark=True, exclude_negative_event_times=True,
        event_time_variable="event_hours", observation_duration_variable="followup_hours", observation_duration_unit="hours",
    )]
    context = context.model_copy(update={"variables": [v.model_copy(update={"analysis_window": f"icu_admission[0,{hours}]h"}) if v.name == exposure else v for v in context.variables]})
    assert family_template_id_for_context(context, analysis_types=("association_study",)) == expected


@pytest.mark.parametrize("family,expected", [("prediction_model", "static_prediction_model"), ("descriptive_epidemiology", "descriptive_exposure_outcome")])
def test_other_headlines_keep_their_templates(family, expected):
    context = baseline_context("sex")
    assert family_template_id_for_context(context, analysis_types=(family,)) == expected


def test_threshold_grouping_reads_the_existing_numeric_source_menu():
    from easyicu.research_agent.planning.capability_gap import threshold_grouping_expressible
    from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_TRANSFORM_ID

    context = baseline_context("age")
    assert not threshold_grouping_expressible(context, "missing_value")
    context.variables[1] = context.variables[1].model_copy(update={"unit_normalization": "stay_level_unique_value"})
    assert threshold_grouping_expressible(context, "age")
    from easyicu.research_agent.planning.exposure_group_compile import compile_exposure_groupings
    from easyicu.research_agent.planning.exposure_group_spec import read_stated_exposure_groupings
    from easyicu.research_agent.contracts.exposure_group_rules import read_grouping_rules, evaluate_grouping
    import pyarrow as pa

    declared = read_stated_exposure_groupings({"groupings": [{
        "id": "x1", "concept": "age", "window": None, "scale": "ordinal",
        "groups": [
            {"id": "g1", "label": "under 65", "rule": {"summary": "value", "op": "<", "value": 65, "unit": "years"}},
            {"id": "g2", "label": "65 or older", "rule": {"summary": "value", "op": ">=", "value": 65, "unit": "years"}},
        ], "unmeasured": {"handling": "exclude"}, "reference": "g1", "contrast": "g2",
        "quote": "age 65 or older versus under 65", "source": "question",
    }]})
    grouped = compile_exposure_groupings(declared, context).groupings[0]
    assert grouped.disposition == "applied"
    rules = read_grouping_rules(grouped.derivation())
    assert evaluate_grouping(rules, pa.table({"age": [64, 65, 66]})).to_pylist() == [1, 2, 2]
    context.variables.append(ConceptDescriptor(
        name=grouped.variable, role="other", dtype="int64", derived_from_concepts=["age"],
        unit_normalization=EXPOSURE_GROUP_TRANSFORM_ID,
        observed_domain={"levels": [1, 2], "n_unique": 2, "is_binary": True},
    ))
    context = context.model_copy(update={"primary_exposure": grouped.variable, "research_question": "Age ≥65 versus <65 and hospital death, adjusted for sex."})
    assert family_template_id_for_context(context, analysis_types=("association_study",)) == "baseline_adjusted_association"


def test_family_spec_rejects_extra_steps_and_omitted_covariate_binding():
    from pydantic import ValidationError
    from easyicu.research_agent.planning.family_spec import FamilyPlanSpec, FamilySpecError, validate_family_plan_spec

    context = baseline_context("chronic_flag", covariates=("age", "sex"))
    request = build_family_spec_request(context, analysis_types=("association_study",),
                                        variable_roster=[v.name for v in context.variables], allowed_literature_citation_keys=())
    payload = dict(family_id=request.family_id, request_sha256=request.request_sha256,
                   adjustment_set=[dict(name="age", coding="continuous", reference_level_index=None,
                                        clinical_rationale="The clinician states this adjustment variable.")],
                   reader_display_labels=[], comparator_applications=[], roster_decision_note="Keep the exact user roster.")
    with pytest.raises(ValidationError):
        FamilyPlanSpec.model_validate({**payload, "steps": [{"module_id": "robustness_replay"}]})
    with pytest.raises(FamilySpecError, match="exact_roster_mismatch"):
        validate_family_plan_spec(FamilyPlanSpec.model_validate(payload), request)


def test_a_stated_adjustment_absent_from_the_roster_is_refused():
    from easyicu.research_agent.planning.family_spec import FamilySpecError

    context = baseline_context("chronic_flag")
    context.user_preferences.data_constraints = json.dumps({"stated_adjustments": [
        {"variable": "age", "concept": "age", "words": "adjusted for age", "source_status": "available"},
    ]})
    with pytest.raises(FamilySpecError, match="stated_adjustment_unavailable"):
        build_family_spec_request(
            context, analysis_types=("association_study",),
            variable_roster=[v.name for v in context.variables if v.name != "age"],
            allowed_literature_citation_keys=(),
        )


@pytest.mark.parametrize("fixed,refused", [(("age", "sex"), None), (("age",), "sex")])
def test_a_partial_question_reading_is_a_subset_of_the_fixed_roster(fixed, refused):
    from easyicu.research_agent.planning.family_spec import FamilySpecError

    context = baseline_context("chronic_flag", covariates=fixed)
    context.user_preferences.covariate_authority = "user"
    context.user_preferences.data_constraints = json.dumps({"stated_adjustments": [
        {"variable": "sex", "concept": "sex", "words": "adjusted for sex", "source_status": "available"},
    ]})
    def build():
        return build_family_spec_request(
            context, analysis_types=("association_study",),
            variable_roster=[v.name for v in context.variables], allowed_literature_citation_keys=(),
        )

    if refused is None:
        assert build().exact_roster == list(fixed)
        return
    with pytest.raises(FamilySpecError, match="exact_roster_mismatch") as stopped:
        build()
    assert (stopped.value.stated_variable, stopped.value.cause_code) == (
        refused, "stated_adjustment_outside_exact_roster")


def test_the_planner_cannot_supply_host_owned_subgroup_terms():
    from easyicu.research_agent.agents.plan_payload import (
        PlannerHostOwnedFieldError, _planner_transport_schema,
        decode_planner_transport_payload, parse_runtime_plan_suffix,
    )

    assert "effect_modifiers" not in json.dumps(_planner_transport_schema())
    step = {"step_id": "primary", "model_requirements": [{
        "requirement_id": "primary", "outcome": "death", "outcome_type": "binary",
        "method_family": "binary_logistic_regression", "exposure_source": "admission_type",
        "effect_modifiers": [{"name": "sex", "role": "covariate", "coding": "binary",
                              "transform": "treatment_contrast", "levels": ["0", "1"],
                              "reference_level": "0"}],
    }]}
    with pytest.raises(PlannerHostOwnedFieldError) as rejected:
        decode_planner_transport_payload({"research_question": "q", "steps": [step]})
    assert rejected.value.fields == ("effect_modifiers",)
    assert rejected.value.path == "steps[0].model_requirements[0]"
    with pytest.raises(PlannerHostOwnedFieldError):
        parse_runtime_plan_suffix(json.dumps({
            "replace_from_step_id": "primary", "replacement_step": step,
            "rationale": "Keep the declared subgroup.",
        }))


def test_the_minimal_family_does_not_satisfy_an_unexecuted_typed_sensitivity():
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec

    context = baseline_context("sex", covariates=("age",))
    context.user_preferences.sensitivity_specs = [PrespecifiedSensitivitySpec(
        spec_id="complete_cases", axis="missing_data", strategy="complete_case",
        execution_variables=("age",),
    )]
    attempt, _ = baseline_plan_attempt(context)
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    plan, _ = runtime.bind_plan(attempt.output)
    review = build_plan_scientific_review(context=context, plan=plan, runtime_authority=runtime.current_case)

    assert not review.approval_allowed
    finding, = [item for item in review.findings if item.code == "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY"]
    assert finding.severity == "blocker"
    assert "complete_cases" in finding.message
    assert finding.requires_user_authorization


def test_first_result_template_has_no_article_steps(monkeypatch):
    from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions
    from easyicu.research_agent.execution import owner_declaration

    context = baseline_context("sex", covariates=("age",))
    attempt, _ = baseline_plan_attempt(
        context,
        planning_contract_context=first_result_instructions(),
    )
    assert attempt.output.design_selection.execution_tier == "first_result"
    assert [step.step_id for step in attempt.output.steps] == [
        "cohort_definition", "table_one", "measurement_audit", "adjusted_association", "absolute_risk_context",
    ]
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    from types import SimpleNamespace
    from easyicu.research_agent.pipeline import _shape_fresh_plan

    shaped = _shape_fresh_plan(
        SimpleNamespace(_scientific_runtime_authorities=runtime, _enable_publication_figure_skill=True, _max_total_steps=8),
        plan=attempt.output, context=context, agent_context=context, long_trajectory_bound=False, findings=[],
    )
    assert [step.step_id for step in shaped.steps] == [step.step_id for step in attempt.output.steps]
    bound, _ = runtime.bind_plan(attempt.output)
    monkeypatch.setattr(owner_declaration, "_verified_host_cohort_step", lambda *args: "cohort_definition")
    owner_declaration.require_host_rendered_plan(plan=bound, current_case_scientific_runtime_authority=runtime.current_case)
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    review = build_plan_scientific_review(context=context, plan=bound, runtime_authority=runtime.current_case, require_reportable_capability=True)
    assert not {"ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED", "CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED",
                "FIGURE_ROLE_COVERAGE_INCOMPLETE", "ARTICLE_CONTENT_ROLES_INCOMPLETE"} & {item.code for item in review.findings}
    assert review.status == "analysis_only" and review.approval_allowed
    assert not review.top_journal_candidate


def test_first_result_keeps_a_question_requested_figure():
    from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions

    context = baseline_context("sex", covariates=("age",))
    context.user_preferences.must_have_outputs = "Generate a figure for the primary association."
    attempt, _ = baseline_plan_attempt(context, planning_contract_context=first_result_instructions())
    assert "visualization" in {step.step_id for step in attempt.output.steps}
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    runtime.bind_plan(attempt.output)


@pytest.mark.parametrize("exposure,outcome,covariates,subgroup", [
    ("sex", "death", (), None), ("admission_type", "unit_death", ("age",), "sex"),
    ("chronic_flag", "death", ("age", "sex", "admission_type"), "sex"),
    ("age", "unit_death", ("sex",), None), ("age", "death", ("sex",), "sex"),
    ("age_group", "death", ("sex",), None),
])
def test_seven_step_template_compiles(exposure, outcome, covariates, subgroup, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step
    from easyicu.research_agent.execution.owner_declaration import resolve_standard_executor
    from easyicu.research_agent.execution.step_executor_registry import StandardExecutorSelection
    from easyicu.research_agent.planning.requirement_coverage import stated_subgroup_coverage

    context = baseline_context(exposure, outcome=outcome, covariates=covariates, subgroup=subgroup)
    result, llm = baseline_plan_attempt(context)
    assert [step.step_id for step in result.output.steps] == [
        "cohort_definition", "table_one", "measurement_audit", "adjusted_association",
        "absolute_risk_context", "visualization", "report",
    ]
    assert result.output.steps[3].model_requirements[0].covariates == list(covariates)
    assert len(llm.calls) == 1
    table_one = result.output.steps[1]
    group = outcome if exposure == "age" else exposure
    assert table_one.table_one_spec.group_by == group
    assert f"Host decision: Table 1 is grouped by Clinical {group.replace('_', ' ')} because " in table_one.intent
    assert ("the exposure is continuous and no threshold grouping was stated" if exposure == "age"
            else "the exposure has declared groups") in table_one.intent
    requirement = result.output.steps[3].model_requirements[0]
    runtime = baseline_runtime(context, result.output)
    authority = runtime.current_case
    if covariates:
        omitted = requirement.model_copy(update={
            "covariates": [], "covariate_rationales": {}, "covariate_temporal_roles": {},
            "model_terms": [term for term in requirement.model_terms if term.role == "exposure"],
        })
        draft = result.output.model_copy(deep=True)
        draft.steps[3].model_requirements = [omitted]
        with pytest.raises(ValueError, match="signed coordinates"):
            authority.seal_adjustment_roster(draft)
    bound, _ = runtime.bind_plan(result.output)
    runtime.seal_for_plan(result.output).validate_plan(bound)
    sealed = runtime.seal_for_plan(result.output).current_case
    from easyicu.research_agent.schema import AnalysisPlan

    assert AnalysisPlan.model_validate_json(bound.model_dump_json()) == bound
    from easyicu.research_agent.contracts.association_execution import baseline_association_execution_verdict, association_execution_verdict
    assert baseline_association_execution_verdict(bound.steps[3]).claimed
    assert not association_execution_verdict(bound.steps[3]).claimed
    assert not resolve_standard_executor(bound.steps[3], plan=bound).claimed_by
    unsigned = bound.steps[3].model_copy(update={"icu_rule_refs": []})
    assert not baseline_association_execution_verdict(unsigned).claimed
    unsigned_plan = bound.model_copy(update={"steps": [*bound.steps[:3], unsigned, *bound.steps[4:]]})
    with pytest.raises(ValueError, match="signed reference"):
        sealed.validate_plan(unsigned_plan)
    wrong_capability = bound.model_copy(deep=True)
    wrong_capability.steps[3].scientific_capability = "association_adjusted_v1"
    with pytest.raises(ValueError, match="capability drifted"):
        sealed.validate_plan(wrong_capability)
    unconsumed = bound.model_copy(deep=True)
    unconsumed.steps[4].input_consumption_contracts = []
    with pytest.raises(ValueError, match="evidence consumption"):
        sealed.validate_plan(unconsumed)
    extra_step = bound.model_copy(update={"steps": [*bound.steps, bound.steps[-1].model_copy(update={"step_id": "extra_report"})]})
    with pytest.raises(ValueError, match="step roster"):
        sealed.validate_plan(extra_step)
    if subgroup:
        assert not association_execution_verdict(bound.steps[3].model_copy(update={"method": "adjusted_association_models"})).claimed
        modifier_payload = bound.steps[3].model_requirements[0].model_dump(mode="json")
        modifier_payload["effect_modifiers"][0]["role"] = "exposure"
        with pytest.raises(ValueError, match="categorical main effects"):
            type(requirement).model_validate(modifier_payload)
        wrong = bound.steps[3].model_dump(mode="json")
        wrong["scientific_action_id"] = "association.adjusted_association"
        from easyicu.research_agent.schema import AnalysisStep
        wrong_step = AnalysisStep.model_validate(wrong)
        assert not baseline_association_execution_verdict(wrong_step).claimed
        with pytest.raises(ValueError, match="subgroup parameter"):
            sealed.validate_plan(bound.model_copy(update={"steps": [*bound.steps[:3], wrong_step, *bound.steps[4:]]}))
    for step in (bound.steps[3], bound.steps[4]):
        decision = resolve_standard_executor(step, plan=bound, current_case_scientific_runtime_authority=sealed,
                                             scientific_runtime_projection_sha256="f" * 64)
        assert decision.claimed_by in {"adjusted_association_estimates", "absolute_risk_context"}
    from easyicu.research_agent.authority.run_input import _declares_host_cohort_products
    assert _declares_host_cohort_products(bound.steps[0])
    for step in [*bound.steps[1:3], bound.steps[-1]]:
        decision = resolve_standard_executor(step, plan=bound)
        assert decision.claimed_by, (step.step_id, decision)
    if subgroup:
        assert stated_subgroup_coverage(context, bound).complete
    from easyicu.research_agent.planning.primary_result_contract import validate_required_primary_result
    validate_required_primary_result(plan=bound, context=context)
    rng = np.random.default_rng(740)
    frame = pd.DataFrame({"stay_id": np.arange(400), "age": rng.uniform(20, 85, 400),
                          "sex": rng.integers(0, 2, 400), "chronic_flag": rng.integers(0, 2, 400),
                          "admission_type": rng.choice(["elective", "emergency", "urgent"], 400),
                          outcome: rng.binomial(1, 0.35, 400).astype(bool)})
    if exposure == "age_group":
        frame["age_group"] = np.where(frame["age"] >= 65, 2, 1)
    if subgroup:
        frame.loc[:3, "sex"] = np.nan
    from easyicu.research_agent.execution.runners import association_effect_modification as owner
    original_fit = owner.fit_estimator
    fit_calls = []
    fits = []

    def recorded_fit(**kwargs):
        fit_calls.append(kwargs)
        result = original_fit(**kwargs)
        fits.append(result)
        return result

    monkeypatch.setattr(owner, "fit_estimator", recorded_fit)
    summary = run_baseline_step(step_id="adjusted_association", plan=bound, authority=sealed, runtime_projection_sha256="f" * 64,
                                frame=frame, output_dir=tmp_path / "primary")
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator

    issues, _ = PrimaryModelContractValidator._planned_model_requirement_issues(
        step=bound.steps[3], contracts=summary["model_contracts"],
    )
    assert issues == []
    if subgroup:
        changed_contract = {**summary["model_contracts"][0], "effect_modifiers": []}
        issues, _ = PrimaryModelContractValidator._planned_model_requirement_issues(
            step=bound.steps[3], contracts=[changed_contract],
        )
        assert issues
    cohort_path = tmp_path / "synthetic.parquet"
    frame.to_parquet(cohort_path)
    assert PrimaryModelContractValidator().audit(
        step=bound.steps[3], step_summary=summary, context=context,
        completed_step_records=[], out_dir=tmp_path / "primary", cohort_path=cohort_path,
    ) == []
    assert summary["n_total"] == (396 if subgroup else 400) and len(summary["model_contracts"]) == 1
    estimates = pd.read_csv(tmp_path / "primary" / "adjusted_association_estimates.csv")
    model_rows = frame.loc[fit_calls[0]["y"].index]
    assert estimates["outcome_proportion"].tolist() == pytest.approx(
        [model_rows[outcome].mean()] * len(estimates)
    )
    if subgroup:
        assert set(estimates["stratum"].dropna().astype(int)) == {0, 1}
        assert estimates.loc[estimates["subgroup"].notna(), "interaction_p_value"].nunique() == 1
    if exposure == "chronic_flag":
        import statsmodels.formula.api as smf
        independent = smf.logit("death_numeric ~ chronic_flag * sex + age + C(admission_type)",
                                data=frame.assign(death_numeric=frame[outcome].astype(int))).fit(disp=False)
        for stratum in (0, 1):
            expected = np.exp(independent.params["chronic_flag"] + stratum * independent.params["chronic_flag:sex"])
            row = estimates.loc[estimates["stratum"].eq(stratum)].iloc[0]
            assert row["estimate"] == pytest.approx(expected, rel=1e-7)
            assert row["interaction_p_value"] == pytest.approx(independent.pvalues["chronic_flag:sex"], rel=1e-7)
    risk = run_baseline_step(step_id="absolute_risk_context", plan=bound, authority=sealed, runtime_projection_sha256="f" * 64,
                            frame=frame, output_dir=tmp_path / "risk", primary_table=estimates)
    assert risk["status"] == "ok"
    assert len(fit_calls) == 1
    # The figure owner receives the schema of the actual upstream product,
    # exactly as execution's typed-input resolver supplies it.
    bindings = {"table:adjusted_association_estimates": {"product_contract": {"columns": list(estimates.columns)}}}
    if "table:absolute_risk_context" in bound.steps[5].inputs:
        risk_table = pd.read_csv(tmp_path / "risk" / "absolute_risk_context.csv")
        bindings["table:absolute_risk_context"] = {"product_contract": {"columns": list(risk_table.columns)}}
    figure = resolve_standard_executor(bound.steps[5], plan=bound, resolved_bindings=bindings)
    assert figure.claimed_by == ("adjusted_association_figure" if exposure == "age" else "association_overview_figure")
    for step in bound.steps:
        decision = resolve_standard_executor(
            step, plan=bound, current_case_scientific_runtime_authority=sealed,
            scientific_runtime_projection_sha256="f" * 64, resolved_bindings=bindings,
        )
        assert decision.claimed_by is not None, (step.step_id, decision)
        selected = decision.render_selection()
        assert isinstance(selected, StandardExecutorSelection)
        compile(selected.code, f"<host:{step.step_id}>", "exec")
        if step.step_id in {"adjusted_association", "absolute_risk_context"}:
            assert "baseline_association_executor import run_baseline_step" in selected.code
    if exposure == "age":
        assert summary["scientific_runtime_receipt"]["observed_knots"] == pytest.approx(
            frame.dropna()[exposure].quantile([0.1, 0.5, 0.9]).tolist(),
        )
        assert [estimates["exposure_level"].min(), estimates["exposure_level"].max()] == pytest.approx(
            frame.dropna()[exposure].quantile([0.1, 0.9]).tolist(),
        )
        assert estimates["reference_level"].unique() == pytest.approx([frame.dropna()[exposure].median()])
        bad_risk = estimates.copy()
        bad_risk["risk_difference"] += 0.1
        with pytest.raises(ValueError, match="risk-difference cross-check"):
            run_baseline_step(step_id="absolute_risk_context", plan=bound, authority=sealed, runtime_projection_sha256="f" * 64,
                              frame=frame, output_dir=tmp_path / "risk", primary_table=bad_risk)
    estimates["n_events"] += 1
    with pytest.raises(ValueError, match="primary counts"):
        run_baseline_step(step_id="absolute_risk_context", plan=bound, authority=sealed, runtime_projection_sha256="f" * 64,
                          frame=frame, output_dir=tmp_path / "risk", primary_table=estimates)
    altered = bound.steps[3].model_copy(update={"model_requirements": [requirement.model_copy(update={"covariates": ["unrequested"]})]})
    with pytest.raises(ValueError, match="drifted"):
        sealed.validate_plan(bound.model_copy(update={"steps": [*bound.steps[:3], altered, *bound.steps[4:]]}))
    if subgroup:
        missing_modifier = bound.steps[3].model_copy(update={"model_requirements": [bound.steps[3].model_requirements[0].model_copy(update={"effect_modifiers": []})]})
        with pytest.raises(ValueError, match="subgroup parameter"):
            sealed.validate_plan(bound.model_copy(update={"steps": [*bound.steps[:3], missing_modifier, *bound.steps[4:]]}))
    from dataclasses import replace
    for fault in ({"separation_detected": True}, {"converged": False}, {"coefficient_covariance": ()}):
        monkeypatch.setattr(owner, "fit_estimator", lambda **kwargs: replace(fits[0], **fault))
        with pytest.raises(ValueError, match="not estimable"):
            owner.fit_baseline_association(frame, requirement=bound.steps[3].model_requirements[0], authority=sealed)
    from easyicu.research_agent.planning.requirement_coverage import stated_adjustment_coverage
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    review = build_plan_scientific_review(context=context, plan=bound, runtime_authority=sealed, require_reportable_capability=True)
    assert not [finding for finding in review.findings if finding.severity == "blocker"], review.findings
    if exposure == "sex":
        missing = context.model_copy(deep=True)
        missing.user_preferences.data_constraints = json.dumps({"stated_adjustments": [
            {"variable": "age", "concept": "age", "words": "adjust for age", "source_status": "available"},
        ]})
        assert not stated_adjustment_coverage(missing, bound).complete
        review = build_plan_scientific_review(context=missing, plan=bound, runtime_authority=sealed)
        assert "STATED_ADJUSTMENT_MISSING" in {finding.code for finding in review.findings}


def test_a_threshold_baseline_plan_stops_when_eligibility_empties_a_level():
    from easyicu.research_agent.planning.cohort_contract import CohortDefinition, ConceptPredicate, TimeWindow
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    context = baseline_context("age_group", covariates=("sex",))
    attempt, _ = baseline_plan_attempt(context)
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    plan, _ = runtime.bind_plan(attempt.output)
    assert build_plan_scientific_review(context=context, plan=plan, runtime_authority=runtime.current_case).approval_allowed
    restricted = plan.model_copy(update={"cohort": CohortDefinition(
        name="only_older_adults", inclusion=(ConceptPredicate(
            "age", TimeWindow("icu_admission", -24.0, 0.0), "first", ">=", 65,
        ),),
    )})
    review = build_plan_scientific_review(context=context, plan=restricted, runtime_authority=runtime.current_case)
    assert not review.approval_allowed
    finding, = [item for item in review.findings if item.code == "POPULATION_EMPTIES_EXPOSURE_LEVEL"]
    assert finding.severity == "blocker" and finding.remediation_route == "agent_plan_revision"
    assert "age_group" in finding.message and "g1" in finding.message
    assert restricted.cohort.inclusion  # Review cannot silently drop study-stated eligibility.


@pytest.mark.parametrize("subgroup", [None, "sex"])
def test_a_baseline_template_revision_passes_the_repair_guard(subgroup):
    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from easyicu.webserver.research_plan_revision import PlanRevisionGuard
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    context = baseline_context("admission_type", outcome="unit_death", covariates=("age",), subgroup=subgroup)
    attempt, _ = baseline_plan_attempt(context)
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    source, _ = runtime.bind_plan(attempt.output)
    # The reviewed primary model needs an input binding repair. This does not
    # authorize another model, figure, or sensitivity in the revision.
    source = source.model_copy(deep=True)
    source.steps[3].input_consumption_contracts = []
    review = build_plan_scientific_review(context=context, plan=source, runtime_authority=runtime.current_case)
    guard = PlanRevisionGuard.from_review(plan=source.model_dump(mode="json"), review=review)
    services = PipelineServices(plan_revision_guard=guard)
    revised_attempt, _ = baseline_plan_attempt(context)
    revised, _ = runtime.bind_plan(revised_attempt.output)
    services.validate_plan_revision(plan=revised, context=context)
    runtime.validate_plan(revised)
    assert build_plan_scientific_review(context=context, plan=revised, runtime_authority=runtime.current_case).approval_allowed
    extra = revised.steps[5].model_copy(update={"step_id": "unrequested_figure"})
    with pytest.raises(ResearchPipelineRunError) as error:
        services.validate_plan_revision(plan=revised.model_copy(update={"steps": [*revised.steps, extra]}), context=context)
    assert error.value.code == "revision_added_unrequested_analysis"
    assert guard.previous_plan == source.model_dump(mode="json")


def test_the_baseline_cohort_executor_only_publishes_verified_plan_products(tmp_path, monkeypatch):
    import pandas as pd
    from easyicu.research_agent.cohort.schema import materialize_locked_analysis_cohort
    from easyicu.research_agent.execution.owner_declaration import resolve_standard_executor
    from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step

    context = baseline_context()
    attempt, _ = baseline_plan_attempt(context)
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    plan, _ = runtime.bind_plan(attempt.output)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = run_dir / "steps/cohort_definition/outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("STEP_OUT_DIR", str(output))
    with pytest.raises(ValueError, match="verified plan materialization"):
        run_baseline_step(step_id="cohort_definition", plan=plan, authority=runtime.current_case,
                          runtime_projection_sha256="f" * 64)
    universe = run_dir / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2, 3], "sex": [0, 1, 1], "death": [False, True, False]}).to_parquet(universe)
    result = materialize_locked_analysis_cohort(run_dir=run_dir, plan=plan, universe_path=universe)
    assert result["status"] == "applied"
    decision = resolve_standard_executor(plan.steps[0], plan=plan, current_case_scientific_runtime_authority=runtime.current_case,
                                         scientific_runtime_projection_sha256="f" * 64)
    untyped = plan.model_copy(deep=True)
    untyped.steps[0].cohort_definition_spec = None
    assert resolve_standard_executor(
        untyped.steps[0], plan=untyped, current_case_scientific_runtime_authority=runtime.current_case,
        scientific_runtime_projection_sha256="f" * 64,
    ).claimed_by is None
    exec(compile(decision.render_selection().code, "<host:cohort_definition>", "exec"), {})
    summary = json.loads((output / "step_summary.json").read_text())
    assert summary["n_analysis_cohort"] == 3
    for product, key in (("artifact:analysis_cohort", "path"), ("table:cohort_flow", "flow_path")):
        assert (output / summary["output_files"][product]).read_bytes() == result[key].read_bytes()
    from easyicu.research_agent.authority.run_input import RunInputIdentityError
    published = output / summary["output_files"]["table:cohort_flow"]
    published.write_text("conflicting,flow\n")
    with pytest.raises(RunInputIdentityError, match="conflicting immutable"):
        exec(compile(decision.render_selection().code, "<host:cohort_definition>", "exec"), {})
    # The owner rechecks the ledger and bytes on every execution.
    result["flow_path"].write_text("broken,flow\n")
    with pytest.raises(ValueError, match="verified plan materialization"):
        exec(compile(decision.render_selection().code, "<host:cohort_definition>", "exec"), {})
