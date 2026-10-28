"""Synthetic checks for explicitly declared ICU outcome estimands."""

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.schema import PlannedModelRequirement


def test_count_outcome_is_a_typed_plan_requirement():
    requirement = PlannedModelRequirement(
        requirement_id="episodes", outcome="episodes", outcome_type="count",
        method_family="statsmodels_negative_binomial", exposure_source="exposure",
        analysis_role="primary", analysis_set="complete_case", covariates=[],
        outcome_definition={
            "estimand": "Adjusted episode rate ratio per unit exposure time",
            "count": {"family": "negative_binomial", "exposure_column": "followup"},
        },
    )
    assert requirement.outcome_definition.count.exposure_column == "followup"


def test_proportional_odds_recovers_common_or_and_reports_score_check():
    from easyicu.research_agent.methods.ordered_trends import fit_outcome_model

    rng = np.random.default_rng(41)
    x = rng.normal(size=2400)
    levels = ["poor", "limited", "good", "excellent"]
    y = np.asarray(levels)[np.digitize(0.7 * x + rng.logistic(size=x.size), [-1, 0.5, 1.7])]
    result = fit_outcome_model(
        X=pd.DataFrame({"x": x}), y=pd.Series(y), term="x", kind="proportional_odds",
        ordered_levels=levels,
    )
    assert result.estimate == pytest.approx(np.exp(0.7), rel=0.15)
    assert result.ci_low < np.exp(0.7) < result.ci_high
    assert result.diagnostics["proportional_odds"]["method"] == "proportional_odds_score"
    assert not result.diagnostics["proportional_odds"]["violated"]


def test_proportional_odds_violation_is_detected():
    from easyicu.research_agent.methods.ordered_trends import fit_outcome_model

    rng = np.random.default_rng(71)
    x = rng.integers(0, 2, size=3000)
    probabilities = np.where(x[:, None] == 0, [0.15, 0.4, 0.35, 0.1], [0.4, 0.1, 0.1, 0.4])
    y = (rng.random(x.size)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
    result = fit_outcome_model(
        X=pd.DataFrame({"x": x}), y=pd.Series(y), term="x", kind="proportional_odds",
        ordered_levels=["0", "1", "2", "3"],
    )
    assert result.diagnostics["proportional_odds"]["violated"]
    assert result.diagnostics["proportional_odds"]["p_value"] < 0.001


def test_negative_binomial_handles_offset_and_overdispersion():
    from easyicu.research_agent.methods.ordered_trends import fit_outcome_model

    rng = np.random.default_rng(17)
    x = rng.integers(0, 2, size=2600)
    exposure_time = np.exp(0.9 * x + rng.normal(0, 0.2, x.size))
    mu = exposure_time * np.exp(0.3 + 0.6 * x)
    y = rng.poisson(rng.gamma(shape=1 / 0.8, scale=mu * 0.8))
    args = dict(X=pd.DataFrame({"x": x}), y=pd.Series(y), term="x", exposure_time=exposure_time)
    nb = fit_outcome_model(kind="negative_binomial", **args)
    poisson = fit_outcome_model(kind="count_poisson", **args)
    assert nb.estimate == pytest.approx(np.exp(0.6), rel=0.15)
    assert nb.ci_low < np.exp(0.6) < nb.ci_high
    assert nb.diagnostics["aic"] < poisson.diagnostics["aic"] - 100
    assert nb.diagnostics["dispersion"]["overdispersed"]
    assert nb.diagnostics["dispersion"]["p_value"] < 0.001


@pytest.mark.parametrize("death_score", [-1, 0])
def test_free_days_death_rule_bounds_and_interval_union(death_score):
    from easyicu.concept.free_days import derive_free_days

    frame = pd.DataFrame({
        "support_intervals": [[[0, 48], [24, 72]], [], [[-24, 800]], [[0, 24]]],
        "death_hours": [np.nan, 400, np.nan, 700],
        "followup_hours": [672, 400, 672, 700],
    })
    values, audit = derive_free_days(
        frame, horizon_days=28, support_concept="mech_vent", death_score=death_score,
    )
    assert values.tolist() == [25, death_score, 0, 27]
    assert audit["by_death_status"]["died_by_horizon"]["n"] == 1
    assert audit["distribution"]["n"] == 4
    assert "rows" not in audit and "patient_ids" not in audit


def test_no_episodes_or_incomplete_followup_is_typed_refusal():
    from easyicu.concept.free_days import FreeDaysDerivationError, derive_free_days

    frame = pd.DataFrame({"death_hours": [np.nan], "followup_hours": [672]})
    with pytest.raises(FreeDaysDerivationError) as absent:
        derive_free_days(frame, horizon_days=28, support_concept="vent_ind", death_score=-1)
    assert absent.value.code == "support_intervals_unavailable"
    frame["support_intervals"] = [[]]
    frame["followup_hours"] = 100
    with pytest.raises(FreeDaysDerivationError) as incomplete:
        derive_free_days(frame, horizon_days=28, support_concept="vent_ind", death_score=0)
    assert incomplete.value.code == "horizon_followup_unavailable"


def _definition(kind):
    if kind == "ordinal":
        return {"estimand": "Adjusted common odds ratio toward higher recovery levels",
                "ordered_levels": ["poor", "limited", "good", "excellent"]}
    if kind == "free_days":
        return {"estimand": "Common odds ratio of more ventilator-free days through ICU day 28, death -1",
                "free_days": {"horizon_days": 28, "support_concept": "mech_vent", "death_score": -1}}
    return {"estimand": "Adjusted count rate ratio per follow-up day",
            "count": {"family": "poisson", "exposure_column": "followup"}}


def _requirement(kind):
    from easyicu.research_agent.contracts.model_terms import OutcomeDefinition

    definition = OutcomeDefinition.model_validate(_definition(kind))
    return PlannedModelRequirement(
        requirement_id="primary", outcome="outcome", outcome_type=kind,
        method_family=definition.estimator, outcome_definition=definition,
        exposure_source="x", analysis_role="primary", analysis_set="complete_case",
        covariates=[], model_terms=[{"name": "x", "role": "exposure",
                                    "coding": "continuous", "transform": "identity"}],
    )


@pytest.mark.parametrize("kind", ["ordinal", "free_days", "count"])
def test_new_types_refuse_an_implicit_estimand_or_wrong_definition(kind):
    payload = _requirement(kind).model_dump(mode="json")
    payload.pop("outcome_definition")
    with pytest.raises(ValueError, match="explicit outcome_definition"):
        PlannedModelRequirement.model_validate(payload)
    payload["outcome_definition"] = _definition("ordinal" if kind != "ordinal" else "count")
    with pytest.raises(ValueError, match="match outcome_type"):
        PlannedModelRequirement.model_validate(payload)


@pytest.mark.parametrize("kind", ["ordinal", "free_days", "count"])
def test_adjusted_association_routes_new_outcomes_and_emits_auditable_products(tmp_path, kind, monkeypatch):
    from easyicu.research_agent.execution.runners import adjusted_association_executor as outcome_models
    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        adjusted_association_executor_owns_step, adjusted_association_executor_code,
        run_adjusted_association_from_env,
    )
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator

    rng = np.random.default_rng(4)
    x = rng.normal(size=650)
    frame = pd.DataFrame({"x": x})
    if kind == "ordinal":
        ranks = np.digitize(0.5 * x + rng.logistic(size=len(x)), [-1, 0, 1])
        frame["outcome"] = np.asarray(_definition(kind)["ordered_levels"])[ranks]
    elif kind == "count":
        frame["followup"] = np.exp(0.8 * x)
        frame["outcome"] = rng.poisson(np.exp(0.4 * x) * frame["followup"])
    else:
        days = np.clip(np.floor(14 + 3 * x + rng.logistic(size=len(x))), 0, 28)
        frame["support_intervals"] = [[[0, (28 - days_i) * 24]] for days_i in days]
        frame["death_hours"] = np.where(rng.random(len(x)) < 0.1, 100, np.nan)
        frame["followup_hours"] = 672
    requirement = _requirement(kind)
    step = AnalysisStep(step_id="primary", intent="Fit the prespecified association.",
                        method="adjusted_association_models", planned_analysis_role="primary",
                        inputs=["artifact:analysis_cohort", "x", "outcome"],
                        expected_outputs=["table:adjusted_association_estimates"],
                        model_requirements=[requirement])
    assert adjusted_association_executor_owns_step(step)
    assert "outcome_definition" in adjusted_association_executor_code(step)
    trace = []
    original = outcome_models.fit_outcome_model
    def tracked(**kwargs):
        trace.append(kwargs["kind"])
        return original(**kwargs)
    monkeypatch.setattr(outcome_models, "fit_outcome_model", tracked)
    from easyicu.research_agent.contracts.association_execution import association_estimator_kind
    summary = run_adjusted_association_from_env(
        requirement_id="primary", exposure="x", outcome="outcome", covariates=[],
        model_terms=requirement.model_terms, estimator_kind=association_estimator_kind(requirement),
        method_family=requirement.method_family, outcome_definition=requirement.outcome_definition.model_dump(),
        analysis_set="complete_case", analysis_role="primary", frame=frame,
        output_dir=tmp_path, emit_step_summary=False,
    )
    assert trace == [association_estimator_kind(requirement)]
    assert summary["model_contracts"][0]["outcome_type"] == kind
    assert summary["model_contracts"][0]["event_n"] is None
    assert summary["outcome_summary"]["n"] == len(x)
    assert (tmp_path / "outcome_distribution.csv").is_file()
    if kind != "count":
        assert (tmp_path / "outcome_distribution.svg").is_file()
    if kind == "free_days":
        assert summary["free_days_derivation_audit"]["death_score"] == -1
        assert "median" in summary["outcome_summary"]
    contract = summary["model_contracts"][0]
    assert PrimaryModelContractValidator._expected_denominator(
        frame=frame, outcome="outcome", outcome_type=kind,
        covariates=[], contract=contract, raw_exposure_source="x",
    ) == (len(x), None)
    from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, VariableRole
    context = ResearchContext(
        research_question="Estimate the prespecified outcome association.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=len(x), id_columns=["stay_id"]),
        variables=[ConceptDescriptor(name="x", role=VariableRole.OTHER, dtype="float64"),
                   ConceptDescriptor(name="outcome", role=VariableRole.OUTCOME, dtype="object" if kind == "ordinal" else "int64")],
        primary_exposure="x", target_outcome="outcome",
    )
    cohort_path = tmp_path / "synthetic.parquet"
    frame.assign(stay_id=np.arange(len(frame))).to_parquet(cohort_path)
    assert PrimaryModelContractValidator().audit(
        step=step, step_summary=summary, context=context, completed_step_records=[],
        out_dir=tmp_path, cohort_path=cohort_path,
    ) == []
    coefficients = pd.read_csv(tmp_path / "adjusted_association_coefficients.csv")
    coefficients["effect_scale"] = "coefficient"
    coefficients.to_csv(tmp_path / "adjusted_association_coefficients.csv", index=False)
    findings = PrimaryModelContractValidator().audit(
        step=step, step_summary=summary, context=context, completed_step_records=[],
        out_dir=tmp_path, cohort_path=cohort_path,
    )
    assert "effect_scale_model_family_mismatch" in str([item.detail for item in findings])


@pytest.mark.parametrize("kind", ["ordinal", "count", "free_days"])
def test_outcome_estimand_change_invalidates_reviewed_plan_signature(kind):
    from easyicu.research_agent.authority.plan_scope import plan_signature
    from types import SimpleNamespace
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewStateDrift, HumanReviewDecision, build_pipeline_workflow,
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ValidationFinding

    requirement = _requirement(kind)
    before = AnalysisPlan(research_question="Estimate the specified association.", steps=[AnalysisStep(
        step_id="primary", intent="Fit the reviewed estimand", model_requirements=[requirement],
        method="adjusted_association_models", expected_outputs=["table:adjusted_association_estimates"],
    )])
    payload = before.model_dump(mode="json")
    definition = payload["steps"][0]["model_requirements"][0].setdefault(
        "outcome_definition", requirement.outcome_definition.model_dump(mode="json"),
    )
    if kind == "free_days":
        definition["free_days"]["death_score"] = 0
    elif kind == "ordinal":
        definition["ordered_levels"] = list(reversed(definition["ordered_levels"]))
    else:
        definition["count"]["exposure_column"] = "another_followup"
    after = AnalysisPlan.model_validate(payload)
    assert plan_signature(before) != plan_signature(after)
    calls = []
    handoff = SimpleNamespace(
        aborted_result=None, plan=before, evidence=SimpleNamespace(records=lambda: []),
        findings=[ValidationFinding(validator="scientific_stop_auditor", severity="error",
                                    message="Confirm the specified scientific estimand.",
                                    detail={"reason": "scientific_stop"})],
    )
    workflow = build_pipeline_workflow(
        plan_invoker=lambda: calls.append("plan") or handoff,
        execute_invoker=lambda _: calls.append("execute"),
        write_invoker=lambda *_: "written", finalise_invoker=lambda *_: "final",
        human_review_invoker=lambda current: human_review_requests_for_plan(
            findings=current.findings, plan=current.plan, evidence=current.evidence),
    )
    paused = workflow.start()
    handoff.plan = after
    with pytest.raises(HumanReviewStateDrift):
        request = paused.requests[0]
        workflow.resume([HumanReviewDecision(
            review_id=request.review_id, authority_sha256=request.authority_sha256,
            decision="approved", reviewer="researcher", decided_at="2026-10-28T00:00:00Z",
        )])
    assert calls == ["plan"]


@pytest.mark.parametrize("kind", ["proportional_odds", "count_poisson", "negative_binomial"])
def test_declared_clusters_determine_covariance(kind):
    from easyicu.research_agent.methods.ordered_trends import fit_outcome_model

    rng = np.random.default_rng(29)
    groups = np.repeat(np.arange(100), 6)
    x = rng.normal(size=len(groups))
    latent = rng.normal(size=100)[groups] + 0.4 * x
    y = (np.digitize(latent + rng.logistic(size=len(x)), [-1, 0, 1])
         if kind == "proportional_odds" else rng.poisson(np.exp(latent)))
    args = dict(X=pd.DataFrame({"x": x}), y=pd.Series(y), term="x", kind=kind)
    if kind == "proportional_odds":
        args["ordered_levels"] = ["0", "1", "2", "3"]
    model = fit_outcome_model(**args)
    clustered = fit_outcome_model(**args, variance_estimator="cluster_robust", cluster_groups=groups)
    assert clustered.variance_estimator == "cluster_robust"
    assert clustered.cluster_count == 100
    assert clustered.estimate == pytest.approx(model.estimate)
    assert clustered.se != pytest.approx(model.se)


@pytest.mark.parametrize("bad_time", [0, -1, np.nan, np.inf])
def test_invalid_exposure_time_is_never_treated_as_unit_exposure(bad_time):
    from easyicu.research_agent.methods.ordered_trends import fit_outcome_model

    x = np.linspace(-1, 1, 40)
    times = np.ones(40)
    times[7] = bad_time
    with pytest.raises(ValueError, match="exposure time"):
        fit_outcome_model(X=pd.DataFrame({"x": x}), y=pd.Series(np.arange(40) % 5),
                          term="x", kind="count_poisson", exposure_time=times)


def test_log_offset_and_exposure_time_are_equivalent_and_aligned():
    from easyicu.research_agent.methods.ordered_trends import fit_outcome_model

    rng = np.random.default_rng(91)
    x = pd.DataFrame({"x": rng.normal(size=300)})
    time = pd.Series(np.exp(0.7 * x["x"]))
    y = pd.Series(rng.poisson(time * np.exp(0.4 * x["x"])))
    args = dict(X=x, y=y, term="x", kind="count_poisson")
    rate = fit_outcome_model(**args, exposure_time=time)
    offset = fit_outcome_model(**args, log_offset=np.log(time))
    assert rate.estimate == pytest.approx(offset.estimate)
    with pytest.raises(ValueError, match="align"):
        fit_outcome_model(**args, exposure_time=time[::-1])


def test_fractional_support_and_death_at_horizon_have_explicit_boundaries():
    from easyicu.concept.free_days import derive_free_days

    frame = pd.DataFrame({"support_intervals": [[[0, 25]], [], []],
                          "death_hours": [np.nan, 672, 673], "followup_hours": [672, 672, 673]})
    result, _ = derive_free_days(frame, horizon_days=28, support_concept="vent_ind", death_score=-1)
    assert result.tolist() == [26, -1, 28]


def test_common_or_is_a_recognised_ratio_in_association_figures():
    from easyicu.research_agent.execution.runners.effect_scale import describe_effect_scale

    scale = describe_effect_scale("common_odds_ratio")
    assert scale.recognised and scale.multiplicative and scale.null_value == 1


def test_numeric_float_counts_keep_their_distribution_frequencies():
    from easyicu.research_agent.contracts.model_terms import OutcomeDefinition
    from easyicu.research_agent.execution.runners.adjusted_association_executor import outcome_summaries

    summary, bars = outcome_summaries(pd.Series([0.0, 1.0, 1.0, np.nan]),
                                     OutcomeDefinition.model_validate(_definition("count")))
    assert bars["n"].tolist() == [1, 2]
    assert summary["missing_n"] == 1


@pytest.mark.parametrize("bad_count", [-1, 1.5, np.inf, np.nan])
def test_count_outcomes_refuse_invalid_counts(bad_count):
    from easyicu.research_agent.methods.ordered_trends import fit_outcome_model

    x = pd.DataFrame({"x": np.linspace(-1, 1, 40)})
    y = pd.Series(np.arange(40) % 5, dtype=float)
    y.iloc[4] = bad_count
    with pytest.raises(ValueError):
        fit_outcome_model(X=x, y=y, term="x", kind="count_poisson")


def test_count_estimator_must_match_its_prespecified_family():
    payload = _requirement("count").model_dump(mode="json")
    payload["method_family"] = "statsmodels_negative_binomial"
    with pytest.raises(ValueError, match="declared outcome estimator"):
        PlannedModelRequirement.model_validate(payload)


def test_free_day_summary_reports_median_and_iqr_and_ordered_bars():
    from easyicu.research_agent.contracts.model_terms import OutcomeDefinition
    from easyicu.research_agent.execution.runners.adjusted_association_executor import outcome_summaries

    summary, bars = outcome_summaries(pd.Series([-1, 0, 5, 28]),
                                     OutcomeDefinition.model_validate(_definition("free_days")))
    assert (summary["q1"], summary["median"], summary["q3"]) == (-0.25, 2.5, 10.75)
    assert bars["n"].tolist() == [1, 1, 1, 1]
    _, bars = outcome_summaries(pd.Series(["poor", "good", "poor"]),
                               OutcomeDefinition.model_validate(_definition("ordinal")))
    assert bars["level"].tolist() == _definition("ordinal")["ordered_levels"]
    assert bars["n"].tolist() == [2, 0, 1, 0]
