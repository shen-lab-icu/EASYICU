"""Synthetic MI integration across ICU outcome families and frozen transport."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
from easyicu.research_agent.contracts.model_terms import DataPreparationPlan, PreparationRefusal
from easyicu.research_agent.execution.runners import adjusted_association_executor as owner
from easyicu.research_agent.methods.missing_data import ImputationRefusal, rubin_pool


def _input(kind, strategy="multiple_imputation"):
    rng = np.random.default_rng(431)
    z, x = rng.normal(size=(2, 240))
    frame = pd.DataFrame({"z": z, "x": x})
    if kind in {"count_poisson", "negative_binomial"}:
        frame["followup"] = np.exp(0.2 * z + rng.normal(0, 0.3, len(z)))
        mean = np.exp(0.3 * z + 0.2 * x) * frame.followup
        frame["y"] = rng.poisson(
            rng.gamma(2, mean / 2) if kind == "negative_binomial" else mean
        )
        definition = {
            "estimand": "Adjusted event rate ratio per observed follow-up time",
            "count": {
                "family": "negative_binomial" if kind == "negative_binomial" else "poisson",
                "exposure_column": "followup",
            },
        }
        method = "statsmodels_" + kind
    else:
        ranks = np.digitize(0.3 * z + 0.2 * x + rng.logistic(size=len(z)), [-0.7, 0.7])
        if kind == "ordinal":
            frame["y"] = np.asarray(["low", "middle", "high"])[ranks]
            definition = {
                "estimand": "Adjusted common odds ratio toward higher recovery levels",
                "ordered_levels": ["low", "middle", "high"],
            }
        else:
            days = np.asarray([0, 7, 28])[ranks]
            frame["support_intervals"] = [[[0, float((28 - value) * 24)]] for value in days]
            frame["death_hours"] = np.nan
            frame["followup_hours"] = 672
            definition = {
                "estimand": "Common odds ratio toward more support-free days at day 28",
                "free_days": {"horizon_days": 28, "support_concept": "mech_vent", "death_score": -1},
            }
        method = "statsmodels_ordered_logit"
    frame.loc[::3, "x"] = np.nan
    choice = DataPreparationPlan(missing_data={
        "strategy": strategy, "m": 4, "iterations": 3,
        "variables": [{"name": "x", "method": "pmm"}],
    })
    arguments = dict(
        requirement_id="fit", exposure="z", outcome="y", covariates=["x"],
        model_terms=[{"name": name, "role": role, "coding": "continuous", "transform": "identity"}
                     for name, role in (("z", "exposure"), ("x", "covariate"))],
        estimator_kind=kind if kind in {"count_poisson", "negative_binomial"} else "proportional_odds",
        method_family=method, outcome_definition=definition,
        analysis_set="source_aware", analysis_role="primary", emit_step_summary=False,
        data_preparation=choice,
    )
    return frame, arguments


@pytest.mark.parametrize("kind", ["ordinal", "free_days", "count_poisson", "negative_binomial"])
@pytest.mark.parametrize("strategy", ["multiple_imputation", "complete_case"])
@pytest.mark.parametrize("clustered", [False, True])
def test_new_outcome_models_pool_the_same_declared_fit_and_keep_sensitivity(tmp_path, monkeypatch, kind, strategy, clustered):
    frame, arguments = _input(kind, strategy)
    if clustered:
        frame["subject_id"] = np.repeat(np.arange(40), 6)
        arguments["dependence"] = {"group_source": "subject_id", "group_derivation": "identity"}
    fits = []
    original = owner.fit_outcome_model

    def tracked(**kwargs):
        fit = original(**kwargs)
        fits.append(fit)
        return fit

    monkeypatch.setattr(owner, "fit_outcome_model", tracked)
    summary = owner.run_adjusted_association_from_env(frame=frame, output_dir=tmp_path, **arguments)
    receipt = json.loads((tmp_path / "data_preparation_provenance.json").read_text())
    n = len(frame) if strategy == "multiple_imputation" else int(frame.x.notna().sum())
    assert summary["n_total"] == n
    contract = summary["model_contracts"][0]
    assert PrimaryModelContractValidator._expected_denominator(
        frame=frame, outcome="y", outcome_type=contract["outcome_type"],
        covariates=["x"], contract=contract, raw_exposure_source="z",
    ) == (n, None)
    pooled = receipt["missing_data"]["pooling"]["z"]
    assert receipt["missing_data"]["m"] == 4
    assert pooled["between_variance"] > 0
    assert pooled["within_variance"] > 0
    assert len(fits) == 5
    slopes = [next(term for term in fit.terms if term.term == "z") for fit in fits[1:]]
    parameter_count = 3 if kind == "count_poisson" else 4
    expected = rubin_pool(
        [np.log(term.estimate) for term in slopes], [term.se**2 for term in slopes],
        complete_df=39 if clustered else len(frame) - parameter_count,
    )
    assert pooled["df"] == pytest.approx(expected.df)
    estimates = pd.read_csv(tmp_path / "adjusted_association_estimates.csv")
    if strategy == "multiple_imputation":
        assert np.log(estimates.estimate.iloc[0]) == pytest.approx(pooled["estimate"])
        assert np.log(estimates.ci_low.iloc[0]) == pytest.approx(pooled["ci_low"])
        assert receipt["sensitivity"]["strategy"] == "complete_case"
        assert receipt["sensitivity"]["n"] < n
        diagnostics = summary["diagnostics"]["multiple_imputation"]
        assert diagnostics["diagnostics_scope"] == "per_imputation"
        assert len(diagnostics["imputations"]) == 4
    else:
        assert receipt["sensitivity"]["strategy"] == "multiple_imputation"
        assert receipt["sensitivity"]["n"] == len(frame)


@pytest.mark.parametrize("error", [None, np.nan, 0, -1])
def test_missing_pooled_variance_refuses_without_a_complete_case_substitute(tmp_path, monkeypatch, error):
    frame, arguments = _input("ordinal")
    original = owner.fit_outcome_model

    def no_variance(**kwargs):
        fit = original(**kwargs)
        return replace(fit, terms=tuple(replace(term, se=error) for term in fit.terms))

    monkeypatch.setattr(owner, "fit_outcome_model", no_variance)
    with pytest.raises(ImputationRefusal) as caught:
        owner.run_adjusted_association_from_env(frame=frame, output_dir=tmp_path, **arguments)
    assert caught.value.reason_code == "mi_analysis_variance_unavailable"
    assert not (tmp_path / "adjusted_association_estimates.csv").exists()


@pytest.mark.parametrize("estimate", [None, np.nan, 0, -1])
def test_an_unpoolable_effect_refuses_with_a_typed_mi_reason(tmp_path, monkeypatch, estimate):
    frame, arguments = _input("ordinal")
    original = owner.fit_outcome_model

    def invalid_effect(**kwargs):
        fit = original(**kwargs)
        return replace(fit, terms=tuple(replace(term, estimate=estimate) for term in fit.terms))

    monkeypatch.setattr(owner, "fit_outcome_model", invalid_effect)
    with pytest.raises(ImputationRefusal, match="mi_analysis_fit_failed"):
        owner.run_adjusted_association_from_env(frame=frame, output_dir=tmp_path, **arguments)
    assert not (tmp_path / "adjusted_association_estimates.csv").exists()


def test_any_imputation_assumption_warning_keeps_the_outcome_claim_ceiling(tmp_path, monkeypatch):
    frame, arguments = _input("ordinal")
    original = owner.fit_outcome_model
    calls = []

    def assumption_warning(**kwargs):
        fit = original(**kwargs)
        calls.append(fit)
        if len(calls) == 5:
            return replace(fit, diagnostics={**fit.diagnostics, "proportional_odds": {"violated": True}},
                           notes="Proportional-odds assumption violated; common OR is model-dependent.")
        return fit

    monkeypatch.setattr(owner, "fit_outcome_model", assumption_warning)
    summary = owner.run_adjusted_association_from_env(frame=frame, output_dir=tmp_path, **arguments)
    assert summary["claim_ceiling"] == "analysis_only"
    assert summary["diagnostics"]["proportional_odds"]["scope"] == "any_imputation"
    table = pd.read_csv(tmp_path / "adjusted_association_estimates.csv")
    assert "model-dependent" in table.notes.iloc[0]


def test_invalid_count_offset_refuses_with_a_typed_mi_reason(tmp_path):
    frame, arguments = _input("count_poisson")
    frame.loc[1, "followup"] = np.nan
    with pytest.raises(ImputationRefusal) as caught:
        owner.run_adjusted_association_from_env(frame=frame, output_dir=tmp_path, **arguments)
    assert caught.value.reason_code == "mi_analysis_coordinate_unobserved"
    assert not (tmp_path / "adjusted_association_estimates.csv").exists()


@pytest.mark.parametrize("strategy", ["multiple_imputation", "complete_case"])
def test_imputed_analysis_errors_keep_the_signed_primary_and_a_typed_reason(tmp_path, monkeypatch, strategy):
    frame, arguments = _input("ordinal", strategy)
    original = owner.fit_outcome_model
    calls = []

    def fail_imputed_fit(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            raise ValueError("Synthetic unavailable analysis fit")
        return original(**kwargs)

    monkeypatch.setattr(owner, "fit_outcome_model", fail_imputed_fit)
    if strategy == "multiple_imputation":
        with pytest.raises(ImputationRefusal, match="mi_analysis_fit_failed"):
            owner.run_adjusted_association_from_env(frame=frame, output_dir=tmp_path, **arguments)
        assert not (tmp_path / "adjusted_association_estimates.csv").exists()
    else:
        summary = owner.run_adjusted_association_from_env(frame=frame, output_dir=tmp_path, **arguments)
        assert summary["n_total"] == int(frame.x.notna().sum())
        sensitivity = summary["data_preparation"]["sensitivity"]
        assert sensitivity == {"strategy": "multiple_imputation", "status": "not_estimable",
                               "reason_code": "mi_analysis_fit_failed"}


@pytest.mark.parametrize("strategy", ["multiple_imputation", "complete_case"])
def test_external_mi_refuses_before_validation_inputs_are_read(tmp_path, strategy):
    from easyicu.research_agent.execution.runners.prediction_validation_executor import run_external_validation_step

    _, arguments = _input("ordinal", strategy)
    with pytest.raises(ImputationRefusal) as caught:
        run_external_validation_step(
            spec={"development_database": "miiv", "validation_databases": ["eicu"],
                  "frozen_model_sha256": "a" * 64, "predictor_concepts": ["x", "z"], "outcome_concept": "y"},
            run_dir=tmp_path, resolved_inputs=tmp_path / "must_not_read.json",
            step_id="external", out_dir=tmp_path / "outputs",
            data_preparation=arguments["data_preparation"],
        )
    assert caught.value.reason_code == "mi_external_validation_model_unavailable"
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("missing_policy", ["multiple_imputation", "complete_case", None])
def test_signed_external_mi_reaches_the_guard_and_is_rejected_by_review(tmp_path, monkeypatch, missing_policy):
    from easyicu.research_agent.contracts.model_terms import preparation_method
    from easyicu.research_agent.contracts.prediction_validation import (
        EXTERNAL_ACTION, EXTERNAL_OUTPUTS, FROZEN_MODEL_INPUT, ExternalValidationSpec,
        external_cohort_input, external_validation_method, external_validation_step_matches,
    )
    from easyicu.research_agent.execution.runners.prediction_validation_executor import prediction_external_executor_code
    from easyicu.research_agent.planning.primary_result_contract import preparation_step_issue
    from easyicu.research_agent.planning.progressive_contract import coordinate_owned_step_fields
    from easyicu.research_agent.schema import AnalysisStep

    _, arguments = _input("ordinal", missing_policy or "multiple_imputation")
    plan = arguments["data_preparation"] if missing_policy else DataPreparationPlan(scaling_coding=[{"variable": "x", "standardize": True}])
    reason = "mi_external_validation_model_unavailable" if missing_policy else "preparation_external_frozen_pipeline"
    spec = ExternalValidationSpec(
        development_database="miiv", validation_databases=("eicu",), frozen_model_sha256="a" * 64,
        predictor_concepts=("x", "z"), outcome_concept="y",
    )
    step = AnalysisStep(
        step_id="external", planned_analysis_role="secondary", scientific_action_id=EXTERNAL_ACTION,
        intent="Apply a frozen development model to an independent validation cohort.",
        method=preparation_method(external_validation_method(spec), plan),
        inputs=[FROZEN_MODEL_INPUT, external_cohort_input("eicu")], expected_outputs=list(EXTERNAL_OUTPUTS),
    )
    assert external_validation_step_matches(step)
    assert "data_preparation" in coordinate_owned_step_fields(
        module_id="custom_analysis", scientific_action_id=EXTERNAL_ACTION, planned_analysis_role="secondary",
    )
    assert preparation_step_issue(step, None) == reason
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(tmp_path / "must_not_read.json"))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "outputs"))
    with pytest.raises(PreparationRefusal, match=reason):
        exec(prediction_external_executor_code(step), {})
