"""Reference models retain the exact host family, references, retention and covariance."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from easyicu.research_agent.execution.runners.adjusted_association_executor import adjusted_association_executor_code
from easyicu.research_agent.skill_packages.adjusted_association_model.scripts.example_data import example_cohort, example_spec
from easyicu.research_agent.skill_packages.adjusted_association_model.scripts.load_cohort import load_cohort
from easyicu.research_agent.skill_packages.adjusted_association_model.scripts.run_analysis import run_analysis
from tests.support.reference_scripts import assert_numeric_parity, run_host_executor

VALUES = ["estimate", "standard_error", "ci_low", "ci_high"]


def _spec(method, clustered, retention=False):
    spec = example_spec()
    payload = spec.model_dump(mode="json")
    model = payload["model"]
    model["method_family"] = method
    if method == "statsmodels_ols":
        model.update(outcome="continuous_outcome", outcome_type="continuous")
    if clustered:
        model["dependence"] = {"group_source": "patient_id", "group_derivation": "identity"}
    if retention:
        model["analysis_set"] = "source_aware"
        model["baseline_missing_handling"] = {"covariates": ["age", "sex"]}
    return type(spec).model_validate(payload)


@pytest.mark.parametrize("method", ["statsmodels_logit_mle", "statsmodels_ols", "statsmodels_glm_poisson_log_robust"])
@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("retention", [False, True])
def test_model_matches_host_family_covariance_and_missing_rows(tmp_path, monkeypatch, method, clustered, retention):
    frame, spec = example_cohort(), _spec(method, clustered, retention)
    if retention:
        # The host retains a missing category only with >=20 estimable rows.
        frame.loc[30:59, "age"] = np.nan
        frame.loc[70:99, "sex"] = None
    fitter = {"statsmodels_logit_mle": sm.Logit, "statsmodels_ols": sm.OLS,
              "statsmodels_glm_poisson_log_robust": sm.GLM}[method]
    original_fit = fitter.fit
    fits = []

    def observe_fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        fits.append(result)
        return result

    # The owner omits p values from its CSV contract. Observe the actual fitted
    # objects to verify beta, p and uncertainty without fabricating z statistics.
    monkeypatch.setattr(fitter, "fit", observe_fit)
    host = run_host_executor(tmp_path, monkeypatch, frame, spec.host_step(), adjusted_association_executor_code)
    out = tmp_path / "standalone"
    summary = run_analysis(load_cohort(frame, spec), spec, out)
    assert len(fits) == 2
    assert fits[0].nobs == fits[1].nobs
    for field in ("params", "bse", "pvalues"):
        np.testing.assert_allclose(getattr(fits[0], field), getattr(fits[1], field), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(fits[0].conf_int(), fits[1].conf_int(), atol=1e-8, rtol=1e-8)
    expected = pd.read_csv(host / "adjusted_association_estimates.csv")
    actual = pd.read_csv(out / "adjusted_association_estimates.csv")
    assert_numeric_parity(actual, expected, counts=["n", "n_events", "cluster_count"], values=VALUES)
    assert actual.reference_level.iloc[0] == expected.reference_level.iloc[0] == 1
    expected_n = (239 if method == "statsmodels_ols" else 238) if retention else (234 if method == "statsmodels_ols" else 233)
    assert int(actual.n.iloc[0]) == expected_n
    variance = "cluster_robust" if clustered else "sandwich_robust" if "poisson" in method else "model_based"
    assert set(actual.variance_estimator) == {variance}
    assert summary["design_columns"] == ["treatment__is_0", "age", "sex__is_female"] or retention
    coefficients = pd.read_csv(out / "adjusted_association_coefficients.csv")
    host_coefficients = pd.read_csv(host / "adjusted_association_coefficients.csv")
    assert_numeric_parity(coefficients, host_coefficients, counts=["term", "term_role", "source_variable"], values=VALUES)
    # Ratio-scale terms encode beta exactly through log(estimate). The owner
    # does not publish coefficient p values; never replace these with a z guess.
    beta = coefficients.estimate if method == "statsmodels_ols" else np.log(coefficients.estimate)
    host_beta = host_coefficients.estimate if method == "statsmodels_ols" else np.log(host_coefficients.estimate)
    np.testing.assert_allclose(beta, host_beta, atol=1e-8, rtol=1e-8)
    assert "p_value" not in actual and "p_value" not in coefficients


def test_other_covariance_fails_association_parity(tmp_path, monkeypatch):
    frame, spec = example_cohort(), _spec("statsmodels_logit_mle", True)
    host = run_host_executor(tmp_path, monkeypatch, frame, spec.host_step(), adjusted_association_executor_code)
    changed = _spec("statsmodels_logit_mle", False)
    out = tmp_path / "different_covariance"
    run_analysis(load_cohort(frame, changed), changed, out)
    expected = pd.read_csv(host / "adjusted_association_estimates.csv")
    actual = pd.read_csv(out / "adjusted_association_estimates.csv")
    assert abs(actual.standard_error.iloc[0] - expected.standard_error.iloc[0]) > 1e-8
    with pytest.raises(AssertionError):
        assert_numeric_parity(actual, expected, counts=["n", "n_events"], values=VALUES)


def test_categorical_exposure_preserves_every_contrast_and_headline(tmp_path, monkeypatch):
    frame, spec = example_cohort(), example_spec()
    frame["treatment"] = np.asarray(["baseline", "middle", "upper"])[np.arange(len(frame)) % 3]
    frame.loc[0, "treatment"] = None
    payload = spec.model_dump(mode="json")
    model = payload["model"]
    model.update(exposure_levels=["baseline", "middle", "upper"],
                 exposure_reference_level="middle", primary_contrast_level="upper")
    model["model_terms"][0].update(coding="categorical", levels=model["exposure_levels"], reference_level="middle")
    spec = type(spec).model_validate(payload)
    host = run_host_executor(tmp_path, monkeypatch, frame, spec.host_step(), adjusted_association_executor_code)
    out = tmp_path / "standalone"
    run_analysis(load_cohort(frame, spec), spec, out)
    actual = pd.read_csv(out / "adjusted_association_estimates.csv")
    expected = pd.read_csv(host / "adjusted_association_estimates.csv")
    assert_numeric_parity(actual, expected, counts=["n", "n_events"], values=VALUES)
    assert actual.exposure_level.tolist() == ["baseline", "upper"]
    assert actual.reference_level.tolist() == ["middle", "middle"]
    assert actual.is_primary_contrast.tolist() == [False, True]
