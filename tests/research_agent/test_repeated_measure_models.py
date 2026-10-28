"""Small synthetic longitudinal fits; no clinical or benchmark records."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from easyicu.research_agent.methods.gee import GEEError, fit_gee
from easyicu.research_agent.methods.mixed_effects import fit_mixed_effects


def panel(seed=39, n=100, visits=6):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n), visits)
    time = np.tile(np.arange(visits, dtype=float), n)
    intercept = rng.normal(0, 2, n)
    slope = rng.normal(0, 0.6, n)
    y = 4 + 1.25 * time + intercept[groups] + slope[groups] * time
    y += rng.normal(0, 1, len(groups))
    return y, pd.DataFrame({"time": time}), groups


def test_random_slopes_recover_growth_covariance_and_baseline_icc():
    y, x, groups = panel()
    result = fit_mixed_effects(y, x, groups, random_slope="time")
    scale = float(x.time.std(ddof=0))
    reference = sm.MixedLM(y, sm.add_constant(x), groups, exog_re=sm.add_constant(x / scale)).fit(reml=True, method="lbfgs")
    assert result.coef[1] == pytest.approx(1.25, abs=0.17)
    assert result.group_var == pytest.approx(4, abs=1.5)
    assert result.random_slope_var == pytest.approx(0.36, abs=0.15)
    assert result.icc == pytest.approx(0.8, abs=0.10)
    np.testing.assert_allclose(result.coef, reference.fe_params, rtol=1e-7)
    inverse_scale = np.diag([1.0, 1.0 / scale])
    np.testing.assert_allclose(result.random_effect_cov, inverse_scale @ reference.cov_re @ inverse_scale, rtol=1e-7)
    assert result.ci_low[1] < 1.25 < result.ci_high[1]
    assert result.to_json()["icc_reference"] == "time_zero"


def test_random_intercept_icc_is_reported():
    rng = np.random.default_rng(90)
    groups = np.repeat(np.arange(80), 5)
    time = np.tile(np.arange(5), 80)
    y = 3 + 0.8 * time + rng.normal(0, 2, 80)[groups] + rng.normal(0, 1, 400)
    result = fit_mixed_effects(y, pd.DataFrame({"time": time}), groups)
    assert result.icc == pytest.approx(0.8, abs=0.10)
    assert result.to_json()["icc_reference"] == "constant"


def test_gee_robust_se_agrees_with_cluster_bootstrap_and_reports_qic():
    y, x, groups = panel(n=120, visits=12)
    result = fit_gee(y, x, groups, family="gaussian", cov_struct="independence")
    rng = np.random.default_rng(14)
    estimates = []
    for _ in range(250):
        selected = rng.integers(0, 120, 120)
        rows = (selected[:, None] * 12 + np.arange(12)).ravel()
        estimates.append(np.linalg.lstsq(sm.add_constant(x.iloc[rows]), y[rows], rcond=None)[0][1])
    assert result.se[1] == pytest.approx(np.std(estimates, ddof=1), rel=0.22)
    assert result.qic is not None and np.isfinite(result.qic)
    assert result.qic_scale > 0
    assert result.to_json()["cov_type"] == "robust"


def test_ar1_fit_keeps_time_outcome_and_cluster_alignment_after_row_shuffle():
    rng = np.random.default_rng(44)
    groups = np.repeat(np.arange(40), 5)
    time = np.tile(np.arange(5, dtype=float), 40)
    noise = rng.normal(size=(40, 5))
    for i in range(1, 5):
        noise[:, i] += 0.65 * noise[:, i - 1]
    y = 2 + 0.4 * time + noise.ravel()
    x = pd.DataFrame({"time": time})
    expected = fit_gee(y, x, groups, family="gaussian", cov_struct="ar1", time=time)
    rows = rng.permutation(len(y))
    shuffled = fit_gee(y[rows], x.iloc[rows], groups[rows], family="gaussian", cov_struct="ar1", time=time[rows])
    np.testing.assert_allclose(shuffled.coef, expected.coef, rtol=1e-8)
    np.testing.assert_allclose(shuffled.se, expected.se, rtol=1e-8)


def test_ar1_refuses_duplicate_times():
    y, x, groups = panel(n=10)
    time = x.time.to_numpy().copy()
    time[1] = time[0]
    with pytest.raises(GEEError, match="unique"):
        fit_gee(y, x, groups, family="gaussian", cov_struct="ar1", time=time)


def test_ar1_refuses_a_gap_in_the_reviewed_grid():
    y, x, groups = panel(n=10)
    time = x.time.to_numpy().copy()
    time[2] += 0.5
    with pytest.raises(GEEError, match="unit-spaced"):
        fit_gee(y, x, groups, family="gaussian", cov_struct="ar1", time=time)
