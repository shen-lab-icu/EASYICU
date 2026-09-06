from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm


def _fit(*, transformation=None, cov_type="cluster"):
    rng = np.random.default_rng(260906)
    n = 480
    x = rng.normal(size=n)
    age = rng.normal(size=n)
    groups = np.repeat(np.arange(120), 4)
    random_intercept = np.repeat(rng.normal(scale=0.7, size=120), 4)
    probability = 1 / (1 + np.exp(-(-0.8 + 0.3 * x + 0.4 * x**2 + random_intercept)))
    y = rng.binomial(1, probability)
    full = np.column_stack([np.ones(n), x, x**2, age])
    if transformation is not None:
        full = full @ transformation
    full = pd.DataFrame(full, columns=["b0", "b1", "b2", "b3"])
    restricted = pd.DataFrame({"const": np.ones(n), "x": x, "age": age})
    kwargs = (
        {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
        if cov_type == "cluster"
        else {}
    )
    fit = sm.GLM(y, full, family=sm.families.Binomial()).fit(**kwargs)
    return fit, restricted


def _run(fit, restricted):
    from easyicu.research_agent.execution.runners.nested_model_comparison import (
        cluster_robust_nested_wald,
    )

    return cluster_robust_nested_wald(fit=fit, restricted_design=restricted)


def test_nested_wald_agrees_with_statsmodels_explicit_nonlinear_restriction():
    fit, restricted = _fit()
    expected = fit.wald_test([[0, 0, 1, 0]], use_f=False, scalar=True)

    result = _run(fit, restricted)

    assert result["method"] == "cluster_robust_nested_wald_chi2"
    assert result["degrees_of_freedom"] == 1
    assert result["statistic"] == pytest.approx(float(expected.statistic), rel=1e-10)
    assert result["p_value"] == pytest.approx(float(expected.pvalue), rel=1e-10)


@pytest.mark.parametrize(
    "transformation",
    [
        np.eye(4)[:, [2, 0, 3, 1]],
        np.array(
            [[1, 0.3, 0.2, 0], [0, 1, 0.2, 0.5], [0.3, 0.2, 1, 0], [0, 0, 0.5, 1]]
        ),
        np.diag([2.0, 0.5, 4.0, 1.0]),
    ],
)
def test_nested_wald_is_invariant_to_full_basis_coordinates(transformation):
    fit, restricted = _fit()
    transformed, _ = _fit(transformation=transformation)

    actual = _run(transformed, restricted)
    expected = _run(fit, restricted)

    assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-8)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-8)


def test_nested_wald_rejects_non_cluster_covariance():
    fit, restricted = _fit(cov_type="nonrobust")
    with pytest.raises(ValueError, match="cluster"):
        _run(fit, restricted)


@pytest.mark.parametrize(
    "mutation", ["non_nested", "rank_deficient", "row_order", "row_count", "nonfinite"]
)
def test_nested_wald_rejects_invalid_restricted_design(mutation):
    fit, restricted = _fit()
    if mutation == "non_nested":
        restricted["new_term"] = restricted["x"] ** 3
    elif mutation == "rank_deficient":
        restricted["duplicate"] = restricted["x"]
    elif mutation == "row_order":
        restricted = restricted.iloc[::-1]
    elif mutation == "row_count":
        restricted = restricted.iloc[:-1]
    else:
        restricted.iloc[0, 1] = np.nan
    with pytest.raises(ValueError):
        _run(fit, restricted)


@pytest.mark.parametrize("bad_covariance", ["singular", "nonfinite", "negative"])
def test_nested_wald_rejects_unusable_covariance_without_silent_rank_downgrade(
    bad_covariance,
):
    fit, restricted = _fit()
    covariance = fit.cov_params().to_numpy().copy()
    if bad_covariance == "singular":
        covariance[:] = 0
    elif bad_covariance == "nonfinite":
        covariance[0, 0] = np.nan
    else:
        covariance *= -1
    stub = SimpleNamespace(
        model=fit.model,
        params=fit.params,
        cov_type=fit.cov_type,
        cov_params=lambda: covariance,
        wald_test=lambda *args, **kwargs: pytest.fail(
            "invalid covariance reached statsmodels"
        ),
    )
    with pytest.raises(ValueError):
        _run(stub, restricted)
