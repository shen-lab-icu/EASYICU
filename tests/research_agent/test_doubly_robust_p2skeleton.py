"""Doubly robust AIPTW: known-answer coverage under randomisation, fail-closed positivity."""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.doubly_robust import (
    DoublyRobustError,
    aiptw_ate,
    positivity_diagnostics,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _randomised_cohort(seed: int, n: int = 1500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomised synthetic cohort: true propensity is known (0.5)."""

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    A = rng.binomial(1, 0.5, size=n).astype(float)
    logit = -0.5 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.7 * A
    Y = (rng.random(n) < _sigmoid(logit)).astype(float)
    return X, A, Y


def _true_ate_monte_carlo() -> float:
    """Ground truth for the DGP above via a large fixed-seed Monte Carlo."""

    rng = np.random.default_rng(999)
    n = 1_000_000
    X = rng.normal(size=(n, 2))
    linear = -0.5 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
    return float((_sigmoid(linear + 0.7) - _sigmoid(linear)).mean())


TRUE_ATE = _true_ate_monte_carlo()
SEEDS = list(range(20260901, 20260913))  # 12 fixed synthetic seeds


def test_aiptw_covers_truth_under_randomisation() -> None:
    """Known-answer test: randomised data, true PS = 0.5 known.

    Across fixed synthetic seeds the estimator must show small average bias
    and reasonable CI coverage of the Monte-Carlo ground truth.
    """

    assert 0.05 < TRUE_ATE < 0.30  # sanity: non-degenerate DGP
    estimates: list[float] = []
    covered = 0
    for seed in SEEDS:
        X, A, Y = _randomised_cohort(seed)
        res = aiptw_ate(X, A, Y, random_state=0)
        assert np.isfinite(res.ate)
        assert res.se > 0
        assert res.ci_low < res.ate < res.ci_high
        assert res.trim_proportion == 0.0
        assert res.n_treated + res.n_control == res.n == 1500
        estimates.append(res.ate)
        covered += int(res.ci_low <= TRUE_ATE <= res.ci_high)

    mean_bias = abs(float(np.mean(estimates)) - TRUE_ATE)
    assert mean_bias < 0.03
    assert covered >= 8  # ~95% nominal; 8/12 is a generous lower bound


def test_aiptw_is_deterministic() -> None:
    X, A, Y = _randomised_cohort(SEEDS[0])
    first = aiptw_ate(X, A, Y, random_state=0)
    second = aiptw_ate(X, A, Y, random_state=0)
    assert first.ate == second.ate
    assert first.se == second.se
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_aiptw_linear_outcome_model_path() -> None:
    X, A, Y = _randomised_cohort(SEEDS[0])
    res = aiptw_ate(X, A, Y, random_state=0, outcome_model="linear")
    assert np.isfinite(res.ate)
    assert res.se > 0
    # Propensity is correct here, so even the misspecified linear outcome
    # model stays consistent; allow a looser band than the logistic path.
    assert abs(res.ate - TRUE_ATE) < 0.08


def test_positivity_trim_triggered_fail_closed() -> None:
    """Near-separated propensity triggers fail-closed with a trim proportion."""

    rng = np.random.default_rng(11)
    n = 400
    A = rng.binomial(1, 0.5, size=n).astype(float)
    # Strongly separating covariate: fitted scores must leave the trim window.
    X = np.column_stack([20.0 * A - 10.0, rng.normal(size=n)])
    Y = (rng.random(n) < 0.3).astype(float)

    diag = positivity_diagnostics(X, A, random_state=0)
    assert diag["trim_proportion"] > 0.0  # non-raising report sees the problem

    with pytest.raises(DoublyRobustError) as excinfo:
        aiptw_ate(X, A, Y, random_state=0)
    assert "positivity" in str(excinfo.value).lower()
    assert excinfo.value.trim_proportion > 0.0


def test_tight_trim_window_fail_closed() -> None:
    # Randomised data carries no X -> A signal, so a tight window needs a
    # confounded cohort where the fitted propensity genuinely spreads.
    rng = np.random.default_rng(20260902)
    n = 1500
    X = rng.normal(size=(n, 2))
    A = (rng.random(n) < _sigmoid(1.0 * X[:, 0])).astype(float)
    Y = (rng.random(n) < 0.3).astype(float)
    with pytest.raises(DoublyRobustError) as excinfo:
        aiptw_ate(X, A, Y, random_state=0, ps_trim=(0.4, 0.6))
    assert excinfo.value.trim_proportion > 0.0


def test_invalid_inputs_fail_closed() -> None:
    X, A, Y = _randomised_cohort(SEEDS[2])
    with pytest.raises(DoublyRobustError):  # non-binary treatment
        aiptw_ate(X, np.where(A == 1.0, 2.0, 0.0), Y)
    with pytest.raises(DoublyRobustError):  # NaN covariate
        bad = X.copy()
        bad[0, 0] = float("nan")
        aiptw_ate(bad, A, Y)
    with pytest.raises(DoublyRobustError):  # single arm
        aiptw_ate(X, np.zeros_like(A), Y)
    with pytest.raises(DoublyRobustError):  # length mismatch
        aiptw_ate(X, A, Y[:-1])
    with pytest.raises(DoublyRobustError):  # unknown outcome model
        aiptw_ate(X, A, Y, outcome_model="forest")  # type: ignore[arg-type]
    with pytest.raises(DoublyRobustError):  # degenerate trim window
        aiptw_ate(X, A, Y, ps_trim=(0.9, 0.1))
    with pytest.raises(DoublyRobustError):  # single-class arm, logistic path
        aiptw_ate(X, A, np.zeros_like(Y))
