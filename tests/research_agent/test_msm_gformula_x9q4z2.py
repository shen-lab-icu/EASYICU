"""MSM stabilised IPTW + point g-formula: synthetic recovery and fail-closed gates.

MSM fixture: 3-visit sequential randomisation. ``A[t] ~ Bernoulli(0.5)``
independent of history, so the true stabilised weights are exactly 1 and the
fitted weights must concentrate near 1 (mean tolerance + max bound).

g-formula fixture: linear DGP ``Y = 1 + 2*A + 1.5*X0 - 0.7*X1 + N(0, 1)`` with
``A | X ~ Bernoulli(sigmoid(0.4*X0))``; the true point-intervention ATE is 2.0.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from easyicu.research_agent.methods.gformula import GFormulaError, gformula_ate
from easyicu.research_agent.methods.msm import MSMError, stabilized_iptw


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# MSM: sequential-randomisation fixture.
# ---------------------------------------------------------------------------


def _seq_randomized(
    seed: int, n: int = 2000, n_visits: int = 3
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 2))
    a_prev = np.zeros(n)
    l_prev = rng.normal(size=n)
    treatments: list[np.ndarray] = []
    den_list: list[np.ndarray] = []
    num_list: list[np.ndarray] = []
    for _ in range(n_visits):
        level = (
            0.5 * base[:, 0] + 0.5 * a_prev + 0.3 * l_prev + rng.normal(size=n)
        )
        treat = rng.binomial(1, 0.5, size=n).astype(float)
        treatments.append(treat)
        den_list.append(np.column_stack([base, level, a_prev, l_prev]))
        num_list.append(base.copy())  # baseline-only stabiliser
        a_prev, l_prev = treat, level
    return treatments, den_list, num_list


def test_msm_weights_concentrate_near_one_under_randomisation() -> None:
    treatments, den_list, num_list = _seq_randomized(20260917)
    res = stabilized_iptw(treatments, den_list, num_list, random_state=0, require_sequential_exchangeability=True, require_correct_history_encoding=True)
    assert res.n == 2000
    assert res.n_visits == 3
    assert res.seeds_used == (0, 1, 2)  # hardcoded seed + visit_index rule
    assert res.trim_proportion == 0.0
    assert res.weight_mean == pytest.approx(1.0, abs=0.05)
    assert res.weight_max < 5.0
    assert res.weight_min > 0.0
    assert res.weight_ess > 0.5 * res.n
    for visit in res.per_visit:
        assert visit.cumul_mean == pytest.approx(1.0, abs=0.08)
        assert visit.cumul_ess > 0.5 * res.n
    assert res.truncated is False
    assert res.evidence_ceiling == "analysis_only"
    assert len(res.limitations) >= 3
    payload = res.to_json()
    assert payload["seeds_used"] == [0, 1, 2]
    assert payload["evidence_ceiling"] == "analysis_only"


def test_msm_deterministic_two_runs_byte_identical() -> None:
    treatments, den_list, num_list = _seq_randomized(20260917)
    first = stabilized_iptw(treatments, den_list, num_list, random_state=0, require_sequential_exchangeability=True, require_correct_history_encoding=True)
    second = stabilized_iptw(treatments, den_list, num_list, random_state=0, require_sequential_exchangeability=True, require_correct_history_encoding=True)
    assert np.array_equal(first.weights, second.weights)
    assert (
        json.dumps(first.to_json(), sort_keys=True)
        == json.dumps(second.to_json(), sort_keys=True)
    )


def test_msm_marginal_numerator_and_truncation() -> None:
    treatments, den_list, _ = _seq_randomized(20260918)
    res = stabilized_iptw(
        treatments,
        den_list,
        [None, None, None],
        random_state=7,
        trunc_quantiles=(0.01, 0.99),
        require_sequential_exchangeability=True, require_correct_history_encoding=True,
    )
    assert res.seeds_used == (7, 8, 9)
    assert res.truncated is True
    assert res.trunc_quantiles == (0.01, 0.99)
    assert res.trunc_low is not None and res.trunc_high is not None
    assert res.trunc_low <= res.trunc_high
    assert bool((res.weights >= res.trunc_low - 1e-12).all())
    assert bool((res.weights <= res.trunc_high + 1e-12).all())
    assert res.weight_mean == pytest.approx(1.0, abs=0.08)


def test_msm_fail_closed() -> None:
    treatments, den_list, num_list = _seq_randomized(20260919)
    with pytest.raises(MSMError):  # non-binary treatment
        stabilized_iptw(
            [np.where(treatments[0] == 1.0, 2.0, 0.0)] + treatments[1:],
            den_list,
            num_list,
        )
    with pytest.raises(MSMError):  # NaN denominator feature
        bad = [frame.copy() for frame in den_list]
        bad[1][0, 0] = float("nan")
        stabilized_iptw(treatments, bad, num_list)
    with pytest.raises(MSMError):  # visit-count mismatch
        stabilized_iptw(treatments, den_list[:-1], num_list)
    with pytest.raises(MSMError):  # row-count mismatch
        stabilized_iptw(treatments, den_list, [frame[:-1] for frame in num_list])
    with pytest.raises(MSMError):  # single-arm visit
        stabilized_iptw(
            [np.zeros_like(treatments[0])] + treatments[1:], den_list, num_list
        )
    with pytest.raises(MSMError):  # degenerate trim window
        stabilized_iptw(treatments, den_list, num_list, ps_trim=(0.9, 0.1))
    with pytest.raises(MSMError):  # degenerate truncation quantiles
        stabilized_iptw(
            treatments, den_list, num_list, trunc_quantiles=(0.9, 0.1)
        )


def test_msm_positivity_violation_fail_closed() -> None:
    rng = np.random.default_rng(11)
    n = 400
    treat = rng.binomial(1, 0.5, size=n).astype(float)
    # Perfectly separating denominator: fitted scores must leave the window.
    den = np.column_stack([20.0 * treat - 10.0, rng.normal(size=n)])
    num = np.column_stack([rng.normal(size=n)])
    with pytest.raises(MSMError, match="ositivity") as excinfo:
        stabilized_iptw(
            [treat],
            [den],
            [num],
            random_state=0,
            require_sequential_exchangeability=True,
            require_correct_history_encoding=True,
        )
    assert excinfo.value.trim_proportion > 0.0


# ---------------------------------------------------------------------------
# g-formula: linear-DGP truth recovery.
# ---------------------------------------------------------------------------


def _linear_dgp(seed: int, n: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    A = (rng.random(n) < _sigmoid(0.4 * X[:, 0])).astype(float)
    Y = 1.0 + 2.0 * A + 1.5 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(size=n)
    return X, A, Y


TRUE_LINEAR_ATE = 2.0
FLAGS = {"require_exchangeability": True, "require_correct_specification": True}


def test_gformula_recovers_linear_truth() -> None:
    X, A, Y = _linear_dgp(20260917)
    res = gformula_ate(
        X, A, Y, outcome_model="linear", n_bootstrap=200, random_state=0, **FLAGS  # type: ignore[arg-type]
    )
    assert res.ate == pytest.approx(TRUE_LINEAR_ATE, abs=0.2)
    assert res.ci_low <= TRUE_LINEAR_ATE <= res.ci_high
    assert res.se > 0
    assert res.ate == pytest.approx(res.mu1_mean - res.mu0_mean)
    assert res.mu1_mean > res.mu0_mean
    assert res.n_successful >= 100
    assert res.trim_proportion == 0.0
    assert res.evidence_ceiling == "analysis_only"
    assert res.to_json()["require_exchangeability"] is True
    assert res.to_json()["require_correct_specification"] is True


def test_gformula_deterministic_two_runs_byte_identical() -> None:
    X, A, Y = _linear_dgp(20260917)
    kw = {"outcome_model": "linear", "n_bootstrap": 200, "random_state": 0, **FLAGS}  # type: ignore[dict-item]
    first = gformula_ate(X, A, Y, **kw)  # type: ignore[arg-type]
    second = gformula_ate(X, A, Y, **kw)  # type: ignore[arg-type]
    assert (
        json.dumps(first.to_json(), sort_keys=True)
        == json.dumps(second.to_json(), sort_keys=True)
    )


def test_gformula_binary_path_finite_and_deterministic() -> None:
    rng = np.random.default_rng(20260920)
    n = 1500
    X = rng.normal(size=(n, 2))
    A = rng.binomial(1, 0.5, size=n).astype(float)
    logit = -0.5 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.7 * A
    Y = (rng.random(n) < _sigmoid(logit)).astype(float)
    kw = {"outcome_model": "logistic", "n_bootstrap": 100, "random_state": 0, **FLAGS}  # type: ignore[dict-item]
    first = gformula_ate(X, A, Y, **kw)  # type: ignore[arg-type]
    second = gformula_ate(X, A, Y, **kw)  # type: ignore[arg-type]
    assert np.isfinite(first.ate)
    assert first.se > 0
    assert first.ci_low <= first.ate <= first.ci_high
    assert (
        json.dumps(first.to_json(), sort_keys=True)
        == json.dumps(second.to_json(), sort_keys=True)
    )


def test_gformula_fail_closed_without_assumption_flags() -> None:
    X, A, Y = _linear_dgp(20260921)
    with pytest.raises(GFormulaError):
        gformula_ate(X, A, Y, outcome_model="linear", n_bootstrap=10)
    with pytest.raises(GFormulaError):
        gformula_ate(
            X, A, Y, outcome_model="linear", n_bootstrap=10,
            require_exchangeability=True, require_correct_specification=False,
        )
    with pytest.raises(GFormulaError):
        gformula_ate(
            X, A, Y, outcome_model="linear", n_bootstrap=10,
            require_exchangeability=False, require_correct_specification=True,
        )


def test_gformula_fail_closed_illegal_inputs() -> None:
    X, A, Y = _linear_dgp(20260922)
    with pytest.raises(GFormulaError):  # non-binary treatment
        gformula_ate(X, np.where(A == 1.0, 2.0, 0.0), Y, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(GFormulaError):  # NaN outcome
        bad = Y.copy()
        bad[0] = float("nan")
        gformula_ate(X, A, bad, outcome_model="linear", **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(GFormulaError):  # NaN covariate
        bad_x = X.copy()
        bad_x[0, 0] = float("nan")
        gformula_ate(bad_x, A, Y, outcome_model="linear", **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(GFormulaError):  # single arm
        gformula_ate(X, np.zeros_like(A), Y, outcome_model="linear", **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(GFormulaError):  # length mismatch
        gformula_ate(X, A, Y[:-1], outcome_model="linear", **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(GFormulaError):  # continuous outcome under logistic
        gformula_ate(X, A, Y, outcome_model="logistic", **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(GFormulaError):  # unknown outcome model
        gformula_ate(X, A, Y, outcome_model="forest", **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(GFormulaError):  # empty bootstrap
        gformula_ate(X, A, Y, outcome_model="linear", n_bootstrap=0, **FLAGS)  # type: ignore[arg-type]


def test_gformula_positivity_violation_fail_closed() -> None:
    rng = np.random.default_rng(13)
    n = 400
    A = rng.binomial(1, 0.5, size=n).astype(float)
    # Strongly (but not exactly) separating covariate: fitted scores leave
    # the trim window while [A, X] stays full rank.
    X = np.column_stack([3.0 * A + 0.5 * rng.normal(size=n), rng.normal(size=n)])
    Y = rng.normal(size=n)
    with pytest.raises(GFormulaError, match="ositivity") as excinfo:
        gformula_ate(X, A, Y, outcome_model="linear", n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    assert excinfo.value.trim_proportion > 0.0


def test_msm_refuses_undeclared_assumptions() -> None:
    treatments, den_list, num_list = _seq_randomized(20260920)
    with pytest.raises(MSMError, match="sequential exchangeability"):
        stabilized_iptw(treatments, den_list, num_list)
    with pytest.raises(MSMError, match="history encoding"):
        stabilized_iptw(
            treatments,
            den_list,
            num_list,
            require_sequential_exchangeability=True,
        )
