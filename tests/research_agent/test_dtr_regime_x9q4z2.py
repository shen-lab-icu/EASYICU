"""Given-regime valuation: known-optimal synthetic regime beats treat-all.

Fixture: ``X0, X1 ~ N(0, 1)``, observed ``A ~ Bernoulli(0.5)`` randomised, and
``Y = X0 + 0.5*X1 + 2.0 * 1{A == 1{X0 > 0}} + N(0, 1)``. The analytic regime
values are ``V(optimal rule 1{X0>0}) = 2.0`` and ``V(treat-all) = 1.0``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from easyicu.research_agent.methods.target_trial import (
    TargetTrialError,
    evaluate_dynamic_regime,
)


def _optimal_regime_dgp(
    seed: int, n: int = 3000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    X = np.column_stack([x0, x1])
    A = rng.binomial(1, 0.5, size=n).astype(float)
    rec_opt = (x0 > 0).astype(float)
    Y = x0 + 0.5 * x1 + 2.0 * (A == rec_opt).astype(float) + rng.normal(size=n)
    return X, A, Y, rec_opt


def test_optimal_regime_beats_treat_all_and_matches_analytic_value() -> None:
    X, A, Y, rec_opt = _optimal_regime_dgp(20260917)
    v_opt = evaluate_dynamic_regime(X, A, Y, rec_opt, random_state=0)
    v_all = evaluate_dynamic_regime(X, A, Y, np.ones_like(A), random_state=0)
    assert v_opt.value > v_all.value
    assert v_opt.value == pytest.approx(2.0, abs=0.25)
    assert v_all.value == pytest.approx(1.0, abs=0.25)
    for res in (v_opt, v_all):
        assert res.se > 0
        assert res.ci_low < res.value < res.ci_high
        assert res.adherence_proportion == pytest.approx(0.5, abs=0.05)
        assert res.n_adherent + (res.n - res.n_adherent) == res.n == 3000
        assert res.trim_proportion == 0.0
        assert res.ess > 0
        assert res.evidence_ceiling == "analysis_only"
    assert any("多阶段优化" in item for item in v_opt.limitations)


def test_regime_valuation_deterministic_two_runs_byte_identical() -> None:
    X, A, Y, rec_opt = _optimal_regime_dgp(20260917)
    first = evaluate_dynamic_regime(X, A, Y, rec_opt, random_state=0)
    second = evaluate_dynamic_regime(X, A, Y, rec_opt, random_state=0)
    assert first.value == second.value
    assert first.se == second.se
    assert (
        json.dumps(first.to_json(), sort_keys=True)
        == json.dumps(second.to_json(), sort_keys=True)
    )


def test_regime_fail_closed_illegal_inputs() -> None:
    X, A, Y, rec_opt = _optimal_regime_dgp(20260918)
    with pytest.raises(TargetTrialError):  # non-binary recommendation
        evaluate_dynamic_regime(
            X, A, Y, np.where(rec_opt == 1.0, 2.0, 0.0), random_state=0
        )
    with pytest.raises(TargetTrialError):  # non-binary observed treatment
        evaluate_dynamic_regime(
            X, np.where(A == 1.0, 2.0, 0.0), Y, rec_opt, random_state=0
        )
    with pytest.raises(TargetTrialError):  # NaN covariate
        bad = X.copy()
        bad[0, 0] = float("nan")
        evaluate_dynamic_regime(bad, A, Y, rec_opt, random_state=0)
    with pytest.raises(TargetTrialError):  # NaN outcome
        bad_y = Y.copy()
        bad_y[0] = float("nan")
        evaluate_dynamic_regime(X, A, bad_y, rec_opt, random_state=0)
    with pytest.raises(TargetTrialError):  # NaN recommendation
        bad_rec = rec_opt.copy()
        bad_rec[0] = float("nan")
        evaluate_dynamic_regime(X, A, Y, bad_rec, random_state=0)
    with pytest.raises(TargetTrialError):  # length mismatch
        evaluate_dynamic_regime(X, A, Y[:-1], rec_opt[:-1], random_state=0)
    with pytest.raises(TargetTrialError):  # single-arm observed treatment
        evaluate_dynamic_regime(X, np.zeros_like(A), Y, rec_opt, random_state=0)
    with pytest.raises(TargetTrialError):  # degenerate trim window
        evaluate_dynamic_regime(X, A, Y, rec_opt, ps_trim=(0.9, 0.1))


def test_regime_low_adherence_fail_closed() -> None:
    X, A, Y, _ = _optimal_regime_dgp(20260919)
    with pytest.raises(TargetTrialError, match="adherence"):
        # Recommending the opposite of what everyone received: adherence 0.
        evaluate_dynamic_regime(X, A, Y, 1.0 - A, random_state=0)


def test_regime_positivity_violation_fail_closed() -> None:
    rng = np.random.default_rng(17)
    n = 400
    A = rng.binomial(1, 0.5, size=n).astype(float)
    X = np.column_stack([20.0 * A - 10.0, rng.normal(size=n)])
    Y = rng.normal(size=n)
    rec = rng.binomial(1, 0.5, size=n).astype(float)
    with pytest.raises(TargetTrialError, match="ositivity"):
        evaluate_dynamic_regime(X, A, Y, rec, random_state=0)
