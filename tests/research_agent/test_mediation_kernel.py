"""Kernel tests: product-method mediation (linear + binary outcomes).

Analytic fixture (noiseless linear chain, exact recovery)::

    X = [0, 0, 1, 1, 2, 2]
    M = 1 + 2*X + [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]   (wiggle breaks collinearity)
    Y = 0.5 + 1.5*X + 3*M

so a = 2, b = 3, c' = 1.5, NIE = 6, NDE = 1.5, total = 7.5, prop = 0.8.
"""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.mediation import mediate

X = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
WIGGLE = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
M = 1.0 + 2.0 * X + WIGGLE
Y = 0.5 + 1.5 * X + 3.0 * M

FLAGS = {"assume_no_interaction": True, "assume_sequential_ignorability": True}


def test_linear_chain_matches_product() -> None:
    res = mediate(X, M, Y, n_bootstrap=50, random_state=0, **FLAGS)  # type: ignore[arg-type]
    assert res.coef_a == pytest.approx(2.0)
    assert res.coef_b == pytest.approx(3.0)
    assert res.coef_c_prime == pytest.approx(1.5)
    assert res.nie == pytest.approx(6.0)
    assert res.nde == pytest.approx(1.5)
    assert res.total_effect == pytest.approx(7.5)
    assert res.nie == pytest.approx(res.coef_a * res.coef_b)
    assert res.total_effect == pytest.approx(res.nde + res.nie)
    assert res.prop_mediated == pytest.approx(0.8)
    assert res.interaction_p_value >= 0.05  # no interaction by construction
    assert res.outcome_type == "linear"
    assert res.claim_ceiling == "analysis_only"


def test_linear_chain_with_covariate_exact() -> None:
    c = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    # Wiggle orthogonalised against [1, X, C] so the structural coefficients
    # (a=2 on X, 0.5 on C) stay exactly identified by OLS.
    design = np.column_stack([np.ones(6), X, c])
    base = np.array([0.3, -0.4, 0.5, -0.2, 0.1, -0.6])
    wiggle = base - design @ np.linalg.lstsq(design, base, rcond=None)[0]
    m2 = 1.0 + 2.0 * X + 0.5 * c + wiggle
    y2 = 0.5 + 1.5 * X + 3.0 * m2 - 1.0 * c
    res = mediate(X, m2, y2, covariates=c, n_bootstrap=50, random_state=0, **FLAGS)  # type: ignore[arg-type]
    assert res.coef_a == pytest.approx(2.0)
    assert res.coef_b == pytest.approx(3.0)
    assert res.coef_c_prime == pytest.approx(1.5)
    assert res.nie == pytest.approx(6.0)
    assert res.n_covariates == 1


def _binary_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 400
    x = rng.binomial(1, 0.5, size=n).astype(float)
    c = rng.normal(size=n)
    m = x + 0.5 * c + rng.normal(size=n)
    logit = -1.0 + 0.6 * x + 0.7 * m + 0.3 * c
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(float)
    return x, m, y, c


def test_binary_outcome_deterministic_and_internally_consistent() -> None:
    x, m, y, c = _binary_fixture()
    first = mediate(
        x, m, y, covariates=c, outcome_type="binary",
        n_bootstrap=100, random_state=3, **FLAGS,  # type: ignore[arg-type]
    )
    second = mediate(
        x, m, y, covariates=c, outcome_type="binary",
        n_bootstrap=100, random_state=3, **FLAGS,  # type: ignore[arg-type]
    )
    assert first.to_json() == second.to_json()
    assert np.isfinite([first.nde, first.nie, first.total_effect]).all()
    assert first.nie == pytest.approx(first.coef_a * first.coef_b)
    assert first.ci_nie[0] <= first.nie <= first.ci_nie[1]
    assert first.ci_nde[0] <= first.nde <= first.ci_nde[1]
    assert first.ci_total[0] <= first.total_effect <= first.ci_total[1]
    assert first.n_successful >= 50
    assert first.coef_a == pytest.approx(1.0, abs=0.25)
    assert first.coef_b == pytest.approx(0.7, abs=0.25)
    assert first.to_json()["claim_ceiling"] == "analysis_only"


def test_linear_deterministic_two_runs_match() -> None:
    kw = {"n_bootstrap": 50, "random_state": 0, **FLAGS}  # type: ignore[dict-item]
    assert mediate(X, M, Y, **kw).to_json() == mediate(X, M, Y, **kw).to_json()  # type: ignore[arg-type]


def test_fail_closed_without_assumption_flags() -> None:
    with pytest.raises(ValueError):
        mediate(X, M, Y, n_bootstrap=10, assume_no_interaction=False,
                assume_sequential_ignorability=True)
    with pytest.raises(ValueError):
        mediate(X, M, Y, n_bootstrap=10, assume_no_interaction=True,
                assume_sequential_ignorability=False)
    with pytest.raises(ValueError):
        mediate(X, M, Y, n_bootstrap=10)  # defaults are False


def test_fail_closed_on_interaction() -> None:
    # Strong exposure-mediator interaction: Y = X + M + 5*X*M.
    x = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    m = np.array([0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0])
    y = x + m + 5.0 * x * m
    with pytest.raises(ValueError, match="interaction"):
        mediate(x, m, y, n_bootstrap=10, random_state=0, **FLAGS)  # type: ignore[arg-type]


def test_fail_closed_illegal_inputs() -> None:
    with pytest.raises(ValueError):  # length mismatch
        mediate([0.0, 1.0], [0.5], [1.0], n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # NaN mediator
        mediate(X, [np.nan] * 6, Y, n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # zero-variance exposure
        mediate(np.ones(6), M, Y, n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # zero-variance mediator
        mediate(X, np.ones(6), Y, n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # bad outcome type
        mediate(X, M, Y, outcome_type="survival", n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # single-class binary outcome
        mediate(X, M, np.zeros(6), outcome_type="binary", n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # non-binary values under binary type
        mediate(X, M, Y, outcome_type="binary", n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # misaligned covariates
        mediate(X, M, Y, covariates=np.zeros((5, 1)), n_bootstrap=10, **FLAGS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # too few rows to screen the interaction
        mediate(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.5, 1.5, 0.0, 2.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            n_bootstrap=10, **FLAGS,  # type: ignore[arg-type]
        )
