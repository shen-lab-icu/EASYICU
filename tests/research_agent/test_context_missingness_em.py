"""Regression tests for the Little-MCAR EM missing-pattern optimisation."""

from __future__ import annotations

import numpy as np

from easyicu.research_agent.context import _estimate_mvn_with_em


def _uncached_reference(
    data: np.ndarray, *, max_iter: int = 100, tol: float = 1e-6
):
    """Small copy of the original row-wise EM for equivalence testing."""

    x = np.asarray(data, dtype=float)
    n, p = x.shape
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    filled = np.where(np.isnan(x), mu, x)
    cov = np.asarray(np.cov(filled, rowvar=False), dtype=float)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov += np.eye(p) * 1e-6

    for _ in range(max_iter):
        expected_rows = []
        second_moments = []
        for row in x:
            obs = np.flatnonzero(~np.isnan(row))
            mis = np.flatnonzero(np.isnan(row))
            if len(mis) == 0:
                row_expectation = row.astype(float)
                second = np.outer(row_expectation, row_expectation)
            elif len(obs) == 0:
                row_expectation = mu.copy()
                second = cov + np.outer(mu, mu)
            else:
                sigma_oo = cov[np.ix_(obs, obs)] + np.eye(len(obs)) * 1e-8
                sigma_mo = cov[np.ix_(mis, obs)]
                sigma_om = cov[np.ix_(obs, mis)]
                sigma_mm = cov[np.ix_(mis, mis)]
                inv_oo = np.linalg.pinv(sigma_oo)
                row_expectation = row.copy().astype(float)
                cond_mean = mu[mis] + sigma_mo @ inv_oo @ (row[obs] - mu[obs])
                cond_cov = sigma_mm - sigma_mo @ inv_oo @ sigma_om
                row_expectation[mis] = cond_mean
                second = np.outer(row_expectation, row_expectation)
                second[np.ix_(mis, mis)] += cond_cov
            expected_rows.append(row_expectation)
            second_moments.append(second)

        expected = np.vstack(expected_rows)
        mu_new = expected.mean(axis=0)
        cov_new = np.mean(second_moments, axis=0) - np.outer(mu_new, mu_new)
        cov_new = (cov_new + cov_new.T) / 2.0
        cov_new += np.eye(p) * 1e-6
        if np.max(np.abs(mu_new - mu)) < tol and np.max(
            np.abs(cov_new - cov)
        ) < tol:
            mu, cov = mu_new, cov_new
            break
        mu, cov = mu_new, cov_new
    return mu, cov


def test_em_pattern_cache_is_numerically_equivalent_to_rowwise_reference() -> None:
    data = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [2.0, 3.0, np.nan, 5.0],
            [3.0, np.nan, 5.0, 6.0],
            [4.0, np.nan, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    expected_mu, expected_cov = _uncached_reference(data, max_iter=8)
    actual_mu, actual_cov = _estimate_mvn_with_em(data, max_iter=8)

    # Grouped matrix multiplication changes floating-point summation order but
    # not the EM equations; tolerate only sub-nanoscopic roundoff.
    np.testing.assert_allclose(actual_mu, expected_mu, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=1e-9, atol=1e-9)


def test_em_computes_one_pseudoinverse_per_missing_pattern(monkeypatch) -> None:
    complete = np.arange(1.0, 25.0).reshape(6, 4)
    repeated_a = np.tile([10.0, 20.0, np.nan, 40.0], (80, 1))
    repeated_b = np.tile([10.0, np.nan, 30.0, 40.0], (60, 1))
    data = np.vstack([complete, repeated_a, repeated_b])

    original_pinv = np.linalg.pinv
    calls = 0

    def counting_pinv(value):
        nonlocal calls
        calls += 1
        return original_pinv(value)

    monkeypatch.setattr(np.linalg, "pinv", counting_pinv)
    _estimate_mvn_with_em(data, max_iter=1)

    assert calls == 2
