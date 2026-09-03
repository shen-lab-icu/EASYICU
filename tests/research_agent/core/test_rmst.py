"""Restricted mean survival time: the PH-free area-under-KM effect measure.

These lock the estimator against hand-verified ground truth (the KM staircase
area with no censoring reduces to ``mean(min(T, tau))``) and the structural
invariants that make RMST safe to report: it is bounded by the horizon, a
horizon before the first event returns the horizon itself, and the two-group
difference recovers the exact contrast when there is no censoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.rmst import RMSTResult, rmst, rmst_difference


# ---------------------------------------------------------------------------
# Hand-verified reference values (exact KM staircase area).
# ---------------------------------------------------------------------------


def test_no_censoring_full_horizon_reference():
    # KM staircase 1 -> .8 -> .6 -> .4 -> .2 -> 0 at the death times; the area
    # on [0, 5] is 1 + 0.8 + 0.6 + 0.4 + 0.2 = 3.0.
    res = rmst([1, 2, 3, 4, 5], [True] * 5, tau=5)
    assert isinstance(res, RMSTResult)
    assert res.rmst == pytest.approx(3.0, abs=1e-9)
    # No censoring -> RMST == mean(min(T, tau)).
    assert res.rmst == pytest.approx(np.mean(np.minimum([1, 2, 3, 4, 5], 5)), abs=1e-9)
    assert res.tau == 5.0


def test_no_censoring_short_horizon_reference():
    # Area on [0, 3] = 1 + 0.8 + 0.6 = 2.4.
    res = rmst([1, 2, 3, 4, 5], [True] * 5, tau=3)
    assert res.rmst == pytest.approx(2.4, abs=1e-9)


def test_tau_before_first_event_returns_tau():
    # S(t) == 1 on [0, tau] when tau is below the first event time, so the area
    # is exactly the horizon.
    res = rmst([2, 3, 4], [True, True, True], tau=1.0)
    assert res.rmst == pytest.approx(1.0, abs=1e-9)
    # With the curve flat at 1 there is no dropped mass -> zero variance.
    assert res.se == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Structural invariants.
# ---------------------------------------------------------------------------


def test_rmst_never_exceeds_tau():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(3, 40))
        durations = rng.exponential(5.0, size=n)
        events = rng.integers(0, 2, size=n)
        tau = float(rng.uniform(0.1, 15.0))
        res = rmst(durations, events, tau=tau)
        assert -1e-12 <= res.rmst <= tau + 1e-9
        assert res.se >= 0.0
        assert res.ci_low <= res.rmst <= res.ci_high


def test_no_censoring_se_matches_integral_form():
    # Hand-verified integral-form variance for durations [1..5], tau=5, no
    # censoring. A_i is the curve area to the right of each event time:
    #   t=1: A=2.0, denom n(n-d)=5*4=20 -> 4.0/20 = 0.20
    #   t=2: A=1.2, denom 4*3=12        -> 1.44/12 = 0.12
    #   t=3: A=0.6, denom 3*2=6         -> 0.36/6  = 0.06
    #   t=4: A=0.2, denom 2*1=2         -> 0.04/2  = 0.02
    #   t=5: n==d -> denom 0, term dropped
    # variance = 0.40, SE = sqrt(0.40).
    res = rmst([1, 2, 3, 4, 5], [True] * 5, tau=5)
    assert res.se == pytest.approx(np.sqrt(0.40), abs=1e-9)
    # Wald CI clipped into [0, tau].
    z = 1.959963984540054
    assert res.ci_low == pytest.approx(res.rmst - z * res.se, abs=1e-9)
    assert res.ci_high == pytest.approx(res.rmst + z * res.se, abs=1e-9)


def test_invalid_tau_and_length_mismatch_raise():
    with pytest.raises(ValueError):
        rmst([1, 2, 3], [True, True, True], tau=0)
    with pytest.raises(ValueError):
        rmst([1, 2, 3], [True, True], tau=2)


# ---------------------------------------------------------------------------
# Two-group difference.
# ---------------------------------------------------------------------------


def test_rmst_difference_exact_contrast_no_censoring():
    # Group a events at [1,2,3,4,5] -> RMST(5) = 3.0.
    # Group b events at [2,4,6,8,10] -> RMST(5) area = 1 + 1 + 0.8 + 0.6 + 0.6
    #   (S = 1 on [0,2), 0.8 on [2,4), 0.6 on [4,5]) = 4.2.
    groups = ["a"] * 5 + ["b"] * 5
    durations = [1, 2, 3, 4, 5, 2, 4, 6, 8, 10]
    events = [True] * 10
    out = rmst_difference(durations, events, groups, tau=5)
    assert out["group_a"] == "a"
    assert out["group_b"] == "b"
    assert out["rmst_a"] == pytest.approx(3.0, abs=1e-9)
    assert out["rmst_b"] == pytest.approx(4.2, abs=1e-9)
    assert out["diff"] == pytest.approx(-1.2, abs=1e-9)
    low, high = out["ci"]
    assert low <= out["diff"] <= high
    assert 0.0 <= out["p_value"] <= 1.0


def test_rmst_difference_requires_two_groups():
    with pytest.raises(ValueError):
        rmst_difference([1, 2, 3], [True, True, True], ["a", "a", "a"], tau=2)
    with pytest.raises(ValueError):
        rmst_difference(
            [1, 2, 3], [True, True, True], ["a", "b", "c"], tau=2
        )
