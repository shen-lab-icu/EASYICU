"""Decision-curve analysis: net benefit across threshold probabilities.

These lock the Vickers & Elkin (2006) net-benefit arithmetic against
hand-verified ground truth, plus the boundary behaviour that makes the sweep
safe: pt -> 1 must not divide by zero, and a model that never beats treat-none
must not report a spuriously positive net benefit.
"""

from __future__ import annotations

import numpy as np

from easyicu.research_agent.methods.decision_curve import (
    net_benefit_at,
    net_benefit_curve,
    summarize_decision_curve,
)

# Hand-verified fixture from the method reference.
_Y_TRUE = [1, 1, 0, 0]
_Y_PROB = [0.9, 0.6, 0.4, 0.2]


def _row_at(frame, threshold):
    """Return the single frame row whose threshold matches, within tol."""
    mask = np.isclose(frame["threshold"].to_numpy(dtype=float), threshold)
    assert mask.sum() == 1, f"expected exactly one row at pt={threshold}"
    return frame[mask].iloc[0]


def test_reference_values_pt_half():
    # pt = 0.5: predicted-positive = prob >= 0.5 = [1,1,0,0] -> TP=2, FP=0.
    frame = net_benefit_curve(_Y_TRUE, _Y_PROB)
    row = _row_at(frame, 0.5)
    assert abs(row["net_benefit_model"] - 0.5) < 1e-9
    assert abs(row["net_benefit_all"] - 0.0) < 1e-9
    assert abs(row["net_benefit_none"] - 0.0) < 1e-9


def test_reference_values_pt_quarter():
    # pt = 0.25: predicted-positive = prob >= 0.25 = [1,1,1,0] -> TP=2, FP=1.
    #   (y_prob = [0.9, 0.6, 0.4, 0.2]; the 0.4 patient (a true negative) is the FP,
    #   the 0.2 patient stays negative). model = 2/4 - (1/4)*(0.25/0.75) = 0.4166666667
    #   all   = 0.5 - 0.5*(0.25/0.75)   = 0.3333333333
    frame = net_benefit_curve(_Y_TRUE, _Y_PROB)
    row = _row_at(frame, 0.25)
    assert abs(row["net_benefit_model"] - 0.41666666666666663) < 1e-9
    assert abs(row["net_benefit_all"] - 0.3333333333333333) < 1e-9
    assert abs(row["net_benefit_none"] - 0.0) < 1e-9


def test_net_benefit_at_matches_curve():
    single = net_benefit_at(_Y_TRUE, _Y_PROB, 0.25)
    assert abs(single["net_benefit_model"] - 0.41666666666666663) < 1e-9
    assert abs(single["net_benefit_all"] - 0.3333333333333333) < 1e-9
    assert abs(single["threshold"] - 0.25) < 1e-9


def test_columns_are_stable_and_trace_friendly():
    frame = net_benefit_curve(_Y_TRUE, _Y_PROB)
    assert list(frame.columns) == [
        "threshold",
        "net_benefit_model",
        "net_benefit_all",
        "net_benefit_none",
    ]
    # treat-none is identically zero everywhere.
    assert np.allclose(frame["net_benefit_none"].to_numpy(dtype=float), 0.0)


def test_default_thresholds_stay_below_one_no_divide_by_zero():
    # Default sweep is arange(0.01, 1.00, 0.01); every pt must be < 1 so
    # pt/(1-pt) never divides by zero, and the frame must be finite.
    frame = net_benefit_curve(_Y_TRUE, _Y_PROB)
    thr = frame["threshold"].to_numpy(dtype=float)
    assert thr.max() < 1.0
    assert np.all(np.isfinite(frame.to_numpy(dtype=float)))


def test_threshold_at_one_is_dropped_not_crashed():
    # A pt at (or above) 1.0 must be skipped, never a ZeroDivisionError / inf.
    frame = net_benefit_curve(_Y_TRUE, _Y_PROB, thresholds=[0.5, 1.0, 1.5])
    thr = frame["threshold"].to_numpy(dtype=float)
    assert np.allclose(thr, [0.5])
    assert np.all(np.isfinite(frame.to_numpy(dtype=float)))


def test_all_negative_outcomes_give_nonpositive_model_net_benefit():
    # Prevalence 0 -> every predicted positive is a false positive, so the
    # model's net benefit is <= 0 across all thresholds (treat-none dominates).
    y_true = [0, 0, 0, 0]
    y_prob = [0.9, 0.6, 0.4, 0.2]
    frame = net_benefit_curve(y_true, y_prob)
    model = frame["net_benefit_model"].to_numpy(dtype=float)
    assert np.all(model <= 1e-12)


def test_summary_reports_peak_net_benefit():
    res = summarize_decision_curve(_Y_TRUE, _Y_PROB)
    assert res.n == 4
    assert abs(res.prevalence - 0.5) < 1e-9
    # Peak model net benefit over the default sweep is 0.5 (reached where all
    # positives are captured with zero false positives).
    assert res.best_net_benefit is not None
    assert abs(res.best_net_benefit - 0.5) < 1e-9
    assert res.best_threshold is not None
    assert 0.0 < res.best_threshold < 1.0


def test_non_binary_labels_rejected():
    import pytest

    with pytest.raises(ValueError):
        net_benefit_curve([0, 1, 2], [0.1, 0.5, 0.9])
