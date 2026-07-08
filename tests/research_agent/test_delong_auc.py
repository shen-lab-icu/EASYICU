"""DeLong AUROC variance / CI / correlated comparison.

These lock the closed-form DeLong inference against hand-verified ground truth:
the point AUC must match sklearn to machine precision, the logit CI must bracket
the estimate strictly inside [0, 1], perfect separation must collapse to
AUC = 1, comparing a model to itself must show no difference, and the DeLong SE
must land inside a bootstrap sanity band. Discrimination inference only --
calibration is a separate question.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from easyicu.research_agent.delong_auc import (
    AUCResult,
    delong_auc_ci,
    delong_auc_variance,
    delong_test,
)


def _fixed_dataset():
    """rng(0), n=400, two Gaussians separated so AUROC ~ 0.8."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    score = rng.normal(0, 1, size=n) + y * 1.2
    return y, score


# ---------------------------------------------------------------------------
# Reference-value assertions (hand-verified ground truth)
# ---------------------------------------------------------------------------


def test_point_auc_matches_sklearn():
    y, score = _fixed_dataset()
    auc, _ = delong_auc_variance(y, score)
    sk = roc_auc_score(y, score)
    assert abs(auc - sk) < 1e-9
    # The dataset was tuned for AUROC ~ 0.8.
    assert 0.75 < auc < 0.9


def test_ci_brackets_auc_inside_unit_interval():
    y, score = _fixed_dataset()
    res = delong_auc_ci(y, score, alpha=0.05)
    assert isinstance(res, AUCResult)
    assert res.ci_low < res.auc < res.ci_high
    assert 0.0 <= res.ci_low <= 1.0
    assert 0.0 <= res.ci_high <= 1.0
    assert res.se > 0.0


def test_perfect_separation_gives_auc_one():
    # All positives scored strictly above all negatives.
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc, var = delong_auc_variance(y, score)
    assert auc == 1.0
    assert var == 0.0
    # A degenerate boundary AUC collapses the CI to the point estimate.
    res = delong_auc_ci(y, score)
    assert res.ci_low == 1.0 and res.ci_high == 1.0


def test_identical_scores_show_no_difference():
    y, score = _fixed_dataset()
    auc_a, auc_b, z, p = delong_test(y, score, score)
    assert auc_a == auc_b
    assert abs(z) < 1e-9
    # A model versus itself: no difference, so p ~ 1.
    assert p > 0.99


def test_delong_se_within_bootstrap_band():
    y, score = _fixed_dataset()
    _, var = delong_auc_variance(y, score)
    delong_se = float(np.sqrt(var))

    # Seeded 2000-resample bootstrap SE of the AUC as an independent sanity band.
    bs = np.random.default_rng(123)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(2000):
        s = bs.choice(idx, size=len(y), replace=True)
        ys = y[s]
        if ys.min() == ys.max():
            continue  # degenerate resample with a single class
        aucs.append(roc_auc_score(ys, score[s]))
    boot_se = float(np.std(aucs, ddof=1))

    rel = abs(delong_se - boot_se) / boot_se
    assert rel < 0.25, f"DeLong SE {delong_se:.4f} vs bootstrap {boot_se:.4f}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_ties_handled_via_midranks():
    # Tied scores across classes must use midranks, matching sklearn's AUC.
    y = np.array([1, 1, 0, 0, 1, 0])
    score = np.array([0.9, 0.9, 0.9, 0.1, 0.5, 0.5])
    auc, _ = delong_auc_variance(y, score)
    assert abs(auc - roc_auc_score(y, score)) < 1e-9


def test_better_model_beats_worse_model():
    # A strongly separating score vs a weakly separating one on the SAME labels
    # must be detected as different (correlated DeLong comparison).
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    strong = rng.normal(0, 1, size=n) + y * 1.2
    weak = rng.normal(0, 1, size=n) + y * 0.4
    auc_a, auc_b, z, p = delong_test(y, strong, weak)
    assert auc_a > auc_b
    assert z > 0
    assert p < 0.01


def test_missing_class_raises():
    # An AUROC is undefined without both classes present.
    y = np.array([1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4])
    try:
        delong_auc_variance(y, score)
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected ValueError for single-class input")
