"""Split-conformal prediction: the distribution-free coverage guarantee.

These lock the property that makes conformal prediction worth reporting: at a
target error rate alpha, empirical coverage on a held-out split is at least
1 - alpha, and the Mondrian variant delivers that guarantee within each class
even under strong outcome imbalance.
"""

from __future__ import annotations

import numpy as np

from easyicu.research_agent.methods.conformal import (
    conformal_calibrate,
    conformal_evaluate,
    conformal_predict_sets,
)


def _well_calibrated_split(n, prevalence, rng):
    """Draw (prob_pos, label) with labels ~ Bernoulli(prob_pos)."""
    true_p = rng.beta(2, 2, size=n) * 0.6 + prevalence * 0.2
    labels = (rng.random(n) < true_p).astype(int)
    # Model probabilities = true probabilities (well calibrated) + mild noise.
    probs = np.clip(true_p + rng.normal(0, 0.03, size=n), 0.01, 0.99)
    return probs, labels


def test_marginal_coverage_meets_target():
    rng = np.random.default_rng(0)
    cal_p, cal_y = _well_calibrated_split(4000, 0.3, rng)
    te_p, te_y = _well_calibrated_split(4000, 0.3, rng)
    res = conformal_evaluate(cal_p, cal_y, te_p, te_y, alpha=0.1, mondrian=False)
    # 1 - alpha = 0.90 target; allow finite-sample slack downward.
    assert res.coverage >= 0.88
    assert 1.0 <= res.mean_set_size <= 2.0


def test_mondrian_covers_minority_class_under_imbalance():
    rng = np.random.default_rng(1)
    # Heavy imbalance: ~8% positives, where marginal coverage can hide minority
    # under-coverage.
    cal_p, cal_y = _well_calibrated_split(6000, 0.08, rng)
    te_p, te_y = _well_calibrated_split(6000, 0.08, rng)
    res = conformal_evaluate(cal_p, cal_y, te_p, te_y, alpha=0.1, mondrian=True)
    # Class-conditional guarantee: BOTH classes covered near/above target.
    assert res.per_class_coverage[1] >= 0.85
    assert res.per_class_coverage[0] >= 0.85


def test_smaller_alpha_gives_larger_sets():
    rng = np.random.default_rng(2)
    cal_p, cal_y = _well_calibrated_split(4000, 0.3, rng)
    te_p, te_y = _well_calibrated_split(4000, 0.3, rng)
    strict = conformal_evaluate(cal_p, cal_y, te_p, te_y, alpha=0.02, mondrian=False)
    loose = conformal_evaluate(cal_p, cal_y, te_p, te_y, alpha=0.2, mondrian=False)
    # Tighter error rate -> higher coverage -> larger (more conservative) sets.
    assert strict.coverage >= loose.coverage
    assert strict.mean_set_size >= loose.mean_set_size


def test_empty_calibration_is_safe_not_spurious():
    thresholds = conformal_calibrate([], [], alpha=0.1)
    # No calibration data -> include every class (trivially valid), never a
    # spuriously tight threshold that would drop the true label.
    sets = conformal_predict_sets([0.1, 0.9], thresholds)
    assert sets == [{0, 1}, {0, 1}]
