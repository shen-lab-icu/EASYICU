"""DeLong variance of the AUROC via the fast Sun & Xu (2014) algorithm.

A reported AUROC (area under the receiver-operating-characteristic curve)
summarises *discrimination* -- the probability that a randomly chosen positive
case is scored above a randomly chosen negative one. To turn a point estimate
into inference we need its sampling variance. DeLong, DeLong & Clarke-Pearson
(1988) gave a nonparametric variance based on the theory of U-statistics; Sun &
Xu (2014, IEEE Signal Processing Letters 21(11):1389-1393) reduced its cost
from O(n^2) to O(n log n) by computing the structural components from *midrank*
placement values.

Method
------
Let there be ``m`` positive and ``n`` negative examples with scores. The AUROC
equals the Mann-Whitney U statistic normalised by ``m * n``. Its two sets of
structural (placement) components are

    V10_i = P(positive_i beats a random negative)   over positives (length m)
    V01_j = P(a random positive beats negative_j)    over negatives (length n)

with ``AUC = mean(V10) = mean(V01)``. Sun & Xu compute these in O(n log n) from
midranks (average ranks over ties) of the combined, positive-only, and
negative-only score vectors::

    V10_i = (midrank_all_of_positive_i - midrank_pos_of_positive_i) / n
    V01_j = 1 - (midrank_all_of_negative_j - midrank_neg_of_negative_j) / m

The single-AUC variance is the standard fastDeLong form with the (m-1)/(n-1)
sample corrections::

    var(AUC) = S10 / m + S01 / n
    S10 = sum((V10 - AUC)^2) / (m - 1)
    S01 = sum((V01 - AUC)^2) / (n - 1)

For two correlated AUROCs measured on the *same* samples, the covariance of the
two structural-component vectors gives a normal-approximation z test of the
difference (DeLong 1988, Section 3).

Confidence interval
-------------------
:func:`delong_auc_ci` builds the interval on the logit scale and back-
transforms. A plain ``auc +- z * se`` interval can exceed 1 (or drop below 0)
near the boundary, which is nonsensical for a probability. Working in logit
space -- ``ci = expit(logit(auc) +- z * se / (auc * (1 - auc)))`` -- keeps the
interval strictly inside ``[0, 1]`` while remaining an asymptotically valid
normal approximation. The result is clipped to ``[0, 1]`` as a final guard.

Scope
-----
This module performs **discrimination inference only**. It says nothing about
whether predicted probabilities are *calibrated*; calibration is a separate
question (see :mod:`easyicu.research_agent.conformal` for a distribution-free
per-patient guarantee, and calibration-slope / Brier reporting elsewhere).

Pure ``numpy`` + ``scipy.stats.norm``; no optional dependency is imported at
module load, so the module is importable with only pandas + numpy + scipy.

References
----------
DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more
correlated receiver operating characteristic curves: a nonparametric approach.
Biometrics 1988;44(3):837-845.

Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the areas
under correlated receiver operating characteristic curves. IEEE Signal
Processing Letters 2014;21(11):1389-1393.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AUCResult:
    """A single AUROC with DeLong standard error and a logit CI."""

    auc: float
    se: float
    ci_low: float
    ci_high: float


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (average ranks over ties) of ``x``, 1-based.

    Ties receive the mean of the ranks they would occupy, which is what makes
    the placement values equal to the true Mann-Whitney U rather than an
    arbitrary tie-broken variant. O(n log n) via a single argsort.
    """

    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # positions i..j (inclusive) are tied; assign the average 1-based rank
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _structural_components(
    pos_scores: np.ndarray, neg_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(V10, V01, auc)`` via midrank placement values (Sun & Xu 2014).

    ``V10`` has one entry per positive, ``V01`` one per negative, and
    ``auc == mean(V10) == mean(V01)`` up to floating-point rounding.
    """

    m = len(pos_scores)
    n = len(neg_scores)
    if m == 0 or n == 0:
        raise ValueError("need at least one positive and one negative example")

    tx = _midrank(pos_scores)
    ty = _midrank(neg_scores)
    tz = _midrank(np.concatenate([pos_scores, neg_scores]))
    tz_pos = tz[:m]
    tz_neg = tz[m:]

    # V10_i = P(pos_i > random neg): the count of negatives it beats, / n.
    v10 = (tz_pos - tx) / n
    # V01_j = P(random pos > neg_j): 1 minus the fraction of positives it beats.
    v01 = 1.0 - (tz_neg - ty) / m

    # AUC is the mean of either placement-component vector (they are equal).
    auc = float(v10.mean())
    return v10, v01, auc


def delong_auc_variance(
    y_true: Sequence[int], y_score: Sequence[float]
) -> Tuple[float, float]:
    """AUROC and its DeLong variance via midrank placement values.

    Parameters
    ----------
    y_true:
        Binary labels; anything ``> 0`` (or truthy) is treated as the positive
        class, everything else as negative.
    y_score:
        Real-valued scores where larger means "more positive".

    Returns
    -------
    ``(auc, var)`` -- the point AUROC and its sampling variance. The variance
    uses the standard fastDeLong form ``var(V10)/m + var(V01)/n`` with the
    ``(m-1)`` / ``(n-1)`` sample corrections.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score length mismatch")
    positive = y_true > 0
    pos = y_score[positive]
    neg = y_score[~positive]
    m = pos.shape[0]
    n = neg.shape[0]
    if m == 0 or n == 0:
        raise ValueError("both classes must be present to compute an AUROC")

    v10, v01, auc = _structural_components(pos, neg)
    # Sample variances of the structural components (ddof=1 -> the m-1 / n-1
    # corrections). With a single element in a class the variance is 0.
    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return auc, float(var)


def delong_auc_ci(
    y_true: Sequence[int], y_score: Sequence[float], alpha: float = 0.05
) -> AUCResult:
    """AUROC with a logit-transformed normal-approximation CI.

    The interval is built on the logit scale so it cannot escape ``[0, 1]``
    (see the module docstring). A degenerate AUC of exactly 0 or 1 -- or a zero
    variance -- collapses the interval to the point estimate, which is the
    correct degenerate answer (perfect or worthless separation carries no
    normal-approximation width).
    """

    from scipy.stats import norm

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    auc, var = delong_auc_variance(y_true, y_score)
    se = float(np.sqrt(var))
    z = float(norm.ppf(1.0 - alpha / 2.0))

    if se == 0.0 or auc <= 0.0 or auc >= 1.0:
        point = min(1.0, max(0.0, auc))
        return AUCResult(auc=auc, se=se, ci_low=point, ci_high=point)

    # Delta-method SE on the logit scale: d/dp logit(p) = 1 / (p (1 - p)).
    logit = np.log(auc / (1.0 - auc))
    logit_se = se / (auc * (1.0 - auc))
    lo = logit - z * logit_se
    hi = logit + z * logit_se
    ci_low = float(1.0 / (1.0 + np.exp(-lo)))
    ci_high = float(1.0 / (1.0 + np.exp(-hi)))
    ci_low = min(1.0, max(0.0, ci_low))
    ci_high = min(1.0, max(0.0, ci_high))
    return AUCResult(auc=auc, se=se, ci_low=ci_low, ci_high=ci_high)


def delong_test(
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Compare two correlated AUROCs measured on the SAME samples.

    Both score vectors are aligned to the same ``y_true`` labels, so the two
    AUROCs are correlated; DeLong (1988, Sec. 3) tests their difference with a
    normal approximation whose variance uses the *covariance* of the two
    structural-component vectors.

    Returns
    -------
    ``(auc_a, auc_b, z, p_value)`` -- the two AUROCs, the z statistic for
    ``auc_a - auc_b``, and the two-sided p-value. Comparing a model to itself
    (identical scores) yields ``z == 0`` and ``p == 1``.
    """

    from scipy.stats import norm

    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if not (y_true.shape[0] == score_a.shape[0] == score_b.shape[0]):
        raise ValueError("y_true, score_a and score_b must share length")
    positive = y_true > 0
    m = int(positive.sum())
    n = int((~positive).sum())
    if m == 0 or n == 0:
        raise ValueError("both classes must be present to compare AUROCs")

    pos_a, neg_a = score_a[positive], score_a[~positive]
    pos_b, neg_b = score_b[positive], score_b[~positive]

    v10_a, v01_a, auc_a = _structural_components(pos_a, neg_a)
    v10_b, v01_b, auc_b = _structural_components(pos_b, neg_b)

    # 2x2 covariance of the structural components, stacked as rows [A, B].
    # cov(V10) uses ddof=1 across the m positives; cov(V01) across the n negs.
    v10 = np.vstack([v10_a, v10_b])
    v01 = np.vstack([v01_a, v01_b])
    s10 = np.cov(v10, ddof=1) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(v01, ddof=1) if n > 1 else np.zeros((2, 2))
    cov = s10 / m + s01 / n  # 2x2 covariance of (auc_a, auc_b)

    # var(auc_a - auc_b) = c^T Sigma c with c = [1, -1].
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    diff = auc_a - auc_b
    if var_diff <= 0.0:
        # Identical (or perfectly co-varying) models: no discernible difference.
        z = 0.0 if diff == 0.0 else float(np.sign(diff)) * np.inf
        p = 1.0 if diff == 0.0 else 0.0
        return float(auc_a), float(auc_b), float(z), float(p)

    z = diff / np.sqrt(var_diff)
    p = 2.0 * norm.sf(abs(z))
    return float(auc_a), float(auc_b), float(z), float(p)


__all__ = [
    "AUCResult",
    "delong_auc_ci",
    "delong_auc_variance",
    "delong_test",
]
