"""Conformal prediction: distribution-free coverage for the prediction family.

A calibrated AUROC/Brier report says *how well* a model discriminates and
calibrates on average; it does not give a per-patient guarantee. Split
(inductive) conformal prediction adds one: for a chosen error rate alpha, the
prediction *set* for each patient contains the true label with probability at
least 1 - alpha, with no distributional assumptions -- only exchangeability of
the calibration and test data.

For an imbalanced ICU outcome (mortality), marginal coverage can be met while
the minority class is systematically under-covered, so the default here is
Mondrian (class-conditional) calibration, which guarantees coverage *within
each class*.

Pure numpy, no SDK. Intended wiring: the prediction step fits a model, holds
out a calibration split, and registers ``conformal_coverage`` /
``conformal_set_size`` statistics that the prediction figure and manuscript can
cite alongside AUROC and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set


@dataclass
class ConformalResult:
    alpha: float
    mondrian: bool
    thresholds: Dict[int, float]
    coverage: float
    mean_set_size: float
    per_class_coverage: Dict[int, float] = field(default_factory=dict)
    empty_fraction: float = 0.0
    uncertain_fraction: float = 0.0


def _conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity scores.

    Uses the ceil((n+1)(1-alpha))/n level with 'higher' interpolation, the
    standard split-conformal correction that yields the >= 1-alpha guarantee.
    An empty calibration set returns 1.0 (include every class -> trivially
    valid, maximally uninformative), never a spurious tight threshold.
    """

    import numpy as np

    scores = np.asarray(list(scores), dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return 1.0
    level = min(1.0, (np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(scores, level, method="higher"))


def _probs_2col(p1):
    import numpy as np

    p1 = np.clip(np.asarray(list(p1), dtype=float), 0.0, 1.0)
    return np.stack([1.0 - p1, p1], axis=1)


def conformal_calibrate(
    cal_prob_pos: Sequence[float],
    cal_labels: Sequence[int],
    *,
    alpha: float = 0.1,
    mondrian: bool = True,
) -> Dict[int, float]:
    """Per-class nonconformity thresholds from a labelled calibration split.

    ``cal_prob_pos`` is the model's predicted P(class=1). The nonconformity
    score of a calibration point is ``1 - p(true class)``.
    """

    import numpy as np

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    probs = _probs_2col(cal_prob_pos)
    y = np.asarray(list(cal_labels), dtype=int)
    if probs.shape[0] != y.shape[0]:
        raise ValueError("calibration probabilities and labels length mismatch")
    scores = 1.0 - probs[np.arange(y.shape[0]), y]
    if mondrian:
        return {c: _conformal_quantile(scores[y == c], alpha) for c in (0, 1)}
    q = _conformal_quantile(scores, alpha)
    return {0: q, 1: q}


def conformal_predict_sets(
    test_prob_pos: Sequence[float],
    thresholds: Dict[int, float],
) -> List[Set[int]]:
    """Prediction set per test point: classes whose nonconformity <= threshold."""

    probs = _probs_2col(test_prob_pos)
    sets: List[Set[int]] = []
    for row in probs:
        s = {c for c in (0, 1) if (1.0 - row[c]) <= thresholds.get(c, 1.0)}
        sets.append(s)
    return sets


def conformal_evaluate(
    cal_prob_pos: Sequence[float],
    cal_labels: Sequence[int],
    test_prob_pos: Sequence[float],
    test_labels: Sequence[int],
    *,
    alpha: float = 0.1,
    mondrian: bool = True,
) -> ConformalResult:
    """Calibrate on the calibration split, evaluate coverage on the test split."""

    import numpy as np

    thresholds = conformal_calibrate(
        cal_prob_pos, cal_labels, alpha=alpha, mondrian=mondrian
    )
    sets = conformal_predict_sets(test_prob_pos, thresholds)
    y = np.asarray(list(test_labels), dtype=int)
    covered = np.array([y[i] in sets[i] for i in range(y.shape[0])], dtype=float)
    sizes = np.array([len(s) for s in sets], dtype=float)
    per_class = {
        c: (float(covered[y == c].mean()) if np.any(y == c) else float("nan"))
        for c in (0, 1)
    }
    return ConformalResult(
        alpha=alpha,
        mondrian=mondrian,
        thresholds={int(k): round(float(v), 4) for k, v in thresholds.items()},
        coverage=round(float(covered.mean()), 4) if covered.size else float("nan"),
        mean_set_size=round(float(sizes.mean()), 4) if sizes.size else float("nan"),
        per_class_coverage={
            k: (round(v, 4) if v == v else v) for k, v in per_class.items()
        },
        empty_fraction=round(float(np.mean(sizes == 0)), 4) if sizes.size else 0.0,
        uncertain_fraction=round(float(np.mean(sizes == 2)), 4) if sizes.size else 0.0,
    )


def conformal_sentence(result: ConformalResult) -> str:
    """A ready-to-cite Results sentence for the conformal coverage."""

    scope = "class-conditional (Mondrian)" if result.mondrian else "marginal"
    return (
        f"At a target error rate of {result.alpha:.0%}, split-conformal "
        f"prediction sets achieved {result.coverage:.1%} empirical {scope} "
        f"coverage on the held-out split (mean set size {result.mean_set_size:.2f}), "
        "giving a distribution-free per-patient guarantee alongside the "
        "discrimination and calibration metrics."
    )


__all__ = [
    "ConformalResult",
    "conformal_calibrate",
    "conformal_evaluate",
    "conformal_predict_sets",
    "conformal_sentence",
]
