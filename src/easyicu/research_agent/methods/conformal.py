"""Conformal prediction: distribution-free coverage for the prediction family.

A calibrated AUROC/Brier report says *how well* a model discriminates and
calibrates on average. Split conformal prediction provides marginal coverage
over exchangeable calibration and test points, with a separately trained model.
It does not guarantee conditional coverage for an individual patient's features.

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

    Selects order statistic ceil((n+1)(1-alpha)). If that rank exceeds n,
    including n=0, return the score upper bound 1 (include every class).
    """

    import numpy as np

    scores = np.asarray(list(scores), dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be a finite 1D vector in [0, 1]")
    n = scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return 1.0
    return float(np.partition(scores, rank - 1)[rank - 1])


def _probs_2col(p1):
    import numpy as np

    p1 = np.asarray(list(p1), dtype=float)
    if p1.ndim != 1 or not np.isfinite(p1).all() or np.any((p1 < 0) | (p1 > 1)):
        raise ValueError("probabilities must be a finite 1D vector in [0, 1]")
    return np.stack([1.0 - p1, p1], axis=1)


def _binary_labels(labels, n):
    import numpy as np

    y = np.asarray(list(labels), dtype=float)
    if y.ndim != 1 or y.size != n:
        raise ValueError("probabilities and labels length/shape mismatch")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    return y.astype(int)


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
    y = _binary_labels(cal_labels, probs.shape[0])
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
    if set(thresholds) != {0, 1} or any(not 0 <= q <= 1 for q in thresholds.values()):
        raise ValueError("thresholds must specify both classes with values in [0, 1]")
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
    y = _binary_labels(test_labels, len(sets))
    covered = np.array([y[i] in sets[i] for i in range(y.shape[0])], dtype=float)
    sizes = np.array([len(s) for s in sets], dtype=float)
    per_class = {
        c: (float(covered[y == c].mean()) if np.any(y == c) else float("nan"))
        for c in (0, 1)
    }
    return ConformalResult(
        alpha=alpha,
        mondrian=mondrian,
        thresholds={int(k): float(v) for k, v in thresholds.items()},
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
        f"prediction sets achieved {result.coverage:.1%} empirical overall "
        f"coverage on the held-out split (mean set size {result.mean_set_size:.2f}), "
        f"using {scope} calibration. The coverage guarantee assumes exchangeable "
        "calibration/test points and a separately trained model; it is not an "
        "individual-patient conditional guarantee."
    )


__all__ = [
    "ConformalResult",
    "conformal_calibrate",
    "conformal_evaluate",
    "conformal_predict_sets",
    "conformal_sentence",
]
