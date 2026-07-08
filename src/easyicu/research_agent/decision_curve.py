"""Decision-curve analysis: net benefit across threshold probabilities.

Discrimination (AUROC) and calibration say how well a risk model ranks and
matches observed frequencies, but neither tells a clinician whether *acting* on
the model does more good than harm. Decision-curve analysis (Vickers & Elkin,
*Med Decis Making* 2006;26:565-574) answers that directly by putting benefits
and harms on one axis -- net benefit -- as a function of the threshold
probability at which a clinician would choose to treat.

At a threshold probability ``pt`` a patient is classified positive iff the
predicted probability is ``>= pt``. The threshold encodes the exchange rate a
decision-maker places on a false positive versus a true positive: someone who
would treat only at a high ``pt`` considers a false-positive intervention
costly, so the false-positive term is up-weighted. The net benefit of the model
over the whole sample of size ``n`` is

    NB(model)     = TP/n - (FP/n) * (pt / (1 - pt))

where ``TP`` and ``FP`` are true- and false-positive counts. The two reference
strategies are

    NB(treat all)  = prevalence - (1 - prevalence) * (pt / (1 - pt))
    NB(treat none) = 0

A model is worth using at a given ``pt`` when its net-benefit curve lies above
both the treat-all and treat-none lines. The units of net benefit are "true
positives per patient", already netted against the harm of false positives at
that threshold's exchange rate; multiplying by ``n`` recovers a count.

Pure numpy/pandas -- no SDK, no optional dependency. Intended wiring: the
prediction step registers a ``net_benefit`` table plus scalar net-benefit values
at clinically relevant thresholds that the decision-curve figure and manuscript
can cite alongside AUROC and calibration.

Reference: Vickers AJ, Elkin EB. Decision curve analysis: a novel method for
evaluating prediction models. Med Decis Making. 2006;26(6):565-574.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# Stable, trace-friendly column names for the returned frame. Downstream
# figure/trace code keys on these, so they must not drift.
_COL_THRESHOLD = "threshold"
_COL_MODEL = "net_benefit_model"
_COL_ALL = "net_benefit_all"
_COL_NONE = "net_benefit_none"


@dataclass
class DecisionCurveResult:
    """Net-benefit summary over a decision-curve sweep."""

    prevalence: float
    n: int
    thresholds: List[float] = field(default_factory=list)
    net_benefit_model: List[float] = field(default_factory=list)
    net_benefit_all: List[float] = field(default_factory=list)
    best_threshold: Optional[float] = None
    best_net_benefit: Optional[float] = None


def _coerce_binary_labels(y_true) -> "object":
    """Return ``y_true`` as an int array of 0/1, raising on anything else."""

    import numpy as np

    y = np.asarray(list(y_true), dtype=float)
    if y.size == 0:
        raise ValueError("y_true is empty")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_true contains non-finite values")
    uniq = set(np.unique(y).tolist())
    if not uniq <= {0.0, 1.0}:
        raise ValueError(f"y_true must be binary 0/1, saw values {sorted(uniq)}")
    return y.astype(int)


def net_benefit_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> "object":
    """Net-benefit curve for a binary risk model (Vickers & Elkin 2006).

    Parameters
    ----------
    y_true:
        Binary outcomes, 0 or 1, length ``n``.
    y_prob:
        Predicted P(outcome = 1) for each patient, same length as ``y_true``.
    thresholds:
        Threshold probabilities ``pt`` to evaluate. Defaults to
        ``numpy.arange(0.01, 1.00, 0.01)``. Thresholds are clipped into
        ``[0, 1)`` and any ``pt`` at or above 1 is dropped so the
        ``pt / (1 - pt)`` odds term never divides by zero.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``net_benefit_model``, ``net_benefit_all``,
        ``net_benefit_none`` (one row per evaluated threshold). ``treat none``
        is identically zero and is included so the frame is self-describing.

    Notes
    -----
    A patient is classified positive iff ``y_prob >= pt``. Counts ``TP`` and
    ``FP`` are taken over the whole sample; net benefit is normalised by ``n``,
    giving "net true positives per patient" at that threshold's exchange rate.
    """

    import numpy as np
    import pandas as pd

    y = _coerce_binary_labels(y_true)
    p = np.asarray(list(y_prob), dtype=float)
    if p.shape[0] != y.shape[0]:
        raise ValueError("y_true and y_prob length mismatch")
    if not np.all(np.isfinite(p)):
        raise ValueError("y_prob contains non-finite values")

    n = int(y.shape[0])
    prevalence = float(y.mean())

    if thresholds is None:
        thr = np.arange(0.01, 1.00, 0.01)
    else:
        thr = np.asarray(list(thresholds), dtype=float)
    # Keep only usable thresholds in [0, 1): pt == 1 divides by zero, and a
    # negative pt is meaningless. This is what makes pt -> 1 safe.
    thr = thr[np.isfinite(thr) & (thr >= 0.0) & (thr < 1.0)]

    rows_thr: List[float] = []
    nb_model: List[float] = []
    nb_all: List[float] = []
    for pt in thr:
        odds = pt / (1.0 - pt)
        predicted_positive = p >= pt
        tp = float(np.sum(predicted_positive & (y == 1)))
        fp = float(np.sum(predicted_positive & (y == 0)))
        model = tp / n - (fp / n) * odds
        treat_all = prevalence - (1.0 - prevalence) * odds
        rows_thr.append(float(pt))
        nb_model.append(model)
        nb_all.append(treat_all)

    return pd.DataFrame(
        {
            _COL_THRESHOLD: rows_thr,
            _COL_MODEL: nb_model,
            _COL_ALL: nb_all,
            _COL_NONE: [0.0] * len(rows_thr),
        }
    )


def summarize_decision_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> DecisionCurveResult:
    """Run :func:`net_benefit_curve` and reduce it to a citable summary.

    ``best_threshold`` / ``best_net_benefit`` report where the model's net
    benefit peaks over the swept thresholds. When the sweep is empty (no usable
    thresholds) both are ``None``.
    """

    import numpy as np

    frame = net_benefit_curve(y_true, y_prob, thresholds)
    y = _coerce_binary_labels(y_true)
    prevalence = float(y.mean())
    n = int(y.shape[0])

    best_threshold: Optional[float] = None
    best_net_benefit: Optional[float] = None
    if len(frame) > 0:
        model_vals = frame[_COL_MODEL].to_numpy(dtype=float)
        idx = int(np.argmax(model_vals))
        best_threshold = float(frame[_COL_THRESHOLD].to_numpy(dtype=float)[idx])
        best_net_benefit = float(model_vals[idx])

    return DecisionCurveResult(
        prevalence=round(prevalence, 6),
        n=n,
        thresholds=[float(v) for v in frame[_COL_THRESHOLD].tolist()],
        net_benefit_model=[float(v) for v in frame[_COL_MODEL].tolist()],
        net_benefit_all=[float(v) for v in frame[_COL_ALL].tolist()],
        best_threshold=best_threshold,
        best_net_benefit=(
            round(best_net_benefit, 6) if best_net_benefit is not None else None
        ),
    )


def net_benefit_at(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    """Net benefit of model / treat-all / treat-none at a single threshold.

    Convenience wrapper for citing one clinically chosen ``pt`` (e.g. 0.10 for a
    mortality model) without materialising the whole sweep.
    """

    frame = net_benefit_curve(y_true, y_prob, [threshold])
    if len(frame) == 0:
        raise ValueError(f"threshold {threshold} is not in [0, 1)")
    row = frame.iloc[0]
    return {
        _COL_THRESHOLD: float(row[_COL_THRESHOLD]),
        _COL_MODEL: float(row[_COL_MODEL]),
        _COL_ALL: float(row[_COL_ALL]),
        _COL_NONE: float(row[_COL_NONE]),
    }


__all__ = [
    "DecisionCurveResult",
    "net_benefit_at",
    "net_benefit_curve",
    "summarize_decision_curve",
]
