"""The AUROC interval a static prediction reports, chosen by how its rows depend.

Owner
-----
This module owns one choice: which interval an AUROC, or the paired
difference of two AUROCs on the same rows, gets.  A prediction's validation
rows are ICU stays, and a patient may contribute several.

- When every patient contributes one row, the rows are independent and the
  DeLong intervals hold (:mod:`.delong_auc`): the logit interval for one
  AUROC, and for a difference the paired normal interval with its z and p.
- When a patient contributes several, the DeLong variance treats those stays
  as independent and is too small.  The patients are resampled instead --
  each draw keeps all of a drawn patient's stays -- 2,000 times from seed
  1729, and the percentile interval is reported.  No p value is computed
  from the draws.

The resampling is stratified, as pROC's bootstrap of an AUROC is by default:
patients with an event stay and patients without one are drawn separately,
each stratum keeping its size, so a draw holds both outcome classes even
when the events are few.  A draw that still holds one class (every patient
has an event, and the drawn ones have no other stay) has no AUROC: it is
skipped and counted, and more than 1% skipped stops the interval
(``auc_bootstrap_degenerate``) instead of reporting one from the draws that
happened to work.

A draw's AUROC is the Mann-Whitney statistic under frequency weights on the
original rows -- a patient drawn k times weighs k on each of their stays --
with ties counting one half, so a draw that keeps every patient once returns
the sample AUROC exactly.  Each score's tie groups are computed once, which
keeps a draw linear in the rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from .delong_auc import delong_auc_ci, delong_difference

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 1729
MAX_SKIPPED_FRACTION = 0.01
DELONG_METHOD = "delong_logit_normal_95pct"
DELONG_PAIRED_METHOD = "delong_paired_normal_95pct"
CLUSTER_BOOTSTRAP_METHOD = "patient_stratified_bootstrap_percentile_95pct"
_Z_975 = 1.959963984540054


class AUCIntervalError(RuntimeError):
    """An interval this owner cannot report; ``code`` says why."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class AUCInterval:
    auc: float
    se: float
    ci_low: float
    ci_high: float
    method: str
    row_n: int
    subject_n: int
    bootstrap_n: int = 0
    bootstrap_skipped_n: int = 0


@dataclass(frozen=True)
class AUCDifference:
    """``auc_a - auc_b`` on the same rows; ``z`` and ``p_value`` only from DeLong."""

    auc_a: float
    auc_b: float
    difference: float
    se: float
    ci_low: float
    ci_high: float
    z: Optional[float]
    p_value: Optional[float]
    method: str
    row_n: int
    subject_n: int
    bootstrap_n: int = 0
    bootstrap_skipped_n: int = 0


def _labels(outcome: Sequence[int]) -> np.ndarray:
    labels = np.asarray(outcome, dtype=float)
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("outcome must be a complete 1D 0/1 vector")
    if labels.min() == labels.max():
        raise ValueError("both outcome classes must be present to compute an AUROC")
    return labels.astype(bool)


def _scores(score: Sequence[float], n: int) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    if values.ndim != 1 or values.size != n or not np.isfinite(values).all():
        raise ValueError("scores must be finite and aligned with the outcome")
    return values


def _subject_codes(subject_ids: Sequence[object], n: int) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(list(subject_ids), dtype=object), sort=False)
    if codes.size != n:
        raise ValueError("subject ids must be aligned with the outcome")
    if (codes < 0).any():
        raise ValueError("subject ids must be complete")
    return codes, len(uniques)


class _TieGroups:
    """One score vector's ordered tie groups, split by outcome class."""

    def __init__(self, score: np.ndarray, positive: np.ndarray) -> None:
        values, group = np.unique(score, return_inverse=True)
        self._size = values.size
        self._positive_rows = np.flatnonzero(positive)
        self._negative_rows = np.flatnonzero(~positive)
        self._positive_group = group[self._positive_rows]
        self._negative_group = group[self._negative_rows]

    def auc(self, weights: np.ndarray) -> Optional[float]:
        positive = np.bincount(
            self._positive_group, weights=weights[self._positive_rows], minlength=self._size
        )
        negative = np.bincount(
            self._negative_group, weights=weights[self._negative_rows], minlength=self._size
        )
        positive_total = float(positive.sum())
        negative_total = float(negative.sum())
        if positive_total == 0.0 or negative_total == 0.0:
            return None
        below = np.cumsum(negative) - negative
        return float((positive * (below + 0.5 * negative)).sum() / (positive_total * negative_total))


def _draw_weights(
    codes: np.ndarray, subject_n: int, positive: np.ndarray
) -> Iterator[np.ndarray]:
    """Row weights of each draw: patients with and without an event, apart."""

    has_event = np.bincount(codes, weights=positive.astype(float), minlength=subject_n) > 0
    strata = [np.flatnonzero(has_event), np.flatnonzero(~has_event)]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        counts = np.zeros(subject_n, dtype=float)
        for stratum in strata:
            if stratum.size:
                drawn = stratum[generator.integers(0, stratum.size, size=stratum.size)]
                counts += np.bincount(drawn, minlength=subject_n)
        yield counts[codes]


def _check_skipped(skipped: int) -> None:
    if skipped > MAX_SKIPPED_FRACTION * BOOTSTRAP_DRAWS:
        raise AUCIntervalError(
            "auc_bootstrap_degenerate",
            f"{skipped} of {BOOTSTRAP_DRAWS} patient resamples held one outcome class",
        )


def _percentiles(values: Sequence[float]) -> tuple[float, float, float]:
    draws = np.asarray(values, dtype=float)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(np.std(draws, ddof=1)), float(low), float(high)


def auc_interval(
    outcome: Sequence[int], score: Sequence[float], subject_ids: Sequence[object]
) -> AUCInterval:
    """The AUROC of ``score`` with the interval its rows' dependence calls for."""

    positive = _labels(outcome)
    values = _scores(score, positive.size)
    codes, subject_n = _subject_codes(subject_ids, positive.size)
    if subject_n == positive.size:
        result = delong_auc_ci(positive.astype(int), values)
        return AUCInterval(
            auc=result.auc,
            se=result.se,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            method=DELONG_METHOD,
            row_n=int(positive.size),
            subject_n=subject_n,
        )
    groups = _TieGroups(values, positive)
    point = groups.auc(np.ones(positive.size))
    assert point is not None  # both classes are present
    draws: list[float] = []
    skipped = 0
    for weights in _draw_weights(codes, subject_n, positive):
        value = groups.auc(weights)
        if value is None:
            skipped += 1
        else:
            draws.append(value)
    _check_skipped(skipped)
    se, low, high = _percentiles(draws)
    return AUCInterval(
        auc=point,
        se=se,
        ci_low=low,
        ci_high=high,
        method=CLUSTER_BOOTSTRAP_METHOD,
        row_n=int(positive.size),
        subject_n=subject_n,
        bootstrap_n=BOOTSTRAP_DRAWS,
        bootstrap_skipped_n=skipped,
    )


def paired_auc_difference(
    outcome: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    subject_ids: Sequence[object],
) -> AUCDifference:
    """``AUROC(a) - AUROC(b)`` on the same rows, with the interval their dependence calls for."""

    positive = _labels(outcome)
    values_a = _scores(score_a, positive.size)
    values_b = _scores(score_b, positive.size)
    codes, subject_n = _subject_codes(subject_ids, positive.size)
    if subject_n == positive.size:
        auc_a, auc_b, difference, variance = delong_difference(
            positive.astype(int), values_a, values_b
        )
        se = float(np.sqrt(max(variance, 0.0)))
        if se == 0.0:
            z = p_value = None
            low = high = difference
        else:
            from scipy.stats import norm

            z = difference / se
            p_value = float(2.0 * norm.sf(abs(z)))
            low, high = difference - _Z_975 * se, difference + _Z_975 * se
        return AUCDifference(
            auc_a=auc_a,
            auc_b=auc_b,
            difference=difference,
            se=se,
            ci_low=float(low),
            ci_high=float(high),
            z=None if z is None else float(z),
            p_value=p_value,
            method=DELONG_PAIRED_METHOD,
            row_n=int(positive.size),
            subject_n=subject_n,
        )
    groups_a = _TieGroups(values_a, positive)
    groups_b = _TieGroups(values_b, positive)
    ones = np.ones(positive.size)
    auc_a, auc_b = groups_a.auc(ones), groups_b.auc(ones)
    assert auc_a is not None and auc_b is not None
    draws: list[float] = []
    skipped = 0
    for weights in _draw_weights(codes, subject_n, positive):
        value_a = groups_a.auc(weights)
        if value_a is None:
            skipped += 1
            continue
        value_b = groups_b.auc(weights)
        assert value_b is not None  # the same weights hold both classes
        draws.append(value_a - value_b)
    _check_skipped(skipped)
    se, low, high = _percentiles(draws)
    return AUCDifference(
        auc_a=auc_a,
        auc_b=auc_b,
        difference=auc_a - auc_b,
        se=se,
        ci_low=low,
        ci_high=high,
        z=None,
        p_value=None,
        method=CLUSTER_BOOTSTRAP_METHOD,
        row_n=int(positive.size),
        subject_n=subject_n,
        bootstrap_n=BOOTSTRAP_DRAWS,
        bootstrap_skipped_n=skipped,
    )


__all__ = [
    "AUCDifference",
    "AUCInterval",
    "AUCIntervalError",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CLUSTER_BOOTSTRAP_METHOD",
    "DELONG_METHOD",
    "DELONG_PAIRED_METHOD",
    "auc_interval",
    "paired_auc_difference",
]
