"""Reclassification metrics: categorical/continuous NRI and IDI (analysis only).

When a new risk model is compared against an old one on the same subjects,
discrimination deltas such as the change in AUC can look flat even when the
new model moves many subjects across clinically meaningful risk thresholds.
The **Net Reclassification Improvement (NRI)** and the **Integrated
Discrimination Improvement (IDI)** (Pencina et al., *Stat Med* 2008;
*Stat Med* 2011) quantify that movement directly from the paired predicted
risks:

* **Categorical NRI** — with risk cutoffs (e.g. ``(0.1, 0.2)``) each subject
  falls into a risk category under the old and the new model. ``up`` means
  the new category is higher, ``down`` means it is lower::

      NRI_cat = [P(up|event) - P(down|event)]
              + [P(down|nonevent) - P(up|nonevent)]

* **Continuous (category-free) NRI** — any increase in predicted risk counts
  as ``up`` and any decrease as ``down`` (exact ties count as no movement).
  Same formula, no cutoffs.
* **IDI** — the change in discrimination slopes (mean risk in events minus
  mean risk in nonevents)::

      IDI = [mean(p_new|event) - mean(p_new|nonevent)]
          - [mean(p_old|event) - mean(p_old|nonevent)]

Uncertainty here is a percentile bootstrap with a fixed seed, so two runs
with the same ``random_state`` are bit-for-bit identical. Inputs are
validated fail-closed: out-of-range probabilities, length mismatches,
non-binary outcomes, or a sample with no events / no nonevents raise
``ValueError`` instead of returning a silent number.

Claim ceiling: ``analysis_only``. These are descriptive reclassification
summaries, not reportable evidence of clinical utility (which additionally
requires calibration, decision-curve, and external-validation evidence).

References
----------
Pencina MJ, D'Agostino RB Sr, D'Agostino RB Jr, Vasan RS. "Evaluating the
added predictive ability of a new marker: from area under the ROC curve to
reclassification and beyond." *Stat Med* 2008;27(2):157-172.
Pencina MJ, D'Agostino RB Sr, Steyerberg EW. "Extensions of net
reclassification improvement calculations to measure usefulness of new
biomarkers." *Stat Med* 2011;30(1):11-21.

Pure numpy — no optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class NRIResult:
    """Net reclassification improvement with a percentile-bootstrap CI."""

    kind: str  # "categorical" or "continuous"
    nri: float
    nri_events: float
    nri_nonevents: float
    ci_low: float
    ci_high: float
    se: float
    n_events: int
    n_nonevents: int
    n_bootstrap: int
    n_successful: int
    ci_level: float
    random_state: int
    cutoffs: Tuple[float, ...] = ()
    claim_ceiling: str = "analysis_only"
    method: str = "pencina_nri_percentile_bootstrap"

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "nri": self.nri,
            "nri_events": self.nri_events,
            "nri_nonevents": self.nri_nonevents,
            "ci": [self.ci_low, self.ci_high],
            "se": self.se,
            "n_events": self.n_events,
            "n_nonevents": self.n_nonevents,
            "n_bootstrap": self.n_bootstrap,
            "n_successful": self.n_successful,
            "ci_level": self.ci_level,
            "random_state": self.random_state,
            "cutoffs": list(self.cutoffs),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


@dataclass(frozen=True)
class IDIResult:
    """Integrated discrimination improvement with a percentile-bootstrap CI."""

    idi: float
    slope_new: float
    slope_old: float
    ci_low: float
    ci_high: float
    se: float
    n_events: int
    n_nonevents: int
    n_bootstrap: int
    n_successful: int
    ci_level: float
    random_state: int
    claim_ceiling: str = "analysis_only"
    method: str = "pencina_idi_percentile_bootstrap"

    def to_json(self) -> Dict[str, Any]:
        return {
            "idi": self.idi,
            "slope_new": self.slope_new,
            "slope_old": self.slope_old,
            "ci": [self.ci_low, self.ci_high],
            "se": self.se,
            "n_events": self.n_events,
            "n_nonevents": self.n_nonevents,
            "n_bootstrap": self.n_bootstrap,
            "n_successful": self.n_successful,
            "ci_level": self.ci_level,
            "random_state": self.random_state,
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_risks(
    y_true: Sequence[int],
    p_old: Sequence[float],
    p_new: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=float)
    po = np.asarray(list(p_old), dtype=float)
    pn = np.asarray(list(p_new), dtype=float)
    if y.ndim != 1 or po.ndim != 1 or pn.ndim != 1:
        raise ValueError("y_true, p_old and p_new must be one-dimensional")
    if not (y.shape[0] == po.shape[0] == pn.shape[0]):
        raise ValueError("y_true, p_old and p_new must have equal length")
    if y.shape[0] < 2:
        raise ValueError("reclassification metrics need at least two subjects")
    if not (np.isfinite(y).all() and np.isfinite(po).all() and np.isfinite(pn).all()):
        raise ValueError("y_true, p_old and p_new must all be finite (no NaN/inf)")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_true must be binary with values in {0, 1}")
    if ((po < 0.0) | (po > 1.0)).any() or ((pn < 0.0) | (pn > 1.0)).any():
        raise ValueError("p_old and p_new must be probabilities in [0, 1]")
    if not (y == 1.0).any() or not (y == 0.0).any():
        raise ValueError("reclassification metrics need at least one event and one nonevent")
    return y, po, pn


def _validate_bootstrap(n_bootstrap: object, ci_level: object) -> Tuple[int, float]:
    n_boot = _as_int(n_bootstrap, "n_bootstrap")
    if n_boot < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(ci_level, bool) or not isinstance(ci_level, (int, float, np.floating)):
        raise ValueError("ci_level must be a number in (0, 1)")
    level = float(ci_level)
    if not (0.0 < level < 1.0):
        raise ValueError("ci_level must be strictly between 0 and 1")
    return n_boot, level


def _validate_cutoffs(cutoffs: Sequence[float]) -> Tuple[float, ...]:
    cuts = tuple(float(c) for c in list(cutoffs))
    if len(cuts) < 1:
        raise ValueError("categorical NRI needs at least one cutoff; use continuous_nri otherwise")
    if not all(np.isfinite(c) for c in cuts):
        raise ValueError("cutoffs must be finite")
    if not all(0.0 < c < 1.0 for c in cuts):
        raise ValueError("cutoffs must lie strictly inside (0, 1)")
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError("cutoffs must be strictly increasing")
    return cuts


def _nri_from_moves(
    y: np.ndarray, up: np.ndarray, down: np.ndarray
) -> Tuple[float, float, float]:
    """Return ``(nri, nri_events, nri_nonevents)`` from boolean move masks."""
    ev = y == 1.0
    ne = y == 0.0
    nri_events = float(np.mean(up[ev])) - float(np.mean(down[ev]))
    nri_nonevents = float(np.mean(down[ne])) - float(np.mean(up[ne]))
    return nri_events + nri_nonevents, nri_events, nri_nonevents


def _bootstrap_distribution(
    statistic: Callable[[np.ndarray], float],
    n: int,
    n_bootstrap: int,
    random_state: int,
) -> Tuple[np.ndarray, int]:
    """Percentile-bootstrap distribution of ``statistic`` over row indices.

    Replicates that leave a single outcome class (statistic undefined, hence
    NaN) are skipped rather than imputed; the count of usable replicates is
    returned alongside. Raises ``ValueError`` when no replicate is usable.
    """
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        stat = float(statistic(idx))
        if np.isfinite(stat):
            values.append(stat)
    if not values:
        raise ValueError("bootstrap produced no usable replicate (degenerate resamples)")
    return np.asarray(values, dtype=float), len(values)


def _summarise_bootstrap(draws: np.ndarray, ci_level: float) -> Tuple[float, float, float]:
    alpha = 1.0 - ci_level
    ci_low = float(np.quantile(draws, alpha / 2.0))
    ci_high = float(np.quantile(draws, 1.0 - alpha / 2.0))
    se = float(np.std(draws, ddof=1)) if draws.shape[0] > 1 else 0.0
    return ci_low, ci_high, se


def _category(values: np.ndarray, cutoffs: Tuple[float, ...]) -> np.ndarray:
    """Risk category per subject: number of cutoffs ``<=`` the risk.

    A risk exactly equal to a cutoff counts into the higher category, i.e.
    categories are ``[0, c1], (c1, c2], ..., (ck, 1]``.
    """
    return np.searchsorted(np.asarray(cutoffs, dtype=float), values, side="right")


def categorical_nri(
    y_true: Sequence[int],
    p_old: Sequence[float],
    p_new: Sequence[float],
    cutoffs: Sequence[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 0,
) -> NRIResult:
    """Categorical NRI for paired old/new risks at the given ``cutoffs``."""
    y, po, pn = _validate_risks(y_true, p_old, p_new)
    cuts = _validate_cutoffs(cutoffs)
    n_boot, level = _validate_bootstrap(n_bootstrap, ci_level)
    seed = _as_int(random_state, "random_state")

    cat_old = _category(po, cuts)
    cat_new = _category(pn, cuts)
    up = cat_new > cat_old
    down = cat_new < cat_old
    nri, nri_ev, nri_ne = _nri_from_moves(y, up, down)

    def _stat(idx: np.ndarray) -> float:
        yy = y[idx]
        if not (yy == 1.0).any() or not (yy == 0.0).any():
            return float("nan")
        return _nri_from_moves(yy, up[idx], down[idx])[0]

    draws, n_ok = _bootstrap_distribution(_stat, y.shape[0], n_boot, seed)
    ci_low, ci_high, se = _summarise_bootstrap(draws, level)
    return NRIResult(
        kind="categorical",
        nri=nri,
        nri_events=nri_ev,
        nri_nonevents=nri_ne,
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        n_events=int(np.sum(y == 1.0)),
        n_nonevents=int(np.sum(y == 0.0)),
        n_bootstrap=n_boot,
        n_successful=n_ok,
        ci_level=level,
        random_state=seed,
        cutoffs=cuts,
    )


def continuous_nri(
    y_true: Sequence[int],
    p_old: Sequence[float],
    p_new: Sequence[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 0,
) -> NRIResult:
    """Continuous (category-free) NRI: any risk increase is ``up``."""
    y, po, pn = _validate_risks(y_true, p_old, p_new)
    n_boot, level = _validate_bootstrap(n_bootstrap, ci_level)
    seed = _as_int(random_state, "random_state")

    up = pn > po
    down = pn < po
    nri, nri_ev, nri_ne = _nri_from_moves(y, up, down)

    def _stat(idx: np.ndarray) -> float:
        yy = y[idx]
        if not (yy == 1.0).any() or not (yy == 0.0).any():
            return float("nan")
        return _nri_from_moves(yy, up[idx], down[idx])[0]

    draws, n_ok = _bootstrap_distribution(_stat, y.shape[0], n_boot, seed)
    ci_low, ci_high, se = _summarise_bootstrap(draws, level)
    return NRIResult(
        kind="continuous",
        nri=nri,
        nri_events=nri_ev,
        nri_nonevents=nri_ne,
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        n_events=int(np.sum(y == 1.0)),
        n_nonevents=int(np.sum(y == 0.0)),
        n_bootstrap=n_boot,
        n_successful=n_ok,
        ci_level=level,
        random_state=seed,
    )


def idi(
    y_true: Sequence[int],
    p_old: Sequence[float],
    p_new: Sequence[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 0,
) -> IDIResult:
    """Integrated discrimination improvement for paired old/new risks."""
    y, po, pn = _validate_risks(y_true, p_old, p_new)
    n_boot, level = _validate_bootstrap(n_bootstrap, ci_level)
    seed = _as_int(random_state, "random_state")

    def _slopes(yy: np.ndarray, pp_old: np.ndarray, pp_new: np.ndarray) -> Tuple[float, float, float]:
        ev = yy == 1.0
        ne = yy == 0.0
        slope_new = float(np.mean(pp_new[ev])) - float(np.mean(pp_new[ne]))
        slope_old = float(np.mean(pp_old[ev])) - float(np.mean(pp_old[ne]))
        return slope_new - slope_old, slope_new, slope_old

    point, slope_new, slope_old = _slopes(y, po, pn)

    def _stat(idx: np.ndarray) -> float:
        yy = y[idx]
        if not (yy == 1.0).any() or not (yy == 0.0).any():
            return float("nan")
        return _slopes(yy, po[idx], pn[idx])[0]

    draws, n_ok = _bootstrap_distribution(_stat, y.shape[0], n_boot, seed)
    ci_low, ci_high, se = _summarise_bootstrap(draws, level)
    return IDIResult(
        idi=point,
        slope_new=slope_new,
        slope_old=slope_old,
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        n_events=int(np.sum(y == 1.0)),
        n_nonevents=int(np.sum(y == 0.0)),
        n_bootstrap=n_boot,
        n_successful=n_ok,
        ci_level=level,
        random_state=seed,
    )


__all__ = [
    "IDIResult",
    "NRIResult",
    "categorical_nri",
    "continuous_nri",
    "idi",
]
