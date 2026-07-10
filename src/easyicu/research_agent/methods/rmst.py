"""Restricted mean survival time (RMST): a PH-free survival effect measure.

A Cox hazard ratio summarises a treatment effect on the hazard scale, but it
is only interpretable when the proportional-hazards (PH) assumption holds. When
the two survival curves cross, plateau, or separate late, the single HR is a
weighted average of a time-varying effect that no reader can map back to a
patient-level quantity. The **restricted mean survival time** avoids that trap.

For a horizon ``tau``, the RMST is the area under the Kaplan-Meier survival
curve on ``[0, tau]``::

    RMST(tau) = integral_{0}^{tau} S(u) du

It has a direct clinical reading: the *average event-free time within the first
tau units of follow-up* (e.g. mean ICU-free days over 28 days). With no
censoring this reduces to ``mean(min(T, tau))``. Because it is a functional of
the whole curve rather than a hazard ratio, it needs **no** PH assumption, so
``RMST_diff = RMST_a(tau) - RMST_b(tau)`` is a valid group contrast even when a
Cox HR would be misleading.

Estimation here is the standard KM-plug-in. The KM survival is a right-
continuous step function that drops at each observed event time; integrating it
over ``[0, tau]`` is a sum of rectangles ``S(t_i) * (t_{i+1} - t_i)`` with the
final rectangle clipped at ``tau``. The variance uses the classic integral form
(Kaplan-Meier / Andersen et al.)::

    Var(RMST) = sum_{t_i <= tau}  A_i^2  *  d_i / (n_i * (n_i - d_i))

    where A_i = integral_{t_i}^{tau} S(u) du   (area of the curve to the right
    of event time t_i, clipped at tau), d_i is the number of events at t_i and
    n_i the number at risk just before t_i. Terms with ``n_i == d_i`` contribute
    zero (no remaining variance to accrue).

Both the point estimate and its SE are computed from the pure-numpy KM step
function above, so results are bit-for-bit reproducible with no optional
dependency. (An earlier version borrowed the SE from lifelines'
``restricted_mean_survival_time(..., return_variance=True)``, but that call
returns the *population* variance of the restricted survival time,
``E[min(T,tau)^2] - E[min(T,tau)]^2`` — a distributional dispersion that does
NOT shrink with n — not the *sampling* SE of the RMST estimator; using it
inflated the CI by ~sqrt(n). The integral-form variance below is the correct
estimator SE.)

Reference
---------
Royston P, Parmar MKB. "Restricted mean survival time: an alternative to the
hazard ratio for the design and analysis of randomized trials with a
time-to-event outcome." *BMC Med Res Methodol* 2013;13:152.
Andersen PK, Hansen MG, Klein JP. "Regression analysis of restricted mean
survival time based on pseudo-observations." *Lifetime Data Anal* 2004;10:335.

Pure numpy/scipy — no optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RMSTResult:
    """Restricted mean survival time up to ``tau`` with a Wald CI."""

    rmst: float
    se: float
    ci_low: float
    ci_high: float
    tau: float


def _clean_inputs(
    durations: Sequence[float],
    event_observed: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(list(durations), dtype=float)
    e = np.asarray(list(event_observed), dtype=float)
    if d.shape[0] != e.shape[0]:
        raise ValueError("durations and event_observed must have equal length")
    finite = np.isfinite(d)
    d, e = d[finite], e[finite]
    if np.any(d < 0):
        raise ValueError("durations must be non-negative")
    # Coerce event flags to 0/1; anything truthy and finite is an event.
    e = (e != 0).astype(float)
    return d, e


def _km_integral(
    durations: np.ndarray,
    event_observed: np.ndarray,
    tau: float,
) -> Tuple[float, float]:
    """Area under the KM curve on ``[0, tau]`` and its integral-form variance.

    Returns ``(rmst, variance)``. Both are computed from the pure-numpy KM step
    function so the point estimate never depends on an optional package.
    """

    if tau <= 0:
        return 0.0, 0.0

    n_total = durations.shape[0]
    if n_total == 0:
        return float(tau), 0.0

    # Distinct event times (censoring times do not drop the curve, but they do
    # reduce the at-risk set for later event times).
    event_times = np.unique(durations[event_observed == 1])
    event_times = event_times[event_times <= tau]

    # Walk the curve. ``surv`` is S(t) on the interval starting at the previous
    # breakpoint; integrate the rectangle up to the next event time (or tau).
    surv = 1.0
    prev_t = 0.0
    rmst = 0.0
    # Per-event-time bookkeeping for the variance: (d_i, n_i, t_i).
    steps: list[Tuple[float, float, float]] = []
    for t in event_times:
        # Rectangle [prev_t, t) carries the survival level *before* this drop.
        rmst += surv * (t - prev_t)
        n_at_risk = float(np.sum(durations >= t))
        d_events = float(np.sum((durations == t) & (event_observed == 1)))
        steps.append((d_events, n_at_risk, float(t)))
        if n_at_risk > 0:
            surv *= 1.0 - d_events / n_at_risk
        prev_t = float(t)
    # Final rectangle from the last event time (or 0) out to tau.
    rmst += surv * (tau - prev_t)

    # Integral-form variance: sum over event times of A_i^2 * d_i/(n_i(n_i-d_i))
    # where A_i is the area of the curve to the RIGHT of t_i (clipped at tau).
    # A_i = rmst - integral_{0}^{t_i} S(u) du.
    variance = 0.0
    area_left = 0.0
    surv = 1.0
    prev_t = 0.0
    for d_events, n_at_risk, t in steps:
        area_left += surv * (t - prev_t)  # integral of S on [0, t_i]
        area_right = rmst - area_left
        if n_at_risk > 0:
            surv *= 1.0 - d_events / n_at_risk
        prev_t = t
        denom = n_at_risk * (n_at_risk - d_events)
        if denom > 0:
            variance += (area_right**2) * d_events / denom

    return float(rmst), float(variance)


def rmst(
    durations: Sequence[float],
    event_observed: Sequence[bool],
    tau: float,
) -> RMSTResult:
    """Restricted mean survival time on ``[0, tau]`` from Kaplan-Meier.

    Parameters
    ----------
    durations:
        Observed follow-up time per subject (event or censoring time).
    event_observed:
        1/True if the event was observed, 0/False if right-censored.
    tau:
        The horizon. RMST is the area under the KM curve on ``[0, tau]``; it is
        always in ``[0, tau]``.

    Both the point estimate and the SE come from the pure-numpy KM integral;
    the SE is the classic integral-form (sampling) variance, which is what a CI
    requires.
    """

    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be positive")
    d, e = _clean_inputs(durations, event_observed)

    point, variance = _km_integral(d, e, tau)
    se = float(np.sqrt(max(variance, 0.0)))

    z = float(stats.norm.ppf(0.975))
    half = z * se
    ci_low = max(0.0, point - half)
    ci_high = min(tau, point + half)
    return RMSTResult(
        rmst=point,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        tau=tau,
    )


def rmst_difference(
    durations: Sequence[float],
    event_observed: Sequence[bool],
    groups: Sequence[object],
    tau: float,
) -> Dict[str, object]:
    """Between-group RMST difference at horizon ``tau``.

    ``groups`` must take exactly two distinct values; the two sorted labels are
    group *a* and group *b* and the reported ``diff`` is ``rmst_a - rmst_b``.
    The difference of two independent KM-plug-in RMSTs has variance
    ``se_a^2 + se_b^2``; the p-value is a two-sided normal-approximation test of
    ``diff == 0``.

    Returns a dict with ``rmst_a``, ``rmst_b``, ``diff``, ``se_diff``,
    ``ci`` (a ``(low, high)`` tuple), ``p_value``, ``tau``, and the two group
    labels ``group_a`` / ``group_b``.
    """

    tau = float(tau)
    d = np.asarray(list(durations), dtype=float)
    e = np.asarray(list(event_observed), dtype=float)
    g = np.asarray(list(groups), dtype=object)
    if not (d.shape[0] == e.shape[0] == g.shape[0]):
        raise ValueError("durations, event_observed and groups must be equal length")

    labels = sorted({x for x in g.tolist()}, key=repr)
    if len(labels) != 2:
        raise ValueError(
            f"rmst_difference needs exactly two groups, got {len(labels)}"
        )
    label_a, label_b = labels[0], labels[1]

    mask_a = g == label_a
    mask_b = g == label_b
    res_a = rmst(d[mask_a], e[mask_a], tau)
    res_b = rmst(d[mask_b], e[mask_b], tau)

    diff = res_a.rmst - res_b.rmst
    se_diff = float(np.sqrt(res_a.se**2 + res_b.se**2))
    if se_diff > 0:
        z = diff / se_diff
        p_value = float(2.0 * stats.norm.sf(abs(z)))
    else:
        # No sampling variability (e.g. no censoring, identical groups): the
        # difference is exact. p is 1.0 unless the point estimates differ.
        p_value = 1.0 if diff == 0.0 else 0.0
    z_crit = float(stats.norm.ppf(0.975))
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    return {
        "rmst_a": res_a.rmst,
        "rmst_b": res_b.rmst,
        "diff": diff,
        "se_diff": se_diff,
        "ci": ci,
        "p_value": p_value,
        "tau": tau,
        "group_a": label_a,
        "group_b": label_b,
    }


__all__ = [
    "RMSTResult",
    "rmst",
    "rmst_difference",
]
