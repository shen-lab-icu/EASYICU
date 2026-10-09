"""What a clone-censor-weight estimate shows about its own weights and clones.

The estimate (:mod:`.clone_censor_weight`) gives the risks; these functions
read the same fit and report what a reviewer needs to judge them, each
computed one way:

* the weights each arm carries past the grace period, before and after
  truncation, with their Kish effective sample size;
* positivity: each eligible stay's modelled probability of starting in the
  grace period, by whether it started, and the share outside the window;
* balance: the two arms' clones are identical at time zero, so each arm's
  clones still uncensored at the end of the grace period (deaths in it
  included) are compared, unweighted and weighted, with the eligible
  population, on the eligible population's unweighted standard deviation;
* adherence: the clones, the artificial censoring by grace-period hour, the
  deaths in the grace period and the deaths by the horizon of each arm, and
  the starts recorded after a stay's death or ICU exit, which are no starts;
* the E-value of the risk ratio.

Nothing here changes an estimate or decides a stop; the executor compares
these numbers with the host's thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

from ..contracts.target_trial_design import TARGET_TRIAL_HOST_POLICY
from .clone_censor_weight import (
    ARMS,
    DEFER,
    INITIATE,
    ArmEstimate,
    CloneCensorWeightEstimate,
)
from .sensitivity import EValueResult, compute_e_value

__all__ = [
    "AdherenceSummary",
    "BalanceRow",
    "PositivitySummary",
    "WeightSummary",
    "adherence_summaries",
    "covariate_balance",
    "grace_icu_exit_share",
    "late_start_count",
    "positivity_summary",
    "risk_ratio_e_value",
    "weight_summaries",
]


@dataclass(frozen=True)
class WeightSummary:
    """The weights one arm carries past the grace period."""

    arm: str
    truncated: bool
    n: int
    mean: float
    sd: float
    minimum: float
    p01: float
    p50: float
    p99: float
    maximum: float
    #: Kish effective sample size, (sum w)^2 / sum w^2.
    ess: float
    n_clipped_low: int
    n_clipped_high: int

    @property
    def ess_share(self) -> float:
        return self.ess / self.n


def _summary(arm: ArmEstimate, *, truncated: bool) -> WeightSummary:
    raw = arm.grace_end_weight[arm.followed_past_grace]
    low, high = arm.truncation_bounds
    weights = np.clip(raw, low, high) if truncated else raw
    p01, p50, p99 = np.percentile(weights, [1.0, 50.0, 99.0])
    return WeightSummary(
        arm=arm.arm,
        truncated=truncated,
        n=int(weights.shape[0]),
        mean=float(weights.mean()),
        sd=float(weights.std(ddof=1)) if weights.shape[0] > 1 else 0.0,
        minimum=float(weights.min()),
        p01=float(p01),
        p50=float(p50),
        p99=float(p99),
        maximum=float(weights.max()),
        ess=float(weights.sum() ** 2 / np.square(weights).sum()),
        n_clipped_low=int((raw < low).sum()) if truncated else 0,
        n_clipped_high=int((raw > high).sum()) if truncated else 0,
    )


def weight_summaries(estimate: CloneCensorWeightEstimate) -> tuple[WeightSummary, ...]:
    """Each arm's weights past the grace period, before then after truncation."""

    return tuple(
        _summary(estimate.arms[arm], truncated=truncated)
        for arm in ARMS
        for truncated in (False, True)
    )


@dataclass(frozen=True)
class PositivitySummary:
    """Modelled probabilities of starting in the grace period."""

    window: tuple[float, float]
    n: int
    n_outside: int
    #: Per observed group: n and the quantiles 0, 1, 25, 50, 75, 99, 100.
    started: tuple[float, ...]
    not_started: tuple[float, ...]
    n_started: int
    n_not_started: int

    @property
    def share_outside(self) -> float:
        return self.n_outside / self.n


_QUANTILES = (0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0)


def _quantiles(values: np.ndarray) -> tuple[float, ...]:
    if values.shape[0] == 0:
        return ()
    return tuple(float(value) for value in np.percentile(values, _QUANTILES))


def positivity_summary(estimate: CloneCensorWeightEstimate) -> PositivitySummary:
    low, high = TARGET_TRIAL_HOST_POLICY["positivity_window"]
    probability = estimate.start_probability
    started = estimate.course.started
    return PositivitySummary(
        window=(float(low), float(high)),
        n=int(probability.shape[0]),
        n_outside=int(((probability <= low) | (probability >= high)).sum()),
        started=_quantiles(probability[started]),
        not_started=_quantiles(probability[~started]),
        n_started=int(started.sum()),
        n_not_started=int((~started).sum()),
    )


@dataclass(frozen=True)
class BalanceRow:
    """One covariate of one arm against the eligible population."""

    covariate: str
    arm: str
    eligible_mean: float
    eligible_sd: float
    unweighted_mean: float
    weighted_mean: float
    smd_unweighted: Optional[float]
    smd_weighted: Optional[float]


def covariate_balance(estimate: CloneCensorWeightEstimate) -> tuple[BalanceRow, ...]:
    """Each arm's uncensored clones at the end of the grace period against time zero."""

    covariates = estimate.eligible.covariates
    rows: list[BalanceRow] = []
    for arm_name in ARMS:
        arm = estimate.arms[arm_name]
        kept = arm.uncensored_at_grace_end
        weights = arm.grace_end_weight[kept]
        for column, name in enumerate(estimate.eligible.covariate_names):
            population = covariates[:, column]
            values = population[kept]
            sd = float(np.std(population, ddof=1))
            mean = float(population.mean())
            unweighted = float(values.mean())
            weighted = float(DescrStatsW(values, weights=weights).mean)
            rows.append(
                BalanceRow(
                    covariate=name,
                    arm=arm_name,
                    eligible_mean=mean,
                    eligible_sd=sd,
                    unweighted_mean=unweighted,
                    weighted_mean=weighted,
                    smd_unweighted=(unweighted - mean) / sd if sd > 0 else None,
                    smd_weighted=(weighted - mean) / sd if sd > 0 else None,
                )
            )
    return tuple(rows)


@dataclass(frozen=True)
class AdherenceSummary:
    """One arm's clones and how the strategy censored them."""

    arm: str
    n_clones: int
    #: Clones artificially censored in each grace-period hour: at a start for
    #: ``defer``, at an ICU exit before a start for ``initiate``.
    censored_by_hour: tuple[int, ...]
    #: ``initiate`` clones still alive without a start at the end of the
    #: grace period, censored then.
    censored_at_grace_end: int
    deaths_in_grace: int
    followed_past_grace: int
    #: Unweighted deaths of uncensored clones by the horizon.
    events_by_horizon: int


def adherence_summaries(
    estimate: CloneCensorWeightEstimate,
) -> tuple[AdherenceSummary, ...]:
    course = estimate.course
    grace = estimate.timing.grace_period_hours
    censored = {
        DEFER: np.bincount(course.start_hour[course.started], minlength=grace),
        INITIATE: np.bincount(
            course.exit_hour[course.leaves_icu_before_start], minlength=grace
        ),
    }
    at_grace_end = {
        DEFER: 0,
        INITIATE: int(
            (
                ~course.started
                & ~course.leaves_icu_before_start
                & ~course.death_in_grace
            ).sum()
        ),
    }
    summaries = []
    for arm_name in ARMS:
        arm = estimate.arms[arm_name]
        summaries.append(
            AdherenceSummary(
                arm=arm_name,
                n_clones=arm.n_clones,
                censored_by_hour=tuple(int(count) for count in censored[arm_name]),
                censored_at_grace_end=at_grace_end[arm_name],
                deaths_in_grace=int(
                    (arm.uncensored_at_grace_end & ~arm.followed_past_grace).sum()
                ),
                followed_past_grace=arm.n_followed_past_grace,
                events_by_horizon=arm.n_events,
            )
        )
    return tuple(summaries)


def grace_icu_exit_share(estimate: CloneCensorWeightEstimate) -> float:
    """Eligible stays that leave the ICU in the grace period before starting."""

    leaves = estimate.course.leaves_icu_before_start
    return float(leaves.sum()) / float(leaves.shape[0])


def late_start_count(estimate: CloneCensorWeightEstimate) -> int:
    """Eligible stays whose recorded start in the grace period follows their death or exit."""

    return int(estimate.course.start_after_death_or_exit.sum())


def risk_ratio_e_value(
    risk_ratio: Optional[float], interval: Optional[tuple[float, float]]
) -> Optional[EValueResult]:
    """The E-value of the risk ratio and of its interval bound nearer the null."""

    if risk_ratio is None or not risk_ratio > 0:
        return None
    return compute_e_value(estimate=risk_ratio, ci=interval, estimate_type="rr")
