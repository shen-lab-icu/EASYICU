"""Clone-censor-weight estimation of a grace-period target trial.

A target trial with one time zero ``T0`` and a grace period of ``G`` hours
compares two strategies from ``T0``: start the treatment within
``[T0, T0 + G)`` (``initiate``) or do not start it within that window
(``defer``; unrestricted afterwards).  Every eligible stay is cloned into both
arms, a clone is censored when its stay deviates from its arm's strategy,
inverse probability of censoring weights restore the stays the censoring
removed, and a weighted Kaplan-Meier estimate per arm gives the risk of death
by the horizon ``H``.  This module owns that mechanism; the host owns its
constants (:mod:`..contracts.target_trial_design`) and the executor owns the
inputs, the stops and the products.

Clock and eligibility
---------------------
Every time is in hours since the stay's ICU admission.  A stay is eligible
when its endpoint is observed, it is alive and in the ICU at ``T0``
(death and ICU exit later than ``T0``) and no treatment start was recorded in
``[0, T0)``; a start at ``T0`` itself is a start in the grace period.  The
treatment-start column must have been captured from ICU admission through
``T0 + G``: a shorter capture would read a missed start as no start, so the
estimate stops instead.

Hours of the grace period and their order
-----------------------------------------
Hour ``k`` of the grace period is ``(T0 + k, T0 + k + 1]``; a start at ``T0``
falls in hour 0 and a start at ``T0 + G`` is not in the grace period.  Within
an hour, a start comes before a death and a death before an ICU exit; equal
times keep that order, except that a death the caller knows to come after the
ICU exit follows an exit at its time.  A start counts only when it is no
later than the stay's death and ICU exit.

Clones and censoring
--------------------
* ``defer``: censored at the start of the hour in which its stay starts.
  Leaving the ICU does not censor it: no start is recorded outside the ICU,
  and the capture reading the researcher confirms states it.
* ``initiate``: censored at the start of the hour in which its stay leaves
  the ICU before starting, and at ``T0 + G`` when it has not started.
* A death in the grace period before a start is compatible with both
  strategies and counts in both arms.  After ``T0 + G`` nothing censors a
  clone before the horizon.

Weights
-------
Two pooled discrete-time logistic models over the hours of the grace period
(statsmodels ``GLM`` binomial): the hazard of starting, among stays at risk of
starting (alive, in the ICU, not yet started), and the hazard of leaving the
ICU before a death, among stays that have not started.  Each has an hour term
(:func:`hour_terms`) and the baseline covariates; its numerator has the hour
term only, so the stabilising factor is common to every clone of an hour and
the Kaplan-Meier hazards do not depend on it.  The ICU-exit model drops the
covariates when its exits are too few for them and rare
(:func:`icu_exit_model_form`); the estimate records which form it fitted.

* ``defer`` in hour ``k``: the common probability of no start through ``k``
  over the stay's own probability over the hours it was at risk.
* ``initiate`` in hour ``k``: the same ratio for not leaving the ICU, frozen
  for a stay once it starts; from ``T0 + G`` a stay that started also carries
  the common probability of starting in the grace period over its own.
* Weights stay constant after ``T0 + G``.  The primary risks use them as
  estimated; a sensitivity analysis clips each arm's weights to the 1st and
  99th percentiles of the weights it carries past ``T0 + G``, another drops
  the weights.

A model that cannot be estimated -- a singular design, separation or no
convergence -- stops the estimate with that cause; nothing is refitted on a
smaller model.

Uncertainty
-----------
:func:`bootstrap_clone_censor_weight` resamples the units the caller names
(patients, or stays when no patient has two) with all their stays, so both
clones of a stay and every stay of a patient are drawn together, and refits
everything.

Evidence ceiling: ``analysis_only``.  The weights adjust for the baseline
covariates the caller supplies; exchangeability, positivity and consistency
are assumptions the researcher confirms, not findings of this module.

References
----------
Hernan MA, Robins JM. Am J Epidemiol 2016;183:758-764.
Cain LE, et al. Int J Biostat 2010;6(2):Article 18.
Hernan MA. BMJ 2018;360:k182.
Maringe C, et al. Int J Epidemiol 2020;49:1719-1729.
Cole SR, Hernan MA. Am J Epidemiol 2008;168:656-664.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..contracts.target_trial_design import (
    ICU_EXIT_MODEL_FORMS,
    MAX_GRACE_PERIOD_HOURS,
    MAX_TIME_ZERO_HOURS,
    MIN_TIME_ZERO_HOURS,
    TARGET_TRIAL_HOST_POLICY,
)
from .target_trial import EligibilityReport, TargetTrialError, reconcile_eligibility

__all__ = [
    "ARMS",
    "ArmEstimate",
    "CloneCensorWeightBootstrap",
    "CloneCensorWeightError",
    "CloneCensorWeightEstimate",
    "DEFER",
    "EffectIntervals",
    "EVIDENCE_CEILING",
    "HazardModelFit",
    "INITIATE",
    "TIME_ZERO_EXCLUSIONS",
    "TIME_ZERO_INCLUSIONS",
    "TrialCourse",
    "TrialStays",
    "TrialTiming",
    "bootstrap_clone_censor_weight",
    "estimate_clone_censor_weight",
    "hour_terms",
    "icu_exit_model_form",
    "resample_trial_units",
    "time_zero_eligibility",
    "trial_course",
    "trial_stays",
]

EVIDENCE_CEILING = TARGET_TRIAL_HOST_POLICY["evidence_ceiling"]
INITIATE = "initiate"
DEFER = "defer"
ARMS = (INITIATE, DEFER)

TIME_ZERO_INCLUSIONS = (
    "endpoint_observed",
    "alive_at_time_zero",
    "in_icu_at_time_zero",
)
TIME_ZERO_EXCLUSIONS = ("treatment_started_before_time_zero",)


class CloneCensorWeightError(ValueError):
    """The estimate refuses to proceed; ``code`` is stable, ``cause`` refines it."""

    def __init__(self, message: str, *, code: str, cause: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


# -- inputs -------------------------------------------------------------------


@dataclass(frozen=True)
class TrialTiming:
    """Time zero, grace period and horizon, in whole hours since ICU admission."""

    time_zero_hours: int
    grace_period_hours: int
    horizon_hours: int

    def __post_init__(self) -> None:
        for name in ("time_zero_hours", "grace_period_hours", "horizon_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise CloneCensorWeightError(
                    f"{name} must be a whole number of hours", code="ccw_timing_invalid"
                )
            object.__setattr__(self, name, int(value))
        if not MIN_TIME_ZERO_HOURS <= self.time_zero_hours <= MAX_TIME_ZERO_HOURS:
            raise CloneCensorWeightError(
                f"time zero {self.time_zero_hours} h is outside "
                f"[{MIN_TIME_ZERO_HOURS}, {MAX_TIME_ZERO_HOURS}]",
                code="ccw_timing_invalid",
            )
        if not 1 <= self.grace_period_hours <= MAX_GRACE_PERIOD_HOURS:
            raise CloneCensorWeightError(
                f"grace period {self.grace_period_hours} h is outside "
                f"[1, {MAX_GRACE_PERIOD_HOURS}]",
                code="ccw_timing_invalid",
            )
        if self.horizon_hours <= self.time_zero_hours + self.grace_period_hours:
            raise CloneCensorWeightError(
                "the horizon must end after the grace period", code="ccw_timing_invalid"
            )

    @property
    def follow_up_hours(self) -> int:
        """Hours from time zero to the horizon."""

        return self.horizon_hours - self.time_zero_hours


@dataclass(frozen=True)
class TrialStays:
    """One row per ICU stay; build it with :func:`trial_stays`."""

    stay_ids: np.ndarray
    group_ids: np.ndarray
    onset_hours: np.ndarray
    death_hours: np.ndarray
    icu_exit_hours: np.ndarray
    #: A death known to come after the ICU exit, so it follows an exit at
    #: its time; any other death at the time of an exit is a death in the ICU.
    death_follows_icu_exit: np.ndarray
    endpoint_observed: np.ndarray
    covariates: np.ndarray
    covariate_names: tuple[str, ...]
    #: ``[start, end)`` of the treatment-start column's capture, in hours.
    onset_window_hours: tuple[float, float]

    @property
    def n(self) -> int:
        return int(self.stay_ids.shape[0])

    def take(self, positions: np.ndarray) -> "TrialStays":
        index = np.asarray(positions, dtype=int)
        return TrialStays(
            stay_ids=self.stay_ids[index],
            group_ids=self.group_ids[index],
            onset_hours=self.onset_hours[index],
            death_hours=self.death_hours[index],
            icu_exit_hours=self.icu_exit_hours[index],
            death_follows_icu_exit=self.death_follows_icu_exit[index],
            endpoint_observed=self.endpoint_observed[index],
            covariates=self.covariates[index],
            covariate_names=self.covariate_names,
            onset_window_hours=self.onset_window_hours,
        )


def _hours(values: Any, *, name: str, n: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CloneCensorWeightError(
            f"{name} must be numeric hours", code="ccw_input_invalid"
        ) from exc
    if array.shape != (n,):
        raise CloneCensorWeightError(
            f"{name} has shape {array.shape}, expected ({n},)", code="ccw_input_invalid"
        )
    if np.isinf(array).any():
        raise CloneCensorWeightError(
            f"{name} holds an infinite time; an absent time is NaN",
            code="ccw_input_invalid",
        )
    return array


def trial_stays(
    *,
    stay_ids: Sequence[Any],
    group_ids: Sequence[Any],
    onset_hours: Sequence[float],
    death_hours: Sequence[float],
    icu_exit_hours: Sequence[float],
    endpoint_observed: Sequence[bool],
    covariates: Any,
    covariate_names: Sequence[str],
    onset_window_hours: tuple[float, float],
    death_follows_icu_exit: Optional[Sequence[bool]] = None,
) -> TrialStays:
    """Validate the per-stay inputs of one emulation.

    ``onset_hours`` is the first treatment start recorded in the capture
    window ``onset_window_hours`` (``[start, end)`` on the same clock, as the
    column's materialization states it), ``death_hours`` the death before the
    horizon and
    ``icu_exit_hours`` the ICU exit; NaN means none recorded (an unknown ICU
    exit makes the stay ineligible).  ``death_follows_icu_exit`` marks the
    deaths known to come after the ICU exit, such as one after hospital
    discharge known only by its day and placed at the exit; such a death is
    at or after a known exit, and none is marked by default.  ``group_ids``
    names each stay's resampling unit.  ``covariates`` is the complete
    numeric baseline design of the weight models, unmeasured states already
    coded by the caller.
    """

    ids = np.asarray(stay_ids, dtype=object)
    if ids.ndim != 1 or ids.shape[0] == 0:
        raise CloneCensorWeightError("no stay was given", code="ccw_input_invalid")
    n = int(ids.shape[0])
    if pd.isna(pd.Series(ids)).any() or pd.Series(ids).duplicated().any():
        raise CloneCensorWeightError(
            "stay ids must be present and unique", code="ccw_input_invalid"
        )
    groups = np.asarray(group_ids, dtype=object)
    if groups.shape != (n,) or pd.isna(pd.Series(groups)).any():
        raise CloneCensorWeightError(
            "every stay needs a resampling unit", code="ccw_input_invalid"
        )
    observed = np.asarray(endpoint_observed)
    if observed.shape != (n,) or observed.dtype != bool:
        raise CloneCensorWeightError(
            "endpoint_observed must be one boolean per stay", code="ccw_input_invalid"
        )
    names = tuple(str(name) for name in covariate_names)
    try:
        matrix = np.asarray(covariates, dtype=float).reshape(n, len(names))
    except (TypeError, ValueError) as exc:
        raise CloneCensorWeightError(
            f"covariates must be a numeric ({n}, {len(names)}) design",
            code="ccw_input_invalid",
        ) from exc
    if not np.isfinite(matrix).all():
        raise CloneCensorWeightError(
            "covariates must be complete and finite; code unmeasured states first",
            code="ccw_input_invalid",
        )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise CloneCensorWeightError(
            "covariate names must be unique and non-empty", code="ccw_input_invalid"
        )
    try:
        window = tuple(float(value) for value in onset_window_hours)
    except (TypeError, ValueError) as exc:
        raise CloneCensorWeightError(
            "the onset capture window must be two hours", code="ccw_input_invalid"
        ) from exc
    if len(window) != 2 or not (np.isfinite(window).all() and window[0] < window[1]):
        raise CloneCensorWeightError(
            "the onset capture window must be a finite [start, end)",
            code="ccw_input_invalid",
        )
    onset = _hours(onset_hours, name="onset_hours", n=n)
    with np.errstate(invalid="ignore"):
        outside = (onset < window[0]) | (onset >= window[1])
    if outside.any():
        raise CloneCensorWeightError(
            "a treatment start lies outside the window it was captured in",
            code="ccw_input_invalid",
        )
    death = _hours(death_hours, name="death_hours", n=n)
    exit_ = _hours(icu_exit_hours, name="icu_exit_hours", n=n)
    follows = (
        np.zeros(n, dtype=bool)
        if death_follows_icu_exit is None
        else np.asarray(death_follows_icu_exit)
    )
    if follows.shape != (n,) or follows.dtype != bool:
        raise CloneCensorWeightError(
            "death_follows_icu_exit must be one boolean per stay",
            code="ccw_input_invalid",
        )
    with np.errstate(invalid="ignore"):
        misplaced = follows & (np.isnan(death) | (death < exit_))
    if misplaced.any():
        raise CloneCensorWeightError(
            "a death known to follow the ICU exit has no time or precedes the exit",
            code="ccw_input_invalid",
        )
    return TrialStays(
        stay_ids=ids,
        group_ids=groups,
        onset_hours=onset,
        death_hours=death,
        icu_exit_hours=exit_,
        death_follows_icu_exit=follows.copy(),
        endpoint_observed=observed.copy(),
        covariates=matrix.copy(),
        covariate_names=names,
        onset_window_hours=(window[0], window[1]),
    )


def time_zero_eligibility(stays: TrialStays, timing: TrialTiming) -> EligibilityReport:
    """The host's time-zero rules as a denominator chain, in protocol order."""

    t0 = float(timing.time_zero_hours)
    with np.errstate(invalid="ignore"):
        inclusions = {
            "endpoint_observed": stays.endpoint_observed,
            "alive_at_time_zero": ~(stays.death_hours <= t0),
            "in_icu_at_time_zero": stays.icu_exit_hours > t0,
        }
        exclusions = {"treatment_started_before_time_zero": stays.onset_hours < t0}
    try:
        return reconcile_eligibility(
            pd.DataFrame(index=range(stays.n)),
            inclusions=inclusions,
            exclusions=exclusions,
        )
    except TargetTrialError as exc:
        raise CloneCensorWeightError(str(exc), code="ccw_no_eligible_stay") from exc


# -- the course of each eligible stay ----------------------------------------


def _grace_hour(times: np.ndarray) -> np.ndarray:
    """Hour ``k`` of ``(k, k + 1]``; a time of exactly 0 is in hour 0."""

    return np.maximum(np.ceil(times) - 1.0, 0.0).astype(int)


@dataclass(frozen=True)
class TrialCourse:
    """What each eligible stay did in the grace period, on the time-zero clock."""

    started: np.ndarray
    start_hour: np.ndarray
    death: np.ndarray
    death_in_grace: np.ndarray
    death_hour: np.ndarray
    leaves_icu_before_start: np.ndarray
    exit_hour: np.ndarray
    last_hour_at_risk: np.ndarray
    #: A start recorded in the grace period after the stay's death or ICU
    #: exit: not a start, and reported rather than dropped silently.
    start_after_death_or_exit: np.ndarray

    @property
    def last_hour_without_start(self) -> np.ndarray:
        """The last hour a stay was at risk and did not start (-1: none)."""

        return np.where(self.started, self.start_hour - 1, self.last_hour_at_risk)


def trial_course(stays: TrialStays, timing: TrialTiming) -> TrialCourse:
    """Code starts, deaths and ICU exits of eligible stays by grace-period hour."""

    t0 = float(timing.time_zero_hours)
    grace = timing.grace_period_hours
    onset = np.nan_to_num(stays.onset_hours - t0, nan=np.inf)
    death = np.nan_to_num(stays.death_hours - t0, nan=np.inf)
    exit_ = stays.icu_exit_hours - t0
    if not (np.all(death > 0) and np.all(exit_ > 0) and np.all(onset >= 0)):
        raise CloneCensorWeightError(
            "a stay that is not eligible at time zero reached the estimate",
            code="ccw_input_invalid",
        )
    started = (onset < grace) & (onset <= death) & (onset <= exit_)
    start_hour = np.where(started, _grace_hour(np.where(started, onset, 0.0)), -1)
    death_in_grace = death <= grace
    death_hour = np.where(
        death_in_grace, _grace_hour(np.where(death_in_grace, death, 1.0)), -1
    )
    exit_in_grace = exit_ <= grace
    any_exit_hour = np.where(
        exit_in_grace, _grace_hour(np.where(exit_in_grace, exit_, 1.0)), grace - 1
    )
    # A death at the time of the exit is a death in the ICU, unless it is
    # known to follow the exit.
    leaves = ~started & exit_in_grace & ((exit_ < death) | stays.death_follows_icu_exit)
    end = np.minimum(np.where(death_in_grace, death_hour, grace - 1), any_exit_hour)
    return TrialCourse(
        started=started,
        start_hour=start_hour,
        death=death,
        death_in_grace=death_in_grace,
        death_hour=death_hour,
        leaves_icu_before_start=leaves,
        exit_hour=np.where(leaves, any_exit_hour, -1),
        last_hour_at_risk=np.where(started, start_hour, end),
        start_after_death_or_exit=(onset < grace) & ~started,
    )


# -- the weight models --------------------------------------------------------


def hour_terms(hours: np.ndarray, grace_period_hours: int) -> np.ndarray:
    """The hour columns of a weight model for hours ``0 .. G - 1``.

    No column for a one-hour grace period, a linear term below the spline minimum,
    otherwise a restricted cubic spline with knots at fixed fractions of the
    grace period (Harrell's form, scaled by the squared outer knot distance).
    """

    k = np.asarray(hours, dtype=float)
    if grace_period_hours < 2:
        return np.empty((k.shape[0], 0))
    if grace_period_hours < TARGET_TRIAL_HOST_POLICY["hour_spline_min_grace_hours"]:
        return k[:, None]
    t1, t2, t3 = (
        fraction * (grace_period_hours - 1)
        for fraction in TARGET_TRIAL_HOST_POLICY["hour_spline_knot_fractions"]
    )

    def cube(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0.0, None) ** 3

    nonlinear = (
        cube(k - t1)
        - cube(k - t2) * (t3 - t1) / (t3 - t2)
        + cube(k - t3) * (t2 - t1) / (t3 - t2)
    ) / (t3 - t1) ** 2
    return np.column_stack([k, nonlinear])


def _hour_term_names(grace_period_hours: int) -> tuple[str, ...]:
    width = hour_terms(np.zeros(1), grace_period_hours).shape[1]
    return ("hour", "hour_spline")[:width]


@dataclass(frozen=True)
class HazardModelFit:
    """One fitted discrete-time hazard model, numerator or denominator."""

    name: str
    terms: tuple[str, ...]
    params: tuple[float, ...]
    n_rows: int
    n_events: int
    #: IRLS iterations to convergence; a model that did not converge stops.
    iterations: int

    @property
    def n_parameters(self) -> int:
        return len(self.terms)

    @property
    def events_per_parameter(self) -> float:
        return self.n_events / self.n_parameters


def _not_estimable(name: str, cause: str) -> CloneCensorWeightError:
    return CloneCensorWeightError(
        f"the {name} model cannot be estimated ({cause})",
        code="ccw_weight_model_not_estimable",
        cause=cause,
    )


def _fit_hazard(
    name: str, exog: np.ndarray, endog: np.ndarray, terms: tuple[str, ...]
) -> HazardModelFit:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import (
        PerfectSeparationError,
        PerfectSeparationWarning,
    )

    n_events = int(endog.sum())
    if n_events in {0, int(endog.shape[0])}:
        raise _not_estimable(name, "separation")
    # Accelerate's matmul can raise spurious floating-point flags; the
    # results are checked instead.
    with np.errstate(all="ignore"):
        gram = exog.T @ exog
    if not np.isfinite(gram).all():
        raise _not_estimable(name, "not_converged")
    if np.linalg.matrix_rank(gram, hermitian=True) < exog.shape[1]:
        raise _not_estimable(name, "singular_design")
    bound = TARGET_TRIAL_HOST_POLICY["separation_probability_bound"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = sm.GLM(endog, exog, family=sm.families.Binomial()).fit(
                method="IRLS",
                maxiter=TARGET_TRIAL_HOST_POLICY["glm_max_iterations"],
                tol=TARGET_TRIAL_HOST_POLICY["glm_tolerance"],
            )
        except PerfectSeparationError as exc:
            raise _not_estimable(name, "separation") from exc
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise _not_estimable(name, "not_converged") from exc
    if any(issubclass(item.category, PerfectSeparationWarning) for item in caught):
        raise _not_estimable(name, "separation")
    params = np.asarray(result.params, dtype=float)
    if not bool(result.converged) or not np.isfinite(params).all():
        raise _not_estimable(name, "not_converged")
    mu = np.asarray(result.mu, dtype=float)
    if float(mu.min()) < bound or float(mu.max()) > 1.0 - bound:
        raise _not_estimable(name, "separation")
    return HazardModelFit(
        name=name,
        terms=terms,
        params=tuple(float(value) for value in params),
        n_rows=int(endog.shape[0]),
        n_events=n_events,
        iterations=int(result.fit_history.get("iteration", 0)),
    )


def _person_hours(last_hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stay position and hour of every stay-hour ``0 .. last_hour``."""

    counts = np.maximum(last_hour + 1, 0)
    stay = np.repeat(np.arange(counts.shape[0]), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    return stay, np.arange(stay.shape[0]) - first


@dataclass(frozen=True)
class _HazardPair:
    denominator: HazardModelFit
    numerator: HazardModelFit
    #: Per stay and grace hour, the modelled hazard; per hour, the common one.
    hazard: np.ndarray
    common: np.ndarray


def _fit_pair(
    name: str,
    stays: TrialStays,
    grace: int,
    last_hour: np.ndarray,
    event_hour: np.ndarray,
    *,
    adjust: bool = True,
) -> _HazardPair:
    stay, hour = _person_hours(last_hour)
    endog = (hour == event_hour[stay]).astype(float)
    basis = hour_terms(hour, grace)
    hour_names = _hour_term_names(grace)
    intercept = np.ones((stay.shape[0], 1))
    numerator = _fit_hazard(
        f"{name}_numerator",
        np.hstack([intercept, basis]),
        endog,
        ("intercept", *hour_names),
    )
    denominator = numerator
    if adjust:
        denominator = _fit_hazard(
            name,
            np.hstack([intercept, basis, stays.covariates[stay]]),
            endog,
            ("intercept", *hour_names, *stays.covariate_names),
        )
    grid = hour_terms(np.arange(grace), grace)
    q = grid.shape[1]
    den = np.asarray(denominator.params)
    num = np.asarray(numerator.params)
    with np.errstate(all="ignore"):
        covariate_part = (
            stays.covariates @ den[1 + q :] if adjust else np.zeros(stays.n)
        )
        linear = den[0] + (grid @ den[1 : 1 + q])[None, :] + covariate_part[:, None]
        common = 1.0 / (1.0 + np.exp(-(num[0] + grid @ num[1:])))
    if not (np.isfinite(linear).all() and np.isfinite(common).all()):
        raise _not_estimable(name, "not_converged")
    return _HazardPair(
        denominator=denominator,
        numerator=numerator,
        hazard=1.0 / (1.0 + np.exp(-linear)),
        common=common,
    )


def icu_exit_model_form(n_exits: int, n_stays: int, n_parameters: int) -> str:
    """The ICU-exit model a grace period with ``n_exits`` exits before a start fits.

    ``n_parameters`` counts the model with its covariates.  Fewer than the
    host's minimum exits per parameter, among at most its share of the
    ``n_stays`` eligible stays, leave the hour terms only.
    """

    if n_exits == 0:
        return ICU_EXIT_MODEL_FORMS[0]
    minimum = TARGET_TRIAL_HOST_POLICY["icu_exit_model_min_events_per_parameter"]
    share = TARGET_TRIAL_HOST_POLICY["icu_exit_model_hour_only_max_share"]
    if n_exits < minimum * n_parameters and n_exits <= share * n_stays:
        return ICU_EXIT_MODEL_FORMS[2]
    return ICU_EXIT_MODEL_FORMS[1]


def _survival_through(hazard: np.ndarray, last_hour: np.ndarray) -> np.ndarray:
    """Per stay and hour ``k``: the product of ``1 - hazard`` over hours ``<= min(k, last)``."""

    hours = np.arange(hazard.shape[1])[None, :]
    log_terms = np.where(hours <= last_hour[:, None], np.log1p(-hazard), 0.0)
    return np.exp(np.cumsum(log_terms, axis=1))


# -- arms and risks -----------------------------------------------------------


@dataclass(frozen=True)
class ArmEstimate:
    """One arm: its clones, the weights they carry and its risk by the horizon."""

    arm: str
    #: Counting-process rows on the time-zero clock: (entry, stop] per hour of
    #: the grace period, then one row past it.
    row_stay: np.ndarray
    row_entry: np.ndarray
    row_stop: np.ndarray
    row_event: np.ndarray
    row_weight: np.ndarray
    row_weight_truncated: np.ndarray
    #: Per clone: not censored by ``T0 + G`` (followed past it, or died in
    #: the grace period), and the weight it carries at that point.
    uncensored_at_grace_end: np.ndarray
    grace_end_weight: np.ndarray
    followed_past_grace: np.ndarray
    truncation_bounds: tuple[float, float]
    #: By the horizon: with the weights, with the truncated weights, unweighted.
    risk: float
    risk_truncated: float
    risk_crude: float
    #: Unweighted deaths of uncensored clones by the horizon.
    n_events: int
    curve_hours: np.ndarray
    curve_risk: np.ndarray

    @property
    def n_clones(self) -> int:
        return int(self.uncensored_at_grace_end.shape[0])

    @property
    def n_followed_past_grace(self) -> int:
        return int(self.followed_past_grace.sum())


def _km(
    entry, stop, event, weight, follow_up: float
) -> tuple[float, np.ndarray, np.ndarray]:
    from lifelines import KaplanMeierFitter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter = KaplanMeierFitter().fit(
            stop, event_observed=event, entry=entry, weights=weight
        )
    survival = fitter.survival_function_.iloc[:, 0]
    times = survival.index.to_numpy(dtype=float)
    keep = times <= follow_up
    at_horizon = float(survival.to_numpy()[keep][-1]) if keep.any() else 1.0
    return 1.0 - at_horizon, times[keep], 1.0 - survival.to_numpy(dtype=float)[keep]


def _arm(
    arm: str,
    course: TrialCourse,
    timing: TrialTiming,
    grace_weights: np.ndarray,
    post_weights: np.ndarray,
    *,
    grace_rows: np.ndarray,
    death_row: np.ndarray,
    followed: np.ndarray,
    secondary: bool,
) -> ArmEstimate:
    grace = timing.grace_period_hours
    follow_up = float(timing.follow_up_hours)
    if not followed.any():
        raise CloneCensorWeightError(
            f"no {arm} clone is followed past the grace period",
            code="ccw_strategy_unobserved",
        )
    stay, hour = _person_hours(grace_rows - 1)
    last = hour == grace_rows[stay] - 1
    dies_here = last & death_row[stay]
    entry = hour.astype(float)
    stop = np.where(dies_here, course.death[stay], hour + 1.0)
    event = dies_here
    weight = grace_weights[stay, hour]
    post = np.flatnonzero(followed)
    entry = np.concatenate([entry, np.full(post.shape[0], float(grace))])
    stop = np.concatenate([stop, np.minimum(course.death[post], follow_up)])
    event = np.concatenate([event, course.death[post] <= follow_up])
    weight = np.concatenate([weight, post_weights[post]])
    stay = np.concatenate([stay, post])

    low_pct, high_pct = TARGET_TRIAL_HOST_POLICY["weight_truncation_percentiles"]
    low, high = (
        float(value) for value in np.percentile(post_weights[post], [low_pct, high_pct])
    )
    truncated = np.clip(weight, low, high)
    risk, curve_hours, curve_risk = _km(entry, stop, event, weight, follow_up)
    risk_truncated = _km(entry, stop, event, truncated, follow_up)[0]
    # A resample needs no unweighted risk.
    risk_crude = float("nan")
    if secondary:
        risk_crude = _km(entry, stop, event, np.ones_like(weight), follow_up)[0]

    uncensored = followed | death_row
    end_weight = np.full(course.started.shape[0], np.nan)
    end_weight[post] = post_weights[post]
    dead = np.flatnonzero(death_row)
    end_weight[dead] = grace_weights[dead, course.death_hour[dead]]
    return ArmEstimate(
        arm=arm,
        row_stay=stay,
        row_entry=entry,
        row_stop=stop,
        row_event=event,
        row_weight=weight,
        row_weight_truncated=truncated,
        uncensored_at_grace_end=uncensored,
        grace_end_weight=end_weight,
        followed_past_grace=followed,
        truncation_bounds=(low, high),
        risk=risk,
        risk_truncated=risk_truncated,
        risk_crude=risk_crude,
        n_events=int(event.sum()),
        curve_hours=curve_hours,
        curve_risk=curve_risk,
    )


@dataclass(frozen=True)
class CloneCensorWeightEstimate:
    """The point estimate of one emulation and what its diagnostics read."""

    timing: TrialTiming
    #: The time-zero denominator chain; a bootstrap resample has none.
    eligibility: Optional[EligibilityReport]
    eligible: TrialStays
    course: TrialCourse
    initiation_model: HazardModelFit
    initiation_numerator: HazardModelFit
    icu_exit_model: Optional[HazardModelFit]
    icu_exit_numerator: Optional[HazardModelFit]
    #: One of the host's ICU-exit model forms (``none`` without exits).
    icu_exit_model_form: str
    #: Per eligible stay, the modelled probability of starting in the grace period.
    start_probability: np.ndarray
    arms: Mapping[str, ArmEstimate]

    @property
    def risk_difference(self) -> float:
        return self.arms[INITIATE].risk - self.arms[DEFER].risk

    @property
    def risk_ratio(self) -> Optional[float]:
        return _ratio(self.arms[INITIATE].risk, self.arms[DEFER].risk)

    @property
    def evidence_ceiling(self) -> str:
        return EVIDENCE_CEILING


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator > 0 else None


def _estimate_eligible(
    eligible: TrialStays,
    timing: TrialTiming,
    eligibility: Optional[EligibilityReport],
    *,
    secondary: bool = True,
) -> CloneCensorWeightEstimate:
    grace = timing.grace_period_hours
    course = trial_course(eligible, timing)
    if not course.started.any():
        raise CloneCensorWeightError(
            "no eligible stay starts the treatment in the grace period",
            code="ccw_strategy_unobserved",
        )
    without_start = course.last_hour_without_start
    starting = _fit_pair(
        "initiation", eligible, grace, course.last_hour_at_risk, course.start_hour
    )
    defer_grace = np.cumprod(1.0 - starting.common)[None, :] / _survival_through(
        starting.hazard, without_start
    )
    start_probability = 1.0 - np.prod(1.0 - starting.hazard, axis=1)
    common_start = 1.0 - float(np.prod(1.0 - starting.common))

    leaving: Optional[_HazardPair] = None
    initiate_grace = np.ones((eligible.n, grace))
    leaving_form = icu_exit_model_form(
        int(course.leaves_icu_before_start.sum()),
        eligible.n,
        1 + len(_hour_term_names(grace)) + len(eligible.covariate_names),
    )
    if leaving_form != ICU_EXIT_MODEL_FORMS[0]:
        leaving = _fit_pair(
            "icu_exit",
            eligible,
            grace,
            without_start,
            course.exit_hour,
            adjust=leaving_form == ICU_EXIT_MODEL_FORMS[1],
        )
        initiate_grace = np.cumprod(1.0 - leaving.common)[None, :] / _survival_through(
            leaving.hazard, without_start
        )

    alive_at_grace_end = ~course.death_in_grace
    defer_followed = ~course.started & alive_at_grace_end
    initiate_followed = course.started & alive_at_grace_end
    defer = _arm(
        DEFER,
        course,
        timing,
        defer_grace,
        defer_grace[:, grace - 1],
        grace_rows=np.where(
            course.started,
            course.start_hour,
            np.where(course.death_in_grace, course.death_hour + 1, grace),
        ),
        death_row=~course.started & course.death_in_grace,
        followed=defer_followed,
        secondary=secondary,
    )
    initiate_dies = course.death_in_grace & ~course.leaves_icu_before_start
    initiate = _arm(
        INITIATE,
        course,
        timing,
        initiate_grace,
        initiate_grace[:, grace - 1] * common_start / start_probability,
        grace_rows=np.where(
            course.leaves_icu_before_start,
            course.exit_hour,
            np.where(initiate_dies, course.death_hour + 1, grace),
        ),
        death_row=initiate_dies,
        followed=initiate_followed,
        secondary=secondary,
    )
    return CloneCensorWeightEstimate(
        timing=timing,
        eligibility=eligibility,
        eligible=eligible,
        course=course,
        initiation_model=starting.denominator,
        initiation_numerator=starting.numerator,
        icu_exit_model=leaving.denominator if leaving else None,
        icu_exit_numerator=leaving.numerator if leaving else None,
        icu_exit_model_form=leaving_form,
        start_probability=start_probability,
        arms={INITIATE: initiate, DEFER: defer},
    )


def estimate_clone_censor_weight(
    stays: TrialStays, timing: TrialTiming
) -> CloneCensorWeightEstimate:
    """Apply the time-zero rules, clone, censor, weight and estimate both risks."""

    start, end = stays.onset_window_hours
    if start > 0.0 or end < timing.time_zero_hours + timing.grace_period_hours:
        raise CloneCensorWeightError(
            f"treatment starts were captured over [{start:g}, {end:g}) h, which "
            "does not span ICU admission to the end of the grace period",
            code="ccw_onset_window_short",
        )
    eligibility = time_zero_eligibility(stays, timing)
    eligible = stays.take(np.asarray(eligibility.analysis_positions, dtype=int))
    return _estimate_eligible(eligible, timing, eligibility)


# -- uncertainty --------------------------------------------------------------


def resample_trial_units(group_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stay positions of one bootstrap resample.

    Units are drawn with replacement, as many as there are; a drawn unit
    brings every one of its stays, each as often as the unit is drawn.
    """

    codes, uniques = pd.factorize(pd.Series(group_ids, dtype=object), sort=True)
    counts = np.bincount(codes, minlength=len(uniques))
    order = np.argsort(codes, kind="stable")
    starts = np.cumsum(counts) - counts
    drawn = rng.integers(0, len(uniques), size=len(uniques))
    sizes = counts[drawn]
    first = np.repeat(np.cumsum(sizes) - sizes, sizes)
    return order[np.repeat(starts[drawn], sizes) + np.arange(sizes.sum()) - first]


@dataclass(frozen=True)
class EffectIntervals:
    """Percentile intervals of both risks, their difference and their ratio."""

    risk: Mapping[str, tuple[float, float]]
    risk_difference: Optional[tuple[float, float]]
    risk_ratio: Optional[tuple[float, float]]


def _interval(values: np.ndarray, level: float) -> Optional[tuple[float, float]]:
    finite = values[np.isfinite(values)]
    if finite.shape[0] == 0:
        return None
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(finite, [tail, 100.0 - tail])
    return float(low), float(high)


def _intervals(table: np.ndarray, level: float) -> EffectIntervals:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(table[:, 1] > 0, table[:, 0] / table[:, 1], np.nan)
    risk = {arm: _interval(table[:, column], level) for column, arm in enumerate(ARMS)}
    return EffectIntervals(
        risk={arm: value for arm, value in risk.items() if value is not None},
        risk_difference=_interval(table[:, 0] - table[:, 1], level),
        risk_ratio=_interval(ratios, level),
    )


@dataclass(frozen=True)
class CloneCensorWeightBootstrap:
    """Percentile intervals from resamples refitted end to end."""

    resamples: int
    seed: int
    confidence_level: float
    #: Per successful resample, the risks of :data:`ARMS` in that order: with
    #: the weights, and with the truncated weights.
    replicate_risks: np.ndarray
    replicate_risks_truncated: np.ndarray
    #: Failed resamples by error code and cause.
    failures: Mapping[str, int]
    intervals: EffectIntervals
    truncated_intervals: EffectIntervals

    @property
    def n_failed(self) -> int:
        return int(sum(self.failures.values()))

    @property
    def failure_share(self) -> float:
        return self.n_failed / self.resamples


def bootstrap_clone_censor_weight(
    estimate: CloneCensorWeightEstimate,
    *,
    resamples: Optional[int] = None,
    seed: Optional[int] = None,
) -> CloneCensorWeightBootstrap:
    """Resample the eligible stays by unit and refit the whole estimate each time."""

    count = int(resamples or TARGET_TRIAL_HOST_POLICY["bootstrap_resamples"])
    used_seed = int(
        TARGET_TRIAL_HOST_POLICY["bootstrap_seed"] if seed is None else seed
    )
    level = float(TARGET_TRIAL_HOST_POLICY["confidence_level"])
    rng = np.random.default_rng(used_seed)
    risks: list[tuple[float, ...]] = []
    failures: dict[str, int] = {}
    for _ in range(count):
        positions = resample_trial_units(estimate.eligible.group_ids, rng)
        try:
            replicate = _estimate_eligible(
                estimate.eligible.take(positions),
                estimate.timing,
                None,
                secondary=False,
            )
        except CloneCensorWeightError as exc:
            key = exc.code if exc.cause is None else f"{exc.code}:{exc.cause}"
            failures[key] = failures.get(key, 0) + 1
            continue
        arms = [replicate.arms[arm] for arm in ARMS]
        risks.append(
            (*(arm.risk for arm in arms), *(arm.risk_truncated for arm in arms))
        )
    table = np.asarray(risks, dtype=float).reshape(-1, 4)
    return CloneCensorWeightBootstrap(
        resamples=count,
        seed=used_seed,
        confidence_level=level,
        replicate_risks=table[:, :2],
        replicate_risks_truncated=table[:, 2:],
        failures=dict(sorted(failures.items())),
        intervals=_intervals(table[:, :2], level),
        truncated_intervals=_intervals(table[:, 2:], level),
    )
