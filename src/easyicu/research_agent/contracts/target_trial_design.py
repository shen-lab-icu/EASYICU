"""The host policy of a grace-period target trial emulated by clone-censor-weight.

Owner
-----
A study states its trial -- the treatment, the time zero ``T0``, the grace
period ``G`` and the fixed-horizon death endpoint -- and the host compiles it
(the planning owner of the trial spec).  Everything else the emulation needs
has exactly one implementation, so it is a constant here rather than a field
someone fills in: the bounds of the time zero and grace period the host
offers, how the hours of the grace period are binned and ordered, the form of
the weight models, the weight truncation, the bootstrap and its seed, the
positivity window and the stop thresholds.  The estimator and its executor
read these constants, and the plan review reads them too, so a reviewed
design and the executed one cannot drift apart.

A signed trial binds :func:`target_trial_host_policy_sha256`, the digest of
every constant here: a trial reviewed under one policy does not execute
under another.

This module never reads patient data and imports nothing of the host but
the canonical JSON digest.
"""

from __future__ import annotations

from types import MappingProxyType

from ..canonical_json import canonical_sha256

__all__ = [
    "BOOTSTRAP_UNSTABLE_CAUSES",
    "MAX_GRACE_PERIOD_HOURS",
    "MAX_TIME_ZERO_HOURS",
    "MIN_TIME_ZERO_HOURS",
    "TARGET_TRIAL_DESIGN_SCHEMA_VERSION",
    "TARGET_TRIAL_HOST_POLICY",
    "TARGET_TRIAL_STOP_REASONS",
    "TARGET_TRIAL_STOP_THRESHOLDS",
    "ICU_EXIT_MODEL_FORMS",
    "WEIGHT_MODEL_NOT_ESTIMABLE_CAUSES",
    "target_trial_host_policy_sha256",
]

TARGET_TRIAL_DESIGN_SCHEMA_VERSION = "easyicu.target_trial_design/1"

#: A treatment start in ``[0, T0)`` marks a prevalent user, so the earliest
#: time zero leaves one hour of ICU stay to see it in.
MIN_TIME_ZERO_HOURS = 1
MAX_TIME_ZERO_HOURS = 72
#: Within the grace period the start of the treatment is adjusted for baseline
#: covariates only, so the host keeps it short.
MAX_GRACE_PERIOD_HOURS = 24

TARGET_TRIAL_HOST_POLICY = MappingProxyType(
    {
        "method": "clone_censor_weight",
        "time_axis": "hours_since_icu_admission",
        # Hour k of the grace period is (T0 + k, T0 + k + 1]; a start at T0
        # itself falls in hour 0.  Death and ICU exit times are read as given.
        "grace_hour_bins": "right_closed_hours_after_time_zero",
        # Within one hour a start comes first, then a death, then an ICU exit.
        # Equal times keep that order: a start at the time of a death or of an
        # exit is a start, and a death at the time of an exit is a death,
        # unless it is known only by its day (below).
        "same_hour_order": ("treatment_start", "death", "icu_exit"),
        # A death without a recorded time is one after hospital discharge,
        # known by its calendar day: it is placed at that day, or at the ICU
        # exit when the day precedes it, and it follows an exit it ties with.
        # A day that ends before the ICU exit contradicts the stay.
        "death_known_by_day": "later_of_its_day_and_the_icu_exit_after_the_exit",
        # The initiating strategy's clone is censored when its stay leaves the
        # ICU in the grace period before starting, and at T0 + G when it has
        # not started; the deferring strategy's clone when its stay starts.
        "initiation_model": "pooled_discrete_time_logistic_glm_binomial",
        "icu_exit_model": "pooled_discrete_time_logistic_glm_binomial",
        # The ICU-exit model adjusts for the covariates.  When its exits are
        # too few for them (under 10 per parameter) and leave at most 2% of
        # the eligible stays, it has the hour terms only: an exit is then
        # taken to depend on time alone, and the estimate says so.  A sparse
        # exit model would otherwise separate and stop an estimate that
        # rare exits barely move.
        "icu_exit_model_min_events_per_parameter": 10,
        "icu_exit_model_hour_only_max_share": 0.02,
        "hour_terms": "restricted_cubic_spline",
        "hour_spline_knot_fractions": (0.1, 0.5, 0.9),
        # Shorter grace periods carry a linear hour term (two hours or more)
        # or none (one hour).
        "hour_spline_min_grace_hours": 4,
        "weight_numerator": "hour_terms_only",
        "weight_after_grace": "constant",
        # The primary risks use the stabilised weights as estimated.  Clipping
        # them trades bias for variance, and under confounding by indication
        # the clipped weights are those of the stays the weights exist for, so
        # the truncated risks are a sensitivity analysis beside the unweighted
        # ones; the extreme-weight stop reads the weights before truncation.
        "primary_weights": "stabilized_untruncated",
        "sensitivity_weights": ("stabilized_truncated", "unweighted"),
        "weight_truncation_percentiles": (1.0, 99.0),
        # The percentiles are those of the weights each arm carries past the
        # grace period; every weight of the arm is clipped to them.
        "weight_truncation_reference": "post_grace_weights_of_the_arm",
        "risk_estimator": "weighted_kaplan_meier_lifelines",
        "bootstrap_resamples": 500,
        "bootstrap_seed": 20261009,
        "confidence_level": 0.95,
        "confidence_interval": "bootstrap_percentile",
        "glm_max_iterations": 100,
        "glm_tolerance": 1e-8,
        # A fitted hazard this close to 0 or 1 marks quasi-complete separation.
        "separation_probability_bound": 1e-8,
        "positivity_window": (0.025, 0.975),
        "balance_reference": "eligible_population_unweighted_sd",
        "evidence_ceiling": "analysis_only",
    }
)

#: The thresholds the executor stops at, each with its reason code.
TARGET_TRIAL_STOP_THRESHOLDS = MappingProxyType(
    {
        # target_trial_sample_insufficient
        "min_eligible_stays": 100,
        # target_trial_events_insufficient: unweighted deaths of uncensored
        # clones by the horizon, per arm.
        "min_events_per_arm": 10,
        # target_trial_strategy_unobserved: starts in the grace period per
        # parameter of the initiation model.
        "min_initiation_events_per_parameter": 10,
        # target_trial_positivity_violated: eligible stays whose modelled
        # probability of starting in the grace period leaves the window.
        "max_positivity_outside_share": 0.05,
        # target_trial_weights_extreme: Kish ESS of an arm's weights before
        # truncation, as a share of the clones it follows past the grace period.
        "min_ess_share_of_adherers": 0.10,
        # target_trial_icu_exit_excessive: eligible stays that leave the ICU in
        # the grace period before starting.
        "max_grace_icu_exit_share": 0.10,
        # target_trial_bootstrap_unstable
        "max_bootstrap_failure_share": 0.05,
        # Not a stop: a weighted |SMD| above it is reported as a finding.
        "balance_smd_flag": 0.1,
    }
)

#: The executor's stop reason codes.  Stable: a published code never changes.
TARGET_TRIAL_STOP_REASONS = (
    "target_trial_sample_insufficient",
    "target_trial_events_insufficient",
    "target_trial_strategy_unobserved",
    "target_trial_positivity_violated",
    "target_trial_weights_extreme",
    "target_trial_icu_exit_excessive",
    "target_trial_weight_model_not_estimable",
    "target_trial_bootstrap_unstable",
)

#: Why a weight model could not be estimated.
WEIGHT_MODEL_NOT_ESTIMABLE_CAUSES = ("separation", "not_converged", "singular_design")
#: Why the bootstrap gives no interval to report: too many resamples failed,
#: or an estimate lies outside its own percentile interval, which then does
#: not describe it.
BOOTSTRAP_UNSTABLE_CAUSES = ("resamples_failed", "estimate_outside_interval")

#: The forms the ICU-exit model takes: none without exits in the grace period.
ICU_EXIT_MODEL_FORMS = ("none", "hour_terms_and_covariates", "hour_terms_only")


def target_trial_host_policy_sha256() -> str:
    """The digest of every constant above, as one record a signed trial binds."""

    return canonical_sha256(
        {
            "schema_version": TARGET_TRIAL_DESIGN_SCHEMA_VERSION,
            "time_zero_hours": [MIN_TIME_ZERO_HOURS, MAX_TIME_ZERO_HOURS],
            "max_grace_period_hours": MAX_GRACE_PERIOD_HOURS,
            "policy": dict(TARGET_TRIAL_HOST_POLICY),
            "stop_thresholds": dict(TARGET_TRIAL_STOP_THRESHOLDS),
            "stop_reasons": list(TARGET_TRIAL_STOP_REASONS),
            "weight_model_not_estimable_causes": list(
                WEIGHT_MODEL_NOT_ESTIMABLE_CAUSES
            ),
            "bootstrap_unstable_causes": list(BOOTSTRAP_UNSTABLE_CAUSES),
            "icu_exit_model_forms": list(ICU_EXIT_MODEL_FORMS),
        }
    )
