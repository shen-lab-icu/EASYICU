"""A synthetic grace-period target trial, shared by its tests.

Test modules may not import one another (``tests/governance/test_test_organization.py``).
The signed target trial suite's authority body and a synthetic cohort with a
known answer serve the authority, executor, claim and figure tests: does
starting a vasopressor within six hours of time zero, six hours after ICU
admission, change 28-day mortality?  No benchmark item asks it.  The cohort
is simulated: sicker stays start sooner and die more often, a start lowers the
hazard of death after the grace period by a fixed factor, and some stays leave
the ICU or die within the grace period.  A death after the ICU stay is timed
to the hour when the hospital records it, else by calendar day only.  Zero
real patient rows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from easyicu.research_agent.contracts.target_trial_design import (
    target_trial_host_policy_sha256,
)

OUTPUTS = (
    "table:target_trial_protocol",
    "table:target_trial_eligibility_flow",
    "table:target_trial_table_one",
    "table:target_trial_risk_curves",
    "table:target_trial_effect_estimates",
    "table:target_trial_weight_models",
    "table:target_trial_weight_diagnostics",
    "table:target_trial_covariate_balance",
    "table:target_trial_positivity",
    "table:target_trial_adherence",
    "table:target_trial_bootstrap_replicates",
    "log:target_trial_runtime_receipt",
    "figure:target_trial_emulation",
)
TIME_ZERO = 6
GRACE = 6
HORIZON_DAYS = 28
COMPILE_SHA256 = "c" * 64


def target_trial_covariates() -> list[dict[str, Any]]:
    return [
        {
            "column": "age",
            "label": "Age",
            "coding": "continuous",
            "levels": [],
            "reference_level": None,
            "unmeasured_state": False,
        },
        {
            "column": "sex",
            "label": "Sex",
            "coding": "binary",
            "levels": ["female", "male"],
            "reference_level": "female",
            "unmeasured_state": False,
        },
        {
            "column": "lactate_max",
            "label": "Lactate, highest before time zero",
            "coding": "continuous",
            "levels": [],
            "reference_level": None,
            "unmeasured_state": True,
        },
        {
            "column": "map_min",
            "label": "Mean arterial pressure, lowest before time zero",
            "coding": "continuous",
            "levels": [],
            "reference_level": None,
            "unmeasured_state": False,
        },
    ]


def target_trial_authority_body(**overrides: Any) -> dict[str, Any]:
    """A complete authority body for the synthetic trial, without its digest.

    The host's builder computes the execution contract digest, so a test of
    the closed contract sees its own field, not a digest mismatch.
    """

    body: dict[str, Any] = {
        "schema_version": "easyicu.target_trial_runtime_authority/1",
        "authority_kind": "target_trial_suite",
        "protocol_content_sha256": "a" * 64,
        "target_trial_compile_sha256": COMPILE_SHA256,
        "target_trial_compile_confirmation_lines": 3,
        "host_policy_sha256": target_trial_host_policy_sha256(),
        "confirmation": {
            "confirmed_by": "researcher",
            "approval_event_id": "approval:card:0001",
            "confirmed_compile_sha256": COMPILE_SHA256,
            "n_lines_confirmed": 3,
        },
        "plan_method": "signed_target_trial_suite",
        "plan_intent": "Execute the signed target trial emulation by clone, censor and weight.",
        "plan_outputs": list(OUTPUTS),
        "development_execution_only_allowed": False,
        "database": "miiv",
        "analysis_unit_label": "ICU stays",
        "eligibility_label": "Adult ICU stays with septic shock",
        "treatment_label": "a vasopressor",
        "initiate_label": "Early vasopressor",
        "defer_label": "No early vasopressor",
        "outcome_label": "Death",
        "unit_id_column": "stay_id",
        "resampling_unit": "patient",
        "patient_group_column": "subject_id",
        "patient_group_derivation": "identity",
        "patient_group_delimiter": None,
        "treatment_onset_columns": [
            "norepinephrine_onset_time",
            "vasopressin_onset_time",
        ],
        "treatment_onset_window_hours": [0.0, 12.0],
        "time_zero_hours": TIME_ZERO,
        "grace_period_hours": GRACE,
        "event_column": "mort_28d",
        "followup_time_column": "followup_days_28d",
        "endpoint_horizon_days": HORIZON_DAYS,
        "endpoint_time_origin": "icu_admission",
        "death_status_column": "death",
        "death_time_column": "death_time",
        "icu_length_of_stay_column": "los_icu",
        "covariates": target_trial_covariates(),
        "estimator": "clone_censor_weight",
        "interpretation": "per_protocol_effect_under_emulation_assumptions",
        "evidence_ceiling": "analysis_only",
        "protocol_product": OUTPUTS[0],
        "eligibility_product": OUTPUTS[1],
        "table_one_product": OUTPUTS[2],
        "risk_curve_product": OUTPUTS[3],
        "effect_product": OUTPUTS[4],
        "weight_model_product": OUTPUTS[5],
        "weight_product": OUTPUTS[6],
        "balance_product": OUTPUTS[7],
        "positivity_product": OUTPUTS[8],
        "adherence_product": OUTPUTS[9],
        "bootstrap_product": OUTPUTS[10],
        "receipt_product": OUTPUTS[11],
        "figure_product": OUTPUTS[12],
    }
    body.update(overrides)
    return body


def _expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def synthetic_target_trial_cohort(
    n: int = 3000,
    seed: int = 20261009,
    *,
    time_zero: int = TIME_ZERO,
    grace: int = GRACE,
) -> pd.DataFrame:
    """A cohort whose stays start, leave the ICU and die by hours since admission.

    Sicker stays (higher lactate, lower MAP) start the vasopressor sooner and
    die more often; a start before ``T0 + G`` multiplies the hazard of death
    after the grace period by 0.7.  Some stays start before time zero, die or
    leave the ICU before it, or are missing lactate (kept as their own state).
    Half the deaths after ICU exit happen after hospital discharge: the
    in-hospital ``death`` status is 0, no ``death_time`` is recorded, and the
    follow-up counts whole calendar days to the date of death.  Patients
    contribute up to two stays.  ``time_zero`` and ``grace`` set the trial the
    stays are simulated for; onsets are captured through its grace period.
    """

    grace_end = time_zero + grace
    rng = np.random.default_rng(seed)
    age = rng.normal(64.0, 14.0, n).clip(18.0, 95.0)
    male = rng.random(n) < 0.56
    lactate = np.exp(rng.normal(0.8, 0.5, n))
    map_min = rng.normal(64.0, 8.0, n)
    severity = 0.6 * (np.log(lactate) - 0.8) / 0.5 - 0.4 * (map_min - 64.0) / 8.0
    onset = np.full(n, np.nan)
    death = np.full(n, np.nan)
    exit_ = np.full(n, np.nan)
    for i in range(n):
        alive, in_icu, started = True, True, False
        for hour in range(grace_end):
            if not started and rng.random() < _expit(-3.4 + 0.9 * severity[i]):
                onset[i] = hour + 0.1
                started = True
            if rng.random() < _expit(-7.0 + 0.8 * severity[i]):
                # A death in the ICU ends the ICU stay too.
                death[i] = exit_[i] = hour + 0.7
                alive = False
                break
            if rng.random() < _expit(-6.0 - 0.6 * severity[i]):
                exit_[i] = hour + 0.4
                in_icu = False
                break
        if alive:
            effect = 0.7 if started and onset[i] < grace_end else 1.0
            rate = math.exp(-7.6 + 0.7 * severity[i] + 0.01 * (age[i] - 64.0)) * effect
            later = grace_end + rng.exponential(1.0 / rate)
            if later <= HORIZON_DAYS * 24:
                death[i] = later
            if in_icu:
                exit_[i] = grace_end + rng.exponential(120.0)
                if not np.isnan(death[i]) and death[i] < exit_[i]:
                    exit_[i] = death[i]
    in_window = np.isnan(onset) | (onset < float(grace_end))
    onset = np.where(in_window, onset, np.nan)
    norepinephrine = np.where(rng.random(n) < 0.8, onset, np.nan)
    vasopressin = np.where(np.isnan(norepinephrine), onset, np.nan)
    died = ~np.isnan(death)
    after_discharge = died & (death > exit_) & (rng.random(n) < 0.5)
    in_hospital = died & ~after_discharge
    followup = np.where(died, death / 24.0, float(HORIZON_DAYS))
    followup = np.where(after_discharge, np.ceil(death / 24.0), followup)
    lactate_recorded = np.where(rng.random(n) < 0.9, lactate, np.nan)
    subject = np.arange(n) // 2 + 1000
    stays = pd.DataFrame(
        {
            "stay_id": np.arange(n) + 30_000_000,
            "subject_id": subject,
            "norepinephrine_onset_time": norepinephrine,
            "vasopressin_onset_time": vasopressin,
            "mort_28d": died.astype(int),
            "followup_days_28d": followup,
            "death": in_hospital.astype(int),
            "death_time": np.where(in_hospital, death, np.nan),
            "los_icu": exit_ / 24.0,
            "age": age,
            "sex": np.where(male, "male", "female"),
            "lactate_max": lactate_recorded,
            "map_min": map_min,
        }
    )
    return stays


__all__ = [
    "COMPILE_SHA256",
    "GRACE",
    "HORIZON_DAYS",
    "OUTPUTS",
    "TIME_ZERO",
    "synthetic_target_trial_cohort",
    "target_trial_authority_body",
    "target_trial_covariates",
]
