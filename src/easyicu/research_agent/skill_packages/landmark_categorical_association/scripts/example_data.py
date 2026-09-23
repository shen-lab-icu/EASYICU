"""Synthetic example cohort in the shape the skill expects.

The example exists so the whole package can be smoke-tested in seconds and so
a reviewer can see every output file before touching patient data.  It is
generated from declared parameters with a fixed seed; nothing in it comes from
a real database and it must never be presented as a clinical result.

The column roster mirrors the E3 (MIMIC-IV strict KDIGO stage / in-hospital
mortality) study context: one row per ICU stay, ``patient_stay_id`` spelled
``p<patient>:s<stay>``, an ordered four-level exposure with an explicit unknown
state, event time and hospital follow-up in hours, ICU length of stay in days,
and three alternate exposure definitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ....contracts.dependence import PlannedDependenceRequirement
from ..spec import CovariateSpec, LandmarkCategoricalSpec, SecondaryOutcomeSpec

EXAMPLE_SEED = 20260921
EXAMPLE_TRUE_LOG_OR = {"0": 0.0, "1": 0.45, "2": 0.90, "3": 1.60}
EXAMPLE_PROVENANCE = "synthetic_example_generated_by_easyicu_skill"


def example_spec() -> LandmarkCategoricalSpec:
    """The E3-shaped specification the synthetic cohort satisfies."""

    return LandmarkCategoricalSpec(
        title="Synthetic example: strict KDIGO stage (0-24 h) and in-hospital death after a 24 h landmark",
        identity_column="patient_stay_id",
        exposure="aki_stage_strict",
        exposure_levels=["0", "1", "2", "3"],
        reference_level="0",
        primary_contrast_level="3",
        exposure_label="Strict KDIGO AKI stage, 0-24 h",
        outcome="death",
        outcome_label="In-hospital death after the 24 h landmark",
        event_time_column="death_time_hours",
        observation_duration_column="hospital_followup_time_hours",
        observation_duration_unit="hours",
        landmark_hours=24.0,
        covariates=[
            CovariateSpec(name="age", coding="continuous", label="Age (years)"),
            CovariateSpec(
                name="sex",
                coding="binary",
                levels=["Female", "Male"],
                reference_level="Female",
                label="Sex",
            ),
            CovariateSpec(name="charlson", coding="continuous", label="Charlson comorbidity index"),
        ],
        dependence=PlannedDependenceRequirement(
            group_source="patient_stay_id",
            group_derivation="prefix_before_delimiter",
            delimiter=":s",
        ),
        alternate_exposures=["aki_stage_creat_strict", "aki_stage_uo_strict", "aki_stage_reference"],
        first_stay_column="first_icu_stay",
        functional_form_covariates=["age", "charlson"],
        functional_form_knots=4,
        secondary_outcomes=[
            SecondaryOutcomeSpec(name="los_icu", unit="days", label="ICU length of stay (days)")
        ],
        measurement_audit_columns=[
            "aki_ascertainment",
            "kidney_complete_negative_observed",
            "kidney_window_row_count",
        ],
        display_labels={
            "aki_stage_creat_strict": "Strict creatinine-domain stage",
            "aki_stage_uo_strict": "Strict urine-output-domain stage",
            "aki_stage_reference": "Public reference stage",
        },
    )


def make_example_cohort(
    n_stays: int = 3000,
    *,
    seed: int = EXAMPLE_SEED,
    unknown_share: float = 0.20,
) -> pd.DataFrame:
    """Generate one deterministic synthetic ICU-stay cohort."""

    if n_stays < 50:
        raise ValueError("the example cohort needs at least 50 stays")
    rng = np.random.default_rng(seed)
    n_patients = int(n_stays * 0.85)
    patient_of_stay = np.concatenate(
        [np.arange(n_patients), rng.integers(0, n_patients, size=n_stays - n_patients)]
    )
    order = np.argsort(patient_of_stay, kind="stable")
    patient_of_stay = patient_of_stay[order]
    stay_ids = 30_000_000 + np.arange(n_stays)
    first_stay = np.r_[True, patient_of_stay[1:] != patient_of_stay[:-1]]

    age = np.clip(rng.normal(64.0, 16.0, size=n_stays), 18.0, 95.0).round(0)
    sex = np.where(rng.random(n_stays) < 0.56, "Male", "Female")
    charlson = np.clip(rng.poisson(3.2, size=n_stays), 0, 16).astype(float)

    # Latent kidney injury severity drives stage, its alternate definitions,
    # ascertainment and outcome.  Stage probabilities are fixed by design.
    stage = rng.choice([0, 1, 2, 3], size=n_stays, p=[0.58, 0.20, 0.12, 0.10]).astype(float)
    creat_stage = np.where(rng.random(n_stays) < 0.80, stage, np.maximum(stage - 1, 0))
    uo_stage = np.where(rng.random(n_stays) < 0.65, stage, np.maximum(stage - 1, 0))
    reference_stage = np.where(rng.random(n_stays) < 0.90, stage, np.minimum(stage + 1, 3))

    # Strict ascertainment: some stays lack an evaluable baseline; they are
    # unknown and are slightly enriched for sicker patients so that recoding
    # them to stage 0 would be visibly wrong.
    unknown_probability = np.clip(unknown_share + 0.05 * (stage >= 2), 0.0, 0.95)
    unknown = rng.random(n_stays) < unknown_probability
    aki_stage_strict = np.where(unknown, np.nan, stage)
    aki_stage_creat_strict = np.where(rng.random(n_stays) < 0.12, np.nan, creat_stage)
    aki_stage_uo_strict = np.where(rng.random(n_stays) < 0.30, np.nan, uo_stage)
    aki_stage_reference = reference_stage  # the public reference never reports unknown
    ascertainment = np.where(
        unknown, 3.0, np.where(stage > 0, 0.0, np.where(rng.random(n_stays) < 0.7, 1.0, 2.0))
    )
    complete_negative = ((~unknown) & (stage == 0)).astype(float)
    window_rows = np.clip(rng.poisson(6.0 + 2.0 * stage, size=n_stays), 0, None).astype(float)

    logit = (
        -3.1
        + np.vectorize(lambda s: EXAMPLE_TRUE_LOG_OR[str(int(s))])(stage)
        + 0.028 * (age - 64.0)
        + 0.10 * (charlson - 3.0)
        + np.where(sex == "Male", 0.08, 0.0)
        + np.where(unknown, 0.25, 0.0)
    )
    death_probability = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n_stays) < death_probability).astype(float)

    # Hospital follow-up in hours; deaths have an event time inside follow-up.
    followup = np.exp(rng.normal(np.log(150.0), 0.75, size=n_stays)) + 2.0
    death_time = np.where(death == 1.0, followup * rng.uniform(0.05, 1.0, size=n_stays), np.nan)
    # A few early deaths and short stays fall before the landmark and must be
    # excluded by the eligibility mask.
    early = rng.random(n_stays) < 0.04
    death_time = np.where(early & (death == 1.0), rng.uniform(0.5, 23.5, size=n_stays), death_time)
    followup = np.where(early & (death == 1.0), death_time, followup)
    short_stay = (rng.random(n_stays) < 0.03) & (death == 0.0)
    followup = np.where(short_stay, rng.uniform(1.0, 23.0, size=n_stays), followup)

    los_icu = np.exp(rng.normal(np.log(2.0) + 0.18 * stage, 0.6, size=n_stays))
    los_icu = np.where(death == 1.0, np.minimum(los_icu, np.nan_to_num(death_time, nan=1e9) / 24.0 + 0.05), los_icu)
    los_icu = np.round(np.clip(los_icu, 0.05, 120.0), 3)

    frame = pd.DataFrame(
        {
            "stay_id": stay_ids.astype("int64"),
            "patient_stay_id": [
                f"p{10_000_000 + int(patient)}:s{int(stay)}"
                for patient, stay in zip(patient_of_stay, stay_ids)
            ],
            "age": age,
            "sex": pd.array(sex, dtype="str"),
            "charlson": charlson,
            "death_time_hours": np.round(death_time, 3),
            "hospital_followup_time_hours": np.round(followup, 3),
            "aki_stage_creat_strict": aki_stage_creat_strict,
            "aki_stage_uo_strict": aki_stage_uo_strict,
            "aki_stage_reference": aki_stage_reference,
            "first_icu_stay": first_stay.astype(float),
            "aki_stage_strict": aki_stage_strict,
            "death": death,
            "aki_ascertainment": ascertainment,
            "kidney_complete_negative_observed": complete_negative,
            "kidney_window_row_count": window_rows,
            "los_icu": los_icu,
        }
    )
    frame.attrs["provenance"] = EXAMPLE_PROVENANCE
    frame.attrs["seed"] = int(seed)
    return frame


def example_truth() -> dict[str, Any]:
    """Design values a test can compare recovered estimates against."""

    return {
        "true_log_or_by_level": dict(EXAMPLE_TRUE_LOG_OR),
        "true_or_stage3_vs_0": float(np.exp(EXAMPLE_TRUE_LOG_OR["3"])),
        "provenance": EXAMPLE_PROVENANCE,
    }


__all__ = [
    "EXAMPLE_PROVENANCE",
    "EXAMPLE_SEED",
    "EXAMPLE_TRUE_LOG_OR",
    "example_spec",
    "example_truth",
    "make_example_cohort",
]
