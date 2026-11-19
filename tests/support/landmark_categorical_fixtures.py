"""Synthetic, case-neutral fixtures for the signed landmark categorical authority.

Shared by the plan-bound roster and the two-group exposure tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.schema import AnalysisPlan


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260922)
    patients = 240
    n = patients * 2
    stage = np.tile(np.arange(4), n // 4)
    age = rng.normal(64.0, 11.0, size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    probability = 1.0 / (1.0 + np.exp(-(-3.2 + 0.38 * stage + 0.012 * (age - 64.0))))
    death = rng.binomial(1, probability, size=n)
    early = (death == 1) & (rng.random(n) < 0.08)
    death_time = np.where(
        death == 1,
        np.where(early, 8.0, rng.uniform(30.0, 160.0, size=n)),
        np.nan,
    )
    followup = rng.uniform(36.0, 180.0, size=n)
    followup[rng.random(n) < 0.04] = 12.0
    return pd.DataFrame(
        {
            "patient_stay_id": [
                f"p{patient}:s{stay}" for patient in range(patients) for stay in (1, 2)
            ],
            "aki_stage_strict": stage,
            "aki_stage_alt": stage,
            "death": death,
            "death_time_hours": death_time,
            "hospital_followup_time_hours": followup,
            "age": age,
            "sex": sex,
            "charlson": rng.integers(0, 8, size=n).astype("float64"),
            "admit_time": pd.date_range("2020-01-01", periods=n, freq="h"),
        }
    )


def _landmark_spec() -> PrespecifiedSensitivitySpec:
    return PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "landmark_24h",
            "axis": "timing",
            "strategy": "landmark",
            "landmark_hours": 24,
            "require_alive_at_landmark": True,
            "exclude_negative_event_times": True,
            "event_time_variable": "death_time_hours",
            "observation_duration_variable": "hospital_followup_time_hours",
            "observation_duration_unit": "hours",
        }
    )


def _dependence() -> PlannedDependenceRequirement:
    return PlannedDependenceRequirement(
        group_source="patient_stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )


def _draft_plan(covariates: list[str] | None = None) -> AnalysisPlan:
    roster = ["age", "sex"] if covariates is None else covariates
    term_by_name = {
        "age": {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
        "charlson": {
            "name": "charlson",
            "role": "covariate",
            "coding": "continuous",
            "transform": "identity",
        },
        "sex": {
            "name": "sex",
            "role": "covariate",
            "coding": "binary",
            "levels": ["Female", "Male"],
            "reference_level": "Female",
            "transform": "treatment_contrast",
        },
    }
    return AnalysisPlan.model_validate(
        {
            "research_question": "Compare KDIGO stages with in-hospital mortality.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "define_landmark_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Define the analysis cohort.",
                    "inputs": ["death", "death_time_hours"],
                    "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
                    "method": "cohort_definition_and_attrition",
                },
                {
                    "step_id": "primary_adjusted_association",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate adjusted stage contrasts.",
                    "inputs": [
                        "artifact:analysis_cohort",
                        "aki_stage_strict",
                        "death",
                        *roster,
                        "patient_stay_id",
                    ],
                    "expected_outputs": ["table:adjusted_association_estimates"],
                    "method": "adjusted_association_models",
                    "sensitivity_spec_ids": ["landmark_24h"],
                    "model_requirements": [
                        {
                            "requirement_id": "primary_stage_model",
                            "outcome": "death",
                            "outcome_type": "binary",
                            "method_family": "statsmodels_logit_mle",
                            "exposure_source": "aki_stage_strict",
                            "analysis_role": "primary",
                            "analysis_set": "source_aware",
                            "covariates": list(roster),
                            "model_terms": [
                                {
                                    "name": "aki_stage_strict",
                                    "role": "exposure",
                                    "coding": "categorical",
                                    "levels": ["0", "1", "2", "3"],
                                    "reference_level": "0",
                                    "transform": "treatment_contrast",
                                },
                                *(
                                    term_by_name.get(
                                        name,
                                        {
                                            "name": name,
                                            "role": "covariate",
                                            "coding": "continuous",
                                            "transform": "identity",
                                        },
                                    )
                                    for name in roster
                                ),
                            ],
                            "exposure_levels": ["0", "1", "2", "3"],
                            "exposure_reference_level": "0",
                            "primary_contrast_level": "3",
                            "dependence": {
                                "group_source": "patient_stay_id",
                                "group_derivation": "prefix_before_delimiter",
                                "delimiter": ":s",
                            },
                        }
                    ],
                },
                {
                    "step_id": "report",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Report results.",
                    "inputs": ["table:adjusted_association_estimates"],
                    "expected_outputs": ["report:study"],
                    "method": "scientific_reporting",
                },
            ],
        }
    )
