"""Seeded synthetic designs for runner integration checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)

from easyicu.research_agent.authority.rmst_runtime import RmstRuntimeAuthority

from easyicu.research_agent.contracts.rmst import RMSTSpec

from tests.support.continuous_survival import continuous_authority_body


def continuous_rows(
    n: int = 1500, *, seed: int = 7, crossing: bool = False, exposure=None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(65.0, 12.0, n)
    sex = rng.choice(["F", "M"], n)
    lab = np.exp(rng.normal(0.7, 0.5, n)) if exposure is None else exposure(rng, n)
    lab[rng.random(n) < 0.08] = np.nan
    rate = np.exp(-4.2 + 0.25 * np.nan_to_num(lab, nan=2.0) + 0.02 * (age - 65.0))
    time = rng.exponential(1.0 / rate)
    if crossing:
        time = np.where(
            time > 10.0, rng.exponential(1.0 / np.exp(-4.0), n) + 10.0, time
        )
    time = np.where(rng.random(n) < 0.05, rng.uniform(0.0, 1.0, n), time)
    return pd.DataFrame(
        {
            "lab_max": lab,
            "mort_28d": (time <= 28.0).astype(int),
            "followup_days_28d": np.minimum(time, 28.0),
            "age": age,
            "sex": sex,
        }
    )


def rmst_spec(**overrides) -> RMSTSpec:
    payload = {
        "time_column": "time_days",
        "event_column": "death",
        "event_code": 1.0,
        "group_column": "lact_group",
        "group_levels": ["high", "low"],
        "tau": 28.0,
        "time_unit": "days",
    }
    payload.update(overrides)
    return RMSTSpec(**payload)


def rmst_authority(spec: RMSTSpec | None = None) -> RmstRuntimeAuthority:
    authority = build_current_case_scientific_runtime_authority(
        {
            "schema_version": "easyicu.rmst_runtime_authority/1",
            "authority_kind": "restricted_mean_survival_difference",
            "protocol_content_sha256": "a" * 64,
            "specification": (spec or rmst_spec()).model_dump(mode="json"),
            "sensitivity_spec_id": "rmst_28d",
            "identity_column": "stay_id",
            "primary_cohort_selection_mode": "all_input_rows",
            "development_execution_only_allowed": True,
            "plan_method": "rmst",
            "plan_intent": "Estimate the prespecified 28-day RMST contrast.",
            "plan_outputs": ["table:rmst_summary", "log:rmst_runtime_receipt"],
        }
    )
    assert isinstance(authority, RmstRuntimeAuthority)
    return authority


def rmst_frame(n: int = 120) -> pd.DataFrame:
    times = 1.0 + 59.0 * (np.arange(n) / max(n - 1, 1))
    events = (np.arange(n) % 3 != 0).astype(float)
    groups = np.where(np.arange(n) % 2 == 0, "high", "low")
    return pd.DataFrame(
        {
            "stay_id": [f"s{i}" for i in range(n)],
            "time_days": times,
            "death": events,
            "lact_group": groups,
        }
    )


def rmst_step(**overrides) -> AnalysisStep:
    payload = {
        "step_id": "rmst_sensitivity",
        "planned_analysis_role": "sensitivity",
        "intent": "Estimate the prespecified 28-day RMST contrast.",
        "inputs": ["artifact:analysis_cohort"],
        "expected_outputs": ["table:rmst_summary", "log:rmst_runtime_receipt"],
        "method": "rmst",
        "sensitivity_spec_ids": ["rmst_28d"],
    }
    payload.update(overrides)
    return AnalysisStep(**payload)


from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)

from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement

from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)

from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
)


def categorical_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260904)
    patients = 240
    n = patients * 2
    stage = np.tile(np.arange(4), n // 4)
    age = rng.normal(64.0, 11.0, size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    probability = 1.0 / (1.0 + np.exp(-(-3.2 + 0.38 * stage + 0.012 * (age - 64.0))))
    death = rng.binomial(1, probability, size=n)
    early = (death == 1) & (rng.random(n) < 0.08)
    death_time = np.where(
        death == 1, np.where(early, 8.0, rng.uniform(30.0, 160.0, size=n)), np.nan
    )
    followup = rng.uniform(36.0, 180.0, size=n)
    followup[rng.random(n) < 0.04] = 12.0
    return pd.DataFrame(
        {
            "patient_stay_id": [
                f"p{patient}:s{stay}" for patient in range(patients) for stay in (1, 2)
            ],
            "aki_stage_strict": stage,
            "death": death,
            "death_time_hours": death_time,
            "hospital_followup_time_hours": followup,
            "age": age,
            "sex": sex,
        }
    )


def categorical_projection(tmp_path):
    universe = tmp_path / "universe.parquet"
    categorical_frame().to_parquet(universe, index=False)
    landmark = PrespecifiedSensitivitySpec.model_validate(
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
    dependence = PlannedDependenceRequirement(
        group_source="patient_stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=(landmark,),
        primary_exposure="aki_stage_strict",
        primary_exposure_source="aki_stage",
        target_outcome="death",
        declared_covariates=("age", "sex"),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="a" * 64,
        dependence=dependence,
    )
    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)
    return (universe, projection, authority)


def categorical_draft_plan() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Compare invented categories with generated binary responses.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "define_landmark_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Define the analysis cohort.",
                    "inputs": ["death", "death_time_hours"],
                    "expected_outputs": [
                        "artifact:analysis_cohort",
                        "table:cohort_flow",
                    ],
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
                        "age",
                        "sex",
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
                            "covariates": ["age", "sex"],
                            "model_terms": [
                                {
                                    "name": "aki_stage_strict",
                                    "role": "exposure",
                                    "coding": "categorical",
                                    "levels": ["0", "1", "2", "3"],
                                    "reference_level": "0",
                                    "transform": "treatment_contrast",
                                },
                                {
                                    "name": "age",
                                    "role": "covariate",
                                    "coding": "continuous",
                                    "transform": "identity",
                                },
                                {
                                    "name": "sex",
                                    "role": "covariate",
                                    "coding": "binary",
                                    "levels": ["Female", "Male"],
                                    "reference_level": "Female",
                                    "transform": "treatment_contrast",
                                },
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
                    "step_id": "duplicate_landmark_sensitivity",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Repeat the landmark analysis.",
                    "inputs": [
                        "artifact:analysis_cohort",
                        "table:adjusted_association_estimates",
                    ],
                    "expected_outputs": ["table:sensitivity_landmark_24h"],
                    "method": "landmark_analysis",
                    "sensitivity_spec_ids": ["landmark_24h"],
                },
                {
                    "step_id": "report",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Report results.",
                    "inputs": [
                        "table:adjusted_association_estimates",
                        "table:sensitivity_landmark_24h",
                    ],
                    "expected_outputs": ["report:study"],
                    "method": "scientific_reporting",
                },
            ],
        }
    )


def binary_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260923)
    patients = 300
    n = patients * 2
    stage = rng.integers(0, 4, size=n)
    age = rng.normal(64.0, 11.0, size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    probability = 1.0 / (
        1.0
        + np.exp(
            -(-3.0 + 0.4 * stage + 0.015 * (age - 64.0) + 0.0004 * (age - 64.0) ** 2)
        )
    )
    death = rng.binomial(1, probability, size=n)
    return pd.DataFrame(
        {
            "patient_stay_id": [f"p{p}:s{s}" for p in range(patients) for s in (1, 2)],
            "first_icu_stay": np.tile([1.0, 0.0], patients),
            "aki_stage_strict": stage,
            "death": death,
            "age": age,
            "sex": sex,
        }
    )


def binary_plan(
    *,
    sensitivity_inputs: list[str] | None = None,
    method: str = "first_stay_association",
    functional_form: dict | None = None,
) -> AnalysisPlan:
    sensitivity = {
        "step_id": "sensitivity_first_icu_stay_only",
        "planned_analysis_role": "sensitivity",
        "intent": "Refit on first ICU stays only.",
        "inputs": sensitivity_inputs
        or [
            "aki_stage_strict",
            "death",
            "age",
            "sex",
            "first_icu_stay",
            "artifact:analysis_cohort",
            "table:adjusted_association_estimates",
        ],
        "expected_outputs": ["table:sensitivity_first_icu_stay_only"],
        "method": method,
        "scientific_capability": "association_freeform_v1",
        "sensitivity_spec_ids": ["first_icu_stay_only"],
    }
    if "table:adjusted_association_estimates" in sensitivity["inputs"]:
        sensitivity["input_consumption_contracts"] = [
            {"input_key": "table:adjusted_association_estimates", "mode": "all_rows"}
        ]
    if functional_form is not None:
        sensitivity = {
            **sensitivity,
            "step_id": "age_functional_form",
            "intent": "Refit with age as a restricted cubic spline.",
            "expected_outputs": ["table:age_functional_form_sensitivity"],
            "sensitivity_spec_ids": ["age_restricted_cubic_spline"],
            "functional_form_spec": functional_form,
        }
    return AnalysisPlan.model_validate(
        {
            "research_question": "Compare invented ordered categories with generated responses.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "define_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Define the analysis cohort.",
                    "inputs": ["death"],
                    "expected_outputs": ["artifact:analysis_cohort"],
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
                        "age",
                        "sex",
                        "patient_stay_id",
                    ],
                    "expected_outputs": ["table:adjusted_association_estimates"],
                    "method": "adjusted_association_models",
                    "model_requirements": [
                        {
                            "requirement_id": "primary_stage_model",
                            "outcome": "death",
                            "outcome_type": "binary",
                            "method_family": "statsmodels_logit_mle",
                            "exposure_source": "aki_stage_strict",
                            "analysis_role": "primary",
                            "analysis_set": "source_aware",
                            "covariates": ["age", "sex"],
                            "model_terms": [
                                {
                                    "name": "aki_stage_strict",
                                    "role": "exposure",
                                    "coding": "categorical",
                                    "levels": ["0", "1", "2", "3"],
                                    "reference_level": "0",
                                    "transform": "treatment_contrast",
                                },
                                {
                                    "name": "age",
                                    "role": "covariate",
                                    "coding": "continuous",
                                    "transform": "identity",
                                },
                                {
                                    "name": "sex",
                                    "role": "covariate",
                                    "coding": "binary",
                                    "levels": ["Female", "Male"],
                                    "reference_level": "Female",
                                    "transform": "treatment_contrast",
                                },
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
                sensitivity,
            ],
        }
    )


from easyicu.research_agent.schema import TrajectoryStabilitySpec
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)


def trajectory_authority():
    columns = ("sofa2_resp__h0_12", "sofa2_resp__h12_24", "lact__h0_12", "lact__h12_24")
    spec = TrajectoryStabilitySpec(
        n_resamples=2,
        sample_fraction=0.75,
        base_seed=1729,
        minimum_successful_resamples=2,
        refit_max_iter=500,
        refit_tolerance=1e-05,
        refit_regularization=1e-06,
        minimum_mean_stability=0.0,
        decision_mode="minimum_mean_threshold",
    )
    return build_trajectory_scientific_runtime_authority(
        {
            "schema_version": "easyicu.trajectory_scientific_runtime_authority/1",
            "protocol_content_sha256": "1" * 64,
            "coordinate_concepts": ["sofa2_resp", "lact"],
            "descriptive_only_concepts": ["sofa2"],
            "window_start_hours": 0,
            "window_end_hours": 24,
            "grid_width_hours": 12,
            "aggregation": "max",
            "representation_columns": list(columns),
            "minimum_available_windows": 2,
            "coordinate_scaling": {
                "method": "pooled_coordinate_wise_z_score",
                "ddof": 0,
                "observed_value_policy": "direct_or_owner_locf_available",
                "missing_value_policy": "preserve_missing_exclude_from_likelihood",
                "zero_variance_action": "fail_closed",
            },
            "evidence_state_policy": {
                "direct_observed": "include",
                "owner_locf_available": "include_and_audit",
                "unavailable": "exclude",
                "additional_clustering_stage_imputation": "none",
            },
            "representation_plan_method": "signed_fixed_window_trajectory_representation",
            "representation_plan_intent": "Build the digest-bound fixed-window trajectory representation exactly as declared by the scientific runtime authority.",
            "representation_plan_inputs": [],
            "representation_required_outputs": [
                "artifact:trajectory_representation",
                "table:trajectory_membership",
                "manifest:trajectory_representation_schema",
            ],
            "model_family": "latent_class_diagonal_gaussian_mixture",
            "fit_method": "observed_data_em_diagonal_gaussian_mixture",
            "covariance_type": "diag",
            "candidate_cluster_counts": [2, 3, 4],
            "selection_criterion": "bic",
            "selection_rule": "minimum",
            "candidate_fit_base_seed": 1729,
            "candidate_fit_max_iter": 500,
            "candidate_fit_tolerance": 1e-05,
            "candidate_fit_regularization": 1e-06,
            "bic_sample_size": "frozen_population_rows",
            "bic_parameter_count": "mixture_weights_k_minus_1_plus_2_k_per_coordinate",
            "bic_tie_break": "smaller_k",
            "upper_boundary_action": "fail_closed_if_selected_at_upper_boundary",
            "upper_boundary_reason_code": "H3_NO_INTERIOR_BIC_OPTIMUM",
            "minimum_cluster_fraction": 0.05,
            "minimum_cluster_fraction_reason_code": "H3_MINIMUM_CLUSTER_FRACTION_NOT_MET",
            "stability_spec": spec.model_dump(mode="json"),
        }
    )
