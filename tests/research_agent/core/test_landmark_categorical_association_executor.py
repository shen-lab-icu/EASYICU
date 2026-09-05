from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.capability_ids import (
    LANDMARK_CATEGORICAL_ASSOCIATION_CAPABILITY_ID,
)
from easyicu.research_agent.execution.runners.landmark_categorical_association_executor import (
    run_landmark_categorical_cohort,
    run_landmark_categorical_primary,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.planning.capability_registry import (
    resolve_primary_capability,
)
from easyicu.research_agent.planning.scientific_review import timing_design_closed
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260904)
    patients = 240
    n = patients * 2
    stage = np.tile(np.arange(4), n // 4)
    age = rng.normal(64.0, 11.0, size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    probability = 1.0 / (
        1.0 + np.exp(-(-3.2 + 0.38 * stage + 0.012 * (age - 64.0)))
    )
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
                f"p{patient}:s{stay}"
                for patient in range(patients)
                for stay in (1, 2)
            ],
            "aki_stage_max": stage,
            "death": death,
            "death_time_hours": death_time,
            "hospital_followup_time_hours": followup,
            "age": age,
            "sex": sex,
        }
    )


def _projection(tmp_path):
    universe = tmp_path / "universe.parquet"
    _frame().to_parquet(universe, index=False)
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
        primary_exposure="aki_stage_max",
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
    return universe, projection, authority


def _draft_plan() -> AnalysisPlan:
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
                        "aki_stage_max",
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
                            "exposure_source": "aki_stage_max",
                            "analysis_role": "primary",
                            "analysis_set": "source_aware",
                            "covariates": ["age", "sex"],
                            "model_terms": [
                                {
                                    "name": "aki_stage_max",
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


def test_signed_landmark_categorical_owner_filters_then_fits(tmp_path) -> None:
    universe, projection, authority = _projection(tmp_path)
    bound, findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(_draft_plan())

    assert findings[0].detail["reason_code"] == (
        "landmark_categorical_association_host_compiled"
    )
    assert "duplicate_landmark_sensitivity" not in {
        step.step_id for step in bound.steps
    }
    # Host binding is not complete until the public plan survives the same
    # serialize/rehydrate boundary used by normalized-plan authority.
    bound = AnalysisPlan.model_validate(bound.model_dump(mode="json"))
    primary = next(
        step for step in bound.steps if step.planned_analysis_role == "primary"
    )
    assert (
        primary.scientific_capability
        == LANDMARK_CATEGORICAL_ASSOCIATION_CAPABILITY_ID
    )
    capability = resolve_primary_capability(
        analysis_type=bound.analysis_type,
        plan=bound,
    )
    assert capability.failure_reason is None
    assert capability.owner_claimed is True
    assert capability.scientific_validation == "reportable"
    report = next(step for step in bound.steps if step.step_id == "report")
    assert "table:sensitivity_landmark_24h" not in report.inputs
    authority.validate_plan(bound)
    assert timing_design_closed(bound) is True

    cohort_step = authority.governed_cohort_step(bound)
    cohort_selection = select_standard_executor(
        cohort_step,
        plan=bound,
        current_case_scientific_runtime_authority=projection.authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )
    assert cohort_selection is not None
    assert cohort_selection.analysis_kind == "signed_landmark_analysis_cohort"
    cohort_summary = run_landmark_categorical_cohort(
        frame=pd.read_parquet(universe),
        source_path=universe,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "cohort",
    )
    assert cohort_summary["n_analysis_cohort"] < cohort_summary["n_source"]
    eligible = pd.read_parquet(tmp_path / "cohort" / "analysis_cohort.parquet")
    assert (eligible["hospital_followup_time_hours"] >= 24).all()
    assert (
        (eligible["death"] == 0) | (eligible["death_time_hours"] > 24)
    ).all()
    assert cohort_step.method == "signed_landmark_analysis_cohort"

    primary = authority.governed_primary_step(bound)
    primary_selection = select_standard_executor(
        primary,
        plan=bound,
        current_case_scientific_runtime_authority=projection.authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )
    assert primary_selection is not None
    assert primary_selection.analysis_kind == "adjusted_association_estimates"
    primary_summary = run_landmark_categorical_primary(
        frame=eligible,
        cohort_path=tmp_path / "cohort" / "analysis_cohort.parquet",
        step=primary,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "primary",
    )
    assert primary_summary["variance_estimator"] == "cluster_robust"
    assert primary_summary["cluster_count"] > 1
    assert len(primary_summary["model_contracts"]) == 1
    assert primary_summary["landmark_runtime_receipt"]["exposure_levels"] == [
        "0",
        "1",
        "2",
        "3",
    ]
