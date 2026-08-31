from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkSplineRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.landmark_spline_validation import (
    landmark_spline_runtime_receipt_valid,
)
from easyicu.research_agent.execution.runners.landmark_spline_executor import (
    run_landmark_spline_association,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
    compile_landmark_spline_runtime_projection,
)


def _specs(*, closed_landmark: bool = True):
    landmark = {
        "spec_id": "landmark_24h_primary",
        "axis": "timing",
        "strategy": "landmark",
        "landmark_hours": 24,
        "require_alive_at_landmark": True,
        "exclude_negative_event_times": True,
    }
    if closed_landmark:
        landmark.update(
            {
                "event_time_variable": "death_time",
                "observation_duration_variable": "los_icu",
                "observation_duration_unit": "days",
            }
        )
    return (
        PrespecifiedSensitivitySpec.model_validate(landmark),
        PrespecifiedSensitivitySpec.model_validate(
            {
                "spec_id": "peak_lactate_restricted_cubic_spline",
                "axis": "functional_form",
                "strategy": "restricted_cubic_spline",
                "execution_variables": ["lact"],
            }
        ),
    )


def _universe(tmp_path):
    path = tmp_path / "universe.parquet"
    pd.DataFrame(
        {
            "lact_max": [1.0, 2.0],
            "death": [0, 1],
            "death_time": [float("nan"), 72.0],
            "los_icu": [2.0, 4.0],
            "age": [50.0, 70.0],
            "sex": ["F", "M"],
            "charlson_first": [1.0, 4.0],
        }
    ).to_parquet(path, index=False)
    return path


def test_closed_web_landmark_spline_compiles_signed_runtime(tmp_path) -> None:
    projection = compile_landmark_spline_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=_specs(),
        primary_exposure="lact_max",
        primary_exposure_source="lact",
        target_outcome="death",
        declared_covariates=("age", "sex", "charlson"),
        covariate_operationalizations={"charlson": "charlson_first"},
        target_is_event_status=True,
        universe_path=_universe(tmp_path),
        scientific_configuration_sha256="a" * 64,
    )

    assert projection is not None
    authority = projection.authority
    assert authority["plan_method"] == "signed_landmark_restricted_cubic_spline"
    assert authority["observation_duration_column"] == "los_icu"
    assert authority["categorical_adjustment_columns"] == ["sex"]
    assert authority["required_adjustment_columns"] == [
        "age",
        "sex",
        "charlson_first",
    ]
    assert len(projection.projection_sha256) == 64


def test_selected_web_landmark_spline_fails_when_runtime_coordinates_are_prose_only(
    tmp_path,
) -> None:
    with pytest.raises(WebScientificRuntimeProjectionError) as caught:
        compile_landmark_spline_runtime_projection(
            study={"covariate_selection": "exact"},
            sensitivity_specs=_specs(closed_landmark=False),
            primary_exposure="lact_max",
            primary_exposure_source="lact",
            target_outcome="death",
            declared_covariates=("age", "sex", "charlson"),
            covariate_operationalizations={"charlson": "charlson_first"},
            target_is_event_status=True,
            universe_path=_universe(tmp_path),
            scientific_configuration_sha256="b" * 64,
        )

    assert caught.value.code == "web_landmark_spline_authority_incomplete"
    assert caught.value.details["missing_fields"] == [
        "landmark.event_time_variable",
        "landmark.observation_duration_variable",
        "landmark.observation_duration_unit",
    ]


def test_landmark_observation_duration_is_a_source_materialization_variable() -> None:
    landmark = _specs()[0]

    assert landmark.source_materialization_variables == ("los_icu",)


def test_landmark_event_time_is_never_requested_from_source_modules() -> None:
    landmark = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "landmark_24h_legacy_web",
            "axis": "timing",
            "strategy": "landmark",
            "execution_variables": ["death_time", "los_icu"],
            "landmark_hours": 24,
            "require_alive_at_landmark": True,
            "exclude_negative_event_times": True,
            "event_time_variable": "death_time",
            "observation_duration_variable": "los_icu",
            "observation_duration_unit": "days",
        }
    )

    assert landmark.source_materialization_variables == ("los_icu",)


def test_web_projection_requires_explicit_repeated_covariate_operationalization(
    tmp_path,
) -> None:
    with pytest.raises(WebScientificRuntimeProjectionError) as caught:
        compile_landmark_spline_runtime_projection(
            study={"covariate_selection": "exact"},
            sensitivity_specs=_specs(),
            primary_exposure="lact_max",
            primary_exposure_source="lact",
            target_outcome="death",
            declared_covariates=("age", "sex", "charlson"),
            covariate_operationalizations={},
            target_is_event_status=True,
            universe_path=_universe(tmp_path),
            scientific_configuration_sha256="c" * 64,
        )

    assert caught.value.code == "web_covariate_operationalization_required"
    assert caught.value.details["missing_operationalizations"] == ["charlson"]


def test_web_landmark_projection_executes_declared_patient_cluster_covariance(
    tmp_path,
) -> None:
    rng = np.random.default_rng(20260831)
    patient_count = 90
    n = patient_count * 2
    lactate = rng.lognormal(mean=0.55, sigma=0.45, size=n)
    age = rng.normal(64.0, 12.0, size=n)
    probability = 1.0 / (1.0 + np.exp(-(-3.5 + 0.5 * lactate)))
    death = rng.binomial(1, probability, size=n)
    universe = tmp_path / "clustered_universe.parquet"
    pd.DataFrame(
        {
            "patient_stay_id": [
                f"p{patient}:s{stay}"
                for patient in range(patient_count)
                for stay in (1, 2)
            ],
            "lact_max": lactate,
            "death": death,
            "death_time": np.where(
                death == 1, rng.uniform(30.0, 180.0, size=n), np.nan
            ),
            "los_icu": rng.uniform(1.2, 8.0, size=n),
            "age": age,
            "sex": rng.choice(["F", "M"], size=n),
            "charlson_first": rng.poisson(3.0, size=n).astype(float),
        }
    ).to_parquet(universe, index=False)
    dependence = PlannedDependenceRequirement(
        group_source="patient_stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )
    projection = compile_landmark_spline_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=_specs(),
        primary_exposure="lact_max",
        primary_exposure_source="lact",
        target_outcome="death",
        declared_covariates=("age", "sex", "charlson"),
        covariate_operationalizations={"charlson": "charlson_first"},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="d" * 64,
        dependence=dependence,
    )

    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(
        projection.authority
    )
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    assert authority.schema_version.endswith("/3")
    assert "patient_stay_id" in authority.required_columns
    summary = run_landmark_spline_association(
        frame=pd.read_parquet(universe),
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "out",
    )

    assert summary["variance_estimator"] == "cluster_robust"
    assert summary["cluster_count"] == patient_count
    assert summary["scientific_runtime_receipt"]["cluster_group_source"] == (
        "patient_stay_id"
    )
    assert landmark_spline_runtime_receipt_valid(summary)
