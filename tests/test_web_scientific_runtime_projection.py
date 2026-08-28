from __future__ import annotations

import pandas as pd
import pytest

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
