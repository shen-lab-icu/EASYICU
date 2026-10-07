"""The Web host seals a survival study's continuous exposure on its own suite.

A survival-family landmark study whose exposure was a laboratory value or a
vital sign failed at launch: the survival projection accepted one binary
exposure only.  A continuous exposure is now compiled into the continuous
suite on the same validated endpoint, landmark, unit and roster, provided its
column is the summary of a window that ends at the landmark: a value recorded
later would let the future enter the exposure.

Synthetic zero-row and seeded universes only.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.landmark_continuous_survival_runtime import (
    LandmarkContinuousSurvivalRuntimeAuthority,
)
from easyicu.research_agent.execution.runners.landmark_continuous_survival_executor import (
    run_landmark_continuous_survival_suite,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.webserver import landmark_continuous_survival_runtime_projection as continuous_projection
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
    compile_web_scientific_runtime_projection,
)


def _landmark(**overrides):
    payload = {
        "spec_id": "landmark_24h",
        "axis": "timing",
        "strategy": "landmark",
        "landmark_hours": 24,
        "require_alive_at_landmark": True,
        "exclude_negative_event_times": True,
        "observation_duration_variable": "followup_days_28d",
        "observation_duration_unit": "days",
    }
    payload.update(overrides)
    return PrespecifiedSensitivitySpec.model_validate(payload)


def _study(**overrides):
    study = {
        "covariate_selection": "exact",
        "analysis_design": {
            "analysis_family": "survival",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
    }
    study.update(overrides)
    return study


def _universe(tmp_path, *, n: int = 800):
    rng = np.random.default_rng(20261007)
    lactate = np.exp(rng.normal(0.6, 0.5, size=n))
    rate = np.exp(-4.0 + 0.3 * lactate)
    time = rng.exponential(1.0 / rate)
    frame = pd.DataFrame(
        {
            "lact_max": lactate,
            "lact_mean": lactate * 0.8,
            "mort_28d": (time <= 28.0).astype("int64"),
            "followup_days_28d": np.minimum(time, 28.0),
            "age": rng.normal(63.0, 14.0, size=n),
            "sex": rng.choice(["F", "M"], size=n),
            "charlson_first": rng.poisson(3.0, size=n).astype(float),
        }
    )
    path = tmp_path / "survival_universe.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def _coordinates(universe, **overrides):
    coordinates = {
        "study": _study(),
        "sensitivity_specs": (_landmark(),),
        "primary_exposure": "lact_max",
        "primary_exposure_source": "lact",
        "target_outcome": "mort_28d",
        "declared_covariates": ("age", "sex", "charlson"),
        "covariate_operationalizations": {"charlson": "charlson_first"},
        "target_is_event_status": True,
        "universe_path": universe,
        "scientific_configuration_sha256": "a" * 64,
    }
    coordinates.update(overrides)
    return coordinates


def test_a_continuous_exposure_compiles_the_continuous_suite_and_executes(tmp_path):
    universe, frame = _universe(tmp_path)

    projection = compile_web_scientific_runtime_projection(**_coordinates(universe))

    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkContinuousSurvivalRuntimeAuthority)
    assert authority.exposure_column == "lact_max"
    assert authority.exposure_window_summary == "max"
    assert authority.exposure_window_hours == (0.0, 24.0)
    assert authority.exposure_label == "Lactate, highest in hours 0 to 24"
    assert authority.exposure_unit == "mmol/L"
    assert (authority.event_column, authority.followup_time_column) == (
        "mort_28d", "followup_days_28d",
    )
    assert authority.landmark_hours == 24.0 and authority.endpoint_horizon_days == 28.0
    assert authority.adjustment_columns == ("age", "sex", "charlson_first")
    assert authority.categorical_adjustment_columns == ("sex",)
    assert authority.time_varying_interval_cutpoints_days == (7.0, 14.0)
    assert authority.analysis_unit_label == "ICU stays"
    assert authority.measurement_audit_product == "table:landmark_continuous_measurement_audit"
    assert authority.development_execution_only_allowed is False
    summary = run_landmark_continuous_survival_suite(
        frame=frame,
        authority=projection.authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "out",
        input_product="table:analysis_cohort",
        input_evidence_id="cohort",
        input_sha256="c" * 64,
    )
    assert summary["reportable_survival_results"]["exposure"] == "lact_max"


def test_the_first_stay_unit_reaches_the_continuous_suite(tmp_path):
    universe, _frame = _universe(tmp_path)
    study = _study(cohort={"preset": "adult_first"})

    projection = compile_web_scientific_runtime_projection(
        **_coordinates(universe, study=study)
    )

    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.analysis_unit_label == "first ICU stays"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        # Recorded once per stay: no time places it before the landmark.
        (
            {
                "primary_exposure": "age",
                "primary_exposure_source": "age",
                "declared_covariates": ("sex", "charlson"),
            },
            "web_landmark_continuous_survival_exposure_untimed",
        ),
        # Not the materializer's summary of its source.
        (
            {"primary_exposure": "charlson_first", "primary_exposure_source": "lact",
             "declared_covariates": ("age", "sex")},
            "web_landmark_continuous_survival_exposure_window_unverified",
        ),
    ],
    ids=["once_per_stay", "not_a_window_summary"],
)
def test_an_exposure_no_window_places_before_the_landmark_is_refused(tmp_path, overrides, code):
    universe, _frame = _universe(tmp_path)

    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        compile_web_scientific_runtime_projection(**_coordinates(universe, **overrides))

    assert excinfo.value.code == code


def _verified(column, *, role, aggregation, transform, window):
    binding = SimpleNamespace(
        metadata=SimpleNamespace(role=role, aggregation=aggregation),
        representation_transform=transform,
        derivation_window=(
            None
            if window is None
            else SimpleNamespace(origin=window[0], start_hours=window[1], end_hours=window[2])
        ),
    )
    return SimpleNamespace(
        sidecar=SimpleNamespace(files=(SimpleNamespace(columns={column: binding}),))
    )


@pytest.mark.parametrize(
    ("role", "aggregation", "transform", "window"),
    [
        # The window runs past the landmark: the future would enter.
        (ConceptColumnRole.NUMERIC_AGGREGATE, "max", "window_numeric_max", ("icu_admission", 0.0, 48.0)),
        (ConceptColumnRole.NUMERIC_AGGREGATE, "max", "window_numeric_max", ("icu_admission", 0.0, 12.0)),
        (ConceptColumnRole.NUMERIC_AGGREGATE, "max", "window_numeric_max", ("hospital_admission", 0.0, 24.0)),
        (ConceptColumnRole.NUMERIC_AGGREGATE, "max", "window_numeric_max", None),
        (ConceptColumnRole.COUNT, "max", "window_nonnull_count", ("icu_admission", 0.0, 24.0)),
        (ConceptColumnRole.NUMERIC_AGGREGATE, "last", "window_numeric_last", ("icu_admission", 0.0, 24.0)),
    ],
    ids=["past_the_landmark", "short_of_the_landmark", "another_origin", "no_window", "a_count", "a_last_value"],
)
def test_verified_metadata_must_prove_the_window_ends_at_the_landmark(
    tmp_path, monkeypatch, role, aggregation, transform, window
):
    universe, _frame = _universe(tmp_path)
    monkeypatch.setattr(
        continuous_projection,
        "load_verified_materialized_cohort_authority",
        lambda _path: _verified(
            "lact_max",
            role=ConceptColumnRole.NUMERIC_AGGREGATE,
            aggregation="max",
            transform="window_numeric_max",
            window=("icu_admission", 0.0, 24.0),
        ),
    )
    assert compile_web_scientific_runtime_projection(**_coordinates(universe)) is not None

    monkeypatch.setattr(
        continuous_projection,
        "load_verified_materialized_cohort_authority",
        lambda _path: _verified(
            "lact_max", role=role, aggregation=aggregation, transform=transform, window=window
        ),
    )
    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        compile_web_scientific_runtime_projection(**_coordinates(universe))
    assert excinfo.value.code == "web_landmark_continuous_survival_exposure_window_unverified"


def test_a_mean_summary_is_its_own_sealed_exposure(tmp_path):
    universe, _frame = _universe(tmp_path)

    projection = compile_web_scientific_runtime_projection(
        **_coordinates(universe, primary_exposure="lact_mean")
    )

    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.exposure_window_summary == "mean"
    assert authority.exposure_label == "Lactate, mean in hours 0 to 24"


def test_a_landmark_without_a_follow_up_cutpoint_is_refused(tmp_path):
    universe, _frame = _universe(tmp_path)

    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        compile_web_scientific_runtime_projection(
            # Follow-up from day 22 to day 28 holds neither cutpoint (days 7, 14).
            **_coordinates(universe, sensitivity_specs=(_landmark(landmark_hours=24 * 22),))
        )

    assert excinfo.value.code == "web_landmark_continuous_survival_followup_unsupported"
