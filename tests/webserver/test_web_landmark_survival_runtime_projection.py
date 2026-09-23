"""Web StudyContext -> sealed landmark survival suite projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.outcome_availability import FIXED_HORIZON_MORTALITY_ENDPOINTS
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkSurvivalRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    run_landmark_survival_suite,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.webserver.landmark_survival_runtime_projection import (
    compile_landmark_survival_runtime_projection,
)
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


def _universe(tmp_path, *, n: int = 600):
    rng = np.random.default_rng(20260922)
    ventilated = rng.binomial(1, 0.45, size=n)
    onset = np.where(
        ventilated == 1,
        rng.choice([-2.0, 3.0, 9.0, 18.0, 30.0], size=n, p=[0.05, 0.4, 0.3, 0.2, 0.05]),
        np.nan,
    )
    risk = 1.0 / (1.0 + np.exp(-(-1.4 + 0.8 * ventilated)))
    died = rng.binomial(1, risk, size=n)
    followup = np.where(died == 1, rng.uniform(0.5, 27.5, size=n), 28.0)
    frame = pd.DataFrame(
        {
            "mech_vent_max": ventilated.astype("int64"),
            "mech_vent_first_time": onset,
            "mort_28d": died.astype("int64"),
            "followup_days_28d": followup,
            "age": rng.normal(63.0, 14.0, size=n),
            "sex": rng.choice(["F", "M"], size=n),
            "charlson_first": rng.poisson(3.0, size=n).astype(float),
            "sofa2_max": rng.integers(0, 16, size=n).astype(float),
        }
    )
    path = tmp_path / "survival_universe.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def _coordinates(universe, **overrides):
    coordinates = {
        "study": _study(),
        "sensitivity_specs": (_landmark(),),
        "primary_exposure": "mech_vent_max",
        "primary_exposure_source": "mech_vent",
        "target_outcome": "mort_28d",
        "declared_covariates": ("age", "sex", "charlson", "sofa2_max"),
        "covariate_operationalizations": {"charlson": "charlson_first"},
        "target_is_event_status": True,
        "universe_path": universe,
        "scientific_configuration_sha256": "a" * 64,
    }
    coordinates.update(overrides)
    return coordinates


def test_survival_family_landmark_compiles_the_sealed_suite_and_executes(tmp_path):
    universe, frame = _universe(tmp_path)

    projection = compile_web_scientific_runtime_projection(**_coordinates(universe))

    assert projection is not None
    assert projection.analysis_only_execution is False
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    assert authority.exposure_status_column == "mech_vent_max"
    assert authority.exposure_onset_column == "mech_vent_first_time"
    assert authority.event_column == "mort_28d"
    assert authority.followup_time_column == "followup_days_28d"
    assert authority.endpoint_horizon_days == 28.0
    assert authority.landmark_hours == 24.0
    assert authority.adjustment_columns == ("age", "sex", "charlson_first", "sofa2_max")
    assert authority.categorical_adjustment_columns == ("sex",)
    assert authority.time_varying_interval_cutpoints_days == (7.0, 14.0)
    assert authority.proportional_hazards_policy == "block_paper_authorization"
    assert authority.development_execution_only_allowed is False
    assert authority.protocol_content_sha256 == "a" * 64
    # The same configuration always signs the same contract.
    again = compile_web_scientific_runtime_projection(**_coordinates(universe))
    assert again is not None and again.projection_sha256 == projection.projection_sha256

    summary = run_landmark_survival_suite(
        frame=frame,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "out",
        input_product="table:analysis_cohort",
        input_evidence_id="cohort",
        input_sha256="b" * 64,
    )
    assert summary["status"] == "ok"
    assert set(summary["output_files"]) == set(authority.analysis_plan_outputs)


def test_sealed_survival_endpoint_and_exposure_bind_the_run_coordinates(tmp_path):
    universe, _frame = _universe(tmp_path)
    projection = compile_landmark_survival_runtime_projection(**_coordinates(universe))
    assert projection is not None
    authorities = ScientificRuntimeAuthorities.load(
        trajectory=None, current_case=projection.authority
    )
    binary = EndpointSpec(
        name="mort_28d",
        kind="binary",
        absence_semantics="no_absent_rows",
        levels=[0, 1],
    )

    endpoint, exposure, prefs = authorities.bind_run_inputs(
        endpoint=binary, primary_exposure="mech_vent_max", user_preferences=None
    )
    assert prefs is None
    assert endpoint.kind == "time_to_event"
    assert endpoint.time_column == "followup_days_28d"
    assert exposure == "mech_vent_max"
    endpoint, exposure, _prefs = authorities.bind_run_inputs(
        endpoint=None, primary_exposure=None, user_preferences=None
    )
    assert endpoint.event_column == "mort_28d" and exposure == "mech_vent_max"

    other = EndpointSpec(
        name="death",
        kind="binary",
        absence_semantics="no_absent_rows",
        levels=[0, 1],
    )
    with pytest.raises(ValueError, match="conflicts with the sealed survival"):
        authorities.bind_run_inputs(endpoint=other, primary_exposure=None, user_preferences=None)
    with pytest.raises(ValueError, match="primary exposure conflicts"):
        authorities.bind_run_inputs(
            endpoint=None, primary_exposure="vaso_ind_max", user_preferences=None
        )
    # Without a sealed suite the caller's coordinates pass through untouched.
    bare = ScientificRuntimeAuthorities(trajectory=None, current_case=None)
    assert bare.bind_run_inputs(
        endpoint=binary, primary_exposure="x", user_preferences={"covariates": []}
    ) == (binary, "x", {"covariates": []})


def test_survival_family_without_landmark_keeps_other_routes(tmp_path):
    universe, _frame = _universe(tmp_path)
    assert (
        compile_landmark_survival_runtime_projection(
            **_coordinates(universe, sensitivity_specs=())
        )
        is None
    )
    association = _study()
    association["analysis_design"]["analysis_family"] = "association_study"
    assert (
        compile_landmark_survival_runtime_projection(
            **_coordinates(universe, study=association)
        )
        is None
    )


@pytest.mark.parametrize(
    ("overrides", "code", "detail_key"),
    [
        (
            {"target_outcome": "death"},
            "web_landmark_survival_endpoint_unsupported",
            "supported_endpoints",
        ),
        (
            {"study": _study(covariate_selection="planner_selectable")},
            "web_landmark_survival_authority_incomplete",
            "missing_fields",
        ),
        (
            {
                "sensitivity_specs": (
                    _landmark(
                        observation_duration_variable="los_icu",
                        observation_duration_unit="days",
                    ),
                )
            },
            "web_landmark_survival_followup_binding_mismatch",
            "required_observation_duration_variable",
        ),
        (
            {"sensitivity_specs": (_landmark(landmark_hours=24 * 30),)},
            "web_landmark_survival_landmark_unsupported",
            "endpoint_horizon_days",
        ),
        (
            {
                "study": _study(
                    analysis_design={
                        "analysis_family": "survival",
                        "analysis_unit": "icu_stay",
                        "variance_estimator": "cluster_robust",
                        "cluster_unit": "patient",
                    }
                )
            },
            "web_landmark_survival_design_unsupported",
            "supported_variance_estimator",
        ),
        (
            {"primary_exposure": "sofa2_max", "primary_exposure_source": "sofa2"},
            "web_landmark_survival_exposure_incompatible",
            "exposure_kind",
        ),
    ],
)
def test_survival_projection_fails_closed_on_unsupported_coordinates(
    tmp_path, overrides, code, detail_key
):
    universe, _frame = _universe(tmp_path)
    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        compile_landmark_survival_runtime_projection(
            **_coordinates(universe, **overrides)
        )
    assert excinfo.value.code == code
    assert detail_key in excinfo.value.details


def test_survival_projection_requires_the_onset_companion_column(tmp_path):
    universe, frame = _universe(tmp_path)
    frame.drop(columns=["mech_vent_first_time"]).to_parquet(universe, index=False)
    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        compile_landmark_survival_runtime_projection(**_coordinates(universe))
    assert excinfo.value.code == "web_scientific_runtime_columns_missing"
    assert excinfo.value.details["missing_columns"] == ["mech_vent_first_time"]


def test_fixed_horizon_vocabulary_pairs_each_event_with_its_followup():
    for event, endpoint in FIXED_HORIZON_MORTALITY_ENDPOINTS.items():
        assert endpoint.event_concept == event
        assert endpoint.followup_concept == f"followup_days_{endpoint.horizon_days}d"
        assert all(0 < cut < endpoint.horizon_days for cut in endpoint.time_varying_cutpoints_days)
        assert tuple(sorted(endpoint.time_varying_cutpoints_days)) == (
            endpoint.time_varying_cutpoints_days
        )
