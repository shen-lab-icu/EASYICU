from __future__ import annotations

from easyicu.research_agent.planning.sensitivity_authority import (
    normalize_prespecified_sensitivities,
)
from easyicu.webserver.agent_pipeline_runs import (
    _runtime_projection_sensitivity_specs,
)


def test_landmark_runtime_adds_plan_owned_rcs_without_mutating_user_specs() -> None:
    configured = normalize_prespecified_sensitivities(
        [
            {
                "spec_id": "landmark_24h",
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
        ]
    )

    runtime = _runtime_projection_sensitivity_specs(
        configured,
        primary_exposure_source="lact",
    )

    assert [item.spec_id for item in configured] == ["landmark_24h"]
    assert [item.spec_id for item in runtime] == [
        "landmark_24h",
        "easyicu_auto_primary_exposure_rcs",
    ]
    assert runtime[-1].execution_variables == ("lact",)


def test_explicit_functional_form_is_never_replaced() -> None:
    configured = normalize_prespecified_sensitivities(
        [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
            },
            {
                "spec_id": "linear_exposure",
                "axis": "functional_form",
                "strategy": "linear_per_unit",
                "execution_variables": ["lact"],
            },
        ]
    )

    assert _runtime_projection_sensitivity_specs(
        configured,
        primary_exposure_source="lact",
    ) == configured


def test_ordinal_landmark_exposure_never_receives_continuous_rcs() -> None:
    configured = normalize_prespecified_sensitivities(
        [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
            }
        ]
    )

    assert _runtime_projection_sensitivity_specs(
        configured,
        primary_exposure_source="aki_stage",
        primary_exposure_dtype="int64",
    ) == configured
