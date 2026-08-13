from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
    normalize_prespecified_sensitivities,
)


def test_prespecified_sensitivity_contract_is_typed_and_immutable() -> None:
    spec = PrespecifiedSensitivitySpec(
        spec_id="timing_landmark",
        axis="timing",
        strategy="landmark",
        execution_variables=("death_time",),
        landmark_hours=24,
        require_alive_at_landmark=True,
        exclude_negative_event_times=True,
    )

    assert spec.landmark_hours == 24
    assert spec.execution_variables == ("death_time",)
    with pytest.raises(ValidationError):
        spec.landmark_hours = 48  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "spec_id": "bad_axis_strategy",
            "axis": "timing",
            "strategy": "restricted_cubic_spline",
        },
        {
            "spec_id": "landmark_without_origin",
            "axis": "timing",
            "strategy": "landmark",
        },
        {
            "spec_id": "restriction_without_variable",
            "axis": "repeated_stays",
            "strategy": "non_readmission_restriction",
        },
        {
            "spec_id": "unknown_field",
            "axis": "missing_data",
            "strategy": "complete_case",
            "answer": "guess",
        },
    ],
)
def test_invalid_sensitivity_authority_fails_at_its_owner(payload: dict) -> None:
    with pytest.raises(ValidationError):
        PrespecifiedSensitivitySpec.model_validate(payload)


def test_sensitivity_ids_are_unique() -> None:
    payload = {
        "spec_id": "missing_complete_case",
        "axis": "missing_data",
        "strategy": "complete_case",
        "execution_variables": ["age"],
    }
    with pytest.raises(ValueError, match="spec_id values must be unique"):
        normalize_prespecified_sensitivities([payload, payload])
