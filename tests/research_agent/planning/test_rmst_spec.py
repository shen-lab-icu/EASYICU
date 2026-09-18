"""Contract tests for the typed RMST step coordinates."""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.rmst import RMSTSpec


def _spec(**overrides):
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


def test_rmst_spec_keeps_exact_coordinates_and_defaults() -> None:
    spec = _spec()
    assert spec.group_levels == ("high", "low")
    assert spec.time_unit == "days"


def test_rmst_spec_rejects_shared_columns() -> None:
    with pytest.raises(ValueError, match="distinct"):
        _spec(event_column="time_days")


def test_rmst_spec_rejects_duplicate_group_levels() -> None:
    with pytest.raises(ValueError, match="distinct labels"):
        _spec(group_levels=["high", "high"])


def test_rmst_spec_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError):
        _spec(tau=0.0)


def test_rmst_spec_reserves_zero_for_censoring() -> None:
    with pytest.raises(ValueError, match="censor code 0"):
        _spec(event_code=0.0)


def test_rmst_spec_is_frozen_and_forbids_unknown_coordinates() -> None:
    spec = _spec()
    with pytest.raises(ValueError):
        spec.tau = 90.0
    with pytest.raises(ValueError):
        _spec(competing_event_column="discharge")


@pytest.mark.parametrize("field", ["tau", "event_code"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_rmst_coordinates_must_be_finite(field, value):
    with pytest.raises(ValueError):
        _spec(**{field: value})


def test_rmst_group_levels_cannot_be_mutated_after_validation():
    spec = _spec()
    with pytest.raises(TypeError):
        spec.group_levels[0] = spec.group_levels[1]
