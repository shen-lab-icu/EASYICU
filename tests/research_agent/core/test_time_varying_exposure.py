"""Contracts for predictable early time-varying exposure panels."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.time_varying_exposure import (
    TimeVaryingExposureError,
    build_early_running_max_exposure_panel,
)


def _followup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "hospital_death": [0, 1, 1, 0],
            "death_time_hours": [math.nan, 8.0, 0.0, math.nan],
            "hospital_followup_time_hours": [50.0, 8.0, 0.0, 0.0],
        }
    )


def _trajectory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 2, 2, 2, 3, 4, 1],
            "charttime": [2.0, 6.0, 24.0, 25.0, 4.0, 8.0, 9.0, 0.0, 0.0, 7.0],
            "concept": ["lact"] * 9 + ["other"],
            "value_num": [2.0, 4.0, 3.0, 99.0, 5.0, 99.0, 8.0, 7.0, 2.0, 4.0],
            "evidence_state": ["direct_observed"] * 10,
        }
    )


def test_panel_preserves_early_events_and_is_predictable() -> None:
    result = build_early_running_max_exposure_panel(
        _trajectory(), _followup(), exposure_concept="lact"
    )

    stay_one = result.panel.loc[result.panel["stay_id"] == 1]
    assert stay_one[
        ["interval_start_hours", "interval_stop_hours"]
    ].values.tolist() == [[0.0, 2.0], [2.0, 6.0], [6.0, 24.0], [24.0, 50.0]]
    assert stay_one["exposure_state"].tolist() == [
        "unmeasured",
        "observed_running_max",
        "observed_running_max",
        "observed_running_max",
    ]
    assert stay_one["exposure_running_max"].tolist()[1:] == [2.0, 4.0, 4.0]

    stay_two = result.panel.loc[result.panel["stay_id"] == 2]
    assert stay_two["interval_stop_hours"].tolist() == [4.0, 8.0]
    assert stay_two["exposure_running_max"].tolist()[1] == 5.0
    assert stay_two["hospital_death"].tolist() == [0, 1]

    zero_time = result.panel.loc[result.panel["stay_id"] == 3].iloc[0]
    assert zero_time["hospital_death"] == 1
    assert zero_time["source_event_time_hours"] == 0.0
    assert zero_time["interval_stop_hours"] == pytest.approx(1e-6)
    assert bool(zero_time["zero_time_event_epsilon_applied"]) is True
    assert zero_time["exposure_state"] == "unmeasured"

    assert result.exclusions.to_dict("records") == [
        {"stay_id": 4, "reason_code": "zero_hospital_followup_without_event"}
    ]
    assert result.receipt["counts"]["input_hospital_deaths"] == 2
    assert result.receipt["counts"]["panel_hospital_deaths"] == 2
    assert result.receipt["counts"]["zero_time_hospital_deaths"] == 1
    assert result.receipt["counts"]["outside_early_window_rows_excluded"] == 1


def test_panel_never_uses_owner_locf_as_a_new_measurement() -> None:
    trajectory = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [3.0, 7.0],
            "concept": ["lact", "lact"],
            "value_num": [2.0, 8.0],
            "evidence_state": ["direct_observed", "owner_locf_available"],
        }
    )
    followup = pd.DataFrame(
        {
            "stay_id": [1],
            "hospital_death": [0],
            "death_time_hours": [math.nan],
            "hospital_followup_time_hours": [12.0],
        }
    )

    result = build_early_running_max_exposure_panel(
        trajectory, followup, exposure_concept="lact"
    )

    assert result.panel["interval_stop_hours"].tolist() == [3.0, 12.0]
    assert result.panel["exposure_running_max"].tolist()[1] == 2.0
    assert result.receipt["counts"]["direct_observed_rows"] == 1


def test_panel_rejects_followup_that_cannot_represent_survival_time() -> None:
    invalid = _followup()
    invalid.loc[1, "death_time_hours"] = 9.0

    with pytest.raises(TimeVaryingExposureError, match="cannot exceed"):
        build_early_running_max_exposure_panel(
            _trajectory(), invalid, exposure_concept="lact"
        )
