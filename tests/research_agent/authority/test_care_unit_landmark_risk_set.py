"""The case-proxy landmark must not turn unknown death into survival."""
import pandas as pd
import pytest

from easyicu.research_agent.authority.care_unit_landmark import case_duration_landmark_risk_set


def test_landmark_requires_ongoing_case_and_known_survival_after_boundary():
    result = case_duration_landmark_risk_set(
        duration_hours=pd.Series([25, 24, 48, 48, 48, None, 30, 30, 30]),
        death_status=pd.Series([False, False, True, True, None, False, True, False, False]),
        death_time_hours=pd.Series([None, None, 25, 24, None, None, None, 30, "invalid"]),
        landmark_hours=24,
    )
    assert result.eligible.tolist() == [True, False, True, False, False, False, False, False, False]
    assert result.closed_or_dead_at_landmark.tolist() == [False, True, False, True, False, False, False, False, False]
    assert result.unidentifiable.tolist() == [False, False, False, False, True, True, True, True, True]


def test_landmark_refuses_mismatched_row_identity_or_invalid_landmark():
    with pytest.raises(ValueError, match="landmark_input_identity_mismatch"):
        case_duration_landmark_risk_set(duration_hours=pd.Series([30], index=[0]),
                                        death_status=pd.Series([False], index=[1]),
                                        death_time_hours=pd.Series([None], index=[0]), landmark_hours=24)
    with pytest.raises(ValueError, match="landmark_hours_invalid"):
        case_duration_landmark_risk_set(duration_hours=pd.Series([30]), death_status=pd.Series([False]),
                                        death_time_hours=pd.Series([None]), landmark_hours=float("nan"))
