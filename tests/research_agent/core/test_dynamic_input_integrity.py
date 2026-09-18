import pandas as pd
import pytest

from easyicu.research_agent.methods.dynamic_prediction import (
    attach_landmark_outcomes, build_landmark_feature_matrix,
)


def test_nonempty_invalid_event_time_is_not_a_nonevent():
    features = pd.DataFrame({"stay_id": [1], "prediction_time_hours": [12]})
    outcomes = pd.DataFrame({"stay_id": [1], "event": ["broken"], "end": [72]})
    with pytest.raises(ValueError, match="event"):
        attach_landmark_outcomes(features, outcomes, event_time_col="event",
            followup_end_col="end", horizon_hours=[24])


def test_conflicting_last_measurement_ties_are_not_arbitrarily_resolved():
    trajectory = pd.DataFrame({"stay_id": [1, 1], "charttime": [2, 2],
        "concept": ["x", "x"], "value_num": [10, 20]})
    for frame in (trajectory, trajectory.iloc[::-1]):
        with pytest.raises(ValueError, match="conflicting"):
            build_landmark_feature_matrix(frame, feature_concepts=["x"],
                landmark_hours=[3], lookback_hours=3, aggregations=["last"])
