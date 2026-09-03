"""Leakage, censoring and metric contracts for dynamic prediction primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.methods.dynamic_prediction import (
    attach_landmark_outcomes,
    build_landmark_feature_matrix,
    evaluate_landmark_probabilities,
)


def test_landmark_features_never_read_after_prediction_time():
    trajectory = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2],
            "charttime": [1.0, 5.0, 7.0, 2.0, 8.0],
            "concept": ["map", "map", "map", "map", "map"],
            "value_num": [80.0, 70.0, 5.0, 90.0, 1.0],
        }
    )

    rows = build_landmark_feature_matrix(
        trajectory,
        feature_concepts=["map"],
        landmark_hours=[6.0],
        lookback_hours=6.0,
        aggregations=["last", "mean"],
    ).set_index("stay_id")

    assert rows.loc[1, "map__last"] == 70.0
    assert rows.loc[1, "map__mean"] == 75.0
    assert rows.loc[2, "map__last"] == 90.0
    assert 5.0 not in rows.to_numpy()
    assert 1.0 not in rows.to_numpy()


def test_landmark_feature_grid_keeps_stays_without_window_measurements():
    trajectory = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [1.0, 10.0],
            "concept": ["map", "map"],
            "value_num": [80.0, 90.0],
        }
    )

    rows = build_landmark_feature_matrix(
        trajectory,
        feature_concepts=["map"],
        landmark_hours=[0.0, 6.0],
        lookback_hours=6.0,
        aggregations=["last"],
    ).set_index(["stay_id", "prediction_time_hours"])

    assert len(rows) == 4
    assert rows.loc[(1, 6.0), "map__last"] == 80.0
    assert np.isnan(rows.loc[(1, 0.0), "map__last"])
    assert np.isnan(rows.loc[(2, 0.0), "map__last"])
    assert np.isnan(rows.loc[(2, 6.0), "map__last"])


def test_landmark_features_reject_missing_identity_or_invalid_time():
    missing_identity = pd.DataFrame(
        {
            "stay_id": [1, None],
            "charttime": [1.0, 2.0],
            "concept": ["map", "map"],
            "value_num": [80.0, 90.0],
        }
    )
    with pytest.raises(ValueError, match="non-missing stay_id"):
        build_landmark_feature_matrix(
            missing_identity,
            feature_concepts=["map"],
            landmark_hours=[6.0],
            lookback_hours=6.0,
        )

    invalid_time = missing_identity.iloc[[0]].copy()
    invalid_time["charttime"] = np.nan
    with pytest.raises(ValueError, match="finite charttime"):
        build_landmark_feature_matrix(
            invalid_time,
            feature_concepts=["map"],
            landmark_hours=[6.0],
            lookback_hours=6.0,
        )


def test_landmark_labels_keep_prevalent_and_censored_horizons_out():
    features = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "prediction_time_hours": [6.0, 6.0, 6.0, 6.0],
            "map__last": [70.0, 80.0, 75.0, 85.0],
        }
    )
    outcomes = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "event_time": [10.0, np.nan, 4.0, np.nan],
            "followup_end": [24.0, 24.0, 24.0, 8.0],
        }
    )

    labelled = attach_landmark_outcomes(
        features,
        outcomes,
        event_time_col="event_time",
        followup_end_col="followup_end",
        horizon_hours=[12.0],
    ).set_index("stay_id")

    assert labelled.loc[1, "outcome"] == 1.0
    assert labelled.loc[2, "outcome"] == 0.0
    assert labelled.loc[3, "eligible_at_landmark"] == 0
    assert np.isnan(labelled.loc[3, "outcome"])
    assert labelled.loc[4, "horizon_observed"] == 0
    assert labelled.loc[4, "eligible_at_landmark"] == 1
    assert np.isnan(labelled.loc[4, "outcome"])


def test_landmark_labels_exclude_followup_that_ended_before_landmark():
    features = pd.DataFrame(
        {"stay_id": [1], "prediction_time_hours": [6.0], "map__last": [70.0]}
    )
    outcomes = pd.DataFrame(
        {"stay_id": [1], "event_time": [np.nan], "followup_end": [4.0]}
    )

    labelled = attach_landmark_outcomes(
        features,
        outcomes,
        event_time_col="event_time",
        followup_end_col="followup_end",
        horizon_hours=[12.0],
    ).iloc[0]

    assert labelled["eligible_at_landmark"] == 0
    assert labelled["horizon_observed"] == 0
    assert np.isnan(labelled["outcome"])


def test_dynamic_metrics_are_separate_by_landmark_and_horizon():
    predictions = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "prediction_time_hours": [6.0] * 4 + [12.0] * 4,
            "target_horizon_hours": [12.0] * 8,
            "outcome": [0, 0, 1, 1, 0, 1, 0, 1],
            "predicted_probability": [
                0.1,
                0.2,
                0.8,
                0.9,
                0.1,
                0.8,
                0.2,
                0.9,
            ],
        }
    )

    result = evaluate_landmark_probabilities(predictions, calibration_bins=2)

    assert result.metrics.shape[0] == 2
    assert result.metrics["auroc"].tolist() == [1.0, 1.0]
    assert set(result.metrics["prediction_time_hours"]) == {6.0, 12.0}
    assert not result.calibration.empty


def test_dynamic_metrics_fail_closed_on_unobserved_or_one_class_rows():
    predictions = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "prediction_time_hours": [6.0, 6.0],
            "target_horizon_hours": [12.0, 12.0],
            "outcome": [0, np.nan],
            "predicted_probability": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="observed labels"):
        evaluate_landmark_probabilities(predictions)

    predictions["outcome"] = [0, 0]
    with pytest.raises(ValueError, match="both outcome classes"):
        evaluate_landmark_probabilities(predictions)


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("prediction_time_hours", np.nan, "prediction times"),
        ("prediction_time_hours", -1.0, "prediction times"),
        ("target_horizon_hours", np.nan, "target horizons"),
        ("target_horizon_hours", 0.0, "target horizons"),
    ],
)
def test_dynamic_metrics_reject_invalid_group_coordinates(column, bad_value, message):
    predictions = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "prediction_time_hours": [6.0, 6.0],
            "target_horizon_hours": [12.0, 12.0],
            "outcome": [0, 1],
            "predicted_probability": [0.1, 0.9],
        }
    )
    predictions.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=message):
        evaluate_landmark_probabilities(predictions)
