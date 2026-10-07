"""A row at a window's end hour was charted after the window.

Chart times are floored to the hourly grid (``io.ts_utils.round_to_interval``),
so a row at hour ``h`` was charted in ``[h, h + 1)``.  Every reader of a window
``(start, end)`` keeps the rows charted in ``[start, end)``, and a landmark ``L``
splits a stay into what was charted before it and what was charted at or after
it.  Read as inside, the row at ``end`` or at ``L`` lets an hour charted after
the window, or after the prediction, decide a summary, a cohort or a label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.cohort import materializer, primitives
from easyicu.research_agent.cohort.schema import _build_cohort_with_flow
from easyicu.research_agent.methods import temporal_features
from easyicu.research_agent.methods.dynamic_prediction import (
    attach_landmark_outcomes,
    build_landmark_feature_matrix,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.scores.aki_strict import summarize_strict_kdigo_window


def _trajectory(rows):
    return pd.DataFrame(
        rows, columns=["stay_id", "charttime", "concept", "value_num", "value_str"]
    )


@pytest.mark.parametrize(
    ("bounds", "kept"),
    [
        ((0.0, 24.0), [0.0, 23.0]),
        # A window before admission ends where the stay's first hour begins.
        ((-6.0, 0.0), [-6.0, -1.0]),
    ],
)
def test_the_cohort_window_keeps_the_rows_charted_inside_it(bounds, kept):
    rows = pd.DataFrame(
        {"stay_id": 1, "charttime": [-7.0, -6.0, -1.0, 0.0, 23.0, 24.0], "map": 70.0}
    )

    assert primitives.window(rows, *bounds)["charttime"].tolist() == kept


def test_the_materializer_reads_its_windows_through_the_primitive():
    assert materializer._window is primitives.window


def test_a_strict_kdigo_window_is_staged_by_the_rows_charted_inside_it():
    negative = {
        "aki_stage_creat_reference": 0,
        "aki_stage_uo_reference": 0,
        "aki_stage_rrt_reference": 0,
        "creatinine_evidence_status": "negative",
        "urine_evidence_status": "negative",
        "rrt_evidence_status": "negative",
    }
    positive = {
        **negative,
        "aki_stage_creat_reference": 3,
        "creatinine_evidence_status": "positive",
    }
    frame = pd.DataFrame(
        [
            {"stay_id": 1, "charttime": 23.0, **negative},
            {"stay_id": 1, "charttime": 24.0, **positive},
            {"stay_id": 2, "charttime": 0.0, **positive},
        ]
    )

    summary = summarize_strict_kdigo_window(
        frame, id_column="stay_id", window_start_hours=0.0, window_end_hours=24.0
    ).set_index("stay_id")

    # Stay 1's stage 3 was charted in the hour after the window.
    assert summary.loc[1, "aki_ascertainment"] == "negative_complete"
    assert summary.loc[1, "aki_stage_strict"] == 0
    assert summary.loc[1, "kidney_window_row_count"] == 1
    assert summary.loc[2, "aki_ascertainment"] == "positive"
    assert summary.loc[2, "aki_stage_strict"] == 3


def test_a_crossing_at_the_window_end_is_no_onset_inside_it():
    trajectory = _trajectory(
        [
            (1, 24.0, "lact", 4.0, "4"),
            (2, 23.0, "lact", 4.0, "4"),
            (3, 0.0, "lact", 4.0, "4"),
        ]
    )

    onset = temporal_features.onset_times(
        trajectory, "lact", op=">", threshold=2.0, window=(0.0, 24.0)
    )

    assert onset.set_index("stay_id")["lact_onset_time"].to_dict() == {2: 23.0, 3: 0.0}


def test_a_landmark_splits_a_stay_at_its_own_hour():
    trajectory = _trajectory(
        [
            (1, 5.0, "aki", 1.0, "1"),
            (2, 6.0, "aki", 1.0, "1"),
            (3, 2.0, "aki", 0.0, "0"),
        ]
    )
    exposure = temporal_features.onset_times(
        _trajectory(
            [
                (2, 5.0, "norepi_rate", 0.1, "0.1"),
                (3, 6.0, "norepi_rate", 0.2, "0.2"),
            ]
        ),
        "norepi_rate",
        op=">",
        threshold=0.0,
    )

    by_stay = temporal_features.landmark_cohort(
        trajectory, outcome_concept="aki", landmark_hours=6.0, exposure_onset=exposure
    ).set_index("stay_id")

    # Stay 1's AKI was charted in [5, 6), before the landmark: not at risk.
    assert by_stay.loc[1, "eligible_at_landmark"] == 0
    # Stay 2's AKI was charted in [6, 7), after it: at risk, an event at time 0.
    assert by_stay.loc[2, "eligible_at_landmark"] == 1
    assert by_stay.loc[2, "event_after_landmark"] == 1
    assert by_stay.loc[2, "time_from_landmark"] == 0.0
    assert by_stay.loc[2, "exposed_by_landmark"] == 1
    # Stay 3's exposure was charted in [6, 7): it began after the landmark.
    assert by_stay.loc[3, "eligible_at_landmark"] == 1
    assert by_stay.loc[3, "exposed_by_landmark"] == 0


def test_landmark_features_read_the_lookback_charted_before_the_prediction():
    trajectory = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "charttime": [-1.0, 0.0, 5.0, 6.0],
            "concept": ["map"] * 4,
            "value_num": [1.0, 60.0, 80.0, 999.0],
        }
    )

    rows = build_landmark_feature_matrix(
        trajectory,
        feature_concepts=["map"],
        landmark_hours=[6.0],
        lookback_hours=6.0,
        aggregations=["last", "min", "max", "mean"],
    ).set_index("stay_id")

    # [0, 6): hour 0 opens the lookback; hour 6 was charted after the prediction.
    assert rows.loc[1, "map__last"] == 80.0
    assert rows.loc[1, "map__min"] == 60.0
    assert rows.loc[1, "map__max"] == 80.0
    assert rows.loc[1, "map__mean"] == 70.0


def test_an_event_at_the_landmark_hour_is_in_its_horizon_and_one_at_its_end_is_not():
    features = pd.DataFrame(
        {"stay_id": [1, 2, 3, 4], "prediction_time_hours": [6.0] * 4}
    )
    outcomes = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "event_time": [6.0, 18.0, 5.0, 17.0],
            "followup_end": [30.0] * 4,
        }
    )

    labelled = attach_landmark_outcomes(
        features,
        outcomes,
        event_time_col="event_time",
        followup_end_col="followup_end",
        horizon_hours=[12.0],
    ).set_index("stay_id")

    # The horizon is [6, 18).
    assert labelled.loc[1, "eligible_at_landmark"] == 1
    assert labelled.loc[1, "outcome"] == 1.0
    assert labelled.loc[2, "horizon_observed"] == 1
    assert labelled.loc[2, "outcome"] == 0.0
    assert labelled.loc[3, "eligible_at_landmark"] == 0
    assert np.isnan(labelled.loc[3, "outcome"])
    assert labelled.loc[4, "outcome"] == 1.0


def _death_predicate(op, value):
    return ConceptPredicate(
        concept_id="death",
        time_window=TimeWindow(
            anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=24.0
        ),
        aggregation="any",
        op=op,
        value=value,
    )


@pytest.mark.parametrize(
    "definition",
    [
        CohortDefinition(
            name="no death in the window",
            inclusion=(),
            exclusion=(_death_predicate("==", 1),),
        ),
        CohortDefinition(
            name="survived the window",
            inclusion=(_death_predicate("==", 0),),
            exclusion=(),
        ),
    ],
    ids=["occurrence", "absence"],
)
def test_an_event_at_the_window_end_lies_outside_its_predicate_window(definition):
    universe = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "death": [1, 1, 0],
            "death_time": [23.0, 24.0, None],
        }
    )

    cohort, flow = _build_cohort_with_flow(definition, universe)

    # Stay 2's death was charted in [24, 25), after the window.
    assert list(cohort["stay_id"]) == [2, 3]
    assert flow[-1]["event_time_end_hours"] == 24.0
