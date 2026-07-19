"""Unit tests for the deterministic temporal-feature primitives.

These exercise the onset / incident / landmark constructors on small synthetic
trajectories so the reusable building blocks are correct in CI — the agent
composes them instead of re-implementing (and mis-implementing) the same logic
per run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.methods import temporal_features as T


def _traj(rows):
    return pd.DataFrame(
        rows, columns=["stay_id", "charttime", "concept", "value_num", "value_str"]
    )


def test_onset_times_first_threshold_crossing_not_first_record():
    # stay 1: MAP 80 -> 60 (<65 at t=2); stay 2: MAP 70 only (never <65)
    traj = _traj(
        [
            (1, 0.0, "map", 80.0, "80"),
            (1, 2.0, "map", 60.0, "60"),
            (1, 5.0, "map", 58.0, "58"),
            (2, 1.0, "map", 70.0, "70"),
        ]
    )
    out = T.onset_times(traj, "map", op="<", threshold=65.0)
    assert list(out.columns) == ["stay_id", "map_onset_time"]
    assert out.loc[out.stay_id == 1, "map_onset_time"].iloc[0] == 2.0
    # stay 2 never crosses -> absent
    assert (out.stay_id == 2).sum() == 0


def test_onset_times_skips_stage_zero_records():
    # KDIGO 0,0,1,2 -> onset at the first >=1 (t=5), NOT the first record (t=1)
    traj = _traj(
        [
            (1, 1.0, "aki", 0.0, "0"),
            (1, 3.0, "aki", 0.0, "0"),
            (1, 5.0, "aki", 1.0, "1"),
            (1, 8.0, "aki", 2.0, "2"),
        ]
    )
    out = T.onset_times(traj, "aki", op=">=", threshold=1.0)
    assert out.loc[out.stay_id == 1, "aki_onset_time"].iloc[0] == 5.0


def test_incident_outcome_cohort_classifies_prevalent_incident_eventfree():
    traj = _traj(
        [
            # stay 1: vent at t=1, AKI onset at t=6  -> incident
            (1, 1.0, "mech_vent", 1.0, "1"),
            (1, 6.0, "aki", 1.0, "1"),
            # stay 2: AKI at t=1 BEFORE vent at t=3   -> prevalent (reverse order)
            (2, 1.0, "aki", 2.0, "2"),
            (2, 3.0, "mech_vent", 1.0, "1"),
            # stay 3: vent at t=2, AKI never crosses (only stage 0) -> event_free
            (3, 2.0, "mech_vent", 1.0, "1"),
            (3, 4.0, "aki", 0.0, "0"),
            # stay 4: AKI onset but never ventilated -> no_index
            (4, 5.0, "aki", 1.0, "1"),
        ]
    )
    out = T.incident_outcome_cohort(
        traj, outcome_concept="aki", index_concept="mech_vent",
        outcome_threshold=1.0, index_threshold=1.0,
    )
    by = out.set_index("stay_id")
    assert by.loc[1, "classification"] == "incident"
    assert by.loc[1, "at_risk"] == 1 and by.loc[1, "incident"] == 1.0
    assert by.loc[1, "time_to_event"] == 5.0  # 6 - 1
    assert by.loc[2, "classification"] == "prevalent"
    assert by.loc[2, "at_risk"] == 0 and np.isnan(by.loc[2, "incident"])
    assert by.loc[3, "classification"] == "event_free"
    assert by.loc[3, "at_risk"] == 1 and by.loc[3, "incident"] == 0.0
    assert by.loc[4, "classification"] == "no_index"
    assert by.loc[4, "at_risk"] == 0


def test_landmark_cohort_excludes_pre_landmark_events_and_flags_exposure():
    traj = _traj(
        [
            # stay 1: AKI at t=3 (before 6h landmark) -> ineligible
            (1, 3.0, "aki", 1.0, "1"),
            # stay 2: AKI at t=10 (after landmark) -> eligible, event
            (2, 10.0, "aki", 1.0, "1"),
            # stay 3: never AKI -> eligible, no event
            (3, 2.0, "aki", 0.0, "0"),
        ]
    )
    exposure = T.onset_times(
        _traj(
            [
                (2, 4.0, "norepi_rate", 0.1, "0.1"),  # exposed before landmark
                (3, 9.0, "norepi_rate", 0.2, "0.2"),  # exposed after landmark
            ]
        ),
        "norepi_rate", op=">", threshold=0.0,
    )
    out = T.landmark_cohort(
        traj, outcome_concept="aki", landmark_hours=6.0, exposure_onset=exposure,
    )
    by = out.set_index("stay_id")
    assert by.loc[1, "eligible_at_landmark"] == 0  # event before landmark
    assert by.loc[2, "eligible_at_landmark"] == 1
    assert by.loc[2, "event_after_landmark"] == 1
    assert by.loc[2, "exposed_by_landmark"] == 1  # exposed at t=4 <= 6
    assert by.loc[3, "eligible_at_landmark"] == 1
    assert by.loc[3, "event_after_landmark"] == 0
    assert by.loc[3, "exposed_by_landmark"] == 0  # exposed at t=9 > 6 (not counted)


def test_onset_times_present_mode_for_categorical_concept():
    # mech_vent is categorical (value_num all NaN, value_str = invasive/none).
    # present-mode onset = first non-negative recorded value.
    traj = _traj(
        [
            (1, 1.0, "mech_vent", np.nan, "none"),       # negative token -> skip
            (1, 3.0, "mech_vent", np.nan, "invasive"),   # onset here
            (1, 5.0, "mech_vent", np.nan, "invasive"),
            (2, 2.0, "mech_vent", np.nan, "noninvasive"),
        ]
    )
    out = T.onset_times(traj, "mech_vent", op="present")
    by = out.set_index("stay_id")["mech_vent_onset_time"]
    assert by.loc[1] == 3.0
    assert by.loc[2] == 2.0
    # restrict to invasive only
    inv = T.onset_times(traj, "mech_vent", positive_values={"invasive"})
    assert inv.set_index("stay_id")["mech_vent_onset_time"].loc[1] == 3.0
    assert (inv.stay_id == 2).sum() == 0  # stay 2 only noninvasive


def test_incident_cohort_with_categorical_index_present_mode():
    traj = _traj(
        [
            (1, 1.0, "mech_vent", np.nan, "invasive"),
            (1, 6.0, "aki", 1.0, "1"),
        ]
    )
    out = T.incident_outcome_cohort(
        traj, outcome_concept="aki", index_concept="mech_vent", index_op="present",
    )
    assert out.set_index("stay_id").loc[1, "classification"] == "incident"


def test_onset_times_empty_when_concept_absent():
    traj = _traj([(1, 1.0, "map", 80.0, "80")])
    out = T.onset_times(traj, "lactate", op=">", threshold=2.0)
    assert out.empty and list(out.columns) == ["stay_id", "lactate_onset_time"]
