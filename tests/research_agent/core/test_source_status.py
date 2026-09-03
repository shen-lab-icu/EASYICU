"""Tests for case-neutral sparse binary event reconciliation."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.methods.source_status import (
    reconcile_binary_event_presence,
    reconcile_conditional_event_time,
    reconcile_measurement_source_status,
)


def test_positive_only_event_triad_retains_negative_rows():
    frame = pd.DataFrame(
        {
            "event_n": [0, 2, 0, 1],
            "event_measured": [0, 1, 0, 1],
            "event_max": [None, 1, 0, 1],
        }
    )

    result = reconcile_binary_event_presence(
        frame,
        count_column="event_n",
        measured_column="event_measured",
        representative_column="event_max",
    )

    assert result.values.tolist() == [0, 1, 0, 1]
    assert result.row_status.tolist() == [
        "event_absent",
        "event_present",
        "event_absent",
        "event_present",
    ]
    assert result.audit["indicator_semantics"] == "binary_event_presence"
    assert result.audit["discordant_n"] == 0
    assert result.status_table["count"].sum() == len(frame)
    assert result.status_table["percentage"].sum() == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("count", "measured", "representative"),
    [
        ([0, 1], [0, 0], [None, 1]),
        ([0, 1], [0, 1], [None, None]),
        ([0, 1], [0, 1], [1, 1]),
        ([0, 1.5], [0, 1], [None, 1]),
        ([0, 1], [0, 1], [None, 2]),
        ([0, 1], [0, 1], [None, "yes"]),
    ],
)
def test_sparse_event_triad_fails_closed_on_any_contradiction(
    count, measured, representative
):
    frame = pd.DataFrame(
        {
            "event_n": count,
            "event_measured": measured,
            "event_max": representative,
        }
    )

    with pytest.raises(ValueError, match="invalid or discordant"):
        reconcile_binary_event_presence(
            frame,
            count_column="event_n",
            measured_column="event_measured",
            representative_column="event_max",
        )


def test_sparse_event_helper_never_guesses_a_missing_column():
    frame = pd.DataFrame({"event_n": [0, 1], "event_measured": [0, 1]})

    with pytest.raises(ValueError, match="columns missing"):
        reconcile_binary_event_presence(
            frame,
            count_column="event_n",
            measured_column="event_measured",
            representative_column="event_max",
        )


def test_conditional_event_time_distinguishes_absence_from_missingness():
    frame = pd.DataFrame(
        {
            "death": [0, 1, 1, 1],
            "death_time": [None, 48.0, None, -2.0],
        }
    )

    result = reconcile_conditional_event_time(
        frame,
        event_status_column="death",
        event_time_column="death_time",
    )

    assert result.row_status.tolist() == [
        "event_absent_not_applicable",
        "event_time_observed",
        "event_time_missing",
        "event_time_before_origin",
    ]
    assert result.audit == {
        "observation_semantics": "conditional_event_time",
        "event_status_column": "death",
        "event_time_column": "death_time",
        "minimum_time": 0.0,
        "n_total": 4,
        "eligible_event_n": 3,
        "not_applicable_event_absent_n": 1,
        "observed_event_time_n": 2,
        "missing_event_time_n": 1,
        "before_origin_n": 1,
        "contradictory_event_absent_with_time_n": 0,
    }
    assert result.status_table["count"].sum() == len(frame)


def test_conditional_event_time_rejects_time_on_event_absent_row():
    frame = pd.DataFrame(
        {
            "death": [0, 1],
            "death_time": [12.0, 48.0],
        }
    )

    with pytest.raises(ValueError, match="event status is absent"):
        reconcile_conditional_event_time(
            frame,
            event_status_column="death",
            event_time_column="death_time",
        )


def test_measurement_source_status_uses_a_closed_audit_only_partition():
    frame = pd.DataFrame(
        {
            "marker_n": [1, 0, 2, 0],
            "marker_measured": [1, 0, 1, 0],
            "marker_max": [2.5, None, None, None],
        }
    )

    result = reconcile_measurement_source_status(
        frame,
        measured_column="marker_measured",
        count_column="marker_n",
        value_column="marker_max",
    )

    assert result.row_status.tolist() == [
        "valid observed",
        "no source",
        "measured/source present but summary missing",
        "no source",
    ]
    assert result.provenance_receipt == {
        "measured_column": "marker_measured",
        "count_column": "marker_n",
        "status": "checked",
        "comparison_n": 4,
        "invalid_pair_n": 0,
        "discordant_n": 0,
        "role": "audit_only",
    }
    assert result.status_table["count"].sum() == len(frame)
    assert result.status_table["percentage"].sum() == pytest.approx(100.0)
    assert result.audit["valid_observed_n"] == 1
    assert result.audit["no_source_n"] == 2
    assert result.audit["measured_source_present_summary_missing_n"] == 1


def test_measurement_source_status_fails_on_value_without_source():
    frame = pd.DataFrame(
        {
            "marker_n": [0, 1],
            "marker_measured": [0, 1],
            "marker_max": [3.2, 1.4],
        }
    )

    with pytest.raises(ValueError, match="source status is contradictory"):
        reconcile_measurement_source_status(
            frame,
            measured_column="marker_measured",
            count_column="marker_n",
            value_column="marker_max",
        )


@pytest.mark.parametrize(
    ("count", "measured"),
    [([0, 1], [1, 1]), ([0, 1.5], [0, 1]), ([0, 1], [0, 2])],
)
def test_measurement_source_status_inherits_fail_closed_pair_validation(
    count, measured
):
    frame = pd.DataFrame(
        {
            "marker_n": count,
            "marker_measured": measured,
            "marker_max": [None, 1.4],
        }
    )

    with pytest.raises(ValueError, match="measurement provenance"):
        reconcile_measurement_source_status(
            frame,
            measured_column="marker_measured",
            count_column="marker_n",
            value_column="marker_max",
        )
