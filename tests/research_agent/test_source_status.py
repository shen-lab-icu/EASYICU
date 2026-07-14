"""Tests for case-neutral sparse binary event reconciliation."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.methods.source_status import (
    reconcile_binary_event_presence,
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
