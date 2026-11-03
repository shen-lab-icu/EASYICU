"""A suppressed count cannot be restored from the proportion beside it.

11-03 external review: the model-facing missingness summary for 3 missing of
120 read ``{"fraction_missing": 0.025, "n_missing": "suppressed",
"n_total": 120}``; 0.025 x 120 restores the 3. Synthetic values only.
"""

from __future__ import annotations

from easyicu.research_agent.research_context.outbound import (
    SUPPRESSED_COUNT,
    project_outbound_counts,
)


def test_the_missingness_proportion_is_suppressed_with_its_count():
    shown = project_outbound_counts({"missingness": {
        "fraction_missing": 0.025, "n_missing": 3, "n_total": 120, "severity": "low",
    }})["missingness"]
    assert shown == {"fraction_missing": SUPPRESSED_COUNT, "n_missing": SUPPRESSED_COUNT,
                     "n_total": 120, "severity": "low"}


def test_a_group_rate_is_suppressed_with_its_small_event_count():
    row = project_outbound_counts({"outcome_events": 4, "outcome_denominator": 63,
                                   "outcome_rate_pct": 6.349206, "exposure_level": "Female"})
    assert row["outcome_events"] == row["outcome_rate_pct"] == SUPPRESSED_COUNT
    assert row["outcome_denominator"] == 63 and row["exposure_level"] == "Female"


def test_proportions_stay_exact_when_no_count_was_suppressed():
    payload = {"fraction_missing": 0.25, "n_missing": 30, "n_total": 120}
    assert project_outbound_counts(payload) == payload
    zero = {"fraction_missing": 0.0, "n_missing": 0, "n_total": 120}
    assert project_outbound_counts(zero) == zero


def test_a_measured_rate_is_not_a_proportion_of_the_count():
    shown = project_outbound_counts({"n": 3, "heart_rate": 88, "resp_rate": 18})
    assert shown == {"n": SUPPRESSED_COUNT, "heart_rate": 88, "resp_rate": 18}
