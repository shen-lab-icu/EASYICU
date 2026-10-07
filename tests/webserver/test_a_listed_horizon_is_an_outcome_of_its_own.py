"""Each horizon a question lists is an outcome of its own, and never a window.

The study intent reader reads fixed-horizon endpoints through the stated
horizon reader.  It read one horizon of a list: "28-day and 90-day
mortality" named only the 90-day endpoint and "mortality at 28 and 90 days"
none.  The hours of an endpoint it could not read became the study's
exposure window: "mortality at 24 and 48 hours", "died within 48 hours of ICU
admission" and "mortality in the first 48 hours" each set a 48-hour window.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.study_intent import deterministic_intent, explicit_outcome_concepts


def _values(result: dict) -> dict:
    return {name: slot["value"] for name, slot in result["slots"].items() if slot}


@pytest.mark.parametrize(
    ("question", "outcomes"),
    [
        ("Is SOFA associated with 28-day and 90-day mortality?", ("mort_28d", "mort_90d")),
        ("Is SOFA associated with 28- and 90-day mortality?", ("mort_28d", "mort_90d")),
        ("Is SOFA associated with mortality at 28 and 90 days?", ("mort_28d", "mort_90d")),
        ("SOFA 与 28 天和 90 天死亡率的关系", ("mort_28d", "mort_90d")),
    ],
)
def test_each_listed_horizon_is_an_outcome_and_the_first_is_the_primary(question, outcomes):
    assert explicit_outcome_concepts(question) == outcomes
    assert _values(deterministic_intent(question))["outcome"] == outcomes[0]


@pytest.mark.parametrize(
    "question",
    [
        "Is SOFA associated with mortality at 24 and 48 hours?",
        "Is lactate associated with having died within 48 hours of ICU admission?",
        "Is ferritin associated with mortality in the first 48 hours?",
    ],
)
def test_an_endpoints_hours_are_no_exposure_window(question):
    assert "time_window_hours" in deterministic_intent(question)["unread"]


def test_the_exposures_hours_stay_its_window_beside_a_criteria_list():
    result = deterministic_intent(
        "Exclusion criteria: death within 24 hours of ICU admission. "
        "Outcome: in-hospital mortality. Exposure: lactate in the first 6 hours."
    )

    assert _values(result)["time_window_hours"] == 6
