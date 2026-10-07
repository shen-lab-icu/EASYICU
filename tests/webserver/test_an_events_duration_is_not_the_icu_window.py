"""An event's duration counts from the event, not from ICU admission.

The intent reader takes the first stated number of hours as the study's
window, which the host materializes from ICU admission.  "The first 24 hours
after suspected infection onset" and "脓毒症发生后24小时内" became a 24-hour
ICU window.  On the aligned Sepsis-3 path the time-anchor gate admits such a
question, so the host would have summarized the first 24 hours after ICU
admission, not after infection onset.  A duration inside a span that the
event time-zero reader states counts from that event: it is not the window,
as an outcome horizon is not.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import study_intent


def _window(question: str):
    return study_intent.deterministic_intent(question)["slots"]["time_window_hours"]["value"]


@pytest.mark.parametrize(
    "question",
    [
        "Lactate in the first 24 hours after suspected infection onset and 28-day mortality",
        "脓毒症发生后24小时内的乳酸与28天死亡",
        "Is the lactate within 6 hours of intubation associated with in-hospital mortality?",
        "插管后6小时内的乳酸与院内死亡",
        "Is the lactate 24 h after sepsis onset associated with in-hospital mortality?",
        "Is the lactate in the first day after intubation associated with in-hospital mortality?",
    ],
)
def test_an_events_duration_is_not_the_window(question):
    assert _window(question) is None


@pytest.mark.parametrize(
    ("question", "hours"),
    [
        ("Lactate in the first 24 hours and 28-day mortality", 24),
        ("Is the lactate within 24 hours of ICU admission associated with in-hospital mortality?", 24),
        ("首日乳酸与死亡", 24),
        ("入ICU后6小时内的乳酸与院内死亡", 6),
        (
            "Among stays ventilated within 6 hours of intubation, is the lactate of the "
            "first 24 hours associated with in-hospital mortality?",
            24,
        ),
    ],
)
def test_a_duration_counted_from_icu_admission_is_still_the_window(question, hours):
    assert _window(question) == hours
