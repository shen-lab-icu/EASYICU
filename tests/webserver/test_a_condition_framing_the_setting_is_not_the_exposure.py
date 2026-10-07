"""A condition that frames a question's setting is not its exposure.

The intent reader takes the first concept it reads, other than an outcome, as
the exposure.  When the studied marker is one it cannot name ("ferritin",
"serum magnesium", "降钙素原"), the condition that frames the setting became
the exposure instead: "is ferritin associated with mortality in sepsis?" was
read as a study of sepsis.  A condition after "in" or "among", or one that
directly qualifies the outcome ("sepsis mortality", "脓毒症28天死亡"), is now
the setting, so an exposure the reader cannot name stays unread.  A condition
that is the subject of the question is still its exposure.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import study_intent


def _read(question: str) -> tuple:
    slots = study_intent.deterministic_intent(question)["slots"]
    return slots["exposure"]["value"], slots["outcome"]["value"]


@pytest.mark.parametrize(
    ("question", "outcome"),
    [
        ("Is serum magnesium associated with 28-day mortality in sepsis?", "mort_28d"),
        ("Is ferritin associated with mortality in sepsis?", "death"),
        ("Is ferritin associated with sepsis mortality?", "death"),
        ("Is the anion gap associated with mortality in AKI?", "death"),
        ("Among sepsis survivors, is ferritin associated with 90-day mortality?", "mort_90d"),
        ("降钙素原与脓毒症28天死亡的关系", "mort_28d"),
    ],
)
def test_the_setting_is_not_read_as_the_exposure(question, outcome):
    assert _read(question) == (None, outcome)


@pytest.mark.parametrize(
    ("question", "exposure", "outcome"),
    [
        ("Is lactate associated with mortality in sepsis?", "lact", "death"),
        ("Is lactate associated with sepsis mortality?", "lact", "death"),
        ("Is lactate associated with AKI in sepsis?", "lact", "aki"),
        ("乳酸与脓毒症28天死亡的关系", "lact", "mort_28d"),
    ],
)
def test_the_studied_marker_is_read_beside_its_setting(question, exposure, outcome):
    assert _read(question) == (exposure, outcome)


@pytest.mark.parametrize(
    ("question", "exposure", "outcome"),
    [
        ("Is sepsis associated with 28-day mortality?", "sep3", "mort_28d"),
        ("脓毒症与28天死亡的关系", "sep3", "mort_28d"),
        ("Is AKI associated with mortality in sepsis?", "aki", "death"),
        ("Is mechanical ventilation associated with in-hospital mortality?", "vent_ind", "death"),
        ("Is the increase in lactate associated with AKI?", "lact", "aki"),
    ],
)
def test_a_condition_or_marker_the_question_studies_is_still_the_exposure(
    question, exposure, outcome
):
    assert _read(question) == (exposure, outcome)
