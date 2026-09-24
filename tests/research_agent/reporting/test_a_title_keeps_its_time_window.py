"""A time window in a heading is a study coordinate, not a reported result.

A neutral title such as "... 28-Day Mortality ..." names when the outcome is
observed.  The heading filter treated its digits as a result next to a
result word and removed the whole title, so the reader manuscript opened
with an empty "#".  A heading that states a value still needs evidence.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.manuscript_claim_policy import (
    _heading_requires_evidence,
    filter_evidence_bound_scaffold,
)

NEUTRAL_TITLES = (
    "Early Serum Lactate Tertile and 28-Day Mortality in Adult ICU Stays: "
    "A Retrospective Cohort Study",
    "A 6-Hour Landmark Cohort Study of Vasopressor Dose and In-Hospital Mortality",
    "Hospital Mortality Within 48 Hours of ICU Readmission: A Cohort Study",
    "Median Arterial Pressure in the First 24 Hours and 90-day Mortality",
)

RESULT_HEADINGS = (
    "28-day mortality was 12% in the highest lactate tertile",
    "Mortality at 28 days was higher in the highest lactate tertile",
    "Median ICU stay of 6 days in the highest lactate tertile",
    "Hazard ratio 1.4 for ICU readmission within 48 hours",
)


@pytest.mark.parametrize("title", NEUTRAL_TITLES)
def test_a_neutral_title_with_a_time_window_needs_no_evidence(title: str) -> None:
    assert _heading_requires_evidence(title) is False


@pytest.mark.parametrize("heading", RESULT_HEADINGS)
def test_a_heading_that_states_a_value_still_needs_evidence(heading: str) -> None:
    assert _heading_requires_evidence(heading) is True


def test_the_filtered_manuscript_keeps_its_title() -> None:
    title = f"# {NEUTRAL_TITLES[0]}"
    scaffold = "\n".join(
        [
            title,
            "",
            "**Keywords:** lactate, intensive care unit, mortality",
            "",
            "## Introduction",
            "",
            "Serum lactate reflects tissue perfusion in critical illness.",
        ]
    )

    result = filter_evidence_bound_scaffold(scaffold, resolve_claim=lambda _ref: None)

    assert result.scaffold.splitlines()[0] == title
