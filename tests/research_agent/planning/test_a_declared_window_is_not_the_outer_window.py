"""A variable's own time window, when it declares one, decides its timing.

The host proves that a dynamic covariate is observed by the landmark from its
materialization window.  A variable without its own window inherits the outer
ICU-admission feature window.  A variable that declared a window this owner
could not place on the ICU-admission axis -- another anchor such as
``event_onset[0,72]h``, or an unreadable label -- also inherited the outer
0-24 h window, so a measurement taken up to 72 h after an event became an
at-or-before-landmark covariate and entered the compiled adjustment set.
Synthetic contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.planning.adjustment_authority import (
    host_proven_temporal_roles,
)
from tests.research_agent.planning.family_spec_fixtures import (
    PLANNER_ROSTER,
    _context,
    _request,
    _run,
    _spec_payload,
)

COVARIATE = "severity_score_24h"


def _with_window(window, *, landmark_hours: float | None = None):
    context = _context(exact=False)
    variables = [
        variable.model_copy(update={"analysis_window": window})
        if variable.name == COVARIATE else variable
        for variable in context.variables
    ]
    update = {"variables": variables}
    if landmark_hours is not None:
        update["user_preferences"] = context.user_preferences.model_copy(
            update={"landmark_hours": landmark_hours}
        )
    return context.model_copy(update=update)


@pytest.mark.parametrize(
    "window",
    [
        pytest.param("event_onset[0,72]h", id="another_anchor"),
        pytest.param("hospital_admission[-12,12]h", id="another_anchor_ending_early"),
        pytest.param("throughout the ICU stay", id="unreadable_label"),
        pytest.param("icu_admission[0,48]h", id="icu_window_ending_after_landmark"),
    ],
)
def test_a_declared_window_the_host_cannot_place_proves_no_timing(window) -> None:
    assert COVARIATE not in host_proven_temporal_roles(_with_window(window))


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(None, id="no_window_inherits_the_outer_window"),
        pytest.param("icu_admission[0,24]h", id="icu_window_ending_at_landmark"),
        pytest.param("icu_admission[-6,24]h", id="icu_window_starting_before_admission"),
        pytest.param("first_24h", id="named_icu_window"),
    ],
)
def test_a_window_on_the_icu_axis_still_proves_its_timing(window) -> None:
    assert host_proven_temporal_roles(_with_window(window))[COVARIATE] == (
        "at_or_before_time_zero"
    )


def test_another_anchor_proves_nothing_at_a_later_landmark_either() -> None:
    # An event-onset window ending at 24 h may still end after a 48 h ICU
    # landmark: the event can occur at any ICU hour.
    context = _with_window("event_onset[0,24]h", landmark_hours=48.0)

    assert COVARIATE not in host_proven_temporal_roles(context)


def test_the_planner_cannot_adjust_for_an_event_window_measurement() -> None:
    context = _with_window("event_onset[0,72]h")
    request = _request(context)
    assert not request.candidate(COVARIATE).selectable

    without = [item for item in PLANNER_ROSTER if item["name"] != COVARIATE]
    llm, outcome = _run(context, [
        json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER)),
        json.dumps(_spec_payload(request, adjustment_set=without)),
    ])

    # The first spec names it and is refused; the retry compiles without it.
    assert len(llm.calls) == 2
    primary = next(step for step in outcome.output.steps if step.step_id == "adjusted_association")
    assert primary.model_requirements[0].covariates == ["age", "sex"]
