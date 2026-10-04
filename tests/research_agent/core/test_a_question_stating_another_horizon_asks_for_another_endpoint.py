"""A question that states another horizon asks for another endpoint.

The context builder compared a question's endpoint with the concept owner's
without its horizon.  A survival question was paired with any ``mort_<h>d``
whose follow-up was in context, so "survival to day 28" became the 90-day
endpoint, censored at day 90.  Only 28-day and 30-day wording was read as a
mortality horizon, and only 28-day and 30-day concepts kept their owner's
definition: "90-day mortality" against the 28-day flag was no request at all,
and "in-hospital mortality" or "28-day mortality" against the 90-day flag
rewrote that flag's definition to the requested one.  Every one of these
passed review as resolved.

The horizons a question states are now read in one vocabulary
(``outcome_availability``), the closed fixed-horizon concepts keep their
owner's horizon (their concept's, whatever their description says), and a
request agrees with an endpoint only when its stated horizon admits the
endpoint's.  Synthetic cohorts (renal replacement therapy);
zero real patients.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.outcome_availability import (
    StatedHorizon,
    fixed_horizon_mortality_endpoint_stated_by,
    stated_mortality_horizons,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.scientific_review import _endpoint_resolved
from easyicu.research_agent.research_context.builder import (
    _enrich_target_outcome_descriptor,
    build_research_context,
)
from easyicu.research_agent.schema import ConceptDescriptor

_ASKS = "Is renal replacement therapy associated with "
_OWNER_DEFINITION = {"mort_28d": "28-day Mortality", "mort_90d": "90-day Mortality", "mort_365d": "1-year Mortality"}


@pytest.mark.parametrize(
    ("text", "stated"),
    [
        ("28-day mortality", ["28-day"]),
        ("90-day all-cause mortality", ["90-day"]),
        ("survival to day 28", ["28-day"]),
        ("death within 90 days of ICU admission", ["90-day"]),
        ("alive at day 28", ["28-day"]),
        ("day-28 mortality", ["28-day"]),
        ("one-year survival", ["1-year"]),
        ("followed up for one year", ["1-year"]),
        ("90 days of follow-up", ["90-day"]),
        ("3-month survival", ["3-month"]),
        ("90 天全因死亡率", ["90-day"]),
        ("一年生存率", ["1-year"]),
        ("三个月内死亡", ["3-month"]),
        ("第 28 天死亡", ["28-day"]),
        ("随访一年", ["1-year"]),
        ("28-day mortality, then survival to day 90", ["28-day", "90-day"]),
        # A number of hours, or days that qualify something else, is no horizon.
        ("exposure within 7 days of admission and in-hospital mortality", []),
        ("a 7-day course and survival", []),
        ("a 24-hour landmark survival analysis", []),
        ("the 3-day SOFA trajectory and 28 deaths", []),
    ],
)
def test_the_horizons_a_question_states(text, stated):
    assert [horizon.adjective for horizon in stated_mortality_horizons(text)] == stated


@pytest.mark.parametrize(
    ("count", "unit", "concept"),
    [
        (28, "day", "mort_28d"),
        (4, "week", "mort_28d"),
        (3, "month", "mort_90d"),
        (12, "month", "mort_365d"),
        (1, "year", "mort_365d"),
        # A month is 30 or 31 days: four weeks are not a month.
        (1, "month", None),
        (30, "day", None),
        (6, "month", None),
        (13, "month", None),
        (2, "year", None),
    ],
)
def test_a_stated_horizon_admits_at_most_one_closed_endpoint(count, unit, concept):
    endpoint = fixed_horizon_mortality_endpoint_stated_by(StatedHorizon(count=count, unit=unit))

    assert (endpoint.event_concept if endpoint is not None else None) == concept


def test_a_long_question_is_read_in_bounded_time():
    import subprocess
    import sys

    program = (
        "from easyicu.outcome_availability import stated_mortality_horizons as read\n"
        "for q in ('28' + ' ' * 20000 + 'x', 'mortality ' + 'at ' * 5000, '1-' * 8000,\n"
        "          'day ' * 6000 + '28', '28-day ' + 'all-cause ' * 4000, '随访' + '至' * 8000):\n"
        "    read(q)\n"
    )
    subprocess.run([sys.executable, "-c", program], check=True, timeout=30)


def _context(question: str, outcome: str, followup: str | None):
    columns = {"stay_id": [1, 2, 3, 4], "rrt": [0, 1, 0, 1], outcome: [0, 1, 0, 1]}
    if followup is not None:
        columns[followup] = [28.0, 3.5, 28.0, 12.0]
    built = build_research_context(
        research_question=question, cohort=pd.DataFrame(columns), cohort_name="synthetic",
        database="synthetic", target_outcome=outcome, primary_exposure="rrt",
        endpoint=EndpointSpec(name=outcome, kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]),
    )
    return getattr(built, "context", built)


@pytest.mark.parametrize(
    ("question", "outcome", "followup", "conflict"),
    [
        ("survival to day 28 in a Cox model?", "mort_90d", "followup_days_90d", True),
        ("one-year survival in a Cox model?", "mort_28d", "followup_days_28d", True),
        ("survival to day 90 in a Cox model?", "mort_90d", "followup_days_90d", False),
        ("3-month survival in a Cox model?", "mort_90d", "followup_days_90d", False),
        ("one-year survival in a Cox model?", "mort_365d", "followup_days_365d", False),
        # A survival question that states no horizon takes the paired endpoint's.
        ("survival in a Cox model?", "mort_90d", "followup_days_90d", False),
    ],
)
def test_a_survival_question_is_paired_only_with_its_stated_horizon(question, outcome, followup, conflict):
    context = _context(_ASKS + question, outcome, followup)
    notes = context.variable(outcome).clinical_caveats

    assert any("Endpoint-definition conflict" in note for note in notes) is conflict
    assert any(f"paired follow-up '{followup}'" in note for note in notes) is not conflict
    assert _endpoint_resolved(context) is not conflict


@pytest.mark.parametrize(
    ("question", "outcome", "conflict"),
    [
        ("in-hospital mortality?", "mort_90d", True),
        ("28-day mortality?", "mort_90d", True),
        ("90-day mortality?", "mort_28d", True),
        ("30-day mortality?", "mort_28d", True),
        # Without its paired follow-up a fixed-horizon flag is no time-to-event endpoint.
        ("survival in a Cox model?", "mort_90d", True),
        ("90-day mortality?", "mort_90d", False),
        ("1-year mortality?", "mort_365d", False),
        ("28-day mortality?", "mort_28d", False),
        ("mortality?", "mort_90d", False),
    ],
)
def test_a_fixed_horizon_concept_keeps_its_owner_definition(question, outcome, conflict):
    context = _context(_ASKS + question, outcome, None)
    descriptor = context.variable(outcome)

    assert descriptor.description == _OWNER_DEFINITION[outcome]
    assert descriptor.source_concept == outcome
    assert any("Endpoint-definition conflict" in note for note in descriptor.clinical_caveats) is conflict
    assert _endpoint_resolved(context) is not conflict


@pytest.mark.parametrize(("question", "conflict"), [("90-day mortality?", False), ("28-day mortality?", True)])
def test_a_closed_endpoint_keeps_its_horizon_whatever_its_description_says(question, conflict):
    # An owner description that states no horizon is still the 90-day concept's.
    definition = "Death from any cause after ICU admission"
    descriptor = ConceptDescriptor(name="mort_90d", dtype="int64", source_concept="mort_90d", description=definition)

    _enrich_target_outcome_descriptor(
        descriptors=[descriptor], research_question=_ASKS + question, target_outcome="mort_90d",
    )

    assert descriptor.description == definition
    assert descriptor.source_concept == "mort_90d"
    assert any("Endpoint-definition conflict" in note for note in descriptor.clinical_caveats) is conflict
