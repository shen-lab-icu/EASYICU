"""A survival question on a fixed-horizon endpoint with its paired follow-up is no conflict.

Event status by day h with its follow-up time censored at h is a time-to-event
endpoint.  The context builder compared the survival question's request
("time-to-event endpoint") with the concept's name ("28-day Mortality") and
marked every such study as an endpoint-definition conflict, so the plan review
asked the researcher to resolve an endpoint the host can execute.  Synthetic
cohorts (renal replacement therapy); zero real patients.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.scientific_review import _endpoint_resolved
from easyicu.research_agent.research_context.builder import build_research_context

_SURVIVAL = "Is renal replacement therapy associated with survival to day 28 in a Cox model?"


def _context(question: str, *, outcome: str = "mort_28d", followup: str | None = "followup_days_28d"):
    columns = {"stay_id": [1, 2, 3, 4], "rrt": [0, 1, 0, 1], outcome: [0, 1, 0, 1]}
    if followup is not None:
        columns[followup] = [28.0, 3.5, 28.0, 12.0]
    built = build_research_context(
        research_question=question, cohort=pd.DataFrame(columns), cohort_name="synthetic",
        database="synthetic", target_outcome=outcome, primary_exposure="rrt",
        endpoint=EndpointSpec(name=outcome, kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]),
    )
    return getattr(built, "context", built)


def _notes(context, outcome: str = "mort_28d") -> list[str]:
    return list(context.variable(outcome).clinical_caveats)


def test_the_paired_fixed_horizon_endpoint_is_the_requested_time_to_event_endpoint():
    context = _context(_SURVIVAL)

    notes = _notes(context)
    assert not any("Endpoint-definition conflict" in note for note in notes)
    assert any("paired follow-up 'followup_days_28d'" in note and "day 28" in note for note in notes)
    assert context.variable("mort_28d").description == "28-day Mortality"
    assert _endpoint_resolved(context)


@pytest.mark.parametrize(
    ("question", "outcome", "followup"),
    [
        # Without its follow-up time the concept is only a binary flag.
        (_SURVIVAL, "mort_28d", None),
        # A setting-defined mortality flag has no fixed horizon to censor at.
        ("Is renal replacement therapy associated with survival in a Cox model?", "death", "followup_days_28d"),
    ],
    ids=["no_followup", "no_fixed_horizon"],
)
def test_an_endpoint_without_a_paired_horizon_still_conflicts(question, outcome, followup):
    context = _context(question, outcome=outcome, followup=followup)

    assert any("Endpoint-definition conflict" in note for note in _notes(context, outcome))
    assert not _endpoint_resolved(context)


def test_a_question_naming_the_fixed_horizon_keeps_its_agreement():
    context = _context("Is renal replacement therapy associated with 28-day mortality?")

    assert not any("Endpoint-definition conflict" in note for note in _notes(context))
    assert _endpoint_resolved(context)
