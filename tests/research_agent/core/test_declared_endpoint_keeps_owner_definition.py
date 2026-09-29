"""A declared outcome whose question names no endpoint keeps its owner definition.

When the research question requests no endpoint definition, the builder records
the target as a bare declared outcome.  That is not a request, so it cannot
conflict with the concept owner's definition; treating it as one marked every
such fixed-horizon endpoint as an endpoint-definition conflict, and the plan
review then refused the endpoint the researcher had chosen.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.scientific_review import _endpoint_resolved
from easyicu.research_agent.research_context.builder import build_research_context

_ENDPOINT = EndpointSpec(
    name="mort_28d", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]
)


def _context(question: str):
    cohort = pd.DataFrame(
        {"stay_id": [1, 2, 3, 4], "lactate": [1.2, 3.4, 2.2, 5.1], "mort_28d": [0, 1, 0, 1]}
    )
    built = build_research_context(
        research_question=question, cohort=cohort, cohort_name="synthetic",
        database="synthetic", target_outcome="mort_28d", endpoint=_ENDPOINT,
    )
    return getattr(built, "context", built)


def _conflicts(context) -> list[str]:
    return [
        note for note in context.variable("mort_28d").clinical_caveats
        if "Endpoint-definition conflict" in note
    ]


def test_a_question_without_an_endpoint_keeps_the_owner_definition():
    context = _context("Do early lactate trajectory classes differ in outcome?")

    assert context.variable("mort_28d").description == "28-day Mortality"
    assert _conflicts(context) == []
    assert _endpoint_resolved(context)


@pytest.mark.parametrize(
    ("question", "conflict"),
    [
        ("Do early lactate trajectory classes differ in hospital mortality?", True),
        ("Do early lactate trajectory classes differ in 28-day mortality?", False),
    ],
)
def test_a_requested_endpoint_is_still_compared_with_the_owner(question, conflict):
    context = _context(question)

    assert bool(_conflicts(context)) is conflict
    assert _endpoint_resolved(context) is not conflict
