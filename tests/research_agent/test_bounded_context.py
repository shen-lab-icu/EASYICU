"""Bounded context assembly and Planner segment metrics."""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.core import PlannerAgent, PlannerPromptBudgetError
from easyicu.research_agent.resources import (
    BoundedContextAssembler,
    ContextBudgetExceeded,
    ContextSegment,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


def test_optional_context_is_evicted_as_a_whole_never_mid_string() -> None:
    segments = (
        ContextSegment("required", "authority", required=True),
        ContextSegment("optional_low", "LOW-SENTINEL", priority=1, required=False),
        ContextSegment("optional_high", "HIGH-SENTINEL", priority=2, required=False),
    )

    assembled = BoundedContextAssembler.assemble(segments, max_bytes=27)

    assert assembled.content == "authorityHIGH-SENTINEL"
    assert "LOW-SENTINEL" not in assembled.content
    assert assembled.receipt.truncated_strings is False
    assert [item.included for item in assembled.receipt.segments] == [True, False, True]


def test_required_or_authority_context_fails_closed_on_overflow() -> None:
    segment = ContextSegment(
        "typed_authority",
        "x" * 20,
        required=True,
        authority_bound=True,
    )

    with pytest.raises(ContextBudgetExceeded, match="required context exceeds"):
        BoundedContextAssembler.assemble((segment,), max_bytes=19)


def test_authority_bound_segment_cannot_be_optional() -> None:
    with pytest.raises(ValueError, match="authority-bound"):
        ContextSegment(
            "bad_authority",
            "x",
            required=False,
            authority_bound=True,
        )


def test_planner_metrics_expose_exact_base_and_protocol_segments() -> None:
    context = ResearchContext(
        research_question="Describe the ICU cohort.",
        cohort=CohortDescriptor(
            cohort_name="fixture",
            database="miiv",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )

    baseline = PlannerAgent.request_metrics(context)
    with_protocol = PlannerAgent.request_metrics(
        context,
        know_how_context='{"cards":[]}',
    )

    assert set(baseline["segments"]) == {"planner_contract_and_typed_context"}
    assert set(with_protocol["segments"]) == {
        "planner_contract_and_typed_context",
        "reviewed_protocol_resources",
    }
    assert with_protocol["total_bytes"] > baseline["total_bytes"]
    assert with_protocol["truncated_strings"] is False


def test_planner_required_context_overflow_preserves_public_error_type() -> None:
    context = ResearchContext(
        research_question="Describe the ICU cohort.",
        cohort=CohortDescriptor(
            cohort_name="fixture",
            database="miiv",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        notes="x" * 90_000,
    )

    with pytest.raises(PlannerPromptBudgetError, match="required context exceeds"):
        PlannerAgent.request_metrics(context)
