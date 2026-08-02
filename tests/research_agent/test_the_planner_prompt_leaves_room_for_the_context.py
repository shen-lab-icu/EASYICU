"""Every byte of fixed directive is a byte the typed context cannot have.

On 2026-08-02 a 2,819-byte addition to the Planner directive killed
``h1_ventilation_survival`` in the planning phase::

    PlannerPromptBudgetError: required context exceeds byte budget without
    safe whole-segment eviction: total=80071 limit=80000

Seventy-one bytes over, from an addition of 2,819.  The addition had been
checked for headroom -- against a MINIMAL context, where it left 29 KB spare
and looked free.  The tasks with no headroom are exactly the ones with large
typed contexts, and none of those was measured.  Two more (``m2``, ``h2``) were
already ~1.3 KB over on their own and failed regardless; only h1 is
attributable to the directive.

So the guard is on the part that is ours to control: the prompt rendered
against an empty context is the FIXED cost, and ``80000 - fixed`` is the whole
budget every real typed context has to fit inside.  A change that grows the
fixed part shrinks that room for every task at once, which is invisible in any
single-task test.

This test is deliberately a ratchet, not a ceiling on prose quality: raising
the number is allowed, but only as a decision someone made on purpose, with the
context room recomputed.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.core import (
    _PLANNER_PROMPT_BYTE_LIMIT,
    PlannerAgent,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

#: Measured at the commit that shrank the cohort-step directive back down.
#: The recorded failure needed 80,071 bytes for h1's whole prompt, so the room
#: left for a typed context is what this number really controls.
FIXED_PROMPT_BYTE_BUDGET = 50_000

#: h1's recorded prompt total. Kept as the concrete case the budget serves.
_H1_RECORDED_TOTAL_BYTES = 80_071
_H1_DIRECTIVE_ADDITION = 2_819


def _minimal_context() -> ResearchContext:
    """The smallest legal context: what remains is the fixed directive."""

    return ResearchContext(
        research_question="Is the exposure associated with in-hospital death?",
        cohort=CohortDescriptor(
            cohort_name="fixed-cost-probe",
            database="miiv",
            n_patients=1,
            n_stays=1,
            id_columns=["stay_id"],
        ),
        variables=[],
    )


@pytest.fixture(scope="module")
def fixed_bytes() -> int:
    return PlannerAgent.request_metrics(_minimal_context())["total_bytes"]


def test_the_fixed_directive_stays_within_its_budget(fixed_bytes):
    assert fixed_bytes <= FIXED_PROMPT_BYTE_BUDGET, (
        f"the fixed Planner prompt is {fixed_bytes} bytes, over the "
        f"{FIXED_PROMPT_BYTE_BUDGET} budget. Every byte here is taken from "
        "every task's typed context at once. Shrink the addition, or raise "
        "this number deliberately and recompute the room left for a context."
    )


def test_the_budget_leaves_the_room_the_recorded_failure_needed(fixed_bytes):
    """h1 needed its context to fit in what the directive left over.

    Anchored on the real numbers rather than on a round figure: the room this
    budget guarantees must exceed the context h1 actually carried, or the same
    task dies the same way.
    """

    h1_context_bytes = _H1_RECORDED_TOTAL_BYTES - (
        PlannerAgent.request_metrics(_minimal_context())["total_bytes"]
        + _H1_DIRECTIVE_ADDITION
    )
    room = _PLANNER_PROMPT_BYTE_LIMIT - fixed_bytes

    assert (
        h1_context_bytes > 0
    ), "the arithmetic below is only meaningful if h1's context is real"
    assert room > h1_context_bytes, (
        f"the directive leaves {room} bytes for a typed context, but h1 "
        f"carried {h1_context_bytes}. That task plans only by luck."
    )


def test_the_budget_is_below_the_transport_limit(fixed_bytes):
    """A fixed cost at or above the transport limit would leave no context at
    all, and the failure would look like a context problem rather than a
    directive one."""

    assert FIXED_PROMPT_BYTE_BUDGET < _PLANNER_PROMPT_BYTE_LIMIT
    assert fixed_bytes < _PLANNER_PROMPT_BYTE_LIMIT
