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
from easyicu.research_agent.providers.prompt_budget import (
    CONSERVATIVE_BYTES_PER_TOKEN,
    DEFAULT_MAX_PROMPT_TOKENS,
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


def test_the_ceiling_is_the_reviewed_transport_number_not_a_second_one():
    """One reviewed envelope, not two -- the older copy is what refused m2.

    ``providers/prompt_budget.py`` already asked "how big may a prompt get
    before assembly has run away", for every declared consumer, and wrote the
    answer down with its reasoning: the guard belongs ABOVE normal traffic.
    The Planner kept a separate hard-coded 80,000 that predates it, and that
    copy is the one two whole tasks died on. Deriving it means the next
    revision of the envelope moves both together instead of leaving one behind.
    """

    assert _PLANNER_PROMPT_BYTE_LIMIT == int(
        DEFAULT_MAX_PROMPT_TOKENS * CONSERVATIVE_BYTES_PER_TOKEN
    ), (
        "the Planner ceiling has drifted from the reviewed transport envelope. "
        "If the envelope really should differ for this consumer, declare it in "
        "PROMPT_TRANSPORT_BUDGETS with a rationale -- do not re-hard-code a "
        "second number here, which is exactly how 80,000 outlived its review."
    )


#: Every Planner prompt the 2026-08-02 nine-task benchmark actually produced,
#: read from each run's own ``planner_prompt_metrics.json``. m2 and h2 were
#: REFUSED at 80,000 and produced zero steps; h3 cleared it by 189 bytes. These
#: are recorded totals, not projections -- anchoring on them is what stops the
#: ceiling drifting back inside real traffic.
_RECORDED_PLANNER_TOTALS = {
    "m2_mortality_prediction": 82_987,
    "h3_trajectory_clustering": 79_811,
    "h1_ventilation_survival": 78_896,
    "e1_sepsis3_prevalence_mortality": 78_670,
    "m1_hepatobiliary_missingness": 77_999,
    "m3_sepsis_subphenotype": 77_334,
    "e2_lactate_mortality": 73_266,
}


@pytest.mark.parametrize(
    ("task", "recorded_bytes"), sorted(_RECORDED_PLANNER_TOTALS.items())
)
def test_a_real_recorded_planner_prompt_is_not_refused(task, recorded_bytes):
    """A guard that real traffic crosses is not catching runaway assembly.

    Seven of nine canonical tasks reached a Planner call on 2026-08-02. Four
    had spent the whole catalog ladder and had no relief left; two were refused
    outright. If this assertion ever fails again, the question to ask is not
    "which prompt can we shrink" -- it is whether the ceiling is once more
    sitting inside normal operating traffic.
    """

    assert recorded_bytes <= _PLANNER_PROMPT_BYTE_LIMIT, (
        f"{task} really produced {recorded_bytes:,} bytes and the ceiling is "
        f"{_PLANNER_PROMPT_BYTE_LIMIT:,}. That task plans zero steps."
    )
