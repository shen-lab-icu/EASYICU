"""A prompt 4.5% over budget shortens its menu; it does not cost the whole task.

MEASURED 2026-07-30 across the nine canonical tasks (batch
``..._88d3983_canonical9_full02``).  Four planned successfully, at 85.8%, 92.9%,
94.5% and **97.8%** of the 80,000-byte planner limit.  ``m2_mortality_prediction``
and ``h2_vasopressor_causal`` came to 83,622 bytes and were refused outright by
``PlannerPromptBudgetError`` -- zero steps, zero provider spend on science, the
whole task lost.  The failing pair is 6.9% larger than the largest success, so
this is not two pathological tasks: nine of nine run against the ceiling and two
fall off it.

Both context segments are ``required=True``, so ``BoundedContextAssembler`` had
nothing to evict and a 4.5% overflow was fatal.

Of that prompt, 47% is the task's typed context (drop it and the Planner invents
columns) and 32% is the plan schema (drop it and the Planner cannot fill the
form -- m3/h1/h3 already died that way).  The analysis-type catalog is the only
part that is pure menu, and it is byte-identical for every task.  So the catalog
descends a detail ladder under budget pressure and nothing else does.

Two properties are load-bearing and each has its own test below:

* the ladder is driven by the byte budget ALONE.  Shortening by inferred family
  would put a guess between the Planner and its options, and that inference is
  known to disagree with the Planner's own declaration (see
  ``describe_article_contract_family_switch``).
* it shortens entries, never drops them.  All 16 families stay selectable at
  every rung, because the Planner may switch away from the inferred family.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent import schema
from easyicu.research_agent.agents import core
from easyicu.research_agent.planning.analysis_types import (
    CATALOG_DETAIL_LADDER,
    list_analysis_types,
    planner_analysis_type_guide,
)


@pytest.fixture
def minimal_research_context() -> schema.ResearchContext:
    """A context small enough that full detail fits, so padding controls the test."""

    return schema.ResearchContext(
        research_question=(
            "Is first-24h peak lactate associated with in-hospital mortality?"
        ),
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[
            schema.ConceptDescriptor(
                name="lact_max", role=schema.VariableRole.LAB, dtype="float64"
            ),
            schema.ConceptDescriptor(
                name="death", role=schema.VariableRole.OUTCOME, dtype="int64"
            ),
        ],
        target_outcome="death",
        primary_exposure="lact_max",
    )


def _budget_headroom(prompt: str) -> int:
    system_bytes = len((core._SYSTEM_GUIDE + core._PRINCIPLES_GUIDE).encode("utf-8"))
    return core._PLANNER_PROMPT_BYTE_LIMIT - system_bytes - len(prompt.encode("utf-8"))


# ---------------------------------------------------------------------------
# The ladder itself
# ---------------------------------------------------------------------------


def test_the_ladder_is_ordered_most_complete_first():
    sizes = [
        len(planner_analysis_type_guide(detail=d).encode("utf-8"))
        for d in CATALOG_DETAIL_LADDER
    ]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > sizes[-1]


def test_every_family_survives_every_rung():
    """Shorten entries, never drop them.

    The Planner may switch away from the inferred family; a rung that hid
    families would make that switch impossible for exactly the tasks under the
    most budget pressure.
    """

    keys = [spec.key for spec in list_analysis_types()]
    assert keys, "the registry must not be empty for this test to mean anything"
    for detail in CATALOG_DETAIL_LADDER:
        guide = planner_analysis_type_guide(detail=detail)
        missing = [key for key in keys if f"- {key}:" not in guide]
        assert not missing, f"{detail} dropped {missing}"


def test_a_shortened_rung_says_so_in_the_prompt():
    """A silently shortened menu reads as a menu with no guardrails."""

    assert "shortened" not in planner_analysis_type_guide(detail="full")
    for detail in CATALOG_DETAIL_LADDER[1:]:
        assert "shortened" in planner_analysis_type_guide(detail=detail)


def test_an_unknown_rung_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown analysis-type catalog detail"):
        planner_analysis_type_guide(detail="whatever_fits")


# ---------------------------------------------------------------------------
# Selection is driven by the budget and by nothing else
# ---------------------------------------------------------------------------


def test_a_prompt_that_fits_keeps_the_full_catalog(minimal_research_context):
    """The four tasks that already planned must not change behaviour."""

    prompt, detail = core._planner_prompt_within_budget(minimal_research_context)
    assert detail == "full"
    assert _budget_headroom(prompt) > 0


def test_an_over_budget_prompt_descends_instead_of_failing(minimal_research_context):
    """Reproduces m2/h2: full detail overflows, a shorter rung fits."""

    full = core._build_planner_user_prompt(
        minimal_research_context, catalog_detail="full"
    )
    # Pad the article-contract segment until full detail is ~4.5% over, the
    # measured m2/h2 overflow.
    over_by = 3_622
    pad = "X" * (_budget_headroom(full) + over_by)

    assert (
        _budget_headroom(
            core._build_planner_user_prompt(
                minimal_research_context,
                planning_contract_context=pad,
                catalog_detail="full",
            )
        )
        < 0
    ), "the padding must actually reproduce an over-budget full-detail prompt"

    prompt, detail = core._planner_prompt_within_budget(
        minimal_research_context, planning_contract_context=pad
    )
    assert detail != "full"
    assert _budget_headroom(prompt) >= 0


_QUESTIONS_BY_FAMILY = {
    "association_study": "Is peak lactate associated with in-hospital mortality?",
    "trajectory_clustering": (
        "Cluster organ-dysfunction trajectories to discover subphenotypes."
    ),
    "prediction_model": (
        "Predict in-hospital mortality with a calibrated model (AUROC)."
    ),
    "survival": "Model time-to-death with Cox regression and censoring.",
}


def _context_asking(minimal_research_context, question: str):
    import copy

    ctx = copy.deepcopy(minimal_research_context)
    object.__setattr__(ctx, "research_question", question)
    return ctx


def _first_rung_that_fits(ctx, pad: str) -> str:
    """Recompute the answer from bytes alone, without consulting production."""

    for detail in CATALOG_DETAIL_LADDER:
        prompt = core._build_planner_user_prompt(
            ctx, planning_contract_context=pad, catalog_detail=detail
        )
        if _budget_headroom(prompt) >= 0:
            return detail
    return CATALOG_DETAIL_LADDER[-1]


def test_the_families_used_below_really_are_different():
    """Otherwise the family-independence test proves nothing."""

    from easyicu.research_agent.planning.analysis_types import infer_analysis_type

    ctxs = {
        family: schema.ResearchContext(
            research_question=question,
            cohort=schema.CohortDescriptor(
                cohort_name="c", database="synthetic", n_patients=100, n_stays=100
            ),
            variables=[
                schema.ConceptDescriptor(
                    name="lact_max", role=schema.VariableRole.LAB, dtype="float64"
                ),
                schema.ConceptDescriptor(
                    name="death", role=schema.VariableRole.OUTCOME, dtype="int64"
                ),
            ],
            target_outcome="death",
            primary_exposure="lact_max",
        )
        for family, question in _QUESTIONS_BY_FAMILY.items()
    }
    inferred = {family: infer_analysis_type(c).key for family, c in ctxs.items()}
    assert inferred == dict(zip(_QUESTIONS_BY_FAMILY, _QUESTIONS_BY_FAMILY))
    assert len(set(inferred.values())) == len(_QUESTIONS_BY_FAMILY)


@pytest.mark.parametrize("family", sorted(_QUESTIONS_BY_FAMILY))
@pytest.mark.parametrize("over_by", [1, 2_000, 3_622, 6_000])
def test_the_chosen_rung_is_the_first_that_fits_whatever_the_family(
    minimal_research_context, family: str, over_by: int
):
    """Budget alone decides, and it decides the FIRST fitting rung.

    An earlier version of this test asserted only that two families agreed.  It
    survived a mutation that reversed the ladder for non-association families,
    because at 3,622 bytes of pressure every ordering converges on the shortest
    rung.  Asserting the exact rung -- against an independent recomputation, at
    several pressures, one of which (2,000) is small enough that the middle rung
    is the right answer -- is what makes a reordering visible.
    """

    ctx = _context_asking(minimal_research_context, _QUESTIONS_BY_FAMILY[family])
    full = core._build_planner_user_prompt(ctx, catalog_detail="full")
    pad = "X" * (_budget_headroom(full) + over_by)

    _, detail = core._planner_prompt_within_budget(ctx, planning_contract_context=pad)
    assert detail == _first_rung_that_fits(ctx, pad)


def test_a_middle_rung_is_actually_reachable():
    """The parametrisation above is only meaningful if 2,000 lands mid-ladder.

    If every pressure resolved to the shortest rung, a reordered ladder would
    still pass. This asserts the ladder has a reachable middle.
    """

    full_bytes = len(planner_analysis_type_guide(detail="full").encode("utf-8"))
    mid_bytes = len(
        planner_analysis_type_guide(detail=CATALOG_DETAIL_LADDER[1]).encode("utf-8")
    )
    assert full_bytes - mid_bytes > 2_000


def test_still_over_at_the_shortest_rung_is_an_explicit_failure(
    minimal_research_context,
):
    """Fail closed, not silent.

    The typed context is never truncated to make a request fit; a request that
    overflows even the shortest menu must still raise.
    """

    pad = "X" * (core._PLANNER_PROMPT_BYTE_LIMIT * 2)
    with pytest.raises(core.PlannerPromptBudgetError):
        core.PlannerAgent.request_metrics(
            minimal_research_context, planning_contract_context=pad
        )


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------


def test_request_metrics_records_which_rung_produced_the_plan(
    minimal_research_context,
):
    metrics = core.PlannerAgent.request_metrics(minimal_research_context)
    assert metrics["analysis_type_catalog_detail"] in CATALOG_DETAIL_LADDER


def test_the_recorded_rung_is_the_one_that_was_sent(minimal_research_context):
    """The receipt must describe the real request, not a second computation."""

    full = core._build_planner_user_prompt(
        minimal_research_context, catalog_detail="full"
    )
    pad = "X" * (_budget_headroom(full) + 3_622)

    messages = core.PlannerAgent.request_messages(
        minimal_research_context, planning_contract_context=pad
    )
    sent = [m for m in messages if m.role == "user"][0].content
    detail = core.PlannerAgent.request_metrics(
        minimal_research_context, planning_contract_context=pad
    )["analysis_type_catalog_detail"]

    assert planner_analysis_type_guide(detail=detail) in sent
