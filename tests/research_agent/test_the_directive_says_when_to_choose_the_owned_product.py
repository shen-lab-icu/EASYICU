"""The directive said what a product name obliges, never when to choose it.

The Planner directive carried one sentence about this product: "A step that
declares the exact output ``table:exposure_outcome_distribution`` MUST also
declare ``exposure_outcome_distribution_spec``."  That is a conditional
obligation.  Nothing told the Planner **when** to name that product, so it
planned the same science under its own labels -- 27 recorded shapes promised
``table:cohort_summary``, 18 ``table:absolute_risk_context`` -- and the Coder
wrote each one with a different shape.  25 of 26 recorded tables have distinct
headers, and every host figure over them fails.

canary34 is the clean demonstration.  E1 reached 9 of 10 steps -- its first
run past the cohort step with the robustness replay AND its figure both green
-- and the one failure was ``05_prevalence_mortality_figure``, whose input was
``table:absolute_risk_context``:

    ValueError: absolute-risk table product contract is unsupported

``prevalence_outcome_figure`` requires an exact column header, which only its
own producer emits.  The refusal is correct; the table simply had no contract.

Refusing that at execution was tried in a1e5cde and overturned by canary33: a
gate demanding the step rename its output asked 48 recorded shapes and only 1
was its business.  Which table a step promises is the Planner's choice, so the
guidance belongs where the Planner can still act on it -- before the plan is
sealed.
"""

from __future__ import annotations

import re

import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT,
)
from easyicu.research_agent.agents.core import _build_planner_user_prompt
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


#: The real prompt builder, not a copy of its text. Reading the source file
#: instead would pass even if the sentence were rendered into a branch the
#: Planner never receives.
def _directive() -> str:
    return _build_planner_user_prompt(
        ResearchContext(
            research_question="Is the exposure associated with the outcome?",
            cohort=CohortDescriptor(
                cohort_name="directive",
                database="test",
                n_patients=10,
                n_stays=10,
                id_columns=["stay_id"],
            ),
            variables=[],
        )
    )


def test_the_directive_says_when_to_choose_this_product_not_only_what_it_obliges():
    """The obligation sentence alone is what let the same science drift."""

    text = _directive()
    product = EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT

    assert product in text
    where = text.index("BY EXPOSURE LEVEL")
    obliges = text.index(
        f"A step that declares the exact output `{product}` MUST also declare"
    )
    assert where < obliges, "the choice guidance must precede the obligation"


def test_it_names_the_three_ways_a_planner_spells_this_science():
    """Prevalence / absolute risk / outcome-by-group are the labels the
    recorded plans actually used for this product's science."""

    text = _directive().casefold()

    for phrase in ("prevalence", "absolute risk", "outcome by group"):
        assert phrase in text, phrase


def test_it_says_why_another_name_costs_the_figure():
    """Guidance without the consequence reads as a style preference."""

    text = _directive()

    assert "different shape every run" in text
    assert "no host figure can consume it" in text


def test_it_keeps_the_reader_facing_name_free():
    """The same split the robustness spec publishes: the label is the
    Planner's, the declared OUTPUT is the contract."""

    text = _directive()

    assert "Name the step whatever your reader should see" in text
    assert "declared OUTPUT that decides who computes it" in text


def test_the_obligation_sentence_is_still_there():
    """Adding the choice guidance must not drop the requirement it explains."""

    text = _directive()

    assert (
        f"A step that declares the exact output `{EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT}` "
        "MUST also declare `exposure_outcome_distribution_spec`" in text
    )


def test_the_guidance_carries_no_case_specific_token():
    """Prompt hygiene: a host-capability fact, not a benchmark fact."""

    text = _directive()
    window = text[
        text.index("BY EXPOSURE LEVEL") : text.index(
            f"A step that declares the exact output "
            f"`{EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT}` MUST also declare"
        )
    ]
    for token in (
        "sep3",
        "kdigo",
        "aki_stage",
        "mimic",
        "sepsis",
        "lactate",
        "e1",
        "e3",
    ):
        assert not re.search(rf"\b{re.escape(token)}\b", window.casefold()), token


def test_it_says_a_host_drawn_figure_takes_exactly_its_own_product():
    """canary35's last failing step, in one sentence.

    E1 took the guidance above and promised
    ``table:exposure_outcome_distribution`` -- that step went ok. Its figure
    then declared three CONTEXT tables alongside it (adjusted estimates, the
    robustness matrix, a measurement audit), and
    ``EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY`` is
    ``required={table:exposure_outcome_distribution}, optional=frozenset()``.
    One extra typed input and no host renderer claims the step; it fell to the
    code generator, which coerced the exposure LEVEL LABEL column to float and
    died on its own non-finite guard.

    The capability's exactness is deliberate -- a renderer must not silently
    ignore an input the Planner declared -- so the guidance goes to the
    Planner, not into the capability.
    """

    text = _directive()

    assert "consumes EXACTLY the typed product it renders" in text
    assert "no host renderer can draw it" in text
    assert "put it in its own figure step" in text


def test_the_figure_rule_is_stated_before_the_obligation_too():
    """Same placement discipline as the choice guidance above."""

    text = _directive()

    assert text.index("consumes EXACTLY the typed product it renders") < text.index(
        f"A step that declares the exact output "
        f"`{EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT}` MUST also declare"
    )
