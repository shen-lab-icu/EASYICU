"""Declaring an analysis_type replaces the contract the Planner was shown.

Measured on the real E1 context (2026-07-29). The prompt-side contract is built
with no ``analysis_type``, so the Planner is shown the family *inferred* from
the research context -- for E1 that is ``survival``:

    baseline_context, cohort_accounting, data_quality, descriptive_result,
    diagnostics, survival_effect, temporal_absolute_risk

The rejecting contract is rebuilt with ``analysis_type=plan.analysis_type``.
The moment the Planner declares ``association_study`` -- which is what E1's
accepted plan declared, and is the right label for a binary in-hospital
mortality outcome -- it is judged against a different set:

    baseline_context, cohort_accounting, data_quality, descriptive_result,
    primary_estimand, robustness

``primary_estimand`` and ``robustness`` were required but never shown;
``diagnostics``, ``survival_effect`` and ``temporal_absolute_risk`` were shown
as required and then were not. A real run spent all five Planner attempts on
five different violations and executed nothing -- one attempt was told to
produce ``table:survival_curve`` for a binary outcome.

Which side is right is genuinely open: the host's inference said survival, the
Planner said association, and the Planner was closer. So the fix is not to make
one side win silently. It is to stop the switch being *invisible*: a rejection
under a re-declared family must name both families and publish the judging
family's whole required-role set, so the next attempt can satisfy it instead of
discovering it one role at a time.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
    render_article_analysis_contract_for_prompt,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


def _time_to_event_context() -> ResearchContext:
    """A context whose inferred family is not the one a Planner would declare."""

    return ResearchContext(
        research_question=(
            "Estimate Sepsis-3 prevalence and its association with in-hospital "
            "mortality."
        ),
        cohort=CohortDescriptor(
            cohort_name="analysis_cohort",
            database="miiv",
            n_patients=1000,
            n_stays=1000,
            id_columns=["stay_id"],
            outcome_columns=["death", "death_time"],
        ),
        variables=[
            ConceptDescriptor(
                name="sep3_sofa2_max",
                role=VariableRole.COMPOSITE_SCORE,
                source_concept="sep3_sofa2",
                dtype="int",
            ),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                source_concept="death",
                dtype="int",
            ),
            ConceptDescriptor(
                name="death_time",
                role=VariableRole.OUTCOME,
                source_concept="death_time",
                dtype="float",
            ),
        ],
        target_outcome="death",
        primary_exposure="sep3_sofa2_max",
        # The real E1 signal, reduced to its cause: a guardrail describing a
        # time-to-event sensitivity is enough to make the host infer the
        # survival family for a question whose outcome is binary. Whether that
        # inference is right is a separate argument; what this file is about is
        # that the Planner is then judged by a contract it was not shown.
        user_preferences=UserPreferences(
            evaluation_focus=(
                "Report a prespecified time-to-event sensitivity for in-hospital "
                "death alongside the primary binary analysis."
            )
        ),
    )


def test_the_declared_family_really_does_replace_the_shown_contract() -> None:
    """The premise, asserted rather than assumed.

    If these two ever coincide the rest of this file is vacuous, so the
    divergence is pinned first.
    """

    context = _time_to_event_context()
    shown = build_article_analysis_contract(context)
    judged = build_article_analysis_contract(context, analysis_type="association_study")

    assert shown.source_analysis_type != "association_study", (
        "this context no longer infers a different family; pick another one or "
        "the switch under test cannot happen"
    )
    assert set(judged.required_roles) - set(
        shown.required_roles
    ), "declaring a different family no longer adds unshown required roles"


def test_the_prompt_says_the_contract_is_bound_to_its_family() -> None:
    """The Planner cannot avoid a trap it is not told about.

    The rendered block already names ``source_analysis_type``. What it never
    said is that re-declaring ``analysis_type`` swaps the whole contract --
    so a Planner correcting what it believes is a mis-inferred family has no
    way to know it just changed the rules it will be judged by.
    """

    rendered = render_article_analysis_contract_for_prompt(
        build_article_analysis_contract(_time_to_event_context())
    )

    assert "analysis_type" in rendered
    lowered = rendered.lower()
    assert "replace" in lowered or "instead of this one" in lowered, (
        "the contract block does not warn that declaring another analysis_type "
        "switches which contract judges the plan"
    )


@pytest.mark.parametrize(
    "declared",
    ["association_study", "prediction_model", "descriptive_epidemiology"],
)
def test_a_family_switch_rejection_publishes_the_whole_new_contract(
    declared: str,
) -> None:
    """One rejection must state the new contract, not leak it a role per attempt.

    Five paid attempts produced five different violations because each
    rejection named only what was missing *this* time. Naming the judging
    family's full required set turns the loop into one correction.
    """

    from easyicu.research_agent.agents.core import (
        describe_article_contract_family_switch,
    )

    context = _time_to_event_context()
    shown = build_article_analysis_contract(context)
    judged = build_article_analysis_contract(context, analysis_type=declared)

    text = describe_article_contract_family_switch(shown=shown, judged=judged)

    assert shown.source_analysis_type in text
    assert judged.source_analysis_type in text
    for role in judged.required_roles:
        assert role in text, f"the judging contract's role {role!r} is not published"


def test_no_switch_text_when_the_family_is_unchanged() -> None:
    """Nothing to say is said by saying nothing."""

    from easyicu.research_agent.agents.core import (
        describe_article_contract_family_switch,
    )

    context = _time_to_event_context()
    shown = build_article_analysis_contract(context)

    assert describe_article_contract_family_switch(shown=shown, judged=shown) == ""


def test_the_switch_note_reaches_the_real_rejection(tmp_path) -> None:
    """Reachability, not just a helper that returns the right string.

    A note nobody threads into the raised error changes nothing about the
    retry loop it exists to shorten, so this drives the real parser with a
    real plan that re-declares the family and asserts the note is in the
    exception the Planner would actually see.
    """

    from easyicu.research_agent.agents.core import (
        PlannerAgent,
        PlannerArticleContractError,
    )

    context = _time_to_event_context()
    shown = build_article_analysis_contract(context)
    assert shown.source_analysis_type != "association_study"

    plan_json = json.dumps(
        {
            "research_question": context.research_question,
            "analysis_type": "association_study",
            "cohort": None,
            "robustness_specs": [],
            "rationale": "Re-declare the family, then under-cover its contract.",
            "steps": [
                {
                    "step_id": "01_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Define the cohort and report attrition.",
                    "inputs": ["sep3_sofa2_max"],
                    "expected_outputs": ["table:cohort_flow"],
                    "method": "cohort_definition_and_attrition",
                    "icu_rule_refs": [],
                }
            ],
        }
    )

    with pytest.raises(PlannerArticleContractError) as excinfo:
        PlannerAgent(llm=object())._parse(
            plan_json,
            context,
            enforce_article_contract=True,
            article_contract_context=context,
        )

    message = str(excinfo.value)
    assert (
        "REPLACED" in message
    ), "the family switch is not reported in the error the Planner receives"
    assert shown.source_analysis_type in message
    for role in build_article_analysis_contract(
        context, analysis_type="association_study"
    ).required_roles:
        assert role in message
