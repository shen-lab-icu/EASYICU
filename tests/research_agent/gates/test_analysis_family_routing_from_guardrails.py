"""A guardrail bundle must not decide the study family.

Two real mis-routes, both measured on a live run's ``research_context.json``
(2026-07-29). The research question was::

    Estimate Sepsis-3 prevalence and its association with in-hospital
    mortality, with a transparent, reproducible cohort definition and
    visible denominator.

The question names no survival vocabulary at all and no cohort-variant
comparison. The benchmark's guardrail/required-output bundle -- which
``_question_text`` concatenates into the same routing text as the question --
supplied both of the words that hijacked it:

1. ``"the landmark row must require survival to 24 hours"`` was the *only*
   occurrence of any survival term. It is an eligibility rule naming who is in
   a cohort variant. It routed the study to the survival family, whose contract
   requires ``survival_effect`` and ``temporal_absolute_risk`` -- a survival
   curve -- for a binary ``death`` flag. Five Planner attempts failed in five
   different ways and nothing executed.

2. With survival masked, ``cohort_definition_sensitivity`` took it instead:
   that predicate asked only whether a definition word and a variation word
   each appeared *somewhere*. Every analysis plan in this system is required to
   carry a robustness/sensitivity step, so the variation half was effectively
   always true, and "reproducible cohort definition" in the question supplied
   the other half.

``test_study_design.py`` already contains a test asserting this exact question
routes to association -- and it stayed green throughout, because it passes the
question with no ``user_preferences`` at all. The bundle is the part that was
never exercised, so every context here carries one.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.analysis_types import infer_analysis_type
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

# Reduced from the live bundle: a landmark eligibility rule, and a robustness
# line that says "sensitivity" about something other than a cohort definition.
LANDMARK_ELIGIBILITY_BUNDLE = (
    "CANONICAL9 REQUIRED OUTPUTS (binding):\n"
    "1. cohort definition summary\n"
    "2. table one\n"
    "3. adjusted association with timing, repeated-stay, and functional-form "
    "sensitivity table and figure\n"
    "4. Emit one row per cohort variant [primary_cohort, landmark_alive_at_24h, "
    "non_readmission_icu_stays]. The landmark row must require survival to 24 "
    "hours and exclude negative death times; the repeated-stay row must "
    "restrict to non-readmission icu stays."
)


def _binary_mortality_context(
    *,
    question: str = (
        "Estimate Sepsis-3 prevalence and its association with in-hospital "
        "mortality, with a transparent, reproducible cohort definition and "
        "visible denominator."
    ),
    must_have_outputs: str = LANDMARK_ELIGIBILITY_BUNDLE,
    evaluation_focus: str = "",
) -> ResearchContext:
    """A binary in-hospital-mortality study carrying a guardrail bundle."""

    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="analysis_cohort",
            database="miiv",
            n_patients=1000,
            n_stays=1000,
            id_columns=["stay_id"],
            outcome_columns=["los_icu", "death"],
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
            # A time column exists, so nothing structural rules survival out --
            # the routing really does turn on the words.
            ConceptDescriptor(
                name="death_time",
                role=VariableRole.TIME,
                source_concept="death_time",
                dtype="float",
            ),
        ],
        target_outcome="death",
        primary_exposure="sep3_sofa2_max",
        user_preferences=UserPreferences(
            must_have_outputs=must_have_outputs,
            evaluation_focus=evaluation_focus,
        ),
    )


def _family(context: ResearchContext) -> str:
    return infer_analysis_type(context).key


def test_a_landmark_eligibility_rule_does_not_make_it_a_survival_study() -> None:
    """The live defect: one eligibility clause, one wrong family."""

    context = _binary_mortality_context()

    assert _family(context) == "association_study"


def test_the_shown_contract_no_longer_demands_a_curve_for_a_binary_outcome() -> None:
    """What the Planner is actually handed, not just the inference key.

    The inference only matters through this object: it is the required-role
    list rendered into the planning prompt and used to reject the plan.
    """

    contract = build_article_analysis_contract(_binary_mortality_context())

    assert contract.source_analysis_type == "association_study"
    assert "survival_effect" not in contract.required_roles
    assert "temporal_absolute_risk" not in contract.required_roles
    assert {"primary_estimand", "robustness"} <= set(contract.required_roles)


@pytest.mark.parametrize(
    "eligibility_phrase",
    [
        "the landmark row must require survival to 24 hours",
        "restricted to patients with survival to 24 hours",
        "survival to 24 hours is required for inclusion",
        "eligible stays must survive to the 24 hour landmark",
        "仅纳入存活至 24 小时的患者",
        "存活满 24 小时者纳入分析",
    ],
)
def test_eligibility_spellings_do_not_assert_a_time_to_event_estimand(
    eligibility_phrase: str,
) -> None:
    context = _binary_mortality_context(
        must_have_outputs=f"1. cohort definition summary\n2. {eligibility_phrase}."
    )

    assert _family(context) != "survival"


@pytest.mark.parametrize(
    "estimand_phrase",
    [
        "report the hazard ratio for death after sepsis onset",
        "plot Kaplan-Meier curves by exposure group",
        "fit a Cox model with time-varying covariates",
        "compare survival to 28 days between exposure groups",
        "report the cumulative incidence accounting for competing risks",
        "报告风险比并绘制生存曲线",
    ],
)
def test_a_real_time_to_event_request_still_routes_to_survival(
    estimand_phrase: str,
) -> None:
    """The mask must not disarm genuine survival work.

    Each of these sits in the *same* bundle as the landmark eligibility rule,
    so the masked clause and the surviving clause coexist -- which is the case
    a blanket veto would have got wrong.
    """

    context = _binary_mortality_context(
        must_have_outputs=LANDMARK_ELIGIBILITY_BUNDLE + f"\n5. {estimand_phrase}."
    )

    assert _family(context) == "survival"


def test_masking_is_scoped_to_the_clause_not_the_sentence() -> None:
    """One sentence, two clauses: the restriction loses its vote, not the ask."""

    context = _binary_mortality_context(
        must_have_outputs=(
            "1. Restrict to patients with survival to 24 hours, then estimate "
            "the hazard of in-hospital death."
        )
    )

    assert _family(context) == "survival"


def test_a_reproducibility_requirement_is_not_a_definition_sensitivity_study() -> None:
    """ "Reproducible cohort definition" + an unrelated "sensitivity" line.

    The two words never refer to each other. Before this was a bound relation,
    their bare co-occurrence anywhere in the text selected the family.
    """

    context = _binary_mortality_context(
        must_have_outputs=(
            "1. cohort definition summary\n"
            "2. adjusted association with timing and functional-form "
            "sensitivity table and figure"
        )
    )

    assert _family(context) == "association_study"


def test_an_explicit_definition_sensitivity_preference_remains_an_adjunct() -> None:
    """A named secondary analysis must not replace the primary estimand.

    This is reduced from the natural Web E1 run.  The primary question asks for
    prevalence and a mortality association; the conversational preferences ask
    to compare timing- and readmission-restricted definitions *as sensitivity
    analyses*.  Before primary-question authority was separated from free-text
    preferences, that perfectly legitimate robustness request selected the
    descriptive cohort-definition family and removed the association analysis.
    """

    context = _binary_mortality_context(
        must_have_outputs=(
            "Estimate prevalence and the adjusted association with mortality. "
            "Assess timing-restricted and non-readmission-restricted cohort "
            "definitions as sensitivity analyses."
        )
    )

    assert _family(context) == "association_study"


@pytest.mark.parametrize(
    "question",
    [
        "Compare alternative Sepsis-3 cohort definitions and report how the "
        "estimated prevalence changes across definitions.",
        "How much does the estimated prevalence vary across different case "
        "definitions of Sepsis-3?",
        "Run a cohort definition sensitivity analysis over three eligibility "
        "criteria for Sepsis-3.",
    ],
)
def test_a_genuine_definition_comparison_still_routes_to_that_family(
    question: str,
) -> None:
    """Binding the cues must not delete the family it guards."""

    context = _binary_mortality_context(question=question)

    assert _family(context) == "cohort_definition_sensitivity"


def test_the_bundle_alone_cannot_carry_both_halves() -> None:
    """Where the words come from is the whole point.

    A definition word in the question and a variation word many lines away in
    a required-outputs list is not a study that varies its cohort definition.
    """

    context = _binary_mortality_context(
        evaluation_focus=(
            "CANONICAL9 SEMANTIC GUARDRAILS (binding):\n"
            "1. State the cohort denominator and inclusion criteria explicitly.\n"
            "2. Report a robustness check for every reported estimate."
        )
    )

    assert _family(context) == "association_study"
