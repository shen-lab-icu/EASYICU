"""A plan no host owner can finish stops before the Provider, with a typed reason.

Final acceptance needs a causal or survival primary step to carry the
Planner-declared ``family_primary_result_requirement`` once the context names
its exposure and outcome.  The Progressive v2 compiler never writes that
field, and the family router offers no template for such a context unless the
host has sealed a suite for it.  A causal question and an unsealed survival
question therefore spent the whole planning budget -- outline, steps, bounded
suffix repairs -- and failed only at final acceptance.  The planner now stops
before its first Provider call with ``progressive_family_result_contract_unwritable``.
Synthetic contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents import progressive_planner
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
)
from easyicu.research_agent.planning.family_spec.request import (
    SEALED_SURVIVAL_SUITE_MARKER,
)
from easyicu.research_agent.planning.primary_result_contract import (
    families_requiring_family_result_contract,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
)

UNWRITABLE = "progressive_family_result_contract_unwritable"
QUESTIONS = {
    "causal_inference": (
        "Among adult ICU stays, what is the effect of a higher first-24 h injury stage "
        "on in-hospital death?"
    ),
    "survival": (
        "Among adult ICU stays, how does the first-24 h injury stage relate to the time "
        "to in-hospital death?"
    ),
}


def _family_context(family: str, **update):
    base = _context()
    return base.model_copy(
        update={
            "research_question": QUESTIONS[family],
            "user_preferences": base.user_preferences.model_copy(
                update={"inferred_analysis_family": family}
            ),
            **update,
        }
    )


def _plan(context, *, strategy: str = FAMILY_SPEC_STRATEGY, **kwargs):
    llm = ScriptedMockLLMClient([])
    try:
        ProgressivePlannerAgent(llm).run_attempt(
            context,
            planner_strategy=strategy,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            direct_comparator_literature_keys=DIRECT_COMPARATORS,
            comparison_literature_keys=DIRECT_COMPARATORS,
            enforce_article_contract=True,
            article_contract_context=context,
            planning_contract_context=kwargs.pop("planning_contract_context", ""),
            required_primary_cohort_selection_mode="predicate_filtered",
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - the test reads which stop it was
        return llm, exc
    return llm, None


@pytest.mark.parametrize("strategy", [FAMILY_SPEC_STRATEGY, "progressive_v2"])
@pytest.mark.parametrize("family", ["causal_inference", "survival"])
def test_a_family_result_no_owner_can_write_stops_before_the_provider(family, strategy):
    context = _family_context(family)
    assert candidate_analysis_types(context) == (family,)

    llm, stopped = _plan(context, strategy=strategy)

    assert isinstance(stopped, ProgressivePlanCompileError)
    assert stopped.reason_code == UNWRITABLE
    assert stopped.details["owner"] == "easyicu.planning.progressive_compiler_v1"
    assert stopped.path == "analysis_type"
    assert llm.calls == []


def test_a_question_with_another_executable_family_still_reaches_the_planner():
    # The association context also lists causal inference among its families;
    # one family the compiler can finish keeps the Planner in play.
    context = _context()
    types = candidate_analysis_types(context)
    assert "association_study" in types and "causal_inference" in types
    assert families_requiring_family_result_contract(
        context, analysis_types=types, sealed_survival_suite=False
    ) == ()

    llm, stopped = _plan(context)

    assert getattr(stopped, "reason_code", None) != UNWRITABLE
    assert llm.calls


def test_a_family_template_route_owns_its_contract_and_is_not_stopped(monkeypatch):
    # A selected family template composes its own sealed contract; the stop
    # guards only the Progressive v2 compiler.
    monkeypatch.setattr(progressive_planner, "_family_spec_fallback_reason", lambda *a, **k: None)

    _llm, stopped = _plan(_family_context("causal_inference"))

    assert getattr(stopped, "reason_code", None) != UNWRITABLE


def test_a_sealed_survival_suite_keeps_survival_planning_open():
    # The host sealed a landmark survival suite for another exposure, so the
    # family router offers no template; the suite's owner is still accepted
    # by its primary method, so planning is not stopped.
    disclosure = SEALED_SURVIVAL_SUITE_MARKER + "\n" + json.dumps(
        {
            "sealed_primary_owner": "signed_landmark_survival_suite",
            "exposure_status_column": "another_exposure",
            "exposure_onset_column": "another_exposure_first_time",
            "event_column": "death",
            "followup_time_column": "followup_days_28d",
            "landmark_hours": 24,
            "endpoint_horizon_days": 28,
            "plan_outputs": ["table:survival_primary"],
        }
    )

    llm, stopped = _plan(_family_context("survival"), planning_contract_context=disclosure)

    assert getattr(stopped, "reason_code", None) != UNWRITABLE
    assert llm.calls


def test_a_design_canary_never_reaches_final_acceptance_and_is_not_stopped():
    llm, stopped = _plan(_family_context("causal_inference"), stop_after_outline=True)

    assert getattr(stopped, "reason_code", None) != UNWRITABLE
    assert llm.calls


def test_the_contract_counts_only_contexts_final_acceptance_would_reject():
    survival = _family_context("survival")
    causal = _family_context("causal_inference")

    assert families_requiring_family_result_contract(
        survival, analysis_types=("survival",), sealed_survival_suite=False
    ) == ("survival",)
    # A host-sealed landmark survival suite is accepted by its primary method.
    assert families_requiring_family_result_contract(
        survival, analysis_types=("survival",), sealed_survival_suite=True
    ) == ()
    assert families_requiring_family_result_contract(
        causal, analysis_types=("causal_inference", "survival"), sealed_survival_suite=True
    ) == ()
    # Without a declared exposure or outcome there is no family headline to require.
    for field in ("primary_exposure", "target_outcome"):
        assert families_requiring_family_result_contract(
            causal.model_copy(update={field: None}),
            analysis_types=("causal_inference",),
            sealed_survival_suite=False,
        ) == ()
    # A sealed fail-closed feasibility scope replaces the effect step.
    feasibility = causal.model_copy(
        update={
            "user_preferences": causal.user_preferences.model_copy(
                update={"formal_result_scope": "source_feasibility_fail_closed"}
            )
        }
    )
    assert families_requiring_family_result_contract(
        feasibility, analysis_types=("causal_inference",), sealed_survival_suite=False
    ) == ()
