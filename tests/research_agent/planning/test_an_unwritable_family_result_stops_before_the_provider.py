"""A plan no host owner can finish stops before the Provider, with a typed reason.

Final acceptance needs a causal or survival primary step to carry the
Planner-declared ``family_primary_result_requirement`` once the context names
its exposure and outcome.  The Progressive v2 compiler never writes that
field, and the family router offers no template for such a context unless the
host has sealed a suite for it.  A causal question and an unsealed survival
question therefore spent the whole planning budget -- outline, steps, bounded
suffix repairs -- and failed only at final acceptance.  The planner now stops
with ``progressive_family_result_contract_unwritable``: before its first Provider
call when every candidate family needs that contract, and otherwise as soon as
the outline selects such a family, without retrying the choice.  When the
family-spec strategy has a host template for the family (a survival question
with a proposable landmark suite), an owner exists that Progressive v2 does
not use, and the stop says so with ``progressive_family_template_required``.
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
from easyicu.research_agent.planning.family_spec import family_template_id_for_context
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
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _outline_context,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _cox_outline_payload,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)
from tests.support.survival_proposal import survival_context

UNWRITABLE = "progressive_family_result_contract_unwritable"
TEMPLATE_REQUIRED = "progressive_family_template_required"
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


def test_a_survival_question_a_family_template_can_plan_names_that_strategy():
    # The host can propose a landmark survival suite for this cohort, so the
    # family-spec strategy plans the question from its template.
    context = survival_context(research_question=QUESTIONS["survival"])
    assert candidate_analysis_types(context) == ("survival",)
    assert family_template_id_for_context(context, analysis_types=("survival",)) is not None

    llm, stopped = _plan(context, strategy="progressive_v2")

    assert isinstance(stopped, ProgressivePlanCompileError)
    assert stopped.reason_code == TEMPLATE_REQUIRED
    assert stopped.path == "analysis_type"
    assert FAMILY_SPEC_STRATEGY in str(stopped)
    assert llm.calls == []

    routed_llm, routed = _plan(context)
    assert getattr(routed, "reason_code", None) not in {UNWRITABLE, TEMPLATE_REQUIRED}
    assert routed_llm.calls


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
    monkeypatch.setattr(
        progressive_planner, "family_spec_fallback_reason", lambda *a, **k: None
    )

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


# The outline stage: the question also offers a family the compiler can
# finish, so the Planner is asked, and its outline commits the family.
OUTLINE_QUESTIONS = {
    "causal_inference": "Estimate the effect of exposure_flag on outcome_flag.",
    "survival": "Estimate time to outcome_flag by exposure_flag (survival).",
}


PRIMARY_ACTIONS = {
    "causal_inference": "causal_emulation.iptw_or",
    "survival": "time_to_event.cox_hr",
}


def _outline_for(family: str) -> dict:
    # The survival fixture's custom primary, with the family's own action.
    outline = _cox_outline_payload()
    outline["analysis_type"] = family
    for candidate in outline["design_selection"]["candidates"]:
        candidate["analysis_type"] = family
    primary = next(step for step in outline["steps"] if step["step_id"] == "05_primary")
    primary["scientific_action_id"] = PRIMARY_ACTIONS[family]
    if family == "causal_inference":
        # A causal article requires a robustness owner.
        outline["steps"].insert(
            outline["steps"].index(primary) + 1,
            {
                **primary,
                "step_id": "06_sensitivity",
                "planned_analysis_role": "sensitivity",
                "objective": "Bound the effect against unmeasured confounding.",
                "depends_on": ["05_primary"],
                "scientific_action_id": "causal_emulation.evalue",
            },
        )
    return outline


def _plan_from_outline(family: str, outline: dict, *, strategy="progressive_v2", **kwargs):
    context = _outline_context().model_copy(
        update={"research_question": OUTLINE_QUESTIONS[family]}
    )
    responses = [outline, _foundation_payload(), *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    try:
        result = ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy=strategy, **kwargs
        )
    except Exception as exc:  # noqa: BLE001 - the test reads which stop it was
        return context, llm, exc
    return context, llm, result


@pytest.mark.parametrize("strategy", [FAMILY_SPEC_STRATEGY, "progressive_v2"])
@pytest.mark.parametrize("family", ["causal_inference", "survival"])
def test_an_outline_that_selects_a_family_no_owner_can_finish_stops_before_its_steps(
    family, strategy
):
    context, llm, stopped = _plan_from_outline(family, _outline_for(family), strategy=strategy)
    types = candidate_analysis_types(context)
    assert family in types
    assert families_requiring_family_result_contract(
        context, analysis_types=types, sealed_survival_suite=False
    ) == ()

    assert isinstance(stopped, ProgressivePlanCompileError)
    assert stopped.reason_code == UNWRITABLE
    assert stopped.path == "analysis_type"
    assert family in str(stopped)
    # The outline was the only request: no foundation, step or retry call.
    assert len(llm.calls) == 1


def test_an_outline_that_selects_an_executable_family_goes_on_to_its_steps():
    outline = _outline_payload()
    assert outline["analysis_type"] == "association_study"

    _context_, llm, result = _plan_from_outline("causal_inference", outline)

    assert getattr(result, "reason_code", None) != UNWRITABLE
    assert len(llm.calls) > 1


def test_a_sealed_survival_suite_keeps_a_survival_outline_open():
    disclosure = SEALED_SURVIVAL_SUITE_MARKER + "\n" + json.dumps(
        {
            "sealed_primary_owner": "signed_landmark_survival_suite",
            "exposure_status_column": "exposure_flag",
            "exposure_onset_column": "exposure_flag_first_time",
            "event_column": "outcome_flag",
            "followup_time_column": "followup_days_28d",
            "landmark_hours": 24,
            "endpoint_horizon_days": 28,
            "plan_outputs": ["table:survival_primary"],
        }
    )

    _context_, llm, result = _plan_from_outline(
        "survival", _outline_for("survival"), planning_contract_context=disclosure
    )

    assert getattr(result, "reason_code", None) != UNWRITABLE
    assert len(llm.calls) > 1


def test_a_design_canary_returns_its_causal_outline():
    _context_, llm, result = _plan_from_outline(
        "causal_inference", _outline_for("causal_inference"), stop_after_outline=True
    )

    assert getattr(result, "reason_code", None) != UNWRITABLE
    assert result.output.analysis_type == "causal_inference"
    assert len(llm.calls) == 1
