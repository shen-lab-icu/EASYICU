"""An outline that commits a family no owner can finish stops at once.

The planner stops when its outline selects a causal or unsealed survival
family -- with ``tte_trial_not_confirmed`` for a causal one, which is planned
only as the emulation of a confirmed target trial, and with
``progressive_family_result_contract_unwritable`` otherwise -- because final
acceptance would reject every later request. That check ran only after the
outline passed its other checks. A causal outline rejected for an unrelated
violation was retried first, and the retries kept the committed family: the
H2 web rerun spent four outline requests before the same stop. The stop is now
read from the first outline that parses. An outline that commits an executable
family, a design canary and a sealed survival suite are validated and retried
as before.

Synthetic contexts only.
"""

from __future__ import annotations

import copy
import json

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.planning.family_spec.request import (
    SEALED_SURVIVAL_SUITE_MARKER,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _cox_outline_payload,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

UNWRITABLE = "progressive_family_result_contract_unwritable"
#: Each family's stop when no owner can write its result contract.
STOPS = {"causal_inference": "tte_trial_not_confirmed", "survival": UNWRITABLE}
QUESTIONS = {
    "causal_inference": "Estimate the effect of exposure_flag on outcome_flag.",
    "survival": "Estimate time to outcome_flag by exposure_flag (survival).",
    "association_study": "Estimate the effect of exposure_flag on outcome_flag.",
}
PRIMARY_ACTIONS = {
    "causal_inference": "causal_emulation.iptw_or",
    "survival": "time_to_event.cox_hr",
}
SEALED_SURVIVAL = SEALED_SURVIVAL_SUITE_MARKER + "\n" + json.dumps(
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


def _outline_for(family: str) -> dict:
    if family == "association_study":
        return _outline_payload()
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


def _with_unregistered_citation(outline: dict) -> dict:
    # Parses, and the outline validator rejects it: an ordinary retryable error.
    rejected = copy.deepcopy(outline)
    rejected["steps"][0]["literature_citation_keys"] = ["unregistered_key_2026"]
    return rejected


def _plan(family: str, first, *, strategy: str = "progressive_v2", **kwargs):
    context = _context().model_copy(update={"research_question": QUESTIONS[family]})
    responses = [
        first if isinstance(first, str) else json.dumps(first),
        json.dumps(_outline_for(family)),
        json.dumps(_foundation_payload()),
        *[json.dumps(item) for item in _materialization_payloads()],
    ]
    llm = ScriptedMockLLMClient(responses)
    try:
        result = ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy=strategy, **kwargs
        )
    except Exception as exc:  # noqa: BLE001 - the test reads which stop it was
        return llm, exc
    return llm, result


@pytest.mark.parametrize("strategy", [FAMILY_SPEC_STRATEGY, "progressive_v2"])
@pytest.mark.parametrize("family", ["causal_inference", "survival"])
def test_a_committed_family_no_owner_can_finish_is_not_retried(family, strategy):
    llm, stopped = _plan(
        family, _with_unregistered_citation(_outline_for(family)), strategy=strategy
    )

    assert isinstance(stopped, ProgressivePlanCompileError)
    assert stopped.reason_code == STOPS[family]
    assert stopped.path == "analysis_type"
    assert family in str(stopped)
    # The first outline was the only request: no retry, foundation or step call.
    assert len(llm.calls) == 1


def test_an_executable_family_with_the_same_violation_is_still_retried():
    llm, result = _plan(
        "association_study", _with_unregistered_citation(_outline_for("association_study"))
    )

    assert getattr(result, "reason_code", None) != UNWRITABLE
    # The rejected outline was retried, then the plan went on to its steps.
    assert len(llm.calls) > 2


def test_a_design_canary_still_validates_and_retries_its_outline():
    llm, result = _plan(
        "causal_inference",
        _with_unregistered_citation(_outline_for("causal_inference")),
        stop_after_outline=True,
    )

    assert getattr(result, "reason_code", None) not in STOPS.values()
    assert result.output.analysis_type == "causal_inference"
    assert len(llm.calls) == 2


def test_a_sealed_survival_suite_still_validates_and_retries_its_outline():
    llm, result = _plan(
        "survival",
        _with_unregistered_citation(_outline_for("survival")),
        planning_contract_context=SEALED_SURVIVAL,
    )

    assert getattr(result, "reason_code", None) != UNWRITABLE
    assert len(llm.calls) > 2


def test_an_outline_that_does_not_parse_is_retried_before_the_stop():
    llm, stopped = _plan("causal_inference", "not an outline")

    # Nothing committed a family, so the parse retry runs; the next outline does.
    assert isinstance(stopped, ProgressivePlanCompileError)
    assert stopped.reason_code == STOPS["causal_inference"]
    assert len(llm.calls) == 2
