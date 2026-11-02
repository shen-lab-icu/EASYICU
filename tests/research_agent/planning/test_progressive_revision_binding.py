"""Progressive Planner revision authority during checkpoint recovery."""

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context, _foundation_payload, _materialization_payloads, _outline_payload,
)


def test_checkpoint_refuses_changed_separately_bound_revision_before_provider():
    dependency_context = {"cohort_file_sha256": "b" * 64, "llm_signature": "codex:gpt-test", "prompt_version": "test-v1"}
    source_llm = ScriptedMockLLMClient([
        json.dumps(_outline_payload()), json.dumps(_foundation_payload()),
        *[json.dumps(item) for item in _materialization_payloads()],
    ])
    source_llm.supports_strict_json_schema = True
    checkpoints = []
    def revision(level):
        return ('Host-bound request to revise the complete candidate plan.\n' + json.dumps({
            "schema_version": "easyicu.plan-change-request/1", "message_origin": "user", "user_history_verified_complete": True,
            "user_message": f"Report {level}% confidence intervals",
        }))
    ProgressivePlannerAgent(source_llm).run(
        _context(), checkpoint_callback=checkpoints.append,
        resume_dependency_context=dependency_context, bound_plan_revision_contract=revision(95),
        planning_contract_context="same planning envelope",
    )
    resumed_llm = ScriptedMockLLMClient([])
    resumed_llm.supports_strict_json_schema = True
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(resumed_llm).run(
            _context(), resume_checkpoint=checkpoints[4], resume_dependency_context=dependency_context,
            bound_plan_revision_contract=revision(90), planning_contract_context="same planning envelope",
        )
    assert caught.value.reason_code == "progressive_resume_dependency_authority_mismatch"
    assert resumed_llm.calls == []


def test_checkpoint_accepts_same_revision_and_refuses_legacy_missing_binding():
    dependency = {"cohort_file_sha256": "b" * 64, "llm_signature": "codex:gpt-test", "prompt_version": "test-v1"}
    revision = ('Host-bound request to revise the complete candidate plan.\n' + json.dumps({
        "schema_version": "easyicu.plan-change-request/1", "message_origin": "user", "user_history_verified_complete": True,
        "user_message": "Report 95% confidence intervals",
    }))
    def source(bound):
        llm = ScriptedMockLLMClient([json.dumps(_outline_payload()), json.dumps(_foundation_payload()),
                                     *[json.dumps(item) for item in _materialization_payloads()]])
        llm.supports_strict_json_schema = True
        checkpoints = []
        ProgressivePlannerAgent(llm).run(
            _context(), checkpoint_callback=checkpoints.append, resume_dependency_context=dependency,
            bound_plan_revision_contract=bound, planning_contract_context="same planning envelope",
        )
        return checkpoints[4]
    checkpoint = source(revision)
    resumed = ScriptedMockLLMClient([json.dumps(item) for item in _materialization_payloads()[3:]])
    resumed.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(resumed)
    agent.run(_context(), resume_checkpoint=checkpoint, resume_dependency_context=dependency,
              bound_plan_revision_contract=revision, planning_contract_context="same planning envelope")
    assert agent.last_result.facts.resume_validated and len(resumed.calls) == 4
    legacy = source("")
    refused = ScriptedMockLLMClient([])
    refused.supports_strict_json_schema = True
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(refused).run(
            _context(), resume_checkpoint=legacy, resume_dependency_context=dependency,
            bound_plan_revision_contract=revision, planning_contract_context="same planning envelope")
    assert caught.value.reason_code == "progressive_resume_dependency_authority_mismatch"
    assert refused.calls == []
