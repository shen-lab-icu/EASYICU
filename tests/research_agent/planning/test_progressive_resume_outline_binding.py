"""A resumed prefix carries the outline rules it was bound under.

The host binds an outline once, when the Planner's outline is parsed. A
checkpoint written before a binding rule changed would otherwise resume with
the old outline and repeat the failure the new rule repairs, so a resume
checks the saved outline against the current binding before any Provider call.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents import progressive_planner
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.planning.progressive_resume import (
    validate_progressive_resume_outline_binding,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_DEPENDENCIES = {
    "cohort_file_sha256": "b" * 64,
    "llm_signature": "codex:gpt-test",
    "prompt_version": "test-v1",
}


def _source_checkpoints():
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    llm.supports_strict_json_schema = True
    checkpoints = []
    ProgressivePlannerAgent(llm).run(
        _context(),
        checkpoint_callback=checkpoints.append,
        resume_dependency_context=_DEPENDENCIES,
    )
    return checkpoints


def test_a_binding_rule_added_after_the_checkpoint_stops_the_resume(monkeypatch) -> None:
    checkpoint = _source_checkpoints()[4]
    last = checkpoint.outline.steps[-1]
    original = progressive_planner._bind_runtime_action_dependencies

    def with_a_new_edge(outline):
        # Stands in for a host rule that now implies one more product edge.
        bound = original(outline)
        steps = list(bound.steps)
        steps[-1] = steps[-1].model_copy(
            update={"depends_on": [*steps[-1].depends_on, "newly_implied_owner"]}
        )
        return bound.model_copy(update={"steps": steps})

    monkeypatch.setattr(
        progressive_planner, "_bind_runtime_action_dependencies", with_a_new_edge
    )
    resumed_llm = ScriptedMockLLMClient([])
    resumed_llm.supports_strict_json_schema = True

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(resumed_llm).run(
            _context(),
            resume_checkpoint=checkpoint,
            resume_dependency_context=_DEPENDENCIES,
        )

    assert caught.value.reason_code == "progressive_resume_outline_binding_changed"
    assert caught.value.details["findings"] == [{"changed_step_ids": [last.step_id]}]
    assert resumed_llm.calls == []


def test_an_outline_bound_by_the_current_host_resumes() -> None:
    checkpoint = _source_checkpoints()[4]

    validate_progressive_resume_outline_binding(
        checkpoint=checkpoint,
        bound_outline=checkpoint.outline,
    )
