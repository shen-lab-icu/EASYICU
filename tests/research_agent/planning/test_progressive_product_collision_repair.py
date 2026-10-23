"""A genuine product collision uses the existing bounded current-step repair."""

from copy import deepcopy
import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

from .progressive_planner_fixtures import (
    _context, _foundation_payload, _materialization_payloads, _outline_payload, _payload,
)


@pytest.mark.parametrize("invalid_attempts", [1, 3])
def test_a_declared_collision_reaches_current_step_repair_with_the_existing_bound(invalid_attempts):
    payload = _payload()
    payload["steps"][4]["outputs"] = [{
        "product_id": "table:adjusted_association_estimates",
        "semantic_role": "adjusted_association_estimates",
    }]
    materializations = _materialization_payloads(payload)
    colliding = deepcopy(materializations[5])
    colliding["step"]["outputs"] = [{
        "product_id": "table:adjusted_association_estimates",
        "semantic_role": "scientific_sensitivity",
    }]
    responses = [
        _outline_payload(payload), _foundation_payload(payload), *materializations[:5],
        *[deepcopy(colliding) for _ in range(invalid_attempts)],
        *materializations[5:],
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    if invalid_attempts == 1:
        plan = agent.run(_context(), checkpoint_callback=checkpoints.append)
        assert len(plan.steps) == 6
        assert len(llm.calls) == 10
        assert agent.last_result.facts.prompt_metrics["compile_revision_count"] == 1
        assert agent.last_result.facts.prompt_metrics["full_revision_count"] == 0
        retained = [checkpoint.materializations[:5] for checkpoint in checkpoints
                    if len(checkpoint.materializations) >= 5]
        assert retained and all(prefix == retained[0] for prefix in retained)
    else:
        with pytest.raises(ProgressivePlanCompileError) as caught:
            agent.run(_context(), checkpoint_callback=checkpoints.append)
        assert caught.value.reason_code == "progressive_product_has_multiple_owners"
        assert caught.value.step_id == "06_sensitivity"
        assert len(llm.calls) == 10  # Three failed attempts; no fourth attempt.
    for index in range(8, min(len(llm.calls), 7 + invalid_attempts)):
        prompt = llm.calls[index][0][-1].content
        assert "HOST COMPILER OBSERVATION FOR THIS CURRENT STEP" in prompt
        assert "progressive_product_has_multiple_owners" in prompt
        assert '"step_id":"06_sensitivity"' in prompt
        assert llm.calls[index][1]["structured_output"].authority_sha256 == (
            llm.calls[7][1]["structured_output"].authority_sha256
        )
    assert "progressive_product_has_multiple_owners" in llm.calls[8][0][-1].content
