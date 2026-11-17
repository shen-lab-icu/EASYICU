"""A family template's host candidates use the shared Progressive repair lane.

A rejected host candidate is repaired by the current-step owner and an
unaccepted final suffix by the bounded suffix owner; neither the family path
nor a host fallback gains a second retry budget.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


@pytest.mark.parametrize("family", ["prediction", "phenotyping"])
@pytest.mark.parametrize("failures", [1, 99])
@pytest.mark.parametrize("malformed", [False, True])
def test_family_final_bindings_use_the_existing_bounded_suffix_lane(monkeypatch, family, failures, malformed):
    from easyicu.research_agent.agents import progressive_planner as owner
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
    from . import family_spec_fixtures as fixture

    context = fixture._prediction_context() if family == "prediction" else fixture._phenotyping_context()
    request = fixture._request(context, cohort_mode=None)
    features = ["hr_max", "lactate_max", "map_min"]
    payload = (fixture._prediction_payload(request, features=features) if family == "prediction"
               else fixture._phenotyping_payload(request, features=features, baseline=["age", "sex"], membership=None))
    fake = ScriptedMockLLMClient([json.dumps(payload)])
    agent = owner.ProgressivePlannerAgent(fake)
    original = owner.ProgressivePlannerAgent._compile_and_accept
    calls, checkpoints = [], []
    before = context.model_dump_json()

    def accept(self, skeleton, **kwargs):
        result = original(self, skeleton, **kwargs)
        calls.append(skeleton)
        if len(calls) <= failures:
            index = next(i for i, step in enumerate(skeleton.steps) if step.planned_analysis_role == "primary")
            if malformed:
                fake.responses.extend(["{}"] * 3)
            else:
                fake.responses.extend(item.model_dump_json() for item in self._attempt.materializations[index:])
            raise ProgressivePlanCompileError(
                "progressive_context_bindings_invalid", "Synthetic prepared-field finding.",
                step_index=index, step_id=skeleton.steps[index].step_id, path="context_bindings",
            )
        return result

    monkeypatch.setattr(owner.ProgressivePlannerAgent, "_compile_and_accept", accept)
    kwargs = dict(
        planner_strategy=owner.FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=fixture.ALLOWED_CITATIONS,
        direct_comparator_literature_keys=fixture.DIRECT_COMPARATORS,
        comparison_literature_keys=fixture.DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=None,
        checkpoint_callback=checkpoints.append,
    )
    if malformed:
        from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure

        with pytest.raises(ProgressivePlanCompileError) as caught:
            agent.run_attempt(context, **kwargs)
        assert caught.value.reason_code == "progressive_context_bindings_invalid"
        assert isinstance(caught.value.__cause__, StructuredResponseFailure)
        assert len(calls) == 1
        assert len(fake.calls) == 4  # one spec and the existing three parse attempts
    elif failures == 99:
        with pytest.raises(ProgressivePlanCompileError) as caught:
            agent.run_attempt(context, **kwargs)
        assert caught.value.reason_code == "progressive_context_bindings_invalid"
        assert len(calls) == 3  # initial acceptance plus exactly two revisions
    else:
        result = agent.run_attempt(context, **kwargs)
        assert result.output.steps
        assert len(calls) == 2
        complete = next(cp for cp in checkpoints if len(cp.materializations) == len(calls[0].steps))
        final = checkpoints[-1]
        index = next(i for i, step in enumerate(calls[0].steps) if step.planned_analysis_role == "primary")
        assert complete.materializations[:index] == final.materializations[:index]
        assert complete.foundation == final.foundation
        assert complete.outline == final.outline
        assert result.facts.prompt_metrics["suffix_revision_count"] == 1
        assert result.facts.prompt_metrics["full_revision_count"] == 0
        assert "progressive_context_bindings_invalid" in fake.calls[1][0][-1].content
    assert context.model_dump_json() == before


@pytest.mark.parametrize("spent", [0, 2])
def test_a_rejected_family_host_candidate_uses_current_step_repair(monkeypatch, spent):
    from easyicu.research_agent.agents import family_spec_planner as family_owner
    from easyicu.research_agent.agents import progressive_planner as owner
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
    from . import family_spec_fixtures as fixture

    context = fixture._prediction_context()
    request = fixture._request(context, cohort_mode=None)
    fake = ScriptedMockLLMClient([json.dumps(fixture._prediction_payload(
        request, features=["hr_max", "lactate_max", "map_min"],
    ))])
    original = family_owner.build_prediction_skeleton

    def draft(*args, **kwargs):
        result = original(*args, **kwargs)
        index = next(i for i, m in enumerate(result.materializations) if m.step.planned_analysis_role == "primary")
        valid = result.materializations[index]
        fake.responses.append(valid.model_dump_json())
        invalid = valid.model_copy(update={"step": valid.step.model_copy(update={
            "raw_inputs": [*valid.step.raw_inputs, "absent_from_the_context"],
        })})
        from dataclasses import replace
        return replace(result, materializations=tuple(
            invalid if i == index else item for i, item in enumerate(result.materializations)
        ))

    monkeypatch.setattr(family_owner, "build_prediction_skeleton", draft)
    agent = ProgressivePlannerAgent(fake)
    compile_prefix = owner.compile_progressive_prefix
    spent_once = False

    def compile_candidate(state, materialization, **kwargs):
        nonlocal spent_once
        if materialization.step.planned_analysis_role == "primary" and not spent_once:
            # Other repairs (for example foundation labels) use this same
            # current-step budget. A host fallback cannot obtain a third edit.
            agent._attempt.prompt_metrics["compile_revision_count"] += spent
            spent_once = True
        return compile_prefix(state, materialization, **kwargs)

    monkeypatch.setattr(owner, "compile_progressive_prefix", compile_candidate)
    kwargs = dict(planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=fixture.ALLOWED_CITATIONS,
        direct_comparator_literature_keys=fixture.DIRECT_COMPARATORS,
        comparison_literature_keys=fixture.DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=None,
    )
    if spent:
        with pytest.raises(ProgressivePlanCompileError) as caught:
            agent.run_attempt(context, **kwargs)
        assert caught.value.reason_code == "progressive_unknown_variable"
        assert len(fake.calls) == 1
        assert agent._attempt.prompt_metrics["compile_revision_count"] == spent
        return
    result = agent.run_attempt(context, **kwargs)
    assert len(fake.calls) == 2
    assert "progressive_unknown_variable" in fake.calls[1][0][-1].content
    assert all("absent_from_the_context" not in step.inputs for step in result.output.steps)
    assert result.facts.prompt_metrics["compile_revision_count"] == 1
