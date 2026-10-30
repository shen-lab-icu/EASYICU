"""Parallel supervisor branches retain each step's reflection deterministically."""

from threading import Barrier, Event, Lock

import pytest

from easyicu.research_agent.agents.roles import RuntimeSupervisor
from easyicu.research_agent.execution.run_coordination import RunCoordinator, merge_parallel_runtime_states
from easyicu.research_agent.schema import AgentRuntimeState, AnalysisStep, ReflectionMemoryEntry


def test_real_supervisor_branches_merge_in_plan_order_despite_reverse_completion():
    supervisor = RuntimeSupervisor()
    seed = ReflectionMemoryEntry(category="reusable_template", summary="existing")
    base = AgentRuntimeState(run_id="synthetic", analysis_family="descriptive", reflection_memory=[seed])
    steps = [AnalysisStep(step_id=name, intent="synthetic") for name in ("first", "second")]
    barrier, second_done, lock = Barrier(2, timeout=5), Event(), Lock()
    completed, states, errors = [], {}, []

    def execute(step):
        branch = base.model_copy(update={"current_step": step})
        state = supervisor.critique_step(state=branch, step_summary={}, evidence_refs=[], findings=[])
        barrier.wait()
        if step.step_id == "first":
            assert second_done.wait(5)
        with lock:
            states[step.step_id] = state
            completed.append(step.step_id)
        if step.step_id == "second":
            second_done.set()
        return {"status": "ok"}

    RunCoordinator().run_parallel(
        steps=steps, max_workers=2, execute_step=execute,
        submit_step=lambda pool, fn, step: pool.submit(fn, step),
        on_worker_error=lambda step, error: errors.append(error),
    )
    assert completed == ["second", "first"]
    merged = merge_parallel_runtime_states(base, [(s.step_id, states[s.step_id]) for s in steps])
    assert [entry.summary for entry in merged.reflection_memory] == [
        "existing", "first triggered critique status=blocked.", "second triggered critique status=blocked."
    ]
    assert merged.current_step.step_id == "second"
    assert [entry.summary for entry in base.reflection_memory] == ["existing"]
    assert not errors


def test_merge_refuses_replaced_prior_memory_or_changed_run():
    seed = ReflectionMemoryEntry(category="reusable_template", summary="existing")
    base = AgentRuntimeState(run_id="run", reflection_memory=[seed])
    with pytest.raises(ValueError, match="prior reflections"):
        merge_parallel_runtime_states(base, [("a", base.model_copy(update={"reflection_memory": []}))])
    with pytest.raises(ValueError, match="scientific scope"):
        merge_parallel_runtime_states(base, [("a", base.model_copy(update={"run_id": "other"}))])


def test_failed_early_step_keeps_last_advanced_step_state():
    base = AgentRuntimeState(run_id="run")
    step = AnalysisStep(step_id="a", intent="synthetic")
    advanced = base.model_copy(update={"current_step": step})
    merged = merge_parallel_runtime_states(base, [("a", advanced), ("b", base)])
    assert merged.current_step.step_id == "a"


@pytest.mark.integration
def test_pipeline_passes_frozen_state_to_parallel_workers(ra, synthetic_cohort, tmp_path, monkeypatch):
    import json
    from easyicu.research_agent.execution import phase
    from easyicu.research_agent.providers import mocks

    mock_plan = mocks._mock_plan_json
    real_merge = phase.merge_parallel_runtime_states
    merge_calls = []

    def independent_plan(*args, **kwargs):
        plan = json.loads(mock_plan(*args, **kwargs))
        for step in plan["steps"]:
            step["inputs"] = [value for value in step.get("inputs", []) if ":" not in value]
        return json.dumps(plan)

    def capture_merge(base, states):
        result = real_merge(base, states)
        merge_calls.append((base, states, result))
        return result

    monkeypatch.setattr(mocks, "_mock_plan_json", independent_plan)
    # This synthetic fixture's augmented plan has figure and cohort dependencies.
    # Force only this offline diagnostic through the existing parallel branch to
    # exercise its state handoff; this is not a scientific scheduling verdict.
    monkeypatch.setattr(phase, "_step_emit_execution_preflight", lambda **kwargs: (False, False))
    monkeypatch.setattr(phase, "merge_parallel_runtime_states", capture_merge)
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path), llm=ra.MockLLMClient(), max_concurrent_steps=2,
        enable_replanning=False, enable_literature=False, enable_visual_qa=False,
        enable_memory=False, enable_latex=False,
    )
    pipeline.run(question="Describe the synthetic ICU cohort", cohort=synthetic_cohort, database="miiv")
    assert len(merge_calls) == 1
    base, states, result = merge_calls[0]
    assert len(states) >= 2
    assert all(state.reflection_memory[:len(base.reflection_memory)] == base.reflection_memory for _, state in states)
    expected_count = len(base.reflection_memory) + sum(len(state.reflection_memory) - len(base.reflection_memory) for _, state in states)
    assert len(result.reflection_memory) == expected_count
