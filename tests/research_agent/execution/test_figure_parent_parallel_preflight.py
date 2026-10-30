"""Legacy figure children retain their parent edge at execution preflight."""

from types import SimpleNamespace
import json

import pytest

from easyicu.research_agent.execution.phase import _step_emit_execution_preflight
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _preflight(steps, *, replan=False):
    findings = []
    flags = _step_emit_execution_preflight(
        plan_block_reason=None,
        plan=AnalysisPlan(research_question="Synthetic study", steps=steps),
        preexecuted_step_ids=set(),
        emit_progress=lambda *args, **kwargs: None,
        run_id="test",
        pipeline=SimpleNamespace(_enable_replanning=replan, _max_concurrent_steps=2),
        steps_to_run=steps,
        findings=findings,
        resumed_step_ids=set(),
    )
    return flags, findings


def test_figure_without_typed_input_still_requires_parent_before_render():
    parent = AnalysisStep(step_id="analysis", intent="Analyze", expected_outputs=["table:result"])
    figure = AnalysisStep(step_id="analysis_figure", intent="Render figure", expected_outputs=["figure:result"])
    (dependent, _), findings = _preflight([parent, figure])
    assert dependent
    assert any(f.validator == "typed_artifact_evidence_lineage" for f in findings)


def test_independent_steps_are_not_serialized_by_this_guard():
    steps = [AnalysisStep(step_id=x, intent="Analyze") for x in ("a", "b")]
    (dependent, _), _ = _preflight(steps)
    assert not dependent


def test_replanning_still_reports_its_sequential_boundary():
    steps = [AnalysisStep(step_id=x, intent="Analyze") for x in ("a", "b")]
    _, findings = _preflight(steps, replan=True)
    assert any(f.validator == "replanner" for f in findings)


@pytest.mark.integration
def test_pipeline_schedules_legacy_figure_after_its_parent(
    ra, synthetic_cohort, tmp_path, monkeypatch
):
    from easyicu.research_agent.execution.run_coordination import RunCoordinator
    from easyicu.research_agent.providers import mocks

    original_plan = mocks._mock_plan_json
    original_sequential = RunCoordinator.run_sequential
    original_parallel = RunCoordinator.run_parallel
    scheduled = []

    def plan_with_child(*args, **kwargs):
        plan = json.loads(original_plan(*args, **kwargs))
        for step in plan["steps"]:
            step["inputs"] = [item for item in step.get("inputs", []) if ":" not in item]
        parent_id = plan["steps"][0]["step_id"]
        plan["steps"].insert(1, {
            "step_id": parent_id + "_figure",
            "intent": "Render the parent's figure",
            "expected_outputs": ["figure:legacy_render"],
            "inputs": [],
            "method": "visualization",
            "planned_analysis_role": "auxiliary",
        })
        return json.dumps(plan)

    def sequential(self, **kwargs):
        from easyicu.research_agent.contracts.declared_product import typed_product
        assert not any(typed_product(value) for step in kwargs["state"].remaining_steps for value in step.inputs)
        scheduled.append(("sequential", [s.step_id for s in kwargs["state"].remaining_steps]))
        return original_sequential(self, **kwargs)

    def parallel(self, **kwargs):
        scheduled.append(("parallel", [s.step_id for s in kwargs["steps"]]))
        return original_parallel(self, **kwargs)

    monkeypatch.setattr(mocks, "_mock_plan_json", plan_with_child)
    monkeypatch.setattr(RunCoordinator, "run_sequential", sequential)
    monkeypatch.setattr(RunCoordinator, "run_parallel", parallel)
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path), llm=ra.MockLLMClient(), max_concurrent_steps=2,
        enable_replanning=False, enable_literature=False, enable_visual_qa=False,
        enable_memory=False, enable_latex=False,
    )
    pipeline.run(question="Describe the synthetic ICU cohort", cohort=synthetic_cohort, database="miiv")
    assert scheduled and scheduled[0][0] == "sequential"
    ids = scheduled[0][1]
    figure = next(i for i in ids if i.endswith("_figure"))
    assert ids.index(figure.removesuffix("_figure")) < ids.index(figure)
