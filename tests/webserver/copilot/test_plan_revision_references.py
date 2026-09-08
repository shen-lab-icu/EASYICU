"""Revision context must carry saved scientific content, never invented plans."""
from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot import tools as owner
from easyicu.webserver.plan_change_request import PlanChangeRequest


def _plan():
    return {"research_question": "Biomarker and outcome", "steps": [
        {"step_id": "baseline", "method": "descriptive_cohort_summary",
         "intent": "Describe the prior clinical roster.", "planned_analysis_role": "auxiliary",
         "inputs": ["age", "severity", "artifact:analysis_cohort"],
         "expected_outputs": ["table:cohort_summary"], "table_one_spec": None},
    ]}


def test_inspect_plan_exposes_actual_variable_and_product_contracts():
    projected = owner._plan_projection(_plan())
    assert projected["steps"][0]["inputs"] == _plan()["steps"][0]["inputs"]
    assert projected["steps"][0]["expected_outputs"] == ["table:cohort_summary"]


def test_revision_includes_only_bound_current_and_user_named_plans(monkeypatch):
    rows = [{"run_id": f"run_{name}", "project_dir": f"/host/{name}"} for name in ("current", "original", "other")]
    context = SimpleNamespace(user_message="请恢复 run_original 的完整基线。")
    monkeypatch.setattr(owner, "_run_rows", lambda _context: rows)
    reads = []
    def read(directory, artifact):
        reads.append((directory, artifact))
        return {"ok": True, "artifact": {"sha256": "a" * 64}, "payload": _plan()}
    monkeypatch.setattr(owner.agent_runs, "read_run_artifact", read)
    references = owner._plan_change_references(context, rows[0])
    assert [ref.run_id for ref in references] == ["run_current", "run_original"]
    assert len(reads) == 2
    change = PlanChangeRequest(source_run_id="run_current", user_message=context.user_message, reference_plans=references)
    assert change.reference_concepts({"age", "severity", "unrelated"}) == ("age", "severity")
    assert "severity" in change.planner_context()
    assert "/host" not in change.planner_context()
    context.user_message = "请恢复 run_foreign 的基线。"
    with pytest.raises(owner.PiCopilotError, match="referenced plan"):
        owner._plan_change_references(context, rows[0])
    assert len(reads) == 2


def test_revision_reference_variables_stay_in_zero_row_menu(tmp_path):
    import json
    import pyarrow.parquet as pq
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.plan_change_request import ReferencedPlan

    plan = _plan()
    plan["steps"][0]["inputs"] = ["age", "hr", "ph", "wbc", "artifact:analysis_cohort"]
    change = PlanChangeRequest(source_run_id="run_current", user_message="Restore the prior baseline.", reference_plans=(
        ReferencedPlan(run_id="run_original", artifact_sha256="b" * 64, plan=plan),
    ))
    llm = ScriptedMockLLMClient([json.dumps({"selected_concepts": ["lact", "death"], "inclusion_exclusion": [], "rationale": "Primary anchors."})])
    result = agent_pipeline_runs._metadata_only_planning_acquisition(
        database="miiv", question="Lactate and hospital mortality", llm=llm,
        output_dir=tmp_path / "metadata", target_outcome="death", plan_change_request=change,
    )
    assert not result.blocked
    assert {"age", "hr", "ph", "wbc"} <= set(pq.read_schema(result.universe_path).names)
    assert pq.read_metadata(result.universe_path).num_rows == 0
    assert "Restore the prior baseline." in llm.calls[0][0][1].content
    assert "run_original" in llm.calls[0][0][1].content


@pytest.mark.parametrize("failure", ["privacy", "digest"])
def test_unreadable_reference_cannot_be_silently_omitted(monkeypatch, failure):
    context = SimpleNamespace(user_message="Revise the plan.")
    row = {"run_id": "run_current", "project_dir": "/host/current"}
    monkeypatch.setattr(owner, "_run_rows", lambda _: [row])
    result = {"ok": False, "error": "artifact_privacy_scan_failed"} if failure == "privacy" else {
        "ok": True, "payload": _plan(), "artifact": {"sha256": "bad"},
    }
    monkeypatch.setattr(owner.agent_runs, "read_run_artifact", lambda *_: result)
    with pytest.raises((owner.PiCopilotError, ValueError)):
        owner._plan_change_references(context, row)
