"""A successful subset of a plan must not be delivered as a completed analysis."""

from easyicu.research_agent.plan_utils import _cap_plan_preserving_figure_steps
from easyicu.research_agent.reporting.completion import run_status_payload
from easyicu.research_agent.reporting.readiness import (
    _plan_truncation_status,
    execution_gate_status,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan():
    return AnalysisPlan(
        research_question="Describe mortality with uncertainty",
        steps=[
            AnalysisStep(step_id="estimate", intent="Estimate mortality", expected_outputs=["table: mortality"]),
            AnalysisStep(step_id="interval", intent="Calculate the confidence interval", expected_outputs=["table: confidence interval"]),
        ],
    )


def _status(plan, findings, records):
    execution = execution_gate_status(plan=plan, per_step_records=records)
    gates = {
        **execution,
        **_plan_truncation_status(findings, plan=plan),
        "analysis_validated": True,
        "manuscript_ready": True,
        "evidence_complete": True,
        "numeric_verified": True,
        "completion_status": "manuscript_ready",
    }
    return run_status_payload(
        gates=gates, research_question=plan.research_question, code_version=None,
        stop_after_analysis=False, writer_probe_mode=False,
        writer_probe_failed_steps=None, force_diagnostic_only=False,
    )


def test_truncated_plan_cannot_be_reported_as_completed():
    original = _plan()
    capped, findings = _cap_plan_preserving_figure_steps(plan=original, cap=1)
    records = [{"step_id": step.step_id, "status": "ok"} for step in capped.steps]
    status = _status(capped, findings, records)
    assert status["gates"]["execution_complete"] is True
    assert status["analysis_outcome"] == {"status": "blocked", "reason_code": "planned_products_missing"}
    assert status["manuscript_outcome"] == {"status": "not_passed", "reason_code": "planned_products_missing"}


def test_restored_plan_requires_all_steps_to_finish():
    original = _plan()
    capped, findings = _cap_plan_preserving_figure_steps(plan=original, cap=1)
    records = [{"step_id": step.step_id, "status": "ok"} for step in original.steps]
    status = _status(original, findings, records)
    assert status["gates"]["plan_truncated"] is False
    assert status["analysis_outcome"] == {"status": "completed", "reason_code": None}
    incomplete = _status(original, findings, records[:1])
    assert incomplete["analysis_outcome"]["status"] == "blocked"


def test_empty_plan_cannot_pass_execution():
    empty = AnalysisPlan(research_question="Describe mortality", steps=[])
    gate = execution_gate_status(plan=empty, per_step_records=[])
    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [{"step_id": "plan", "status": "no_steps"}]
