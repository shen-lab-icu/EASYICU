"""Synthetic completed analysis and failed derived manuscript outcomes."""

import json

import pytest

from easyicu.webserver.agent_pipeline_runs import _write_projection
from easyicu.webserver.pi_copilot.projections import project_job, project_run_outcome, project_run_row
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.run_record import RunGate
from easyicu.research_agent.reporting.completion import completion_outcomes
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)


def _project(tmp_path, **changes):
    axes = {
        "execution_complete": True,
        "analysis_validated": True,
        "evidence_complete": False,
        "numeric_verified": True,
        "manuscript_ready": False,
        "manuscript_generated": True,
        "missing_evidence_count": 0,
        "evidence_error_count": 1,
        "publication_ready": False,
        "paper_authorized": False,
        **changes,
    }
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(json.dumps({"gates": axes}))
    return _write_projection(
        wrapper_dir=tmp_path / "wrapper",
        study=complete_study(),
        provider={"provider": "openai", "model": "synthetic"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
    )


def test_a_completed_analysis_keeps_its_status_when_the_manuscript_fails(tmp_path):
    result = _project(tmp_path)
    gate = result["gate"]
    assert gate["status"] == "analysis_only"
    assert gate["analysis_outcome"] == {"status": "completed", "reason_code": None}
    assert gate["manuscript_outcome"] == {
        "status": "not_passed", "reason_code": "manuscript_checks_not_passed",
    }
    assert gate["reportable"] is False
    assert gate["draft_unlocked"] is False
    checks = {row["id"]: row["passed"] for row in gate["checks"]}
    assert checks["manuscript_ready"] is False
    assert checks["publication_ready"] is False
    assert checks["paper_authorized"] is False
    projection = project_job({
        "id": "synthetic-job", "kind": "agent-run", "status": "done", "result": result,
    })
    assert projection["analysis_results_available"] is True
    assert projection["analysis_outcome"] == gate["analysis_outcome"]
    assert projection["manuscript_outcome"] == gate["manuscript_outcome"]
    durable = project_run_outcome({"ok": True, **result})
    assert durable["analysis_outcome"] == gate["analysis_outcome"]
    assert durable["manuscript_outcome"] == gate["manuscript_outcome"]


@pytest.mark.parametrize("axis", ["execution_complete", "analysis_validated"])
def test_a_failed_analysis_stays_blocked(tmp_path, axis):
    gate = _project(tmp_path, **{axis: False})["gate"]
    assert gate["status"] == "blocked"
    assert gate["analysis_outcome"] == {
        "status": "blocked", "reason_code": f"{axis}_not_satisfied",
    }
    assert gate["manuscript_outcome"]["status"] == "not_passed"


def test_workflow_keeps_the_manuscript_reason_on_its_own_stage(tmp_path):
    gate = _project(tmp_path)["gate"]
    run = {
        "run_id": "synthetic", "run_type": "full", "engine": "easyicu.research_agent.pipeline",
        "gate_status": gate["status"], "gate_reason": gate["reason"],
        "gate_checks": {row["id"]: row["passed"] for row in gate["checks"]},
        "analysis_outcome": gate["analysis_outcome"], "manuscript_outcome": gate["manuscript_outcome"],
        "artifact_names": ["agent_plan.json", "evidence_ledger.json", "source_run_manifest.json", "result_tables.json", "manuscript_draft.json"],
    }
    snapshot = build_research_workflow_snapshot(study={"id": "synthetic", "question": "Describe the aggregate."},
        active_export_present=False, active_job=None, latest_run=run)
    stages = {row.id: row for row in snapshot.stages}
    assert stages["analysis"].status == "complete"
    assert stages["manuscript"].status == "blocked"
    assert stages["manuscript"].reason_code == "manuscript_checks_not_passed"
    assert snapshot.analysis_validation_retry_available is False
    summary = project_run_row(run)
    assert summary["analysis_outcome"] == run["analysis_outcome"]
    assert summary["manuscript_outcome"] == run["manuscript_outcome"]


def test_legacy_record_separates_a_manuscript_stop_without_promoting_any_check():
    payload = {
        "status": "blocked", "reason": "research_agent_pipeline_failed_closed",
        "detail": {"reason_code": "manuscript_checks_not_passed"},
        "reportable": False, "draft_unlocked": False,
        "checks": [{"id": "execution_complete", "passed": True}, {"id": "analysis_validated", "passed": True},
                   {"id": "evidence_complete", "passed": False}, {"id": "numeric_verified", "passed": False}],
    }
    record = RunGate.from_payload(payload)
    assert record.status == "analysis_only"
    assert record.analysis_outcome.status == "completed"
    assert record.manuscript_outcome.reason_code == "manuscript_checks_not_passed"
    assert record.to_dict()["checks"] == payload["checks"]
    assert record.to_dict()["manuscript_detail"] == payload["detail"]
    assert "detail" not in record.to_dict()
    assert payload["status"] == "blocked"


@pytest.mark.parametrize("change,status,reason", [
    ({"manuscript_requested": False}, "not_requested", None),
    ({"evidence_complete": False}, "not_passed", "evidence_complete_not_satisfied"),
    ({"numeric_verified": False}, "not_passed", "numeric_verified_not_satisfied"),
    ({"writer_stop": {"reason_code": "writer_provider_transport_unavailable", "provider_http_status": 503},
      "evidence_complete": False}, "not_passed", "writer_provider_transport_unavailable"),
    ({}, "passed", None),
])
def test_deliverable_verdicts_use_typed_codes_without_changing_analysis(change, status, reason):
    axes = {"execution_complete": True, "analysis_validated": True,
            "evidence_complete": True, "numeric_verified": True, "manuscript_ready": True, **change}
    outcomes = completion_outcomes(axes)
    assert outcomes["analysis_outcome"]["status"] == "completed"
    assert outcomes["manuscript_outcome"] == {"status": status, "reason_code": reason}
    pending = completion_outcomes(axes, pending=True)
    assert pending["analysis_outcome"]["status"] == "pending"
    assert pending["manuscript_outcome"]["status"] == "pending"
