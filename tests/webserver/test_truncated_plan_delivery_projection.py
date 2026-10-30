"""Readiness survives the web job boundary without becoming a false success."""

import json

from easyicu.webserver.agent_pipeline_runs import (
    _failed_closed_detail, _gate_from_axes, _readiness_axes,
)


def test_web_job_remains_blocked_until_truncated_products_are_restored(tmp_path):
    gates = {
        "execution_complete": True,
        "analysis_validated": True,
        "evidence_complete": True,
        "numeric_verified": True,
        "manuscript_ready": True,
        "manuscript_requested": True,
        "plan_truncated": True,
    }
    path = tmp_path / "run_status.json"
    path.write_text(json.dumps({"gates": gates}), encoding="utf-8")
    axes = _readiness_axes(tmp_path)
    projection = _gate_from_axes(axes, pending=False)
    assert projection["status"] == "blocked"
    assert projection["analysis_outcome"] == {"status": "blocked", "reason_code": "planned_products_missing"}
    assert _failed_closed_detail(axes) == {"reason_code": "planned_products_missing"}

    gates["plan_truncated"] = False
    path.write_text(json.dumps({"gates": gates}), encoding="utf-8")
    projection = _gate_from_axes(_readiness_axes(tmp_path), pending=False)
    assert projection["status"] == "analysis_only"
    assert projection["analysis_outcome"]["status"] == "completed"
