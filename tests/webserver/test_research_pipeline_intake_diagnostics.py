"""Real intake failures retain actionable diagnostics across Web and Pi adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from easyicu.webserver import dataio, research_run_submission
from easyicu.webserver.pi_copilot.projections import ensure_safe_projection
from easyicu.webserver.routes import agent as agent_route
from tests.webserver.copilot.test_pi_copilot_research_workflow import (
    _run_submission_rejection,
    _write_pipeline_export,
)


def _rejected(export: Path) -> dict:
    with pytest.raises(dataio.ExportCohortError) as raised:
        dataio.validate_research_pipeline_source(str(export), database="miiv")
    return raised.value.detail


def test_broken_manifest_returns_the_intake_failure_through_pi(tmp_path, monkeypatch):
    export = _write_pipeline_export(tmp_path / "private-package")
    (export / "_manifest.json").write_text("{not-json")

    detail = _rejected(export)
    result = _run_submission_rejection(monkeypatch, detail)

    assert result["status"] == "blocked"
    assert result["code"] == "research_pipeline_manifest_invalid"
    assert result["details"]["intake_error_code"] == "export_manifest_json_invalid"
    assert "JSON" in result["details"]["intake_error_message"]
    assert "easyicu_start_extraction" in result["summary"]
    assert str(export) not in json.dumps(result)


@pytest.mark.parametrize(
    "member",
    [
        "missing.csv",
        "tables/missing.csv",
        "tables/../missing.csv",
        "/Users/private/export.csv",
        r"C:\Users\private\export.csv",
        r"C:private.csv",
        r"\\server\share\export.csv",
        "bad\nname.csv",
        "bad\x00name.csv",
        "bad\x1bname.csv",
        "patient_id_123.csv",
        "sk-sensitivecredential.csv",
    ],
)
def test_manifest_member_diagnostics_never_reflect_private_coordinates(tmp_path, member):
    export = _write_pipeline_export(tmp_path / "export")
    manifest_path = export / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["file"] = member
    manifest_path.write_text(json.dumps(manifest))

    detail = _rejected(export)

    assert detail["intake_error_code"] in {"manifest_file_missing", "manifest_path_escape"}
    assert detail["intake_error_message"]
    assert "intake_member" not in detail
    assert detail["intake_member_sha256"] == hashlib.sha256(member.encode()).hexdigest()
    assert member not in json.dumps(detail)
    assert ensure_safe_projection(detail) == detail


def test_public_metadata_member_name_and_its_failure_remain_readable(tmp_path):
    export = _write_pipeline_export(tmp_path / "export")
    manifest_path = export / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["file"] = "feature_definitions.csv"
    manifest_path.write_text(json.dumps(manifest))

    detail = _rejected(export)

    assert detail["intake_error_code"] == "manifest_file_missing"
    assert detail["intake_member"] == "feature_definitions.csv"
    assert "missing" in detail["intake_error_message"]
    assert ensure_safe_projection(detail) == detail


def test_http_adapter_preserves_real_manifest_diagnostic_before_launch(tmp_path, monkeypatch):
    export = _write_pipeline_export(tmp_path / "export")
    (export / "_manifest.json").write_text("{not-json")
    assert dataio.describe_export_source(str(export))["ok"] is True
    study = {
        "id": "intake-diagnostic-study",
        "revision": 1,
        "question": "Describe the available cohort.",
        "data_source": {"path": str(export), "database": "miiv"},
    }
    monkeypatch.setattr(research_run_submission.context_store, "get_context", lambda _: study)
    monkeypatch.setattr(
        research_run_submission,
        "build_research_workflow_snapshot",
        lambda **_: SimpleNamespace(planning_prerequisites_missing=[]),
    )
    monkeypatch.setattr(
        research_run_submission,
        "_submit_job",
        lambda *_: pytest.fail("Invalid intake must not submit a job"),
    )
    app = FastAPI()
    app.include_router(agent_route.control_router)

    response = TestClient(app).post(
        "/api/jobs/agent-run",
        json={
            "engine": "research_agent_pipeline",
            "study_context_id": study["id"],
            "candidate_plan_only": True,
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "research_pipeline_manifest_invalid"
    assert detail["intake_error_code"] == "export_manifest_json_invalid"
    assert ensure_safe_projection(detail) == detail
    assert str(export) not in response.text
