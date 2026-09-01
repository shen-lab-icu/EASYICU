from __future__ import annotations

import json

from fastapi.testclient import TestClient

from easyicu.webserver import agent_runs, run_artifact_disclosure
from easyicu.webserver.app import app


def test_disclosure_owner_applies_both_browser_and_artifact_policies() -> None:
    browser_scan = run_artifact_disclosure.scan_browser_projection(
        {"run_context.json": {"reason": "/Users/reviewer/private.csv"}}
    )
    artifact_scan = run_artifact_disclosure.scan_artifact_payloads(
        {"figure_gallery.json": {"patient_rows": [{"age": 72}]}}
    )

    assert browser_scan["passed"] is False
    assert artifact_scan["passed"] is False
    blocked = run_artifact_disclosure.privacy_blocked_projection(
        run_context={"run_id": "run-safe", "study_id": "study-safe"},
        scan=browser_scan,
    )
    assert blocked["quality_gate.json"]["gate"]["reportable"] is False
    assert "/Users/reviewer" not in json.dumps(blocked)


def test_row_level_json_is_withheld_from_preview_download_and_bundle(tmp_path) -> None:
    artifact_path = tmp_path / "figure_gallery.json"
    artifact_path.write_text(
        json.dumps({"patient_rows": [{"stay_id": "synthetic-secret"}]}),
        encoding="utf-8",
    )

    preview = agent_runs.read_run_artifact(str(tmp_path), artifact_path.name)
    download = agent_runs.read_run_artifact_bytes(str(tmp_path), artifact_path.name)
    bundle = agent_runs.build_run_bundle(str(tmp_path))

    for result in (preview, download, bundle):
        assert result["ok"] is False
        assert result["error"] == "artifact_privacy_scan_failed"
        assert result["privacy_scan"]["passed"] is False
        assert "payload" not in result
        assert "content" not in result


def test_patient_rows_are_withheld_even_without_explicit_identifiers(tmp_path) -> None:
    artifact_path = tmp_path / "figure_gallery.json"
    artifact_path.write_text(
        json.dumps({"patient_rows": [{"age": 72, "sex": "F"}]}),
        encoding="utf-8",
    )

    result = agent_runs.read_run_artifact(str(tmp_path), artifact_path.name)

    assert result["ok"] is False
    assert result["error"] == "artifact_privacy_scan_failed"
    assert result["privacy_scan"]["row_level_markers"] == [
        {"path": "figure_gallery.json.patient_rows", "marker": "patient_rows"}
    ]


def test_row_level_json_routes_return_conflict_without_payload(tmp_path) -> None:
    artifact_path = tmp_path / "figure_gallery.json"
    artifact_path.write_text(
        json.dumps({"patient_rows": [{"stay_id": "synthetic-secret"}]}),
        encoding="utf-8",
    )
    client = TestClient(app)

    for endpoint in (
        "/api/agent-runs/artifact",
        "/api/agent-runs/download-artifact",
        "/api/agent-runs/download-bundle",
    ):
        response = client.post(
            endpoint,
            json={"project_dir": str(tmp_path), "artifact": artifact_path.name},
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "artifact_privacy_scan_failed"
        assert detail["privacy_scan"]["passed"] is False
        assert "synthetic-secret" not in response.text
