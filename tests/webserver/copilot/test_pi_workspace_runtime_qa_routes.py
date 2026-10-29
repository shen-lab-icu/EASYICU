"""The Host binds browser receipts to checked workspace bytes and one nonce."""

import json

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
from easyicu.webserver.pi_copilot.workspace_runtime_qa import RuntimeQaStore, SCHEMA
from easyicu.webserver.routes import pi_copilot as routes


@pytest.fixture
def browser_receipts(tmp_path, monkeypatch):
    class Service:
        get_workspace_preview = PiCopilotService.get_workspace_preview
        begin_workspace_runtime_qa = PiCopilotService.begin_workspace_runtime_qa
        get_workspace_runtime_qa = PiCopilotService.get_workspace_runtime_qa
        record_workspace_runtime_qa = PiCopilotService.record_workspace_runtime_qa

        def _assert_project_initialized(self, project_id):
            if project_id != "project-a":
                raise PiCopilotError("pi_project_not_initialized", "Not initialized", status_code=409)
            return project_id

    service = Service()
    service.workspace = ProjectWorkspace(tmp_path / "workspace")
    service.runtime_qa = RuntimeQaStore()
    contract = {"schema": SCHEMA, "checks": [
        {"id": "total", "kind": "text_number", "selector": "#n", "expected": 200},
    ]}
    source = '<p id="n">200</p><script type="application/json" data-easyicu-runtime-qa>' + json.dumps(contract) + '</script>'
    service.workspace.write_file("project-a", "demo.html", source)
    checked = service.workspace.check_file("project-a", "demo.html")
    body = {"file": "demo.html", "checked_sha256": checked["checked_sha256"]}
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    return service, TestClient(app), body


def _report(digest, *, status="passed"):
    return {"schema": SCHEMA, "source_sha256": digest, "runtime_errors": 0, "warnings": [],
            "checks": [{"id": "total", "kind": "text_number", "status": status,
                        "code": "number_matches" if status == "passed" else "number_mismatch"}]}


def test_runtime_receipt_needs_real_current_browser_report(browser_receipts):
    service, client, body = browser_receipts
    url = "/api/copilot/pi/projects/project-a/workspace/runtime-qa"
    assert client.get(url, params=body).json()["status"] == "partial"
    request = client.post(url, json=body).json()
    assert request["code"] == "browser_runtime_pending"
    assert request["scientific_verified"] is False
    preview = client.get("/api/copilot/pi/projects/project-a/workspace/preview", params={**body, "runtime_nonce": request["nonce"]})
    assert preview.status_code == 200
    assert "easyicu:runtime-qa-child" in preview.text
    assert "allow-same-origin" not in preview.text
    assert "allow-downloads" not in preview.headers["content-security-policy"]
    response = client.post(url + "/report", json={**body, "nonce": request["nonce"], "report": _report(body["checked_sha256"])})
    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert response.json()["browser_download_verified"] is False
    assert response.json()["scientific_verified"] is False
    assert client.get(url, params=body).json()["status"] == "complete"
    # Neither a previous report nor its capability can certify new bytes.
    service.workspace.edit_file("project-a", "demo.html", old_text='>200</p>', new_text='>201</p>', expected_sha256=body["checked_sha256"])
    assert client.get(url, params=body).status_code == 409
    assert client.post(url + "/report", json={**body, "nonce": request["nonce"], "report": _report(body["checked_sha256"])}).status_code == 409


def test_runtime_nonce_is_bound_and_renewal_revokes_previous(browser_receipts):
    _, client, body = browser_receipts
    url = "/api/copilot/pi/projects/project-a/workspace/runtime-qa"
    old = client.post(url, json=body).json()
    current = client.post(url, json=body).json()
    assert old["nonce"] != current["nonce"]
    for nonce in ("a" * 64, old["nonce"]):
        assert client.get("/api/copilot/pi/projects/project-a/workspace/preview", params={**body, "runtime_nonce": nonce}).status_code == 409
        assert client.post(url + "/report", json={**body, "nonce": nonce, "report": _report(body["checked_sha256"])}).status_code == 409
    wrong_source = _report("b" * 64)
    assert client.post(url + "/report", json={**body, "nonce": current["nonce"], "report": wrong_source}).status_code == 409


def test_runtime_sha_coordinates_use_canonical_workspace_digest(browser_receipts):
    service, client, body = browser_receipts
    url = "/api/copilot/pi/projects/project-a/workspace/runtime-qa"
    uppercase = {**body, "checked_sha256": body["checked_sha256"].upper()}
    assert client.post(url, json=uppercase).status_code == 422
    request = service.begin_workspace_runtime_qa(
        project_id="project-a", relative_file=body["file"], checked_sha256=uppercase["checked_sha256"],
    )
    assert request["source_sha256"] == body["checked_sha256"]
    assert client.get("/api/copilot/pi/projects/project-a/workspace/preview", params={**body, "runtime_nonce": request["nonce"]}).status_code == 200
    assert service.record_workspace_runtime_qa(project_id="project-a", relative_file=body["file"],
        checked_sha256=uppercase["checked_sha256"], nonce=request["nonce"], report=_report(body["checked_sha256"]))["status"] == "complete"
    assert service.get_workspace_runtime_qa(project_id="project-a", relative_file=body["file"], checked_sha256=uppercase["checked_sha256"])["status"] == "complete"


def test_runtime_report_is_bounded_and_normalized(browser_receipts):
    _, client, body = browser_receipts
    url = "/api/copilot/pi/projects/project-a/workspace/runtime-qa"
    request = client.post(url, json=body).json()
    report = _report(body["checked_sha256"], status="failed")
    report.update(status="complete", scientific_verified=True, browser_download_verified=True)
    response = client.post(url + "/report", json={**body, "nonce": request["nonce"], "report": report})
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["scientific_verified"] is False
    assert response.json()["browser_download_verified"] is False
    assert client.post(url + "/report", content=b"x" * 32769).status_code == 413
    assert client.post(url + "/report", json={**body, "nonce": request["nonce"], "report": report, "extra": True}).status_code == 422


def test_absent_contract_stays_partial_even_for_empty_error_free_report(browser_receipts):
    service, client, _ = browser_receipts
    service.workspace.write_file("project-a", "plain.html", "<p>demo</p>")
    body = {"file": "plain.html", "checked_sha256": service.workspace.check_file("project-a", "plain.html")["checked_sha256"]}
    url = "/api/copilot/pi/projects/project-a/workspace/runtime-qa"
    request = client.post(url, json=body).json()
    report = _report(body["checked_sha256"])
    report["checks"] = []
    response = client.post(url + "/report", json={**body, "nonce": request["nonce"], "report": report})
    assert response.json()["status"] == "partial"
