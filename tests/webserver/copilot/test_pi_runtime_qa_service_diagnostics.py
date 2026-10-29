"""Contract failures reach both the model and browser without pending loops."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.workspace_runtime_qa import RuntimeQaError


@pytest.mark.parametrize("declared", [False, True])
def test_service_qa_readback_explains_schema_or_missing_contract_without_polling(declared) -> None:
    source = '<h1 id="n">2%</h1>'
    if declared:
        contract = {"schema": "easyicu.web-runtime-qa/1", "checks": [
            {"id": "number", "kind": "text_number", "selector": "#n", "expected": "2%"},
        ]}
        source += '<script type="application/json" data-easyicu-runtime-qa>' + json.dumps(contract) + '</script>'
    owner = SimpleNamespace(
        get_workspace_preview=lambda **_: {"artifact": {"file": "brief.html", "checked_sha256": "a" * 64, "text": source}},
        runtime_qa=SimpleNamespace(read=lambda *_: {"status": "partial", "csv_export_ready": True, "runtime_errors": []}),
    )
    receipt = PiCopilotService.get_workspace_runtime_qa(owner, project_id="project", relative_file="brief.html",
                                                      checked_sha256="a" * 64)
    assert receipt["status"] == ("failed" if declared else "partial")
    assert receipt["polling_allowed"] is False
    assert receipt["next_action"]
    if declared:
        assert receipt["contract_diagnostics"]["issues"][0]["path"] == "/checks/0/expected"
        assert "2%" not in json.dumps(receipt)
    else:
        assert receipt["csv_export_ready"] is True


def test_runtime_qa_begin_route_preserves_safe_issue_paths(monkeypatch) -> None:
    from easyicu.webserver.app import app
    from easyicu.webserver.routes import pi_copilot as routes

    class Service:
        def begin_workspace_runtime_qa(self, **kwargs):
            raise RuntimeQaError("runtime_qa_contract_invalid", issues=[
                {"path": "/checks/0/expected", "code": "runtime_qa_number_invalid", "expected": "finite number"},
            ])

    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: Service())
    with TestClient(app) as client:
        response = client.post("/api/copilot/pi/projects/project/workspace/runtime-qa",
                               json={"file": "brief.html", "checked_sha256": "a" * 64})
    assert response.status_code == 422
    assert response.json()["detail"]["details"]["issues"][0]["path"] == "/checks/0/expected"
