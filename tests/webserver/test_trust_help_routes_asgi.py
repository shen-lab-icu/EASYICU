"""Run on the lead's Mac: TestClient socket sends are blocked in the server lane."""

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyicu.webserver import run_refs
from easyicu.webserver.routes import trust_help
from tests.support.trust_help import call, ledger


def test_http_trust_and_help_contracts(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(trust_help.router)
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call(payload="SYNTHETIC_PRIVATE")])
    ref = run_refs.issue(str(tmp_path))
    service = SimpleNamespace(gateway=SimpleNamespace(session_dir=tmp_path),
        _get_record=lambda identity: SimpleNamespace(session_id=identity))
    monkeypatch.setattr(trust_help, "get_pi_copilot_service", lambda: service)
    with TestClient(app) as client:
        egress = client.get(f"/api/trust/runs/{ref}/egress")
        assert egress.status_code == 200 and "SYNTHETIC_PRIVATE" not in egress.text
        assert egress.headers["cache-control"] == "no-store"
        assert client.get("/api/trust/runs/invalid/egress").status_code == 404
        assert client.get("/api/trust/conversations/synthetic/egress").json()["status"] == "not_recorded"
        release = client.get("/api/help/whats-new").json()
        assert release["show_after_update"]
        assert not client.get("/api/help/whats-new", params={"last_seen_version": release["latest_version"]}).json()["entries"]
        assert client.get("/api/help/whats-new?last_seen_version=invalid").status_code == 400
        bundle = client.get(f"/api/help/runs/{ref}/problem-bundle")
        assert bundle.status_code == 200 and bundle.headers["content-type"] == "application/zip"
        assert "attachment" in bundle.headers["content-disposition"]
        assert "project_dir" not in json.dumps(egress.json())
