"""ASGI workflow metadata check; run on the Mac, not the socket-blocked server."""

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver import capabilities


def test_research_workflow_route_is_read_only_and_respects_the_master_switch(monkeypatch):
    monkeypatch.setattr(capabilities, "capability_settings", lambda: {"science_skills_enabled": False})
    response = TestClient(app).get("/api/capabilities/research-workflows")
    assert response.status_code == 200
    rows = response.json()["items"]
    assert rows and all(row["read_only"] and not row["enabled"] for row in rows)
    assert all(len(row["databases"]) == 6 and row["steps"] and row["outputs"] for row in rows)
