"""ASGI contracts: run on the lead machine; AF_UNIX is blocked on the server."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.personal_preferences import PersonalPreferencesStore
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.routes import personal_preferences as routes


def test_preferences_http_codes_and_partial_updates(tmp_path, monkeypatch):
    service = SimpleNamespace(preferences=PersonalPreferencesStore(tmp_path / "preferences.json"))
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    route = "/api/copilot/pi/preferences"
    assert client.get(route).json()["preferences"]["instructions"] == ""
    assert client.patch(route, json={"instructions": "Short prose"}).status_code == 200
    response = client.patch(route, json={"interface_language": "zh"})
    assert response.status_code == 200
    assert response.json()["preferences"]["instructions"] == "Short prose"
    response = client.patch(route, json={"figure_export_format": "exe"})
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "preferences_figure_export_format_invalid"
    assert client.patch(route, json={"execution": True}).json()["detail"]["error"] == "preferences_field_unknown"


def test_usage_routes_enforce_scope_and_identifiers(monkeypatch):
    def refused(*args, **kwargs):
        raise PiCopilotError("pi_research_run_not_found", "Unknown", status_code=404)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: SimpleNamespace(_research_run_row=refused, _scoped_record=refused))
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    assert client.get("/api/copilot/pi/projects/project-a/runs/run-a/usage").status_code == 404
    assert client.get("/api/copilot/pi/sessions/session-a/usage", params={"project_id": "project-a"}).status_code == 404
    assert client.get("/api/copilot/pi/sessions/session-a/usage", params={"project_id": "../invalid"}).status_code == 422
