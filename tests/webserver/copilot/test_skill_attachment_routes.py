"""ASGI transport checks for the lead's Mac; direct owners run on the server."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.routes import skill_attachments as routes
from easyicu.webserver.pi_copilot.contracts import PiCopilotError


def test_attachment_catalog_and_draft_replacement(monkeypatch):
    calls = []
    service = SimpleNamespace(
        attachable_skills=lambda *args, **kwargs: {"items": [], "max_attached_skills": 8},
        set_draft_skills=lambda *args, **kwargs: calls.append((args, kwargs)) or {"attached_skills": []},
    )
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    client = TestClient(app)
    response = client.get("/api/copilot/pi/sessions/synthetic/skills", params={"project_id": "project"})
    assert response.status_code == 200 and response.json()["items"] == []
    response = client.put("/api/copilot/pi/sessions/synthetic/draft-skills", json={"project_id": "project", "attached_skills": []})
    assert response.status_code == 200
    assert calls == [(("synthetic",), {"project_id": "project", "attached_skills": []})]


def test_project_defaults_read_and_write(monkeypatch):
    calls = []
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: SimpleNamespace(
        project_default_skills=lambda **kwargs: calls.append(kwargs) or {"attached_skills": []},
    ))
    client = TestClient(app)
    assert client.get("/api/copilot/pi/projects/project/default-skills").status_code == 200
    assert client.put("/api/copilot/pi/projects/project/default-skills", json={"attached_skills": []}).status_code == 200
    assert calls == [{"project_id": "project"}, {"project_id": "project", "attached_skills": []}]


def test_refusals_are_stable_codes(monkeypatch):
    def refused(*args, **kwargs):
        raise PiCopilotError("skill_attachment_stale", "placeholder", status_code=409)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: SimpleNamespace(set_draft_skills=refused))
    response = TestClient(app).put("/api/copilot/pi/sessions/synthetic/draft-skills", json={"project_id": "project", "attached_skills": []})
    assert response.status_code == 409
    assert response.json() == {"detail": {"error": "skill_attachment_stale"}}
