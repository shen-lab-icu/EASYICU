"""HTTP contract tests for the Pi Copilot route owner."""

from __future__ import annotations

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.routes import pi_copilot as route_module


class FakeService:
    def runtime_status(self) -> dict:
        return {"ok": True, "runtime": {"status": "ready"}}

    def create_session(self, **kwargs) -> dict:
        return {"ok": True, "received": kwargs}

    def configure_provider(self, **kwargs) -> dict:
        assert kwargs["api_key"] == "route-private-key"
        return {
            "ok": True,
            "runtime": {"status": "ready"},
            "configuration": {
                "credential_present": True,
                "connection_verified": True,
                "secrets_returned": False,
            },
        }

    def send_message(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}

    def list_sessions(self, **kwargs) -> dict:
        return {"ok": True, "sessions": [], "received": kwargs}

    def get_session(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}


def test_status_and_create_routes_preserve_strict_boolean_opt_in(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    assert client.get("/api/copilot/pi/status").json()["runtime"]["status"] == "ready"
    created = client.post(
        "/api/copilot/pi/sessions",
        json={
            "project_id": "guided-project-1",
            "title": "Review session",
            "language": "zh",
            "thinking_level": "high",
            "external_llm_opt_in": True,
        },
    )
    assert created.status_code == 200
    assert created.json()["received"]["external_llm_opt_in"] is True

    string_boolean = client.post(
        "/api/copilot/pi/sessions",
        json={"project_id": "guided-project-1", "external_llm_opt_in": "true"},
    )
    assert string_boolean.status_code == 422
    unknown = client.post(
        "/api/copilot/pi/sessions",
        json={
            "project_id": "guided-project-1",
            "external_llm_opt_in": True,
            "api_key": "must-not-be-accepted",
        },
    )
    assert unknown.status_code == 422
    assert created.json()["received"]["project_id"] == "guided-project-1"


def test_session_queries_are_scoped_to_one_research_project(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    assert client.get("/api/copilot/pi/sessions").status_code == 422
    listed = client.get(
        "/api/copilot/pi/sessions",
        params={"project_id": "guided-project-2", "limit": 7},
    )
    assert listed.status_code == 200
    assert listed.json()["received"] == {
        "project_id": "guided-project-2",
        "limit": 7,
    }

    assert client.get("/api/copilot/pi/sessions/pi-test").status_code == 422
    opened = client.get(
        "/api/copilot/pi/sessions/pi-test",
        params={"project_id": "guided-project-2"},
    )
    assert opened.status_code == 200
    assert opened.json()["received"]["project_id"] == "guided-project-2"


def test_provider_setup_route_is_typed_and_never_returns_secret(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    response = client.post(
        "/api/copilot/pi/provider-config",
        json={
            "provider": "easyicu-local",
            "api_key": "route-private-key",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gpt5.6 luna",
            "api_transport": "openai-completions",
            "enable_ai": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["runtime"]["status"] == "ready"
    assert "route-private-key" not in response.text
    assert client.post(
        "/api/copilot/pi/provider-config",
        json={
            "provider": "easyicu-local",
            "api_key": "route-private-key",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gpt5.6 luna",
            "api_transport": "openai-completions",
            "enable_ai": "true",
        },
    ).status_code == 422
    anthropic = client.post(
        "/api/copilot/pi/provider-config",
        json={
            "provider": "anthropic",
            "api_key": "route-private-key",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-sonnet-4-6",
            "api_transport": "anthropic-messages",
            "enable_ai": True,
        },
    )
    assert anthropic.status_code == 200
    assert client.post(
        "/api/copilot/pi/provider-config",
        json={
            "provider": "easyicu-local",
            "api_key": "route-private-key",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gpt5.6 luna",
            "api_transport": "unknown",
            "enable_ai": True,
        },
    ).status_code == 422


def test_message_route_rejects_unknown_actions_and_fields(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    accepted = client.post(
        "/api/copilot/pi/sessions/pi-test/message",
        json={
            "message": "Save setup and inspect aggregate validation",
            "allowed_actions": ["configure", "run"],
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["received"]["allowed_actions"] == ["configure", "run"]

    assert client.post(
        "/api/copilot/pi/sessions/pi-test/message",
        json={"message": "Inspect", "allowed_actions": ["bash"]},
    ).status_code == 422
    assert client.post(
        "/api/copilot/pi/sessions/pi-test/message",
        json={"message": "Inspect", "raw_rows": True},
    ).status_code == 422


def test_owner_error_keeps_stable_code_and_owner(monkeypatch) -> None:
    class BlockedService(FakeService):
        def create_session(self, **kwargs) -> dict:
            raise PiCopilotError(
                "external_llm_opt_in_required",
                "Explicit opt-in is required.",
                status_code=403,
            )

    monkeypatch.setattr(
        route_module,
        "get_pi_copilot_service",
        lambda: BlockedService(),
    )
    response = TestClient(app).post(
        "/api/copilot/pi/sessions",
        json={"project_id": "guided-project-1", "external_llm_opt_in": True},
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["error"] == "external_llm_opt_in_required"
    assert detail["owner"] == "easyicu.webserver.pi_copilot"
