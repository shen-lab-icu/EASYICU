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

    def send_message(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}


def test_status_and_create_routes_preserve_strict_boolean_opt_in(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    assert client.get("/api/copilot/pi/status").json()["runtime"]["status"] == "ready"
    created = client.post(
        "/api/copilot/pi/sessions",
        json={
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
        json={"external_llm_opt_in": "true"},
    )
    assert string_boolean.status_code == 422
    unknown = client.post(
        "/api/copilot/pi/sessions",
        json={"external_llm_opt_in": True, "api_key": "must-not-be-accepted"},
    )
    assert unknown.status_code == 422


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
        json={"external_llm_opt_in": True},
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["error"] == "external_llm_opt_in_required"
    assert detail["owner"] == "easyicu.webserver.pi_copilot"
