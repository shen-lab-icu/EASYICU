from __future__ import annotations

from fastapi.testclient import TestClient

from easyicu.webserver import settings as settings_store
from easyicu.webserver.app import app


def test_study_intent_route_rejects_string_boolean_opt_in() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/copilot/study-intent",
        json={
            "question": "Does lactate predict mortality?",
            "llm_provider": "openai",
            "external_llm_opt_in": "false",
        },
    )

    assert response.status_code == 422


def test_study_intent_route_uses_server_ai_setting_not_request_body(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": False})

    response = client.post(
        "/api/copilot/study-intent",
        json={
            "question": "Does lactate predict mortality?",
            "llm_provider": "openai",
            "external_llm_opt_in": True,
            "ai_enabled": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "deterministic"
    assert payload["provider_block"]["error"] == "external_llm_opt_in_required"
