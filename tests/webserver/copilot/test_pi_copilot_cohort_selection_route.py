"""HTTP boundary for structured primary-cohort selection events."""

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.routes import pi_copilot as route_module


def test_cohort_selection_route_accepts_only_typed_host_coordinates(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    class Service:
        def confirm_cohort_eligibility(self, session_id: str, **kwargs) -> dict:
            calls.append({"session_id": session_id, **kwargs})
            return {"ok": True, "code": "cohort_eligibility_confirmed"}

    monkeypatch.setattr(route_module, "get_pi_copilot_service", Service)
    client = TestClient(app)
    payload = {
        "project_id": "project-cohort-route",
        "option_id": "confirm_current_cohort",
        "expected_revision": 7,
        "primary_cohort_contract_sha256": "a" * 64,
        "selection_event_id": "b" * 64,
    }

    accepted = client.post(
        "/api/copilot/pi/sessions/pi-cohort-route/cohort-eligibility-selection",
        json=payload,
    )
    assert accepted.status_code == 200
    assert calls == [{"session_id": "pi-cohort-route", **payload}]

    malformed = client.post(
        "/api/copilot/pi/sessions/pi-cohort-route/cohort-eligibility-selection",
        json={**payload, "selection_event_id": "not-a-digest"},
    )
    assert malformed.status_code == 422
    assert len(calls) == 1

    extra = client.post(
        "/api/copilot/pi/sessions/pi-cohort-route/cohort-eligibility-selection",
        json={**payload, "conversation_text": "I agree"},
    )
    assert extra.status_code == 422
    assert len(calls) == 1
