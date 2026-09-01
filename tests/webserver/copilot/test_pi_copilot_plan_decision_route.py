"""HTTP boundary for host-owned scientific plan choices."""

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.routes import pi_copilot as route_module


def test_plan_decision_route_accepts_only_typed_coordinates(monkeypatch) -> None:
    calls: list[dict] = []

    class Service:
        def confirm_plan_decision(self, session_id: str, **kwargs) -> dict:
            calls.append({"session_id": session_id, **kwargs})
            return {"ok": True, "code": "plan_decision_confirmed"}

    monkeypatch.setattr(route_module, "get_pi_copilot_service", Service)
    client = TestClient(app)
    payload = {
        "project_id": "project-plan-choice",
        "decision_code": "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
        "option_id": "landmark_24h",
        "expected_revision": 7,
        "run_id": "run_20260831T065226_d3c42d",
    }

    accepted = client.post(
        "/api/copilot/pi/sessions/pi-plan-choice/plan-decision-selection",
        json=payload,
    )
    assert accepted.status_code == 200
    assert calls == [{"session_id": "pi-plan-choice", **payload}]

    extra = client.post(
        "/api/copilot/pi/sessions/pi-plan-choice/plan-decision-selection",
        json={**payload, "conversation_text": "I agree"},
    )
    assert extra.status_code == 422
    assert len(calls) == 1
