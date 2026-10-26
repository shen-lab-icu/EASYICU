"""ASGI acceptance; run on the lead's Mac (TestClient stalls in the lane)."""

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.routes import pi_copilot as routes


def test_quick_click_routes_reject_chat_approval_and_forward_exact_coordinates(monkeypatch):
    calls = []

    class Service:
        def approve_quick_answer(self, session_id, **kwargs):
            calls.append(("approve", session_id, kwargs))
            return {"ok": True}

        def expand_quick_answer(self, session_id, **kwargs):
            calls.append(("expand", session_id, kwargs))
            return {"ok": True}

    monkeypatch.setattr(routes, "get_pi_copilot_service", Service)
    with TestClient(app) as client:
        prefix = "/api/copilot/pi/sessions/pi-synthetic/quick-answers/qa_synthetic"
        body = {"project_id": "project-synthetic", "card_sha256": "a" * 64, "accept_substitutions": True}
        assert client.post(prefix + "/approve", json={**body, "conversation_text": "yes"}).status_code == 422
        assert client.post(prefix + "/approve", json={**body, "accept_substitutions": "yes"}).status_code == 422
        assert not calls
        assert client.post(prefix + "/approve", json=body).status_code == 200
        assert calls[-1] == ("approve", "pi-synthetic", {"quick_answer_id": "qa_synthetic", **body})
        assert client.post(prefix + "/expand", json={"project_id": "project-synthetic", "reading": {}}).status_code == 422
        assert client.post(prefix + "/expand", json={"project_id": "project-synthetic"}).status_code == 200
        assert len(calls) == 2
