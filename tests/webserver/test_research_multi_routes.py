"""ASGI contracts for the lead's Mac; AF_UNIX transport is blocked in the lane."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from easyicu.webserver.routes import pi_copilot, research_multi


@pytest.mark.parametrize("suffix,owner,extra", [
    ("", "propose", {"study_context_id": "study", "targets": [{"database": "aumc"}, {"database": "sic"}]}),
    ("/approve", "approve", {"batch_id": "a" * 64, "card_sha256": "b" * 64}),
    ("/cancel", "cancel", {"batch_id": "a" * 64, "database": "aumc"}),
    ("/follow-up", "dispatch_followup", {"run_id": "run", "suggestion_id": "c" * 64,
        "expected_source_sha256": "d" * 64, "targets": [{"database": "aumc"}]}),
])
def test_combined_card_routes_use_the_normal_copilot_mount(monkeypatch, suffix, owner, extra):
    service, calls = object(), []
    monkeypatch.setattr(research_multi, "get_pi_copilot_service", lambda: service)
    def dispatch(current, **kwargs):
        calls.append((current, kwargs))
        return {"multi_database_batch_id": "a" * 64}
    monkeypatch.setattr(research_multi.actions, owner, dispatch)
    app = FastAPI()
    app.include_router(pi_copilot.router)
    with TestClient(app) as client:
        response = client.post("/api/copilot/pi/multi-database" + suffix,
                               json={"project_id": "p", "session_id": "s", **extra})
        assert response.status_code == 200
        assert response.json() == {"multi_database_batch_id": "a" * 64}
        assert len(calls) == 1 and calls[0][0] is service
        bound = calls[0][1]
        if owner == "propose":
            bound = bound["request"].model_dump(mode="json")
        assert bound["project_id"] == "p" and bound["session_id"] == "s"
        if owner == "approve":
            assert bound["card_sha256"] == "b" * 64


def test_browser_cannot_supply_a_plan_or_clearance_and_status_has_stable_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(research_multi, "get_pi_copilot_service", lambda: calls.append("service"))
    app = FastAPI()
    app.include_router(pi_copilot.router)
    with TestClient(app) as client:
        response = client.post("/api/copilot/pi/multi-database/approve", json={
            "project_id": "p", "session_id": "s", "batch_id": "a" * 64, "card_sha256": "b" * 64,
            "plan": {}, "clearance": "verified_public_or_synthetic",
        })
        assert response.status_code == 422
        response = client.get("/api/copilot/pi/multi-database/invalid", params={"project_id": "p", "session_id": "s"})
        assert response.status_code == 422
        assert response.json() == {"detail": {"error": "multi_database_request_invalid"}}
        assert calls == []
