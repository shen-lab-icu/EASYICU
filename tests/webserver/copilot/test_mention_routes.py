"""ASGI transport contracts for the lead; never run in the server sandbox."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.routes import mentions as routes


def test_picker_resolver_and_draft_are_registered(monkeypatch):
    calls = []
    def call(*args, **kwargs):
        calls.append((args, kwargs))
        return {"items": []}
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: SimpleNamespace(
        search_mentions=call, resolve_mention=call, set_draft_mentions=call,
    ))
    client = TestClient(app)
    prefix = "/api/copilot/pi/sessions/synthetic"
    assert client.get(prefix + "/mentions", params={"project_id": "project", "kind": "concept", "limit": 2}).status_code == 200
    assert client.post(prefix + "/mentions/resolve", json={"project_id": "project", "kind": "concept", "id": "lact"}).status_code == 200
    assert client.put(prefix + "/draft-mentions", json={"project_id": "project", "mentions": []}).status_code == 200
    assert len(calls) == 3 and all(args == ("synthetic",) for args, _ in calls)


def test_invalid_kind_and_unbounded_search_never_reach_the_owner(monkeypatch):
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: None)
    client = TestClient(app)
    for params in ({"kind": "unknown"}, {"kind": "concept", "limit": 41}, {"kind": "concept", "query": "x" * 121}):
        assert client.get("/api/copilot/pi/sessions/synthetic/mentions", params={"project_id": "project", **params}).status_code == 422


def test_stale_mentions_return_only_the_stable_code(monkeypatch):
    def refused(*args, **kwargs):
        raise PiCopilotError("mention_stale", "placeholder", status_code=409)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: SimpleNamespace(set_draft_mentions=refused))
    response = TestClient(app).put("/api/copilot/pi/sessions/synthetic/draft-mentions", json={"project_id": "project", "mentions": []})
    assert response.status_code == 409
    assert response.json() == {"detail": {"error": "mention_stale"}}
