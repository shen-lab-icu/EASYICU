"""HTTP wiring; run on the lead's Mac because TestClient cannot run in the lane."""

import pytest
from threading import RLock
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore
from easyicu.webserver.routes import attention
from tests.webserver.attention_fixtures import archive, result_ready


@pytest.fixture
def client(tmp_path, monkeypatch):
    from types import SimpleNamespace

    replay = PiConversationReplayStore(tmp_path / "replay")
    record = PiSessionRecord(session_id="conversation-1", project_id="project-1")
    service = SimpleNamespace(store_path=tmp_path / "sessions.json", replay_store=replay,
                              _lock=RLock(),
                              _read_records=lambda: [record])
    monkeypatch.setattr(attention, "get_pi_copilot_service", lambda: service)
    archive(replay, result=result_ready())
    return TestClient(app, base_url="http://127.0.0.1:8765")


def test_get_and_post_ack_are_registered_and_private(client):
    response = client.get("/api/attention", params={"client_id": "browser-1", "after": 0})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert len(response.json()["items"]) == 1
    assert client.post("/api/attention/ack", json={"client_id": "browser-1", "sequences": [1]}).status_code == 200
    assert client.get("/api/attention", params={"client_id": "browser-1"}).json()["items"] == []
    assert len(client.get("/api/attention", params={"client_id": "desktop-1"}).json()["items"]) == 1


@pytest.mark.parametrize("params", [{}, {"client_id": "../other"}, {"client_id": "b", "limit": 101},
                                    {"client_id": "b", "after": -1}])
def test_http_query_validation(client, params):
    assert client.get("/api/attention", params=params).status_code == 422


def test_unknown_ack_and_cross_origin_ack_are_refused(client):
    body = {"client_id": "browser-1", "sequences": [999]}
    assert client.post("/api/attention/ack", json=body).status_code == 409
    assert client.post("/api/attention/ack", json=body,
                       headers={"origin": "https://outside.example"}).status_code == 403


def test_hosted_cannot_select_another_users_store(client, tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    response = client.get("/api/attention", params={"client_id": "browser-1", "project_id": "another-user"},
                          headers={"X-User-Id": "another-user"})
    assert response.status_code == 200
    assert {item["project_id"] for item in response.json()["items"]} == {"project-1"}
    # Even a mistakenly reused singleton cannot open another user's journal.
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "another-user"))
    assert client.get("/api/attention", params={"client_id": "browser-1"}).status_code == 403
