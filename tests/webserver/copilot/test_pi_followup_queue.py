"""Queued follow-ups are separate authorized turns, never inherited grants."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from easyicu.webserver import jobs
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.followup_queue import FollowupQueue
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway,
    allow_unrelated_message_test,
    study_state as study_state,
)


class HeldGateway(FakeGateway):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.prompts = []

    def request(self, method, params, **kwargs):
        if method == "session.prompt":
            self.prompts.append((params["message"], kwargs["tool_context"]))
            if len(self.prompts) == 1:
                self.entered.set()
                assert self.release.wait(4)
        if method == "session.abort":
            self.release.set()
        return super().request(method, params, **kwargs)


def wait_until(predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def held(tmp_path, study_state, monkeypatch):
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager(max_running=1))
    gateway = HeldGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    session_id = service.create_session(project_id="queue-project", external_llm_opt_in=True)["session"]["session_id"]
    allow_unrelated_message_test(service, session_id)
    first = service.send_message(session_id, project_id="queue-project", message="What is available?", allowed_actions=["run"])
    assert gateway.entered.wait(2)
    yield service, gateway, session_id, first
    gateway.release.set()
    wait_until(lambda: session_id not in service._busy_sessions)
    service.close()


def enqueue(service, session_id, message="What happened next?"):
    return service.send_message(session_id, project_id="queue-project", message=message, message_origin="typed", queue_if_busy=True)


def test_queue_dispatches_fifo_with_separate_tool_authority(held):
    service, gateway, session_id, first = held
    queued = enqueue(service, session_id)
    second = enqueue(service, session_id, "Which evidence is available?")
    assert queued["status"] == "queued"
    assert len(gateway.prompts) == 1
    gateway.release.set()
    wait_until(lambda: len(gateway.prompts) == 3 and session_id not in service._busy_sessions)
    assert [text for text, _ in gateway.prompts] == ["What is available?", "What happened next?", "Which evidence is available?"]
    assert gateway.prompts[0][1].allowed_actions == frozenset({"run"})
    assert gateway.prompts[1][1].allowed_actions == frozenset()
    assert gateway.prompts[1][1] is not gateway.prompts[0][1]
    items = service.followup_messages(session_id, project_id="queue-project")["queue"]["items"]
    assert {item["id"] for item in items} == {queued["queue_item_id"], second["queue_item_id"]}
    assert all(item["status"] == "submitted" and item["job_id"] != first["job_id"] for item in items)


def test_queue_rechecks_live_study_scope_at_delivery(held, study_state):
    service, gateway, session_id, _ = held
    enqueue(service, session_id)
    study_state["revision"] += 1
    gateway.release.set()
    wait_until(lambda: service.followup_messages(session_id, project_id="queue-project")["queue"]["items"][0]["status"] == "interrupted")
    item = service.followup_messages(session_id, project_id="queue-project")["queue"]["items"][0]
    assert item["status"] == "interrupted"
    assert item["error"] == "pi_session_authority_stale"
    assert len(gateway.prompts) == 1


def test_abort_cancels_pending_followups(held):
    service, gateway, session_id, first = held
    enqueue(service, session_id)
    service.abort_session(session_id, project_id="queue-project", message_job_id=first["job_id"])
    wait_until(lambda: session_id not in service._busy_sessions)
    assert len(gateway.prompts) == 1
    assert service.followup_messages(session_id, project_id="queue-project")["queue"]["items"][0]["status"] == "cancelled"


def test_queue_cancellation_and_cross_project_scope(held):
    service, gateway, session_id, _ = held
    item = enqueue(service, session_id)
    with pytest.raises(PiCopilotError):
        service.cancel_followup(session_id, project_id="other-project", item_id=item["queue_item_id"])
    service.cancel_followup(session_id, project_id="queue-project", item_id=item["queue_item_id"])
    gateway.release.set()
    wait_until(lambda: session_id not in service._busy_sessions)
    assert len(gateway.prompts) == 1


def test_queue_rejects_embedded_grants_and_bound_source_mutation(held):
    service, gateway, session_id, _ = held
    with pytest.raises(PiCopilotError) as error:
        service.send_message(session_id, project_id="queue-project", message="What happened?", message_origin="typed", allowed_actions=["run"], queue_if_busy=True)
    assert error.value.code == "pi_followup_plain_message_required"
    with pytest.raises(PiCopilotError) as error:
        enqueue(service, session_id, "Generate the analysis plan.")
    assert error.value.code == "pi_followup_action_requires_idle"


def test_persisted_queue_restart_is_interrupted_and_bounded(tmp_path: Path):
    path = tmp_path / "queue.json"
    queue = FollowupQueue(path)
    for index in range(queue.MAX_PENDING):
        queue.enqueue(session_id="s", project_id="p", message=f"question {index}", binding={"revision": 1})
    with pytest.raises(PiCopilotError) as error:
        queue.enqueue(session_id="s", project_id="p", message="overflow", binding={})
    assert error.value.code == "pi_followup_queue_full"
    restarted = FollowupQueue(path)
    assert restarted.take("s") is None
    assert all(row["status"] == "interrupted" and row["error"] == "server_restarted" for row in restarted.snapshot("s")["items"])
    assert all("binding" not in row for row in restarted.snapshot("s")["items"])


@pytest.mark.parametrize("raw", ["broken", '[{"id":"q","session_id":"s","project_id":"p","message":"text","binding":{},"status":[]}]'])
def test_corrupt_queue_does_not_become_executable(tmp_path: Path, raw):
    path = tmp_path / "queue.json"
    path.write_text(raw, encoding="utf-8")
    queue = FollowupQueue(path)
    assert queue.snapshot("s")["available"] is False
    with pytest.raises(PiCopilotError, match="could not be read"):
        queue.enqueue(session_id="s", project_id="p", message="hello", binding={})


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_failed_queue_write_never_leaves_an_executable_message(tmp_path, monkeypatch, operation):
    from easyicu.webserver.pi_copilot import followup_queue
    queue = FollowupQueue(tmp_path / "queue.json")
    queue.enqueue(session_id="s", project_id="p", message="first", binding={})
    def fail(*args):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(followup_queue.os, operation, fail)
    with pytest.raises(PiCopilotError) as error:
        queue.enqueue(session_id="s", project_id="p", message="rejected", binding={})
    assert error.value.code == "pi_followup_store_unavailable"
    assert queue.take("s") is None
    assert queue.snapshot("s")["available"] is False
    assert all(row["status"] == "interrupted" for row in queue.snapshot("s")["items"])


def test_queue_disk_failure_does_not_prevent_stopping_active_message(held, monkeypatch):
    service, gateway, session_id, first = held
    enqueue(service, session_id)
    def fail():
        raise OSError("synthetic queue disk failure")
    monkeypatch.setattr(service.followups, "_write_store", fail)
    result = service.abort_session(session_id, project_id="queue-project", message_job_id=first["job_id"])
    assert result["cancel_requested"] is True
    assert result["pi_aborted"] is True
    assert jobs.MANAGER.get(first["job_id"]).cancel_requested is True
    assert service.followups.snapshot(session_id)["available"] is False
    wait_until(lambda: session_id not in service._busy_sessions)
    assert len(gateway.prompts) == 1


def test_queue_cannot_defer_registered_source_confirmation(held):
    service, _, session_id, _ = held
    with pytest.raises(PiCopilotError) as error:
        enqueue(service, session_id, "Use the local prepared data")
    assert error.value.code == "pi_followup_action_requires_idle"


def test_abort_revokes_queued_delivery_during_preflight_without_global_lock(held, monkeypatch):
    service, gateway, session_id, _ = held
    enqueue(service, session_id)
    entered = threading.Event()
    release = threading.Event()
    original = service._ensure_open
    def held_open(record):
        entered.set()
        assert release.wait(3)
        return original(record)
    monkeypatch.setattr(service, "_ensure_open", held_open)
    gateway.release.set()
    assert entered.wait(3)
    # Delivery I/O must not hold the lock shared by every conversation.
    assert service._lock.acquire(timeout=0.2)
    service._lock.release()
    service.abort_session(session_id, project_id="queue-project")
    release.set()
    wait_until(lambda: session_id not in service._dispatching_followups)
    assert len(gateway.prompts) == 1
    assert service.followups.snapshot(session_id)["items"][0]["status"] == "cancelled"


def test_corrupt_store_is_not_overwritten_by_cancel(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text("broken", encoding="utf-8")
    queue = FollowupQueue(path)
    with pytest.raises(PiCopilotError):
        queue.cancel("s")
    assert path.read_text(encoding="utf-8") == "broken"


def test_malformed_queue_status_is_unavailable_instead_of_crashing(tmp_path):
    import json
    path = tmp_path / "queue.json"
    path.write_text(json.dumps([{"id": "q", "session_id": "s", "project_id": "p", "message": "hello", "binding": {}, "status": []}]), encoding="utf-8")
    assert FollowupQueue(path).snapshot("s")["available"] is False


def test_http_queue_roundtrip_and_project_authority(held, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from easyicu.webserver.routes import pi_copilot as routes

    service, gateway, session_id, _ = held
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    endpoint = f"/api/copilot/pi/sessions/{session_id}"
    body = {"project_id": "queue-project", "message": "Which evidence is available?",
            "message_origin": "typed", "queue_if_busy": True}
    response = client.post(endpoint + "/message", json=body)
    assert response.status_code == 200
    item_id = response.json()["queue_item_id"]
    assert client.get(endpoint + "/followups", params={"project_id": "other"}).status_code == 409
    queue = client.get(endpoint + "/followups", params={"project_id": "queue-project"}).json()["queue"]
    assert queue["items"][0]["id"] == item_id
    assert "binding" not in queue["items"][0]
    response = client.post(endpoint + f"/followups/{item_id}/cancel", json={"project_id": "queue-project"})
    assert response.status_code == 200
    assert response.json()["queue"]["items"][0]["status"] == "cancelled"
    assert client.post(endpoint + "/message", json={**body, "queue_if_busy": "true"}).status_code == 422
    gateway.release.set()
    wait_until(lambda: session_id not in service._busy_sessions)
    assert len(gateway.prompts) == 1


def test_failed_turn_does_not_deliver_followups(held):
    service, gateway, session_id, _ = held
    enqueue(service, session_id)
    original = FakeGateway.request

    def fail_prompt(method, params, **kwargs):
        if method == "session.prompt":
            gateway.release.wait(2)
            raise RuntimeError("synthetic provider failure")
        return original(gateway, method, params, **kwargs)

    # The first prompt is already waiting; make its eventual response fail.
    from unittest.mock import patch
    with patch.object(FakeGateway, "request", lambda self, *args, **kwargs: fail_prompt(*args, **kwargs)):
        gateway.release.set()
        wait_until(lambda: service.followups.snapshot(session_id)["items"][0]["status"] == "interrupted")
    assert len(gateway.prompts) == 1
    assert service.followups.snapshot(session_id)["items"][0]["error"] == "previous_turn_failed"


def test_native_prompt_does_not_accept_same_turn_steering():
    from easyicu.webserver.pi_copilot import service as service_module

    source = (Path(service_module.__file__).parent / "node_app/src/main.mjs").read_text()
    request_guard = source.split("async function promptSession(requestId, params) {", 1)[1].splitlines()[1]
    assert "assertExactKeys" in request_guard
    assert "streaming_behavior" not in request_guard


def test_terminal_scheduler_cancels_after_a_failed_job(held):
    from types import SimpleNamespace

    service, _, session_id, _ = held
    enqueue(service, session_id)
    service._schedule_followups(session_id, SimpleNamespace(status="failed"))
    wait_until(lambda: service.followups.snapshot(session_id)["items"][0]["status"] == "interrupted")
    assert service.followups.snapshot(session_id)["items"][0]["error"] == "previous_turn_failed"
