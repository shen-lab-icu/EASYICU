"""Replaced failures remain diagnostic history, never active branch authority."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore


def test_failed_attempt_history_excludes_old_text_resources_and_grants(tmp_path: Path) -> None:
    store = PiConversationReplayStore(tmp_path / "replay")
    store.start_turn(session_id="session", project_id="project", job_id="old-failure",
                     allowed_actions=["workspace_write"], user_message="private old question")
    store.append_event(session_id="session", project_id="project", job_id="old-failure", event={
        "type": "tool_end", "at": "2026-10-29T09:00:20Z", "tool_name": "easyicu_write_project_file",
        "tool_call_id": "old-call", "status": "error", "code": "pi_gateway_timeout",
        "resources": [{"file": "old.html", "sha256": "a" * 64}], "summary": "old resource summary",
    })
    store.finish_turn(session_id="session", project_id="project", job_id="old-failure", status="failed",
                      assistant_reply="private previous answer")
    store.supersede_from_turn_index(session_id="session", project_id="project", turn_index=0)
    store.start_turn(session_id="session", project_id="project", job_id="current", allowed_actions=[])
    store.finish_turn(session_id="session", project_id="project", job_id="current", status="done",
                      assistant_entry_id="completed-entry")
    active = store.snapshot(session_id="session", project_id="project")
    assert [r["job_id"] for r in active["turns"]] == ["current"]
    assert active["turns"][0]["assistant_entry_id"] == "completed-entry"
    old = store.prior_failed_attempts(session_id="session", project_id="project")
    assert len(old) == 1 and old[0]["job_id"] == "old-failure"
    assert set(old[0]) == {"job_id", "status", "started_at", "ended_at", "events"}
    assert old[0]["events"][0]["tool_call_id"] == "old-call"
    serialized = json.dumps(old)
    for private in ("private", "old.html", "resources", "allowed_actions", "old resource summary"):
        assert private not in serialized
    with pytest.raises(PiCopilotError, match="another project"):
        store.prior_failed_attempts(session_id="session", project_id="other-project")


def test_retry_route_accepts_only_failed_entry_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from easyicu.webserver.app import app
    from easyicu.webserver.routes import pi_copilot as routes

    calls = []

    class Service:
        def retry_unstarted_reply(self, session_id, **kwargs):
            calls.append((session_id, kwargs))
            return {"ok": True, "job_id": "retry-job"}

    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: Service())
    with TestClient(app) as client:
        path = "/api/copilot/pi/sessions/session/retry-unstarted"
        payload = {"project_id": "project", "source_error_entry_id": "failed-entry"}
        response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        assert calls == [("session", payload)]
        for extra in ({"message": "replace request"}, {"allowed_actions": ["full_run"]}, {"budget": 999}):
            rejected = client.post(path, json={**payload, **extra})
            assert rejected.status_code == 422
        assert len(calls) == 1


@pytest.mark.parametrize("recovery_kind", [None, "retry_unstarted"])
def test_recovery_without_new_user_does_not_shift_regeneration_index(tmp_path: Path, recovery_kind) -> None:
    store = PiConversationReplayStore(tmp_path / "replay")
    store.start_turn(session_id="session", project_id="project", job_id="user-0",
                     allowed_actions=[], user_message="first question")
    store.finish_turn(session_id="session", project_id="project", job_id="user-0", status="failed")
    store.start_turn(session_id="session", project_id="project", job_id="recovery-0", allowed_actions=[],
                     source_job_id="user-0", source_error_entry_id="error-0", recovery_kind=recovery_kind)
    store.finish_turn(session_id="session", project_id="project", job_id="recovery-0", status="done")
    skill = {"source": "builtin", "id": "web-prototype", "version": "v1"}
    store.start_turn(session_id="session", project_id="project", job_id="user-1", allowed_actions=[],
                     user_message="second question", attached_skills=[skill])
    store.finish_turn(session_id="session", project_id="project", job_id="user-1", status="done")
    assert store.attachments_for_turn(session_id="session", project_id="project", turn_index=1) == [skill]
    words = store.question_wording_context(session_id="session", project_id="project")
    assert [row["text"] for row in words["user_turns"]] == ["first question", "second question"]
    store.supersede_from_turn_index(session_id="session", project_id="project", turn_index=1)
    active = store.snapshot(session_id="session", project_id="project")
    assert [row["job_id"] for row in active["turns"]] == ["user-0", "recovery-0"]
