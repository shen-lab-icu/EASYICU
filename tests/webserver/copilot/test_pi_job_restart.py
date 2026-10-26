"""Lifecycle recovery preserves conversation meaning and archived evidence."""

from pathlib import Path

import pytest

from easyicu.webserver import jobs, settings
from easyicu.webserver.job_journal import JobJournal
from easyicu.webserver.pi_copilot.projections import project_job
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway


def test_restart_receipt_is_an_interruption_in_conversation(tmp_path, monkeypatch):
    journal_path = tmp_path / "jobs.sqlite3"
    JobJournal(journal_path).create(jobs.Job("lost-message", "pi-copilot-message").lifecycle_receipt())
    manager = jobs.JobManager()
    manager.enable_persistence(journal_path)
    monkeypatch.setattr(jobs, "MANAGER", manager)
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway())
    session_id = service.create_session(project_id="restart-test", external_llm_opt_in=True)["session"]["session_id"]
    record = service._get_record(session_id)
    record.active_message_job_id = "lost-message"
    record.last_turn_status = "running"
    service._save_record(record)
    public = service.get_session(session_id, project_id="restart-test")["session"]
    assert public["active_message_job_id"] is None
    assert public["last_turn_status"] == "interrupted"
    projected = project_job(manager.get("lost-message").snapshot())
    assert projected["interrupted"] is True
    assert projected["recovered"] is True
    assert projected["result_omitted"] is True
    assert projected["error_code"] == "job_interrupted_by_server_restart"
    assert projected["next_actions"] == []


@pytest.mark.parametrize("previous_status", ["running", "done"])
def test_lifecycle_recovery_cannot_erase_archived_evidence(tmp_path: Path, previous_status):
    store = PiConversationReplayStore(tmp_path / "replay")
    scope = {"session_id": "session-1", "project_id": "project-1"}
    store.record_host_action(**scope, action_id="action-1", action_code="generate_plan", child_job_id="job-1")
    original = {
        "job_id": "job-1", "kind": "agent-run", "status": previous_status,
        "run_id": "run-1", "progress": [{"step": "plan", "seq": 1}],
        "artifact_refs": [{"kind": "research_artifact", "run_id": "run-1",
                           "artifact": "quality_gate.json", "sha256": "b" * 64}],
    }
    store.archive_child_job(**scope, job=original)
    restored = project_job(jobs.Job.from_lifecycle_receipt({
        "id": "job-1", "kind": "agent-run", "status": "running",
        "created": 1.0, "finished": None, "cancel_requested": False,
        "next_event_seq": 256, "interrupted": False,
    }).snapshot())
    archived = store.archive_child_job(**scope, job=restored)
    assert archived["run_id"] == original["run_id"]
    assert archived["artifact_refs"] == original["artifact_refs"]
    assert archived["progress"] == original["progress"]
    if previous_status == "done":
        assert archived == original
    else:
        assert archived["interrupted"] is True
        assert archived["status"] == "failed"
    assert store.snapshot(**scope)["child_jobs"] == [archived]


@pytest.mark.parametrize("previous_state", ["needs_user", "stopped", "running"])
def test_omitted_job_result_cannot_satisfy_research_review(tmp_path, monkeypatch, previous_state):
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway())
    session_id = service.create_session(project_id="review-test", external_llm_opt_in=True)["session"]["session_id"]
    service._note_research_job(session_id, {
        "job_id": "job-review", "created_at_epoch": 1.0, "finished_at_epoch": 2.0,
        "status": "running" if previous_state == "running" else "done",
        "human_review_pending": previous_state == "needs_user",
        "gate_status": "blocked" if previous_state == "stopped" else "",
    })
    assert service._get_record(session_id).latest_research_job.state == previous_state
    restored = project_job(jobs.Job.from_lifecycle_receipt({
        "id": "job-review", "kind": "agent-run", "status": "done",
        "created": 1.0, "finished": 2.0, "cancel_requested": False,
        "next_event_seq": 8, "interrupted": False,
    }).snapshot())
    service._note_research_job(session_id, restored)
    expected = "stopped" if previous_state == "running" else previous_state
    assert service._get_record(session_id).latest_research_job.state == expected
