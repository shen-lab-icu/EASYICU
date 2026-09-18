"""Track 2 long-task memory: session event log over EvidenceStore."""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceStore,
    SessionEventKind,
    SessionEventLog,
    SessionStatus,
)


def _make_log(tmp_path):
    store = EvidenceStore(tmp_path / "run")
    return store, SessionEventLog(store)


def test_typed_events_cover_attempt_artifact_failure_diagnosis(tmp_path):
    store, log = _make_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first try with X")
    log.log_completed_artifact(
        "proj", "task", "sess", evidence_id="table_one", produced_by_step="01_desc"
    )
    log.log_failure_reason("proj", "task", "sess", reason="boom", step_id="02_model")
    log.log_next_diagnosis(
        "proj", "task", "sess", diagnosis="rerun 02_model", step_id="02_model"
    )
    kinds = [event.kind for event in log.events()]
    assert kinds == [
        SessionEventKind.ATTEMPT_RATIONALE.value,
        SessionEventKind.COMPLETED_ARTIFACT.value,
        SessionEventKind.FAILURE_REASON.value,
        SessionEventKind.NEXT_DIAGNOSIS.value,
    ]
    assert store is not None


def test_three_level_hierarchy_projects_tasks_sessions(tmp_path):
    _, log = _make_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="why")
    payload = log.to_dict()
    assert [p["project_id"] for p in payload["projects"]] == ["proj"]
    assert [t["task_id"] for t in payload["tasks"]] == ["task"]
    assert [s["session_id"] for s in payload["sessions"]] == ["sess"]


def test_suspend_and_resume_cycle(tmp_path):
    _, log = _make_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="initial")
    log.log_next_diagnosis("proj", "task", "sess", diagnosis="continue later")
    log.suspend_session("proj", "task", "sess")
    assert log.get_session("proj", "task", "sess").status == SessionStatus.SUSPENDED.value
    log.resume_session("proj", "task", "sess", attempt_rationale="resume now")
    node = log.get_session("proj", "task", "sess")
    assert node.status == SessionStatus.RUNNING.value
    assert node.attempt_count == 2


def test_fail_session_retains_valid_artifacts(tmp_path):
    _, log = _make_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="try")
    log.log_completed_artifact("proj", "task", "sess", evidence_id="good_table")
    log.fail_session(
        "proj", "task", "sess", reason="solver failed", next_diagnosis="retry step"
    )
    node = log.get_session("proj", "task", "sess")
    assert node.status == SessionStatus.FAILED.value
    assert node.valid_evidence_ids == ["good_table"]
    assert node.failure_reason == "solver failed"
    assert node.next_diagnosis == "retry step"


def test_resumable_requires_next_diagnosis(tmp_path):
    _, log = _make_log(tmp_path)
    log.start_session("proj", "task", "s1", attempt_rationale="a")
    log.log_failure_reason("proj", "task", "s1", reason="bad")
    # No next diagnosis -> not resumable.
    assert log.resumable_sessions() == []
    log.log_next_diagnosis("proj", "task", "s1", diagnosis="do this")
    assert [s.session_id for s in log.resumable_sessions()] == ["s1"]


def test_save_and_load_latest_roundtrip(tmp_path):
    store, log = _make_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="why")
    log.log_completed_artifact("proj", "task", "sess", evidence_id="ev1")
    log.fail_session("proj", "task", "sess", reason="err", next_diagnosis="next")
    record = log.save()
    assert record.evidence_id.startswith("session_event_log")
    restored = SessionEventLog.load_latest(store)
    node = restored.get_session("proj", "task", "sess")
    assert node.valid_evidence_ids == ["ev1"]
    assert node.failure_reason == "err"
    assert node.next_diagnosis == "next"
    assert len(restored.events()) == 4


def test_load_latest_empty_when_no_save(tmp_path):
    store = EvidenceStore(tmp_path / "empty")
    restored = SessionEventLog.load_latest(store)
    assert restored.events() == []
    assert restored.resumable_sessions() == []


def test_invalid_ids_rejected(tmp_path):
    _, log = _make_log(tmp_path)
    with pytest.raises(ValueError):
        log.start_session("", "task", "sess", attempt_rationale="x")
    with pytest.raises(ValueError):
        log.start_session("proj", "task", "sess", attempt_rationale="")
