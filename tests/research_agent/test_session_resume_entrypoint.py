"""Session resume entrypoint: explicit key in, persisted log out (or fail closed)."""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceStore,
    SessionEventLog,
)
from easyicu.research_agent.pipeline import (
    _resolve_session_resume_inputs,
    _validate_resume_session_key,
)


def _stored_log(tmp_path):
    store = EvidenceStore(tmp_path / "entry")
    log = SessionEventLog(store)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    log.log_failure_reason("proj", "task", "sess", reason="boom", step_id="s1")
    log.suspend_session("proj", "task", "sess")
    log.save()
    return store


def test_key_shape_validation():
    assert _validate_resume_session_key(None) is None
    assert _validate_resume_session_key(("a", "b", "c")) == ("a", "b", "c")
    assert _validate_resume_session_key(["a", "b", "c"]) == ("a", "b", "c")
    with pytest.raises(ValueError, match="triple"):
        _validate_resume_session_key(("a", "b"))
    with pytest.raises(ValueError, match="triple"):
        _validate_resume_session_key(("a", "", "c"))
    with pytest.raises(ValueError, match="triple"):
        _validate_resume_session_key("abc")


def test_no_key_resolves_to_no_resume(tmp_path):
    store = EvidenceStore(tmp_path / "empty")
    assert (
        _resolve_session_resume_inputs(resume_session_key=None, evidence=store)
        is None
    )


def test_unknown_session_fails_closed_before_any_step(tmp_path):
    store = EvidenceStore(tmp_path / "empty")
    with pytest.raises(ValueError, match="no recorded session"):
        _resolve_session_resume_inputs(
            resume_session_key=("p", "t", "nope"), evidence=store
        )


def test_known_session_resolves_from_persisted_log(tmp_path):
    store = _stored_log(tmp_path)
    fresh_view = EvidenceStore(tmp_path / "entry")
    resolved = _resolve_session_resume_inputs(
        resume_session_key=("proj", "task", "sess"), evidence=fresh_view
    )
    assert resolved is not None
    log, key = resolved
    assert key == ("proj", "task", "sess")
    assert log.get_session(*key).status == "suspended"


def test_missing_evidence_store_fails_closed():
    with pytest.raises(ValueError, match="evidence store"):
        _resolve_session_resume_inputs(
            resume_session_key=("p", "t", "s"), evidence=None
        )
