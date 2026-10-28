"""Direct synthetic rail checks: finished replies are not finished research."""

from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot.conversation_status import (
    conversation_status,
    research_job_state,
    session_rail_status,
)

EARLY = "2026-01-01T10:00:00Z"
LATE = "2026-01-01T11:00:00Z"


def status(*, research=None, choice=None, message="done", running=False):
    return conversation_status(
        message_status=message,
        message_running=running,
        message_at=LATE,
        research_state=research,
        research_at=EARLY if research else None,
        data_choice_status=choice,
    )


def test_finished_reply_including_scope_refusal_is_replied():
    assert status() == "replied"


@pytest.mark.parametrize("choice", ["pending", "selection_in_progress"])
def test_source_choice_stays_pending_after_reply(choice):
    assert status(choice=choice) == "needs_user"


@pytest.mark.parametrize("research", ["stopped", "needs_user", "done"])
def test_late_successful_chat_keeps_research_state(research):
    assert status(research=research) == research


@pytest.mark.parametrize("research", ["stopped", "needs_user", "done", None])
def test_active_reply_still_shows_running(research):
    assert status(research=research, running=True) == "running"


@pytest.mark.parametrize("message", ["failed", "cancelled", "interrupted"])
def test_failed_later_reply_is_stopped(message):
    assert status(message=message, research="done") == "stopped"


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ({"status": "failed", "run_type": "full"}, "stopped"),
        ({"status": "done", "human_review_pending": True}, "needs_user"),
        ({"status": "done", "plan_decision_pending": True}, "needs_user"),
        ({"status": "done", "plan_decision_pending": False}, "done"),
        ({"status": "done", "gate_status": "blocked"}, "stopped"),
        ({"status": "done", "analysis_outcome": {"status": "completed"}}, "done"),
        # Persisted done is task completion; it carries no analysis-stage claim.
        ({"status": "done", "kind": "extract"}, "done"),
    ],
)
def test_research_job_state_keeps_persisted_contract(job, expected):
    assert research_job_state(job) == expected


@pytest.mark.parametrize("research", [None, "stopped", "needs_user", "done", "running"])
def test_saved_session_reload_keeps_research_after_later_chat(research):
    record = SimpleNamespace(
        last_turn_status="done", active_message_job_id=None,
        conversation_updated_at=LATE, data_source_authorization=None,
        latest_research_job=(SimpleNamespace(state=research, at=EARLY, job_id="synthetic")
                             if research else None),
    )
    assert session_rail_status(record, job_is_running=lambda _: False) == (
        "stopped" if research == "running" else research or "replied"
    )
