"""Blocked tool receipts identify the host action without changing authority."""

from unittest.mock import Mock

import pytest

from easyicu.webserver import research_run_submission
from easyicu.webserver.pi_copilot import study_context_update, tools
from easyicu.webserver.pi_copilot.contracts import (
    HostTurnGrant,
    PiSessionRecord,
    ToolExecutionContext,
    TURN_CAPABILITIES,
)
from easyicu.webserver.pi_copilot.service import (
    ALLOWED_TURN_ACTIONS,
    PRIVILEGED_ONE_SHOT_TURN_ACTIONS,
)


def _context(actions=(), *, user_message=""):
    return ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-action-receipt"),
        allowed_actions=actions,
        user_message=user_message,
    )


def _assert_blocked(result, action, *, consumed=False, provider_gate=False):
    assert result["status"] == "blocked"
    assert result["code"] == (
        "pi_action_grant_consumed" if consumed else "pi_action_authorization_required"
    )
    assert result["summary"] == (
        "This provider run requires its explicit one-use action grant."
        if provider_gate else
        f"The one-use {action} grant for this message was already consumed."
        if consumed else
        f"This action requires a one-use {action} grant for the current message."
    )
    assert result["owner"] == (
        "easyicu.webserver.provider_gate" if provider_gate
        else "easyicu.webserver.pi_copilot"
    )
    assert result["details"]["required_action"] == action
    assert result["details"]["grantable_by_turn"] is (
        action not in PRIVILEGED_ONE_SHOT_TURN_ACTIONS
    )


@pytest.mark.parametrize("action", sorted(ALLOWED_TURN_ACTIONS))
def test_missing_tool_grant_names_the_action_and_preserves_the_summary(action):
    _assert_blocked(tools._consume_action(_context(), action), action)


@pytest.mark.parametrize("action", sorted(ALLOWED_TURN_ACTIONS - TURN_CAPABILITIES))
def test_consumed_tool_grant_still_names_the_action(action):
    context = _context((action,))
    assert tools._consume_action(context, action) is None
    _assert_blocked(tools._consume_action(context, action), action, consumed=True)


@pytest.mark.parametrize("action", sorted(PRIVILEGED_ONE_SHOT_TURN_ACTIONS))
def test_privileged_browser_grant_is_not_click_grantable_or_consumed(action):
    context = _context((action,), user_message="Please explain the current setup.")
    _assert_blocked(tools._consume_action(context, action), action)
    assert action in context.allowed_actions


@pytest.mark.parametrize("consumed", [False, True])
def test_study_update_receipts_name_configure(consumed):
    context = _context(("configure",) if consumed else ())
    if consumed:
        assert study_context_update._consume_action(context, "configure") is None
    _assert_blocked(
        study_context_update._consume_action(context, "configure"),
        "configure", consumed=consumed,
    )


@pytest.fixture
def provider_submission(monkeypatch):
    monkeypatch.setattr(tools, "_bound_context", lambda binding: {"id": "study-test"})
    monkeypatch.setattr(tools, "_workflow_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(tools, "_account_environment_for_research_provider", lambda c: (None, None))

    def submit(request, *, authorize, **kwargs):
        authorize()
        pytest.fail("A blocked request must not submit a provider run")

    monkeypatch.setattr(research_run_submission, "submit_research_run", submit)


@pytest.mark.parametrize("action", ["provider_run", "report_revision"])
def test_provider_gate_names_the_required_privileged_action(provider_submission, action):
    context = _context((action,), user_message="Please explain the current setup.")
    result = tools._run(
        context, {"run_type": "full"},
        report_source_run_id="run-synthetic" if action == "report_revision" else "",
    )
    _assert_blocked(result, action, provider_gate=True)


@pytest.mark.parametrize("action", ["provider_run", "report_revision"])
@pytest.mark.parametrize("outcome", ["missing", "consumed", "inference_denied"])
def test_provider_submission_authorization_receipts(provider_submission, monkeypatch, action, outcome):
    context = _context((action,))
    # Submission rechecks authority after validation. Exercise both that
    # defense and a grant no longer available when the owner consumes it.
    monkeypatch.setattr(
        tools, "_privileged_action_inferred",
        Mock(side_effect=[True, outcome != "inference_denied"]),
    )
    if outcome != "inference_denied":
        monkeypatch.setattr(context.grant, "consume_once", lambda requested: outcome)
    result = tools._run(
        context, {"run_type": "full"},
        report_source_run_id="run-synthetic" if action == "report_revision" else "",
    )
    _assert_blocked(result, action, consumed=outcome == "consumed")


def test_receipts_do_not_change_the_privileged_action_set():
    assert tools._PRIVILEGED_ONE_SHOT_ACTIONS == PRIVILEGED_ONE_SHOT_TURN_ACTIONS
    context = _context(("configure",))
    assert isinstance(context.grant, HostTurnGrant)
    assert tools._consume_action(context, "configure") is None
