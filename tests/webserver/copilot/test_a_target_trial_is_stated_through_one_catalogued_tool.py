"""A causal study's target trial is stated through one catalogued tool.

The conversation's tool hands the statement to the setup owner
(``target_trial_setup``).  Whatever the owner refuses before a compile starts
is refused before the one-use Configure grant is spent; a statement the host
starts compiling spends the grant and the turn's authority, so the model writes
its reply from the job it was handed.  The tool also waits, as every tool that
reads data does, for the conversation's confirmed data source.  The setup owner
is replaced here: its own tests start real compile jobs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.webserver import target_trial_setup
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiCopilotError,
    PiSessionDataSourceAuthorization,
    PiSessionRecord,
    ToolExecutionContext,
)

TOOL = target_trial_setup.TARGET_TRIAL_STATE_TOOL
STUDY = {"id": "study-trial", "revision": 4}
STATEMENT = {"spec": {"treatment": {}}, "population_spec": {"criteria": []}}


def _context(*, actions=("configure",), source_status="confirmed") -> ToolExecutionContext:
    return ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-target-trial",
            binding=AuthorityBinding(study_context_id="study-trial", study_revision=4),
            data_source_authorization=PiSessionDataSourceAuthorization(
                status=source_status,
                reason=(
                    "project_source_confirmation_required"
                    if source_status == "pending"
                    else None
                ),
            ),
        ),
        allowed_actions=set(actions),
    )


class _Owner:
    """The setup owner's three calls, recorded; a refusal is set per test."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.check_refusal: target_trial_setup.TargetTrialSetupError | None = None
        self.start_refusal: target_trial_setup.TargetTrialSetupError | None = None

    def check(self, study: dict, params: dict) -> Any:
        self.calls.append("check")
        assert (study, params) == (STUDY, STATEMENT)
        if self.check_refusal is not None:
            raise self.check_refusal
        return "statement"

    def submit(self, study: dict, statement: Any) -> Any:
        self.calls.append("submit")
        assert statement == "statement"
        if self.start_refusal is not None:
            raise self.start_refusal
        return SimpleNamespace(id="job-trial", kind="target_trial_compile", status="running")


@pytest.fixture
def owner(monkeypatch: pytest.MonkeyPatch) -> _Owner:
    fake = _Owner()
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(STUDY))
    monkeypatch.setattr(
        tool_module.target_trial_setup, "check_target_trial_statement", fake.check
    )
    monkeypatch.setattr(
        tool_module.target_trial_setup, "submit_target_trial_compile", fake.submit
    )
    return fake


def _authority_spent(context: ToolExecutionContext) -> str | None:
    try:
        context.assert_authority_fresh()
    except PiCopilotError as exc:
        return str(exc.details.get("reason"))
    return None


def test_a_started_compile_spends_the_grant_and_the_turn(owner: _Owner) -> None:
    context = _context()

    result = tool_module.execute_tool(TOOL, STATEMENT, context)

    assert (result["status"], result["code"]) == (
        "ok",
        target_trial_setup.TARGET_TRIAL_COMPILE_SUBMITTED,
    )
    assert result["details"]["job_id"] == "job-trial"
    assert result["details"]["study_context_id"] == "study-trial"
    assert owner.calls == ["check", "submit"]
    assert context.grant.consume_once("configure") == "consumed"
    assert _authority_spent(context) == target_trial_setup.TARGET_TRIAL_COMPILE_SUBMITTED


def test_a_statement_the_host_refuses_keeps_the_grant(owner: _Owner) -> None:
    owner.check_refusal = target_trial_setup.TargetTrialSetupError(
        "target_trial_statement_invalid",
        "The host refuses these fields.",
        status_code=422,
        details={"fields": ["spec.grace_period.hours"]},
    )
    context = _context()

    result = tool_module.execute_tool(TOOL, STATEMENT, context)

    assert (result["status"], result["code"]) == (
        "blocked",
        "target_trial_statement_invalid",
    )
    assert result["details"] == {"fields": ["spec.grace_period.hours"]}
    assert owner.calls == ["check"]
    assert context.grant.consume_once("configure") == "granted"
    assert _authority_spent(context) is None


def test_without_the_configure_grant_no_compile_starts(owner: _Owner) -> None:
    context = _context(actions=())

    result = tool_module.execute_tool(TOOL, STATEMENT, context)

    assert (result["status"], result["code"]) == (
        "blocked",
        "pi_action_authorization_required",
    )
    assert owner.calls == ["check"]
    assert _authority_spent(context) is None


def test_a_compile_refused_as_it_starts_says_why(owner: _Owner) -> None:
    owner.start_refusal = target_trial_setup.TargetTrialSetupError(
        "target_trial_compile_busy",
        "Another target trial is compiling; state this one when it finishes.",
    )
    context = _context()

    result = tool_module.execute_tool(TOOL, STATEMENT, context)

    assert (result["status"], result["code"]) == ("blocked", "target_trial_compile_busy")
    assert owner.calls == ["check", "submit"]
    assert _authority_spent(context) is None


def test_without_a_bound_study_nothing_is_checked(
    owner: _Owner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: None)
    context = _context()

    result = tool_module.execute_tool(TOOL, STATEMENT, context)

    assert (result["status"], result["code"]) == ("blocked", "study_context_required")
    assert owner.calls == []
    assert context.grant.consume_once("configure") == "granted"


def test_the_tool_waits_for_the_confirmed_data_source(owner: _Owner) -> None:
    context = _context(source_status="pending")

    result = tool_module.execute_tool(TOOL, STATEMENT, context)

    assert (result["status"], result["code"]) == (
        "blocked",
        "pi_session_data_source_confirmation_required",
    )
    assert owner.calls == []


def test_an_argument_outside_the_catalog_is_refused(owner: _Owner) -> None:
    with pytest.raises(PiCopilotError) as caught:
        tool_module.execute_tool(TOOL, {**STATEMENT, "approved": True}, _context())

    assert caught.value.code == "pi_tool_unknown_arguments"
    assert owner.calls == []
