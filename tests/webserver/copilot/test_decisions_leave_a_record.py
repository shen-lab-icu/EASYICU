"""A decision leaves its record, and the rail reads the latest job.

Counted eICU Q1' attempts 7-10 (server, 2026-10-20/21): after 「确认方案并准备数据」
or 「批准并开始分析」 the gate card vanished and nothing in the conversation said
who decided what and when; a conversation whose research job failed read
「已完成」 in the rail.  Each decision now leaves one record read from the owner
that persisted it -- the host's action row, the data-source authorization, the
provider-recovery receipt -- so a reload shows the same line; a system action
is its own actor; and the rail state comes from the conversation's latest job.
All sessions, questions and stamps here are synthetic.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from easyicu.webserver import host_action_jobs, host_action_starting, jobs, settings
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.host_action_contracts import (
    DecisionKeyFacts,
    ExecutionRetryDecision,
    HostActionRequest,
    HostDecisionOffers,
    PlanReviewDecision,
    PlanTransitionDecision,
    decision_key,
    decision_key_facts,
    retry_options_sha256,
)
from easyicu.webserver.pi_copilot import run_authority
from easyicu.webserver.pi_copilot.contracts import PiSessionResearchJob
from easyicu.webserver.pi_copilot.conversation_status import (
    conversation_status,
    later_research_job,
    research_job_state,
)
from easyicu.webserver.pi_copilot.decision_records import project_decision_records
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.provider_job_recovery import RECEIPT_NAME, read_recovery_receipt
from easyicu.webserver.run_record import RunDirectory
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

DIGEST = "a" * 64
PREPARE = PlanTransitionDecision(
    next_action_code="plan_execution_upgrade_required",
    scientific_configuration_sha256=DIGEST,
    source_run_id="run_candidate",
)
GENERATE = PlanTransitionDecision(
    next_action_code="provider_ready_to_generate_plan",
    scientific_configuration_sha256=DIGEST,
)
REVIEW = PlanReviewDecision(
    run_id="run_candidate",
    scientific_configuration_sha256=DIGEST,
    review_authority_sha256="b" * 64,
)
RETRY = ExecutionRetryDecision(
    source_run_id="run_failed",
    gate_reason="research_pipeline_execution_failed",
    scientific_configuration_sha256=DIGEST,
)
OPENAI = retry_options_sha256(report_only=False, provider="openai", credential_source="pi_verified")


def _turn(action_code: str, key: str, at: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "host_action", "action_code": action_code, "action_key": key,
        "started_at": at, "child_job_id": "job-" + action_code, **extra,
    }


def _receipt(wrapper: Path, *, job_id: str, rows: list[dict[str, Any]], **extra: Any) -> Path:
    wrapper.mkdir(parents=True, exist_ok=True)
    path = wrapper / RECEIPT_NAME
    path.write_text(json.dumps({
        "schema_version": "easyicu.provider_job_recovery/1", "job_id": job_id,
        "window_seconds": 1800.0, "max_automatic_resumes": 2, "probe_interval_seconds": 120.0,
        "automatic_resumes": rows, "stop_reason": None, "continuation_refusal": None, **extra,
    }), encoding="utf-8")
    return path


def _resume(at: str, number: int = 1, actor: str = "system") -> dict[str, Any]:
    return {"at": at, "actor": actor, "cause_code": "provider_timeout",
            "checkpoint": {"kind": "planner"}, "number": number}


# --- what a host-made key says was decided -----------------------------------------


def test_a_decision_key_reads_back_what_was_decided() -> None:
    assert decision_key_facts(decision_key(PREPARE)) == DecisionKeyFacts(
        family="plan_transition", run_id="run_candidate",
        next_action_code="plan_execution_upgrade_required",
    )
    assert decision_key_facts(decision_key(GENERATE)).run_id == ""
    for verdict in ("approved", "rejected"):
        assert decision_key_facts(decision_key(REVIEW, review_decision=verdict)) == DecisionKeyFacts(
            family="plan_review", run_id="run_candidate", review_decision=verdict,
        )
    assert decision_key_facts(decision_key(RETRY, retry_options=OPENAI)) == DecisionKeyFacts(
        family="execution_retry", run_id="run_failed",
    )
    # A key the host did not make proves nothing (browser-era rows, fragments).
    for other in ("", None, "approve", "prepare:run_candidate",
                  "review:run_candidate:approved:" + "a" * 16,
                  "plan:" + "a" * 15 + ":run_candidate:plan_execution_upgrade_required"):
        assert decision_key_facts(other) is None


# --- the records, from their owners ------------------------------------------------


def test_each_answered_gate_is_one_record_of_the_researchers() -> None:
    turns = [
        _turn("generate_plan", decision_key(GENERATE), "2026-10-21T05:59:00Z"),
        _turn("prepare_analysis_data", decision_key(PREPARE), "2026-10-21T06:03:00Z"),
        _turn("execute_plan", decision_key(REVIEW, review_decision="approved"), "2026-10-21T13:53:00Z"),
        _turn("execute_plan", decision_key(REVIEW, review_decision="rejected"), "2026-10-21T14:10:00Z"),
        # The same action under a key the host did not make, and a non-host row.
        _turn("prepare_analysis_data", "prepare:legacy-browser-row", "2026-10-21T14:20:00Z"),
        {"kind": "message", "action_code": "prepare_analysis_data",
         "action_key": decision_key(PREPARE), "started_at": "2026-10-21T14:30:00Z"},
    ]
    authorization = {
        "status": "confirmed", "confirmation_mode": "select_registered_source",
        "confirmed_at": "2026-10-21T05:58:00+00:00",
        "source": {"label": "eICU", "reference_release": "2.0", "identity_sha256": DIGEST},
    }

    records = project_decision_records(
        host_action_turns=turns, data_source_authorization=authorization, recovery_receipts=[],
    )

    assert [(row["code"], row["actor"], row["at"]) for row in records] == [
        ("data_chosen", "user", "2026-10-21T05:58:00Z"),
        ("plan_confirmed", "user", "2026-10-21T06:03:00Z"),
        ("plan_approved", "user", "2026-10-21T13:53:00Z"),
        ("plan_rejected", "user", "2026-10-21T14:10:00Z"),
    ]
    assert records[0]["source_label"] == "eICU" and records[0]["source_release"] == "2.0"
    assert {row["run_id"] for row in records[1:]} == {"run_candidate"}


@pytest.mark.parametrize(
    "authorization",
    [
        {},
        {"status": "pending", "confirmation_mode": "select_registered_source"},
        # The system's own policy and a pre-choice session are not the researcher's choice.
        {"status": "confirmed", "confirmation_mode": "agent_default_study_required",
         "confirmed_at": "2026-10-21T05:58:00Z", "source": {"label": "eICU"}},
        {"status": "legacy_confirmed", "confirmation_mode": "legacy_session"},
        {"status": "confirmed", "confirmation_mode": "select_local_source",
         "confirmed_at": "not a time", "source": {"label": "eICU"}},
    ],
)
def test_no_data_choice_record_without_the_researchers_confirmed_choice(authorization) -> None:
    assert project_decision_records(
        host_action_turns=[], data_source_authorization=authorization, recovery_receipts=[],
    ) == []


def test_a_system_continuation_is_recorded_as_the_systems(tmp_path: Path) -> None:
    wrapper = tmp_path / "run_job-1"
    _receipt(wrapper, job_id="job-1", rows=[
        _resume("2026-10-21T06:14:05.123456+00:00", 1), _resume("2026-10-21T06:40:00+00:00", 2),
    ])
    receipt = read_recovery_receipt(wrapper)
    assert receipt == {"job_id": "job-1", "max_automatic_resumes": 2, "automatic_resumes": [
        {"at": "2026-10-21T06:14:05Z", "number": 1, "cause_code": "provider_timeout"},
        {"at": "2026-10-21T06:40:00Z", "number": 2, "cause_code": "provider_timeout"},
    ]}
    # The receipt names no folder, checkpoint body or message to the page.
    assert "checkpoint" not in json.dumps(receipt)

    records = project_decision_records(
        host_action_turns=[_turn("prepare_analysis_data", decision_key(PREPARE), "2026-10-21T06:14:05Z")],
        data_source_authorization={}, recovery_receipts=[receipt],
    )
    # A click and a continuation in the same second: the click comes first.
    assert [(row["code"], row["actor"]) for row in records] == [
        ("plan_confirmed", "user"), ("provider_resumed", "system"), ("provider_resumed", "system"),
    ]
    assert [(row["number"], row["limit"]) for row in records[1:]] == [(1, 2), (2, 2)]
    assert all(row["actor"] == "system" for row in records if row["code"] == "provider_resumed")
    assert all(row["code"] != "provider_resumed" for row in records if row["actor"] == "user")


@pytest.mark.parametrize(
    "damage",
    ["user_actor", "naive_time", "over_limit", "wrong_schema", "symlink", "oversized"],
)
def test_a_receipt_that_is_not_a_system_record_is_not_read(tmp_path: Path, damage: str) -> None:
    wrapper = tmp_path / "run_job-1"
    rows = [_resume("2026-10-21T06:14:05+00:00")]
    extra: dict[str, Any] = {}
    if damage == "user_actor":
        rows = [_resume("2026-10-21T06:14:05+00:00", actor="user")]
    elif damage == "naive_time":
        rows = [_resume("2026-10-21T06:14:05")]
    elif damage == "over_limit":
        rows = [_resume("2026-10-21T06:14:05+00:00", number=3)]
    elif damage == "wrong_schema":
        extra = {"schema_version": "easyicu.provider_job_recovery/0"}
    elif damage == "oversized":
        extra = {"padding": "x" * (70 * 1024)}
    path = _receipt(wrapper, job_id="job-1", rows=rows, **extra)
    if damage == "symlink":
        real = tmp_path / "elsewhere.json"
        path.rename(real)
        os.symlink(real, path)
    assert read_recovery_receipt(wrapper) is None


# --- the rail ----------------------------------------------------------------------


EARLY, LATE = "2026-10-21T06:00:00Z", "2026-10-21T07:00:00Z"


@pytest.mark.parametrize(
    ("message", "research", "data_choice", "expected"),
    [
        # running: a reply or a research job still running
        (("running", True, LATE), (None, None), None, "running"),
        (("done", False, LATE), ("running", EARLY), None, "running"),
        # waiting for the researcher: a paused plan review, or a data choice
        (("done", False, LATE), ("needs_user", EARLY), None, "needs_user"),
        (("done", False, LATE), (None, None), "pending", "needs_user"),
        # attempt 9: the reply finished, the research job after it failed
        (("done", False, EARLY), ("stopped", LATE), None, "stopped"),
        (("failed", False, LATE), (None, None), None, "stopped"),
        (("interrupted", False, LATE), (None, None), None, "stopped"),
        # completed
        (("done", False, LATE), ("done", EARLY), None, "done"),
        (("done", False, EARLY), ("done", LATE), None, "done"),
        # a later finished reply does not settle a stopped research job
        (("done", False, LATE), ("stopped", EARLY), None, "stopped"),
        # a later failed reply after a completed research job
        (("failed", False, LATE), ("done", EARLY), None, "stopped"),
        # nothing ran yet
        ((None, False, None), (None, None), None, None),
    ],
)
def test_the_rail_state_is_the_latest_jobs(message, research, data_choice, expected) -> None:
    status, running, message_at = message
    state, research_at = research
    assert conversation_status(
        message_status=status, message_running=running, message_at=message_at,
        research_state=state, research_at=research_at, data_choice_status=data_choice,
    ) == expected


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ({"status": "running"}, "running"),
        ({"status": "done", "human_review_pending": True, "gate_status": "blocked"}, "needs_user"),
        ({"status": "done", "gate_status": "blocked"}, "stopped"),
        ({"status": "failed"}, "stopped"),
        ({"status": "cancelled"}, "stopped"),
        ({"status": "interrupted"}, "stopped"),
        ({"status": "done", "gate_status": "pass"}, "done"),
    ],
)
def test_a_research_jobs_terminal_state_maps_to_one_rail_state(job, expected) -> None:
    assert research_job_state(job) == expected


def test_the_latest_started_job_wins_and_an_ended_job_never_reopens() -> None:
    def job(job_id: str, state: str, started: str, at: str) -> PiSessionResearchJob:
        return PiSessionResearchJob(job_id=job_id, state=state, started_at=started, at=at)

    first = job("job-a", "running", EARLY, EARLY)
    ended = job("job-a", "stopped", EARLY, LATE)
    assert later_research_job(None, first) == first
    assert later_research_job(first, ended) == ended
    # A late sight of the job running does not reopen it.
    assert later_research_job(ended, first) == ended
    # A newer job replaces it; an older job's late end does not.
    newer = job("job-b", "running", LATE, LATE)
    assert later_research_job(ended, newer) == newer
    assert later_research_job(newer, job("job-a", "done", EARLY, "2026-10-21T08:00:00Z")) == newer


# --- through the service: persisted, reloaded, and in the rail ----------------------


class Conversation:
    """One conversation of one study; its research jobs end when told."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.project = f"project-records-{uuid.uuid4().hex[:10]}"
        self.service = self.reload()
        self.session_id = self.service.create_session(
            project_id=self.project, external_llm_opt_in=True
        )["session"]["session_id"]
        self.study_id = self.service._get_record(self.session_id).binding.study_context_id
        self.offers = HostDecisionOffers(plan_transition=PREPARE, plan_review=REVIEW)
        self.releases: list[threading.Event] = []

    def reload(self) -> PiCopilotService:
        """A new service over the same stores, as after a page or process reload."""

        return PiCopilotService(store_path=self.tmp_path / "sessions.json", gateway=FakeGateway())

    def start(self, action_code: str, decision: Any, result: Callable[[], dict]) -> str:
        release = threading.Event()
        self.releases.append(release)

        def submit() -> dict:
            study = context_store.get_context(self.study_id)

            def runner(job: Any) -> dict:
                release.wait(10)
                try:
                    return result()
                finally:
                    context_store.clear_active_job_if(self.study_id, job.id, current_stage="review")

            job = jobs.MANAGER.submit("agent-run", runner)
            context_store.handoff_context(
                self.study_id, current_stage="analyze", last_route="agent",
                active_job_id=job.id, expected_revision=int(study["revision"]),
            )
            return {"job_id": job.id, "kind": job.kind, "status": job.status}

        receipt = host_action_jobs.submit_with_host_action(
            HostActionRequest(session_id=self.session_id, project_id=self.project,
                              action_code=action_code, decision=decision),
            study_context_id=self.study_id, submit=submit, offered=lambda _study: self.offers,
            conversations=self.service,
            **({"review_decision": "approved"} if action_code == "execute_plan" else {}),
        )
        return receipt["job_id"]

    def finish(self, job_id: str, state: str) -> None:
        self.releases[-1].set()
        deadline = time.monotonic() + 10
        while True:
            noted = self.service._get_record(self.session_id).latest_research_job
            if noted is not None and noted.job_id == job_id and noted.state == state:
                return
            assert time.monotonic() < deadline, noted
            time.sleep(0.02)

    def rail(self, service: Optional[PiCopilotService] = None) -> Optional[str]:
        rows = (service or self.service).list_sessions(project_id=self.project)["sessions"]
        return next(row for row in rows if row["session_id"] == self.session_id)["conversation_status"]

    def records(self, service: Optional[PiCopilotService] = None) -> list[dict[str, Any]]:
        session = (service or self.service).get_session(self.session_id, project_id=self.project)["session"]
        return session["decision_records"]


@pytest.fixture
def conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    host_action_starting.clear_all_for_tests()
    host_action_jobs.clear_started_for_tests()
    state = Conversation(tmp_path)
    yield state
    for release in state.releases:
        release.set()
    host_action_starting.clear_all_for_tests()
    host_action_jobs.clear_started_for_tests()


def test_an_agent_timing_decision_survives_a_conversation_reload(conversation: Conversation) -> None:
    from easyicu.webserver.pi_copilot.plan_decisions import (
        AGENT_POST_BASELINE_LANDMARK_SPEC_ID, landmark_timing_specification,
    )

    saved = context_store.upsert_context({
        "id": conversation.study_id,
        "sensitivity_specs": [landmark_timing_specification(
            outcome="death", landmark_hours=24, spec_id=AGENT_POST_BASELINE_LANDMARK_SPEC_ID,
        )],
        "confirmations": {"plan_timing_landmark_24h": True},
    }, _server_agent_landmark_decision=True)
    expected = [{"code": "post_baseline_landmark_selected", "actor": "agent", "at": saved["updated_at"]}]
    assert conversation.records() == expected
    assert conversation.records(conversation.reload()) == expected


def test_reload_after_each_decision_shows_the_same_records(conversation: Conversation) -> None:
    prepare = conversation.start("prepare_analysis_data", PREPARE, lambda: {"human_review_pending": True})
    conversation.finish(prepare, "needs_user")
    [confirmed] = conversation.records()
    assert (confirmed["code"], confirmed["actor"]) == ("plan_confirmed", "user")
    [row] = conversation.service.replay_store.host_action_turns(
        session_id=conversation.session_id, project_id=conversation.project
    )
    assert confirmed["at"] == row["started_at"]
    assert conversation.records(conversation.reload()) == [confirmed]

    # The provider outage during preparation, continued by the system.
    resumed_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    root = run_authority.research_pipeline_workspace().project_root(conversation.study_id)
    _receipt(RunDirectory.create(root, conversation.study_id, prepare).path, job_id=prepare,
             rows=[_resume(resumed_at)])

    approve = conversation.start("execute_plan", REVIEW, lambda: {})
    conversation.finish(approve, "done")
    shown = conversation.records()
    assert [(row["code"], row["actor"]) for row in shown] == [
        ("plan_confirmed", "user"), ("plan_approved", "user"), ("provider_resumed", "system"),
    ]
    assert shown[2]["job_id"] == prepare and (shown[2]["number"], shown[2]["limit"]) == (1, 2)
    # Reload: a new service over the same stores shows exactly the same lines.
    assert conversation.records(conversation.reload()) == shown
    # The records name no folder.
    assert str(root) not in json.dumps(shown) and conversation.study_id not in json.dumps(shown)


def test_the_rail_follows_the_latest_jobs_state(conversation: Conversation) -> None:
    service = conversation.service
    # A finished reply from before the research job.
    record = service._get_record(conversation.session_id)
    record.last_turn_status = "done"
    record.conversation_updated_at = "2026-01-01T00:00:00Z"
    service._save_record(record)
    # The reply still asks which data to use.
    assert record.data_source_authorization.status == "pending"
    assert conversation.rail() == "needs_user"
    record.data_source_authorization = record.data_source_authorization.model_copy(
        update={"status": "not_required"}
    )
    service._save_record(record)
    assert conversation.rail() == "done"

    prepare = conversation.start("prepare_analysis_data", PREPARE, lambda: {"human_review_pending": True})
    assert conversation.rail() == "running"
    stale_copy = service._get_record(conversation.session_id)
    conversation.finish(prepare, "needs_user")
    assert conversation.rail() == "needs_user"

    def fails() -> dict:
        raise RuntimeError("research_pipeline_execution_failed")

    approve = conversation.start("execute_plan", REVIEW, fails)
    conversation.finish(approve, "stopped")
    # attempt 9: the reply said done; the conversation's latest job failed.
    assert service._get_record(conversation.session_id).last_turn_status == "done"
    assert conversation.rail() == "stopped"
    assert conversation.rail(conversation.reload()) == "stopped"
    # A writer holding a copy read before the job's note does not erase it.
    assert stale_copy.latest_research_job.job_id == prepare
    service._save_record(stale_copy)
    noted = service._get_record(conversation.session_id).latest_research_job
    assert (noted.job_id, noted.state) == (approve, "stopped")
    assert conversation.rail() == "stopped"


def test_a_job_the_process_no_longer_runs_stopped(
    conversation: Conversation, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation.start("prepare_analysis_data", PREPARE, lambda: {})
    assert conversation.rail() == "running"
    # The Web process restarted: its jobs are gone, the note still says running.
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    assert conversation.rail() == "stopped"
    conversation.records()  # opening the conversation reconciles the note
    noted = conversation.service._get_record(conversation.session_id).latest_research_job
    assert noted is not None and noted.state == "stopped"
