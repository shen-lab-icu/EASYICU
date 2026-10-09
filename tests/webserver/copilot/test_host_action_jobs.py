"""One job per host decision of a study, recorded by the host.

On 2026-10-09 at 02:22Z a click on 「确认方案并准备数据」 took about 50 seconds
to return: the submission's start checks run before its job exists, and only
then did the browser record the click in the conversation.  A page closed in
between lost the record, and the click repeated after reopening met a 409.

The submission now carries the decision it answers; the host starts one job
per decision of the study, tells a repeat the decision is starting, reuses the
running job, refuses a decision the study no longer offers, and writes the
conversation row itself.  These tests drive the owner directly, as a browser
that has already gone away would.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

from easyicu.webserver import host_action_jobs, host_action_starting, jobs, settings
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.host_action_contracts import (
    ExecutionRetryDecision,
    HostActionRequest,
    HostDecisionOffers,
    PlanReviewDecision,
    PlanTransitionDecision,
    retry_options_sha256,
)
from easyicu.webserver.host_action_jobs import HostActionJobError
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.workflow import build_project_workflow_projection
from easyicu.webserver.research_run_submission import ResearchRunSubmissionError
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

DIGEST = "a" * 64
PREPARE = PlanTransitionDecision(
    next_action_code="plan_execution_upgrade_required",
    scientific_configuration_sha256=DIGEST,
    source_run_id="run_candidate",
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
OPENAI = retry_options_sha256(
    report_only=False, provider="openai", credential_source="pi_verified"
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    host_action_starting.clear_all_for_tests()
    host_action_jobs.clear_started_for_tests()
    yield
    host_action_starting.clear_all_for_tests()
    host_action_jobs.clear_started_for_tests()


class Study:
    """One study, two conversations of its project, and what it offers now."""

    def __init__(self, tmp_path: Path) -> None:
        # Study contexts live in the suite's shared state home: one project
        # (and so one study) per test.
        self.project = f"project-host-action-{uuid.uuid4().hex[:10]}"
        self.service = PiCopilotService(
            store_path=tmp_path / "sessions.json", gateway=FakeGateway()
        )
        first = self.service.create_session(
            project_id=self.project, external_llm_opt_in=True
        )
        second = self.service.create_session(
            project_id=self.project, external_llm_opt_in=True
        )
        self.sessions = (
            first["session"]["session_id"],
            second["session"]["session_id"],
        )
        self.study_id = first["session"]["binding"]["study_context_id"]
        assert second["session"]["binding"]["study_context_id"] == self.study_id
        self.offers = HostDecisionOffers(
            plan_transition=PREPARE, plan_review=REVIEW, execution_retry=RETRY
        )
        self.release = threading.Event()
        self.submitted: list[str] = []

    def ask(
        self,
        action_code: str = "prepare_analysis_data",
        decision: Any = PREPARE,
        session: int = 0,
    ) -> HostActionRequest:
        return HostActionRequest(
            session_id=self.sessions[session],
            project_id=self.project,
            action_code=action_code,
            decision=decision,
        )

    def submit(
        self, *, fails: bool = False, clears_pointer: bool = True
    ) -> Callable[[], dict]:
        """The existing submission: start a job and point the study at it."""

        def run() -> dict:
            study = context_store.get_context(self.study_id)

            def runner(job: Any) -> dict:
                self.release.wait(10)
                try:
                    if fails:
                        raise RuntimeError("research_pipeline_execution_failed")
                    return {}
                finally:
                    if clears_pointer:
                        context_store.clear_active_job_if(
                            self.study_id, job.id, current_stage="review"
                        )

            job = jobs.MANAGER.submit("agent-run", runner)
            context_store.handoff_context(
                self.study_id,
                current_stage="analyze",
                last_route="agent",
                active_job_id=job.id,
                expected_revision=int(study["revision"]),
            )
            self.submitted.append(job.id)
            return {"job_id": job.id, "kind": job.kind, "status": job.status}

        return run

    def start(self, request: HostActionRequest, **kwargs: Any) -> dict:
        submit = kwargs.pop("submit", None) or self.submit()
        if request.action_code == "execute_plan":
            kwargs.setdefault("review_decision", "approved")
        if request.action_code == "retry_analysis":
            kwargs.setdefault("retry_options", OPENAI)
        return host_action_jobs.submit_with_host_action(
            request,
            study_context_id=self.study_id,
            submit=submit,
            offered=lambda _study: self.offers,
            conversations=self.service,
            **kwargs,
        )

    def rows(self, session: int = 0) -> list[dict]:
        return self.service.replay_store.host_action_turns(
            session_id=self.sessions[session], project_id=self.project
        )

    def finish(self, job_id: str, status: str) -> None:
        self.release.set()
        deadline = time.monotonic() + 10
        while jobs.MANAGER.get(job_id).status != status:
            assert time.monotonic() < deadline, jobs.MANAGER.get(job_id).status
            time.sleep(0.01)


@pytest.fixture
def study(tmp_path: Path):
    state = Study(tmp_path)
    yield state
    state.release.set()


def refused(call: Callable[[], Any]) -> HostActionJobError:
    with pytest.raises(HostActionJobError) as caught:
        call()
    return caught.value


def test_the_host_records_the_click_in_the_request_that_starts_its_job(
    study: Study,
) -> None:
    receipt = study.start(study.ask())

    assert receipt["reused"] is False
    assert receipt["host_action"]["recorded"] is True
    [row] = study.rows()
    assert row["action_code"] == "prepare_analysis_data"
    assert row["child_job_id"] == receipt["job_id"]
    assert row["action_key"] == receipt["host_action"]["decision_key"]
    assert row["job_id"] == receipt["host_action"]["action_id"]
    assert row["status"] == "running"
    assert (
        jobs.MANAGER.get(receipt["job_id"]).host_action.decision_key
        == row["action_key"]
    )


def test_a_repeat_while_the_job_runs_gets_the_same_job(study: Study) -> None:
    first = study.start(study.ask())
    second = study.start(study.ask())

    assert second["reused"] is True
    assert second["job_id"] == first["job_id"]
    assert study.submitted == [first["job_id"]]
    assert len(study.rows()) == 1


def test_two_conversations_of_one_study_start_one_job_and_each_records_it(
    study: Study,
) -> None:
    first = study.start(study.ask(session=0))
    second = study.start(study.ask(session=1))

    assert second["reused"] is True
    assert second["job_id"] == first["job_id"]
    assert study.submitted == [first["job_id"]]
    assert [row["child_job_id"] for row in study.rows(0)] == [first["job_id"]]
    assert [row["child_job_id"] for row in study.rows(1)] == [first["job_id"]]


def test_a_repeat_during_the_start_checks_is_told_the_decision_is_starting(
    study: Study,
) -> None:
    entered, proceed = threading.Event(), threading.Event()
    inner = study.submit()

    def slow_checks() -> dict:
        entered.set()
        proceed.wait(10)
        return inner()

    result: dict = {}
    first = threading.Thread(
        target=lambda: result.update(
            receipt=study.start(study.ask(), submit=slow_checks)
        )
    )
    first.start()
    assert entered.wait(5)
    try:
        began = time.monotonic()
        error = refused(lambda: study.start(study.ask(session=1)))
        assert time.monotonic() - began < 2  # told, not kept waiting
        assert (error.status_code, error.code) == (409, "host_action_in_progress")
        assert error.detail["details"]["action_code"] == "prepare_analysis_data"
        browser = build_project_workflow_projection(
            study_context_id=study.study_id, include_starting=True
        )
        assert browser.workflow.next_action_code == "starting"
        assert browser.starting.action_code == "prepare_analysis_data"
        # Every authority read keeps the state the decision answers.
        assert (
            build_project_workflow_projection(
                study_context_id=study.study_id
            ).workflow.next_action_code
            != "starting"
        )
    finally:
        proceed.set()
        first.join(10)
    assert host_action_starting.starting_for(study.study_id) is None
    again = study.start(study.ask(session=1))
    assert again["reused"] is True
    assert again["job_id"] == result["receipt"]["job_id"]
    assert len(study.submitted) == 1


def test_another_decision_of_the_study_waits_for_the_running_one(study: Study) -> None:
    first = study.start(study.ask())
    other = PlanTransitionDecision(
        next_action_code="plan_configuration_superseded",
        scientific_configuration_sha256=DIGEST,
        source_run_id="run_candidate",
    )

    error = refused(lambda: study.start(study.ask("generate_plan", other, session=1)))

    assert (error.status_code, error.code) == (409, "study_job_running")
    assert error.detail["details"]["job_id"] == first["job_id"]
    assert study.submitted == [first["job_id"]]


def test_a_repeat_between_tag_and_clear_is_never_taken_for_another_decision(
    study: Study, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = jobs.Job.tag_host_action
    repeat: dict = {}

    def tag_while_a_repeat_arrives(job: jobs.Job, value: Any) -> bool:
        def ask_again() -> None:
            try:
                repeat["outcome"] = (
                    "reused",
                    study.start(study.ask(session=1))["job_id"],
                )
            except HostActionJobError as exc:
                repeat["outcome"] = (exc.code, None)

        thread = threading.Thread(target=ask_again)
        thread.start()
        thread.join(0.3)
        repeat["thread"] = thread
        return tag(job, value)

    monkeypatch.setattr(jobs.Job, "tag_host_action", tag_while_a_repeat_arrives)
    first = study.start(study.ask())
    repeat["thread"].join(10)

    assert repeat["outcome"] in {
        ("reused", first["job_id"]),
        ("host_action_in_progress", None),
    }
    assert study.submitted == [first["job_id"]]


def test_a_failed_approval_whose_review_was_consumed_is_refused_as_stale(
    study: Study,
) -> None:
    first = study.start(
        study.ask("execute_plan", REVIEW), submit=study.submit(fails=True)
    )
    study.finish(first["job_id"], "failed")
    # The resumed review is gone; the plan's next step is something else.
    study.offers = HostDecisionOffers(plan_transition=PREPARE)

    error = refused(lambda: study.start(study.ask("execute_plan", REVIEW)))

    assert (error.status_code, error.code) == (409, "host_action_decision_stale")
    assert error.detail["details"] == {
        "mismatched_fields": ["family"],
        "job_id": first["job_id"],
        "job_status": "failed",
    }
    assert study.submitted == [first["job_id"]]


def test_a_job_that_ended_without_clearing_its_pointer_holds_no_decision(
    study: Study,
) -> None:
    # The runner clears the study's pointer in a finally that swallows its own
    # failure (the external disk's I/O errors of 10-04): the pointer can
    # outlive the job it names, which the job manager knows has ended.
    first = study.start(
        study.ask(), submit=study.submit(fails=True, clears_pointer=False)
    )
    study.finish(first["job_id"], "failed")
    assert context_store.get_context(study.study_id)["active_job_id"] == first["job_id"]

    again = study.start(study.ask(session=1))

    assert again["reused"] is False
    assert again["job_id"] != first["job_id"]
    assert len(study.submitted) == 2


def test_a_page_opened_before_the_job_finished_is_answered_by_that_job(
    study: Study,
) -> None:
    first = study.start(study.ask())
    study.finish(first["job_id"], "done")
    study.offers = HostDecisionOffers(
        plan_transition=PREPARE.model_copy(
            update={"next_action_code": "operator_plan_approval_required"}
        )
    )

    again = study.start(study.ask(session=1))

    assert again["reused"] is True
    assert again["job_id"] == first["job_id"]
    assert again["status"] == "done"
    assert study.submitted == [first["job_id"]]


def test_a_refused_start_leaves_the_decision_free_to_ask_again(study: Study) -> None:
    def capacity_full() -> dict:
        raise ResearchRunSubmissionError(
            {"error": "job_capacity_exceeded"}, status_code=429
        )

    with pytest.raises(ResearchRunSubmissionError):
        study.start(study.ask("execute_plan", REVIEW), submit=capacity_full)
    assert host_action_starting.starting_for(study.study_id) is None
    assert study.rows() == []

    # The paused review is still offered, so the same approval starts its job.
    receipt = study.start(study.ask("execute_plan", REVIEW))
    assert receipt["reused"] is False
    assert study.submitted == [receipt["job_id"]]


def test_after_a_restart_a_lost_job_no_longer_holds_its_decision(
    study: Study, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = study.start(study.ask())
    # A restart: the study still names the job, the new process has no jobs.
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    host_action_starting.clear_all_for_tests()
    host_action_jobs.clear_started_for_tests()
    assert context_store.get_context(study.study_id)["active_job_id"] == first["job_id"]
    browser = build_project_workflow_projection(
        study_context_id=study.study_id, include_starting=True
    )
    assert browser.workflow.next_action_code not in {
        "starting",
        "research_planning_running",
    }

    study.offers = HostDecisionOffers(
        plan_transition=PREPARE.model_copy(
            update={"next_action_code": "question_required"}
        )
    )
    error = refused(lambda: study.start(study.ask()))
    assert error.code == "host_action_decision_stale"
    assert error.detail["details"]["job_id"] == first["job_id"]
    assert error.detail["details"]["job_status"] == "interrupted"

    study.offers = HostDecisionOffers(plan_transition=PREPARE)
    again = study.start(study.ask())
    assert again["reused"] is False
    assert again["job_id"] != first["job_id"]


def test_a_failed_conversation_write_is_backfilled_when_the_conversation_is_read(
    study: Study, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = study.service.replay_store.record_host_action

    def disk_error(**_kwargs: Any) -> dict:
        raise PiCopilotError("pi_replay_write_failed", "I/O error", status_code=500)

    monkeypatch.setattr(study.service.replay_store, "record_host_action", disk_error)
    receipt = study.start(study.ask())
    assert receipt["host_action"]["recorded"] is False
    assert receipt["host_action"]["record_error"] == "pi_replay_write_failed"
    assert study.rows() == []
    # The job and the study's pointer stand; the next read writes the row.
    assert jobs.MANAGER.get(receipt["job_id"]).status == "running"

    monkeypatch.setattr(study.service.replay_store, "record_host_action", write)
    study.service.get_session(study.sessions[0], project_id=study.project)
    assert [row["child_job_id"] for row in study.rows()] == [receipt["job_id"]]
    assert study.rows(1) == []


def test_the_browser_endpoint_returns_the_hosts_row_and_writes_none_of_its_own(
    study: Study,
) -> None:
    receipt = study.start(study.ask())

    answered = study.service.record_host_action(
        study.sessions[0],
        project_id=study.project,
        action_code="prepare_analysis_data",
        action_key=receipt["job_id"],
        child_job_id=receipt["job_id"],
    )
    assert (
        answered["host_action"]["action_key"] == receipt["host_action"]["decision_key"]
    )
    assert len(study.rows()) == 1
    with pytest.raises(PiCopilotError) as caught:
        study.service.record_host_action(
            study.sessions[1],
            project_id=study.project,
            action_code="prepare_analysis_data",
            action_key=receipt["job_id"],
            child_job_id=receipt["job_id"],
        )
    assert (caught.value.status_code, caught.value.code) == (
        409,
        "host_action_server_recorded",
    )
    assert study.rows(1) == []
    # An action that starts no job is still the browser's to record.
    study.service.record_host_action(
        study.sessions[1],
        project_id=study.project,
        action_code="review_results",
        action_key="run_candidate:results",
    )
    assert [row["action_code"] for row in study.rows(1)] == ["review_results"]


def test_a_retry_with_another_model_connection_is_another_decision(
    study: Study,
) -> None:
    first = study.start(study.ask("retry_analysis", RETRY))
    same = study.start(study.ask("retry_analysis", RETRY, session=1))
    assert same["job_id"] == first["job_id"]

    codex = retry_options_sha256(
        report_only=False, provider="codex", credential_source="codex_user_auth"
    )
    error = refused(
        lambda: study.start(study.ask("retry_analysis", RETRY), retry_options=codex)
    )
    assert error.code == "study_job_running"
    report_only = retry_options_sha256(
        report_only=True, provider="openai", credential_source="pi_verified"
    )
    error = refused(
        lambda: study.start(
            study.ask("retry_analysis", RETRY), retry_options=report_only
        )
    )
    assert error.code == "study_job_running"
    assert study.submitted == [first["job_id"]]


def test_the_echoed_decision_is_checked_field_by_field(study: Study) -> None:
    changed = PREPARE.model_copy(update={"scientific_configuration_sha256": "c" * 64})

    error = refused(lambda: study.start(study.ask(decision=changed)))

    assert (error.status_code, error.code) == (409, "host_action_decision_stale")
    assert error.detail["details"]["mismatched_fields"] == [
        "scientific_configuration_sha256"
    ]
    assert error.detail["details"]["job_id"] is None
    assert study.submitted == []
    assert host_action_starting.starting_for(study.study_id) is None


def test_a_conversation_of_another_project_or_study_cannot_ask(
    study: Study, tmp_path: Path
) -> None:
    other = PiCopilotService(store_path=tmp_path / "other.json", gateway=FakeGateway())
    foreign = other.create_session(
        project_id="project-elsewhere", external_llm_opt_in=True
    )
    request = HostActionRequest(
        session_id=study.sessions[0],
        project_id="project-elsewhere",
        action_code="prepare_analysis_data",
        decision=PREPARE,
    )
    error = refused(lambda: study.start(request))
    assert error.code == "pi_session_project_mismatch"

    error = refused(
        lambda: host_action_jobs.submit_with_host_action(
            study.ask(),
            study_context_id=foreign["session"]["binding"]["study_context_id"],
            submit=study.submit(),
            offered=lambda _study: study.offers,
            conversations=study.service,
        )
    )
    assert error.code == "host_action_study_mismatch"
    assert study.submitted == []


def test_a_host_action_names_one_family_and_no_key() -> None:
    with pytest.raises(HostActionJobError) as caught:
        host_action_jobs.parse_host_action(
            {
                "session_id": "pi_1",
                "project_id": "p1",
                "action_code": "execute_plan",
                "decision": PREPARE.model_dump(),
            }
        )
    assert caught.value.code == "host_action_invalid"
    with pytest.raises(HostActionJobError):
        host_action_jobs.parse_host_action(
            {
                "session_id": "pi_1",
                "project_id": "p1",
                "action_code": "prepare_analysis_data",
                "decision": PREPARE.model_dump(),
                "decision_key": "plan:forged",
            }
        )
    assert host_action_jobs.parse_host_action(None) is None


def test_the_replayed_0222_click_starts_one_job_and_one_row(study: Study) -> None:
    """Approve, close the page during the checks, reopen, approve again."""

    entered, proceed = threading.Event(), threading.Event()
    inner = study.submit()

    def checks() -> dict:
        entered.set()
        proceed.wait(10)
        return inner()

    # The first request runs on; its page is gone and never reads the answer.
    first = threading.Thread(target=lambda: study.start(study.ask(), submit=checks))
    first.start()
    assert entered.wait(5)
    reopened = build_project_workflow_projection(
        study_context_id=study.study_id, include_starting=True
    )
    assert reopened.workflow.next_action_code == "starting"
    assert refused(lambda: study.start(study.ask())).code == "host_action_in_progress"
    proceed.set()
    first.join(10)

    again = study.start(study.ask())
    assert again["reused"] is True
    assert len(study.submitted) == 1
    assert [row["child_job_id"] for row in study.rows()] == study.submitted
