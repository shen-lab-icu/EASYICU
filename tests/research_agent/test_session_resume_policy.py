"""Track 2 resume semantics + pairwise tournament ranking."""

from __future__ import annotations

import pytest

from easyicu.research_agent import pipeline as pipeline_module
from easyicu.research_agent.discovery.hypothesis_generator import (
    HypothesisCandidate,
    pairwise_tournament_rank,
)
from easyicu.research_agent.execution import phase as phase_module


def _session(
    project="proj",
    task="task",
    session="sess",
    status="suspended",
    attempts=1,
    diagnosis="rerun step",
    valid=(),
):
    return {
        "project_id": project,
        "task_id": task,
        "session_id": session,
        "status": status,
        "attempt_count": attempts,
        "next_diagnosis": diagnosis,
        "valid_evidence_ids": list(valid),
    }


def _candidate(cid, score):
    return HypothesisCandidate(
        hypothesis_family_id="fam",
        candidate_id=cid,
        predictor=f"pred_{cid}",
        outcome="death",
        question=f"q {cid}",
        variable_coverage=0.8,
        literature_saturation_signal=0.1,
        icu_gate=1.0,
        priority_score=score,
        rationale="r",
    )


def test_pipeline_resumes_suspended_and_retains_only_valid():
    decision = pipeline_module.session_resume_decision(
        sessions=[_session(status="suspended", valid=["ev_keep", "ev_gone"])],
        valid_evidence_ids=["ev_keep"],
        max_attempts=3,
    )
    assert decision.stopped is False
    assert decision.resume_session_keys == (("proj", "task", "sess"),)
    assert decision.retained_evidence_ids == ("ev_keep",)


def test_pipeline_stops_at_max_attempts():
    decision = pipeline_module.session_resume_decision(
        sessions=[_session(status="failed", attempts=3)],
        valid_evidence_ids=[],
        max_attempts=3,
    )
    assert decision.stopped is True
    assert decision.reason == "stop_condition_reached"
    assert decision.resume_session_keys == ()


def test_pipeline_drops_session_without_next_diagnosis():
    decision = pipeline_module.session_resume_decision(
        sessions=[_session(status="failed", diagnosis="")],
        valid_evidence_ids=[],
        max_attempts=3,
    )
    assert decision.stopped is True
    assert decision.resume_session_keys == ()


def test_pipeline_completed_sessions_only_retain():
    decision = pipeline_module.session_resume_decision(
        sessions=[_session(session="done", status="completed", valid=["ev1"])],
        valid_evidence_ids=["ev1"],
        max_attempts=3,
    )
    assert decision.stopped is True
    assert decision.reason == "no_resumable_sessions"
    assert decision.retained_evidence_ids == ("ev1",)


def test_pipeline_from_event_log_dict_wrapper():
    log_dict = {
        "sessions": [_session(status="suspended")],
        "events": [],
    }
    decision = pipeline_module.session_resume_from_event_log_dict(
        log_dict=log_dict, valid_evidence_ids=[], max_attempts=3
    )
    assert decision.stopped is False
    with pytest.raises(ValueError):
        pipeline_module.session_resume_from_event_log_dict(
            log_dict="nope", valid_evidence_ids=[]
        )


def test_phase_retains_valid_and_reruns_from_earliest_affected():
    plan = phase_module.plan_session_resume_steps(
        ordered_step_ids=["01", "02", "03"],
        retained_step_ids=["01", "02"],
        failed_or_suspended_step_ids=["02"],
        attempts_by_step={"02": 1},
        has_diagnosis_by_step={"02": True},
        max_attempts=3,
    )
    assert plan.stopped is False
    assert plan.steps_skipped_retained == ("01",)
    assert plan.steps_to_run == ("02", "03")


def test_phase_stops_without_next_diagnosis_or_at_max_attempts():
    stopped_no_diag = phase_module.plan_session_resume_steps(
        ordered_step_ids=["01", "02"],
        retained_step_ids=["01"],
        failed_or_suspended_step_ids=["02"],
        attempts_by_step={"02": 1},
        has_diagnosis_by_step={"02": False},
        max_attempts=3,
    )
    assert stopped_no_diag.stopped is True
    stopped_max = phase_module.plan_session_resume_steps(
        ordered_step_ids=["01", "02"],
        retained_step_ids=["01"],
        failed_or_suspended_step_ids=["02"],
        attempts_by_step={"02": 3},
        has_diagnosis_by_step={"02": True},
        max_attempts=3,
    )
    assert stopped_max.stopped is True
    assert phase_module.should_stop_session_retry(
        attempts=3, max_attempts=3, has_next_diagnosis=True
    ) is True
    assert phase_module.should_stop_session_retry(
        attempts=1, max_attempts=3, has_next_diagnosis=True
    ) is False


def test_tournament_default_orders_by_priority_score():
    candidates = [_candidate("c1", 0.5), _candidate("c2", 0.9), _candidate("c3", 0.7)]
    ranked = pairwise_tournament_rank(candidates)
    assert [c.candidate_id for c in ranked] == ["c2", "c3", "c1"]
    # Input is not mutated.
    assert [c.candidate_id for c in candidates] == ["c1", "c2", "c3"]


def test_tournament_custom_beats_and_deterministic_tiebreak():
    low = _candidate("a_low", 0.1)
    high = _candidate("z_high", 0.2)

    def beats(a, b):
        # Deliberately prefer the low-priority candidate.
        if a.candidate_id == "a_low" and b.candidate_id == "z_high":
            return 1
        if a.candidate_id == "z_high" and b.candidate_id == "a_low":
            return -1
        return 0

    ranked = pairwise_tournament_rank([high, low], beats=beats)
    assert [c.candidate_id for c in ranked] == ["a_low", "z_high"]
    tied = [_candidate("b", 0.5), _candidate("a", 0.5)]
    ranked_tied = pairwise_tournament_rank(tied)
    assert [c.candidate_id for c in ranked_tied] == ["a", "b"]
    with pytest.raises(ValueError):
        pairwise_tournament_rank([high, low], beats=lambda a, b: 42)


"""Track 2 wiring: execute-phase session filter + outcome recording.

Store-verified semantics (review finding): a log entry is a claim, store
presence is the fact. The filter skips a retained step only when every
output the log claims for it resolves in the live EvidenceStore; the
recorder logs the step records' real ``evidence_ids`` and never invents
ids.
"""

from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import (
    EvidenceStore,
    SessionEventLog,
    SessionStatus,
)


def _wiring_log(tmp_path):
    store = EvidenceStore(tmp_path / "wiring")
    return store, SessionEventLog(store)


def _register(store, evidence_id):
    store.register_json(
        kind="log",
        description="session resume wiring test artefact",
        payload={"ok": True},
        filename=f"{evidence_id}.json",
        evidence_id=evidence_id,
    )


def _steps(*ids):
    return [SimpleNamespace(step_id=step_id) for step_id in ids]


def test_wiring_fresh_session_passes_queue_through(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    findings = []
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1", "s2"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=findings,
        evidence_store=store,
    )
    assert [step.step_id for step in out] == ["s1", "s2"]
    assert findings == []


def test_wiring_completed_session_refuses_rerun(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="done once")
    log.complete_session("proj", "task", "sess")
    with pytest.raises(ValueError, match="already completed"):
        phase_module.session_resume_step_filter(
            steps_to_run=_steps("s1"),
            session_event_log=log,
            session_key=("proj", "task", "sess"),
            findings=[],
            evidence_store=store,
        )


def test_wiring_unknown_session_key_fails_closed(tmp_path):
    store, log = _wiring_log(tmp_path)
    with pytest.raises(KeyError):
        phase_module.session_resume_step_filter(
            steps_to_run=_steps("s1"),
            session_event_log=log,
            session_key=("proj", "task", "nope"),
            findings=[],
            evidence_store=store,
        )


def test_wiring_requires_an_evidence_store(tmp_path):
    _, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    with pytest.raises(ValueError, match="requires an evidence store"):
        phase_module.session_resume_step_filter(
            steps_to_run=_steps("s1"),
            session_event_log=log,
            session_key=("proj", "task", "sess"),
            findings=[],
            evidence_store=None,
        )


def test_wiring_suspended_session_restarts_at_earliest_affected(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    _register(store, "ev_s1")
    log.log_completed_artifact(
        "proj", "task", "sess", evidence_id="ev_s1", produced_by_step="s1"
    )
    log.log_failure_reason("proj", "task", "sess", reason="s2 blew up", step_id="s2")
    log.log_next_diagnosis(
        "proj", "task", "sess", diagnosis="rerun s2 with fixed input", step_id="s2"
    )
    log.suspend_session("proj", "task", "sess")
    findings = []
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1", "s2", "s3"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=findings,
        evidence_store=store,
    )
    assert [step.step_id for step in out] == ["s2", "s3"]
    assert findings == []


def test_wiring_phantom_evidence_reruns_with_finding(tmp_path):
    """Review repro: zero real evidence must never skip a step."""

    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    log.log_completed_artifact(
        "proj", "task", "sess", evidence_id="ev_ghost", produced_by_step="s1"
    )
    log.log_failure_reason("proj", "task", "sess", reason="s2 blew up", step_id="s2")
    log.log_next_diagnosis(
        "proj", "task", "sess", diagnosis="rerun everything", step_id="s2"
    )
    log.suspend_session("proj", "task", "sess")
    findings = []
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1", "s2"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=findings,
        evidence_store=store,
    )
    assert [step.step_id for step in out] == ["s1", "s2"]
    kinds = [finding.detail.get("kind") for finding in findings]
    assert "session_resume_phantom_evidence" in kinds
    assert all(finding.severity == "error" for finding in findings)


def test_wiring_stop_condition_empties_queue_with_finding(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    log.log_failure_reason("proj", "task", "sess", reason="s1 blew up", step_id="s1")
    log.suspend_session("proj", "task", "sess")
    findings = []
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1", "s2"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=findings,
        evidence_store=store,
    )
    assert out == []
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["kind"] == "session_resume_stopped"


def test_wiring_log_pair_rule():
    phase_module.require_session_log_pair(None, None)
    phase_module.require_session_log_pair(object(), ("a", "b", "c"))
    with pytest.raises(ValueError, match="together"):
        phase_module.require_session_log_pair(object(), None)
    with pytest.raises(ValueError, match="together"):
        phase_module.require_session_log_pair(None, ("a", "b", "c"))
    with pytest.raises(ValueError, match="triple"):
        phase_module.session_resume_step_filter(
            steps_to_run=[], session_event_log=object(),
            session_key=("a", "b"), findings=[],
            evidence_store=object(),
        )


def test_wiring_records_real_evidence_ids_and_completes(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    phase_module.record_session_step_outcomes(
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        per_step_records=[
            {"step_id": "s1", "status": "ok", "evidence_ids": ["ev_a", "ev_b"]},
            {"step_id": "s2", "status": "ok", "evidence_ids": ["ev_c"]},
        ],
    )
    node = log.get_session("proj", "task", "sess")
    assert node.status == SessionStatus.COMPLETED.value
    assert node.valid_evidence_ids == ["ev_a", "ev_b", "ev_c"]


def test_wiring_records_nothing_without_registered_ids(tmp_path):
    """No synthetic ids are invented: an ok record without evidence ids
    retains nothing (review finding)."""

    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    phase_module.record_session_step_outcomes(
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        per_step_records=[
            {"step_id": "s1", "status": "ok"},
            {"step_id": "s2", "status": "ok", "evidence_ids": []},
        ],
    )
    node = log.get_session("proj", "task", "sess")
    assert node.status == SessionStatus.COMPLETED.value
    assert node.valid_evidence_ids == []


def test_wiring_records_failure_and_keeps_valid_artefacts(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    phase_module.record_session_step_outcomes(
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        per_step_records=[
            {"step_id": "s1", "status": "ok", "evidence_ids": ["ev_a"]},
            {"step_id": "s2", "status": "contract_failed"},
        ],
    )
    node = log.get_session("proj", "task", "sess")
    assert node.status == SessionStatus.FAILED.value
    assert node.valid_evidence_ids == ["ev_a"]
    assert "s2" in (node.failure_reason or "")


def test_wiring_empty_records_leave_session_untouched(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    phase_module.record_session_step_outcomes(
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        per_step_records=[],
    )
    node = log.get_session("proj", "task", "sess")
    assert node.status == SessionStatus.RUNNING.value


def test_wiring_roundtrip_record_then_filter_skips_verified(tmp_path):
    """End-to-end across the two seams with one real store: recorded real
    ids verify on the next run and the step is skipped."""

    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    _register(store, "ev_real")
    phase_module.record_session_step_outcomes(
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        per_step_records=[
            {"step_id": "s1", "status": "ok", "evidence_ids": ["ev_real"]},
            {"step_id": "s2", "status": "contract_failed"},
        ],
    )
    log.log_next_diagnosis(
        "proj", "task", "sess", diagnosis="rerun s2", step_id="s2"
    )
    log.suspend_session("proj", "task", "sess")
    findings = []
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1", "s2"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=findings,
        evidence_store=store,
    )
    assert [step.step_id for step in out] == ["s2"]
    assert findings == []


def test_wiring_deleted_artifact_bytes_rerun_with_finding(tmp_path):
    """Review repro: deleting the actual product file must un-skip the step,
    even though the log entry and the store record still exist."""

    from easyicu.research_agent.authority.runtime_artifacts import (
        verified_run_evidence_path,
    )

    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    _register(store, "ev_gone")
    record = store.get("ev_gone")
    assert record is not None
    assert verified_run_evidence_path(store.root, record) is not None
    log.log_completed_artifact(
        "proj", "task", "sess", evidence_id="ev_gone", produced_by_step="s1"
    )
    log.log_failure_reason("proj", "task", "sess", reason="s2 blew up", step_id="s2")
    log.log_next_diagnosis(
        "proj", "task", "sess", diagnosis="rerun everything", step_id="s2"
    )
    log.suspend_session("proj", "task", "sess")
    doomed = verified_run_evidence_path(store.root, record)
    assert doomed is not None and doomed.is_file()
    doomed.unlink()
    assert store.get("ev_gone") is not None

    findings = []
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1", "s2"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=findings,
        evidence_store=store,
    )
    assert [step.step_id for step in out] == ["s1", "s2"]
    kinds = [finding.detail.get("kind") for finding in findings]
    assert "session_resume_phantom_evidence" in kinds


def test_wiring_retry_advances_attempt_count(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    _register(store, "ev_s1")
    log.log_completed_artifact(
        "proj", "task", "sess", evidence_id="ev_s1", produced_by_step="s1"
    )
    log.log_failure_reason("proj", "task", "sess", reason="s2 blew up", step_id="s2")
    log.log_next_diagnosis(
        "proj", "task", "sess", diagnosis="rerun s2", step_id="s2"
    )
    log.suspend_session("proj", "task", "sess")
    assert log.get_session("proj", "task", "sess").attempt_count == 1
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1", "s2"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=[],
        evidence_store=store,
    )
    assert [step.step_id for step in out] == ["s2"]
    node = log.get_session("proj", "task", "sess")
    assert node.attempt_count == 2
    assert node.status == "running"


def test_wiring_refused_run_consumes_no_attempt(tmp_path):
    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    log.log_failure_reason("proj", "task", "sess", reason="s1 blew up", step_id="s1")
    node = log.get_session("proj", "task", "sess")
    node.attempt_count = 3
    log.suspend_session("proj", "task", "sess")
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1"),
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        findings=[],
        evidence_store=store,
    )
    assert out == []
    assert log.get_session("proj", "task", "sess").attempt_count == 3


def test_wiring_last_permitted_attempt_runs_before_stop(tmp_path):
    store, log = _wiring_log(tmp_path)
    key = ("proj", "task", "sess")
    log.start_session(*key, attempt_rationale="first attempt")
    log.log_failure_reason(*key, reason="first failure", step_id="s1")
    log.log_next_diagnosis(*key, diagnosis="try again", step_id="s1")
    log.suspend_session(*key)
    log.resume_session(*key, attempt_rationale="second attempt")
    log.log_failure_reason(*key, reason="second failure", step_id="s1")
    log.log_next_diagnosis(*key, diagnosis="last diagnostic", step_id="s1")
    log.suspend_session(*key)

    findings = []
    out = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1"),
        session_event_log=log,
        session_key=key,
        findings=findings,
        evidence_store=store,
    )
    assert [step.step_id for step in out] == ["s1"]
    assert log.get_session(*key).attempt_count == 3
    assert log.get_session(*key).status == "running"
    assert findings == []

    log.log_failure_reason(*key, reason="third failure", step_id="s1")
    log.log_next_diagnosis(*key, diagnosis="no budget left", step_id="s1")
    log.suspend_session(*key)
    refused = phase_module.session_resume_step_filter(
        steps_to_run=_steps("s1"),
        session_event_log=log,
        session_key=key,
        findings=[],
        evidence_store=store,
    )
    assert refused == []
    assert log.get_session(*key).attempt_count == 3
    assert log.get_session(*key).status == "suspended"


def test_wiring_phase_end_persists_log_for_next_entry(tmp_path):
    """Review repro: memory-completed must reload as completed, not suspended."""

    from easyicu.research_agent.authority.evidence_store import SessionEventLog

    store, log = _wiring_log(tmp_path)
    log.start_session("proj", "task", "sess", attempt_rationale="first attempt")
    phase_module._record_session_outcomes_best_effort(
        session_event_log=log,
        session_key=("proj", "task", "sess"),
        per_step_records=[
            {"step_id": "s1", "status": "ok", "evidence_ids": ["ev_a"]},
        ],
        findings=[],
    )
    reloaded = SessionEventLog.load_latest(store)
    assert reloaded.get_session("proj", "task", "sess").status == "completed"
