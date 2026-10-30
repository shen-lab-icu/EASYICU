"""One reviewed preparation offer owns one host job, even during races."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import os
import subprocess
import sys

import pytest

from easyicu.webserver import host_action_jobs, host_action_starting, jobs, study_contexts
from easyicu.webserver.host_action_contracts import HostActionRequest, HostDecisionOffers, RawPreparationReviewDecision


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {})
    host_action_jobs.clear_started_for_tests()
    host_action_starting.clear_all_for_tests()
    class Conversations:
        def __init__(self):
            self.rows = {}
            self.lock = Lock()
            self.first_record = Event()
            self.release_record = Event()
            self.block_first = False
            self.calls = 0
        def host_action_session(self, *args, **kwargs):
            return None
        def host_action_child_job(self, session_id, *, project_id, action_key):
            with self.lock:
                return self.rows.get((session_id, project_id, action_key))
        def record_job_host_action(self, session_id, *, project_id, action_code, action_key, child_job_id):
            with self.lock:
                self.calls += 1
                first = self.calls == 1
            if first and self.block_first:
                self.first_record.set()
                assert self.release_record.wait(10)
            with self.lock:
                self.rows[(session_id, project_id, action_key)] = {"child_job_id": child_job_id, "status": "running"}
            return {}
    owner = Conversations()
    choice = RawPreparationReviewDecision(run_id="run", scientific_configuration_sha256="a"*64,
        checkpoint_sha256="b"*64, scope_sha256="c"*64, review_authority_sha256="d"*64, nonce="e"*64)
    action = HostActionRequest(session_id="session", project_id="project", action_code="review_raw_preparation", decision=choice)
    calls = []
    def submit():
        job = jobs.MANAGER.reserve("raw-preparation-review")
        calls.append(job.id)
        return {"job_id": job.id}
    def invoke(*, action=action, decision="approved"):
        return host_action_jobs.submit_with_host_action(action, study_context_id="study", submit=submit,
            review_decision=decision, offered=lambda _: HostDecisionOffers(raw_preparation_review=choice),
            conversations=owner)
    yield owner, action, choice, calls, invoke
    host_action_jobs.clear_started_for_tests()
    host_action_starting.clear_all_for_tests()


def test_same_current_offer_reuses_recorded_job(host):
    _, _, _, calls, invoke = host
    first = invoke()
    again = invoke()
    assert again["reused"] and again["job_id"] == first["job_id"] and calls == [first["job_id"]]


def test_concurrent_repeat_reuses_job_before_first_conversation_write_returns(host, monkeypatch):
    owner, _, _, calls, invoke = host
    monkeypatch.setattr(host_action_jobs, "_STARTED_LIMIT", 1)
    owner.block_first = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(invoke)
        assert owner.first_record.wait(10)
        # Evict the normal cache while the first record is not yet durable.
        host_action_jobs._remember_started("other-study", "other-key", "other-job")
        second = pool.submit(invoke).result(timeout=10)
        owner.release_record.set()
        first = first_future.result(timeout=10)
    assert second["reused"] and second["job_id"] == first["job_id"] and calls == [first["job_id"]]


def test_opposite_answer_and_other_session_cannot_reuse(host):
    _, action, _, calls, invoke = host
    invoke()
    with pytest.raises(host_action_jobs.HostActionJobError, match="already_answered"):
        invoke(decision="rejected")
    with pytest.raises(host_action_jobs.HostActionJobError, match="original_job_mismatch"):
        invoke(action=action.model_copy(update={"session_id": "other"}))
    assert len(calls) == 1


def test_new_nonce_is_a_distinct_offer(host):
    owner, action, choice, calls, invoke = host
    first = invoke()
    new_choice = choice.model_copy(update={"nonce": "f" * 64})
    new_action = action.model_copy(update={"decision": new_choice})
    def submit():
        job = jobs.MANAGER.reserve("raw-preparation-review")
        calls.append(job.id)
        return {"job_id": job.id}
    second = host_action_jobs.submit_with_host_action(new_action, study_context_id="study", submit=submit,
        review_decision="approved", offered=lambda _: HostDecisionOffers(raw_preparation_review=new_choice),
        conversations=owner)
    assert not second["reused"] and first["job_id"] != second["job_id"] and len(calls) == 2


def test_missing_or_failed_original_never_restarts(host, monkeypatch):
    _, _, _, calls, invoke = host
    first = invoke()
    jobs.MANAGER.get(first["job_id"]).fail_from_runner("synthetic_failure")
    with pytest.raises(host_action_jobs.HostActionJobError, match="original_job_terminal"):
        invoke()
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    host_action_jobs.clear_started_for_tests()
    with pytest.raises(host_action_jobs.HostActionJobError, match="original_job_unavailable"):
        invoke()
    assert len(calls) == 1


def test_signed_job_binding_survives_cache_eviction_and_cross_session(host, monkeypatch):
    _, action, _, calls, invoke = host
    first = invoke()
    host_action_jobs.clear_started_for_tests()
    assert invoke()["job_id"] == first["job_id"]
    host_action_jobs.clear_started_for_tests()
    with pytest.raises(host_action_jobs.HostActionJobError, match="original_job_mismatch"):
        invoke(action=action.model_copy(update={"session_id": "other"}))
    assert calls == [first["job_id"]]


def test_tampered_job_binding_fails_closed(host, tmp_path):
    from easyicu import state_paths
    _, _, choice, calls, invoke = host
    invoke()
    path = state_paths.state_root() / "private-preparation-decisions" / (choice.nonce + ".host-job.json")
    path.write_text('{}')
    with pytest.raises(host_action_jobs.HostActionJobError, match="original_job_mismatch"):
        invoke()
    assert len(calls) == 1


def test_missing_job_binding_with_decision_receipt_never_starts_another_job(host):
    from easyicu import state_paths
    _, _, choice, calls, invoke = host
    invoke()
    root = state_paths.state_root() / "private-preparation-decisions"
    (root / (choice.nonce + ".host-job.json")).unlink()
    (root / (choice.nonce + ".json")).write_text("{}")
    host_action_jobs.clear_started_for_tests()
    with pytest.raises(host_action_jobs.HostActionJobError, match="original_job_unavailable"):
        invoke()
    assert len(calls) == 1


def test_job_binding_publication_failure_cannot_be_reused(host, monkeypatch):
    _, _, _, calls, invoke = host
    def fail_publication(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(host_action_jobs, "_write_raw_claim", fail_publication)
    with pytest.raises(OSError, match="disk failure"):
        invoke()
    with pytest.raises(host_action_jobs.HostActionJobError, match="original_job_unavailable"):
        invoke()
    assert len(calls) == 1


def test_original_job_reused_after_independent_process_restart(tmp_path):
    script = r'''
import sys
from pathlib import Path
from easyicu.webserver import host_action_jobs, jobs, study_contexts
from easyicu.webserver.host_action_contracts import HostActionRequest, HostDecisionOffers, RawPreparationReviewDecision
root=Path(sys.argv[1]); phase=sys.argv[2]
jobs.MANAGER.enable_persistence(root / 'jobs.sqlite3')
study_contexts.get_context=lambda _: {}
class Conversations:
    def host_action_session(self, *a, **k): pass
    def host_action_child_job(self, *a, **k): return None
    def record_job_host_action(self, *a, **k): return {}
choice=RawPreparationReviewDecision(run_id='run',scientific_configuration_sha256='a'*64,checkpoint_sha256='b'*64,
    scope_sha256='c'*64,review_authority_sha256='d'*64,nonce='e'*64)
action=HostActionRequest(session_id='session',project_id='project',action_code='review_raw_preparation',decision=choice)
def submit():
    if phase != 'first': raise AssertionError('restart submitted a new job')
    return {'job_id': jobs.MANAGER.reserve('raw-preparation-review',job_id='same-job').id}
result=host_action_jobs.submit_with_host_action(action,study_context_id='study',submit=submit,review_decision='approved',
    offered=lambda _: HostDecisionOffers(raw_preparation_review=choice),conversations=Conversations())
assert result['job_id']=='same-job'
if phase=='first':
    assert not result['reused']
    jobs.MANAGER.get('same-job').complete_from_runner({})
else:
    assert result['reused']
'''
    env = os.environ.copy()
    env["EASYICU_HOME"] = str(tmp_path / "home")
    for phase in ("first", "restart"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), phase], env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
def test_crash_after_job_reservation_does_not_start_another_job(tmp_path):
    script = r'''
import os, signal, sys
from pathlib import Path
from easyicu.webserver import host_action_jobs, jobs, study_contexts
from easyicu.webserver.host_action_contracts import HostActionRequest, HostDecisionOffers, RawPreparationReviewDecision
root=Path(sys.argv[1]); phase=sys.argv[2]
jobs.MANAGER.enable_persistence(root / 'jobs.sqlite3')
study_contexts.get_context=lambda _: {}
class Conversations:
    def host_action_session(self, *a, **k): pass
    def host_action_child_job(self, *a, **k): return None
    def record_job_host_action(self, *a, **k): return {}
choice=RawPreparationReviewDecision(run_id='run',scientific_configuration_sha256='a'*64,checkpoint_sha256='b'*64,
    scope_sha256='c'*64,review_authority_sha256='d'*64,nonce='e'*64)
action=HostActionRequest(session_id='session',project_id='project',action_code='review_raw_preparation',decision=choice)
def submit():
    if phase != 'first': raise AssertionError('recovery submitted a second job')
    jobs.MANAGER.reserve('raw-preparation-review',job_id='first-job')
    os.kill(os.getpid(), signal.SIGKILL)
try:
    host_action_jobs.submit_with_host_action(action,study_context_id='study',submit=submit,review_decision='approved',
        offered=lambda _: HostDecisionOffers(raw_preparation_review=choice),conversations=Conversations())
except host_action_jobs.HostActionJobError as exc:
    assert phase == 'restart' and exc.code == 'raw_preparation_original_job_unavailable'
else:
    raise AssertionError('incomplete admission was accepted')
'''
    env = os.environ.copy()
    env["EASYICU_HOME"] = str(tmp_path / "home")
    first = subprocess.run([sys.executable, "-c", script, str(tmp_path), "first"], env=env,
                           capture_output=True, text=True, timeout=30)
    assert first.returncode == -9, first.stderr
    second = subprocess.run([sys.executable, "-c", script, str(tmp_path), "restart"], env=env,
                            capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr


def test_original_route_accepts_identical_checkpoint_published_by_replay(tmp_path, monkeypatch):
    from easyicu.webserver import raw_preparation_candidate, raw_preparation_decisions
    from easyicu.webserver.routes import agent as route
    from tests.webserver.test_raw_preparation_decisions import _host_decision

    pending, args, values = _host_decision(tmp_path, monkeypatch)
    path = tmp_path / "raw_preparation_checkpoint.json"
    path.write_text(pending.model_dump_json())
    action = args["host_action"]
    monkeypatch.setattr(raw_preparation_candidate, "available_host_review", lambda **_: (
        action.decision, pending, values["scope"], tmp_path))
    monkeypatch.setattr(route, "_submit_host_decision", lambda *a, **k: {
        "reused": False, "job_id": "job", "host_action": {"recorded": True}})
    original = raw_preparation_decisions.record_host_review
    def publish_during_original(**kwargs):
        decided = original(**kwargs)
        # Another request already recovered and committed these exact bytes.
        path.write_text(decided.model_dump_json())
        return decided
    monkeypatch.setattr(raw_preparation_decisions, "record_host_review", publish_during_original)
    completed = []
    monkeypatch.setattr(jobs.MANAGER, "run_reserved", lambda job_id, callback: completed.append((job_id, callback(None))))
    result = route.raw_preparation_review({"study_context_id": "study", "decision": "approved",
                                           "host_action": action.model_dump(mode="json")})
    assert result["decision"] == "approved" and not result["analysis_authorized"]
    assert completed == [("job", {"decision": "approved", "analysis_authorized": False})]
