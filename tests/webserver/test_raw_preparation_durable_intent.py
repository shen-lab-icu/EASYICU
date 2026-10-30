"""Pre-consumption intent: immutable original decision, no replay authority."""
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.webserver import raw_preparation_durable as durable
from easyicu.webserver.jobs import Job
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding


@pytest.fixture
def intent(tmp_path, monkeypatch):
    scope = {'adapter_id': 'registered-parquet-projections/1', 'inputs': ['synthetic']}
    payload = {'raw_preparation_scope': scope}
    request = HumanReviewRequest.create(kind='scientific_stop', summary='Synthetic only', payload=payload,
                                        authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256='c' * 64, request=request)
    checkpoint = pending.decide(HumanReviewDecision(review_id=request.review_id,
        authority_sha256=request.authority_sha256, decision='approved', reviewer='test',
        decided_at=datetime.now(timezone.utc).isoformat()))
    binding = RawPreparationHostBinding(project_id='p', session_id='s', study_context_id='study',
                                        run_id='run', nonce='b' * 64)
    private = tmp_path / 'private'; private.mkdir(mode=0o700)
    work = tmp_path / 'work'; work.mkdir()
    receipt = {'job_id':'job', 'decision_key':'decision', 'scientific_configuration_sha256':'a'*64}
    monkeypatch.setattr(durable, '_private_directory', lambda: private)
    monkeypatch.setattr(durable, '_signature', lambda value: hashlib.sha256(canonical_json(value).encode()).hexdigest())
    monkeypatch.setattr(durable, 'runtime_digest', lambda: 'e' * 64)
    monkeypatch.setattr(durable, '_review', lambda **kwargs: (receipt, work))
    job = Job('job', 'raw-preparation-review')
    durable.admit_preparation_intent(job_id='job', checkpoint=checkpoint, binding=binding, scope=scope)
    return SimpleNamespace(private=private, work=work, job=job, scope=scope,
                           checkpoint=checkpoint, binding=binding)


def test_intent_reuses_exact_job_and_requires_prior_lease_on_retry(intent, monkeypatch):
    calls = []
    monkeypatch.setattr(durable, 'assert_host_execution_quiescent',
                        lambda path, **kwargs: calls.append((path, kwargs['allow_missing'])))
    monkeypatch.setattr(durable, 'guarded_multitable_projection', lambda **kwargs: kwargs['host_work_root'] / 'output')
    for _ in range(2):
        assert durable.recover_preparation_before_consumption(job_id='job', job=intent.job, timeout=5) == intent.work / 'output'
    assert [value for _, value in calls] == [True, False]
    assert calls[0][0] == intent.work / '.preparation-execution-job'
    with pytest.raises(FileExistsError):
        durable.admit_preparation_intent(job_id='job', checkpoint=intent.checkpoint,
                                          binding=intent.binding, scope=intent.scope)


def test_intent_refuses_changed_code_tamper_cancel_and_consumed_decision(intent, monkeypatch):
    monkeypatch.setattr(durable, 'runtime_digest', lambda: 'f' * 64)
    with pytest.raises(ValueError, match='runtime_changed'):
        durable.recover_preparation_before_consumption(job_id='job', job=intent.job, timeout=5)
    monkeypatch.setattr(durable, 'runtime_digest', lambda: 'e' * 64)
    intent.job.is_cancel_requested = lambda: True
    with pytest.raises(ValueError, match='cancelled'):
        durable.recover_preparation_before_consumption(job_id='job', job=intent.job, timeout=5)
    intent.job.is_cancel_requested = lambda: False
    def consumed(**kwargs):
        raise ValueError('raw_preparation_host_nonce_replayed')
    monkeypatch.setattr(durable, '_review', consumed)
    with pytest.raises(ValueError, match='nonce_replayed'):
        durable.recover_preparation_before_consumption(job_id='job', job=intent.job, timeout=5)
    path = intent.private / 'job.preparation-intent.json'
    data = json.loads(path.read_text()); data['root'] = '/tmp/changed'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='intent_invalid'):
        durable.recover_preparation_before_consumption(job_id='job', job=intent.job, timeout=5)


def test_started_without_lease_fails_closed(intent, monkeypatch):
    from easyicu.runtime.host_execution_lease import HostExecutionLeaseError

    def crash_before_lease(**kwargs):
        raise RuntimeError('simulated_crash_before_lease')

    monkeypatch.setattr(durable, 'guarded_multitable_projection', crash_before_lease)
    with pytest.raises(RuntimeError, match='simulated_crash_before_lease'):
        durable.recover_preparation_before_consumption(job_id='job', job=intent.job, timeout=5)
    assert (intent.private / 'job.preparation-started').is_file()
    assert not (intent.work / '.preparation-execution-job').exists()
    with pytest.raises(HostExecutionLeaseError, match='host_execution_receipt_missing'):
        durable.recover_preparation_before_consumption(job_id='job', job=intent.job, timeout=5)
