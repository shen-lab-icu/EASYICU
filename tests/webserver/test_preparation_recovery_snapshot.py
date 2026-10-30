"""A sealed snapshot retains exact identity across preparation attempts."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from easyicu import state_paths
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.webserver import preparation_recovery_snapshot as recovery
from easyicu.webserver import raw_preparation_decisions as decisions
from easyicu.webserver.raw_case_snapshot import SnapshotFile, StagedRawSnapshot
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding


@pytest.fixture
def staged(tmp_path, monkeypatch):
    state = tmp_path / 'state'; state.mkdir()
    monkeypatch.setattr(state_paths, 'state_root', lambda: state)
    work = tmp_path / 'work'; work.mkdir()
    parent = work / 'reviewed-multitable-snapshot-12345678'
    source = parent / 'snapshot'; source.mkdir(parents=True)
    (source / 'a.parquet').write_bytes(b'synthetic a')
    (source / 'b.parquet').write_bytes(b'synthetic b')
    scope = {'source_identity': {'database': 'synthetic'}, 'inputs': [
        {'table': 'a', 'path': 'a.parquet', 'columns': ['x']},
        {'table': 'b', 'path': 'b.parquet', 'columns': ['y']}], 'plan_sha256': 'c' * 64}
    payload = {'raw_preparation_scope': scope}
    request = HumanReviewRequest.create(kind='scientific_stop', summary='Synthetic only',
        payload=payload, authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256='c' * 64, request=request)
    checkpoint = pending.decide(HumanReviewDecision(review_id=request.review_id,
        authority_sha256=request.authority_sha256, decision='approved', reviewer='test',
        decided_at=datetime.now(timezone.utc).isoformat()))
    binding = RawPreparationHostBinding(project_id='project', session_id='session',
        study_context_id='study', run_id='run', nonce='b' * 64)
    private = decisions._private_directory()
    receipt = {'schema_version': 'easyicu.raw-preparation-host-decision/1', 'actor': 'host_user',
        'decision': 'approved', **binding.model_dump(mode='python'), 'job_id': 'job',
        'scope_sha256': canonical_sha256(scope), 'decided_checkpoint_sha256': checkpoint.checkpoint_sha256,
        'request_authority_sha256': request.authority_sha256, 'decision_key': 'decision',
        'scientific_configuration_sha256': 'a' * 64,
        'expires_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}
    def write_decision():
        (private / (binding.nonce + '.json')).write_text(canonical_json({**receipt, 'signature': decisions._signature(receipt)}))
        (private / (binding.nonce + '.used')).write_text(canonical_json(
            {'decision_sha256': canonical_sha256(receipt), 'nonce': binding.nonce}))
    write_decision()
    monkeypatch.setattr(recovery, 'runtime_digest', lambda: 'e' * 64)
    monkeypatch.setattr(recovery, 'worker_identity', lambda: 'f' * 64)
    monkeypatch.setattr(recovery, 'verify_host_worker', lambda *args: 'f' * 64)
    intent = {'schema_version': 'easyicu.preparation-intent/1', 'job_id': 'job',
        'binding': binding.model_dump(mode='json'), 'checkpoint': checkpoint.model_dump(mode='json'),
        'scope': scope, 'root': str(work), 'scientific_configuration_sha256': 'a' * 64,
        'decision_key': 'decision', 'runtime_sha256': 'e' * 64}
    (private / 'job.preparation-intent.json').write_text(canonical_json({**intent, 'signature': decisions._signature(intent)}))
    snapshot = StagedRawSnapshot(canonical_sha256(scope['source_identity']), tuple(
        SnapshotFile(name, hashlib.sha256((source / name).read_bytes()).hexdigest(), (source / name).stat().st_size)
        for name in ['a.parquet', 'b.parquet']))
    arguments = {'job_id': 'job', 'checkpoint': checkpoint, 'binding': binding, 'scope': scope,
        'host_work_root': work, 'lease_dir': work / '.preparation-execution-job', 'lease_nonce': 'lease'}
    return arguments, snapshot, parent, private, receipt, write_decision


def _seal(staged):
    args, snapshot, parent, *_ = staged
    recovery.seal_preparation_snapshot(**args, snapshot_parent=parent, snapshot=snapshot)


def test_sealed_snapshot_restores_without_original_source(staged):
    args, snapshot, parent, *_ = staged
    _seal(staged)
    # There is no registered or original source at all in this test.
    path, actual = recovery.verify_preparation_snapshot(**args)
    assert path == parent / 'snapshot'
    assert actual == snapshot
    assert actual.public_receipt()['point_in_time_source_snapshot_verified'] is False
    with pytest.raises(FileExistsError):
        _seal(staged)


@pytest.mark.parametrize('change', ['bytes', 'missing', 'extra'])
def test_changed_or_incomplete_snapshot_fails_closed(staged, change):
    args, _, parent, *_ = staged
    _seal(staged)
    path = parent / 'snapshot'
    if change == 'bytes':
        (path / 'a.parquet').write_bytes(b'changed')
    elif change == 'missing':
        (path / 'a.parquet').unlink()
    else:
        (path / 'unexpected').write_bytes(b'extra')
    with pytest.raises(ValueError, match='raw_snapshot_'):
        recovery.verify_preparation_snapshot(**args)


@pytest.mark.parametrize('change', ['runtime', 'worker', 'guard', 'scope', 'intent', 'seal', 'missing_seal', 'expired'])
def test_recovery_rejects_changed_authority_or_coordinates(staged, monkeypatch, change):
    args, _, _, private, receipt, write_decision = staged
    _seal(staged)
    if change == 'runtime':
        monkeypatch.setattr(recovery, 'runtime_digest', lambda: '1' * 64)
    elif change == 'worker':
        monkeypatch.setattr(recovery, 'worker_identity', lambda: '1' * 64)
    elif change == 'guard':
        args = {**args, 'lease_dir': args['host_work_root'] / 'different'}
    elif change == 'scope':
        args = {**args, 'scope': {**args['scope'], 'inputs': []}}
    elif change == 'intent':
        path = private / 'job.preparation-intent.json'
        value = json.loads(path.read_text()); value['runtime_sha256'] = '1' * 64; path.write_text(json.dumps(value))
    elif change == 'seal':
        path = private / 'job.preparation-snapshot.json'
        value = json.loads(path.read_text()); value['directory'] = 'reviewed-multitable-snapshot-abcdefgh'; path.write_text(json.dumps(value))
    elif change == 'missing_seal':
        (private / 'job.preparation-snapshot.json').unlink()
    else:
        receipt['expires_at'] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        write_decision()
    with pytest.raises((ValueError, OSError)):
        recovery.verify_preparation_snapshot(**args)


def test_sealing_rejects_wrong_roster_before_publishing(staged):
    args, snapshot, parent, private, *_ = staged
    incomplete = StagedRawSnapshot(snapshot.source_identity_sha256, snapshot.files[:1])
    with pytest.raises(ValueError, match='preparation_recovery_roster_invalid'):
        recovery.seal_preparation_snapshot(**args, snapshot_parent=parent, snapshot=incomplete)
    assert not (private / 'job.preparation-snapshot.json').exists()


def test_recovery_rechecks_runtime_after_reading_snapshot(staged, monkeypatch):
    args, *_ = staged
    _seal(staged)
    values = iter(['e' * 64, '1' * 64])
    monkeypatch.setattr(recovery, 'runtime_digest', lambda: next(values))
    with pytest.raises(ValueError, match='preparation_recovery_intent_mismatch'):
        recovery.verify_preparation_snapshot(**args)
