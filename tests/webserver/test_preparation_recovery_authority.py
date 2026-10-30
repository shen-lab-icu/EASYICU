"""Consumed authorization for continuation is distinct from result access."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from easyicu import state_paths
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.webserver import raw_preparation_decisions as decisions
from easyicu.webserver.preparation_recovery_authority import verify_recovery_authority
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding


@pytest.fixture
def authority(tmp_path, monkeypatch):
    monkeypatch.setattr(state_paths, 'state_root', lambda: tmp_path)
    scope = {'adapter_id': 'synthetic', 'inputs': ['file.parquet']}
    payload = {'raw_preparation_scope': scope}
    request = HumanReviewRequest.create(kind='scientific_stop', summary='Synthetic only',
        payload=payload, authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256='c' * 64, request=request)
    checkpoint = pending.decide(HumanReviewDecision(review_id=request.review_id,
        authority_sha256=request.authority_sha256, decision='approved', reviewer='test',
        decided_at=datetime.now(timezone.utc).isoformat()))
    binding = RawPreparationHostBinding(project_id='project', session_id='session',
        study_context_id='study', run_id='run', nonce='b' * 64)
    directory = decisions._private_directory()
    body = {'schema_version': 'easyicu.raw-preparation-host-decision/1', 'actor': 'host_user',
        'decision': 'approved', **binding.model_dump(mode='python'), 'job_id': 'job',
        'scope_sha256': canonical_sha256(scope),
        'decided_checkpoint_sha256': checkpoint.checkpoint_sha256,
        'request_authority_sha256': checkpoint.request.authority_sha256,
        'expires_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}

    def save():
        (directory / (binding.nonce + '.json')).write_text(canonical_json({**body, 'signature': decisions._signature(body)}))
        (directory / (binding.nonce + '.used')).write_text(canonical_json(
            {'decision_sha256': canonical_sha256(body), 'nonce': binding.nonce}))

    save()
    return body, save, checkpoint, binding, scope, directory


def test_live_consumed_decision_is_bound_to_original_job_and_scope(authority):
    body, _, checkpoint, binding, scope, _ = authority
    result = verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope=scope, job_id='job')
    assert result == body


@pytest.mark.parametrize('expiry', [
    (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    '2026-10-30T00:00:00',
    'invalid',
    None,
])
def test_expiry_blocks_work_but_preserves_read_only_result_access(authority, expiry):
    body, save, checkpoint, binding, scope, _ = authority
    body['expires_at'] = expiry
    save()
    with pytest.raises(ValueError, match='preparation_continuation_expir'):
        verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope=scope, job_id='job')
    assert decisions.verify_consumed_host_review(checkpoint=checkpoint, binding=binding, scope=scope) == body


def test_tampered_decision_is_rejected(authority):
    _, _, checkpoint, binding, scope, directory = authority
    path = directory / (binding.nonce + '.json')
    altered = json.loads(path.read_text()); altered['job_id'] = 'other-job'; path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='raw_preparation_host_receipt_invalid'):
        verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope=scope, job_id='job')


def test_exact_scope_job_binding_and_consumption_are_required(authority):
    _, _, checkpoint, binding, scope, directory = authority
    with pytest.raises(ValueError, match='preparation_recovery_scope_mismatch'):
        verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope={'inputs': ['other']}, job_id='job')
    with pytest.raises(ValueError, match='preparation_job_binding_mismatch'):
        verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope=scope, job_id='other')
    other = binding.model_copy(update={'session_id': 'another'})
    with pytest.raises(ValueError, match='raw_preparation_host_binding_mismatch'):
        verify_recovery_authority(checkpoint=checkpoint, binding=other, scope=scope, job_id='job')
    (directory / (binding.nonce + '.used')).unlink()
    with pytest.raises(ValueError, match='raw_preparation_host_receipt_invalid'):
        verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope=scope, job_id='job')


def test_invalid_job_id_is_rejected(authority):
    _, _, checkpoint, binding, scope, _ = authority
    with pytest.raises(ValueError, match='preparation_job_invalid'):
        verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope=scope, job_id='../job')
