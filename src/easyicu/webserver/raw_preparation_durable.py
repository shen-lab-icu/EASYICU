"""Durable pre-consumption intent for an exactly approved preparation.

Only preparations that have not consumed their decision can be automatically
started after a crash. Consumed decisions need a separate, stronger recovery
contract and are deliberately refused here. This module does not enqueue jobs.
"""
from __future__ import annotations

import hmac
import json
from pathlib import Path
import re
from typing import Any, Mapping

from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
from easyicu.webserver import agent_pipeline_runs, study_contexts
from easyicu.webserver.durable_research_store import runtime_digest
from easyicu.webserver.multitable_guarded_worker import guarded_multitable_projection
from easyicu.webserver.raw_preparation_decisions import (
    RawPreparationHostBinding, _private_directory, _publish_once, _signature, verify_host_review,
)
from easyicu.webserver.reviewed_multitable_projection import require_reviewable_multitable_scope

_JOB = re.compile(r'^[A-Za-z0-9_-]{1,128}$')


def _owner(binding: RawPreparationHostBinding, receipt: Mapping[str, Any], job_id: str) -> Path:
    from easyicu.webserver.pi_copilot.service import get_pi_copilot_service
    if not _JOB.fullmatch(job_id) or receipt.get('job_id') != job_id:
        raise ValueError('preparation_intent_job_mismatch')
    owner = get_pi_copilot_service()
    owner.host_action_session(binding.session_id, project_id=binding.project_id,
                              study_context_id=binding.study_context_id)
    row = owner.host_action_child_job(binding.session_id, project_id=binding.project_id,
                                      action_key=receipt.get('decision_key'))
    if not row or row.get('child_job_id') != job_id:
        raise ValueError('preparation_intent_action_mismatch')
    pending, _ = agent_pipeline_runs.raw_preparation_review_owner(binding.run_id, binding.study_context_id)
    root = Path(pending.run_dir)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('preparation_intent_root_invalid')
    current = study_contexts.get_context(binding.study_context_id)
    if not current or study_contexts.scientific_configuration_sha256(current) != receipt.get('scientific_configuration_sha256'):
        raise ValueError('preparation_intent_configuration_changed')
    return root


def _review(*, checkpoint: RawPreparationCheckpoint, binding: RawPreparationHostBinding,
            scope: Mapping[str, Any], job_id: str) -> tuple[dict, Path]:
    if canonical_sha256(checkpoint.request.payload.get('raw_preparation_scope')) != canonical_sha256(dict(scope)):
        raise ValueError('preparation_intent_scope_mismatch')
    require_reviewable_multitable_scope(scope)
    receipt = dict(verify_host_review(checkpoint=checkpoint, scope=scope, **binding.model_dump(mode='python')))
    return receipt, _owner(binding, receipt, job_id)


def admit_preparation_intent(*, job_id: str, checkpoint: RawPreparationCheckpoint,
                             binding: RawPreparationHostBinding, scope: Mapping[str, Any]) -> None:
    """Save immutable intent before launching; does not consume the decision."""
    receipt, root = _review(checkpoint=checkpoint, binding=binding, scope=scope, job_id=job_id)
    body = {'schema_version': 'easyicu.preparation-intent/1', 'job_id': job_id,
            'binding': binding.model_dump(mode='json'), 'checkpoint': checkpoint.model_dump(mode='json'),
            'scope': json.loads(canonical_json(dict(scope))), 'root': str(root),
            'scientific_configuration_sha256': receipt['scientific_configuration_sha256'],
            'decision_key': receipt['decision_key'], 'runtime_sha256': runtime_digest()}
    content = canonical_json({**body, 'signature': _signature(body)}).encode()
    # The host decision directory already uses owner-only permissions and safe
    # anchored publication. A distinct suffix cannot collide with a decision.
    with AnchoredDirectory.open(_private_directory()) as directory:
        _publish_once(directory, job_id + '.preparation-intent.json', content)
        directory.assert_still_selected()


def recover_preparation_before_consumption(*, job_id: str, job: Any, timeout: float) -> Path:
    """Continue the original job only while its signed decision is unconsumed.

    Missing/unsafe prior lifecycle proof, changed inputs or a consumed nonce
    refuse. Repeated callers still contend for the same exclusive host lease.
    """
    return _recover_preparation(job_id=job_id, job=job, timeout=timeout, sealed_recovery=False)


def recover_preparation_with_sealed_snapshot(*, job_id: str, job: Any, timeout: float) -> Path:
    """Continue an exact intent; consumed decisions require a sealed snapshot.

    This entry point is not wired to product dispatch. The guarded worker must
    verify the seal and never read the original source on a consumed retry.
    """
    return _recover_preparation(job_id=job_id, job=job, timeout=timeout, sealed_recovery=True)


def _recover_preparation(*, job_id: str, job: Any, timeout: float, sealed_recovery: bool) -> Path:
    if not _JOB.fullmatch(job_id) or not job or getattr(job, 'id', None) != job_id:
        raise ValueError('preparation_intent_job_mismatch')
    if job.is_cancel_requested():
        raise ValueError('preparation_intent_cancelled')
    with AnchoredDirectory.open(_private_directory()) as directory:
        value = json.loads(directory.read_bytes(job_id + '.preparation-intent.json', max_bytes=1024 * 1024))
        directory.assert_still_selected()
    if not isinstance(value, dict):
        raise ValueError('preparation_intent_invalid')
    sig = value.pop('signature', None)
    if not isinstance(sig, str) or not hmac.compare_digest(_signature(value), sig):
        raise ValueError('preparation_intent_invalid')
    if value.get('schema_version') != 'easyicu.preparation-intent/1' or value.get('job_id') != job_id:
        raise ValueError('preparation_intent_invalid')
    if value.get('runtime_sha256') != runtime_digest():
        raise ValueError('preparation_intent_runtime_changed')
    checkpoint = RawPreparationCheckpoint.model_validate(value.get('checkpoint'))
    binding = RawPreparationHostBinding.model_validate(value.get('binding'))
    scope = value.get('scope')
    if not isinstance(scope, dict):
        raise ValueError('preparation_intent_invalid')
    with AnchoredDirectory.open(_private_directory()) as directory:
        consumed = not directory.is_absent(binding.nonce + '.used')
        directory.assert_still_selected()
    if consumed and sealed_recovery:
        # No raw source resolver runs on a consumed retry. The signed intent,
        # original host decision and current conversation owner remain binding.
        from easyicu.webserver.preparation_recovery_authority import verify_recovery_authority
        receipt = dict(verify_recovery_authority(checkpoint=checkpoint, binding=binding,
                                                 scope=scope, job_id=job_id))
        root = _owner(binding, receipt, job_id)
    else:
        receipt, root = _review(checkpoint=checkpoint, binding=binding, scope=scope, job_id=job_id)
    if (str(root) != value.get('root') or receipt.get('decision_key') != value.get('decision_key')
            or receipt.get('scientific_configuration_sha256') != value.get('scientific_configuration_sha256')):
        raise ValueError('preparation_intent_binding_changed')
    # A restarted Job has no in-memory action tag. Rebind only after the
    # persisted conversation action and signed decision were both verified.
    from easyicu.webserver.host_action_contracts import HostActionTag, host_action_id
    tag = HostActionTag(session_id=binding.session_id, project_id=binding.project_id,
        study_context_id=binding.study_context_id, action_code='review_raw_preparation',
        decision_key=receipt['decision_key'],
        action_id=host_action_id(binding.session_id, 'review_raw_preparation', receipt['decision_key']))
    if not job.tag_host_action(tag):
        raise ValueError('preparation_intent_action_mismatch')
    lease_dir = root / ('.preparation-execution-' + job_id)
    # Record launch intent before the lease exists. A crash in that narrow
    # window stays blocked instead of treating a missing receipt as proof.
    first = False
    with AnchoredDirectory.open(_private_directory()) as directory:
        if consumed and sealed_recovery and directory.is_absent(job_id + '.preparation-started'):
            raise ValueError('preparation_intent_consumed_start_missing')
        try:
            _publish_once(directory, job_id + '.preparation-started', canonical_json({
                'job_id': job_id, 'intent_sha256': canonical_sha256(value),
                'signature': _signature(value),
            }).encode())
            first = True
        except FileExistsError:
            marker = json.loads(directory.read_bytes(job_id + '.preparation-started', max_bytes=4096))
            if marker != {'job_id': job_id, 'intent_sha256': canonical_sha256(value),
                          'signature': _signature(value)}:
                raise ValueError('preparation_intent_start_invalid')
        directory.assert_still_selected()
    assert_host_execution_quiescent(lease_dir, allow_missing=first, wait_timeout=2)
    if job.is_cancel_requested():
        raise ValueError('preparation_intent_cancelled')
    unregister = []
    def prepared(nonce: str) -> None:
        from easyicu.runtime.host_execution_lease import request_host_execution_stop
        unregister.append(job.register_cancel_callback(lambda: request_host_execution_stop(lease_dir, nonce)))
    try:
        arguments = dict(lease_dir=lease_dir, checkpoint=checkpoint, binding=binding, scope=scope,
                         host_work_root=root, timeout=timeout, on_prepared=prepared)
        if sealed_recovery:
            arguments['job_id'] = job_id
        return guarded_multitable_projection(**arguments)
    finally:
        for remove in unregister:
            remove()
