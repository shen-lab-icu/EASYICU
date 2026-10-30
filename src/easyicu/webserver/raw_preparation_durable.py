"""Durable pre-consumption intent for an exactly approved preparation.

Only preparations that have not consumed their decision can be automatically
started after a crash. Consumed decisions need a separate, stronger recovery
contract and are deliberately refused here. This module does not enqueue jobs.
"""
from __future__ import annotations

import hmac
import json
import math
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
    pending, planned = agent_pipeline_runs.raw_preparation_review_owner(binding.run_id, binding.study_context_id)
    root = Path(pending.run_dir)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('preparation_intent_root_invalid')
    current = study_contexts.get_context(binding.study_context_id)
    if (not current or not isinstance(planned, Mapping)
            or study_contexts.scientific_configuration_sha256(current) != receipt.get('scientific_configuration_sha256')
            or study_contexts.scientific_configuration_sha256(planned) != receipt.get('scientific_configuration_sha256')):
        raise ValueError('preparation_intent_configuration_changed')
    return root


def _scope_matches(checkpoint: RawPreparationCheckpoint, scope: Mapping[str, Any]) -> bool:
    scope_key = ('first_observed_icu_preparation_scope'
                 if scope.get('adapter_id') == 'eicu-demo-first-observed-single-hospital-adult/1'
                 else 'raw_preparation_scope')
    return canonical_sha256(checkpoint.request.payload.get(scope_key)) == canonical_sha256(dict(scope))


def _validate_current_scope(*, root: Path, checkpoint: RawPreparationCheckpoint, scope: Mapping[str, Any]) -> None:
    if scope.get('adapter_id') == 'eicu-demo-first-observed-single-hospital-adult/1':
        from easyicu.webserver.first_observed_preparation import _host_inputs
        expected, _ = _host_inputs(run_dir=root, request=checkpoint.request)
        if expected != dict(scope):
            raise ValueError('preparation_intent_scope_mismatch')
    else:
        require_reviewable_multitable_scope(scope)


def _review(*, checkpoint: RawPreparationCheckpoint, binding: RawPreparationHostBinding,
            scope: Mapping[str, Any], job_id: str) -> tuple[dict, Path]:
    if not _scope_matches(checkpoint, scope):
        raise ValueError('preparation_intent_scope_mismatch')
    receipt = dict(verify_host_review(checkpoint=checkpoint, scope=scope, **binding.model_dump(mode='python')))
    root = _owner(binding, receipt, job_id)
    _validate_current_scope(root=root, checkpoint=checkpoint, scope=scope)
    return receipt, root


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


def admit_preparation_handoff(*, job_id: str, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: Mapping[str, Any], timeout: float, store_path: Path,
        journal_path: Path, job_created: float) -> None:
    """Accept one explicit /3 handoff after the host decision and intent exist.

    Atomic publication is the admission point. This is not a new decision and
    cannot authorize research analysis. Caller owns the deployment lease.
    """
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or not 0 < timeout <= 86400):
        raise ValueError('preparation_handoff_timeout_invalid')
    if (not Path(store_path).is_absolute() or '..' in Path(store_path).parts
            or not Path(journal_path).is_absolute() or '..' in Path(journal_path).parts
            or type(job_created) not in {int, float} or not math.isfinite(job_created) or job_created <= 0):
        raise ValueError('preparation_handoff_store_invalid')
    admit_preparation_intent(job_id=job_id, checkpoint=checkpoint, binding=binding, scope=scope)
    from easyicu.webserver.preparation_recovery_snapshot import _read_signed
    intent = _read_signed(job_id + '.preparation-intent.json')
    body = {'schema_version': 'easyicu.preparation-handoff/1', 'job_id': job_id,
            'intent_sha256': canonical_sha256(intent), 'runtime_sha256': intent['runtime_sha256'],
            'study_context_id': binding.study_context_id, 'timeout_seconds': timeout,
            'command_schema': 'easyicu.durable-preparation/3', 'store_path': str(store_path),
            'journal_path': str(journal_path), 'job_created': job_created}
    encoded = canonical_json({**body, 'signature': _signature(body)}).encode()
    with AnchoredDirectory.open(_private_directory()) as directory:
        _publish_once(directory, job_id + '.preparation-handoff.json', encoded)
        directory.assert_still_selected()


def recover_approved_unstarted_handoff(*, job_id: str, job: Any, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: Mapping[str, Any], timeout: float,
        store_path: Path, journal_path: Path) -> dict[str, Any]:
    """Finish a missing /3 handoff only for the original unstarted host job.

    The caller must hold the deployment lease and admission lock. This creates
    no approval, cannot replay a consumed grant, and does not dispatch work.
    """
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.webserver.host_action_contracts import HostActionRequest, RawPreparationReviewDecision, decision_key, host_action_id
    from easyicu.webserver.host_action_jobs import _Ask, _raw_claim_occupied
    from easyicu.webserver.pi_copilot.service import get_pi_copilot_service
    from easyicu.webserver.preparation_recovery_snapshot import _read_signed
    from easyicu.webserver.raw_preparation_decisions import recover_host_review

    if scope.get('adapter_id') != 'eicu-demo-first-observed-single-hospital-adult/1':
        raise ValueError('preparation_handoff_recovery_adapter_unsupported')
    if (job is None or job.id != job_id or job.kind != 'raw-preparation-review' or job.is_cancel_requested()
            or not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or not 0 < timeout <= 86400
            or not Path(store_path).is_absolute() or '..' in Path(store_path).parts
            or not Path(journal_path).is_absolute() or '..' in Path(journal_path).parts):
        raise ValueError('preparation_handoff_recovery_invalid')
    state = job.snapshot()
    if (state.get('status') != 'running'
            and not (state.get('status') == 'failed' and state.get('recovered') and state.get('interrupted'))):
        raise ValueError('preparation_handoff_recovery_terminal')
    if checkpoint.state != 'approved':
        raise ValueError('preparation_handoff_recovery_approval_missing')
    pending_body = checkpoint.model_dump(mode='json', exclude={'checkpoint_sha256'})
    pending_body.update(state='pending', decision=None)
    pending = RawPreparationCheckpoint.model_validate({**pending_body, 'checkpoint_sha256': canonical_sha256(pending_body)})
    receipt, root = _review(checkpoint=checkpoint, binding=binding, scope=scope, job_id=job_id)
    with AnchoredDirectory.open(root) as directory:
        saved = RawPreparationCheckpoint.model_validate_json(directory.read_bytes(
            'raw_preparation_checkpoint.json', max_bytes=1024 * 1024))
        directory.assert_still_selected()
    if saved != checkpoint:
        raise ValueError('preparation_handoff_recovery_checkpoint_changed')
    if recover_host_review(pending=pending, binding=binding, scope=scope, job_id=job_id, run_dir=root) != checkpoint:
        raise ValueError('preparation_handoff_recovery_approval_changed')
    PlanReviewAuthority.model_validate(checkpoint.request.payload['plan_review_authority'])
    choice = RawPreparationReviewDecision(run_id=binding.run_id,
        scientific_configuration_sha256=receipt['scientific_configuration_sha256'],
        checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(dict(scope)),
        review_authority_sha256=checkpoint.request.authority_sha256, nonce=binding.nonce)
    action = HostActionRequest(session_id=binding.session_id, project_id=binding.project_id,
                               action_code='review_raw_preparation', decision=choice)
    key = decision_key(choice, review_decision='approved')
    if receipt.get('decision_key') != key:
        raise ValueError('preparation_handoff_recovery_action_changed')
    owner = get_pi_copilot_service()
    owner.host_action_session(binding.session_id, project_id=binding.project_id,
                              study_context_id=binding.study_context_id)
    ask = _Ask(request=action, study_context_id=binding.study_context_id, key=key,
               action_id=host_action_id(binding.session_id, action.action_code, key), conversations=owner)
    claimed = _raw_claim_occupied(ask, decision_key(choice, review_decision='rejected'))
    if claimed is None or not claimed.same_decision or claimed.job is not job:
        raise ValueError('preparation_handoff_recovery_job_changed')
    lease = root / ('.preparation-execution-' + job_id)
    if lease.exists() or lease.is_symlink():
        raise ValueError('preparation_handoff_lifecycle_present')
    with AnchoredDirectory.open(_private_directory()) as directory:
        for name in (binding.nonce + '.used', job_id + '.preparation-started',
                     job_id + '.preparation-snapshot.json', job_id + '.preparation-completed.json'):
            if not directory.is_absent(name):
                raise ValueError('preparation_handoff_execution_evidence_present')
        intent_exists = not directory.is_absent(job_id + '.preparation-intent.json')
        handoff_exists = not directory.is_absent(job_id + '.preparation-handoff.json')
        directory.assert_still_selected()
    if handoff_exists:
        return verify_unstarted_preparation_handoff(job_id=job_id, job=job, store_path=store_path, journal_path=journal_path)
    if not intent_exists:
        admit_preparation_intent(job_id=job_id, checkpoint=checkpoint, binding=binding, scope=scope)
    intent = _read_signed(job_id + '.preparation-intent.json')
    expected = {'schema_version': 'easyicu.preparation-intent/1', 'job_id': job_id,
        'binding': binding.model_dump(mode='json'), 'checkpoint': checkpoint.model_dump(mode='json'),
        'scope': json.loads(canonical_json(dict(scope))), 'root': str(root),
        'scientific_configuration_sha256': receipt['scientific_configuration_sha256'],
        'decision_key': key, 'runtime_sha256': runtime_digest()}
    if intent != expected:
        raise ValueError('preparation_handoff_recovery_intent_changed')
    body = {'schema_version': 'easyicu.preparation-handoff/1', 'job_id': job_id,
        'intent_sha256': canonical_sha256(intent), 'runtime_sha256': intent['runtime_sha256'],
        'study_context_id': binding.study_context_id, 'timeout_seconds': timeout,
        'command_schema': 'easyicu.durable-preparation/3', 'store_path': str(store_path),
        'journal_path': str(journal_path), 'job_created': job.created}
    with AnchoredDirectory.open(_private_directory()) as directory:
        _publish_once(directory, job_id + '.preparation-handoff.json',
                      canonical_json({**body, 'signature': _signature(body)}).encode())
        directory.assert_still_selected()
    return verify_unstarted_preparation_handoff(job_id=job_id, job=job, store_path=store_path, journal_path=journal_path)


def verify_unstarted_preparation_handoff(*, job_id: str, job: Any, store_path: Path,
                                         journal_path: Path) -> dict[str, Any]:
    """Verify a /3 handoff with no prior command or worker launch evidence.

    The caller must own the deployment lease and serialize admissions. This
    function cannot prove that a prior worker was cleaned up and consequently
    refuses any existing lifecycle, consumption, or snapshot receipt.
    """
    if (not isinstance(job_id, str) or not _JOB.fullmatch(job_id) or job is None
            or getattr(job, 'id', None) != job_id or getattr(job, 'kind', None) != 'raw-preparation-review'
            or job.is_cancel_requested()):
        raise ValueError('preparation_handoff_job_invalid')
    state = job.snapshot()
    if (state.get('status') not in {'running', 'failed'}
            or (state.get('status') == 'failed' and not (state.get('recovered') and state.get('interrupted')))):
        raise ValueError('preparation_handoff_job_terminal')
    from easyicu.webserver.preparation_recovery_snapshot import _read_signed
    handoff = _read_signed(job_id + '.preparation-handoff.json')
    intent = _read_signed(job_id + '.preparation-intent.json')
    if (set(handoff) != {'schema_version', 'job_id', 'intent_sha256', 'runtime_sha256',
                          'study_context_id', 'timeout_seconds', 'command_schema', 'store_path',
                          'journal_path', 'job_created'}
            or handoff.get('schema_version') != 'easyicu.preparation-handoff/1'
            or handoff.get('command_schema') != 'easyicu.durable-preparation/3'
            or handoff.get('store_path') != str(store_path)
            or handoff.get('journal_path') != str(journal_path)
            or handoff.get('job_created') != job.created
            or handoff.get('job_id') != job_id or handoff.get('intent_sha256') != canonical_sha256(intent)
            or handoff.get('runtime_sha256') != runtime_digest()
            or intent.get('runtime_sha256') != handoff.get('runtime_sha256')
            or intent.get('schema_version') != 'easyicu.preparation-intent/1'
            or intent.get('job_id') != job_id):
        raise ValueError('preparation_handoff_binding_invalid')
    timeout = handoff.get('timeout_seconds')
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or not 0 < timeout <= 86400):
        raise ValueError('preparation_handoff_timeout_invalid')
    checkpoint = RawPreparationCheckpoint.model_validate(intent.get('checkpoint'))
    binding = RawPreparationHostBinding.model_validate(intent.get('binding'))
    scope = intent.get('scope')
    if not isinstance(scope, dict) or binding.study_context_id != handoff.get('study_context_id'):
        raise ValueError('preparation_handoff_binding_invalid')
    with AnchoredDirectory.open(_private_directory()) as directory:
        for name in (binding.nonce + '.used', job_id + '.preparation-started',
                     job_id + '.preparation-snapshot.json', job_id + '.preparation-completed.json'):
            if not directory.is_absent(name):
                raise ValueError('preparation_handoff_execution_evidence_present')
        directory.assert_still_selected()
    saved_root = Path(str(intent.get('root') or ''))
    if not saved_root.is_absolute() or '..' in saved_root.parts or saved_root.is_symlink():
        raise ValueError('preparation_handoff_owner_changed')
    lease_dir = saved_root / ('.preparation-execution-' + job_id)
    if lease_dir.exists() or lease_dir.is_symlink():
        raise ValueError('preparation_handoff_lifecycle_present')
    if not _scope_matches(checkpoint, scope):
        raise ValueError('preparation_handoff_binding_invalid')
    receipt = dict(verify_host_review(checkpoint=checkpoint, scope=scope, **binding.model_dump(mode='python')))
    root = _owner(binding, receipt, job_id)
    if (str(root) != intent.get('root') or receipt.get('decision_key') != intent.get('decision_key')
            or receipt.get('scientific_configuration_sha256') != intent.get('scientific_configuration_sha256')):
        raise ValueError('preparation_handoff_owner_changed')
    _validate_current_scope(root=root, checkpoint=checkpoint, scope=scope)
    return {'schema_version': 'easyicu.durable-preparation/3', 'study_context_id': binding.study_context_id,
            'timeout_seconds': timeout, 'handoff_sha256': canonical_sha256(handoff)}


def verify_preparation_handoff_command(*, job_id: str, job: Any, payload: Mapping[str, Any],
                                       store_path: Path, journal_path: Path) -> None:
    """Recheck the immutable /3 command binding on every DBOS attempt."""
    from easyicu.webserver.preparation_recovery_snapshot import _read_signed
    handoff = _read_signed(job_id + '.preparation-handoff.json')
    intent = _read_signed(job_id + '.preparation-intent.json')
    expected = {'schema_version': 'easyicu.durable-preparation/3',
                'study_context_id': handoff.get('study_context_id'),
                'timeout_seconds': handoff.get('timeout_seconds'),
                'handoff_sha256': canonical_sha256(handoff)}
    actual = {key: value for key, value in dict(payload).items() if key != '_resume_attempt'}
    if (actual != expected or handoff.get('schema_version') != 'easyicu.preparation-handoff/1'
            or handoff.get('command_schema') != 'easyicu.durable-preparation/3'
            or handoff.get('store_path') != str(store_path)
            or handoff.get('journal_path') != str(journal_path)
            or handoff.get('job_created') != job.created
            or handoff.get('job_id') != job_id or intent.get('job_id') != job_id
            or handoff.get('intent_sha256') != canonical_sha256(intent)
            or handoff.get('runtime_sha256') != runtime_digest()
            or intent.get('runtime_sha256') != handoff.get('runtime_sha256')):
        raise ValueError('preparation_handoff_command_mismatch')


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
    first_observed = scope.get('adapter_id') == 'eicu-demo-first-observed-single-hospital-adult/1'
    if consumed and sealed_recovery and first_observed:
        from easyicu.webserver.raw_preparation_decisions import verify_consumed_host_review
        receipt = dict(verify_consumed_host_review(checkpoint=checkpoint, binding=binding, scope=scope))
        root = _owner(binding, receipt, job_id)
    elif consumed and sealed_recovery:
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
    if consumed and sealed_recovery and first_observed:
        # Never replay the raw read after consumption. Only a complete,
        # signed coordinate can survive this interruption window.
        from easyicu.webserver.first_observed_preparation import resolve_prepared_first_observed_coordinate
        return resolve_prepared_first_observed_coordinate(
            run_dir=root, checkpoint=checkpoint, binding=binding, scope=scope).path
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


def inspect_admitted_first_observed(*, job_id: str, job: Any, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: Mapping[str, Any], command: Mapping[str, Any],
        store_path: Path, journal_path: Path) -> dict[str, Any]:
    """Read an existing outbox's status after rechecking its signed owner.

    This cannot admit, dispatch, consume a decision, reset attempts, or read
    patient rows. An uncertain or changed binding is refused.
    """
    import hashlib
    import os
    import stat
    from easyicu import state_paths
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.webserver.host_action_contracts import RawPreparationReviewDecision, decision_key
    from easyicu.webserver.first_observed_preparation import _private_root

    if (scope.get('adapter_id') != 'eicu-demo-first-observed-single-hospital-adult/1'
            or not _JOB.fullmatch(job_id) or job is None or job.id != job_id
            or job.kind != 'raw-preparation-review' or checkpoint.state != 'approved'
            or not _scope_matches(checkpoint, scope)):
        raise ValueError('preparation_admitted_binding_invalid')
    root = state_paths.state_root() / 'private-preparation-decisions'
    with AnchoredDirectory.open(root) as directory:
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError('preparation_admitted_directory_invalid')
        with directory.open_regular('host.key') as handle:
            info = os.fstat(handle.fileno())
            if info.st_mode & 0o077 or info.st_uid != os.getuid():
                raise ValueError('preparation_admitted_key_invalid')
            key = handle.read(33)
        if len(key) != 32:
            raise ValueError('preparation_admitted_key_invalid')
        def signed(name: str) -> dict:
            value = json.loads(directory.read_bytes(name, max_bytes=4 * 1024 * 1024))
            if not isinstance(value, dict):
                raise ValueError('preparation_admitted_record_invalid')
            signature = value.pop('signature', None)
            digest = hmac.new(key, canonical_json(value).encode(), hashlib.sha256).hexdigest()
            if not isinstance(signature, str) or not hmac.compare_digest(signature, digest):
                raise ValueError('preparation_admitted_record_invalid')
            return value
        receipt = signed(binding.nonce + '.json')
        pending_claim = signed(binding.nonce + '.host-job-pending.json')
        claim = signed(binding.nonce + '.host-job.json')
        intent = signed(job_id + '.preparation-intent.json')
        handoff = signed(job_id + '.preparation-handoff.json')
        consumed = not directory.is_absent(binding.nonce + '.used')
        if consumed:
            marker = json.loads(directory.read_bytes(binding.nonce + '.used', max_bytes=64 * 1024))
            if marker != {'decision_sha256': canonical_sha256(receipt), 'nonce': binding.nonce}:
                raise ValueError('preparation_admitted_consumption_invalid')
        directory.assert_still_selected()
    pending_body = checkpoint.model_dump(mode='json', exclude={'checkpoint_sha256'})
    pending_body.update(state='pending', decision=None)
    pending_sha = canonical_sha256(pending_body)
    choice = RawPreparationReviewDecision(run_id=binding.run_id,
        scientific_configuration_sha256=receipt.get('scientific_configuration_sha256'),
        checkpoint_sha256=pending_sha, scope_sha256=canonical_sha256(dict(scope)),
        review_authority_sha256=checkpoint.request.authority_sha256, nonce=binding.nonce)
    decision = decision_key(choice, review_decision='approved')
    expected_receipt = {'schema_version': 'easyicu.raw-preparation-host-decision/1', 'actor': 'host_user',
        'decision': 'approved', **binding.model_dump(mode='python'), 'job_id': job_id,
        'scope_sha256': canonical_sha256(dict(scope)), 'pending_checkpoint_sha256': pending_sha,
        'decided_checkpoint_sha256': checkpoint.checkpoint_sha256,
        'request_authority_sha256': checkpoint.request.authority_sha256,
        'decided_review': checkpoint.decision.model_dump(mode='json'), 'decision_key': decision}
    if any(receipt.get(k) != v for k, v in expected_receipt.items()):
        raise ValueError('preparation_admitted_receipt_changed')
    expected_pending = {'schema_version': 'easyicu.raw-preparation-host-job-pending/1',
        'nonce': binding.nonce, 'study_context_id': binding.study_context_id,
        'project_id': binding.project_id, 'session_id': binding.session_id,
        'action_code': 'review_raw_preparation', 'decision_key': decision}
    expected_claim = {**expected_pending, 'schema_version': 'easyicu.raw-preparation-host-job/1',
                      'job_id': job_id, 'pending_sha256': canonical_sha256(expected_pending)}
    if pending_claim != expected_pending or claim != expected_claim:
        raise ValueError('preparation_admitted_claim_changed')
    owner_root = _owner(binding, receipt, job_id)
    with AnchoredDirectory.open(owner_root) as directory:
        saved = RawPreparationCheckpoint.model_validate_json(directory.read_bytes(
            'raw_preparation_checkpoint.json', max_bytes=1024 * 1024))
        directory.assert_still_selected()
    if saved != checkpoint:
        raise ValueError('preparation_admitted_checkpoint_changed')
    pending, _ = agent_pipeline_runs.raw_preparation_review_owner(binding.run_id, binding.study_context_id)
    expected_plan = PlanReviewAuthority.model_validate(checkpoint.request.payload['plan_review_authority'])
    authorities = [PlanReviewAuthority.model_validate(item.payload['plan_review_authority'])
        for item in pending.requests if isinstance(item.payload.get('plan_review_authority'), Mapping)]
    if not authorities or any(item != expected_plan for item in authorities):
        raise ValueError('preparation_admitted_plan_changed')
    _validate_current_scope(root=owner_root, checkpoint=checkpoint, scope=scope)
    runtime = runtime_digest()
    expected_intent = {'schema_version': 'easyicu.preparation-intent/1', 'job_id': job_id,
        'binding': binding.model_dump(mode='json'), 'checkpoint': checkpoint.model_dump(mode='json'),
        'scope': json.loads(canonical_json(dict(scope))), 'root': str(owner_root),
        'scientific_configuration_sha256': receipt['scientific_configuration_sha256'],
        'decision_key': decision, 'runtime_sha256': runtime}
    timeout = handoff.get('timeout_seconds')
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or not 0 < timeout <= 86400):
        raise ValueError('preparation_admitted_handoff_invalid')
    expected_handoff = {'schema_version': 'easyicu.preparation-handoff/1', 'job_id': job_id,
        'intent_sha256': canonical_sha256(intent), 'runtime_sha256': runtime,
        'study_context_id': binding.study_context_id, 'timeout_seconds': timeout,
        'command_schema': 'easyicu.durable-preparation/3', 'store_path': str(store_path),
        'journal_path': str(journal_path), 'job_created': job.created}
    expected_command = {'schema_version': 'easyicu.durable-preparation/3',
        'study_context_id': binding.study_context_id, 'timeout_seconds': timeout,
        'handoff_sha256': canonical_sha256(handoff)}
    payload = command.get('payload')
    if (intent != expected_intent or handoff != expected_handoff or command.get('id') != job_id
            or command.get('code_sha256') != runtime or command.get('status') not in {'active', 'terminal'}
            or not isinstance(payload, dict) or
            {k: v for k, v in payload.items() if k not in {'_resume_attempt', '_cancel_requested'}} != expected_command):
        raise ValueError('preparation_admitted_command_changed')
    state = job.snapshot()
    status = state.get('status')
    if status not in {'running', 'done', 'failed', 'cancelled'}:
        raise ValueError('preparation_admitted_job_invalid')
    if status == 'done':
        if not consumed:
            raise ValueError('preparation_admitted_result_unconsumed')
        with AnchoredDirectory.open(_private_root(create=False)) as directory:
            document = json.loads(directory.read_bytes(binding.nonce + '.json', max_bytes=64 * 1024))
            with directory.open_regular(binding.nonce + '.parquet') as handle:
                coordinate_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            directory.assert_still_selected()
        if not isinstance(document, dict) or set(document) != {'body', 'signature'}:
            raise ValueError('preparation_admitted_result_invalid')
        body, signature = document['body'], document['signature']
        if not isinstance(body, dict) or not isinstance(signature, str) or not hmac.compare_digest(
                signature, hmac.new(key, canonical_json(body).encode(), hashlib.sha256).hexdigest()):
            raise ValueError('preparation_admitted_result_invalid')
        version = body.get('schema_version')
        if version not in {'easyicu.first-observed-coordinate-attestation/1',
                           'easyicu.first-observed-coordinate-attestation/2'}:
            raise ValueError('preparation_admitted_result_invalid')
        expected_result = {'schema_version': version, 'scope_sha256': canonical_sha256(dict(scope)),
            'checkpoint_sha256': checkpoint.checkpoint_sha256,
            'binding_sha256': canonical_sha256(binding.model_dump(mode='json')),
            'coordinate_sha256': coordinate_digest, 'source_complete_six_columns': True,
            'analysis_authorized': False}
        if version.endswith('/2'):
            flow = body.get('private_source_flow')
            if not isinstance(flow, dict):
                raise ValueError('preparation_admitted_result_invalid')
            expected_result['private_source_flow'] = flow
        if body != expected_result:
            raise ValueError('preparation_admitted_result_invalid')
    result = 'complete' if status == 'done' else 'preparing' if status == 'running' else 'stopped'
    return {'job_id': job_id, 'preparation_status': result, 'job_status': status, 'analysis_authorized': False}
