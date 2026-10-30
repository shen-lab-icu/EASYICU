"""One consumed decision, one sealed snapshot, and resumable projections.

Only the original guarded job may stage the source. If it crashes before the
snapshot receipt is sealed, it cannot re-read the source automatically. A
later guarded attempt projects the same verified private snapshot only. The
caller must establish prior worker cleanup, owner, cancellation, and durable
dispatch; this module neither does so nor authorizes scientific analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.runtime.host_execution_lease import verify_host_worker
from easyicu.webserver.durable_research_store import runtime_digest
from easyicu.webserver.multitable_guarded_worker import identity as worker_identity
from easyicu.webserver.preparation_recovery_snapshot import (
    _coordinates, _read_signed, seal_preparation_snapshot, verify_preparation_snapshot,
)
from easyicu.webserver.raw_case_snapshot import stage_reviewed_source_files
from easyicu.webserver.raw_preparation_attestation import implementation_identity, verify_attestation
from easyicu.webserver.raw_preparation_decisions import (
    RawPreparationHostBinding, _private_directory, _publish_once, _signature,
    consume_host_review, verify_consumed_host_review, verify_host_review,
)
from easyicu.webserver.reviewed_multitable_projection import (
    _ParquetProjectionImplementation, _project_staged, _scope_shape, _validated_scope,
)

_JOB = re.compile(r'[A-Za-z0-9_-]{1,128}')
_RESULT = re.compile(r'reviewed-multitable-projection-[a-z0-9_]{8}')


def _exists(name: str) -> bool:
    with AnchoredDirectory.open(_private_directory()) as directory:
        exists = not directory.is_absent(name)
        directory.assert_still_selected()
        return exists


def _initial_authority(*, job_id: str, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: dict, host_work_root: Path,
        lease_dir: Path, lease_nonce: str) -> None:
    """Check the same intent and guardian before the first source read."""
    if not _JOB.fullmatch(job_id) or Path(lease_dir) != host_work_root / ('.preparation-execution-' + job_id):
        raise ValueError('preparation_recovery_guard_invalid')
    if (not host_work_root.is_absolute() or '..' in host_work_root.parts or host_work_root.is_symlink()
            or checkpoint.plan_sha256 != scope['plan_sha256']
            or canonical_sha256(checkpoint.request.payload.get('raw_preparation_scope')) != canonical_sha256(scope)):
        raise ValueError('preparation_recovery_binding_invalid')
    with AnchoredDirectory.open(host_work_root) as directory:
        directory.assert_still_selected()
    receipt = verify_host_review(checkpoint=checkpoint, scope=scope, **binding.model_dump(mode='python'))
    worker = verify_host_worker(lease_dir, lease_nonce)
    if worker != worker_identity():
        raise ValueError('preparation_recovery_guard_invalid')
    expected = {'schema_version': 'easyicu.preparation-intent/1', 'job_id': job_id,
        'binding': binding.model_dump(mode='json'), 'checkpoint': checkpoint.model_dump(mode='json'),
        'scope': json.loads(canonical_json(scope)), 'root': str(host_work_root),
        'scientific_configuration_sha256': receipt.get('scientific_configuration_sha256'),
        'decision_key': receipt.get('decision_key'), 'runtime_sha256': runtime_digest()}
    if receipt.get('job_id') != job_id or _read_signed(job_id + '.preparation-intent.json') != expected:
        raise ValueError('preparation_recovery_intent_mismatch')


def read_completed_multitable_projection(*, job_id: str, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: Mapping[str, Any], host_work_root: Path) -> Path:
    """Read a signed completed output without granting another execution."""
    if not isinstance(job_id, str) or not _JOB.fullmatch(job_id):
        raise ValueError('preparation_job_invalid')
    checked = _scope_shape(scope)
    receipt = verify_consumed_host_review(checkpoint=checkpoint, binding=binding, scope=checked)
    if receipt.get('job_id') != job_id:
        raise ValueError('preparation_job_binding_mismatch')
    body = _read_signed(job_id + '.preparation-completed.json')
    coordinates, name = body.get('coordinates'), body.get('directory')
    if (set(body) != {'schema_version', 'coordinates', 'directory', 'snapshot_sha256'}
            or body.get('schema_version') != 'easyicu.preparation-completed/1'
            or not isinstance(coordinates, dict) or not isinstance(name, str) or not _RESULT.fullmatch(name)
            or coordinates.get('job_id') != job_id
            or coordinates.get('host_work_root') != str(host_work_root)
            or coordinates.get('scope_sha256') != canonical_sha256(checked)
            or coordinates.get('checkpoint_sha256') != checkpoint.checkpoint_sha256
            or coordinates.get('binding_sha256') != canonical_sha256(binding.model_dump(mode='json'))):
        raise ValueError('preparation_completion_binding_invalid')
    root = Path(host_work_root) / name
    from easyicu.webserver import reviewed_multitable_projection
    verified = verify_attestation(root=root, scope=checked, checkpoint_sha256=checkpoint.checkpoint_sha256,
        binding=binding, adapter=_ParquetProjectionImplementation(), worker=Path(reviewed_multitable_projection.__file__))
    if verified['snapshot']['snapshot_sha256'] != body['snapshot_sha256']:
        raise ValueError('preparation_completion_snapshot_mismatch')
    return root / 'output'


def run_recoverable_multitable_projection(*, job_id: str, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: Mapping[str, Any], host_work_root: Path,
        lease_dir: Path, lease_nonce: str) -> Path:
    """Stage once after consumption, then resume only from sealed bytes."""
    checked = _scope_shape(scope)
    root = Path(host_work_root)
    if not isinstance(job_id, str) or not _JOB.fullmatch(job_id):
        raise ValueError('preparation_job_invalid')
    if _exists(job_id + '.preparation-completed.json'):
        return read_completed_multitable_projection(job_id=job_id, checkpoint=checkpoint, binding=binding,
                                                     scope=checked, host_work_root=root)
    args = dict(job_id=job_id, checkpoint=checkpoint, binding=binding, scope=checked,
                host_work_root=root, lease_dir=lease_dir, lease_nonce=lease_nonce)
    consumed = _exists(binding.nonce + '.used')
    if not consumed:
        if _exists(job_id + '.preparation-snapshot.json'):
            raise ValueError('preparation_recovery_unexpected_snapshot')
        _initial_authority(**args)
        consume_host_review(checkpoint=checkpoint, scope=checked, **binding.model_dump(mode='python'))
        # Source metadata and files are touched only after consuming the exact
        # decision. A crash before sealing deliberately prevents another read.
        _, source = _validated_scope(checked)
        parent = Path(tempfile.mkdtemp(prefix='reviewed-multitable-snapshot-', dir=root))
        snapshot = stage_reviewed_source_files(source_root=source, staging_root=parent / 'snapshot',
            relative_files=(entry['path'] for entry in checked['inputs']),
            source_identity_sha256=canonical_sha256(checked['source_identity']))
        with AnchoredDirectory.open(root) as directory:
            directory._directory.fsync()
            directory.assert_still_selected()
        seal_preparation_snapshot(**args, snapshot_parent=parent, snapshot=snapshot)
    snapshot_root, snapshot = verify_preparation_snapshot(**args)
    attempt = Path(tempfile.mkdtemp(prefix='reviewed-multitable-projection-', dir=root))
    (attempt / 'output').mkdir(mode=0o700)
    copied = stage_reviewed_source_files(source_root=snapshot_root, staging_root=attempt / 'snapshot',
        relative_files=(entry['path'] for entry in checked['inputs']),
        source_identity_sha256=canonical_sha256(checked['source_identity']))
    if copied.public_receipt() != snapshot.public_receipt():
        raise ValueError('preparation_recovery_snapshot_copy_changed')
    from easyicu.webserver import reviewed_multitable_projection
    implementation = implementation_identity(_ParquetProjectionImplementation(), Path(reviewed_multitable_projection.__file__))
    def recheck() -> None:
        _, current = verify_preparation_snapshot(**args)
        if current != snapshot:
            raise ValueError('preparation_recovery_snapshot_changed')
    result = _project_staged(root=attempt, snapshot=copied, checked=checked, checkpoint=checkpoint,
        binding=binding, implementation=implementation, before_publish=recheck)
    recheck()
    coordinates = _coordinates(**args)
    with AnchoredDirectory.open(attempt) as directory:
        directory._directory.fsync()
        directory.assert_still_selected()
    with AnchoredDirectory.open(root) as directory:
        directory._directory.fsync()
        directory.assert_still_selected()
    body = {'schema_version': 'easyicu.preparation-completed/1', 'coordinates': coordinates,
            'directory': attempt.name, 'snapshot_sha256': snapshot.public_receipt()['snapshot_sha256']}
    with AnchoredDirectory.open(_private_directory()) as directory:
        _publish_once(directory, job_id + '.preparation-completed.json',
                      canonical_json({**body, 'signature': _signature(body)}).encode())
        directory.assert_still_selected()
    if read_completed_multitable_projection(job_id=job_id, checkpoint=checkpoint, binding=binding,
                                             scope=checked, host_work_root=root) != result:
        raise ValueError('preparation_completion_output_mismatch')
    return result
