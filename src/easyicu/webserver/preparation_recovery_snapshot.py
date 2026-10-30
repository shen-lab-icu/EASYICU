"""Seal and recheck an existing private preparation snapshot.

This standalone primitive does not stage sources, run a producer, or dispatch
work. It only makes a host-signed claim about the exact staged bytes it sees;
it cannot prove source provenance, an atomic source snapshot, actual worker
reads, scientific validity, or permission to run analysis.
"""
from __future__ import annotations

import hmac
import json
from pathlib import Path
import re
from typing import Any, Mapping

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.runtime.host_execution_lease import verify_host_worker
from easyicu.webserver.durable_research_store import runtime_digest
from easyicu.webserver.multitable_guarded_worker import identity as worker_identity
from easyicu.webserver.preparation_recovery_authority import verify_recovery_authority
from easyicu.webserver.raw_case_snapshot import SnapshotFile, StagedRawSnapshot, verify_staged_source_files
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, _private_directory, _publish_once, _signature

_JOB = re.compile(r'[A-Za-z0-9_-]{1,128}')
_DIR = re.compile(r'reviewed-multitable-snapshot-[a-z0-9_]{8}')
_HASH = re.compile(r'[a-f0-9]{64}')


def _read_signed(name: str) -> dict:
    try:
        with AnchoredDirectory.open(_private_directory()) as directory:
            value = json.loads(directory.read_bytes(name, max_bytes=4 * 1024 * 1024))
            directory.assert_still_selected()
    except (AuthorityFilesystemError, OSError, ValueError, UnicodeError) as exc:
        raise ValueError('preparation_recovery_record_invalid') from exc
    if not isinstance(value, dict):
        raise ValueError('preparation_recovery_record_invalid')
    signature = value.pop('signature', None)
    if not isinstance(signature, str) or not hmac.compare_digest(signature, _signature(value)):
        raise ValueError('preparation_recovery_record_invalid')
    return value


def _coordinates(*, job_id: str, checkpoint: RawPreparationCheckpoint,
                 binding: RawPreparationHostBinding, scope: Mapping[str, Any],
                 host_work_root: Path, lease_dir: Path, lease_nonce: str) -> dict:
    if not isinstance(job_id, str) or not _JOB.fullmatch(job_id):
        raise ValueError('preparation_job_invalid')
    root = Path(host_work_root)
    if not root.is_absolute() or '..' in root.parts or root.is_symlink():
        raise ValueError('preparation_recovery_root_invalid')
    with AnchoredDirectory.open(root) as directory:
        directory.assert_still_selected()
    if Path(lease_dir) != root / ('.preparation-execution-' + job_id):
        raise ValueError('preparation_recovery_guard_invalid')
    receipt = verify_recovery_authority(checkpoint=checkpoint, binding=binding, scope=scope, job_id=job_id)
    worker = verify_host_worker(lease_dir, lease_nonce)
    if not isinstance(worker, str) or not _HASH.fullmatch(worker) or worker_identity() != worker:
        raise ValueError('preparation_recovery_guard_invalid')
    intent = _read_signed(job_id + '.preparation-intent.json')
    runtime = runtime_digest()
    expected = {'schema_version': 'easyicu.preparation-intent/1', 'job_id': job_id,
        'binding': binding.model_dump(mode='json'), 'checkpoint': checkpoint.model_dump(mode='json'),
        'scope': json.loads(canonical_json(dict(scope))), 'root': str(root),
        'scientific_configuration_sha256': receipt.get('scientific_configuration_sha256'),
        'decision_key': receipt.get('decision_key'), 'runtime_sha256': runtime}
    if intent != expected:
        raise ValueError('preparation_recovery_intent_mismatch')
    return {'job_id': job_id, 'scope_sha256': canonical_sha256(dict(scope)),
        'checkpoint_sha256': checkpoint.checkpoint_sha256,
        'binding_sha256': canonical_sha256(binding.model_dump(mode='json')),
        'intent_sha256': canonical_sha256(intent), 'runtime_sha256': runtime,
        'worker_identity': worker, 'host_work_root': str(root)}


def _check_snapshot(snapshot: StagedRawSnapshot, scope: Mapping[str, Any], path: Path) -> None:
    entries = scope.get('inputs')
    if not isinstance(entries, list) or not entries or not isinstance(scope.get('source_identity'), dict):
        raise ValueError('preparation_recovery_roster_invalid')
    paths = [entry.get('path') if isinstance(entry, dict) else None for entry in entries]
    if (snapshot.schema_version != 'easyicu.raw-source-stage/1'
            or any(not isinstance(name, str) or not name for name in paths)
            or len(paths) != len(set(paths))
            or sorted(paths) != sorted(item.relative_path for item in snapshot.files)
            or snapshot.source_identity_sha256 != canonical_sha256(scope['source_identity'])):
        raise ValueError('preparation_recovery_roster_invalid')
    verify_staged_source_files(path, snapshot)


def seal_preparation_snapshot(*, job_id: str, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: Mapping[str, Any], host_work_root: Path,
        lease_dir: Path, lease_nonce: str, snapshot_parent: Path, snapshot: StagedRawSnapshot) -> None:
    """Publish a signed snapshot receipt once, under the active guardian."""
    arguments = dict(job_id=job_id, checkpoint=checkpoint, binding=binding, scope=scope,
                     host_work_root=host_work_root, lease_dir=lease_dir, lease_nonce=lease_nonce)
    coordinates = _coordinates(**arguments)
    parent = Path(snapshot_parent)
    if parent.parent != Path(host_work_root) or not _DIR.fullmatch(parent.name) or parent.is_symlink():
        raise ValueError('preparation_recovery_snapshot_path_invalid')
    if not isinstance(snapshot, StagedRawSnapshot):
        raise ValueError('preparation_recovery_snapshot_invalid')
    _check_snapshot(snapshot, scope, parent / 'snapshot')
    if _coordinates(**arguments) != coordinates:
        raise ValueError('preparation_recovery_binding_changed')
    body = {'schema_version': 'easyicu.preparation-snapshot/1', 'coordinates': coordinates,
            'directory': parent.name, 'snapshot': snapshot.public_receipt()}
    encoded = canonical_json({**body, 'signature': _signature(body)}).encode()
    if len(encoded) > 4 * 1024 * 1024:
        raise ValueError('preparation_recovery_record_too_large')
    with AnchoredDirectory.open(_private_directory()) as directory:
        _publish_once(directory, job_id + '.preparation-snapshot.json', encoded)
        directory.assert_still_selected()


def verify_preparation_snapshot(*, job_id: str, checkpoint: RawPreparationCheckpoint,
        binding: RawPreparationHostBinding, scope: Mapping[str, Any], host_work_root: Path,
        lease_dir: Path, lease_nonce: str) -> tuple[Path, StagedRawSnapshot]:
    """Recheck the sealed bytes and coordinates without reading source files."""
    arguments = dict(job_id=job_id, checkpoint=checkpoint, binding=binding, scope=scope,
                     host_work_root=host_work_root, lease_dir=lease_dir, lease_nonce=lease_nonce)
    coordinates = _coordinates(**arguments)
    body = _read_signed(job_id + '.preparation-snapshot.json')
    name, stored = body.get('directory'), body.get('snapshot')
    if (set(body) != {'schema_version', 'coordinates', 'directory', 'snapshot'}
            or body.get('schema_version') != 'easyicu.preparation-snapshot/1'
            or body.get('coordinates') != coordinates
            or not isinstance(name, str) or not _DIR.fullmatch(name)
            or not isinstance(stored, dict)):
        raise ValueError('preparation_recovery_snapshot_binding_mismatch')
    try:
        snapshot = StagedRawSnapshot(source_identity_sha256=stored['source_identity_sha256'],
                                     files=tuple(SnapshotFile(**item) for item in stored['files']))
    except (KeyError, TypeError) as exc:
        raise ValueError('preparation_recovery_snapshot_invalid') from exc
    if snapshot.public_receipt() != stored:
        raise ValueError('preparation_recovery_snapshot_invalid')
    parent = Path(host_work_root) / name
    if parent.is_symlink():
        raise ValueError('preparation_recovery_snapshot_path_invalid')
    path = parent / 'snapshot'
    _check_snapshot(snapshot, scope, path)
    if _coordinates(**arguments) != coordinates:
        raise ValueError('preparation_recovery_binding_changed')
    return path, snapshot
