"""Fixed host-guarded worker for an already-authorized multi-table projection.

This gives the caller a process-tree cleanup receipt. It does not itself persist
an outbox or authorize replay after consuming a host decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from easyicu.runtime.host_execution_lease import (
    acknowledge_host_execution_parent, assert_host_execution_quiescent,
    claim_host_execution, complete_host_worker_cleanup, prepare_host_execution,
    request_host_execution_stop, verify_host_worker,
)

_MAX_MESSAGE = 2 * 1024 * 1024


def identity() -> str:
    from easyicu.webserver import reviewed_multitable_projection
    h = hashlib.sha256()
    for path in (Path(__file__), Path(reviewed_multitable_projection.__file__)):
        h.update(path.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


def _worker(lease_dir: Path, nonce: str, expected: str) -> int:
    if verify_host_worker(lease_dir, nonce) != expected or identity() != expected:
        return 125
    status = 1
    try:
        raw = sys.stdin.buffer.read(_MAX_MESSAGE + 1)
        if len(raw) > _MAX_MESSAGE:
            raise ValueError('preparation_request_too_large')
        request = json.loads(raw)
        if set(request) != {'checkpoint', 'binding', 'scope', 'host_work_root'}:
            raise ValueError('preparation_request_invalid')
        from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
        from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
        from easyicu.webserver.reviewed_multitable_projection import run_reviewed_multitable_projection
        result = run_reviewed_multitable_projection(
            checkpoint=RawPreparationCheckpoint.model_validate(request['checkpoint']),
            binding=RawPreparationHostBinding.model_validate(request['binding']),
            scope=request['scope'], host_work_root=Path(request['host_work_root']))
        payload = {'ok': True, 'output_name': result.parent.name}
        status = 0
    except BaseException:
        payload = {'ok': False, 'error': 'guarded_preparation_failed'}
    try:
        sys.stdout.buffer.write(json.dumps(payload, separators=(',', ':')).encode())
        sys.stdout.buffer.flush()
    except (BrokenPipeError, OSError):
        # The parent may have died and closed its capture pipe. The worker
        # must still acknowledge completion and remain for guardian cleanup.
        status = 1
    complete_host_worker_cleanup(lease_dir, nonce, exitcode=status)
    while True:
        time.sleep(60)


def _supervise(lease_dir: Path, nonce: str, expected: str, timeout: float) -> int:
    if identity() != expected or timeout <= 0:
        return 125
    lease = claim_host_execution(lease_dir, nonce)
    worker = None
    try:
        if lease.script_sha256 != expected:
            lease.mark_quiesced('worker_identity_mismatch', cleanup_proven=True)
            return 125
        if not lease.parent_alive() or lease.stop_requested():
            lease.mark_quiesced('cancelled_before_worker', cleanup_proven=True)
            return 125
        worker = subprocess.Popen([sys.executable, '-m', 'easyicu.webserver.multitable_guarded_worker', 'worker', str(lease_dir), nonce, expected, str(timeout)],
                                  start_new_session=os.name == 'posix')
        lease.record_worker(worker.pid)
        deadline = time.monotonic() + timeout
        while True:
            if worker.poll() is not None:
                lease.mark_unresolved('worker_exit_without_cleanup')
                return 125
            completion = lease.worker_completion_status()
            reason = (('worker_completed' if completion == 0 else 'worker_failed') if completion is not None
                      else 'parent_lost' if not lease.parent_alive()
                      else 'stop_requested' if lease.stop_requested()
                      else 'timeout' if time.monotonic() >= deadline else None)
            if reason is not None:
                stopped = lease.terminate_worker_tree()
                try:
                    worker.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    stopped = False
                lease.mark_quiesced(reason, cleanup_proven=True) if stopped else lease.mark_unresolved('worker_cleanup_unconfirmed')
                return completion if completion is not None and stopped else 125
            time.sleep(0.05)
    except BaseException:
        stopped = worker is None
        if worker is not None:
            try:
                stopped = lease.terminate_worker_tree()
                worker.wait(timeout=3)
            except BaseException:
                stopped = False
        try:
            lease.mark_quiesced('guardian_failed', cleanup_proven=True) if stopped else lease.mark_unresolved('guardian_cleanup_unconfirmed')
        except BaseException:
            pass
        return 125
    finally:
        lease.release()


def guarded_multitable_projection(*, lease_dir: Path, checkpoint: Any, binding: Any,
                                  scope: Mapping[str, Any], host_work_root: Path, timeout: float,
                                  on_prepared: Callable[[str], None] | None = None) -> Path:
    """Execute once under the original decision; never replay consumed authority."""
    if getattr(sys, 'frozen', False):
        raise RuntimeError('guarded_preparation_frozen_unsupported')
    request = {'checkpoint': checkpoint.model_dump(mode='json'), 'binding': binding.model_dump(mode='json'),
               'scope': dict(scope), 'host_work_root': str(host_work_root)}
    encoded = json.dumps(request, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > _MAX_MESSAGE:
        raise ValueError('preparation_request_too_large')
    lease = prepare_host_execution(lease_dir, script_sha256=identity())
    spawned = False
    process = None
    try:
        if on_prepared is not None:
            on_prepared(lease.nonce)
        lease.mark_spawning()
        process = subprocess.Popen([sys.executable, '-m', 'easyicu.webserver.multitable_guarded_worker', 'supervise', str(lease.path), lease.nonce,
                                    lease.script_sha256, str(timeout)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=os.name == 'posix')
        lease.bind_guardian(process.pid)
        spawned = True
        lease.release()
        try:
            output, _ = process.communicate(encoded, timeout=timeout + 15)
        except subprocess.TimeoutExpired:
            request_host_execution_stop(lease.path, lease.nonce)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError('guarded_preparation_supervisor_timeout') from None
        assert_host_execution_quiescent(lease.path, allow_missing=False, allow_current_parent=True, wait_timeout=2)
        if len(output) > _MAX_MESSAGE:
            raise ValueError('preparation_response_too_large')
        value = json.loads(output)
        name = value.get('output_name') if isinstance(value, dict) and value.get('ok') is True else None
        if not isinstance(name, str) or not name.startswith('reviewed-multitable-projection-') or Path(name).name != name:
            raise ValueError('guarded_preparation_failed')
        root = Path(host_work_root) / name
        from easyicu.webserver.reviewed_multitable_projection import verify_reviewed_multitable_projection
        verify_reviewed_multitable_projection(root=root, checkpoint_sha256=checkpoint.checkpoint_sha256,
                                             binding=binding, scope=scope)
        return root / 'output'
    finally:
        if not spawned:
            try:
                if process is None:
                    lease.abort()
            finally:
                lease.release()
        else:
            acknowledge_host_execution_parent(lease.path, lease.nonce)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('supervise', 'worker'))
    parser.add_argument('lease_dir', type=Path)
    parser.add_argument('nonce')
    parser.add_argument('expected')
    parser.add_argument('timeout', type=float)
    args = parser.parse_args()
    if args.action == 'worker':
        return _worker(args.lease_dir, args.nonce, args.expected)
    return _supervise(args.lease_dir, args.nonce, args.expected, args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
