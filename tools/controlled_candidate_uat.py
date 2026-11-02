#!/usr/bin/env python3
"""Host-owned development UAT. Default is an in-host preflight; --launch admits once.

When explicitly launched as a separate process, take the normal Web deployment
lease and start its durable owner only after proving no other command or private
preparation handoff could be resumed. Keep this process alive until its job is
terminal. No HTTP route or client-selected ledger is introduced.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time


def _close_owned_host():
    from easyicu.webserver import deployment_lease, durable_research
    while True:
        try:
            if durable_research.stop_service():
                deployment_lease.release_single_process_lease()
                return
            print(json.dumps({'shutdown': 'waiting_for_durable_cleanup'}), flush=True)
            time.sleep(1)
        except KeyboardInterrupt:
            # A second interrupt cannot abandon live work and release the OS lease.
            continue
        except Exception:
            print(json.dumps({'shutdown': 'cleanup_unverified'}), flush=True)
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                pass


def _start_isolated_durable_host():
    from easyicu.webserver import deployment_lease, durable_research, jobs, state_paths
    root = state_paths.state_root()
    deployment_lease.acquire_single_process_lease()
    try:
        path = root / 'research-commands.sqlite3'
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('original durable store missing or unsafe')
        connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            active = connection.execute("SELECT count(*) FROM commands WHERE status='active'").fetchone()[0]
        finally:
            connection.close()
        handoffs = root / 'private-preparation-decisions'
        if active or (handoffs.exists() and any(handoffs.glob('*.preparation-handoff.json'))):
            raise RuntimeError('other durable work exists; isolated host refused')
        jobs.MANAGER.enable_persistence(root / 'webserver-jobs.sqlite3')
        durable_research.start_service(root)
        return durable_research.current_service()
    except BaseException:
        _close_owned_host()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-job-id', required=True)
    parser.add_argument('--historical-worker-tree', required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--authorized-preprovider-successor', action='store_true')
    parser.add_argument('--failed-candidate-job-id')
    parser.add_argument('--expected-calls-sha256')
    parser.add_argument('--expected-failed-task-sha256')
    parser.add_argument('--expected-failure-sha256')
    args = parser.parse_args()
    from types import SimpleNamespace
    import re
    successor = None
    if args.authorized_preprovider_successor:
        if (not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.failed_candidate_job_id or "")
                or not re.fullmatch(r"[0-9a-f]{64}", args.expected_calls_sha256 or "")
                or not re.fullmatch(r"[0-9a-f]{64}", args.expected_failed_task_sha256 or "")
                or not re.fullmatch(r"[0-9a-f]{64}", args.expected_failure_sha256 or "")):
            parser.error('successor needs original job, failed job and exact calls digest')
        successor = SimpleNamespace(authorized=True, failed_job_id=args.failed_candidate_job_id,
                                    expected_calls_sha256=args.expected_calls_sha256,
                                    expected_task_sha256=args.expected_failed_task_sha256,
                                    expected_failure_sha256=args.expected_failure_sha256)
    elif args.failed_candidate_job_id or args.expected_calls_sha256 or args.expected_failed_task_sha256 or args.expected_failure_sha256:
        parser.error('successor parameters need explicit authorization flag')
    from easyicu.webserver import durable_research
    from easyicu.webserver.research_job_recovery import (
        prepare_controlled_candidate_from_original_job,
        submit_controlled_candidate_from_original_job,
    )
    service = durable_research.current_service()
    owns_service = False
    if service is None and args.launch:
        service = _start_isolated_durable_host()
        owns_service = True
    if service is None:
        parser.error('preflight must execute in a Web host process; no service was started')
    try:
        binding = prepare_controlled_candidate_from_original_job(service=service, original_job_id=args.original_job_id,
                                                                  historical_worker_tree=args.historical_worker_tree, _successor=successor)
        ledger = binding.task.ledger.snapshot()
        row = next(item for item in ledger['tasks'] if item['task_id'] == binding.task.task_id)
        result = {'preflight': 'passed', 'original_job_id': args.original_job_id,
                  'status': row['status'], 'provider_attempts': ledger['totals']['provider_attempts'],
                  'accounted_tokens': ledger['totals']['accounted_tokens'],
                  'accounted_estimated_cost_usd': ledger['totals']['accounted_estimated_cost_usd'],
                  'active_seconds': row['elapsed_seconds'], 'launched': False}
        if args.launch:
            receipt = submit_controlled_candidate_from_original_job(original_job_id=args.original_job_id,
                historical_worker_tree=args.historical_worker_tree, _successor=successor)
            result.update(launched=True, job_id=receipt.job_id)
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.launch and owns_service:
            while True:
                state = service.manager.get(receipt.job_id).snapshot()['status']
                if state != 'running':
                    print(json.dumps({'job_id': receipt.job_id, 'terminal_status': state}), flush=True)
                    break
                time.sleep(1)
    finally:
        if owns_service:
            _close_owned_host()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
