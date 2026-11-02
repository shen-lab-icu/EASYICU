from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, ProviderHardStopLedgerError
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver.provider_adapter import web_research_agent_hard_stop_limits


def setup_book(tmp_path):
    limits = web_research_agent_hard_stop_limits('planner_canary')
    path = (tmp_path / 'ledger.json').resolve()
    def book(resume):
        return ProviderHardStopLedger(path=path, task_ids=('task',), limits=limits, batch_id='task',
            declaration_sha256='a' * 64, resume_existing=resume, require_existing=resume)
    ledger = book(False)
    task = ledger.start_task('task')
    for _ in range(3):
        attempt = ledger.reserve_transport_attempt(task_id='task', role='planner', model='synthetic', messages=[], max_tokens=16, prior_attempt_id=None)
        ledger.finish_transport_attempt(task_id='task', attempt_id=attempt, usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}, error_type=None)
    task.pause()
    task.claim_controlled_candidate('failed')
    task.finish(error='synthetic_pre_provider_failure')
    return book, ledger


def test_atomic_successor_preserves_history_and_only_one_winner(tmp_path):
    book, ledger = setup_book(tmp_path)
    old = ledger.snapshot()
    oldrow = old['tasks'][0]
    digest = canonical_sha256(oldrow['calls'])
    books = [book(True), book(True)]
    barrier = Barrier(2)
    def claim(i):
        barrier.wait()
        try:
            books[i].claim_preprovider_successor('task', failed_job_id='failed', successor_job_id=f'new-{i}', expected_calls_sha256=digest, expected_task_sha256=canonical_sha256(oldrow))
            return True
        except ProviderHardStopLedgerError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, range(2))) == [False, True]
    new = ledger.snapshot()
    row = new['tasks'][0]
    assert row['status'] == 'running' and row['calls'] == oldrow['calls']
    assert row['elapsed_seconds'] == oldrow['elapsed_seconds']
    assert row['terminal_attempts'][-1]['controlled_candidate_job_id'] == 'failed'
    assert row['terminal_attempts'][-1]['error'] == oldrow['error']
    assert new['limits'] == old['limits'] and new['declaration_sha256'] == old['declaration_sha256']
    with pytest.raises(ProviderHardStopLedgerError):
        ledger.claim_preprovider_successor('task', failed_job_id='failed', successor_job_id='again', expected_calls_sha256=digest, expected_task_sha256=canonical_sha256(oldrow))


@pytest.mark.parametrize('failed,digest', [('other', '0' * 64), ('failed', '0' * 64)])
def test_successor_rejects_wrong_binding_without_mutation(tmp_path, failed, digest):
    _, ledger = setup_book(tmp_path)
    old = ledger.snapshot()
    with pytest.raises(ProviderHardStopLedgerError):
        ledger.claim_preprovider_successor('task', failed_job_id=failed, successor_job_id='new', expected_calls_sha256=digest, expected_task_sha256=canonical_sha256(old['tasks'][0]))
    assert ledger.snapshot() == old


def test_successor_runner_fails_once_and_does_not_retry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop
    from easyicu.webserver import research_job_recovery
    from easyicu.webserver.jobs import Job
    _, ledger = setup_book(tmp_path)
    task = TaskProviderHardStop(ledger, 'task')
    digest = canonical_sha256(ledger.snapshot()['tasks'][0]['calls'])
    monkeypatch.setattr(research_job_recovery.study_contexts, 'scientific_configuration_sha256', lambda _: 'a' * 64)
    monkeypatch.setattr(research_job_recovery, '_historical_worker_identity', lambda _: 'b' * 64)
    monkeypatch.setattr(research_job_recovery, '_continuation', lambda *a: pytest.fail('retry'))
    called = []
    def fail(job):
        called.append(job.id)
        raise RuntimeError('synthetic failure')
    with pytest.raises(RuntimeError, match='synthetic failure'):
        research_job_recovery.run_controlled_candidate(runner=fail, job=Job('successor', 'agent-run'),
            request=SimpleNamespace(no_automatic_retries=True, allow_environment_resume=False, study_context={}),
            task=task, original_job_id='old', historical_worker_tree='a' * 40,
            _successor=SimpleNamespace(failed_job_id='failed', expected_calls_sha256=digest, expected_task_sha256=canonical_sha256(ledger.snapshot()['tasks'][0]), expected_failure_sha256='a'*64))
    assert called == ['successor']
    row = ledger.snapshot()['tasks'][0]
    assert row['status'] == 'failed' and row['preprovider_successor_job_id'] == 'successor'
    assert len(row['calls']) == 3 and row['terminal_attempts'][-1]['controlled_candidate_job_id'] == 'failed'


def test_host_preflight_requires_original_failed_claim_and_quiescent_guard(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from easyicu.runtime import host_execution_lease
    from easyicu.webserver import research_auto_resume, research_job_recovery, metadata_provider_concurrency
    attempt = 'a' * 32
    row = {'status': 'failed', 'controlled_candidate_job_id': '4a1a2bcf8057', 'calls': [{'attempt_id': attempt, 'state': 'completed'} for _ in range(3)]}
    seed = SimpleNamespace(study={'id': 'study'}, scientific_configuration_sha256='b' * 64, hard_stop_ledger_path='/synthetic', hard_stop_task_id='task', hard_stop_declaration_sha256='b'*64)
    descriptor = {'study_context_id': 'study', 'scientific_configuration_sha256': 'b' * 64, 'provider_binding': {'hard_stop_ledger_path': '/synthetic', 'hard_stop_task_id': 'task', 'hard_stop_declaration_sha256': 'b'*64}}
    manager = SimpleNamespace(get=lambda job: SimpleNamespace(snapshot=lambda: {'status': 'failed'}))
    store = SimpleNamespace(get=lambda job: {'status': 'terminal', 'payload': descriptor})
    service = SimpleNamespace(manager=manager, store=store)
    guard = tmp_path / '.runtime' / '.metadata-provider-execution' / attempt
    guard.mkdir(parents=True)
    (guard / 'worker-cleanup.json').write_text(json.dumps({'nonce': 'n', 'worker': 'w', 'exitcode': 0}))
    monkeypatch.setattr(research_auto_resume, '_authority', lambda *a, **k: ({}, seed, tmp_path, row))
    monkeypatch.setattr(research_auto_resume, '_original_task', lambda *a, **k: 'task')
    monkeypatch.setattr(research_job_recovery, '_historical_worker_identity', lambda _: 'digest')
    monkeypatch.setattr(metadata_provider_concurrency, 'get', lambda _: None)
    monkeypatch.setattr(host_execution_lease, 'read_host_execution_receipt', lambda _: {'script_sha256': 'digest', 'nonce': 'n', 'worker': 'w', 'state': 'quiesced', 'reason': 'worker_completed'})
    monkeypatch.setattr(host_execution_lease, 'assert_host_execution_quiescent', lambda *a, **k: None)
    failed_wrapper = tmp_path / 'run_4a1a2bcf8057'
    (failed_wrapper / '.runtime').mkdir(parents=True)
    (failed_wrapper / 'diagnostics').mkdir()
    lineage = {'schema_version': 'easyicu.controlled-candidate-lineage/1', 'original_job_id': '733c01fc120e', 'candidate_job_id': '4a1a2bcf8057', 'historical_worker_tree': 'a'*40, 'task_id': 'task', 'declaration_sha256': 'b'*64}
    lineage['sha256'] = canonical_sha256(lineage)
    (failed_wrapper / '.runtime' / 'controlled_candidate_lineage.json').write_text(json.dumps(lineage))
    failure = json.dumps({'status':'failed', 'traceback_frames': [{'file':'research_job_recovery.py','function':'bind_provider_recovery'}, {'file':'raw_preparation_decisions.py','function':'_publish_once'}, {'file':'filesystem.py','function':'publish_temporary_no_clobber'}]}).encode()
    (failed_wrapper / 'diagnostics' / 'research_pipeline_failure.json').write_bytes(failure)
    import hashlib
    monkeypatch.setattr(research_auto_resume, '_owned_wrapper', lambda *a: failed_wrapper)
    successor = SimpleNamespace(authorized=True, failed_job_id='4a1a2bcf8057', expected_calls_sha256=canonical_sha256(row['calls']), expected_task_sha256=canonical_sha256(row), expected_failure_sha256=hashlib.sha256(failure).hexdigest())
    binding = research_job_recovery.prepare_controlled_candidate_from_original_job(service=service,
        original_job_id='733c01fc120e', historical_worker_tree='a'*40, _successor=successor)
    assert binding.task == 'task' and binding.successor is successor
    (failed_wrapper / 'diagnostics' / 'research_pipeline_failure.json').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='failure_evidence_changed'):
        research_job_recovery.prepare_controlled_candidate_from_original_job(service=service,
            original_job_id='733c01fc120e', historical_worker_tree='a'*40, _successor=successor)
    (failed_wrapper / 'diagnostics' / 'research_pipeline_failure.json').write_bytes(failure)
    successor.authorized = False
    with pytest.raises(ValueError, match='successor_not_authorized'):
        research_job_recovery.prepare_controlled_candidate_from_original_job(service=service,
            original_job_id='733c01fc120e', historical_worker_tree='a'*40, _successor=successor)
    successor.authorized = True
    monkeypatch.setattr(metadata_provider_concurrency, 'get', lambda _: {'held': True})
    with pytest.raises(ValueError, match='slot_held'):
        research_job_recovery.prepare_controlled_candidate_from_original_job(service=service,
            original_job_id='733c01fc120e', historical_worker_tree='a'*40, _successor=successor)


def test_successor_rejects_in_progress_calls_without_reopening(tmp_path):
    book, ledger = setup_book(tmp_path)
    # Synthetic corruption models an unclean worker; do not mutate actual UAT state.
    with ledger._durable_transaction():
        row = ledger._task_locked('task')
        row['calls'][0]['state'] = 'in_progress'
        ledger._persist_locked()
    before = ledger.snapshot()
    with pytest.raises(ProviderHardStopLedgerError):
        book(True).claim_preprovider_successor('task', failed_job_id='failed', successor_job_id='new',
            expected_calls_sha256=canonical_sha256(before['tasks'][0]['calls']), expected_task_sha256=canonical_sha256(before['tasks'][0]))
    assert ledger.snapshot() == before


def test_successor_rejects_exhausted_wall_clock_without_reopening(tmp_path):
    book, ledger = setup_book(tmp_path)
    with ledger._durable_transaction():
        ledger._task_locked('task')['elapsed_seconds'] = ledger.limits.max_wall_clock_seconds_per_task
        ledger._persist_locked()
    before = ledger.snapshot()
    with pytest.raises(ProviderHardStopLedgerError, match='budget unavailable'):
        book(True).claim_preprovider_successor('task', failed_job_id='failed', successor_job_id='new',
            expected_calls_sha256=canonical_sha256(before['tasks'][0]['calls']), expected_task_sha256=canonical_sha256(before['tasks'][0]))
    assert ledger.snapshot() == before


def test_successor_preserves_three_accounted_synthetic_calls(tmp_path):
    _, ledger = setup_book(tmp_path)
    # Build a separate ledger using actual accounting APIs; no client or network.
    path = (tmp_path / 'with-calls.json').resolve()
    source = ProviderHardStopLedger(path=path, task_ids=('task',),
        limits=web_research_agent_hard_stop_limits('planner_canary'), batch_id='task', declaration_sha256='a'*64)
    task = source.start_task('task')
    for _ in range(3):
        attempt = source.reserve_transport_attempt(task_id='task', role='planner', model='synthetic',
            messages=[], max_tokens=16, prior_attempt_id=None)
        source.finish_transport_attempt(task_id='task', attempt_id=attempt,
            usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}, error_type=None)
    task.pause()
    task.claim_controlled_candidate('failed')
    task.finish(error='synthetic')
    old = source.snapshot()
    calls = old['tasks'][0]['calls']
    source.claim_preprovider_successor('task', failed_job_id='failed', successor_job_id='new',
        expected_calls_sha256=canonical_sha256(calls), expected_task_sha256=canonical_sha256(old['tasks'][0]))
    new = source.snapshot()
    assert len(calls) == 3 and new['tasks'][0]['calls'] == calls
    for key in ('provider_attempts', 'accounted_tokens', 'accounted_estimated_cost_usd'):
        assert new['totals'][key] == old['totals'][key]


def _process_claim(path, ready, start, results, index, calls_digest, task_digest):
    ledger = ProviderHardStopLedger(path=path, task_ids=('task',),
        limits=web_research_agent_hard_stop_limits('planner_canary'), batch_id='task',
        declaration_sha256='a'*64, resume_existing=True, require_existing=True)
    ready.put(index)
    start.wait()
    try:
        ledger.claim_preprovider_successor('task', failed_job_id='failed', successor_job_id=f'proc-{index}',
            expected_calls_sha256=calls_digest, expected_task_sha256=task_digest)
        results.put(True)
    except ProviderHardStopLedgerError:
        results.put(False)


def test_two_processes_have_exactly_one_successor(tmp_path):
    import multiprocessing
    _, ledger = setup_book(tmp_path)
    row = ledger.snapshot()['tasks'][0]
    ctx = multiprocessing.get_context('spawn')
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=_process_claim, args=(ledger.path, ready, start, results, i,
        canonical_sha256(row['calls']), canonical_sha256(row))) for i in range(2)]
    for worker in workers:
        worker.start()
    for _ in workers:
        ready.get(timeout=10)
    start.set()
    assert sorted(results.get(timeout=10) for _ in workers) == [False, True]
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0


def test_successor_rejects_task_drift_even_when_calls_match(tmp_path):
    _, ledger = setup_book(tmp_path)
    before = ledger.snapshot()
    with pytest.raises(ProviderHardStopLedgerError, match='claim refused'):
        ledger.claim_preprovider_successor('task', failed_job_id='failed', successor_job_id='new',
            expected_calls_sha256=canonical_sha256(before['tasks'][0]['calls']), expected_task_sha256='0'*64)
    assert ledger.snapshot() == before
