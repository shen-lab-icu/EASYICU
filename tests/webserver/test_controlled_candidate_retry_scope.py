import asyncio
import json

import pytest

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure, automatic_retry_limit, automatic_retry_scope,
    call_llm_with_structured_retry,
)


def test_scoped_parse_failure_is_one_call_and_default_is_preserved():
    client = ScriptedMockLLMClient(['not json', '{"ok": true}'])
    with automatic_retry_scope(disabled=True):
        with pytest.raises(StructuredResponseFailure):
            call_llm_with_structured_retry(client, [], parser=json.loads, max_retries=2)
    assert len(client.calls) == 1
    client = ScriptedMockLLMClient(['not json', '{"ok": true}'])
    assert call_llm_with_structured_retry(client, [], parser=json.loads, max_retries=2) == {'ok': True}
    assert len(client.calls) == 2


def test_scopes_propagate_to_async_tasks_without_cross_job_leakage():
    async def child():
        await asyncio.sleep(0)
        return automatic_retry_limit(3)
    async def task(disabled):
        with automatic_retry_scope(disabled=disabled):
            await asyncio.sleep(0)
            return await asyncio.create_task(child())
    async def run():
        return await asyncio.gather(task(True), task(False))
    assert asyncio.run(run()) == [0, 3]
    assert automatic_retry_limit(3) == 3


def test_nested_scope_cannot_widen_outer_policy():
    with automatic_retry_scope(disabled=True):
        with automatic_retry_scope(disabled=False):
            assert automatic_retry_limit(2) == 0


def test_guarded_worker_fails_once_and_keeps_cumulative_ledger(tmp_path, monkeypatch):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency as slots, provider_adapter

    monkeypatch.setattr(slots.state_paths, 'state_root', lambda: tmp_path / 'state')
    sent = []
    monkeypatch.setattr(metadata_provider_client, 'guarded_call', lambda *a, **k: sent.append(1) or {
        'ok': False, 'type': 'InternalServerError', 'status_code': 500})
    env = {'OPENAI_API_KEY': 'synthetic-secret', 'OPENAI_BASE_URL': 'http://127.0.0.1:1/v1',
           'OPENAI_MODEL': 'synthetic', 'EASYICU_DISABLE_PROVIDER_ENV_FILE': '1',
           'EASYICU_PROVIDER_DATA_HANDLING': 'user_accepted_external'}
    provider = {'provider': 'openai', 'external': True}
    inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=10)
    book = ProviderHardStopLedger(path=tmp_path / 'budget.json', task_ids=('task',),
        limits=provider_adapter.web_research_agent_hard_stop_limits('planner_canary'),
        batch_id='task', declaration_sha256='a' * 64)
    task = book.start_task('task')
    guarded = metadata_provider_client.MetadataProviderClient(
        inner=inner, task=task, provider=provider, environment=env, public=public, timeout=10)
    with automatic_retry_scope(disabled=True), pytest.raises(metadata_provider_client.MetadataProviderError):
        HardStopClient(guarded, role='planner', task=task).complete([LLMMessage(role='user', content='synthetic')], max_tokens=32)
    calls = book.snapshot()['tasks'][0]['calls']
    assert len(sent) == len(calls) == 1
    assert calls[0]['state'] == 'failed_usage_unknown'


def test_fresh_candidate_reuses_paused_task_and_never_auto_continues(tmp_path):
    from types import SimpleNamespace
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import provider_adapter, research_job_recovery, study_contexts
    from easyicu.webserver.jobs import Job

    study = {'id': 'synthetic', 'question': 'synthetic'}
    digest = study_contexts.scientific_configuration_sha256(study)
    book = ProviderHardStopLedger(path=tmp_path / 'budget.json', task_ids=('task',),
        limits=provider_adapter.web_research_agent_hard_stop_limits('planner_canary'),
        batch_id='task', declaration_sha256=digest)
    task = book.start_task('task')
    prior = ScriptedMockLLMClient(['prior'])
    HardStopClient(prior, role='planner', task=task).complete([LLMMessage(role='user', content='synthetic')], max_tokens=32)
    task.pause()
    old_calls = book.snapshot()['tasks'][0]['calls']
    request = SimpleNamespace(no_automatic_retries=True, allow_environment_resume=False, study_context=study)
    sent = []
    def runner(job):
        assert research_job_recovery.existing_provider_task(job) is task
        sent.append(1)
        assert automatic_retry_limit(3) == 0
        HardStopClient(ScriptedMockLLMClient(['second']), role='planner', task=task).complete(
            [LLMMessage(role='user', content='synthetic')], max_tokens=32)
        raise ConnectionError('synthetic')
    with pytest.raises(ConnectionError):
        research_job_recovery.run_controlled_candidate(runner=runner, request=request, job=Job('new', 'agent-run'), task=task)
    row = book.snapshot()['tasks'][0]
    assert sent == [1] and row['status'] == 'failed'
    assert row['calls'][:len(old_calls)] == old_calls
    assert len(row['calls']) == len(old_calls) + 1


def test_policy_survives_config_roundtrip_and_resume_lifecycle(tmp_path):
    from types import SimpleNamespace
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.instance_lifecycle import pipeline_instance_lifecycle

    config = PipelineConfig(workdir=tmp_path, no_automatic_retries=True)
    restored = PipelineConfig.from_recovery_payload(config.recovery_payload(), expected_digest=config.canonical_digest())
    assert restored.no_automatic_retries is True
    assert PipelineConfig(workdir=tmp_path).canonical_payload().get('no_automatic_retries') is None
    class Pipeline:
        _config = restored
        workdir = tmp_path
        _pending_human_review = {"pending": SimpleNamespace(run_id="synthetic")}
        @pipeline_instance_lifecycle('resume')
        def resume(self):
            return automatic_retry_limit(3)
    assert Pipeline().resume() == 0
    assert automatic_retry_limit(3) == 3


@pytest.mark.parametrize("changes", [
    {"no_automatic_retries": False}, {"allow_environment_resume": True},
    {"plan_change_request": object()}, {"execution_resume_source_run_id": "old"},
    {"development_resume_source_job_id": "old"}, {"plan_revision_source_run_id": "old"},
])
def test_controlled_candidate_rejects_unsafe_launch_before_task_access(changes):
    from types import SimpleNamespace
    from easyicu.webserver.research_job_recovery import run_controlled_candidate
    values = {"no_automatic_retries": True, "allow_environment_resume": False}
    values.update(changes)
    with pytest.raises(ValueError):
        run_controlled_candidate(runner=lambda job: pytest.fail("executed"),
                                 request=SimpleNamespace(**values), job=object(), task=object())


def test_review_failure_does_not_start_automatic_continuation(monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver import research_job_recovery
    from easyicu.webserver.jobs import Job
    monkeypatch.setattr(research_job_recovery, '_continuation', lambda *a: pytest.fail('automatic continuation'))
    attempts = []
    def operation(*, job):
        recovery = research_job_recovery.recovery_for_job(job)
        recovery.task = SimpleNamespace(ledger=SimpleNamespace(snapshot=lambda: {'tasks': []}), task_id='task')
        recovery.review_launch = {'synthetic': True}
        entry = SimpleNamespace(pipeline=SimpleNamespace(_config=SimpleNamespace(no_automatic_retries=True)),
                                provider_client=None, provider_hard_stop=None)
        research_job_recovery.bind_review_recovery(job, entry, {})
        attempts.append(1)
        raise ConnectionError('synthetic')
    with pytest.raises(ConnectionError):
        research_job_recovery.recover_review(operation)(job=Job('review', 'agent-review'))
    assert attempts == [1]


@pytest.mark.parametrize('row,digest_matches', [
    ({'task_id': 'other', 'status': 'paused', 'calls': []}, True),
    ({'task_id': 'task', 'status': 'running', 'calls': []}, True),
    ({'task_id': 'task', 'status': 'paused', 'calls': [{'state': 'in_progress'}]}, True),
    ({'task_id': 'task', 'status': 'paused', 'calls': []}, False),
])
def test_candidate_refuses_unverified_task_without_mutating(row, digest_matches):
    from types import SimpleNamespace
    from easyicu.webserver import research_job_recovery, study_contexts
    study = {'id': 'synthetic', 'question': 'synthetic'}
    digest = study_contexts.scientific_configuration_sha256(study) if digest_matches else 'different'
    task = SimpleNamespace(task_id='task', ledger=SimpleNamespace(snapshot=lambda: {
        'declaration_sha256': digest, 'tasks': [row]}), resume=lambda: pytest.fail('resumed'))
    request = SimpleNamespace(no_automatic_retries=True, allow_environment_resume=False, study_context=study)
    with pytest.raises(ValueError):
        research_job_recovery.run_controlled_candidate(runner=lambda job: pytest.fail('executed'),
                                                        request=request, job=object(), task=task)


def test_host_reference_preflight_fails_closed_on_worker_identity_or_slot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.runtime import host_execution_lease
    from easyicu.webserver import research_auto_resume, research_job_recovery, metadata_provider_concurrency, metadata_provider_worker
    attempt = 'a' * 32
    row = {'status': 'paused', 'calls': [{'attempt_id': attempt, 'state': 'completed'}]}
    seed = SimpleNamespace()
    service = SimpleNamespace(store=SimpleNamespace(get=lambda _: {'status': 'terminal', 'payload': {}}),
                              manager=SimpleNamespace(get=lambda _: SimpleNamespace(snapshot=lambda: {'status': 'failed'})))
    monkeypatch.setattr(research_auto_resume, '_authority', lambda *a: ({}, seed, tmp_path, row))
    monkeypatch.setattr(research_auto_resume, '_original_task', lambda _: 'task')
    monkeypatch.setattr(research_job_recovery, '_historical_worker_identity', lambda _: 'good')
    monkeypatch.setattr(host_execution_lease, 'read_host_execution_receipt', lambda _: {'script_sha256': 'bad'})
    monkeypatch.setattr(host_execution_lease, 'assert_host_execution_quiescent', lambda *a, **k: None)
    monkeypatch.setattr(metadata_provider_concurrency, 'get', lambda _: None)
    with pytest.raises(ValueError, match='worker_identity_changed'):
        research_job_recovery.prepare_controlled_candidate_from_original_job(service=service, original_job_id='old', historical_worker_tree='a' * 40)
    monkeypatch.setattr(host_execution_lease, 'read_host_execution_receipt', lambda _: {'script_sha256': 'good'})
    monkeypatch.setattr(metadata_provider_concurrency, 'get', lambda _: {'held': True})
    with pytest.raises(ValueError, match='slot_held'):
        research_job_recovery.prepare_controlled_candidate_from_original_job(service=service, original_job_id='old', historical_worker_tree='a' * 40)


def test_controlled_task_claim_is_atomic_and_once_only(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, ProviderHardStopLedgerError
    from easyicu.webserver import provider_adapter
    path = (tmp_path / 'budget.json').resolve()
    limits = provider_adapter.web_research_agent_hard_stop_limits('planner_canary')
    def open_book(resume):
        return ProviderHardStopLedger(path=path, task_ids=('task',), limits=limits,
            batch_id='task', declaration_sha256='a' * 64, resume_existing=resume, require_existing=resume)
    initial = open_book(False)
    initial.start_task('task').pause()
    tasks = [open_book(True).start_task('task') for _ in range(2)]
    barrier = Barrier(2)
    def claim(index):
        barrier.wait()
        try:
            tasks[index].claim_controlled_candidate(f'job-{index}')
            return True
        except ProviderHardStopLedgerError:
            return False
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, range(2)))
    assert sorted(results) == [False, True]
    row = initial.snapshot()['tasks'][0]
    assert row['controlled_candidate_job_id'] in {'job-0', 'job-1'}
    winner = tasks[results.index(True)]
    winner.pause()
    with pytest.raises(ProviderHardStopLedgerError):
        winner.claim_controlled_candidate('another')


@pytest.mark.parametrize('failure', ['missing_receipt', 'worker_active', 'budget_exhausted'])
def test_host_preflight_stops_before_candidate_on_missing_proof_or_exhausted_budget(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from easyicu.runtime import host_execution_lease
    from easyicu.webserver import research_auto_resume, research_job_recovery, metadata_provider_concurrency, metadata_provider_worker
    row = {'status': 'paused', 'calls': [{'attempt_id': 'a' * 32, 'state': 'completed'}]}
    service = SimpleNamespace(store=SimpleNamespace(get=lambda _: {'status': 'terminal', 'payload': {}}),
                              manager=SimpleNamespace(get=lambda _: SimpleNamespace(snapshot=lambda: {'status': 'failed'})))
    monkeypatch.setattr(research_auto_resume, '_authority', lambda *a: ({}, object(), tmp_path, row))
    monkeypatch.setattr(research_job_recovery, '_historical_worker_identity', lambda _: 'good')
    monkeypatch.setattr(metadata_provider_concurrency, 'get', lambda _: None)
    guard_path = tmp_path / '.runtime' / '.metadata-provider-execution' / ('a' * 32)
    guard_path.mkdir(parents=True)
    (guard_path / 'worker-cleanup.json').write_text(json.dumps({'nonce': 'n', 'worker': {'pid': 42}, 'exitcode': 0}))
    receipt = {'script_sha256': 'good', 'nonce': 'n', 'worker': {'pid': 42},
               'state': 'quiesced', 'reason': 'worker_completed'}
    monkeypatch.setattr(host_execution_lease, 'read_host_execution_receipt',
        lambda _: None if failure == 'missing_receipt' else receipt)
    def guard(*a, **k):
        if failure == 'worker_active':
            raise host_execution_lease.HostExecutionLeaseError('host_execution_active')
    monkeypatch.setattr(host_execution_lease, 'assert_host_execution_quiescent', guard)
    def original_task(seed):
        if failure == 'budget_exhausted':
            raise RuntimeError('synthetic budget exhausted')
        pytest.fail('must refuse before task')
    monkeypatch.setattr(research_auto_resume, '_original_task', original_task)
    with pytest.raises((ValueError, RuntimeError, host_execution_lease.HostExecutionLeaseError)):
        research_job_recovery.prepare_controlled_candidate_from_original_job(service=service, original_job_id='old', historical_worker_tree='a' * 40)


def test_standalone_host_refuses_existing_durable_work_before_start(tmp_path, monkeypatch):
    import importlib.util
    import sqlite3
    from pathlib import Path
    from easyicu.webserver import deployment_lease, durable_research, state_paths
    spec = importlib.util.spec_from_file_location('controlled_candidate_uat_test',
        Path(__file__).resolve().parents[2] / 'tools/controlled_candidate_uat.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    db = sqlite3.connect(tmp_path / 'research-commands.sqlite3')
    db.execute('create table commands (status text)')
    db.execute("insert into commands values ('active')")
    db.commit()
    db.close()
    monkeypatch.setattr(state_paths, 'state_root', lambda: tmp_path)
    monkeypatch.setattr(deployment_lease, 'acquire_single_process_lease', lambda: None)
    monkeypatch.setattr(deployment_lease, 'release_single_process_lease', lambda: None)
    monkeypatch.setattr(durable_research, 'stop_service', lambda: True)
    monkeypatch.setattr(durable_research, 'start_service', lambda *_: pytest.fail('started other work'))
    with pytest.raises(RuntimeError, match='other durable work'):
        module._start_isolated_durable_host()


def test_controlled_task_claim_competes_across_processes(tmp_path):
    import multiprocessing
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, ProviderHardStopLedgerError
    from easyicu.webserver import provider_adapter
    path = (tmp_path / 'process-budget.json').resolve()
    limits = provider_adapter.web_research_agent_hard_stop_limits('planner_canary')
    def open_book(resume):
        return ProviderHardStopLedger(path=path, task_ids=('task',), limits=limits,
            batch_id='task', declaration_sha256='a' * 64, resume_existing=resume, require_existing=resume)
    open_book(False).start_task('task').pause()
    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip('fork process race is unavailable on this platform')
    ctx = multiprocessing.get_context('fork')
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    def run(index):
        task = open_book(True).start_task('task')
        barrier.wait(timeout=10)
        try:
            task.claim_controlled_candidate(f'process-{index}')
            results.put(True)
        except ProviderHardStopLedgerError:
            results.put(False)
    children = [ctx.Process(target=run, args=(index,)) for index in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(15)
        assert child.exitcode == 0
    assert sorted([results.get(timeout=2), results.get(timeout=2)]) == [False, True]


def test_standalone_host_keeps_lease_until_service_stops(monkeypatch):
    import importlib.util
    from pathlib import Path
    from easyicu.webserver import deployment_lease, durable_research
    spec = importlib.util.spec_from_file_location('controlled_candidate_shutdown_test',
        Path(__file__).resolve().parents[2] / 'tools/controlled_candidate_uat.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    states = iter([False, True])
    monkeypatch.setattr(durable_research, 'stop_service', lambda: events.append('stop') or next(states))
    monkeypatch.setattr(deployment_lease, 'release_single_process_lease', lambda: events.append('release'))
    monkeypatch.setattr(module.time, 'sleep', lambda _: events.append('wait'))
    module._close_owned_host()
    assert events == ['stop', 'wait', 'stop', 'release']


def test_deferred_review_stage_keeps_original_canary_limits_and_distinct_profile():
    from easyicu.webserver import provider_adapter, research_launch_runtime
    from easyicu.research_agent.orchestration.profiles import get_submission_profile
    assert (provider_adapter.web_research_agent_hard_stop_limits('deferred_reviewed').canonical_payload()
            == provider_adapter.web_research_agent_hard_stop_limits('planner_canary').canonical_payload())
    canary = get_submission_profile(research_launch_runtime._submission_profile_ref(
        budget_mode='planner_canary', live_pubmed=False))
    deferred = get_submission_profile(research_launch_runtime._submission_profile_ref(
        budget_mode='deferred_reviewed', live_pubmed=False))
    assert canary.planner_only is True
    assert deferred.planner_only is False


def test_deferred_stage_is_one_claim_and_blocks_transport_before_review(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, ProviderHardStopLedgerError
    from easyicu.webserver import provider_adapter
    ledger = ProviderHardStopLedger(path=(tmp_path / 'budget.json').resolve(), task_ids=('task',),
        limits=provider_adapter.web_research_agent_hard_stop_limits('planner_canary'),
        batch_id='task', declaration_sha256='a' * 64)
    task = ledger.start_task('task')
    task.pause()
    task.claim_controlled_candidate('candidate')
    task.pause()
    task.claim_deferred_review_stage('stage', expected_candidate_job_id='candidate')
    with pytest.raises(ProviderHardStopLedgerError, match='cannot send'):
        ledger.reserve_transport_attempt(task_id='task', role='planner', model='synthetic',
            messages=[], max_tokens=32, prior_attempt_id=None)
    row = ledger.snapshot()['tasks'][0]
    assert row['calls'] == [] and row['deferred_stage_no_send'] is True
    task.pause()
    with pytest.raises(ProviderHardStopLedgerError):
        task.claim_deferred_review_stage('second', expected_candidate_job_id='candidate')


def test_deferred_stage_runner_reuses_task_and_never_sends_before_review(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, ProviderHardStopLedgerError
    from easyicu.webserver import provider_adapter, research_job_recovery
    from easyicu.webserver.jobs import Job
    ledger = ProviderHardStopLedger(path=(tmp_path / 'budget.json').resolve(), task_ids=('task',),
        limits=provider_adapter.web_research_agent_hard_stop_limits('planner_canary'),
        batch_id='task', declaration_sha256='a' * 64)
    task = ledger.start_task('task')
    task.pause()
    task.claim_controlled_candidate('candidate')
    task.pause()
    def runner(job):
        assert research_job_recovery.existing_provider_task(job) is task
        with pytest.raises(ProviderHardStopLedgerError, match='cannot send'):
            ledger.reserve_transport_attempt(task_id='task', role='acquisition', model='synthetic',
                messages=[], max_tokens=16, prior_attempt_id=None)
        task.pause()
        return {'gate': {'status': 'pending'}}
    result = research_job_recovery.run_deferred_review_stage(
        runner=runner, job=Job('stage', 'agent-run'), task=task, expected_candidate_job_id='candidate')
    assert result['gate']['status'] == 'pending'
    row = ledger.snapshot()['tasks'][0]
    assert row['status'] == 'paused' and row['deferred_stage_no_send'] is True and row['calls'] == []


def test_candidate_lineage_is_sealed_in_new_wrapper(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.webserver import provider_adapter, research_job_recovery
    from easyicu.webserver.jobs import Job
    ledger = ProviderHardStopLedger(path=(tmp_path / 'study' / 'run_old' / 'budget.json').resolve(), task_ids=('task',),
        limits=provider_adapter.web_research_agent_hard_stop_limits('planner_canary'),
        batch_id='task', declaration_sha256='a' * 64)
    task = ledger.start_task('task')
    task.pause()
    wrapper = tmp_path / 'study' / 'run_candidate'
    wrapper.mkdir(parents=True)
    monkeypatch.setattr(research_job_recovery, '_historical_worker_identity', lambda _: 'b' * 64)
    def runner(job):
        research_job_recovery.bind_provider_recovery(job, wrapper_dir=wrapper, task=task, client=object())
        task.pause()
        return {'gate': {'status': 'pending'}}
    request = SimpleNamespace(no_automatic_retries=True, allow_environment_resume=False, study_context={'id': 's'})
    monkeypatch.setattr(research_job_recovery.study_contexts, 'scientific_configuration_sha256', lambda _: 'a' * 64)
    research_job_recovery.run_controlled_candidate(runner=runner, request=request, job=Job('candidate', 'agent-run'),
        task=task, original_job_id='old', historical_worker_tree='c' * 40)
    payload = json.loads((wrapper / '.runtime' / 'controlled_candidate_lineage.json').read_bytes())
    digest = payload.pop('sha256')
    assert canonical_sha256(payload) == digest
    assert payload['candidate_job_id'] == 'candidate' and payload['task_id'] == 'task'
    assert payload['original_job_id'] == 'old' and payload['historical_worker_tree'] == 'c' * 40


def test_signed_preparation_handoff_admits_one_deferred_stage(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver import raw_preparation_decisions, preparation_recovery_snapshot, raw_preparation_durable, research_job_recovery
    binding = {'project_id': 'project', 'session_id': 'session', 'study_context_id': 'study',
               'run_id': 'candidate-run', 'nonce': 'a' * 64}
    intent = {'job_id': 'prep-job', 'binding': binding}
    monkeypatch.setattr(preparation_recovery_snapshot, '_read_signed', lambda _: intent)
    monkeypatch.setattr(raw_preparation_decisions, '_private_directory', lambda: tmp_path)
    monkeypatch.setattr(raw_preparation_decisions, '_signature', lambda _: 'synthetic')
    calls = []
    monkeypatch.setattr(research_job_recovery, 'submit_deferred_review_stage',
                        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(job_id='stage-job'))
    output = tmp_path / 'private-first-observed' / 'result'
    assert raw_preparation_durable.launch_deferred_review_after_first_observed(job_id='prep-job', output=output) == {
        'status': 'preparing_review', 'job_id': 'stage-job', 'analysis_authorized': False}
    with pytest.raises(FileExistsError):
        raw_preparation_durable.launch_deferred_review_after_first_observed(job_id='prep-job', output=output)
    assert calls == [{'preparation_run_id': 'candidate-run', 'study_context_id': 'study',
                      'project_id': 'project', 'session_id': 'session'}]
    assert raw_preparation_durable.launch_deferred_review_after_first_observed(
        job_id='prep-job', output=tmp_path / 'unrelated' / 'result') is None
