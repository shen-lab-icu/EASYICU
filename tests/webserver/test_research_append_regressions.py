"""Round-three synthetic regressions for completed-study additions."""
from types import SimpleNamespace

import pytest

from tests.support import pytest_state  # noqa: F401
from tests.support.result_append import completed_source
from easyicu.webserver import agent_pipeline_runs, agent_runs, study_contexts
from easyicu.webserver.research_append_run import approve_append_run, propose_append_run
from easyicu.webserver.research_append_source import read_json, seal_completed_source
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore


def source_at(tmp_path, *, study=None):
    wrapper, run, _, _, baseline = completed_source(tmp_path)
    return seal_completed_source(
        wrapper=wrapper, run_dir=run, study=study or {'id': 'study'},
        baseline=baseline.model_dump(mode='json'), prepared_package_binding={'synthetic': True},
    )


@pytest.fixture(scope='module')
def finished(tmp_path_factory):
    source = source_at(tmp_path_factory.mktemp('review2-finished'))
    before = {str(p.relative_to(source.wrapper)): p.read_bytes()
              for p in source.wrapper.rglob('*') if p.is_file()}
    child = propose_append_run(source=source, kind='subgroup', wrapper=source.wrapper.parent / 'run_append_finished',
                               study={'id': 'study'}, registered_source_changed=False)
    approve_append_run(child.wrapper, decision='approved', reviewer='synthetic-reviewer')
    return source, child, before


def test_result_click_keeps_the_conversation_bound_to_the_review_it_created(tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding
    from easyicu.webserver.pi_copilot.service import PiCopilotService
    from easyicu.webserver.research_append_actions import propose_for_conversation

    source = source_at(tmp_path)
    study = {'id': 'study', 'revision': 0}
    record = SimpleNamespace(project_id='project', agent_mode='other', binding=AuthorityBinding(study_context_id='study', study_revision=0, run_id='source'))
    service = object.__new__(PiCopilotService)
    service.replay_store = PiConversationReplayStore(tmp_path / "replay")
    service.project_store = SimpleNamespace(assert_matches=lambda *args: 'study')
    monkeypatch.setattr(service, '_scoped_record', lambda *args, **kwargs: record)
    saved = []
    monkeypatch.setattr(service, '_save_record', lambda record: saved.append(record.binding.model_dump()))
    monkeypatch.setattr(service, '_ensure_open', lambda record: None)
    monkeypatch.setattr(service, '_public_session', lambda *args, **kwargs: {})
    monkeypatch.setattr(service, '_research_run_row', lambda *args: {'project_dir': str(source.wrapper), 'study_id': 'study'})

    def latest(*args, **kwargs):
        rows = agent_runs.list_run_history(study_id='study', project_root=str(tmp_path))['runs']
        return rows[0]['run_id'] if rows else 'source'

    monkeypatch.setattr(service, '_latest_run_id', latest)
    monkeypatch.setattr(study_contexts, 'get_context', lambda _: study)
    assert service._binding_stale_details(record.binding, project_id='project')['stale'] is False
    response = propose_for_conversation(service, session_id='conversation', project_id='project', run_id='source',
                                        kind='subgroup', expected_source_sha256=source.digest)
    stale = service._binding_stale_details(record.binding, project_id='project')
    assert stale['stale'] is False, (response, stale)
    assert saved[-1]['run_id'] == response['run_id']
    item, = service.replay_store.attention.page(client_id="browser-1")["items"]
    assert (item["run_id"], item["kind"], item["reason_code"]) == (
        response["run_id"], "needs_review", "human_plan_review_required",
    )
    service.replay_store.attention.acknowledge(client_id="browser-1", sequences=[item["sequence"]])
    repeated = propose_for_conversation(service, session_id='conversation', project_id='project', run_id='source',
                                        kind='subgroup', expected_source_sha256=source.digest)
    assert repeated == response
    assert service.replay_store.attention.page(client_id="browser-1")["items"] == []


def test_completed_addition_merges_source_and_new_results_without_writing_the_source(finished):
    source, child, before = finished
    assert {str(p.relative_to(source.wrapper)): p.read_bytes()
            for p in source.wrapper.rglob('*') if p.is_file()} == before
    tables = read_json(child.wrapper / 'result_tables.json')['tables']
    producers = {table['produced_by_step'] for table in tables}
    assert {'adjusted_association', 'append_subgroup'} <= producers, producers
    original = next(t for t in tables if t['produced_by_step'] == 'adjusted_association')
    assert original['origin']['run_id'] == 'source'
    assert original['sha256'] == source.seal['products']['table:adjusted_association_estimates']['sha256']
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
    registered = EvidenceStore(child.run_dir).get(original['evidence_id'])
    assert registered.sha256 == original['sha256']
    assert verified_run_evidence_path(child.run_dir, registered) is not None


def test_completed_addition_still_offers_the_next_executable_study_addition(finished, monkeypatch):
    from easyicu.webserver.research_append_actions import append_options

    _, child, _ = finished
    service = SimpleNamespace(_research_run_row=lambda *args: {'project_dir': str(child.wrapper), 'study_id': 'study'})
    monkeypatch.setattr(study_contexts, 'get_context', lambda _: {'id': 'study'})
    options = append_options(service, project_id='project', run_id=child.checkpoint.run_id)
    available = {row['kind'] for row in options['options'] if row['available']}
    assert 'sensitivity' in available, available


def test_completed_append_step_has_a_complete_output_trace(finished):
    from easyicu.webserver.run_step_trace import project_step_detail

    _, child, _ = finished
    step = project_step_detail(child.run_dir, child.plan.steps[0].step_id)
    assert (step['state'], bool(step['outputs'])) == ('complete', True), step
    assert step['execution'] == {'mode': 'standard', 'executor': 'result_append_subgroup'}
    assert step['timing']['started_at'] and step['timing']['ended_at']
    manifest = read_json(child.run_dir / 'manifest.json')
    assert manifest['per_step_records'][0]['step_id'] == child.plan.steps[0].step_id
    assert manifest['per_step_records'][0]['evidence_ids']


def test_cancelled_append_job_does_not_start_its_model(tmp_path):
    source = source_at(tmp_path)
    child = propose_append_run(source=source, kind='subgroup', wrapper=source.wrapper.parent / 'run_append_cancelled',
                               study={'id': 'study'}, registered_source_changed=False)
    result = None
    try:
        result = agent_pipeline_runs.resume_research_pipeline(
            run_id=child.checkpoint.run_id, study_context_id='study', decision='approved',
            reviewer='synthetic-reviewer', note='', current_study_context={'id': 'study'},
            job=SimpleNamespace(id='cancelled-append', cancel_requested=True, emit=lambda _: None),
        )
    except Exception:
        pass
    assert not (child.run_dir / 'steps').exists(), (
        'A cancelled append still executed its new model', result,
    )


def test_source_verification_stop_is_projected_instead_of_remaining_approvable(tmp_path):
    source = source_at(tmp_path)
    child = propose_append_run(source=source, kind='subgroup', wrapper=source.wrapper.parent / 'run_append_stopped',
                               study={'id': 'study'}, registered_source_changed=False)
    manifest = source.run_dir / 'manifest.json'
    manifest.write_bytes(manifest.read_bytes() + b' ')
    with pytest.raises(ValueError):
        agent_pipeline_runs.resume_research_pipeline(
            run_id=child.checkpoint.run_id, study_context_id='study', decision='approved',
            reviewer='synthetic-reviewer', note='', current_study_context={'id': 'study'},
            job=SimpleNamespace(id='stopped-append', cancel_requested=False, emit=lambda _: None),
        )
    projection = read_json(child.wrapper / 'source_run_manifest.json')
    assert projection['plan_approval_allowed'] is False, projection['status']
    assert projection.get('failure_code'), projection
    assert projection['message_zh'].startswith('这次结果所用的数据或结果文件已被改动')
    assert projection['next_action_zh'] == '进入数据准备'
    assert agent_pipeline_runs.pending_review(child.checkpoint.run_id) is None
    from easyicu.webserver.stop_next_actions import project_stop_next_actions
    gate = read_json(child.wrapper / 'quality_gate.json')['gate']
    assert [action['kind'] for action in project_stop_next_actions(gate)] == ['reexport_data']


def test_earlier_export_is_attributed_to_the_sealed_source_not_the_new_registry_selection(tmp_path):
    old = {'id': 'study', 'data_source': {'label': 'Earlier synthetic export', 'database': 'synthetic',
                                       'path': str(tmp_path / 'earlier-export')}}
    source = source_at(tmp_path, study=old)
    current = {'id': 'study', 'data_source': {'label': 'Different synthetic export', 'database': 'synthetic-next',
                                           'path': str(tmp_path / 'different-export')}}
    child = propose_append_run(source=source, kind='subgroup', wrapper=source.wrapper.parent / 'run_append_new_source',
                               study=current, registered_source_changed=True)
    projection = read_json(child.wrapper / 'run_context.json')
    assert projection['source'] == {key: old['data_source'][key] for key in ('label', 'database')}, projection['source']


@pytest.mark.parametrize("kind", ["sensitivity", "subgroup"])
def test_next_addition_preserves_both_earlier_runs(finished, monkeypatch, kind):
    from easyicu.webserver.research_append_actions import append_options, propose_for_conversation
    from easyicu.webserver.research_append_run import append_run_for_id

    source, prior, _ = finished
    before = {str(p): p.read_bytes() for root in (source.wrapper, prior.wrapper)
              for p in root.rglob('*') if p.is_file()}
    service = SimpleNamespace(_research_run_row=lambda *args: {'project_dir': str(prior.wrapper), 'study_id': 'study'},
                              replay_store=PiConversationReplayStore(source.wrapper.parent / "replay"),
                              host_action_session=lambda *args, **kw: None, rebind_session=lambda *args, **kw: None)
    monkeypatch.setattr(study_contexts, 'get_context', lambda _: {'id': 'study'})
    offered = append_options(service, project_id='project', run_id=prior.checkpoint.run_id)
    response = propose_for_conversation(service, project_id='project', session_id='session',
        run_id=prior.checkpoint.run_id, expected_source_sha256=offered['source_run_sha256'], kind=kind)
    child = append_run_for_id(response['run_id'])
    assert child.record['predecessor']['run_id'] == prior.checkpoint.run_id
    approve_append_run(child.wrapper, decision='approved', reviewer='synthetic')
    assert {str(p): p.read_bytes() for root in (source.wrapper, prior.wrapper)
            for p in root.rglob('*') if p.is_file()} == before
    tables = read_json(child.wrapper / 'result_tables.json')['tables']
    assert {'adjusted_association', 'append_subgroup', child.plan.steps[0].step_id} <= {t['produced_by_step'] for t in tables}
    assert {t['origin']['run_id'] for t in tables if t.get('origin')} == {'source', prior.checkpoint.run_id}
    own = [t for t in tables if not t.get('origin')]
    assert len(own) == 1, 'earlier same-kind tables must not be projected again as new results'


def test_cancellation_between_steps_stops_the_remaining_roster(tmp_path):
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    source = source_at(tmp_path)
    child = propose_append_run(source=source, kind='exposure', wrapper=source.wrapper.parent / 'run_cancel_between',
                               study={'id': 'study'}, registered_source_changed=False)
    events = []
    job = SimpleNamespace(id='cancel-between', cancel_requested=False)
    def emit(event):
        events.append(event)
        if event.get('status') == 'completed':
            job.cancel_requested = True
    job.emit = emit
    with pytest.raises(ResearchPipelineRunError) as stop:
        agent_pipeline_runs.resume_research_pipeline(run_id=child.checkpoint.run_id, study_context_id='study',
            decision='approved', reviewer='synthetic', note='', current_study_context={'id': 'study'}, job=job)
    assert stop.value.code == 'research_pipeline_cancelled'
    assert {path.name for path in (child.run_dir / 'steps').iterdir()} == {child.plan.steps[0].step_id}
    assert len(events) == 2
    assert read_json(child.wrapper / 'source_run_manifest.json')['plan_approval_allowed'] is False


def test_refresh_with_changed_source_removes_approval(tmp_path):
    source = source_at(tmp_path)
    child = propose_append_run(source=source, kind='subgroup', wrapper=source.wrapper.parent / 'run_changed',
                               study={'id': 'study'}, registered_source_changed=False)
    path = source.run_dir / 'manifest.json'
    path.write_bytes(path.read_bytes() + b' ')
    assert agent_pipeline_runs.pending_review(child.checkpoint.run_id) is None
    assert read_json(child.wrapper / 'source_run_manifest.json')['plan_approval_allowed'] is False
