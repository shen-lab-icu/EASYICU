"""Source candidates are evidence-derived, review-only, and scope-bound."""
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyicu.databases.research_plan_requirements import _constraints
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import non_executable_proposal_view as view
from easyicu.webserver import source_partition_candidates as discovery
from tests.support.signed_case_duration import signed_case_duration_case
from tests.webserver.test_non_executable_proposal_view import Service, setup
from tests.webserver.copilot.pi_copilot_static_fixtures import _load_guided_pi_module_harness  # noqa: F401


def test_only_recorded_concepts_form_candidates_not_guessed_outcome(tmp_path, monkeypatch):
    context, *_ = signed_case_duration_case()
    variable = next(item for item in context.variables if item.name == 'lact').model_copy(update={'source_concept': 'lact'})
    identifier = next(item for item in context.variables if item.name == 'CaseID')
    derived_identifier = identifier.model_copy(update={'name': 'patient_stay_id'})
    non_id = identifier.model_copy(update={'name': 'LaboratoryValue', 'role': 'other'})
    context = context.model_copy(update={'variables': [variable, identifier, derived_identifier, non_id],
        'cohort': context.cohort.model_copy(update={'id_columns': ['CaseID', 'patient_stay_id']})})
    identity = SourceIdentity.from_record(_constraints(context)['source_identity'])
    study = {'data_source': {'database': 'sic', 'path': '/registered'}}
    monkeypatch.setattr(discovery.sources, 'registered_source_identity', lambda _: identity)
    EvidenceStore(tmp_path).register_text(kind='log', description='context',
        text=context.model_dump_json(), filename='context.json', evidence_id='research_context')
    captured = {}
    def fake_discover(*, identity, requested_columns):
        captured.update(requested_columns)
        return {'candidate_sha256': 'a' * 64, 'files': [
                    {'table': 'laboratory', 'path': 'laboratory/1.parquet',
                     'columns': requested_columns['laboratory']}], 'full_source_roster_verified': False,
                'files_content_verified': False, 'analysis_authorized': False}
    monkeypatch.setattr(discovery, 'discover_source_partition_candidates', fake_discover)
    candidate = discovery.candidate_from_bound_research_context(study=study, run_dir=tmp_path)
    assert captured == {'laboratory': ['CaseID', 'LaboratoryID', 'LaboratoryValue', 'Offset']}
    assert candidate['features'][0]['concept'] == 'lact'
    assert candidate['identifier_candidates'] == [{'variable': 'CaseID', 'table': 'laboratory',
        'column': 'CaseID', 'actual_records_verified': False, 'join_or_derived_identifier_verified': False}]
    assert candidate['unmapped_variables'] == ['patient_stay_id', 'LaboratoryValue']
    assert candidate['outcome_and_cohort_dependencies_complete'] is False
    assert candidate['analysis_authorized'] is False
    monkeypatch.setattr(discovery.sources, 'registered_source_identity', lambda _: SourceIdentity(database='sic', registration_id='other'))
    with pytest.raises(ValueError, match='context_source_mismatch'):
        discovery.candidate_from_bound_research_context(study=study, run_dir=tmp_path)


@pytest.mark.parametrize('case', ['not_cohort_id', 'not_id_role', 'missing_schema_column'])
def test_identifier_candidate_requires_declared_id_and_selected_schema(case, tmp_path, monkeypatch):
    context, *_ = signed_case_duration_case()
    lact = next(item for item in context.variables if item.name == 'lact').model_copy(update={'source_concept': 'lact'})
    identifier = next(item for item in context.variables if item.name == 'CaseID')
    if case == 'not_id_role':
        identifier = identifier.model_copy(update={'role': 'other'})
    cohort = context.cohort.model_copy(update={'id_columns': [] if case == 'not_cohort_id' else ['CaseID']})
    context = context.model_copy(update={'variables': [lact, identifier], 'cohort': cohort})
    identity = SourceIdentity.from_record(_constraints(context)['source_identity'])
    monkeypatch.setattr(discovery.sources, 'registered_source_identity', lambda _: identity)
    EvidenceStore(tmp_path).register_text(kind='log', description='context',
        text=context.model_dump_json(), filename='context.json', evidence_id='research_context')
    monkeypatch.setattr(discovery, 'discover_source_partition_candidates', lambda **kw: {
        'files': [{'table': 'laboratory', 'path': 'laboratory/1.parquet',
                   'columns': ['LaboratoryValue'] if case == 'missing_schema_column'
                   else kw['requested_columns']['laboratory']}],
        'analysis_authorized': False})
    result = discovery.candidate_from_bound_research_context(
        study={'data_source': {'database': 'sic', 'path': '/registered'}}, run_dir=tmp_path)
    if case == 'missing_schema_column':
        assert result['identifier_candidates'] == []
    else:
        assert result['identifier_candidates'] == [{'variable': None, 'table': 'laboratory',
            'column': 'CaseID', 'actual_records_verified': False, 'join_or_derived_identifier_verified': False}]
    assert result['unmapped_variables'] == ['CaseID']
    assert result['analysis_authorized'] is False


def test_source_identity_is_visible_when_exported_identifier_has_no_verified_mapping(tmp_path, monkeypatch):
    context, *_ = signed_case_duration_case()
    lact = next(item for item in context.variables if item.name == 'lact').model_copy(update={'source_concept': 'lact'})
    case_id = next(item for item in context.variables if item.name == 'CaseID').model_copy(update={'role': 'other'})
    exported_id = case_id.model_copy(update={'name': 'patient_stay_id', 'role': 'id'})
    context = context.model_copy(update={'variables': [lact, case_id, exported_id],
        'cohort': context.cohort.model_copy(update={'id_columns': ['patient_stay_id']})})
    identity = SourceIdentity.from_record(_constraints(context)['source_identity'])
    monkeypatch.setattr(discovery.sources, 'registered_source_identity', lambda _: identity)
    EvidenceStore(tmp_path).register_text(kind='log', description='context',
        text=context.model_dump_json(), filename='context.json', evidence_id='research_context')
    monkeypatch.setattr(discovery, 'discover_source_partition_candidates', lambda **kw: {
        'files': [{'table': 'laboratory', 'path': 'laboratory/1.parquet',
                   'columns': kw['requested_columns']['laboratory']}], 'analysis_authorized': False})
    result = discovery.candidate_from_bound_research_context(
        study={'data_source': {'database': 'sic', 'path': '/registered'}}, run_dir=tmp_path)
    assert result['identifier_candidates'] == [{'variable': None, 'table': 'laboratory',
        'column': 'CaseID', 'actual_records_verified': False, 'join_or_derived_identifier_verified': False}]
    assert result['unmapped_variables'] == ['CaseID', 'patient_stay_id']
    assert result['analysis_authorized'] is False


def test_only_bound_project_session_and_run_can_request_preview(tmp_path, monkeypatch):
    setup(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(discovery, 'candidate_from_bound_research_context', lambda **kw: (
        calls.append(kw) or {'files': [], 'candidate_sha256': 'f' * 64}))
    kw = dict(service=Service(), project_id='project1', session_id='session1', run_id='run1', include_source_candidate=True)
    assert view.project_proposal(**kw)['source_partition_candidate']['status'] == 'candidate'
    with pytest.raises(ValueError, match='session_not_bound'):
        view.project_proposal(**{**kw, 'session_id': 'other'})
    assert len(calls) == 1 and calls[0]['run_dir'] == Path(tmp_path)


def test_candidate_view_escapes_manifest_data_and_marks_all_boundaries(monkeypatch):
    from easyicu.webserver.routes import non_executable_proposal as route
    monkeypatch.setattr(route, 'get_pi_copilot_service', lambda: object())
    monkeypatch.setattr(route, 'project_proposal', lambda **kw: {
        'proposal': {'original_question': 'question', 'decisions': []}, 'literature_candidates': None,
        'source_partition_candidate': {'status': 'candidate', 'candidate_sha256': 'f' * 64,
            'unmapped_variables': ['<unknown>'], 'identifier_candidates': [
                {'variable': '<id>', 'table': '<table>', 'column': '<column>'},
                {'variable': None, 'table': 'cases', 'column': 'CaseID'}],
            'features': [{'concept': '<lact>', 'table': 'lab', 'columns': ['field'], 'derivation_or_callback_present': True}],
            'files': [{'table': 'lab', 'path': '<file>.parquet', 'columns': ['field']}]}})
    app = FastAPI(); app.include_router(route.router)
    response = TestClient(app).get('/api/pi/proposal/project/session/run/source-candidate')
    assert response.status_code == 200
    assert '&lt;lact&gt;' in response.text and '&lt;file&gt;' in response.text and '<lact>' not in response.text
    assert '&lt;id&gt;' in response.text and '&lt;column&gt;' in response.text
    assert '未验证实际记录、表间关联或派生标识' in response.text
    assert '未绑定研究变量' in response.text and '不代表研究变量已绑定' in response.text
    assert '尚未验证原始文件内容' in response.text and '结局和队列依赖尚未完整建立' in response.text
    assert '不授权提取患者记录或分析' in response.text and '存在额外转换规则' in response.text


def test_blocked_preparation_exposes_metadata_without_approving(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver import agent_pipeline_runs, raw_preparation_candidate, sources, study_contexts
    from tests.webserver.test_sic_raw_preparation_grant import _case

    result, source_root, _, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    (tmp_path / 'research_context.json').write_text(result.context.model_dump_json())
    pending = SimpleNamespace(run_id='run', run_dir=str(tmp_path), requests=(checkpoint.request,))
    study = {'id': 'study', 'revision': 1}
    monkeypatch.setattr(agent_pipeline_runs, 'raw_preparation_review_owner', lambda *args: (pending, study))
    monkeypatch.setattr(sources, 'load_registry', lambda: {'sources': [{'path': str(source_root)}]})
    monkeypatch.setattr(discovery, 'candidate_from_bound_research_context', lambda **kw: {
        'files': [{'table': 'observations', 'path': 'observations/1.parquet', 'columns': ['value']}],
        'analysis_authorized': False, 'outcome_and_cohort_dependencies_complete': False})
    monkeypatch.setattr(study_contexts, 'get_context', lambda _: study)
    monkeypatch.setattr(study_contexts, 'scientific_configuration_sha256', lambda s: s['revision'])
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id='run', study_context_id='study')
    assert outcome['status'] == 'blocked' and outcome['blocker_codes']
    assert outcome['source_partition_candidate']['analysis_authorized'] is False
    assert not (tmp_path / 'raw_preparation_checkpoint.json').exists()
    monkeypatch.setattr(study_contexts, 'get_context', lambda _: {'id': 'study', 'revision': 2})
    stale = raw_preparation_candidate.create_candidate_for_review(run_id='run', study_context_id='study')
    assert stale['status'] == 'blocked' and 'source_partition_candidate' not in stale


def test_blocked_preparation_preview_escapes_sources_and_never_offers_approval():
    import shutil
    from tests.support.node import run_node
    from tests.webserver.copilot.pi_copilot_static_fixtures import _read

    node = shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read('js/screens-guided-pi-confirmation.js')!r});
      const esc=v=>String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
      const workflow={{next_action_code:'starting',starting:{{action_code:'prepare_analysis_data'}},
        raw_preparation_candidate_run_id:'run',raw_preparation_candidate_result:{{status:'blocked',partial_multitable_projection:{{partial_projection_only:true}},
          blocker_codes:['dependency_cohort_inputs_unknown','dependency_cohort_concepts_unrecorded'],source_partition_candidate:{{
            files:[{{table:'<table>',path:'<file>',columns:['id','<value>']}}],
            unmapped_variables:['<missing>'],identifier_candidates:[{{variable:'<id>',table:'<table>',column:'<column>'}},{{variable:null,table:'cases',column:'CaseID'}}]}}}}}};
      const owner=window.EU_GUIDED_PI_CONFIRMATION.create({{tr:(en,zh)=>zh,esc,iconHtml:()=>'',resourceButton:()=>'',
        sessionIsStale:()=>false,busy:()=>false,workflow:()=>workflow,session:()=>({{archived_child_jobs:[]}})}});
      process.stdout.write(owner.workflowConfirmationHtml());
    """
    result = run_node(node, script, check=False)
    assert result.returncode == 0, result.stderr
    html = result.stdout
    assert '&lt;table>' in html and '&lt;file>' in html and '&lt;missing>' in html
    assert '&lt;id>' in html and '&lt;column>' in html and '未验证实际记录、表间关联或派生标识' in html
    assert '未绑定研究变量' in html and '不代表研究变量已绑定' in html
    assert '不筛选队列、不关联表、不计算派生特征' in html and '暂不可审批或执行' in html
    assert '来源映射尚未覆盖已审阅人群条件使用的全部概念' in html
    assert '<file>' not in html and '不授权数据准备或分析' in html
    assert 'data-gpi-preparation-review' not in html


def test_click_forwards_candidate_and_ignores_stale_run_response():
    import json
    import shutil
    from tests.support.node import run_node
    from tests.webserver.copilot.pi_copilot_static_fixtures import _read

    node = shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    script = 'global.window = {};\n' + _read('js/screens-guided-pi-events.js') + '''
      let click, resolve;
      const host={querySelector:()=>null,addEventListener:(name,fn)=>{if(name==='click')click=fn;}};
      const state={host,session:{session_id:'session',binding:{study_context_id:'study'}},
        workflow:{raw_preparation_candidate_run_id:'run'}};
      const owner=window.EasyICU.guidedPi.require('events').create({state,projectId:()=> 'project',render:()=>{},
        api:()=>({previewRawPreparation:()=>new Promise(r=>resolve=r)}),loadWorkflow:async()=>{}});
      owner.wire();
      const target={closest:s=>s==='[data-gpi-preparation-preview]' ? {dataset:{gpiPreparationPreview:'run'}} : null};
      click({target});
      resolve({status:'blocked',blocker_codes:['dependency_cohort_inputs_unknown'],
        partial_multitable_projection:{partial_projection_only:true,approvable:false},
        source_partition_candidate:{files:[{table:'lab',path:'lab/1.parquet',columns:['value']}],unmapped_variables:['id']}});
      setImmediate(()=>{
        const forwarded=state.workflow.raw_preparation_candidate_result;
        click({target});
        state.workflow.raw_preparation_candidate_run_id='other';
        resolve({status:'blocked',source_partition_candidate:{files:[{path:'stale'}]}});
        setImmediate(()=>process.stdout.write(JSON.stringify({forwarded,state:state.workflow.raw_preparation_candidate_result})));
      });
    '''
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result['forwarded']['status'] == 'blocked'
    assert result['forwarded']['source_partition_candidate']['files'][0]['path'] == 'lab/1.parquet'
    assert result['forwarded']['partial_multitable_projection'] == {'partial_projection_only': True, 'approvable': False}
    assert result['state']['status'] == 'loading'
