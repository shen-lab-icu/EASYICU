"""Source selection and artifact reading do not manufacture research decisions."""
import json
import shutil
import subprocess

import pytest

from easyicu.webserver.pi_copilot.projections import project_run_outcome

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    return json.loads(subprocess.check_output([node, "--eval", script], text=True))


def test_source_confirmation_continues_once_but_selection_only_opens_picker():
    result = run_js(f"""
      global.window = {{}};
      eval({_read('js/screens-guided-pi-data-binding.js')!r});
      let session = {{session_id:'s'}};
      const calls = [];
      const owner = window.EU_GUIDED_PI_DATA_BINDING.create({{
        session: () => session, busy: () => false, projectId: () => 'p',
        api: () => ({{authorizePiCopilotDataSource: async (id, request) => {{
          calls.push(request.action);
          return {{session: {{session_id:id, data_source_authorization: {{status:
            request.action === 'use_study_required_data' ? 'confirmed' : 'selection_in_progress'}}}}}};
        }}}}),
        setSession: value => session = value, rememberSession: () => {{}},
        setError: value => {{if(value) throw Error(value);}}, errorText: String,
        render: () => {{}}, loadWorkflow: async () => calls.push('workflow'),
        continueAfterDataSourceConfirmation: async () => calls.push('continue'),
      }});
      (async () => {{
        await owner.authorizeDataSource('use_study_required_data');
        await owner.authorizeDataSource('begin_local_selection');
        process.stdout.write(JSON.stringify(calls));
      }})();
    """)
    assert result == ['use_study_required_data', 'workflow', 'continue', 'begin_local_selection']


@pytest.mark.parametrize('lookup', ['exact', 'missing', 'wrong', 'switched'])
def test_source_picker_uses_host_context_when_browser_history_is_full(lookup):
    result = run_js(f"""
      global.window = global;
      global.location = {{hash:'#guided'}};
      const storage = new Map();
      global.localStorage = {{getItem:k=>storage.get(k)||null,
        setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)}};
      global.CustomEvent = class {{constructor(type,init){{this.type=type;this.detail=init.detail;}}}};
      global.addEventListener = () => {{}};
      global.dispatchEvent = () => {{}};
      global.document = {{readyState:'loading',addEventListener:()=>{{}}}};
      const target = {{id:'new-project',revision:4,question:'New research question',
        data_source:{{path:'/exports/new',database:'miiv'}},current_stage:'study_setup'}};
      const old = Array.from({{length:80}},(_,i)=>({{id:'history-'+i,revision:2,
        question:'Historical question '+i,data_source:{{path:'/exports/old-'+i,database:'miiv'}}}}));
      const calls = [];
      let activeId = old[79].id;
      global.EU_API = {{
        loadActiveStudyContext:async()=>({{context:old[79]}}),
        listStudyContexts:async()=>({{contexts:[target,...old],active_id:activeId}}),
        loadStudyContext:async id=>{{
          calls.push('load:'+id);
          if ({lookup!r}==='missing') throw Error('HTTP 404');
          if ({lookup!r}==='switched') await window.EU_STUDY_CONTEXT.activate(old[0].id);
          return {{context:{lookup!r}==='wrong'?old[0]:target}};
        }},
        saveStudyContext:async body=>{{
          calls.push(body);activeId=body.id;
          return {{context:[target,...old].find(c=>c.id===body.id)}};
        }},
      }};
      eval({_read('js/study-context.js')!r});
      global.EU_GUIDED_PI_PREVIEW = {{open:()=>calls.push('open')}};
      eval({_read('js/screens-guided-pi-data-binding.js')!r});
      let session = {{session_id:'s',binding:{{study_context_id:target.id}}}};
      let error = '';
      const owner = window.EU_GUIDED_PI_DATA_BINDING.create({{
        session:()=>session,busy:()=>false,projectId:()=> 'p',
        api:()=>({{authorizePiCopilotDataSource:async()=>({{
          session:{{...session,data_source_authorization:{{status:'selection_in_progress'}}}},
          resource:{{kind:'native_workspace',study_context_id:target.id}}
        }})}}),
        setSession:s=>session=s,rememberSession:()=>{{}},
        setError:s=>error=s,errorText:e=>e.message,render:()=>{{}},
        loadWorkflow:async()=>calls.push('workflow'),
        continueAfterDataSourceConfirmation:async()=>calls.push('continue'),
      }});
      (async()=>{{
        await owner.authorizeDataSource('begin_local_selection');
        process.stdout.write(JSON.stringify({{calls,error,active:window.EU_STUDY_CONTEXT.active(),
          authorization:session.data_source_authorization.status}}));
      }})();
    """)
    assert 'load:new-project' in result['calls']
    assert result['authorization'] == 'selection_in_progress'
    assert 'continue' not in result['calls']
    if lookup == 'exact':
        assert result['error'] == ''
        assert result['active']['id'] == 'new-project'
        assert result['active']['data_source']['path'] == '/exports/new'
        assert {'id': 'new-project'} in result['calls']
        assert result['calls'][-1] == 'open'
    else:
        assert result['error']
        assert result['active']['id'] == ('history-0' if lookup == 'switched' else 'history-79')
        assert {'id': 'new-project'} not in result['calls']
        assert 'open' not in result['calls']


@pytest.mark.parametrize('status', ['confirmed', 'pending', 'selection_in_progress'])
def test_local_source_receipt_is_visible_without_inventing_a_chat_decision(status):
    result = run_js(f"""
      global.window = {{}};
      eval({_read('js/html-escape.js')!r});
      eval({_read('js/screens-guided-pi-data-consent.js')!r});
      const session = {{data_source_authorization:{{status:{status!r},
        confirmation_mode:'select_local_source',confirmed_at:'2026-09-08T08:12:48Z',
        source:{{label:'MIMIC-IV <img src=x>',reference_release:'3.1'}}}}}};
      const before = JSON.stringify(session);
      const html = window.EU_GUIDED_PI_DATA_CONSENT.renderSelectedSource(session,
        {{esc:window.EU_HTML.esc,tr:(_en,zh)=>zh}});
      process.stdout.write(JSON.stringify({{html,same:before===JSON.stringify(session)}}));
    """)
    assert result['same']
    if status != 'confirmed':
        assert result['html'] == ''
    else:
        assert 'MIMIC-IV &lt;img src=x&gt; v3.1' in result['html']
        assert '2026-09-08T08:12:48Z' in result['html']
        assert '<img' not in result['html']
        assert 'data-gpi-data-source-action' not in result['html']


def test_resource_cards_preserve_coordinates_and_escape_labels():
    result = run_js(f"""
      global.window = {{EU_LANG:'zh'}};
      eval({_read('js/html-escape.js')!r});
      eval({_read('js/screens-guided-pi-resources.js')!r});
      const owner = window.EU_GUIDED_PI_RESOURCES.create({{esc:window.EU_HTML.esc}});
      const row = {{resources:[
        {{kind:'research_artifact', run_id:'run_safe', artifact:'result_tables.json',
          sha256:'abc', label:'打开结果表 <img src=x onerror=bad()>', media_type:'application/json'}},
        {{kind:'research_artifact', run_id:'run_safe', artifact:'evidence_ledger.json', label:'证据台账'}},
      ]}};
      const before = JSON.stringify(row);
      const html = owner.renderForMessage(row);
      process.stdout.write(JSON.stringify({{html, unchanged: before === JSON.stringify(row)}}));
    """)
    assert result['unchanged']
    html = result['html']
    assert 'gpi-file-card' in html
    assert 'data-gpi-resource-run="run_safe"' in html
    assert 'data-gpi-resource-digest="abc"' in html
    assert '<img' not in html
    assert '<details class="gpi-resource-technical">' in html
    assert '交互预览' in html


def test_result_summary_does_not_invent_complete_cases_or_merge_distributions():
    result = run_js(f"""
      global.window = {{}};
      eval({_read('js/screens-guided-pi-result-summary.js')!r});
      const table = {{headers:['row_role','exposure_level','n_rows','exposure_denominator','exposure_pct','outcome_events','outcome_denominator','outcome_rate_pct'],
        rows:[['overall','',10,10,100,2,10,20],['exposure_level','0.0',10,10,100,2,10,20]]}};
      const summarize = window.EU_GUIDED_PI_RESULT_SUMMARY.summarize;
      const plan = {{display_labels:{{'x=0':'Reference','y':'Outcome'}},steps:[{{planned_analysis_role:'primary',exposure_outcome_distribution_spec:{{exposure:'x',outcome:'y',exposure_levels:[0]}}}}]}};
      process.stdout.write(JSON.stringify({{one:summarize({{tables:[table]}},plan),two:summarize({{tables:[table,{{...table}}]}},plan)}}));
    """)
    claims = result['one']['claims']
    assert not any(row['source_field'] == 'n_complete_case' for row in claims)
    assert next(row for row in claims if row['source_field'] == 'overall_outcome.risk_pct')['display_value'] == '20.00%'
    assert result['one']['exposureLevels'][0]['label'] == 'Reference'
    assert result['two']['exposureLevels'] == []
    assert result['two']['claims'] == []


@pytest.mark.parametrize('matching_digest', [True, False])
def test_full_report_reads_separately_bound_revision_without_promoting_source(matching_digest):
    digest = 'a' * 64
    result = run_js(f"""
      global.window = {{EU_LANG:'zh',AGENT_RENDER:{{manuscriptProvenanceView:()=>'<article>Verified revision</article>'}}}};
      eval({_read('js/html-escape.js')!r});
      eval({_read('js/screens-guided-pi-analysis-report.js')!r});
      const payload = {{source_manifest:{{readiness:{{manuscript_ready:false}}}},
        manuscript_provenance:{{manuscript_sha256:{digest!r},report_revision:{{
          schema_version:'easyicu.web-report-revision/1',status:'pass',claim_ceiling:'analysis_only',
          publication_authorized:false,analysis_steps_executed:0,
          output_sha256:{(digest if matching_digest else 'b' * 64)!r}}}}}}};
      const before = JSON.stringify(payload);
      process.stdout.write(JSON.stringify({{html:window.EU_GUIDED_PI_ANALYSIS_REPORT.render(payload),same:before===JSON.stringify(payload)}}));
    """)
    assert result['same']
    assert ('Verified revision' in result['html']) is matching_digest


def test_full_report_shows_real_results_plan_and_article_without_raw_html():
    result = run_js(f"""
      global.window = {{EU_LANG:'zh',AGENT_RENDER:{{manuscriptProvenanceView:()=>'<article>Bound article and references</article>'}}}};
      eval({_read('js/html-escape.js')!r});
      eval({_read('js/screens-guided-pi-result-summary.js')!r});
      eval({_read('js/screens-guided-pi-analysis-report.js')!r});
      const payload = {{run_context:{{question:'<img src=x>',source:{{label:'Selected source'}}}},
        source_manifest:{{readiness:{{manuscript_ready:true}}}},
        plan:{{steps:[{{intent:'Describe observed counts'}}]}},
        result_tables:{{tables:[{{headers:['row_role','exposure_level','n_rows','exposure_denominator','exposure_pct','outcome_events','outcome_denominator','outcome_rate_pct'],
          rows:[['overall','',100,100,100,3,100,3],['exposure_level','A',100,100,100,3,100,3]]}}]}}}};
      process.stdout.write(JSON.stringify(window.EU_GUIDED_PI_ANALYSIS_REPORT.render(payload)));
    """)
    assert '<img' not in result
    assert '3.00%' in result and '3 / 100' in result
    assert 'Describe observed counts' in result
    assert 'Bound article and references' in result
    assert '完整变量行' not in result


def test_execution_history_keeps_user_decisions_latest_failure_and_live_activity_visible():
    result = run_js(f"""
      global.window = {{}};
      eval({_read('js/screens-guided-pi-activity.js')!r});
      const owner = window.EU_GUIDED_PI_ACTIVITY.create({{esc:String,tr:(_en,zh)=>zh}});
      const rows = [{{id:'question',role:'user'}},...['complete','failed','failed'].map((status,index)=>({{id:'attempt'+index,role:'activity',status}})),
        {{id:'decision',role:'user'}},{{id:'live',role:'activity',status:'running'}}];
      const before = JSON.stringify(rows);
      process.stdout.write(JSON.stringify({{html:owner.renderTimeline(rows,row=>`<p>${{row.id}}</p>`),same:before===JSON.stringify(rows)}}));
    """)
    assert result['same']
    html = result['html']
    assert html.count('<details') == 1
    assert html.index('</details>') < html.index('attempt2') < html.index('decision') < html.index('live')
    assert all(html.count(name) == 1 for name in ('question', 'attempt0', 'attempt1', 'attempt2', 'decision', 'live'))


@pytest.mark.parametrize('same_source', [True, False])
def test_report_revision_draft_availability_is_separate_from_scientific_gate(same_source):
    review = {'ok': True, 'run_id': 'run_a', 'gate': {'reportable': False},
        'artifact_payloads': {'manuscript_provenance.json': {
            'manuscript_sha256': 'a' * 64,
            'report_revision': {'schema_version': 'easyicu.web-report-revision/1',
                'source_run_id': 'run_a' if same_source else 'run_b', 'status': 'pass',
                'analysis_steps_executed': 0, 'claim_ceiling': 'analysis_only',
                'publication_authorized': False, 'output_sha256': 'a' * 64}}}}
    result = project_run_outcome(review)
    assert result['report_revision_ready'] is same_source
    assert result['manuscript_ready'] is False
    assert result['reportable'] is False


@pytest.mark.parametrize('revision', [True, False])
def test_revised_report_never_advertises_the_source_runs_old_pdf(revision):
    html = run_js(f"""
      global.window = {{}};
      eval({_read('js/screens-guided-pi-run-outcome.js')!r});
      const owner = window.EU_GUIDED_PI_RUN_OUTCOME.create({{
        tr:(_en,zh)=>zh,esc:String,iconHtml:()=>'',resourceButton:(row)=>row.artifact,
      }});
      process.stdout.write(JSON.stringify(owner.render({{
        present:true,analysis_results_available:true,analysis_validated:true,numeric_verified:true,
        manuscript_ready:true,report_revision_ready:{json.dumps(revision)},run_id:'run_a',
        artifact_refs:[{{artifact:'evidence_ledger.json',sha256:'a'.repeat(64)}},{{artifact:'manuscript_scaffold.pdf'}}],
      }},{{stages:[{{id:'analysis',status:'complete'}}],analysis_validation_retry_available:true}})));
    """)
    assert ('manuscript_scaffold.pdf' in html) is not revision
    assert 'data-gpi-run-outcome-retry="report_only"' in html


@pytest.mark.parametrize('mismatch', [None, 'pdf_hash', 'revision_id', 'manuscript_hash', 'missing'])
def test_revision_pdf_requires_the_exact_reader_revision_and_ledger(mismatch):
    pdf = {'name': 'manuscript_revision.pdf', 'sha256': 'b' * 64,
           'revision_id': 'revision_a', 'manuscript_sha256': 'a' * 64}
    if mismatch == 'pdf_hash':
        pdf['sha256'] = 'c' * 64
    elif mismatch == 'revision_id':
        pdf['revision_id'] = 'revision_old'
    elif mismatch == 'manuscript_hash':
        pdf['manuscript_sha256'] = 'c' * 64
    review = {'ok': True, 'run_id': 'run_a', 'gate': {'checks': {'execution_complete': True, 'analysis_validated': True}},
        'artifacts': [] if mismatch == 'missing' else [{'name': 'manuscript_revision.pdf', 'sha256': 'b' * 64}],
        'artifact_payloads': {'manuscript_provenance.json': {
            'manuscript_sha256': 'a' * 64,
            'report_revision': {'schema_version': 'easyicu.web-report-revision/1',
                'revision_id': 'revision_a', 'source_run_id': 'run_a', 'status': 'pass',
                'analysis_steps_executed': 0, 'claim_ceiling': 'analysis_only',
                'publication_authorized': False, 'output_sha256': 'a' * 64, 'pdf_artifact': pdf}}}}
    result = project_run_outcome(review)
    assert result['report_revision_pdf_ready'] is (mismatch is None)
    assert any(row['artifact'] == 'manuscript_revision.pdf' for row in result['artifact_refs']) is (mismatch is None)
    assert result['manuscript_ready'] is False
    assert result['reportable'] is False
