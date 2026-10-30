import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import _load_guided_pi_module_harness, _read  # noqa: F401


def test_proposal_display_requires_host_pending_and_escapes_summary():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read('js/screens-guided-pi-confirmation.js')!r});
      let workflow = {{next_action_code:'starting', starting:{{action_code:'prepare_analysis_data'}}}};
      const escape = v => String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
      const owner = window.EU_GUIDED_PI_CONFIRMATION.create({{
        tr: (en, zh)=>zh, esc:escape, iconHtml:()=>'', resourceButton:()=>'',
        sessionIsStale:()=>false, busy:()=>false, workflow:()=>workflow,
        session:()=>({{archived_child_jobs:[]}}),
      }});
      const absent = owner.workflowConfirmationHtml();
      workflow.raw_preparation_candidate_run_id = 'run-1';
      workflow.raw_preparation_candidate_result = {{status:'loading'}};
      const loading = owner.workflowConfirmationHtml();
      workflow.raw_preparation = {{status:'awaiting_consumption_closure', summary:'<script>bad</script>'}};
      const pending = owner.workflowConfirmationHtml();
      process.stdout.write(JSON.stringify({{absent, loading, pending}}));
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert "data-gpi-preparation-preview" not in result["absent"]
    assert 'data-gpi-preparation-preview="run-1" disabled' in result["loading"]
    assert "&lt;script>bad&lt;/script>" in result["pending"]
    assert "<script>bad</script>" not in result["pending"]
    assert "暂时不能开始准备" in result["pending"]


def test_preview_click_uses_current_binding_and_stale_response_is_ignored():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = "global.window = {};\n" + _read("js/screens-guided-pi-events.js") + """
      let click, resolve;
      const calls = [];
      let project = 'project';
      const host = {querySelector:()=>null, addEventListener:(name, fn)=>{if(name==='click')click=fn;}};
      const state = {host, session:{session_id:'session',binding:{study_context_id:'study'}},
        workflow:{raw_preparation_candidate_run_id:'run'}};
      const owner = window.EasyICU.guidedPi.require('events').create({
        state, projectId:()=>project, render:()=>calls.push('render'),
        api:()=>({previewRawPreparation:body=>{calls.push(body); return new Promise(r=>resolve=r);}}),
        loadWorkflow:()=>calls.push('reload'),
      });
      owner.wire();
      click({target:{closest:s=>s==='[data-gpi-preparation-preview]' ? {dataset:{gpiPreparationPreview:'run'}} : null}});
      const loading=state.workflow.raw_preparation_candidate_result.status;
      project='another-project'; resolve({status:'pending'});
      setImmediate(()=>process.stdout.write(JSON.stringify({loading,calls,state:state.workflow.raw_preparation_candidate_result})));
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result["loading"] == "loading"
    assert {"run_id": "run", "study_context_id": "study", "session_id": "session", "project_id": "project"} in result["calls"]
    assert "reload" not in result["calls"]


def test_review_controls_require_a_host_offer_and_show_every_escaped_dependency():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read('js/screens-guided-pi-confirmation.js')!r});
      const esc = v => String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
      let workflow = {{next_action_code:'starting', starting:{{action_code:'prepare_analysis_data'}},
        raw_preparation:{{status:'awaiting_consumption_closure', summary:'summary', scope:{{dependencies:{{files:[
          {{table:'table',path:'<one>',columns:['id'],window:'first day'}},
          {{table:'table',path:'second',columns:['value'],window:'first day'}}]}}}}}}}};
      const owner=window.EU_GUIDED_PI_CONFIRMATION.create({{tr:(en,zh)=>zh,esc,iconHtml:()=>'',resourceButton:()=>'',
        sessionIsStale:()=>false,busy:()=>false,workflow:()=>workflow,session:()=>({{archived_child_jobs:[]}})}});
      const unavailable=owner.workflowConfirmationHtml();
      workflow.host_decisions={{raw_preparation_review:{{family:'raw_preparation_review'}}}};
      const offered=owner.workflowConfirmationHtml();
      workflow.raw_preparation_review_loading=true;
      const loading=owner.workflowConfirmationHtml();
      process.stdout.write(JSON.stringify({{unavailable,offered,loading}}));
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert "data-gpi-preparation-review" not in result["unavailable"]
    assert "&lt;one>" in result["offered"] and "second" in result["offered"]
    assert 'data-gpi-preparation-review="approved"' in result["offered"]
    assert 'data-gpi-preparation-review="approved" disabled' in result["loading"]


def test_review_click_carries_only_the_current_host_offer():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = "global.window = {};\n" + _read("js/screens-guided-pi-events.js") + """
      let click; const calls=[];
      const host={querySelector:()=>null,addEventListener:(name,fn)=>{if(name==='click')click=fn;}};
      const offer={family:'raw_preparation_review',run_id:'run'};
      const state={host,session:{session_id:'session',binding:{study_context_id:'study'}},
        workflow:{host_decisions:{raw_preparation_review:offer}}};
      const owner=window.EasyICU.guidedPi.require('events').create({state,projectId:()=> 'project',render:()=>{},
        api:()=>({reviewRawPreparation:async body=>calls.push(body)}),loadWorkflow:async()=>calls.push('reload'),tr:(en,zh)=>zh});
      owner.wire();
      click({target:{closest:s=>s==='[data-gpi-preparation-review]' ? {dataset:{gpiPreparationReview:'approved'}} : null}});
      setImmediate(()=>process.stdout.write(JSON.stringify(calls)));
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result[0]["host_action"]["action_code"] == "review_raw_preparation"
    assert result[0]["host_action"]["decision"] == {"family": "raw_preparation_review", "run_id": "run"}
    assert result[0]["decision"] == "approved"


def test_blocked_candidate_shows_specific_reasons_without_untrusted_codes():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read('js/screens-guided-pi-confirmation.js')!r});
      const esc=v=>String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;');
      const workflow={{next_action_code:'starting',starting:{{action_code:'prepare_analysis_data'}},
        raw_preparation_candidate_run_id:'run',raw_preparation_candidate_result:{{status:'blocked',
          blocker_codes:['dependency_cohort_inputs_unknown','dependency_partition_roster_unknown','<script>','__proto__']}}}};
      const owner=window.EU_GUIDED_PI_CONFIRMATION.create({{tr:(en,zh)=>zh,esc,iconHtml:()=>'',resourceButton:()=>'',
        sessionIsStale:()=>false,busy:()=>false,workflow:()=>workflow,session:()=>({{archived_child_jobs:[]}})}});
      process.stdout.write(owner.workflowConfirmationHtml());
    """
    html = run_node(node, script, check=True).stdout
    assert "尚未完整识别筛选患者所需的来源表" in html
    assert "尚未确认全部来源分区文件" in html
    assert "<script>" not in html
    assert "[object Object]" not in html
    assert "data-gpi-preparation-review" not in html
