"""A named demo cannot look clickable while the conversation is busy."""

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


@pytest.mark.requires_node
@pytest.mark.usefixtures('_load_guided_pi_module_harness')
@pytest.mark.parametrize('fallback', [False, True])
def test_busy_demo_offer_is_explained_and_does_not_start_until_idle(fallback: bool) -> None:
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is not installed')
    script = f"""
      global.window = global; window.EU_LANG = 'zh';
      eval({_ESCAPE_OWNER!r});
      eval({_read('js/database-names.js')!r});
      eval({_read('js/screens-guided-pi-next-actions.js')!r});
      eval({_read('js/screens-guided-pi-host-jobs.js')!r});
      eval({_read('js/screens-guided-pi-data-consent.js')!r});
      const consent = window.EasyICU.guidedPi.require('dataConsent');
      const demo = {{ id: 'eicu_demo_v2_0_1', database: 'eicu_demo', title: 'eICU Collaborative Research Database Demo', label: 'eICU Demo', status: {{ export_ready: false }} }};
      const fallback = {json.dumps(fallback)};
      const session = {{session_id: 's1', data_source_authorization: {{status:'pending'}}}};
      let busy = true; let errors = []; let prepared = [];
      const host = {{
        tr: (en, zh) => zh, dataConsent: consent, session: () => session, projectId: () => 'p1',
        messages: () => [{{role: 'user', text: fallback ? '在 eICU 数据里评估乳酸和死亡率' : '在 eICU Demo 数据里评估乳酸和死亡率'}}],
        busy: () => busy, render() {{}}, setError: x => errors.push(x), errorText: e => String(e),
        workflowReceipts: () => [], setWorkflowReceipts() {{}},
        api: () => ({{
          loadOfficialDemoSources: async () => ({{sources: [demo]}}),
          loadPiCopilotRegisteredSourceOffer: async () => ({{state:'ready', choices:[], official_demo:demo}}),
          startOfficialDemoSourcePrepare: async id => {{ prepared.push(id); return {{job_id:'job1'}}; }},
        }}),
      }};
      const owner = window.EasyICU.guidedPi.require('hostJobs').create(host);
      const ctx = () => ({{tr:host.tr, esc:EU_HTML.esc, icon:()=>'', namedDemo:owner.namedDemo(), namedDatabase:owner.namedDatabase()}});
      const tick = () => new Promise(resolve => setImmediate(resolve));
      (async () => {{
        ctx(); await tick(); ctx(); await tick();
        const busyHtml = consent.render(session, ctx());
        const denied = await owner.useNamedDemo(demo.id);
        const busyPrepared = prepared.length;
        const busyNotice = errors.at(-1);
        busy = false;
        const idleHtml = consent.render(session, ctx());
        const started = await owner.useNamedDemo(demo.id);
        const pendingHtml = consent.render(session, ctx());
        owner.stopAll();
        process.stdout.write(JSON.stringify({{busyHtml,idleHtml,pendingHtml,denied,busyPrepared,busyNotice,started,prepared}}));
      }})().catch(err => {{ console.error(err); process.exitCode=1; }});
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert 'data-gpi-named-demo="eicu_demo_v2_0_1" disabled' in result['busyHtml']
    assert '等待当前回复或任务结束' in result['busyHtml']
    assert result['denied'] is False
    assert result['busyPrepared'] == 0
    assert '等待当前回复或研究任务结束' in result['busyNotice']
    assert 'data-gpi-named-demo="eicu_demo_v2_0_1" disabled' not in result['idleHtml']
    assert '准备' in result['idleHtml']
    assert result['started'] is True
    assert result['prepared'] == ['eicu_demo_v2_0_1']
    assert 'data-gpi-named-demo="eicu_demo_v2_0_1" disabled' in result['pendingHtml']
    assert '正在准备' in result['pendingHtml']


@pytest.mark.requires_node
@pytest.mark.usefixtures('_load_guided_pi_module_harness')
def test_late_demo_start_does_not_auto_use_in_another_session() -> None:
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is not installed')
    script = f"""
      global.window = global; window.EU_LANG = 'en';
      eval({_ESCAPE_OWNER!r});
      eval({_read('js/screens-guided-pi-host-jobs.js')!r});
      const demo = {{id:'eicu-demo',title:'eICU Demo',status:{{export_ready:false}}}};
      let session = {{session_id:'s1'}}, project='p1', complete;
      let snapshots=0, errors=[];
      const host = {{
        tr: (en, zh) => en, dataConsent: {{requiresConfirmation:()=>true}},
        session:()=>session, projectId:()=>project,
        messages:()=>[{{role:'user',text:'Use eICU Demo'}}],
        busy:()=>false, render(){{}}, workflowReceipts:()=>[], setWorkflowReceipts(){{}},
        setError:x=>errors.push(x), errorText:e=>String(e),
        api:()=>({{
          loadOfficialDemoSources:async()=>({{sources:[demo]}}),
          startOfficialDemoSourcePrepare:()=>new Promise(resolve=>{{complete=resolve}}),
          loadJobSnapshot:async()=>{{snapshots++;return {{status:'done'}}}},
        }}),
      }};
      const owner=window.EasyICU.guidedPi.require('hostJobs').create(host);
      const tick=()=>new Promise(resolve=>setImmediate(resolve));
      (async()=>{{
        owner.namedDemo(); await tick();
        const started=owner.useNamedDemo('eicu-demo');
        session={{session_id:'s2'}}; project='p2';
        complete({{job_id:'job-old'}});
        const result=await started; await tick();
        const offer=owner.namedDemo(); owner.stopAll();
        process.stdout.write(JSON.stringify({{result,snapshots,errors,pending:offer.pending}}));
      }})().catch(err=>{{console.error(err);process.exitCode=1}});
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result == {'result': False, 'snapshots': 0, 'errors': [''], 'pending': False}


@pytest.mark.requires_node
@pytest.mark.usefixtures('_load_guided_pi_module_harness')
@pytest.mark.parametrize('fallback', [False, True])
def test_directory_confirmation_error_links_to_the_existing_data_page(fallback: bool) -> None:
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is not installed')
    script = f"""
      global.window=global; window.EU_LANG='zh';
      eval({_ESCAPE_OWNER!r});
      eval({_read('js/database-names.js')!r});
      eval({_read('js/screens-guided-pi-next-actions.js')!r});
      eval({_read('js/screens-guided-pi-host-jobs.js')!r});
      eval({_read('js/screens-guided-pi-data-consent.js')!r});
      const consent=window.EasyICU.guidedPi.require('dataConsent');
      const demo={{id:'eicu-demo',database:'eicu_demo',title:'eICU Demo',status:{{export_ready:false}}}};
      let session={{session_id:'s1',data_source_authorization:{{status:'pending'}}}}, project='p1', errors=[];
      let saves=0;
      const host={{tr:(en,zh)=>zh,dataConsent:consent,session:()=>session,projectId:()=>project,
        messages:()=>[{{role:'user',text:{json.dumps('在 eICU 数据里研究乳酸' if fallback else '在 eICU Demo 数据里研究乳酸')}}}],
        busy:()=>false,render(){{}},workflowReceipts:()=>[],setWorkflowReceipts(){{}},
        setError:x=>errors.push(x),errorText:e=>String(e),
        api:()=>({{loadOfficialDemoSources:async()=>({{sources:[demo]}}),
          loadPiCopilotRegisteredSourceOffer:async()=>({{choices:[],official_demo:demo}}),
          startOfficialDemoSourcePrepare:async()=>{{throw Object.assign(new Error('x'),{{code:'data_directory_confirmation_required'}})}},
          saveSettings:async()=>{{saves++}},
        }}),
      }};
      const owner=window.EasyICU.guidedPi.require('hostJobs').create(host);
      const ctx=()=>({{tr:host.tr,esc:EU_HTML.esc,icon:()=>'',namedDemo:owner.namedDemo(),namedDatabase:owner.namedDatabase()}});
      const tick=()=>new Promise(resolve=>setImmediate(resolve));
      (async()=>{{
        ctx();await tick();ctx();await tick();
        const before=consent.render(session,ctx());
        const result=await owner.useNamedDemo('eicu-demo');
        const help=consent.render(session,ctx());
        session={{session_id:'s2',data_source_authorization:{{status:'pending'}}}};project='p2';
        ctx();await tick();
        const other=consent.render(session,ctx());
        owner.stopAll();
        process.stdout.write(JSON.stringify({{before,result,help,other,errors,saves}}));
      }})().catch(err=>{{console.error(err);process.exitCode=1}});
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert 'href="#patient"' not in result['before']
    assert result['result'] is False
    assert 'href="#patient"' in result['help']
    assert '下载到这里' in result['help']
    assert '编辑设置' in result['help']
    assert '返回本次对话' in result['help']
    assert 'href="#patient"' not in result['other']
    assert '请先在患者审阅的数据选择页确认 Demo 下载位置' in result['errors'][-1]
    assert result['saves'] == 0
