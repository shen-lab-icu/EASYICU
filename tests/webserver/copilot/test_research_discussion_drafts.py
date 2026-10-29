"""Advisory design discussion remains usable before data, without new authority."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
from tests.support.node import run_node

APP_DIR = Path(tools.__file__).with_name('node_app')


def _context():
    return ToolExecutionContext(session=PiSessionRecord(session_id='pi-discussion-test'))


def test_discussion_hint_is_response_metadata_not_scientific_state_or_a_grant(monkeypatch):
    captured = []
    original = {'status': 'ok', 'code': 'study_context_updated',
                'details': {'study': {'question': 'User question'}, 'workflow': {'missing_setup_fields': ['data_source']}}}
    before = deepcopy(original)

    def owner(context, params, **kwargs):
        captured.append((context, deepcopy(params), kwargs))
        return original

    monkeypatch.setattr(tools, 'update_study_context', owner)
    context = _context()
    params = {'question': 'User question', 'discussion_only': True}
    result = tools._update_study_context(context, params)
    assert captured[0][0] is context
    assert captured[0][1] == {'question': 'User question'}
    assert captured[0][2] == {'load_context': tools._bound_context, 'project_workflow': tools._workflow_snapshot}
    assert context.allowed_actions == frozenset()
    assert result['details']['discussion_only'] is True
    assert result['details']['study'] == before['details']['study']
    assert original == before
    assert params == {'question': 'User question', 'discussion_only': True}


@pytest.mark.parametrize('flag', [False, None, 'true', 1, [], {}])
def test_invalid_hint_is_rejected_and_false_is_stripped(monkeypatch, flag):
    calls = []
    monkeypatch.setattr(tools, 'update_study_context', lambda context, params, **kw:
                        calls.append(params) or {'status': 'blocked', 'code': 'pi_action_authorization_required', 'details': {}})
    result = tools._update_study_context(_context(), {'question': 'Question', 'discussion_only': flag})
    if flag is False:
        assert calls == [{'question': 'Question'}]
        assert result['code'] == 'pi_action_authorization_required'
        assert 'discussion_only' not in result['details']
    else:
        assert not calls
        assert result['code'] == 'invalid_discussion_only'


@pytest.mark.parametrize('binding', [{'bind_source_id': 'source-1'}, {'bind_active_export': True}])
def test_discussion_hint_cannot_carry_source_binding(monkeypatch, binding):
    def forbidden(*args, **kwargs):
        pytest.fail('Source binding must not reach the scientific owner in discussion-only mode')
    monkeypatch.setattr(tools, 'update_study_context', forbidden)
    result = tools._update_study_context(_context(), {'question': 'Question', 'discussion_only': True, **binding})
    assert result['status'] == 'blocked'
    assert result['code'] == 'discussion_source_binding_not_allowed'


def test_blocked_scientific_owner_receipt_cannot_become_a_successful_discussion_hint(monkeypatch):
    original = {'status': 'blocked', 'code': 'pi_action_authorization_required', 'details': {}}
    monkeypatch.setattr(tools, 'update_study_context', lambda *args, **kwargs: original)
    assert tools._update_study_context(_context(), {'discussion_only': True, 'question': 'Q'}) is original
    assert 'discussion_only' not in original['details']


def test_reply_hint_does_not_bypass_the_actual_configure_grant(monkeypatch):
    from easyicu.webserver.pi_copilot import study_context_update as owner
    monkeypatch.setattr(tools, '_bound_context', lambda binding: {'id': 's1', 'revision': 1, 'question': 'Old question'})
    monkeypatch.setattr(owner.study_contexts, 'upsert_context', lambda *a, **kw: pytest.fail('No Configure grant must mean no write'))
    result = tools._update_study_context(_context(), {'discussion_only': True, 'question': 'New question'})
    assert result['status'] == 'blocked'
    assert result['code'] == 'pi_action_authorization_required'
    assert 'discussion_only' not in result.get('details', {})


def test_host_reply_hint_keeps_model_turn_open_but_unmarked_source_flow_stays_owned():
    node = shutil.which('node')
    if not node or not (APP_DIR / 'node_modules').is_dir():
        pytest.skip('Pinned Pi Node runtime is unavailable')
    module = APP_DIR / 'src' / 'post-tool-finalization.mjs'
    script = r'''
      import { hostPostToolFinalization } from MODULE;
      const model = { api:'openai-completions', provider:'test', id:'test' };
      const question = '只讨论设计草案，不绑定数据、不提取、不批准执行。';
      function context({hint, modelHint, ready=false, error=false}={}) {
        return {messages:[
          {role:'user',content:[{type:'text',text:question}]},
          {role:'assistant',content:[{type:'toolCall',id:'u1',name:'easyicu_update_study_context',
            arguments:{question,...(modelHint ? {discussion_only:true} : {})}}]},
          {role:'toolResult',toolCallId:'u1',toolName:'easyicu_update_study_context',isError:error,content:[],
            details:{status:error ? 'blocked' : 'ok',code:error ? 'denied' : 'study_context_updated',details:{
              ...(hint !== undefined ? {discussion_only:hint} : {}),
              workflow:{missing_setup_fields:ready ? [] : ['data_source'],
                next_action_code:ready ? 'provider_ready_to_generate_plan' : 'study_setup_incomplete',
                study_setup_receipt:{configuration:{data_source:ready ? {database:'miiv'} : {}}}}
            }}}
        ]};
      }
      const results = {};
      for (const hint of [true, false, 'true']) {
        const stream = hostPostToolFinalization(model,context({hint}),'zh');
        results[String(hint)+':'+typeof hint] = stream ? (await stream.result()).content[0].text : null;
      }
      const modelHint = hostPostToolFinalization(model,context({modelHint:true}),'zh');
      results.modelOnly = (await modelHint.result()).content[0].text;
      results.readyDiscussion = hostPostToolFinalization(model,context({hint:true,ready:true}),'zh');
      const ready = hostPostToolFinalization(model,context({ready:true}),'zh');
      results.readyDefault = (await ready.result()).content[0].text;
      results.blocked = hostPostToolFinalization(model,context({hint:true,error:true}),'zh');
      console.log(JSON.stringify(results));
    '''.replace('MODULE', json.dumps(module.as_uri()))
    completed = run_node(node, script, module=True, cwd=APP_DIR, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)
    assert result['true:boolean'] is None
    assert result['readyDiscussion'] is None
    assert '请先选择' in result['false:boolean']
    assert result['true:string'] == result['false:boolean']
    assert result['modelOnly'] == result['false:boolean']
    assert '候选研究计划' in result['readyDefault']
    assert result['blocked'] is None
