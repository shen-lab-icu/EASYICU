"""Explicit, scope-bound authorization for a blocked candidate-plan turn."""
import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import _load_guided_pi_module_harness, _read  # noqa: F401


def test_plan_authorization_requires_exact_configuration_and_fresh_host_decision():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{ EU_HTML: {{esc: String}}, t: (en, zh) => zh }};
      eval({_read('js/screens-guided-pi-access.js')!r});
      const access = window.EasyICU.guidedPi.require('access');
      const pending = access.pending([{{role:'user',text:'x'}},{{role:'activity',steps:[
        {{kind:'tool',code:'external_llm_opt_in_required'}}]}}]);
      const html = access.render(pending, {{disabled:false,mode:'assist'}});
      let state = {{projectId:'p',sessionId:'s',studyId:'study',code:'provider_ready_to_generate_plan',
        decision:{{family:'plan_transition',next_action_code:'provider_ready_to_generate_plan',scientific_configuration_sha256:'a'}},userMessage:'candidate question'}};
      let question = 'candidate question', starts = [], errors = [], confirms = [];
      const cb = {{ planContext:()=>state, loadStudy:async()=>({{question,data_source:{{database:'sic'}}}}),
        confirmPlan:msg=>{{confirms.push(msg);return true}},
        startPlan:async(...args)=>starts.push(args), planError:msg=>errors.push(msg) }};
      (async()=>{{
        await access.authorizePlan(cb);
        question='old ICU question'; await access.authorizePlan(cb);
        question='candidate question'; state={{...state,decision:null}}; await access.authorizePlan(cb);
        state={{...state,decision:{{family:'plan_transition',next_action_code:'provider_ready_to_generate_plan',scientific_configuration_sha256:'a'}}}};
        cb.confirmPlan=()=>{{state={{...state,decision:{{...state.decision,scientific_configuration_sha256:'b'}}}};return true}};
        await access.authorizePlan(cb);
        process.stdout.write(JSON.stringify({{pending,html,starts,errors,confirms}}));
      }})().catch(e=>{{console.error(e);process.exit(1)}});
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result["pending"] == ["provider_plan"]
    assert "data-gpi-review-plan-authorization" in result["html"]
    assert "data-gpi-action-ask-routine" not in result["html"]
    assert len(result["starts"]) == 1
    assert result["starts"][0][1]["requireConfirmedDecision"] is True
    assert len(result["errors"]) == 3
    assert "candidate question" in result["confirms"][0]


def test_plan_submit_rechecks_decision_after_async_study_load():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{}};
      eval({_read('js/screens-guided-pi-plan-actions.js')!r});
      const oldDecision = {{family:'plan_transition',next_action_code:'provider_ready_to_generate_plan',
        scientific_configuration_sha256:'a'.repeat(64),source_run_id:''}};
      let decision = oldDecision, calls = [], errors = [];
      const host = {{tr:(en,zh)=>zh,errorText:e=>String(e.message || e),regeneration:{{}},nextActions:{{}},replay:{{}},
        session:()=>({{session_id:'s',binding:{{study_context_id:'study'}},research_provider:{{provider:'api'}}}}),
        workflow:()=>({{next_action_code:'provider_ready_to_generate_plan',host_decisions:{{plan_transition:decision}}}}),
        projectId:()=> 'p',busy:()=>false,sessionIsStale:()=>false,researchSourceReady:()=>true,
        api:()=>({{loadStudyContext:async()=>{{decision={{...oldDecision,scientific_configuration_sha256:'b'.repeat(64)}};
          return {{context:{{question:'question',data_source:{{path:'/synthetic'}}}}}}}},
          startAgentRun:async body=>{{calls.push(body);return {{job_id:'j'}}}}}}),
        turnGrants:()=>[],appendMessage:()=>{{}},render:()=>{{}},setBusy:()=>{{}},setError:e=>errors.push(e),
        watchChildJob:()=>{{}},refreshSession:async()=>{{}},loadWorkflow:async()=>{{}}}};
      const actions=window.EU_GUIDED_PI_PLAN_ACTIONS.create(host);
      (async()=>{{await actions.startFormalPlanGeneration('provider_ready_to_generate_plan',
        {{requireConfirmedDecision:true,confirmedDecision:JSON.stringify(oldDecision)}});
        process.stdout.write(JSON.stringify({{calls,errors}}));}})().catch(e=>{{console.error(e);process.exit(1)}});
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result["calls"] == []
    assert result["errors"]


def test_history_replay_keeps_blocked_activity_after_its_user_turn():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{EU_HTML:{{esc:String}}}};
      eval({_read('js/screens-guided-pi-replay.js')!r});
      eval({_read('js/screens-guided-pi-transcript.js')!r});
      eval({_read('js/screens-guided-pi-access.js')!r});
      const upsert=(a,s)=>{{const i=a.steps.findIndex(x=>x.id===s.id);
        if(i>=0)a.steps[i]={{...a.steps[i],...s}};else a.steps.push(s)}};
      const owner=window.EasyICU.guidedPi.require('transcript').create({{
        activity:{{focusLatest:r=>r,startTurn:()=>{{}},finishTurn:()=>{{}}}},upsertActivityStep:upsert,
        timeMs:v=>Date.parse(v||''),resourceKey:r=>JSON.stringify(r||null),modelErrorText:x=>x,
        workflowActionCode:()=>''}});
      const session={{transcript:[
        {{role:'user',entry_id:'u1',timestamp:'2026-10-30T05:12:18.682Z',content:[{{type:'text',text:'candidate'}}]}},
        {{role:'assistant',entry_id:'a1',timestamp:'2026-10-30T05:12:28.193Z',content:[{{type:'text',text:'blocked'}}]}}
      ],conversation_replay:{{turns:[{{job_id:'j1',assistant_entry_id:'a1',status:'done',
        started_at:'2026-10-30T05:12:18Z',ended_at:'2026-10-30T05:12:29Z',events:[
          {{type:'tool_end',tool_call_id:'call',tool_name:'easyicu_request_replan',
            code:'external_llm_opt_in_required',is_error:true,at:'2026-10-30T05:12:28Z'}}]}}]}}}};
      const rows=owner.transcriptMessages(session);
      const access=window.EasyICU.guidedPi.require('access');
      const blocked=access.pending(rows);
      const activity=rows.find(r=>r.role==='activity');
      const newer=[...rows,{{role:'user',text:'new unrelated request'}}];
      process.stdout.write(JSON.stringify({{roles:rows.map(r=>r.role),blocked,
        stale:access.pending(newer),startedAt:activity.startedAt,timelineAt:activity.timelineAt}}));
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result["roles"].index("user") < result["roles"].index("activity")
    assert result["blocked"] == ["provider_plan"]
    assert result["stale"] == []
    assert result["startedAt"] < result["timelineAt"]
