"""The offer client transports source choices and typed backend errors."""

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node

JS = Path(__file__).parents[3] / "src/easyicu/webserver/static/js"


def _node(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    harness = """
      global.window = global;
      global.document = {readyState:'loading', addEventListener:()=>{}, querySelector:()=>null};
    """ + (JS / "api.js").read_text() + script
    result = run_node(node, harness)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_offer_client_encodes_coordinates_and_posts_the_exact_source_id() -> None:
    result = _node("""
      const calls = [];
      global.fetch = async (url, options) => {
        calls.push({url, options});
        return {ok:true,json:async()=>({database:'eicu',recommended:{source_id:'src_registered'}})};
      };
      (async () => {
        const offer = await EU_API.loadPiCopilotRegisteredSourceOffer('session/a', 'project & b', 'eicu');
        await EU_API.authorizePiCopilotDataSource('session/a', {
          project_id:'project & b',action:'use_registered_source',database:'eicu',source_id:offer.recommended.source_id,
        });
        console.log(JSON.stringify(calls));
      })().catch(e=>{console.error(e);process.exit(1);});
    """)
    assert result[0]["url"] == (
        "/api/copilot/pi/sessions/session%2Fa/registered-source-offer?project_id=project+%26+b&database=eicu")
    assert result[1]["url"] == "/api/copilot/pi/sessions/session%2Fa/data-source-authorization"
    assert result[1]["options"]["method"] == "POST"
    assert json.loads(result[1]["options"]["body"]) == {
        "project_id": "project & b", "action": "use_registered_source", "database": "eicu",
        "source_id": "src_registered",
    }


def test_offer_and_authorize_clients_preserve_backend_error_codes() -> None:
    result = _node("""
      global.fetch = async () => ({ok:false,status:409,json:async()=>({detail:{
        error:'pi_data_source_not_offered',reason:'Refresh this choice.',details:{database:'eicu'},
      }})});
      (async () => {
        const errors = [];
        for (const action of [
          () => EU_API.loadPiCopilotRegisteredSourceOffer('s','p','eicu'),
          () => EU_API.authorizePiCopilotDataSource('s',{source_id:'stale'}),
        ]) { try { await action(); } catch(e) { errors.push({code:e.code,status:e.status,details:e.details}); } }
        console.log(JSON.stringify(errors));
      })().catch(e=>{console.error(e);process.exit(1);});
    """)
    assert result == [{"code": "pi_data_source_not_offered", "status": 409, "details": {"database": "eicu"}}] * 2


@pytest.mark.parametrize("mode", ["reuse_project_source", "select_registered_source"])
def test_confirmed_registered_choice_continues_planning_once(mode) -> None:
    source = (JS / "screens-guided-pi-modules.js").read_text() + (JS / "screens-guided-pi-data-binding.js").read_text()
    result = _node(source + """
      const calls = [];
      let session = {session_id:'s'};
      const owner = EasyICU.guidedPi.require('dataBinding').create({
        api:()=>({authorizePiCopilotDataSource:async()=>({session:{session_id:'s',data_source_authorization:{
          status:'confirmed',confirmation_mode:MODE,extraction_scope:'study_required',
        }}})}), session:()=>session,projectId:()=>'p',busy:()=>false,
        setSession:value=>session=value,rememberSession:()=>{},render:()=>{},setError:()=>{},
        loadWorkflow:async()=>calls.push('workflow'),continueAfterDataSourceConfirmation:async()=>calls.push('plan'),
      });
      (async()=>{await owner.authorizeDataSource('use_registered_source');console.log(JSON.stringify(calls));})();
    """.replace("MODE", json.dumps(mode)))
    assert result == ["workflow", "plan"]


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize(("status", "detail", "sentences"), [
    (422, [{"loc": ["body", "database"], "msg": "Input should be a profile key"}],
     {"en": "This step did not complete (HTTP 422).", "zh": "这一步没有完成（HTTP 422）。"}),
    (409, {"error": "pi_data_source_not_offered", "reason": "private transport detail"},
     {"en": "That export is no longer offered. Reload the page and choose again.",
      "zh": "这份数据已不在可选范围内，请刷新页面后重新选择。"}),
    (409, {"error": "pi_data_source_database_mismatch", "reason": "private transport detail"},
     {"en": "That export is not of the database your question names.",
      "zh": "这份数据不是你的问题指定的数据库。"}),
    (500, None,
     {"en": "The EasyICU server hit an error on this step (HTTP 500).",
      "zh": "EasyICU 服务器处理这一步时出错了（HTTP 500）。"}),
])
def test_a_failed_registered_choice_renders_the_error_owner_sentence(language, status, detail, sentences) -> None:
    owners = "".join((JS / name).read_text() for name in (
        "screens-guided-pi-modules.js", "html-escape.js", "screens-guided-pi-error-text.js",
        "screens-guided-pi-data-binding.js", "screens-guided-pi-session-view.js",
    ))
    result = _node(owners + """
      const tr = (en, zh) => LANGUAGE === 'zh' ? zh : en;
      const empty = () => '';
      const state = {session:{session_id:'s',data_source_authorization:{status:'pending'}},
        messages:[],workflowReceipts:[],draft:'',error:''};
      const originalSession = state.session;
      const modules = EasyICU.guidedPi;
      const errorText = modules.require('errorText').create({tr,staticPreview:()=>false}).errorText;
      const view = modules.require('sessionView').create({
        state,modules,tr,esc:EU_HTML.esc,iconHtml:empty,projectId:()=>'p',
        projectTitle:empty,navigationSessionTitle:empty,sessionIsStale:()=>false,agentMode:()=>'research',
        transcript:{latestTurnCompletedIdeaExploration:()=>false},runFiles:{notice:empty},
        activity:{renderTimeline:empty},
        runOutcome:{collection:()=>[]},aside:{layoutOptions:()=>({})},
        header:{render:empty,renderModelControl:empty},
        studyWorkspace:{renderReference:empty,renderSkillReference:empty,renderMaterials:empty,
          renderSkillPicker:empty,renderAccessMode:empty},
      });
      let html = '', requests = [];
      global.fetch = async (url, options) => {
        requests.push({url,body:JSON.parse(options.body)});
        return {ok:false,status:STATUS,json:async()=>({detail:DETAIL})};
      };
      const binding = modules.require('dataBinding').create({
        api:()=>EU_API,session:()=>state.session,projectId:()=>'p',busy:()=>false,errorText,
        setError:value=>state.error=value,render:()=>{html=view.sessionPanel();},
        setSession:()=>{throw new Error('A refusal must not replace the session');},
      });
      (async()=>{
        await binding.authorizeDataSource('use_registered_source',{database:'eicu_demo',sourceId:'src_demo'});
        console.log(JSON.stringify({html,error:state.error,requests,unchanged:state.session===originalSession}));
      })().catch(e=>{console.error(e);process.exit(1);});
    """.replace("LANGUAGE", json.dumps(language)).replace("STATUS", json.dumps(status)).replace("DETAIL", json.dumps(detail)))
    assert 'class="gpi-error" role="alert"' in result["html"]
    assert sentences[language] in result["error"]
    assert sentences[language] in result["html"]
    if status == 422:
        assert result["error"] == sentences[language]
    assert "private transport detail" not in result["html"]
    assert "Input should be a profile key" not in result["html"]
    assert result["unchanged"] is True
    assert result["requests"] == [{
        "url": "/api/copilot/pi/sessions/s/data-source-authorization",
        "body": {"project_id": "p", "action": "use_registered_source", "database": "eicu_demo", "source_id": "src_demo"},
    }]
