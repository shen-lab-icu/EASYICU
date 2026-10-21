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
    return json.loads(run_node(node, harness, check=True).stdout)


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
