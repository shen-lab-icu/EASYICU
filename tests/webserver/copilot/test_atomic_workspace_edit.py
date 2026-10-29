"""Atomic exact edits preserve old behavior and separate saved writes from checks."""
from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.tool_catalog import TOOL_MODEL_SCHEMAS
from easyicu.webserver.pi_copilot.tools import execute_tool
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace


def setup_edit(tmp_path, source="<html><p>A</p><p>B</p><p>C</p><p>D</p></html>"):
    workspace = ProjectWorkspace(tmp_path / "workspace")
    written = workspace.write_file("atomic-project", "page.html", source)
    context = ToolExecutionContext(session=PiSessionRecord(session_id="atomic-session", project_id="atomic-project", agent_mode="workspace"),
        workspace=workspace, allowed_actions=frozenset({"workspace_write"}))
    return workspace, context, written, source


def edits(*pairs):
    return [{"old_text": old, "new_text": new} for old, new in pairs]


def test_batch_commits_four_changes_once_and_static_check_has_same_digest(tmp_path, monkeypatch):
    workspace, context, before, _ = setup_edit(tmp_path)
    writes, digests = [], []
    atomic, expected = workspace._atomic_write, workspace._expected_sha256
    monkeypatch.setattr(workspace, "_atomic_write", lambda path, raw: (writes.append(raw), atomic(path, raw))[-1])
    monkeypatch.setattr(workspace, "_expected_sha256", lambda digest: (digests.append(digest), expected(digest))[-1])
    result = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": before["sha256"],
        "edits": edits((">A<", ">AA<"), (">B<", ">BB<"), (">C<", ">CC<"), (">D<", ">DD<")), "check_after": True}, context)
    assert len(writes) == len(digests) == 1
    assert result["status"] == "ok" and result["code"] == "pi_workspace_file_edited"
    assert result["details"]["replacements"] == 4
    assert result["details"]["mutation_receipt"]["ordinal"] == 1
    assert result["details"]["static_check"]["status"] == "passed"
    assert result["details"]["checked_sha256"] == result["details"]["sha256"] == hashlib.sha256(writes[0]).hexdigest()
    assert result["details"]["resource"]["checked_sha256"] == result["details"]["sha256"]
    assert result["details"]["runtime_status"] == "pending"
    assert result["details"]["scientific_evidence"] is False


def test_targets_are_resolved_against_original_text_without_replacement_cascades(tmp_path):
    workspace, _, before, _ = setup_edit(tmp_path, "AA BB")
    result = workspace.edit_file("atomic-project", "page.html", expected_sha256=before["sha256"], edits=edits(("AA", "BB"), ("BB", "CC")))
    assert workspace.read_file("atomic-project", "page.html")["text"] == "BB CC"
    assert result["replacements"] == 2


@pytest.mark.parametrize("source,replacements", [
    ("AA BB", edits(("AA", "new"), ("missing", "later"))),
    ("AA AA BB", edits(("AA", "new"))),
    ("alpha", edits(("alpha", "new"), ("alp", "later"))),
    ("alpha", edits(("alpha", "new"), ("alpha", "again"))),
    ("aaa", edits(("aa", "new"))),
])
def test_any_missing_ambiguous_overlapping_or_duplicate_target_leaves_original_bytes(tmp_path, monkeypatch, source, replacements):
    workspace, _, before, _ = setup_edit(tmp_path, source)
    monkeypatch.setattr(workspace, "_atomic_write", lambda *args: (_ for _ in ()).throw(AssertionError("partial write")))
    with pytest.raises(PiCopilotError):
        workspace.edit_file("atomic-project", "page.html", expected_sha256=before["sha256"], edits=replacements)
    assert workspace.read_file("atomic-project", "page.html")["sha256"] == before["sha256"]


@pytest.mark.parametrize("changes", [[], edits(*[("A", "B")] * 17), [{"old_text": "A"}],
    [{"old_text": "", "new_text": "x"}], [{"old_text": "A", "new_text": "x", "extra": True}],
    [{"old_text": 1, "new_text": "x"}], [{"old_text": "A", "new_text": "x" * 120001}]])
def test_batch_shape_and_item_limits_are_checked_before_writing(tmp_path, changes):
    workspace, _, before, _ = setup_edit(tmp_path)
    with pytest.raises(PiCopilotError):
        workspace.edit_file("atomic-project", "page.html", expected_sha256=before["sha256"], edits=changes)
    assert workspace.read_file("atomic-project", "page.html")["sha256"] == before["sha256"]


def test_mutual_modes_bad_boolean_missing_digest_and_stale_digest_leave_original_unchanged(tmp_path):
    workspace, context, before, _ = setup_edit(tmp_path)
    for extra in [{"old_text": "A", "edits": edits(("B", "C"))}, {"edits": edits(("A", "B")), "check_after": "true"},
                  {"edits": edits(("A", "B")), "expected_sha256": "a" * 64}]:
        with pytest.raises(PiCopilotError):
            execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": before["sha256"], **extra}, context)
    with pytest.raises(PiCopilotError):
        workspace.edit_file("atomic-project", "page.html", edits=edits(("A", "B")))
    assert workspace.read_file("atomic-project", "page.html")["sha256"] == before["sha256"]


def test_unicode_output_byte_limit_is_not_replaced_by_character_limit(tmp_path):
    workspace, _, before, _ = setup_edit(tmp_path, "A B C")
    with pytest.raises(PiCopilotError) as caught:
        workspace.edit_file("atomic-project", "page.html", expected_sha256=before["sha256"], edits=edits(("A", "😀" * 70000)))
    assert caught.value.code == "pi_workspace_file_too_large"
    assert workspace.read_file("atomic-project", "page.html")["sha256"] == before["sha256"]


def test_check_failure_keeps_successful_write_and_exposes_no_checked_resource(tmp_path):
    workspace, context, before, _ = setup_edit(tmp_path)
    invalid = '<script type="application/json" data-easyicu-runtime-qa>{bad json}</script>'
    result = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": before["sha256"],
        "edits": edits(("<p>A</p>", invalid)), "check_after": True}, context)
    assert result["status"] == "ok" and result["code"] == "pi_workspace_file_edited"
    assert result["details"]["static_check"]["valid"] is False
    assert result["details"]["static_check"]["code"] == "pi_workspace_runtime_contract_invalid"
    assert "resource" not in result["details"] and "checked_sha256" not in result["details"]
    assert workspace.read_file("atomic-project", "page.html")["sha256"] == result["details"]["sha256"] != before["sha256"]
    assert result["details"]["mutation_receipt"]["ordinal"] == 1


def test_html_check_never_executes_scripts_and_mismatched_check_digest_is_not_ready(tmp_path, monkeypatch):
    workspace, context, before, _ = setup_edit(tmp_path)
    monkeypatch.setattr("easyicu.webserver.pi_copilot.workspace.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("executed script")))
    result = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": before["sha256"],
        "edits": edits(("<p>A</p>", "<script>throw new Error('not executed')</script>")), "check_after": True}, context)
    assert result["details"]["static_check"]["valid"] is True
    monkeypatch.setattr(workspace, "check_file", lambda *args: {"valid": True, "checked_sha256": "b" * 64})
    result = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": result["details"]["sha256"],
        "edits": edits((">B<", ">BB<")), "check_after": True}, context)
    assert result["details"]["static_check"]["status"] == "unconfirmed"
    assert "checked_sha256" not in result["details"] and "resource" not in result["details"]


def test_legacy_single_edit_and_its_deletion_behavior_are_unchanged(tmp_path):
    workspace, context, before, _ = setup_edit(tmp_path, "aaa")
    result = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": before["sha256"], "old_text": "aa", "new_text": "b"}, context)
    assert workspace.read_file("atomic-project", "page.html")["text"] == "ba"
    assert result["summary"] == "Edited page.html with one exact replacement."
    assert result["details"]["replacements"] == 1 and "static_check" not in result["details"]
    result = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": result["details"]["sha256"], "old_text": "b"}, context)
    assert workspace.read_file("atomic-project", "page.html")["text"] == "a"


def test_concurrent_batches_with_same_digest_have_only_one_complete_winner(tmp_path):
    workspace, _, before, _ = setup_edit(tmp_path, "A B")
    def replace(label):
        try:
            workspace.edit_file("atomic-project", "page.html", expected_sha256=before["sha256"], edits=edits(("A", label + "1"), ("B", label + "2")))
            return label
        except PiCopilotError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(replace, ["x", "y"]))
    assert results.count("pi_workspace_file_changed") == 1
    assert workspace.read_file("atomic-project", "page.html")["text"] in {"x1 x2", "y1 y2"}


def test_batch_still_spends_one_of_eight_original_workspace_mutation_slots(tmp_path):
    workspace, context, before, _ = setup_edit(tmp_path, "value-0")
    digest = before["sha256"]
    for index in range(8):
        result = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": digest, "edits": edits((f"value-{index}", f"value-{index+1}"))}, context)
        digest = result["details"]["sha256"]
        assert result["details"]["mutation_receipt"]["ordinal"] == index + 1
    refused = execute_tool("easyicu_edit_project_file", {"file": "page.html", "expected_sha256": digest, "edits": edits(("value-8", "value-9"))}, context)
    assert refused["code"] == "pi_workspace_mutation_limit_reached"
    assert workspace.read_file("atomic-project", "page.html")["text"] == "value-8"


def test_catalog_schema_accepts_legacy_and_batch_but_not_mixed_or_empty_batch():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(TOOL_MODEL_SCHEMAS["easyicu_edit_project_file"])
    base = {"file": "page.html", "expected_sha256": "a" * 64}
    assert validator.is_valid({**base, "old_text": "A", "new_text": "B"})
    assert validator.is_valid({**base, "edits": edits(("A", "B")), "check_after": True})
    assert not validator.is_valid({**base, "old_text": "A", "edits": edits(("A", "B"))})
    assert not validator.is_valid({**base, "edits": []})


@pytest.mark.requires_node
@pytest.mark.parametrize("tool_name", ["easyicu_edit_project_file", "easyicu_write_project_file"])
def test_actual_sdk_preparation_rejects_raw_coercion_cases_before_host_execution(tool_name):
    from tests.support.node import run_node

    root = Path(__file__).resolve().parents[3]
    app = root / "src/easyicu/webserver/pi_copilot/node_app"
    node = shutil.which("node")
    if not node or not (app / "node_modules").exists():
        pytest.skip("Pinned SDK is unavailable")
    script = r"""
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {Check} from 'typebox/value';
import {Type} from 'typebox';
import {defineTool} from '@earendil-works/pi-coding-agent';
import {wrapToolDefinition} from './node_modules/@earendil-works/pi-coding-agent/dist/core/tools/tool-definition-wrapper.js';
import {runAgentLoop} from './node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-agent-core/dist/agent-loop.js';
import assert from 'node:assert/strict';
const source=readFileSync('./src/main.mjs','utf8');
const toolName=__TOOL_NAME__;
const line=source.split('\n').find(line=>line.includes('hostTool(sessionId, { name: "'+toolName+'"'));
const schema=JSON.parse(readFileSync('../tool_catalog.json','utf8')).tools.find(row=>row.name===toolName).model_schema;
const guardStart=source.indexOf('function strictCatalogArguments(');
const guard=source.slice(guardStart,source.indexOf('\nconst RESEARCH_TOOL_NAMES',guardStart));
let executed=0;
const [definition]=runInNewContext(guard+'\n['+line.trim()+']',{sessionId:'strict-artifact',Check,
  TOOL_CATALOG_BY_NAME:new Map([[toolName,{model_schema:schema}]]),
  catalogParameters:()=>Type.Unsafe(schema),hostTool:(_id,definition)=>definition});
const tool=wrapToolDefinition(defineTool({...definition,execute:async()=>{executed++;return {content:[{type:'text',text:'saved'}],details:{}};}}));
const base={file:'page.html',expected_sha256:'a'.repeat(64)};
const cases=toolName==='easyicu_edit_project_file'?[
  [{...base,old_text:'a',new_text:'b'},true],
  [{...base,edits:[{old_text:'a',new_text:'b'}],check_after:true},true],
  [{...base,old_text:12,new_text:'b'},false],
  [{...base,old_text:'a',new_text:false},false],
  [{...base,edits:[{old_text:12,new_text:'b'}]},false],
  [{...base,edits:[{old_text:'a',new_text:false}]},false],
  [{...base,edits:[{old_text:'a',new_text:'b',extra:true}]},false],
  [{...base,edits:[{old_text:'a',new_text:'b'}],check_after:'false'},false],
  [{...base,old_text:'a',new_text:'b',check_after:true,preview_after:true},true],
  [{...base,old_text:'a',new_text:'b',preview_after:true},false],
  [{...base,old_text:'a',new_text:'b',check_after:false,preview_after:true},false],
  [{...base,old_text:'a',new_text:'b',check_after:true,preview_after:'false'},false],
]:[
  [{file:'page.html',content:'<p>new</p>'},true],
  [{file:'page.html',content:'<p>new</p>',check_after:true,preview_after:true},true],
  [{file:'page.html',content:12},false],
  [{file:'page.html',content:'<p>new</p>',check_after:'true'},false],
  [{file:'page.html',content:'<p>new</p>',check_after:true,preview_after:'true'},false],
  [{file:'page.html',content:'<p>new</p>',preview_after:true},false],
  [{file:'page.html',content:'<p>new</p>',check_after:false,preview_after:true},false],
  [{file:'page.html',content:'<p>new</p>',extra:true},false],
];
for (const [args,valid] of cases){
  const before=executed,results=[];
  const assistant=(content,stopReason)=>({role:'assistant',content,stopReason,timestamp:Date.now(),
    api:'openai-completions',provider:'local',model:'test',usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
  const responses=[assistant([{type:'toolCall',id:'strict-call',name:tool.name,arguments:args}],'toolUse'),assistant([{type:'text',text:'done'}],'stop')];
  await runAgentLoop([{role:'user',content:[{type:'text',text:'edit'}],timestamp:Date.now()}],
    {systemPrompt:'test',messages:[],tools:[tool]},
    {model:{api:'openai-completions',provider:'local',id:'test'},convertToLlm:async rows=>rows},
    async event=>{if(event.type==='tool_execution_end')results.push(event);},undefined,
    async()=>{const message=responses.shift();return {async *[Symbol.asyncIterator](){yield {type:'start',partial:message};yield {type:'done',reason:message.stopReason,message};},result:async()=>message};});
  assert.equal(executed-before,valid?1:0,'raw arguments must be checked before SDK coercion');
  assert.equal(results[0].isError,!valid);
}
console.log(JSON.stringify({cases:cases.length,valid_host_executions:executed,invalid_host_executions:0}));
"""
    completed = run_node(node, script.replace("__TOOL_NAME__", json.dumps(tool_name)), module=True, cwd=app, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    expected = (12, 3) if tool_name == "easyicu_edit_project_file" else (8, 2)
    assert json.loads(completed.stdout) == {"cases": expected[0], "valid_host_executions": expected[1], "invalid_host_executions": 0}
