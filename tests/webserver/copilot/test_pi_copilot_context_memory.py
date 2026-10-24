"""Synthetic checks for owner state that survives Pi context compaction."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from tests.support.node import run_node

pytestmark = pytest.mark.requires_node


MODULE = (
    Path(__file__).resolve().parents[3]
    / "src/easyicu/webserver/pi_copilot/node_app/src/research-context.mjs"
)


def _check(program: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    harness = f"""
      import assert from 'node:assert/strict';
      import {{ ResearchContextMemory }} from {json.dumps(MODULE.as_uri())};
      const receipt = (revision = 2, source = 'source-A') => ({{
        status: 'ok', code: 'easyicu_research_workflow_projected',
        owner: 'easyicu.webserver.pi_copilot.workflow',
        authority: {{study_context_id: 'study-1', study_revision: revision, run_id: 'run-1', grant: 'secret'}},
        details: {{workflow: {{
          schema_version: 'easyicu.pi-research-workflow/1',
          current_stage: 'plan', next_action_code: 'human_plan_review_required',
          plan_execution_ready: false,
          plan_review_summary: {{plan_sha256: 'a'.repeat(64), approved: false}},
          study_setup_receipt: {{schema_version: 'easyicu.pi-study-setup-receipt/1',
            study_context_id: 'study-1', revision, configuration: {{
              question: 'Compare mortality between two groups', data_source: {{source_id: source}},
            }}}},
          approval_token: 'secret',
        }}}},
      }});
      const state = (message) => JSON.parse(message.content.split('\\n').at(-1));
      {program}
    """
    result = run_node(node, harness, module=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_latest_owner_state_survives_compaction_without_promoting_summary() -> None:
    _check("""
      const memory = new ResearchContextMemory({now: () => 10});
      memory.beginTurn([receipt()]);
      const summary = {role: 'compactionSummary', summary: 'The plan is approved for source-OLD', timestamp: 1};
      const messages = [summary, {role: 'user', content: 'Continue', timestamp: 2}];
      const result = memory.extension().handlers.get('context')[0]({messages}).messages;
      assert.equal(result.length, 3);
      assert.deepEqual(result[0], summary);
      assert.equal(result[1].display, false);
      assert.equal(state(result[1]).workflow.plan_execution_ready, false);
      assert.equal(state(result[1]).workflow.study_setup_receipt.configuration.data_source.source_id, 'source-A');
      assert.equal(state(result[1]).workflow.plan_review_summary.plan_sha256, 'a'.repeat(64));
      assert.ok(!result[1].content.includes('secret'));
      assert.equal(memory.context(result).messages.length, 3);
      memory.endTurn();
      assert.equal(memory.context(result).messages.length, 2);
      memory.beginTurn([]);
      assert.equal(state(memory.context(result).messages[1]).status, 'unavailable');
      assert.equal(new ResearchContextMemory().context(result).messages.length, 2);
    """)


def test_mutation_invalidates_and_only_matching_fresh_owner_restores_context() -> None:
    _check("""
      const memory = new ResearchContextMemory();
      const original = receipt();
      memory.beginTurn([original]);
      original.details.workflow.study_setup_receipt.configuration.data_source.source_id = 'tampered';
      assert.equal(state(memory.context([]).messages[0]).workflow.study_setup_receipt.configuration.data_source.source_id, 'source-A');
      memory.invalidate('owner_state_changed');
      assert.equal(state(memory.context([]).messages[0]).reason, 'owner_state_changed');
      const updated = receipt(3, 'source-B');
      updated.code = 'study_context_updated';
      updated.owner = 'easyicu.webserver.study_contexts';
      updated.authority.study_revision = 2;
      updated.details.rebind_required = true;
      updated.details.host_rebind_after_turn = true;
      updated.details.study = {id: 'study-1', revision: 3};
      assert.equal(memory.captureReceipt(updated), true);
      assert.equal(state(memory.context([]).messages[0]).status, 'pending_rebind');
      assert.equal(state(memory.context([]).messages[0]).binding_state, 'rebind_required');
      assert.equal(state(memory.context([]).messages[0]).receipt_binding.study_revision, 2);
      assert.equal(state(memory.context([]).messages[0]).workflow.study_setup_receipt.revision, 3);
      assert.equal(memory.captureReceipt(receipt(2, 'source-OLD')), false);
      assert.equal(state(memory.context([]).messages[0]).binding_state, 'rebind_required');
      assert.equal(memory.captureReceipt(receipt(3, 'source-B')), false);
      memory.endTurn();
      memory.beginTurn([receipt(3, 'source-B')]);
      assert.equal(state(memory.context([]).messages[0]).binding_state, 'current');
      assert.equal(memory.captureReceipt(receipt(2, 'source-OLD')), false);
      assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
      assert.equal(state(memory.context([]).messages[0]).reason, 'stale_owner_receipt');
      const foreign = receipt(4);
      foreign.authority.study_context_id = 'other-study';
      foreign.details.workflow.study_setup_receipt.study_context_id = 'other-study';
      assert.equal(memory.captureReceipt(foreign), false);
      assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
      assert.equal(state(memory.context([]).messages[0]).reason, 'invalid_owner_receipt');
      memory.endTurn();
      assert.equal(memory.captureReceipt(receipt()), false);
    """)


@pytest.mark.parametrize("rejection", ["oversize", "invalid_configuration"])
def test_a_rejected_newer_revision_prevents_a_late_older_receipt(rejection):
    _check(f"""
      const memory = new ResearchContextMemory({{maxBytes: 4096}});
      memory.beginTurn([receipt(2)]);
      const newer = receipt(4);
      if ({json.dumps(rejection)} === 'oversize') {{
        newer.details.workflow.study_setup_receipt.configuration.question = '重'.repeat(3000);
      }} else {{
        newer.details.workflow.study_setup_receipt.configuration = [];
      }}
      assert.equal(memory.captureReceipt(newer), false);
      assert.equal(memory.captureReceipt(receipt(3, 'source-STALE')), false);
      assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
      assert.ok(!memory.context([]).messages[0].content.includes('source-STALE'));
      assert.equal(memory.captureReceipt(receipt(4, 'source-FRESH')), true);
    """)


def test_incomplete_receipts_state_unknown_fields_and_a_capture_time():
    _check("""
      const memory = new ResearchContextMemory({now: () => 10});
      const partial = receipt();
      delete partial.details.workflow.plan_execution_ready;
      memory.beginTurn([partial]);
      const projected = state(memory.context([]).messages[0]);
      assert.equal(projected.workflow.plan_execution_ready, null);
      assert.equal(projected.workflow.missing_setup_fields, null);
      assert.equal(projected.captured_at, 10);
    """)


def test_a_lost_mutation_fences_reads_started_before_or_during_the_write():
    _check("""
      const memory = new ResearchContextMemory();
      memory.beginTurn([receipt(2)]);
      const before = memory.startTool();
      const write = memory.startTool(true);
      const during = memory.startTool();
      memory.finishTool(write); // Host mutation may have saved, but its reply was lost.
      assert.equal(memory.finishTool(before, receipt(2)), false);
      assert.equal(memory.finishTool(during, receipt(2)), false);
      assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
      const fresh = memory.startTool();
      assert.equal(memory.finishTool(fresh, receipt(3)), true);
      const previousTurn = memory.startTool();
      memory.endTurn(); memory.beginTurn([receipt(3)]);
      assert.equal(memory.finishTool(previousTurn, receipt(3)), false);
    """)


def test_invalid_or_oversize_receipt_is_omitted_as_a_whole() -> None:
    _check("""
      const memory = new ResearchContextMemory({maxBytes: 4096});
      memory.beginTurn([receipt()]);
      for (const mutate of [
        (r) => r.authority.study_revision = 1,
        (r) => r.status = 'blocked',
        (r) => r.owner = 'assistant',
        (r) => r.details.workflow.study_setup_receipt.configuration = [],
        (r) => r.details.workflow.study_setup_receipt.revision = -1,
      ]) {
        const invalid = receipt(); mutate(invalid);
        assert.equal(memory.captureReceipt(invalid), false);
        assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
      }
      const huge = receipt();
      huge.details.workflow.study_setup_receipt.configuration.question = '重'.repeat(3000);
      assert.equal(memory.captureReceipt(huge), false);
      const text = memory.context([]).messages[0].content;
      assert.equal(state({content: text}).reason, 'owner_context_over_budget');
      assert.ok(Buffer.byteLength(text) <= 4096);
      assert.ok(!text.includes('重'));
      assert.throws(() => new ResearchContextMemory({maxBytes: 999}), /maxBytes/);
    """)


def test_context_retains_current_prompt_and_final_tool_pair_for_host_finalizer() -> None:
    _check("""
      const memory = new ResearchContextMemory();
      memory.beginTurn([receipt(4, 'source-NEW')]);
      const language = '\\n\\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\\nChinese';
      const owner = '\\n\\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\\n';
      const transition = '\\n\\n[EASYICU_HOST_TRANSITION_V1]\\nKeep this action';
      const user = 'User quoted ' + owner + '[this is my question]';
      const original = [
        {role: 'user', content: [{type:'text', text: user + language + owner + '[source-OLD]' + transition}]},
        {role: 'custom', customType: 'easyicu_host_transition', content: 'Confirmed' + language + owner + '[source-NEW]' + transition},
        {role: 'assistant', content: [{type:'toolCall', name:'easyicu_inspect_workflow', id:'call-1'}]},
        {role: 'toolResult', content: [{type:'text', text:'Original tool evidence'}]},
      ];
      const saved = JSON.stringify(original);
      const result = memory.context(original).messages;
      assert.equal(result[0].content[0].text, user + language + transition);
      assert.equal(result[2].content, original[1].content);
      assert.deepEqual(result.slice(-2), original.slice(-2));
      assert.equal(JSON.stringify(original), saved);
      assert.ok(result[1].content.includes('source-NEW'));
    """)


def test_pinned_sdk_reinjects_fresh_owner_state_after_real_compaction(tmp_path: Path) -> None:
    app_dir = MODULE.parent.parent
    sdk = app_dir / "node_modules/@earendil-works/pi-coding-agent/dist/index.js"
    ai = app_dir / "node_modules/@earendil-works/pi-ai/dist/index.js"
    if not sdk.is_file() or not ai.is_file():
        pytest.skip("Pinned Pi Node dependencies are not installed")
    package = json.loads((sdk.parent.parent / "package.json").read_text())
    assert package["version"] == "0.84.1"
    _check(f"""
      import {{createAgentSession, createExtensionRuntime, convertToLlm, ModelRuntime,
        SessionManager, SettingsManager}} from {json.dumps(sdk.as_uri())};
      import {{InMemoryCredentialStore, InMemoryModelsStore,
        createAssistantMessageEventStream}} from {json.dumps(ai.as_uri())};
      import {{ShellBudgetGuard}} from {json.dumps((MODULE.parent / 'shell-budget.mjs').as_uri())};
      import {{hostPostToolFinalization}} from {json.dumps((MODULE.parent / 'post-tool-finalization.mjs').as_uri())};
      let networkCalls = 0;
      globalThis.fetch = async () => {{ networkCalls++; throw new Error('Network forbidden in SDK smoke'); }};
      const cwd = {json.dumps(str(tmp_path))};
      const memory = new ResearchContextMemory();
      const manager = SessionManager.inMemory(cwd);
      const runtime = await ModelRuntime.create({{
        credentials: new InMemoryCredentialStore(), modelsStore: new InMemoryModelsStore(),
        modelsPath: null, refreshOnCreate: false, allowModelNetwork: false,
      }});
      runtime.registerProvider('context-test', {{
        api: 'openai-completions', baseUrl: 'https://invalid.example',
        models: [{{id:'offline', name:'offline', reasoning:false, input:['text'],
          contextWindow:200000, maxTokens:1024,
          cost:{{input:0,output:0,cacheRead:0,cacheWrite:0}}}}],
      }});
      await runtime.setRuntimeApiKey('context-test', 'offline-synthetic-key');
      const model = runtime.getModel('context-test', 'offline');
      assert.ok(model);
      const loader = {{
        getExtensions: () => ({{extensions:[memory.extension()], errors:[], runtime:createExtensionRuntime()}}),
        getSkills: () => ({{skills:[],diagnostics:[]}}),
        getPrompts: () => ({{prompts:[],diagnostics:[]}}),
        getThemes: () => ({{themes:[],diagnostics:[]}}),
        getAgentsFiles: () => ({{agentsFiles:[]}}),
        getSystemPrompt: () => 'Offline synthetic test.', getSystemPromptSource: () => undefined,
        getAppendSystemPrompt: () => [], getAppendSystemPromptSources: () => [],
        extendResources: () => {{}}, reload: async () => {{}},
      }};
      const {{session}} = await createAgentSession({{
        cwd, agentDir:cwd, modelRuntime:runtime, model, thinkingLevel:'off',
        resourceLoader:loader, sessionManager:manager, noTools:'all', tools:[],
        settingsManager:SettingsManager.inMemory({{
          retry:{{enabled:false,maxRetries:0}},
          compaction:{{enabled:false,reserveTokens:512,keepRecentTokens:128}},
        }}),
      }});
      const guardOptions = {{tokenBudget:1000000, maxOutputTokens:1024,
        maxProviderCallsPerMessage:10, maxProviderCallsPerSession:20,
        consumedTokens:() => session.getSessionStats().tokens.total}};
      const guard = new ShellBudgetGuard(guardOptions);
      const seen = [];
      let compacting = false;
      session.agent.streamFunction = (selected, context, options = {{}}) => {{
        guard.authorize(context, options);
        const budgetReceipt = guard.receipt();
        manager.appendCustomEntry(budgetReceipt.schema_version, budgetReceipt);
        seen.push({{compacting, context:structuredClone(context)}});
        const stream = createAssistantMessageEventStream();
        const message = {{
          role:'assistant', api:selected.api, provider:selected.provider, model:selected.id,
          content:[{{type:'text',text: compacting
            ? 'Historical summary falsely says source-OLD is approved. No current authority.'
            : 'Synthetic response. '.repeat(80)}}],
          stopReason:'stop', timestamp:Date.now(),
          usage:{{input:100,output:50,cacheRead:0,cacheWrite:0,totalTokens:150,
            cost:{{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}}},
        }};
        stream.push({{type:'start',partial:message}});
        stream.push({{type:'done',reason:'stop',message}});
        stream.end(message);
        return stream;
      }};
      try {{
        guard.beginMessage(); memory.beginTurn([receipt(2)]);
        await session.prompt('Synthetic initial question. '.repeat(80));
        assert.ok(JSON.stringify(seen[0].context).includes('easyicu.pi-current-research-state/1'));
        assert.ok(JSON.stringify(seen[0].context).includes('source-A'));
        const before = guard.state().provider_calls;
        compacting = true;
        await session.compact();
        compacting = false;
        assert.ok(guard.state().provider_calls > before, 'Compaction must cross the same budget guard');
        assert.ok(seen.some((call) => call.compacting));
        assert.ok(!JSON.stringify(seen.filter((call) => call.compacting))
          .includes('easyicu.pi-current-research-state/1'), 'Transient context is not a compaction input');
        assert.ok(manager.getEntries().some((entry) => entry.type === 'compaction'));
        memory.endTurn(); guard.endMessage();
        guard.beginMessage(); memory.beginTurn([receipt(3, 'source-CURRENT')]);
        await session.prompt('Continue the synthetic question');
        const last = seen.at(-1).context;
        assert.ok(JSON.stringify(last).includes('source-OLD'), 'Hostile historical summary stays visibly historical');
        const projected = last.messages.find((message) => JSON.stringify(message.content).includes('easyicu.pi-current-research-state/1'));
        assert.ok(projected);
        const text = projected.content.filter((part) => part.type === 'text').map((part) => part.text).join('');
        assert.equal(state({{content:text}}).workflow.study_setup_receipt.revision, 3);
        assert.equal(state({{content:text}}).workflow.study_setup_receipt.configuration.data_source.source_id, 'source-CURRENT');
        assert.equal(last.messages.at(-1).content[0].text, 'Continue the synthetic question');
        const toolMessages = [
          {{role:'user', content:[{{type:'text',text:'Save this synthetic question'}}]}},
          {{role:'assistant',content:[{{type:'toolCall',id:'call-1',name:'easyicu_update_study_context',
            arguments:{{question:'Synthetic question'}}}}]}},
          {{role:'toolResult',toolCallId:'call-1',toolName:'easyicu_update_study_context',isError:false,content:[],
            details:{{status:'ok',code:'study_context_updated',details:{{workflow:{{
              next_action_code:'provider_ready_to_generate_plan',missing_setup_fields:['outcome'],
              study_setup_receipt:{{configuration:{{data_source:{{database:'miiv'}}}}}},
            }}}}}}}},
        ];
        const baseline = hostPostToolFinalization(model, {{messages:toolMessages}}, 'zh');
        const withMemory = hostPostToolFinalization(model,
          {{messages:convertToLlm(memory.context(toolMessages).messages)}}, 'zh');
        assert.ok(baseline && withMemory, 'Context injection must preserve deterministic host finalization');
        assert.deepEqual((await withMemory.result()).content, (await baseline.result()).content);
        const entries = manager.getEntries();
        assert.ok(!JSON.stringify(entries).includes('easyicu.pi-current-research-state/1'), 'Transient owner state must not be persisted');
        const restored = new ShellBudgetGuard({{...guardOptions,persistedEntries:entries}});
        assert.equal(restored.state().provider_calls, guard.state().provider_calls);
        assert.ok(restored.state().provider_calls >= 3);
        assert.equal(networkCalls, 0);
      }} finally {{
        memory.endTurn(); guard.endMessage(); session.dispose();
      }}
    """)


def test_real_host_tool_invalidates_before_request_and_captures_only_fresh_receipts() -> None:
    _check(f"""
      import {{readFileSync}} from 'node:fs';
      const main = readFileSync({json.dumps(str(MODULE.parent / 'main.mjs'))}, 'utf8');
      const body = main.slice(main.indexOf('function hostTool('), main.indexOf('function validateToolDefinitions('));
      const memory = new ResearchContextMemory();
      memory.beginTurn([receipt(2)]);
      const records = new Map([['s1', {{researchContext:memory, session:{{setActiveToolsByName:()=>{{}}}}}}]]);
      const catalog = new Map([
        ['easyicu_update_study_context', {{model_schema:null,host_mutating:true,execution_mode:'sequential'}}],
        ['easyicu_inspect_workflow', {{model_schema:null,host_mutating:false,execution_mode:'parallel'}}],
      ]);
      const request = async (session, name) => {{
        assert.equal(session, 's1');
        if (name === 'easyicu_update_study_context') {{
          assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
          throw Object.assign(new Error('Synthetic lost reply'), {{code:'lost_reply'}});
        }}
        return receipt(3, 'source-FRESH');
      }};
      const hostTool = new Function('TOOL_CATALOG_BY_NAME','CATALOG_PARAMETERS','defineTool',
        'sessions','requestHostTool','boundedText', body + '; return hostTool;')(
          catalog, new WeakMap(), (value)=>value, records, request, (value)=>String(value));
      const lost = await hostTool('s1', {{name:'easyicu_update_study_context',parameters:{{}}}}).execute('c1', {{}});
      assert.equal(lost.isError, true);
      assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
      await hostTool('s1', {{name:'easyicu_inspect_workflow',parameters:{{}}}}).execute('c2', {{}});
      assert.equal(state(memory.context([]).messages[0]).workflow.study_setup_receipt.revision, 3);
      assert.equal(state(memory.context([]).messages[0]).workflow.study_setup_receipt.configuration.data_source.source_id, 'source-FRESH');
    """)


@pytest.mark.parametrize("read_finishes_first", [False, True])
def test_real_host_tool_fences_concurrent_reads_and_keeps_rebind_pending(read_finishes_first):
    _check(f"""
      import {{readFileSync}} from 'node:fs';
      const main = readFileSync({json.dumps(str(MODULE.parent / 'main.mjs'))}, 'utf8');
      const body = main.slice(main.indexOf('function hostTool('), main.indexOf('function validateToolDefinitions('));
      const memory = new ResearchContextMemory();
      memory.beginTurn([receipt(2)]);
      const records = new Map([['s1', {{researchContext:memory, session:{{setActiveToolsByName:()=>{{}}}}}}]]);
      const catalog = new Map([
        ['easyicu_update_study_context', {{model_schema:null,host_mutating:true,execution_mode:'sequential'}}],
        ['easyicu_inspect_workflow', {{model_schema:null,host_mutating:false,execution_mode:'parallel'}}],
      ]);
      const readers = [];
      let finishUpdate;
      const request = (session, name) => new Promise((resolve) => {{
        if (name === 'easyicu_update_study_context') finishUpdate = resolve;
        else readers.push(resolve);
      }});
      const hostTool = new Function('TOOL_CATALOG_BY_NAME','CATALOG_PARAMETERS','defineTool',
        'sessions','requestHostTool','boundedText', body + '; return hostTool;')(
          catalog, new WeakMap(), (value)=>value, records, request, (value)=>String(value));
      const read = hostTool('s1', {{name:'easyicu_inspect_workflow',parameters:{{}}}});
      const write = hostTool('s1', {{name:'easyicu_update_study_context',parameters:{{}}}});
      const oldRead = read.execute('old', {{}});
      const mutation = write.execute('write', {{}});
      const overlappingRead = read.execute('overlap', {{}});
      if ({json.dumps(read_finishes_first)}) {{
        readers[0](receipt(2)); await oldRead;
        assert.equal(state(memory.context([]).messages[0]).status, 'unavailable');
      }}
      const update = receipt(3, 'source-NEW');
      Object.assign(update, {{code:'study_context_updated',owner:'easyicu.webserver.study_contexts'}});
      update.authority.study_revision = 2;
      Object.assign(update.details, {{rebind_required:true,host_rebind_after_turn:true,
        study:{{id:'study-1',revision:3}}}});
      finishUpdate(update); await mutation;
      assert.equal(state(memory.context([]).messages[0]).status, 'pending_rebind');
      if (!{json.dumps(read_finishes_first)}) {{ readers[0](receipt(2)); await oldRead; }}
      readers[1](receipt(2)); await overlappingRead;
      assert.equal(state(memory.context([]).messages[0]).status, 'pending_rebind');
      const postUpdateRead = read.execute('post-update', {{}});
      readers[2](receipt(2)); await postUpdateRead;
      assert.equal(state(memory.context([]).messages[0]).binding_state, 'rebind_required');
      assert.equal(state(memory.context([]).messages[0]).workflow.study_setup_receipt.revision, 3);
      memory.endTurn(); memory.beginTurn([receipt(3, 'source-NEW')]);
      assert.equal(state(memory.context([]).messages[0]).status, 'current');
    """)
