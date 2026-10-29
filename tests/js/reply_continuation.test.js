'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { createHash } = require('node:crypto');
const { spawn } = require('node:child_process');
const readline = require('node:readline');
const vm = require('node:vm');
const { pathToFileURL } = require('node:url');
const { test } = require('node:test');

const root = path.resolve(__dirname, '../..');
const app = path.join(root, 'src/easyicu/webserver/pi_copilot/node_app');
const js = path.join(root, 'src/easyicu/webserver/static/js');
const imports = Promise.all([
  import(pathToFileURL(path.join(app, 'src/reply-continuation.mjs'))),
  import(pathToFileURL(path.join(app, 'src/shell-budget.mjs'))),
]);
const model = { api: 'openai-completions', provider: 'local', id: 'local-recovery' };
const assistant = (content, stopReason = 'toolUse') => ({
  role: 'assistant', content, stopReason, ...model, model: model.id,
  ...(stopReason === 'error' ? { errorMessage: 'HTTP 503 Service Unavailable' } : {}),
});
const entry = (id, message) => ({ type: 'message', id, message });
const receipt = calls => ({ type: 'custom', customType: 'easyicu.shell-budget/1', data: { provider_calls: calls } });

async function fixture({ maxCalls = 8, previousCalls = 0 } = {}) {
  const [, { ShellBudgetGuard }] = await imports;
  const entries = previousCalls ? [entry('old-user', { role: 'user', content: 'Earlier turn' }), receipt(previousCalls)] : [];
  entries.push(entry('user', { role: 'user', content: 'Save and check brief.html' }), receipt(previousCalls + 1),
    entry('tools', assistant([{ type: 'toolCall', id: 'write', name: 'easyicu_write_project_file' },
      { type: 'toolCall', id: 'check', name: 'easyicu_check_project_file' }])),
    entry('written', { role: 'toolResult', toolCallId: 'write', toolName: 'easyicu_write_project_file',
      details: { status: 'ok', code: 'pi_workspace_file_written' }, isError: false }),
    entry('checked', { role: 'toolResult', toolCallId: 'check', toolName: 'easyicu_check_project_file',
      details: { status: 'ok', code: 'pi_workspace_file_checked' }, isError: false }),
    receipt(previousCalls + 5), entry('failed', assistant([], 'error')));
  const record = {
    agentMode: 'workspace', language: 'zh',
    session: { model, messages: entries.filter(item => item.type === 'message').map(item => item.message),
      isStreaming: false, pendingMessageCount: 0 },
    sessionManager: { getEntries: () => entries, getBranch: () => entries },
    budgetGuard: new ShellBudgetGuard({ tokenBudget: 100000, maxOutputTokens: 256,
      maxProviderCallsPerMessage: maxCalls, maxProviderCallsPerSession: 100,
      consumedTokens: () => 0, initialProviderCalls: previousCalls + 5 }),
  };
  return { record, entries };
}

test('continuation restores the original message budget and preserves the failed branch', async () => {
  const [{ replyContinuationState, prepareReplyContinuation }] = await imports;
  const { record, entries } = await fixture({ previousCalls: 7 });
  const before = JSON.stringify(entries);
  assert.equal(replyContinuationState(record).available, true);
  assert.equal(record.budgetGuard.state().provider_calls, 12, 'eligibility consumes no budget');
  const continuation = prepareReplyContinuation(record, 'failed', { status: 'failed', summary: 'CSV runtime check failed' });
  assert.equal(record.budgetGuard.state().message_provider_calls, 5);
  assert.equal(record.budgetGuard.state().provider_calls, 12);
  assert.equal(continuation.message.display, false);
  assert.equal(continuation.message.details.tools_disabled, true);
  assert.match(continuation.message.content, /"status":"failed"/);
  assert.match(continuation.message.content, /not runtime acceptance/);
  assert.equal(JSON.stringify(entries), before, 'planning a continuation never rewinds or edits history');
  record.budgetGuard.authorize({ messages: [] });
  assert.equal(record.budgetGuard.state().message_provider_calls, 6);
  assert.equal(record.budgetGuard.state().provider_calls, 13);
});

test('unknown outcomes, live tools, source errors, authority changes and stale targets fail closed', async () => {
  const [{ replyContinuationState, prepareReplyContinuation }] = await imports;
  for (const mutate of [
    ({ entries }) => entries.splice(entries.findIndex(item => item.id === 'checked'), 1),
    ({ entries }) => { entries.find(item => item.id === 'written').message.details.status = 'unknown'; },
    ({ entries }) => { entries.find(item => item.id === 'written').message.isError = true; },
    ({ entries }) => { entries.find(item => item.id === 'tools').message.content[0].name = 'easyicu_run'; },
    ({ record }) => { record.session.pendingMessageCount = 1; },
    ({ record }) => { record.session.model = { ...model, id: 'different-model' }; },
    ({ record }) => { record.agentMode = 'research'; },
    ({ entries }) => { entries.find(item => item.id === 'failed').message.errorMessage = 'pi_host_tool_outcome_unknown'; },
    ({ entries }) => { entries.find(item => item.id === 'failed').message.stopReason = 'aborted'; },
  ]) {
    const f = await fixture(); mutate(f);
    assert.equal(replyContinuationState(f.record).available, false);
  }
  const f = await fixture();
  assert.throws(() => prepareReplyContinuation(f.record, 'older-failed'), /no longer current/);
});

test('a completed runtime-QA read after a saved artifact permits explanation recovery without promoting acceptance', async () => {
  const [{ replyContinuationState, prepareReplyContinuation }] = await imports;
  const { record, entries } = await fixture();
  const index = entries.findIndex(item => item.id === 'failed');
  entries.splice(index, 0,
    entry('qa-call', assistant([{ type: 'toolCall', id: 'runtime-qa', name: 'easyicu_get_project_runtime_qa' }])),
    entry('qa-read', { role: 'toolResult', toolCallId: 'runtime-qa', toolName: 'easyicu_get_project_runtime_qa',
      details: { status: 'ok', code: 'pi_workspace_runtime_qa_read', details: { runtime_qa: { status: 'failed' } } }, isError: false }));
  assert.equal(replyContinuationState(record).available, true);
  const recovery = prepareReplyContinuation(record, 'failed', { status: 'failed', summary: 'Current exact-byte QA failed' });
  assert.match(recovery.message.content, /"status":"failed"/);
  assert.match(recovery.message.content, /must not be called accepted/);
});

test('original message call/token/cost exhaustion and unprovable legacy budgets close recovery', async () => {
  const [{ replyContinuationState }, { ShellBudgetGuard }] = await imports;
  const f = await fixture({ maxCalls: 5 });
  assert.equal(replyContinuationState(f.record).reason_code, 'pi_shell_message_provider_call_budget_exhausted');
  const legacy = await fixture();
  legacy.entries.unshift(entry('unsaved-old-turn', { role: 'user', content: 'Old legacy turn' }));
  assert.equal(replyContinuationState(legacy.record).reason_code, 'pi_reply_continuation_budget_unknown');
  const t = await fixture();
  t.record.budgetGuard.tokenBudget = 100;
  assert.equal(replyContinuationState(t.record).reason_code, 'pi_shell_token_budget_exhausted');
  const pricing = { inputPriceUsdPerMillionTokens: 1, outputPriceUsdPerMillionTokens: 1,
    maxCostUsdPerMessage: 0.01, maxCostUsdPerSession: 1 };
  const guard = new ShellBudgetGuard({ tokenBudget: 100000, maxOutputTokens: 256,
    maxProviderCallsPerMessage: 8, maxProviderCallsPerSession: 100, consumedTokens: () => 0, pricing });
  guard.beginMessage(); guard.authorize({ messages: [] }); guard.authorize({ messages: [] }); guard.endMessage();
  const cost = guard.receipt().reserved_cost_micro_usd;
  assert.throws(() => guard.resumeMessage({ messageProviderCalls: 2, messageReservedCostMicroUsd: cost,
    sessionProviderCalls: 2, sessionReservedCostMicroUsd: cost }), error => error.code === 'pi_shell_message_cost_budget_exhausted');
  assert.equal(guard.state().provider_calls, 2);
});

test('artifact validation coverage is incomplete when identities are missing or more than eight artifacts were saved', async () => {
  const [{ replyContinuationState }] = await imports;
  const missing = await fixture();
  assert.equal(replyContinuationState(missing.record).resources_complete, false);
  const f = await fixture();
  const start = f.entries.findIndex(item => item.id === 'tools');
  const end = f.entries.findIndex(item => item.id === 'failed');
  const rows = [];
  for (let index = 0; index < 9; index++) {
    const file = `artifact-${index}.html`;
    rows.push(entry(`tool-${index}`, assistant([{ type: 'toolCall', id: `write-${index}`, name: 'easyicu_write_project_file', arguments: { file } }])),
      entry(`result-${index}`, { role: 'toolResult', toolCallId: `write-${index}`, toolName: 'easyicu_write_project_file',
        details: { status: 'ok', code: 'pi_workspace_file_written', details: { file, sha256: 'a'.repeat(64) } }, isError: false }));
  }
  rows.push(receipt(5));
  f.entries.splice(start, end - start, ...rows);
  const state = replyContinuationState(f.record);
  assert.equal(state.available, true, 'explanation remains possible even though validation coverage is incomplete');
  assert.equal(state.completed_resource_total, 9);
  assert.equal(state.completed_resources.length, 8);
  assert.equal(state.resources_complete, false);
});

test('message action substitutes an explicit explanation continuation for unsafe regeneration', () => {
  let owner; const calls = [];
  const context = vm.createContext({ window: {
    EU_HTML: { esc: String }, EasyICU: { guidedPi: { declare: (_name, value) => { owner = value; } } },
  } });
  vm.runInContext(fs.readFileSync(path.join(js, 'screens-guided-pi-message-actions.js'), 'utf8'), context);
  const rows = [{ id: 'u', role: 'user', entryId: 'u', text: 'Save brief' },
    { id: 'a', role: 'assistant', text: 'Explanation interrupted', errorCode: 'pi_model_provider_unavailable' }];
  const actions = owner.create({ tr: (_en, zh) => zh, iconHtml: () => '', rows: () => rows,
    regenerate: () => calls.push('regenerate'), continueReply: (...args) => calls.push(args) });
  const options = { canRetry: true, retryUserEntryId: 'u', hasCompletedAction: true,
    canContinueReply: true, sourceErrorEntryId: 'failed' };
  const html = actions.render(rows[1], options).actionsHtml;
  assert.match(html, /继续完成说明/);
  assert.doesNotMatch(html, /data-gpi-message-retry/);
  function click(selector) {
    const button = { closest: () => ({ dataset: { gpiMessageId: 'a' } }) };
    actions.handleClick({ target: { closest: query => query === selector ? button : null } });
  }
  click('[data-gpi-message-retry]'); assert.equal(calls.length, 0, 'stale retry DOM cannot replay a completed operation');
  click('[data-gpi-message-continue-reply]'); assert.deepEqual(calls, [['failed', 'a']]);
  assert.doesNotMatch(actions.render(rows[1], { ...options, canContinueReply: false }).actionsHtml, /data-gpi-message-retry/);
  assert.match(actions.render({ ...rows[1], errorCode: '' }, options).actionsHtml, /data-gpi-message-retry/);
  const pending = actions.render(rows[1], { ...options, canContinueReply: false, continuationPending: true,
    continuationReasonCode: 'pi_reply_continuation_available' }).actionsHtml;
  assert.match(pending, /正在继续完成说明/);
  assert.match(pending, /disabled aria-busy="true"/);
  assert.doesNotMatch(pending, /gpi-message-continuation-note/);
  assert.doesNotMatch(actions.render({ ...rows[1], errorCode: '', replyContinuation: true }, options).actionsHtml, /data-gpi-message-retry/);
  assert.doesNotMatch(actions.render(rows[1], { ...options, hasCompletedAction: false, hasSideEffectAttempt: true,
    canContinueReply: false }).actionsHtml, /data-gpi-message-retry/);
});

function hash(file) { return createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }

test('persisted recovery replay marks only its assistant explanation after a reload', () => {
  let transcript;
  const context = vm.createContext({ window: { EasyICU: { guidedPi: { declare: (_name, value) => { transcript = value; } } } } });
  vm.runInContext(fs.readFileSync(path.join(js, 'screens-guided-pi-transcript.js'), 'utf8'), context);
  const rows = [{ role: 'assistant', text: 'Saved; runtime unverified.', timelineAt: 10500 },
    { role: 'assistant', text: 'Saved; runtime unverified.', timelineAt: 25000 },
    { role: 'assistant', text: 'Original provider failure.', timelineAt: 9500 }];
  transcript.markReplyContinuations(rows, [{ kind: 'reply_continuation', assistant_reply: 'Saved; runtime unverified.',
    started_at: 10000, ended_at: 11000 }], Number);
  assert.equal(rows[0].replyContinuation, true);
  assert.equal(rows[1].replyContinuation, undefined);
  assert.equal(rows[2].replyContinuation, undefined);
});

test('real sidecar: write/check once, fail Provider, reload, continue explanation without tool replay', {
  skip: !fs.existsSync(path.join(app, 'node_modules/@earendil-works/pi-ai/package.json')),
  timeout: 60000,
}, async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'easyicu-reply-continuation-'));
  const artifact = path.join(temp, 'brief.html');
  const events = []; const requests = []; let recovered = false; let writes = 0; let checks = 0;
  const server = http.createServer(async (req, res) => {
    let text = ''; for await (const part of req) text += part;
    const body = JSON.parse(text); requests.push(body);
    if (requests.length > 1 && !recovered) { res.writeHead(503); res.end('Synthetic upstream unavailable'); return; }
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const delta = requests.length === 1 ? { role: 'assistant', tool_calls: [
      { index: 0, id: 'write-1', type: 'function', function: { name: 'easyicu_write_project_file', arguments: JSON.stringify({ file: 'brief.html', content: '<!doctype html><title>Saved once</title><p>synthetic</p>' }) } },
      { index: 1, id: 'check-1', type: 'function', function: { name: 'easyicu_check_project_file', arguments: JSON.stringify({ file: 'brief.html' }) } },
    ] } : { role: 'assistant', content: '文件已经保存；静态检查通过。当前运行时验收失败，尚未完成，不能称为已验收。' };
    const chunk = (delta, finish_reason) => JSON.stringify({ id: 'synthetic', object: 'chat.completion.chunk', created: 1,
      model: 'local-recovery', choices: [{ index: 0, delta, finish_reason }] });
    res.write('data: ' + chunk(delta, null) + '\n\n');
    res.write('data: ' + chunk({}, requests.length === 1 ? 'tool_calls' : 'stop') + '\n\n');
    res.end('data: [DONE]\n\n');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const env = { PATH: process.env.PATH, EASYICU_PI_API_KEY: 'local-synthetic-only',
    EASYICU_PI_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`,
    EASYICU_PI_MODEL: 'local-recovery', EASYICU_PI_PROVIDER: 'local', EASYICU_PI_API: 'openai-completions',
    EASYICU_PI_SESSION_DIR: path.join(temp, 'sessions'), EASYICU_PI_CWD: temp,
    EASYICU_PI_MAX_TOKENS: '256', EASYICU_PI_MAX_PROVIDER_CALLS_PER_MESSAGE: '6' };
  let child; let sequence = 0; let pending;
  function start() {
    child = spawn(process.execPath, [path.join(app, 'src/main.mjs')], { cwd: app, env, stdio: ['pipe', 'pipe', 'pipe'] });
    pending = new Map(); let stderr = '';
    child.stderr.on('data', part => { stderr += part; });
    child.once('exit', code => { for (const task of pending.values()) task.reject(new Error(`sidecar exit ${code}: ${stderr}`)); });
    readline.createInterface({ input: child.stdout }).on('line', line => {
      const row = JSON.parse(line);
      if (row.kind === 'event') events.push(row);
      else if (row.kind === 'tool_request') {
        const { name, arguments: args } = row.params;
        let result = { status: 'ok', code: 'workflow_inspected', summary: 'Synthetic workflow', details: { workflow: { missing_setup_fields: [] } } };
        if (name === 'easyicu_write_project_file') {
          writes++; fs.writeFileSync(artifact, args.content, { flag: 'wx' });
          result = { status: 'ok', code: 'pi_workspace_file_written', summary: 'Saved brief.html', details: { file: args.file, sha256: hash(artifact) } };
        } else if (name === 'easyicu_check_project_file') {
          checks++; assert.match(fs.readFileSync(artifact, 'utf8'), /<!doctype html>/);
          result = { status: 'ok', code: 'pi_workspace_file_checked', summary: 'Static structure checked', details: { file: args.file, sha256: hash(artifact) } };
        } else assert.equal(name, 'easyicu_inspect_workflow', 'continuation must not ask for any tool');
        child.stdin.write(JSON.stringify({ protocol_version: 'easyicu.pi-copilot/1', kind: 'tool_response', request_id: row.request_id, ok: true, result }) + '\n');
      } else if (pending.has(row.request_id)) {
        const task = pending.get(row.request_id); pending.delete(row.request_id);
        row.ok ? task.resolve(row.result) : task.reject(Object.assign(new Error(row.error.message), row.error));
      }
    });
  }
  function rpc(method, params) {
    const request_id = `test-${++sequence}`;
    return new Promise((resolve, reject) => {
      pending.set(request_id, { resolve, reject });
      child.stdin.write(JSON.stringify({ protocol_version: 'easyicu.pi-copilot/1', kind: 'request', request_id, method, params }) + '\n');
    });
  }
  async function stop() {
    if (!child || child.exitCode !== null) return;
    const exited = new Promise(resolve => child.once('exit', resolve)); child.stdin.end(); await exited;
  }
  try {
    start();
    await rpc('session.create', { session_id: 'recovery-test', agent_mode: 'workspace', language: 'zh' });
    const failed = await rpc('session.prompt', { session_id: 'recovery-test', message: '保存并检查 brief.html，然后说明结果。' });
    assert.equal(writes, 1); assert.equal(checks, 1); assert.equal(requests.length, 5, 'one tool decision plus four guarded failed final explanations');
    assert.equal(failed.reply_continuation.available, true, JSON.stringify(failed.reply_continuation));
    assert.equal(failed.shell_usage.message_provider_calls, 5);
    const savedHash = hash(artifact); const originalErrorId = failed.reply_continuation.source_error_entry_id;
    assert.ok(failed.transcript.some(row => row.error_code === 'pi_model_provider_unavailable'));
    await stop(); start();
    const restored = await rpc('session.create', { session_id: 'recovery-test', session_file: failed.session_file, agent_mode: 'workspace', language: 'zh' });
    assert.equal(restored.reply_continuation.available, true, 'recovery remains available after sidecar reload');
    recovered = true;
    const continued = await rpc('session.continue_reply', { session_id: 'recovery-test', source_error_entry_id: originalErrorId,
      validation_context: { status: 'failed', summary: 'Synthetic current runtime acceptance failed' } });
    assert.equal(writes, 1); assert.equal(checks, 1); assert.equal(hash(artifact), savedHash);
    assert.equal(requests.length, 6); assert.equal((requests.at(-1).tools || []).length, 0, 'continuation request has zero tool capability');
    assert.match(JSON.stringify(requests.at(-1).messages), /"status\\":\\"failed/);
    assert.equal(continued.shell_usage.message_provider_calls, 6, 'original message budget accumulates rather than resetting');
    assert.equal(continued.shell_usage.provider_calls, 6);
    assert.equal(continued.reply_continuation.available, false, 'successful explanation cannot be continued again');
    assert.ok(continued.transcript.some(row => row.error_code === 'pi_model_provider_unavailable' && row.entry_id === originalErrorId), 'the original failure is retained');
    assert.ok(continued.transcript.some(row => JSON.stringify(row.content).includes('当前运行时验收失败')));
    assert.ok(events.some(row => row.event.type === 'message_end' && row.event.error_code === 'pi_model_provider_unavailable'));
    assert.ok(events.some(row => row.event.type === 'text_delta' && row.event.delta.includes('当前运行时验收失败')));
    const jsonl = fs.readFileSync(failed.session_file, 'utf8').trim().split('\n').map(JSON.parse);
    assert.ok(jsonl.some(row => row.customType === 'easyicu.reply-continuation/1' && row.data.state === 'completed'));
    await assert.rejects(rpc('session.continue_reply', { session_id: 'recovery-test', source_error_entry_id: originalErrorId }));
    assert.equal(requests.length, 6);
    const outDir = process.env.EASYICU_REPLY_RECOVERY_RECEIPT_DIR;
    if (outDir) {
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, 'deterministic-sidecar-recovery.json'), JSON.stringify({
        schema_version: 'easyicu.reply-continuation-test/1', provider: 'loopback deterministic HTTP',
        original_failure_preserved: true, sidecar_reloaded_before_continuation: true,
        write_count: writes, check_count: checks, provider_calls: requests.length,
        original_message_provider_calls: continued.shell_usage.message_provider_calls,
        artifact_sha256_before: savedHash, artifact_sha256_after: hash(artifact),
        continuation_tools: (requests.at(-1).tools || []).length,
        failed_current_qa_retained: true, actual_home_mutated: false,
      }, null, 2) + '\n');
    }
  } finally { await stop(); await new Promise(resolve => server.close(resolve)); fs.rmSync(temp, { recursive: true, force: true }); }
});
