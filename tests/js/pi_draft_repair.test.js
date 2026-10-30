'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const test = require('node:test');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const readline = require('node:readline');
const { spawn } = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const source = path.join(root, 'src/easyicu/webserver/pi_copilot/node_app/src');
const imports = Promise.all(['draft-repair.mjs', 'shell-budget.mjs'].map(file => import(pathToFileURL(path.join(source, file)).href)));
const request = '只更新讨论草案，不读取患者数据，不批准研究或运行分析';
const body = (overrides = {}) => ({ schema: 'easyicu.research-draft/1', title: '草案', question: '问题',
  population: '', exposure: '目录声明，待核验', outcome: '', candidate_data: '', approach: '',
  data_requirements: [], open_questions: [], changes: [], ...overrides });
const block = data => '```easyicu-research-draft\n' + JSON.stringify(data) + '\n```';
const answer = text => ({ role: 'assistant', stopReason: 'stop', content: [{ type: 'text', text }] });

async function fixture({ maxCalls = 4, response = answer(block(body())), onSend } = {}) {
  const [{ repairDraftOnce }, { ShellBudgetGuard }] = await imports;
  const guard = new ShellBudgetGuard({ tokenBudget: 100000, maxOutputTokens: 256,
    maxProviderCallsPerMessage: maxCalls, maxProviderCallsPerSession: 100, consumedTokens: () => 0 });
  guard.beginMessage(); guard.authorize({ systemPrompt: '', messages: [] });
  let tools = ['easyicu_list_source_concepts'];
  const calls = [];
  const record = { agentMode: 'research', abortRevision: 0, budgetGuard: guard,
    session: { messages: [answer(block(body({ exposure: '旧的未核验声明' }))),
      { role: 'user', content: [{ type: 'text', text: request }] }, answer('已查到字典信息。')],
      agent: { state: { systemPrompt: '' } },
      getActiveToolNames: () => [...tools],
      setActiveToolsByName: value => { tools = [...value]; },
      sendCustomMessage: async (message, options) => {
        calls.push({ message, options, tools: [...tools] });
        guard.authorize({ systemPrompt: '', messages: record.session.messages });
        if (onSend) await onSend(record);
        if (response instanceof Error) throw response;
        record.session.messages.push(response);
      } } };
  return { repairDraftOnce, record, guard, calls, activeTools: () => tools };
}

test('missing draft retries once with no tools and cumulative budget; shared parser accepts reply', async () => {
  const f = await fixture();
  assert.equal(await f.repairDraftOnce(f.record, request, 0), 'completed');
  assert.equal(f.calls.length, 1);
  assert.deepEqual(f.calls[0].tools, []);
  assert.equal(f.calls[0].message.display, false);
  assert.equal(f.calls[0].options.triggerTurn, true);
  assert.deepEqual(f.activeTools(), ['easyicu_list_source_concepts']);
  assert.equal(f.guard.messageProviderCalls, 2);
  assert.equal(f.guard.sessionProviderCalls, 2);
  assert.equal(f.guard.messageActive, true);
});

test('valid schema, non-revision, cancelled turn, errors and exhausted budget never trigger retry', async () => {
  for (const variant of ['valid', 'other', 'cancelled', 'error', 'exhausted', 'inactive', 'no_prior']) {
    const f = await fixture({ maxCalls: variant === 'exhausted' ? 1 : 4 });
    if (variant === 'valid') f.record.session.messages.push(answer(block(body())));
    if (variant === 'cancelled') f.record.abortRevision = 1;
    if (variant === 'error') f.record.session.messages.push({ ...answer('x'), stopReason: 'aborted' });
    if (variant === 'inactive') f.guard.endMessage();
    if (variant === 'no_prior') f.record.session.messages.shift();
    const result = await f.repairDraftOnce(f.record, variant === 'other' ? '解释字典' : request, 0);
    assert.equal(result, variant === 'exhausted' ? 'budget_unavailable' : 'not_applicable', variant);
    assert.equal(f.calls.length, 0, variant);
    assert.equal(f.guard.messageProviderCalls, 1, variant);
  }
});

test('malformed output receives only one attempt and cancellation or failure never replays tools', async () => {
  for (const variant of ['invalid', 'cancel', 'exception']) {
    const f = await fixture({ response: variant === 'exception' ? new Error('provider error') : answer(block(body({ schema: 'bad' }))),
      onSend: variant === 'cancel' ? record => { record.abortRevision += 1; } : null });
    const result = await f.repairDraftOnce(f.record, request, 0);
    assert.equal(result, variant === 'cancel' ? 'cancelled' : 'failed');
    assert.equal(f.calls.length, 1);
    assert.equal(await f.repairDraftOnce(f.record, request, 0), 'not_applicable');
    assert.equal(f.calls.length, 1);
    assert.deepEqual(f.activeTools(), ['easyicu_list_source_concepts']);
    assert.equal(f.guard.messageProviderCalls, 2);
  }
});

test('real sidecar repairs an omitted draft once with no active tools', { timeout: 60000 }, async () => {
  const app = path.join(root, 'src/easyicu/webserver/pi_copilot/node_app');
  if (!fs.existsSync(path.join(app, 'node_modules/@earendil-works/pi-ai/package.json'))) return;
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'easyicu-draft-repair-'));
  const requests = [];
  const server = http.createServer(async (req, res) => {
    let data = ''; for await (const part of req) data += part;
    const payload = JSON.parse(data); requests.push(payload);
    const text = requests.length === 1 ? block(body({ exposure: '旧的未核验声明' }))
      : requests.length === 3 ? '字典元数据已核对，但未生成草案。' : block(body({ exposure: '目录声明，待核验' }));
    const toolCall = requests.length === 2;
    const chunk = (delta, finish_reason) => JSON.stringify({ id: 'synthetic', object: 'chat.completion.chunk', created: 1,
      model: 'draft-test', choices: [{ index: 0, delta, finish_reason }] });
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const delta = toolCall ? { role: 'assistant', tool_calls: [{ index: 0, id: 'catalog-1', type: 'function',
      function: { name: 'easyicu_list_source_concepts', arguments: JSON.stringify({ source_id: 'synthetic-source', query: 'lactate', limit: 20 }) } }] }
      : { role: 'assistant', content: text };
    res.write('data: ' + chunk(delta, null) + '\n\n');
    res.write('data: ' + chunk({}, toolCall ? 'tool_calls' : 'stop') + '\n\n');
    res.end('data: [DONE]\n\n');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const env = { PATH: process.env.PATH, EASYICU_PI_API_KEY: 'synthetic',
    EASYICU_PI_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`, EASYICU_PI_MODEL: 'draft-test',
    EASYICU_PI_PROVIDER: 'local', EASYICU_PI_API: 'openai-completions', EASYICU_PI_SESSION_DIR: path.join(temp, 'sessions'),
    EASYICU_PI_CWD: temp, EASYICU_PI_MAX_TOKENS: '1024', EASYICU_PI_MAX_PROVIDER_CALLS_PER_MESSAGE: '4' };
  const child = spawn(process.execPath, [path.join(app, 'src/main.mjs')], { cwd: app, env, stdio: ['pipe', 'pipe', 'pipe'] });
  const pending = new Map(); let sequence = 0; let stderr = ''; const toolCalls = [];
  child.stderr.on('data', part => { stderr += part; });
  child.once('exit', code => { for (const item of pending.values()) item.reject(new Error(`sidecar exited ${code}: ${stderr}`)); });
  readline.createInterface({ input: child.stdout }).on('line', line => {
    const row = JSON.parse(line);
    if (row.kind === 'event') return;
    if (row.kind === 'tool_request') {
      toolCalls.push(row.params.name);
      child.stdin.write(JSON.stringify({ protocol_version: 'easyicu.pi-copilot/1', kind: 'tool_response', request_id: row.request_id,
        ok: true, result: { status: 'ok', code: 'workflow_inspected', summary: 'synthetic', details: { workflow: { missing_setup_fields: [] } } } }) + '\n');
    } else if (pending.has(row.request_id)) {
      const item = pending.get(row.request_id); pending.delete(row.request_id);
      row.ok ? item.resolve(row.result) : item.reject(new Error(row.error.message));
    }
  });
  function rpc(method, params) {
    const request_id = `request-${++sequence}`;
    return new Promise((resolve, reject) => { pending.set(request_id, { resolve, reject });
      child.stdin.write(JSON.stringify({ protocol_version: 'easyicu.pi-copilot/1', kind: 'request', request_id, method, params }) + '\n'); });
  }
  try {
    await rpc('session.create', { session_id: 'draft-test', agent_mode: 'research', language: 'zh' });
    await rpc('session.prompt', { session_id: 'draft-test', message: '请形成一份研究讨论稿。' });
    const result = await rpc('session.prompt', { session_id: 'draft-test', message: request });
    assert.equal(requests.length, 4);
    assert.equal((requests[3].tools || []).length, 0);
    assert.equal(result.shell_usage.message_provider_calls, 3);
    assert.equal(result.shell_usage.provider_calls, 4);
    assert.equal(result.transcript.filter(item => item.role === 'user').length, 2);
    assert.ok(result.transcript.some(item => JSON.stringify(item.content).includes('目录声明，待核验')));
    assert.equal(toolCalls.filter(name => name === 'easyicu_list_source_concepts').length, 1);
    assert.ok(toolCalls.every(name => ['easyicu_inspect_workflow', 'easyicu_list_source_concepts'].includes(name)));
  } finally {
    if (child.exitCode === null) { const exited = new Promise(resolve => child.once('exit', resolve)); child.stdin.end(); await exited; }
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
