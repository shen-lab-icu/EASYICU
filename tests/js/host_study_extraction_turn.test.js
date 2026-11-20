'use strict';
// B19 routing: the host already started this study's extraction for the
// researcher's own sent turn. The sidecar reports the host's receipt with no
// tools: it does not preload owner context (the submission changed authority,
// so any further host tool would be refused) and offers the model no tool.
const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const readline = require('node:readline');
const { spawn } = require('node:child_process');

const root = path.resolve(__dirname, '../..');
const app = path.join(root, 'src/easyicu/webserver/pi_copilot/node_app');
const intent = 'report_host_study_extraction';
const receipt = {
  status: 'ok', code: 'easyicu_extraction_submitted',
  summary: 'Submitted EasyICU feature extraction job extract-1 from the bound typed study setup.',
  owner: 'easyicu.webserver.routes.jobs', details: { job_id: 'extract-1' }, authority: {},
};
const sent = '我确认：按本研究现有的人群条件进行数据提取，生成本研究的数据包。';

test('a host extraction turn reports the host receipt with no tools', { timeout: 60000 }, async () => {
  if (!fs.existsSync(path.join(app, 'node_modules/@earendil-works/pi-ai/package.json'))) return;
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'easyicu-host-extraction-'));
  const requests = [];
  const server = http.createServer(async (req, res) => {
    let data = ''; for await (const part of req) data += part;
    requests.push(JSON.parse(data));
    const chunk = (delta, finish_reason) => JSON.stringify({ id: 'synthetic', object: 'chat.completion.chunk', created: 1,
      model: 'host-extraction-test', choices: [{ index: 0, delta, finish_reason }] });
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write('data: ' + chunk({ role: 'assistant', content: '本研究的数据提取已开始。' }, null) + '\n\n');
    res.write('data: ' + chunk({}, 'stop') + '\n\n');
    res.end('data: [DONE]\n\n');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const env = { PATH: process.env.PATH, TMPDIR: temp, EASYICU_PI_API_KEY: 'synthetic',
    EASYICU_PI_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`, EASYICU_PI_MODEL: 'host-extraction-test',
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
      row.ok ? item.resolve(row.result) : item.reject(Object.assign(new Error(row.error.message), { code: row.error.code }));
    }
  });
  function rpc(method, params) {
    const request_id = `request-${++sequence}`;
    return new Promise((resolve, reject) => { pending.set(request_id, { resolve, reject });
      child.stdin.write(JSON.stringify({ protocol_version: 'easyicu.pi-copilot/1', kind: 'request', request_id, method, params }) + '\n'); });
  }
  try {
    await rpc('session.create', { session_id: 'host-extraction', agent_mode: 'research', language: 'zh' });
    // A receipt belongs to this intent, and this intent needs its receipt.
    await assert.rejects(rpc('session.prompt', { session_id: 'host-extraction', message: sent, host_receipt: receipt }),
      { code: 'pi_prompt_invalid' });
    await assert.rejects(rpc('session.prompt', { session_id: 'host-extraction', message: sent, intent }),
      { code: 'pi_prompt_invalid' });
    await assert.rejects(rpc('session.prompt', { session_id: 'host-extraction', message: sent, intent, host_receipt: [receipt] }),
      { code: 'pi_prompt_invalid' });
    assert.equal(requests.length, 0);

    const result = await rpc('session.prompt', { session_id: 'host-extraction', message: sent, intent,
      message_origin: 'typed', host_receipt: receipt });
    assert.deepEqual(toolCalls, []);
    assert.equal(requests.length, 1);
    assert.equal((requests[0].tools || []).length, 0);
    const prompt = JSON.stringify(requests[0].messages);
    assert.ok(prompt.includes('easyicu_extraction_submitted'));
    assert.ok(prompt.includes('[EASYICU_HOST_TRANSITION_V1]'));
    assert.ok(prompt.includes('the host already acted on it before this reply'));
    const users = result.transcript.filter(item => item.role === 'user');
    assert.equal(users.length, 1);
    assert.ok(JSON.stringify(users[0].content).includes(sent));
    assert.ok(!JSON.stringify(users[0].content).includes('EASYICU_HOST_TRANSITION'));
  } finally {
    if (child.exitCode === null) { const exited = new Promise(resolve => child.once('exit', resolve)); child.stdin.end(); await exited; }
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
