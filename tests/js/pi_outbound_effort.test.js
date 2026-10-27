// Exercise the real bridge and pinned SDK, with fetch replaced before import.
// No socket, provider credential, relay, or installed user runtime is used.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');
const { existsSync, mkdtempSync, readFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join, resolve } = require('node:path');

const root = resolve(__dirname, '../..');
const app = join(root, 'src/easyicu/webserver/pi_copilot/node_app');
const installed = existsSync(join(app, 'node_modules/@earendil-works/pi-coding-agent/dist/index.js'));

// Serialized into a Node preload so the SDK's actual HTTP request builder
// reaches this fake fetch. Capture the body before returning valid SSE.
async function fakeFetch(input, init = {}) {
  const url = String(input?.url || input);
  if (!['https://effort.invalid/v1/chat/completions', 'https://effort.invalid/v1/responses'].includes(url)) {
    throw new Error(`Unexpected endpoint: ${url}`);
  }
  const headers = new Headers(init.headers || input?.headers);
  if (headers.get('authorization') !== 'Bearer test-key') throw new Error('Expected dummy key');
  const body = JSON.parse(init.body || await input.text());
  appendFileSync(captureFile, JSON.stringify({ url, body }) + '\n');
  const text = 'Synthetic response.';
  const output = {
    id: 'msg_synthetic', type: 'message', role: 'assistant', status: 'completed',
    content: [{ type: 'output_text', text, annotations: [] }],
  };
  const response = {
    id: 'resp_synthetic', object: 'response', created_at: 1700000000,
    model: body.model, status: 'completed', output: [output],
    usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12,
      input_tokens_details: { cached_tokens: 0 }, output_tokens_details: { reasoning_tokens: 0 } },
  };
  const events = url.endsWith('/responses') ? [
    { type: 'response.created', response: { ...response, status: 'in_progress', output: [] } },
    { type: 'response.output_item.added', output_index: 0,
      item: { ...output, status: 'in_progress', content: [] } },
    { type: 'response.content_part.added', item_id: output.id, output_index: 0, content_index: 0,
      part: { type: 'output_text', text: '', annotations: [] } },
    { type: 'response.output_text.delta', item_id: output.id, output_index: 0, content_index: 0, delta: text },
    { type: 'response.output_text.done', item_id: output.id, output_index: 0, content_index: 0, text },
    { type: 'response.content_part.done', item_id: output.id, output_index: 0, content_index: 0,
      part: output.content[0] },
    { type: 'response.output_item.done', output_index: 0, item: output },
    { type: 'response.completed', response },
  ] : [
    { id: 'chatcmpl-synthetic', object: 'chat.completion.chunk', created: 1700000000,
      model: body.model, choices: [{ index: 0, delta: { role: 'assistant', content: text }, finish_reason: null }] },
    { id: 'chatcmpl-synthetic', object: 'chat.completion.chunk', created: 1700000000,
      model: body.model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 } },
  ];
  const sse = events.map((event, index) => (
    (event.type ? `event: ${event.type}\n` : '')
    + 'data: ' + JSON.stringify({ ...event, ...(event.type ? { sequence_number: index } : {}) }) + '\n\n'
  )).join('') + (url.endsWith('/responses') ? '' : 'data: [DONE]\n\n');
  return new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}

function bridge(api, directory) {
  const capture = join(directory, 'requests.jsonl');
  const preload = `import { appendFileSync } from 'node:fs';
    const captureFile = ${JSON.stringify(capture)};
    globalThis.fetch = ${fakeFetch.toString()};`;
  const child = spawn(process.execPath, [
    '--import', 'data:text/javascript;base64,' + Buffer.from(preload).toString('base64'),
    join(app, 'src/main.mjs'),
  ], {
    cwd: root,
    // Keep provider and account settings from the invoking shell out of the test.
    env: {
      PATH: process.env.PATH, TMPDIR: directory,
      EASYICU_PI_SESSION_DIR: join(directory, 'sessions'), EASYICU_PI_CWD: directory,
      EASYICU_PI_API_KEY: 'test-key', EASYICU_PI_PROVIDER: 'effort-test',
      EASYICU_PI_BASE_URL: 'https://effort.invalid/v1', EASYICU_PI_MODEL: 'synthetic-reasoner',
      EASYICU_PI_API: api,
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  const pending = new Map();
  let sequence = 0;
  let stderr = '';
  const lines = createInterface({ input: child.stdout });
  const rejectAll = error => {
    for (const request of pending.values()) request.reject(error);
    pending.clear();
  };
  child.stderr.on('data', data => { stderr += data; });
  child.on('error', rejectAll);
  const closed = new Promise(resolveClosed => child.once('close', (code, signal) => {
    rejectAll(new Error(`Bridge exited ${code || signal}: ${stderr}`));
    resolveClosed();
  }));
  lines.on('line', line => {
    const message = JSON.parse(line);
    if (message.kind === 'tool_request') {
      // Before each prompt the bridge preloads the host's workflow receipt
      // (main.mjs currentTurnOwnerContext); answer it like an idle project.
      // Any other host tool would mean the synthetic reply called one.
      if (message.params && message.params.name === 'easyicu_inspect_workflow') {
        child.stdin.write(JSON.stringify({
          protocol_version: 'easyicu.pi-copilot/1', kind: 'tool_response', request_id: message.request_id, ok: true,
          result: { status: 'ok', code: 'workflow_inspected', summary: 'Synthetic idle workflow.', owner: 'test', details: { workflow: {} }, authority: {} },
        }) + '\n');
        return;
      }
      rejectAll(new Error('The synthetic response must not call a host tool: ' + String(message.params && message.params.name)));
    }
    if (message.kind !== 'response') return;
    const request = pending.get(message.request_id);
    if (!request) return;
    pending.delete(message.request_id);
    if (message.ok) request.resolve(message.result);
    else request.reject(new Error(JSON.stringify(message.error)));
  });
  return {
    request(method, params) {
      const request_id = String(++sequence);
      return new Promise((resolveRequest, reject) => {
        const timer = setTimeout(() => {
          pending.delete(request_id);
          reject(new Error(`Bridge timed out on ${method}: ${stderr}`));
        }, 15000);
        pending.set(request_id, {
          resolve: result => { clearTimeout(timer); resolveRequest(result); },
          reject: error => { clearTimeout(timer); reject(error); },
        });
        child.stdin.write(JSON.stringify({
          protocol_version: 'easyicu.pi-copilot/1', kind: 'request', request_id, method, params,
        }) + '\n');
      });
    },
    bodies: () => readFileSync(capture, 'utf8').trim().split('\n').map(line => JSON.parse(line).body),
    async close() {
      lines.close();
      child.kill('SIGTERM');
      const timer = setTimeout(() => child.kill('SIGKILL'), 2000);
      try { await closed; } finally { clearTimeout(timer); }
    },
  };
}

for (const api of ['openai-completions', 'openai-responses']) {
  test(`${api} sends session effort on create, change, and off`, {
    skip: installed ? false : 'Pinned Pi dependencies are not installed in this checkout',
    timeout: 45000,
  }, async t => {
    const directory = mkdtempSync(join(tmpdir(), 'outbound-effort-'));
    const client = bridge(api, directory);
    t.after(async () => {
      await client.close();
      rmSync(directory, { recursive: true, force: true });
    });
    const assertEffort = (body, effort) => {
      if (effort === 'off' && api === 'openai-completions') {
        assert.equal(Object.hasOwn(body, 'reasoning_effort'), false);
      } else if (effort === 'off') {
        // The Responses API turns reasoning off explicitly (pi-ai's default
        // thinkingLevelMap.off), rather than leaving it to the service.
        assert.equal(body.reasoning?.effort, 'none');
      } else if (api === 'openai-completions') {
        assert.equal(body.reasoning_effort, effort);
      } else {
        assert.equal(body.reasoning?.effort, effort);
      }
    };
    let requests = 0;
    const prompt = async (session_id, effort) => {
      await client.request('session.prompt', { session_id, message: 'Reply with a synthetic greeting.' });
      const bodies = client.bodies();
      assert.equal(bodies.length, ++requests, 'exactly one outbound request per prompt');
      assert.equal(bodies.at(-1).model, 'synthetic-reasoner');
      assertEffort(bodies.at(-1), effort);
    };
    for (const effort of ['low', 'high', 'off']) {
      const session_id = `effort-${effort}`;
      const state = await client.request('session.create', { session_id, thinking_level: effort });
      assert.equal(state.thinking_level, effort);
      await prompt(session_id, effort);
    }
    for (const effort of ['high', 'off', 'low']) {
      const changed = await client.request('session.set_thinking_level', {
        session_id: 'effort-low', thinking_level: effort,
      });
      assert.equal(changed.thinking_level, effort);
      await prompt('effort-low', effort);
    }
  });
}
