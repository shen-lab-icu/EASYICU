'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
const fs = require('node:fs');
const vm = require('node:vm');
const modulePath = path.resolve(__dirname, '../../src/easyicu/webserver/pi_copilot/node_app/src/conversation-model.mjs');
const owner = import(pathToFileURL(modulePath));
const runtime = { selected: { id: 'old', provider: 'fixture', baseUrl: 'https://synthetic.invalid/v1', api: 'openai-completions' }, config: {} };

test('API models retain the connection and only the chosen session changes', async () => {
  const { switchConversationModel, conversationModel } = await owner;
  const first = { externalId: 'first', session: { model: runtime.selected, async setModel(model) { this.model = model; } } };
  const other = { ...runtime.selected };
  await switchConversationModel(first, new Map(), runtime, 'new');
  assert.deepEqual(first.session.model, { ...runtime.selected, id: 'new', name: 'new' });
  assert.equal(other.id, 'old');
  assert.equal(runtime.selected.id, 'old');
  assert.equal(conversationModel(runtime).id, 'old');
  assert.equal(conversationModel(runtime, 'new').id, 'new'); // Restored session choice.
});

test('an active request or streaming session refuses before changing the SDK model', async () => {
  const { switchConversationModel } = await owner;
  for (const streaming of [false, true]) {
    const record = { externalId: 's', session: { isStreaming: streaming, setModel() { assert.fail('busy model change'); } } };
    await assert.rejects(switchConversationModel(record, new Map(streaming ? [] : [['s', 1]]), runtime, 'new'), { code: 'pi_session_busy' });
  }
});

test('account models use their own registry and unknown or invalid models fail closed', async () => {
  const { conversationModel } = await owner;
  const known = { id: 'account-next', provider: 'account' };
  const account = { ...runtime, config: { codexAuthFile: '/synthetic/unused', provider: 'account' },
    runtime: { getModel(provider, id) { assert.equal(provider, 'account'); return id === 'account-next' ? known : null; } } };
  assert.equal(conversationModel(account, 'account-next'), known);
  for (const model of ['', ' ', 'x'.repeat(257), {}]) assert.throws(() => conversationModel(runtime, model), { code: 'pi_session_model_unavailable' });
  for (const model of ['unknown', '', ' ', 'x'.repeat(257), {}]) {
    assert.throws(() => conversationModel(account, model), { code: 'pi_session_model_unavailable' });
  }
});

test('browser API scopes catalogue and selection and preserves typed refusals', async () => {
  const calls = [], window = {};
  const document = { readyState: 'loading', addEventListener() {}, querySelector() { return null; } };
  const fetch = async (url, options) => {
    calls.push([url, options]);
    return { ok: false, status: 409, json: async () => ({ detail: { error: 'pi_session_busy', reason: 'Wait.' } }) };
  };
  vm.runInNewContext(fs.readFileSync(path.resolve(__dirname, '../../src/easyicu/webserver/static/js/api.js'), 'utf8'), { window, document, fetch, URLSearchParams });
  await assert.rejects(window.EU_API.setPiCopilotConversationModel('session/a', { project_id: 'project-a', model: 'new' }), { code: 'pi_session_busy' });
  assert.equal(calls[0][0], '/api/copilot/pi/sessions/session%2Fa/model');
  assert.equal(calls[0][1].method, 'POST');
  assert.deepEqual(JSON.parse(calls[0][1].body), { project_id: 'project-a', model: 'new' });
  await assert.rejects(window.EU_API.loadPiCopilotConversationModels('session/a', 'project/b'), { code: 'pi_session_busy' });
  assert.equal(calls[1][0], '/api/copilot/pi/sessions/session%2Fa/models?project_id=project%2Fb');
});

test('the sidecar wires saved model restore and per-session switching', () => {
  const source = fs.readFileSync(path.join(path.dirname(modulePath), 'main.mjs'), 'utf8');
  assert.match(source, /conversationModel\(modelRuntime, params\.model\)/);
  assert.match(source, /case "session.set_model"/);
  assert.match(source, /switchConversationModel\(record, activeRequestBySession, await getModelRuntime\(\), params.model\)/);
});
