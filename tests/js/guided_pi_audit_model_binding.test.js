/* Synthetic, offline contracts for the saved conversation's model menu. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const files = process.argv.slice(2);
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const owners = files.length ? files : ['screens-guided-pi-model-menu.js',
  'screens-guided-pi-header.js', 'screens-guided-pi-session-view.js'].map(name => path.join(root, name));
const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const copy = value => JSON.parse(JSON.stringify(value));
const defer = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const tick = () => new Promise(resolve => setImmediate(resolve));
function session(provider = 'openai', id = 'conversation-a', model = 'model-a') {
  return { session_id: id, project_id: 'project-a', model: { id: model }, model_connection: {
    schema_version: 'easyicu.research-provider-binding/1', provider, model,
    credential_source: provider === 'codex' ? 'codex_user_auth' : 'pi_verified',
    authentication_mode: provider === 'codex' ? 'chatgpt_account' : 'api_key',
  } };
}
function harness(language = 'zh', initial = session()) {
  const declared = {}, listeners = [], calls = [];
  const state = { session: initial, messages: [], workflowReceipts: [], draft: '', sessions: [initial] };
  let project = initial.project_id, hosted = false, stale = false;
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const document = {
    addEventListener: (type, handler) => listeners.push({ type, handler }),
    createElement: () => ({ className: '', innerHTML: '', remove() {} }),
    querySelector: () => null,
  };
  const api = {
    loadPiCopilotSession: async (id, projectId) => { calls.push(['session', id, projectId]); return { session: copy(state.session) }; },
    loadPiCopilotConversationModels: async (id, projectId) => {
      calls.push(['catalogue', id, projectId]);
      return { models: state.session.model_connection.provider === 'codex' ? ['model-a', 'account-next'] : ['model-a', 'model-b'] };
    },
    setPiCopilotConversationModel: async (id, body) => {
      calls.push(['switch', id, body]);
      return { session: { ...copy(state.session), model: { id: body.model },
        model_connection: { ...copy(state.session.model_connection), model: body.model } } };
    },
  };
  const modules = { declare: (name, owner) => { declared[name] = owner; }, optional: name => declared[name] || null };
  const window = { EU_LANG: language, EU_HTML: { esc }, EU_API: api, EasyICU: { guidedPi: modules } };
  for (const file of owners) vm.runInNewContext(fs.readFileSync(file, 'utf8'), { window, document }, { filename: file });
  declared.hosted = { active: () => hosted };
  const view = declared.sessionView.create({
    state, modules, projectId: () => project, sessionIsStale: () => stale, agentMode: () => 'research',
    header: declared.header, tr, esc, iconHtml: () => '', projectTitle: () => 'Study', navigationSessionTitle: () => 'Conversation',
    starters: { render: options => options.composer },
    transcript: { latestTurnCompletedIdeaExploration: () => false }, runFiles: { timeline: rows => rows, notice: () => '' },
    activity: { renderTimeline: () => '' }, runOutcome: { render: () => '', collection: () => [] },
    studyWorkspace: { renderReference: () => '', renderSkillReference: () => '', renderMaterials: () => '', renderSkillPicker: () => '', renderAccessMode: () => '' },
    aside: { layoutOptions: () => ({}) }, resourceOwner: {},
  });
  function control() {
    const html = view.sessionPanel();
    const key = (html.match(/data-gpi-model-key="([^"]*)"/) || [])[1] || '';
    const span = { textContent: [state.session.model_connection?.provider, state.session.model_connection?.model].filter(Boolean).join(' · '), title: '' };
    const rows = [];
    const dataset = { gpiModelKey: key.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&amp;/g, '&') };
    const popover = {
      dataset,
      querySelectorAll: () => { rows.length = 0; return []; }, querySelector: () => null,
      appendChild: node => rows.push(node), insertBefore: node => rows.push(node),
    };
    const node = {
      dataset,
      open: true, isConnected: true, span, rows,
      querySelector: selector => selector === '.gpi-model-binding > span' ? span : popover,
      closest: () => node, html: () => rows.map(row => row.innerHTML).join(''),
    };
    return node;
  }
  function clickModel(node, model = 'model-b') {
    const choice = { dataset: { gpiPickModel: model }, closest: selector => selector === '[data-gpi-pick-model]' ? choice : node };
    for (const listener of listeners.filter(row => row.type === 'click')) listener.handler({ target: choice, preventDefault() {} });
  }
  return { state, api, calls, control, clickModel, view, menu: declared.modelMenu, header: declared.header, tr,
    project: value => { project = value; }, hosted: value => { hosted = value; }, stale: value => { stale = value; } };
}

test('API and account selections update only the acknowledged conversation and header', async () => {
  for (const provider of ['openai', 'codex']) {
    const h = harness('zh', session(provider)); const node = h.control();
    await h.menu.load(node);
    const model = provider === 'codex' ? 'account-next' : 'model-b';
    assert.deepEqual(h.calls[1], ['catalogue', 'conversation-a', 'project-a']);
    assert.match(node.html(), /data-gpi-pick-model/);
    h.clickModel(node, model); await tick();
    assert.equal(h.calls.filter(row => row[0] === 'switch').length, 1);
    assert.deepEqual(copy(h.calls[2]), ['switch', 'conversation-a', { project_id: 'project-a', model }]);
    assert.equal(h.state.session.model_connection.model, model);
    assert.equal(h.state.sessions[0].model_connection.model, model);
    assert.equal(node.span.textContent, provider + ' · ' + model);
    assert.equal(node.open, false);
  }
});

test('legacy, incomplete and incoherent bindings never guess from the printed label', async () => {
  for (const change of [null, {}, { provider: 'codex', model: 'model-a' },
    { ...session().model_connection, authentication_mode: 'chatgpt_account' }]) {
    const h = harness(); h.state.session.model_connection = change;
    const node = h.control(); node.span.textContent = 'openai · model-a';
    await h.menu.load(node);
    assert.equal(h.calls.length, 0);
    assert.doesNotMatch(node.html(), /data-gpi-pick-model/);
    assert.ok(node.html());
  }
});

test('missing API owners and failed readback leave the saved model intact', async () => {
  for (const missing of ['loadPiCopilotSession', 'loadPiCopilotConversationModels']) {
    const h = harness(); delete h.api[missing]; const node = h.control();
    await h.menu.load(node);
    assert.match(node.html(), /不可用|无法/);
    assert.doesNotMatch(node.html(), /data-gpi-pick-model/);
    assert.equal(node.span.textContent, 'openai · model-a');
  }
  const h = harness(); h.api.loadPiCopilotSession = async () => { throw Object.assign(new Error('Forbidden'), { status: 403 }); };
  const node = h.control(); await h.menu.load(node);
  assert.match(node.html(), /HTTP 403/);
  assert.equal(h.calls.length, 0);
});

test('session, project, provider and model changes each invalidate the catalogue', async () => {
  for (const change of ['session', 'project', 'provider', 'model']) {
    const h = harness(); await h.menu.load(h.control());
    if (change === 'session') h.state.session = session('openai', 'conversation-b');
    if (change === 'project') { h.state.session.project_id = 'project-b'; h.project('project-b'); }
    if (change === 'provider') h.state.session = session('codex');
    if (change === 'model') h.state.session = session('openai', 'conversation-a', 'model-z');
    const node = h.control(); await h.menu.load(node);
    assert.equal(h.calls.filter(row => row[0] === 'catalogue').length, 2, change);
    if (change === 'provider') { assert.match(node.html(), /account-next/); assert.doesNotMatch(node.html(), /model-b/); }
  }
});

test('reopening and a reload read the acknowledged binding and current catalogue again', async () => {
  const h = harness(); const first = h.control(); await h.menu.load(first);
  h.api.loadPiCopilotConversationModels = async () => ({ models: ['new-catalogue'] });
  first.open = false; first.open = true; await h.menu.load(first);
  assert.match(first.html(), /new-catalogue/); assert.doesNotMatch(first.html(), /model-b/);
  const second = h.control(); await h.menu.load(second);
  assert.match(second.html(), /new-catalogue/);
  assert.equal(h.calls.filter(row => row[0] === 'session').length, 3);
  const reload = harness('zh', copy(h.state.session)); await reload.menu.load(reload.control());
  assert.equal(reload.state.session.model_connection.model, 'model-a');
});

test('delayed catalogue success or rejection cannot repaint another binding', async () => {
  for (const rejected of [false, true]) {
    const h = harness(); const pending = defer(); h.api.loadPiCopilotConversationModels = () => h.state.session.model_connection.provider === 'openai' ? pending.promise : Promise.resolve({ models: ['account-next'] });
    const old = h.control(); const loading = h.menu.load(old); await tick();
    h.state.session = session('codex', 'conversation-b');
    const current = h.control(); await h.menu.load(current);
    const before = current.html(), oldBefore = old.html();
    if (rejected) pending.reject(new Error('old request failed')); else pending.resolve({ models: ['stale-model'] });
    await loading;
    assert.equal(current.html(), before); assert.equal(old.html(), oldBefore);
    assert.equal(h.menu.state.current, 'model-a');
    assert.equal(h.menu.state.loading, false);
    assert.equal(h.menu.state.error, '');
  }
});

test('a changed binding, detached control or new busy task discards an outstanding response', async () => {
  for (const change of ['session', 'project', 'binding', 'model', 'research-model', 'chat-model', 'detached', 'busy']) {
    const h = harness(); const pending = defer(); h.api.loadPiCopilotConversationModels = () => pending.promise;
    const node = h.control(); const loading = h.menu.load(node); await tick(); const before = node.html();
    if (change === 'binding') h.state.session = session('codex');
    if (change === 'session') h.state.session = session('openai', 'conversation-b');
    if (change === 'project') { h.state.session.project_id = 'project-b'; h.project('project-b'); }
    if (change === 'model') h.state.session = session('openai', 'conversation-a', 'model-b');
    if (change === 'research-model') h.state.session.model_connection.model = 'model-b';
    if (change === 'chat-model') h.state.session.model.id = 'model-b';
    if (change === 'detached') node.isConnected = false;
    if (change === 'busy') h.state.busy = true;
    pending.resolve({ models: ['stale-model'] }); await loading;
    assert.equal(node.html(), before, change);
    assert.equal(h.menu.state.models.includes('stale-model'), false, change);
  }
});

test('a control rendered for a previous binding cannot initiate a new request', async () => {
  const h = harness(); const node = h.control(); h.state.session = session('codex', 'conversation-b');
  await h.menu.load(node);
  assert.equal(h.calls.length, 0);
});

test('an older request for the same binding cannot overwrite a newer reopen', async () => {
  const h = harness(); const pending = defer(); h.api.loadPiCopilotConversationModels = () => pending.promise;
  const old = h.control(); const loading = h.menu.load(old); await tick();
  h.api.loadPiCopilotConversationModels = async () => ({ models: ['latest-model'] });
  const current = h.control(); await h.menu.load(current); const before = current.html();
  pending.resolve({ models: ['old-model'] }); await loading;
  assert.equal(current.html(), before); assert.equal(h.menu.state.models.includes('old-model'), false);
});

test('wrong-session and changed-binding readbacks cannot start catalogue discovery', async () => {
  for (const kind of ['session', 'project', 'binding', 'conversation-model']) {
    const h = harness(); const other = copy(h.state.session);
    if (kind === 'session') other.session_id = 'other';
    if (kind === 'project') other.project_id = 'other';
    if (kind === 'binding') other.model_connection.model = 'other';
    if (kind === 'conversation-model') other.model.id = 'other';
    h.api.loadPiCopilotSession = async () => ({ session: other });
    const node = h.control(); await h.menu.load(node);
    assert.equal(h.calls.length, 0); assert.match(node.html(), /重新打开/);
    assert.equal(node.span.textContent, 'openai · model-a');
  }
});

test('a divergent conversation model is labelled separately and cannot use the saved research catalogue', async () => {
  for (const language of ['zh', 'en']) {
    const h = harness(language); h.state.session.model.id = 'chat-model';
    const html = h.view.sessionPanel();
    assert.ok(html.includes(h.tr('Conversation: chat-model', '对话：chat-model')));
    assert.ok(html.includes(h.tr('Analysis: openai · model-a', '分析：openai · model-a')));
    const node = h.control(); await h.menu.load(node);
    assert.equal(h.calls.length, 0);
    assert.ok(node.html().includes(h.tr('differ', '不一致')));
  }
});

test('a missing or legacy binding has readable labels in both languages', () => {
  for (const language of ['zh', 'en']) {
    const h = harness(language); h.state.session.model_connection = null; h.state.session.model = null;
    let html = h.view.sessionPanel();
    assert.ok(html.includes(h.tr('Model connection unavailable', '模型连接暂不可用')));
    assert.doesNotMatch(html, /legacy model binding/);
    h.state.session.model = { id: 'chat-only' };
    h.state.session.research_provider = { provider: 'codex', model: 'research-only' };
    html = h.view.sessionPanel();
    assert.ok(html.includes(h.tr('Conversation: chat-only', '对话：chat-only')));
    assert.ok(html.includes(h.tr('Analysis: codex · research-only', '分析：codex · research-only')));
  }
});

test('a delayed session readback cannot discover models for a newly selected conversation', async () => {
  const h = harness(); const pending = defer(); const saved = copy(h.state.session);
  h.api.loadPiCopilotSession = () => pending.promise;
  const old = h.control(); const loading = h.menu.load(old);
  h.state.session = session('codex', 'conversation-b');
  pending.resolve({ session: saved }); await loading;
  assert.equal(h.calls.length, 0);
  assert.equal(h.menu.state.models.length, 0);
});

test('empty, malformed and rejected catalogues never reuse a previous list', async () => {
  for (const payload of [{ models: [] }, { models: null }, null, 'reject']) {
    const h = harness(); const node = h.control(); await h.menu.load(node);
    h.api.loadPiCopilotConversationModels = async () => {
      if (payload === 'reject') throw new Error('synthetic catalogue unavailable');
      return payload;
    };
    await h.menu.load(node);
    assert.doesNotMatch(node.html(), /model-b/);
    assert.equal(h.menu.state.models.length, 0);
    assert.equal(h.menu.state.loading, false);
    assert.equal(node.span.textContent, 'openai · model-a');
  }
});

test('catalogue text is escaped and private binding fields never enter the menu key', async () => {
  const h = harness(); h.state.session.model_connection.account_session_sha256 = 'synthetic-private-locator';
  h.api.loadPiCopilotConversationModels = async () => ({ models: ['<img src=x>', {}, { id: 'one' }, { id: 'one' }] });
  const node = h.control(); await h.menu.load(node);
  assert.match(node.html(), /&lt;img/); assert.doesNotMatch(node.html(), /<img|\[object Object\]/);
  assert.equal((node.html().match(/data-gpi-pick-model="one"/g) || []).length, 1);
  assert.doesNotMatch(node.dataset.gpiModelKey, /synthetic-private-locator|account_session_sha256/);
});

test('busy, preparing and streaming conversations do not offer a switch', async () => {
  for (const kind of ['busy', 'projectLoading', 'planStarting', 'streaming']) {
    const h = harness();
    if (kind === 'stale') h.stale(true);
    else if (kind === 'streaming') h.state.session.streaming = true;
    else h.state[kind] = kind === 'workflow' ? { active_job: { present: true } } : true;
    const node = h.control(); await h.menu.load(node); h.clickModel(node); await tick();
    assert.equal(h.calls.length, 0, kind);
    assert.doesNotMatch(node.html(), /data-gpi-pick-model/);
  }
});

test('both languages offer terse model labels and retain the popover owner', async () => {
  for (const language of ['zh', 'en']) for (const hosted of [false, true]) {
    const h = harness(language); h.hosted(hosted); const node = h.control(); await h.menu.load(node);
    const panel = h.view.sessionPanel();
    assert.equal(panel.includes('data-gpi-config'), !hosted);
    assert.match(node.html(), />model-b<\/span>/);
    assert.doesNotMatch(node.html(), /<small|new conversation|新对话|gpi-model-note/);
    assert.match(panel, /<details class="gpi-model-control" data-popover-menu>/);
  }
});

for (const [code, en, zh] of [
  ['pi_session_busy', 'Wait for the current reply', '请等待当前回复'],
  ['pi_session_model_unavailable', 'not available', '没有此模型'],
  ['pi_session_project_mismatch', 'another project', '另一个项目'],
  ['pi_session_connection_changed', 'saved connection changed', '连接已变化'],
  ['pi_session_connection_unpinned', 'Start a new conversation', '这条对话创建于旧版本，不能切换模型。新建对话即可选择其他模型。'],
]) test('typed refusal ' + code + ' is localized and preserves the model', async () => {
  for (const language of ['zh', 'en']) {
    const h = harness(language), node = h.control(); await h.menu.load(node);
    h.api.setPiCopilotConversationModel = async () => { throw Object.assign(new Error('internal detail'), { code }); };
    h.clickModel(node); await tick();
    assert.ok(node.html().includes(language === 'zh' ? zh : en), node.html());
    assert.equal(h.state.session.model_connection.model, 'model-a');
    assert.equal(node.open, true);
    assert.doesNotMatch(node.html(), /internal detail/);
  }
});

test('selection races cannot mutate another conversation or repeat an in-flight write', async () => {
  for (const change of ['session', 'project', 'busy', 'detached']) {
    const h = harness(), node = h.control(); await h.menu.load(node);
    const pending = defer(), old = copy(h.state.session);
    let calls = 0;
    h.api.setPiCopilotConversationModel = () => { ++calls; return pending.promise; };
    h.clickModel(node); h.clickModel(node); await tick();
    assert.equal(calls, 1);
    if (change === 'session') h.state.session = session('codex', 'conversation-b');
    if (change === 'project') h.project('project-b');
    if (change === 'busy') h.state.busy = true;
    if (change === 'detached') node.isConnected = false;
    pending.resolve({ session: { ...old, model: { id: 'model-b' }, model_connection: { ...old.model_connection, model: 'model-b' } } });
    await tick();
    assert.equal(h.state.session.model_connection.model, 'model-a');
    assert.equal(node.span.textContent, 'openai · model-a');
  }
});

test('unacknowledged and unknown model selections never update the saved choice', async () => {
  const h = harness(), node = h.control(); await h.menu.load(node);
  h.clickModel(node, 'unknown'); await tick();
  assert.equal(h.calls.filter(row => row[0] === 'switch').length, 0);
  h.api.setPiCopilotConversationModel = async () => ({ session: session('openai', 'other', 'model-b') });
  h.clickModel(node); await tick();
  assert.equal(h.state.session.model_connection.model, 'model-a');
  assert.match(node.html(), /无法确认/);
});

test('each acknowledgement coordinate must match before accepting a model change', async () => {
  for (const field of ['project', 'provider', 'authentication', 'credential', 'connection-model', 'runtime-model']) {
    const h = harness(), node = h.control(); await h.menu.load(node);
    const updated = copy(h.state.session);
    updated.model_connection.model = updated.model.id = 'model-b';
    if (field === 'project') updated.project_id = 'another-project';
    if (field === 'provider') updated.model_connection.provider = 'codex';
    if (field === 'authentication') updated.model_connection.authentication_mode = 'chatgpt_account';
    if (field === 'credential') updated.model_connection.credential_source = 'codex_user_auth';
    if (field === 'connection-model') updated.model_connection.model = 'other-model';
    if (field === 'runtime-model') updated.model.id = 'other-model';
    h.api.setPiCopilotConversationModel = async () => ({ session: updated });
    h.clickModel(node); await tick();
    assert.equal(h.state.session.model_connection.model, 'model-a', field);
    assert.equal(node.span.textContent, 'openai · model-a', field);
    assert.equal(node.open, true, field);
    assert.match(node.html(), /无法确认/, field);
  }
});

test('the everyday header omits presentation controls and the stale reviewer branch is gone', () => {
  for (const language of ['zh', 'en']) {
    const h = harness(language); const html = h.view.sessionPanel();
    assert.doesNotMatch(html, /data-gpi-presentation-pin|data-gpi-demo(?:\s|>)/);
    assert.match(html, /data-gpi-study-setup/);
    const workflow = h.view.workflowHtml({ kind: 'reviewer_validation_demo', stages: [{ id: 'question' }, { id: 'setup' }, { id: 'manuscript' }] });
    assert.ok(workflow.includes(h.tr('Research workflow', '科研流程')));
    assert.ok(workflow.includes(h.tr('Question', '问题')));
    assert.doesNotMatch(workflow, /Reviewer workflow|审稿协议|数据合同|审稿报告/);
    assert.match(h.view.projectRequiredPanel(), language === 'zh'
      ? /data-gpi-demo>.*演示：从提问到结果/
      : /data-gpi-demo>.*Demo: from question to result/);
  }
  const jsRoot = path.dirname(owners[0]);
  assert.match(fs.readFileSync(path.join(jsRoot, 'screens-guided-pi-demo.js'), 'utf8'), /research_workflow_demo/);
});


test('an admitted scientific run does not freeze the next conversation model', async () => {
  const h = harness(); h.state.childJobId = 'scientific-job';
  h.state.workflow = { active_job: { present: true } }; h.stale(true);
  const node = h.control(); await h.menu.load(node); h.clickModel(node); await tick();
  assert.equal(h.state.session.model_connection.model, 'model-b');
  assert.equal(h.state.childJobId, 'scientific-job');
});
