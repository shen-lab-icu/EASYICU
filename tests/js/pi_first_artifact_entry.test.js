'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const js = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const read = name => fs.readFileSync(path.join(js, `screens-guided-pi-${name}.js`), 'utf8');
const shell = fs.readFileSync(path.join(js, 'screens-guided-pi.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
function context(names) {
  const ctx = vm.createContext({ window: { EU_LANG: 'zh', EU_HTML: { esc } } });
  vm.runInContext(read('modules'), ctx);
  names.forEach(name => vm.runInContext(read(name), ctx));
  return ctx;
}
function section(begin, end) {
  const start = shell.indexOf(begin), stop = shell.indexOf(end, start);
  assert.ok(start >= 0 && stop > start);
  return shell.slice(start, stop);
}
function wire(ctx, state, switchMode, sent) {
  let click;
  state.host = { addEventListener(name, fn) { if (name === 'click') click = fn; }, querySelector() { return null; } };
  ctx.window.EasyICU.guidedPi.require('events').create({ state,
    tr: en => en, RUN_FILES: { handleClick: () => false }, MESSAGE_ACTIONS: { handleClick: () => false },
    STARTERS: { actionFromEvent: () => null }, COHORT_ELIGIBILITY: { actionFromEvent: () => null },
    STUDY_WORKSPACE: { removeReference() {}, removeSkill() {}, removeMethod() {} },
    switchMode, sendText: () => sent.push('sent'),
  }).wire();
  return () => click({ target: { closest: selector => selector === '[data-gpi-new-workspace]' ? {} : null },
    preventDefault() {}, stopPropagation() {} });
}

test('a new project with no session has a visible explicit artifact entry in both UI languages', () => {
  for (const language of ['zh', 'en']) {
    const ctx = context(['session-view']);
    const state = { session: null, sessions: [], project: { title: 'Fresh project' }, draft: 'Keep this draft' };
    const view = ctx.window.EasyICU.guidedPi.require('sessionView').create({ state,
      modules: ctx.window.EasyICU.guidedPi, tr: (en, zh) => language === 'zh' ? zh : en,
      esc, iconHtml: () => '', projectId: () => 'draft_fresh', sessionMatchesUiLanguage: () => true,
      providerBindingSummary: () => '', resourceOwner: { button: () => '' },
    });
    const html = view.activatePanel();
    assert.match(html, /class="btn" type="button" data-gpi-new-workspace/);
    assert.match(html, language === 'zh' ? /新建产物任务/ : /New artifact task/);
    assert.match(html, /data-gpi-create/);
    assert.doesNotMatch(html.match(/<button[^>]*data-gpi-new-workspace[^>]*>/)[0], /hidden|disabled/);
    assert.equal(state.session, null); assert.equal(state.draft, 'Keep this draft');
    for (const flag of ['creating', 'projectLoading', 'busy']) {
      state[flag] = true;
      assert.match(view.activatePanel().match(/<button[^>]*data-gpi-new-workspace[^>]*>/)[0], /disabled/);
      state[flag] = false;
    }
    state.projectIssue = 'pi_project_study_context_missing';
    assert.doesNotMatch(view.activatePanel(), /data-gpi-new-workspace/);
  }
});

test('the first artifact action creates one workspace session in the current project, preserving its draft without sending', async () => {
  const ctx = context(['events']), calls = [], sent = [], pending = [];
  const modules = ctx.window.EasyICU.guidedPi;
  modules.declare('replay', {});
  modules.declare('project', { syncLocation: (...args) => calls.push(['location', ...args]) });
  const state = { session: null, sessions: [{ session_id: 'old_project_workspace', agent_mode: 'workspace' }],
    sessionSelectionRevision: 0, agentMode: 'research', project: { title: 'Fresh project' },
    projectInitialization: { required: true }, draft: 'Create the current project’s synthetic 24-row brief',
    taskDraft: 'stale task', researchProvider: 'openai', researchModel: 'unchanged-model', messages: [] };
  let finishCreation;
  Object.assign(ctx, { state, MODULES: modules, IDEA_SOURCE: { reset() {} },
    agentMode: () => state.agentMode, projectId: () => 'draft_fresh', uiLanguage: () => 'zh',
    closeSource() {}, closeChildSource() {}, closeSourceView() {}, render() {}, hydrateProjectedJob() {},
    displayProjectTitle: String, tr: (_en, zh) => zh, errorText: String,
    CONNECT: { ask: () => true }, EFFORT_MENU: { preferred: () => 'high' },
    STUDY_WORKSPACE: { applyHubIntent() {}, takeHubAttachment: () => null },
    transcriptMessages: () => [], rememberSession: id => calls.push(['remember', id]),
    openSession: id => { throw new Error('Must not open the old project session: ' + id); },
    loadWorkflow: async () => {},
    api: () => ({
      async initializePiCopilotProject(value) { calls.push(['initialize', value]); return { status: 'ready' }; },
      createPiCopilotSession(value) { calls.push(['create', value]); return new Promise(resolve => { finishCreation = resolve; }); },
      loadPiCopilotSessions() { throw new Error('A new task must not select a saved session'); },
      sendPiCopilotMessage() { sent.push('api-send'); throw new Error('Entry must not send a message'); },
    }),
  });
  vm.runInContext(section('  async function createSession()', '  async function renameSession('), ctx);
  vm.runInContext(section('  async function switchMode(', '  const LIVE_STREAM ='), ctx);
  const click = wire(ctx, state, (...args) => { const work = ctx.switchMode(...args); pending.push(work); return work; }, sent);
  click(); click();
  assert.equal(state.creating, true);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls.filter(row => row[0] === 'create').length, 1);
  const created = calls.find(row => row[0] === 'create')[1];
  assert.equal(created.project_id, 'draft_fresh'); assert.equal(created.agent_mode, 'workspace');
  assert.equal(created.research_provider, 'openai');
  assert.equal(calls.find(row => row[0] === 'initialize')[1].project_id, 'draft_fresh');
  finishCreation({ session: { session_id: 'fresh_workspace', agent_mode: 'workspace', transcript: [] } });
  await Promise.all(pending);
  assert.equal(state.session.session_id, 'fresh_workspace'); assert.equal(state.agentMode, 'workspace');
  assert.equal(state.draft, 'Create the current project’s synthetic 24-row brief');
  assert.equal(state.researchProvider, 'openai'); assert.equal(state.researchModel, 'unchanged-model');
  assert.deepEqual(sent, []); assert.equal(state.creating, false);
  assert.equal(calls.filter(row => row[0] === 'create').length, 1);
});

test('stale artifact-entry clicks are inert during creation, project loading or an active turn', () => {
  for (const lock of [{ creating: true }, { projectLoading: true }, { busy: true }, { session: { streaming: true } }]) {
    const ctx = context(['events']), calls = [], sent = [], state = { draft: 'Keep draft', messages: [], ...lock };
    wire(ctx, state, (...args) => calls.push(args), sent)();
    assert.deepEqual(calls, []); assert.deepEqual(sent, []); assert.equal(state.draft, 'Keep draft');
  }
});
