'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const shell = fs.readFileSync(path.join(JS, 'screens-guided-pi.js'), 'utf8');
function section(start, end) {
  const begin = shell.indexOf(start);
  const finish = shell.indexOf(end, begin);
  assert.ok(begin >= 0 && finish > begin, `production section exists: ${start}`);
  return shell.slice(begin, finish);
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function row(id, extra = {}) { return { session_id: id, language: 'zh', messages: [], ...extra }; }
function plain(value) { return JSON.parse(JSON.stringify(value)); }
const flush = () => new Promise(resolve => setImmediate(resolve));

function harness(overrides = {}) {
  const state = {
    project: { id: 'project-a', title: 'Project A' }, host: {},
    session: row('session-a'), sessions: [row('session-a')], messages: [],
    workflow: {}, workflowReceipts: [], currentTurnResources: [],
    sessionSelectionRevision: 0, pendingLanguageReload: false,
    agentMode: 'research', researchProvider: 'test-provider', researchModel: '',
    busy: false, childJobId: '', creating: false, projectLoading: false,
    error: '', draft: '', taskDraft: '', ...overrides,
  };
  const calls = { create: [], load: [], list: [], rename: [], remove: [], remember: [], sync: [], opened: [], transcriptUpdates: [], closeSource: 0, closeChild: 0, previewClose: 0, renders: 0, removeReference: 0, removeSkill: 0, removeMethod: 0 };
  const api = {
    createPiCopilotSession: async body => { calls.create.push(plain(body)); return { session: row('created-a') }; },
    loadPiCopilotSession: async (id, project) => { calls.load.push({ id, project }); return { session: row(id) }; },
    loadPiCopilotSessions: async (limit, project) => { calls.list.push({ limit, project }); return { sessions: [] }; },
    renamePiCopilotSession: async (id, body) => { calls.rename.push({ id, ...plain(body) }); return { session: row(id, { title: body.title }) }; },
    deletePiCopilotSession: async (id, body) => { calls.remove.push({ id, ...plain(body) }); },
  };
  const context = vm.createContext({
    state, URLSearchParams, document: {}, window: { EU_LANG: 'zh', location: { search: '' }, prompt: () => '  Renamed\nconversation  ' },
    projectId: () => state.project.id, api: () => api,
    tr: (_en, zh) => zh, displayProjectTitle: (title, fallback) => title || fallback,
    connectionReady: () => true, CONNECT: { ask: () => true },
    closeSourceView() {}, closeSource() { calls.closeSource++; }, closeChildSource() { calls.closeChild++; },
    EFFORT_MENU: { preferred: () => 'medium' },
    loadWorkflow: async () => {}, transcriptMessages: session => session.messages || [],
    TRANSCRIPT: { markSupersededPlanAttempts: (messages, session) => calls.transcriptUpdates.push({ messages, session }) },
    hydrateProjectedJob() {}, errorText: error => error.message || String(error),
    render() { calls.renders++; }, navigationSessionTitle: session => session.title || session.session_id,
    sessionMatchesUiLanguage: session => session && session.language === context.window.EU_LANG,
    rememberedSession: () => '', uiLanguage: () => context.window.EU_LANG,
    rememberSession: id => calls.remember.push({ project: state.project.id, id }),
    setShell() {}, reconcileSettledSession() {}, watchJob() {},
    HOST_JOBS: { sync() {} }, PLAN_ACTIONS: { continueSystemOwnedPlanProgression: async () => {} },
    STUDY_WORKSPACE: {
      applyHubIntent() {}, takeHubAttachment: () => null,
      isEmptyConversation: session => session.empty === true,
      removeReference() { calls.removeReference++; },
      removeSkill() { calls.removeSkill++; }, removeMethod() { calls.removeMethod++; },
    },
  });
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-guided-pi-modules.js'), 'utf8'), context);
  const modules = context.window.EasyICU.guidedPi;
  modules.declare('project', {
    requestedSessionId: () => '',
    syncLocation: (project, id) => calls.sync.push({ project, id }),
  });
  modules.declare('preview', { close() { calls.previewClose++; } });
  modules.declare('replay', { hydrate: async (_api, session) => session });
  context.MODULES = modules;
  vm.runInContext(section('  async function openSession(', '  function assistantRow('), context);
  const productionOpen = context.openSession;
  context.openSession = async (...args) => { calls.opened.push(args[0]); return productionOpen(...args); };
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-guided-pi-session-navigation.js'), 'utf8'), context);
  const nav = modules.require('sessionNavigation').create({
    state, api: context.api, projectId: context.projectId,
    connectionReady: context.connectionReady, closeSourceView: context.closeSourceView, CONNECT: context.CONNECT,
    displayProjectTitle: context.displayProjectTitle, tr: context.tr, EFFORT_MENU: context.EFFORT_MENU,
    loadWorkflow: (...args) => context.loadWorkflow(...args), transcriptMessages: context.transcriptMessages,
    STUDY_WORKSPACE: context.STUDY_WORKSPACE, hydrateProjectedJob: context.hydrateProjectedJob,
    rememberSession: context.rememberSession, errorText: context.errorText, render: context.render,
    navigationSessionTitle: context.navigationSessionTitle, sessionMatchesUiLanguage: context.sessionMatchesUiLanguage,
    rememberedSession: context.rememberedSession, openSession: context.openSession, uiLanguage: context.uiLanguage,
    closeSource: context.closeSource, closeChildSource: context.closeChildSource, setShell: context.setShell,
  });
  function switchProject() {
    state.project = { id: 'project-b', title: 'Project B' };
    state.sessionSelectionRevision++;
    state.session = row('session-b');
    state.sessions = [row('session-b')];
    state.messages = [{ role: 'user', text: 'Project B question' }];
    state.error = 'Project B current status';
    state.creating = false;
  }
  return { state, calls, api, context, nav, modules, switchProject };
}

test('a delayed create response cannot select its conversation in another project', async () => {
  const h = harness();
  const pending = deferred();
  h.api.createPiCopilotSession = body => { h.calls.create.push(plain(body)); return pending.promise; };
  const run = h.nav.createSession();
  assert.equal(h.calls.create[0].project_id, 'project-a');
  h.switchProject();
  pending.resolve({ session: row('late-a') });
  await run;
  assert.equal(h.state.session.session_id, 'session-b');
  assert.equal(h.state.messages[0].text, 'Project B question');
  assert.deepEqual(h.calls.remember, []);
  assert.deepEqual(h.calls.sync, []);
});

test('a delayed create error leaves the current project error intact', async () => {
  const h = harness();
  const pending = deferred();
  h.api.createPiCopilotSession = () => pending.promise;
  const run = h.nav.createSession();
  h.switchProject();
  pending.reject(new Error('Project A request failed'));
  await run;
  assert.equal(h.state.error, 'Project B current status');
});

test('a project switch while initialization is pending stops the old create', async () => {
  const h = harness({ projectInitialization: { required: true } });
  const pending = deferred();
  const initialized = [];
  h.api.initializePiCopilotProject = body => { initialized.push(plain(body)); return pending.promise; };
  const run = h.nav.createSession();
  h.switchProject();
  pending.resolve({ status: 'ready' });
  await run;
  assert.equal(initialized[0].project_id, 'project-a');
  assert.deepEqual(h.calls.create, []);
  assert.equal(h.state.session.session_id, 'session-b');
});

test('a project switch while initialized workflow loads stops the old create', async () => {
  const h = harness({ projectInitialization: { required: true } });
  const pending = deferred();
  let loading = 0;
  h.api.initializePiCopilotProject = async () => ({ status: 'ready' });
  h.context.loadWorkflow = () => { loading++; return pending.promise; };
  const run = h.nav.createSession();
  await flush();
  assert.equal(loading, 1);
  h.switchProject();
  pending.resolve();
  await run;
  assert.deepEqual(h.calls.create, [], 'an abandoned create must not create a conversation with the new project title');
  assert.equal(h.state.session.session_id, 'session-b');
});

test('a delayed hub attachment cannot finish selecting the old project session', async () => {
  const h = harness();
  const pending = deferred();
  const attached = [];
  h.context.STUDY_WORKSPACE.takeHubAttachment = () => ({ source: 'catalog', id: 'test-skill' });
  h.modules.declare('skillAttach', { ensure: (project, session, source, id) => {
    attached.push({ project, session: session.session_id, source, id });
    return pending.promise;
  } });
  const run = h.nav.createSession();
  await flush();
  assert.equal(attached.length, 1);
  h.switchProject();
  pending.resolve();
  await run;
  assert.equal(h.state.session.session_id, 'session-b');
  assert.equal(h.state.error, 'Project B current status');
  assert.deepEqual(h.calls.remember, []);
  assert.deepEqual(h.calls.sync, []);
});

test('an old create finishing cannot clear another project active create flag', async () => {
  const h = harness();
  const pending = deferred();
  h.api.createPiCopilotSession = () => pending.promise;
  const run = h.nav.createSession();
  h.switchProject();
  h.state.creating = true;
  pending.resolve({ session: row('late-a') });
  await run;
  assert.equal(h.state.creating, true);
  assert.equal(h.state.session.session_id, 'session-b');
});

test('a delayed project-session list cannot replace another project list or open a row', async () => {
  const h = harness();
  const pending = deferred();
  h.api.loadPiCopilotSessions = (limit, project) => { h.calls.list.push({ limit, project }); return pending.promise; };
  const run = h.nav.loadProjectSessions(false);
  h.switchProject();
  pending.resolve({ sessions: [row('late-a')] });
  await run;
  assert.equal(h.calls.list[0].project, 'project-a');
  assert.deepEqual(plain(h.state.sessions).map(session => session.session_id), ['session-b']);
  assert.deepEqual(h.calls.opened, []);
});

test('the production openSession ignores a delayed session response after project switch', async () => {
  const h = harness();
  const pending = deferred();
  h.api.loadPiCopilotSession = (id, project) => { h.calls.load.push({ id, project }); return pending.promise; };
  const run = h.context.openSession('session-a-2');
  h.switchProject();
  pending.resolve({ session: row('session-a-2') });
  await run;
  assert.deepEqual(h.calls.load[0], { id: 'session-a-2', project: 'project-a' });
  assert.equal(h.state.session.session_id, 'session-b');
  assert.deepEqual(h.calls.remember, []);
});

test('the production openSession ignores a delayed error after project switch', async () => {
  const h = harness();
  const pending = deferred();
  h.api.loadPiCopilotSession = () => pending.promise;
  const run = h.context.openSession('session-a-2');
  h.switchProject();
  pending.reject(new Error('Project A session failed'));
  await run;
  assert.equal(h.state.error, 'Project B current status');
  assert.deepEqual(h.calls.remember, []);
});

test('a session workflow finishing after project switch cannot progress the new project', async () => {
  const h = harness();
  const pending = deferred();
  const progressed = [];
  let loading = 0;
  h.context.loadWorkflow = () => { loading++; return pending.promise; };
  h.context.PLAN_ACTIONS.continueSystemOwnedPlanProgression = async () => progressed.push(h.state.project.id);
  const run = h.context.openSession('session-a-2');
  await flush();
  assert.equal(loading, 1);
  assert.equal(h.calls.transcriptUpdates.length, 1);
  assert.equal(h.calls.transcriptUpdates[0].session.session_id, 'session-a-2');
  h.switchProject();
  pending.resolve();
  await run;
  assert.deepEqual(progressed, []);
  assert.equal(h.state.session.session_id, 'session-b');
});

test('a language change waits for the active turn before replacing its session', async () => {
  const h = harness({ busy: true, jobId: 'active-message-job' });
  h.context.window.EU_LANG = 'en';
  h.nav.handleLanguageChange();
  assert.equal(h.state.pendingLanguageReload, true);
  assert.equal(h.state.session.session_id, 'session-a');
  assert.equal(h.state.busy, true);
  assert.equal(h.state.jobId, 'active-message-job');
  assert.equal(h.calls.closeSource, 0);
  assert.deepEqual(h.calls.list, []);
  // This is the same owner entry point invoked by watchJob after the server
  // turn settles; no server turn is fabricated or advanced by this test.
  h.state.busy = false;
  await h.nav.reloadSessionsForLanguage();
  assert.equal(h.state.pendingLanguageReload, false);
  assert.equal(h.state.jobId, '');
  assert.equal(h.calls.closeSource, 1);
  assert.deepEqual(h.calls.list, [{ limit: 100, project: 'project-a' }]);
});

test('new conversation reuses the currently empty draft without creating or opening another', async () => {
  const h = harness();
  await h.nav.startEntry();
  assert.equal(h.state.session.session_id, 'session-a');
  assert.deepEqual(h.calls.create, []);
  assert.deepEqual(h.calls.opened, []);
});

test('new conversation reuses an existing empty draft of the current project', async () => {
  const h = harness({ messages: [{ role: 'user', text: 'Existing discussion' }], sessions: [row('session-a'), row('empty-a', { empty: true })] });
  await h.nav.startEntry();
  assert.equal(h.state.session.session_id, 'empty-a');
  assert.deepEqual(h.calls.create, []);
  assert.deepEqual(h.calls.opened, ['empty-a']);
  assert.deepEqual(h.calls.load, [{ id: 'empty-a', project: 'project-a' }]);
});

test('rename sends a normalized title with the selected project identity', async () => {
  const h = harness();
  await h.nav.renameSession(h.state.session);
  assert.deepEqual(h.calls.rename, [{ id: 'session-a', project_id: 'project-a', title: 'Renamed conversation' }]);
  assert.equal(h.state.session.title, 'Renamed conversation');
  assert.equal(h.state.sessions[0].title, 'Renamed conversation');
});

test('a late rename response cannot clear another project status', async () => {
  const h = harness();
  const pending = deferred();
  h.api.renamePiCopilotSession = () => pending.promise;
  const run = h.nav.renameSession(h.state.session);
  h.switchProject();
  pending.resolve({ session: row('session-a', { title: 'Renamed conversation' }) });
  await run;
  assert.equal(h.state.error, 'Project B current status');
  assert.equal(h.state.session.session_id, 'session-b');
});

test('remove sends the selected project identity and opens the next matching conversation', async () => {
  const h = harness({ sessions: [row('session-a'), row('session-a-next')] });
  await h.nav.removeSession(h.state.session);
  assert.deepEqual(h.calls.remove, [{ id: 'session-a', project_id: 'project-a', confirm: true }]);
  assert.equal(h.state.session.session_id, 'session-a-next');
  assert.deepEqual(h.calls.opened, ['session-a-next']);
});

test('a delayed selected-conversation delete cannot clear the new project conversation', async () => {
  const h = harness();
  const pending = deferred();
  h.api.deletePiCopilotSession = () => pending.promise;
  const run = h.nav.removeSession(h.state.session);
  h.switchProject();
  pending.resolve({});
  await run;
  assert.equal(h.state.session.session_id, 'session-b');
  assert.equal(h.state.messages[0].text, 'Project B question');
  assert.equal(h.state.error, 'Project B current status');
  assert.deepEqual(h.calls.remember, []);
  assert.deepEqual(h.calls.sync, []);
});

test('deleting an old selected conversation cannot clear a newer selection in the same project', async () => {
  const h = harness({ sessions: [row('session-a'), row('session-a-next')] });
  const pending = deferred();
  h.api.deletePiCopilotSession = () => pending.promise;
  const run = h.nav.removeSession(h.state.session);
  await h.context.openSession('session-a-next');
  const rememberedBefore = h.calls.remember.length;
  const syncedBefore = h.calls.sync.length;
  pending.resolve({});
  await run;
  assert.equal(h.state.session.session_id, 'session-a-next');
  assert.equal(h.calls.remember.length, rememberedBefore, 'do not clear then restore the new selected conversation');
  assert.equal(h.calls.sync.length, syncedBefore, 'do not replace the new conversation URL');
});

test('stale default offers a recovery only for the current project after failed create', async () => {
  const h = harness({ session: null });
  let callback;
  h.modules.declare('skillAttach', {
    noteDefaultFailure: async (project, error) => project === 'project-a' && error.code === 'skill_attachment_stale',
    bindDefaultRecovery: (_host, _project, onResult) => { callback = onResult; },
    clearDefaultFailure() {},
  });
  h.api.createPiCopilotSession = async () => { throw Object.assign(new Error('stale'), { code: 'skill_attachment_stale' }); };
  await h.nav.createSession();
  assert.equal(h.state.session, null);
  assert.match(h.state.error, /项目默认技能已更改或不可用/);
  assert.equal(typeof callback, 'function');
  callback('project-a', true, '');
  assert.match(h.state.error, /已移除本项目默认技能/);
});

test('late stale-default lookup cannot attach recovery to a different project', async () => {
  const h = harness({ session: null });
  const pending = deferred();
  let bound = false;
  let isCurrent;
  h.modules.declare('skillAttach', {
    noteDefaultFailure: (_project, _error, predicate) => { isCurrent = predicate; return pending.promise; },
    bindDefaultRecovery: () => { bound = true; },
  });
  h.api.createPiCopilotSession = async () => { throw Object.assign(new Error('stale'), { code: 'skill_attachment_stale' }); };
  const run = h.nav.createSession();
  await flush();
  h.switchProject();
  assert.equal(isCurrent(), false);
  pending.resolve(true);
  await run;
  assert.equal(bound, false);
  assert.equal(h.state.error, 'Project B current status');
});

test('a stale hub attachment after session creation is not mistaken for a stale project default', async () => {
  const h = harness({ session: null });
  let checked = false;
  h.modules.declare('skillAttach', { noteDefaultFailure: async () => { checked = true; return true; }, clearDefaultFailure() {} });
  h.context.STUDY_WORKSPACE.applyHubIntent = () => { throw Object.assign(new Error('attachment stale'), { code: 'skill_attachment_stale' }); };
  await h.nav.createSession();
  assert.equal(h.state.session.session_id, 'created-a');
  assert.equal(h.state.error, 'attachment stale');
  assert.equal(checked, false);
});
