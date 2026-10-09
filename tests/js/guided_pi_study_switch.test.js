/* A study write that waits belongs to the conversation it started in. The
   researcher can open another conversation while it waits; the write then
   stops instead of landing on the study that is open now. Covered here with
   the browser's own StudyContext store over an in-memory host:
   - the opening question a source confirmation carries into the setup;
   - the demo source a host notice binds to the study;
   - the study a data-source selection activates;
   - the continue message the shell sends after the carry. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const [modulesFile, bindingFile, hostJobsFile, storeFile, shellFile] = process.argv.slice(2).map(file => path.resolve(file));
global.window = global;
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
const storage = new Map();
const listeners = {};
global.localStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, String(value)), removeItem: key => storage.delete(key) };
global.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
global.addEventListener = (type, handler) => (listeners[type] = listeners[type] || []).push(handler);
global.dispatchEvent = event => (listeners[event.type] || []).forEach(handler => handler(event));
global.document = { readyState: 'complete', addEventListener: global.addEventListener, dispatchEvent: global.dispatchEvent };
global.location = { hash: '#guided' };

// The host saves in memory. An activation (a save that names only the id)
// is recorded, and the held one waits until the test releases it.
const saved = new Map();
const activations = [];
const held = { id: '', reached: null, released: null, release: null };
global.EU_API = {
  loadActiveStudyContext: async () => ({ context: null }),
  listStudyContexts: async () => ({ contexts: [], active_id: null }),
  loadStudyContext: async id => ({ context: { ...saved.get(id) } }),
  saveStudyContext: async body => {
    if (Object.keys(body).length === 1) {
      activations.push(body.id);
      if (body.id === held.id) { held.id = ''; held.reached(); await held.released; }
      return { context: { ...saved.get(body.id) } };
    }
    const { expected_revision: _revision, ...fields } = body;
    const row = { ...saved.get(body.id), ...fields, revision: ((saved.get(body.id) || {}).revision || 0) + 1 };
    saved.set(body.id, row);
    return { context: { ...row } };
  },
};
require(modulesFile);
require(bindingFile);
require(hostJobsFile);
require(storeFile);
const modules = global.EasyICU.guidedPi;
const store = global.EU_STUDY_CONTEXT;
const quietHydrate = async () => store.active();
store.hydrate = quietHydrate;
function hold(id) {
  held.id = id;
  held.released = new Promise(resolve => { held.release = resolve; });
  return new Promise(resolve => { held.reached = resolve; });
}

let session = null;
const conversation = (id, studyId) => ({ session_id: id, binding: studyId ? { study_context_id: studyId } : {} });
const errors = [];
function bindingOwner(question) {
  return modules.require('dataBinding').create({
    api: () => ({ authorizePiCopilotDataSource: async () => ({ session, resource: { study_context_id: session.binding.study_context_id } }) }),
    render() {}, projectId: () => 'project_current', loadWorkflow: async () => {}, dataConsent: {},
    errorText: String, rememberSession() {}, continueAfterDataSourceConfirmation: async () => {},
    session: () => session, busy: () => false, setError(error) { if (error) errors.push(error); }, setSession() {}, rebind: async () => {},
    workflow: () => ({ missing_setup_fields: ['question'] }), researchQuestion: () => question,
  });
}
const demo = { ok: true, id: 'src_demo', path: '/demo/eicu', label: 'eICU demo', database: 'eicu_demo' };
global.EU_SOURCES = { activeSource: () => demo };
let receipts = [];
let onRebind = async () => {};
let authorized = 0;
const hostJobs = modules.require('hostJobs').create({
  tr: en => en, api: () => ({}), iconHtml: () => '', busy: () => false, messages: () => [],
  dataConsent: { requiresConfirmation: () => true, selectionInProgress: () => false },
  session: () => session, workflowReceipts: () => receipts, setWorkflowReceipts: rows => { receipts = rows; },
  setError(error) { if (error) errors.push(error); }, errorText: String, render() {},
  rebind: () => onRebind(), authorizeDataSource: async () => { authorized += 1; }, confirmDataSourceBinding: async () => {},
});
const notice = () => {
  receipts = [{ id: 'demo-notice', role: 'host_notice', session_id: session.session_id, source_label: demo.label }];
  return receipts[0];
};

(async () => {
  const studyA = store.startNew({ title: 'Project A', question: '' }, { persist: false });
  await store.persist();
  const studyB = store.startNew({ title: 'Project B', question: 'Original question for project B' }, { persist: false });
  await store.persist();

  // The question carry of conversation A waits on activating study A while
  // the researcher opens conversation B: neither study gets A's question.
  session = conversation('session-A', studyA.id);
  let reached = hold(studyA.id);
  let pending = bindingOwner('Research question from project A').carryQuestionIntoSetup();
  await reached;
  session = conversation('session-B', studyB.id);
  let switched = store.activate(studyB.id);
  held.release();
  await switched;
  assert.equal(await pending, false);
  assert.equal(saved.get(studyB.id).question, 'Original question for project B');
  assert.equal(saved.get(studyA.id).question, '');
  // Undisturbed, it activates its own study and saves the question there.
  session = conversation('session-A', studyA.id);
  assert.equal(await bindingOwner('Research question from project A').carryQuestionIntoSetup(), true);
  assert.equal(saved.get(studyA.id).question, 'Research question from project A');
  assert.equal(saved.get(studyB.id).question, 'Original question for project B');
  // A conversation bound to no study has nowhere to save it.
  session = conversation('session-C', '');
  assert.equal(await bindingOwner('A question').carryQuestionIntoSetup(), false);

  // The demo source of conversation A waits the same way: study B keeps its
  // source, and A's notice can be used again when A is reopened.
  await store.activate(studyB.id);
  const sourceA = saved.get(studyA.id).data_source;
  const sourceB = saved.get(studyB.id).data_source;
  session = conversation('session-A', studyA.id);
  let row = notice();
  reached = hold(studyA.id);
  pending = hostJobs.handleAction('use', row.id);
  await reached;
  session = conversation('session-B', studyB.id);
  switched = store.activate(studyB.id);
  held.release();
  await switched;
  await pending;
  assert.deepEqual(saved.get(studyB.id).data_source, sourceB);
  assert.deepEqual(saved.get(studyA.id).data_source, sourceA);
  assert.equal(row.pending, false, 'The notice is usable again in conversation A');
  session = conversation('session-A', studyA.id);
  row = notice();
  await hostJobs.handleAction('use', row.id);
  assert.equal(saved.get(studyA.id).data_source.path, demo.path);
  assert.deepEqual(saved.get(studyB.id).data_source, sourceB);
  // Unbound, it says so and writes nothing.
  session = conversation('session-C', '');
  row = notice();
  errors.length = 0;
  await hostJobs.handleAction('use', row.id);
  assert.match(errors.join(' '), /not linked to a study/);
  // A rebind that leaves conversation A bound to another study authorizes
  // no source for it.
  session = conversation('session-A', studyA.id);
  row = notice();
  const authorizedBefore = authorized;
  onRebind = async () => { session = conversation('session-A', studyB.id); };
  await hostJobs.handleAction('use', row.id);
  onRebind = async () => {};
  assert.equal(authorized, authorizedBefore);
  assert.equal(row.pending, false);

  // A data-source selection in conversation A refreshes the stores first;
  // if the researcher opens conversation B meanwhile, study A is not made
  // the active study over B, here or on the host.
  await store.activate(studyB.id);
  session = conversation('session-A', studyA.id);
  let releaseHydrate;
  let hydrateReached;
  const hydrated = new Promise(resolve => { hydrateReached = resolve; });
  store.hydrate = () => { hydrateReached(); return new Promise(resolve => { releaseHydrate = () => resolve(store.active()); }); };
  const selection = bindingOwner('').authorizeDataSource('begin_local_selection');
  await hydrated;
  session = conversation('session-B', studyB.id);
  const before = activations.length;
  releaseHydrate();
  await selection;
  store.hydrate = quietHydrate;
  assert.equal(store.active().id, studyB.id);
  assert.deepEqual(activations.slice(before), []);

  // The shell's continue message after a source confirmation waits on the
  // question carry. Opening conversation B meanwhile, or B and then A
  // again, sends nothing; undisturbed, it is sent once.
  const shell = fs.readFileSync(shellFile, 'utf8');
  const begin = shell.indexOf('  async function continueAfterDataSourceConfirmation(');
  const continueSource = shell.slice(begin, shell.indexOf('  async function sendMessage(', begin));
  for (const route of [['session-B'], ['session-B', 'session-A'], []]) {
    let releaseCarry;
    const sent = [];
    const shellState = { session: { session_id: 'session-A' }, sessionSelectionRevision: 1, busy: false, childJobId: '' };
    const context = vm.createContext({
      state: shellState, projectId: () => 'project_current', sessionIsStale: () => false, tr: en => en,
      DATA_BINDING: { carryQuestionIntoSetup: () => new Promise(resolve => { releaseCarry = resolve; }) },
      sendText: async (...args) => { sent.push(args); },
    });
    vm.runInContext(continueSource, context);
    const continued = context.continueAfterDataSourceConfirmation();
    for (const id of route) {
      shellState.session = { session_id: id };
      shellState.sessionSelectionRevision += 1;
    }
    releaseCarry(false);
    assert.equal(await continued, route.length === 0, route.join(' -> '));
    assert.equal(sent.length, route.length === 0 ? 1 : 0, route.join(' -> '));
    if (!route.length) assert.equal(sent[0][2], 'advance_after_data_source_confirmation');
  }
  process.stdout.write('A study write that waits stops when its conversation is no longer open.\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
