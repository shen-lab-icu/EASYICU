'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function sessionRefresh(currentId) {
  const source = fs.readFileSync(path.join(JS, 'screens-guided-pi.js'), 'utf8');
  const begin = source.indexOf('  async function openSession(');
  const end = source.indexOf('  function assistantRow(', begin);
  const panel = { open: true, closes: 0 };
  const state = { session: { session_id: currentId }, sessions: [], workflow: {}, sessionSelectionRevision: 0 };
  const context = vm.createContext({
    state, URLSearchParams, window: { location: { search: '' } }, document: {},
    projectId: () => 'project-1', closeSourceView() {}, closeChildSource() {},
    MODULES: {
      optional: () => ({ close() { panel.open = false; panel.closes += 1; } }),
      require: () => null,
    },
    api: () => ({ loadPiCopilotSession: async id => ({ session: { session_id: id } }) }),
    sessionMatchesUiLanguage: () => true, transcriptMessages: () => [],
    hydrateProjectedJob() {}, reconcileSettledSession() {}, HOST_JOBS: { sync() {} },
    rememberSession() {}, setShell() {}, render() {}, loadWorkflow: async () => {},
    PLAN_ACTIONS: { continueSystemOwnedPlanProgression: async () => {} }, errorText: String,
  });
  vm.runInContext(source.slice(begin, end), context);
  return { panel, state, refresh: id => context.openSession(id) };
}

test('restoring the same conversation after the picker enters real-data mode keeps its panel open', async () => {
  const ui = sessionRefresh('session-1');
  await ui.refresh('session-1');
  assert.equal(ui.state.error, '');
  assert.equal(ui.state.session.session_id, 'session-1');
  assert.equal(ui.panel.open, true);
  assert.equal(ui.panel.closes, 0);
});

test('opening another conversation still closes the previous data-source panel', async () => {
  const ui = sessionRefresh('session-1');
  await ui.refresh('session-2');
  assert.equal(ui.state.error, '');
  assert.equal(ui.state.session.session_id, 'session-2');
  assert.equal(ui.panel.open, false);
  assert.equal(ui.panel.closes, 1);
});

test('source binding removes the synthetic extraction-mode control while normal extraction retains it', () => {
  const modes = { removed: 0, remove() { this.removed += 1; } };
  const root = {
    isConnected: true, innerHTML: '', querySelector: () => null,
    querySelectorAll: selector => selector.includes('[data-datamode-switch]') ? [modes] : [],
  };
  const context = vm.createContext({
    icon: () => '', t: en => en,
    window: {
      EU_HTML: { esc: value => String(value ?? '') },
      EU_EXTRACTION_NATIVE_OWNER: {
        render: () => '<div data-datamode-switch></div>', bind() {}, isReal: () => true,
        isPreparedExport: () => false, sourceBindingSnapshot: () => ({ ready: false }),
        beginSourceBinding() {},
      },
    },
  });
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-extraction-embedded.js'), 'utf8'), context);
  const embedded = context.window.EU_EXTRACTION_EMBEDDED_WORKSPACE;
  embedded.mount(root, { resource: { state: 'setup', entry_mode: 'source_binding', study_context_id: 'study-1' } });
  assert.equal(modes.removed, 1, 'a source decision must not offer synthetic extraction mode');
  embedded.mount(root, { resource: { state: 'setup', study_context_id: 'study-1' } });
  assert.equal(modes.removed, 1, 'the native extraction mode control remains available outside source binding');
});
