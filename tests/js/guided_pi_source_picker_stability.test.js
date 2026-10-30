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
    TRANSCRIPT: { markSupersededPlanAttempts() {} },
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

test('extraction offers no synthetic data-mode control, embedded or not', () => {
  // Every source takes the one source-backed path; there is no Demo/Real
  // switch for the native page to render or the source-binding panel to strip.
  for (const name of ['screens-extraction.js', 'screens-extraction-embedded.js']) {
    const source = fs.readFileSync(path.join(JS, name), 'utf8');
    assert.doesNotMatch(source, /data-datamode/, `${name} must not offer a data-mode control`);
    assert.doesNotMatch(source, /data-ex-sample/, `${name} must not offer the synthetic sample study`);
  }
});
