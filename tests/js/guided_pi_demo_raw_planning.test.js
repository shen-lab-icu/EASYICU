'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function screen() {
  const declared = {};
  const state = {
    project: 'project-1', busy: false, error: '', calls: [], renders: 0,
    session: { session_id: 'pi-1', binding: { study_context_id: 'study-1', study_revision: 4 } },
    workflow: { next_action_code: 'bound_export_cohort_mismatch', bound_export_cohort_refusal: {
      planning_source: { action: 'use_official_demo_raw_for_plan', study_id: 'study-1',
        expected_revision: 4, offer_sha256: 'a'.repeat(64), database: 'eicu_demo' },
    } },
  };
  const context = vm.createContext({
    window: { EasyICU: { guidedPi: { declare: (name, owner) => { declared[name] = owner; }, optional: () => null } } },
  });
  for (const name of ['confirmation', 'plan-actions']) {
    vm.runInContext(fs.readFileSync(path.join(JS, `screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  const api = { authorizePiCopilotDataSource: async (id, body) => { state.calls.push({ id, body }); } };
  const host = {
    tr: (en, zh) => zh || en, esc: String, iconHtml: () => '', resourceButton: () => '',
    session: () => state.session, workflow: () => state.workflow, sessionIsStale: () => false,
    projectId: () => state.project, busy: () => state.busy, researchSourceReady: () => true,
    api: () => api, setBusy: value => { state.busy = value; }, setError: value => { state.error = value; },
    render: () => { state.renders++; }, errorText: String,
    refreshSession: async () => { state.calls.push('refresh'); },
    loadWorkflow: async () => { state.calls.push('workflow'); },
    startAgentRun: () => assert.fail('source selection cannot start a Provider run'),
    sendText: () => assert.fail('source selection cannot send a turn or grant execution'),
  };
  return { state, api, confirmation: declared.confirmation.create(host), actions: declared.planActions.create(host) };
}

test('only an available same-source offer has a choice button and it grants no execution', () => {
  const ui = screen();
  const offer = ui.confirmation.workflowConfirmation();
  assert.equal(offer.nonApprovable, false);
  assert.equal(JSON.stringify(offer.grants), '[]');
  assert.match(ui.confirmation.workflowConfirmationHtml(), /data-gpi-confirm-action/);
  delete ui.state.workflow.bound_export_cohort_refusal.planning_source;
  assert.equal(ui.confirmation.workflowConfirmation().nonApprovable, true);
  assert.doesNotMatch(ui.confirmation.workflowConfirmationHtml(), /data-gpi-confirm-action/);
});

test('one click consumes the displayed version without paths, grants or starting analysis', async () => {
  const ui = screen();
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  assert.deepEqual(JSON.parse(JSON.stringify(ui.state.calls)), [
    { id: 'pi-1', body: { project_id: 'project-1', action: 'use_official_demo_raw_for_plan',
      expected_revision: 4, offer_sha256: 'a'.repeat(64) } }, 'refresh', 'workflow',
  ]);
  assert.equal(ui.state.busy, false);
  assert.equal(ui.state.error, '');
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  assert.equal(ui.state.calls.length, 3, 'the same offer cannot be consumed twice');
});

test('an offer for an old study revision never submits', async () => {
  const ui = screen();
  ui.state.session.binding.study_revision++;
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  assert.equal(ui.state.calls.length, 0);
});

test('a slow choice does not refresh a different conversation after navigation', async () => {
  const ui = screen();
  let finish;
  ui.api.authorizePiCopilotDataSource = () => new Promise(resolve => { finish = resolve; });
  const pending = ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  ui.state.session = { session_id: 'pi-2', binding: { study_context_id: 'study-2', study_revision: 1 } };
  const rendersBefore = ui.state.renders;
  finish({ ok: true });
  await pending;
  assert.equal(ui.state.calls.length, 0);
  assert.equal(ui.state.renders, rendersBefore);
});
