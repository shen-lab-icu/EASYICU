'use strict';

// UX leg 3 (2026-11-19, B6): a settings change on a raw official-Demo study
// ended its source choice and the automatic revision stopped with no way
// forward. The workflow now offers the same choice again; the card reads as
// an expected step, and confirming it continues the plan.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const CODE = 'demo_planning_source_reconfirm_required';
const COPY = {
  zh: { title: '再确认一次数据源', note: '研究设置已更新，请再确认一次数据源后继续。', approve: '确认数据源并继续' },
  en: { title: 'Confirm the data source again', note: 'The study settings were updated. Confirm the data source once more to continue.',
    approve: 'Confirm the data source and continue' },
};

function screen(language = 'zh', { offer = true } = {}) {
  const declared = {};
  const state = {
    project: 'project-1', busy: false, error: '', calls: [], renders: 0,
    session: { session_id: 'pi-1', binding: { study_context_id: 'study-1', study_revision: 7 } },
    workflow: { next_action_code: CODE, bound_export_cohort_refusal: {
      code: CODE, reason_codes: [],
      ...(offer ? { planning_source: { action: 'use_official_demo_raw_for_plan', study_id: 'study-1',
        expected_revision: 7, offer_sha256: 'b'.repeat(64), database: 'miiv' } } : {}),
    } },
    after: { next_action_code: 'plan_configuration_superseded' },
  };
  const context = vm.createContext({
    window: { EasyICU: { guidedPi: { declare: (name, owner) => { declared[name] = owner; }, require: name => declared[name], optional: () => null } } },
  });
  for (const name of ['confirmation-data', 'confirmation', 'plan-actions']) {
    vm.runInContext(fs.readFileSync(path.join(JS, `screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  const api = {
    authorizePiCopilotDataSource: async (id, body) => { state.calls.push({ id, body }); },
    loadStudyContext: async () => { state.calls.push('study'); return { id: 'study-1', revision: 8, question: 'q', data_source: { path: 'raw', database: 'miiv' } }; },
    startAgentRun: async () => { state.calls.push('plan'); return { job_id: 'job-1' }; },
    applyPiCopilotAgentPlanConfiguration: async () => { state.calls.push('configure'); return { next_action: 'fresh_plan' }; },
  };
  const known = {
    tr: (en, zh) => (language === 'zh' ? zh : en), esc: String, iconHtml: () => '', resourceButton: () => '',
    session: () => state.session, workflow: () => state.workflow, sessionIsStale: () => false,
    projectId: () => state.project, busy: () => state.busy, researchSourceReady: () => true,
    api: () => api, setBusy: value => { state.busy = value; }, setError: value => { state.error = value; },
    render: () => { state.renders++; }, errorText: String,
    refreshSession: async () => { state.calls.push('refresh'); state.session.binding.study_revision = 8; },
    loadWorkflow: async () => { state.calls.push('workflow'); state.workflow = state.after; },
    sendText: () => assert.fail('the card sends no conversation turn'),
  };
  // The plan start's other lifecycle adapters are inert here.
  const host = new Proxy(known, { get: (item, key) => (key in item ? item[key] : () => undefined) });
  return { state, confirmation: declared.confirmation.create(host), actions: declared.planActions.create(host) };
}

for (const language of ['zh', 'en']) {
  test(`${language}: the card explains the expected step and offers one confirmation`, () => {
    const ui = screen(language);
    const html = ui.confirmation.workflowConfirmationHtml();
    for (const text of Object.values(COPY[language])) assert.ok(html.includes(text), text);
    assert.match(html, /data-gpi-confirm-action/);
    const card = ui.confirmation.workflowConfirmation();
    assert.equal(card.nonApprovable, false);
    assert.equal(JSON.stringify(card.grants), '[]');
  });

  test(`${language}: without an offer the card names the other ways forward and confirms nothing`, () => {
    const ui = screen(language, { offer: false });
    const html = ui.confirmation.workflowConfirmationHtml();
    assert.doesNotMatch(html, /data-gpi-confirm-action/);
    assert.match(html, /data-gpi-confirm-edit/);
    assert.ok(!html.includes(COPY[language].note));
  });
}

test('confirming signs the displayed offer, then the plan continues on its own', async () => {
  const ui = screen();
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  const calls = JSON.parse(JSON.stringify(ui.state.calls));
  assert.deepEqual(calls.slice(0, 3), [
    { id: 'pi-1', body: { project_id: 'project-1', action: 'use_official_demo_raw_for_plan',
      expected_revision: 7, offer_sha256: 'b'.repeat(64) } }, 'refresh', 'workflow',
  ]);
  assert.ok(calls.includes('plan'), 'the revision starts without another click');
  assert.equal(calls.filter(call => call === 'plan').length, 1);
});

test('a first source choice still only selects the source', async () => {
  const ui = screen();
  ui.state.workflow.next_action_code = 'bound_export_cohort_mismatch';
  ui.state.after = { next_action_code: 'provider_ready_to_generate_plan' };
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  assert.ok(!ui.state.calls.includes('plan'));
});

test('the confirmation does not continue when the settings changed again meanwhile', async () => {
  const ui = screen();
  ui.state.after = { next_action_code: CODE, bound_export_cohort_refusal: { code: CODE, reason_codes: [] } };
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  assert.ok(!ui.state.calls.includes('plan'));
});

test('a first plan after confirming again continues, keeping its own holds', async () => {
  // Carbon b6 3: the button says "continue" on a study that has no plan yet.
  const ui = screen();
  ui.state.after = { next_action_code: 'provider_ready_to_generate_plan' };
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  assert.ok(ui.state.calls.includes('plan'));

  const held = screen();
  held.state.after = { next_action_code: 'provider_ready_to_generate_plan' };
  held.state.session.idea_exploration_open = true;
  await held.actions.confirmWorkflow(held.confirmation.workflowConfirmation());
  assert.ok(!held.state.calls.includes('plan'), 'mined ideas still wait for a choice');
});

// Carbon b9 2: applying the plan settings (for example narrowing to first
// stays) ends a raw Demo choice; the revision waits for its card instead of
// starting on a raw folder that no longer resolves.
for (const [after, starts] of [[CODE, false], ['plan_configuration_superseded', true]]) {
  test(`after the plan settings apply, the revision ${starts ? 'starts' : 'waits for the data-source card'}`, async () => {
    const ui = screen();
    ui.state.session.binding.run_id = 'run-1';
    ui.state.workflow = { next_action_code: 'agent_plan_configuration_required' };
    ui.state.after = after === CODE
      ? { next_action_code: CODE, bound_export_cohort_refusal: { code: CODE, reason_codes: [] } }
      : { next_action_code: after };
    await ui.actions.continueSystemOwnedPlanProgression();
    assert.ok(ui.state.calls.includes('configure'));
    assert.equal(ui.state.calls.includes('plan'), starts);
  });
}

// B10 (UX leg 3 en/zh, 2026-11-19): the compile, the reconfirmation and a
// fresh plan repeated without end. The host compiles a configuration once;
// the plan made on it is reviewed with its own action, which starts nothing
// without new planner repairs. Counted from the compile: one fresh plan.
for (const review of ['plan_scientific_changes_required', 'plan_execution_upgrade_required']) test(`after one compile and its fresh plan, ${review} starts no further plan`, async () => {
  const ui = screen();
  ui.state.session.binding.run_id = 'run-1';
  ui.state.workflow = { next_action_code: 'agent_plan_configuration_required' };
  ui.state.after = { next_action_code: 'plan_configuration_superseded' };
  await ui.actions.continueSystemOwnedPlanProgression();
  // The fresh plan's review keeps the runtime gaps (the host does not
  // compile its own configuration again) and asks for no planner repair.
  ui.state.session.binding.run_id = 'run-2';
  ui.state.workflow = { next_action_code: review, plan_review_summary: {
    authorization_questions: [], remediation_buckets: { agent_plan_revision: [],
      runtime_capability: ['POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED'] } } };
  for (let poll = 0; poll < 3; poll++) await ui.actions.continueSystemOwnedPlanProgression();
  const calls = JSON.parse(JSON.stringify(ui.state.calls));
  assert.equal(calls.filter(call => call === 'configure').length, 1);
  assert.equal(calls.filter(call => call === 'plan').length, 1);
});

// S1474: the raw-Demo sequence counted end to end. The compile is automatic,
// the reconfirmation is the researcher's click and starts one plan, and that
// plan's review with the same runtime gaps starts nothing more.
for (const review of ['plan_scientific_changes_required', 'plan_execution_upgrade_required']) test(`compile, reconfirm click, one plan, then ${review} starts nothing more`, async () => {
  const ui = screen();
  ui.state.session.binding.run_id = 'run-1';
  ui.state.workflow = { next_action_code: 'agent_plan_configuration_required' };
  ui.state.after = { next_action_code: CODE, bound_export_cohort_refusal: { code: CODE, reason_codes: [],
    planning_source: { action: 'use_official_demo_raw_for_plan', study_id: 'study-1', expected_revision: 8,
      offer_sha256: 'b'.repeat(64), database: 'miiv' } } };
  await ui.actions.continueSystemOwnedPlanProgression();
  const count = name => JSON.parse(JSON.stringify(ui.state.calls)).filter(call => call === name).length;
  assert.equal(count('plan'), 0, 'the plan waits for the data-source card');
  ui.state.session.binding.study_revision = 8;
  ui.state.after = { next_action_code: 'plan_configuration_superseded' };
  await ui.actions.confirmWorkflow(ui.confirmation.workflowConfirmation());
  assert.equal(count('plan'), 1);
  ui.state.session.binding.run_id = 'run-2';
  ui.state.workflow = { next_action_code: review, plan_review_summary: {
    authorization_questions: [], remediation_buckets: { agent_plan_revision: [],
      runtime_capability: ['POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED'] } } };
  for (let poll = 0; poll < 3; poll++) await ui.actions.continueSystemOwnedPlanProgression();
  assert.equal(count('configure'), 1);
  assert.equal(count('plan'), 1);
});

// Carbon b10r2 F1: when the fresh plan's review also asks the planner for
// repairs, the planner revises it once; that revision continues the same
// compiled candidate, so its review is not compiled again either.
test('a planner repair after the compile revises once and is not compiled again', async () => {
  const ui = screen();
  const count = name => JSON.parse(JSON.stringify(ui.state.calls)).filter(call => call === name).length;
  ui.state.session.binding.run_id = 'run-1';
  ui.state.workflow = { next_action_code: 'agent_plan_configuration_required' };
  ui.state.after = { next_action_code: 'plan_configuration_superseded' };
  await ui.actions.continueSystemOwnedPlanProgression();
  ui.state.session.binding.run_id = 'run-2';
  ui.state.workflow = { next_action_code: 'plan_scientific_changes_required', plan_review_summary: {
    run_id: 'run-2', authorization_questions: [], remediation_buckets: {
      agent_plan_revision: ['PLAN_STEP_ORDER_INCOMPLETE'], runtime_capability: ['POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED'] } } };
  for (let poll = 0; poll < 3; poll++) await ui.actions.continueSystemOwnedPlanProgression();
  assert.equal(count('plan'), 2, 'one fresh plan and one planner revision');
  ui.state.session.binding.run_id = 'run-3';
  ui.state.workflow = { next_action_code: 'plan_scientific_changes_required', plan_review_summary: {
    run_id: 'run-3', authorization_questions: [], remediation_buckets: {
      agent_plan_revision: [], runtime_capability: ['POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED'] } } };
  for (let poll = 0; poll < 3; poll++) await ui.actions.continueSystemOwnedPlanProgression();
  assert.equal(count('configure'), 1);
  assert.equal(count('plan'), 2);
});
