'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh' };
for (const file of process.argv.slice(2)) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
let language = 'zh';
const tr = (en, zh) => language === 'zh' ? zh : en;
const action = (kind, route, params = {}) => ({ kind, action_id: `${kind}_0123456789abcdef`,
  label_key: `stop_next_actions.${kind}`, params: { route, ...params } });
const edit = action('edit_plan', 'plan_edit', { intent: 'candidate_plan', planner_start_mode: 'fresh', requires_explicit_choice: true, requires_plan_approval: true });
const run = { run_ref: 'r1', run_id: 'synthetic', run_status: 'failed',
  artifact_names: ['source_run_manifest.json'], next_actions: [action('retry', 'retry'), edit] };
let workflow = { next_action_code: 'failed_pipeline_execution_retry_available', runs: [{ ...run, authoritative: true }] };
const owner = modules.require('runStop').create({ tr, runFailureText: () => tr('Stopped.', '已停止。'),
  workflow: () => workflow, latestRun: () => workflow.runs.find(row => row.authoritative),
  resourceButton: (resource, label, options) => `<button class="${(options && options.className) || 'gpi-resource-link'}" data-resource="${resource.artifact}">${label}</button>`,
  canAct: () => true, canPreview: () => true, hasRecovery: () => true });
let statedDetail;
modules.require('runStop').create({ tr, runFailureText: (_code, detail) => {
  statedDetail = detail;
  return detail.statedVariable;
} }).render({ savedRuns: [{ ...run, gate_reason_code: 'research_pipeline_progressive_compile_failed',
  gate_detail: { reason_code: 'progressive_family_spec_stated_adjustment_unavailable',
    cause_code: 'stated_adjustment_name_unresolved', stated_variable: '试验量表甲' } }] });
assert.equal(statedDetail.statedVariable, '试验量表甲');
assert.equal(statedDetail.cause, 'stated_adjustment_name_unresolved');
const events = {};
const calls = [];
const recovery = { scrollIntoView() { calls.push('reveal'); }, setAttribute() {}, focus() {} };
const state = { session: {}, busy: false, workflow, latestRun: run,
  host: { addEventListener: (name, callback) => { events[name] = callback; }, querySelector: selector => selector === '.gpi-confirmation' ? recovery : null } };
modules.require('events').create({ state, tr,
  RUN_FILES: { handleClick: () => false },
  RESOURCE_OWNER: { fromButton: () => null },
  retryFailedExecution: () => calls.push('retry'), confirmWorkflowAction: () => calls.push('confirm'),
  editWorkflow: () => calls.push('edit'),
  COHORT_ELIGIBILITY: { actionFromEvent: () => null },
}).wire();
function click(kind) {
  const button = { dataset: { gpiStopAction: `${kind}_0123456789abcdef`, gpiStopRun: 'synthetic' } };
  events.click({ target: { closest: selector => selector === '[data-gpi-stop-action]' ? button : null }, preventDefault() {} });
}
for (language of ['zh', 'en']) {
  window.EU_LANG = language;
  const html = owner.render({ savedRuns: [run] });
  assert.ok(html.includes('data-gpi-stop-action="retry_0123456789abcdef"'));
  assert.ok(html.includes(tr('Retry from the failed step', '从失败步骤重试')));
  assert.ok(html.includes(tr('Regenerate the plan', '重新生成计划')));
  assert.ok(html.includes('source_run_manifest.json'));
  click('retry');
  click('edit_plan');
  assert.deepEqual(calls.splice(0), ['retry', 'edit']);
  workflow.next_action_code = 'failed_pipeline_requires_fresh_plan';
  assert.ok(!owner.render({ savedRuns: [run] }).includes('data-gpi-stop-action="retry_'));
  click('retry');
  assert.deepEqual(calls, [], 'a stale retry click cannot start a run');
  click('edit_plan');
  assert.deepEqual(calls.splice(0), ['confirm']);
  workflow.next_action_code = 'failed_pipeline_execution_retry_available';
  state.busy = true;
  click('retry');
  assert.deepEqual(calls, []);
  state.busy = false;
  const historical = { ...run, run_id: 'old', run_ref: 'r0' };
  const old = owner.render({ savedRuns: [historical] });
  assert.ok(!old.includes('data-gpi-stop-action'));
  assert.ok(old.includes('source_run_manifest.json'), 'historical stops retain their own receipt');
  assert.ok(old.includes('class="btn sm" data-resource="source_run_manifest.json"'), 'the receipt is a button like the other choices');
  assert.equal(modules.require('runStop').currentRun(run,
    { runs: [{ run_id: 'new_authoritative_run', authoritative: true }] }, run), false,
  'the workflow authority wins over a stale local latest-run pointer');
  const data = { ...run, next_actions: [action('reexport_data', 'data_extraction', { requires_explicit_choice: true })] };
  assert.ok(owner.render({ savedRuns: [data] }).includes('href="#extraction"'));
  // No stop asks how the model handles data: an old saved settings choice
  // offers no button of its own and enters the recovery card.
  const declare = owner.render({ savedRuns: [{ ...run, next_actions: [action('open_provider_settings', 'provider_settings')] }] });
  assert.ok(!declare.includes('data-gpi-stop-declare'));
  assert.ok(!declare.includes('数据处理方式') && !declare.includes('handles data'));
  assert.ok(declare.includes('data-gpi-stop-recovery'));
  const forged = { ...run, next_actions: [{ ...edit, params: { ...edit.params, route: 'retry' } }] };
  assert.ok(!owner.render({ savedRuns: [forged] }).includes('data-gpi-stop-action'));
  const unknown = { ...run, next_actions: [action('imagined_action', 'retry')] };
  assert.ok(!owner.render({ savedRuns: [unknown] }).includes('imagined_action'));
  assert.ok(owner.render({ savedRuns: [unknown] }).includes('data-gpi-stop-recovery'), 'unsupported choices enter the existing recovery card');
  events.click({ target: { closest: selector => selector === '[data-gpi-stop-recovery]' ? {} : null } });
  assert.deepEqual(calls.splice(0), ['reveal']);
  workflow.next_actions = [];
  assert.ok(!owner.render({ savedRuns: [run] }).includes('data-gpi-stop-action'), 'the current workflow withdraws historical offers');
  delete workflow.next_actions;
  const noOwners = modules.require('runStop').create({ tr, workflow: () => workflow, latestRun: () => run });
  assert.ok(!noOwners.render({ savedRuns: [run] }).includes('<button'));
}
console.log('stop actions: bilingual rendering, owner dispatch, stale/busy/historical guards passed');
