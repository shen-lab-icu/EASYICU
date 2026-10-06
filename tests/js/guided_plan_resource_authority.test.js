'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function confirmation(review) {
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/static/js/screens-guided-pi-modules.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/static/js/screens-guided-pi-confirmation.js'), 'utf8'), context);
  return context.window.EasyICU.guidedPi.require('confirmation').create({
    workflow: () => ({ next_action_code: 'plan_scientific_changes_required', plan_review_summary: review }),
    session: () => ({ binding: { run_id: 'newer-failed-child' }, archived_child_jobs: [] }),
    tr: en => en, esc: String, iconHtml: () => '', resourceButton: () => '',
    sessionIsStale: () => false,
  }).workflowConfirmation();
}

function confirmationOwner(workflow) {
  const context = { window: {} };
  vm.createContext(context);
  for (const name of ['modules', 'confirmation']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('confirmation').create({
    workflow: () => workflow, session: () => ({}), busy: () => false,
    tr: en => en, esc: value => String(value).replaceAll('<', '&lt;'), iconHtml: () => '',
    resourceButton: ref => `[resource:${ref.run_id}:${ref.artifact}]`,
    sessionIsStale: () => false,
  });
}

function confirmationWithFailureText(workflow, session) {
  const context = { window: {} };
  vm.createContext(context);
  for (const name of ['modules', 'confirmation']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('confirmation').create({
    workflow: () => workflow, session: () => session, busy: () => false,
    tr: en => en, esc: String, iconHtml: () => '', resourceButton: () => '',
    sessionIsStale: () => false,
    // Echo which cause the card asked for: the code and its typed detail.
    runFailureText: (code, detail) => [code, detail && detail.code].filter(Boolean).join(' / '),
  }).workflowConfirmation();
}

test('plan evidence keeps the reviewed run when a later child fails before planning', () => {
  const card = confirmation({ run_id: 'reviewed-plan', remediation_buckets: {} });
  assert.equal(card.reviewResources.length, 3);
  for (const resource of card.reviewResources) assert.equal(resource.run_id, 'reviewed-plan');
});

test('missing plan authority never falls back to the latest session run', () => {
  const card = confirmation({ remediation_buckets: {} });
  for (const resource of card.reviewResources) assert.equal(resource.run_id, '');
});

test('a preserved candidate and its failed preparation have separate evidence links', () => {
  const workflow = {
    next_action_code: 'planner_checkpoint_resume_available',
    plan_review_summary: { run_id: 'reviewed-plan' },
    latest_attempt_failure: {
      run_id: 'failed-preparation', candidate_run_id: 'reviewed-plan',
      reason: 'provider_unavailable', checkpoint_resume_available: true,
      raw_error: '<script>private response</script>',
    },
  };
  const owner = confirmationOwner(workflow);
  const html = owner.workflowConfirmationHtml();
  assert.match(html, /gpi-confirmation-failure/);
  assert.match(html, /provider request failed/i);
  assert.match(html, /failed-preparation:source_run_manifest.json/);
  assert.match(html, /reviewed-plan:agent_plan.json/);
  assert.doesNotMatch(html, /private response|failed-preparation:agent_plan.json/);
  assert.equal(owner.workflowConfirmation().grants.includes('extract'), false);
});

test('a plan that can no longer resume is not blamed on a changed study', () => {
  const card = code => confirmationOwner({
    next_action_code: code, plan_review_summary: { run_id: 'old-plan' },
  }).workflowConfirmation();
  const lost = card('plan_review_not_resumable');
  assert.match(lost.title, /can no longer be approved or resumed/);
  assert.doesNotMatch(lost.title, /study changed/);
  assert.match(card('plan_configuration_superseded').title, /The study changed/);
});

test('the fresh-plan card names the cause its failed run recorded', () => {
  // The workflow's action comes from the authoritative failed run; its typed
  // detail names a cause that the job's bare code would generalize.
  const failedJob = { kind: 'agent-run', status: 'failed', error_code: 'research_pipeline_progressive_compile_failed' };
  const card = workflow => confirmationWithFailureText(workflow, { archived_child_jobs: [failedJob] });
  const recorded = card({
    next_action_code: 'failed_pipeline_requires_fresh_plan',
    runs: [
      { run_id: 'other', authoritative: false, run_status: 'failed',
        gate_reason_code: 'research_pipeline_execution_failed' },
      { run_id: 'failed-plan', authoritative: true, run_status: 'failed', gate_status: 'blocked',
        gate_reason_code: 'research_pipeline_progressive_compile_failed',
        gate_detail_code: 'progressive_family_result_contract_unwritable' },
    ],
  });
  assert.equal(recorded.reason, 'research_pipeline_progressive_compile_failed / progressive_family_result_contract_unwritable');
  // A run awaiting review is not a failure; the job's code is all there is.
  const pending = card({
    next_action_code: 'failed_pipeline_requires_fresh_plan',
    runs: [{ run_id: 'paused', authoritative: true, run_status: 'human_review_pending', gate_reason_code: 'human_plan_review_required' }],
  });
  assert.equal(pending.reason, 'research_pipeline_progressive_compile_failed');
});

test('failure notice cannot drift onto another reviewed candidate', () => {
  const owner = confirmationOwner({
    next_action_code: 'plan_execution_upgrade_required',
    plan_review_summary: { run_id: 'new-plan' },
    latest_attempt_failure: { run_id: 'failed', candidate_run_id: 'old-plan', reason: 'provider_unavailable' },
  });
  assert.doesNotMatch(owner.workflowConfirmationHtml(), /gpi-confirmation-failure/);
});
