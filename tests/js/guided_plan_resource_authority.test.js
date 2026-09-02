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

test('plan evidence keeps the reviewed run when a later child fails before planning', () => {
  const card = confirmation({ run_id: 'reviewed-plan', remediation_buckets: {} });
  assert.equal(card.reviewResources.length, 3);
  for (const resource of card.reviewResources) assert.equal(resource.run_id, 'reviewed-plan');
});

test('missing plan authority never falls back to the latest session run', () => {
  const card = confirmation({ remediation_buckets: {} });
  for (const resource of card.reviewResources) assert.equal(resource.run_id, '');
});
