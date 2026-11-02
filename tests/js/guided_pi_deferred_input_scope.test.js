'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
global.window = { EU_LANG: 'zh' };
for (const name of ['screens-guided-pi-modules', 'html-escape', 'screens-guided-pi-plan-actions', 'screens-guided-pi-confirmation-data', 'screens-guided-pi-confirmation']) {
  require(path.join(__dirname, '../../src/easyicu/webserver/static/js', name + '.js'));
}

function render(scope, language = 'zh') {
  const workflow = { next_action_code: 'operator_plan_approval_required',
    plan_review_summary: { deferred_input_scope: scope } };
  const before = JSON.stringify(workflow);
  const host = { tr: (en, zh) => language === 'zh' ? zh : en, esc: window.EU_HTML.esc,
    iconHtml: () => '', workflow: () => workflow, session: () => ({ session_id: 'session', binding: { study_context_id: 'study' } }),
    busy: () => false, sessionIsStale: () => false, researchSourceReady: () => true,
    projectId: () => 'project', resourceButton: () => '', api: () => { throw Error('read-only card must not call API'); } };
  const html = window.EasyICU.guidedPi.require('confirmation').create(host).workflowConfirmationHtml();
  assert.equal(JSON.stringify(workflow), before);
  return html;
}

const scope = {
  schema_version: 'easyicu.first-observed-review-scope/1', verified: true,
  recipe_sha256: 'a'.repeat(64), plan_sha256: 'b'.repeat(64), source_manifest_sha256: 'c'.repeat(64),
  pending_materialization: true, actual_cohort_exists: false, analysis_authorized: false,
  question: 'By sex, report mortality and a 90% interval.', source_label: 'eICU <Demo>',
  population_summary: 'Adult first observable ICU; lifetime first is not established.',
  interval_summary: 'Wilson interval, confidence level 90%',
  snapshot_files: [{ file: 'patient.parquet', sha256: null, hash_status: 'computed_after_authorization' }],
  decoded_files: [{ file: 'patient.parquet', columns: ['patientunitstayid', 'hospitaldischargestatus', '<unsafe>'] }],
  budget: { max_calls: 32, max_tokens: 1200000, max_cost_usd: 30, max_wall_time_seconds: 1800,
    used_calls: 3, used_tokens: 18294, used_cost_usd: 0.21088 },
};

test('one complete plan card shows proposed outcome reads and cumulative budget without creating a second approval', () => {
  for (const language of ['zh', 'en']) {
    const html = render(scope, language);
    for (const text of ['patient.parquet', 'hospitaldischargestatus', '90%', '3/32', '18294/1200000', '$0.21088/$30', scope.recipe_sha256]) assert.ok(html.includes(text));
    assert.match(html, /eICU &lt;Demo&gt;/);
    assert.match(html, /&lt;unsafe&gt;/);
    assert.doesNotMatch(html, /95%|data-gpi-preparation-review|<unsafe>/);
    assert.equal((html.match(/data-gpi-confirm-action/g) || []).length, 1);
    assert.doesNotMatch(html, /data-gpi-confirm-action disabled/);
  }
});

test('missing proof or malformed read scope disables the existing approval action', () => {
  for (const invalid of [
    { ...scope, verified: false }, { ...scope, recipe_sha256: 'wrong' },
    { ...scope, analysis_authorized: true }, { ...scope, decoded_files: [] },
    { ...scope, snapshot_files: [{ file: 'patient.parquet', sha256: null }] },
  ]) {
    const html = render(invalid);
    assert.match(html, /无法核验本计划的数据读取范围/);
    assert.match(html, /data-gpi-confirm-action disabled/);
    assert.doesNotMatch(html, /data-gpi-deferred-input-scope/);
  }
});

test('ordinary plans keep their existing card and action', () => {
  const html = render(undefined);
  assert.doesNotMatch(html, /data-gpi-deferred-input-scope|data-gpi-confirm-action disabled/);
  assert.equal((html.match(/data-gpi-confirm-action/g) || []).length, 1);
});
