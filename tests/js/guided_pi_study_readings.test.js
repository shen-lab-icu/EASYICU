/* The real renderers display host readings without granting approval. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const files = process.argv.slice(2);
assert.equal(files.length, 8, 'Run through tools/run_js_contracts.py with all eight owners');
const modules = {};
const sandbox = {
  window: { EU_LANG: 'zh', EasyICU: { guidedPi: {
    declare: (name, api) => { modules[name] = api; },
    optional: name => modules[name] || null,
    require: name => modules[name],
  } } },
};
vm.createContext(sandbox);
for (const file of files) {
  vm.runInContext(fs.readFileSync(file, 'utf8'), sandbox, { filename: file });
}
const esc = sandbox.window.EU_HTML.esc;
const helpers = { esc, tr: (en, zh) => sandbox.window.EU_LANG === 'zh' ? zh || en : en };
const reading = { role: 'exposure', concept: 'lact', column: 'lact_max', category: 'blood gas',
  basis: 'keyword', label_en: 'Lactate', label_zh: '乳酸', evidence: 'lactate' };
const family = { family: 'association_study', source: 'question_wording',
  elements: [{ element: 'relationship', evidence: '是否与死亡相关' }],
  conflict: { design_family: 'descriptive_epidemiology' } };
const trial = { source: 'question_trial_shape', elements: [
  { element: 'time_zero', evidence: 'hour 6' },
  { element: 'initiate_strategy', evidence: 'start norepinephrine within 24 h' },
  { element: 'defer_strategy', evidence: 'not start' },
  { element: 'fixed_horizon_outcome', evidence: '28-day mortality' },
] };

for (const language of ['zh', 'en']) {
  sandbox.window.EU_LANG = language;
  const coordinate = modules.questionRequirements.notesHtml(
    { coordinate_readings: { status: 'shown', items: [reading] } }, helpers,
  );
  assert.ok(coordinate.includes(language === 'zh' ? '乳酸' : 'Lactate'));
  assert.ok(coordinate.includes(language === 'zh' ? '血气检验' : 'blood gas'));
  assert.ok(coordinate.includes(language === 'zh' ? '未经你单独确认' : 'not a separate choice you confirmed'));
  assert.equal(coordinate.includes('<button'), false);
  const unavailable = modules.questionRequirements.notesHtml(
    { coordinate_readings: { status: 'unavailable', items: [] } }, helpers,
  );
  assert.ok(unavailable.includes(language === 'zh' ? '无法核验' : 'could not be verified'));
  assert.equal(modules.questionRequirements.notesHtml({}, helpers), '');

  const workflow = { study_setup_receipt: { configuration: {
    study_family_reading: family,
    design_variance_basis: { variance_estimator: 'none_counts_only', conflict: 'counts_only_not_requested' },
    causal_trial_reading: trial,
  } }, next_action_code: 'target_trial_statement_needed' };
  const before = JSON.stringify(workflow);
  const familyHtml = modules.studyReadings.notesHtml(workflow, helpers);
  assert.ok(familyHtml.includes(language === 'zh' ? '关联研究' : 'association research'));
  assert.ok(familyHtml.includes(language === 'zh' ? '描述性研究' : 'descriptive research'));
  assert.ok(familyHtml.includes(language === 'zh' ? '没有记录到' : 'no researcher request'));
  const trialHtml = modules.targetTrial.bodyHtml(workflow, helpers);
  for (const row of trial.elements) assert.ok(trialHtml.includes(row.evidence));
  const decision = modules.targetTrial.confirmation(workflow, helpers.tr);
  assert.equal(decision.nonApprovable, true);
  assert.equal(modules.targetTrial.approvalRequest(workflow, { binding: {} }, 'p'), null);
  assert.equal(JSON.stringify(workflow), before);

  const confirmation = modules.confirmation.create({ ...helpers,
    iconHtml: () => '', resourceButton: () => '',
    sessionIsStale: () => false, busy: () => false,
    session: () => ({ binding: { run_id: 'r' } }), workflow: () => workflow,
  }).workflowConfirmationHtml();
  assert.ok(confirmation.includes(familyHtml));
  assert.ok(confirmation.includes('28-day mortality'));

  const errors = modules.errorText.create({ tr: helpers.tr, staticPreview: () => false });
  assert.ok(errors.errorText({ code: 'study_causal_trial_design_unset' })
    .includes(language === 'zh' ? '目标试验' : 'target-trial'));
  assert.ok(errors.errorText({ code: 'target_trial_database_out_of_scope' }).includes('MIMIC-IV'));
  assert.ok(errors.errorText({ code: 'target_trial_treatment_not_registered', details: {
    causal_trial_gap: { treatment_concepts: [{ label_en: 'Albumin', label_zh: '白蛋白' }],
      supported_classes: [{ label_en: 'Vasopressors', label_zh: '升压药' }] },
  } }).includes(language === 'zh' ? '白蛋白' : 'Albumin'));
}

const hostile = '<img src=x onerror=alert(1)>';
sandbox.window.EU_LANG = 'zh';
const injection = modules.questionRequirements.notesHtml({ coordinate_readings: {
  status: 'shown', items: [{ ...reading, label_zh: hostile, evidence: hostile }],
} }, helpers);
assert.equal(injection.includes(hostile), false);
assert.ok(injection.includes('&lt;img'));
const hostileTrial = modules.targetTrial.bodyHtml({ study_setup_receipt: { configuration: {
  causal_trial_reading: { ...trial, elements: [{ element: 'time_zero', evidence: hostile }] },
} } }, helpers);
assert.equal(hostileTrial.includes(hostile), false);
assert.ok(hostileTrial.includes('&lt;img'));
const hostileFamily = modules.studyReadings.notesHtml({ study_setup_receipt: { configuration: {
  study_family_reading: { ...family, elements: [{ element: 'relationship', evidence: hostile }] },
} } }, helpers);
assert.equal(hostileFamily.includes(hostile), false);
assert.ok(hostileFamily.includes('&lt;img'));
const denied = modules.questionRequirements.notesHtml({ coordinate_readings: {
  status: 'shown', items: [{ ...reading, basis: 'configured' }],
} }, helpers);
assert.equal(denied, '');
console.log('Host reading display, authority and escaping contracts passed');
