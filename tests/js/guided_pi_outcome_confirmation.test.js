'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function owner(workflow, language) {
  const context = { window: {} };
  vm.createContext(context);
  for (const name of ['modules', 'confirmation-data', 'confirmation']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('confirmation').create({
    workflow: () => workflow, session: () => ({}), busy: () => false,
    tr: (en, zh) => language === 'zh' ? zh : en, esc: String, iconHtml: () => '', resourceButton: () => '',
    sessionIsStale: () => false,
  });
}

test('the plan step offers the proposed outcome as the one choice to confirm', () => {
  const workflow = { next_action_code: 'study_primary_outcome_confirmation_required', unconfirmed_outcome: '院内死亡' };
  const zh = owner(workflow, 'zh').workflowConfirmation();
  // The message designates the exact label, which the study update accepts.
  assert.equal(zh.message, '主要结局：院内死亡');
  // The click saves the outcome, so it carries the one-time configure grant.
  assert.deepEqual([...zh.grants], ['configure']);
  assert.equal(zh.nonApprovable, false);
  assert.match(zh.title, /院内死亡/);
  assert.equal(zh.editLabel, '改用其他结局');
  assert.equal(owner(workflow, 'en').workflowConfirmation().message, 'Primary outcome: 院内死亡');

  // Without a label there is nothing to confirm by one click.
  const unlabeled = owner({ next_action_code: 'study_primary_outcome_confirmation_required' }, 'zh').workflowConfirmation();
  assert.equal(unlabeled.nonApprovable, true);
  assert.equal(unlabeled.message, '');
  assert.deepEqual([...unlabeled.grants], []);
});

test('a per-hospitalization first stay says why when the source cannot order a patient', () => {
  const preparation = {
    available: true, status: 'available', criteria: [{ id: 'c2', quote: '首次入 ICU', disposition: 'requires_extraction' }],
    changes: [{ field: 'first_icu_stay_per_hospitalization', before: false, after: true }],
  };
  const workflow = substitutes => ({
    next_action_code: 'population_inclusion_requires_extraction',
    plan_review_summary: { population_preparation: preparation },
    ...(substitutes ? { patient_first_stay_substitutes: substitutes } : {}),
  });
  const eicu = owner(workflow(['first_icu_stay_per_hospitalization', 'all_icu_stays']), 'zh').workflowConfirmationHtml();
  assert.match(eicu, /仅保留每次住院的首次 ICU 入住：否 → 是/);
  assert.doesNotMatch(eicu, /first_icu_stay_per_hospitalization/);
  assert.match(eicu, /无法确定同一患者跨不同住院的首次 ICU 入住/);
  assert.doesNotMatch(owner(workflow(null), 'zh').workflowConfirmationHtml(), /无法确定同一患者/);
});
