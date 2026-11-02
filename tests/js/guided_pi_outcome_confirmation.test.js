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
  assert.deepEqual([...zh.grants], []);
  assert.equal(zh.nonApprovable, false);
  assert.match(zh.title, /院内死亡/);
  assert.equal(zh.editLabel, '改用其他结局');
  assert.equal(owner(workflow, 'en').workflowConfirmation().message, 'Primary outcome: 院内死亡');

  // Without a label there is nothing to confirm by one click.
  const unlabeled = owner({ next_action_code: 'study_primary_outcome_confirmation_required' }, 'zh').workflowConfirmation();
  assert.equal(unlabeled.nonApprovable, true);
  assert.equal(unlabeled.message, '');
});
