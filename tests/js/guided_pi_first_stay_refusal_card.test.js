'use strict';

// UX leg 3 en (2026-11-19, B9): the host refused the automatic configuration
// because the source could not order a patient's stays. The card showed the
// generic "the run stopped" text and promised a choice below that it never
// rendered. It now names the reason and the ways forward that work (a plan
// change keeps the stated first-stay decision, so it is not offered as one).

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const CODE = 'agent_plan_first_stay_coordinate_unavailable';
const COPY = {
  zh: { reason: '当前数据源无法确认同一患者哪次入住在先', note: '新建一个保留全部 ICU 入住的项目', promised: '下方的选择会更新研究配置' },
  en: { reason: 'cannot show which of a patient\'s stays came first', note: 'start a new project that keeps every ICU stay',
    promised: 'Your choice below updates the study' },
};

function card(language, refusal) {
  const declared = {};
  const context = vm.createContext({
    window: { EasyICU: { guidedPi: { declare: (name, owner) => { declared[name] = owner; }, require: name => declared[name], optional: () => null } } },
  });
  for (const file of ['html-escape.js', ...['error-text', 'confirmation-data', 'confirmation'].map(name => `screens-guided-pi-${name}.js`)]) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context);
  }
  const tr = (en, zh) => (language === 'zh' ? zh : en);
  const known = {
    tr, esc: String, iconHtml: () => '', resourceButton: () => '',
    runFailureText: declared.errorText.create({ tr }).runFailureText,
    session: () => ({ session_id: 'pi-1', binding: { study_context_id: 'study-1', study_revision: 3, run_id: 'run-1' } }),
    workflow: () => ({ next_action_code: 'agent_plan_configuration_required' }),
    planConfigurationError: () => refusal, sessionIsStale: () => false, busy: () => false,
  };
  const host = new Proxy(known, { get: (item, key) => (key in item ? item[key] : () => undefined) });
  return declared.confirmation.create(host);
}

for (const language of ['zh', 'en']) {
  test(`${language}: a first-stay refusal names its reason and promises no missing choice`, () => {
    const owner = card(language, CODE);
    const html = owner.workflowConfirmationHtml();
    assert.ok(html.includes(COPY[language].reason), 'typed reason');
    assert.ok(html.includes(COPY[language].note));
    assert.ok(!html.includes(COPY[language].promised));
    assert.equal(owner.workflowConfirmation().nonApprovable, true);
  });

  test(`${language}: another refusal keeps its own reason and the cohort choice note`, () => {
    const html = card(language, 'agent_plan_patient_grouping_unavailable').workflowConfirmationHtml();
    assert.ok(html.includes(COPY[language].promised));
    assert.ok(!html.includes(COPY[language].reason));
  });
}
