'use strict';

// S1542 backstop (B13 was the second retry-path loop): a plan whose
// execution settings were applied twice with no new plan in between stops
// with its own card, not a third "Apply execution settings". Its ways
// forward are a fresh plan and a change request.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const CODE = 'agent_plan_configuration_loop_stopped';

function card(language) {
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
    workflow: () => ({ next_action_code: CODE, plan_review_summary: { run_id: 'run-1' } }),
    planConfigurationError: () => '', sessionIsStale: () => false, busy: () => false,
  };
  const host = new Proxy(known, { get: (item, key) => (key in item ? item[key] : () => undefined) });
  return { owner: declared.confirmation.create(host), runFailureText: known.runFailureText };
}

for (const language of ['zh', 'en']) {
  test(`${language}: the stopped compile loop names itself and offers a fresh plan and a change request`, () => {
    const { owner, runFailureText } = card(language);
    const confirmation = owner.workflowConfirmation();
    assert.equal(confirmation.code, CODE);
    assert.ok(!confirmation.nonApprovable);
    const html = owner.workflowConfirmationHtml();
    assert.match(html, language === 'zh' ? /已应用过两次/ : /already applied twice/);
    assert.ok(!html.includes(language === 'zh' ? '应用执行设置' : 'Apply execution settings'), 'no third apply');
    assert.ok(html.includes(language === 'zh' ? '重新生成计划' : 'Generate fresh plan'));
    assert.ok(html.includes(language === 'zh' ? '提出修改' : 'Request changes'));
    // The service refusal of a third compile reads the same way.
    assert.notEqual(runFailureText(CODE), runFailureText('easyicu_unlisted_stop'));
    assert.match(runFailureText(CODE), language === 'zh' ? /已应用过两次/ : /already applied twice/);
  });
}
