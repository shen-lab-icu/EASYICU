'use strict';

// m23 (UX mining walk, 11-19): while mined ideas awaited a choice, the
// confirmed data source offered "查看数据准备确认", whose turn could do
// nothing (allowed_actions []), and the clinician escaped only by typing a
// freeze. The step now offers the freeze itself, sent as a composer message.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const STATIC = path.resolve(__dirname, '../../src/easyicu/webserver/static');
const QUESTION = '初始乳酸与 24 小时峰值乳酸的预后信息比较';

function nextActions() {
  const declared = {};
  const window = { EU_LANG: 'zh', EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; }, optional: name => declared[name] } } };
  const context = vm.createContext({ window });
  for (const file of ['html-escape.js', 'screens-guided-pi-next-actions.js']) {
    vm.runInContext(fs.readFileSync(path.join(STATIC, 'js', file), 'utf8'), context, { filename: file });
  }
  return declared.nextActions;
}

const STEP = { choices: ['确认数据来源后继续'], asking: '', prompt: '', explicit: true };
const confirmed = { status: 'confirmed', database: 'miiv', label: 'MIMIC-IV Demo' };

function rendered(owner, exploration, language = 'zh') {
  const step = owner.project('**下一步：**\n- 先完成数据包注册，再生成研究计划');
  return owner.render(step || STEP, {
    language, dataSourceAuthorization: confirmed, workflowActionCode: 'idea_mining_available',
    ideaExploration: exploration,
  });
}

test('an open exploration offers the freeze instead of the dead data-preparation button', () => {
  const owner = nextActions();
  const html = rendered(owner, { open: true, question: QUESTION });
  assert.match(html, /data-gpi-idea-freeze-step/);
  assert.match(html, /冻结一个方向并继续/);
  assert.ok(html.includes(`data-gpi-idea-freeze-question="${QUESTION}"`));
  assert.doesNotMatch(html, /data-gpi-data-source-continue/);
  assert.doesNotMatch(html, /查看数据准备确认/);
  assert.match(rendered(owner, { open: true, question: QUESTION }, 'en'), /Freeze a direction and continue/);
});

test('without a direction under discussion the step asks for one and carries nothing', () => {
  const html = rendered(nextActions(), { open: true, question: '' });
  assert.match(html, /选择一个方向/);
  assert.doesNotMatch(html, /data-gpi-idea-freeze-question/);
  assert.doesNotMatch(html, /data-gpi-data-source-continue/);
});

test('a closed exploration keeps the data-source step unchanged', () => {
  const owner = nextActions();
  for (const exploration of [null, { open: false, question: QUESTION }]) {
    const html = rendered(owner, exploration);
    assert.doesNotMatch(html, /data-gpi-idea-freeze/);
    assert.match(html, /data-gpi-data-source-continue/);
  }
});

function eventsHarness(question, selector = '[data-gpi-idea-freeze]', dataset = null) {
  const listeners = {};
  const calls = [];
  const input = { value: '', focus() { calls.push(['focus', 'input']); }, matches: () => false };
  const choice = { focus() { calls.push(['focus', 'choice']); } };
  const button = { dataset: dataset || (question ? { gpiIdeaFreezeQuestion: question } : {}) };
  const host = {
    addEventListener: (name, fn) => { (listeners[name] ||= []).push(fn); },
    querySelector: selector => selector === '[data-gpi-input]' ? input : null,
    querySelectorAll: selector => selector === '[data-gpi-next-choice]' ? [choice] : [],
  };
  const state = { host, session: { session_id: 's1' }, draft: '', busy: false };
  const declared = {};
  const window = {
    EU_LANG: 'zh',
    EU_COMPOSER_KEYBOARD: { enterShouldSend: () => false },
    EU_POPOVER_MENUS: { closeAll() {}, fit() {} },
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; }, optional: name => declared[name] } },
    EU_API: {},
  };
  const document = { activeElement: null, createElement: () => ({ dataset: {}, setAttribute() {}, remove() {} }) };
  const context = vm.createContext({ window, document, location: { hash: '' } });
  for (const file of ['html-escape.js', 'screens-guided-pi-events.js']) {
    vm.runInContext(fs.readFileSync(path.join(STATIC, 'js', file), 'utf8'), context, { filename: file });
  }
  const events = declared.events.create({
    state, projectId: () => 'p1', render: () => {}, tr: (en, zh) => zh || en,
    COHORT_ELIGIBILITY: { actionFromEvent: () => null }, MESSAGE_ACTIONS: { handleClick: () => false },
    sendMessage: () => { calls.push(['send', state.draft, input.value]); },
    sendText: (...args) => { calls.push(['sendText', ...args]); },
    continueAfterDataSourceConfirmation: () => { calls.push(['continue']); },
  });
  events.wire();
  const click = () => {
    const target = { closest: wanted => wanted === selector ? button : null, matches: () => false };
    for (const fn of listeners.click || []) fn({ target, preventDefault() {}, stopPropagation() {} });
  };
  return { click, calls, state, input };
}

test('the freeze is the typed composer message, never a host turn without actions', () => {
  const { click, calls } = eventsHarness(QUESTION);
  click();
  const text = `把这个方向冻结为本项目的研究问题，然后继续：${QUESTION}`;
  assert.deepEqual(calls, [['send', text, text]]);
});

test('with no direction the click focuses the direction choice and sends nothing', () => {
  const { click, calls } = eventsHarness('');
  click();
  assert.deepEqual(calls, [['focus', 'choice']]);
});

test('a timing request button fills and focuses the message box but sends nothing (A′)', () => {
  const draft = '把结局改为院内死亡，乳酸（最大值）保持不变，然后重新生成研究计划。';
  const { click, calls, state, input } = eventsHarness('', '[data-gpi-composer-prefill]', { gpiComposerPrefill: draft });
  click();
  assert.equal(state.draft, draft);
  assert.equal(input.value, draft);
  assert.deepEqual(calls, [['focus', 'input']]);
});
