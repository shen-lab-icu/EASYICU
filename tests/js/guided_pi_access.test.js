'use strict';

// The access owner: what each mode grants, and the question asked when the
// host held a step for the researcher's go-ahead.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const js = path.join(__dirname, '..', '..', 'src', 'easyicu', 'webserver', 'static', 'js');

function owner(language = 'zh', storage = new Map()) {
  const localStorage = {
    getItem: key => (storage.has(key) ? storage.get(key) : null),
    setItem: (key, value) => { storage.set(key, String(value)); },
  };
  const context = vm.createContext({ console, window: { EU_LANG: language, localStorage } });
  for (const name of ['screens-guided-pi-modules.js', 'html-escape.js', 'screens-guided-pi-access.js']) {
    vm.runInContext(fs.readFileSync(path.join(js, name), 'utf8'), context);
  }
  return { access: context.window.EasyICU.guidedPi.require('access'), storage };
}

const blocked = (action, grantable = true) => ({
  kind: 'tool', status: 'error', code: 'pi_action_authorization_required', requiredAction: action, grantable,
});

test('each mode grants its routine actions; stopping a run and privileged actions are never routine', () => {
  const { access } = owner();
  assert.deepEqual([...access.MODES], ['ask', 'assist']);
  assert.deepEqual([...access.grants('ask', 'research')], []);
  assert.deepEqual([...access.grants('assist', 'research')], ['idea', 'literature', 'configure', 'run', 'mcp_read']);
  assert.ok(access.grants('assist', 'workspace').includes('workspace_write'));
  // The retired "allow more" mode reads as the default.
  assert.deepEqual([...access.grants('full', 'research')], [...access.grants('assist', 'research')]);
  for (const mode of access.MODES) {
    for (const action of ['cancel', 'extract', 'provider_run', 'report_revision']) {
      assert.ok(!access.grants(mode, 'workspace').includes(action), `${mode} must not grant ${action}`);
    }
  }
});

test('the chosen mode is remembered in this browser, and a broken store falls back to the default', () => {
  const { access, storage } = owner();
  assert.equal(access.preferred(), 'assist');
  access.remember('ask');
  assert.equal(storage.get('easyicu.pi.accessMode.v1'), 'ask');
  assert.equal(access.preferred(), 'ask');
  storage.set('easyicu.pi.accessMode.v1', 'full');
  assert.equal(access.preferred(), 'assist');
});

test('only steps held in the latest turn, that a turn grant can allow, are asked about', () => {
  const { access } = owner();
  const messages = [
    { role: 'user', text: 'first' },
    { role: 'activity', steps: [blocked('configure')] },
    { role: 'user', text: 'second' },
    { role: 'activity', steps: [
      blocked('literature'), blocked('literature'), blocked('provider_run', false), blocked('extract', false),
      blocked('unknown_action'), { kind: 'tool', status: 'complete', code: 'study_context_updated' },
    ] },
    { role: 'activity', steps: [blocked('cancel')] },
    { role: 'assistant', text: 'needs approval' },
  ];
  assert.deepEqual([...access.pending(messages)], ['literature', 'cancel']);
  assert.deepEqual([...access.pending(messages.concat({ role: 'user', text: 'third' }))], []);
});

test('the card names the actions; in ask mode it also offers switching to routine confirmation', () => {
  const { access } = owner('zh');
  const ask = access.render(['literature', 'configure'], { mode: 'ask', agentMode: 'research' });
  assert.match(ask, /助手要检索文献、修改研究配置。/);
  assert.match(ask, /data-gpi-action-ask-allow="literature configure"[^>]*>允许这一次</);
  assert.match(ask, /data-gpi-action-ask-routine="literature configure"[^>]*>常规操作自动确认</);
  // Stopping a run is never routine, so only this once is offered.
  const stop = access.render(['cancel'], { mode: 'ask', agentMode: 'research' });
  assert.match(stop, /助手要停止正在进行的运行。/);
  assert.doesNotMatch(stop, /data-gpi-action-ask-routine/);
  assert.doesNotMatch(access.render(['literature'], { mode: 'assist', agentMode: 'research' }), /data-gpi-action-ask-routine/);
  assert.match(access.render(['run'], { mode: 'ask', agentMode: 'research', disabled: true }), /disabled/);
  assert.equal(access.render([], { mode: 'ask' }), '');
  const en = owner('en').access.render(['literature'], { mode: 'ask', agentMode: 'research' });
  assert.match(en, /The assistant wants to search the literature\./);
});

function click(attributes) {
  const button = {
    dataset: attributes.dataset,
    hasAttribute: name => Boolean(attributes.routine) && name === 'data-gpi-action-ask-routine',
  };
  return { target: { closest: () => button }, preventDefault() {} };
}

test('allowing sends one follow-up with exactly those grants added to the mode', () => {
  const { access, storage } = owner('zh');
  const sent = [];
  let mode = 'ask';
  const callbacks = {
    busy: () => false, setMode: value => { mode = value; },
    grants: () => access.grants(mode, 'research'),
    send: (text, grants) => sent.push({ text, grants: [...grants] }),
  };
  assert.equal(access.handleClick(click({ dataset: { gpiActionAskAllow: 'literature configure' } }), callbacks), true);
  assert.deepEqual(sent[0], { text: '同意这一步，请继续。', grants: ['literature', 'configure'] });
  assert.equal(mode, 'ask');

  assert.equal(access.handleClick(click({ routine: true, dataset: { gpiActionAskRoutine: 'literature' } }), callbacks), true);
  assert.equal(mode, 'assist');
  assert.equal(storage.get('easyicu.pi.accessMode.v1'), 'assist');
  assert.deepEqual([...sent[1].grants].sort(), [...access.grants('assist', 'research')].sort());

  // An action outside the vocabulary, or a busy conversation, sends nothing.
  access.handleClick(click({ dataset: { gpiActionAskAllow: 'provider_run_typo' } }), callbacks);
  access.handleClick(click({ dataset: { gpiActionAskAllow: 'run' } }), { ...callbacks, busy: () => true });
  assert.equal(sent.length, 2);
  assert.equal(access.handleClick({ target: { closest: () => null } }, callbacks), false);
});
