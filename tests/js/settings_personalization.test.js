'use strict';

// Standing instructions are read from and saved to the host's preferences
// owner; only the instructions field is sent, and a refusal is said in words.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise(resolve => setImmediate(resolve)); };

function owner(update) {
  const sent = [];
  const window = {
    EU_LANG: 'zh',
    EU_API: {
      loadPersonalPreferences: async () => ({ preferences: { instructions: 'Use <b>SOFA</b>', figure_export_format: 'svg' }, options: { max_instructions_chars: 20 } }),
      updatePersonalPreferences: async body => { sent.push(body); return update ? update(body) : { preferences: { instructions: body.instructions } }; },
    },
  };
  const context = vm.createContext({ window });
  for (const file of ['html-escape.js', 'screens-settings-personalization.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  const listeners = {};
  const nodes = { button: { disabled: true }, counter: { textContent: '', classList: { toggle() {} } } };
  const box = {
    innerHTML: '', isConnected: true,
    addEventListener: (name, handler) => { listeners[name] = handler; },
    querySelector: selector => selector === '[data-set-personal-save]' ? nodes.button : selector === '[data-set-personal-count]' ? nodes.counter : null,
  };
  const root = { querySelector: selector => selector === '[data-set-personal]' ? box : null };
  return { ui: window.EU_SETTINGS_PERSONALIZATION, root, box, listeners, nodes, sent };
}

test('saved instructions load as text and only a change can be saved', async () => {
  const { ui, root, box, listeners, nodes, sent } = owner();
  ui.bind(root);
  await settle();
  assert.match(box.innerHTML, />Use &lt;b&gt;SOFA&lt;\/b&gt;<\/textarea>/);
  assert.match(box.innerHTML, /data-set-personal-save disabled/);
  listeners.input({ target: { matches: () => true, value: 'Answer briefly.' } });
  assert.equal(nodes.button.disabled, false);
  listeners.input({ target: { matches: () => true, value: 'x'.repeat(21) } });
  assert.equal(nodes.button.disabled, true);
  listeners.input({ target: { matches: () => true, value: 'Answer briefly.' } });
  listeners.click({ target: { closest: selector => selector === '[data-set-personal-save]' ? {} : null } });
  await settle();
  assert.deepEqual(JSON.parse(JSON.stringify(sent)), [{ instructions: 'Answer briefly.' }]);
  assert.match(box.innerHTML, /role="status">已保存，新对话起生效。/);
});

test('a refusal keeps the text and says why', async () => {
  const { ui, root, box, listeners } = owner(() => { throw Object.assign(new Error('x'), { code: 'pi_message_phi_risk' }); });
  ui.bind(root);
  await settle();
  listeners.input({ target: { matches: () => true, value: 'MRN 12345' } });
  listeners.click({ target: { closest: selector => selector === '[data-set-personal-save]' ? {} : null } });
  await settle();
  assert.match(box.innerHTML, /role="alert">这段文字疑似含有患者信息，请删去后再保存。/);
  assert.match(box.innerHTML, />MRN 12345<\/textarea>/);
});
