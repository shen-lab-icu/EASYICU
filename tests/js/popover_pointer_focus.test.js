/* Safari may focus a menu's ancestor during an inside pointer press. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');

const [ownerFile] = process.argv.slice(2).map(file => path.resolve(file));
const listeners = new Map();
const menus = [];
const viewListeners = new Map();
global.window = global;
global.addEventListener = (type, listener) => viewListeners.set(type, listener);
global.document = {
  addEventListener(type, listener) {
    if (!listeners.has(type)) listeners.set(type, []);
    listeners.get(type).push(listener);
  },
  querySelectorAll(selector) {
    return selector === 'details[data-popover-menu][open]' ? menus.filter(menu => menu.open) : [];
  },
};
require(ownerFile);
function dispatch(type, target, extra = {}) {
  const event = { target, key: '', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
  for (const listener of listeners.get(type) || []) listener(event);
  return event;
}
function node(parent = null) {
  return { parentElement: parent, contains(other) {
    for (let current = other; current; current = current.parentElement) if (current === this) return true;
    return false;
  } };
}
const app = node();
const menu = Object.assign(node(app), { open: true });
const button = node(menu);
const outside = node(app);
const outsideControl = node(outside);
let focused = 0;
menu.querySelector = selector => selector === ':scope > summary' ? { focus() { focused++; } } : null;
menus.push(menu);
function reset() { dispatch('pointercancel', button); menu.open = true; }

// Actual Safari ordering: pointerdown button, focusin ancestor, click button.
reset();
dispatch('pointerdown', button);
dispatch('focusin', app);
assert.equal(menu.open, true, 'ancestor focus during an inside press keeps the menu open');
dispatch('pointerup', button);

reset();
dispatch('pointerdown', button);
dispatch('focusin', button);
assert.equal(menu.open, true, 'focus on an inside button keeps the menu open');

reset();
dispatch('pointerdown', outsideControl);
assert.equal(menu.open, false, 'an outside press closes immediately');

reset();
dispatch('pointerdown', button);
dispatch('focusin', outsideControl);
assert.equal(menu.open, false, 'focus on an unrelated outside control closes');

for (const end of ['pointerup', 'pointercancel']) {
  reset();
  dispatch('pointerdown', button);
  dispatch(end, button);
  dispatch('focusin', app);
  assert.equal(menu.open, false, `${end} clears the inside press`);
}
reset();
dispatch('pointerdown', button);
assert.equal(typeof viewListeners.get('blur'), 'function', 'window blur listener is registered');
viewListeners.get('blur')({ type: 'blur' });
dispatch('focusin', app);
assert.equal(menu.open, false, 'window blur clears the inside press');

reset();
dispatch('pointerdown', button);
dispatch('keydown', button, { key: 'Tab' });
dispatch('focusin', app);
assert.equal(menu.open, false, 'keyboard input clears the inside press');

reset();
dispatch('pointerdown', button);
const escape = dispatch('keydown', button, { key: 'Escape' });
assert.equal(menu.open, false, 'Escape closes the menu');
assert.equal(escape.defaultPrevented, true);
assert.equal(focused, 1, 'Escape returns focus to the summary');
process.stdout.write('Pointer focus preserves inside actions and outside dismissal.\n');
