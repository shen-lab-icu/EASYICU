'use strict';

// The first visit records the version silently; after an update the newer
// entries show once and closing acknowledges them; Settings opens them all.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const ENTRY = { version: '1.1.0', date: '2026-11-02', changes: [{ code: 'x', zh: '新增 <b>搜索</b>', en: 'Adds search' }] };

function harness(stored) {
  const calls = [];
  const store = new Map(stored ? [['easyicu.whatsNew.seen', stored]] : []);
  let dialog = null;
  const window = {
    EU_LANG: 'zh',
    localStorage: { getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, String(value)) },
    EU_API: {
      loadWhatsNew: async seen => {
        calls.push(seen);
        const newer = seen === '1.1.0' ? [] : [ENTRY];
        return { latest_version: '1.1.0', entries: newer, show_after_update: newer.length > 0 };
      },
    },
    addEventListener() {},
  };
  const document = {
    body: { appendChild() {} },
    createElement: () => {
      const listeners = {};
      dialog = {
        className: '', innerHTML: '', open: false, dataset: {},
        addEventListener(name, handler) { listeners[name] = handler; },
        showModal() { this.open = true; },
        close() { this.open = false; },
        fire(name, event) { listeners[name](event); },
      };
      return dialog;
    },
  };
  const context = vm.createContext({ window, document, setTimeout });
  for (const file of ['html-escape.js', 'help-whats-new.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return { whatsNew: window.EU_WHATS_NEW, calls, store, dialog: () => dialog };
}

test('the first visit records the current version without showing anything', async () => {
  const { whatsNew, calls, store, dialog } = harness();
  await whatsNew.check();
  assert.deepEqual(calls, ['']);
  assert.equal(store.get('easyicu.whatsNew.seen'), '1.1.0');
  assert.equal(dialog(), null);
});

test('after an update the newer entries show once, as text, and closing acknowledges them', async () => {
  const { whatsNew, calls, store, dialog } = harness('1.0.0');
  await whatsNew.check();
  assert.deepEqual(calls, ['1.0.0']);
  assert.equal(dialog().open, true);
  assert.match(dialog().innerHTML, /<h2>EasyICU 已更新到 1\.1\.0<\/h2>/);
  assert.match(dialog().innerHTML, /<h3>1\.1\.0<small>2026-11-02<\/small><\/h3><ul><li>新增 &lt;b&gt;搜索&lt;\/b&gt;<\/li><\/ul>/);
  assert.equal(store.get('easyicu.whatsNew.seen'), '1.0.0', 'reading never acknowledges');
  dialog().fire('click', { target: { closest: selector => selector === '[data-whats-new-close]' ? {} : null } });
  assert.equal(dialog().open, false);
  assert.equal(store.get('easyicu.whatsNew.seen'), '1.1.0');
  await whatsNew.check();
  assert.deepEqual(calls, ['1.0.0', '1.1.0']);
});

test('Settings opens every entry, whatever was acknowledged', async () => {
  const { whatsNew, calls, dialog } = harness('1.1.0');
  await whatsNew.open();
  assert.deepEqual(calls, ['']);
  assert.match(dialog().innerHTML, /<h2>版本说明<\/h2>/);
  assert.match(dialog().innerHTML, /1\.1\.0/);
});
