/* A refusal code reads as a sentence in the UI language, never as the code. Synthetic paths only. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const en = (english) => english;
const zh = (english, chinese) => chinese;
function load(file, context) { vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file }); }

function errorOwner() {
  const window = {};
  window.window = window;
  const context = vm.createContext(window);
  for (const file of ['screens-guided-pi-modules.js', 'html-escape.js', 'screens-guided-pi-error-text.js']) load(file, context);
  return window.EasyICU.guidedPi.require('errorText');
}

// The detail a refused POST /api/settings carries (settings.py SettingsValidationError).
function rejectedSave(code, reason) {
  const error = new Error(`data_dir: ${reason}`);
  error.code = 'settings_patch_rejected';
  error.detail = { error: 'settings_patch_rejected', reason: `data_dir: ${reason}`,
    rejected: [{ key: 'data_dir', code, reason }] };
  return error;
}

test('a refused storage folder reads in the UI language without the key prefix', () => {
  const owner = errorOwner();
  const error = rejectedSave('storage_path_not_creatable', 'This folder does not exist, and EasyICU cannot create it there.');
  const english = owner.storageErrorText(error, en);
  assert.match(english, /^This folder does not exist, and EasyICU cannot create it there\./);
  assert.doesNotMatch(english, /data_dir|[一-鿿]/);
  assert.match(owner.storageErrorText(error, zh), /^此文件夹不存在/);
  assert.equal(owner.storageErrorText(rejectedSave('storage_path_not_writable', 'x'), en),
    'EasyICU does not have permission to write to this folder.');
  // An unknown code keeps the server sentence, still without "data_dir:".
  assert.equal(owner.storageErrorText(rejectedSave('storage_future_code', 'A new server reason.'), en), 'A new server reason.');
});

test('the export-folder codes the Data pages and folder pickers receive read as sentences', () => {
  const owner = errorOwner();
  for (const code of ['no_export_files', 'not_a_directory', 'permission_denied']) {
    const english = owner.scanErrorText(code, en);
    assert.notEqual(english, code);
    assert.doesNotMatch(english, /_/);
    assert.match(owner.scanErrorText(code, zh), /[一-鿿]/);
  }
  assert.match(owner.scanErrorText('no_export_files', en), /no EasyICU export files/);
});

test('Settings shows a refused data folder as one sentence in English', async () => {
  const owner = errorOwner();
  const storage = new Map();
  const window = {
    EU_SETTINGS: { data_dir: '/synthetic/old', about: {} },
    EU_DATABASE_NAMES: { keys: () => [], name: value => value },
    sessionStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    __euRender() {},
    EasyICU: { guidedPi: { optional: name => (name === 'errorText' ? owner : null) } },
    EU_API: { saveSettings: async () => { throw rejectedSave('storage_path_not_creatable', 'This folder does not exist, and EasyICU cannot create it there.'); } },
  };
  class FormData { constructor(form) { this.data = form.data; } get(key) { return this.data[key] ?? null; } }
  const node = (attrs = {}, data = {}) => ({ attrs, data, handlers: {}, disabled: false, getAttribute: key => attrs[key],
    addEventListener(name, handler) { this.handlers[name] = handler; }, querySelector: () => ({}) });
  const root = (nodes = {}) => ({ querySelectorAll: selector => nodes[selector] || [], querySelector: () => null });
  const context = vm.createContext({ window, t: en, icon: () => '', FormData });
  load('screens-settings.js', context);
  const screen = window.SCREENS.settings;
  const change = node({ 'data-storage-change': 'data_dir' });
  screen.afterRender(root({ '[data-storage-change]': [change] }));
  change.handlers.click();
  const form = node({ 'data-storage-form': 'data_dir' }, { path: '/synthetic/missing/folder' });
  screen.afterRender(root({ '[data-storage-form]': [form] }));
  await form.handlers.submit({ preventDefault() {} });
  const html = screen.render();
  assert.match(html, /role="alert">This folder does not exist, and EasyICU cannot create it there\./);
  assert.doesNotMatch(html, /data_dir:|没有此文件夹的写入权限/);
});
