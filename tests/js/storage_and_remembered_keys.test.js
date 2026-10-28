'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const settle = async () => { for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve)); };
const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const tr = (en, zh) => zh;
function storage() {
  const values = new Map();
  return { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
}
function node(attrs = {}, data = {}) {
  return { attrs, data, handlers: {}, disabled: false, getAttribute: key => attrs[key],
    addEventListener(name, handler) { this.handlers[name] = handler; },
    querySelector: () => null };
}
function root(nodes = {}) {
  return { querySelectorAll: selector => nodes[selector] || [], querySelector: () => null };
}
function load(file, context) { vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file }); }

async function demo() {
  const calls = [];
  const row = { id: 'synthetic', title: 'Synthetic source', version: '1', scope: {}, download: {}, provenance: {},
    status: { state: 'not_downloaded', destination: '/synthetic/数据<&', data_dir_confirmed: false } };
  const window = { location: { hash: '#data' }, sessionStorage: storage(), EU_API: {
    loadOfficialDemoSources: async () => ({ sources: [row] }),
    saveSettings: async patch => { calls.push(['save', JSON.parse(JSON.stringify(patch))]); row.status.data_dir_confirmed = true; },
    startOfficialDemoSourcePrepare: async id => { calls.push(['start', id]); return { job_id: 'synthetic-job' }; },
    loadJobSnapshot: async () => ({ status: 'done' }),
  } };
  const context = vm.createContext({ window, localStorage: storage(), setTimeout: () => 1, clearTimeout() {} });
  load('screens-viz-patient-demo-sources.js', context);
  const owner = window.EU_OFFICIAL_DEMO_SOURCES;
  await owner.ensureLoaded();
  return { window, owner, calls, row };
}

test('download confirms the displayed destination once and later cards reuse it', async () => {
  const { owner, calls } = await demo();
  const html = owner.render({ t: tr });
  assert.match(html, /将下载到：/);
  assert.match(html, /下载到这里/);
  assert.match(html, /更改位置/);
  assert.match(html, /数据&lt;&amp;/);
  assert.doesNotMatch(html, /data-demo-source-prepare=/);
  const stale = node({ 'data-demo-source-prepare': 'synthetic' });
  owner.bind(root({ '[data-demo-source-prepare]': [stale] }));
  stale.handlers.click({ preventDefault() {} });
  assert.deepEqual(calls, []);
  const confirm = node({ 'data-demo-source-confirm': 'synthetic' });
  owner.bind(root({ '[data-demo-source-confirm]': [confirm] }));
  await confirm.handlers.click();
  await settle();
  assert.deepEqual(calls, [['save', { data_dir: '/synthetic/数据<&', data_dir_confirmed: true }], ['start', 'synthetic']]);
  const reused = owner.render({ t: tr });
  assert.match(reused, /official-demo-destination/);
  assert.doesNotMatch(reused, /data-demo-source-confirm=/);
});

test('changing destination opens Storage settings and keeps the return card', async () => {
  const { owner, window, calls } = await demo();
  const change = node({ 'data-demo-source-location': 'synthetic' });
  owner.bind(root({ '[data-demo-source-location]': [change] }));
  change.handlers.click();
  assert.equal(window.location.hash, '#settings');
  assert.equal(window.sessionStorage.getItem('easyicu.settings.storageReturn'), '#data');
  assert.equal(window.sessionStorage.getItem('easyicu.settings.storageCard'), 'synthetic');
  assert.equal(window.sessionStorage.getItem('easyicu.settings.openStorage'), '1');
  assert.deepEqual(calls, []);
});

test('an already prepared legacy source can be activated before choosing a new default', async () => {
  const { owner, row, calls } = await demo();
  row.status.state = 'prepared'; row.status.registered = true; row.status.active = false;
  assert.doesNotMatch(owner.render({ t: tr }), /data-demo-source-confirm=/);
  const button = node({ 'data-demo-source-prepare': 'synthetic' });
  owner.bind(root({ '[data-demo-source-prepare]': [button] }));
  button.handlers.click({ preventDefault() {} });
  await settle();
  assert.deepEqual(calls, [['start', 'synthetic']]);
});

function provider() {
  const owners = {};
  const window = { location: {}, EasyICU: { guidedPi: { declare: (key, value) => { owners[key] = value; }, optional: key => owners[key] } } };
  class FormData { constructor(form) { this.data = form.data; } get(key) { return this.data[key] ?? null; } has(key) { return Object.hasOwn(this.data, key); } }
  const context = vm.createContext({ window, FormData });
  load('screens-guided-pi-provider.js', context);
  load('screens-guided-pi-provider-control.js', context);
  return { window, owners };
}

test('desktop remembers by default; an unchecked choice survives repaint and is submitted', async () => {
  const { owners } = provider();
  const config = { remember_key_supported: true };
  const state = { researchProvider: 'api' };
  const options = { state, config, runtime: {}, tr, esc, option: () => '', providerPreset: () => 'deepseek' };
  const render = () => owners.provider.renderConnectionBody(options);
  assert.match(render(), /name="remember_key" checked/);
  assert.match(render(), /在本机记住此密钥/);
  config.remember_key_supported = false;
  assert.doesNotMatch(render(), /name="remember_key"/);
  config.remember_key_supported = true;
  const data = { provider: 'deepseek', base_url: 'https://api.deepseek.com/v1', api_transport: 'openai-completions', model: 'fixture', api_key: 'synthetic-form-key' };
  const form = node({}, data);
  form.querySelector = selector => selector === '[name="remember_key"]' ? { checked: false } : null;
  const host = { querySelector: () => form };
  const sent = [];
  const control = owners.providerControl.create({ state, tr, render() {}, api: () => ({ savePiCopilotProviderConfig: async body => { sent.push(body); return {}; } }),
    runtimeReady: () => true, shellReady: () => true, connectionConfigured: () => true, connectionReady: () => true, errorText: error => error.message });
  control.captureForm(host);
  assert.equal(state.apiDraft.remember_key, false);
  assert.doesNotMatch(render(), /name="remember_key" checked/);
  assert.equal(JSON.stringify(state).includes('synthetic-form-key'), false);
  state.modelCatalog = { manual: true, models: [], signature: owners.provider.connectionSignature(data) };
  await control.configureProvider(form);
  assert.equal(sent[0].remember_key, false);
  assert.equal(sent[0].enable_ai, true);
});

test('Settings lists and deletes remembered credentials using only their IDs', async () => {
  const calls = [];
  let refuse = true;
  const row = { id: 'synthetic-id', provider: 'Synthetic provider', base_url: 'https://fixture.invalid' };
  const window = { EU_SETTINGS: { about: { deployment: 'desktop' } }, EU_DATABASE_NAMES: { keys: () => [], name: value => value },
    sessionStorage: storage(), __euRender() {}, EU_API: {
      loadRememberedProviderKeys: async () => ({ credentials: [row] }),
      deleteRememberedProviderKey: async id => { calls.push(id); if (refuse) throw new Error('系统凭据库已锁定'); return { credentials: [] }; },
    } };
  const context = vm.createContext({ window, t: tr, icon: () => '' });
  load('screens-settings.js', context);
  const screen = window.SCREENS.settings;
  screen.afterRender(root());
  await settle();
  assert.match(screen.render(), /已记住密钥/);
  assert.match(screen.render(), /删除/);
  const button = node({ 'data-forget-key': 'synthetic-id' });
  screen.afterRender(root({ '[data-forget-key]': [button] }));
  await button.handlers.click();
  assert.match(screen.render(), /role="alert"/);
  assert.match(screen.render(), /系统凭据库已锁定/);
  assert.match(screen.render(), /data-forget-key=/);
  refuse = false;
  await button.handlers.click();
  assert.deepEqual(calls, ['synthetic-id', 'synthetic-id']);
  assert.doesNotMatch(screen.render(), /data-forget-key=/);
  assert.doesNotMatch(screen.render(), /role="alert"/);
});

test('an unavailable desktop store disables remembering and explains why', () => {
  const { owners } = provider();
  const reason = '系统凭据库不可用；密钥仅用于本次启动。';
  const html = owners.provider.renderConnectionBody({ state: { researchProvider: 'api' },
    config: { remember_key_supported: false, remember_key_unavailable_reason: reason },
    runtime: {}, tr, esc, option: () => '', providerPreset: () => 'deepseek' });
  assert.match(html, /name="remember_key" disabled/);
  assert.match(html, /在本机记住此密钥/);
  assert.ok(html.includes(reason));
  assert.doesNotMatch(html, /name="remember_key" checked/);
});

test('a rejected storage location stays editable and is reported as an error', async () => {
  const window = { EU_SETTINGS: { data_dir: '/synthetic/old', about: {} },
    EU_DATABASE_NAMES: { keys: () => [], name: value => value }, sessionStorage: storage(), __euRender() {},
    EU_API: { saveSettings: async () => { throw new Error('没有此文件夹的写入权限。'); } } };
  class FormData { constructor(form) { this.data = form.data; } get(key) { return this.data[key] ?? null; } }
  const context = vm.createContext({ window, t: tr, icon: () => '', FormData });
  load('screens-settings.js', context);
  const screen = window.SCREENS.settings;
  const change = node({ 'data-storage-change': 'data_dir' });
  screen.afterRender(root({ '[data-storage-change]': [change] }));
  change.handlers.click();
  const form = node({ 'data-storage-form': 'data_dir' }, { path: '/synthetic/unwritable' });
  form.querySelector = () => ({});
  screen.afterRender(root({ '[data-storage-form]': [form] }));
  await form.handlers.submit({ preventDefault() {} });
  const html = screen.render();
  assert.match(html, /role="alert"/);
  assert.match(html, /没有此文件夹的写入权限。/);
  assert.match(html, /value="\/synthetic\/unwritable"/);
  assert.doesNotMatch(html, /设置已更新/);
});

test('Settings saves both paths and returns to the card without a popup', async () => {
  const sent = [];
  const window = { EU_SETTINGS: { data_dir: '/synthetic/数据<&', project_dir: '/synthetic/项目', about: {} },
    EU_DATABASE_NAMES: { keys: () => [], name: value => value }, sessionStorage: storage(), location: { hash: '#settings' },
    EU_API: { saveSettings: async patch => { sent.push(JSON.parse(JSON.stringify(patch))); Object.assign(window.EU_SETTINGS, patch); } },
    __euRender() {} };
  class FormData { constructor(form) { this.data = form.data; } get(key) { return this.data[key] ?? null; } }
  const context = vm.createContext({ window, t: tr, icon: () => '', FormData });
  load('screens-settings.js', context);
  const screen = window.SCREENS.settings;
  const html = screen.render();
  for (const copy of ['存储位置', '数据目录', '默认项目目录', '更改']) assert.ok(html.includes(copy), copy);
  assert.match(html, /数据&lt;&amp;/);
  const change = node({ 'data-storage-change': 'data_dir' });
  screen.afterRender(root({ '[data-storage-change]': [change] }));
  change.handlers.click();
  assert.match(screen.render(), /data-storage-form="data_dir"/);
  const submit = {};
  const form = node({ 'data-storage-form': 'data_dir' }, { path: '/synthetic/new-data' });
  form.querySelector = () => submit;
  screen.afterRender(root({ '[data-storage-form]': [form] }));
  await form.handlers.submit({ preventDefault() {} });
  assert.deepEqual(sent, [{ data_dir: '/synthetic/new-data', data_dir_confirmed: true }]);
  window.sessionStorage.setItem('easyicu.settings.storageReturn', '#data');
  const back = node();
  screen.afterRender(root({ '[data-storage-return]': [back] }));
  back.handlers.click();
  assert.equal(window.location.hash, '#data');
});

test('a project picker uses the latest saved default after leaving Settings', () => {
  const window = { EU_SETTINGS: { project_dir: '/synthetic/new-projects' } };
  const context = vm.createContext({ window, localStorage: storage() });
  load('screens-guided-projects.js', context);
  assert.equal(window.EU_GUIDED_PROJECTS.defaultParentDir({ guidedDrafts: { data: { default_parent_dir: '/synthetic/old-projects' } } }), '/synthetic/new-projects');
});
