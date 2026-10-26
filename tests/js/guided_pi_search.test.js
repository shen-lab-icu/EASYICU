'use strict';

// Search shows the host's hits as text and opens them through the owners that
// already open a conversation, a project and a Drive file; it never opens a
// conversation while a reply is running.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

const SNIPPET = '\nCompare <b>survival</b> 🌱 by lactate';
const HITS = [
  { kind: 'message', project_id: 'p1', title: '', snippet: SNIPPET,
    matches: [{ start: SNIPPET.indexOf('survival'), end: SNIPPET.indexOf('survival') + 8 }, { start: 40, end: 90 }], offset_unit: 'utf16',
    open: { kind: 'conversation', source: 'pi', project_id: 'p1', conversation_id: 'c1', message_id: 'e1' } },
  { kind: 'conversation', project_id: 'p2', title: 'Sepsis cohort', snippet: 'Sepsis cohort', matches: [],
    offset_unit: 'utf16', open: { kind: 'conversation', source: 'pi', project_id: 'p2', conversation_id: 'c9' } },
  { kind: 'file', project_id: 'p2', title: 'Report', snippet: '', matches: [], offset_unit: 'utf16',
    open: { kind: 'file', project_id: 'p2', run_id: 'r1', run_ref: 'ref', artifact: 'report.md', file_id: 'report.md' } },
];

function harness({ canOpen = true } = {}) {
  const calls = [];
  const listeners = {};
  const input = { value: '', focus() {}, select() {}, matches: selector => selector === '[data-gpi-search-input]' };
  const results = { innerHTML: '' };
  const dialog = {
    className: '', innerHTML: '', open: false, attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, handler) { listeners[name] = handler; },
    querySelector(selector) {
      if (selector === '[data-gpi-search-input]') return input;
      if (selector === '[data-gpi-search-results]') return results;
      return null;
    },
    querySelectorAll() { return []; },
    showModal() { this.open = true; },
    close() { this.open = false; },
    remove() {},
  };
  const material = { dataset: { gpiRailMaterialKey: 'r1/report.md' }, click() { calls.push(['material', 'r1/report.md']); } };
  const declared = {};
  const window = {
    EU_LANG: 'zh',
    icon: () => '',
    EasyICU: { guidedPi: {
      declare: (name, api) => { declared[name] = api; },
      optional: name => name === 'project' ? { syncLocation: (...args) => calls.push(['location', ...args]) } : null,
    } },
    EU_API: {
      searchWorkspace: async params => { calls.push(['search', params]); return { hits: HITS, has_more: true, index_truncated: false, issues: [] }; },
      loadGuidedDrafts: async () => ({ drafts: [{ id: 'p2', title: 'ICU sepsis' }] }),
    },
    EU_GUIDED_CONTEXT: { openProject: async id => { calls.push(['project', id]); return true; } },
  };
  const document = {
    createElement: () => dialog,
    body: { appendChild() {} },
    activeElement: null,
    querySelectorAll: selector => selector === '[data-gpi-rail-material-key]' ? [material] : [],
  };
  const context = vm.createContext({ window, document, setTimeout, clearTimeout });
  for (const file of ['html-escape.js', 'screens-guided-pi-search.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  const ctx = {
    projectId: () => 'p1', projectTitle: () => 'Lactate study',
    openSession: async id => { calls.push(['session', id]); }, canOpen: () => canOpen,
  };
  const search = declared.globalSearch;
  const type = async value => { input.value = value; listeners.input({ target: input }); await wait(230); };
  const clickHit = async index => {
    await listeners.click({ target: { closest: selector => selector === '[data-gpi-search-hit]' ? { dataset: { gpiSearchHit: String(index) } } : null } });
    await wait(10);
  };
  return { search, ctx, dialog, results, calls, type, clickHit, listeners };
}

test('a hit is shown as text, with the host’s UTF-16 matches marked', async () => {
  const { search, ctx, results, calls, type } = harness();
  search.open(ctx);
  await type('survival');
  assert.deepEqual(JSON.parse(JSON.stringify(calls.find(row => row[0] === 'search')[1])), { q: 'survival', limit: 20 });
  assert.match(results.innerHTML, /Compare &lt;b&gt;<mark>survival<\/mark>&lt;\/b&gt; 🌱 by lactate/);
  assert.match(results.innerHTML, /<small>消息 · Lactate study<\/small>/);
  assert.match(results.innerHTML, /<small>对话 · ICU sepsis<\/small>/);
  assert.match(results.innerHTML, /只显示前 3 条，多加一个词可以缩小范围。/);
  // A mark past the snippet's end is dropped, not clipped.
  assert.equal((results.innerHTML.match(/<mark>/g) || []).length, 1);
});

test('a hit in this project opens its conversation; another project opens through the project owner', async () => {
  const first = harness();
  first.search.open(first.ctx);
  await first.type('survival');
  await first.clickHit(0);
  assert.deepEqual(first.calls.filter(row => row[0] === 'session'), [['session', 'c1']]);
  assert.equal(first.dialog.open, false);

  const second = harness();
  second.search.open(second.ctx);
  await second.type('sepsis');
  await second.clickHit(1);
  assert.deepEqual(second.calls.filter(row => row[0] !== 'search').map(row => [...row]), [['location', 'p2', 'c9'], ['project', 'p2']]);

  const third = harness();
  third.search.open(third.ctx);
  await third.type('report');
  await third.clickHit(2);
  assert.deepEqual(third.calls.filter(row => row[0] !== 'search').map(row => [...row]),
    [['location', 'p2', ''], ['project', 'p2'], ['material', 'r1/report.md']]);
});

test('a conversation does not open while a reply is running', async () => {
  const { search, ctx, dialog, results, calls, type, clickHit } = harness({ canOpen: false });
  search.open(ctx);
  await type('survival');
  await clickHit(0);
  assert.equal(calls.some(row => row[0] === 'session'), false);
  assert.equal(dialog.open, true);
  assert.match(results.innerHTML, /role="alert">请等当前回复结束后再打开。/);
});

test('the scope narrows the host query and the backdrop closes the dialog', async () => {
  const { search, ctx, dialog, calls, listeners, type } = harness();
  search.open(ctx);
  await type('lactate');
  listeners.click({ target: { closest: selector => selector === '[data-gpi-search-scope]' ? { dataset: { gpiSearchScope: 'files' } } : null } });
  await wait(10);
  assert.deepEqual(JSON.parse(JSON.stringify(calls.filter(row => row[0] === 'search').pop()[1])), { q: 'lactate', scope: 'files', limit: 20 });
  listeners.click({ target: dialog });
  assert.equal(dialog.open, false);
});
