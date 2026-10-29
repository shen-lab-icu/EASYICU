'use strict';

// Search shows the host's hits as text and opens them through the owners that
// already open a conversation, a project and a Drive file, then reveals the
// matched message; it never opens a conversation while a reply is running.
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

function harness({ canOpen = true, hits = HITS, includeMarkdown = true } = {}) {
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
  const classes = new Set();
  const message = {
    dataset: { gpiEntryId: 'e1' },
    scrollIntoView(options) { calls.push(['reveal', 'e1', options.block]); },
    classList: { add: name => classes.add(name), remove: name => classes.delete(name) },
  };
  const declared = {};
  const window = {
    EU_LANG: 'zh',
    icon: () => '',
    EasyICU: { guidedPi: {
      declare: (name, api) => { declared[name] = api; },
      optional: name => name === 'project' ? {
        syncLocation: (...args) => calls.push(['location', ...args]),
        openListedProject: id => { calls.push(['project', id]); return true; },
      } : (includeMarkdown ? declared[name] || null : null),
    } },
    EU_API: {
      searchWorkspace: async params => { calls.push(['search', params]); return { hits, has_more: true, index_truncated: false, issues: [] }; },
      loadGuidedDrafts: async () => ({ drafts: [{ id: 'p2', title: 'ICU sepsis' }] }),
    },
  };
  const document = {
    createElement: () => dialog,
    body: { appendChild() {} },
    activeElement: null,
    querySelectorAll: selector => selector === '[data-gpi-rail-material-key]' ? [material]
      : selector === '[data-gpi-entry-id]' ? [message] : [],
  };
  const context = vm.createContext({ window, document, setTimeout, clearTimeout });
  for (const file of ['html-escape.js', 'screens-guided-pi-markdown.js', 'screens-guided-pi-search.js']) {
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
  return { search, ctx, dialog, results, calls, type, clickHit, listeners, classes };
}

test('a hit is shown as text, with the host’s UTF-16 matches marked', async () => {
  const { search, ctx, results, calls, type } = harness();
  search.open(ctx);
  await type('survival');
  assert.deepEqual(JSON.parse(JSON.stringify(calls.find(row => row[0] === 'search')[1])), { q: 'survival', limit: 20 });
  assert.match(results.innerHTML, /Compare &lt;b&gt;<mark>survival<\/mark>&lt;\/b&gt; 🌱 by lactate/);
  assert.match(results.innerHTML, /<small>对话 · 1 条匹配消息 · Lactate study<\/small>/);
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
  // The matched message is scrolled into view and marked for a moment.
  assert.deepEqual(first.calls.filter(row => row[0] === 'reveal'), [['reveal', 'e1', 'center']]);
  assert.equal(first.classes.has('is-search-hit'), true);

  const second = harness();
  second.search.open(second.ctx);
  await second.type('sepsis');
  await second.clickHit(1);
  // A conversation hit names no message, so nothing is revealed.
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

test('matching messages are grouped by project and conversation without losing message navigation', async () => {
  const message = HITS[0];
  const same = { ...message, message_id: 'e2', snippet: '**Second** matching message', matches: [],
    open: { ...message.open, message_id: 'e2' } };
  const otherProject = { ...message, project_id: 'p2',
    open: { ...message.open, project_id: 'p2' } };
  const named = { ...message, kind: 'conversation', title: 'Lactate design',
    snippet: 'Lactate design', open: { ...message.open, message_id: undefined } };
  const { search, ctx, results, type, clickHit, calls } = harness({ hits: [named, message, same, otherProject] });
  search.open(ctx);
  await type('matching');
  assert.match(results.innerHTML, /2 个结果/);
  assert.match(results.innerHTML, /<strong>Lactate design<\/strong>/);
  assert.match(results.innerHTML, /对话 · 2 条匹配消息 · Lactate study/);
  assert.match(results.innerHTML, /展开匹配消息/);
  assert.match(results.innerHTML, /Second matching message/);
  assert.doesNotMatch(results.innerHTML, /\*\*Second\*\*/);
  await clickHit(1);
  assert.deepEqual(calls.filter(row => row[0] === 'session'), [['session', 'c1']]);
  assert.deepEqual(calls.filter(row => row[0] === 'reveal'), [['reveal', 'e1', 'center']]);
});

test('the current-project switch is a host query filter', async () => {
  const { search, ctx, calls, listeners, type } = harness();
  search.open(ctx);
  await type('lactate');
  const filter = { checked: true, matches: selector => selector === '[data-gpi-search-project]' };
  listeners.input({ target: filter });
  await wait(10);
  assert.deepEqual(JSON.parse(JSON.stringify(calls.filter(row => row[0] === 'search').pop()[1])),
    { q: 'lactate', project: 'p1', limit: 20 });
  filter.checked = false;
  listeners.input({ target: filter });
  await wait(10);
  assert.deepEqual(JSON.parse(JSON.stringify(calls.filter(row => row[0] === 'search').pop()[1])),
    { q: 'lactate', limit: 20 });
});

test('cleaning Markdown retains UTF-16 highlights and treats HTML as text', () => {
  const { search } = harness();
  const snippet = '### **模型** [结果](https://example.test/report) <img src=x onerror=alert(1)> 🌱';
  const start = snippet.indexOf('模型');
  const html = search.snippetHtml({ snippet, matches: [{ start, end: start + 2 }], offset_unit: 'utf16' });
  assert.match(html, /<mark>模型<\/mark> 结果/);
  assert.doesNotMatch(html, /\*\*|###|https:\/\/example/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt; 🌱/);
});

test('search snippets preserve scientific code and literal underscores while cleaning paired prose', () => {
  const { search } = harness();
  for (const snippet of ['`dose__adjusted` and `x**2`', 'dose__adjusted and x**2', '```python\ndose__adjusted = x**2\n```']) {
    const start = snippet.indexOf('__');
    const html = search.snippetHtml({ snippet, matches: [{ start, end: start + 2 }], offset_unit: 'utf16' });
    assert.match(html, /dose<mark>__<\/mark>adjusted/);
    assert.match(html, /x\*\*2/);
  }
  assert.equal(search.snippetHtml({ snippet: '__Adjusted model__ and **mortality**', matches: [] }), 'Adjusted model and mortality');
});

// Real UAT replies place bold phrases directly after Chinese punctuation.
test('Chinese sentence punctuation does not leak paired Markdown markers', () => {
  const { search } = harness();
  const snippet = '这个问题：**成人 ICU 乳酸变化**。仍需 __确认设计__。';
  const start = snippet.indexOf('乳酸');
  const html = search.snippetHtml({snippet, matches: [{start,end:start+2}], offset_unit:'utf16'});
  assert.equal(html, '这个问题：成人 ICU <mark>乳酸</mark>变化。仍需 确认设计。');
});


test('supported math snippets remain readable with exact repeated UTF-16 highlights', () => {
  const { search } = harness();
  const snippet = String.raw`🌱 清除率：\[\frac{\text{基线乳酸}-\text{后续乳酸}}{\text{基线乳酸}}\times100\%\]；模型。`;
  const first = snippet.indexOf('基线乳酸'), second = snippet.lastIndexOf('基线乳酸');
  const html = search.snippetHtml({ snippet, offset_unit: 'utf16', matches: [
    { start: first, end: first + 4 }, { start: second, end: second + 4 },
  ] });
  assert.match(html, /🌱 清除率：\(<mark>基线乳酸<\/mark>-后续乳酸\) \/ \(<mark>基线乳酸<\/mark>\) × 100%/);
  assert.doesNotMatch(html, /\\(?:frac|text|times|\[|\])/);
});

test('inline and block maths support nested fractions and independent adjacent hits', () => {
  const { search } = harness();
  const snippet = String.raw`公式\(\beta + A\) 和 $$\frac{x}{\frac{y}{2}}$$，最后是结局`;
  const starts = [snippet.indexOf('A'), snippet.indexOf('y'), snippet.indexOf('结局')];
  const html = search.snippetHtml({ snippet, offset_unit: 'utf16', matches: starts.map((start, index) => ({start, end: start + (index === 2 ? 2 : 1)})) });
  assert.match(html, /公式β \+ <mark>A<\/mark> 和 \(x\) \/ \(\(<mark>y<\/mark>\) \/ \(2\)\)，最后是<mark>结局<\/mark>/);
});

test('a hit on TeX control syntax keeps source so the marked query stays exact', () => {
  const { search } = harness();
  const snippet = String.raw`\[\frac{a}{b}\]`;
  const start = snippet.indexOf('frac');
  const html = search.snippetHtml({ snippet, offset_unit: 'utf16', matches: [{ start, end: start + 4 }] });
  assert.equal(html, String.raw`\[\<mark>frac</mark>{a}{b}\]`);
});

test('unknown, partial and code formulas stay literal and cannot inject HTML', () => {
  const { search } = harness();
  const snippets = [
    String.raw`\[\href{javascript:alert(1)}{x}\]`,
    String.raw`\[\unknown{[text](url)}\]`,
    String.raw`\[\frac{a}{b`,
    '`' + String.raw`\(\frac{a}{b}\)` + '`',
    '```python\n' + String.raw`\[\frac{a}{b}\]` + '\n```',
  ];
  for (const snippet of snippets) {
    const html = search.snippetHtml({ snippet, matches: [] });
    if (snippet.includes('frac')) assert.match(html, /\\frac\{a\}\{b/);
    if (snippet.includes('unknown')) assert.match(html, /\\unknown\{\[text\]\(url\)\}/);
    if (snippet.includes('href')) assert.match(html, /\\href\{javascript:alert\(1\)\}/);
    assert.doesNotMatch(html, /<a\b|<script|<img/);
  }
  const malicious = String.raw`\[\text{<img src=x onerror=alert(1)>}\]`;
  const html = search.snippetHtml({ snippet: malicious, matches: [] });
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.doesNotMatch(html, /<img/);
});

test('no markdown owner and escaped delimiters leave formulas unchanged', () => {
  const noOwner = harness({ includeMarkdown: false }).search;
  const source = String.raw`\[\frac{a}{b}\]`;
  assert.equal(noOwner.snippetHtml({ snippet: source, matches: [] }), source);
  const escaped = String.raw`\\[\frac{a}{b}\]`;
  assert.equal(harness().search.snippetHtml({ snippet: escaped, matches: [] }), escaped);
});


test('indented and tilde fences and multi-backtick spans keep TeX literal', () => {
  const { search } = harness();
  const formula = String.raw`\[\frac{a}{b}\]`;
  for (const snippet of [`~~~tex\n${formula}\n~~~`, `  \`\`\`\`tex\n${formula}\n  \`\`\`\``, `\`\`${formula}\`\``]) {
    const html = search.snippetHtml({ snippet, matches: [] });
    assert.match(html, /\\frac\{a\}\{b\}/);
    assert.doesNotMatch(html, /\(a\) \/ \(b\)/);
  }
});

test('an incomplete formula with Markdown-looking text is not partly reformatted', () => {
  const { search } = harness();
  const snippet = String.raw`计算：\[\unknown{**待确认**}`;
  assert.equal(search.snippetHtml({ snippet, matches: [] }), snippet);
});
