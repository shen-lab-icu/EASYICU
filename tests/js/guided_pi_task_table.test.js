/* The project task table: every conversation with its state and times,
   filtered in place, sorted by activity or creation, opening through the host. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const jsRoot = path.dirname(path.resolve(process.argv[2]));
global.window = global;
require(path.join(jsRoot, 'screens-guided-pi-modules.js'));
require(path.join(jsRoot, 'screens-guided-pi-task-table.js'));
const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const table = global.EasyICU.guidedPi.require('taskTable').create({ tr: en => en, esc, iconHtml: name => `[${name}]` });

const sessions = [
  { session_id: 'a', title: 'Older plan', created_at: '2026-10-01T08:00:00Z', last_activity_at: '2026-10-27T08:00:00Z', key: 'needs_user' },
  { session_id: 'b', title: 'Newest <run>', created_at: '2026-10-20T08:00:00Z', last_activity_at: '2026-10-28T08:00:00Z', key: 'running' },
  { session_id: 'c', title: 'Draft', created_at: '2026-10-25T08:00:00Z', key: 'idle' },
];
const calls = [];
const ctx = { sessions, title: row => row.title, status: row => row.key, railKey: row => row.key, locked: false,
  open: id => calls.push(`open:${id}`), create: () => calls.push('new'), close: () => calls.push('close') };
const html = table.panel(ctx);
assert.match(html, /<h2>Conversations<span>3<\/span><\/h2>/);
assert.match(html, /Newest &lt;run>/);
const order = ['b', 'a', 'c'].map(id => html.indexOf(`data-gpi-task-row="${id}"`));
assert.deepEqual([...order].sort((x, y) => x - y), order, 'latest activity first');
assert.match(html, /class="gpi-task-state is-needs_user"/);
assert.match(html, /<th aria-sort="descending"><button type="button" data-gpi-task-sort="activity">/);
assert.match(html, /<p class="gpi-task-empty" data-gpi-task-empty hidden>/);
for (const [key] of [['all'], ['needs_user'], ['running'], ['stopped'], ['done'], ['idle']]) assert.match(html, new RegExp(`<option value="${key}"`));

// Delegated handlers on the host: open, new, close, sort, search and state filter.
const listeners = {};
const host = { addEventListener: (type, fn) => { listeners[type] = fn; } };
table.bind(host);
table.bind(host);
const rows = sessions.map(row => ({ dataset: { gpiTaskRow: row.session_id }, hidden: false }));
const empty = { hidden: true };
const body = { innerHTML: '' };
const root = { querySelectorAll: () => rows, querySelector: query => (query === '[data-gpi-task-empty]' ? empty : body) };
const click = (attr, value) => ({ target: { closest: query => {
  if (query === '[data-gpi-task-table]') return root;
  return query === `[${attr}]` ? { dataset: value } : null;
} } });
listeners.click(click('data-gpi-task-open', { gpiTaskOpen: 'a' }));
listeners.click(click('data-gpi-task-new', {}));
listeners.click(click('data-gpi-task-close', {}));
assert.deepEqual(calls, ['open:a', 'new', 'close']);
listeners.click(click('data-gpi-task-sort', { gpiTaskSort: 'created' }));
assert.ok(body.innerHTML.indexOf('data-gpi-task-row="c"') < body.innerHTML.indexOf('data-gpi-task-row="a"'), 'creation order on request');
assert.match(body.innerHTML, /<th aria-sort="descending"><button type="button" data-gpi-task-sort="created">/);
const field = (attr, value) => ({ target: { value, matches: query => query === `[${attr}]`, closest: () => root } });
listeners.input(field('data-gpi-task-search', 'PLAN'));
assert.deepEqual(rows.map(row => row.hidden), [false, true, true]);
listeners.input(field('data-gpi-task-search', ''));
listeners.change(field('data-gpi-task-filter', 'running'));
assert.deepEqual(rows.map(row => row.hidden), [true, false, true]);
listeners.change(field('data-gpi-task-filter', 'stopped'));
assert.equal(empty.hidden, false, 'an empty filter says so');
listeners.change(field('data-gpi-task-filter', 'bogus'));
assert.deepEqual(rows.map(row => row.hidden), [false, false, false]);

// A busy host blocks opening and starting, never closing.
calls.length = 0;
table.panel({ ...ctx, locked: true });
listeners.click(click('data-gpi-task-open', { gpiTaskOpen: 'a' }));
listeners.click(click('data-gpi-task-new', {}));
listeners.click(click('data-gpi-task-close', {}));
assert.deepEqual(calls, ['close']);
assert.match(table.panel({ ...ctx, locked: true }), /data-gpi-task-new disabled/);
