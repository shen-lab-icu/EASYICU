'use strict';

// The attention feed marks rows by its strongest item, counts open items in the
// tab title, acknowledges a conversation only while it is open and visible,
// and hands each new item once to a desktop bridge when one exists.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise(resolve => setImmediate(resolve)); };
const ITEMS = [
  { project_id: 'p1', conversation_id: 'c1', run_id: 'r1', kind: 'completed', reason_code: 'run_completed', sequence: 41, timestamp: '2026-10-26T12:00:00Z' },
  { project_id: 'p1', conversation_id: 'c1', run_id: 'r2', kind: 'needs_review', reason_code: 'human_plan_review_required', sequence: 42, timestamp: '2026-10-26T12:01:00Z' },
  { project_id: 'p2', conversation_id: 'c7', run_id: 'r3', kind: 'failed', reason_code: 'run_failed', sequence: 43, timestamp: '2026-10-26T12:02:00Z' },
];

function harness({ visible = true, bridge = null } = {}) {
  const calls = [];
  const declared = {};
  const store = new Map();
  const window = {
    EU_LANG: 'zh',
    crypto: { randomUUID: () => '0b1c2d3e-0000-4000-8000-000000000000' },
    localStorage: { getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, String(value)) },
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; }, optional: () => null } },
    EU_API: {
      loadAttention: async params => { calls.push(['feed', { ...params }]); return { items: params.after ? [] : ITEMS, next_after: 43, has_more: false, unacknowledged_count: 3 }; },
      acknowledgeAttention: async body => { calls.push(['ack', { ...body, sequences: [...body.sequences] }]); return { acknowledged: [...body.sequences] }; },
    },
    easyicuAttentionBridge: bridge,
  };
  const document = { title: 'EasyICU', visibilityState: visible ? 'visible' : 'hidden', addEventListener() {} };
  const intervals = [];
  const context = vm.createContext({ window, document, setInterval: fn => { intervals.push(fn); return 1; } });
  for (const file of ['html-escape.js', 'screens-guided-pi-attention.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  let selected = '';
  const changes = [];
  const attention = declared.attention;
  const bind = sessionId => {
    selected = sessionId;
    attention.bind({ projectId: () => 'p1', selectedId: () => selected, openSession: async id => calls.push(['open', id]), onChange: () => changes.push(document.title) });
  };
  return { attention, calls, document, store, bind, select: id => { selected = id; }, changes, intervals };
}

test('rows carry their strongest mark and the tab title counts open items', async () => {
  const { attention, calls, document, bind } = harness();
  bind('');
  await settle();
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0])), ['feed', { client_id: 'web-0b1c2d3e-0000-4000-8000-000000000000', after: 0, limit: 100 }]);
  assert.match(attention.conversationMarkHtml('p1', 'c1'), /class="gpi-attention-mark is-needs_review"[^>]*aria-label="需要你确认"/);
  assert.match(attention.projectMarkHtml('p2'), /is-failed[^>]*title="运行失败"/);
  assert.equal(attention.conversationMarkHtml('p1', 'c9'), '');
  assert.equal(document.title, '(3) EasyICU');
  assert.equal(calls.some(row => row[0] === 'ack'), false, 'reading the feed never acknowledges');
});

test('the open, visible conversation is acknowledged; a hidden window acknowledges nothing', async () => {
  const shown = harness();
  shown.bind('c1');
  await settle();
  assert.deepEqual(JSON.parse(JSON.stringify(shown.calls.filter(row => row[0] === 'ack'))),
    [['ack', { client_id: 'web-0b1c2d3e-0000-4000-8000-000000000000', sequences: [41, 42] }]]);
  assert.equal(shown.attention.conversationMarkHtml('p1', 'c1'), '');
  assert.notEqual(shown.attention.projectMarkHtml('p2'), '');
  assert.equal(shown.document.title, '(1) EasyICU');

  const hidden = harness({ visible: false });
  hidden.bind('c1');
  await settle();
  assert.equal(hidden.calls.some(row => row[0] === 'ack'), false);
});

test('a desktop bridge receives each new item once, then the count', async () => {
  const published = [];
  const badges = [];
  const bridge = {
    publish: async payload => { published.push(payload.item.sequence); return { delivered: true }; },
    badge: async payload => { badges.push(payload.unacknowledged_count); },
    onOpen: () => () => {},
  };
  const { attention, bind, store } = harness({ bridge });
  bind('');
  await settle();
  assert.deepEqual(published, [41, 42, 43]);
  assert.deepEqual(badges, [3]);
  assert.equal(store.get('easyicu.attention.published'), '43');
  await attention.poll();
  assert.deepEqual(published, [41, 42, 43]);
});
