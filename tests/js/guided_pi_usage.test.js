'use strict';

// The model menu states what the host's accounting records: completed calls
// with reported tokens and cost; failed, in-progress and unknown calls apart;
// a null cost is never shown as free.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const empty = { calls: 0, provider_reported_calls: 0, provider_reported_tokens: 0, usage_unknown_calls: 0, estimated_cost_usd: 0 };

function owner(apiOverrides = {}) {
  const declared = {};
  const listeners = {};
  const window = {
    EU_LANG: 'zh',
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } },
    EU_API: apiOverrides,
  };
  const document = {
    addEventListener: (name, handler) => { listeners[name] = handler; },
    createElement: () => ({ className: '', innerHTML: '', attributes: {}, setAttribute(name, value) { this.attributes[name] = value; } }),
  };
  const context = vm.createContext({ window, document });
  for (const file of ['html-escape.js', 'screens-guided-pi-usage.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return { usage: declared.usage, listeners };
}

test('completed calls carry the tokens and cost; the rest are counted apart', () => {
  const { usage } = owner();
  const text = usage.summaryText({
    accounting_complete: true,
    by_state: {
      completed: { ...empty, calls: 3, provider_reported_calls: 2, provider_reported_tokens: 12480, usage_unknown_calls: 1, estimated_cost_usd: 0.0412 },
      failed: { ...empty, calls: 1 }, in_flight: { ...empty, calls: 2 }, unknown: { ...empty },
    },
  });
  assert.equal(text, '3 次调用 · 12,480 tokens · 约 $0.04；另有 1 次未报告用量、1 次失败、2 次进行中');
});

test('an unknown cost is left out, a tiny one is not rounded to zero', () => {
  const { usage } = owner();
  const unknown = usage.summaryText({ accounting_complete: true, by_state: { completed: { ...empty, calls: 1, provider_reported_calls: 1, provider_reported_tokens: 90, estimated_cost_usd: null } } });
  assert.equal(unknown, '1 次调用 · 90 tokens');
  const tiny = usage.summaryText({ accounting_complete: true, by_state: { completed: { ...empty, calls: 1, provider_reported_calls: 1, provider_reported_tokens: 90, estimated_cost_usd: 0.0004 } } });
  assert.equal(tiny, '1 次调用 · 90 tokens · 约 <$0.01');
});

test('incomplete accounting says so instead of showing totals', () => {
  const { usage } = owner();
  assert.equal(usage.summaryText({ accounting_complete: false, reason_code: 'conversation_usage_history_unavailable' }),
    '这段对话创建于旧版本，没有用量记录。');
  assert.equal(usage.summaryText({ accounting_complete: false }), '用量记录不完整。');
  assert.equal(usage.summaryText(null), '');
});

test('opening the model menu reads this conversation and its latest analysis', async () => {
  const calls = [];
  const { usage, listeners } = owner({
    loadPiSessionUsage: async (...args) => { calls.push(['session', ...args]); return { usage: { accounting_complete: true, by_state: { completed: { ...empty, calls: 2, provider_reported_calls: 2, provider_reported_tokens: 300, estimated_cost_usd: 0.5 } } } }; },
    loadPiRunUsage: async () => { throw Object.assign(new Error('bad'), { code: 'provider_usage_ledger_invalid' }); },
  });
  const appended = [];
  const popover = { querySelectorAll: () => [], appendChild: node => appended.push(node) };
  const control = { open: true, isConnected: true, querySelector: () => popover, closest: () => control };
  usage.bind({ session: () => ({ session_id: 's1', project_id: 'p1', binding: { run_id: 'r1' } }), projectId: () => 'p1' });
  listeners.toggle({ target: control });
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, [['session', 's1', 'p1']]);
  assert.equal(appended.length, 1);
  assert.match(appended[0].innerHTML, /<p><span>本对话<\/span>2 次调用 · 300 tokens · 约 \$0\.50<\/p>/);
  assert.match(appended[0].innerHTML, /<p><span>最近一次分析<\/span>分析的用量记录无法核对。<\/p>/);
});
