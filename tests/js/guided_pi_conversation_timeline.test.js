const assert = require('node:assert/strict');
const fs = require('node:fs');

const ownerPath = process.argv[2];
assert.ok(ownerPath, 'activity owner path is required');

let activityApi = null;
global.window = {
  EasyICU: {
    guidedPi: {
      declare(name, api) {
        assert.equal(name, 'activity');
        activityApi = api;
      },
    },
  },
};

eval(fs.readFileSync(ownerPath, 'utf8'));
assert.ok(activityApi, 'activity owner must register');

const esc = value => String(value == null ? '' : value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');
const activity = activityApi.create({
  tr: (_en, zh) => zh,
  esc,
  iconHtml: name => `<i>${esc(name)}</i>`,
  resourceName: resource => String(resource && resource.name || ''),
  resourceKey: resource => String(resource && resource.name || ''),
  resourceButton: resource => `<button>${esc(resource && resource.name || '')}</button>`,
});

const completed = {
  id: 'trace-1', role: 'activity', status: 'complete',
  startedAt: Date.parse('2026-09-19T00:00:00Z'),
  endedAt: Date.parse('2026-09-19T00:00:02Z'),
  steps: [{ id: 'tool-1', kind: 'tool', toolName: 'load_workflow', status: 'complete' }],
};
const rows = [
  { id: 'user-1', role: 'user', text: '问题' },
  completed,
  { id: 'assistant-1', role: 'assistant', text: '正式回答' },
];
const html = activity.renderTimeline(rows, row => {
  if (row.role === 'activity') return activity.render(row);
  return `<article data-role="${row.role}">${row.text}</article>`;
});

assert.ok(html.indexOf('data-role="user"') < html.indexOf('data-role="assistant"'));
assert.ok(html.indexOf('data-role="assistant"') < html.indexOf('gpi-turn-trace'));
assert.match(html, /查看执行过程/);
assert.doesNotMatch(html, /执行明细/);

const grouped = activity.renderTimeline([
  completed,
  { ...completed, id: 'trace-2', status: 'error' },
  { id: 'assistant-2', role: 'assistant', text: '带诊断的回答' },
], row => row.role === 'activity' ? activity.render(row) : `<article>${row.text}</article>`);
assert.ok(grouped.indexOf('带诊断的回答') < grouped.indexOf('gpi-turn-traces'));
assert.match(grouped, /查看执行过程 · 2 条记录 · 1 条未完成/);

const running = activity.renderTimeline([
  { ...completed, id: 'running', status: 'running' },
  { id: 'assistant-stream', role: 'assistant', text: '正在生成' },
], row => `<span data-row="${row.id}"></span>`);
assert.ok(running.indexOf('data-row="running"') < running.indexOf('data-row="assistant-stream"'));

console.log('guided conversation timeline contract passed');
