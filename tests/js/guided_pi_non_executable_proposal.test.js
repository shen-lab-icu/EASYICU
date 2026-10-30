'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
global.window = global;
global.EU_LANG = 'zh';
global.EU_HTML = { esc: v => String(v ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;') };
global.AGENT_RENDER = { artifactTitle: name => name, artifactSummary: () => '', runStatusLabel: value => value === 'blocked' ? '已阻断' : value };
require(path.join(root, 'screens-guided-pi-modules.js'));
require(path.join(root, 'screens-guided-pi-run-files.js'));
const modules = global.EasyICU.guidedPi;
const run = { run_id: 'run_1', run_ref: 'opaque_ref', study_id: 'study_1', run_status: 'blocked' };
const context = { projectId: 'project_1', studyId: 'study_1', sessionId: 'session_1', runId: '', busy: false };
const calls = [];
let response = { ok: true };
const api = {
  listStudyContexts: async () => ({ contexts: [{ id: 'study_1', title: 'Study' }] }),
  loadIdeaAgentProjects: async () => ({ projects: [] }),
  loadGuidedDrafts: async () => ({ drafts: [] }),
  loadAgentRunHistory: async () => ({ runs: [run], count: 1 }),
  loadAgentRunReview: async () => ({ run_ref: run.run_ref, run_id: run.run_id, study_id: run.study_id, artifacts: [], readiness: { signable: false } }),
};
global.fetch = async (...args) => { calls.push(args); return response; };
function owner() { return modules.require('runFiles').create({ api: () => api, context: () => context, changed() {}, resourceButton: () => '' }); }
function html(o) { return o.render({ savedRuns: [run] }); }
test('excerpt appears only for the selected bound run and explains its limits', async () => {
  calls.length = 0; response = { ok: true };
  const o = owner(); await o.sync(); assert.equal(calls.length, 0);
  await o.open(run.run_ref); assert.equal(calls.length, 1);
  assert.equal(calls[0][0], '/api/pi/proposal/project_1/session_1/run_1');
  assert.deepEqual(calls[0][1], { method: 'GET', credentials: 'same-origin', cache: 'no-store', redirect: 'error' });
  const result = html(o);
  assert.match(result, /查看待审研究配置摘录/);
  assert.match(result, /不是完整方案；来源和文献尚未逐项绑定，也不授权执行/);
  assert.match(result, /原运行：已阻断/);
  assert.doesNotMatch(result, /批准方案|批准执行/);
  o.reset();
});
test('an inaccessible or stale excerpt never hides the original blocked run', async () => {
  calls.length = 0; response = { ok: false, status: 409 };
  const o = owner(); await o.sync(); await o.open(run.run_ref);
  assert.doesNotMatch(html(o), /查看待审研究配置摘录/);
  assert.match(html(o), /原运行：已阻断/);
  response = { ok: true }; await o.open(run.run_ref);
  assert.match(html(o), /查看待审研究配置摘录/);
  context.sessionId = 'session_2'; await o.sync();
  assert.doesNotMatch(html(o), /查看待审研究配置摘录/);
  context.sessionId = 'session_1'; o.reset();
});
test('a response from an old session cannot publish a link into a new one', async () => {
  let finish; calls.length = 0;
  global.fetch = async (...args) => { calls.push(args); return new Promise(resolve => { finish = resolve; }); };
  const o = owner(); await o.sync(); const pending = o.open(run.run_ref);
  while (!finish) await new Promise(resolve => setImmediate(resolve));
  context.sessionId = 'session_2'; await o.sync(); finish({ ok: true }); await pending;
  assert.doesNotMatch(html(o), /查看待审研究配置摘录/);
  context.sessionId = 'session_1'; global.fetch = async (...args) => { calls.push(args); return response; }; o.reset();
});
