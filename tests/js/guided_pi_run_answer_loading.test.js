'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const digest = 'a'.repeat(64);
const run = { run_id: 'synthetic-run', present: true, analysis_results_available: true,
  artifact_refs: ['result_tables.json', 'agent_plan.json'].map(artifact => ({
    artifact, run_id: 'synthetic-run', sha256: digest,
  })) };
// The host's compact, verified aggregate survives a truncated display preview.
// Values and group names are synthetic; no run copy is used in these tests.
const plan = { analysis_type: 'descriptive_epidemiology',
  display_labels: { duration: '住院时长', unit_type: 'ICU 类型' },
  steps: [{ step_id: 'grouped_summary', planned_analysis_role: 'primary' }] };
const tables = { tables: [{ name: 'summary.csv', produced_by_step: 'grouped_summary',
  evidence_id: 'synthetic_summary', sha256: digest, source_status: 'current',
  preview_truncated: true, headers: [], rows: [],
  descriptive_summary: { schema_version: 'easyicu.web-descriptive-summary/1', categorical: [],
    continuous: ['Medical', 'Surgical', 'Mixed'].map((group, i) => ({
      variable: 'duration', group_variable: 'unit_type', group, unit: 'days',
      n_nonmissing: 10 + i, median: 2 + i, q25: 1 + i, q75: 3 + i,
    })) },
  }] };

function harness(read, options = {}) {
  const calls = [], waits = [];
  let project = 'synthetic-project';
  const window = { EU_LANG: 'zh' };
  const context = vm.createContext({ window, setTimeout: (resolve, ms) => {
    waits.push(ms);
    if (options.sleep) options.sleep(resolve, ms);
    else queueMicrotask(resolve);
  } });
  for (const name of ['screens-guided-pi-modules.js', 'screens-guided-pi-result-summary.js',
    'screens-guided-pi-run-answer.js', 'screens-guided-pi-run-outcome.js', 'screens-guided-pi-events.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, name), 'utf8'), context, { filename: name });
  }
  const api = { loadPiCopilotResearchArtifact: async (...args) => {
    calls.push(args);
    const response = await read?.(...args);
    return response === undefined ? { ok: true, payload: args[2] === 'agent_plan.json' ? plan : tables } : response;
  } };
  const deps = { tr: (en, zh) => zh, esc: value => String(value ?? '').replace(/</g, '&lt;'),
    api: () => api, projectId: () => project, resourceButton: () => '', iconHtml: () => '' };
  const modules = window.EasyICU.guidedPi;
  const owner = modules.require('runAnswer').create(deps);
  return { owner, calls, waits, deps, modules, setProject: value => { project = value; } };
}
const failure = (status, code) => Object.assign(new Error(code || `HTTP ${status}`), { status, code });
const resultCalls = calls => calls.filter(call => call[2] === 'result_tables.json');
const answer = html => {
  assert.match(html, /中位数［四分位距］/);
  for (const text of ['Medical 2.00［1.00–3.00］，n=10', 'Surgical 3.00［2.00–4.00］，n=11',
    'Mixed 4.00［3.00–5.00］，n=12']) assert.ok(html.includes(text), html);
  assert.doesNotMatch(html, /暂未生成|is-unavailable|is-failed/);
};

for (const transient of [() => failure(503), () => new TypeError('Failed to fetch')]) {
  test(`a transient ${transient().message} retries and answers continuous groups`, async () => {
    let attempts = 0;
    const { owner, calls, waits } = harness(async (_project, _run, artifact) => {
      if (artifact === 'result_tables.json' && ++attempts === 1) throw transient();
    });
    assert.equal(await owner.load(run), true);
    answer(owner.render(run));
    assert.equal(attempts, 2);
    assert.equal(waits.length, 1);
    assert.ok(waits[0] > 0 && waits[0] <= 1000);
    assert.ok(calls.every(call => call[0] === 'synthetic-project' && call[1] === run.run_id && call[3] === digest));
    assert.equal(calls.filter(call => call[2] === 'agent_plan.json').length, 1);
    assert.equal(await owner.load(run), false, 'a successful projection stays cached');
  });
}

test('exhausted 5xx reads stop, show a truthful error, and allow an explicit retry', async () => {
  let broken = true;
  const { owner, calls, waits } = harness(async (_project, _run, artifact) => {
    if (broken && artifact === 'result_tables.json') throw failure(503);
  });
  await owner.load(run);
  const html = owner.render(run);
  assert.match(html, /结果.*加载|加载.*结果/);
  assert.match(html, /data-gpi-run-answer-retry="synthetic-run"/);
  assert.doesNotMatch(html, /暂未生成|未生成结果摘要/);
  assert.equal(resultCalls(calls).length, 3);
  assert.equal(waits.length, 2);
  assert.ok(waits[1] > waits[0] && waits[1] <= 1500);
  assert.equal(await owner.load(run), false, 'workflow refresh does not loop failed reads');
  assert.equal(resultCalls(calls).length, 3);
  broken = false;
  await owner.retry(run);
  answer(owner.render(run));
  assert.equal(resultCalls(calls).length, 4);
});

for (const status of [400, 401, 403, 404, 409, 422]) {
  test(`HTTP ${status} is refused without automatic retries`, async () => {
    const { owner, calls, waits } = harness(async (_project, _run, artifact) => {
      if (artifact === 'result_tables.json') throw failure(status, status === 409 ? 'pi_run_artifact_sha256_mismatch' : undefined);
    });
    await owner.load(run);
    assert.match(owner.render(run), /data-gpi-run-answer-retry/);
    assert.doesNotMatch(owner.render(run), /Medical|暂未生成/);
    assert.equal(resultCalls(calls).length, 1);
    assert.equal(waits.length, 0);
    assert.equal(await owner.load(run), false);
  });
}

test('an explicit retry cannot bypass an unchanged sha mismatch', async () => {
  const { owner, calls, waits } = harness(async () => { throw failure(409, 'pi_run_artifact_sha256_mismatch'); });
  await owner.load(run);
  await owner.retry(run);
  assert.equal(resultCalls(calls).length, 2);
  assert.ok(calls.every(call => call[3] === digest));
  assert.equal(waits.length, 0);
  assert.doesNotMatch(owner.render(run), /Medical|暂未生成/);
});

test('a project change during backoff cancels the stale read', async () => {
  let release;
  const { owner, calls, setProject } = harness(async (_project, _run, artifact) => {
    if (artifact === 'result_tables.json') throw failure(503);
  }, { sleep: resolve => { release = resolve; } });
  const pending = owner.load(run);
  for (let i = 0; i < 8 && !release; i++) await Promise.resolve();
  assert.equal(typeof release, 'function');
  setProject('another-project');
  release();
  assert.equal(await pending, false);
  assert.equal(resultCalls(calls).length, 1);
  assert.equal(owner.render(run), '');
});

test('an artifact ref from another run never initiates a read', async () => {
  const { owner, calls } = harness();
  const foreign = { ...run, artifact_refs: run.artifact_refs.map(ref => ({ ...ref, run_id: 'foreign-run' })) };
  assert.equal(await owner.load(foreign), false);
  assert.equal(calls.length, 0);
});

test('the retry control reaches the answer owner and repaints the completed run', async () => {
  let broken = true;
  const h = harness(async (_project, _run, artifact) => {
    if (broken && artifact === 'result_tables.json') throw failure(503);
  });
  let click;
  const card = { dataset: { gpiRunOutcome: run.run_id }, outerHTML: '' };
  const host = { addEventListener: (name, handler) => { if (name === 'click') click = handler; },
    querySelectorAll: () => [card] };
  const workflow = { stages: [{ id: 'analysis', status: 'complete' }] };
  const outcome = h.modules.require('runOutcome').create({ ...h.deps, host: () => host });
  await outcome.loadScientificReview(run, workflow);
  for (let i = 0; i < 30; i++) await Promise.resolve();
  assert.match(card.outerHTML, /data-gpi-run-answer-retry/);
  h.modules.require('events').create({ state: { host, latestRun: run, workflow }, RUN_OUTCOME: outcome }).wire();
  broken = false;
  const control = { dataset: { gpiRunAnswerRetry: run.run_id } };
  click({ target: { closest: selector => selector === '[data-gpi-run-answer-retry]' ? control : null } });
  for (let i = 0; i < 30; i++) await Promise.resolve();
  answer(card.outerHTML);
});
