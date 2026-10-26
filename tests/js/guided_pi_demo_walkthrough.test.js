/* The workflow demo replays its registered run: it stops at every
   traceability card, waits for the reader's own plan confirmation, shows
   recorded times only, and says so when a reader skips the confirmation. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
require('./guided_pi_module_harness.cjs');

const repo = path.resolve(__dirname, '../..');
const trace = JSON.parse(fs.readFileSync(path.join(repo, 'src/easyicu/webserver/static/assets/demo/sofa2-run-trace.json'), 'utf8'));

// A replay advances only through timers and the trace request; both are
// driven by hand so each test reads the state at a stop.
let timers = [];
global.setTimeout = fn => { timers.push(fn); return timers.length; };
global.clearTimeout = id => { if (id) timers[id - 1] = null; };
let traceResponse = () => ({ ok: true, json: async () => trace });
global.fetch = async () => traceResponse();
const strip = { dataset: {}, innerHTML: '' };
const root = { querySelector: selector => (selector === '[data-gpi-walk-workflow]' ? strip : null), contains: () => true };
global.document = { querySelector: selector => (selector === '[data-gpi-walk-root]' ? root : null) };
global.window = { EU_LANG: 'zh' };
for (const file of process.argv.slice(2)) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
const walk = modules.require('demoWalkthrough');

let asideSyncs = 0;
const deps = {
  messageHtml: row => `<article data-row="${row.id}">${row.label || row.text || (row.steps || []).map(step => step.label).join('|')}</article>`,
  workflowHtml: () => '<nav></nav>',
  iconHtml: () => '',
  button: (resource, label) => `<button data-resource="${resource && resource.artifact}">${label || ''}</button>`,
  durationText: (start, end) => `${((end - start) / 1000).toFixed(1)} s`,
  syncAside: () => { asideSyncs += 1; },
};
async function settle() {
  for (let round = 0; round < 400; round += 1) {
    await new Promise(resolve => setImmediate(resolve));
    const next = timers.findIndex(Boolean);
    if (next < 0) return;
    const fn = timers[next];
    timers[next] = null;
    fn();
  }
  throw new Error('the replay did not settle');
}
function click(action, data = {}) {
  const control = { dataset: { gpiWalk: action, ...data } };
  return walk.handleClick({ target: { closest: selector => (selector === '[data-gpi-walk]' ? control : null) }, preventDefault() {} });
}
function html() { return walk.panelHtml(deps); }

test('the replay stops at each card and waits for the reader to confirm the plan', async () => {
  timers = [];
  walk.panelHtml(deps);
  walk.start();
  await settle();
  assert.equal(walk.state().waiting, 'dictionary');
  assert.match(html(), /问题里的每个词，用的是 MIMIC-IV 的哪些数据/);
  assert.match(html(), /admissions\.hospital_expire_flag · 时间 deathtime/);
  assert.match(html(), /chartevents · itemid 220052, 220181, 225312/);
  assert.match(html(), new RegExp(trace.sources.concept_dict.commit.slice(0, 9)));
  assert.match(html(), /2 \/ 6 · 数据映射/);

  click('play');
  await settle();
  assert.equal(walk.state().waiting, 'literature');
  assert.match(html(), /PMID 17938396/);
  assert.match(html(), /85\/100/);
  assert.match(html(), /未照搬：/);

  click('play');
  await settle();
  assert.equal(walk.state().waiting, 'confirm');
  assert.match(html(), /在这次真实运行里，研究者看了 329\.5 s 才确认/);
  assert.match(html(), /在上方确认方案后继续/);
  assert.doesNotMatch(html(), /data-gpi-walk="play"/);
  // Nothing moves on until the reader presses the control.
  await settle();
  assert.equal(walk.state().waiting, 'confirm');

  click('confirm');
  await settle();
  const state = walk.state();
  assert.equal(state.waiting, 'result');
  assert.equal(state.confirmed, true);
  assert.equal(state.skipped, false);
  assert.match(html(), /已确认方案：「确认方案并准备数据」/);
  assert.match(html(), /4,480 ÷ 33,997 = 13\.18%/);
  assert.match(html(), /table_step_artifact_257971d9f784c410/);
  assert.match(html(), /列 outcome_rate_pct，行 exposure = 1/);
  // The run's other figures sit under the main one, each its own redrawn table.
  assert.match(html(), /sofa2-cohort-flow-rerendered\.png/);
  assert.match(html(), /数据完整性（补充）/);

  click('select', { gpiWalkTarget: 'figure' });
  assert.match(html(), /这张图从哪来/);
  assert.match(html(), new RegExp(trace.evidence.records.figure_step_artifact_1c86f17d397e23b3.sha256.slice(0, 9)));
  assert.match(html(), /B 图的数据/);
  // The shown figure is the result table redrawn; the run's own figure is named beside it.
  assert.match(html(), /上面这张图由当前绘图程序根据下方结果表绘制/);
  assert.match(html(), new RegExp(trace.evidence.rerender.png_sha256.slice(0, 9)));
  assert.match(html(), /运行原图/);
  click('select', { gpiWalkTarget: 'cell:1:stays' });
  assert.match(html(), /94,458 个 ICU 入住中的 60,461 个/);

  click('play');
  await settle();
  assert.equal(walk.state().finished, true);
  assert.match(html(), /回放结束/);
  assert.match(html(), /data-gpi-demo-exit/);
  assert.match(html(), /这个结果能读到哪一步/);
  assert.match(html(), /descriptive_only/);
  assert.match(html(), /sofa2-study-report\.zh\.html/);
  assert.match(html(), /打开研究结果报告/);
  click('followup', { gpiWalkIndex: '0' });
  assert.match(html(), /连接模型后，就能在你自己的项目里接着问/);
  assert.ok(asideSyncs >= 5, `the right rail followed ${asideSyncs} chapter changes`);

  const done = walk.workflow();
  assert.deepEqual(done.stages.map(stage => stage.status), [...Array(7).fill('complete'), 'blocked']);
});

test('recorded times come from the audit log and nothing else', () => {
  const demo = modules.require('demo');
  const activities = demo.messages(trace).filter(row => row.role === 'activity');
  const planning = activities.find(row => row.id === 'reviewer-planning');
  const draft = planning.steps.find(step => step.id === 'p-draft');
  assert.equal(draft.endedAt - draft.startedAt, Date.parse(trace.spans['p-draft'][1]) - Date.parse(trace.spans['p-draft'][0]));
  assert.equal(planning.endedAt - planning.startedAt, Date.parse(trace.spans['p-review'][1]) - Date.parse(trace.spans['p-literature'][0]));
  // The pre-executed cohort step and the wrapper's report have no audit span.
  const execution = activities.find(row => row.id === 'reviewer-execution');
  assert.equal(execution.steps.find(step => step.id === 'e-1').durationKnown, false);
  // Without the trace record no activity claims a total.
  assert.ok(demo.messages(null).filter(row => row.role === 'activity').every(row => row.durationKnown === false));
  assert.deepEqual(demo.chapters(trace).map(chapter => chapter.id), ['question', 'data', 'plan', 'confirm', 'analysis', 'interpretation']);
  assert.deepEqual(demo.chapters(trace).flatMap(chapter => chapter.beats).filter(beat => beat.stop).map(beat => [beat.card, beat.stop]),
    [['dictionary', true], ['literature', true], ['confirm', 'confirm'], ['result', true], ['followups', 'end']]);
});

test('a skipped confirmation is shown as skipped, and the start state replays from the question', async () => {
  timers = [];
  walk.start();
  assert.deepEqual(walk.workflow().stages.slice(0, 2).map(stage => stage.status), ['running', 'pending']);
  await settle();
  click('skip');
  assert.equal(walk.state().finished, true);
  assert.equal(walk.state().skipped, true);
  assert.match(html(), /演示中跳过了确认/);
  assert.doesNotMatch(html(), /已确认方案：「确认方案并准备数据」/);
});

test('without the trace record the conversation still replays and claims no recorded time', async () => {
  timers = [];
  traceResponse = () => ({ ok: false });
  // A fresh page: the loaded record is cached per page load.
  global.window = { EU_LANG: 'en' };
  for (const file of process.argv.slice(2)) { delete require.cache[require.resolve(path.resolve(file))]; require(path.resolve(file)); }
  const fresh = window.EasyICU.guidedPi.require('demoWalkthrough');
  fresh.panelHtml(deps);
  fresh.start();
  await settle();
  assert.equal(fresh.state().status, 'failed');
  const control = action => ({ target: { closest: selector => (selector === '[data-gpi-walk]' ? { dataset: { gpiWalk: action } } : null) }, preventDefault() {} });
  for (let guard = 0; guard < 12 && !fresh.state().finished; guard += 1) {
    const waiting = fresh.state().waiting;
    fresh.handleClick(control(waiting === 'confirm' ? 'confirm' : 'play'));
    await settle();
  }
  assert.equal(fresh.state().finished, true);
  const page = fresh.panelHtml(deps);
  assert.doesNotMatch(page, /data-gpi-trace-card="dictionary"/);
  assert.match(page, /assets\/demo\/sofa2-phenotype-mortality-rerendered\.png/);
  assert.match(page, /assets\/demo\/sofa2-phenotype-mortality\.png/);
});
