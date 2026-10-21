/* A run's trace rows are the actions it took, each with its own time, and a
   plan step opens the code it ran and the files it made. Synthetic events
   and payloads only; no run directory is read. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
require(path.resolve('tests/js/guided_pi_module_harness.cjs'));

const owners = process.argv.slice(1).filter(arg => /screens-guided-pi-[a-z-]+\.js$/.test(arg)).map(file => fs.readFileSync(path.resolve(file), 'utf8'));
assert.equal(owners.length, 3, 'activity, step-detail and source-view owners');
const at = seconds => new Date(Date.UTC(2026, 0, 1) + seconds * 1000).toISOString();
const RUN = 'run_20260101T000000_synthetic';
const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const visible = html => html.replace(/<[^>]*>/g, ' ');
const operations = html => [...html.matchAll(/data-gpi-operation="([^"]*)"/g)]
  .map(match => JSON.parse(decodeURIComponent(match[1].replace(/&quot;/g, '"').replace(/&amp;/g, '&'))));
const event = (seconds, step, label, extra) => ({ type: 'progress', step, label, status: 'running',
  timestamp: at(seconds), run_id: RUN, ...extra });
const plan = (seconds, phase, current, extra) => event(seconds, 'planning', '', { action: 'plan_validation',
  planning_unit: 'structure', retry_phase: phase, current, total: 3, ...extra });
const PNG = 'data:image/png;base64,iVBORw0KGgo=';

const events = [
  event(0, 'runtime', '', { action: 'runtime_validated', status: 'complete' }),
  event(1, 'context', '', { action: 'context_built', status: 'complete', source_label: 'MIMIC-IV demo', source_database: 'miiv' }),
  event(2, 'context', '', { action: 'exposure_groups_staged', group_variable_count: 1 }),
  event(3, 'hypothesis', '', { action: 'literature_build', query_count: 2, record_count: 3, queries: ['sepsis glucose mortality'] }),
  plan(10, 'started', 1), plan(14, 'rejected', 1), plan(15, 'started', 2), plan(19, 'accepted', 2),
  event(20, 'plan', '', { action: 'plan_ready', plan_step_count: 4, status: 'complete' }),
  event(21, 'step', 'Skipped step already completed by pre-execution: cohort_accounting.', { step_id: 'cohort_accounting', status: 'complete' }),
  event(22, 'step', 'Step 2/4 started: baseline_table.', { step_id: 'baseline_table', current_step: 2, total_steps: 4 }),
  event(23, 'coder', '', { step_id: 'baseline_table', action: 'standard_executor', executor: 'grouped_table_one' }),
  event(24, 'runner', '', { step_id: 'baseline_table', action: 'run', run_kind: 'standard' }),
  event(30, 'step', 'Step 2/4 complete: baseline_table.', { step_id: 'baseline_table', status: 'complete', current_step: 2, total_steps: 4 }),
  event(31, 'step', 'Step 3/4 started: outcome_distribution.', { step_id: 'outcome_distribution', current_step: 3, total_steps: 4 }),
  event(32, 'coder', '', { step_id: 'outcome_distribution', action: 'code_generation' }),
  event(40, 'runner', '', { step_id: 'outcome_distribution', action: 'run', run_kind: 'generated' }),
  event(41, 'runner', '', { step_id: 'outcome_distribution', action: 'step_error', status: 'error' }),
  event(42, 'coder', '', { step_id: 'outcome_distribution', action: 'repair', repair_kind: 'runtime', attempt: 1 }),
  event(50, 'runner', '', { step_id: 'outcome_distribution', action: 'run', run_kind: 'generated', attempt: 1 }),
  event(55, 'step', 'Step 3/4 complete: outcome_distribution.', { step_id: 'outcome_distribution', status: 'complete', current_step: 3, total_steps: 4 }),
  event(56, 'step', 'Step 4/4 started: running_step.', { step_id: 'running_step', current_step: 4, total_steps: 4 }),
  event(57, 'coder', '', { step_id: 'running_step', action: 'code_generation' }),
];

const stepPayloads = {
  baseline_table: { schema_version: 'easyicu.web-run-step-detail/1', step_id: 'baseline_table', state: 'complete',
    execution: { mode: 'standard', executor: 'grouped_table_one' },
    code: { display_name: 'analysis.py', language: 'python', text: 'import pandas as pd\nvalue = 42\nprint(value)\n', line_count: 3, truncated: false },
    log: { shown: false, reason: 'runner_output_stays_local' },
    outputs: [
      { name: 'figure.png', kind: 'figure', bytes: 2048, thumbnail: PNG },
      { name: 'figure.svg', kind: 'figure', bytes: 4096, thumbnail: 'data:image/svg+xml;base64,PHN2Zz4=' },
      { name: 'step_summary.json', kind: 'summary', bytes: 300 },
      { name: 'table_one.csv', kind: 'table', bytes: 900, evidence: { id: 'table_baseline', sha256: 'a'.repeat(64) } },
      { name: '../escape.csv', kind: 'table', bytes: 1 },
    ],
    withheld_output_count: 1, more_output_count: 0,
    headline: { en: 'Baseline table built for 3 groups.', zh: '已生成 3 组的基线特征表。' } },
  running_step: { schema_version: 'easyicu.web-run-step-detail/1', step_id: 'running_step', state: 'running',
    execution: { mode: 'generated', executor: '' },
    code: { display_name: 'analysis.py', language: 'python', text: 'rows = 1\n', line_count: 1, truncated: false },
    log: { shown: false, reason: 'runner_output_stays_local' }, outputs: [], withheld_output_count: 0, more_output_count: 0 },
  mismatched: { step_id: 'another_step', state: 'complete', execution: {}, outputs: [] },
};

(async () => {
  for (const language of ['zh', 'en']) {
    const tr = (en, zh) => (language === 'zh' ? zh : en);
    const calls = { step: [], evidence: [], timers: 0 };
    let html = '';
    let listener = null;
    const host = {
      hidden: true,
      closest: () => ({ classList: { toggle() {} } }),
      addEventListener: (type, handler) => { if (type === 'click') listener = handler; },
      replaceChildren: () => { html = ''; },
      set innerHTML(value) { html = value; },
      get innerHTML() { return html; },
    };
    global.window = {
      EU_LANG: language, EU_HTML: { esc }, icon: name => `<i data-icon="${name}"></i>`,
      setTimeout: () => { calls.timers += 1; return calls.timers; }, clearTimeout() {},
      EU_API: {
        loadPiCopilotRunStep: async (...args) => { calls.step.push(args); return { ok: true, run_id: args[1], step: stepPayloads[args[2]] }; },
        loadPiCopilotResearchEvidence: async (...args) => { calls.evidence.push(args); return { ok: true, payload: { display_name: 'table_one.csv' } }; },
      },
    };
    owners.forEach(source => eval(source));
    window.EasyICU.guidedPi.declare('evidencePreview', { render: payload => `<div class="stub-evidence">${esc(payload.display_name)}</div>` });
    const activity = window.EasyICU.guidedPi.require('activity').create({
      tr, esc, iconHtml: name => `<i data-icon="${name}"></i>`, resourceName: () => '', resourceKey: () => '',
      resourceButton: () => '', runFailureText: () => '',
    });

    // (1) Rows are the run's actions, in order, with their own times.
    const turn = { status: 'running', steps: [] };
    events.forEach(row => activity.notePipelineEvent(turn, row));
    const rows = turn.steps;
    const plans = rows.filter(row => row.step === 'plan_step');
    assert.deepEqual(plans.map(row => row.stepId), ['cohort_accounting', 'baseline_table', 'outcome_distribution', 'running_step']);
    const baseline = plans[1];
    assert.equal(baseline.execution, 'standard');
    assert.equal(baseline.executor, 'grouped_table_one');
    assert.equal(baseline.runId, RUN);
    assert.equal(baseline.endedAt - baseline.startedAt, 8000);
    assert.equal(plans[2].execution, 'generated');
    assert.equal(activity.stepLabel(baseline), tr('Build the baseline characteristics table by exposure group', '生成基线特征表（按暴露分组）'));
    assert.equal(activity.stepLabel(plans[2]), tr('Summarize the outcome distribution', '汇总结局分布'));
    const validations = rows.filter(row => row.action === 'plan_validation');
    assert.deepEqual(validations.map(row => [row.current, row.status, row.endedAt - row.startedAt]),
      [[1, 'error', 4000], [2, 'complete', 4000]]);
    const literature = rows.find(row => row.action === 'literature_build');
    assert.equal(literature.status, 'complete', 'the next event ends the literature span');
    assert.equal(literature.endedAt - literature.startedAt, 7000);
    const repairs = rows.filter(row => row.action === 'repair');
    assert.equal(repairs.length, 1);
    assert.equal(repairs[0].endedAt - repairs[0].startedAt, 8000, 'a repair runs until the re-run starts');
    assert.ok(rows.some(row => row.action === 'step_error' && row.status === 'error'));
    // Only the running step animates (the turn header keeps its own spinner).
    const rendered = activity.render(turn);
    const rowSpinners = [...rendered.matchAll(/class="gpi-activity-step-open"[\s\S]*?<\/button>/g)]
      .filter(match => match[0].includes('gpi-running-spinner'));
    assert.equal(rowSpinners.length, 1);
    assert.deepEqual(rows.filter(row => row.status === 'running'), [plans[3]]);

    activity.finishPipelineSteps(turn, Date.parse(at(60)), 'complete');
    turn.status = 'complete';
    const trace = activity.render(turn);
    const text = visible(trace);
    for (const label of [
      tr('Build the baseline characteristics table by exposure group', '生成基线特征表（按暴露分组）'),
      tr('Already completed during preparation; skipped', '已在准备阶段完成，跳过'),
      tr('The run failed; repairing the code (attempt 1)', '运行失败，修复代码（第 1 次）'),
      tr('This step hit an error', '这一步运行出错'),
      tr('Search the literature and build the hypotheses', '检索文献并构建研究假设'),
      tr('2 queries · 3 records', '2 条检索式 · 3 条记录'),
      tr('Form the exposure groups', '划分暴露分组'),
      tr('Analysis plan ready: 4 steps', '分析计划就绪：共 4 步'),
      tr('8.0 s', '8.0 秒'),
    ]) assert.ok(text.includes(label), `${language}: ${label}`);
    // Kind icons: a registered standard method versus code written for this study.
    assert.match(trace, /<i data-icon="layers"><\/i>[\s\S]{0,400}生成基线特征表|<i data-icon="layers"><\/i>[\s\S]{0,400}Build the baseline/);
    assert.match(trace, new RegExp(`<i data-icon="edit"></i>[\\s\\S]{0,400}${tr('Summarize the outcome distribution', '汇总结局分布')}`));
    // Step coordinates ride in the operation payload; the visible text names no step id or path.
    assert.doesNotMatch(text, /baseline_table|outcome_distribution|cohort_accounting|\/Users|\/private/);
    const ops = operations(trace);
    const baselineOp = ops.find(op => op.step === 'baseline_table');
    assert.equal(baselineOp.run, RUN);
    assert.equal(baselineOp.execution, 'standard');
    assert.equal(baselineOp.executorName, tr('baseline characteristics table by exposure group', '基线特征表（按暴露分组）'));
    assert.deepEqual([baselineOp.stepIndex, baselineOp.stepTotal], [2, 4]);
    assert.equal(ops.find(op => op.step === 'cohort_accounting').execution, 'host');
    const contextOp = ops.find(op => op.title === tr('Build the research context', '构建研究上下文'));
    assert.deepEqual(contextOp.facts, [{ label: tr('Data source', '数据源'), value: 'MIMIC-IV demo (miiv)' }]);
    const literatureOp = ops.find(op => op.title === tr('Search the literature and build the hypotheses', '检索文献并构建研究假设'));
    assert.deepEqual(literatureOp.facts.map(fact => fact.value), ['2', '3', 'sepsis glucose mortality']);
    // A finished run's audit rows carry the same typed facts without a label.
    const hydrated = { status: 'complete', steps: [] };
    events.map(row => (row.step === 'step' ? row : { ...row, label: undefined })).forEach(row => activity.notePipelineEvent(hydrated, row));
    const hydratedBaseline = hydrated.steps.find(row => row.stepId === 'baseline_table' && row.step === 'plan_step');
    assert.equal(hydratedBaseline.executor, 'grouped_table_one');
    assert.equal(hydratedBaseline.text, tr('Ran the standard executor script; no model call.', '运行标准执行脚本，不调用模型。'));

    // (2) The step opens its code and outputs from the step route.
    const sourceView = window.EasyICU.guidedPi.require('sourceView');
    sourceView.mount(host);
    const button = op => ({ dataset: { gpiOperation: encodeURIComponent(JSON.stringify(op)) } });
    assert.equal(sourceView.openOperation(button(baselineOp), 'project_1', '', () => {}), true);
    for (let i = 0; i < 4; i += 1) await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(calls.step[0], ['project_1', RUN, 'baseline_table']);
    assert.match(html, /<i>1<\/i>[\s\S]*<i>3<\/i>/, 'line numbers');
    assert.ok(html.includes('<b>PYTHON</b>'));
    assert.ok(html.includes('data-gpi-step-copy'));
    assert.ok(html.includes(`${tr('Standard method', '标准方法')}: ${tr('baseline characteristics table by exposure group', '基线特征表（按暴露分组）')}`));
    assert.ok(html.includes(tr('Step 2/4', '第 2/4 步')));
    assert.ok(html.includes(`<img src="${PNG}"`), 'a registered PNG renders inline');
    assert.doesNotMatch(html, /svg\+xml|escape\.csv/, 'only PNG data and valid names render');
    assert.ok(html.includes('data-gpi-step-evidence="3"'), 'the table opens in the evidence preview');
    assert.ok(html.includes(tr('Baseline table built for 3 groups.', '已生成 3 组的基线特征表。')));
    assert.ok(html.includes(tr('The run log stays in the run record and is not shown here.', '运行日志保存在运行记录中，不在这里显示。')));
    assert.ok(html.includes(tr('1 more file(s) are not listed', '另有 1 个文件未列出')));
    assert.doesNotMatch(html, /parquet|\/Users|\/private|run\.log/);
    listener({ target: { closest: selector => (selector === '[data-gpi-step-evidence]' ? { dataset: { gpiStepEvidence: '3' } } : null) } });
    for (let i = 0; i < 4; i += 1) await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(calls.evidence[0], ['project_1', RUN, 'table_baseline', 'a'.repeat(64)]);
    assert.ok(html.includes('<div class="stub-evidence">table_one.csv</div>'));

    // (4) A running step shows its code once written and no output yet.
    const runningOp = operations(activity.render({ status: 'running', steps: [plans[3]] })).find(op => op.step === 'running_step');
    sourceView.openOperation(button(runningOp), 'project_1', '', () => {});
    for (let i = 0; i < 4; i += 1) await new Promise(resolve => setImmediate(resolve));
    assert.ok(html.includes('rows = <span class="tok-number">1</span>'));
    assert.ok(html.includes(tr('The output appears when this step finishes.', '这一步完成后显示输出。')));
    assert.doesNotMatch(html, /gpi-step-files|<img/);
    assert.equal(calls.timers, 1, 'a running step is read again');

    // A payload for another step is refused, not shown.
    sourceView.openOperation(button({ ...baselineOp, id: 'other', step: 'mismatched' }), 'project_1', '', () => {});
    for (let i = 0; i < 4; i += 1) await new Promise(resolve => setImmediate(resolve));
    assert.ok(html.includes(tr('This step cannot be displayed safely.', '这一步无法安全展示。')));
    // A host action shows its typed inputs.
    sourceView.openOperation(button(literatureOp), 'project_1', '', () => {});
    assert.ok(html.includes(tr('Records found', '检索到的记录')) && html.includes('sepsis glucose mortality'));
    // An operation's two blocks read Input and Output (输入 / 输出).
    assert.ok(html.includes(`<strong>${tr('Input', '输入')}</strong></div>\n        <dl class="gpi-operation-facts">`));
    assert.ok(html.includes(`<strong>${tr('Output', '输出')}</strong></div>`));
    sourceView.close();
  }
  process.stdout.write('step detail contracts passed\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
