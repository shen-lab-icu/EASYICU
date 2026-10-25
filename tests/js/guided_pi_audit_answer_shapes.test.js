/* Synthetic aggregate contracts for the completed conversation answer. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const root = process.argv[2] || path.resolve(__dirname, '../..');
global.window = global;
global.EU_LANG = 'zh';
const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
global.EU_HTML = { esc };
for (const file of ['screens-guided-pi-modules.js', 'screens-guided-pi-resources.js',
  'screens-guided-pi-result-summary.js', 'screens-guided-pi-run-answer.js', 'screens-guided-pi-run-outcome.js']) {
  require(path.join(root, 'src/easyicu/webserver/static/js', file));
}
const modules = global.EasyICU.guidedPi;
const sha = char => char.repeat(64);
const source = { evidence_id: 'summary_registered', sha256: sha('a'), source_status: 'current',
  produced_by_step: 'describe', name: 'registered.csv', label: 'Registered aggregate' };
const plan = { analysis_type: 'descriptive', display_labels: { chloride: '氯离子', balance: '液体平衡' },
  steps: [{ step_id: 'describe', planned_analysis_role: 'primary', method: 'descriptive_distribution' }] };
const continuous = () => ({ ...source, headers: ['variable', 'group', 'n', 'median', 'q25', 'q75', 'unit'],
  rows: [['chloride', '甲组', '6', '104.250', '101.125', '107.875', 'mmol/L'],
    ['chloride', '乙组', '5', '99.125', '97.500', '102.250', 'mmol/L']] });
const model = (extra = {}) => ({ ...source, headers: ['model_id', 'fit_status', 'estimate', 'ci_low', 'ci_high',
  'effect_scale', 'is_primary_contrast', 'analysis_role', 'contrast', 'n'],
  rows: [['model_one', 'fitted', '1.876', '1.201', '2.604', 'odds_ratio', true, 'primary', 'W vs V', 11]], ...extra });
let sequence = 0;
async function answer(tables, options = {}) {
  global.EU_LANG = options.lang || 'zh';
  const runId = `synthetic_${++sequence}`;
  const payloads = { 'agent_plan.json': options.plan || plan, 'result_tables.json': { tables, ...options.result },
    'run_context.json': {}, ...(options.provenance ? { 'manuscript_provenance.json': options.provenance } : {}) };
  const run = { run_id: runId, analysis_results_available: true, analysis_validated: true,
    artifact_refs: Object.keys(payloads).map((artifact, index) => ({ kind: 'research_artifact', run_id: runId,
      artifact, sha256: sha(String(index + 1)) })) };
  const calls = [];
  const resources = modules.require('resources').create({ esc });
  const owner = modules.require('runAnswer').create({ esc,
    tr: (en, zh) => global.EU_LANG === 'zh' ? zh : en, projectId: () => 'synthetic_project',
    resourceButton: (resource, label, opts) => { calls.push(resource); return resources.button(resource, label, opts); },
    api: () => ({ loadPiCopilotResearchArtifact: async (_project, _run, artifact) => {
      if (options.fail) throw new Error('Synthetic read failure');
      return { payload: payloads[artifact] };
    } }) });
  await owner.load(run);
  return { html: owner.render(run), calls, owner, run };
}
const fallback = '分析已完成，暂未生成结果摘要。请查看结果表。';
const summary = (tables, selectedPlan = plan) => modules.require('resultSummary').summarize({ tables }, selectedPlan, {});

test('continuous medians retain quartiles, units, analysed stays and the exact source without manuscript provenance', async () => {
  const { html, calls, owner, run } = await answer([continuous()]);
  assert.match(html, /氯离子.*甲组.*中位数 104\.250.*101\.125–107\.875.*mmol\/L.*6 个 ICU 入住记录/);
  assert.match(html, /乙组.*99\.125.*97\.500–102\.250.*5 个 ICU 入住记录/);
  assert.match(html, /不能说明因果/);
  assert.match(owner.limitation(run), /不能说明因果/);
  assert.doesNotMatch(html, /103\.125|名患者/);
  assert.ok(calls.some(r => r.evidence_id === source.evidence_id && r.evidence_sha256 === sha('a')));
});

test('means with standard deviations preserve zero and signed values in English', async () => {
  const table = { ...source, headers: ['variable', 'group', 'n_nonmissing', 'group_n', 'mean', 'sd', 'unit'],
    rows: [['balance', 'West', 4, 7, '0', '0', 'mL'], ['balance', 'East', 3, 8, '-125.50', '32.25', 'mL']] };
  const { html } = await answer([table], { lang: 'en', plan: { ...plan, display_labels: { balance: 'Fluid balance' } } });
  assert.match(html, /Fluid balance.*West.*mean 0.*SD 0.*mL.*4 ICU stays/);
  assert.match(html, /East.*mean -125\.50.*SD 32\.25.*3 ICU stays/);
  assert.doesNotMatch(html, /7 ICU stays|8 ICU stays/);
});

test('typed continuous group rows exclude the overall row and use nonmissing counts', async () => {
  const table = { ...source, headers: ['variable', 'group', 'row_role', 'group_n', 'n_nonmissing', 'median', 'q25', 'q75', 'unit'],
    rows: [['chloride', 'Overall', 'overall', 11, 9, 103, 99, 106, 'mmol/L'],
      ['chloride', 'North', 'group', 6, 5, 104, 101, 108, 'mmol/L'],
      ['chloride', 'South', 'group', 5, 4, 100, 98, 102, 'mmol/L']] };
  const { html } = await answer([table], { lang: 'en' });
  assert.match(html, /North.*5 ICU stays/);
  assert.match(html, /South.*4 ICU stays/);
  assert.doesNotMatch(html, /Overall|11 ICU stays/);
});

test('zero observations and NA never become a zero estimate; missing units are explicit', async () => {
  const table = continuous();
  table.rows = [['chloride', 'Empty', 0, 'NA', '', null, ''], ['chloride', 'Measured', 2, 0, 0, 0, '']];
  const { html } = await answer([table], { lang: 'en' });
  assert.match(html, /Empty.*unavailable.*0 ICU stays/);
  assert.match(html, /Measured.*median 0.*0–0/);
  assert.match(html, /unit unavailable/);
  assert.doesNotMatch(html, /Empty[^<]*median 0|NaN|undefined/);
});

test('known primary producer excludes auxiliary continuous tables', async () => {
  const secondary = { ...continuous(), evidence_id: 'auxiliary', produced_by_step: 'baseline' };
  secondary.rows = [['balance', 'Decoy', 99, 998, 997, 999, 'mL']];
  const { html } = await answer([secondary, continuous()], { plan: { ...plan, steps: [...plan.steps,
    { step_id: 'baseline', planned_analysis_role: 'auxiliary' }] } });
  assert.match(html, /104\.250/);
  assert.doesNotMatch(html, /998|Decoy/);
});

test('a typed continuous summary respects the registered statistic and analysed denominator', async () => {
  const table = { ...source, headers: ['variable', 'group', 'denominator_n', 'nonmissing_n', 'median', 'q25', 'q75', 'mean', 'sd', 'unit'],
    rows: [['temperature', 'Ward A', 9, 7, '37.1', '36.8', '37.4', '37.05', '0.25', '°C']] };
  for (const [kind, expected, absent] of [['mean_sd', /mean 37\.05.*SD 0\.25/, /median/],
    ['median_iqr', /median 37\.1.*36\.8–37\.4/, /mean/], ['both', /median 37\.1.*mean 37\.05/, /nothing/]]) {
    const selectedPlan = { ...plan, display_labels: { temperature: 'Temperature' }, steps: [{ ...plan.steps[0],
      table_one_spec: { variables: [{ name: 'temperature', variable_kind: 'continuous', summary: kind }] } }] };
    const { html } = await answer([table], { plan: selectedPlan, lang: 'en' });
    assert.match(html, expected);
    assert.doesNotMatch(html, absent);
    assert.match(html, /°C.*7 ICU stays/);
    assert.doesNotMatch(html, /9 ICU stays/);
  }
});

test('a continuous primary needs a bound producer; copies alone never answer', async () => {
  for (const table of [{ ...continuous(), produced_by_step: '' }, { ...continuous(), is_source_copy: true }]) {
    const { html, calls } = await answer([table]);
    assert.ok(html.includes(fallback));
    assert.doesNotMatch(html, /104\.250/);
    assert.ok(calls.every(r => !r.evidence_id));
  }
});

test('source and publication copies do not become competing results, regardless of order', async () => {
  const main = continuous();
  const copies = [{ ...main, evidence_id: 'copy_one', is_source_copy: true },
    { ...main, evidence_id: 'publication_figure_source_two', name: 'publication_figure_source_two.csv' }];
  for (const tables of [[...copies, main], [main, ...copies]]) {
    const { html, calls } = await answer(tables);
    assert.match(html, /104\.250/);
    assert.equal(calls.filter(r => r.evidence_id).length, 1);
    assert.equal(calls.find(r => r.evidence_id).evidence_id, main.evidence_id);
  }
});

test('repeated same evidence identity is one producer, never value-based deduplication', () => {
  const main = model();
  assert.equal(summary([main, structuredClone(main)]).estimates.length, 1);
  assert.equal(summary([main, { ...main, evidence_id: 'independent' }]).estimates.length, 0);
  const changed = structuredClone(main); changed.rows[0][2] = '2.9';
  assert.equal(summary([main, changed]).estimates.length, 0);
});

test('a known sensitivity producer cannot supersede the primary producer', () => {
  const primary = model();
  const sensitivity = model({ evidence_id: 'sensitivity_table', produced_by_step: 'sensitivity' });
  const result = summary([sensitivity, primary], { ...plan, steps: [...plan.steps,
    { step_id: 'sensitivity', planned_analysis_role: 'sensitivity' }] });
  assert.equal(result.estimates.length, 1);
  assert.equal(result.estimates[0].source.evidence_id, primary.evidence_id);
});

test('conflicting primary models and multiple primary contrasts never choose the first estimate', async () => {
  const main = model();
  const extra = structuredClone(main.rows[0]); extra[2] = '8.5';
  for (const tables of [[main, model({ evidence_id: 'another_primary' })], [{ ...main, rows: [...main.rows, extra] }],
    [{ ...main, row_model_ids: ['one', 'two'], headers: main.headers.slice(1), rows: [main.rows[0].slice(1), extra.slice(1)] }]]) {
    assert.equal(summary(tables).primaryModelAmbiguous, true);
    assert.equal(summary(tables).estimates.length, 0);
    const { html, calls } = await answer(tables, { plan: { ...plan, analysis_type: 'association_study' },
      provenance: { claims: [{ source_field: 'primary_or', display_value: '12.345' }] } });
    assert.match(html, /多个主要模型|暂未生成结果摘要/);
    assert.doesNotMatch(html, /OR 1\.88|OR 8\.50|12\.345/);
    assert.ok(calls.some(r => r.evidence_id === main.evidence_id));
  }
});

test('association ambiguity cannot suppress a separately registered prediction answer', async () => {
  const { html } = await answer([model(), model({ evidence_id: 'other_model' })], {
    plan: { ...plan, analysis_type: 'prediction_model' }, result: { prediction_performance: {
      schema_version: 'easyicu.web-prediction-performance/1', product: 'table:model_performance',
      evidence_id: 'prediction_performance', authority_scope: 'analysis_only',
      auroc: 0.745, auroc_ci_low: 0.701, auroc_ci_high: 0.789, brier_score: 0.125,
      validation_n: 12, validation_subject_n: 12, validation_event_n: 4,
    } },
  });
  assert.match(html, /AUROC 0\.745/);
  assert.doesNotMatch(html, /多个主要模型|OR 1\.88/);
});

test('two continuous primary tables get a bounded fallback with both exact table actions', async () => {
  const { html, calls } = await answer([continuous(), { ...continuous(), evidence_id: 'second_primary', sha256: sha('b'),
    rows: [['balance', 'Third group', 4, 25, 20, 30, 'mL']] }]);
  assert.ok(html.includes(fallback));
  assert.doesNotMatch(html, /104\.250/);
  assert.deepEqual(calls.filter(r => r.evidence_id).map(r => r.evidence_id).sort(), ['second_primary', source.evidence_id].sort());
});

test('truncated or invalid continuous summaries fall back to their exact table', async () => {
  for (const change of [{ preview_truncated: true }, { rows: [['chloride', 'X', 2, 4, 5, 6, 'mmol/L']] },
    { rows: [['chloride', 'X', -1, 4, 3, 5, 'mmol/L']] },
    { rows: [['chloride', 'X', 1.5, 4, 3, 5, 'mmol/L']] },
    { headers: ['variable', 'group', 'n', 'mean', 'sd', 'unit'], rows: [['balance', 'X', 2, 0, -1, 'mL']] }]) {
    const { html, calls } = await answer([{ ...continuous(), ...change }]);
    assert.ok(html.includes(fallback));
    assert.ok(calls.some(r => r.evidence_id === source.evidence_id));
  }
});

test('unavailable, stale, or absent table provenance never creates evidence links or a continuous answer', async () => {
  for (const change of [{ source_status: 'stale' }, { source_status: 'unavailable' }, { sha256: '' },
    { sha256: 'invalid' }, { evidence_id: '../private' }]) {
    const { html, calls } = await answer([{ ...continuous(), ...change }]);
    assert.ok(html.includes(fallback));
    assert.doesNotMatch(html, /104\.250|data-gpi-evidence-open/);
    assert.ok(calls.every(r => !r.evidence_id));
  }
});

test('missing results and failed reads show C01 in both languages without a scientific caveat', async () => {
  for (const options of [{}, { fail: true }, { lang: 'en' }]) {
    const { html, calls } = await answer([], options);
    assert.ok(html.includes(options.lang === 'en'
      ? 'Analysis complete. A result summary is unavailable; open the result table.' : fallback));
    assert.doesNotMatch(html, /未调整|without adjustment|可从下方打开/);
    assert.ok(calls.some(r => r.artifact === 'result_tables.json' && !r.evidence_id));
  }
});

test('the completed card also uses C01 before its answer can be loaded', () => {
  const latest = { run_id: 'synthetic', present: true, analysis_results_available: true, analysis_validated: true,
    artifact_refs: [{ artifact: 'result_tables.json', run_id: 'synthetic', kind: 'research_artifact', sha256: sha('a') }] };
  for (const lang of ['zh', 'en']) {
    global.EU_LANG = lang;
    const outcome = modules.require('runOutcome').create({ esc, tr: (en, zh) => lang === 'zh' ? zh : en,
      iconHtml: () => '', resourceButton: () => '', canPreview: () => true });
    const html = outcome.render(latest, { stages: [{ id: 'analysis', status: 'complete' }] });
    assert.ok(html.includes(lang === 'zh' ? fallback
      : 'Analysis complete. A result summary is unavailable; open the result table.'));
    assert.doesNotMatch(html, /可从下方打开|Open the study overview/);
  }
});

test('malicious labels stay escaped and a different run cannot reuse a cached answer', async () => {
  const table = continuous(); table.rows[0][1] = '<script>bad()</script>';
  const { html, owner, run } = await answer([table]);
  assert.match(html, /&lt;script>/);
  assert.doesNotMatch(html, /<script>/);
  assert.equal(owner.render({ ...run, run_id: 'another_run' }), '');
});
