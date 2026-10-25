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
for (const file of ['data-catalog.js', 'screens-guided-pi-modules.js', 'screens-guided-pi-resources.js',
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

test('continuous medians display compact quartiles, units and counts with the exact source', async () => {
  const { html, calls, owner, run } = await answer([continuous()]);
  assert.ok(html.includes('氯离子（mmol/L），中位数［四分位距］：甲组 104［101–108］，n=6；乙组 99［98–102］，n=5。'));
  assert.match(html, /不能说明因果/);
  assert.match(owner.limitation(run), /不能说明因果/);
  assert.doesNotMatch(html, /103\.125|名患者/);
  assert.deepEqual(summary([continuous()]).continuousGroups[0].statistics[0].values,
    ['104.250', '101.125', '107.875']);
  assert.ok(calls.some(r => r.evidence_id === source.evidence_id && r.evidence_sha256 === sha('a')));
});

test('means with standard deviations preserve zero and signed values in English', async () => {
  const table = { ...source, headers: ['variable', 'group', 'n_nonmissing', 'group_n', 'mean', 'sd', 'unit'],
    rows: [['balance', 'West', 4, 7, '0', '0', 'mL'], ['balance', 'East', 3, 8, '-125.50', '32.25', 'mL']] };
  const { html } = await answer([table], { lang: 'en', plan: { ...plan, display_labels: { balance: 'Fluid balance' } } });
  assert.ok(html.includes('Fluid balance (mL), mean [SD]: West 0 [0], n=4; East -126 [32], n=3.'));
  assert.doesNotMatch(html, /7 ICU stays|8 ICU stays/);
});

test('typed continuous group rows exclude the overall row and use nonmissing counts', async () => {
  const table = { ...source, headers: ['variable', 'group', 'row_role', 'group_n', 'n_nonmissing', 'median', 'q25', 'q75', 'unit'],
    rows: [['chloride', 'Overall', 'overall', 11, 9, 103, 99, 106, 'mmol/L'],
      ['chloride', 'North', 'group', 6, 5, 104, 101, 108, 'mmol/L'],
      ['chloride', 'South', 'group', 5, 4, 100, 98, 102, 'mmol/L']] };
  const { html } = await answer([table], { lang: 'en' });
  assert.match(html, /North 104 \[101–108\], n=5/);
  assert.match(html, /South 100 \[98–102\], n=4/);
  assert.doesNotMatch(html, /Overall|11 ICU stays/);
});

test('zero observations and NA never become a zero estimate; missing units are explicit', async () => {
  const table = continuous();
  table.rows = [['chloride', 'Empty', 0, 'NA', '', null, ''], ['chloride', 'Measured', 2, 0, 0, 0, '']];
  const { html } = await answer([table], { lang: 'en' });
  assert.match(html, /Empty.*unavailable.*n=0/);
  assert.match(html, /Measured 0\.00 \[0\.00–0\.00\], n=2/);
  assert.match(html, /unit unavailable/);
  assert.doesNotMatch(html, /Empty[^;<>]*median 0|NaN|undefined/);
});

test('known primary producer excludes auxiliary continuous tables', async () => {
  const secondary = { ...continuous(), evidence_id: 'auxiliary', produced_by_step: 'baseline' };
  secondary.rows = [['balance', 'Decoy', 99, 998, 997, 999, 'mL']];
  const { html } = await answer([secondary, continuous()], { plan: { ...plan, steps: [...plan.steps,
    { step_id: 'baseline', planned_analysis_role: 'auxiliary' }] } });
  assert.match(html, /104［101–108］/);
  assert.doesNotMatch(html, /998|Decoy/);
});

test('a typed continuous summary prefers registered medians and the nonmissing denominator', async () => {
  const table = { ...source, headers: ['variable', 'group', 'denominator_n', 'nonmissing_n', 'median', 'q25', 'q75', 'mean', 'sd', 'unit'],
    rows: [['temperature', 'Ward A', 9, 7, '37.1', '36.8', '37.4', '37.05', '0.25', '°C']] };
  for (const kind of ['mean_sd', 'median_iqr', 'both']) {
    const selectedPlan = { ...plan, display_labels: { temperature: 'Temperature' }, steps: [{ ...plan.steps[0],
      table_one_spec: { variables: [{ name: 'temperature', variable_kind: 'continuous', summary: kind }] } }] };
    const { html } = await answer([table], { plan: selectedPlan, lang: 'en' });
    assert.ok(html.includes('Temperature (°C), median [IQR]: Ward A 37.1 [36.8–37.4], n=7.'));
    assert.doesNotMatch(html, /mean|37\.05|0\.25/);
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
    assert.match(html, /104［101–108］/);
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

const compact = (continuous = [], categorical = []) => ({ ...source,
  headers: ['variable', 'statistic', 'value'], rows: [['chloride', 'median', '104.250']],
  preview_truncated: true, descriptive_summary: {
    schema_version: 'easyicu.web-descriptive-summary/1', continuous, categorical,
  } });
const compactRow = (group, extra = {}) => ({ variable: 'chloride', group, group_variable: 'ward',
  row_role: 'exposure_level', n_nonmissing: '6', unit: 'mmol/L',
  median: '104.250', q25: '101.125', q75: '107.875', ...extra });
const answerText = html => html.match(/<div class="gpi-run-answer-text"><p>(.*?)<\/p>/)?.[1] || '';

test('continuous display precision follows the variable location magnitude and leaves registered strings exact', async () => {
  for (const [value, expected] of [['2.5699999', '2.57'], ['87.04', '87.0'], ['1234.6', '1235']]) {
    const table = compact([compactRow('North', { median: value, q25: value, q75: value,
      mean: '999.99000', sd: '12.34000' })]);
    const original = JSON.stringify(table);
    const { html } = await answer([table]);
    assert.ok(answerText(html).includes(`North ${expected}［${expected}–${expected}］，n=6`));
    assert.doesNotMatch(answerText(html), /999\.99|均值|标准差/);
    assert.equal(summary([table]).continuousGroups[0].statistics[0].values[0], value);
    assert.equal(JSON.stringify(table), original);
  }
  const table = compact([compactRow('North', { median: '2.5699999', q25: '1.4', q75: '3.05' }),
    compactRow('South', { median: '-87.04', q25: '-90.25', q75: '-80.14' })]);
  assert.match(answerText((await answer([table])).html), /North 2\.6［1\.4–3\.0］.*South -87\.0［-90\.3–-80\.1］/);
  const mean = compact([compactRow('North', { median: '', q25: '', q75: '', mean: '87.04', sd: '2.5699999' })]);
  assert.match(answerText((await answer([mean])).html), /均值［标准差］：North 87\.0［2\.6］，n=6/);
});

test('descriptive labels use the catalog and bound aggregation/window, with language-specific fallbacks', async () => {
  const table = compact([compactRow('North', { variable: 'observed_peak', group_variable: 'unit_column',
    median: '2.5699999', q25: '1.4', q75: '3.05' })]);
  table.descriptive_summary.variable_display = { observed_peak: { source_concept: 'glu',
    description: 'source glucose description', unit_normalization: 'window_numeric_max',
    analysis_window: 'icu_admission[0,18]h' }, unit_column: { source_concept: 'icu_unit_type' } };
  const selectedPlan = { ...plan, display_labels: { observed_peak: 'English maximum glucose description' } };
  for (const [lang, expected] of [
    ['zh', '各 ICU 类型入 ICU 18 小时内最高血糖（mmol/L），中位数［四分位距］：North 2.57［1.40–3.05］，n=6。'],
    ['en', 'Maximum glucose in the first 18 h of ICU (mmol/L) by ICU Unit Type, median [IQR]: North 2.57 [1.40–3.05], n=6.'],
  ]) assert.equal(answerText((await answer([table], { lang, plan: selectedPlan })).html), expected);

  for (const [variable, labels, lang, expected] of [
    ['glu', { glu: 'English glucose description' }, 'zh', '血糖'],
    ['mystery', { mystery: 'Only English description' }, 'zh', 'mystery'],
    ['mystery', { mystery: 'Only English description' }, 'en', 'Only English description'],
    ['mystery', { mystery: '已注册名称' }, 'zh', '已注册名称'],
    ['mystery', {}, 'en', 'mystery'],
  ]) {
    const unknown = compact([compactRow('North', { variable, group_variable: '' })]);
    const text = answerText((await answer([unknown], { lang, plan: { ...plan, display_labels: labels } })).html);
    assert.ok(text.startsWith(expected), text);
  }
  const unknown = compact([compactRow('North', { variable: 'mystery', group_variable: '' })]);
  unknown.descriptive_summary.variable_display = { mystery: { description: 'Bound English description' } };
  assert.ok(answerText((await answer([unknown], { lang: 'en' })).html).startsWith('Bound English description'));
  assert.ok(answerText((await answer([unknown])).html).startsWith('mystery'));
});

test('overall rows are omitted per variable when named groups exist and retained when alone', async () => {
  const overall = compactRow('Overall', { n_nonmissing: '1000', median: '999', q25: '998', q75: '1001' });
  const table = compact([overall, compactRow('South'), compactRow('North'),
    compactRow('Overall', { variable: 'balance', row_role: 'overall', group_variable: '' })]);
  const text = answerText((await answer([table])).html);
  assert.doesNotMatch(text, /Overall|999|n=1,000/);
  assert.match(text, /South 104［101–108］，n=6；North 104［101–108］，n=6/);
  assert.match((await answer([table])).html, /液体平衡（mmol\/L），中位数［四分位距］：104［101–108］，n=6。/);
  const only = answerText((await answer([compact([overall])])).html);
  assert.match(only, /999［998–1001］，n=1,000/);
  assert.doesNotMatch(only, /Overall/);
});

test('bound reader labels preserve clinical acronyms and nonzero admission windows', async () => {
  for (const [concept, expected] of [['ph', 'blood pH'], ['pafi', 'PaO2/FiO2 Ratio']]) {
    const table = compact([compactRow('North', { variable: 'observed_peak', group_variable: '' })]);
    table.descriptive_summary.variable_display = { observed_peak: { source_concept: concept,
      unit_normalization: 'window_numeric_min', analysis_window: 'hospital_admission[6,18]h' } };
    assert.ok(answerText((await answer([table], { lang: 'en' })).html)
      .startsWith(`Minimum ${expected} 6–18 h after hospital admission`));
    assert.ok(answerText((await answer([table])).html).startsWith('入院 6–18 小时内最低'));
  }
});

test('unmeasured spread stays absent and rounded negative zero and mixed units stay readable', async () => {
  const table = compact([compactRow('North', { median: '-0.00001', q25: '', q75: '', unit: 'mg/L' }),
    compactRow('South', { median: '0', q25: '', q75: '', unit: 'mmol/L' })]);
  const text = answerText((await answer([table])).html);
  assert.match(text, /中位数：North 0\.00 mg\/L，n=6；South 0\.00 mmol\/L，n=6/);
  assert.doesNotMatch(text, /四分位距|-0\.00|标准差/);
});

test('an overall categorical fact displays a one-decimal percentage with its registered fraction', async () => {
  const rows = [{ variable: 'vital_status', group: 'Overall', row_role: 'overall', level: 'dead',
    numerator: '37', denominator: '301', percentage: '12.2923588039867' }];
  const selectedPlan = { ...plan, display_labels: { vital_status: '死亡', 'vital_status="dead"': '死亡' } };
  assert.equal(answerText((await answer([compact([], rows)], { plan: selectedPlan })).html), '死亡，比例：12.3%（37/301）。');
  const grouped = { ...rows[0], group: 'North', row_role: 'group', numerator: '1', denominator: '4', percentage: '25' };
  const text = answerText((await answer([compact([], [...rows, grouped])], { plan: selectedPlan })).html);
  assert.match(text, /North 25\.0%（1\/4）/);
  assert.doesNotMatch(text, /Overall|37\/301/);
});

test('more than twelve groups show the largest by observed n in registered order with a table tail', async () => {
  const rows = Array.from({ length: 15 }, (_, i) => compactRow(`Ward ${i}`, { n_nonmissing: String(i + 1) }));
  for (const lang of ['zh', 'en']) {
    const text = answerText((await answer([compact(rows)], { lang })).html);
    assert.doesNotMatch(text, /Ward [012](?!\d)/);
    assert.deepEqual([...text.matchAll(/Ward (\d+)/g)].map(match => Number(match[1])),
      Array.from({ length: 12 }, (_, i) => i + 3));
    assert.ok(text.endsWith(lang === 'zh' ? '另有 3 组见结果表。' : '3 more groups in the result table.'));
  }
  const tied = rows.map(row => ({ ...row, n_nonmissing: '5' }));
  const text = answerText((await answer([compact(tied)])).html);
  assert.deepEqual([...text.matchAll(/Ward (\d+)/g)].map(match => Number(match[1])),
    Array.from({ length: 12 }, (_, i) => i));
  const categories = rows.map(({ group, n_nonmissing }) => ({ variable: 'status', group,
    level: 'yes', numerator: '1', denominator: n_nonmissing, percentage: '12.345' }));
  const categoricalText = answerText((await answer([compact([], categories)])).html);
  assert.match(categoricalText, /Ward 3 12\.3%（1\/4）.*另有 3 组见结果表/);
  assert.deepEqual([...categoricalText.matchAll(/Ward (\d+)/g)].map(match => Number(match[1])),
    Array.from({ length: 12 }, (_, i) => i + 3));
  assert.doesNotMatch(answerText((await answer([compact(rows.slice(0, 12))])).html), /另有|more groups/);
});

test('a complete primary long-form projection answers every group in one sentence despite a truncated display preview', async () => {
  const rows = Array.from({ length: 8 }, (_, index) => compactRow(`Ward ${index}`));
  const secondary = { ...continuous(), produced_by_step: 'secondary', evidence_id: 'secondary' };
  const { html, calls } = await answer([secondary, compact(rows)], { plan: { ...plan, steps: [...plan.steps,
    { step_id: 'secondary', planned_analysis_role: 'secondary' }] } });
  const text = html.match(/<div class="gpi-run-answer-text"><p>(.*?)<\/p>/)?.[1] || '';
  for (let i = 0; i < 8; i++) assert.ok(text.includes(`Ward ${i}`));
  assert.equal((text.match(/。/g) || []).length, 1);
  assert.match(text, /mmol\/L.*Ward 0 104［101–108］，n=6/);
  assert.doesNotMatch(html, /暂未生成结果摘要|p 值|P =|显著|甲组/);
  assert.equal(calls.filter(row => row.evidence_id).length, 1);
  assert.equal(calls.find(row => row.evidence_id).evidence_id, source.evidence_id);
});

test('grouped medians only and means only can answer without inventing a spread', async () => {
  for (const [statistic, value, expected, absent] of [
    ['median', '102.5000', /中位数：North 103，n=3；South 103，n=3/, /四分位距|标准差|均值/],
    ['mean', '-0.12500', /均值：North -0\.13，n=3；South -0\.13，n=3/, /四分位距|标准差|中位数/],
  ]) {
    const rows = ['North', 'South'].map(group => ({ variable: 'chloride', group,
      row_role: 'exposure_level', n_nonmissing: '3', unit: 'mmol/L', [statistic]: value }));
    const { html } = await answer([compact(rows)]);
    assert.match(html, expected);
    assert.match(html, /North.*South/);
    assert.match(html, /n=3/);
    assert.doesNotMatch(html, absent);
    assert.doesNotMatch(html, /暂未生成结果摘要/);
    const wide = { ...source, headers: ['variable', 'group', 'n', statistic, 'unit'],
      rows: [['chloride', 'North', '3', value, 'mmol/L'], ['chloride', 'South', '3', value, 'mmol/L']] };
    assert.match((await answer([wide])).html, expected);
    if (statistic === 'mean') {
      const single = compact([{ ...rows[0], n_nonmissing: '1', sd: '' }]);
      const singleHtml = (await answer([single])).html;
      assert.match(singleHtml, /均值：North -0\.13，n=1/);
      assert.doesNotMatch(singleHtml, /标准差/);
    }
  }
});

test('categorical outcome proportions round percentages and retain observed numerators and denominators', async () => {
  const rows = [
    { variable: 'status', group: 'North', level: 'yes', numerator: '2', denominator: '7', percentage: '28.5714285714' },
    { variable: 'status', group: 'South', level: 'yes', numerator: '0', denominator: '5', percentage: '0.0000' },
  ].map(row => ({ ...row, row_role: 'exposure_level', unit: '' }));
  const { html, calls, owner, run } = await answer([compact([], rows)], {
    plan: { ...plan, display_labels: { status: '恢复状态', 'status="yes"': '已恢复' } },
  });
  assert.ok(html.includes('恢复状态（已恢复），比例：North 28.6%（2/7）；South 0.0%（0/5）。'));
  assert.equal(summary([compact([], rows)]).categoricalGroups[0].percentage, '28.5714285714');
  assert.doesNotMatch(html, /暂未生成结果摘要|风险比|OR |p 值|显著/);
  assert.match(owner.limitation(run), /不能说明因果/);
  assert.ok(calls.some(row => row.evidence_id === source.evidence_id));
});

test('an ungrouped distribution summary answers its registered overall statistics without a fabricated group', async () => {
  const { html } = await answer([compact([compactRow('', { row_role: 'overall', group_variable: '' })])]);
  assert.ok(html.includes('氯离子（mmol/L），中位数［四分位距］：104［101–108］，n=6。'));
  assert.doesNotMatch(html, /「」|Overall|暂未生成结果摘要|undefined/);
  // Support/frequency rows alone contain no registered location statistic.
  const { html: empty } = await answer([{ ...source, headers: ['variable', 'value', 'count', 'proportion'],
    rows: [['chloride', '102.5', '2', '0.25']] }]);
  assert.ok(empty.includes(fallback));
});

test('compact summaries keep primary ambiguity and source verification fail-closed', async () => {
  const main = compact([compactRow('North')]);
  for (const tables of [[main, { ...main, evidence_id: 'independent' }],
    [{ ...main, source_status: 'unavailable' }], [{ ...main, source_status: 'stale' }],
    [{ ...main, produced_by_step: 'unknown' }], [{ ...main, sha256: '' }],
    [{ ...main, is_source_copy: true }],
    [{ ...main, descriptive_summary: { ...main.descriptive_summary, schema_version: 'future/9' } }]]) {
    const { html } = await answer(tables);
    assert.ok(html.includes(fallback));
    assert.doesNotMatch(html, /104\.250/);
  }
  const blank = compact([compactRow('North', { median: '', q25: '', q75: '' })]);
  assert.ok((await answer([blank])).html.includes(fallback));
  const changed = structuredClone(main);
  changed.descriptive_summary.continuous[0].median = '103.125';
  assert.ok((await answer([main, changed])).html.includes(fallback));
  const oversized = compact(Array.from({ length: 65 }, (_, i) => compactRow(`Ward ${i}`)));
  assert.ok((await answer([oversized])).html.includes(fallback));
});

test('invalid compact values cannot become descriptive facts or bypass output escaping', async () => {
  for (const row of [compactRow('North', { median: 'NaN' }), compactRow('North', { n_nonmissing: '-1' }),
    compactRow('North', { q25: '110' }), compactRow('North', { q75: '' })]) {
    const { html } = await answer([compact([row])]);
    assert.ok(html.includes(fallback));
  }
  for (const row of [
    { variable: 'status', group: 'North', level: 'yes', numerator: '8', denominator: '7', percentage: '80' },
    { variable: 'status', group: 'North', level: 'yes', numerator: '1', denominator: '0', percentage: '100' },
    { variable: 'status', group: 'North', level: 'yes', numerator: '0', denominator: '0', percentage: '0' },
    { variable: 'status', group: 'North', level: 'yes', numerator: '1', denominator: '7', percentage: '101' },
  ]) {
    const { html } = await answer([compact([], [row])]);
    assert.ok(html.includes(fallback));
  }
  const category = { variable: 'status', group: 'North', level: 'yes', numerator: '1', denominator: '7', percentage: '14.25' };
  assert.ok((await answer([compact([], [category, category])])).html.includes(fallback));
  const { html } = await answer([compact([compactRow('<img src=x onerror=alert(1)>')])]);
  assert.match(html, /&lt;img/);
  assert.doesNotMatch(html, /<img src=x/);
});
