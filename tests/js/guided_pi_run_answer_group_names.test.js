/* Synthetic groups: every answer line reads the run's registered names. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const root = process.argv[2] || path.resolve(__dirname, '../..');
global.window = global;
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
for (const file of ['screens-guided-pi-modules.js', 'screens-agent-exposure-levels.js',
  'screens-guided-pi-result-summary.js', 'screens-guided-pi-run-answer.js']) {
  require(path.join(root, 'src/easyicu/webserver/static/js', file));
}
const modules = global.EasyICU.guidedPi;
const variable = 'synthetic_band';
const names = [['Below threshold', '阈值以下'], ['At threshold', '达到阈值'], ['Above threshold', '阈值以上']];
let serial = 0;

async function answer({ language, size = 2, binding = 'model', registry = true, labels = {}, omit = [] }) {
  global.EU_LANG = language;
  global.t = (en, zh) => language === 'zh' ? zh : en;
  const levels = Array.from({ length: size }, (_, i) => i + 1);
  const eventTotal = size * (size + 1) / 2;
  const plan = { display_labels: labels, steps: [{ step_id: 'primary', planned_analysis_role: 'primary',
    ...(binding !== 'distribution' ? { model_requirements: [{ analysis_role: 'primary', exposure_source: variable,
      exposure_levels: levels.map(String), exposure_reference_level: '1', outcome: 'synthetic_outcome' },
    { analysis_role: 'secondary', exposure_source: 'decoy_band' },
    ...(binding === 'ambiguous' ? [{ analysis_role: 'primary', exposure_source: 'other_band' }] : [])] }
      : { exposure_outcome_distribution_spec: { exposure: variable, exposure_levels: levels, outcome: 'synthetic_outcome' } }) },
  { step_id: 'sensitivity', planned_analysis_role: 'secondary', model_requirements: [
    { analysis_role: 'primary', exposure_source: 'sensitivity_band' },
  ] }] };
  const tables = [{ name: 'risk.csv', produced_by_step: 'risk_context',
    headers: ['row_role', 'exposure_level', 'n_rows', 'exposure_denominator', 'exposure_pct',
      'outcome_events', 'outcome_denominator', 'outcome_rate_pct'],
    rows: levels.map(level => ['exposure_level', `${level}.0`, 10, size * 10, 100 / size, level, 10, level * 10])
      .concat([['overall', '', size * 10, size * 10, 100, eventTotal, size * 10, eventTotal / (size * 10) * 100]]) },
  { name: 'estimate.csv', produced_by_step: 'primary', evidence_id: 'synthetic_estimate',
    sha256: 'b'.repeat(64), source_status: 'current',
    headers: ['fit_status', 'estimate', 'ci_low', 'ci_high', 'effect_scale', 'is_primary_contrast',
      'exposure', 'exposure_level', 'reference_level'],
    rows: [['fitted', 1.7, 1.1, 2.5, 'odds_ratio', true, variable, `${size}.0`, '1.0']] }];
  const context = { source: { reader_label: 'Synthetic ICU' }, ...(registry ? { exposure_groups: {
    status: 'shown', groupings: [{ variable, status: 'labelled', concept_label_en: 'Band', concept_label_zh: '区间',
      reference: 1, levels: levels.filter(level => !omit.includes(level)).map(level => ({ code: level,
        label: names[level - 1][1], reader_name: names[level - 1][0] })) }],
  } } : {}) };
  const payloads = { 'agent_plan.json': plan, 'result_tables.json': { tables }, 'run_context.json': context };
  const run_id = `synthetic_${++serial}`;
  const run = { run_id, artifact_refs: Object.keys(payloads).map(artifact => ({ artifact, run_id, sha256: 'a'.repeat(64) })) };
  const widget = modules.require('runAnswer').create({ tr: global.t, esc: global.EU_HTML.esc,
    projectId: () => 'synthetic', api: () => ({ loadPiCopilotResearchArtifact: async (_project, _run, artifact) => ({ ok: true, payload: payloads[artifact] }) }) });
  await widget.load(run);
  return { html: widget.render(run), summary: modules.require('resultSummary').summarize({ tables }, plan, context) };
}

test('unlabelled text codes keep their variable in distributions', () => {
  const plan = { steps: [{ planned_analysis_role: 'primary', exposure_outcome_distribution_spec: {
    exposure: variable, exposure_levels: ['arm_a', 'arm_b'], outcome: 'synthetic_outcome',
  } }] };
  const summary = modules.require('resultSummary').summarize({ tables: [{
    headers: ['row_role', 'exposure_level', 'n_rows', 'exposure_denominator', 'exposure_pct',
      'outcome_events', 'outcome_denominator', 'outcome_rate_pct'],
    rows: [['exposure_level', 'arm_b', 10, 20, 50, 2, 10, 20]],
  }] }, plan, {});
  assert.equal(summary.exposureLevels[0].label, `${variable}=arm_b`);
});

for (const language of ['zh', 'en']) {
  for (const size of [2, 3]) {
    test(`${language}: ${size} model-bound groups use the shared registered names`, async () => {
      const { html, summary } = await answer({ language, size, labels: { [`${variable}=2`]: 'Stale manuscript label' } });
      const expected = names.slice(0, size).map(pair => pair[language === 'zh' ? 1 : 0]);
      assert.deepEqual(summary.exposureLevels.map(row => row.label), expected);
      for (const name of expected) assert.ok(html.includes(name), html);
      assert.ok(summary.estimates[0].label.includes(expected.at(-1)));
      assert.ok(summary.estimates[0].label.includes(expected[0]));
      assert.doesNotMatch(html, /Stale manuscript label|「[123](?:\.0)?」|“[123](?:\.0)?”/);
      if (size === 2) {
        assert.ok(html.indexOf(expected[1]) < html.indexOf(expected[0]), 'the existing focal-first order is preserved');
        assert.ok(html.includes(language === 'zh'
          ? '在 Synthetic ICU 的 20 个 ICU 入住记录中，达到阈值有 10 个（50.00%）。'
          : '10 (50.00%) were in the group “At threshold”'));
        assert.ok(html.includes(language === 'zh'
          ? '结局：达到阈值组 20.00%（2/10），阈值以下组 10.00%（1/10）；总体 15.00%。'
          : '20.00% (2/10) in “At threshold” vs 10.00% (1/10) in “Below threshold”'));
      } else {
        assert.ok(html.indexOf(expected[0]) < html.indexOf(expected[1]));
        assert.ok(html.indexOf(expected[1]) < html.indexOf(expected[2]));
      }
    });
  }
  test(`${language}: distribution specs still use the shared names`, async () => {
    const { summary } = await answer({ language, binding: 'distribution' });
    assert.deepEqual(summary.exposureLevels.map(row => row.label), names.slice(0, 2).map(pair => pair[language === 'zh' ? 1 : 0]));
  });
  test(`${language}: numeric display-label keys name string model levels`, async () => {
    const labels = Object.fromEntries(names.map((pair, i) => [`${variable}=${i + 1}`, pair[language === 'zh' ? 1 : 0]]));
    const { summary } = await answer({ language, registry: false, labels });
    assert.deepEqual(summary.exposureLevels.map(row => row.label), Object.values(labels).slice(0, 2));
    assert.ok(summary.estimates[0].label.includes(Object.values(labels)[1]));
    assert.ok(summary.estimates[0].label.includes(Object.values(labels)[0]));
  });
  test(`${language}: quoted display-label keys follow the plan level type`, async () => {
    const labels = Object.fromEntries(names.map((pair, i) => [`${variable}=${JSON.stringify(String(i + 1))}`, pair[language === 'zh' ? 1 : 0]]));
    const { summary } = await answer({ language, registry: false, labels });
    assert.deepEqual(summary.exposureLevels.map(row => row.label), Object.values(labels).slice(0, 2));
    assert.ok(summary.estimates[0].label.includes(Object.values(labels)[1]));
    assert.ok(summary.estimates[0].label.includes(Object.values(labels)[0]));
  });
  test(`${language}: ambiguous primary exposures do not borrow another groups name`, async () => {
    const { summary } = await answer({ language, binding: 'ambiguous' });
    assert.deepEqual(summary.exposureLevels.map(row => row.label), ['exposure=1', 'exposure=2']);
  });
  test(`${language}: absent or partial labels show variable=code`, async () => {
    for (const registry of [false, true]) {
      const { html, summary } = await answer({ language, registry, omit: [2] });
      assert.equal(summary.exposureLevels[1].label, `${variable}=2`);
      assert.ok(summary.estimates[0].label.includes(`${variable}=2`));
      assert.ok(html.includes(`${variable}=2`));
      assert.doesNotMatch(html, /「2(?:\.0)?」|“2(?:\.0)?”/);
    }
  });
  test(`${language}: blank display labels use the same qualified fallback`, async () => {
    for (const label of ['', '   ']) {
      const { summary } = await answer({ language, registry: false, labels: { [`${variable}=2`]: label } });
      assert.equal(summary.exposureLevels[1].label, `${variable}=2`);
      assert.ok(summary.estimates[0].label.includes(`${variable}=2`));
    }
  });
}
