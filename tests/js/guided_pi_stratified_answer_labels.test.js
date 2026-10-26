/* Synthetic aggregate results: strata and scales use the run's reader names. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const root = process.argv[2] || path.resolve(__dirname, '../..');
global.window = global;
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.EU_CATALOG = { dict: { marker: ['Marker', '标志物', 'g/dL'] } };
for (const file of ['screens-guided-pi-modules.js', 'screens-agent-exposure-levels.js',
  'screens-guided-pi-result-summary.js', 'screens-guided-pi-run-answer.js']) {
  require(path.join(root, 'src/easyicu/webserver/static/js', file));
}
const modules = global.EasyICU.guidedPi;
let serial = 0;

async function sentence({ language = 'zh', categorical = false, registered = true, registry = true,
  exposure = 'marker_min', modifier, displayLabel, prefix = true, unit = '1 g/dL', partial = false } = {}) {
  global.EU_LANG = language;
  const tr = global.t = (en, zh) => language === 'zh' ? zh : en;
  if (categorical) exposure = 'synthetic_band';
  const contrast = `${prefix ? `${exposure}: ` : ''}${categorical ? '2 vs 1' : `per ${unit} increase`}`;
  const display_labels = { ward: modifier || tr('ward', '病区'),
    ...(registered ? { [exposure]: displayLabel || tr('Minimum marker', '最低标志物') } : {}),
    ...(categorical && !registry && registered ? {
      [`${exposure}=1`]: tr('Below threshold', '阈值以下'), [`${exposure}=2`]: tr('Above threshold', '阈值以上'),
    } : {}),
  };
  const plan = { display_labels, steps: [
    { step_id: 'primary', planned_analysis_role: 'primary', model_requirements: [
      { analysis_role: 'primary', exposure_source: exposure, outcome: 'synthetic_event' },
    ] },
    { step_id: 'modifier', planned_analysis_role: 'secondary', scientific_action_id: 'association.effect_modification',
      inputs: [exposure, 'ward'], expected_outputs: ['table:synthetic_strata'] },
  ] };
  const table = { name: 'synthetic_strata.csv', produced_by_step: 'modifier', evidence_id: 'synthetic_strata',
    sha256: 'b'.repeat(64), source_status: 'current', headers: [], rows: [],
    stratified_summary: { schema_version: 'easyicu.web-stratified-summary/1', product: 'table:synthetic_strata',
      subgroup_column: 'ward', contrast, effect_scale: 'odds_ratio', interaction_p_value: 0.141,
      strata: [{ level: tr('North', '北区'), estimate: 0.79, ci_low: 0.69, ci_high: 0.92 },
        { level: tr('South', '南区'), estimate: 0.91, ci_low: 0.82, ci_high: 1.00 }] } };
  const context = categorical && registry ? { exposure_groups: { status: 'shown', groupings: [
    { variable: exposure, status: 'labelled', reference: 1, levels: [
      { code: 1, label: '阈值以下', reader_name: 'Below threshold' },
      ...(!partial ? [{ code: 2, label: '阈值以上', reader_name: 'Above threshold' }] : []),
    ] },
  ] } } : {};
  const primary = { name: 'primary.csv', produced_by_step: 'primary',
    headers: ['fit_status', 'estimate', 'ci_low', 'ci_high', 'effect_scale', 'is_primary_contrast', 'contrast'],
    rows: [['fitted', 0.85, 0.75, 0.96, 'odds_ratio', true, 'primary contrast']] };
  const payloads = { 'agent_plan.json': plan, 'result_tables.json': { tables: [primary, table] }, 'run_context.json': context };
  const run_id = `synthetic_strata_${++serial}`;
  const run = { run_id, artifact_refs: Object.keys(payloads).map(artifact => ({ artifact, run_id, sha256: 'a'.repeat(64) })) };
  const owner = modules.require('runAnswer').create({ tr, esc: global.EU_HTML.esc,
    projectId: () => 'synthetic', api: () => ({ loadPiCopilotResearchArtifact: async (_project, _run, artifact) => ({ payload: payloads[artifact] }) }) });
  await owner.load(run);
  const html = owner.render(run);
  const match = html.match(language === 'zh' ? /按[^<]*交互作用 p[^<]*。/ : /Stratified by[^<]*interaction p[^<]*\./);
  assert.ok(match, html);
  return match[0];
}

for (const language of ['zh', 'en']) {
  test(`${language}: the continuous scale uses its registered exposure name and localized unit phrase`, async () => {
    assert.equal(await sentence({ language }), language === 'zh'
      ? '按病区分层（最低标志物每升高 1 g/dL）：北区 OR 0.79（95% CI 0.69–0.92）；南区 OR 0.91（95% CI 0.82–1.00）；交互作用 p = 0.141。'
      : 'Stratified by ward (per 1 g/dL higher minimum marker): North OR 0.79 (95% CI 0.69–0.92); South OR 0.91 (95% CI 0.82–1.00); interaction p = 0.141.');
  });
  test(`${language}: a registered exposure name takes precedence over the catalog fallback`, async () => {
    const displayLabel = language === 'zh' ? '最低血清标志物' : 'Minimum serum marker';
    const text = await sentence({ language, displayLabel });
    assert.ok(text.includes(language === 'zh' ? '最低血清标志物每升高 1 g/dL' : 'per 1 g/dL higher minimum serum marker'), text);
  });
  for (const registry of [true, false]) {
    test(`${language}: categorical contrast uses ${registry ? 'grouping' : 'display'} level labels`, async () => {
      const text = await sentence({ language, categorical: true, registry });
      assert.ok(text.includes(language === 'zh' ? '（阈值以上 vs 阈值以下）' : '(Above threshold vs Below threshold)'), text);
      assert.doesNotMatch(text, /synthetic_band|2 vs 1|参照|reference/);
    });
  }
  test(`${language}: a contrast without an exposure prefix follows the primary model binding`, async () => {
    const text = await sentence({ language, categorical: true, prefix: false });
    assert.ok(text.includes(language === 'zh' ? '（阈值以上 vs 阈值以下）' : '(Above threshold vs Below threshold)'), text);
  });
  test(`${language}: a per-unit clause without a prefix follows the primary exposure`, async () => {
    const text = await sentence({ language, prefix: false });
    assert.ok(text.includes(language === 'zh' ? '最低标志物每升高 1 g/dL' : 'per 1 g/dL higher minimum marker'), text);
  });
  test(`${language}: a missing display label falls back to the existing catalog display owner`, async () => {
    const text = await sentence({ language, registered: false });
    assert.ok(text.includes(language === 'zh' ? '最低标志物每升高 1 g/dL' : 'per 1 g/dL higher minimum marker'), text);
    assert.doesNotMatch(text, /marker_min|per .*increase/);
  });
  test(`${language}: an unknown exposure stays readable without exposing its code`, async () => {
    const text = await sentence({ language, registered: false, exposure: 'synthetic_unknown' });
    assert.ok(text.includes(language === 'zh' ? '暴露变量每升高 1 g/dL' : 'per 1 g/dL higher exposure'), text);
    assert.doesNotMatch(text, /synthetic_unknown/);
  });
  test(`${language}: an unregistered categorical level keeps its variable`, async () => {
    const text = await sentence({ language, categorical: true, partial: true });
    assert.ok(text.includes(language === 'zh' ? 'synthetic_band=2 vs 阈值以下' : 'synthetic_band=2 vs Below threshold'), text);
  });
  test(`${language}: the registered scale magnitude and unit remain unchanged`, async () => {
    const text = await sentence({ language, unit: '2.5 mg/L' });
    assert.ok(text.includes(language === 'zh' ? '每升高 2.5 mg/L' : 'per 2.5 mg/L higher minimum marker'), text);
  });
}
test('Chinese modifier labels only insert spaces at CJK/Latin boundaries', async () => {
  assert.ok((await sentence({ modifier: 'Ward' })).startsWith('按 Ward 分层'));
  assert.ok((await sentence({ modifier: 'ICU病区' })).startsWith('按 ICU 病区分层'));
});
test('an English manuscript label uses the localized catalog aggregate on a Chinese page', async () => {
  const text = await sentence({ displayLabel: 'Minimum marker' });
  assert.ok(text.includes('最低标志物每升高 1 g/dL'), text);
  assert.doesNotMatch(text, /Minimum marker|per .*increase|marker_min/);
});
