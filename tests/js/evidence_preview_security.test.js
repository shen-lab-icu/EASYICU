'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const modulesSource = fs.readFileSync(process.argv[2], 'utf8');
const source = fs.readFileSync(process.argv[3], 'utf8');
const escape = value => String(value == null ? '' : value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const context = {
  window: { EU_LANG: 'en', EU_HTML: { esc: escape } },
  globalThis: { pwned: false },
};
vm.createContext(context);
vm.runInContext(modulesSource, context);
vm.runInContext(source, context);
const renderer = context.window.EasyICU.guidedPi.require('evidencePreview');

const code = renderer.render({
  renderer: 'code', previewable: true, kind: 'code', evidence_id: 'code_1',
  sha256: 'a'.repeat(64), text: '<img src=x onerror="globalThis.pwned=true">\nestimate = 1.2',
}, { pointer: '/estimate', value: '<script>bad()</script>' });
assert.ok(code.includes('&lt;img src=x onerror=&quot;globalThis.pwned=true&quot;&gt;'));
assert.ok(code.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!code.includes('<img src=x'));
assert.equal(context.globalThis.pwned, false, 'preview code must never execute');

const table = renderer.render({
  renderer: 'table', previewable: true, kind: 'table', evidence_id: 'table_1',
  sha256: 'b'.repeat(64), headers: ['estimate'], rows: [['<svg onload=bad()>']],
}, {});
assert.ok(table.includes('&lt;svg onload=bad()&gt;'));
assert.ok(!table.includes('<svg onload=bad()>'));

const statistic = renderer.render({
  renderer: 'json', previewable: true, kind: 'statistic', evidence_id: 'statistic_1',
  sha256: 'd'.repeat(64), display_name: 'step_summary.json',
  relative_path: 'evidence/statistic_1__<bad>.json',
  description: '<img src=x onerror="bad()">', role: 'summarize_absolute_risk',
  producer: 'runner', generation_mode: 'fallback',
  declared_lineage: [
    { relation: 'analysis_code', status: 'registered', evidence_id: 'code_1', kind: 'code', sha256: 'a'.repeat(64), display_name: 'analysis.py', relative_path: 'evidence/code_1__analysis.py' },
    { relation: 'input_data', status: 'unregistered', evidence_id: 'missing_input' },
  ],
  run_authority: {
    status: 'recorded', run_id: 'run_1', git_sha: 'abc123', git_dirty: true,
    runner_image_digest: 'sha256:image', environment_identity_sha256: 'e'.repeat(64),
    links: [{ relation: 'run_plan_authority', status: 'registered', evidence_id: 'analysis_plan', kind: 'log', sha256: 'f'.repeat(64), display_name: 'analysis_plan.json', relative_path: 'evidence/analysis_plan__analysis_plan.json' }],
  },
  value: {
    analysis_family: 'absolute_risk_context', n_total: 50640, outcome: 'death',
    method: '<img src=x onerror="bad()">',
    reportable_descriptive_results: {
      overall_outcome: { outcome: 'death', n: 50640, event_n: 7006, risk_pct: 13.8, risk_ci_low_pct: 13.5, risk_ci_high_pct: 14.1 },
      exposures: [{ exposure: 'lact_max', groups: [{ label: 'above threshold', n: 100, outcome_n: 100, outcome_event_n: 20, outcome_risk_pct: 20, outcome_risk_ci_low_pct: 12, outcome_risk_ci_high_pct: 31 }] }],
    },
  },
}, { pointer: '/n_total', value: 50640 });
assert.ok(statistic.includes('Readable result'));
assert.ok(statistic.includes('Overall risk'));
assert.ok(statistic.includes('13.8%'));
assert.ok(statistic.includes('Registered source record'));
assert.ok(statistic.includes('Registry digest verified'));
assert.ok(statistic.includes('Reproduction path'));
assert.ok(statistic.includes('Code that generated this record'));
assert.ok(statistic.includes('data-evidence-id="code_1"'));
assert.ok(statistic.includes('missing_input'));
assert.ok(statistic.includes('Full file audit'));
assert.ok(statistic.includes('Continue to upstream run provenance'));
assert.ok(statistic.includes('data-evidence-id="analysis_plan"'));
assert.ok(statistic.includes('evidence/statistic_1__&lt;bad&gt;.json'));
assert.ok(statistic.includes('&lt;img src=x'));
assert.ok(statistic.includes('Open raw JSON for audit'));
assert.ok(!statistic.includes('<img src=x'));
assert.ok(statistic.indexOf('Readable result') < statistic.indexOf('Registered source record'));
const readerLayer = statistic.slice(0, statistic.indexOf('Full file audit'));
assert.ok(!readerLayer.includes('JSON pointer'));
assert.ok(!readerLayer.includes('statistic_1'));
assert.ok(!readerLayer.includes('d'.repeat(64)));
assert.ok(!readerLayer.includes('evidence/statistic_1__&lt;bad&gt;.json'));

const selectedPayload = {
  renderer: 'json', previewable: true, kind: 'statistic', evidence_id: 'primary_summary',
  sha256: 'd'.repeat(64), display_name: 'step_summary.json',
  value: { estimates: [{ rate: 23.45678 }, { rate: 0 }], 'escaped/key~': 8 },
};
const selected = renderer.render(selectedPayload, {
  pointer: '/estimates/0/rate', value: '23.45678', display: '23.46%',
});
const selectedReader = selected.slice(0, selected.indexOf('Full file audit'));
assert.ok(selectedReader.includes('Selected number source'));
assert.ok(selectedReader.includes('23.46%'));
assert.ok(selectedReader.includes('23.45678'));
assert.ok(selectedReader.includes('Source field and value match'));
assert.ok(!selectedReader.includes('/estimates/0/rate'), 'raw pointers remain in the audit layer');
const zero = renderer.render(selectedPayload, { pointer: '/estimates/1/rate', value: '0', display: '0%' });
assert.ok(zero.includes('Source field and value match'));
assert.ok(zero.includes('<strong>0</strong>'));
const escapedPointer = renderer.render(selectedPayload, { pointer: '/escaped~1key~0', value: '8' });
assert.ok(escapedPointer.includes('Source field and value match'));
for (const locator of [
  { pointer: '/estimates/0/rate', value: '999' },
  { pointer: '/estimates/1/rate', value: '23.45678' },
  { pointer: '/estimates/9/rate', value: '23.45678' },
  { pointer: '/estimates/0', value: '[object Object]' },
  { pointer: '/constructor', value: '8' },
  { pointer: '/escaped~2key~', value: '8' },
]) {
  const mismatch = renderer.render(selectedPayload, locator);
  assert.ok(mismatch.includes('Selected number could not be matched'));
  assert.ok(!mismatch.includes('Source field and value match'));
}
const selectedXss = renderer.render(selectedPayload, {
  pointer: '/estimates/0/rate', value: '23.45678', display: '<img src=x onerror="bad()">',
});
assert.ok(!selectedXss.includes('<img src=x'));
assert.ok(selectedXss.includes('&lt;img src=x'));

const distribution = {
  ...selectedPayload,
  value: {
    exposure: 'event_status', outcome: 'hospital_endpoint',
    descriptive_estimates: {
      schema_version: 'easyicu.exposure_outcome_descriptive_estimates/1',
      exposure_prevalence: [
        { level: 0, n: 60, denominator: 100, estimate_pct: 60 },
        { level: 1, n: 40, denominator: 100, estimate_pct: 40 },
      ],
      outcome_absolute_risks: [
        { level: 0, events: 6, denominator: 60, estimate_pct: 10 },
        { level: 1, events: 8, denominator: 40, estimate_pct: 20 },
      ],
    },
  },
};
const originalDistribution = JSON.stringify(distribution);
function selectionOnly(payload, pointer, value) {
  const rendered = renderer.render(payload, { pointer, value: String(value), display: String(value) });
  return rendered.slice(0, rendered.indexOf('</section>') + '</section>'.length);
}
const prevalenceSelection = selectionOnly(distribution, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40);
assert.ok(prevalenceSelection.includes('Group share of the cohort'));
assert.ok(prevalenceSelection.includes('40 / 100'));
assert.ok(prevalenceSelection.includes('Recorded group value'));
assert.ok(!prevalenceSelection.includes('hospital_endpoint'));
assert.ok(!prevalenceSelection.includes('8 / 40'));
const outcomeSelection = selectionOnly(distribution, '/descriptive_estimates/outcome_absolute_risks/1/estimate_pct', 20);
assert.ok(outcomeSelection.includes('Observed outcome proportion within the group'));
assert.ok(outcomeSelection.includes('8 / 40'));
assert.ok(outcomeSelection.includes('hospital_endpoint'));
assert.ok(!outcomeSelection.includes('40 / 100'));
for (const [family, metric, field, count] of [
  ['exposure_prevalence', 'Records in this group', 'n', 40],
  ['exposure_prevalence', 'Cohort denominator', 'denominator', 100],
  ['outcome_absolute_risks', 'Outcome events in this group', 'events', 8],
  ['outcome_absolute_risks', 'Outcome denominator in this group', 'denominator', 40],
]) {
  assert.ok(selectionOnly(distribution, `/descriptive_estimates/${family}/1/${field}`, count).includes(metric));
}
const wrongSelectedValue = renderer.render(distribution, {pointer:'/descriptive_estimates/exposure_prevalence/0/estimate_pct',value:'40'});
assert.ok(wrongSelectedValue.includes('Selected number could not be matched'));
assert.ok(!wrongSelectedValue.includes('Selected metric'));
const unsupported = JSON.parse(originalDistribution);
unsupported.value.descriptive_estimates.schema_version = 'unknown';
assert.ok(!selectionOnly(unsupported, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40).includes('Selected metric'));
const invalidDenominator = JSON.parse(originalDistribution);
invalidDenominator.value.descriptive_estimates.exposure_prevalence[1].denominator = -1;
assert.ok(!selectionOnly(invalidDenominator, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40).includes('Group records / cohort records'));
const dangerousGroup = JSON.parse(originalDistribution);
dangerousGroup.value.descriptive_estimates.exposure_prevalence[1].level = '<img src=x onerror=bad()>';
const escapedGroup = selectionOnly(dangerousGroup, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40);
assert.ok(escapedGroup.includes('&lt;img src=x'));
assert.ok(!escapedGroup.includes('<img src=x'));
assert.equal(JSON.stringify(distribution), originalDistribution, 'source JSON remains unchanged');
const selectedWithContext = renderer.render(distribution, {pointer:'/descriptive_estimates/exposure_prevalence/1/estimate_pct',value:'40'});
assert.ok(selectedWithContext.includes('<summary>Whole-file context (not the selected metric)</summary>'));

for (const [confidence, label] of [[0.90, '90% CI'], [null, 'Confidence interval'], [95, 'Confidence interval']]) {
  const html = renderer.render({...selectedPayload, value:{estimate:1.2,ci:[1,1.4],confidence_level:confidence}});
  assert.ok(html.includes(`<span>${label}</span>`));
  assert.ok(!html.includes('95% CI'), 'never infer a confidence level from two endpoints');
  const grouped = renderer.render({...selectedPayload, value:{
    reportable_descriptive_results:{exposures:[{exposure:'event_status',groups:[{
      label:'recorded',outcome_risk_ci_low_pct:10,outcome_risk_ci_high_pct:30,confidence_level:confidence,
    }]}]},
  }});
  assert.ok(grouped.includes('<th>Confidence interval</th>'));
  assert.ok(!grouped.includes('95% CI'));
  if (confidence === 0.90) assert.ok(grouped.includes('(90% CI)'));
}
const declared95 = renderer.render({...selectedPayload, value:{estimate:1.2,ci:[1,1.4],confidence_level:0.95}});
assert.ok(declared95.includes('<span>95% CI</span>'), 'retain explicitly recorded confidence levels');

const withheld = renderer.render({
  renderer: 'metadata', previewable: false, kind: 'table', evidence_id: 'cohort_1',
  sha256: 'c'.repeat(64), withheld_reason: 'patient_level_rows_withheld', bytes: 1024,
}, { pointer: '/estimates/0/rate', value: '23.45678', display: '23.46%' });
assert.ok(withheld.includes('Patient-level cohort rows are withheld'));
assert.ok(!withheld.includes('<table'));
assert.ok(!withheld.includes('Source field and value match'));
assert.ok(!withheld.includes('23.46%'));

process.stdout.write(JSON.stringify({ ok: true, suites: ['safe_preview', 'exact_source_selection', 'typed_metric_context', 'confidence_level_metadata'] }));
