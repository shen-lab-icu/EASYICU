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

const withheld = renderer.render({
  renderer: 'metadata', previewable: false, kind: 'table', evidence_id: 'cohort_1',
  sha256: 'c'.repeat(64), withheld_reason: 'patient_level_rows_withheld', bytes: 1024,
}, {});
assert.ok(withheld.includes('Patient-level cohort rows are withheld'));
assert.ok(!withheld.includes('<table'));

process.stdout.write(JSON.stringify({ ok: true, cases: 4 }));
