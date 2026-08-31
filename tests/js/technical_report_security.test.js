'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const escape = value => String(value == null ? '' : value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const context = {
  window: {
    EU_LANG: 'en', EU_HTML: { esc: escape },
    AGENT_RENDER: { figureGallery: () => '<div class="safe-gallery">figures</div>' },
  },
};
vm.createContext(context);
vm.runInContext(source, context);
const report = context.window.EU_GUIDED_PI_TECHNICAL_REPORT;

function claim(id, field, value, displayValue) {
  return {
    claim_id: id, source_field: field, canonical_value: value,
    display_value: displayValue, source_json_pointer: `/${field}`,
    source_value: String(value),
    evidence: {
      evidence_id: `evidence_${id}`, sha256: id.repeat(64).slice(0, 64),
      kind: 'statistic', description: '<img src=x onerror=bad()>',
    },
  };
}

const html = report.render({
  run_context: { question: '<svg onload=bad()> ICU outcome?' },
  source_manifest: { readiness: { analysis_validated: true } },
  manuscript_provenance: { claims: [
    claim('a', 'cohort.n_stays', 94458, '94,458'),
    claim('b', 'n_complete_case', 44111, '44,111'),
    claim('c', 'reportable_descriptive_results.overall_outcome.risk_pct', 10, '10.0%'),
    claim('d', 'primary_or', 1.644, '1.644'),
    claim('e', 'primary_or_ci[0]', 1.587, '1.587'),
    claim('f', 'primary_or_ci[1]', 1.703, '1.703'),
    claim('1', 'reportable_descriptive_results.exposures[0].groups[0].outcome_risk_pct', 13.8, '13.8%'),
    claim('2', 'reportable_descriptive_results.exposures[0].groups[1].outcome_risk_pct', 5.6, '5.6%'),
  ] },
  figure_gallery: { figures: [] },
});

assert.ok(html.includes('Technical analysis report'));
assert.ok(html.includes('94,458'));
assert.ok(html.includes('OR 1.644'));
assert.ok(html.includes('13.8%'));
assert.ok(html.includes('data-gpi-evidence-open'));
assert.ok(html.includes('data-gpi-report-artifact="result_tables.json"'));
assert.ok(html.includes('safe-gallery'));
assert.ok(html.includes('&lt;svg onload=bad()&gt;'));
assert.ok(html.includes('&lt;img src=x onerror=bad()&gt;'));
assert.ok(!html.includes('<svg onload=bad()>'));
assert.ok(!html.includes('<img src=x onerror=bad()>'));

const mismatchedHtml = report.render({
  run_context: { question: 'Mismatch probe' },
  source_manifest: { readiness: {} },
  manuscript_provenance: { claims: [
    claim('m', 'primary_or', 2.0, '2.0'),
    claim('n', 'robustness_rows[0].ci_low', 0.8, '0.8'),
    claim('o', 'robustness_rows[0].ci_high', 1.2, '1.2'),
  ] },
  figure_gallery: {},
});
assert.ok(!mismatchedHtml.includes('OR 2.0 (95% CI 0.8–1.2)'));

process.stdout.write(JSON.stringify({ ok: true, metrics: 4, evidence_links: true }));
