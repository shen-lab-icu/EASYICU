'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const analysisSource = fs.readFileSync(process.argv[2], 'utf8');
const articleSource = fs.readFileSync(process.argv[3], 'utf8');
const escape = value => String(value == null ? '' : value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const context = {
  window: {
    EU_LANG: 'en', EU_HTML: { esc: escape },
    AGENT_RENDER: {
      figureGallery: () => '<div class="safe-gallery">figures</div>',
      manuscriptProvenanceView: payload => `<article>${escape(payload.marker)}${payload.figure_gallery ? '<figure>bound</figure>' : ''}</article>`,
    },
  },
};
vm.createContext(context);
vm.runInContext(analysisSource, context);
vm.runInContext(articleSource, context);

const claim = (id, field, value, displayValue) => ({
  claim_id: id, source_field: field, canonical_value: value,
  display_value: displayValue, source_json_pointer: `/${field}`,
  source_value: String(value),
  evidence: {
    evidence_id: `evidence_${id}`, sha256: id.repeat(64).slice(0, 64),
    kind: 'statistic', description: '<img src=x onerror=bad()>',
  },
});
const html = context.window.EU_GUIDED_PI_ANALYSIS_REPORT.render({
  run_context: { question: '<svg onload=bad()> E2?' },
  manuscript_provenance: { claims: [
    claim('a', 'cohort.n_stays', 94458, '94,458'),
    claim('b', 'n_total', 50640, '50,640'),
    claim('c', 'n_complete_case', 44095, '44,095'),
    claim('d', 'n_events', 5480, '5,480'),
    claim('e', 'reportable_descriptive_results.overall_outcome.risk_pct', 13.8, '13.8%'),
    claim('f', 'primary_or', 1.656, '1.656'),
    claim('1', 'primary_or_ci[0]', 1.599, '1.599'),
    claim('2', 'primary_or_ci[1]', 1.716, '1.716'),
  ] },
  figure_gallery: { presentation_variant: true, figures: [] },
});
assert.ok(html.includes('Complete analysis report'));
assert.ok(html.includes('Result interpretation'));
assert.ok(html.includes('safe-gallery'));
assert.ok(html.includes('data-gpi-evidence-open'));
assert.ok(html.includes('&lt;svg onload=bad()&gt;'));
assert.ok(html.includes('&lt;img src=x onerror=bad()&gt;'));
assert.ok(!html.includes('<svg onload=bad()>'));
assert.ok(!html.includes('<img src=x onerror=bad()>'));

const mismatchedHtml = context.window.EU_GUIDED_PI_ANALYSIS_REPORT.render({
  run_context: { question: 'Mismatch probe' },
  manuscript_provenance: { claims: [
    claim('m', 'primary_or', 2.0, '2.0'),
    claim('n', 'robustness_rows[0].ci_low', 0.8, '0.8'),
    claim('o', 'robustness_rows[0].ci_high', 1.2, '1.2'),
  ] },
  figure_gallery: {},
});
assert.ok(!mismatchedHtml.includes('OR 2.0 (95% CI 0.8–1.2)'));

const unrelatedHtml = context.window.EU_GUIDED_PI_ANALYSIS_REPORT.render({
  run_context: { question: 'Does vasopressor exposure predict acute kidney injury?' },
  manuscript_provenance: { claims: [] },
  figure_gallery: {},
});
assert.ok(unrelatedHtml.includes('Does vasopressor exposure predict acute kidney injury?'));
for (const caseSpecificText of [
  'peak lactate', 'MIMIC-IV', '24-hour landmark', '1–5 mmol/L',
  'Charlson comorbidity index', 'in-hospital mortality', 'governed E2 run',
  'Before landmark eligibility',
]) {
  assert.ok(!unrelatedHtml.includes(caseSpecificText), caseSpecificText);
}

const articleHtml = context.window.EU_GUIDED_PI_ARTICLE_REPORT.render({
  marker: '<script>bad()</script>', figure_gallery: { presentation_variant: true },
});
assert.ok(articleHtml.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(articleHtml.includes('<figure>bound</figure>'));

process.stdout.write(JSON.stringify({ ok: true, reports: 2 }));
