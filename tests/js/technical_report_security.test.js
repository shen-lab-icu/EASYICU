'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { test } = require('node:test');

const modulesSource = fs.readFileSync(process.argv[2], 'utf8');
const source = fs.readFileSync(process.argv[3], 'utf8');
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
vm.runInContext(modulesSource, context);
vm.runInContext(source, context);
vm.runInContext(fs.readFileSync(path.join(path.dirname(process.argv[3]), 'screens-guided-pi-evidence-preview.js'), 'utf8'), context);
const report = context.window.EasyICU.guidedPi.require('technicalReport');
const preview = context.window.EasyICU.guidedPi.require('evidencePreview');

function claim(id, field, value, displayValue) {
  return {
    claim_id: id, source_field: field, canonical_value: value,
    display_value: displayValue, source_json_pointer: `/${field.replaceAll('.', '/').replace(/\[(\d+)\]/g, '/$1')}`,
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

const riskField = 'reportable_descriptive_results.overall_outcome.risk_pct';
const exactRisk = 23.456789012345;
const sourceValue = {
  cohort: { n_stays: 0 }, n_complete_case: 18,
  reportable_descriptive_results: { overall_outcome: { risk_pct: exactRisk } },
  primary_or: 1.234567890123,
};
const digest = createHash('sha256').update(JSON.stringify(sourceValue)).digest('hex');
function bound(field, value, shown = String(value)) {
  const row = claim('a', field, value, shown);
  row.source_value = typeof value === 'number' ? String(Number(value.toPrecision(6))) : String(value);
  row.status = 'current';
  row.evidence = { evidence_id: 'primary_summary', sha256: digest, kind: 'statistic', status: 'current' };
  return row;
}
function render(rows, language = 'en') {
  context.window.EU_LANG = language;
  return report.render({ manuscript_provenance: { claims: rows },
    source_manifest: { readiness: { analysis_validated: true } } });
}
function targets(rendered) {
  return Array.from(rendered.matchAll(/<button\b[^>]*\bdata-gpi-evidence-open[^>]*>/g), match =>
    Object.fromEntries(Array.from(match[0].matchAll(/data-evidence-([a-z0-9-]+)="([^"]*)"/g), attr => [attr[1], attr[2]])));
}
function open(target) {
  assert.equal(target.sha256, digest, 'the registered digest is retained');
  return preview.render({ previewable: true, renderer: 'json', kind: 'statistic',
    evidence_id: target.id, sha256: target.sha256, value: sourceValue,
  }, { pointer: target.pointer, value: target['source-value'] });
}

test('technical metric buttons round only their visible text, including zero', () => {
  const rendered = render([
    bound('cohort.n_stays', 0, '0'), bound('n_complete_case', 18, '18'),
    bound(riskField, exactRisk, '23.46%'), bound('primary_or', sourceValue.primary_or, '1.235'),
  ]);
  const buttons = targets(rendered);
  assert.equal(buttons.length, 4);
  for (const button of buttons) {
    assert.ok(open(button).includes('Source field and value match'), button.pointer);
  }
  assert.ok(rendered.includes('23.46%'));
  assert.equal(buttons.find(row => row.pointer === '/cohort/n_stays')['source-value'], '0');
  assert.equal(buttons.find(row => row.pointer.endsWith('/risk_pct'))['source-value'], String(exactRisk));
});

test('technical metric uses a legacy source value only when canonical is absent', () => {
  for (const value of [0, exactRisk]) {
    const row = bound(riskField, value, `${value}%`);
    delete row.canonical_value;
    row.source_value = value;
    const buttons = targets(render([row]));
    assert.equal(buttons.length, 1, 'legacy source still has an evidence action');
    assert.equal(buttons[0]['source-value'], String(value));
  }
});

test('technical metric refuses stale claims and stale evidence', () => {
  for (const stale of ['claim', 'evidence']) {
    const row = bound(riskField, exactRisk);
    if (stale === 'claim') row.status = 'stale';
    else row.evidence.status = 'stale';
    assert.equal(targets(render([row])).length, 0, stale);
  }
});

test('technical metric refuses invalid canonical values without falling back to display', () => {
  for (const value of [NaN, Infinity, -Infinity, '23.456789012345', true]) {
    const row = bound(riskField, value, '23.46%');
    row.source_value = String(exactRisk);
    assert.equal(targets(render([row])).length, 0, String(value));
  }
  const missing = bound(riskField, null, '23.46%');
  delete missing.source_value;
  assert.equal(targets(render([missing])).length, 0);
});

test('technical metric requires an evidence identity, digest and scalar pointer', () => {
  for (const changed of [{ evidence_id: '../unregistered' }, { sha256: '' }, { sha256: 'z'.repeat(64) }]) {
    const row = bound(riskField, exactRisk);
    Object.assign(row.evidence, changed);
    assert.equal(targets(render([row])).length, 0);
  }
  const row = bound(riskField, exactRisk);
  delete row.source_json_pointer;
  assert.equal(targets(render([row])).length, 0);
});

test('technical metric preserves an exact reader mismatch instead of searching other fields', () => {
  const row = bound(riskField, exactRisk);
  row.source_json_pointer = '/n_complete_case';
  const selected = open(targets(render([row]))[0]);
  assert.ok(selected.includes('Selected number could not be matched'));
  assert.ok(!selected.includes('Source field and value match'));
});

const intervalFamilies = [
  ['primary_or_ci[0]', 'primary_or_ci[1]'],
  ['primary_or_ci_low', 'primary_or_ci_high'],
  ['primary_ci_low', 'primary_ci_high'],
];
test('technical report consumes each primary interval family from the estimate source', () => {
  for (const [low, high] of intervalFamilies) {
    const rows = [bound('primary_or', 1.25, '1.250'), bound(low, 0.875, '0.875'), bound(high, 1.625, '1.625')];
    rows[2].evidence.sha256 = digest.toUpperCase();
    const rendered = render(rows, 'zh');
    assert.ok(rendered.includes('95% CI 0.875–1.625'), low);
    assert.ok(rendered.includes('class="gpi-tech-effect"'), low);
    assert.ok(!rendered.includes('未授权调整后效应估计'));
    assert.ok(rendered.includes('分析已验证 · 等待审阅'));
  }
});

test('technical interval cannot come from a different source, pointer or stale record', () => {
  for (const invalid of ['id', 'digest', 'pointer', 'status', 'effect_pointer', 'effect_status']) {
    const effect = bound('primary_or', 1.25, '1.250');
    const low = bound('primary_ci_low', 0.875, '0.875');
    const high = bound('primary_ci_high', 1.625, '1.625');
    if (invalid === 'id') low.evidence.evidence_id = 'sensitivity_summary';
    if (invalid === 'digest') high.evidence.sha256 = 'f'.repeat(64);
    if (invalid === 'pointer') high.source_json_pointer = '/sensitivity/primary_ci_high';
    if (invalid === 'status') low.status = 'stale';
    if (invalid === 'effect_pointer') effect.source_json_pointer = '/another_model/primary_or';
    if (invalid === 'effect_status') effect.status = 'stale';
    const rendered = render([effect, low, high]);
    assert.ok(!rendered.includes('95% CI 0.875–1.625'), invalid);
    assert.ok(!rendered.includes('class="gpi-tech-effect"'), invalid);
  }
});

test('technical interval keeps endpoint families together and ignores sensitivity intervals', () => {
  for (const fields of [['primary_ci_low', 'primary_or_ci_high'], ['robustness_rows[0].ci_low', 'robustness_rows[0].ci_high']]) {
    const rendered = render([bound('primary_or', 1.25), bound(fields[0], 0.875), bound(fields[1], 1.625)]);
    assert.ok(!rendered.includes('95% CI 0.875–1.625'));
    assert.ok(!rendered.includes('class="gpi-tech-effect"'));
  }
});

test('technical interval finds the matching source after an unrelated model', () => {
  const unrelated = bound('primary_ci_low', 9.5);
  unrelated.evidence.evidence_id = 'sensitivity_summary';
  const rendered = render([bound('primary_or', 1.25), unrelated,
    bound('primary_ci_low', 0.875), bound('primary_ci_high', 1.625)]);
  assert.ok(rendered.includes('95% CI 0.875–1.625'));
});

test('technical missing display data does not assert a model authorization decision', () => {
  for (const rows of [[], [bound('primary_or', 1.25)]]) {
    const rendered = render(rows, 'zh');
    assert.ok(!rendered.includes('未授权调整后效应估计'));
    assert.ok(rendered.includes('当前视图'));
    assert.ok(rendered.includes('分析已验证 · 等待审阅'));
  }
});

// Full-analysis regressions share the registered report harness to keep P2 owner-local.
{
  const assert = require('node:assert/strict');
  const fs = require('node:fs');
  const path = require('node:path');
  const vm = require('node:vm');
  const { createHash } = require('node:crypto');
  const { test } = require('node:test');

  // Explicit owner arguments also let the same regression tests exercise the read-only base.
  const owner = path.join(path.dirname(process.argv[3]), 'screens-guided-pi-analysis-report.js');
  const escape = value => String(value == null ? '' : value)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
  const context = vm.createContext({ window: { EU_LANG: 'en', EU_HTML: { esc: escape } } });
  for (const file of [process.argv[2], owner,
    path.join(path.dirname(owner), 'screens-guided-pi-evidence-preview.js')]) {
    vm.runInContext(fs.readFileSync(file, 'utf8'), context);
  }
  const report = context.window.EasyICU.guidedPi.require('analysisReport');
  const preview = context.window.EasyICU.guidedPi.require('evidencePreview');
  const riskField = 'reportable_descriptive_results.overall_outcome.risk_pct';
  const exactRisk = 23.456789012345;
  const sourceValue = { cohort: { n_stays: 24 }, n_total: 18, n_events: 0,
    reportable_descriptive_results: { overall_outcome: { risk_pct: exactRisk } } };
  const digest = createHash('sha256').update(JSON.stringify(sourceValue)).digest('hex');
  function claim(field, value, shown = String(value)) {
    return { source_field: field, canonical_value: value, display_value: shown, status: 'current',
      source_value: typeof value === 'number' ? String(Number(value.toPrecision(6))) : String(value),
      source_json_pointer: `/${field.replaceAll('.', '/').replace(/\[(\d+)\]/g, '/$1')}`,
      evidence: { evidence_id: 'primary_summary', sha256: digest, kind: 'statistic', status: 'current' } };
  }
  function render(rows, language = 'en') {
    context.window.EU_LANG = language;
    return report.render({ manuscript_provenance: { claims: rows },
      source_manifest: { readiness: { manuscript_ready: false } } });
  }
  function targets(html) {
    return Array.from(html.matchAll(/<button\b[^>]*\bdata-gpi-evidence-open[^>]*>/g), match =>
      Object.fromEntries(Array.from(match[0].matchAll(/data-evidence-([a-z0-9-]+)="([^"]*)"/g), attr => [attr[1], attr[2]])));
  }
  function open(target) {
    assert.equal(target.sha256, digest);
    return preview.render({ previewable: true, renderer: 'json', kind: 'statistic',
      evidence_id: target.id, sha256: target.sha256, value: sourceValue,
    }, { pointer: target.pointer, value: target['source-value'] });
  }

  test('analysis metric buttons round only visible text and open their exact source values', () => {
    const html = render([claim('cohort.n_stays', 24), claim('n_total', 18), claim('n_events', 0, '0'),
      claim(riskField, exactRisk, '23.456789%')]);
    const buttons = targets(html);
    assert.equal(buttons.length, 4);
    for (const button of buttons) assert.ok(open(button).includes('Source field and value match'), button.pointer);
    assert.ok(html.includes('<strong>23.46%</strong>'));
    assert.equal(buttons.find(row => row.pointer.endsWith('/risk_pct'))['source-value'], String(exactRisk));
    assert.equal(buttons.find(row => row.pointer === '/n_events')['source-value'], '0');
  });

  test('analysis metric retains a legacy source value including numeric zero', () => {
    for (const value of [0, exactRisk]) {
      const row = claim(riskField, value);
      delete row.canonical_value;
      row.source_value = value;
      const buttons = targets(render([row]));
      assert.equal(buttons.length, 1, 'legacy source still has an evidence action');
      assert.equal(buttons[0]['source-value'], String(value));
    }
  });

  test('analysis metric refuses stale claims and evidence', () => {
    for (const stale of ['claim', 'evidence']) {
      const row = claim(riskField, exactRisk);
      if (stale === 'claim') row.status = 'stale';
      else row.evidence.status = 'stale';
      assert.equal(targets(render([row])).length, 0, stale);
    }
  });

  test('analysis metric does not turn invalid canonical or missing source values into links', () => {
    for (const value of [NaN, Infinity, -Infinity, '23.456789012345', true]) {
      const row = claim(riskField, value, '23.46%');
      row.source_value = String(exactRisk);
      assert.equal(targets(render([row])).length, 0, String(value));
    }
    const missing = claim(riskField, null, '23.46%');
    delete missing.source_value;
    assert.equal(targets(render([missing])).length, 0);
  });

  test('analysis metric requires the registered digest, identity and pointer', () => {
    for (const changed of [{ evidence_id: '../unregistered' }, { sha256: '' }, { sha256: 'z'.repeat(64) }]) {
      const row = claim(riskField, exactRisk);
      Object.assign(row.evidence, changed);
      assert.equal(targets(render([row])).length, 0);
    }
    const row = claim(riskField, exactRisk);
    delete row.source_json_pointer;
    assert.equal(targets(render([row])).length, 0);
  });

  test('analysis metric with a different pointer fails the existing exact reader check', () => {
    const row = claim(riskField, exactRisk);
    row.source_json_pointer = '/n_total';
    const selected = open(targets(render([row]))[0]);
    assert.ok(selected.includes('Selected number could not be matched'));
    assert.ok(!selected.includes('Source field and value match'));
  });

  const families = [
    ['primary_or_ci[0]', 'primary_or_ci[1]'],
    ['primary_or_ci_low', 'primary_or_ci_high'],
    ['primary_ci_low', 'primary_ci_high'],
  ];
  test('analysis report consumes all supported primary interval shapes from the same source', () => {
    for (const [low, high] of families) {
      const rows = [claim('primary_or', 1.25, '1.250'), claim(low, 0.875), claim(high, 1.625)];
      rows[2].evidence.sha256 = digest.toUpperCase();
      const html = render(rows, 'zh');
      assert.ok(html.includes('OR 1.250（95% CI 0.875–1.625）'), low);
      assert.ok(html.includes('ANALYSIS ONLY'));
    }
  });

  test('analysis report refuses intervals from other sources, pointers or stale claims', () => {
    for (const invalid of ['id', 'digest', 'pointer', 'status', 'effect_pointer', 'effect_status']) {
      const effect = claim('primary_or', 1.25, '1.250');
      const low = claim('primary_ci_low', 0.875);
      const high = claim('primary_ci_high', 1.625);
      if (invalid === 'id') low.evidence.evidence_id = 'sensitivity_summary';
      if (invalid === 'digest') high.evidence.sha256 = 'f'.repeat(64);
      if (invalid === 'pointer') high.source_json_pointer = '/sensitivity/primary_ci_high';
      if (invalid === 'status') low.status = 'stale';
      if (invalid === 'effect_pointer') effect.source_json_pointer = '/another_model/primary_or';
      if (invalid === 'effect_status') effect.status = 'stale';
      assert.ok(!render([effect, low, high]).includes('95% CI 0.875–1.625'), invalid);
    }
  });

  test('analysis report does not mix interval families or borrow sensitivity endpoints', () => {
    for (const fields of [['primary_ci_low', 'primary_or_ci_high'], ['robustness_rows[0].ci_low', 'robustness_rows[0].ci_high']]) {
      assert.ok(!render([claim('primary_or', 1.25), claim(fields[0], 0.875), claim(fields[1], 1.625)])
        .includes('95% CI 0.875–1.625'));
    }
  });

  test('analysis report selects matching endpoints after unrelated model claims', () => {
    const unrelated = claim('primary_ci_low', 9.5);
    unrelated.evidence.evidence_id = 'sensitivity_summary';
    assert.ok(render([claim('primary_or', 1.25), unrelated, claim('primary_ci_low', 0.875), claim('primary_ci_high', 1.625)])
      .includes('95% CI 0.875–1.625'));
  });

  test('analysis report distinguishes missing interval display from a descriptive-only analysis', () => {
    const html = render([claim('primary_or', 1.25)], 'zh');
    assert.ok(html.includes('OR 1.25'));
    assert.ok(html.includes('当前视图缺少与主要估计匹配的置信区间'));
    assert.ok(!html.includes('没有估计调整后的关联'));
    assert.ok(!html.includes('未授权调整后效应估计'));
    assert.ok(html.includes('ANALYSIS ONLY'));
    assert.ok(render([], 'zh').includes('当前视图暂无匹配的主要估计'));
  });
}
