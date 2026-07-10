/* Executable contract for Patient Review scope and unknown-coverage rendering. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
require(path.resolve(process.argv[2]));

const helpers = {
  t: en => en,
  esc: value => String(value == null ? '' : value),
  fmtInt: value => Number(value).toLocaleString('en-US'),
  fmtNum: (value, digits = 1) => Number(value).toFixed(digits),
  fmtPct: value => `${Number(value).toFixed(1)}%`,
  icon: () => '',
  moduleLabel: item => item.label || item.module,
};

const drill = {
  summary: {
    entities: 94458,
    review_entities: 500,
    review_entity_cap: 500,
    review_scope: 'browser_bounded_entity_sample',
  },
  module_profiles: [
    { module: 'vitals', label: 'Vitals', entities: 400, coverage_pct: 80, feature_count: 4 },
    { module: 'labs', label: 'Laboratory', entities: null, coverage_pct: null, feature_count: 8, rows: 1000000 },
  ],
  quality: [
    { module: 'vitals', metric_kind: 'coverage', entities: 400, coverage_pct: 80, quality_status: 'ok' },
    { module: 'labs', metric_kind: 'coverage', entities: null, coverage_pct: null, quality_status: 'unknown' },
  ],
};

const html = global.EU_PATIENT_OVERVIEW.renderQualityAudit({ drill }, helpers);
assert.match(html, /500 reviewed \/ 94,458 full/);
assert.match(html, /400 \/ 500/);
assert.doesNotMatch(html, /400 \/ 94,458/);
assert.match(html, /1 modules have inventory metadata only/);
assert.doesNotMatch(html, /data-patient-missingness-module="labs"/);
assert.doesNotMatch(html, /coverage across all selected entities/);
assert.doesNotMatch(html, /Laboratory[\s\S]{0,300}100\.0%/);

process.stdout.write(JSON.stringify({ ok: true }));
