/* Executable contract for Patient Review quality and missingness truth. */
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
  quality_metrics: {
    summary: {
      total_records: 1000400,
      weighted_duplicate_time_pct: 0.2,
      weighted_out_of_physio_pct: 0.1,
    },
    features: [
      { feature: 'heart_rate', module: 'vitals', missing_pct: 20, records: 400 },
      { feature: 'creatinine', module: 'labs', missing_pct: 35, records: 1000000 },
      { feature: 'sepsis3', module: 'outcome', missing_pct: 90, records: 500 },
    ],
  },
};

const html = global.EU_PATIENT_OVERVIEW.renderQualityAudit({ drill }, helpers);
assert.match(html, /Feature-level missingness and observation volume/);
assert.match(html, /1,000,400 records audited/);
assert.match(html, /heart_rate: 20% missing, 400 records/);
assert.match(html, /creatinine: 35% missing, 1000000 records/);
assert.doesNotMatch(html, /sepsis3/);
assert.doesNotMatch(html, /coverage across all selected entities/);

process.stdout.write(JSON.stringify({ ok: true }));
