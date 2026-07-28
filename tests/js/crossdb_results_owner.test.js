/* Executable contract for the Cross-DB result workspace owner. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_CROSSDB_CHARTS = {
  begin() {},
  dispose() {},
  mount() { return 1; },
  render() {
    return '<div class="xdb-main-chart"><div data-crossdb-echart="test"></div></div>';
  },
};

require(path.resolve(process.argv[2]));

const owner = global.EU_CROSSDB_RESULTS;
assert(owner, 'Cross-DB results owner must be published');

const payload = {
  source_type: 'registered_exports',
  sources: [
    { label: 'MIMIC-IV Demo', database: 'miiv', path_hash: 'abc', summary: { total_records: 100 } },
    { label: 'eICU Demo', database: 'eicu', path_hash: 'def', summary: { total_records: 2500 } },
  ],
  shared_modules: ['vitals'],
  availability: [
    { module: 'vitals', shared: true, values: [{ present: true, coverage_pct: 98 }, { present: true, coverage_pct: 91 }] },
  ],
  rows: [
    { key: 'total_records', label: 'Total records', values: [100, 2500], delta: 2400 },
  ],
  compatibility_gate: { status: 'compatible', comparison_mode: 'descriptive_only' },
  blocked_features: [{ id: 'matched_cohort' }],
  privacy: { raw_rows_returned: false },
  provenance: { inference: 'blocked_until_numeric_evidence_gate' },
  feature_distributions: [
    {
      module: 'vitals',
      shared_feature_count: 1,
      features: [
        {
          feature: 'hr',
          values: [
            { present: true, non_null: 100, min: 50, max: 140, points: [{ x: 50, density: 0 }, { x: 90, density: 1 }, { x: 140, density: 0 }] },
            { present: true, non_null: 200, min: 40, max: 150, points: [{ x: 40, density: 0 }, { x: 85, density: 0.8 }, { x: 150, density: 0 }] },
          ],
        },
        {
          feature: 'spo2',
          values: [
            { present: true, non_null: 90, min: 70, max: 100, points: [{ x: 70, density: 0 }, { x: 97, density: 1 }, { x: 100, density: 0 }] },
            { present: false },
          ],
        },
      ],
    },
  ],
};

let repaintCount = 0;
let exported = null;
let expanded = false;
const config = {
  catalogTotals: { modules: 19, features: 281 },
  coreFeatures: ['hr'],
  helpers: {
    catalogFeatureMeta(key) {
      return {
        hr: { name: 'Heart Rate', unit: 'bpm' },
        spo2: { name: 'Oxygen Saturation', unit: '%' },
      }[key] || { name: key, unit: '' };
    },
    catalogModuleLabel: key => ({ vitals: 'Vital signs' }[key] || key),
    esc: value => String(value == null ? '' : value),
    fmtInt: value => Number(value).toLocaleString(),
    fmtNum: value => String(value),
    fmtPct: value => `${value}%`,
    icon: () => '',
    metricLabel: value => value,
    statusLabel: value => value,
    t: english => english,
    term: value => value,
  },
  expandScope() { expanded = true; },
  exportPayload(value) { exported = value; },
  repaint() { repaintCount += 1; },
};

function fakeControl(dataset = {}) {
  const handlers = {};
  return {
    dataset,
    value: '',
    addEventListener(type, handler) { handlers[type] = handler; },
    fire(type, event = {}) {
      assert.equal(typeof handlers[type], 'function', `${type} handler must be bound`);
      handlers[type]({ preventDefault() {}, stopPropagation() {}, ...event });
    },
  };
}

function rootWith(mapping) {
  return {
    querySelectorAll(selector) {
      return mapping[selector] || [];
    },
  };
}

const initial = owner.render(payload, config);
assert.match(initial, /data-crossdb-result-tab="overview"/);
assert.match(initial, /Cross-database consistency workspace/);
assert.match(initial, /Comparable feature profiles/);
assert.doesNotMatch(initial, /data-crossdb-export/);
assert.doesNotMatch(initial, /xdb-density-features/);

const distributionsTab = fakeControl({ crossdbResultTab: 'distributions' });
owner.bind(rootWith({ '[data-crossdb-result-tab]': [distributionsTab] }), payload, config);
distributionsTab.fire('click');
assert.equal(repaintCount, 1);

const distributions = owner.render(payload, config);
assert.match(distributions, /data-crossdb-result-panel="distributions"/);
assert.match(distributions, /data-crossdb-scope="core"/);
assert.match(distributions, /Heart Rate/);
assert.match(distributions, /Oxygen Saturation/);
assert.match(distributions, /xdb-main-chart/);
assert.equal((distributions.match(/class="xdb-main-chart"/g) || []).length, 1);

const coreScope = fakeControl({ crossdbScope: 'core' });
owner.bind(rootWith({ '[data-crossdb-scope]': [coreScope] }), payload, config);
coreScope.fire('click');
const coreFeatures = owner.render(payload, config);
assert.doesNotMatch(coreFeatures, /Oxygen Saturation/, 'core scope remains available as an explicit filter');

const allScope = fakeControl({ crossdbScope: 'all' });
owner.bind(rootWith({ '[data-crossdb-scope]': [allScope] }), payload, config);
allScope.fire('click');
const allFeatures = owner.render(payload, config);
assert.match(allFeatures, /Oxygen Saturation/);
assert.match(allFeatures, /All mapped features/);

const featureQuery = fakeControl();
featureQuery.value = 'oxygen';
owner.bind(rootWith({ '[data-crossdb-feature-query]': [featureQuery] }), payload, config);
featureQuery.fire('keydown', { key: 'Enter' });
const searchedFeatures = owner.render(payload, config);
assert.match(searchedFeatures, /Oxygen Saturation/);
assert.doesNotMatch(searchedFeatures, /Heart Rate/);

const qualityTab = fakeControl({ crossdbResultTab: 'quality' });
owner.bind(rootWith({ '[data-crossdb-result-tab]': [qualityTab] }), payload, config);
qualityTab.fire('click');
const quality = owner.render(payload, config);
assert.match(quality, /Quality, scope, and provenance/);
assert.match(quality, /matched_cohort/);
assert.match(quality, /raw rows returned/);

const exportButton = fakeControl();
owner.bind(rootWith({ '[data-crossdb-export]': [exportButton] }), payload, config);
exportButton.fire('click');
assert.equal(exported, payload);

owner.reset();
const partialPayload = {
  ...payload,
  source_type: 'raw_database_root',
  provenance: { feature_scope: 'curated_core', inference: 'blocked_until_numeric_evidence_gate' },
};
const partial = owner.render(partialPayload, config);
assert.match(partial, /data-crossdb-partial-scope/);
assert.match(partial, /1 \/ 19/);
assert.match(partial, /2 \/ 281/);
assert.match(partial, /data-crossdb-expand-scope/);
const expandButton = fakeControl();
owner.bind(rootWith({ '[data-crossdb-expand-scope]': [expandButton] }), partialPayload, config);
expandButton.fire('click');
assert.equal(expanded, true);

owner.reset();
assert.deepEqual(owner.snapshot(), {
  tab: 'overview',
  scope: 'all',
  module: 'all',
  feature: null,
  query: '',
});

process.stdout.write(JSON.stringify({
  complete_catalog_filter: true,
  partial_scope_disclosed: true,
  full_scope_handoff: true,
  no_duplicate_actions: true,
  one_main_chart: true,
  result_tabs: true,
}));
