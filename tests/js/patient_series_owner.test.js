/* Executable boundary contract for Patient feature availability and time pairing. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {
  console,
  Date,
  Math,
  Map,
  Number,
  Set,
  EU_LANG: 'en',
  t(en) { return en; },
};
context.window = context;
const sandbox = vm.createContext(context);
process.argv.slice(2).forEach(path => {
  vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, { filename: path });
});

const demo = context.VIZ_DEMO;
const features = context.EU_PATIENT_FEATURES;
const series = context.EU_PATIENT_SERIES;
assert(demo && features && series);

const paired = series.numericSamples({
  values: [80, null, 90],
  times: [0, 1, 4],
});
assert.deepEqual(
  JSON.parse(JSON.stringify(paired)),
  { values: [80, 90], times: [0, 4] },
  'dropping a missing value must not move a later observation to the wrong time',
);

const lanes = features.catalogLanes(demo.demoTimeLanes(0));
const declaredFeatures = lanes.flatMap(lane => lane.features || []);
const numericSignals = lanes.flatMap(lane => lane.signals || [])
  .filter(signal => series.numericValues(signal).length >= 2);
assert.equal(declaredFeatures.length, context.EU_CATALOG.totalConcepts);
assert.equal(
  declaredFeatures.filter(feature => feature.trajectory).length,
  numericSignals.length,
  'the owner trajectory count must equal the number of drawable numeric signals',
);
assert.equal(numericSignals.length, 83);

for (const featureId of ['mech_vent', 'vent_mode', 'vent_breath_seq']) {
  const feature = declaredFeatures.find(row => row.feature === featureId);
  assert(feature, `${featureId} must remain discoverable`);
  assert.equal(feature.observed, true);
  assert.equal(feature.trajectory, false);
  assert.equal(feature.status, 'observed_categorical');
}

const coverage = {
  modules: [{
    module: 'vitals',
    features: [{
      feature: 'hr',
      status: 'observed',
      materialized: true,
      numeric: true,
      non_null_count: 250,
      trajectory_candidate: true,
      loadable: true,
    }],
  }, {
    module: 'chemistry',
    features: [{
      feature: 'pct',
      status: 'materialized_unknown',
      materialized: true,
      numeric: true,
      non_null_count: null,
      trajectory_candidate: false,
      loadable: true,
    }],
  }],
};
const coveredLanes = features.catalogLanes([], coverage, feature => (
  feature === 'hr' ? { loading: false, loaded: false } : {}
));
const heartRate = coveredLanes
  .flatMap(lane => lane.features || [])
  .find(row => row.feature === 'hr');
assert(heartRate);
assert.equal(heartRate.status, 'available_unloaded');
assert.equal(heartRate.loadable, true);
const coveredHtml = series.renderModulePanels(coveredLanes, {
  esc: value => String(value),
  fmtInt: value => String(value),
  t: en => en,
});
assert.match(coveredHtml, /data-patient-feature-load="hr"/);
assert.match(coveredHtml, /data-patient-feature-load="pct"/);
assert.match(coveredHtml, /data-patient-module-load="vitals"/);
assert.match(coveredHtml, /data-patient-inventory-toggle="open"/);
assert.match(coveredHtml, /observed · load chart/);
assert.match(coveredHtml, /materialized · verify on load/);
assert.match(
  coveredHtml,
  new RegExp(`${context.EU_CATALOG.totalConcepts} features across ${context.EU_CATALOG.groups.length} modules`),
);
const workspaceHtml = series.renderTimeSeriesWorkspace(
  { lanes: coveredLanes, mode: 'lanes' },
  { esc: value => String(value), fmtInt: value => String(value), t: en => en },
);
assert.match(workspaceHtml, /Module overview/);
assert.match(workspaceHtml, /Trajectory gallery/);
assert.match(workspaceHtml, /Cross-patient comparison/);
const loadedStaticLanes = features.catalogLanes([], coverage, feature => (
  feature === 'pct'
    ? {
      loaded: true,
      payload: {
        status: 'observed_numeric_static',
        observation: { current: 3.2, observation_count: 1 },
      },
    }
    : {}
));
const loadedStaticHtml = series.renderModulePanels(loadedStaticLanes, {
  esc: value => String(value),
  fmtInt: value => String(value),
  t: en => en,
});
assert.match(loadedStaticHtml, /value: 3.2/);
assert.match(loadedStaticHtml, /Module data loaded/);

process.stdout.write(JSON.stringify({
  all_features_discoverable: true,
  categorical_observations_distinct: true,
  numeric_trajectory_count_truthful: true,
  paired_time_filtering: true,
}));
