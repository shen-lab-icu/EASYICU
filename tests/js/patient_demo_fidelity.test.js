/* Executable fidelity contract for the offline Patient Review fallback. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const catalogSource = fs.readFileSync(process.argv[2], 'utf8');
const demoSource = fs.readFileSync(process.argv[3], 'utf8');
const demoDrilldownSource = fs.readFileSync(process.argv[4], 'utf8');
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
vm.runInContext(catalogSource, sandbox, { filename: process.argv[2] });
vm.runInContext(demoSource, sandbox, { filename: process.argv[3] });
vm.runInContext(demoDrilldownSource, sandbox, { filename: process.argv[4] });

const demo = context.VIZ_DEMO;
const demoDrilldown = context.VIZ_DEMO_DRILLDOWN;
const catalog = context.EU_CATALOG;
assert(demo && demoDrilldown && catalog);
assert.doesNotMatch(demoSource, /Math\.(sin|random)\s*\(/);
assert.deepEqual(Object.keys(demoDrilldown), ['buildPatientDrilldown']);

const drilldown = demoDrilldown.buildPatientDrilldown('demo_ent_3');
assert.equal(drilldown.selected.ref, 'demo_ent_3');
assert.equal(drilldown.data_tables.loaded_summary.review_features, catalog.totalConcepts);
assert.equal(drilldown.data_tables.table_previews.length, catalog.groups.length);

const lanes = demo.demoTimeLanes(0);
assert.equal(lanes.length, catalog.groups.length);
assert.equal(
  lanes.reduce((sum, lane) => sum + lane.features.length, 0),
  catalog.totalConcepts,
  'every catalog feature must remain discoverable in its owning module',
);
assert(lanes.every(lane => Array.isArray(lane.features) && lane.features.length > 0));
assert(lanes.some(lane => lane.features.some(feature => feature.status === 'metadata_only')));

const repeatA = JSON.stringify(demo.demoTimeLanes(2));
const repeatB = JSON.stringify(demo.demoTimeLanes(2));
assert.equal(repeatA, repeatB, 'the same entity must generate byte-stable trajectories');

const signal = (feature, entity = 0) => demo.demoSignal(feature, entity);
const hrRecovery = signal('hr', 0);
const mapRecovery = signal('map', 0);
const lactRecovery = signal('lact', 0);
assert.deepEqual(hrRecovery.times, demo.DEMO_CHART_HOURS);
assert(hrRecovery.times.some((value, index, rows) => index > 1 && value - rows[index - 1] !== rows[1] - rows[0]));
assert(hrRecovery.values[0] > hrRecovery.values.at(-1), 'shock-recovery HR should improve');
assert(mapRecovery.values[0] < mapRecovery.values.at(-1), 'shock-recovery MAP should improve');
assert(lactRecovery.values[0] > lactRecovery.values.at(-1), 'shock-recovery lactate should clear');

const hrDeterioration = signal('hr', 2);
const mapDeterioration = signal('map', 2);
const lactDeterioration = signal('lact', 2);
assert(hrDeterioration.values[0] < hrDeterioration.values.at(-1));
assert(mapDeterioration.values[0] > mapDeterioration.values.at(-1));
assert(lactDeterioration.values[0] < lactDeterioration.values.at(-1));

for (let entity = 0; entity < 5; entity += 1) {
  for (let timeIndex = 0; timeIndex < demo.DEMO_CHART_HOURS.length; timeIndex += 1) {
    const dbp = Number(demo.demoTableValue('dbp', entity, timeIndex));
    const map = Number(demo.demoTableValue('map', entity, timeIndex));
    const sbp = Number(demo.demoTableValue('sbp', entity, timeIndex));
    assert(dbp < map && map < sbp, `blood-pressure ordering failed at ${entity}/${timeIndex}`);
    assert(Number(demo.demoTableValue('spo2', entity, timeIndex)) <= 100);
    assert(Number(demo.demoTableValue('gcs', entity, timeIndex)) >= 3);

    const fio2 = Number(demo.demoTableValue('fio2', entity, timeIndex));
    const spo2 = Number(demo.demoTableValue('spo2', entity, timeIndex));
    const safi = Number(demo.demoTableValue('safi', entity, timeIndex));
    assert(Math.abs(safi - (100 * spo2 / fio2)) <= 0.25, `SaFi identity failed at ${entity}/${timeIndex}`);

    const ventModeValue = demo.demoTableValue('mech_vent', entity, timeIndex);
    const ventWindowValue = demo.demoTableValue('vent_ind', entity, timeIndex);
    const supplementalOxygen = demo.demoTableValue('supp_o2', entity, timeIndex);
    const advancedSupport = demo.demoTableValue('adv_resp', entity, timeIndex);
    assert([null, 'invasive', 'noninvasive'].includes(ventModeValue));
    assert.equal(Boolean(ventWindowValue), ventModeValue != null);
    assert.equal(Boolean(advancedSupport), ventModeValue != null);
    assert.equal(Boolean(supplementalOxygen), Boolean(ventWindowValue) || fio2 > 21);

    const controlMode = demo.demoTableValue('vent_mode', entity, timeIndex);
    const breathSequence = demo.demoTableValue('vent_breath_seq', entity, timeIndex);
    assert(['volume', 'pressure', 'dual_adaptive', 'proportional', 'unspecified', 'standby'].includes(controlMode));
    assert(['controlled', 'assisted', 'simv', 'spontaneous', 'standby'].includes(breathSequence));
    const controlledDrivingPressure = demo.demoTableValue('driving_pres_controlled', entity, timeIndex);
    assert.equal(controlledDrivingPressure != null, breathSequence === 'controlled');

    const sofa = Number(demo.demoTableValue('sofa', entity, timeIndex));
    const components = ['resp', 'coag', 'liver', 'cardio', 'cns', 'renal']
      .map(component => Number(demo.demoTableValue(`sofa_${component}`, entity, timeIndex)));
    assert.equal(sofa, components.reduce((sum, value) => sum + value, 0));
  }
}

assert(signal('temp', 0).values.length < hrRecovery.values.length);
assert(signal('crea', 0).values.length < hrRecovery.values.length);
assert(hrRecovery.point_count > hrRecovery.values.length);
assert.equal(hrRecovery.times.length, hrRecovery.values.length);
assert(hrRecovery.times.every((value, index, rows) => index === 0 || value > rows[index - 1]));
assert.equal(demo.demoTableValue('hr', 0, 3), hrRecovery.values[3]);
assert.equal(demo.demoTableValue('apache_iv', 0, 0), null);

const rrt = signal('rrt', 3);
assert(rrt.values.every(value => value === 0 || value === 1));
const ventMode = signal('mech_vent', 0);
assert(
  ventMode.values.every(value => value == null || value === 'invasive' || value === 'noninvasive'),
  'mechanical ventilation mode must remain categorical rather than a fabricated binary',
);
assert.equal(typeof demo.demoTableValue('mech_vent', 0, 0), 'string');
assert.equal(ventMode.min, null);
assert.equal(ventMode.max, null);
assert.equal(ventMode.mean, null);
assert.equal(demo.demoSignalDelta(ventMode.values), null);
const ventWindow = signal('vent_ind', 0);
assert(ventWindow.values.every(value => value === 0 || value === 1));
assert.equal(typeof demo.demoTableValue('vent_ind', 0, 0), 'boolean');
assert.doesNotMatch(demoSource, /endsWith\(['"]60['"]\)/);
const norepi = signal('norepi_rate', 0);
assert(norepi.values.every(value => Number.isFinite(Number(value)) && Number(value) >= 0));

process.stdout.write(JSON.stringify({
  all_catalog_features_grouped: true,
  clinically_correlated: true,
  deterministic: true,
  derived_scores_consistent: true,
  irregular_cadence: true,
  unmodeled_not_invented: true,
}));
