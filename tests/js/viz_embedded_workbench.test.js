/* Executable contract for reusing the native data-workbench renderers in Copilot. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '../..');
function load(relative) {
  vm.runInThisContext(fs.readFileSync(path.join(ROOT, relative), 'utf8'), { filename: relative });
}

global.window = global;
global.EU_LANG = 'en';
global.EU_HTML = { esc: value => String(value == null ? '' : value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;') };
global.t = (en) => en;
global.icon = name => `<i>${name}</i>`;

load('src/easyicu/webserver/static/js/screens-viz-context.js');

test('visualization context keeps snapshot and hydration behind one explicit contract', () => {
  const hydrated = [];
  global.EU_VIZ_CONTEXT_OWNER.init({
    activePath: () => '/private/export',
    sources: () => [{ id: 'mimic', path: '/private/export', label: 'MIMIC-IV' }],
    defaultExportPath: () => '/private/export',
    patient: () => ({ summary: { entities: 3 }, module_profiles: [{ module: 'vitals' }] }),
    cohort: () => ({ summary: { cohort_size: 120 }, feature_selection: { selected: [] }, feature_catalog: { modules: [] } }),
    cohortComparison: () => 'outcome',
    cohortOutcome: () => 'mort_28d',
    crossdb: () => ({ source_count: 2, shared_modules: ['vitals'], selection_receipt: { selection_digest: 'abc' } }),
    hydrate: (route, payload) => hydrated.push([route, payload]),
    cohortPanels: { coverage: () => '<section data-native-cohort="coverage"></section>' },
    cohortMount: () => 1,
    patientSeriesHelpers: () => ({ native: true }),
    crossdbResultsConfig: repaint => ({ native: true, repaint }),
  });

  assert.equal(global.EU_VIZ_CONTEXT.snapshot('cohort').cohort.cohort_size, 120);
  assert.match(global.EU_VIZ_CONTEXT.renderCohortPanel({}, 'coverage'), /data-native-cohort="coverage"/);
  assert.equal(global.EU_VIZ_CONTEXT.hydratePreview('patient', { selected: { label: 'Entity 3' } }), true);
  assert.equal(global.EU_VIZ_CONTEXT.hydratePreview('settings', {}), false);
  assert.deepEqual(hydrated, [['patient', { selected: { label: 'Entity 3' } }]]);
});

global.EU_PATIENT_SERIES = {
  renderTimeSeriesWorkspace: payload => `<section data-native-patient="${payload.mode}">${payload.selected.label}</section>`,
};
global.EU_PATIENT_CHARTS = { mount: () => 1 };
global.EU_CROSSDB_RESULTS = {
  render: payload => `<section data-native-crossdb>${payload.source_count}</section>`,
  bind: () => {},
  mount: () => 1,
};

load('src/easyicu/webserver/static/js/screens-viz-embedded.js');

test('embedded preview delegates patient, cohort, and cross-db bodies to native owners', () => {
  const patient = global.EU_VIZ_EMBEDDED_WORKBENCH.render({
    source: { label: 'MIMIC-IV' },
    selected: { label: 'Entity 3', ref: 'browser-only-ref' },
    time_lanes: [],
  }, 'patient_timeline', { patientMode: 'single' });
  assert.match(patient, /data-native-patient="single"/);
  assert.match(patient, /Open full Patient Review/);
  assert.doesNotMatch(patient, /browser-only-ref/);

  const cohort = global.EU_VIZ_EMBEDDED_WORKBENCH.render({
    source: { label: 'MIMIC-IV' }, summary: { cohort_size: 120 },
  }, 'cohort_summary', { cohortPanel: 'coverage' });
  assert.match(cohort, /data-native-cohort="coverage"/);
  assert.match(cohort, /Native Cohort Statistics/);

  const crossdb = global.EU_VIZ_EMBEDDED_WORKBENCH.render({
    source_count: 2, sources: [{ label: 'MIMIC-IV' }, { label: 'eICU' }],
  }, 'crossdb_comparison', {});
  assert.match(crossdb, /data-native-crossdb/);
  assert.match(crossdb, /Native Cross-DB workspace/);
});

test('guided preview delegates to the embedded native-workbench owner', () => {
  load('src/easyicu/webserver/static/js/screens-guided-pi-data-preview.js');
  const html = global.EU_GUIDED_PI_DATA_PREVIEW.render({
    source_count: 2, sources: [{ label: 'MIMIC-IV' }, { label: 'eICU' }],
  }, 'crossdb_comparison');
  assert.match(html, /data-native-crossdb/);
  assert.doesNotMatch(html, /gpi-data-table-row/);
});

test('embedded mount keeps native downstream actions read-only', () => {
  const source = fs.readFileSync(path.join(ROOT, 'src/easyicu/webserver/static/js/screens-viz-embedded.js'), 'utf8');
  assert.match(source, /querySelectorAll\('\[data-study-handoff\],\[data-nav\]'\)/);
  assert.match(source, /control\.hidden = true/);
  assert.match(source, /data-gpi-open-native/);
});
