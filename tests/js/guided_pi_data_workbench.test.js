/* Executable contract for Copilot conversation resources and data previews. */
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
global.EU_VIZ_EMBEDDED_WORKBENCH = {
  render(payload, view) {
    const source = payload && payload.source || {};
    const selected = payload && payload.selected || {};
    return `<section data-native-workbench="${view}"><h2>${source.label || selected.label || ''}</h2></section>`;
  },
  mount(host, payload, view) { host.innerHTML = this.render(payload, view); },
};

load('src/easyicu/webserver/static/js/screens-guided-pi-resources.js');
load('src/easyicu/webserver/static/js/screens-guided-pi-data-preview.js');

test('data workbench resource keeps only immutable coordinates in the button', () => {
  const owner = global.EU_GUIDED_PI_RESOURCES.create({ esc: global.EU_HTML.esc });
  const resource = {
    kind: 'data_workbench_snapshot',
    view: 'cohort_summary',
    snapshot_sha256: 'd'.repeat(64),
    label: 'Cohort and filter flow',
  };
  assert.equal(owner.key(resource), `data-workbench:cohort_summary:${'d'.repeat(64)}`);
  const html = owner.button(resource);
  assert.match(html, /data-gpi-resource-kind="data_workbench_snapshot"/);
  assert.match(html, /data-gpi-resource-view="cohort_summary"/);
  assert.doesNotMatch(html, /source_path|stay_id|patient_id/);
});

test('cohort preview delegates to the native embedded workbench', () => {
  const html = global.EU_GUIDED_PI_DATA_PREVIEW.render({
    source: { label: 'MIMIC-IV export' },
    summary: { cohort_size: 120, modules: 2, mortality_pct: 12.5 },
    quality: { median_coverage_pct: 91.7 },
    eligibility_flow: {
      initial_count: 150,
      steps: [
        { label: 'All ICU stays', count: 150 },
        { label: 'Final cohort', count: 120, excluded: 30, excluded_pct_of_previous: 20 },
      ],
    },
    selected_feature_distributions: [{
      label: 'Heart rate', module: 'vitals', aggregation: 'entity_median',
      observed_pct: 91.7, bins: [{ low: 40, high: 80, count: 45 }],
      summary: { median: 72, min: 40, max: 140 },
    }],
  }, 'feature_distribution');
  assert.match(html, /data-native-workbench="feature_distribution"/);
  assert.match(html, /MIMIC-IV export/);
  assert.doesNotMatch(html, /gpi-data-bars/);
});

test('patient preview delegates without exposing direct identifier labels', () => {
  const html = global.EU_GUIDED_PI_DATA_PREVIEW.render({
    selected: { label: 'Entity 3', outcome: 'Survived' },
    time_lanes: [{
      label: 'Vitals', status: 'ready', signals: [{
        feature: 'hr', name: 'Heart rate', values: [80, 92, 85],
        current: 85, min: 80, max: 92, time_axis: { label_en: 'ICU hour' },
      }],
    }],
  }, 'patient_timeline');
  assert.match(html, /Entity 3/);
  assert.match(html, /data-native-workbench="patient_timeline"/);
  assert.doesNotMatch(html, /stay_id|subject_id|hadm_id/);
});
