/* Executable contract for the shared Patient/Cohort/Cross-DB ECharts shell. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'en';
global.location = { hash: '#crossdb' };
global.addEventListener = () => {};
global.document = { documentElement: {} };
global.getComputedStyle = () => ({
  getPropertyValue(name) {
    return {
      '--accent': '#0f766e',
      '--hair': '#e2e8f0',
      '--ink': '#17202a',
      '--ink-4': '#64748b',
      '--surface': '#ffffff',
    }[name] || '';
  },
});

const observers = [];
global.ResizeObserver = function ResizeObserver(callback) {
  this.callback = callback;
  this.disconnected = false;
  this.observe = () => {};
  this.disconnect = () => { this.disconnected = true; };
  observers.push(this);
};

const calls = [];
global.echarts = {
  init(element, theme, options) {
    const call = { disposed: false, element, options, resizeCount: 0, value: null };
    calls.push(call);
    return {
      dispose() { call.disposed = true; },
      resize() { call.resizeCount += 1; },
      setOption(value) { call.value = value; },
    };
  },
};

/* Load the dependency-free owners first, exactly as index.html does: the
   chart owners destructure `esc` from window.EU_HTML at the top of their
   IIFE, and read window.EU_PALETTE for series colours and strokes. */
['html-escape.js', 'chart-palette.js'].forEach(owner => {
  require(path.join(path.dirname(path.resolve(process.argv[2])), owner));
});

require(path.resolve(process.argv[2]));
require(path.resolve(process.argv[3]));
require(path.resolve(process.argv[4]));

const shared = global.EU_ECHARTS;
const crossdb = global.EU_CROSSDB_CHARTS;
const cohort = global.EU_COHORT_CHARTS;
assert(shared && crossdb && cohort);
assert.equal(shared.VERSION, '6.1.0');
assert.equal(shared.available(), true);
assert.equal(shared.tooltip(() => '', 'item').renderMode, 'richText');
assert.equal(shared.axis({ type: 'value', name: 'bpm' }).name, 'bpm');

const helpers = {
  catalogFeatureMeta: () => ({ unit: 'bpm' }),
  esc: value => String(value),
  fmtPct: value => `${value}%`,
  t: english => english,
};
const densityItem = {
  row: {
    feature: 'hr',
    values: [
      { present: true, points: [{ x: 40, density: 0 }, { x: 80, density: 1 }, { x: 120, density: 0 }] },
      { present: true, points: [{ x: 45, density: 0 }, { x: 85, density: 0.8 }, { x: 125, density: 0 }] },
    ],
  },
};
crossdb.begin();
const densityHtml = crossdb.render(densityItem, ['MIMIC-IV', 'eICU'], helpers);
assert.match(densityHtml, /data-crossdb-echart=/);
const densityOption = crossdb.densityOption([
  { label: 'MIMIC-IV', value: densityItem.row.values[0] },
  { label: 'eICU', value: densityItem.row.values[1] },
], 'bpm', 'Aggregate density');
assert.equal(densityOption.series.length, 2);
assert.equal(densityOption.series[0].smooth, false);
assert.equal(densityOption.tooltip.renderMode, 'richText');
assert.equal(densityOption.xAxis.name, 'bpm');
assert.equal(densityOption.aria.show, true);

const densityId = densityHtml.match(/data-crossdb-echart="([^"]+)"/)[1];
const densityFallback = { removeAttribute() {} };
const densityElement = {
  getAttribute(name) { return name === 'data-crossdb-echart' ? densityId : null; },
  closest() { return { querySelector() { return densityFallback; } }; },
};
assert.equal(crossdb.mount({ querySelectorAll() { return [densityElement]; } }), 1);
assert.equal(calls[0].options.renderer, 'svg');
crossdb.begin();
assert.equal(calls[0].disposed, true);
assert.equal(observers[0].disconnected, true);

const survivalSpec = {
  description: 'Survival probability by group',
  xLabel: 'Days',
  yLabel: 'Survival probability',
  groupLabel: 'Group',
  eventsLabel: 'events',
  censoredLabel: 'censored',
  finalLabel: 'Final survival',
  atRiskLabel: 'Number at risk',
  horizon: 28,
  atRisk: {
    times: [0, 7, 14, 28],
    rows: [
      { label: 'Sepsis', values: [120, 96, 71, 40] },
      { label: 'Non-sepsis', values: [130, 121, 110, 88] },
    ],
  },
  groups: [
    {
      label: 'Sepsis', n: 120, events: 44, censored: 76,
      points: [{ time: 0, survival: 100 }, { time: 7, survival: 82 }],
      censorMarks: [{ time: 3, survival: 100 }, { time: 11, survival: 82 }],
    },
    {
      label: 'Non-sepsis', n: 130, events: 21, censored: 109,
      points: [{ time: 0, survival: 100 }, { time: 7, survival: 93 }],
      censorMarks: [{ time: 5, survival: 100 }],
    },
  ],
};
const survival = cohort.survivalOption(survivalSpec);
// Two step lines plus one censoring-tick series per group.
assert.equal(survival.series.length, 4);
assert.equal(survival.series[0].step, 'end');
assert.equal(survival.series[0].smooth, false);
assert.equal(survival.yAxis.min, 0);
assert.equal(survival.yAxis.max, 100);

// An explicit x max is what the number-at-risk columns are positioned against.
assert.equal(survival.xAxis.max, 28);

const censorSeries = survival.series.filter(s => String(s.id).startsWith('km-censor:'));
assert.equal(censorSeries.length, 2);
assert.equal(censorSeries[0].type, 'scatter');
assert.equal(censorSeries[0].symbol, 'rect');
assert.equal(censorSeries[0].silent, true);
assert.equal(censorSeries[0].data.length, 2);
// Ticks share the group's series name so the legend stays one entry per group.
assert.equal(censorSeries[0].name, 'Sepsis');

// A group with no censoring contributes no tick series.
const noCensor = cohort.survivalOption({
  ...survivalSpec,
  groups: [{ label: 'Only', points: [{ time: 0, survival: 100 }], censorMarks: [] }],
});
assert.equal(noCensor.series.length, 1);

// Censor ticks must not duplicate every group in the axis tooltip.
const tooltipText = survival.tooltip.formatter([
  { seriesId: 'km-line:0', seriesName: 'Sepsis', value: [7, 82] },
  { seriesId: 'km-censor:0', seriesName: 'Sepsis', value: [7, 82] },
]);
assert.equal(tooltipText.split('\n').filter(line => line.includes('Sepsis')).length, 1);

// The fail-closed table has to state the same facts as the plot.
const survivalFallback = cohort.survivalFallback(survivalSpec);
assert(survivalFallback.includes('censored'));
assert(survivalFallback.includes('Number at risk'));
assert(survivalFallback.includes('109'));

const bins = Array.from({ length: 13 }, (_, index) => String(index));
const matrix = bins.map((label, row) => ({
  label,
  cells: bins.map((_, column) => ({
    count: row === column ? 20 : 1,
    pct: row === column ? 10 : 0.5,
    intensity: row === column ? 1 : 0.05,
  })),
}));
const heatmap = cohort.heatmapOption({
  bins,
  matrix,
  mode: 'pct',
  xLabel: 'SOFA-2',
  yLabel: 'SOFA-1',
  valueLabel: '%',
  description: 'SOFA transition matrix',
});
assert.equal(heatmap.series[0].type, 'heatmap');
assert.equal(heatmap.series[0].label.show, false);
assert.equal(heatmap.dataZoom.length, 4);
assert.equal(heatmap.tooltip.renderMode, 'richText');

let failedChartDisposed = false;
let failedFallbackVisible = false;
let failedElementHidden = false;
global.echarts.init = () => ({
  dispose() { failedChartDisposed = true; },
  setOption() { throw new Error('renderer failed'); },
});
assert.equal(shared.mount(
  { setAttribute(name) { if (name === 'hidden') failedElementHidden = true; } },
  shared.baseOption('Failure contract'),
  { fallback: { removeAttribute(name) { if (name === 'hidden') failedFallbackVisible = true; } } },
), false);
assert.equal(failedChartDisposed, true);
assert.equal(failedElementHidden, true);
assert.equal(failedFallbackVisible, true);

process.stdout.write(JSON.stringify({
  cohort_heatmap: true,
  cohort_survival: true,
  crossdb_density: true,
  fail_closed_fallback: true,
  resize_dispose: true,
  shared_svg_renderer: true,
}));
