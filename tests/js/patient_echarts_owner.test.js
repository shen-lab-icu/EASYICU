/* Executable contract for the Patient Review ECharts widget owner. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const chartCalls = [];
const observers = [];

function FakeResizeObserver(callback) {
  this.callback = callback;
  this.disconnected = false;
  this.observe = () => {};
  this.disconnect = () => { this.disconnected = true; };
  observers.push(this);
}

const context = {
  console,
  Date,
  Intl,
  Map,
  Number,
  ResizeObserver: FakeResizeObserver,
  document: { documentElement: {} },
  getComputedStyle() {
    return {
      getPropertyValue(name) {
        return {
          '--accent': '#0f766e',
          '--hair': '#e2e8f0',
          '--ink': '#17202a',
          '--ink-4': '#64748b',
          '--surface': '#ffffff',
        }[name] || '';
      },
    };
  },
};
context.window = context;
context.echarts = {
  init(element, theme, options) {
    const record = {
      disposed: false,
      element,
      options,
      resizeCount: 0,
      setOptionValue: null,
    };
    chartCalls.push(record);
    return {
      dispose() { record.disposed = true; },
      resize() { record.resizeCount += 1; },
      setOption(value) { record.setOptionValue = value; },
    };
  },
};

vm.runInNewContext(source, vm.createContext(context), { filename: process.argv[2] });
const charts = context.EU_PATIENT_CHARTS;
assert(charts, 'Patient chart owner must publish window.EU_PATIENT_CHARTS');
assert.equal(charts.VERSION, '6.1.0');
assert.equal(charts.available(), true);

const signal = charts.signalOption({
  feature: 'hr',
  label: 'Heart rate',
  unit: 'bpm',
  values: [88, 92, 84],
  times: [0.2, 1, 48],
  timeAxis: { kind: 'relative_hours', label_en: 'ICU hour', unit: 'hour' },
  thresholds: [{ value: 50, label: 'Low' }, { value: 120, label: 'High' }],
});
assert.equal(signal.animation, false);
assert.equal(signal.aria.show, true);
assert.doesNotMatch(signal.aria.label.description, /focus/i);
assert.equal(signal.tooltip.renderMode, 'richText');
assert.equal(signal.xAxis.type, 'value');
assert.deepEqual(
  JSON.parse(JSON.stringify(signal.series[0].data)),
  [[0.2, 88], [1, 92], [48, 84]],
  'irregular source time spacing must survive option construction',
);
assert.equal(signal.series[0].smooth, false, 'clinical trajectories must not be visually smoothed');
assert.equal(signal.series[0].step, false);
assert.deepEqual(
  JSON.parse(JSON.stringify(signal.series[0].markLine.data.map(row => row.name))),
  ['Median', 'Low', 'High'],
);

const stepSignal = charts.signalOption({
  feature: 'norepi_rate',
  label: 'Norepinephrine rate',
  unit: 'mcg/kg/min',
  values: [0.2, 0.12, 0],
  times: [0, 2, 7],
  timeAxis: { kind: 'relative_hours' },
});
assert.equal(stepSignal.series[0].step, 'end', 'intervention rates must render as step functions');

const minuteSignal = charts.signalOption({
  feature: 'map',
  label: 'MAP',
  unit: 'mmHg',
  values: [70, 72, 68],
  times: [0, 60, 180],
  timeAxis: { kind: 'relative_minutes', unit: 'minute' },
});
assert.deepEqual(
  JSON.parse(JSON.stringify(minuteSignal.series[0].data.map(row => row[0]))),
  [0, 1, 3],
  'minute offsets must be normalized to hours',
);

const comparison = charts.comparisonOption({
  feature: 'hr',
  label: 'Heart rate',
  unit: 'bpm',
  traces: [
    {
      label: 'Entity 1',
      values: [80, 90],
      times: ['2026-01-01 00:00', '2026-01-01 01:00'],
      time_axis: { kind: 'datetime' },
    },
    {
      label: 'Entity 2',
      values: [70, 75],
      times: ['2026-02-10 06:00', '2026-02-10 08:00'],
      time_axis: { kind: 'datetime' },
    },
  ],
});
assert.equal(comparison.series.length, 2);
assert.deepEqual(
  JSON.parse(JSON.stringify(comparison.series[0].data.map(row => row[0]))),
  [0, 1],
);
assert.deepEqual(
  JSON.parse(JSON.stringify(comparison.series[1].data.map(row => row[0]))),
  [0, 2],
  'dated entity traces must align by elapsed hours without erasing spacing',
);
assert.equal(comparison.dataZoom[1].type, 'slider');

charts.begin();
const slot = charts.signalSlot({
  label: 'HR <img src=x onerror=alert(1)>',
  unit: 'bpm',
  values: [80, 82],
  times: [0, 1],
  timeAxis: { kind: 'relative_hours' },
}, 'fallback');
assert.doesNotMatch(slot, /<img/i);
assert.match(slot, /&lt;img/);
assert.match(slot, /data-patient-echart-fallback hidden/);
const id = slot.match(/data-patient-echart="([^"]+)"/)[1];
const element = {
  getAttribute(name) {
    return name === 'data-patient-echart' ? id : null;
  },
};
const root = {
  querySelectorAll(selector) {
    return selector === '[data-patient-echart]' ? [element] : [];
  },
};
assert.equal(charts.mount(root), 1);
assert.equal(chartCalls.length, 1);
assert.equal(chartCalls[0].options.renderer, 'svg');
assert.equal(chartCalls[0].setOptionValue.tooltip.renderMode, 'richText');

charts.begin();
assert.equal(chartCalls[0].disposed, true, 'repaint must dispose the previous chart instance');
assert.equal(observers[0].disconnected, true, 'repaint must disconnect the previous resize observer');

let chartHidden = false;
let fallbackShown = false;
context.echarts.init = () => { throw new Error('renderer unavailable'); };
context.console = { warn() {} };
const failedSlot = charts.signalSlot({
  label: 'MAP',
  values: [70, 72],
  times: [0, 1],
  timeAxis: { kind: 'relative_hours' },
}, '<svg data-fallback="true"></svg>');
const failedId = failedSlot.match(/data-patient-echart="([^"]+)"/)[1];
const fallback = { removeAttribute(name) { if (name === 'hidden') fallbackShown = true; } };
const failedElement = {
  getAttribute(name) { return name === 'data-patient-echart' ? failedId : null; },
  setAttribute(name) { if (name === 'hidden') chartHidden = true; },
  closest() { return { querySelector() { return fallback; } }; },
};
assert.equal(charts.mount({ querySelectorAll() { return [failedElement]; } }), 0);
assert.equal(chartHidden, true);
assert.equal(fallbackShown, true);

process.stdout.write(JSON.stringify({
  aria: true,
  elapsed_time_alignment: true,
  html_tooltip_disabled: true,
  irregular_spacing: true,
  local_svg_renderer: true,
  renderer_failure_fallback: true,
  repaint_disposes: true,
  step_interventions: true,
  thresholds: true,
}));
