'use strict';
// A hosted extraction saves in the clinician's own workspace on the server.
// The administrator's data folders are read-only: the request carries no
// out_dir, the page offers no folder to choose or create, a raw folder is not
// converted by the clinician, and the result names the workspace instead of a
// server path. A local build keeps its destination folder and its actions.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// tools/run_js_contracts.py passes the page owner, its hosted copy owner and the HTML escaper.
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const OWNER = 'screens-extraction.js';
const file = (arg, name) => fs.readFileSync(arg ? path.resolve(arg) : path.join(root, name), 'utf8');
const source = file(process.argv[2], OWNER);
const hostedCopy = file(process.argv[3], 'screens-extraction-hosted.js');
const html = file(process.argv[4], 'html-escape.js');

function section(begin, end) {
  const a = source.indexOf(begin), b = source.indexOf(end, a + begin.length);
  assert(a >= 0 && b > a, `${OWNER}: owner boundaries must exist (${begin.trim()})`);
  return source.slice(a, b);
}

const SERVER_PATH = '/home/clinic/instances/a/home/exports/easyicu_export_x';

function page(hosted, values = {}) {
  const ctx = vm.createContext({
    window: { EU_SETTINGS: { export_dir: '/Users/me/exports' }, t: (_en, zh) => zh },
    t: (_en, zh) => zh, icon: () => '', hostedServer: () => hosted, dataMode: () => 'real',
    exExportDir: null, exAdvExport: true, exFormat: 'parquet', exIncludeDefinitions: true, exMerge: 'separate',
    advRow: (k, v) => `${k}${v}`, switchEl: () => '', pathDisplay: x => x, exPath: '/data/eicu',
    ...values,
  });
  vm.runInContext(html, ctx);
  vm.runInContext(hostedCopy, ctx);
  ctx.escHtml = ctx.window.EU_HTML.esc;
  vm.runInContext(section('  function hostedCopy(', '  const DEFAULT_OBSERVATION_WINDOW_HOURS'), ctx);
  vm.runInContext(section('  function exportDestinationLabel() {', '  // Consume the Copilot handoff config'), ctx);
  return ctx;
}

function extractor(hosted, values = {}) {
  const posted = [];
  const c = page(hosted, {
    runModuleKeys: () => ['vitals'], cohortExportSupport: () => ({ ok: true }), abandonExtractionContinuity() {},
    repaint() {}, backgroundRepaint() {}, exScanResult: { db_key: 'eicu' }, selectedConceptPayload: () => ({}),
    exMaxPatients: 0, recommendedCohortContract: () => ({}), cohortContract: () => ({}), ...values,
  });
  c.window.EventSource = function EventSource() {};
  c.window.EU_EXTRACTION_JOB_CONTINUITY = { prepare: () => 'ticket', attach: () => true, isPending: () => false, abandon() {} };
  c.window.EU_API = { postJSON: (url, body) => { posted.push({ url, body }); return Promise.resolve({ job_id: 'j' }); } };
  vm.runInContext(section('  function runExtract(mode) {', '  function resetToCore()'), c);
  return { c, posted };
}

test('a hosted extraction request carries no output folder; a local one carries the chosen folder', () => {
  for (const hosted of [true, false]) {
    const { c, posted } = extractor(hosted);
    c.runExtract('recommended');
    assert.equal(posted.length, 1);
    assert.equal(posted[0].url, '/api/jobs/extract');
    if (hosted) assert.equal(Object.prototype.hasOwnProperty.call(posted[0].body, 'out_dir'), false);
    else assert.equal(posted[0].body.out_dir, '/Users/me/exports');
  }
});

test('a local extraction without a destination still stops for one; a hosted one never asks', () => {
  for (const hosted of [true, false]) {
    const { c, posted } = extractor(hosted);
    c.window.EU_SETTINGS = {};
    c.exAdvExport = false;
    c.runExtract('recommended');
    assert.equal(posted.length, hosted ? 1 : 0);
    assert.equal(c.exAdvExport, !hosted, 'a local run opens the export settings to choose a folder');
  }
});

test('hosted export settings name the workspace and offer no folder to browse or create', () => {
  const hosted = page(true);
  vm.runInContext(section('  function exportCfg(options) {', '  /* ---- compact action summary ---- */'), hosted);
  const cfg = hosted.exportCfg();
  assert(cfg.includes('服务器上你的工作区'));
  assert(cfg.includes('管理员准备的数据不会被改动'));
  assert(!cfg.includes('data-ex-export-browse'));
  assert(!cfg.includes('data-ex-export-create'));
  assert(!cfg.includes('本机'));

  const local = page(false);
  vm.runInContext(section('  function exportCfg(options) {', '  /* ---- compact action summary ---- */'), local);
  const localCfg = local.exportCfg();
  assert(localCfg.includes('data-ex-export-browse'));
  assert(localCfg.includes('data-ex-export-create'));
  assert(localCfg.includes('/Users/me/exports'));
});

test('a hosted raw folder points to the administrator instead of offering a conversion', () => {
  const raw = { db: 'eICU', layout: ['flat', '平铺'], source: 'raw', tables: 31, modules: 19, ready: false, size_hint: '5 GB' };
  for (const hosted of [true, false]) {
    const c = page(hosted, { exScanError: null, exScanResult: raw, runsWhere: () => '' });
    vm.runInContext(section('  function scanFacts() {', '  function scanErrorState() {'), c);
    vm.runInContext(section('  function scanResultState() {', '  // Driven by the live convert job'), c);
    const card = c.scanResultState();
    assert.equal(card.includes('data-ex-startconv'), !hosted);
    assert.equal(card.includes('请联系管理员'), hosted);
  }
});

test('a hosted result names the workspace, shows no server path and opens nothing', () => {
  const result = { out_dir: SERVER_PATH, files: [{ file: 'vitals.parquet', module: 'vitals', rows: 3 }], file_count: 1, total_rows: 3 };
  for (const hosted of [true, false]) {
    const c = page(hosted, {
      exportResult: result, exportJobId: 'j', exportRunModules: ['vitals'], EX_EXT: { parquet: 'parquet' },
      extractionSupportFiles: () => [], cohortScaleNote: () => '', modKeys: () => ['vitals'],
      exOutputNotice: '', exOutputError: '', exSyncNotice: '', exSyncError: '',
    });
    vm.runInContext(section('  function doneState() {', '  S.extraction = {'), c);
    const done = c.doneState();
    assert.equal(done.includes(SERVER_PATH), !hosted);
    assert.equal(done.includes('data-ex-open-output'), !hosted);
    assert.equal(done.includes('数据始终留在医院服务器上'), hosted);
    assert.equal(done.includes('全部留在你的机器上'), !hosted);
  }
});
