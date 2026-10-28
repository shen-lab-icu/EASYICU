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
const names = fs.readFileSync(path.join(process.argv[2] ? path.dirname(path.resolve(process.argv[2])) : root, 'database-names.js'), 'utf8');

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
    advRow: (k, v) => `${k}${v}`, switchEl: () => '', sampleCapCtl: () => '<cap>', pathDisplay: x => x, exPath: '/data/eicu',
    ...values,
  });
  vm.runInContext(names, ctx);
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
    const { c, posted } = extractor(hosted, { exFormat: 'excel' });
    c.runExtract('recommended');
    assert.equal(posted.length, 1);
    assert.equal(posted[0].url, '/api/jobs/extract');
    // Patient-level files stay on the server (no download), so a hosted run
    // is written in the analysis format whatever format was remembered.
    assert.equal(posted[0].body.format, hosted ? 'parquet' : 'excel');
    if (hosted) assert.equal(Object.prototype.hasOwnProperty.call(posted[0].body, 'out_dir'), false);
    else assert.equal(posted[0].body.out_dir, '/Users/me/exports');
    // A hosted export is listed under a name the clinician recognises.
    if (hosted) assert.match(posted[0].body.label, /^我的抽取 · eICU-CRD · \d\d-\d\d \d\d:\d\d$/);
    else assert.equal(Object.prototype.hasOwnProperty.call(posted[0].body, 'label'), false);
  }
});

test('a hosted export label names the database and the local time', () => {
  const c = page(true);
  const label = c.window.EU_EXTRACTION_HOSTED.exportLabel;
  assert.equal(label('miiv', new Date(2026, 9, 22, 9, 5)), '我的抽取 · MIMIC-IV · 10-22 09:05');
  assert.equal(label('eicu', new Date(2026, 0, 3, 23, 59)), '我的抽取 · eICU-CRD · 01-03 23:59');
  assert.equal(label('newdb', new Date(2026, 0, 3, 0, 0)), '我的抽取 · NEWDB · 01-03 00:00');
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
  // How many stays to take is an export setting, not a cohort criterion.
  assert(cfg.includes('抽取住院数上限<cap>'));

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
    assert.equal(done.includes('不能下载；可以在患者审阅、队列统计和研究对话中使用'), hosted);
    assert.equal(done.includes('全部留在你的机器上'), !hosted);
  }
});

test('a hosted folder is named by its folder name, never its server path', () => {
  for (const hosted of [true, false]) {
    const c = page(hosted);
    vm.runInContext(section('  function pathDisplay(path) {', '  function pathNeedsFolderChoice('), c);
    assert.equal(c.pathDisplay('/home/clinic/data/eicu-crd-2.0/'), hosted ? 'eicu-crd-2.0' : '/home/clinic/data/eicu-crd-2.0/');
  }
});

test('a finished extraction the job registered is re-read, not registered again', async () => {
  // Registering it again dropped the job's label ('我的抽取 · …' became 'EICU')
  // and its comparison choice (pilot1, 2026-10-22).
  for (const byJob of [true, false]) {
    const calls = [];
    const c = page(true, { backgroundRepaint() {}, localStorage: { setItem() {} } });
    c.window.EU_API = {
      hydrateWorkspaceRegistry: () => { calls.push('read'); return Promise.resolve({ sources: [] }); },
      registerWorkspaceSource: (p, o) => { calls.push(['register', p, o]); return Promise.resolve({ sources: [] }); },
    };
    vm.runInContext(section('  function rememberExportPath(', '  function exportDestinationLabel() {'), c);
    await c.rememberExportPath(SERVER_PATH, null, byJob);
    // The options object comes from the page's realm: compare its JSON.
    assert.equal(JSON.stringify(calls), JSON.stringify(byJob ? ['read'] : [['register', SERVER_PATH, { active: true, crossdb: true }]]));
  }
});
