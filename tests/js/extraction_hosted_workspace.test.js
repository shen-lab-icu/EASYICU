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
const minuteSource = file(null, 'screens-extraction-minute.js');
const studyContext = file(null, 'screens-extraction-study-context.js');

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

test('minute chooser sends only explicit, database-covered modules and renders the result receipt', async () => {
  let hosted = false;
  let stale = false;
  const groups = {
    hirid: [
      { id: 'subhour_signals', label: ['Bedside signals', '床旁信号'], signals: ['hr'], default_selected: false },
      { id: 'vent_native', label: ['Ventilator mechanics', '呼吸力学'], signals: ['peep_total'], default_selected: false },
    ],
    sic: [
      { id: 'subhour_signals', label: ['Bedside signals', '床旁信号'], signals: ['hr'], default_selected: false },
      { id: 'vent_native', label: ['Ventilator mechanics', '呼吸力学'], signals: [], unavailable_reason: 'no_whitelisted_signals' },
    ],
  };
  const context = vm.createContext({ window: { EU_API: {
    async getJSON(url) {
      const key = url.split('database=')[1];
      return { minute_resolution: { modules: groups[key] } };
    },
  } } });
  vm.runInContext(html, context);
  vm.runInContext(minuteSource, context);
  const owner = context.window.EU_EXTRACTION_MINUTE.create({
    t: en => en, escHtml: context.window.EU_HTML.esc,
    hosted: () => hosted, repaint() {}, backgroundRepaint() {},
  });
  const render = async key => {
    owner.render(key);
    await new Promise(resolve => setImmediate(resolve));
    return owner.render(key);
  };
  const initial = await render('hirid');
  assert.match(initial, /Minute resolution/);
  assert.match(initial, /data-ex-minute-module="vent_native"/);
  assert.doesNotMatch(initial, /checked|peep_total/);
  assert.deepEqual(Array.from(owner.keys()), []);

  const input = {
    dataset: { exMinuteModule: 'vent_native' }, checked: true,
    addEventListener(_event, listener) { this.toggle = listener; },
    focus() {},
  };
  owner.bind({ querySelector: () => null, querySelectorAll: () => [input] }, () => { stale = true; });
  input.toggle();
  assert.equal(stale, true);
  assert.deepEqual(Array.from(owner.keys()), ['vent_native']);
  assert.deepEqual(Object.keys(owner.payload(new Set(['vitals']))), []);
  assert.deepEqual(Array.from(owner.payload(new Set(['vent_native'])).vent_native), ['peep_total']);
  assert.match(owner.render('hirid'), /hirid<\/b> · peep_total/);
  assert.match(owner.render('hirid'), /larger and take longer/);
  assert.equal(owner.resultDetail({
    time_resolution: { interval_minutes: 1, signals: { peep_total: { rows: 10 } } },
  }), '1-minute grid · peep_total');
  assert.equal(owner.resultDetail({ module: 'vitals', rows: 10 }), '');

  const switched = await render('sic');
  assert.match(switched, /No covered signals/);
  assert.doesNotMatch(switched, /data-ex-minute-module="vent_native"/);
  assert.deepEqual(Array.from(owner.keys()), []);
  hosted = true;
  assert.match(owner.render('sic'), /Patient-level minute exports are unavailable on hosted instances/);
  assert.doesNotMatch(owner.render('sic'), /data-ex-minute-module/);
});

const coverage = { minute_resolution: { modules: [
  { id: 'haemo_native', label: ['Arterial line readings', '动脉压读数'], signals: ['sbp_invasive'] },
  { id: 'subhour_signals', label: ['Bedside monitor readings', '床旁监护读数'], signals: ['map_invasive'] },
] } };
const recommendedModules = ['demographics', 'vitals', 'chemistry', 'sofa2_score', 'sepsis3_sofa2', 'outcome'];
const settle = () => new Promise(resolve => setImmediate(resolve));

function minuteScreen(getOptions = async () => coverage) {
  const posted = [];
  const requests = [];
  const c = vm.createContext({
    t: en => en, icon: () => '', location: { hash: '#extraction' }, setTimeout() {},
    window: {
      EU_SETTINGS: { export_dir: '/fixture/output' },
      EU_CATALOG: { groupConcepts: { vitals: ['sbp'] } },
      EventSource() {},
      EU_API: {
        async scanPath() {
          return { ok: true, db_key: 'sic', db: 'SICdb', source: 'prepared', ready: true };
        },
        getJSON(url) { requests.push(url); return getOptions(); },
        async postJSON(url, body) {
          posted.push({ url, body: JSON.parse(JSON.stringify(body)) });
          return { job_id: 'fixture-job' };
        },
      },
      EU_EXTRACTION_JOB_CONTINUITY: {
        prepare: record => record, abandon() {}, isPending: () => false,
        attach(record, jobId) {
          c.window.EU_EXTRACTION_JOB_HOST.begin({ ...record, job_id: jobId });
          return true;
        },
      },
    },
  });
  for (const script of [names, html, hostedCopy, studyContext, minuteSource, source]) vm.runInContext(script, c);
  const screen = c.window.SCREENS.extraction;
  const render = () => screen.render();

  // Bind each interaction to a control actually rendered by the page.
  function interact(attribute, value, event = 'click', properties = {}) {
    const field = value === undefined ? attribute : attribute + '="' + value + '"';
    const tag = render().match(/<(?:input|button)\b[^>]*>/g)
      .find(candidate => new RegExp(' ' + field + '(?=[\\s>/])').test(candidate));
    assert.ok(tag, 'Missing control ' + field);
    assert.doesNotMatch(tag, /\sdisabled(?=[\s>])/);
    const selector = attribute === 'id' ? '#' + value : '[' + attribute + ']';
    const datasetKey = attribute.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    let handle;
    const node = {
      dataset: { [datasetKey]: value }, ...properties,
      addEventListener(name, listener) { if (name === event) handle = listener; },
      focus() {},
    };
    const query = selectors => selectors.split(', ').includes(selector) ? [node] : [];
    screen.afterRender({ querySelectorAll: query, querySelector: selectors => query(selectors)[0] });
    assert.ok(handle, 'Missing ' + event + ' handler for ' + field);
    handle();
  }

  async function connect(path = '/fixture/sic-converted') {
    interact('id', 'exPathInput', 'input', { value: path });
    interact('data-ex-analyze');
    await settle();
    interact('data-ex-usedata');
    await settle();
    render();
    await settle();
  }

  async function submit() {
    interact('data-ex-step-next', 'cohort');
    interact('data-ex-step-next', 'export');
    interact('data-ex-run', 'custom');
    await settle();
    assert.equal(posted.length, 1);
    assert.equal(posted[0].url, '/api/jobs/extract');
    return posted[0].body;
  }

  return { c, render, interact, connect, submit, requests };
}

test('an explicitly checked minute module survives the real extraction form submission', async () => {
  const screen = minuteScreen();
  await screen.connect();
  screen.interact('data-ex-core');
  screen.interact('data-ex-minute-module', 'haemo_native', 'change', { checked: true });
  assert.match(screen.render(), /SICdb<\/b> · sbp_invasive/);
  const body = await screen.submit();
  assert.equal(body.path, '/fixture/sic-converted');
  assert.equal(body.database, 'sic');
  assert.deepEqual(body.modules, [...recommendedModules, 'haemo_native']);
  assert.deepEqual(body.concepts, { vitals: ['sbp'], haemo_native: ['sbp_invasive'] });
});

test('Select all submits hourly modules without adding minute options', async () => {
  const screen = minuteScreen();
  await screen.connect();
  screen.interact('data-ex-selectall');
  const body = await screen.submit();
  assert.equal(body.modules.length, 20);
  assert.ok(body.modules.includes('vasopressors'));
  assert.ok(coverage.minute_resolution.modules.every(row => !body.modules.includes(row.id)));
  assert.deepEqual(body.concepts, { vitals: ['sbp'] });
});

test('Use recommended modules clears an explicit minute selection before submitting', async () => {
  const screen = minuteScreen();
  await screen.connect();
  screen.interact('data-ex-minute-module', 'haemo_native', 'change', { checked: true });
  screen.interact('data-ex-core');
  const body = await screen.submit();
  assert.deepEqual(body.modules, recommendedModules);
  assert.deepEqual(body.concepts, { vitals: ['sbp'] });
});

test('an empty coverage reply offers Retry and can recover', async () => {
  let attempts = 0;
  const screen = minuteScreen(async () => ++attempts === 1 ? {} : coverage);
  await screen.connect();
  const markup = screen.render();
  assert.ok(markup.includes('Could not load coverage'), 'An empty response must show a coverage error');
  assert.ok(!markup.includes('Checking database coverage'));
  screen.interact('data-ex-minute-retry');
  await settle();
  assert.match(screen.render(), /Arterial line readings/);
  assert.deepEqual(screen.requests, Array(2).fill('/api/extraction/module-options?database=sic'));
});

test('a new extraction and a fresh scan can reload coverage for the same database', async () => {
  let attempts = 0;
  const screen = minuteScreen(async () => {
    if (++attempts < 3) throw new Error('Coverage endpoint temporarily unavailable');
    return coverage;
  });
  await screen.connect();
  assert.match(screen.render(), /Could not load coverage/);
  screen.interact('data-ex-selectall');
  await screen.submit();
  screen.c.window.EU_EXTRACTION_JOB_HOST.applyEvent({ kind: 'extract', job_id: 'fixture-job' }, {
    type: 'end', status: 'error', error: 'Export destination unavailable',
  });
  screen.interact('data-ex-reset');
  screen.render();
  assert.equal(attempts, 2, 'Back to setup must retry coverage for the same database');
  await settle();
  assert.match(screen.render(), /Could not load coverage/);
  screen.c.window.__euExtractReset();
  await screen.connect('/fixture/second-sic-folder');
  assert.match(screen.render(), /Arterial line readings/);
  assert.equal(attempts, 3);
});

test('rendering with no scanned source removes loaded minute choices', async () => {
  const c = vm.createContext({ window: { EU_API: { getJSON: async () => coverage } } });
  vm.runInContext(minuteSource, c);
  const owner = c.window.EU_EXTRACTION_MINUTE.create({
    t: en => en, escHtml: text => text, hosted: () => false,
    repaint() {}, backgroundRepaint() {},
  });
  owner.render('sic');
  await settle();
  assert.match(owner.render('sic'), /Arterial line readings/);
  assert.doesNotMatch(owner.render(''), /data-ex-minute-module|data-ex-minute-retry/);
  assert.equal(owner.count(), 0);
});

test('starting source binding discards an in-flight coverage response', async () => {
  let finishPending;
  let repaints = 0;
  const screen = minuteScreen(() => new Promise(resolve => { finishPending = resolve; }));
  await screen.connect();
  screen.c.window.__euRender = () => { repaints++; };
  screen.c.window.EU_EXTRACTION_NATIVE_OWNER.beginSourceBinding();
  finishPending(coverage);
  await settle();
  assert.equal(repaints, 0, 'a disconnected lookup must not publish coverage');
});
