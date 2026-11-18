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
const jobContinuity = file(null, 'screens-extraction-job-continuity.js');

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
    minuteModules: null,
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
    owner.render({ path: '/fixture/' + key, database: key, ready: true });
    await new Promise(resolve => setImmediate(resolve));
    return owner.markup();
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
  assert.match(owner.markup(), /hirid<\/b> · peep_total/);
  assert.match(owner.markup(), /larger and take longer/);
  assert.equal(owner.resultDetail({
    time_resolution: { interval_minutes: 1, signals: { peep_total: { rows: 10 } } },
  }), '1-minute grid · peep_total');
  assert.equal(owner.resultDetail({ module: 'vitals', rows: 10 }), '');

  const switched = await render('sic');
  assert.match(switched, /No covered signals/);
  assert.doesNotMatch(switched, /data-ex-minute-module="vent_native"/);
  assert.deepEqual(Array.from(owner.keys()), []);
  hosted = true;
  assert.match(owner.markup(), /Patient-level minute exports are unavailable on hosted instances/);
  assert.doesNotMatch(owner.markup(), /data-ex-minute-module/);
});

const coverage = { minute_resolution: { modules: [
  { id: 'haemo_native', label: ['Arterial line readings', '动脉压读数'], signals: ['sbp_invasive'] },
  { id: 'subhour_signals', label: ['Bedside monitor readings', '床旁监护读数'], signals: ['map_invasive'] },
] } };
const recommendedModules = ['demographics', 'vitals', 'chemistry', 'sofa2_score', 'sepsis3_sofa2', 'outcome'];
const settle = () => new Promise(resolve => setImmediate(resolve));

function minuteScreen(getOptions = async () => coverage, storage = new Map(), status = 'running') {
  const posted = [];
  const requests = [];
  let events;
  function EventSource() { events = this; this.close = () => {}; }
  const c = vm.createContext({
    t: en => en, icon: () => '', location: { hash: '#extraction' }, setTimeout() {}, EventSource,
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    window: {
      EU_SETTINGS: { export_dir: '/fixture/output' },
      EU_CATALOG: { groupConcepts: { vitals: ['sbp'] } },
      EventSource,
      EU_API: {
        async scanPath() {
          return { ok: true, db_key: 'sic', db: 'SICdb', source: 'prepared', ready: true };
        },
        getJSON(url) { requests.push(url); return getOptions(); },
        async loadJobSnapshot(id) {
          return { id, kind: 'extract', status, error: 'Archive volume became read-only' };
        },
        async postJSON(url, body) {
          posted.push({ url, body: JSON.parse(JSON.stringify(body)) });
          return { job_id: 'fixture-job' };
        },
      },
    },
  });
  for (const script of [names, html, hostedCopy, studyContext, minuteSource, source, jobContinuity]) vm.runInContext(script, c);
  const screen = c.window.SCREENS.extraction;
  const render = () => screen.render();
  let markup;
  let paints = 0;
  c.window.__euRender = () => {
    if (c.location.hash === '#guided') return;
    markup = render();
    paints++;
  };

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

  return {
    c, render, interact, connect, submit, requests,
    get markup() { return markup; },
    get paints() { return paints; },
    jobEvent(message) { events.onmessage({ data: JSON.stringify(message) }); },
  };
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

for (const [change, patch] of [
  ['feature edit', { modules: ['chemistry'] }],
  ['cohort edit', { cohort: { age_min: 30 } }],
]) {
  test(`a same-folder study ${change} replaces locally checked bedside signals`, async () => {
    const screen = minuteScreen();
    const { EU_EXTRACTION_STUDY_CONTEXT: bridge, EU_EXTRACTION_CONTEXT: context } = screen.c.window;
    const study = {
      id: 'overnight-lab-audit', revision: 3,
      data_source: { path: '/fixture/overnight-lab-audit', database: 'sic' },
      modules: ['blood_gas'],
    };
    bridge.hydrate(study);
    await screen.connect(study.data_source.path);
    screen.interact('data-ex-minute-module', 'subhour_signals', 'change', { checked: true });
    assert.ok(context.snapshot().modules.includes('subhour_signals'));

    bridge.hydrate({ ...study, ...patch, revision: 4 });
    const snapshot = context.snapshot();
    const body = await screen.submit();
    const modules = patch.modules || study.modules;
    assert.deepEqual({ snapshot: Array.from(snapshot.modules), submitted: body.modules, concepts: body.concepts },
      { snapshot: modules, submitted: modules, concepts: undefined });
    assert.equal(snapshot.data_source.path, study.data_source.path);
    assert.equal(screen.requests.length, 1);
  });
}

const arterialCoverage = { minute_resolution: { modules: [
  { ...coverage.minute_resolution.modules[0], signals: ['sbp_invasive', 'dbp_invasive'] },
  coverage.minute_resolution.modules[1],
  { id: 'vent_native', label: ['Ventilator settings', '呼吸机设置'], signals: [], unavailable_reason: 'no_whitelisted_signals' },
] } };

test('a saved waveform study selects its minute module after its folder is scanned', async () => {
  const screen = minuteScreen(async () => arterialCoverage);
  screen.c.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate({
    id: 'waveform-recheck', data_source: { database: 'sic' }, modules: ['haemo_native'],
  });
  await screen.connect('/fixture/waveform-recheck');
  const body = await screen.submit();
  assert.deepEqual(body.modules, ['haemo_native']);
  assert.deepEqual(body.concepts, { haemo_native: ['sbp_invasive', 'dbp_invasive'] });
});

test('an updated arterial study intersects its saved projection when coverage arrives', async () => {
  let resolveCoverage;
  const screen = minuteScreen(() => new Promise(resolve => { resolveCoverage = resolve; }));
  const { EU_EXTRACTION_STUDY_CONTEXT: bridge, EU_EXTRACTION_CONTEXT: context } = screen.c.window;
  const study = { id: 'diastolic-calibration', data_source: { database: 'sic' }, modules: ['haemo_native'] };
  bridge.hydrate(study);
  await screen.connect('/fixture/diastolic-calibration');
  bridge.hydrate({
    ...study,
    modules: ['haemo_native', 'subhour_signals', 'vent_native', 'unlisted_waveforms'],
    concepts: { haemo_native: ['dbp_invasive', 'expired_channel'], subhour_signals: ['expired_channel'] },
  });
  assert.deepEqual(Array.from(context.snapshot().modules), []);
  resolveCoverage(arterialCoverage);
  await settle();

  assert.match(screen.markup, /data-ex-minute-module="haemo_native" checked/);
  assert.match(screen.markup, /SICdb<\/b> · dbp_invasive</);
  assert.deepEqual(Array.from(context.snapshot().modules), ['haemo_native']);
  const body = await screen.submit();
  assert.deepEqual(body.modules, ['haemo_native']);
  assert.deepEqual(body.concepts, { haemo_native: ['dbp_invasive'] });
});

test('guided perfusion prefill carries its signal projection into extraction', async () => {
  const screen = minuteScreen(async () => arterialCoverage);
  let handoff = { config: { modules: ['haemo_native'], concepts: { haemo_native: ['dbp_invasive'] } } };
  screen.c.window.EU_GUIDED_HANDOFF = {
    take(route) {
      assert.equal(route, 'extraction');
      const next = handoff;
      handoff = null;
      return next;
    },
  };
  await screen.connect('/fixture/guided-perfusion');
  const body = await screen.submit();
  assert.deepEqual(body.modules, ['haemo_native']);
  assert.deepEqual(body.concepts, { haemo_native: ['dbp_invasive'] });
});

for (const status of ['running', 'failed']) {
  test(`a reloaded ${status} pressure export returns to setup with the original minute projection`, async () => {
    const storage = new Map();
    const originalCoverage = { minute_resolution: { modules: [
      { ...arterialCoverage.minute_resolution.modules[0], signals: ['dbp_invasive'] },
    ] } };
    const original = minuteScreen(async () => originalCoverage, storage);
    await original.connect('/fixture/pressure-archive');
    original.interact('data-ex-minute-module', 'haemo_native', 'change', { checked: true });
    const submitted = await original.submit();
    assert.deepEqual(submitted.modules, ['haemo_native']);
    assert.deepEqual(submitted.concepts, { haemo_native: ['dbp_invasive'] });

    const restored = minuteScreen(async () => arterialCoverage, storage, status);
    assert.ok(await restored.c.window.EU_EXTRACTION_JOB_CONTINUITY.ready);
    await settle();
    if (status === 'running') {
      restored.jobEvent({ type: 'end', status: 'failed', error: 'Archive volume became read-only' });
    }
    restored.interact('data-ex-reset');

    assert.match(restored.markup, /data-ex-minute-module="haemo_native" checked/);
    assert.deepEqual(Array.from(restored.c.window.EU_EXTRACTION_CONTEXT.snapshot().modules), ['haemo_native']);
    const body = await restored.submit();
    assert.deepEqual(body.modules, ['haemo_native']);
    assert.deepEqual(body.concepts, { haemo_native: ['dbp_invasive'] });
  });
}

test('a padded telemetry path keeps coverage and minute choices when the job attaches', async () => {
  const screen = minuteScreen();
  await screen.connect('  /fixture/telemetry/weekly  ');
  screen.c.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate({
    id: 'weekly-telemetry', data_source: { path: '/fixture/telemetry/weekly', database: 'sic' },
  });
  screen.interact('data-ex-minute-module', 'subhour_signals', 'change', { checked: true });
  const body = await screen.submit();
  assert.deepEqual(body.modules, ['subhour_signals']);
  screen.jobEvent({ type: 'end', status: 'failed', error: 'Archive volume became read-only' });
  screen.interact('data-ex-reset');
  assert.equal(screen.requests.length, 1);
  assert.match(screen.markup, /data-ex-minute-module="subhour_signals" checked/);
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

test('returning to setup preserves the failed lookup until a fresh scan', async () => {
  let attempts = 0;
  const screen = minuteScreen(async () => {
    if (++attempts === 1) throw new Error('Coverage endpoint temporarily unavailable');
    return coverage;
  });
  await screen.connect();
  assert.match(screen.render(), /Could not load coverage/);
  screen.interact('data-ex-selectall');
  await screen.submit();
  screen.jobEvent({ type: 'end', status: 'error', error: 'Export destination unavailable' });
  screen.interact('data-ex-reset');
  assert.match(screen.markup, /Could not load coverage/);
  assert.equal(attempts, 1, 'Back to setup keeps coverage for the connected folder');
  screen.c.window.__euExtractReset();
  await screen.connect('/fixture/second-sic-folder');
  assert.match(screen.markup, /Arterial line readings/);
  assert.equal(attempts, 2);
});

function minuteOwner(getJSON) {
  const c = vm.createContext({ window: { EU_API: { getJSON } } });
  vm.runInContext(minuteSource, c);
  return c.window.EU_EXTRACTION_MINUTE.create({
    t: en => en, escHtml: text => text, hosted: () => false,
    repaint() {}, backgroundRepaint() {},
  });
}

test('rendering without a source drops a checked minute module', async () => {
  const owner = minuteOwner(async () => coverage);
  const connected = { path: '/fixture/sic', database: 'sic', ready: true };
  owner.render(connected);
  await settle();
  assert.match(owner.markup(), /data-ex-minute-module="haemo_native"/);
  let select;
  const input = {
    dataset: { exMinuteModule: 'haemo_native' }, checked: true, focus() {},
    addEventListener(_event, listener) { select = listener; },
  };
  owner.bind({ querySelector: () => null, querySelectorAll: () => [input] }, () => {});
  select();
  assert.equal(owner.count(), 1);
  owner.render({ path: '', database: '', ready: false });
  assert.doesNotMatch(owner.markup(), /data-ex-minute-module/);
  assert.equal(owner.count(), 0);
  owner.render(connected);
  await settle();
  assert.match(owner.markup(), /data-ex-minute-module="haemo_native"/);
  assert.equal(owner.count(), 0, 'Reconnecting must leave the module unchecked');
});

test('removing the source also drops Retry for the failed lookup', async () => {
  const owner = minuteOwner(async () => { throw new Error('Coverage service is offline'); });
  owner.render({ path: '/fixture/unreachable-sic', database: 'sic', ready: true });
  await settle();
  assert.match(owner.markup(), /data-ex-minute-retry/);
  owner.render({ path: '', database: '', ready: false });
  const empty = owner.markup();
  assert.match(empty, /Database coverage is unavailable/);
  assert.doesNotMatch(empty, /data-ex-minute-retry/);
});

for (const pending of [true, false]) {
  test(`saving a validated monitor folder preserves ${pending ? 'pending coverage' : 'checked minute signals'}`, async () => {
    let finish;
    const screen = minuteScreen(() => pending ? new Promise(resolve => { finish = resolve; }) : Promise.resolve(coverage));
    const { EU_EXTRACTION_STUDY_CONTEXT: bridge, EU_EXTRACTION_NATIVE_OWNER: owner } = screen.c.window;
    const path = '/fixture/bedside-collection/winter';
    const study = {
      id: 'bedside-collection', revision: 1,
      data_source: pending ? null : { path: '/fixture/bedside-collection/autumn', database: 'sic' },
    };
    bridge.hydrate(study, 'sic');
    await screen.connect(path);
    if (pending) assert.match(screen.markup, /Checking database coverage/);
    else screen.interact('data-ex-minute-module', 'haemo_native', 'change', { checked: true });
    let patch;
    screen.c.window.EU_STUDY_CONTEXT = {
      update(value) { patch = value; },
      async persist() {
        const saved = { ...study, ...patch, revision: 2 };
        assert.ok(bridge.hydrate(saved, 'sic'));
        return saved;
      },
    };

    assert.equal((await owner.bindSourceToCopilot()).study_revision, 2);
    assert.equal(patch.data_source.path, path);
    if (pending) {
      finish(coverage);
      await settle();
    } else {
      assert.ok(screen.c.window.EU_EXTRACTION_CONTEXT.snapshot().modules.includes('haemo_native'));
      assert.match(screen.markup, /data-ex-minute-module="haemo_native" checked/);
    }
    assert.match(screen.markup, /Arterial line readings/);
    assert.equal(screen.requests.length, 1);
  });
}

test('removing the calibration source disconnects the screen and drops its pending coverage', async () => {
  let finishPending;
  const screen = minuteScreen(() => new Promise(resolve => { finishPending = resolve; }));
  const study = {
    id: 'monitor-calibration', revision: 1,
    data_source: { path: '/fixture/calibration/incoming', database: 'sic' },
  };
  const bridge = screen.c.window.EU_EXTRACTION_STUDY_CONTEXT;
  bridge.hydrate(study, 'sic');
  await screen.connect(study.data_source.path);
  assert.match(screen.markup, /Checking database coverage/);

  screen.c.location.hash = '#guided';
  bridge.hydrate({ ...study, revision: 2, data_source: null }, 'sic');
  const paints = screen.paints;
  screen.c.location.hash = '#extraction';
  finishPending(coverage);
  await settle();
  assert.equal(screen.paints, paints, 'An unbound calibration lookup must never repaint the workspace');
  const rendered = screen.render();
  assert.equal(screen.requests.length, 1, 'Rendering after unbinding must not repeat the old lookup');
  assert.match(rendered, /id="exPathInput"/);
  assert.doesNotMatch(rendered, /Checking database coverage|data-ex-minute-module/);
});

test('a source change in Copilot drops minute choices from the handoff before extraction renders', async () => {
  const screen = minuteScreen();
  const { EU_EXTRACTION_STUDY_CONTEXT: bridge, EU_EXTRACTION_CONTEXT: context } = screen.c.window;
  const study = {
    id: 'calibration-review', revision: 1,
    data_source: { path: '/fixture/calibration/ward-copy', database: 'sic' },
  };
  bridge.hydrate(study);
  await screen.connect(study.data_source.path);
  screen.interact('data-ex-minute-module', 'subhour_signals', 'change', { checked: true });
  assert.ok(context.snapshot().modules.includes('subhour_signals'));

  screen.c.location.hash = '#guided';
  const paints = screen.paints;
  const replacement = { path: '/fixture/calibration/verified', database: 'sic' };
  assert.ok(bridge.hydrate({ ...study, revision: 2, data_source: replacement }));
  const snapshot = context.snapshot();
  assert.equal(snapshot.data_source, null);
  assert.ok(!snapshot.modules.includes('subhour_signals'), 'The handoff must drop signals from the former folder');
  assert.equal(screen.paints, paints);
});

test('a running bedside export defers a source-only correction until the job ends', async () => {
  const screen = minuteScreen();
  const bridge = screen.c.window.EU_EXTRACTION_STUDY_CONTEXT;
  const study = {
    id: 'bedside-export', revision: 3, modules: ['vitals'],
    data_source: { path: '/fixture/bedside-export/working-copy', database: 'sic' },
  };
  bridge.hydrate(study);
  await screen.connect(study.data_source.path);
  await screen.submit();
  const revision = {
    ...study, revision: 4, data_source: { ...study.data_source, path: '/fixture/bedside-export/reconciled' },
  };
  assert.match(screen.markup, /Extracting/);
  assert.equal(bridge.hydrate(revision), null);
  assert.ok(bridge.hydrate(study), 'Reopening the active export with its original binding is safe');
  assert.equal(screen.c.window.EU_EXTRACTION_CONTEXT.snapshot().data_source.path, study.data_source.path);
  screen.jobEvent({ type: 'end', status: 'error', error: 'Destination quota reached' });
  assert.match(screen.markup, /Destination quota reached/);
  assert.ok(bridge.hydrate(revision));
  assert.match(screen.markup, /id="exPathInput"/);
  assert.equal(screen.requests.length, 1);
});

test('reattaching an active archive conversion drops coverage from before conversion', async () => {
  let finishPending;
  const screen = minuteScreen(() => new Promise(resolve => { finishPending = resolve; }));
  const study = {
    id: 'monitor-archive', revision: 1,
    data_source: { path: '/fixture/archive/quarterly', database: 'sic' },
  };
  const bridge = screen.c.window.EU_EXTRACTION_STUDY_CONTEXT;
  bridge.hydrate(study);
  await screen.connect(study.data_source.path);
  assert.match(screen.markup, /Checking database coverage/);
  const jobs = screen.c.window.EU_EXTRACTION_JOB_CONTINUITY;
  const conversion = jobs.prepare({ kind: 'convert', source: study.data_source });
  screen.c.location.hash = '#guided';
  const paints = screen.paints;
  jobs.attach(conversion, 'archive-convert');

  const deferred = bridge.hydrate({ ...study, revision: 2, data_source: null }, 'sic');
  screen.c.location.hash = '#extraction';
  finishPending(coverage);
  await settle();
  assert.equal(screen.paints, paints, 'A running conversion must discard the earlier coverage response');
  assert.doesNotMatch(screen.render(), /exMinuteHeading/);
  assert.equal(deferred, null);
  assert.equal(screen.requests.length, 1);
});

test('a replacement observations folder removes stale Retry and waits for a new scan', async () => {
  let attempts = 0;
  const screen = minuteScreen(async () => {
    if (++attempts === 1) throw new Error('Coverage lookup interrupted');
    return coverage;
  });
  const study = {
    id: 'night-shift-observations', revision: 4,
    data_source: { path: '/fixture/observations/initial', database: 'sic' },
  };
  const bridge = screen.c.window.EU_EXTRACTION_STUDY_CONTEXT;
  bridge.hydrate(study);
  await screen.connect(study.data_source.path);
  assert.match(screen.markup, /data-ex-minute-retry/);
  bridge.hydrate({ ...study, revision: 5 });
  assert.match(screen.markup, /data-ex-minute-retry/);
  assert.equal(attempts, 1);

  const corrected = '/fixture/observations/reconciled';
  bridge.hydrate({ ...study, revision: 6, data_source: { ...study.data_source, path: corrected } });
  assert.match(screen.markup, /id="exPathInput"/);
  assert.doesNotMatch(screen.markup, /data-ex-minute-retry|Checking database coverage/);
  assert.equal(attempts, 1);
  await screen.connect(corrected);
  assert.match(screen.markup, /data-ex-minute-module="haemo_native"/);
  assert.equal(attempts, 2);
});

for (const [transition, disconnect] of [
  ['starting source binding', win => win.EU_EXTRACTION_NATIVE_OWNER.beginSourceBinding()],
  ['resetting after a workspace source switch', win => win.__euExtractReset()],
]) {
  test(`${transition} discards an in-flight coverage response`, async () => {
    let finishPending;
    const screen = minuteScreen(() => new Promise(resolve => { finishPending = resolve; }));
    await screen.connect('/fixture/telemetry/ward-transfer');
    const paints = screen.paints;
    disconnect(screen.c.window);
    finishPending(coverage);
    await settle();
    assert.equal(screen.paints, paints, 'A disconnected lookup must not publish coverage');
    assert.match(screen.render(), /id="exPathInput"/);
  });
}
