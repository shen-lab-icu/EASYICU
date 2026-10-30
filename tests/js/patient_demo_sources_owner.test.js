/* Executable contract for the official Patient Review demo-source owner. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const ownerSource = fs.readFileSync(process.argv[2], 'utf8');
const catalogPayload = {
  ok: true,
  cache: { location: 'local EasyICU cache' },
  sources: [
    {
      id: 'mimic_iv_demo_v2_2',
      title: 'MIMIC-IV Clinical Database Demo',
      version: '2.2',
      database: 'miiv',
      description: '100 deidentified patients.',
      scope: { patients: 100 },
      download: {
        size_label: '15.5 MB',
        preferred_transport: 'github_release',
      },
      provenance: {
        provider: 'PhysioNet',
        landing_page: 'https://physionet.org/content/mimic-iv-demo/2.2/',
        license: { name: 'ODbL 1.0' },
      },
      status: {
        state: 'not_downloaded',
        registered: false,
        resume_available: true,
        partial_bytes: 4 * 1024 * 1024,
      },
    },
    {
      id: 'eicu_demo_v2_0_1',
      title: 'eICU Collaborative Research Database Demo',
      version: '2.0.1',
      database: 'eicu_demo',
      description: 'More than 2,500 deidentified ICU stays.',
      scope: { icu_stays: 2500 },
      download: { size_label: '130.6 MB' },
      provenance: {
        provider: 'PhysioNet',
        landing_page: 'https://physionet.org/content/eicu-crd-demo/2.0.1/',
        license: { name: 'ODbL 1.0' },
      },
      status: { state: 'prepared', registered: true, active: false },
    },
    {
      id: 'mimic_iii_demo_v1_4',
      title: 'MIMIC-III Clinical Database Demo',
      version: '1.4',
      database: 'mimic_demo',
      description: 'Official deidentified MIMIC-III demonstration release for testing the complete EasyICU concept and visualization workflow.',
      scope: { summary: '100 patients, 136 ICU stays', patients: 100, icu_stays: '136' },
      download: { size_label: '10.8 MB', preferred_transport: 'physionet', mirror_url: null },
      provenance: {
        provider: 'PhysioNet',
        landing_page: 'https://physionet.org/content/mimiciii-demo/1.4/',
        license: { name: 'ODbL 1.0' },
      },
      status: { state: 'not_downloaded', registered: false },
    },
  ],
};

let catalogLoads = 0;
let jobSnapshotLoads = 0;
let jobSnapshotMode = 'done';
let registryLoads = 0;
let registryFails = false;
const prepareStarts = [];
let dataModeContext = null;
const localStorageValues = new Map();
const scheduledTimers = [];
const context = {
  console,
  Promise,
  setTimeout(callback) {
    scheduledTimers.push(callback);
    return callback;
  },
  clearTimeout(callback) {
    const index = scheduledTimers.indexOf(callback);
    if (index >= 0) scheduledTimers.splice(index, 1);
  },
  localStorage: {
    getItem(key) { return localStorageValues.has(key) ? localStorageValues.get(key) : null; },
    setItem(key, value) { localStorageValues.set(key, String(value)); },
    removeItem(key) { localStorageValues.delete(key); },
  },
  setDataModeContext(value) {
    dataModeContext = value;
  },
  EU_API: {
    hydrateWorkspaceRegistry() {
      registryLoads += 1;
      return registryFails ? Promise.reject(new Error('registry unavailable')) : Promise.resolve({ sources: [] });
    },
    loadOfficialDemoSources() {
      catalogLoads += 1;
      return Promise.resolve(catalogPayload);
    },
    startOfficialDemoSourcePrepare(sourceId) {
      prepareStarts.push(sourceId);
      assert.equal(sourceId, 'mimic_iv_demo_v2_2');
      return Promise.resolve({ job_id: 'job-demo-1', status: 'running' });
    },
    loadJobSnapshot(jobId) {
      jobSnapshotLoads += 1;
      assert.equal(jobId, 'job-demo-1');
      if (jobSnapshotMode === 'failed') {
        const error = 'demo_source_download_failed: ' + JSON.stringify({
          code: 'demo_source_download_failed',
          detail: {
            cause: {
              code: 'unstructured_exception',
              detail: { message: '<urlopen error Tunnel connection failed: 503 Service Unavailable>' },
              structured: false,
              type: 'URLError',
            },
            phase: 'download',
          },
        });
        return Promise.resolve({
          id: jobId,
          status: 'failed',
          error,
          events: [
            { type: 'progress', phase: 'download', stage: 'starting', bytes_received: 0 },
            { type: 'end', status: 'failed', error },
          ],
        });
      }
      if (jobSnapshotMode === 'streaming') {
        return Promise.resolve({
          id: jobId,
          status: 'running',
          events: [
            {
              type: 'progress',
              phase: 'download',
              stage: 'streaming',
              bytes_received: 4 * 1024 * 1024,
              bytes_total: 16 * 1024 * 1024,
              resume_from_bytes: 2 * 1024 * 1024,
              download_rate_bps: 2 * 1024 * 1024,
              eta_seconds: 6,
            },
          ],
        });
      }
      return Promise.resolve({
        id: jobId,
        status: 'done',
        events: [
          { type: 'progress', phase: 'download', current: 1, total: 1 },
          { type: 'end', status: 'done' },
        ],
        result: { source_id: 'mimic_iv_demo_v2_2' },
      });
    },
  },
};
context.window = context;
const sandbox = vm.createContext(context);
vm.runInContext(ownerSource, sandbox, { filename: process.argv[2] });

const owner = context.EU_PATIENT_DEMO_SOURCES;
assert(owner);
assert.equal(context.EU_OFFICIAL_DEMO_SOURCES, owner);
const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  assert.equal(await owner.ensureLoaded(), true);
  const html = owner.render({ t: en => en, esc: value => String(value) });
  assert.match(html, /mimic_iv_demo_v2_2/);
  assert.match(html, /eicu_demo_v2_0_1/);
  assert.match(html, /data-demo-source-prepare="mimic_iii_demo_v1_4"/);
  assert.match(html, /MIMIC-III Clinical Database Demo/);
  assert.match(html, /10\.8 MB/);
  assert.match(html, /100 patients, 136 ICU stays/);
  assert.match(html, /Official dataset page/);
  assert.match(html, /Deidentified real records/);
  // Descriptions come from the owner's own copy, not the English release text.
  assert.match(html, /Deidentified MIMIC-IV demonstration release/);
  assert.doesNotMatch(html, /100 deidentified patients/);
  assert.doesNotMatch(html, />miiv<|>eicu_demo</);
  assert.match(html, /Download paused/);
  assert.match(html, /Resume download and prepare/);
  // The download transport is the host's concern, not the researcher's.
  assert.doesNotMatch(html, /GitHub fast mirror/);
  // No synthetic fallback: a public demo is prepared data, not a mode.
  assert.doesNotMatch(html, /synthetic/i);
  assert.match(html, /4\.0 MB saved/);
  assert.doesNotMatch(html, /data-gen/);
  assert.match(html, /data-demo-source-prepare="eicu_demo_v2_0_1"/);
  assert.match(html, /data-demo-source-open-after-prepare="true"/);
  assert.match(html, /Activate and open/);
  assert.doesNotMatch(html, /data-demo-source-open="eicu_demo_v2_0_1"/);
  assert.equal(owner.source('eicu_demo_v2_0_1').status.active, false);
  const pendingWithoutRegistry = owner.render({ t: en => en, esc: value => String(value) }, { pendingOnly: true, registryRows: [] });
  assert.match(pendingWithoutRegistry, /data-demo-source="eicu_demo_v2_0_1"/,
    'a server-registered demo stays visible until this page can show its registry row');
  const pendingWithRegistry = owner.render({ t: en => en, esc: value => String(value) }, {
    pendingOnly: true,
    registryRows: [{ ok: true, path: '/exports/eicu', source_kind: 'public_demo', source_id: 'eicu_demo_v2_0_1' }],
  });
  assert.doesNotMatch(pendingWithRegistry, /data-demo-source="eicu_demo_v2_0_1"/);

  const prepareClicks = new Map();
  const prepareButton = {
    getAttribute(name) {
      return name === 'data-demo-source-prepare' ? 'mimic_iv_demo_v2_2' : null;
    },
    addEventListener(name, handler) {
      if (name === 'click') prepareClicks.set('mimic_iv_demo_v2_2', handler);
    },
  };
  const concurrentButton = {
    getAttribute(name) {
      return name === 'data-demo-source-prepare' ? 'eicu_demo_v2_0_1' : null;
    },
    addEventListener(name, handler) {
      if (name === 'click') prepareClicks.set('eicu_demo_v2_0_1', handler);
    },
  };
  const rebindRoot = { querySelectorAll() { return []; } };
  const prepareConfig = {
    refresh() {
      // Patient Review repaints and immediately rebinds its owners. Terminal
      // cleanup must therefore happen before refresh or the stored job revives.
      owner.bind(rebindRoot, prepareConfig);
    },
  };
  owner.bind(
    {
      querySelectorAll(selector) {
        return selector === '[data-demo-source-prepare]'
          ? [prepareButton, concurrentButton]
          : [];
      },
    },
    prepareConfig,
  );
  assert.equal(typeof prepareClicks.get('mimic_iv_demo_v2_2'), 'function');
  prepareClicks.get('mimic_iv_demo_v2_2')({ preventDefault() {} });
  prepareClicks.get('eicu_demo_v2_0_1')({ preventDefault() {} });
  assert.deepEqual(prepareStarts, ['mimic_iv_demo_v2_2']);
  const runningHtml = owner.render({ t: en => en, esc: value => String(value) });
  assert.match(runningHtml, /Waiting for current preparation/);
  await tick();
  await tick();
  await tick();
  const snapshot = owner.snapshot();
  assert.equal(snapshot.job.status, 'done');
  assert.equal(jobSnapshotLoads, 1, 'terminal repaint must not restart the completed job');
  assert.equal(snapshot.error, null);
  assert(catalogLoads >= 2, 'successful preparation must refresh source status');
  assert.equal(registryLoads, 1, 'successful preparation must refresh the prepared-data registry');
  assert.equal(
    localStorageValues.has('easyicu_patient_official_demo_job_v1'),
    false,
    'terminal jobs must clear the reconnect pointer',
  );

  localStorageValues.set(
    'easyicu_patient_official_demo_job_v1',
    JSON.stringify({
      id: 'job-demo-1',
      sourceId: 'mimic_iv_demo_v2_2',
      openAfterPrepare: false,
    }),
  );
  const snapshotsBeforeResume = jobSnapshotLoads;
  owner.bind(
    { querySelectorAll() { return []; } },
    { refresh() {} },
  );
  await tick();
  await tick();
  assert(jobSnapshotLoads > snapshotsBeforeResume, 'page rebind must reconnect a remembered job');
  assert.equal(localStorageValues.has('easyicu_patient_official_demo_job_v1'), false);

  jobSnapshotMode = 'streaming';
  localStorageValues.set(
    'easyicu_patient_official_demo_job_v1',
    JSON.stringify({
      id: 'job-demo-1',
      sourceId: 'mimic_iv_demo_v2_2',
      openAfterPrepare: false,
    }),
  );
  owner.bind(
    { querySelectorAll() { return []; } },
    { refresh() {} },
  );
  await tick();
  await tick();
  const progressHtml = owner.render({ t: en => en, esc: value => String(value) });
  assert.match(progressHtml, /Resuming verified official download/);
  assert.match(progressHtml, /25%/);
  assert.match(progressHtml, /4\.0 MB \/ 16\.0 MB/);
  assert.match(progressHtml, /2\.0 MB\/s/);
  assert.match(progressHtml, /about 6s left/);
  assert.match(progressHtml, /class="official-demo-progress"/);
  assert.doesNotMatch(progressHtml, /NaN|Infinity/);
  jobSnapshotMode = 'done';
  assert.equal(scheduledTimers.length, 1);
  scheduledTimers.shift()();
  await tick();
  await tick();

  // A failed preparation stays on its card after the catalog refresh, as a
  // sentence with the lower-layer cause folded below; pressing the button
  // again clears it.
  jobSnapshotMode = 'failed';
  let failClick;
  const failButton = {
    getAttribute(name) {
      return name === 'data-demo-source-prepare' ? 'mimic_iv_demo_v2_2' : null;
    },
    addEventListener(name, handler) {
      if (name === 'click') failClick = handler;
    },
  };
  const catalogLoadsBeforeFailure = catalogLoads;
  owner.bind(
    { querySelectorAll(selector) { return selector === '[data-demo-source-prepare]' ? [failButton] : []; } },
    { refresh() {} },
  );
  failClick({ preventDefault() {} });
  await tick();
  await tick();
  await tick();
  const failedSnapshot = owner.snapshot();
  assert(catalogLoads > catalogLoadsBeforeFailure, 'a failed preparation still refreshes source status');
  assert.equal(failedSnapshot.error, null);
  assert.equal(failedSnapshot.failure.sourceId, 'mimic_iv_demo_v2_2');
  assert.equal(failedSnapshot.failure.phase, 'download');
  const failedHtml = owner.render({ t: en => en, esc: value => String(value) });
  assert.match(failedHtml, /data-demo-source-failure="mimic_iv_demo_v2_2"/);
  assert.match(failedHtml, /Could not download the archive from PhysioNet/);
  assert.match(failedHtml, /Tunnel connection failed: 503/);
  assert.doesNotMatch(failedHtml, /demo_source_download_failed/);
  assert.equal(failedHtml.match(/data-demo-source-failure=/g).length, 1, 'only the failed source carries the note');
  jobSnapshotMode = 'done';
  failClick({ preventDefault() {} });
  assert.equal(owner.snapshot().failure, null);
  assert.doesNotMatch(owner.render({ t: en => en, esc: value => String(value) }), /data-demo-source-failure/);
  await tick();
  await tick();
  await tick();

  // A registration refresh failure must stay visible and not open a stale
  // selection; the user can refresh the prepared-data list and try again.
  registryFails = true;
  let staleOpen = false;
  const staleOpenButton = {
    getAttribute(name) {
      if (name === 'data-demo-source-prepare') return 'mimic_iv_demo_v2_2';
      return name === 'data-demo-source-open-after-prepare' ? 'true' : null;
    },
    addEventListener(name, handler) { if (name === 'click') this.click = handler; },
  };
  owner.bind(
    { querySelectorAll(selector) { return selector === '[data-demo-source-prepare]' ? [staleOpenButton] : []; } },
    { refresh() {}, openPrepared() { staleOpen = true; } },
  );
  staleOpenButton.click({ preventDefault() {} });
  await tick(); await tick(); await tick();
  assert.equal(staleOpen, false, 'must not open using a stale prepared-data registry');
  assert.match(owner.render({ t: (en, zh) => zh, esc: value => String(value) }), /无法刷新已准备的数据/);
  registryFails = false;
  let retryClick;
  owner.bind({ querySelectorAll(selector) {
    return selector === '[data-demo-source-retry]'
      ? [{ addEventListener(name, handler) { if (name === 'click') retryClick = handler; } }]
      : [];
  } }, { refresh() {} });
  const beforeRetry = registryLoads;
  retryClick({ preventDefault() {} });
  await tick(); await tick(); await tick();
  assert.equal(registryLoads, beforeRetry + 1, 'retry refreshes the prepared-data registry');
  assert.equal(owner.snapshot().error, null);

  // Repainting while a terminal refresh is in progress starts a shared
  // catalog request; both callers must receive its completion callback.
  const normalLoadCatalog = context.EU_API.loadOfficialDemoSources;
  let releaseCatalog;
  context.EU_API.loadOfficialDemoSources = () => new Promise(resolve => { releaseCatalog = resolve; });
  let sharedCallbackCount = 0;
  const firstCatalog = owner.ensureLoaded(null, true);
  const secondCatalog = owner.ensureLoaded(() => { sharedCallbackCount += 1; }, true);
  releaseCatalog(catalogPayload);
  await firstCatalog;
  await secondCatalog;
  assert.equal(sharedCallbackCount, 1, 'joining an in-flight refresh must still run its callback');
  context.EU_API.loadOfficialDemoSources = normalLoadCatalog;

  catalogPayload.sources[1].status.active = true;
  const activeHtml = owner.render({ t: en => en, esc: value => String(value) });
  assert.match(activeHtml, /data-demo-source-open="eicu_demo_v2_0_1"/);
  assert.equal(owner.rememberOpened('eicu_demo_v2_0_1').id, 'eicu_demo_v2_0_1');
  assert.deepEqual(JSON.parse(JSON.stringify(dataModeContext)), {
    display_mode: 'demo',
    processing_mode: 'real',
    kind: 'official_demo',
    source_id: 'eicu_demo_v2_0_1',
    source_label: 'eICU Collaborative Research Database Demo v2.0.1',
  });
  assert.equal(
    owner.activeMetadata(
      [{ path: '/exports/eicu', label: 'eICU Collaborative Research Database Demo v2.0.1' }],
      '/exports/eicu',
    ).source_id,
    'eicu_demo_v2_0_1',
  );
  assert.equal(
    owner.activeMetadata(
      [{ path: '/exports/mimic', label: 'MIMIC-IV Clinical Database Demo v2.2' }],
      '/exports/mimic',
    ),
    null,
    'remembered demo provenance must not attach to a different active export',
  );
  const officialPair = owner.rememberPair([
    { ok: true, path: '/exports/mimic', label: 'MIMIC-IV Clinical Database Demo v2.2' },
    { ok: true, path: '/exports/eicu', label: 'eICU Collaborative Research Database Demo v2.0.1' },
    { ok: true, path: '/exports/local', label: 'Unrelated local export' },
  ]);
  assert.deepEqual(
    JSON.parse(JSON.stringify(officialPair.map(source => source.path))),
    ['/exports/mimic', '/exports/eicu'],
  );
  assert.deepEqual(JSON.parse(JSON.stringify(dataModeContext)), {
    display_mode: 'demo',
    processing_mode: 'real',
    kind: 'official_demo_pair',
    source_id: 'mimic_iv_demo_v2_2,eicu_demo_v2_0_1',
    source_label: 'MIMIC-IV Clinical Database Demo 2.2 + eICU Collaborative Research Database Demo 2.0.1',
  });

  let openedSource = null;
  let openClick;
  const openButton = {
    getAttribute(name) {
      return name === 'data-demo-source-open' ? 'eicu_demo_v2_0_1' : null;
    },
    addEventListener(name, handler) {
      if (name === 'click') openClick = handler;
    },
  };
  owner.bind(
    {
      querySelectorAll(selector) {
        return selector === '[data-demo-source-open]' ? [openButton] : [];
      },
    },
    { openPrepared(sourceId) { openedSource = sourceId; } },
  );
  openClick({ preventDefault() {} });
  assert.equal(openedSource, 'eicu_demo_v2_0_1');

  process.stdout.write(JSON.stringify({
    done_status_supported: true,
    official_sources_rendered: true,
    progress_telemetry_visible: true,
    prepare_single_flight: true,
    refresh_reconnect_supported: true,
    prepared_source_openable: true,
    provenance_visible: true,
    shared_owner_contract: true,
    official_pair_resolved: true,
    user_mode_remains_demo: true,
    no_synthetic_fallback: true,
    prepare_failure_on_card: true,
  }));
})().catch(error => {
  process.stderr.write(String(error && error.stack || error));
  process.exitCode = 1;
});
