/* Executable contract for the registered-export Cross-DB source action. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'en';

let paths = ['/exports/mimic-iv', '/exports/eicu', '/exports/mimic-iv'];
let runCount = 0;
let officialRunCount = 0;
let officialBindConfig = null;
let officialEnsureCount = 0;
global.EU_CROSSDB_SOURCE_HOST = {
  registeredPaths: () => paths,
  runRegistered: () => { runCount += 1; },
  officialPaths: () => ['/exports/mimic-iv', '/exports/eicu'],
  openOfficial: () => {},
  repaint: () => {},
  runOfficial: () => { officialRunCount += 1; },
};
global.EU_OFFICIAL_DEMO_SOURCES = {
  ensureLoaded(callback) {
    officialEnsureCount += 1;
    if (callback) callback(true);
  },
  bind(root, config) {
    officialBindConfig = config;
  },
  render() {
    return '<div data-official-demo-sources>official demos</div>';
  },
  snapshot() {
    return {
      catalog: {
        sources: [
          {
            id: 'mimiciv_demo',
            title: 'MIMIC-IV Clinical Database Demo',
            version: '2.2',
            database: 'miiv',
            scope: { patients: 100 },
            download: { size_label: '15.5 MB' },
            provenance: { provider: 'PhysioNet', license: { name: 'ODbL 1.0' } },
            status: { active: true, state: 'prepared' },
          },
          {
            id: 'eicu_demo',
            title: 'eICU Collaborative Research Database Demo',
            version: '2.0.1',
            database: 'eicu',
            scope: { icu_stays: 2500 },
            download: { size_label: '130.6 MB' },
            provenance: { provider: 'PhysioNet', license: { name: 'ODbL 1.0' } },
            status: { active: true, state: 'prepared' },
          },
        ],
      },
      error: null,
      job: null,
    };
  },
};

// Load the real escaping owner rather than stubbing it: a stub here would be
// one more copy of the contract this module was just made to stop re-rolling.
const moduleUnderTest = path.resolve(process.argv[2]);
require(path.join(path.dirname(moduleUnderTest), 'html-escape.js'));
require(moduleUnderTest);

function fakeButton(disabled) {
  const handlers = {};
  const attrs = disabled ? { 'aria-disabled': 'true' } : {};
  return {
    dataset: {},
    addEventListener(type, handler) { handlers[type] = handler; },
    getAttribute(name) { return attrs[name] || null; },
    click() {
      let prevented = false;
      let stopped = false;
      handlers.click({
        preventDefault() { prevented = true; },
        stopPropagation() { stopped = true; },
      });
      assert.equal(prevented, true);
      assert.equal(stopped, true);
    },
  };
}

const owner = global.EU_CROSSDB_SOURCE_CHOICE;
assert(owner);

const readyHtml = owner.render({ registryHtml: '<div data-test-registry>picker</div>' });
assert.match(readyHtml, /2 selected exports/);
assert.match(readyHtml, /data-test-registry/);
assert.match(readyHtml, /data-crossdb-run-registered/);
assert.doesNotMatch(readyHtml, /data-crossdb-run-registered aria-disabled="true"/);
const loadingHtml = owner.renderLoading();
assert.match(loadingHtml, /data-crossdb-registered-loading/);
assert.match(loadingHtml, /2 local exports · aggregate-only/);
assert.match(loadingHtml, /data-crossdb-cancel/);
const demoHtml = owner.renderDemo({
  sourceModeHtml: '<div data-test-source-mode>source mode</div>',
  syntheticHtml: '<div data-test-synthetic>synthetic fallback</div>',
});
assert.match(demoHtml, /data-crossdb-demo-source-choice/);
assert.match(demoHtml, /data-official-demo-sources/);
assert.match(demoHtml, /Start consistency check/);
assert.match(demoHtml, /MIMIC-IV Clinical Database Demo/);
assert.match(demoHtml, /eICU Collaborative Research Database Demo/);
assert.match(demoHtml, /data-crossdb-synthetic-fallback/);
assert.match(demoHtml, /UI rehearsal only/);

const readyButton = fakeButton(false);
const readyRoot = { querySelectorAll: () => [readyButton] };
owner.wire(readyRoot);
owner.wire(readyRoot);
readyButton.click();
assert.equal(runCount, 1, 'one click must invoke the registered-export host exactly once');

const officialButton = fakeButton(false);
const officialRoot = {
  querySelector(selector) {
    return selector === '[data-crossdb-demo-source-choice]' ? {} : null;
  },
  querySelectorAll(selector) {
    return selector === '[data-crossdb-run-official]' ? [officialButton] : [];
  },
};
owner.wire(officialRoot);
officialButton.click();
assert.equal(officialRunCount, 1);
assert.equal(officialEnsureCount, 1);
assert.equal(typeof officialBindConfig.openPrepared, 'function');

paths = ['/exports/mimic-iv'];
const blockedHtml = owner.render();
assert.match(blockedHtml, /1 selected export/);
assert.match(blockedHtml, /data-crossdb-run-registered aria-disabled="true"/);
assert.match(blockedHtml, /at least two EasyICU exports/);

const blockedButton = fakeButton(true);
owner.wire({ querySelectorAll: () => [blockedButton] });
blockedButton.click();
assert.equal(runCount, 1, 'fewer than two registered exports must not invoke the host');

process.stdout.write(JSON.stringify({
  official_pair_ready: true,
  official_run_count: officialRunCount,
  ready_sources: 2,
  run_count: runCount,
}));
