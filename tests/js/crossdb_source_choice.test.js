/* Executable contract for the registered-export Cross-DB source action. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'en';

let paths = ['/exports/mimic-iv', '/exports/eicu', '/exports/mimic-iv'];
let runCount = 0;
let openedOfficial = null;
let officialBindConfig = null;
let officialEnsureCount = 0;
global.EU_CROSSDB_SOURCE_HOST = {
  registeredPaths: () => paths,
  runRegistered: () => { runCount += 1; },
  officialPaths: () => [],
  openOfficial: sourceId => { openedOfficial = sourceId; },
  repaint: () => {},
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
            status: { active: false, registered: true, state: 'prepared' },
          },
          {
            id: 'eicu_demo',
            title: 'eICU Collaborative Research Database Demo',
            version: '2.0.1',
            database: 'eicu',
            scope: { icu_stays: 2500 },
            download: { size_label: '130.6 MB' },
            provenance: { provider: 'PhysioNet', license: { name: 'ODbL 1.0' } },
            status: { active: false, registered: true, state: 'prepared' },
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
// Public demos are prepared data like any other: the unprepared ones are
// offered under the picker, and there is no separate demo or synthetic path.
assert.equal(owner.renderDemo, undefined);
const offersHtml = owner.render({
  registryHtml: '<div data-test-registry>picker</div>',
  offersHtml: '<div data-official-demo-sources>official demos</div>',
});
assert.ok(offersHtml.indexOf('data-crossdb-run-registered') < offersHtml.indexOf('data-official-demo-sources'));
assert.doesNotMatch(offersHtml, /synthetic|Synthetic/);

const readyButton = fakeButton(false);
const readyRoot = { querySelectorAll: () => [readyButton] };
owner.wire(readyRoot);
owner.wire(readyRoot);
readyButton.click();
assert.equal(runCount, 1, 'one click must invoke the registered-export host exactly once');

const officialRoot = {
  querySelector(selector) {
    return selector === '[data-crossdb-source-choice]' ? {} : null;
  },
  querySelectorAll: () => [],
};
owner.wire(officialRoot);
assert.equal(officialEnsureCount, 1);
officialBindConfig.openPrepared('eicu_demo');
assert.equal(openedOfficial, 'eicu_demo');

paths = ['/exports/mimic-iv'];
const blockedHtml = owner.render();
assert.match(blockedHtml, /1 selected export/);
assert.match(blockedHtml, /data-crossdb-run-registered aria-disabled="true"/);
assert.match(blockedHtml, /at least two prepared datasets; public demos count/);

const blockedButton = fakeButton(true);
owner.wire({ querySelectorAll: () => [blockedButton] });
blockedButton.click();
assert.equal(runCount, 1, 'fewer than two registered exports must not invoke the host');

process.stdout.write(JSON.stringify({
  opened_official: openedOfficial,
  ready_sources: 2,
  run_count: runCount,
}));
