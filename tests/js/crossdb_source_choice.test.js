/* Executable contract for the registered-export Cross-DB source action. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'en';

let paths = ['/exports/mimic-iv', '/exports/eicu', '/exports/mimic-iv'];
let runCount = 0;
global.EU_CROSSDB_SOURCE_HOST = {
  registeredPaths: () => paths,
  runRegistered: () => { runCount += 1; },
};

require(path.resolve(process.argv[2]));

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
assert.doesNotMatch(loadingHtml, /data-crossdb-cancel/);

const readyButton = fakeButton(false);
const readyRoot = { querySelectorAll: () => [readyButton] };
owner.wire(readyRoot);
owner.wire(readyRoot);
readyButton.click();
assert.equal(runCount, 1, 'one click must invoke the registered-export host exactly once');

paths = ['/exports/mimic-iv'];
const blockedHtml = owner.render();
assert.match(blockedHtml, /1 selected export/);
assert.match(blockedHtml, /data-crossdb-run-registered aria-disabled="true"/);
assert.match(blockedHtml, /exports below/);

const blockedButton = fakeButton(true);
owner.wire({ querySelectorAll: () => [blockedButton] });
blockedButton.click();
assert.equal(runCount, 1, 'fewer than two registered exports must not invoke the host');

process.stdout.write(JSON.stringify({ ready_sources: 2, run_count: runCount }));
