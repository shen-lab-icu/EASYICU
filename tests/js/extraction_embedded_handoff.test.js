'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const ownerPath = process.argv[2];
assert.ok(ownerPath, 'screens-extraction-embedded.js path is required');

let html = '';
let syncClick = null;
let syncCalls = 0;
let rebindCalls = 0;
let receivedReceipt = null;
const syncButton = {
  disabled: false,
  textContent: '',
  addEventListener(type, handler) {
    if (type === 'click') syncClick = handler;
  },
};
const host = {
  isConnected: true,
  querySelector(selector) {
    if (selector === '[data-gpi-extraction-sync]') return syncButton;
    if (selector === '[data-gpi-extraction-embed]') return { scrollTop: 0 };
    return null;
  },
  querySelectorAll() { return []; },
  set innerHTML(value) { html = value; },
};

global.icon = () => '';
global.t = en => en;
global.escHtml = value => String(value == null ? '' : value);
global.window = {
  EU_EXTRACTION_NATIVE_OWNER: {
    render: () => '<section></section>',
    bind: () => {},
    isReal: () => true,
    isPreparedExport: () => false,
    syncToCopilot: async () => {
      syncCalls += 1;
      return {
        id: 'extract-job-1',
        database: 'MIMIC-IV',
        output_dir: '/Users/test/easyicu_export/run-1',
        data_file_count: 1,
        support_file_count: 5,
        total_rows: 42,
        receipt_kind: 'extraction_result',
        study_revision: 8,
      };
    },
  },
  EU_GUIDED_PI: {
    rebind: async () => { rebindCalls += 1; },
    notifyExtractionHandoff: receipt => { receivedReceipt = receipt; },
  },
};

vm.runInThisContext(fs.readFileSync(ownerPath, 'utf8'), { filename: ownerPath });

(async () => {
  window.EU_EXTRACTION_EMBEDDED_WORKSPACE.mount(host, {});
  assert.ok(syncClick, 'sync button must be interactive');
  await syncClick({ currentTarget: syncButton });
  await new Promise(resolve => setImmediate(resolve));

  assert.equal(syncCalls, 1);
  assert.equal(rebindCalls, 1);
  assert.equal(receivedReceipt.id, 'extract-job-1');
  assert.match(html, /Synced to Copilot/);
  assert.match(html, /StudyContext revision 8 now contains this extraction result/);
  assert.match(html, /The next Copilot turn reads that typed state/);
  console.log('extraction embedded handoff contract passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
