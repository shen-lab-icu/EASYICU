/* Executable state contract for the Cross-DB setup/scan owner. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const context = { console, EU_DATA: 'real' };
context.window = context;
vm.runInNewContext(source, vm.createContext(context), { filename: process.argv[2] });

const setup = context.EU_CROSSDB_SETUP;
assert(setup, 'Cross-DB setup owner must publish window.EU_CROSSDB_SETUP');
assert.equal(setup.sourceMethod(), 'registered');

function esc(value) {
  return String(value == null ? '' : value).replace(/[&<>"']/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[character]));
}

function helpers() {
  return {
    esc,
    fmtInt: value => String(value == null ? '—' : value),
    icon: () => '',
    progressMessage: value => String(value || ''),
    statusLabel: value => String(value || ''),
    t: english => english,
    term: value => String(value || ''),
  };
}

function rawRunTag(html) {
  const match = String(html).match(/<button[^>]*data-crossdb-run-raw[^>]*>/);
  assert(match, 'real setup must render one raw-run button');
  return match[0];
}

function fakeButton() {
  const attributes = new Map();
  const handlers = {};
  return {
    dataset: {},
    addEventListener(type, handler) { handlers[type] = handler; },
    getAttribute(name) { return attributes.has(name) ? attributes.get(name) : null; },
    setAttribute(name, value) { attributes.set(name, String(value)); },
    removeAttribute(name) { attributes.delete(name); },
    click() {
      assert.equal(typeof handlers.click, 'function', 'button must receive a click handler');
      handlers.click({
        preventDefault() {},
        stopPropagation() {},
        target: this,
      });
    },
  };
}

(async () => {
  let errorMessage = null;
  let repaintCount = 0;
  let resolveStaleScan;
  const scanCalls = [];
  const detectedPayload = {
    ok: true,
    detected: [
      {
        key: 'miiv',
        label: 'MIMIC-IV <img src=x onerror=alert(1)>',
        folder_name: '<script>alert(1)</script>',
      },
      { key: 'eicu', label: 'eICU-CRD', folder_name: 'eicu' },
    ],
    unrecognized_folders: ['<svg onload=alert(1)>'],
    unrecognized_count: 1,
    aliases: {
      miiv: { label: '<b>MIMIC-IV</b>', aliases: ['<i>miiv</i>'] },
    },
  };
  const config = {
    api: {
      scanCrossdbRawRoot(body) {
        scanCalls.push(body);
        if (body.data_root === '/scan-a') {
          return new Promise(resolve => { resolveStaleScan = resolve; });
        }
        return Promise.resolve(detectedPayload);
      },
    },
    getError() { return errorMessage; },
    helpers: helpers(),
    repaint() { repaintCount += 1; },
    registryHtml() { return ''; },
    setError(value) { errorMessage = value; },
  };

  assert.equal(setup.featureScope(), 'all', 'raw comparison defaults to the complete catalog');
  setup.setSelectedKeys(['miiv', 'eicu']);
  const defaultSetupHtml = setup.renderReal(config);
  assert.match(defaultSetupHtml, /data-crossdb-source-path="registered"/);
  assert.doesNotMatch(defaultSetupHtml, /data-crossdb-run-raw/);
  setup.setSourceMethod('raw');
  assert.equal(setup.sourceMethod(), 'raw');
  setup.changeRawRoot('/scan-a');
  const staleScan = setup.scan('/scan-a', config);
  assert.equal(typeof resolveStaleScan, 'function');
  setup.changeRawRoot('/scan-b');
  resolveStaleScan(detectedPayload);
  assert.equal(await staleScan, false, 'a scan from the previous root must be ignored');
  assert.equal(setup.snapshot(config).scanReady, false);

  assert.equal(await setup.scan('/scan-b', config), true);
  assert.equal(scanCalls.length, 2);
  assert.deepEqual(Array.from(scanCalls[1].databases), ['miiv', 'eicu']);
  assert.equal(setup.snapshot(config).scanReady, true);
  assert.doesNotMatch(rawRunTag(setup.renderReal(config)), /aria-disabled/);

  setup.setSelectedKeys(['miiv', 'eicu', 'sic']);
  assert.equal(setup.snapshot(config).scanReady, false, 'a newly selected missing database must fail closed');
  assert.doesNotMatch(rawRunTag(setup.renderReal(config)), /aria-disabled/, 'the primary action re-checks the folder instead of requiring a separate scan click');
  setup.setSelectedKeys(['miiv', 'eicu']);
  assert.equal(setup.snapshot(config).scanReady, true);
  assert.equal(scanCalls.length, 2, 'selection changes must re-use the root scan instead of re-scanning');

  const escapedHtml = setup.renderReal(config);
  assert.doesNotMatch(escapedHtml, /<img src=x|<script>alert|<svg onload|<i>miiv<\/i>/i);
  assert.match(escapedHtml, /&lt;img/);
  assert.match(escapedHtml, /&lt;script&gt;/);
  assert.match(escapedHtml, /&lt;svg/);

  assert.equal(setup.sourceIdentity(), 'eicu,miiv');
  assert.equal(setup.identityKeys({ source_identity: 'miiv,eicu' }).length, 0, 'resume identity must already be canonical');
  assert.equal(setup.identityKeys({ source_identity: 'eicu,unknown' }).length, 0);
  assert.equal(setup.identityKeys({ source_identity: 'miiv' }).length, 0);
  assert.equal(setup.acceptResume({
    raw_root: '/other-root',
    source_identity: 'eicu,miiv',
    sample_mode: 'standard',
  }), false, 'a saved job from another root must not attach');
  assert.equal(setup.acceptResume({
    raw_root: '/scan-b',
    source_identity: 'eicu,miiv',
    sample_mode: 'standard',
    feature_scope: 'all_catalog',
  }), true);
  assert.equal(setup.matchesSource({ raw_root: '/scan-b', source_identity: 'eicu,miiv' }), true);
  const standard = setup.snapshot(config);
  assert.equal(standard.sampleMode, 'standard');
  assert.equal(standard.sampleProfile.maxPatients, 300);
  assert.equal(standard.sampleProfile.sampleSize, 1500);
  assert.equal(standard.featureScope, 'all');

  const featureScopeButton = fakeButton();
  featureScopeButton.dataset.crossdbFeatureScope = 'core';
  setup.bind({
    querySelector() { return null; },
    querySelectorAll(selector) {
      return selector === '[data-crossdb-feature-scope]' ? [featureScopeButton] : [];
    },
  }, {
    helpers: helpers(),
    repaint() { repaintCount += 1; },
  });
  featureScopeButton.click();
  assert.equal(setup.featureScope(), 'core');
  assert.match(setup.renderReal(config), /data-crossdb-feature-scope="all"/);
  setup.setFeatureScope('all');

  const rawRunButton = fakeButton();
  const rawRunRoot = {
    querySelector(selector) {
      if (selector === '[data-crossdb-root]') return { value: '/scan-b' };
      return null;
    },
    querySelectorAll(selector) {
      if (selector === '[data-crossdb-run-raw]') return [rawRunButton];
      return [];
    },
  };
  let rawRunSnapshot = null;
  setup.setView('idle');
  setup.bind(rawRunRoot, {
    helpers: helpers(),
    repaint() { repaintCount += 1; },
    runRaw(done, options) {
      rawRunSnapshot = options.setup;
      done(true);
    },
  });
  rawRunButton.click();
  assert.equal(rawRunSnapshot.rawRoot, '/scan-b');
  assert.deepEqual(Array.from(rawRunSnapshot.selectedKeys), ['miiv', 'eicu']);
  assert.equal(rawRunSnapshot.featureScope, 'all');
  assert.equal(setup.view(), 'loaded', 'a successful raw completion must keep the owner loaded');

  setup.setView('idle');
  setup.changeRawRoot('/scan-c');
  setup.setSelectedKeys(['miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic']);
  let oneClickSnapshot = null;
  const oneClickButton = fakeButton();
  setup.bind({
    querySelector(selector) {
      if (selector === '[data-crossdb-root]') return { value: '/scan-c' };
      return null;
    },
    querySelectorAll(selector) {
      return selector === '[data-crossdb-run-raw]' ? [oneClickButton] : [];
    },
  }, {
    ...config,
    runRaw(done, options) {
      oneClickSnapshot = options.setup;
      done(true);
    },
  });
  oneClickButton.click();
  await new Promise(resolve => setImmediate(resolve));
  assert(oneClickSnapshot, 'one primary click must scan and start the raw comparison');
  assert.deepEqual(Array.from(oneClickSnapshot.selectedKeys), ['miiv', 'eicu']);
  assert.equal(oneClickSnapshot.featureScope, 'all');

  setup.setView('idle');
  setup.changeRawRoot('');
  setup.setSelectedKeys(['miiv', 'eicu']);
  const typedRootButton = fakeButton();
  typedRootButton.setAttribute('aria-disabled', 'true');
  const typedRootInput = {
    value: '',
    handlers: {},
    addEventListener(type, handler) { this.handlers[type] = handler; },
  };
  setup.bind({
    querySelector() { return null; },
    querySelectorAll(selector) {
      if (selector === '[data-crossdb-root]') return [typedRootInput];
      if (selector === '[data-crossdb-run-raw]') return [typedRootButton];
      return [];
    },
  }, config);
  typedRootInput.value = '/typed-root';
  typedRootInput.handlers.input();
  assert.equal(
    typedRootButton.getAttribute('aria-disabled'),
    null,
    'typing a valid root must enable the primary action without requiring blur',
  );

  let rawActive = true;
  let rawCancelCount = 0;
  context.EU_CROSSDB_PROGRESS = {
    clear() {},
    requestCancel(options) {
      rawCancelCount += 1;
      if (options && typeof options.onStateChange === 'function') options.onStateChange();
      return true;
    },
    snapshot() { return rawActive ? { jobId: 'raw-job', starting: false } : { jobId: '', starting: false }; },
  };
  const registryFenceOperation = setup.beginOperation();
  setup.setView('loading');
  assert.equal(setup.onRegistryChanged(), false, 'registry changes must not hide an active raw job');
  assert.equal(setup.operationCurrent(registryFenceOperation), true);
  const rawResetButton = fakeButton();
  let rawResetCount = 0;
  setup.bind({
    querySelector() { return null; },
    querySelectorAll(selector) { return selector === '[data-viz-reset]' ? [rawResetButton] : []; },
  }, {
    helpers: helpers(),
    repaint() { repaintCount += 1; },
    resetResult() { rawResetCount += 1; },
  });
  rawResetButton.click();
  assert.equal(rawCancelCount, 1, 'reset during a raw job must request cooperative cancellation');
  assert.equal(rawResetCount, 0, 'active raw reset must not abandon the job and clear the screen');
  rawActive = false;

  let missingApiError = null;
  const missingApiConfig = {
    api: {},
    getError() { return missingApiError; },
    helpers: helpers(),
    repaint() { repaintCount += 1; },
    setError(value) { missingApiError = value; },
  };
  assert.equal(await setup.scan('/missing-api', missingApiConfig), false);
  assert.match(missingApiError, /folder check API is unavailable/);
  assert.match(setup.renderReal(missingApiConfig), /folder check API is unavailable/);

  context.EU_DATA = 'demo';
  setup.setSelectedKeys(['miiv', 'eicu']);
  setup.setView('idle');
  const runButton = fakeButton();
  const resetButton = fakeButton();
  const root = {
    querySelector() { return null; },
    querySelectorAll(selector) {
      if (selector === '[data-crossdb-run-demo]') return [runButton];
      if (selector === '[data-viz-reset]') return [resetButton];
      return [];
    },
  };
  let lateCompletion = null;
  let operationId = null;
  let resetCount = 0;
  const operationConfig = {
    helpers: helpers(),
    repaint() { repaintCount += 1; },
    resetResult() { resetCount += 1; },
    runDemo(done, runSnapshot) {
      lateCompletion = done;
      operationId = runSnapshot.operationId;
    },
  };
  setup.bind(root, operationConfig);
  runButton.click();
  assert.equal(setup.view(), 'loading');
  assert.equal(setup.operationCurrent(operationId), true);
  resetButton.click();
  assert.equal(setup.view(), 'idle');
  assert.equal(setup.operationCurrent(operationId), false);
  assert.equal(resetCount, 1);
  const repaintAfterReset = repaintCount;
  lateCompletion(true);
  assert.equal(setup.view(), 'idle', 'a late completion must not restore loaded state after reset');
  assert.equal(repaintCount, repaintAfterReset, 'a fenced completion must not repaint stale state');

  process.stdout.write(JSON.stringify({
    bounded_profile: true,
    explicit_feature_scope: true,
    identity_resume_fail_closed: true,
    missing_api_visible: true,
    one_click_full_default: true,
    typed_root_enables_primary: true,
    raw_completion_loaded: true,
    raw_registry_guard: true,
    raw_reset_cancel: true,
    operation_reset_fence: true,
    progressive_source_choice: true,
    scan_reused: true,
    selection_revalidated: true,
    server_text_escaped: true,
    stale_scan_blocked: true,
  }));
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
