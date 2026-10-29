'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const STATIC = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const SHA = 'a'.repeat(64), NONCE = 'b'.repeat(64), SCHEMA = 'easyicu.web-runtime-qa/1';

// This harness executes only the checked-in trusted probe against typed test
// elements. Generated artifact scripts are never evaluated by a Host process.
function fixture(alternate, brokenBar = false, brokenFilter = false) {
  const ids = alternate
    ? ['[data-metric="amount"]', '[data-metric="events"]', '.rate-label', 'section > .column', 'select[data-choice]', '.save-csv']
    : ['#records', '#events', '#percentage', '#column', '#filter', '#save'];
  const nodes = new Map();
  function element(text) { return { textContent: String(text), disabled: false, getAttribute: () => null }; }
  const [total, events, rate, bar, filter, download] = ids;
  nodes.set(total, element(12)); nodes.set(events, element(3)); nodes.set(rate, element('25.0%'));
  nodes.set(bar, { getBoundingClientRect: () => ({ height: brokenBar ? 2 : Number(nodes.get(rate).textContent.replace('%', '')) * 1.9, width: 50 }) });
  nodes.set(download, element('Save CSV'));
  const select = { tagName: 'SELECT', value: 'all', options: [{ value: 'all' }, { value: 'subset' }],
    dispatchEvent() {
      if (brokenFilter) return;
      const subset = this.value === 'subset';
      nodes.get(total).textContent = subset ? '7' : '12';
      nodes.get(events).textContent = subset ? '2' : '3';
      nodes.get(rate).textContent = subset ? '28.6%' : '25.0%';
    } };
  nodes.set(filter, select);
  const contract = { schema: SCHEMA, checks: [
    { id: 'total', kind: 'text_number', selector: total, expected: 12, tolerance: 0.000001 },
    { id: 'ratio', kind: 'ratio', numerator: events, denominator: total, output: rate, scale: 100, tolerance: 0.051 },
    { id: 'geometry', kind: 'bar_geometry', selector: bar, value_selector: rate, axis: 'height', domain_max: 100, extent: 190, min_extent: 2, tolerance_px: 0.5 },
    { id: 'selection', kind: 'select_cases', selector: filter, cases: [{ value: 'subset', checks: [
      { id: 'subset-total', kind: 'text_number', selector: total, expected: 7, tolerance: 0.000001 },
      { id: 'subset-ratio', kind: 'ratio', numerator: events, denominator: total, output: rate, scale: 100, tolerance: 0.051 },
      { id: 'subset-geometry', kind: 'bar_geometry', selector: bar, value_selector: rate, axis: 'height', domain_max: 100, extent: 190, min_extent: 2, tolerance_px: 0.5 },
    ] }] },
    { id: 'csv', kind: 'download_control', selector: download },
  ] };
  return { nodes, ids, select, contract, document: { readyState: 'complete', querySelectorAll(selector) {
    return nodes.has(selector) ? [nodes.get(selector)] : [];
  } } };
}

async function probe(document, contract) {
  let done;
  const result = new Promise(resolve => { done = resolve; });
  const source = fs.readFileSync(path.join(STATIC, 'workspace-runtime-probe.js'), 'utf8')
    .replace('/*EASYICU_RUNTIME_QA_CONFIG*/null', JSON.stringify({ nonce: NONCE, source_sha256: SHA, contract }));
  const finished = [];
  document.dispatchEvent = event => finished.push(event.type);
  const window = {};
  vm.runInNewContext(source, { window, document,
    Event: class { constructor(type) { this.type = type; } }, CustomEvent: class { constructor(type) { this.type = type; } },
    addEventListener() {}, setTimeout, requestAnimationFrame: callback => queueMicrotask(callback),
    parent: { postMessage: value => { assert.equal(window.__easyicuRuntimeQaRunning, false);
      assert.ok(finished.includes('easyicu:runtime-qa-finished')); done(value); } } });
  return result;
}

for (const alternate of [false, true]) {
  test(`generic geometry red/green and linked select restores state (alternate DOM ${alternate})`, async () => {
    const bad = fixture(alternate, true);
    const red = await probe(bad.document, bad.contract);
    assert.equal(red.type, 'easyicu:runtime-qa-child');
    assert.equal(red.report.checks.find(row => row.id === 'geometry').code, 'bar_geometry_mismatch');
    assert.equal(red.report.checks.find(row => row.id === 'subset-geometry').status, 'failed');
    assert.equal(bad.select.value, 'all');
    const good = fixture(alternate);
    const green = await probe(good.document, good.contract);
    assert.ok(green.report.checks.filter(row => row.kind !== 'download_control').every(row => row.status === 'passed'));
    assert.equal(green.report.checks.find(row => row.id === 'csv').status, 'pending');
    assert.equal(green.report.csv_export_ready, false);
    assert.equal(green.report.browser_download_verified, false);
    assert.equal(good.select.value, 'all');
    assert.equal(good.nodes.get(good.ids[0]).textContent, '12');
  });
}

test('filter linkage and ratio mismatch fail, rather than inferring success from a clickable control', async () => {
  const data = fixture(true, false, true);
  data.nodes.get(data.ids[2]).textContent = '99%';
  const result = await probe(data.document, data.contract);
  assert.equal(result.report.checks.find(row => row.id === 'ratio').code, 'ratio_mismatch');
  assert.equal(result.report.checks.find(row => row.id === 'subset-total').code, 'number_mismatch');
  assert.equal(result.report.checks.find(row => row.id === 'selection').status, 'failed');
  assert.equal(data.select.value, 'all');
});

test('missing contract yields observations only; heuristic does not declare a correct short bar failed', async () => {
  const document = { readyState: 'complete', querySelectorAll(selector) {
    return selector.includes('progressbar') ? [{ textContent: '35%', getBoundingClientRect: () => ({ height: 2, width: 80 }) }] : [];
  } };
  const result = await probe(document, null);
  assert.equal(result.report.checks.length, 0);
  assert.deepEqual(Array.from(result.report.warnings), ['suspicious_percent_bar']);
  assert.equal(result.report.scientific_verified, false);
});

test('unexpected observation errors release the local QA/CSV handshake and report failure', async () => {
  const result = await probe({ readyState: 'complete', querySelectorAll() { throw new Error('blocked'); } }, null);
  assert.equal(result.report.runtime_errors, 1);
  assert.equal(result.report.checks.length, 0);
});

function host(options) {
  const listeners = {}, api = {};
  const element = () => ({ dataset: {}, isConnected: true, children: [], textContent: '',
    setAttribute() {}, replaceChildren(...children) { this.children = children; }, appendChild(child) { this.children.push(child); } });
  const window = { EU_LANG: 'zh', addEventListener(type, fn) { listeners[type] = fn; },
    EasyICU: { guidedPi: { declare(name, value) { api[name] = value; } } } };
  vm.runInNewContext(fs.readFileSync(path.join(STATIC, 'screens-guided-pi-runtime-qa.js'), 'utf8'),
    { window, document: { createElement: element }, TextEncoder });
  let current = 'project|page.html|' + SHA;
  const owner = api.runtimeQa.create(() => current, options);
  const frame = { isConnected: true, contentWindow: {} }, slot = element();
  owner.prepare(current, { nonce: NONCE }); owner.bind(frame, slot, { projectId: 'project', file: 'page.html', sha256: SHA });
  const event = (changes = {}) => ({ source: frame.contentWindow, origin: 'null',
    data: { type: 'easyicu:runtime-qa', nonce: NONCE, report: { schema: SCHEMA, source_sha256: SHA } }, ...changes });
  return { owner, frame, slot, event, receive: listeners.message, setCurrent: value => { current = value; } };
}

test('Host accepts only current frame, nonce, hash and normalized readback; issue text is inert', async () => {
  let calls = 0;
  const ui = host({ publish: async () => { calls += 1; return { status: 'failed', source_sha256: SHA,
    coverage: ['bar_geometry'], warnings: [], checks: [{ id: 'height', status: 'failed', code: '<img src=x>', observed: 2, expected: 66.5 }] }; } });
  await ui.receive(ui.event({ source: {} })); await ui.receive(ui.event({ origin: 'http://host' }));
  const badNonce = ui.event(); badNonce.data.nonce = SHA; await ui.receive(badNonce);
  const badHash = ui.event(); badHash.data.report.source_sha256 = NONCE; await ui.receive(badHash);
  assert.equal(calls, 0);
  await ui.receive(ui.event()); assert.equal(calls, 1);
  assert.equal(ui.owner.snapshot().status, 'failed');
  assert.equal(ui.slot.children[0].dataset.gpiRuntimeQa, 'failed');
  const details = ui.slot.children[1];
  assert.ok(details.children[1].children[0].textContent.includes('<img src=x>'));
  assert.equal(details.children[1].children[0].children.length, 0);
});

test('late publish and CSV readiness cannot upgrade a closed or switched preview', async () => {
  let resolve, refreshes = 0;
  const ui = host({ publish: () => new Promise(done => { resolve = done; }), refresh: async () => { refreshes += 1; return { status: 'complete' }; } });
  const pending = ui.receive(ui.event());
  ui.owner.noteCsvReady({ file: 'other.html', sha256: SHA }); assert.equal(refreshes, 0);
  ui.owner.reset(); resolve({ status: 'complete' }); await pending;
  assert.equal(ui.owner.snapshot(), null);
  ui.owner.noteCsvReady({ file: 'page.html', sha256: SHA }); assert.equal(refreshes, 0);
});

test('API contract failures display bounded paths and expectations without raw transport text', () => {
  const ui = host({});
  const text = ui.owner.failureText({ code: 'runtime_qa_contract_invalid', message: 'DO_NOT_ECHO_TRANSPORT',
    details: { issues: [{ path: '/checks/0/expected', expected: 'finite JSON number' },
      { path: '/checks/1/cases/0/checks/0/id', expected: 'unique id' }] } });
  assert.match(text, /合同格式错误/);
  assert.match(text, /浏览器检查尚未运行/);
  assert.match(text, /\/checks\/0\/expected/);
  assert.match(text, /\/checks\/1\/cases\/0\/checks\/0\/id/);
  assert.doesNotMatch(text, /DO_NOT_ECHO_TRANSPORT/);
});

test('Host-normalized preflight failure shows schema issues rather than claiming runtime execution', async () => {
  const ui = host({ publish: async () => ({ status: 'failed', source_sha256: SHA, coverage: [], checks: [],
    contract_diagnostics: { status: 'invalid', issues: [{ path: '/checks/0/expected', expected: 'finite JSON number' }] } }) });
  await ui.receive(ui.event());
  assert.match(ui.slot.children[0].textContent, /浏览器检查尚未运行/);
  const list = ui.slot.children[1].children[1];
  assert.match(list.children[0].textContent, /\/checks\/0\/expected/);
  assert.match(list.children[1].textContent, /不要轮询/);
});
