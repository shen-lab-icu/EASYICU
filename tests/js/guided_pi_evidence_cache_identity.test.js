'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const owner = fs.readFileSync(path.resolve(__dirname, '../../src/easyicu/webserver/static/js/screens-guided-pi-preview.js'), 'utf8');
const start = owner.indexOf('  async function openEvidence(button) {');
const end = owner.indexOf('  async function loadResource()', start);
assert.ok(start >= 0 && end > start, 'production evidence click owner exists');
const source = owner.slice(start, end);
const sha = letter => letter.repeat(64);
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function harness(api) {
  const state = { resource: { kind: 'research_artifact', run_id: 'run_1' }, projectId: 'project', request: 1,
    evidenceTabs: [], activeEvidenceId: '', activeClaimId: '', governance: null };
  const renders = [];
  const context = vm.createContext({ state, isEvidenceBoundResource: () => true, tr: en => en,
    window: { EU_API: { loadPiCopilotResearchEvidence: api }, EasyICU: { guidedPi: {
      require: () => ({ kindLabel: () => 'JSON' }),
    } } }, render: () => { const item = state.evidenceTabs.find(row => row.evidenceId === state.activeEvidenceId);
      renders.push(item && { sha256: item.sha256, loading: item.loading, error: item.error, payload: item.payload }); },
    syncLocation() {},
  });
  vm.runInContext(source, context);
  return { state, renders, open: context.openEvidence };
}
function button(digest, pointer = '') {
  return { dataset: { evidenceId: 'summary', evidenceSha256: digest, evidencePointer: pointer,
    evidenceSourceValue: '1', evidenceKind: 'statistic' }, textContent: '1' };
}

test('a different digest triggers backend revalidation instead of cached evidence', async () => {
  const calls = [];
  const h = harness(async (...args) => {
    calls.push(args);
    if (args[3] === sha('b')) throw new Error('requested digest does not match');
    return { payload: { value: 'first' } };
  });
  await h.open(button(sha('a'), '/estimate'));
  await h.open(button(sha('a'), '/other'));
  assert.equal(calls.length, 1, 'same digest keeps its existing verified payload');
  assert.equal(h.state.evidenceTabs[0].locator.pointer, '/other');
  await h.open(button(sha('b')));
  assert.equal(calls.length, 2, 'a different digest must be sent to the backend');
  assert.equal(calls[1][3], sha('b'));
  assert.equal(h.state.evidenceTabs.length, 1);
  assert.equal(h.state.evidenceTabs[0].sha256, sha('b'));
  assert.match(h.state.evidenceTabs[0].error, /digest does not match/);
  assert.equal(h.state.evidenceTabs[0].payload, null, 'the old result must not be presented as the new digest');
});

test('a displaced request cannot overwrite newer governance or render stale data', async () => {
  const first = deferred(), second = deferred();
  const h = harness((_project, _run, _id, digest) => digest === sha('a') ? first.promise : second.promise);
  const oldRequest = h.open(button(sha('a')));
  const newRequest = h.open(button(sha('b')));
  second.resolve({ payload: { value: 'new' }, governance: { revision: 'new' } });
  await newRequest;
  first.resolve({ payload: { value: 'old' }, governance: { revision: 'old' } });
  await oldRequest;
  assert.equal(h.state.evidenceTabs.length, 1);
  assert.equal(h.state.evidenceTabs[0].sha256, sha('b'));
  assert.equal(h.state.evidenceTabs[0].payload.value, 'new');
  assert.equal(h.state.governance.revision, 'new');
  assert.equal(h.renders.at(-1).payload.value, 'new');
});
