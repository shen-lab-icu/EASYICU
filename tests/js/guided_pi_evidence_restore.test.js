'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
global.window = global;
global.EU_HTML = { esc: v => String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.EU_LANG = 'en';
global.icon = () => '';
const classes = new Set();
const classList = { contains: v => classes.has(v), toggle: (v, on) => on ? classes.add(v) : classes.delete(v) };
const main = { classList, querySelector: () => null };
const aside = { classList, closest: () => main };
global.document = { getElementById: id => id === 'gdContextAside' ? aside : null, addEventListener() {} };
const host = { innerHTML: '', addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; }, replaceChildren() { this.innerHTML = ''; } };
require(path.join(root, 'screens-guided-pi-modules.js'));
require(path.join(root, 'screens-guided-pi-resources.js'));
require(path.join(root, 'screens-guided-pi-evidence-preview.js'));
global.EasyICU.guidedPi.declare('literature', { renderArtifact: () => '' });
require(path.join(root, 'screens-guided-pi-preview.js'));
const preview = global.EasyICU.guidedPi.require('preview');
preview.mount(host);
const artifactSha = 'a'.repeat(64), evidenceSha = 'b'.repeat(64);
const url = `http://127.0.0.1:8770/?pi_project=project_1&pi_view=evidence&pi_view_kind=research_artifact&pi_view_run=run_1&pi_view_artifact=manuscript_provenance.json&pi_view_sha=${artifactSha}&pi_view_evidence=result&pi_view_evidence_sha=${evidenceSha}&pi_view_evidence_kind=statistic&pi_view_evidence_pointer=%2Festimate#guided`;
global.history = { state: null, replaceState(_state, _title, value) { global.location = new URL(value); } };
const settle = async () => { for (let i = 0; i < 10; i++) await new Promise(resolve => setImmediate(resolve)); };
function claim(value = 0.82, digest = evidenceSha) {
  return { claim_id: 'claim_1', canonical_value: value, source_json_pointer: '/estimate',
    evidence: { evidence_id: 'result', sha256: digest, kind: 'statistic' } };
}
async function reopen(claims, source = 0.82) {
  preview.close();
  global.location = new URL(url);
  global.EU_API = {
    loadPiCopilotResearchArtifact: async () => ({ payload: { schema_version: 'easyicu.manuscript-provenance/1', claims } }),
    loadPiCopilotResearchEvidence: async () => ({ payload: { renderer: 'json', kind: 'statistic', previewable: true, value: { estimate: source } } }),
  };
  assert.equal(preview.restoreFromLocation('project_1', {}), true);
  await settle();
  return host.innerHTML;
}
test('reopening the same bound result recovers its value from the digest-pinned parent', async () => {
  const html = await reopen([claim()]);
  assert.match(html, /Source field and value match/);
  assert.doesNotMatch(html, /Selected number could not be matched/);
  assert.doesNotMatch(global.location.href, /0\.82/, 'scientific values do not enter the URL');
});
test('missing, stale or conflicting parent claims cannot certify the restored result', async () => {
  for (const claims of [[], [claim(0.82, 'c'.repeat(64))], [claim(0.82), claim(0.91)], [{...claim(), status: 'superseded'}], [claim(true)]]) {
    const html = await reopen(claims);
    assert.doesNotMatch(html, /Source field and value match/);
    assert.match(html, /Cannot verify the selected number after reopening/);
  }
});
test('a changed result is still rejected after its parent claim is restored', async () => {
  const html = await reopen([claim()], 0.83);
  assert.match(html, /Selected number could not be matched/);
  assert.doesNotMatch(html, /Source field and value match/);
});
test('a slow parent cannot cause a false match before its bound claim arrives', async () => {
  preview.close();
  global.location = new URL(url);
  let finishParent;
  global.EU_API = {
    loadPiCopilotResearchArtifact: () => new Promise(resolve => { finishParent = resolve; }),
    loadPiCopilotResearchEvidence: async () => ({ payload: { renderer: 'json', kind: 'statistic', previewable: true, value: { estimate: 0.82 } } }),
  };
  assert.equal(preview.restoreFromLocation('project_1', {}), true);
  await settle();
  assert.match(host.innerHTML, /Loading governed artifact/);
  assert.doesNotMatch(host.innerHTML, /Source field and value match/);
  finishParent({ payload: { schema_version: 'easyicu.manuscript-provenance/1', claims: [claim()] } });
  await settle();
  assert.match(host.innerHTML, /Source field and value match/);
});
