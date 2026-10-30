'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
global.window = global;
global.EU_LANG = 'zh';
global.t = (en, zh) => zh;
global.icon = () => '';
const classes = new Set();
const classList = { contains: v => classes.has(v), toggle: (v, on) => on ? classes.add(v) : classes.delete(v) };
const main = { classList, querySelector: () => null };
const aside = { classList, closest: () => main };
global.document = { getElementById: id => id === 'gdContextAside' ? aside : null, addEventListener() {} };
const host = { innerHTML: '', addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; }, replaceChildren() { this.innerHTML = ''; } };
for (const f of ['html-escape.js', 'screens-agent-reader-vocab.js', 'screens-agent-report-reader.js', 'screens-agent-render.js', 'screens-guided-pi-modules.js', 'screens-guided-pi-resources.js', 'screens-guided-pi-evidence-preview.js']) require(path.join(root, f));
global.EasyICU.guidedPi.declare('literature', { renderArtifact: () => '' });
require(path.join(root, 'screens-guided-pi-preview.js'));
const preview = global.EasyICU.guidedPi.require('preview');
preview.mount(host);
const sha = 'a'.repeat(64);
global.history = { state: null, replaceState(_state, _title, value) { global.location = new URL(value); } };
const calls = [];
global.EU_API = { loadPiCopilotResearchArtifact: async (...args) => { calls.push(args); return { payload: {} }; } };
const settle = async () => { for (let i = 0; i < 10; i++) await new Promise(resolve => setImmediate(resolve)); };
async function restore(name, digest = sha) {
  preview.close(); calls.length = 0;
  global.location = new URL(`http://127.0.0.1:8770/?pi_view=artifact&pi_view_kind=research_artifact&pi_view_run=run_1&pi_view_artifact=${name}&pi_view_sha=${digest}#guided`);
  const opened = preview.restoreFromLocation('project_1', {});
  await settle();
  return opened;
}
test('restore renders registered title and preserves immutable artifact and digest', async () => {
  assert.equal(await restore('scientific_readiness.json'), true);
  assert.match(host.innerHTML, /研究审阅/);
  assert.ok(calls.some(args => args.includes('scientific_readiness.json') && JSON.stringify(args).includes(sha)));
  assert.equal(global.location.searchParams.get('pi_view_artifact'), 'scientific_readiness.json');
  assert.equal(global.location.searchParams.get('pi_view_sha'), sha);
});
test('restore keeps unknown filename and rejects an invalid digest', async () => {
  assert.equal(await restore('custom-report.json'), true);
  assert.match(host.innerHTML, /custom-report\.json/);
  assert.equal(await restore('custom-report.json', 'invalid'), false);
  assert.equal(calls.length, 0);
});
