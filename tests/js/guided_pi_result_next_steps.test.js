'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh', requestAnimationFrame: fn => fn() };
for (const file of process.argv.slice(2)) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
let language = 'zh';
let available = true;
const tr = (en, zh) => language === 'zh' ? zh : en;
const latest = { run_id: 'synthetic', present: true, analysis_results_available: true, analysis_validated: true,
  artifact_refs: ['evidence_ledger.json', 'result_tables.json'].map(artifact => ({ artifact, run_id: 'synthetic', kind: 'research_artifact' })) };
const workflow = { stages: [{ id: 'analysis', status: 'complete' }] };
const outcome = modules.require('runOutcome').create({ tr, esc: window.EU_HTML.esc, iconHtml: () => '',
  resourceButton: () => '', canPreview: () => available });
const callbacks = {};
const state = { session: {}, latestRun: latest, workflow, draft: '',
  host: { addEventListener: (name, fn) => { callbacks[name] = fn; }, querySelector: () => null } };
modules.require('events').create({ state, tr, render() {}, RUN_OUTCOME: outcome,
  RUN_FILES: { handleClick: () => false }, COHORT_ELIGIBILITY: { actionFromEvent: () => null } }).wire();
for (language of ['zh', 'en']) {
  window.EU_LANG = language;
  const html = outcome.render(latest, workflow);
  assert.ok(html.lastIndexOf('gpi-followups') > html.lastIndexOf('gpi-response-review'), 'next steps finish the result card');
  assert.ok(html.includes('data-gpi-followup="0"'));
  const button = { dataset: { gpiFollowup: '0' } };
  callbacks.click({ target: { closest: selector => selector === '[data-gpi-followup]' ? button : null } });
  assert.equal(state.draft, outcome.followUps(latest, workflow)[0]);
  state.draft = '';
  state.session.stale = { stale: true };
  callbacks.click({ target: { closest: selector => selector === '[data-gpi-followup]' ? button : null } });
  assert.equal(state.draft, '');
  state.session.stale = null;
  available = false;
  assert.match(outcome.render(latest, workflow), /data-gpi-followup="0" disabled/);
  available = true;
}
console.log('result next steps: bilingual final placement, composer dispatch and stale guard passed');
