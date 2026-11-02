'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve('src/easyicu/webserver/static/js');
(async () => {
  const streams = [];
  class EventSource { constructor(url) { this.url = url; streams.push(this); } close() {} }
  const ctx = vm.createContext({ window: { EU_LANG: 'en', EU_HTML: {esc: String} }, EventSource,
    document: {getElementById: () => null}, Date, Promise, setTimeout });
  for (const owner of ['modules', 'error-text', 'activity', 'replay', 'childjob'])
    vm.runInContext(fs.readFileSync(path.join(root, `screens-guided-pi-${owner}.js`), 'utf8'), ctx);
  const modules = ctx.window.EasyICU.guidedPi;
  const tr = en => en;
  const errors = modules.require('errorText').create({tr});
  const activity = modules.require('activity').create({tr, esc: String, iconHtml: () => '',
    resourceName: () => '', resourceKey: () => '', resourceButton: () => '', runFailureText: errors.runFailureText});
  let watched = 'prep', source = null, advanced = 0;
  const messages = [];
  const child = modules.require('childJob').create({tr, activity,
    upsertActivityStep(target, row) { const old = target.steps.find(item => item.id === row.id); if (old) Object.assign(old, row); else target.steps.push(row); },
    render() {}, messages: () => messages, session: () => ({session_id:'session'}),
    childJobId: () => watched, setChildJobId: id => { watched = id; },
    childSource: () => source, setChildSource: value => { source = value; },
    archiveChildJob: async () => {}, refreshSession: async () => {}, loadWorkflow: async () => {},
    sessionIsStale: () => false, rebind: async () => {},
    continueSystemOwnedPlanProgression: async () => { advanced++; return false; },
  });
  child.handleChildJobEvent('prep', 'easyicu_raw_preparation_submitted', {
    type:'end', status:'done', result:{deferred_review:{status:'preparing_review',job_id:'stage'}},
  });
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(watched, 'stage');
  assert.match(streams[0].url, /\/api\/jobs\/stage\/events$/);
  assert.equal(advanced, 0);
  watched = 'prep-blocked'; source = null;
  child.handleChildJobEvent('prep-blocked', 'easyicu_raw_preparation_submitted', {
    type:'end', status:'done', result:{deferred_review:{status:'blocked',reason:'ValueError'}},
  });
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(advanced, 0);
  assert.equal(messages.find(item => item.childJobId === 'prep-blocked').status, 'error');
  console.log('deferred review handoff contracts passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
