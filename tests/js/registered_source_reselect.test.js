'use strict';

// Selecting a folder that is already registered must not register it again.
// A fresh registration replaced the official demo's release name and receipt
// with the bare database code ("MIIV"), so two different exports looked alike.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const calls = [];
const demo = '/data/mimic_iv_demo_v2_2/exports/abc';
const window = {
  EU_API: {},
  EU_WORKSPACE_REGISTRY: {
    sources: [{ path: demo, ok: true, label: 'MIMIC-IV Clinical Database Demo v2.2', source_id: 'mimic_iv_demo_v2_2' }],
    active_path: null, crossdb_paths: [],
  },
  addEventListener() {},
};
const document = { readyState: 'loading', querySelector() { return null; }, addEventListener() {} };
async function fetch(path, init) {
  calls.push({ path, body: init && init.body ? JSON.parse(init.body) : null });
  return { ok: true, status: 200, async json() { return { sources: [], crossdb_paths: [] }; } };
}
vm.runInNewContext(source, { window, document, fetch, console, setTimeout, clearTimeout, URLSearchParams, location: { hash: '' } });

(async () => {
  await window.EU_API.registerWorkspaceSource(demo + '/', { active: true, crossdb: true });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, '/api/workspaces/registry');
  assert.deepEqual(calls[0].body, {
    sources: [{ path: demo, label: 'MIMIC-IV Clinical Database Demo v2.2' }],
    crossdb_paths: [demo], active_path: demo, active_source: 'registered_source',
  });

  await window.EU_API.registerWorkspaceSource('/data/new_export', { active: true, crossdb: true });
  assert.equal(calls[1].path, '/api/workspaces/register');
  assert.equal(calls[1].body.path, '/data/new_export');
  console.log('registered_source_reselect: ok');
})().catch(error => { console.error(error); process.exit(1); });
