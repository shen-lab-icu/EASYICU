/* Registry identity survives rename in shared badges and pending demo offers. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const demo = {
  id: 'mimic_iv_demo_v2_2', title: 'MIMIC-IV Clinical Database Demo', version: '2.2', database: 'miiv',
  download: { sha256: 'a'.repeat(64) }, status: { state: 'not_downloaded', registered: false },
};
const originalLabel = `${demo.title} v${demo.version}`;
const renamed = {
  ok: true, path: '/private/fixture', label: 'Teaching collection', database: 'miiv',
  source_id: demo.id, archive_sha256: demo.download.sha256, source_kind: 'public_demo',
};

async function setup({ hosted = false, catalog = true } = {}) {
  let helpers;
  const context = {
    console, t: (en, zh) => zh, icon: () => '',
    fmtInt: String, fmtNum: String, fmtPct: String, fmtP: String,
    EU_HTML: { esc: value => String(value == null ? '' : value) },
    EU_VIZ_PATIENT: { init() {} }, EU_VIZ_COHORT: { init(value) { helpers = value; } },
    VIZ_DEMO: {}, EU_CAPABILITIES: { deployment: { hosted } },
    EU_DATABASE_NAMES: { isDemo: database => database === 'eicu_demo' },
    EU_API: { loadOfficialDemoSources: async () => ({ sources: [demo] }) },
  };
  context.window = context;
  vm.createContext(context);
  for (const file of ['screens-viz-patient-demo-sources.js', 'screens-viz.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  if (catalog) await context.EU_OFFICIAL_DEMO_SOURCES.ensureLoaded();
  function select(row) {
    context.EU_WORKSPACE_REGISTRY = { sources: [row], active_path: row.path, crossdb_paths: [row.path] };
  }
  return { context, helpers, select, owner: context.EU_OFFICIAL_DEMO_SOURCES };
}

test('a renamed MIMIC-IV demo still reads 公开 Demo before the catalog loads', async () => {
  const { helpers, select } = await setup({ catalog: false });
  select(renamed);
  assert.equal(helpers.activeSourceKind().kind, 'demo');
  assert.equal(helpers.activeSourceKind().label, '公开 Demo');
  for (const mode of ['single', 'multi']) {
    const html = helpers.sourceRegistryBlock(mode);
    assert.match(html, /Teaching collection/);
    assert.match(html, /data-src-kind="demo">公开 Demo/);
  }
});

test('the host kind wins over label and database heuristics; synthetic stays synthetic', async () => {
  const { helpers, select, owner } = await setup();
  for (const kind of ['own', 'synthetic']) {
    const row = { ...renamed, source_kind: kind, label: originalLabel, database: 'eicu_demo' };
    select(row);
    assert.equal(helpers.activeSourceKind().kind, kind);
    assert.equal(owner.registeredSources([row]).length, 0);
  }
});

test('a renamed demo is registered and never offered as pending preparation', async () => {
  const { owner } = await setup();
  const registered = owner.registeredSources([renamed]);
  assert.equal(registered.length, 1);
  assert.equal(registered[0].source_id, demo.id);
  assert.equal(registered[0].path, renamed.path);
  assert.equal(owner.render({}, { pendingOnly: true, registryRows: [renamed] }), '');
});

test('old registries retain title and database fallbacks but cannot override a receipt', async () => {
  const { helpers, select, owner } = await setup();
  const legacy = { ok: true, path: '/private/legacy', database: 'miiv', label: ` ${originalLabel.toUpperCase()} ` };
  select(legacy);
  assert.equal(helpers.activeSourceKind().kind, 'demo');
  assert.equal(owner.registeredSources([legacy]).length, 1);
  select({ ...legacy, database: 'eicu_demo', label: 'Older export' });
  assert.equal(helpers.activeSourceKind().kind, 'demo');
  const stored = { ...renamed };
  delete stored.source_kind;
  assert.equal(owner.registeredSources([stored]).length, 1);
  for (const identity of [
    { source_id: 'unknown' }, { source_id: demo.id },
    { source_id: demo.id, archive_sha256: 'b'.repeat(64) }, { archive_sha256: demo.download.sha256 },
  ]) {
    assert.equal(owner.registeredSources([{ ...legacy, ...identity }]).length, 0);
  }
});

test('hosted registry rows show the renamed demo badge without visible paths or source IDs', async () => {
  const { helpers, select } = await setup({ hosted: true });
  select(renamed);
  for (const mode of ['single', 'multi']) {
    const html = helpers.sourceRegistryBlock(mode);
    const visible = html.replace(/<[^>]*>/g, '');
    assert.match(visible, /Teaching collection/);
    assert.match(visible, /公开 Demo/);
    assert.doesNotMatch(visible, /\/private\/|mimic_iv_demo_v2_2/);
    assert.doesNotMatch(html, /src-path|title="\/private\/|data-src-rename|data-src-remove|source_id/);
  }
});

test('a data-page selection is not presented as research authorization', async () => {
  for (const hosted of [false, true]) {
    const { helpers, select } = await setup({ hosted });
    select(renamed);
    const html = helpers.sourceRegistryBlock('single');
    assert.match(html, /数据页已选择/);
    assert.match(html, /研究对话中还需单独确认本次使用的数据来源/);
    assert.doesNotMatch(html, /会同时用于各数据页面和研究对话/);
    assert.doesNotMatch(helpers.sourceRegistryBlock('multi'), /数据页已选择/);
  }
});
