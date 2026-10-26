'use strict';

// Synthetic source metadata only. An optional checkout root exercises the
// same contract against the read-only base without copying a fixture there.
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(process.argv[2] || path.join(__dirname, '../..'));
const js = path.join(root, 'src/easyicu/webserver/static/js');
// The stream transport is not under test and needs no installed provider SDK.
// Evaluate the unmodified owner body with a minimal in-memory event sink.
const finalizerSource = fs.readFileSync(path.join(root,
  'src/easyicu/webserver/pi_copilot/node_app/src/post-tool-finalization.mjs'), 'utf8');
const finalizer = vm.runInNewContext(finalizerSource
  .replace(/^import .* from "@earendil-works\/pi-ai";\n/m, '')
  .replace(/^export /gm, '') + '\n({ hostPostToolFinalization, OWNER_CONTEXT_MARKER });', {
  createAssistantMessageEventStream() {
    let message;
    return { push(event) { if (event.type === 'done') message = event.message; },
      result: async () => message };
  },
});
// Every research database takes the same source path; what a study can do on
// it is the capability gate's decision before planning, not a fixed list here.
const supported = { miiv: 'MIMIC-IV', eicu: 'eICU', hirid: 'HiRID', aumc: 'AmsterdamUMCdb', sic: 'SICdb' };
const exportRow = { source_id: 'src_synthetic', label: 'Synthetic export', stays: 12,
  module_count: 2, generated_date: '2026-01-02' };
const empty = { state: 'ready', recommended: null, choices: [] };

function card(database, label, { hosted = false, language = 'zh', offer = empty,
  authorization = { status: 'pending' }, namedDemo = null } = {}) {
  const window = { EU_CAPABILITIES: { deployment: { hosted } },
    EU_SETTINGS: { support_contact: 'Support <desk>' } };
  const context = vm.createContext({ window });
  for (const name of ['database-names.js', 'screens-guided-pi-modules.js', 'html-escape.js',
    'screens-guided-pi-hosted.js', 'screens-guided-pi-data-consent.js']) {
    vm.runInContext(fs.readFileSync(path.join(js, name), 'utf8'), context, { filename: name });
  }
  return window.EasyICU.guidedPi.require('dataConsent').render(
    { data_source_authorization: authorization },
    { tr: (en, zh) => language === 'zh' ? zh : en, esc: window.EU_HTML.esc, icon: () => '',
      namedDatabase: { key: database, label, offer }, namedDemo },
  );
}

function refusal(label, language) {
  return language === 'zh' ? `本版本暂不支持使用 ${label} 完成这项研究。`
    : `This version does not support this study with ${label}.`;
}

function missing(label, language) {
  return language === 'zh' ? `尚未添加 ${label} 数据。` : `${label} data have not been added.`;
}

async function narration(database, label, { hosted = false, language = 'zh',
  extra = {}, toolResult = false, question = `Summarize temperature records in ${label}.` } = {}) {
  const { hostPostToolFinalization, OWNER_CONTEXT_MARKER } = await finalizer;
  const catalog = { status: 'ok', code: 'easyicu_data_sources_listed', details: {
    selected_database: { database, label }, local_folder_selection_available: !hosted,
    supported_databases: [{ database: 'miiv', display_label: 'MIMIC-IV' }], ...extra,
  } };
  const messages = [{ role: 'user', content: [{ type: 'text',
    text: question + (toolResult ? '' : OWNER_CONTEXT_MARKER + JSON.stringify([catalog])) }] }];
  if (toolResult) messages.push({ role: 'toolResult', toolName: 'easyicu_list_data_sources',
    isError: false, details: catalog });
  messages.push({ role: 'assistant', content: [{ type: 'toolCall', id: 'synthetic-update',
    name: 'easyicu_update_study_context', arguments: { question } }] },
  { role: 'toolResult', toolCallId: 'synthetic-update', toolName: 'easyicu_update_study_context',
    isError: false, details: { status: 'ok', code: 'study_context_updated', details: { workflow: {
      next_action_code: 'study_setup_incomplete', missing_setup_fields: ['data_source'],
      study_setup_receipt: { configuration: { data_source: {} } },
    } } } });
  const stream = hostPostToolFinalization({ api: 'test', provider: 'test', id: 'test' }, { messages }, language);
  assert.ok(stream, 'the saved initial question has a host-owned source reply');
  return (await stream.result()).content[0].text;
}

for (const [database, label] of Object.entries(supported)) {
  for (const language of ['zh', 'en']) for (const hosted of [false, true]) {
    test(`${database} missing data keeps deployment-specific guidance (${language}, hosted=${hosted})`, async () => {
      const html = card(database, label, { hosted, language });
      assert.ok(html.includes(missing(label, language)), html);
      assert.ok(!html.includes(refusal(label, language)));
      if (hosted) {
        assert.match(html, /Support &lt;desk&gt;/);
        assert.doesNotMatch(html, /data-gpi-data-source-database=/);
      } else {
        assert.ok(html.includes(`data-gpi-data-source-database="${database}"`));
        assert.doesNotMatch(html, /Support|管理员|administrator|server/);
      }
      const loading = card(database, label, { hosted, language, offer: { state: 'loading' } });
      assert.ok(!loading.includes(missing(label, language)), 'an unfinished lookup is not absent data');
      assert.match(loading, /aria-busy="true"/);
      const text = await narration(database, label, { hosted, language });
      assert.ok(text.includes(missing(label, language)), text);
      if (hosted) assert.doesNotMatch(text, /目录|folder/);
      else assert.match(text, /目录|folder/);
    });
    test(`${database} registered choices preserve identity and consent (${language}, hosted=${hosted})`, async () => {
      for (const offer of [{ ...empty, recommended: exportRow }, { ...empty, choices: [exportRow] }]) {
        const html = card(database, label, { hosted, language, offer });
        assert.ok(html.includes('data-gpi-data-source-id="src_synthetic"'));
        assert.ok(html.includes(`data-gpi-data-source-database="${database}"`));
        assert.match(html, /不等于批准分析|does not approve analysis/);
        assert.ok(!html.includes(missing(label, language)));
      }
      const text = await narration(database, label, { hosted, language, extra: {
        recommended_source: { ...exportRow, database, availability: 'available_in_easyicu', aggregate: { stays: 12 } },
      } });
      assert.match(text, /不等于批准分析|does not approve analysis/);
      assert.match(text, /12/);
      assert.ok(!text.includes(missing(label, language)));
    });
  }
}

test('a pending HiRID project source is confirmed like any other source', () => {
  const html = card('eicu', 'eICU', { authorization: { status: 'pending',
    reason: 'project_source_confirmation_required', source: { database: 'hirid', label: 'HiRID <fixture>' } } });
  assert.match(html, /这项研究使用哪份数据/);
  assert.ok(!html.includes(refusal('HiRID &lt;fixture&gt;', 'zh')));
  assert.doesNotMatch(html, /<fixture>/);
});

test('confirmed and in-progress source states are not new scope refusals', () => {
  for (const status of ['confirmed', 'selection_in_progress']) {
    const html = card('hirid', 'HiRID', { authorization: { status } });
    assert.ok(!html.includes(refusal('HiRID', 'zh')));
  }
});

test('a supported official demo remains explicitly separate from full data', () => {
  const html = card('eicu', 'eICU', { namedDemo: { id: 'synthetic_demo', database: 'eicu_demo',
    label: 'eICU Demo', prepared: false } });
  assert.match(html, /data-gpi-named-demo="synthetic_demo"/);
  assert.match(html, /仅官方 Demo 数据/);
  assert.match(html, /确认数据源不等于批准分析/);
});

test('scope narration still requires the researcher and catalog to name the same database', async () => {
  for (const options of [{ question: 'Summarize temperature records in MIMIC-IV.' },
    { extra: { database_selection_deferred: true } }]) {
    const text = await narration('hirid', 'HiRID', options);
    assert.ok(!text.includes(refusal('HiRID', 'zh')));
    assert.match(text, /请先选择/);
  }
});
