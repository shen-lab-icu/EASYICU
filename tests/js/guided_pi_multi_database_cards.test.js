'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const BATCH = 'b'.repeat(64);
const CARD = 'c'.repeat(64);

function owner(statusPayload, language = 'zh') {
  const declared = {};
  const esc = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const window = {
    EU_LANG: language,
    EU_HTML: { esc },
    EU_CATALOG: { dict: { age: ['Age', '年龄'] } },
    EU_DATABASE_NAMES: { name: key => ({ miiv: 'MIMIC-IV', eicu: 'eICU-CRD', hirid: 'HiRID', sic: 'SICdb' }[key] || '') },
    EasyICU: { guidedPi: { declare: (name, value) => { declared[name] = value; }, optional: name => declared[name] || null } },
  };
  const context = vm.createContext({ window, setTimeout: () => 0, Date });
  for (const file of ['screens-guided-pi-source-adjustments.js', 'screens-guided-pi-multi-database.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context);
  }
  const calls = [];
  let payload = statusPayload;
  let repaints = 0;
  const api = {
    getPiMultiDatabase: async (...args) => { calls.push(['status', ...args]); return payload; },
    approvePiMultiDatabase: async body => { calls.push(['approve', body]); },
    cancelPiMultiDatabase: async body => { calls.push(['cancel', body]); },
  };
  const session = {
    session_id: 'pi-1',
    conversation_replay: {
      turns: [{ job_id: 'job-1', started_at: '2026-10-26T10:00:00Z', ended_at: '2026-10-26T10:00:04Z' }],
      multi_database_batches: [{ batch_id: BATCH, study_context_id: 'study-1', source_run_id: null,
        message_job_id: 'job-1', created_at: '2026-10-26T10:00:03Z' }],
    },
  };
  const tr = (en, zh) => (language === 'zh' ? zh : en);
  const cards = declared.multiDatabase.create({
    tr, esc, api: () => api, session: () => session, projectId: () => 'project-1',
    disabled: () => false, repaint: () => { repaints += 1; }, errorText: String,
  });
  return { cards, calls, session, setPayload: next => { payload = next; }, repaints: () => repaints };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

const REVIEW = {
  status: 'review', card_sha256: CARD, answer: {},
  card: { batch_id: BATCH, targets: [
    { database: 'eicu', disposition: 'exact', substitutions: [], definition_codes: [], reason_codes: [] },
    { database: 'hirid', disposition: 'adjusted', definition_codes: ['binned_covariate:age'], reason_codes: [], substitutions: [
      { capability: 'threshold:age', reason_code: 'binned', original: { concept: 'age', operator: '>=', value: 18 },
        substitute: { concept: 'age', operator: '>=', value: 20 } }] },
    { database: 'sic', disposition: 'not_run', substitutions: [], definition_codes: [], reason_codes: ['multi_database_source_unavailable'] },
  ] },
};

test('the combined card lists every database and approves the displayed digest once', async () => {
  const { cards, calls, session } = owner(REVIEW);
  const rows = cards.rows(session);
  assert.equal(rows[0].role, 'multi_database');
  assert.equal(rows[0].timelineAt, Date.parse('2026-10-26T10:00:04Z') + 1);
  await flush();
  assert.deepEqual(calls[0], ['status', BATCH, 'project-1', 'pi-1']);
  const html = cards.render(rows[0]);
  assert.match(html, /eICU-CRD<\/strong><span>精确对应/);
  assert.match(html, /HiRID<\/strong><span>口径有调整/);
  assert.match(html, /年龄 ≥ 20 岁/);
  assert.match(html, /按区间记录：年龄/);
  assert.match(html, /SICdb<\/strong><span>不运行<\/span><p>这个数据库没有已登记的数据包。/);
  assert.match(html, /结果分别报告，不合并/);
  assert.match(html, new RegExp(`data-gpi-multi-approve="${BATCH}" data-gpi-multi-card="${CARD}"[^>]*>批准并运行 2 个数据库<`));
  const button = { disabled: false, dataset: { gpiMultiApprove: BATCH, gpiMultiCard: CARD } };
  assert.equal(cards.handleClick({ target: { closest: s => (s === '[data-gpi-multi-approve]' ? button : null) } }), true);
  assert.equal(button.disabled, true);
  await flush();
  assert.deepEqual(JSON.parse(JSON.stringify(calls[1])), ['approve', { project_id: 'project-1', session_id: 'pi-1', batch_id: BATCH, card_sha256: CARD }]);
  assert.equal(cards.handleClick({ target: { closest: s => (s === '[data-gpi-multi-approve]' ? button : null) } }), true);
  await flush();
  assert.equal(calls.filter(call => call[0] === 'approve').length, 1);
});

test('results stay per database with the descriptive range, never a pooled estimate', async () => {
  const primary = (database, estimate, low, high, n) => ({ database, unit: 'odds_ratio', estimate, ci_low: low, ci_high: high, population_n: n });
  const { cards, session } = owner({
    status: 'completed', card_sha256: CARD, card: REVIEW.card,
    answer: {
      reference: { state: 'completed', primary: primary('miiv', 1.3, 1.1, 1.5, 9000) },
      targets: [
        { database: 'eicu', state: 'completed', reason_code: null, primary: primary('eicu', 1.2, 1.05, 1.37, 12000) },
        { database: 'hirid', state: 'failed', reason_code: 'multi_database_execution_failed' },
      ],
      heterogeneity: { n_databases: 2, minimum: 1.2, maximum: 1.3, spread: 0.1, null_value: 1, description_code: 'directions_agree' },
    },
  });
  const rows = cards.rows(session);
  await flush();
  const html = cards.render(rows[0]);
  assert.match(html, /MIMIC-IV<\/strong><span>原研究<\/span><b>比值比（OR） 1.30（1.10–1.50） · n = 9,000/);
  assert.match(html, /eICU-CRD<\/strong><span>已完成<\/span><b>比值比（OR） 1.20（1.05–1.37）/);
  assert.match(html, /HiRID<\/strong><span>未完成<\/span><p>运行没有完成。/);
  assert.match(html, /2 个数据库的估计在 1.20 至 1.30 之间，都在 1.00 的同一侧/);
  assert.doesNotMatch(html, /pooled|合并估计/);
  assert.doesNotMatch(html, /data-gpi-multi-approve/);
});

test('a running database can be cancelled from its own row', async () => {
  const { cards, calls, session } = owner({ status: 'running', card_sha256: CARD, card: REVIEW.card,
    answer: { targets: [{ database: 'eicu', state: 'running', reason_code: null }] } });
  const rows = cards.rows(session);
  await flush();
  const html = cards.render(rows[0]);
  assert.match(html, new RegExp(`data-gpi-multi-cancel="${BATCH}" data-gpi-multi-database="eicu"`));
  const button = { disabled: false, dataset: { gpiMultiCancel: BATCH, gpiMultiDatabase: 'eicu' } };
  cards.handleClick({ target: { closest: s => (s === '[data-gpi-multi-cancel]' ? button : null) } });
  await flush();
  assert.deepEqual(JSON.parse(JSON.stringify(calls.find(call => call[0] === 'cancel'))),
    ['cancel', { project_id: 'project-1', session_id: 'pi-1', batch_id: BATCH, database: 'eicu' }]);
});

test('a batch still preparing says so and is read again later', async () => {
  const { cards, calls, session } = owner({ batch_id: BATCH, status: 'preparing' });
  const rows = cards.rows(session);
  await flush();
  assert.match(cards.render(rows[0]), /正在为每个数据库准备审阅/);
  assert.equal(calls.length, 1);
});
