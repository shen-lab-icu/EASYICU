'use strict';

// The itemised review shows the host's items in plain words with their source;
// numbers without evidence come first and are marked, never hidden; a value is
// text, not markup.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const SHA = 'a'.repeat(64);
const source = (artifact, extra = {}) => ({ artifact, artifact_sha256: SHA, json_pointer: '/x', evidence_id: '', sha256: '', ...extra });
const page = items => ({ items, total: items.length, offset: 0, next_offset: null });
const REVIEW = {
  run_id: 'r1', analysis_only: true, publication_authorized: false,
  approach: page([
    { code: 'SR_APPROACH_OUTCOME', value: 'in-hospital death', order: 2, source: source('agent_plan.json') },
    { code: 'SR_APPROACH_POPULATION', value: { age: '>= 18', first_stay: true }, order: 1, source: source('agent_plan.json') },
  ]),
  materials: page([
    { code: 'SR_DATA_DATABASE', value: 'miiv', status: 'recorded', source: source('run_context.json') },
    { code: 'SR_DATA_EXPORT_ID', value: null, status: 'not_recorded', source: source('run_context.json') },
    { code: 'SR_FUTURE_CODE', text: 'Future code.', value: '<b>x</b>', source: source('run_context.json') },
  ]),
  limitations: page([
    { code: 'SR_FINDING', value: { code: 'X' }, source: source('scientific_readiness.json') },
    { code: 'SR_CLAIM_CEILING', value: 'analysis_only', source: source('scientific_plan_review.json') },
  ]),
  number_check: {
    ...page([
      { code: 'SR_NUMBER_BOUND', value: '0.82', surface: 'answer', source: source('result_tables.json', { evidence_id: 'ev1', sha256: 'b'.repeat(64) }) },
      { code: 'SR_NUMBER_UNBOUND', value: '1.37', surface: 'manuscript', source: source('manuscript_draft.json') },
    ]),
    counts: { bound: 1, unbound: 1, suppressed: 0 }, coverage: { answer: 'recorded_inputs', manuscript: 'unavailable' },
  },
};

function owner(load) {
  const declared = {};
  const window = { EU_LANG: 'zh', EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } } };
  const context = vm.createContext({ window });
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-guided-pi-review-items.js'), 'utf8'), context, { filename: 'review-items' });
  const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const buttons = [];
  const resourceButton = (ref, label) => { buttons.push(JSON.parse(JSON.stringify(ref))); return `<button data-ref="${esc(ref.artifact)}">${esc(label)}</button>`; };
  return { items: declared.reviewItems.create({
    tr: (en, zh) => zh, esc, projectId: () => 'p1', resourceButton,
    api: () => ({ loadPiRunScientificReview: load || (async () => ({ review: REVIEW })) }),
  }), buttons };
}

test('approach items follow the reviewed order, in plain words, with their source', async () => {
  const { items, buttons } = owner();
  const run = { run_id: 'r1' };
  assert.equal(items.panelHtml(run, 'approach'), '', 'nothing before the host review is read');
  assert.equal(await items.load(run), true);
  const html = items.panelHtml(run, 'approach');
  assert.ok(html.indexOf('研究人群') < html.indexOf('结局'));
  assert.match(html, /<p>age: &gt;= 18；first_stay: 是<\/p>/);
  assert.match(html, />查看来源<\/button>/);
  assert.deepEqual(buttons[0], { kind: 'research_artifact', run_id: 'r1', artifact: 'agent_plan.json', sha256: SHA, label: 'agent_plan.json' });
});

test('a missing data identity says so, and an unknown code falls back to the host text as text', async () => {
  const { items } = owner();
  await items.load({ run_id: 'r1' });
  const html = items.panelHtml({ run_id: 'r1' }, 'materials');
  assert.match(html, /data-review-item="SR_DATA_EXPORT_ID"><div><strong>导出编号<\/strong><p>未记录<\/p>/);
  assert.match(html, /class="gpi-review-item is-defect" data-review-item="SR_DATA_EXPORT_ID"/);
  assert.match(html, /<strong>Future code\.<\/strong><p>&lt;b&gt;x&lt;\/b&gt;<\/p>/);
});

test('numbers without evidence lead, are marked, and the counts say what was checked', async () => {
  const { items, buttons } = owner();
  await items.load({ run_id: 'r1' });
  const html = items.panelHtml({ run_id: 'r1' }, 'numbers');
  assert.match(html, /1 个数字有证据记录，1 个没有，0 个是按小样本规则隐藏的数。尚无稿件，只核对了回答中的数字。/);
  assert.ok(html.indexOf('1.37') < html.indexOf('0.82'));
  assert.match(html, /is-defect" data-review-item="SR_NUMBER_UNBOUND"><div><strong>1\.37<\/strong><p>没有证据记录 · 稿件<\/p>/);
  assert.equal(buttons.find(ref => ref.evidence_id === 'ev1').evidence_pointer, '/x');
});

test('items the card already shows can be left out; a failed read leaves the card as it was', async () => {
  const { items } = owner();
  await items.load({ run_id: 'r1' });
  const html = items.panelHtml({ run_id: 'r1' }, 'limitations', item => item.code === 'SR_FINDING');
  assert.doesNotMatch(html, /审阅意见/);
  assert.match(html, /结论可用范围/);
  const failing = owner(async () => { throw Object.assign(new Error('x'), { code: 'scientific_review_privacy_blocked' }); }).items;
  await failing.load({ run_id: 'r1' });
  assert.equal(failing.panelHtml({ run_id: 'r1' }, 'approach'), '');
  assert.equal(failing.panelHtml({ run_id: 'r1' }, 'numbers'), '');
});
