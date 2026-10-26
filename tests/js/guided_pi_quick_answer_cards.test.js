'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function owners(language = 'zh') {
  const declared = {};
  const esc = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const window = {
    EU_LANG: language,
    EU_HTML: { esc },
    EU_CATALOG: { dict: { lact: ['Lactate', '乳酸'], sex: ['Sex', '性别'], age: ['Age', '年龄'] } },
    EU_DATABASE_NAMES: { name: key => ({ eicu_demo: 'eICU-CRD Demo', hirid: 'HiRID' }[key] || '') },
    EasyICU: { guidedPi: { declare: (name, value) => { declared[name] = value; }, optional: name => declared[name] || null } },
  };
  const context = vm.createContext({ window });
  for (const file of ['screens-guided-pi-source-adjustments.js', 'screens-guided-pi-quick-answers.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context);
  }
  const acts = [];
  let off = false;
  const tr = (en, zh) => (language === 'zh' ? zh : en);
  const cards = declared.quickAnswers.create({ tr, esc, act: async choice => acts.push(choice), disabled: () => off });
  return { cards, acts, disable: () => { off = true; }, owner: declared.quickAnswers };
}

const READING = {
  question: '入 ICU 首日最高乳酸的中位数？', intent: 'describe', population: { criteria: [] },
  measure: { concept: 'lact', statistic: 'median', summary: 'max', window: { start_hours: 0, end_hours: 24 }, level: null },
  grouping: [], analysis_unit: 'icu_stay', adjustments: [], endpoint_kind: 'observed', ambiguities: [],
};
const SOURCE = { database: 'eicu_demo', registration_id: 'src_synthetic' };
const evidence = field => ({ evidence_id: 'table_quick_result_1', sha256: 'e'.repeat(64), source_field: field, source: SOURCE });

function attachment(overrides = {}) {
  return {
    quick_answer_id: 'qa_1', message_job_id: 'job-1', study_context_id: 'study-1', answer: null,
    proposal: {
      status: 'approval_required', reason: 'quick_qualified',
      expand: { action: 'expand_into_full_study', reading: READING, source: SOURCE },
      card: {
        sha256: 'c'.repeat(64), source: SOURCE,
        plan: { reading: READING, effective_reading: READING, qualification: [] },
        capability: { status: 'exact', database: 'eicu_demo', substitutions: [], missing: [] },
      },
    },
    ...overrides,
  };
}

function rendered(cards, item) {
  return cards.render({ role: 'quick_answer', id: 'quick-answer-' + item.quick_answer_id, attachment: item });
}

test('a card shows the host reading and sends the displayed digest on its own click', async () => {
  const { cards, acts } = owners();
  const html = rendered(cards, attachment());
  assert.match(html, /快速回答/);
  assert.match(html, /eICU-CRD Demo/);
  assert.match(html, /入 ICU 0–24 小时乳酸最高值的中位数/);
  assert.match(html, /全部 ICU 入住/);
  assert.match(html, /data-gpi-quick-approve="qa_1" data-gpi-quick-card="c{64}" data-gpi-quick-accept="false"/);
  assert.match(html, />计算</);
  const button = { disabled: false, dataset: { gpiQuickApprove: 'qa_1', gpiQuickCard: 'c'.repeat(64), gpiQuickAccept: 'false' } };
  assert.equal(cards.handleClick({ target: { closest: selector => (selector === '[data-gpi-quick-approve]' ? button : null) } }), true);
  assert.deepEqual(JSON.parse(JSON.stringify(acts)), [{ code: 'approve_quick_answer', quick_answer_id: 'qa_1',
    card_sha256: 'c'.repeat(64), accept_substitutions: false }]);
});

test('a substitution is listed and accepted only by the button that names it', () => {
  const { cards } = owners();
  const item = attachment();
  item.proposal.card.capability = { status: 'substitution', database: 'hirid', missing: [], substitutions: [{
    capability: 'threshold:age', original: { concept: 'age', operator: '>=', value: 18 },
    substitute: { concept: 'age', operator: '>=', value: 20 }, source_field: null,
  }] };
  const html = rendered(cards, item);
  assert.match(html, /HiRID 上的口径调整/);
  assert.match(html, /年龄 ≥ 18 岁/);
  assert.match(html, /年龄 ≥ 20 岁/);
  assert.match(html, /data-gpi-quick-accept="true"[^>]*>同意口径调整并计算</);
});

test('an answer shows only the values of its evidence-bound rows', () => {
  const { cards } = owners();
  const rows = [
    { statistic: 'cohort_n', variable: '__cohort__', group: 'Overall', level: '', unit: '', value: 60, evidence: evidence('rows[0].value') },
    { statistic: 'nonmissing_n', variable: 'lact_max', group: 'Overall', level: '', unit: '', value: 58, evidence: evidence('rows[1].value') },
    { statistic: 'missing_n', variable: 'lact_max', group: 'Overall', level: '', unit: '', value: 2, evidence: evidence('rows[2].value') },
    { statistic: 'median', variable: 'lact_max', group: 'Overall', level: '', unit: 'mmol/L', value: 4, evidence: evidence('rows[3].value') },
  ];
  const html = rendered(cards, attachment({ answer: { status: 'ok', reason: 'quick_answer_ready', source: SOURCE, rows,
    sentences: rows.map(row => ({ text: `The recorded value is ${row.value}.`, evidence: row.evidence })),
    expand: { reading: READING, source: SOURCE } } }));
  assert.match(html, /<td>4 mmol\/L<\/td>/);
  assert.match(html, /60 例中 58 例有值/);
  assert.doesNotMatch(html, /The recorded value/);
  assert.doesNotMatch(html, /data-gpi-quick-approve/);
  assert.match(html, /data-gpi-quick-expand="qa_1"/);
});

test('a grouped comparison reads each group and the test from their own rows', () => {
  const { cards } = owners('en');
  const rows = [];
  for (const [group, n, value] of [['Male', 30, 3.5], ['Female', 30, 4.5]]) {
    rows.push({ statistic: 'cohort_n', variable: '__cohort__', group, level: '', unit: '', value: n, evidence: evidence('x') });
    rows.push({ statistic: 'median', variable: 'lact_max', group, level: '', unit: 'mmol/L', value, evidence: evidence('y') });
  }
  rows.push({ statistic: 'p_value:mann_whitney_or_kruskal', variable: 'lact_max', group: 'Comparison', level: '', unit: '', value: 0.0412, evidence: evidence('z') });
  const html = rendered(cards, attachment({ answer: { status: 'ok', reason: 'quick_answer_ready', source: SOURCE, rows, sentences: [], expand: {} } }));
  assert.match(html, /<th scope="row">Male<\/th><td>3.5 mmol\/L<\/td><td>30 in the group<\/td>/);
  assert.match(html, /Between groups: P = 0.041 \(Mann–Whitney \/ Kruskal–Wallis test\)/);
});

test('a question that is not quick says why and offers the full study, never a calculation', () => {
  const { cards } = owners();
  const html = rendered(cards, attachment({ proposal: { status: 'full_study', reason: 'quick_requires_adjusted_study',
    expand: { action: 'expand_into_full_study', reading: READING, source: SOURCE } } }));
  assert.match(html, /需要调整其他因素/);
  assert.match(html, /class="btn primary sm" type="button" data-gpi-quick-expand="qa_1"/);
  assert.doesNotMatch(html, /data-gpi-quick-approve/);
  const unknown = rendered(cards, attachment({ proposal: { status: 'stopped', reason: 'quick_unlisted_code', expand: {} } }));
  assert.match(unknown, /这个问题不适合快速回答/);
  assert.doesNotMatch(unknown, /quick_unlisted_code/);
});

test('a busy or stale conversation dispatches nothing', () => {
  const { cards, acts, disable } = owners();
  disable();
  const html = rendered(cards, attachment());
  assert.match(html, /data-gpi-quick-approve="qa_1"[^>]* disabled/);
  const button = { disabled: false, dataset: { gpiQuickExpand: 'qa_1' } };
  assert.equal(cards.handleClick({ target: { closest: selector => (selector === '[data-gpi-quick-expand]' ? button : null) } }), true);
  assert.equal(acts.length, 0);
});

test('each card follows the turn that produced it', () => {
  const { cards } = owners();
  const session = {
    quick_answers: [attachment(), attachment({ quick_answer_id: 'qa_2', message_job_id: 'job-unpaged' })],
    conversation_replay: { turns: [{ job_id: 'job-1', started_at: '2026-10-26T10:00:00Z', ended_at: '2026-10-26T10:00:05Z' }] },
  };
  const placed = cards.place([
    { id: 'u1', role: 'user', timelineAt: Date.parse('2026-10-26T10:00:00Z') },
    { id: 'u2', role: 'user', timelineAt: Date.parse('2026-10-26T10:05:00Z') },
  ], cards.rows(session));
  assert.deepEqual(placed.map(row => row.id), ['u1', 'quick-answer-qa_1', 'u2', 'quick-answer-qa_2']);
});

test('a qualified card holds the automatic full plan until it becomes a full study', () => {
  const { owner } = owners();
  const card = (overrides = {}) => attachment({ study_context_id: null, ...overrides });
  const asked = { study_setup_receipt: { configuration: { question: READING.question } } };
  const other = { study_setup_receipt: { configuration: { question: '另一个研究问题：乳酸与死亡的关系？' } } };
  // The card the turn just made, and a card on the study's own question.
  assert.equal(owner.holdsAutomaticPlan({ last_message_job_id: 'job-1', quick_answers: [card()] }, other), true);
  assert.equal(owner.holdsAutomaticPlan({ last_message_job_id: 'job-later', quick_answers: [card()] }, asked), true);
  const answered = card({ answer: { status: 'ok', rows: [] } });
  assert.equal(owner.holdsAutomaticPlan({ last_message_job_id: 'job-later', quick_answers: [answered] }, asked), true);
  // Expanded into a full study, an earlier card on another question, or none.
  assert.equal(owner.holdsAutomaticPlan({ last_message_job_id: 'job-1', quick_answers: [card({ study_context_id: 'study-1' })] }, asked), false);
  assert.equal(owner.holdsAutomaticPlan({ last_message_job_id: 'job-later', quick_answers: [card()] }, other), false);
  assert.equal(owner.holdsAutomaticPlan({ last_message_job_id: 'job-1', quick_answers: [] }, asked), false);
  assert.equal(owner.holdsAutomaticPlan(null, null), false);
  // A proposal the host stopped or handed to a full study holds nothing.
  for (const status of ['stopped', 'full_study']) {
    const stopped = card({ proposal: { status, reason: 'quick_requires_adjusted_study', card: null,
      expand: { action: 'expand_into_full_study', reading: READING, source: SOURCE } } });
    assert.equal(owner.holdsAutomaticPlan({ last_message_job_id: 'job-1', quick_answers: [stopped] }, asked), false);
  }
});
