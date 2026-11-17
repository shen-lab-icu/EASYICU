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

test('a median is shown with its quartiles from their own rows', () => {
  const rows = [
    { statistic: 'cohort_n', variable: '__cohort__', group: 'Overall', level: '', unit: '', value: 60, evidence: evidence('a') },
    { statistic: 'nonmissing_n', variable: 'lact_max', group: 'Overall', level: '', unit: '', value: 58, evidence: evidence('b') },
    { statistic: 'median', variable: 'lact_max', group: 'Overall', level: '', unit: 'mmol/L', value: 1.8, evidence: evidence('c') },
    { statistic: 'q1', variable: 'lact_max', group: 'Overall', level: '', unit: 'mmol/L', value: 1.2, evidence: evidence('d') },
    { statistic: 'q3', variable: 'lact_max', group: 'Overall', level: '', unit: 'mmol/L', value: 3.1, evidence: evidence('e') },
  ];
  const answer = { answer: { status: 'ok', reason: 'quick_answer_ready', source: SOURCE, rows, sentences: [], expand: {} } };
  const zh = rendered(owners().cards, attachment(answer));
  assert.match(zh, /<td>1.8（1.2–3.1） mmol\/L<\/td>/);
  assert.match(zh, /乳酸最高值的中位数（四分位数间距）/);
  const en = rendered(owners('en').cards, attachment(answer));
  assert.match(en, /<td>1.8 \(IQR 1.2–3.1\) mmol\/L<\/td>/);
});

test('a continuous distribution card names and renders the fixed summary', () => {
  const { cards } = owners('en');
  const item = attachment();
  const reading = { ...READING, question: 'Describe the distribution of age by sex.', grouping: ['sex'],
    measure: { concept: 'age', statistic: 'distribution' } };
  Object.assign(item.proposal.card.plan, { reading, effective_reading: reading, measure_levels: [] });
  const plan = rendered(cards, item);
  assert.match(plan, /N and missing count, median \(IQR\), and mean \(SD\) of Age/);
  assert.match(plan, /<dt>Groups<\/dt><dd>Sex<\/dd>/);
  const rows = Object.entries({ cohort_n: 30, nonmissing_n: 18, missing_n: 12, median: 60, q1: 50, q3: 70,
    mean: 59, std: 9.2 }).map(([statistic, value]) => ({ statistic, value, group: 'Female', level: '',
    unit: statistic.endsWith('_n') ? '' : 'years', evidence: evidence(statistic) }));
  item.answer = { status: 'ok', source: SOURCE, rows };
  const answer = rendered(cards, item);
  assert.match(answer, /<th scope="row">Female<\/th>/);
  assert.match(answer, /60 \(IQR 50–70\) years/);
  assert.match(answer, /Mean 59 \(SD 9.2\) years/);
  assert.match(answer, /18 of 30 with a value.*12 missing/);
  assert.doesNotMatch(answer, /<svg|<img|<canvas/);
});

test('a categorical distribution card renders every level and its proportion', () => {
  const { cards } = owners();
  const item = attachment();
  const reading = { ...READING, question: '性别分布如何？', measure: { concept: 'sex', statistic: 'distribution' } };
  Object.assign(item.proposal.card.plan, { reading, effective_reading: reading, measure_levels: ['Female', 'Male'] });
  assert.match(rendered(cards, item), /按类别的计数和比例/);
  const rows = [['Female', 18, 60], ['Male', 12, 40]].flatMap(([level, count, percent]) =>
    [['numerator', count], ['denominator', 30], ['percentage', percent]].map(([statistic, value]) =>
      ({ statistic, value, group: 'Overall', level, unit: statistic === 'percentage' ? '%' : '', evidence: evidence(statistic) })));
  item.answer = { status: 'ok', source: SOURCE, rows };
  const answer = rendered(cards, item);
  assert.match(answer, /Female 18 \/ 30（60%）/);
  assert.match(answer, /Male 12 \/ 30（40%）/);
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

test('a pooled group is named and a withheld statistic says why, never a blank cell', () => {
  const rows = [];
  for (const [group, n, value] of [['Male', 30, 3.5], ['Other groups (pooled)', 12, 4.1], ['Female', 18, null]]) {
    rows.push({ statistic: 'cohort_n', variable: '__cohort__', group, level: '', unit: '', value: n, evidence: evidence('x') });
    if (value !== null) rows.push({ statistic: 'median', variable: 'lact_max', group, level: '', unit: 'mmol/L', value, evidence: evidence('y') });
  }
  const answer = { answer: { status: 'ok', reason: 'quick_answer_ready', source: SOURCE, rows, sentences: [], expand: {} } };
  const zh = rendered(owners().cards, attachment(answer));
  assert.match(zh, /<th scope="row">其他组（合并）<\/th><td>4.1 mmol\/L<\/td>/);
  assert.match(zh, /<th scope="row">Female<\/th><td>有值例数过少，按隐私规则不显示<\/td><td>共 18 例<\/td>/);
  assert.match(zh, /人数过少的组按隐私规则合并为“其他组”显示。/);
  const en = rendered(owners('en').cards, attachment(answer));
  assert.match(en, /<th scope="row">Other groups \(pooled\)<\/th>/);
  assert.match(en, /Too few values to show under the privacy rule/);
});

test('a comparison over more than two groups says no test ran, before and after approval', () => {
  const { cards } = owners('zh');
  const compare = { ...READING, intent: 'compare', grouping: ['sex'] };
  const base = attachment();
  base.proposal.card.plan = { reading: compare, effective_reading: compare, qualification: [], test: null };
  assert.match(rendered(cards, base), /未做组间检验：快速回答只对两组做检验。/);
  const rows = [];
  for (const [group, value] of [['Male', 3.5], ['Female', 4.5], ['Unknown', 2.5]]) {
    rows.push({ statistic: 'median', variable: 'lact_max', group, level: '', unit: 'mmol/L', value, evidence: evidence('y') });
  }
  const answered = attachment({ answer: { status: 'ok', reason: 'quick_answer_ready', source: SOURCE, rows, sentences: [], expand: {} } });
  answered.proposal.card.plan = base.proposal.card.plan;
  const html = rendered(cards, answered);
  assert.match(html, /<th scope="row">Unknown<\/th>/);
  assert.match(html, /未做组间检验/);
  assert.doesNotMatch(html, /组间比较：P =/);
  const tested = attachment({ answer: { status: 'ok', reason: 'quick_answer_ready', source: SOURCE, sentences: [], expand: {},
    rows: [...rows.slice(0, 2), { statistic: 'p_value:mann_whitney_or_kruskal', variable: 'lact_max', group: 'Comparison', level: '', unit: '', value: 0.04, evidence: evidence('z') }] } });
  tested.proposal.card.plan = { ...base.proposal.card.plan, test: 'mann_whitney_or_kruskal' };
  assert.doesNotMatch(rendered(cards, tested), /未做组间检验/);
});

test('a question that is not quick says why and offers the full study, never a calculation', () => {
  const { cards } = owners();
  const html = rendered(cards, attachment({ proposal: { status: 'full_study', reason: 'quick_requires_adjusted_study',
    expand: { action: 'expand_into_full_study', reading: READING, source: SOURCE } } }));
  assert.match(html, /需要调整其他因素/);
  assert.match(html, /class="btn primary sm" type="button" data-gpi-quick-expand="qa_1"/);
  assert.doesNotMatch(html, /data-gpi-quick-approve/);
  // Every kind of full study the host qualification names has its own reason.
  for (const [reason, pattern] of [['quick_requires_interval_study', /需要报告置信区间/], ['quick_requires_prediction_study', /需要建立预测模型/],
    ['quick_requires_causal_study', /涉及因果效应/], ['quick_requires_clustering_study', /需要识别患者分型/]]) {
    assert.match(rendered(cards, attachment({ proposal: { status: 'full_study', reason, expand: {} } })), pattern);
  }
  const unknown = rendered(cards, attachment({ proposal: { status: 'stopped', reason: 'quick_unlisted_code', expand: {} } }));
  assert.match(unknown, /这个问题不适合快速回答/);
  assert.doesNotMatch(unknown, /quick_unlisted_code/);
});

test('a stopped comparison over repeated stays offers the first-stay question only where it can be prepared', () => {
  const cases = [
    ['quick_repeated_units_ask_first_stay', 'zh', /同一患者有多次 ICU 入住。可改问每位患者的首次 ICU 入住，或改为完整研究。/],
    ['quick_repeated_units_ask_first_stay', 'en', /The same patient has several ICU stays\. Ask about each patient's first ICU stay, or make it a full study\./],
    ['quick_repeated_units_require_full_study', 'zh', /同一患者有多次 ICU 入住，这个比较需要在完整研究里处理。/],
    ['quick_repeated_units_require_full_study', 'en', /The same patient has several ICU stays; this comparison needs the full study\./],
  ];
  for (const [reason, language, pattern] of cases) {
    const { cards } = owners(language);
    const html = rendered(cards, attachment({ answer: { status: 'stopped', reason,
      source: SOURCE, rows: [], sentences: [], expand: { reading: READING, source: SOURCE } } }));
    assert.match(html, pattern);
    if (reason === 'quick_repeated_units_require_full_study') assert.doesNotMatch(html, /首次 ICU 入住|first ICU stay/);
    assert.match(html, /data-gpi-quick-expand="qa_1"/);
    assert.doesNotMatch(html, /data-gpi-quick-approve/);
  }
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
