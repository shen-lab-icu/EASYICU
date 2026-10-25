'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh', EU_CATALOG: { dict: {
  age: ['Age', '年龄'], sex: ['Sex', '性别'], lact: ['Lactate', '乳酸'], death: ['In-hospital Mortality', '院内死亡'],
  los_icu: ['ICU Length of Stay', 'ICU住院时长'],
  hr: ['Heart Rate', '心率'], sofa: ['SOFA Score', 'SOFA评分'], gcs: ['Glasgow Coma Scale', '格拉斯哥昏迷评分'],
} } };
for (const file of process.argv.slice(2)) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
let language = 'zh';
const tr = (en, zh) => language === 'zh' ? zh : en;
const esc = window.EU_HTML.esc;
const lines = [
  { key: 'population', criteria: [{ concept: 'age', op: '>=', value: 18 }], unresolved: false },
  { key: 'comparison', mode: 'association', variables: [{ concept: 'lact' }], groups: [{ en: '<4 mmol/L', zh: '<4 mmol/L' }, { en: '≥4 mmol/L', zh: '≥4 mmol/L' }], unresolved: false },
  { key: 'outcome', variables: [{ concept: 'death' }], unresolved: false },
  { key: 'adjustment', mode: 'adjusted', variables: [{ concept: 'age' }, { concept: 'sex' }], unresolved: false },
  { key: 'data_checks', missing: 'complete_case' },
];
const workflow = { next_action_code: 'operator_plan_approval_required',
  plan_review_summary: { run_id: 'synthetic' },
  plan_conversation_preview: { items: [{ key: 'population_and_unit', text: 'patient_stay_id arbitrary model prose' }],
    clinical_summary: { lines, timing: { anchor: 'icu_admit', hours: 24, risk_set: true } },
    design: { primary_method: 'Exact method retained' } },
};
const confirmation = modules.require('confirmation').create({ tr, esc, iconHtml: () => '',
  resourceButton: () => '', sessionIsStale: () => false, workflow: () => workflow,
  session: () => ({ archived_child_jobs: [], binding: { run_id: 'synthetic' } }), busy: () => false });
for (language of ['zh', 'en']) {
  window.EU_LANG = language;
  const html = confirmation.workflowConfirmationHtml();
  assert.ok(!html.includes('patient_stay_id') && !html.includes('arbitrary model prose'));
  const summary = html.match(/<ol class="gpi-clinical-lines">([\s\S]*?)<\/ol>/)?.[1];
  assert.ok(summary, 'five clinical lines are rendered by the confirmation owner');
  assert.equal((summary.match(/<li>/g) || []).length, 5);
  const labels = language === 'zh' ? ['研究人群', '比较', '结局', '调整', '数据检查'] : ['Population', 'Comparison', 'Outcome', 'Adjustment', 'Data checks'];
  assert.deepEqual([...summary.matchAll(/<b>(.*?)<\/b>/g)].map(row => row[1]), labels);
  const names = window.EU_PRODUCT_LABELS;
  const card = names.clinicalPlan(workflow.plan_conversation_preview.clinical_summary, tr);
  assert.deepEqual(card.lines.map(row => row.text), language === 'zh' ? [
    '年龄 ≥18 岁的 ICU 患者', '乳酸 <4 mmol/L 与 ≥4 mmol/L 两组', '院内死亡', '年龄、性别',
    '完整病例分析；运行前核对缺失与可用性',
  ] : [
    'ICU patients aged ≥18', 'Lactate <4 vs ≥4 mmol/L', 'In-hospital mortality', 'Age, sex',
    'Complete-case analysis; availability checked before the run',
  ]);
  assert.equal(card.timing, tr(
    'Follow-up starts 24 hours after ICU admission and includes only patients still in hospital then.',
    '从入 ICU 后第 24 小时开始随访，只纳入此时仍在院的患者。'));
  assert.ok(card.lines.every(row => !/[.。]$/.test(row.text)), 'only timing ends with a full stop');
  if (language === 'zh') {
    for (const row of summary.matchAll(/<em>(.*?)<\/em>/g)) assert.ok(row[1].replaceAll('&lt;', '<').length <= 48, row[1]);
    assert.ok(html.includes('从入 ICU 后第 24 小时开始随访，只纳入此时仍在院的患者。'));
  }
  assert.ok(html.includes('≥4 mmol/L') && html.includes(language === 'zh' ? '&lt;4 mmol/L' : '&lt;4 vs'));
  assert.ok(html.includes('Exact method retained'));
  assert.ok(!html.includes('<footer>'));
  workflow.plan_conversation_preview.clinical_summary.timing = null;
  assert.ok(!confirmation.workflowConfirmationHtml().includes('gpi-plan-timing'));
  workflow.plan_conversation_preview.clinical_summary.timing = { anchor: 'icu_admit', hours: 24, risk_set: true };
  const changed = structuredClone(lines);
  changed[1].variables = [{ concept: 'unknown_internal_column' }];
  changed[1].groups = [];
  const projected = names.clinicalPlan({ lines: changed }, tr);
  assert.ok(!JSON.stringify(projected).includes('unknown_internal_column'));
  assert.match(projected.lines[1].text, /待确认|to be confirmed/);
  changed[2].variables = [{ concept: 'los_icu' }];
  assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[2].text, tr('ICU length of stay', 'ICU 住院时长'));
  changed[1] = { key: 'comparison', mode: 'association', continuous: true,
    variables: [{ concept: 'measured_marker', label: tr('Maximum lactate', '最高乳酸') }] };
  assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[1].text, tr('Maximum lactate levels', '最高乳酸水平'));
  changed[1].variables = [{ concept: 'patient_stay_id', label: tr('Age', '年龄') }];
  assert.match(names.clinicalPlan({ lines: changed }, tr).lines[1].text, /待确认|to be confirmed/);
  changed[1].variables = [{ concept: 'lact', label: '<img src=x onerror=bad()>' }];
  assert.ok(!JSON.stringify(names.clinicalPlan({ lines: changed }, tr)).includes('<img'));
  changed[1] = { key: 'comparison', mode: 'descriptive', variables: [{ concept: 'sex' }] };
  changed[3] = { key: 'adjustment', mode: 'descriptive' };
  changed[0] = { key: 'population', criteria: [], all_rows: true };
  changed[4].missing = 'unresolved';
  const descriptive = names.clinicalPlan({ lines: changed }, tr);
  assert.deepEqual(descriptive.lines.map(row => row.label), language === 'zh'
    ? ['研究人群', '分组', '结局', '调整', '数据检查'] : ['Population', 'Groups', 'Outcome', 'Adjustment', 'Data checks']);
  assert.deepEqual(descriptive.lines.map(row => row.text), language === 'zh' ? [
    '所提供队列的全部患者', '按性别分层', 'ICU 住院时长', '不做调整，仅分层描述',
    '缺失处理待确认；运行前核对缺失与可用性',
  ] : [
    'All patients in the supplied cohort', 'By sex', 'ICU length of stay', 'None; stratified description only',
    'Missing-data handling to be confirmed; availability checked before the run',
  ]);
  changed[1] = { key: 'comparison', mode: 'prediction', variables: [{ concept: 'age' }, { concept: 'sex' }, { concept: 'lact' }] };
  changed[3] = { key: 'adjustment', mode: 'prediction' };
  changed[4].missing = 'training_imputation';
  const prediction = names.clinicalPlan({ lines: changed }, tr);
  assert.deepEqual(prediction.lines.map(row => row.label), language === 'zh'
    ? ['研究人群', '预测变量', '结局', '评估', '数据检查'] : ['Population', 'Predictors', 'Outcome', 'Evaluation', 'Data checks']);
  assert.equal(prediction.lines[1].text, tr('Age, sex, lactate', '年龄、性别、乳酸'));
  assert.equal(prediction.lines[3].text, tr('Predictions against observed outcomes', '预测与实际结局对照'));
  assert.equal(prediction.lines[4].text, tr(
    'Imputation using training-set information only; availability checked before the run',
    '仅用训练集信息填补缺失；运行前核对缺失与可用性'));
  changed[3].evaluation = 'discrimination_calibration';
  assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[3].text, tr(
    'Predictions against observed outcomes (discrimination and calibration)', '预测与实际结局对照（区分度与校准）'));
  for (const key of [1, 3]) {
    changed[key] = { key: key === 1 ? 'comparison' : 'adjustment', mode: key === 1 ? 'prediction' : 'adjusted',
      variables: ['age', 'sex', 'lact', 'hr'].map(concept => ({ concept })) };
    assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[key].text, tr('Age, sex, lactate, heart rate', '年龄、性别、乳酸、心率'));
    changed[key].variables.push({ concept: 'sofa' });
    assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[key].text, tr('Age, sex, lactate and 2 more', '年龄、性别、乳酸等 5 项'));
    changed[key].variables.push({ concept: 'gcs' });
    assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[key].text, tr('Age, sex, lactate and 3 more', '年龄、性别、乳酸等 6 项'));
  }
  changed[3].variables = [{ concept: 'age' }, { concept: 'gcs' }, { concept: 'sofa' }];
  assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[3].text, tr(
    'Age, Glasgow coma scale, SOFA score', '年龄、格拉斯哥昏迷评分、SOFA 评分'));
  changed[3].variables = [{ concept: 'baseline_score', label: tr('Pre-ICU SOFA Score', '入ICU时SOFA评分') }];
  assert.equal(names.clinicalPlan({ lines: changed }, tr).lines[3].text, tr('Pre-ICU SOFA score', '入 ICU 时 SOFA 评分'));
  changed[3] = { key: 'adjustment', mode: 'adjusted', unresolved: true, variables: [
    { concept: 'first_marker', label: tr('A long but registered baseline measurement label', '已登记的较长临床测量名称'.repeat(3)) },
    { concept: 'second_marker', label: tr('Another long registered baseline measurement label', '另一项已登记的较长测量名称'.repeat(3)) },
  ] };
  const bounded = names.clinicalPlan({ lines: changed }, tr).lines[3].text;
  assert.ok(bounded.length <= (language === 'zh' ? 48 : 120));
  assert.match(bounded, /待确认|to be confirmed/, 'shortening never clips uncertainty');
  assert.ok(!/^(Include|Measure|Adjust|Compare|Predict|纳入|观察|调整|比较|用)/.test(bounded));
  const zero = names.clinicalPlan({ lines, timing: { anchor: 'icu_admit', hours: 0 } }, tr);
  assert.equal(zero.timing, tr('Follow-up starts at ICU admission.', '从入 ICU 开始随访。'));
}
console.log('clinical plan card: 2 languages, 3 families, timing, disclosure and unknown identifiers passed');
