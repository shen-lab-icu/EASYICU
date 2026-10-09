/* A finished run answers in its own registered numbers; traces name each
   approved step; a job submitted from a turn keeps its outcome after reload. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'zh';
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.EU_CATALOG = { dict: {
  age: ['Age', '年龄', 'years'], sex: ['Sex', '性别', ''], adm: ['Admission Type', '入院类型', ''],
  charlson: ['Charlson Comorbidity Index', 'Charlson 合并症指数', 'points'],
} };
for (const file of process.argv.slice(2)) require(path.resolve(file));
const modules = global.EasyICU.guidedPi;
const tr = (en, zh) => zh;
const esc = global.EU_HTML.esc;
const resources = modules.require('resources').create({ esc });

const sha = char => char.repeat(64);
const ref = (artifact, char) => ({ kind: 'research_artifact', run_id: 'run_e1', artifact, sha256: sha(char) });
const latest = {
  present: true, analysis_results_available: true, analysis_validated: true, numeric_verified: true,
  run_id: 'run_e1', figure_count: 2,
  artifact_refs: [ref('evidence_ledger.json', 'a'), ref('result_tables.json', 'b'), ref('agent_plan.json', 'c'),
    ref('run_context.json', 'd'), ref('figure_gallery.json', 'e'), ref('manuscript_provenance.json', 'f')],
};
const workflow = { current_stage: 'interpretation', stages: [{ id: 'analysis', status: 'complete' }] };
const table = (name, headers, rows) => ({ name, label: `Table ${name} from step x.`, evidence_id: name, headers, rows });
const payloads = {
  'result_tables.json': { tables: [
    table('flow__cohort_analysis_flow.csv', ['n_before', 'n_excluded', 'n_remaining'], [['94458', '0', '94458']]),
    table('step__table_one.csv', ['variable', 'group'], [['age', 'Overall']]),
    table('step__exposure_outcome_distribution.csv',
      ['row_role', 'exposure_level', 'n_rows', 'exposure_denominator', 'exposure_pct', 'outcome_events', 'outcome_denominator', 'outcome_rate_pct'],
      [['exposure_level', '0.0', '62862', '94458', '66.550213', '6376', '62862', '10.142853'],
        ['exposure_level', '1.0', '31596', '94458', '33.449787', '4974', '31596', '15.742499'],
        ['overall', '', '94458', '94458', '100.0', '11350', '94458', '12.015922']]),
    table('step__missingness_audit.csv', ['concept', 'n_total'], [['death', '94458']]),
  ] },
  'agent_plan.json': {
    display_labels: {
      death: '院内死亡状态', sep3_sofa1_max: 'Sepsis-3脓毒症诊断状态（疑似感染伴传统SOFA评分至少增加2分）',
      'sep3_sofa1_max=0': '未记录Sepsis-3脓毒症诊断', 'sep3_sofa1_max=1': '记录Sepsis-3脓毒症诊断',
    },
    steps: [
      { step_id: 'baseline_context', method: 'table_one', intent: '按 Sepsis-3 分组呈现基线特征。',
        table_one_spec: { group_by: 'sep3_sofa1_max' },
        inputs: ['sep3_sofa1_max', 'age', 'sex', 'adm', 'charlson_first', 'artifact:analysis_cohort', 'sep3_sofa1_n', 'charlson_measured'] },
      { step_id: 'exposure_outcome_distribution', method: 'descriptive', planned_analysis_role: 'primary',
        intent: '比较两组的院内死亡比例。',
        exposure_outcome_distribution_spec: { exposure: 'sep3_sofa1_max', outcome: 'death', exposure_levels: [0, 1] } },
      { step_id: '05_descriptive_context_figure', method: 'visualization', intent: 'Render the figure.' },
    ],
  },
  'run_context.json': { question: 'Sepsis-3 与院内死亡', source: { label: 'MIMIC-IV' } },
  'figure_gallery.json': { figures: [
    { placement: 'main', caption: 'Exposure distribution', data_url: 'data:image/png;base64,AAAA' },
    { placement: 'supplementary', caption: 'Data quality', data_url: 'data:image/png;base64,BBBB' },
  ] },
  'manuscript_provenance.json': { claims: [] },
};
const api = { loadPiCopilotResearchArtifact: async (_project, _run, artifact) => ({ ok: true, payload: payloads[artifact] }) };
const outcome = modules.require('runOutcome').create({
  tr, esc, iconHtml: () => '', resourceButton: resources.button, api: () => api, projectId: () => 'project_e1',
  host: () => null,
});

(async () => {
  // Before the run's numbers are read the card keeps one short sentence.
  assert.doesNotMatch(outcome.render(latest, workflow), /gpi-run-answer"/);
  await outcome.loadScientificReview(latest, workflow);
  for (let turn = 0; turn < 5; turn += 1) await new Promise(resolve => setImmediate(resolve));
  const card = outcome.render(latest, workflow);
  assert.match(card, /在 MIMIC-IV 的 94,458 个 ICU 入住记录中，「记录Sepsis-3脓毒症诊断」有 31,596 个（33\.45%）。/);
  assert.match(card, /院内死亡状态：「记录Sepsis-3脓毒症诊断」组 15\.74%（4,974\/31,596），「未记录Sepsis-3脓毒症诊断」组 10\.14%（6,376\/62,862）；总体 12\.02%。/);
  assert.match(card, /这是未调整的描述性比较，不能说明因果/);
  assert.match(card, /<img src="data:image\/png;base64,AAAA"/);
  assert.match(card, /打开全部 2 张图/);
  assert.match(card, /4 张表：分组比例与结局、基线特征、队列流程，另有 1 张数据质量审计表/);
  assert.match(card, /1 张主图、1 张补充图/);
  assert.doesNotMatch(card, /66\.550213|15\.742499/);
  const followUps = outcome.followUps(latest, workflow);
  assert.equal(followUps[0], '在调整年龄、性别、入院类型、Charlson 合并症指数后，Sepsis-3脓毒症诊断状态与院内死亡状态的关联还成立吗？');
  assert.ok(followUps.some(text => text.includes('按年龄分层')));
  assert.ok(followUps.every(text => !text.includes('sep3_sofa1')));

  // Traces: one named row per approved plan step, with how it ran.
  const activity = modules.require('activity').create({
    tr, esc, iconHtml: () => '', resourceName: () => '', resourceKey: () => '', resourceButton: () => '',
  });
  const facts = activity.planStepFacts({ step: 'step', label: 'Step 4/7 complete: measurement_audit.' });
  assert.deepEqual(facts, { index: 4, total: 7, phase: 'complete', stepId: 'measurement_audit' });
  const rows = [
    activity.planStepRow(activity.planStepFacts({ step: 'step', label: 'Step 3/7 complete: exposure_outcome_distribution.' }), 1),
    activity.planStepRow(facts, 2),
    activity.planStepRow(activity.planStepFacts({ step: 'step', label: 'Step 5/7 complete: kdigo_stage_logistic_model.' }), 3),
  ];
  assert.ok(activity.notePlanSubStep(rows, activity.planSubStepFacts({ step: 'runner', label: 'Running standard executor script for measurement_audit.' })));
  assert.equal(rows[1].text, '运行标准执行脚本，不调用模型。');
  assert.equal(activity.notePlanSubStep(rows, activity.planSubStepFacts({ step: 'coder', label: 'Using renderer for missing_step.' })), false);
  const trace = activity.render({ role: 'activity', status: 'complete', startedAt: 0, endedAt: 1000,
    steps: rows.concat([{ id: 'pipeline-step', kind: 'pipeline', step: 'resume', status: 'complete', label: '' }]) });
  assert.match(trace, /第 3\/7 步：分组比例与结局/);
  assert.match(trace, /第 4\/7 步：测量与缺失审计/);
  assert.match(trace, /第 5\/7 步：KDIGO分期Logistic模型/);
  assert.match(trace, /复用了这次获批运行已完成的步骤/);
  assert.doesNotMatch(trace, /已完成批准的分析步骤|研究任务进度已更新/);

  // A report rewrite submitted from a turn keeps its outcome after reload.
  modules.declare('replay', { lifecycleTurns: () => [] });
  const transcript = modules.require('transcript').create({
    tr, activity, timeMs: value => Date.parse(value) || 0, resourceKey: resource => (resource ? `${resource.run_id}:${resource.artifact}` : ''),
    modelErrorText: () => '', activityHasCompletedAction: () => false, workflowActionCode: () => '',
    runFailureText: code => (code === 'WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED' ? '新版本报告没有通过数字与证据校验，保留原版本。' : ''),
    upsertActivityStep(target, step) {
      const found = target.steps.find(item => item.id === step.id);
      if (found) Object.assign(found, step); else target.steps.push(step);
    },
  });
  const turn = (id, at, jobId) => [
    { role: 'user', content: [{ type: 'text', text: '请修订当前报告：统一术语。' }], timestamp: at },
    { role: 'assistant', content: [{ type: 'tool_call', tool_call_id: id, tool_name: 'easyicu_repair_report' }], timestamp: at },
    { role: 'tool', content: [{ type: 'tool_result', tool_call_id: id, tool_name: 'easyicu_repair_report', code: 'easyicu_full_run_submitted', job_id: jobId }], timestamp: at },
    { role: 'assistant', content: [{ type: 'text', text: '已提交报告修订任务，任务 ID：' + jobId }], timestamp: at },
  ];
  const messages = transcript.transcriptMessages({
    transcript: turn('call-1', '2026-09-08T19:04:28Z', 'job-ok').concat(turn('call-2', '2026-09-08T19:11:38Z', 'job-failed')),
    archived_child_jobs: [
      { job_id: 'job-ok', kind: 'agent-run', status: 'done', report_only: true, report_revision_ready: true, finished_at_epoch: 1788894926,
        artifact_refs: [{ run_id: 'run_e1', artifact: 'manuscript_revision.pdf', sha256: sha('9') }] },
      { job_id: 'job-failed', kind: 'agent-run', status: 'failed', report_only: true,
        error_code: 'WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED', finished_at_epoch: 1788895000 },
    ],
  });
  const texts = messages.filter(row => row.role === 'assistant').map(row => row.text);
  assert.ok(texts.includes('报告已基于封存的分析结果重写，并生成了新的 PDF；没有重跑分析。'));
  assert.ok(texts.includes('报告重写没有完成：新版本报告没有通过数字与证据校验，保留原版本。'));
  assert.ok(!texts.some(text => text.includes('任务 ID')), 'The handoff reply with its job id stays hidden');
  const done = messages.find(row => row.id === 'submitted-job-job-ok');
  assert.equal(done.resources[0].conversation_label, '当前报告 PDF');
  process.stdout.write('Run answer, plan-step traces and submitted-job outcomes passed.\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
