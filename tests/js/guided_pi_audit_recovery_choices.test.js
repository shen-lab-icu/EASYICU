'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const root = process.env.EASYICU_TEST_OWNER_ROOT || path.resolve(__dirname, '../..');
global.window = { EU_LANG: 'zh' };
for (const file of ['screens-guided-pi-modules', 'html-escape', 'screens-guided-pi-plan-actions',
  'screens-guided-pi-confirmation', 'screens-guided-pi-regeneration', 'screens-guided-pi-question-requirements',
  'screens-guided-pi-study-readings', 'screens-guided-pi-target-trial-copy', 'screens-guided-pi-target-trial']) {
  require(path.join(root, 'src/easyicu/webserver/static/js', file + '.js'));
}
const modules = window.EasyICU.guidedPi;
const tr = (en, zh) => window.EU_LANG === 'zh' ? zh : en;
const esc = window.EU_HTML.esc;
const summary = () => ({ run_id: 'candidate-a', authorization_questions: [], automatic_revision_blockers: [],
  remediation_buckets: { agent_plan_revision: ['SYNTHETIC_PLAN_REPAIR'] } });
function fixture(code, extra = {}) {
  const workflow = { next_action_code: code, plan_review_summary: summary(), ...extra };
  const calls = [];
  const session = { session_id: 'session-a', binding: { study_context_id: 'study-a', study_revision: 7, run_id: 'candidate-a' } };
  const host = { tr, esc, iconHtml: () => '', workflow: () => workflow, session: () => session,
    busy: () => false, sessionIsStale: () => false, researchSourceReady: () => true,
    resourceButton: (ref, label) => `<button data-file="${ref.artifact}" data-run="${ref.run_id}">${esc(label)}</button>`,
    studyFamilyGapText: () => ({ refusal: 'This source does not support the design.', instead: 'Change the design or source.' }),
    projectId: () => 'project-a', render: () => {}, setBusy: () => {}, setError: () => {},
    appendMessage: () => {}, errorText: e => { throw e; }, watchChildJob: (...args) => calls.push(args),
    api: () => ({
      loadStudyContext: async () => ({ question: 'Compare synthetic measurements.', data_source: { path: '/synthetic/source' } }),
      startAgentRun: async request => { calls.push(request); return { job_id: 'job-a' }; },
    }),
  };
  return { workflow, session, calls, host, card: modules.require('confirmation').create(host) };
}

for (const lang of ['zh', 'en']) {
  test(`${lang}: resumable progress and reprepare have one distinct next action`, () => {
    window.EU_LANG = lang;
    const resumable = fixture('planner_checkpoint_resume_available');
    assert.equal(resumable.card.workflowConfirmation().title, tr('Plan progress is saved. Continue?', '计划生成进度已保存，继续生成？'));
    const failure = { run_id: 'attempt-b', candidate_run_id: 'candidate-a', checkpoint_resume_available: false, reason: 'preparation_failed' };
    const reprepare = fixture('plan_execution_upgrade_required', { latest_attempt_failure: failure });
    assert.equal(reprepare.card.workflowConfirmation().note, tr(
      'Plan generation did not finish. Your candidate is saved. Prepare the data again, then review the new plan.',
      '这次计划生成未完成。候选方案已保留，请重新准备数据，再核对新计划。'));
    for (const item of [resumable, reprepare]) {
      const html = item.card.workflowConfirmationHtml();
      assert.equal((html.match(/data-gpi-confirm-action/g) || []).length, 1);
      assert.equal(html.includes('data-gpi-confirm-edit'), false);
      assert.doesNotMatch(html, /Planner|checkpoint|检查点/);
    }
    assert.match(reprepare.card.workflowConfirmationHtml(), new RegExp(tr('Data preparation did not finish', '数据准备未完成')));
    assert.equal(reprepare.card.workflowConfirmationHtml().includes('data-run="candidate-a"'), true);
    const changed = fixture('plan_configuration_superseded').card.workflowConfirmation();
    assert.doesNotMatch(changed.note, /\brun\b|\bdigest\b|配置摘要/);
  });

  test(`${lang}: provider failure explains the automatic retry policy without inventing a receipt`, () => {
    window.EU_LANG = lang;
    const f = fixture('plan_scientific_changes_required', { latest_attempt_failure: {
      run_id: 'attempt-b', candidate_run_id: 'candidate-a', checkpoint_resume_available: false, reason: 'provider_unavailable',
    } });
    const html = f.card.workflowConfirmationHtml();
    assert.ok(html.includes(tr('Transient timeouts are retried once automatically', '短暂超时会自动重试一次')));
    assert.ok(html.includes(tr('The candidate is saved; analysis has not started. Resolve the issues below first.', '候选计划已保存，分析尚未开始。请先处理下列问题。')));
    assert.equal((html.match(/data-gpi-confirm-action/g) || []).length, 1);
    f.workflow.latest_attempt_failure.candidate_run_id = 'unrelated';
    assert.equal(f.card.workflowConfirmationHtml().includes('gpi-confirmation-failure'), false);
    delete f.workflow.latest_attempt_failure;
    assert.equal(f.card.workflowConfirmationHtml().includes('gpi-confirmation-failure'), false);
    const fresh = fixture('failed_pipeline_requires_fresh_plan', { runs: [{ authoritative: true,
      run_status: 'failed', gate_reason_code: 'research_pipeline_provider_timeout' }] });
    assert.ok(fresh.card.workflowConfirmation().note.includes(tr('Transient timeouts are retried once automatically', '短暂超时会自动重试一次')));
    assert.equal(fresh.card.workflowConfirmationHtml().includes('data-gpi-confirm-edit'), false);
    f.workflow.latest_attempt_failure = { run_id: 'attempt-b', candidate_run_id: 'candidate-a',
      checkpoint_resume_available: false, reason: 'preparation_failed' };
    f.workflow.runs = [{ run_id: 'attempt-b', gate_reason_code: 'research_pipeline_provider_timeout' }];
    assert.ok(f.card.workflowConfirmationHtml().includes(tr('Transient timeouts are retried once automatically', '短暂超时会自动重试一次')));
    f.workflow.runs[0].run_id = 'unrelated';
    assert.equal(f.card.workflowConfirmationHtml().includes(tr('Transient timeouts are retried once automatically', '短暂超时会自动重试一次')), false);
    f.workflow.runs = [{ run_id: 'attempt-b', gate_reason_code: 'research_pipeline_plan_contract_exhausted' }];
    assert.ok(f.card.workflowConfirmationHtml().includes(tr('Plan generation did not finish', '计划生成未完成')));
  });

  test(`${lang}: source and causal refusals draft design/source changes without approval`, () => {
    window.EU_LANG = lang;
    for (const code of ['study_family_database_out_of_scope', 'study_family_reading_disagreement', 'target_trial_review']) {
      const f = fixture(code, { target_trial_card: { state: 'stopped', reason_code: 'target_trial_database_out_of_scope' } });
      assert.equal(f.card.planChangeDraft(), tr('I want to change the study design or data source: ', '我想修改研究设计或数据来源：'));
      assert.equal(f.card.workflowConfirmation().nonApprovable, true);
      assert.equal(f.card.workflowConfirmation().grants.length, 0);
      assert.ok(f.card.workflowConfirmationHtml().includes('data-gpi-confirm-edit'));
      assert.equal(f.card.workflowConfirmationHtml().includes('data-gpi-confirm-action'), false);
      if (code === 'target_trial_review') assert.ok(f.card.workflowConfirmation().note.includes(tr(
        'Describe which study design or data source you want to change in the conversation.',
        '请在对话里说明想修改的研究设计或数据来源。')));
    }
    assert.ok(fixture('operator_plan_approval_required').card.planChangeDraft().includes(tr('complete research plan', '整份研究计划')));
  });

  test(`${lang}: study readings put reasons and variance method in details`, () => {
    window.EU_LANG = lang;
    const workflow = { study_setup_receipt: { configuration: {
      study_family_reading: { source: 'question_wording', family: 'association_study',
        elements: [{ evidence: '<synthetic association>' }], conflict: { design_family: 'descriptive_epidemiology' } },
      design_variance_basis: { basis: 'source_patient_grouping', replaced: 'none_counts_only' },
      causal_trial_reading: { source: 'question_causal_method', elements: [{ element: 'causal_method', evidence: '<synthetic method>' }] },
    } } };
    const before = JSON.stringify(workflow);
    const html = modules.require('studyReadings').notesHtml(workflow, { tr, esc });
    assert.ok(html.includes(tr('Study type: association research.', '研究类型：关联研究。')));
    assert.ok(html.includes(tr('Confidence intervals will account for repeated stays from the same patient.', '同一患者的多次入住将一并考虑，用于估计置信区间。')));
    assert.match(html, /<details[\s\S]*&lt;synthetic association&gt;[\s\S]*<\/details>/);
    assert.equal(html.includes('<details open'), false);
    assert.ok(html.includes(tr('saved design', '已保存设计')));
    assert.ok(html.includes(tr('Patient-clustered', '按患者聚类')));
    assert.equal(JSON.stringify(workflow), before);
    assert.equal(html.includes('<synthetic'), false);
    assert.equal(html.includes('<button'), false);
    const trial = modules.require('targetTrial').bodyHtml(workflow, { tr, esc });
    assert.match(trial, /<details[^>]*><summary>[\s\S]*&lt;synthetic method&gt;[\s\S]*<\/details>/);
    assert.equal(trial.includes(' open'), false);
    workflow.study_setup_receipt.configuration.study_family_reading.elements = [];
    assert.ok(modules.require('studyReadings').notesHtml(workflow, { tr, esc }).includes(tr('Study type: association research.', '研究类型：关联研究。')));
    const notes = modules.require('questionRequirements').notesHtml({ question_requirements: {
      status: 'shown', items: [{ quote: '<synthetic result>', disposition: 'claimed', source: 'plan' }],
    } }, { tr, esc });
    assert.ok(notes.includes(tr('Verification is still needed.', '尚需核对。')));
    assert.ok(notes.includes('&lt;synthetic result&gt;'));
  });

  test(`${lang}: evidence blockers offer the existing review without promising automatic repair`, () => {
    window.EU_LANG = lang;
    const f = fixture('plan_scientific_changes_required', { plan_review_summary: {
      ...summary(), automatic_revision_blockers: ['MISSING_EVIDENCE'],
      remediation_buckets: { agent_plan_revision: ['SYNTHETIC_PLAN_REPAIR'], external_evidence: ['MISSING_EVIDENCE'],
        independent_review: ['INDEPENDENT_REVIEW'], runtime_capability: ['SOURCE_UNAVAILABLE'] },
    } });
    const html = f.card.workflowConfirmationHtml();
    assert.ok(html.includes(tr('System plan repairs: 1; independent evidence or review: 2; unresolved analysis or data requirements: 1.',
      '1 项属于系统计划修订；2 项需要补充独立证据或人工审阅；1 项分析或数据条件尚未落实。')));
    assert.ok(html.includes(tr('Generating the same plan again cannot supply missing data or independent evidence.',
      '重复生成计划不能补足数据或独立证据')));
    assert.ok(html.includes('data-file="scientific_plan_review.json" data-run="candidate-a"'));
    assert.equal(html.includes('data-gpi-confirm-action'), false);
    assert.equal(html.includes('data-gpi-confirm-edit'), false);
  });
}

test('repair guards apply to explicit clicks as well as automatic progression', async () => {
  for (const fields of [{ automatic_revision_blockers: ['MISSING_EVIDENCE'] },
    { authorization_questions: [{ code: 'OUTCOME_DEFINITION_UNRESOLVED' }] },
    { remediation_buckets: { agent_plan_revision: [] } }]) {
    const f = fixture('plan_scientific_changes_required', { plan_review_summary: { ...summary(), ...fields } });
    const actions = modules.require('planActions').create(f.host);
    assert.equal(f.card.workflowConfirmation().nonApprovable, true);
    await actions.startFormalPlanGeneration('plan_scientific_changes_required');
    await actions.startFormalPlanGeneration('plan_scientific_changes_required', { automatic: true });
    assert.equal(f.calls.length, 0, JSON.stringify(fields));
  }
});

test('recovery submits the reviewed candidate and the original host decision', async () => {
  for (const code of ['plan_execution_upgrade_required', 'plan_scientific_changes_required', 'planner_checkpoint_resume_available']) {
    const f = fixture(code);
    f.session.binding.run_id = 'failed-attempt';
    f.workflow.host_decisions = { plan_transition: { expected_revision: 7, scientific_configuration_sha256: 'a'.repeat(64) } };
    const actions = modules.require('planActions').create(f.host);
    await actions.confirmWorkflow(f.card.workflowConfirmation());
    const request = f.calls[0];
    assert.ok(request, code);
    assert.equal(request.planner_start_mode, code === 'planner_checkpoint_resume_available' ? 'resume_checkpoint' : 'auto');
    assert.equal(request.plan_revision_source_run_id, code === 'planner_checkpoint_resume_available' ? '' : 'candidate-a');
    assert.deepEqual(request.host_action.decision, f.workflow.host_decisions.plan_transition);
    assert.equal(request.host_action.action_code, code === 'plan_execution_upgrade_required' ? 'prepare_analysis_data' : 'generate_plan');
    assert.equal(request.execution_resume_source_run_id, undefined);
  }
});

test('a fresh plan after a failed planning run does not resume its checkpoint', async () => {
  const f = fixture('failed_pipeline_requires_fresh_plan');
  await modules.require('planActions').create(f.host).confirmWorkflow(f.card.workflowConfirmation());
  assert.equal(f.calls[0].planner_start_mode, 'fresh');
  assert.equal(f.calls[0].plan_revision_source_run_id, '');
  assert.equal(f.calls[0].execution_resume_source_run_id, undefined);
});

test('replanning a failed approved execution retains its reviewed source', async () => {
  const f = fixture('failed_pipeline_execution_retry_futile');
  await modules.require('planActions').create(f.host).startFormalPlanGeneration('failed_pipeline_requires_fresh_plan');
  assert.equal(f.calls[0].planner_start_mode, 'auto');
  assert.equal(f.calls[0].plan_revision_source_run_id, 'candidate-a');
  assert.equal(f.calls[0].execution_resume_source_run_id, undefined);
});

test('stale, busy and passive pages never start recovery or prepare data automatically', async () => {
  for (const blocker of ['busy', 'sessionIsStale']) {
    const f = fixture('planner_checkpoint_resume_available');
    f.host[blocker] = () => true;
    await modules.require('planActions').create(f.host).confirmWorkflow(f.card.workflowConfirmation());
    assert.equal(f.calls.length, 0);
  }
  const f = fixture('plan_execution_upgrade_required');
  const actions = modules.require('planActions').create(f.host);
  await actions.startFormalPlanGeneration('plan_execution_upgrade_required', { automatic: true });
  await actions.continueSystemOwnedPlanProgression({ passive: true });
  assert.equal(f.calls.length, 0);
});

test('start-plan wording retains the original regeneration branch coordinate', () => {
  for (const text of ['开始生成研究计划', '开始生成候选研究计划。', 'Start plan generation', 'Start generating the candidate research plan.']) {
    const result = modules.require('regeneration').latestPlanRequest([
      { role: 'user', entryId: 'entry-a', text }, { role: 'assistant', id: 'answer-a' },
    ]);
    assert.deepEqual(result, { userEntryId: 'entry-a', targetMessageId: 'answer-a' }, text);
  }
  assert.equal(modules.require('regeneration').isPlanActionText('Please change the study population.'), false);
});
