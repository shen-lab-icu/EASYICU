/* One agent turn reads as a reference research agent's does: an opening
   sentence, a trace whose rows carry their own times and whose muted lines
   appear only where a job event says something changed, the step's Input and
   Output, and -- for a finished run -- a 文件 | 内容 table, notes on the run,
   an action row and follow-ups; a stopped run says why once. Synthetic jobs,
   runs and step payloads only; no run directory or patient row is read. */
'use strict';
process.env.TZ = 'UTC';
const assert = require('node:assert/strict');
const path = require('node:path');

require(path.resolve('tests/js/guided_pi_module_harness.cjs'));
const files = process.argv.slice(2);
assert.equal(files.length, 16, 'Run through tools/run_js_contracts.py with all sixteen owners');
let language = 'zh';
global.window = { EU_LANG: language, document: { addEventListener() {} },
  t: (en, zh) => (language === 'zh' ? zh : en), icon: name => (name ? `<i>${name}</i>` : '') };
global.document = { addEventListener() {}, getElementById: () => null };
for (const file of files) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
modules.declare('nextActions', { project: () => null, bodyText: () => '', render: () => '', renderPast: () => '' });
const esc = window.EU_HTML.esc;
const tr = (en, zh) => (language === 'zh' ? zh : en);
const iconHtml = name => (name ? `<i>${name}</i>` : '');
const errorText = modules.require('errorText').create({ tr, staticPreview: () => false });
const runFailureText = (...args) => errorText.runFailureText(...args);
const at = seconds => new Date(Date.UTC(2026, 0, 1) + seconds * 1000).toISOString();
const RUN = 'run_20260101T000000_turn';
const visible = html => html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');
const operations = html => [...html.matchAll(/data-gpi-operation="([^"]*)"/g)]
  .map(match => JSON.parse(decodeURIComponent(match[1].replace(/&quot;/g, '"').replace(/&amp;/g, '&'))));
const count = (html, needle) => html.split(needle).length - 1;

function activityOwner() {
  return modules.require('activity').create({ tr, esc, iconHtml, resourceName: () => '', resourceKey: () => '',
    resourceButton: () => '', runFailureText });
}
const step = (seconds, id, n, total, phase, extra) => ({ type: 'progress', step: 'step', step_id: id,
  current_step: n, total_steps: total, status: phase === 'started' ? 'running' : 'complete', timestamp: at(seconds),
  label: `Step ${n}/${total} ${phase}: ${id}.`, run_id: RUN, ...extra });
function job(events, extra) {
  const activity = { id: 'easyicu-job-j1', role: 'activity', childJobId: 'j1', status: 'complete', steps: [],
    startedAt: Date.parse(at(0)), endedAt: Date.parse(at(200)), ...extra };
  const owner = activityOwner();
  events.forEach(event => owner.notePipelineEvent(activity, event));
  if (activity.status !== 'running') owner.finishPipelineSteps(activity, Date.parse(at(200)), activity.status === 'error' ? 'error' : 'complete');
  return { owner, activity };
}

for (language of ['zh', 'en']) {
  window.EU_LANG = language;

  // --- an automatically continued analysis -------------------------------
  const continued = job([
    step(10, 'baseline_table', 1, 3, 'started'), step(13, 'baseline_table', 1, 3, 'complete'),
    { type: 'progress', step: 'provider_recovery_wait', actor: 'system', reason: 'provider_timeout', current: 1, total: 2, timestamp: at(20) },
    { type: 'progress', step: 'provider_auto_resume', actor: 'system', reason: 'provider_timeout', current: 1, total: 2, timestamp: at(80) },
    step(90, 'outcome_model', 2, 3, 'started'), step(96, 'outcome_model', 2, 3, 'complete'),
    step(100, 'robustness', 3, 3, 'started'), step(110, 'robustness', 3, 3, 'complete'),
  ], { jobCode: 'execute_plan' });
  const html = continued.owner.render(continued.activity);
  // (1) The opening sentence comes from the job's kind, before the trace.
  assert.ok(html.startsWith(`<p class="gpi-turn-opening">${tr(
    'I will run the analysis as the approved plan states and write up the results.',
    '我按批准的计划运行分析，并写出结果说明。')}</p>`));
  // (2) Each step row carries its own duration.
  assert.ok(visible(html).includes(tr('3.0 s', '3.0 秒')) && visible(html).includes(tr('6.0 s', '6.0 秒'))
    && visible(html).includes(tr('10.0 s', '10.0 秒')));
  // The wait for the model service is one muted line, with its attempt, in place of a row.
  const wait = tr('The model service is temporarily unavailable; EasyICU waits for it and then continues from the checkpoint automatically (attempt 1/2).',
    '模型服务暂时不可用，等待恢复后从检查点自动继续（第 1/2 次）。');
  assert.equal(count(html, wait), 1);
  assert.equal(count(html, 'gpi-trace-note'), 1, 'one host line, for the one event that states a change');
  assert.ok(!operations(html).some(row => /provider_recovery_wait/.test(row.id)), 'the wait is not also a row');
  // A finished trace is collapsed.
  assert.match(html, /<details class="gpi-activity complete">/);
  assert.doesNotMatch(html, /<details class="gpi-activity complete" open/);
  assert.match(visible(html), new RegExp(tr('4 steps', '4 个步骤')));

  // --- narration only with a receipt -----------------------------------
  const plain = job([step(10, 'baseline_table', 1, 1, 'started'), step(12, 'baseline_table', 1, 1, 'complete')], { jobCode: 'execute_plan' });
  assert.equal(count(plain.owner.render(plain.activity), 'gpi-trace-note'), 0);
  // A job without a known kind opens with no sentence.
  const unknown = job([step(10, 'baseline_table', 1, 1, 'started'), step(12, 'baseline_table', 1, 1, 'complete')], { jobCode: 'agent-run' });
  assert.doesNotMatch(unknown.owner.render(unknown.activity), /gpi-turn-opening/);

  // Steps done before this run: one line before the first of them.
  const prepared = job([
    { ...step(1, 'cohort_accounting', 1, 3, 'complete'), label: 'Skipped step already completed by pre-execution: cohort_accounting.' },
    { ...step(2, 'cohort_flow', 2, 3, 'complete'), label: 'Skipped step already completed by pre-execution: cohort_flow.' },
    step(3, 'outcome_model', 3, 3, 'started'), step(9, 'outcome_model', 3, 3, 'complete'),
  ], { jobCode: 'prepare_analysis_data' });
  const preparedHtml = prepared.owner.render(prepared.activity);
  assert.equal(count(preparedHtml, tr('Preparation already finished; the checked data are used directly.', '准备阶段已完成，直接使用已核对的数据。')), 1);
  assert.ok(preparedHtml.indexOf('gpi-trace-note') < preparedHtml.indexOf('data-gpi-operation'), 'the line precedes the rows it explains');
  assert.ok(preparedHtml.startsWith(`<p class="gpi-turn-opening">${tr(
    'I will first prepare the data as the confirmed design states, then plan the analysis again on the prepared data.',
    '我先按确认的方案准备数据，然后在准备好的数据上重新规划分析。')}</p>`));

  // A transport retry the runtime numbered reads as one line.
  const retried = activityOwner().render({ id: 'turn-1', role: 'activity', status: 'complete', startedAt: 0, endedAt: 5000, steps: [
    { id: 'tool-1', kind: 'tool', toolName: 'easyicu_inspect_plan', status: 'complete', startedAt: 0, endedAt: 1000 },
    { id: 'retry-2', kind: 'retry', status: 'complete', attempt: 2, maxAttempts: 3, startedAt: 1000, endedAt: 1000 },
  ] });
  assert.equal(count(retried, tr('Retry 2.', '重试第 2 次。')), 1);

  // (3) A failed row is red and gives its cause in one line from the error-text owner.
  const failed = job([
    step(10, 'baseline_table', 1, 2, 'started'), step(13, 'baseline_table', 1, 2, 'complete'),
    step(20, 'outcome_model', 2, 2, 'started'),
  ], { jobCode: 'execute_plan', status: 'error', failureCode: 'research_pipeline_execution_failed' });
  const failedHtml = failed.owner.render(failed.activity);
  const cause = runFailureText('research_pipeline_execution_failed');
  assert.equal(count(failedHtml, `<span class="gpi-activity-step-cause" title="${esc(cause)}">${esc(cause)}</span>`), 1);
  assert.match(failedHtml, /<li class="error kind-pipeline">/);
  assert.equal(operations(failedHtml).find(row => row.status === 'error').cause, cause, 'the detail panel receives the same cause');
  // A finished job's repaired error carries no cause: no failure code names a stop.
  const repaired = job([
    step(10, 'outcome_model', 1, 1, 'started'),
    { type: 'progress', step: 'runner', step_id: 'outcome_model', action: 'step_error', status: 'error', timestamp: at(12), run_id: RUN },
    step(20, 'outcome_model', 1, 1, 'complete'),
  ], { jobCode: 'execute_plan' });
  assert.doesNotMatch(repaired.owner.render(repaired.activity), /gpi-activity-step-cause/);
  // The cause rides only on a job's stop code, whatever the error-text owner would say.
  const generic = modules.require('activity').create({ tr, esc, iconHtml, resourceName: () => '', resourceKey: () => '',
    resourceButton: () => '', runFailureText: () => 'generic cause' });
  const erroredRow = failureCode => ({ id: 'turn-3', role: 'activity', status: 'error', startedAt: 1, endedAt: 2000, failureCode,
    steps: [{ id: 'p-1', kind: 'pipeline', step: 'runner', status: 'error', startedAt: 1, endedAt: 1000, label: 'Runner stopped' }] });
  assert.match(generic.render(erroredRow('research_pipeline_execution_failed')), /gpi-activity-step-cause/);
  assert.doesNotMatch(generic.render(erroredRow('')), /gpi-activity-step-cause/);

  // (4) A kind icon only when the row's kind is known.
  const icons = activityOwner().render({ id: 'turn-2', role: 'activity', status: 'complete', startedAt: 0, endedAt: 3000, steps: [
    { id: 'tool-a', kind: 'tool', toolName: 'easyicu_inspect_plan', status: 'complete', startedAt: 0, endedAt: 1000 },
    { id: 'tool-b', kind: 'tool', toolName: 'opaque_future_tool', status: 'complete', startedAt: 1000, endedAt: 2000 },
  ] });
  const iconRows = operations(icons);
  assert.equal(iconRows.find(row => row.id === 'tool-a').icon, 'list');
  assert.equal(iconRows.find(row => row.id === 'tool-b').icon, '');
  assert.doesNotMatch(icons, /<i>spark<\/i>/);

  // (5) Running: the list stays open in a scrolling window, with 展开全部 / 收起 kept across renders.
  const live = job([1, 2, 3, 4, 5, 6].flatMap(n => [step(n * 10, `step_${n}`, n, 7, 'started'), step(n * 10 + 2, `step_${n}`, n, 7, 'complete')])
    .concat([step(80, 'step_7', 7, 7, 'started')]), { status: 'running', jobCode: 'execute_plan' });
  let liveHtml = live.owner.render(live.activity);
  assert.match(liveHtml, /<div class="gpi-activity-body gpi-trace-live is-clipped" data-gpi-trace-key="j1"><div class="gpi-trace-window"><ol>/);
  assert.match(liveHtml, new RegExp(`aria-expanded="false"><i>external</i><span>${tr('Expand all', '展开全部')}</span>`));
  assert.equal(count(liveHtml, 'gpi-running-spinner'), 2, 'the running row and the running header');
  modules.require('activity').traceExpanded('j1', true);
  liveHtml = live.owner.render(live.activity);
  assert.match(liveHtml, /gpi-trace-live is-clipped is-expanded/);
  assert.match(liveHtml, new RegExp(`aria-expanded="true"><i>external</i><span>${tr('Collapse', '收起')}</span>`));
  modules.require('activity').traceExpanded('j1', false);

  // Durations past a minute read 12 min 3 s / 12 分 3 秒.
  assert.equal(live.owner.durationText(1000, 724000), tr('12 min 3 s', '12 分 3 秒'));

  // --- the top bar while a job runs -----------------------------------
  const header = modules.require('header').render({ tr, esc, icon: iconHtml, projectTitle: 'P', sessionTitle: 'S', busy: true,
    runningSince: 1000, runningElapsed: tr('12 min 3 s', '12 分 3 秒'), layout: {} });
  assert.match(header, new RegExp(`<span class="gpi-live is-running"><span role="status">${tr('Running', '运行中')}</span><span aria-hidden="true"> · </span><time data-gpi-live-elapsed="1000">${tr('12 min 3 s', '12 分 3 秒')}</time></span>`));
  const idleHeader = modules.require('header').render({ tr, esc, icon: iconHtml, projectTitle: 'P', sessionTitle: 'S', busy: false, layout: {} });
  assert.doesNotMatch(idleHeader, /is-running/);
}
language = 'zh';
window.EU_LANG = 'zh';

// --- the step detail: status chip, duration, receipt facts, no-file sentence
const stepDetail = modules.require('stepDetail');
const helpers = { codeRows: text => esc(text), statusLabel: value => ({ complete: '已完成', error: '需要处理', running: '运行中' }[value] || value) };
const detail = stepDetail.validDetail({
  step_id: 'outcome_model', state: 'complete', execution: { mode: 'standard', executor: 'grouped_table_one' },
  code: { display_name: 'analysis.py', language: 'python', text: 'print(1)\n', line_count: 1, truncated: false },
  outputs: [], withheld_output_count: 0, more_output_count: 0, headline: {},
  inputs: { source_label: 'eICU-CRD demo', source_database: 'eicu', stay_count: 1841, variable_count: 4, step_index: 3, step_total: 8 },
  timing: { started_at: at(10), ended_at: at(73) },
});
for (language of ['zh', 'en']) {
  window.EU_LANG = language;
  const operation = { title: 'Fit', icon: '', status: 'complete', duration: '', stepIndex: 3, stepTotal: 8, execution: 'standard', executorName: '' };
  const rendered = stepDetail.render({ operation, stepDetail: detail }, helpers);
  assert.match(rendered, /<b class="gpi-status-chip status-complete">/);
  assert.ok(rendered.includes(tr('1 min 3 s', '1 分 3 秒')), 'duration from the step’s own audit times');
  assert.ok(rendered.includes(`<strong>${tr('Input', '输入')}</strong>`) && rendered.includes(`<strong>${tr('Output', '输出')}</strong>`));
  for (const fact of ['eICU-CRD demo (eicu)', '1841', '4', tr('Step 3 of 8', '第 3/8 步')]) assert.ok(rendered.includes(`<dd>${fact}</dd>`), fact);
  const factsAt = rendered.indexOf('<dl class="gpi-operation-facts gpi-step-facts">');
  assert.ok(factsAt >= 0 && factsAt < rendered.indexOf('gpi-source-code'), 'facts above the code');
  assert.ok(rendered.includes(tr('This step updated the run record and produced no file.', '这一步没有生成文件，只更新了运行记录。')));
  // A failed step: the red box carries the cause.
  const failedDetail = { ...detail, state: 'failed' };
  const failedHtml = stepDetail.render({ operation: { ...operation, status: 'error', cause: '分析步骤失败。' }, stepDetail: failedDetail }, helpers);
  assert.match(failedHtml, /<div class="gpi-step-error" role="alert">.*分析步骤失败。/s);
  assert.doesNotMatch(failedHtml, /produced no file|没有生成文件/);
}
// A detail that carries no receipt fact shows none.
const bare = stepDetail.validDetail({ step_id: 'x', state: 'complete', outputs: [] });
assert.doesNotMatch(stepDetail.render({ operation: { title: 'X', status: 'complete', stepIndex: 0, stepTotal: 0 }, stepDetail: bare }, helpers), /gpi-step-facts/);
language = 'zh';
window.EU_LANG = 'zh';

// --- run files: a 文件 | 内容 table with no run id or host path ----------
const resources = modules.require('resources').create({ esc });
const REF = 'rr_' + 'a'.repeat(32);
const savedRun = { run_id: RUN, run_ref: REF, study_id: 's', run_status: 'failed', gate_reason: 'research_pipeline_execution_failed', updated_at: '2026-01-01T00:10:00Z', updated_at_epoch: 600 };
const runFilesApi = {
  listStudyContexts: async () => ({ contexts: [{ id: 's', title: 'S' }] }),
  loadIdeaAgentProjects: async () => ({ projects: [] }),
  loadGuidedDrafts: async () => ({ drafts: [] }),
  loadAgentRunHistory: async () => ({ ok: true, runs: [savedRun], count: 1 }),
  loadAgentRunReview: async () => ({ ok: true, run_id: RUN, run_ref: REF, study_id: 's', engine: 'easyicu.research_agent.pipeline',
    readiness: { signable: false, non_human_failures: [] }, signed: false,
    artifacts: [{ name: 'result_tables.json', sha256: 'b'.repeat(64) }, { name: 'figure_gallery.json', sha256: 'c'.repeat(64) }] }),
};
(async () => {
  const runFiles = modules.require('runFiles').create({ api: () => runFilesApi, changed() {}, resourceButton: resources.button,
    context: () => ({ projectId: 'project-1', studyId: 's', sessionId: 'session-1', runId: '', busy: false }) });
  await runFiles.sync();
  await runFiles.open(REF);
  const filesHtml = runFiles.render({ savedRuns: [savedRun] });
  assert.match(filesHtml, /<table class="gpi-run-file-list"><thead><tr><th scope="col">文件<\/th><th scope="col">内容<\/th>/);
  assert.equal(count(filesHtml, '<tr class="gpi-run-file">'), 2);
  assert.ok(!visible(filesHtml).includes(RUN), 'the run id stays in the run record');
  assert.doesNotMatch(filesHtml, /\/Users\/|\/Volumes\/|project_dir/);
  // One place per run, even when two roots list it.
  const projected = modules.require('runFiles').projectTimeline([], [savedRun, { ...savedRun, run_ref: 'rr_' + 'd'.repeat(32) }]);
  assert.equal(projected.flatMap(row => row.savedRuns).length, 1);

  // --- a stopped run: its cause once, no button, no run id ---
  // The action that works stays on the workflow confirmation card, the one
  // owner that offers every choice and records the answer.
  const confirmation = { code: 'failed_pipeline_execution_retry_available', approve: '从失败步骤重试分析' };
  const stop = modules.require('runStop').create({ tr, runFailureText });
  const older = { ...savedRun, run_id: 'run_older', run_ref: 'rr_' + 'e'.repeat(32), updated_at_epoch: 100, gate_reason: 'data_foundation_blocked', run_status: 'blocked' };
  const timeline = [{ role: 'saved_run', savedRuns: [older] }, { role: 'saved_run', savedRuns: [savedRun] }];
  const newest = stop.render(timeline[1]);
  assert.equal(count(newest, 'class="gpi-run-stop"'), 1);
  assert.ok(newest.includes(`<p>${esc(runFailureText('research_pipeline_execution_failed'))}</p>`));
  assert.doesNotMatch(newest, /<button|data-gpi-confirm-action/);
  assert.ok(!newest.includes(RUN));
  const olderHtml = stop.render(timeline[0]);
  assert.ok(olderHtml.includes(esc(runFailureText('data_foundation_blocked'))) && !olderHtml.includes('<button'));
  // A review waiting on a blocked gate is not a stop.
  assert.equal(stop.render({ savedRuns: [{ ...savedRun, run_status: 'blocked', gate_reason: 'human_plan_review_required' }] }), '');
  // A run whose record names no cause still says it stopped.
  assert.ok(stop.render({ savedRuns: [{ ...savedRun, gate_reason: '', run_status: 'failed' }] }).includes('<p>这次运行在完成前停止。</p>'));

  // The conversation shows the stop block in place of the old sentence, and
  // the server note above the composer while a job runs.
  const stub = (overrides = {}) => new Proxy(overrides, { get: (target, key) => (key in target ? target[key] : () => '') });
  const conversation = (messages, overrides = {}) => modules.require('sessionView').create({
    state: { session: { session_id: 'session-1', binding: {} }, messages, workflowReceipts: [], busy: false, projectLoading: false,
      draft: '', regenerating: false, planStarting: null },
    modules, dataConsent: stub({ requiresConfirmation: () => false }), starters: null, ideaSource: null,
    header: stub(), regeneration: null, studyWorkspace: stub({ messageView: row => ({ text: row.text }) }),
    activity: stub({ renderTimeline: (rows, renderRow) => rows.map(row => renderRow(row)).join(''), render: () => '', durationText: () => '5 秒' }),
    runFiles: stub({ timeline: rows => rows, render: () => '', notice: () => '' }),
    resourceOwner: stub(), messageActions: stub({ render: () => ({}) }),
    transcript: stub({ latestTurnCompletedIdeaExploration: () => false }),
    runOutcome: stub({ collection: () => [] }), aside: stub({ layoutOptions: () => ({}) }),
    cohortEligibility: stub(), tr, esc, iconHtml, projectId: () => 'project-1',
    publicAssistantText: text => text, assistantTextHtml: text => esc(text), sessionIsStale: () => false,
    agentMode: () => 'research', accessModeLabel: () => '', projectTitle: () => 'Project',
    navigationSessionTitle: () => 'Session', workflowConfirmationHtml: () => '',
    hostJobs: stub(), followUps: null, effortMenu: null,
    workflowConfirmation: () => confirmation, runFailureText, ...overrides,
  }).sessionPanel();
  const stoppedPanel = conversation([{ id: 'q', role: 'user', text: '问题', complete: true }, { id: 'saved', role: 'saved_run', savedRuns: [savedRun] }]);
  assert.equal(count(stoppedPanel, 'class="gpi-run-stop"'), 1);
  assert.doesNotMatch(stoppedPanel, /这次运行没有完成，文件作为历史保留在下方/);
  assert.ok(!visible(stoppedPanel).includes(RUN));
  const note = 'EasyICU 在服务器上运行，可以切换对话或关闭页面，回来后继续显示进度。';
  assert.doesNotMatch(stoppedPanel, /gpi-compose-server-note/);
  const runningPanel = conversation([{ id: 'q', role: 'user', text: '问题', complete: true },
    { id: 'easyicu-job-j1', role: 'activity', childJobId: 'j1', status: 'running', startedAt: 1000, steps: [] }]);
  assert.ok(runningPanel.includes(`<p class="gpi-compose-server-note">${note}</p><div class="gpi-compose">`));

  // --- a finished run's answer: File | Contents, notes, actions, follow-ups
  const ref = (artifact, digest) => ({ kind: 'research_artifact', run_id: RUN, artifact, sha256: digest.repeat(64) });
  const latestRun = { present: true, run_id: RUN, analysis_results_available: true, analysis_validated: true, numeric_verified: true,
    reportable: false, updated_at: '2026-01-01T00:10:00Z',
    artifact_refs: ['result_tables.json', 'figure_gallery.json', 'evidence_ledger.json', 'literature_evidence.json',
      'scientific_readiness.json', 'agent_plan.json'].map((name, index) => ref(name, 'abcdef'[index])) };
  const workflow = (stage, status) => ({ current_stage: stage, stages: [{ id: 'analysis', status: 'complete' }, { id: stage, status }],
    runs: [{ run_id: RUN }, { run_id: 'run_older' }] });
  const session = { archived_child_jobs: [
    { kind: 'agent-run', run_id: RUN, progress: [
      { step: 'planning', retry_phase: 'rejected', planning_unit: 'structure', current: 1, label: '' },
      { step: 'provider_auto_resume', current: 1, total: 2 },
      { step: 'step', label: 'Skipped step already completed by pre-execution: cohort_accounting.' },
    ] },
    { kind: 'agent-run', run_id: 'run_other', progress: [{ step: 'provider_auto_resume', current: 1, total: 2 }, { step: 'provider_auto_resume', current: 2, total: 2 }] },
  ] };
  const outcome = modules.require('runOutcome').create({ tr, esc, iconHtml, resourceButton: resources.button, api: () => ({}),
    projectId: () => 'project-1', host: () => null, canPreview: () => false, preview: () => null, workflowContext: () => ({}),
    errorText: () => '', recordHostAction: async () => null, onError() {}, session: () => session });
  let card = outcome.render(latestRun, workflow('interpretation', 'blocked'));
  assert.match(card, /<thead><tr><th>文件<\/th><th>内容<\/th><\/tr><\/thead>/);
  assert.ok(card.indexOf('gpi-deliverables') < card.indexOf('gpi-run-notes') && card.indexOf('gpi-run-notes') < card.indexOf('gpi-answer-actions')
    && card.indexOf('gpi-answer-actions') < card.indexOf('gpi-followups'), 'table, notes, actions, then follow-ups');
  const notes = (card.match(/<section class="gpi-run-notes"[^>]*><h4>过程说明<\/h4><p>([^<]*)<\/p>/) || [])[1] || '';
  assert.equal(notes, '模型服务中断后，系统从检查点自动继续了 1 次。 计划有 1 次没有通过校验，已自动修正。 1 个步骤已在准备数据时完成，没有重复运行。 结果仅供分析审阅，未获投稿授权。');
  assert.ok(card.includes('data-gpi-run-answer-copy><i>copy</i><span>复制</span>'));
  assert.ok(card.includes('data-gpi-open-run-record><i>list</i><span>运行记录 2</span>'));
  assert.doesNotMatch(card, />审查</, 'no review action unless the review owner offers one');
  assert.doesNotMatch(card, /thumb/i);
  assert.match(card, /<span>后续问题<\/span>/);
  assert.ok(card.includes('<span aria-hidden="true">↳</span>'));
  card = outcome.render({ ...latestRun, artifact_refs: latestRun.artifact_refs }, workflow('interpretation', 'ready'));
  assert.ok(card.includes('>审查</button>'), 'the workflow offers the review for this run');
  // A run whose receipts state nothing has no notes beyond its scope.
  const quiet = modules.require('runOutcome').create({ tr, esc, iconHtml, resourceButton: resources.button, api: () => ({}),
    projectId: () => 'project-1', host: () => null, canPreview: () => false, preview: () => null, workflowContext: () => ({}),
    errorText: () => '', recordHostAction: async () => null, onError() {}, session: () => ({ archived_child_jobs: [] }) });
  assert.doesNotMatch(quiet.render({ ...latestRun, reportable: true }, workflow('interpretation', 'blocked')), /gpi-run-notes/);

  // --- the right rail: name, time, preview, download; a filter past five files
  window.EU_GUIDED_CONTRACTS = { fmtRunTime: () => '01-01 00:10' };
  const shelf = outcome.renderShelf(latestRun, workflow('interpretation', 'blocked'));
  assert.match(shelf, /本次运行 · <time>01-01 00:10<\/time> · 6 个文件/);
  assert.match(shelf, /data-gpi-results-search/);
  const small = { ...latestRun, artifact_refs: latestRun.artifact_refs.slice(0, 4).concat(ref('evidence_ledger.json', 'c')) };
  assert.doesNotMatch(outcome.renderShelf(small, workflow('interpretation', 'blocked')), /data-gpi-results-search/);

  // 复制 copies the answer as shown and keeps its label.
  const copied = [];
  window.navigator = { clipboard: { writeText: async text => { copied.push(text); } } };
  const answerNode = { textContent: ' 28 天死亡率为 12%。 ' };
  const answerCard = { querySelector: selector => (selector === '.gpi-run-answer-text' ? answerNode : null) };
  const attributes = {};
  const copyButton = { dataset: {}, innerHTML: '', closest: selector => (selector === '[data-gpi-run-outcome]' ? answerCard : null),
    setAttribute: (key, value) => { attributes[key] = value; } };
  const actions = modules.require('messageActions').create({ tr, iconHtml, rows: () => [], host: () => null });
  const handled = actions.handleClick({ target: { closest: selector => (selector === '[data-gpi-run-answer-copy]' ? copyButton : null) } });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(handled, true);
  assert.deepEqual(copied, ['28 天死亡率为 12%。']);
  assert.equal(copyButton.dataset.copied, 'true');
  assert.equal(copyButton.innerHTML, '<i>check</i><span>已复制</span>');

  // --- the opening sentence and run notes have one owner -----------------
  const narration = modules.require('jobNarration');
  assert.equal(narration.opening('easyicu_full_run_upgrade_submitted', tr), '我先按确认的方案准备数据，然后在准备好的数据上重新规划分析。');
  assert.equal(narration.opening('easyicu_review_submitted', tr), '我按批准的计划运行分析，并写出结果说明。');
  assert.equal(narration.opening('', tr), '');
  process.stdout.write('turn layout contracts passed\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
