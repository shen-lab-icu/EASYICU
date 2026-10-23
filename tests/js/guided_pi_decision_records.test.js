/* Every answered gate leaves its record, the rail reads the latest job, and
   the conversation's attributes carry no host folder.

   Counted eICU Q1' attempts 7-10 (2026-10-20/21): a clicked plan card left
   nothing behind, a failed conversation read 「已完成」 in the rail, and the
   run files carried absolute folders in `data-run-files`. These run the real
   owners on synthetic sessions only. */
'use strict';
process.env.TZ = 'UTC';
const assert = require('node:assert/strict');
const path = require('node:path');

require(path.resolve('tests/js/guided_pi_module_harness.cjs'));
const files = process.argv.slice(2);
assert.equal(files.length, 11, 'Run through tools/run_js_contracts.py with all eleven owners');
let language = 'zh';
// `t` and `icon` are the page's shared copy and icon helpers, which the
// run-file reader uses.
global.window = { EU_LANG: language, document: { addEventListener() {} },
  t: (en, zh) => (language === 'zh' ? zh : en), icon: () => '' };
global.document = { addEventListener() {}, getElementById: () => null };
for (const file of files) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
// Owners this page loads elsewhere; only what the conversation asks of them.
modules.declare('nextActions', { project: () => null, bodyText: () => '', render: () => '', renderPast: () => '' });
const esc = window.EU_HTML.esc;
const tr = (en, zh) => (language === 'zh' ? zh : en);
const records = modules.require('decisionRecords');

const AUTHORIZATION = {
  status: 'confirmed', confirmation_mode: 'select_registered_source', confirmed_at: '2026-10-21T05:58:00Z',
  source: { label: 'eICU', reference_release: '2.0' },
};
const RECORDS = [
  { code: 'data_chosen', actor: 'user', at: '2026-10-21T05:58:00Z', source_label: 'eICU', source_release: '2.0' },
  { code: 'plan_confirmed', actor: 'user', at: '2026-10-21T06:03:00Z', run_id: 'run_candidate' },
  { code: 'provider_resumed', actor: 'system', at: '2026-10-21T06:14:00Z', job_id: 'job-prepare', number: 1, limit: 2, cause_code: 'provider_timeout' },
  { code: 'plan_approved', actor: 'user', at: '2026-10-21T13:53:00Z', run_id: 'run_candidate' },
];
const at = text => Date.parse(text);
const LINES = {
  zh: [
    '✓ 你选择了数据：eICU v2.0',
    '✓ 你确认了方案，开始准备数据 · 06:03',
    '系统操作：模型服务恢复后，从检查点自动继续（第 1/2 次）· 06:14',
    '✓ 你批准了计划，开始分析 · 13:53',
  ],
  en: [
    '✓ You chose the data: eICU v2.0',
    '✓ You confirmed the plan and started data preparation · 06:03',
    'System action: the model service recovered; continued automatically from the checkpoint (1/2) · 06:14',
    '✓ You approved the plan and started the analysis · 13:53',
  ],
};

// --- the conversation (session-view with the real record, run-file and consent owners)

const stub = (overrides = {}) => new Proxy(overrides, { get: (target, key) => (key in target ? target[key] : () => '') });
function conversation(session, messages, runFiles) {
  const state = {
    session, messages, workflowReceipts: [], busy: false, projectLoading: false, draft: '',
    regenerating: false, planStarting: null,
  };
  const view = modules.require('sessionView').create({
    state, modules, dataConsent: modules.require('dataConsent'), starters: null, ideaSource: null,
    header: stub(), regeneration: null, studyWorkspace: stub({ messageView: row => ({ text: row.text }) }),
    activity: stub({
      renderTimeline: (rows, renderRow) => rows.map(row => renderRow(row)).join(''),
      render: row => `<div data-activity="${esc(row.id)}"></div>`,
    }),
    runFiles: runFiles || stub({ timeline: rows => rows }),
    resourceOwner: stub(), messageActions: stub({ render: () => ({}) }),
    transcript: stub({ latestTurnCompletedIdeaExploration: () => false }),
    runOutcome: stub({ collection: () => [] }), aside: stub({ layoutOptions: () => ({}) }),
    cohortEligibility: stub(), tr, esc, iconHtml: () => '', projectId: () => 'project-1',
    publicAssistantText: text => text, assistantTextHtml: text => esc(text), sessionIsStale: () => false,
    agentMode: () => 'research', accessModeLabel: () => '', projectTitle: () => 'Project',
    navigationSessionTitle: () => 'Session', workflowConfirmationHtml: () => '',
    hostJobs: stub(), followUps: null, effortMenu: null,
  });
  return view.sessionPanel();
}
const recordLines = html => Array.from(html.matchAll(/<p class="gpi-decision-record[^"]*"[^>]*data-gpi-decision-actor="(\w+)"[^>]*>([^<]*)<\/p>/g))
  .map(match => [match[1], match[2].replace(/&#39;/g, "'")]);
const order = html => Array.from(html.matchAll(/data-(?:activity|gpi-message-id|gpi-decision-record)="([^"]+)"/g)).map(match => match[1]);

// What the page holds after a reload: the session as the host sends it, and
// the timeline rebuilt from the persisted transcript -- no optimistic rows.
const MESSAGES = [
  { id: 'question', role: 'user', text: 'eICU 中脓毒症患者的乳酸与死亡率？', complete: true, timelineAt: at('2026-10-21T05:50:00Z') },
  { id: 'plan-ready', role: 'assistant', text: '候选计划已生成。', complete: true, timelineAt: at('2026-10-21T06:00:00Z') },
  { id: 'job-prepare', role: 'activity', status: 'complete', steps: [], startedAt: at('2026-10-21T06:03:00Z'), timelineAt: at('2026-10-21T06:03:00Z') },
  { id: 'prepared', role: 'assistant', text: '数据已准备。', complete: true, timelineAt: at('2026-10-21T06:40:00Z') },
  { id: 'job-analysis', role: 'activity', status: 'complete', steps: [], startedAt: at('2026-10-21T13:53:00Z'), timelineAt: at('2026-10-21T13:53:00Z') },
];

for (language of ['zh', 'en']) {
  window.EU_LANG = language;
  const session = { session_id: 'session-1', binding: {}, data_source_authorization: AUTHORIZATION, decision_records: RECORDS };
  const html = conversation(session, MESSAGES.map(row => ({ ...row })));
  assert.deepEqual(recordLines(html), [
    ['user', LINES[language][0]], ['user', LINES[language][1]], ['system', LINES[language][2]], ['user', LINES[language][3]],
  ]);
  // Each line stands where its decision was made: above the work it started.
  assert.deepEqual(order(html), [
    'question', 'data_chosen', 'plan-ready', 'plan_confirmed', 'job-prepare', 'provider_resumed', 'prepared',
    'plan_approved', 'job-analysis',
  ]);
  // A reload rebuilds the same lines from the same session.
  const reloaded = conversation(JSON.parse(JSON.stringify(session)), MESSAGES.map(row => ({ ...row })));
  assert.deepEqual(recordLines(reloaded), recordLines(html));
  // No record without its owner's record: an empty session shows none.
  assert.deepEqual(recordLines(conversation({ ...session, decision_records: [] }, MESSAGES.map(row => ({ ...row })))), []);
}
language = 'zh';
window.EU_LANG = 'zh';

// A system continuation is its own actor: never a check mark, never "you",
// and a record claiming otherwise is dropped rather than shown as someone else's.
const system = records.rows({ decision_records: [RECORDS[2]] });
assert.equal(system.length, 1);
const systemHtml = records.render(system[0], { tr, esc });
assert.match(systemHtml, /class="gpi-decision-record is-system"/);
assert.match(systemHtml, /data-gpi-decision-actor="system"/);
assert.doesNotMatch(systemHtml, /✓|你|You /);
assert.deepEqual(records.rows({ decision_records: [
  { ...RECORDS[2], actor: 'user' },
  { ...RECORDS[1], actor: 'system' },
  { ...RECORDS[1], at: 'not a time' },
  { code: 'invented', actor: 'user', at: RECORDS[1].at },
] }), []);
assert.equal(records.clock(at('2026-10-21T06:03:59Z')), '06:03');

// A host-selected first-day landmark is the Agent's decision and explains
// the surviving population in both languages, including after a reload.
const timing = { code: 'post_baseline_landmark_selected', actor: 'agent', at: '2026-10-21T05:59:00Z' };
for (language of ['zh', 'en']) {
  window.EU_LANG = language;
  const session = { decision_records: [timing] };
  const html = conversation(JSON.parse(JSON.stringify(session)), MESSAGES.map(row => ({ ...row })));
  assert.deepEqual(recordLines(html), [['agent', language === 'zh'
    ? 'Agent 决策：暴露在入 ICU 后第一天内测量，因此分析从第 24 小时开始，仅纳入该时点仍存活的患者 · 05:59'
    : 'Agent decision: the exposure is measured during the first ICU day, so the analysis starts at 24 hours among patients alive then · 05:59']]);
  assert.deepEqual(order(html).slice(0, 3), ['question', 'post_baseline_landmark_selected', 'plan-ready']);
}
assert.deepEqual(records.rows({ decision_records: [
  { ...timing, actor: 'user' }, { ...timing, actor: 'system' }, { ...timing, at: 'invalid' },
] }), []);
language = 'zh';
window.EU_LANG = 'zh';

// A data source the researcher typed already shows its past choice under
// that reply, so the record line is not repeated.
const typed = conversation(
  { session_id: 'session-1', binding: {}, data_source_authorization: AUTHORIZATION, decision_records: RECORDS.slice(0, 1) },
  [{ id: 'typed', role: 'user', text: 'eICU v2.0', complete: true, timelineAt: at('2026-10-21T05:57:00Z') },
    { id: 'answer', role: 'assistant', text: '好的。', complete: true, timelineAt: at('2026-10-21T05:58:30Z') }],
);
assert.deepEqual(recordLines(typed), []);
assert.match(typed, /gpi-past-decision/);

// --- the run files: only the host's opaque coordinate reaches an attribute

(async () => {
  const HOST_PATHS = /\/home\/|\/Users\/|\/Volumes\//;
  const REF = 'rr_' + '1'.repeat(32);
  // A row shaped as an older host sent it, folder and all: the page must
  // still address the run by its ref only.
  const run = { run_id: 'run_candidate', study_id: 'study-1', run_status: 'completed', updated_at_epoch: 1,
    run_ref: REF, project_dir: '/Users/researcher/EasyICU/projects/study-1/run_candidate' };
  const requests = [];
  const api = {
    listStudyContexts: async () => ({ contexts: [{ id: 'study-1', title: 'Study' }] }),
    loadIdeaAgentProjects: async () => ({ projects: [{ study_id: 'study-1', project_dir: '/Volumes/disk/seed/study-1',
      runs: [{ run_id: 'seed-only', label: 'seed-only', project_dir: '/home/researcher/seed/study-1/run_seed-only' }] }] }),
    loadGuidedDrafts: async () => ({ drafts: [] }),
    loadAgentRunHistory: async request => { requests.push(request); return { ok: true, runs: [run], count: 1 }; },
    loadAgentRunReview: async coordinate => { requests.push(['review', coordinate]); return {
      ok: true, run_ref: coordinate.run_ref, run_id: run.run_id, study_id: 'study-1', signed: false,
      readiness: { signable: false }, artifacts: [{ name: 'result_tables.json', sha256: 'a'.repeat(64), relative_path: 'result_tables.json' }] }; },
  };
  const runFiles = modules.require('runFiles').create({
    api: () => api, changed() {}, resourceButton: () => '',
    context: () => ({ projectId: 'project-1', studyId: 'study-1', sessionId: 'session-1', runId: '', busy: false }),
  });
  await runFiles.sync();
  await runFiles.open(REF);
  const session = { session_id: 'session-1', binding: {}, data_source_authorization: AUTHORIZATION, decision_records: RECORDS };
  const html = conversation(session, [
    ...MESSAGES.map(row => ({ ...row })),
    { id: 'results', role: 'assistant', text: '结果如下。', complete: true, timelineAt: at('2026-10-21T15:00:00Z'),
      resources: [{ kind: 'research_artifact', run_id: 'run_candidate', artifact: 'result_tables.json' }] },
  ], runFiles);
  assert.match(html, new RegExp(`data-run-files="${REF}"`));
  const attributes = Array.from(html.matchAll(/\s([\w:-]+)="([^"]*)"/g));
  assert.ok(attributes.length > 10);
  const leaked = attributes.filter(([, , value]) => HOST_PATHS.test(value)).map(([, name, value]) => `${name}=${value}`);
  assert.deepEqual(leaked, []);
  assert.ok(requests.filter(request => !Array.isArray(request)).every(request => request.path_free === true));
  assert.deepEqual(requests.find(request => Array.isArray(request))[1], { run_ref: REF });

  // --- the rail: the conversation's latest job decides its state

  const workspace = modules.require('studyWorkspace').create({ tr, esc, iconHtml: () => '' });
  const table = [
    [{ conversation_status: 'running' }, '进行中'],
    [{ conversation_status: 'needs_user', last_turn_status: 'done' }, '待确认'],
    // attempt 9: the reply finished, the research job that followed failed
    [{ conversation_status: 'stopped', last_turn_status: 'done' }, '已停止'],
    [{ conversation_status: 'done', last_turn_status: 'done' }, '已完成'],
    [{ conversation_status: null, history_turn_count: 0 }, '未开始'],
    // a message turn this page started is running whatever the index said
    [{ conversation_status: 'stopped', active_message_job_id: 'job' }, '进行中'],
    // rows from before the host reported a state read their last reply
    [{ last_turn_status: 'failed' }, '已停止'],
    [{ last_turn_status: 'done' }, '已完成'],
  ];
  assert.deepEqual(table.map(([row]) => workspace.sessionStatusLabel(row)), table.map(([, label]) => label));
  language = 'en';
  assert.deepEqual(
    ['running', 'needs_user', 'stopped', 'done'].map(state => workspace.sessionStatusLabel({ conversation_status: state })),
    ['Running', 'Needs you', 'Stopped', 'Done'],
  );
  language = 'zh';

  // --- a stale gate says so instead of offering its buttons

  const gate = stale => modules.require('confirmation').create({
    tr, esc, iconHtml: () => '', resourceButton: () => '', sessionIsStale: () => stale, busy: () => false,
    session: () => ({ binding: { run_id: '' }, archived_child_jobs: [] }),
    workflow: () => ({ next_action_code: 'provider_ready_to_generate_plan' }),
  }).workflowConfirmationHtml();
  const live = gate(false);
  assert.match(live, /data-gpi-confirm-action/);
  const stale = gate(true);
  assert.match(stale, /data-gpi-stale-gate/);
  assert.match(stale, /已不能在这里作答/);
  assert.doesNotMatch(stale, /<button|data-gpi-confirm-/);

  // --- a host action's replay renders with the real activity owner

  modules.declare('replay', { lifecycleTurns: session => session.conversation_replay.turns });
  const activity = modules.require('activity').create({ tr, esc, iconHtml: () => '', resourceName: () => '',
    resourceKey: () => '', resourceButton: () => '', runFailureText: () => '' });
  const transcript = modules.require('transcript').create({
    tr, activity, timeMs: value => Date.parse(value) || 0, resourceKey: () => '', modelErrorText: () => '',
    activityHasCompletedAction: () => false, workflowActionCode: () => '', runFailureText: () => '',
    upsertActivityStep(target, step) {
      const found = target.steps.find(item => item.id === step.id);
      if (found) Object.assign(found, step); else target.steps.push(step);
    },
  });
  const replayed = transcript.transcriptMessages({
    transcript: [],
    conversation_replay: { turns: [{ job_id: 'host_1', kind: 'host_action', action_code: 'prepare_analysis_data',
      action_key: 'plan:' + 'a'.repeat(16) + ':run_candidate:plan_execution_upgrade_required',
      child_job_id: 'job-prepare', status: 'done', started_at: '2026-10-21T06:03:00Z', ended_at: '2026-10-21T06:40:00Z' }] },
    archived_child_jobs: [{ job_id: 'job-prepare', kind: 'agent-run', status: 'done', human_review_pending: true,
      created_at_epoch: at('2026-10-21T06:03:00Z') / 1000, finished_at_epoch: at('2026-10-21T06:40:00Z') / 1000,
      progress: [{ type: 'progress', step: 'planning', status: 'running', timestamp: '2026-10-21T06:05:00Z' }] }],
  });
  assert.ok(replayed.some(row => row.role === 'activity' && row.childJobId === 'job-prepare'));
  assert.ok(replayed.some(row => row.role === 'assistant' && row.hostActionCode === 'prepare_analysis_data'));
  // The replayed action keys the job's opening sentence; a finished job names no stop.
  const replayedJob = replayed.find(row => row.role === 'activity' && row.childJobId === 'job-prepare');
  assert.equal(replayedJob.jobCode, 'prepare_analysis_data');
  assert.equal(replayedJob.failureCode, '');
  const failedReplay = transcript.transcriptMessages({
    transcript: [],
    conversation_replay: { turns: [{ job_id: 'host_2', kind: 'host_action', action_code: 'execute_plan',
      action_key: 'plan:' + 'b'.repeat(16) + ':run_candidate:plan_execution',
      child_job_id: 'job-execute', status: 'failed', started_at: '2026-10-21T07:00:00Z', ended_at: '2026-10-21T07:20:00Z' }] },
    archived_child_jobs: [{ job_id: 'job-execute', kind: 'agent-run', status: 'failed', error_code: 'research_pipeline_execution_failed',
      created_at_epoch: at('2026-10-21T07:00:00Z') / 1000, finished_at_epoch: at('2026-10-21T07:20:00Z') / 1000, progress: [] }],
  }).find(row => row.role === 'activity' && row.childJobId === 'job-execute');
  assert.equal(failedReplay && failedReplay.failureCode, 'research_pipeline_execution_failed', 'a failed job gives its failed row a cause');
  const limited = transcript.transcriptMessages({
    transcript: [],
    conversation_replay: { turns: [{ job_id: 'host-render', kind: 'host_action', action_code: 'execute_plan',
      child_job_id: 'job-render', status: 'done', started_at: '2026-10-21T07:00:00Z', ended_at: '2026-10-21T07:20:00Z' }] },
    archived_child_jobs: [{ job_id: 'job-render', kind: 'agent-run', status: 'done',
      analysis_results_available: true, analysis_validated: true, numeric_verified: true,
      limitations: [{ step_id: 'presentation', message: '可选图表 presentation 未完成，不能用于支持结论。' }], progress: [] }],
  }).find(row => row.role === 'assistant' && row.hostActionCode === 'execute_plan');
  assert.match(limited.text, /可选图表 presentation 未完成/);
  assert.doesNotMatch(limited.text, /已在本轮完成.*图件重绘/);
  const planNote = '已省略可选绘图步骤 presentation：没有标准绘图执行器，且没有必需输出或后续步骤依赖它。';
  assert.ok(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {
    research_question: 'Synthetic distribution', steps: [], rationale: planNote,
  }).includes(planNote));
  process.stdout.write('Decision records, rail state, path-free run files, stale gate and host-action replay passed.\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
