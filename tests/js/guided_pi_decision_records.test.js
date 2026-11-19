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
const { test } = require('node:test');

require(path.resolve('tests/js/guided_pi_module_harness.cjs'));
const files = process.argv.slice(2);
assert.equal(files.length, 13, 'Run through tools/run_js_contracts.py with all thirteen owners');
let language = 'zh';
// `t` and `icon` are the page's shared copy and icon helpers, which the
// run-file reader uses.
global.window = { EU_LANG: language, document: { addEventListener() {} },
  t: (en, zh) => (language === 'zh' ? zh : en), icon: () => '' };
global.document = { addEventListener() {}, getElementById: () => null };
for (const file of files) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
for (const name of ['target-trial-copy', 'target-trial', 'question-requirements', 'run-stop', 'plan-actions']) {
  require(path.resolve(`src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`));
}
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
function conversation(session, messages, runFiles, workflow) {
  const state = {
    session, sessions: [], messages, workflow, workflowReceipts: [], busy: false, projectLoading: false, draft: '',
    regenerating: false, planStarting: null,
  };
  const confirmation = modules.require('confirmation').create({
    tr, esc, iconHtml: () => '', resourceButton: () => '', sessionIsStale: () => false, busy: () => false,
    session: () => session, workflow: () => workflow,
  });
  const view = modules.require('sessionView').create({
    state, modules, dataConsent: modules.require('dataConsent'), starters: null, ideaSource: null,
    header: stub(), regeneration: null, studyWorkspace: stub({ messageView: row => ({ text: row.text }) }),
    activity: stub({
      renderTimeline: (rows, renderRow) => rows.map(row => renderRow(row)).join(''),
      render: row => `<div data-activity="${esc(row.id)}"></div>`,
    }),
    runFiles: runFiles || stub({ timeline: rows => rows }),
    resourceOwner: stub(), messageActions: stub({ render: () => ({}) }),
    transcript: modules.require('transcript'),
    runOutcome: stub({ collection: () => [] }), aside: stub({ layoutOptions: () => ({}) }),
    cohortEligibility: stub(), tr, esc, iconHtml: () => '', projectId: () => 'project-1',
    publicAssistantText: text => text, assistantTextHtml: text => esc(text), sessionIsStale: () => false,
    agentMode: () => 'research', accessModeLabel: () => '', projectTitle: () => 'Project',
    navigationSessionTitle: () => 'Session',
    workflowConfirmation: confirmation.workflowConfirmation, workflowConfirmationHtml: confirmation.workflowConfirmationHtml,
    hostJobs: stub(), followUps: null, effortMenu: null,
  });
  return view.sessionPanel();
}
const recordLines = html => Array.from(html.matchAll(/<p class="gpi-decision-record[^"]*"[^>]*data-gpi-decision-actor="(\w+)"[^>]*>([^<]*)<\/p>/g))
  .map(match => [match[1], match[2].replace(/&#39;/g, "'")]);
const order = html => Array.from(html.matchAll(/data-(?:activity|gpi-message-id|gpi-decision-record)="([^"]+)"/g)).map(match => match[1]);

// A mixed literature and planning turn can end with a real study decision.
const pendingSession = { session_id: 'session-pending', binding: {}, data_source_authorization: AUTHORIZATION };
const researchTurn = [
  { id: 'question', role: 'user', text: 'Compare two fluid strategies for renal recovery.' },
  { id: 'review', role: 'activity', status: 'complete', steps: [
    { kind: 'tool', toolName: 'easyicu_search_literature', status: 'complete' },
  ] },
];
const trial = {
  state: 'approvable', approvable: true, compile_sha256: 'd'.repeat(64), confirmation_lines: 1,
  confirmations: [{ text: 'Assign the strategy at the start of the first shift.' }],
};
const pendingTrial = { next_action_code: 'target_trial_review', target_trial_card: trial };
function pendingDecisionHtml(workflow, cardLanguage = 'en') {
  const previousLanguage = language;
  language = window.EU_LANG = cardLanguage;
  try {
    return conversation(pendingSession, researchTurn, null, workflow);
  } finally {
    language = window.EU_LANG = previousLanguage;
  }
}
test('pending target-trial assumptions and explicit approval appear in the main log', () => {
  const trialHtml = pendingDecisionHtml(pendingTrial);
  assert.match(trialHtml, /data-gpi-log[^]*Assign the strategy at the start of the first shift/);
  assert.match(trialHtml, /data-gpi-confirm-action disabled[^>]*>Approve the target trial<\/button>/);
  assert.equal(trial.state, 'approvable');
});

test('a pending trial statement asks for free text in the main log', () => {
  const clarification = pendingDecisionHtml({
    next_action_code: 'target_trial_statement_needed',
  });
  assert.match(clarification, /data-gpi-log[^]*State the target trial this study emulates/);
  assert.match(clarification, /data-gpi-confirm-edit[^>]*>State it in the conversation<\/button>/);
});

test('a proposed exposure landmark shows its assumptions and explicit choice in the main log', () => {
  const proposal = { card_sha256: 'f'.repeat(64), exposure: 'early fluid', landmark_hours: 18, horizon_days: 42 };
  const workflow = {
    next_action_code: 'plan_scientific_changes_required', plan_decision: proposal,
    plan_review_summary: { authorization_questions: [{
      code: 'SURVIVAL_LANDMARK_OWNER_NOT_SEALED', requires_user_authorization: true, decision_context: proposal,
    }] },
  };
  const before = JSON.stringify(workflow);
  const exposure = pendingDecisionHtml(workflow);
  assert.match(exposure, /data-gpi-log[^]*18-hour landmark · 42-day survival/);
  assert.match(exposure, /Deaths before hour 18 are not part of the comparison/);
  assert.equal((exposure.match(/data-gpi-plan-decision-option=/g) || []).length, 1);
  assert.match(exposure, /data-gpi-plan-decision-option="fixed_landmark_survival"[^>]*><strong>Use the 18-hour landmark/);
  assert.doesNotMatch(exposure, /data-gpi-confirm-action/);
  assert.equal(JSON.stringify(workflow), before);
});

test('a pending exposure choice asks for an answer in the main log', () => {
  const exposure = pendingDecisionHtml({
    next_action_code: 'question_requirement_not_covered',
    plan_review_summary: { run_id: 'candidate-pending', question_requirements: {
      status: 'shown', items: [{
        quote: 'Compare both fluid strategies against the same outcome',
        disposition: 'not_covered',
        exposure_choice: { primary_exposure: 'early fluid', additional_exposures: ['late fluid'] },
      }],
    } },
  });
  assert.match(exposure, /data-gpi-log[^]*Reply in the conversation before approval/);
  assert.match(exposure, /data-gpi-confirm-edit[^>]*>Answer this question<\/button>/);
  assert.doesNotMatch(exposure, /data-gpi-confirm-action/);
});

test('a completed literature turn ignores a stale plan summary from a failed run', () => {
  const run = { run_id: 'earlier-candidate', run_ref: 'earlier', authoritative: true, run_status: 'failed',
    gate_reason_code: 'research_pipeline_progressive_compile_failed' };
  for (const code of ['failed_pipeline_requires_fresh_plan', 'failed_pipeline_execution_retry_available']) {
    const html = pendingDecisionHtml({
      next_action_code: code, plan_review_summary: { run_id: run.run_id }, runs: [run],
    });
    assert.match(html, /Compare two fluid strategies for renal recovery/);
    assert.doesNotMatch(html, /data-gpi-run-stop|data-gpi-confirm-action/);
  }
});

test('an open study decision is inline when the transcript has not arrived', () => {
  const session = { ...pendingSession, binding: {} };
  const html = conversation(session, [], null, { next_action_code: 'target_trial_statement_needed' });
  const label = tr('State it in the conversation', '在对话里说明');
  assert.match(html, /data-gpi-log[^]*data-gpi-confirm-edit/);
  assert.ok(html.includes(label));
  assert.match(html, /data-gpi-input/);
  assert.doesNotMatch(html, /data-gpi-confirm-action/);
});

test('reading target-trial assumptions leaves approval to the explicit action', () => {
  const owner = modules.require('targetTrial');
  const session = { ...pendingSession, binding: { study_context_id: 'study-1', study_revision: 4 } };
  assert.equal(owner.approvalRequest(pendingTrial, session, 'project-1'), null);
  owner.noteTick(trial.compile_sha256, 0, true);
  try {
    const html = pendingDecisionHtml(pendingTrial);
    assert.match(html, /data-gpi-log[^]*data-gpi-confirm-action>Approve the target trial<\/button>/);
    assert.equal(trial.state, 'approvable');
    assert.deepEqual(owner.approvalRequest(pendingTrial, session, 'project-1'), {
      project_id: 'project-1', study_context_id: 'study-1', expected_revision: 4,
      compile_sha256: trial.compile_sha256, n_lines_confirmed: 1,
    });
  } finally {
    owner.noteTick(trial.compile_sha256, 0, false);
  }
});

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
    [{ conversation_status: 'stopped', last_turn_status: 'done' }, '需要处理'],
    [{ conversation_status: 'done', last_turn_status: 'done' }, '任务已结束'],
    [{ conversation_status: null, history_turn_count: 0 }, '未开始'],
    // a message turn this page started is running whatever the index said
    [{ conversation_status: 'stopped', active_message_job_id: 'job' }, '进行中'],
    // rows from before the host reported a state read their last reply
    [{ last_turn_status: 'failed' }, '回复失败'],
    [{ last_turn_status: 'done' }, '已回复'],
  ];
  assert.deepEqual(table.map(([row]) => workspace.sessionStatusLabel(row)), table.map(([, label]) => label));
  language = 'en';
  assert.deepEqual(
    ['running', 'needs_user', 'stopped', 'done'].map(state => workspace.sessionStatusLabel({ conversation_status: state })),
    ['Running', 'Needs you', 'Needs attention', 'Task ended'],
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

test('a first-result delivery conflict prompts for a revision inside the main log', () => {
  const html = pendingDecisionHtml({
    next_action_code: 'first_result_delivery_requires_article',
    plan_review_summary: { run_id: 'shift-recovery-plan', deferred_input_scope: { verified: false } },
  });
  const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
  assert.match(log, /data-gpi-confirm-edit[^>]*>Revise in the conversation<\/button>/);
  assert.match(log, /separate article run/);
  assert.doesNotMatch(log, /data-gpi-confirm-action/);
});

test('a current plan whose requirement record is unreadable can be regenerated from the main log', () => {
  const html = pendingDecisionHtml({
    next_action_code: 'question_requirements_unreadable',
    plan_review_summary: { run_id: 'interrupted-shift-plan' },
  });
  const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
  assert.match(log, /data-gpi-confirm-action[^>]*>Generate fresh plan<\/button>/);
});

test('a pending cohort-source decision from the plan stage is inline after a literature turn', () => {
  const code = 'plan_icd_source_unverified';
  const html = pendingDecisionHtml({
    next_action_code: code,
    stages: [{ id: 'plan', status: 'review_required', reason_code: code }],
    plan_review_summary: { run_id: 'cohort-source-plan', finding_codes: ['HOST_ICD_SOURCE_UNVERIFIED'] },
  });
  const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
  assert.match(log, /data-gpi-confirm-edit[^>]*>Change cohort<\/button>/);
  assert.doesNotMatch(log, /data-gpi-confirm-action/);
});

test('a delivery clarification is actionable inside the main conversation even without a plan summary', () => {
  const code = 'host_delivery_request_ambiguous';
  for (const messages of [[], researchTurn]) {
    const previousLanguage = language;
    language = window.EU_LANG = 'en';
    try {
      const html = conversation(pendingSession, messages, null, {
        next_action_code: code,
        stages: [{ id: 'plan', status: 'review_required', reason_code: code }],
      });
      const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
      assert.match(log, /data-gpi-confirm-edit[^>]*>Revise in the conversation<\/button>/);
      assert.match(log, /which deliverables/i);
      assert.doesNotMatch(log, /data-gpi-confirm-action/);
    } finally {
      language = window.EU_LANG = previousLanguage;
    }
  }
});

test('a pending plan action is inline when the workflow arrives before the transcript and plan summary', () => {
  const code = 'operator_plan_approval_required';
  const html = conversation(pendingSession, [], null, {
    next_action_code: code,
    stages: [{ id: 'plan', status: 'review_required', reason_code: code }],
  });
  const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
  assert.match(log, /data-gpi-confirm-action disabled[^>]*>批准并开始分析<\/button>/);
});

test('the inline plan-review card requires a choice for each uncertain deliverable', () => {
  for (const messages of [[], researchTurn]) {
    const html = conversation(pendingSession, messages, null, {
      next_action_code: 'host_delivery_request_ambiguous',
      stages: [{ id: 'plan', status: 'review_required', reason_code: 'host_delivery_request_ambiguous' }],
      plan_review_summary: {
        run_id: 'ward-recovery-plan',
        deliverables: { source: 'detector_prefill', requested: ['references'], check: ['manuscript_pdf'], confirmed: false },
      },
    });
    const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
    assert.equal((log.match(/aria-label="需要确认科研流程"/g) || []).length, 1);
    assert.match(log, /交付物/);
    assert.match(log, /<form[^>]*data-gpi-plan-deliverables[^>]*>/);
    assert.match(log, /<select name="manuscript_pdf" required[^>]*>\s*<option value="" selected/);
    assert.match(log, /<select name="references" required[^>]*>[\s\S]*?<option value="yes" selected/);
    assert.match(log, /data-gpi-deliverables-apply[^>]*>重新生成计划<\/button>/);
    assert.doesNotMatch(log, /data-gpi-confirm-action/);
  }
});

test('a reviewable plan keeps the existing approval alongside editable deliverables', () => {
  const html = pendingDecisionHtml({
    next_action_code: 'operator_plan_approval_required',
    plan_review_summary: {
      run_id: 'morning-recovery-plan',
      deliverables: { source: 'user_edit', requested: ['article_preview'], check: [], confirmed: true },
    },
  });
  const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
  assert.match(log, /<select name="article_preview" required[^>]*>[\s\S]*?<option value="yes" selected/);
  assert.match(log, /data-gpi-confirm-action[^>]*>Approve and start analysis<\/button>/);
});

test('choosing an article output for an analysis-only review shows the composer route', () => {
  const fields = ['manuscript_pdf', 'article_preview', 'references'].map(name => ({
    name, value: 'no', dataset: { gpiDeliverableOriginal: 'no' },
  }));
  const approve = { disabled: false, dataset: {} };
  const route = { hidden: true };
  const forward = { disabled: true };
  const form = {
    querySelectorAll: () => fields,
    querySelector: selector => selector.includes('article') ? route : forward,
    closest: () => ({ querySelector: () => approve }),
    hasAttribute: name => name === 'data-gpi-analysis-deliverables',
  };
  const owner = modules.require('planActions');
  const html = owner.deliverablesField({
    requested: ['manuscript_pdf'], check: [], confirmed: true, article_required: true,
  }, { tr: en => en, esc, disabled: false });
  assert.match(html, /separate article run/);
  assert.match(html, /data-gpi-confirm-edit[^>]*>Revise in the conversation<\/button>/);
  fields[0].value = 'yes';
  assert.deepEqual(owner.deliverableSelection(form), { requested: [], deferred: ['manuscript_pdf'], excluded: [] });
  assert.equal(route.hidden, false);
  assert.equal(approve.disabled, true);
  fields[0].value = 'no';
  assert.equal(owner.deliverableSelection(form), null);
  assert.equal(route.hidden, true);
  assert.equal(approve.disabled, false);
  fields[0].dataset.gpiDeliverableOriginal = '';
  assert.deepEqual(owner.deliverableSelection(form), { requested: [], deferred: [], excluded: ['manuscript_pdf'] });
  assert.equal(forward.disabled, false);
  fields[0].value = 'yes';
  assert.deepEqual(owner.deliverableSelection(form), { requested: [], deferred: ['manuscript_pdf'], excluded: [] });
});

test('saving a check choice starts a new plan review and never submits approval', async () => {
  const calls = [];
  const session = {
    session_id: 'session-ward',
    binding: { study_context_id: 'study-ward', study_revision: 6, run_id: 'review-ward' },
  };
  let workflow = {
    next_action_code: 'host_delivery_request_ambiguous',
    plan_review_summary: { run_id: 'review-ward', deliverables: { confirmed: false, check: ['manuscript_pdf'] } },
  };
  let busy = false;
  const owner = modules.require('planActions').create({
    tr: en => en,
    session: () => session,
    sessionIsStale: () => false,
    projectId: () => 'project-ward',
    workflow: () => workflow,
    busy: () => busy,
    setBusy: value => { busy = value; },
    setError: value => assert.equal(value, ''),
    render: () => {},
    refreshSession: async () => {},
    loadWorkflow: async () => { workflow = { next_action_code: 'plan_configuration_superseded' }; },
    appendMessage: () => {},
    watchChildJob: id => calls.push(['watch', id]),
    api: () => ({
      saveStudyContext: async request => calls.push(['edit', request]),
      loadStudyContext: async () => ({ context: { question: 'Describe shift recovery.', data_source: { path: '/synthetic/ward' } } }),
      startAgentRun: async request => { calls.push(['generate', request]); return { job_id: 'candidate-job' }; },
      submitAgentRunReview: () => assert.fail('Selecting a deliverable is never approval'),
    }),
  });
  await owner.confirmDecision({ deliverables: { requested: [], deferred: [], excluded: ['manuscript_pdf'] } });
  assert.deepEqual(calls[0], ['edit', {
    id: 'study-ward', expected_revision: 6, active: false,
    deliverables_source_run_id: 'review-ward',
    plan_deliverables: { requested: [], deferred: [], excluded: ['manuscript_pdf'], confirmed: true, source: 'user_edit' },
  }]);
  assert.equal(calls[1][0], 'generate');
  assert.equal(calls[1][1].question, 'Describe shift recovery.');
  assert.equal(calls[1][1].planner_start_mode, 'fresh');
  assert.deepEqual(calls[2], ['watch', 'candidate-job']);
});

test('a result-product choice outside the three fields still opens the inline composer action', () => {
  const html = pendingDecisionHtml({
    next_action_code: 'host_delivery_request_ambiguous',
    plan_review_summary: {
      run_id: 'ward-results-plan',
      deliverables: { requested: [], check: [], confirmed: true, source: 'detector_prefill' },
    },
  });
  const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
  assert.match(log, /data-gpi-plan-deliverables/);
  assert.match(log, /data-gpi-confirm-edit[^>]*>Revise in the conversation<\/button>/);
  assert.doesNotMatch(log, /data-gpi-confirm-action/);
});

test('the pending ward transfer plan offers an inline analysis-only choice for requested article files', () => {
  for (const [cardLanguage, description, forward] of [
    ['en', 'Generate a fresh plan for analysis only', 'Generate fresh plan'],
    ['zh', '重新生成仅分析计划', '重新生成计划'],
  ]) {
    const html = pendingDecisionHtml({
      next_action_code: 'first_result_delivery_requires_article',
      plan_review_summary: {
        run_id: 'transfer-outcomes-plan',
        deliverables: { article_required: true, source: 'user_edit', requested: ['manuscript_pdf'],
          check: [], confirmed: true },
      },
    }, cardLanguage);
    const log = html.slice(html.indexOf('data-gpi-log'), html.indexOf('data-gpi-input'));
    assert.ok(log.includes(description));
    assert.ok(log.includes(`data-gpi-deliverables-apply>${forward}</button>`));
    assert.doesNotMatch(log, /data-gpi-confirm-action/);
  }
});
