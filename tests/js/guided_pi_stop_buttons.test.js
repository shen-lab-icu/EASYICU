'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh', requestAnimationFrame: () => {} };
const catalog = process.argv[2] === '--catalog';
for (const file of process.argv.slice(catalog ? 3 : 2)) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
let language = 'zh';
const tr = (en, zh) => language === 'zh' ? zh : en;
const action = (kind, route, params = {}) => ({ kind, action_id: `${kind}_0123456789abcdef`,
  label_key: `stop_next_actions.${kind}`, params: { route, ...params } });
const edit = action('edit_plan', 'plan_edit', { intent: 'candidate_plan', planner_start_mode: 'fresh', requires_explicit_choice: true, requires_plan_approval: true });
const run = { run_ref: 'r1', run_id: 'synthetic', run_status: 'failed',
  artifact_names: ['source_run_manifest.json'], next_actions: [action('retry', 'retry', { stage: 'execution' }), edit] };
let workflow = { next_action_code: 'failed_pipeline_execution_retry_available', runs: [{ ...run, authoritative: true }] };
const { runFailureText = () => tr('Stopped.', '已停止。') } = modules.optional('errorText')?.create({
  tr, staticPreview: () => false,
}) || {};
const owner = modules.require('runStop').create({ tr, runFailureText,
  workflow: () => workflow, latestRun: () => workflow.runs.find(row => row.authoritative),
  resourceButton: (resource, label, options) => `<button class="${(options && options.className) || 'gpi-resource-link'}" data-resource="${resource.artifact}">${label}</button>`,
  canAct: () => true, canPreview: () => true,
  connectionChecks: () => ({
    credentialVerifiedAt: state.runtime?.configuration?.verified_at,
    codexAuth: state.codexAuth,
  }) });
let statedDetail;
modules.require('runStop').create({ tr, runFailureText: (_code, detail) => {
  statedDetail = detail;
  return detail.statedVariable;
} }).render({ savedRuns: [{ ...run, gate_reason_code: 'research_pipeline_progressive_compile_failed',
  gate_detail: { reason_code: 'progressive_family_spec_stated_adjustment_unavailable',
    cause_code: 'stated_adjustment_name_unresolved', stated_variable: '试验量表甲' } }] });
assert.equal(statedDetail.statedVariable, '试验量表甲');
assert.equal(statedDetail.cause, 'stated_adjustment_name_unresolved');
const confirmation = modules.optional('confirmation')?.create({
  tr, esc: window.EU_HTML.esc, iconHtml: () => '', resourceButton: () => '',
  workflow: () => workflow, session: () => state.session, busy: () => false,
  sessionIsStale: () => false, runFailureText,
  studyFamilyGapText: () => ({ refusal: 'The data do not support this study.', instead: 'Change the study or source.' }),
});
const events = {};
const drafts = [];
const calls = [];
const authorizationPopup = {};
const recovery = { scrollIntoView() { calls.push('reveal'); }, setAttribute() {}, focus() {} };
const state = { session: {}, busy: false, workflow, latestRun: run,
  host: { addEventListener: (name, callback) => { events[name] = callback; }, querySelector: selector => selector === '.gpi-confirmation[data-gpi-review-recovery]' ? recovery : null } };
modules.require('events').create({ state, tr,
  RUN_FILES: { handleClick: () => false },
  RESOURCE_OWNER: { fromButton: () => null },
  retryFailedExecution: reason => calls.push(reason || 'retry'), confirmWorkflowAction: () => calls.push('confirm'),
  editWorkflow: () => calls.push('edit'), render: () => drafts.push(state.draft),
  openAuthorizationPopup: () => authorizationPopup,
  startCodexLogin: (flow, popup) => calls.push([flow, popup]),
  authorizeDataSource: action => calls.push(action), confirmPlanDecision: selection => calls.push(selection),
  COHORT_ELIGIBILITY: { actionFromEvent: () => null },
}).wire();
function clickControl(selector, dataset) {
  const button = { dataset };
  events.click({ target: { closest: candidate => candidate === selector ? button : null }, preventDefault() {} });
}
function clickAction(id, actionCode = workflow.next_action_code) {
  clickControl('[data-gpi-stop-action]', { gpiStopAction: id, gpiStopRun: 'synthetic', gpiStopWorkflow: actionCode });
}
function click(kind) {
  clickAction(`${kind}_0123456789abcdef`);
}
for (language of ['zh', 'en']) {
  window.EU_LANG = language;
  const html = owner.render({ savedRuns: [run] });
  assert.ok(html.includes('data-gpi-stop-action="retry_0123456789abcdef"'));
  assert.ok(html.includes(tr('Retry from the failed step', '从失败步骤重试')));
  assert.ok(html.includes(tr('Regenerate the plan', '重新生成计划')));
  assert.ok(html.includes('source_run_manifest.json'));
  click('retry');
  click('edit_plan');
  assert.deepEqual(calls.splice(0), ['retry', 'edit']);
  workflow.next_action_code = 'failed_pipeline_requires_fresh_plan';
  clickAction(edit.action_id, 'failed_pipeline_execution_retry_available');
  assert.deepEqual(calls, [], 'changing the workflow cannot turn a stale edit into a start');
  assert.ok(!owner.render({ savedRuns: [run] }).includes('data-gpi-stop-action="retry_'));
  click('retry');
  assert.deepEqual(calls, [], 'a stale retry click cannot start a run');
  click('edit_plan');
  assert.deepEqual(calls.splice(0), ['confirm']);
  workflow.next_action_code = 'failed_pipeline_execution_retry_available';
  state.busy = true;
  click('retry');
  assert.deepEqual(calls, []);
  state.busy = false;
  const historical = { ...run, run_id: 'old', run_ref: 'r0' };
  const old = owner.render({ savedRuns: [historical] });
  assert.ok(!old.includes('data-gpi-stop-action'));
  assert.ok(old.includes('source_run_manifest.json'), 'historical stops retain their own receipt');
  assert.ok(old.includes('class="btn sm" data-resource="source_run_manifest.json"'), 'the receipt is a button like the other choices');
  assert.equal(modules.require('runStop').currentRun(run,
    { runs: [{ run_id: 'new_authoritative_run', authoritative: true }] }, run), false,
  'the workflow authority wins over a stale local latest-run pointer');
  const data = { ...run, next_actions: [action('reexport_data', 'data_extraction', { requires_explicit_choice: true })] };
  assert.ok(owner.render({ savedRuns: [data] }).includes('data-gpi-stop-action="reexport_data_'));
  workflow.runs[0].next_actions = data.next_actions;
  click('reexport_data');
  assert.deepEqual(calls.splice(0), ['begin_local_selection']);
  workflow.runs[0].next_actions = run.next_actions;
  // No stop asks how the model handles data: an old saved settings choice
  // offers no button of its own and enters the recovery card.
  const declare = owner.render({ savedRuns: [{ ...run, next_actions: [action('open_provider_settings', 'provider_settings')] }] }, { hasRecovery: true });
  assert.ok(!declare.includes('data-gpi-stop-declare'));
  assert.ok(!declare.includes('数据处理方式') && !declare.includes('handles data'));
  assert.ok(declare.includes('data-gpi-stop-recovery'));
  const forged = { ...run, next_actions: [{ ...edit, params: { ...edit.params, route: 'retry' } }] };
  assert.ok(!owner.render({ savedRuns: [forged] }).includes('data-gpi-stop-action'));
  const unknown = { ...run, next_actions: [action('imagined_action', 'retry')] };
  assert.ok(!owner.render({ savedRuns: [unknown] }, { hasRecovery: true }).includes('imagined_action'));
  assert.ok(owner.render({ savedRuns: [unknown] }, { hasRecovery: true }).includes('data-gpi-stop-recovery'), 'unsupported choices enter the existing recovery card');
  events.click({ target: { closest: selector => selector === '[data-gpi-stop-recovery]' ? {} : null } });
  assert.deepEqual(calls.splice(0), ['reveal']);
  workflow.next_actions = [];
  assert.ok(!owner.render({ savedRuns: [run] }).includes('data-gpi-stop-action'), 'the current workflow withdraws historical offers');
  delete workflow.next_actions;
  const noOwners = modules.require('runStop').create({ tr, workflow: () => workflow, latestRun: () => run });
  assert.ok(!noOwners.render({ savedRuns: [run] }).includes('<button'));
}
console.log('stop actions: bilingual rendering, owner dispatch, stale/busy/historical guards passed');

if (!catalog) {
  for (const [reason, kind, route, resolved] of [
    ['no_export_files', 'extract', 'data', 'provider_ready_to_generate_plan'],
    ['research_pipeline_pi_verified_credentials_required', 'agent-run', 'connect', 'plan_execution_upgrade_required'],
  ]) {
    const offer = route === 'data'
      ? action('reexport_data', 'data_extraction', { requires_explicit_choice: true })
      : action('check_connection', 'model_connection');
    const job = {
      job_id: 'night_shift_export', kind, status: 'failed', error_code: reason,
      finished_at_epoch: Date.parse('2026-01-05T18:00:00Z') / 1000, next_actions: [offer],
    };
    state.session.archived_child_jobs = [job];
    state.runtime = { configuration: { verified_at: '2026-01-05T16:00:00Z' } };
    workflow = state.workflow = { next_action_code: reason, next_actions: [offer], runs: [] };
    const clickJob = () => clickControl('[data-gpi-stop-action]', {
      gpiStopJob: job.job_id, gpiStopAction: offer.action_id, gpiStopWorkflow: workflow.next_action_code,
    });
    assert.match(owner.renderJob(state.session), /data-gpi-job-stop/);
    clickJob();
    assert.ok(route === 'data' ? calls.pop() === 'begin_local_selection' : state.showSetup);
    state.showSetup = false;
    workflow.next_actions = [];
    workflow.active_job = { job_id: job.job_id, status: 'failed' };
    assert.equal(owner.renderJob(state.session), '', 'a withdrawn offer for this job clears its card');
    clickJob();
    assert.deepEqual(calls, []);
    assert.equal(state.showSetup, false);
    workflow.next_actions = [offer];
    delete workflow.active_job;
    workflow.next_action_code = resolved;
    assert.match(owner.renderJob(state.session), /data-gpi-job-stop/,
      'a change in the workflow label alone cannot dismiss a stop');
    const supersedingDate = '2026-01-05T19:00:00Z';
    for (const [where, setRevision, resetRevision] of [
      ['study', () => { workflow.study_setup_receipt = { updated_at: supersedingDate }; },
        () => { delete workflow.study_setup_receipt; }],
      ['session binding', () => {
        state.session.binding = { study_context_id: 'evening_assays', study_revision: 4 };
        workflow.study_setup_receipt = { study_context_id: 'evening_assays', revision: 5 };
      }, () => { delete state.session.binding; delete workflow.study_setup_receipt; }],
      ...(kind === 'extract'
        ? [['export', () => { workflow.bound_export_updated_at = supersedingDate; },
          () => { delete workflow.bound_export_updated_at; }]]
        : [['credentials', () => { state.runtime = { configuration: { verified_at: supersedingDate } }; },
          () => { delete state.runtime; }]]),
    ]) {
      setRevision();
      assert.equal(owner.renderJob(state.session), '', `a new ${where} revision supersedes the job`);
      clickJob();
      assert.deepEqual(calls, []);
      assert.equal(state.showSetup, false);
      resetRevision();
    }
    if (kind === 'agent-run') {
      state.session.research_provider = { provider: 'codex' };
      state.runtime = { configuration: { verified_at: supersedingDate } };
      assert.match(owner.renderJob(state.session), /data-gpi-job-stop/,
        'the API verification is unrelated to the evening Codex attempt');
      delete state.session.research_provider;
      delete state.runtime;
    }
    workflow.next_action_code = reason;
    workflow.runs = [{ run_id: 'night_shift_candidate', updated_at: '2026-01-05T19:00:00Z' }];
    assert.equal(owner.renderJob(state.session), '', 'a newer run owns the workflow');
    clickJob();
    assert.deepEqual(calls, []);
    assert.equal(state.showSetup, false);
  }
  const job = {
    job_id: 'dusk_assay_export', kind: 'extract', status: 'failed', error_code: 'no_export_files',
    next_actions: [action('reexport_data', 'data_extraction', { requires_explicit_choice: true })],
  };
  state.session.archived_child_jobs = [job];
  workflow = state.workflow = { next_action_code: 'provider_ready_to_generate_plan',
    next_actions: [], runs: [{ run_id: 'prior_dusk_assay' }] };
  assert.match(owner.renderJob(state.session), /data-gpi-job-stop/,
    'a failed export still offers recovery when the study has a prior export and an undated run');
  clickControl('[data-gpi-stop-action]', {
    gpiStopJob: job.job_id, gpiStopAction: job.next_actions[0].action_id, gpiStopWorkflow: workflow.next_action_code,
  });
  assert.deepEqual(calls.splice(0), ['begin_local_selection']);

  workflow.next_action_code = job.error_code;
  workflow.next_actions = job.next_actions;
  for (const [finishedAt, runUpdatedAt] of [
    [Date.parse('2026-01-05T18:00:00Z') / 1000, undefined],
    [undefined, '2026-01-05T17:00:00Z'],
  ]) {
    job.finished_at_epoch = finishedAt;
    workflow.runs[0].updated_at = runUpdatedAt;
    assert.match(owner.renderJob(state.session), /data-gpi-job-stop/,
      'missing dates cannot establish that an old run replaced the evening export failure');
  }
  state.session.conversation_replay = {
    updated_at: '2026-01-05T21:00:00Z',
    turns: [{
      kind: 'host_action', child_job_id: job.job_id, status: 'failed', ended_at: '2026-01-05T18:00:00Z',
    }],
  };
  workflow.bound_export_updated_at = '2026-01-05T19:00:00Z';
  assert.equal(owner.renderJob(state.session), '',
    'the overnight replacement export supersedes the failed handover even after later messages');
  delete state.session.conversation_replay;
  console.log('job stops: current workflow offer and supersession checks passed');
}

async function testSupersededDecisionRefresh() {
  const steps = [];
  const planActions = modules.require('planActions').create({
    session: () => ({ session_id: 'evening_review', binding: { study_revision: 4, run_id: 'overnight_candidate' } }),
    busy: () => false, sessionIsStale: () => false, projectId: () => 'pulse_project',
    api: () => ({
      async confirmPiCopilotPlanDecision(session, request) {
        steps.push([session, request.run_id, request.option_id]);
        throw { code: 'plan_decision_source_superseded', message: 'The evening review was replaced by a morning review.' };
      },
    }),
    async refreshSession(force) {
      await Promise.resolve();
      steps.push(['refreshSession', force]);
    },
    async loadWorkflow() {
      await Promise.resolve();
      steps.push('loadWorkflow');
    },
    setBusy: value => steps.push(['busy', value]),
    setError: value => steps.push(['error', value]),
    errorText: error => error.message,
    render: () => steps.push('render'),
  });
  await planActions.confirmDecision({ decision_code: 'meal_window', option_id: 'overnight' });
  assert.deepEqual(steps, [
    ['busy', true], ['error', ''], 'render', ['evening_review', 'overnight_candidate', 'overnight'],
    ['refreshSession', true], 'loadWorkflow', ['busy', false],
    ['error', 'The evening review was replaced by a morning review.'], 'render',
  ]);
  console.log('superseded decision refreshes the session and workflow before unlocking');
}

if (catalog) {
  const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
  const failures = [];
  const outcome = modules.require('runOutcome').create({
    tr, esc: window.EU_HTML.esc, iconHtml: () => '', resourceButton: () => '',
    manuscriptNotPassedText: code => code,
  });
  for (language of ['en', 'zh']) {
    window.EU_LANG = language;
    for (const { code, cause, surface, design, routes, drafts: reviewDrafts, label, run, job, workflow: offer,
      expect_edit, missing, recovered, superseded_workflow, initial_provider, verified_provider, provider } of cases) {
      const name = [surface, code, cause, language].filter(Boolean).join('/');
      try {
        calls.length = drafts.length = 0;
        workflow = state.workflow = { ...offer, runs: run ? [{ ...run, authoritative: true }] : offer.runs || [] };
        state.latestRun = run || null;
        state.session.archived_child_jobs = job ? [job] : [];
        if (surface === 'confirmation-edit') {
          const html = confirmation.workflowConfirmationHtml();
          assert.match(html, /<button(?![^>]* disabled)[^>]*data-gpi-confirm-action/, 'retry remains enabled');
          assert.equal(html.includes('data-gpi-confirm-edit'), expect_edit, 'a plan recovery offer controls fresh planning');
          clickControl('[data-gpi-confirm-action]', {});
          assert.deepEqual(calls.splice(0), ['confirm']);
          if (expect_edit) {
            assert.match(html, /<button(?![^>]* disabled)[^>]*data-gpi-confirm-edit/);
            clickControl('[data-gpi-confirm-edit]', {});
            assert.deepEqual(calls, ['edit']);
          }
          continue;
        }
        if (surface === 'job-supersession') {
          state.session.conversation_replay = { updated_at: '2026-01-05T18:00:00Z' };
          if (provider) state.session.research_provider = { provider };
          try {
            Object.assign(state, initial_provider);
            const [choice] = job.next_actions;
            const html = owner.renderJob(state.session);
            assert.match(html, /data-gpi-job-stop/, 'the failed job keeps its recovery card');
            const buttons = [...html.matchAll(/(<button(?![^>]* disabled)[^>]*>)([^<]*)<\/button>/g)];
            const control = buttons.find(([, button]) => button.includes(`data-gpi-stop-action="${choice.action_id}"`));
            assert.ok(control, 'the recovery action is enabled');
            const [, button, buttonLabel] = control;
            if (label) assert.equal(buttonLabel, tr(...label));
            const clickJob = (workflowCode = readAttribute('workflow')) => clickControl('[data-gpi-stop-action]', {
              gpiStopJob: readAttribute('job'), gpiStopAction: readAttribute('action'), gpiStopWorkflow: workflowCode,
            });
            function readAttribute(name) {
              return new RegExp(`data-gpi-stop-${name}="([^"]+)"`).exec(button)?.[1];
            }
            clickJob();
            if (label) assert.deepEqual(calls.splice(0), [['browser', authorizationPopup]]);
            else if (provider) assert.equal(state.showSetup, true);
            else assert.deepEqual(calls.splice(0), ['begin_local_selection']);
            state.showSetup = false;
            if (initial_provider) {
              Object.assign(state, verified_provider);
              assert.match(owner.renderJob(state.session), /data-gpi-job-stop/, 'the active auth stop still needs recovery');
              Object.assign(state, initial_provider);
            }
            Object.assign(workflow, superseded_workflow);
            if (provider) assert.match(owner.renderJob(state.session), /data-gpi-job-stop/, 'a workflow label does not verify a connection');
            Object.assign(state, verified_provider);
            assert.equal(owner.renderJob(state.session), '', 'the archived job was superseded');
            clickJob(workflow.next_action_code);
            assert.deepEqual(calls, []);
            assert.equal(state.showSetup, false);
          } finally {
            delete state.session.conversation_replay;
            delete state.session.research_provider;
            delete state.runtime;
            delete state.codexAuth;
          }
          continue;
        }
        if (surface === 'coordinate-recovery') {
          const card = confirmation.workflowConfirmationHtml();
          const stop = owner.render({ savedRuns: [run] });
          const html = card + stop;
          assert.ok(html.trim());
          const actions = offer.next_actions;
          if (code === 'study_coordinate_unrecognized') {
            assert.doesNotMatch(html, /Choose a database that records it|请换一个记录了它的数据库/);
            assert.doesNotMatch(html, /data-gpi-confirm-action/, 'the unknown coordinate needs clarification');
          }
          if (missing) {
            for (const concept of missing) assert.ok(stop.includes(concept), `the stop text names ${concept}`);
            assert.ok(!stop.includes(recovered), 'the stop text omits a coordinate the source now has');
          }
          const canRetry = actions.some(choice => choice.kind === 'retry' && choice.params.stage === 'execution');
          if (code === 'failed_pipeline_execution_retry_available'
            && /^research_pipeline_(provider_timeout|execution_failed)\//.test(cause)) {
            assert.equal(canRetry, true, 'the eligible failed step projects an execution retry');
            assert.deepEqual(actions.map(choice => choice.kind), ['retry', 'edit_plan']);
            assert.match(card, /data-gpi-confirm-action/, 'the projected retry offers execution');
          }
          const canPlan = actions.some(({ kind, params }) => (kind === 'retry' && params.stage === 'planning')
            || kind === 'drop_variable' || kind === 'rerun_without_element'
            || (kind === 'edit_plan' && params.change_question !== true));
          if (canPlan && code.startsWith('failed_pipeline_')) {
            if (code === 'failed_pipeline_execution_retry_available') {
              assert.match(card, /data-gpi-confirm-edit/, 'the retry card also offers fresh planning');
              assert.equal(card.includes('data-gpi-confirm-action'), canRetry, 'only a projected retry offers execution');
            } else {
              assert.match(card, /data-gpi-confirm-action/, 'the failed attempt keeps its workflow confirmation');
            }
          }
          for (const choice of actions) {
            assert.ok(html.includes(choice.action_id), `the ${choice.kind} choice stays visible`);
          }
          const controls = [...html.matchAll(/<button(?![^>]* disabled)[^>]*>/g)].map(match => match[0]);
          assert.ok(controls.length, 'at least one visible executable action');
          for (const button of controls) {
            calls.length = drafts.length = 0;
            for (const [attribute, expected] of [
              ['data-gpi-confirm-action', 'confirm'], ['data-gpi-confirm-edit', 'edit'],
            ]) {
              if (button.includes(attribute)) {
                clickControl(`[${attribute}]`, {});
                assert.deepEqual(calls, [expected]);
              }
            }
            const id = /data-gpi-stop-action="([^"]+)"/.exec(button)?.[1];
            if (id) {
              const choice = actions.find(action => action.action_id === id);
              assert.ok(choice, 'the button has a projected choice');
              const route = modules.require('runStop').actionRoute(choice, workflow);
              clickControl('[data-gpi-stop-action]', {
                gpiStopAction: id, gpiStopWorkflow: code,
                ...(button.includes('data-gpi-stop-review') ? { gpiStopReview: '' } : { gpiStopRun: run.run_id }),
              });
              if (route === 'draft') assert.equal(drafts.length, 1);
              else assert.deepEqual(calls, [route]);
            }
            assert.equal(calls.length + drafts.length, 1, 'every visible button reaches its owner');
          }
          continue;
        }
        if (surface === 'outcome') {
          workflow.stages = [{ id: 'analysis', status: 'complete' }];
          const html = outcome.render(run, workflow);
          assert.match(html, /data-manuscript-outcome="not_passed"/);
          const mode = workflow.report_rewrite_available ? 'report_only' : 'restore';
          assert.ok(html.includes('data-gpi-run-outcome-retry="' + mode + '"'));
          clickControl('[data-gpi-run-outcome-retry]', { gpiRunOutcomeRetry: mode });
          assert.deepEqual(calls, [mode]);
          continue;
        }
        if (surface === 'review') {
          const html = confirmation.workflowConfirmationHtml();
          assert.match(html, / data-gpi-review-recovery/, 'the review exposes a forward action');
          if (reviewDrafts) {
            assert.ok(confirmation.planChangeDraft().includes(reviewDrafts[0]));
            const controls = [...html.matchAll(/data-gpi-stop-review data-gpi-stop-action="([^"]+)" data-gpi-stop-workflow="([^"]+)"/g)];
            assert.equal(controls.length, reviewDrafts.length);
            for (const [i, [, id, workflowCode]] of controls.entries()) {
              clickControl('[data-gpi-stop-action]', { gpiStopReview: '', gpiStopAction: id, gpiStopWorkflow: workflowCode });
              assert.ok(drafts.pop().includes(reviewDrafts[i]));
            }
            assert.deepEqual(calls, [], 'review edits never start work');
            continue;
          }
          const primary = /<button[^>]*data-gpi-confirm-action[^>]*>/.exec(html)?.[0];
          if (primary && !primary.includes(' disabled')) {
            assert.ok(confirmation.workflowConfirmation().approve || confirmation.workflowConfirmation().retryLabel);
            clickControl('[data-gpi-confirm-action]', {});
            assert.deepEqual(calls, ['confirm']);
          } else if (html.includes('data-gpi-plan-decision-option')) {
            const decision = /data-gpi-plan-decision-code="([^"]+)" data-gpi-plan-decision-option="([^"]+)" data-gpi-plan-decision-card="([^"]+)"/.exec(html);
            assert.ok(decision);
            const [, decisionCode, optionId, digest] = decision;
            clickControl('[data-gpi-plan-decision-option]', {
              gpiPlanDecisionCode: decisionCode, gpiPlanDecisionOption: optionId, gpiPlanDecisionCard: digest,
            });
            assert.deepEqual(calls, [{ decision_code: decisionCode, option_id: optionId, card_sha256: digest }]);
          } else {
            assert.match(html, /data-gpi-confirm-edit/);
            assert.ok(confirmation.planChangeDraft().trim());
            clickControl('[data-gpi-confirm-edit]', {});
            assert.deepEqual(calls, ['edit']);
          }
          continue;
        }
        const html = job ? owner.renderJob(state.session) : owner.render({ savedRuns: [run] });
        if (job && design) {
          assert.equal(html, '', 'this workflow has no current job stop action');
          continue;
        }
        assert.match(html, job ? /data-gpi-job-stop/ : /data-gpi-run-stop=/);
        const ids = [...html.matchAll(/data-gpi-stop-action="([^"]+)"/g)].map(match => match[1]);
        if (label) assert.ok(html.includes(tr(...label)));
        if (routes) {
          assert.deepEqual(run.next_actions.map(choice => modules.require('runStop').actionRoute(choice, workflow)), routes);
          if (!routes.includes('confirm')) {
            assert.doesNotMatch(confirmation.workflowConfirmationHtml(), /data-gpi-confirm-action/,
              'the companion confirmation must not restart the unchanged study');
          }
        }
        if (design) {
          assert.deepEqual(ids, [], 'revisit the design exception once an owner can act');
          continue;
        }
        assert.ok(ids.length, 'the projected stop renders a forward action without a receipt or generic fallback');
        for (const [i, choice] of (job || run).next_actions.entries()) {
          assert.ok(ids.includes(choice.action_id), `unrendered action ${choice.kind}`);
          calls.length = drafts.length = 0;
          state.showSetup = false;
          if (job) clickControl('[data-gpi-stop-action]', {
            gpiStopJob: job.job_id, gpiStopAction: choice.action_id, gpiStopWorkflow: offer.next_action_code,
          });
          else clickAction(choice.action_id);
          const expected = routes?.[i] || (
            ['edit_study', 'rephrase_in_scope', 'use_alternative_concept', 'drop_variable', 'rerun_without_element'].includes(choice.kind)
              || (choice.kind === 'edit_plan' && (choice.params.concept || choice.params.step_id || choice.params.element || choice.params.change_question))
              ? 'draft' : choice.kind === 'check_connection' ? 'connect'
                : choice.kind === 'reexport_data' ? 'data'
                  : choice.kind === 'retry' ? (offer.next_action_code === 'failed_pipeline_execution_retry_available' ? 'retry' : 'confirm')
                    : ['failed_pipeline_requires_fresh_plan', 'failed_pipeline_execution_retry_futile'].includes(offer.next_action_code)
                      ? 'confirm' : 'edit'
          );
          if (expected === 'draft') {
            assert.equal(drafts.length, 1, 'the existing composer opens a draft');
            assert.ok(drafts[0].trim());
            assert.deepEqual(calls, [], 'a draft does not start any work');
            if (choice.kind === 'use_alternative_concept') {
              assert.ok(drafts[0].includes(choice.params.alternative_concept));
              assert.ok(drafts[0].includes(choice.params.concept));
            }
          } else if (expected === 'connect') {
            assert.equal(state.showSetup, true, 'the current model setup opens');
          } else {
            assert.deepEqual(calls, [expected === 'data' ? 'begin_local_selection' : expected]);
          }
        }
      } catch (error) {
        failures.push(`${name} — ${error.message}`);
      }
    }
  }
  assert.equal(failures.length, 0, failures.join('\n'));
  console.log(`stop catalog: ${cases.length} cases, ${2 * cases.length} bilingual checks`);
}

if (!catalog) testSupersededDecisionRefresh().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
