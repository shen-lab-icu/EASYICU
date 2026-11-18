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
const owner = modules.require('runStop').create({ tr, runFailureText: () => tr('Stopped.', '已停止。'),
  workflow: () => workflow, latestRun: () => workflow.runs.find(row => row.authoritative),
  resourceButton: (resource, label, options) => `<button class="${(options && options.className) || 'gpi-resource-link'}" data-resource="${resource.artifact}">${label}</button>`,
  canAct: () => true, canPreview: () => true });
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
  sessionIsStale: () => false,
  studyFamilyGapText: () => ({ refusal: 'The data do not support this study.', instead: 'Change the study or source.' }),
});
const events = {};
const drafts = [];
const calls = [];
const recovery = { scrollIntoView() { calls.push('reveal'); }, setAttribute() {}, focus() {} };
const state = { session: {}, busy: false, workflow, latestRun: run,
  host: { addEventListener: (name, callback) => { events[name] = callback; }, querySelector: selector => selector === '.gpi-confirmation[data-gpi-review-recovery]' ? recovery : null } };
modules.require('events').create({ state, tr,
  RUN_FILES: { handleClick: () => false },
  RESOURCE_OWNER: { fromButton: () => null },
  retryFailedExecution: reason => calls.push(reason || 'retry'), confirmWorkflowAction: () => calls.push('confirm'),
  editWorkflow: () => calls.push('edit'), render: () => drafts.push(state.draft),
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

if (catalog) {
  const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
  const failures = [];
  const outcome = modules.require('runOutcome').create({
    tr, esc: window.EU_HTML.esc, iconHtml: () => '', resourceButton: () => '',
    manuscriptNotPassedText: code => code,
  });
  for (language of ['en', 'zh']) {
    window.EU_LANG = language;
    for (const { code, cause, surface, design, run, job, workflow: offer } of cases) {
      const name = [surface, code, cause, language].filter(Boolean).join('/');
      try {
        calls.length = drafts.length = 0;
        workflow = state.workflow = { ...offer, runs: run ? [{ ...run, authoritative: true }] : [] };
        state.latestRun = run || null;
        state.session.archived_child_jobs = job ? [job] : [];
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
        assert.match(html, job ? /data-gpi-job-stop/ : /data-gpi-run-stop=/);
        const ids = [...html.matchAll(/data-gpi-stop-action="([^"]+)"/g)].map(match => match[1]);
        if (design) {
          assert.deepEqual(ids, [], 'revisit the design exception once an owner can act');
          continue;
        }
        assert.ok(ids.length, 'the projected stop renders a forward action without a receipt or generic fallback');
        for (const choice of (job || run).next_actions) {
          assert.ok(ids.includes(choice.action_id), `unrendered action ${choice.kind}`);
          calls.length = drafts.length = 0;
          state.showSetup = false;
          if (job) clickControl('[data-gpi-stop-action]', {
            gpiStopJob: job.job_id, gpiStopAction: choice.action_id, gpiStopWorkflow: offer.next_action_code,
          });
          else clickAction(choice.action_id);
          if (['edit_study', 'rephrase_in_scope', 'use_alternative_concept', 'drop_variable', 'rerun_without_element'].includes(choice.kind)
            || (choice.kind === 'edit_plan' && (choice.params.concept || choice.params.step_id || choice.params.element || choice.params.change_question))) {
            assert.equal(drafts.length, 1, 'the existing composer opens a draft');
            assert.ok(drafts[0].trim());
            assert.deepEqual(calls, [], 'a draft does not start any work');
            if (choice.kind === 'use_alternative_concept') {
              assert.ok(drafts[0].includes(choice.params.alternative_concept));
              assert.ok(drafts[0].includes(choice.params.concept));
            }
          } else if (choice.kind === 'check_connection') {
            assert.equal(state.showSetup, true, 'the current model setup opens');
          } else {
            const expected = choice.kind === 'reexport_data' ? 'begin_local_selection'
              : choice.kind === 'retry' ? (offer.next_action_code === 'failed_pipeline_execution_retry_available' ? 'retry' : 'confirm')
                : ['failed_pipeline_requires_fresh_plan', 'failed_pipeline_execution_retry_futile'].includes(offer.next_action_code)
                  ? 'confirm' : 'edit';
            assert.deepEqual(calls, [expected]);
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
