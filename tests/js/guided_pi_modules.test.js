'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');

function load(window = {}) {
  const context = { window };
  vm.createContext(context);
  vm.runInContext(source, context);
  return context.window.EasyICU.guidedPi;
}

const modules = load();
const api = { render: () => 'ok' };
assert.equal(modules.declare('preview', api), api);
assert.equal(modules.require('preview').render(), 'ok');
assert.equal(modules.optional('missing'), null);
assert.throws(() => modules.require('missing'), /is not declared: missing/);
assert.throws(() => modules.declare('preview', {}), /already declared: preview/);
assert.throws(() => modules.declare('invalid-name', {}), /non-empty identifier/);
assert.throws(() => modules.declare('emptyApi', null), /API must be an object/);
assert.throws(() => load({ EasyICU: { guidedPi: {} } }), /namespace already exists/);
assert.equal(Object.isFrozen(modules.require('preview')), true);

// D-P3-2: errorText owner contract, including the raw fallback branch.
// The registry test above only covers declare/require; the errorText module
// must stay loadable beside it and keep its fallback returning raw transport
// text verbatim (callers esc() it per D-P2-2).
(function testErrorText() {
  const dir = path.dirname(path.resolve(process.argv[2]));
  let errorTextPath = path.join(dir, 'screens-guided-pi-error-text.js');
  if (!fs.existsSync(errorTextPath)) {
    errorTextPath = path.join(__dirname, '..', '..', 'src', 'easyicu', 'webserver', 'static', 'js', 'screens-guided-pi-error-text.js');
  }
  const errorTextSource = fs.readFileSync(errorTextPath, 'utf8');
  const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const context = { window: { EU_HTML: { esc }, EU_LANG: 'en' } };
  vm.createContext(context);
  vm.runInContext(source, context);
  vm.runInContext(errorTextSource, context);
  const guided = context.window.EasyICU.guidedPi;
  const tr = (en, zh) => en;
  const live = guided.require('errorText').create({ tr, staticPreview: () => false }).errorText;
  const preview = guided.require('errorText').create({ tr, staticPreview: () => true }).errorText;

  // Empty input stays empty (no fallback).
  assert.equal(live(''), '');
  assert.equal(live(null), '');
  assert.equal(live(undefined), '');

  // Known code maps through tr (not the fallback).
  assert.match(live({ code: 'pi_provider_auth_failed' }), /model service rejected/);

  // Static-preview Failed-to-fetch branch (terminology preserved).
  assert.match(preview({ message: 'Failed to fetch' }), /static preview/);

  // A blocked data foundation names its lower-layer cause from codes only:
  // the gate detail code and the concept ids the data lacked, never free text.
  const runFailure = guided.require('errorText').create({ tr, staticPreview: () => false }).runFailureText;
  assert.match(runFailure('data_foundation_blocked'), /Data preparation did not pass/);
  const detailed = runFailure('data_foundation_blocked', { code: 'required_concepts_unavailable', missing: ['icu_unit_type', 'charlson'] });
  assert.match(detailed, /Data preparation did not pass/);
  assert.match(detailed, /icu_unit_type, charlson/);
  assert.match(runFailure('data_foundation_blocked', { code: 'outcome_concept_undeclared' }), /No executable outcome variable/);
  assert.equal(runFailure('data_foundation_blocked', { code: 'unknown_detail_code' }), runFailure('data_foundation_blocked'));

  // A typed compile stop names its own cause; the generic compile sentence
  // would name one that did not happen.
  const unwritable = runFailure('research_pipeline_progressive_compile_failed', { code: 'progressive_family_result_contract_unwritable' });
  assert.match(unwritable, /this survival question needs/);
  assert.doesNotMatch(unwritable, /variable level or model term/);
  assert.match(runFailure('research_pipeline_progressive_compile_failed'), /variable level or model term/);

  // A family request refused before the model call names the study change
  // that lifts it; any other family-template stop says only what is true of
  // every such stop, with its code, and names no plan step.
  const lateCohort = { code: 'progressive_family_spec_cohort_eligibility_after_time_zero' };
  const late = runFailure('research_pipeline_progressive_compile_failed', lateCohort);
  assert.match(late, /before the model was called/);
  assert.match(late, /move time zero later/);
  assert.doesNotMatch(late, /variable level or model term/);
  const runFailureZh = guided.require('errorText').create({ tr: (en, zh) => zh, staticPreview: () => false }).runFailureText;
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', { code: 'progressive_family_result_contract_unwritable' }), /固定时间内发生/);
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', lateCohort), /把时间零点后移/);
  const familyStop = runFailure('research_pipeline_progressive_compile_failed', { code: 'progressive_family_spec_accepted_baseline_grouping_unsupported' });
  assert.match(familyStop, /progressive_family_spec_accepted_baseline_grouping_unsupported/);
  assert.match(familyStop, /no analysis was run/);
  assert.doesNotMatch(familyStop, /variable level or model term/);
  // Any other candidate-plan stop says only what is true of every such stop,
  // with its code; the compile sentence is true only of a variable or level
  // the data cannot resolve.
  const planStop = runFailure('research_pipeline_progressive_compile_failed', { code: 'progressive_step_invalid' });
  assert.match(planStop, /EasyICU check of the candidate plan \(code: progressive_step_invalid\)/);
  assert.match(planStop, /no analysis was run/);
  assert.doesNotMatch(planStop, /variable level or model term/);
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', { code: 'progressive_step_invalid' }), /候选计划没有通过 EasyICU 的检查/);
  ['progressive_unknown_variable', 'progressive_model_levels_unavailable'].forEach(code => {
    assert.match(runFailure('research_pipeline_progressive_compile_failed', { code }), /variable level or model term/);
  });
  // Two owners of one result, and a reused result no earlier step produces,
  // name their own cause.
  const owners = runFailure('research_pipeline_progressive_compile_failed', { code: 'progressive_product_has_multiple_owners' });
  assert.match(owners, /Two steps of the candidate plan produce the same result/);
  assert.doesNotMatch(owners, /variable level or model term/);
  assert.match(runFailure('research_pipeline_progressive_compile_failed', { code: 'progressive_outline_replay_producer_absent' }), /no earlier step runs/);
  assert.match(runFailure('research_pipeline_progressive_compile_failed', { code: 'progressive_outline_trajectory_comparison_unowned' }), /trajectory groups/);

  // A prediction stop names the study change that lifts it, not the generic
  // family-template sentence.
  const predictionStops = {
    progressive_family_spec_prediction_risk_set_unavailable: /Declare the study's analysis as a prediction model/,
    progressive_family_spec_icu_stay_unit_unread: /Prepare the export again with the unit recorded/,
  };
  Object.entries(predictionStops).forEach(([code, remedy]) => {
    const text = runFailure('research_pipeline_progressive_compile_failed', { code });
    assert.match(text, remedy);
    assert.match(text, /before the model was called/);
    assert.doesNotMatch(text, /EasyICU check of the study's template plan/);
  });
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', { code: 'progressive_family_spec_prediction_risk_set_unavailable' }), /声明为预测模型/);

  // A failed-closed run says which check it did not pass. A stop its executor
  // named carries its own cause, and the remedy follows that cause.
  const failedClosed = 'research_agent_pipeline_failed_closed';
  const intervalStop = 'continuous_survival_interval_result_not_estimable';
  assert.match(runFailure(failedClosed), /did not pass EasyICU's checks/);
  const noEvents = runFailure(failedClosed, { code: intervalStop, cause: 'interval_without_event' });
  assert.match(noEvents, /could not be estimated: an interval had no events/);
  assert.match(noEvents, /Choose interval cut points with events in every interval/);
  assert.doesNotMatch(noEvents, /did not pass EasyICU's checks/);
  const notConverged = runFailure(failedClosed, { code: intervalStop, cause: 'did_not_converge' });
  assert.match(notConverged, /the interval model did not converge/);
  assert.match(notConverged, /Adjust for fewer variables or merge sparse categories/);
  const unnamed = runFailure(failedClosed, { code: intervalStop, cause: 'unregistered_cause' });
  assert.match(unnamed, /could not be estimated\. The run has no primary result/);
  assert.match(unnamed, /follow-up intervals or its adjustment/);
  assert.match(runFailureZh(failedClosed, { code: intervalStop, cause: 'follow_up_ends_by_final_cutpoint' }), /随访在最后一个切点之前就结束了/);
  const oneValue = runFailure(failedClosed, { code: 'continuous_survival_exposure_has_one_value' });
  assert.match(oneValue, /had the same value of the continuous exposure/);
  assert.match(oneValue, /Choose an exposure, summary or window whose values differ between stays/);
  assert.doesNotMatch(oneValue, /did not pass EasyICU's checks/);
  assert.match(runFailureZh(failedClosed, { code: 'continuous_survival_exposure_has_one_value' }), /取值都相同/);
  const refitFailed = runFailure(failedClosed, { code: 'trajectory_stability_refit_failed' });
  assert.match(refitFailed, /needs every planned refit to succeed, and a refit could not be completed/);
  assert.match(refitFailed, /Revise the plan, for example to consider fewer classes/);
  assert.doesNotMatch(refitFailed, /did not pass EasyICU's checks/);
  assert.match(runFailureZh(failedClosed, { code: 'trajectory_stability_refit_failed' }), /每次计划的重拟合都成功/);
  // The binary landmark suite stops under the same rule, with the same remedy.
  const binaryStop = 'landmark_survival_interval_result_not_estimable';
  const binaryNoEvents = runFailure(failedClosed, { code: binaryStop, cause: 'interval_without_event' });
  assert.match(binaryNoEvents, /could not be estimated: an interval had no events/);
  assert.match(binaryNoEvents, /Choose interval cut points with events in every interval/);
  assert.doesNotMatch(binaryNoEvents, /did not pass EasyICU's checks/);
  assert.match(runFailureZh(failedClosed, { code: binaryStop, cause: 'did_not_converge' }), /区间模型没有收敛/);
  // The target trial emulation names its stop, and a weight model or its
  // bootstrap why; the remedy revises the confirmed design.
  const weightModel = runFailure(failedClosed, { code: 'target_trial_weight_model_not_estimable', cause: 'separation' });
  assert.match(weightModel, /weights could not be estimated: the model predicted almost perfectly when stays started the treatment or left the ICU/);
  assert.match(weightModel, /Revise the target trial's design, for example with merged sparse categories or fewer covariates, confirm it again/);
  assert.doesNotMatch(weightModel, /did not pass EasyICU's checks/);
  assert.match(runFailure(failedClosed, { code: 'target_trial_weight_model_not_estimable', cause: 'singular_design' }), /covariates repeated each other's information/);
  for (const cause of ['unregistered_cause', 'constructor', 'toString']) {
    const unnamedWeights = runFailure(failedClosed, { code: 'target_trial_weight_model_not_estimable', cause });
    assert.match(unnamedWeights, /could not be estimated, so the run has no primary result/);
    assert.doesNotMatch(unnamedWeights, /function|native code/);
  }
  // Dropping a confounder is no remedy for positivity or extreme weights.
  for (const code of ['target_trial_positivity_violated', 'target_trial_weights_extreme']) {
    assert.doesNotMatch(runFailure(failedClosed, { code }), /fewer covariates/);
    assert.doesNotMatch(runFailureZh(failedClosed, { code }), /减少协变量/);
    assert.match(runFailure(failedClosed, { code }), /covariates that affect starting the treatment but not the outcome/);
  }
  // Every stop the suite names has its own sentence in both languages.
  for (const code of [
    'target_trial_sample_insufficient', 'target_trial_events_insufficient',
    'target_trial_strategy_unobserved', 'target_trial_positivity_violated',
    'target_trial_weights_extreme', 'target_trial_icu_exit_excessive',
    'target_trial_weight_model_not_estimable', 'target_trial_bootstrap_unstable',
  ]) {
    assert.doesNotMatch(runFailure(failedClosed, { code }), /did not pass EasyICU's checks/);
    assert.match(runFailure(failedClosed, { code }), /the run has no primary result\. Revise the target trial's design/);
    assert.match(runFailureZh(failedClosed, { code }), /这次运行没有主结果。请修订目标试验的设计/);
  }
  assert.match(runFailure(failedClosed, { code: 'target_trial_bootstrap_unstable', cause: 'estimate_outside_interval' }), /outside its own bootstrap interval/);
  // A causal plan stops before the Planner without a confirmed target trial,
  // or when the approved trial no longer compiles on the run's data.
  const compileFailed = 'research_pipeline_progressive_compile_failed';
  assert.match(runFailure(compileFailed, { code: 'tte_trial_not_confirmed' }), /no confirmed target trial, so planning stopped before any model was called/);
  assert.match(runFailureZh(compileFailed, { code: 'tte_trial_not_confirmed' }), /还没有已确认的目标试验/);
  const drifted = runFailure(compileFailed, { code: 'target_trial_compile_drifted' });
  assert.match(drifted, /differs from the version approved on its confirmation card/);
  assert.match(drifted, /approve it again, then generate the plan/);
  assert.match(runFailureZh(compileFailed, { code: 'target_trial_compile_drifted' }), /请在确认卡片上复核目标试验并重新批准/);
  // The stops at run start name what to prepare or change.
  assert.match(runFailure('target_trial_materialization_mismatch'), /Prepare the data again for the target trial's windows/);
  assert.match(runFailure('target_trial_family_mismatch'), /not declared as causal inference/);
  assert.match(runFailure('target_trial_configuration_invalid'), /Set up the target trial again in the conversation/);
  assert.match(runFailure('research_pipeline_conflicting_sealed_suites'), /a run can follow only one of them/);
  assert.match(runFailureZh('target_trial_materialization_mismatch'), /请按目标试验的时间窗重新准备数据/);
  assert.match(runFailure(failedClosed, { code: 'target_trial_icu_exit_excessive' }), /left the ICU within the grace period before starting the treatment/);
  assert.match(runFailureZh(failedClosed, { code: 'target_trial_positivity_violated' }), /重新确认后再生成计划/);
  assert.match(runFailureZh(failedClosed, { code: 'target_trial_bootstrap_unstable', cause: 'resamples_failed' }), /自助法重抽样无法估计/);
  const axisSentences = {
    execution_complete_not_satisfied: /An analysis step did not finish/,
    analysis_validated_not_satisfied: /automated validation did not pass/,
    evidence_complete_not_satisfied: /lack the evidence they rest on/,
    numeric_verified_not_satisfied: /could not be checked against the results/,
  };
  Object.entries(axisSentences).forEach(([code, sentence]) => {
    assert.match(runFailure(failedClosed, { code }), sentence);
  });
  // The Writer met an unavailable model service after every analysis step
  // finished: whichever retry limit ended it, the results are kept.
  for (const cause of ['transport_retry_attempts_exhausted', 'transport_retry_window_exhausted', 'transport_retry_wall_clock_exhausted']) {
    const writerStop = runFailure(failedClosed, { code: 'writer_provider_transport_unavailable', cause });
    assert.match(writerStop, /model service was unavailable while the manuscript was being drafted; the analysis finished/);
    assert.doesNotMatch(writerStop, /did not pass EasyICU's checks/);
  }
  assert.match(runFailureZh(failedClosed, { code: 'writer_provider_transport_unavailable' }), /写稿时模型服务不可用；分析已完成，结果已保留。可以重试这次运行重新写稿/);

  // A runner image built from other EasyICU source needs a rebuild; telling the
  // researcher to start Docker would send them to a runtime that is already up.
  const stale = live({ code: 'research_pipeline_runner_image_mismatch' });
  assert.match(stale, /Rebuild it from the current commit/);
  assert.doesNotMatch(stale, /Docker Desktop|colima/);
  assert.match(runFailure('research_pipeline_runner_image_mismatch'), /runner image did not match/);

  // An approved run that failed for good is not offered for approval again;
  // one whose review stayed open is.
  const unresumable = runFailure('research_pipeline_approved_run_failed');
  assert.match(unresumable, /cannot resume/);
  assert.doesNotMatch(unresumable, /approved again/);
  assert.match(runFailure('research_pipeline_review_resume_failed'), /can be approved again/);

  // A launch refusing the bound export names what comes first, not its code
  // (the transport puts the bare code in the message).
  const exportCohort = live({
    code: 'research_pipeline_export_cohort_mismatch',
    message: 'research_pipeline_export_cohort_mismatch',
  });
  assert.match(exportCohort, /Extract this study’s cohort, or choose another data source/);
  assert.doesNotMatch(exportCohort, /research_pipeline_export_cohort_mismatch/);
  assert.match(live({ code: 'research_pipeline_export_cohort_invalid' }), /Check the study’s cohort/);

  // Fallback branch: raw transport text verbatim.
  assert.equal(live({ message: 'Failed to fetch' }), 'Failed to fetch');
  assert.equal(live({ message: 'boom' }), 'boom');
  assert.equal(live({ code: 'custom_code' }), 'custom_code');
  assert.equal(live('raw-oops'), 'raw-oops');

  // Harness keeps the legacy alias for the same owner.
  let harnessPath = path.join(dir, '..', '..', '..', 'tests', 'js', 'guided_pi_module_harness.cjs');
  if (!fs.existsSync(harnessPath)) {
    harnessPath = path.join(__dirname, 'guided_pi_module_harness.cjs');
  }
  const harness = fs.readFileSync(harnessPath, 'utf8');
  assert.match(harness, /EU_GUIDED_PI_ERROR_TEXT/);
  assert.match(harness, /['"]errorText['"]/);
})();

// The workspace panel reads a failed-closed run as a failure, and both its
// failure line and the run's row in the run record ask for the run's own cause.
(function testAsideFailedClosedRun() {
  const dir = path.dirname(path.resolve(process.argv[2]));
  const asideSource = fs.readFileSync(path.join(dir, 'screens-guided-pi-aside.js'), 'utf8');
  const head = { innerHTML: '' };
  const body = { innerHTML: '', querySelector: () => null, querySelectorAll: () => [] };
  const context = { window: {}, document: { getElementById: id => (
    id === 'gdStudyAside' ? { querySelector: () => head } : id === 'gdAsideBody' ? body : null) } };
  vm.createContext(context);
  vm.runInContext(source, context);
  vm.runInContext(asideSource, context);
  const render = run => {
    context.window.EasyICU.guidedPi.require('aside').create({
      tr: en => en, esc: value => String(value ?? ''), iconHtml: () => '',
      projectId: () => 'project', displayProjectTitle: value => value,
      demoMode: () => false, shell: () => 'pi', project: () => ({}),
      workflow: () => ({ current_stage: 'analysis', runs: [run],
        stages: [{ id: 'analysis', status: 'blocked', reason_code: 'failed_pipeline_requires_fresh_plan' }] }),
      latestRun: () => run,
      runFailureText: (code, detail) => [code, detail && detail.code, detail && detail.cause].filter(Boolean).join(' / '),
    }).syncProjectWorkflowAside();
    return body.innerHTML;
  };
  const run = { run_id: 'run_closed', present: true, authoritative: true, run_type: 'full',
    run_status: 'blocked', gate_status: 'blocked', gate_reason_code: 'research_agent_pipeline_failed_closed',
    gate_detail_code: 'continuous_survival_interval_result_not_estimable',
    gate_detail_cause_code: 'interval_without_event' };
  const cause = 'research_agent_pipeline_failed_closed / continuous_survival_interval_result_not_estimable / interval_without_event';
  const closed = render(run);
  assert.ok(closed.includes(`<div class="si-s gpi-run-failure" title="run_closed">${cause}</div>`), closed);
  assert.ok(closed.includes(`<small class="gpi-run-cause">${cause}</small>`), closed);
  // A run waiting for its interpretation review has not failed.
  const waiting = render({ ...run, gate_reason_code: 'research_agent_pipeline_complete_human_interpretation_required',
    gate_detail_code: undefined, gate_detail_cause_code: undefined });
  assert.ok(!waiting.includes('gpi-run-failure'), waiting);
})();

process.stdout.write(JSON.stringify({ ok: true, fail_closed_cases: 5 }));
