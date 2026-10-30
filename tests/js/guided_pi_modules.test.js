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

  // Every launch-resume refusal is readable in both the toast and failure card.
  const resumeSource = fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/research_launch_resume.py'), 'utf8');
  const resumeCodes = new Set(resumeSource.match(/research_pipeline_development_resume_[a-z_]+/g));
  resumeCodes.add('research_pipeline_development_resume_future_integrity_check');
  for (const language of ['en', 'zh']) {
    const errors = guided.require('errorText').create({
      tr: (en, zh) => language === 'zh' ? zh : en, staticPreview: () => false,
    });
    for (const code of resumeCodes) {
      const text = errors.errorText({ code, message: code });
      assert.ok(text.length > 30, `${language}: ${code}`);
      assert.doesNotMatch(text, /research_pipeline_development_resume_/);
      assert.equal(errors.runFailureText(code), text);
      assert.equal(errors.errorText(code), text);
      assert.equal(errors.errorText({ code: 'transport_error', message: code }), text);
      assert.match(text, language === 'zh' ? /计划|检查点/ : /plan|checkpoint/i);
    }
  }

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
  // The study's own outcome or exposure: no advice to drop it from the question.
  const coordinate = runFailure('data_foundation_blocked', { code: 'study_coordinate_unavailable', missing: ['icu_death'] });
  assert.match(coordinate, /outcome or exposure from this database: icu_death/);
  assert.match(coordinate, /before calling the model/);
  assert.doesNotMatch(coordinate, /drop/);
  // A concept the source provides but the menu lacks is EasyICU's gap.
  const menuGap = runFailure('data_foundation_blocked', { code: 'planning_menu_missing_supported_concept', missing: ['icu_death'] });
  assert.match(menuGap, /EasyICU's own planning list is missing a variable it can read from this database: icu_death/);
  assert.match(menuGap, /not in your data/);
  assert.doesNotMatch(menuGap, /another database|Choose a database|drop/);
  const unknownName = runFailure('data_foundation_blocked', { code: 'study_coordinate_unrecognized', missing: ['icu_mortality_72h'] });
  assert.match(unknownName, /did not recognise the name it gave this study's outcome or exposure: icu_mortality_72h/);
  assert.doesNotMatch(unknownName, /another database|Choose a database|drop/);
  assert.equal(runFailure('data_foundation_blocked', { code: 'unknown_detail_code' }), runFailure('data_foundation_blocked'));

  // A typed compile stop names its own cause; the generic compile sentence
  // would name one that did not happen.
  const unwritable = runFailure('research_pipeline_progressive_compile_failed', { code: 'progressive_family_result_contract_unwritable' });
  assert.match(unwritable, /this survival question needs/);
  assert.match(unwritable, /prepared input and design/);
  assert.match(unwritable, /confirmation/);
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
  const errorTextZh = guided.require('errorText').create({ tr: (en, zh) => zh, staticPreview: () => false });
  const runFailureZh = errorTextZh.runFailureText;
  const missingFollowup = { code: 'progressive_family_result_contract_unwritable',
    cause: 'required_concepts_unavailable', missing: ['followup_days_90d'] };
  assert.match(runFailure('research_pipeline_progressive_compile_failed', missingFollowup), /prepared data lack the event or censoring time/);
  assert.match(runFailure('research_pipeline_progressive_compile_failed', missingFollowup), /followup_days_90d/);
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', missingFollowup), /缺少.*事件或删失时间/);
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', missingFollowup), /followup_days_90d/);
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', { code: missingFollowup.code }), /确认/);
  assert.equal(runFailure('research_pipeline_progressive_compile_failed', { ...missingFollowup, cause: 'unknown_cause' }), unwritable);
  assert.equal(runFailure('research_pipeline_progressive_compile_failed', { ...missingFollowup, missing: [] }), unwritable);
  assert.doesNotMatch(runFailure('research_pipeline_progressive_compile_failed', { ...missingFollowup, code: 'progressive_unknown_literature_source' }), /censoring time/);
  const desktopStop = 'research_pipeline_plan_step_needs_sandbox';
  const desktopStopEn = 'One step in this plan needs code written by the model during the run, and the desktop app does not run such code. Remove or change that step, then approve the plan. No analysis has been run.';
  const desktopStopZh = '计划里有一步要靠模型临时写代码才能完成，桌面版不运行这类代码。请删除或修改这一步，再批准计划。尚未运行任何分析。';
  assert.equal(runFailure(desktopStop), desktopStopEn);
  assert.equal(live({ code: desktopStop }), desktopStopEn);
  assert.equal(runFailureZh(desktopStop), desktopStopZh);
  assert.equal(errorTextZh.errorText({ code: desktopStop }), desktopStopZh);
  // Exhausted message budgets preserve completed work and close this turn.
  const modelErrorZh = errorTextZh.modelErrorText;
  for (const code of ['pi_shell_message_provider_call_budget_exhausted', 'pi_shell_message_cost_budget_exhausted']) {
    assert.match(modelErrorZh(code, false), /已保留/);
    assert.match(modelErrorZh(code, false), /本轮无法继续/);
    assert.doesNotMatch(modelErrorZh(code, false), /发送“继续”|新建|重试/);
  }
  assert.match(modelErrorZh('pi_shell_session_cost_budget_exhausted', false), /新建后续对话/);
  const splineCause = { code: 'landmark_spline_binary_sensitivity_unsupported' };
  const splineFailure = runFailure('research_pipeline_scientific_authority_refused', splineCause);
  assert.match(splineFailure, /two-group comparison/);
  assert.match(splineFailure, /own grouped model/);
  assert.doesNotMatch(splineFailure, /landmark_spline_binary_sensitivity_unsupported/);
  const splineFailureZh = runFailureZh('research_pipeline_scientific_authority_refused', splineCause);
  assert.match(splineFailureZh, /两组比较/);
  assert.match(splineFailureZh, /单独按分组建模/);
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
  // Each adjustment refusal reaches a planning stop under its progressive_ code.
  const adjustmentStops = {
    progressive_family_spec_stated_adjustment_unavailable: [
      'The question asks to adjust for a variable that the extracted data does not provide, so planning stopped and no analysis was run. Choose data that includes it, or remove it from the adjustment variables, then generate the plan again.',
      '问题要求调整的某个变量在提取的数据里没有对应的列，因此规划已停止，没有运行分析。请选择包含该变量的数据，或把它从调整变量中去掉，然后重新生成计划。',
    ],
    progressive_family_spec_exact_roster_mismatch: [
      "The plan's adjustment variables differ from the ones named in the question, so planning stopped and no analysis was run. Generate the plan again.",
      '计划中的调整变量与问题里写明的不一致，因此规划已停止，没有运行分析。请重新生成计划。',
    ],
    progressive_family_spec_covariate_timing_unproven: [
      'An adjustment variable is not known by the landmark time, so it cannot be used to adjust this analysis. Planning stopped and no analysis was run. Give that variable a time window that ends before the landmark, or remove it, then generate the plan again.',
      '有一个调整变量在 landmark 时间点之前还无法确定，不能用于这项分析的调整。规划已停止，没有运行分析。请为该变量设定在 landmark 之前结束的时间窗，或将其去掉，然后重新生成计划。',
    ],
  };
  for (const [code, [en, zh]] of Object.entries(adjustmentStops)) {
    assert.equal(runFailure('research_pipeline_progressive_compile_failed', { code }), en);
    assert.equal(runFailureZh('research_pipeline_progressive_compile_failed', { code }), zh);
  }
  for (const cause of ['stated_adjustment_name_unresolved', 'stated_adjustment_source_unavailable',
    'stated_adjustment_column_unavailable', 'stated_adjustment_column_ambiguous']) {
    const detail = { code: 'progressive_family_spec_stated_adjustment_unavailable',
      cause, statedVariable: '试验量表甲' };
    const en = runFailure('research_pipeline_progressive_compile_failed', detail);
    const zh = runFailureZh('research_pipeline_progressive_compile_failed', detail);
    assert.match(en, /试验量表甲/);
    assert.match(zh, /试验量表甲/);
    assert.match(en, /remove/);
    assert.match(zh, /去掉/);
    if (cause === 'stated_adjustment_name_unresolved') {
      assert.match(en, /could not identify/);
      assert.match(zh, /无法识别/);
    } else {
      assert.match(zh, /数据/);
    }
  }
  // A fixed study roster that leaves out a stated variable names it; the plan is not blamed.
  const omitted = { code: 'progressive_family_spec_exact_roster_mismatch',
    cause: 'stated_adjustment_outside_exact_roster', statedVariable: '性别' };
  assert.equal(runFailure('research_pipeline_progressive_compile_failed', omitted),
    "The study's fixed adjustment variables leave out “性别”, which the question asks to adjust for. Add it to the adjustment variables, or remove it from the question. Planning stopped and no analysis was run. Generate the plan again after that change.");
  assert.equal(runFailureZh('research_pipeline_progressive_compile_failed', omitted),
    '研究设置中固定的调整变量没有包含问题要求调整的「性别」。请把它加入调整变量，或在问题中去掉它。规划已停止，没有运行分析。修改后请重新生成计划。');
  [
    ['progressive_unknown_variable', /variable outside the study's available inputs/],
    ['progressive_model_levels_unavailable', /category levels that the data description has not established/],
  ].forEach(([code, cause]) => {
    const sentence = runFailure('research_pipeline_progressive_compile_failed', { code });
    assert.match(sentence, cause);
    assert.doesNotMatch(sentence, /progressive_/);
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
  // An exposure grouping stop names what the researcher changes, in both
  // languages, not the generic compile sentence.
  const groupingStops = {
    exposure_group_requires_extraction: [/Prepare data that include this value/, /请准备包含该数值的数据/],
    exposure_group_not_applied: [/the detail names the grouping and the reason/, /详情里写明了是哪一个分组以及原因/],
    exposure_grouping_unanswered: [/keep the run record for diagnosis/, /请保留运行记录供排查/],
    exposure_group_variable_unbound: [/This is an EasyICU fault/, /这是 EasyICU 的内部故障/],
    exposure_group_candidate_drift: [/runs only the grouping that was approved/, /EasyICU 只运行已批准的分组/],
    exposure_group_level_empty: [/does not drop or merge a group on its own/, /不会自行删除或合并分组/],
    progressive_family_spec_exposure_group_contrast_unavailable: [/reference or comparison group/, /参照组或对比组不在模板能提供的水平里/],
    progressive_exposure_group_contrast_mismatch: [/conflicting contrasts/, /不一致的比较方向/],
  };
  Object.entries(groupingStops).forEach(([code, [en, zh]]) => {
    const text = runFailure('research_pipeline_progressive_compile_failed', { code });
    assert.match(text, en);
    assert.match(text, /no analysis was run/);
    assert.doesNotMatch(text, /EasyICU check of the (?:candidate|study's template) plan/);
    assert.match(runFailureZh('research_pipeline_progressive_compile_failed', { code }), zh);
  });
  // A plan that reads a substance in the other form than the question says
  // stops with that cause, not the generic compile sentence.
  const substance = { code: 'progressive_question_substance_form_substituted' };
  assert.match(runFailure('research_pipeline_progressive_compile_failed', substance), /reads it in the other, so the plan would answer another question/);
  assert.doesNotMatch(runFailure('research_pipeline_progressive_compile_failed', substance), /EasyICU check of the candidate plan/);
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', substance), /主要暴露读成了另一种形式/);
  // A variable listed by the source with no value stops before planning; a
  // prediction step whose predictor or comparator holds no value stops the run.
  assert.match(runFailure('research_pipeline_required_concept_without_values'), /the source holds no value of it/);
  assert.match(runFailureZh('research_pipeline_required_concept_without_values'), /一个值都没有/);
  assert.match(runFailure('research_agent_pipeline_failed_closed', { code: 'prediction_predictor_unobserved' }), /revise the plan's predictors/);
  assert.match(runFailureZh('research_agent_pipeline_failed_closed', { code: 'prediction_predictor_unobserved' }), /开发集里一个值都没有/);
  assert.match(runFailure('research_agent_pipeline_failed_closed', { code: 'benchmark_comparator_unobserved' }), /nothing to compare/);
  assert.match(runFailureZh('research_agent_pipeline_failed_closed', { code: 'benchmark_comparator_unobserved' }), /验证集里一个值都没有/);
  // A static prediction planned without its family template names the template.
  const template = { code: 'progressive_static_prediction_requires_family_template' };
  assert.match(runFailure('research_pipeline_progressive_compile_failed', template), /planned from the prediction template/);
  assert.doesNotMatch(runFailure('research_pipeline_progressive_compile_failed', template), /EasyICU check of the candidate plan/);
  assert.match(runFailureZh('research_pipeline_progressive_compile_failed', template), /要用预测模板规划.*这份计划没有用模板/);

  // A failed-closed run says which check it did not pass. A stop its executor
  // named carries its own cause, and the remedy follows that cause.
  const failedClosed = 'research_agent_pipeline_failed_closed';
  assert.equal(runFailure(failedClosed, { code: 'manuscript_checks_not_passed' }), 'The manuscript draft did not pass its automated checks.');
  const manuscriptReasons = {
    manuscript_checks_not_passed: ['The manuscript draft did not pass its automated checks.', '论文稿没有通过自动检查。'],
    evidence_complete_not_satisfied: ['Some statements in the manuscript draft are not linked to a result from this analysis.', '论文稿中有部分表述没有对应到本次分析的结果。'],
    numeric_verified_not_satisfied: ['Some numbers in the manuscript draft do not match the analysis results.', '论文稿中有部分数字与分析结果不一致。'],
    writer_provider_transport_unavailable: ['Every analysis step finished, but the model service could not be reached while the manuscript draft was being written. Try again later.', '所有分析步骤都已完成，但撰写论文稿时一直无法连接模型服务。请稍后重试。'],
  };
  for (const language of ['en', 'zh']) {
    const errors = guided.require('errorText').create({
      tr: (en, zh) => language === 'zh' ? zh : en, staticPreview: () => false,
    });
    // One notice: the analysis status once, then only a distinct manuscript reason.
    const lead = language === 'zh' ? '分析已完成。论文草稿仍需修改。'
      : 'Analysis complete. The manuscript draft still needs revision.';
    const distinct = {
      evidence_complete_not_satisfied: manuscriptReasons.evidence_complete_not_satisfied,
      numeric_verified_not_satisfied: manuscriptReasons.numeric_verified_not_satisfied,
      writer_provider_transport_unavailable: ['The model service was unavailable while writing the draft. Try again later.', '撰写草稿时无法连接模型服务，请稍后重试。'],
    };
    for (const [code, translations] of Object.entries(manuscriptReasons)) {
      const sentence = translations[language === 'zh' ? 1 : 0];
      assert.equal(errors.runFailureText(failedClosed, { code }), sentence);
      const notice = errors.manuscriptNotPassedText(code);
      const reason = distinct[code] ? distinct[code][language === 'zh' ? 1 : 0] : '';
      assert.equal(notice, lead + (reason ? ' ' + reason : ''));
      assert.doesNotMatch(notice, /TODO\(lead copy\)|[a-z]+_[a-z_]+|[\[\]]/);
    }
    for (const code of [undefined, null, '', 'unknown_manuscript_reason', 'analysis_validated_not_satisfied', 'toString']) {
      assert.equal(errors.manuscriptNotPassedText(code), lead);
    }
  }
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
    evidence_complete_not_satisfied: /Some statements in the manuscript draft are not linked to a result from this analysis/,
    numeric_verified_not_satisfied: /Some numbers in the manuscript draft do not match the analysis results/,
  };
  Object.entries(axisSentences).forEach(([code, sentence]) => {
    assert.match(runFailure(failedClosed, { code }), sentence);
  });
  // The Writer met an unavailable model service after every analysis step
  // finished: whichever retry limit ended it, the results are kept.
  for (const cause of ['transport_retry_attempts_exhausted', 'transport_retry_window_exhausted', 'transport_retry_wall_clock_exhausted']) {
    const writerStop = runFailure(failedClosed, { code: 'writer_provider_transport_unavailable', cause });
    assert.equal(writerStop, manuscriptReasons.writer_provider_transport_unavailable[0]);
    assert.doesNotMatch(writerStop, /did not pass EasyICU's checks/);
  }
  assert.equal(runFailureZh(failedClosed, { code: 'writer_provider_transport_unavailable' }), manuscriptReasons.writer_provider_transport_unavailable[1]);

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

  // Fallback branch: a server sentence in the reader's language stays as it
  // is; a code, a transport string or a bare word becomes a plain sentence
  // with the original in brackets for troubleshooting.
  assert.equal(live({ message: 'The selected folder is empty.' }), 'The selected folder is empty.');
  assert.equal(live({ message: 'Failed to fetch' }), 'EasyICU could not reach its server. Check the network, then try again.');
  assert.equal(live({ message: 'boom' }), 'This step did not complete (boom). If it keeps happening, keep this error message for troubleshooting.');
  assert.equal(live({ code: 'custom_code' }), 'This step did not complete (custom_code). If it keeps happening, keep this error message for troubleshooting.');
  assert.equal(live('raw-oops'), 'This step did not complete (raw-oops). If it keeps happening, keep this error message for troubleshooting.');
  assert.equal(live({ message: '/api/x -> HTTP 503', status: 503 }), 'The EasyICU server hit an error on this step (HTTP 503). Try again shortly; if it keeps happening, keep this error message for troubleshooting.');
  const zhLive = guided.require('errorText').create({ tr: (_en, zh) => zh, staticPreview: () => false }).errorText;
  assert.match(zhLive({ status: 500 }), /请保留错误信息以便排查/);
  assert.doesNotMatch(zhLive({ status: 500 }), /管理员/);
  guided.declare('hosted', { active: () => true, supportContact: () => 'Help desk' });
  assert.match(live({ status: 500 }), /contact the administrator \(Help desk\)/);
  assert.match(zhLive({ status: 500 }), /请联系管理员（Help desk）/);

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
  const render = (run, failureText) => {
    context.window.EasyICU.guidedPi.require('aside').create({
      tr: en => en, esc: value => String(value ?? ''), iconHtml: () => '',
      projectId: () => 'project', displayProjectTitle: value => value,
      demoMode: () => false, shell: () => 'pi', project: () => ({}),
      workflow: () => ({ current_stage: 'analysis', runs: [run],
        stages: [{ id: 'analysis', status: 'blocked', reason_code: 'failed_pipeline_requires_fresh_plan' }] }),
      latestRun: () => run,
      runFailureText: failureText || ((code, detail) => [code, detail && detail.code, detail && detail.cause].filter(Boolean).join(' / ')),
    }).syncProjectWorkflowAside();
    return body.innerHTML;
  };
  const run = { run_id: 'run_closed', present: true, authoritative: true, run_type: 'full',
    run_status: 'blocked', gate_status: 'blocked', gate_reason_code: 'research_agent_pipeline_failed_closed',
    gate_detail_code: 'continuous_survival_interval_result_not_estimable',
    gate_detail_cause_code: 'interval_without_event' };
  const cause = 'research_agent_pipeline_failed_closed / continuous_survival_interval_result_not_estimable / interval_without_event';
  const closed = render(run);
  assert.ok(closed.includes(`<div class="si-s gpi-run-failure" title="${cause}">${cause}</div>`), closed);
  assert.ok(closed.includes(`<small class="gpi-run-cause">${cause}</small>`), closed);
  context.window.EU_HTML = { esc: String };
  vm.runInContext(fs.readFileSync(path.join(dir, 'screens-guided-pi-error-text.js'), 'utf8'), context);
  const errors = context.window.EasyICU.guidedPi.require('errorText').create({ tr: (_en, zh) => zh, staticPreview: () => false });
  for (const detail of [{ gate_detail_mismatch_fields: ['dependence'] }, { gate_detail: { mismatch_fields: ['dependence'] } }]) {
    const failed = { ...run, gate_reason_code: 'research_pipeline_scientific_authority_refused',
      gate_detail_code: 'baseline_association_design_mismatch', ...detail };
    const sentence = errors.runFailureText(failed.gate_reason_code, { code: failed.gate_detail_code, mismatchFields: ['dependence'] });
    assert.ok(sentence.includes('患者内多次 ICU 入住的相关性处理'));
    const html = render(failed, errors.runFailureText);
    assert.ok(html.includes(`<div class="si-s gpi-run-failure" title="${sentence}">${sentence}</div>`));
    assert.ok(html.includes(`<small class="gpi-run-cause">${sentence}</small>`));
  }
  // A run waiting for its interpretation review has not failed.
  const waiting = render({ ...run, gate_reason_code: 'research_agent_pipeline_complete_human_interpretation_required',
    gate_detail_code: undefined, gate_detail_cause_code: undefined });
  assert.ok(!waiting.includes('gpi-run-failure'), waiting);
})();

process.stdout.write(JSON.stringify({ ok: true, fail_closed_cases: 5 }));
