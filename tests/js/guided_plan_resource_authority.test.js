'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function confirmation(review) {
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/static/js/screens-guided-pi-modules.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/static/js/screens-guided-pi-confirmation.js'), 'utf8'), context);
  return context.window.EasyICU.guidedPi.require('confirmation').create({
    workflow: () => ({ next_action_code: 'plan_scientific_changes_required', plan_review_summary: review }),
    session: () => ({ binding: { run_id: 'newer-failed-child' }, archived_child_jobs: [] }),
    tr: en => en, esc: String, iconHtml: () => '', resourceButton: () => '',
    sessionIsStale: () => false,
  }).workflowConfirmation();
}

function confirmationOwner(workflow, language = 'en') {
  const context = { window: {} };
  vm.createContext(context);
  for (const name of ['modules', 'confirmation']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('confirmation').create({
    workflow: () => workflow, session: () => ({}), busy: () => false,
    tr: (en, zh) => language === 'zh' ? zh : en, esc: value => String(value).replaceAll('<', '&lt;'), iconHtml: () => '',
    resourceButton: ref => `[resource:${ref.run_id}:${ref.artifact}]`,
    sessionIsStale: () => false,
  });
}

function confirmationWithFailureText(workflow, session) {
  const context = { window: {} };
  vm.createContext(context);
  for (const name of ['modules', 'confirmation']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('confirmation').create({
    workflow: () => workflow, session: () => session, busy: () => false,
    tr: en => en, esc: String, iconHtml: () => '', resourceButton: () => '',
    sessionIsStale: () => false,
    // Echo which cause the card asked for: the code, its typed detail and
    // that detail's own cause.
    runFailureText: (code, detail) => [code, detail && detail.code, detail && detail.cause].filter(Boolean).join(' / '),
  }).workflowConfirmation();
}

test('the reviewed desktop profile is visible before approval in both languages', () => {
  for (const name of ['desktop_first_result_dev', 'desktop_first_result_live_pubmed_dev']) {
    for (const language of ['en', 'zh']) {
      const owner = confirmationOwner({
        next_action_code: 'operator_plan_approval_required',
        plan_review_summary: { run_id: 'desktop', submission_profile_ref: `${name}/20261023` },
      }, language);
      const profileLine = language === 'en'
        ? 'This run produces analysis tables and figures, not a manuscript.'
        : '本次只生成分析表格和图形，不写论文。';
      assert.ok(owner.workflowConfirmationHtml().includes(
        `<p class="gpi-confirmation-profile">${profileLine}</p>`));
    }
  }
  const docker = confirmationOwner({
    next_action_code: 'operator_plan_approval_required',
    plan_review_summary: { run_id: 'docker', submission_profile_ref: 'npj_dm/1.0.0' },
  });
  assert.doesNotMatch(docker.workflowConfirmationHtml(), /gpi-confirmation-profile/);
});

// The conversation's run-stop block for the same runs, with the same echo.
function runStopText(runs) {
  const context = { window: { EU_HTML: { esc: String } } };
  vm.createContext(context);
  for (const name of ['modules', 'run-stop']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('runStop').create({
    tr: en => en,
    runFailureText: (code, detail) => [code, detail && detail.code, detail && detail.cause].filter(Boolean).join(' / '),
  }).render({ savedRuns: runs });
}

test('plan evidence keeps the reviewed run when a later child fails before planning', () => {
  const card = confirmation({ run_id: 'reviewed-plan', remediation_buckets: {} });
  assert.equal(card.reviewResources.length, 3);
  for (const resource of card.reviewResources) assert.equal(resource.run_id, 'reviewed-plan');
});

test('a revision the launcher refuses for a non-plan blocker is not offered', () => {
  // Hosted pilot: the lactate-group association card offered a revision that
  // the launcher refused (plan_revision_owner_resolution_required), silently.
  const buckets = { agent_plan_revision: ['CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED'] };
  assert.equal(confirmation({ run_id: 'p', remediation_buckets: buckets }).nonApprovable, false);
  assert.equal(confirmation({
    run_id: 'p', remediation_buckets: buckets,
    automatic_revision_blockers: ['POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED'],
  }).nonApprovable, true);
});

test('a blocked revision explains the next step without promising an automatic repair', () => {
  for (const language of ['en', 'zh']) {
    const workflow = {
      next_action_code: 'plan_scientific_changes_required',
      plan_review_summary: {
        run_id: 'retained-candidate', authorization_questions: [],
        remediation_buckets: {
          agent_plan_revision: ['FIGURE_ROLE_COVERAGE_INCOMPLETE'],
          runtime_capability: ['REPEATED_STAY_IDENTITY_UNAVAILABLE'],
        },
        automatic_revision_blockers: ['REPEATED_STAY_IDENTITY_UNAVAILABLE'],
      },
      latest_attempt_failure: {
        run_id: 'failed-preparation', candidate_run_id: 'retained-candidate',
        reason: 'preparation_failed', checkpoint_resume_available: false,
      },
    };
    const owner = confirmationOwner(workflow, language);
    const before = JSON.stringify(workflow);
    const html = owner.workflowConfirmationHtml();
    assert.equal(owner.workflowConfirmation().nonApprovable, true);
    assert.doesNotMatch(html, /data-gpi-confirm-action|data-gpi-confirm-edit/);
    assert.doesNotMatch(html, /will revise the candidate plan|将在分析前修订候选计划/);
    assert.doesNotMatch(html, /use the action below to recover|请使用下方操作恢复/);
    assert.match(html, language === 'en'
      ? /Open the review details.*cannot supply missing data or independent evidence/
      : /审阅详情.*不能补足数据或独立证据/);
    assert.match(html, /retained-candidate:scientific_plan_review.json/);
    assert.match(html, /failed-preparation:source_run_manifest.json/);
    assert.equal(JSON.stringify(workflow), before);
  }
});

test('an actionable revision keeps its revision button and recovery guidance', () => {
  const workflow = {
    next_action_code: 'plan_scientific_changes_required',
    plan_review_summary: {
      run_id: 'retained-candidate', authorization_questions: [],
      remediation_buckets: { agent_plan_revision: ['FIGURE_ROLE_COVERAGE_INCOMPLETE'] },
      automatic_revision_blockers: [],
    },
    latest_attempt_failure: {
      run_id: 'failed-preparation', candidate_run_id: 'retained-candidate',
      reason: 'preparation_failed', checkpoint_resume_available: false,
    },
  };
  for (const language of ['en', 'zh']) {
    const owner = confirmationOwner(workflow, language);
    const html = owner.workflowConfirmationHtml();
    assert.equal(owner.workflowConfirmation().nonApprovable, false);
    assert.match(html, /data-gpi-confirm-action/);
    assert.match(html, language === 'en' ? /Follow the next action below/ : /请按下方提示处理/);
    assert.doesNotMatch(html, /contact support|请联系支持人员/);
  }
});

test('a review without an executable repair explains how to resolve its evidence gap', () => {
  for (const language of ['en', 'zh']) {
    const owner = confirmationOwner({
      next_action_code: 'plan_scientific_changes_required',
      plan_review_summary: {
        run_id: 'evidence-review', authorization_questions: [],
        remediation_buckets: { external_evidence: ['LITERATURE_SUPPORT_INSUFFICIENT'] },
      },
    }, language);
    const html = owner.workflowConfirmationHtml();
    assert.equal(owner.workflowConfirmation().nonApprovable, true);
    assert.doesNotMatch(html, /data-gpi-confirm-action|will revise the candidate plan|将在分析前修订候选计划/);
    assert.match(html, language === 'en' ? /independent evidence or review/ : /补充独立证据或人工审阅/);
    assert.match(html, /evidence-review:scientific_plan_review.json/);
  }
});

test('a user decision keeps its answer action after a preparation failure', () => {
  for (const language of ['en', 'zh']) {
    const owner = confirmationOwner({
      next_action_code: 'plan_scientific_changes_required',
      plan_review_summary: {
        run_id: 'endpoint-review',
        authorization_questions: [{ code: 'OUTCOME_DEFINITION_UNRESOLVED' }],
        remediation_buckets: { agent_plan_revision: ['FIGURE_ROLE_COVERAGE_INCOMPLETE'] },
      },
      latest_attempt_failure: {
        run_id: 'failed-preparation', candidate_run_id: 'endpoint-review',
        reason: 'preparation_failed', checkpoint_resume_available: false,
      },
    }, language);
    const html = owner.workflowConfirmationHtml();
    assert.match(html, /data-gpi-confirm-edit/);
    assert.match(html, language === 'en' ? /Follow the next action below/ : /请按下方提示处理/);
    assert.doesNotMatch(html, /contact support|请联系支持人员/);
  }
});

test('a refused scope expansion tells the user how to request a revision', () => {
  const context = { window: { EU_HTML: { esc: String } } };
  vm.createContext(context);
  for (const name of ['modules', 'error-text']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  for (const language of ['en', 'zh']) {
    const errors = context.window.EasyICU.guidedPi.require('errorText').create({
      tr: (en, zh) => language === 'zh' ? zh : en, staticPreview: () => false,
    });
    const code = 'revision_added_unrequested_analysis';
    const text = errors.runFailureText(code);
    assert.equal(errors.errorText({ code }), text);
    assert.match(text, language === 'en' ? /previous plan was kept/ : /已保留修订前的计划/);
    assert.match(text, language === 'en' ? /review.*generate a revised candidate/i : /审阅.*生成修订版候选计划/);
    assert.match(text, language === 'en' ? /if that button is shown/ : /如有.*若没有/);
    assert.match(text, language === 'en' ? /failure receipt.*support/ : /失败回执.*支持人员/);
  }
});

test('missing plan authority never falls back to the latest session run', () => {
  const card = confirmation({ remediation_buckets: {} });
  for (const resource of card.reviewResources) assert.equal(resource.run_id, '');
});

test('a preserved candidate and its failed preparation have separate evidence links', () => {
  const workflow = {
    next_action_code: 'planner_checkpoint_resume_available',
    plan_review_summary: { run_id: 'reviewed-plan' },
    latest_attempt_failure: {
      run_id: 'failed-preparation', candidate_run_id: 'reviewed-plan',
      reason: 'provider_unavailable', checkpoint_resume_available: true,
      raw_error: '<script>private response</script>',
    },
  };
  const owner = confirmationOwner(workflow);
  const html = owner.workflowConfirmationHtml();
  assert.match(html, /gpi-confirmation-failure/);
  assert.match(html, /model service did not respond/i);
  assert.match(html, /Transient timeouts are retried once automatically/);
  assert.match(html, /failed-preparation:source_run_manifest.json/);
  assert.match(html, /reviewed-plan:agent_plan.json/);
  assert.doesNotMatch(html, /private response|failed-preparation:agent_plan.json/);
  assert.equal(owner.workflowConfirmation().grants.includes('extract'), false);
});

test('a plan that can no longer resume is not blamed on a changed study', () => {
  const card = code => confirmationOwner({
    next_action_code: code, plan_review_summary: { run_id: 'old-plan' },
  }).workflowConfirmation();
  const lost = card('plan_review_not_resumable');
  assert.match(lost.title, /can no longer be approved or resumed/);
  assert.doesNotMatch(lost.title, /study changed/);
  assert.match(card('plan_configuration_superseded').title, /The study changed/);
});

test('an unavailable checkpoint offers preparation of the same retained candidate', () => {
  const workflow = {
    next_action_code: 'plan_execution_upgrade_required',
    plan_review_summary: { run_id: 'retained-candidate' },
    latest_attempt_failure: {
      run_id: 'failed-preparation', candidate_run_id: 'retained-candidate',
      reason: 'provider_unavailable', checkpoint_resume_available: false,
    },
  };
  const owner = confirmationOwner(workflow);
  const card = owner.workflowConfirmation();
  assert.match(card.approve, /Prepare data again/);
  assert.match(card.note, /Your candidate is saved.*Prepare the data again/);
  assert.doesNotMatch(owner.workflowConfirmationHtml(), /data-gpi-confirm-edit|checkpoint/);
  assert.doesNotMatch(card.message, /approve.*analysis/i);
  assert.ok(card.grants.includes('extract'));
  for (const resource of card.reviewResources) assert.equal(resource.run_id, 'retained-candidate');
  const chinese = confirmationOwner(workflow, 'zh');
  assert.equal(chinese.workflowConfirmation().approve, '重新准备数据');
  assert.match(chinese.workflowConfirmation().note, /候选方案已保留.*重新准备数据/);
  assert.doesNotMatch(chinese.workflowConfirmationHtml(), /检查点/);
});

test('the cause a failed run recorded is said once, by its run-stop block', () => {
  // The workflow's action comes from the authoritative failed run; its typed
  // detail names a cause that the job's bare code would generalize. The
  // conversation's run-stop block says it right above the card, so the card
  // does not repeat it.
  const failedJob = { kind: 'agent-run', status: 'failed', error_code: 'research_pipeline_progressive_compile_failed' };
  const card = workflow => confirmationWithFailureText(workflow, { archived_child_jobs: [failedJob] });
  const failedRuns = [
    { run_id: 'other', authoritative: false, run_status: 'failed',
      gate_reason_code: 'research_pipeline_execution_failed' },
    { run_id: 'failed-plan', run_ref: 'r-failed-plan', authoritative: true, run_status: 'failed', gate_status: 'blocked',
      gate_reason_code: 'research_pipeline_progressive_compile_failed',
      gate_detail_code: 'progressive_family_result_contract_unwritable' },
  ];
  const recorded = card({ next_action_code: 'failed_pipeline_requires_fresh_plan', runs: failedRuns });
  assert.equal(recorded.reason, '');
  assert.match(runStopText(failedRuns), /research_pipeline_progressive_compile_failed \/ progressive_family_result_contract_unwritable/);
  // A run awaiting review is not a failure and has no run-stop block; the
  // job's code is all there is, so the card says it.
  const pending = card({
    next_action_code: 'failed_pipeline_requires_fresh_plan',
    runs: [{ run_id: 'paused', authoritative: true, run_status: 'human_review_pending', gate_reason_code: 'human_plan_review_required' }],
  });
  assert.equal(pending.reason, 'research_pipeline_progressive_compile_failed');
});

test('a failed-closed run is a failure whose run-stop block names its cause', () => {
  // A run that did not pass a check after it started records which check and
  // why; the bare gate code alone would name no cause.
  const card = workflow => confirmationWithFailureText(workflow, { archived_child_jobs: [] });
  const closedRuns = [{ run_id: 'closed', run_ref: 'r-closed', authoritative: true, run_status: 'blocked', gate_status: 'blocked',
    gate_reason_code: 'research_agent_pipeline_failed_closed',
    gate_detail_code: 'continuous_survival_interval_result_not_estimable',
    gate_detail_cause_code: 'interval_without_event' }];
  const closed = card({ next_action_code: 'failed_pipeline_requires_fresh_plan', runs: closedRuns });
  assert.equal(closed.reason, '');
  assert.match(runStopText(closedRuns), /research_agent_pipeline_failed_closed \/ continuous_survival_interval_result_not_estimable \/ interval_without_event/);
  // Another research_agent_pipeline_* gate is not read as a failure.
  const other = card({
    next_action_code: 'failed_pipeline_requires_fresh_plan',
    runs: [{ run_id: 'done', authoritative: true, run_status: 'blocked', gate_status: 'blocked',
      gate_reason_code: 'research_agent_pipeline_complete_human_interpretation_required' }],
  });
  assert.equal(other.reason, '');
});

test('failure notice cannot drift onto another reviewed candidate', () => {
  const owner = confirmationOwner({
    next_action_code: 'plan_execution_upgrade_required',
    plan_review_summary: { run_id: 'new-plan' },
    latest_attempt_failure: { run_id: 'failed', candidate_run_id: 'old-plan', reason: 'provider_unavailable' },
  });
  assert.doesNotMatch(owner.workflowConfirmationHtml(), /gpi-confirmation-failure/);
});
