'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');

const js = path.resolve('src/easyicu/webserver/static/js');
global.window = {};
for (const name of ['modules', 'data-consent', 'cohort-eligibility', 'confirmation-data',
  'question-requirements', 'confirmation', 'target-trial-copy', 'target-trial', 'transcript',
  'session-view', 'next-actions', 'plan-actions']) {
  if (name === 'modules') require(path.join(js, 'html-escape.js'));
  require(path.join(js, `screens-guided-pi-${name}.js`));
}
const modules = window.EasyICU.guidedPi;
const { esc } = window.EU_HTML;
const stub = (overrides = {}) => new Proxy(overrides, { get: (item, key) => item[key] || (() => '') });

function render(language, workflow, sessionPatch, ideaExplored = true) {
  window.EU_LANG = language;
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const session = {
    session_id: 'session', binding: { study_context_id: 'study', study_revision: 7 },
    data_source_authorization: { status: 'confirmed', source: { label: 'Example ICU' } },
    ...sessionPatch,
  };
  const state = {
    session, workflow, workflowReceipts: [], sessions: [], draft: '',
    messages: [
      { role: 'user', text: 'Are two admission measures associated with recovery?' },
      ...(ideaExplored ? [{ role: 'activity', steps: [{ kind: 'tool', status: 'complete', toolName: 'easyicu_mine_ideas' }] }] : []),
      { role: 'assistant', complete: true, text: 'Setup recorded.\n\nNext step:\n- Unrelated suggestion' },
    ],
  };
  const host = { tr, esc, iconHtml: () => '', resourceButton: () => '', session: () => session,
    workflow: () => workflow, sessionIsStale: () => false, busy: () => false };
  const confirmation = modules.require('confirmation').create(host);
  const view = modules.require('sessionView').create({
    ...host, state, modules, dataConsent: modules.require('dataConsent'), header: stub(),
    studyWorkspace: stub({ messageView: row => ({ text: row.text }) }),
    activity: stub({ renderTimeline: (rows, renderRow) => rows.map(row => renderRow(row)).join('') }),
    runFiles: stub({ timeline: rows => rows }), resourceOwner: stub(), messageActions: stub({ render: () => ({}) }),
    transcript: modules.require('transcript'), runOutcome: stub(), aside: stub({ layoutOptions: () => ({}) }),
    cohortEligibility: modules.require('cohortEligibility').create(host),
    projectId: () => 'project', publicAssistantText: String, assistantTextHtml: esc,
    agentMode: () => 'research', projectTitle: () => 'Project', navigationSessionTitle: () => 'Session',
    workflowConfirmationHtml: confirmation.workflowConfirmationHtml,
    workflowConfirmation: confirmation.workflowConfirmation,
  });
  return { html: view.sessionPanel(), confirmation };
}

for (const language of ['en', 'zh']) {
  const steps = [
    [{ next_action_code: 'active_export_or_setup_required' }, {
      data_source_authorization: { status: 'pending', reason: 'project_source_confirmation_required',
        source: { label: 'Example ICU' } },
    }, /data-gpi-data-source-action="use_study_required_data"/],
    [{ next_action_code: 'study_primary_outcome_confirmation_required', unconfirmed_outcome: 'recovery' },
      {}, /data-gpi-confirm-action/],
    [{ next_action_code: 'cohort_eligibility_confirmation_required' }, {
      cohort_eligibility_selection: {
        present: true, blocker_code: 'cohort_eligibility_confirmation_required',
        primary_cohort_contract_sha256: 'selected',
        options: [{ id: 'no_eligibility_filter', primary_cohort_contract_sha256: 'selected',
          label: { en: 'All stays', zh: '全部入住' } }, { id: 'first_admission_only' }],
      },
    }, /data-gpi-cohort-option="no_eligibility_filter"/],
    [{ next_action_code: 'provider_ready_to_generate_plan' }, {}, /data-gpi-confirm-action/],
    [{ next_action_code: 'agent_plan_configuration_required' }, {}, /data-gpi-confirm-action/],
  ];
  for (const [workflow, session, action] of steps) {
    test(`${language}: ${workflow.next_action_code} is inline after exploration`, () => {
      const { html } = render(language, workflow, session);
      const log = html.split('data-gpi-log>')[1].split('<div class="gpi-compose">')[0];
      const primary = log.match(/<button\b[^>]*class="btn [^"]*\bprimary\b[^"]*"[^>]*>/g) || [];
      assert.equal(primary.length, 1, workflow.next_action_code);
      assert.match(primary[0], action);
      assert.doesNotMatch(log, /data-gpi-next-choice|data-gpi-aside-pending/);
    });
  }

  test(`${language}: one exposure arrangement is proposed on the current plan card`, async () => {
    const { html, confirmation } = render(language, {
      next_action_code: 'question_requirement_not_covered',
      plan_review_summary: {
        run_id: 'candidate',
        exposure_arrangement: {
          primary: 'measure_a', additional: ['measure_b'], default_option: 'primary_and_secondary',
          options: {
            primary_and_secondary: { en: 'Primary and secondary', zh: '主要和次要' },
            analyse_both: { en: 'All exposures', zh: '所有暴露' },
            only_primary: { en: 'Only primary', zh: '仅主要暴露' },
          },
        },

        question_requirements: { status: 'shown', judged_on_plan_under_review: true, items: [{
          id: 'r1', disposition: 'not_covered', source: 'host_verified', quote: 'Compare two measures',
          exposure_choice: { primary_exposure: 'measure_a', additional_exposures: ['measure_b'] },
        }] },
      },
    }, {}, false);
    assert.match(html, /data-gpi-confirm-action/);
    assert.equal((html.match(/data-gpi-exposure-arrangement=/g) || []).length, 2);
    const proposal = confirmation.workflowConfirmation();
    assert.equal(proposal.exposureArrangement.defaultOption, 'primary_and_secondary');
    assert.match(proposal.note, /measure_a.*measure_b/);
    const messages = [];
    const actions = modules.require('planActions').create({ sendText: (...args) => messages.push(args) });
    assert.deepEqual(messages, []);
    await actions.confirmWorkflow(proposal);
    assert.deepEqual(messages, [[proposal.exposureArrangement.options.primary_and_secondary,
      ['provider_run', 'literature'], 'confirm_exposure_arrangement', true, 'host_prompt']]);
  });
}

for (const language of ['en', 'zh']) {
  test(`${language}: target trial proposal, approval, and clarification are distinct inline states`, () => {
    const trial = modules.require('targetTrial');
    const workflow = {
      next_action_code: 'target_trial_statement_needed',
      target_trial_card: {
        state: 'proposal',
        proposal: { expected_revision: 7, digest: 'a'.repeat(64), missing: [], elements: [
          { key: 'treatment', value: 'sample therapy', source: 'question' },
          { key: 'eligibility_at_zero', value: '', source: 'design_choice' },
        ] },
      },
    };
    const session = { binding: { study_context_id: 'study', study_revision: 7 } };
    const shown = render(language, workflow, {});
    assert.match(shown.html, /sample therapy/);
    assert.match(shown.html, language === 'zh' ? /EasyICU 默认/ : /EasyICU default/);
    assert.match(shown.html, /data-gpi-confirm-action/);
    assert.ok(shown.confirmation.workflowConfirmation().compileTrialProposal);
    assert.equal(trial.proposalRequest(workflow, session, 'project').proposal_sha256, 'a'.repeat(64));
    assert.equal(trial.approvalRequest(workflow, session, 'project'), null);

    const clarification = { ...workflow, target_trial_card: {
      state: 'clarification', proposal: { ...workflow.target_trial_card.proposal, digest: null, missing: ['treatment'] },
    } };
    const missing = render(language, clarification, {}).html;
    assert.equal((missing.match(/data-gpi-clarification/g) || []).length, 1);
    assert.match(missing, language === 'zh' ? /你要研究哪种治疗/ : /Which treatment should be studied/);
    assert.match(missing, /data-gpi-confirm-edit/);
    assert.doesNotMatch(missing, /data-gpi-confirm-action/);

    const review = {
      next_action_code: 'target_trial_review',
      target_trial_card: { state: 'approvable', approvable: true, confirm_all_on_approval: true,
        compile_sha256: 'b'.repeat(64), confirmation_lines: 1, confirmations: [{ text: 'Review this assumption' }] },
    };
    assert.match(render(language, review, {}).html, /Review this assumption/);
    assert.equal(trial.approvalRequest(review, session, 'project').n_lines_confirmed, 1);
    review.target_trial_card.confirm_all_on_approval = false;
    assert.equal(trial.approvalRequest(review, session, 'project'), null);
  });
}
