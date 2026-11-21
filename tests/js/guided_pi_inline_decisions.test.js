'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');

const js = path.resolve('src/easyicu/webserver/static/js');
global.window = {};
require(path.join(js, 'screens-agent-reader-vocab.js'));
for (const name of ['modules', 'data-consent', 'cohort-eligibility', 'confirmation-data',
  'question-requirements', 'confirmation', 'target-trial-copy', 'target-trial', 'transcript',
  'session-view', 'next-actions', 'plan-actions', 'events', 'run-readiness']) {
  if (name === 'modules') require(path.join(js, 'html-escape.js'));
  require(path.join(js, `screens-guided-pi-${name}.js`));
}
const modules = window.EasyICU.guidedPi;
const { esc } = window.EU_HTML;
const stub = (overrides = {}) => new Proxy(overrides, { get: (item, key) => item[key] || (() => '') });

function render(language, workflow, sessionPatch, ideaExplored = true, hostPatch = {}, statePatch = {}) {
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
    ...statePatch,
  };
  const host = { tr, esc, iconHtml: () => '', resourceButton: () => '', session: () => session,
    workflow: () => workflow, sessionIsStale: () => false, busy: () => false, ...hostPatch };
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
  return { html: view.sessionPanel(), confirmation, state, view };
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
      const log = conversationLog(html);
      const primary = primaryButtons(log);
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
    const actions = modules.require('planActions').create({
      session: () => ({ session_id: 'session' }), busy: () => false, sessionIsStale: () => false,
      sendText: (...args) => messages.push(args),
    });
    assert.deepEqual(messages, []);
    await actions.confirmWorkflow(proposal);
    assert.deepEqual(messages, [[proposal.exposureArrangement.options.primary_and_secondary,
      ['provider_run', 'literature'], 'confirm_exposure_arrangement', true, 'host_prompt']]);
  });
}

function conversationLog(html) {
  return html.split('data-gpi-log>')[1].split('<div class="gpi-compose">')[0];
}

function primaryButtons(html) {
  return html.match(/<button\b[^>]*class="btn [^"]*\bprimary\b[^"]*"[^>]*>/g) || [];
}

const cohortSelection = {
  present: true, stated: false, blocker_code: 'cohort_eligibility_confirmation_required',
  options: [{ id: 'no_eligibility_filter' }, { id: 'first_admission_only' }],
};

function m3Workflow(shape) {
  const populationStop = shape === 'C';
  const nextAction = populationStop
    ? 'population_criterion_decision_unconfirmed' : 'question_requirement_not_covered';
  const items = (shape === 'B'
    ? [['not_covered', 'host_verified'], ['attested', 'plan_declared'], ['attested', 'plan_declared']]
    : [['not_covered', 'host_verified'], ['not_covered', 'host_verified']]
  ).map(([disposition, source], index) => ({
    id: `r${index + 1}`, kind: 'analysis', disposition, source, quote: `示例分析 ${index + 1}`,
  }));
  return {
    current_stage: 'plan',
    next_action_code: nextAction,
    next_actions: [{
      action_id: 'edit_plan_0123456789abcdef', kind: 'edit_plan', label_key: 'stop_next_actions.edit_plan',
      params: { route: 'plan_edit', intent: 'candidate_plan', planner_start_mode: 'fresh',
        requires_explicit_choice: true, requires_plan_approval: true },
    }],
    plan_decision: null,
    stages: [
      { id: 'idea', status: 'complete' },
      { id: 'question', status: 'complete' },
      populationStop ? { id: 'setup', status: 'complete' }
        : { id: 'setup', status: 'ready', reason_code: 'cohort_eligibility_confirmation_required' },
      { id: 'plan', status: 'review_required', reason_code: nextAction },
      populationStop ? { id: 'extraction', status: 'complete', reason_code: 'bound_research_input_prepared' }
        : { id: 'extraction', status: 'blocked', reason_code: 'metadata_only_input_not_prepared' },
    ],
    plan_review_summary: {
      status: 'analysis_only', run_id: 'synthetic-m3-review',
      question_requirements: { status: 'shown', judged_on_plan_under_review: true, items },
      ...(populationStop ? { reviewed_population_spec: {
        schema_version: 'easyicu.reviewed_population_spec_projection/1',
        criteria: [{
          criterion: { kind: 'condition_present', role: 'include', quote: '示例纳入条件',
            concepts_all_of: ['sample_condition'], window: null, missing_policy: null },
          disposition: 'not_applied',
        }],
      } } : {}),
    },
  };
}

for (const [shape, recording] of [['A', 's15'], ['B', 's15b'], ['C', 's15c']]) {
  test(`Dev9 M3 shape ${shape} (${recording}) offers one primary forward action and the inline revision`, () => {
    const workflow = m3Workflow(shape);
    const researchProvider = { provider: 'synthetic', model: 'scripted', credential_source: 'pi_verified' };
    const readiness = modules.require('runReadiness').create({
      tr: (en, zh) => zh,
      state: { workflow, session: { research_provider: researchProvider },
        runtime: { status: 'ready', api_transport: 'openai-completions' } },
    });
    const { html, confirmation, state, view } = render('zh', workflow, {
      research_provider: researchProvider,
      ...(shape === 'C' ? {} : { cohort_eligibility_selection: cohortSelection }),
    }, false, { runReadiness: readiness.forAction }, {
      messages: [
        { role: 'user', text: '请为两个示例指标设计分析。' },
        { role: 'user', text: '使用这份数据' },
        { role: 'user', text: '确认结局' },
        ...(shape === 'C' ? [{ role: 'user', text: '确认方案并准备数据' }] : []),
        { role: 'assistant', complete: true, text: '候选计划已保存。' },
      ],
    });
    const log = conversationLog(html);
    assert.match(log, shape === 'C' ? /请明确人群条件的判定方式/ : /题面要求的一项分析，这份计划没有回答/);
    if (shape === 'C') {
      assert.match(log, /示例纳入条件/);
      assert.match(log, /无法判定时: 未注明/);
    } else {
      assert.match(log, /示例分析 1/);
    }
    assert.doesNotMatch(log, /data-gpi-aside-pending|data-gpi-cohort-option|data-gpi-exposure-arrangement/);
    const primary = primaryButtons(log);
    assert.equal(primary.length, 1, 'the conversation must contain exactly one primary forward action');
    if (shape === 'C') {
      assert.match(primary[0], /data-gpi-confirm-edit>/, 'the primary action must open the existing edit path');
      assert.doesNotMatch(log, /data-gpi-confirm-action/);
    } else {
      // S1757: an unanswered question requirement keeps a fresh plan as the
      // forward action; the edit path stays beside it, never alone.
      assert.match(primary[0], /data-gpi-confirm-action>$/);
      assert.match(log, /<button class="btn primary sm" type="button" data-gpi-confirm-action>重新生成计划<\/button>/);
    }
    assert.match(log, /data-gpi-confirm-edit>提出修改<\/button>/);
    const edit = log.match(/<button\b[^>]*data-gpi-confirm-edit[^>]*>/)[0];

    const handlers = {};
    state.host = { addEventListener: (name, handler) => { handlers[name] = handler; } };
    modules.require('events').create({
      state, COHORT_ELIGIBILITY: stub(), MESSAGE_ACTIONS: stub(),
      confirmWorkflowAction: () => assert.fail('Revision must not submit approval or regenerate automatically'),
      editWorkflow: () => { state.draft = confirmation.planChangeDraft(); },
      sendMessage: () => {
        assert.match(state.draft, /请补全示例要求。$/);
        state.workflowReceipts.push({ role: 'assistant', text: '已收到示例修订意见。', complete: true });
      },
    }).wire();
    const disabled = /\sdisabled(?:[\s=>])/;
    const sendButton = rendered => rendered.match(/<button\b[^>]*data-gpi-send[^>]*>/)?.[0];
    const click = button => {
      assert.ok(button);
      if (disabled.test(button)) return;
      handlers.click({
        target: { closest: selector => button.includes(selector.slice(1, -1)) ? { dataset: {} } : null },
      });
    };
    assert.equal(state.draft, '');
    click(edit);
    assert.match(state.draft, /请根据以下审阅意见修订整份研究计划/);
    assert.match(state.draft, /不要开始分析/);
    assert.deepEqual(state.workflowReceipts, []);
    state.draft += '请补全示例要求。';
    state.projectLoading = true;
    const lockedSend = sendButton(view.sessionPanel());
    assert.match(lockedSend, disabled);
    click(lockedSend);
    assert.deepEqual(state.workflowReceipts, []);

    state.projectLoading = false;
    const updated = view.sessionPanel();
    assert.match(updated, /请补全示例要求。<\/textarea>/);
    const send = sendButton(updated);
    assert.doesNotMatch(send, disabled);
    click(send);
    assert.equal(state.workflowReceipts.length, 1);
    assert.equal(state.workflowReceipts[0].text, '已收到示例修订意见。');
  });
}

test('S1758: an unanswered population condition is named on the card that offers a fresh plan', () => {
  const workflow = m3Workflow('A');
  workflow.plan_review_summary.question_population_unlisted = ['示例条件'];
  const researchProvider = { provider: 'synthetic', model: 'scripted', credential_source: 'pi_verified' };
  const readiness = modules.require('runReadiness').create({
    tr: (en, zh) => zh,
    state: { workflow, session: { research_provider: researchProvider },
      runtime: { status: 'ready', api_transport: 'openai-completions' } },
  });
  const { html } = render('zh', workflow, {
    research_provider: researchProvider, cohort_eligibility_selection: cohortSelection,
  }, false, { runReadiness: readiness.forAction }, {
    messages: [{ role: 'user', text: '确认结局' }, { role: 'assistant', complete: true, text: '候选计划已保存。' }],
  });
  const log = conversationLog(html);
  assert.match(log, /问题限定了人群“示例条件”，方案未包含对应的人群条件，请确认。/);
  const primary = primaryButtons(log);
  assert.equal(primary.length, 1);
  assert.match(log, /<button class="btn primary sm" type="button" data-gpi-confirm-action>重新生成计划<\/button>/);
  assert.match(log, /data-gpi-confirm-edit>提出修改<\/button>/);
});

for (const shape of ['A', 'C']) {
  test(`Dev9 M3 shape ${shape} keeps Check connection primary when disconnected`, () => {
    const workflow = m3Workflow(shape);
    const tr = (en, zh) => zh;
    const readiness = modules.require('runReadiness').create({ state: { workflow }, tr });
    const readinessChecks = [];
    const { html, confirmation, state } = render('zh', workflow, {}, false, {
      runReadiness(code) {
        readinessChecks.push(code);
        return readiness.forAction(code);
      },
    });
    assert.ok(readinessChecks.length > 0, 'the offered edit must query run readiness');
    assert.ok(readinessChecks.every(code => code === workflow.next_action_code));
    assert.equal(confirmation.workflowConfirmation().readiness?.connect, true);
    const log = conversationLog(html);
    const primary = primaryButtons(log);
    assert.equal(primary.length, 1);
    assert.match(primary[0], /data-gpi-config>/);
    assert.match(log, /data-gpi-config>检查连接<\/button>/);
    assert.match(log, /data-gpi-run-readiness>请先为此对话连接模型。<\/small>/);
    assert.doesNotMatch(log, /data-gpi-confirm-action/);

    let click;
    let renders = 0;
    state.error = 'Previous connection error';
    state.host = { addEventListener: (event, handler) => { if (event === 'click') click = handler; } };
    modules.require('events').create({
      state, COHORT_ELIGIBILITY: stub(), MESSAGE_ACTIONS: stub(), render: () => { renders++; },
    }).wire();
    click({ target: { closest: selector => selector === '[data-gpi-config]' ? {} : null } });
    assert.equal(state.showSetup, true);
    assert.equal(state.error, '');
    assert.equal(renders, 1);
  });
}

test('Pending review focuses the inline decision and the stop link requires a recovery card', () => {
  const recovery = '.gpi-confirmation[data-gpi-review-recovery]';
  const confirmation = '.gpi-confirmation';
  const typedCards = '.gpi-cohort-eligibility, .gpi-data-consent';
  const calls = [];
  const card = name => ({
    scrollIntoView: () => calls.push(['scroll', name]),
    setAttribute: (...args) => calls.push(args),
    focus: () => calls.push(['focus', name]),
  });
  const cards = new Map();
  let click;
  const state = { host: {
    querySelector: selector => cards.get(selector),
    addEventListener: (event, handler) => { if (event === 'click') click = handler; },
  } };
  const events = modules.require('events').create({ state });
  events.wire();
  const clickStopLink = () => click({
    target: { closest: selector => selector === '[data-gpi-stop-recovery]' ? {} : null },
  });

  cards.set(typedCards, card('typed'));
  events.revealPendingReview();
  cards.set(confirmation, card('confirmation'));
  events.revealPendingReview();
  clickStopLink();
  cards.set(recovery, card('recovery'));
  events.revealPendingReview();
  clickStopLink();
  assert.deepEqual(calls, ['typed', 'confirmation', 'recovery', 'recovery'].flatMap(name => [
    ['scroll', name], ['tabindex', '-1'], ['focus', name],
  ]));
});

const trialProposal = {
  next_action_code: 'target_trial_statement_needed',
  target_trial_card: { state: 'proposal', proposal: {
    expected_revision: 7, digest: 'a'.repeat(64), missing: [],
    elements: [{ key: 'treatment', value: 'example infusion', source: 'question' }],
  } },
};
const trialReview = {
  next_action_code: 'target_trial_review',
  target_trial_card: { state: 'approvable', approvable: true, confirm_all_on_approval: true,
    compile_sha256: 'b'.repeat(64), confirmation_lines: 1, confirmations: [{ text: 'Review the index stay' }] },
};

function trialActions() {
  const state = { selection: 0, busy: false, workflow: trialProposal, error: '', watched: [] };
  const session = { session_id: 'session', binding: { study_context_id: 'study', study_revision: 7 } };
  const api = {};
  const actions = modules.require('planActions').create({
    session: () => session, workflow: () => state.workflow, projectId: () => 'project',
    selectionRevision: () => state.selection, sessionIsStale: () => false,
    busy: () => state.busy, setBusy: busy => { state.busy = busy; },
    api: () => api, setError: error => { state.error = error; }, errorText: error => error.message,
    render() {}, refreshSession: async () => {}, loadWorkflow: async () => { await state.onLoad?.(); },
    watchChildJob: (...args) => { state.watched.push([state.busy, ...args]); },
  });
  const click = () => actions.confirmWorkflow(render('en', state.workflow, session).confirmation.workflowConfirmation());
  return { state, api, click };
}

test('Continue releases the action before watching the job and only an Approve click submits approval', async () => {
  const { state, api, click } = trialActions();
  const approvals = [];
  api.compilePiCopilotTargetTrialProposal = async (session, body) => {
    assert.equal(session, 'session');
    assert.equal(body.proposal_sha256, 'a'.repeat(64));
    assert.equal(state.busy, true);
    return { job_id: 'trial-job' };
  };
  api.approvePiCopilotTargetTrial = async (session, body) => { approvals.push([session, body]); };
  await click();
  assert.equal(state.busy, false);
  assert.deepEqual(state.watched, [[false, 'trial-job', 'easyicu_target_trial_compile_submitted']]);
  state.workflow = trialReview;
  assert.deepEqual(approvals, []);
  await click();
  assert.equal(state.busy, false);
  assert.equal(approvals.length, 1);
  assert.equal(approvals[0][1].compile_sha256, 'b'.repeat(64));
  assert.equal(approvals[0][1].n_lines_confirmed, 1);
});

for (const [step, workflow, method] of [
  ['Continue', trialProposal, 'compilePiCopilotTargetTrialProposal'],
  ['Approve', trialReview, 'approvePiCopilotTargetTrial'],
]) {
  for (const exit of ['current error', 'stale success', 'stale error', 'stale after reload']) {
    test(`${step} updates busy state only for its own selection on ${exit}`, async () => {
      const { state, api, click } = trialActions();
      state.workflow = workflow;
      let loads = 0;
      const switchSelection = () => {
        state.selection += 1;
        state.busy = true; // The newly selected conversation has work in flight.
      };
      state.onLoad = () => {
        loads += 1;
        if (exit === 'stale after reload') switchSelection();
      };
      api[method] = async () => {
        if (['stale success', 'stale error'].includes(exit)) switchSelection();
        if (exit !== 'stale success') throw new Error('Synthetic transport failure');
        return { job_id: 'old-job' };
      };
      await click();
      assert.equal(state.busy, exit !== 'current error');
      assert.deepEqual(state.watched, []);
      assert.equal(loads, ['current error', 'stale after reload'].includes(exit) ? 1 : 0);
      assert.equal(state.error, exit === 'current error' ? 'Synthetic transport failure' : '');
    });
  }
}

function exposureWorkflow(reviewPatch = {}) {
  return {
    next_action_code: 'question_requirement_not_covered',
    plan_review_summary: {
      run_id: 'synthetic-review',
      exposure_arrangement: {
        primary: 'signal_a', additional: ['signal_b'], default_option: 'primary_and_secondary',
        options: {
          primary_and_secondary: { en: 'Keep the first admission signal primary.' },
          analyse_both: { en: 'Estimate each admission signal in parallel.' },
        },
      },
      question_requirements: { status: 'shown', judged_on_plan_under_review: true, items: [{
        source: 'host_verified', disposition: 'not_covered', quote: 'Compare the two admission signals.',
        exposure_choice: { primary_exposure: 'signal_a', additional_exposures: ['signal_b'] },
      }] },
      ...reviewPatch,
    },
  };
}

test('A pending scientific-review decision has one primary connection action in the conversation', () => {
  const workflow = {
    next_action_code: 'plan_scientific_changes_required',
    plan_review_summary: { finding_codes: ['HOST_ICD_SOURCE_UNVERIFIED'] },
  };
  const readiness = modules.require('runReadiness').create({ state: { workflow }, tr: en => en });
  const log = conversationLog(render('en', workflow, {}, true, { runReadiness: readiness.forAction }).html);
  const primary = primaryButtons(log);
  assert.equal(primary.length, 1);
  assert.match(primary[0], /data-gpi-config/);
  assert.match(log, /data-gpi-confirm-edit/);
  assert.doesNotMatch(log, /data-gpi-aside-pending/);
});

for (const language of ['zh', 'en']) {
  test(`${language}: a survival decision without a digest asks for the missing design inline`, () => {
    const code = 'SURVIVAL_LANDMARK_OWNER_NOT_SEALED';
    const question = 'For survival after admission, which mortality endpoint, follow-up duration, landmark and baseline variables should we use?';
    const workflow = {
      next_action_code: 'plan_scientific_changes_required',
      plan_review_summary: {
        run_id: 'synthetic-survival-review',
        authorization_questions: [{ code, question, evidence: 'Follow-up and adjustment are unspecified.' }],
        remediation_buckets: { study_authority_change: [code] },
      },
    };
    const log = conversationLog(render(language, workflow, {}).html);
    const expected = language === 'zh' ? window.AGENT_READER_VOCAB.question(code) : question;
    if (language === 'zh') assert.match(expected, /哪些调整协变量/);
    assert.equal(log.match(/class="gpi-confirmation-body">.*?<small>(.*?)<\/small>/s)?.[1], expected);
    assert.match(log, /data-gpi-confirm-edit/);
    assert.doesNotMatch(log, /data-gpi-plan-decision-(?:card|option)|data-gpi-aside-pending/);
  });
}

test('A disconnected candidate-plan action shows only Check connection as primary', () => {
  const workflow = { next_action_code: 'provider_ready_to_generate_plan' };
  const readiness = modules.require('runReadiness').create({ state: { workflow }, tr: en => en });
  const log = conversationLog(render('en', workflow, {}, true, { runReadiness: readiness.forAction }).html);
  const primary = primaryButtons(log);
  assert.equal(primary.length, 1);
  assert.match(primary[0], /data-gpi-config/);
  assert.doesNotMatch(log, /data-gpi-confirm-action/);
});

test('A disconnected exposure arrangement uses Check connection as its only primary action', () => {
  const workflow = exposureWorkflow();
  const readiness = modules.require('runReadiness').create({ state: { workflow }, tr: en => en });
  const log = conversationLog(render('en', workflow, {}, true, { runReadiness: readiness.forAction }).html);
  const primary = primaryButtons(log);
  assert.equal(primary.length, 1);
  assert.match(primary[0], /data-gpi-config/);
  assert.match(log, /data-gpi-exposure-arrangement="analyse_both" disabled/);
});

test('Request changes remains inline with another requirement and no exposure offer', () => {
  const workflow = exposureWorkflow({ exposure_arrangement: null });
  workflow.plan_review_summary.question_requirements.items.push({
    source: 'host_verified', disposition: 'not_covered', quote: 'Show estimates by ICU admission shift.',
  });
  const log = conversationLog(render('en', workflow, {}).html);
  assert.match(log, /data-gpi-confirm-edit>Request changes/);
  assert.match(log, /Show estimates by ICU admission shift/);
  assert.doesNotMatch(log, /data-gpi-exposure-arrangement/);
});

test('Exposure card suppresses only the note for the matching verified uncovered arrangement', () => {
  const workflow = exposureWorkflow();
  const { items } = workflow.plan_review_summary.question_requirements;
  items.push(
    { ...items[0], source: 'plan_declared' },
    { ...items[0], exposure_choice: { primary_exposure: 'signal_a', additional_exposures: ['signal_d'] } },
  );
  const log = conversationLog(render('en', workflow, {}).html);
  assert.match(log, /Not answered by this plan: “Compare the two admission signals.”/);
  assert.equal((log.match(/The question asks to analyse signal_a and signal_b/g) || []).length, 1);
  assert.match(log, /The question asks to analyse signal_a and signal_d/);
});

for (const gate of ['ready', 'invalid scope', 'unready', 'busy', 'stale', 'source pending']) {
  test(`An exposure alternative sent through the card respects ${gate}`, async () => {
    const workflow = exposureWorkflow(gate === 'invalid scope' ? { deferred_input_scope: {} } : {});
    const hostPatch = { runReadiness: () => gate === 'unready' ? { reason: 'Check connection', connect: true } : null };
    const view = render('en', workflow, {}, true, hostPatch);
    const messages = [];
    const sendText = (...args) => messages.push(args);
    const session = { session_id: 'session' };
    const actions = modules.require('planActions').create({
      session: () => session, busy: () => gate === 'busy', sessionIsStale: () => gate === 'stale',
      researchSourceReady: () => gate !== 'source pending', sendText,
    });
    const handlers = {};
    modules.require('events').create({
      state: { session, workflow, busy: gate === 'busy', host: {
        addEventListener: (name, handler) => { handlers[name] = handler; },
      } },
      COHORT_ELIGIBILITY: stub(), tr: en => en, sendText,
      confirmWorkflowAction: option => actions.confirmWorkflow(view.confirmation.workflowConfirmation(), option),
    }).wire();
    const alternative = { dataset: { gpiExposureArrangement: 'analyse_both' } };
    await handlers.click({ target: {
      closest: selector => selector === '[data-gpi-exposure-arrangement]' ? alternative : null,
    } });
    assert.deepEqual(messages, gate === 'ready'
      ? [['Estimate each admission signal in parallel.', ['provider_run', 'literature'], 'confirm_exposure_arrangement', true, 'host_prompt']]
      : []);
    if (['invalid scope', 'unready'].includes(gate)) {
      assert.match(view.html, /data-gpi-exposure-arrangement="analyse_both" disabled/);
    }
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
    const missing = conversationLog(render(language, clarification, {}).html);
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

for (const language of ['en', 'zh']) {
  test(`${language}: unread target trial wording is quoted on the inline clarification`, () => {
    const residual = ['２４　 hours  after ICU admission', '插管 <后>'];
    const workflow = {
      next_action_code: 'target_trial_statement_needed',
      target_trial_card: {
        state: 'clarification',
        proposal: {
          digest: null, missing: ['unread_words'], unread_words: residual,
          elements: [{ key: 'outcome', value: '90-day mortality', source: 'question' }],
        },
      },
    };
    const html = conversationLog(render(language, workflow, {}).html);
    assert.match(html, /data-gpi-clarification/);
    assert.match(html, language === 'zh' ? /请说明以下措辞的含义/ : /What do the following words mean/);
    assert.match(html, /90-day mortality/);
    assert.match(html, /data-gpi-confirm-edit/);
    assert.doesNotMatch(html, /data-gpi-confirm-action/);
    for (const fragment of residual) assert.ok(html.includes(esc(fragment)));
  });
}
