'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function harness() {
  let owner;
  const calls = [];
  const session = { session_id: 'pi-synthetic', quick_answers: [{ quick_answer_id: 'qa_synthetic',
    proposal: { status: 'approval_required', card: { sha256: 'a'.repeat(64) } } }] };
  let busy = false;
  const api = {
    approvePiQuickAnswer: async (...args) => calls.push(['approve', ...args]),
    expandPiQuickAnswer: async (...args) => calls.push(['expand', ...args]),
  };
  const context = vm.createContext({ window: { EasyICU: { guidedPi: { declare: (_name, value) => { owner = value; } } } } });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/static/js/screens-guided-pi-plan-actions.js'), 'utf8'), context);
  const actions = owner.create({
    session: () => session, projectId: () => 'project-synthetic', api: () => api,
    busy: () => busy, sessionIsStale: () => false, researchSourceReady: () => true,
    setBusy: value => { busy = value; }, setError: error => assert.fail(error), errorText: String,
    refreshSession: async () => calls.push('refresh'), loadWorkflow: async () => calls.push('workflow'),
    render: () => {}, sendText: () => assert.fail('An approval is never a chat turn'),
  });
  return { actions, calls };
}

test('the plan approval dispatcher sends the displayed digest and explicit substitution click', async () => {
  const { actions, calls } = harness();
  await actions.confirmWorkflow({ code: 'approve_quick_answer', quick_answer_id: 'qa_synthetic',
    card_sha256: 'a'.repeat(64), accept_substitutions: true });
  assert.deepEqual(JSON.parse(JSON.stringify(calls)), [
    ['approve', 'pi-synthetic', 'qa_synthetic', { project_id: 'project-synthetic', card_sha256: 'a'.repeat(64), accept_substitutions: true }],
    'refresh', 'workflow',
  ]);
});

test('a stale digest makes no request and substitution acceptance is never inferred', async () => {
  const { actions, calls } = harness();
  await actions.confirmWorkflow({ code: 'approve_quick_answer', quick_answer_id: 'qa_synthetic', card_sha256: 'b'.repeat(64) });
  assert.equal(calls.length, 0);
  await actions.confirmWorkflow({ code: 'approve_quick_answer', quick_answer_id: 'qa_synthetic', card_sha256: 'a'.repeat(64), accept_substitutions: 'yes' });
  assert.equal(calls[0][3].accept_substitutions, false);
});

test('expansion passes only the host attachment coordinate and refreshes the prefilled study', async () => {
  const { actions, calls } = harness();
  await actions.confirmWorkflow({ code: 'expand_into_full_study', quick_answer_id: 'qa_synthetic', reading: { invented: true } });
  assert.deepEqual(JSON.parse(JSON.stringify(calls)), [
    ['expand', 'pi-synthetic', 'qa_synthetic', { project_id: 'project-synthetic' }], 'refresh', 'workflow',
  ]);
});

// After the data-source transition (or any turn) the automatic full plan
// waits behind a qualified quick card and starts as before after a stop.
function progression(quickAnswers) {
  const declared = {};
  const window = { EasyICU: { guidedPi: {
    declare: (name, value) => { declared[name] = value; },
    optional: name => declared[name] || null,
  } } };
  const context = vm.createContext({ window });
  for (const file of ['screens-guided-pi-quick-answers.js', 'screens-guided-pi-plan-actions.js']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../src/easyicu/webserver/static/js', file), 'utf8'), context);
  }
  const question = '入 ICU 第一天最高乳酸的中位数是多少？男女之间有差别吗？';
  const session = {
    session_id: 'pi-synthetic', last_message_job_id: 'job-transition', quick_answers: quickAnswers(question),
    binding: { study_context_id: 'study-synthetic', study_revision: 4 },
    research_provider: { provider: 'easyicu-local', credential_source: 'local' },
  };
  const workflow = { next_action_code: 'provider_ready_to_generate_plan',
    study_setup_receipt: { configuration: { question } } };
  const started = [];
  let busy = false;
  const api = {
    loadStudyContext: async id => ({ context: { id, question, data_source: { path: '/synthetic/export' } } }),
    startAgentRun: async payload => { started.push(payload); return { job_id: 'job-plan' }; },
  };
  const actions = declared.planActions.create({
    tr: (en, zh) => zh, session: () => session, workflow: () => workflow, projectId: () => 'project-synthetic',
    api: () => api, busy: () => busy, sessionIsStale: () => false, researchSourceReady: () => true,
    setBusy: value => { busy = value; }, setError: () => {}, errorText: String, render: () => {},
    refreshSession: async () => {}, loadWorkflow: async () => {}, watchChildJob: () => {},
  });
  return { actions, started };
}

const proposal = (question, status) => ({
  status, reason: status === 'approval_required' ? 'quick_qualified' : 'quick_requires_adjusted_study',
  card: status === 'approval_required' ? { sha256: 'a'.repeat(64) } : null,
  expand: { action: 'expand_into_full_study', reading: { question }, source: {} },
});

test('a quick card from the source transition holds the automatic plan', async () => {
  const { actions, started } = progression(question => [{ quick_answer_id: 'qa_synthetic',
    message_job_id: 'job-transition', study_context_id: null, answer: null, proposal: proposal(question, 'approval_required') }]);
  assert.equal(await actions.continueSystemOwnedPlanProgression(), false);
  assert.equal(started.length, 0);
});

for (const status of ['stopped', 'full_study']) {
  test(`a ${status} quick proposal still starts the automatic plan`, async () => {
    const { actions, started } = progression(question => [{ quick_answer_id: 'qa_synthetic',
      message_job_id: 'job-transition', study_context_id: null, answer: null, proposal: proposal(question, status) }]);
    assert.equal(await actions.continueSystemOwnedPlanProgression(), true);
    assert.equal(started.length, 1);
    assert.equal(started[0].host_action, undefined);
    assert.equal(started[0].study_context_id, 'study-synthetic');
  });
}

test('a study question without a quick card keeps its automatic plan', async () => {
  const { actions, started } = progression(() => []);
  assert.equal(await actions.continueSystemOwnedPlanProgression(), true);
  assert.equal(started.length, 1);
  assert.equal(await actions.continueSystemOwnedPlanProgression({ passive: true }), false);
});
