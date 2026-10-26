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
