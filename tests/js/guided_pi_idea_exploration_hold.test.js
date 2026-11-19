'use strict';

// Idea Mining saves its request as the study question with no outcome, so
// confirming a data source made the conversation start a full plan on the
// mining request itself.  While mined ideas await a choice the automatic plan
// waits, the plan card keeps its generate button, and that click starts the
// plan; a study question with an outcome keeps its automatic plan.
const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh' };
for (const file of process.argv.slice(2)) require(path.resolve(file));
const modules = window.EasyICU.guidedPi;
const tr = (en, zh) => zh;

function conversation(ideaExplorationOpen) {
  const question = ideaExplorationOpen
    ? '围绕乳酸与脓毒症结局，发掘几个官方 MIMIC-IV 演示数据能支持的研究想法。'
    : '入 ICU 第一天最高乳酸超过 2 mmol/L 的患者，28 天死亡率是否更高？';
  const session = {
    session_id: 'pi-synthetic', last_message_job_id: 'job-transition', quick_answers: [],
    idea_exploration_open: ideaExplorationOpen,
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
  const actions = modules.require('planActions').create({
    tr, session: () => session, workflow: () => workflow, projectId: () => 'project-synthetic',
    api: () => api, busy: () => busy, sessionIsStale: () => false, researchSourceReady: () => true,
    setBusy: value => { busy = value; }, setError: () => {}, errorText: String, render: () => {},
    refreshSession: async () => {}, loadWorkflow: async () => {}, watchChildJob: () => {}, appendMessage: () => {},
  });
  const card = modules.require('confirmation').create({ tr, esc: window.EU_HTML.esc, iconHtml: () => '',
    resourceButton: () => '', sessionIsStale: () => false, workflow: () => workflow,
    session: () => session, busy: () => busy });
  return { actions, card, started };
}

test('mined ideas awaiting a choice hold the automatic plan and keep its button', async () => {
  const { actions, card, started } = conversation(true);
  assert.equal(await actions.continueSystemOwnedPlanProgression(), false);
  assert.equal(started.length, 0);
  const html = card.workflowConfirmationHtml();
  assert.match(html, /data-gpi-confirm-action>生成候选计划</);
  await actions.confirmWorkflow({ code: 'provider_ready_to_generate_plan' });
  assert.equal(started.length, 1);
  assert.equal(started[0].study_context_id, 'study-synthetic');
});

test('a study question with an outcome still starts its plan automatically', async () => {
  const { actions, started } = conversation(false);
  assert.equal(await actions.continueSystemOwnedPlanProgression(), true);
  assert.equal(started.length, 1);
});
