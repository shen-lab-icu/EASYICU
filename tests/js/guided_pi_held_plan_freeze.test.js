'use strict';

// m23 follow-up 1 (S1442): while mined ideas await a choice the automatic
// plan is held, and the UX walks (zh and en) found no freeze anywhere: the
// held plan card offered only "generate candidate plan", and the freeze
// lived in a reply's next-step block that the agent did not always write.
// The held card now carries the freeze itself, the only one in view.

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');

const js = path.resolve('src/easyicu/webserver/static/js');
global.window = {};
require(path.join(js, 'screens-agent-reader-vocab.js'));
for (const name of ['modules', 'data-consent', 'cohort-eligibility', 'confirmation-data',
  'question-requirements', 'confirmation', 'target-trial-copy', 'target-trial', 'transcript',
  'research-draft', 'session-view', 'next-actions', 'plan-actions', 'events', 'run-readiness']) {
  if (name === 'modules') require(path.join(js, 'html-escape.js'));
  require(path.join(js, `screens-guided-pi-${name}.js`));
}
const modules = window.EasyICU.guidedPi;
const { esc } = window.EU_HTML;
const stub = (overrides = {}) => new Proxy(overrides, { get: (item, key) => item[key] || (() => '') });

const draftBlock = question => '```easyicu-research-draft\n' + JSON.stringify({
  schema: 'easyicu.research-draft/1', title: 'Direction', question, population: '', exposure: '',
  outcome: '', candidate_data: '', approach: '', data_requirements: [], open_questions: [], changes: [],
}) + '\n```';

function render(language, { open = true, code = 'provider_ready_to_generate_plan', question = '' } = {}) {
  window.EU_LANG = language;
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const workflow = { next_action_code: code };
  const session = {
    session_id: 'session', binding: { study_context_id: 'study', study_revision: 3 },
    data_source_authorization: { status: 'confirmed', source: { label: 'Example ICU' } },
    idea_exploration_open: open,
  };
  const state = {
    session, workflow, workflowReceipts: [], sessions: [], draft: '',
    messages: [
      { role: 'user', text: 'Mine a few directions around one bedside measure.' },
      { role: 'activity', steps: [{ kind: 'tool', status: 'complete', toolName: 'easyicu_mine_ideas' }] },
      // The reply names its next step, so the freeze could appear twice.
      { role: 'assistant', complete: true, text: `Directions listed.${question ? `\n\n${draftBlock(question)}` : ''}\n\n**下一步：**\n- 先完成数据包注册，再生成研究计划` },
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
  return view.sessionPanel();
}

const count = (html, pattern) => (html.match(pattern) || []).length;
const QUESTION = { zh: '首日峰值乳酸与院内死亡', en: 'First-day peak lactate and in-hospital death' };
const COPY = {
  zh: { generate: '生成候选计划', freeze: '冻结一个方向并继续', choose: '写下要冻结的方向' },
  en: { generate: 'Generate candidate plan', freeze: 'Freeze a direction and continue', choose: 'Write the direction to freeze' },
};

for (const language of ['zh', 'en']) {
  test(`${language}: the held plan card carries the one freeze in view`, () => {
    const html = render(language, { question: QUESTION[language] });
    assert.ok(html.includes(COPY[language].generate), 'the generate button stays');
    assert.equal(count(html, /<button[^>]*data-gpi-idea-freeze[\s>]/g), 1);
    assert.ok(html.includes(`data-gpi-idea-freeze-question="${QUESTION[language]}"`));
    assert.ok(html.includes(COPY[language].freeze));
    // The card's freeze follows the card, not the reply above it.
    assert.ok(html.indexOf('data-gpi-idea-freeze-step') > html.indexOf('data-gpi-confirm-action'));
  });

  test(`${language}: with no direction under discussion the held card asks for one`, () => {
    const html = render(language);
    assert.equal(count(html, /<button[^>]*data-gpi-idea-freeze[\s>]/g), 1);
    assert.doesNotMatch(html, /data-gpi-idea-freeze-question/);
    assert.ok(html.includes(COPY[language].choose));
  });

  test(`${language}: a conversation that is not exploring shows no freeze on its plan card`, () => {
    const html = render(language, { open: false, question: QUESTION[language] });
    assert.ok(html.includes(COPY[language].generate));
    assert.doesNotMatch(html, /data-gpi-idea-freeze/);
  });
}

test('another held step keeps the freeze in the reply, never on its card', () => {
  const html = render('zh', { code: 'study_primary_outcome_confirmation_required', question: QUESTION.zh });
  assert.ok(count(html, /<button[^>]*data-gpi-idea-freeze[\s>]/g) <= 1);
  const log = html.split('data-gpi-log>')[1] || html;
  const cardAt = log.indexOf('data-gpi-confirm-action');
  const freezeAt = log.indexOf('data-gpi-idea-freeze-step');
  assert.ok(freezeAt < 0 || cardAt < 0 || freezeAt < cardAt);
});
