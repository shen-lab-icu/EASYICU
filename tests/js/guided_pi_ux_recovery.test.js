'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const read = name => fs.readFileSync(path.join(root, 'src/easyicu/webserver', name), 'utf8');

test('first-turn research and article markers preserve the exact replay snapshot', () => {
  const main = read('pi_copilot/node_app/src/main.mjs');
  const sandbox = { boundedText: (value, max) => String(value || '').slice(0, max),
    OWNER_CONTEXT_MARKER: '\n\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\n',
    HOST_TRANSITION_MARKER: '\n\n[EASYICU_HOST_TRANSITION_V1]\n',
    ATTACHED_SKILL_MARKER: '\n\n[EASYICU_ATTACHED_SKILLS_V1]\n', MENTION_MARKER: '\n\n[EASYICU_MENTIONS_V1]\n' };
  vm.createContext(sandbox);
  vm.runInContext(main.slice(main.indexOf('function userPromptStudySnapshot'), main.indexOf('function resourceLoader')), sandbox);
  const receipt = { code: 'easyicu_research_workflow_projected', authority: { study_context_id: 'study-new', study_revision: 1 },
    details: { workflow: { study_setup_receipt: { schema_version: 'easyicu.pi-study-setup-receipt/1', study_context_id: 'study-new', revision: 1, configuration: { question: '', covariates: [] } } } } };
  for (const marker of ['', '\n\n[EASYICU_RESEARCH_ENTRY_V1]\n', '\n\n[EASYICU_IDEA_SOURCE_V1]\n', sandbox.HOST_TRANSITION_MARKER]) {
    const prompt = 'Please discuss my ICU question.' + sandbox.OWNER_CONTEXT_MARKER + JSON.stringify([receipt]) + (marker ? marker + 'Host instruction after the receipt.' : '');
    const restored = sandbox.userPromptStudySnapshot(prompt);
    assert.equal(restored.study_context_id, 'study-new');
    assert.equal(restored.source_revision, 1);
    assert.equal(JSON.stringify(restored.configuration), JSON.stringify(receipt.details.workflow.study_setup_receipt.configuration));
  }
  receipt.authority.study_revision = 2;
  assert.equal(sandbox.userPromptStudySnapshot(sandbox.OWNER_CONTEXT_MARKER + JSON.stringify([receipt])), undefined);
  assert.equal(sandbox.userPromptStudySnapshot('No saved receipt'), undefined);
});

function moduleOwner(file, name) {
  const modules = {}, window = { EU_LANG: 'zh', EU_HTML: { esc: value => String(value) },
    EasyICU: { guidedPi: { declare: (key, value) => { modules[key] = value; } } } };
  vm.runInNewContext(read('static/js/' + file), { window });
  return modules[name];
}

test('lightweight routing preserves a draft and does not mistake real-data work for a demo', () => {
  const owner = moduleOwner('screens-guided-pi-task-scope.js', 'taskScope');
  assert.equal(owner.kind('请生成200条合成成人ICU记录，不读取真实数据，做中文演示。'), 'synthetic');
  assert.equal(owner.kind('Create 200 synthetic ICU records and a chart.'), 'synthetic');
  assert.equal(owner.kind('先讨论 MIMIC-IV 研究方案，不要执行。'), 'discussion');
  assert.equal(owner.kind('只讨论 MIMIC-IV 研究方案。'), 'discussion');
  assert.equal(owner.kind('先设计研究方案，不运行分析。'), 'discussion');
  assert.equal(owner.kind('First discuss the study plan; do not run analysis.'), 'discussion');
  assert.equal(owner.kind('Just discuss a study design for MIMIC-IV.'), 'discussion');
  for (const question of [
    '请先设计研究方案，然后用 MIMIC-IV 真实数据执行分析。',
    '我想用 eICU 数据，先讨论研究方法，再完成完整分析。',
    '请生成200条合成数据，然后分析真实队列数据。',
    'Create 200 synthetic ICU records, then analyze real cohort data.',
    'First discuss the MIMIC-IV study design, then run the analysis.',
    '先讨论研究方案，再完成完整分析。',
  ]) assert.equal(owner.kind(question), '');
  for (const question of ['不是合成数据，请分析MIMIC-IV。', 'Compare real MIMIC-IV records to synthetic data.', '合成数据是否足够？', '在 eICU 中估计院内死亡率。']) assert.equal(owner.kind(question), '');
  const task = owner.current({ taskDraft: '请生成合成数据演示', messages: [] });
  assert.equal(task.text, '请生成合成数据演示');
  assert.equal(task.kind, 'synthetic');
});

test('example search reaches collapsed examples without sending any question', () => {
  const owner = moduleOwner('screens-guided-pi-starters.js', 'starters');
  const cards = [{ textContent: 'HiRID lactate', hidden: false }, { textContent: 'SICdb survival', hidden: false }];
  const more = { open: false };
  const root = { querySelectorAll: () => cards, querySelector: () => more };
  owner.filter(root, 'sicdb');
  assert.equal(cards[0].hidden, true); assert.equal(cards[1].hidden, false); assert.equal(more.open, true);
  owner.filter(root, '');
  assert.ok(cards.every(card => !card.hidden)); assert.equal(more.open, false);
});

test('a new workspace task carries its question without the old entry intent or article', async () => {
  const source = read('static/js/screens-guided-pi.js');
  const switchMode = source.slice(source.indexOf('  async function switchMode('), source.indexOf('  const LIVE_STREAM ='));
  const question = 'Create 200 synthetic ICU records and a chart.';
  for (const busy of [false, true]) {
    const state = { busy, agentMode: 'research', session: { session_id: 'old' }, sessions: [],
      messages: [{ role: 'user', text: 'Old question' }], draft: 'Old draft', taskDraft: question,
      pendingEntryIntent: 'data_first_entry', followUpDraft: { sessionId: 'old' } };
    let articleResets = 0, created = null;
    const sandbox = { state, agentMode: () => state.agentMode,
      IDEA_SOURCE: { reset: () => { articleResets += 1; } },
      MODULES: { optional: () => null, require: () => ({}) },
      closeSource() {}, closeChildSource() {}, closeSourceView() {},
      projectId: () => 'project', uiLanguage: () => 'en', rememberSession() {},
      createSession: async () => { created = { ...state }; } };
    vm.runInNewContext(switchMode, sandbox);
    await sandbox.switchMode('workspace', { newSession: true, draft: question });
    if (busy) {
      assert.equal(created, null); assert.equal(articleResets, 0);
      assert.equal(state.pendingEntryIntent, 'data_first_entry');
      continue;
    }
    assert.equal(created.agentMode, 'workspace'); assert.equal(created.session, null);
    assert.equal(created.draft, question); assert.equal(created.pendingEntryIntent, '');
    assert.equal(created.followUpDraft, null); assert.equal(articleResets, 1);
  }
});

test('the model menu keeps an unavailable saved choice visible as a warning', () => {
  const owner = moduleOwner('screens-guided-pi-model-menu.js', 'modelMenu');
  Object.assign(owner.state, { models: ['available-chat'], current: 'removed-chat', loading: false, error: '' });
  const html = owner.rowsHtml();
  assert.match(html, /removed-chat/); assert.match(html, /available-chat/);
  assert.doesNotMatch(html, /aria-checked="true"/);
});
