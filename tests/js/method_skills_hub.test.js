/* Method Skills are projected from real capability contracts and start a task. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
global.window = global;
global.EU_LANG = 'zh';
global.t = (_en, zh) => zh;
global.icon = () => '';
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.SCREENS = {};
global.EU_EXTENSIONS = { activation_sha256: 'a'.repeat(64), skills: [] };
global.EU_CAPABILITIES = { capabilities: {
  publication_skills: { enabled: true, items: [] },
  method_skills: { enabled: true, items: [
    { id: 'survival-time-to-event', title: 'Survival', title_zh: '生存分析（时间到事件）',
      category: 'Survival analysis', category_zh: '生存分析', description: 'Contract-bound survival.',
      description_zh: '运行契约绑定的 Cox 分析。', version: 'easyicu.method-skills/1', enabled: true,
      capability_id: 'survival_time_to_event_v1', action_ids: ['time_to_event.cox_hr', 'time_to_event.ph_check'],
      layer: 'research_workflow', included_module_ids: ['cohort-characterization-table-one'],
      not_for: [{ situation_en: 'Exposure changes during follow-up', situation_zh: '暴露在随访期间变化' }],
      execution_mode: 'deterministic_host', claim_ceiling: 'reportable', scope: 'Cox estimand',
      inputs: ['time origin', 'event and censoring'], outputs: ['primary CSV'], diagnostics: ['PH diagnostic'],
      prompt: 'Use survival.', prompt_zh: '请使用生存分析（时间到事件）方法。' },
    { id: 'target-trial-emulation', title: 'Target trial', title_zh: '目标试验模拟',
      category: 'Causal inference', category_zh: '因果推断', description: 'Specify a target trial.',
      description_zh: '明确目标试验方案。', version: 'easyicu.method-skills/1', enabled: true,
      capability_id: 'causal_target_trial_v1', action_ids: [], execution_mode: 'agent_coded_with_host_gates',
      layer: 'research_workflow', included_module_ids: [],
      claim_ceiling: 'analysis_only', scope: 'Causal contrast', inputs: ['time zero'], outputs: ['registered result'],
      diagnostics: ['positivity and balance'], prompt: 'Use target trial.', prompt_zh: '请使用目标试验模拟方法。' },
    { id: 'cohort-characterization-table-one', title: 'Table 1', title_zh: '队列描述与 Table 1',
      category: 'Descriptive', category_zh: '描述性研究', description: 'Reusable baseline table.',
      description_zh: '可复用的基线特征表。', version: 'easyicu.method-skills/1', enabled: true,
      capability_id: 'descriptive_measurement_v1', action_ids: ['descriptive.table_one'],
      layer: 'analysis_module', included_module_ids: [], execution_mode: 'deterministic_host',
      claim_ceiling: 'analysis_only', scope: 'Table 1', inputs: ['group'], outputs: ['table_one.csv'],
      diagnostics: ['denominators'], prompt: 'Use Table 1.', prompt_zh: '请使用 Table 1 模块。' },
  ], components: [
    { id: 'prediction.decision_curve', title: 'Decision-curve analysis', title_zh: '决策曲线与净获益',
      category: 'Prediction', category_zh: '预测模型', description: 'Clinical utility.',
      description_zh: '比较临床阈值范围内的净获益。', version: 'easyicu.method-skills/1', enabled: true,
      method_family: 'prediction', method_key: 'decision_curve', tier: 'standard_supporting',
      implementation: 'llm_coded', execution_mode: 'agent_coded_with_host_gates', claim_ceiling: 'analysis_only',
      outputs: ['decision_curve.csv'], reporting_items: ['TRIPOD+AI 19'], kernel_modules: ['decision_curve'],
      prompt: 'Use decision curve.', prompt_zh: '请使用决策曲线与净获益方法组件。' },
    { id: 'cross_database.meta_analysis', title: 'Per-database meta-analysis', title_zh: '跨数据库效应合并',
      category: 'Evidence synthesis', category_zh: '证据合成', description: 'Pool compatible effects.',
      description_zh: '合并估计目标一致的数据库效应。', enabled: true, method_family: 'cross_database',
      method_key: 'meta_analysis', tier: 'standard_supporting', implementation: 'deterministic',
      execution_mode: 'deterministic_host', claim_ceiling: 'analysis_only', outputs: ['forest.svg'] },
  ], workflow_count: 2, analysis_module_count: 1, available_method_count: 2, planned_method_count: 0 },
} };
const pending = new Map();
global.sessionStorage = { setItem: (key, value) => pending.set(key, value), getItem: key => pending.get(key) || null };
global.location = { hash: '' };
global.requestAnimationFrame = () => {};
global.__euRender = () => {};
require(path.resolve(process.argv[2]));
const screen = global.SCREENS.skills;
let html = screen.render();
// The skills tab lists only what a clinician picks; method components live in the method library.
assert.match(html, /全部 <span>3<\/span>/);
assert.match(html, /EasyICU <span>3<\/span>/);
assert.match(html, /生存分析（时间到事件）/);
assert.match(html, /目标试验模拟/);
assert.match(html, /完整研究 <small>2<\/small>/);
assert.match(html, /单项分析 <small>1<\/small>/);
assert.match(html, /队列描述与 Table 1/);
assert.doesNotMatch(html, /决策曲线与净获益/);
assert.match(html, /data-sk-mode="skills"[^>]*>技能 <span>3<\/span>/);
assert.match(html, /data-sk-mode="methods"[^>]*>模块 <span>2<\/span>/);
assert.match(html, /<span>审查后可用于报告<\/span>/);
assert.match(html, /<span>仅分析级<\/span>/);
assert.doesNotMatch(html, /eusk-catalog-summary|EasyICU 已验证|<small>[^<]*<\/small><\/button>/);

const handlers = {};
const hub = { addEventListener: (name, handler) => { handlers[name] = handler; } };
screen.afterRender({ querySelector: selector => selector === '.eusk-shell' ? hub : null });
const click = dataset => handlers.click({ target: { closest: selector => selector === 'button' ? { dataset } : null } });
click({ skMode: 'methods' });
html = screen.render();
assert.match(html, /全部 <span>2<\/span>/);
assert.match(html, /EasyICU <span>2<\/span>/);
assert.match(html, /预测模型 <small>1<\/small>/);
assert.match(html, /决策曲线与净获益/);
assert.match(html, /证据合成 <small>1<\/small>/);
assert.match(html, /跨数据库效应合并/);
assert.equal((html.match(/class="eusk-card /g) || []).length, 2);
assert.match(html, /<span>配套方法<\/span>/);
assert.doesNotMatch(html, /生存分析（时间到事件）/);
assert.doesNotMatch(html, /eusk-catalog-summary/);
click({ skOpen: 'builtin:survival-time-to-event' });
html = screen.render();
assert.match(html, /生存分析 · 完整研究/);
assert.match(html, /<span>能力<\/span><code>survival_time_to_event_v1<\/code>/);
assert.match(html, /由 EasyICU 直接计算/);
assert.match(html, /<span>结果用途<\/span><strong>审查后可用于报告<\/strong>/);
assert.match(html, /<h2>输入<\/h2><ul><li>time origin<\/li><li>event and censoring<\/li>/);
assert.match(html, /<h2>何时不适用<\/h2><ul><li>暴露在随访期间变化<\/li>/);
assert.match(html, /以实际运行证据为准/);
assert.doesNotMatch(html, /EasyICU 已验证/);
assert.match(html, /<h2>包含的单项分析<\/h2><ul><li>队列描述与 Table 1<\/li>/);
assert.match(html, /启动工作流/);
assert.doesNotMatch(html, /data-sk-toggle="builtin:survival-time-to-event"/);
click({ skTab: 'files' });
html = screen.render();
assert.match(html, /survival_time_to_event_v1/);
assert.match(html, /time_to_event\.cox_hr/);
assert.match(html, /PH diagnostic/);
click({ skUse: 'builtin:survival-time-to-event' });
assert.equal(global.location.hash, '#guided');
assert.equal(pending.get('easyicu.skillHub.question'), '请使用生存分析（时间到事件）方法。');
assert.deepEqual(JSON.parse(pending.get('easyicu.skillHub.method')), {
  id: 'survival-time-to-event', title: '生存分析（时间到事件）', kind: 'method',
  layer: 'research_workflow',
  capability_id: 'survival_time_to_event_v1', action_ids: ['time_to_event.cox_hr', 'time_to_event.ph_check'],
  included_module_ids: ['cohort-characterization-table-one'],
  method_family: '', method_key: '', claim_ceiling: 'reportable',
});
global.location.hash = '';
click({ skBack: '' });
click({ skOpen: 'builtin:prediction.decision_curve' });
html = screen.render();
assert.match(html, /返回模块/);
assert.match(html, /预测模型 · 模块/);
assert.match(html, /<span>方法族<\/span><strong>预测模型<\/strong>/);
assert.match(html, /模型编写代码，EasyICU 核验/);
click({ skTab: 'files' });
html = screen.render();
assert.match(html, /decision_curve\.csv/);
assert.match(html, /TRIPOD\+AI 19/);
assert.equal((html.match(/<li>decision_curve<\/li>/g) || []).length, 1);
click({ skUse: 'builtin:prediction.decision_curve' });
assert.equal(pending.get('easyicu.skillHub.question'), '请使用决策曲线与净获益方法组件。');
assert.deepEqual(JSON.parse(pending.get('easyicu.skillHub.method')), {
  id: 'prediction.decision_curve', title: '决策曲线与净获益', kind: 'method_component',
  layer: 'method_component',
  capability_id: '', action_ids: [], method_family: 'prediction', method_key: 'decision_curve',
  included_module_ids: [],
  claim_ceiling: 'analysis_only',
});
// Publication skills have an internal source, without a research acceptance claim.
global.EU_CAPABILITIES.capabilities.publication_skills.items.push({
  id: 'nature-writing', stage: 'Writing', enabled: true, version: 'test',
});
click({ skOpen: 'builtin:nature-writing' });
html = screen.render();
assert.match(html, /写作 · EasyICU 内置/);
assert.match(html, /以实际运行证据为准/);
assert.doesNotMatch(html, /EasyICU 已验证/);
process.stdout.write('Method Skill catalogue, contract and task handoff passed.\n');
