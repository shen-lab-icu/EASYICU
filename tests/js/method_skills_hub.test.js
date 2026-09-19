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
    { id: 'survival-time-to-event', title: 'Survival', title_zh: '生存与时间结局分析',
      category: 'Survival analysis', category_zh: '生存分析', description: 'Contract-bound survival.',
      description_zh: '运行契约绑定的 Cox 分析。', version: 'easyicu.method-skills/1', enabled: true,
      capability_id: 'survival_time_to_event_v1', action_ids: ['time_to_event.cox_hr', 'time_to_event.ph_check'],
      execution_mode: 'deterministic_host', claim_ceiling: 'reportable', scope: 'Cox estimand',
      inputs: ['time origin', 'event and censoring'], outputs: ['primary CSV'], diagnostics: ['PH diagnostic'],
      prompt: 'Use survival.', prompt_zh: '请使用生存与时间结局分析方法。' },
    { id: 'target-trial-emulation', title: 'Target trial', title_zh: '目标试验模拟',
      category: 'Causal inference', category_zh: '因果推断', description: 'Specify a target trial.',
      description_zh: '明确目标试验方案。', version: 'easyicu.method-skills/1', enabled: true,
      capability_id: 'causal_target_trial_v1', action_ids: [], execution_mode: 'agent_coded_with_host_gates',
      claim_ceiling: 'analysis_only', scope: 'Causal contrast', inputs: ['time zero'], outputs: ['registered result'],
      diagnostics: ['positivity and balance'], prompt: 'Use target trial.', prompt_zh: '请使用目标试验模拟方法。' },
  ], components: [
    { id: 'prediction.decision_curve', title: 'Decision-curve analysis', title_zh: '决策曲线与净获益',
      category: 'Prediction / risk modelling', category_zh: '预测建模', description: 'Clinical utility.',
      description_zh: '比较临床阈值范围内的净获益。', version: 'easyicu.method-skills/1', enabled: true,
      method_family: 'prediction', method_key: 'decision_curve', tier: 'standard_supporting',
      implementation: 'llm_coded', execution_mode: 'agent_coded_with_host_gates', claim_ceiling: 'analysis_only',
      outputs: ['decision_curve.csv'], reporting_items: ['TRIPOD+AI 19'], kernel_modules: ['decision_curve'],
      prompt: 'Use decision curve.', prompt_zh: '请使用决策曲线与净获益方法组件。' },
  ], workflow_count: 2, available_method_count: 1, planned_method_count: 0 },
} };
const pending = new Map();
global.sessionStorage = { setItem: (key, value) => pending.set(key, value), getItem: key => pending.get(key) || null };
global.location = { hash: '' };
global.requestAnimationFrame = () => {};
global.__euRender = () => {};
require(path.resolve(process.argv[2]));
const screen = global.SCREENS.skills;
let html = screen.render();
assert.match(html, /全部 <span>2<\/span>/);
assert.match(html, /EasyICU <span>2<\/span>/);
assert.match(html, /生存与时间结局分析/);
assert.match(html, /目标试验模拟/);
assert.match(html, /研究工作流 <small>2<\/small>/);
assert.match(html, /data-sk-mode="skills"[^>]*>技能 <span>2<\/span>/);
assert.match(html, /data-sk-mode="methods"[^>]*>方法库 <span>1<\/span>/);
assert.doesNotMatch(html, /决策曲线与净获益/);
assert.match(html, /2 个研究工作流与 0 个写作、图件技能/);
assert.match(html, /1 个分析方法在“方法库”中单独浏览/);

const handlers = {};
const hub = { addEventListener: (name, handler) => { handlers[name] = handler; } };
screen.afterRender({ querySelector: selector => selector === '.eusk-shell' ? hub : null });
const click = dataset => handlers.click({ target: { closest: selector => selector === 'button' ? { dataset } : null } });
click({ skMode: 'methods' });
html = screen.render();
assert.match(html, /全部 <span>1<\/span>/);
assert.match(html, /EasyICU <span>1<\/span>/);
assert.match(html, /预测与验证 <small>1<\/small>/);
assert.match(html, /决策曲线与净获益/);
assert.doesNotMatch(html, /生存与时间结局分析/);
assert.match(html, /1 个可用方法，分为六个方法族/);
click({ skOpen: 'builtin:survival-time-to-event' });
html = screen.render();
assert.match(html, /能力契约/);
assert.match(html, /在新任务中使用/);
assert.match(html, /可报告契约/);
assert.doesNotMatch(html, /data-sk-toggle="builtin:survival-time-to-event"/);
click({ skTab: 'files' });
html = screen.render();
assert.match(html, /survival_time_to_event_v1/);
assert.match(html, /time_to_event\.cox_hr/);
assert.match(html, /PH diagnostic/);
click({ skUse: 'builtin:survival-time-to-event' });
assert.equal(global.location.hash, '#guided');
assert.equal(pending.get('easyicu.skillHub.question'), '请使用生存与时间结局分析方法。');
assert.deepEqual(JSON.parse(pending.get('easyicu.skillHub.method')), {
  id: 'survival-time-to-event', title: '生存与时间结局分析', kind: 'method',
  capability_id: 'survival_time_to_event_v1', action_ids: ['time_to_event.cox_hr', 'time_to_event.ph_check'],
  method_family: '', method_key: '', claim_ceiling: 'reportable',
});
global.location.hash = '';
click({ skBack: '' });
click({ skOpen: 'builtin:prediction.decision_curve' });
html = screen.render();
assert.match(html, /方法坐标/);
assert.match(html, /prediction\.decision_curve/);
assert.match(html, /Agent 编码 \+ 主机门禁/);
click({ skTab: 'files' });
html = screen.render();
assert.match(html, /decision_curve\.csv/);
assert.match(html, /TRIPOD\+AI 19/);
assert.equal((html.match(/<li>decision_curve<\/li>/g) || []).length, 1);
click({ skUse: 'builtin:prediction.decision_curve' });
assert.equal(pending.get('easyicu.skillHub.question'), '请使用决策曲线与净获益方法组件。');
assert.deepEqual(JSON.parse(pending.get('easyicu.skillHub.method')), {
  id: 'prediction.decision_curve', title: '决策曲线与净获益', kind: 'method_component',
  capability_id: '', action_ids: [], method_family: 'prediction', method_key: 'decision_curve',
  claim_ceiling: 'analysis_only',
});
process.stdout.write('Method Skill catalogue, contract and task handoff passed.\n');
