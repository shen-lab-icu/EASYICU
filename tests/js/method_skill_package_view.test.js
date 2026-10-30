/* Built-in method details load and switch a reviewed documentation package. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'zh';
global.t = (_en, zh) => zh;
global.icon = () => '';
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.SCREENS = {};
const sourcePath = 'src/easyicu/research_agent/methods/survival.py';
const sourceDigest = 'd'.repeat(64);
global.EU_EXTENSIONS = { activation_sha256: 'a'.repeat(64), skills: [] };
global.EasyICU = { guidedPi: { optional: name => name === 'markdown' ? {
  render: text => `<article class="rendered-doc">${String(text).replace(/^# (.+)$/m, '<h1>$1</h1>')}</article>`,
} : null } };
global.EU_CAPABILITIES = { capabilities: {
  publication_skills: { enabled: true, items: [] },
  method_skills: { enabled: true, items: [{
    id: 'survival-time-to-event', title: 'Survival', title_zh: '生存与时间结局分析',
    category: 'Survival', category_zh: '生存分析', description: 'Survival workflow.',
    description_zh: '生存流程。', version: 'easyicu.method-skills/1', enabled: true,
    capability_id: 'survival_time_to_event_v1', action_ids: ['time_to_event.cox_hr'],
    layer: 'research_workflow', included_module_ids: ['cohort-characterization-table-one'],
    execution_mode: 'deterministic_host', claim_ceiling: 'reportable', scope: 'Cox',
    inputs: ['time origin'], outputs: ['result.csv'], diagnostics: ['PH'],
    prompt: 'Use survival.', prompt_zh: '请使用生存方法。',
  }, {
    id: 'landmark-association', title: 'Landmark association', title_zh: '固定 Landmark 关联研究',
    category: 'Association', category_zh: '关联研究', description: 'Landmark.', description_zh: '固定时点。',
    version: 'easyicu.method-skills/1', enabled: true, capability_id: 'landmark_v1', action_ids: [],
    layer: 'research_workflow', included_module_ids: [], execution_mode: 'deterministic_host', claim_ceiling: 'reportable',
  }], components: [], workflow_count: 1, available_method_count: 0, planned_method_count: 0 },
} };
global.EU_API = {
  loadBuiltinSkillPackage: async id => id === 'landmark-association' ? {
    schema_version: 'easyicu.method-skill-package/3', skill_id: id, package_sha256: 'c'.repeat(64), read_only: true,
    overview_paths: { en: 'SKILL.md', zh: 'SKILL.zh.md' },
    implementation: { mode: 'unavailable', entrypoints: [], examples: [] },
    not_for: [
      { situation_en: 'Exposure changes during follow-up', situation_zh: '暴露在随访期间变化', sibling_skill_id: 'survival-time-to-event' },
      { situation_en: 'Confounding is not recorded', situation_zh: '混杂信息不足', sibling_skill_id: null },
    ],
    files: [
      { path: 'SKILL.md', language: 'markdown', size_bytes: 40, content: '# Landmark association\n\n## When not to use\n\n- Exposure changes during follow-up → Survival\n' },
      { path: 'SKILL.zh.md', language: 'markdown', size_bytes: 40, content: '# 固定 Landmark 关联研究\n\n## 何时不适用\n\n- 暴露在随访期间变化 → 生存与时间结局分析\n- 混杂信息不足\n' },
    ],
  } : ({
    schema_version: 'easyicu.method-skill-package/3', skill_id: id,
    package_sha256: 'b'.repeat(64), read_only: true,
    overview_paths: { en: 'SKILL.md', zh: 'SKILL.zh.md' },
    implementation: { mode: 'fixed_host', summary_zh: '通过固定的 Cox 模型函数估计风险比。', entrypoints: [{
      name: 'fit_cox', qualname: 'fit_cox', module: 'easyicu.research_agent.methods.survival',
      signature: 'def fit_cox(data: DataFrame, spec: CoxSpec) -> CoxResult', role: 'entrypoint',
      source_path: sourcePath, source_sha256: sourceDigest, line_start: 120, line_end: 121,
      file_path: 'implementation/research_agent.methods.survival.py', excerpt_line_start: 7, excerpt_line_end: 8,
      docstring: 'Fit a reviewed Cox model and return hazard ratios.',
    }], examples: [] },
    files: [
      { path: 'SKILL.md', language: 'markdown', size_bytes: 240, content: '---\nname: survival\ndescription: machine metadata\n---\n# Survival\nUse carefully.\n\n## When to use this workflow\nConfirm time origin.\n\n## Validation and quality checks\nCheck proportional hazards.' },
      { path: 'SKILL.zh.md', language: 'markdown', size_bytes: 240, content: '---\nname: survival\ndescription: machine metadata\n---\n# 生存分析说明\n\n## 输入\n每位患者的随访时间、事件状态和协变量。\n\n## 输出\n- `cox_estimates.csv`\n\n## 必做检查\n完整的比例风险诊断说明。' },
      { path: 'references/workflow_contract.md', language: 'markdown', size_bytes: 32, content: '# Workflow contract\n' },
      { path: 'implementation/research_agent.methods.survival.py', role: 'host_implementation',
        source_path: sourcePath, source_sha256: sourceDigest, language: 'python', size_bytes: 350,
        content: `# ${sourcePath}:10-11 sha256=${sourceDigest}\ndef internal_helper():\n    return None\n\n\n# ${sourcePath}:120-121 sha256=${sourceDigest}\ndef fit_cox(data, spec):\n    return model.fit(data, spec)` },
    ],
  }),
};
const pending = new Map();
global.sessionStorage = { setItem: (key, value) => pending.set(key, value), getItem: key => pending.get(key) || null };
global.location = { hash: '' };
global.requestAnimationFrame = () => {};
global.__euRender = () => {};

require(path.resolve(process.argv[2] || 'src/easyicu/webserver/static/js/screens-skills-implementation.js'));
require(path.resolve(process.argv[3] || 'src/easyicu/webserver/static/js/screens-skills.js'));
const screen = global.SCREENS.skills;
const handlers = {};
const hub = { addEventListener: (name, handler) => { handlers[name] = handler; } };
screen.afterRender({ querySelector: selector => selector === '.eusk-shell' ? hub : null });
const click = dataset => handlers.click({ target: { closest: selector => selector === 'button' ? { dataset } : null } });

(async () => {
  click({ skOpen: 'builtin:survival-time-to-event' });
  assert.match(screen.render(), /正在读取已审阅的 SKILL\.md/);
  await new Promise(resolve => setImmediate(resolve));
  let html = screen.render();
  assert.match(html, /data-sk-overview-source="SKILL\.zh\.md"/);
  assert.match(html, /所需输入/);
  assert.match(html, /每位研究对象的随访时长、事件或删失状态/);
  assert.match(html, /survival_effect_estimates\.csv/);
  assert.match(html, /固定内置实现/);
  assert.doesNotMatch(html, /data-sk-function="0"|def fit_cox\(data: DataFrame/);
  assert.doesNotMatch(html, /完整的比例风险诊断说明/);
  assert.doesNotMatch(html, /machine metadata/);
  assert.doesNotMatch(html, /只读内置技能包/);
  click({ skTab: 'files' });
  html = screen.render();
  assert.match(html, /只读内置技能包/);
  assert.match(html, /SKILL\.md/);
  assert.match(html, /<h2>SKILL\.zh\.md/);
  assert.match(html, /references/);
  assert.doesNotMatch(html, /scripts\//);
  assert.match(html, /workflow_contract\.md/);
  assert.doesNotMatch(html, /adapter\.py/);
  assert.match(html, /技能包文件 <span>4<\/span>/);
  assert.match(html, /eusk-rendered-markdown/);
  assert.match(html, /<h1>生存分析说明<\/h1>/);
  assert.doesNotMatch(html, /<pre># Survival/);
  assert.doesNotMatch(html, /machine metadata/);
  assert.match(html, /implementation 保存只读源码摘录/);
  assert.match(html, /完整的比例风险诊断说明/);

  // The implementation tab opens the exact excerpt, not the helper at the top of its file.
  click({ skTab: 'overview' });
  click({ skTab: 'implementation' });
  html = screen.render();
  assert.match(html, /data-sk-tab="implementation"/);
  assert.match(html, /def fit_cox\(data: DataFrame, spec: CoxSpec\) -> CoxResult/);
  assert.match(html, /data-source-line="120"/);
  assert.match(html, /data-source-line="121"/);
  assert.doesNotMatch(html, /internal_helper/);
  assert.match(html, /survival\.py:120–121/);
  click({ skPackageFile: 'implementation/research_agent.methods.survival.py' });
  html = screen.render();
  assert.match(html, /此文件不能作为独立脚本运行/);
  assert.match(html, /internal_helper/);
  assert.match(html, /data-source-line="10"/);
  assert.match(html, /data-source-line="120"/);

  // A bilingual package opens on the reader's language, and a sibling it names
  // opens that Skill; a line without a sibling stays text.
  click({ skOpen: 'builtin:landmark-association' });
  await new Promise(resolve => setImmediate(resolve));
  html = screen.render();
  assert.match(html, /data-sk-overview-source="SKILL\.zh\.md"/);
  assert.match(html, /暂未提供函数元数据/);
  assert.match(html, /aria-label="实现待补齐">实现待补齐<\/span>/);
  assert.match(html, /data-sk-use="builtin:landmark-association" >在新任务中讨论/);
  assert.doesNotMatch(html, /✓ 可用/);
  assert.equal(global.EU_CAPABILITIES.capabilities.method_skills.items[1].enabled, true);
  assert.match(html, /- 暴露在随访期间变化 → <button type="button" class="eusk-sibling" data-sk-open="builtin:survival-time-to-event">生存与时间结局分析<\/button>/);
  assert.match(html, /- 混杂信息不足/);
  assert.doesNotMatch(html, /EUSKSIBLING/);
  click({ skTab: 'files' });
  html = screen.render();
  assert.match(html, /<h2>SKILL\.zh\.md/);
  assert.match(html, /<h1>固定 Landmark 关联研究<\/h1>/);
  click({ skUse: 'builtin:landmark-association' });
  assert.equal(global.location.hash, '#guided');
  assert.equal(JSON.parse(pending.get('easyicu.skillHub.method')).id, 'landmark-association');
  // Reopening a Skill discards the previous file choice and again selects its language.
  click({ skOpen: 'builtin:survival-time-to-event' });
  await new Promise(resolve => setImmediate(resolve));
  click({ skTab: 'files' });
  assert.match(screen.render(), /<h2>SKILL\.zh\.md/);
  process.stdout.write('Built-in method Skill package file view passed.\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
