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
  }], components: [], workflow_count: 1, available_method_count: 0, planned_method_count: 0 },
} };
global.EU_API = {
  loadBuiltinSkillPackage: async id => ({
    schema_version: 'easyicu.method-skill-package/3', skill_id: id,
    package_sha256: 'b'.repeat(64), read_only: true,
    files: [
      { path: 'SKILL.md', language: 'markdown', size_bytes: 240, content: '---\nname: survival\ndescription: machine metadata\n---\n# Survival\nUse carefully.\n\n## When to use this workflow\nConfirm time origin.\n\n## Validation and quality checks\nCheck proportional hazards.' },
      { path: 'references/workflow_contract.md', language: 'markdown', size_bytes: 32, content: '# Workflow contract\n' },
    ],
  }),
};
global.sessionStorage = { setItem: () => {}, getItem: () => null };
global.location = { hash: '' };
global.requestAnimationFrame = () => {};
global.__euRender = () => {};

require(path.resolve(process.argv[2]));
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
  assert.match(html, /data-sk-overview-source="SKILL\.md"/);
  assert.match(html, /eusk-overview-document/);
  assert.match(html, /<h1>Survival<\/h1>/);
  assert.match(html, /When to use this workflow/);
  assert.match(html, /Validation and quality checks/);
  assert.doesNotMatch(html, /machine metadata/);
  assert.doesNotMatch(html, /只读内置技能包/);
  click({ skTab: 'files' });
  html = screen.render();
  assert.match(html, /只读内置技能包/);
  assert.match(html, /SKILL\.md/);
  assert.match(html, /references/);
  assert.doesNotMatch(html, /scripts\//);
  assert.match(html, /workflow_contract\.md/);
  assert.doesNotMatch(html, /adapter\.py/);
  assert.match(html, /技能包文件 <span>2<\/span>/);
  assert.match(html, /eusk-rendered-markdown/);
  assert.match(html, /<h1>Survival<\/h1>/);
  assert.doesNotMatch(html, /<pre># Survival/);
  assert.doesNotMatch(html, /machine metadata/);
  assert.match(html, /只有该能力确有专用实现时才显示 scripts/);
  process.stdout.write('Built-in method Skill package file view passed.\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
