/* A built-in Skill is copied into a personal, instructions-only Skill that names its source. */
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
      category: 'Survival', category_zh: '生存分析', description: 'Compare survival.',
      description_zh: '比较不同分组的生存。', version: 'easyicu.method-skills/3', enabled: true,
      capability_id: 'survival_time_to_event_v1', action_ids: ['time_to_event.cox_hr'],
      layer: 'research_workflow', included_module_ids: [], execution_mode: 'deterministic_host',
      claim_ceiling: 'reportable', outputs: ['primary CSV'], diagnostics: ['PH diagnostic'],
      prompt: 'Use survival.', prompt_zh: '请使用生存分析方法。' },
  ], components: [] },
} };
global.location = { hash: '', pathname: '/', search: '' };
global.history = { pushState: (_state, _title, url) => { global.location.hash = String(url).replace(/^[^#]*/, ''); } };
global.requestAnimationFrame = () => {};
global.__euRender = () => {};
global.sessionStorage = { setItem: () => {}, getItem: () => null };

const copied = (name, sourceUpdated) => ({
  name, category: '', description: '比较不同分组的生存。', digest_sha256: 'c'.repeat(64), size_bytes: 900,
  stages: ['conversation'], updated_at: '2026-10-26T10:00:00Z', enabled: false,
  execution_boundary: 'instructions_only', claim_ceiling: 'analysis_only',
  source: { builtin_id: 'survival-time-to-event', registry_version: 'easyicu.method-skills/3',
    package_sha256: 'd'.repeat(64), copied_at: '2026-10-26T10:00:00Z', omitted_reference_paths: [] },
  source_updated: sourceUpdated, source_available: true,
});
const posts = [];
global.EU_API = {
  loadBuiltinSkillPackage: async () => { throw new Error('offline'); },
  loadExtensionSkill: async name => ({ name, skill_md: '---\nname: x\n---\n' }),
  loadExtensions: async () => global.EU_EXTENSIONS,
  postJSON: async (url, body) => {
    posts.push([url, body]);
    if (body.name === 'survival-time-to-event') {
      throw Object.assign(new Error('taken'), { code: 'extension_skill_name_conflict' });
    }
    const skill = copied(body.name, false);
    return { ok: true, skill, extensions: { activation_sha256: 'b'.repeat(64), skills: [skill] } };
  },
};
require(path.resolve(process.argv[2]));
const screen = global.SCREENS.skills;
const handlers = {};
screen.afterRender({ querySelector: selector => selector === '.eusk-shell' ? { addEventListener: (name, handler) => { handlers[name] = handler; } } : null });
const click = dataset => handlers.click({ target: { closest: selector => selector === 'button' ? { dataset } : null } });
const settle = async () => { for (let i = 0; i < 12; i += 1) await new Promise(resolve => setImmediate(resolve)); };

(async () => {
  click({ skOpen: 'builtin:survival-time-to-event' });
  await settle();
  let html = screen.render();
  assert.match(html, /data-sk-copy="builtin:survival-time-to-event"[^>]*>复制为我的技能<\/button>/);

  click({ skCopy: 'builtin:survival-time-to-event' });
  await settle();
  // A taken name moves to the next free suffix; the copy is requested against the revision the page saw.
  assert.deepEqual(posts.map(([url, body]) => [url, body.builtin_id, body.name, body.expected_sha256]), [
    ['/api/extensions/skills/copy', 'survival-time-to-event', 'survival-time-to-event', 'a'.repeat(64)],
    ['/api/extensions/skills/copy', 'survival-time-to-event', 'survival-time-to-event-2', 'a'.repeat(64)],
  ]);
  assert.ok(posts.every(([, body]) => !('enabled' in body)));
  html = screen.render();
  assert.match(html, /<h1>survival-time-to-event-2<\/h1>/);
  assert.match(html, /复制自「生存分析（时间到事件）」/);
  assert.match(html, /副本默认未启用/);
  assert.doesNotMatch(html, /data-sk-copy=/);
  assert.doesNotMatch(html, /eusk-source-note/);

  click({ skBack: '' });
  html = screen.render();
  assert.match(html, /<span>复制自「生存分析（时间到事件）」<\/span>/);

  // A later change to the built-in is reported, never merged into the copy.
  global.EU_EXTENSIONS = { activation_sha256: 'b'.repeat(64), skills: [copied('survival-time-to-event-2', true)] };
  html = screen.render();
  assert.match(html, /<span>复制自「生存分析（时间到事件）」 · 原技能已更新<\/span>/);
  click({ skOpen: 'user:survival-time-to-event-2' });
  await settle();
  html = screen.render();
  assert.match(html, /class="eusk-source-note">原技能已更新，你的副本保持不变。 <button type="button" data-sk-open="builtin:survival-time-to-event">查看原技能<\/button>/);

  // A hosted instance's skills belong to its administrator: no copy action.
  global.EasyICU = { guidedPi: { optional: name => name === 'hosted' ? { active: () => true } : null } };
  click({ skOpen: 'builtin:survival-time-to-event' });
  await settle();
  assert.doesNotMatch(screen.render(), /data-sk-copy=/);
  process.stdout.write('Built-in Skill copy, provenance and source-updated note passed.\n');
})().catch(error => { console.error(error); process.exit(1); });
