'use strict';
const assert = require('node:assert/strict');
global.window = global;
global.t = (_en, zh) => zh;
global.EU_LANG = 'zh';
global.icon = () => '';
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.SCREENS = {};
global.EU_CAPABILITIES = { capabilities: {} };
global.EU_EXTENSIONS = { activation_sha256: 'a'.repeat(64), skills: [] };
global.location = { hash: '', pathname: '/', search: '' };
global.sessionStorage = { getItem: () => null, setItem() {} };
global.requestAnimationFrame = () => {};
global.__euRender = () => {};
const saved = [];
const content = '---\nname: audit-skill\ndescription: Example.\n---\nCheck the report.\n';
global.EU_API = {
  installExtensionSkill: async body => {
    saved.push(body);
    const skill = { name: 'audit-skill', stages: body.stages, enabled: body.enabled };
    global.EU_EXTENSIONS.skills = [skill];
    return { skill };
  },
  loadExtensionSkill: async () => ({ name: 'audit-skill', skill_md: content, stages: ['writing'], enabled: true }),
};
require('../../src/easyicu/webserver/static/js/screens-skills.js');
const screen = global.SCREENS.skills;
const handlers = {};
screen.afterRender({ querySelector: () => ({ addEventListener: (event, handler) => { handlers[event] = handler; } }) });
const click = dataset => handlers.click({ target: { closest: () => ({ dataset }) } });
const paste = () => handlers.input({ target: { matches: selector => selector === '[data-sk-draft]', value: content } });
const settle = async () => { for (let i = 0; i < 12; i += 1) await new Promise(resolve => setImmediate(resolve)); };

(async () => {
  click({ skUpload: '' });
  assert.doesNotMatch(screen.render(), /data-sk-install-enabled checked/);
  paste(); click({ skInstall: '' }); await settle();
  assert.equal(saved[0].enabled, false, 'pasted skill should not activate before review');

  // Editing preserves the old state, but opening a fresh import must not inherit it.
  click({ skEdit: '' }); await settle();
  assert.match(screen.render(), /data-sk-install-enabled checked/);
  click({ skBack: '' }); click({ skUpload: '' });
  assert.doesNotMatch(screen.render(), /data-sk-install-enabled checked/);
  paste(); click({ skInstall: '' }); await settle();
  assert.equal(saved[1].enabled, false);
  assert.deepEqual(saved[1].stages, ['conversation']);

  // Users can explicitly opt in when installing a reviewed skill.
  click({ skBack: '' }); click({ skUpload: '' });
  handlers.change({ target: { matches: selector => selector === '[data-sk-install-enabled]', checked: true } });
  paste(); click({ skInstall: '' }); await settle();
  assert.equal(saved[2].enabled, true);
  process.stdout.write('Skill imports stay disabled until explicitly enabled.\n');
})().catch(error => { console.error(error); process.exit(1); });
