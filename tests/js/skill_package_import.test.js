'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
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
const sent = [];
const main = '---\nname: synthetic-upload\ndescription: Synthetic guidance.\n---\nRead references/a.md.\n';
let detail = { name: 'synthetic-upload', skill_md: main, instructions: 'Synthetic guidance.', reference_files: { 'references/a.md': 'Old reference.' }, stages: ['conversation'], enabled: false };
const skill = { ...detail, description: 'Synthetic guidance.', digest: 'b'.repeat(64) };
const result = () => { global.EU_EXTENSIONS.skills = [skill]; return { skill }; };
global.EU_API = {
  importExtensionSkillPackage: async body => { sent.push(['import', body]); return result(); },
  installExtensionSkill: async body => { sent.push(['edit', body]); return result(); },
  loadExtensionSkill: async () => detail,
};
const downloads = [];
global.document = { body: { append() {} }, createElement: () => ({ click() { downloads.push([this.href, this.download]); }, remove() {} }) };
require(path.resolve(__dirname, '../../src/easyicu/webserver/static/js/screens-skills.js'));
const screen = global.SCREENS.skills;
const handlers = {};
screen.afterRender({ querySelector: () => ({ addEventListener: (event, handler) => { handlers[event] = handler; } }) });
const click = dataset => handlers.click({ target: { closest: selector => selector === 'button' ? { dataset } : null } });
const pick = (selector, files) => handlers.change({ target: { matches: value => value === selector, files } });
const settle = async () => { for (let i = 0; i < 12; i += 1) await new Promise(resolve => setImmediate(resolve)); };

(async () => {
  // Hosted users have the same personal importer; built-ins still have no edit action.
  global.EasyICU = { guidedPi: { optional: name => name === 'hosted' ? { active: () => true } : null } };
  assert.match(screen.render(), /data-sk-upload/);
  click({ skUpload: '' });
  assert.match(screen.render(), /data-sk-folder/);
  pick('[data-sk-folder]', [
    { name: 'SKILL.md', webkitRelativePath: 'local/SKILL.md', size: main.length, text: async () => main },
    { name: 'a.md', webkitRelativePath: 'local/references/a.md', size: 10, text: async () => 'Reference.' },
  ]);
  await settle();
  click({ skInstall: '' });
  await settle();
  assert.equal(sent[0][0], 'import');
  assert.deepEqual(sent[0][1].files, { 'local/SKILL.md': main, 'local/references/a.md': 'Reference.' });
  assert.equal(sent[0][1].expected_sha256, 'a'.repeat(64));
  assert.equal(sent[0][1].enabled, false);

  click({ skEdit: '' });
  await settle();
  assert.match(screen.render(), /data-sk-reference="references\/a.md"/);
  handlers.input({ target: { matches: value => value === '[data-sk-reference]', dataset: { skReference: 'references/a.md' }, value: 'Edited reference.' } });
  click({ skInstall: '' });
  await settle();
  assert.deepEqual(sent[1][1].reference_files, { 'references/a.md': 'Edited reference.' });
  click({ skDownload: '' });
  await settle();
  assert.deepEqual(downloads, [['/api/extensions/skills/synthetic-upload/package.zip', 'synthetic-upload.zip']]);

  click({ skBack: '' }); click({ skUpload: '' });
  pick('[data-sk-file]', [{ name: 'local.zip', size: 3, arrayBuffer: async () => Uint8Array.from([80, 75, 3]).buffer }]);
  await settle(); click({ skInstall: '' }); await settle();
  assert.equal(sent[2][1].archive_base64, 'UEsD');
  assert.equal('path' in sent[2][1], false);

  click({ skBack: '' }); click({ skUpload: '' });
  pick('[data-sk-file]', [{ name: 'large.zip', size: 256001 }]);
  await settle();
  assert.match(screen.render(), /ZIP 超过 256 KB/);
  assert.equal(sent.length, 3);
  process.stdout.write('Skill folder/ZIP upload, reference editing, export and hosted access passed.\n');
})().catch(error => { console.error(error); process.exit(1); });
