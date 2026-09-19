/* Large-catalog Skill Hub contract: truthfully render the actual registry. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
global.window = global;
global.t = en => en;
global.icon = () => '';
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.SCREENS = {};
global.EU_CAPABILITIES = { capabilities: { publication_skills: { enabled: true, items: [] } } };
global.EU_EXTENSIONS = { activation_sha256: 'a'.repeat(64), skills: Array.from({ length: 81 }, (_, i) => ({
  name: `skill-${String(i + 1).padStart(2, '0')}`,
  description: `A reviewed workflow for topic ${i + 1}.`,
  category: i % 2 ? 'Clinical Research' : 'Literature',
  digest: String(i % 10).repeat(64), stages: ['conversation'], enabled: i % 3 !== 0,
})) };
const pendingHubIntent = new Map();
global.sessionStorage = { setItem: (key, value) => pendingHubIntent.set(key, value),
  getItem: key => pendingHubIntent.get(key) || null };
global.location = { hash: '' };
require(path.resolve(process.argv[2]));
const screen = global.SCREENS.skills;
let html = screen.render();
assert.match(html, /All <span>81<\/span>/);
assert.match(html, /Mine <span>81<\/span>/);
assert.match(html, /data-sk-mode="skills"[^>]*>Skills <span>81<\/span>/);
assert.match(html, /data-sk-mode="methods"[^>]*>Method library <span>0<\/span>/);
assert.equal((html.match(/class="eusk-card /g) || []).length, 81);
assert.match(html, /Upload SKILL\.md/);
assert.match(html, /Create with EasyICU/);
const handlers = {};
const hub = { addEventListener: (name, handler) => { handlers[name] = handler; } };
global.__euRender = () => {};
global.requestAnimationFrame = () => {};
screen.afterRender({ querySelector: selector => selector === '.eusk-shell' ? hub : null });
handlers.input({ target: { matches: selector => selector === '[data-sk-search]', value: 'skill-42' } });
html = screen.render();
assert.match(html, /skill-42/);
assert.doesNotMatch(html, /skill-43/);
handlers.input({ target: { matches: selector => selector === '[data-sk-search]', value: '' } });
handlers.change({ target: { matches: selector => selector === '[data-sk-hide]', checked: true } });
html = screen.render();
assert.equal((html.match(/class="eusk-card /g) || []).length, 54);
handlers.change({ target: { matches: selector => selector === '[data-sk-category]', value: 'Clinical Research' } });
html = screen.render();
assert.equal((html.match(/class="eusk-card /g) || []).length, 27);
const click = dataset => handlers.click({ target: { closest: selector => selector === 'button' ? { dataset } : null } });
click({ skUpload: '' });
assert.match(screen.render(), /Upload a skill/);
assert.match(screen.render(), /Supporting-file packages are not yet accepted/);
click({ skCancel: '' });
click({ skCreateWith: '' });
assert.equal(global.location.hash, '#guided');
assert.match(pendingHubIntent.get('easyicu.skillHub.question'), /reusable EasyICU Skill/);
assert.equal(JSON.parse(pendingHubIntent.get('easyicu.skillHub.builder')).mode, 'create');
process.stdout.write('81-item Skill Hub search and enabled filters passed.\n');
