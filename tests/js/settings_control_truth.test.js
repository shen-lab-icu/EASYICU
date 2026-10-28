'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function settings() {
  const tabs = {};
  const window = { EU_LANG: 'zh', SCREENS: {}, EU_SETTINGS: {}, EU_CAPABILITIES: { capabilities: {
    mcp_tools: { allowed_tools: [], blocked_tools: [] }, tool_audit: { event_count: 10 },
  } }, EU_DATABASE_NAMES: { keys: () => [], name: x => x } };
  const context = vm.createContext({ window, t: (_en, zh) => zh, icon: () => '', location: {}, document: {} });
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-settings.js'), 'utf8'), context);
  const screen = window.SCREENS.settings;
  screen.afterRender({ querySelector: () => null, querySelectorAll: selector => selector === '[data-settings-cap-tab]'
    ? ['overview', 'skills', 'connectors', 'mcp', 'prompts', 'audit'].map(id => ({
      getAttribute: () => id, addEventListener: (_name, handler) => { tabs[id] = handler; },
    })) : [] });
  return { window, screen, select: id => tabs[id]({ preventDefault() {} }) };
}

test('overview counts only live switches and real MCP tools', () => {
  const { screen } = settings();
  const html = screen.render();
  assert.match(html, />3\/5<\/b>/);
  assert.match(html, /0\/0/);
  assert.match(html, /10/);
  assert.doesNotMatch(html, /4\/7|2\/5/);
});

test('retired switches cannot return as shells; readonly lines lead to the owner', () => {
  const { select, screen } = settings();
  select('prompts');
  let html = screen.render();
  assert.doesNotMatch(html, /data-setting="prompt_contracts_enabled"/);
  assert.match(html, /data-settings-open="guided"/);
  select('audit');
  html = screen.render();
  assert.doesNotMatch(html, /data-setting="remote_compute_enabled"/);
  assert.match(html, /仅本地计算/);
  select('connectors');
  html = screen.render();
  assert.match(html, /PubMed.*OpenAlex/);
  assert.doesNotMatch(html, /Zotero 自动连接/);
});

test('publication status respects the master switch', () => {
  const { select, screen, window } = settings();
  window.EU_SETTINGS.science_skills_enabled = false;
  select('skills');
  const html = screen.render();
  assert.match(html, /0\/2/);
  assert.doesNotMatch(html, /settings-cap-status on/);
});

test('MCP extension state changes carry the reviewed revision', async () => {
  const sent = [];
  let click;
  const window = { EU_LANG: 'zh', EU_HTML: { esc: String }, EU_EXTENSIONS: {
    activation_sha256: 'a'.repeat(64), skills: [], mcp_servers: [],
  }, EU_API: { setExtensionState: async body => { sent.push(body); } } };
  vm.runInNewContext(fs.readFileSync(path.join(JS, 'screens-settings-extensions.js'), 'utf8'), { window });
  const button = { getAttribute: name => ({ 'data-ext-toggle': 'mcp', 'data-ext-name': 'synthetic', 'data-ext-enabled': 'false' })[name], addEventListener: (_event, handler) => { click = handler; } };
  window.EU_SETTINGS_EXTENSIONS.bind({ querySelector: () => null, querySelectorAll: selector => selector === '[data-ext-toggle]' ? [button] : [] });
  await click();
  assert.deepEqual(JSON.parse(JSON.stringify(sent)), [{ kind: 'mcp', name: 'synthetic', enabled: true, expected_sha256: 'a'.repeat(64) }]);
  assert.match(window.EU_SETTINGS_EXTENSIONS.renderSkills(), /href="#skills"/);
  assert.doesNotMatch(window.EU_SETTINGS_EXTENSIONS.renderSkills(), /data-ext-install-skill|data-ext-skill-file/);
});
