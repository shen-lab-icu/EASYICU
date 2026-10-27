'use strict';

// The connection decides which models exist; the menu lists newer versions
// first and keeps an unversioned name after them, in the connection's order.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const owner = process.argv[2];
const declared = {};
const context = vm.createContext({
  window: { EU_HTML: { esc: value => String(value) }, EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } } },
});
vm.runInContext(fs.readFileSync(owner, 'utf8'), context);
const { newestFirst } = declared.modelMenu;

assert.deepEqual(
  [...newestFirst(['codex-auto-review', 'gpt-5.3-codex-spark', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-6.0'])],
  ['gpt-6.0', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.3-codex-spark', 'codex-auto-review'],
);
assert.deepEqual([...newestFirst(['model-a', 'model-b'])], ['model-a', 'model-b']);
assert.deepEqual([...newestFirst(['gpt-4o', 'o3', 'claude-sonnet-5-5'])], ['claude-sonnet-5-5', 'gpt-4o', 'o3']);
assert.deepEqual([...newestFirst([])], []);
