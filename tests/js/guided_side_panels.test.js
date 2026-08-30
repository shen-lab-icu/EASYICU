'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
const localStorageValues = new Map();
global.localStorage = {
  getItem(key) { return localStorageValues.has(key) ? localStorageValues.get(key) : null; },
  setItem(key, value) { localStorageValues.set(key, String(value)); },
};

require(path.resolve(process.argv[2]));

const classes = new Set();
const main = {
  classList: {
    toggle(name, active) { if (active) classes.add(name); else classes.delete(name); },
  },
};
const ctx = { t: en => en, icon: () => '' };
const owner = global.EU_GUIDED_PANELS;

assert.match(owner.renderContextAsideRestore(ctx), /data-context-aside-toggle/);
assert.match(owner.renderContextAsideCollapse(ctx), /aria-controls="gdContextAside"/);
assert.equal(owner.isContextAsideCollapsed(), false);
owner.setContextAsideCollapsed(true, main);
assert.equal(owner.isContextAsideCollapsed(), true);
assert.equal(owner.contextAsideClass(), 'gd-context-aside-collapsed');
assert.equal(localStorageValues.get('easyicu.guided.contextAsideCollapsed.v1'), '1');
assert.equal(classes.has('gd-context-aside-collapsed'), true);
owner.setContextAsideCollapsed(false, main);
assert.equal(owner.isContextAsideCollapsed(), false);
assert.equal(classes.has('gd-context-aside-collapsed'), false);

process.stdout.write('{"bilateralCollapse":true,"previewCanExpand":true}');
