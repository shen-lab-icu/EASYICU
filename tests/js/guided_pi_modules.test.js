'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');

function load(window = {}) {
  const context = { window };
  vm.createContext(context);
  vm.runInContext(source, context);
  return context.window.EasyICU.guidedPi;
}

const modules = load();
const api = { render: () => 'ok' };
assert.equal(modules.declare('preview', api), api);
assert.equal(modules.require('preview').render(), 'ok');
assert.equal(modules.optional('missing'), null);
assert.throws(() => modules.require('missing'), /is not declared: missing/);
assert.throws(() => modules.declare('preview', {}), /already declared: preview/);
assert.throws(() => modules.declare('invalid-name', {}), /non-empty identifier/);
assert.throws(() => modules.declare('emptyApi', null), /API must be an object/);
assert.throws(() => load({ EasyICU: { guidedPi: {} } }), /namespace already exists/);
assert.equal(Object.isFrozen(modules.require('preview')), true);

process.stdout.write(JSON.stringify({ ok: true, fail_closed_cases: 5 }));
