'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
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

// D-P3-2: errorText owner contract, including the raw fallback branch.
// The registry test above only covers declare/require; the errorText module
// must stay loadable beside it and keep its fallback returning raw transport
// text verbatim (callers esc() it per D-P2-2).
(function testErrorText() {
  const dir = path.dirname(path.resolve(process.argv[2]));
  let errorTextPath = path.join(dir, 'screens-guided-pi-error-text.js');
  if (!fs.existsSync(errorTextPath)) {
    errorTextPath = path.join(__dirname, '..', '..', 'src', 'easyicu', 'webserver', 'static', 'js', 'screens-guided-pi-error-text.js');
  }
  const errorTextSource = fs.readFileSync(errorTextPath, 'utf8');
  const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const context = { window: { EU_HTML: { esc }, EU_LANG: 'en' } };
  vm.createContext(context);
  vm.runInContext(source, context);
  vm.runInContext(errorTextSource, context);
  const guided = context.window.EasyICU.guidedPi;
  const tr = (en, zh) => en;
  const live = guided.require('errorText').create({ tr, staticPreview: () => false }).errorText;
  const preview = guided.require('errorText').create({ tr, staticPreview: () => true }).errorText;

  // Empty input stays empty (no fallback).
  assert.equal(live(''), '');
  assert.equal(live(null), '');
  assert.equal(live(undefined), '');

  // Known code maps through tr (not the fallback).
  assert.match(live({ code: 'pi_provider_auth_failed' }), /model service rejected/);

  // Static-preview Failed-to-fetch branch (terminology preserved).
  assert.match(preview({ message: 'Failed to fetch' }), /static preview/);

  // Fallback branch: raw transport text verbatim.
  assert.equal(live({ message: 'Failed to fetch' }), 'Failed to fetch');
  assert.equal(live({ message: 'boom' }), 'boom');
  assert.equal(live({ code: 'custom_code' }), 'custom_code');
  assert.equal(live('raw-oops'), 'raw-oops');

  // Harness keeps the legacy alias for the same owner.
  let harnessPath = path.join(dir, '..', '..', '..', 'tests', 'js', 'guided_pi_module_harness.cjs');
  if (!fs.existsSync(harnessPath)) {
    harnessPath = path.join(__dirname, 'guided_pi_module_harness.cjs');
  }
  const harness = fs.readFileSync(harnessPath, 'utf8');
  assert.match(harness, /EU_GUIDED_PI_ERROR_TEXT/);
  assert.match(harness, /['"]errorText['"]/);
})();

process.stdout.write(JSON.stringify({ ok: true, fail_closed_cases: 5 }));
