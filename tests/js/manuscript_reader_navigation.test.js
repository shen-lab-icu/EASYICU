'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
require('./guided_pi_module_harness.cjs');
global.window = { EU_HTML: { esc: String } };
global.document = { getElementById: () => null };
require(path.resolve(process.argv[2]));
let handler;
let prevented = 0;
let scrolled = 0;
const host = {
  addEventListener: (type, callback) => { handler = callback; },
  querySelector: selector => {
    assert.equal(selector, '#gpi-reference-2');
    return { scrollIntoView: options => {
      assert.equal(options.block, 'start', 'show the reference instead of leaving it at the viewport foot');
      scrolled += 1;
    } };
  },
};
window.EasyICU.guidedPi.require('preview').mount(host);
const click = number => handler({
  preventDefault: () => { prevented += 1; },
  target: { closest: selector => selector === '[data-gpi-reference]'
    ? { dataset: { gpiReference: number } } : null },
});
click('2');
assert.equal(prevented, 1, 'citations must not change the SPA route hash');
assert.equal(scrolled, 1, 'the exact local reference must scroll into view');
click('2] .unrelated');
assert.equal(scrolled, 1, 'invalid citation identities cannot construct selectors');
let focused = 0;
const displayAnchor = {
  dataset: { gpiDisplayAnchor: 'Table 1' },
  scrollIntoView: options => {
    assert.equal(options.block, 'start', 'locate the registered display in place');
    scrolled += 1;
  },
  classList: {
    add: name => { assert.equal(name, 'is-focused'); focused += 1; },
    remove: () => {},
  },
};
host.querySelectorAll = selector => {
  assert.equal(selector, '[data-gpi-display-anchor]');
  return [displayAnchor];
};
const clickDisplay = id => handler({
  preventDefault: () => { prevented += 1; },
  target: { closest: selector => selector === '[data-gpi-display]'
    ? { dataset: { gpiDisplay: id } } : null },
});
clickDisplay('Table 1');
assert.equal(scrolled, 2, 'the registered display must scroll into view');
assert.equal(focused, 1, 'the located display must be highlighted briefly');
clickDisplay('Table 1] .unrelated');
assert.equal(scrolled, 2, 'invalid display identities cannot construct selectors');
process.stdout.write(JSON.stringify({ ok: true, cases: 5 }));
