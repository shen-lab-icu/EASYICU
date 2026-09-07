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
process.stdout.write(JSON.stringify({ ok: true, cases: 3 }));
