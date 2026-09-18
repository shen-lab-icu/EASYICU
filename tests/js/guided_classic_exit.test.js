'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../src/easyicu/webserver/static/js/screens-guided.js'), 'utf8');
const start = source.indexOf("const openEl = e.target.closest('[data-open]');");
const end = source.indexOf('// clickable Study panel step', start);
assert.ok(start >= 0 && end > start);
for (const state of [null, { path: '/synthetic' }]) {
  const handoffs = [];
  const context = {
    EXTRACT: { state: () => state },
    e: { target: { closest: () => ({ dataset: { open: 'extraction' } }) } },
    window: { EU_GUIDED_HANDOFF: { set: value => handoffs.push(value) } },
    location: { hash: '' },
    guidedAgentHandoffPrefill: () => ({ question: 'synthetic' }),
    guidedExtractionClassicConfig: () => ({ path: '/synthetic' }),
  };
  vm.runInNewContext(`(function () { ${source.slice(start, end)} })()`, context);
  assert.equal(context.location.hash, '#extraction');
  assert.equal(handoffs.length, state ? 1 : 0);
  if (state) assert.equal(handoffs[0].requires_user_confirm, true);
}
