'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const ownerPath = process.argv[2];
assert.ok(ownerPath, 'screens-icd.js path is required');

global.window = global;
global.t = en => en;
global.icon = () => '';
global.escHtml = value => String(value == null ? '' : value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;');
global.__euRender = () => {};
global.setTimeout = () => 1;
global.clearTimeout = () => {};

vm.runInThisContext(fs.readFileSync(ownerPath, 'utf8'), { filename: ownerPath });

const html = window.EUIcd.block({
  database: 'miiv',
  databaseLabel: 'MIMIC-IV 3.1',
  real: true,
});
assert.match(html, /MIMIC-IV 3\.1/);
assert.match(html, /Bound to the current local source/);
assert.match(html, /No estimated patient count or synthetic code frequency is shown/);
assert.doesNotMatch(html, /MIMIC-III|eICU|\/ 412|Top matching ICD codes/);

const handlers = {};
const inputs = {
  '#icdIncInput': {
    value: '',
    addEventListener(name, callback) { handlers[`inc:${name}`] = callback; },
  },
  '#icdExcInput': {
    value: '',
    addEventListener(name, callback) { handlers[`exc:${name}`] = callback; },
  },
};
window.EUIcd.bind({ querySelector: selector => inputs[selector] || null });
inputs['#icdIncInput'].value = 'N18, I50';
handlers['inc:input']();
inputs['#icdExcInput'].value = 'C';
handlers['exc:input']();

assert.deepEqual(window.EUIcd.contract(), {
  icd_include: 'N18, I50',
  icd_exclude: 'C',
  include_diagnoses: ['N18', 'I50'],
  exclude_diagnoses: ['C'],
});

const selected = window.EUIcd.block({ database: 'miiv', databaseLabel: '<MIMIC-IV>', real: true });
assert.match(selected, /&lt;MIMIC-IV&gt;/);
assert.doesNotMatch(selected, /<MIMIC-IV>/);
assert.doesNotMatch(selected, /74 \/ 412|18\.0%/);

console.log('extraction ICD source binding contract passed');
