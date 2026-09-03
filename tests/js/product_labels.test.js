'use strict';

const assert = require('assert');

global.window = {};
require(process.argv[2]);

const labels = window.EU_PRODUCT_LABELS;
assert(labels);
assert.strictEqual(labels.copilotTitle('Pi Copilot'), 'EasyICU Copilot');
assert.strictEqual(labels.copilotTitle('EasyICU Copilot'), 'EasyICU Copilot');
assert.strictEqual(
  labels.copilotTitle('Pi Copilot', 'Sepsis mortality calibration'),
  'Sepsis mortality calibration',
);
assert.strictEqual(labels.copilotTitle('Named study', 'fallback'), 'Named study');
assert.strictEqual(
  labels.projectTitle('Pi Copilot', 'Sepsis mortality calibration'),
  'Sepsis mortality calibration',
);
assert.strictEqual(
  labels.projectTitle('Untitled ICU study', 'AKI trajectory study'),
  'AKI trajectory study',
);
assert.strictEqual(labels.projectTitle('Named study', 'fallback'), 'Named study');
window.EU_LANG = 'zh';
assert.strictEqual(labels.projectTitle('Untitled guided study'), '研究项目');
assert(Object.isFrozen(labels));

console.log('product label contract ok');
