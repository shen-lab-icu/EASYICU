'use strict';

// Search and attention open another project exactly as picking its rail row
// does, and say so when the rail does not list it.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

test('a listed project opens through its rail row; an unlisted one does not', () => {
  const clicks = [];
  const declared = {};
  const rows = ['p1', 'p2'].map(id => ({ dataset: { projectId: id }, click: () => clicks.push(id) }));
  const window = { EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } } };
  const document = { querySelectorAll: selector => selector === '.gd-sess[data-project-id]' ? rows : [] };
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-guided-pi-project.js'), 'utf8'), vm.createContext({ window, document }));
  const owner = declared.project;
  assert.equal(owner.openListedProject('p2'), true);
  assert.deepEqual(clicks, ['p2']);
  assert.equal(owner.openListedProject('p9'), false);
  assert.equal(owner.openListedProject(''), false);
  assert.deepEqual(clicks, ['p2']);
});
