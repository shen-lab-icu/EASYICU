'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'zh';
global.EU_STUDY_CONTEXT = {
  all: () => [
    { id: 'legacy', title: 'Pi Copilot', question: 'Sepsis mortality calibration' },
    { id: 'study_4e3e37e9aad64496', title: 'EasyICU Copilot', updated_at: '2026-08-23T08:18:46Z' },
    { id: 'study_8cd181b313e80dbd', title: 'EasyICU Copilot', updated_at: '2026-08-23T08:18:46Z' },
    { id: 'study-651b4b5b-0ccf-4f16-b5a6-0febe5fcc943', title: 'Untitled ICU study', updated_at: '2026-08-23T09:51:45Z' },
    { id: 'named', title: 'AKI trajectory study', question: 'Question' },
  ],
  active: () => null,
};

require(path.resolve(process.argv[2]));
require(path.resolve(process.argv[3]));

const projects = global.EU_AGENT_STUDY_CONTEXT.projects();
const names = projects.map(row => row.name[0]);
assert.equal(names[0], 'Sepsis mortality calibration');
assert.equal(names.at(-1), 'AKI trajectory study');
assert.equal(new Set(names.slice(1, 4)).size, 3, 'legacy fallback titles must remain distinguishable');
assert(names[1].startsWith('研究项目 · '));
assert(names[1].endsWith('4e3e37'));
assert(names[2].endsWith('8cd181'));
assert(names[3].endsWith('651b4b'));
assert.equal(projects.some(row => /Pi Copilot/.test(row.name[0])), false);

console.log('project title projection contract ok');
