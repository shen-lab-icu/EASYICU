'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function owner() {
  let result;
  const window = { EU_LANG: 'zh', EasyICU: { guidedPi: { declare: (_name, value) => { result = value; } } } };
  const source = path.resolve(__dirname, '../../src/easyicu/webserver/static/js/screens-guided-pi-research-draft.js');
  vm.runInNewContext(fs.readFileSync(source, 'utf8'), { window });
  return result;
}
function draft(exposure = '待核对的指标') {
  return { schema: 'easyicu.research-draft/1', title: '讨论草案', question: '研究问题', population: '',
    exposure, outcome: '', candidate_data: '', approach: '', data_requirements: [], open_questions: [], changes: [] };
}
const block = value => '```easyicu-research-draft\n' + JSON.stringify(value) + '\n```';

test('explicit draft revision reports missing or malformed updated output without inventing a draft', () => {
  const draftOwner = owner();
  const old = { role: 'assistant', complete: true, text: block(draft()), entryId: 'old' };
  const ask = { role: 'user', text: '核对来源后，只更新讨论草案，不读取患者数据。' };
  const plain = { role: 'assistant', complete: true, stopReason: 'stop', text: '已经核对来源字典。' };
  const snapshot = draftOwner.current([old, ask, plain]);
  assert.equal(snapshot.updateFailed, true);
  assert.equal(snapshot.draft.exposure, '待核对的指标');
  const rendered = draftOwner.render(snapshot, { tr: (_en, zh) => zh, esc: value => String(value) });
  assert.match(rendered, /本轮未能更新草案/);
  assert.match(rendered, /下方仍是上一份草案/);
  assert.match(rendered, /请再次要求更新/);
  assert.equal(draftOwner.current([old, ask, { ...plain, text: '```easyicu-research-draft\n{bad}\n```' }]).updateFailed, true);
  assert.equal(draftOwner.current([old, { role: 'user', text: 'Please update the discussion draft.' }, plain]).updateFailed, true);
  assert.equal(draftOwner.current([old, { role: 'user', text: '不要更新草案，只回答问题' }, plain]).updateFailed, false);
  assert.equal(draftOwner.current([old, { role: 'user', text: '继续检查字典' }, plain]).updateFailed, false);
  assert.equal(draftOwner.current([old, ask]).updateFailed, false);
  assert.equal(draftOwner.current([old, ask, { ...plain, complete: false }]).updateFailed, false);
  assert.equal(draftOwner.current([old, ask, { ...plain, stopReason: 'toolUse' }]).updateFailed, false);
  assert.equal(draftOwner.current([old, ask, { ...plain, stopReason: 'error' }]).updateFailed, true);
  const revised = { ...plain, text: block(draft('已核对的字典声明')) };
  const updated = draftOwner.current([old, ask, revised]);
  assert.equal(updated.updateFailed, false);
  assert.equal(updated.pendingUpdate, false);
  assert.equal(updated.draft.exposure, '已核对的字典声明');
});
