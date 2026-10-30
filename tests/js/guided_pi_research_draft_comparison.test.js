'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = path.resolve(__dirname, '../../src/easyicu/webserver/static/js/screens-guided-pi-research-draft.js');
const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const tr = (en, zh) => zh;
const draft = overrides => ({
  schema: 'easyicu.research-draft/1', title: '首次讨论', question: '原研究问题', population: '初始人群',
  exposure: '初始暴露', outcome: '院内死亡', candidate_data: '数据候选', approach: '原方法',
  data_requirements: ['原始数据需求'], open_questions: ['原待定项'], changes: ['开始讨论'], ...overrides,
});
function owner() {
  let result;
  const window = { EU_LANG: 'zh', EasyICU: { guidedPi: { declare: (_, value) => { result = value; } } } };
  vm.runInNewContext(fs.readFileSync(source, 'utf8'), { window });
  return result;
}
const block = value => '```easyicu-research-draft\n' + JSON.stringify(value) + '\n```';

test('earlier draft compares every recorded field with the currently displayed draft without changing either', () => {
  const api = owner();
  const older = draft({ title: '旧标题 <img>', population: '旧人群', outcome: '院内死亡',
    exposure: '', data_requirements: ['旧需求 <script>'], open_questions: ['旧待定'], changes: ['旧变更'] });
  const latest = draft({ title: '新标题', population: '新人群', outcome: 'ICU 住院时长',
    exposure: '新暴露', data_requirements: ['新需求'], open_questions: ['新待定'], changes: ['改为时长'] });
  const rows = [{ role: 'assistant', complete: true, entryId: 'old', text: block(older) },
    { role: 'user', text: '更改结局' }, { role: 'assistant', complete: true, entryId: 'new', text: block(latest) }];
  const before = JSON.stringify(rows);
  const html = api.render(api.current(rows), { tr, esc });
  assert.match(html, /较早草案/);
  assert.match(html, /当前显示的草案/);
  for (const text of ['旧标题 &lt;img&gt;', '新标题', '旧人群', '新人群', '院内死亡', 'ICU 住院时长',
    '旧需求 &lt;script&gt;', '新需求', '旧待定', '新待定', '旧变更', '改为时长', '未记录']) {
    assert.ok(html.includes(text), text);
  }
  for (const label of ['标题', '研究问题', '研究人群', '暴露', '主要结局', '候选数据', '拟议方法', '需要的数据', '待定事项', '本次草案变更']) {
    assert.ok(html.includes(`<h5>${label}</h5>`), label);
  }
  assert.doesNotMatch(html, /<img>|<script>|data-gpi-.*(?:approve|execute|configure)/);
  assert.match(html, /两者均不是已批准的计划/);
  assert.equal(JSON.stringify(rows), before);
});

test('failed updates keep the earlier displayed draft and compare only accepted history', () => {
  const api = owner();
  const rows = [
    { role: 'assistant', entryId: 'a', text: block(draft({ title: '最早' })) },
    { role: 'assistant', entryId: 'b', text: block(draft({ title: '最近成功' })) },
    { role: 'user', text: '修改结局' },
    { role: 'assistant', entryId: 'c', errorCode: 'failed', text: block(draft({ title: '失败草案' })) },
  ];
  const snapshot = api.current(rows);
  const html = api.render(snapshot, { tr, esc });
  assert.equal(snapshot.pendingUpdate, true);
  assert.match(html, /上一份草案 · 待更新/);
  assert.match(html, /最早/);
  assert.match(html, /最近成功/);
  assert.doesNotMatch(html, /失败草案/);
});
