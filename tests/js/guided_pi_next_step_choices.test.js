'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function nextActions() {
  const esc = value => String(value).replace(/[&<>"']/g, ch => `&#${ch.charCodeAt(0)};`);
  const context = { window: { EU_HTML: { esc } } };
  vm.createContext(context);
  for (const name of ['modules', 'next-actions']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('nextActions');
}

test('choices the host rewrites to one action are offered once', () => {
  const owner = nextActions();
  const html = owner.render({
    explicit: true, body: '', prompt: '请明确确认结局定义，并选择数据来源：', asking: false,
    choices: ['确认院内死亡为结局，并下载准备 eICU Demo', '下载并准备 eICU Demo', '我想先修改研究问题再选数据'],
  }, { language: 'zh' });
  assert.equal((html.match(/下载并准备 eICU Demo/g) || []).length, 1);
  assert.match(html, /我想先修改研究问题再选数据/);
});
