'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const js = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function owner(language) {
  const context = vm.createContext({ window: { EU_HTML: { esc: String } } });
  for (const name of ['modules', 'error-text']) {
    vm.runInContext(fs.readFileSync(path.join(js, `screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return context.window.EasyICU.guidedPi.require('errorText').create({
    tr: (en, zh) => language === 'zh' ? zh : en, staticPreview: () => false,
  });
}

for (const language of ['zh', 'en']) {
  test(`${language}: stale Demo authority asks to reselect the source`, () => {
    const errors = owner(language);
    const code = 'research_pipeline_demo_planning_authority_stale';
    for (const text of [errors.errorText({ code }), errors.jobErrorText(code), errors.runFailureText(code)]) {
      assert.match(text, language === 'zh' ? /重新选择 Demo 数据源.*审阅/ : /Select the Demo source again.*review/);
      assert.doesNotMatch(text, /retry|重试|删除|remove/i);
    }
  });
  test(`${language}: missing first ICU proof preserves the required population`, () => {
    const errors = owner(language);
    const code = 'research_pipeline_first_observed_input_binding_unavailable';
    for (const text of [errors.errorText({ code }), errors.jobErrorText(code), errors.runFailureText(code)]) {
      assert.match(text, language === 'zh' ? /尚无法将首次 ICU 人群准备结果绑定到实际研究数据/ : /first ICU population cannot currently be bound to the research input/);
      assert.match(text, language === 'zh' ? /分析尚未开始/ : /analysis has not started/);
      assert.match(text, language === 'zh' ? /核对数据源和准备结果.*联系支持人员/ : /Check the data source and preparation result.*contact support/);
      assert.doesNotMatch(text, /retry|重试|删除|remove/i);
    }
  });
}
