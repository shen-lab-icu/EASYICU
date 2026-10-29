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
  test(`${language}: message-call exhaustion explains its terminal budget across raw and typed errors`, () => {
    const errors = owner(language);
    const code = 'pi_shell_message_provider_call_budget_exhausted';
    const variants = [code, `  ${code}: private-debug-detail`, { code, message: 'private-debug-detail' },
      { message: code }, new Error(`${code}: private-debug-detail`)];
    for (const value of variants) {
      for (const text of [errors.errorText(value), errors.jobErrorText(value),
        errors.modelErrorText(value, false), errors.modelErrorText(value, true)]) {
        assert.match(text, language === 'zh' ? /模型调用额度已耗尽/ : /used its allowed model calls/);
        assert.match(text, language === 'zh' ? /文件和执行记录已保留/ : /files and execution records are preserved/);
        assert.match(text, language === 'zh' ? /本轮无法继续/ : /turn cannot continue/);
        assert.doesNotMatch(text, /管理员|administrator|发送.*继续|Send.*continue|新建|new conversation|reset|重置|private-debug-detail/i);
      }
    }
  });

  test(`${language}: a gateway timeout directs the user to bounded recovery without replay or hidden debug text`, () => {
    const errors = owner(language);
    const code = 'pi_gateway_timeout';
    for (const value of [code, `${code}: private-debug-detail`, { code }, new Error(`${code}: private-debug-detail`)]) {
      for (const text of [errors.errorText(value), errors.jobErrorText(value), errors.modelErrorText(value, true)]) {
        assert.match(text, language === 'zh' ? /等待已超时/ : /wait timed out/);
        assert.match(text, language === 'zh' ? /文件和执行记录已保留/ : /files and execution records are preserved/);
        assert.match(text, language === 'zh' ? /恢复提示.*状态未确认.*不要重复执行/ : /recovery guidance.*do not repeat operations/);
        assert.doesNotMatch(text, /管理员|administrator|发送.*继续|Send.*continue|reset|重置|private-debug-detail/i);
      }
    }
  });

  test(`${language}: message-cost exhaustion stays closed while other failures keep their existing meanings`, () => {
    const errors = owner(language);
    const exhausted = errors.modelErrorText('pi_shell_message_cost_budget_exhausted', true);
    assert.match(exhausted, language === 'zh' ? /模型费用额度已耗尽/ : /model cost budget/);
    assert.match(exhausted, language === 'zh' ? /本轮无法继续/ : /turn cannot continue/);
    assert.doesNotMatch(exhausted, /发送.*继续|Send.*continue|reset|重置/i);
    assert.match(errors.modelErrorText('pi_model_provider_unavailable', true),
      language === 'zh' ? /工具操作已完成/ : /tool action completed/);
    assert.match(errors.modelErrorText('pi_model_provider_unavailable', false),
      language === 'zh' ? /连接中断/ : /connection was interrupted/);
    assert.doesNotMatch(errors.errorText('pi_gateway_timeout_other'), language === 'zh' ? /本次等待已超时/ : /This wait timed out/);
  });
}
