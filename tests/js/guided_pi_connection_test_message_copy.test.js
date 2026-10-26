/* A failed test message before saving names its cause in the interface language. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const root = process.argv[2] || path.resolve(__dirname, '../..');
global.window = global;
global.EU_HTML = { esc: value => String(value ?? '') };
for (const file of ['screens-guided-pi-modules.js', 'screens-guided-pi-error-text.js']) {
  require(path.join(root, 'src/easyicu/webserver/static/js', file));
}
const modules = global.EasyICU.guidedPi;
const backendMessage = 'The selected model did not accept a bounded verification request.';
const failed = status => ({
  code: 'pi_provider_inference_verification_failed',
  message: backendMessage,
  details: status === undefined ? {} : { status_code: status },
});

for (const language of ['zh', 'en']) {
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const { errorText } = modules.require('errorText').create({ tr });

  for (const status of [500, 502, 503, 429]) {
    test(`${language}: status ${status} reads as a temporary outage`, () => {
      assert.equal(errorText(failed(status)), tr(
        `The model service is temporarily unavailable (status ${status}). Try again in a moment.`,
        `模型服务暂时不可用（状态码 ${status}），请稍后重试。`));
    });
  }

  test(`${language}: a rejected test message points at the model`, () => {
    assert.equal(errorText(failed(400)), tr(
      'The selected model did not answer a test message (status 400). Check the model name or choose another model.',
      '所选模型没有回应测试消息（状态码 400），请检查模型名或换一个模型。'));
  });

  test(`${language}: an unknown status keeps the sentence whole`, () => {
    assert.equal(errorText(failed(undefined)), tr(
      'The selected model did not answer a test message. Check the model name or choose another model.',
      '所选模型没有回应测试消息，请检查模型名或换一个模型。'));
  });

  test(`${language}: no verification code falls back to the backend sentence`, () => {
    for (const code of ['pi_provider_inference_verification_failed', 'pi_provider_inference_response_invalid',
      'pi_provider_response_too_large']) {
      const text = errorText({ code, message: backendMessage, details: { status_code: 503 } });
      assert.notEqual(text, backendMessage, code);
      if (language === 'zh') assert.match(text, /[一-鿿]/, code);
    }
  });
}
