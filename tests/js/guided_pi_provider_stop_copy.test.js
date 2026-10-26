/* Provider failures keep their typed cause in the run-stop top line. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const root = process.argv[2] || path.resolve(__dirname, '../..');
global.window = global;
global.EU_HTML = { esc: value => String(value ?? '') };
for (const file of ['screens-guided-pi-modules.js', 'screens-guided-pi-error-text.js', 'screens-guided-pi-run-stop.js']) {
  require(path.join(root, 'src/easyicu/webserver/static/js', file));
}
const modules = global.EasyICU.guidedPi;
for (const language of ['zh', 'en']) {
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const { runFailureText } = modules.require('errorText').create({ tr });
  const stop = modules.require('runStop').create({ tr, runFailureText });
  for (const reason of ['provider_http_error', 'provider_timeout', 'research_pipeline_provider_timeout']) {
    for (const runner of ['not_started', 'host', undefined]) {
      for (const flattened of [false, true]) {
        test(`${language}: ${reason}, ${runner}, ${flattened ? 'flat' : 'nested'}`, () => {
          const run = { run_id: 'synthetic', run_status: 'failed',
            gate_reason: reason === 'research_pipeline_provider_timeout' ? reason : 'research_pipeline_execution_failed',
            ...(flattened ? { gate_detail_code: reason, gate_detail_runner_kind: runner }
              : { gate_detail: { reason_code: reason, runner_kind: runner } }) };
          const html = stop.render({ savedRuns: [run] });
          const expected = tr('The model service returned an error, so the run stopped. Retry once the connection or service is back.',
            '模型服务返回错误，本次运行已停止。网络或服务恢复后可直接重试。')
            + (runner === 'not_started' ? tr(' Analysis has not started.', '分析尚未开始。') : '');
          assert.ok(html.includes(`<p>${expected}</p>`), html);
          assert.doesNotMatch(html, /支持人员|Send the failure receipt/);
        });
      }
    }
  }
  test(`${language}: unrelated failures retain their existing top line`, () => {
    const html = stop.render({ savedRuns: [{ run_status: 'failed', gate_reason: 'research_pipeline_execution_failed',
      gate_detail: { reason_code: 'pipeline_operation_raised', runner_kind: 'not_started' } }] });
    assert.ok(html.includes(runFailureText('research_pipeline_execution_failed', { code: 'pipeline_operation_raised' })));
    assert.doesNotMatch(html, /模型服务返回错误|model service returned an error/);
  });
}
