/* B11: a launch on an unreadable package names it and its re-export. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const root = process.argv[2] || path.resolve(__dirname, '../..');
global.window = global;
global.EU_HTML = { esc: value => String(value ?? '') };
for (const file of ['screens-guided-pi-modules.js', 'product-labels.js', 'screens-guided-pi-error-text.js', 'screens-guided-pi-run-stop.js']) {
  require(path.join(root, 'src/easyicu/webserver/static/js', file));
}
const modules = global.EasyICU.guidedPi;
const action = {
  kind: 'reexport_data', action_id: 'reexport_data_0123456789abcdef', label_key: 'stop_next_actions.reexport_data',
  params: { route: 'data_extraction', requires_explicit_choice: true },
};
for (const language of ['zh', 'en']) {
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const { runFailureText } = modules.require('errorText').create({ tr });
  test(`${language}: the stop names the missing manifest, not the generic stop`, () => {
    const text = runFailureText('research_pipeline_manifest_invalid');
    assert.notEqual(text, runFailureText('easyicu_unlisted_stop'));
    assert.match(text, language === 'zh' ? /清单文件.*重新准备数据/ : /no readable EasyICU manifest.*Prepare the data again/);
  });
  test(`${language}: the job stop's primary choice is the re-export`, () => {
    const stop = modules.require('runStop').create({
      tr, runFailureText, canAct: () => true, workflow: () => ({ next_action_code: 'research_pipeline_manifest_invalid' }),
    });
    const job = { job_id: 'b11_job', kind: 'agent-run', status: 'failed', error_code: 'research_pipeline_manifest_invalid',
      created_at_epoch: 1, finished_at_epoch: 2, next_actions: [action] };
    const html = stop.renderJob({ archived_child_jobs: [job] });
    assert.ok(html.includes(`<p>${runFailureText('research_pipeline_manifest_invalid')}</p>`), html);
    const buttons = html.match(/<button[^>]*>[^<]*<\/button>/g) || [];
    assert.equal(buttons.length, 1, html);
    assert.match(buttons[0], /data-gpi-stop-action="reexport_data_0123456789abcdef"/);
    assert.ok(buttons[0].includes(`>${global.EU_PRODUCT_LABELS.stopAction('reexport_data', tr)}<`), buttons[0]);
  });
}
