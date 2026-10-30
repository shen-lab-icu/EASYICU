'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
global.window = { EU_LANG: 'zh' };
for (const name of ['screens-guided-pi-modules', 'html-escape', 'screens-guided-pi-plan-actions', 'screens-guided-pi-confirmation']) {
  require(path.join(__dirname, '../../src/easyicu/webserver/static/js', name + '.js'));
}

function render(scope, offer) {
  const workflow = { next_action_code: 'operator_plan_approval_required', raw_preparation: {
    status: 'awaiting_consumption_closure', scope, summary: '检查准备范围',
  }, host_decisions: offer ? { raw_preparation_review: { scope_sha256: 'a'.repeat(64) } } : {} };
  const host = { tr: (_en, zh) => zh, esc: window.EU_HTML.esc, iconHtml: () => '', workflow: () => workflow,
    session: () => ({ session_id: 'session', binding: { study_context_id: 'study' } }),
    busy: () => false, sessionIsStale: () => false, researchSourceReady: () => true,
    projectId: () => 'project', resourceButton: () => '', render: () => {}, setBusy: () => {}, setError: () => {},
    appendMessage: () => {}, api: () => ({}) };
  return window.EasyICU.guidedPi.require('confirmation').create(host).workflowConfirmationHtml();
}

test('single table review displays exact source, file, output columns and full read scope before offering approval', () => {
  const scope = { adapter_id: 'registered-csv-projection/1', purpose: 'prepare_single_table_only',
    source_identity: { database: 'sic', registration_id: 'source<1>' },
    relative_file: 'table.csv', columns: ['id', 'value'] };
  const html = render(scope, true);
  assert.match(html, /source&lt;1&gt;/);
  for (const text of ['table.csv', 'id, value', '全部行和全部字段', '不构成队列或研究分析', '批准后将开始这项单表数据准备']) assert.ok(html.includes(text));
  assert.match(html, /data-gpi-preparation-review="approved"/);
  const unavailable = render(scope, false);
  assert.doesNotMatch(unavailable, /data-gpi-preparation-review="approved"/);
  assert.match(unavailable, /中断恢复和结果交付尚未接通/);
  assert.doesNotMatch(render({ ...scope, columns: [] }, true), /data-gpi-preparation-review="approved"/);
});
