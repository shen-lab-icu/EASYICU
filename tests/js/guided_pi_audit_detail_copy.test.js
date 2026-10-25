/* Audit P10: readable detail views retain the registered evidence boundaries.
   All fixtures are synthetic aggregates, public receipts, or inert code. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = process.env.EASYICU_DETAIL_OWNER_ROOT || path.resolve(__dirname, '../..');
const js = path.join(root, 'src/easyicu/webserver/static/js');
const sha = 'a'.repeat(64);
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
function owners(language, names) {
  const window = { EU_LANG: language, EU_HTML: { esc }, icon: () => '', setTimeout() {}, clearTimeout() {} };
  const context = vm.createContext({ window });
  for (const name of ['modules', ...names]) {
    vm.runInContext(fs.readFileSync(path.join(js, `screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return window;
}
function host() {
  const listeners = {};
  const query = { value: '', addEventListener: (_, handler) => { listeners.query = handler; } };
  const status = { value: 'all', addEventListener: (_, handler) => { listeners.status = handler; } };
  const results = { innerHTML: '' };
  return {
    innerHTML: '', hidden: true, query, status, results, listeners,
    querySelector: selector => ({ '[data-gpi-wb-query]': query, '[data-gpi-wb-status]': status, '[data-gpi-wb-results]': results }[selector] || null),
    closest: () => ({ classList: { toggle() {} } }),
    addEventListener: (type, handler) => { listeners[type] = handler; },
    replaceChildren() { this.innerHTML = ''; },
  };
}
const evidence = {
  renderer: 'json', previewable: true, kind: 'statistic', display_name: 'step_summary.json',
  evidence_id: 'summary_synthetic', sha256: sha, relative_path: 'evidence/summary_synthetic.json',
  value: { n_total: 7, estimate: 1.234567, ci: [1.1, 1.4], confidence_level: 0.9 },
  declared_lineage: [
    { relation: 'analysis_code', status: 'registered', evidence_id: 'code_synthetic', sha256: 'b'.repeat(64), display_name: 'analysis.py', kind: 'code' },
    { relation: 'input_data', status: 'unregistered', evidence_id: 'unavailable_input' },
  ],
  run_authority: {
    status: 'recorded', run_id: 'run_synthetic', execution_identity_sha256: 'c'.repeat(64),
    links: [{ relation: 'run_plan_authority', status: 'registered', evidence_id: 'plan_synthetic', sha256: 'd'.repeat(64) }],
  },
};
const step = {
  step_id: 'summary_step', state: 'complete', execution: { mode: 'generated' },
  code: { text: 'value = 7\nprint(value)', language: 'python', display_name: 'analysis.py', line_count: 2 },
  inputs: { source_label: 'Synthetic source', stay_count: 7, variable_count: 1 },
  outputs: [{ name: 'step_summary.json', kind: 'summary', bytes: 80, evidence: { id: 'summary_synthetic', sha256: sha } }],
  headline: { en: 'Seven stays summarized.', zh: '已汇总 7 次入住。' },
  extraction_record: { status: 'partial', analysis_unit: 'icu_stay', extracted: 9, retained: 7,
    concepts: [{ concept: 'sodium', column: 'sodium_mean', table: 'assay', unit: 'mmol/L' }],
    steps: [{ step_id: 'summary_step', index: 1, before: 7, after: 7, before_unit: 'icu_stay', unit: 'icu_stay', events: 0 }] },
};
const operation = { id: 'summary_operation', title: 'Synthetic summary', kind: 'pipeline', status: 'complete',
  run: 'run_synthetic', step: 'summary_step', execution: 'generated', facts: [], resources: [] };
const operationButton = { dataset: { gpiOperation: encodeURIComponent(JSON.stringify(operation)) } };
const settle = async () => { for (let i = 0; i < 4; i++) await new Promise(resolve => setImmediate(resolve)); };

for (const language of ['zh', 'en']) {
  const tr = (en, zh) => language === 'zh' ? zh : en;
  test(`${language}: variables, source support, and blank records use plain labels`, () => {
    const window = owners(language, ['workbench-preview']);
    const target = host();
    window.EasyICU.guidedPi.require('workbenchPreview').mount(target, { concepts: [
      { concept_id: 'sodium_mean', study_role: 'plan_input', availability_status: 'semantic_review_required', reason_code: 'plan_bound_column_has_missing_values', physical_coverage_pct: 0 },
      { concept_id: 'signal_count', study_role: 'supporting_variable', availability_status: 'structurally_unavailable', reason_code: 'plan_bound_source_structurally_unavailable' },
      { concept_id: '', study_role: 'other_materialized_column', availability_status: 'not_extracted' },
    ] });
    for (const label of [
      tr('Check the variable definition', '需核对变量含义'),
      tr('Some records are blank; interpret missingness using the variable definition.', '有空白记录，需按变量定义判断是否缺失。'),
      tr('This source does not support this variable for analysis, even if records are present.', '当前数据源尚不支持这一变量；有记录也不能直接用于分析。'),
      tr('Variable screening', '变量筛选'), tr('Find variable', '查找变量'),
      tr('Other prepared variable; not a direct plan input', '其他已准备的变量；非计划直接输入'),
      tr('Unnamed variable', '未命名变量'), tr('Role / variable', '用途 / 变量'),
      tr('Role, variable, module…', '用途、变量、模块…'),
    ]) assert.ok(target.innerHTML.includes(label), label);
    assert.ok(target.innerHTML.includes('0.0%'), 'recorded zero coverage remains zero');
    assert.ok(target.innerHTML.includes(`aria-label="${tr('Variable availability', '变量可用性')}"`));
    assert.ok(target.innerHTML.includes(`<span>${tr('Variables', '变量')}</span>`));
    assert.ok(target.innerHTML.includes(tr('Not shown', '不展示')), 'missing coverage is not zero');
    target.query.value = 'no_matching_synthetic_variable';
    target.listeners.query();
    assert.ok(target.results.innerHTML.includes(tr('No variables match these filters.', '没有符合筛选条件的变量。')));
    target.query.value = '';
    target.listeners.query();
    target.status.value = 'structurally_unavailable';
    target.listeners.status();
    assert.ok(target.results.innerHTML.includes('signal_count'));
    assert.ok(!target.results.innerHTML.includes('sodium_mean'), 'filters still change only the view');
    const empty = host();
    window.EasyICU.guidedPi.require('workbenchPreview').mount(empty, { review_stage: 'post_plan' });
    assert.ok(empty.innerHTML.includes(tr('Coverage has not been recorded.', '覆盖情况暂未记录。')));
    const chart = host();
    window.EasyICU.guidedPi.require('workbenchPreview').mount(chart, {
      review_stage: 'post_plan', concepts: [{ study_role: 'plan_input', physical_coverage_pct: 50 }],
    });
    assert.ok(chart.innerHTML.includes(`<span title="">${tr('Unnamed variable', '未命名变量')}</span>`));
  });

  test(`${language}: source records retain lineage and digests in disclosures`, () => {
    const window = owners(language, ['evidence-preview']);
    const renderer = window.EasyICU.guidedPi.require('evidencePreview');
    const original = JSON.stringify(evidence);
    const html = renderer.render(evidence, { pointer: '/estimate', value: '1.234567' });
    assert.ok(html.includes(`<summary>${tr('Source records for this analysis', '本次分析的来源记录')}</summary>`));
    assert.doesNotMatch(html, /权威锚点|run-level authority anchors|gpi-evidence-readonly/);
    assert.ok(html.indexOf(tr('Selected number source', '所选数字的来源')) < html.indexOf('gpi-evidence-audit-details'));
    for (const value of ['data-evidence-id="code_synthetic"', 'data-evidence-id="plan_synthetic"', 'unavailable_input', '1.234567', sha, 'c'.repeat(64), 'evidence/summary_synthetic.json']) assert.ok(html.includes(value), value);
    assert.ok(!html.includes('data-evidence-id="unavailable_input"'), 'unregistered lineage never becomes an action');
    const audit = html.slice(html.indexOf('gpi-evidence-audit-details'), html.indexOf('gpi-evidence-authority'));
    assert.ok(audit.includes(sha) && audit.includes('/estimate'), 'identity and pointer remain in audit details');
    assert.ok(renderer.render({ ...evidence, run_authority: {} }).includes(`<summary>${tr('Source records for this analysis', '本次分析的来源记录')}</summary>`));
    const noLinks = renderer.render({ ...evidence, declared_lineage: [], run_authority: { status: 'recorded', links: [] } });
    assert.ok(noLinks.includes(tr('No source files were registered for this analysis.', '本次分析未登记可查看的来源文件。')));
    assert.ok(!noLinks.includes('data-gpi-evidence-open'));
    assert.equal(JSON.stringify(evidence), original);
    const mismatch = renderer.render(evidence, { pointer: '/estimate', value: '1.234568' });
    assert.ok(mismatch.includes(tr('Selected number could not be matched', '所选数字未能与来源匹配')));
    assert.ok(!mismatch.includes(tr('Source field and value match', '来源字段与原值已核对')));
  });

  test(`${language}: unavailable previews show file information without a fabricated action`, () => {
    const window = owners(language, ['evidence-preview']);
    const renderer = window.EasyICU.guidedPi.require('evidencePreview');
    const payload = { renderer: 'metadata', previewable: false, display_name: 'attachment.bin',
      evidence_id: 'attachment_synthetic', sha256: sha, bytes: 128, text: '<script>bad()</script>' };
    for (const rendererName of ['metadata', 'future_renderer']) {
      const html = renderer.render({ ...payload, renderer: rendererName });
      assert.ok(html.includes(tr('File information', '文件信息')));
      assert.ok(html.includes(tr('Preview is unavailable for this file.', '此文件暂不支持预览。')));
      assert.doesNotMatch(html, /<button|<a |<script|data-gpi-source-copy|browser renderer|浏览器渲染器/);
      assert.ok(html.includes(sha) && html.includes('128'));
    }
    const withheld = renderer.render({ ...payload, withheld_reason: 'patient_level_rows_withheld' });
    assert.ok(withheld.includes(tr('Patient-level cohort rows are withheld', '患者级队列行已隐藏')));
    assert.doesNotMatch(withheld, /<table|<script/);
    const unknown = renderer.render({ ...payload, withheld_reason: 'future_preview_restriction' });
    const message = unknown.slice(unknown.lastIndexOf('<div class="gpi-evidence-withheld">'));
    assert.ok(message.includes(tr('Preview is unavailable for this file.', '此文件暂不支持预览。')));
    assert.ok(!message.includes('future_preview_restriction'));
    assert.ok(unknown.includes('future_preview_restriction'), 'the diagnostic reason remains in file details');
    const code = renderer.render({ ...payload, previewable: true, renderer: 'code', kind: 'code' });
    assert.ok(code.includes(tr('Read-only evidence code', '只读证据代码')));
    assert.ok(code.includes('&lt;script&gt;bad()&lt;/script&gt;'));
    assert.doesNotMatch(code, /gpi-evidence-readonly|<script/);
  });

  test(`${language}: step results precede the source record and code stays available`, () => {
    const window = owners(language, ['step-detail']);
    const owner = window.EasyICU.guidedPi.require('stepDetail');
    const detail = owner.validDetail(step);
    const html = owner.render({ operation, stepDetail: detail }, { codeRows: esc, statusLabel: () => tr('Complete', '已完成') });
    assert.ok(html.indexOf(tr('Seven stays summarized.', '已汇总 7 次入住。')) < html.indexOf('gpi-extraction-record'));
    assert.ok(html.includes(`<details><summary>${tr('Analysis code', '分析代码')}</summary>`));
    for (const value of ['data-gpi-step-copy', 'data-gpi-step-evidence="0"', 'print(value)', 'assay', 'mmol/L', tr('Not recorded', '未记录'), tr('0 events', '0 个事件')]) assert.ok(html.includes(value), value);
    assert.ok(html.includes(tr('Code written for this run', '本次编写的代码')));
  });

  test(`${language}: attention and model-service fallback copy describe the recorded state`, () => {
    const window = owners(language, ['activity']);
    const activity = window.EasyICU.guidedPi.require('activity').create({ tr, esc, iconHtml: () => '',
      resourceName: () => '', resourceKey: () => '', resourceButton: () => '', runFailureText: () => tr('Retry stopped', '重试已停止') });
    for (const [step, en, zh] of [
      ['pause', 'The task is paused.', '任务已暂停。'],
      ['human_review', 'The study needs manual review.', '这项研究需要人工核对。'],
      ['scientific_action_gap', 'The study cannot continue with the current settings. Review the outstanding issues.', '当前设置下研究无法继续，请查看待处理问题。'],
      ['cancel_requested', 'Cancellation requested; waiting for the task to stop.', '已请求取消，正在等待任务停止。'],
    ]) assert.equal(activity.pipelineEventLabel({ step, status: 'action_required', label: 'untrusted internal wording' }), tr(en, zh));
    for (const [status, en, zh] of [
      ['running', 'Checking the model service.', '正在检查模型服务。'],
      ['failed', 'The model service reported a problem. Review the failure details.', '模型服务出现问题，请查看失败原因。'],
      ['complete', 'Model service status updated.', '模型服务状态已更新。'],
    ]) assert.equal(activity.pipelineEventLabel({ step: 'provider_future_event', status }), tr(en, zh));
    assert.ok(activity.pipelineEventLabel({ step: 'provider_recovery_wait' }).includes(tr('waiting', '等待')));
    assert.equal(activity.pipelineEventLabel({ step: 'provider_continuation_refused' }), tr('Retry stopped', '重试已停止'));
  });

  test(`${language}: a saved setup without a detailed receipt does not imply a complete plan`, () => {
    const window = owners(language, ['activity']);
    const activity = window.EasyICU.guidedPi.require('activity').create({ tr, esc, iconHtml: () => '',
      resourceName: () => '', resourceKey: () => '', resourceButton: () => '', runFailureText: () => '' });
    const saved = { id: 'save_synthetic', kind: 'tool', toolName: 'easyicu_update_study_context', status: 'complete', code: 'study_context_updated', text: 'untrusted family and saved fields' };
    const html = activity.render({ status: 'complete', steps: [saved] });
    assert.ok(html.includes(tr('This update is saved. Review the study setup for any remaining fields.', '本次修改已保存；请在研究设置中核对仍需填写的内容。')));
    assert.doesNotMatch(html, /untrusted family|analysis ready|分析已就绪/);
    assert.ok(!activity.render({ status: 'complete', steps: [{ ...saved, status: 'error' }] }).includes(tr('This update is saved.', '本次修改已保存')));
    const unknown = activity.render({ status: 'complete', steps: [{ ...saved, code: '' }] });
    assert.ok(unknown.includes(tr('Review the study setup; this record does not list the updated fields.', '请核对研究设置；这条记录未列出修改的具体内容。')));
    assert.ok(!unknown.includes(tr('This update is saved.', '本次修改已保存')));
  });

  test(`${language}: a missing evidence renderer is explained and unavailable source code stays closed`, async () => {
    const window = owners(language, ['step-detail', 'source-view']);
    window.EU_API = {
      loadPiCopilotRunStep: async () => ({ ok: true, run_id: operation.run, step }),
      loadPiCopilotResearchEvidence: async () => ({ ok: true, payload: { ...evidence, renderer: 'metadata', previewable: false } }),
    };
    const target = host();
    const view = window.EasyICU.guidedPi.require('sourceView');
    view.mount(target);
    assert.equal(view.openOperation(operationButton, 'project_synthetic', operation.run), true);
    await settle();
    target.listeners.click({ target: { closest: selector => selector === '[data-gpi-step-evidence]' ? { dataset: { gpiStepEvidence: '0' } } : null } });
    await settle();
    assert.ok(target.innerHTML.includes(tr('The evidence preview could not be displayed.', '暂时无法显示证据预览。')));
    assert.doesNotMatch(target.innerHTML, /data-gpi-source-copy|data-gpi-step-copy|data-gpi-retry/);
    await view.open({ dataset: { sourceEvidenceId: 'code_synthetic', sourceSha256: sha } }, 'project_synthetic', operation.run);
    assert.ok(target.innerHTML.includes(tr('Source unavailable', '源代码暂不可用')));
    assert.doesNotMatch(target.innerHTML, /gpi-source-code|data-gpi-source-copy|print\(value\)/);
    assert.equal(await view.open({ dataset: { sourceEvidenceId: '../bad', sourceSha256: sha } }, 'project_synthetic', operation.run), false);
    view.close();
  });
}
