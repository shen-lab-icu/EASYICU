'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const A = 'a'.repeat(64);
const B = 'b'.repeat(64);
const FILE = 'icu_synthetic_demo.html';

function owners(language = 'zh') {
  const esc = value => String(value == null ? '' : value)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
  const context = { URL, window: { EU_HTML: { esc }, EU_LANG: language } };
  vm.createContext(context);
  for (const name of ['modules', 'resources', 'markdown']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return {
    resources: context.window.EasyICU.guidedPi.require('resources').create({ esc }),
    markdown: context.window.EasyICU.guidedPi.require('markdown'),
    context, esc,
  };
}

function file(fields = {}) {
  return { kind: 'file', file: FILE, label: '乳酸演示简报', media_type: 'text/html',
    scientific_evidence: false, claim_ceiling: 'unsupported', ...fields };
}

function installLiveResources(context, resources) {
  const shell = fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/static/js/screens-guided-pi.js'), 'utf8');
  const body = shell.split('  function addAssistantResources(resources) {', 2)[1]
    .split('  function completeLatestAssistant', 1)[0];
  context.RESOURCE_OWNER = resources;
  vm.runInContext(`function addAssistantResources(resources) {${body}`, context);
}

test('read, write, check and webpage snapshots share one workspace file entry', () => {
  const { resources } = owners();
  const read = file({ sha256: A, label: '初次读取' });
  const check = file({ checked_sha256: A, label: '已检查简报' });
  const webpage = file({ kind: 'webpage', checked_sha256: A, label: '打开简报' });
  const laterRead = file({ sha256: A, label: '乳酸演示简报' });
  const source = [read, check, webpage, laterRead];
  const before = JSON.stringify(source);
  const rows = resources.merge([], source);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].checked_sha256, A);
  assert.equal(rows[0].label, '乳酸演示简报');
  assert.equal(rows[0].scientific_evidence, false);
  assert.equal(rows[0].claim_ceiling, 'unsupported');
  assert.equal(JSON.stringify(source), before, 'merge must not rewrite Host snapshots');
  assert.equal(resources.key(read), resources.key(webpage));
  const html = resources.renderForMessage({ resources: source });
  assert.equal((html.match(/class="gpi-file-card"/g) || []).length, 1);
  assert.match(html, new RegExp(`data-gpi-resource-checked-digest="${A}"`));
});

test('incremental aggregation retains a genuine check only for identical bytes', () => {
  const { resources } = owners();
  let rows = resources.merge([file({ sha256: A })], [file({ checked_sha256: A })]);
  rows = resources.merge(rows, [file({ sha256: A })]);
  assert.equal(rows[0].checked_sha256, A);
  rows = resources.merge(rows, [file({ sha256: B, label: '已修改版本' })]);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].checked_sha256, undefined);
  assert.equal(rows[0].sha256, B);
  assert.equal(rows[0].label, '已修改版本');
  assert.doesNotMatch(resources.button(rows[0]), new RegExp(`checked-digest="${A}"`));
  rows = resources.merge(rows, [file({ checked_sha256: B })]);
  assert.equal(rows[0].checked_sha256, B);
});

test('unknown, conflicting and malformed versions cannot inherit a check', () => {
  const { resources } = owners();
  for (const next of [file(), file({ sha256: B, checked_sha256: A }),
    file({ sha256: 'invalid', checked_sha256: A }), file({ sha256: A, checked_sha256: 'invalid' })]) {
    const rows = resources.merge([file({ checked_sha256: A })], [next]);
    assert.equal(rows[0].checked_sha256, undefined);
  }
  assert.equal(resources.merge([], [file({ sha256: A })])[0].checked_sha256, undefined);
  assert.equal(resources.merge([], [file({ checked_sha256: A })])[0].checked_sha256, A);
});

test('ordinary webpage hashes are never upgraded to Host checks by DOM transport', () => {
  const { resources } = owners();
  const unverified = resources.fromButton({ dataset: {
    gpiResourceKind: 'webpage', gpiResourceFile: FILE, gpiResourceDigest: A,
  } });
  assert.equal(unverified.sha256, A);
  assert.equal(unverified.checked_sha256, undefined);
  const checked = resources.fromButton({ dataset: {
    gpiResourceKind: 'webpage', gpiResourceFile: FILE,
    gpiResourceDigest: A, gpiResourceCheckedDigest: A,
  } });
  assert.equal(checked.checked_sha256, A);
});

test('workspace merging keeps other files and governed research coordinates separate', () => {
  const { resources } = owners();
  const governed = { kind: 'research_artifact', run_id: 'run-a', artifact: FILE, state: 'reviewed' };
  const rows = resources.merge([file({ sha256: A }), governed], [
    file({ file: 'data/synthetic.csv', sha256: B }),
    { ...governed, state: 'later' }, { ...governed, run_id: 'run-b' },
  ]);
  assert.equal(rows.length, 4);
  assert.equal(rows.find(row => row.run_id === 'run-a').state, 'reviewed');
  assert.equal(rows.filter(row => row.run_id).length, 2);
});

test('workspace outputs use task labels while plan and literature retain their labels', () => {
  for (const language of ['zh', 'en']) {
    const { resources } = owners(language);
    const workspace = resources.renderForMessage({ resources: [file()] });
    assert.match(workspace, language === 'zh' ? /aria-label="本轮产物"/ : /aria-label="Task outputs"/);
    const plan = resources.renderForMessage({ resources: [{ kind: 'idea_plan', run_id: 'plan' }] });
    assert.match(plan, language === 'zh' ? /研究方案/ : /Research plan/);
    const combined = resources.renderForMessage({ resources: [file(),
      { kind: 'literature_source', pmid: '123', title: 'Topic' },
      { kind: 'literature_source', pmid: '456', title: 'Method', authority_class: 'literature_method' }] });
    assert.match(combined, language === 'zh' ? /本轮产物与文献/ : /Task outputs and literature/);
    assert.match(combined, language === 'zh' ? /方法学参考/ : /Method references/);
  }
});

test('relative Markdown links open the exact attached resource through controlled preview', () => {
  const { markdown } = owners();
  const html = markdown.render(`已生成[打开简报](${FILE})，并有[合成 CSV](data/synthetic.csv)。`, {
    resources: [file({ checked_sha256: A }), file({ file: 'data/synthetic.csv', media_type: 'text/csv', sha256: B })],
  });
  assert.equal((html.match(/data-gpi-resource-kind=/g) || []).length, 2);
  assert.match(html, new RegExp(`data-gpi-resource-file="${FILE}"`));
  assert.match(html, new RegExp(`data-gpi-resource-checked-digest="${A}"`));
  assert.match(html, /data-gpi-resource-file="data\/synthetic.csv"/);
  assert.doesNotMatch(html, /\[打开简报\]|\[合成 CSV\]|href="(?:file|data):/);
  const encoded = markdown.render('[中文文件](./%E5%90%88%E6%88%90.csv)', {
    resources: [file({ file: '合成.csv' })],
  });
  assert.match(encoded, /data-gpi-resource-file="合成.csv"/);
});

test('unbound relative Markdown paths explain the file without granting preview access', () => {
  const { markdown } = owners();
  const html = markdown.render('[查看结果](results/unattached.html)', { resources: [file()] });
  assert.match(html, /查看结果.*<code>results\/unattached.html<\/code>/);
  assert.doesNotMatch(html, /data-gpi-resource|href=|\[查看结果\]/);
  const otherTurn = markdown.render(`[上轮文件](${FILE})`);
  assert.doesNotMatch(otherTurn, /data-gpi-resource|href=/);
});

test('relative-link resolution rejects traversal, protocols and markup injection', () => {
  const { resources, markdown } = owners();
  for (const unsafe of ['../secret.csv', '%2e%2e/secret.csv', '/absolute.csv',
    '//outside.example/a', 'file:///private/file', 'javascript:alert(1)',
    'data:text/html,hello', 'safe.csv#fragment', 'safe.csv?query=1', 'safe%5cfile.csv', '%00bad.csv']) {
    assert.equal(resources.relativeLink([file({ file: unsafe })], unsafe, 'x'), '');
    assert.doesNotMatch(markdown.render(`[x](${unsafe})`, { resources: [file({ file: unsafe })] }),
      /data-gpi-resource|href=/);
  }
  const html = markdown.render(`[<img src=x>](${FILE}) <script>alert(1)</script>`, {
    resources: [file({ label: '"><img src=x>', checked_sha256: A })],
  });
  assert.doesNotMatch(html, /<img|<script/);
  assert.match(html, /&lt;img/);
  assert.match(html, /&lt;script&gt;/);
});

test('HTTPS references and block formatting remain supported beside file links', () => {
  const { markdown } = owners();
  const html = markdown.render(`# 简报\n- [下载表格](data/synthetic.csv)\n- [PubMed](https://pubmed.ncbi.nlm.nih.gov/26903338/)\n\n|文件|说明|\n|---|---|\n|[简报](${FILE})|合成示例|`, {
    resources: [file({ checked_sha256: A }), file({ file: 'data/synthetic.csv', sha256: B })],
  });
  assert.match(html, /class="gpi-md-heading"/);
  assert.match(html, /<ul class="gpi-md-list">/);
  assert.match(html, /<table/);
  assert.match(html, /href="https:\/\/pubmed.ncbi.nlm.nih.gov\/26903338\/"/);
  assert.match(html, /rel="noopener noreferrer"/);
  assert.equal((html.match(/data-gpi-resource-kind=/g) || []).length, 2);
});

test('the prior synthetic ICU demo link and its actual byte versions regress safely', () => {
  // Captured from the completed pi_96eaf4e0ef83c65b7419 / draft_2ea91044d505
  // task, not a new Provider execution or patient-data read. Source receipts:
  // easyicu_biomni_retest_20261029/{easyicu-completed-dom.txt,
  // easyicu-first-output-source.html,easyicu-final-download-fix.html}.
  const firstSha = '10eebcb5edab24effe498de7275823f77f310d6f833631ee003356b34ff0670e';
  const finalSha = 'b896e32cba9e77104a7147d45250b61f634a5bf1a51d7b3a8b34950a892425fc';
  const text = '[打开交互式中文 HTML 简报](icu_synthetic_demo.html)';
  const { resources, markdown } = owners();
  let rows = resources.merge([], [file({ sha256: firstSha }),
    file({ checked_sha256: firstSha }), file({ kind: 'webpage', checked_sha256: firstSha })]);
  let html = markdown.render(text, { resources: rows });
  assert.match(html, new RegExp(`data-gpi-resource-checked-digest="${firstSha}"`));
  assert.doesNotMatch(html, /\[打开交互式/);
  assert.equal((resources.renderForMessage({ resources: rows }).match(/class="gpi-file-card"/g) || []).length, 1);
  rows = resources.merge(rows, [file({ sha256: finalSha })]);
  assert.equal(rows[0].checked_sha256, undefined);
  rows = resources.merge(rows, [file({ kind: 'webpage', checked_sha256: finalSha })]);
  html = markdown.render(text, { resources: rows });
  assert.match(html, new RegExp(`data-gpi-resource-checked-digest="${finalSha}"`));
  assert.doesNotMatch(html, new RegExp(firstSha));
});

test('persisted transcript aggregation replaces the early read with its later checked snapshot', () => {
  const { resources, context } = owners();
  for (const name of ['activity', 'transcript']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  context.window.EasyICU.guidedPi.declare('replay', { lifecycleTurns: () => [] });
  const activity = { focusLatest: rows => rows };
  const owner = context.window.EasyICU.guidedPi.require('transcript').create({
    activity, tr: en => en, timeMs: value => Date.parse(value),
    modelErrorText: () => 'The provider did not finish the explanation.',
    resourceKey: resources.key, mergeResources: resources.merge,
    upsertActivityStep: (row, step) => {
      const index = row.steps.findIndex(item => item.id === step.id);
      if (index < 0) row.steps.push(step); else Object.assign(row.steps[index], step);
    },
  });
  const transcript = [{ role: 'user', timestamp: '2026-10-29T03:00:00Z',
    content: [{ type: 'text', text: '创建合成演示简报' }] }];
  [file({ sha256: A }), file({ checked_sha256: A }),
    file({ kind: 'webpage', checked_sha256: A })].forEach((resource, index) => transcript.push({
    role: 'tool', timestamp: `2026-10-29T03:00:0${index + 1}Z`,
    content: [{ type: 'tool_result', tool_call_id: `t${index}`, tool_name: 'easyicu_read_file', resource }],
  }));
  transcript.push({ role: 'assistant', timestamp: '2026-10-29T03:00:05Z',
    content: [{ type: 'text', text: `[打开简报](${FILE})` }] });
  const rows = owner.transcriptMessages({ transcript });
  const answer = rows.find(row => row.role === 'assistant');
  assert.equal(answer.resources.length, 1);
  assert.equal(answer.resources[0].checked_sha256, A);
  const failedRows = owner.transcriptMessages({ transcript: transcript.slice(0, -1).concat({
    role: 'assistant', entry_id: 'failed-entry', timestamp: '2026-10-29T03:00:05Z',
    content: [], error_code: 'pi_model_provider_unavailable',
  }) });
  const failedAnswer = failedRows.find(row => row.role === 'assistant');
  assert.equal(failedAnswer.errorCode, 'pi_model_provider_unavailable');
  assert.equal(failedAnswer.resources.length, 1);
  assert.equal(failedAnswer.resources[0].checked_sha256, A);
});

test('activity detail combines the initial resource with checked resource snapshots', () => {
  const { resources, context, esc } = owners();
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,
    '../../src/easyicu/webserver/static/js/screens-guided-pi-activity.js'), 'utf8'), context);
  const activity = context.window.EasyICU.guidedPi.require('activity').create({
    tr: en => en, esc, iconHtml: () => '', resourceName: resources.name,
    resourceKey: resources.key, resourceButton: resources.button, mergeResources: resources.merge,
  });
  const html = activity.render({ id: 'historical-activity', status: 'complete', startedAt: 1, endedAt: 2,
    steps: [{ id: 'tool-read', kind: 'tool', toolName: 'easyicu_read_file', status: 'complete',
      resource: file({ sha256: A }), resources: [file({ checked_sha256: A }),
        file({ kind: 'webpage', checked_sha256: A })] }] });
  assert.equal((html.match(/data-gpi-resource-kind=/g) || []).length, 1);
  assert.match(html, new RegExp(`data-gpi-resource-checked-digest="${A}"`));
});

test('live assistant aggregation updates both current-turn resources and the active reply', () => {
  const { resources, context } = owners();
  context.agentMode = () => 'research';
  context.state = { currentTurnResources: [file({ sha256: A })],
    messages: [{ role: 'assistant', complete: false }] };
  installLiveResources(context, resources);
  context.addAssistantResources([file({ checked_sha256: A }), file({ kind: 'webpage', checked_sha256: A })]);
  assert.equal(context.state.currentTurnResources.length, 1);
  assert.equal(context.state.messages[0].resources[0].checked_sha256, A);
  context.addAssistantResources([file({ sha256: B })]);
  assert.equal(context.state.messages[0].resources[0].checked_sha256, undefined);
  assert.equal(context.state.messages[0].resources[0].sha256, B);
});

test('live workspace HTML opens each newly checked version once without opening research artifacts', () => {
  const { resources, context } = owners();
  const opened = [];
  let mode = 'workspace';
  let project = 'project-a';
  context.agentMode = () => mode;
  context.projectId = () => project;
  context.previewWorkflowContext = () => ({ nextActionCode: 'idle' });
  context.autoPreviewedWorkspaceVersions = new Set();
  context.MODULES = { optional: () => ({ open: (resource, id) => opened.push({ resource, id }) }) };
  context.state = { currentTurnResources: [], messages: [{ role: 'assistant', complete: false }] };
  installLiveResources(context, resources);
  context.addAssistantResources([file({ sha256: A })]);
  assert.equal(opened.length, 0, 'an ordinary SHA does not open unchecked HTML');
  context.addAssistantResources([file({ checked_sha256: A })]);
  assert.equal(opened.length, 1);
  context.addAssistantResources([file({ kind: 'webpage', checked_sha256: A })]);
  context.addAssistantResources([file({ sha256: A })]);
  assert.equal(opened.length, 1, 'same-version snapshots cannot reopen a manually closed preview');
  context.addAssistantResources([file({ sha256: B })]);
  assert.equal(opened.length, 1);
  context.addAssistantResources([file({ checked_sha256: B })]);
  assert.equal(opened.length, 2);
  context.addAssistantResources([{ kind: 'research_artifact', run_id: 'governed', artifact: 'report.html', checked_sha256: A }]);
  context.addAssistantResources([file({ file: 'other.csv', media_type: 'text/csv', checked_sha256: A })]);
  assert.equal(opened.length, 2);
  mode = 'research';
  context.addAssistantResources([file({ file: 'research.html', checked_sha256: A })]);
  assert.equal(opened.length, 2);
  mode = 'workspace'; project = 'project-b';
  context.addAssistantResources([file({ checked_sha256: B })]);
  assert.equal(opened.length, 3);
  assert.equal(opened[2].id, 'project-b');
});

test('workspace workflow traces use task wording while research keeps its scientific labels', () => {
  for (const language of ['zh', 'en']) {
    const { resources, context, esc } = owners(language);
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      '../../src/easyicu/webserver/static/js/screens-guided-pi-activity.js'), 'utf8'), context);
    let mode = 'workspace';
    const activity = context.window.EasyICU.guidedPi.require('activity').create({
      tr: (en, zh) => language === 'zh' ? zh : en, esc, iconHtml: () => '',
      resourceName: resources.name, resourceKey: resources.key, resourceButton: resources.button,
      agentMode: () => mode,
    });
    const step = { kind: 'tool', toolName: 'easyicu_inspect_workflow', status: 'running' };
    assert.match(activity.stepLabel(step), language === 'zh' ? /任务工作流程/ : /task workflow/);
    assert.doesNotMatch(activity.stepLabel(step), /科研|research/);
    assert.match(activity.stepLabel({ ...step, status: 'complete' }), language === 'zh' ? /已检查任务工作流程/ : /Checked task workflow/);
    mode = 'research';
    assert.match(activity.stepLabel(step), language === 'zh' ? /检查科研流程/ : /Inspect research workflow/);
    assert.match(activity.stepLabel({ ...step, status: 'complete' }), language === 'zh' ? /已读取科研流程/ : /Read research workflow/);
  }
});

test('failed tool details show the safe typed cause and never the successful no-file fallback', () => {
  for (const language of ['zh', 'en']) {
    const { resources, context, esc } = owners(language);
    context.window.icon = () => '';
    for (const name of ['activity', 'source-view']) {
      vm.runInContext(fs.readFileSync(path.resolve(__dirname,
        `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
    }
    const activity = context.window.EasyICU.guidedPi.require('activity').create({
      tr: (en, zh) => language === 'zh' ? zh : en, esc, iconHtml: () => '',
      resourceName: resources.name, resourceKey: resources.key, resourceButton: resources.button,
      agentMode: () => 'workspace', runFailureText: code => `Safe reason (${code})`,
    });
    const view = context.window.EasyICU.guidedPi.require('sourceView');
    let rendered = '';
    view.mount({ addEventListener() {}, closest: () => null, replaceChildren() {},
      set innerHTML(value) { rendered = value; }, get innerHTML() { return rendered; } });
    for (const code of ['pi_workspace_write_authorization_required', 'pi_workspace_file_unavailable', '',
      '<script>sk-private-secret</script>']) {
      const trace = activity.render({ status: 'failed', startedAt: 1, endedAt: 2,
        steps: [{ id: 'write', kind: 'tool', toolName: 'easyicu_write_project_file', status: 'error', code,
          summary: 'sk-private-secret', text: 'sk-private-secret', receipt: { summary: 'sk-private-secret' } }] });
      const payload = trace.match(/data-gpi-operation="([^"]*)"/)[1];
      const operation = JSON.parse(decodeURIComponent(payload));
      assert.match(operation.detail, language === 'zh' ? /未完成：/ : /Not completed:/);
      if (code === 'pi_workspace_write_authorization_required') {
        assert.match(operation.detail, language === 'zh' ? /没有获得写入项目文件的授权/ : /did not authorize writing project files/);
      }
      assert.ok(view.openOperation({ dataset: { gpiOperation: payload } }, 'project-a'));
      assert.doesNotMatch(rendered, /只更新了运行记录|updated the run record and produced no file|sk-private-secret|<script>/);
      assert.match(rendered, language === 'zh' ? /未完成：/ : /Not completed:/);
    }
  }
});
