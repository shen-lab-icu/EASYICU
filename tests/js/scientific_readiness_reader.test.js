'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const JS = path.join(__dirname, '../../src/easyicu/webserver/static/js');

function renderer(lang) {
  const window = { EU_LANG: lang, t: (en, zh) => lang === 'zh' ? zh : en, icon: () => '' };
  const context = vm.createContext({ window });
  for (const file of ['html-escape.js', 'screens-agent-reader-vocab.js', 'screens-agent-report-reader.js', 'screens-agent-render.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return window.AGENT_RENDER;
}
const finding = n => ({ code: `UNKNOWN_${n}`, domain: 'analysis', severity: 'blocker', message: `detail ${n}`, remediation: `remedy ${n}` });
const payload = {
  schema_version: 'easyicu.web-scientific-readiness/1', run_id: 'run_test', status: 'blocked',
  claim_ceiling: 'unsupported', human_review_required: true, paper_authorized: false,
  domains: [{ domain: 'literature', status: 'review_required', evidence_refs: ['a', 'b'], summary: 'Domain evidence is incomplete.' }],
  findings: [finding(1), finding(2), finding(3), finding(4)],
};
for (const lang of ['zh', 'en']) {
  test(`${lang}: readiness reader retains boundary, evidence count and every finding in a disclosure`, () => {
    const html = renderer(lang).artifactStructuredView('scientific_readiness.json', payload);
    for (const expected of ['run_test', 'detail 1', 'remedy 1', 'detail 4', 'remedy 4', 'Domain evidence is incomplete.', 'ag-readiness-domain', 'ag-readiness-finding', '<details class="ag-readiness-findings">']) assert.ok(html.includes(expected), expected);
    assert.ok(html.includes(lang === 'zh' ? '2 项证据' : '2 references'));
    assert.ok(html.includes(lang === 'zh' ? '未授权' : 'Not authorized'));
    assert.ok(!html.includes('<table'), 'readiness remains readable in a narrow panel');
  });
  test(`${lang}: a small or empty finding list is expanded and source text is escaped`, () => {
    const html = renderer(lang).artifactStructuredView('scientific_readiness.json', {
      ...payload, findings: [{ ...finding(1), message: '<script>unsafe</script>' }], domains: [],
    });
    assert.ok(!html.includes('<details class="ag-readiness-findings">'));
    assert.ok(html.includes('&lt;script&gt;unsafe&lt;/script&gt;'));
    assert.ok(!html.includes('<script>unsafe</script>'));
    assert.ok(html.includes(lang === 'zh' ? '没有可展示的审阅领域' : 'No review domains are present'));
  });
}

test('zh: historical and current blocked summaries do not invent completed analysis', () => {
  const render = renderer('zh');
  for (const summary of [
    'Execution completeness does not close the open scientific design defects.',
    'Analysis completion has not been confirmed; the scientific assessment remains open.',
  ]) {
    const html = render.artifactStructuredView('scientific_readiness.json', {
      ...payload, domains: [{ domain: 'analysis', status: 'blocked', summary }],
    });
    assert.ok(!html.includes('分析执行完成，不代表'));
    assert.ok(html.includes(summary.startsWith('Execution') ? '即使分析执行完整' : '尚未确认分析已完成'));
  }
});
