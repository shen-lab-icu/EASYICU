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

test('missing review metadata remains unknown, distinct from explicit false and empty arrays', () => {
  const render = renderer('zh');
  const html = render.scientificReadinessView({ domains: [{ domain: 'literature' }] });
  assert.match(html, /<dt>人工审阅<\/dt><dd>未提供<\/dd>/);
  assert.match(html, /<dt>投稿使用<\/dt><dd>未提供<\/dd>/);
  assert.match(html, /<dt>论断上限<\/dt><dd>未提供<\/dd>/);
  assert.match(html, /未提供证据数量/);
  assert.match(html, /未提供科学问题清单/);
  const explicit = render.scientificReadinessView({
    human_review_required: false, paper_authorized: false,
    domains: [{ domain: 'literature', evidence_refs: [] }], findings: [],
  });
  assert.match(explicit, /<dt>人工审阅<\/dt><dd>不需要<\/dd>/);
  assert.match(explicit, /<dt>投稿使用<\/dt><dd>未授权<\/dd>/);
  assert.match(explicit, /0 项证据/);
  assert.match(explicit, /没有待处理科学问题/);
});

test('known review statements have faithful Chinese copy; unknown text stays intact', () => {
  const render = renderer('zh');
  const html = render.scientificReadinessView({ domains: [
    { domain: 'literature', summary: 'The current literature package cannot establish currency or design support.' },
    { domain: 'data', summary: 'Provenance exists, but scientific population scope is not fully established.' },
    { domain: 'analysis', summary: 'Unrecognised source statement.' },
  ] });
  assert.match(html, /当前文献资料尚不能证明证据的时效性或对研究设计的支持/);
  assert.match(html, /已有来源记录，但研究人群的范围尚未完全明确/);
  assert.match(html, /Unrecognised source statement\./);
});
