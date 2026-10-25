/* Plain report labels keep the saved manuscript state, units and evidence. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// An optional checkout root exercises exactly the same fixtures on the base.
const ROOT = path.resolve(process.argv[2] || path.join(__dirname, '../..'));
const JS = path.join(ROOT, 'src/easyicu/webserver/static/js');

function reader(lang) {
  const window = { EU_LANG: lang, t: (en, zh) => lang === 'zh' ? zh : en, icon: () => '' };
  const context = vm.createContext({ window });
  for (const file of [
    'html-escape.js', 'screens-guided-pi-modules.js', 'screens-agent-reader-vocab.js',
    'screens-agent-render.js', 'screens-guided-pi-resources.js',
    'screens-guided-pi-analysis-report.js', 'screens-guided-pi-article-report.js',
    'screens-guided-pi-data-preview.js',
  ]) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return {
    window, renderer: window.AGENT_RENDER,
    resources: window.EasyICU.guidedPi.require('resources').create({ esc: window.EU_HTML.esc }),
    analysis: window.EasyICU.guidedPi.require('analysisReport'),
    article: window.EasyICU.guidedPi.require('articleReport'),
    data: window.EasyICU.guidedPi.require('dataPreview'),
  };
}

const SHA = 'a'.repeat(64);
function claim(field, value, display, overrides = {}) {
  return {
    source_field: field, source_json_pointer: `/${field.replace(/\[(\d+)\]/g, '/$1').replaceAll('.', '/')}`,
    source_value: String(value), canonical_value: value, display_value: display,
    status: 'current', evidence: { evidence_id: 'synthetic_result', sha256: SHA, status: 'current' },
    ...overrides,
  };
}
function report() {
  return {
    plan: { steps: [{ method: 'summary', intent: 'Recorded marker κ (ng/mL)' }] },
    run_context: { question: 'Compare recorded marker bands.' },
    manuscript_provenance: {
      claims: [
        claim('cohort.n_stays', 12, '12'), claim('n_total', 9, '9'),
        claim('primary_or', 1.23456, '1.235'),
        claim('primary_or_ci[0]', 0.98765, '0.988'),
        claim('primary_or_ci[1]', 1.54321, '1.543'),
      ],
      article_blocks: [
        { kind: 'heading', level: 2, segments: [{ text: 'Limitations' }] },
        { kind: 'paragraph', segments: [{ text: 'Repeated records may be dependent; this association does not establish causation.' }] },
      ],
    },
  };
}

for (const lang of ['zh', 'en']) {
  const zh = lang === 'zh';
  test(`${lang}: plan and study-review titles use plain labels`, () => {
    const { renderer } = reader(lang);
    assert.equal(renderer.artifactTitle('agent_plan.json'), zh ? '分析计划' : 'Analysis plan');
    assert.equal(renderer.artifactTitle('scientific_readiness.json'), zh ? '研究审阅' : 'Study review');
  });

  test(`${lang}: a manuscript title is pending only for its saved pending status`, () => {
    const { renderer } = reader(lang);
    const pending = zh ? '论文草稿（待审阅）' : 'Manuscript draft (for review)';
    const neutral = zh ? '论文草稿' : 'Manuscript draft';
    const payload = { status: 'locked_pending_human_review', claims: [] };
    assert.equal(renderer.artifactTitle('manuscript_draft.json', payload), pending);
    assert.match(renderer.artifactStructuredView('manuscript_draft.json', payload), new RegExp(pending.replace(/[()]/g, '\\$&')));
    for (const status of [undefined, '', 'locked', 'blocked', 'paused', 'not_generated', 'signed', 'publication_ready', '<img src=x>']) {
      assert.equal(renderer.artifactTitle('manuscript_draft.json', { status }), neutral, String(status));
    }
  });

  test(`${lang}: a not-generated manuscript marker overrides a pending wrapper status`, () => {
    const { renderer } = reader(lang);
    const neutral = zh ? '论文草稿' : 'Manuscript draft';
    for (const markdown_preview of [
      'Manuscript scaffold not generated\n\nA saved diagnostic.',
      '# Manuscript scaffold not generated\n\nA saved diagnostic.',
      ...['blocked', 'paused', 'probe'].map(kind => `# Saved diagnostic\n<!-- easyicu:manuscript-state kind=${kind} reason_code=synthetic_stop -->`),
    ]) {
      const payload = { status: 'locked_pending_human_review', markdown_preview, claims: [] };
      assert.equal(renderer.artifactTitle('manuscript_draft.json', payload), neutral);
      const html = renderer.artifactStructuredView('manuscript_draft.json', payload);
      assert.doesNotMatch(html, /论文草稿（待审阅）|Manuscript draft \(for review\)/);
    }
  });

  test(`${lang}: report resources use reader titles and retain host-owned labels`, () => {
    const { resources } = reader(lang);
    for (const [artifact, en, chinese] of [
      ['full_analysis_report.json', 'Research results and interpretation', '研究结果与解读'],
      ['article_report.json', 'Manuscript draft', '论文草稿'],
      ['technical_report.json', 'Technical report', '技术报告'],
    ]) {
      assert.equal(resources.label({ kind: 'research_report', artifact, label: artifact }), zh ? chinese : en);
    }
    const resource = { kind: 'research_artifact', artifact: 'agent_plan.json', conversation_label: 'Saved owner title' };
    assert.equal(resources.label(resource), 'Saved owner title');
  });

  test(`${lang}: resource titles retain manuscript state through the existing transport`, () => {
    const { resources } = reader(lang);
    const resource = { kind: 'research_artifact', run_id: 'run_synthetic', artifact: 'manuscript_draft.json',
      state: 'locked_pending_human_review', sha256: SHA };
    const pending = zh ? '论文草稿（待审阅）' : 'Manuscript draft (for review)';
    assert.equal(resources.label(resource), pending);
    const html = resources.button(resource);
    const decoded = resources.fromButton({ dataset: Object.fromEntries(Array.from(
      html.matchAll(/data-([a-z-]+)="([^"]*)"/g), ([, key, value]) =>
        [key.replace(/-([a-z])/g, (_, char) => char.toUpperCase()), value],
    )) });
    assert.equal(decoded.state, resource.state);
    assert.equal(decoded.sha256, SHA);
    assert.equal(decoded.run_id, resource.run_id);
    assert.equal(decoded.artifact, resource.artifact);
    assert.equal(resources.label(decoded), pending);
    assert.equal(resources.label({ ...resource, state: 'signed' }), zh ? '论文草稿' : 'Manuscript draft');
  });

  test(`${lang}: raw artifact names remain in folded file details`, () => {
    const { resources } = reader(lang);
    const resource = { kind: 'research_artifact', run_id: 'run_synthetic', artifact: 'agent_plan.json', sha256: SHA,
      evidence_id: 'synthetic_result', evidence_sha256: SHA, evidence_pointer: '/cohort/n_stays' };
    const html = resources.renderForMessage({ resources: [resource] });
    assert.match(html, new RegExp(`>${zh ? '分析计划' : 'Analysis plan'}</button>`));
    assert.match(html, /<details[^>]*><summary>[^<]+<\/summary><span>agent_plan\.json<\/span><\/details>/);
    assert.doesNotMatch(html, /<details[^>]*\bopen(?:\s|>)/);
    assert.match(html, /data-gpi-resource-run="run_synthetic"/);
    assert.match(html, /data-evidence-pointer="\/cohort\/n_stays"/);
    assert.match(html, new RegExp(`data-evidence-sha256="${SHA}"`));
  });

  test(`${lang}: stay counts, exact evidence and original units survive the copy change`, () => {
    const { analysis } = reader(lang);
    const payload = report();
    const original = JSON.stringify(payload);
    const html = analysis.render(payload);
    for (const label of zh ? ['筛选前 ICU 入住记录', '符合条件的 ICU 入住记录'] : ['ICU stays before screening', 'Eligible ICU stays']) {
      assert.ok(html.includes(`<span>${label}</span>`), label);
    }
    assert.match(html, /<strong>12<\/strong>/);
    assert.match(html, /<strong>9<\/strong>/);
    assert.match(html, /OR 1\.235 ?[（(]95% CI 0\.988–1\.543[）)]/);
    assert.match(html, /data-evidence-source-value="12"/);
    assert.match(html, /data-evidence-pointer="\/cohort\/n_stays"/);
    assert.match(html, /Recorded marker κ \(ng\/mL\)/);
    assert.equal(JSON.stringify(payload), original);
  });

  test(`${lang}: interpretation stays beside results without the implementation caption`, () => {
    const { analysis } = reader(lang);
    const html = analysis.render(report());
    const summary = html.match(/<section class="gpi-analysis-summary">([\s\S]*?)<\/section>/)[1];
    assert.match(summary, /Repeated records may be dependent; this association does not establish causation\./);
    assert.ok(html.includes(zh
      ? '主要分析和敏感性分析可能回答不同问题，请按各自的人群、比较和结局解读。'
      : 'Primary and sensitivity analyses may address different questions; interpret each using its population, comparison and outcome.'));
    assert.doesNotMatch(html, /所有展示的数值和图件均投影|Every displayed number and figure is projected|本 Web 报告不重新计算|This Web report does not recalculate/);
    if (zh) assert.doesNotMatch(html, /estimand/);
    const fallbackReport = analysis.render({});
    if (zh) assert.doesNotMatch(fallbackReport, /estimand/);
    const fallback = fallbackReport.match(/<section class="gpi-analysis-summary">([\s\S]*?)<\/section>/)[1];
    assert.ok(fallback.includes(zh ? '不一定是相互独立的患者' : 'not necessarily independent patients'));
  });

  test(`${lang}: stale evidence and unrelated intervals cannot become matching evidence`, () => {
    const { analysis } = reader(lang);
    const payload = report();
    payload.manuscript_provenance.claims[2].status = 'stale';
    const html = analysis.render(payload);
    assert.doesNotMatch(html, /data-evidence-pointer="\/primary_or"/);
    assert.doesNotMatch(html, /95% CI 0\.988–1\.543/);
    assert.ok(html.includes(zh ? '缺少与主要估计匹配的置信区间' : 'no matching primary confidence interval'));
    const other = report();
    other.manuscript_provenance.claims[3].evidence.sha256 = 'b'.repeat(64);
    assert.doesNotMatch(analysis.render(other), /95% CI 0\.988–1\.543/);
  });

  test(`${lang}: absent preview owners explain the failure without invented controls`, () => {
    const { window, data, article } = reader(lang);
    const text = zh ? '暂时无法显示数据预览，请重试。' : 'The data preview could not be displayed. Please retry.';
    const host = { innerHTML: '' };
    data.mount(host, {}, 'cohort_summary');
    assert.ok(host.innerHTML.includes(text));
    assert.doesNotMatch(host.innerHTML, /<button|data-.*retry|renderer|Workbench/);
    delete window.AGENT_RENDER;
    assert.throws(() => article.render({}), error => error.message === (zh
      ? '暂时无法显示论文草稿，请重试。' : 'The manuscript preview could not be displayed. Please retry.'));
  });

  test(`${lang}: available preview owners receive the original payload`, () => {
    const { window, data, article } = reader(lang);
    const payload = { marker: 'Synthetic registered display (μg/L)' };
    const calls = [];
    window.EU_VIZ_EMBEDDED_WORKBENCH = {
      render: (...args) => { calls.push(args); return '<section>Native preview</section>'; },
      mount: (...args) => { calls.push(args); },
    };
    assert.equal(data.render(payload, 'feature_distribution'), '<section>Native preview</section>');
    const host = {};
    data.mount(host, payload, 'feature_distribution');
    assert.equal(calls[0][0], payload);
    assert.equal(calls[0][1], 'feature_distribution');
    assert.equal(calls[1][0], host);
    assert.equal(calls[1][1], payload);
    window.AGENT_RENDER.manuscriptProvenanceView = actual => {
      assert.equal(actual, payload); return '<article>Saved manuscript</article>';
    };
    assert.equal(article.render(payload), '<article>Saved manuscript</article>');
  });
}
