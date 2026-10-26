'use strict';

// The claim drawer words the provenance owner's typed lineage nodes; it never
// infers a link the owner did not prove and keeps IDs/checksums one click away.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function reader(language = 'en') {
  const window = {
    t: (english, chinese) => (language === 'zh' ? chinese : english),
    icon: () => '',
  };
  const context = vm.createContext({ window, Intl });
  for (const file of ['html-escape.js', 'screens-agent-report-reader.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return window.AGENT_REPORT_READER;
}

const hooks = {
  artifactTable: (title, columns, rows) => `<table data-title="${title}">${rows.map(row => `<tr><td>${row.join('</td><td>')}</td></tr>`).join('')}</table>`,
  manuscriptUnavailable: () => false,
};
const sha = character => character.repeat(64);
const node = (kind, status, extra = {}) => ({
  kind, status, sha256: status === 'missing' ? null : sha('a'), evidence_id: status === 'missing' ? null : `${kind}_record`,
  facts: {}, missing_reason: null, ...extra,
});
function view(lineage, language) {
  return reader(language).manuscriptProvenanceView({
    article_blocks: [{ kind: 'paragraph', segments: [{ kind: 'claim', text: '0.82', claim_id: 'claim_1' }] }],
    claims: [{
      claim_id: 'claim_1', display_value: '0.82', source_field: 'estimate', source_json_pointer: '/estimate',
      step_id: 'primary', evidence: { evidence_id: 'summary', sha256: sha('b') }, related_artifacts: [],
      ...(lineage ? { lineage } : {}),
    }],
  }, hooks);
}
const fullChain = () => [
  node('statistic', 'current', { facts: { method: 'logistic regression <img src=x>' } }),
  node('analysis_input', 'current', { facts: { rows: 1234 } }),
  node('cohort', 'current', { facts: { rows: 1180, window_start_hours: 0, window_end_hours: 24 } }),
  node('column_definitions', 'current', { facts: { count: 7 } }),
  node('concept_dictionary', 'current', { facts: { concept_count: 256, base_sha256: sha('c') } }),
  node('data_source', 'current', { facts: {
    database: 'eicu_demo', version: '2.0.1', official_sha256_matches: true, doi: '10.13026/4mxk-na84',
  } }),
];
const chainOf = html => {
  const match = html.match(/<ol class="gpi-lineage">([\s\S]*?)<\/ol>/);
  assert.ok(match, 'a claim with lineage renders the chain');
  return match[1].split('</li>').filter(item => item.includes('<li'));
};

test('a proven chain reads from the number back to its data source', () => {
  const html = view(fullChain());
  const items = chainOf(html);
  assert.deepEqual(items.map(item => item.match(/<strong>([^<]+)<\/strong>/)[1]),
    ['Result', 'Analysis table', 'Study population', 'Variable definitions', 'Concept dictionary', 'Data source']);
  assert.ok(items.every((item, index) => item.includes(`class="is-current" style="--i:${index}"`)), 'staggered reveal index per node');
  assert.match(items[0], /logistic regression &lt;img src=x&gt;/, 'facts are escaped');
  assert.match(items[1], /1,234 rows/);
  assert.match(items[2], /1,180 included · window 0–24 h/);
  assert.match(items[3], /7 variables/);
  assert.match(items[4], /256 concepts/);
  assert.match(items[5], /eICU-CRD Demo 2\.0\.1 · <a href="https:\/\/doi\.org\/10\.13026\/4mxk-na84"[^>]*>matches the official PhysioNet release<\/a>/);
  // Each node opens its own record; the tab is named after the node and the
  // preview renderer follows the record kind.
  assert.match(items[1], /data-evidence-id="analysis_input_record"[^>]*data-evidence-kind="table"[^>]*data-evidence-label="Analysis table"[^>]*>Open<\/button>/);
  assert.match(items[0], /data-evidence-kind="statistic"/);
  assert.match(items[2], /data-evidence-kind="artifact"/);
  assert.match(html, /<details class="gpi-claim-technical"><summary>IDs and checksums<\/summary><table data-title="Exact result source">/,
    'IDs, pointers and digests move behind one disclosure');
});

test('a broken link says what is missing and offers nothing to open', () => {
  const lineage = fullChain();
  lineage[1] = node('analysis_input', 'stale', { missing_reason: 'input_digest_mismatch' });
  lineage[2] = node('cohort', 'missing', { missing_reason: 'upstream_unproven' });
  lineage[3] = node('column_definitions', 'missing', { missing_reason: 'some_future_code' });
  lineage[5] = node('data_source', 'current', { facts: { database: 'eicu', version: '2.0', official_sha256_matches: false } });
  const items = chainOf(view(lineage));
  assert.match(items[1], /class="is-stale"[\s\S]*<span class="pill warn">changed<\/span>[\s\S]*differs from the registered one/);
  assert.doesNotMatch(items[1], /<button/, 'a stale record is not offered as the source');
  assert.match(items[2], /class="is-missing"[\s\S]*the step above could not be verified/);
  assert.doesNotMatch(items[2], /<button/);
  assert.match(items[3], /<p>Not recorded<\/p>/, 'an unknown reason falls back to a neutral line');
  lineage[3] = node('column_definitions', 'missing', { missing_reason: 'upstream_unproven' });
  const repeated = chainOf(view(lineage));
  assert.match(repeated[2], /the step above could not be verified/, 'the first unverified step says why');
  assert.match(repeated[3], /<strong>Variable definitions<\/strong><\/div>/, 'later unverified steps show only their marker');
  assert.match(items[5], /eICU-CRD 2\.0 · <span class="is-warn">does not match the official release<\/span>/);
});

test('unsafe or unknown values never become links or nodes', () => {
  const lineage = fullChain();
  lineage[5].facts.doi = '10.13026/x" onclick="alert(1)';
  lineage.push(node('patient_rows', 'current'));
  const html = view(lineage);
  const items = chainOf(html);
  assert.equal(items.length, 6, 'only the six owner-declared kinds render');
  assert.doesNotMatch(items[5], /<a /, 'an invalid DOI keeps the plain sentence');
  assert.match(items[5], /matches the official PhysioNet release/);
  assert.doesNotMatch(html, /onclick=/);
});

test('readers saved before lineage keep their tables in view', () => {
  const html = view(null);
  assert.doesNotMatch(html, /gpi-lineage|gpi-claim-technical/);
  assert.match(html, /<table data-title="Exact result source">/);
});

test('the chain is worded for Chinese readers', () => {
  const items = chainOf(view(fullChain(), 'zh'));
  assert.match(view(fullChain(), 'zh'), /<h3>这个数字的来源<\/h3>/);
  assert.match(items[2], /<strong>研究人群<\/strong><p>纳入 1,180 例 · 时间窗 0–24 小时<\/p>/);
  assert.match(items[5], /与 PhysioNet 官方发布一致/);
  assert.match(items[1], />打开<\/button>/);
});
