'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.resolve(process.argv[2]), 'utf8');
const sandbox = {
  window: {
    t: (english) => english,
    icon: () => '',
  },
};
/* html-escape.js owns esc/escAttr and this module destructures them at the
   top of its IIFE, so it loads first here exactly as it does in index.html. */
const escapeOwner = path.join(path.dirname(path.resolve(process.argv[2])), 'html-escape.js');
const context = vm.createContext(sandbox);
vm.runInNewContext(fs.readFileSync(escapeOwner, 'utf8'), context, { filename: escapeOwner });
vm.runInNewContext(source, context, { filename: process.argv[2] });
const renderer = sandbox.window.AGENT_RENDER;

const hostileLabel = 'figure" onerror="globalThis.pwned=1';
const safePng = 'data:image/png;base64,iVBORw0KGgo=';
const escaped = renderer.figureGallery({
  figures: [{ label: hostileLabel, data_url: safePng }],
});
assert.ok(escaped.includes('&quot;'), 'attribute quotes must be entity escaped');
assert.ok(!escaped.includes('alt="figure" onerror='), 'label must not create a new attribute');

const captioned = renderer.figureGallery({
  figures: [{ label: 'Coverage', data_url: safePng,
    caption: 'Unknown is not zero. <script>alert(1)</script>' }],
});
assert.match(captioned, /<figcaption>[\s\S]*<p class="ag-figure-caption">Unknown is not zero\./,
  'the explanatory caption must be outside the image in readable page text');
assert.ok(captioned.includes('&lt;script&gt;'), 'caption markup must be escaped');
assert.ok(!captioned.includes('<script>'), 'caption must not execute markup');
assert.ok(!escaped.includes('ag-figure-caption'), 'legacy figures need no empty caption paragraph');

const hostileSource = renderer.figureGallery({
  figures: [{
    label: 'bad source',
    data_url: 'data:image/png;base64,AAAA" onerror="globalThis.pwned=1',
  }],
});
assert.equal(hostileSource, '', 'malformed image data URL must not create an image');

const activeData = renderer.figureGallery({
  figures: [{ label: 'html', data_url: 'data:text/html,<script>alert(1)</script>' }],
});
assert.equal(activeData, '', 'only bounded PNG data URLs may render');

const hostileKey = '<img src=x onerror="globalThis.pwned=1">';
const hostileStructured = renderer.artifactStructuredView('fallback.json', {
  rows: [{ [hostileKey]: 'value' }],
});
assert.ok(hostileStructured.includes('&lt;img'), 'object keys must be rendered as text');
assert.ok(!hostileStructured.includes(`<th>${hostileKey}</th>`), 'object key must not create table-header markup');

const readiness = renderer.artifactStructuredView('scientific_readiness.json', {
  schema_version: 'easyicu.web-scientific-readiness/1', run_id: 'run_review', status: 'analysis_only',
  claim_ceiling: 'analysis_only', human_review_required: true, paper_authorized: false,
  domains: [{ domain: 'idea', status: 'not_assessed', evidence_refs: ['idea_handoff'],
    summary: 'Technical executability is not evidence of novelty or publication value.' }],
  findings: [{ severity: 'blocker', domain: 'idea', code: 'IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED',
    message: 'No accepted, digest-bound prior-art review proves that this question is a reliable or sufficiently differentiated research idea.',
    remediation: 'Run Idea Mining prior-art retrieval, inspect same-topic hits, and accept the refreshed handoff before treating the idea as publishable.' }],
});
assert.match(readiness, /Scientific review/);
assert.match(readiness, /Idea and novelty/);
assert.match(readiness, /Prior-art evidence is not established/);
assert.doesNotMatch(readiness, /IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED|not_assessed/,
  'readable review must not expose backend status and finding codes');

const hostileTableChrome = renderer.artifactTable(
  '<svg onload="globalThis.pwned=2">',
  ['safe'],
  [],
  '<img src=x onerror="globalThis.pwned=3">',
);
assert.ok(hostileTableChrome.includes('&lt;svg'), 'table title must be rendered as text');
assert.ok(hostileTableChrome.includes('&lt;img'), 'empty-state text must be rendered as text');
assert.ok(!hostileTableChrome.includes('<svg onload='), 'table title must not create markup');

const planRows = renderer.stepRowsFrom({
  steps: [{
    step_id: '06_primary_adjusted_association',
    intent: 'Estimate the adjusted binary association.',
    planned_analysis_role: 'primary',
    expected_outputs: ['table:adjusted_association_estimates'],
  }],
});
assert.deepEqual(
  JSON.parse(JSON.stringify(planRows)),
  [[
    '06_primary_adjusted_association',
    'Estimate the adjusted binary association.',
    'planned · primary',
    'table:adjusted_association_estimates',
  ]],
  'plan previews must expose the typed step identity, intent, role, and outputs',
);

assert.equal(renderer.fmtCount(null), '—', 'missing denominators must not render as zero');
assert.equal(renderer.fmtCount(undefined), '—');
assert.equal(renderer.fmtCount(''), '—');

const manuscriptReader = renderer.manuscriptProvenanceView({
  schema_version: 'easyicu.manuscript-provenance/1',
  article_blocks: [{
    kind: 'paragraph',
    segments: [
      { kind: 'text', text: '**Results:** <img src=x onerror="globalThis.pwned=4"> [research_context] [@paper_key]' },
      { kind: 'claim', text: '0.5', claim_id: 'claim_1" onclick="globalThis.pwned=5' },
    ],
  }],
  claims: [{
    claim_id: 'claim_1" onclick="globalThis.pwned=5',
    display_value: '0.5',
    source_value: '0.5',
    source_field: 'runtime.spline_knot_quantiles[1]',
    source_json_pointer: '/runtime/spline_knot_quantiles[1]',
    step_id: 'primary',
    evidence: { evidence_id: 'summary', sha256: 'a'.repeat(64) },
    related_artifacts: [],
  }],
});
assert.ok(manuscriptReader.includes('data-gpi-claim='), 'bound numbers must be interactive');
assert.match(
  manuscriptReader,
  /class="gpi-bound-number"[^>]*data-gpi-evidence-open[^>]*data-evidence-id="summary"[^>]*data-evidence-sha256="a{64}"/,
  'bound numbers with valid evidence must open the exact result preview directly',
);
assert.ok(manuscriptReader.includes('open its exact result evidence preview'), 'reader must explain the primary click action');
assert.ok(manuscriptReader.includes('JSON field'), 'reader must expose the exact JSON field');
assert.ok(manuscriptReader.includes('Open registered evidence'), 'reader must expose execution lineage');
assert.ok(manuscriptReader.includes('data-gpi-evidence-open'), 'registered evidence must be actionable');
assert.ok(manuscriptReader.includes('<strong>Results:</strong>'), 'readable prose must render basic emphasis');
assert.ok(manuscriptReader.includes('class="gpi-reader-citation"'), 'literature bindings need a readable citation marker');
assert.ok(!manuscriptReader.includes('[research_context]'), 'internal evidence ids must stay out of readable prose');
assert.ok(!manuscriptReader.includes('<img src=x'), 'article text must be escaped');
assert.ok(!manuscriptReader.includes('onclick="globalThis.pwned=5'), 'claim ids must not create handlers');

const assembledPayload = {
  schema_version: 'easyicu.manuscript-provenance/1',
  article_blocks: [
    { kind: 'heading', level: 2, segments: [{ kind: 'text', text: 'Results' }] },
    { kind: 'paragraph', segments: [{ kind: 'text', text: 'See Table 1. Sources [@first; @second]. [00_probe] Preserve [95% CI].' }] },
    { kind: 'heading', level: 2, segments: [{ kind: 'text', text: 'Discussion' }] },
  ],
  tables: [{ label: 'Table 1', caption: 'Baseline', columns: ['Variable', 'N'], rows: [['<svg onload=bad()>', '120']], notes: ['Unknown is not zero.'] }],
  references: [
    { key: 'first', number: 1, title: '<script>bad()</script>', authors: ['A Author'], year: '2020', url: 'javascript:bad()', bibliographic_notices: ['Correction: <img src=x onerror=bad()>'] },
    { key: 'second', number: 2, title: 'Second source.', authors: [], year: '2021', doi: '10.1234/test' },
  ],
  figure_gallery: { figures: [{ label: 'Figure', data_url: safePng, caption: 'Bound caption.' }] },
  report_revision: { status: 'pass', revision_id: 'revision-1' },
};
const assembled = renderer.artifactStructuredView('manuscript_draft.json', { reader: assembledPayload });
assert.ok(assembled.includes('gpi-manuscript-article'), 'new drafts use the actual article reader');
assert.ok(assembled.includes('Table 1. Baseline'));
assert.equal((assembled.match(/Bound caption\./g) || []).length, 1);
assert.ok(assembled.indexOf('Table 1. Baseline') < assembled.indexOf('<h2>Discussion</h2>'));
assert.ok(assembled.includes('href="#gpi-reference-1"') && assembled.includes('href="#gpi-reference-2"'));
assert.ok(assembled.includes('References') && assembled.includes('A Author'));
assert.ok(!assembled.includes('[00_probe]'), 'numeric-prefixed internal step labels are not article prose');
assert.ok(assembled.includes('[95% CI]'), 'scientific bracketed labels must remain visible');
assert.ok(assembled.includes('Some source records have no author metadata'));
assert.ok(!assembled.includes('source.</a>.'), 'source title punctuation must not be duplicated');
assert.ok(assembled.includes('Correction: &lt;img'));
assert.ok(assembled.includes('revision-1'));
assert.ok(!assembled.includes('href="javascript:'));
assert.ok(!assembled.includes('<script>') && !assembled.includes('<svg onload'));
const withoutDiscussion = renderer.manuscriptProvenanceView({ ...assembledPayload, article_blocks: [] });
assert.equal((withoutDiscussion.match(/Bound caption\./g) || []).length, 1, 'figures are not lost when a heading is missing');

const preciseClaim = {
  claim_id: 'precision', display_value: '23.46%', source_value: '23.4568',
  canonical_value: 23.456789, source_json_pointer: '/estimate_pct',
  evidence: { evidence_id: 'summary', sha256: 'a'.repeat(64), kind: 'statistic' },
};
const precisionReader = claim => renderer.manuscriptProvenanceView({
  claims: [claim], article_blocks: [{ kind: 'paragraph', segments: [
    { kind: 'claim', claim_id: claim.claim_id, text: claim.display_value },
  ] }],
});
const precise = precisionReader(preciseClaim);
assert.equal((precise.match(/data-evidence-source-value="23.456789"/g) || []).length, 2,
  'the number and audit link must carry the canonical source, not its rounded lexical label');
assert.ok(!precise.includes('data-evidence-source-value="23.4568"'));
assert.ok(precisionReader({ ...preciseClaim, canonical_value: 0 }).includes('data-evidence-source-value="0"'));
const { canonical_value: _canonical, ...legacyClaim } = preciseClaim;
assert.ok(precisionReader(legacyClaim).includes('data-evidence-source-value="23.4568"'),
  'legacy readers retain their source label for exact verification, not guessed extra precision');
for (const invalid of [NaN, Infinity, '23.456789']) {
  const rejected = precisionReader({ ...preciseClaim, canonical_value: invalid });
  assert.ok(rejected.includes('data-evidence-source-value=""'));
  assert.ok(!rejected.includes('data-evidence-source-value="23.4568"'));
}


const readerContext = renderer.manuscriptProvenanceView({
  claims: [{
    claim_id: 'claim_7',
    display_value: '1.96',
    source_field: 'adjusted.odds_ratio',
    source_json_pointer: '/adjusted/odds_ratio',
    source_value: '1.96',
    canonical_value: 1.964,
    step_id: 'primary_adjusted_model',
    status: 'stale',
    method_summary: { intent: 'Estimate the adjusted association.' },
    evidence: { evidence_id: 'primary_summary', sha256: 'a'.repeat(64), kind: 'statistic' },
    related_artifacts: [{
      evidence_id: 'table_one', role: 'supporting_artifact', kind: 'table',
      sha256: 'b'.repeat(64), display_id: 'Table 1', display_contract_sha256: 'c'.repeat(64),
    }],
  }],
  article_blocks: [
    { kind: 'paragraph', segments: [{ kind: 'claim', claim_id: 'claim_7', text: '1.96' }] },
    { kind: 'verification_notes', notes: [{
      code: 'strict_untraceable_numeric_sentence_removed', severity: 'warning',
      text: 'A numeric sentence without a registered evidence source was removed.',
    }] },
  ],
  tables: [{ display_id: 'Table 1', label: 'Table 1', caption: 'Registered table.' }],
  figure_gallery: { figures: [{ display_id: 'Figure 1', label: 'Figure 1', data_url: safePng }] },
});
const staleNumber = readerContext.match(/<button[^>]*data-gpi-claim="claim_7"[^>]*>/)[0];
assert.ok(!staleNumber.includes('data-gpi-evidence-open'), 'a stale chain must open its status and lineage before any evidence preview');
assert.match(readerContext, /data-gpi-display="Table 1"/, 'a registered display id must render a locate control');
assert.match(readerContext, /data-gpi-display-anchor="Table 1"/, 'the display block must carry its anchor');
assert.match(readerContext, /<figure[^>]*data-gpi-display-anchor="Figure 1"/, 'the actual figure must carry a locate target');
assert.match(readerContext, /gpi-verification-notes/, 'typed verification notes must render as reader text');
assert.match(readerContext, /strict_untraceable_numeric_sentence_removed/, 'the note code must remain visible');
assert.match(readerContext, /<span class="pill warn">stale<\/span>/, 'stale links must show their status');
assert.ok(readerContext.includes('Estimate the adjusted association.'), 'method summary must stay readable');
assert.ok(!readerContext.includes('relative_path'), 'reader markup must not leak host paths');

const scrubbedUrls = renderer.scrubDataUrls({
  metadata_url: 'https://example.test/metadata.json',
  Data_URL: 'data:image/png;base64,AAAA',
  image_data_url: 'data:image/png;base64,BBBB',
});
assert.equal(scrubbedUrls.metadata_url, 'https://example.test/metadata.json', 'ordinary *_url metadata must remain visible');
assert.equal(scrubbedUrls.Data_URL, '[embedded image hidden in JSON preview]');
assert.equal(scrubbedUrls.image_data_url, '[embedded image hidden in JSON preview]');

process.stdout.write(JSON.stringify({ ok: true, cases: 23 }));
