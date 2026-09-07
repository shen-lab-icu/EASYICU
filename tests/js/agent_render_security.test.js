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
    { kind: 'paragraph', segments: [{ kind: 'text', text: 'See Table 1. Sources [@first; @second].' }] },
    { kind: 'heading', level: 2, segments: [{ kind: 'text', text: 'Discussion' }] },
  ],
  tables: [{ label: 'Table 1', caption: 'Baseline', columns: ['Variable', 'N'], rows: [['<svg onload=bad()>', '120']], notes: ['Unknown is not zero.'] }],
  references: [
    { key: 'first', number: 1, title: '<script>bad()</script>', authors: ['A Author'], year: '2020', url: 'javascript:bad()', bibliographic_notices: ['Correction: <img src=x onerror=bad()>'] },
    { key: 'second', number: 2, title: 'Second source', authors: [], year: '2021', doi: '10.1234/test' },
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
assert.ok(assembled.includes('Correction: &lt;img'));
assert.ok(assembled.includes('revision-1'));
assert.ok(!assembled.includes('href="javascript:'));
assert.ok(!assembled.includes('<script>') && !assembled.includes('<svg onload'));
const withoutDiscussion = renderer.manuscriptProvenanceView({ ...assembledPayload, article_blocks: [] });
assert.equal((withoutDiscussion.match(/Bound caption\./g) || []).length, 1, 'figures are not lost when a heading is missing');

process.stdout.write(JSON.stringify({ ok: true, cases: 15 }));
