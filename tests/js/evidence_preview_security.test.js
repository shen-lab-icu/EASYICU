'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const escape = value => String(value == null ? '' : value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const context = {
  window: { EU_LANG: 'en', EU_HTML: { esc: escape } },
  globalThis: { pwned: false },
};
vm.createContext(context);
vm.runInContext(source, context);
const renderer = context.window.EU_GUIDED_PI_EVIDENCE_PREVIEW;

const code = renderer.render({
  renderer: 'code', previewable: true, kind: 'code', evidence_id: 'code_1',
  sha256: 'a'.repeat(64), text: '<img src=x onerror="globalThis.pwned=true">\nestimate = 1.2',
}, { pointer: '/estimate', value: '<script>bad()</script>' });
assert.ok(code.includes('&lt;img src=x onerror=&quot;globalThis.pwned=true&quot;&gt;'));
assert.ok(code.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!code.includes('<img src=x'));
assert.equal(context.globalThis.pwned, false, 'preview code must never execute');

const table = renderer.render({
  renderer: 'table', previewable: true, kind: 'table', evidence_id: 'table_1',
  sha256: 'b'.repeat(64), headers: ['estimate'], rows: [['<svg onload=bad()>']],
}, {});
assert.ok(table.includes('&lt;svg onload=bad()&gt;'));
assert.ok(!table.includes('<svg onload=bad()>'));

const withheld = renderer.render({
  renderer: 'metadata', previewable: false, kind: 'table', evidence_id: 'cohort_1',
  sha256: 'c'.repeat(64), withheld_reason: 'patient_level_rows_withheld', bytes: 1024,
}, {});
assert.ok(withheld.includes('Patient-level cohort rows are withheld'));
assert.ok(!withheld.includes('<table'));

process.stdout.write(JSON.stringify({ ok: true, cases: 3 }));
