'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const js = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const read = name => fs.readFileSync(path.join(js, name), 'utf8');
const digest = 'a'.repeat(64);

function fixture() {
  const reads = [], downloads = [], forbidden = [];
  const host = {
    hidden: false, innerHTML: '', addEventListener() {},
    querySelector() { return null; }, querySelectorAll() { return []; },
    replaceChildren() { this.innerHTML = ''; },
  };
  const window = { EU_LANG: 'en', EU_API: {
    async loadPiCopilotWorkspaceFile(projectId, file) {
      reads.push([projectId, file]);
      return { artifact: { file, text: 'A bounded note.', sha256: 'b'.repeat(64) } };
    },
    piCopilotWorkspaceDownloadUrl(projectId, file, sha256) {
      downloads.push([projectId, file, sha256]);
      return '/workspace/download?file=' + encodeURIComponent(file) + '&expected_sha256=' + sha256;
    },
    piCopilotWorkspacePreviewUrl() { forbidden.push('preview'); throw new Error('CSV must not become an executable preview'); },
    beginPiCopilotWorkspaceRuntimeQa() { forbidden.push('qa'); throw new Error('CSV must not become checked HTML'); },
  } };
  const context = vm.createContext({ window, document: { getElementById() { return null; } } });
  for (const file of ['html-escape.js', 'screens-guided-pi-modules.js', 'screens-guided-pi-resources.js', 'screens-guided-pi-preview.js']) {
    vm.runInContext(read(file), context, { filename: file });
  }
  const preview = window.EasyICU.guidedPi.require('preview');
  preview.mount(host);
  return { preview, host, reads, downloads, forbidden };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('a delivered CSV card keeps its Host SHA through the real preview owner and exposes an exact attachment link without reading rows', async () => {
  const { preview, host, reads, downloads, forbidden } = fixture();
  assert.equal(preview.open({ kind: 'file', file: 'synthetic_icu_glucose_brief.csv', media_type: 'text/csv', sha256: digest }, 'draft-synthetic'), true);
  await settle();
  assert.match(host.innerHTML, /data-gpi-csv-attachment/);
  assert.match(host.innerHTML, /\/workspace\/download\?file=synthetic_icu_glucose_brief\.csv&amp;expected_sha256=a{64}/);
  assert.match(host.innerHTML, /download="synthetic_icu_glucose_brief\.csv"/);
  assert.ok(downloads.length > 0);
  assert.ok(downloads.every(([project, file, sha]) => project === 'draft-synthetic' && file === 'synthetic_icu_glucose_brief.csv' && sha === digest));
  assert.deepEqual(reads, []);
  assert.deepEqual(forbidden, []);
  assert.doesNotMatch(host.innerHTML, /Preview unavailable|Model workspace reads|<iframe|data-gpi-preview-downloads/);
  assert.match(host.innerHTML, /Workspace artifact · Unvalidated/);
  assert.doesNotMatch(host.innerHTML, /Static check passed|Browser checks passed|scientific validation passed/);
});

test('CSV attachment recognition uses its bounded file suffix and cannot be turned into an executable HTML preview by media type', async () => {
  const { preview, host, reads, forbidden } = fixture();
  preview.open({ kind: 'file', file: 'nested/synthetic.CSV', media_type: 'text/html', sha256: digest, checked_sha256: 'c'.repeat(64) }, 'draft-synthetic');
  await settle();
  assert.match(host.innerHTML, /data-gpi-csv-attachment/);
  assert.match(host.innerHTML, /download="synthetic\.CSV"/);
  assert.deepEqual(reads, []); assert.deepEqual(forbidden, []);
  assert.doesNotMatch(host.innerHTML, /<iframe|data-gpi-preview-downloads/);
});

test('CSV cards without a valid delivered SHA remain metadata only and cannot mint a download identity from their label', async () => {
  for (const sha256 of [undefined, '', 'not-a-digest', 'a'.repeat(63)]) {
    const { preview, host, reads, downloads } = fixture();
    preview.open({ kind: 'file', file: 'synthetic.csv', label: digest, sha256 }, 'draft-synthetic');
    await settle();
    assert.match(host.innerHTML, /data-gpi-csv-attachment/);
    assert.deepEqual(reads, []); assert.deepEqual(downloads, []);
    assert.doesNotMatch(host.innerHTML, /<a[^>]*download=/);
  }
});

test('a forged checked SHA alone does not replace the CSV artifact SHA used for its download', async () => {
  const { preview, host, reads, downloads } = fixture();
  preview.open({ kind: 'file', file: 'synthetic.csv', checked_sha256: 'c'.repeat(64) }, 'draft-synthetic');
  await settle();
  assert.match(host.innerHTML, /data-gpi-csv-attachment/);
  assert.deepEqual(reads, []);
  assert.deepEqual(downloads, []);
});

test('unsafe CSV card paths are rejected before rendering or asking for a download URL', async () => {
  for (const file of ['../synthetic.csv', '/tmp/synthetic.csv', 'nested/../../synthetic.csv', 'nested//synthetic.csv', 'nested/./synthetic.csv', 'bad\0.csv']) {
    const { preview, host, reads, downloads } = fixture();
    assert.equal(preview.open({ kind: 'file', file, sha256: digest }, 'draft-synthetic'), undefined);
    await settle();
    assert.deepEqual(reads, []); assert.deepEqual(downloads, []);
    assert.doesNotMatch(host.innerHTML, /data-gpi-csv-attachment|<a[^>]*download=/);
  }
});

test('ordinary text cards keep their existing bounded reader and download the loaded snapshot', async () => {
  const { preview, host, reads, downloads, forbidden } = fixture();
  preview.open({ kind: 'file', file: 'note.md', sha256: digest }, 'draft-synthetic');
  await settle();
  assert.deepEqual(reads, [['draft-synthetic', 'note.md']]);
  assert.ok(downloads.some(([, file, sha]) => file === 'note.md' && sha === 'b'.repeat(64)));
  assert.match(host.innerHTML, /A bounded note\./);
  assert.doesNotMatch(host.innerHTML, /data-gpi-csv-attachment/);
  assert.deepEqual(forbidden, []);
});
