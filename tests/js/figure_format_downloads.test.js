'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const js = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const window = { EU_LANG: 'en', icon: () => '', t: (en, zh) => window.EU_LANG === 'zh' ? zh : en };
const context = vm.createContext({ window });
for (const owner of ['html-escape.js', 'screens-agent-report-reader.js']) {
  vm.runInContext(fs.readFileSync(path.join(js, owner), 'utf8'), context);
}
const locator = { evidence_id: 'synthetic_pdf', sha256: 'a'.repeat(64), kind: 'figure' };
const row = {
  label: 'Synthetic aggregate figure', data_url: 'data:image/png;base64,c3ludGhldGlj',
  export_locators: { pdf: locator, svg: { ...locator, evidence_id: 'synthetic_svg' } },
};
const gallery = figures => window.AGENT_REPORT_READER.figureGallery({ run_ref: 'rr_' + 'b'.repeat(32), figures });
for (const language of ['en', 'zh']) {
  window.EU_LANG = language;
  const html = gallery([row, { ...row, export_locators: { tiff: { ...locator, evidence_id: 'synthetic_tiff' } } }]);
  assert.equal((html.match(/class="ag-figure-download"/g) || []).length, 2, 'one compact control per figure');
  assert.match(html, /PDF · SVG/);
  assert.equal((html.match(/data-gpi-figure-download/g) || []).length, 3);
  assert.match(html, /data-figure-format="pdf"/);
  assert.match(html, /data-figure-format="tiff"/);
  assert.doesNotMatch(html, /data-figure-format="png"/);
  assert.ok(html.includes(language === 'zh' ? '下载图件' : 'Download figure'));
  assert.ok(html.includes(language === 'zh' ? '下载 PDF' : 'Download PDF'));
  const absent = gallery([{ ...row, export_locators: {} }]);
  assert.doesNotMatch(absent, /ag-figure-download|data-gpi-figure-download/);
  const unsafe = gallery([{ ...row, export_locators: {
    pdf: { ...locator, evidence_id: '" onclick="bad' },
    svg: { ...locator, sha256: 'invalid' }, tiff: { ...locator, kind: 'table' },
    html: locator, png: { ...locator, relative_path: '/private/image.png' },
  } }]);
  assert.doesNotMatch(unsafe, /data-gpi-figure-download|onclick=|private\/image/);
}
const button = { dataset: {
  figureRunRef: 'rr_' + 'b'.repeat(32), figureFormat: 'pdf',
  figureEvidenceId: locator.evidence_id, figureSha256: locator.sha256,
} };
const request = window.AGENT_REPORT_READER.figureDownloadRequest(button);
assert.equal(JSON.stringify(request), JSON.stringify({
  run_ref: button.dataset.figureRunRef, figure_export: { format: 'pdf', ...locator },
}));
for (const format of ['pdf', 'svg', 'tiff', 'png']) {
  assert.equal(window.AGENT_REPORT_READER.figureDownloadRequest({ dataset: {
    ...button.dataset, figureFormat: format,
  } }).figure_export.format, format, 'the clicked format must reach the host unchanged');
}
assert.doesNotMatch(window.AGENT_REPORT_READER.figureGallery({ figures: [row] }), /data-gpi-figure-download/);
for (const patch of [{ figureFormat: 'html' }, { figureRunRef: '/host/run' }, { figureSha256: '' }]) {
  assert.equal(window.AGENT_REPORT_READER.figureDownloadRequest({ dataset: { ...button.dataset, ...patch } }), null);
}

(async () => {
  const calls = [];
  const downloads = [];
  const document = {
    querySelector: () => null, getElementById: () => null,
    body: { appendChild() {} },
    createElement: () => ({ click() { downloads.push(this.download); }, remove() {} }),
  };
  const transport = vm.createContext({ window, document, setTimeout: fn => fn(),
    URL: { createObjectURL: () => 'blob:synthetic', revokeObjectURL() {} },
    fetch: async (url, options) => {
      calls.push({ url, body: JSON.parse(options.body) });
      return { ok: true, headers: { get: () => 'attachment; filename="synthetic.pdf"' }, blob: async () => ({}) };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(js, 'api.js'), 'utf8'), transport);
  await window.EU_API.downloadAgentRunArtifact({ run_ref: request.run_ref }, request.figure_export);
  assert.equal(calls[0].url, '/api/agent-runs/download-artifact');
  assert.equal(JSON.stringify(calls[0].body), JSON.stringify(request));
  assert.deepEqual(downloads, ['synthetic.pdf']);
  await window.EU_API.downloadAgentRunArtifact({ run_ref: request.run_ref }, 'result_tables.json');
  assert.deepEqual(calls[1].body, { run_ref: request.run_ref, artifact: 'result_tables.json' });

  const owners = new Map();
  window.EasyICU = { guidedPi: { declare: (name, owner) => owners.set(name, owner), optional: () => null } };
  vm.runInContext(fs.readFileSync(path.join(js, 'screens-guided-pi-preview.js'), 'utf8'), transport);
  let click;
  owners.get('preview').mount({ addEventListener: (_name, fn) => { click = fn; } });
  const event = { target: { closest: selector => selector === '[data-gpi-figure-download]' ? button : null } };
  click(event);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls.length, 3, 'preview click must reach the governed download route');
  assert.equal(JSON.stringify(calls[2].body), JSON.stringify(request));

  vm.runInContext(fs.readFileSync(path.join(js, 'screens-guided-pi-run-files.js'), 'utf8'), transport);
  const controller = owners.get('runFiles').createController(window.EU_API);
  controller.state.review = { run_ref: request.run_ref, artifacts: [] };
  await controller.download(request.figure_export);
  assert.equal(calls.length, 4, 'run files must accept a figure identity without treating it as an artifact name');
  assert.equal(JSON.stringify(calls[3].body), JSON.stringify(request));
  const run = { run_ref: request.run_ref, study_id: 'synthetic', run_id: 'run_synthetic' };
  const api = {
    ...window.EU_API,
    listStudyContexts: async () => ({ contexts: [{ id: run.study_id }] }),
    loadIdeaAgentProjects: async () => ({ projects: [] }),
    loadGuidedDrafts: async () => ({ drafts: [] }),
    loadAgentRunHistory: async () => ({ ok: true, runs: [run] }),
    loadAgentRunReview: async () => ({ ok: true, ...run, artifacts: [] }),
  };
  const view = owners.get('runFiles').create({
    api: () => api, context: () => ({ studyId: run.study_id }), changed() {},
  });
  await view.sync();
  await view.open(request.run_ref);
  const section = { dataset: { runFiles: request.run_ref } };
  assert.equal(view.handleClick({ target: { closest: selector => (
    selector === '[data-run-files]' ? section : selector === '[data-gpi-figure-download]' ? button : null
  ) } }), true);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls.length, 5, 'run-file click must download the selected format');
  assert.equal(JSON.stringify(calls[4].body), JSON.stringify(request));
  console.log('Figure format downloads: rendering, click and transport contracts passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
