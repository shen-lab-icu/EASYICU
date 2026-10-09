/* A run's only report leads its file list instead of folding away as history,
   and the host's file guide owns what every file is called and is for: only
   the guide may call the first report the manuscript. With a revision the
   first report is the historical original. The real resource owner renders
   the buttons, since it re-titles previewable files itself. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
global.EU_LANG = 'zh';
global.t = (en, zh) => zh;
global.icon = () => '';
global.document = { addEventListener() {} };
for (const file of process.argv.slice(2)) require(path.resolve(file));
const modules = global.EasyICU.guidedPi;
const runFiles = modules.require('runFiles');
const resources = modules.require('resources').create({ esc: global.EU_HTML.esc });

const DIR = '/runs/a/run_flagship';
const run = { run_id: 'run_flagship', study_id: 'a', project_dir: DIR };
function listFor(names, readiness, fileGuide) {
  const api = {
    listStudyContexts: async () => ({ contexts: [{ id: 'a', title: 'A' }] }),
    loadIdeaAgentProjects: async () => ({ projects: [] }),
    loadGuidedDrafts: async () => ({ drafts: [] }),
    loadAgentRunHistory: async () => ({ ok: true, runs: [run], count: 1 }),
    loadAgentRunReview: async () => ({ ok: true, ...run, engine: 'easyicu.research_agent.pipeline', signed: false, readiness,
      artifacts: names.map(name => ({ name, sha256: 'a'.repeat(64) })), ...(fileGuide ? { file_guide: fileGuide } : {}) }),
  };
  const view = runFiles.create({
    api: () => api, changed() {}, resourceButton: resources.button,
    context: () => ({ projectId: 'draft_a', studyId: 'a', sessionId: 's', runId: '', busy: false }),
  });
  return (async () => {
    await view.sync();
    await view.open(DIR);
    const html = view.render({ savedRuns: [run] });
    const [primary, extra = ''] = html.split('<details class="gpi-run-extra"');
    return { primary, extra };
  })();
}
// The button a file renders as: its visible text and the title the reader opens with.
function button(html, artifact) {
  const at = html.indexOf(`data-gpi-resource-artifact="${artifact}"`);
  if (at < 0) return null;
  const rest = html.slice(at);
  return { reader: rest.match(/data-gpi-resource-label="([^"]*)"/)[1], text: rest.match(/>([^<]*)<\/button>/)[1] };
}
const executed = { signable: true, non_human_failures: [] };
const names = ['manuscript_provenance.json', 'manuscript_scaffold.pdf', 'result_tables.json', 'figure_gallery.json', 'evidence_ledger.json'];

(async () => {
  // No revision and no guide: the report leads under a name that claims
  // nothing, even when no gate failure is listed (a missing check is not a pass).
  let list = await listFor(names, executed);
  assert.deepEqual(button(list.primary, 'manuscript_scaffold.pdf'), { reader: '运行报告（PDF）', text: '运行报告（PDF）' });
  assert.match(list.primary, /本次运行生成的报告，可下载。/);
  assert.equal(button(list.extra, 'manuscript_scaffold.pdf'), null);
  assert.doesNotMatch(list.primary + list.extra, /历史 PDF|稿件 PDF/);

  // The host's guide names every file it covers, the first report included,
  // on the button and in the reader alike.
  const guide = [
    { name: 'manuscript_scaffold.pdf', folder: 'manuscript', title: { en: 'Report draft (PDF)', zh: '报告草稿（PDF）' }, purpose: { en: 'A draft.', zh: '本次运行生成的报告草稿；稿件审阅尚未通过。' } },
    { name: 'result_tables.json', folder: 'results', title: { en: 'Result tables', zh: '结果表（含 CSV）' }, purpose: { en: 'Tables.', zh: '每张表另附 CSV。' } },
  ];
  list = await listFor(names, { signable: false, non_human_failures: ['manuscript_ready'] }, guide);
  assert.deepEqual(button(list.primary, 'manuscript_scaffold.pdf'), { reader: '报告草稿（PDF）', text: '报告草稿（PDF）' });
  assert.match(list.primary, /本次运行生成的报告草稿；稿件审阅尚未通过。/);
  assert.deepEqual(button(list.primary, 'result_tables.json'), { reader: '结果表（含 CSV）', text: '结果表（含 CSV）' });
  assert.match(list.primary, /每张表另附 CSV。/);
  // A file the guide leaves out keeps the shared vocabulary.
  assert.deepEqual(button(list.primary, 'figure_gallery.json'), { reader: '图件画廊', text: '图件画廊' });

  // With a revision the revision leads and the first report is history.
  list = await listFor(names.concat('manuscript_revision.pdf'), executed);
  assert.equal(button(list.primary, 'manuscript_revision.pdf').text, '当前报告修订（PDF）');
  assert.equal(button(list.primary, 'manuscript_scaffold.pdf'), null);
  assert.equal(button(list.extra, 'manuscript_scaffold.pdf').text, '原运行报告（历史 PDF）');
  process.stdout.write('A run\'s only report leads its files under the guide\'s name.\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
