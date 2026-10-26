'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh' };
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
require(path.join(root, 'html-escape.js'));
require(path.join(root, 'screens-guided-pi-result-append.js'));
require(path.join(root, 'screens-guided-pi-run-answer.js'));
require(path.join(root, 'screens-guided-pi-run-outcome.js'));
let language = 'zh';
let project = 'project';
let ready = true;
let posts = [];
const kinds = ['sensitivity', 'exposure', 'subgroup', 'article'];
const labels = [
  ['Add a sensitivity analysis', '加敏感性分析'], ['Change the exposure definition', '换暴露定义'],
  ['Add a subgroup', '加亚组'], ['Extend to a full article', '完善为论文级分析'],
];
const options = kinds.map((kind, i) => ({ kind, label_en: labels[i][0], label_zh: labels[i][1],
  available: kind !== 'subgroup', owner: kind, status_en: 'No eligible stratifier.', status_zh: '没有可用分层变量。' }));
const owner = window.EasyICU.guidedPi.require('resultAppend').create({
  tr: (en, zh) => language === 'zh' ? zh : en, esc: window.EU_HTML.esc,
  projectId: () => project, canPreview: () => ready, host: () => null,
  session: () => ({ session_id: 'session' }), onError: value => { throw Error(value); }, errorText: String,
  api: () => ({
    loadPiCopilotAppendOptions: async () => ({ run_id: 'source', source_run_sha256: 'a'.repeat(64), source_changed: true, options }),
    proposePiCopilotAppend: async (session, payload) => { posts.push([session, payload]); return { run_id: 'append', human_review_pending: true }; },
  }),
});
(async () => {
  await owner.load({ run_id: 'source' });
  for (language of ['zh', 'en']) {
    const html = owner.render({ run_id: 'source' });
    for (const label of labels) assert.ok(html.includes(label[language === 'zh' ? 1 : 0]));
    assert.match(html, /data-gpi-result-append="sensitivity"/);
    assert.doesNotMatch(html, /data-gpi-result-append="subgroup"/);
    assert.match(html, /data-gpi-append-owner="subgroup"/);
    assert.match(html, language === 'zh' ? /沿用这次结果的数据版本/ : /same data version as this result/);
  }
  assert.equal(await owner.propose('subgroup'), false);
  assert.equal(posts.length, 0);
  assert.equal(await owner.propose('exposure'), true);
  assert.deepEqual(posts[0], ['session', { project_id: 'project', run_id: 'source', kind: 'exposure', expected_source_sha256: 'a'.repeat(64) }]);
  assert.equal('decision' in posts[0][1], false, 'the result button does not approve');
  ready = false;
  assert.equal(await owner.propose('sensitivity'), false);
  project = 'different';
  assert.doesNotMatch(owner.render({ run_id: 'source' }), /data-gpi-result-append=/);
  const resultRun = { run_id: 'addition', artifact_refs: ['agent_plan.json', 'result_tables.json'].map(artifact => ({
    run_id: 'addition', artifact, sha256: 'b'.repeat(64),
  })) };
  const answer = window.EasyICU.guidedPi.require('runAnswer').create({
    tr: (en, zh) => zh, esc: window.EU_HTML.esc, projectId: () => 'project', resourceButton: () => '',
    api: () => ({ loadPiCopilotResearchArtifact: async (_project, _run, artifact) => ({ ok: true,
      payload: artifact === 'agent_plan.json' ? { steps: [], analysis_type: 'association_study' } : {
        tables: [], append: { kind: 'subgroup', label_en: 'Add a subgroup', label_zh: '加亚组', strata: [
          { subgroup: 'stratum', stratum: 'A', estimate: 1.4, ci_low: 1.1, ci_high: 1.8, interaction_p_value: 0.03 },
          { subgroup: 'stratum', stratum: 'B', estimate: 0.8, ci_low: 0.5, ci_high: 1.2, interaction_p_value: 0.03 },
        ] },
      } }) }),
  });
  await answer.load(resultRun);
  const answerHtml = answer.render(resultRun);
  assert.match(answerHtml, /= A: OR 1\.40/);
  assert.match(answerHtml, /= B: OR 0\.800/);
  assert.match(answerHtml, /交互检验 P/);
  assert.match(answerHtml, /沿用之前的数据版本/);
  let offered = 0;
  const outcome = window.EasyICU.guidedPi.require('runOutcome').create({
    tr: en => en, esc: window.EU_HTML.esc, projectId: () => 'project', host: () => null,
    api: () => ({ loadPiCopilotAppendOptions: async () => {
      offered += 1;
      return { run_id: 'source', options };
    } }),
  });
  const result = { run_id: 'source', present: true, analysis_results_available: true, artifact_refs: [] };
  await outcome.loadScientificReview(result, { stages: [{ id: 'analysis', status: 'running' }] });
  assert.equal(offered, 0);
  const finished = { stages: [{ id: 'analysis', status: 'complete' }] };
  await outcome.loadScientificReview(result, finished);
  await outcome.loadScientificReview(result, finished);
  assert.equal(offered, 1, 'the existing result-loading lifecycle discovers executable additions once');
  console.log('result append: executable buttons, owner entry, earlier export, scoped proposal and one approval passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
