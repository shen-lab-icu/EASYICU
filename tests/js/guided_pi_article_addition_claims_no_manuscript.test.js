'use strict';
// S1799: an article addition reports analyses only. Its answer never says a
// manuscript exists or awaits review, and it names the way to a paper.
const assert = require('node:assert/strict');
const path = require('node:path');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh' };
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
require(path.join(root, 'html-escape.js'));
require(path.join(root, 'screens-guided-pi-run-answer.js'));

(async () => {
  const run = { run_id: 'addition', artifact_refs: ['agent_plan.json', 'result_tables.json'].map(artifact => ({
    run_id: 'addition', artifact, sha256: 'c'.repeat(64),
  })) };
  for (const [language, tr] of [['zh', (en, zh) => zh], ['en', en => en]]) {
    const answer = window.EasyICU.guidedPi.require('runAnswer').create({
      tr, esc: window.EU_HTML.esc, projectId: () => 'project', resourceButton: () => '',
      api: () => ({ loadPiCopilotResearchArtifact: async (_project, _run, artifact) => ({ ok: true,
        payload: artifact === 'agent_plan.json' ? { steps: [], analysis_type: 'association_study' } : {
          tables: [], append: { kind: 'article', label_en: 'Extend to a full article', label_zh: '完善为论文级分析' },
        } }) }),
    });
    await answer.load(run);
    const html = answer.render(run);
    if (language === 'zh') {
      assert.match(html, /没有生成稿件、PDF 或参考文献/);
      assert.match(html, /新提一个问题并写明需要论文/);
      assert.doesNotMatch(html, /稿件还没有审阅|论文稿中|论文草稿/);
    } else {
      assert.match(html, /no manuscript, PDF or references were written/);
      assert.match(html, /ask a new question that requests a paper/);
      assert.doesNotMatch(html, /manuscript has not been reviewed|manuscript draft/);
    }
  }
  console.log('article addition: the answer claims no manuscript and names the paper path');
})().catch(error => { console.error(error); process.exitCode = 1; });
