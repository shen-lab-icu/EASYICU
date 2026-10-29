'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const base = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
for (const language of ['en', 'zh']) {
  const context = vm.createContext({ window: { t: (en, zh) => language === 'zh' ? zh : en, icon: () => '' } });
  for (const name of ['html-escape.js', 'screens-agent-reader-vocab.js', 'screens-agent-report-reader.js', 'screens-agent-render.js']) {
    vm.runInContext(fs.readFileSync(path.join(base, name), 'utf8'), context);
  }
  const choice = { missing_data: { strategy: 'multiple_imputation', m: 7, seed: 123,
    variables: [{ name: '<img src=x>', method: 'pmm' }], auxiliaries: ['auxiliary'] },
    outliers: [], windows: [], scaling_coding: [], unit_harmonisation: true };
  const plan = { research_question: 'Synthetic association question', steps: [{ step_id: 'fit',
    intent: 'Fit the prespecified synthetic association', planned_analysis_role: 'primary',
    method: 'adjusted_association_models with easyicu.data_preparation/1:' + JSON.stringify(choice),
    inputs: [], expected_outputs: ['table:adjusted_association_estimates'] }] };
  const html = context.window.AGENT_RENDER.artifactStructuredView('agent_plan.json', plan);
  assert.match(html, language === 'zh' ? /数据预处理/ : /Data preparation/);
  assert.match(html, language === 'zh' ? /主分析：多重插补；敏感性分析：完整病例/ : /Primary: Multiple imputation; sensitivity: Complete case/);
  assert.match(html, /m=7/);
  assert.match(html, /123/);
  assert.match(html, /auxiliary/);
  assert.doesNotMatch(html, /<img src=x>/);
}
console.log('2 preparation plan language checks passed');
