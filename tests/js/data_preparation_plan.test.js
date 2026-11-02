'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const base = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
for (const language of ['en', 'zh']) {
  const context = vm.createContext({ window: { EU_LANG: language, t: (en, zh) => language === 'zh' ? zh : en, icon: () => '' } });
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
  // The saved plan alone owns what it can answer, regardless of interval method.
  for (const method of [
    'Counts and proportions; no confidence interval (typed counts-only design).',
    'Counts and proportions with 95% confidence intervals.',
  ]) {
    const supports = 'Row-level counts and proportions & <img src=x>.';
    const saved = { research_question: 'Synthetic description', design_selection: { candidates: [{
      disposition: 'selected', analysis_type: 'descriptive_epidemiology',
      primary_method: method, supports,
    }] } };
    const rendered = context.window.AGENT_RENDER.artifactStructuredView('agent_plan.json', saved);
    assert.match(rendered, /Row-level counts and proportions &amp; &lt;img src=x&gt;\./);
    assert.doesNotMatch(rendered, /<img src=x>/);
    assert.doesNotMatch(rendered, /方向、大小和不确定性|direction, size, and uncertainty/);
  }
}
console.log('2 preparation plan language checks passed');
