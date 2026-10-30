'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const ownerSource = fs.readFileSync(path.join(root, 'src/easyicu/webserver/static/js/screens-guided-pi-error-text.js'), 'utf8');
const compiler = fs.readFileSync(path.join(root, 'src/easyicu/research_agent/planning/progressive_compiler.py'), 'utf8');
// Read source with Node only: frontend CI has no project Python dependencies.
const codes = new Set([...compiler.matchAll(/["'](progressive_[a-z_]+)["']/g)]
  .map(match => match[1]).filter(code => !compiler.includes(`def ${code}(`)));
const features = fs.readFileSync(path.join(root, 'src/easyicu/research_agent/contracts/phenotyping_features.py'), 'utf8');
for (const match of (compiler + features).matchAll(/(?:ValueError\(|reason = )f?["']((?:phenotyping_feature_|phenotype_comparison_)[a-z_]+)(?=:|["'])/g)) {
  codes.add(`progressive_${match[1]}`);
}
const contrasts = fs.readFileSync(path.join(root, 'src/easyicu/research_agent/planning/exposure_group_compile.py'), 'utf8');
for (const match of contrasts.matchAll(/["'](progressive_[a-z_]+)["']/g)) codes.add(match[1]);
const contract = fs.readFileSync(path.join(root, 'src/easyicu/research_agent/planning/progressive_contract.py'), 'utf8');
for (const match of contract.matchAll(/\{prefix\}_([a-z_]+)/g)) {
  for (const prefix of ['progressive', 'progressive_outline']) codes.add(`${prefix}_${match[1]}`);
}
codes.add('progressive_outline_module_unavailable');
assert.ok(codes.size > 80);
// These checks do not yet have a more specific user-facing explanation.
// Keep this explicit so a mapped message cannot silently regress to a code.
const knownUnmapped = new Set([
  'progressive_stated_adult_population_missing',
  'progressive_adult_age_source_unverified',
  'progressive_diagnostic_identity_not_bound',
]);
for (const lang of ['en', 'zh']) {
  let api;
  const context = { window: { EU_LANG: lang, EU_HTML: { esc: String }, EasyICU: { guidedPi: {
    declare: (_name, value) => { api = value; },
  } } } };
  vm.runInNewContext(ownerSource, context);
  const owner = api.create({ tr: (en, zh) => lang === 'zh' ? zh : en, staticPreview: () => false });
  for (const code of codes) {
    const sentence = owner.runFailureText('research_pipeline_progressive_compile_failed', { code });
    // Unmapped checks deliberately retain their diagnostic code; they do not
    // assume that regenerating a plan will resolve the underlying stop.
    if (knownUnmapped.has(code)) {
      assert.equal(sentence, lang === 'zh'
        ? `候选计划没有通过 EasyICU 的检查（代码：${code}），规划停止，没有运行分析。`
        : `Planning stopped at an EasyICU check of the candidate plan (code: ${code}); no analysis was run.`);
      continue;
    }
    assert.ok(!sentence.includes('progressive_'), `${lang}: ${code}: ${sentence}`);
    assert.ok(sentence.includes(lang === 'zh' ? '规划已停止，没有运行分析' : 'no analysis was run'), `${lang}: ${code}: ${sentence}`);
    assert.ok((lang === 'zh' ? sentence.includes('请') : /[Gg]enerate/.test(sentence)), `${lang}: ${code}: ${sentence}`);
  }
  for (const code of ['progressive_unmapped_test_stop', 'progressive_family_spec_unmapped_test_stop']) {
    const sentence = owner.runFailureText('research_pipeline_progressive_compile_failed', { code });
    assert.ok(sentence.includes(code), `diagnostic code is retained: ${sentence}`);
    assert.ok(sentence.includes(lang === 'zh' ? '没有运行分析' : 'no analysis was run'));
    assert.doesNotMatch(sentence, /重新生成|[Gg]enerate the plan again/);
  }
  assert.equal(owner.runFailureText('research_pipeline_progressive_compile_failed', { code: 'progressive_analysis_module_unavailable' }),
    lang === 'zh' ? '候选计划中有一个分析步骤不适用于这类研究，因此规划已停止，没有运行分析。请重新生成计划。'
      : 'The candidate plan used an analysis step that EasyICU cannot run for this kind of study, so planning stopped and no analysis was run. Generate the plan again.');
}
console.log(`PASS: ${codes.size} compiler stops in both languages`);
