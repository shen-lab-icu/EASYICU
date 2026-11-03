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
// Only require_phenotyping_features errors are explicitly prefixed by the
// compiler. The separate AnalysisStep validator is wrapped in the generic
// progressive_analysis_step_invalid stop, not its internal ValueError text.
const validatorStart = features.indexOf('\ndef validate_phenotyping_step(');
assert.ok(validatorStart > 0, 'phenotyping step validator boundary is present');
const directFeatureChecks = features.slice(0, validatorStart);
for (const match of (compiler + directFeatureChecks).matchAll(/(?:ValueError\(|reason = )f?["']((?:phenotyping_feature_|phenotype_comparison_)[a-z_]+)(?=:|["'])/g)) {
  codes.add(`progressive_${match[1]}`);
}
assert.ok(compiler.includes('"progressive_analysis_step_invalid"'), 'compiled step validation has a generic stop');
assert.ok(features.slice(validatorStart).includes('"phenotype_comparison_input_mismatch"'));
assert.ok(codes.has('progressive_analysis_step_invalid'));
assert.ok(!codes.has('progressive_phenotype_comparison_input_mismatch'));
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
    // Unmapped checks point to the failure receipt, which keeps the code;
    // they do not assume that regenerating a plan resolves the stop.  A raw
    // code in the sentence read as a malfunction to the researcher (11-03).
    if (knownUnmapped.has(code)) {
      assert.equal(sentence, lang === 'zh'
        ? '候选计划没有通过 EasyICU 的一项检查，规划停止，没有运行分析。失败回执里写明了是哪一项检查。'
        : 'Planning stopped at an EasyICU check of the candidate plan; no analysis was run. The failure receipt names the check.');
      continue;
    }
    assert.ok(!sentence.includes('progressive_'), `${lang}: ${code}: ${sentence}`);
    assert.ok(sentence.includes(lang === 'zh' ? '规划已停止，没有运行分析' : 'no analysis was run'), `${lang}: ${code}: ${sentence}`);
    assert.ok((lang === 'zh' ? sentence.includes('请') : /[Gg]enerate/.test(sentence)), `${lang}: ${code}: ${sentence}`);
  }
  for (const code of ['progressive_unmapped_test_stop', 'progressive_family_spec_unmapped_test_stop']) {
    const sentence = owner.runFailureText('research_pipeline_progressive_compile_failed', { code });
    assert.ok(!sentence.includes(code), `no raw code in the sentence: ${sentence}`);
    assert.ok(sentence.includes(lang === 'zh' ? '失败回执' : 'failure receipt'), sentence);
    assert.ok(sentence.includes(lang === 'zh' ? '没有运行分析' : 'no analysis was run'));
    assert.doesNotMatch(sentence, /重新生成|[Gg]enerate the plan again/);
  }
  for (const code of ['progressive_article_required_roles_missing', 'progressive_outline_scientific_citation_missing']) {
    const sentence = owner.runFailureText('research_pipeline_progressive_compile_failed', { code });
    assert.ok(!sentence.includes('progressive_') && !sentence.includes('失败回执') && !sentence.includes('failure receipt'), sentence);
    assert.ok(sentence.includes(lang === 'zh' ? '请重新生成计划' : 'Generate the plan again'), sentence);
  }
  assert.equal(owner.runFailureText('research_pipeline_progressive_compile_failed', { code: 'progressive_analysis_module_unavailable' }),
    lang === 'zh' ? '候选计划中有一个分析步骤不适用于这类研究，因此规划已停止，没有运行分析。请重新生成计划。'
      : 'The candidate plan used an analysis step that EasyICU cannot run for this kind of study, so planning stopped and no analysis was run. Generate the plan again.');
}
console.log(`PASS: ${codes.size} compiler stops in both languages`);
