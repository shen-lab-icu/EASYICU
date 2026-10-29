/* Implementation metadata opens exact source without running it. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const context = { EU_LANG: 'zh', t: (_en, zh) => zh,
  EU_HTML: { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') } };
context.window = context;
vm.createContext(context);
const ownerFile = path.resolve(process.argv[2] || 'src/easyicu/webserver/static/js/screens-skills-implementation.js');
vm.runInContext(fs.readFileSync(ownerFile, 'utf8'), context);
const owner = context.EU_SKILL_IMPLEMENTATION;
const digest = 'a'.repeat(64);
const sourcePath = 'src/easyicu/research_agent/methods/analysis.py';
const source = { path: 'implementation/research_agent.methods.analysis.py', role: 'host_implementation',
  source_path: sourcePath, source_sha256: digest, language: 'python',
  content: `# ${sourcePath}:10-11 sha256=${digest}\ndef helper():\n    return None\n\n\n# ${sourcePath}:120-122 sha256=${digest}\ndef estimate(data):\n    """Return an estimate."""\n    return data["<img onerror='bad()'>"]` };
const entry = { name: 'estimate', qualname: 'estimate', signature: 'def estimate(data: DataFrame) -> Result',
  role: 'kernel', file_path: source.path, source_path: sourcePath, source_sha256: digest,
  line_start: 120, line_end: 122, excerpt_line_start: 7, excerpt_line_end: 9,
  docstring: 'Input: a typed cohort. Output: estimate and uncertainty.' };
const pkg = { overview_paths: { en: 'SKILL.md', zh: 'SKILL.zh.md' }, files: [
  { path: 'SKILL.md', content: '# Estimate\n\n## Inputs\nEnglish input.\n\n## Outputs\nEnglish output.' },
  { path: 'SKILL.zh.md', content: '# 估计\n\n## 用途\n估计风险。\n\n## 计划将提出哪些内容供您审阅\nLONG_PLAN_TEXT\n输入：每位患者一行的队列与分组。\n\n## 输出\n- `estimate.csv`\n\n## 使用边界\nLONG_CONTRACT_TEXT' }, source,
], implementation: { mode: 'fixed_host', summary_zh: '使用登记的统计实现。', entrypoints: [
  { ...entry, name: 'helper', qualname: 'helper', signature: 'def helper()', role: 'helper', line_start: 10, line_end: 11,
    excerpt_line_start: 2, excerpt_line_end: 3 }, entry,
], examples: [] } };
assert.equal(owner.defaultPath(pkg), 'SKILL.zh.md');
context.EU_LANG = 'en';
assert.equal(owner.defaultPath(pkg), 'SKILL.md');
context.EU_LANG = 'zh';
assert.equal(owner.defaultPath({ files: [{ path: 'references/a.md' }, { path: 'SKILL.md' }] }), 'SKILL.md');

const overview = owner.overview(pkg, { description: '按分组估计风险。', inputs: ['English fallback'], outputs: ['fallback.csv'] }, text => `<article>${context.EU_HTML.esc(text)}</article>`);
assert.match(overview, /按分组估计风险/);
assert.match(overview, /每位患者一行的队列与分组/);
assert.match(overview, /estimate\.csv/);
assert.doesNotMatch(overview, /English fallback|LONG_PLAN_TEXT|LONG_CONTRACT_TEXT/);
const preview = owner.preview(pkg);
assert.ok(preview.indexOf('data-sk-function="1"') < preview.indexOf('data-sk-function="0"'), 'kernel must precede helper');
assert.match(preview, /def estimate\(data: DataFrame\) -&gt; Result|def estimate\(data: DataFrame\) -> Result/);

const rendered = owner.render(pkg, 1);
const code = rendered.slice(rendered.indexOf('<pre class="eusk-source-code"'));
assert.match(code, /data-source-line="120"/);
assert.match(code, /data-source-line="122"/);
assert.doesNotMatch(code, /data-source-line="10"|def helper/);
assert.match(code, /eusk-syntax-keyword/);
assert.match(code, /&lt;img/);
assert.doesNotMatch(rendered, /<img|<script|data-sk-run|data-sk-use|调用示例/);
assert.match(rendered, /打开源文件/);
assert.match(rendered, /analysis\.py:120–122/);
assert.match(rendered, new RegExp(`sha256:${digest}`));
assert.match(rendered, /Input: a typed cohort. Output: estimate and uncertainty./);

// Marker fallback uses original source lines; it never treats excerpt line 1 as source line 1.
const markerPkg = structuredClone(pkg);
delete markerPkg.implementation.entrypoints[1].excerpt_line_start;
delete markerPkg.implementation.entrypoints[1].excerpt_line_end;
assert.match(owner.render(markerPkg, 1), /data-source-line="120"/);
const stale = structuredClone(pkg);
stale.implementation.entrypoints[1].source_sha256 = 'b'.repeat(64);
assert.match(owner.render(stale, 1), /未能定位此函数的源码片段/);
assert.doesNotMatch(owner.render(stale, 1), /data-source-line="120"/);
const missing = structuredClone(pkg);
missing.implementation.entrypoints[1].file_path = 'not-in-package.py';
assert.match(owner.render(missing, 1), /未能定位此函数的源码片段/);
assert.doesNotMatch(owner.render(missing, 1), /打开源文件/);

const fullFile = owner.filePreview(source);
assert.match(fullFile, /data-source-line="10"/);
assert.match(fullFile, /data-source-line="120"/);
assert.doesNotMatch(fullFile, /data-source-line="1"/);
assert.match(owner.filePreview({ language: 'r', path: 'scripts/main.R', content: 'x <- function() TRUE' }), /data-source-line="1"/);

for (const [mode, label] of [['fixed_host', '固定内置实现'], ['generated_code', '研究代码按任务生成'],
  ['owner_projection', '数据或定义工具'], ['unavailable', '暂未登记实现']]) {
  const data = { files: [], implementation: { mode, entrypoints: [] } };
  assert.match(owner.render(data), new RegExp(label));
}
const generated = { ...pkg, implementation: { mode: 'generated_code', summary_zh: '<script>bad()</script>',
  entrypoints: [{ ...entry, role: 'helper' }], examples: [] } };
assert.match(owner.preview(generated), /可复用辅助函数/);
assert.match(owner.render(generated), /本次研究的分析代码会随任务生成/);
assert.doesNotMatch(owner.preview(generated), /固定内置实现/);
const generatedWithGate = { ...pkg, implementation: { mode: 'generated_code', entrypoints: [
  { ...entry, name: 'initial_code_origin', qualname: 'initial_code_origin', role: 'gate' },
  { ...entry, name: 'run_time_varying_association', qualname: 'run_time_varying_association', role: 'helper' },
] } };
const generatedPreview = owner.preview(generatedWithGate);
assert.ok(generatedPreview.indexOf('data-sk-function="1"') < generatedPreview.indexOf('data-sk-function="0"'), 'generated-code helper must precede internal gate');
const generatedDefault = owner.render(generatedWithGate);
assert.match(generatedDefault, /<h2>run_time_varying_association<\/h2>/);
assert.doesNotMatch(generatedDefault, /<h2>initial_code_origin<\/h2>/);
assert.match(owner.render(generated), /&lt;script>/);
assert.doesNotMatch(owner.render(generated), /<script>/);
assert.match(owner.render(null), /暂未提供函数元数据/);
assert.doesNotMatch(owner.render({ implementation: 'deterministic' }), /data-sk-function/);
const example = { files: [], implementation: { mode: 'fixed_host', examples: [{ language: 'python', label_zh: '实际示例', code: 'print("example")' }] } };
assert.match(owner.render(example), /调用示例（仅展示）/);
assert.doesNotMatch(owner.render(example), /<button|data-sk-run/);
for (const [id, required, output] of [
  ['cohort-characterization-table-one', '分组列及各组取值', '实际分母'],
  ['descriptive.table_one', '待汇总的变量', '缺失人数'],
  ['time_to_event.rmst', '共同的随访截止时间', '平均无事件时间'],
  ['restricted-mean-survival-time', '事件是否发生', '两组差值'],
  ['survival-time-to-event', '随访起点', '比例风险假设检查'],
  ['time_to_event.cox_hr', '随访起点', '比例风险假设检查'],
  ['cross_database.meta_analysis', '无需患者级数据', '每次排除一个数据库'],
]) {
  const text = owner.overview({ ...pkg, skill_id: id }, { description: '用途' }, value => context.EU_HTML.esc(value));
  assert.match(text, new RegExp(required));
  assert.match(text, new RegExp(output));
  assert.doesNotMatch(text, /digest-bound|TableOneSpec|所属方法要求的数据|group rmst\/ci/);
}
for (const [family, key, input, output] of [
  ['data_preparation', 'window_aggregation', '原始时间', '测量数量'],
  ['descriptive', 'descriptive_summary', '研究队列的数据表', '分布汇总表'],
  ['association', 'mixed_effects', '多次记录的长表', '方差与诊断'],
  ['time_to_event', 'ph_check', '事件或删失状态', 'Schoenfeld'],
  ['prediction', 'dynamic_prediction', '预测时点与回看窗口', '风险预测表现'],
  ['causal_emulation', 'mediation', '中介变量', '直接、间接和总效应'],
  ['phenotyping', 'early_subtype_assignment', '轨迹观察窗之前', '亚型概率'],
  ['cross_database', 'meta_analysis', '汇总效应和标准误', '合并效应'],
]) {
  const id = `${family}.${key}`;
  const text = owner.overview({ ...pkg, skill_id: id }, {
    methodFamily: family, methodKey: key, description: '本方法的用途', outputs: ['existing_result.csv + digest-bound receipt'],
  }, value => context.EU_HTML.esc(value));
  assert.match(text, new RegExp(input));
  assert.match(text, new RegExp(output));
  assert.doesNotMatch(text, /所属方法要求的数据|digest-bound|TableOneSpec|LONG_PLAN_TEXT/);
  if (!['cross_database', 'time_to_event'].includes(family)) assert.match(text, /existing_result\.csv/);
  assert.doesNotMatch(text, /invented_result\.csv/);
}
const unknownMethod = owner.overview({ ...pkg, skill_id: 'association.new_unrecorded_method' }, {
  methodFamily: 'association', methodKey: 'new_unrecorded_method', description: '已登记用途', outputs: ['real_output.json'],
}, value => context.EU_HTML.esc(value));
assert.match(unknownMethod, /尚缺中文阅读说明/);
assert.match(unknownMethod, /real_output\.json/);
assert.doesNotMatch(unknownMethod, /digest-bound|TableOneSpec/);
const predictionInput = owner.overview({ ...pkg, skill_id: 'prediction.discrimination_calibration' }, {
  methodFamily: 'prediction', methodKey: 'discrimination_calibration', description: '训练或评价模型', outputs: [],
}, value => value);
assert.match(predictionInput, /拟合新模型时需要开发与验证数据的特征、结局、按研究对象划分的数据分区及预处理方式/);
assert.match(predictionInput, /评价已有预测时需要逐病例配对的真实结局与预测概率/);
const robustnessInput = owner.overview({ ...pkg, skill_id: 'association.robustness_panel' }, {
  methodFamily: 'association', methodKey: 'robustness_panel', description: '稳健性分析', outputs: [],
}, value => value);
assert.match(robustnessInput, /汇总既有变体时需要/);
assert.match(robustnessInput, /按已确定规格重放主模型时，还需原分析数据、主模型及相应规格/);
const evalueInput = owner.overview({ ...pkg, skill_id: 'association.evalue' }, {
  methodFamily: 'association', methodKey: 'evalue', description: '未测混杂敏感性', outputs: ['e_values.csv'],
}, value => value);
assert.match(evalueInput, /使用 OR 时还需该队列实际的结局发生率/);
assert.match(evalueInput, /置信区间中最接近无效值的端点对应值/);
assert.doesNotMatch(evalueInput, /区间下界/);
const phInput = owner.overview({ ...pkg, skill_id: 'time_to_event.ph_check' }, {
  methodFamily: 'time_to_event', methodKey: 'ph_check', description: '比例风险假设检查',
  outputs: ['schoenfeld_test.csv (covariate, chi2, p) + schoenfeld_plot (diagnostics panel slot)'],
}, value => value);
assert.match(phInput, /Schoenfeld 检验结果与相关诊断/);
assert.doesNotMatch(phInput, /schoenfeld_test\.csv|schoenfeld_plot|已登记文件/);
const missingnessInput = owner.overview({ ...pkg, skill_id: 'association.missingness_audit' }, {
  methodFamily: 'association', methodKey: 'missingness_audit', description: '缺失审计', outputs: ['missingness_summary.csv'],
}, value => value);
assert.match(missingnessInput, /待审计的变量或测量产物/);
assert.match(missingnessInput, /可观察对象的人数/);
assert.doesNotMatch(missingnessInput, /包含暴露、结局及调整变量/);
const causalInput = owner.overview({ ...pkg, skill_id: 'causal_emulation.covariate_balance' }, {
  methodFamily: 'causal_emulation', methodKey: 'covariate_balance', description: '协变量平衡', outputs: [],
}, value => value);
assert.match(causalInput, /0\/1 治疗标记/);
assert.match(causalInput, /已有权重为可选输入/);
assert.doesNotMatch(causalInput, /估计因果效应时需要/);
const calibrationInput = owner.overview({ ...pkg, skill_id: 'prediction.calibration_metrics' }, {
  methodFamily: 'prediction', methodKey: 'calibration_metrics', description: '校准指标', outputs: [],
}, value => value);
assert.match(calibrationInput, /校准截距、校准斜率与 Brier 分数/);
assert.doesNotMatch(calibrationInput, /总体校准/);
const coxReading = owner.overview({ ...pkg, skill_id: 'time_to_event.cox_hr' }, { description: 'Cox' }, value => value);
assert.match(coxReading, /暴露变量或比较分组/);
assert.doesNotMatch(fs.readFileSync(ownerFile, 'utf8'), /fetch\s*\(|EU_API|eval\s*\(|new Function|location\.(?:hash|href)\s*=/);
process.stdout.write('Skill implementation modes, precise source mapping, escaping and read-only contract passed.\n');
