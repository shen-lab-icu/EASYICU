'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const modulesSource = fs.readFileSync(process.argv[2], 'utf8');
const source = fs.readFileSync(process.argv[3], 'utf8');
const escape = value => String(value == null ? '' : value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const context = {
  window: { EU_LANG: 'en', EU_HTML: { esc: escape } },
  globalThis: { pwned: false },
};
vm.createContext(context);
vm.runInContext(modulesSource, context);
vm.runInContext(source, context);
const renderer = context.window.EasyICU.guidedPi.require('evidencePreview');

const literature = renderer.render({
  renderer: 'literature_cards', previewable: true, kind: 'log', evidence_id: 'preplan_literature_bundle',
  sha256: 'a'.repeat(64), value: {
    cards: [{ title: '<img src=x onerror=bad()>', evidence_role: 'database', access_mode: 'abstract',
      source_url: 'https://pubmed.ncbi.nlm.nih.gov/12345/',
      evidence: [{ review_role: 'data_source', source_backed_summary: '<script>bad()</script>', locator: 'Abstract' }] },
    { title: 'Unsafe link', access_mode: 'unknown', source_url: 'javascript:alert(1)', evidence: [] }],
    review: {shortfalls: [{role: 'definition', actual: 0, target: 3}]},
  },
}, {});
assert.ok(literature.includes('Abstract excerpt; full text not reviewed'));
assert.ok(literature.includes('Relevance and support for each claim still require review'));
assert.ok(literature.includes('https://pubmed.ncbi.nlm.nih.gov/12345/'));
assert.ok(literature.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!literature.includes('<img src=x'));
assert.ok(!literature.includes('href="javascript:'));
assert.equal(context.globalThis.pwned, false);

const code = renderer.render({
  renderer: 'code', previewable: true, kind: 'code', evidence_id: 'code_1',
  sha256: 'a'.repeat(64), text: '<img src=x onerror="globalThis.pwned=true">\nestimate = 1.2',
}, { pointer: '/estimate', value: '<script>bad()</script>' });
assert.ok(code.includes('&lt;img src=x onerror=&quot;globalThis.pwned=true&quot;&gt;'));
assert.ok(code.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!code.includes('<img src=x'));
assert.equal(context.globalThis.pwned, false, 'preview code must never execute');

const table = renderer.render({
  renderer: 'table', previewable: true, kind: 'table', evidence_id: 'table_1',
  sha256: 'b'.repeat(64), headers: ['estimate'], rows: [['<svg onload=bad()>']],
}, {});
assert.ok(table.includes('&lt;svg onload=bad()&gt;'));
assert.ok(!table.includes('<svg onload=bad()>'));

const statistic = renderer.render({
  renderer: 'json', previewable: true, kind: 'statistic', evidence_id: 'statistic_1',
  sha256: 'd'.repeat(64), display_name: 'step_summary.json',
  relative_path: 'evidence/statistic_1__<bad>.json',
  description: '<img src=x onerror="bad()">', role: 'summarize_absolute_risk',
  producer: 'runner', generation_mode: 'fallback',
  declared_lineage: [
    { relation: 'analysis_code', status: 'registered', evidence_id: 'code_1', kind: 'code', sha256: 'a'.repeat(64), display_name: 'analysis.py', relative_path: 'evidence/code_1__analysis.py' },
    { relation: 'input_data', status: 'unregistered', evidence_id: 'missing_input' },
  ],
  run_authority: {
    status: 'recorded', run_id: 'run_1', git_sha: 'abc123', git_dirty: true,
    runner_image_digest: 'sha256:image', environment_identity_sha256: 'e'.repeat(64),
    links: [{ relation: 'run_plan_authority', status: 'registered', evidence_id: 'analysis_plan', kind: 'log', sha256: 'f'.repeat(64), display_name: 'analysis_plan.json', relative_path: 'evidence/analysis_plan__analysis_plan.json' }],
  },
  value: {
    analysis_family: 'absolute_risk_context', n_total: 50640, outcome: 'death',
    method: '<img src=x onerror="bad()">',
    reportable_descriptive_results: {
      overall_outcome: { outcome: 'death', n: 50640, event_n: 7006, risk_pct: 13.8, risk_ci_low_pct: 13.5, risk_ci_high_pct: 14.1 },
      exposures: [{ exposure: 'lact_max', groups: [{ label: 'above threshold', n: 100, outcome_n: 100, outcome_event_n: 20, outcome_risk_pct: 20, outcome_risk_ci_low_pct: 12, outcome_risk_ci_high_pct: 31 }] }],
    },
  },
}, { pointer: '/n_total', value: 50640 });
assert.ok(statistic.includes('Readable result'));
assert.ok(statistic.includes('Overall risk'));
assert.ok(statistic.includes('13.8%'));
assert.ok(statistic.includes('Registered source record'));
assert.ok(statistic.includes('Registry digest verified'));
assert.ok(statistic.includes('Reproduction path'));
assert.ok(statistic.includes('Code that generated this record'));
assert.ok(statistic.includes('data-evidence-id="code_1"'));
assert.ok(statistic.includes('missing_input'));
assert.ok(statistic.includes('Full file audit'));
assert.ok(statistic.includes('Continue to upstream run provenance'));
assert.ok(statistic.includes('data-evidence-id="analysis_plan"'));
assert.ok(statistic.includes('evidence/statistic_1__&lt;bad&gt;.json'));
assert.ok(statistic.includes('&lt;img src=x'));
assert.ok(statistic.includes('Open raw JSON for audit'));
assert.ok(!statistic.includes('<img src=x'));
assert.ok(statistic.indexOf('Readable result') < statistic.indexOf('Registered source record'));
const readerLayer = statistic.slice(0, statistic.indexOf('Full file audit'));
assert.ok(!readerLayer.includes('JSON pointer'));
assert.ok(!readerLayer.includes('statistic_1'));
assert.ok(!readerLayer.includes('d'.repeat(64)));
assert.ok(!readerLayer.includes('evidence/statistic_1__&lt;bad&gt;.json'));

const selectedPayload = {
  renderer: 'json', previewable: true, kind: 'statistic', evidence_id: 'primary_summary',
  sha256: 'd'.repeat(64), display_name: 'step_summary.json',
  value: { estimates: [{ rate: 23.45678 }, { rate: 0 }], 'escaped/key~': 8 },
};
const selected = renderer.render(selectedPayload, {
  pointer: '/estimates/0/rate', value: '23.45678', display: '23.46%',
});
const selectedReader = selected.slice(0, selected.indexOf('Full file audit'));
assert.ok(selectedReader.includes('Selected number source'));
assert.ok(selectedReader.includes('23.46%'));
assert.ok(selectedReader.includes('23.45678'));
assert.ok(selectedReader.includes('Source field and value match'));
assert.ok(!selectedReader.includes('/estimates/0/rate'), 'raw pointers remain in the audit layer');
const zero = renderer.render(selectedPayload, { pointer: '/estimates/1/rate', value: '0', display: '0%' });
assert.ok(zero.includes('Source field and value match'));
assert.ok(zero.includes('<strong>0</strong>'));
const escapedPointer = renderer.render(selectedPayload, { pointer: '/escaped~1key~0', value: '8' });
assert.ok(escapedPointer.includes('Source field and value match'));
for (const locator of [
  { pointer: '/estimates/0/rate', value: '999' },
  { pointer: '/estimates/1/rate', value: '23.45678' },
  { pointer: '/estimates/9/rate', value: '23.45678' },
  { pointer: '/estimates/0', value: '[object Object]' },
  { pointer: '/constructor', value: '8' },
  { pointer: '/escaped~2key~', value: '8' },
]) {
  const mismatch = renderer.render(selectedPayload, locator);
  assert.ok(mismatch.includes('Selected number could not be matched'));
  assert.ok(!mismatch.includes('Source field and value match'));
}
const selectedXss = renderer.render(selectedPayload, {
  pointer: '/estimates/0/rate', value: '23.45678', display: '<img src=x onerror="bad()">',
});
assert.ok(!selectedXss.includes('<img src=x'));
assert.ok(selectedXss.includes('&lt;img src=x'));

const distribution = {
  ...selectedPayload,
  value: {
    exposure: 'event_status', outcome: 'hospital_endpoint',
    descriptive_estimates: {
      schema_version: 'easyicu.exposure_outcome_descriptive_estimates/1',
      exposure_prevalence: [
        { level: 0, n: 60, denominator: 100, estimate_pct: 60 },
        { level: 1, n: 40, denominator: 100, estimate_pct: 40 },
      ],
      outcome_absolute_risks: [
        { level: 0, events: 6, denominator: 60, estimate_pct: 10 },
        { level: 1, events: 8, denominator: 40, estimate_pct: 20 },
      ],
    },
  },
};
const originalDistribution = JSON.stringify(distribution);
function selectionOnly(payload, pointer, value) {
  const rendered = renderer.render(payload, { pointer, value: String(value), display: String(value) });
  return rendered.slice(0, rendered.indexOf('</section>') + '</section>'.length);
}
const prevalenceSelection = selectionOnly(distribution, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40);
assert.ok(prevalenceSelection.includes('Group share of the cohort'));
assert.ok(prevalenceSelection.includes('40 / 100'));
assert.ok(prevalenceSelection.includes('Recorded group value'));
assert.ok(!prevalenceSelection.includes('hospital_endpoint'));
assert.ok(!prevalenceSelection.includes('8 / 40'));
const outcomeSelection = selectionOnly(distribution, '/descriptive_estimates/outcome_absolute_risks/1/estimate_pct', 20);
assert.ok(outcomeSelection.includes('Observed outcome proportion within the group'));
assert.ok(outcomeSelection.includes('8 / 40'));
assert.ok(outcomeSelection.includes('hospital_endpoint'));
assert.ok(!outcomeSelection.includes('40 / 100'));
for (const [family, metric, field, count] of [
  ['exposure_prevalence', 'Records in this group', 'n', 40],
  ['exposure_prevalence', 'Cohort denominator', 'denominator', 100],
  ['outcome_absolute_risks', 'Outcome events in this group', 'events', 8],
  ['outcome_absolute_risks', 'Outcome denominator in this group', 'denominator', 40],
]) {
  assert.ok(selectionOnly(distribution, `/descriptive_estimates/${family}/1/${field}`, count).includes(metric));
}
const wrongSelectedValue = renderer.render(distribution, {pointer:'/descriptive_estimates/exposure_prevalence/0/estimate_pct',value:'40'});
assert.ok(wrongSelectedValue.includes('Selected number could not be matched'));
assert.ok(!wrongSelectedValue.includes('Selected metric'));
const unsupported = JSON.parse(originalDistribution);
unsupported.value.descriptive_estimates.schema_version = 'unknown';
assert.match(selectionOnly(unsupported, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40), /Selected metric<\/span><strong>Not recorded/);
const invalidDenominator = JSON.parse(originalDistribution);
invalidDenominator.value.descriptive_estimates.exposure_prevalence[1].denominator = -1;
assert.ok(!selectionOnly(invalidDenominator, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40).includes('Group records / cohort records'));
const dangerousGroup = JSON.parse(originalDistribution);
dangerousGroup.value.descriptive_estimates.exposure_prevalence[1].level = '<img src=x onerror=bad()>';
const escapedGroup = selectionOnly(dangerousGroup, '/descriptive_estimates/exposure_prevalence/1/estimate_pct', 40);
assert.ok(escapedGroup.includes('&lt;img src=x'));
assert.ok(!escapedGroup.includes('<img src=x'));
assert.equal(JSON.stringify(distribution), originalDistribution, 'source JSON remains unchanged');
const selectedWithContext = renderer.render(distribution, {pointer:'/descriptive_estimates/exposure_prevalence/1/estimate_pct',value:'40'});
assert.ok(selectedWithContext.includes('<summary>Whole-file context (not the selected metric)</summary>'));

for (const [confidence, label] of [[0.90, '90% CI'], [null, 'Confidence interval'], [95, 'Confidence interval']]) {
  const html = renderer.render({...selectedPayload, value:{estimate:1.2,ci:[1,1.4],confidence_level:confidence}});
  assert.ok(html.includes(`<span>${label}</span>`));
  assert.ok(!html.includes('95% CI'), 'never infer a confidence level from two endpoints');
  const grouped = renderer.render({...selectedPayload, value:{
    reportable_descriptive_results:{exposures:[{exposure:'event_status',groups:[{
      label:'recorded',outcome_risk_ci_low_pct:10,outcome_risk_ci_high_pct:30,confidence_level:confidence,
    }]}]},
  }});
  assert.ok(grouped.includes('<th>Confidence interval</th>'));
  assert.ok(!grouped.includes('95% CI'));
  if (confidence === 0.90) assert.ok(grouped.includes('(90% CI)'));
}
const declared95 = renderer.render({...selectedPayload, value:{estimate:1.2,ci:[1,1.4],confidence_level:0.95}});
assert.ok(declared95.includes('<span>95% CI</span>'), 'retain explicitly recorded confidence levels');

const withheld = renderer.render({
  renderer: 'metadata', previewable: false, kind: 'table', evidence_id: 'cohort_1',
  sha256: 'c'.repeat(64), withheld_reason: 'patient_level_rows_withheld', bytes: 1024,
}, { pointer: '/estimates/0/rate', value: '23.45678', display: '23.46%' });
assert.ok(withheld.includes('Patient-level cohort rows are withheld'));
assert.ok(!withheld.includes('<table'));
assert.ok(!withheld.includes('Source field and value match'));
assert.ok(!withheld.includes('23.46%'));

// Recorded producer semantics apply only to the exact selected field.
const tableOne = {...selectedPayload, value:{
  analysis_family:'grouped_table_one', cohort_n:24, group_by:'category',
  reportable_descriptive_results:{overall_outcome:{outcome:'unrelated_endpoint',n:99}},
}};
assert.match(selectionOnly(tableOne, '/cohort_n', 24), /Table 1 cohort denominator/);
const model = {...selectedPayload, value:{
  analysis_family:'association', effect_scale:'odds_ratio', primary_or:1.25,
  primary_ci_low:1.1, primary_ci_high:1.4, n_total:24, n_events:0,
  primary_contrast:{exposure_level:'upper',reference_level:'lower'},
  adjustment_covariates:['baseline_covariate'],
  landmark_runtime_receipt:{schema_version:'easyicu.landmark_categorical_association_runtime_receipt/1',landmark_hours:12},
  unrelated:{estimate:1.25,n_total:99,outcome:'unrelated_endpoint'},
}};
const modelSelection = selectionOnly(model, '/primary_or', 1.25);
for (const fact of ['Odds ratio', 'upper', 'lower', '24', 'baseline_covariate', '12']) assert.ok(modelSelection.includes(fact));
assert.match(modelSelection, /<span>Group<\/span><strong>upper<\/strong>/);
assert.match(modelSelection, /<span>Reference group<\/span><strong>lower<\/strong>/);
assert.ok(!modelSelection.includes('unrelated_endpoint'));
assert.match(selectionOnly(model, '/primary_ci_low', 1.1), /lower bound/);
assert.match(selectionOnly(model, '/n_events', 0), /Model outcome events/);
const unrelatedSelection = selectionOnly(model, '/unrelated/estimate', 1.25);
assert.match(unrelatedSelection, /Selected metric<\/span><strong>Not recorded/);
assert.ok(!unrelatedSelection.includes('baseline_covariate'));
const futureModel = {...model, value:{...model.value,schema_version:'future/99'}};
assert.ok(!selectionOnly(futureModel, '/primary_or', 1.25).includes('Odds ratio'));
const futureReceipt = {...model,value:{...model.value,landmark_runtime_receipt:{schema_version:'future/1',landmark_hours:12}}};
assert.match(selectionOnly(futureReceipt,'/primary_or',1.25), /Time window \/ anchor<\/span><strong>Not recorded/);
const unsafeModel = {...model,value:{...model.value,primary_contrast:{exposure_level:'<svg onload=bad()>',reference_level:0},adjustment_covariates:['<img src=x onerror=bad()>']}};
const unsafeModelHtml = selectionOnly(unsafeModel,'/primary_or',1.25);
assert.ok(!unsafeModelHtml.includes('<svg onload'));
assert.ok(!unsafeModelHtml.includes('<img src=x'));
assert.ok(unsafeModelHtml.includes('&lt;svg onload'));
assert.match(unsafeModelHtml, /Reference group<\/span><strong>0/);
const missingFacts = selectionOnly({...model,value:{analysis_family:'association',primary_or:1.25}}, '/primary_or', 1.25);
for (const label of ['Group', 'Reference group', 'Denominator / sample', 'Unit', 'Time window / anchor', 'Adjustment']) {
  assert.ok(missingFacts.includes(`<span>${label}</span><strong>Not recorded</strong>`), label);
}
const risk = {...selectedPayload,value:{
  analysis_family:'absolute_risk_context',n_total:24,
  reportable_descriptive_results:{schema_version:'easyicu.absolute_risk_reporting/1',
    overall_outcome:{outcome:'overall_only',n:24,event_n:6,risk_pct:25},
    exposures:[{exposure:'category',unit:'category',groups:[
      {label:'upper',n:8,outcome_n:8,outcome_event_n:0,outcome_risk_pct:0,prevalence_pct:100/3},
    ]}],
  },
}};
const groupRisk = selectionOnly(risk, '/reportable_descriptive_results/exposures/0/groups/0/outcome_risk_pct', 0);
for (const fact of ['Observed outcome proportion within the group','upper','8','%']) assert.ok(groupRisk.includes(fact));
assert.ok(!groupRisk.includes('overall_only'));
assert.match(selectionOnly(risk, '/reportable_descriptive_results/overall_outcome/risk_pct', 25), /Overall outcome proportion/);
assert.match(selectionOnly(risk, '/n_total', 24), /Absolute-risk analysis sample/);
assert.match(selectionOnly(risk, '/reportable_descriptive_results/exposures/0/groups/0/prevalence_pct', 100/3), /Group share of the cohort/);
const sensitivity = {...model,value:{...model.value,analysis_rows:[
  {odds_ratio:1.3,ci_low:1.05,effect_measure:'odds_ratio',n_stays:18,strategy:'functional_form'},
]}};
const sensitivitySelection = selectionOnly(sensitivity, '/analysis_rows/0/odds_ratio', 1.3);
assert.match(sensitivitySelection, /Odds ratio/);
assert.match(sensitivitySelection, /<strong>18<\/strong>/);
assert.ok(!sensitivitySelection.includes('baseline_covariate'));
assert.ok(!sensitivitySelection.includes('upper'));
assert.match(selectionOnly(sensitivity, '/analysis_rows/0/ci_low', 1.05), /lower bound/);
const mismatchContext = renderer.render(model,{pointer:'/primary_or',value:'1.2'});
assert.ok(!mismatchContext.includes('Selected metric'));
assert.ok(mismatchContext.includes('Selected number could not be matched'));

// Record summaries lead; unknown records and all raw JSON stay in audit.
context.window.EU_LANG = 'zh';
const logPayload = {renderer:'json',previewable:true,kind:'log',sha256:'b'.repeat(64)};
const records = [
  ['run_status.json', '运行状态', {schema_version:'easyicu.run_status/2',status:'manuscript_ready',gates:{execution_complete:true,paper_authorized:false}}, ['稿件已生成','投稿使用权限','未通过']],
  ['reviewer_report.json', '科学审阅', {summary:{aggregated_recommendation:'major_revision',review_mode:'simulated_deterministic',counts:{major:1,minor:0}},critiques:[{comments:[{severity:'major',message:'需要检查区间。'}]}]}, ['需重大修订','模拟审阅','需要检查区间。']],
  ['scientific_maturity_audit.json','待处理问题',{schema_version:'easyicu.scientific_maturity/1',status:'analysis_only',findings:[{severity:'blocker',message:'还需核对设计。'}]},['仅供分析','还需核对设计。']],
  ['critique_report.json','步骤检查',{status:'pass',concerns:[],unsupported_claims:[],missing_evidence_refs:[],suggested_repairs:[]},['通过','0']],
];
for (const [display_name, purpose, value, facts] of records) {
  const payload = {...logPayload,display_name,value};
  const html = renderer.render(payload);
  assert.equal(renderer.title(payload),purpose);
  const first = html.slice(0,html.indexOf('完整文件审计'));
  assert.ok(first.includes(purpose));
  for (const fact of facts) assert.ok(first.includes(fact), fact);
  assert.ok(!first.includes('gpi-evidence-json'));
  assert.equal((html.match(/<pre /g)||[]).length,1,'raw JSON appears once, under audit');
}
const maturityStatistic = renderer.render({...logPayload,kind:'statistic',display_name:'scientific_maturity_audit.json',
  value:{schema_version:'easyicu.scientific_maturity/1',status:'analysis_only',findings:[{message:'核对设计。'}]},
});
assert.ok(maturityStatistic.slice(0,maturityStatistic.indexOf('完整文件审计')).includes('仅供分析'));
const unsafeReview = renderer.render({...logPayload,display_name:'critique_report.json',value:{
  status:'needs_revision',concerns:['<img src=x onerror=bad()>'],missing_evidence_refs:['missing_source'],suggested_repairs:['核对来源'],unsupported_claims:[],
}});
assert.ok(unsafeReview.includes('&lt;img src=x'));
assert.ok(!unsafeReview.includes('<img src=x'));
assert.ok(unsafeReview.includes('有待处理项'));
assert.ok(!unsafeReview.includes('data-gpi-evidence-open'));
for (const value of [{status:'pass'}, {schema_version:'future/1',status:'pass',concerns:[],unsupported_claims:[],missing_evidence_refs:[],suggested_repairs:[]}]) {
  const unknownReview = renderer.render({...logPayload,display_name:'critique_report.json',value});
  const first = unknownReview.slice(0,unknownReview.indexOf('完整文件审计'));
  assert.ok(first.includes('未记录'));
  assert.ok(!first.includes('<strong>通过</strong>'));
}
const unknownJson = renderer.render({...logPayload,display_name:'unknown.json',value:{status:'pass'}});
assert.ok(unknownJson.slice(0,unknownJson.indexOf('完整文件审计')).includes('未记录'));

// Exercise the real run card without changing the shared runner's owner map.
const path = require('node:path');
vm.runInContext(fs.readFileSync(path.join(path.dirname(process.argv[3]),'screens-guided-pi-run-outcome.js'),'utf8'),context);
async function checkRecordLabels() {
  const buttons = [];
  const refs = ['run_status.json','reviewer_report.md','scientific_maturity_audit.json','critique_report.json','missing_record.json'];
  const latest = {run_id:'run_synthetic',present:true,analysis_results_available:true,
    artifact_refs:[{run_id:'run_synthetic',artifact:'scientific_readiness.json',sha256:'d'.repeat(64)}],
    review_evidence_refs:refs.slice(0,4).map((reference,i)=>({reference,evidence_id:`record_${i}`,sha256:'b'.repeat(64),kind:'log'})),
  };
  const workflow = {stages:[{id:'analysis',status:'complete'}]};
  const outcome = context.window.EasyICU.guidedPi.require('runOutcome').create({
    tr:(en,zh)=>zh,esc:escape,iconHtml:()=>'',projectId:()=> 'project_synthetic',
    resourceButton:(ref,label)=>{buttons.push({ref,label});return `<button>${escape(label)}</button>`;},
    manuscriptNotPassedText:()=>'',
    api:()=>({loadPiCopilotResearchArtifact:async()=>({ok:true,payload:{run_id:latest.run_id,
      domains:[{domain:'analysis',status:'warning',summary:'检查记录',evidence_refs:refs.slice(0,3)},
        {domain:'manuscript',status:'warning',summary:'核对记录',evidence_refs:refs.slice(3)}],findings:[],
    }})}),
  });
  await outcome.loadScientificReview(latest,workflow);
  const html = outcome.render(latest,workflow);
  for (const [index,label] of ['运行状态','科学审阅','待处理问题','步骤检查'].entries()) {
    assert.ok(buttons.some(b=>b.ref.evidence_id===`record_${index}` && b.label===label), label);
  }
  assert.ok(!buttons.some(b=>b.label==='missing_record.json'));
  assert.ok(html.includes('gpi-review-evidence-missing'));
  assert.ok(!html.includes('>missing_record.json<'));
  latest.review_evidence_refs[0].sha256='invalid';
  latest.review_evidence_refs[1].evidence_id='../wrong';
  buttons.length=0;
  outcome.render(latest,workflow);
  assert.ok(!buttons.some(b=>['record_0','../wrong'].includes(b.ref.evidence_id)));
  latest.review_evidence_refs.push({...latest.review_evidence_refs[2]});
  buttons.length=0;
  outcome.render(latest,workflow);
  assert.ok(!buttons.some(b=>b.ref.evidence_id==='record_2'));
}
checkRecordLabels().then(()=>process.stdout.write(JSON.stringify({ ok: true, suites: [
  'safe_preview','exact_source_selection','typed_metric_context','confidence_level_metadata',
  'producer_metric_context','readable_records','review_record_labels',
] }))).catch(error=>{console.error(error);process.exitCode=1;});
