/* Review is a next step, never publication approval. Synthetic records only. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
global.window = global;
global.EU_LANG = 'zh';
global.t = (en, zh) => EU_LANG === 'zh' ? zh : en;
global.icon = () => '';
global.document = { addEventListener() {} };
for (const name of ['screens-guided-pi-modules.js', 'html-escape.js', 'screens-guided-pi-error-text.js',
  'screens-guided-pi-resources.js', 'screens-agent-reader-vocab.js', 'screens-agent-report-reader.js', 'screens-agent-render.js',
  'screens-guided-pi-run-files.js', 'screens-guided-pi-job-narration.js', 'screens-guided-pi-run-outcome.js']) {
  require(path.resolve('src/easyicu/webserver/static/js', name));
}
const modules = EasyICU.guidedPi;
const esc = EU_HTML.esc;
const resources = modules.require('resources').create({ esc });
const errors = modules.require('errorText').create({ tr: t, staticPreview: () => false });
const workflow = { stages: [{ id: 'analysis', status: 'complete' }] };
const ref = artifact => ({ kind: 'research_artifact', artifact, run_id: 'run_synthetic', sha256: 'a'.repeat(64) });
const latest = () => ({ present: true, run_id: 'run_synthetic', analysis_results_available: true,
  execution_complete: true, analysis_validated: true, numeric_verified: true, manuscript_ready: false,
  manuscript_outcome: { status: 'not_passed', reason_code: 'manuscript_checks_not_passed' },
  artifact_refs: ['scientific_readiness.json', 'evidence_ledger.json', 'result_tables.json'].map(ref) });
const finding = (code, extra = {}) => ({ code, domain: 'analysis', severity: 'major',
  message: 'A synthetic scientific issue needs assessment.', remediation: 'Inspect the saved study design.', ...extra });
async function outcome(findings = [], run = latest(), options = {}) {
  const payload = { run_id: run.run_id, domains: [], findings, ...options.payload };
  const view = modules.require('runOutcome').create({ tr: t, esc, iconHtml: () => '',
    resourceButton: resources.button, manuscriptNotPassedText: errors.manuscriptNotPassedText,
    projectId: options.projectId || (() => 'project_synthetic'), host: () => null,
    api: () => ({ loadPiCopilotResearchArtifact: options.load || (async (project, id, name, sha) => {
      assert.deepEqual([project, id, name, sha], ['project_synthetic', run.run_id, 'scientific_readiness.json', 'a'.repeat(64)]);
      return { ok: true, payload };
    }) }), canPreview: () => true,
  });
  if (!options.defer) await view.loadScientificReview(run, workflow);
  return { view, payload, html: view.render(run, workflow) };
}
const original = html => html.match(/<details class="gpi-review-original">[\s\S]*?<\/details>/)?.[0] || '';
const visible = html => html.replace(/<details\b[^>]*>[\s\S]*?<\/details>/g, '');
async function files(run = latest(), readOnly = false, signable = true, signed = false) {
  const saved = { ...run, study_id: 'study_synthetic', run_ref: 'rr_' + 'b'.repeat(32),
    analysis_outcome: { status: 'completed' } };
  const api = {
    listStudyContexts: async () => ({ contexts: [{ id: saved.study_id }] }),
    loadIdeaAgentProjects: async () => ({ projects: [{ study_id: saved.study_id, read_only: readOnly }] }),
    loadGuidedDrafts: async () => ({ drafts: [] }),
    loadAgentRunHistory: async () => ({ runs: [saved], count: 1 }),
    loadAgentRunReview: async () => ({ ...saved, readiness: { signable }, signed, artifacts: [] }),
  };
  const view = modules.require('runFiles').create({ api: () => api, changed() {}, resourceButton: resources.button,
    manuscriptNotPassedText: errors.manuscriptNotPassedText,
    context: () => ({ projectId: 'project_synthetic', studyId: saved.study_id, sessionId: 'session_synthetic' }) });
  await view.sync();
  await view.open(saved.run_ref);
  return view.render({ savedRuns: [saved] });
}

test('one stable run-keyed manuscript notice across result and file cards', async () => {
  const run = latest();
  const { view, html } = await outcome([], run);
  const combined = html + await files(run);
  assert.equal((combined.match(/data-manuscript-outcome="not_passed"/g) || []).length, 1);
  assert.match(html, /data-manuscript-run="run_synthetic"/);
  assert.match(html, /分析已完成。论文草稿仍需修改。/);
  assert.doesNotMatch(html, /论文稿没有通过自动检查|只有论文稿未通过检查/);
  assert.equal(view.render(run, workflow), html, 'rerendering must retain the notice');
  assert.doesNotMatch(view.render({ ...run, manuscript_ready: true }, workflow), /data-manuscript-outcome/);
  assert.doesNotMatch(view.render({ ...run, report_revision_ready: true }, workflow), /data-manuscript-outcome/);
});
test('a specific manuscript reason appears once in both languages', () => {
  for (const language of ['zh', 'en']) {
    EU_LANG = language;
    const lead = t('Analysis complete. The manuscript draft still needs revision.', '分析已完成。论文草稿仍需修改。');
    for (const code of [undefined, 'unknown_reason', 'manuscript_checks_not_passed']) {
      assert.equal(errors.manuscriptNotPassedText(code), lead);
    }
    const reason = t('Some numbers in the manuscript draft do not match the analysis results.', '论文稿中有部分数字与分析结果不一致。');
    assert.equal(errors.manuscriptNotPassedText('numeric_verified_not_satisfied'), lead + ' ' + reason);
  }
  EU_LANG = 'zh';
});
const known = [
  ['IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED', '既往研究依据尚未建立', '检索'],
  ['SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN', '重大科学修订尚未关闭', '逐项'],
  ['NOVELTY_POSITIONING_NOT_ESTABLISHED', '创新性定位尚未建立', '对照'],
  ['REPORTING_CHECKLIST_ITEMS_OPEN', '报告规范项目尚未关闭', '逐项'],
  ['PAPER_AUTHORITY_NOT_GRANTED', '尚未取得投稿使用权限', '审阅'],
  ['MODEL_INCOMPLETE', '预设模型尚未完成', '模型'],
  ['COHORT_SOURCE_SCOPE_NOT_EXPLICIT', '队列来源范围尚未写明', '来源人群'],
  ['DIRECT_COMPARATOR_SCREENING_NOT_ESTABLISHED', '直接对照文献筛选未完成', '筛选'],
  ['POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED', '暴露判定晚于时间零点', '随访'],
  ['REPEATED_STAY_DEPENDENCE_UNRESOLVED', '同一患者多次入住未处理', '患者标识'],
  ['ROBUSTNESS_AXES_TOO_NARROW', '稳健性分析维度不足', '方案'],
  ['MANUSCRIPT_CORE_SECTIONS_TOO_THIN', '稿件核心章节过于单薄', '补充'],
  ['RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED', '近期同类研究尚未核对', '检索'],
  ['ROBUSTNESS_REPLAY_DUPLICATES_PRIMARY', '稳健性分析重复了主要分析', '方案'],
  ['MANUSCRIPT_EXACT_LITERATURE_BINDING_INCOMPLETE', '稿件引用尚未对应到文献', '引用'],
  ['MANUSCRIPT_REQUIRED_SECTIONS_INCOMPLETE', '稿件缺少必要章节', '补齐'],
  ['MANUSCRIPT_RENDER_QA_INCOMPLETE', '论文 PDF 尚未完成核对', 'PDF'],
  ['FORMAL_PREREGISTRATION_NOT_ESTABLISHED', '研究预注册尚未核实', '人工核对'],
];
for (const [code, title, action] of known) test(`known finding: ${code}`, async () => {
  const { html } = await outcome([finding(code)]);
  const primary = visible(html);
  assert.ok(primary.includes(title));
  assert.ok(primary.includes(action));
  assert.doesNotMatch(primary.replace(/<[^>]*>/g, ''), /synthetic scientific|Inspect the saved|_[A-Z_]+|\.py|\.json/);
  assert.match(html, /<details class="gpi-response-review">/);
  EU_LANG = 'en';
  const english = visible((await outcome([finding(code)])).html);
  assert.match(english, /Next step:/);
  assert.doesNotMatch(english, /An analysis issue needs manual review|[\u3400-\u9fff]/);
  EU_LANG = 'zh';
});
test('unknown findings explicitly need manual review and preserve their full original text', async () => {
  const item = finding('UNLISTED_REVIEW_CODE', { domain: 'future_domain',
    message: 'Synthetic wording <script>unsafe()</script> ' + 'long statement '.repeat(100),
    remediation: 'Keep every sentence and punctuation.', authorization_question: 'An original question?' });
  const { html, payload } = await outcome([item]);
  assert.match(visible(html), /有一项分析问题需人工核对/);
  assert.match(visible(html), /请展开原始审阅意见/);
  assert.ok(original(html).includes(esc(JSON.stringify(payload, null, 2))), 'no truncation or translation of the original record');
  assert.match(original(html), /原始审阅意见/);
  assert.doesNotMatch(html, /<script>|gpi-review-original" open/);
});
test('a blocker takes priority, and authorization remains a manual decision', async () => {
  const { html } = await outcome([finding('ROBUSTNESS_AXES_TOO_NARROW'), finding('FORMAL_PREREGISTRATION_NOT_ESTABLISHED', {
    severity: 'blocker', requires_user_authorization: true, authorization_question: 'Confirm a new synthetic protocol?',
  })]);
  const primary = visible(html);
  assert.match(primary, /研究预注册尚未核实/);
  assert.match(primary, /需要你确认/);
  assert.doesNotMatch(primary, /data-gpi-run-outcome-retry|data-run-files-sign/);
});
test('resolved issues do not lead and unfamiliar domains still get a readable issue', async () => {
  const { html, payload } = await outcome([
    finding('MODEL_INCOMPLETE', { severity: 'blocker', status: 'resolved' }),
    finding('ROBUSTNESS_AXES_TOO_NARROW'),
    finding('FUTURE_ISSUE', { domain: 'future_domain' }),
    ...Array.from({ length: 18 }, (_, i) => finding(`EXTRA_${i}`, { domain: 'future_domain' })),
  ]);
  assert.doesNotMatch(visible(html), /预设模型尚未完成/);
  assert.match(html.replace(original(html), ''), /data-review-code="FUTURE_ISSUE"/);
  assert.ok(original(html).includes(esc(JSON.stringify(payload, null, 2))));
});
test('evidence is clickable only through a unique registered run and digest', async () => {
  const reference = 'preplan_literature_bundle.json.screening_decisions';
  const issue = finding('DIRECT_COMPARATOR_SCREENING_NOT_ESTABLISHED', { evidence_refs: [reference] });
  const entry = { reference, evidence_id: 'screening_record', sha256: 'b'.repeat(64), kind: 'log', pointer: '/screening_decisions' };
  const run = latest();
  for (const entries of [[], [{ ...entry, sha256: '' }], [entry, entry]]) {
    run.review_evidence_refs = entries;
    const { html } = await outcome([issue], run);
    assert.doesNotMatch(html, /data-evidence-id="screening_record"/);
    assert.match(visible(html), /文献筛选记录（暂不可查看）/);
  }
  run.review_evidence_refs = [entry];
  let rendered = await outcome([issue], run);
  assert.match(visible(rendered.html), /data-evidence-id="screening_record"/);
  assert.match(visible(rendered.html), /文献筛选记录/);
  assert.match(rendered.html, new RegExp(`data-evidence-sha256="${'b'.repeat(64)}"`));
  run.review_evidence_refs = [];
  run.artifact_refs.push(ref('preplan_literature_bundle.json'));
  rendered = await outcome([issue], run);
  assert.match(visible(rendered.html), /data-gpi-resource-artifact="preplan_literature_bundle.json"/);
  run.artifact_refs.at(-1).run_id = 'run_elsewhere';
  assert.doesNotMatch(visible((await outcome([issue], run)).html), /data-gpi-resource-artifact="preplan_literature_bundle.json"/);
  run.artifact_refs.at(-1).run_id = run.run_id;
  run.artifact_refs.at(-1).sha256 = 'invalid';
  assert.doesNotMatch(visible((await outcome([issue], run)).html), /data-gpi-resource-artifact="preplan_literature_bundle.json"/);
});
test('invalid, stale or failed review loads cannot supply a finding', async () => {
  for (const payload of [{ run_id: 'run_other', domains: [], findings: [finding('MODEL_INCOMPLETE')] }, null]) {
    const { html } = await outcome([], latest(), { load: async () => ({ ok: true, payload }) });
    assert.doesNotMatch(html, /预设模型尚未完成/);
  }
  let project = 'project_synthetic', complete;
  const run = latest();
  const { view } = await outcome([], run, { defer: true, projectId: () => project,
    load: () => new Promise(resolve => { complete = resolve; }) });
  const pending = view.loadScientificReview(run, workflow);
  project = 'project_new';
  complete({ ok: true, payload: { run_id: run.run_id, domains: [], findings: [finding('MODEL_INCOMPLETE')] } });
  await pending;
  assert.doesNotMatch(view.render(run, workflow), /预设模型尚未完成/);
});
test('domain-level review records keep their registered evidence links', async () => {
  const run = latest();
  run.review_evidence_refs = [{ reference: 'reviewer_report.json', evidence_id: 'domain_review',
    sha256: 'b'.repeat(64), kind: 'log' }];
  const { html } = await outcome([], run, { payload: { domains: [{ domain: 'analysis', status: 'blocked',
    summary: 'A synthetic domain needs expert assessment.', evidence_refs: ['reviewer_report.json'] }] } });
  assert.match(html, /data-evidence-id="domain_review"/);
  assert.ok(original(html).includes('A synthetic domain needs expert assessment.'));
});
test('result validity, numeric checking and manuscript status stay separate', async () => {
  for (const language of ['zh', 'en']) {
    EU_LANG = language;
    const { view, html } = await outcome([finding('MODEL_INCOMPLETE')]);
    assert.ok(html.includes(t('Results for review · manuscript review incomplete', '结果待审阅 · 论文尚未完成审阅')));
    assert.ok(html.includes(t('Numbers were checked against the result tables; conclusions still need expert review.', '已核对数字与结果表；研究结论仍需专业审阅。')));
    const pending = view.render({ ...latest(), analysis_validated: false, numeric_verified: false }, workflow);
    assert.doesNotMatch(pending, /已核对数字与结果表|Numbers were checked|分析已完成。论文|Analysis complete\. The manuscript/);
    assert.doesNotMatch(html, /依据运行闸门|numeric gate|幻觉审查/);
  }
  EU_LANG = 'zh';
});
test('recording review requires signability, confirmations and writable state', async () => {
  const checks = ['evidence_reviewed', 'claims_remain_locked', 'no_patient_rows_persisted'];
  for (const [readOnly, signable, signed] of [[false, true, false], [true, true, false], [false, false, false], [false, true, true]]) {
    const permitted = !readOnly && signable && !signed;
    const html = await files(latest(), readOnly, signable, signed);
    assert.equal(html.includes('data-run-files-sign'), permitted);
    if (permitted) assert.match(html, /本地审阅不授予发表权限/);
    let calls = 0;
    const controller = modules.require('runFiles').createController({ signoffAgentRun: async () => {
      calls++; return { signed: true };
    } });
    controller.state.selected = { readOnly };
    controller.state.review = { run_ref: 'rr_synthetic', readiness: { signable }, signed };
    await controller.sign(checks.slice(1));
    assert.equal(calls, 0);
    await controller.sign(checks);
    assert.equal(calls, Number(permitted));
  }
});
test('target trial narration uses clinical wording in both languages', () => {
  for (const language of ['zh', 'en']) {
    EU_LANG = language;
    assert.equal(modules.require('jobNarration').opening('target-trial-compile', t),
      t('I am preparing the analysis from the target-trial protocol.', '我正在按目标试验方案准备分析。'));
  }
  EU_LANG = 'zh';
});
