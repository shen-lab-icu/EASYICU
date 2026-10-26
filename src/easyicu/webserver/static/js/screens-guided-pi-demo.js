/* Guided Pi reviewer-demo transcript.
   Owner: read-only reviewer fixtures and their safe structured preview.
   It never starts a provider job or mutates a real EasyICU project. */
(function () {
  'use strict';

  const SOURCE_RUN_ID = 'run_20260815T061842_5049c6';
  // The answer shows the registered result table drawn by the current figure
  // renderer; the run's own figure stays unchanged and one click away.
  const FIGURE_URL = 'assets/demo/sofa2-phenotype-mortality-rerendered.png?v=20261026-fig2';
  const ORIGINAL_FIGURE_URL = 'assets/demo/sofa2-phenotype-mortality.png?v=20260922-demo2';
  // The registered result laid out as a journal-style descriptive report
  // (tools/build_demo_study_report.py); it is not the withheld manuscript.
  const STUDY_REPORT_URL = { en: 'assets/demo/sofa2-study-report.en.html?v=20261026-fig1', zh: 'assets/demo/sofa2-study-report.zh.html?v=20261026-fig1' };
  const WRAPPER_RUN_ID = 'e59d1a54feff';
  const SOURCE_AUTHORITY = 'bounded_reviewer_projection_from_registered_run';

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  const RUN_FILES_URL = 'assets/demo/run-files/';
  const RUN_FILES_TOKEN = '20261026-run-files1';
  const RUN_FILES = {
    'run_context.json': () => tr('Run context', '运行上下文'),
    'cohort_summary.json': () => tr('Cohort summary', '队列摘要'),
    'quality_gate.json': () => tr('Evidence verification', '证据核验'),
    'agent_plan.json': () => tr('Research plan', '研究计划'),
    'literature_evidence.json': () => tr('Literature evidence', '文献证据'),
    'scientific_plan_review.json': () => tr('Scientific plan review', '科学计划审阅'),
    'scientific_readiness.json': () => tr('Scientific review', '科学审阅'),
    'manuscript_draft.json': () => tr('Locked manuscript draft', '锁定论文草稿'),
    'figure_gallery.json': () => tr('Figure gallery', '图件画廊'),
    'result_tables.json': () => tr('Research result tables', '科研结果表'),
    'source_run_manifest.json': () => tr('Source run manifest', '原始运行清单'),
    'evidence_ledger.json': () => tr('Evidence ledger', '证据账本'),
  };
  function demoArtifact(name, title, summary, metrics, sections, extra) {
    return Object.assign({
      schema_version: 'easyicu.pi-reviewer-demo-artifact/1',
      artifact: name,
      title,
      summary,
      status: 'reviewer_demo_complete',
      source_run_id: SOURCE_RUN_ID,
      source_authority: SOURCE_AUTHORITY,
      authority_class: 'engineering_validation_only',
      reportable: false,
      publication_authorized: false,
      claim_ceiling: 'descriptive_only',
      metrics: metrics || [],
      sections: sections || [],
    }, extra || {});
  }
  function artifacts() {
    const rows = {
      'descriptive_results.json': demoArtifact(
        'descriptive_results.json',
        tr('Registered aggregate results', '已登记聚合结果'),
        tr('All values are copied from the run-bound descriptive evidence. No inferential result is added by the demo.', '所有数值均复制自运行绑定的描述性证据；Demo 不新增任何推断结果。'),
        [
          { label: tr('Adult ICU stays', '成人 ICU stays'), value: '94,458' },
          { label: tr('Phenotype present', '表型阳性'), value: '33,997 / 94,458 (35.99%)' },
          { label: tr('Observed deaths · absent', '观察死亡 · 阴性'), value: '4,986 / 60,461 (8.25%)' },
          { label: tr('Observed deaths · present', '观察死亡 · 阳性'), value: '4,480 / 33,997 (13.18%)' },
        ],
        [
          { heading: tr('Interpretation ceiling', '解读上限'), items: [tr('These are observed descriptive proportions, not causal effects or ordinary baseline-exposure associations.', '这些是观察到的描述性比例，不是因果效应或普通基线暴露关联。'), tr('The first-24-hour ascertainment period leaves exposure-opportunity and early-event timing unresolved.', '入 ICU 后 24 小时判定窗口仍存在暴露机会与早期事件时间未闭合问题。')] },
        ],
        {
          tables: [{
            label: tr('Counts-only phenotype and mortality distribution', '仅计数的表型与死亡分布'),
            headers: [tr('Phenotype', '表型'), tr('ICU stays', 'ICU stays'), tr('Cohort share', '队列占比'), tr('Deaths', '死亡'), tr('Observed mortality', '观察死亡率')],
            rows: [
              [tr('Absent', '阴性'), '60,461', '64.01%', '4,986', '8.25%'],
              [tr('Present', '阳性'), '33,997', '35.99%', '4,480', '13.18%'],
              [tr('Overall', '总体'), '94,458', '100%', '9,466', '10.02%'],
            ],
          }],
        },
      ),
      'authority_verdict.json': demoArtifact(
        'authority_verdict.json',
        tr('Reviewer verdict: demonstration complete', '审稿结论：演示完整完成'),
        tr('The systems demonstration passed its engineering criteria. The clinical manuscript remains unauthorized because that is a separate scientific gate.', '系统演示通过工程标准；临床稿件仍未授权，因为它属于另一套科学闸门。'),
        [
          { label: tr('Reviewer demo', '审稿人 Demo'), value: 'COMPLETE' },
          { label: tr('Engineering validation', '工程验证'), value: 'COMPLETE' },
          { label: tr('Clinical manuscript', '临床稿件'), value: 'WITHHELD' },
          { label: tr('Publication authority', '发表权限'), value: 'NOT GRANTED' },
        ],
        [
          { heading: tr('Why this is not a failed Demo', '为什么这不是 Demo 失败'), items: [tr('The reviewer question is whether the system completes governed analysis and preserves authority boundaries. Both behaviors were observed.', '审稿问题是系统能否完成受治理分析并保持权限边界；两项行为均已观察到。'), tr('Calling the whole workflow “blocked” conflates product completion with clinical publication readiness. The interface now reports them separately.', '把整个流程称为“阻断”混淆了产品完成度与临床投稿就绪度；界面现已分别报告。')] },
          { heading: tr('What remains for a systems paper', '系统论文仍需完成'), items: [tr('Prespecified multi-task and multi-database benchmarks.', '预先规定的多任务、多数据库 benchmark。'), tr('Governed-versus-ungoverned baselines and authority-boundary ablations.', '受治理与不受治理 baseline 及权限边界消融。'), tr('Independent expert evaluation and reproducibility/time/cost comparison.', '独立专家评估及可复现性、时间、成本比较。')] },
        ],
      ),
    };
    // The standard run files are the source run's own files, published by
    // tools/build_demo_run_files.py and read by the same views as any run.
    Object.keys(RUN_FILES).forEach(name => { rows[name] = { artifact: name, title: RUN_FILES[name](), run_file: true }; });
    return rows;
  }

  function artifactResource(name, label) {
    const item = artifacts()[name];
    return {
      kind: 'demo_artifact', artifact: name, label: label || name,
      title: item ? item.title : name, run_id: SOURCE_RUN_ID,
      media_type: 'application/json',
    };
  }
  function documentResource(name, label, mediaType) {
    return { kind: 'demo_document', artifact: name, label: label || name, run_id: WRAPPER_RUN_ID, media_type: mediaType };
  }
  function reviewResources() {
    return [
      documentResource('system-validation-report.html', tr('Open reviewer dossier', '打开审稿人报告'), 'text/html'),
      documentResource('system-validation-report.pdf', tr('Open six-page PDF', '打开 6 页 PDF'), 'application/pdf'),
    ];
  }
  function activity(id, startedAt, endedAt, steps, extra) { return Object.assign({ id, role: 'activity', status: 'complete', startedAt, endedAt, steps, expanded: true }, extra || {}); }
  function message(id, role, text, resources) { return { id, role, text, complete: true, resources: resources || [] }; }
  function standardRunResources() {
    return [
      artifactResource('run_context.json', tr('Run context', '运行上下文')),
      artifactResource('cohort_summary.json', tr('Cohort summary', '队列摘要')),
      artifactResource('quality_gate.json', tr('Evidence verification', '证据核验')),
      artifactResource('agent_plan.json', tr('Research plan', '研究计划')),
      artifactResource('literature_evidence.json', tr('Literature evidence', '文献证据')),
      artifactResource('scientific_plan_review.json', tr('Scientific plan review', '科学计划审阅')),
      artifactResource('scientific_readiness.json', tr('Scientific review', '科学审阅')),
      artifactResource('manuscript_draft.json', tr('Locked manuscript draft', '锁定论文草稿')),
      artifactResource('figure_gallery.json', tr('Figure gallery', '图件画廊')),
      artifactResource('result_tables.json', tr('Research result tables', '科研结果表')),
      artifactResource('source_run_manifest.json', tr('Source run manifest', '原始运行清单')),
      artifactResource('evidence_ledger.json', tr('Evidence ledger', '证据账本')),
    ];
  }

  // A demo trace row: named for a reader and kept as its own row. It shows a
  // time only when the run's audit log recorded one (sofa2-run-trace.json).
  function traceStep(id, label, text, resource, resources) {
    return { id, kind: 'pipeline', step: 'plan_step', status: 'complete', label, text: text || '', resource: resource || null, resources: resources || [], durationKnown: false };
  }
  function recordedSpan(trace, id) {
    const span = trace && trace.spans && trace.spans[id];
    if (!Array.isArray(span)) return null;
    const started = Date.parse(span[0]);
    const ended = Date.parse(span[1]);
    return Number.isFinite(started) && Number.isFinite(ended) && ended >= started ? [started, ended] : null;
  }
  function timed(trace, step) {
    const span = recordedSpan(trace, step.id);
    return span ? Object.assign(step, { startedAt: span[0], endedAt: span[1], durationKnown: true }) : step;
  }
  // An activity's total is its recorded first start to last end; an activity
  // with an unrecorded row claims no total.
  function timedActivity(id, steps, extra) {
    const known = steps.filter(step => step.durationKnown !== false);
    const row = activity(id, null, null, steps, extra);
    if (!known.length) return Object.assign(row, { durationKnown: false });
    return Object.assign(row, {
      startedAt: Math.min(...known.map(step => step.startedAt)),
      endedAt: Math.max(...known.map(step => step.endedAt)),
    });
  }

  /* The walkthrough reads as one ordinary research conversation in six
     chapters: the question, how its words map to the data, the plan and its
     literature, the researcher's confirmation, the analysis, and the reading
     of the result. A `card` beat is a traceability stop the replay owner
     (screens-guided-pi-demo-walkthrough.js) renders from the run's trace. */
  function chapters(trace) {
    const documents = reviewResources();
    const run = standardRunResources();
    const resource = name => run.find(row => row.artifact === name);
    const step = (...args) => timed(trace, traceStep(...args));
    const evidenceCount = trace && trace.evidence && Number(trace.evidence.count);
    const registered = Number.isFinite(evidenceCount) && evidenceCount > 0 ? evidenceCount.toLocaleString('en-US') : '125';
    return [
      { id: 'question', label: tr('Question', '提问'), beats: [
        message('reviewer-user-1', 'user', tr(
          'In MIMIC-IV adult ICU stays, how common is the experimental SOFA-2 sepsis phenotype within the first 24 hours of admission, and how does in-hospital mortality differ with and without it?',
          '在 MIMIC-IV 的成人 ICU 入住中，入 ICU 后 24 小时内出现实验性 SOFA-2 脓毒症表型的比例有多高？有和没有这个表型的患者，院内死亡率差多少？',
        )),
      ] },
      { id: 'data', label: tr('Data mapping', '数据映射'), beats: [
        timedActivity('reviewer-context', [
          step('p-scope', tr('Checked the question and the data scope', '核对研究问题与数据范围'), tr('MIMIC-IV and adult ICU stays confirmed.', '已确认使用 MIMIC-IV 与成人 ICU 入住。'), resource('run_context.json')),
          step('p-cohort', tr('Analysis population: 94,458 adult ICU stays', '分析人群：94,458 个成人 ICU 入住'), tr('Age ≥ 18; no stay was excluded.', '年龄 ≥ 18 岁，没有入住被排除。'), resource('cohort_summary.json')),
        ], { displayTitle: tr('The question is mapped to the data', '问题已对应到数据') }),
        { card: 'dictionary', stop: true },
      ] },
      { id: 'plan', label: tr('Plan and literature', '计划与文献'), beats: [
        timedActivity('reviewer-planning', [
          step('p-literature', tr('Collected the methods references', '整理方法学依据'), tr('Nine methods references kept (STROBE, RECORD, Sepsis-3 and others); no novelty search was run.', '保留 9 篇方法学文献（STROBE、RECORD、Sepsis-3 等）；这次没有做新颖性检索。'), resource('literature_evidence.json')),
          step('p-draft', tr('Plan draft 1 passed the contract checks: six steps', '计划草案第 1 版通过合同检查：共 6 个步骤'), '', resource('agent_plan.json')),
          step('p-review', tr('Scientific review of the plan', '计划科学审阅'), tr('Ready to run; the missing novelty search and independent review are kept as limitations.', '可以执行；缺少的新颖性检索和独立审阅记为局限。'), resource('scientific_plan_review.json')),
        ], { displayTitle: tr('Plan ready for your review', '计划已就绪，等待你确认'), childJobId: 'reviewer-plan-run' }),
        message('reviewer-assistant-1', 'assistant', tr(
          'I drafted a six-step descriptive plan. Please confirm it before the analysis starts:\n\n- **Population**: all adult ICU stays in MIMIC-IV (94,458)\n- **Exposure**: the experimental SOFA-2 sepsis phenotype within 24 hours of ICU admission (yes / no)\n- **Outcome**: in-hospital death\n- **Method**: counts and proportions by phenotype; no adjustment and no significance test\n- **Data quality**: a separate audit of missing values that keeps not-applicable death times apart from true missingness\n\nThe complete plan and its methods references are below.',
          '我拟定了一份 6 步的描述性分析计划，开始分析前请你确认：\n\n- **人群**：MIMIC-IV 中全部成人 ICU 入住（94,458 个）\n- **暴露**：入 ICU 后 24 小时内是否出现实验性 SOFA-2 脓毒症表型\n- **结局**：院内死亡\n- **方法**：按有无表型计数并计算比例，不做调整，不做显著性检验\n- **数据质量**：单独审计缺失值，并把“不适用”的死亡时间和真正的缺失分开\n\n完整计划和方法学依据可以从下方打开。',
        ), [resource('agent_plan.json'), resource('literature_evidence.json'), resource('scientific_plan_review.json')]),
        { card: 'literature', stop: true },
      ] },
      { id: 'confirm', label: tr('Your confirmation', '你确认'), beats: [
        // In a project only the review control approves a plan; a chat
        // message cannot, so the walkthrough waits for that control.
        { card: 'confirm', stop: 'confirm' },
        {
          id: 'reviewer-plan-approval', role: 'demo_action',
          label: tr('Plan confirmed: “Confirm plan and prepare data”', '已确认方案：「确认方案并准备数据」'),
          note: tr('Pressed in the review control under the plan; typing in the conversation does not approve a plan.',
            '在计划下方的审阅控件中点击；在对话里打字不会批准计划。'),
        },
      ] },
      { id: 'analysis', label: tr('Analysis and figures', '分析与图表'), beats: [
        timedActivity('reviewer-execution', [
          step('e-1', tr('Step 1/6: Cohort and denominators', '第 1/6 步：队列与分母核算'), tr('All 94,458 stays retained.', '94,458 个入住全部保留。'), resource('cohort_summary.json')),
          step('e-2', tr('Step 2/6: Measurement and missingness audit', '第 2/6 步：测量与缺失审计'), tr('Death time applies only to the 9,466 deaths; 0 of them missing.', '死亡时间只对 9,466 例死亡适用，其中缺失 0 例。'), resource('result_tables.json')),
          step('e-3', tr('Step 3/6: Group proportions and outcomes', '第 3/6 步：分组比例与结局'), tr('Counts and proportions only; no inferential estimate added.', '只计数和比例，不新增推断估计。'), resource('result_tables.json')),
          step('e-4', tr('Step 4/6: Result figure', '第 4/6 步：结果图'), '', resource('figure_gallery.json')),
          step('e-5', tr('Step 5/6: Cohort flow figure', '第 5/6 步：队列流程图'), '', resource('figure_gallery.json')),
          step('e-6', tr('Step 6/6: Data-quality figure', '第 6/6 步：数据质量图'), '', resource('figure_gallery.json')),
          step('e-figures', tr('Audited the figures', '审核图件'), '', resource('figure_gallery.json')),
        ], { displayTitle: tr('Analysis complete', '分析已完成'), childJobId: 'reviewer-execution-run' }),
        { card: 'result', stop: true },
      ] },
      { id: 'interpretation', label: tr('Interpretation', '解读与追问'), beats: [
        message('reviewer-assistant-2', 'assistant', tr(
          '**Result**: of 94,458 adult ICU stays, 33,997 (35.99%) had the experimental SOFA-2 sepsis phenotype within 24 hours of ICU admission.\n\nIn-hospital death: 13.18% (4,480/33,997) with the phenotype and 8.25% (4,986/60,461) without it; 10.02% (9,466/94,458) overall.\n\nThis is an unadjusted descriptive comparison and does not show cause and effect. The phenotype is assessed over the first 24 hours, and deaths during that window are not handled separately, so read the difference with that in mind.',
          '**结果**：在 94,458 个成人 ICU 入住中，33,997 个（35.99%）在入 ICU 后 24 小时内出现实验性 SOFA-2 脓毒症表型。\n\n院内死亡：有表型组 13.18%（4,480/33,997），无表型组 8.25%（4,986/60,461）；总体 10.02%（9,466/94,458）。\n\n这是未调整的描述性比较，不能说明因果。表型在入 ICU 后 24 小时内判定，这段时间里的早期死亡没有单独处理，解读差异时要注意这一点。',
        ), [resource('result_tables.json'), resource('figure_gallery.json'), documents[0]]),
        { card: 'boundary' },
        timedActivity('reviewer-writing', [
          step('e-writer', tr('Manuscript stopped by the evidence check', '撰写论文：被证据检查拦下'), tr('Strict evidence enforcement blocked the manuscript, so no clinical paper was written.', '严格证据检查拦下了论文生成，没有写出临床论文。'), resource('manuscript_draft.json')),
          step('e-register', tr(`Registered ${registered} evidence records`, `登记 ${registered} 条证据`), tr('Every number in the answer opens to its result table and code.', '回答里的每个数字都能点开，追到结果表和代码。'), resource('evidence_ledger.json'), [resource('quality_gate.json'), resource('source_run_manifest.json')]),
          step('e-report', tr('Wrote the review report', '生成审阅报告'), tr('No clinical manuscript: the novelty search and independent scientific review are not done, a boundary kept by design.', '没有生成临床论文：新颖性检索和独立科学审阅尚未完成，这是按设计保留的边界。'), documents[0], documents.concat([resource('scientific_readiness.json'), resource('manuscript_draft.json')])),
        ], { displayTitle: tr('Run complete', '运行结束') }),
        { card: 'followups', stop: 'end' },
      ] },
    ];
  }
  // The conversation rows alone, without the replay's traceability stops.
  function messages(trace) {
    return chapters(trace || null).flatMap(chapter => chapter.beats).filter(beat => !beat.card);
  }

  // Next questions for this result; shown, not sent, in the read-only demo.
  function followUps() {
    return [
      tr('After adjusting for age and sex, is the phenotype still associated with in-hospital death?', '在调整年龄和性别后，这个表型与院内死亡的关联还成立吗？'),
      tr('With a 24-hour landmark that excludes deaths in the first 24 hours, how do the results change?', '改用入 ICU 后 24 小时 landmark、排除这段时间内死亡的患者后，结果会怎样？'),
      tr('Repeat the analysis with the standard Sepsis-3 (SOFA-1) definition and compare the two.', '用标准 Sepsis-3（SOFA-1）定义重复这项分析，比较两种定义。'),
      tr('Replicate this analysis in eICU.', '在 eICU 中复现这项分析。'),
    ];
  }

  function figureCaption() {
    return tr('Main figure, redrawn from the registered result table · A: share of stays by phenotype; B: in-hospital mortality by phenotype, dashed line = all stays', '主图（按登记的结果表重新绘制）· A：有无表型的入住占比；B：各组院内死亡率，虚线为全部入住');
  }
  function studyReportLink(esc) {
    return `<a class="gpi-link" href="${esc(tr(STUDY_REPORT_URL.en, STUDY_REPORT_URL.zh))}" target="_blank" rel="noopener">${esc(tr('Open the result report', '打开研究结果报告'))}</a>`;
  }
  function originalFigureLink(esc) {
    return `<a class="gpi-link" href="${esc(ORIGINAL_FIGURE_URL)}" target="_blank" rel="noopener">${esc(tr('Original run figure', '原始运行图'))}</a>`;
  }
  // The folded review and governance record under the answer.
  function governanceHtml(ctx) {
    const esc = ctx.esc;
    const button = typeof ctx.button === 'function' ? ctx.button : () => '';
    const run = standardRunResources();
    const resource = name => run.find(row => row.artifact === name);
    const documents = reviewResources();
    return `<details class="gpi-response-review gpi-demo-governance"><summary>${esc(tr('Review and governance record', '审阅与治理记录'))}</summary>
        <div class="gpi-demo-governance-body">
          <p><strong>${esc(tr('Done', '已完成'))}</strong>${esc(tr(': plan reviewed before analysis (six steps); 6/6 steps run; the numbers in the answer come from the result tables; the browser received aggregate tables only (no patient rows, paths or credentials).', '：分析前审阅计划（6 步）；6/6 步执行完成；回答里的数字都取自结果表；浏览器只收到聚合表（无患者行、路径或凭据）。'))}</p>
          <p><strong>${esc(tr('Kept open by design', '按设计保留'))}</strong>${esc(tr(': no novelty search and no independent scientific review, so no clinical manuscript was written. This does not change the analysis above.', '：没有做新颖性检索和独立科学审阅，所以没有生成临床论文；这不影响上面的分析结果。'))}</p>
          <p><strong>${esc(tr('Model use', '模型用量'))}</strong>${esc(tr(': 14 calls · 162,256 tokens · about $2.31', '：14 次调用 · 162,256 tokens · 约 $2.31'))}</p>
          <div class="gpi-demo-governance-links">${studyReportLink(esc)}${documents.map(doc => button(doc)).join('')}${button(resource('scientific_readiness.json'), tr('Scientific review', '科学审阅'))}</div>
        </div>
      </details>`;
  }
  /* The answer's figure and table, the follow-up questions, and the folded
     governance record. The figure is the run's registered result table drawn
     by the current renderer; the registered main figure (sha256 37a29be8…,
     byte-identical to the reviewer dossier's) is linked beside it. */
  function resultHtml(ctx) {
    const esc = ctx.esc;
    const button = typeof ctx.button === 'function' ? ctx.button : () => '';
    const run = standardRunResources();
    const resource = name => run.find(row => row.artifact === name);
    const rows = [
      [tr('With the phenotype', '有表型'), '33,997', '35.99%', '4,480', '13.18%'],
      [tr('Without the phenotype', '无表型'), '60,461', '64.01%', '4,986', '8.25%'],
      [tr('All stays', '全部入住'), '94,458', '100%', '9,466', '10.02%'],
    ];
    const headers = [tr('Group', '分组'), tr('ICU stays', 'ICU 入住'), tr('Share', '占比'), tr('Deaths', '死亡'), tr('In-hospital mortality', '院内死亡率')];
    return `<section class="gpi-demo-result" aria-label="${esc(tr('Demo result', '演示结果'))}">
      <figure class="gpi-run-answer-figure">
        <img src="${esc(FIGURE_URL)}" alt="${esc(tr('Phenotype share and in-hospital mortality by phenotype', '表型占比与各组院内死亡率'))}" loading="lazy" decoding="async">
        <figcaption>${esc(figureCaption())} · ${originalFigureLink(esc)} · ${button(resource('figure_gallery.json'), tr('All figures', '全部图表'))}</figcaption>
      </figure>
      <div class="gpi-demo-table"><table><thead><tr>${headers.map(label => `<th scope="col">${esc(label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr><th scope="row">${esc(row[0])}</th>${row.slice(1).map(cell => `<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
      ${ctx.figureOnly ? '' : `<div class="gpi-followups gpi-demo-followups"><div class="gpi-followups-head">${esc(tr('Follow-up questions', '可以继续问'))}</div><div class="gpi-followups-list">${followUps().map(text => `<div class="gpi-followup-row"><span class="gpi-followup-prompt" aria-disabled="true"><span aria-hidden="true">↳</span>${esc(text)}</span></div>`).join('')}</div></div>`}
      ${ctx.figureOnly ? '' : governanceHtml(ctx)}
    </section>`;
  }

  // The demo's key files for the side panel, in reading order.
  function shelfResources() {
    const run = standardRunResources();
    return ['result_tables.json', 'figure_gallery.json', 'agent_plan.json', 'literature_evidence.json', 'scientific_readiness.json']
      .map(name => run.find(row => row.artifact === name)).filter(Boolean)
      .concat(reviewResources());
  }

  function workflow() {
    return {
      // The locked manuscript is the current row, so the to-do list says why.
      kind: 'research_workflow_demo', current_stage: 'manuscript',
      completed_required_stages: 7, required_stage_count: 8,
      next_action_code: 'reviewer_demo_complete',
      stages: [
        ['question', 'complete', 'question_bound'],
        ['idea', 'complete', 'methods_references_kept'],
        ['setup', 'complete', 'prepared_data_contract_verified'],
        ['extraction', 'complete', 'aggregate_projection_verified'],
        ['plan', 'complete', 'exact_plan_reviewed'],
        ['analysis', 'complete', 'six_of_six_steps_complete'],
        ['interpretation', 'complete', 'descriptive_ceiling_preserved'],
        // No clinical manuscript: novelty search and independent review are open.
        ['manuscript', 'blocked', 'manuscript_withheld_by_design'],
      ].map(([id, status, reason_code]) => ({ id, status, reason_code })),
    };
  }
  function artifact(name) { return clone(artifacts()[String(name || '')] || null); }
  async function previewArtifact(name) {
    const key = String(name || '');
    if (!Object.prototype.hasOwnProperty.call(RUN_FILES, key)) return artifact(key);
    if (typeof fetch !== 'function') throw new Error(tr('The demo run file cannot be loaded here.', '此处无法加载演示运行文件。'));
    const response = await fetch(`${RUN_FILES_URL}${key}?v=${RUN_FILES_TOKEN}`, { credentials: 'same-origin' });
    if (!response.ok) throw new Error(tr('The demo run file is unavailable.', '演示运行文件不可用。'));
    return response.json();
  }
  function hasArtifact(name) { return Object.prototype.hasOwnProperty.call(artifacts(), String(name || '')); }
  function artifactLabel(name) { const item = artifacts()[String(name || '')]; return item ? item.title : String(name || ''); }

  window.EasyICU.guidedPi.declare('demo', {
    messages, chapters, workflow, artifact, previewArtifact, hasArtifact, artifactLabel,
    reviewResources, primaryDocument: () => artifactResource('figure_gallery.json', tr('Figures', '图表')),
    followUps, resultHtml, figureCaption, originalFigureLink, studyReportLink, FIGURE_URL, governanceHtml, shelfResources,
    standardResources: standardRunResources, sourceRunId: SOURCE_RUN_ID,
  });
})();
