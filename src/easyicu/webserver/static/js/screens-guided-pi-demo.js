/* Guided Pi reviewer-demo transcript.
   Owner: read-only reviewer fixtures and their safe structured preview.
   It never starts a provider job or mutates a real EasyICU project. */
(function () {
  'use strict';

  const SOURCE_RUN_ID = 'run_20260815T061842_5049c6';
  const WRAPPER_RUN_ID = 'e59d1a54feff';
  const SOURCE_AUTHORITY = 'bounded_reviewer_projection_from_registered_run';

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
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
  function literatureRecords() {
    return [
      {
        key: 'strobe_2007', year: '2007', venue: 'Annals of Internal Medicine', pmid: '17938396',
        title: 'The STROBE statement: guidelines for reporting observational studies.',
        relevance: tr('Supports explicit cohort, denominator, uncertainty, and limitation reporting.', '支持明确报告队列、分母、不确定性与局限。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/17938396/',
      },
      {
        key: 'record_2015', year: '2015', venue: 'PLOS Medicine', pmid: '26440803',
        title: 'The RECORD statement for studies using routinely collected health data.',
        relevance: tr('Supports source-data, code-list, and reproducibility reporting.', '支持来源数据、代码表与可复现性报告。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/26440803/',
      },
      {
        key: 'anderson_landmark_1983', year: '1983', venue: 'Journal of Clinical Oncology', pmid: '6668489',
        title: 'Analysis of survival by tumor response.',
        relevance: tr('Frames the temporal limitation of a first-24-hour phenotype.', '界定入 ICU 后 24 小时表型的时间学局限。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/6668489/',
      },
      {
        key: 'suissa_immortal_time_2008', year: '2008', venue: 'American Journal of Epidemiology', pmid: '18056625',
        title: 'Immortal time bias in pharmacoepidemiology.',
        relevance: tr('Supports keeping post-baseline exposure opportunity visible.', '支持显式保留基线后暴露机会问题。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/18056625/',
      },
      {
        key: 'singer_sepsis3_2016', year: '2016', venue: 'JAMA', pmid: '26903338',
        title: 'The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).',
        relevance: tr('Frames the source Sepsis-3 organ-dysfunction definition.', '界定来源 Sepsis-3 器官功能障碍定义。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
      },
      {
        key: 'durrleman_splines_1989', year: '1989', venue: 'Statistics in Medicine', pmid: '2657958',
        title: 'Flexible regression models with cubic splines.',
        relevance: tr('Documents flexible modelling of continuous covariates.', '记录连续协变量的灵活建模方法。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/2657958/',
      },
      {
        key: 'sterne_missing_data_2009', year: '2009', venue: 'BMJ', pmid: '19564179',
        title: 'Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls.',
        relevance: tr('Frames assumptions behind complete-case and missing-data handling.', '界定完整病例与缺失数据处理的假设。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/19564179/',
      },
      {
        key: 'ricu_2023', year: '2023', venue: 'GigaScience', pmid: '37318234',
        title: "ricu: R's interface to intensive care data.",
        relevance: tr("Conceptual ancestor of EasyICU's concept dictionary and table model.", 'EasyICU 概念字典与表模型的概念先驱。'),
        url: 'https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giad041/7198370',
      },
      {
        key: 'johnson_mimiciv_2023', year: '2023', venue: 'Scientific Data', pmid: '36596836',
        title: 'MIMIC-IV, a freely accessible electronic health record dataset.',
        relevance: tr('Primary source database used by this EasyICU run.', '本次 EasyICU 运行使用的主要来源数据库。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/36596836/',
      },
    ];
  }
  function artifacts() {
    const rows = {
      'reviewer_protocol.json': demoArtifact(
        'reviewer_protocol.json',
        tr('Prespecified reviewer demonstration protocol', '预先规定的审稿人演示协议'),
        tr('A real, read-only run is evaluated against explicit workflow criteria. Clinical novelty is intentionally outside this systems demonstration.', '使用真实只读运行按明确流程标准进行评估；临床新颖性明确不属于本次系统演示。'),
        [
          { label: tr('Prepared ICU stays', '准备后 ICU stays'), value: '94,458' },
          { label: tr('Analysis mode', '分析模式'), value: 'descriptive · counts only' },
          { label: tr('Reviewer criteria', '审稿标准'), value: '6 prespecified checks' },
          { label: tr('Patient rows in browser', '浏览器患者行'), value: '0' },
        ],
        [
          { heading: tr('System question', '系统问题'), items: [tr('Can the Agent preserve an exact approved plan, execute it, expose aggregate evidence, and stop unsupported authority escalation?', 'Agent 能否保留精确批准的计划、完成执行、展示聚合证据，并阻止无依据的权限升级？')] },
          { heading: tr('Pass criteria', '通过标准'), items: [tr('Typed plan is inspectable and digest-bound.', 'Typed Plan 可审阅且绑定摘要。'), tr('All required steps complete without plan drift.', '所有必需步骤完成且无计划漂移。'), tr('Tables, figures, run identity, Provider usage, and source bindings remain inspectable.', '表格、图件、运行身份、Provider 使用及来源绑定均可审阅。'), tr('No identifier column, patient row, credential, or host path reaches the browser.', '浏览器不接收标识列、患者行、凭据或宿主路径。'), tr('Unsupported manuscript authority is withheld.', '无依据的稿件权限被正确拒绝。'), tr('A self-contained reviewer dossier is produced.', '生成自包含审稿人报告。')] },
          { heading: tr('Authority boundary', '权限边界'), items: [tr('The reviewer demo may complete even when a clinical manuscript is withheld. These are separate outcomes.', '即使临床稿件被拒绝，审稿人 Demo 仍可完整通过；二者是不同结果。')] },
        ],
      ),
      'analysis_plan.json': demoArtifact(
        'analysis_plan.json',
        tr('Exact reviewed six-step plan', '精确审阅的六步计划'),
        tr('The approved plan is descriptive only: no p-values, confidence intervals, causal effects, or independence-sensitive inference.', '批准计划仅限描述：不计算 P 值、置信区间、因果效应或依赖独立性假设的推断。'),
        [
          { label: tr('Typed steps', 'Typed steps'), value: '6' },
          { label: tr('Primary analysis', '主要分析'), value: 'counts + proportions' },
          { label: tr('Variance mode', '方差模式'), value: 'none_counts_only' },
          { label: tr('Claim ceiling', '结论上限'), value: 'descriptive_only' },
        ],
        [{ heading: tr('Locked question', '锁定问题'), items: [tr('Estimate the first-24-hour experimental SOFA-2 phenotype prevalence and observed in-hospital mortality by phenotype status among adult ICU stays.', '在成人 ICU stays 中估计入 ICU 后 24 小时实验性 SOFA-2 表型比例，并按表型状态报告观察到的院内死亡。')] }],
        {
          projection_note: tr('The four displayed methods references are exact keys retained by the run. The source receipt records curated seeds rather than a completed novelty search.', '展示的 4 条方法学文献是该运行保留的精确 key；来源回执记录的是人工种子，而不是已完成的新颖性检索。'),
          steps: [
            { step_id: '01_define_analysis_cohort', method: 'cohort definition + attrition', intent: tr('Materialize adult ICU stays and preserve exact denominator accounting.', '生成成人 ICU stay 队列并保留精确分母账本。'), inputs: ['age ≥ 18', 'prepared ICU universe'], outputs: ['analysis_cohort', 'cohort_flow'], citation_keys: ['record_2015'] },
            { step_id: '02_missingness_and_measurement_audit', method: 'typed measurement audit', intent: tr('Separate measurement availability, binary event status, and conditional event-time applicability.', '区分测量可得性、二元事件状态与条件事件时间适用性。'), inputs: ['age', 'sex', 'sep3_sofa2', 'death', 'death_time'], outputs: ['missingness_data_quality'], citation_keys: ['record_2015'] },
            { step_id: '03_exposure_outcome_distribution', method: 'descriptive counts only', intent: tr('Report phenotype prevalence and observed mortality with exact denominators.', '按精确分母报告表型比例及观察到的死亡。'), inputs: ['analysis_cohort', 'sep3_sofa2_max', 'death'], outputs: ['exposure_outcome_distribution'], citation_keys: ['strobe_2007', 'record_2015', 'anderson_landmark_1983', 'suissa_immortal_time_2008'] },
            { step_id: '04_visualize_exposure_outcome_distribution', method: 'deterministic rendering', intent: tr('Render registered counts and proportions without re-analysis.', '不重新分析，仅绘制已登记计数和比例。'), inputs: ['exposure_outcome_distribution'], outputs: ['phenotype_mortality_figure'], citation_keys: ['strobe_2007'] },
            { step_id: '05_visualize_cohort_accounting', method: 'deterministic rendering', intent: tr('Render the cohort denominator and eligibility accounting.', '绘制队列分母与纳入账本。'), inputs: ['cohort_flow'], outputs: ['cohort_flow_figure'], citation_keys: ['record_2015'] },
            { step_id: '06_visualize_data_quality', method: 'applicability-aware rendering', intent: tr('Render true missingness separately from not-applicable conditional event times.', '将真实缺失与不适用的条件事件时间分开绘制。'), inputs: ['missingness_data_quality'], outputs: ['data_quality_figure'], citation_keys: ['record_2015'] },
          ],
          citations: literatureRecords(),
        },
      ),
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
      'applicability_audit.json': demoArtifact(
        'applicability_audit.json',
        tr('Applicability-aware data-quality audit', '适用性敏感的数据质量审计'),
        tr('The audit prevents event prevalence from being mislabelled as measurement coverage.', '该审计避免把事件比例误标为测量覆盖率。'),
        [
          { label: tr('Death status available', '死亡状态可得'), value: '94,458 / 94,458' },
          { label: tr('Death-time applicable', '死亡时间适用'), value: '9,466' },
          { label: tr('Missing among applicable', '适用者中缺失'), value: '0 / 9,466' },
          { label: tr('Not applicable', '不适用'), value: '84,992' },
        ],
        [
          { heading: tr('Semantic correction', '语义修正'), items: [tr('10.02% is the death-event prevalence and therefore the share for which death_time is applicable. It is not a death-time measurement rate.', '10.02% 是死亡事件比例，因此也是 death_time 的适用比例；它不是死亡时间测量率。'), tr('Twenty-eight death times precede the ICU origin and remain a separate timing-protocol flag, not missingness.', '28 个死亡时间早于 ICU origin，作为独立时间协议标记保留，不计为缺失。')] },
        ],
        {
          tables: [{
            label: tr('Typed observation semantics', 'Typed observation semantics'),
            headers: [tr('Variable', '变量'), tr('Semantic type', '语义类型'), tr('Applicable', '适用'), tr('Missing in applicable', '适用者中缺失'), tr('Not applicable', '不适用')],
            rows: [
              ['age', 'measurement_availability', '94,458', '0', '0'],
              ['sex', 'measurement_availability', '94,458', '0', '0'],
              ['sep3_sofa2', 'binary_event_presence', '94,458', '0', '0'],
              ['death', 'measurement_availability', '94,458', '0', '0'],
              ['death_time', 'conditional_event_time', '9,466', '0', '84,992'],
            ],
          }],
        },
      ),
      'execution_receipt.json': demoArtifact(
        'execution_receipt.json',
        tr('Execution, provenance, and privacy receipt', '执行、来源与隐私回执'),
        tr('One exact reviewed plan completed and produced a bounded, inspectable browser projection.', '一份精确审阅计划完成执行，并生成有界、可审阅的浏览器投影。'),
        [
          { label: tr('Execution', '执行'), value: '6 / 6 steps' },
          { label: tr('Registered evidence', '已登记证据'), value: '125 records' },
          { label: tr('Review surfaces', '审阅界面'), value: '12 tables / 3 figures' },
          { label: tr('Provider usage', 'Provider 使用'), value: '14 calls / 162,256 tokens' },
          { label: tr('Estimated cost', '估算成本'), value: '$2.30776' },
          { label: tr('Source bindings', '来源绑定'), value: '11 SHA-256 bindings' },
        ],
        [
          { heading: tr('Privacy boundary', '隐私边界'), items: [tr('Aggregate tables only; zero patient rows, identifier columns, host paths, or credentials in the browser projection.', '浏览器投影仅含聚合表；患者行、标识列、宿主路径和凭据均为 0。')] },
          { heading: tr('Reproducibility boundary', '可复现性边界'), items: [tr('The report, HTML, PDF, corrected figure source, evidence ledger, and private review/Provider receipts are digest-bound.', '报告、HTML、PDF、修正图源、证据账本及私有审阅/Provider 回执均绑定摘要。')] },
        ],
        {
          tables: [{
            label: tr('Reviewer workflow outcome', '审稿人流程结果'),
            headers: [tr('Boundary', '边界'), tr('Outcome', '结果'), tr('Meaning', '含义')],
            rows: [
              [tr('Typed plan', 'Typed Plan'), tr('Verified', '已核验'), tr('Six exact steps', '精确六步')],
              [tr('Development review', '开发审阅'), tr('Verified', '已核验'), tr('Exact-plan approval', '精确计划批准')],
              [tr('Execution', '执行'), tr('Verified', '已核验'), tr('6/6 complete', '6/6 完成')],
              [tr('Browser projection', '浏览器投影'), tr('Verified', '已核验'), tr('Aggregate-only privacy pass', '仅聚合隐私检查通过')],
              [tr('Clinical manuscript', '临床稿件'), tr('Withheld as designed', '按设计拒绝'), tr('STRICT authority gate', 'STRICT 权限闸门')],
              [tr('Reviewer dossier', '审稿人报告'), tr('Complete', '完整'), tr('HTML + six-page PDF', 'HTML + 6 页 PDF')],
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
    rows['run_context.json'] = demoArtifact(
      'run_context.json', tr('Run context', '运行上下文'),
      tr('Path-free identity and scientific scope derived from the registered source run.', '从登记 source run 派生的无路径身份与科学范围。'),
      [
        { label: tr('Pipeline run', 'Pipeline run'), value: SOURCE_RUN_ID },
        { label: tr('Wrapper job', 'Wrapper job'), value: WRAPPER_RUN_ID },
        { label: tr('Analysis family', '分析家族'), value: 'descriptive_epidemiology' },
        { label: tr('Claim ceiling', '结论上限'), value: 'descriptive_only' },
      ],
      [{ heading: tr('Bound question', '绑定问题'), items: [tr('Estimate the experimental first-24-hour SOFA-2 phenotype prevalence and observed in-hospital mortality with exact denominators.', '按精确分母估计实验性入 ICU 后 24 小时 SOFA-2 表型比例及观察到的院内死亡。')] }],
      {
        run_id: SOURCE_RUN_ID, study_id: 'e1-luna-canary-20260814-a56657b', run_type: 'full',
        mode: 'research_agent_pipeline', database_scope: 'miiv', cohort_size: 94458,
        local_first: { uploads: 0 },
        question: tr('Estimate the experimental first-24-hour SOFA-2 phenotype prevalence and observed in-hospital mortality using exact denominators.', '按精确分母估计实验性入 ICU 后 24 小时 SOFA-2 表型比例及观察到的院内死亡。'),
      },
    );
    rows['cohort_summary.json'] = demoArtifact(
      'cohort_summary.json', tr('Cohort summary', '队列摘要'),
      tr('The host-materialized adult ICU-stay universe and exact attrition accounting.', 'Host 生成的成人 ICU stay 分析全集与精确队列账本。'),
      [
        { label: tr('Prepared universe', '准备后全集'), value: '94,458 ICU stays' },
        { label: tr('Adult criterion', '成人标准'), value: 'age ≥ 18' },
        { label: tr('Excluded', '排除'), value: '0' },
        { label: tr('Analysis cohort', '分析队列'), value: '94,458 ICU stays' },
      ],
      [{ heading: tr('Cohort contract', '队列合同'), items: [tr('One prepared row per ICU stay; no patient-level independence claim is made.', '每个 ICU stay 一条准备后记录；不主张患者层独立性。')] }],
      {
        run_id: SOURCE_RUN_ID, status: 'complete', database_scope: 'miiv', cohort_size: 94458,
        analysis_unit: 'ICU stay', included: 94458, excluded: 0,
        criteria: [tr('Adult ICU stays', '成人 ICU stays'), 'age >= 18'],
      },
    );
    rows['quality_gate.json'] = Object.assign(clone(rows['authority_verdict.json']), {
      artifact: 'quality_gate.json',
      title: tr('Evidence verification', '证据核验'),
      summary: tr('Execution completed, while manuscript and publication authority were withheld by separate checks.', '执行已完成；稿件与发表权限由独立检查按设计拒绝。'),
      gate: {
        status: 'blocked', reportable: false, draft_unlocked: false,
        reason: 'research_agent_pipeline_failed_closed',
        checks: [
          { id: 'execution_complete', passed: true, status: 'passed', evidence: '6 / 6 steps', reason: '' },
          { id: 'analysis_validated', passed: false, status: 'failed', evidence: '', reason: 'analysis_validated_not_satisfied' },
          { id: 'evidence_complete', passed: false, status: 'failed', evidence: '', reason: 'evidence_complete_not_satisfied' },
          { id: 'numeric_verified', passed: false, status: 'failed', evidence: '', reason: 'numeric_verified_not_satisfied' },
          { id: 'manuscript_ready', passed: false, status: 'failed', evidence: '', reason: 'manuscript_ready_not_satisfied' },
          { id: 'publication_ready', passed: false, status: 'failed', evidence: '', reason: 'publication_ready_not_satisfied' },
          { id: 'paper_authorized', passed: false, status: 'failed', evidence: '', reason: 'paper_authorized_not_satisfied' },
        ],
      },
    });
    rows['agent_plan.json'] = Object.assign(clone(rows['analysis_plan.json']), {
      artifact: 'agent_plan.json', title: tr('Research plan', '研究计划'),
    });
    rows['literature_evidence.json'] = demoArtifact(
      'literature_evidence.json', tr('Literature evidence', '文献证据'),
      tr('Exact retained methodology keys are inspectable; the source receipt honestly records that no live novelty retrieval completed.', '可审阅精确保留的方法学 key；来源回执如实记录未完成实时新颖性检索。'),
      [
        { label: tr('Retained curated records', '保留人工文献'), value: '9' },
        { label: tr('Live retrieval completed', '实时检索完成'), value: tr('No', '否') },
        { label: tr('Plan mapping', '计划映射'), value: 'complete' },
        { label: tr('Novelty authority', '新颖性权限'), value: 'not established' },
      ],
      [{ heading: tr('Search receipt', '检索回执'), items: ['search_conducted=false', 'sources_enabled=[]', tr('Curated methods evidence cannot establish novelty.', '人工方法学证据不能建立新颖性。')] }],
      {
        citations: literatureRecords(),
        search: {
          search_conducted: false, sources_returning: [],
          note: tr('The run retained curated methodology records; no live novelty search completed.', '该运行保留人工方法学文献；未完成实时新颖性检索。'),
        },
        evidence_boundary: tr('Literature supports design rationale; patient and result evidence remain separately governed.', '文献支持设计依据；患者与结果证据由独立链路治理。'),
        step_citation_map: clone(rows['analysis_plan.json'].steps).map(step => ({
          step_id: step.step_id, intent: step.intent,
          planned_analysis_role: step.step_id.startsWith('0') && Number(step.step_id.slice(0, 2)) > 3 ? 'auxiliary' : 'scientific',
          citation_keys: step.citation_keys,
        })),
      },
    );
    rows['scientific_plan_review.json'] = demoArtifact(
      'scientific_plan_review.json', tr('Scientific plan review', '科学计划审阅'),
      tr('The six-step counts-only plan passed development review for execution, not publication review.', '六步仅计数计划通过开发执行审阅，但不是投稿审阅。'),
      [
        { label: tr('Plan contract', '计划合同'), value: 'passed' },
        { label: tr('Typed steps', 'Typed steps'), value: '6' },
        { label: tr('Execution approval', '执行批准'), value: 'development only' },
        { label: tr('Publication review', '投稿审阅'), value: 'not granted' },
      ],
      [{ heading: tr('Open scientific limits', '未闭合科学限制'), items: [tr('Post-baseline exposure opportunity remains unresolved.', '基线后暴露机会仍未闭合。'), tr('Independent novelty and scientific review are unavailable.', '缺少独立新颖性与科学审阅。')] }],
      {
        review_scope: 'pre_execution_plan', rendered_outputs_assessed: false,
        dimension_scores: { literature: 40, novelty: 70, literature_to_plan: 100, icu_clinical_design: 100, statistical_design: 100, robustness: 70, figures: 100, content_completeness: 100 },
        findings: [
          { severity: 'major', remediation_route: 'literature', code: 'LITERATURE_RETRIEVAL_NOT_CONDUCTED', message: tr('Curated seeds do not establish current novelty.', '人工种子不能建立当前新颖性。'), remediation: tr('Run a dated, inspectable retrieval and independent review.', '执行带日期、可检查的检索与独立审阅。') },
          { severity: 'major', remediation_route: 'study_design', code: 'POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED', message: tr('The first-24-hour phenotype is post-baseline.', '入 ICU 后 24 小时表型属于基线后暴露。'), remediation: tr('Keep the current run descriptive or create a new landmark design.', '保持当前运行仅描述，或创建新的 landmark 设计。') },
        ],
        facts: { score_interpretation: { figures: tr('Planned roles only; rendered visual quality was assessed after execution.', '仅规划角色；渲染后再评估视觉质量。'), content_completeness: tr('Planned article-role coverage only.', '仅表示计划文章角色覆盖。') } },
      },
    );
    rows['scientific_readiness.json'] = Object.assign(clone(rows['authority_verdict.json']), {
      artifact: 'scientific_readiness.json', title: tr('Scientific review', '科学审阅'),
      summary: tr('Engineering validation is complete; clinical and publication readiness remain separately withheld.', '工程验证已完成；临床与投稿就绪度仍由独立边界拒绝。'),
      metrics: [], sections: [], claim_ceiling: 'unsupported',
      domains: [
        { domain: 'idea', status: 'not_assessed', summary: tr('Technical execution does not establish novelty.', '技术执行不能建立新颖性。') },
        { domain: 'literature', status: 'blocked', summary: tr('No live novelty retrieval completed.', '未完成实时新颖性检索。') },
        { domain: 'data', status: 'review_required', summary: tr('Prepared-data provenance exists; publication population scope remains open.', '准备后数据来源存在；投稿人群范围仍未闭合。') },
        { domain: 'analysis', status: 'analysis_only', summary: tr('Six descriptive steps completed without inferential authority.', '六个描述性步骤完成，但无推断权限。') },
        { domain: 'manuscript', status: 'blocked', summary: tr('STRICT evidence binding withheld the formal draft.', 'STRICT 证据绑定拒绝正式稿件。') },
      ],
    });
    rows['manuscript_draft.json'] = demoArtifact(
      'manuscript_draft.json', tr('Locked manuscript draft', '锁定论文草稿'),
      tr('STRICT evidence enforcement stopped before a formal manuscript could be authorized.', 'STRICT 证据执行在正式稿件获得授权前停止。'),
      [
        { label: tr('Status', '状态'), value: 'withheld_as_designed' },
        { label: tr('Formal manuscript', '正式稿件'), value: tr('Not generated', '未生成') },
        { label: tr('Authorized sentences', '授权句子'), value: '0' },
        { label: tr('Publication authority', '发表权限'), value: 'false' },
      ],
      [{ heading: tr('Deterministic reason', '确定性原因'), items: ['STRICT evidence mode: manuscript prose lacks deterministic evidence or scientific claim authority.', tr('This is a safety result, not missing execution output.', '这是安全结果，不是分析执行产物缺失。')] }],
      {
        run_id: SOURCE_RUN_ID, status: 'locked_pending_human_review',
        claims: [{ id: 'claim_000', text: tr('Formal manuscript was not generated.', '未生成正式稿件。'), evidence_ids: [], status: 'diagnostic_only' }],
      },
    );
    rows['figure_gallery.json'] = demoArtifact(
      'figure_gallery.json', tr('Figure gallery', '图件画廊'),
      tr('Three source-bound supporting figures are available; no primary publication figure bundle is claimed.', '3 张来源绑定支持图可用；不声称存在主投稿图件包。'),
      [
        { label: tr('Supporting figures', '支持图'), value: '3' },
        { label: tr('Primary publication figures', '主投稿图'), value: '0' },
        { label: tr('Embedded images', '嵌入图像'), value: '3' },
        { label: tr('Semantic corrections', '语义修正'), value: '1' },
      ],
      [{ heading: tr('Registered figures', '登记图件'), items: [tr('Phenotype prevalence and observed mortality.', '表型比例与观察死亡。'), tr('Adult ICU cohort accounting.', '成人 ICU 队列账本。'), tr('Applicability-aware data quality.', '适用性敏感的数据质量。')] }],
      {
        schema_version: 'easyicu.web-pipeline-figure-gallery/1', kind: 'figure_gallery',
        status: 'no_primary_publication_figure', embedded_count: 3, primary_count: 0, supporting_count: 3,
        figures: [
          { label: tr('Phenotype prevalence and observed in-hospital mortality', '表型比例与观察院内死亡'), name: 'sep3_sofa2_mortality_distribution.png', relative_path: 'steps/04_visualize_exposure_outcome_distribution/outputs/sep3_sofa2_mortality_distribution.png', status: 'supporting' },
          { label: tr('Adult ICU cohort accounting', '成人 ICU 队列账本'), name: 'cohort_flow.png', relative_path: 'steps/05_visualize_cohort_accounting/outputs/cohort_flow.png', status: 'supporting' },
          { label: tr('Applicability-aware data quality', '适用性敏感的数据质量'), name: 'data_quality.png', relative_path: 'steps/06_visualize_data_quality/outputs/data_quality.png', status: 'supporting' },
        ],
      },
    );
    rows['result_tables.json'] = Object.assign(clone(rows['descriptive_results.json']), {
      artifact: 'result_tables.json', title: tr('Research result tables', '科研结果表'),
    });
    rows['source_run_manifest.json'] = Object.assign(clone(rows['execution_receipt.json']), {
      artifact: 'source_run_manifest.json', title: tr('Source run manifest', '原始运行清单'),
      run_id: SOURCE_RUN_ID, status: 'blocked', evidence_count: 125, figure_count: 3,
      result_table_count: 12, system_validation_report_available: true,
      provider: { provider: 'openai', model: 'gpt-5.6-luna', provider_gate: 'research_agent_provider_ready' },
      readiness: { execution_complete: true, failed_steps: [], missing_steps: [], manuscript_generated: false, paper_authorized: false, publication_ready: false },
    });
    rows['evidence_ledger.json'] = Object.assign(clone(rows['execution_receipt.json']), {
      artifact: 'evidence_ledger.json', title: tr('Evidence ledger', '证据账本'),
      summary: tr('Digest-bound inventory of the browser-safe run projection and registered reviewer documents.', '浏览器安全运行投影与登记审稿文档的摘要绑定清单。'),
      artifacts: [
        ['run_context.json', 'eb96c56e38ddcda5e4781226bc068654dd82753e5c58d8e63efed01450e16695'],
        ['cohort_summary.json', '6569849a2e0a8f27f0246066f4cb0a42820d209b5db0422ec712fc6fbce64e40'],
        ['quality_gate.json', '01d1be6ffa605c4e44f04d66ee63331466c2f1e120f8460dffa26fe11f2133d7'],
        ['agent_plan.json', '04796b6430c1e75b22ee4d73826f14869bd7d2e85eb812be0b8854775863e44e'],
        ['literature_evidence.json', 'bc5c6dffdf90ba37b7265da8f35110f96c509f6cdcf8777f432af56dca132760'],
        ['scientific_readiness.json', '9707075317783a2c943364197ada7515f5e18be0e2f8c152d7ecab47c0336c85'],
        ['manuscript_draft.json', '5c41b834bf45b364c600111838daf364abb44739ab9a9743ef0da86850490913'],
        ['result_tables.json', '16c14d7f8d456eb5334df6df9fef59f028fd8d303dbf31cb198fe75e62372089'],
      ].map(([name, sha256]) => ({ name, sha256, kind: 'json', media_type: 'application/json' })),
    });
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

  // A demo trace row: named for a reader and kept as its own row. The fixture
  // records only each activity's total time, so rows carry no per-step time.
  function traceStep(id, label, text, resource, resources) {
    return { id, kind: 'pipeline', step: 'plan_step', status: 'complete', label, text: text || '', resource: resource || null, resources: resources || [], durationKnown: false };
  }

  /* The walkthrough reads as one ordinary research conversation: a clinical
     question, the plan to confirm, the answer in the run's own numbers, and
     the questions to ask next. The governance record stays one click away. */
  function messages() {
    const documents = reviewResources();
    const run = standardRunResources();
    const resource = name => run.find(row => row.artifact === name);
    return [
      message('reviewer-user-1', 'user', tr(
        'In MIMIC-IV adult ICU stays, how common is the experimental SOFA-2 sepsis phenotype within the first 24 hours of admission, and how does in-hospital mortality differ with and without it?',
        '在 MIMIC-IV 的成人 ICU 入住中，入 ICU 后 24 小时内出现实验性 SOFA-2 脓毒症表型的比例有多高？有和没有这个表型的患者，院内死亡率差多少？',
      )),
      activity('reviewer-planning', 1000, 194000, [
        traceStep('p-scope', tr('Checked the question and the data scope', '核对研究问题与数据范围'), tr('MIMIC-IV and adult ICU stays confirmed.', '已确认使用 MIMIC-IV 与成人 ICU 入住。'), resource('run_context.json')),
        traceStep('p-cohort', tr('Analysis population: 94,458 adult ICU stays', '分析人群：94,458 个成人 ICU 入住'), tr('Age ≥ 18; no stay was excluded.', '年龄 ≥ 18 岁，没有入住被排除。'), resource('cohort_summary.json')),
        traceStep('p-literature', tr('Collected the methods references', '整理方法学依据'), tr('Nine methods references kept (STROBE, RECORD, Sepsis-3 and others); no novelty search was run.', '保留 9 篇方法学文献（STROBE、RECORD、Sepsis-3 等）；这次没有做新颖性检索。'), resource('literature_evidence.json')),
        traceStep('p-draft-1', tr('Plan draft 1 failed the scientific checks and was revised', '计划草案第 1 版未通过科学检查，已自动修订'), tr('A rejected draft never becomes the plan.', '被否决的草案不会成为计划。')),
        traceStep('p-draft-2', tr('Plan draft 2 passed: six steps', '计划草案第 2 版通过检查：共 6 个步骤'), '', resource('agent_plan.json')),
        traceStep('p-review', tr('Scientific review of the plan', '计划科学审阅'), tr('Ready to run; the missing novelty search and independent review are kept as limitations.', '可以执行；缺少的新颖性检索和独立审阅记为局限。'), resource('scientific_plan_review.json')),
      ], { displayTitle: tr('Plan ready for your review', '计划已就绪，等待你确认'), childJobId: 'reviewer-plan-run' }),
      message('reviewer-assistant-1', 'assistant', tr(
        'I drafted a six-step descriptive plan. Please confirm it before the analysis starts:\n\n- **Population**: all adult ICU stays in MIMIC-IV (94,458)\n- **Exposure**: the experimental SOFA-2 sepsis phenotype within 24 hours of ICU admission (yes / no)\n- **Outcome**: in-hospital death\n- **Method**: counts and proportions by phenotype; no adjustment and no significance test\n- **Data quality**: a separate audit of missing values that keeps not-applicable death times apart from true missingness\n\nThe complete plan and its methods references are below.',
        '我拟定了一份 6 步的描述性分析计划，开始分析前请你确认：\n\n- **人群**：MIMIC-IV 中全部成人 ICU 入住（94,458 个）\n- **暴露**：入 ICU 后 24 小时内是否出现实验性 SOFA-2 脓毒症表型\n- **结局**：院内死亡\n- **方法**：按有无表型计数并计算比例，不做调整，不做显著性检验\n- **数据质量**：单独审计缺失值，并把“不适用”的死亡时间和真正的缺失分开\n\n完整计划和方法学依据可以从下方打开。',
      ), [resource('agent_plan.json'), resource('literature_evidence.json'), resource('scientific_plan_review.json')]),
      message('reviewer-user-2', 'user', tr('Approve this plan and start the analysis.', '批准这份计划，开始分析。')),
      activity('reviewer-execution', 200000, 578000, [
        traceStep('e-1', tr('Step 1/6: Cohort and denominators', '第 1/6 步：队列与分母核算'), tr('All 94,458 stays retained.', '94,458 个入住全部保留。'), resource('cohort_summary.json')),
        traceStep('e-2', tr('Step 2/6: Measurement and missingness audit', '第 2/6 步：测量与缺失审计'), tr('Death time applies only to the 9,466 deaths; 0 of them missing.', '死亡时间只对 9,466 例死亡适用，其中缺失 0 例。'), artifactResource('applicability_audit.json', tr('Applicability audit', '适用性审计'))),
        traceStep('e-3', tr('Step 3/6: Group proportions and outcomes', '第 3/6 步：分组比例与结局'), tr('Counts and proportions only; no inferential estimate added.', '只计数和比例，不新增推断估计。'), resource('result_tables.json')),
        traceStep('e-4', tr('Step 4/6: Result figure', '第 4/6 步：结果图'), '', resource('figure_gallery.json')),
        traceStep('e-5', tr('Step 5/6: Cohort flow figure', '第 5/6 步：队列流程图'), '', resource('figure_gallery.json')),
        traceStep('e-6', tr('Step 6/6: Data-quality figure', '第 6/6 步：数据质量图'), '', resource('figure_gallery.json')),
        traceStep('e-verify', tr('Checked every number against the result tables', '核对数字与分母'), tr('125 evidence records registered; each reported number traces to a result table.', '登记 125 条证据，报告中的数字都能对回结果表。'), resource('evidence_ledger.json'), [resource('quality_gate.json'), resource('source_run_manifest.json')]),
        traceStep('e-report', tr('Wrote the review report', '生成审阅报告'), tr('No clinical manuscript: the novelty search and independent scientific review are not done, a boundary kept by design.', '没有生成临床论文：新颖性检索和独立科学审阅尚未完成，这是按设计保留的边界。'), documents[0], documents.concat([resource('scientific_readiness.json'), resource('manuscript_draft.json')])),
      ], { displayTitle: tr('Analysis complete', '分析已完成'), childJobId: 'reviewer-execution-run' }),
      message('reviewer-assistant-2', 'assistant', tr(
        '**Result**: of 94,458 adult ICU stays, 33,997 (35.99%) had the experimental SOFA-2 sepsis phenotype within 24 hours of ICU admission.\n\nIn-hospital death: 13.18% (4,480/33,997) with the phenotype and 8.25% (4,986/60,461) without it; 10.02% (9,466/94,458) overall.\n\nThis is an unadjusted descriptive comparison and does not show cause and effect. The phenotype is assessed over the first 24 hours, and deaths during that window are not handled separately, so read the difference with that in mind.',
        '**结果**：在 94,458 个成人 ICU 入住中，33,997 个（35.99%）在入 ICU 后 24 小时内出现实验性 SOFA-2 脓毒症表型。\n\n院内死亡：有表型组 13.18%（4,480/33,997），无表型组 8.25%（4,986/60,461）；总体 10.02%（9,466/94,458）。\n\n这是未调整的描述性比较，不能说明因果。表型在入 ICU 后 24 小时内判定，这段时间里的早期死亡没有单独处理，解读差异时要注意这一点。',
      ), [resource('result_tables.json'), resource('figure_gallery.json'), documents[0]]),
    ];
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

  /* The answer's figure and table, the follow-up questions, and the folded
     governance record. The figure is the run's registered main figure,
     extracted unchanged from the reviewer dossier (sha256 37a29be8…). */
  function resultHtml(ctx) {
    const esc = ctx.esc;
    const button = typeof ctx.button === 'function' ? ctx.button : () => '';
    const run = standardRunResources();
    const resource = name => run.find(row => row.artifact === name);
    const documents = reviewResources();
    const rows = [
      [tr('With the phenotype', '有表型'), '33,997', '35.99%', '4,480', '13.18%'],
      [tr('Without the phenotype', '无表型'), '60,461', '64.01%', '4,986', '8.25%'],
      [tr('All stays', '全部入住'), '94,458', '100%', '9,466', '10.02%'],
    ];
    const headers = [tr('Group', '分组'), tr('ICU stays', 'ICU 入住'), tr('Share', '占比'), tr('Deaths', '死亡'), tr('In-hospital mortality', '院内死亡率')];
    return `<section class="gpi-demo-result" aria-label="${esc(tr('Demo result', '演示结果'))}">
      <figure class="gpi-run-answer-figure">
        <img src="assets/demo/sofa2-phenotype-mortality.png?v=20260922-demo2" alt="${esc(tr('Phenotype share and in-hospital mortality by phenotype', '表型占比与各组院内死亡率'))}" loading="lazy" decoding="async">
        <figcaption>${esc(tr('Main figure · A: share of stays by phenotype (0 = no, 1 = yes); B: in-hospital mortality by phenotype', '主图 · A：有无表型的入住占比（0 = 无，1 = 有）；B：各组院内死亡率'))} · ${button(resource('figure_gallery.json'), tr('All figures', '全部图表'))}</figcaption>
      </figure>
      <div class="gpi-demo-table"><table><thead><tr>${headers.map(label => `<th scope="col">${esc(label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr><th scope="row">${esc(row[0])}</th>${row.slice(1).map(cell => `<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
      <div class="gpi-followups gpi-demo-followups"><div class="gpi-followups-head">${esc(tr('Follow-up questions', '可以继续问'))}</div><div class="gpi-followups-list">${followUps().map(text => `<div class="gpi-followup-row"><span class="gpi-followup-prompt" aria-disabled="true"><span aria-hidden="true">↳</span>${esc(text)}</span></div>`).join('')}</div></div>
      <details class="gpi-response-review gpi-demo-governance"><summary>${esc(tr('Review and governance record', '审阅与治理记录'))}</summary>
        <div class="gpi-demo-governance-body">
          <p><strong>${esc(tr('Done', '已完成'))}</strong>${esc(tr(': plan reviewed before analysis (six steps); 6/6 steps run; every number checked against the result tables; the browser received aggregate tables only (no patient rows, paths or credentials).', '：分析前审阅计划（6 步）；6/6 步执行完成；所有数字对回结果表；浏览器只收到聚合表（无患者行、路径或凭据）。'))}</p>
          <p><strong>${esc(tr('Kept open by design', '按设计保留'))}</strong>${esc(tr(': no novelty search and no independent scientific review, so no clinical manuscript was written. This does not change the analysis above.', '：没有做新颖性检索和独立科学审阅，所以没有生成临床论文；这不影响上面的分析结果。'))}</p>
          <p><strong>${esc(tr('Model use', '模型用量'))}</strong>${esc(tr(': 14 calls · 162,256 tokens · about $2.31', '：14 次调用 · 162,256 tokens · 约 $2.31'))}</p>
          <div class="gpi-demo-governance-links">${documents.map(doc => button(doc)).join('')}${button(resource('scientific_readiness.json'), tr('Scientific review', '科学审阅'))}</div>
        </div>
      </details>
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
    const item = artifact(name);
    if (!item || String(name || '') !== 'figure_gallery.json' || typeof fetch !== 'function') return item;
    try {
      const response = await fetch('/assets/demo/system-validation-report.html?v=20260815-reviewer-demo1', { credentials: 'same-origin' });
      if (!response.ok) return item;
      const html = await response.text();
      const images = [];
      const pattern = /\bsrc=(["'])(data:image\/png;base64,[A-Za-z0-9+/=]+)\1/gi;
      let match;
      while ((match = pattern.exec(html)) && images.length < 3) images.push(match[2]);
      if (images.length !== 3) return item;
      item.figures.forEach((figure, index) => { figure.data_url = images[index]; });
    } catch (_) {
      // The metadata table remains usable if the registered dossier is unavailable.
    }
    return item;
  }
  function hasArtifact(name) { return Object.prototype.hasOwnProperty.call(artifacts(), String(name || '')); }
  function artifactLabel(name) { const item = artifacts()[String(name || '')]; return item ? item.title : String(name || ''); }

  window.EasyICU.guidedPi.declare('demo', {
    messages, workflow, artifact, previewArtifact, hasArtifact, artifactLabel,
    reviewResources, primaryDocument: () => artifactResource('figure_gallery.json', tr('Figures', '图表')),
    followUps, resultHtml, shelfResources, sourceRunId: SOURCE_RUN_ID,
  });
})();
