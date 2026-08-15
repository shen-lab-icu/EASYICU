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
        title: 'Analysis of survival by tumor response and other time-dependent outcome comparisons.',
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
        key: 'ricu_2023', year: '2023', venue: 'Software', pmid: '',
        title: "ricu: R's interface to intensive care data.",
        relevance: tr("Conceptual ancestor of EasyICU's concept dictionary and table model.", 'EasyICU 概念字典与表模型的概念先驱。'),
        url: 'https://github.com/eth-mds/ricu',
      },
      {
        key: 'johnson_mimiciv_2023', year: '2023', venue: 'Scientific Data', pmid: '',
        title: 'MIMIC-IV, a freely accessible electronic health record dataset.',
        relevance: tr('Primary source database used by this EasyICU run.', '本次 EasyICU 运行使用的主要来源数据库。'),
        url: '',
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
          { label: tr('Phenotype present', '表型阳性'), value: '33,997 / 94,458 (35.991658%)' },
          { label: tr('Observed deaths · absent', '观察死亡 · 阴性'), value: '4,986 / 60,461 (8.246638%)' },
          { label: tr('Observed deaths · present', '观察死亡 · 阳性'), value: '4,480 / 33,997 (13.177633%)' },
        ],
        [
          { heading: tr('Interpretation ceiling', '解读上限'), items: [tr('These are observed descriptive proportions, not causal effects or ordinary baseline-exposure associations.', '这些是观察到的描述性比例，不是因果效应或普通基线暴露关联。'), tr('The first-24-hour ascertainment period leaves exposure-opportunity and early-event timing unresolved.', '入 ICU 后 24 小时判定窗口仍存在暴露机会与早期事件时间未闭合问题。')] },
        ],
        {
          tables: [{
            label: tr('Counts-only phenotype and mortality distribution', '仅计数的表型与死亡分布'),
            headers: [tr('Phenotype', '表型'), tr('ICU stays', 'ICU stays'), tr('Cohort share', '队列占比'), tr('Deaths', '死亡'), tr('Observed mortality', '观察死亡率')],
            rows: [
              [tr('Absent', '阴性'), '60,461', '64.008342%', '4,986', '8.246638%'],
              [tr('Present', '阳性'), '33,997', '35.991658%', '4,480', '13.177633%'],
              [tr('Overall', '总体'), '94,458', '100.000000%', '9,466', '10.021385%'],
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
          { heading: tr('Semantic correction', '语义修正'), items: [tr('10.021% is the death-event prevalence and therefore the share for which death_time is applicable. It is not a death-time measurement rate.', '10.021% 是死亡事件比例，因此也是 death_time 的适用比例；它不是死亡时间测量率。'), tr('Twenty-eight death times precede the ICU origin and remain a separate timing-protocol flag, not missingness.', '28 个死亡时间早于 ICU origin，作为独立时间协议标记保留，不计为缺失。')] },
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
      artifact: 'agent_plan.json', title: tr('Agent plan', 'Agent 计划'),
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
      artifact: 'scientific_readiness.json', title: 'Scientific Readiness',
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
  function tool(id, name, text, resource, resources) { return { id, kind: 'tool', toolName: name, status: 'complete', text: text || '', resource: resource || null, resources: resources || [] }; }
  function pipeline(id, label, text, resource, resources, extra) { return Object.assign({ id, kind: 'pipeline', status: 'complete', label, text: text || '', resource: resource || null, resources: resources || [] }, extra || {}); }
  function submitted(id, label, text, code) { return { id, kind: 'submitted', status: 'complete', label, text: text || '', code: code || '', owner: 'EasyICU' }; }
  function retry(id, label, text) { return { id, kind: 'retry', status: 'complete', label, text: text || '', owner: 'agent-run' }; }
  function message(id, role, text, resources) { return { id, role, text, complete: true, resources: resources || [] }; }
  function standardRunResources() {
    return [
      artifactResource('run_context.json', tr('Run context', '运行上下文')),
      artifactResource('cohort_summary.json', tr('Cohort summary', '队列摘要')),
      artifactResource('quality_gate.json', tr('Evidence verification', '证据核验')),
      artifactResource('agent_plan.json', tr('Agent plan', 'Agent 计划')),
      artifactResource('literature_evidence.json', tr('Literature evidence', '文献证据')),
      artifactResource('scientific_plan_review.json', tr('Scientific plan review', '科学计划审阅')),
      artifactResource('scientific_readiness.json', 'Scientific Readiness'),
      artifactResource('manuscript_draft.json', tr('Locked manuscript draft', '锁定论文草稿')),
      artifactResource('figure_gallery.json', tr('Figure gallery', '图件画廊')),
      artifactResource('result_tables.json', tr('Research result tables', '科研结果表')),
      artifactResource('source_run_manifest.json', tr('Source run manifest', '原始运行清单')),
      artifactResource('evidence_ledger.json', tr('Evidence ledger', '证据账本')),
    ];
  }

  function messages() {
    const documents = reviewResources();
    const runResources = standardRunResources();
    return [
      message('reviewer-user-1', 'user', tr(
        'Run a complete governed Research Agent demonstration on the prepared ICU data. Show the real planning lifecycle, pause for plan review, and keep every inspectable receipt available.',
        '请在准备后的 ICU 数据上运行完整的受治理 Research Agent Demo。展示真实规划生命周期，在计划审阅处暂停，并保留所有可检查回执。',
      )),
      activity('reviewer-planning', 1000, 194000, [
        submitted('plan-submit', tr('EasyICU preflight task submitted', 'EasyICU 预检任务已提交'), WRAPPER_RUN_ID, 'easyicu_run_submitted'),
        pipeline('provider', tr('Research Agent provider authorized', 'Research Agent Provider 已授权'), tr('Provider budget and credential fingerprint were bound before planning.', '规划前已绑定 Provider 预算与凭据指纹。')),
        pipeline('select', tr('Selecting concepts and materializing a typed analysis universe', '正在选择概念并生成 typed 分析全集'), tr('The prepared-data contract was used; no raw CSV path entered the pipeline.', '使用准备后数据合同；没有原始 CSV 路径进入流水线。')),
        pipeline('planning', tr('Research Agent planning started; execution remains blocked pending human plan review', 'Research Agent 开始规划；执行仍暂停等待人工计划审阅'), ''),
        pipeline('run-start', tr('Starting research-agent run.', '正在启动 Research Agent 运行。'), ''),
        pipeline('cohort', tr('Cohort materialised to parquet.', '队列已生成 parquet。'), tr('94,458 adult ICU stays.', '94,458 个成人 ICU stays。'), runResources[1]),
        pipeline('runtime', tr('Execution runtime validated before planning.', '规划前已验证执行运行时。'), ''),
        pipeline('context', tr('Research context built.', '研究上下文已构建。'), '', runResources[0]),
        pipeline('audit', tr('Initial cohort audit passed.', '初始队列审计已通过。'), '', runResources[2]),
        pipeline('literature', tr('Building pre-plan literature and hypothesis blueprint.', '正在构建计划前文献与假设蓝图。'), '', runResources[4]),
        pipeline('draft-1', tr('Generating plan draft 1/5.', '正在生成计划草案 1/5。'), '1/5'),
        retry('retry-1', tr('Plan draft 1/5 did not satisfy the scientific contract; retrying.', '计划草案 1/5 未满足科学合同；正在重试。'), tr('The rejected draft was not promoted to plan authority.', '被拒草案未提升为计划权限。')),
        pipeline('draft-2', tr('Generating plan draft 2/5.', '正在生成计划草案 2/5。'), '2/5'),
        pipeline('draft-pass', tr('Plan draft 2/5 passed contract validation.', '计划草案 2/5 通过合同验证。'), '2/5', runResources[5]),
        pipeline('plan-ready', tr('Analysis plan ready with 6 step(s).', '分析计划已就绪，共 6 个步骤。'), '', runResources[3]),
        pipeline('plan-pause', tr('Plan contract passed; analysis paused for human review', '计划合同已通过；分析已暂停，等待人工审阅'), tr('The pause binds this exact plan and does not grant publication authority.', '暂停点绑定这份精确计划，不授予发表权限。'), runResources[3], runResources.slice(0, 8), { code: 'blocked', owner: 'agent-run' }),
      ], { displayTitle: tr('Analysis plan ready for review', '分析计划已就绪，等待审阅'), childJobId: 'reviewer-plan-run' }),
      message('reviewer-assistant-1', 'assistant', tr(
        '**The plan contract passed and analysis is paused.** Open the plan, literature, scientific review, cohort, or run context directly from the lifecycle receipt before approving execution.',
        '**计划合同已通过，分析已暂停。** 批准执行前，可以直接从生命周期回执打开计划、文献、科学审阅、队列或运行上下文。',
      ), runResources.slice(0, 8)),
      message('reviewer-user-2', 'user', tr('I approve this exact reviewed plan. Resume it without changing the study configuration.', '我批准这份精确审阅计划。请在不改变研究配置的情况下恢复执行。')),
      activity('reviewer-execution', 200000, 578000, [
        submitted('execute-submit', tr('EasyICU research task submitted', 'EasyICU 科研任务已提交'), SOURCE_RUN_ID, 'easyicu_full_run_submitted'),
        pipeline('execute-provider', tr('Research Agent provider authorized', 'Research Agent Provider 已授权'), tr('Cumulative Provider accounting resumed from the same durable ledger.', '从同一持久 Provider ledger 恢复累计计费。')),
        pipeline('resume', tr('Exact reviewed plan restored from the human-review checkpoint.', '已从人工审阅 checkpoint 恢复精确计划。'), '', runResources[5]),
        pipeline('execute-start', tr('Deterministic execution started.', '确定性执行已开始。'), ''),
        pipeline('step-1-start', tr('Running 1/6 · cohort definition and attrition', '正在运行 1/6 · 队列定义与纳排'), ''),
        pipeline('step-1-done', tr('Completed 1/6 · adult ICU cohort', '完成 1/6 · 成人 ICU 队列'), '94,458 / 94,458 stays retained.', runResources[1]),
        pipeline('step-2-start', tr('Running 2/6 · typed measurement audit', '正在运行 2/6 · typed 测量审计'), ''),
        pipeline('step-2-done', tr('Completed 2/6 · applicability-aware data quality', '完成 2/6 · 适用性敏感的数据质量'), tr('death_time: 9,466 applicable; 0 missing among applicable.', 'death_time：9,466 适用；适用者中 0 缺失。'), artifactResource('applicability_audit.json', tr('Open applicability audit', '打开适用性审计'))),
        pipeline('step-3-start', tr('Running 3/6 · exposure-outcome distribution', '正在运行 3/6 · 暴露-结局分布'), ''),
        pipeline('step-3-done', tr('Completed 3/6 · counts-only result table', '完成 3/6 · 仅计数结果表'), tr('No inferential estimate was added.', '未新增推断估计。'), runResources[9]),
        pipeline('step-4', tr('Completed 4/6 · phenotype and mortality figure', '完成 4/6 · 表型与死亡图'), '', runResources[8]),
        pipeline('step-5', tr('Completed 5/6 · cohort accounting figure', '完成 5/6 · 队列账本图'), '', runResources[8]),
        pipeline('step-6', tr('Completed 6/6 · data-quality figure', '完成 6/6 · 数据质量图'), tr('Conditional event-time applicability was separated from missingness.', '条件事件时间适用性已与缺失分开。'), runResources[8]),
        pipeline('evidence', tr('Registered 125 evidence records.', '已登记 125 条证据记录。'), '', runResources[11]),
        pipeline('numeric', tr('Verified registered descriptive numbers and denominators.', '已核验登记的描述性数值与分母。'), '', runResources[2]),
        pipeline('provider-ledger', tr('Provider ledger completed.', 'Provider ledger 已完成。'), '14 calls · 162,256 tokens · $2.30776', runResources[10]),
        pipeline('writer', tr('Writer phase started under STRICT evidence enforcement.', 'Writer 阶段在 STRICT 证据执行下启动。'), ''),
        pipeline('writer-stop', tr('Formal manuscript withheld by the deterministic authority gate.', '正式稿件被确定性权限闸门拒绝。'), tr('Execution output remains available; publication authority was not granted.', '执行产物保持可用；未授予发表权限。'), runResources[7], [], { code: 'withheld_as_designed', owner: 'agent-run' }),
        pipeline('readiness', tr('Scientific readiness projected with open blockers.', 'Scientific Readiness 已投影未闭合问题。'), '', runResources[6]),
        pipeline('privacy', tr('Aggregate-only browser privacy projection passed.', '仅聚合浏览器隐私投影已通过。'), tr('No patient rows, identifier columns, credentials, or host paths.', '无患者行、标识列、凭据或宿主路径。')),
        pipeline('dossier', tr('Reviewer HTML and PDF dossier registered.', '审稿 HTML 与 PDF 报告已登记。'), '', documents[0], documents),
        pipeline('end', tr('Execution complete; manuscript authority withheld as designed.', '执行完成；稿件权限按设计拒绝。'), '', runResources[10], runResources.slice(8), { code: 'engineering_validation_complete', owner: 'agent-run' }),
      ], { displayTitle: tr('Analysis complete; reviewer evidence ready', '分析完成；审稿证据已就绪'), childJobId: 'reviewer-execution-run' }),
      message('reviewer-assistant-2', 'assistant', tr(
        '**Execution complete: 6/6 steps.** The cohort contained **94,458 ICU stays**; the phenotype was present in **33,997 (35.991658%)**. Observed mortality was **4,986/60,461 (8.246638%)** without the phenotype and **4,480/33,997 (13.177633%)** with it.\n\nThese are descriptive counts and proportions only. The projection exposes 125 evidence records, 12 tables, three figures, and 14 Provider calls without exposing patient rows or host paths.',
        '**执行完成：6/6 步。** 队列包含 **94,458 个 ICU stays**；其中 **33,997（35.991658%）** 为表型阳性。表型阴性组观察死亡为 **4,986/60,461（8.246638%）**，阳性组为 **4,480/33,997（13.177633%）**。\n\n这些仅是描述性计数与比例。投影展示 125 条证据、12 张表、3 张图和 14 次 Provider 调用，同时不暴露患者行或宿主路径。',
      ), runResources.slice(6).concat(documents)),
      message('reviewer-user-3', 'user', tr('The manuscript gate says it was withheld. Does that mean the Demo failed?', '稿件闸门显示被拒绝，这是否意味着 Demo 失败？')),
      activity('reviewer-authority', 580000, 586000, [
        tool('authority', 'easyicu_inspect_scientific_readiness', tr('Loaded the exact scientific-readiness and quality-gate projections.', '已加载精确 Scientific Readiness 与质量闸门投影。'), runResources[6], [runResources[2], runResources[7], artifactResource('authority_verdict.json', tr('Open authority verdict', '打开权限裁决'))]),
        pipeline('writer-gate', tr('Clinical manuscript · withheld as designed', '临床稿件 · 按设计拒绝'), tr('STRICT evidence and scientific-authority requirements were not satisfied.', 'STRICT 证据与科学权限要求未满足。'), runResources[7]),
        pipeline('demo-gate', tr('Reviewer demonstration · complete', '审稿人演示 · 完整完成'), tr('Planning, review pause, execution, evidence projection, and authority separation were demonstrated.', '规划、审阅暂停、执行、证据投影与权限分离均已展示。'), documents[0]),
      ], { displayTitle: tr('Authority boundary inspected', '权限边界已检查') }),
      message('reviewer-assistant-3', 'assistant', tr(
        '**No. The reviewer Demo is complete.** The product question was whether the governed workflow could execute, expose evidence, preserve privacy, and prevent unsupported authority escalation. It did all four.\n\nThe clinical manuscript is a different deliverable. Withholding it is the expected safety result because novelty retrieval, independent scientific review, publication-figure authority, and exact manuscript evidence binding are not closed. The interface now marks this amber as **withheld as designed**, not as a failed Demo.',
        '**不是。审稿人 Demo 已完整完成。** 产品问题是受治理流程能否完成执行、展示证据、保护隐私并阻止无依据的权限升级；这四项均已实现。\n\n临床稿件是另一项交付物。由于新颖性检索、独立科学审阅、投稿图件权限及稿件精确证据绑定尚未闭合，拒绝稿件是预期安全结果。界面现在以琥珀色标记为**按设计拒绝**，不再把它显示成 Demo 失败。',
      )),
      message('reviewer-user-4', 'user', tr('Package the complete evidence so I can review it.', '把完整证据整理好供我审阅。')),
      activity('reviewer-package', 590000, 592300, [
        tool('dossier', 'easyicu_inspect_run', tr('Loaded the complete standard artifact set, self-contained reviewer dossier, and registered PDF.', '已加载完整标准产物集、自包含审稿报告及登记 PDF。'), documents[0], [documents[1]].concat(runResources)),
      ], { displayTitle: tr('Complete reviewer package loaded', '完整审稿包已加载') }),
      message('reviewer-assistant-4', 'assistant', tr(
        '**Reviewer package complete.** The dossier contains the protocol, exact lifecycle, aggregate case table, three figures, strongest unresolved scientific requirements, Provider usage, and 11 SHA-256 source bindings. Open the HTML for interactive review or the six-page PDF for circulation.',
        '**审稿包已完整生成。** 报告包含协议、精确生命周期、聚合案例表、3 张图、最重要的未闭合科学要求、Provider 使用及 11 个 SHA-256 来源绑定。可打开 HTML 交互审阅，或使用 6 页 PDF 传阅。',
      )),
    ];
  }

  function workflow() {
    return {
      kind: 'reviewer_validation_demo', current_stage: 'manuscript',
      completed_required_stages: 8, required_stage_count: 8,
      next_action_code: 'reviewer_demo_complete',
      stages: [
        ['question', 'complete', 'reviewer_protocol_bound'],
        ['idea', 'complete', 'bounded_validation_objective_selected'],
        ['setup', 'complete', 'prepared_data_contract_verified'],
        ['extraction', 'complete', 'aggregate_projection_verified'],
        ['plan', 'complete', 'exact_plan_reviewed'],
        ['analysis', 'complete', 'six_of_six_steps_complete'],
        ['interpretation', 'complete', 'descriptive_ceiling_preserved'],
        ['manuscript', 'complete', 'reviewer_dossier_complete'],
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

  window.EU_GUIDED_PI_DEMO = {
    messages, workflow, artifact, previewArtifact, hasArtifact, artifactLabel,
    reviewResources, primaryDocument: () => reviewResources()[0], sourceRunId: SOURCE_RUN_ID,
  };
})();
