/* Guided Pi complete research-demo transcript.
   Owner: read-only product-demo fixtures and their safe structured preview.
   It never starts a provider job or mutates a real EasyICU project. */
(function () {
  'use strict';

  const SOURCE_RUN_ID = 'run_20260811T030843_4d45a8';
  const SOURCE_AUTHORITY = 'engineering_canary_demo_only';
  const SOURCE_FIGURE = '/assets/demo/e1-publication-figure.png';

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function demoArtifact(name, title, summary, metrics, sections, sourceKind, extra) {
    return Object.assign({
      schema_version: 'easyicu.pi-product-demo-artifact/1',
      artifact: name,
      title,
      summary,
      status: 'read_only_product_demo',
      source_run_id: SOURCE_RUN_ID,
      source_authority: sourceKind || SOURCE_AUTHORITY,
      reportable: false,
      claim_ceiling: 'analysis_only',
      metrics: metrics || [],
      sections: sections || [],
    }, extra || {});
  }
  function literatureRecords() {
    return [
      {
        key: 'singer_sepsis3_2016', year: '2016', venue: 'JAMA', pmid: '26903338',
        title: 'The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).',
        relevance: tr('Clinical definition of Sepsis-3 and SOFA-aligned organ dysfunction.', 'Sepsis-3 与 SOFA 器官功能障碍的临床定义。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
      },
      {
        key: 'strobe_2007', year: '2007', venue: 'Annals of Internal Medicine', pmid: '17938396',
        title: 'The STROBE statement: guidelines for reporting observational studies.',
        relevance: tr('Prespecifies transparent cohort, model, uncertainty, and limitation reporting.', '规定观察性研究的队列、模型、不确定性与局限性报告。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/17938396/',
      },
      {
        key: 'record_2015', year: '2015', venue: 'PLOS Medicine', pmid: '26440803',
        title: 'The RECORD statement for studies using routinely collected health data.',
        relevance: tr('Adds source-data, code-list, linkage, and reproducibility obligations.', '补充常规医疗数据的来源、代码表、链接与可复现性要求。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/26440803/',
      },
      {
        key: 'suissa_immortal_time_2008', year: '2008', venue: 'American Journal of Epidemiology', pmid: '18056625',
        title: 'Immortal time bias in pharmacoepidemiology.',
        relevance: tr('Motivates explicit time zero and exposure-opportunity checks.', '支持明确 time zero 与暴露机会审计。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/18056625/',
      },
      {
        key: 'anderson_landmark_1983', year: '1983', venue: 'Journal of Clinical Oncology', pmid: '6668489',
        title: 'Analysis of survival by tumor response and other time-dependent outcome comparisons.',
        relevance: tr('Classical rationale for landmarking post-baseline classifications.', '为基线后分类的 landmark 分析提供经典方法学依据。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/6668489/',
      },
      {
        key: 'durrleman_splines_1989', year: '1989', venue: 'Statistics in Medicine', pmid: '2657958',
        title: 'Flexible regression models with cubic splines.',
        relevance: tr('Supports checking nonlinear continuous-covariate effects.', '支持检查连续协变量的非线性效应。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/2657958/',
      },
      {
        key: 'sterne_missing_data_2009', year: '2009', venue: 'BMJ', pmid: '19564179',
        title: 'Multiple imputation for missing data in epidemiological and clinical research.',
        relevance: tr('Frames missing-data assumptions and complete-case sensitivity analysis.', '界定缺失数据假设与完整病例敏感性分析。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/19564179/',
      },
      {
        key: 'ricu_2023', year: '2023', venue: 'Open-source software',
        title: "ricu: R's interface to intensive care data.",
        relevance: tr('Concept-dictionary and cross-database design precedent.', '概念字典与跨数据库设计的先例。'),
        url: 'https://github.com/eth-mds/ricu',
      },
      {
        key: 'johnson_mimiciv_2023', year: '2023', venue: 'Scientific Data', pmid: '36596836',
        title: 'MIMIC-IV, a freely accessible electronic health record dataset.',
        relevance: tr('Primary source description for the database used in this demonstration.', '本演示所用数据库的主要来源说明。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/36596836/',
      },
    ];
  }

  function currentAuditRecords() {
    return [
      {
        key: 'ranzani_sofa2_2025', year: '2025', venue: 'JAMA',
        title: 'Development and Validation of the Sequential Organ Failure Assessment (SOFA)-2 Score.',
        relevance: tr('Current primary SOFA-2 development and validation study: 3.3 million adult ICU encounters across 10 cohorts. This directly limits the novelty of a small SOFA-2–mortality association study.', '当前 SOFA-2 开发与验证主文献：10 个队列、330 万例成人 ICU encounter。它直接限制了小样本 SOFA-2–死亡关联研究的新颖性。'),
        url: 'https://jamanetwork.com/journals/jama/fullarticle/2840822',
      },
      {
        key: 'bai_sofa2_mimiciv_2026', year: '2026', venue: 'Journal of Intensive Care', pmid: '41877184',
        title: 'SOFA-2 versus SOFA-1 for mortality prediction in critically ill patients.',
        relevance: tr('Direct MIMIC-IV prior art using 65,366 critically ill patients; the historical 140-stay canary is not a credible novelty claim against this evidence.', '直接使用 MIMIC-IV 65,366 例危重患者的同主题先前研究；历史 140-stay canary 无法据此主张可信新颖性。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/41877184/',
      },
      {
        key: 'wei_sofa2_pneumonia_sepsis_2026', year: '2026', venue: 'Critical Care',
        title: 'SOFA-2 score predicts mortality in pneumonia-associated sepsis: a retrospective cohort study.',
        relevance: tr('Related MIMIC-IV sepsis analysis with 7,150 patients plus external validation; it demonstrates the expected scope for a publishable comparative study.', '相关 MIMIC-IV 脓毒症分析包含 7,150 例并做外部验证，体现了可投稿比较研究应有的范围。'),
        url: 'https://doi.org/10.1186/s13054-026-06027-4',
      },
      {
        key: 'lee_tarmos_2021', year: '2021', venue: 'Journal of Clinical Epidemiology', pmid: '33539930',
        title: 'Framework for the treatment and reporting of missing data in observational studies (TARMOS).',
        relevance: tr('A newer structured framework for missing-data assumptions and reporting than the historical 2009 seed alone.', '相比仅使用 2009 年历史种子，它提供了更新的观察性研究缺失数据处理与报告框架。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/33539930/',
      },
      {
        key: 'surviving_sepsis_2021', year: '2021', venue: 'Intensive Care Medicine', pmid: '34605781',
        title: 'Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2021.',
        relevance: tr('Current management context for adult sepsis; clinical guidance must remain distinct from validation of an experimental score representation.', '成人脓毒症当前管理背景；临床指南不能被当作实验性评分表征的验证证据。'),
        url: 'https://pubmed.ncbi.nlm.nih.gov/34605781/',
      },
      {
        key: 'icmje_current', year: '2026', venue: 'ICMJE Recommendations',
        title: 'Preparing a Manuscript for Submission to a Medical Journal.',
        relevance: tr('Current submission expectations for reporting guidelines, source population, reproducible methods, uncertainty, limitations, and contextualization.', '当前投稿对报告规范、来源人群、可复现方法、不确定性、局限与证据语境的要求。'),
        url: 'https://www.icmje.org/recommendations/browse/manuscript-preparation/preparing-for-submission.html',
      },
    ];
  }

  function artifacts() {
    return {
      'idea_shortlist.json': demoArtifact(
        'idea_shortlist.json',
        tr('Research opportunity shortlist', '研究机会候选'),
        tr('Three technically executable directions reconstructed for the product walkthrough. The historical run did not perform a live prior-art search, so none is authorized as a reliable publication idea.', '为产品流程演示重建的 3 个技术可执行方向。历史运行没有执行实时先前研究检索，因此任何一个都未获授权成为可靠投稿 Idea。'),
        [
          { label: tr('Technically executable candidates', '技术可执行候选'), value: '3' },
          { label: tr('Live prior-art search in source run', '源运行实时先前研究检索'), value: tr('No', '否') },
          { label: tr('Publication recommendation', '投稿推荐'), value: tr('No · novelty not established', '否 · 新颖性未成立') },
        ],
        [
          { heading: tr('1 · Descriptive', '1 · 描述性'), items: [tr('Estimate standard Sepsis-3 prevalence with the current SOFA-1-based definition and a visible ICU-stay denominator.', '使用当前基于 SOFA-1 的标准 Sepsis-3 定义估计比例，并明确 ICU stay 分母。')] },
          { heading: tr('2 · Historical canary association study', '2 · 历史 canary 关联研究'), items: [tr('Useful for exercising the pipeline, but not a novel paper idea: current studies already developed and validated SOFA-2 against mortality in much larger cohorts, including MIMIC-IV.', '可用于验证流水线，但不是新颖论文 Idea：当前研究已在远大于本队列的样本中开发并验证 SOFA-2 与死亡的关系，其中包括 MIMIC-IV。'), tr('This indicator is not the standard Sepsis-3 definition and the 140-stay engineering subset cannot establish generalizability.', '该指标不是标准 Sepsis-3 定义，且 140-stay 工程子集无法建立可推广性。')] },
          { heading: tr('3 · Extension', '3 · 延伸方向'), items: [tr('Repeat the same transparent definition in another compatible ICU database.', '在另一兼容 ICU 数据库复现相同的透明定义。')] },
          { heading: tr('Adjudication', '裁决'), items: [tr('Keep direction 2 only as a read-only engineering validation case. A publishable idea requires a fresh, outcome-blind prior-art screen and a differentiated question.', '方向 2 仅保留为只读工程验证案例。可投稿 Idea 必须重新进行不看结果的先前研究筛查，并形成明确差异化问题。')] },
        ],
        'independent_product_audit_20260811',
      ),
      'literature_evidence.json': demoArtifact(
        'literature_evidence.json',
        tr('Literature search and evidence registry', '文献检索与证据登记'),
        tr('The historical E1 bundle contains nine curated seeds, not a completed search. A separate dated product audit adds current 2025–2026 evidence and shows that the canary idea is not novel.', '历史 E1 bundle 包含 9 条人工种子，并非完成的检索。另行开展的定期产品审计补充了 2025–2026 年当前证据，并表明 canary Idea 不具新颖性。'),
        [
          { label: tr('Historical curated seeds', '历史人工种子'), value: '9' },
          { label: tr('Historical search conducted', '历史检索是否执行'), value: tr('No', '否') },
          { label: tr('Retrieval sources enabled', '启用检索源'), value: '0' },
          { label: tr('Independent current-audit records', '独立当前审计记录'), value: '6' },
        ],
        [
          { heading: tr('Historical source receipt', '历史来源回执'), items: [tr('search_conducted = false; curated_seed_count = 9; sources_enabled = []. No PRISMA flow exists for the historical run.', 'search_conducted = false；curated_seed_count = 9；sources_enabled = []。历史运行不存在 PRISMA 流程。')] },
          { heading: tr('Relevance adjudication', '相关性裁决'), items: [tr('Sepsis-3, STROBE, RECORD, and MIMIC-IV remain useful foundational sources.', 'Sepsis-3、STROBE、RECORD 与 MIMIC-IV 仍是有用的基础来源。'), tr('Immortal-time and landmark references are not directly applied by this fixed-window association plan; the spline paper does not justify a model that treats age linearly.', '固定时间窗关联计划并未直接应用 immortal-time 与 landmark 方法；样条文献也不能为线性处理年龄的模型提供依据。')] },
          { heading: tr('Current audit · 2026-08-11', '当前审计 · 2026-08-11'), items: [tr('Current SOFA-2 development, MIMIC-IV validation, sepsis validation, missing-data, and submission guidance were checked separately. These sources disconfirm novelty; they were not retroactively inserted into the historical Agent run.', '另行核查了当前 SOFA-2 开发、MIMIC-IV 验证、脓毒症验证、缺失数据与投稿规范。这些来源否定了新颖性；它们没有被追溯性写入历史 Agent 运行。')] },
        ],
        'independent_product_audit_20260811',
        {
          citations: literatureRecords().concat(currentAuditRecords()),
          historical_search_receipt: { search_conducted: false, curated_seed_count: 9, sources_enabled: [], sources_returning: [], prisma: null },
          current_audit_receipt: { audited_at: '2026-08-11', record_count: 6, authority: 'independent_product_audit_not_source_run_authority' },
        },
      ),
      'extraction_quality.json': demoArtifact(
        'extraction_quality.json',
        tr('Data package and quality review', '数据包与质量审阅'),
        tr('Aggregate-only projection of a prepared 140-stay MIMIC-IV engineering subset. File provenance is present, but this projection does not establish the full source population or representativeness.', '准备后的 140-stay MIMIC-IV 工程子集的仅聚合投影。文件来源可追踪，但该投影没有建立完整来源人群或代表性。'),
        [
          { label: tr('ICU stays', 'ICU stays'), value: '140' },
          { label: tr('Experimental SOFA-2 sensitivity indicator present', '实验性 SOFA-2 敏感性指标阳性'), value: '53 / 140 (37.9%)' },
          { label: tr('Primary complete cases', '主分析完整病例'), value: '140 / 140' },
          { label: tr('Full-database population scope', '全库来源人群范围'), value: tr('Not closed', '未闭合') },
        ],
        [
          { heading: tr('Registered fields', '已登记字段'), items: ['sep3_sofa2_max', 'death', 'age', 'sex'] },
          { heading: tr('Time and denominator', '时间与分母'), items: [tr('Exposure window: ICU admission through 24 hours.', '暴露窗：入 ICU 至 24 小时。'), tr('Analysis unit: one prepared row per ICU stay.', '分析单位：每个 ICU stay 一条准备后记录。')] },
          { heading: tr('What is verified', '已核验内容'), items: [tr('The prepared files are digest-traceable and contain a 140-stay denominator with complete primary fields.', '准备后文件具备摘要来源链，分母为 140 个 stay，主分析字段完整。')] },
          { heading: tr('What is not verified', '未核验内容'), items: [tr('The projection does not close the MIMIC-IV source population, eligibility/exclusion flow, sampling mechanism, or representativeness. “140 complete rows” is not the same as “data extraction is scientifically adequate.”', '该投影没有闭合 MIMIC-IV 来源人群、纳排流程、抽样机制或代表性。“140 行完整”不等于“数据提取在科学上充分”。')] },
          { heading: tr('Governance', '治理边界'), items: [tr('Engineering-canary aggregate only; not formal paper evidence.', '仅工程试跑聚合结果，不是正式论文证据。')] },
        ],
        SOURCE_AUTHORITY,
        {
          tables: [{
            label: tr('Prepared aggregate cohort receipt', '准备后聚合队列回执'),
            headers: [tr('Field', '字段'), tr('Value', '值'), tr('Missing', '缺失')],
            rows: [
              ['sep3_sofa2_max', '53 positive / 87 negative', '0 / 140'],
              ['death', '15 deaths / 125 survivors', '0 / 140'],
              ['age', 'mean 62.0 years', '0 / 140'],
              ['sex', '63 female / 77 male', '0 / 140'],
            ],
          }],
        },
      ),
      'agent_plan.json': demoArtifact(
        'agent_plan.json',
        tr('Evidence-bound analysis plan', '证据绑定的分析计划'),
        tr('The historical plan is executable, but it is not a comprehensive publication plan: five of eleven steps are rendering, the complete-case replay is identical because no primary field is missing, and current SOFA-2 prior art was absent.', '历史计划可以执行，但不是完整投稿分析计划：11 步中 5 步是绘图；因主字段无缺失，完整病例复跑与主分析相同；且缺少当前 SOFA-2 先前研究。'),
        [
          { label: tr('Analysis / audit steps', '分析 / 审计步骤'), value: '6' },
          { label: tr('Render-only steps', '仅绘图步骤'), value: '5' },
          { label: tr('Primary model', '主模型'), value: tr('Logistic regression', 'Logistic 回归') },
          { label: tr('Adjustment', '调整变量'), value: tr('Age + sex', '年龄 + 性别') },
        ],
        [
          { heading: tr('Primary analysis', '主分析'), items: [tr('Estimate the odds ratio for in-hospital death: experimental SOFA-2 sensitivity indicator present versus absent.', '估计实验性 SOFA-2 敏感性指标阳性对阴性的院内死亡 OR。'), tr('Report group counts, risks, adjusted OR, and 95% confidence interval.', '报告分组人数、风险、调整 OR 与 95% 置信区间。')] },
          { heading: tr('Quality and sensitivity', '质量与敏感性'), items: [tr('Audit variable availability and value missingness before modelling.', '建模前审计变量可得性与数值缺失。'), tr('Replay the same specification in complete cases without changing the estimand.', '在完整病例中按同一设定复跑，不改变 estimand。')] },
          { heading: tr('Literature anchors', '文献依据'), items: [tr('Sepsis-3 supports the clinical framing only and does not validate the experimental SOFA-2 indicator; STROBE and RECORD govern reporting.', 'Sepsis-3 仅支持临床背景，不验证实验性 SOFA-2 指标；报告遵循 STROBE 与 RECORD。')] },
          { heading: tr('Open scientific defects', '未闭合科学缺口'), items: [tr('No current SOFA-2 same-topic prior-art comparison was available to the Planner.', 'Planner 未获得当前 SOFA-2 同主题先前研究比较。'), tr('Only 15 outcome events support the exposure, age, and sex model; the plan did not prespecify an information-adequacy decision or a meaningful small-sample sensitivity analysis.', '暴露、年龄和性别模型仅有 15 个结局事件；计划未预先规定信息充分性裁决或有意义的小样本敏感性分析。'), tr('The robustness replay is not an independent sensitivity analysis when all 140 rows are already complete; external validation and a differentiated estimand are absent.', '当 140 行本就全部完整时，完整病例复跑不是独立敏感性分析；外部验证与差异化 estimand 均缺失。')] },
          { heading: tr('Citation-fit defect', '引文匹配缺口'), items: [tr('The spline paper is listed for a model that uses linear age; immortal-time and landmark papers are retained but not actually implemented. Exact relevance must be adjudicated, not inferred from a bibliography.', '样条文献被列给了线性年龄模型；immortal-time 与 landmark 文献虽保留却未实际实施。引文相关性必须逐项裁决，不能从参考文献列表推断。')] },
        ],
        'demo_projection_from_historical_agent_plan',
        {
          projection_note: tr('The 11 scientific steps come from the completed Agent plan. The citation alignment shown here is an explicit product-review projection from the same run\'s pre-plan LiteratureBundle; the historical canary predates the native per-step binding field now required by current runs.', '11 个科学步骤来自已完成的 Agent 计划。这里的引文对应关系是由同一次运行的 pre-plan LiteratureBundle 生成的产品审阅投影；该历史 canary 早于当前运行强制要求的逐步原生绑定字段。'),
          steps: [
            { step_id: '01_define_analysis_cohort', method: 'cohort definition + attrition', intent: tr('Materialize the prepared ICU cohort, preserve the visible denominator, and keep the sepsis framing distinct from the experimental exposure.', '生成准备后 ICU 队列、保留可见分母，并将脓毒症临床背景与实验性暴露明确区分。'), inputs: ['prepared MIMIC-IV export'], outputs: ['analysis_cohort', 'cohort_flow'], citation_keys: ['singer_sepsis3_2016', 'record_2015', 'johnson_mimiciv_2023'] },
            { step_id: '02_baseline_table_by_primary_exposure', method: 'descriptive · Table 1', intent: tr('Describe age, sex, and outcome overall and by exposure group.', '按暴露组描述年龄、性别和结局。'), inputs: ['analysis_cohort', 'sep3_sofa2_max', 'age', 'sex', 'death'], outputs: ['table_one'], citation_keys: ['strobe_2007', 'record_2015'] },
            { step_id: '03_exposure_outcome_distribution', method: 'descriptive risks + Wilson CI', intent: tr('Report experimental-indicator prevalence and mortality risk by exposure level.', '报告实验性指标比例及分组死亡风险。'), inputs: ['analysis_cohort', 'sep3_sofa2_max', 'death'], outputs: ['exposure_outcome_distribution'], citation_keys: ['strobe_2007', 'record_2015'] },
            { step_id: '04_measurement_missingness_audit', method: 'availability + missingness audit', intent: tr('Audit every primary-model field before interpretation.', '解读前审计所有主模型字段。'), inputs: ['sep3_sofa2_max', 'death', 'age', 'sex'], outputs: ['missingness_measurement_audit'], citation_keys: ['record_2015', 'sterne_missing_data_2009'] },
            { step_id: '05_primary_adjusted_association', method: 'age/sex-adjusted logistic regression', intent: tr('Estimate the adjusted odds ratio for in-hospital death.', '估计院内死亡的调整后 OR。'), inputs: ['analysis_cohort', 'sep3_sofa2_max', 'death', 'age', 'sex'], outputs: ['adjusted_association_estimates'], citation_keys: ['strobe_2007', 'record_2015', 'durrleman_splines_1989'] },
            { step_id: '06_primary_association_figure', method: 'dot-interval visualization', intent: tr('Render the primary estimate with its 95% confidence interval.', '绘制主要估计及 95% 置信区间。'), inputs: ['adjusted_association_estimates'], outputs: ['primary_adjusted_association'], citation_keys: ['strobe_2007'] },
            { step_id: '07_absolute_risk_figure', method: 'absolute-risk visualization', intent: tr('Show exposure prevalence and group-specific mortality risk.', '展示暴露比例与分组死亡风险。'), inputs: ['exposure_outcome_distribution'], outputs: ['descriptive_absolute_risk'], citation_keys: ['strobe_2007'] },
            { step_id: '08_robustness_replay', method: 'locked complete-case replay', intent: tr('Repeat the same estimand without silently changing exposure, outcome, or model.', '不改变暴露、结局或模型，按相同 estimand 完整病例复跑。'), inputs: ['analysis_cohort', 'sep3_sofa2_max', 'death', 'age', 'sex'], outputs: ['primary_or', 'complete_case_n', 'robustness_matrix'], citation_keys: ['sterne_missing_data_2009', 'record_2015'] },
            { step_id: '08_robustness_replay_figure', method: 'render-only robustness plot', intent: tr('Render the registered replay outputs without re-analysis.', '不重新分析，仅绘制已登记复跑产物。'), inputs: ['primary_or', 'robustness_matrix'], outputs: ['robustness_plot'], citation_keys: ['strobe_2007'] },
            { step_id: '09_robustness_figure', method: 'sensitivity interval plot', intent: tr('Display estimate stability across the locked specification.', '展示锁定设定下估计的稳定性。'), inputs: ['robustness_matrix'], outputs: ['robustness_sensitivity'], citation_keys: ['sterne_missing_data_2009'] },
            { step_id: '10_data_quality_figure', method: 'availability visualization', intent: tr('Display measurement availability for all analytic fields.', '展示所有分析字段的测量可得性。'), inputs: ['missingness_measurement_audit'], outputs: ['data_quality_missingness'], citation_keys: ['record_2015'] },
          ],
          citations: literatureRecords().concat(currentAuditRecords()),
        },
      ),
      'result_tables.json': demoArtifact(
        'result_tables.json',
        tr('Aggregate result tables', '聚合结果表'),
        tr('Bounded projections of Agent-produced CSV tables from the completed engineering canary.', '已完成工程试跑中 Agent 生成 CSV 表格的有界投影。'),
        [
          { label: tr('Table 1 rows shown', 'Table 1 展示行'), value: '6' },
          { label: tr('Primary complete cases', '主分析完整病例'), value: '140 / 140' },
          { label: tr('Primary model events', '主模型事件数'), value: '15' },
        ],
        [],
        SOURCE_AUTHORITY,
        {
          tables: [
            {
              label: tr('Table 1 · Baseline characteristics by experimental SOFA-2 sensitivity indicator', 'Table 1 · 按实验性 SOFA-2 敏感性指标分组的基线特征'),
              headers: [tr('Characteristic', '特征'), tr('Overall (n=140)', '总体 (n=140)'), tr('Absent (n=87)', '阴性 (n=87)'), tr('Present (n=53)', '阳性 (n=53)'), 'SMD'],
              rows: [
                [tr('Age, median [IQR], y', '年龄，中位数 [IQR]，岁'), '63 [52–72]', '63 [50–70.5]', '63 [56–78]', '0.299'],
                [tr('Female, n (%)', '女性，n (%)'), '63 (45.0)', '46 (52.9)', '17 (32.1)', '−0.430'],
                [tr('Male, n (%)', '男性，n (%)'), '77 (55.0)', '41 (47.1)', '36 (67.9)', '0.430'],
                [tr('In-hospital death, n (%)', '院内死亡，n (%)'), '15 (10.7)', '7 (8.0)', '8 (15.1)', '0.222'],
              ],
            },
            {
              label: tr('Primary adjusted association', '主要调整关联'),
              headers: [tr('Contrast', '对比'), tr('Adjusted OR', '调整 OR'), '95% CI', 'N', tr('Events', '事件')],
              rows: [[tr('Indicator present vs absent', '指标阳性 vs 阴性'), '1.50', '0.49–4.60', '140', '15']],
            },
            {
              label: tr('Absolute mortality risk', '绝对死亡风险'),
              headers: [tr('Group', '组别'), tr('Deaths / N', '死亡 / N'), tr('Risk', '风险'), '95% CI'],
              rows: [
                [tr('Indicator absent', '指标阴性'), '7 / 87', '8.0%', '4.0%–15.7%'],
                [tr('Indicator present', '指标阳性'), '8 / 53', '15.1%', '7.9%–27.1%'],
              ],
            },
          ],
        },
      ),
      'figure_gallery.json': demoArtifact(
        'figure_gallery.json',
        tr('Agent-produced publication figure', 'Agent 生成的论文图'),
        tr('The displayed PNG is the unchanged primary figure emitted by the completed E1 engineering canary.', '展示的 PNG 是 E1 工程试跑原样产出的主图。'),
        [
          { label: tr('Primary figures', '主图'), value: '1' },
          { label: tr('Supporting figures', '补充图'), value: '4' },
          { label: tr('Panels', '面板'), value: '2' },
        ],
        [{ heading: tr('Panels', '面板'), items: [tr('A · Adjusted odds ratio with 95% CI.', 'A · 调整后 OR 与 95% CI。'), tr('B · Feature-missingness audit.', 'B · 特征缺失审计。')] }],
        SOURCE_AUTHORITY,
        {
          projection_note: tr('The PNG is preserved exactly as emitted by the historical canary, including legacy indicator wording. Its sep3_sofa2_max exposure is an experimental SOFA-2 sensitivity representation, not standard Sepsis-3.', 'PNG 保持历史 canary 原样，包括旧指标措辞。其中 sep3_sofa2_max 暴露是实验性 SOFA-2 敏感性表征，不是标准 Sepsis-3。'),
          images: [{ src: SOURCE_FIGURE, alt: tr('Primary adjusted association and feature missingness figure', '主要调整关联与特征缺失图'), caption: tr('Historical engineering-canary figure · unchanged Agent output', '历史工程 canary 图 · Agent 原样产出') }],
        },
      ),
      'result_summary.json': demoArtifact(
        'result_summary.json',
        tr('Analysis results', '分析结果'),
        tr('The point estimate was above one, but the confidence interval was wide and included the null.', '点估计高于 1，但置信区间较宽且包含无效值。'),
        [
          { label: tr('Experimental SOFA-2 sensitivity indicator prevalence', '实验性 SOFA-2 敏感性指标比例'), value: '53 / 140 (37.9%)' },
          { label: tr('Mortality · indicator present', '死亡率 · 指标阳性'), value: '8 / 53 (15.1%)' },
          { label: tr('Mortality · indicator absent', '死亡率 · 指标阴性'), value: '7 / 87 (8.0%)' },
          { label: tr('Adjusted odds ratio', '调整后 OR'), value: '1.50 (95% CI 0.49–4.60)' },
        ],
        [
          { heading: tr('Interpretation', '结果解读'), items: [tr('The estimate is compatible with lower, similar, or higher odds of death; it is not conclusive evidence of an association.', '该区间同时兼容更低、相近或更高的死亡优势比，不能作为明确关联证据。'), tr('This observational result does not support a causal claim.', '该观察性结果不支持因果结论。')] },
          { heading: tr('Next evidence needed', '下一步证据'), items: [tr('Larger cohorts and external validation with the same concept and time-window contract.', '在更大队列中按相同概念与时间窗合同进行外部验证。')] },
        ],
      ),
      'quality_gate.json': demoArtifact(
        'quality_gate.json',
        tr('Evidence and reportability gate', '证据与可报告性闸门'),
        tr('Five automated engineering checks passed. Scientific review, literature currency, cohort scope, and publication authority did not pass; the persisted reviewer recommendation is major revision.', '5 项自动工程检查通过。科学审阅、文献时效性、队列范围与论文授权均未通过；持久化 reviewer 建议为 major revision。'),
        [
          { label: tr('Execution', '执行'), value: tr('Passed', '通过') },
          { label: tr('Analysis validation', '分析验证'), value: tr('Passed', '通过') },
          { label: tr('Evidence completeness', '证据完整性'), value: tr('Passed', '通过') },
          { label: tr('Numeric verification', '数值核验'), value: tr('Passed', '通过') },
          { label: tr('Reviewer recommendation', 'Reviewer 建议'), value: 'major_revision' },
          { label: tr('Publication ready', '可投稿'), value: 'false' },
          { label: tr('Reportability', '可报告性'), value: tr('Analysis-only', '仅供分析') },
        ],
        [{ heading: tr('Current gate', '当前闸门'), items: [tr('reportable = false; publication_ready = false; paper_authorized = false.', 'reportable = false；publication_ready = false；paper_authorized = false。'), tr('Human review alone cannot repair missing novelty, incomplete source-population authority, or an inadequate analysis plan.', '仅靠人工审阅不能修复缺失的新颖性、不完整的来源人群授权或不充分的分析计划。')] }],
      ),
      'scientific_readiness.json': demoArtifact(
        'scientific_readiness.json',
        tr('Scientific and publication readiness', '科学与投稿就绪度'),
        tr('Deterministic adjudication of the historical source artifacts. Engineering completion is preserved, while scientific and publication claims fail closed.', '对历史源产物的确定性裁决。保留工程完成事实，同时对科学与投稿论断保持 fail-closed。'),
        [
          { label: tr('Idea', 'Idea'), value: tr('Blocked · novelty not established', '阻断 · 新颖性未成立') },
          { label: tr('Literature', '文献'), value: tr('Blocked · curated seeds only', '阻断 · 仅人工种子') },
          { label: tr('Data', '数据'), value: tr('Review required · source scope open', '需审阅 · 来源范围未闭合') },
          { label: tr('Analysis', '分析'), value: tr('Blocked · major revision', '阻断 · major revision') },
          { label: tr('Manuscript', '稿件'), value: tr('Draft only · not publication ready', '仅初稿 · 不可投稿') },
        ],
        [
          { heading: 'LITERATURE_RETRIEVAL_NOT_CONDUCTED · blocker', items: [tr('The historical bundle explicitly records search_conducted=false. It cannot prove recency, relevance, or novelty.', '历史 bundle 明确记录 search_conducted=false，不能证明时效性、相关性或新颖性。')] },
          { heading: 'IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED · blocker', items: [tr('Current 2025–2026 same-topic studies make the 140-stay SOFA-2 mortality question unsuitable as a novelty claim.', '当前 2025–2026 年同主题研究使 140-stay SOFA-2 死亡问题不适合作为新颖性论断。')] },
          { heading: 'COHORT_SOURCE_SCOPE_NOT_EXPLICIT · major', items: [tr('The prepared denominator is traceable, but the full source population and selection path are not closed in the Web projection.', '准备后分母可追踪，但 Web 投影没有闭合完整来源人群和选择路径。')] },
          { heading: 'SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN · blocker', items: [tr('The persisted three-role reviewer report contains one major finding and aggregates to major_revision.', '持久化的三角色 reviewer 报告包含 1 个 major finding，汇总建议为 major_revision。')] },
          { heading: 'PAPER_AUTHORITY_NOT_GRANTED · blocker', items: [tr('Draft generation and 11/11 execution do not grant publication authority.', '生成初稿与 11/11 执行不授予论文发表权限。')] },
        ],
        'independent_product_audit_20260811',
        { scientific_status: 'analysis_only', publication_ready: false, paper_authorized: false, citations: currentAuditRecords() },
      ),
      'manuscript_draft.json': demoArtifact(
        'manuscript_draft.json',
        tr('Evidence-bound manuscript draft', '证据绑定的论文初稿'),
        tr('A structured evidence-bound draft was generated, but it does not meet publication requirements because novelty, source-population scope, analytical adequacy, and major reviewer findings remain open.', '已生成结构化证据绑定初稿，但由于新颖性、来源人群范围、分析充分性与 major reviewer finding 尚未闭合，它不满足投稿要求。'),
        [
          { label: tr('Status', '状态'), value: 'scientifically_blocked_draft' },
          { label: tr('Cohort', '队列'), value: '140 ICU stays' },
          { label: tr('Primary estimate', '主要估计'), value: 'OR 1.50 (95% CI 0.49–4.60)' },
        ],
        [
          { heading: tr('Draft conclusion', '初稿结论'), items: [tr('The experimental early SOFA-2 sensitivity indicator had an imprecise adjusted association with in-hospital mortality in this prepared cohort.', '在该准备队列中，早期实验性 SOFA-2 敏感性指标与院内死亡的调整关联估计不精确。')] },
          { heading: tr('Required author review', '作者必须审阅'), items: [tr('Confirm clinical interpretation, limitations, citation fit, and the analysis-only claim ceiling before any external use.', '对外使用前确认临床解读、局限性、引用匹配和仅供分析的结论上限。')] },
          { heading: tr('Publication blockers', '投稿阻断项'), items: [tr('The question is not differentiated from current SOFA-2 development and MIMIC-IV validation literature.', '问题未与当前 SOFA-2 开发及 MIMIC-IV 验证文献形成差异化。'), tr('The 140-stay engineering subset and 15 events do not support a generalizable, adequately stress-tested paper claim.', '140-stay 工程子集与 15 个事件不足以支撑可推广且经充分稳健性检验的论文论断。'), tr('The persisted reviewer package remains major_revision; human sign-off must not bypass these scientific blockers.', '持久化 reviewer 包仍为 major_revision；人工签署不能绕过这些科学阻断项。')] },
        ],
        SOURCE_AUTHORITY,
        {
          manuscript_sections: [
            { heading: tr('Title', '标题'), text: 'Retrospective MIMIC-IV ICU Cohort Study of an Experimental SOFA-2 Sensitivity Indicator and In-Hospital Mortality' },
            { heading: tr('Abstract · Methods', '摘要 · 方法'), text: tr('Observational ICU cohort study using the complete denominator of 140 stays. The exposure was the experimental first-24-hour sep3_sofa2_max sensitivity indicator, not standard Sepsis-3; the outcome was in-hospital death. Logistic regression adjusted for age and sex.', '观察性 ICU 队列研究，完整分母为 140 个 stay。暴露为入 ICU 后 24 小时实验性 sep3_sofa2_max 敏感性指标，而非标准 Sepsis-3；结局为院内死亡，Logistic 回归调整年龄和性别。') },
            { heading: tr('Abstract · Results', '摘要 · 结果'), text: tr('The adjusted odds ratio was 1.50 (95% CI 0.49–4.60). All 140 stays were retained in the complete-case replay, which reproduced the same estimate.', '调整后 OR 为 1.50（95% CI 0.49–4.60）。完整病例复跑保留全部 140 个 stay，并复现相同估计。') },
            { heading: tr('Interpretation', '结果解读'), text: tr('The interval spans associations below and above the null. The result is therefore imprecise, observational, and not evidence of causality.', '区间同时覆盖无效值两侧，因此结果不精确、仅属观察性关联，不能作为因果证据。') },
            { heading: tr('Limitations', '局限性'), text: tr('Single prepared database cohort, small event count, residual confounding, an experimental SOFA-2 sensitivity representation that is not standard Sepsis-3, and an LLM-in-the-loop workflow whose generated code was governed but not manually reviewed line by line.', '单一准备数据库队列、事件数较少、残余混杂、并非标准 Sepsis-3 的实验性 SOFA-2 敏感性表征，以及生成代码受治理但未逐行人工审阅的 LLM-in-the-loop 工作流。') },
            { heading: tr('Conclusion', '结论'), text: tr('External validation in larger and independently prepared ICU cohorts is warranted. The draft remains locked pending clinical and methods review.', '需要在更大且独立准备的 ICU 队列中外部验证。稿件在临床与方法学审阅前保持锁定。') },
          ],
        },
      ),
    };
  }

  function artifactResource(name, label) {
    const item = artifacts()[name];
    return {
      kind: 'demo_artifact', artifact: name, label: label || name,
      title: item ? item.title : name, run_id: SOURCE_RUN_ID,
      media_type: 'application/json',
    };
  }
  function literatureResource(record) {
    const item = record || literatureRecords()[0];
    return Object.assign({ kind: 'literature_source', label: `${item.key} · ${item.year}` }, item);
  }
  function literatureResources(keys) {
    const allowed = Array.isArray(keys) && keys.length ? new Set(keys) : null;
    return literatureRecords().concat(currentAuditRecords()).filter(item => !allowed || allowed.has(item.key)).map(literatureResource);
  }
  function activity(id, startedAt, endedAt, steps) {
    return { id, role: 'activity', status: 'complete', startedAt, endedAt, steps, expanded: true };
  }
  function tool(id, name, text, resource, resources) {
    return { id, kind: 'tool', toolName: name, status: 'complete', text: text || '', resource: resource || null, resources: resources || [] };
  }
  function pipeline(id, label, text, resource) {
    return { id, kind: 'pipeline', status: 'complete', label, text: text || '', resource: resource || null };
  }
  function message(id, role, text) { return { id, role, text, complete: true }; }

  function messages() {
    const allLiterature = literatureResources();
    return [
      message('demo-user-1', 'user', tr(
        'Help me find a feasible MIMIC-IV study about early sepsis and in-hospital death. Show me the literature and data feasibility before deciding.',
        '帮我从 MIMIC-IV 里找一个关于早期脓毒症和院内死亡、而且数据可做的研究方向。先把文献依据和数据可行性给我看，再决定。',
      )),
      activity('demo-activity-1', 1000, 7200, [
        tool('ideas', 'easyicu_mine_ideas', tr('Reconstructed technically executable directions from the historical case; no novelty authority was granted.', '根据历史案例重建技术可执行方向；未授予新颖性权限。'), artifactResource('idea_shortlist.json', tr('Open the idea adjudication', '打开 Idea 裁决'))),
        tool('literature', 'easyicu_inspect_literature', tr('Read the source receipt: nine curated seeds, no historical retrieval search, and no PRISMA flow.', '读取源回执：9 条人工种子，历史运行未执行检索，也没有 PRISMA 流程。'), artifactResource('literature_evidence.json', tr('Open historical and current literature audit', '打开历史与当前文献审计')), allLiterature),
        tool('readiness', 'easyicu_inspect_scientific_readiness', tr('Separated technical feasibility from idea, scientific, and publication readiness.', '将技术可行性与 Idea、科学及投稿就绪度分开。'), artifactResource('scientific_readiness.json', tr('Open scientific readiness findings', '打开科学就绪度 findings'))),
        pipeline('feasibility', tr('Checked prepared-field feasibility', '已检查准备后字段可行性'), tr('The prepared subset contains the required fields; full source-population scope remains open.', '准备子集包含所需字段；完整来源人群范围仍未闭合。')),
      ]),
      message('demo-assistant-1', 'assistant', tr(
        '**I found three technically executable directions, but none is yet a reliable publication idea.** The historical source used nine curated references and did not run prior-art retrieval. A current audit found 2025–2026 SOFA-2 development and MIMIC-IV mortality studies that directly weaken the novelty of direction 2.\n\nWe can continue direction 2 only as an engineering validation case, not as a paper claim.',
        '**我找到了 3 个技术可执行方向，但目前没有一个是可靠的投稿 Idea。** 历史来源只用了 9 条人工参考文献，没有执行先前研究检索；当前审计发现了 2025–2026 年 SOFA-2 开发及 MIMIC-IV 死亡研究，直接削弱了方向 2 的新颖性。\n\n可以继续用方向 2 验证工程流程，但不能把它当成论文论断。',
      )),
      message('demo-user-2', 'user', tr('Continue with the second one as an engineering validation case. Use MIMIC-IV.', '把第 2 个作为工程验证案例继续，使用 MIMIC-IV。')),
      activity('demo-activity-2', 9000, 13200, [
        tool('accept', 'easyicu_accept_idea_handoff', tr('Bound the selected engineering case to this Copilot project without granting novelty authority.', '将选定工程案例绑定到 Copilot 项目，但不授予新颖性权限。')),
        tool('setup', 'easyicu_update_study_context', tr('Prepared the cohort, exposure, outcome, window, and adjustment slots.', '准备队列、暴露、结局、时间窗与调整变量配置。')),
      ]),
      message('demo-assistant-2', 'assistant', tr(
        'Engineering setup is ready: **MIMIC-IV prepared ICU stays**, exposure = experimental first-24-hour `sep3_sofa2` sensitivity indicator, outcome = in-hospital death, adjustment = age and sex. This is not standard Sepsis-3 and is not a publication-authorized idea.\n\nShall I materialize the aggregate validation package?',
        '工程配置已准备好：**MIMIC-IV 准备后 ICU stay**，暴露为入 ICU 后 24 小时实验性 `sep3_sofa2` 敏感性指标，结局为院内死亡，调整年龄和性别。该指标不是标准 Sepsis-3，也不是已获投稿授权的 Idea。\n\n是否生成聚合验证数据包？',
      )),
      message('demo-user-3', 'user', tr('Yes. Continue with data extraction.', '可以，继续提取数据。')),
      activity('demo-activity-3', 15000, 23600, [
        tool('extract', 'easyicu_start_extraction', tr('Materialized the registered fields with provenance receipts.', '按已登记字段生成带来源回执的数据包。'), artifactResource('extraction_quality.json', tr('Open data and quality review', '打开数据与质量审阅'))),
        pipeline('denominator', tr('Verified the prepared denominator', '已核验准备后分母'), tr('140 prepared ICU stays; the upstream source population and selection path remain open.', '准备后 140 个 ICU stay；上游来源人群与选择路径仍未闭合。')),
        pipeline('missingness', tr('Audited analysis-variable missingness', '已审计分析变量缺失'), tr('All 140 prepared stays were complete for exposure, outcome, age, and sex.', '准备后的 140 个 stay 在暴露、结局、年龄和性别字段均完整。')),
      ]),
      message('demo-assistant-3', 'assistant', tr(
        'The prepared engineering subset contains **140 ICU stays**; **53/140 (37.9%)** are indicator-positive and all primary fields are complete. This verifies the prepared denominator, not the full MIMIC-IV source population, selection path, or representativeness.\n\nI can prepare an engineering analysis plan, but the data package is not paper-authorized.',
        '准备后的工程子集包含 **140 个 ICU stay**；其中 **53/140（37.9%）** 为指标阳性，主分析字段完整。这只核验了准备后的分母，并未核验完整 MIMIC-IV 来源人群、选择路径或代表性。\n\n可以生成工程分析计划，但该数据包未获论文授权。',
      )),
      message('demo-user-4', 'user', tr('Prepare the analysis plan for this validation case.', '按这个验证案例制定分析计划。')),
      activity('demo-activity-4', 25000, 33600, [
        tool('inspect-lit', 'easyicu_inspect_literature', tr('Compared historical citation roles with the current evidence audit.', '将历史引文作用与当前证据审计进行比较。'), artifactResource('literature_evidence.json', tr('Open the literature audit', '打开文献审计')), allLiterature),
        tool('plan', 'easyicu_run', tr('Loaded the historical 11-step digest-bound plan and stopped before analysis.', '加载历史 11 步摘要绑定计划，并在分析前暂停。'), artifactResource('agent_plan.json', tr('Open the analysis plan', '打开分析计划'))),
        tool('plan-audit', 'easyicu_inspect_scientific_readiness', tr('Flagged missing novelty, source scope, information adequacy, and meaningful sensitivity analysis.', '标记缺失的新颖性、来源范围、信息充分性与有意义的敏感性分析。'), artifactResource('scientific_readiness.json', tr('Open plan-readiness findings', '打开计划就绪度 findings'))),
        pipeline('plan-gate', tr('Stopped for plan review', '已停在计划审阅门'), tr('No model was executed before the user approval shown next.', '在下一条用户批准前没有执行模型。')),
      ]),
      message('demo-assistant-4', 'assistant', tr(
        'The plan is executable, but not publication-complete. It has **6 analysis/audit steps and 5 render-only steps**. With only 15 deaths, it lacks a prespecified information-adequacy decision and meaningful small-sample sensitivity; the complete-case replay is identical because no primary field is missing. Current SOFA-2 prior art was absent from the historical Planner.\n\nApprove only if you want to continue the engineering validation case—not a paper analysis.',
        '计划可以执行，但不具备完整投稿分析。它包含 **6 个分析/审计步骤和 5 个仅绘图步骤**。在只有 15 个死亡事件时，计划缺少预先规定的信息充分性裁决和有意义的小样本敏感性分析；因为主字段无缺失，完整病例复跑与主分析完全相同。历史 Planner 也未获得当前 SOFA-2 先前研究。\n\n仅在你要继续工程验证案例时批准，不能视为论文分析批准。',
      )),
      message('demo-user-5', 'user', tr('Continue the engineering validation only. Keep all scientific and publication claims blocked.', '只继续工程验证，所有科学与投稿论断保持阻断。')),
      activity('demo-activity-5', 35000, 73100, [
        tool('resume', 'easyicu_resume', tr('Resumed the approved engineering plan.', '恢复已批准的工程计划。')),
        pipeline('step-1', tr('1/11 · Built the prepared cohort flow', '1/11 · 构建准备后队列流程'), tr('Prepared denominator remained 140 ICU stays.', '准备后分母保持为 140 个 ICU stay。')),
        pipeline('step-2', tr('2/11 · Generated Table 1', '2/11 · 生成 Table 1'), tr('Summarized age and sex by exposure group.', '按暴露组汇总年龄和性别。'), artifactResource('result_tables.json', tr('Open Table 1 and model tables', '打开 Table 1 与模型表'))),
        pipeline('step-3', tr('3/11 · Estimated prevalence and mortality', '3/11 · 估计比例与死亡率'), tr('53 indicator-positive and 87 indicator-negative stays.', '指标阳性 53 个，阴性 87 个 stay。')),
        pipeline('step-4', tr('4/11 · Completed missingness audit', '4/11 · 完成缺失审计'), tr('Primary complete-case denominator: 140.', '主分析完整病例分母：140。')),
        pipeline('step-5', tr('5/11 · Fit the adjusted model', '5/11 · 拟合调整模型'), tr('Age/sex-adjusted logistic regression completed with 15 outcome events.', '年龄、性别调整 Logistic 回归完成，共 15 个结局事件。')),
        pipeline('step-6', tr('6–10/11 · Rendered estimates and audit figures', '6–10/11 · 生成估计与审计图表'), tr('These are rendering steps, not five additional independent analyses.', '这些是绘图步骤，不是另外 5 个独立分析。'), artifactResource('figure_gallery.json', tr('Open the Agent-produced figure', '打开 Agent 生成的图件'))),
        pipeline('step-11', tr('11/11 · Bound evidence and verified numbers', '11/11 · 绑定证据并核验数值'), tr('Five engineering checks passed; scientific readiness remains blocked.', '5 项工程检查通过；科学就绪度仍被阻断。'), artifactResource('quality_gate.json', tr('Open the evidence gate', '打开证据闸门'))),
        pipeline('science-gate', tr('Scientific/publication gate · blocked', '科学/投稿闸门 · 阻断'), tr('Idea novelty, literature search, source scope, major review, and paper authority remain open.', 'Idea 新颖性、文献检索、来源范围、major review 与论文授权仍未闭合。'), artifactResource('scientific_readiness.json', tr('Open all scientific blockers', '打开全部科学阻断项'))),
        pipeline('results', tr('Prepared aggregate results for review', '已准备聚合结果供审阅'), tr('The run remains analysis-only.', '该运行仍为仅供分析。'), artifactResource('result_summary.json', tr('Open result summary', '打开结果摘要'))),
      ]),
      message('demo-assistant-5', 'assistant', tr(
        '**Engineering analysis complete.** Mortality was **15.1% (8/53)** with the indicator and **8.0% (7/87)** without it; the adjusted odds ratio was **1.50 (95% CI 0.49–4.60)**.\n\nThe estimate is highly imprecise and the run has only 15 events. Five engineering checks passed, but the persisted reviewer recommendation is **major revision** and scientific/publication readiness remains blocked.',
        '**工程分析完成。** 指标阳性组死亡率为 **15.1%（8/53）**，阴性组为 **8.0%（7/87）**；调整 OR 为 **1.50（95% CI 0.49–4.60）**。\n\n估计非常不精确，且运行只有 15 个事件。5 项工程检查通过，但持久化 reviewer 建议为 **major revision**，科学与投稿就绪度仍被阻断。',
      )),
      message('demo-user-6', 'user', tr('Interpret the result and organize it into a manuscript draft.', '帮我解读结果，并整理成论文初稿。')),
      activity('demo-activity-6', 75000, 91800, [
        tool('interpret', 'easyicu_inspect_interpretation', tr('Separated the observed estimate from causal and clinical claims.', '将观察到的估计与因果、临床结论分开。'), artifactResource('result_summary.json', tr('Open evidence-bound interpretation', '打开证据约束的解读'))),
        tool('manuscript', 'easyicu_inspect_manuscript', tr('Organized a bounded draft without upgrading its authority.', '整理有界初稿，但不提升其权限。'), artifactResource('manuscript_draft.json', tr('Open manuscript draft', '打开论文初稿'))),
        pipeline('lock', tr('Scientifically blocked the draft', '已在科学闸门阻断初稿'), tr('Human sign-off alone cannot upgrade this engineering case to a publication.', '仅靠人工签署不能把该工程案例升级为论文。'), artifactResource('scientific_readiness.json', tr('Open publication blockers', '打开投稿阻断项'))),
      ]),
      message('demo-assistant-6', 'assistant', tr(
        'I generated an evidence-bound draft, but it does **not** meet publication requirements. The idea is not differentiated from current SOFA-2 literature; the 140-stay source scope is incomplete; the 15-event analysis is not adequately stress-tested; and the reviewer package remains major revision.\n\nThe correct next step is to mine a genuinely differentiated idea and rebuild the data/plan authority—not to sign or polish this draft into a paper.',
        '我生成了证据绑定初稿，但它**不满足投稿要求**。该 Idea 未与当前 SOFA-2 文献形成差异化；140-stay 来源范围未闭合；15 个事件的分析没有得到充分稳健性检验；reviewer 包仍为 major revision。\n\n正确下一步是重新寻找真正差异化的 Idea，并重建数据与 Plan 权限，而不是签署或润色这份初稿去投稿。',
      )),
    ];
  }

  function workflow() {
    return {
      current_stage: 'manuscript', completed_required_stages: 4, required_stage_count: 8,
      next_action_code: 'new_idea_prior_art_required',
      stages: [
        ['question', 'complete', 'question_bound'],
        ['idea', 'review_required', 'prior_art_authority_not_established'],
        ['setup', 'complete', 'study_setup_complete'],
        ['extraction', 'review_required', 'source_population_scope_open'],
        ['plan', 'review_required', 'publication_analysis_incomplete'],
        ['analysis', 'complete', 'validated_analysis_complete'],
        ['interpretation', 'complete', 'interpretation_complete'],
        ['manuscript', 'blocked', 'paper_authority_not_granted'],
      ].map(([id, status, reason_code]) => ({ id, status, reason_code })),
    };
  }
  function artifact(name) { return clone(artifacts()[String(name || '')] || null); }
  function hasArtifact(name) { return Object.prototype.hasOwnProperty.call(artifacts(), String(name || '')); }
  function artifactLabel(name) {
    const item = artifacts()[String(name || '')];
    return item ? item.title : String(name || '');
  }
  function safeExternalUrl(value) {
    try {
      const parsed = new URL(String(value || ''));
      if (parsed.protocol !== 'https:') return '';
      if (![
        'pubmed.ncbi.nlm.nih.gov', 'github.com', 'doi.org',
        'jamanetwork.com', 'www.icmje.org',
      ].includes(parsed.hostname)) return '';
      return parsed.href;
    } catch (error) { return ''; }
  }
  function safeDemoImage(value) {
    const path = String(value || '');
    return /^\/assets\/demo\/[A-Za-z0-9_.-]+\.png$/.test(path) ? path : '';
  }
  function tableHtml(table) {
    const headers = Array.isArray(table && table.headers) ? table.headers : [];
    const rows = Array.isArray(table && table.rows) ? table.rows : [];
    return `<section class="gpi-demo-table-section">
      <h4>${esc(table && table.label || tr('Evidence table', '证据表'))}</h4>
      <div class="gpi-demo-table-wrap"><table><thead><tr>${headers.map(value => `<th>${esc(value)}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(row => `<tr>${(Array.isArray(row) ? row : []).map(value => `<td>${esc(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    </section>`;
  }
  function citationHtml(citation) {
    const url = safeExternalUrl(citation && citation.url);
    const title = esc(citation && citation.title || citation && citation.key || 'citation');
    const source = [citation && citation.venue, citation && citation.year, citation && citation.pmid ? `PMID ${citation.pmid}` : ''].filter(Boolean).join(' · ');
    return `<article class="gpi-demo-citation" data-citation-key="${esc(citation && citation.key || '')}">
      <div><code>${esc(citation && citation.key || '')}</code><span>${esc(source)}</span></div>
      ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${title}</a>` : `<strong>${title}</strong>`}
      <p>${esc(citation && citation.relevance || '')}</p>
    </article>`;
  }
  function renderArtifact(payload) {
    const item = payload && typeof payload === 'object' ? payload : {};
    const metrics = Array.isArray(item.metrics) ? item.metrics : [];
    const sections = Array.isArray(item.sections) ? item.sections : [];
    const tables = Array.isArray(item.tables) ? item.tables : [];
    const citations = Array.isArray(item.citations) ? item.citations : [];
    const steps = Array.isArray(item.steps) ? item.steps : [];
    const images = Array.isArray(item.images) ? item.images : [];
    const manuscript = Array.isArray(item.manuscript_sections) ? item.manuscript_sections : [];
    const citationByKey = new Map(citations.map(citation => [citation.key, citation]));
    return `<div class="gpi-demo-artifact">
      <div class="gpi-demo-artifact-intro"><strong>${esc(item.title || item.artifact || tr('Demo artifact', '演示产物'))}</strong><p>${esc(item.summary || '')}</p></div>
      ${item.projection_note ? `<div class="gpi-demo-projection-note" role="note"><strong>${esc(tr('Historical projection note', '历史投影说明'))}</strong><span>${esc(item.projection_note)}</span></div>` : ''}
      ${metrics.length ? `<dl>${metrics.map(metric => `<div><dt>${esc(metric.label || '')}</dt><dd>${esc(metric.value || '')}</dd></div>`).join('')}</dl>` : ''}
      ${sections.map(section => `<section><h4>${esc(section.heading || '')}</h4><ul>${(Array.isArray(section.items) ? section.items : []).map(value => `<li>${esc(value)}</li>`).join('')}</ul></section>`).join('')}
      ${citations.length ? `<section class="gpi-demo-citations"><h4>${esc(tr('Inspectable literature records', '可审阅文献记录'))}</h4>${citations.map(citationHtml).join('')}</section>` : ''}
      ${steps.length ? `<section class="gpi-demo-plan"><h4>${esc(tr('Plan steps · input → method → output → literature', '计划步骤 · 输入 → 方法 → 输出 → 文献'))}</h4>${steps.map((step, index) => {
        const keys = Array.isArray(step.citation_keys) ? step.citation_keys : [];
        return `<article><header><span>${String(index + 1).padStart(2, '0')}</span><strong>${esc(step.step_id || '')}</strong><em>${esc(step.method || '')}</em></header><p>${esc(step.intent || '')}</p><div><small>${esc(tr('Inputs', '输入'))}</small><span>${esc((step.inputs || []).join(' · '))}</span></div><div><small>${esc(tr('Outputs', '输出'))}</small><span>${esc((step.outputs || []).join(' · '))}</span></div><footer>${keys.map(key => {
          const citation = citationByKey.get(key);
          const url = safeExternalUrl(citation && citation.url);
          return url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(key)}</a>` : `<code>${esc(key)}</code>`;
        }).join('')}</footer></article>`;
      }).join('')}</section>` : ''}
      ${tables.map(tableHtml).join('')}
      ${images.map(image => {
        const src = safeDemoImage(image && image.src);
        return src ? `<figure class="gpi-demo-figure"><img src="${esc(src)}" alt="${esc(image.alt || '')}"><figcaption>${esc(image.caption || '')}</figcaption></figure>` : '';
      }).join('')}
      ${manuscript.length ? `<article class="gpi-demo-manuscript">${manuscript.map(section => `<section><h3>${esc(section.heading || '')}</h3><p>${esc(section.text || '')}</p></section>`).join('')}</article>` : ''}
    </div>`;
  }

  window.EU_GUIDED_PI_DEMO = {
    messages, workflow, artifact, hasArtifact, artifactLabel, renderArtifact,
    sourceRunId: SOURCE_RUN_ID,
  };
})();
