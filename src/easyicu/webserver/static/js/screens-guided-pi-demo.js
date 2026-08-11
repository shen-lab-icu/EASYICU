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

  function artifacts() {
    return {
      'idea_shortlist.json': demoArtifact(
        'idea_shortlist.json',
        tr('Research opportunity shortlist', '研究机会候选'),
        tr('Three feasible MIMIC-IV directions grounded in sepsis literature and prepared EasyICU concept coverage; the historical canary selected an explicitly experimental SOFA-2 sensitivity representation.', '结合脓毒症文献与 EasyICU 已准备概念覆盖形成的 3 个 MIMIC-IV 可行方向；历史 canary 选择的是明确标注为实验性的 SOFA-2 敏感性表征。'),
        [
          { label: tr('Candidates', '候选数'), value: '3' },
          { label: tr('Literature records screened', '筛选文献'), value: '9' },
          { label: tr('Selected direction', '推荐方向'), value: tr('Experimental SOFA-2 sensitivity indicator and hospital death', '实验性 SOFA-2 敏感性指标与院内死亡') },
        ],
        [
          { heading: tr('1 · Descriptive', '1 · 描述性'), items: [tr('Estimate standard Sepsis-3 prevalence with the current SOFA-1-based definition and a visible ICU-stay denominator.', '使用当前基于 SOFA-1 的标准 Sepsis-3 定义估计比例，并明确 ICU stay 分母。')] },
          { heading: tr('2 · Historical canary association study', '2 · 历史 canary 关联研究'), items: [tr('Compare in-hospital mortality with versus without the experimental first-24-hour SOFA-2 sensitivity indicator.', '比较入 ICU 后 24 小时实验性 SOFA-2 敏感性指标有无两组的院内死亡。'), tr('This indicator is not the standard Sepsis-3 definition; prespecify age and sex adjustment and avoid causal language.', '该指标不是标准 Sepsis-3 定义；预先规定年龄与性别调整，不使用因果措辞。')] },
          { heading: tr('3 · Extension', '3 · 延伸方向'), items: [tr('Repeat the same transparent definition in another compatible ICU database.', '在另一兼容 ICU 数据库复现相同的透明定义。')] },
        ],
        'demo_orchestration_reconstructed_from_bound_literature',
      ),
      'literature_evidence.json': demoArtifact(
        'literature_evidence.json',
        tr('Literature search and evidence registry', '文献检索与证据登记'),
        tr('All nine records retained by the E1 engineering-canary LiteratureBundle, with their design role and inspectable source link.', 'E1 工程试跑 LiteratureBundle 保留的 9 条记录，逐条展示设计作用与可打开来源。'),
        [
          { label: tr('Identified', '检出'), value: '14' },
          { label: tr('Duplicates removed', '去重'), value: '5' },
          { label: tr('Screened / included', '筛选 / 纳入'), value: '9 / 9' },
          { label: tr('Patient/result evidence', '患者/结果证据'), value: tr('No · design evidence only', '否 · 仅设计依据') },
        ],
        [
          { heading: tr('Search receipt', '检索回执'), items: [tr('Search was conducted before planning; records were normalized into stable citation keys.', '检索在制定计划前完成；记录被规范为稳定 citation key。'), tr('Literature supports definitions and methods; it does not verify this cohort\'s numeric results.', '文献支持定义与方法，不核验本队列数值结果。')] },
        ],
        SOURCE_AUTHORITY,
        { citations: literatureRecords() },
      ),
      'extraction_quality.json': demoArtifact(
        'extraction_quality.json',
        tr('Data package and quality review', '数据包与质量审阅'),
        tr('Aggregate-only projection of the prepared MIMIC-IV engineering-canary cohort; no patient rows or identifiers are exposed.', '真实 MIMIC-IV 工程试跑队列的仅聚合投影；不展示患者行或标识符。'),
        [
          { label: tr('ICU stays', 'ICU stays'), value: '140' },
          { label: tr('Experimental SOFA-2 sensitivity indicator present', '实验性 SOFA-2 敏感性指标阳性'), value: '53 / 140 (37.9%)' },
          { label: tr('Primary complete cases', '主分析完整病例'), value: '140 / 140' },
        ],
        [
          { heading: tr('Registered fields', '已登记字段'), items: ['sep3_sofa2_max', 'death', 'age', 'sex'] },
          { heading: tr('Time and denominator', '时间与分母'), items: [tr('Exposure window: ICU admission through 24 hours.', '暴露窗：入 ICU 至 24 小时。'), tr('Analysis unit: one prepared row per ICU stay.', '分析单位：每个 ICU stay 一条准备后记录。')] },
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
        tr('Retrospective association analysis using the complete prepared cohort, explicit timing, and a locked robustness replay.', '使用完整准备队列、明确时间锚点与锁定稳健性复跑的回顾性关联分析。'),
        [
          { label: tr('Planned steps', '计划步骤'), value: '11' },
          { label: tr('Primary model', '主模型'), value: tr('Logistic regression', 'Logistic 回归') },
          { label: tr('Adjustment', '调整变量'), value: tr('Age + sex', '年龄 + 性别') },
        ],
        [
          { heading: tr('Primary analysis', '主分析'), items: [tr('Estimate the odds ratio for in-hospital death: experimental SOFA-2 sensitivity indicator present versus absent.', '估计实验性 SOFA-2 敏感性指标阳性对阴性的院内死亡 OR。'), tr('Report group counts, risks, adjusted OR, and 95% confidence interval.', '报告分组人数、风险、调整 OR 与 95% 置信区间。')] },
          { heading: tr('Quality and sensitivity', '质量与敏感性'), items: [tr('Audit variable availability and value missingness before modelling.', '建模前审计变量可得性与数值缺失。'), tr('Replay the same specification in complete cases without changing the estimand.', '在完整病例中按同一设定复跑，不改变 estimand。')] },
          { heading: tr('Literature anchors', '文献依据'), items: [tr('Sepsis-3 supports the clinical framing only and does not validate the experimental SOFA-2 indicator; STROBE and RECORD govern reporting.', 'Sepsis-3 仅支持临床背景，不验证实验性 SOFA-2 指标；报告遵循 STROBE 与 RECORD。')] },
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
          citations: literatureRecords(),
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
        tr('All five engineering checks passed, but the manuscript remains locked for human scientific review.', '5 项工程检查均通过，但稿件仍锁定，等待人工科学审阅。'),
        [
          { label: tr('Execution', '执行'), value: tr('Passed', '通过') },
          { label: tr('Analysis validation', '分析验证'), value: tr('Passed', '通过') },
          { label: tr('Evidence completeness', '证据完整性'), value: tr('Passed', '通过') },
          { label: tr('Numeric verification', '数值核验'), value: tr('Passed', '通过') },
          { label: tr('Reportability', '可报告性'), value: tr('Analysis-only', '仅供分析') },
        ],
        [{ heading: tr('Current gate', '当前闸门'), items: [tr('reportable = false', 'reportable = false'), tr('Human interpretation and sign-off are still required.', '仍需要人工解读与签署。')] }],
      ),
      'manuscript_draft.json': demoArtifact(
        'manuscript_draft.json',
        tr('Evidence-bound manuscript draft', '证据绑定的论文初稿'),
        tr('A structured draft was generated from Agent-produced results and citations, then locked pending human review.', '初稿由 Agent 产出的结果与引用生成，随后锁定等待人工审阅。'),
        [
          { label: tr('Status', '状态'), value: 'locked_pending_human_review' },
          { label: tr('Cohort', '队列'), value: '140 ICU stays' },
          { label: tr('Primary estimate', '主要估计'), value: 'OR 1.50 (95% CI 0.49–4.60)' },
        ],
        [
          { heading: tr('Draft conclusion', '初稿结论'), items: [tr('The experimental early SOFA-2 sensitivity indicator had an imprecise adjusted association with in-hospital mortality in this prepared cohort.', '在该准备队列中，早期实验性 SOFA-2 敏感性指标与院内死亡的调整关联估计不精确。')] },
          { heading: tr('Required author review', '作者必须审阅'), items: [tr('Confirm clinical interpretation, limitations, citation fit, and the analysis-only claim ceiling before any external use.', '对外使用前确认临床解读、局限性、引用匹配和仅供分析的结论上限。')] },
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
    return literatureRecords().filter(item => !allowed || allowed.has(item.key)).map(literatureResource);
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
    const lit = literatureResource();
    const allLiterature = literatureResources();
    return [
      message('demo-user-1', 'user', tr(
        'Help me find a feasible MIMIC-IV study about early sepsis and in-hospital death. Show me the literature and data feasibility before deciding.',
        '帮我从 MIMIC-IV 里找一个关于早期脓毒症和院内死亡、而且数据可做的研究方向。先把文献依据和数据可行性给我看，再决定。',
      )),
      activity('demo-activity-1', 1000, 7200, [
        tool('ideas', 'easyicu_mine_ideas', tr('Compared the scientific question with available ICU concepts.', '将科学问题与可用 ICU 概念做可行性匹配。'), artifactResource('idea_shortlist.json', tr('Open the three research ideas', '打开 3 个研究方向'))),
        tool('literature', 'easyicu_search_literature', tr('Screened and retained nine definition, methods, reporting, and database records.', '筛选并保留 9 篇定义、方法、报告与数据库文献。'), artifactResource('literature_evidence.json', tr('Open all nine literature records', '打开全部 9 篇文献')), allLiterature),
        pipeline('feasibility', tr('Checked MIMIC-IV concept feasibility', '已检查 MIMIC-IV 概念可行性'), tr('The prepared source contains the required exposure, outcome, age, and sex fields.', '准备数据源包含所需暴露、结局、年龄和性别字段。')),
      ]),
      message('demo-assistant-1', 'assistant', tr(
        '**I found three feasible directions.** For this historical engineering walkthrough, I recommend the second: compare in-hospital mortality with versus without the experimental first-24-hour SOFA-2 sensitivity indicator. MIMIC-IV has the required fields and the design can use an explicit denominator and time window.\n\nThis exposure is not standard Sepsis-3. The consensus paper anchors the clinical context only; the plan will use non-causal language and prespecified age/sex adjustment.',
        '**我找到了 3 个可行方向。** 这次历史工程流程演示建议选第 2 个：比较入 ICU 后 24 小时实验性 SOFA-2 敏感性指标阳性与阴性患者的院内死亡。MIMIC-IV 具备所需字段，而且可以明确分母与时间窗。\n\n该暴露不是标准 Sepsis-3。共识文献只用于临床背景；分析会使用非因果措辞，并预先规定年龄与性别调整。',
      )),
      message('demo-user-2', 'user', tr('Choose the second one. Use MIMIC-IV.', '选择第 2 个，使用 MIMIC-IV。')),
      activity('demo-activity-2', 9000, 13200, [
        tool('accept', 'easyicu_accept_idea_handoff', tr('Bound the selected idea to this Copilot project.', '将所选想法绑定到当前 Copilot 项目。')),
        tool('setup', 'easyicu_update_study_context', tr('Prepared the cohort, exposure, outcome, window, and adjustment slots.', '准备队列、暴露、结局、时间窗与调整变量配置。')),
      ]),
      message('demo-assistant-2', 'assistant', tr(
        'Study setup is ready: **MIMIC-IV ICU stays**, exposure = experimental first-24-hour `sep3_sofa2` sensitivity indicator, outcome = in-hospital death, adjustment = age and sex. This is not the standard Sepsis-3 definition.\n\nData extraction is a real data operation. Shall I create the aggregate research package and quality report?',
        '研究配置已准备好：**MIMIC-IV ICU stay**，暴露为入 ICU 后 24 小时实验性 `sep3_sofa2` 敏感性指标，结局为院内死亡，调整年龄和性别。该指标不是标准 Sepsis-3 定义。\n\n数据提取属于真实数据操作。是否生成聚合研究数据包和质量报告？',
      )),
      message('demo-user-3', 'user', tr('Yes. Continue with data extraction.', '可以，继续提取数据。')),
      activity('demo-activity-3', 15000, 23600, [
        tool('extract', 'easyicu_start_extraction', tr('Materialized the registered MIMIC-IV fields with provenance receipts.', '按已登记字段生成带来源回执的 MIMIC-IV 数据包。'), artifactResource('extraction_quality.json', tr('Open data and quality review', '打开数据与质量审阅'))),
        pipeline('denominator', tr('Verified the cohort denominator', '已核验队列分母'), tr('140 ICU stays; no hidden exclusion was applied.', '140 个 ICU stay；没有隐藏排除。')),
        pipeline('missingness', tr('Audited analysis-variable missingness', '已审计分析变量缺失'), tr('All 140 stays were complete for exposure, outcome, age, and sex.', '暴露、结局、年龄和性别在 140 个 stay 中均完整。')),
      ]),
      message('demo-assistant-3', 'assistant', tr(
        'The data package contains **140 ICU stays**. The experimental SOFA-2 sensitivity indicator is present in **53/140 (37.9%)**; all 140 stays are complete for the primary variables. No patient rows or identifiers are shown here.\n\nThe aggregate quality review is ready. I can now prepare the evidence-bound analysis plan.',
        '数据包包含 **140 个 ICU stay**。其中 **53/140（37.9%）** 为实验性 SOFA-2 敏感性指标阳性；主分析变量在 140 个 stay 中均完整。这里不展示患者行或标识符。\n\n聚合质量审阅已完成。接下来可以生成证据绑定的分析计划。',
      )),
      message('demo-user-4', 'user', tr('Prepare the analysis plan for this question.', '按这个问题制定分析计划。')),
      activity('demo-activity-4', 25000, 33600, [
        tool('inspect-lit', 'easyicu_inspect_literature', tr('Mapped the clinical definition and reporting guidance to the scientific steps.', '将临床定义与报告规范映射到科学步骤。'), artifactResource('literature_evidence.json', tr('Open the literature registry', '打开文献登记表')), literatureResources(['singer_sepsis3_2016', 'strobe_2007', 'record_2015', 'sterne_missing_data_2009'])),
        tool('plan', 'easyicu_run', tr('Generated an 11-step digest-bound plan and stopped before analysis.', '生成 11 步摘要绑定计划，并在分析前暂停。'), artifactResource('agent_plan.json', tr('Open the analysis plan', '打开分析计划'))),
        pipeline('plan-gate', tr('Stopped for plan review', '已停在计划审阅门'), tr('No model was executed before the user approval shown next.', '在下一条用户批准前没有执行模型。')),
      ]),
      message('demo-assistant-4', 'assistant', tr(
        'The plan uses one age/sex-adjusted logistic regression for in-hospital death, reports group counts and absolute risks, audits missingness, and repeats the same estimand in a locked complete-case sensitivity analysis.\n\nSepsis-3 is cited for clinical framing only; the experimental SOFA-2 exposure remains explicitly distinct. Reporting choices are tied to STROBE and RECORD. Approve this plan to continue?',
        '计划采用年龄、性别调整的 Logistic 回归分析院内死亡，同时报告分组人数和绝对风险、审计缺失，并用相同 estimand 做锁定的完整病例敏感性分析。\n\nSepsis-3 仅用于临床背景；实验性 SOFA-2 暴露保持明确区分，报告选择绑定 STROBE 与 RECORD。是否批准该计划继续？',
      )),
      message('demo-user-5', 'user', tr('The plan looks good. Continue the analysis.', '计划可以，继续分析。')),
      activity('demo-activity-5', 35000, 73100, [
        tool('resume', 'easyicu_resume', tr('Resumed the approved digest-bound plan.', '恢复已批准的摘要绑定计划。')),
        pipeline('step-1', tr('1/11 · Built the transparent cohort flow', '1/11 · 构建透明队列流程'), tr('Denominator remained 140 ICU stays.', '分母保持为 140 个 ICU stay。')),
        pipeline('step-2', tr('2/11 · Generated Table 1', '2/11 · 生成 Table 1'), tr('Summarized age and sex by exposure group.', '按暴露组汇总年龄和性别。'), artifactResource('result_tables.json', tr('Open Table 1 and model tables', '打开 Table 1 与模型表'))),
        pipeline('step-3', tr('3/11 · Estimated prevalence and mortality', '3/11 · 估计比例与死亡率'), tr('53 indicator-positive and 87 indicator-negative stays.', '指标阳性 53 个，阴性 87 个 stay。')),
        pipeline('step-4', tr('4/11 · Completed missingness audit', '4/11 · 完成缺失审计'), tr('Primary complete-case denominator: 140.', '主分析完整病例分母：140。')),
        pipeline('step-5', tr('5/11 · Fit the adjusted model', '5/11 · 拟合调整模型'), tr('Age/sex-adjusted logistic regression completed.', '年龄、性别调整的 Logistic 回归完成。')),
        pipeline('step-6', tr('6–10/11 · Rendered estimates and sensitivity figures', '6–10/11 · 生成估计与敏感性图表'), tr('Primary estimate, absolute risk, robustness, and data-quality panels.', '主要估计、绝对风险、稳健性和数据质量图。'), artifactResource('figure_gallery.json', tr('Open the Agent-produced figure', '打开 Agent 生成的图件'))),
        pipeline('step-11', tr('11/11 · Bound evidence and verified numbers', '11/11 · 绑定证据并核验数值'), tr('All five engineering checks passed.', '5 项工程检查全部通过。'), artifactResource('quality_gate.json', tr('Open the evidence gate', '打开证据闸门'))),
        pipeline('results', tr('Prepared aggregate results for review', '已准备聚合结果供审阅'), tr('The run remains analysis-only pending human interpretation.', '该运行仍为仅供分析，等待人工解读。'), artifactResource('result_summary.json', tr('Open result summary', '打开结果摘要'))),
      ]),
      message('demo-assistant-5', 'assistant', tr(
        '**Analysis complete.** Mortality was **15.1% (8/53)** with the indicator and **8.0% (7/87)** without it. The age/sex-adjusted odds ratio was **1.50 (95% CI 0.49–4.60)**.\n\nThe interval includes the null, so this is not conclusive evidence of an association. The run passed five engineering checks but remains **analysis-only** pending human scientific review.',
        '**分析完成。** 指标阳性组死亡率为 **15.1%（8/53）**，阴性组为 **8.0%（7/87）**；年龄、性别调整后的 OR 为 **1.50（95% CI 0.49–4.60）**。\n\n置信区间包含无效值，因此不能认为已经得到明确关联。运行通过 5 项工程检查，但在人工科学审阅前仍为**仅供分析**。',
      )),
      message('demo-user-6', 'user', tr('Interpret the result and organize it into a manuscript draft.', '帮我解读结果，并整理成论文初稿。')),
      activity('demo-activity-6', 75000, 91800, [
        tool('interpret', 'easyicu_inspect_interpretation', tr('Separated the observed estimate from causal and clinical claims.', '将观察到的估计与因果、临床结论分开。'), artifactResource('result_summary.json', tr('Open evidence-bound interpretation', '打开证据约束的解读'))),
        tool('manuscript', 'easyicu_inspect_manuscript', tr('Organized methods, results, interpretation, and limitations with evidence references.', '按证据引用组织方法、结果、解读与局限。'), artifactResource('manuscript_draft.json', tr('Open manuscript draft', '打开论文初稿'))),
        pipeline('lock', tr('Locked the draft for human review', '已锁定初稿等待人工审阅'), tr('No external or paper-ready claim is authorized by this demo.', '该演示未授权任何对外或论文级结论。')),
      ]),
      message('demo-assistant-6', 'assistant', tr(
        'The result suggests a higher point estimate for mortality in indicator-positive stays, but the data are too imprecise to establish a reliable association and cannot support causality. The main limitations are the small single-database cohort, residual confounding, and dependence on the registered 24-hour indicator definition.\n\nI organized the evidence-bound manuscript draft and kept it locked for author review. The next step is clinical/methods review, not automatic publication.',
        '结果的点估计提示指标阳性 stay 的死亡率可能更高，但当前数据不够精确，不能确认稳定关联，更不能支持因果结论。主要局限包括单库小样本、残余混杂，以及对已登记 24 小时指标定义的依赖。\n\n我已整理证据绑定的论文初稿，并保持锁定等待作者审阅。下一步是临床与方法学审阅，不是自动发表。',
      )),
    ];
  }

  function workflow() {
    return {
      current_stage: 'manuscript', completed_required_stages: 7, required_stage_count: 8,
      next_action_code: 'human_review_and_reporting',
      stages: [
        ['question', 'complete', 'question_bound'],
        ['idea', 'complete', 'idea_handoff_accepted'],
        ['setup', 'complete', 'study_setup_complete'],
        ['extraction', 'complete', 'active_export_ready'],
        ['plan', 'complete', 'operator_plan_approved'],
        ['analysis', 'complete', 'validated_analysis_complete'],
        ['interpretation', 'complete', 'interpretation_complete'],
        ['manuscript', 'review_required', 'human_review_required'],
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
      if (!['pubmed.ncbi.nlm.nih.gov', 'github.com', 'doi.org'].includes(parsed.hostname)) return '';
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
