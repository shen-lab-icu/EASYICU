/* Owner: shared artifact reader vocabulary. */
/* Reader-facing names for the host codes the shared artifact renderers show:
   plan-review findings, result-table columns and titles, method cards and the
   design elements a method source supports.  Text only -- nothing here
   derives, rounds, or reinterprets a registered value. */
(function () {
  'use strict';

  // The renderers' own i18n global decides the language, so a reader never
  // sees a table in one language and its column names in another.
  function pick(pair) {
    return typeof window.t === 'function' ? window.t(pair[0], pair[1]) : (window.EU_LANG === 'zh' ? pair[1] : pair[0]);
  }
  function zh() { return pick(['en', 'zh']) === 'zh'; }

  // [Chinese title, Chinese detail].  English readers keep the host's own
  // finding message, so only the Chinese reading is supplied here.
  const FINDINGS = {
    ACCEPTED_BASELINE_CONTENT_MISSING: ['已接受的基线表内容缺失', '此前已接受的基线表有变量或数据坐标没有保留；需先恢复这些来源绑定的数据再重新规划。'],
    ACCEPTED_BASELINE_MATERIALIZATION_MISSING: ['已接受的基线表数据缺失', '此前已接受的基线表有变量或数据坐标没有保留；需先恢复这些来源绑定的数据再重新规划。'],
    ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND: ['调整变量缺少理由或时间定位', '有协变量缺少混杂理由，或没有证明它在时间零点之前。'],
    APPLICABLE_METHOD_LAYERS_NOT_BOUND: ['方法依据没有覆盖适用环节', '引用的文献没有覆盖时间对齐、相关性、缺失数据、函数形式、解读或报告中适用的环节。'],
    ARTICLE_CONTENT_ROLES_INCOMPLETE: ['文章内容模块不完整', '计划缺少队列、基线、数据质量、描述、主分析或稳健性中应有的证据模块。'],
    CLINICAL_DEFINITION_DATABASE_CONFORMANCE_NOT_ESTABLISHED: ['临床定义与数据库的一致性未确认', '这个临床定义在当前数据库上的实现是否与原定义一致，尚未确认。'],
    CLINICAL_DEFINITION_INDEPENDENT_REVIEW_PENDING: ['临床定义待独立审阅', '临床定义还需独立审阅，之后才能用于正式结论。'],
    CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED: ['连续协变量的函数形式未检查', '连续协变量以线性形式进入模型，但没有可执行的非线性检查（如样条）。'],
    DESCRIPTIVE_INTERVAL_DEPENDENCE_UNRESOLVED: ['描述性区间的独立性未解决', '描述性区间缺少独立单位或患者分组依据；应只报告计数与比例。'],
    DESCRIPTIVE_POPULATION_SCOPE_UNRESOLVED: ['描述性步骤的人群未确定', '描述性风险步骤没有可执行的人群选择，分母无法确定。'],
    DESIGN_ANALOGUE_NOT_BOUND_TO_PRIMARY_PLAN: ['可参照研究没有约束主分析', '已筛到可参照的研究，但它没有约束任何主分析步骤；需写明借鉴或有意不同的设计要素。'],
    DIRECT_COMPARATOR_NOT_BOUND_TO_PRIMARY_PLAN: ['可比研究没有约束主分析', '已筛到直接可比的研究，但它没有约束任何主分析步骤；需写明借鉴或有意不同的设计要素。'],
    DIRECT_COMPARATOR_NOT_ESTABLISHED: ['尚无直接可比研究', '还没有找到与本研究直接可比的已发表研究。'],
    DISTRIBUTION_MISSINGNESS_AUTHORITY_INVALID: ['分布或缺失分析的设定无效', '某个分布或缺失分析步骤的设定不能被执行组件接受。'],
    LITERATURE_DESIGN_ROUTE_NOT_EXPLICIT: ['文献对设计的作用未写明', '引用了文献，但没有写明每篇文献支撑了哪个设计要素。'],
    LITERATURE_RETRIEVAL_NOT_CONDUCTED: ['未进行文献检索', '只有人工种子文献，没有完成可核查的实时检索。'],
    LITERATURE_SEARCH_PROVENANCE_INCOMPLETE: ['文献检索记录不完整', '检索的来源、日期或筛选记录不完整。'],
    LITERATURE_SEARCH_QUERY_NOT_RECORDED: ['检索式未记录', '检索回执没有保存确切的检索式。'],
    METHOD_SOURCE_DESIGN_ELEMENT_UNSUPPORTED: ['方法文献与设计要素不匹配', '有方法文献被绑定到它的方法卡并不支持的设计要素上。'],
    MISSINGNESS_UNEXAMINED_COMPLETE_CASE: ['完整病例分析的假设未检验', '所有模型都只用完整病例，也没有缺失数据敏感性分析。'],
    MODEL_TERM_CODING_CONFLICTS_WITH_DECLARED_DOMAIN: ['变量编码与取值类型冲突', '有分类或有序变量被当作连续变量编码。'],
    NOVELTY_POSITIONING_REVIEW_REQUIRED: ['创新性定位待独立审阅', '已筛到可比研究，但检索与自动筛选不能证明创新性；人群、暴露、时间零点、估计目标、分析路线和临床贡献仍需独立评估。'],
    PHENOTYPING_COMPARISON_CONTRACT_INVALID: ['表型比较的设定无效', '聚类后的描述性比较没有绑定到确切的主聚类队列与分配结果。'],
    PHENOTYPING_FIT_ROSTER_INVALID: ['聚类拟合变量清单无效', '需要把用于聚类拟合的变量和仅供阅读的变量分开声明。'],
    PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE: ['聚类后的结局比较不完整', '没有可执行的步骤覆盖所请求结局的分群比较。'],
    PLANNER_ADJUSTMENT_PROPOSAL_INCOMPLETE: ['调整变量方案不完整', 'Agent 提出的调整变量缺少混杂理由或基线时间证明；由 Agent 修订，不需要你补填。'],
    PLAN_INPUT_STRUCTURALLY_UNAVAILABLE: ['所需变量在数据源中不可得', '计划需要的科学变量在这个数据源中结构性缺失；不会改用别的结局或暴露替代。'],
    PLAN_POPULATION_REQUIREMENT_DRIFT: ['研究人群发生了变化', '描述性结果改变了研究人群；需要在新计划中审阅这一变更。'],
    POPULATION_SCOPE_AMENDMENT_DECLARED: ['研究人群已声明变更', '描述性结果改变了研究人群；需要在新计划中审阅这一变更。'],
    POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED: ['暴露在时间零点之后才判定', '暴露在入 ICU 之后才判定，但计划没有可执行的 landmark 或时变设计来处理暴露机会和早期事件。'],
    PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH: ['暴露定义的时间锚点不一致', '主暴露临床定义的时间锚点与研究声明的时间零点不一致。'],
    PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED: ['暴露定义的时间锚点未核实', '主暴露临床定义的时间锚点是否与研究的时间零点一致，尚未核实。'],
    PRIMARY_POPULATION_EXECUTION_OWNER_MISSING: ['主模型人群没有执行组件', '描述性步骤要求使用主模型人群，但没有执行组件绑定这个人群。'],
    RECENT_DESIGN_ANALOGUE_NOT_ESTABLISHED: ['尚无近期可参照设计', '没有找到近期发表、方法可参照的研究。'],
    RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED: ['尚无近期直接可比研究', '没有找到近期发表的直接可比研究。'],
    REPEATED_STAY_DEDUP_UNDECLARED: ['重复入住的处理规则未声明', '可能存在同一患者多次入住，但没有步骤写明是只取首次入住，还是按患者聚类或混合模型处理。'],
    REPEATED_STAY_IDENTITY_UNAVAILABLE: ['无法识别同一患者的多次入住', '数据源没有给出患者身份，无法排除同一患者多次入住；可以继续做开发性分析，但不能按患者独立处理，也不能作论文结论。'],
    REPEATED_STAY_METHOD_NOT_DECLARED: ['重复入住没有对应的统计方法', '有患者身份，但没有可执行的方法处理同一患者的多次入住；由 Agent 选定方法。'],
    REQUESTED_OUTCOME_COVERAGE_INCOMPLETE: ['部分结局没有分析步骤', '问题里提到的结局并非都有可执行的分析。'],
    REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY: ['要求的敏感性分析只写在文字里', '你要求的敏感性分析缺失，或只有描述、没有可执行步骤。'],
    REVIEWABLE_PLAN_SPECIFICATION_MISSING: ['缺少完整的推荐设定', '所选设计没有给出完整、可审阅的推荐设定。'],
    ROBUSTNESS_AXES_TOO_NARROW: ['稳健性分析维度不足', '可执行的稳健性分析维度少于这类研究的要求。'],
    ROBUSTNESS_DIAGNOSTIC_DISPLAY_MISMATCH: ['稳健性图混入了诊断结果', '稳健性图绑定了不是效应估计的函数形式诊断。'],
    ROBUSTNESS_SPECS_NOT_EXECUTABLE: ['部分稳健性分析没有执行组件', '计划声明的部分稳健性分析没有任何组件会执行。'],
    SCIENTIFIC_STEP_METHOD_SOURCE_NOT_BOUND: ['科学步骤缺少方法依据', '有科学步骤没有引用约束其方法的文献。'],
    TABLE_ONE_INDEPENDENT_TESTS_IGNORE_REPEATED_UNITS: ['基线表检验忽略了重复单位', '队列保留了重复单位，基线表却要求按独立样本做检验。'],
    TIME_VARYING_RUNTIME_UNAVAILABLE: ['时变分析无法执行', '这个数据源或设计目前还不能执行时变分析。'],
    TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED: ['顶刊可比研究检索未完成', '还没有完成针对顶刊可比研究的检索。'],
    UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE: ['主关联没有调整混杂', '主要关联未做调整，只能作描述性解读。'],
  };

  const COLUMNS = {
    // Run and artifact fields.
    run_id: ['Run ID', '运行 ID'],
    run_type: ['Run type', '运行类型'],
    study_id: ['Study ID', '研究 ID'],
    status: ['Status', '状态'],
    mode: ['Mode', '模式'],
    question: ['Question', '问题'],
    local_first: ['Local-first', '本地优先'],
    cohort_size: ['Cohort size', '队列规模'],
    evidence_count: ['Evidence items', '证据项'],
    missing_evidence: ['Missing evidence', '缺失证据'],
    signed: ['Signed', '已签署'],
    provider: ['Provider', 'Provider'],
    database_scope: ['Database scope', '数据库范围'],
    stage: ['Stage', '阶段'],
    population_rule: ['Population rule', '人群规则'],
    excluded_from_previous: ['Excluded', '上一步排除'],
    exposure_value: ['Exposure', '暴露值'],
    reference_exposure_value: ['Reference', '参考值'],
    adjusted_absolute_risk: ['Adjusted risk', '校正后风险'],
    adjusted_odds_ratio: ['Adjusted OR', '校正后 OR'],
    ci_low: ['95% CI low', '95% CI 下限'],
    ci_high: ['95% CI high', '95% CI 上限'],
    standardization_n: ['Standardized N', '标准化样本量'],
    standardization_method: ['Standardization', '标准化方法'],
    estimate_type: ['Estimate', '估计类型'],
    point_estimate: ['Estimate', '估计值'],
    effect_scale: ['Scale', '效应尺度'],
    converged: ['Converged', '已收敛'],
    model_id: ['Model', '模型'],
    spec_id: ['Specification', '规格'],
    // Result-table columns.
    row_role: ['Row type', '行类型'],
    exposure_level: ['Exposure level', '暴露水平'],
    exposure_level_index: ['Level index', '水平序号'],
    n_rows: ['Records', '记录数'],
    exposure_denominator: ['Exposure denominator', '暴露分母'],
    exposure_pct: ['Share (%)', '占比 (%)'],
    exposure_ci_low_pct: ['Share 95% CI low (%)', '占比 95% CI 下限 (%)'],
    exposure_ci_high_pct: ['Share 95% CI high (%)', '占比 95% CI 上限 (%)'],
    exposure_standard_error_pct: ['Share SE (%)', '占比标准误 (%)'],
    exposure_interval_covariance: ['Interval variance', '区间方差类型'],
    exposure_interval_cluster_count: ['Clusters', '聚类数'],
    outcome_observed_n: ['Records with outcome', '有结局记录数'],
    outcome_events: ['Outcome events', '结局事件数'],
    outcome_denominator: ['Outcome denominator', '结局分母'],
    outcome_rate_pct: ['Outcome rate (%)', '结局比例 (%)'],
    outcome_ci_low_pct: ['Rate 95% CI low (%)', '比例 95% CI 下限 (%)'],
    outcome_ci_high_pct: ['Rate 95% CI high (%)', '比例 95% CI 上限 (%)'],
    step_order: ['Step', '步骤'],
    predicate_kind: ['Rule type', '条件类型'],
    concept_id: ['Concept', '概念'],
    resolved_column: ['Column', '列'],
    aggregation: ['Aggregation', '汇总方式'],
    op: ['Operator', '运算'],
    value: ['Value', '取值'],
    n_before: ['Before step', '本步之前'],
    n_excluded: ['Excluded', '排除'],
    n_remaining: ['Remaining', '剩余'],
    event_time_column: ['Event-time column', '事件时间列'],
    event_time_start_hours: ['Event-time start (h)', '事件时间起点（小时）'],
    variable: ['Variable', '变量'],
    fraction_missing: ['Missing fraction', '缺失比例'],
    n_missing: ['Missing', '缺失数'],
    n_unique_non_missing: ['Distinct values', '不同取值数'],
    variable_type: ['Variable type', '变量类型'],
    variable_order: ['Order', '顺序'],
    category: ['Category', '类别'],
    group: ['Group', '分组'],
    group_order: ['Group order', '分组顺序'],
    group_level: ['Group level', '分组水平'],
    denominator_n: ['Denominator', '分母'],
    nonmissing_n: ['Non-missing', '非缺失数'],
    missing_n: ['Missing', '缺失数'],
    missing_pct: ['Missing (%)', '缺失比例 (%)'],
    cohort_n: ['Cohort', '队列人数'],
    group_n: ['Group', '分组人数'],
    group_pct_of_cohort: ['Share of cohort (%)', '占队列 (%)'],
    group_missing_excluded_n: ['Missing group excluded', '分组缺失排除'],
    n_nonmissing: ['Non-missing', '非缺失数'],
    median: ['Median', '中位数'],
    q25: ['25th percentile', '下四分位数'],
    q75: ['75th percentile', '上四分位数'],
    mean: ['Mean', '均值'],
    sd: ['SD', '标准差'],
    analysis_set: ['Analysis set', '分析集'],
    required_variables: ['Required variables', '所需变量'],
    n_total: ['Total', '总数'],
    n_complete: ['Complete', '完整数'],
    n_excluded_missing: ['Excluded for missing values', '因缺失排除'],
    complete_pct: ['Complete (%)', '完整比例 (%)'],
    estimate: ['Estimate', '估计值'],
    fit_status: ['Fit status', '拟合状态'],
    exposure: ['Exposure', '暴露'],
    outcome: ['Outcome', '结局'],
    covariates: ['Covariates', '协变量'],
    estimator_kind: ['Estimator', '估计方法'],
    analysis_role: ['Analysis role', '分析角色'],
    contrast: ['Comparison', '对比'],
    is_primary_contrast: ['Primary comparison', '主要对比'],
    reference_level: ['Reference level', '参照水平'],
    standard_error: ['Standard error', '标准误'],
    variance_estimator: ['Variance estimator', '方差估计'],
    cluster_count: ['Clusters', '聚类数'],
    notes: ['Notes', '备注'],
    n: ['N', '人数'],
    n_events: ['Events', '事件数'],
    p_value: ['P value', 'P 值'],
    p_value_holm: ['P value (Holm)', 'P 值（Holm 校正）'],
    level: ['Level', '水平'],
    in_primary_model: ['In primary model', '进入主模型'],
    test_id: ['Test', '检验'],
    statistic: ['Statistic', '统计量'],
    requirement_id: ['Requirement', '分析需求'],
    eligible_n: ['Applicable', '适用数'],
    not_applicable_n: ['Not applicable', '不适用数'],
    event_present_n: ['Event present', '有事件'],
    event_absent_n: ['Event absent', '无事件'],
    before_origin_n: ['Before time zero', '早于时间零点'],
    value_missing_n: ['Value missing', '值缺失数'],
    value_missing_pct: ['Value missing (%)', '值缺失比例 (%)'],
    indicator_semantics: ['Indicator meaning', '指示含义'],
    missingness_kind: ['Missingness type', '缺失类型'],
    measured_n: ['Measured', '有测量数'],
    measured_pct: ['Measured (%)', '测量比例 (%)'],
    measured_one_n: ['Measured', '有测量数'],
    raw_indicator_one_n: ['Raw indicator = 1', '原始指示为 1'],
    n_stratum: ['Stratum', '分层人数'],
    exposure_variable: ['Exposure variable', '暴露变量'],
    exposure_category: ['Exposure category', '暴露类别'],
    value_column: ['Value column', '取值列'],
    measurement_count_column: ['Measurement-count column', '测量次数列'],
    measurement_total_n: ['Measurements', '测量总次数'],
    measurement_count_median_when_measured: ['Median measurements', '测量次数中位数'],
    measurement_count_max: ['Most measurements', '最多测量次数'],
    repeat_measured_n: ['Measured more than once', '多次测量数'],
    value_present_but_measured_zero_n: ['Value without measurement flag', '有值但测量标记为 0'],
    raw_value_missing_n: ['Raw value missing', '原始值缺失数'],
  };

  // Tested before the renderer's own broader patterns, so a specific audit
  // table is never folded into a generic "measurement" title.
  const TABLES = [
    [/table_one/, 'Baseline characteristics (Table 1)', '基线特征表（Table 1）'],
    [/analytic_denominators/, 'Analysis denominators', '分析分母'],
    [/event_timing_audit/, 'Event-timing audit', '事件时间审计'],
    [/exposure_component_completeness/, 'Exposure component completeness', '暴露分量完整性'],
    [/measurement_process_audit/, 'Measurement-process audit', '测量过程审计'],
    [/measurement_source_audit/, 'Measurement-source audit', '测量来源审计'],
    [/missingness_audit/, 'Missingness audit', '缺失审计'],
    [/distribution_prevalence/, 'Distribution and share by group', '分组分布与占比'],
    [/probe_variable_profile/, 'Variable profile', '变量概况'],
    [/adjusted_association_estimates/, 'Adjusted association estimates', '调整后关联估计'],
    [/adjusted_association_coefficients/, 'Model coefficients', '模型系数'],
    [/key_metrics/, 'Key metrics', '关键指标'],
    [/ordinal_trend|trend_tests?/, 'Ordered trend tests', '有序趋势检验'],
    [/cohort_analysis_flow|exact sequential attrition ledger/, 'Cohort flow (sequential exclusions)', '队列流程（逐步排除）'],
  ];

  const METHOD_CARDS = {
    reporting_observational_study: ['Observational-study reporting', '观察性研究报告规范'],
    reporting_routinely_collected_data: ['Routinely collected data reporting', '常规收集数据的报告规范'],
    time_zero_and_immortal_time: ['Time zero and immortal time', '时间零点与不朽时间偏倚'],
    landmark_analysis: ['Landmark analysis', 'Landmark 分析'],
    proportional_hazards_diagnostics: ['Proportional-hazards diagnostics', '比例风险诊断'],
    restricted_mean_survival_time: ['Restricted mean survival time', '限制性平均生存时间'],
    repeated_units_per_patient: ['Repeated units per patient', '同一患者的重复单位'],
    continuous_covariate_functional_form: ['Functional form of continuous covariates', '连续协变量的函数形式'],
    missing_data_handling: ['Missing-data handling', '缺失数据处理'],
    absolute_and_relative_effects: ['Absolute and relative effects', '绝对与相对效应'],
  };

  const DESIGN_ELEMENTS = {
    adjustment: ['adjustment', '调整变量'],
    dependence: ['dependence', '相关性结构'],
    estimand: ['estimand', '估计目标'],
    exposure: ['exposure', '暴露'],
    missing_data: ['missing data', '缺失数据'],
    outcome: ['outcome', '结局'],
    reporting: ['reporting', '报告规范'],
    robustness: ['robustness', '稳健性'],
    time_zero: ['time zero', '时间零点'],
  };

  function finding(code) {
    const row = FINDINGS[String(code || '')];
    return row && zh() ? { title: row[0], detail: row[1] } : null;
  }
  function column(key) {
    const row = COLUMNS[String(key || '')];
    return row ? pick(row) : '';
  }
  function tableTitle(...tokens) {
    const text = tokens.map(value => String(value || '').toLowerCase()).join(' ');
    const row = TABLES.find(([pattern]) => pattern.test(text));
    return row ? pick([row[1], row[2]]) : '';
  }
  // What a cited method source is used for, from its typed binding rather
  // than the host's English application sentence.
  function citationUse(row) {
    const support = row && row.method_card_support && typeof row.method_card_support === 'object'
      ? row.method_card_support : {};
    const cards = (Array.isArray(support.matched_card_ids) ? support.matched_card_ids : [])
      .map(id => METHOD_CARDS[String(id)]).filter(Boolean).map(pick);
    const elements = (Array.isArray(row && row.design_elements) ? row.design_elements : [])
      .map(id => DESIGN_ELEMENTS[String(id)]).filter(Boolean).map(pick);
    if (!cards.length && !elements.length) return '';
    const chinese = zh();
    const parts = [];
    if (cards.length) parts.push(chinese ? `方法：${cards.join('、')}` : `Method: ${cards.join(', ')}`);
    if (elements.length) parts.push(chinese ? `支撑：${elements.join('、')}` : `Supports: ${elements.join(', ')}`);
    return parts.join(chinese ? '；' : '; ');
  }

  window.AGENT_READER_VOCAB = Object.freeze({ finding, column, tableTitle, citationUse });
})();
