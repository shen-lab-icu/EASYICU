/* Owner: read-only Skill implementation metadata and source views. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;
  const tr = (en, zh) => window.t(en, zh);
  const files = pkg => Array.isArray(pkg && pkg.files) ? pkg.files : [];
  const metadata = pkg => pkg && pkg.implementation && typeof pkg.implementation === 'object' ? pkg.implementation : null;
  const entries = pkg => Array.isArray(metadata(pkg)?.entrypoints) ? metadata(pkg).entrypoints.filter(row => row && typeof row === 'object') : [];
  const modeLabel = mode => ({
    fixed_host: tr('Built-in implementation', '固定内置实现'),
    generated_code: tr('Code generated for each study', '研究代码按任务生成'),
    owner_projection: tr('Data or definition tool', '数据或定义工具'),
    unavailable: tr('Implementation not recorded', '暂未登记实现'),
  }[mode] || tr('Implementation not recorded', '暂未登记实现'));
  const roleLabel = role => ({ entrypoint: tr('Entry point', '主入口'), kernel: tr('Method function', '方法函数'),
    helper: tr('Helper', '辅助函数'), gate: tr('Check', '检查函数') }[role] || tr('Function', '函数'));
  // Reading labels for the existing scientific inputs and products. The full
  // owner contract remains in Files; these labels do not change execution.
  const tableOneGuide = {
    inputs: '每位研究对象一行的数据表；分组列及各组取值；待汇总的变量、单位和汇总方式（如中位数或人数与比例）。',
    outputs: '- `table_one.csv`：各组基线特征，并列出每个变量有记录的人数、缺失人数和实际分母。',
  };
  const rmstGuide = {
    inputs: '每位研究对象的随访时长、事件是否发生，以及共同的随访截止时间 τ；比较两组时还需要分组列。时长与 τ 使用同一单位。',
    outputs: '- `rmst_summary.csv`：各组在 τ 内的平均无事件时间、两组差值及其不确定性区间。\n- 运行记录：保存本次使用的输入与方法。',
  };
  const coxInput = '每位研究对象的随访时长、事件或删失状态、暴露变量或比较分组及调整变量；同时给出随访起点、时间单位和事件定义。';
  const READING_ZH = {
    'cohort-characterization-table-one': tableOneGuide,
    'descriptive.table_one': tableOneGuide,
    'restricted-mean-survival-time': rmstGuide,
    'time_to_event.rmst': rmstGuide,
    'survival-time-to-event': {
      inputs: coxInput,
      outputs: '- `table_one.csv`：分组基线特征。\n- `survival_effect_estimates.csv`：方案中选择的生存效应估计及区间。\n- `survival_ph_diagnostic.csv`：Cox 比例风险假设检查。\n- `survival_analysis_receipt.json`：本次分析的输入与方法记录。',
    },
    'time_to_event.cox_hr': {
      inputs: coxInput,
      outputs: '- 调整后的 Cox 风险比及其不确定性区间。\n- 比例风险假设检查。\n- 本次分析的输入、模型和运行记录。',
    },
    'cross_database.meta_analysis': {
      inputs: '各独立数据库的汇总效应和标准误，使用相同的估计目标、效应尺度与定义，并有各自的审核记录。无需患者级数据或患者标识。',
      outputs: '- `forest.svg`：各数据库效应及合并效应的森林图。\n- 数据库间异质性统计。\n- 每次排除一个数据库后的敏感性分析。\n- 未纳入合并的数据库及原因。',
    },
    'association.adjusted_association': {
      inputs: '含暴露、结局和调整变量的数据表；各变量的编码与参照组，以及采用的回归模型和目标比较。',
      outputs: '- 调整后的暴露与结局关联估计及其不确定性区间。\n- 模型与系数记录，以及对应的森林图。',
    },
    'missingness-measurement-audit': {
      inputs: '研究队列及要检查的变量，保留缺失标记；检查测量过程时还需测量时间、可观察窗口和对应人数。',
      outputs: '- 各变量有记录、缺失和可测量的人数与比例。\n- 相应的测量审计与队列流转说明。',
    },
    'exposure-outcome-distribution': {
      inputs: '每位研究对象的暴露分组、结局状态及其缺失标记，并说明分组取值和结局定义。',
      outputs: '- 各暴露组的结局人数、实际分母、发生比例与不确定性区间。\n- 对应的描述性结果记录。',
    },
    'adjusted-association-model': {
      inputs: '含暴露、结局和调整变量的数据表，以及变量编码、参照组、目标比较和选用的回归模型。',
      outputs: '- 调整后的效应估计与不确定性区间，以及相应的模型与结果记录。',
    },
    'ordinal-dose-response': {
      inputs: '至少三级有序暴露、二元结局和调整变量；同时说明各级顺序、参照组及主要比较。',
      outputs: '- 有序暴露的调整后趋势估计、各级比较及相应结果记录。',
    },
    'landmark-categorical-association': {
      inputs: '含事件时间、观察时长、分类暴露和调整变量的数据表；明确 landmark 时间点、此前的暴露分组与参照组。',
      outputs: '- landmark 后各暴露组的调整后关联估计及区间，以及本次分析记录。',
    },
    'landmark-spline-dose-response': {
      inputs: '含事件时间、观察时长、连续暴露和调整变量的数据表；明确 landmark、暴露窗口、样条结点及参考值。',
      outputs: '- landmark 后连续暴露的剂量反应曲线、效应区间及本次分析记录。',
    },
    'time-varying-cox': {
      inputs: '按研究对象和起止时间排列的区间长表，记录各区间的暴露、事件与随访状态；另需基线调整变量及时间单位。',
      outputs: '- 随时间更新暴露的风险比、区间、模型诊断和运行记录。',
    },
    'treatment-patterns': {
      inputs: '按 ICU 入住整理的分组、药物类别使用状态、首次用药时间、ICU 随访时长及死亡或存活出科状态；同时说明药物类别的记录覆盖范围。',
      outputs: '- 各组药物类别的使用人数与比例。\n- 已用药者的首次用药时间中位数与四分位距。\n- 将死亡和存活出科作为竞争事件的首次用药累积发生率。',
    },
  };
  const FAMILY_INPUT_ZH = {
    data_preparation: '待整理的数据表、各列的含义与单位。涉及时间窗时还需原始测量时间；涉及模型预处理时还需训练与验证分区。具体使用哪些列和处理规则由本方法的完整计划确定。',
    descriptive: '研究队列的数据表、变量含义与单位；按组描述时还需分组信息。描述测量过程时保留测量时间、缺失标记和相应人数。具体列与汇总规则由完整计划确定。',
    association: '包含暴露、结局及调整变量的分析数据表，以及变量编码与目标比较。重复测量方法还需研究对象标识与时间列；仅汇总既有模型的方法使用已完成的估计与诊断。具体输入由完整计划确定。',
    time_to_event: '研究对象的观察时长、事件或删失状态、分组及所需调整变量，并说明时间起点与单位。竞争风险方法还需区分目标事件和竞争事件；模型诊断使用相应模型及其分析数据。具体配置由完整计划确定。',
    prediction: '训练模型时需要特征、结局和数据分区；评估已有模型时需要同一批病例的预测值与真实结局。解释模型还需已拟合模型和对应特征，动态预测还需测量时间。各方法具体采用的数据形态由完整计划确定。',
    causal_emulation: '估计因果效应时需要治疗或暴露、结局、治疗前协变量及相关时间信息。诊断协变量平衡、重叠或倾向评分时，按所选分支使用治疗标记与协变量，或已有倾向评分、权重及配对结果。纵向效应方法还需按对象与时间排列的记录。具体输入、比较和识别假设由完整计划确定。',
    phenotyping: '发现亚型时使用研究对象的特征矩阵；轨迹方法使用含对象标识和时间的重复测量。描述或验证已确定亚型时使用既有类别标记、分配结果与所需结局。具体特征和时间窗由完整计划确定。',
    cross_database: '各数据库已经得到的汇总估计、标准误和估计目标说明，并记录来源与各自的审核结果。合并前核对效应尺度与定义是否一致，无需患者级数据。',
  };
  const METHOD_INPUT_ZH = {
    unit_harmonisation: '要查看的概念和数据库，以及数据源已记录的单位与换算状态。本工具只读查看这些定义与状态。',
    window_aggregation: '带研究对象标识、测量值和原始时间的数据表，以及相应变量的时间窗与汇总规则。',
    mixed_effects: '每位研究对象有多次记录的长表，包含对象标识、测量时间、连续结局和所需解释变量。',
    gee: '每位研究对象有多次记录的长表，包含对象标识、测量时间、二元或连续结局和所需解释变量。',
    missingness_audit: '待审计的变量或测量产物、缺失与测量可用性标记、相应时间窗，以及研究队列和可观察对象的人数。',
    mixed_effects_growth: '包含对象标识、测量时间、连续测量及所需解释变量的重复测量长表；另保留随访及死亡脱落信息。',
    multiple_testing: '属于同一组比较的原始 P 值，以及要控制的比较范围和校正规则。',
    evalue: '已估计的效应、其置信区间与效应尺度；使用 OR 时还需该队列实际的结局发生率。无需重新输入患者级数据。',
    robustness_panel: '汇总既有变体时需要同一研究在不同模型或队列规格下的效应估计、区间和规格说明；按已确定规格重放主模型时，还需原分析数据、主模型及相应规格。',
    multiple_adjustment: '同一分析人群的暴露、结局，以及粗模型、最小调整和完整调整所用的协变量集合。',
    count_outcome: '研究对象的事件次数、暴露或观察时长、主要暴露与调整变量，以及计数模型和过度离散的处理选择。',
    proportional_odds: '有序分类结局、各结局等级的明确顺序，以及暴露和调整变量。',
    free_days: '观察期限内的支持治疗记录、死亡与随访信息，用于推导无支持天数；另需暴露和调整变量。',
    mediation: '暴露、中介变量、结局和相关协变量，以及它们的时间顺序与相应识别假设。',
    cluster_sizes: '唯一研究对象清单、已冻结的类别分配表、预期类别名单及缺失标签策略；轨迹分型还需对应冻结回执。',
    covariate_balance: '每个研究对象一行的队列表，含唯一对象标识、0/1 治疗标记、预设连续或分类协变量；已有权重为可选输入。',
    outcome_by_cluster: '已确定的类别标记，与同一批研究对象的临床变量、结局和缺失标记。',
    cluster_stability: '原聚类所用的特征、类别标记和模型配置，以及相应重采样方案。',
    trajectory_cluster_stability: '已选轨迹模型、原始重复测量与类别分配，以及对应的重采样和重拟合配置。',
    early_subtype_assignment: '已确定的亚型标记、轨迹观察窗之前的临床特征，以及独立的训练与验证分区。',
    dynamic_prediction: '含对象标识和测量时间的纵向记录，预测时点与回看窗口、结局和随访时间，以及按对象划分的开发与验证集。',
    external_validation: '已拟合的开发模型及其预处理方式，以及独立验证数据库中含同一概念特征和结局的数据。',
    reclassification: '同一批病例的真实二元结局、基线模型和更新模型各自的预测概率；分层重分类另需临床风险阈值。',
    discrimination_calibration: '拟合新模型时需要开发与验证数据的特征、结局、按研究对象划分的数据分区及预处理方式；评价已有预测时需要逐病例配对的真实结局与预测概率。',
    calibration_metrics: '同一批病例的真实结局和模型预测概率。',
    delong_ci: '同一批病例的真实二元结局和模型评分；比较两模型时保留逐病例配对的两组评分。',
    decision_curve: '同一批病例的真实二元结局、预测概率，以及要比较的临床阈值范围。',
    threshold_metrics: '同一批病例的真实结局和预测概率，以及要评价的决策阈值。',
    subgroup_fairness: '同一份冻结二分类预测、真实结局、预设验证集和闭合亚组标记；按唯一对象标识连接。',
  };
  const METHOD_OUTPUT_ZH = {
    outlier_rules: '异常值规则及移除或截尾数量', window_aggregation: '时间窗、汇总方式及测量数量记录',
    unit_harmonisation: '各数据库已有单位换算状态', missing_data: '合并后的系数、组内与组间方差、自由度、缺失信息比例和汇总诊断',
    scaling_coding: '模型设计列与标准化、编码选择记录', descriptive_summary: '队列与变量分布汇总表及相应图件',
    missingness_audit: '逐变量缺失、覆盖及测量可用性汇总；声明敏感性比较时包含其结果',
    mixed_effects: '重复测量效应、区间、方差与诊断及观察到的时间变化', gee: '重复测量平均效应、稳健标准误、诊断及时间变化',
    ordinal_trend: '有序暴露的趋势估计、剂量反应结果与森林图', proportional_odds: '共同优势比及区间、比例优势检验和有序分布图',
    free_days: '无支持天数的推导审计、分布和有序结局关联', count_outcome: '发生率比及区间、离散检验和暴露时间记录',
    multiple_adjustment: '不同调整模型的效应对照表', effect_modification: '交互检验、亚组效应与森林图',
    multiple_testing: '校正后的 P 值表', rcs_spline: '非线性暴露曲线及其不确定性区间', evalue: '未测混杂强度的 E-value，以及置信区间中最接近无效值的端点对应值',
    robustness_panel: '不同分析规格下的效应稳定性汇总', km_logrank: '分组生存曲线数据与 log-rank 检验',
    ph_check: '各协变量的 Schoenfeld 检验结果与相关诊断', subgroup_hr: '亚组风险比、交互结果与森林图',
    competing_risks_cif: '分组累积发生率、Gray 检验、原因别与亚分布风险比及运行记录',
    discrimination_calibration: 'AUROC、校准曲线和 ROC 图', calibration_metrics: '校准截距、校准斜率与 Brier 分数',
    delong_ci: 'AUROC 及置信区间；比较两模型时含配对差异检验', benchmark_comparison: '同批病例上模型与现有评分的区分度对照；适用时包含校准比较',
    decision_curve: '各临床阈值下模型、全治与全不治策略的净获益', threshold_metrics: '指定阈值下的敏感度、特异度、预测值和混淆矩阵',
    feature_attribution: '特征贡献或重要性汇总及相应图件', subgroup_fairness: '各预设亚组的区分度、校准及 Brier 点估计、分母、稀疏标记与无法估计的原因；不作规范性公平判断',
    internal_validation: '内部验证后的模型表现及乐观偏倚估计', conformal_intervals: '预测集合或区间的覆盖率汇总',
    external_validation: '分数据库的外部验证表现、校准、决策曲线及病例构成；再校准结果单列',
    reclassification: '基线与更新模型的 NRI、IDI 及重采样区间', dynamic_prediction: '各预测时点的风险预测表现',
    iptw_or: '因果效应估计表及相应目标试验与识别假设说明', covariate_balance: '加权前后协变量标准化差异、有效样本量、缺失分母和来源回执；不估计治疗效应',
    positivity_overlap: '倾向评分分布、极端权重与截断记录', propensity_adjustment: '倾向评分、匹配对或稳定权重及平衡表',
    negative_control: '阴性对照比较结果', doubly_robust: '双重稳健平均处理效应、标准误、区间与重叠诊断',
    mediation: '直接、间接和总效应及重采样区间', g_methods: '纵向或单时点效应估计、权重诊断与运行记录',
    cluster_solution: '类别分配、稳定性与特征热图及类别结局图', k_selection: '候选类别数对应的选择指标曲线',
    cluster_stability: '聚类稳定性报告', trajectory_cluster_stability: '重采样重拟合记录、调整 Rand 指数、对齐类别分配及冻结决定',
    cluster_sizes: '各冻结类别的规模与比例、空类和未分配对象；按声明阈值标记近空类，保留来源回执', outcome_by_cluster: '各类别临床与结局汇总、真实分母和缺失人数',
    trajectory_feature_clustering: '轨迹特征表示、类别分配、稳定性及类别规模检查与类别结局比较',
    early_subtype_assignment: '验证集亚型概率、分类表现、校准与混淆及系数表', mixed_effects_growth: '增长曲线固定效应、随机效应协方差、ICC 与观察时间曲线',
  };
  function familyReading(pkg, skill) {
    const id = String(pkg && pkg.skill_id || skill.name || '');
    const family = skill.methodFamily || id.split('.')[0];
    if (!FAMILY_INPUT_ZH[family]) return null;
    const key = skill.methodKey || id.split('.')[1];
    // PH's legacy catalog filename differs from its runtime product; retain
    // the diagnostic meaning here and leave exact files to the run's receipt.
    const names = key === 'ph_check' ? [] : [...new Set((skill.outputs || []).flatMap(value => String(value).match(/[A-Za-z0-9_.-]+\.(?:csv|tsv|json|svg|pdf|png|md|parquet)\b/g) || []))];
    const result = METHOD_OUTPUT_ZH[key] || '该方法的结果内容尚缺中文阅读说明，请查看文件中的已登记产物要求。';
    return { inputs: METHOD_INPUT_ZH[key] || FAMILY_INPUT_ZH[family],
      outputs: `- ${result}。${names.length ? `\n- 已登记文件：${names.map(name => '`' + name + '`').join('、')}。` : ''}` };
  }
  function defaultPath(pkg) {
    const preferred = pkg && pkg.overview_paths && pkg.overview_paths[window.EU_LANG === 'zh' ? 'zh' : 'en'];
    return (files(pkg).find(file => file.path === preferred) || files(pkg).find(file => file.path === 'SKILL.md') || files(pkg)[0] || {}).path || '';
  }
  function guideSection(markdown, headings) {
    const sections = String(markdown || '').split(/^##\s+/m).slice(1);
    for (const section of sections) {
      const [title, ...body] = section.split('\n');
      if (headings.includes(title.trim().toLowerCase())) return body.join('\n').trim();
    }
    return '';
  }
  function overview(pkg, skill, renderMarkdown) {
    const main = files(pkg).find(file => file.path === defaultPath(pkg));
    const markdown = String(main && main.content || '');
    const reading = window.EU_LANG === 'zh' && (READING_ZH[pkg && pkg.skill_id] || familyReading(pkg, skill));
    const inputs = reading && reading.inputs || guideSection(markdown, ['输入', '所需输入', 'inputs'])
      || (markdown.match(/^(?:输入[：:]|Inputs:)\s*(.+)$/m) || [])[1];
    const outputs = reading && reading.outputs || guideSection(markdown, ['输出', '产出', 'outputs', 'expected outputs']);
    const fallback = values => values && values.length ? `<ul>${values.map(value => `<li>${esc(value)}</li>`).join('')}</ul>` : `<p>${tr('See the method guide in Files.', '详见文件中的方法说明。')}</p>`;
    return `<section class="eusk-quick-guide" data-sk-overview-source="${esc(main && main.path || '')}"><h2>${tr('Purpose', '用途')}</h2><p>${esc(skill.description || '')}</p><div class="eusk-quick-io"><div><h2>${tr('Required input', '所需输入')}</h2>${inputs ? renderMarkdown(inputs) : fallback(skill.inputs)}</div><div><h2>${tr('Outputs', '输出')}</h2>${outputs ? renderMarkdown(outputs) : fallback(skill.outputs)}</div></div></section>`;
  }
  function summary(pkg) {
    const impl = metadata(pkg);
    const mode = impl && impl.mode || 'unavailable';
    const text = impl && tr(impl.summary_en || impl.summary_zh || '', impl.summary_zh || impl.summary_en || '');
    return `<div class="eusk-implementation-summary" data-sk-implementation-mode="${esc(mode)}"><span class="eusk-implementation-mode">${esc(modeLabel(mode))}</span>${text ? `<p>${esc(text)}</p>` : `<p>${tr('No function metadata is available for this package. The complete guide is in Files.', '此技能包暂未提供函数元数据，完整说明仍可在文件中查看。')}</p>`}${mode === 'generated_code' ? `<p>${tr('The implementation functions support or check generated code. The study-specific analysis code is created with the task.', '实现函数用于辅助或检查生成代码；本次研究的分析代码会随任务生成。')}</p>` : ''}</div>`;
  }
  function primaryIndices(pkg) {
    const rows = entries(pkg);
    const ranks = metadata(pkg)?.mode === 'generated_code'
      ? { entrypoint: 0, kernel: 1, helper: 2, gate: 3 }
      : { entrypoint: 0, kernel: 1, gate: 2, helper: 3 };
    return rows.map((row, index) => index).sort((a, b) => {
      const rank = row => ranks[row.role] ?? 4;
      return rank(rows[a]) - rank(rows[b]);
    });
  }
  function entryButton(entry, index, selected = false) {
    return `<button type="button" class="eusk-function-link${selected ? ' active' : ''}" data-sk-function="${index}"${selected ? ' aria-current="true"' : ''}><span>${esc(roleLabel(entry.role))}</span><strong>${esc(entry.qualname || entry.name || '')}</strong><code>${esc(entry.signature || '')}</code></button>`;
  }
  function preview(pkg) {
    const rows = entries(pkg);
    const helpers = metadata(pkg)?.mode === 'generated_code' && !rows.some(row => ['entrypoint', 'kernel'].includes(row.role));
    return `<section class="eusk-implementation-preview"><h2>${helpers ? tr('Reusable helper functions', '可复用辅助函数') : tr('Implementation / functions', '实现 / 函数')}</h2>${summary(pkg)}<button type="button" class="eusk-open-implementation" data-sk-tab="implementation">${tr('Inspect implementation and source', '查看实现与源码')} →</button></section>`;
  }
  // A host file contains separate excerpts, each bound to its original lines.
  // Their marker lines are display metadata and never receive invented line numbers.
  function sourceRows(file) {
    const raw = String(file && file.content || '').split('\n');
    if (!(file && (file.role === 'host_implementation' || String(file.path || '').startsWith('implementation/')))) {
      return raw.map((text, index) => ({ text, line: index + 1 }));
    }
    let line = null;
    let end = 0;
    return raw.map(text => {
      const marker = /^# (src\/[^:]+):(\d+)-(\d+) sha256=[a-f0-9]+$/.exec(text);
      if (marker) { line = Number(marker[2]); end = Number(marker[3]); return { text, line: null }; }
      const row = { text, line: line !== null && line <= end ? line : null };
      if (line !== null) line += 1;
      return row;
    });
  }
  function sourceFor(pkg, entry) {
    const file = files(pkg).find(row => row.path === entry.file_path);
    if (!file) return null;
    if ((file.source_path && file.source_path !== entry.source_path)
      || (file.source_sha256 && entry.source_sha256 && file.source_sha256 !== entry.source_sha256)) return null;
    const start = Number(entry.line_start);
    const end = Number(entry.line_end);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) return null;
    const excerptStart = Number(entry.excerpt_line_start);
    const excerptEnd = Number(entry.excerpt_line_end);
    if (Number.isInteger(excerptStart) && Number.isInteger(excerptEnd) && excerptStart > 0 && excerptEnd >= excerptStart) {
      const raw = String(file.content || '').split('\n');
      if (excerptEnd <= raw.length && excerptEnd - excerptStart === end - start) {
        return { file, rows: raw.slice(excerptStart - 1, excerptEnd).map((text, index) => ({ text, line: start + index })) };
      }
    }
    if (file.source_path !== entry.source_path) return null;
    const rows = sourceRows(file).filter(row => row.line !== null && row.line >= start && row.line <= end);
    return rows.length === end - start + 1 ? { file, rows } : null;
  }
  function highlighted(text, language) {
    if (!['python', 'r', 'javascript'].includes(String(language || '').toLowerCase())) return esc(text);
    const token = /(#.*$|\/\/.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:async|await|def|class|return|if|else|elif|for|while|in|import|from|as|with|raise|try|except|finally|yield|None|True|False|function|NULL|TRUE|FALSE|const|let|new)\b)/g;
    let cursor = 0;
    let html = '';
    for (const match of String(text).matchAll(token)) {
      html += esc(text.slice(cursor, match.index));
      const kind = /^(#|\/\/)/.test(match[0]) ? 'comment' : /^["']/.test(match[0]) ? 'string' : 'keyword';
      html += `<span class="eusk-syntax-${kind}">${esc(match[0])}</span>`;
      cursor = match.index + match[0].length;
    }
    return html + esc(text.slice(cursor));
  }
  function codeHtml(rows, language) {
    return `<pre class="eusk-source-code" tabindex="0" aria-label="${tr('Read-only source code', '只读源代码')}"><code>${rows.map(row => `<span class="eusk-source-line"${row.line === null ? '' : ` data-source-line="${row.line}"`}><span class="eusk-line-number" aria-hidden="true">${row.line === null ? '' : row.line}</span><span>${highlighted(row.text, language) || ' '}</span></span>`).join('')}</code></pre>`;
  }
  function filePreview(file) { return codeHtml(sourceRows(file), file.language); }
  function render(pkg, selected) {
    const rows = entries(pkg);
    const index = Number.isInteger(selected) && rows[selected] ? selected : primaryIndices(pkg)[0];
    const entry = rows[index];
    const source = entry && sourceFor(pkg, entry);
    const examples = Array.isArray(metadata(pkg)?.examples) ? metadata(pkg).examples : [];
    return `${summary(pkg)}${entry ? `<div class="eusk-implementation-layout"><nav class="eusk-function-list" aria-label="${tr('Implementation functions', '实现函数')}">${primaryIndices(pkg).map(i => entryButton(rows[i], i, i === index)).join('')}</nav><section class="eusk-function-detail"><div class="eusk-function-heading"><span>${esc(roleLabel(entry.role))}</span><h2>${esc(entry.qualname || entry.name)}</h2><code>${esc(entry.signature || '')}</code>${entry.docstring ? `<p class="eusk-function-docstring">${esc(entry.docstring)}</p>` : ''}<p class="eusk-source-location">${esc(entry.source_path || '')}:${esc(entry.line_start)}–${esc(entry.line_end)}<br><span>sha256:${esc(entry.source_sha256 || '')}</span></p>${source ? `<button type="button" data-sk-package-file="${esc(source.file.path)}">${tr('Open source file', '打开源文件')} ↗</button>` : ''}</div>${source ? `<p class="eusk-file-kind">${tr('Read-only source excerpt · original line numbers', '只读源码摘录 · 原始行号')}</p>${codeHtml(source.rows, source.file.language)}` : `<p class="eusk-source-missing">${tr('This function’s source excerpt could not be located. The package files remain available.', '未能定位此函数的源码片段，可继续查看技能包文件。')}</p>`}</section></div>` : ''}${examples.length ? `<section class="eusk-call-examples"><h2>${tr('Call examples (display only)', '调用示例（仅展示）')}</h2>${examples.map(example => `<h3>${esc(tr(example.label_en || example.path || '', example.label_zh || example.label_en || example.path || ''))}</h3>${codeHtml(String(example.code || '').split('\n').map((text, i) => ({ text, line: i + 1 })), example.language)}`).join('')}</section>` : ''}`;
  }
  window.EU_SKILL_IMPLEMENTATION = { defaultPath, overview, preview, render, filePreview };
})();
