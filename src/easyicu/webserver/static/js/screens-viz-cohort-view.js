/* Cohort Statistics rendering owner. State and effects stay in screens-viz-cohort.js. */
(function () {
  let state;
  let survival;
  let demo;
  let t;
  let icon;
  let esc;
  let fmtInt;
  let fmtNum;
  let fmtPct;
  let fmtP;
  let workspaceSamplingNote;
  let cohortCharts;
  let cohortReview;
  const { demoCatalogModules, demoRowsForModule } = window.VIZ_DEMO;

  /* In-page panels (aligned with cohort_redesign.py _SUBTABS): Group contrast,
     Coverage audit, Cohort profile, SOFA reclassification. Demo-only panels use
     fixed catalog-shaped previews; real mode still requires a cohort-review payload. */
  function cohortTabs() {
    const tabs = [
      ['groups',   t('Group contrast', '组间对照'),       'layers'],
      ['survival', t('Survival curves', '生存曲线'),       'chart'],
      ['coverage', t('Coverage audit', '覆盖审计'),      'shield'],
      ['snapshot', t('Cohort profile', '队列画像'),       'cohort'],
      ['sofa',     t('SOFA reclassification', 'SOFA 重分层'), 'refresh'],
    ];
    return `<div class="tabs" id="cohtabs">${tabs.map(([k, lab, ic]) =>
      `<button class="tab ${state.panel === k ? 'active' : ''}" data-cohtab="${k}">${icon(ic, 14)} ${lab}</button>`).join('')}</div>`;
  }

  function cohortPanelBody() {
    if (cohortCharts && typeof cohortCharts.begin === 'function') cohortCharts.begin();
    const review = cohortReview();
    const demoLoaded = window.EU_DATA !== 'real' && state.view === 'loaded';
    switch (state.panel) {
      case 'survival': return review ? survival.body(review) : survival.demoBody();
      case 'coverage': return review ? cohortCoverageBody(review) : (demoLoaded ? cohortCoverageBody(demo.coverageReview(), { demo: true }) : cohortUnavailablePanel('coverage'));
      case 'sofa':     return review ? cohortSofaBody(review) : (demoLoaded ? cohortSofaBody(demo.sofaReview(), { demo: true }) : cohortUnavailablePanel('sofa'));
      case 'snapshot': return cohortSnapshotBody();
      default:         return cohortGroupsBody();
    }
  }

  function cohortUnavailablePanel(kind) {
    const isSofa = kind === 'sofa';
    return `
      <div class="state empty mt-16">
        <div class="ico">${icon(isSofa ? 'refresh' : 'shield', 18)}</div>
        <div class="t">${isSofa ? t('SOFA reclassification requires a real cohort review', 'SOFA 重分层需要真实队列审阅') : t('Coverage audit requires a real cohort review', '覆盖审计需要真实队列审阅')}</div>
        <div class="d">${t('The old seeded audit panel has been removed. Switch to Real mode, register an EasyICU export, then load Cohort Statistics to compute this aggregate-only panel.', '旧的种子审计面板已移除。请切换到真实模式，注册 EasyICU 导出，再加载队列统计以计算这个仅聚合的面板。')}</div>
      </div>`;
  }

  function cohortProfileValue(row, value) {
    if (value == null || value === '') return '—';
    if (row.kind === 'count') return fmtInt(value);
    if (row.kind === 'percent') return fmtPct(value);
    return fmtNum(value, 1);
  }
  function cohortText(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      'Workspace': '工作区',
      'Local export': '本地导出',
      'Demo cohort': '演示队列',
      'Not configured': '未配置',
      'Cohort statistics': '队列统计',
      'Agent preflight': 'Agent 预检',
      'current session': '当前会话',
      'Input package': '输入包',
      'Backend evidence checks': '后端证据检查',
      'Draft review': '草稿核验',
      'demo concept set': '演示概念集',
      'manifest parsed · denominators previewed · aggregate payload returned': 'manifest 已解析 · 分母已预览 · 聚合载荷已返回',
      'coverage + denominators ready': '覆盖率 + 分母已就绪',
      'locked · requires reviewer sign-off': '已锁定 · 需要审阅者签署',
      'Analysis table': '分析表',
      'Real cohort aggregate': '真实队列聚合',
      'Local export group contrast': '本地导出分组对照',
      'Cohort size': '队列规模',
      'Total stays': '总住院数',
      'Total patients': '总患者数',
      'Mean age': '平均年龄',
      'Median age': '年龄中位数',
      'Female': '女性',
      'Female %': '女性比例',
      'Male %': '男性比例',
      'Mortality': '死亡率',
      'Median SOFA-2': 'SOFA-2 中位数',
      'Median SOFA': 'SOFA 中位数',
      'Sepsis-3 +': 'Sepsis-3 阳性',
      'Local export cohort review ready': '本地导出队列审阅已就绪',
      'Source': '来源',
      'Database': '数据库',
      'Path hash': '路径哈希',
      'Scope': '范围',
      'aggregate-only payload': '仅聚合载荷',
      'Comparison': '对照',
      'Select descriptive split': '选择描述性分组',
      'Summary': '摘要',
      'Overview': '概览',
      'Descriptive profile': '描述性画像',
      'Aggregate-only group characteristics': '仅聚合的分组特征',
      'Metric': '指标',
      'Status': '状态',
      'descriptive': '描述性',
      'Fail-closed': '保守拦截',
      'Blocked cohort functions': '已拦截的队列功能',
      'row_level_filters': '行级筛选',
      'inferential_statistics': '推断统计',
      'matched_cohort': '匹配队列',
      'paired_sofa_reclassification': '配对 SOFA 重分层',
      'custom_threshold': '自定义阈值',
      'p_value_smd': 'p 值 / SMD',
      'blocked': '已拦截',
      'supported': '已支持',
      'Age Groups': '年龄分组',
      'Female vs Male': '女性 vs 男性',
      'Short vs Long Stay': '短住院 vs 长住院',
      'Survived vs Deceased': '存活 vs 死亡',
      'Sepsis vs Non-sepsis': 'Sepsis vs 非 Sepsis',
      'Survived': '存活',
      'Deceased': '死亡',
      'Non-sepsis': '非 Sepsis',
      'Sepsis': 'Sepsis',
      'Female': '女性',
      'Male': '男性',
      'Unknown': '未知',
      'Known': '已知',
      'Short/median': '短住院/中位数以下',
      'Long': '长住院',
      'N': 'N',
      'Mortality %': '死亡率 %',
      'Median ICU LOS': 'ICU 住院时长中位数',
      'years': '年',
      'days': '天',
      'Survival analysis': '生存分析',
      'Kaplan-Meier module': 'Kaplan-Meier 模块',
      'Demo simulated KM preview': '演示模拟 KM 预览',
      'Seeded demo only': '仅 seeded 演示',
      'Demo hospital mortality by Sepsis vs Non-sepsis': '演示院内死亡 · 按 Sepsis vs 非 Sepsis 分组',
      'Demo follow-up days': '演示随访天数',
      'Kaplan-Meier curves and log-rank': 'Kaplan-Meier 曲线与 log-rank',
      'Hospital mortality': '院内死亡',
      'ICU mortality': 'ICU 死亡',
      '28-day mortality': '28 天死亡',
      '30-day display window': '30 天显示窗口',
      '28-day window': '28 天窗口',
      'dedicated flag + follow-up': '专用事件标志 + 随访时间',
      'Hospital LOS / follow-up days': '住院时长 / 随访天数',
      'Outcome': '结局',
      'Outcome overview': '结局概览',
      'Grouping': '分组',
      'events': '事件',
      'No outcome module': '没有结局模块',
      'not available': '不可用',
      'KM-ready': '可画 KM',
      'KM curve endpoint': 'KM 曲线结局',
      'Event rate summary': '事件率',
      'rate only': '仅事件率',
      'time window': '时间窗',
      'unavailable': '不可用',
      'Survival analysis blocked': '生存分析已拦截',
      'Current export is loaded, but the cohort is above the interactive KM preview limit; continue with an audited local analysis job on this same export.': '当前导出已加载，但队列超过交互式 KM 预览上限；请在同一个导出上继续运行本地审计分析任务。',
      'Exploratory · unadjusted': '探索性 · 未调整',
      'Time-to-event': '事件时间',
      'Log-rank': 'Log-rank',
      'df 1 · exploratory only · point estimates, no CI': 'df 1 · 仅探索 · 点估计，无置信区间',
      'not enough events': '事件数不足',
      'Not manuscript-ready by itself': '不能单独用于稿件结论',
      'Number at risk': '风险人数',
      'Group': '分组',
      'Days': '天',
      'Survival probability': '生存概率',
      'Real export required': '需要真实导出',
      'Coverage audit': '覆盖审计',
      'Real module coverage and quality': '真实模块覆盖率与质量',
      'Demo module coverage and quality': '演示模块覆盖率与质量',
      'Modules OK': '正常模块',
      'Watchlist': '观察名单',
      'Median coverage': '覆盖率中位数',
      'Neutral event modules': '事件/暴露模块',
      'Presence-rate modules': '事件/暴露模块',
      'Event/exposure rows show cohort incidence or exposure prevalence, not missingness coverage; they are excluded from the coverage watchlist.': '事件/暴露行显示队列发生率或暴露率，不是缺失覆盖率；它们不会进入低覆盖观察名单。',
      'Unknown coverage': '未知覆盖率',
      'Module': '模块',
      'Records': '记录数',
      'Fields': '字段数',
      'Covered entities': '覆盖实体',
      'Entities': '实体数',
      'Coverage': '覆盖率',
      'Coverage / rate': '覆盖率 / 发生率',
      'Event rate': '发生率',
      'Exposure rate': '暴露率',
      'Fail-closed scope': '保守拦截范围',
      'Interpretation': '解释',
      'Ready': '正常',
      'Watch': '观察',
      'Low coverage': '低覆盖',
      'Rate only': '仅比例',
      'SOFA reclassification': 'SOFA 重分层',
      'SOFA-2 aggregate review': 'SOFA-2 聚合审阅',
      'Demo SOFA-2 aggregate preview': '演示 SOFA-2 聚合预览',
      'Paired entities': '配对实体',
      'SOFA-2 higher': 'SOFA-2 更高',
      'SOFA-2 lower': 'SOFA-2 更低',
      'Median delta': '差值中位数',
      'SOFA-2 minus SOFA-1': 'SOFA-2 减 SOFA-1',
      'Mean SOFA-2': 'SOFA-2 均值',
      'Age': '年龄',
      'ICU LOS days': 'ICU 住院天数',
      'Min': '最小值',
      'Max': '最大值',
      'registered export aggregate': '注册导出聚合',
      'bounded column read': '有界列读取',
      'SOFA-2 severity bins': 'SOFA-2 严重度分箱',
      'SOFA-1 to SOFA-2 movement': 'SOFA-1 到 SOFA-2 变化',
      'Worst-ICU severity transition matrix': 'ICU 最严重 SOFA 转移矩阵',
      'Matrix value': '矩阵数值',
      'Percent': '百分比',
      'Count': '人数',
      'Granularity': '粒度',
      'Coarse': '粗略',
      'Medium': '中等',
      'Fine': '细粒度',
      'Exact': '逐分',
      '4 bands': '4 档',
      '6 bands': '6 档',
      '12 bands': '12 档',
      '25 scores': '25 分',
      'Rows are SOFA-1 severity bands; columns are SOFA-2 bands. Color intensity follows the selected value.': '行是 SOFA-1 严重度分层，列是 SOFA-2 分层；颜色深浅随当前显示值变化。',
      'Rows are SOFA-1 score bands; columns are SOFA-2 score bands. Use the granularity control to move from clinical bands to exact 0-24 scores.': '行是 SOFA-1 分数分箱，列是 SOFA-2 分数分箱。可用粒度控件从临床分层切到 0-24 逐分矩阵。',
      'Same severity band': '同一严重度层级',
      'SOFA-2 higher band': 'SOFA-2 更高层级',
      'SOFA-2 lower band': 'SOFA-2 更低层级',
      'Paired aggregate ready': '配对聚合已就绪',
      'Paired reclassification blocked': '配对重分层已拦截',
      'Cohort profile': '队列画像',
      'Real cohort aggregate': '真实队列聚合',
      'Aggregate ranges': '聚合范围',
      'Source provenance': '来源溯源',
      'Export measures': '导出指标',
      'Files loaded': '已加载文件',
      'Rows reviewed': '已审阅行数',
      'Outcome groups': '结局分组',
      'Table one': 'Table One',
      'Baseline characteristics comparison': '基线特征对照',
      'Characteristic': '特征',
      'Overall': '总体',
      'p-value': 'p 值',
      'Group Contrast Table': '分组对照表',
      'Select comparison mode': '选择对照模式',
      'Features': '特征',
      'Select feature modules': '选择特征模块',
      'Demographics': '人口统计',
      'Outcome': '结局',
      'Vital Signs': '生命体征',
      'Features to load': '待加载特征数',
      'Selected modules': '已选模块',
      'Catalog available': 'Catalog 可用范围',
      'Recommended modules': '推荐模块',
      'All catalog modules': '全部 Catalog 模块',
      'Load all modules': '加载全部模块',
      'Use recommended modules': '恢复推荐模块',
      'Default load': '默认加载',
      'Custom Threshold': '自定义阈值',
      'Short vs Long Stay': '短住院 vs 长住院',
      'Above threshold': '高于阈值',
      'Below threshold': '低于阈值',
      'Example': '示例',
      'Ratio': '比例',
      'Survived vs Deceased': '存活 vs 死亡',
      'Male vs Female': '男性 vs 女性',
      'Short vs Long Stay': '短住院 vs 长住院',
      'Age < 65': '年龄 < 65',
      'Age ≥ 65': '年龄 ≥ 65',
      'LOS < 5d': 'ICU 住院 < 5 天',
      'LOS ≥ 5d': 'ICU 住院 ≥ 5 天',
      'Sepsis-3 +': 'Sepsis-3 阳性',
      'Sepsis-3 -': 'Sepsis-3 阴性',
      'Age, mean (SD)': '年龄，均值（SD）',
      'Male, n (%)': '男性，n (%)',
      'SOFA, median': 'SOFA，中位数',
      'Lactate, mmol/L': '乳酸，mmol/L',
      'ICU LOS, days': 'ICU 住院天数',
      'Mortality, n (%)': '死亡，n (%)',
      'Sepsis-3, n (%)': 'Sepsis-3，n (%)',
      'Ventilation, n (%)': '机械通气，n (%)',
      'Real': '真实',
      'Demo': '演示',
    };
    if (Object.prototype.hasOwnProperty.call(map, raw)) return t(raw, map[raw]);
    if (/^SOFA-2 <= ([^ ]+) vs > ([^ ]+)$/.test(raw)) {
      const m = raw.match(/^SOFA-2 <= ([^ ]+) vs > ([^ ]+)$/);
      return t(raw, `SOFA-2 <= ${m[1]} vs > ${m[2]}`);
    }
    if (/^(.+) by (.+)$/.test(raw)) {
      const m = raw.match(/^(.+) by (.+)$/);
      return t(raw, `${cohortText(m[1])} · 按 ${cohortText(m[2])} 分组`);
    }
    return t(raw, raw);
  }
  function cohortReason(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      'Cohort Review accepts only registered-source aggregate review in Stage17.': 'Cohort Review 当前只接受已注册来源的聚合审阅。',
      'Generic Table One/group p-values, SMDs, and confidence intervals remain blocked; survival log-rank is scoped to the KM module when timed outcomes exist.': '通用 Table One / 分组 p 值、SMD 和置信区间仍被拦截；只有存在事件时间时才在 KM 模块中提供 log-rank。',
      'Matched cohorts belong to Cross-DB parity and audit-gated analysis.': '匹配队列属于跨库 parity 与审计后的分析流程。',
      'Custom group thresholds require row-level validation before display.': '自定义分组阈值显示前需要行级校验。',
      'Custom thresholds require audited row-level cohort construction.': '自定义阈值需要经过审计的行级队列构建。',
      'Inferential statistics are withheld until the numeric evidence audit gate.': '推断统计会等到数值证据审计后再开放。',
      'Matched cohorts require an audit-bound analysis plan.': '匹配队列需要绑定审计的分析计划。',
      'Matched cohort logic is not part of Stage17 Cohort Review.': '匹配队列逻辑不属于当前 Stage17 队列审阅范围。',
      'Table One p-values, SMDs, and row-level baseline tables require the numeric evidence audit gate. Survival log-rank is scoped to the audited KM module when time-to-event data exist.': 'Table One p 值、SMD 和行级基线表需要数值证据审计；有事件时间时，survival log-rank 只在已审计 KM 模块中提供。',
      'Paired SOFA-1/SOFA-2 reclassification is not available for this export.': '此导出无法做配对 SOFA-1/SOFA-2 重分层。',
      'No outcome has both an event column and a time-to-event/censoring column in this export.': '此导出没有同时具备事件列与事件时间/删失时间列的结局。',
      'No supported two-group split is available for this cohort.': '此队列没有可用的双组分组。',
      'No survival curve could be computed from the available timed records.': '可用时间记录不足，无法计算生存曲线。',
      'Current export is loaded, but the cohort is above the interactive KM preview limit; continue with an audited local analysis job on this same export.': '当前导出已加载，但队列超过交互式 KM 预览上限；请在同一个导出上继续运行本地审计分析任务。',
      'Outcome module is not present in the registered export.': '注册导出中没有结局模块。',
      'Fewer than two cohort entities have valid survival time values.': '有效生存时间值少于两个队列实体。',
      'Demo threshold uses SOFA ≥ 6. Real custom thresholds remain fail-closed until a bounded cohort-builder backend is available.': '演示阈值使用 SOFA ≥ 6。真实自定义阈值会在有界队列构建后端可用前保持保守拦截。',
      'This export does not expose an outcome with both event and time-to-event columns.': '此导出没有同时包含事件列和事件时间列的结局。',
      'ICU mortality is unavailable because this export does not include ICU-specific event and time columns.': 'ICU 死亡不可用，因为当前导出没有 ICU 专用死亡事件列和 ICU 时间列。',
      'ICU mortality is unavailable because this export does not include an ICU-specific event column.': 'ICU 死亡不可用，因为当前导出没有 ICU 专用死亡事件列。',
      'ICU-specific event column is not present in the registered export.': '当前注册导出没有 ICU 专用死亡事件列。',
      'ICU mortality event rate is available, but KM/log-rank needs ICU-specific time columns.': 'ICU 死亡事件率可用，但 KM/log-rank 需要 ICU 专用时间列。',
      'Unavailable for this export': '此导出不可用',
      'Unavailable': '不可用',
    };
    if (/^No event column found for (.+)\.$/.test(raw)) {
      const label = raw.match(/^No event column found for (.+)\.$/)[1];
      return t(raw, `未找到“${cohortText(label)}”事件列。`);
    }
    if (/^(.+) is available only as an event flag; KM\/log-rank needs time-to-event or censoring time\.$/.test(raw)) {
      const label = raw.match(/^(.+) is available only as an event flag; KM\/log-rank needs time-to-event or censoring time\.$/)[1];
      return t(raw, `“${cohortText(label)}”只有事件标志；KM/log-rank 需要事件时间或删失时间。`);
    }
    return t(raw, map[raw] || raw);
  }

  function cohortDemoCatalogScope() {
    const modules = demoCatalogModules();
    const byKey = new Map(modules.map(module => [module.module, module]));
    const defaultKeys = ['demographics', 'outcome', 'vitals', 'sepsis3_sofa2'];
    const recommended = defaultKeys.map(key => byKey.get(key)).filter(Boolean);
    const fallback = modules.slice(0, Math.min(4, modules.length));
    const selected = state.featureScope === 'all' ? modules : (recommended.length ? recommended : fallback);
    const totalFeatureCount = (window.EU_CATALOG || {}).totalConcepts
      || modules.reduce((acc, module) => acc + (module.features || []).length, 0)
      || selected.reduce((acc, module) => acc + (module.features || []).length, 0);
    const selectedFeatureCount = selected.reduce((acc, module) => acc + (module.features || []).length, 0);
    return {
      allModules: modules,
      selectedModules: selected,
      isAll: state.featureScope === 'all',
      totalModuleCount: modules.length,
      selectedModuleCount: selected.length,
      totalFeatureCount,
      selectedFeatureCount,
    };
  }

  function cohortDemoFeaturePicker() {
    const scope = cohortDemoCatalogScope();
    const chips = scope.selectedModules.map(module => `
      <span class="chip solid" title="${esc(module.module)}">
        ${esc(module.label)}
        <span class="mono" style="font-size:10.5px;color:inherit;opacity:.72;">${fmtInt((module.features || []).length)}</span>
      </span>`).join('');
    const nextScope = scope.isAll ? 'recommended' : 'all';
    const actionLabel = scope.isAll ? cohortText('Use recommended modules') : cohortText('Load all modules');
    const badgeLabel = scope.isAll ? cohortText('All catalog modules') : cohortText('Recommended modules');
    const scopeNote = scope.isAll
      ? t('All catalog modules are selected for this demo review. The simulated preview can take a little longer, but it still does not scan a real export.', '已选择全部 catalog 模块用于这次演示审阅；演示预览可能稍慢一点，但不会扫描真实导出。')
      : t('Default loads a focused subset; use Load all modules to include every catalog feature.', '默认只加载推荐模块；点击“加载全部模块”即可纳入所有 catalog 特征。');
    return `
      <div class="card pad" style="padding:14px 16px;" data-cohort-catalog-scope="${scope.isAll ? 'all' : 'recommended'}">
        <div class="row wrap gap-8" style="align-items:center;">
          <span class="pill ${scope.isAll ? 'ok' : 'demo'}"><span class="dot"></span>${badgeLabel}</span>
          <span style="font-size:12px;color:var(--ink-3);">${cohortText('Catalog available')}: ${fmtInt(scope.totalModuleCount)} ${t('modules', '个模块')} · ${fmtInt(scope.totalFeatureCount)} ${t('features', '个特征')}</span>
          <span class="grow"></span>
          <button class="btn sm" type="button" data-cohort-feature-scope="${nextScope}">${scope.isAll ? icon('sliders', 12) : icon('layers', 12)} ${actionLabel}</button>
        </div>
        <div class="row wrap gap-6" style="margin-top:10px;">${chips}</div>
        <div class="row wrap gap-10" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--hair);justify-content:space-between;">
          <span class="row gap-6" style="font-size:12px;color:var(--ink-3);">${icon('flask', 13)} ${cohortText('Features to load')}: ${fmtInt(scope.selectedFeatureCount)} / ${fmtInt(scope.totalFeatureCount)}</span>
          <span class="row gap-6" style="font-size:12px;color:var(--ink-3);">${icon('layers', 13)} ${cohortText('Selected modules')}: ${fmtInt(scope.selectedModuleCount)} / ${fmtInt(scope.totalModuleCount)}</span>
        </div>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:8px;">${scopeNote}</div>
      </div>`;
  }

  function cohortRealModuleSummary(review) {
    const rows = (review && review.coverage) || [];
    if (!rows.length) return '';
    const exactRows = rows.filter(row => row.coverage_basis === 'unique_entity_intersection');
    const metadataRows = rows.filter(row => row.coverage_basis === 'metadata_row_count_only');
    const okish = rows.filter(row => ['ok', 'neutral'].includes(row.quality_status)).length;
    const chips = rows.map(row => {
      const cls = row.quality_status === 'ok' || row.quality_status === 'neutral' ? 'solid' : (row.quality_status === 'unknown' ? 'demo' : '');
      const label = row.coverage_basis === 'metadata_row_count_only'
        ? t('manifest rows', '清单行数')
        : cohortCoverageMetricValue(row);
      return `<span class="chip ${cls}" title="${esc(row.module)}">${esc(row.module)} <span class="mono" style="font-size:10.5px;color:inherit;opacity:.72;">${label}</span></span>`;
    }).join('');
    return `
      <div class="card pad mt-14" data-cohort-real-modules>
        <div class="row wrap gap-8" style="align-items:center;">
          <span class="pill ok"><span class="dot"></span>${t('Current export loaded', '当前导出已加载')}</span>
          <span style="font-size:12px;color:var(--ink-3);">${fmtInt(rows.length)} ${t('modules', '个模块')} · ${fmtInt(okish)} ${t('ready or event modules', '个已就绪/事件模块')} · ${fmtInt(exactRows.length)} ${t('exact coverage scans', '个精确覆盖扫描')}</span>
          <span class="grow"></span>
          <button class="btn sm" type="button" data-cohgo="coverage">${icon('shield', 12)} ${t('Open coverage audit', '打开覆盖审计')}</button>
        </div>
        <div class="row wrap gap-6" style="margin-top:10px;">${chips}</div>
        ${metadataRows.length ? `<div style="font-size:11.5px;color:var(--ink-4);margin-top:8px;">${t('Some non-Parquet or very large modules may show manifest-confirmed row counts instead of exact unique-stay coverage; they are still part of this export.', '部分非 Parquet 或超大模块可能显示清单确认的行数，而不是精确唯一 stay 覆盖率；它们仍属于当前导出。')}</div>` : ''}
      </div>`;
  }

  function cohortCoverageMetricLabel(row) {
    if (!row || row.coverage_basis === 'metadata_row_count_only') return t('row count only', '仅行数');
    if (row.metric_kind === 'event_rate') return cohortText('Event rate');
    if (row.metric_kind === 'exposure_rate') return cohortText('Exposure rate');
    return cohortText('Coverage');
  }

  function cohortCoverageMetricValue(row) {
    if (!row) return '—';
    if (row.coverage_basis === 'metadata_row_count_only') return t('row count only', '仅行数');
    return `${cohortCoverageMetricLabel(row)} ${fmtPct(row.coverage_pct)}`;
  }

  function cohortQualityStatusClass(row) {
    if (!row) return 'demo';
    if (row.coverage_basis === 'metadata_row_count_only') return 'ok';
    if (row.metric_kind === 'event_rate' || row.metric_kind === 'exposure_rate') return 'ok';
    if (row.quality_status === 'ok' || row.quality_status === 'neutral') return 'ok';
    if (row.quality_status === 'unknown') return 'demo';
    return 'warn';
  }

  function cohortQualityStatusLabel(row) {
    if (!row) return cohortText('Unknown');
    if (row.coverage_basis === 'metadata_row_count_only') return t('loaded', '已加载');
    if (row.metric_kind === 'event_rate') return cohortText('Event rate');
    if (row.metric_kind === 'exposure_rate') return cohortText('Exposure rate');
    if (row.quality_status === 'ok') return cohortText('Ready');
    if (row.quality_status === 'warn') return cohortText('Watch');
    if (row.quality_status === 'bad') return cohortText('Low coverage');
    if (row.quality_status === 'neutral') return cohortText('Rate only');
    return cohortText('Unknown');
  }

  function cohortRealFeaturePicker(review) {
    const catalog = (review && review.feature_catalog) || {};
    const selection = (review && review.feature_selection) || {};
    const modules = (catalog.modules || []).filter(module => module && module.feature_count);
    if (!modules.length) return '';
    const selectedRows = selection.selected || [];
    const selectedIds = new Set(selectedRows.map(row => row && row.id).filter(Boolean));
    const defaultIds = selection.default_ids || [];
    const maxSelected = selection.max_selected_features || catalog.max_selected_features || 48;
    if (state.featureModule !== 'all' && !modules.some(module => module.module === state.featureModule)) state.featureModule = 'all';
    const visibleModules = state.featureModule === 'all' ? modules : modules.filter(module => module.module === state.featureModule);
    const moduleChips = [
      `<button class="chip ${state.featureModule === 'all' ? 'solid' : ''}" type="button" data-cohort-feature-module="all">${t('All modules', '全部模块')} <span class="mono">${fmtInt(catalog.total_features)}</span></button>`,
      ...modules.map(module => `<button class="chip ${state.featureModule === module.module ? 'solid' : ''}" type="button" data-cohort-feature-module="${esc(module.module)}">${esc(module.label || module.module)} <span class="mono">${fmtInt(module.feature_count)}</span></button>`),
    ].join('');
    const atMax = selectedIds.size >= maxSelected;
    const featureButtons = visibleModules.flatMap(module => (module.features || []).map(feature => {
      const on = selectedIds.has(feature.id);
      const locked = atMax && !on;
      const attr = locked
        ? `aria-disabled="true" title="${esc(t('Selection limit reached; remove a feature before adding another.', '已达到选择上限；请先移除一个特征再添加。'))}"`
        : `data-cohort-feature-toggle="${esc(feature.id)}"`;
      return `
        <button class="chip ${on ? 'solid' : ''} ${locked ? 'disabled' : ''}" type="button" ${attr}>
          ${on ? icon('check', 11) : icon('plus', 11)}
          <span>${esc(feature.label || feature.column || feature.id)}</span>
          <span class="mono" style="opacity:.72;">${esc(module.module)}</span>
        </button>`;
    })).join('');
    const selectedChips = selectedRows.map(feature => `
      <button class="chip solid" type="button" data-cohort-feature-toggle="${esc(feature.id)}" title="${esc(t('Remove feature', '移除特征'))}">
        ${icon('check', 11)} ${esc(feature.label || feature.column || feature.id)}
        <span class="mono" style="opacity:.72;">${esc(feature.module || '')}</span>
      </button>`).join('');
    return `
      <div class="card pad mt-14" data-cohort-feature-picker>
        <div class="row wrap gap-8" style="align-items:center;">
          <span class="pill ok"><span class="dot"></span>${t('Full export feature catalog', '全量导出特征目录')}</span>
          <span style="font-size:12px;color:var(--ink-3);">${fmtInt(selection.module_count || catalog.total_modules)} ${t('modules', '个模块')} · ${fmtInt(selection.available_count || catalog.total_features)} ${t('available comparison features', '个可比较特征')} · ${fmtInt(selection.selected_count || selectedIds.size)} ${t('selected', '个已选择')}</span>
          <span class="grow"></span>
          <button class="btn sm" type="button" data-cohort-feature-default>${icon('sliders', 12)} ${t('Restore default features', '恢复默认特征')}</button>
          <button class="btn sm ghost" type="button" data-cohort-feature-clear>${icon('close', 12)} ${t('Clear added features', '清空已选特征')}</button>
        </div>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:8px;">${t('Default starts with key ICU variables, but every feature present in the loaded modules can be added to the descriptive group table. Values remain aggregate-only; no patient rows are returned.', '默认先选关键 ICU 变量，但已加载模块中的每个特征都可以加入描述性分组表。这里只返回聚合结果，不返回患者行。')}</div>
        <div class="row wrap gap-6" style="margin-top:10px;">${moduleChips}</div>
        ${selectedChips ? `<div class="row wrap gap-6" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--hair);"><span style="font-size:11px;color:var(--ink-4);padding-top:4px;">${t('Selected', '已选择')}</span>${selectedChips}</div>` : ''}
        <div class="row wrap gap-6" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--hair);max-height:180px;overflow:auto;">${featureButtons}</div>
        <div style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Interactive comparison is capped to keep large local exports responsive.', '为保证大型本地导出交互流畅，单次交互比较会限制选中特征数量。')} ${fmtInt(selectedIds.size)} / ${fmtInt(maxSelected)}</div>
      </div>`;
  }

  function cohortCoverageBody(review, opts = {}) {
    const rows = review.coverage || [];
    const q = review.quality || {};
    const metadataOnlyCount = rows.filter(row => row.coverage_basis === 'metadata_row_count_only').length;
    const rateRows = rows.filter(row => row.metric_kind === 'event_rate' || row.metric_kind === 'exposure_rate').length;
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('Coverage audit')}</div><h2>${cohortText(opts.demo ? 'Demo module coverage and quality' : 'Real module coverage and quality')}</h2></div>
      ${opts.demo ? demo.panelNote('coverage') : ''}
      ${metadataOnlyCount ? `<div class="note info mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('Large export coverage optimized', '大导出覆盖率已优化')}</div><div class="d">${t('Some non-Parquet or very large modules are shown with manifest-confirmed row counts first to avoid a slow full stay-id scan. They are loaded modules, not missing modules.', '部分非 Parquet 或超大模块会先显示清单确认的行数，避免缓慢的全量 stay_id 扫描。它们是已加载模块，不是缺失模块。')}</div></div></div>` : ''}
      ${rateRows ? `<div class="note info mt-12"><div class="ico">${icon('activity', 14)}</div><div class="body"><div class="t">${cohortText('Presence-rate modules')}</div><div class="d">${cohortText('Event/exposure rows show cohort incidence or exposure prevalence, not missingness coverage; they are excluded from the coverage watchlist.')}</div></div></div>` : ''}
      ${cohortCoverageForest(review)}
      <div class="audit-cards compact">
        ${[
          ['Modules OK', fmtInt(q.modules_ok)],
          ['Watchlist', fmtInt(q.watchlist_count)],
          ['Median coverage', fmtPct(q.median_coverage_pct)],
          ['Presence-rate modules', fmtInt(q.modules_neutral)],
          ['Unknown coverage', fmtInt(q.modules_unknown)],
        ].map(([k, v]) => `<div class="audit-card"><div class="ac-k">${cohortText(k)}</div><div class="ac-v mono">${v}</div></div>`).join('')}
      </div>
      <details class="cohort-coverage-details mt-16"><summary>${t('Open exact module table', '展开精确模块表')}</summary><div class="table-wrap table-scroll mt-10">
        <table class="eu-table">
          <thead><tr><th>${cohortText('Module')}</th><th class="num">${cohortText('Records')}</th><th class="num">${cohortText('Fields')}</th><th class="num">${cohortText('Entities')}</th><th class="num">${cohortText('Coverage / rate')}</th><th>${cohortText('Interpretation')}</th></tr></thead>
          <tbody>
            ${rows.map(row => `<tr>
              <td class="key">${esc(row.module)}</td>
              <td class="num">${fmtInt(row.rows)}</td>
              <td class="num">${fmtInt(row.column_count)}</td>
              <td class="num">${row.coverage_basis === 'metadata_row_count_only' ? t('manifest confirmed', '清单确认') : fmtInt(row.covered_entities)}</td>
              <td class="num">${cohortCoverageMetricValue(row)}</td>
              <td><span class="pill ${cohortQualityStatusClass(row)}" style="height:20px;">${cohortQualityStatusLabel(row)}</span></td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div></details>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Fail-closed scope')}</div><div class="d">${t('Coverage is aggregate-only. Row-level filtering, subgroup missingness, and eligibility waterfalls remain blocked until a bounded cohort-builder backend exists.', '覆盖率是仅聚合结果。行级筛选、亚组缺失率和纳排瀑布图会在有界队列构建后端就绪前保持拦截。')}</div></div></div>`;
  }

  function cohortCoverageForest(review) {
    const rows = (review.coverage || []).filter(row => row && row.covered_entities != null && row.coverage_pct != null);
    const total = Number(review.summary && review.summary.cohort_size) || Math.max(0, ...rows.map(row => Number(row.covered_entities || 0)));
    if (!rows.length || !total) return '';
    const z = 1.96;
    const interval = row => {
      const n = total;
      const p = Math.max(0, Math.min(1, Number(row.covered_entities || 0) / n));
      const denominator = 1 + z * z / n;
      const centre = (p + z * z / (2 * n)) / denominator;
      const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator;
      return [Math.max(0, (centre - margin) * 100), Math.min(100, (centre + margin) * 100)];
    };
    return `<section class="cohort-coverage-forest" data-cohort-coverage-forest>
      <div class="cohort-forest-head"><div><b>${t('Module coverage with 95% Wilson intervals', '模块覆盖率与 95% Wilson 区间')}</b><span>${t('Presence-rate modules remain labelled as event or exposure rates, not missingness.', '事件或暴露模块仍按发生率标记，不解释为缺失率。')}</span></div><strong>N = ${fmtInt(total)}</strong></div>
      <div class="cohort-forest-axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
      <div class="cohort-forest-rows">${rows.map(row => { const value = Math.max(0, Math.min(100, Number(row.coverage_pct))); const [lo, hi] = interval(row); return `<div class="cohort-forest-row"><div><b>${esc(row.module)}</b><span>${esc(cohortText(row.metric_kind === 'event_rate' ? 'event rate' : row.metric_kind === 'exposure_rate' ? 'exposure rate' : 'coverage'))}</span></div><div class="cohort-forest-track"><i class="ci" style="left:${lo.toFixed(2)}%;width:${Math.max(.8, hi-lo).toFixed(2)}%"></i><i class="point" style="left:${value.toFixed(2)}%"></i></div><strong>${fmtPct(value)} <small>${fmtInt(row.covered_entities)}/${fmtInt(total)}</small></strong></div>`; }).join('')}</div>
    </section>`;
  }

  const SOFA_MATRIX_GRANULARITIES = {
    coarse: {
      label: 'Coarse',
      detail: '4 bands',
      bins: [
        { label: '0-5', min: 0, max: 5 },
        { label: '6-8', min: 6, max: 8 },
        { label: '9-11', min: 9, max: 11 },
        { label: '12+', min: 12, max: 24 },
      ],
    },
    medium: {
      label: 'Medium',
      detail: '6 bands',
      bins: [
        { label: '0-3', min: 0, max: 3 },
        { label: '4-7', min: 4, max: 7 },
        { label: '8-11', min: 8, max: 11 },
        { label: '12-15', min: 12, max: 15 },
        { label: '16-19', min: 16, max: 19 },
        { label: '20-24', min: 20, max: 24 },
      ],
    },
    fine: {
      label: 'Fine',
      detail: '12 bands',
      bins: Array.from({ length: 12 }, (_, index) => {
        const min = index * 2;
        const max = index === 11 ? 24 : min + 1;
        return { label: min === max ? String(min) : `${min}-${max}`, min, max };
      }),
    },
    exact: {
      label: 'Exact',
      detail: '25 scores',
      bins: Array.from({ length: 25 }, (_, score) => ({ label: String(score), min: score, max: score })),
    },
  };

  function cohortSofaGranularityButtons(hasExactMatrix) {
    if (!hasExactMatrix) return '';
    const order = ['coarse', 'medium', 'fine', 'exact'];
    return `
      <div class="sofa-matrix-control">
        <span>${cohortText('Granularity')}</span>
        <div class="sofa-matrix-toggle" role="group" aria-label="${cohortText('Granularity')}">
          ${order.map(key => {
            const opt = SOFA_MATRIX_GRANULARITIES[key];
            return `<button class="${state.sofaMatrixGranularity === key ? 'active' : ''}" data-cohort-sofa-granularity="${key}" type="button">${cohortText(opt.label)} <small>${cohortText(opt.detail)}</small></button>`;
          }).join('')}
        </div>
      </div>`;
  }

  function cohortSofaExactMatrixMap(reclass) {
    const exact = reclass.exact_score_matrix || [];
    if (!Array.isArray(exact) || !exact.length) return null;
    const map = new Map();
    exact.forEach(row => {
      const source = Number(row && row.label);
      if (!Number.isFinite(source)) return;
      (row.cells || []).forEach(cell => {
        const target = Number(cell && cell.label);
        if (!Number.isFinite(target)) return;
        map.set(`${source}|${target}`, Number(cell.count) || 0);
      });
    });
    return map.size ? map : null;
  }

  function cohortSofaBinnedMatrix(reclass) {
    const exactMap = cohortSofaExactMatrixMap(reclass);
    if (!exactMap) {
      return {
        bins: reclass.severity_bins || [],
        matrix: reclass.transition_matrix || [],
        exact: false,
      };
    }
    const granularity = SOFA_MATRIX_GRANULARITIES[state.sofaMatrixGranularity] ? state.sofaMatrixGranularity : 'exact';
    const bins = SOFA_MATRIX_GRANULARITIES[granularity].bins;
    const paired = Number(reclass.paired_count) || Array.from(exactMap.values()).reduce((acc, value) => acc + value, 0) || 0;
    const matrix = bins.map(sourceBin => {
      const cells = bins.map(targetBin => {
        let count = 0;
        for (let source = sourceBin.min; source <= sourceBin.max; source += 1) {
          for (let target = targetBin.min; target <= targetBin.max; target += 1) {
            count += exactMap.get(`${source}|${target}`) || 0;
          }
        }
        return {
          label: targetBin.label,
          count,
          pct: paired ? Number((count / paired * 100).toFixed(1)) : 0,
        };
      });
      return {
        label: sourceBin.label,
        count: cells.reduce((acc, cell) => acc + cell.count, 0),
        cells,
      };
    });
    return {
      bins: bins.map(row => row.label),
      matrix,
      exact: true,
    };
  }

  function cohortSofaHeatmap(reclass) {
    const binned = cohortSofaBinnedMatrix(reclass);
    const bins = binned.bins || [];
    const matrix = binned.matrix || [];
    if (!bins.length || !matrix.length) {
      return `<div class="muted" style="font-size:11px;">${t('No paired SOFA-1/SOFA-2 bins in this export.', '此导出没有配对 SOFA-1/SOFA-2 分箱。')}</div>`;
    }
    const mode = state.sofaMatrixMode === 'count' ? 'count' : 'pct';
    const hasExactMatrix = !!binned.exact;
    const pairedExact = hasExactMatrix && state.sofaMatrixGranularity === 'exact';
    const maxValue = Math.max(
      1,
      ...matrix.flatMap(row => (row.cells || []).map(cell => (
        mode === 'count' ? Number(cell.count) || 0 : Number(cell.pct) || 0
      ))),
    );
    const chartMatrix = matrix.map(row => ({
      label: row.label,
      cells: (row.cells || []).map(cell => ({
        count: Number(cell.count) || 0,
        pct: Number(cell.pct) || 0,
        intensity: (mode === 'count' ? Number(cell.count) || 0 : Number(cell.pct) || 0) / maxValue,
      })),
    }));
    return `
      <div class="sofa-matrix-head mt-12">
        <div>
          <div class="rc-sec-t">${cohortText('Worst-ICU severity transition matrix')}</div>
          <p>${pairedExact ? t('Both axes run from 0 to 24; bubble size is the paired count and the dashed line marks identical scores.', '横轴与纵轴均为 0–24 分；气泡大小表示配对人数，虚线表示两版评分完全相同。') : hasExactMatrix ? cohortText('Rows are SOFA-1 score bands; columns are SOFA-2 score bands. Use the granularity control to move from clinical bands to exact 0-24 scores.') : cohortText('Rows are SOFA-1 severity bands; columns are SOFA-2 bands. Color intensity follows the selected value.')}</p>
        </div>
        <div class="sofa-matrix-controls">
          ${cohortSofaGranularityButtons(hasExactMatrix)}
          <div class="sofa-matrix-control">
            <span>${cohortText('Matrix value')}</span>
            <div class="sofa-matrix-toggle" role="group" aria-label="${cohortText('Matrix value')}">
              <button class="${mode === 'pct' ? 'active' : ''}" data-cohort-sofa-matrix-mode="pct" type="button">${cohortText('Percent')}</button>
              <button class="${mode === 'count' ? 'active' : ''}" data-cohort-sofa-matrix-mode="count" type="button">N</button>
            </div>
          </div>
        </div>
      </div>
      ${cohortCharts && typeof cohortCharts.heatmapSlot === 'function'
        ? cohortCharts.heatmapSlot({
          label: pairedExact ? cohortText('Exact SOFA-1 / SOFA-2 paired matrix') : cohortText('Worst-ICU severity transition matrix'),
          description: hasExactMatrix
            ? cohortText('Rows are SOFA-1 score bands; columns are SOFA-2 score bands. Use the granularity control to move from clinical bands to exact 0-24 scores.')
            : cohortText('Rows are SOFA-1 severity bands; columns are SOFA-2 bands. Color intensity follows the selected value.'),
          bins,
          matrix: chartMatrix,
          mode,
          xLabel: 'SOFA-2',
          yLabel: 'SOFA-1',
          valueLabel: mode === 'count' ? 'N' : '%',
          sameLabel: pairedExact ? t('Same score', '相同分数') : cohortText('Same severity band'),
          upLabel: pairedExact ? t('SOFA-2 higher score', 'SOFA-2 分数更高') : cohortText('SOFA-2 higher band'),
          downLabel: pairedExact ? t('SOFA-2 lower score', 'SOFA-2 分数更低') : cohortText('SOFA-2 lower band'),
          pairedExact,
        })
        : ''}
      <div class="viz-cap"><b>${t('How to read', '怎么读')}</b><span>${t('Cells on the diagonal are patients scored the same by SOFA-1 and SOFA-2; off-diagonal cells are patients the two definitions disagree on — large off-diagonal cells are where switching score versions would change your cohort.', '对角线上的格子是 SOFA-1 与 SOFA-2 评分一致的患者；对角线以外是两套定义不一致的患者 —— 偏离对角线的大格子意味着换用评分版本会改变你的队列。')}</span></div>`;
  }

  function cohortSofaBody(review, opts = {}) {
    const s = review.summary || {};
    const sofa = s.sofa2 || {};
    const reclass = review.sofa_reclassification || {};
    const bins = sofa.bins || [];
    const maxBin = Math.max(1, ...bins.map(b => b.count || 0));
    const movement = reclass.direction_counts || {};
    const delta = reclass.delta_summary || {};
    const movementCards = reclass.status === 'ready' ? [
      [fmtInt(reclass.paired_count), 'Paired entities', `${fmtPct(reclass.coverage_pct)} ${t('of cohort', '的队列覆盖')}`, 'n'],
      [fmtInt(movement.up && movement.up.count), 'SOFA-2 higher', fmtPct(movement.up && movement.up.pct), 'up'],
      [fmtInt(movement.down && movement.down.count), 'SOFA-2 lower', fmtPct(movement.down && movement.down.pct), 'down'],
      [fmtNum(delta.median, 1), 'Median delta', 'SOFA-2 minus SOFA-1', 'delta'],
    ] : [];
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('SOFA reclassification')}</div><h2>${cohortText(opts.demo ? 'Demo SOFA-2 aggregate preview' : 'SOFA-2 aggregate review')}</h2></div>
      ${opts.demo ? demo.panelNote('sofa') : ''}
      <div class="rc-kpis">
        ${[
          [fmtNum(sofa.median, 1), 'Median SOFA-2', `${fmtInt(sofa.count)} ${t('entities with score', '个实体有评分')}`, 'delta'],
          [fmtNum(sofa.mean, 1), 'Mean SOFA-2', 'registered export aggregate', 'n'],
          [fmtNum(sofa.min, 1), 'Min', 'bounded column read', 'down'],
          [fmtNum(sofa.max, 1), 'Max', 'bounded column read', 'up'],
        ].map(([v, label, hint, kind]) => `
          <div class="rc-kpi rc-${kind}">
            <div class="rk-top"><span class="rk-ico">${icon(kind === 'up' ? 'arrow' : kind === 'down' ? 'arrow' : 'layers', 13)}</span><span class="rk-label">${cohortText(label)}</span></div>
            <div class="rk-val mono">${v}</div>
            <div class="rk-hint">${cohortText(hint)}</div>
          </div>`).join('')}
      </div>
      <div class="card pad mt-16">
        <div class="rc-sec-t">${cohortText('SOFA-2 severity bins')}</div>
        <div class="rc-groups">
          ${bins.map(bin => `
            <div class="rc-grow">
              <div class="rg-head"><span class="rg-name">${esc(bin.label)}</span><span class="rg-pct mono">${fmtPct(bin.pct)}</span></div>
              <div class="rg-bar"><div class="rg-fill same" style="width:${((bin.count || 0) / maxBin * 100).toFixed(0)}%;"></div></div>
              <div class="rg-meta"><span>${fmtInt(bin.count)} ${t('entities', '个实体')}</span></div>
            </div>`).join('')}
        </div>
      </div>
      ${reclass.status === 'ready' ? `
        <div class="card pad mt-16">
          <div class="rc-sec-t">${cohortText('SOFA-1 to SOFA-2 movement')}</div>
          <div class="rc-kpis compact">
            ${movementCards.map(([v, label, hint, kind]) => `
              <div class="rc-kpi rc-${kind}">
                <div class="rk-top"><span class="rk-ico">${icon(kind === 'up' ? 'arrow' : kind === 'down' ? 'arrow' : 'layers', 13)}</span><span class="rk-label">${cohortText(label)}</span></div>
                <div class="rk-val mono">${v}</div>
                <div class="rk-hint">${cohortText(hint)}</div>
              </div>`).join('')}
          </div>
          ${cohortSofaHeatmap(reclass)}
        </div>
        <div class="note ok mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Paired aggregate ready')}</div><div class="d">${t('Worst-ICU SOFA-1/SOFA-2 movement is computed from bounded per-entity score aggregates only. No paired patient rows or inferential statistics are returned.', 'ICU 最严重 SOFA-1/SOFA-2 变化只由有界实体级评分聚合计算；不返回配对患者行或推断统计。')}</div></div></div>
      ` : `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${cohortText('Paired reclassification blocked')}</div><div class="d">${esc(cohortReason(reclass.reason || 'Paired SOFA-1/SOFA-2 reclassification is not available for this export.'))}</div></div></div>`}`;
  }

  function cohortDistBars(bins) {
    const arr = (bins || []).filter(Boolean);
    if (!arr.length) return `<div class="muted" style="font-size:11px;">${t('No binned values in this export.', '此导出无可分箱数值。')}</div>`;
    const maxN = Math.max(1, ...arr.map(b => b.count || 0));
    return arr.map(b => `<div class="qrow"><span>${esc(b.label)}</span><div class="qbar"><span style="width:${((b.count || 0) / maxN * 100).toFixed(0)}%"></span></div><span class="qv">${fmtInt(b.count)}</span></div>`).join('');
  }

  function cohortCompositionBars(rows) {
    return (rows || []).map(([label, pct]) => `<div class="qrow"><span>${cohortText(label)}</span><div class="qbar"><span style="width:${pct == null ? 0 : Math.max(0, Math.min(100, pct)).toFixed(0)}%"></span></div><span class="qv">${fmtPct(pct)}</span></div>`).join('');
  }

  function cohortProfileLabel(row) {
    const item = row || {};
    if (window.EU_LANG === 'zh') return item.label_zh || item.zh || item.label || item.id || '';
    return item.label || item.label_en || item.label_zh || item.id || '';
  }

  function cohortProfileReason(row) {
    const item = row || {};
    if (window.EU_LANG === 'zh') return item.reason_zh || item.text_zh || item.reason || item.text || '';
    return item.reason || item.text || item.reason_zh || item.text_zh || '';
  }

  function cohortProfileStatusText(status) {
    const key = String(status || 'unknown');
    const labels = {
      ready: [t('ready', '已就绪')],
      partial: [t('partial', '部分可用')],
      unavailable: [t('not in export', '当前导出未提供')],
      schema_only: [t('schema only', '仅结构可见')],
      ok: [t('ok', '正常')],
      warn: [t('watch', '关注')],
      bad: [t('low', '偏低')],
      unknown: [t('unknown', '未知')],
    };
    return (labels[key] && labels[key][0]) || key;
  }

  function cohortProfileUnit(item) {
    const row = item || {};
    return window.EU_LANG === 'zh' ? (row.unit_zh || row.unit || '') : (row.unit || row.unit_zh || '');
  }

  function cohortProfileItemValue(item) {
    const row = item || {};
    if (row.kind === 'numeric') {
      if (row.value == null) return '—';
      const unit = cohortProfileUnit(row);
      return `${fmtNum(row.value, 1)}${unit ? ` ${esc(unit)}` : ''}`;
    }
    if (row.kind === 'proportion' || row.kind === 'event_rate' || row.kind === 'module_coverage') {
      return row.pct == null ? '—' : fmtPct(row.pct);
    }
    if (row.kind === 'count') {
      return fmtInt(row.count);
    }
    if (row.kind === 'category') {
      const first = (row.bins || [])[0];
      return first ? `${esc(first.label)} · ${fmtPct(first.pct)}` : '—';
    }
    return row.value == null ? '—' : esc(row.value);
  }

  function cohortProfileDetail(item) {
    const row = item || {};
    if (row.status === 'unavailable') return cohortProfileReason(row) || t('Not present in this export.', '当前导出未提供。');
    if (row.kind === 'numeric') {
      const unit = cohortProfileUnit(row);
      return `${t('range', '范围')} ${fmtNum(row.min, 1)}-${fmtNum(row.max, 1)}${unit ? ` ${esc(unit)}` : ''} · n=${fmtInt(row.count)}`;
    }
    if (row.kind === 'proportion' || row.kind === 'event_rate' || row.kind === 'module_coverage') {
      const base = row.count == null ? t('entity denominator unavailable', '实体分母不可用') : `${fmtInt(row.count)} / ${fmtInt(row.denominator)}`;
      const modules = (row.modules || []).length ? ` · ${esc((row.modules || []).join(', '))}` : '';
      const records = row.rows ? ` · ${fmtInt(row.rows)} ${t('records', '记录')}` : '';
      return `${base}${modules}${records}`;
    }
    if (row.kind === 'count') {
      return row.denominator ? `${fmtInt(row.count)} / ${fmtInt(row.denominator)}` : '';
    }
    if (row.kind === 'category') {
      return (row.bins || []).slice(0, 3).map(bin => `${esc(bin.label)} ${fmtPct(bin.pct)}`).join(' · ') || t('No categorical column available.', '没有可用分类列。');
    }
    return cohortProfileReason(row);
  }

  function cohortProfileItem(item) {
    const row = item || {};
    const status = String(row.status || 'unknown').replace(/[^a-z0-9_-]/gi, '');
    const pct = typeof row.pct === 'number' ? Math.max(0, Math.min(100, row.pct)) : null;
    const bar = pct == null ? '' : `<div class="cprof-bar"><span style="width:${pct.toFixed(1)}%"></span></div>`;
    return `
      <div class="cprof-item ${status}">
        <div class="cprof-k">${esc(cohortProfileLabel(row))}</div>
        <div class="cprof-v">${cohortProfileItemValue(row)}</div>
        ${bar}
        <div class="cprof-d">${esc(cohortProfileDetail(row))}</div>
      </div>`;
  }

  function cohortClinicalProfile(profile) {
    const domains = (profile && profile.domains) || [];
    if (!domains.length) return '';
    return `
      <div class="cprof-grid">
        ${domains.map(domain => `
          <section class="cprof-domain ${esc(String(domain.status || 'unknown'))}">
            <div class="cprof-head">
              <div>
                <div class="eyebrow">${esc(cohortProfileLabel(domain))}</div>
                <h3>${esc(cohortProfileLabel(domain))}</h3>
              </div>
              <span class="pill ${domain.status === 'ready' ? 'ok' : 'dashed'}">${esc(cohortProfileStatusText(domain.status || 'unknown'))}</span>
            </div>
            <div class="cprof-items">${(domain.items || []).map(cohortProfileItem).join('')}</div>
          </section>
        `).join('')}
      </div>
      ${(profile.notes || []).length ? `<div class="note info mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body">${(profile.notes || []).map(note => `<div class="t">${esc(cohortProfileLabel(note))}</div><div class="d">${esc(cohortProfileReason(note))}</div>`).join('')}</div></div>` : ''}`;
  }

  function demoNumericProfile(id, label, labelZh, value, unit, unitZh, min, max, count = 10) {
    return { id, label, label_zh: labelZh, kind: 'numeric', status: 'ready', value, unit, unit_zh: unitZh, min, max, count };
  }

  function demoPctProfile(id, label, labelZh, pct, count, denominator = 10, kind = 'proportion') {
    return { id, label, label_zh: labelZh, kind, status: 'ready', pct, count, denominator };
  }

  function demoCohortClinicalProfile() {
    return {
      status: 'seeded_demo_clinical_shape',
      payload_scope: 'demo_cohort_aggregate_no_patient_rows',
      domains: [
        {
          id: 'demo_demographics',
          label: 'Demographics',
          label_zh: '人口统计',
          status: 'ready',
          items: [
            demoNumericProfile('age', 'Median age', '年龄中位数', 63, 'years', '岁', 28, 91),
            demoPctProfile('female', 'Female', '女性', 44, 4),
            {
              id: 'admission',
              label: 'Admission type',
              label_zh: '入院类型',
              kind: 'category',
              status: 'ready',
              count: 10,
              distinct: 3,
              bins: [
                { label: t('Emergency', '急诊'), count: 5, pct: 50 },
                { label: t('Transfer', '转入'), count: 3, pct: 30 },
                { label: t('Elective', '择期'), count: 2, pct: 20 },
              ],
            },
          ],
        },
        {
          id: 'demo_severity_outcomes',
          label: 'Severity and outcomes',
          label_zh: '严重程度与结局',
          status: 'ready',
          items: [
            demoNumericProfile('sofa2', 'Worst SOFA-2', '最严重 SOFA-2', 6, 'points', '分', 1, 18),
            demoPctProfile('sepsis3', 'Sepsis-3 incidence', 'Sepsis-3 发生率', 60, 6, 10, 'event_rate'),
            demoPctProfile('hospital_mortality', 'Hospital mortality', '院内死亡率', 20, 2, 10, 'event_rate'),
            demoNumericProfile('icu_los', 'ICU length of stay', 'ICU 住院时长', 5.6, 'days', '天', 1.1, 21.4),
          ],
        },
        {
          id: 'demo_treatments',
          label: 'Treatments and organ support',
          label_zh: '治疗暴露与器官支持',
          status: 'ready',
          items: [
            demoPctProfile('mechanical_ventilation', 'Mechanical ventilation', '机械通气', 50, 5, 10, 'event_rate'),
            demoPctProfile('vasopressors', 'Vasopressor exposure', '血管活性药物暴露', 40, 4, 10, 'event_rate'),
            demoPctProfile('rrt', 'Renal replacement therapy', '肾脏替代治疗', 10, 1, 10, 'event_rate'),
            demoPctProfile('antibiotics', 'Antibiotic exposure', '抗感染治疗', 70, 7, 10, 'event_rate'),
          ],
        },
        {
          id: 'demo_diagnoses',
          label: 'Diagnoses and comorbidities',
          label_zh: '诊断与共病',
          status: 'ready',
          items: [
            demoPctProfile('aki', 'AKI / renal dysfunction', 'AKI / 肾功能异常', 30, 3, 10, 'event_rate'),
            demoPctProfile('respiratory_failure', 'Respiratory failure', '呼吸衰竭', 40, 4, 10, 'event_rate'),
            demoPctProfile('shock', 'Shock phenotype', '休克表型', 30, 3, 10, 'event_rate'),
            demoPctProfile('infection', 'Suspected infection', '疑似感染', 70, 7, 10, 'event_rate'),
          ],
        },
        {
          id: 'demo_vitals_labs',
          label: 'Vitals and laboratory profile',
          label_zh: '生命体征与实验室',
          status: 'ready',
          items: [
            demoNumericProfile('map', 'Mean arterial pressure', '平均动脉压', 76, 'mmHg', 'mmHg', 45, 126),
            demoNumericProfile('lactate', 'Lactate', '乳酸', 2.4, 'mmol/L', 'mmol/L', 0.8, 8.9),
            demoNumericProfile('creatinine', 'Creatinine', '肌酐', 1.3, 'mg/dL', 'mg/dL', 0.5, 4.8),
            demoNumericProfile('platelets', 'Platelets', '血小板', 168, '10^9/L', '10^9/L', 38, 420),
          ],
        },
        {
          id: 'demo_completeness',
          label: 'Data completeness',
          label_zh: '数据覆盖',
          status: 'ready',
          items: [
            demoPctProfile('demographics_module', 'Demographics module', '人口统计模块', 100, 10, 10, 'module_coverage'),
            demoPctProfile('vitals_module', 'Vital signs module', '生命体征模块', 100, 10, 10, 'module_coverage'),
            demoPctProfile('labs_module', 'Laboratory modules', '实验室模块', 90, 9, 10, 'module_coverage'),
            demoPctProfile('outcome_module', 'Outcome module', '结局模块', 100, 10, 10, 'module_coverage'),
          ],
        },
      ],
      notes: [
        {
          label: 'Demo-only clinical shape',
          label_zh: '仅演示临床结构',
          text: 'The demo shows the dimensions a real cohort profile should expose; values are seeded UI examples, not research results.',
          text_zh: '演示页展示真实队列画像应覆盖的维度；数值是界面示例，不是研究结果。',
        },
      ],
    };
  }

  function cohortSnapshotBody(reviewOverride) {
    const review = reviewOverride || cohortReview();
    if (review && review.summary) {
      const s = review.summary;
      return `
      <div class="sec-stack"><div class="lbl">${cohortText('Cohort profile')}</div><h2>${cohortText('Real cohort aggregate')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Cohort size')}</div><div class="val">${fmtInt(s.cohort_size)}</div></div>
        <div class="stat"><div class="label">${cohortText('Median age')}</div><div class="val">${fmtNum(s.age && s.age.median, 1)}</div></div>
        <div class="stat"><div class="label">${cohortText('Female')}</div><div class="val">${fmtPct(s.sex && s.sex.female_pct)}</div></div>
        <div class="stat"><div class="label">${cohortText('Sepsis-3 +')}</div><div class="val">${fmtPct(s.sepsis_pct)}</div></div>
        <div class="stat"><div class="label">${cohortText('Median SOFA-2')}</div><div class="val">${fmtNum(s.sofa2 && s.sofa2.median, 1)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">${fmtPct(s.mortality_pct)}</div></div>
      </div>
      <div class="card pad mt-16">
        <div class="sec-stack mini"><div class="lbl">${t('Clinical phenotype', '临床画像')}</div><h3>${t('Interpretable cohort dimensions', '可解释的队列维度')}</h3></div>
        ${cohortClinicalProfile(s.clinical_profile)}
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Age distribution', '年龄分布')}</div>
          ${cohortDistBars(s.age && s.age.bins)}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('SOFA-2 severity', 'SOFA-2 严重度')}</div>
          ${cohortDistBars(s.sofa2 && s.sofa2.bins)}
        </div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('ICU LOS distribution', 'ICU 住院时长分布')}</div>
          ${cohortDistBars(s.los_icu_days && s.los_icu_days.bins)}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Cohort composition', '队列构成')}</div>
          ${cohortCompositionBars([
            ['Female', s.sex && s.sex.female_pct],
            ['Mortality', s.mortality_pct],
            ['Sepsis-3 +', s.sepsis_pct],
          ])}
          ${(s.admission && s.admission.bins && s.admission.bins.length) ? `<div class="eyebrow" style="margin:12px 0 8px;">${t('Admission type', '入院类型')}</div>${cohortDistBars(s.admission.bins)}` : ''}
        </div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${cohortText('Aggregate ranges')}</div>
          ${[
            ['Age', s.age],
            ['SOFA-2', s.sofa2],
            ['ICU LOS days', s.los_icu_days],
          ].map(([label, item]) => `<div class="setup-row"><span class="k">${cohortText(label)}</span><span class="vv">${t('median', '中位数')} ${fmtNum(item && item.median, 1)} · ${t('range', '范围')} ${fmtNum(item && item.min, 1)}-${fmtNum(item && item.max, 1)}</span></div>`).join('')}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${cohortText('Source provenance')}</div>
          <div class="setup-row"><span class="k">${cohortText('Source')}</span><span class="vv">${esc((review.source || {}).label || cohortText('Local export'))}</span></div>
          <div class="setup-row"><span class="k">${cohortText('Database')}</span><span class="vv">${esc((review.source || {}).database || cohortText('unknown'))}</span></div>
          <div class="setup-row"><span class="k">${cohortText('Path hash')}</span><span class="vv mono">${esc((review.source || {}).path_hash || '')}</span></div>
          <div class="setup-row"><span class="k">${cohortText('Scope')}</span><span class="vv">${esc((review.provenance || {}).payload_scope || 'cohort_aggregate_only')}</span></div>
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Real registered export aggregate. Row-level filters, generic Table One p-values, matched cohorts, and paired SOFA reclassification remain blocked; timed survival outcomes are handled in the KM module.', '真实注册导出聚合。行级筛选、通用 Table One p 值、匹配队列和配对 SOFA 重分层仍保持拦截；有事件时间的生存结局由 KM 模块处理。')}</p>`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && ws.summary) {
      const s = ws.summary;
      const ageBars = [
        ['Mean age', s.mean_age],
        ['Female %', s.female_pct],
        ['Mortality %', s.mortality],
        ['Sepsis-3 %', s.sepsis_pct],
      ];
      return `
      <div class="sec-stack"><div class="lbl">Cohort profile</div><h2>${t('Local export snapshot', '本地导出队列概览')}</h2></div>
      ${workspaceSamplingNote(s)}
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${t('Stays', '住院数')}</div><div class="val">${fmtInt(s.stays)}</div></div>
        <div class="stat"><div class="label">${t('Mean age', '平均年龄')}</div><div class="val">${fmtNum(s.mean_age, 1)}</div></div>
        <div class="stat"><div class="label">${t('Female', '女性')}</div><div class="val">${fmtPct(s.female_pct)}</div></div>
        <div class="stat"><div class="label">${t('Sepsis-3 +', 'Sepsis-3 阳性')}</div><div class="val">${fmtPct(s.sepsis_pct)}</div></div>
        <div class="stat"><div class="label">${t('Median SOFA-2', 'SOFA-2 中位数')}</div><div class="val">${fmtNum(s.median_sofa2, 1)}</div></div>
        <div class="stat accent"><div class="label">${t('Mortality', '死亡率')}</div><div class="val">${fmtPct(s.mortality)}</div></div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Export measures', '导出指标')}</div>
          ${ageBars.map(([lab, n]) => `<div class="qrow"><span>${lab}</span><div class="qbar"><span style="width:${n == null ? 0 : Math.max(0, Math.min(100, n))}%"></span></div><span class="qv">${fmtNum(n, 1)}</span></div>`).join('')}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Files loaded', '已加载文件')}</div>
          ${(ws.files || []).slice(0, 6).map(f => `<div class="setup-row"><span class="k">${esc(f.module || f.file)}</span><span class="vv">${fmtInt(f.rows)} rows</span></div>`).join('')}
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Real local export summary. Formal analyses still require the evidence-bound agent path.', '真实本地导出摘要。正式分析仍需走 evidence-bound agent 路径。')}</p>`;
    }
    const demoProfile = demoCohortClinicalProfile();
    // [label, bar% (0-100), displayed value, hint]. Bar length and the shown
    // value are decoupled so a raw SOFA score is scaled to its own max (0-24)
    // instead of sharing the prevalence-% axis at an arbitrary length.
    const domains = [
      [t('Severity', '严重程度'), 6 / 24 * 100, '6 / 24 SOFA-2', t('median severity', '严重程度中位数')],
      [t('Sepsis', 'Sepsis'), 60, '60%', t('incidence', '发生率')],
      [t('Ventilation', '机械通气'), 50, '50%', t('exposure', '暴露率')],
      [t('Vasopressors', '血管活性药'), 40, '40%', t('exposure', '暴露率')],
      [t('AKI', 'AKI'), 30, '30%', t('phenotype', '表型')],
      [t('Mortality', '死亡'), 20, '20%', t('event rate', '事件率')],
    ];
    return `
      <div class="sec-stack"><div class="lbl">Cohort profile</div><h2>${t('Demo clinical cohort profile', '演示临床队列画像')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${t('Patients', '患者数')}</div><div class="val">10</div></div>
        <div class="stat"><div class="label">${t('Median age', '年龄中位数')}</div><div class="val">56</div></div>
        <div class="stat"><div class="label">${t('Female', '女性')}</div><div class="val">70%</div></div>
        <div class="stat"><div class="label">${t('Sepsis-3 +', 'Sepsis-3 阳性')}</div><div class="val">60%</div></div>
        <div class="stat"><div class="label">${t('Median SOFA', 'SOFA 中位数')}</div><div class="val">6</div></div>
        <div class="stat accent"><div class="label">${t('Mortality', '死亡率')}</div><div class="val">20%</div></div>
      </div>
      <div class="card pad mt-16">
        <div class="sec-stack mini"><div class="lbl">${t('Clinical domains', '临床维度')}</div><h3>${t('What a real cohort profile should summarize', '真实队列画像应总结哪些信息')}</h3></div>
        ${cohortClinicalProfile(demoProfile)}
      </div>
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:8px;">${t('At-a-glance phenotype balance', '一屏临床表型概览')}</div>
        <div class="cprof-spark-grid">
          ${domains.map(([label, barPct, display, hint]) => `<div class="cprof-spark">
            <div class="cprof-spark-head"><span>${esc(label)}</span><b class="mono">${esc(display)}</b></div>
            <div class="cprof-bar"><span style="width:${Math.max(4, Math.min(100, barPct)).toFixed(1)}%"></span></div>
            <div class="cprof-d">${esc(hint)}</div>
          </div>`).join('')}
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Demo / seeded example values for UI preview — not a real run output.', '演示 / 示例数据，仅用于界面预览 —— 非真实运行结果。')}</p>`;
  }

  function cohortGroupsBody(reviewOverride) {
    const review = reviewOverride || cohortReview();
    if (review && review.summary) {
      const s = review.summary || {};
      const source = review.source || {};
      const supported = (review.groups || {}).supported || [];
      const blocked = (review.groups || {}).blocked || [];
      const active = supported.find(row => row.id === state.compare) || supported[0] || {};
      const activeGroups = active.groups || [];
      const activeProfile = active.profile || {};
      const profileColumns = activeProfile.columns || [];
      const profileRows = activeProfile.rows || [];
      const radio = (row) => `<label class="radio ${active.id === row.id ? 'on' : ''}" role="button" tabindex="0" data-cohort-comp="${esc(row.id)}"><span class="mk"></span> ${esc(cohortText(row.label || row.id))}</label>`;
      return `
      ${workspaceSamplingNote(s)}
      <div class="coh-jump">
        <button class="cj-card" data-cohgo="coverage">
          <span class="cj-ico">${icon('shield', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${cohortText('Coverage audit')}</span><span class="cj-d">${t('Review module coverage before analysis', '分析前审阅模块覆盖率')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
        <button class="cj-card" data-cohgo="snapshot">
          <span class="cj-ico">${icon('cohort', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${cohortText('Cohort profile')}</span><span class="cj-d">${t('Inspect real registered export aggregates', '查看后端计算的本地导出聚合')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Analysis table')}</div><h2>${cohortText('Real cohort aggregate')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Cohort size')}</div><div class="val">${fmtInt(s.cohort_size)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">${fmtPct(s.mortality_pct)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Median age')}</div><div class="val">${fmtNum(s.age && s.age.median, 1)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Median SOFA-2')}</div><div class="val">${fmtNum(s.sofa2 && s.sofa2.median, 1)}</div></div>
      </div>
      <div class="note mt-12"><div class="ico">${icon('folder', 14)}</div><div class="body"><div class="t">${cohortText('Local export cohort review ready')}</div><div class="d">${cohortText('Source')} ${esc(source.label || cohortText('Local export'))} · ${esc(source.database || cohortText('unknown'))} · ${cohortText('Path hash')} <span class="mono">${esc(source.path_hash || '')}</span> · ${cohortText('aggregate-only payload')}.</div></div></div>
      ${cohortRealModuleSummary(review)}
      ${cohortRealFeaturePicker(review)}

      <div class="sec-stack"><div class="lbl">${cohortText('Comparison')}</div><h2>${cohortText('Select descriptive split')}</h2></div>
      <div class="radio-row">
        ${supported.map(row => radio(row)).join('')}
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Summary')}</div><h2>${esc(cohortText(active.label || 'Descriptive split'))} ${cohortText('Overview')}</h2></div>
      <div class="cols-3">
        ${activeGroups.map((g, i) => `<div class="stat ${i === 0 ? 'accent' : ''}"><div class="label">${esc(cohortText(g.label))}</div><div class="val">${fmtInt(g.count)}</div><div style="font-size:11px;color:var(--ink-4);margin-top:4px;">${fmtPct(g.pct)}</div></div>`).join('')}
      </div>

      <div class="sec-stack"><div class="lbl">Table One</div><h2>${cohortText('Aggregate-only group characteristics')}</h2></div>
      <div class="table-wrap table-scroll cohort-table-one" data-cohort-table-one>
        <table class="eu-table">
          <thead><tr><th>${cohortText('Metric')}</th>${profileColumns.map(col => `<th class="num">${esc(cohortText(col))}</th>`).join('')}<th>${cohortText('Status')}</th></tr></thead>
          <tbody>
            ${profileRows.map(row => `<tr>
              <td class="key">${esc(cohortText(row.metric))}${row.unit ? ` <span class="mono" style="color:var(--ink-4);font-weight:500;">${esc(cohortText(row.unit))}</span>` : ''}</td>
              ${(row.values || []).map(value => `<td class="num">${cohortProfileValue(row, value)}</td>`).join('')}
              <td><span class="pill ok" style="height:20px;">${cohortText('descriptive')}</span></td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Fail-closed')}</div><h2>${cohortText('Blocked cohort functions')}</h2></div>
      <div class="cols-3">
        ${blocked.map(item => `<div class="stat"><div class="label">${esc(cohortText(item.id))}</div><div class="val" style="font-size:13px;line-height:1.35;font-family:var(--font-body);font-weight:600;">${esc(cohortText(item.status))}</div><div style="font-size:11px;color:var(--ink-4);margin-top:6px;">${esc(cohortReason(item.reason))}</div></div>`).join('')}
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('No row-level filters, generic Table One p-values, SMDs, matched cohort, or paired SOFA reclassification are exposed here. Use the Survival curves tab for audited KM/log-rank when timed outcomes exist.', '这里不开放行级筛选、通用 Table One p 值、SMD、匹配队列或配对 SOFA 重分层。若存在事件时间，请在「生存曲线」页查看已审计的 KM/log-rank。')}</p>`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && ws.cohort) {
      const s = ws.summary || {};
      const c = ws.cohort || {};
      const chars = c.characteristics || [];
      return `
      <div class="coh-jump">
        <button class="cj-card" data-cohgo="coverage">
          <span class="cj-ico">${icon('shield', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('Coverage audit', '覆盖审计')}</span><span class="cj-d">${t('Review module coverage before analysis', '分析前审阅模块覆盖率')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
        <button class="cj-card" data-cohgo="snapshot">
          <span class="cj-ico">${icon('cohort', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('Cohort profile', '队列画像')}</span><span class="cj-d">${t('Inspect the local export snapshot', '查看本地导出摘要')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Analysis table')}</div><h2>${cohortText('Local export group contrast')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Total stays')}</div><div class="val">${fmtInt(s.stays)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mean age')}</div><div class="val">${fmtNum(s.mean_age, 1)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Female %')}</div><div class="val">${fmtPct(s.female_pct)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">${fmtPct(s.mortality)}</div></div>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Summary')}</div><h2>${cohortText('Outcome groups')}</h2></div>
      <div class="cols-3">
        <div class="stat"><div class="label">${cohortText('Survived')}</div><div class="val">${fmtInt(c.survived)}</div></div>
        <div class="stat"><div class="label">${cohortText('Deceased')}</div><div class="val">${fmtInt(c.deceased)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Rows reviewed')}</div><div class="val">${fmtInt((ws.tableRows || []).length)}</div></div>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Table one')}</div><h2>${cohortText('Baseline characteristics comparison')}</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${cohortText('Characteristic')}</th><th class="num">${cohortText('Overall')}</th><th class="num">${cohortText('Survived')}</th><th class="num">${cohortText('Deceased')}</th></tr></thead>
          <tbody>
            ${chars.map(r => `<tr><td class="key">${esc(cohortText(r[0]))}</td>${r.slice(1).map(c => `<td class="num">${fmtNum(c, 2)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="viz-cap"><b>${t('How to read', '怎么读')}</b><span>${t('Each row is a baseline characteristic; columns summarize it per group. Rows that differ noticeably flag confounding to address when Guided Copilot prepares the formal analysis.', '每行是一个基线特征，各列是分组内的汇总值。差异明显的行提示存在混杂 —— 「研究引导」准备正式分析时需要处理它们。')}</span></div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Real local export summary. P-values and manuscript claims are intentionally withheld from this UI preview.', '真实本地导出摘要。此 UI 预览不会直接给出 p 值或稿件声明。')}</p>`;
    }
    const comparisons = {
      outcome: {
        title: 'Survived vs Deceased',
        groups: [['Survived', '8'], ['Deceased', '2'], ['Ratio', '80.0% / 20.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '52.1 (15.4)', '65.5 (17.0)', '0.31'],
          ['Male, n (%)', '3 (30.0)', '3 (37.5)', '0 (0.0)', '0.47'],
          ['SOFA, median', '6', '5', '11', '0.08'],
          ['Lactate, mmol/L', '2.4', '2.1', '4.8', '0.12'],
          ['ICU LOS, days', '5.6', '5.1', '8.4', '0.22'],
        ],
      },
      age: {
        title: 'Age Groups',
        groups: [['Age < 65', '6'], ['Age ≥ 65', '4'], ['Ratio', '60.0% / 40.0%']],
        table: [
          ['Mortality, n (%)', '2 (20.0)', '0 (0.0)', '2 (50.0)', '0.13'],
          ['SOFA, median', '6', '4', '8', '0.21'],
          ['Lactate, mmol/L', '2.4', '1.9', '3.2', '0.28'],
          ['ICU LOS, days', '5.6', '3.9', '7.8', '0.18'],
          ['Sepsis-3, n (%)', '6 (60.0)', '3 (50.0)', '3 (75.0)', '0.58'],
        ],
      },
      sex: {
        title: 'Male vs Female',
        groups: [['Female', '7'], ['Male', '3'], ['Ratio', '70.0% / 30.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '52.0 (14.1)', '61.3 (18.7)', '0.42'],
          ['Mortality, n (%)', '2 (20.0)', '2 (28.6)', '0 (0.0)', '0.51'],
          ['SOFA, median', '6', '6', '5', '0.74'],
          ['Lactate, mmol/L', '2.4', '2.2', '2.8', '0.68'],
          ['ICU LOS, days', '5.6', '5.8', '5.1', '0.81'],
        ],
      },
      los: {
        title: 'Short vs Long Stay',
        groups: [['LOS < 5d', '6'], ['LOS ≥ 5d', '4'], ['Ratio', '60.0% / 40.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '50.6 (13.9)', '61.1 (18.5)', '0.37'],
          ['Mortality, n (%)', '2 (20.0)', '0 (0.0)', '2 (50.0)', '0.13'],
          ['SOFA, median', '6', '4', '9', '0.09'],
          ['Lactate, mmol/L', '2.4', '1.8', '3.9', '0.11'],
          ['Ventilation, n (%)', '5 (50.0)', '2 (33.3)', '3 (75.0)', '0.29'],
        ],
      },
      sepsis: {
        title: 'Sepsis vs Non-sepsis',
        groups: [['Sepsis-3 +', '6'], ['Sepsis-3 -', '4'], ['Ratio', '60.0% / 40.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '58.1 (16.8)', '49.9 (14.7)', '0.45'],
          ['Mortality, n (%)', '2 (20.0)', '2 (33.3)', '0 (0.0)', '0.25'],
          ['SOFA, median', '6', '8', '3', '0.06'],
          ['Lactate, mmol/L', '2.4', '3.0', '1.6', '0.16'],
          ['ICU LOS, days', '5.6', '6.7', '3.9', '0.30'],
        ],
      },
      custom: {
        title: 'Custom Threshold',
        groups: [['Above threshold', '5'], ['Below threshold', '5'], ['Example', 'SOFA ≥ 6']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '60.4 (15.2)', '49.2 (15.7)', '0.33'],
          ['Mortality, n (%)', '2 (20.0)', '2 (40.0)', '0 (0.0)', '0.17'],
          ['Lactate, mmol/L', '2.4', '3.3', '1.6', '0.14'],
          ['ICU LOS, days', '5.6', '7.2', '4.0', '0.23'],
          ['Sepsis-3, n (%)', '6 (60.0)', '4 (80.0)', '2 (40.0)', '0.52'],
        ],
        note: 'Demo threshold uses SOFA ≥ 6. Real custom thresholds remain fail-closed until a bounded cohort-builder backend is available.',
      },
    };
    const comp = comparisons[state.compare] || comparisons.outcome;
    const radio = (key, label) => `<label class="radio ${state.compare === key ? 'on' : ''}" role="button" tabindex="0" data-cohort-comp="${key}"><span class="mk"></span> ${cohortText(label)}</label>`;
    return `
      <div class="coh-jump">
        <button class="cj-card" data-cohgo="coverage">
          <span class="cj-ico">${icon('shield', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('Coverage audit', '覆盖审计')}</span><span class="cj-d">${t('Check module coverage before it biases a denominator', '在偏差分母之前检查模块覆盖度')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
        <button class="cj-card" data-cohgo="sofa">
          <span class="cj-ico">${icon('refresh', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('SOFA reclassification', 'SOFA 重分层')}</span><span class="cj-d">${t('See who moves under the 2025 SOFA-2 standard', '看哪些患者在 2025 版 SOFA-2 下重分层')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Analysis table')}</div><h2>${cohortText('Group Contrast Table')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Total patients')}</div><div class="val">10</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mean age')}</div><div class="val">54.8</div></div>
        <div class="stat accent"><div class="label">${cohortText('Male %')}</div><div class="val">30.0%</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">20.0%</div></div>
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Comparison')}</div><h2>${cohortText('Select comparison mode')}</h2></div>
      <div class="radio-row">
        ${radio('outcome', 'Survived vs Deceased')}
        ${radio('age', 'Age Groups')}
        ${radio('sex', 'Male vs Female')}
        ${radio('los', 'Short vs Long Stay')}
        ${radio('sepsis', 'Sepsis vs Non-sepsis')}
        ${radio('custom', 'Custom Threshold')}
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Features')}</div><h2>${cohortText('Select feature modules')}</h2></div>
      ${cohortDemoFeaturePicker()}

      <div class="sec-stack"><div class="lbl">${cohortText('Summary')}</div><h2>${esc(cohortText(comp.title))} ${cohortText('Overview')}</h2></div>
      <div class="cols-3">
        ${comp.groups.map((g, i) => `<div class="stat ${i === 2 ? 'accent' : ''}"><div class="label">${esc(cohortText(g[0]))}</div><div class="val">${esc(g[1])}</div></div>`).join('')}
      </div>
      ${comp.note ? `<div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="d" style="margin:0;">${esc(cohortReason(comp.note))}</div></div></div>` : ''}

      <div class="sec-stack"><div class="lbl">${cohortText('Table one')}</div><h2>${cohortText('Baseline characteristics comparison')}</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${cohortText('Characteristic')}</th><th class="num">${cohortText('Overall')} (n=10)</th><th class="num">${esc(cohortText(comp.groups[0][0]))} (n=${esc(comp.groups[0][1])})</th><th class="num">${esc(cohortText(comp.groups[1][0]))} (n=${esc(comp.groups[1][1])})</th></tr></thead>
          <tbody>
            ${comp.table.map(r => `<tr><td class="key">${esc(cohortText(r[0]))}</td>${r.slice(1, -1).map(c => `<td class="num">${esc(c)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="viz-cap"><b>${t('How to read', '怎么读')}</b><span>${t('Each row is a baseline characteristic; columns summarize it per group. Rows that differ noticeably flag confounding to address when Guided Copilot prepares the formal analysis.', '每行是一个基线特征，各列是分组内的汇总值。差异明显的行提示存在混杂 —— 「研究引导」准备正式分析时需要处理它们。')}</span></div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Demo / seeded example values for UI preview — not a real run output.', '演示 / 示例数据，仅用于界面预览 —— 非真实运行结果。')}</p>`;
  }
  function init(config) {
    state = config.state;
    t = config.t;
    icon = config.icon;
    esc = config.esc;
    fmtInt = config.fmtInt;
    fmtNum = config.fmtNum;
    fmtPct = config.fmtPct;
    fmtP = config.fmtP;
    workspaceSamplingNote = config.workspaceSamplingNote;
    cohortCharts = config.cohortCharts;
    cohortReview = config.cohortReview;
    survival = window.EU_VIZ_COHORT_SURVIVAL;
    demo = window.EU_VIZ_COHORT_DEMO;
    survival.init({
      state, t, icon, esc, fmtInt, fmtNum, fmtPct, fmtP,
      cohortCharts, cohortReview, cohortText, cohortReason,
    });
    demo.init({ t, icon, esc, fmtInt, cohortText, demoRowsForModule });
  }

  window.EU_VIZ_COHORT_VIEW = {
    init,
    tabs: cohortTabs,
    panelBody: cohortPanelBody,
    text: cohortText,
    reason: cohortReason,
    demoCatalogScope: cohortDemoCatalogScope,
    coverageBody: cohortCoverageBody,
    survivalBody(review) { return survival.body(review); },
    sofaBody: cohortSofaBody,
    snapshotBody: cohortSnapshotBody,
    groupsBody: cohortGroupsBody,
    hasSofaGranularity(value) { return !!SOFA_MATRIX_GRANULARITIES[value]; },
  };
})();
