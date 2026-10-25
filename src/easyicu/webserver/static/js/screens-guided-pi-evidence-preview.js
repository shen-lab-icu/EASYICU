/* Owner: read-only renderer dispatch for digest-pinned Research Agent evidence. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function text(value, limit) { return String(value == null ? '' : value).slice(0, limit || 500); }
  function recordPurpose(name) {
    const purposes = {
      'run_status.json': tr('Run status', '运行状态'),
      'reviewer_report.json': tr('Scientific review', '科学审阅'),
      'reviewer_report_post_readiness.json': tr('Scientific review', '科学审阅'),
      'scientific_maturity_audit.json': tr('Open issues', '待处理问题'),
      'critique_report.json': tr('Step check', '步骤检查'),
    };
    return purposes[name] || '';
  }
  function title(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    if (recordPurpose(p.display_name)) return recordPurpose(p.display_name);
    if (p.renderer === 'json' && p.kind !== 'statistic') return tr('Evidence record', '证据记录');
    return text(p.display_name || p.evidence_id || tr('Evidence', '证据'), 160);
  }
  function kindLabel(payload) {
    const renderer = String(payload && payload.renderer || 'metadata');
    const labels = {
      code: tr('Code', '代码'), json: 'JSON', table: tr('Table', '表格'),
      metadata: tr('File', '文件'),
    };
    return labels[renderer] || tr('File', '文件');
  }
  function locatorView(locator) {
    const source = locator && typeof locator === 'object' ? locator : {};
    if (!source.pointer && (source.value === '' || source.value == null)) return '';
    return `<div class="gpi-evidence-locator"><div><span>${esc(tr('JSON pointer', 'JSON 指针'))}</span><code>${esc(text(source.pointer, 500) || '—')}</code></div><div><span>${esc(tr('Bound source value', '绑定源数值'))}</span><code>${esc(text(source.value, 500) || '—')}</code></div></div>`;
  }
  function relationLabel(relation) {
    const labels = {
      analysis_code: tr('Code that generated this record', '生成这条记录的代码'),
      input_data: tr('Declared input evidence', '登记的输入证据'),
      run_plan_authority: tr('Approved analysis plan', '运行所依据的分析计划'),
      run_research_context: tr('Research context', '研究问题与配置'),
      run_cohort_authority: tr('Cohort authority', '队列定义权威'),
    };
    return labels[String(relation || '')] || tr('Registered parent evidence', '已登记父证据');
  }
  function evidenceLink(row) {
    const item = row && typeof row === 'object' ? row : {};
    const evidenceId = text(item.evidence_id, 160);
    const sha256 = text(item.sha256, 64).toLowerCase();
    const status = String(item.status || 'unregistered');
    const label = relationLabel(item.relation);
    if (status !== 'registered' || !/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) {
      return `<div class="gpi-evidence-lineage-gap"><span>${esc(label)}</span><strong>${esc(evidenceId || tr('Not recorded', '未记录'))}</strong><small>${esc(status)}</small></div>`;
    }
    const fileName = text(item.display_name || evidenceId, 180);
    return `<button type="button" class="gpi-evidence-lineage-link" data-gpi-evidence-open data-evidence-id="${esc(evidenceId)}" data-evidence-sha256="${esc(sha256)}" data-evidence-kind="${esc(text(item.kind || 'artifact', 80))}" data-evidence-label="${esc(fileName)}"><span>${esc(label)}</span><strong>${esc(fileName)}</strong></button>`;
  }
  function recordView(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    return `<section class="gpi-evidence-record"><div class="gpi-evidence-record-head"><span>${esc(tr('Registered source record', '真实登记文件'))}</span><strong>${esc(text(p.display_name || p.evidence_id, 180))}</strong></div><p class="gpi-evidence-record-status">${esc(tr('Registry digest verified', '登记摘要已核对'))}</p></section>`;
  }
  function reproductionPathView(payload, rows) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const stages = [p.kind === 'code' ? tr('Code', '代码') : tr('Result', '结果')];
    if (rows.some(row => row && row.relation === 'analysis_code' && row.status === 'registered')) stages.push(tr('Code', '代码'));
    if (rows.some(row => row && row.relation === 'input_data' && row.status === 'registered')) stages.push(tr('Input', '输入'));
    if (p.run_authority && p.run_authority.status === 'recorded') stages.push(tr('Run', '运行'));
    return `<div class="gpi-evidence-path" aria-label="${esc(tr('Reproduction path', '复现路径'))}">${stages.map((stage, index) => `${index ? '<b aria-hidden="true">→</b>' : ''}<span>${esc(stage)}</span>`).join('')}</div>`;
  }
  function fileAuditView(payload, locator) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const facts = [
      [tr('File name', '文件名'), p.display_name],
      [tr('Run-relative file', '运行内真实文件'), p.relative_path],
      [tr('Analysis step', '生成步骤'), p.role],
      [tr('Producer', '生成 owner'), p.producer],
      [tr('Generation mode', '生成方式'), p.generation_mode],
      [tr('Preview status', '预览状态'), p.withheld_reason],
    ].filter(([, value]) => value !== '' && value != null);
    const rows = Array.isArray(p.declared_lineage) ? p.declared_lineage : [];
    const hasCode = rows.some(row => row && row.relation === 'analysis_code' && row.status === 'registered');
    const rawJson = p.renderer === 'json'
      ? `<details class="gpi-evidence-json-details"><summary>${esc(tr('Open raw JSON for audit', '展开原始 JSON 审计内容'))}</summary>${jsonView(p)}</details>` : '';
    return `<details class="gpi-evidence-audit-details"><summary>${esc(tr('Full file audit', '完整文件审计'))}</summary><div class="gpi-evidence-audit-body">${locatorView(locator)}${p.description ? `<p>${esc(text(p.description, 500))}</p>` : ''}<dl>${facts.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd><code>${esc(text(value, 500))}</code></dd></div>`).join('')}</dl><div class="gpi-evidence-record-identity"><span>${esc(text(p.kind || 'artifact', 80))}</span><code>${esc(text(p.evidence_id, 160))}</code><code>${esc(text(p.sha256, 64))}</code></div>${hasCode ? `<p class="gpi-evidence-line-note">${esc(tr('The run records the complete generating script, but not a field-to-line span.', '当前运行登记了完整生成脚本，但未登记字段到具体代码行的映射。'))}</p>` : ''}${rawJson}</div></details>`;
  }
  function declaredLineageView(payload) {
    const rows = Array.isArray(payload && payload.declared_lineage) ? payload.declared_lineage : [];
    return `<section class="gpi-evidence-lineage"><div class="gpi-evidence-section-head"><span>${esc(tr('Reproduction path', '复现路径'))}</span></div>${reproductionPathView(payload, rows)}${rows.length ? `<div class="gpi-evidence-lineage-list">${rows.map(evidenceLink).join('')}</div>` : `<p>${esc(tr('This record has no direct parent evidence registered.', '该文件没有登记直接父证据。'))}</p>`}</section>`;
  }
  function runAuthorityView(payload) {
    const authority = payload && payload.run_authority && typeof payload.run_authority === 'object' ? payload.run_authority : {};
    if (authority.status !== 'recorded') {
      return `<details class="gpi-evidence-authority is-missing"><summary>${esc(tr('Source records for this analysis', '本次分析的来源记录'))}</summary><p>${esc(tr('No valid run manifest was recorded for this evidence.', '该证据没有可用的运行清单记录。'))}</p></details>`;
    }
    const facts = [
      [tr('Run', '运行'), authority.run_id],
      ['Git', authority.git_sha ? `${authority.git_sha}${authority.git_dirty === true ? ' · dirty' : ''}` : ''],
      [tr('Runner image', '执行镜像'), authority.runner_image_digest],
      [tr('Environment identity', '环境身份'), authority.environment_identity_sha256],
      [tr('Prompt pack', '提示包'), authority.prompt_pack_sha256 || authority.prompt_pack_version],
      [tr('Execution identity', '执行身份'), authority.execution_identity_sha256],
    ].filter(([, value]) => value !== '' && value != null);
    const links = Array.isArray(authority.links) ? authority.links : [];
    return `<details class="gpi-evidence-authority" title="${esc(tr('Continue to upstream run provenance', '查看这次分析的运行来源'))}"><summary>${esc(tr('Source records for this analysis', '本次分析的来源记录'))}</summary><div class="gpi-evidence-authority-body"><dl>${facts.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd><code>${esc(text(value, 500))}</code></dd></div>`).join('')}</dl>${links.length ? `<div class="gpi-evidence-lineage-list">${links.map(evidenceLink).join('')}</div>` : `<p>${esc(tr('No source files were registered for this analysis.', '本次分析未登记可查看的来源文件。'))}</p>`}</div></details>`;
  }
  function codeView(payload) {
    const lines = text(payload.text, 600000).split('\n');
    return `<div class="gpi-evidence-code" role="region" aria-label="${esc(tr('Read-only evidence code', '只读证据代码'))}" tabindex="0"><ol>${lines.map(line => `<li><code>${esc(line || ' ')}</code></li>`).join('')}</ol></div>`;
  }
  function jsonView(payload) {
    let value = '';
    try { value = JSON.stringify(payload.value, null, 2); } catch (_error) { value = tr('JSON preview unavailable.', 'JSON 预览不可用。'); }
    return `<pre class="gpi-preview-code gpi-evidence-json" tabindex="0"><code>${esc(value)}</code></pre>`;
  }
  function scalar(value, limit) {
    if (value == null || value === '') return '—';
    if (typeof value === 'number' && Number.isFinite(value)) return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
    if (Array.isArray(value)) return value.slice(0, 8).map(item => scalar(item, 80)).join(', ');
    return text(value, limit || 180);
  }
  function percent(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(Math.abs(numeric) >= 10 ? 1 : 2)}%` : '—';
  }
  function confidenceInterval(low, high) {
    if (low == null || high == null) return '—';
    return `${percent(low)}–${percent(high)}`;
  }
  function confidenceLabel(level) {
    return typeof level === 'number' && Number.isFinite(level) && level > 0 && level < 1
      ? `${Number((level * 100).toFixed(4))}% CI` : tr('Confidence interval', '置信区间');
  }
  function recorded(value) {
    return typeof value === 'string' && value.trim() ? text(value)
      : typeof value === 'number' && Number.isFinite(value) ? String(value) : '';
  }
  function contextFacts(context) {
    return [
      [tr('Selected metric', '当前指标'), context.metric],
      [tr('Group', '分组'), context.group],
      [tr('Reference group', '参照组'), context.reference],
      [tr('Denominator / sample', '分母 / 样本量'), context.denominator],
      [tr('Unit', '单位'), context.unit],
      [tr('Time window / anchor', '时间窗 / 时间锚点'), context.window],
      [tr('Adjustment', '调整因素'), context.adjustment],
    ].map(([label, value]) => [label, recorded(value) || tr('Not recorded', '未记录')]);
  }
  function producerStatisticFacts(value, pointer) {
    const context = {};
    // These producers publish unversioned step summaries. A future schema is
    // not silently interpreted as today's contract.
    if (value.schema_version) return contextFacts(context);
    if (value.analysis_family === 'grouped_table_one' && pointer === '/cohort_n') {
      context.metric = tr('Table 1 cohort denominator', '表 1 队列分母');
      context.denominator = value.cohort_n;
    }
    const modelField = /^\/(primary_or|primary_estimate|adjusted_effect|primary_ci_low|primary_ci_high|n_total|n_events|primary_or_ci\/[01]|primary_estimate_interval\/[01])$/.exec(pointer);
    if (value.analysis_family === 'association' && modelField) {
      const field = modelField[1];
      const scale = {odds_ratio:tr('Odds ratio', '优势比（OR）'),risk_ratio:tr('Risk ratio', '风险比（RR）'),hazard_ratio:tr('Hazard ratio', '风险率比（HR）')}[value.effect_scale];
      context.metric = field === 'n_total' ? tr('Model sample', '模型样本量')
        : field === 'n_events' ? tr('Model outcome events', '模型结局事件数')
          : scale ? scale + (/low$|\/0$/.test(field) ? tr(' — interval lower bound', '：区间下限') : /high$|\/1$/.test(field) ? tr(' — interval upper bound', '：区间上限') : '') : '';
      const contrast = value.primary_contrast || {};
      context.group = contrast.exposure_level;
      context.reference = contrast.reference_level;
      context.denominator = value.n_total;
      const covariates = value.adjustment_covariates;
      if (Array.isArray(covariates) && covariates.every(item => typeof item === 'string' && item.trim())) {
        context.adjustment = covariates.length ? covariates.map(item => text(item, 100)).join(', ') : tr('Unadjusted', '未调整');
      }
      const receipt = value.landmark_runtime_receipt || {};
      if (receipt.schema_version === 'easyicu.landmark_categorical_association_runtime_receipt/1'
          && typeof receipt.landmark_hours === 'number' && Number.isFinite(receipt.landmark_hours) && receipt.landmark_hours >= 0) {
        context.window = tr('Landmark: ', '界标：') + receipt.landmark_hours + tr(' hours', ' 小时');
      }
    }
    const analysisRow = /^\/analysis_rows\/(0|[1-9]\d*)\/(odds_ratio|ci_low|ci_high|n_stays|n_deaths)$/.exec(pointer);
    if (value.analysis_family === 'association' && analysisRow && Array.isArray(value.analysis_rows)) {
      const row = value.analysis_rows[Number(analysisRow[1])];
      if (row && row.strategy === 'functional_form' && row.effect_measure === 'odds_ratio') {
        const field = analysisRow[2];
        context.metric = field === 'n_stays' ? tr('Model stays', '模型入住数') : field === 'n_deaths' ? tr('Model deaths', '模型死亡数')
          : tr('Odds ratio', '优势比（OR）') + (field === 'ci_low' ? tr(' — interval lower bound', '：区间下限') : field === 'ci_high' ? tr(' — interval upper bound', '：区间上限') : '');
        context.denominator = row.n_stays;
        // A sensitivity row cannot inherit the primary model's contrast,
        // adjustment roster, or time anchor from elsewhere in the file.
      }
    }
    const report = value.reportable_descriptive_results;
    if (report && report.schema_version === 'easyicu.absolute_risk_reporting/1') {
      if (value.analysis_family === 'absolute_risk_context' && pointer === '/n_total') {
        context.metric = tr('Absolute-risk analysis sample', '绝对风险分析样本量');
        context.denominator = value.n_total;
      }
      const overall = /^\/reportable_descriptive_results\/overall_outcome\/(n|event_n|risk_pct|risk_ci_low_pct|risk_ci_high_pct)$/.exec(pointer);
      const group = /^\/reportable_descriptive_results\/exposures\/(0|[1-9]\d*)\/groups\/(0|[1-9]\d*)\/(n|outcome_n|outcome_event_n|outcome_risk_pct|outcome_risk_ci_low_pct|outcome_risk_ci_high_pct|prevalence_pct|prevalence_ci_low_pct|prevalence_ci_high_pct)$/.exec(pointer);
      const exposure = group && Array.isArray(report.exposures) ? report.exposures[Number(group[1])] : null;
      const row = overall ? report.overall_outcome : exposure && Array.isArray(exposure.groups) ? exposure.groups[Number(group[2])] : null;
      const field = overall ? overall[1] : group ? group[3] : '';
      if (row && typeof row === 'object') {
        const prevalence = field.startsWith('prevalence_');
        const metrics = {
          n: tr('Records in this population', '当前人群记录数'),
          outcome_n: tr('Outcome denominator in this group', '该分组结局分母'),
          event_n: tr('Outcome events', '结局事件数'),
          outcome_event_n: tr('Outcome events in this group', '该分组结局事件数'),
        };
        context.metric = metrics[field] || (prevalence ? tr('Group share of the cohort', '分组占队列比例')
          : overall ? tr('Overall outcome proportion', '总体结局比例') : tr('Observed outcome proportion within the group', '组内观察到的结局比例'));
        if (field.includes('_ci_')) context.metric += field.includes('_low_') ? tr(' — interval lower bound', '：区间下限') : tr(' — interval upper bound', '：区间上限');
        context.denominator = overall ? row.n : prevalence ? value.n_total : field === 'n' ? row.n : row.outcome_n;
        context.group = overall ? tr('Overall', '总体') : recorded(row.label) || row.group_value;
        context.unit = field.endsWith('_pct') ? '%' : '';
      }
    }
    if (!Number.isSafeInteger(context.denominator) || context.denominator < 0) delete context.denominator;
    return contextFacts(context);
  }
  function selectedStatisticFacts(payload, pointer) {
    const value = payload.value || {};
    const estimates = value.descriptive_estimates;
    const match = /^\/descriptive_estimates\/(exposure_prevalence|outcome_absolute_risks)\/(0|[1-9]\d*)\/(n|events|denominator|estimate_pct)$/.exec(pointer);
    if (!match || !estimates || estimates.schema_version !== 'easyicu.exposure_outcome_descriptive_estimates/1') return producerStatisticFacts(value, pointer);
    const prevalence = match[1] === 'exposure_prevalence';
    if ((prevalence && match[3] === 'events') || (!prevalence && match[3] === 'n')) return contextFacts({});
    const rows = estimates[match[1]];
    const row = Array.isArray(rows) ? rows[Number(match[2])] : null;
    if (!row || typeof row !== 'object') return contextFacts({});
    const count = row[prevalence ? 'n' : 'events'];
    const denominator = row.denominator;
    const metric = {
      n: tr('Records in this group', '该分组记录数'),
      events: tr('Outcome events in this group', '该分组结局事件数'),
      denominator: prevalence ? tr('Cohort denominator', '队列分母') : tr('Outcome denominator in this group', '该分组结局分母'),
      estimate_pct: prevalence ? tr('Group share of the cohort', '分组占队列比例') : tr('Observed outcome proportion within the group', '组内观察到的结局比例'),
    }[match[3]];
    const facts = contextFacts({metric, group:typeof row.level === 'boolean' ? String(row.level) : row.level,
      denominator:Number.isSafeInteger(denominator) && denominator > 0 ? denominator : null,
      unit:match[3] === 'estimate_pct' ? '%' : ''});
    if (typeof value.exposure === 'string' && value.exposure) facts.push([tr('Exposure field', '分组变量'), text(value.exposure)]);
    if (typeof row.level === 'string' || typeof row.level === 'number' || typeof row.level === 'boolean') {
      facts.push([tr('Recorded group value', '原始分组值'), String(row.level)]);
    }
    if (!prevalence && typeof value.outcome === 'string' && value.outcome) facts.push([tr('Outcome field', '结局变量'), text(value.outcome)]);
    // These are direct siblings of the selected pointer, not a search for
    // matching numbers in the file and not a reconstructed denominator.
    if (Number.isSafeInteger(count) && Number.isSafeInteger(denominator)
        && count >= 0 && denominator > 0 && count <= denominator) {
      facts.push([prevalence ? tr('Group records / cohort records', '分组记录数 / 队列记录数') : tr('Events / outcome records in this group', '该分组事件数 / 有结局记录数'), `${scalar(count)} / ${scalar(denominator)}`]);
    }
    return facts;
  }
  function selectedNumberView(payload, locator) {
    if (!locator || !locator.pointer) return '';
    const pointer = String(locator.pointer);
    let value = payload.value;
    if (!pointer.startsWith('/') || /~(?:[^01]|$)/.test(pointer)) value = undefined;
    else for (const part of pointer.slice(1).split('/')) {
      const key = part.replace(/~1/g, '/').replace(/~0/g, '~');
      if (value == null || typeof value !== 'object' || !Object.prototype.hasOwnProperty.call(value, key)) {
        value = undefined;
        break;
      }
      value = value[key];
    }
    // Read the exact admitted source field, never search for the same number or
    // infer a percentage/unit. The clicked text is display-only, not authority.
    if (typeof value !== 'number' || !Number.isFinite(value) || String(value) !== String(locator.value)) {
      return `<div class="gpi-evidence-withheld" role="alert"><strong>${esc(tr('Selected number could not be matched', '所选数字未能与来源匹配'))}</strong><p>${esc(tr('Check the full evidence lineage; this preview does not verify the selected number.', '请核对完整证据链；此预览未确认所选数字。'))}</p></div>`;
    }
    const facts = [
      ...selectedStatisticFacts(payload, pointer),
      ...(locator.display ? [[tr('In the manuscript', '正文显示'), text(locator.display, 120)]] : []),
      [tr('Exact source value', '来源原值'), String(value)],
    ];
    return `<section class="gpi-evidence-statistic gpi-evidence-selected-number" aria-label="${esc(tr('Selected number source', '所选数字的来源'))}"><div class="gpi-evidence-statistic-head"><strong>${esc(tr('Selected number source', '所选数字的来源'))}</strong></div><div class="gpi-evidence-statistic-facts">${facts.map(([label, fact]) => `<div class="gpi-evidence-statistic-card"><span>${esc(label)}</span><strong>${esc(fact)}</strong></div>`).join('')}</div><p class="gpi-evidence-record-status">${esc(tr('Source field and value match', '来源字段与原值已核对'))}</p></section>`;
  }
  function statisticView(payload) {
    const value = payload && payload.value && typeof payload.value === 'object' && !Array.isArray(payload.value)
      ? payload.value : {};
    const report = value.reportable_descriptive_results && typeof value.reportable_descriptive_results === 'object'
      ? value.reportable_descriptive_results : value;
    const overall = report.overall_outcome && typeof report.overall_outcome === 'object'
      ? report.overall_outcome : {};
    const facts = [];
    const nTotal = value.n_total != null ? value.n_total : overall.n;
    if (nTotal != null) facts.push([tr('Sample', '样本量'), scalar(nTotal)]);
    const outcome = overall.outcome || value.outcome;
    if (outcome) facts.push([tr('Outcome', '结局'), scalar(outcome)]);
    if (overall.event_n != null || value.outcome_event_n != null) facts.push([tr('Events', '事件数'), scalar(overall.event_n != null ? overall.event_n : value.outcome_event_n)]);
    if (overall.risk_pct != null) facts.push([tr('Overall risk', '总体风险'), `${percent(overall.risk_pct)} (${confidenceInterval(overall.risk_ci_low_pct, overall.risk_ci_high_pct)})`]);
    if (value.estimate != null) facts.push([tr('Estimate', '估计值'), scalar(value.estimate)]);
    if (Array.isArray(value.ci) && value.ci.length >= 2) facts.push([confidenceLabel(value.confidence_level), `${scalar(value.ci[0])}–${scalar(value.ci[1])}`]);
    if (Array.isArray(value.exposure_columns) && value.exposure_columns.length) facts.push([tr('Exposure', '暴露'), scalar(value.exposure_columns)]);
    const factCards = facts.map(([label, fact]) => `<div class="gpi-evidence-statistic-card"><span>${esc(label)}</span><strong>${esc(fact)}</strong></div>`).join('');
    const method = value.method || report.method || '';
    const exposures = Array.isArray(report.exposures) ? report.exposures.slice(0, 12) : [];
    const exposureSections = exposures.map(exposure => {
      const groups = Array.isArray(exposure && exposure.groups) ? exposure.groups.slice(0, 24) : [];
      const distribution = exposure && exposure.continuous_distribution && typeof exposure.continuous_distribution === 'object'
        ? exposure.continuous_distribution : null;
      const groupTable = groups.length ? `<div class="gpi-evidence-table-wrap"><table class="gpi-evidence-table"><thead><tr><th>${esc(tr('Group', '分组'))}</th><th>${esc(tr('N', '例数'))}</th><th>${esc(tr('Events / N', '事件 / N'))}</th><th>${esc(tr('Risk', '风险'))}</th><th>${esc(tr('Confidence interval', '置信区间'))}</th></tr></thead><tbody>${groups.map(group => `<tr><td>${esc(scalar(group.label || group.group_value))}</td><td>${esc(scalar(group.n))}</td><td>${esc(`${scalar(group.outcome_event_n)} / ${scalar(group.outcome_n)}`)}</td><td>${esc(percent(group.outcome_risk_pct))}</td><td>${esc(confidenceInterval(group.outcome_risk_ci_low_pct, group.outcome_risk_ci_high_pct))}${typeof group.confidence_level === 'number' && group.confidence_level > 0 && group.confidence_level < 1 ? ` (${esc(confidenceLabel(group.confidence_level))})` : ''}</td></tr>`).join('')}</tbody></table></div>` : '';
      const distributionTable = distribution ? `<div class="gpi-evidence-table-wrap"><table class="gpi-evidence-table"><thead><tr><th>${esc(tr('N', '例数'))}</th><th>${esc(tr('Median', '中位数'))}</th><th>${esc(tr('IQR', '四分位距'))}</th><th>${esc(tr('Range', '范围'))}</th></tr></thead><tbody><tr><td>${esc(scalar(distribution.n))}</td><td>${esc(scalar(distribution.median))}</td><td>${esc(`${scalar(distribution.q25)}–${scalar(distribution.q75)}`)}</td><td>${esc(`${scalar(distribution.minimum)}–${scalar(distribution.maximum)}`)}</td></tr></tbody></table></div>` : '';
      if (!groupTable && !distributionTable) return '';
      return `<section class="gpi-evidence-statistic-section"><h3>${esc(scalar(exposure.exposure || tr('Exposure', '暴露')))}${exposure.unit ? ` <small>${esc(scalar(exposure.unit, 80))}</small>` : ''}</h3>${groupTable}${distributionTable}</section>`;
    }).join('');
    const readable = factCards || method || exposureSections
      ? `<div class="gpi-evidence-statistic"><div class="gpi-evidence-statistic-head"><span>${esc(tr('Readable result', '结果摘要'))}</span><strong>${esc(tr('Registered statistic', '已登记统计量'))}</strong></div>${factCards ? `<div class="gpi-evidence-statistic-facts">${factCards}</div>` : ''}</div>` : '';
    const details = method || exposureSections
      ? `<details class="gpi-evidence-result-details"><summary>${esc(tr('Method and complete result tables', '方法与完整结果表'))}</summary><div>${method ? `<p class="gpi-evidence-statistic-method"><strong>${esc(tr('Method', '方法'))}</strong>${esc(scalar(method, 500))}</p>` : ''}${exposureSections}</div></details>` : '';
    return `${readable}${details}`;
  }
  function tableView(payload) {
    const headers = Array.isArray(payload.headers) ? payload.headers.slice(0, 24) : [];
    const rows = Array.isArray(payload.rows) ? payload.rows.slice(0, 100) : [];
    return `<div class="gpi-evidence-table-wrap" tabindex="0"><table class="gpi-evidence-table"><thead><tr>${headers.map(value => `<th>${esc(text(value, 160))}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${headers.map((_, index) => `<td>${esc(text(Array.isArray(row) ? row[index] : '', 1000))}</td>`).join('')}</tr>`).join('')}</tbody></table>${payload.rows_truncated || payload.columns_truncated ? `<p>${esc(tr('Preview is bounded; additional rows or columns are not shown.', '只预览了一部分；其余行或列未展示。'))}</p>` : ''}</div>`;
  }
  function reviewRecordView(payload) {
    const value = payload.value && typeof payload.value === 'object' && !Array.isArray(payload.value) ? payload.value : {};
    const unknown = tr('Not recorded', '未记录');
    const states = {
      analysis_only:tr('Analysis only', '仅供分析'), manuscript_ready:tr('Manuscript generated', '稿件已生成'),
      publication_ready:tr('Publication ready', '具备投稿条件'), blocked:tr('Blocked', '未通过'),
      failed:tr('Failed', '未通过'), diagnostic_only:tr('Diagnostic only', '仅供诊断'),
      pass:tr('Passed', '通过'), needs_revision:tr('Open issues', '有待处理项'),
      major_revision:tr('Major revision required', '需重大修订'), minor_revision:tr('Minor revision required', '需小修'),
      accept:tr('Accepted by this review', '本轮审阅接受'), reject:tr('Rejected by this review', '本轮审阅不接受'),
    };
    const severity = {blocker:tr('Blocking', '阻断'),reject:tr('Reject', '拒绝'),major:tr('Major', '重大'),minor:tr('Minor', '次要'),info:tr('Information', '提示')};
    const facts = [];
    let status = '', findings = null, repairs = [], boundary = '';
    if (value.schema_version === 'easyicu.run_status/2') {
      status = value.status;
      const gates = value.gates && typeof value.gates === 'object' ? value.gates : {};
      for (const [key, en, zh] of [
        ['execution_complete','Analysis execution','分析执行'],['analysis_validated','Analysis validation','分析校验'],
        ['evidence_complete','Evidence completeness','证据完整性'],['numeric_verified','Numeric verification','数字核验'],
        ['manuscript_ready','Manuscript generation','稿件生成'],['paper_authorized','Publication permission','投稿使用权限'],
      ]) facts.push([tr(en, zh), gates[key] === true ? tr('Passed', '通过') : gates[key] === false ? tr('Not passed', '未通过') : unknown]);
      findings = Array.isArray(gates.scientific_maturity_findings) ? gates.scientific_maturity_findings : null;
    } else if (value.schema_version === 'easyicu.scientific_maturity/1') {
      status = value.status;
      findings = Array.isArray(value.findings) ? value.findings : null;
    } else if (!value.schema_version && /^reviewer_report(?:_post_readiness)?\.json$/.test(payload.display_name || '')
        && value.summary && typeof value.summary === 'object' && Array.isArray(value.critiques)) {
      status = value.summary.aggregated_recommendation;
      findings = value.critiques.flatMap(row => row && Array.isArray(row.comments) ? row.comments : []);
      const counts = value.summary.counts || {};
      Object.entries(severity).forEach(([key,label]) => {
        if (Number.isSafeInteger(counts[key]) && counts[key] >= 0) facts.push([label, String(counts[key])]);
      });
      if (value.summary.review_mode === 'simulated_deterministic') {
        boundary = tr('Simulated review; it is not independent external review or publication permission.', '模拟审阅；不代表独立外部审阅，也不授予投稿使用权限。');
      } else if (typeof value.summary.claim_boundary === 'string') boundary = value.summary.claim_boundary;
    } else if (!value.schema_version && payload.display_name === 'critique_report.json'
        && ['concerns','unsupported_claims','missing_evidence_refs','suggested_repairs'].every(key => Array.isArray(value[key]))) {
      status = value.status;
      findings = [...value.concerns, ...value.unsupported_claims].map(message => ({message}));
      repairs = value.suggested_repairs;
      facts.push([tr('Missing evidence references', '缺失证据引用'), String(value.missing_evidence_refs.length)]);
    }
    const wording = (message, label) => {
      const clean = recorded(message);
      if (!clean) return '';
      return window.EU_LANG === 'zh' && !/[\u3400-\u9fff]/.test(clean)
        ? `<details><summary>${esc(label || tr('Original wording', '原文（英文）'))}</summary><p>${esc(clean)}</p></details>` : `<p>${esc(clean)}</p>`;
    };
    facts.unshift([tr('Status', '状态'), states[status] || unknown]);
    facts.push([tr('Recorded findings', '已记录问题'), findings ? String(findings.length) : unknown]);
    const issues = (findings || []).slice(0, 20).filter(row => row && typeof row === 'object').map(row =>
      `<li>${severity[row.severity] ? `<strong>${esc(severity[row.severity])}</strong>` : ''}${wording(row.message)}${wording(row.remediation, tr('Suggested repair', '处理建议（英文）'))}</li>`).join('');
    const repairList = repairs.slice(0, 20).map(item => `<li>${wording(item)}</li>`).join('');
    return `<section class="gpi-evidence-statistic"><div class="gpi-evidence-statistic-head"><strong>${esc(title(payload))}</strong></div><div class="gpi-evidence-statistic-facts">${facts.map(([label,fact]) => `<div class="gpi-evidence-statistic-card"><span>${esc(label)}</span><strong>${esc(fact)}</strong></div>`).join('')}</div>${wording(boundary)}${issues ? `<ul>${issues}</ul>` : ''}${repairList ? `<h3>${esc(tr('Suggested repairs', '处理建议'))}</h3><ul>${repairList}</ul>` : ''}${findings && findings.length > 20 ? `<p>${esc(tr('Additional findings are retained in the full file audit.', '其余问题见完整文件审计。'))}</p>` : ''}</section>`;
  }
  function metadataView(payload) {
    const reasons = {
      patient_level_rows_withheld: tr('Patient-level cohort rows are withheld. File identity and digest remain visible.', '患者级队列行已隐藏；文件身份与摘要仍可核对。'),
      direct_identifier_columns_withheld: tr('This table contains direct identifier columns, so row preview is blocked.', '该表含直接标识符列，因此禁止预览数据行。'),
      non_result_table_rows_withheld: tr('Only result-owned aggregate tables can expose rows in this view.', '此处只允许预览分析结果所属的聚合表。'),
      preview_size_limit: tr('The file exceeds the bounded preview size.', '文件太大，无法预览。'),
      evidence_preview_host_path_detected: tr('The text contains a host path and was withheld.', '文本包含主机路径，已隐藏。'),
      evidence_preview_encoding_unsupported: tr('The text encoding is not supported for safe preview.', '文本编码不符合安全预览要求。'),
      unsupported_evidence_type: tr('Preview is unavailable for this file.', '此文件暂不支持预览。'),
    };
    const reason = String(payload.withheld_reason || 'unsupported_evidence_type');
    return `<div class="gpi-evidence-withheld"><strong>${esc(tr('File information', '文件信息'))}</strong><p>${esc(reasons[reason] || reasons.unsupported_evidence_type)}</p>${payload.bytes != null ? `<span>${esc(String(payload.bytes))} bytes</span>` : ''}</div>`;
  }
  function render(payload, locator) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const isStatistic = p.previewable && p.renderer === 'json' && p.kind === 'statistic' && !recordPurpose(p.display_name);
    let body = metadataView(p);
    if (p.previewable && p.renderer === 'code') body = codeView(p);
    else if (isStatistic) {
      const selected = selectedNumberView(p, locator);
      const wholeFile = statisticView(p);
      // A file-level outcome summary is not the meaning of every number in
      // that file. Keep it accessible, explicitly separated from the selection.
      body = selected + (selected && wholeFile
        ? `<details class="gpi-evidence-result-details"><summary>${esc(tr('Whole-file context (not the selected metric)', '整份结果背景（不是当前所选指标）'))}</summary>${wholeFile}</details>` : wholeFile);
    }
    else if (p.previewable && p.renderer === 'json') body = reviewRecordView(p);
    else if (p.previewable && p.renderer === 'table') body = tableView(p);
    const summaryFirst = p.previewable && p.renderer === 'json';
    return `<div class="gpi-evidence-view">${summaryFirst ? body : ''}${recordView(p)}${declaredLineageView(p)}${fileAuditView(p, locator)}${runAuthorityView(p)}${summaryFirst ? '' : body}</div>`;
  }

  window.EasyICU.guidedPi.declare('evidencePreview', { render, title, kindLabel });
})();
