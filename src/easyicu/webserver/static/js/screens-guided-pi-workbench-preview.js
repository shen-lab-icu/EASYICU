/* Guided Pi data-workbench preview owner.
   Renders one digest-bound, result-blind data-package review. It never owns
   StudyContext state and never applies cohort/filter changes. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  const view = { query: '', status: 'all' };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function number(value) {
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }
  function percent(value) {
    const numeric = number(value);
    if (numeric == null) return null;
    return Math.max(0, Math.min(100, numeric));
  }
  function statusLabel(value) {
    const labels = {
      ready: tr('Ready', '可用'),
      partial: tr('Partial', '部分可用'),
      not_extracted: tr('Not extracted', '未提取'),
      semantic_review_required: tr('Review required', '需要语义审阅'),
    };
    return labels[String(value || '')] || String(value || tr('Unknown', '未知'));
  }
  function statusClass(value) {
    return ['ready', 'partial', 'not_extracted', 'semantic_review_required'].includes(String(value || ''))
      ? String(value) : 'unknown';
  }
  function roleLabel(value) {
    const labels = {
      outcome: tr('Outcome', '结局'),
      plan_input: tr('Plan input', '计划变量'),
      supporting_variable: tr('Supporting variable', '辅助变量'),
    };
    return labels[String(value || '')] || String(value || '');
  }
  function reasonLabel(value) {
    const labels = {
      plan_bound_column_complete: tr('Complete for all analysis rows', '全部分析记录均有值'),
      plan_bound_column_has_missing_values: tr('Some analysis rows are missing values', '部分分析记录缺少数值'),
      plan_bound_null_count_unavailable: tr('Missingness requires review', '缺失情况需要进一步检查'),
    };
    return labels[String(value || '')] || String(value || '');
  }
  function conceptRows(payload) {
    const rows = Array.isArray(payload && payload.concepts) ? payload.concepts : [];
    const query = view.query.trim().toLowerCase();
    return rows.filter(row => {
      const status = String(row && row.availability_status || 'unknown');
      if (view.status !== 'all' && status !== view.status) return false;
      if (!query) return true;
      const haystack = [row.study_role, row.concept_id, row.module, row.reason_code]
        .map(value => String(value || '').toLowerCase()).join(' ');
      return haystack.includes(query);
    });
  }
  function summaryCard(label, value, detail) {
    return `<div class="gpi-wb-card"><span>${esc(label)}</span><strong>${esc(value)}</strong>${detail ? `<small>${esc(detail)}</small>` : ''}</div>`;
  }
  function qualityStrip(payload) {
    const quality = payload && payload.quality && typeof payload.quality === 'object' ? payload.quality : {};
    const rows = [
      ['ok', Number(quality.modules_ok || 0), tr('OK', '正常')],
      ['warn', Number(quality.modules_warn || 0), tr('Watch', '关注')],
      ['bad', Number(quality.modules_bad || 0), tr('Blocked', '阻断')],
      ['unknown', Number(quality.modules_unknown || 0), tr('Unknown', '未知')],
    ];
    const total = rows.reduce((sum, row) => sum + row[1], 0);
    if (!total) return `<p class="gpi-wb-muted">${esc(tr('No module-quality summary was issued.', '尚未生成模块质量摘要。'))}</p>`;
    return `<div class="gpi-wb-quality" role="img" aria-label="${esc(tr('Module quality distribution', '模块质量分布'))}">
      <div class="gpi-wb-quality-bar">${rows.filter(row => row[1] > 0).map(row => `<span class="${row[0]}" style="width:${(row[1] / total * 100).toFixed(2)}%" title="${esc(row[2] + ': ' + row[1])}"></span>`).join('')}</div>
      <div class="gpi-wb-legend">${rows.map(row => `<span><i class="${row[0]}"></i>${esc(row[2])} <strong>${row[1]}</strong></span>`).join('')}</div>
    </div>`;
  }
  function conceptTable(payload) {
    const rows = conceptRows(payload);
    if (!rows.length) return `<div class="gpi-wb-empty">${esc(tr('No execution concept matches this view.', '当前筛选下没有执行概念。'))}</div>`;
    return `<div class="gpi-wb-table" role="table" aria-label="${esc(tr('Execution concept availability', '执行概念可用性'))}">
      <div class="gpi-wb-row head" role="row"><span>${esc(tr('Role / concept', '角色 / 概念'))}</span><span>${esc(tr('Availability', '可用性'))}</span><span>${esc(tr('Observed coverage', '观测覆盖'))}</span></div>
      ${rows.map(row => {
        const denominator = number(row.denominator_count);
        const evaluable = number(row.evaluable_count);
        const pct = denominator && evaluable != null ? percent(evaluable / denominator * 100) : percent(row.physical_coverage_pct);
        const coverage = pct == null ? tr('Owner receipt required', '需要 owner 回执') : `${pct.toFixed(1)}%`;
        return `<div class="gpi-wb-row" role="row">
          <span><strong>${esc(row.concept_id || tr('Unnamed concept', '未命名概念'))}</strong><small>${esc(roleLabel(row.study_role))}${row.module ? ` · ${esc(row.module)}` : ''}</small></span>
          <span><em class="gpi-wb-status ${statusClass(row.availability_status)}">${esc(statusLabel(row.availability_status))}</em><small>${esc(reasonLabel(row.reason_code))}</small></span>
          <span><b>${esc(coverage)}</b>${pct == null ? '' : `<i class="gpi-wb-meter"><u style="width:${pct.toFixed(2)}%"></u></i>`}</span>
        </div>`;
      }).join('')}
    </div>`;
  }
  function moduleList(payload) {
    const rows = Array.isArray(payload && payload.configured_modules) ? payload.configured_modules : [];
    if (!rows.length) return `<span class="gpi-wb-muted">${esc(tr('No configured modules', '尚未配置模块'))}</span>`;
    return `<div class="gpi-wb-modules">${rows.map(row => `<span class="${statusClass(row.availability_status)}"><strong>${esc(row.module || '')}</strong><small>${esc(statusLabel(row.availability_status))}</small></span>`).join('')}</div>`;
  }
  function readinessChart(payload) {
    const rows = (Array.isArray(payload && payload.concepts) ? payload.concepts : [])
      .map(row => {
        const denominator = number(row && row.denominator_count);
        const evaluable = number(row && row.evaluable_count);
        const coverage = denominator && evaluable != null
          ? percent(evaluable / denominator * 100)
          : percent(row && row.physical_coverage_pct);
        return { row, coverage };
      })
      .filter(item => item.coverage != null)
      .sort((left, right) => left.coverage - right.coverage)
      .slice(0, 8);
    if (!rows.length) return `<p class="gpi-wb-muted">${esc(tr('Coverage will appear after the source owner publishes aggregate receipts.', '数据源 owner 发布聚合回执后，这里会显示覆盖情况。'))}</p>`;
    return `<div class="gpi-wb-coverage-chart" role="img" aria-label="${esc(tr('Lowest observed coverage among planned variables', '计划变量中覆盖率最低的字段'))}">
      ${rows.map(item => `<div class="gpi-wb-coverage-row"><span title="${esc(item.row.concept_id || '')}">${esc(item.row.concept_id || tr('Unnamed concept', '未命名概念'))}</span><i><u style="width:${item.coverage.toFixed(2)}%"></u></i><strong>${item.coverage.toFixed(1)}%</strong></div>`).join('')}
    </div>`;
  }
  function pendingAnalysisSteps() {
    const rows = [
      tr('Build the final analytic cohort and apply the Plan\'s time-zero and eligibility rules', '生成最终分析队列，并应用计划中的时间起点与纳排规则'),
      tr('Apply missing-data handling, variable coding, and any prespecified transformations', '执行缺失数据处理、变量编码及预先设定的转换'),
      tr('Fit the models and produce result tables and figures', '拟合模型，并生成结果表与图表'),
    ];
    return `<div class="gpi-wb-pending"><strong>${esc(tr('Runs only after approval', '批准后才会执行'))}</strong><ol>${rows.map(row => `<li>${esc(row)}</li>`).join('')}</ol></div>`;
  }
  function render(payload) {
    const denominator = payload && payload.denominator && number(payload.denominator.count);
    const modules = Array.isArray(payload && payload.configured_modules) ? payload.configured_modules : [];
    const concepts = Array.isArray(payload && payload.concepts) ? payload.concepts : [];
    const readyCount = concepts.filter(row => row && row.availability_status === 'ready').length;
    const quality = payload && payload.quality && typeof payload.quality === 'object' ? payload.quality : {};
    const cohort = payload && payload.cohort_review && typeof payload.cohort_review === 'object' ? payload.cohort_review : {};
    const source = payload && payload.source && typeof payload.source === 'object' ? payload.source : {};
    const postPlan = payload && payload.review_stage === 'post_plan';
    const isReady = payload && ['ready_for_plan', 'ready_for_analysis'].includes(payload.status);
    return `<section class="gpi-wb" data-gpi-workbench>
      <header class="gpi-wb-intro"><div><span class="gpi-wb-eyebrow">${esc(tr('Embedded data workbench', '嵌入式数据工作台'))}</span><h3>${esc(postPlan ? tr('Pre-analysis data readiness check', '分析前数据准备检查') : tr('Registered source review', '已登记数据源检查'))}</h3><p>${esc(postPlan ? tr('This result-blind view confirms the reusable source and availability of Plan-named variables. It is not the final analytic cohort and no preprocessing or model has run.', '这个不看结果的视图用于确认可复用数据源及计划变量是否可用；它不是最终分析队列，尚未运行预处理或模型。') : tr('This is a read-only projection of the registered export. It cannot silently change the cohort or reveal analysis results.', '这是已登记数据源的只读投影，不能静默修改队列，也不会提前泄露分析结果。'))}</p></div><em class="${isReady ? 'ready' : 'blocked'}">${esc(isReady ? (postPlan ? tr('Source ready', '数据源已准备') : tr('Ready for plan', '可进入计划')) : tr('Review blocked', '审阅受阻'))}</em></header>
      <div class="gpi-wb-cards">
        ${summaryCard(postPlan ? tr('Source denominator', '来源分母') : tr('Registered denominator', '登记分母'), denominator == null ? '—' : denominator.toLocaleString(), payload && payload.denominator && payload.denominator.analysis_unit)}
        ${summaryCard(postPlan ? tr('Bound source modules', '已绑定数据模块') : tr('Configured modules', '已配置模块'), modules.length, modules.filter(row => row && row.availability_status === 'ready').length + ' ' + tr('ready', '可用'))}
        ${summaryCard(postPlan ? tr('Planned variables', '计划变量') : tr('Execution concepts', '执行概念'), `${readyCount}/${concepts.length}`, tr('fully available', '完全可用'))}
        ${summaryCard(tr('Quality watchlist', '质量关注项'), Number(quality.watchlist_count || 0), quality.median_coverage_pct == null ? '' : `${tr('median coverage', '中位覆盖')} ${quality.median_coverage_pct}%`)}
      </div>
      <div class="gpi-wb-context"><div><span>${esc(tr('Registered source', '已登记数据源'))}</span><strong>${esc(source.label || source.database || 'EasyICU')}</strong></div><div><span>${esc(postPlan ? tr('Source population', '来源人群') : tr('Cohort', '队列'))}</span><strong>${esc(cohort.label || tr('Bound StudyContext cohort', 'StudyContext 已绑定队列'))}</strong></div></div>
      ${postPlan ? pendingAnalysisSteps() : ''}
      <section class="gpi-wb-section"><div class="gpi-wb-section-head"><div><h4>${esc(postPlan ? tr('Planned-variable coverage', '计划变量覆盖情况') : tr('Module quality', '模块质量'))}</h4><p>${esc(tr('Aggregate availability only; no event rates or effect estimates.', '仅展示聚合可用性，不展示事件率或效应量。'))}</p></div></div>${postPlan ? readinessChart(payload) : ''}${qualityStrip(payload)}${moduleList(payload)}</section>
      <section class="gpi-wb-section"><div class="gpi-wb-section-head"><div><h4>${esc(tr('Execution-concept screening', '执行概念筛选'))}</h4><p>${esc(tr('Filter this view locally. Scientific filter changes must be proposed and confirmed in chat.', '此处筛选只改变视图；科学筛选修改必须回到对话提出并确认。'))}</p></div></div>
        <div class="gpi-wb-controls"><label><span>${esc(tr('Find concept', '查找概念'))}</span><input type="search" data-gpi-wb-query value="${esc(view.query)}" placeholder="${esc(tr('Role, concept, module…', '角色、概念、模块…'))}" /></label><label><span>${esc(tr('Status', '状态'))}</span><select data-gpi-wb-status><option value="all">${esc(tr('All', '全部'))}</option>${['ready', 'partial', 'semantic_review_required', 'not_extracted'].map(value => `<option value="${value}" ${view.status === value ? 'selected' : ''}>${esc(statusLabel(value))}</option>`).join('')}</select></label></div>
        <div data-gpi-wb-results>${conceptTable(payload)}</div>
      </section>
      <footer class="gpi-wb-foot"><strong>${esc(tr('Want to change the cohort or filters?', '需要修改队列或筛选条件？'))}</strong><span>${esc(tr('Describe the change in the conversation. EasyICU will create a typed proposal and ask for confirmation before updating StudyContext or extracting data.', '请在对话中描述修改。EasyICU 会生成结构化提案，并在更新 StudyContext 或提取数据前请求确认。'))}</span></footer>
    </section>`;
  }
  function mount(host, payload) {
    if (!host) return;
    host.innerHTML = render(payload || {});
    const query = host.querySelector('[data-gpi-wb-query]');
    const status = host.querySelector('[data-gpi-wb-status]');
    const results = host.querySelector('[data-gpi-wb-results]');
    const repaint = () => { if (results) results.innerHTML = conceptTable(payload || {}); };
    if (query) query.addEventListener('input', () => { view.query = query.value; repaint(); });
    if (status) status.addEventListener('change', () => { view.status = status.value; repaint(); });
  }

  window.EU_GUIDED_PI_WORKBENCH_PREVIEW = { mount };
})();
