/* Guided Pi data-workbench preview owner.
   Renders one digest-bound, result-blind data-package review. It never owns
   StudyContext state and never applies cohort/filter changes. */
(function () {
  'use strict';

  const view = { query: '', status: 'all' };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
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
          <span><strong>${esc(row.concept_id || tr('Unnamed concept', '未命名概念'))}</strong><small>${esc(row.study_role || '')}${row.module ? ` · ${esc(row.module)}` : ''}</small></span>
          <span><em class="gpi-wb-status ${statusClass(row.availability_status)}">${esc(statusLabel(row.availability_status))}</em><small>${esc(row.reason_code || '')}</small></span>
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
  function render(payload) {
    const denominator = payload && payload.denominator && number(payload.denominator.count);
    const modules = Array.isArray(payload && payload.configured_modules) ? payload.configured_modules : [];
    const concepts = Array.isArray(payload && payload.concepts) ? payload.concepts : [];
    const ready = concepts.filter(row => row && row.availability_status === 'ready').length;
    const quality = payload && payload.quality && typeof payload.quality === 'object' ? payload.quality : {};
    const cohort = payload && payload.cohort_review && typeof payload.cohort_review === 'object' ? payload.cohort_review : {};
    const source = payload && payload.source && typeof payload.source === 'object' ? payload.source : {};
    return `<section class="gpi-wb" data-gpi-workbench>
      <header class="gpi-wb-intro"><div><span class="gpi-wb-eyebrow">${esc(tr('Embedded data workbench', '嵌入式数据工作台'))}</span><h3>${esc(tr('Screen the analysis package without leaving the conversation', '无需离开对话即可审阅分析数据包'))}</h3><p>${esc(tr('This is a read-only projection of the registered export. It cannot silently change the cohort or reveal analysis results.', '这是已登记数据源的只读投影，不能静默修改队列，也不会提前泄露分析结果。'))}</p></div><em class="${payload && payload.status === 'ready_for_plan' ? 'ready' : 'blocked'}">${esc(payload && payload.status === 'ready_for_plan' ? tr('Ready for plan', '可进入计划') : tr('Review blocked', '审阅受阻'))}</em></header>
      <div class="gpi-wb-cards">
        ${summaryCard(tr('Analysis denominator', '分析分母'), denominator == null ? '—' : denominator.toLocaleString(), payload && payload.denominator && payload.denominator.analysis_unit)}
        ${summaryCard(tr('Configured modules', '已配置模块'), modules.length, modules.filter(row => row && row.availability_status === 'ready').length + ' ' + tr('ready', '可用'))}
        ${summaryCard(tr('Execution concepts', '执行概念'), `${ready}/${concepts.length}`, tr('fully available', '完全可用'))}
        ${summaryCard(tr('Quality watchlist', '质量关注项'), Number(quality.watchlist_count || 0), quality.median_coverage_pct == null ? '' : `${tr('median coverage', '中位覆盖')} ${quality.median_coverage_pct}%`)}
      </div>
      <div class="gpi-wb-context"><div><span>${esc(tr('Registered source', '已登记数据源'))}</span><strong>${esc(source.label || source.database || 'EasyICU')}</strong></div><div><span>${esc(tr('Cohort', '队列'))}</span><strong>${esc(cohort.label || tr('Bound StudyContext cohort', 'StudyContext 已绑定队列'))}</strong></div></div>
      <section class="gpi-wb-section"><div class="gpi-wb-section-head"><div><h4>${esc(tr('Module quality', '模块质量'))}</h4><p>${esc(tr('Aggregate availability only; no event rates or effect estimates.', '仅展示聚合可用性，不展示事件率或效应量。'))}</p></div></div>${qualityStrip(payload)}${moduleList(payload)}</section>
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
