/* Cross-DB result workspace owner.
   Owns result navigation and rendering only; setup, transport, and API calls
   remain in their existing owners. */
(function () {
  'use strict';

  const state = {
    tab: 'overview',
    scope: 'all',
    module: 'all',
    feature: null,
    query: '',
  };
  const TABS = new Set(['overview', 'coverage', 'distributions', 'quality']);
  const SCOPES = new Set(['core', 'all']);
  const chartOwner = window.EU_CROSSDB_CHARTS;

  function helpersOf(config) {
    return Object.assign({
      catalogFeatureMeta: key => ({ name: key, unit: '' }),
      catalogModuleLabel: key => key,
      esc: value => String(value == null ? '' : value),
      fmtInt: value => String(value == null ? '—' : value),
      fmtNum: value => String(value == null ? '—' : value),
      fmtPct: value => value == null ? '—' : `${value}%`,
      icon: () => '',
      metricLabel: key => key,
      statusLabel: key => key,
      t: english => english,
      term: value => value,
    }, config && config.helpers || {});
  }

  function palette(index) {
    const colors = [
      'var(--accent)',
      'oklch(62% 0.11 255)',
      'oklch(64% 0.10 35)',
      'oklch(62% 0.10 145)',
      'oklch(58% 0.10 300)',
      'oklch(60% 0.10 75)',
    ];
    return colors[index % colors.length];
  }

  function beginCharts() {
    if (chartOwner && typeof chartOwner.begin === 'function') chartOwner.begin();
  }

  function featureKey(module, row) {
    return `${module.module || 'module'}::${row.feature || row.label || 'feature'}`;
  }

  function cleanModules(payload) {
    return (payload && payload.feature_distributions || [])
      .filter(module => module && Array.isArray(module.features) && module.features.length);
  }

  function featureCounts(modules, coreSet) {
    let all = 0;
    let core = 0;
    let shared = 0;
    (modules || []).forEach(module => {
      (module.features || []).forEach(row => {
        all += 1;
        if (coreSet.has(String(row.feature || '').toLowerCase())) core += 1;
        const values = row.values || [];
        if (values.length && values.every(value => value && value.present)) shared += 1;
      });
    });
    return { all, core, shared };
  }

  function scopedModules(payload, config) {
    const modules = cleanModules(payload);
    const coreSet = new Set((config && config.coreFeatures || []).map(value => String(value || '').toLowerCase()));
    const counts = featureCounts(modules, coreSet);
    const useCore = state.scope === 'core' && counts.core > 0;
    const query = state.query.trim().toLowerCase();
    return {
      counts,
      modules: modules.map(module => ({
        ...module,
        features: (module.features || []).filter(row => {
          const key = String(row.feature || '').toLowerCase();
          if (useCore && !coreSet.has(key)) return false;
          if (state.module !== 'all' && module.module !== state.module) return false;
          if (!query) return true;
          const meta = helpersOf(config).catalogFeatureMeta(row.feature) || {};
          return `${key} ${meta.name || ''} ${module.module || ''}`.toLowerCase().includes(query);
        }),
      })).filter(module => module.features.length),
    };
  }

  function selectedFeature(payload, config) {
    const scoped = scopedModules(payload, config);
    const rows = [];
    scoped.modules.forEach(module => {
      (module.features || []).forEach(row => rows.push({ module, row, key: featureKey(module, row) }));
    });
    let selected = rows.find(item => item.key === state.feature) || rows[0] || null;
    if (selected && selected.key !== state.feature) state.feature = selected.key;
    return { ...scoped, rows, selected };
  }

  function formatDensity(value) {
    if (value == null || !Number.isFinite(Number(value))) return '—';
    const number = Number(value);
    if (Math.abs(number) >= 1000) return Math.round(number).toLocaleString();
    if (Math.abs(number) >= 100) return number.toFixed(0);
    if (Math.abs(number) >= 10) return number.toFixed(1);
    return number.toFixed(2);
  }

  function metricValue(key, value, helpers) {
    if (value == null) return '—';
    if (['stays', 'cohort_size', 'modules', 'total_rows', 'total_records', 'feature_rows', 'concepts_present'].includes(key)) {
      return helpers.fmtInt(value);
    }
    if (['female_pct', 'mortality', 'mortality_pct', 'sepsis_pct', 'coverage_median_pct'].includes(key)) {
      return helpers.fmtPct(value);
    }
    return helpers.fmtNum(value, 1);
  }

  function sourceLabels(payload) {
    return (payload && payload.sources || []).map(source => source.label || source.database || 'local');
  }

  function catalogScope(payload, config) {
    const modules = cleanModules(payload);
    const counts = featureCounts(modules, new Set((config && config.coreFeatures || []).map(value => String(value || '').toLowerCase())));
    const totals = config && config.catalogTotals || {};
    const totalModules = Number(totals.modules) || modules.length;
    const totalFeatures = Number(totals.features) || counts.all;
    const provenance = payload && payload.provenance || {};
    const raw = payload && payload.source_type === 'raw_database_root';
    const runScope = String(provenance.feature_scope || '');
    const partial = raw && runScope !== 'all_catalog';
    return {
      counts,
      modules,
      partial,
      runScope,
      totalFeatures,
      totalModules,
    };
  }

  function partialScopeNotice(payload, config) {
    const h = helpersOf(config);
    const scope = catalogScope(payload, config);
    if (!scope.partial) return '';
    return `<div class="note warn mt-16" data-crossdb-partial-scope>
      <div class="ico">${h.icon('alert', 15)}</div>
      <div class="body">
        <div class="t">${h.t('This is a quick core run, not the full clinical catalog', '当前是快速核心运行，不是完整临床目录')}</div>
        <div class="d">${h.t('This result computed', '本次结果仅计算')} ${h.fmtInt(scope.modules.length)} / ${h.fmtInt(scope.totalModules)} ${h.t('modules and', '个模块、')} ${h.fmtInt(scope.counts.all)} / ${h.fmtInt(scope.totalFeatures)} ${h.t('feature profiles. Choose the explicit full-catalog scope to compare every mapped concept with bounded sampling.', '个特征剖面。选择显式的完整目录范围后，才会以有界抽样比较全部映射概念。')}</div>
        <div class="mt-10"><button class="btn sm primary" type="button" data-crossdb-expand-scope>${h.icon('layers', 12)} ${h.t('Set up full-catalog comparison', '设置完整目录对比')}</button></div>
      </div>
    </div>`;
  }

  function statusBar(payload, config) {
    const h = helpersOf(config);
    const sources = payload.sources || [];
    const scope = catalogScope(payload, config);
    const moduleText = scope.partial
      ? `${h.fmtInt(scope.modules.length)} / ${h.fmtInt(scope.totalModules)}`
      : h.fmtInt(scope.modules.length);
    const featureText = scope.partial
      ? `${h.fmtInt(scope.counts.all)} / ${h.fmtInt(scope.totalFeatures)}`
      : h.fmtInt(scope.counts.all);
    return `<div class="xdb-result-status" role="status">
      <span class="pill ${scope.partial ? 'warn' : 'ok'}"><span class="dot"></span>${scope.partial ? h.t('Quick scope', '快速范围') : h.t('Ready', '已就绪')}</span>
      <div>
        <b>${h.t('Cross-database consistency workspace', '跨库一致性检查工作区')}</b>
        <span>${h.fmtInt(sources.length)} ${h.t('sources', '个数据源')} · ${moduleText} ${h.t('modules', '个模块')} · ${featureText} ${h.t('available feature profiles', '个可用特征剖面')}</span>
      </div>
    </div>`;
  }

  function tabBar(config) {
    const h = helpersOf(config);
    const tabs = [
      ['overview', h.t('Overview', '概览')],
      ['coverage', h.t('Coverage', '覆盖')],
      ['distributions', h.t('Distributions', '分布')],
      ['quality', h.t('Quality & provenance', '质量与溯源')],
    ];
    return `<div class="xdb-result-tabs" role="tablist" aria-label="${h.esc(h.t('Cross-database result sections', '跨库结果分区'))}">
      ${tabs.map(([key, label]) => `<button type="button" role="tab" class="${state.tab === key ? 'active' : ''}" data-crossdb-result-tab="${key}" aria-selected="${state.tab === key ? 'true' : 'false'}">${h.esc(label)}</button>`).join('')}
    </div>`;
  }

  function recordCards(payload, config) {
    const h = helpersOf(config);
    return `<div class="xdb-source-summary-grid">
      ${(payload.sources || []).map((source, index) => {
        const summary = source.summary || {};
        const records = summary.total_records != null ? summary.total_records : summary.cohort_size;
        return `<article class="xdb-source-summary" style="--xdb-source-color:${palette(index)}">
          <div><span class="xdb-source-dot"></span><b>${h.esc(source.label || source.database || `${h.t('Source', '来源')} ${index + 1}`)}</b></div>
          <strong>${records == null ? '—' : h.fmtInt(records)}</strong>
          <span>${h.t('records in this bounded comparison', '条记录纳入本次有界对比')}</span>
        </article>`;
      }).join('')}
    </div>`;
  }

  function overview(payload, config) {
    const h = helpersOf(config);
    const modules = cleanModules(payload);
    const coreSet = new Set((config && config.coreFeatures || []).map(value => String(value || '').toLowerCase()));
    const counts = featureCounts(modules, coreSet);
    const gate = payload.compatibility_gate || {};
    const sharedModules = payload.shared_modules || [];
    return `
      <section class="xdb-result-section" data-crossdb-result-panel="overview">
        <div class="xdb-kpi-grid">
          <article><span>${h.t('Data sources', '数据源')}</span><strong>${h.fmtInt((payload.sources || []).length)}</strong><small>${h.t('independent exports or databases', '独立导出或数据库')}</small></article>
          <article><span>${h.t('Shared modules', '共享模块')}</span><strong>${h.fmtInt(sharedModules.length)}</strong><small>${h.t('available across every selected source', '在所有所选来源中可用')}</small></article>
          <article><span>${h.t('Comparable feature profiles', '可对比特征剖面')}</span><strong>${h.fmtInt(counts.shared)}</strong><small>${h.t('present in every selected source', '在全部所选来源中存在')}</small></article>
          <article><span>${h.t('Comparison mode', '比较模式')}</span><strong class="textual">${h.esc(h.statusLabel(gate.comparison_mode || 'descriptive_only'))}</strong><small>${h.esc(h.statusLabel(gate.status || 'compatible'))}</small></article>
        </div>
        ${partialScopeNotice(payload, config)}
        ${recordCards(payload, config)}
        <div class="viz-cap"><b>${h.t('What this shows', '这组数字是什么')}</b><span>${h.t('Per-source record volume in this bounded comparison. It is a size reference, not an outcome result.', '本次有界对比中各来源的记录规模；它只是样本量参照，不是结局结果。')}</span></div>
        <div class="note info mt-16">
          <div class="ico">${h.icon('benchmark', 16)}</div>
          <div class="body">
            <div class="t">${h.t('What this page can answer', '这个页面能回答什么')}</div>
            <div class="d">${h.t('It checks whether the same clinical concepts exist and have comparable aggregate distributions across databases. It does not prove matched cohorts or permit inferential claims.', '它用于检查相同临床概念在不同数据库中是否存在、聚合分布是否可比；不证明队列已经匹配，也不允许直接做推断性结论。')}</div>
          </div>
        </div>
        <div class="nextbar accent mt-16">
          <div class="nb-ico">${h.icon('arrow', 16)}</div>
          <div class="grow"><div class="nb-t">${h.t('Comparison checked — what’s next?', '跨库对比已检查 —— 下一步？')}</div><div class="nb-d">${h.t('Review one cohort in detail or carry this bounded comparison into an analysis plan.', '深入审阅一个队列，或把这次有界对比带入分析计划。')}</div></div>
          <button class="btn" data-nav="cohort">${h.icon('cohort', 13)} ${h.t('Back to Cohort Statistics', '返回队列统计')}</button>
          <button class="btn primary" data-nav="agent">${h.icon('agent', 13)} ${h.t('Create analysis plan', '创建分析计划')}</button>
        </div>
      </section>`;
  }

  function availabilityCell(value, config) {
    const h = helpersOf(config);
    if (!value || !value.present) return `<td class="num xdb-avail-cell missing">${h.t('Missing', '缺失')}</td>`;
    const percent = typeof value.coverage_pct === 'number' ? value.coverage_pct : null;
    const tone = percent == null ? 'present' : (percent >= 80 ? 'high' : (percent >= 50 ? 'medium' : 'low'));
    return `<td class="num xdb-avail-cell ${tone}">${percent == null ? h.t('Present', '存在') : h.fmtPct(percent)}</td>`;
  }

  function coverage(payload, config) {
    const h = helpersOf(config);
    const labels = sourceLabels(payload);
    const rows = payload.availability || [];
    const scope = catalogScope(payload, config);
    return `
      <section class="xdb-result-section" data-crossdb-result-panel="coverage">
        <div class="xdb-section-head">
          <div><h2>${h.t('Module coverage matrix', '模块覆盖矩阵')}</h2><p>${h.t('Start here to see which clinical domains can be compared before inspecting individual feature distributions.', '先查看哪些临床模块能够跨库比较，再进入单个特征的分布检查。')}</p></div>
          <span class="pill">${h.fmtInt(rows.length)}${scope.partial ? ` / ${h.fmtInt(scope.totalModules)}` : ''} ${h.t('modules audited', '个模块已审计')}</span>
        </div>
        ${partialScopeNotice(payload, config)}
        <div class="table-wrap table-scroll">
          <table class="eu-table">
            <thead><tr><th>${h.t('Module', '模块')}</th>${labels.map(label => `<th class="num">${h.esc(label)}</th>`).join('')}<th class="num">${h.t('Shared', '共享')}</th></tr></thead>
            <tbody>${rows.map(row => `<tr><td class="key">${h.esc(h.catalogModuleLabel(row.module))}</td>${(row.values || []).map(value => availabilityCell(value, config)).join('')}<td class="num"><span class="pill ${row.shared ? 'ok' : 'warn'}">${row.shared ? h.t('Yes', '是') : h.t('No', '否')}</span></td></tr>`).join('')}</tbody>
          </table>
        </div>
        <div class="xdb-shared-modules">
          <b>${h.t('Shared exported modules', '共享导出模块')}</b>
          <div>${(payload.shared_modules || []).length ? (payload.shared_modules || []).map(module => `<span class="chip solid">${h.esc(h.catalogModuleLabel(module))}</span>`).join('') : `<span class="pill warn">${h.t('No shared modules detected', '未检测到共享模块')}</span>`}</div>
        </div>
      </section>`;
  }

  function numericChart(item, labels, config) {
    const h = helpersOf(config);
    if (!chartOwner || typeof chartOwner.render !== 'function') return '';
    return chartOwner.render(item, labels, h);
  }

  function featureDetail(item, labels, config) {
    if (!item) return '';
    const h = helpersOf(config);
    const meta = h.catalogFeatureMeta(item.row.feature) || {};
    const values = item.row.values || [];
    return `<article class="xdb-feature-detail">
      <div class="xdb-section-head">
        <div><h2>${h.esc(meta.name || item.row.feature)}</h2><p>${h.esc(h.catalogModuleLabel(item.module.module))} · <span class="mono">${h.esc(item.row.feature)}</span>${meta.unit ? ` · ${h.esc(meta.unit)}` : ''}</p></div>
        <span class="pill dashed">${h.fmtInt(values.filter(value => value && value.present).length)} / ${h.fmtInt(labels.length)} ${h.t('sources present', '个来源存在')}</span>
      </div>
      ${numericChart(item, labels, config)}
      <div class="viz-cap"><b>${h.t('How to read', '怎么读')}</b><span>${h.t('Overlapping curves suggest similar aggregate measurement distributions. Shifted or missing curves can indicate unit, definition, or coverage differences that should be checked before pooling.', '曲线重叠表示聚合测量分布较一致；曲线错位或缺失可能提示单位、定义或覆盖差异，合并分析前应进一步核查。')}</span></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${h.t('Source', '来源')}</th><th class="num">${h.t('Non-null values', '非空值')}</th><th class="num">${h.t('Observed range', '观测范围')}</th><th class="num">${h.t('Summary points', '摘要点')}</th></tr></thead>
          <tbody>${values.map((value, index) => `<tr>
            <td class="key"><i class="xdb-table-dot" style="background:${palette(index)}"></i>${h.esc(labels[index] || value.source || `${h.t('Source', '来源')} ${index + 1}`)}</td>
            <td class="num">${value.present ? h.fmtInt(value.non_null || value.n || 0) : h.t('Missing', '缺失')}</td>
            <td class="num">${value.present && value.min != null && value.max != null ? `${formatDensity(value.min)} – ${formatDensity(value.max)}` : '—'}</td>
            <td class="num">${Array.isArray(value.points) ? h.fmtInt(value.points.length) : (Array.isArray(value.categories) ? h.fmtInt(value.categories.length) : '—')}</td>
          </tr>`).join('')}</tbody>
        </table>
      </div>
    </article>`;
  }

  function distributions(payload, config) {
    const h = helpersOf(config);
    const modules = cleanModules(payload);
    const view = selectedFeature(payload, config);
    const allCount = view.counts.all;
    const coreCount = view.counts.core;
    const catalog = catalogScope(payload, config);
    const moduleOptions = [`<option value="all">${h.t('All modules', '全部模块')}</option>`]
      .concat(modules.map(module => `<option value="${h.esc(module.module)}" ${state.module === module.module ? 'selected' : ''}>${h.esc(h.catalogModuleLabel(module.module))} (${h.fmtInt((module.features || []).length)})</option>`))
      .join('');
    return `
      <section class="xdb-result-section" data-crossdb-result-panel="distributions">
        <div class="xdb-section-head">
          <div><h2>${h.t('Feature distribution comparison', '特征分布对比')}</h2><p>${h.t('Choose one feature at a time. The main chart overlays aggregate source distributions; the list covers the complete mapped catalog available in this run.', '每次选择一个特征。主图叠加各来源的聚合分布；左侧列表覆盖本次运行中可用的完整映射目录。')}</p></div>
          <span class="pill">${h.fmtInt(view.rows.length)} ${h.t('matching features', '个匹配特征')}</span>
        </div>
        <div class="xdb-feature-toolbar">
          <div class="xdb-scope-switch" role="group" aria-label="${h.esc(h.t('Feature scope', '特征范围'))}">
            <button type="button" class="${state.scope === 'core' ? 'active' : ''}" data-crossdb-scope="core">${h.t('Core concepts', '核心概念')} <span>${h.fmtInt(coreCount)}</span></button>
            <button type="button" class="${state.scope === 'all' ? 'active' : ''}" data-crossdb-scope="all">${catalog.partial ? h.t('All in this run', '本次全部特征') : h.t('All mapped features', '全部映射特征')} <span>${h.fmtInt(allCount)}</span></button>
          </div>
          <label><span>${h.t('Module', '模块')}</span><select data-crossdb-module>${moduleOptions}</select></label>
          <label class="xdb-feature-search"><span>${h.t('Search', '搜索')}</span><input type="search" value="${h.esc(state.query)}" data-crossdb-feature-query placeholder="${h.esc(h.t('Name or feature key', '名称或特征键'))}"></label>
        </div>
        ${partialScopeNotice(payload, config)}
        <div class="xdb-feature-workspace">
          <nav class="xdb-feature-list" aria-label="${h.esc(h.t('Available feature profiles', '可用特征剖面'))}">
            ${view.rows.length ? view.rows.map(item => {
              const meta = h.catalogFeatureMeta(item.row.feature) || {};
              const present = (item.row.values || []).filter(value => value && value.present).length;
              return `<button type="button" class="${state.feature === item.key ? 'selected' : ''}" data-crossdb-feature="${h.esc(item.key)}">
                <span><b>${h.esc(meta.name || item.row.feature)}</b><small>${h.esc(h.catalogModuleLabel(item.module.module))} · ${h.esc(item.row.feature)}</small></span>
                <em>${h.fmtInt(present)}/${h.fmtInt(sourceLabels(payload).length)}</em>
              </button>`;
            }).join('') : `<div class="xdb-feature-empty">${h.t('No features match the current filters.', '当前筛选条件下没有匹配特征。')}</div>`}
          </nav>
          ${view.selected ? featureDetail(view.selected, sourceLabels(payload), config) : ''}
        </div>
      </section>`;
  }

  function quality(payload, config) {
    const h = helpersOf(config);
    const labels = sourceLabels(payload);
    const provenance = payload.provenance || {};
    const privacy = payload.privacy || {};
    const blocked = payload.blocked_features || [];
    const gate = payload.compatibility_gate || {};
    const sourceMode = payload.source_type === 'raw_database_root' ? h.t('root hash', '根目录哈希') : h.t('path hash', '路径哈希');
    return `
      <section class="xdb-result-section" data-crossdb-result-panel="quality">
        <div class="xdb-section-head">
          <div><h2>${h.t('Quality, scope, and provenance', '质量、范围与溯源')}</h2><p>${h.t('Review source identity, aggregate summaries, and fail-closed restrictions before exporting or planning downstream analysis.', '导出或规划下游分析前，核查来源身份、聚合摘要和默认拦截范围。')}</p></div>
          <span class="pill ${gate.status === 'compatible' ? 'ok' : 'warn'}">${h.esc(h.statusLabel(gate.status || 'compatible'))}</span>
        </div>
        <div class="xdb-provenance-grid">
          ${(payload.sources || []).map(source => `<article>
            <span>${h.icon('db', 15)}</span>
            <div><b>${h.esc(source.label || h.t('Local source', '本地来源'))}</b><small>${h.esc((source.database || 'local').toUpperCase())} · ${sourceMode} <span class="mono">${h.esc(source.path_hash || '—')}</span></small></div>
          </article>`).join('')}
        </div>
        <div class="table-wrap table-scroll mt-16">
          <table class="eu-table">
            <thead><tr><th>${h.t('Aggregate metric', '聚合指标')}</th>${labels.map(label => `<th class="num">${h.esc(label)}</th>`).join('')}<th class="num">${h.t('Range', '范围')}</th></tr></thead>
            <tbody>${(payload.rows || []).map(row => `<tr><td class="key">${h.esc(h.metricLabel(row.label || row.key))}</td>${(row.values || []).map(value => `<td class="num">${metricValue(row.key, value, h)}</td>`).join('')}<td class="num">${metricValue(row.key, row.delta, h)}</td></tr>`).join('')}</tbody>
          </table>
        </div>
        <div class="note warn mt-16">
          <div class="ico">${h.icon('lock', 16)}</div>
          <div class="body">
            <div class="t">${h.t('Fail-closed analysis scope', '默认拦截的分析范围')}</div>
            <div class="d">${blocked.length ? blocked.map(item => h.esc(h.statusLabel(item.id))).join(' · ') : h.t('Inferential analyses remain blocked', '推断性分析保持拦截')} · ${h.t('raw rows returned', '返回原始行')}=${privacy.raw_rows_returned === true ? 'true' : 'false'} · ${h.t('inference', '推断')}=${h.esc(h.statusLabel(provenance.inference || 'blocked_until_numeric_evidence_gate'))}</div>
          </div>
        </div>
      </section>`;
  }

  function render(payload, config) {
    if (!payload) return '';
    beginCharts();
    let body = '';
    if (state.tab === 'coverage') body = coverage(payload, config);
    else if (state.tab === 'distributions') body = distributions(payload, config);
    else if (state.tab === 'quality') body = quality(payload, config);
    else body = overview(payload, config);
    return `${statusBar(payload, config)}${tabBar(config)}${body}`;
  }

  function repaint(config) {
    if (config && typeof config.repaint === 'function') config.repaint();
  }

  function bind(root, payload, config) {
    if (!root) return;
    root.querySelectorAll('[data-crossdb-result-tab]').forEach(button => button.addEventListener('click', () => {
      const next = button.dataset.crossdbResultTab;
      if (!TABS.has(next) || next === state.tab) return;
      state.tab = next;
      repaint(config);
    }));
    root.querySelectorAll('[data-crossdb-scope]').forEach(button => button.addEventListener('click', () => {
      const next = button.dataset.crossdbScope;
      if (!SCOPES.has(next) || next === state.scope) return;
      state.scope = next;
      state.feature = null;
      repaint(config);
    }));
    root.querySelectorAll('[data-crossdb-module]').forEach(select => select.addEventListener('change', () => {
      state.module = select.value || 'all';
      state.feature = null;
      repaint(config);
    }));
    root.querySelectorAll('[data-crossdb-feature-query]').forEach(input => {
      input.addEventListener('keydown', event => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        state.query = input.value || '';
        state.feature = null;
        repaint(config);
      });
      input.addEventListener('change', () => {
        state.query = input.value || '';
        state.feature = null;
        repaint(config);
      });
      input.addEventListener('search', () => {
        state.query = input.value || '';
        state.feature = null;
        repaint(config);
      });
    });
    root.querySelectorAll('[data-crossdb-feature]').forEach(button => button.addEventListener('click', () => {
      state.feature = button.dataset.crossdbFeature || null;
      repaint(config);
    }));
    root.querySelectorAll('[data-crossdb-expand-scope]').forEach(button => button.addEventListener('click', () => {
      if (config && typeof config.expandScope === 'function') config.expandScope(payload);
    }));
    root.querySelectorAll('[data-crossdb-export]').forEach(button => button.addEventListener('click', () => {
      if (config && typeof config.exportPayload === 'function') config.exportPayload(payload);
    }));
  }

  function mount(root) {
    return chartOwner && typeof chartOwner.mount === 'function' ? chartOwner.mount(root) : 0;
  }

  function reset() {
    if (chartOwner && typeof chartOwner.dispose === 'function') chartOwner.dispose();
    state.tab = 'overview';
    state.scope = 'all';
    state.module = 'all';
    state.feature = null;
    state.query = '';
  }

  window.EU_CROSSDB_RESULTS = {
    bind,
    mount,
    render,
    reset,
    snapshot: () => ({ ...state }),
  };
})();
