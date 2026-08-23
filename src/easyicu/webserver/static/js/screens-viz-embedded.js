/* Read-only embedded form of the native Patient/Cohort/Cross-DB workspaces.
   It reuses their renderers; it does not own transport or scientific data. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function fmt(value, digits) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: digits == null ? 1 : digits }) : '—';
  }
  function routeOf(view) {
    if (view === 'patient_timeline') return 'patient';
    if (view === 'crossdb_comparison') return 'crossdb';
    return 'cohort';
  }
  function metaOf(view) {
    const route = routeOf(view);
    if (route === 'patient') return {
      eyebrow: tr('Native Patient Review', '原生患者审阅'),
      title: tr('Patient timeline', '患者时间序列'),
      open: tr('Open full Patient Review', '打开完整患者审阅'),
    };
    if (route === 'crossdb') return {
      eyebrow: tr('Native Cross-DB workspace', '原生跨库工作台'),
      title: tr('Cross-database comparison', '跨数据库比较'),
      open: tr('Open full Cross-DB workspace', '打开完整跨库工作台'),
    };
    return {
      eyebrow: tr('Native Cohort Statistics', '原生队列统计'),
      title: view === 'feature_distribution' ? tr('Cohort and selected features', '队列与已选特征') : tr('Cohort review', '队列审阅'),
      open: tr('Open full Cohort Statistics', '打开完整队列统计'),
    };
  }
  function sourceLabel(payload, route) {
    const source = payload && payload.source || {};
    if (source.label || source.database) return source.label || source.database;
    if (route === 'crossdb') {
      const labels = (payload && payload.sources || []).map(row => row && (row.label || row.database)).filter(Boolean);
      if (labels.length) return labels.join(' · ');
    }
    return tr('Registered EasyICU export', '已登记 EasyICU 导出');
  }
  function shellHead(payload, view) {
    const route = routeOf(view); const meta = metaOf(view);
    return `<header class="gpi-viz-embed-head">
      <div><span>${esc(meta.eyebrow)}</span><h3>${esc(meta.title)}</h3><p>${esc(sourceLabel(payload, route))}</p></div>
      <button class="btn sm" type="button" data-gpi-open-native data-nav="${route}">${window.icon ? window.icon('arrow', 13) : ''} ${esc(meta.open)}</button>
    </header>`;
  }
  function tabs(items, active) {
    return `<div class="gpi-viz-embed-tabs" role="tablist">${items.map(item => `<button type="button" role="tab" data-gpi-embedded-tab="${item[0]}" aria-selected="${item[0] === active}">${esc(item[1])}</button>`).join('')}</div>`;
  }
  function featureDistributions(payload) {
    const rows = Array.isArray(payload && payload.selected_feature_distributions) ? payload.selected_feature_distributions : [];
    if (!rows.length) return `<div class="state empty"><div class="t">${esc(tr('No selected-feature distribution is available.', '暂无已选特征分布。'))}</div></div>`;
    return `<div class="sec-stack"><div class="lbl">${esc(tr('Selected features', '已选特征'))}</div><h2>${esc(tr('Stay-level descriptive distributions', '住院级描述性分布'))}</h2></div>
      <div class="cols-2">${rows.map(row => {
        const bins = Array.isArray(row.bins) && row.bins.length ? row.bins.map(bin => ({ label: `${fmt(bin.low, 2)}–${fmt(bin.high, 2)}`, count: Number(bin.count) || 0 })) : (row.categories || []).map(item => ({ label: item.label || '', count: Number(item.count) || 0 }));
        const max = Math.max(1, ...bins.map(item => item.count));
        return `<section class="card pad gpi-viz-feature-card"><div class="panel-head"><div><div class="eyebrow">${esc(row.module || tr('Feature', '特征'))}</div><div class="panel-title">${esc(row.label || row.column || '')}</div></div><span class="pill ok">${fmt(row.observed_pct, 1)}%</span></div>
          <div class="gpi-viz-feature-bars">${bins.map(item => `<div><span>${esc(item.label)}</span><i><u style="width:${(item.count / max * 100).toFixed(2)}%"></u></i><b>${fmt(item.count, 0)}</b></div>`).join('')}</div>
          <div class="viz-cap"><b>${esc(tr('How to read', '怎么读'))}</b><span>${esc(tr('Descriptive stay-level aggregation from the Cohort owner; no inferential comparison.', '来自 Cohort owner 的住院级描述性聚合；不包含推断性比较。'))}</span></div></section>`;
      }).join('')}</div>`;
  }
  function cohortBody(payload, view, state) {
    const context = window.EU_VIZ_CONTEXT || {};
    const items = [
      ['groups', tr('Group contrast', '组间对照')],
      ['profile', tr('Cohort profile', '队列画像')],
      ['coverage', tr('Coverage', '覆盖审计')],
      ['survival', tr('Survival', '生存曲线')],
      ['sofa', tr('SOFA', 'SOFA 重分层')],
    ];
    if ((payload.selected_feature_distributions || []).length) items.splice(1, 0, ['features', tr('Selected features', '已选特征')]);
    const active = state.cohortPanel || (view === 'feature_distribution' && items.some(item => item[0] === 'features') ? 'features' : 'groups');
    const body = active === 'features' ? featureDistributions(payload) : (context.renderCohortPanel ? context.renderCohortPanel(payload, active) : '');
    return `${tabs(items, active)}<div data-gpi-native-body data-gpi-native-cohort="${active}">${body}</div>`;
  }
  function patientBody(payload, state) {
    const context = window.EU_VIZ_CONTEXT || {};
    const owner = window.EU_PATIENT_SERIES;
    const mode = state.patientMode === 'lanes' ? 'lanes' : 'single';
    const review = payload.trajectory_review || {};
    const lanes = Array.isArray(review.lanes) ? review.lanes : (payload.time_lanes || []);
    const body = owner && typeof owner.renderTimeSeriesWorkspace === 'function' ? owner.renderTimeSeriesWorkspace({
      drill: payload, review, lanes, selected: payload.selected || {}, mode,
    }, context.patientSeriesHelpers ? context.patientSeriesHelpers() : {}) : '';
    return `${tabs([['single', tr('Trajectory gallery', '轨迹画廊')], ['lanes', tr('Module overview', '模块总览')]], mode)}<div data-gpi-native-body data-gpi-native-patient="${mode}">${body}</div>`;
  }
  function crossdbBody(payload, repaint) {
    const context = window.EU_VIZ_CONTEXT || {};
    const owner = window.EU_CROSSDB_RESULTS;
    const config = context.crossdbResultsConfig ? context.crossdbResultsConfig(repaint) : { repaint };
    return owner && typeof owner.render === 'function' ? `<div data-gpi-native-body data-gpi-native-crossdb>${owner.render(payload, config)}</div>` : '';
  }
  function render(payload, view, state) {
    const cleanView = String(view || 'cohort_summary');
    const local = state || {};
    const route = routeOf(cleanView);
    const body = route === 'patient' ? patientBody(payload || {}, local)
      : route === 'crossdb' ? crossdbBody(payload || {}, local.repaint)
      : cohortBody(payload || {}, cleanView, local);
    return `<div class="gpi-viz-embed" data-gpi-viz-embed data-gpi-viz-route="${route}">${shellHead(payload || {}, cleanView)}<div class="gpi-viz-embed-body">${body}</div><footer><strong>${esc(tr('Native renderer · Read-only snapshot', '原生渲染器 · 只读快照'))}</strong><span>${esc(tr('Use the conversation to change requirements, or explicitly open the full workspace to continue there.', '如需改变需求请继续对话，或明确打开完整工作台继续操作。'))}</span></footer></div>`;
  }
  function handoff(route, payload) {
    const context = window.EU_VIZ_CONTEXT || {};
    if (context.hydratePreview) context.hydratePreview(route, payload);
    const bridge = window.EU_GUIDED_HANDOFF;
    if (bridge && typeof bridge.set === 'function') bridge.set({
      target_route: route,
      prefill: {
        question_hint: tr('Opened from the exact Copilot data snapshot.', '从当前 Copilot 数据快照打开。'),
        cohort_hint: payload && payload.summary && payload.summary.cohort_size != null ? `${fmt(payload.summary.cohort_size, 0)} ICU stays` : '',
        module_hint: (payload && payload.shared_modules || []).slice(0, 6).join(', '),
      },
    });
  }
  function mount(host, payload, view) {
    if (!host) return;
    const state = {};
    const repaint = () => paint();
    state.repaint = repaint;
    function paint() {
      host.innerHTML = render(payload || {}, view, state);
      const route = routeOf(view);
      host.querySelectorAll('[data-gpi-embedded-tab]').forEach(button => button.addEventListener('click', () => {
        if (route === 'patient') state.patientMode = button.dataset.gpiEmbeddedTab;
        else state.cohortPanel = button.dataset.gpiEmbeddedTab;
        paint();
      }));
      const open = host.querySelector('[data-gpi-open-native]');
      if (open) open.addEventListener('click', () => handoff(route, payload || {}));
      const body = host.querySelector('[data-gpi-native-body]');
      if (!body) return;
      body.querySelectorAll('[data-study-handoff],[data-nav]').forEach(control => {
        control.hidden = true;
        control.setAttribute('aria-disabled', 'true');
        control.setAttribute('tabindex', '-1');
      });
      if (route === 'patient' && window.EU_PATIENT_CHARTS && typeof window.EU_PATIENT_CHARTS.mount === 'function') window.EU_PATIENT_CHARTS.mount(body);
      if (route === 'cohort' && window.EU_VIZ_CONTEXT && typeof window.EU_VIZ_CONTEXT.mountCohortCharts === 'function') {
        window.EU_VIZ_CONTEXT.mountCohortCharts(body);
        body.querySelectorAll('[data-cohort-comp],[data-cohgo],[data-cohort-feature-toggle],[data-cohort-feature-default],[data-cohort-feature-clear],[data-cohort-surv-group],[data-cohort-sofa-matrix-mode],[data-cohort-sofa-granularity]').forEach(control => {
          control.setAttribute('aria-disabled', 'true'); control.setAttribute('tabindex', '-1');
        });
      }
      if (route === 'crossdb' && window.EU_CROSSDB_RESULTS) {
        const config = window.EU_VIZ_CONTEXT && window.EU_VIZ_CONTEXT.crossdbResultsConfig ? window.EU_VIZ_CONTEXT.crossdbResultsConfig(repaint) : { repaint };
        if (typeof window.EU_CROSSDB_RESULTS.mount === 'function') window.EU_CROSSDB_RESULTS.mount(body);
        if (typeof window.EU_CROSSDB_RESULTS.bind === 'function') window.EU_CROSSDB_RESULTS.bind(body, payload || {}, config);
      }
    }
    paint();
  }

  window.EU_VIZ_EMBEDDED_WORKBENCH = { mount, render, routeOf };
})();
