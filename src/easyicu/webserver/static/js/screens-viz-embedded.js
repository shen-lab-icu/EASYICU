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
    if (view === 'icd_cohort_preview') return 'extraction';
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
    if (route === 'extraction') return {
      eyebrow: tr('Native Data Extraction', '原生数据提取'),
      title: tr('ICD cohort preview', 'ICD 队列预览'),
      open: tr('Open full Data Extraction', '打开完整数据提取'),
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
    const rawLanes = Array.isArray(review.lanes) ? review.lanes : (payload.time_lanes || []);
    const detailByFeature = new Map((payload.loaded_feature_details || []).map(detail => [
      String(detail && detail.feature && detail.feature.feature || ''), detail,
    ]).filter(row => row[0]));
    const loadedByModule = new Map();
    detailByFeature.forEach(detail => {
      if (!detail || !detail.signal) return;
      const module = String(detail.feature && detail.feature.module || '');
      if (!module) return;
      const signals = loadedByModule.get(module) || [];
      signals.push({ ...detail.signal, module, lazy_loaded: true });
      loadedByModule.set(module, signals);
    });
    const lanesByModule = new Map((rawLanes || []).map(lane => [String(lane && lane.lane || ''), lane]));
    const augmentedLanes = Array.from(new Set([...lanesByModule.keys(), ...loadedByModule.keys()])).filter(Boolean).map(module => {
      const lane = lanesByModule.get(module) || { lane: module, label: module, signals: [], signal_count: 0, status: 'unavailable' };
      const loaded = loadedByModule.get(module) || [];
      const loadedKeys = new Set(loaded.map(signal => String(signal.feature || signal.key || '')));
      const signals = loaded.concat((lane.signals || []).filter(signal => !loadedKeys.has(String(signal && (signal.feature || signal.key) || ''))));
      return { ...lane, signals, signal_count: signals.length, status: signals.length ? 'ready' : lane.status };
    });
    const featureOwner = window.EU_PATIENT_FEATURES;
    const lanes = featureOwner && typeof featureOwner.catalogLanes === 'function'
      ? featureOwner.catalogLanes(augmentedLanes, payload.feature_coverage, feature => {
        const detail = detailByFeature.get(String(feature || ''));
        return detail ? { payload: detail, loaded: true } : {};
      })
      : augmentedLanes;
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
  function icdCohortBody(payload) {
    const summary = payload && payload.summary || {};
    const report = payload && payload.cohort_report || {};
    const icd = report.icd || {};
    const sourceTotal = Number(summary.source_total != null ? summary.source_total : report.source_total);
    const beforeIcd = Number(summary.selected_before_icd != null ? summary.selected_before_icd : report.selected_before_icd);
    const selected = Number(summary.cohort_size != null ? summary.cohort_size : report.selected);
    const steps = [
      { label: tr('Source ICU stays', '数据源 ICU 住院'), count: sourceTotal },
      { label: tr('After demographic/stay filters', '人口学与住院筛选后'), count: beforeIcd },
      { label: tr('Final ICD cohort', '最终 ICD 队列'), count: selected },
    ].filter(item => Number.isFinite(item.count));
    const max = Math.max(1, ...steps.map(item => item.count));
    const tokens = (label, values, tone) => `<div class="gpi-icd-token-row"><b>${esc(label)}</b><div>${(Array.isArray(values) && values.length ? values : [tr('None', '无')]).map(value => `<span class="pill ${tone || ''}">${esc(value)}</span>`).join('')}</div></div>`;
    return `<div data-gpi-native-body data-gpi-icd-flow>
      <div class="gpi-icd-kpis">
        <section><span>${esc(tr('Final cohort', '最终队列'))}</span><strong>${fmt(selected, 0)}</strong><small>${esc(tr('ICU stays', 'ICU 住院'))}</small></section>
        <section><span>${esc(tr('Include matches', '纳入匹配'))}</span><strong>${fmt(icd.include_matches, 0)}</strong><small>${esc(tr('before exclusions', '排除前'))}</small></section>
        <section><span>${esc(tr('Exclude matches', '排除匹配'))}</span><strong>${fmt(icd.exclude_matches, 0)}</strong><small>${esc(tr('source-wide matches', '全数据源匹配'))}</small></section>
      </div>
      <section class="card pad gpi-icd-flow"><div class="panel-head"><div><div class="eyebrow">${esc(tr('Owner-computed filter funnel', 'Owner 计算的筛选漏斗'))}</div><div class="panel-title">${esc(tr('How the ICD cohort was resolved', 'ICD 队列如何得到'))}</div></div><span class="pill ok">${esc(tr('Read-only', '只读'))}</span></div>
        <div class="gpi-icd-flow-bars">${steps.map((item, index) => `<div><span>${esc(item.label)}</span><i><u style="width:${(item.count / max * 100).toFixed(2)}%"></u></i><b>${fmt(item.count, 0)}</b>${index ? `<em>${fmt(steps[index - 1].count > 0 ? item.count / steps[index - 1].count * 100 : 0, 1)}%</em>` : '<em>100%</em>'}</div>`).join('')}</div>
      </section>
      <section class="card pad gpi-icd-codes"><div class="panel-title">${esc(tr('ICD prefix rules', 'ICD 前缀规则'))}</div>${tokens(tr('Include', '纳入'), icd.include_tokens, 'ok')}${tokens(tr('Exclude', '排除'), icd.exclude_tokens, 'warn')}</section>
      <div class="viz-cap"><b>${esc(tr('Scope', '范围'))}</b><span>${esc(tr('Counts are ICU stays from the Data Extraction cohort owner. No extraction job was started, and no identifiers or raw rows entered the conversation.', '计数单位为 Data Extraction 队列 owner 计算的 ICU 住院。本次未启动提取任务，患者标识和原始数据行也未进入对话。'))}</span></div>
    </div>`;
  }
  function render(payload, view, state) {
    const cleanView = String(view || 'cohort_summary');
    const local = state || {};
    const route = routeOf(cleanView);
    const body = route === 'patient' ? patientBody(payload || {}, local)
      : route === 'crossdb' ? crossdbBody(payload || {}, local.repaint)
      : route === 'extraction' ? icdCohortBody(payload || {})
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
