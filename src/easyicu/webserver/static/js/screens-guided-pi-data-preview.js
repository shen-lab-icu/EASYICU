/* Conversational Data Workbench renderer.
   Payloads are loaded from project-scoped immutable snapshots. This owner only
   presents bounded cohort aggregates and browser-only pseudonymous timelines. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function num(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; }
  function fmt(value, digits) {
    const parsed = num(value);
    return parsed == null ? '—' : parsed.toLocaleString(undefined, { maximumFractionDigits: digits == null ? 1 : digits });
  }
  function metric(label, value, note) {
    return `<div class="gpi-data-metric"><span>${esc(label)}</span><strong>${esc(value)}</strong>${note ? `<small>${esc(note)}</small>` : ''}</div>`;
  }
  function section(title, note, body) {
    return `<section class="gpi-data-section"><header><div><h4>${esc(title)}</h4>${note ? `<p>${esc(note)}</p>` : ''}</div></header>${body}</section>`;
  }
  function empty(message) { return `<div class="gpi-data-empty">${esc(message)}</div>`; }

  function sparkline(values, color) {
    const points = (Array.isArray(values) ? values : []).map(num).filter(value => value != null);
    if (!points.length) return '';
    const width = 300; const height = 88; const pad = 8;
    const low = Math.min(...points); const high = Math.max(...points); const span = high - low || 1;
    const coords = points.map((value, index) => {
      const x = points.length === 1 ? width / 2 : pad + index / (points.length - 1) * (width - pad * 2);
      const y = height - pad - (value - low) / span * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    return `<svg class="gpi-data-spark" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(tr('Bounded time series', '有界时间序列'))}"><line x1="8" y1="80" x2="292" y2="80"></line><polyline points="${coords}" style="stroke:${esc(color || '#5965d8')}"></polyline></svg>`;
  }

  function funnel(flow) {
    const steps = Array.isArray(flow && flow.steps) ? flow.steps : [];
    if (!steps.length) return empty(tr('No stepwise cohort report is available for this export.', '该导出没有逐步队列筛选回执。'));
    const initial = Math.max(1, num(flow.initial_count) || num(steps[0].count) || 1);
    return `<div class="gpi-data-funnel">${steps.map((step, index) => {
      const count = Math.max(0, num(step.count) || 0);
      const width = Math.max(9, count / initial * 100);
      const label = step.label_i18n && step.label_i18n[window.EU_LANG] || step.label || step.id;
      const note = step.note_i18n && step.note_i18n[window.EU_LANG] || step.note || '';
      return `<div class="gpi-data-funnel-row"><div class="gpi-data-funnel-label"><span>${index + 1}</span><div><strong>${esc(label)}</strong>${note ? `<small>${esc(note)}</small>` : ''}</div><b>${fmt(count, 0)}</b></div><div class="gpi-data-funnel-track"><i style="width:${width.toFixed(2)}%"></i></div>${step.excluded ? `<em>−${fmt(step.excluded, 0)} (${fmt(step.excluded_pct_of_previous, 1)}%)</em>` : ''}</div>`;
    }).join('')}</div>`;
  }

  function histogram(distribution) {
    const bins = Array.isArray(distribution && distribution.bins) ? distribution.bins : [];
    const categories = Array.isArray(distribution && distribution.categories) ? distribution.categories : [];
    const rows = bins.length ? bins.map(bin => ({
      label: `${fmt(bin.low, 2)}–${fmt(bin.high, 2)}`, count: num(bin.count) || 0,
    })) : categories.map(row => ({ label: row.label || '', count: num(row.count) || 0 }));
    const max = Math.max(1, ...rows.map(row => row.count));
    return `<article class="gpi-data-distribution"><header><div><strong>${esc(distribution.label || distribution.column || '')}</strong><small>${esc([distribution.module, distribution.aggregation].filter(Boolean).join(' · '))}</small></div><span>${fmt(distribution.observed_pct, 1)}% ${esc(tr('observed', '有观测'))}</span></header><div class="gpi-data-bars">${rows.map(row => `<div><span>${esc(row.label)}</span><i><u style="width:${(row.count / max * 100).toFixed(2)}%"></u></i><b>${fmt(row.count, 0)}</b></div>`).join('')}</div>${distribution.summary ? `<footer>${esc(tr('Median', '中位数'))} <strong>${fmt(distribution.summary.median, 2)}</strong> · ${esc(tr('Range', '范围'))} ${fmt(distribution.summary.min, 2)}–${fmt(distribution.summary.max, 2)}</footer>` : ''}</article>`;
  }

  function cohortView(payload) {
    const summary = payload && payload.summary || {};
    const distributions = Array.isArray(payload && payload.selected_feature_distributions) ? payload.selected_feature_distributions : [];
    const quality = payload && payload.quality || {};
    const cards = `<div class="gpi-data-metrics">${metric(tr('ICU stays', 'ICU 住院'), fmt(summary.cohort_size, 0), tr('analysis denominator', '分析分母'))}${metric(tr('Modules', '模块'), fmt(summary.modules, 0), `${fmt(quality.median_coverage_pct, 1)}% ${tr('median coverage', '中位覆盖')}`)}${metric(tr('Mortality', '死亡率'), summary.mortality_pct == null ? '—' : `${fmt(summary.mortality_pct, 1)}%`, tr('descriptive only', '仅描述性'))}${metric(tr('Selected features', '已选特征'), fmt(distributions.length, 0), tr('bounded aggregate view', '有界聚合视图'))}</div>`;
    return `${cards}${section(tr('Cohort filter funnel', '队列筛选漏斗'), tr('Counts come from the export owner receipt; no row-level data are shown.', '计数来自导出 owner 回执，不展示行级数据。'), funnel(payload && payload.eligibility_flow))}${distributions.length ? section(tr('Feature distributions', '特征分布'), tr('Stay-level descriptive aggregation; not an inferential result.', '住院级描述性聚合；不是推断性结果。'), `<div class="gpi-data-distributions">${distributions.map(histogram).join('')}</div>`) : ''}`;
  }

  function patientTimeline(payload) {
    const selected = payload && payload.selected || {};
    const lanes = (Array.isArray(payload && payload.time_lanes) ? payload.time_lanes : []).filter(lane => lane && lane.status === 'ready');
    const signals = lanes.flatMap(lane => (lane.signals || []).map(signal => ({ ...signal, lane_label: lane.label })));
    const cards = `<div class="gpi-data-metrics">${metric(tr('Pseudonymous entity', '伪匿名实体'), selected.label || tr('Selected entity', '所选实体'), tr('ordinal + one-way token', '序号 + 单向令牌'))}${metric(tr('Time lanes', '时间通道'), fmt(lanes.length, 0), tr('available lanes', '可用通道'))}${metric(tr('Signals', '信号'), fmt(signals.length, 0), tr('at most 12 points each', '每个最多 12 个点'))}${metric(tr('ICU outcome', 'ICU 结局'), selected.outcome || '—', tr('local browser view', '本地浏览器视图'))}</div>`;
    const plots = signals.length ? `<div class="gpi-data-timelines">${signals.slice(0, 12).map((signal, index) => `<article><header><div><strong>${esc(signal.name || signal.feature || '')}</strong><small>${esc([signal.lane_label, signal.unit].filter(Boolean).join(' · '))}</small></div><span>${fmt(signal.current, 2)}</span></header>${sparkline(signal.values, ['#5965d8', '#10a37f', '#e07a3f', '#c458a0'][index % 4])}<footer><span>${esc(signal.time_axis && (window.EU_LANG === 'zh' ? signal.time_axis.label_zh : signal.time_axis.label_en) || tr('Recorded time', '记录时间'))}</span><span>${esc(tr('min', '最低'))} ${fmt(signal.min, 2)} · ${esc(tr('max', '最高'))} ${fmt(signal.max, 2)}</span></footer></article>`).join('')}</div>` : empty(tr('No time-indexed signals are available for this entity.', '该实体没有可用的时间索引信号。'));
    return `${cards}${section(tr('Patient time series', '患者时间序列'), tr('Browser-only pseudonymous review. Raw identifiers, notes and full rows are withheld.', '仅浏览器伪匿名审阅；不返回原始标识符、病历文本和完整数据行。'), plots)}`;
  }

  function crossdbView(payload) {
    const sources = Array.isArray(payload && payload.sources) ? payload.sources : [];
    const rows = Array.isArray(payload && payload.rows) ? payload.rows : [];
    const shared = Array.isArray(payload && payload.shared_modules) ? payload.shared_modules : [];
    const gate = payload && payload.compatibility_gate || {};
    const cards = `<div class="gpi-data-metrics">${metric(tr('Databases', '数据库'), fmt(sources.length, 0), sources.map(row => row.database || row.label).filter(Boolean).join(' · '))}${metric(tr('Shared modules', '共有模块'), fmt(shared.length, 0), tr('concept-aligned', '概念层对齐'))}${metric(tr('Compatibility', '兼容性'), gate.status || '—', tr('owner gate', 'owner 闸门'))}${metric(tr('Claim level', '结论级别'), gate.claim_level || tr('descriptive only', '仅描述性'), tr('no matched cohort', '未做匹配队列'))}</div>`;
    const table = rows.length ? `<div class="gpi-data-table" role="table" style="--gpi-data-source-count:${Math.max(1, sources.length)}"><div class="gpi-data-table-row head" role="row"><span>${esc(tr('Metric', '指标'))}</span>${sources.map(source => `<span>${esc(source.label || source.database || '')}</span>`).join('')}<span>Δ</span></div>${rows.map(row => `<div class="gpi-data-table-row" role="row"><strong>${esc(row.label || row.key || '')}</strong>${(row.values || []).map(value => `<span>${fmt(value, 2)}</span>`).join('')}<b>${fmt(row.delta, 2)}</b></div>`).join('')}</div>` : empty(tr('No comparable aggregate metrics were issued.', '没有可比较的聚合指标。'));
    const modules = shared.length ? `<div class="gpi-data-chips">${shared.map(value => `<span>${esc(value)}</span>`).join('')}</div>` : empty(tr('No shared modules.', '没有共有模块。'));
    return `${cards}${section(tr('Cross-database metrics', '跨数据库指标'), tr('Descriptive ranges from registered exports; not matched or inferential.', '来自登记导出的描述性范围；不是匹配或推断分析。'), table)}${section(tr('Shared concept modules', '共有概念模块'), tr('These modules are physically present across the selected exports.', '这些模块在所选导出中均实际存在。'), modules)}`;
  }

  function render(payload, view) {
    const source = payload && payload.source || {};
    const body = view === 'patient_timeline' ? patientTimeline(payload)
      : view === 'crossdb_comparison' ? crossdbView(payload)
      : cohortView(payload);
    return `<div class="gpi-data-preview" data-gpi-data-preview><header class="gpi-data-intro"><div><span>${esc(tr('Conversational Data Workbench', '对话式数据工作台'))}</span><h3>${esc(view === 'patient_timeline' ? tr('Patient timeline', '患者时间序列') : view === 'crossdb_comparison' ? tr('Cross-database comparison', '跨数据库比较') : view === 'feature_distribution' ? tr('Feature distribution', '特征分布') : tr('Cohort review', '队列审阅'))}</h3><p>${esc(source.label || source.database || tr('Registered EasyICU export', '已登记 EasyICU 导出'))}</p></div><em>${esc(tr('Descriptive review', '描述性审阅'))}</em></header>${body}<footer class="gpi-data-governance"><strong>${esc(tr('Governed local view', '受治理的本地视图'))}</strong><span>${esc(tr('Continue in this conversation to change the cohort, feature, patient ordinal, database selection, or export request.', '如需更改队列、特征、患者序号、数据库选择或导出要求，直接在当前对话中继续提出。'))}</span></footer></div>`;
  }

  function mount(host, payload, view) {
    if (host) host.innerHTML = render(payload || {}, String(view || 'cohort_summary'));
  }

  window.EU_GUIDED_PI_DATA_PREVIEW = { mount, render };
})();
