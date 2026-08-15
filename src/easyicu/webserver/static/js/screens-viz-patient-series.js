(function () {
  const COLORS = ['var(--accent)', '#2563eb', '#0f766e', '#b45309', '#7c3aed', '#be123c'];
  // Only concepts with an absolute, payload-supplied reference are eligible.
  // Relative definitions such as ΔSOFA are intentionally excluded.
  const ABSOLUTE_REFERENCE_FEATURES = new Set([
    'hr', 'map', 'sbp', 'dbp', 'spo2', 'o2sat', 'sao2', 'resp', 'temp',
    'lact', 'crea', 'ph', 'glu', 'k', 'na', 'plt', 'hgb', 'inr_pt', 'pafi', 'bili',
  ]);
  const MAX_SIGNALS_PER_MODULE = 8;

  function signalKey(sig) {
    return String((sig && (sig.feature || sig.key || sig.name)) || '').toLowerCase();
  }

  function numericSamples(sig) {
    const rawValues = (sig && Array.isArray(sig.values)) ? sig.values : [];
    const rawTimes = (sig && Array.isArray(sig.times)) ? sig.times : [];
    const alignedTimes = rawTimes.length === rawValues.length;
    const values = [];
    const times = [];
    rawValues.forEach((rawValue, index) => {
      if (rawValue == null || rawValue === '') return;
      const value = Number(rawValue);
      if (!Number.isFinite(value)) return;
      values.push(value);
      if (alignedTimes) times.push(rawTimes[index]);
    });
    return { values, times: alignedTimes ? times : rawTimes.slice(0, values.length) };
  }

  function numericValues(sig) {
    return numericSamples(sig).values;
  }

  function fallbackEsc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    })[ch]);
  }

  function hEsc(helpers, value) {
    return helpers && helpers.esc ? helpers.esc(value) : fallbackEsc(value);
  }

  function hT(helpers, en, zh) {
    return helpers && helpers.t ? helpers.t(en, zh) : en;
  }

  function hFmtInt(helpers, value) {
    return helpers && helpers.fmtInt ? helpers.fmtInt(value) : String(value == null ? '' : value);
  }

  function hFmtNum(helpers, value, digits = 1) {
    return helpers && helpers.fmtNum ? helpers.fmtNum(value, digits) : Number(value).toFixed(digits);
  }

  function signalLabel(sig, helpers) {
    if (helpers && helpers.signalLabel) return helpers.signalLabel(sig);
    return (sig && (sig.label || sig.name || sig.feature || sig.key)) || '';
  }

  function formatTimeLabel(raw, helpers) {
    const numeric = Number(raw);
    if (Number.isFinite(numeric)) return `${hFmtNum(helpers, numeric, numeric % 1 ? 1 : 0)}h`;
    return raw == null || raw === '' ? '' : String(raw);
  }

  function signalTimeLabels(signals, helpers, fallbackEnd = null) {
    const withTimes = (signals || []).find(sig => Array.isArray(sig && sig.times) && sig.times.length);
    const samples = numericSamples(withTimes);
    const times = samples.times;
    const first = times.length ? formatTimeLabel(times[0], helpers) : '';
    const last = times.length ? formatTimeLabel(times[times.length - 1], helpers) : '';
    if (first || last) return [first || '0h', last || hT(helpers, 'last point', '末点')];
    if (fallbackEnd) return ['0h', `${fallbackEnd}h`];
    return null;
  }

  function moduleLabel(lane, helpers) {
    const raw = (lane && (lane.label || lane.lane)) || hT(helpers, 'Module', '模块');
    return helpers && helpers.seriesLabel ? helpers.seriesLabel(raw) : raw;
  }

  function median(values) {
    if (!values.length) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }

  function thresholdsFor(sig, helpers) {
    const key = signalKey(sig);
    if (!ABSOLUTE_REFERENCE_FEATURES.has(key)) return [];
    return ((sig && sig.thresholds) || [])
      .map((row, index) => ({
        value: Number(row && row.value),
        label: (row && row.label) || hT(helpers, 'Clinical reference', '临床参考线'),
        color: (row && row.color) || (index % 2 ? '#f97316' : '#ef4444'),
        dash: (row && row.dash) || '4 4',
      }))
      .filter(row => Number.isFinite(row.value))
      .slice(0, 4);
  }

  function fallbackThresholds(sig, helpers) {
    const values = numericValues(sig);
    const midpoint = median(values);
    const rows = thresholdsFor(sig, helpers);
    if (Number.isFinite(midpoint)) {
      rows.unshift({
        value: midpoint,
        label: hT(helpers, 'Median', '中位数'),
        color: '#94a3b8',
        dash: '4 4',
      });
    }
    return rows;
  }

  function signalCell(sig, index, helpers, xLabels) {
    const samples = numericSamples(sig);
    const values = samples.values;
    if (values.length < 2 || !helpers || !helpers.axisSpark) return '';
    const label = signalLabel(sig, helpers);
    const color = COLORS[index % COLORS.length];
    const unit = sig && sig.unit ? sig.unit : '';
    const fallbackChart = helpers.axisSpark(values, 360, 132, color, {
      unit,
      label,
      thresholds: fallbackThresholds(sig, helpers),
      xLabels,
    });
    const chartOwner = window.EU_PATIENT_CHARTS;
    const chart = chartOwner && chartOwner.signalSlot
      ? chartOwner.signalSlot({
        color,
        feature: signalKey(sig),
        label,
        latestLabel: hT(helpers, 'Latest', '最新'),
        thresholds: thresholdsFor(sig, helpers),
        timeAxis: sig && sig.time_axis,
        times: samples.times,
        unit,
        values,
      }, fallbackChart)
      : fallbackChart;
    return `
      <div class="pt-vsm-cell" data-patient-series-feature="${hEsc(helpers, signalKey(sig))}">
        <div class="pt-vsm-title">${hEsc(helpers, label)}${unit ? ` <span class="mono">${hEsc(helpers, unit)}</span>` : ''}</div>
        ${chart}
      </div>`;
  }

  function featureAvailability(feature, helpers) {
    if (feature && feature.trajectory) {
      return {
        cssClass: 'available',
        label: hT(helpers, 'loaded trajectory', '已加载轨迹'),
      };
    }
    const status = String(feature && feature.status || '');
    const rows = {
      available_unloaded: ['available-unloaded', hT(helpers, 'observed · load chart', '已有观测 · 加载图表')],
      observed_categorical: ['observed', hT(helpers, 'observed category', '分类观测')],
      observed_static: ['observed', hT(helpers, 'observed value', '单点观测')],
      observed_numeric_static: ['observed', hT(helpers, 'observed value', '单点观测')],
      selected_entity_unavailable: ['unavailable', hT(helpers, 'no value for this entity', '该实体无观测')],
      all_null: ['unavailable', hT(helpers, 'no observations in sample', '样本中无观测')],
      structurally_unavailable: ['unsupported', hT(helpers, 'unsupported for source', '该数据源不支持')],
      not_materialized: ['metadata-only', hT(helpers, 'not materialized', '未物化')],
      materialized_unknown: ['unknown', hT(helpers, 'materialized · verify on load', '已物化 · 加载核验')],
    };
    const match = rows[status];
    if (match) return { cssClass: match[0], label: match[1] };
    if (feature && feature.observed) {
      return {
        cssClass: 'observed',
        label: hT(helpers, 'observed', '已有观测'),
      };
    }
    return {
      cssClass: 'metadata-only',
      label: hT(helpers, 'catalog metadata', '目录元数据'),
    };
  }

  function featureObservationLabel(feature, helpers) {
    const observation = feature && feature.observation;
    if (!observation || observation.current == null) return '';
    const value = String(observation.current);
    const unit = feature && feature.unit ? ` ${feature.unit}` : '';
    return hT(helpers, `value: ${value}${unit}`, `值：${value}${unit}`);
  }

  function renderModulePanels(lanes, helpers = {}) {
    const usable = (lanes || [])
      .map(lane => {
        const signals = ((lane && lane.signals) || []).filter(sig => numericValues(sig).length >= 2);
        const declaredFeatures = Array.isArray(lane && lane.features) ? lane.features : [];
        const features = declaredFeatures.length
          ? declaredFeatures
          : signals.map(sig => ({
            feature: signalKey(sig),
            name: signalLabel(sig, helpers),
            unit: (sig && sig.unit) || '',
            status: 'trajectory',
            trajectory: true,
          }));
        return { lane, signals, features };
      })
      .filter(item => item.signals.length || item.features.length);
    if (!usable.length) return '';

    const cards = usable.map((item, moduleIndex) => {
      const visibleSignals = item.signals.slice(0, MAX_SIGNALS_PER_MODULE);
      const xLabels = signalTimeLabels(visibleSignals, helpers, helpers.demoHours ? helpers.demoHours() : null);
      const hidden = Math.max(0, item.signals.length - visibleSignals.length);
      const declared = Number(item.lane && item.lane.signal_count);
      const totalFeatures = Number.isFinite(declared) && declared > 0 ? declared : item.features.length;
      const observedDeclared = Number(item.lane && item.lane.export_observed_count);
      const observedFeatures = Number.isFinite(observedDeclared)
        ? observedDeclared
        : item.features.filter(feature => feature && feature.observed).length;
      const loadedTrajectories = item.features.filter(feature => feature && feature.trajectory).length || item.signals.length;
      const loadableFeatures = item.features.filter(feature => (
        feature && feature.loadable && !feature.trajectory && !feature.lazy_loaded
      ));
      const loadedNonTrajectory = item.features.filter(feature => (
        feature && feature.lazy_loaded && !feature.trajectory
      )).length;
      const moduleLoading = loadableFeatures.some(feature => feature && feature.loading);
      const moduleId = (item.lane && item.lane.lane) || String(moduleIndex);
      return `
        <section class="pt-module-card" data-patient-series-module="${hEsc(helpers, moduleId)}">
          <div class="pt-module-head">
            <div>
              <div class="pt-module-title">${hEsc(helpers, moduleLabel(item.lane, helpers))}</div>
              <div class="pt-module-meta mono">${hFmtInt(helpers, totalFeatures)} ${hT(helpers, 'catalog features', '个目录特征')} · ${hFmtInt(helpers, observedFeatures)} ${hT(helpers, 'observed in export', '个在导出中有观测')} · ${hFmtInt(helpers, loadedTrajectories)} ${hT(helpers, 'loaded trajectories', '个已加载轨迹')}</div>
            </div>
            <div class="pt-module-actions">
              ${loadableFeatures.length ? `
                <button class="btn sm" type="button" data-patient-module-load="${hEsc(helpers, moduleId)}"${moduleLoading ? ' disabled aria-busy="true"' : ''}>
                  ${hEsc(helpers, moduleLoading
                    ? hT(helpers, 'Loading module…', '正在加载本模块…')
                    : hT(helpers, `Load module data (${loadableFeatures.length})`, `加载本模块数据（${loadableFeatures.length}）`))}
                </button>` : ''}
              <span class="chip">${hFmtInt(helpers, item.features.length)} / ${hFmtInt(helpers, totalFeatures)}</span>
            </div>
          </div>
          ${visibleSignals.length ? `
            <div class="pt-module-grid">
              ${visibleSignals.map((sig, i) => signalCell(sig, i, helpers, xLabels)).join('')}
            </div>
            ${hidden ? `<div class="pt-feature-overflow-note">${hT(helpers, 'Additional trajectories remain listed in the module inventory.', '其余轨迹仍列在模块特征清单中。')} +${hFmtInt(helpers, hidden)}</div>` : ''}` : `
            <div class="pt-module-no-series">
              ${loadableFeatures.length
                ? hT(
                  helpers,
                  `${loadableFeatures.length} source-backed features can be checked for this entity. Load the module to separate trajectories, single values, categories, and true missingness.`,
                  `有 ${loadableFeatures.length} 个来源可追溯的特征可核查。加载本模块后会区分轨迹、单点值、分类值和该患者确实缺失。`,
                )
                : loadedNonTrajectory
                  ? hT(
                    helpers,
                    `Module data loaded. No multi-point numeric trajectory exists for this entity; expand the feature list to review ${loadedNonTrajectory} values, categories, or missingness findings.`,
                    `本模块已加载。该患者没有可绘制的多时点数值轨迹；展开特征清单可查看 ${loadedNonTrajectory} 个单点值、分类值或缺失结论。`,
                  )
                  : hT(helpers, 'No bounded numeric trajectory is available for this module in the active payload; catalog metadata remains reviewable.', '当前载荷中该模块没有可用的有界数值轨迹；目录元数据仍可审阅。')}
            </div>`}
          <details class="pt-feature-inventory">
            <summary>
              <span>${hT(helpers, 'Feature list and loading status', '特征清单与加载状态')}</span>
              <span class="mono">${hFmtInt(helpers, item.features.length)} ${hT(helpers, 'features', '个特征')}</span>
            </summary>
            <div class="pt-feature-inventory-grid">
              ${item.features.map(feature => {
                const key = (feature && (feature.feature || feature.key)) || '';
                const label = (feature && (feature.name || feature.label)) || key;
                const unit = feature && feature.unit ? feature.unit : '';
                const hasTrajectory = Boolean(feature && feature.trajectory);
                const availability = featureAvailability(feature, helpers);
                const canLoad = Boolean(feature && feature.loadable && !hasTrajectory && !feature.lazy_loaded);
                const loading = Boolean(feature && feature.loading);
                const error = feature && feature.load_error ? String(feature.load_error) : '';
                const tag = canLoad ? 'button' : 'div';
                const attributes = canLoad
                  ? ` type="button" data-patient-feature-load="${hEsc(helpers, key)}"${loading ? ' disabled aria-busy="true"' : ''}`
                  : '';
                const stateClass = error ? ' load-error' : '';
                const availabilityLabel = loading
                  ? hT(helpers, 'loading…', '加载中…')
                  : (error
                    ? hT(helpers, 'retry load', '重试加载')
                    : (featureObservationLabel(feature, helpers) || availability.label));
                return `<${tag} class="pt-feature-inventory-item ${availability.cssClass}${canLoad ? ' loadable' : ''}${stateClass}"${attributes}${error ? ` title="${hEsc(helpers, error)}"` : ''}>
                  <span class="pt-feature-status-dot" aria-hidden="true"></span>
                  <span class="pt-feature-inventory-copy">
                    <b>${hEsc(helpers, label)}</b>
                    <span class="mono">${hEsc(helpers, key)}${unit ? ` · ${hEsc(helpers, unit)}` : ''}</span>
                  </span>
                  <em>${hEsc(helpers, availabilityLabel)}</em>
                </${tag}>`;
              }).join('')}
            </div>
          </details>
        </section>`;
    }).join('');
    const totalCatalogFeatures = usable.reduce((sum, item) => sum + item.features.length, 0);
    const totalTrajectories = usable.reduce((sum, item) => sum + item.signals.length, 0);
    const totalObservedFeatures = usable.reduce((sum, item) => {
      const declared = Number(item.lane && item.lane.export_observed_count);
      return sum + (Number.isFinite(declared)
        ? declared
        : item.features.filter(feature => feature && feature.observed).length);
    }, 0);

    return `
      <div class="sec-stack mt-16">
        <div class="lbl">${hT(helpers, 'Time series and feature catalog by module', '按模块分组的时间序列与特征目录')}</div>
        <div class="pt-catalog-scope mono">${hFmtInt(helpers, totalCatalogFeatures)} ${hT(helpers, 'features across', '个特征，分布于')} ${hFmtInt(helpers, usable.length)} ${hT(helpers, 'modules', '个模块')} · ${hFmtInt(helpers, totalObservedFeatures)} ${hT(helpers, 'observed in export', '个在导出中有观测')} · ${hFmtInt(helpers, totalTrajectories)} ${hT(helpers, 'bounded trajectories loaded', '条已加载的有界轨迹')}</div>
      </div>
      <div class="pt-vsm-legend">
        <span><i style="background:var(--accent);"></i>${hT(helpers, 'Patient trajectory', '患者轨迹')}</span>
        <span><i class="dash" style="color:#94a3b8;"></i>${hT(helpers, 'Median', '中位数')}</span>
        <span><i class="dash" style="color:#ef4444;"></i>${hT(helpers, 'Clinical reference guide', '临床参考线')}</span>
        ${window.EU_PATIENT_CHARTS && window.EU_PATIENT_CHARTS.available()
          ? `<span class="pt-echart-hint">${hT(helpers, 'Hover for exact time and value', '悬停查看精确时间和值')}</span>`
          : ''}
      </div>
      <div class="pt-feature-toolbar">
        <span>${hT(helpers, 'All catalog features are grouped below. Expand the lists to inspect every feature and its honest loading state.', '全部目录特征已按模块分组。展开清单即可查看每个特征及其真实加载状态。')}</span>
        <div>
          <button class="btn sm" type="button" data-patient-inventory-toggle="open">${hT(helpers, 'Expand all feature lists', '展开全部特征清单')}</button>
          <button class="btn sm" type="button" data-patient-inventory-toggle="close">${hT(helpers, 'Collapse all', '全部收起')}</button>
        </div>
      </div>
      <div class="pt-module-stack">${cards}</div>
      <div class="pt-vsm-axis">${hT(helpers, 'Source-recorded time spacing · bounded payload', '按源记录时间间隔展示 · 有界载荷')}</div>`;
  }

  function flattenSignals(lanes) {
    const rows = [];
    (lanes || []).forEach((lane, laneIndex) => {
      ((lane && lane.signals) || []).forEach((sig, signalIndex) => {
        if (numericValues(sig).length >= 2) rows.push({ lane, sig, laneIndex, signalIndex });
      });
    });
    return rows;
  }

  function renderModeBar(mode, helpers) {
    const modes = [
      ['lanes', hT(helpers, 'Module overview', '模块总览'), hT(helpers, 'All modules, features, and loading states', '全部模块、特征与加载状态')],
      ['single', hT(helpers, 'Trajectory gallery', '轨迹画廊'), hT(helpers, 'Loaded charts for the current entity', '当前患者已加载的逐项图表')],
      ['compare', hT(helpers, 'Cross-patient comparison', '跨患者对比'), hT(helpers, 'The same feature across entities', '同一特征跨患者比较')],
    ];
    return `
      <div class="pt-series-modebar" data-patient-series-modebar>
        ${modes.map(([id, title, sub]) => `
          <button class="pt-series-mode ${mode === id ? 'active' : ''}" data-patient-series-mode="${id}" type="button">
            <span class="pt-series-mode-title">${hEsc(helpers, title)}</span>
            <span>${hEsc(helpers, sub)}</span>
          </button>`).join('')}
      </div>`;
  }

  function renderClinicalLanes(lanes, helpers) {
    const totalCatalogFeatures = (lanes || []).reduce(
      (total, lane) => total + ((lane && lane.features) || []).length,
      0,
    );
    return `
      <div class="pt-series-panel" data-patient-series-panel="lanes">
        <div class="pt-series-panel-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Module overview', '模块总览'))}</div>
            <h2>${hEsc(helpers, hT(helpers, `Complete feature catalog grouped by clinical module (${hFmtInt(helpers, totalCatalogFeatures)} features)`, `${hFmtInt(helpers, totalCatalogFeatures)} 个特征，按临床模块组织`))}</h2>
            <p>${hEsc(helpers, hT(helpers, 'Every catalog concept remains discoverable. Features with bounded observations receive a chart; metadata-only concepts stay explicit instead of receiving fabricated demo values.', '所有目录概念均可查找；有有界观测的特征显示图表，仅有元数据的概念保持明确标识，不会为了演示而伪造数值。'))}</p>
          </div>
          <span class="pill ok">${hEsc(helpers, hT(helpers, 'single entity', '单实体'))}</span>
        </div>
        ${renderModulePanels(lanes, helpers)}
      </div>`;
  }

  function renderSinglePatient(lanes, selected, helpers) {
    const signals = flattenSignals(lanes).slice(0, 18);
    const xLabels = signalTimeLabels(signals.map(row => row.sig), helpers, helpers.demoHours ? helpers.demoHours() : null);
    if (!signals.length) return '';
    return `
      <div class="pt-series-panel" data-patient-series-panel="single">
        <div class="pt-series-panel-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Current-patient trajectory gallery', '当前患者轨迹画廊'))}</div>
            <h2>${hEsc(helpers, selected && (selected.label || selected.entity_ref) || hT(helpers, 'Selected entity', '已选实体'))}</h2>
            <p>${hEsc(helpers, hT(helpers, 'Feature cards are intentionally separated, so SOFA scores, labs and interventions do not collapse into one unreadable multi-line chart.', '每个特征单独成卡，避免把 SOFA、实验室和治疗支持堆进一张难读的多折线图。'))}</p>
          </div>
          <span class="pill">${hFmtInt(helpers, signals.length)} ${hEsc(helpers, hT(helpers, 'features shown', '个特征展示'))}</span>
        </div>
        <div class="pt-single-grid">
          ${signals.map((row, i) => `
            <article class="pt-single-card">
              <div class="pt-single-meta">
                <span>${hEsc(helpers, moduleLabel(row.lane, helpers))}</span>
                <span>${hEsc(helpers, signalKey(row.sig))}</span>
              </div>
              ${signalCell(row.sig, i, helpers, xLabels)}
            </article>`).join('')}
        </div>
      </div>`;
  }

  function traceValues(trace) {
    return numericSamples(trace).values;
  }

  function multiTraceChart(traces, comparison, helpers) {
    const series = (traces || []).map((trace, index) => {
      const samples = numericSamples(trace);
      return {
        trace: { ...trace, values: samples.values, times: samples.times },
        values: samples.values,
        color: COLORS[index % COLORS.length],
      };
    }).filter(row => row.values.length >= 2);
    if (series.length < 2) return '';
    const all = series.flatMap(row => row.values);
    const min = Math.min(...all);
    const max = Math.max(...all);
    const span = max > min ? max - min : 1;
    const width = 760;
    const height = 250;
    const left = 54;
    const right = 18;
    const top = 20;
    const bottom = 38;
    const plotW = width - left - right;
    const plotH = height - top - bottom;
    const maxPoints = Math.max(...series.map(row => row.values.length));
    const x = i => left + (maxPoints <= 1 ? 0 : (i / (maxPoints - 1)) * plotW);
    const y = value => top + (1 - ((value - min) / span)) * plotH;
    const yTicks = [max, min + span * 0.5, min];
    const unit = comparison && comparison.unit ? comparison.unit : '';
    const label = comparison && (comparison.label || comparison.feature) || hT(helpers, 'Selected feature', '已选特征');
    const timeLabels = signalTimeLabels(series.map(row => row.trace), helpers, null);
    const fallbackChart = `
      <div class="pt-multi-chart" role="img" aria-label="${hEsc(helpers, label)}">
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          ${yTicks.map(value => `
            <line x1="${left}" y1="${y(value).toFixed(1)}" x2="${width - right}" y2="${y(value).toFixed(1)}" class="grid" />
            <text x="10" y="${(y(value) + 4).toFixed(1)}" class="axis">${hEsc(helpers, `${(helpers.fmtNum ? helpers.fmtNum(value, 2) : value.toFixed(2))}${unit ? ` ${unit}` : ''}`)}</text>
          `).join('')}
          <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" class="axis-line" />
          <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" class="axis-line" />
          ${series.map(row => {
            const points = row.values.map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(' ');
            const last = row.values[row.values.length - 1];
            return `
              <polyline points="${points}" fill="none" stroke="${row.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
              <circle cx="${x(row.values.length - 1).toFixed(1)}" cy="${y(last).toFixed(1)}" r="4" fill="${row.color}" />`;
          }).join('')}
          <text x="${left}" y="${height - 10}" class="axis">${hEsc(helpers, hT(helpers, 'obs 1', '第1点'))}</text>
          <text x="${width - right - 42}" y="${height - 10}" class="axis">${hEsc(helpers, hT(helpers, 'obs N', '第N点'))}</text>
        </svg>
      </div>`;
    const chartOwner = window.EU_PATIENT_CHARTS;
    const chart = chartOwner && chartOwner.comparisonSlot
      ? chartOwner.comparisonSlot({
        feature: comparison && comparison.feature,
        label,
        timeAxis: comparison && comparison.time_axis,
        traces: series.map(row => row.trace),
        unit,
      }, fallbackChart)
      : fallbackChart;
    return `
      ${chart}
      <p class="pt-multi-note">${hEsc(helpers, hT(
        helpers,
        'Numeric ICU offsets retain their source spacing; dated traces align to hours since each entity’s first observed point. Hover for exact bounded values, or drag the navigator to zoom.',
        '数值型 ICU 偏移保留源时间间隔；日期型轨迹按各实体首个观测点后的小时数对齐。悬停查看有界精确值，拖动导航条可缩放。',
      ))}</p>`;
  }

  function renderAggregateFallback(rows, helpers) {
    if (!rows.length) {
      return `
        <div class="pt-series-panel" data-patient-series-panel="compare">
          <div class="state empty">
            <div class="t">${hEsc(helpers, hT(helpers, 'No multi-patient comparison payload', '没有多患者对比载荷'))}</div>
            <div class="d">${hEsc(helpers, hT(helpers, 'Load a real export or a bounded demo payload with feature coverage to compare one feature across entities.', '加载真实导出或带特征覆盖的有界演示载荷后，才能比较同一特征在多个实体中的表现。'))}</div>
          </div>
        </div>`;
    }
    const maxRecords = Math.max(1, ...rows.map(row => Number(row.records || row.rows || 0)));
    return `
      <div class="pt-compare-table table-wrap table-scroll">
        <table class="eu-table">
          <thead>
            <tr>
              <th>${hEsc(helpers, hT(helpers, 'Feature', '特征'))}</th>
              <th>${hEsc(helpers, hT(helpers, 'Module', '模块'))}</th>
              <th class="num">${hEsc(helpers, hT(helpers, 'Entities', '实体'))}</th>
              <th class="num">${hEsc(helpers, hT(helpers, 'Records', '记录'))}</th>
              <th class="num">${hEsc(helpers, hT(helpers, 'Coverage', '覆盖'))}</th>
              <th>${hEsc(helpers, hT(helpers, 'Record density', '记录密度'))}</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(row => {
              const records = Number(row.records || row.rows || 0);
              const coverage = Number(row.coverage_pct);
              const width = Math.max(2, Math.min(100, Number.isFinite(coverage) ? coverage : (records / maxRecords) * 100));
              return `<tr>
                <td class="key">${hEsc(helpers, row.label || row.name || row.feature)}</td>
                <td>${hEsc(helpers, row.module || '')}</td>
                <td class="num">${hFmtInt(helpers, row.entities)}</td>
                <td class="num">${hFmtInt(helpers, records)}</td>
                <td class="num">${Number.isFinite(coverage) ? (helpers.fmtPct ? helpers.fmtPct(coverage) : `${coverage}%`) : '—'}</td>
                <td><div class="pt-compare-bar"><i style="width:${width}%;"></i></div></td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>`;
  }

  function renderCompare(review, helpers) {
    const comparison = ((review || {}).multi_entity_comparison || {});
    const traces = (comparison.traces || []).filter(trace => traceValues(trace).length >= 2);
    const rows = (comparison.features || [])
      .filter(row => row && (row.feature || row.label || row.name))
      .slice(0, 24);
    if (traces.length >= 2) {
      return `
        <div class="pt-series-panel" data-patient-series-panel="compare">
          <div class="pt-series-panel-head">
            <div>
              <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Cross-patient comparison', '跨患者对比'))}</div>
              <h2>${hEsc(helpers, hT(helpers, 'Same feature across pseudonymous entities', '同一特征在多个伪匿名实体中的轨迹'))}</h2>
              <p>${hEsc(helpers, hT(helpers, 'This restores the old Patient Review comparison mode: one selected feature, several bounded entities, no direct identifiers.', '这里恢复旧版患者审阅的对比模式：一个已选特征、多个有界实体、没有直接标识符。'))}</p>
            </div>
            <span class="pill ok">${hFmtInt(helpers, traces.length)} ${hEsc(helpers, hT(helpers, 'entities', '个实体'))}</span>
          </div>
          <div class="pt-compare-feature">
            <div>
              <div class="eyebrow">${hEsc(helpers, comparison.module_label || comparison.module || hT(helpers, 'Module', '模块'))}</div>
              <strong>${hEsc(helpers, comparison.label || comparison.feature || hT(helpers, 'Selected feature', '已选特征'))}</strong>
              ${comparison.unit ? `<span class="mono">${hEsc(helpers, comparison.unit)}</span>` : ''}
            </div>
            <span class="pill">${hEsc(helpers, hT(helpers, 'bounded trace payload', '有界轨迹载荷'))}</span>
          </div>
          ${multiTraceChart(traces, comparison, helpers)}
          <div class="pt-trace-legend">
            ${traces.map((trace, index) => {
              const values = traceValues(trace);
              const last = values[values.length - 1];
              const hidden = Math.max(0, Number(trace.point_count || values.length) - values.length);
              return `<div class="pt-trace-key">
                <i style="background:${COLORS[index % COLORS.length]};"></i>
                <span>${hEsc(helpers, trace.label || trace.ref || `${hT(helpers, 'Entity', '实体')} ${index + 1}`)}</span>
                <b>${hEsc(helpers, helpers.fmtNum ? helpers.fmtNum(last, 2) : String(last))}</b>
                ${hidden ? `<em>+${hFmtInt(helpers, hidden)}</em>` : ''}
              </div>`;
            }).join('')}
          </div>
        </div>`;
    }
    return `
      <div class="pt-series-panel" data-patient-series-panel="compare">
        <div class="pt-series-panel-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Cross-patient comparison', '跨患者对比'))}</div>
            <h2>${hEsc(helpers, hT(helpers, 'No comparable multi-entity trajectories', '暂无可对比的多患者轨迹'))}</h2>
            <p>${hEsc(helpers, hT(helpers, 'The active bounded payload does not contain one time-indexed feature observed in at least two entities. Aggregate coverage is shown only as a fallback audit.', '当前有界载荷里没有至少两个实体同时观测到的同一时序特征；下面的聚合覆盖只作为 fallback 审计。'))}</p>
          </div>
          <span class="pill warn">${hEsc(helpers, hT(helpers, 'fallback audit', 'fallback 审计'))}</span>
        </div>
        ${renderAggregateFallback(rows, helpers)}
      </div>`;
  }

  function renderTimeSeriesWorkspace(payload = {}, helpers = {}) {
    const review = payload.review || {};
    const lanes = Array.isArray(payload.lanes) ? payload.lanes : [];
    const mode = ['lanes', 'single', 'compare'].includes(payload.mode) ? payload.mode : 'lanes';
    const chartOwner = window.EU_PATIENT_CHARTS;
    if (chartOwner && chartOwner.begin) chartOwner.begin();
    const body = mode === 'single'
      ? renderSinglePatient(lanes, payload.selected || {}, helpers)
      : mode === 'compare'
        ? renderCompare(review, helpers)
        : renderClinicalLanes(lanes, helpers);
    return `
      <section class="pt-series-workbench" data-patient-series-workbench>
        <div class="pt-series-workbench-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Time-series workspace', '时间序列工作区'))}</div>
            <h2>${hEsc(helpers, hT(helpers, 'Clinical trajectory review', '临床轨迹审阅'))}</h2>
            <p>${hEsc(helpers, hT(helpers, 'Use the module overview to load source-backed features, then inspect loaded charts or compare the same feature across patients.', '先在模块总览加载来源可追溯的特征，再查看轨迹画廊或进行跨患者同特征对比。'))}</p>
          </div>
        </div>
        ${renderModeBar(mode, helpers)}
        ${body}
      </section>`;
  }

  window.EU_PATIENT_SERIES = {
    renderModulePanels,
    renderTimeSeriesWorkspace,
    signalKey,
    numericSamples,
    numericValues,
  };
})();
