/* ============================================================
   screens-viz-patient-charts.js — interactive chart widget owner
   for Patient Review. ECharts is vendored locally and optional:
   if it is unavailable, screens-viz-patient-series.js keeps its
   existing semantic SVG fallback.
   ============================================================ */
(function () {
  /* Chart labels keep their control-character + whitespace normalisation
     (`text`) before escaping; only the escaping itself is shared. */
  const esc = value => window.EU_HTML.esc(text(value));
  const VERSION = '6.1.0';
  const MAX_TEXT = 180;
  const STEP_FEATURES = new Set([
    'abx', 'cort', 'rrt', 'mech_vent', 'vent_ind', 'supp_o2', 'adv_resp', 'vaso_ind',
    'norepi_rate', 'norepi_equiv', 'epi_rate', 'dopa_rate', 'dobu_rate', 'adh_rate',
    'phn_rate', 'fio2', 'peep', 'ins',
  ]);
  let renderId = 0;
  let pending = new Map();
  const live = new Map();

  function chartCore() {
    return window.EU_ECHARTS || null;
  }

  /* Straight to the palette owner rather than through the chart shell: this
     file also renders when the shell is missing, and a local fallback here is
     how the strokes silently diverged from the colours before. */
  function traceLineStyle(index, width) {
    return window.EU_PALETTE.lineStyle(index, { width });
  }

  function text(value, fallback = '') {
    const normalized = String(value == null ? fallback : value)
      .replace(/[\u0000-\u001f\u007f]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    return normalized.slice(0, MAX_TEXT);
  }


  function finite(value) {
    if (value == null || (typeof value === 'string' && !value.trim())) return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatNumber(value, digits = 1) {
    const number = finite(value);
    if (number == null) return '—';
    const precision = Math.abs(number) < 1 ? Math.max(2, digits) : digits;
    return Number(number.toFixed(precision)).toLocaleString('en-US');
  }

  function median(values) {
    const numbers = values.map(finite).filter(value => value != null).sort((a, b) => a - b);
    if (!numbers.length) return null;
    const middle = Math.floor(numbers.length / 2);
    return numbers.length % 2
      ? numbers[middle]
      : (numbers[middle - 1] + numbers[middle]) / 2;
  }

  function cssColor(name, fallback) {
    try {
      const root = document.documentElement;
      const value = getComputedStyle(root).getPropertyValue(name).trim();
      return value || fallback;
    } catch (_) {
      return fallback;
    }
  }

  function palette() {
    const core = chartCore();
    if (core && typeof core.palette === 'function') return core.palette();
    return {
      accent: cssColor('--accent', '#0f766e'),
      ink: cssColor('--ink', '#17202a'),
      muted: cssColor('--ink-4', '#64748b'),
      hair: cssColor('--hair', '#e2e8f0'),
      surface: cssColor('--surface', '#ffffff'),
      series: window.EU_PALETTE.series(),
    };
  }

  function normalizedAxisMeta(meta) {
    const source = meta && typeof meta === 'object' ? meta : {};
    return {
      kind: text(source.kind, ''),
      label: text(source.label || (window.EU_LANG === 'zh' ? source.label_zh : source.label_en) || source.label_en || source.label_zh, ''),
      unit: text(source.unit, ''),
      sourceColumn: text(source.source_column, ''),
    };
  }

  function numericPairs(times, values, axisMeta) {
    const meta = normalizedAxisMeta(axisMeta);
    const pairs = [];
    values.forEach((rawValue, index) => {
      const y = finite(rawValue);
      const rawTime = times[index];
      const x = finite(rawTime);
      if (x == null || y == null) return;
      const normalizedX = meta.kind === 'relative_minutes' ? x / 60 : x;
      pairs.push([normalizedX, y]);
    });
    if (pairs.length < 2) return null;
    let label = meta.label;
    if (!label) {
      if (meta.kind === 'relative_hours') label = 'ICU hour';
      else if (meta.kind === 'relative_minutes') label = 'ICU hour';
      else label = 'Recorded offset';
    }
    return {
      kind: 'value',
      label,
      pairs,
      tooltip: value => `${formatNumber(value, 1)}${meta.kind.startsWith('relative_') ? ' h' : ''}`,
    };
  }

  function datedPairs(times, values, axisMeta, relativeToFirst) {
    const parsed = [];
    values.forEach((rawValue, index) => {
      const y = finite(rawValue);
      const stamp = Date.parse(String(times[index] == null ? '' : times[index]));
      if (!Number.isFinite(stamp) || y == null) return;
      parsed.push([stamp, y]);
    });
    if (parsed.length < 2) return null;
    if (relativeToFirst) {
      const first = parsed[0][0];
      return {
        kind: 'value',
        label: 'Hours since first observed point',
        pairs: parsed.map(([stamp, value]) => [(stamp - first) / 3600000, value]),
        tooltip: value => `${formatNumber(value, 1)} h`,
      };
    }
    const meta = normalizedAxisMeta(axisMeta);
    return {
      kind: 'time',
      label: meta.label || 'Recorded time',
      pairs: parsed,
      tooltip: value => {
        const date = new Date(Number(value));
        return Number.isNaN(date.getTime()) ? text(value) : date.toLocaleString();
      },
    };
  }

  function categoryPairs(times, values) {
    const rows = [];
    values.forEach((rawValue, index) => {
      const y = finite(rawValue);
      if (y == null) return;
      rows.push([text(times[index], `Observation ${index + 1}`), y]);
    });
    if (rows.length < 2) return null;
    return {
      kind: 'category',
      label: 'Recorded order',
      categories: rows.map(row => row[0]),
      pairs: rows.map(row => row[1]),
      tooltip: value => text(value),
    };
  }

  function axisData(times, values, axisMeta, options = {}) {
    const cleanValues = Array.isArray(values) ? values : [];
    const cleanTimes = Array.isArray(times) ? times.slice(0, cleanValues.length) : [];
    const meta = normalizedAxisMeta(axisMeta);
    if (cleanTimes.length === cleanValues.length && cleanTimes.length >= 2) {
      if (cleanTimes.every(value => finite(value) != null)) {
        return numericPairs(cleanTimes, cleanValues, meta);
      }
      const dated = datedPairs(cleanTimes, cleanValues, meta, Boolean(options.relativeDates));
      if (dated) return dated;
      return categoryPairs(cleanTimes, cleanValues);
    }
    return categoryPairs(cleanValues.map((_, index) => `Observation ${index + 1}`), cleanValues);
  }

  function xAxisOption(axis, colors, compact) {
    const base = {
      type: axis.kind,
      name: compact ? '' : axis.label,
      nameLocation: 'middle',
      nameGap: compact ? 0 : 30,
      nameTextStyle: { color: colors.muted, fontSize: 12 },
      axisLine: { lineStyle: { color: colors.hair } },
      axisTick: { show: false },
      axisLabel: {
        color: colors.muted,
        fontSize: 12,
        hideOverlap: true,
        formatter: axis.kind === 'value'
          ? value => formatNumber(value, 1)
          : undefined,
      },
      splitLine: { show: false },
      boundaryGap: false,
    };
    if (axis.kind === 'category') base.data = axis.categories;
    return base;
  }

  function markLines(values, thresholds, unit, colors) {
    const rows = [];
    const midpoint = median(values);
    if (midpoint != null) {
      rows.push({
        name: 'Median',
        yAxis: midpoint,
        lineStyle: { color: '#94a3b8', type: 'dashed', width: 1 },
      });
    }
    (Array.isArray(thresholds) ? thresholds : []).slice(0, 4).forEach((row, index) => {
      const value = finite(row && row.value);
      if (value == null) return;
      rows.push({
        name: text((row && row.label) || `Threshold ${index + 1}`),
        yAxis: value,
        lineStyle: {
          color: (row && row.color) || (index % 2 ? '#f97316' : '#ef4444'),
          type: row && row.dash ? 'dashed' : 'dashed',
          width: 1,
        },
        label: {
          formatter: `${formatNumber(value, 1)}${unit ? ` ${text(unit)}` : ''}`,
        },
      });
    });
    return {
      silent: true,
      symbol: ['none', 'none'],
      label: { show: false },
      data: rows,
    };
  }

  function tooltipFormatter(axis, unit) {
    return params => {
      const rows = Array.isArray(params) ? params : [params];
      if (!rows.length) return '';
      const firstValue = rows[0] && rows[0].value;
      const rawX = Array.isArray(firstValue) ? firstValue[0] : rows[0].axisValue;
      const lines = [axis.tooltip(rawX)];
      rows.forEach(row => {
        const value = Array.isArray(row.value) ? row.value[1] : row.value;
        lines.push(`${text(row.seriesName, 'Value')}: ${formatNumber(value, 2)}${unit ? ` ${text(unit)}` : ''}`);
      });
      return lines.join('\n');
    };
  }

  function signalOption(spec) {
    const colors = palette();
    const values = (spec.values || []).map(finite).filter(value => value != null);
    const axis = axisData(spec.times, spec.values, spec.timeAxis);
    if (!axis || values.length < 2) return null;
    const label = text(spec.label || spec.feature, 'Clinical signal');
    const unit = text(spec.unit, '');
    return {
      animation: false,
      color: [spec.color || colors.accent],
      aria: {
        show: true,
        label: {
          description: `${label}. ${values.length} bounded observations. The adjacent summary reports the latest value; pointer users can hover for exact plotted values.`,
        },
      },
      grid: { left: 58, right: 18, top: 16, bottom: 34, containLabel: false },
      tooltip: {
        trigger: 'axis',
        renderMode: 'richText',
        confine: true,
        appendToBody: false,
        formatter: tooltipFormatter(axis, unit),
        axisPointer: { type: 'line', snap: true },
      },
      xAxis: Object.assign(xAxisOption(axis, colors, true), axis.kind === 'value' && Array.isArray(spec.xDomain) ? { min: spec.xDomain[0], max: spec.xDomain[1] } : {}),
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: colors.muted,
          fontSize: 12,
          formatter: value => formatNumber(value, 1),
        },
        splitLine: { lineStyle: { color: colors.hair, type: 'solid' } },
        ...(Array.isArray(spec.yDomain) ? { min: spec.yDomain[0], max: spec.yDomain[1] } : {}),
      },
      series: [{
        name: label,
        type: 'line',
        data: axis.pairs,
        showSymbol: axis.pairs.length <= 24,
        symbol: 'circle',
        symbolSize: 6,
        smooth: false,
        step: STEP_FEATURES.has(text(spec.feature, '').toLowerCase()) ? 'end' : false,
        connectNulls: false,
        sampling: 'lttb',
        lineStyle: { width: 2 },
        itemStyle: { borderColor: colors.surface, borderWidth: 1 },
        emphasis: { lineStyle: { width: 3 } },
        markLine: markLines(values, spec.thresholds, unit, colors),
      }],
    };
  }

  function comparisonOption(spec) {
    const colors = palette();
    const traces = (Array.isArray(spec.traces) ? spec.traces : [])
      .map((trace, index) => {
        const axis = axisData(trace.times, trace.values, trace.time_axis || spec.timeAxis, { relativeDates: true });
        if (!axis) return null;
        return {
          axis,
          label: text(trace.label || trace.ref, `Entity ${index + 1}`),
          color: colors.series[index % colors.series.length],
        };
      })
      .filter(Boolean);
    if (traces.length < 2) return null;
    const axis = traces[0].axis;
    const label = text(spec.label || spec.feature, 'Selected feature');
    const unit = text(spec.unit, '');
    return {
      animation: false,
      color: traces.map(row => row.color),
      aria: {
        show: true,
        label: {
          description: `${label}. ${traces.length} bounded pseudonymous entity trajectories compared on a shared elapsed-time axis.`,
        },
      },
      legend: {
        type: 'scroll',
        top: 0,
        left: 8,
        right: 8,
        itemWidth: 16,
        itemHeight: 3,
        textStyle: { color: colors.muted, fontSize: 12 },
      },
      grid: { left: 54, right: 18, top: 38, bottom: 58, containLabel: false },
      tooltip: {
        trigger: 'axis',
        renderMode: 'richText',
        confine: true,
        appendToBody: false,
        formatter: tooltipFormatter(axis, unit),
        axisPointer: { type: 'line', snap: true },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
          zoomOnMouseWheel: 'shift',
          moveOnMouseWheel: false,
          moveOnMouseMove: true,
        },
        {
          type: 'slider',
          xAxisIndex: 0,
          filterMode: 'none',
          height: 18,
          bottom: 8,
          borderColor: colors.hair,
          backgroundColor: colors.surface,
          fillerColor: 'rgba(15, 118, 110, .12)',
          handleStyle: { color: colors.accent },
          showDetail: false,
        },
      ],
      xAxis: xAxisOption(axis, colors, false),
      yAxis: {
        type: 'value',
        scale: true,
        name: unit,
        nameTextStyle: { color: colors.muted, fontSize: 12 },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: colors.muted,
          fontSize: 12,
          formatter: value => formatNumber(value, 1),
        },
        splitLine: { lineStyle: { color: colors.hair } },
      },
      series: traces.map((row, index) => ({
        name: row.label,
        type: 'line',
        data: row.axis.pairs,
        showSymbol: row.axis.pairs.length <= 24,
        symbolSize: 4,
        smooth: false,
        connectNulls: false,
        sampling: 'lttb',
        lineStyle: traceLineStyle(index, 2),
        emphasis: { lineStyle: { width: 3 } },
      })),
    };
  }

  function available() {
    const core = chartCore();
    if (core && typeof core.available === 'function') return core.available();
    return Boolean(window.echarts && typeof window.echarts.init === 'function');
  }

  function begin() {
    dispose();
    pending = new Map();
    renderId += 1;
  }

  function chartSummary(label, values, unit) {
    const numbers = (values || []).map(finite).filter(value => value != null);
    if (!numbers.length) return label;
    const suffix = unit ? ` ${unit}` : '';
    return `${label}. ${numbers.length} bounded observations. Minimum ${formatNumber(Math.min(...numbers), 2)}${suffix}. Maximum ${formatNumber(Math.max(...numbers), 2)}${suffix}. Latest ${formatNumber(numbers[numbers.length - 1], 2)}${suffix}.`;
  }

  function slot(kind, spec, fallbackHtml) {
    if (!available()) return fallbackHtml || '';
    const id = `pt-ec-${renderId}-${pending.size + 1}`;
    pending.set(id, { kind, spec });
    const label = text(spec.label || spec.feature, 'Clinical chart');
    const values = kind === 'signal' ? (spec.values || []) : [];
    const current = values.length ? finite(values[values.length - 1]) : null;
    const unit = text(spec.unit, '');
    const ariaLabel = kind === 'signal'
      ? chartSummary(label, values, unit)
      : `${label}. Bounded pseudonymous entity comparison.`;
    const currentHtml = current == null ? '' : `
      <span class="pt-echart-current">${esc(spec.latestLabel || 'Latest')}:
        <b>${esc(formatNumber(current, 2))}${unit ? ` ${esc(unit)}` : ''}</b>
      </span>`;
    return `
      <div class="pt-echart-shell" data-patient-chart-kind="${esc(kind)}">
        ${currentHtml}
        <div class="pt-echart ${kind === 'comparison' ? 'comparison' : ''}"
             data-patient-echart="${esc(id)}"
             role="img"
             aria-label="${esc(ariaLabel)}"></div>
        <div class="pt-echart-fallback" data-patient-echart-fallback hidden>${fallbackHtml || ''}</div>
      </div>`;
  }

  function signalSlot(spec, fallbackHtml) {
    return slot('signal', spec, fallbackHtml);
  }

  function comparisonSlot(spec, fallbackHtml) {
    return slot('comparison', spec, fallbackHtml);
  }

  function mount(root) {
    if (!available() || !root || typeof root.querySelectorAll !== 'function') return 0;
    const elements = Array.from(root.querySelectorAll('[data-patient-echart]'));
    let mounted = 0;
    const showFallback = element => {
      const shell = element && typeof element.closest === 'function' ? element.closest('.pt-echart-shell') : null;
      const fallback = shell && typeof shell.querySelector === 'function'
        ? shell.querySelector('[data-patient-echart-fallback]')
        : null;
      if (element && typeof element.setAttribute === 'function') element.setAttribute('hidden', '');
      if (fallback && typeof fallback.removeAttribute === 'function') fallback.removeAttribute('hidden');
    };
    elements.forEach(element => {
      const id = element.getAttribute('data-patient-echart');
      const record = pending.get(id);
      if (!record || live.has(element)) return;
      const option = record.kind === 'comparison'
        ? comparisonOption(record.spec)
        : signalOption(record.spec);
      if (!option) {
        showFallback(element);
        pending.delete(id);
        return;
      }
      try {
        const core = chartCore();
        if (core && typeof core.mount === 'function') {
          const shell = element && typeof element.closest === 'function'
            ? element.closest('.pt-echart-shell')
            : null;
          const fallback = shell && typeof shell.querySelector === 'function'
            ? shell.querySelector('[data-patient-echart-fallback]')
            : null;
          const ok = core.mount(element, option, {
            owner: 'patient',
            fallback,
            onError(error) {
              if (window.console && typeof window.console.warn === 'function') {
                window.console.warn('Patient chart fell back to semantic SVG.', error);
              }
            },
          });
          if (ok) mounted += 1;
          pending.delete(id);
          return;
        }
        const chart = window.echarts.init(element, null, { renderer: 'svg' });
        chart.setOption(option, { notMerge: true, lazyUpdate: false });
        let observer = null;
        if (typeof ResizeObserver === 'function') {
          observer = new ResizeObserver(() => chart.resize());
          observer.observe(element);
        }
        live.set(element, { chart, observer });
        mounted += 1;
      } catch (error) {
        showFallback(element);
        if (window.console && typeof window.console.warn === 'function') {
          window.console.warn('Patient chart fell back to semantic SVG.', error);
        }
      } finally {
        pending.delete(id);
      }
    });
    return mounted;
  }

  function dispose() {
    const core = chartCore();
    if (core && typeof core.dispose === 'function') core.dispose('patient');
    live.forEach(record => {
      if (record.observer) record.observer.disconnect();
      if (record.chart && typeof record.chart.dispose === 'function') record.chart.dispose();
    });
    live.clear();
  }

  if (typeof window.addEventListener === 'function') {
    window.addEventListener('hashchange', () => {
      const route = String((window.location && window.location.hash) || '').replace(/^#/, '');
      if (route !== 'patient') dispose();
    });
  }

  window.EU_PATIENT_CHARTS = {
    VERSION,
    available,
    begin,
    comparisonOption,
    comparisonSlot,
    dispose,
    mount,
    signalOption,
    signalSlot,
  };
})();
