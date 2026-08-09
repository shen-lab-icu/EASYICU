/* ============================================================
   screens-viz-echarts.js — shared visualization-shell owner.
   Owns the local ECharts theme, SVG renderer lifecycle, resize
   handling, and semantic fallback handoff. Route owners only
   translate their reviewed payloads into chart options.
   ============================================================ */
(function () {
  const VERSION = '6.1.0';
  const live = new Map();

  function text(value, fallback = '') {
    return String(value == null ? fallback : value)
      .replace(/[\u0000-\u001f\u007f]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 240);
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

  function cssColor(name, fallback) {
    try {
      const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return value || fallback;
    } catch (_) {
      return fallback;
    }
  }

  function palette() {
    return {
      accent: cssColor('--accent', '#0f766e'),
      blue: '#2563eb',
      violet: '#7c3aed',
      gold: '#b45309',
      rose: '#be123c',
      ink: cssColor('--ink', '#17202a'),
      muted: cssColor('--ink-4', '#64748b'),
      hair: cssColor('--hair', '#e2e8f0'),
      surface: cssColor('--surface', '#ffffff'),
      series: [
        cssColor('--accent', '#0f766e'),
        '#2563eb',
        '#7c3aed',
        '#b45309',
        '#be123c',
      ],
    };
  }

  function available() {
    return Boolean(window.echarts && typeof window.echarts.init === 'function');
  }

  function baseOption(description) {
    const colors = palette();
    return {
      animation: false,
      backgroundColor: 'transparent',
      color: colors.series.slice(),
      textStyle: {
        color: colors.ink,
        fontFamily: 'IBM Plex Sans, system-ui, sans-serif',
      },
      aria: {
        show: true,
        label: { description: text(description, 'Clinical data visualization.') },
      },
    };
  }

  function grid(options = {}) {
    return {
      left: options.left == null ? 52 : options.left,
      right: options.right == null ? 18 : options.right,
      top: options.top == null ? 24 : options.top,
      bottom: options.bottom == null ? 42 : options.bottom,
      containLabel: Boolean(options.containLabel),
    };
  }

  function axis(options = {}) {
    const colors = palette();
    const result = {
      type: options.type || 'value',
      name: text(options.name, ''),
      nameLocation: options.nameLocation || 'middle',
      nameGap: options.nameGap == null ? 30 : options.nameGap,
      nameTextStyle: { color: colors.muted, fontSize: 10 },
      axisLine: { show: options.axisLine !== false, lineStyle: { color: colors.hair } },
      axisTick: { show: false },
      axisLabel: {
        color: colors.muted,
        fontFamily: 'IBM Plex Mono, ui-monospace, monospace',
        fontSize: 9,
        hideOverlap: true,
        formatter: options.formatter,
      },
      splitLine: {
        show: options.splitLine !== false,
        lineStyle: { color: colors.hair, type: options.splitType || 'solid' },
      },
      boundaryGap: options.boundaryGap == null ? false : options.boundaryGap,
      scale: Boolean(options.scale),
    };
    if (Array.isArray(options.data)) result.data = options.data;
    if (options.min != null) result.min = options.min;
    if (options.max != null) result.max = options.max;
    if (options.inverse != null) result.inverse = Boolean(options.inverse);
    return result;
  }

  function tooltip(formatter, trigger = 'axis') {
    return {
      trigger,
      renderMode: 'richText',
      confine: true,
      appendToBody: false,
      formatter,
      axisPointer: trigger === 'axis' ? { type: 'line', snap: true } : undefined,
    };
  }

  function legend(options = {}) {
    const colors = palette();
    return {
      type: 'scroll',
      top: options.top == null ? 0 : options.top,
      left: options.left == null ? 8 : options.left,
      right: options.right == null ? 8 : options.right,
      itemWidth: options.itemWidth == null ? 16 : options.itemWidth,
      itemHeight: options.itemHeight == null ? 3 : options.itemHeight,
      textStyle: { color: colors.muted, fontSize: 10 },
    };
  }

  function revealFallback(element, fallback) {
    if (element && typeof element.setAttribute === 'function') element.setAttribute('hidden', '');
    if (fallback && typeof fallback.removeAttribute === 'function') fallback.removeAttribute('hidden');
  }

  function mount(element, option, options = {}) {
    if (!available() || !element || !option) {
      revealFallback(element, options.fallback);
      return false;
    }
    if (live.has(element)) return true;
    let chart = null;
    try {
      chart = window.echarts.init(element, null, { renderer: 'svg' });
      chart.setOption(option, { notMerge: true, lazyUpdate: false });
      let observer = null;
      if (typeof ResizeObserver === 'function') {
        observer = new ResizeObserver(() => chart.resize());
        observer.observe(element);
      }
      live.set(element, {
        chart,
        observer,
        owner: text(options.owner, 'shared'),
      });
      return true;
    } catch (error) {
      if (chart && typeof chart.dispose === 'function') chart.dispose();
      revealFallback(element, options.fallback);
      if (typeof options.onError === 'function') options.onError(error);
      return false;
    }
  }

  function dispose(owner) {
    const target = owner == null ? null : text(owner);
    live.forEach((record, element) => {
      if (target && record.owner !== target) return;
      if (record.observer) record.observer.disconnect();
      if (record.chart && typeof record.chart.dispose === 'function') record.chart.dispose();
      live.delete(element);
    });
  }

  window.EU_ECHARTS = {
    VERSION,
    available,
    axis,
    baseOption,
    dispose,
    finite,
    formatNumber,
    grid,
    legend,
    mount,
    palette,
    tooltip,
  };
})();
