/* ============================================================
   screens-viz-crossdb-charts.js — Cross-DB chart renderer owner.
   Receives one selected aggregate feature and builds either a
   density-line or categorical-bar ECharts view. Result browsing,
   filters, source tables, and route state remain separate.
   ============================================================ */
(function () {
  'use strict';

  let renderId = 0;
  let pending = new Map();

  function core() {
    return window.EU_ECHARTS || null;
  }

  function palette(index) {
    return window.EU_PALETTE.color(index);
  }

  /* SVG equivalent of the ECharts stroke: `[7, 4]` becomes "7 4". */
  function dashArray(index) {
    const dash = window.EU_PALETTE.dashPattern(index);
    return dash ? ` stroke-dasharray="${dash.join(' ')}"` : '';
  }

  function begin() {
    const chartCore = core();
    if (chartCore && typeof chartCore.dispose === 'function') chartCore.dispose('crossdb');
    pending = new Map();
    renderId += 1;
  }

  function slot(kind, option, fallbackHtml, ariaLabel) {
    const chartCore = core();
    if (!chartCore || typeof chartCore.available !== 'function' || !chartCore.available() || !option) {
      return fallbackHtml || '';
    }
    const id = `xdb-ec-${renderId}-${pending.size + 1}`;
    pending.set(id, option);
    return `<div class="xdb-main-chart" data-crossdb-chart-kind="${kind}">
      <div class="xdb-echart" data-crossdb-echart="${id}" role="img" aria-label="${ariaLabel}"></div>
      <div data-crossdb-echart-fallback hidden>${fallbackHtml || ''}</div>
    </div>`;
  }

  function densityOption(series, unit, description) {
    const chartCore = core();
    if (!chartCore) return null;
    const colors = chartCore.palette();
    return Object.assign(chartCore.baseOption(description), {
      color: series.map((_, index) => colors.series[index % colors.series.length]),
      grid: chartCore.grid({ left: 70, right: 30, top: 54, bottom: 58 }),
      legend: chartCore.legend({ top: 4 }),
      tooltip: chartCore.tooltip(params => {
        const rows = Array.isArray(params) ? params : [params];
        if (!rows.length) return '';
        const first = Array.isArray(rows[0].value) ? rows[0].value[0] : rows[0].axisValue;
        const lines = [`${chartCore.formatNumber(first, 2)}${unit ? ` ${unit}` : ''}`];
        rows.forEach(row => {
          const value = Array.isArray(row.value) ? row.value[1] : row.value;
          lines.push(`${row.seriesName}: ${chartCore.formatNumber(value, 4)}`);
        });
        return lines.join('\n');
      }),
      xAxis: chartCore.axis({
        type: 'value',
        name: unit,
        nameGap: 28,
        splitLine: true,
        formatter: value => chartCore.formatNumber(value, 2),
      }),
      yAxis: chartCore.axis({
        type: 'value',
        name: window.EU_LANG === 'zh' ? '相对密度' : 'Relative density',
        nameGap: 38,
        scale: true,
        axisLine: false,
        formatter: value => chartCore.formatNumber(value, 3),
      }),
      series: series.map((row, index) => ({
        name: row.label,
        type: 'line',
        data: row.value.points
          .filter(point => Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.density)))
          .map(point => [Number(point.x), Number(point.density)]),
        showSymbol: row.value.points.length <= 18,
        symbol: index % 2 ? 'emptyCircle' : 'circle',
        symbolSize: 6,
        smooth: false,
        connectNulls: false,
        sampling: 'lttb',
        /* Shared per-index stroke: the old `index % 3 === 2` rule left series
           0 and 1 — the common two-database comparison — both solid, so they
           were separable by colour alone. */
        lineStyle: chartCore.lineStyle(index, { width: 2.4 }),
        emphasis: { focus: 'series', lineStyle: { width: 3.2 } },
      })),
    });
  }

  function numericFallback(series, unit, h) {
    const xs = series.flatMap(row => row.value.points.map(point => Number(point.x)).filter(Number.isFinite));
    const ys = series.flatMap(row => row.value.points.map(point => Number(point.density)).filter(Number.isFinite));
    if (!xs.length || !ys.length) return '';
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const maxY = Math.max(0.000001, ...ys);
    const width = 760;
    const height = 250;
    const left = 42;
    const right = 18;
    const top = 18;
    const bottom = 38;
    const xScale = value => left + ((Number(value) - minX) / ((maxX - minX) || 1)) * (width - left - right);
    const yScale = value => top + (1 - Number(value) / maxY) * (height - top - bottom);
    const paths = series.map((row, index) => {
      const points = row.value.points.filter(point => Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.density)));
      const line = points.map((point, position) => `${position ? 'L' : 'M'}${xScale(point.x).toFixed(1)},${yScale(point.density).toFixed(1)}`).join(' ');
      return `<path class="xdb-main-density-line" d="${line}" stroke="${row.color}"${dashArray(index)}></path>`;
    }).join('');
    const tickValues = [minX, minX + (maxX - minX) / 2, maxX];
    return `<div class="xdb-main-chart xdb-chart-fallback">
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${h.esc(h.t('Overlaid aggregate density curves', '叠加聚合密度曲线'))}">
        <line class="xdb-main-axis" x1="${left}" x2="${width - right}" y1="${height - bottom}" y2="${height - bottom}"></line>
        ${tickValues.map(value => `<line class="xdb-main-grid" x1="${xScale(value)}" x2="${xScale(value)}" y1="${top}" y2="${height - bottom}"></line><text x="${xScale(value)}" y="${height - 14}" text-anchor="middle">${h.esc(formatDensity(value))}</text>`).join('')}
        ${paths}
        <text x="${left}" y="12">${h.esc(h.t('Relative density', '相对密度'))}</text>
        <text x="${width - right}" y="${height - 14}" text-anchor="end">${h.esc(unit)}</text>
      </svg>
      <div class="xdb-main-legend">${series.map(row => `<span><i style="background:${row.color}"></i>${h.esc(row.label)}</span>`).join('')}</div>
    </div>`;
  }

  function formatDensity(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    if (Math.abs(number) >= 1000 || (Math.abs(number) > 0 && Math.abs(number) < 0.01)) {
      return number.toExponential(2);
    }
    return Number(number.toFixed(2)).toLocaleString();
  }

  function categoricalOption(series, categories, description) {
    const chartCore = core();
    if (!chartCore) return null;
    return Object.assign(chartCore.baseOption(description), {
      grid: chartCore.grid({ left: 122, right: 24, top: 42, bottom: 36 }),
      legend: chartCore.legend({ top: 4 }),
      tooltip: chartCore.tooltip(params => {
        const rows = Array.isArray(params) ? params : [params];
        return rows.map(row => `${row.seriesName}: ${chartCore.formatNumber(row.value, 1)}%`).join('\n');
      }),
      xAxis: chartCore.axis({
        type: 'value',
        name: '%',
        min: 0,
        max: 100,
        formatter: value => `${chartCore.formatNumber(value, 0)}%`,
      }),
      yAxis: chartCore.axis({
        type: 'category',
        data: categories,
        inverse: true,
        splitLine: false,
        boundaryGap: true,
      }),
      series: series.map(row => ({
        name: row.label,
        type: 'bar',
        data: categories.map(category => {
          const found = row.categories.find(item => item.label === category);
          return found ? Number(found.pct) || 0 : 0;
        }),
        barMaxWidth: 18,
        itemStyle: { borderRadius: [0, 3, 3, 0] },
        emphasis: { focus: 'series' },
      })),
    });
  }

  function categoricalFallback(values, labels, h) {
    return `<div class="xdb-category-chart">
      ${values.map((value, index) => `<div>
        <b><i style="background:${palette(index)}"></i>${h.esc(labels[index] || value.source || `${h.t('Source', '来源')} ${index + 1}`)}</b>
        ${value.present ? (value.categories || []).slice(0, 8).map(category => `<span><em>${h.esc(category.label)}</em><i style="--xdb-cat-width:${Math.max(2, Number(category.pct) || 0)}%;background:${palette(index)}"></i><small>${h.fmtPct(category.pct)}</small></span>`).join('') : `<span class="missing">${h.t('Missing', '缺失')}</span>`}
      </div>`).join('')}
    </div>`;
  }

  function categoricalChart(item, labels, h) {
    const values = item.row.values || [];
    const series = values.map((value, index) => ({
      label: labels[index] || value.source || `${h.t('Source', '来源')} ${index + 1}`,
      categories: value && value.present && Array.isArray(value.categories) ? value.categories.slice(0, 8) : [],
    })).filter(row => row.categories.length);
    if (!series.length) return categoricalFallback(values, labels, h);
    const categories = Array.from(new Set(series.flatMap(row => row.categories.map(category => category.label)))).slice(0, 12);
    return slot(
      'category',
      categoricalOption(series, categories, h.t('Aggregate category percentage comparison.', '聚合分类百分比对比。')),
      categoricalFallback(values, labels, h),
      h.esc(h.t('Interactive aggregate category comparison', '交互式聚合分类对比')),
    );
  }

  function render(item, labels, h) {
    const values = item.row.values || [];
    const series = values.map((value, index) => ({
      color: palette(index),
      label: labels[index] || value.source || `${h.t('Source', '来源')} ${index + 1}`,
      value,
    })).filter(row => row.value && row.value.present && Array.isArray(row.value.points) && row.value.points.length >= 2);
    if (!series.length) return categoricalChart(item, labels, h);
    const unit = (h.catalogFeatureMeta(item.row.feature) || {}).unit || '';
    return slot(
      'density',
      densityOption(
        series,
        unit,
        `${h.t('Aggregate distribution comparison', '聚合分布对比')}. ${series.length} ${h.t('sources', '个来源')}.`,
      ),
      numericFallback(series, unit, h),
      h.esc(h.t('Interactive overlaid aggregate density curves', '交互式叠加聚合密度曲线')),
    );
  }

  function mount(root) {
    const chartCore = core();
    if (!chartCore || typeof chartCore.mount !== 'function' || !root) return 0;
    let mounted = 0;
    root.querySelectorAll('[data-crossdb-echart]').forEach(element => {
      const id = element.getAttribute('data-crossdb-echart');
      const option = pending.get(id);
      if (!option) return;
      const shell = typeof element.closest === 'function' ? element.closest('.xdb-main-chart') : null;
      const fallback = shell && typeof shell.querySelector === 'function'
        ? shell.querySelector('[data-crossdb-echart-fallback]')
        : null;
      if (chartCore.mount(element, option, {
        owner: 'crossdb',
        fallback,
        onError(error) {
          if (window.console && typeof window.console.warn === 'function') {
            window.console.warn('Cross-DB chart fell back to semantic markup.', error);
          }
        },
      })) mounted += 1;
      pending.delete(id);
    });
    return mounted;
  }

  function dispose() {
    const chartCore = core();
    if (chartCore && typeof chartCore.dispose === 'function') chartCore.dispose('crossdb');
    pending.clear();
  }

  window.EU_CROSSDB_CHARTS = {
    begin,
    categoricalOption,
    densityOption,
    dispose,
    mount,
    render,
  };
})();
