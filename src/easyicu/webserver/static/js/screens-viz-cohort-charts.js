/* ============================================================
   screens-viz-cohort-charts.js — Cohort Statistics chart owner.
   Converts reviewed aggregate survival and SOFA transition data
   into the shared ECharts contract. Tables, controls, and route
   state remain owned by screens-viz.js.
   ============================================================ */
(function () {
  'use strict';

  let renderId = 0;
  let pending = new Map();

  function core() {
    return window.EU_ECHARTS || null;
  }

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, character => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    })[character]);
  }

  function finite(value) {
    const chartCore = core();
    if (chartCore && typeof chartCore.finite === 'function') return chartCore.finite(value);
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function fill(tone, intensity) {
    const alpha = Math.max(0.12, Math.min(0.78, 0.16 + Number(intensity || 0) * 0.54));
    if (tone === 'up') return `rgba(190, 76, 76, ${alpha.toFixed(3)})`;
    if (tone === 'down') return `rgba(42, 111, 178, ${alpha.toFixed(3)})`;
    return `rgba(34, 137, 122, ${alpha.toFixed(3)})`;
  }

  function begin() {
    const chartCore = core();
    if (chartCore && typeof chartCore.dispose === 'function') chartCore.dispose('cohort');
    pending = new Map();
    renderId += 1;
  }

  function slot(kind, option, fallbackHtml, label, height) {
    const chartCore = core();
    if (!chartCore || typeof chartCore.available !== 'function' || !chartCore.available() || !option) {
      return fallbackHtml || '';
    }
    const id = `coh-ec-${renderId}-${pending.size + 1}`;
    pending.set(id, option);
    return `<div class="cohort-echart-shell ${kind}" data-cohort-chart-kind="${kind}">
      <div class="cohort-echart ${kind}" style="--cohort-chart-height:${Number(height) || 320}px"
           data-cohort-echart="${id}" role="img" aria-label="${esc(label)}"></div>
      <div data-cohort-echart-fallback hidden>${fallbackHtml || ''}</div>
    </div>`;
  }

  function survivalOption(spec) {
    const chartCore = core();
    if (!chartCore) return null;
    const groups = (spec.groups || []).filter(group => Array.isArray(group.points) && group.points.length);
    if (!groups.length) return null;
    return Object.assign(chartCore.baseOption(spec.description), {
      grid: chartCore.grid({ left: 58, right: 22, top: 44, bottom: 48 }),
      legend: chartCore.legend({ top: 4 }),
      tooltip: chartCore.tooltip(params => {
        const rows = Array.isArray(params) ? params : [params];
        if (!rows.length) return '';
        const first = Array.isArray(rows[0].value) ? rows[0].value[0] : rows[0].axisValue;
        const lines = [`${chartCore.formatNumber(first, 1)} ${spec.xLabel}`];
        rows.forEach(row => {
          const value = Array.isArray(row.value) ? row.value[1] : row.value;
          lines.push(`${row.seriesName}: ${chartCore.formatNumber(value, 1)}%`);
        });
        return lines.join('\n');
      }),
      xAxis: chartCore.axis({
        type: 'value',
        name: spec.xLabel,
        min: 0,
        formatter: value => chartCore.formatNumber(value, 1),
      }),
      yAxis: chartCore.axis({
        type: 'value',
        name: spec.yLabel,
        min: 0,
        max: 100,
        axisLine: false,
        formatter: value => `${chartCore.formatNumber(value, 0)}%`,
      }),
      series: groups.map((group, index) => ({
        name: group.label,
        type: 'line',
        data: group.points
          .map(point => [finite(point.time), finite(point.survival)])
          .filter(point => point[0] != null && point[1] != null),
        step: 'end',
        smooth: false,
        showSymbol: false,
        symbol: index % 2 ? 'emptyCircle' : 'circle',
        connectNulls: false,
        lineStyle: { width: 2.6, type: index % 3 === 2 ? 'dashed' : 'solid' },
        emphasis: { focus: 'series', lineStyle: { width: 3.4 } },
      })),
    });
  }

  function survivalFallback(spec) {
    return `<div class="table-wrap table-scroll cohort-chart-fallback">
      <table class="eu-table">
        <thead><tr><th>${esc(spec.groupLabel)}</th><th class="num">n</th><th class="num">${esc(spec.eventsLabel)}</th><th class="num">${esc(spec.finalLabel)}</th></tr></thead>
        <tbody>${(spec.groups || []).map(group => {
          const points = group.points || [];
          const last = points.length ? points[points.length - 1] : null;
          return `<tr><td class="key">${esc(group.label)}</td><td class="num">${esc(group.n)}</td><td class="num">${esc(group.events)}</td><td class="num">${last ? `${esc(last.survival)}%` : '—'}</td></tr>`;
        }).join('')}</tbody>
      </table>
    </div>`;
  }

  function survivalSlot(spec) {
    return slot(
      'survival',
      survivalOption(spec),
      survivalFallback(spec),
      spec.label,
      330,
    );
  }

  function heatmapOption(spec) {
    const chartCore = core();
    if (!chartCore) return null;
    const bins = spec.bins || [];
    const matrix = spec.matrix || [];
    if (!bins.length || !matrix.length) return null;
    const values = matrix.flatMap((row, rowIndex) => (row.cells || []).map((cell, colIndex) => {
      const count = Number(cell.count) || 0;
      const pct = Number(cell.pct) || 0;
      const tone = colIndex > rowIndex ? 'up' : colIndex < rowIndex ? 'down' : 'same';
      return {
        value: [colIndex, rowIndex, spec.mode === 'count' ? count : pct, count, pct],
        itemStyle: { color: fill(tone, cell.intensity) },
      };
    }));
    const dense = bins.length > 12;
    const option = Object.assign(chartCore.baseOption(spec.description), {
      grid: chartCore.grid({
        left: dense ? 78 : 72,
        right: dense ? 64 : 24,
        top: 28,
        bottom: dense ? 72 : 52,
      }),
      tooltip: chartCore.tooltip(params => {
        const value = params && params.value || [];
        const source = bins[value[1]] || value[1];
        const target = bins[value[0]] || value[0];
        return `${spec.yLabel} ${source} → ${spec.xLabel} ${target}\nN ${value[3]} · ${chartCore.formatNumber(value[4], 1)}%`;
      }, 'item'),
      xAxis: chartCore.axis({
        type: 'category',
        name: spec.xLabel,
        data: bins,
        boundaryGap: true,
        splitLine: false,
      }),
      yAxis: chartCore.axis({
        type: 'category',
        name: spec.yLabel,
        data: bins,
        boundaryGap: true,
        splitLine: false,
        inverse: true,
      }),
      series: [{
        name: spec.valueLabel,
        type: 'heatmap',
        data: values,
        label: {
          show: !dense,
          color: '#17202a',
          fontFamily: 'IBM Plex Mono, ui-monospace, monospace',
          fontSize: 10,
          formatter: params => spec.mode === 'count'
            ? chartCore.formatNumber(params.value[3], 0)
            : `${chartCore.formatNumber(params.value[4], 1)}%`,
        },
        itemStyle: {
          borderColor: 'rgba(255,255,255,.78)',
          borderWidth: 2,
          borderRadius: 3,
        },
        emphasis: {
          itemStyle: { borderColor: '#17202a', borderWidth: 1.5 },
        },
      }],
    });
    if (dense) {
      option.dataZoom = [
        { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
        {
          type: 'slider',
          xAxisIndex: 0,
          filterMode: 'none',
          height: 16,
          bottom: 8,
          showDetail: false,
        },
        { type: 'inside', yAxisIndex: 0, filterMode: 'none' },
        {
          type: 'slider',
          yAxisIndex: 0,
          filterMode: 'none',
          width: 16,
          right: 8,
          showDetail: false,
        },
      ];
    }
    return option;
  }

  function heatmapFallback(spec) {
    return `<div class="table-wrap table-scroll cohort-chart-fallback">
      <table class="eu-table">
        <thead><tr><th>${esc(spec.yLabel)} \\ ${esc(spec.xLabel)}</th>${(spec.bins || []).map(bin => `<th class="num">${esc(bin)}</th>`).join('')}</tr></thead>
        <tbody>${(spec.matrix || []).map(row => `<tr><td class="key">${esc(row.label)}</td>${(row.cells || []).map(cell => `<td class="num">${spec.mode === 'count' ? esc(cell.count) : `${esc(cell.pct)}%`}</td>`).join('')}</tr>`).join('')}</tbody>
      </table>
    </div>`;
  }

  function heatmapSlot(spec) {
    const dense = (spec.bins || []).length > 12;
    return `${slot(
      'heatmap',
      heatmapOption(spec),
      heatmapFallback(spec),
      spec.label,
      dense ? 560 : 410,
    )}
    <div class="cohort-heat-legend">
      <span><i class="same"></i>${esc(spec.sameLabel)}</span>
      <span><i class="up"></i>${esc(spec.upLabel)}</span>
      <span><i class="down"></i>${esc(spec.downLabel)}</span>
    </div>`;
  }

  function mount(root) {
    const chartCore = core();
    if (!chartCore || typeof chartCore.mount !== 'function' || !root) return 0;
    let mounted = 0;
    root.querySelectorAll('[data-cohort-echart]').forEach(element => {
      const id = element.getAttribute('data-cohort-echart');
      const option = pending.get(id);
      if (!option) return;
      const shell = typeof element.closest === 'function' ? element.closest('.cohort-echart-shell') : null;
      const fallback = shell && typeof shell.querySelector === 'function'
        ? shell.querySelector('[data-cohort-echart-fallback]')
        : null;
      if (chartCore.mount(element, option, {
        owner: 'cohort',
        fallback,
        onError(error) {
          if (window.console && typeof window.console.warn === 'function') {
            window.console.warn('Cohort chart fell back to its semantic table.', error);
          }
        },
      })) mounted += 1;
      pending.delete(id);
    });
    return mounted;
  }

  function dispose() {
    const chartCore = core();
    if (chartCore && typeof chartCore.dispose === 'function') chartCore.dispose('cohort');
    pending.clear();
  }

  if (typeof window.addEventListener === 'function') {
    window.addEventListener('hashchange', () => {
      const route = String((window.location && window.location.hash) || '').replace(/^#/, '');
      if (route !== 'cohort') dispose();
    });
  }

  window.EU_COHORT_CHARTS = {
    begin,
    dispose,
    heatmapOption,
    heatmapSlot,
    mount,
    survivalOption,
    survivalSlot,
  };
})();
