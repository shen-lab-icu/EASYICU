/* ============================================================
   screens-viz-cohort-charts.js — Cohort Statistics chart owner.
   Converts reviewed aggregate survival and SOFA transition data
   into the shared ECharts contract. Tables, controls, and route
   state remain owned by screens-viz.js.
   ============================================================ */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  let renderId = 0;
  let pending = new Map();

  function core() {
    return window.EU_ECHARTS || null;
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

  /* `belowHtml` renders under the plot on the rendered path only. The
     fail-closed fallback already carries the same numbers in its own table, so
     appending it there too would print them twice. */
  function slot(kind, option, fallbackHtml, label, height, belowHtml) {
    const chartCore = core();
    if (!chartCore || typeof chartCore.available !== 'function' || !chartCore.available() || !option) {
      return fallbackHtml || '';
    }
    const id = `coh-ec-${renderId}-${pending.size + 1}`;
    pending.set(id, option);
    return `<div class="cohort-echart-shell ${kind}" data-cohort-chart-kind="${kind}">
      <div class="cohort-echart ${kind}" style="--cohort-chart-height:${Number(height) || 320}px"
           data-cohort-echart="${id}" role="img" aria-label="${esc(label)}"></div>
      ${belowHtml || ''}
      <div data-cohort-echart-fallback hidden>${fallbackHtml || ''}</div>
    </div>`;
  }

  /* Shared by the chart grid and the number-at-risk table so the table columns
     stay under the axis ticks they label. Change one, both move.

     The left inset widens when a risk table is present: 58px is sized for the
     y-axis "100%" labels, but the table's row headers are group names in the
     same gutter, and at 58px "Sepsis-3 positive" ellipsised into the first
     count. Published KM figures widen the left margin for exactly this reason
     rather than letting the labels collide with the numbers. */
  const SURVIVAL_GRID = { left: 58, right: 22, top: 44, bottom: 48 };
  const SURVIVAL_RISK_GUTTER = 124;

  function survivalGrid(spec) {
    return hasRiskTable(spec)
      ? Object.assign({}, SURVIVAL_GRID, { left: SURVIVAL_RISK_GUTTER })
      : SURVIVAL_GRID;
  }

  function hasRiskTable(spec) {
    const risk = spec.atRisk || {};
    const times = (risk.times || []).map(finite).filter(time => time != null);
    const horizon = survivalHorizon(spec);
    return times.length >= 2
      && Boolean((risk.rows || []).length)
      && horizon != null
      && horizon > 0;
  }

  function survivalOption(spec) {
    const chartCore = core();
    if (!chartCore) return null;
    const groups = (spec.groups || []).filter(group => Array.isArray(group.points) && group.points.length);
    if (!groups.length) return null;
    return Object.assign(chartCore.baseOption(spec.description), {
      grid: chartCore.grid(survivalGrid(spec)),
      legend: chartCore.legend({ top: 4 }),
      tooltip: chartCore.tooltip(params => {
        const all = Array.isArray(params) ? params : [params];
        /* Censoring ticks share their group's series name, so without this
           filter the axis tooltip would list every group twice. */
        const rows = all.filter(row => !String(row.seriesId || '').startsWith('km-censor:'));
        if (!rows.length) return '';
        const first = Array.isArray(rows[0].value) ? rows[0].value[0] : rows[0].axisValue;
        const lines = [`${chartCore.formatNumber(first, 1)} ${spec.xLabel}`];
        rows.forEach(row => {
          const value = Array.isArray(row.value) ? row.value[1] : row.value;
          lines.push(`${row.seriesName}: ${chartCore.formatNumber(value, 1)}%`);
        });
        return lines.join('\n');
      }),
      /* An explicit max is what lets the number-at-risk table line up with the
         plot: without it ECharts picks its own "nice" upper bound and the
         table columns drift off the ticks they label. The horizon is also
         meaningful on its own — the payload declares a display window. */
      xAxis: chartCore.axis({
        type: 'value',
        name: spec.xLabel,
        min: 0,
        max: survivalHorizon(spec),
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
      series: groups.flatMap((group, index) => {
        const line = {
          name: group.label,
          id: `km-line:${index}`,
          type: 'line',
          data: group.points
            .map(point => [finite(point.time), finite(point.survival)])
            .filter(point => point[0] != null && point[1] != null),
          step: 'end',
          smooth: false,
          showSymbol: false,
          symbol: index % 2 ? 'emptyCircle' : 'circle',
          connectNulls: false,
          lineStyle: chartCore.lineStyle(index),
          emphasis: { focus: 'series', lineStyle: { width: 3.4 } },
        };
        const marks = (group.censorMarks || [])
          .map(mark => [finite(mark.time), finite(mark.survival)])
          .filter(mark => mark[0] != null && mark[1] != null);
        if (!marks.length) return [line];
        /* Standard KM censoring ticks. Same series `name` as the line so the
           legend stays one entry per group and toggling a group hides both.
           A flat stretch with no ticks is a genuinely different fact from a
           flat stretch full of them — that is why these are not decoration. */
        return [line, {
          name: group.label,
          id: `km-censor:${index}`,
          type: 'scatter',
          data: marks,
          symbol: 'rect',
          symbolSize: [1.6, 11],
          silent: true,
          legendHoverLink: false,
          z: 5,
        }];
      }),
    });
  }

  function survivalHorizon(spec) {
    const declared = finite(spec.horizon);
    if (declared != null && declared > 0) return declared;
    let max = 0;
    (spec.groups || []).forEach(group => {
      (group.points || []).forEach(point => {
        const time = finite(point.time);
        if (time != null && time > max) max = time;
      });
    });
    return max > 0 ? max : null;
  }

  /* Number at risk, drawn as a real table rather than extra chart series.

     A Kaplan-Meier curve without it is incomplete by every reporting standard:
     the reader cannot tell whether the tail is a finding or three remaining
     patients. The backend already computed it — cohort_review.py returns
     `number_at_risk` — it simply was not being rendered.

     The table inherits the chart's grid insets so its columns land on the axis
     ticks they label. It stays HTML (not a second ECharts grid) so the numbers
     are selectable and survive the fail-closed path where no renderer exists.

     The explicit ARIA roles are load-bearing, not belt-and-braces: positioning
     the columns requires `display:block` on the rows and cells, and CSS
     `display` overrides the implicit role of a table element — without these
     roles the markup is a real <table> that assistive tech reads as a stack of
     anonymous text, with the scope attributes and <caption> inert. */
  function riskTableHtml(spec) {
    if (!hasRiskTable(spec)) return '';
    const risk = spec.atRisk || {};
    const rows = risk.rows || [];
    const horizon = survivalHorizon(spec);
    const allTimes = (risk.times || []).map(finite).filter(time => time != null);
    const times = spacedRiskTimes(allTimes, horizon);
    const position = time => `${Math.min(100, Math.max(0, (time / horizon) * 100)).toFixed(3)}%`;
    const header = times
      .map(({ time }) => `<th role="columnheader" scope="col" style="left:${position(time)}">${esc(formatTime(time))}</th>`)
      .join('');
    const body = rows.map(row => `<tr role="row">
        <th role="rowheader" scope="row">${esc(row.label)}</th>
        ${times.map(({ time, index }) => {
          const value = (row.values || [])[index];
          return `<td role="cell" style="left:${position(time)}">${value == null ? '—' : esc(value)}</td>`;
        }).join('')}
      </tr>`).join('');
    return `<div class="km-risk" style="--km-inset-left:${survivalGrid(spec).left}px;--km-inset-right:${SURVIVAL_GRID.right}px">
      <table class="km-risk-table" role="table" aria-label="${esc(spec.atRiskLabel)}">
        <caption>${esc(spec.atRiskLabel)}</caption>
        <thead role="rowgroup"><tr role="row"><th role="columnheader" scope="col" class="km-risk-corner">${esc(spec.xLabel)}</th>${header}</tr></thead>
        <tbody role="rowgroup">${body}</tbody>
      </table>
    </div>`;
  }

  /* The backend's risk grid is clinical (0, 1, 3, 7, 14, 28) and the early
     entries matter for ICU mortality — but on a 28-day axis day 1 sits 3.6%
     along, close enough to day 0 that three-digit counts collide into
     "214201". Drop only what would overlap, always keeping the first and last
     column so the table still spans the plot. Positions are a fraction of the
     axis, so this holds at any container width. */
  const RISK_MIN_GAP = 0.07;

  /* Returns {time, index} so a dropped column cannot shift the counts: the
     values array is indexed by the backend's original grid. */
  function spacedRiskTimes(times, horizon) {
    const columns = times.map((time, index) => ({ time, index }));
    if (columns.length < 3 || !horizon) return columns;
    const last = columns[columns.length - 1];
    const kept = [columns[0]];
    columns.slice(1, -1).forEach(column => {
      const previous = kept[kept.length - 1];
      if ((column.time - previous.time) / horizon < RISK_MIN_GAP) return;
      if ((last.time - column.time) / horizon < RISK_MIN_GAP) return;
      kept.push(column);
    });
    kept.push(last);
    return kept;
  }

  function formatTime(time) {
    const chartCore = core();
    return chartCore ? chartCore.formatNumber(time, Number.isInteger(time) ? 0 : 1) : String(time);
  }

  /* The no-renderer path must state the same facts as the plot, so censored
     counts and the number at risk belong here too — not only in the chart. */
  function survivalFallback(spec) {
    const risk = spec.atRisk || {};
    const riskTimes = (risk.times || []).map(finite).filter(time => time != null);
    const riskByLabel = new Map((risk.rows || []).map(row => [row.label, row.values || []]));
    return `<div class="table-wrap table-scroll cohort-chart-fallback">
      <table class="eu-table">
        <thead><tr>
          <th>${esc(spec.groupLabel)}</th>
          <th class="num">n</th>
          <th class="num">${esc(spec.eventsLabel)}</th>
          <th class="num">${esc(spec.censoredLabel)}</th>
          <th class="num">${esc(spec.finalLabel)}</th>
          ${riskTimes.map(time => `<th class="num">${esc(spec.atRiskLabel)} @${esc(formatTime(time))}</th>`).join('')}
        </tr></thead>
        <tbody>${(spec.groups || []).map(group => {
          const points = group.points || [];
          const last = points.length ? points[points.length - 1] : null;
          const values = riskByLabel.get(group.label) || [];
          return `<tr>
            <td class="key">${esc(group.label)}</td>
            <td class="num">${esc(group.n)}</td>
            <td class="num">${esc(group.events)}</td>
            <td class="num">${group.censored == null ? '—' : esc(group.censored)}</td>
            <td class="num">${last ? `${esc(last.survival)}%` : '—'}</td>
            ${riskTimes.map((_time, index) => `<td class="num">${values[index] == null ? '—' : esc(values[index])}</td>`).join('')}
          </tr>`;
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
      riskTableHtml(spec),
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
    riskTableHtml,
    survivalFallback,
    survivalOption,
    survivalSlot,
  };
})();
