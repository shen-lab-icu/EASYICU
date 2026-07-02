(function () {
  const COLORS = ['var(--accent)', '#2563eb', '#0f766e', '#b45309', '#7c3aed', '#be123c'];
  const VITAL_THRESHOLDS = {
    hr: { low: 50, high: 120 },
    map: { low: 60, high: 120 },
    sbp: { low: 90, high: 160 },
    dbp: { low: 50, high: 100 },
    spo2: { low: 90 },
    resp: { low: 10, high: 30 },
    temp: { low: 36, high: 38.5 },
  };
  const MAX_SIGNALS_PER_MODULE = 8;

  function signalKey(sig) {
    return String((sig && (sig.feature || sig.key || sig.name)) || '').toLowerCase();
  }

  function numericValues(sig) {
    return ((sig && sig.values) || []).map(Number).filter(Number.isFinite);
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
    const times = withTimes ? withTimes.times : [];
    const first = times.length ? formatTimeLabel(times[0], helpers) : '';
    const last = times.length ? formatTimeLabel(times[Math.min(times.length - 1, numericValues(withTimes).length - 1)], helpers) : '';
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
    const values = numericValues(sig);
    const med = median(values);
    const bounds = VITAL_THRESHOLDS[signalKey(sig)] || {};
    const thresholds = [];
    if (Number.isFinite(med)) {
      thresholds.push({
        value: med,
        label: hT(helpers, 'Median', '中位数'),
        color: '#94a3b8',
        dash: '4 4',
      });
    }
    if (Number.isFinite(bounds.low)) {
      thresholds.push({ value: bounds.low, label: hT(helpers, 'Low threshold', '低阈值'), color: '#ef4444' });
    }
    if (Number.isFinite(bounds.high)) {
      thresholds.push({ value: bounds.high, label: hT(helpers, 'High threshold', '高阈值'), color: '#f97316' });
    }
    return thresholds;
  }

  function signalCell(sig, index, helpers, xLabels) {
    const values = numericValues(sig);
    if (values.length < 2 || !helpers || !helpers.axisSpark) return '';
    const label = signalLabel(sig, helpers);
    const color = COLORS[index % COLORS.length];
    const unit = sig && sig.unit ? sig.unit : '';
    return `
      <div class="pt-vsm-cell" data-patient-series-feature="${hEsc(helpers, signalKey(sig))}">
        <div class="pt-vsm-title">${hEsc(helpers, label)}${unit ? ` <span class="mono">${hEsc(helpers, unit)}</span>` : ''}</div>
        ${helpers.axisSpark(values, 360, 132, color, { unit, label, thresholds: thresholdsFor(sig, helpers), xLabels })}
      </div>`;
  }

  function renderModulePanels(lanes, helpers = {}) {
    const usable = (lanes || [])
      .map(lane => {
        const signals = ((lane && lane.signals) || []).filter(sig => numericValues(sig).length >= 2);
        return { lane, signals };
      })
      .filter(item => item.signals.length);
    if (!usable.length) return '';

    const cards = usable.map((item, moduleIndex) => {
      const visibleSignals = item.signals.slice(0, MAX_SIGNALS_PER_MODULE);
      const xLabels = signalTimeLabels(visibleSignals, helpers, helpers.demoHours ? helpers.demoHours() : null);
      const hidden = Math.max(0, item.signals.length - visibleSignals.length);
      const declared = Number(item.lane && item.lane.signal_count);
      const totalSignals = Number.isFinite(declared) && declared > 0 ? declared : item.signals.length;
      return `
        <section class="pt-module-card" data-patient-series-module="${hEsc(helpers, (item.lane && item.lane.lane) || moduleIndex)}">
          <div class="pt-module-head">
            <div>
              <div class="pt-module-title">${hEsc(helpers, moduleLabel(item.lane, helpers))}</div>
              <div class="pt-module-meta mono">${hFmtInt(helpers, visibleSignals.length)} / ${hFmtInt(helpers, totalSignals)} ${hT(helpers, 'signals shown', '个信号展示')}</div>
            </div>
            ${hidden ? `<span class="chip">${hT(helpers, 'plus', '另有')} ${hFmtInt(helpers, hidden)}</span>` : ''}
          </div>
          <div class="pt-module-grid">
            ${visibleSignals.map((sig, i) => signalCell(sig, i, helpers, xLabels)).join('')}
          </div>
        </section>`;
    }).join('');

    return `
      <div class="sec-stack mt-16"><div class="lbl">${hT(helpers, 'Time series by module', '按模块分组的时间序列')}</div></div>
      <div class="pt-vsm-legend">
        <span><i style="background:var(--accent);"></i>${hT(helpers, 'Patient trajectory', '患者轨迹')}</span>
        <span><i class="dash" style="color:#94a3b8;"></i>${hT(helpers, 'Median', '中位数')}</span>
        <span><i class="dash" style="color:#ef4444;"></i>${hT(helpers, 'Low threshold', '低阈值')}</span>
        <span><i class="dash" style="color:#f97316;"></i>${hT(helpers, 'High threshold', '高阈值')}</span>
      </div>
      <div class="pt-module-stack">${cards}</div>
      <div class="pt-vsm-axis">${hT(helpers, 'Time since ICU admission (hours)', 'ICU 入科后时间（小时）')}</div>`;
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
      ['lanes', hT(helpers, 'Clinical lanes', '临床泳道'), hT(helpers, 'Vitals, labs, scores, support', '生命体征、实验室、评分、治疗支持')],
      ['single', hT(helpers, 'Single patient', '单患者'), hT(helpers, 'One entity, feature-by-feature', '一个实体逐特征审阅')],
      ['compare', hT(helpers, 'Multi-patient comparison', '多患者对比'), hT(helpers, 'Same feature across the sample', '同一特征跨样本对比')],
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
    return `
      <div class="pt-series-panel" data-patient-series-panel="lanes">
        <div class="pt-series-panel-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Clinical lanes', '临床泳道'))}</div>
            <h2>${hEsc(helpers, hT(helpers, 'Signals grouped by clinical meaning', '按临床语义分组的时间序列'))}</h2>
            <p>${hEsc(helpers, hT(helpers, 'This restores the old Patient Review mental model: each lane is a clinical domain, and each chart follows the selected entity over time.', '这里恢复旧版患者审阅的逻辑：每条泳道对应一个临床域，每张图追踪当前实体的时间变化。'))}</p>
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
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Single patient trajectory', '单患者轨迹'))}</div>
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
    return ((trace && trace.values) || []).map(Number).filter(Number.isFinite);
  }

  function multiTraceChart(traces, comparison, helpers) {
    const series = (traces || []).map((trace, index) => ({
      trace,
      values: traceValues(trace),
      color: COLORS[index % COLORS.length],
    })).filter(row => row.values.length >= 2);
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
    return `
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
          <text x="${left}" y="${height - 10}" class="axis">${hEsc(helpers, timeLabels ? timeLabels[0] : '0h')}</text>
          <text x="${width - right - 42}" y="${height - 10}" class="axis">${hEsc(helpers, timeLabels ? timeLabels[1] : hT(helpers, 'last point', '末点'))}</text>
        </svg>
      </div>`;
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
              <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Multi-patient comparison', '多患者对比'))}</div>
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
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Multi-patient comparison', '多患者对比'))}</div>
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
            <p>${hEsc(helpers, hT(helpers, 'Choose the old review modes without leaving the native Patient Review page.', '在原生患者明细页内切换旧版审阅模式。'))}</p>
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
    numericValues,
  };
})();
