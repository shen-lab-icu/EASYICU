(function () {
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
    return helpers && helpers.fmtNum ? helpers.fmtNum(value, digits) : String(value == null ? '' : value);
  }

  function hFmtPct(helpers, value) {
    return helpers && helpers.fmtPct ? helpers.fmtPct(value) : `${hFmtNum(helpers, value, 1)}%`;
  }

  function hIcon(helpers, name, size) {
    return helpers && helpers.icon ? helpers.icon(name, size) : '';
  }

  function overviewText(value, helpers) {
    const map = {
      'Age / sex': hT(helpers, 'Age / sex', '年龄 / 性别'),
      'SOFA-2 max': hT(helpers, 'SOFA-2 max', 'SOFA-2 最大值'),
      'Sepsis-3': hT(helpers, 'Sepsis-3', 'Sepsis-3'),
      'Outcome': hT(helpers, 'Outcome', '结局'),
      'ICU LOS': hT(helpers, 'ICU LOS', 'ICU 住院天数'),
      'Patient Summary': hT(helpers, 'Patient Summary', '患者摘要'),
      'Available data': hT(helpers, 'Available data', '可用数据'),
      'Case review': hT(helpers, 'Case review', '病例审阅'),
      'Clinical review': hT(helpers, 'Clinical review', '临床审阅'),
    };
    return map[String(value || '')] || value;
  }

  function toneClass(tone) {
    if (tone === 'bad') return 'bad';
    if (tone === 'warn') return 'warn';
    if (tone === 'ok') return 'ok';
    return 'accent';
  }

  function signalLabel(card) {
    return (card && (card.label || card.name || card.feature || card.key)) || 'signal';
  }

  function signalValue(card, helpers) {
    if (!card || card.current == null) return '—';
    const unit = card.unit ? ` ${card.unit}` : '';
    const numeric = Number(card.current);
    return Number.isFinite(numeric) ? `${hFmtNum(helpers, numeric, 1)}${unit}` : `${card.current}${unit}`;
  }

  function deltaNote(card, helpers) {
    const n = Number(card && card.delta);
    if (!Number.isFinite(n)) return hT(helpers, 'latest', '最新');
    if (Math.abs(n) < 1e-9) return hT(helpers, 'stable', '稳定');
    return `Δ ${n > 0 ? '+' : ''}${hFmtNum(helpers, n, 1)}`;
  }

  function cardFeature(card) {
    return (card && (card.feature || card.key || card.name || card.label)) || '';
  }

  function asNumber(value) {
    if (value == null || value === '' || typeof value === 'boolean') return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  }

  function clampPct(value) {
    const numeric = asNumber(value);
    if (numeric == null) return null;
    return Math.max(0, Math.min(100, numeric));
  }

  function moduleDisplayLabel(item, helpers) {
    if (!item) return hT(helpers, 'Module', '模块');
    if (helpers && helpers.moduleLabel) return helpers.moduleLabel(item);
    return item.label || item.module || hT(helpers, 'Module', '模块');
  }

  function moduleAuditRows(drill) {
    const rows = new Map();
    const profiles = Array.isArray(drill && drill.module_profiles) ? drill.module_profiles : [];
    const quality = Array.isArray(drill && drill.quality) ? drill.quality : [];
    profiles.forEach(item => {
      const key = String((item && item.module) || '');
      if (key) rows.set(key, Object.assign({}, item));
    });
    quality.forEach(item => {
      const key = String((item && item.module) || '');
      if (!key) return;
      rows.set(key, Object.assign({}, rows.get(key) || {}, item));
    });
    return Array.from(rows.values()).filter(item => item && (item.module || item.label));
  }

  function isPresenceRateRow(item) {
    const kind = String((item && item.metric_kind) || '').toLowerCase();
    const status = String((item && item.quality_status) || '').toLowerCase();
    const module = String((item && item.module) || '').trim().toLowerCase();
    const presenceModules = new Set([
      'sepsis3_sofa1',
      'sepsis3_sofa2',
      'vasopressor',
      'vasopressors',
      'ventilation',
      'ventilator',
    ]);
    return (kind && kind !== 'coverage') || status === 'neutral' || presenceModules.has(module);
  }

  function cohortDenominator(drill, rows) {
    const loaded = drill && drill.data_tables && drill.data_tables.loaded_summary;
    const summary = (drill && drill.summary) || {};
    const boundedReview = summary.review_scope === 'browser_bounded_entity_sample';
    const candidates = [
      boundedReview && summary.review_entities,
      summary.entities,
      summary.stays,
      loaded && loaded.entities,
      loaded && loaded.stays,
      ...rows.map(item => item && item.entities),
    ];
    for (const value of candidates) {
      const numeric = asNumber(value);
      if (numeric && numeric > 0) return numeric;
    }
    return null;
  }

  function median(values) {
    const nums = values.map(asNumber).filter(value => value != null).sort((a, b) => a - b);
    if (!nums.length) return null;
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  }

  function missingSeverity(missingPct) {
    if (missingPct == null) return 'neutral';
    if (missingPct >= 30) return 'bad';
    if (missingPct >= 10) return 'warn';
    return 'ok';
  }

  function quantile(values, probability) {
    const sorted = values.map(asNumber).filter(value => value != null).sort((a, b) => a - b);
    if (!sorted.length) return null;
    const position = (sorted.length - 1) * probability;
    const lower = Math.floor(position);
    const remainder = position - lower;
    return sorted[lower + 1] == null ? sorted[lower] : sorted[lower] + remainder * (sorted[lower + 1] - sorted[lower]);
  }

  function renderDistributionLatest(drill, helpers) {
    const source = drill && drill.trajectory_review && drill.trajectory_review.single_entity;
    const rows = (Array.isArray(source && source.signals) ? source.signals : [])
      .map(signal => {
        const values = (Array.isArray(signal && signal.values) ? signal.values : []).map(asNumber).filter(value => value != null);
        if (values.length < 4) return null;
        const minimum = Math.min(...values);
        const maximum = Math.max(...values);
        const span = maximum - minimum || 1;
        const pct = value => Math.max(0, Math.min(100, (value - minimum) / span * 100));
        return {
          signal,
          values,
          minimum,
          maximum,
          q1: quantile(values, .25),
          median: quantile(values, .5),
          q3: quantile(values, .75),
          latest: values[values.length - 1],
          pct,
        };
      })
      .filter(Boolean)
      .slice(0, 6);
    if (!rows.length) return '';
    return `<section class="pt-distribution-latest mt-16" data-patient-distribution-latest>
      <div class="pt-distribution-head">
        <div><div class="eyebrow">${hEsc(helpers, hT(helpers, 'Distribution · latest value', '分布—末值图'))}</div><h2>${hEsc(helpers, hT(helpers, 'Within-patient value distributions', '单患者多变量分布与末值'))}</h2><p>${hEsc(helpers, hT(helpers, 'Each row summarizes the bounded observed values for one signal. The diamond is the latest recorded value; this view intentionally contains no second time trajectory.', '每行汇总一个信号的有界观测值；菱形标记最后一次记录值。本图刻意不重复时间轨迹。'))}</p></div>
        <span class="pill ok">${hEsc(helpers, (source && (source.selected_label || source.selected_ref)) || hT(helpers, 'selected entity', '已选实体'))}</span>
      </div>
      <div class="pt-distribution-rows">
        ${rows.map(row => {
          const label = row.signal.name || row.signal.label || row.signal.feature;
          const unit = row.signal.unit || '';
          return `<div class="pt-distribution-row" data-patient-distribution-feature="${hEsc(helpers, row.signal.feature || label)}">
            <div class="pt-distribution-name"><b>${hEsc(helpers, label)}</b><span>${hEsc(helpers, row.signal.feature || '')} · n=${hFmtInt(helpers, row.values.length)}</span></div>
            <div class="pt-boxplot" aria-label="${hEsc(helpers, `${label}: ${row.minimum} to ${row.maximum}`)}"><i class="whisker"></i><i class="box" style="left:${row.pct(row.q1).toFixed(2)}%;width:${Math.max(1.5, row.pct(row.q3) - row.pct(row.q1)).toFixed(2)}%"></i><i class="median" style="left:${row.pct(row.median).toFixed(2)}%"></i><i class="latest" style="left:${row.pct(row.latest).toFixed(2)}%"></i></div>
            <div class="pt-distribution-values"><span>${hFmtNum(helpers, row.minimum, 1)}</span><b>${hFmtNum(helpers, row.latest, 1)}${unit ? ` ${hEsc(helpers, unit)}` : ''}</b><span>${hFmtNum(helpers, row.maximum, 1)}</span></div>
          </div>`;
        }).join('')}
      </div>
      <div class="pt-distribution-legend"><span><i class="box"></i>${hEsc(helpers, hT(helpers, 'IQR', '四分位距'))}</span><span><i class="median"></i>${hEsc(helpers, hT(helpers, 'median', '中位数'))}</span><span><i class="latest"></i>${hEsc(helpers, hT(helpers, 'latest', '末值'))}</span></div>
    </section>`;
  }

  function renderCaseReview(selected, summaryCards, sections, drill, helpers) {
    const totalSignals = (sections || []).reduce((acc, section) => acc + Number(section.available_count || 0), 0);
    const activeSections = (sections || []).filter(section => Number(section.available_count || 0) > 0);
    return `
      <section class="pt-overview-workbench mt-16" data-patient-overview-workbench>
        <div class="pt-overview-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, overviewText('Case review', helpers))}</div>
            <h2>${hEsc(helpers, hT(helpers, 'Clinical case overview', '病例画像工作台'))} · ${hEsc(helpers, (selected && selected.label) || hT(helpers, 'Selected entity', '已选实体'))}</h2>
            <p>${hEsc(helpers, hT(helpers, 'Snapshot-level clinical interpretation for one pseudonymous entity. Use Time Series only when you need trajectories.', '面向单个去标识实体的临床画像；需要看轨迹时再进入时间序列页。'))}</p>
          </div>
          <span class="pill ${drill && drill.demo ? 'demo' : 'ok'}">${drill && drill.demo ? hT(helpers, 'demo workspace', '演示工作区') : hT(helpers, 'local export', '本地导出')}</span>
        </div>
        <div class="pt-overview-stat-grid">
          ${(summaryCards || []).map(card => `
            <div class="pt-overview-stat ${toneClass(card && card.tone)}">
              <span>${hEsc(helpers, overviewText(card && card.label, helpers))}</span>
              <b>${hEsc(helpers, card && card.value == null ? '—' : card.value)}</b>
            </div>`).join('')}
        </div>
        <div class="pt-overview-availability">
          <div>
            <span>${hEsc(helpers, hT(helpers, 'Signals available for this entity', '该实体可用信号'))}</span>
            <b>${hFmtInt(helpers, totalSignals)}</b>
          </div>
          <div>
            <span>${hEsc(helpers, hT(helpers, 'Clinical modules represented', '有值的临床模块'))}</span>
            <b>${hFmtInt(helpers, activeSections.length)}</b>
          </div>
          <div>
            <span>${hEsc(helpers, hT(helpers, 'Source-row policy', '源行策略'))}</span>
            <b>${hEsc(helpers, hT(helpers, 'bounded summary only', '仅有界摘要'))}</b>
          </div>
        </div>
      </section>`;
  }

  function renderMissingnessCoverage(drill, helpers) {
    const quality = drill && drill.quality_metrics || {};
    const summary = quality.summary || {};
    const eventPattern = /(outcome|death|sepsis|sep3|vasopressor|ventilat|infection|culture)/i;
    const rows = (Array.isArray(quality.features) ? quality.features : [])
      .filter(row => row && !eventPattern.test(`${row.feature || ''} ${row.module || ''}`))
      .map(row => ({ ...row, missing: clampPct(row.missing_pct), records: Math.max(1, Number(row.records || 0)) }))
      .filter(row => row.missing != null && Number.isFinite(row.records));
    if (!rows.length) return '';
    const maxLog = Math.max(1, ...rows.map(row => Math.log10(row.records)));
    const labelled = rows.slice().sort((a, b) => b.missing - a.missing).slice(0, 8);
    const width = 860; const height = 390; const left = 70; const right = 28; const top = 28; const bottom = 56;
    const x = value => left + value / 100 * (width - left - right);
    const y = records => top + (1 - Math.log10(records) / maxLog) * (height - top - bottom);
    return `<section class="pt-missingness-workbench mt-16" data-patient-missing-record-scatter>
      <div class="pt-missingness-head"><div><div class="eyebrow">${hEsc(helpers, hT(helpers, 'Missingness · record volume', '缺失—记录散点'))}</div><h2>${hEsc(helpers, hT(helpers, 'Feature-level missingness and observation volume', '特征层面的缺失率与记录量'))}</h2><p>${hEsc(helpers, hT(helpers, 'Each point is a measured feature from the active export. Event and exposure prevalence are excluded because absence of an event is not missing data.', '每个点都是当前导出中的实测特征；事件与暴露发生率被排除，因为“事件未发生”不等于数据缺失。'))}</p></div><span class="pill ok">${hFmtInt(helpers, rows.length)} ${hEsc(helpers, hT(helpers, 'features', '个特征'))}</span></div>
      <div class="pt-missing-scatter"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${hEsc(helpers, hT(helpers, 'Missingness by record count', '缺失率与记录量散点图'))}">
        ${[0,25,50,75,100].map(tick => `<line x1="${x(tick)}" x2="${x(tick)}" y1="${top}" y2="${height-bottom}" class="grid"></line><text x="${x(tick)}" y="${height-25}" class="axis" text-anchor="middle">${tick}%</text>`).join('')}
        ${Array.from({length:Math.floor(maxLog)+1},(_,tick)=>`<line x1="${left}" x2="${width-right}" y1="${y(10 ** tick)}" y2="${y(10 ** tick)}" class="grid"></line><text x="${left-12}" y="${y(10 ** tick)+4}" class="axis" text-anchor="end">10${tick ? `<tspan baseline-shift="super" font-size="12">${tick}</tspan>` : ''}</text>`).join('')}
        ${rows.map(row => { const show = labelled.includes(row); const label = row.name || row.feature; return `<g><circle cx="${x(row.missing).toFixed(1)}" cy="${y(row.records).toFixed(1)}" r="${show ? 6 : 4}" class="${row.missing >= 30 ? 'bad' : row.missing >= 10 ? 'warn' : 'ok'}"><title>${hEsc(helpers, `${label}: ${row.missing}% missing, ${row.records} records`)}</title></circle>${show ? `<text x="${(x(row.missing)+8).toFixed(1)}" y="${(y(row.records)-7).toFixed(1)}" class="point-label">${hEsc(helpers, label)}</text>` : ''}</g>`; }).join('')}
        <text x="${(left+width-right)/2}" y="${height-5}" class="axis-title" text-anchor="middle">${hEsc(helpers, hT(helpers, 'Missing values (%)', '缺失率 (%)'))}</text><text x="16" y="${(top+height-bottom)/2}" class="axis-title" text-anchor="middle" transform="rotate(-90 16 ${(top+height-bottom)/2})">${hEsc(helpers, hT(helpers, 'Records (log scale)', '记录数（对数尺度）'))}</text>
      </svg></div>
      <div class="pt-cleaning-receipt" data-patient-cleaning-receipt><div><span>${hEsc(helpers, hT(helpers, 'Cleaning receipt', '清洗回执'))}</span><b>${hFmtInt(helpers, summary.total_records || 0)} ${hEsc(helpers, hT(helpers, 'records audited', '条记录已审计'))}</b></div><div><span>${hEsc(helpers, hT(helpers, 'Post-cleaning duplicate-time rate', '清洗后重复时间戳率'))}</span><b>${hFmtPct(helpers, summary.weighted_duplicate_time_pct || 0)}</b></div><div><span>${hEsc(helpers, hT(helpers, 'Post-rule out-of-range rate', '规则处理后越界率'))}</span><b>${hFmtPct(helpers, summary.weighted_out_of_physio_pct || 0)}</b></div></div>
      <p class="pt-missingness-note">${hEsc(helpers, hT(helpers, 'The receipt verifies the configured post-cleaning state; it does not re-label resolved duplicates or out-of-range source values as new problems.', '回执核验的是清洗后的状态，不会把已经解决的重复时间戳或越界源值重新包装成“新问题”。'))}</p>
    </section>`;
  }

  function renderSignalCard(card, helpers) {
    return `
      <div class="pt-overview-signal ${toneClass(card && card.tone)}">
        <div class="pt-overview-signal-main">
          <span>${hEsc(helpers, signalLabel(card))}</span>
          <b>${hEsc(helpers, signalValue(card, helpers))}</b>
        </div>
        <div class="pt-overview-signal-meta">
          <span class="mono">${hEsc(helpers, cardFeature(card))}</span>
          <em>${hEsc(helpers, deltaNote(card, helpers))}</em>
        </div>
      </div>`;
  }

  function renderCategorySummary(sections, helpers) {
    const usable = (sections || []).filter(section => section && (section.cards || []).length);
    if (!usable.length) {
      return `
        <div class="empty mt-16">
          <div class="glyph">${hIcon(helpers, 'grid', 22)}</div>
          <div class="t">${hEsc(helpers, hT(helpers, 'No module signals available', '暂无模块信号'))}</div>
          <div class="d">${hEsc(helpers, hT(helpers, 'The active export has no bounded patient-level values for this entity.', '当前导出没有这个实体的有界患者层面数值。'))}</div>
        </div>`;
    }
    return `
      <section class="pt-category-summary mt-18" data-patient-category-review>
        <div class="sec-stack">
          <div class="lbl">${hEsc(helpers, overviewText('Clinical review', helpers))}</div>
          <h2>${hEsc(helpers, hT(helpers, 'Category signal summary', '分类信号摘要'))}</h2>
        </div>
        <p class="pt-category-intro">${hEsc(helpers, hT(helpers, 'This is not another trajectory view: each module shows the selected entity’s latest or ever-observed clinical state. Open Time Series for curves and Data Tables for exact cells.', '这里不是另一组曲线：每个模块只展示该实体的最新值或曾出现状态。曲线去时间序列页，精确格子去数据表页。'))}</p>
        <div class="pt-category-list">
          ${usable.map(section => {
            const cards = (section.cards || []).slice(0, 10);
            const riskCount = cards.filter(card => ['bad', 'warn'].includes(card && card.tone)).length;
            return `
              <article class="pt-category-section" data-patient-category="${hEsc(helpers, section.id || '')}">
                <div class="pt-category-head">
                  <div>
                    <h3>${hEsc(helpers, overviewText(section.title || section.id, helpers))}</h3>
                    <p>${hFmtInt(helpers, section.available_count || (section.cards || []).length)} ${hEsc(helpers, hT(helpers, 'signals available for this entity', '个信号可用于该实体'))}</p>
                  </div>
                  <span class="pill ${riskCount ? 'warn' : 'ok'}">${riskCount ? `${hFmtInt(helpers, riskCount)} ${hT(helpers, 'watch', '需关注')}` : hT(helpers, 'stable', '稳定')}</span>
                </div>
                <div class="pt-category-values">${cards.map(card => renderSignalCard(card, helpers)).join('')}</div>
              </article>`;
          }).join('')}
        </div>
      </section>`;
  }

  function renderModuleLedger(drill, helpers) {
    const modules = Array.isArray(drill && drill.module_profiles) && drill.module_profiles.length
      ? drill.module_profiles
      : (Array.isArray(drill && drill.quality) ? drill.quality : []);
    const rows = modules.filter(item => item && (item.module || item.label)).slice(0, 32);
    if (!rows.length) return '';
    const maxRows = Math.max(1, ...rows.map(item => Number(item.rows || 0)));
    const totalFeatures = rows.reduce((acc, item) => acc + Number(item.review_features != null ? item.review_features : (item.feature_count || 0)), 0);
    const loaded = (drill && drill.data_tables && drill.data_tables.loaded_summary) || {};
    const observed = loaded.observed_features != null ? loaded.observed_features : rows.reduce((acc, item) => acc + Number(item.observed_features || 0), 0);
    return `
      <section class="pt-module-ledger mt-16" data-patient-overview-module-ledger>
        <div class="pt-module-ledger-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Module map', '模块图谱'))}</div>
            <h2>${hEsc(helpers, hT(helpers, 'Export module availability', '导出模块可用性'))}</h2>
            <p>${hFmtInt(helpers, rows.length)} ${hEsc(helpers, hT(helpers, 'modules', '个模块'))} · ${hFmtInt(helpers, totalFeatures)} ${hEsc(helpers, hT(helpers, 'review features', '个审阅特征'))} · ${hFmtInt(helpers, observed)} ${hEsc(helpers, hT(helpers, 'observed in selected browser sample', '个在当前浏览样本中可观测'))}</p>
          </div>
          <span class="pill ok">${hEsc(helpers, hT(helpers, 'active export', '当前导出'))}</span>
        </div>
        <div class="pt-module-ledger-grid">
          ${rows.map(item => {
            const label = (helpers && helpers.moduleLabel ? helpers.moduleLabel(item) : (item.label || item.module)) || hT(helpers, 'Module', '模块');
            const featureCount = item.review_features != null ? item.review_features : item.feature_count;
            const rowCount = Number(item.rows || 0);
            const entityCount = Number(item.entities);
            const coverage = Number(item.coverage_pct);
            const hasCoverage = Number.isFinite(coverage) && Number.isFinite(entityCount) && entityCount > 0;
            const width = hasCoverage ? Math.max(2, Math.min(100, coverage)) : Math.max(2, Math.min(100, (rowCount / maxRows) * 100));
            const tone = hasCoverage ? (coverage >= 80 ? 'ok' : coverage > 0 ? 'warn' : 'accent') : (rowCount ? 'accent' : 'neutral');
            const subtitle = hasCoverage ? hFmtPct(helpers, coverage) : `${hFmtInt(helpers, rowCount)} ${hT(helpers, 'rows', '行')}`;
            return `
              <div class="pt-module-ledger-card ${tone}" data-patient-overview-module-card="${hEsc(helpers, item.module || label)}">
                <div class="pt-module-ledger-card-top">
                  <b>${hEsc(helpers, label)}</b>
                  <span>${hEsc(helpers, subtitle)}</span>
                </div>
                <div class="pt-module-ledger-card-meta">
                  <span>${hFmtInt(helpers, featureCount)} ${hEsc(helpers, hT(helpers, 'features', '特征'))}</span>
                  <span>${Number.isFinite(entityCount) && entityCount > 0 ? `${hFmtInt(helpers, entityCount)} ${hT(helpers, 'entities', '实体')}` : hEsc(helpers, item.shape || item.status || item.quality_status || '')}</span>
                </div>
                <div class="pt-module-ledger-bar"><i style="width:${width}%;"></i></div>
              </div>`;
          }).join('')}
        </div>
      </section>`;
  }

  function renderOverview(payload, helpers = {}) {
    const selected = (payload && payload.selected) || {};
    const summaryCards = (payload && payload.summaryCards) || [];
    const sections = (payload && payload.sections) || [];
    const drill = payload && payload.drill;
    return [
      renderCaseReview(selected, summaryCards, sections, drill, helpers),
      renderDistributionLatest(drill, helpers),
    ].join('');
  }

  function renderQualityAudit(payload, helpers = {}) {
    const drill = (payload && payload.drill) || payload;
    return renderMissingnessCoverage(drill, helpers);
  }

  window.EU_PATIENT_OVERVIEW = {
    renderOverview,
    renderQualityAudit,
  };
})();
