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
    const candidates = [
      drill && drill.summary && drill.summary.entities,
      drill && drill.summary && drill.summary.stays,
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
    const rows = moduleAuditRows(drill);
    if (!rows.length) return '';
    const denominator = cohortDenominator(drill, rows);
    const coverageRows = rows
      .map(item => {
        const coverage = clampPct(item.coverage_pct);
        if (coverage == null) return null;
        return Object.assign({}, item, {
          coverage_pct: coverage,
          missing_pct: Math.max(0, 100 - coverage),
          is_presence_rate: isPresenceRateRow(item),
        });
      })
      .filter(Boolean);
    const trueMissingRows = coverageRows
      .filter(item => !item.is_presence_rate)
      .sort((a, b) => (b.missing_pct || 0) - (a.missing_pct || 0));
    const presenceRows = coverageRows
      .filter(item => item.is_presence_rate)
      .sort((a, b) => (b.coverage_pct || 0) - (a.coverage_pct || 0));
    const medianCoverage = median(trueMissingRows.map(item => item.coverage_pct));
    const topMissing = trueMissingRows.length ? trueMissingRows[0].missing_pct : null;
    const watchCount = trueMissingRows.filter(item => (item.missing_pct || 0) >= 10).length;
    const visibleMissing = trueMissingRows.slice(0, 8);
    const visiblePresence = presenceRows.slice(0, 6);
    const denomText = denominator ? `${hFmtInt(helpers, denominator)} ${hT(helpers, 'entities', '实体')}` : hT(helpers, 'active export', '当前导出');
    return `
      <section class="pt-missingness-workbench mt-16" data-patient-overview-missingness>
        <div class="pt-missingness-head">
          <div>
            <div class="eyebrow">${hEsc(helpers, hT(helpers, 'Missingness audit', '缺失率审计'))}</div>
            <h2>${hEsc(helpers, hT(helpers, 'Missingness and coverage', '缺失率与覆盖率'))}</h2>
            <p>${hEsc(helpers, hT(helpers, 'Coverage is calculated at entity level for true data-availability modules. Event or exposure modules are labelled separately and are not treated as missingness.', '真正的数据可用性模块按实体层面计算覆盖率；事件或暴露模块单独标注，不当作缺失率。'))}</p>
          </div>
          <span class="pill ok">${hEsc(helpers, denomText)}</span>
        </div>
        <div class="pt-missingness-stat-grid">
          <div>
            <span>${hEsc(helpers, hT(helpers, 'Audited availability modules', '已审计可用性模块'))}</span>
            <b>${hFmtInt(helpers, trueMissingRows.length)}</b>
          </div>
          <div>
            <span>${hEsc(helpers, hT(helpers, 'Median coverage', '覆盖率中位数'))}</span>
            <b>${medianCoverage == null ? '—' : hFmtPct(helpers, medianCoverage)}</b>
          </div>
          <div>
            <span>${hEsc(helpers, hT(helpers, 'Highest missingness', '最高缺失率'))}</span>
            <b>${topMissing == null ? '—' : hFmtPct(helpers, topMissing)}</b>
          </div>
          <div>
            <span>${hEsc(helpers, hT(helpers, 'Watchlist modules', '需关注模块'))}</span>
            <b>${hFmtInt(helpers, watchCount)}</b>
          </div>
        </div>
        <div class="pt-missingness-body">
          <div class="pt-missingness-panel">
            <div class="pt-missingness-panel-head">
              <b>${hEsc(helpers, hT(helpers, 'Top module missingness', '模块缺失率排行'))}</b>
              <span>${hEsc(helpers, hT(helpers, 'coverage denominator: selected cohort entities', '分母：当前队列实体'))}</span>
            </div>
            <div class="pt-missingness-list">
              ${visibleMissing.length ? visibleMissing.map(item => {
                const label = moduleDisplayLabel(item, helpers);
                const coverage = clampPct(item.coverage_pct) || 0;
                const missingPct = Math.max(0, 100 - coverage);
                const tone = missingSeverity(missingPct);
                const entities = asNumber(item.entities);
                const width = Math.max(2, Math.min(100, missingPct));
                return `
                  <div class="pt-missingness-row ${tone}" data-patient-missingness-module="${hEsc(helpers, item.module || label)}">
                    <div class="pt-missingness-name">
                      <b>${hEsc(helpers, label)}</b>
                      <span class="mono">${hEsc(helpers, item.module || '')}</span>
                    </div>
                    <div class="pt-missingness-track" aria-hidden="true"><i style="width:${width}%;"></i></div>
                    <div class="pt-missingness-metric">
                      <span>${hEsc(helpers, hT(helpers, 'missing', '缺失'))}</span>
                      <b>${hFmtPct(helpers, missingPct)}</b>
                    </div>
                    <div class="pt-missingness-metric">
                      <span>${hEsc(helpers, hT(helpers, 'coverage', '覆盖'))}</span>
                      <b>${hFmtPct(helpers, coverage)}</b>
                    </div>
                    <div class="pt-missingness-meta">${Number.isFinite(entities) ? hFmtInt(helpers, entities) : '—'} / ${denominator ? hFmtInt(helpers, denominator) : '—'} · ${hFmtInt(helpers, item.feature_count || item.review_features || 0)} ${hEsc(helpers, hT(helpers, 'features', '特征'))}</div>
                  </div>`;
              }).join('') : `
                <div class="pt-missingness-empty">${hEsc(helpers, hT(helpers, 'No true missingness module is available in this payload.', '当前载荷没有可计算真实缺失率的模块。'))}</div>`}
            </div>
          </div>
          ${visiblePresence.length ? `
            <div class="pt-presence-panel" data-patient-presence-rate-modules>
              <div class="pt-missingness-panel-head">
                <b>${hEsc(helpers, hT(helpers, 'Event / exposure prevalence', '事件 / 暴露发生率'))}</b>
                <span>${hEsc(helpers, hT(helpers, 'not missingness', '不是缺失率'))}</span>
              </div>
              <div class="pt-presence-list">
                ${visiblePresence.map(item => {
                  const label = moduleDisplayLabel(item, helpers);
                  return `
                    <div class="pt-presence-row">
                      <span>${hEsc(helpers, label)}</span>
                      <b>${hFmtPct(helpers, clampPct(item.coverage_pct) || 0)}</b>
                      <em>${hEsc(helpers, String(item.metric_kind || hT(helpers, 'presence rate', '发生率')))}</em>
                    </div>`;
                }).join('')}
              </div>
            </div>` : ''}
        </div>
        <div class="pt-missingness-note">${hEsc(helpers, hT(helpers, 'Clinical flags such as Sepsis-3, outcome, vasopressor exposure, or ventilation prevalence describe event occurrence. They are deliberately excluded from the missingness watchlist.', 'Sepsis-3、结局、血管活性药物暴露、机械通气等临床标志描述事件发生；这里刻意不把它们放进缺失率风险列表。'))}</div>
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
      renderCategorySummary(sections, helpers),
      renderModuleLedger(drill, helpers),
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
