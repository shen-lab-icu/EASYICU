/* Patient Review domain owner.
   Owns Patient view state, source-backed loading, complete review rendering,
   and route event dispatch. The Data Workbench host retains only shared source
   selection, navigation, and generic formatting primitives. */
(function () {
  'use strict';

  const S = (window.SCREENS = window.SCREENS || {});
  const { DEMO_DURATION_HOURS } = window.VIZ_DEMO;
  const { buildPatientDrilldown: buildDemoPatientDrilldown } = window.VIZ_DEMO_DRILLDOWN;
  const {
    signalKey: ptSignalKey,
    catalogLanes: patientCatalogLanes,
  } = window.EU_PATIENT_FEATURES;

  let host = {
    t: (en) => en,
    icon: () => '',
    esc: value => String(value == null ? '' : value),
    fmtInt: value => String(value == null ? '—' : value),
    fmtNum: value => String(value == null ? '—' : value),
    fmtPct: value => String(value == null ? '—' : value),
    axisSpark: () => '',
    workspaceSamplingNote: () => '',
    vizRail: () => '',
    registrySources: () => [],
    registryActivePath: () => null,
    sourceLine: () => '',
    sourceRegistryBlock: () => '',
    bindSourceRegistry: () => {},
    repaintScreen: () => {},
    skeletonWorkspace: () => '',
    downloadJsonFile: () => {},
  };

  const t = (en, zh) => host.t(en, zh);
  const icon = (...args) => host.icon(...args);
  const esc = value => host.esc(value);
  const fmtInt = (...args) => host.fmtInt(...args);
  const fmtNum = (...args) => host.fmtNum(...args);
  const fmtPct = (...args) => host.fmtPct(...args);
  const axisSpark = (...args) => host.axisSpark(...args);
  const workspaceSamplingNote = summary => host.workspaceSamplingNote(summary);
  const vizRail = active => host.vizRail(active);
  const registrySources = () => host.registrySources();
  const registryActivePath = () => host.registryActivePath();
  const sourceLine = source => host.sourceLine(source);
  const sourceRegistryBlock = mode => host.sourceRegistryBlock(mode);
  const bindSourceRegistry = (root, screenId) => host.bindSourceRegistry(root, screenId);
  const repaintScreen = id => host.repaintScreen(id);
  const skeletonWorkspace = mode => host.skeletonWorkspace(mode);
  const downloadJsonFile = (filename, payload) => host.downloadJsonFile(filename, payload);

  let patientView = 'idle';
  let patientTab = 'tables';
  let patientSeriesMode = 'lanes';
  let vizErr = null;

  function patientDrilldown() {
    if (window.EU_PATIENT_DRILLDOWN) return window.EU_PATIENT_DRILLDOWN;
    /* In demo mode every entry path (incl. the global "load demo workspace", which only
       seeds the thin EU_VIZ_WORKSPACE) should still get the rich catalog-shaped drill so
       Tables / Time Series / Quality render full review content instead of thin fallbacks.
       Build once and cache; buildDemoPatientDrilldown is hoisted. */
    if (window.EU_DATA !== 'real' && patientView === 'loaded') {
      window.EU_PATIENT_DRILLDOWN = buildDemoPatientDrilldown();
      return window.EU_PATIENT_DRILLDOWN;
    }
    return null;
  }
  function patientTablePreviews(payload) {
    const tables = (payload || patientDrilldown() || {}).data_tables || {};
    return Array.isArray(tables.table_previews) ? tables.table_previews : [];
  }
  function activePatientTablePreview(payload) {
    const owner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.tables;
    if (owner && typeof owner.activePreview === 'function') {
      return owner.activePreview(payload || patientDrilldown());
    }
    const previews = patientTablePreviews(payload);
    if (!previews.length) return null;
    const tables = (payload || patientDrilldown() || {}).data_tables || {};
    const fallback = (tables.module_picker || {}).default_module || (previews[0] && previews[0].module);
    return previews.find(row => row.module === fallback) || previews[0] || null;
  }
  function fmtCell(v) {
    if (v == null || v === '') return '—';
    if (typeof v === 'number') return Number.isFinite(v) ? fmtNum(v, 3) : '—';
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    return String(v);
  }
  function patientI18nLabel(meta, fallback) {
    const row = meta || {};
    if (window.EU_LANG === 'zh') return row.label_zh || row.zh || row.name_zh || row.label_en || row.en || fallback || '';
    return row.label_en || row.en || row.name_en || row.label_zh || row.zh || fallback || '';
  }
  function patientModuleLabel(row) {
    return patientI18nLabel(row && row.label_i18n, row && (row.label || row.module));
  }

  function patientFeatureLabel(row) {
    return patientI18nLabel(row && row.name_i18n, row && (row.name || row.feature));
  }
  function patientColumnLabelMap(preview) {
    const out = {};
    ((preview && preview.display_column_labels) || []).forEach(row => {
      if (row && row.column) out[row.column] = row;
    });
    return out;
  }
  function patientColumnLabel(col, preview) {
    const labels = patientColumnLabelMap(preview);
    if (labels[col]) return patientI18nLabel(labels[col], col);
    if (col === 'entity') return t('Pseudonymous entity', '伪匿名实体');
    return String(col || '').replace(/[_-]+/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase());
  }
  function patientFlowText(row, key, fallback) {
    const i18n = row && row[key + '_i18n'];
    if (i18n) return patientI18nLabel(i18n, fallback || row[key] || '');
    return (row && row[key]) || fallback || '';
  }
  function patientEligibilityFlow(flow) {
    const steps = Array.isArray(flow && flow.steps) ? flow.steps.filter(s => s && s.count != null) : [];
    if (!steps.length) return '';
    const title = flow.title_i18n ? patientI18nLabel(flow.title_i18n, flow.title) : (flow.title || t('Eligibility flow (ICU stays)', '入组筛选流程（ICU 住院）'));
    const subtitle = flow.has_stepwise_report
      ? t('Computed from the export cohort report; no patient rows are returned to the browser.', '来自导出 cohort report；浏览器不返回患者行。')
      : t('This export does not contain a stepwise filter log, so only the available denominator is shown.', '这个导出没有逐步筛选日志，因此只显示可用分母。');
    return `
      <div class="patient-flow-card mt-16" data-patient-eligibility-flow>
        <div class="patient-flow-head">
          <div>
            <h2 class="patient-flow-title">${esc(title)}</h2>
            <div class="patient-flow-sub">${esc(subtitle)}</div>
          </div>
          <span class="pill ${flow.has_stepwise_report ? 'ok' : 'warn'}" style="height:22px;">${flow.has_stepwise_report ? t('stepwise', '逐步') : t('summary only', '仅摘要')}</span>
        </div>
        <div class="patient-flow-diagram" role="img" aria-label="${esc(title)}">
          ${steps.map((step, idx) => {
            const isLast = idx === steps.length - 1;
            const excluded = Number(step.excluded);
            const hasExcluded = Number.isFinite(excluded) && excluded > 0;
            const note = patientFlowText(step, 'note', '');
            const pctNum = Number(step.pct_of_initial);
            const pct = step.pct_of_initial == null || (idx === 0 && Math.abs(pctNum - 100) < 0.05) ? '' : `(${fmtPct(step.pct_of_initial)})`;
            const exclPct = step.excluded_pct_of_previous == null ? '' : `(${fmtPct(step.excluded_pct_of_previous)})`;
            return `
              <div class="patient-flow-node ${idx === 0 ? 'first' : ''} ${step.final || isLast ? 'final' : ''} ${isLast ? 'last' : 'has-next'}">
                <div>
                  <div class="patient-flow-label">${esc(patientFlowText(step, 'label', step.id))}</div>
                  ${note ? `<div class="patient-flow-note">(${esc(note)})</div>` : ''}
                  <div class="patient-flow-count">${fmtInt(step.count)}</div>
                  ${pct ? `<div class="patient-flow-pct">${esc(pct)}</div>` : ''}
                </div>
              </div>
              <div class="patient-flow-side-link ${hasExcluded ? '' : 'empty'}" aria-hidden="true"></div>
              <div class="patient-flow-excluded ${hasExcluded ? '' : 'empty'}">
                ${hasExcluded ? `
                  <div>
                    <div class="patient-flow-ex-title">${t('Excluded', '排除')}</div>
                    <div class="patient-flow-ex-count">${fmtInt(excluded)}</div>
                    <div class="patient-flow-ex-pct">${esc(exclPct || t('from previous step', '相对上一步'))}</div>
                  </div>` : ''}
              </div>`;
          }).join('')}
        </div>
      </div>`;
  }
  function patientShapeLabel(module) {
    const shape = module && module.shape;
    if (shape === 'time_indexed') return t('time indexed', '时序');
    if (shape === 'static') return t('static', '静态');
    return shape || (module && module.time_indexed ? t('time indexed', '时序') : t('static', '静态'));
  }
  function patientSignalLabel(signal) {
    return (signal && (signal.label || signal.name || signal.feature || signal.key)) || 'signal';
  }
  function patientMatrixCell(value, values, unit) {
    const num = Number(value);
    if (!Number.isFinite(num)) return `<td class="num muted">—</td>`;
    const nums = (values || []).map(v => Number(v)).filter(v => Number.isFinite(v));
    const min = nums.length ? Math.min(...nums) : num;
    const max = nums.length ? Math.max(...nums) : num;
    const ratio = max > min ? Math.max(0, Math.min(1, (num - min) / (max - min))) : 0.5;
    const shade = Math.round(7 + ratio * 31);
    const title = `${fmtNum(num, 3)}${unit ? ` ${unit}` : ''} · range ${fmtNum(min, 3)}-${fmtNum(max, 3)}`;
    return `<td class="num mono" title="${esc(title)}" style="background:color-mix(in srgb, var(--accent) ${shade}%, var(--surface));">${esc(fmtNum(num, 2))}</td>`;
  }
  function patientFeatureMatrix(lane, drill, opts = {}) {
    const signals = ((lane || {}).signals || [])
      .filter(s => Array.isArray(s.values) && s.values.some(v => Number.isFinite(Number(v))));
    const pointCap = Math.max(1, Math.min(Number(opts.maxRows) || 24, 24));
    const pointCount = signals.length ? Math.max(...signals.map(s => (s.values || []).length)) : 0;
    const shownPoints = Math.min(pointCount, pointCap);
    const firstTimes = signals.length ? (signals[0].times || signals[0].charttimes || signals[0].charttime || []) : [];
    const timeLabel = index => {
      const raw = Array.isArray(firstTimes) ? firstTimes[index] : null;
      const numeric = Number(raw);
      if (Number.isFinite(numeric)) return `${fmtNum(numeric, numeric % 1 ? 1 : 0)}h`;
      if (raw != null && raw !== '') return String(raw);
      return `t${index}`;
    };
    const entityLabel = ((drill || {}).selected || {}).label || t('selected entity', '已选实体');
    const sourceBadge = (drill && drill.demo) ? t('demo', '演示') : t('real', '真实');
    if (!signals.length) {
      return `
        <div class="empty mt-16"><div class="glyph">${icon('rows', 22)}</div><div class="t">${t('No feature matrix values', '暂无特征矩阵值')}</div><div class="d">${t('This lane has no bounded numeric time-window values.', '这个分组没有有界的数值型时间窗口数据。')}</div></div>`;
    }
    return `
      <div class="card pad mt-16" data-patient-feature-matrix="${esc((lane || {}).lane || 'signals')}">
        <div class="mc-head">
          <div>
            <div style="font-weight:600;font-size:13px;">${esc((lane || {}).label || (lane || {}).lane || t('Selected signals', '已选信号'))}</div>
            <div class="mono" style="font-size:10.5px;color:var(--ink-4);">${fmtInt(shownPoints)} ${t('time windows', '个时间窗口')} × ${fmtInt(signals.length)} ${t('features', '个特征')} · ${esc(entityLabel)}</div>
          </div>
          <span class="pill ${(drill && drill.demo) ? 'demo' : 'ok'}" style="height:22px;">${sourceBadge}</span>
        </div>
        <div class="table-wrap table-scroll mt-12">
          <table class="eu-table">
            <thead>
              <tr>
                <th>${t('Window', '时间窗')}</th>
                ${signals.map(s => `<th class="num"><span>${esc(patientSignalLabel(s))}</span>${s.unit ? `<br><span class="mono" style="font-size:10px;color:var(--ink-4);">${esc(s.unit)}</span>` : ''}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
              ${Array.from({ length: shownPoints }, (_, i) => `
                <tr>
                  <td class="key mono">${esc(timeLabel(i))}</td>
                  ${signals.map(s => patientMatrixCell((s.values || [])[i], s.values || [], s.unit || '')).join('')}
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
        <div class="row wrap gap-6 mt-8" style="font-size:11.5px;color:var(--ink-4);">
          <span>${t('Rows are time windows; columns are selected features.', '行是时间窗口；列是已选特征。')}</span>
          ${pointCount > shownPoints ? `<span>${t('matrix preview capped', '矩阵预览已截断')} ${fmtInt(shownPoints)} / ${fmtInt(pointCount)}</span>` : ''}
        </div>
      </div>`;
  }
  function patientOverviewWorkbench(selected, summaryCards, sections, drill) {
    const renderer = window.EU_PATIENT_OVERVIEW && window.EU_PATIENT_OVERVIEW.renderOverview;
    if (typeof renderer !== 'function') {
      return `<div class="empty mt-16"><div class="glyph">${icon('grid', 22)}</div><div class="t">${t('Patient overview renderer unavailable', '患者概览渲染器不可用')}</div><div class="d">${t('Reload the page; the owner module did not load.', '请刷新页面；对应 owner 模块没有加载。')}</div></div>`;
    }
    return renderer({ selected, summaryCards, sections, drill }, {
      t,
      esc,
      icon,
      fmtInt,
      fmtNum,
      fmtPct,
      signalLabel: patientSignalLabel,
      moduleLabel: patientModuleLabel,
    });
  }

  function patientEntityNavigator(drill, selected, opts = {}) {
    const owner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.navigation;
    if (owner && typeof owner.render === 'function') {
      return owner.render({
        drill,
        selected,
        opts,
        helpers: { t, esc, fmtInt, icon },
      });
    }
    return '';
  }
  function patientMatrixAudit(drill, lanesOverride = null) {
    const review = drill ? (drill.trajectory_review || {}) : {};
    const lanes = Array.isArray(lanesOverride) ? lanesOverride : (Array.isArray(review.lanes) ? review.lanes : (drill && Array.isArray(drill.time_lanes) ? drill.time_lanes : []));
    const readyLanes = lanes.filter(lane => (lane.signals || []).some(sig => Array.isArray(sig.values) && sig.values.some(v => Number.isFinite(Number(v)))));
    if (!drill || !readyLanes.length) return '';
    const signalScope = drill.demo
      ? t('Seeded matrix values are capped at', '演示矩阵值上限为')
      : t('Local matrix values are capped at', '本地矩阵值上限为');
    return `
      <details class="pt-matrix-details pt-matrix-audit mt-16">
        <summary>
          <span>${t('Exact value audit matrices', '精确值审计矩阵')}</span>
          <span class="mono">${fmtInt(readyLanes.length)} ${t('modules', '个模块')}</span>
        </summary>
        <div class="pt-matrix-details-body">
          <div class="note info">
            <div class="ico">${icon('rows', 16)}</div>
            <div class="body"><span class="t">${t('Data-table companion audit', '数据表配套审计')}</span> <span class="d" style="display:inline;">— ${t('Use this only when you need cell-level time-window values. The main Time Series tab stays focused on per-feature clinical trajectories.', '只有需要核对时间窗格子值时再展开；主时间序列页只保留单特征临床曲线。')}</span></div>
          </div>
          ${readyLanes.map(lane => patientFeatureMatrix(lane, drill)).join('')}
          <p class="pt-audit-foot">${signalScope} ${fmtInt((drill.privacy || {}).max_points_per_signal)} ${t('points per feature for browser review; lane membership follows the EasyICU clinical concept catalog.', '个点/特征用于浏览器审阅；分组来自 EasyICU 临床概念目录。')}</p>
        </div>
      </details>`;
  }
  function patientQualityText(value) {
    const map = {
      'Quality dashboard': t('Quality dashboard', '质量审阅'),
      'QC concepts': t('QC concepts', '质控特征'),
      'Records': t('Records', '记录'),
      'Seeded observations': t('Seeded observations', '种子观测值'),
      'Weighted missing': t('Weighted missing', '加权缺失'),
      'Out-of-physio': t('Out-of-physio', '生理范围外'),
      'Duplicate TS': t('Duplicate TS', '重复时间戳'),
      'QC ledger': t('QC ledger', '质控台账'),
      'Catalog concept scope': t('Catalog concept scope', '目录特征范围'),
      'Missingness gate': t('Missingness check', '缺失率检查'),
      'Physiologic range': t('Physiologic range', '生理范围'),
      'Temporal integrity': t('Temporal integrity', '时间完整性'),
      'Missingness': t('Missingness', '缺失率'),
      'Out-of-Physio': t('Out-of-Physio', '生理范围外'),
      'Temporal Integrity': t('Temporal Integrity', '时间完整性'),
      'Per-module entity coverage': t('Per-module entity coverage', '模块实体覆盖'),
      'Per-module review metrics': t('Per-module review metrics', '逐模块审阅指标'),
      'Top concept quality issues': t('Top concept quality issues', '主要特征质量问题'),
      'Local export bounded review': t('Local export bounded review', '本地导出有界审阅'),
      'Catalog demo bounded review': t('Catalog demo bounded review', '目录演示有界审阅'),
    };
    return map[String(value || '')] || value;
  }
  function patientQualityPanel(panel) {
    const rows = Array.isArray(panel && panel.rows) ? panel.rows : [];
    const metricLabel = panel && panel.id === 'temporal'
      ? t('Duplicate TS', '重复时间戳')
      : panel && panel.id === 'outliers'
        ? t('Out-of-range', '范围外')
        : t('Missing', '缺失');
    return `
      <section class="pt-qc-panel" data-patient-qc-panel="${esc(panel && panel.id || '')}">
        <div class="pt-qc-panel-head">
          <div>
            <div class="eyebrow">${esc(patientQualityText(panel && panel.label || panel && panel.id || 'Panel'))}</div>
            <div class="pt-qc-panel-sub">${t('Top bounded feature-level flags for the active review scope.', '当前审阅范围内最高的有界特征级质量标记。')}</div>
          </div>
          <span class="pill ${rows.length ? 'warn' : 'ok'}">${fmtInt(rows.length)} ${t('rows', '行')}</span>
        </div>
        <div class="table-wrap table-scroll mt-10">
          <table class="eu-table">
            <thead><tr><th>${t('Feature', '特征')}</th><th class="num">${metricLabel}</th><th class="num">${t('Records', '记录')}</th></tr></thead>
            <tbody>
              ${rows.slice(0, 8).map(row => `<tr><td class="key">${esc(row.feature || row.name || row.id || '')}</td><td class="num">${fmtPct(row.value)}</td><td class="num">${fmtInt(row.records)}</td></tr>`).join('') || `<tr><td colspan="3" class="muted">${t('No flags in this panel.', '这个面板没有质量标记。')}</td></tr>`}
            </tbody>
          </table>
        </div>
      </section>`;
  }
  function patientQualityWorkbook(review) {
    const panels = Array.isArray(review && review.panels) ? review.panels : [];
    if (!panels.length) return '';
    return `
      <div class="pt-qc-workbook mt-16" data-patient-qc-workbook>
        <div class="pt-qc-title">
          <div>
            <div class="eyebrow">${t('QC workbook', '质控工作簿')}</div>
            <h2>${t('Missingness, physiologic range, and temporal integrity', '缺失率、生理范围和时间完整性')}</h2>
          </div>
          <span class="pill ok">${t('bounded review', '有界审阅')}</span>
        </div>
        <div class="pt-qc-panels">
          ${panels.map(patientQualityPanel).join('')}
        </div>
      </div>`;
  }
  function patientSeriesLabel(value) {
    const label = String(value || '');
    const mapped = {
      'Entity scope': t('Entity scope', '实体范围'),
      'Loaded signals': t('Loaded signals', '已加载信号'),
      'Clinical lanes': t('Module overview', '模块总览'),
      'Clinical Lanes': t('Module overview', '模块总览'),
      'Feature matrices': t('Module coverage', '模块覆盖'),
      'Feature Matrix': t('Module coverage', '模块覆盖'),
      'Review mode': t('Review mode', '审阅模式'),
      'Single Patient': t('Trajectory gallery', '轨迹画廊'),
      'Multi-Patient Comparison': t('Cross-patient comparison', '跨患者对比'),
    };
    return mapped[label] || label;
  }
  function patientSeriesDetail(value) {
    let detail = String(value || '');
    const legacyAggregateDetail = ['time windows x features', 'single entity', ['aggregate', 'comparison'].join(' ')].join(' / ');
    detail = detail.replace(legacyAggregateDetail, 'module overview / trajectory gallery / cross-patient comparison');
    detail = detail.replace(
      'clinical lanes / single entity / multi-entity same-feature traces',
      'module overview / trajectory gallery / cross-patient comparison',
    );
    detail = detail.replace('catalog lane signals', 'catalog signals');
    detail = detail.replace('matrix groups available', 'modules available');
    if (window.EU_LANG === 'zh') {
      detail = detail.replace('module overview / trajectory gallery / cross-patient comparison', '模块总览 / 轨迹画廊 / 跨患者对比');
      detail = detail.replace('catalog signals', '目录信号');
      detail = detail.replace('modules available', '个模块可用');
      detail = detail.replace('selected-entity signals', '个已选实体信号');
      detail = detail.replace('pseudonymous options exposed', '个去标识实体选项');
    }
    return detail;
  }

  function patientWorkspaceFromDrilldown(payload) {
    const s = payload && payload.summary ? payload.summary : {};
    return {
      ok: true,
      route: 'patient',
      mode: payload && payload.mode ? payload.mode : 'real',
      demo: !!(payload && payload.demo),
      database: payload && payload.source ? payload.source.database : null,
      summary: {
        stays: s.entities,
        modules: s.modules,
        file_count: s.file_count,
        total_rows: s.total_rows,
        mean_age: s.mean_age,
        female_pct: s.female_pct,
        mortality: s.mortality,
        median_los_icu: s.median_los_icu,
        median_sofa2: s.median_sofa2,
        sepsis_pct: s.sepsis_pct,
      },
    };
  }

  function patientSourcePayload() {
    return window.EU_PATIENT_SOURCES || null;
  }
  function patientActiveSourceMeta() {
    const payload = patientSourcePayload();
    if (payload && payload.active_source) return payload.active_source;
    const active = registryActivePath();
    if (!active) return null;
    return registrySources().find(s => s.path === active) || null;
  }
  function patientSourceReadyCard() {
    const payload = patientSourcePayload();
    const active = patientActiveSourceMeta();
    const sourceCount = payload ? payload.source_count : registrySources().length;
    if (!active) {
      return `
      <div class="note warn mt-12" data-patient-source-ready="false">
        <div class="ico">${icon('alert', 14)}</div>
        <div class="body">
          <div class="t">${t('No active local export is ready', '还没有可用的本地导出')}</div>
          <div class="d">${t('Add or choose an EasyICU export folder below, or run Data Extraction first.', '请在下方添加或选择一个 EasyICU 导出文件夹，或先运行数据抽取。')}</div>
          <div class="d mono" style="margin-top:4px;">registered_sources=${fmtInt(sourceCount)}</div>
          <div class="row mt-8"><button class="btn sm" data-nav="extraction" type="button">${icon('extract', 13)} ${t('Open Data Extraction', '打开数据抽取')}</button></div>
        </div>
      </div>`;
    }
    const patientReady = active.patient_ready !== false;
    const sum = active.summary || {};
    const readyLine = [
      sum.entities != null ? `${fmtInt(sum.entities)} ${t('entities', '个实体')}` : (sum.stays != null ? `${fmtInt(sum.stays)} ${t('stays', '条住院')}` : null),
      sum.modules != null ? `${fmtInt(sum.modules)} ${t('modules', '个模块')}` : null,
      sum.total_rows != null ? `${fmtInt(sum.total_rows)} ${t('rows', '行')}` : null,
    ].filter(Boolean).join(' · ') || sourceLine(active);
    return `
      <div class="note ${patientReady ? 'ok' : 'warn'} mt-12" data-patient-source-ready="${patientReady ? 'true' : 'false'}">
        <div class="ico">${icon(patientReady ? 'check' : 'alert', 14)}</div>
        <div class="body">
          <div class="t">${patientReady ? t('Ready to load local export', '本地导出已就绪，可以加载') : t('Registered export needs review', '已注册导出需要检查')}</div>
          <div class="d"><b>${esc(active.label || active.database || t('Local export', '本地导出'))}</b> · ${esc(readyLine)}</div>
          <div class="d mono" style="margin-top:4px;">path_hash=${esc(active.path_hash || '—')} · ${t('local-only metadata', '仅本地元数据')}</div>
        </div>
      </div>`;
  }

  function loadRealPatient(done, entityRef) {
    if (!(window.EU_API && window.EU_API.loadPatientReviewDrilldown)) {
      vizErr = 'Patient Review API is unavailable.';
      done && done(false);
      return;
    }
    const active = registryActivePath();
    if (!active) {
      window.EU_PATIENT_DRILLDOWN = null;
      window.EU_VIZ_WORKSPACE = null;
      vizErr = 'No registered export is active. Add an EasyICU export folder or run Data Extraction first.';
      done && done(false);
      return;
    }
    const body = {};
    body.source_path = active;
    if (entityRef) body.entity_ref = entityRef;
    const tableOwner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.tables;
    const tableState = tableOwner && typeof tableOwner.snapshot === 'function'
      ? tableOwner.snapshot(patientDrilldown())
      : {};
    if (tableState.module) body.table_module = tableState.module;
    body.table_page = 1;
    body.table_page_size = tableState.pageSize || 24;
    window.EU_API.loadPatientReviewDrilldown(body).then(payload => {
      window.EU_PATIENT_DRILLDOWN = payload;
      window.EU_VIZ_WORKSPACE = patientWorkspaceFromDrilldown(payload);
      const reviewOwner = window.EU_PATIENT_REVIEW || {};
      if (reviewOwner.navigation && reviewOwner.navigation.prime) reviewOwner.navigation.prime(payload);
      if (reviewOwner.tables && reviewOwner.tables.prime) reviewOwner.tables.prime(payload);
      if (reviewOwner.features && reviewOwner.features.prime) reviewOwner.features.prime(payload);
      vizErr = null;
      window.EU_HASWORK = true;
      done && done(true);
    }).catch(err => {
      window.EU_PATIENT_DRILLDOWN = null;
      window.EU_VIZ_WORKSPACE = null;
      vizErr = String(err && err.message || err);
      done && done(false);
    });
  }
  function loadPatientSources(done) {
    if (!(window.EU_API && window.EU_API.loadPatientReviewSources)) {
      done && done(false);
      return;
    }
    window.EU_PATIENT_SOURCES_LOADING = true;
    window.EU_API.loadPatientReviewSources({}).then(payload => {
      window.EU_PATIENT_SOURCES = payload;
      vizErr = null;
      done && done(true, payload);
    }).catch(err => {
      window.EU_PATIENT_SOURCES = null;
      vizErr = String(err && err.message || err);
      done && done(false);
    }).finally(() => {
      window.EU_PATIENT_SOURCES_LOADING = false;
    });
  }

  function patientTabs() {
    const tabs = [
      ['tables', t('Data Tables', '数据表'), 'rows'],
      ['series', t('Time Series', '时间序列'), 'viz'],
      ['patient', t('Patient Overview', '患者概览'), 'patient'],
      ['quality', t('Data Quality', '数据质量'), 'shield'],
    ];
    return `<div class="tabs" id="ptabs">${tabs.map(([k, lab, ic]) =>
      `<button class="tab ${patientTab === k ? 'active' : ''}" data-ptab="${k}">${icon(ic, 14)} ${lab}</button>`).join('')}</div>`;
  }

  function ptTables() {
    const drill = patientDrilldown();
    if (drill && drill.summary) {
      const s = drill.summary || {};
      const dt = drill.data_tables || {};
      const loaded = dt.loaded_summary || {};
      const detailGate = dt.detail_gate || {};
      const picker = dt.module_picker || {};
      const modules = drill.module_profiles || [];
      const activePreview = activePatientTablePreview(drill);
      const tableOwner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.tables;
      const tableState = tableOwner && typeof tableOwner.snapshot === 'function'
        ? tableOwner.snapshot(drill)
        : { module: activePreview && activePreview.module, page: 1, pageSize: 24 };
      const previewColumns = activePreview && Array.isArray(activePreview.display_columns) ? activePreview.display_columns : [];
      const previewRows = activePreview && Array.isArray(activePreview.rows) ? activePreview.rows : [];
      const previewPage = (activePreview && activePreview.pagination) || {};
      const page = Number(previewPage.page || activePreview && activePreview.page || tableState.page || 1);
      const pageCount = Number(previewPage.page_count || activePreview && activePreview.page_count || 1);
      const rowStart = Number(previewPage.row_start || activePreview && activePreview.row_start || 0);
      const rowEnd = Number(previewPage.row_end || activePreview && activePreview.row_end || 0);
      const hasPrevious = Boolean(previewPage.has_previous || activePreview && activePreview.has_previous);
      const hasNext = Boolean(previewPage.has_next || activePreview && activePreview.has_next);
      const pageSize = Number(previewPage.page_size || activePreview && activePreview.page_size || tableState.pageSize || 24);
      /* Labels and provenance captions both go through t(): this table is the
         only place in Patient Review that still rendered bare English in the
         Chinese locale. The caption states what each number is computed from,
         so it has to read in the user's language, not just the label. */
      const basisScope = drill.demo
        ? t('clinically constrained synthetic aggregate', '临床约束合成聚合值')
        : t('demographics aggregate', '人口学聚合值');
      const rows = [
        [t('Entities', '实体'), fmtInt(s.entities), drill.demo
          ? t('synthetic fallback denominator', '合成兜底分母')
          : t('cohort denominator from active export', '来自当前导出的队列分母')],
        [t('Mean age', '平均年龄'), fmtNum(s.mean_age, 1), basisScope],
        [t('Female', '女性'), fmtPct(s.female_pct), basisScope],
        [t('Mortality', '病死率'), fmtPct(s.mortality), drill.demo
          ? t('synthetic fallback outcome', '合成兜底结局')
          : t('outcome aggregate', '结局聚合值')],
        [t('Median SOFA-2', 'SOFA-2 中位数'), fmtNum(s.median_sofa2, 1), drill.demo
          ? t('synthetic fallback score', '合成兜底评分')
          : t('score aggregate', '评分聚合值')],
        [t('Sepsis-3 positive', 'Sepsis-3 阳性'), fmtPct(s.sepsis_pct), drill.demo
          ? t('synthetic fallback event', '合成兜底事件')
          : t('event aggregate', '事件聚合值')],
      ];
      const reviewModules = (dt.modules && dt.modules.length ? dt.modules : modules).slice(0, drill.demo ? 32 : 64);
      const activeModule = reviewModules.find(m => m.module === tableState.module) || reviewModules[0] || {};
      const previewFeatures = activeModule.preview_features || [];
      const workspaceCopy = drill.demo
        ? t('Clinically constrained synthetic fallback: modules and feature names come from the EasyICU catalog; modeled values share one deterministic phenotype model and are for UI rehearsal only.', '临床约束合成兜底：模块和特征名来自 EasyICU 目录；建模数值共享同一个确定性表型模型，仅用于界面演练。')
        : t('module table previews, feature counts, source scope and optional detail gate are computed from the active export.', '模块表格预览、特征数量、来源范围和详情边界都从当前导出计算。');
      return `
      <div class="st-stats mt-16">
        ${[
          [t('Entities', '实体'), fmtInt(loaded.entities != null ? loaded.entities : s.entities), 'ok'],
          [t('Review features', '审阅特征'), fmtInt(loaded.review_features), 'accent'],
          [t('Modules', '模块'), fmtInt(loaded.module_count), 'accent'],
          [t('Observed features', '已观测特征'), fmtInt(loaded.observed_features), 'accent'],
        ].map(([l, v, c]) => `<div class="stat ${c}"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('')}
      </div>
      ${workspaceSamplingNote(s)}
      <div class="note ok mt-16">
        <div class="ico">${icon('rows', 16)}</div>
        <div class="body"><span class="t">${t('Table preview', '表格预览')}</span> <span class="d" style="display:inline;">— ${drill.demo ? t('Seeded demo rows for UI preview.', '演示行仅用于界面预览。') : t('Capped local rows from the active export; identifiers are replaced by pseudonymous entity tokens.', '来自当前本地导出的有界行预览；标识符已替换为去标识化实体 token。')}</span></div>
      </div>
      ${patientEligibilityFlow(drill.eligibility_flow)}
      ${reviewModules.length ? `
      <div class="row wrap gap-6 mt-12" data-pt-table-picker>
        ${reviewModules.map(module => {
          const active = module.module === tableState.module;
          const status = tableOwner && typeof tableOwner.moduleStatus === 'function'
            ? tableOwner.moduleStatus(module, activePreview, { t })
            : (module.review_status === 'inventory_only'
              ? t('available · load', '可用 · 按需加载')
              : t('reviewed', '已审阅'));
          return `<button type="button" class="chip ${active ? 'solid' : ''}" data-pt-table-module="${esc(module.module)}" aria-pressed="${active ? 'true' : 'false'}">${esc(patientModuleLabel(module))} <span class="mono" style="font-size:10.5px;color:var(--ink-4);">${fmtInt(module.rows)} ${t('rows', '行')} · ${status}</span></button>`;
        }).join('')}
      </div>
      ${tableOwner && tableOwner.statusHtml ? tableOwner.statusHtml({ t, esc, icon }) : ''}
      <div class="patient-table-frame mt-12">
        <div class="patient-id-note">${drill.demo ? t('Synthetic entity tokens are UI-only references.', '合成实体 token 仅用于界面引用。') : t('Entity tokens are local pseudonymous references. Direct clinical identifiers stay on disk.', '实体 token 是本地伪匿名引用；直接临床标识符保留在磁盘上。')}</div>
        <div class="patient-table-scroll" data-patient-table-preview style="--pt-cols:${Math.max(6, previewColumns.length)};">
        <table class="eu-table patient-preview-table" aria-label="${esc(`${patientModuleLabel(activeModule)} ${t('bounded table preview', '有界表格预览')}`)}">
          <thead><tr>${previewColumns.map(c => `<th${c === 'entity' ? ' class="patient-entity-col"' : ' class="num"'}>${esc(patientColumnLabel(c, activePreview))}</th>`).join('')}</tr></thead>
          <tbody>
            ${previewRows.length ? previewRows.map(r => `<tr>${previewColumns.map(c => `<td class="${c === 'entity' ? 'key mono patient-entity-token' : 'num'}">${esc(fmtCell(r[c]))}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${Math.max(1, previewColumns.length)}" class="muted">${esc(activePreview && activePreview.reason ? activePreview.reason : t('No preview rows available for this module.', '这个模块没有可预览行。'))}</td></tr>`}
          </tbody>
        </table>
        </div>
      </div>
      <div class="patient-table-pager mt-8" role="group" aria-label="${esc(t('Table page controls', '表格分页控件'))}">
        <button type="button" class="btn sm" data-pt-page-prev ${hasPrevious ? '' : 'disabled'}>${icon('arrow-left', 13)} ${t('Previous', '上一页')}</button>
        <div class="patient-page-readout">
          <span class="mono">${esc(activePreview && activePreview.module || '')}</span>
          <span>${rowStart && rowEnd ? `${fmtInt(rowStart)}-${fmtInt(rowEnd)}` : fmtInt(activePreview && activePreview.row_count)} / ${fmtInt(activePreview && activePreview.rows_total)} ${t('rows', '行')}</span>
          <span>${window.EU_LANG === 'zh' ? `第 ${fmtInt(page)} / ${fmtInt(pageCount)} 页` : `Page ${fmtInt(page)} / ${fmtInt(pageCount)}`}</span>
        </div>
        <label class="patient-page-size">${t('Rows', '行数')}
          <select data-pt-page-size>
            ${[24, 50, 100].map(n => `<option value="${n}" ${pageSize === n ? 'selected' : ''}>${n}</option>`).join('')}
          </select>
        </label>
        <button type="button" class="btn sm" data-pt-page-next ${hasNext ? '' : 'disabled'}>${t('Next', '下一页')} ${icon('arrow', 13)}</button>
      </div>
      <div class="row wrap gap-6 mt-8" style="font-size:11.5px;color:var(--ink-4);">
        <span>${fmtInt(previewColumns.length)} / ${fmtInt(activePreview && activePreview.columns_total)} ${t('columns', '列')}</span>
        ${(activePreview && activePreview.truncated_rows) ? `<span>${t('server-paged preview', '服务端分页预览')}</span>` : ''}
        ${(activePreview && activePreview.truncated_columns) ? `<span>${t('column preview capped', '列预览已截断')}</span>` : ''}
      </div>` : `
      <div class="empty mt-16"><div class="glyph">${icon('rows', 22)}</div><div class="t">${t('No table preview available', '暂无表格预览')}</div><div class="d">${t('This export has module metadata but no displayable table columns after identifier removal.', '这个导出有模块元数据，但去除标识符后没有可显示的表格列。')}</div></div>`}
      <div class="note info mt-16">
        <div class="ico">${icon('rows', 16)}</div>
        <div class="body"><span class="t">${t('Review workspace summary', '审阅工作区摘要')}</span> <span class="d" style="display:inline;">— ${esc(workspaceCopy)}</span></div>
      </div>
      <div class="table-wrap table-scroll mt-12">
        <table class="eu-table">
          <thead><tr><th>${t('Aggregate', '聚合项')}</th><th class="num">${t('Value', '数值')}</th><th>${t('Basis', '依据')}</th></tr></thead>
          <tbody>
            ${rows.map(r => `<tr><td class="key">${esc(r[0])}</td><td class="num">${esc(r[1])}</td><td>${esc(r[2])}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      ${reviewModules.length ? `
      <div class="split-320 mt-16" style="grid-template-columns:1fr 310px;">
        <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${t('Module table overview', '模块表格概览')}</th><th class="num">${t('Features', '特征')}</th><th class="num">${t('Rows', '行')}</th><th class="num">${t('Entities', '实体')}</th><th class="num">${t('Coverage', '覆盖率')}</th><th>${t('Shape', '形态')}</th></tr></thead>
          <tbody>
            ${reviewModules.map(m => `<tr><td class="key">${esc(patientModuleLabel(m))}</td><td class="num">${fmtInt(m.review_features != null ? m.review_features : m.feature_count)}</td><td class="num">${fmtInt(m.rows)}</td><td class="num">${fmtInt(m.entities)}</td><td class="num">${fmtPct(m.coverage_pct)}</td><td>${esc(patientShapeLabel(m))} · ${fmtInt(m.dynamic_features || 0)} ${t('dynamic', '动态')}</td></tr>`).join('')}
          </tbody>
        </table>
        </div>
        <div class="card pad">
          <div class="eyebrow">${t('Module at a glance', '模块速览')}</div>
          <div style="font-weight:600;font-size:15px;margin-top:6px;">${esc(patientModuleLabel(activeModule) || t('Selected module', '已选模块'))}</div>
          <div class="col gap-6 mt-12" style="font-size:12.5px;">
            <div class="setup-row"><span class="k">${t('Review features', '审阅特征')}</span><span class="vv">${fmtInt(activeModule.review_features != null ? activeModule.review_features : activeModule.feature_count)}</span></div>
            <div class="setup-row"><span class="k">${t('Share', '占比')}</span><span class="vv">${fmtPct(activeModule.share_pct)}</span></div>
            <div class="setup-row"><span class="k">${t('Coverage', '覆盖率')}</span><span class="vv">${fmtPct(activeModule.coverage_pct)}</span></div>
            <div class="setup-row"><span class="k">${t('Status', '状态')}</span><span class="vv">${esc(activeModule.status || 'ready')}</span></div>
          </div>
          <div class="row wrap gap-6 mt-12">
            ${previewFeatures.slice(0, 6).map(f => `<span class="chip">${esc(patientFeatureLabel(f))}${f.unit ? ` · ${esc(f.unit)}` : ''}</span>`).join('') || `<span class="chip">${t('metadata only', '仅元数据')}</span>`}
          </div>
        </div>
      </div>` : ''}
      ${patientMatrixAudit(drill)}
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">${esc(patientI18nLabel(detailGate.title_i18n, detailGate.title || t('Source records are optional', '源始记录是可选的')))}</span> <span class="d" style="display:inline;">— ${esc(patientI18nLabel(detailGate.reason_i18n, detailGate.reason || t('Native Patient Review exposes cohort aggregates and one pseudonymous entity drilldown. Direct identifier tables stay out of the browser payload.', '原生患者审阅只暴露队列聚合值和单个假名化实体的下钻。直接标识表不会进入浏览器载荷。')))}</span></div>
      </div>`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && ws.tableRows) {
      const s = ws.summary || {};
      const rows = ws.tableRows.slice(0, 12);
      return `
      <div class="st-stats mt-16">
        ${[
          ['Stays', fmtInt(s.stays), 'ok'],
          ['Mean age', fmtNum(s.mean_age, 1), 'accent'],
          ['Mortality', fmtPct(s.mortality), 'accent'],
          ['Median SOFA-2', fmtNum(s.median_sofa2, 1), 'accent'],
        ].map(([l, v, c]) => `<div class="stat ${c}"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('')}
      </div>
      <div class="table-wrap table-scroll mt-16">
        <table class="eu-table">
          <thead><tr><th>stay_id</th><th class="num">age</th><th>sex</th><th class="num">SOFA-2</th><th class="num">LOS (d)</th><th>outcome</th></tr></thead>
          <tbody>
            ${rows.map(r => `<tr><td class="key mono">${esc(r.stay_id)}</td><td class="num">${fmtNum(r.age, 0)}</td><td>${esc(r.sex || '')}</td><td class="num">${fmtNum(r.sofa2, 1)}</td><td class="num">${fmtNum(r.los_icu, 1)}</td><td>${r.outcome === 'Deceased' ? '<span class="pill bad" style="height:20px;"><span class="dot"></span>Deceased</span>' : (r.outcome === 'Survived' ? '<span class="pill ok" style="height:20px;"><span class="dot"></span>Survived</span>' : '<span class="pill" style="height:20px;">Unknown</span>')}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Real local export · <span class="mono">${esc(ws.path)}</span></p>`;
    }
    const rows = [
      ['20001', '67', 'M', '6', '2.4', '5.6', 'Survived'],
      ['20002', '54', 'F', '4', '1.9', '4.1', 'Survived'],
      ['20003', '72', 'M', '11', '4.8', '8.4', 'Deceased'],
      ['20004', '49', 'F', '3', '1.6', '3.2', 'Survived'],
      ['20005', '61', 'M', '8', '3.1', '6.0', 'Survived'],
    ];
    return `
      <div class="st-stats mt-16">
        ${[['Stays', '10', 'ok'], ['Mean age', '54.8', 'accent'], ['Mortality', '20.0%', 'accent'], ['Mech vent', '52.1%', 'accent']].map(([l, v, c]) =>
          `<div class="stat ${c}"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('')}
      </div>
      <div class="table-wrap table-scroll mt-16">
        <table class="eu-table">
          <thead><tr><th>stay_id</th><th class="num">age</th><th>sex</th><th class="num">SOFA</th><th class="num">lactate</th><th class="num">LOS (d)</th><th>outcome</th></tr></thead>
          <tbody>
            ${rows.map(r => `<tr><td class="key mono">${r[0]}</td><td class="num">${r[1]}</td><td>${r[2]}</td><td class="num">${r[3]}</td><td class="num">${r[4]}</td><td class="num">${r[5]}</td><td>${r[6] === 'Deceased' ? '<span class="pill bad" style="height:20px;"><span class="dot"></span>Deceased</span>' : '<span class="pill ok" style="height:20px;"><span class="dot"></span>Survived</span>'}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Demo / seeded example values for UI preview — not a real run output.</p>`;
  }

  function patientVitalSmallMultiples(lanes) {
    const renderer = window.EU_PATIENT_SERIES && window.EU_PATIENT_SERIES.renderModulePanels;
    if (typeof renderer !== 'function') return '';
    return renderer(lanes, {
      t,
      esc,
      fmtInt,
      fmtNum,
      axisSpark,
      signalLabel: patientSignalLabel,
      seriesLabel: patientSeriesLabel,
      signalKey: ptSignalKey,
      demoHours: () => (window.EU_DATA !== 'real' && DEMO_DURATION_HOURS) ? DEMO_DURATION_HOURS : null,
    });
  }
  function patientTimeSeriesWorkbench(drill, review, lanes) {
    const renderer = window.EU_PATIENT_SERIES && window.EU_PATIENT_SERIES.renderTimeSeriesWorkspace;
    if (typeof renderer !== 'function') return patientVitalSmallMultiples(lanes);
    return renderer({
      drill,
      review,
      lanes,
      selected: drill && drill.selected,
      mode: patientSeriesMode,
    }, {
      t,
      esc,
      fmtInt,
      fmtNum,
      fmtPct,
      icon,
      axisSpark,
      signalLabel: patientSignalLabel,
      seriesLabel: patientSeriesLabel,
      signalKey: ptSignalKey,
      demoHours: () => (window.EU_DATA !== 'real' && DEMO_DURATION_HOURS) ? DEMO_DURATION_HOURS : null,
    });
  }
  function ptSeries() {
    const drill = patientDrilldown();
    const review = drill ? (drill.trajectory_review || {}) : {};
    const signals = drill && drill.selected ? (drill.selected.signals || []) : [];
    const rawLanes = Array.isArray(review.lanes) ? review.lanes : (drill && Array.isArray(drill.time_lanes) ? drill.time_lanes : []);
    const featureOwner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.features;
    const augmentedLanes = featureOwner && featureOwner.augmentLanes
      ? featureOwner.augmentLanes(rawLanes, drill)
      : rawLanes;
    const lanes = patientCatalogLanes(
      augmentedLanes,
      drill && drill.feature_coverage,
      feature => featureOwner && featureOwner.stateFor
        ? featureOwner.stateFor(feature, drill)
        : {},
    );
    const readyLanes = lanes.filter(lane => (lane.signals || []).length || (lane.features || []).length);
    if (drill && readyLanes.length) {
      return `
      ${patientEntityNavigator(drill, drill.selected, {
        detail: t('All three views use the same selected entity; switch between module coverage, loaded charts, and cross-patient comparison.', '三个视图共用当前患者：分别查看模块覆盖、已加载轨迹和跨患者同特征对比。'),
      })}
      ${patientTimeSeriesWorkbench(drill, review, readyLanes)}
      <div class="note ok mt-16">
        <div class="ico">${icon('rows', 16)}</div>
        <div class="body"><span class="t">${t('Three distinct review views', '三种审阅视图')}</span> <span class="d" style="display:inline;">— ${t('Module overview shows all catalog features and loading states; trajectory gallery shows loaded charts for the current entity; cross-patient comparison aligns the same feature across entities.', '模块总览展示全部特征及加载状态；轨迹画廊展示当前患者已加载图表；跨患者对比查看同一特征在不同患者中的表现。')}</span></div>
      </div>
      <div class="grid cards-4 mt-16">
        ${(review.contract || []).map(row => `
          <div class="stat ${row.status === 'ready' ? 'ok' : row.status === 'warn' ? 'warn' : 'accent'}">
            <div class="label">${esc(row.index || '')} · ${esc(patientSeriesLabel(row.label))}</div>
            <div class="val" style="font-size:13px;">${esc(patientSeriesDetail(row.detail))}</div>
          </div>`).join('')}
      </div>
      <div class="row wrap gap-6 mt-16">
        ${(review.modes || []).map(mode => `<span class="chip ${mode.status === 'ready' ? 'solid' : ''}">${esc(patientSeriesLabel(mode.label || mode.id))} · ${esc(mode.status || 'available')}</span>`).join('')}
      </div>
      ${patientMatrixAudit(drill, readyLanes)}`;
    }
    if (drill && signals.length) {
      return `
      ${patientFeatureMatrix({ lane: 'selected', label: t('Selected signals', '已选信号'), signals, signal_count: signals.length }, drill)}
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Signal arrays are capped at ${fmtInt((drill.privacy || {}).max_points_per_signal)} points for browser review.</p>`;
    }
    if (drill) {
      return `<div class="empty mt-16"><div class="glyph">${icon('viz', 22)}</div><div class="t">${t('No bounded signals in this export', '该导出没有可用的有界信号')}</div><div class="d">${t('The active export did not include supported vitals columns for the selected entity.', '当前导出中，所选实体没有受支持的生命体征列。')}</div></div>`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && Array.isArray(ws.series) && ws.series.length) {
      return `
      ${patientFeatureMatrix({ lane: 'workspace_series', label: t('Loaded signals', '已加载信号'), signals: ws.series, signal_count: ws.series.length }, { selected: { label: t('local export', '本地导出') } })}`;
    }
    if (ws) {
      return `<div class="empty mt-16"><div class="glyph">${icon('viz', 22)}</div><div class="t">No time-series module in this export</div><div class="d">Run extraction with vitals selected to populate trend panels.</div></div>`;
    }
    const series = [
      ['hr', 'Heart rate', 'bpm', '92', [88,90,95,101,98,94,92,96,99,93]],
      ['map', 'MAP', 'mmHg', '82', [78,80,76,70,74,79,82,85,83,81]],
      ['spo2', 'SpO₂', '%', '96', [98,97,95,93,94,96,96,97,95,96]],
      ['resp', 'Respiratory rate', '/min', '18', [16,18,20,22,19,17,18,21,19,18]],
    ];
    const seededSignals = series.map(([feature, name, unit, current, values]) => ({ feature, name, unit, current, values }));
    return `
      ${patientVitalSmallMultiples([{ lane: 'seeded_vitals', label: t('Seeded vitals', '演示生命体征'), signals: seededSignals }])}
      ${patientFeatureMatrix({
        lane: 'seeded_vitals',
        label: t('Seeded vitals', '演示生命体征'),
        signals: seededSignals,
        signal_count: seededSignals.length,
      }, { demo: true, selected: { label: t('demo entity', '演示实体') } })}`;
  }

  function ptPatient() {
    const drill = patientDrilldown();
    if (drill && drill.selected) {
      const selected = drill.selected || {};
      const overview = drill.patient_overview || {};
      const dashboard = overview.dashboard || {};
      const category = overview.category_view || {};
      const dataTable = overview.data_table || {};
      const summaryCards = dashboard.summary_cards || [];
      const sections = category.sections || [];
      return `
      ${patientEntityNavigator(drill, selected)}
      ${patientOverviewWorkbench(selected, summaryCards, sections, drill)}
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">${t('Pseudonymous drilldown', '去标识患者审阅')}</span> <span class="d" style="display:inline;">— ${t('entity refs are one-way browser tokens for the active local export; direct clinical identifiers are not returned.', '实体引用是当前本地导出的单向浏览器 token；不会返回直接临床标识符。')}</span></div>
      </div>
      ${dataTable.row_preview === 'blocked' ? `
      <div class="note warn mt-12">
        <div class="ico">${icon('lock', 14)}</div>
        <div class="body"><span class="t">${t('Data Table preview blocked', '数据表预览已阻断')}</span> <span class="d" style="display:inline;">— ${esc(dataTable.reason || t('Native Patient Overview keeps source rows out of the browser payload.', '原生患者概览不会把源行放入浏览器载荷。'))}</span></div>
      </div>` : ''}`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && ws.patient) {
      const p = ws.patient;
      const ids = (ws.tableRows || []).slice(0, 5).map(r => r.stay_id);
      const trend = ws.series || [];
      return `
      <div class="row wrap gap-6 mt-16">
        <span class="eyebrow" style="align-self:center;margin-right:4px;">Select stay</span>
        ${ids.map((id, i) => `<span class="chip ${i === 0 ? 'solid' : ''}" style="${i === 0 ? 'border-color:var(--ink);color:var(--ink);' : ''}">${esc(id)}</span>`).join('')}
      </div>
      <div class="split-320 mt-16" style="grid-template-columns:300px 1fr;">
        <div class="card pad">
          <div class="eyebrow">Patient summary</div>
          <div style="font-weight:600;font-size:15px;margin-top:6px;">Stay ${esc(p.stay_id || '')}</div>
          <div class="col gap-6 mt-12" style="font-size:12.5px;">
            <div class="setup-row"><span class="k">Age · sex</span><span class="vv">${fmtNum(p.age, 0)} · ${esc(p.sex || '—')}</span></div>
            <div class="setup-row"><span class="k">SOFA-2 (max)</span><span class="vv">${fmtNum(p.sofa2, 1)}</span></div>
            <div class="setup-row"><span class="k">Sepsis-3</span><span class="vv">${p.sepsis3 == null ? '—' : (p.sepsis3 ? 'Positive' : 'Negative')}</span></div>
            <div class="setup-row"><span class="k">ICU LOS</span><span class="vv">${fmtNum(p.los_icu, 1)} d</span></div>
            <div class="setup-row"><span class="k">Outcome</span><span class="vv">${esc(p.outcome || 'Unknown')}</span></div>
          </div>
        </div>
        <div class="mini-chart">
          <div class="mc-head"><div style="font-weight:600;font-size:13px;">Vitals · stay ${esc(p.stay_id || '')}</div><span class="mono" style="font-size:10.5px;color:var(--ink-4);">local export</span></div>
          <div class="col gap-12 mt-8">
            ${trend.slice(0, 4).map((s, i) => `
              <div>
                <div class="row" style="justify-content:space-between;font-size:11px;"><span class="mono">${esc(s.key || s.name)}</span><span class="mono" style="color:var(--ink-4);">${fmtNum(s.current, 1)} ${esc(s.unit || '')}</span></div>
                <div style="height:78px;">${axisSpark(s.values || [], 440, 78, ['var(--accent)', 'var(--accent)', 'var(--ok)', 'var(--warn)'][i % 4], { unit: s.unit || '', label: s.name || s.key || 'signal' })}</div>
              </div>`).join('') || '<div style="font-size:12px;color:var(--ink-4);">No vitals trend available in this export.</div>'}
          </div>
        </div>
      </div>`;
    }
    const ids = ['20001', '20002', '20003', '20004', '20005'];
    return `
      <div class="row wrap gap-6 mt-16">
        <span class="eyebrow" style="align-self:center;margin-right:4px;">Select stay</span>
        ${ids.map((id, i) => `<span class="chip ${i === 0 ? 'solid' : ''}" style="${i === 0 ? 'border-color:var(--ink);color:var(--ink);' : ''}">${id}</span>`).join('')}
      </div>
      <div class="split-320 mt-16" style="grid-template-columns:300px 1fr;">
        <div class="card pad">
          <div class="eyebrow">Patient summary</div>
          <div style="font-weight:600;font-size:15px;margin-top:6px;">Stay 20001</div>
          <div class="col gap-6 mt-12" style="font-size:12.5px;">
            <div class="setup-row"><span class="k">Age · sex</span><span class="vv">67 · M</span></div>
            <div class="setup-row"><span class="k">SOFA (max)</span><span class="vv">6</span></div>
            <div class="setup-row"><span class="k">Sepsis-3</span><span class="vv">Positive</span></div>
            <div class="setup-row"><span class="k">ICU LOS</span><span class="vv">5.6 d</span></div>
            <div class="setup-row"><span class="k">Outcome</span><span class="vv">Survived</span></div>
          </div>
        </div>
        <div class="mini-chart">
          <div class="mc-head"><div style="font-weight:600;font-size:13px;">Vitals · stay 20001</div><span class="mono" style="font-size:10.5px;color:var(--ink-4);">24h</span></div>
          <div class="col gap-12 mt-8">
            ${[['HR', [88,90,95,101,98,94,92,96], 'var(--accent)'], ['MAP', [78,80,76,70,74,79,82,85], 'var(--accent)'], ['SpO₂', [98,97,95,93,94,96,96,97], 'var(--ok)']].map(([n, vals, col]) => `
              <div>
                <div class="row" style="justify-content:space-between;font-size:11px;"><span class="mono">${n}</span><span class="mono" style="color:var(--ink-4);">demo</span></div>
                <div style="height:78px;">${axisSpark(vals, 440, 78, col, { label: n || 'signal' })}</div>
              </div>`).join('')}
          </div>
        </div>
      </div>`;
  }

  function ptQuality() {
    const drill = patientDrilldown();
    if (drill && Array.isArray(drill.quality)) {
      const review = drill.quality_review || {};
      const qm = drill.quality_metrics || {};
      const qsum = qm.summary || {};
      const topIssues = qm.top_issues || [];
      const qualityAuditRenderer = window.EU_PATIENT_OVERVIEW && window.EU_PATIENT_OVERVIEW.renderQualityAudit;
      const qualityAudit = typeof qualityAuditRenderer === 'function'
        ? qualityAuditRenderer({ drill, review }, {
          t,
          esc,
          fmtInt,
          fmtNum,
          fmtPct,
          icon,
          signalLabel: patientSignalLabel,
          moduleLabel: patientModuleLabel,
        })
        : '';
      const boundedTitle = drill.demo ? patientQualityText('Catalog demo bounded review') : patientQualityText('Local export bounded review');
      const boundedDetail = drill.demo
        ? t('Coverage and quality indicators describe the clinically constrained synthetic fallback only. Load an official demo or local export for source-backed denominators.', '覆盖率和质量指标只描述临床约束合成兜底。请加载官方演示数据或本地导出，以获得有来源支撑的分母。')
        : t('Shown coverage, missingness, physiologic-range flags and duplicate timestamp rates are computed over the bounded local review sample. Modules without computed entity coverage remain inventory-only; formal claims stay locked to the evidence-bound agent path.', '此处显示的覆盖率、缺失率、生理范围标记和重复时间戳率基于本地有界审阅样本计算。未计算实体覆盖率的模块仅显示文件清单信息；正式结论仍锁定在证据绑定的 Agent 路径。');
      return `
      <div class="note ok mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">${patientQualityText('Quality dashboard')}</span> <span class="d" style="display:inline;">— ${t('QC workbook semantics: computed module coverage within the stated review scope, missingness, physiologic range, temporal integrity, and action-oriented issues.', '质控工作簿语义：在明确审阅范围内计算的模块覆盖率、缺失率、生理范围、时间完整性和可处理的问题清单。')}</span></div>
      </div>
      ${(review.summary_cards || []).length ? `
      <div class="st-stats mt-16">
        ${(review.summary_cards || []).map(card => `<div class="stat ${card.tone || 'accent'}"><div class="label">${esc(patientQualityText(card.label))}</div><div class="val">${card.unit === '%' ? fmtPct(card.value) : esc(card.value == null ? '—' : fmtInt(card.value))}</div></div>`).join('')}
      </div>` : (qsum.concept_count != null ? `
      <div class="st-stats mt-16">
        ${[
          [patientQualityText('QC concepts'), fmtInt(qsum.concept_count), 'ok'],
          [patientQualityText('Records'), fmtInt(qsum.total_records), 'accent'],
          [t('Weighted missing', '加权缺失'), fmtPct(qsum.weighted_missing_pct), 'accent'],
          [patientQualityText('Out-of-physio'), fmtPct(qsum.weighted_out_of_physio_pct), qsum.weighted_out_of_physio_pct > 0 ? 'warn' : 'ok'],
        ].map(([l, v, c]) => `<div class="stat ${c}"><div class="label">${esc(l)}</div><div class="val">${v}</div></div>`).join('')}
      </div>` : '')}
      ${(review.contract || []).length ? `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:8px;">${patientQualityText('QC ledger')}</div>
        <div class="grid cards-4">
          ${(review.contract || []).map(row => `
            <div class="stat ${row.status === 'ok' || row.status === 'ready' ? 'ok' : row.status === 'warn' ? 'warn' : row.status === 'bad' ? 'bad' : 'accent'}">
              <div class="label">${esc(row.index || '')} · ${esc(patientQualityText(row.label || ''))}</div>
              <div class="val" style="font-size:13px;">${esc(row.detail || '')}</div>
            </div>`).join('')}
        </div>
      </div>` : ''}
      ${qualityAudit}
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">${patientQualityText('Per-module review metrics')}</div>
        <div class="panel-sub" style="margin-bottom:10px;">${t('Computed entity coverage uses the bounded review denominator. Inventory-only modules show row counts and are not painted as 0% coverage.', '已计算的实体覆盖率使用有界审阅分母。仅有文件清单的模块显示行数，不会被渲染成 0% 覆盖率。')}</div>
        ${drill.quality.map(q => {
          const hasCoverage = q.coverage_pct != null && Number.isFinite(Number(q.coverage_pct));
          const coverage = hasCoverage ? Math.max(0, Math.min(100, Number(q.coverage_pct))) : null;
          const metricKind = String(q.metric_kind || 'coverage');
          const metricLabel = metricKind === 'event_rate'
            ? t('event rate', '事件率')
            : (metricKind === 'exposure_rate' ? t('exposure rate', '暴露率') : t('coverage', '覆盖率'));
          const metric = hasCoverage
            ? `${fmtPct(coverage)} ${metricLabel}`
            : `${fmtInt(q.rows)} ${t('rows · coverage not computed', '行 · 未计算覆盖率')}`;
          return `
          <div class="qrow" data-patient-module-coverage="${hasCoverage ? 'computed' : 'not-computed'}"><span>${esc(q.module)}</span><div class="qbar ${hasCoverage && q.quality_status === 'ok' ? '' : (hasCoverage ? q.quality_status : 'neutral')}">${hasCoverage ? `<span style="width:${coverage}%"></span>` : ''}</div><span class="qv">${metric}</span></div>`;
        }).join('')}
      </div>
      ${patientQualityWorkbook(review)}
      ${topIssues.length ? `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">${patientQualityText('Top concept quality issues')}</div>
        <div class="table-wrap table-scroll">
          <table class="eu-table">
            <thead><tr><th>${t('Concept', '概念')}</th><th>${t('Module', '模块')}</th><th class="num">${t('Records', '记录')}</th><th class="num">${t('Missing', '缺失')}</th><th class="num">${t('Outlier', '异常值')}</th><th class="num">${t('Duplicate TS', '重复时间戳')}</th></tr></thead>
            <tbody>
              ${topIssues.map(row => `<tr><td class="key">${esc(row.feature)}</td><td>${esc(row.module)}</td><td class="num">${fmtInt(row.records)}</td><td class="num">${fmtPct(row.missing_pct)}</td><td class="num">${fmtPct(row.out_of_physio_pct)}</td><td class="num">${fmtPct(row.duplicate_time_pct)}</td></tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>` : ''}
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><div class="t">${boundedTitle}</div><div class="d">${boundedDetail}</div></div>
      </div>
      ${(drill.blocked_features || []).map(item => `
        <div class="note warn mt-12">
          <div class="ico">${icon('lock', 14)}</div>
          <div class="body"><span class="t">${esc(item.id)}</span> <span class="d" style="display:inline;">— ${esc(item.reason || 'blocked')}</span></div>
        </div>`).join('')}`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && Array.isArray(ws.quality)) {
      return `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">${t('Per-module stay-id presence', '各模块 stay-id 出现情况')}</div>
        ${ws.quality.map(q => {
          const hasCoverage = q.coverage_pct != null && Number.isFinite(Number(q.coverage_pct));
          const coverage = hasCoverage ? Math.max(0, Math.min(100, Number(q.coverage_pct))) : null;
          return `
          <div class="qrow" data-patient-module-coverage="${hasCoverage ? 'computed' : 'not-computed'}"><span>${esc(q.module || q.file)}</span><div class="qbar ${hasCoverage && q.status === 'ok' ? '' : (hasCoverage ? q.status : 'neutral')}">${hasCoverage ? `<span style="width:${coverage}%"></span>` : ''}</div><span class="qv">${hasCoverage ? fmtPct(coverage) : `${fmtInt(q.rows)} ${t('rows · coverage not computed', '行 · 未计算覆盖率')}`}</span></div>`;
        }).join('')}
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><div class="t">${t('Local export snapshot', '本地导出快照')}</div><div class="d">${t('Percentages are unique stay_id values found in each file divided by the loaded stay set. Event-only modules can be sparse by design; analysis gates still resolve denominators separately.', '百分比 = 各文件中出现的唯一 stay_id 数 ÷ 已载入的 stay 集合。仅事件类模块本就可能稀疏；分析闸门仍会单独解析分母。')}</div></div>
      </div>`;
    }
    const cov = [['Vitals', 98, 'ok'], ['Labs', 88, 'ok'], ['SOFA / SOFA-2', 94, 'ok'], ['Sepsis-3', 90, 'ok'], ['Fluids', 72, 'warn'], ['Ventilation', 58, 'bad']];
    return `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">${t('Per-concept coverage', '各概念覆盖率')}</div>
        ${cov.map(([n, pct, c]) => `
          <div class="qrow"><span>${n}</span><div class="qbar ${c === 'ok' ? '' : c}"><span style="width:${pct}%"></span></div><span class="qv">${pct}%</span></div>`).join('')}
      </div>
      <div class="note warn mt-16">
        <div class="ico">${icon('beaker', 16)}</div>
        <div class="body"><div class="t">${t('Ventilation coverage below threshold', '通气覆盖率低于阈值')}</div><div class="d">${t('Demo figures: 58% of stays have ventilation fields. In a real export the agent flags affected denominators before any analysis uses them.', '演示数值：58% 的 stay 具有通气字段。在真实导出中，代理会在任何分析使用这些分母前先行标记。')}</div></div>
      </div>`;
  }

  function patientTabBody() {
    if (patientTab !== 'series') {
      const chartOwner = window.EU_PATIENT_CHARTS;
      if (chartOwner && typeof chartOwner.dispose === 'function') chartOwner.dispose();
    }
    switch (patientTab) {
      case 'tables': return ptTables();
      case 'series': return ptSeries();
      case 'patient': return ptPatient();
      case 'quality': return ptQuality();
    }
  }
  function patientBrowseConfig() {
    return {
      drill: patientDrilldown,
      sourcePath: registryActivePath,
      repaint: () => repaintScreen('patient'),
      selectDemo(ref) {
        const payload = buildDemoPatientDrilldown(ref);
        window.EU_PATIENT_DRILLDOWN = payload;
        window.EU_VIZ_WORKSPACE = patientWorkspaceFromDrilldown(payload);
        patientView = 'loaded';
        repaintScreen('patient');
      },
    };
  }
  function resetPatientBrowseOwners() {
    const owner = window.EU_PATIENT_REVIEW || {};
    if (owner.navigation && owner.navigation.reset) owner.navigation.reset();
    if (owner.tables && owner.tables.reset) owner.tables.reset();
    if (owner.features && owner.features.reset) owner.features.reset();
  }
  function bindPatientTableControls(root) {
    const owner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.tables;
    if (owner && typeof owner.bind === 'function') owner.bind(root, patientBrowseConfig());
  }
  function bindPatientSeriesControls(root) {
    const chartOwner = window.EU_PATIENT_CHARTS;
    if (chartOwner && typeof chartOwner.mount === 'function') chartOwner.mount(root);
    const featureOwner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.features;
    if (featureOwner && typeof featureOwner.bind === 'function') {
      featureOwner.bind(root, patientBrowseConfig());
    }
    root.querySelectorAll('[data-patient-series-mode]').forEach(b => b.addEventListener('click', e => {
      e.preventDefault();
      const next = b.dataset.patientSeriesMode;
      if (!next || next === patientSeriesMode) return;
      patientSeriesMode = next;
      const body = document.querySelector('#ptbody');
      if (!body) {
        repaintScreen('patient');
        return;
      }
      body.innerHTML = patientTabBody();
      bindPatientTableControls(body);
      bindPatientEntitySelection(body);
      bindPatientSeriesControls(body);
    }));
  }
  function bindPatientEntitySelection(root) {
    const owner = window.EU_PATIENT_REVIEW && window.EU_PATIENT_REVIEW.navigation;
    if (owner && typeof owner.bind === 'function') owner.bind(root, patientBrowseConfig());
  }

  S.patient = {
    section: 'viz', nav: 'viz', sub: 'patient',
    crumbs: ['Home', 'Data Workspace','Patient Review'],
    get actionHtml() {
      // Topbar actions only exist once a workspace is loaded — before that the
      // page body owns the single primary action, and a context-free "Render"
      // button up here just reads as noise.
      return patientView === 'loaded'
        ? `<button class="btn" data-viz-reset>${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button><button class="btn primary" data-gen>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>`
        : '';
    },
    rail: () => vizRail('patient'),
    render() {
      if (window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.take) window.EU_GUIDED_HANDOFF.take('patient');
      const guidedNote = window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.noteHtml ? window.EU_GUIDED_HANDOFF.noteHtml('patient') : '';
      const dataMode = window.getDataMode
        ? window.getDataMode()
        : (window.EU_DATA === 'real' ? 'real' : 'demo');
      const realMode = dataMode === 'real';
      /* Every render branch below opens with this. Patient Review is a dense
         workspace whose visible title lives in the card eyebrow and the loaded
         bar, so the route heading is screen-reader-only — same treatment the
         Guided fullscreen route uses. It is not decoration: app.js resolves
         both the document title and the post-navigation focus target from the
         route's h1, and without one this screen rendered with no heading of
         any level at all. */
      const patientHeading = `<h1 class="shell-sr-only" tabindex="-1">${t('Patient Review', '患者审阅')}</h1>`;
      if (patientView === 'loading') {
        return `${patientHeading}${guidedNote}<div class="card pad">${skeletonWorkspace(window.EU_DATA)}</div>`;
      }
      if (patientView === 'loaded') {
        const drill = patientDrilldown();
        const ws = window.EU_VIZ_WORKSPACE;
        const s = drill ? drill.summary : (ws && ws.summary);
        const demoSourceOwner = window.EU_PATIENT_DEMO_SOURCES;
        const officialDemo = demoSourceOwner && demoSourceOwner.activeMetadata
          ? demoSourceOwner.activeMetadata(registrySources(), registryActivePath())
          : null;
        const readyTitle = drill
          ? (drill.demo
            ? t('Clinically constrained synthetic fallback ready', '临床约束合成兜底已就绪')
            : (officialDemo
              ? `${officialDemo.title} ${t('official demo ready', '官方演示已就绪')}`
              : t('Local export patient drilldown ready', '本地导出患者审阅已就绪')))
          : (ws ? t('Local export workspace ready', '本地导出工作区已就绪') : t('Demo review workspace ready', '演示审阅工作区已就绪'));
        const boundedReview = drill && s && s.review_scope === 'browser_bounded_entity_sample';
        const entityStats = boundedReview
          ? `${t('full cohort', '完整队列')} ${fmtInt(s.entities)} ${t('entities', '个实体')} · ${t('bounded browser review', '浏览器有界审阅')} ${fmtInt(s.review_entities)} ${t('entities', '个实体')}`
          : `${fmtInt(s && (s.entities != null ? s.entities : s.stays))} ${drill ? t('entities', '个实体') : t('stays', '次住院')}`;
        const loadedFeatureCount = drill && drill.data_tables && drill.data_tables.loaded_summary
          ? drill.data_tables.loaded_summary.review_features
          : null;
        const readyStats = s
          ? (drill && drill.demo
            ? `${fmtInt(s.entities)} ${t('synthetic entities', '个合成实体')} · ${fmtInt(s.modules)} ${t('modules', '个模块')} · ${fmtInt(loadedFeatureCount)} ${t('catalog features', '个目录特征')}`
            : `${entityStats} · ${fmtInt(s.modules)} ${t('modules', '个模块')} · ${fmtInt(s.total_rows)} ${t('rows', '行')}`)
          : `48 ${t('synthetic entities', '个合成实体')} · 19 ${t('modules', '个模块')} · ${t('catalog features', '目录特征')}`;
        const demoLoadedNote = (drill && drill.demo) ? `
        <div class="note warn mt-12">
          <div class="ico">${icon('beaker', 16)}</div>
          <div class="body">
            <div class="t">${t('Clinically constrained synthetic fallback', '临床约束合成兜底')}</div>
            <div class="d">${t('This offline fallback uses deterministic, correlated ICU trajectories derived from one synthetic phenotype model. It contains no real records and must not be used for clinical inference or manuscript results.', '这个离线兜底使用由同一合成表型模型派生的确定性相关 ICU 轨迹；其中不含真实记录，不得用于临床推断或稿件结果。')}</div>
          </div>
          <button class="btn sm" data-patient-use-real>${icon('db', 13)} ${t('Use real export', '使用真实导出')}</button>
        </div>` : officialDemo ? `
        <div class="note info mt-12" data-patient-official-demo="${esc(officialDemo.source_id)}">
          <div class="ico">${icon('db', 16)}</div>
          <div class="body">
            <div class="t">${esc(officialDemo.title)} · ${esc(officialDemo.version)}</div>
            <div class="d">${t('This Patient Review is backed by official deidentified demo records processed through the normal EasyICU conversion and concept-mapping pipeline.', '当前患者审阅来自官方去标识化 Demo 记录，并经过 EasyICU 正常的转换与概念映射流程。')} ${esc(officialDemo.provider)} · ${esc(officialDemo.license)}</div>
          </div>
        </div>` : (!drill && !ws) ? `
        <div class="note warn mt-12">
          <div class="ico">${icon('beaker', 16)}</div>
          <div class="body">
            <div class="t">${t('Synthetic fallback workspace', '合成兜底工作区')}</div>
            <div class="d">${t('This tab is showing synthetic UI data. Load an official demo or local export for source-backed module tables, feature matrices, and quality metrics.', '此标签页显示合成界面数据。请加载官方演示数据或本地导出，以查看有来源支撑的模块表、特征矩阵和质量指标。')}</div>
          </div>
          <button class="btn sm" data-patient-use-real>${icon('db', 13)} ${t('Use real export', '使用真实导出')}</button>
        </div>` : '';
        return `
        ${patientHeading}
        <div class="loaded-bar">
          <span class="pill ok"><span class="dot"></span>${t('Loaded', '已加载')}</span>
          <div class="grow"><span style="font-weight:600;font-size:13px;">${readyTitle}</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${readyStats}</span></div>
          <button class="btn sm" data-viz-reset>${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button>
          <button class="btn sm" data-patient-export>${icon('download', 13)} ${t('Export', '导出')}</button>
        </div>
        ${guidedNote}
        ${demoLoadedNote}
        <div class="mt-16">${patientTabs()}</div>
        <div id="ptbody">${patientTabBody()}</div>
        <div class="nextbar accent mt-16">
          <div class="nb-ico">${icon('arrow', 16)}</div>
          <div class="grow"><div class="nb-t">${t('Reviewed the data — what\u2019s next?', '\u6570\u636e\u5df2\u5ba1\u9605 \u2014\u2014 \u4e0b\u4e00\u6b65\uff1f')}</div><div class="nb-d">${t('Compare groups in Cohort Statistics, or ask Guided Copilot to assemble an auditable analysis and review-ready draft.', '\u5728\u300c\u961f\u5217\u7edf\u8ba1\u300d\u505a\u7ec4\u95f4\u5bf9\u6bd4\uff0c\u6216\u8ba9\u300c\u7814\u7a76\u5f15\u5bfc\u300d\u7ec4\u88c5\u53ef\u5ba1\u8ba1\u5206\u6790\u4e0e\u5f85\u6838\u9a8c\u8349\u7a3f\u3002')}</div></div>
          <button class="btn" data-nav="cohort">${icon('cohort', 13)} ${t('Cohort Statistics', '\u961f\u5217\u7edf\u8ba1')}</button>
          <button class="btn primary" data-study-handoff data-study-source="patient" data-study-target="guided">${icon('agent', 13)} ${t('Continue in Guided Copilot','\u5728\u7814\u7a76\u5f15\u5bfc\u4e2d\u7ee7\u7eed')}</button>
        </div>`;
      }
      /* idle */
      const demoSourceOwner = window.EU_PATIENT_DEMO_SOURCES;
      const officialDemoSources = demoSourceOwner && typeof demoSourceOwner.render === 'function'
        ? demoSourceOwner.render({ t, esc })
        : `<div class="note warn"><div class="body"><div class="d">${t('Official demo-source controls are unavailable; the clinically constrained synthetic fallback remains usable.', '官方演示数据源控件暂不可用；仍可使用带临床约束的合成兜底。')}</div></div><button class="btn sm" data-gen>${t('Load synthetic fallback', '加载合成兜底')}</button></div>`;
      return `
      ${patientHeading}
      ${guidedNote}
      <div class="card pad">
        <div class="panel-head">
          <div>
            <div class="eyebrow">${t('Patient Review', '患者审阅')}</div>
            <!-- h2, not div: this is the section title under the route's h1.
                 margin-bottom is zeroed because there is no global heading
                 reset, and the UA default would otherwise add ~14px here. -->
            <h2 class="panel-title" style="margin-top:4px;margin-bottom:0;font-size:17px;">${t('Load a review workspace', '加载审阅工作区')}</h2>
            <div class="panel-sub">${realMode ? t('Load a local EasyICU export folder. Nothing is uploaded.', '加载本地 EasyICU 导出文件夹，不上传任何数据。') : t('Choose an official deidentified ICU demo, or use the offline synthetic fallback for interaction rehearsal only.', '选择官方去标识化 ICU 演示数据，或仅在离线界面演练时使用合成兜底。')}</div>
          </div>
        </div>

        <div style="border-top:1px solid var(--hair);padding-top:16px;">
          <div class="eyebrow" style="margin-bottom:10px;">${t('Data source', '数据源')}</div>
          <div class="radio-row">
            <label class="radio ${realMode ? 'on' : ''}" role="button" tabindex="0" data-datamode="real"><span class="mk"></span> ${t('Previously exported data', '此前导出的数据')}</label>
            <label class="radio ${!realMode ? 'on' : ''}" role="button" tabindex="0" data-datamode="demo"><span class="mk"></span> ${t('Demo data', '演示数据')}</label>
          </div>
        </div>

        ${realMode ? `
        <div class="card sunken pad mt-16">
          <div class="eyebrow" style="margin-bottom:4px;">${t('Local export', '本地导出')}</div>
          <div style="font-weight:600;font-size:14px;">${t('Load exported EasyICU tables', '加载已导出的 EasyICU 数据表')}</div>
          <div class="panel-sub" style="margin-top:2px;">${t('Pick a registered local export, or add one by path.', '选择已注册的本地导出，或按路径添加一个。')}</div>
          ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
          ${patientSourceReadyCard()}
          ${sourceRegistryBlock('single')}
          <p style="font-size:11.5px;color:var(--ink-4);margin:14px 0 0;">${t('Use Data Extraction first to create or refresh this folder. The last successful export is remembered locally.', '请先用数据抽取创建或刷新该文件夹。上次成功的导出会被本地记住。')}</p>
          <button class="btn primary block lg mt-16" data-gen>${icon('folder', 14)} ${t('Load local export', '加载本地导出')}</button>
        </div>` : `
        <div class="card sunken pad mt-16">${officialDemoSources}</div>`}
      </div>

      <div class="empty mt-16">
        <div class="glyph">${icon('viz', 22)}</div>
        <div class="t">${t('Preview workspace awaits data', '预览工作区等待数据')}</div>
        <div class="d">${t('Generate demo data or load exported files above; the review tabs will appear here as a compact multi-view workspace.', '在上方生成演示数据或加载导出文件；审阅标签页会作为紧凑的多视图工作区出现在这里。')}</div>
      </div>`;
    },
    afterRender(root) {
      const dataMode = window.getDataMode
        ? window.getDataMode()
        : (window.EU_DATA === 'real' ? 'real' : 'demo');
      const realMode = dataMode === 'real';
      bindSourceRegistry(root, 'patient');
      if (realMode && patientView === 'idle' && !window.EU_PATIENT_SOURCES && !window.EU_PATIENT_SOURCES_LOADING) {
        loadPatientSources(ok => {
          if (ok && window.location.hash === '#patient' && patientView === 'idle') repaintScreen('patient');
        });
      }
      const demoSourceOwner = window.EU_PATIENT_DEMO_SOURCES;
      if (!realMode && patientView === 'idle' && demoSourceOwner) {
        demoSourceOwner.ensureLoaded(() => {
          if (window.location.hash === '#patient' && patientView === 'idle' && window.getDataMode() !== 'real') {
            repaintScreen('patient');
          }
        });
        demoSourceOwner.bind(root, {
          refresh: () => {
            if (window.location.hash === '#patient' && patientView === 'idle') repaintScreen('patient');
          },
          openPrepared: sourceId => {
            if (patientView === 'loading') return;
            const source = demoSourceOwner.rememberOpened && demoSourceOwner.rememberOpened(sourceId);
            if (!source || !source.status || !source.status.active) {
              vizErr = t(
                'The selected official demo is not active yet. Activate it before opening.',
                '所选官方演示尚未激活，请先激活后再打开。',
              );
              repaintScreen('patient');
              return;
            }
            resetPatientBrowseOwners();
            patientView = 'loading';
            window.EU_DATA = 'real';
            window.EU_VIZ_WORKSPACE = null;
            window.EU_PATIENT_DRILLDOWN = null;
            window.EU_PATIENT_SOURCES = null;
            vizErr = null;
            repaintScreen('patient');
            const hydration = window.EU_API && window.EU_API.hydrateWorkspaceRegistry
              ? window.EU_API.hydrateWorkspaceRegistry()
              : Promise.resolve();
            Promise.resolve(hydration).then(() => {
              loadRealPatient(ok => {
                patientView = ok ? 'loaded' : 'idle';
                repaintScreen('patient');
              });
            }).catch(error => {
              vizErr = String((error && error.message) || error);
              patientView = 'idle';
              repaintScreen('patient');
            });
          },
        });
      }
      root.querySelectorAll('.radio[data-datamode]').forEach(b => b.addEventListener('keydown', e => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        e.preventDefault();
        if (window.setDataMode) window.setDataMode(b.dataset.datamode);
      }));
      root.querySelectorAll('[data-gen]').forEach(b => b.addEventListener('click', () => {
        if (patientView === 'loading') return;
        const rerunningLoadedWorkspace = patientView === 'loaded';
        const useSourceBackedPipeline = window.EU_DATA === 'real'
          && (realMode || rerunningLoadedWorkspace);
        resetPatientBrowseOwners();
        patientView = 'loading';
        repaintScreen('patient');
        if (useSourceBackedPipeline) {
          loadRealPatient(ok => { patientView = ok ? 'loaded' : 'idle'; repaintScreen('patient'); });
        } else {
          if (window.setDataModeContext) window.setDataModeContext(null);
          window.EU_DATA = 'demo';
          try { localStorage.setItem('easyicu_home_data', 'demo'); } catch (e) {}
          setTimeout(() => {
            const payload = buildDemoPatientDrilldown();
            window.EU_PATIENT_DRILLDOWN = payload;
            window.EU_VIZ_WORKSPACE = patientWorkspaceFromDrilldown(payload);
            patientView = 'loaded';
            window.EU_HASWORK = true;
            vizErr = null;
            repaintScreen('patient');
          }, 450);
        }
      }));
      root.querySelectorAll('[data-viz-reset]').forEach(b => b.addEventListener('click', () => {
        patientView = 'idle'; resetPatientBrowseOwners(); window.EU_VIZ_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; repaintScreen('patient');
      }));
      root.querySelectorAll('[data-patient-use-real]').forEach(b => b.addEventListener('click', () => {
        patientView = 'idle';
        resetPatientBrowseOwners();
        if (window.setDataModeContext) window.setDataModeContext(null);
        window.EU_DATA = 'real';
        window.EU_VIZ_WORKSPACE = null;
        window.EU_PATIENT_DRILLDOWN = null;
        window.EU_PATIENT_SOURCES = null;
        try { localStorage.setItem('easyicu_home_data', 'real'); } catch (e) {}
        repaintScreen('patient');
      }));
      root.querySelectorAll('[data-patient-export]').forEach(b => b.addEventListener('click', () => {
        const payload = patientDrilldown();
        if (!payload) {
          vizErr = 'No Patient Review payload is loaded yet.';
          repaintScreen('patient');
          return;
        }
        downloadJsonFile('easyicu-patient-review-drilldown.json', {
          exported_at: new Date().toISOString(),
          payload_scope: 'bounded_patient_review_drilldown',
          patient_review: payload,
        });
      }));
      const tabsEl = root.querySelector('#ptabs');
      if (tabsEl) tabsEl.addEventListener('click', e => {
        const b = e.target.closest('[data-ptab]'); if (!b) return;
        patientTab = b.dataset.ptab;
        tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.ptab === patientTab));
        const body = root.querySelector('#ptbody');
        body.innerHTML = patientTabBody();
        bindPatientTableControls(body);
        bindPatientEntitySelection(body);
        bindPatientSeriesControls(body);
      });
      const patientBody = root.querySelector('#ptbody') || root;
      bindPatientTableControls(patientBody);
      bindPatientEntitySelection(patientBody);
      bindPatientSeriesControls(patientBody);
    },
  };


  function resetForSourceChange() {
    patientView = 'idle';
    vizErr = null;
  }

  function resetForDataMode() {
    resetPatientBrowseOwners();
    patientView = 'idle';
    vizErr = null;
    window.EU_PATIENT_SOURCES = null;
  }

  function presetLoaded() {
    patientView = 'loaded';
  }

  function hydrate(payload) {
    window.EU_DATA = 'real';
    patientView = 'loaded';
    window.EU_PATIENT_DRILLDOWN = payload;
    window.EU_VIZ_WORKSPACE = patientWorkspaceFromDrilldown(payload);
    vizErr = null;
    window.EU_HASWORK = true;
    const reviewOwner = window.EU_PATIENT_REVIEW || {};
    if (reviewOwner.navigation && reviewOwner.navigation.prime) reviewOwner.navigation.prime(payload);
    if (reviewOwner.tables && reviewOwner.tables.prime) reviewOwner.tables.prime(payload);
    if (reviewOwner.features && reviewOwner.features.prime) reviewOwner.features.prime(payload);
  }

  window.EU_VIZ_PATIENT = {
    init(bindings) { host = Object.assign({}, host, bindings || {}); },
    activeSourceMeta: patientActiveSourceMeta,
    drilldown: patientDrilldown,
    hydrate,
    presetLoaded,
    resetForDataMode,
    resetForSourceChange,
    seriesLabel: patientSeriesLabel,
    signalKey: ptSignalKey,
    signalLabel: patientSignalLabel,
    setError(value) { vizErr = value == null ? null : String(value); },
    state() { return { view: patientView, tab: patientTab, seriesMode: patientSeriesMode, error: vizErr }; },
    workspaceFromDrilldown: patientWorkspaceFromDrilldown,
  };
})();
