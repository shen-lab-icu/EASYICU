/* Screens: Data Visualization — Patient Review, Cohort Statistics, Cross-DB Benchmark */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  // demo/fixture data layer lives in screens-viz-demo.js (loaded first);
  // rebind its exports so existing call sites stay unchanged.
  const {
    catalogModuleLabel, catalogFeatureMeta,
    DEMO_ENTITY_COUNT, DEMO_DURATION_HOURS, DEMO_CHART_HOURS, DEMO_CLINICAL_LANES, DEMO_THRESHOLDS,
    demoCatalogModules, demoIsTimeIndexed, demoFeatureModule, demoCoverageForFeature,
    demoRowsForModule, demoReviewStatus, demoQualityStatus, demoRateTone,
    demoThresholds, demoBaseValue, demoTableValue, demoCharttimeAt, demoSignal, demoTimeLanes, demoSignalDelta,
    demoFeatureTone, demoCategorySection, demoQualityPanelRows,
  } = window.VIZ_DEMO;

  function vizRail(active) {
    const real = window.EU_DATA === 'real';
    const xdb = active === 'crossdb' ? window.EU_CROSSDB_WORKSPACE : null;
    const drill = active === 'patient' ? patientDrilldown() : null;
    const cohort = active === 'cohort' ? cohortReview() : null;
    const ws = window.EU_VIZ_WORKSPACE;
    const label = real ? t('Real', '真实') : t('Demo', '演示');
    const xdbRaw = xdb && xdb.source_type === 'raw_database_root';
    const xdbDemo = xdb && xdb.source_type === 'legacy_simulated_multidb_feature_frames';
    let dataset;
    let cohortLine;
    let variables;
    if (xdb) {
      dataset = `${fmtInt(xdb.source_count)} ${xdbRaw || xdbDemo ? t('databases', '个数据库') : t('exports', '个导出')}`;
      cohortLine = xdbRaw ? t('raw feature densities', '原始特征密度') : (xdbDemo ? t('seeded simulated densities', '种子模拟密度') : t('matched exports required', '需要匹配导出'));
      variables = `${fmtInt((xdb.shared_modules || []).length)} ${t('shared modules', '个共享模块')}`;
    } else if (drill) {
      const loaded = (drill.data_tables || {}).loaded_summary || {};
      dataset = (drill.source || {}).label || (drill.demo ? t('Demo · EasyICU catalog', '演示 · EasyICU 字典') : t('Local export', '本地导出'));
      cohortLine = drill.demo
        ? `${fmtInt(drill.summary && drill.summary.entities)} ${t('seeded entities', '个种子实体')}`
        : `${fmtInt(drill.summary && drill.summary.entities)} ${t('entities', '个实体')}`;
      variables = drill.demo
        ? `${fmtInt(drill.summary && drill.summary.modules)} ${t('modules', '个模块')} · ${fmtInt(loaded.review_features)} ${t('features', '个特征')}`
        : `${fmtInt(drill.summary && drill.summary.modules)} ${t('modules', '个模块')}`;
    } else if (cohort) {
      const fsel = cohort.feature_selection || {};
      dataset = (cohort.source || {}).label || t('Local export', '本地导出');
      cohortLine = `${fmtInt(cohort.summary && cohort.summary.cohort_size)} ${t('entities', '个实体')}`;
      variables = `${fmtInt(cohort.summary && cohort.summary.modules)} ${t('modules', '个模块')} · ${fmtInt(fsel.selected_count)} / ${fmtInt(fsel.available_count)} ${t('features', '个特征')}`;
    } else if (ws) {
      dataset = (ws.path || '').split('/').filter(Boolean).slice(-2).join('/') || t('Local export', '本地导出');
      cohortLine = `${fmtInt(ws.summary && ws.summary.stays)} ${t('stays', '次住院')}`;
      variables = `${fmtInt(ws.summary && ws.summary.modules)} ${t('modules', '个模块')}`;
    } else {
      const cat = window.EU_CATALOG || {};
      const moduleCount = Array.isArray(cat.groups) ? cat.groups.length : 19;
      const featureCount = cat.totalConcepts || Object.values(cat.groupConcepts || {}).reduce((a, b) => a + (Array.isArray(b) ? b.length : 0), 0) || 247;
      if (!real && active === 'cohort' && cohortView === 'loaded') {
        const scope = cohortDemoCatalogScope();
        dataset = t('Demo · EasyICU catalog', '演示 · EasyICU 字典');
        cohortLine = `10 ${t('stays', '次住院')}`;
        variables = `${fmtInt(scope.selectedModuleCount)} / ${fmtInt(scope.totalModuleCount || moduleCount)} ${t('modules', '个模块')} · ${fmtInt(scope.selectedFeatureCount)} / ${fmtInt(scope.totalFeatureCount || featureCount)} ${t('features', '个特征')}`;
      } else {
        dataset = real ? t('No export loaded', '尚未加载导出') : t('Demo · EasyICU catalog', '演示 · EasyICU 字典');
        cohortLine = real ? t('load exported tables', '加载导出表') : `${DEMO_ENTITY_COUNT} ${t('seeded entities', '个种子实体')}`;
        variables = real ? t('from export manifest', '来自导出清单') : `${fmtInt(moduleCount)} ${t('modules', '个模块')} · ${fmtInt(featureCount)} ${t('features', '个特征')}`;
      }
    }
    return `
    <div class="rail-sep"></div>
    <div class="rail-block">
      <div class="rail-head"><span class="t">${t('Current setup', '当前配置')}</span><span class="pill ${real ? 'ok' : 'demo'}" style="height:20px;"><span class="dot"></span>${label}</span></div>
      <div class="setup-row"><span class="k">${t('Dataset', '数据集')}</span><span class="vv">${esc(dataset)}</span></div>
      <div class="setup-row"><span class="k">${t('Cohort', '队列')}</span><span class="vv">${cohortLine}</span></div>
      <div class="setup-row"><span class="k">${t('Variables', '变量')}</span><span class="vv">${variables}</span></div>
      <button class="btn sm block" data-viz-reset style="margin-top:12px;">${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button>
    </div>`;
  }

  /* view state for the interactive viz screens */
  let patientView = 'idle';   // idle | loading | loaded
  let patientTab = 'tables';
  let patientTableModule = null;
  let patientTablePage = 1;
  let patientTablePageSize = 24;
  let patientSeriesMode = 'lanes';
  let crossView = 'idle';     // idle | loading | loaded
  let crossDensityModule = 'all';
  let crossDensityFeature = null;
  let crossDensityScope = 'core'; // core | all — restore the old curated "one subplot per canonical concept" default
  // Canonical clinical concepts of the legacy Figure-3 Cross-DB panel (paper_figures._render_paper_crossdb_panel).
  // The native grid otherwise dumps all ~247 catalog features; the curated default keeps the old small-multiples look.
  const CROSS_DENSITY_CANON = ['hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2', 'crea', 'lact', 'wbc', 'plt', 'gluc'];
  let crossRawES = null;
  let crossRawJobId = null;
  let crossRawProg = null;
  let crossRawCancelRequested = false;
  let crossRawJobStarting = false;
  let crossRawRootDraft = '';
  let crossRawRootScan = null;
  let crossRawRootScanPath = '';
  let crossRawRootScanning = false;
  let crossRawSampleMode = 'quick';
  let cohortView = 'idle';    // idle | loaded | loading
  let cohortPanel = 'groups'; // groups | coverage | snapshot | sofa
  let cohortCompare = 'outcome';
  let cohortFeatureScope = 'recommended'; // recommended | all
  let cohortFeatureModule = 'all';
  let cohortSelectedFeatures = [];
  let cohortSurvivalOutcome = 'mort_28d';
  let cohortSurvivalGroup = 'sepsis';
  let cohortSofaMatrixMode = 'pct'; // pct | count
  let cohortSofaMatrixGranularity = 'medium'; // coarse | medium | fine | exact
  let vizErr = null;

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }
  function fmtInt(v) { return v == null ? '—' : Number(v).toLocaleString(); }
  function fmtNum(v, digits = 1) {
    if (v == null || Number.isNaN(Number(v))) return '—';
    return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  function fmtPct(v) { return v == null ? '—' : `${fmtNum(v, 1)}%`; }
  function fmtP(v) {
    if (v == null || Number.isNaN(Number(v))) return '—';
    const n = Number(v);
    if (n > 0 && n < 0.001) {
      const exponent = Math.floor(Math.log10(n));
      const mantissa = n / Math.pow(10, exponent);
      return `${mantissa.toLocaleString(undefined, { maximumSignificantDigits: 4 })} × 10^${exponent}`;
    }
    return n.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  function downloadJsonFile(filename, payload) {
    const blob = new Blob([JSON.stringify(payload || {}, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'easyicu-patient-review.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function registrySources() {
    const reg = window.EU_SOURCES && window.EU_SOURCES.registry ? window.EU_SOURCES.registry() : (window.EU_WORKSPACE_REGISTRY || {});
    return (reg.sources || []).filter(s => s && s.ok && s.path);
  }
  function registryActivePath() {
    if (window.EU_SOURCES && window.EU_SOURCES.activePath) return window.EU_SOURCES.activePath();
    const reg = window.EU_WORKSPACE_REGISTRY || {};
    return reg.active_path || null;
  }
  function registryCrossdbPaths() {
    if (window.EU_SOURCES && window.EU_SOURCES.crossdbPaths) return window.EU_SOURCES.crossdbPaths();
    const reg = window.EU_WORKSPACE_REGISTRY || {};
    return Array.isArray(reg.crossdb_paths) ? reg.crossdb_paths : [];
  }
  function defaultExportPath() {
    if (window.EU_LAST_EXPORT && window.EU_LAST_EXPORT.out_dir) return window.EU_LAST_EXPORT.out_dir;
    const active = registryActivePath();
    if (active) return active;
    try {
      const v = localStorage.getItem('easyicu_last_export_dir');
      if (v) return v;
    } catch (e) {}
    return '';
  }
  function defaultCrossdbPaths() {
    const paths = [];
    registryCrossdbPaths().forEach(p => { if (p) paths.push(String(p)); });
    try {
      const raw = localStorage.getItem('easyicu_crossdb_export_dirs');
      if (raw) {
        const parsed = raw.trim().startsWith('[') ? JSON.parse(raw) : raw.split(/[,\n;]/);
        if (Array.isArray(parsed)) parsed.forEach(p => { if (p) paths.push(String(p)); });
      }
    } catch (e) {}
    const last = defaultExportPath();
    if (last) paths.push(last);
    return Array.from(new Set(paths.map(p => p.trim()).filter(Boolean)));
  }
  function defaultRawCrossdbRoot() {
    return crossRawRootDraft || '';
  }
  function selectedCrossDbKeys() {
    return CROSS_DBS.filter(d => d[1]).map(d => d[2]);
  }
  function selectedCrossDbCount() {
    return CROSS_DBS.filter(d => d[1]).length;
  }
  function crossRawSampleProfiles() {
    return [
      {
        id: 'quick',
        label: t('Quick preview', '快速预览'),
        note: t('Fast first look for module-level density checks.', '优先快速看模块级分布。'),
        maxPatients: 200,
        sampleSize: 600,
      },
      {
        id: 'standard',
        label: t('Standard sample', '标准抽样'),
        note: t('Balanced default for smoother density curves.', '平衡速度和曲线稳定性。'),
        maxPatients: 300,
        sampleSize: 1500,
      },
      {
        id: 'deeper',
        label: t('Deeper sample', '较深抽样'),
        note: t('More stable, but can take longer on six raw databases.', '更稳定，但六库原始数据会更慢。'),
        maxPatients: 800,
        sampleSize: 3000,
      },
    ];
  }
  function crossRawSampleProfile() {
    return crossRawSampleProfiles().find(row => row.id === crossRawSampleMode)
      || crossRawSampleProfiles()[0];
  }
  function crossRawSampleSummary(profile) {
    const p = profile || crossRawSampleProfile();
    return `${p.label} · ≤${fmtInt(p.maxPatients)} ${t('entities/db', '实体/库')} · ≤${fmtInt(p.sampleSize)} ${t('values/feature', '值/特征')}`;
  }
  function crossRawDbLabel(dbKey) {
    const row = CROSS_DBS.find(d => d[2] === dbKey);
    return row ? row[0] : String(dbKey || '');
  }
  function crossRawPathValue(path) {
    return String(path || '').trim();
  }
  function invalidateCrossRawRootScan() {
    crossRawRootScan = null;
    crossRawRootScanPath = '';
    crossRawRootScanning = false;
  }
  function crossRawSelectionStatusFor(path) {
    const rawRoot = crossRawPathValue(path);
    const current = rawRoot && crossRawRootScan && crossRawRootScanPath === rawRoot
      ? crossRawRootScan
      : null;
    const selectedKeys = selectedCrossDbKeys();
    const detectedKeys = new Set(
      current && current.ok
        ? (current.detected || []).map(row => row && row.key).filter(Boolean)
        : []
    );
    const detectedSelectedKeys = selectedKeys.filter(key => detectedKeys.has(key));
    const missingSelectedKeys = selectedKeys.filter(key => !detectedKeys.has(key));
    return {
      current,
      selectedKeys,
      detectedKeys,
      detectedSelectedKeys,
      missingSelectedKeys,
      runnable: !!(rawRoot && current && current.ok && detectedSelectedKeys.length >= 2),
    };
  }
  function crossRawScanReadyFor(path) {
    return crossRawSelectionStatusFor(path).runnable;
  }
  function crossRawScanCurrentFor(path) {
    const rawRoot = crossRawPathValue(path);
    return !!(rawRoot && crossRawRootScan && crossRawRootScanPath === rawRoot);
  }
  function scanCrossdbRawRoot(path) {
    const rawRoot = crossRawPathValue(path || defaultRawCrossdbRoot());
    crossRawRootDraft = rawRoot;
    if (!rawRoot) {
      invalidateCrossRawRootScan();
      vizErr = t('Choose a local ICU data root before checking Cross-DB folders.', '检查跨库文件夹前，请先选择本地 ICU 数据根目录。');
      repaintScreen('crossdb');
      return Promise.resolve(false);
    }
    if (!window.EU_API || !window.EU_API.scanCrossdbRawRoot) {
      invalidateCrossRawRootScan();
      vizErr = t('Raw Cross-DB folder check API is unavailable in this browser session.', '当前浏览器会话无法使用原始跨库文件夹检查 API。');
      repaintScreen('crossdb');
      return Promise.resolve(false);
    }
    crossRawRootScanning = true;
    crossRawRootScanPath = rawRoot;
    vizErr = null;
    repaintScreen('crossdb');
    return window.EU_API.scanCrossdbRawRoot({
      data_root: rawRoot,
      databases: selectedCrossDbKeys(),
    }).then(scan => {
      crossRawRootScanning = false;
      crossRawRootScan = scan || null;
      crossRawRootScanPath = rawRoot;
      if (scan && scan.ok === false) {
        vizErr = scan.hint || scan.error || 'Could not check that raw ICU data root.';
      } else {
        vizErr = null;
      }
      repaintScreen('crossdb');
      return crossRawScanReadyFor(rawRoot);
    }).catch(err => {
      crossRawRootScanning = false;
      crossRawRootScan = null;
      crossRawRootScanPath = rawRoot;
      vizErr = String(err && err.message || err);
      repaintScreen('crossdb');
      return false;
    });
  }
  function teardownCrossRawES() {
    if (crossRawES) {
      try { crossRawES.close(); } catch (e) {}
    }
    crossRawES = null;
  }
  function cancelCrossRawJob() {
    if (crossRawCancelRequested) return;
    const jobId = crossRawJobId;
    crossRawCancelRequested = true;
    crossRawJobStarting = false;
    crossRawJobId = null;
    crossRawProg = null;
    teardownCrossRawES();
    crossView = 'idle';
    vizErr = t('Raw Cross-DB density job cancellation requested.', '已请求取消原始跨库密度任务。');
    repaintScreen('crossdb');
    if (!jobId || !window.EU_API || !window.EU_API.postJSON) return;
    window.EU_API.postJSON('/api/jobs/' + jobId + '/cancel', { reason: 'user_requested' })
      .catch(err => {
        vizErr = String(err && err.message || err);
        repaintScreen('crossdb');
      });
  }
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
    const previews = patientTablePreviews(payload);
    if (!previews.length) return null;
    const tables = (payload || patientDrilldown() || {}).data_tables || {};
    const fallback = (tables.module_picker || {}).default_module || (previews[0] && previews[0].module);
    if (!patientTableModule || !previews.some(row => row.module === patientTableModule)) {
      patientTableModule = fallback;
    }
    return previews.find(row => row.module === patientTableModule) || previews[0] || null;
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
  function demoTablePreviewRowContext(rowIndex, timeIndexed) {
    const idx = Math.max(0, Number(rowIndex) || 0);
    if (!timeIndexed) {
      return {
        entityIndex: idx,
        timeIndex: 0,
        entityRef: `demo_ent_${idx + 1}`,
        charttime: null,
        valueSeed: idx,
      };
    }
    const timepointsPerEntity = 12;
    const entityIndex = Math.floor(idx / timepointsPerEntity);
    const timeIndex = idx % timepointsPerEntity;
    return {
      entityIndex,
      timeIndex,
      entityRef: `demo_ent_${entityIndex + 1}`,
      charttime: demoCharttimeAt(timeIndex),
      valueSeed: entityIndex * 13 + timeIndex,
    };
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
            <div class="patient-flow-title">${esc(title)}</div>
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
    const entities = Array.isArray(drill && drill.entities) ? drill.entities : [];
    if (!entities.length) return '';
    const selectedRef = selected && selected.ref;
    const title = opts.title || t('Case navigator', '病例导航');
    const detail = opts.detail || t('Switch entity once; tables, trends, overview, and quality keep the same pseudonymous case context.', '切换一次实体后，数据表、趋势、概览和质量页都会使用同一个去标识病例上下文。');
    return `
      <div class="pt-entity-nav mt-16">
        <div>
          <div class="eyebrow">${esc(title)}</div>
          <div class="pt-entity-detail">${esc(detail)}</div>
        </div>
        <div class="pt-entity-chiprow">
          ${entities.map(item => `<button type="button" class="chip ${item.ref === selectedRef ? 'solid' : ''}" data-patient-entity="${esc(item.ref)}" style="${item.ref === selectedRef ? 'border-color:var(--ink);color:var(--ink);' : ''}">${esc(item.label || item.ref)}</button>`).join('')}
        </div>
      </div>`;
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
      'Clinical lanes': t('Clinical lanes', '临床泳道'),
      'Clinical Lanes': t('Clinical lanes', '临床泳道'),
      'Feature matrices': t('Feature matrices', '特征矩阵'),
      'Feature Matrix': t('Feature Matrix', '特征矩阵'),
      'Review mode': t('Review mode', '审阅模式'),
      'Single Patient': t('Single Patient', '单患者'),
      'Multi-Patient Comparison': t('Multi-Patient Comparison', '多患者同特征对比'),
    };
    return mapped[label] || label;
  }
  function patientSeriesDetail(value) {
    let detail = String(value || '');
    const legacyAggregateDetail = ['time windows x features', 'single entity', ['aggregate', 'comparison'].join(' ')].join(' / ');
    detail = detail.replace(legacyAggregateDetail, 'clinical lanes / single entity / same-feature comparison');
    detail = detail.replace('catalog lane signals', 'catalog signals');
    detail = detail.replace('matrix groups available', 'lanes available');
    if (window.EU_LANG === 'zh') {
      detail = detail.replace('clinical lanes / single entity / same-feature comparison', '临床泳道 / 单实体 / 同特征对比');
      detail = detail.replace('catalog signals', '目录信号');
      detail = detail.replace('lanes available', '条临床泳道可用');
      detail = detail.replace('selected-entity signals', '个已选实体信号');
      detail = detail.replace('pseudonymous options exposed', '个去标识实体选项');
    }
    return detail;
  }
  function cohortReview() {
    return window.EU_COHORT_REVIEW || null;
  }
  function cohortLoaded() {
    const review = cohortReview();
    return cohortView === 'loaded' && (window.EU_DATA !== 'real' || !!(review && review.summary));
  }
  function reloadStaleRealCohortIfNeeded(review) {
    if (window.EU_DATA !== 'real' || cohortView !== 'loaded' || (review && review.summary)) return false;
    if (!registryActivePath()) {
      cohortView = 'idle';
      return false;
    }
    cohortView = 'loading';
    setTimeout(() => loadRealCohort(ok => { cohortView = ok ? 'loaded' : 'idle'; repaintScreen('cohort'); }), 0);
    return true;
  }
  function resetCohortFeatureSelection() {
    cohortFeatureModule = 'all';
    cohortSelectedFeatures = [];
  }
  function cohortSelectedFeatureIds(review) {
    const selected = ((review || cohortReview() || {}).feature_selection || {}).selected || [];
    return selected.map(row => row && row.id).filter(Boolean);
  }
  function syncCohortFeatureSelection(payload) {
    const ids = cohortSelectedFeatureIds(payload);
    if (ids.length) cohortSelectedFeatures = ids;
  }
  function reloadCohortForFeatureSelection() {
    if (window.EU_DATA !== 'real') {
      repaintScreen('cohort');
      return;
    }
    cohortView = 'loading';
    repaintScreen('cohort');
    loadRealCohort(ok => {
      cohortView = ok ? 'loaded' : 'idle';
      repaintScreen('cohort');
    });
  }
  function patientWorkspaceFromDrilldown(payload) {
    const s = payload && payload.summary ? payload.summary : {};
    return {
      ok: true,
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
  function cohortWorkspaceFromReview(payload) {
    const s = payload && payload.summary ? payload.summary : {};
    return {
      ok: true,
      mode: 'real',
      database: payload && payload.source ? payload.source.database : null,
      cohortReview: payload,
      summary: {
        stays: s.cohort_size,
        modules: s.modules,
        file_count: s.file_count,
        total_rows: s.total_records,
        mean_age: s.age && s.age.mean,
        female_pct: s.sex && s.sex.female_pct,
        mortality: s.mortality_pct,
        median_los_icu: s.los_icu_days && s.los_icu_days.median,
        median_sofa2: s.sofa2 && s.sofa2.median,
        sepsis_pct: s.sepsis_pct,
      },
    };
  }
  function sourceLine(s) {
    const sum = s.summary || {};
    const parts = [];
    if (sum.stays != null) parts.push(`${fmtInt(sum.stays)} ${t('stays', '次住院')}`);
    if (sum.entities != null && sum.stays == null) parts.push(`${fmtInt(sum.entities)} ${t('entities', '个实体')}`);
    if (sum.modules != null) parts.push(`${fmtInt(sum.modules)} ${t('modules', '个模块')}`);
    if (sum.total_rows != null) parts.push(`${fmtInt(sum.total_rows)} ${t('rows', '行')}`);
    return parts.join(' · ') || t('export folder', '导出文件夹');
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
          <div class="t">No active local export is ready</div>
          <div class="d">Add or choose an EasyICU export folder below, or run Data Extraction first.</div>
          <div class="d mono" style="margin-top:4px;">registered_sources=${fmtInt(sourceCount)}</div>
        </div>
      </div>`;
    }
    const patientReady = active.patient_ready !== false;
    const sum = active.summary || {};
    const readyLine = [
      sum.entities != null ? `${fmtInt(sum.entities)} entities` : (sum.stays != null ? `${fmtInt(sum.stays)} stays` : null),
      sum.modules != null ? `${fmtInt(sum.modules)} modules` : null,
      sum.total_rows != null ? `${fmtInt(sum.total_rows)} rows` : null,
    ].filter(Boolean).join(' · ') || sourceLine(active);
    return `
      <div class="note ${patientReady ? 'ok' : 'warn'} mt-12" data-patient-source-ready="${patientReady ? 'true' : 'false'}">
        <div class="ico">${icon(patientReady ? 'check' : 'alert', 14)}</div>
        <div class="body">
          <div class="t">${patientReady ? 'Ready to load local export' : 'Registered export needs review'}</div>
          <div class="d"><b>${esc(active.label || active.database || 'Local export')}</b> · ${esc(readyLine)}</div>
          <div class="d mono" style="margin-top:4px;">path_hash=${esc(active.path_hash || '—')} · local-only metadata</div>
        </div>
      </div>`;
  }
  function sourceRegistryBlock(mode) {
    const multi = mode === 'multi';
    const active = defaultExportPath();
    const selected = new Set(defaultCrossdbPaths());
    const sources = registrySources().slice().sort((a, b) => {
      const aOn = multi ? selected.has(a.path) : a.path === active;
      const bOn = multi ? selected.has(b.path) : b.path === active;
      if (aOn !== bOn) return aOn ? -1 : 1;
      return 0;
    });
    const title = multi ? t('Local export sources', '本地导出来源') : t('Current local export', '当前本地导出');
    const empty = multi
      ? t('No registered exports yet. Add two EasyICU export folders below.', '还没有注册导出。请在下方添加两个 EasyICU 导出文件夹。')
      : t('No registered export yet. Add an EasyICU export folder below.', '还没有注册导出。请在下方添加一个 EasyICU 导出文件夹。');
    return `
      <div class="src-registry" data-src-mode="${multi ? 'multi' : 'single'}">
        <div class="src-head">
          <div><div class="eyebrow">${title}</div><div class="src-sub">${multi ? t('Choose at least two exports for Cross-DB preview.', '请选择至少两个导出用于跨库预览。') : t('This active export is shared by Patient, Cohort, Agent, and Copilot.', '这个 active 导出会被患者明细、队列统计、Agent 和 Copilot 共用。')}</div></div>
          <button class="btn sm ghost" data-src-refresh>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
        </div>
        <div class="src-list">
          ${sources.length ? sources.map(s => {
            const on = multi ? selected.has(s.path) : s.path === active;
            const attr = multi ? `data-src-cross="${esc(s.path)}"` : `data-src-active="${esc(s.path)}"`;
            const label = s.label || s.database || t('local', '本地');
            return `
              <div class="src-row ${on ? 'on' : ''}" ${attr}>
                <span class="src-ico">${icon(multi && on ? 'check' : 'folder', 14, multi && on ? 2.6 : undefined)}</span>
                <span class="src-body"><span class="src-name">${esc(label)}</span><span class="src-meta">${esc(sourceLine(s))}</span><span class="src-path mono">${esc(s.path)}</span></span>
                <span class="pill ${on ? 'ok' : 'dashed'}" style="height:20px;">${on ? (multi ? t('selected', '已选择') : t('active', '当前')) : (multi ? t('add', '添加') : t('use', '使用'))}</span>
                <span class="src-actions">
                  <button class="btn icon sm ghost" data-src-action data-src-rename="${esc(s.path)}" data-src-label="${esc(label)}" title="${esc(t('Rename source', '重命名来源'))}">${icon('edit', 12)}</button>
                  <button class="btn icon sm ghost" data-src-action data-src-remove="${esc(s.path)}" title="${esc(t('Remove registration only; files stay on disk', '仅移除注册记录；磁盘文件保留'))}">${icon('close', 12)}</button>
                </span>
              </div>`;
          }).join('') : `<div class="empty compact"><div class="glyph">${icon('folder', 20)}</div><div class="t">${empty}</div></div>`}
        </div>
        <div class="path-field editable src-add">
          <span class="pf-ico">${icon('folder', 14)}</span>
          <input class="pf-input" data-src-path-input type="text" spellcheck="false" autocomplete="off" placeholder="${esc(t('Paste a local EasyICU export folder', '粘贴本地 EasyICU 导出文件夹'))}" aria-label="${esc(t('EasyICU export path', 'EasyICU 导出路径'))}" />
          <button class="btn sm" data-src-browse>${icon('folder', 12)} ${t('Browse...', '浏览...')}</button>
          <button class="btn sm primary" data-src-add>${icon('plus', 12)} ${t('Add', '添加')}</button>
        </div>
        <div class="note warn src-add-feedback" data-src-add-feedback hidden aria-hidden="true" role="status" style="display:none;">
          <div class="ico" data-src-add-feedback-icon>${icon('alert', 14)}</div>
          <div class="body"><div class="d" data-src-add-feedback-text style="margin:0;"></div></div>
        </div>
      </div>`;
  }
  function setSourceAddFeedback(container, message, kind) {
    const box = container && container.querySelector('[data-src-add-feedback]');
    if (!box) return;
    const clean = message == null ? '' : String(message).trim();
    const text = box.querySelector('[data-src-add-feedback-text]');
    if (!clean) {
      box.hidden = true;
      box.setAttribute('aria-hidden', 'true');
      box.style.display = 'none';
      if (text) text.textContent = '';
      return;
    }
    const level = kind || 'warn';
    box.hidden = false;
    box.removeAttribute('aria-hidden');
    box.style.display = '';
    box.classList.remove('warn', 'ok', 'info');
    box.classList.add(level);
    if (text) text.textContent = clean;
    const glyph = box.querySelector('[data-src-add-feedback-icon]');
    if (glyph) glyph.innerHTML = icon(level === 'ok' ? 'check' : (level === 'info' ? 'db' : 'alert'), 14);
  }
  function registerSourceFromInput(container, screenId, button) {
    const input = container && container.querySelector('[data-src-path-input]');
    const path = input && input.value ? input.value.trim() : '';
    if (!path) {
      vizErr = null;
      if (input) {
        input.setAttribute('aria-invalid', 'true');
        input.focus();
      }
      setSourceAddFeedback(container, t('Use Browse to choose a local EasyICU export folder, or paste its path before pressing Add.', '请点击“浏览”选择本地 EasyICU 导出文件夹，或粘贴路径后再点击添加。'), 'warn');
      return;
    }
    if (!(window.EU_API && window.EU_API.registerWorkspaceSource)) {
      setSourceAddFeedback(container, t('Local workspace API is not ready. Refresh the page and try again.', '本地工作区 API 尚未就绪。请刷新页面后重试。'), 'warn');
      return;
    }
    if (button && button.getAttribute('aria-disabled') === 'true') return;
    if (input) input.removeAttribute('aria-invalid');
    if (button) button.setAttribute('aria-disabled', 'true');
    setSourceAddFeedback(container, t('Checking and adding this local export...', '正在检查并添加这个本地导出...'), 'info');
    const multi = container && container.dataset && container.dataset.srcMode === 'multi';
    window.EU_API.registerWorkspaceSource(path, { active: !multi, crossdb: true }).then(() => {
      vizErr = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; window.EU_COHORT_REVIEW = null; resetCohortFeatureSelection(); crossView = 'idle'; patientView = 'idle'; repaintScreen(screenId);
    }).catch(err => {
      const msg = String(err && err.message || err);
      vizErr = msg;
      if (button) button.removeAttribute('aria-disabled');
      setSourceAddFeedback(container, msg, 'warn');
    });
  }
  let sourcePickerEl = null;
  function ensureSourcePickerStyles() {
    if (document.getElementById('euPickerCss')) return;
    const s = document.createElement('style'); s.id = 'euPickerCss';
    s.textContent = `
      .eu-pick-back{position:fixed;inset:0;background:rgba(15,18,24,.42);backdrop-filter:blur(2px);z-index:1000;display:flex;align-items:center;justify-content:center;}
      .eu-pick{width:min(560px,92vw);max-height:78vh;display:flex;flex-direction:column;background:var(--surface,#fff);border:1px solid var(--line,#e6e8ee);border-radius:14px;box-shadow:0 24px 60px rgba(15,18,24,.28);overflow:hidden;}
      .eu-pick-h{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line,#e6e8ee);}
      .eu-pick-h .t{font-weight:650;font-size:14px;}
      .eu-pick-cur{font-family:var(--mono,monospace);font-size:11px;color:var(--ink-4,#8a91a0);padding:8px 16px;border-bottom:1px solid var(--line,#eef0f4);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
      .eu-pick-sc{display:flex;gap:6px;padding:8px 16px;flex-wrap:wrap;border-bottom:1px solid var(--line,#eef0f4);}
      .eu-pick-sc button{font-size:11.5px;padding:3px 10px;border:1px solid var(--line,#e0e3ea);border-radius:999px;background:var(--surface-2,#f7f8fb);cursor:pointer;color:var(--ink-2,#3a4150);}
      .eu-pick-list{overflow:auto;flex:1;padding:6px;}
      .eu-pick-row{display:flex;align-items:center;gap:10px;width:100%;padding:8px 10px;border:0;background:none;text-align:left;cursor:pointer;border-radius:8px;font-size:13px;color:var(--ink-1,#1a2030);}
      .eu-pick-row:hover{background:var(--surface-2,#f2f4f8);}
      .eu-pick-row .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
      .eu-pick-row .hint{font-size:10px;color:var(--ink-4,#8a91a0);border:1px solid var(--line,#e0e3ea);border-radius:5px;padding:0 5px;flex:none;}
      .eu-pick-f{display:flex;align-items:center;gap:8px;padding:12px 16px;border-top:1px solid var(--line,#e6e8ee);}
      .eu-pick-empty{padding:24px;text-align:center;color:var(--ink-4,#8a91a0);font-size:12px;}`;
    document.head.appendChild(s);
  }
  function closeSourcePicker() {
    if (sourcePickerEl) { sourcePickerEl.remove(); sourcePickerEl = null; }
    document.removeEventListener('keydown', sourcePickerKey);
  }
  function sourcePickerKey(e) { if (e.key === 'Escape') closeSourcePicker(); }
  function openSourceFolderPicker(startPath, onPick, title) {
    if (!(window.EU_API && window.EU_API.listDir)) {
      if (onPick) onPick('');
      return;
    }
    ensureSourcePickerStyles();
    closeSourcePicker();
    let cur = startPath || '';
    const pickerTitle = title || t('Choose EasyICU export folder', '选择 EasyICU 导出文件夹');
    const back = document.createElement('div'); back.className = 'eu-pick-back';
    back.innerHTML = `
      <div class="eu-pick" role="dialog" aria-label="${esc(pickerTitle)}">
        <div class="eu-pick-h">
          <span style="color:var(--ink-3);">${icon('folder', 16)}</span>
          <span class="t">${esc(pickerTitle)}</span>
          <span class="grow" style="flex:1;"></span>
          <button class="btn sm ghost" data-pk-close>${icon('close', 13)}</button>
        </div>
        <div class="eu-pick-cur" data-pk-cur></div>
        <div class="eu-pick-sc" data-pk-sc></div>
        <div class="eu-pick-list" data-pk-list><div class="eu-pick-empty">${t('Loading...', '加载中...')}</div></div>
        <div class="eu-pick-f">
          <button class="btn ghost sm" data-pk-up>${icon('back', 13)} ${t('Up', '上一级')}</button>
          <span style="flex:1;"></span>
          <button class="btn primary" data-pk-use>${icon('check', 13)} ${t('Use this folder', '选择此文件夹')}</button>
        </div>
      </div>`;
    document.body.appendChild(back); sourcePickerEl = back;
    const listEl = back.querySelector('[data-pk-list]');
    const curEl = back.querySelector('[data-pk-cur]');
    const scEl = back.querySelector('[data-pk-sc]');
    back.addEventListener('click', e => { if (e.target === back) closeSourcePicker(); });
    back.querySelector('[data-pk-close]').addEventListener('click', closeSourcePicker);
    back.querySelector('[data-pk-use]').addEventListener('click', () => { closeSourcePicker(); if (cur && onPick) onPick(cur); });
    document.addEventListener('keydown', sourcePickerKey);

    function load(path) {
      listEl.innerHTML = `<div class="eu-pick-empty">${t('Loading...', '加载中...')}</div>`;
      window.EU_API.listDir(path).then(r => {
        cur = r.path || path || '';
        curEl.textContent = cur || '/';
        const up = back.querySelector('[data-pk-up]');
        up.disabled = !r.parent;
        up.onclick = () => r.parent && load(r.parent);
        scEl.innerHTML = '';
        (r.shortcuts || []).forEach(s => {
          const b = document.createElement('button'); b.textContent = s.name;
          b.onclick = () => load(s.path); scEl.appendChild(b);
        });
        if (!r.entries || !r.entries.length) {
          listEl.innerHTML = `<div class="eu-pick-empty">${r.ok === false ? t('Cannot read this folder.', '无法读取该文件夹。') : t('No sub-folders here.', '此处没有子文件夹。')}</div>`;
          return;
        }
        listEl.innerHTML = '';
        r.entries.forEach(en => {
          const b = document.createElement('button'); b.className = 'eu-pick-row';
          b.innerHTML = `<span style="color:var(--ink-3);flex:none;">${icon('folder', 15)}</span><span class="nm">${esc(en.name)}</span>${en.hint ? `<span class="hint">${esc(en.hint)}</span>` : ''}`;
          b.onclick = () => load(en.path); listEl.appendChild(b);
        });
      }).catch(err => {
        listEl.innerHTML = `<div class="eu-pick-empty">${t('Failed to list folder', '列目录失败')}: ${esc(String(err && err.message || err))}</div>`;
      });
    }
    load(cur);
  }
  function bindSourceRegistry(root, screenId) {
    root.querySelectorAll('[data-src-active]').forEach(b => b.addEventListener('click', e => {
      if (e.target.closest('[data-src-action]')) return;
      const path = b.dataset.srcActive;
      if (!path || !(window.EU_API && window.EU_API.saveWorkspaceRegistry)) return;
      window.EU_API.saveWorkspaceRegistry({ active_path: path }).then(() => {
        try { localStorage.setItem('easyicu_last_export_dir', path); } catch (e) {}
        window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; window.EU_COHORT_REVIEW = null; resetCohortFeatureSelection(); window.EU_STALE = true;
        patientView = 'idle'; crossView = 'idle';
        if (screenId === 'cohort' && window.EU_DATA === 'real') {
          cohortView = 'loading';
          repaintScreen('cohort');
          loadRealCohort(ok => { cohortView = ok ? 'loaded' : 'idle'; repaintScreen('cohort'); });
          return;
        }
        repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-cross]').forEach(b => b.addEventListener('click', e => {
      if (e.target.closest('[data-src-action]')) return;
      const path = b.dataset.srcCross;
      const cur = defaultCrossdbPaths().filter(Boolean);
      const next = cur.includes(path) ? cur.filter(p => p !== path) : cur.concat([path]);
      if (!(window.EU_API && window.EU_API.saveWorkspaceRegistry)) return;
      window.EU_API.saveWorkspaceRegistry({ crossdb_paths: next }).then(() => {
        window.EU_CROSSDB_WORKSPACE = null; window.EU_COHORT_REVIEW = null; resetCohortFeatureSelection(); crossView = 'idle'; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-add]').forEach(b => b.addEventListener('click', () => {
      registerSourceFromInput(b.closest('[data-src-mode]') || root, screenId, b);
    }));
    root.querySelectorAll('[data-src-browse]').forEach(b => b.addEventListener('click', () => {
      const container = b.closest('[data-src-mode]') || root;
      const input = container.querySelector('[data-src-path-input]');
      openSourceFolderPicker((input && input.value.trim()) || defaultExportPath(), picked => {
        if (!picked || !input) {
          setSourceAddFeedback(container, t('Local folder picker API is not ready. Paste a path instead.', '本地文件夹选择 API 尚未就绪。请改为粘贴路径。'), 'warn');
          return;
        }
        input.value = picked;
        input.removeAttribute('aria-invalid');
        setSourceAddFeedback(container, t('Folder selected. Registering and switching to this export...', '已选择文件夹，正在注册并切换到这个导出...'), 'info');
        registerSourceFromInput(container, screenId, container.querySelector('[data-src-add]'));
      });
    }));
    root.querySelectorAll('[data-src-path-input]').forEach(input => {
      input.addEventListener('input', () => {
        input.removeAttribute('aria-invalid');
        setSourceAddFeedback(input.closest('[data-src-mode]') || root, '', 'warn');
      });
      input.addEventListener('keydown', e => {
        if (e.key !== 'Enter') return;
        e.preventDefault();
        const container = input.closest('[data-src-mode]') || root;
        registerSourceFromInput(container, screenId, container.querySelector('[data-src-add]'));
      });
    });
    root.querySelectorAll('[data-src-rename]').forEach(b => b.addEventListener('click', e => {
      e.preventDefault(); e.stopPropagation();
      const path = b.dataset.srcRename;
      const current = b.dataset.srcLabel || '';
      if (!path || !(window.EU_API && window.EU_API.renameWorkspaceSource)) return;
      const next = window.prompt(t('Source label', '来源名称'), current);
      if (next === null) return;
      window.EU_API.renameWorkspaceSource(path, next).then(() => {
        vizErr = null; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-remove]').forEach(b => b.addEventListener('click', e => {
      e.preventDefault(); e.stopPropagation();
      const path = b.dataset.srcRemove;
      if (!path || !(window.EU_API && window.EU_API.removeWorkspaceSource)) return;
      if (!window.confirm(t('Remove this source from the registry? Export files stay on disk.', '从注册表中移除此来源？导出文件仍会保留在磁盘上。'))) return;
      window.EU_API.removeWorkspaceSource(path).then(() => {
        vizErr = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; window.EU_COHORT_REVIEW = null; resetCohortFeatureSelection(); crossView = 'idle'; patientView = 'idle'; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-refresh]').forEach(b => b.addEventListener('click', () => {
      if (!(window.EU_API && window.EU_API.hydrateWorkspaceRegistry)) return;
      window.EU_API.hydrateWorkspaceRegistry().then(() => { window.EU_PATIENT_SOURCES = null; repaintScreen(screenId); }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
  }
  function loadRealWorkspace(done) {
    if (!(window.EU_API && window.EU_API.loadWorkspaceSummary)) {
      vizErr = 'Live API is unavailable.';
      done && done(false);
      return;
    }
    const path = defaultExportPath();
    window.EU_API.loadWorkspaceSummary(path).then(ws => {
      window.EU_VIZ_WORKSPACE = ws;
      vizErr = null;
      window.EU_HASWORK = true;
      try { localStorage.setItem('easyicu_last_export_dir', ws.path || path); } catch (e) {}
      if (window.EU_API && window.EU_API.registerWorkspaceSource) {
        window.EU_API.registerWorkspaceSource(ws.path || path, { active: true, crossdb: false }).catch(() => {});
      }
      done && done(true);
    }).catch(err => {
      vizErr = String(err && err.message || err);
      done && done(false);
    });
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
    if (patientTableModule) body.table_module = patientTableModule;
    body.table_page = patientTablePage;
    body.table_page_size = patientTablePageSize;
    window.EU_API.loadPatientReviewDrilldown(body).then(payload => {
      window.EU_PATIENT_DRILLDOWN = payload;
      window.EU_VIZ_WORKSPACE = patientWorkspaceFromDrilldown(payload);
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
  function cohortMissingExportMessage() {
    return t('Choose or add a local EasyICU export before loading Cohort Statistics.', '请先选择或添加本地 EasyICU 导出，再加载队列统计。');
  }
  function loadRealCohort(done) {
    if (!(window.EU_API && window.EU_API.loadCohortReviewSummary)) {
      vizErr = 'Cohort Review API is unavailable.';
      done && done(false);
      return;
    }
    const active = registryActivePath();
    if (!active) {
      window.EU_COHORT_REVIEW = null;
      window.EU_VIZ_WORKSPACE = null;
      vizErr = cohortMissingExportMessage();
      done && done(false);
      return;
    }
    const body = { source_path: active };
    if (cohortSelectedFeatures.length) body.selected_features = cohortSelectedFeatures.slice();
    window.EU_API.loadCohortReviewSummary(body).then(payload => {
      window.EU_COHORT_REVIEW = payload;
      syncCohortFeatureSelection(payload);
      window.EU_VIZ_WORKSPACE = cohortWorkspaceFromReview(payload);
      vizErr = null;
      window.EU_HASWORK = true;
      done && done(true);
    }).catch(err => {
      window.EU_COHORT_REVIEW = null;
      window.EU_VIZ_WORKSPACE = null;
      vizErr = String(err && err.message || err);
      done && done(false);
    });
  }
  function loadRealCrossdb(done, opts) {
    teardownCrossRawES();
    crossRawJobId = null;
    crossRawProg = null;
    crossRawCancelRequested = false;
    window.EU_CROSSDB_WORKSPACE = null;
    window.EU_COHORT_REVIEW = null;
    resetCohortFeatureSelection();
    crossDensityModule = 'all';
    crossDensityFeature = null;
    const paths = defaultCrossdbPaths();
    if (paths.length >= 2 && window.EU_API && (window.EU_API.loadCrossdbReviewSummary || window.EU_API.loadCrossdbSummary)) {
      const loader = window.EU_API.loadCrossdbReviewSummary
        ? window.EU_API.loadCrossdbReviewSummary({ paths: paths })
        : window.EU_API.loadCrossdbSummary(paths);
      loader.then(xdb => {
        window.EU_CROSSDB_WORKSPACE = xdb;
        const first = xdb.sources && xdb.sources[0];
        if (first) window.EU_VIZ_WORKSPACE = { database: first.database, summary: first.summary };
        vizErr = null;
        window.EU_HASWORK = true;
        done && done(true);
      }).catch(err => {
        vizErr = String(err && err.message || err);
        done && done(false);
      });
      return;
    }
    const requestedRawRoot = opts && opts.rawRoot ? String(opts.rawRoot).trim() : '';
    const rawRootInput = document.querySelector('[data-crossdb-root]');
    const rawRoot = requestedRawRoot || (rawRootInput && rawRootInput.value ? rawRootInput.value.trim() : '');
    const rawDatabases = selectedCrossDbKeys();
    const sampleProfile = crossRawSampleProfile();
    if (!rawRoot) {
      vizErr = t('Choose a local ICU data root before loading real Cross-DB densities.', '加载真实跨库密度前，请先选择本地 ICU 数据根目录。');
      done && done(false);
      return;
    }
    if (!crossRawScanReadyFor(rawRoot)) {
      vizErr = t('Check the ICU data root first so EasyICU can confirm at least two selected database folders.', '请先检查 ICU 数据根目录，确认至少两个已选数据库文件夹可识别。');
      done && done(false);
      return;
    }
    if (rawRoot && rawDatabases.length >= 2 && window.EU_API && window.EU_API.startCrossdbRawDistributionJob && window.EventSource) {
      if (crossRawJobStarting) return;
      crossRawJobStarting = true;
      crossRawRootDraft = rawRoot;
      window.EU_API.startCrossdbRawDistributionJob({
        data_root: rawRoot,
        databases: rawDatabases,
        feature_scope: 'all_catalog',
        coverage_min: 2,
        max_patients: sampleProfile.maxPatients,
        sample_size: sampleProfile.sampleSize,
      }).then(r => {
        if (crossRawCancelRequested) {
          crossRawJobStarting = false;
          if (r && r.job_id && window.EU_API && window.EU_API.postJSON) {
            window.EU_API.postJSON('/api/jobs/' + r.job_id + '/cancel', { reason: 'user_requested' }).catch(() => {});
          }
          return;
        }
        crossRawJobId = r.job_id;
        crossRawProg = {
          phase: 'queued',
          max_patients: sampleProfile.maxPatients,
          sample_size: sampleProfile.sampleSize,
          message: `${t('Queued local raw Cross-DB density job.', '本地原始跨库密度任务已排队。')} ${crossRawSampleSummary(sampleProfile)}`,
        };
        crossRawES = new EventSource('/api/jobs/' + r.job_id + '/events');
        crossRawES.onmessage = ev => {
          let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
          if (m.type === 'progress') {
            crossRawProg = m;
          } else if (m.type === 'cancel_requested') {
            crossRawCancelRequested = true;
            crossRawProg = {
              phase: 'cancel',
              message: t('Cancel requested. The current database read may finish before the job stops.', '已请求取消。当前数据库读取可能会先完成，然后任务才停止。'),
            };
          } else if (m.type === 'end') {
            teardownCrossRawES();
            crossRawJobStarting = false;
            if (m.status === 'done') {
              const xdb = m.result || {};
              window.EU_CROSSDB_WORKSPACE = xdb;
              const first = xdb.sources && xdb.sources[0];
              if (first) window.EU_VIZ_WORKSPACE = { database: first.database, summary: first.summary };
              vizErr = null;
              window.EU_HASWORK = true;
              done && done(true);
            } else if (m.status === 'cancelled') {
              vizErr = t('Raw Cross-DB density job cancelled before completion.', '原始跨库密度任务已在完成前取消。');
              done && done(false);
            } else {
              vizErr = m.error || t('Raw Cross-DB density job failed.', '原始跨库密度任务失败。');
              done && done(false);
            }
          }
          repaintScreen('crossdb');
        };
        crossRawES.onerror = () => {
        crossRawJobStarting = false;
          if (!window.EU_CROSSDB_WORKSPACE && !vizErr) vizErr = t('Lost connection to the raw Cross-DB density job.', '与原始跨库密度任务的连接已断开。');
          teardownCrossRawES();
          done && done(false);
          repaintScreen('crossdb');
        };
        repaintScreen('crossdb');
      }).catch(err => {
        crossRawJobStarting = false;
        if (crossRawCancelRequested) return;
        vizErr = String(err && err.message || err);
        done && done(false);
      });
      return;
    }
    vizErr = rawDatabases.length < 2
      ? t('Select at least two databases before loading real Cross-DB densities.', '加载真实跨库密度前，请至少选择两个数据库。')
      : t('Raw Cross-DB density job API is unavailable in this browser session.', '当前浏览器会话无法使用原始跨库密度任务 API。');
    done && done(false);
  }
  function loadDemoCrossdb(done) {
    window.EU_CROSSDB_WORKSPACE = null;
    window.EU_VIZ_WORKSPACE = null;
    crossDensityModule = 'all';
    crossDensityFeature = null;
    const databases = selectedCrossDbKeys();
    if (databases.length < 2) {
      vizErr = t('Select at least two demo databases.', '请至少选择两个演示数据库。');
      done && done(false);
      return;
    }
    if (!window.EU_API || !window.EU_API.loadCrossdbDemoDistribution) {
      vizErr = t('Demo distribution endpoint is unavailable.', '演示分布接口不可用。');
      done && done(false);
      return;
    }
    window.EU_API.loadCrossdbDemoDistribution({
      databases,
      feature_scope: 'all_catalog',
      records_per_feature: 96,
    }).then(xdb => {
      window.EU_CROSSDB_WORKSPACE = xdb;
      const first = xdb.sources && xdb.sources[0];
      if (first) window.EU_VIZ_WORKSPACE = { database: first.database, summary: first.summary };
      vizErr = null;
      window.EU_HASWORK = true;
      done && done(true);
    }).catch(err => {
      window.EU_CROSSDB_WORKSPACE = null;
      vizErr = String(err && err.message || err);
      done && done(false);
    });
  }

  /* allow the print harness to preset loaded states for a richer PDF */
  window.__euVizPreset = function (which) {
    if (!which || which === 'patient') patientView = 'loaded';
    if (!which || which === 'crossdb') crossView = 'loaded';
  };
  window.__euVizResetForDataMode = function () {
    patientView = 'idle';
    crossView = 'idle';
    cohortView = 'idle';
    vizErr = null;
    window.EU_PATIENT_SOURCES = null;
  };

  function repaintScreen(id) {
    if (window.__euRender) { window.__euRender(); return; }
    const app = document.getElementById('app');
    const content = app && app.querySelector('.content');
    if (!content) return;
    content.innerHTML = S[id].render();
    if (S[id].afterRender) S[id].afterRender(app);
    content.scrollTop = 0;
  }

  /* tiny seeded sparkline */
  function spark(vals, w = 132, h = 36, color = 'var(--accent)') {
    if (!vals || vals.length === 0) return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"></svg>`;
    if (vals.length === 1) vals = [vals[0], vals[0]];
    const max = Math.max(...vals), min = Math.min(...vals), span = (max - min) || 1;
    const pts = vals.map((v, i) => {
      const x = (i / (vals.length - 1)) * (w - 4) + 2;
      const y = h - 4 - ((v - min) / span) * (h - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }

  function axisSpark(vals, w = 440, h = 86, color = 'var(--accent)', opts = {}) {
    const nums = (vals || []).map(v => Number(v)).filter(v => Number.isFinite(v));
    if (!nums.length) {
      return `<svg class="spark axis-spark" data-axis-chart="true" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><text x="44" y="${Math.round(h / 2)}" fill="#94a3b8" font-size="10">no numeric points</text></svg>`;
    }
    const seriesVals = nums.length === 1 ? [nums[0], nums[0]] : nums;
    const thresholdRows = (opts.thresholds || [])
      .map(th => ({ value: Number(th && th.value), label: String((th && th.label) || 'threshold'), color: (th && th.color) || '#d97706', dash: (th && th.dash) || '3 3' }))
      .filter(th => Number.isFinite(th.value));
    const rawMin = Math.min(...seriesVals, ...thresholdRows.map(th => th.value));
    const rawMax = Math.max(...seriesVals, ...thresholdRows.map(th => th.value));
    const rawSpan = (rawMax - rawMin) || 1;
    const min = rawMin - rawSpan * 0.08;
    const max = rawMax + rawSpan * 0.08;
    const span = (max - min) || 1;
    const left = 46;
    const right = 14;
    const top = 9;
    const bottom = 24;
    const innerW = Math.max(24, w - left - right);
    const innerH = Math.max(24, h - top - bottom);
    const xFor = i => left + (i / Math.max(seriesVals.length - 1, 1)) * innerW;
    const yFor = v => top + (1 - ((v - min) / span)) * innerH;
    const pts = seriesVals.map((v, i) => `${xFor(i).toFixed(1)},${yFor(v).toFixed(1)}`).join(' ');
    const yTop = yFor(rawMax);
    const yMid = yFor((rawMax + rawMin) / 2);
    const yBottom = yFor(rawMin);
    const unit = opts.unit ? ` ${opts.unit}` : '';
    const label = opts.label || 'value';
    const current = seriesVals[seriesVals.length - 1];
    const thresholds = thresholdRows.slice(0, 3).map(th => {
      const y = yFor(th.value);
      if (y < top - 1 || y > top + innerH + 1) return '';
      return `<line x1="${left}" y1="${y.toFixed(1)}" x2="${(left + innerW).toFixed(1)}" y2="${y.toFixed(1)}" stroke="${th.color}" stroke-width="1" stroke-dasharray="${th.dash}" opacity=".72"><title>${esc(th.label)} ${fmtNum(th.value, 1)}${esc(unit)}</title></line>`;
    }).join('');
    return `
      <svg class="spark axis-spark" data-axis-chart="true" data-axis-label="${esc(label)}" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="${esc(label)} chart with x and y axes">
        <line x1="${left}" y1="${top}" x2="${left}" y2="${top + innerH}" stroke="#cbd5e1" stroke-width="1"/>
        <line x1="${left}" y1="${top + innerH}" x2="${left + innerW}" y2="${top + innerH}" stroke="#cbd5e1" stroke-width="1"/>
        <line x1="${left}" y1="${yTop.toFixed(1)}" x2="${left + innerW}" y2="${yTop.toFixed(1)}" stroke="#eef2f7" stroke-width="1"/>
        <line x1="${left}" y1="${yMid.toFixed(1)}" x2="${left + innerW}" y2="${yMid.toFixed(1)}" stroke="#eef2f7" stroke-width="1"/>
        <line x1="${left}" y1="${yBottom.toFixed(1)}" x2="${left + innerW}" y2="${yBottom.toFixed(1)}" stroke="#eef2f7" stroke-width="1"/>
        ${thresholds}
        <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="${xFor(seriesVals.length - 1).toFixed(1)}" cy="${yFor(current).toFixed(1)}" r="2.6" fill="${color}" stroke="#fff" stroke-width="1.2"/>
        <text x="2" y="${Math.max(11, yTop + 3).toFixed(1)}" fill="#64748b" font-size="9">${fmtNum(rawMax, 1)}${esc(unit)}</text>
        <text x="2" y="${Math.min(h - 18, yBottom + 3).toFixed(1)}" fill="#64748b" font-size="9">${fmtNum(rawMin, 1)}${esc(unit)}</text>
        <text x="${left}" y="${h - 6}" fill="#64748b" font-size="9">${esc((opts.xLabels && opts.xLabels[0]) || 't0')}</text>
        <text x="${Math.max(left + 28, left + innerW - 46).toFixed(1)}" y="${h - 6}" fill="#64748b" font-size="9">${esc((opts.xLabels && opts.xLabels[1]) || `t${seriesVals.length - 1}`)}</text>
        <text x="${Math.max(left + 70, left + innerW - 116).toFixed(1)}" y="11" fill="#64748b" font-size="9">current ${fmtNum(current, 1)}${esc(unit)}</text>
      </svg>`;
  }

  function skeletonWorkspace() {
    return `
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow">
          <div style="font-weight:600;font-size:12.75px;">Generating demo review workspace…</div>
          <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">reproducible · no outbound calls</div>
        </div>
        <button class="btn sm" data-viz-reset>${icon('stop', 13)} Cancel</button>
      </div>
      <div class="indet mt-12"></div>
      <div class="st-stats mt-16">
        ${[0,1,2,3].map(() => `<div class="sk-stat"><div class="sk sk-line sm" style="width:52%"></div><div class="sk" style="height:22px;width:64%;margin-top:10px;"></div></div>`).join('')}
      </div>
      <div class="sk-table mt-16">
        <div class="sk-trow head">${[42,28,28,28,28].map(w => `<div class="sk sk-line sm" style="width:${w}%"></div>`).join('')}</div>
        ${[0,1,2,3,4].map(() => `<div class="sk-trow">${[70,55,48,52,40].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}
      </div>`;
  }

  function buildDemoPatientDrilldown(selectedRef) {
    const modules = demoCatalogModules();
    const totalFeatures = modules.reduce((acc, row) => acc + row.features.length, 0);
    const moduleProfiles = modules.map((row, idx) => {
      const coverage = Math.max(58, Math.min(100, 84 + (idx % 5) * 3 - (row.features.length > 25 ? 4 : 0)));
      const entities = Math.round(DEMO_ENTITY_COUNT * coverage / 100);
      const timeIndexed = demoIsTimeIndexed(row.module);
      return {
        module: row.module,
        label: row.label,
        rows: demoRowsForModule(row.module, row.features.length, coverage),
        feature_count: row.features.length,
        observed_features: row.features.length,
        entities,
        coverage_pct: Number(coverage.toFixed(1)),
        time_indexed: timeIndexed,
        dynamic_features: timeIndexed ? row.features.length : 0,
        static_features: timeIndexed ? 0 : row.features.length,
        preview_features: row.features.slice(0, 6),
      };
    });
    const totalRows = moduleProfiles.reduce((acc, row) => acc + (Number(row.rows) || 0), 0);
    const tableModules = moduleProfiles.map(row => ({
      module: row.module,
      label: row.label,
      review_features: row.feature_count,
      observed_features: row.observed_features,
      rows: row.rows,
      entities: row.entities,
      coverage_pct: row.coverage_pct,
      share_pct: totalFeatures ? Number((row.feature_count / totalFeatures * 100).toFixed(1)) : null,
      shape: row.time_indexed ? 'time_indexed' : 'static',
      dynamic_features: row.dynamic_features,
      static_features: row.static_features,
      preview_features: row.preview_features.map(feature => {
        const meta = catalogFeatureMeta(feature);
        return { feature, name: meta.name || feature, unit: meta.unit || '', group: row.label };
      }),
      status: demoReviewStatus(row.coverage_pct, row.feature_count),
    }));
    const selectedIndex = Math.max(0, Math.min(4, Number(String(selectedRef || 'demo_ent_1').replace(/\D+/g, '')) - 1 || 0));
    const entities = [0, 1, 2, 3, 4].map(idx => ({
      ref: `demo_ent_${idx + 1}`,
      label: `Demo entity ${idx + 1}`,
      ordinal: idx + 1,
      outcome: idx === 2 ? 'Deceased' : 'Survived',
      severity: `SOFA-2 ${6 + idx}`,
    }));
    const timeLanes = demoTimeLanes(selectedIndex);
    const signalIndex = {};
    timeLanes.forEach(lane => (lane.signals || []).forEach(sig => { if (!signalIndex[sig.feature]) signalIndex[sig.feature] = sig; }));
    const selected = {
      ref: entities[selectedIndex].ref,
      label: entities[selectedIndex].label,
      ordinal: selectedIndex + 1,
      demographics: { age: 58 + selectedIndex * 6, sex: selectedIndex % 2 ? 'F' : 'M' },
      scores: { sofa2_max: 6 + selectedIndex, sepsis3_sofa2: selectedIndex !== 1 },
      outcomes: { status: entities[selectedIndex].outcome, icu_los_days: Number((4.2 + selectedIndex * 1.3).toFixed(1)) },
      signals: (timeLanes.find(l => l.lane === 'vitals') || {}).signals || [],
    };
    const qualityRows = [];
    modules.forEach((moduleRow, moduleIdx) => {
      moduleRow.features.forEach((feature, featureIdx) => {
        const coverage = demoCoverageForFeature(feature, moduleIdx + featureIdx);
        const missing = Number((100 - coverage).toFixed(1));
        const timeIndexed = demoIsTimeIndexed(moduleRow.module);
        const demoPointCount = Array.isArray(DEMO_CHART_HOURS) && DEMO_CHART_HOURS.length ? DEMO_CHART_HOURS.length : 12;
        const records = timeIndexed
          ? Math.max(1, Math.round(DEMO_ENTITY_COUNT * demoPointCount * (coverage / 100)))
          : Math.max(1, Math.round(DEMO_ENTITY_COUNT * (coverage / 100)));
        const outlier = DEMO_THRESHOLDS[feature] ? Number((((featureIdx + moduleIdx) % 4) * 0.4).toFixed(1)) : 0;
        const duplicate = timeIndexed ? Number((((featureIdx + 1) % 5) * 0.08).toFixed(2)) : 0;
        const meta = catalogFeatureMeta(feature);
        qualityRows.push({
          feature,
          name: meta.name || feature,
          module: moduleRow.module,
          records,
          entities: Math.max(1, Math.round(DEMO_ENTITY_COUNT * coverage / 100)),
          coverage_pct: coverage,
          missing_pct: missing,
          out_of_physio_pct: outlier,
          duplicate_time_pct: duplicate,
          density_per_entity: Number((records / DEMO_ENTITY_COUNT).toFixed(3)),
          time_indexed: timeIndexed,
          status: demoQualityStatus(missing, outlier, duplicate),
        });
      });
    });
    const weight = qualityRows.reduce((acc, row) => acc + Math.max(row.records, 1), 0) || 1;
    const weighted = key => Number((qualityRows.reduce((acc, row) => acc + (Number(row[key]) || 0) * Math.max(row.records, 1), 0) / weight).toFixed(1));
    const qualitySummary = {
      concept_count: qualityRows.length,
      total_records: qualityRows.reduce((acc, row) => acc + row.records, 0),
      weighted_missing_pct: weighted('missing_pct'),
      weighted_out_of_physio_pct: weighted('out_of_physio_pct'),
      weighted_duplicate_time_pct: weighted('duplicate_time_pct'),
      denominator_entities: DEMO_ENTITY_COUNT,
    };
    const topIssues = qualityRows.slice().sort((a, b) =>
      (b.missing_pct - a.missing_pct) || (b.out_of_physio_pct - a.out_of_physio_pct) || (b.records - a.records)
    ).slice(0, 5);
    const quality = moduleProfiles.map(row => ({
      module: row.module,
      rows: row.rows,
      column_count: row.feature_count,
      covered_entities: row.entities,
      coverage_pct: row.coverage_pct,
      quality_status: row.coverage_pct >= 80 ? 'ok' : (row.coverage_pct >= 50 ? 'warn' : 'bad'),
    }));
    const readyLanes = timeLanes.filter(row => row.status === 'ready' && (row.signals || []).length);
    const loadedSignals = readyLanes.reduce((acc, row) => acc + row.signal_count, 0);
    const comparisonFeatures = qualityRows.filter(row => row.time_indexed).sort((a, b) => b.records - a.records).slice(0, 8)
      .map(row => ({ feature: row.feature, name: row.name, module: row.module, records: row.records, entities: row.entities, coverage_pct: row.coverage_pct, density_per_entity: row.density_per_entity }));
    const compareFeature = (comparisonFeatures[0] && comparisonFeatures[0].feature) || 'hr';
    const compareMeta = catalogFeatureMeta(compareFeature);
    const compareModule = demoFeatureModule(compareFeature);
    const comparisonTraces = entities.map((entity, idx) => {
      const signal = demoSignal(compareFeature, idx);
      const values = (signal.values || []).map(Number).filter(Number.isFinite);
      return {
        ref: entity.ref,
        label: entity.label,
        values,
        times: values.map((_, timeIndex) => demoCharttimeAt(timeIndex)),
        point_count: values.length,
        bounded: true,
        max_points: values.length,
      };
    }).filter(trace => trace.values.length >= 2);
    const sections = [
      demoCategorySection('vitals', 'Vital Signs Snapshot', ['hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2'], signalIndex),
      demoCategorySection('labs', 'Key Laboratory Snapshot', ['lact', 'crea', 'plt', 'wbc', 'hgb', 'bili', 'glu'], signalIndex),
      demoCategorySection('scores', 'Scores and sepsis flags', ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs'], signalIndex),
      demoCategorySection('support', 'Support and therapies', ['mech_vent', 'vent_ind', 'rrt', 'norepi_rate', 'epi_rate', 'peep'], signalIndex),
    ];
    const summary = {
      entities: DEMO_ENTITY_COUNT,
      modules: modules.length,
      file_count: modules.length,
      total_rows: totalRows,
      review_entities: DEMO_ENTITY_COUNT,
      review_entity_cap: DEMO_ENTITY_COUNT,
      review_scope: 'catalog_seeded_demo_full_feature_set',
      static_aggregate_scope: 'easyicu_catalog_seeded_demo',
      dynamic_aggregate_scope: 'seeded_48h_demo_window',
      mean_age: 63.4,
      female_pct: 42.6,
      mortality: 20.8,
      median_los_icu: 5.4,
      median_sofa2: 7,
      sepsis_pct: 62.5,
    };
    const tablePreviews = tableModules.slice(0, 32).map((row, moduleIdx) => {
      const timeIndexed = row.shape === 'time_indexed';
      const features = (row.preview_features || []).map(f => f.feature || f.name).filter(Boolean).slice(0, timeIndexed ? 6 : 8);
      const displayColumns = ['entity'].concat(timeIndexed ? ['charttime'] : []).concat(features);
      const previewLimit = timeIndexed ? 24 : 8;
      const previewRows = Array.from({ length: previewLimit }, (_, idx) => {
        const context = demoTablePreviewRowContext(idx, timeIndexed);
        const out = { entity: context.entityRef };
        if (timeIndexed) out.charttime = context.charttime;
        features.forEach((feature, featureIdx) => {
          out[feature] = demoTableValue(feature, context.valueSeed + featureIdx + moduleIdx);
        });
        return out;
      });
      return {
        module: row.module,
        label: row.label,
        file: `${row.module}.demo`,
        rows_total: row.rows,
        columns_total: Math.max(row.review_features + (timeIndexed ? 2 : 1), displayColumns.length),
        display_columns: displayColumns,
        hidden_columns: Math.max(0, row.review_features - features.length),
        row_cap: previewRows.length,
        column_cap: displayColumns.length,
        pseudonymous_entity_column: true,
        status: 'ready',
        rows: previewRows,
        row_count: previewRows.length,
        truncated_rows: row.rows > previewRows.length,
        truncated_columns: row.review_features > features.length,
        payload_scope: 'seeded_pseudonymous_module_table_preview',
      };
    });
    return {
      ok: true,
      mode: 'demo',
      demo: true,
      source: { label: 'Demo · EasyICU feature catalog', database: 'demo', path_hash: 'catalog-demo' },
      provenance: {
        computed_from: ['window.EU_CATALOG.groups', 'window.EU_CATALOG.groupConcepts', 'deterministic_seeded_patient_signals'],
        payload_scope: 'catalog_shaped_seeded_demo_no_real_patient_rows',
        signals: 'deterministic_seeded_values_for_ui_preview',
      },
      privacy: {
        raw_rows_returned: false,
        direct_identifiers_returned: false,
        max_entity_options: 5,
        max_points_per_signal: 12,
        payload_tables_are_aggregated: true,
      },
      summary,
      eligibility_flow: {
        title: 'Eligibility flow (ICU stays)',
        title_i18n: { en: 'Eligibility flow (ICU stays)', zh: '入组筛选流程（ICU 住院）' },
        has_stepwise_report: true,
        payload_scope: 'demo_cohort_attrition_metadata_only',
        privacy: { patient_rows_returned: false, direct_identifiers_returned: false },
        steps: [
          {
            id: 'source_total',
            label: 'All ICU stays',
            label_i18n: { en: 'All ICU stays', zh: '全部 ICU 住院' },
            count: 72,
            denominator: 72,
            pct_of_initial: 100,
            excluded: null,
            excluded_pct_of_previous: null,
            note: 'catalog-shaped demo source pool',
            note_i18n: { en: 'catalog-shaped demo source pool', zh: '目录形演示来源池' },
            basis: 'seeded_demo',
          },
          {
            id: 'adult_stay_filter',
            label: 'Adult first ICU stay',
            label_i18n: { en: 'Adult first ICU stay', zh: '成人首次 ICU 住院' },
            count: 56,
            denominator: 72,
            pct_of_initial: 77.8,
            excluded: 16,
            excluded_pct_of_previous: 22.2,
            note: 'age >= 18 · first stay',
            note_i18n: { en: 'age >= 18 · first stay', zh: '年龄 ≥ 18 · 首次 ICU' },
            basis: 'seeded_demo',
          },
          {
            id: 'target_clinical_cohort',
            label: 'Sepsis-3 cohort',
            label_i18n: { en: 'Sepsis-3 cohort', zh: 'Sepsis-3 脓毒症队列' },
            count: DEMO_ENTITY_COUNT,
            denominator: 72,
            pct_of_initial: Number((DEMO_ENTITY_COUNT / 72 * 100).toFixed(1)),
            excluded: 8,
            excluded_pct_of_previous: 14.3,
            note: 'suspected infection + SOFA signal',
            note_i18n: { en: 'suspected infection + SOFA signal', zh: '疑似感染 + SOFA 信号' },
            basis: 'seeded_demo',
          },
          {
            id: 'final_cohort',
            label: 'Final review cohort',
            label_i18n: { en: 'Final review cohort', zh: '最终审阅队列' },
            count: DEMO_ENTITY_COUNT,
            denominator: 72,
            pct_of_initial: Number((DEMO_ENTITY_COUNT / 72 * 100).toFixed(1)),
            excluded: 0,
            excluded_pct_of_previous: 0,
            note: 'UI preview only',
            note_i18n: { en: 'UI preview only', zh: '仅用于界面预览' },
            basis: 'seeded_demo',
            final: true,
          },
        ],
      },
      module_profiles: moduleProfiles,
      entities,
      selected,
      time_lanes: timeLanes,
      quality,
      quality_metrics: {
        summary: qualitySummary,
        features: qualityRows.slice(0, 80),
        top_issues: topIssues,
        payload_scope: 'catalog_seeded_quality_metrics_no_row_payload',
      },
      data_tables: {
        loaded_summary: {
          entities: DEMO_ENTITY_COUNT,
          review_features: totalFeatures,
          observed_features: totalFeatures,
          module_count: modules.length,
          source_count: 1,
        },
        module_picker: {
          default_module: tableModules[0] && tableModules[0].module,
          module_count: tableModules.length,
          selection_mode: 'module_then_feature',
        },
        detail_gate: {
          title: 'Catalog-shaped demo; no source rows',
          default_open: false,
          reason: 'Demo rows are deterministic seeded values, while modules and feature names come from the real EasyICU concept catalog.',
          available_detail_modes: ['module_glance', 'single_feature_metadata'],
        },
        modules: tableModules,
        table_previews: tablePreviews,
        payload_scope: 'easyicu_catalog_demo_without_row_payload',
      },
      trajectory_review: {
        contract: [
          { index: '01', label: 'Entity scope', detail: `${entities.length} demo entity options exposed`, status: 'ready' },
          { index: '02', label: 'Loaded signals', detail: `${loadedSignals} catalog signals`, status: 'ready' },
          { index: '03', label: 'Feature matrices', detail: `${readyLanes.length} matrix groups available`, status: 'ready' },
          { index: '04', label: 'Review mode', detail: 'clinical lanes / single entity / same-feature comparison', status: 'ready' },
        ],
        modes: [
          { id: 'feature_matrix', label: 'Feature Matrix', status: 'ready', description: 'Bounded time-window by feature matrices for grouped EasyICU catalog signals.' },
          { id: 'single_entity', label: 'Single Patient', status: 'ready', description: 'Selected seeded demo entity trends and latest values.' },
          { id: 'multi_entity_comparison', label: 'Multi-Patient Comparison', status: 'ready', description: 'One selected feature compared across bounded pseudonymous entities.' },
        ],
        lanes: readyLanes,
        single_entity: { selected_ref: selected.ref, selected_label: selected.label, signals: selected.signals.slice(0, 12) },
        multi_entity_comparison: {
          selection_cap: 5,
          normalization_available: true,
          feature: compareFeature,
          label: compareMeta.name || compareFeature,
          unit: compareMeta.unit || '',
          module: compareModule,
          module_label: catalogModuleLabel(compareModule),
          traces: comparisonTraces,
          compared_entities: comparisonTraces.length,
          features: comparisonFeatures,
          payload_scope: 'seeded_pseudonymous_multi_entity_same_feature_traces',
        },
        payload_scope: 'catalog_demo_feature_matrix_semantics_bounded',
      },
      patient_overview: {
        navigator: {
          current: selected.label,
          ordinal: selected.ordinal,
          options: entities.map(item => ({ ref: item.ref, label: item.label, outcome: item.outcome, severity: item.severity })),
          actions: ['first', 'previous', 'next', 'last', 'random'],
        },
        dashboard: {
          mode: 'Dashboard',
          summary_cards: [
            { label: 'Age / sex', value: `${selected.demographics.age} / ${selected.demographics.sex}`, tone: 'neutral' },
            { label: 'SOFA-2 max', value: String(selected.scores.sofa2_max), tone: selected.scores.sofa2_max < 10 ? 'warn' : 'bad' },
            { label: 'Sepsis-3', value: selected.scores.sepsis3_sofa2 ? 'Positive' : 'Negative', tone: selected.scores.sepsis3_sofa2 ? 'warn' : 'ok' },
            { label: 'Outcome', value: selected.outcomes.status, tone: selected.outcomes.status === 'Deceased' ? 'bad' : 'ok' },
            { label: 'ICU LOS', value: `${selected.outcomes.icu_los_days} d`, tone: 'neutral' },
          ],
          trend_panels: sections.filter(s => s.available_count).slice(0, 3),
          sofa_comparator: signalIndex.sofa && signalIndex.sofa2
            ? { status: 'ready', features: [{ feature: 'sofa', label: 'SOFA-1', current: signalIndex.sofa.current, values: signalIndex.sofa.values }, { feature: 'sofa2', label: 'SOFA-2', current: signalIndex.sofa2.current, values: signalIndex.sofa2.values }] }
            : { status: 'unavailable', reason: 'SOFA-1 and SOFA-2 signals are both required.' },
        },
        category_view: { mode: 'Category View', sections },
        data_table: {
          mode: 'Data Table',
          available_features: totalFeatures,
          row_preview: 'blocked',
          reason: 'Demo preserves the Patient Overview contract without returning source rows.',
        },
        payload_scope: 'catalog_demo_patient_overview_semantics_pseudonymous',
      },
      quality_review: {
        summary_cards: [
          { label: 'QC concepts', value: qualitySummary.concept_count, tone: 'ok' },
          { label: 'Seeded observations', value: qualitySummary.total_records, tone: 'accent' },
          { label: 'Weighted missing', value: qualitySummary.weighted_missing_pct, unit: '%', tone: demoRateTone(qualitySummary.weighted_missing_pct, 5, 20) },
          { label: 'Out-of-physio', value: qualitySummary.weighted_out_of_physio_pct, unit: '%', tone: demoRateTone(qualitySummary.weighted_out_of_physio_pct, 1, 5) },
          { label: 'Duplicate TS', value: qualitySummary.weighted_duplicate_time_pct, unit: '%', tone: demoRateTone(qualitySummary.weighted_duplicate_time_pct, 0.5, 2) },
        ],
        contract: [
          { index: '01', label: 'Catalog concept scope', detail: `${qualitySummary.concept_count} concepts · ${DEMO_ENTITY_COUNT} demo entities · ${qualitySummary.total_records} seeded observations`, status: 'ready' },
          { index: '02', label: 'Missingness gate', detail: `${qualitySummary.weighted_missing_pct}% weighted missing`, status: demoRateTone(qualitySummary.weighted_missing_pct, 5, 20) },
          { index: '03', label: 'Physiologic range', detail: `${qualitySummary.weighted_out_of_physio_pct}% out-of-range values`, status: demoRateTone(qualitySummary.weighted_out_of_physio_pct, 1, 5) },
          { index: '04', label: 'Temporal integrity', detail: `${qualitySummary.weighted_duplicate_time_pct}% duplicate time rows`, status: demoRateTone(qualitySummary.weighted_duplicate_time_pct, 0.5, 2) },
        ],
        panels: [
          { id: 'missingness', label: 'Missingness', rows: demoQualityPanelRows(qualityRows.slice().sort((a, b) => b.missing_pct - a.missing_pct), 'missing_pct') },
          { id: 'outliers', label: 'Out-of-Physio', rows: demoQualityPanelRows(qualityRows.slice().sort((a, b) => b.out_of_physio_pct - a.out_of_physio_pct), 'out_of_physio_pct') },
          { id: 'temporal', label: 'Temporal Integrity', rows: demoQualityPanelRows(qualityRows.slice().sort((a, b) => b.duplicate_time_pct - a.duplicate_time_pct), 'duplicate_time_pct') },
        ],
        top_issues: topIssues,
        module_coverage: quality,
        payload_scope: 'catalog_demo_quality_semantics_aggregate_only',
      },
      blocked_features: [
        { id: 'demo_not_manuscript_result', status: 'blocked', reason: 'Seeded demo values exercise the UI only; load a real export for analysis.' },
        { id: 'raw_identifier_table', status: 'blocked', reason: 'Patient Review returns aggregates and pseudonymous demo entities only.' },
      ],
    };
  }

  /* ---------------- PATIENT REVIEW ---------------- */
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
      const previews = patientTablePreviews(drill);
      const activePreview = activePatientTablePreview(drill);
      const previewColumns = activePreview && Array.isArray(activePreview.display_columns) ? activePreview.display_columns : [];
      const previewRows = activePreview && Array.isArray(activePreview.rows) ? activePreview.rows : [];
      const previewPage = (activePreview && activePreview.pagination) || {};
      const page = Number(previewPage.page || activePreview && activePreview.page || patientTablePage || 1);
      const pageCount = Number(previewPage.page_count || activePreview && activePreview.page_count || 1);
      const rowStart = Number(previewPage.row_start || activePreview && activePreview.row_start || 0);
      const rowEnd = Number(previewPage.row_end || activePreview && activePreview.row_end || 0);
      const hasPrevious = Boolean(previewPage.has_previous || activePreview && activePreview.has_previous);
      const hasNext = Boolean(previewPage.has_next || activePreview && activePreview.has_next);
      patientTablePage = page;
      patientTablePageSize = Number(previewPage.page_size || activePreview && activePreview.page_size || patientTablePageSize || 24);
      const basisScope = drill.demo ? 'catalog-seeded demo aggregate' : 'demographics aggregate';
      const rows = [
        ['Entities', fmtInt(s.entities), drill.demo ? 'seeded demo denominator' : 'cohort denominator from active export'],
        ['Mean age', fmtNum(s.mean_age, 1), basisScope],
        ['Female', fmtPct(s.female_pct), basisScope],
        ['Mortality', fmtPct(s.mortality), drill.demo ? 'catalog-seeded demo outcome' : 'outcome aggregate'],
        ['Median SOFA-2', fmtNum(s.median_sofa2, 1), drill.demo ? 'catalog-seeded demo score' : 'score aggregate'],
        ['Sepsis-3 positive', fmtPct(s.sepsis_pct), drill.demo ? 'catalog-seeded demo event' : 'event aggregate'],
      ];
      const reviewModules = (dt.modules && dt.modules.length ? dt.modules : modules).slice(0, drill.demo ? 32 : 64);
      const activeModule = reviewModules.find(m => m.module === picker.default_module) || reviewModules[0] || {};
      const previewFeatures = activeModule.preview_features || [];
      const workspaceCopy = drill.demo
        ? t('catalog-shaped seeded demo: module counts and feature names come from the EasyICU concept catalog; seeded values are UI preview only.', '目录形演示：模块数量和特征名来自 EasyICU 概念目录，数值只是界面预览。')
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
      <div class="note ok mt-16">
        <div class="ico">${icon('rows', 16)}</div>
        <div class="body"><span class="t">${t('Table preview', '表格预览')}</span> <span class="d" style="display:inline;">— ${drill.demo ? t('Seeded demo rows for UI preview.', '演示行仅用于界面预览。') : t('Capped local rows from the active export; identifiers are replaced by pseudonymous entity tokens.', '来自当前本地导出的有界行预览；标识符已替换为去标识化实体 token。')}</span></div>
      </div>
      ${patientEligibilityFlow(drill.eligibility_flow)}
      ${previews.length ? `
      <div class="row wrap gap-6 mt-12" data-pt-table-picker>
        ${previews.map(p => `<button type="button" class="chip ${activePreview && p.module === activePreview.module ? 'solid' : ''}" data-pt-table-module="${esc(p.module)}" style="${activePreview && p.module === activePreview.module ? 'border-color:var(--ink);color:var(--ink);' : ''}">${esc(patientModuleLabel(p))} <span class="mono" style="font-size:10.5px;color:var(--ink-4);">${fmtInt(p.rows_total)} ${t('rows', '行')}</span></button>`).join('')}
      </div>
      <div class="patient-table-frame mt-12">
        <div class="patient-id-note">${drill.demo ? t('Demo entity tokens are seeded UI references.', '演示实体 token 是种子界面引用。') : t('Entity tokens are local pseudonymous references. Direct clinical identifiers stay on disk.', '实体 token 是本地伪匿名引用；直接临床标识符保留在磁盘上。')}</div>
        <div class="patient-table-scroll" data-patient-table-preview style="--pt-cols:${Math.max(6, previewColumns.length)};">
        <table class="eu-table patient-preview-table">
          <thead><tr>${previewColumns.map(c => `<th${c === 'entity' ? ' class="patient-entity-col"' : ' class="num"'}>${esc(patientColumnLabel(c, activePreview))}</th>`).join('')}</tr></thead>
          <tbody>
            ${previewRows.length ? previewRows.map(r => `<tr>${previewColumns.map(c => `<td class="${c === 'entity' ? 'key mono patient-entity-token' : 'num'}">${esc(fmtCell(r[c]))}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${Math.max(1, previewColumns.length)}" class="muted">${esc(activePreview && activePreview.reason ? activePreview.reason : t('No preview rows available for this module.', '这个模块没有可预览行。'))}</td></tr>`}
          </tbody>
        </table>
        </div>
      </div>
      <div class="patient-table-pager mt-8">
        <button type="button" class="btn sm" data-pt-page-prev ${hasPrevious ? '' : 'disabled'}>${icon('arrow-left', 13)} ${t('Previous', '上一页')}</button>
        <div class="patient-page-readout">
          <span class="mono">${esc(activePreview && activePreview.module || '')}</span>
          <span>${rowStart && rowEnd ? `${fmtInt(rowStart)}-${fmtInt(rowEnd)}` : fmtInt(activePreview && activePreview.row_count)} / ${fmtInt(activePreview && activePreview.rows_total)} ${t('rows', '行')}</span>
          <span>${window.EU_LANG === 'zh' ? `第 ${fmtInt(page)} / ${fmtInt(pageCount)} 页` : `Page ${fmtInt(page)} / ${fmtInt(pageCount)}`}</span>
        </div>
        <label class="patient-page-size">${t('Rows', '行数')}
          <select data-pt-page-size>
            ${[24, 50, 100].map(n => `<option value="${n}" ${Number(patientTablePageSize) === n ? 'selected' : ''}>${n}</option>`).join('')}
          </select>
        </label>
        <button type="button" class="btn sm" data-pt-page-next ${hasNext ? '' : 'disabled'}>${t('Next', '下一页')} ${icon('arrow-right', 13)}</button>
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
        <div class="body"><span class="t">${esc(detailGate.title || 'Source records are optional')}</span> <span class="d" style="display:inline;">— ${esc(detailGate.reason || 'Native Patient Review exposes cohort aggregates and one pseudonymous entity drilldown. Direct identifier tables stay out of the browser payload.')}</span></div>
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

  function ptSignalKey(sig) {
    return String((sig && (sig.feature || sig.key || sig.name)) || '').toLowerCase();
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
    const lanes = Array.isArray(review.lanes) ? review.lanes : (drill && Array.isArray(drill.time_lanes) ? drill.time_lanes : []);
    const readyLanes = lanes.filter(lane => (lane.signals || []).length);
    if (drill && readyLanes.length) {
      return `
      ${patientEntityNavigator(drill, drill.selected, {
        detail: t('This tab restores the old clinical-lane review modes while keeping the current bounded entity controls.', '这里恢复旧版临床泳道审阅模式，同时保留当前有界实体切换。'),
      })}
      ${patientTimeSeriesWorkbench(drill, review, readyLanes)}
      <div class="note ok mt-16">
        <div class="ico">${icon('rows', 16)}</div>
        <div class="body"><span class="t">${t('Old review logic restored', '旧版审阅逻辑已恢复')}</span> <span class="d" style="display:inline;">— ${t('Clinical lanes, single-patient trajectories and same-feature multi-patient comparison are primary; exact value matrices remain available below as an audit view.', '临床泳道、单患者轨迹和多患者同特征对比是主视图；精确值矩阵保留在下方作为审计视图。')}</span></div>
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
      return `<div class="empty mt-16"><div class="glyph">${icon('viz', 22)}</div><div class="t">No bounded signals in this export</div><div class="d">The active export did not include supported vitals columns for the selected entity.</div></div>`;
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
        ? t('Coverage, missingness and physiologic-range flags are deterministic seeded values over the real EasyICU feature catalog. Load a real export for analysis-ready denominators.', '覆盖率、缺失率和生理范围标记是基于真实 EasyICU 特征目录的确定性演示值；分析级分母需要加载真实导出。')
        : t('Coverage, missingness, physiologic-range flags and duplicate timestamp rates are computed from bounded local columns. Formal claims remain locked to the evidence-bound agent path.', '覆盖率、缺失率、生理范围标记和重复时间戳率都从有界本地列计算；正式结论仍锁定在证据绑定的 Agent 路径。');
      return `
      <div class="note ok mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">${patientQualityText('Quality dashboard')}</span> <span class="d" style="display:inline;">— ${t('QC workbook semantics: module coverage, missingness, physiologic range, temporal integrity, and action-oriented issues.', '质控工作簿语义：模块覆盖、缺失率、生理范围、时间完整性和可处理的问题清单。')}</span></div>
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
        <div class="eyebrow" style="margin-bottom:6px;">${patientQualityText('Per-module entity coverage')}</div>
        ${drill.quality.map(q => `
          <div class="qrow"><span>${esc(q.module)}</span><div class="qbar ${q.quality_status === 'ok' ? '' : q.quality_status}"><span style="width:${q.coverage_pct == null ? 0 : Math.max(0, Math.min(100, q.coverage_pct))}%"></span></div><span class="qv">${q.coverage_pct == null ? fmtInt(q.rows) : fmtPct(q.coverage_pct)}</span></div>`).join('')}
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
        <div class="eyebrow" style="margin-bottom:6px;">Per-module stay-id presence</div>
        ${ws.quality.map(q => `
          <div class="qrow"><span>${esc(q.module || q.file)}</span><div class="qbar ${q.status === 'ok' ? '' : q.status}"><span style="width:${q.coverage_pct == null ? 0 : Math.max(0, Math.min(100, q.coverage_pct))}%"></span></div><span class="qv">${q.coverage_pct == null ? fmtInt(q.rows) : fmtPct(q.coverage_pct)}</span></div>`).join('')}
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><div class="t">Local export snapshot</div><div class="d">Percentages are unique stay_id values found in each file divided by the loaded stay set. Event-only modules can be sparse by design; analysis gates still resolve denominators separately.</div></div>
      </div>`;
    }
    const cov = [['Vitals', 98, 'ok'], ['Labs', 88, 'ok'], ['SOFA / SOFA-2', 94, 'ok'], ['Sepsis-3', 90, 'ok'], ['Fluids', 72, 'warn'], ['Ventilation', 58, 'bad']];
    return `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">Per-concept coverage</div>
        ${cov.map(([n, pct, c]) => `
          <div class="qrow"><span>${n}</span><div class="qbar ${c === 'ok' ? '' : c}"><span style="width:${pct}%"></span></div><span class="qv">${pct}%</span></div>`).join('')}
      </div>
      <div class="note warn mt-16">
        <div class="ico">${icon('beaker', 16)}</div>
        <div class="body"><div class="t">Ventilation coverage below threshold</div><div class="d">58% of stays have ventilation fields; the agent will flag affected denominators before any analysis uses them.</div></div>
      </div>`;
  }

  function patientTabBody() {
    switch (patientTab) {
      case 'tables': return ptTables();
      case 'series': return ptSeries();
      case 'patient': return ptPatient();
      case 'quality': return ptQuality();
    }
  }
  function refreshPatientTablePage() {
    if (window.EU_DATA !== 'real') {
      repaintScreen('patient');
      return;
    }
    patientView = 'loading';
    repaintScreen('patient');
    loadRealPatient(ok => { patientView = ok ? 'loaded' : 'idle'; repaintScreen('patient'); });
  }
  function bindPatientTableControls(root) {
    root.querySelectorAll('[data-pt-table-module]').forEach(b => b.addEventListener('click', e => {
      e.preventDefault();
      const nextModule = b.dataset.ptTableModule || null;
      if (nextModule === patientTableModule) return;
      patientTableModule = nextModule;
      patientTablePage = 1;
      refreshPatientTablePage();
    }));
    const prev = root.querySelector('[data-pt-page-prev]');
    if (prev) prev.addEventListener('click', e => {
      e.preventDefault();
      if (prev.disabled || patientTablePage <= 1) return;
      patientTablePage = Math.max(1, patientTablePage - 1);
      refreshPatientTablePage();
    });
    const next = root.querySelector('[data-pt-page-next]');
    if (next) next.addEventListener('click', e => {
      e.preventDefault();
      if (next.disabled) return;
      patientTablePage += 1;
      refreshPatientTablePage();
    });
    const pageSize = root.querySelector('[data-pt-page-size]');
    if (pageSize) pageSize.addEventListener('change', () => {
      const parsed = Number(pageSize.value);
      patientTablePageSize = [24, 50, 100].includes(parsed) ? parsed : 24;
      patientTablePage = 1;
      refreshPatientTablePage();
    });
  }
  function bindPatientSeriesControls(root) {
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
    root.querySelectorAll('[data-patient-entity]').forEach(b => b.addEventListener('click', () => {
      const ref = b.dataset.patientEntity;
      if (!ref) return;
      if (window.EU_DATA !== 'real') {
        const payload = buildDemoPatientDrilldown(ref);
        window.EU_PATIENT_DRILLDOWN = payload;
        window.EU_VIZ_WORKSPACE = patientWorkspaceFromDrilldown(payload);
        patientView = 'loaded';
        repaintScreen('patient');
        return;
      }
      patientView = 'loading';
      repaintScreen('patient');
      loadRealPatient(ok => { patientView = ok ? 'loaded' : 'idle'; repaintScreen('patient'); }, ref);
    }));
  }

  S.patient = {
    section: 'viz', nav: 'viz', sub: 'patient',
    crumbs: ['Home', 'Data Visualization', 'Patient Review'],
    get actionHtml() {
      return patientView === 'loaded'
        ? `<button class="btn" data-viz-reset>${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button><button class="btn primary" data-gen>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>`
        : `<button class="btn primary" data-gen ${patientView === 'loading' ? 'aria-disabled="true"' : ''}>${icon('play', 13)} ${t('Render', '渲染')}</button>`;
    },
    rail: () => vizRail('patient'),
    render() {
      if (patientView === 'loading') {
        return `<div class="card pad">${skeletonWorkspace()}</div>`;
      }
      if (patientView === 'loaded') {
        const drill = patientDrilldown();
        const ws = window.EU_VIZ_WORKSPACE;
        const s = drill ? drill.summary : (ws && ws.summary);
        const readyTitle = drill
          ? (drill.demo ? t('Catalog-shaped demo review workspace ready', '目录形演示审阅工作区已就绪') : t('Local export patient drilldown ready', '本地导出患者明细已就绪'))
          : (ws ? t('Local export workspace ready', '本地导出工作区已就绪') : t('Demo review workspace ready', '演示审阅工作区已就绪'));
        const reviewStats = drill && s && s.review_scope === 'browser_bounded_entity_sample'
          ? ` · ${t('browser review', '浏览器审阅')} ${fmtInt(s.review_entities)}/${fmtInt(s.entities)} ${t('entities', '个实体')}`
          : '';
        const loadedFeatureCount = drill && drill.data_tables && drill.data_tables.loaded_summary
          ? drill.data_tables.loaded_summary.review_features
          : null;
        const readyStats = s
          ? (drill && drill.demo
            ? `${fmtInt(s.entities)} ${t('seeded entities', '个种子实体')} · ${fmtInt(s.modules)} ${t('modules', '个模块')} · ${fmtInt(loadedFeatureCount)} ${t('catalog features', '个目录特征')}`
            : `${fmtInt(s.entities != null ? s.entities : s.stays)} ${drill ? t('entities', '个实体') : t('stays', '次住院')} · ${fmtInt(s.modules)} ${t('modules', '个模块')} · ${fmtInt(s.total_rows)} ${t('rows', '行')}${reviewStats}`)
          : `48 ${t('seeded entities', '个种子实体')} · 19 ${t('modules', '个模块')} · ${t('catalog features', '目录特征')}`;
        const demoLoadedNote = (drill && drill.demo) ? `
        <div class="note warn mt-12">
          <div class="ico">${icon('beaker', 16)}</div>
          <div class="body">
            <div class="t">${t('Catalog-shaped seeded demo', '目录形种子演示')}</div>
            <div class="d">${t('Modules and feature names come from the real EasyICU catalog; values and row counts are deterministic seeded UI examples, not a local export or manuscript result.', '模块与特征名取自真实的 EasyICU 目录；数值和行数是确定性的界面种子示例，不是本地导出或稿件结果。')}</div>
          </div>
          <button class="btn sm" data-patient-use-real>${icon('db', 13)} ${t('Use real export', '使用真实导出')}</button>
        </div>` : (!drill && !ws) ? `
        <div class="note warn mt-12">
          <div class="ico">${icon('beaker', 16)}</div>
          <div class="body">
            <div class="t">${t('Seeded demo workspace', '种子演示工作区')}</div>
            <div class="d">${t('This tab is showing seeded demo rows. The real Patient Review backend appears after switching to Real and loading a local export; it shows module table overview, feature matrices, and concept-level quality metrics.', '此标签页显示的是种子演示行。切换到真实模式并加载本地导出后会出现真实的患者审阅后端；它展示模块表概览、特征矩阵和概念级质量指标。')}</div>
          </div>
          <button class="btn sm" data-patient-use-real>${icon('db', 13)} ${t('Use real export', '使用真实导出')}</button>
        </div>` : '';
        return `
        <div class="loaded-bar">
          <span class="pill ok"><span class="dot"></span>${t('Loaded', '已加载')}</span>
          <div class="grow"><span style="font-weight:600;font-size:13px;">${readyTitle}</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${readyStats}</span></div>
          <button class="btn sm" data-viz-reset>${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button>
          <button class="btn sm" data-patient-export>${icon('download', 13)} ${t('Export', '导出')}</button>
        </div>
        ${demoLoadedNote}
        <div class="mt-16">${patientTabs()}</div>
        <div id="ptbody">${patientTabBody()}</div>
        <div class="nextbar accent mt-16">
          <div class="nb-ico">${icon('arrow', 16)}</div>
          <div class="grow"><div class="nb-t">${t('Reviewed the data — what\u2019s next?', '\u6570\u636e\u5df2\u5ba1\u9605 \u2014\u2014 \u4e0b\u4e00\u6b65\uff1f')}</div><div class="nb-d">${t('Compare groups in Cohort Statistics, or assemble an auditable analysis and review-ready draft in Agent Projects.', '\u5728\u300c\u961f\u5217\u7edf\u8ba1\u300d\u505a\u7ec4\u95f4\u5bf9\u6bd4\uff0c\u6216\u5728\u300c\u7814\u7a76\u9879\u76ee\u300d\u7ec4\u88c5\u53ef\u5ba1\u8ba1\u5206\u6790\u4e0e\u5f85\u6838\u9a8c\u8349\u7a3f\u3002')}</div></div>
          <button class="btn" data-nav="cohort">${icon('cohort', 13)} ${t('Cohort Statistics', '\u961f\u5217\u7edf\u8ba1')}</button>
          <button class="btn primary" data-nav="agent">${icon('agent', 13)} ${t('Analyze in Agent', '\u8fdb\u5165\u7814\u7a76\u9879\u76ee')}</button>
        </div>`;
      }
      /* idle */
      return `
      <div class="card pad">
        <div class="panel-head">
          <div>
            <div class="eyebrow">${t('Quick visualization', '快速可视化')}</div>
            <div class="panel-title" style="margin-top:4px;font-size:17px;">${t('Load a review workspace', '加载审阅工作区')}</div>
            <div class="panel-sub">${window.EU_DATA === 'real' ? t('Load a local EasyICU export folder. Nothing is uploaded.', '加载本地 EasyICU 导出文件夹，不上传任何数据。') : t('Start with exported EasyICU tables or generate a catalog-shaped demo; review tabs appear immediately after loading.', '从已导出的 EasyICU 数据表开始，或生成目录形演示；加载后审阅标签页立即出现。')}</div>
          </div>
        </div>

        <div style="border-top:1px solid var(--hair);padding-top:16px;">
          <div class="eyebrow" style="margin-bottom:10px;">${t('Data source', '数据源')}</div>
          <div class="radio-row">
            <label class="radio ${window.EU_DATA === 'real' ? 'on' : ''}" role="button" tabindex="0" data-datamode="real"><span class="mk"></span> ${t('Previously exported data', '此前导出的数据')}</label>
            <label class="radio ${window.EU_DATA !== 'real' ? 'on' : ''}" role="button" tabindex="0" data-datamode="demo"><span class="mk"></span> ${t('Demo data', '演示数据')}</label>
          </div>
        </div>

        <div class="card sunken pad mt-16">
          <div class="eyebrow" style="margin-bottom:4px;">${window.EU_DATA === 'real' ? t('Local export', '本地导出') : t('Demo review', '演示审阅')}</div>
          <div style="font-weight:600;font-size:14px;">${window.EU_DATA === 'real' ? t('Load exported EasyICU tables', '加载已导出的 EasyICU 数据表') : t('Generate a catalog-shaped demo review workspace', '生成目录形演示审阅工作区')}</div>
          <div class="panel-sub" style="margin-top:2px;">${window.EU_DATA === 'real' ? t('Pick a registered local export, or add one by path.', '选择已注册的本地导出，或按路径添加一个。') : t('Uses the real EasyICU feature catalog for tables, trends, patient overview, and quality checks; seeded values are only for UI preview.', '表格、趋势、患者概览和质量检查均使用真实的 EasyICU 特征目录；种子数值仅用于界面预览。')}</div>
          ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
          ${window.EU_DATA === 'real' ? `
          ${patientSourceReadyCard()}
          ${sourceRegistryBlock('single')}
          <p style="font-size:11.5px;color:var(--ink-4);margin:14px 0 0;">${t('Use Data Extraction first to create or refresh this folder. The last successful export is remembered locally.', '请先用数据抽取创建或刷新该文件夹。上次成功的导出会被本地记住。')}</p>
          <button class="btn primary block lg mt-16" data-gen>${icon('folder', 14)} ${t('Load local export', '加载本地导出')}</button>` : `
          <div class="cols-2 mt-16" style="gap:28px;">
            <div>
              <div class="row" style="justify-content:space-between;"><label style="font-size:12.5px;font-weight:500;color:var(--ink-2);">${t('Number of patients', '患者数量')}</label><span class="mono" style="font-size:12px;">48</span></div>
              <div class="slider"><div class="track"><div class="fill" style="width:100%"></div><div class="knob" style="left:100%"></div></div><div class="ends"><span>10</span><span>48</span></div></div>
            </div>
            <div>
              <div class="row" style="justify-content:space-between;"><label style="font-size:12.5px;font-weight:500;color:var(--ink-2);">${t('Data duration (hours)', '数据时长（小时）')}</label><span class="mono" style="font-size:12px;">48</span></div>
              <div class="slider"><div class="track"><div class="fill" style="width:100%"></div><div class="knob" style="left:100%"></div></div><div class="ends"><span>24</span><span>48</span></div></div>
            </div>
          </div>
          <p style="font-size:11.5px;color:var(--ink-4);margin:14px 0 0;">${t('Demo profile: all EasyICU catalog modules and concepts, with seeded ICU-like values for preview only.', '演示配置：全部 EasyICU 目录模块与概念，配以种子化的类 ICU 数值，仅供预览。')}</p>
          <button class="btn primary block lg mt-16" data-gen>${icon('play', 14)} ${t('Generate and load demo workspace', '生成并加载演示工作区')}</button>`}
        </div>
      </div>

      <div class="empty mt-16">
        <div class="glyph">${icon('viz', 22)}</div>
        <div class="t">${t('Preview workspace awaits data', '预览工作区等待数据')}</div>
        <div class="d">${t('Generate demo data or load exported files above; the review tabs will appear here as a compact multi-view workspace.', '在上方生成演示数据或加载导出文件；审阅标签页会作为紧凑的多视图工作区出现在这里。')}</div>
      </div>`;
    },
    afterRender(root) {
      bindSourceRegistry(root, 'patient');
      if (window.EU_DATA === 'real' && patientView === 'idle' && !window.EU_PATIENT_SOURCES && !window.EU_PATIENT_SOURCES_LOADING) {
        loadPatientSources(ok => {
          if (ok && window.location.hash === '#patient' && patientView === 'idle') repaintScreen('patient');
        });
      }
      root.querySelectorAll('.radio[data-datamode]').forEach(b => b.addEventListener('keydown', e => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        e.preventDefault();
        if (window.setDataMode) window.setDataMode(b.dataset.datamode);
      }));
      root.querySelectorAll('[data-gen]').forEach(b => b.addEventListener('click', () => {
        if (patientView === 'loading') return;
        patientTablePage = 1;
        patientView = 'loading';
        repaintScreen('patient');
        if (window.EU_DATA === 'real') {
          loadRealPatient(ok => { patientView = ok ? 'loaded' : 'idle'; repaintScreen('patient'); });
        } else {
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
        patientView = 'idle'; patientTablePage = 1; window.EU_VIZ_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; repaintScreen('patient');
      }));
      root.querySelectorAll('[data-patient-use-real]').forEach(b => b.addEventListener('click', () => {
        patientView = 'idle';
        patientTablePage = 1;
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

  /* ---------------- COHORT STATISTICS ---------------- */
  /* In-page panels (aligned with cohort_redesign.py _SUBTABS): Group contrast,
     Coverage audit, Cohort profile, SOFA reclassification. Demo-only panels use
     fixed catalog-shaped previews; real mode still requires a cohort-review payload. */
  function cohortTabs() {
    const tabs = [
      ['groups',   t('Group contrast', '组间对照'),       'layers'],
      ['survival', t('Survival curves', '生存曲线'),       'chart'],
      ['coverage', t('Coverage audit', '覆盖审计'),      'shield'],
      ['snapshot', t('Cohort profile', '队列画像'),       'cohort'],
      ['sofa',     t('SOFA reclassification', 'SOFA 重分层'), 'refresh'],
    ];
    return `<div class="tabs" id="cohtabs">${tabs.map(([k, lab, ic]) =>
      `<button class="tab ${cohortPanel === k ? 'active' : ''}" data-cohtab="${k}">${icon(ic, 14)} ${lab}</button>`).join('')}</div>`;
  }

  function cohortPanelBody() {
    const review = cohortReview();
    const demoLoaded = window.EU_DATA !== 'real' && cohortView === 'loaded';
    switch (cohortPanel) {
      case 'survival': return review ? cohortSurvivalBody(review) : cohortSurvivalDemoBody();
      case 'coverage': return review ? cohortCoverageBody(review) : (demoLoaded ? cohortCoverageBody(cohortDemoCoverageReview(), { demo: true }) : cohortUnavailablePanel('coverage'));
      case 'sofa':     return review ? cohortSofaBody(review) : (demoLoaded ? cohortSofaBody(cohortDemoSofaReview(), { demo: true }) : cohortUnavailablePanel('sofa'));
      case 'snapshot': return cohortSnapshotBody();
      default:         return cohortGroupsBody();
    }
  }

  function cohortUnavailablePanel(kind) {
    const isSofa = kind === 'sofa';
    return `
      <div class="state empty mt-16">
        <div class="ico">${icon(isSofa ? 'refresh' : 'shield', 18)}</div>
        <div class="t">${isSofa ? t('SOFA reclassification requires a real cohort review', 'SOFA 重分层需要真实队列审阅') : t('Coverage audit requires a real cohort review', '覆盖审计需要真实队列审阅')}</div>
        <div class="d">${t('The old seeded audit panel has been removed. Switch to Real mode, register an EasyICU export, then load Cohort Statistics to compute this aggregate-only panel.', '旧的种子审计面板已移除。请切换到真实模式，注册 EasyICU 导出，再加载队列统计以计算这个仅聚合的面板。')}</div>
      </div>`;
  }

  function cohortProfileValue(row, value) {
    if (value == null || value === '') return '—';
    if (row.kind === 'count') return fmtInt(value);
    if (row.kind === 'percent') return fmtPct(value);
    return fmtNum(value, 1);
  }
  function cohortText(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      'Workspace': '工作区',
      'Local export': '本地导出',
      'Demo cohort': '演示队列',
      'Not configured': '未配置',
      'Cohort statistics': '队列统计',
      'Agent preflight': 'Agent 预检',
      'current session': '当前会话',
      'Input package': '输入包',
      'Backend evidence checks': '后端证据检查',
      'Draft review': '草稿核验',
      'demo concept set': '演示概念集',
      'manifest parsed · denominators previewed · aggregate payload returned': 'manifest 已解析 · 分母已预览 · 聚合载荷已返回',
      'coverage + denominators ready': '覆盖率 + 分母已就绪',
      'locked · requires Agent sign-off': '已锁定 · 需要 Agent 签署',
      'Analysis table': '分析表',
      'Real cohort aggregate': '真实队列聚合',
      'Local export group contrast': '本地导出分组对照',
      'Cohort size': '队列规模',
      'Total stays': '总住院数',
      'Total patients': '总患者数',
      'Mean age': '平均年龄',
      'Median age': '年龄中位数',
      'Female': '女性',
      'Female %': '女性比例',
      'Male %': '男性比例',
      'Mortality': '死亡率',
      'Median SOFA-2': 'SOFA-2 中位数',
      'Median SOFA': 'SOFA 中位数',
      'Sepsis-3 +': 'Sepsis-3 阳性',
      'Local export cohort review ready': '本地导出队列审阅已就绪',
      'Source': '来源',
      'Database': '数据库',
      'Path hash': '路径哈希',
      'Scope': '范围',
      'aggregate-only payload': '仅聚合载荷',
      'Comparison': '对照',
      'Select descriptive split': '选择描述性分组',
      'Summary': '摘要',
      'Overview': '概览',
      'Descriptive profile': '描述性画像',
      'Aggregate-only group characteristics': '仅聚合的分组特征',
      'Metric': '指标',
      'Status': '状态',
      'descriptive': '描述性',
      'Fail-closed': '保守拦截',
      'Blocked cohort functions': '已拦截的队列功能',
      'row_level_filters': '行级筛选',
      'inferential_statistics': '推断统计',
      'matched_cohort': '匹配队列',
      'paired_sofa_reclassification': '配对 SOFA 重分层',
      'custom_threshold': '自定义阈值',
      'p_value_smd': 'p 值 / SMD',
      'blocked': '已拦截',
      'supported': '已支持',
      'Age Groups': '年龄分组',
      'Female vs Male': '女性 vs 男性',
      'Short vs Long Stay': '短住院 vs 长住院',
      'Survived vs Deceased': '存活 vs 死亡',
      'Sepsis vs Non-sepsis': 'Sepsis vs 非 Sepsis',
      'Survived': '存活',
      'Deceased': '死亡',
      'Non-sepsis': '非 Sepsis',
      'Sepsis': 'Sepsis',
      'Female': '女性',
      'Male': '男性',
      'Unknown': '未知',
      'Known': '已知',
      'Short/median': '短住院/中位数以下',
      'Long': '长住院',
      'N': 'N',
      'Mortality %': '死亡率 %',
      'Median ICU LOS': 'ICU 住院时长中位数',
      'years': '年',
      'days': '天',
      'Survival analysis': '生存分析',
      'Kaplan-Meier module': 'Kaplan-Meier 模块',
      'Demo simulated KM preview': '演示模拟 KM 预览',
      'Seeded demo only': '仅 seeded 演示',
      'Demo hospital mortality by Sepsis vs Non-sepsis': '演示院内死亡 · 按 Sepsis vs 非 Sepsis 分组',
      'Demo follow-up days': '演示随访天数',
      'Kaplan-Meier curves and log-rank': 'Kaplan-Meier 曲线与 log-rank',
      'Hospital mortality': '院内死亡',
      'ICU mortality': 'ICU 死亡',
      '28-day mortality': '28 天死亡',
      '30-day display window': '30 天显示窗口',
      '28-day window': '28 天窗口',
      'derived from hospital death + LOS': '由院内死亡 + 住院时长派生',
      'Hospital LOS / follow-up days': '住院时长 / 随访天数',
      'Outcome': '结局',
      'Outcome overview': '结局概览',
      'Grouping': '分组',
      'events': '事件',
      'No outcome module': '没有结局模块',
      'not available': '不可用',
      'KM-ready': '可画 KM',
      'KM curve endpoint': 'KM 曲线结局',
      'Event rate summary': '事件率',
      'rate only': '仅事件率',
      'time window': '时间窗',
      'unavailable': '不可用',
      'Survival analysis blocked': '生存分析已拦截',
      'Current export is loaded, but the cohort is above the interactive KM preview limit; continue with an audited local analysis job on this same export.': '当前导出已加载，但队列超过交互式 KM 预览上限；请在同一个导出上继续运行本地审计分析任务。',
      'Exploratory · unadjusted': '探索性 · 未调整',
      'Time-to-event': '事件时间',
      'Log-rank': 'Log-rank',
      'df 1 · exploratory only': 'df 1 · 仅探索',
      'not enough events': '事件数不足',
      'Not manuscript-ready by itself': '不能单独用于稿件结论',
      'Number at risk': '风险人数',
      'Group': '分组',
      'Days': '天',
      'Survival probability': '生存概率',
      'Real export required': '需要真实导出',
      'Coverage audit': '覆盖审计',
      'Real module coverage and quality': '真实模块覆盖率与质量',
      'Demo module coverage and quality': '演示模块覆盖率与质量',
      'Modules OK': '正常模块',
      'Watchlist': '观察名单',
      'Median coverage': '覆盖率中位数',
      'Neutral event modules': '事件/暴露模块',
      'Presence-rate modules': '事件/暴露模块',
      'Event/exposure rows show cohort incidence or exposure prevalence, not missingness coverage; they are excluded from the coverage watchlist.': '事件/暴露行显示队列发生率或暴露率，不是缺失覆盖率；它们不会进入低覆盖观察名单。',
      'Unknown coverage': '未知覆盖率',
      'Module': '模块',
      'Records': '记录数',
      'Fields': '字段数',
      'Covered entities': '覆盖实体',
      'Entities': '实体数',
      'Coverage': '覆盖率',
      'Coverage / rate': '覆盖率 / 发生率',
      'Event rate': '发生率',
      'Exposure rate': '暴露率',
      'Fail-closed scope': '保守拦截范围',
      'Interpretation': '解释',
      'Ready': '正常',
      'Watch': '观察',
      'Low coverage': '低覆盖',
      'Rate only': '仅比例',
      'SOFA reclassification': 'SOFA 重分层',
      'SOFA-2 aggregate review': 'SOFA-2 聚合审阅',
      'Demo SOFA-2 aggregate preview': '演示 SOFA-2 聚合预览',
      'Paired entities': '配对实体',
      'SOFA-2 higher': 'SOFA-2 更高',
      'SOFA-2 lower': 'SOFA-2 更低',
      'Median delta': '差值中位数',
      'SOFA-2 minus SOFA-1': 'SOFA-2 减 SOFA-1',
      'Mean SOFA-2': 'SOFA-2 均值',
      'Age': '年龄',
      'ICU LOS days': 'ICU 住院天数',
      'Min': '最小值',
      'Max': '最大值',
      'registered export aggregate': '注册导出聚合',
      'bounded column read': '有界列读取',
      'SOFA-2 severity bins': 'SOFA-2 严重度分箱',
      'SOFA-1 to SOFA-2 movement': 'SOFA-1 到 SOFA-2 变化',
      'Worst-ICU severity transition matrix': 'ICU 最严重 SOFA 转移矩阵',
      'Matrix value': '矩阵数值',
      'Percent': '百分比',
      'Count': '人数',
      'Granularity': '粒度',
      'Coarse': '粗略',
      'Medium': '中等',
      'Fine': '细粒度',
      'Exact': '逐分',
      '4 bands': '4 档',
      '6 bands': '6 档',
      '12 bands': '12 档',
      '25 scores': '25 分',
      'Rows are SOFA-1 severity bands; columns are SOFA-2 bands. Color intensity follows the selected value.': '行是 SOFA-1 严重度分层，列是 SOFA-2 分层；颜色深浅随当前显示值变化。',
      'Rows are SOFA-1 score bands; columns are SOFA-2 score bands. Use the granularity control to move from clinical bands to exact 0-24 scores.': '行是 SOFA-1 分数分箱，列是 SOFA-2 分数分箱。可用粒度控件从临床分层切到 0-24 逐分矩阵。',
      'Same severity band': '同一严重度层级',
      'SOFA-2 higher band': 'SOFA-2 更高层级',
      'SOFA-2 lower band': 'SOFA-2 更低层级',
      'Paired aggregate ready': '配对聚合已就绪',
      'Paired reclassification blocked': '配对重分层已拦截',
      'Cohort profile': '队列画像',
      'Real cohort aggregate': '真实队列聚合',
      'Aggregate ranges': '聚合范围',
      'Source provenance': '来源溯源',
      'Export measures': '导出指标',
      'Files loaded': '已加载文件',
      'Rows reviewed': '已审阅行数',
      'Outcome groups': '结局分组',
      'Table one': 'Table One',
      'Baseline characteristics comparison': '基线特征对照',
      'Characteristic': '特征',
      'Overall': '总体',
      'p-value': 'p 值',
      'Group Contrast Table': '分组对照表',
      'Select comparison mode': '选择对照模式',
      'Features': '特征',
      'Select feature modules': '选择特征模块',
      'Demographics': '人口统计',
      'Outcome': '结局',
      'Vital Signs': '生命体征',
      'Features to load': '待加载特征数',
      'Selected modules': '已选模块',
      'Catalog available': 'Catalog 可用范围',
      'Recommended modules': '推荐模块',
      'All catalog modules': '全部 Catalog 模块',
      'Load all modules': '加载全部模块',
      'Use recommended modules': '恢复推荐模块',
      'Default load': '默认加载',
      'Custom Threshold': '自定义阈值',
      'Short vs Long Stay': '短住院 vs 长住院',
      'Above threshold': '高于阈值',
      'Below threshold': '低于阈值',
      'Example': '示例',
      'Ratio': '比例',
      'Survived vs Deceased': '存活 vs 死亡',
      'Male vs Female': '男性 vs 女性',
      'Short vs Long Stay': '短住院 vs 长住院',
      'Age < 65': '年龄 < 65',
      'Age ≥ 65': '年龄 ≥ 65',
      'LOS < 5d': 'ICU 住院 < 5 天',
      'LOS ≥ 5d': 'ICU 住院 ≥ 5 天',
      'Sepsis-3 +': 'Sepsis-3 阳性',
      'Sepsis-3 -': 'Sepsis-3 阴性',
      'Age, mean (SD)': '年龄，均值（SD）',
      'Male, n (%)': '男性，n (%)',
      'SOFA, median': 'SOFA，中位数',
      'Lactate, mmol/L': '乳酸，mmol/L',
      'ICU LOS, days': 'ICU 住院天数',
      'Mortality, n (%)': '死亡，n (%)',
      'Sepsis-3, n (%)': 'Sepsis-3，n (%)',
      'Ventilation, n (%)': '机械通气，n (%)',
      'Real': '真实',
      'Demo': '演示',
    };
    if (Object.prototype.hasOwnProperty.call(map, raw)) return t(raw, map[raw]);
    if (/^SOFA-2 <= ([^ ]+) vs > ([^ ]+)$/.test(raw)) {
      const m = raw.match(/^SOFA-2 <= ([^ ]+) vs > ([^ ]+)$/);
      return t(raw, `SOFA-2 <= ${m[1]} vs > ${m[2]}`);
    }
    if (/^(.+) by (.+)$/.test(raw)) {
      const m = raw.match(/^(.+) by (.+)$/);
      return t(raw, `${cohortText(m[1])} · 按 ${cohortText(m[2])} 分组`);
    }
    return t(raw, raw);
  }
  function cohortReason(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      'Cohort Review accepts only registered-source aggregate review in Stage17.': 'Cohort Review 当前只接受已注册来源的聚合审阅。',
      'Generic Table One/group p-values, SMDs, and confidence intervals remain blocked; survival log-rank is scoped to the KM module when timed outcomes exist.': '通用 Table One / 分组 p 值、SMD 和置信区间仍被拦截；只有存在事件时间时才在 KM 模块中提供 log-rank。',
      'Matched cohorts belong to Cross-DB parity and audit-gated analysis.': '匹配队列属于跨库 parity 与审计后的分析流程。',
      'Custom group thresholds require row-level validation before display.': '自定义分组阈值显示前需要行级校验。',
      'Custom thresholds require audited row-level cohort construction.': '自定义阈值需要经过审计的行级队列构建。',
      'Inferential statistics are withheld until the numeric evidence audit gate.': '推断统计会等到数值证据审计后再开放。',
      'Matched cohorts require an audit-bound analysis plan.': '匹配队列需要绑定审计的分析计划。',
      'Matched cohort logic is not part of Stage17 Cohort Review.': '匹配队列逻辑不属于当前 Stage17 队列审阅范围。',
      'Table One p-values, SMDs, and row-level baseline tables require the numeric evidence audit gate. Survival log-rank is scoped to the audited KM module when time-to-event data exist.': 'Table One p 值、SMD 和行级基线表需要数值证据审计；有事件时间时，survival log-rank 只在已审计 KM 模块中提供。',
      'Paired SOFA-1/SOFA-2 reclassification is not available for this export.': '此导出无法做配对 SOFA-1/SOFA-2 重分层。',
      'No outcome has both an event column and a time-to-event/censoring column in this export.': '此导出没有同时具备事件列与事件时间/删失时间列的结局。',
      'No supported two-group split is available for this cohort.': '此队列没有可用的双组分组。',
      'No survival curve could be computed from the available timed records.': '可用时间记录不足，无法计算生存曲线。',
      'Current export is loaded, but the cohort is above the interactive KM preview limit; continue with an audited local analysis job on this same export.': '当前导出已加载，但队列超过交互式 KM 预览上限；请在同一个导出上继续运行本地审计分析任务。',
      'Outcome module is not present in the registered export.': '注册导出中没有结局模块。',
      'Fewer than two cohort entities have valid survival time values.': '有效生存时间值少于两个队列实体。',
      'Demo threshold uses SOFA ≥ 6. Real custom thresholds remain fail-closed until a bounded cohort-builder backend is available.': '演示阈值使用 SOFA ≥ 6。真实自定义阈值会在有界队列构建后端可用前保持保守拦截。',
      'This export does not expose an outcome with both event and time-to-event columns.': '此导出没有同时包含事件列和事件时间列的结局。',
      'ICU mortality is unavailable because this export does not include ICU-specific event and time columns.': 'ICU 死亡不可用，因为当前导出没有 ICU 专用死亡事件列和 ICU 时间列。',
      'ICU mortality is unavailable because this export does not include an ICU-specific event column.': 'ICU 死亡不可用，因为当前导出没有 ICU 专用死亡事件列。',
      'ICU-specific event column is not present in the registered export.': '当前注册导出没有 ICU 专用死亡事件列。',
      'ICU mortality event rate is available, but KM/log-rank needs ICU-specific time columns.': 'ICU 死亡事件率可用，但 KM/log-rank 需要 ICU 专用时间列。',
      'Unavailable for this export': '此导出不可用',
      'Unavailable': '不可用',
    };
    if (/^No event column found for (.+)\.$/.test(raw)) {
      const label = raw.match(/^No event column found for (.+)\.$/)[1];
      return t(raw, `未找到“${cohortText(label)}”事件列。`);
    }
    if (/^(.+) is available only as an event flag; KM\/log-rank needs time-to-event or censoring time\.$/.test(raw)) {
      const label = raw.match(/^(.+) is available only as an event flag; KM\/log-rank needs time-to-event or censoring time\.$/)[1];
      return t(raw, `“${cohortText(label)}”只有事件标志；KM/log-rank 需要事件时间或删失时间。`);
    }
    return t(raw, map[raw] || raw);
  }

  function cohortDemoCatalogScope() {
    const modules = demoCatalogModules();
    const byKey = new Map(modules.map(module => [module.module, module]));
    const defaultKeys = ['demographics', 'outcome', 'vitals', 'sepsis3_sofa2'];
    const recommended = defaultKeys.map(key => byKey.get(key)).filter(Boolean);
    const fallback = modules.slice(0, Math.min(4, modules.length));
    const selected = cohortFeatureScope === 'all' ? modules : (recommended.length ? recommended : fallback);
    const totalFeatureCount = (window.EU_CATALOG || {}).totalConcepts
      || modules.reduce((acc, module) => acc + (module.features || []).length, 0)
      || selected.reduce((acc, module) => acc + (module.features || []).length, 0);
    const selectedFeatureCount = selected.reduce((acc, module) => acc + (module.features || []).length, 0);
    return {
      allModules: modules,
      selectedModules: selected,
      isAll: cohortFeatureScope === 'all',
      totalModuleCount: modules.length,
      selectedModuleCount: selected.length,
      totalFeatureCount,
      selectedFeatureCount,
    };
  }

  function cohortDemoFeaturePicker() {
    const scope = cohortDemoCatalogScope();
    const chips = scope.selectedModules.map(module => `
      <span class="chip solid" title="${esc(module.module)}">
        ${esc(module.label)}
        <span class="mono" style="font-size:10.5px;color:inherit;opacity:.72;">${fmtInt((module.features || []).length)}</span>
      </span>`).join('');
    const nextScope = scope.isAll ? 'recommended' : 'all';
    const actionLabel = scope.isAll ? cohortText('Use recommended modules') : cohortText('Load all modules');
    const badgeLabel = scope.isAll ? cohortText('All catalog modules') : cohortText('Recommended modules');
    const scopeNote = scope.isAll
      ? t('All catalog modules are selected for this demo review. The simulated preview can take a little longer, but it still does not scan a real export.', '已选择全部 catalog 模块用于这次演示审阅；演示预览可能稍慢一点，但不会扫描真实导出。')
      : t('Default loads a focused subset; use Load all modules to include every catalog feature.', '默认只加载推荐模块；点击“加载全部模块”即可纳入所有 catalog 特征。');
    return `
      <div class="card pad" style="padding:14px 16px;" data-cohort-catalog-scope="${scope.isAll ? 'all' : 'recommended'}">
        <div class="row wrap gap-8" style="align-items:center;">
          <span class="pill ${scope.isAll ? 'ok' : 'demo'}"><span class="dot"></span>${badgeLabel}</span>
          <span style="font-size:12px;color:var(--ink-3);">${cohortText('Catalog available')}: ${fmtInt(scope.totalModuleCount)} ${t('modules', '个模块')} · ${fmtInt(scope.totalFeatureCount)} ${t('features', '个特征')}</span>
          <span class="grow"></span>
          <button class="btn sm" type="button" data-cohort-feature-scope="${nextScope}">${scope.isAll ? icon('sliders', 12) : icon('layers', 12)} ${actionLabel}</button>
        </div>
        <div class="row wrap gap-6" style="margin-top:10px;">${chips}</div>
        <div class="row wrap gap-10" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--hair);justify-content:space-between;">
          <span class="row gap-6" style="font-size:12px;color:var(--ink-3);">${icon('flask', 13)} ${cohortText('Features to load')}: ${fmtInt(scope.selectedFeatureCount)} / ${fmtInt(scope.totalFeatureCount)}</span>
          <span class="row gap-6" style="font-size:12px;color:var(--ink-3);">${icon('layers', 13)} ${cohortText('Selected modules')}: ${fmtInt(scope.selectedModuleCount)} / ${fmtInt(scope.totalModuleCount)}</span>
        </div>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:8px;">${scopeNote}</div>
      </div>`;
  }

  function cohortRealModuleSummary(review) {
    const rows = (review && review.coverage) || [];
    if (!rows.length) return '';
    const exactRows = rows.filter(row => row.coverage_basis === 'unique_entity_intersection');
    const metadataRows = rows.filter(row => row.coverage_basis === 'metadata_row_count_only');
    const okish = rows.filter(row => ['ok', 'neutral'].includes(row.quality_status)).length;
    const chips = rows.map(row => {
      const cls = row.quality_status === 'ok' || row.quality_status === 'neutral' ? 'solid' : (row.quality_status === 'unknown' ? 'demo' : '');
      const label = row.coverage_basis === 'metadata_row_count_only'
        ? t('manifest rows', '清单行数')
        : cohortCoverageMetricValue(row);
      return `<span class="chip ${cls}" title="${esc(row.module)}">${esc(row.module)} <span class="mono" style="font-size:10.5px;color:inherit;opacity:.72;">${label}</span></span>`;
    }).join('');
    return `
      <div class="card pad mt-14" data-cohort-real-modules>
        <div class="row wrap gap-8" style="align-items:center;">
          <span class="pill ok"><span class="dot"></span>${t('Current export loaded', '当前导出已加载')}</span>
          <span style="font-size:12px;color:var(--ink-3);">${fmtInt(rows.length)} ${t('modules', '个模块')} · ${fmtInt(okish)} ${t('ready or event modules', '个已就绪/事件模块')} · ${fmtInt(exactRows.length)} ${t('exact coverage scans', '个精确覆盖扫描')}</span>
          <span class="grow"></span>
          <button class="btn sm" type="button" data-cohgo="coverage">${icon('shield', 12)} ${t('Open coverage audit', '打开覆盖审计')}</button>
        </div>
        <div class="row wrap gap-6" style="margin-top:10px;">${chips}</div>
        ${metadataRows.length ? `<div style="font-size:11.5px;color:var(--ink-4);margin-top:8px;">${t('Some non-Parquet or very large modules may show manifest-confirmed row counts instead of exact unique-stay coverage; they are still part of this export.', '部分非 Parquet 或超大模块可能显示清单确认的行数，而不是精确唯一 stay 覆盖率；它们仍属于当前导出。')}</div>` : ''}
      </div>`;
  }

  function cohortCoverageMetricLabel(row) {
    if (!row || row.coverage_basis === 'metadata_row_count_only') return t('row count only', '仅行数');
    if (row.metric_kind === 'event_rate') return cohortText('Event rate');
    if (row.metric_kind === 'exposure_rate') return cohortText('Exposure rate');
    return cohortText('Coverage');
  }

  function cohortCoverageMetricValue(row) {
    if (!row) return '—';
    if (row.coverage_basis === 'metadata_row_count_only') return t('row count only', '仅行数');
    return `${cohortCoverageMetricLabel(row)} ${fmtPct(row.coverage_pct)}`;
  }

  function cohortQualityStatusClass(row) {
    if (!row) return 'demo';
    if (row.coverage_basis === 'metadata_row_count_only') return 'ok';
    if (row.metric_kind === 'event_rate' || row.metric_kind === 'exposure_rate') return 'ok';
    if (row.quality_status === 'ok' || row.quality_status === 'neutral') return 'ok';
    if (row.quality_status === 'unknown') return 'demo';
    return 'warn';
  }

  function cohortQualityStatusLabel(row) {
    if (!row) return cohortText('Unknown');
    if (row.coverage_basis === 'metadata_row_count_only') return t('loaded', '已加载');
    if (row.metric_kind === 'event_rate') return cohortText('Event rate');
    if (row.metric_kind === 'exposure_rate') return cohortText('Exposure rate');
    if (row.quality_status === 'ok') return cohortText('Ready');
    if (row.quality_status === 'warn') return cohortText('Watch');
    if (row.quality_status === 'bad') return cohortText('Low coverage');
    if (row.quality_status === 'neutral') return cohortText('Rate only');
    return cohortText('Unknown');
  }

  function cohortRealFeaturePicker(review) {
    const catalog = (review && review.feature_catalog) || {};
    const selection = (review && review.feature_selection) || {};
    const modules = (catalog.modules || []).filter(module => module && module.feature_count);
    if (!modules.length) return '';
    const selectedRows = selection.selected || [];
    const selectedIds = new Set(selectedRows.map(row => row && row.id).filter(Boolean));
    const defaultIds = selection.default_ids || [];
    const maxSelected = selection.max_selected_features || catalog.max_selected_features || 48;
    if (cohortFeatureModule !== 'all' && !modules.some(module => module.module === cohortFeatureModule)) cohortFeatureModule = 'all';
    const visibleModules = cohortFeatureModule === 'all' ? modules : modules.filter(module => module.module === cohortFeatureModule);
    const moduleChips = [
      `<button class="chip ${cohortFeatureModule === 'all' ? 'solid' : ''}" type="button" data-cohort-feature-module="all">${t('All modules', '全部模块')} <span class="mono">${fmtInt(catalog.total_features)}</span></button>`,
      ...modules.map(module => `<button class="chip ${cohortFeatureModule === module.module ? 'solid' : ''}" type="button" data-cohort-feature-module="${esc(module.module)}">${esc(module.label || module.module)} <span class="mono">${fmtInt(module.feature_count)}</span></button>`),
    ].join('');
    const atMax = selectedIds.size >= maxSelected;
    const featureButtons = visibleModules.flatMap(module => (module.features || []).map(feature => {
      const on = selectedIds.has(feature.id);
      const locked = atMax && !on;
      const attr = locked
        ? `aria-disabled="true" title="${esc(t('Selection limit reached; remove a feature before adding another.', '已达到选择上限；请先移除一个特征再添加。'))}"`
        : `data-cohort-feature-toggle="${esc(feature.id)}"`;
      return `
        <button class="chip ${on ? 'solid' : ''} ${locked ? 'disabled' : ''}" type="button" ${attr}>
          ${on ? icon('check', 11) : icon('plus', 11)}
          <span>${esc(feature.label || feature.column || feature.id)}</span>
          <span class="mono" style="opacity:.72;">${esc(module.module)}</span>
        </button>`;
    })).join('');
    const selectedChips = selectedRows.map(feature => `
      <button class="chip solid" type="button" data-cohort-feature-toggle="${esc(feature.id)}" title="${esc(t('Remove feature', '移除特征'))}">
        ${icon('check', 11)} ${esc(feature.label || feature.column || feature.id)}
        <span class="mono" style="opacity:.72;">${esc(feature.module || '')}</span>
      </button>`).join('');
    return `
      <div class="card pad mt-14" data-cohort-feature-picker>
        <div class="row wrap gap-8" style="align-items:center;">
          <span class="pill ok"><span class="dot"></span>${t('Full export feature catalog', '全量导出特征目录')}</span>
          <span style="font-size:12px;color:var(--ink-3);">${fmtInt(selection.module_count || catalog.total_modules)} ${t('modules', '个模块')} · ${fmtInt(selection.available_count || catalog.total_features)} ${t('available comparison features', '个可比较特征')} · ${fmtInt(selection.selected_count || selectedIds.size)} ${t('selected', '个已选择')}</span>
          <span class="grow"></span>
          <button class="btn sm" type="button" data-cohort-feature-default>${icon('sliders', 12)} ${t('Restore default features', '恢复默认特征')}</button>
          <button class="btn sm ghost" type="button" data-cohort-feature-clear>${icon('close', 12)} ${t('Clear added features', '清空已选特征')}</button>
        </div>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:8px;">${t('Default starts with key ICU variables, but every feature present in the loaded modules can be added to the descriptive group table. Values remain aggregate-only; no patient rows are returned.', '默认先选关键 ICU 变量，但已加载模块中的每个特征都可以加入描述性分组表。这里只返回聚合结果，不返回患者行。')}</div>
        <div class="row wrap gap-6" style="margin-top:10px;">${moduleChips}</div>
        ${selectedChips ? `<div class="row wrap gap-6" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--hair);"><span style="font-size:11px;color:var(--ink-4);padding-top:4px;">${t('Selected', '已选择')}</span>${selectedChips}</div>` : ''}
        <div class="row wrap gap-6" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--hair);max-height:180px;overflow:auto;">${featureButtons}</div>
        <div style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Interactive comparison is capped to keep large local exports responsive.', '为保证大型本地导出交互流畅，单次交互比较会限制选中特征数量。')} ${fmtInt(selectedIds.size)} / ${fmtInt(maxSelected)}</div>
      </div>`;
  }

  function cohortSurvivalBody(review) {
    const survival = review.survival_analysis || {};
    const outcomes = survival.outcomes || [];
    const groups = survival.group_options || [];
    const readyOutcomes = outcomes.filter(row => row.status === 'ready');
    const readyGroups = groups.filter(row => row.status === 'ready');
    const selectedOutcome = survival.default_outcome || (readyOutcomes[0] && readyOutcomes[0].id) || cohortSurvivalOutcome;
    cohortSurvivalOutcome = selectedOutcome || cohortSurvivalOutcome;
    const selectedGroup = readyGroups.some(row => row.id === cohortSurvivalGroup)
      ? cohortSurvivalGroup
      : (survival.default_group || (readyGroups[0] && readyGroups[0].id));
    const curve = (survival.curves || []).find(row => row.outcome_id === selectedOutcome && row.group_id === selectedGroup);
    const outcomeCards = cohortSurvivalOutcomeCards(outcomes, selectedOutcome);
    const groupButtons = groups.map(row => {
      const ready = row.status === 'ready';
      const cls = `seg-btn ${selectedGroup === row.id ? 'active' : ''} ${ready ? '' : 'disabled'}`;
      const attr = ready ? `data-cohort-surv-group="${esc(row.id)}"` : `aria-disabled="true" title="${esc(cohortReason(row.reason || 'Unavailable'))}"`;
      const n = (row.groups || []).map(g => fmtInt(g.count)).join(' / ');
      return `<button class="${cls}" ${attr}><span>${esc(cohortText(row.label || row.id))}</span><b>${ready ? n : cohortText('blocked')}</b></button>`;
    }).join('');
    const blockedOutcomes = outcomes.filter(row => row.status !== 'ready');
    if (!curve) {
      return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Kaplan-Meier module')}</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">${cohortText('Outcome overview')}</div>${outcomeCards}</div>
        <div><div class="surv-label">${cohortText('Grouping')}</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${cohortText('Survival analysis blocked')}</div><div class="d">${esc(cohortReason(survival.reason || 'This export does not expose an outcome with both event and time-to-event columns.'))}</div></div></div>
      ${cohortSurvivalSourceHint(survival)}
      ${cohortSurvivalBlockedList(blockedOutcomes)}`;
    }
    const logrank = curve.logrank || {};
    const pValueLabel = logrank.p_value_label || fmtP(logrank.p_value);
    const windowNote = cohortSurvivalWindowNote(curve);
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Kaplan-Meier curves and log-rank')}</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">${cohortText('Outcome overview')}</div>${outcomeCards}</div>
        <div><div class="surv-label">${cohortText('Grouping')}</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      <div class="surv-card mt-14">
        <div class="surv-head">
          <div>
            <div class="eyebrow">${cohortText('Exploratory · unadjusted')}</div>
            <h3>${esc(cohortText(curve.label || 'Kaplan-Meier curve'))}</h3>
            <p>${esc(cohortText(curve.time_label || 'Time-to-event'))} · ${t('event', '事件')} <span class="mono">${esc(curve.event_column || '')}</span> · ${t('time', '时间')} <span class="mono">${esc(curve.time_column || '')}</span></p>
            ${windowNote ? `<p>${esc(windowNote)}</p>` : ''}
          </div>
          <div class="surv-logrank">
            <span>${cohortText('Log-rank')}</span>
            <strong>${logrank.status === 'ready' ? `χ² ${fmtNum(logrank.chi_square, 2)} · p = ${esc(pValueLabel)}` : cohortText('unavailable')}</strong>
            <small>${logrank.status === 'ready' ? cohortText('df 1 · exploratory only') : esc(cohortReason(logrank.reason || 'not enough events'))}</small>
          </div>
        </div>
        ${cohortSurvivalChart(curve)}
        ${cohortRiskTable(curve)}
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Not manuscript-ready by itself')}</div><div class="d">${t('KM/log-rank is computed from bounded cohort aggregates and marked exploratory. Any claim still needs the evidence-bound Agent check and human review.', 'KM/log-rank 由有界队列聚合计算，标记为探索性。任何稿件声明仍需要证据绑定 Agent 检查和人工审阅。')}</div></div></div>
      ${cohortSurvivalBlockedList(blockedOutcomes)}`;
  }

  function cohortSurvivalOutcomeCards(outcomes, selectedOutcome) {
    const rows = outcomes || [];
    if (!rows.length) {
      return `<div class="surv-outcome-grid"><div class="surv-outcome-card muted"><span>${cohortText('No outcome module')}</span><b>${cohortText('not available')}</b></div></div>`;
    }
    return `
      <div class="surv-outcome-grid">
        ${rows.map(row => {
          const summary = row.event_summary || {};
          const hasRate = summary.status === 'available' && summary.event_rate_pct != null;
          const selected = row.id === selectedOutcome;
          const cls = `surv-outcome-card ${selected ? 'active' : ''} ${hasRate ? '' : 'muted'}`;
          const rate = hasRate ? fmtPct(summary.event_rate_pct) : cohortText('not available');
          const events = hasRate
            ? `${fmtInt(summary.event_count)} / ${fmtInt(summary.denominator)} ${cohortText('events')}`
            : cohortReason(summary.reason || row.reason || 'No event column found');
          const meta = cohortSurvivalOutcomeMeta(row);
          return `
            <div class="${cls}">
              <span>${esc(cohortText(row.label || row.id))}</span>
              <strong>${esc(rate)}</strong>
              <b>${esc(events)}</b>
              ${meta ? `<em>${esc(meta)}</em>` : ''}
            </div>`;
        }).join('')}
      </div>`;
  }

  function cohortSurvivalOutcomeMeta(row) {
    if (!row) return '';
    const parts = [];
    const summary = row.event_summary || {};
    if (row.id === 'mort_28d' && row.status === 'ready') {
      parts.push(cohortText('KM curve endpoint'));
    } else if (summary.status === 'available') {
      parts.push(cohortText('Event rate summary'));
    }
    if (summary.basis === 'derived_time_window') {
      parts.push(cohortText(summary.time_window_label || row.window_label || 'time window'));
    } else if (row.window_label && row.id === 'mort_28d') {
      parts.push(cohortText(row.window_label));
    }
    if (summary.basis === 'derived_time_window' || row.derived_from === 'hospital_mortality_time_window') {
      parts.push(cohortText('derived from hospital death + LOS'));
    }
    return parts.join(' · ');
  }

  function cohortSurvivalOutcomeUnavailable(row) {
    const reason = cohortReason(row && row.reason);
    if (reason.includes('ICU') || reason.includes('专用')) {
      return t('unavailable · missing ICU event/time columns', '不可用 · 缺少 ICU 事件/时间列');
    }
    return cohortText('unavailable');
  }

  function cohortSurvivalWindowNote(curve) {
    if (!curve || curve.display_horizon_days == null) return '';
    const days = fmtNum(curve.display_horizon_days, 0);
    const base = t(
      `Displayed on a ${days}-day window; later observations are censored at the window boundary.`,
      `默认显示 ${days} 天窗口；窗口之后的观测在边界处按删失处理。`
    );
    if (curve.derived_from === 'hospital_mortality_time_window') {
      return `${base} ${t('This 28-day endpoint is derived from hospital death plus hospital LOS because dedicated 28-day columns were not present.', '这个 28 天结局由院内死亡 + 住院时长派生，因为导出中没有单独的 28 天结局列。')}`;
    }
    return base;
  }

  function cohortSurvivalSourceHint(survival) {
    const reason = cohortReason((survival && survival.reason) || 'This export does not expose an outcome with both event and time-to-event columns.');
    const review = cohortReview();
    const source = (review && review.source) || {};
    const summary = (review && review.summary) || {};
    return `
      <div class="note info mt-12" data-survival-source-hint>
        <div class="ico">${icon('db', 14)}</div>
        <div class="body">
          <div class="t">${t('Current export is already loaded', '当前导出已加载')}</div>
          <div class="d">${t('Cohort Review is using the active EasyICU export. KM/log-rank will run here when the Outcome module has both an event flag and a time-to-event or censoring-time column; otherwise continue from this same export in an audited local Agent analysis. No re-import is required.', '队列审阅正在使用当前 active 的 EasyICU 导出。只要 Outcome/结局模块同时有事件标志和事件时间或删失时间列，KM/log-rank 就会直接在这里运行；否则也应从同一个导出进入本地 Agent 审计分析，不需要重新导入。')}</div>
          <div class="d mono" style="margin-top:4px;">${esc(reason)}</div>
        </div>
      </div>
      <div class="card pad mt-12" data-survival-current-export>
        <div class="sec-stack" style="margin-bottom:10px;"><div class="lbl">${t('Loaded source', '已加载来源')}</div><h2>${t('KM uses the current export snapshot', 'KM 复用当前导出快照')}</h2></div>
        <div class="setup-row"><span class="k">${cohortText('Source')}</span><span class="vv">${esc(source.label || cohortText('Local export'))}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Path hash')}</span><span class="vv mono">${esc(source.path_hash || '')}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Cohort size')}</span><span class="vv">${fmtInt(summary.cohort_size)} ${t('entities', '个实体')}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Outcome')}</span><span class="vv">${t('Outcome module is registered in this export; blocked rows below explain any missing event/time pair.', '此导出已注册 Outcome/结局模块；下方拦截行会说明缺少的事件/时间组合。')}</span></div>
      </div>`;
  }

  function cohortSurvivalBlockedList(rows) {
    if (!rows || !rows.length) return '';
    return `
      <div class="surv-blocked mt-12">
        ${rows.map(row => `<div class="surv-blocked-row"><span>${esc(cohortText(row.label || row.id))}</span><em>${esc(cohortReason(row.reason || 'Unavailable for this export'))}</em></div>`).join('')}
      </div>`;
  }

  function cohortSurvivalChart(curve) {
    const groups = curve.groups || [];
    const allPoints = groups.flatMap(g => g.points || []);
    const maxTime = Math.max(1, ...allPoints.map(p => Number(p.time) || 0), ...(((curve.number_at_risk || {}).times || []).map(Number)));
    const w = 760, h = 300, l = 56, r = 22, tpad = 22, b = 46;
    const plotW = w - l - r, plotH = h - tpad - b;
    const x = value => l + (Number(value || 0) / maxTime) * plotW;
    const y = value => tpad + (1 - (Number(value || 0) / 100)) * plotH;
    const colors = ['#0f766e', '#2563eb', '#8b5cf6', '#b45309'];
    const xticks = [0, maxTime / 4, maxTime / 2, maxTime * 0.75, maxTime].map(v => Math.round(v * 10) / 10);
    const yticks = [0, 25, 50, 75, 100];
    const stepPath = (points) => {
      const pts = (points && points.length) ? points : [{ time: 0, survival: 100 }];
      let d = `M ${x(pts[0].time)} ${y(pts[0].survival)}`;
      for (let i = 1; i < pts.length; i += 1) {
        d += ` H ${x(pts[i].time)} V ${y(pts[i].survival)}`;
      }
      return d;
    };
    return `
      <div class="km-chart-wrap">
        <svg class="km-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(curve.label || 'Kaplan-Meier curve')}">
          <rect x="${l}" y="${tpad}" width="${plotW}" height="${plotH}" rx="4" fill="#fbfbf8" stroke="#e5e2da"></rect>
          ${yticks.map(v => `<line x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}" class="km-grid"></line><text x="${l - 10}" y="${y(v) + 4}" text-anchor="end" class="km-axis">${v}%</text>`).join('')}
          ${xticks.map(v => `<line x1="${x(v)}" x2="${x(v)}" y1="${tpad}" y2="${h - b}" class="km-grid faint"></line><text x="${x(v)}" y="${h - 18}" text-anchor="middle" class="km-axis">${fmtNum(v, 1)}</text>`).join('')}
          <text x="${l + plotW / 2}" y="${h - 2}" text-anchor="middle" class="km-axis-title">${cohortText('Days')}</text>
          <text x="14" y="${tpad + plotH / 2}" transform="rotate(-90 14 ${tpad + plotH / 2})" text-anchor="middle" class="km-axis-title">${cohortText('Survival probability')}</text>
          ${groups.map((g, i) => `<path d="${stepPath(g.points || [])}" class="km-line" style="stroke:${colors[i % colors.length]};"></path>`).join('')}
        </svg>
        <div class="km-legend">
          ${groups.map((g, i) => `<span><i style="background:${colors[i % colors.length]};"></i>${esc(cohortText(g.label))} · n ${fmtInt(g.n)} · ${cohortText('events')} ${fmtInt(g.events)}</span>`).join('')}
        </div>
      </div>`;
  }

  function cohortRiskTable(curve) {
    const risk = curve.number_at_risk || {};
    const times = risk.times || [];
    const rows = risk.rows || [];
    if (!times.length || !rows.length) return '';
    return `
      <div class="risk-table-wrap">
        <div class="surv-label">${cohortText('Number at risk')}</div>
        <table class="risk-table">
          <thead><tr><th>${cohortText('Group')}</th>${times.map(tick => `<th>${fmtNum(tick, 1)}d</th>`).join('')}</tr></thead>
          <tbody>
            ${rows.map(row => `<tr><td>${esc(cohortText(row.label))}</td>${(row.values || []).map(value => `<td>${fmtInt(value)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }

  function cohortSurvivalDemoBody() {
    const curve = {
      label: 'Demo hospital mortality by Sepsis vs Non-sepsis',
      time_label: 'Demo follow-up days',
      event_column: 'demo_hospital_death',
      time_column: 'demo_followup_days',
      logrank: { status: 'ready', chi_square: 5.42, p_value: 0.0198 },
      groups: [
        {
          label: 'Non-sepsis',
          n: 160,
          events: 18,
          points: [
            { time: 0, survival: 100 },
            { time: 1, survival: 98.8 },
            { time: 3, survival: 96.9 },
            { time: 7, survival: 93.8 },
            { time: 14, survival: 90.7 },
            { time: 28, survival: 88.8 },
          ],
        },
        {
          label: 'Sepsis',
          n: 140,
          events: 34,
          points: [
            { time: 0, survival: 100 },
            { time: 1, survival: 97.9 },
            { time: 3, survival: 93.6 },
            { time: 7, survival: 88.5 },
            { time: 14, survival: 81.8 },
            { time: 28, survival: 75.7 },
          ],
        },
      ],
      number_at_risk: {
        times: [0, 1, 3, 7, 14, 28],
        rows: [
          { label: 'Non-sepsis', values: [160, 158, 154, 148, 141, 132] },
          { label: 'Sepsis', values: [140, 137, 129, 119, 105, 92] },
        ],
      },
    };
    const logrank = curve.logrank;
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Demo simulated KM preview')}</h2></div>
      <div class="note warn mt-12" data-demo-survival-simulated><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Seeded demo only')}</div><div class="d">${t('This Kaplan-Meier curve is a fixed simulated preview for the demo workspace. It exercises the chart, log-rank, and number-at-risk UI only; it is not derived from a local export and must not be used for manuscript claims.', '这条 Kaplan-Meier 曲线是演示工作区的固定模拟预览，只用于展示图表、log-rank 和风险人数表交互；它不是来自本地导出，不能用于稿件结论。')}</div></div></div>
      <div class="surv-card mt-14">
        <div class="surv-head">
          <div>
            <div class="eyebrow">${cohortText('Seeded demo only')}</div>
            <h3>${esc(cohortText(curve.label))}</h3>
            <p>${esc(cohortText(curve.time_label))} · ${t('event', '事件')} <span class="mono">${esc(curve.event_column)}</span> · ${t('time', '时间')} <span class="mono">${esc(curve.time_column)}</span></p>
          </div>
          <div class="surv-logrank">
            <span>${cohortText('Log-rank')}</span>
            <strong>χ² ${fmtNum(logrank.chi_square, 2)} · p = ${fmtP(logrank.p_value)}</strong>
            <small>${cohortText('Seeded demo only')}</small>
          </div>
        </div>
        ${cohortSurvivalChart(curve)}
        ${cohortRiskTable(curve)}
      </div>`;
  }

  function cohortDemoPanelNote(kind) {
    const detail = kind === 'sofa'
      ? t('This SOFA-2 reclassification panel uses fixed seeded values to preview the chart and table layout. It is not computed from a local export and must not be used for manuscript claims.', '这个 SOFA-2 重分层面板使用固定 seeded 数值来预览图表和表格布局；它不是从本地导出计算的，不能用于稿件结论。')
      : t('This coverage audit uses fixed EasyICU catalog-shaped demo values to preview the module coverage workflow. It is not computed from a local export and must not be used for manuscript claims.', '这个覆盖审计使用固定的 EasyICU catalog-shaped 演示值来预览模块覆盖工作流；它不是从本地导出计算的，不能用于稿件结论。');
    return `<div class="note warn mt-12" data-demo-cohort-panel="${esc(kind)}"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Seeded demo only')}</div><div class="d">${detail}</div></div></div>`;
  }

  function cohortDemoCoverageReview() {
    const modules = demoCatalogModules();
    const coverage = modules.map((module, index) => {
      const featureCount = (module.features || []).length;
      const penalty = (index % 5) * 4 + Math.max(0, featureCount - 12) * 0.25;
      const coveragePct = Math.max(58, Math.min(100, 98 - penalty));
      const rows = demoRowsForModule(module.module, featureCount, coveragePct);
      const qualityStatus = coveragePct >= 85 ? 'ok' : (coveragePct >= 70 ? 'warn' : 'neutral');
      return {
        module: module.label,
        module_key: module.module,
        rows,
        column_count: featureCount,
        covered_entities: Math.max(1, Math.round(10 * coveragePct / 100)),
        coverage_pct: Number(coveragePct.toFixed(1)),
        quality_status: qualityStatus,
      };
    });
    const coverageValues = coverage.map(row => row.coverage_pct).sort((a, b) => a - b);
    const median = coverageValues.length ? coverageValues[Math.floor(coverageValues.length / 2)] : null;
    return {
      demo: true,
      coverage,
      quality: {
        modules_ok: coverage.filter(row => row.quality_status === 'ok').length,
        watchlist_count: coverage.filter(row => row.quality_status === 'warn').length,
        median_coverage_pct: median,
        modules_neutral: coverage.filter(row => row.quality_status === 'neutral').length,
        modules_unknown: 0,
      },
    };
  }

  function cohortDemoSofaExactMatrix(pairs) {
    const labels = Array.from({ length: 25 }, (_, score) => String(score));
    const total = pairs.length || 1;
    return labels.map(sourceLabel => {
      const sourceScore = Number(sourceLabel);
      const cells = labels.map(targetLabel => {
        const targetScore = Number(targetLabel);
        const count = pairs.filter(([sofa1, sofa2]) => sofa1 === sourceScore && sofa2 === targetScore).length;
        return { label: targetLabel, count, pct: Number((count / total * 100).toFixed(1)) };
      });
      return { label: sourceLabel, count: cells.reduce((acc, cell) => acc + cell.count, 0), cells };
    });
  }

  function cohortDemoSofaReview() {
    const demoPairs = [[2, 2], [4, 7], [8, 5], [7, 7], [6, 9], [10, 13], [12, 11], [3, 3], [5, 5], [9, 9]];
    return {
      demo: true,
      summary: {
        cohort_size: 10,
        sofa2: {
          count: 10,
          median: 7,
          mean: 7.4,
          min: 1,
          max: 16,
          bins: [
            { label: '0-5', count: 4, pct: 40.0 },
            { label: '6-9', count: 3, pct: 30.0 },
            { label: '10-13', count: 2, pct: 20.0 },
            { label: '14+', count: 1, pct: 10.0 },
          ],
        },
      },
      sofa_reclassification: {
        status: 'ready',
        paired_count: 10,
        coverage_pct: 100.0,
        direction_counts: {
          up: { count: 3, pct: 30.0 },
          down: { count: 2, pct: 20.0 },
          same: { count: 5, pct: 50.0 },
        },
        delta_summary: { median: 1.0 },
        severity_bins: ['0-5', '6-9', '10-13', '14+'],
        transition_matrix: [
          { label: '0-5', cells: [{ count: 3, pct: 30.0 }, { count: 1, pct: 10.0 }, { count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }] },
          { label: '6-9', cells: [{ count: 1, pct: 10.0 }, { count: 2, pct: 20.0 }, { count: 1, pct: 10.0 }, { count: 0, pct: 0.0 }] },
          { label: '10-13', cells: [{ count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }, { count: 1, pct: 10.0 }, { count: 1, pct: 10.0 }] },
          { label: '14+', cells: [{ count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }] },
        ],
        exact_score_bins: Array.from({ length: 25 }, (_, score) => String(score)),
        exact_score_matrix: cohortDemoSofaExactMatrix(demoPairs),
        score_scale: { min: 0, max: 24, unit: 'SOFA points', aggregation: 'nearest_integer_clamped_0_24' },
      },
    };
  }

  function cohortCoverageBody(review, opts = {}) {
    const rows = review.coverage || [];
    const q = review.quality || {};
    const metadataOnlyCount = rows.filter(row => row.coverage_basis === 'metadata_row_count_only').length;
    const rateRows = rows.filter(row => row.metric_kind === 'event_rate' || row.metric_kind === 'exposure_rate').length;
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('Coverage audit')}</div><h2>${cohortText(opts.demo ? 'Demo module coverage and quality' : 'Real module coverage and quality')}</h2></div>
      ${opts.demo ? cohortDemoPanelNote('coverage') : ''}
      ${metadataOnlyCount ? `<div class="note info mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('Large export coverage optimized', '大导出覆盖率已优化')}</div><div class="d">${t('Some non-Parquet or very large modules are shown with manifest-confirmed row counts first to avoid a slow full stay-id scan. They are loaded modules, not missing modules.', '部分非 Parquet 或超大模块会先显示清单确认的行数，避免缓慢的全量 stay_id 扫描。它们是已加载模块，不是缺失模块。')}</div></div></div>` : ''}
      ${rateRows ? `<div class="note info mt-12"><div class="ico">${icon('activity', 14)}</div><div class="body"><div class="t">${cohortText('Presence-rate modules')}</div><div class="d">${cohortText('Event/exposure rows show cohort incidence or exposure prevalence, not missingness coverage; they are excluded from the coverage watchlist.')}</div></div></div>` : ''}
      <div class="audit-cards">
        ${[
          ['Modules OK', fmtInt(q.modules_ok)],
          ['Watchlist', fmtInt(q.watchlist_count)],
          ['Median coverage', fmtPct(q.median_coverage_pct)],
          ['Presence-rate modules', fmtInt(q.modules_neutral)],
          ['Unknown coverage', fmtInt(q.modules_unknown)],
        ].map(([k, v]) => `<div class="audit-card"><div class="ac-k">${cohortText(k)}</div><div class="ac-v mono">${v}</div></div>`).join('')}
      </div>
      <div class="table-wrap table-scroll mt-16">
        <table class="eu-table">
          <thead><tr><th>${cohortText('Module')}</th><th class="num">${cohortText('Records')}</th><th class="num">${cohortText('Fields')}</th><th class="num">${cohortText('Entities')}</th><th class="num">${cohortText('Coverage / rate')}</th><th>${cohortText('Interpretation')}</th></tr></thead>
          <tbody>
            ${rows.map(row => `<tr>
              <td class="key">${esc(row.module)}</td>
              <td class="num">${fmtInt(row.rows)}</td>
              <td class="num">${fmtInt(row.column_count)}</td>
              <td class="num">${row.coverage_basis === 'metadata_row_count_only' ? t('manifest confirmed', '清单确认') : fmtInt(row.covered_entities)}</td>
              <td class="num">${cohortCoverageMetricValue(row)}</td>
              <td><span class="pill ${cohortQualityStatusClass(row)}" style="height:20px;">${cohortQualityStatusLabel(row)}</span></td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Fail-closed scope')}</div><div class="d">${t('Coverage is aggregate-only. Row-level filtering, subgroup missingness, and eligibility waterfalls remain blocked until a bounded cohort-builder backend exists.', '覆盖率是仅聚合结果。行级筛选、亚组缺失率和纳排瀑布图会在有界队列构建后端就绪前保持拦截。')}</div></div></div>`;
  }

  function cohortSofaCellFill(tone, intensity) {
    const a = Math.max(0.08, Math.min(0.72, 0.12 + intensity * 0.52));
    if (tone === 'up') return `rgba(190, 76, 76, ${a.toFixed(3)})`;
    if (tone === 'down') return `rgba(42, 111, 178, ${a.toFixed(3)})`;
    return `rgba(34, 137, 122, ${a.toFixed(3)})`;
  }

  const SOFA_MATRIX_GRANULARITIES = {
    coarse: {
      label: 'Coarse',
      detail: '4 bands',
      bins: [
        { label: '0-5', min: 0, max: 5 },
        { label: '6-8', min: 6, max: 8 },
        { label: '9-11', min: 9, max: 11 },
        { label: '12+', min: 12, max: 24 },
      ],
    },
    medium: {
      label: 'Medium',
      detail: '6 bands',
      bins: [
        { label: '0-3', min: 0, max: 3 },
        { label: '4-7', min: 4, max: 7 },
        { label: '8-11', min: 8, max: 11 },
        { label: '12-15', min: 12, max: 15 },
        { label: '16-19', min: 16, max: 19 },
        { label: '20-24', min: 20, max: 24 },
      ],
    },
    fine: {
      label: 'Fine',
      detail: '12 bands',
      bins: Array.from({ length: 12 }, (_, index) => {
        const min = index * 2;
        const max = index === 11 ? 24 : min + 1;
        return { label: min === max ? String(min) : `${min}-${max}`, min, max };
      }),
    },
    exact: {
      label: 'Exact',
      detail: '25 scores',
      bins: Array.from({ length: 25 }, (_, score) => ({ label: String(score), min: score, max: score })),
    },
  };

  function cohortSofaGranularityButtons(hasExactMatrix) {
    if (!hasExactMatrix) return '';
    const order = ['coarse', 'medium', 'fine', 'exact'];
    return `
      <div class="sofa-matrix-control">
        <span>${cohortText('Granularity')}</span>
        <div class="sofa-matrix-toggle" role="group" aria-label="${cohortText('Granularity')}">
          ${order.map(key => {
            const opt = SOFA_MATRIX_GRANULARITIES[key];
            return `<button class="${cohortSofaMatrixGranularity === key ? 'active' : ''}" data-cohort-sofa-granularity="${key}" type="button">${cohortText(opt.label)} <small>${cohortText(opt.detail)}</small></button>`;
          }).join('')}
        </div>
      </div>`;
  }

  function cohortSofaExactMatrixMap(reclass) {
    const exact = reclass.exact_score_matrix || [];
    if (!Array.isArray(exact) || !exact.length) return null;
    const map = new Map();
    exact.forEach(row => {
      const source = Number(row && row.label);
      if (!Number.isFinite(source)) return;
      (row.cells || []).forEach(cell => {
        const target = Number(cell && cell.label);
        if (!Number.isFinite(target)) return;
        map.set(`${source}|${target}`, Number(cell.count) || 0);
      });
    });
    return map.size ? map : null;
  }

  function cohortSofaBinnedMatrix(reclass) {
    const exactMap = cohortSofaExactMatrixMap(reclass);
    if (!exactMap) {
      return {
        bins: reclass.severity_bins || [],
        matrix: reclass.transition_matrix || [],
        exact: false,
      };
    }
    const granularity = SOFA_MATRIX_GRANULARITIES[cohortSofaMatrixGranularity] ? cohortSofaMatrixGranularity : 'medium';
    const bins = SOFA_MATRIX_GRANULARITIES[granularity].bins;
    const paired = Number(reclass.paired_count) || Array.from(exactMap.values()).reduce((acc, value) => acc + value, 0) || 0;
    const matrix = bins.map(sourceBin => {
      const cells = bins.map(targetBin => {
        let count = 0;
        for (let source = sourceBin.min; source <= sourceBin.max; source += 1) {
          for (let target = targetBin.min; target <= targetBin.max; target += 1) {
            count += exactMap.get(`${source}|${target}`) || 0;
          }
        }
        return {
          label: targetBin.label,
          count,
          pct: paired ? Number((count / paired * 100).toFixed(1)) : 0,
        };
      });
      return {
        label: sourceBin.label,
        count: cells.reduce((acc, cell) => acc + cell.count, 0),
        cells,
      };
    });
    return {
      bins: bins.map(row => row.label),
      matrix,
      exact: true,
    };
  }

  function cohortSofaHeatmap(reclass) {
    const binned = cohortSofaBinnedMatrix(reclass);
    const bins = binned.bins || [];
    const matrix = binned.matrix || [];
    if (!bins.length || !matrix.length) {
      return `<div class="muted" style="font-size:11px;">${t('No paired SOFA-1/SOFA-2 bins in this export.', '此导出没有配对 SOFA-1/SOFA-2 分箱。')}</div>`;
    }
    const mode = cohortSofaMatrixMode === 'count' ? 'count' : 'pct';
    const hasExactMatrix = !!binned.exact;
    const cellMin = bins.length > 12 ? 54 : (bins.length > 6 ? 72 : 92);
    const matrixMinWidth = Math.max(620, 112 + bins.length * cellMin);
    const maxCount = Math.max(1, ...matrix.flatMap(row => (row.cells || []).map(cell => Number(cell.count) || 0)));
    const maxPct = Math.max(1, ...matrix.flatMap(row => (row.cells || []).map(cell => Number(cell.pct) || 0)));
    const corner = `${t('SOFA-1', 'SOFA-1')} \\ ${t('SOFA-2', 'SOFA-2')}`;
    const headers = bins.map(label => `<div class="sofa-heat-head col">${esc(label)}</div>`).join('');
    const rows = matrix.map((row, rowIndex) => {
      const cells = (row.cells || []).map((cell, colIndex) => {
        const count = Number(cell.count) || 0;
        const pct = Number(cell.pct) || 0;
        const value = mode === 'count' ? fmtInt(count) : fmtPct(pct);
        const intensity = mode === 'count' ? count / maxCount : pct / maxPct;
        const tone = colIndex > rowIndex ? 'up' : colIndex < rowIndex ? 'down' : 'same';
        const label = `${row.label} to ${bins[colIndex] || ''}: ${fmtInt(count)} · ${fmtPct(pct)}`;
        return `<div class="sofa-heat-cell ${tone}" style="--heat-bg:${cohortSofaCellFill(tone, intensity)};" title="${esc(label)}" aria-label="${esc(label)}">
          <span class="sofa-heat-value mono">${value}</span>
        </div>`;
      }).join('');
      return `<div class="sofa-heat-head row">${esc(row.label)}</div>${cells}`;
    }).join('');
    return `
      <div class="sofa-matrix-head mt-12">
        <div>
          <div class="rc-sec-t">${cohortText('Worst-ICU severity transition matrix')}</div>
          <p>${hasExactMatrix ? cohortText('Rows are SOFA-1 score bands; columns are SOFA-2 score bands. Use the granularity control to move from clinical bands to exact 0-24 scores.') : cohortText('Rows are SOFA-1 severity bands; columns are SOFA-2 bands. Color intensity follows the selected value.')}</p>
        </div>
        <div class="sofa-matrix-controls">
          ${cohortSofaGranularityButtons(hasExactMatrix)}
          <div class="sofa-matrix-control">
            <span>${cohortText('Matrix value')}</span>
            <div class="sofa-matrix-toggle" role="group" aria-label="${cohortText('Matrix value')}">
              <button class="${mode === 'pct' ? 'active' : ''}" data-cohort-sofa-matrix-mode="pct" type="button">${cohortText('Percent')}</button>
              <button class="${mode === 'count' ? 'active' : ''}" data-cohort-sofa-matrix-mode="count" type="button">N</button>
            </div>
          </div>
        </div>
      </div>
      <div class="sofa-heat-scroll">
        <div class="sofa-heatmap" style="--sofa-data-cols:${bins.length}; --sofa-cell-min:${cellMin}px; --sofa-min-width:${matrixMinWidth}px;">
          <div class="sofa-heat-head corner">${esc(corner)}</div>
          ${headers}
          ${rows}
        </div>
      </div>
      <div class="sofa-heat-legend">
        <span><i class="same"></i>${cohortText('Same severity band')}</span>
        <span><i class="up"></i>${cohortText('SOFA-2 higher band')}</span>
        <span><i class="down"></i>${cohortText('SOFA-2 lower band')}</span>
      </div>`;
  }

  function cohortSofaBody(review, opts = {}) {
    const s = review.summary || {};
    const sofa = s.sofa2 || {};
    const reclass = review.sofa_reclassification || {};
    const bins = sofa.bins || [];
    const maxBin = Math.max(1, ...bins.map(b => b.count || 0));
    const movement = reclass.direction_counts || {};
    const delta = reclass.delta_summary || {};
    const movementCards = reclass.status === 'ready' ? [
      [fmtInt(reclass.paired_count), 'Paired entities', `${fmtPct(reclass.coverage_pct)} ${t('of cohort', '的队列覆盖')}`, 'n'],
      [fmtInt(movement.up && movement.up.count), 'SOFA-2 higher', fmtPct(movement.up && movement.up.pct), 'up'],
      [fmtInt(movement.down && movement.down.count), 'SOFA-2 lower', fmtPct(movement.down && movement.down.pct), 'down'],
      [fmtNum(delta.median, 1), 'Median delta', 'SOFA-2 minus SOFA-1', 'delta'],
    ] : [];
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('SOFA reclassification')}</div><h2>${cohortText(opts.demo ? 'Demo SOFA-2 aggregate preview' : 'SOFA-2 aggregate review')}</h2></div>
      ${opts.demo ? cohortDemoPanelNote('sofa') : ''}
      <div class="rc-kpis">
        ${[
          [fmtNum(sofa.median, 1), 'Median SOFA-2', `${fmtInt(sofa.count)} ${t('entities with score', '个实体有评分')}`, 'delta'],
          [fmtNum(sofa.mean, 1), 'Mean SOFA-2', 'registered export aggregate', 'n'],
          [fmtNum(sofa.min, 1), 'Min', 'bounded column read', 'down'],
          [fmtNum(sofa.max, 1), 'Max', 'bounded column read', 'up'],
        ].map(([v, label, hint, kind]) => `
          <div class="rc-kpi rc-${kind}">
            <div class="rk-top"><span class="rk-ico">${icon(kind === 'up' ? 'arrow' : kind === 'down' ? 'arrow' : 'layers', 13)}</span><span class="rk-label">${cohortText(label)}</span></div>
            <div class="rk-val mono">${v}</div>
            <div class="rk-hint">${cohortText(hint)}</div>
          </div>`).join('')}
      </div>
      <div class="card pad mt-16">
        <div class="rc-sec-t">${cohortText('SOFA-2 severity bins')}</div>
        <div class="rc-groups">
          ${bins.map(bin => `
            <div class="rc-grow">
              <div class="rg-head"><span class="rg-name">${esc(bin.label)}</span><span class="rg-pct mono">${fmtPct(bin.pct)}</span></div>
              <div class="rg-bar"><div class="rg-fill same" style="width:${((bin.count || 0) / maxBin * 100).toFixed(0)}%;"></div></div>
              <div class="rg-meta"><span>${fmtInt(bin.count)} ${t('entities', '个实体')}</span></div>
            </div>`).join('')}
        </div>
      </div>
      ${reclass.status === 'ready' ? `
        <div class="card pad mt-16">
          <div class="rc-sec-t">${cohortText('SOFA-1 to SOFA-2 movement')}</div>
          <div class="rc-kpis compact">
            ${movementCards.map(([v, label, hint, kind]) => `
              <div class="rc-kpi rc-${kind}">
                <div class="rk-top"><span class="rk-ico">${icon(kind === 'up' ? 'arrow' : kind === 'down' ? 'arrow' : 'layers', 13)}</span><span class="rk-label">${cohortText(label)}</span></div>
                <div class="rk-val mono">${v}</div>
                <div class="rk-hint">${cohortText(hint)}</div>
              </div>`).join('')}
          </div>
          ${cohortSofaHeatmap(reclass)}
        </div>
        <div class="note ok mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Paired aggregate ready')}</div><div class="d">${t('Worst-ICU SOFA-1/SOFA-2 movement is computed from bounded per-entity score aggregates only. No paired patient rows or inferential statistics are returned.', 'ICU 最严重 SOFA-1/SOFA-2 变化只由有界实体级评分聚合计算；不返回配对患者行或推断统计。')}</div></div></div>
      ` : `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${cohortText('Paired reclassification blocked')}</div><div class="d">${esc(cohortReason(reclass.reason || 'Paired SOFA-1/SOFA-2 reclassification is not available for this export.'))}</div></div></div>`}`;
  }

  function cohortDistBars(bins) {
    const arr = (bins || []).filter(Boolean);
    if (!arr.length) return `<div class="muted" style="font-size:11px;">${t('No binned values in this export.', '此导出无可分箱数值。')}</div>`;
    const maxN = Math.max(1, ...arr.map(b => b.count || 0));
    return arr.map(b => `<div class="qrow"><span>${esc(b.label)}</span><div class="qbar"><span style="width:${((b.count || 0) / maxN * 100).toFixed(0)}%"></span></div><span class="qv">${fmtInt(b.count)}</span></div>`).join('');
  }

  function cohortCompositionBars(rows) {
    return (rows || []).map(([label, pct]) => `<div class="qrow"><span>${cohortText(label)}</span><div class="qbar"><span style="width:${pct == null ? 0 : Math.max(0, Math.min(100, pct)).toFixed(0)}%"></span></div><span class="qv">${fmtPct(pct)}</span></div>`).join('');
  }

  function cohortProfileLabel(row) {
    const item = row || {};
    if (window.EU_LANG === 'zh') return item.label_zh || item.zh || item.label || item.id || '';
    return item.label || item.label_en || item.label_zh || item.id || '';
  }

  function cohortProfileReason(row) {
    const item = row || {};
    if (window.EU_LANG === 'zh') return item.reason_zh || item.text_zh || item.reason || item.text || '';
    return item.reason || item.text || item.reason_zh || item.text_zh || '';
  }

  function cohortProfileStatusText(status) {
    const key = String(status || 'unknown');
    const labels = {
      ready: [t('ready', '已就绪')],
      partial: [t('partial', '部分可用')],
      unavailable: [t('not in export', '当前导出未提供')],
      schema_only: [t('schema only', '仅结构可见')],
      ok: [t('ok', '正常')],
      warn: [t('watch', '关注')],
      bad: [t('low', '偏低')],
      unknown: [t('unknown', '未知')],
    };
    return (labels[key] && labels[key][0]) || key;
  }

  function cohortProfileUnit(item) {
    const row = item || {};
    return window.EU_LANG === 'zh' ? (row.unit_zh || row.unit || '') : (row.unit || row.unit_zh || '');
  }

  function cohortProfileValue(item) {
    const row = item || {};
    if (row.kind === 'numeric') {
      if (row.value == null) return '—';
      const unit = cohortProfileUnit(row);
      return `${fmtNum(row.value, 1)}${unit ? ` ${esc(unit)}` : ''}`;
    }
    if (row.kind === 'proportion' || row.kind === 'event_rate' || row.kind === 'module_coverage') {
      return row.pct == null ? '—' : fmtPct(row.pct);
    }
    if (row.kind === 'count') {
      return fmtInt(row.count);
    }
    if (row.kind === 'category') {
      const first = (row.bins || [])[0];
      return first ? `${esc(first.label)} · ${fmtPct(first.pct)}` : '—';
    }
    return row.value == null ? '—' : esc(row.value);
  }

  function cohortProfileDetail(item) {
    const row = item || {};
    if (row.status === 'unavailable') return cohortProfileReason(row) || t('Not present in this export.', '当前导出未提供。');
    if (row.kind === 'numeric') {
      const unit = cohortProfileUnit(row);
      return `${t('range', '范围')} ${fmtNum(row.min, 1)}-${fmtNum(row.max, 1)}${unit ? ` ${esc(unit)}` : ''} · n=${fmtInt(row.count)}`;
    }
    if (row.kind === 'proportion' || row.kind === 'event_rate' || row.kind === 'module_coverage') {
      const base = row.count == null ? t('entity denominator unavailable', '实体分母不可用') : `${fmtInt(row.count)} / ${fmtInt(row.denominator)}`;
      const modules = (row.modules || []).length ? ` · ${esc((row.modules || []).join(', '))}` : '';
      const records = row.rows ? ` · ${fmtInt(row.rows)} ${t('records', '记录')}` : '';
      return `${base}${modules}${records}`;
    }
    if (row.kind === 'count') {
      return row.denominator ? `${fmtInt(row.count)} / ${fmtInt(row.denominator)}` : '';
    }
    if (row.kind === 'category') {
      return (row.bins || []).slice(0, 3).map(bin => `${esc(bin.label)} ${fmtPct(bin.pct)}`).join(' · ') || t('No categorical column available.', '没有可用分类列。');
    }
    return cohortProfileReason(row);
  }

  function cohortProfileItem(item) {
    const row = item || {};
    const status = String(row.status || 'unknown').replace(/[^a-z0-9_-]/gi, '');
    const pct = typeof row.pct === 'number' ? Math.max(0, Math.min(100, row.pct)) : null;
    const bar = pct == null ? '' : `<div class="cprof-bar"><span style="width:${pct.toFixed(1)}%"></span></div>`;
    return `
      <div class="cprof-item ${status}">
        <div class="cprof-k">${esc(cohortProfileLabel(row))}</div>
        <div class="cprof-v">${cohortProfileValue(row)}</div>
        ${bar}
        <div class="cprof-d">${esc(cohortProfileDetail(row))}</div>
      </div>`;
  }

  function cohortClinicalProfile(profile) {
    const domains = (profile && profile.domains) || [];
    if (!domains.length) return '';
    return `
      <div class="cprof-grid">
        ${domains.map(domain => `
          <section class="cprof-domain ${esc(String(domain.status || 'unknown'))}">
            <div class="cprof-head">
              <div>
                <div class="eyebrow">${esc(cohortProfileLabel(domain))}</div>
                <h3>${esc(cohortProfileLabel(domain))}</h3>
              </div>
              <span class="pill ${domain.status === 'ready' ? 'ok' : 'dashed'}">${esc(cohortProfileStatusText(domain.status || 'unknown'))}</span>
            </div>
            <div class="cprof-items">${(domain.items || []).map(cohortProfileItem).join('')}</div>
          </section>
        `).join('')}
      </div>
      ${(profile.notes || []).length ? `<div class="note info mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body">${(profile.notes || []).map(note => `<div class="t">${esc(cohortProfileLabel(note))}</div><div class="d">${esc(cohortProfileReason(note))}</div>`).join('')}</div></div>` : ''}`;
  }

  function demoNumericProfile(id, label, labelZh, value, unit, unitZh, min, max, count = 10) {
    return { id, label, label_zh: labelZh, kind: 'numeric', status: 'ready', value, unit, unit_zh: unitZh, min, max, count };
  }

  function demoPctProfile(id, label, labelZh, pct, count, denominator = 10, kind = 'proportion') {
    return { id, label, label_zh: labelZh, kind, status: 'ready', pct, count, denominator };
  }

  function demoCohortClinicalProfile() {
    return {
      status: 'seeded_demo_clinical_shape',
      payload_scope: 'demo_cohort_aggregate_no_patient_rows',
      domains: [
        {
          id: 'demo_demographics',
          label: 'Demographics',
          label_zh: '人口统计',
          status: 'ready',
          items: [
            demoNumericProfile('age', 'Median age', '年龄中位数', 63, 'years', '岁', 28, 91),
            demoPctProfile('female', 'Female', '女性', 44, 4),
            {
              id: 'admission',
              label: 'Admission type',
              label_zh: '入院类型',
              kind: 'category',
              status: 'ready',
              count: 10,
              distinct: 3,
              bins: [
                { label: t('Emergency', '急诊'), count: 5, pct: 50 },
                { label: t('Transfer', '转入'), count: 3, pct: 30 },
                { label: t('Elective', '择期'), count: 2, pct: 20 },
              ],
            },
          ],
        },
        {
          id: 'demo_severity_outcomes',
          label: 'Severity and outcomes',
          label_zh: '严重程度与结局',
          status: 'ready',
          items: [
            demoNumericProfile('sofa2', 'Worst SOFA-2', '最严重 SOFA-2', 6, 'points', '分', 1, 18),
            demoPctProfile('sepsis3', 'Sepsis-3 incidence', 'Sepsis-3 发生率', 60, 6, 10, 'event_rate'),
            demoPctProfile('hospital_mortality', 'Hospital mortality', '院内死亡率', 20, 2, 10, 'event_rate'),
            demoNumericProfile('icu_los', 'ICU length of stay', 'ICU 住院时长', 5.6, 'days', '天', 1.1, 21.4),
          ],
        },
        {
          id: 'demo_treatments',
          label: 'Treatments and organ support',
          label_zh: '治疗暴露与器官支持',
          status: 'ready',
          items: [
            demoPctProfile('mechanical_ventilation', 'Mechanical ventilation', '机械通气', 50, 5, 10, 'event_rate'),
            demoPctProfile('vasopressors', 'Vasopressor exposure', '血管活性药物暴露', 40, 4, 10, 'event_rate'),
            demoPctProfile('rrt', 'Renal replacement therapy', '肾脏替代治疗', 10, 1, 10, 'event_rate'),
            demoPctProfile('antibiotics', 'Antibiotic exposure', '抗感染治疗', 70, 7, 10, 'event_rate'),
          ],
        },
        {
          id: 'demo_diagnoses',
          label: 'Diagnoses and comorbidities',
          label_zh: '诊断与共病',
          status: 'ready',
          items: [
            demoPctProfile('aki', 'AKI / renal dysfunction', 'AKI / 肾功能异常', 30, 3, 10, 'event_rate'),
            demoPctProfile('respiratory_failure', 'Respiratory failure', '呼吸衰竭', 40, 4, 10, 'event_rate'),
            demoPctProfile('shock', 'Shock phenotype', '休克表型', 30, 3, 10, 'event_rate'),
            demoPctProfile('infection', 'Suspected infection', '疑似感染', 70, 7, 10, 'event_rate'),
          ],
        },
        {
          id: 'demo_vitals_labs',
          label: 'Vitals and laboratory profile',
          label_zh: '生命体征与实验室',
          status: 'ready',
          items: [
            demoNumericProfile('map', 'Mean arterial pressure', '平均动脉压', 76, 'mmHg', 'mmHg', 45, 126),
            demoNumericProfile('lactate', 'Lactate', '乳酸', 2.4, 'mmol/L', 'mmol/L', 0.8, 8.9),
            demoNumericProfile('creatinine', 'Creatinine', '肌酐', 1.3, 'mg/dL', 'mg/dL', 0.5, 4.8),
            demoNumericProfile('platelets', 'Platelets', '血小板', 168, '10^9/L', '10^9/L', 38, 420),
          ],
        },
        {
          id: 'demo_completeness',
          label: 'Data completeness',
          label_zh: '数据覆盖',
          status: 'ready',
          items: [
            demoPctProfile('demographics_module', 'Demographics module', '人口统计模块', 100, 10, 10, 'module_coverage'),
            demoPctProfile('vitals_module', 'Vital signs module', '生命体征模块', 100, 10, 10, 'module_coverage'),
            demoPctProfile('labs_module', 'Laboratory modules', '实验室模块', 90, 9, 10, 'module_coverage'),
            demoPctProfile('outcome_module', 'Outcome module', '结局模块', 100, 10, 10, 'module_coverage'),
          ],
        },
      ],
      notes: [
        {
          label: 'Demo-only clinical shape',
          label_zh: '仅演示临床结构',
          text: 'The demo shows the dimensions a real cohort profile should expose; values are seeded UI examples, not research results.',
          text_zh: '演示页展示真实队列画像应覆盖的维度；数值是界面示例，不是研究结果。',
        },
      ],
    };
  }

  function cohortSnapshotBody() {
    const review = cohortReview();
    if (review && review.summary) {
      const s = review.summary;
      return `
      <div class="sec-stack"><div class="lbl">${cohortText('Cohort profile')}</div><h2>${cohortText('Real cohort aggregate')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Cohort size')}</div><div class="val">${fmtInt(s.cohort_size)}</div></div>
        <div class="stat"><div class="label">${cohortText('Median age')}</div><div class="val">${fmtNum(s.age && s.age.median, 1)}</div></div>
        <div class="stat"><div class="label">${cohortText('Female')}</div><div class="val">${fmtPct(s.sex && s.sex.female_pct)}</div></div>
        <div class="stat"><div class="label">${cohortText('Sepsis-3 +')}</div><div class="val">${fmtPct(s.sepsis_pct)}</div></div>
        <div class="stat"><div class="label">${cohortText('Median SOFA-2')}</div><div class="val">${fmtNum(s.sofa2 && s.sofa2.median, 1)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">${fmtPct(s.mortality_pct)}</div></div>
      </div>
      <div class="card pad mt-16">
        <div class="sec-stack mini"><div class="lbl">${t('Clinical phenotype', '临床画像')}</div><h3>${t('Interpretable cohort dimensions', '可解释的队列维度')}</h3></div>
        ${cohortClinicalProfile(s.clinical_profile)}
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Age distribution', '年龄分布')}</div>
          ${cohortDistBars(s.age && s.age.bins)}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('SOFA-2 severity', 'SOFA-2 严重度')}</div>
          ${cohortDistBars(s.sofa2 && s.sofa2.bins)}
        </div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('ICU LOS distribution', 'ICU 住院时长分布')}</div>
          ${cohortDistBars(s.los_icu_days && s.los_icu_days.bins)}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Cohort composition', '队列构成')}</div>
          ${cohortCompositionBars([
            ['Female', s.sex && s.sex.female_pct],
            ['Mortality', s.mortality_pct],
            ['Sepsis-3 +', s.sepsis_pct],
          ])}
          ${(s.admission && s.admission.bins && s.admission.bins.length) ? `<div class="eyebrow" style="margin:12px 0 8px;">${t('Admission type', '入院类型')}</div>${cohortDistBars(s.admission.bins)}` : ''}
        </div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${cohortText('Aggregate ranges')}</div>
          ${[
            ['Age', s.age],
            ['SOFA-2', s.sofa2],
            ['ICU LOS days', s.los_icu_days],
          ].map(([label, item]) => `<div class="setup-row"><span class="k">${cohortText(label)}</span><span class="vv">${t('median', '中位数')} ${fmtNum(item && item.median, 1)} · ${t('range', '范围')} ${fmtNum(item && item.min, 1)}-${fmtNum(item && item.max, 1)}</span></div>`).join('')}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${cohortText('Source provenance')}</div>
          <div class="setup-row"><span class="k">${cohortText('Source')}</span><span class="vv">${esc((review.source || {}).label || cohortText('Local export'))}</span></div>
          <div class="setup-row"><span class="k">${cohortText('Database')}</span><span class="vv">${esc((review.source || {}).database || cohortText('unknown'))}</span></div>
          <div class="setup-row"><span class="k">${cohortText('Path hash')}</span><span class="vv mono">${esc((review.source || {}).path_hash || '')}</span></div>
          <div class="setup-row"><span class="k">${cohortText('Scope')}</span><span class="vv">${esc((review.provenance || {}).payload_scope || 'cohort_aggregate_only')}</span></div>
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Real registered export aggregate. Row-level filters, generic Table One p-values, matched cohorts, and paired SOFA reclassification remain blocked; timed survival outcomes are handled in the KM module.', '真实注册导出聚合。行级筛选、通用 Table One p 值、匹配队列和配对 SOFA 重分层仍保持拦截；有事件时间的生存结局由 KM 模块处理。')}</p>`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && ws.summary) {
      const s = ws.summary;
      const ageBars = [
        ['Mean age', s.mean_age],
        ['Female %', s.female_pct],
        ['Mortality %', s.mortality],
        ['Sepsis-3 %', s.sepsis_pct],
      ];
      return `
      <div class="sec-stack"><div class="lbl">Cohort profile</div><h2>${t('Local export snapshot', '本地导出队列概览')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${t('Stays', '住院数')}</div><div class="val">${fmtInt(s.stays)}</div></div>
        <div class="stat"><div class="label">${t('Mean age', '平均年龄')}</div><div class="val">${fmtNum(s.mean_age, 1)}</div></div>
        <div class="stat"><div class="label">${t('Female', '女性')}</div><div class="val">${fmtPct(s.female_pct)}</div></div>
        <div class="stat"><div class="label">${t('Sepsis-3 +', 'Sepsis-3 阳性')}</div><div class="val">${fmtPct(s.sepsis_pct)}</div></div>
        <div class="stat"><div class="label">${t('Median SOFA-2', 'SOFA-2 中位数')}</div><div class="val">${fmtNum(s.median_sofa2, 1)}</div></div>
        <div class="stat accent"><div class="label">${t('Mortality', '死亡率')}</div><div class="val">${fmtPct(s.mortality)}</div></div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Export measures', '导出指标')}</div>
          ${ageBars.map(([lab, n]) => `<div class="qrow"><span>${lab}</span><div class="qbar"><span style="width:${n == null ? 0 : Math.max(0, Math.min(100, n))}%"></span></div><span class="qv">${fmtNum(n, 1)}</span></div>`).join('')}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Files loaded', '已加载文件')}</div>
          ${(ws.files || []).slice(0, 6).map(f => `<div class="setup-row"><span class="k">${esc(f.module || f.file)}</span><span class="vv">${fmtInt(f.rows)} rows</span></div>`).join('')}
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Real local export summary. Formal analyses still require the evidence-bound agent path.', '真实本地导出摘要。正式分析仍需走 evidence-bound agent 路径。')}</p>`;
    }
    const demoProfile = demoCohortClinicalProfile();
    const domains = [
      [t('Severity', '严重程度'), 6, t('median SOFA-2', 'SOFA-2 中位数')],
      [t('Sepsis', 'Sepsis'), 60, t('incidence', '发生率')],
      [t('Ventilation', '机械通气'), 50, t('exposure', '暴露率')],
      [t('Vasopressors', '血管活性药'), 40, t('exposure', '暴露率')],
      [t('AKI', 'AKI'), 30, t('phenotype', '表型')],
      [t('Mortality', '死亡'), 20, t('event rate', '事件率')],
    ];
    return `
      <div class="sec-stack"><div class="lbl">Cohort profile</div><h2>${t('Demo clinical cohort profile', '演示临床队列画像')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${t('Patients', '患者数')}</div><div class="val">10</div></div>
        <div class="stat"><div class="label">${t('Median age', '年龄中位数')}</div><div class="val">56</div></div>
        <div class="stat"><div class="label">${t('Female', '女性')}</div><div class="val">70%</div></div>
        <div class="stat"><div class="label">${t('Sepsis-3 +', 'Sepsis-3 阳性')}</div><div class="val">60%</div></div>
        <div class="stat"><div class="label">${t('Median SOFA', 'SOFA 中位数')}</div><div class="val">6</div></div>
        <div class="stat accent"><div class="label">${t('Mortality', '死亡率')}</div><div class="val">20%</div></div>
      </div>
      <div class="card pad mt-16">
        <div class="sec-stack mini"><div class="lbl">${t('Clinical domains', '临床维度')}</div><h3>${t('What a real cohort profile should summarize', '真实队列画像应总结哪些信息')}</h3></div>
        ${cohortClinicalProfile(demoProfile)}
      </div>
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:8px;">${t('At-a-glance phenotype balance', '一屏临床表型概览')}</div>
        <div class="cprof-spark-grid">
          ${domains.map(([label, value, hint]) => `<div class="cprof-spark">
            <div class="cprof-spark-head"><span>${esc(label)}</span><b class="mono">${typeof value === 'number' && value <= 100 && value !== 6 ? fmtPct(value) : fmtNum(value, 1)}</b></div>
            <div class="cprof-bar"><span style="width:${Math.max(4, Math.min(100, value === 6 ? 50 : value)).toFixed(1)}%"></span></div>
            <div class="cprof-d">${esc(hint)}</div>
          </div>`).join('')}
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Demo / seeded example values for UI preview — not a real run output.', '演示 / 示例数据，仅用于界面预览 —— 非真实运行结果。')}</p>`;
  }

  function cohortGroupComparisonChart(rows, columns) {
    const cols = columns || [];
    const metrics = (rows || []).filter(r => (r.values || []).some(v => typeof v === 'number' && Number.isFinite(v)));
    if (!metrics.length || !cols.length) return '';
    const colors = ['#0f766e', '#2563eb', '#8b5cf6', '#b45309'];
    return `
      <div class="cgc">
        <div class="cgc-legend">${cols.map((c, i) => `<span><i style="background:${colors[i % colors.length]};"></i>${esc(cohortText(c))}</span>`).join('')}</div>
        ${metrics.map(row => {
          const vals = (row.values || []).map(v => (typeof v === 'number' && Number.isFinite(v)) ? v : null);
          const maxV = Math.max(1, ...vals.map(v => v == null ? 0 : Math.abs(v)));
          return `
          <div class="cgc-row">
            <div class="cgc-metric">${esc(cohortText(row.metric))}${row.unit ? ` <span class="mono">${esc(cohortText(row.unit))}</span>` : ''}</div>
            <div class="cgc-bars">
              ${vals.map((v, i) => `<div class="cgc-bar"><div class="cgc-fill" style="width:${v == null ? 0 : (Math.abs(v) / maxV * 100).toFixed(0)}%;background:${colors[i % colors.length]};"></div><span class="cgc-val">${cohortProfileValue(row, v)}</span></div>`).join('')}
            </div>
          </div>`;
        }).join('')}
      </div>`;
  }

  function cohortGroupsBody() {
    const review = cohortReview();
    if (review && review.summary) {
      const s = review.summary || {};
      const source = review.source || {};
      const supported = (review.groups || {}).supported || [];
      const blocked = (review.groups || {}).blocked || [];
      const active = supported.find(row => row.id === cohortCompare) || supported[0] || {};
      const activeGroups = active.groups || [];
      const activeProfile = active.profile || {};
      const profileColumns = activeProfile.columns || [];
      const profileRows = activeProfile.rows || [];
      const radio = (row) => `<label class="radio ${active.id === row.id ? 'on' : ''}" role="button" tabindex="0" data-cohort-comp="${esc(row.id)}"><span class="mk"></span> ${esc(cohortText(row.label || row.id))}</label>`;
      return `
      <div class="coh-jump">
        <button class="cj-card" data-cohgo="coverage">
          <span class="cj-ico">${icon('shield', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${cohortText('Coverage audit')}</span><span class="cj-d">${t('Review module coverage before analysis', '分析前审阅模块覆盖率')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
        <button class="cj-card" data-cohgo="snapshot">
          <span class="cj-ico">${icon('cohort', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${cohortText('Cohort profile')}</span><span class="cj-d">${t('Inspect real registered export aggregates', '查看后端计算的本地导出聚合')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Analysis table')}</div><h2>${cohortText('Real cohort aggregate')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Cohort size')}</div><div class="val">${fmtInt(s.cohort_size)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">${fmtPct(s.mortality_pct)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Median age')}</div><div class="val">${fmtNum(s.age && s.age.median, 1)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Median SOFA-2')}</div><div class="val">${fmtNum(s.sofa2 && s.sofa2.median, 1)}</div></div>
      </div>
      <div class="note mt-12"><div class="ico">${icon('folder', 14)}</div><div class="body"><div class="t">${cohortText('Local export cohort review ready')}</div><div class="d">${cohortText('Source')} ${esc(source.label || cohortText('Local export'))} · ${esc(source.database || cohortText('unknown'))} · ${cohortText('Path hash')} <span class="mono">${esc(source.path_hash || '')}</span> · ${cohortText('aggregate-only payload')}.</div></div></div>
      ${cohortRealModuleSummary(review)}
      ${cohortRealFeaturePicker(review)}

      <div class="sec-stack"><div class="lbl">${cohortText('Comparison')}</div><h2>${cohortText('Select descriptive split')}</h2></div>
      <div class="radio-row">
        ${supported.map(row => radio(row)).join('')}
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Summary')}</div><h2>${esc(cohortText(active.label || 'Descriptive split'))} ${cohortText('Overview')}</h2></div>
      <div class="cols-3">
        ${activeGroups.map((g, i) => `<div class="stat ${i === 0 ? 'accent' : ''}"><div class="label">${esc(cohortText(g.label))}</div><div class="val">${fmtInt(g.count)}</div><div style="font-size:11px;color:var(--ink-4);margin-top:4px;">${fmtPct(g.pct)}</div></div>`).join('')}
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Descriptive profile')}</div><h2>${cohortText('Aggregate-only group characteristics')}</h2></div>
      ${cohortGroupComparisonChart(profileRows, profileColumns)}
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${cohortText('Metric')}</th>${profileColumns.map(col => `<th class="num">${esc(cohortText(col))}</th>`).join('')}<th>${cohortText('Status')}</th></tr></thead>
          <tbody>
            ${profileRows.map(row => `<tr>
              <td class="key">${esc(cohortText(row.metric))}${row.unit ? ` <span class="mono" style="color:var(--ink-4);font-weight:500;">${esc(cohortText(row.unit))}</span>` : ''}</td>
              ${(row.values || []).map(value => `<td class="num">${cohortProfileValue(row, value)}</td>`).join('')}
              <td><span class="pill ok" style="height:20px;">${cohortText('descriptive')}</span></td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Fail-closed')}</div><h2>${cohortText('Blocked cohort functions')}</h2></div>
      <div class="cols-3">
        ${blocked.map(item => `<div class="stat"><div class="label">${esc(cohortText(item.id))}</div><div class="val" style="font-size:13px;line-height:1.35;font-family:var(--font-body);font-weight:600;">${esc(cohortText(item.status))}</div><div style="font-size:11px;color:var(--ink-4);margin-top:6px;">${esc(cohortReason(item.reason))}</div></div>`).join('')}
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('No row-level filters, generic Table One p-values, SMDs, matched cohort, or paired SOFA reclassification are exposed here. Use the Survival curves tab for audited KM/log-rank when timed outcomes exist.', '这里不开放行级筛选、通用 Table One p 值、SMD、匹配队列或配对 SOFA 重分层。若存在事件时间，请在「生存曲线」页查看已审计的 KM/log-rank。')}</p>`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && ws.cohort) {
      const s = ws.summary || {};
      const c = ws.cohort || {};
      const chars = c.characteristics || [];
      return `
      <div class="coh-jump">
        <button class="cj-card" data-cohgo="coverage">
          <span class="cj-ico">${icon('shield', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('Coverage audit', '覆盖审计')}</span><span class="cj-d">${t('Review module coverage before analysis', '分析前审阅模块覆盖率')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
        <button class="cj-card" data-cohgo="snapshot">
          <span class="cj-ico">${icon('cohort', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('Cohort profile', '队列画像')}</span><span class="cj-d">${t('Inspect the local export snapshot', '查看本地导出摘要')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Analysis table')}</div><h2>${cohortText('Local export group contrast')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Total stays')}</div><div class="val">${fmtInt(s.stays)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mean age')}</div><div class="val">${fmtNum(s.mean_age, 1)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Female %')}</div><div class="val">${fmtPct(s.female_pct)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">${fmtPct(s.mortality)}</div></div>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Summary')}</div><h2>${cohortText('Outcome groups')}</h2></div>
      <div class="cols-3">
        <div class="stat"><div class="label">${cohortText('Survived')}</div><div class="val">${fmtInt(c.survived)}</div></div>
        <div class="stat"><div class="label">${cohortText('Deceased')}</div><div class="val">${fmtInt(c.deceased)}</div></div>
        <div class="stat accent"><div class="label">${cohortText('Rows reviewed')}</div><div class="val">${fmtInt((ws.tableRows || []).length)}</div></div>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Table one')}</div><h2>${cohortText('Baseline characteristics comparison')}</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${cohortText('Characteristic')}</th><th class="num">${cohortText('Overall')}</th><th class="num">${cohortText('Survived')}</th><th class="num">${cohortText('Deceased')}</th></tr></thead>
          <tbody>
            ${chars.map(r => `<tr><td class="key">${esc(cohortText(r[0]))}</td>${r.slice(1).map(c => `<td class="num">${fmtNum(c, 2)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Real local export summary. P-values and manuscript claims are intentionally withheld from this UI preview.', '真实本地导出摘要。此 UI 预览不会直接给出 p 值或稿件声明。')}</p>`;
    }
    const comparisons = {
      outcome: {
        title: 'Survived vs Deceased',
        groups: [['Survived', '8'], ['Deceased', '2'], ['Ratio', '80.0% / 20.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '52.1 (15.4)', '65.5 (17.0)', '0.31'],
          ['Male, n (%)', '3 (30.0)', '3 (37.5)', '0 (0.0)', '0.47'],
          ['SOFA, median', '6', '5', '11', '0.08'],
          ['Lactate, mmol/L', '2.4', '2.1', '4.8', '0.12'],
          ['ICU LOS, days', '5.6', '5.1', '8.4', '0.22'],
        ],
      },
      age: {
        title: 'Age Groups',
        groups: [['Age < 65', '6'], ['Age ≥ 65', '4'], ['Ratio', '60.0% / 40.0%']],
        table: [
          ['Mortality, n (%)', '2 (20.0)', '0 (0.0)', '2 (50.0)', '0.13'],
          ['SOFA, median', '6', '4', '8', '0.21'],
          ['Lactate, mmol/L', '2.4', '1.9', '3.2', '0.28'],
          ['ICU LOS, days', '5.6', '3.9', '7.8', '0.18'],
          ['Sepsis-3, n (%)', '6 (60.0)', '3 (50.0)', '3 (75.0)', '0.58'],
        ],
      },
      sex: {
        title: 'Male vs Female',
        groups: [['Female', '7'], ['Male', '3'], ['Ratio', '70.0% / 30.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '52.0 (14.1)', '61.3 (18.7)', '0.42'],
          ['Mortality, n (%)', '2 (20.0)', '2 (28.6)', '0 (0.0)', '0.51'],
          ['SOFA, median', '6', '6', '5', '0.74'],
          ['Lactate, mmol/L', '2.4', '2.2', '2.8', '0.68'],
          ['ICU LOS, days', '5.6', '5.8', '5.1', '0.81'],
        ],
      },
      los: {
        title: 'Short vs Long Stay',
        groups: [['LOS < 5d', '6'], ['LOS ≥ 5d', '4'], ['Ratio', '60.0% / 40.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '50.6 (13.9)', '61.1 (18.5)', '0.37'],
          ['Mortality, n (%)', '2 (20.0)', '0 (0.0)', '2 (50.0)', '0.13'],
          ['SOFA, median', '6', '4', '9', '0.09'],
          ['Lactate, mmol/L', '2.4', '1.8', '3.9', '0.11'],
          ['Ventilation, n (%)', '5 (50.0)', '2 (33.3)', '3 (75.0)', '0.29'],
        ],
      },
      sepsis: {
        title: 'Sepsis vs Non-sepsis',
        groups: [['Sepsis-3 +', '6'], ['Sepsis-3 -', '4'], ['Ratio', '60.0% / 40.0%']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '58.1 (16.8)', '49.9 (14.7)', '0.45'],
          ['Mortality, n (%)', '2 (20.0)', '2 (33.3)', '0 (0.0)', '0.25'],
          ['SOFA, median', '6', '8', '3', '0.06'],
          ['Lactate, mmol/L', '2.4', '3.0', '1.6', '0.16'],
          ['ICU LOS, days', '5.6', '6.7', '3.9', '0.30'],
        ],
      },
      custom: {
        title: 'Custom Threshold',
        groups: [['Above threshold', '5'], ['Below threshold', '5'], ['Example', 'SOFA ≥ 6']],
        table: [
          ['Age, mean (SD)', '54.8 (16.2)', '60.4 (15.2)', '49.2 (15.7)', '0.33'],
          ['Mortality, n (%)', '2 (20.0)', '2 (40.0)', '0 (0.0)', '0.17'],
          ['Lactate, mmol/L', '2.4', '3.3', '1.6', '0.14'],
          ['ICU LOS, days', '5.6', '7.2', '4.0', '0.23'],
          ['Sepsis-3, n (%)', '6 (60.0)', '4 (80.0)', '2 (40.0)', '0.52'],
        ],
        note: 'Demo threshold uses SOFA ≥ 6. Real custom thresholds remain fail-closed until a bounded cohort-builder backend is available.',
      },
    };
    const comp = comparisons[cohortCompare] || comparisons.outcome;
    const radio = (key, label) => `<label class="radio ${cohortCompare === key ? 'on' : ''}" role="button" tabindex="0" data-cohort-comp="${key}"><span class="mk"></span> ${cohortText(label)}</label>`;
    return `
      <div class="coh-jump">
        <button class="cj-card" data-cohgo="coverage">
          <span class="cj-ico">${icon('shield', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('Coverage audit', '覆盖审计')}</span><span class="cj-d">${t('Check module coverage before it biases a denominator', '在偏差分母之前检查模块覆盖度')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
        <button class="cj-card" data-cohgo="sofa">
          <span class="cj-ico">${icon('refresh', 16)}</span>
          <span class="cj-tx"><span class="cj-t">${t('SOFA reclassification', 'SOFA 重分层')}</span><span class="cj-d">${t('See who moves under the 2025 SOFA-2 standard', '看哪些患者在 2025 版 SOFA-2 下重分层')}</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
      </div>
      <div class="sec-stack"><div class="lbl">${cohortText('Analysis table')}</div><h2>${cohortText('Group Contrast Table')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${cohortText('Total patients')}</div><div class="val">10</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mean age')}</div><div class="val">54.8</div></div>
        <div class="stat accent"><div class="label">${cohortText('Male %')}</div><div class="val">30.0%</div></div>
        <div class="stat accent"><div class="label">${cohortText('Mortality')}</div><div class="val">20.0%</div></div>
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Comparison')}</div><h2>${cohortText('Select comparison mode')}</h2></div>
      <div class="radio-row">
        ${radio('outcome', 'Survived vs Deceased')}
        ${radio('age', 'Age Groups')}
        ${radio('sex', 'Male vs Female')}
        ${radio('los', 'Short vs Long Stay')}
        ${radio('sepsis', 'Sepsis vs Non-sepsis')}
        ${radio('custom', 'Custom Threshold')}
      </div>

      <div class="sec-stack"><div class="lbl">${cohortText('Features')}</div><h2>${cohortText('Select feature modules')}</h2></div>
      ${cohortDemoFeaturePicker()}

      <div class="sec-stack"><div class="lbl">${cohortText('Summary')}</div><h2>${esc(cohortText(comp.title))} ${cohortText('Overview')}</h2></div>
      <div class="cols-3">
        ${comp.groups.map((g, i) => `<div class="stat ${i === 2 ? 'accent' : ''}"><div class="label">${esc(cohortText(g[0]))}</div><div class="val">${esc(g[1])}</div></div>`).join('')}
      </div>
      ${comp.note ? `<div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="d" style="margin:0;">${esc(cohortReason(comp.note))}</div></div></div>` : ''}

      <div class="sec-stack"><div class="lbl">${cohortText('Table one')}</div><h2>${cohortText('Baseline characteristics comparison')}</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${cohortText('Characteristic')}</th><th class="num">${cohortText('Overall')} (n=10)</th><th class="num">${cohortText('Survived')} (n=8)</th><th class="num">${cohortText('Deceased')} (n=2)</th><th class="num">${cohortText('p-value')}</th></tr></thead>
          <tbody>
            ${comp.table.map(r => `<tr><td class="key">${esc(cohortText(r[0]))}</td>${r.slice(1).map(c => `<td class="num">${esc(c)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Demo / seeded example values for UI preview — not a real run output.', '演示 / 示例数据，仅用于界面预览 —— 非真实运行结果。')}</p>`;
  }

  S.cohort = {
    section: 'viz', nav: 'viz', sub: 'cohort',
    crumbs: ['Home', 'Data Visualization', 'Cohort Statistics'],
    get actionHtml() {
      if (cohortLoaded()) {
        return `<button class="btn" data-viz-reset>${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button><button class="btn primary" data-cohort-run>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>`;
      }
      const label = window.EU_DATA === 'real' ? t('Load export', '加载导出') : t('Run demo review', '运行演示审阅');
      const realMissingExport = window.EU_DATA === 'real' && !registryActivePath();
      return `<button class="btn primary" data-cohort-run ${cohortView === 'loading' || realMissingExport ? 'aria-disabled="true"' : ''}>${icon('play', 13)} ${label}</button>`;
    },
    rail: () => vizRail('cohort'),
    afterRender(root) {
      bindSourceRegistry(root, 'cohort');
      root.querySelectorAll('[data-cohort-run]').forEach(b => b.addEventListener('click', () => {
        if (cohortView === 'loading') return;
        if (window.EU_DATA === 'real') {
          if (!registryActivePath()) {
            cohortView = 'idle';
            window.EU_COHORT_REVIEW = null;
            window.EU_VIZ_WORKSPACE = null;
            vizErr = cohortMissingExportMessage();
            repaintScreen('cohort');
            return;
          }
          cohortView = 'loading'; repaintScreen('cohort');
          loadRealCohort(ok => { cohortView = ok ? 'loaded' : 'idle'; repaintScreen('cohort'); });
        } else {
          vizErr = null;
          cohortView = 'loading'; repaintScreen('cohort');
          setTimeout(() => { cohortView = 'loaded'; window.EU_HASWORK = true; repaintScreen('cohort'); }, 1300);
        }
      }));
      const tabsEl = root.querySelector('#cohtabs');
      if (tabsEl) tabsEl.addEventListener('click', e => {
        const b = e.target.closest('[data-cohtab]'); if (!b) return;
        if (b.dataset.cohtab === cohortPanel) return;
        cohortPanel = b.dataset.cohtab;
        repaintScreen('cohort');
      });
      root.querySelectorAll('[data-cohgo]').forEach(b => b.addEventListener('click', () => {
        cohortPanel = b.dataset.cohgo;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-viz-reset]').forEach(b => b.addEventListener('click', () => {
        cohortView = 'idle';
        cohortFeatureScope = 'recommended';
        window.EU_COHORT_REVIEW = null;
        window.EU_VIZ_WORKSPACE = null;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-use-real]').forEach(b => b.addEventListener('click', () => {
        cohortView = 'idle';
        window.EU_DATA = 'real';
        window.EU_COHORT_REVIEW = null;
        window.EU_VIZ_WORKSPACE = null;
        cohortFeatureScope = 'recommended';
        try { localStorage.setItem('easyicu_home_data', 'real'); } catch (e) {}
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-scope]').forEach(b => b.addEventListener('click', () => {
        const next = b.dataset.cohortFeatureScope === 'all' ? 'all' : 'recommended';
        if (next === cohortFeatureScope) return;
        cohortFeatureScope = next;
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-comp]').forEach(b => {
        const choose = () => {
          if (b.dataset.cohortComp === cohortCompare) return;
          cohortCompare = b.dataset.cohortComp || 'outcome';
          window.EU_STALE = true;
          repaintScreen('cohort');
        };
        b.addEventListener('click', choose);
        b.addEventListener('keydown', e => {
          if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            choose();
          }
        });
      });
      root.querySelectorAll('[data-cohort-surv-group]').forEach(b => b.addEventListener('click', () => {
        if (b.dataset.cohortSurvGroup === cohortSurvivalGroup) return;
        cohortSurvivalGroup = b.dataset.cohortSurvGroup || 'sepsis';
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-sofa-matrix-mode]').forEach(b => b.addEventListener('click', () => {
        const next = b.dataset.cohortSofaMatrixMode === 'count' ? 'count' : 'pct';
        if (next === cohortSofaMatrixMode) return;
        cohortSofaMatrixMode = next;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-sofa-granularity]').forEach(b => b.addEventListener('click', () => {
        const next = b.dataset.cohortSofaGranularity || 'medium';
        if (!SOFA_MATRIX_GRANULARITIES[next] || next === cohortSofaMatrixGranularity) return;
        cohortSofaMatrixGranularity = next;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-module]').forEach(b => b.addEventListener('click', () => {
        cohortFeatureModule = b.dataset.cohortFeatureModule || 'all';
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-toggle]').forEach(b => b.addEventListener('click', () => {
        const id = b.dataset.cohortFeatureToggle;
        if (!id || b.getAttribute('aria-disabled') === 'true') return;
        const selected = new Set(cohortSelectedFeatures.length ? cohortSelectedFeatures : cohortSelectedFeatureIds(cohortReview()));
        if (selected.has(id)) selected.delete(id);
        else selected.add(id);
        cohortSelectedFeatures = Array.from(selected);
        window.EU_STALE = true;
        reloadCohortForFeatureSelection();
      }));
      root.querySelectorAll('[data-cohort-feature-default]').forEach(b => b.addEventListener('click', () => {
        resetCohortFeatureSelection();
        window.EU_STALE = true;
        reloadCohortForFeatureSelection();
      }));
      root.querySelectorAll('[data-cohort-feature-clear]').forEach(b => b.addEventListener('click', () => {
        cohortSelectedFeatures = [];
        const review = cohortReview();
        if (review && review.feature_selection) review.feature_selection.selected = [];
        if (review && review.groups && review.groups.supported) {
          review.groups.supported.forEach(row => {
            if (row.profile && Array.isArray(row.profile.rows)) {
              row.profile.rows = row.profile.rows.filter(metric => !metric.feature_id);
            }
          });
        }
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
    },
    render() {
      if (window.__euCohortPanel) { cohortPanel = window.__euCohortPanel; window.__euCohortPanel = null; }
      const ws = window.EU_VIZ_WORKSPACE;
      let review = cohortReview();
      if (reloadStaleRealCohortIfNeeded(review)) review = null;
      const loaded = cohortLoaded();
      const head = `
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>${cohortText('Workspace')}</span> ${icon('chevron', 11)} <span>${ws ? cohortText('Local export') : (loaded ? cohortText('Demo cohort') : cohortText('Not configured'))}</span> ${icon('chevron', 11)} <span style="color:var(--ink-2);">${cohortText('Cohort statistics')}</span>
      </div>
      <div class="page-head" style="margin-bottom:16px;">
        <h1 style="margin-top:0;">${loaded ? (ws ? t('Local export cohort', '本地导出队列') : t('Sepsis vs Non-sepsis', 'Sepsis 与非 Sepsis 对照')) : t('Cohort Statistics', '队列统计')}</h1>
        <p class="lead">${loaded ? (ws ? t('Real exported module tables · local-only summary', '真实导出模块表 · 仅本地汇总') : t('Group contrast · coverage audit · cohort profile · SOFA reclassification', '组间对照 · 覆盖审计 · 队列画像 · SOFA 重分层')) : t('Choose a demo cohort review or load a registered local export before viewing group contrasts, coverage, survival curves, and SOFA reclassification.', '先运行演示队列审阅或加载已注册的本地导出，然后再查看组间对照、覆盖率、生存曲线和 SOFA 重分层。')}</p>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:9px;">${t('Key terms', '关键术语')}: ${window.gloss('cohort', t('cohort', '队列'))} · ${window.gloss('denominator', t('denominator', '分母'))} · ${window.gloss('SOFA')} · ${window.gloss('Sepsis-3')}</div>
      </div>`;
      if (cohortView !== 'loading' && !loaded) {
        if (window.EU_DATA !== 'real') {
          return head + `<div class="card pad" style="max-width:760px;" data-cohort-config-required="true">
            <div class="panel-head">
              <div>
                <div class="eyebrow">${t('Review setup required', '需要先配置审阅')}</div>
                <div class="panel-title" style="font-size:17px;">${t('Run or load a cohort review first', '请先运行或加载队列审阅')}</div>
                <div class="panel-sub mt-4">${t('Cohort Statistics no longer opens with preloaded seeded results. Start a demo review intentionally, or switch to Real and load a registered export.', '队列统计不再默认打开预加载的 seeded 结果。请明确启动演示审阅，或切换到真实模式并加载已注册导出。')}</div>
              </div>
              <span class="pill demo"><span class="dot"></span>${t('Demo available', '可用演示')}</span>
            </div>
            ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
            <div class="note info mt-16">
              <div class="ico">${icon('shield', 14)}</div>
              <div class="body">
                <div class="t">${t('Explicit setup check', '显式配置检查')}</div>
                <div class="d">${t('Demo values are seeded UI examples, not findings. Real cohort review is computed from your active local EasyICU export.', '演示值只是 seeded UI 示例，不是研究发现。真实队列审阅会从当前 active 的本地 EasyICU 导出计算。')}</div>
              </div>
            </div>
            <div class="row wrap gap-8 mt-16">
              <button class="btn primary" data-cohort-run>${icon('play', 13)} ${t('Run demo cohort review', '运行演示队列审阅')}</button>
              <button class="btn" data-cohort-use-real>${icon('db', 13)} ${t('Use real export', '使用真实导出')}</button>
              <button class="btn" data-nav="extraction">${icon('extract', 13)} ${t('Open Data Extraction', '打开数据抽取')}</button>
            </div>
          </div>
          <div class="empty mt-16" data-cohort-empty-preview="true">
            <div class="glyph">${icon('cohort', 22)}</div>
            <div class="t">${t('Cohort review awaits setup', '队列审阅等待配置')}</div>
            <div class="d">${t('After setup, this page will show group contrast, KM/log-rank, coverage audit, cohort profile, and SOFA reclassification.', '配置后，这里会显示组间对照、KM/log-rank、生存风险表、覆盖审计、队列画像和 SOFA 重分层。')}</div>
          </div>`;
        }
        return head + `<div class="card pad" style="max-width:720px;">
          <div class="panel-title" style="font-size:17px;">${t('Load a local export first', '请先加载本地导出')}</div>
          <div class="panel-sub mt-4">${t('Cohort Statistics uses the same export snapshot as Patient Review.', '队列统计使用与患者明细相同的导出快照。')}</div>
          ${!registryActivePath() ? `<div class="note info mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('No active export selected', '尚未选择 active 导出')}</div><div class="d">${cohortMissingExportMessage()}</div></div></div>` : ''}
          ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
          ${sourceRegistryBlock('single')}
          <button class="btn primary mt-16" data-cohort-run ${!registryActivePath() ? 'aria-disabled="true"' : ''}>${icon('folder', 14)} ${t('Load local export', '加载本地导出')}</button>
        </div>`;
      }
      if (cohortView === 'loading') {
        return head + `<div class="card pad">
          <div class="load-strip">
            <span class="spin accent"></span>
            <div class="grow"><div style="font-weight:600;font-size:12.75px;">${t('Recomputing cohort statistics…', '正在重新计算队列统计…')}</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${t('reproducible · no outbound calls', '可复现 · 无外部调用')}</div></div>
          </div>
          <div class="indet mt-12"></div>
          <div class="st-stats mt-16">${[0,1,2,3].map(() => `<div class="sk-stat"><div class="sk sk-line sm" style="width:52%"></div><div class="sk" style="height:22px;width:64%;margin-top:10px;"></div></div>`).join('')}</div>
          <div class="sk-table mt-16">${[0,1,2,3,4].map(() => `<div class="sk-trow">${[60,40,40,40,30].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}</div>
        </div>`;
      }
      const demoScope = cohortDemoCatalogScope();
      const preflightItems = [
        [
          cohortText('Input package'),
          ws ? `${fmtInt(ws.summary && ws.summary.stays)} ${t('entities', '个实体')} · ${fmtInt(ws.summary && ws.summary.modules)} ${t('modules', '个模块')}` : `10 ${t('stays', '次住院')} · ${fmtInt(demoScope.selectedModuleCount)} / ${fmtInt(demoScope.totalModuleCount)} ${t('modules', '个模块')} · ${fmtInt(demoScope.selectedFeatureCount)} / ${fmtInt(demoScope.totalFeatureCount)} ${t('features', '个特征')}`,
          'ok',
          null,
        ],
        [
          cohortText('Backend evidence checks'),
          review ? cohortText('manifest parsed · denominators previewed · aggregate payload returned') : cohortText('coverage + denominators ready'),
          'ok',
          null,
        ],
        [
          cohortText('Draft review'),
          cohortText('locked · requires Agent sign-off'),
          'warn',
          'agent',
        ],
      ];
      return head + `
      <div class="card" style="padding:0;overflow:hidden;">
        <div class="row" style="justify-content:space-between;padding:11px 16px;border-bottom:1px solid var(--hair);">
          <span style="font-weight:600;font-size:12.5px;">${cohortText('Agent preflight')}</span>
          <span class="mono" style="font-size:11px;color:var(--ink-4);">${cohortText('current session')}</span>
        </div>
        <div class="preflight">
          ${preflightItems.map(([tt, d, s, nav]) => `
            <div class="pf-cell" ${nav ? `data-nav="${nav}" role="button" tabindex="0" style="cursor:pointer;"` : ''}>
              <div class="eyebrow" style="display:flex;align-items:center;gap:6px;">
                <span class="dot-${s}"></span>${tt}${nav ? `<span style="margin-left:auto;color:var(--ink-4);">${icon('arrow', 12)}</span>` : ''}
              </div>
              <div style="font-size:12.5px;color:var(--ink-2);margin-top:6px;">${d}</div>
              ${nav ? `<div style="font-size:11px;color:var(--ink-4);margin-top:4px;">${review ? t('Aggregate payload is ready; open Agent for evidence-bound draft review.', '聚合载荷已就绪；打开 Agent 做证据绑定草稿核验。') : t('Demo review is local-only; open Agent only after choosing a real export.', '演示审阅仅限本地预览；选择真实导出后再打开 Agent。')}</div>` : ''}
            </div>`).join('')}
        </div>
      </div>

      ${cohortTabs()}
      <div id="cohbody">${cohortPanelBody()}</div>`;
    },
  };

  /* ---------------- CROSS-DB BENCHMARK ---------------- */
  const CROSS_DBS = [
    ['MIMIC-IV', true, 'miiv'], ['eICU-CRD', true, 'eicu'], ['AmsterdamUMCdb', true, 'aumc'],
    ['HiRID', true, 'hirid'], ['MIMIC-III', true, 'mimic'], ['SICdb', true, 'sic'],
  ];

  function crossTerm(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      'Workspace': '工作区',
      'Demo simulated frames': '演示模拟特征帧',
      'Local exports': '本地导出',
      'Local export': '本地导出',
      'Demo cohort': '演示队列',
      'Not configured': '未配置',
      'Cross-DB benchmark': '跨库对比',
      'Cross-DB benchmark ready': '跨库对比已就绪',
      'Raw ICU data root': '原始 ICU 数据根目录',
      'Real raw database mode': '真实原始数据库模式',
      'Databases': '数据库',
      'Available databases': '可用数据库',
      'selected': '已选择',
      'add': '添加',
      'Loaded': '已加载',
      'Change selection': '更改选择',
      'Export JSON': '导出 JSON',
      'Re-run': '重新运行',
      'Run': '运行',
      'Run benchmark': '运行对比',
      'Load real density benchmark': '加载真实密度对比',
      'Loaded seeded distribution summary': '已加载的种子分布摘要',
      'Loaded raw-database distribution summary': '已加载的原始数据库分布摘要',
      'Loaded cross-database export summary': '已加载的跨库导出摘要',
      'Source provenance': '来源溯源',
      'Module availability matrix': '模块可用性矩阵',
      'Shared exported modules': '共享导出模块',
      'Metric': '指标',
      'Module': '模块',
      'Shared': '共享',
      'Missing': '缺失',
      'Present': '存在',
      'Yes': '是',
      'No': '否',
      'Database': '数据库',
      'Values': '取值数',
      'Range': '范围',
      'Density points': '密度点',
      'All modules': '全部模块',
      'Fail-closed scope': '保守拦截范围',
      'Select databases to compare': '选择要对比的数据库',
      'Demo benchmark was not loaded': '演示对比尚未加载',
      'No shared modules detected': '未识别到共享模块',
      'unsupported analyses': '不支持的分析',
      'aggregate density only': '仅聚合密度',
      'database curves': '条数据库曲线',
      'values': '个取值',
      'folder check ready': '文件夹检查通过',
      'check folders · need ≥ 2 detected': '检查文件夹 · 至少需识别 2 个',
      'all supported catalog concepts': '全部受支持的标准概念',
      'local only': '仅本地',
      'local-only · nothing uploaded': '仅本地 · 不上传',
      'root hash': '根目录哈希',
      'path hash': '路径哈希',
      'demo seed': '演示种子',
    };
    return t(raw, map[raw] || raw);
  }

  function crossMetricLabel(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      'Feature rows': '特征行数',
      'Concepts present': '已识别概念',
      'stays': '住院数',
      'cohort_size': '队列规模',
      'modules': '模块数',
      'total_rows': '总行数',
      'total_records': '总记录数',
      'female_pct': '女性比例',
      'mortality': '死亡率',
      'mortality_pct': '死亡率',
      'sepsis_pct': 'Sepsis-3 比例',
      'coverage_median_pct': '覆盖率中位数',
      'sofa2 median': 'SOFA-2 中位数',
      'hr median': '心率中位数',
    };
    if (map[raw]) return t(raw, map[raw]);
    const sofa = raw.match(/^sofa2_([a-z]+)_median$/);
    if (sofa) {
      const organs = {
        resp: '呼吸',
        coag: '凝血',
        liver: '肝脏',
        cardio: '循环',
        cns: '中枢神经',
        renal: '肾脏',
      };
      return t(raw, `SOFA-2 ${organs[sofa[1]] || sofa[1]} 中位数`);
    }
    return raw.replace(/_/g, ' ');
  }

  function crossStatusLabel(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      compatible: '可对比',
      descriptive_only: '仅描述性',
      blocked: '已拦截',
      blocked_until_numeric_evidence_gate: '待数值证据核验通过前拦截',
      matched_cohort: '匹配队列',
      inferential_statistics: '推断统计',
      row_level_filters: '行级筛选',
      queued: '排队中',
      cancel: '取消中',
      running: '运行中',
      done: '完成',
      failed: '失败',
    };
    return t(raw, map[raw] || raw);
  }

  function crossProgressMessage(value) {
    const raw = String(value == null ? '' : value);
    const map = {
      'Queued local raw Cross-DB density job.': '本地原始跨库密度任务已排队。',
      'Cancel requested. The current database read may finish before the job stops.': '已请求取消。当前数据库读取可能会先完成，然后任务才停止。',
      'Starting local raw Cross-DB density job…': '正在启动本地原始跨库密度任务…',
      'Building seeded density frames…': '正在生成种子密度特征帧…',
      'Loading real feature densities from local databases…': '正在从本地数据库加载真实特征密度…',
      'Loading seeded frames for selected databases…': '正在为所选数据库加载种子特征帧…',
    };
    return t(raw, map[raw] || raw);
  }

  function crossHeader() {
    const ws = window.EU_VIZ_WORKSPACE;
    const xdb = window.EU_CROSSDB_WORKSPACE;
    const xdbDemo = xdb && xdb.source_type === 'legacy_simulated_multidb_feature_frames';
    const sourceLabel = xdb
      ? (xdbDemo ? crossTerm('Demo simulated frames') : crossTerm('Local exports'))
      : (ws ? crossTerm('Local export') : (window.EU_DATA === 'real' ? crossTerm('Not configured') : crossTerm('Demo cohort')));
    return `
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>${crossTerm('Workspace')}</span> ${icon('chevron', 11)} <span>${sourceLabel}</span> ${icon('chevron', 11)} <span style="color:var(--ink-2);">${crossTerm('Cross-DB benchmark')}</span>
      </div>
      <div class="page-head" style="margin-bottom:14px;">
        <h1 style="margin-top:0;">${crossTerm('Cross-DB benchmark')}</h1>
        <p class="lead">${window.EU_DATA === 'real' ? t('Load real ICU database folders and compare feature density distributions by module.', '加载本地真实 ICU 数据库文件夹，并按模块对比特征密度分布。') : t('Same cohort definition compared across ≥2 ICU databases.', '用同一个队列定义对比两个或更多 ICU 数据库。')}</p>
      </div>`;
  }
  function crossFmt(key, value) {
    if (value == null) return '—';
    if (key === 'stays' || key === 'cohort_size' || key === 'modules' || key === 'total_rows' || key === 'total_records') return fmtInt(value);
    if (key === 'feature_rows' || key === 'concepts_present') return fmtInt(value);
    if (key === 'female_pct' || key === 'mortality' || key === 'mortality_pct' || key === 'sepsis_pct' || key === 'coverage_median_pct') return fmtPct(value);
    return fmtNum(value, 1);
  }
  function rawCrossdbAliasSummary(scan) {
    const aliases = scan && scan.aliases ? scan.aliases : {};
    return Object.keys(aliases).map(key => {
      const row = aliases[key] || {};
      const names = (row.aliases || []).slice(0, 4).join('/');
      return `${row.label || key}: ${names}`;
    }).join(' · ');
  }
  function rawCrossdbDbStatus(dbKey, selected) {
    const path = defaultRawCrossdbRoot();
    if (crossRawRootScanning && crossRawRootScanPath === path) {
      return { cls: 'dashed', label: t('checking', '检查中'), sub: t('checking folder', '正在检查文件夹') };
    }
    const current = crossRawScanCurrentFor(path) ? crossRawRootScan : null;
    if (!current || current.ok === false) {
      return {
        cls: selected ? 'ok' : 'dashed',
        label: selected ? t('selected', '已选择') : t('add', '添加'),
        sub: t('not checked yet', '尚未检查'),
      };
    }
    const detected = (current.detected || []).find(row => row.key === dbKey);
    if (detected) {
      return {
        cls: selected ? 'ok' : 'dashed',
        label: selected ? t('detected', '已识别') : t('not selected', '未选择'),
        sub: detected.folder_name ? `${t('folder', '文件夹')} ${detected.folder_name}` : t('recognized folder', '已识别文件夹'),
      };
    }
    return {
      cls: selected ? 'warn' : 'dashed',
      label: selected ? t('missing', '缺失') : t('not found', '未找到'),
      sub: selected ? t('not found in root', '根目录中未找到') : t('not detected', '未识别'),
    };
  }
  function rawCrossdbScanPanel() {
    const path = defaultRawCrossdbRoot();
    if (!path) {
      return `<div class="note info mt-12">
        <div class="ico">${icon('folder', 14)}</div>
        <div class="body"><div class="t">${t('Choose a parent folder first', '先选择一个父文件夹')}</div><div class="d">${t('It should contain database subfolders. Accepted aliases include mimiciv, mimic-iv, miiv, eicu, eicu-crd, aumc, amsterdamumc, hirid, mimiciii, sicdb, and sic.', '它应包含数据库子文件夹。可识别别名包括 mimiciv、mimic-iv、miiv、eicu、eicu-crd、aumc、amsterdamumc、hirid、mimiciii、sicdb、sic。')}</div></div>
      </div>`;
    }
    if (crossRawRootScanning && crossRawRootScanPath === path) {
      return `<div class="note info mt-12">
        <div class="ico"><span class="spin accent"></span></div>
        <div class="body"><div class="t">${t('Checking database folders', '正在检查数据库文件夹')}</div><div class="d">${t('EasyICU is matching top-level folders against supported database aliases. No patient rows are read.', 'EasyICU 正在用支持的数据库别名匹配顶层文件夹；不会读取患者行。')}</div></div>
      </div>`;
    }
    if (!crossRawScanCurrentFor(path)) {
      return `<div class="note warn mt-12">
        <div class="ico">${icon('alert', 14)}</div>
        <div class="body"><div class="t">${t('Folder not checked', '文件夹尚未检查')}</div><div class="d">${t('Check this root before running so missing or custom-named database folders are visible.', '运行前先检查这个根目录，这样缺失或自定义命名的数据库文件夹会直接显示出来。')}</div></div>
      </div>`;
    }
    const scan = crossRawRootScan || {};
    if (scan.ok === false) {
      return `<div class="note warn mt-12">
        <div class="ico">${icon('alert', 14)}</div>
        <div class="body"><div class="t">${t('Folder check failed', '文件夹检查失败')}</div><div class="d">${esc(scan.hint || scan.error || t('Could not check this folder.', '无法检查该文件夹。'))}</div></div>
      </div>`;
    }
    const status = crossRawSelectionStatusFor(path);
    const detected = scan.detected || [];
    const selectedDetected = new Set(status.detectedSelectedKeys);
    const missing = status.missingSelectedKeys.map(key => ({
      key,
      label: crossRawDbLabel(key),
    }));
    const unknown = scan.unrecognized_folders || [];
    const tone = status.runnable ? 'ok' : 'warn';
    const title = status.runnable ? t('Folder check ready', '文件夹检查通过') : t('Folder check needs attention', '文件夹检查需要处理');
    return `<div class="note ${tone} mt-12">
      <div class="ico">${icon(status.runnable ? 'check' : 'alert', 14)}</div>
      <div class="body">
        <div class="t">${title}</div>
        <div class="d">${t('Detected database folders', '已识别数据库文件夹')}: ${fmtInt(detected.length)} · ${t('selected recognized', '已选且识别')}: ${fmtInt(status.detectedSelectedKeys.length)}/${fmtInt(status.selectedKeys.length)} · ${t('need at least 2', '至少需要 2 个')}.</div>
        <div class="row gap-8 mt-8" style="flex-wrap:wrap;">
          ${detected.length ? detected.map(row => `<span class="chip ${selectedDetected.has(row.key) ? 'solid' : ''}">${esc(row.label || row.key)} · ${esc(row.folder_name || row.key)}${selectedDetected.has(row.key) ? '' : ` · ${t('not selected', '未选择')}`}</span>`).join('') : `<span class="pill warn">${t('No supported database folders detected', '未识别到支持的数据库文件夹')}</span>`}
        </div>
        ${missing.length ? `<div class="d mt-8">${t('Missing selected database folders', '已选但缺失的数据库文件夹')}: ${missing.map(row => esc(row.label || row.key)).join(', ')}</div>` : ''}
        ${unknown.length ? `<div class="d mt-8">${t('Unrecognized folders', '未识别文件夹')}: ${unknown.map(esc).join(', ')}${scan.unrecognized_count > unknown.length ? ` +${fmtInt(scan.unrecognized_count - unknown.length)}` : ''}</div>` : ''}
        <div class="d mt-8">${t('Accepted aliases', '可识别别名')}: ${esc(rawCrossdbAliasSummary(scan))}</div>
      </div>
    </div>`;
  }
  function rawCrossdbSetup() {
    const sel = CROSS_DBS.filter(d => d[1]).length;
    const rawRoot = defaultRawCrossdbRoot();
    const canRun = sel >= 2 && crossRawScanReadyFor(rawRoot);
    const sampleProfile = crossRawSampleProfile();
    const sampleProfiles = crossRawSampleProfiles();
    return `
      <div class="note info">
        <div class="ico">${icon('benchmark', 16)}</div>
        <div class="body"><span class="t">${crossTerm('Real raw database mode')}</span> <span class="d" style="display:inline;">— ${t('Choose a local ICU data root containing database folders, then compare all catalog concepts with cross-database support. No rows leave this machine.', '选择一个包含数据库子文件夹的本地 ICU 数据根目录，然后对比所有具备跨库支持的标准概念。不会有任何行级数据离开本机。')}</span></div>
      </div>
      ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
      <div class="card pad mt-16">
        <div class="row between gap-12" style="align-items:flex-start;">
          <div>
            <div class="panel-title">${crossTerm('Raw ICU data root')}</div>
            <div class="panel-sub mt-4">${t('Folder that contains database subfolders such as mimiciv, eicu, aumc, hirid, mimiciii, or sic.', '包含数据库子文件夹的目录，例如 mimiciv、eicu、aumc、hirid、mimiciii 或 sic。')}</div>
          </div>
          <span class="pill ok"><span class="dot"></span>${crossTerm('local only')}</span>
        </div>
        <div class="path-field editable mt-14">
          <span class="pf-ico">${icon('folder', 14)}</span>
          <input class="pf-input" data-crossdb-root type="text" spellcheck="false" autocomplete="off" value="${esc(defaultRawCrossdbRoot())}" placeholder="${esc(t('Paste a local ICU database root folder', '粘贴本地 ICU 数据库根目录'))}" aria-label="${esc(t('ICU data root', 'ICU 数据根目录'))}" />
          <button class="btn sm" type="button" data-crossdb-root-browse>${icon('folder', 12)} ${t('Browse...', '浏览...')}</button>
          <button class="btn sm" type="button" data-crossdb-root-scan>${icon('search', 12)} ${t('Check folders', '检查文件夹')}</button>
        </div>
        ${rawCrossdbScanPanel()}
      </div>
      <div class="card pad mt-16">
        <div class="row between gap-12" style="align-items:flex-start;">
          <div>
            <div class="panel-title">${t('Sampling budget before plotting', '绘图前抽样预算')}</div>
            <div class="panel-sub mt-4">${t('Raw Cross-DB density uses bounded local sampling so six databases do not trigger an unbounded full-table scan.', '原始跨库密度使用有界本地抽样，避免六个数据库触发无界全表扫描。')}</div>
          </div>
          <span class="pill ok">${esc(crossRawSampleSummary(sampleProfile))}</span>
        </div>
        <div class="db-grid mt-14" style="grid-template-columns:repeat(3,minmax(0,1fr));">
          ${sampleProfiles.map(profile => `
            <button class="db-card ${profile.id === sampleProfile.id ? 'sel' : ''}" type="button" data-crossdb-sample-mode="${esc(profile.id)}" style="text-align:left;">
              <div style="min-width:0;">
                <div style="font-weight:650;font-size:12.5px;">${esc(profile.label)}</div>
                <div class="mono" style="font-size:10.5px;color:var(--ink-4);">≤${fmtInt(profile.maxPatients)} ${t('entities/database', '实体/数据库')} · ≤${fmtInt(profile.sampleSize)} ${t('values/feature', '值/特征')}</div>
                <div style="font-size:11px;color:var(--ink-3);margin-top:4px;">${esc(profile.note)}</div>
              </div>
              <span class="db-mk pill ${profile.id === sampleProfile.id ? 'ok' : 'dashed'}" style="flex:none;height:20px;">${profile.id === sampleProfile.id ? `<span class="dot"></span>${t('selected', '已选择')}` : t('choose', '选择')}</span>
            </button>`).join('')}
        </div>
      </div>
      <div class="sec-stack"><div class="lbl">${crossTerm('Databases')} · <span id="dbcount">${sel}</span> ${crossTerm('selected')}</div></div>
      <div class="db-grid" id="dbgrid">
        ${CROSS_DBS.map(([n, on, key]) => {
          const status = rawCrossdbDbStatus(key, on);
          return `
          <div class="db-card ${on ? 'sel' : ''}" data-db="${CROSS_DBS.findIndex(d => d[0] === n)}">
            <div class="row gap-8" style="min-width:0;">
              <span class="${on ? '' : 'ink-4'}" style="flex:none;color:${on ? 'var(--accent-ink)' : 'var(--ink-4)'};">${icon('db', 15)}</span>
              <div style="min-width:0;">
                <div style="font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${n}</div>
                <div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc(status.sub)}</div>
              </div>
            </div>
            <span class="db-mk pill ${status.cls}" style="flex:none;height:20px;">${status.cls === 'ok' ? '<span class="dot"></span>' : ''}${esc(status.label)}</span>
          </div>`;
        }).join('')}
      </div>
      <div class="gate-strip mt-20">
        <span class="pill"><span style="color:var(--ink-3);">${icon('benchmark', 12)}</span> <span id="runhint">${sel} / 6 · ${canRun ? crossTerm('folder check ready') : crossTerm('check folders · need ≥ 2 detected')}</span></span>
        <span class="pill">${crossTerm('all supported catalog concepts')}</span>
        <span class="pill">${esc(crossRawSampleSummary(sampleProfile))}</span>
        <div class="grow"></div>
        <button class="btn primary" data-run ${canRun ? '' : 'aria-disabled="true"'}>${icon('play', 13)} ${crossTerm('Load real density benchmark')}</button>
      </div>`;
  }
  function crossAvailCell(v) {
    if (!v || !v.present) return `<td class="num xdb-avail-cell missing">${crossTerm('Missing')}</td>`;
    const pct = (typeof v.coverage_pct === 'number') ? v.coverage_pct : null;
    if (pct == null) return `<td class="num xdb-avail-cell present">${crossTerm('Present')}</td>`;
    const bg = pct >= 80 ? 'rgba(15,118,110,0.18)' : pct >= 50 ? 'rgba(180,83,9,0.16)' : 'rgba(190,18,60,0.14)';
    return `<td class="num xdb-avail-cell" style="background:${bg};">${fmtPct(pct)}</td>`;
  }

  function crossRealLoaded(xdb) {
    const sources = xdb.sources || [];
    const labels = sources.map(s => s.label || s.database || 'local');
    const shared = xdb.shared_modules || [];
    const availability = xdb.availability || [];
    const provenance = xdb.provenance || {};
    const privacy = xdb.privacy || {};
    const blocked = xdb.blocked_features || [];
    const gate = xdb.compatibility_gate || {};
    const gateStatus = gate.status || 'compatible';
    const mode = gate.comparison_mode || 'descriptive_only';
    const rawMode = xdb.source_type === 'raw_database_root';
    const demoMode = xdb.source_type === 'legacy_simulated_multidb_feature_frames';
    const readyTitle = demoMode
      ? t('Seeded simulated density benchmark ready', '种子模拟密度对比已就绪')
      : (rawMode ? t('Real raw-database density benchmark ready', '真实原始数据库密度对比已就绪') : t('Real cross-database benchmark ready', '真实跨数据库对比已就绪'));
    const sourceUnit = rawMode || demoMode ? t('databases', '个数据库') : t('exports', '个导出');
    const sourceMode = demoMode ? crossTerm('demo seed') : (rawMode ? crossTerm('root hash') : crossTerm('path hash'));
    const noteTitle = demoMode
      ? t('Legacy seeded feature-frame distribution', '旧版种子特征帧分布')
      : (rawMode ? t('Raw ICU database distribution', '原始 ICU 数据库分布') : t('Registered export comparison', '已注册导出对比'));
    const noteDetail = demoMode
      ? t('Feature density curves are computed from the old clinically-shaped demo generator, then aggregated by module; this is not a user database.', '特征密度曲线来自旧版临床形态演示生成器，并按模块聚合；这不是用户数据库。')
      : (rawMode ? t('Feature density curves are computed from local ICU database folders through easyicu.load_concepts; no patient rows are returned.', '特征密度曲线通过 easyicu.load_concepts 从本地 ICU 数据库文件夹计算；不会返回患者行级数据。') : t('Cross-DB aggregate-only payload from the local source registry. Matched cohort definitions and formal claims still require the evidence-bound agent path.', '这是来自本地来源注册表的跨库仅聚合载荷。匹配队列定义和正式声明仍需走证据绑定 Agent 路径。'));
    const blockedIds = blocked.map(item => crossStatusLabel(item.id)).join(', ') || crossTerm('unsupported analyses');
    return `
      <div class="loaded-bar">
        <span class="pill ok"><span class="dot"></span>${crossTerm('Loaded')}</span>
        <div class="grow"><span style="font-weight:600;font-size:13px;">${readyTitle}</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${fmtInt(sources.length)} ${sourceUnit} · ${fmtInt(shared.length)} ${t('shared modules', '个共享模块')}</span></div>
        <button class="btn sm" data-viz-reset>${icon('sliders', 13)} ${crossTerm('Change selection')}</button>
        <button class="btn sm" data-crossdb-export>${icon('download', 13)} ${crossTerm('Export JSON')}</button>
      </div>
      ${crossDbRecordCards(sources, labels)}
      ${crossRealFeatureDensityByModule(xdb.feature_distributions || [], labels)}
      <div class="note info mt-16">
        <div class="ico">${icon('benchmark', 16)}</div>
        <div class="body"><span class="t">${noteTitle}</span> <span class="d" style="display:inline;">— ${noteDetail}</span></div>
      </div>
      <div class="note ok mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">${t('Compatibility gate', '兼容性核验')}: ${esc(crossStatusLabel(gateStatus))}</span> <span class="d" style="display:inline;">— ${esc(crossStatusLabel(mode))} · ${crossStatusLabel('matched_cohort')}=false · ${crossStatusLabel('inferential_statistics')}=false.</span></div>
      </div>
      <details class="xdb-audit mt-16">
      <summary>${t('Provenance & audit', '溯源与审计')} · ${t('source hashes, distribution summary, availability matrix, scope', '来源哈希、分布摘要、可用性矩阵、范围')}</summary>
      <div class="sec-stack"><div class="lbl">${crossTerm('Source provenance')}</div></div>
      <div class="src-grid">
        ${sources.map(source => `
          <div class="src-card">
            <div class="row gap-8" style="min-width:0;">
              <span style="color:var(--accent-ink);">${icon('db', 15)}</span>
              <div style="min-width:0;">
                <div style="font-weight:650;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(source.label || crossTerm('Local export'))}</div>
                <div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc((source.database || t('local', '本地')).toUpperCase())} · ${sourceMode} ${esc(source.path_hash || '—')}</div>
              </div>
            </div>
          </div>`).join('')}
      </div>
      <div class="sec-stack"><div class="lbl">${demoMode ? crossTerm('Loaded seeded distribution summary') : (rawMode ? crossTerm('Loaded raw-database distribution summary') : crossTerm('Loaded cross-database export summary'))}</div></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${crossTerm('Metric')}</th>${labels.map(c => `<th class="num">${esc(c)}</th>`).join('')}<th class="num">Δ ${t('range', '范围')}</th></tr></thead>
          <tbody>
            ${(xdb.rows || []).map(row => `<tr><td class="key">${esc(crossMetricLabel(row.label || row.key))}</td>${(row.values || []).map(v => `<td class="num">${crossFmt(row.key, v)}</td>`).join('')}<td class="num" style="color:var(--ink-3);">${crossFmt(row.key, row.delta)}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="sec-stack"><div class="lbl">${crossTerm('Module availability matrix')}</div></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>${crossTerm('Module')}</th>${labels.map(c => `<th class="num">${esc(c)}</th>`).join('')}<th class="num">${crossTerm('Shared')}</th></tr></thead>
          <tbody>
            ${availability.map(row => `<tr><td class="key">${esc(catalogModuleLabel(row.module))}</td>${(row.values || []).map(v => crossAvailCell(v)).join('')}<td class="num">${row.shared ? crossTerm('Yes') : crossTerm('No')}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="sec-stack"><div class="lbl">${crossTerm('Shared exported modules')}</div></div>
      <div class="row wrap gap-6">
        ${shared.length ? shared.map(m => `<span class="chip solid">${esc(catalogModuleLabel(m))}</span>`).join('') : `<span class="pill warn">${crossTerm('No shared modules detected')}</span>`}
      </div>
      <div class="note warn mt-16">
        <div class="ico">${icon('lock', 16)}</div>
        <div class="body"><span class="t">${crossTerm('Fail-closed scope')}</span> <span class="d" style="display:inline;">— ${blockedIds} ${t('remain blocked', '保持拦截')}。${t('Raw rows returned', '是否返回原始行')}=${privacy.raw_rows_returned === true ? 'true' : 'false'}；${t('inference', '推断')}=${esc(crossStatusLabel(provenance.inference || 'blocked_until_numeric_evidence_gate'))}。</span></div>
      </div>
      </details>`;
  }

  function crossDbRecordCards(sources, labels) {
    /* Legacy Figure-3 Cross-DB header: one record-count card per database, color-keyed to the density legend. */
    const cards = (sources || []).map((source, i) => {
      const label = labels[i] || source.label || source.database || `DB ${i + 1}`;
      const summary = source.summary || {};
      const records = summary.total_records != null ? summary.total_records : summary.cohort_size;
      const color = densityPalette(i);
      return `
        <div class="xdb-rec-card" style="border-left-color:${color};">
          <div class="xdb-rec-name"><span class="xdb-rec-dot" style="background:${color};"></span>${esc(label)}</div>
          <div class="xdb-rec-value">${records != null ? fmtInt(records) : '—'}</div>
          <div class="xdb-rec-unit">${t('records', '条记录')}</div>
        </div>`;
    }).join('');
    if (!cards) return '';
    return `<div class="xdb-rec-cards mt-16">${cards}</div>`;
  }

  function crossRealFeatureDensityByModule(modules, labels) {
    return crossFeatureDensityPanel(
      t('Multi-database feature density grid', '多数据库特征密度网格'),
      t('Old Cross-DB layout: one subplot per feature, grouped by module; each subplot overlays the selected database density curves. No patient rows are returned.', '旧版 Cross-DB 布局：每个特征一个小图，按模块分组；每个小图叠加所选数据库的密度曲线。不会返回患者行级数据。'),
      modules,
      labels,
    );
  }

  function crossFeatureDensityPanel(title, subtitle, modules, labels) {
    const allCleaned = (modules || []).filter(module => module && (module.features || []).length);
    if (!allCleaned.length) return '';
    /* Curated default: restore the legacy Figure-3 "one subplot per canonical concept" grid
       instead of dumping all ~247 catalog features. 'all' is opt-in. */
    const canonSet = new Set(CROSS_DENSITY_CANON);
    const coreModules = allCleaned
      .map(module => ({ ...module, features: (module.features || []).filter(f => canonSet.has(String(f.feature || '').toLowerCase())) }))
      .filter(module => module.features.length);
    const scope = (crossDensityScope === 'all' || !coreModules.length) ? 'all' : 'core';
    const cleaned = scope === 'core' ? coreModules : allCleaned;
    const allFeatureCount = allCleaned.reduce((acc, module) => acc + (module.features || []).length, 0);
    const coreFeatureCount = coreModules.reduce((acc, module) => acc + (module.features || []).length, 0);
    let selectedModule = crossDensityModule || 'all';
    if (selectedModule !== 'all' && !cleaned.some(module => module.module === selectedModule)) selectedModule = 'all';
    const visible = selectedModule === 'all' ? cleaned : cleaned.filter(module => module.module === selectedModule);
    const totalFeatures = cleaned.reduce((acc, module) => acc + (module.features || []).length, 0);
    const sharedFeatures = cleaned.reduce((acc, module) => acc + Number(module.shared_feature_count || 0), 0);
    const visibleFeatures = visible.reduce((acc, module) => acc + (module.features || []).length, 0);
    const labelRow = (labels || []).map((label, i) => `<span><i style="background:${densityPalette(i)};"></i>${esc(label)}</span>`).join('');
    const moduleOptions = [
      `<option value="all" ${selectedModule === 'all' ? 'selected' : ''}>${esc(t('All modules', '全部模块'))}</option>`,
      ...cleaned.map(module => `<option value="${esc(module.module)}" ${selectedModule === module.module ? 'selected' : ''}>${esc(catalogModuleLabel(module.module))} (${fmtInt((module.features || []).length)})</option>`),
    ].join('');
    const detail = findCrossDensityFeature(visible, crossDensityFeature) || (visible[0] && visible[0].features && { module: visible[0], row: visible[0].features[0] });
    if (detail && (!crossDensityFeature || !findCrossDensityFeature(visible, crossDensityFeature))) crossDensityFeature = crossFeatureKey(detail.module, detail.row);
    const scopeToggle = coreModules.length ? `
        <div class="xdb-density-scope">
          <button class="chip ${scope === 'core' ? 'solid' : ''}" data-density-scope="core">${t('Core concepts', '核心概念')} <span class="mono">${fmtInt(coreFeatureCount)}</span></button>
          <button class="chip ${scope === 'all' ? 'solid' : ''}" data-density-scope="all">${t('All features', '全部特征')} <span class="mono">${fmtInt(allFeatureCount)}</span></button>
        </div>` : '';
    return `
      <div class="sec-stack"><div class="lbl">${esc(title)}</div></div>
      <div class="xdb-density-panel" data-density-total="${totalFeatures}">
        <div class="xdb-density-top">
          <div>
            <div class="xdb-density-title">${esc(title)}</div>
            <div class="xdb-density-sub">${esc(subtitle)}</div>
            <div class="xdb-density-meta mono">${scope === 'core' ? `${t('curated core concepts', '精选核心概念')} · ` : ''}${fmtInt(cleaned.length)} ${t('modules', '个模块')} · ${fmtInt(totalFeatures)} ${t('features', '个特征')} · ${fmtInt(sharedFeatures)} ${t('shared across selected databases', '个在所选数据库间共享')} · ${t('showing', '正在显示')} ${fmtInt(visibleFeatures)}</div>
          </div>
          <div class="xdb-density-legend">${labelRow}</div>
        </div>
        ${scopeToggle}
        ${scope === 'all' ? `
        <div class="xdb-density-selectrow">
          <label for="xdbDensityModule">${t('Module to display', '选择展示模块')}</label>
          <select id="xdbDensityModule" data-density-module-select>${moduleOptions}</select>
          <span class="mono">${selectedModule === 'all' ? t('showing every catalog module', '正在显示全部概念模块') : t('showing selected module only', '仅显示所选模块')}</span>
        </div>
        <div class="xdb-density-controls">
          <button class="chip ${selectedModule === 'all' ? 'solid' : ''}" data-density-module-filter="all">${crossTerm('All modules')}</button>
          ${cleaned.map(module => `<button class="chip ${selectedModule === module.module ? 'solid' : ''}" data-density-module-filter="${esc(module.module)}">${esc(catalogModuleLabel(module.module))} <span class="mono">${fmtInt((module.features || []).length)}</span></button>`).join('')}
        </div>` : ''}
        ${detail ? crossFeatureDensityDetail(detail.module, detail.row, labels || []) : ''}
        ${visible.map(module => crossFeatureDensityModule(module, labels || [])).join('')}
      </div>`;
  }

  function crossFeatureDensityModule(module, labels) {
    const features = module.features || [];
    const maxDensity = Math.max(1, ...features.flatMap(row => (row.values || []).map(v => Number(v.density_per_100_entities)).filter(Number.isFinite)));
    const moduleLabel = catalogModuleLabel(module.module);
    return `
      <section class="xdb-density-module" data-density-module="${esc(module.module)}">
        <div class="xdb-density-module-head">
          <div><h3>${esc(moduleLabel)}</h3><p>${fmtInt(features.length)} ${t('features', '个特征')} · ${fmtInt(module.shared_feature_count || 0)} ${crossTerm('Shared')}</p></div>
          <span class="pill dashed">${esc(module.module)}</span>
        </div>
        <div class="xdb-density-features" style="--xdb-grid-cols:${Math.min(4, Math.max(1, Math.ceil(Math.sqrt(features.length || 1))))}">
          ${features.map(row => crossFeatureDensityFeature(module, row, labels, maxDensity)).join('')}
        </div>
      </section>`;
  }

  function crossFeatureKey(module, row) {
    return `${module.module || 'module'}::${row.feature || row.label || 'feature'}`;
  }

  function findCrossDensityFeature(modules, key) {
    if (!key) return null;
    for (const module of modules || []) {
      for (const row of module.features || []) {
        if (crossFeatureKey(module, row) === key) return { module, row };
      }
    }
    return null;
  }

  function crossFeatureDensityFeature(module, row, labels, maxDensity) {
    const meta = catalogFeatureMeta(row.feature);
    const curve = crossFeatureCurve(row, labels);
    const key = crossFeatureKey(module, row);
    return `
      <button class="xdb-density-feature ${crossDensityFeature === key ? 'selected' : ''}" data-density-feature="${esc(row.feature)}" data-density-feature-key="${esc(key)}" type="button">
        <div class="xdb-density-name"><span>${esc(meta.name)}</span><small>${esc(row.feature)}${meta.unit ? ` · ${esc(meta.unit)}` : ''}</small></div>
        ${curve}
      </button>`;
  }

  function crossFeatureDensityDetail(module, row, labels) {
    const meta = catalogFeatureMeta(row.feature);
    const values = row.values || [];
    const moduleLabel = catalogModuleLabel(module.module);
    return `
      <div class="xdb-density-detail" data-density-detail="${esc(crossFeatureKey(module, row))}">
        <div class="xdb-density-detail-head">
          <div>
            <div class="xdb-density-detail-title">${esc(meta.name)}</div>
            <div class="xdb-density-detail-sub">${esc(moduleLabel)} · ${esc(row.feature)}${meta.unit ? ` · ${esc(meta.unit)}` : ''} · ${crossTerm('aggregate density only')}</div>
          </div>
          <span class="pill dashed">${fmtInt(values.filter(v => v.present).length)} ${t('curves', '条曲线')}</span>
        </div>
        <div class="xdb-density-detail-plot">${crossFeatureCurve(row, labels)}</div>
        <div class="table-wrap table-scroll xdb-density-detail-table">
          <table class="eu-table">
            <thead><tr><th>${crossTerm('Database')}</th><th class="num">${crossTerm('Values')}</th><th class="num">${crossTerm('Range')}</th><th class="num">${crossTerm('Density points')}</th></tr></thead>
            <tbody>
              ${values.map((v, i) => `<tr>
                <td class="key"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${densityPalette(i)};margin-right:6px;"></span>${esc(labels[i] || v.source || `${crossTerm('Database')} ${i + 1}`)}</td>
                <td class="num">${v.present ? fmtInt(v.non_null || v.n || 0) : crossTerm('Missing')}</td>
                <td class="num">${v.present && v.min != null && v.max != null ? `${fmtDensity(v.min)}-${fmtDensity(v.max)}` : '—'}</td>
                <td class="num">${Array.isArray(v.points) ? fmtInt(v.points.length) : (Array.isArray(v.categories) ? fmtInt(v.categories.length) : '—')}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  }

  function crossFeatureCurve(row, labels) {
    const series = (row.values || [])
      .map((value, i) => ({ value, label: labels[i] || value.source || `${t('export', '导出')} ${i + 1}`, color: densityPalette(i) }))
      .filter(item => item.value && item.value.present && Array.isArray(item.value.points) && item.value.points.length >= 2);
    if (!series.length) return crossCategoricalBars(row, labels);
    const xs = series.flatMap(item => item.value.points.map(p => Number(p.x)).filter(Number.isFinite));
    const ys = series.flatMap(item => item.value.points.map(p => Number(p.density)).filter(Number.isFinite));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const maxY = Math.max(0.000001, ...ys);
    const w = 360, h = 72, padX = 8, padTop = 8, padBottom = 14;
    const xScale = x => padX + ((Number(x) - minX) / ((maxX - minX) || 1)) * (w - padX * 2);
    const yScale = y => padTop + (1 - Number(y) / maxY) * (h - padTop - padBottom);
    const paths = series.map(item => {
      const pts = item.value.points.filter(p => Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.density)));
      const line = pts.map((p, idx) => `${idx ? 'L' : 'M'}${xScale(p.x).toFixed(1)},${yScale(p.density).toFixed(1)}`).join(' ');
      const area = `${line} L${xScale(pts[pts.length - 1].x).toFixed(1)},${(h - padBottom).toFixed(1)} L${xScale(pts[0].x).toFixed(1)},${(h - padBottom).toFixed(1)} Z`;
      return `<path class="xdb-density-area" d="${area}" fill="${item.color}"></path><path class="xdb-density-line" d="${line}" stroke="${item.color}"></path>`;
    }).join('');
    const totalN = series.reduce((acc, item) => acc + Number(item.value.non_null || item.value.n || 0), 0);
    const stats = `<span>${fmtInt(series.length)} ${crossTerm('database curves')}</span><span>x ${fmtDensity(minX)}-${fmtDensity(maxX)}</span><span>n=${fmtInt(totalN)}</span>`;
    return `
      <div class="xdb-density-plot">
        <svg class="xdb-density-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
          <line class="xdb-density-axis-line" x1="${padX}" x2="${w - padX}" y1="${h - padBottom}" y2="${h - padBottom}"></line>
          ${paths}
        </svg>
        <div class="xdb-density-stats">${stats}</div>
      </div>`;
  }

  function crossCategoricalBars(row, labels) {
    return `
      <div class="xdb-density-values">
        ${(row.values || []).map((v, i) => {
          const label = labels[i] || v.source || `${t('export', '导出')} ${i + 1}`;
          const cats = (v.categories || []).slice(0, 4);
          return `
            <div class="xdb-density-cat ${v.present ? '' : 'missing'}">
              <span class="xdb-density-src">${esc(label)}</span>
              <span class="xdb-density-cat-bars">${cats.map(cat => `<i title="${esc(cat.label)} ${fmtPct(cat.pct)}" style="width:${Math.max(2, Number(cat.pct) || 0)}%;background:${densityPalette(i)};"></i>`).join('')}</span>
              <span class="xdb-density-num mono">${v.present ? `${fmtInt(v.non_null || 0)} ${crossTerm('values')}` : crossTerm('Missing')}</span>
            </div>`;
        }).join('')}
      </div>`;
  }

  function densityPalette(i) {
    const palette = ['var(--accent)', 'oklch(62% 0.11 255)', 'oklch(64% 0.10 35)', 'oklch(62% 0.10 145)', 'oklch(58% 0.10 300)', 'oklch(60% 0.10 75)'];
    return palette[i % palette.length];
  }

  function fmtDensity(value) {
    if (value == null || !Number.isFinite(Number(value))) return '—';
    const n = Number(value);
    if (n >= 1000) return Math.round(n).toLocaleString();
    if (n >= 100) return n.toFixed(0);
    if (n >= 10) return n.toFixed(1);
    return n.toFixed(2);
  }


  function crossDistributionPanel(title, subtitle, rows) {
    const palette = ['var(--accent)', 'oklch(62% 0.11 255)', 'oklch(64% 0.10 35)', 'oklch(62% 0.10 145)', 'oklch(58% 0.10 300)', 'oklch(60% 0.10 75)'];
    const legend = (rows[0] && rows[0].values || []).map((v, i) => `<span><i style="background:${palette[i % palette.length]};"></i>${esc(v.label)}</span>`).join('');
    return `
      <div class="sec-stack"><div class="lbl">${esc(title)}</div></div>
      <div class="xdb-dist-panel">
        <div class="xdb-dist-top">
          <div class="xdb-dist-sub">${esc(subtitle)}</div>
          <div class="xdb-dist-legend">${legend}</div>
        </div>
        <div class="xdb-dist-rows">
          ${rows.map(row => crossDistributionRow(row, palette)).join('')}
        </div>
      </div>`;
  }

  function crossDistributionRow(row, palette) {
    const nums = (row.values || []).flatMap(v => {
      const out = [];
      if (typeof v.value === 'number') out.push(v.value);
      if (typeof v.low === 'number') out.push(v.low);
      if (typeof v.high === 'number') out.push(v.high);
      return out;
    });
    if (!nums.length) return '';
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const pad = Math.max((max - min) * 0.08, max === min ? Math.max(1, max * 0.05) : 0.01);
    const lo = min - pad;
    const hi = max + pad;
    const pos = value => Math.max(0, Math.min(100, ((value - lo) / (hi - lo)) * 100));
    const fmt = value => {
      if (value == null) return '—';
      const n = Number(value);
      if (Math.abs(n) >= 100) return n.toFixed(0);
      if (Math.abs(n) >= 10) return n.toFixed(1);
      return n.toFixed(2);
    };
    return `
      <div class="xdb-dist-row">
        <div class="xdb-dist-label"><span>${esc(row.label)}</span><small>${esc(row.unit || '')}</small></div>
        <div class="xdb-dist-axis">
          ${(row.values || []).map((v, i) => {
            if (typeof v.value !== 'number') return '';
            const color = palette[i % palette.length];
            const band = typeof v.low === 'number' && typeof v.high === 'number'
              ? `<span class="xdb-dist-band" style="left:${pos(v.low)}%;width:${Math.max(1.2, pos(v.high) - pos(v.low))}%;background:${color};"></span>`
              : '';
            return `${band}<span class="xdb-dist-dot" title="${esc(v.label)} ${fmt(v.value)}" style="left:${pos(v.value)}%;background:${color};"></span>`;
          }).join('')}
        </div>
        <div class="xdb-dist-range mono">${fmt(min)}–${fmt(max)}</div>
      </div>`;
  }

  function crossLoadingState() {
    const p = crossRawProg || {};
    const cur = p.current || 0;
    const tot = p.total || 0;
    const pct = tot ? Math.round((cur / tot) * 100) : 0;
    const sampleMax = p.max_patients || p.maxPatients || (window.EU_DATA === 'real' ? crossRawSampleProfile().maxPatients : null);
    const sampleValues = p.sample_size || p.sampleSize || (window.EU_DATA === 'real' ? crossRawSampleProfile().sampleSize : null);
    const sampleText = sampleMax && sampleValues
      ? ` · ≤${fmtInt(sampleMax)} ${t('entities/db', '实体/库')} · ≤${fmtInt(sampleValues)} ${t('values/feature', '值/特征')}`
      : '';
    const loadingTitle = window.EU_DATA === 'real'
      ? crossProgressMessage('Loading real feature densities from local databases…')
      : crossProgressMessage('Loading seeded frames for selected databases…');
    const progressText = p.message || (window.EU_DATA === 'real'
      ? crossProgressMessage('Starting local raw Cross-DB density job…')
      : crossProgressMessage('Building seeded density frames…'));
    return `<div class="card pad">
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow"><div style="font-weight:600;font-size:12.75px;">${loadingTitle}</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${crossTerm('local-only · nothing uploaded')}${p.phase ? ` · ${esc(crossStatusLabel(p.phase))}` : ''}${sampleText}</div></div>
        ${tot ? `<span class="mono" style="font-size:11px;color:var(--ink-3);">${cur}/${tot}</span>` : ''}
        <button class="btn sm" ${window.EU_DATA === 'real' ? 'data-crossdb-cancel' : 'data-viz-reset'} ${crossRawCancelRequested ? 'disabled' : ''}>${icon('stop', 13)} ${crossRawCancelRequested ? t('Cancel requested', '已请求取消') : t('Cancel', '取消')}</button>
      </div>
      ${tot ? `<div style="height:8px;border-radius:999px;background:var(--surface-2,#eef0f4);overflow:hidden;margin:12px 0 8px;"><div style="height:100%;width:${pct}%;background:var(--accent,#2f7d6b);transition:width .25s;"></div></div>` : '<div class="indet mt-12"></div>'}
      <div style="font-size:12px;color:var(--ink-3);min-height:18px;margin-top:8px;">${esc(crossProgressMessage(progressText))}</div>
      <div class="sk-table mt-16">
        <div class="sk-trow head">${[30,18,18,18,18].map(w => `<div class="sk sk-line sm" style="width:${w}%"></div>`).join('')}</div>
        ${[0,1,2,3,4,5].map(() => `<div class="sk-trow">${[55,40,40,40,40].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}
      </div>
    </div>`;
  }

  S.crossdb = {
    section: 'viz', nav: 'viz', sub: 'crossdb', wide: true,
    crumbs: ['Home', 'Data Visualization', 'Cross-DB Benchmark'],
    get actionHtml() {
      return crossView === 'loaded' || (window.EU_DATA === 'real' && window.EU_CROSSDB_WORKSPACE)
        ? `<button class="btn" data-viz-reset>${icon('sliders', 13)} ${crossTerm('Change selection')}</button><button class="btn" data-crossdb-export>${icon('download', 13)} ${crossTerm('Export JSON')}</button><button class="btn primary" data-run>${icon('refresh', 13)} ${crossTerm('Re-run')}</button>`
        : `<button class="btn primary" data-run ${crossView === 'loading' ? 'aria-disabled="true"' : ''}>${icon('play', 13)} ${crossTerm('Run')}</button>`;
    },
    rail: () => vizRail('crossdb'),
    render() {
      if (crossView === 'loading') {
        return crossHeader() + crossLoadingState();
      }
      if (window.EU_DATA === 'real') {
        const xdb = window.EU_CROSSDB_WORKSPACE;
        if (xdb) {
          return crossHeader() + crossRealLoaded(xdb);
        }
        return crossHeader() + rawCrossdbSetup();
      }
      if (crossView === 'loaded') {
        const xdb = window.EU_CROSSDB_WORKSPACE;
        if (xdb) return crossHeader() + crossRealLoaded(xdb);
        return crossHeader() + `<div class="note warn">
          <div class="ico">${icon('alert', 14)}</div>
          <div class="body"><span class="t">${crossTerm('Demo benchmark was not loaded')}</span> <span class="d" style="display:inline;">— ${t('Run the benchmark again so the backend can build the seeded distribution payload.', '请重新运行对比，让后端生成种子分布载荷。')}</span></div>
        </div>`;
      }
      /* idle — select databases */
      const sel = CROSS_DBS.filter(d => d[1]).length;
      return crossHeader() + `
        <div class="note info">
          <div class="ico">${icon('benchmark', 16)}</div>
          <div class="body"><span class="t">${crossTerm('Select databases to compare')}</span> <span class="d" style="display:inline;">— ${t('Pick two or more standardized ICU sources, then run the benchmark. Each uses an independent seeded feature frame in Demo Mode.', '选择两个或更多标准化 ICU 来源后运行对比。演示模式下，每个数据库使用独立的种子特征帧。')}</span></div>
        </div>
        <div class="sec-stack"><div class="lbl">${crossTerm('Available databases')} · <span id="dbcount">${sel}</span> ${crossTerm('selected')}</div></div>
        <div class="db-grid" id="dbgrid">
          ${CROSS_DBS.map(([n, on], i) => `
            <div class="db-card ${on ? 'sel' : ''}" data-db="${i}">
              <div class="row gap-8" style="min-width:0;">
                <span class="${on ? '' : 'ink-4'}" style="flex:none;color:${on ? 'var(--accent-ink)' : 'var(--ink-4)'};">${icon('db', 15)}</span>
                <div style="min-width:0;">
                  <div style="font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${n}</div>
                  <div class="mono" style="font-size:10.5px;color:var(--ink-4);">${crossTerm('all supported catalog concepts')}</div>
                </div>
              </div>
              <span class="db-mk pill ${on ? 'ok' : 'dashed'}" style="flex:none;height:20px;">${on ? `<span class="dot"></span>${crossTerm('selected')}` : crossTerm('add')}</span>
            </div>`).join('')}
        </div>
        <div class="gate-strip mt-20">
          <span class="pill"><span style="color:var(--ink-3);">${icon('benchmark', 12)}</span> <span id="runhint">${sel} / 6 · ${t('need ≥ 2', '至少需要 2 个')}</span></span>
          <div class="grow"></div>
          <button class="btn primary" data-run ${sel < 2 ? 'aria-disabled="true"' : ''}>${icon('play', 13)} ${crossTerm('Run benchmark')}</button>
        </div>`;
    },
    afterRender(root) {
      bindSourceRegistry(root, 'crossdb');
      root.querySelectorAll('[data-density-scope]').forEach(b => b.addEventListener('click', () => {
        crossDensityScope = b.dataset.densityScope || 'core';
        crossDensityModule = 'all';
        crossDensityFeature = null;
        repaintScreen('crossdb');
      }));
      root.querySelectorAll('[data-density-module-filter]').forEach(b => b.addEventListener('click', () => {
        crossDensityModule = b.dataset.densityModuleFilter || 'all';
        crossDensityFeature = null;
        repaintScreen('crossdb');
      }));
      root.querySelectorAll('[data-density-module-select]').forEach(select => select.addEventListener('change', () => {
        crossDensityModule = select.value || 'all';
        crossDensityFeature = null;
        repaintScreen('crossdb');
      }));
      root.querySelectorAll('[data-density-feature-key]').forEach(b => b.addEventListener('click', () => {
        crossDensityFeature = b.dataset.densityFeatureKey || null;
        repaintScreen('crossdb');
      }));
      root.querySelectorAll('[data-crossdb-sample-mode]').forEach(b => b.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        crossRawSampleMode = b.dataset.crossdbSampleMode || 'quick';
        repaintScreen('crossdb');
      }));
      const grid = root.querySelector('#dbgrid');
      if (grid) grid.addEventListener('click', e => {
        const card = e.target.closest('[data-db]'); if (!card) return;
        const i = +card.dataset.db;
        CROSS_DBS[i][1] = !CROSS_DBS[i][1];
        if (window.EU_DATA === 'real') {
          // Keep the last folder scan: toggling a database changes selection,
          // not whether sibling folders were recognized under the same root.
          vizErr = null;
          repaintScreen('crossdb');
          return;
        }
        const on = CROSS_DBS[i][1];
        card.classList.toggle('sel', on);
        const mk = card.querySelector('.db-mk');
        mk.className = `db-mk pill ${on ? 'ok' : 'dashed'}`;
        mk.innerHTML = on ? `<span class="dot"></span>${crossTerm('selected')}` : crossTerm('add');
        card.querySelector('span[style*="flex:none"]').style.color = on ? 'var(--accent-ink)' : 'var(--ink-4)';
        const sel = CROSS_DBS.filter(d => d[1]).length;
        const cnt = root.querySelector('#dbcount'); if (cnt) cnt.textContent = sel;
        const hint = root.querySelector('#runhint'); if (hint) hint.textContent = `${sel} / 6 · ${t('need ≥ 2', '至少需要 2 个')}`;
        root.querySelectorAll('[data-run]').forEach(b => { if (sel < 2) b.setAttribute('aria-disabled', 'true'); else b.removeAttribute('aria-disabled'); });
      });
      root.querySelectorAll('[data-run]').forEach(b => {
        if (b.dataset.crossdbRunBound === '1') return;
        b.dataset.crossdbRunBound = '1';
        b.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        if (b.getAttribute('aria-disabled') === 'true' || crossView === 'loading') return;
        if (window.EU_DATA === 'real') {
          const rawRootInput = root.querySelector('[data-crossdb-root]');
          let rawRoot = rawRootInput && rawRootInput.value ? rawRootInput.value.trim() : '';
          crossRawRootDraft = rawRoot;
          if (!crossRawScanReadyFor(rawRoot)) {
            scanCrossdbRawRoot(rawRoot);
            return;
          }
          b.setAttribute('aria-disabled', 'true');
          crossView = 'loading'; repaintScreen('crossdb');
          loadRealCrossdb(() => { crossView = 'idle'; repaintScreen('crossdb'); }, { rawRoot });
        } else {
          b.setAttribute('aria-disabled', 'true');
          crossView = 'loading'; repaintScreen('crossdb');
          loadDemoCrossdb(ok => { crossView = ok ? 'loaded' : 'idle'; repaintScreen('crossdb'); });
        }
        });
      });
      root.querySelectorAll('[data-crossdb-root]').forEach(input => {
        if (input.dataset.crossdbRootBound === '1') return;
        input.dataset.crossdbRootBound = '1';
        input.addEventListener('input', () => {
          const next = (input.value || '').trim();
          if (next !== crossRawRootDraft) invalidateCrossRawRootScan();
          crossRawRootDraft = next;
        });
        input.addEventListener('change', () => {
          const next = (input.value || '').trim();
          if (next !== crossRawRootDraft) invalidateCrossRawRootScan();
          crossRawRootDraft = next;
          repaintScreen('crossdb');
        });
      });
      root.querySelectorAll('[data-crossdb-root-browse]').forEach(b => b.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        const input = root.querySelector('[data-crossdb-root]');
        openSourceFolderPicker(
          (input && input.value.trim()) || crossRawRootDraft,
          picked => {
            if (!picked || !input) {
              vizErr = t('Local folder picker API is not ready. Paste a raw ICU data root path instead.', '本地文件夹选择 API 尚未就绪。请改为粘贴原始 ICU 数据根目录路径。');
              repaintScreen('crossdb');
              return;
            }
            crossRawRootDraft = picked;
            vizErr = null;
            input.value = picked;
            input.focus();
            scanCrossdbRawRoot(picked);
          },
          t('Choose local ICU data root', '选择本地 ICU 数据根目录')
        );
      }));
      root.querySelectorAll('[data-crossdb-root-scan]').forEach(b => b.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        const input = root.querySelector('[data-crossdb-root]');
        const rawRoot = input && input.value ? input.value.trim() : crossRawRootDraft;
        scanCrossdbRawRoot(rawRoot);
      }));
      root.querySelectorAll('[data-crossdb-cancel]').forEach(b => b.addEventListener('click', cancelCrossRawJob));
      root.querySelectorAll('[data-crossdb-export]').forEach(b => b.addEventListener('click', () => {
        const payload = window.EU_CROSSDB_WORKSPACE;
        if (!payload) {
          vizErr = t('No Cross-DB payload is loaded yet.', '尚未加载 Cross-DB 载荷。');
          repaintScreen('crossdb');
          return;
        }
        downloadJsonFile('easyicu-crossdb-review.json', {
          exported_at: new Date().toISOString(),
          payload_scope: 'bounded_crossdb_review',
          crossdb_review: payload,
        });
      }));
      root.querySelectorAll('[data-viz-reset]').forEach(b => b.addEventListener('click', () => {
        teardownCrossRawES();
        crossRawJobStarting = false;
        crossRawJobId = null;
        crossRawProg = null;
        crossRawCancelRequested = false;
        invalidateCrossRawRootScan();
        crossView = 'idle'; crossDensityModule = 'all'; crossDensityFeature = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; repaintScreen('crossdb');
      }));
    },
  };
})();
