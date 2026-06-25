/* Screens: Data Visualization — Patient Review, Cohort Statistics, Cross-DB Benchmark */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  function vizRail(active) {
    const real = window.EU_DATA === 'real';
    const xdb = active === 'crossdb' ? window.EU_CROSSDB_WORKSPACE : null;
    const drill = active === 'patient' ? patientDrilldown() : null;
    const cohort = active === 'cohort' ? cohortReview() : null;
    const ws = window.EU_VIZ_WORKSPACE;
    const label = real ? 'Real' : 'Demo';
    const xdbRaw = xdb && xdb.source_type === 'raw_database_root';
    const xdbDemo = xdb && xdb.source_type === 'legacy_simulated_multidb_feature_frames';
    const dataset = xdb ? `${fmtInt(xdb.source_count)} ${xdbRaw || xdbDemo ? 'databases' : 'exports'}` : (drill ? ((drill.source || {}).label || 'Local export') : (cohort ? ((cohort.source || {}).label || 'Local export') : (ws ? ((ws.path || '').split('/').filter(Boolean).slice(-2).join('/') || 'Local export') : (real ? 'No export loaded' : 'Demo · 10 patients'))));
    const cohortLine = xdb ? (xdbRaw ? 'raw feature densities' : (xdbDemo ? 'seeded simulated densities' : 'matched exports required')) : (drill ? `${fmtInt(drill.summary && drill.summary.entities)} entities` : (cohort ? `${fmtInt(cohort.summary && cohort.summary.cohort_size)} entities` : (ws ? `${fmtInt(ws.summary && ws.summary.stays)} stays` : (real ? 'load exported tables' : 'demo defaults'))));
    const variables = xdb ? `${fmtInt((xdb.shared_modules || []).length)} shared modules` : (drill ? `${fmtInt(drill.summary && drill.summary.modules)} modules` : (cohort ? `${fmtInt(cohort.summary && cohort.summary.modules)} modules` : (ws ? `${fmtInt(ws.summary && ws.summary.modules)} modules` : (real ? 'from export manifest' : 'demo defaults'))));
    return `
    <div class="rail-sep"></div>
    <div class="rail-block">
      <div class="rail-head"><span class="t">Current setup</span><span class="pill ${real ? 'ok' : 'demo'}" style="height:20px;"><span class="dot"></span>${label}</span></div>
      <div class="setup-row"><span class="k">Dataset</span><span class="vv">${esc(dataset)}</span></div>
      <div class="setup-row"><span class="k">Cohort</span><span class="vv">${cohortLine}</span></div>
      <div class="setup-row"><span class="k">Variables</span><span class="vv">${variables}</span></div>
      <button class="btn sm block" data-viz-reset style="margin-top:12px;">${icon('sliders', 13)} Edit setup</button>
    </div>`;
  }

  /* view state for the interactive viz screens */
  let patientView = 'idle';   // idle | loading | loaded
  let patientTab = 'tables';
  let crossView = 'idle';     // idle | loading | loaded
  let crossDensityModule = 'all';
  let crossDensityFeature = null;
  let crossRawES = null;
  let crossRawJobId = null;
  let crossRawProg = null;
  let crossRawCancelRequested = false;
  let crossRawJobStarting = false;
  let cohortView = 'loaded';  // loaded | loading
  let cohortPanel = 'groups'; // groups | coverage | snapshot | sofa
  let cohortCompare = 'outcome';
  let cohortSurvivalOutcome = 'hospital_death';
  let cohortSurvivalGroup = 'sepsis';
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
    if (n < 0.001) return '<0.001';
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
    try {
      const v = localStorage.getItem('easyicu_crossdb_data_root') || localStorage.getItem('easyicu_raw_data_root');
      if (v && !/\/easyicu\/exports(\/|$)/.test(v)) return v;
    } catch (e) {}
    const active = registryActivePath();
    if (active && !/\/easyicu\/exports(\/|$)/.test(active)) {
      const parts = active.split('/').filter(Boolean);
      const last = (parts[parts.length - 1] || '').toLowerCase();
      const dbLike = ['mimiciv', 'mimic-iv', 'miiv', 'eicu', 'eicu-crd', 'aumc', 'hirid', 'mimiciii', 'mimic-iii', 'sic', 'sicdb'];
      if (dbLike.some(x => last.includes(x)) && parts.length > 1) {
        const parent = '/' + parts.slice(0, -1).join('/');
        if (!/\/easyicu\/exports(\/|$)/.test(parent)) return parent;
      }
    }
    return '';
  }
  function selectedCrossDbKeys() {
    return CROSS_DBS.filter(d => d[1]).map(d => d[2]);
  }
  function teardownCrossRawES() {
    if (crossRawES) {
      try { crossRawES.close(); } catch (e) {}
    }
    crossRawES = null;
  }
  function cancelCrossRawJob() {
    if (!crossRawJobId || crossRawCancelRequested || !window.EU_API || !window.EU_API.postJSON) return;
    crossRawCancelRequested = true;
    crossRawProg = {
      phase: 'cancel',
      message: 'Cancel requested. The current database read may finish before the job stops.',
    };
    repaintScreen('crossdb');
    window.EU_API.postJSON('/api/jobs/' + crossRawJobId + '/cancel', { reason: 'user_requested' })
      .catch(err => { vizErr = String(err && err.message || err); repaintScreen('crossdb'); });
  }
  function patientDrilldown() {
    return window.EU_PATIENT_DRILLDOWN || null;
  }
  function cohortReview() {
    return window.EU_COHORT_REVIEW || null;
  }
  function patientWorkspaceFromDrilldown(payload) {
    const s = payload && payload.summary ? payload.summary : {};
    return {
      ok: true,
      mode: 'real',
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
    if (sum.stays != null) parts.push(`${fmtInt(sum.stays)} stays`);
    if (sum.entities != null && sum.stays == null) parts.push(`${fmtInt(sum.entities)} entities`);
    if (sum.modules != null) parts.push(`${fmtInt(sum.modules)} modules`);
    if (sum.total_rows != null) parts.push(`${fmtInt(sum.total_rows)} rows`);
    return parts.join(' · ') || 'export folder';
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
    const sources = registrySources();
    const active = defaultExportPath();
    const selected = new Set(defaultCrossdbPaths());
    const title = multi ? 'Local export sources' : 'Current local export';
    const empty = multi
      ? 'No registered exports yet. Add two EasyICU export folders below.'
      : 'No registered export yet. Add an EasyICU export folder below.';
    return `
      <div class="src-registry">
        <div class="src-head">
          <div><div class="eyebrow">${title}</div><div class="src-sub">${multi ? 'Choose at least two exports for Cross-DB preview.' : 'This active export is shared by Patient, Cohort, Agent, and Copilot.'}</div></div>
          <button class="btn sm ghost" data-src-refresh>${icon('refresh', 12)} Refresh</button>
        </div>
        <div class="src-list">
          ${sources.length ? sources.map(s => {
            const on = multi ? selected.has(s.path) : s.path === active;
            const attr = multi ? `data-src-cross="${esc(s.path)}"` : `data-src-active="${esc(s.path)}"`;
            const label = s.label || s.database || 'local';
            return `
              <div class="src-row ${on ? 'on' : ''}" ${attr}>
                <span class="src-ico">${icon(multi && on ? 'check' : 'folder', 14, multi && on ? 2.6 : undefined)}</span>
                <span class="src-body"><span class="src-name">${esc(label)}</span><span class="src-meta">${esc(sourceLine(s))}</span><span class="src-path mono">${esc(s.path)}</span></span>
                <span class="pill ${on ? 'ok' : 'dashed'}" style="height:20px;">${on ? (multi ? 'selected' : 'active') : (multi ? 'add' : 'use')}</span>
                <span class="src-actions">
                  <button class="btn icon sm ghost" data-src-action data-src-rename="${esc(s.path)}" data-src-label="${esc(label)}" title="Rename source">${icon('edit', 12)}</button>
                  <button class="btn icon sm ghost" data-src-action data-src-remove="${esc(s.path)}" title="Remove registration only; files stay on disk">${icon('close', 12)}</button>
                </span>
              </div>`;
          }).join('') : `<div class="empty compact"><div class="glyph">${icon('folder', 20)}</div><div class="t">${empty}</div></div>`}
        </div>
        <div class="path-field editable src-add">
          <span class="pf-ico">${icon('folder', 14)}</span>
          <input class="pf-input" data-src-path-input type="text" spellcheck="false" autocomplete="off" placeholder="${esc('Paste a local EasyICU export folder')}" aria-label="EasyICU export path" />
          <button class="btn sm primary" data-src-add>${icon('plus', 12)} Add</button>
        </div>
      </div>`;
  }
  function bindSourceRegistry(root, screenId) {
    root.querySelectorAll('[data-src-active]').forEach(b => b.addEventListener('click', e => {
      if (e.target.closest('[data-src-action]')) return;
      const path = b.dataset.srcActive;
      if (!path || !(window.EU_API && window.EU_API.saveWorkspaceRegistry)) return;
      window.EU_API.saveWorkspaceRegistry({ active_path: path }).then(() => {
        try { localStorage.setItem('easyicu_last_export_dir', path); } catch (e) {}
        window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; window.EU_COHORT_REVIEW = null; window.EU_STALE = true;
        patientView = 'idle'; crossView = 'idle'; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-cross]').forEach(b => b.addEventListener('click', e => {
      if (e.target.closest('[data-src-action]')) return;
      const path = b.dataset.srcCross;
      const cur = defaultCrossdbPaths().filter(Boolean);
      const next = cur.includes(path) ? cur.filter(p => p !== path) : cur.concat([path]);
      if (!(window.EU_API && window.EU_API.saveWorkspaceRegistry)) return;
      window.EU_API.saveWorkspaceRegistry({ crossdb_paths: next }).then(() => {
        window.EU_CROSSDB_WORKSPACE = null; window.EU_COHORT_REVIEW = null; crossView = 'idle'; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-add]').forEach(b => b.addEventListener('click', () => {
      const input = root.querySelector('[data-src-path-input]');
      const path = input && input.value ? input.value.trim() : '';
      if (!path || !(window.EU_API && window.EU_API.registerWorkspaceSource)) return;
      const multi = !!root.querySelector('[data-src-cross]');
      window.EU_API.registerWorkspaceSource(path, { active: !multi, crossdb: true }).then(() => {
        vizErr = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; window.EU_COHORT_REVIEW = null; crossView = 'idle'; patientView = 'idle'; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-rename]').forEach(b => b.addEventListener('click', e => {
      e.preventDefault(); e.stopPropagation();
      const path = b.dataset.srcRename;
      const current = b.dataset.srcLabel || '';
      if (!path || !(window.EU_API && window.EU_API.renameWorkspaceSource)) return;
      const next = window.prompt('Source label', current);
      if (next === null) return;
      window.EU_API.renameWorkspaceSource(path, next).then(() => {
        vizErr = null; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-remove]').forEach(b => b.addEventListener('click', e => {
      e.preventDefault(); e.stopPropagation();
      const path = b.dataset.srcRemove;
      if (!path || !(window.EU_API && window.EU_API.removeWorkspaceSource)) return;
      if (!window.confirm('Remove this source from the registry? Export files stay on disk.')) return;
      window.EU_API.removeWorkspaceSource(path).then(() => {
        vizErr = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; window.EU_COHORT_REVIEW = null; crossView = 'idle'; patientView = 'idle'; repaintScreen(screenId);
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
  function loadRealCohort(done) {
    if (!(window.EU_API && window.EU_API.loadCohortReviewSummary)) {
      vizErr = 'Cohort Review API is unavailable.';
      done && done(false);
      return;
    }
    const active = registryActivePath();
    const body = {};
    if (active) body.source_path = active;
    window.EU_API.loadCohortReviewSummary(body).then(payload => {
      window.EU_COHORT_REVIEW = payload;
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
    const rawRoot = requestedRawRoot || (rawRootInput && rawRootInput.value ? rawRootInput.value.trim() : defaultRawCrossdbRoot());
    const rawDatabases = selectedCrossDbKeys();
    if (rawRoot && rawDatabases.length >= 2 && window.EU_API && window.EU_API.startCrossdbRawDistributionJob && window.EventSource) {
      if (crossRawJobStarting) return;
      crossRawJobStarting = true;
      try { localStorage.setItem('easyicu_crossdb_data_root', rawRoot); } catch (e) {}
      window.EU_API.startCrossdbRawDistributionJob({
        data_root: rawRoot,
        databases: rawDatabases,
        feature_scope: 'all_catalog',
        coverage_min: 2,
        max_patients: 300,
        sample_size: 1500,
        max_features: 90,
      }).then(r => {
        crossRawJobId = r.job_id;
        crossRawProg = { phase: 'queued', message: 'Queued local raw Cross-DB density job.' };
        crossRawES = new EventSource('/api/jobs/' + r.job_id + '/events');
        crossRawES.onmessage = ev => {
          let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
          if (m.type === 'progress') {
            crossRawProg = m;
          } else if (m.type === 'cancel_requested') {
            crossRawCancelRequested = true;
            crossRawProg = {
              phase: 'cancel',
              message: 'Cancel requested. The current database read may finish before the job stops.',
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
              vizErr = 'Raw Cross-DB density job cancelled before completion.';
              done && done(false);
            } else {
              vizErr = m.error || 'Raw Cross-DB density job failed.';
              done && done(false);
            }
          }
          repaintScreen('crossdb');
        };
        crossRawES.onerror = () => {
          crossRawJobStarting = false;
          if (!window.EU_CROSSDB_WORKSPACE && !vizErr) vizErr = 'Lost connection to the raw Cross-DB density job.';
          teardownCrossRawES();
          done && done(false);
          repaintScreen('crossdb');
        };
        repaintScreen('crossdb');
      }).catch(err => {
        crossRawJobStarting = false;
        vizErr = String(err && err.message || err);
        done && done(false);
      });
      return;
    }
    loadRealWorkspace(done);
  }
  function loadDemoCrossdb(done) {
    window.EU_CROSSDB_WORKSPACE = null;
    window.EU_VIZ_WORKSPACE = null;
    crossDensityModule = 'all';
    crossDensityFeature = null;
    const databases = selectedCrossDbKeys();
    if (databases.length < 2) {
      vizErr = 'Select at least two demo databases.';
      done && done(false);
      return;
    }
    if (!window.EU_API || !window.EU_API.loadCrossdbDemoDistribution) {
      vizErr = 'Demo distribution endpoint is unavailable.';
      done && done(false);
      return;
    }
    window.EU_API.loadCrossdbDemoDistribution({
      databases,
      feature_scope: 'legacy_demo_supported_features',
      records_per_feature: 192,
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
    cohortView = 'loaded';
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
      .map(t => ({ value: Number(t && t.value), label: String((t && t.label) || 'threshold') }))
      .filter(t => Number.isFinite(t.value));
    const rawMin = Math.min(...seriesVals, ...thresholdRows.map(t => t.value));
    const rawMax = Math.max(...seriesVals, ...thresholdRows.map(t => t.value));
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
    const thresholds = thresholdRows.slice(0, 3).map(t => {
      const y = yFor(t.value);
      if (y < top - 1 || y > top + innerH + 1) return '';
      return `<line x1="${left}" y1="${y.toFixed(1)}" x2="${(left + innerW).toFixed(1)}" y2="${y.toFixed(1)}" stroke="#d97706" stroke-width="1" stroke-dasharray="3 3" opacity=".72"><title>${esc(t.label)} ${fmtNum(t.value, 1)}${esc(unit)}</title></line>`;
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
        <text x="${left}" y="${h - 6}" fill="#64748b" font-size="9">t0</text>
        <text x="${Math.max(left + 28, left + innerW - 46).toFixed(1)}" y="${h - 6}" fill="#64748b" font-size="9">t${seriesVals.length - 1}</text>
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

  /* ---------------- PATIENT REVIEW ---------------- */
  function patientTabs() {
    const tabs = [['tables', 'Data Tables', 'rows'], ['series', 'Time Series', 'viz'], ['patient', 'Patient Overview', 'patient'], ['quality', 'Data Quality', 'shield']];
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
      const rows = [
        ['Entities', fmtInt(s.entities), 'cohort denominator from active export'],
        ['Mean age', fmtNum(s.mean_age, 1), 'demographics aggregate'],
        ['Female', fmtPct(s.female_pct), 'demographics aggregate'],
        ['Mortality', fmtPct(s.mortality), 'outcome aggregate'],
        ['Median SOFA-2', fmtNum(s.median_sofa2, 1), 'score aggregate'],
        ['Sepsis-3 positive', fmtPct(s.sepsis_pct), 'event aggregate'],
      ];
      const reviewModules = (dt.modules && dt.modules.length ? dt.modules : modules).slice(0, 12);
      const activeModule = reviewModules.find(m => m.module === picker.default_module) || reviewModules[0] || {};
      const previewFeatures = activeModule.preview_features || [];
      return `
      <div class="st-stats mt-16">
        ${[
          ['Entities', fmtInt(loaded.entities != null ? loaded.entities : s.entities), 'ok'],
          ['Review features', fmtInt(loaded.review_features), 'accent'],
          ['Modules', fmtInt(loaded.module_count), 'accent'],
          ['Observed features', fmtInt(loaded.observed_features), 'accent'],
        ].map(([l, v, c]) => `<div class="stat ${c}"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('')}
      </div>
      <div class="note ok mt-16">
        <div class="ico">${icon('rows', 16)}</div>
        <div class="body"><span class="t">Review workspace summary</span> <span class="d" style="display:inline;">— migrated from the old Data Tables page: module-first review, feature counts, source scope and optional detail gate are computed from the active export.</span></div>
      </div>
      <div class="table-wrap table-scroll mt-16">
        <table class="eu-table">
          <thead><tr><th>Aggregate</th><th class="num">Value</th><th>Basis</th></tr></thead>
          <tbody>
            ${rows.map(r => `<tr><td class="key">${esc(r[0])}</td><td class="num">${esc(r[1])}</td><td>${esc(r[2])}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      ${reviewModules.length ? `
      <div class="split-320 mt-16" style="grid-template-columns:1fr 310px;">
        <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Module table overview</th><th class="num">Features</th><th class="num">Rows</th><th class="num">Entities</th><th class="num">Coverage</th><th>Shape</th></tr></thead>
          <tbody>
            ${reviewModules.map(m => `<tr><td class="key">${esc(m.label || m.module)}</td><td class="num">${fmtInt(m.observed_features != null ? m.observed_features : (m.review_features != null ? m.review_features : m.feature_count))}</td><td class="num">${fmtInt(m.rows)}</td><td class="num">${fmtInt(m.entities)}</td><td class="num">${fmtPct(m.coverage_pct)}</td><td>${esc(m.shape || (m.time_indexed ? 'time_indexed' : 'static'))} · ${fmtInt(m.dynamic_features || 0)} dynamic</td></tr>`).join('')}
          </tbody>
        </table>
        </div>
        <div class="card pad">
          <div class="eyebrow">Module at a glance</div>
          <div style="font-weight:600;font-size:15px;margin-top:6px;">${esc(activeModule.label || activeModule.module || 'Selected module')}</div>
          <div class="col gap-6 mt-12" style="font-size:12.5px;">
            <div class="setup-row"><span class="k">Review features</span><span class="vv">${fmtInt(activeModule.review_features != null ? activeModule.review_features : activeModule.feature_count)}</span></div>
            <div class="setup-row"><span class="k">Share</span><span class="vv">${fmtPct(activeModule.share_pct)}</span></div>
            <div class="setup-row"><span class="k">Coverage</span><span class="vv">${fmtPct(activeModule.coverage_pct)}</span></div>
            <div class="setup-row"><span class="k">Status</span><span class="vv">${esc(activeModule.status || 'ready')}</span></div>
          </div>
          <div class="row wrap gap-6 mt-12">
            ${previewFeatures.slice(0, 6).map(f => `<span class="chip">${esc(f.feature || f.name)}${f.unit ? ` · ${esc(f.unit)}` : ''}</span>`).join('') || '<span class="chip">metadata only</span>'}
          </div>
        </div>
      </div>` : ''}
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

  function ptSeries() {
    const drill = patientDrilldown();
    const review = drill ? (drill.trajectory_review || {}) : {};
    const signals = drill && drill.selected ? (drill.selected.signals || []) : [];
    const lanes = Array.isArray(review.lanes) ? review.lanes : (drill && Array.isArray(drill.time_lanes) ? drill.time_lanes : []);
    const readyLanes = lanes.filter(lane => (lane.signals || []).length);
    if (drill && readyLanes.length) {
      const palette = ['var(--accent)', 'var(--accent)', 'var(--ok)', 'var(--warn)', '#64748b'];
      return `
      <div class="note ok mt-16">
        <div class="ico">${icon('viz', 16)}</div>
        <div class="body"><span class="t">Trajectory ledger</span> <span class="d" style="display:inline;">— old Time Series logic restored as clinical lanes, single-entity review, and aggregate multi-entity comparison without browser row traces.</span></div>
      </div>
      <div class="grid cards-4 mt-16">
        ${(review.contract || []).map(row => `
          <div class="stat ${row.status === 'ready' ? 'ok' : row.status === 'warn' ? 'warn' : 'accent'}">
            <div class="label">${esc(row.index || '')} · ${esc(row.label || '')}</div>
            <div class="val" style="font-size:13px;">${esc(row.detail || '')}</div>
          </div>`).join('')}
      </div>
      <div class="row wrap gap-6 mt-16">
        ${(review.modes || []).map(mode => `<span class="chip ${mode.status === 'ready' ? 'solid' : ''}">${esc(mode.label || mode.id)} · ${esc(mode.status || 'available')}</span>`).join('')}
      </div>
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">Clinical lanes</div>
        <div class="grid cards-2">
          ${readyLanes.map(lane => `
            <div class="mini-chart">
              <div class="mc-head">
                <div><div style="font-weight:600;font-size:13px;">${esc(lane.label || lane.lane)}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">${fmtInt(lane.signal_count)} signals · ${(drill.selected || {}).label || 'selected entity'}</div></div>
                <span class="pill ok" style="height:22px;">real</span>
              </div>
              <div class="col gap-12 mt-8">
                ${(lane.signals || []).slice(0, 4).map((s, i) => `
                  <div>
                    <div class="row" style="justify-content:space-between;font-size:11px;"><span class="mono">${esc(s.feature)}</span><span class="mono" style="color:var(--ink-4);">${fmtNum(s.current, 1)} ${esc(s.unit || '')}</span></div>
                    <div style="height:86px;">${axisSpark(s.values || [], 440, 86, palette[i % palette.length], { unit: s.unit || '', label: s.feature || s.name || 'signal', thresholds: s.thresholds || [] })}</div>
                  </div>`).join('')}
              </div>
            </div>`).join('')}
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Signal arrays are capped at ${fmtInt((drill.privacy || {}).max_points_per_signal)} points for browser review; lane membership follows the EasyICU clinical concept catalog.</p>`;
    }
    if (drill && signals.length) {
      const palette = ['var(--accent)', 'var(--accent)', 'var(--ok)', 'var(--warn)'];
      return `
      <div class="cols-2 mt-16">
        ${signals.map((s, i) => `
          <div class="mini-chart">
            <div class="mc-head">
              <div><div style="font-weight:600;font-size:13px;">${esc(s.name)}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc((drill.selected || {}).label || 'Selected entity')} · capped local signal</div></div>
              <div style="text-align:right;"><div class="mono" style="font-size:18px;font-weight:500;">${fmtNum(s.current, 1)}</div><div class="mono" style="font-size:10px;color:var(--ink-4);">${esc(s.unit || '')}</div></div>
            </div>
            <div style="height:86px;">${axisSpark(s.values || [], 520, 86, palette[i % palette.length], { unit: s.unit || '', label: s.name || s.key || 'signal', thresholds: s.thresholds || [] })}</div>
          </div>`).join('')}
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Signal arrays are capped at ${fmtInt((drill.privacy || {}).max_points_per_signal)} points for browser review.</p>`;
    }
    if (drill) {
      return `<div class="empty mt-16"><div class="glyph">${icon('viz', 22)}</div><div class="t">No bounded signals in this export</div><div class="d">The active export did not include supported vitals columns for the selected entity.</div></div>`;
    }
    const ws = window.EU_VIZ_WORKSPACE;
    if (ws && Array.isArray(ws.series) && ws.series.length) {
      const palette = ['var(--accent)', 'var(--accent)', 'var(--ok)', 'var(--warn)'];
      return `
      <div class="cols-2 mt-16">
        ${ws.series.map((s, i) => `
          <div class="mini-chart">
            <div class="mc-head">
              <div><div style="font-weight:600;font-size:13px;">${esc(s.name)}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">first available window · local export</div></div>
              <div style="text-align:right;"><div class="mono" style="font-size:18px;font-weight:500;">${fmtNum(s.current, 1)}</div><div class="mono" style="font-size:10px;color:var(--ink-4);">${esc(s.unit || '')}</div></div>
            </div>
            <div style="height:86px;">${axisSpark(s.values || [], 520, 86, palette[i % palette.length], { unit: s.unit || '', label: s.name || s.key || 'signal' })}</div>
          </div>`).join('')}
      </div>`;
    }
    if (ws) {
      return `<div class="empty mt-16"><div class="glyph">${icon('viz', 22)}</div><div class="t">No time-series module in this export</div><div class="d">Run extraction with vitals selected to populate trend panels.</div></div>`;
    }
    const series = [
      ['Heart rate', 'bpm', '92', [88,90,95,101,98,94,92,96,99,93], 'var(--accent)'],
      ['MAP', 'mmHg', '82', [78,80,76,70,74,79,82,85,83,81], 'var(--accent)'],
      ['SpO₂', '%', '96', [98,97,95,93,94,96,96,97,95,96], 'var(--ok)'],
      ['Temp', '°C', '37.0', [36.8,37.0,37.4,37.6,37.2,37.0,36.9,37.1,37.3,37.0], 'var(--warn)'],
    ];
    return `
      <div class="cols-2 mt-16">
        ${series.map(([n, u, cur, vals, col]) => `
          <div class="mini-chart">
            <div class="mc-head">
              <div><div style="font-weight:600;font-size:13px;">${n}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">first 24h · hourly</div></div>
              <div style="text-align:right;"><div class="mono" style="font-size:18px;font-weight:500;">${cur}</div><div class="mono" style="font-size:10px;color:var(--ink-4);">${u}</div></div>
            </div>
            <div style="height:86px;">${axisSpark(vals, 520, 86, col, { unit: u || '', label: n || 'signal' })}</div>
          </div>`).join('')}
      </div>`;
  }

  function ptPatient() {
    const drill = patientDrilldown();
    if (drill && drill.selected) {
      const selected = drill.selected || {};
      const overview = drill.patient_overview || {};
      const dashboard = overview.dashboard || {};
      const category = overview.category_view || {};
      const dataTable = overview.data_table || {};
      const demo = selected.demographics || {};
      const scores = selected.scores || {};
      const outcomes = selected.outcomes || {};
      const signals = selected.signals || [];
      const entities = drill.entities || [];
      const summaryCards = dashboard.summary_cards || [];
      const sections = category.sections || [];
      return `
      <div class="row wrap gap-6 mt-16">
        <span class="eyebrow" style="align-self:center;margin-right:4px;">Case navigator</span>
        ${entities.map(item => `<button type="button" class="chip ${item.ref === selected.ref ? 'solid' : ''}" data-patient-entity="${esc(item.ref)}" style="${item.ref === selected.ref ? 'border-color:var(--ink);color:var(--ink);' : ''}">${esc(item.label || item.ref)}</button>`).join('')}
      </div>
      ${summaryCards.length ? `
      <div class="st-stats mt-16">
        ${summaryCards.map(card => `<div class="stat ${card.tone || 'accent'}"><div class="label">${esc(card.label)}</div><div class="val">${esc(card.value == null ? '—' : card.value)}</div></div>`).join('')}
      </div>` : ''}
      <div class="split-320 mt-16" style="grid-template-columns:300px 1fr;">
        <div class="card pad">
          <div class="eyebrow">Dashboard</div>
          <div style="font-weight:600;font-size:15px;margin-top:6px;">${esc(selected.label || 'Selected entity')}</div>
          <div class="col gap-6 mt-12" style="font-size:12.5px;">
            <div class="setup-row"><span class="k">Age · sex</span><span class="vv">${fmtNum(demo.age, 0)} · ${esc(demo.sex || '—')}</span></div>
            <div class="setup-row"><span class="k">SOFA-2 (max)</span><span class="vv">${fmtNum(scores.sofa2_max, 1)}</span></div>
            <div class="setup-row"><span class="k">Sepsis-3</span><span class="vv">${scores.sepsis3_sofa2 == null ? '—' : (scores.sepsis3_sofa2 ? 'Positive' : 'Negative')}</span></div>
            <div class="setup-row"><span class="k">ICU LOS</span><span class="vv">${fmtNum(outcomes.icu_los_days, 1)} d</span></div>
            <div class="setup-row"><span class="k">Outcome</span><span class="vv">${esc(outcomes.status || 'Unknown')}</span></div>
          </div>
        </div>
        <div class="mini-chart">
          <div class="mc-head"><div style="font-weight:600;font-size:13px;">Selected entity trend tiles · ${esc(selected.label || 'selected entity')}</div><span class="mono" style="font-size:10.5px;color:var(--ink-4);">local export · bounded</span></div>
          <div class="col gap-12 mt-8">
            ${signals.slice(0, 4).map((s, i) => `
              <div>
                <div class="row" style="justify-content:space-between;font-size:11px;"><span class="mono">${esc(s.key || s.name)}</span><span class="mono" style="color:var(--ink-4);">${fmtNum(s.current, 1)} ${esc(s.unit || '')}</span></div>
                <div style="height:78px;">${axisSpark(s.values || [], 440, 78, ['var(--accent)', 'var(--accent)', 'var(--ok)', 'var(--warn)'][i % 4], { unit: s.unit || '', label: s.name || s.key || 'signal', thresholds: s.thresholds || [] })}</div>
              </div>`).join('') || '<div style="font-size:12px;color:var(--ink-4);">No vitals trend available in this export.</div>'}
          </div>
        </div>
      </div>
      ${sections.length ? `
      <div class="card pad mt-16">
        <div class="eyebrow">Category View</div>
        <div class="grid cards-2 mt-12">
          ${sections.map(section => `
            <div class="mini-chart">
              <div class="mc-head">
                <div><div style="font-weight:600;font-size:13px;">${esc(section.title || section.id)}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">${fmtInt(section.available_count || 0)} available signals</div></div>
              </div>
              <div class="col gap-8 mt-8">
                ${(section.cards || []).slice(0, 6).map((card, i) => `
                  <div>
                    <div class="row" style="justify-content:space-between;font-size:11px;"><span class="mono">${esc(card.feature || card.label)}</span><span class="mono" style="color:var(--ink-4);">${fmtNum(card.current, 1)} ${esc(card.unit || '')}${card.delta == null ? '' : ` · Δ ${fmtNum(card.delta, 1)}`}</span></div>
                    <div style="height:70px;">${axisSpark(card.values || [], 420, 70, ['var(--accent)', 'var(--ok)', 'var(--warn)', '#64748b'][i % 4], { unit: card.unit || '', label: card.label || card.feature || 'signal', thresholds: card.thresholds || [] })}</div>
                  </div>`).join('') || '<div style="font-size:12px;color:var(--ink-4);">No signals in this category for the selected entity.</div>'}
              </div>
            </div>`).join('')}
        </div>
      </div>` : ''}
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">Pseudonymous drilldown</span> <span class="d" style="display:inline;">— entity refs are one-way browser tokens for the active local export; direct clinical identifiers are not returned.</span></div>
      </div>
      ${dataTable.row_preview === 'blocked' ? `
      <div class="note warn mt-12">
        <div class="ico">${icon('lock', 14)}</div>
        <div class="body"><span class="t">Data Table preview blocked</span> <span class="d" style="display:inline;">— ${esc(dataTable.reason || 'Native Patient Overview keeps source rows out of the browser payload.')}</span></div>
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
      return `
      <div class="note ok mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">Quality dashboard</span> <span class="d" style="display:inline;">— old quality review semantics restored: missingness, physiologic range, temporal integrity, module coverage and action-oriented issues.</span></div>
      </div>
      ${(review.summary_cards || []).length ? `
      <div class="st-stats mt-16">
        ${(review.summary_cards || []).map(card => `<div class="stat ${card.tone || 'accent'}"><div class="label">${esc(card.label)}</div><div class="val">${card.unit === '%' ? fmtPct(card.value) : esc(card.value == null ? '—' : fmtInt(card.value))}</div></div>`).join('')}
      </div>` : (qsum.concept_count != null ? `
      <div class="st-stats mt-16">
        ${[
          ['QC concepts', fmtInt(qsum.concept_count), 'ok'],
          ['Records', fmtInt(qsum.total_records), 'accent'],
          ['Missing', fmtPct(qsum.weighted_missing_pct), 'accent'],
          ['Out-of-physio', fmtPct(qsum.weighted_out_of_physio_pct), qsum.weighted_out_of_physio_pct > 0 ? 'warn' : 'ok'],
        ].map(([l, v, c]) => `<div class="stat ${c}"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('')}
      </div>` : '')}
      ${(review.contract || []).length ? `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:8px;">QC ledger</div>
        <div class="grid cards-4">
          ${(review.contract || []).map(row => `
            <div class="stat ${row.status === 'ok' || row.status === 'ready' ? 'ok' : row.status === 'warn' ? 'warn' : row.status === 'bad' ? 'bad' : 'accent'}">
              <div class="label">${esc(row.index || '')} · ${esc(row.label || '')}</div>
              <div class="val" style="font-size:13px;">${esc(row.detail || '')}</div>
            </div>`).join('')}
        </div>
      </div>` : ''}
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">Per-module entity coverage</div>
        ${drill.quality.map(q => `
          <div class="qrow"><span>${esc(q.module)}</span><div class="qbar ${q.quality_status === 'ok' ? '' : q.quality_status}"><span style="width:${q.coverage_pct == null ? 0 : Math.max(0, Math.min(100, q.coverage_pct))}%"></span></div><span class="qv">${q.coverage_pct == null ? fmtInt(q.rows) : fmtPct(q.coverage_pct)}</span></div>`).join('')}
      </div>
      ${(review.panels || []).length ? `
      <div class="grid cards-3 mt-16">
        ${(review.panels || []).map(panel => `
          <div class="card pad">
            <div class="eyebrow">${esc(panel.label || panel.id)}</div>
            <div class="col gap-8 mt-12">
              ${(panel.rows || []).slice(0, 5).map(row => `
                <div class="setup-row"><span class="k">${esc(row.feature || row.name)}</span><span class="vv">${fmtPct(row.value)} · ${fmtInt(row.records)} rec</span></div>`).join('') || '<div style="font-size:12px;color:var(--ink-4);">No flags in this panel.</div>'}
            </div>
          </div>`).join('')}
      </div>` : ''}
      ${topIssues.length ? `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">Top concept quality issues</div>
        <div class="table-wrap table-scroll">
          <table class="eu-table">
            <thead><tr><th>Concept</th><th>Module</th><th class="num">Records</th><th class="num">Missing</th><th class="num">Outlier</th><th class="num">Duplicate TS</th></tr></thead>
            <tbody>
              ${topIssues.map(row => `<tr><td class="key">${esc(row.feature)}</td><td>${esc(row.module)}</td><td class="num">${fmtInt(row.records)}</td><td class="num">${fmtPct(row.missing_pct)}</td><td class="num">${fmtPct(row.out_of_physio_pct)}</td><td class="num">${fmtPct(row.duplicate_time_pct)}</td></tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>` : ''}
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><div class="t">Local export bounded review</div><div class="d">Coverage, missingness, physiologic-range flags and duplicate timestamp rates are computed from bounded local columns. Formal claims remain locked to the evidence-bound agent path.</div></div>
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
  function bindPatientEntitySelection(root) {
    root.querySelectorAll('[data-patient-entity]').forEach(b => b.addEventListener('click', () => {
      const ref = b.dataset.patientEntity;
      if (!ref || !(window.EU_DATA === 'real')) return;
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
        ? `<button class="btn" data-viz-reset>${icon('sliders', 13)} Edit setup</button><button class="btn primary" data-gen>${icon('refresh', 13)} Re-run</button>`
        : `<button class="btn primary" data-gen ${patientView === 'loading' ? 'aria-disabled="true"' : ''}>${icon('play', 13)} Render</button>`;
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
        const readyTitle = drill ? 'Local export patient drilldown ready' : (ws ? 'Local export workspace ready' : 'Demo review workspace ready');
        const readyStats = s
          ? `${fmtInt(s.entities != null ? s.entities : s.stays)} ${drill ? 'entities' : 'stays'} · ${fmtInt(s.modules)} modules · ${fmtInt(s.total_rows)} rows`
          : '10 stays · 19 modules · 0 errors';
        const demoLoadedNote = (!drill && !ws) ? `
        <div class="note warn mt-12">
          <div class="ico">${icon('beaker', 16)}</div>
          <div class="body">
            <div class="t">Seeded demo workspace</div>
            <div class="d">This tab is showing seeded demo rows. The real Patient Review backend appears after switching to Real and loading a local export; it shows module table overview, clinical lanes, and concept-level quality metrics.</div>
          </div>
          <button class="btn sm" data-patient-use-real>${icon('db', 13)} Use real export</button>
        </div>` : '';
        return `
        <div class="loaded-bar">
          <span class="pill ok"><span class="dot"></span>Loaded</span>
          <div class="grow"><span style="font-weight:600;font-size:13px;">${readyTitle}</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${readyStats}</span></div>
          <button class="btn sm" data-viz-reset>${icon('sliders', 13)} Edit setup</button>
          <button class="btn sm" data-patient-export>${icon('download', 13)} Export</button>
        </div>
        ${demoLoadedNote}
        <div class="mt-16">${patientTabs()}</div>
        <div id="ptbody">${patientTabBody()}</div>
        <div class="nextbar accent mt-16">
          <div class="nb-ico">${icon('arrow', 16)}</div>
          <div class="grow"><div class="nb-t">${t('Reviewed the data — what\u2019s next?', '\u6570\u636e\u5df2\u5ba1\u9605 \u2014\u2014 \u4e0b\u4e00\u6b65\uff1f')}</div><div class="nb-d">${t('Compare groups in Cohort Statistics, or assemble an auditable analysis and gated draft in Agent Projects.', '\u5728\u300c\u961f\u5217\u7edf\u8ba1\u300d\u505a\u7ec4\u95f4\u5bf9\u6bd4\uff0c\u6216\u5728\u300c\u7814\u7a76\u9879\u76ee\u300d\u7ec4\u88c5\u53ef\u5ba1\u8ba1\u5206\u6790\u4e0e\u53d7\u95f8\u8349\u7a3f\u3002')}</div></div>
          <button class="btn" data-nav="cohort">${icon('cohort', 13)} ${t('Cohort Statistics', '\u961f\u5217\u7edf\u8ba1')}</button>
          <button class="btn primary" data-nav="agent">${icon('agent', 13)} ${t('Analyze in Agent', '\u8fdb\u5165\u7814\u7a76\u9879\u76ee')}</button>
        </div>`;
      }
      /* idle */
      return `
      <div class="card pad">
        <div class="panel-head">
          <div>
            <div class="eyebrow">Quick visualization</div>
            <div class="panel-title" style="margin-top:4px;font-size:17px;">Load a review workspace</div>
            <div class="panel-sub">${window.EU_DATA === 'real' ? 'Load a local EasyICU export folder. Nothing is uploaded.' : 'Start with exported EasyICU tables or generate a compact demo set; review tabs appear immediately after loading.'}</div>
          </div>
        </div>

        <div style="border-top:1px solid var(--hair);padding-top:16px;">
          <div class="eyebrow" style="margin-bottom:10px;">Data source</div>
          <div class="radio-row">
            <label class="radio ${window.EU_DATA === 'real' ? 'on' : ''}" role="button" tabindex="0" data-datamode="real"><span class="mk"></span> Previously exported data</label>
            <label class="radio ${window.EU_DATA !== 'real' ? 'on' : ''}" role="button" tabindex="0" data-datamode="demo"><span class="mk"></span> Demo data</label>
          </div>
        </div>

        <div class="card sunken pad mt-16">
          <div class="eyebrow" style="margin-bottom:4px;">${window.EU_DATA === 'real' ? 'Local export' : 'Demo review'}</div>
          <div style="font-weight:600;font-size:14px;">${window.EU_DATA === 'real' ? 'Load exported EasyICU tables' : 'Generate a lightweight demo review workspace'}</div>
          <div class="panel-sub" style="margin-top:2px;">${window.EU_DATA === 'real' ? 'Pick a registered local export, or add one by path.' : 'Loads a fast core ICU concept set for tables, trends, patient overview, and quality checks.'}</div>
          ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
          ${window.EU_DATA === 'real' ? `
          ${patientSourceReadyCard()}
          ${sourceRegistryBlock('single')}
          <p style="font-size:11.5px;color:var(--ink-4);margin:14px 0 0;">Use Data Extraction first to create or refresh this folder. The last successful export is remembered locally.</p>
          <button class="btn primary block lg mt-16" data-gen>${icon('folder', 14)} Load local export</button>` : `
          <div class="cols-2 mt-16" style="gap:28px;">
            <div>
              <div class="row" style="justify-content:space-between;"><label style="font-size:12.5px;font-weight:500;color:var(--ink-2);">Number of patients</label><span class="mono" style="font-size:12px;">10</span></div>
              <div class="slider"><div class="track"><div class="fill" style="width:0%"></div><div class="knob" style="left:0%"></div></div><div class="ends"><span>10</span><span>48</span></div></div>
            </div>
            <div>
              <div class="row" style="justify-content:space-between;"><label style="font-size:12.5px;font-weight:500;color:var(--ink-2);">Data duration (hours)</label><span class="mono" style="font-size:12px;">24</span></div>
              <div class="slider"><div class="track"><div class="fill" style="width:0%"></div><div class="knob" style="left:0%"></div></div><div class="ends"><span>24</span><span>48</span></div></div>
            </div>
          </div>
          <p style="font-size:11.5px;color:var(--ink-4);margin:14px 0 0;">Fast demo profile: core vitals, labs, SOFA/SOFA-2, Sepsis-3, AKI, interventions, demographics, and outcomes.</p>
          <button class="btn primary block lg mt-16" data-gen>${icon('play', 14)} Generate and load demo workspace</button>`}
        </div>
      </div>

      <div class="empty mt-16">
        <div class="glyph">${icon('viz', 22)}</div>
        <div class="t">Preview workspace awaits data</div>
        <div class="d">Generate demo data or load exported files above; the review tabs will appear here as a compact multi-view workspace.</div>
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
        patientView = 'loading';
        repaintScreen('patient');
        if (window.EU_DATA === 'real') {
          loadRealPatient(ok => { patientView = ok ? 'loaded' : 'idle'; repaintScreen('patient'); });
        } else {
          setTimeout(() => { patientView = 'loaded'; window.EU_HASWORK = true; repaintScreen('patient'); }, 1400);
        }
      }));
      root.querySelectorAll('[data-viz-reset]').forEach(b => b.addEventListener('click', () => {
        patientView = 'idle'; window.EU_VIZ_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; repaintScreen('patient');
      }));
      root.querySelectorAll('[data-patient-use-real]').forEach(b => b.addEventListener('click', () => {
        patientView = 'idle';
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
        root.querySelector('#ptbody').innerHTML = patientTabBody();
        bindPatientEntitySelection(root);
      });
      bindPatientEntitySelection(root);
    },
  };

  /* ---------------- COHORT STATISTICS ---------------- */
  /* In-page panels (aligned with cohort_redesign.py _SUBTABS): Group contrast,
     Coverage audit, Cohort profile, SOFA reclassification. Coverage + SOFA
     delegate to the shared EUAudit / EUSofa renderers (formerly standalone
     screens). */
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
    switch (cohortPanel) {
      case 'survival': return review ? cohortSurvivalBody(review) : cohortSurvivalDemoBody();
      case 'coverage': return review ? cohortCoverageBody(review) : (window.EUAudit ? window.EUAudit.panel() : '');
      case 'sofa':     return review ? cohortSofaBody(review) : (window.EUSofa ? window.EUSofa.panel() : '');
      case 'snapshot': return cohortSnapshotBody();
      default:         return cohortGroupsBody();
    }
  }

  function cohortProfileValue(row, value) {
    if (value == null || value === '') return '—';
    if (row.kind === 'count') return fmtInt(value);
    if (row.kind === 'percent') return fmtPct(value);
    return fmtNum(value, 1);
  }

  function cohortSurvivalBody(review) {
    const survival = review.survival_analysis || {};
    const outcomes = survival.outcomes || [];
    const groups = survival.group_options || [];
    const readyOutcomes = outcomes.filter(row => row.status === 'ready');
    const readyGroups = groups.filter(row => row.status === 'ready');
    const selectedOutcome = readyOutcomes.some(row => row.id === cohortSurvivalOutcome)
      ? cohortSurvivalOutcome
      : (survival.default_outcome || (readyOutcomes[0] && readyOutcomes[0].id));
    const selectedGroup = readyGroups.some(row => row.id === cohortSurvivalGroup)
      ? cohortSurvivalGroup
      : (survival.default_group || (readyGroups[0] && readyGroups[0].id));
    const curve = (survival.curves || []).find(row => row.outcome_id === selectedOutcome && row.group_id === selectedGroup);
    const outcomeButtons = outcomes.map(row => {
      const ready = row.status === 'ready';
      const cls = `seg-btn ${selectedOutcome === row.id ? 'active' : ''} ${ready ? '' : 'disabled'}`;
      const attr = ready ? `data-cohort-surv-outcome="${esc(row.id)}"` : `aria-disabled="true" title="${esc(row.reason || 'Unavailable')}"`;
      return `<button class="${cls}" ${attr}><span>${esc(row.label || row.id)}</span>${ready ? `<b>${fmtInt(row.event_count)} events</b>` : '<b>blocked</b>'}</button>`;
    }).join('');
    const groupButtons = groups.map(row => {
      const ready = row.status === 'ready';
      const cls = `seg-btn ${selectedGroup === row.id ? 'active' : ''} ${ready ? '' : 'disabled'}`;
      const attr = ready ? `data-cohort-surv-group="${esc(row.id)}"` : `aria-disabled="true" title="${esc(row.reason || 'Unavailable')}"`;
      const n = (row.groups || []).map(g => fmtInt(g.count)).join(' / ');
      return `<button class="${cls}" ${attr}><span>${esc(row.label || row.id)}</span><b>${ready ? n : 'blocked'}</b></button>`;
    }).join('');
    const blockedOutcomes = outcomes.filter(row => row.status !== 'ready');
    if (!curve) {
      return `
      <div class="sec-stack"><div class="lbl">Survival analysis</div><h2>Kaplan-Meier module</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">Outcome</div><div class="surv-segments">${outcomeButtons}</div></div>
        <div><div class="surv-label">Grouping</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">Survival analysis blocked</div><div class="d">${esc(survival.reason || 'This export does not expose an outcome with both event and time-to-event columns.')}</div></div></div>
      ${cohortSurvivalBlockedList(blockedOutcomes)}`;
    }
    const logrank = curve.logrank || {};
    return `
      <div class="sec-stack"><div class="lbl">Survival analysis</div><h2>Kaplan-Meier curves and log-rank</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">Outcome</div><div class="surv-segments">${outcomeButtons}</div></div>
        <div><div class="surv-label">Grouping</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      <div class="surv-card mt-14">
        <div class="surv-head">
          <div>
            <div class="eyebrow">Exploratory · unadjusted</div>
            <h3>${esc(curve.label || 'Kaplan-Meier curve')}</h3>
            <p>${esc(curve.time_label || 'Time-to-event')} · event <span class="mono">${esc(curve.event_column || '')}</span> · time <span class="mono">${esc(curve.time_column || '')}</span></p>
          </div>
          <div class="surv-logrank">
            <span>Log-rank</span>
            <strong>${logrank.status === 'ready' ? `χ² ${fmtNum(logrank.chi_square, 2)} · p ${fmtP(logrank.p_value)}` : 'blocked'}</strong>
            <small>${logrank.status === 'ready' ? 'df 1 · exploratory only' : esc(logrank.reason || 'not enough events')}</small>
          </div>
        </div>
        ${cohortSurvivalChart(curve)}
        ${cohortRiskTable(curve)}
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">Not manuscript-ready by itself</div><div class="d">KM/log-rank is computed from bounded cohort aggregates and marked exploratory. Any claim still needs the evidence-bound Agent gate and human review.</div></div></div>
      ${cohortSurvivalBlockedList(blockedOutcomes)}`;
  }

  function cohortSurvivalBlockedList(rows) {
    if (!rows || !rows.length) return '';
    return `
      <div class="surv-blocked mt-12">
        ${rows.map(row => `<div class="surv-blocked-row"><span>${esc(row.label || row.id)}</span><em>${esc(row.reason || 'Unavailable for this export')}</em></div>`).join('')}
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
          <text x="${l + plotW / 2}" y="${h - 2}" text-anchor="middle" class="km-axis-title">Days</text>
          <text x="14" y="${tpad + plotH / 2}" transform="rotate(-90 14 ${tpad + plotH / 2})" text-anchor="middle" class="km-axis-title">Survival probability</text>
          ${groups.map((g, i) => `<path d="${stepPath(g.points || [])}" class="km-line" style="stroke:${colors[i % colors.length]};"></path>`).join('')}
        </svg>
        <div class="km-legend">
          ${groups.map((g, i) => `<span><i style="background:${colors[i % colors.length]};"></i>${esc(g.label)} · n ${fmtInt(g.n)} · events ${fmtInt(g.events)}</span>`).join('')}
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
        <div class="surv-label">Number at risk</div>
        <table class="risk-table">
          <thead><tr><th>Group</th>${times.map(tick => `<th>${fmtNum(tick, 1)}d</th>`).join('')}</tr></thead>
          <tbody>
            ${rows.map(row => `<tr><td>${esc(row.label)}</td>${(row.values || []).map(value => `<td>${fmtInt(value)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }

  function cohortSurvivalDemoBody() {
    return `
      <div class="sec-stack"><div class="lbl">Survival analysis</div><h2>Kaplan-Meier module</h2></div>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">Real export required</div><div class="d">KM curves, log-rank, and number-at-risk tables require a registered local export with event and time-to-event columns. Demo mode does not fabricate survival curves.</div></div></div>`;
  }

  function cohortCoverageBody(review) {
    const rows = review.coverage || [];
    const q = review.quality || {};
    return `
      <div class="sec-stack"><div class="lbl">Coverage audit</div><h2>Real module coverage and quality</h2></div>
      <div class="audit-cards">
        ${[
          ['Modules OK', fmtInt(q.modules_ok)],
          ['Watchlist', fmtInt(q.watchlist_count)],
          ['Median coverage', fmtPct(q.median_coverage_pct)],
          ['Neutral event modules', fmtInt(q.modules_neutral)],
          ['Unknown coverage', fmtInt(q.modules_unknown)],
        ].map(([k, v]) => `<div class="audit-card"><div class="ac-k">${k}</div><div class="ac-v mono">${v}</div></div>`).join('')}
      </div>
      <div class="table-wrap table-scroll mt-16">
        <table class="eu-table">
          <thead><tr><th>Module</th><th class="num">Records</th><th class="num">Fields</th><th class="num">Covered entities</th><th class="num">Coverage</th><th>Status</th></tr></thead>
          <tbody>
            ${rows.map(row => `<tr>
              <td class="key">${esc(row.module)}</td>
              <td class="num">${fmtInt(row.rows)}</td>
              <td class="num">${fmtInt(row.column_count)}</td>
              <td class="num">${fmtInt(row.covered_entities)}</td>
              <td class="num">${fmtPct(row.coverage_pct)}</td>
              <td><span class="pill ${row.quality_status === 'ok' || row.quality_status === 'neutral' ? 'ok' : 'warn'}" style="height:20px;">${esc(row.quality_status || 'unknown')}</span></td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">Fail-closed scope</div><div class="d">Coverage is aggregate-only. Row-level filtering, subgroup missingness, and eligibility waterfalls remain blocked until a bounded cohort-builder backend exists.</div></div></div>`;
  }

  function cohortSofaBody(review) {
    const s = review.summary || {};
    const sofa = s.sofa2 || {};
    const reclass = review.sofa_reclassification || {};
    const bins = sofa.bins || [];
    const maxBin = Math.max(1, ...bins.map(b => b.count || 0));
    const movement = reclass.direction_counts || {};
    const delta = reclass.delta_summary || {};
    const matrix = reclass.transition_matrix || [];
    const movementCards = reclass.status === 'ready' ? [
      [fmtInt(reclass.paired_count), 'Paired entities', `${fmtPct(reclass.coverage_pct)} of cohort`, 'n'],
      [fmtInt(movement.up && movement.up.count), 'SOFA-2 higher', fmtPct(movement.up && movement.up.pct), 'up'],
      [fmtInt(movement.down && movement.down.count), 'SOFA-2 lower', fmtPct(movement.down && movement.down.pct), 'down'],
      [fmtNum(delta.median, 1), 'Median delta', 'SOFA-2 minus SOFA-1', 'delta'],
    ] : [];
    return `
      <div class="sec-stack"><div class="lbl">SOFA reclassification</div><h2>SOFA-2 aggregate review</h2></div>
      <div class="rc-kpis">
        ${[
          [fmtNum(sofa.median, 1), 'Median SOFA-2', `${fmtInt(sofa.count)} entities with score`, 'delta'],
          [fmtNum(sofa.mean, 1), 'Mean SOFA-2', 'registered export aggregate', 'n'],
          [fmtNum(sofa.min, 1), 'Min', 'bounded column read', 'down'],
          [fmtNum(sofa.max, 1), 'Max', 'bounded column read', 'up'],
        ].map(([v, label, hint, kind]) => `
          <div class="rc-kpi rc-${kind}">
            <div class="rk-top"><span class="rk-ico">${icon(kind === 'up' ? 'arrow' : kind === 'down' ? 'arrow' : 'layers', 13)}</span><span class="rk-label">${label}</span></div>
            <div class="rk-val mono">${v}</div>
            <div class="rk-hint">${hint}</div>
          </div>`).join('')}
      </div>
      <div class="card pad mt-16">
        <div class="rc-sec-t">SOFA-2 severity bins</div>
        <div class="rc-groups">
          ${bins.map(bin => `
            <div class="rc-grow">
              <div class="rg-head"><span class="rg-name">${esc(bin.label)}</span><span class="rg-pct mono">${fmtPct(bin.pct)}</span></div>
              <div class="rg-bar"><div class="rg-fill same" style="width:${((bin.count || 0) / maxBin * 100).toFixed(0)}%;"></div></div>
              <div class="rg-meta"><span>${fmtInt(bin.count)} entities</span></div>
            </div>`).join('')}
        </div>
      </div>
      ${reclass.status === 'ready' ? `
        <div class="card pad mt-16">
          <div class="rc-sec-t">SOFA-1 to SOFA-2 movement</div>
          <div class="rc-kpis compact">
            ${movementCards.map(([v, label, hint, kind]) => `
              <div class="rc-kpi rc-${kind}">
                <div class="rk-top"><span class="rk-ico">${icon(kind === 'up' ? 'arrow' : kind === 'down' ? 'arrow' : 'layers', 13)}</span><span class="rk-label">${label}</span></div>
                <div class="rk-val mono">${v}</div>
                <div class="rk-hint">${hint}</div>
              </div>`).join('')}
          </div>
          <div class="rc-sec-t mt-12">Worst-ICU severity transition matrix</div>
          <table class="mini-table">
            <thead><tr><th>SOFA-1 \\ SOFA-2</th>${(reclass.severity_bins || []).map(label => `<th>${esc(label)}</th>`).join('')}</tr></thead>
            <tbody>
              ${matrix.map(row => `
                <tr>
                  <td>${esc(row.label)}</td>
                  ${(row.cells || []).map(cell => `<td><span class="mono">${fmtInt(cell.count)}</span><span class="muted"> ${fmtPct(cell.pct)}</span></td>`).join('')}
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
        <div class="note ok mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">Paired aggregate ready</div><div class="d">Worst-ICU SOFA-1/SOFA-2 movement is computed from bounded per-entity score aggregates only. No paired patient rows or inferential statistics are returned.</div></div></div>
      ` : `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">Paired reclassification blocked</div><div class="d">${esc(reclass.reason || 'Paired SOFA-1/SOFA-2 reclassification is not available for this export.')}</div></div></div>`}`;
  }

  function cohortSnapshotBody() {
    const review = cohortReview();
    if (review && review.summary) {
      const s = review.summary;
      return `
      <div class="sec-stack"><div class="lbl">Cohort profile</div><h2>Real cohort aggregate</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">Cohort size</div><div class="val">${fmtInt(s.cohort_size)}</div></div>
        <div class="stat"><div class="label">Median age</div><div class="val">${fmtNum(s.age && s.age.median, 1)}</div></div>
        <div class="stat"><div class="label">Female</div><div class="val">${fmtPct(s.sex && s.sex.female_pct)}</div></div>
        <div class="stat"><div class="label">Sepsis-3 +</div><div class="val">${fmtPct(s.sepsis_pct)}</div></div>
        <div class="stat"><div class="label">Median SOFA-2</div><div class="val">${fmtNum(s.sofa2 && s.sofa2.median, 1)}</div></div>
        <div class="stat accent"><div class="label">Mortality</div><div class="val">${fmtPct(s.mortality_pct)}</div></div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">Aggregate ranges</div>
          ${[
            ['Age', s.age],
            ['SOFA-2', s.sofa2],
            ['ICU LOS days', s.los_icu_days],
          ].map(([label, item]) => `<div class="setup-row"><span class="k">${label}</span><span class="vv">median ${fmtNum(item && item.median, 1)} · range ${fmtNum(item && item.min, 1)}-${fmtNum(item && item.max, 1)}</span></div>`).join('')}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">Source provenance</div>
          <div class="setup-row"><span class="k">Source</span><span class="vv">${esc((review.source || {}).label || 'Local export')}</span></div>
          <div class="setup-row"><span class="k">Database</span><span class="vv">${esc((review.source || {}).database || 'unknown')}</span></div>
          <div class="setup-row"><span class="k">Path hash</span><span class="vv mono">${esc((review.source || {}).path_hash || '')}</span></div>
          <div class="setup-row"><span class="k">Scope</span><span class="vv">${esc((review.provenance || {}).payload_scope || 'cohort_aggregate_only')}</span></div>
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Real registered export aggregate. Row-level filters, generic Table One p-values, matched cohorts, and paired SOFA reclassification remain blocked; timed survival outcomes are handled in the KM module.</p>`;
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
    const ages = [['18–39', 2], ['40–59', 3], ['60–74', 4], ['≥75', 1]];
    const sofa = [['0–5', 4], ['6–8', 3], ['9–11', 2], ['≥12', 1]];
    const maxA = Math.max(...ages.map(a => a[1])), maxS = Math.max(...sofa.map(s => s[1]));
    return `
      <div class="sec-stack"><div class="lbl">Cohort profile</div><h2>${t('Demo cohort snapshot', '演示队列概览')}</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">${t('Patients', '患者数')}</div><div class="val">10</div></div>
        <div class="stat"><div class="label">${t('Median age', '年龄中位数')}</div><div class="val">56</div></div>
        <div class="stat"><div class="label">${t('Female', '女性')}</div><div class="val">70%</div></div>
        <div class="stat"><div class="label">${t('Sepsis-3 +', 'Sepsis-3 阳性')}</div><div class="val">60%</div></div>
        <div class="stat"><div class="label">${t('Median SOFA', 'SOFA 中位数')}</div><div class="val">6</div></div>
        <div class="stat accent"><div class="label">${t('Mortality', '死亡率')}</div><div class="val">20%</div></div>
      </div>
      <div class="cols-2 mt-16">
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('Age distribution', '年龄分布')}</div>
          ${ages.map(([lab, n]) => `<div class="qrow"><span>${lab}</span><div class="qbar"><span style="width:${(n / maxA * 100)}%"></span></div><span class="qv">${n}</span></div>`).join('')}
        </div>
        <div class="card pad">
          <div class="eyebrow" style="margin-bottom:8px;">${t('SOFA severity', 'SOFA 严重度')}</div>
          ${sofa.map(([lab, n]) => `<div class="qrow"><span>${lab}</span><div class="qbar"><span style="width:${(n / maxS * 100)}%"></span></div><span class="qv">${n}</span></div>`).join('')}
        </div>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">${t('Demo / seeded example values for UI preview — not a real run output.', '演示 / 示例数据，仅用于界面预览 —— 非真实运行结果。')}</p>`;
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
      const radio = (row) => `<label class="radio ${active.id === row.id ? 'on' : ''}" role="button" tabindex="0" data-cohort-comp="${esc(row.id)}"><span class="mk"></span> ${esc(row.label || row.id)}</label>`;
      return `
      <div class="coh-jump">
        <button class="cj-card" data-cohgo="coverage">
          <span class="cj-ico">${icon('shield', 16)}</span>
          <span class="cj-tx"><span class="cj-t">Coverage audit</span><span class="cj-d">Review module coverage before analysis</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
        <button class="cj-card" data-cohgo="snapshot">
          <span class="cj-ico">${icon('cohort', 16)}</span>
          <span class="cj-tx"><span class="cj-t">Cohort profile</span><span class="cj-d">Inspect real registered export aggregates</span></span>
          <span class="cj-go">${icon('arrow', 13)}</span>
        </button>
      </div>
      <div class="sec-stack"><div class="lbl">Analysis table</div><h2>Real cohort aggregate</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">Cohort size</div><div class="val">${fmtInt(s.cohort_size)}</div></div>
        <div class="stat accent"><div class="label">Mortality</div><div class="val">${fmtPct(s.mortality_pct)}</div></div>
        <div class="stat accent"><div class="label">Median age</div><div class="val">${fmtNum(s.age && s.age.median, 1)}</div></div>
        <div class="stat accent"><div class="label">Median SOFA-2</div><div class="val">${fmtNum(s.sofa2 && s.sofa2.median, 1)}</div></div>
      </div>
      <div class="note mt-12"><div class="ico">${icon('folder', 14)}</div><div class="body"><div class="t">Local export cohort review ready</div><div class="d">Source ${esc(source.label || 'Local export')} · ${esc(source.database || 'unknown')} · path hash <span class="mono">${esc(source.path_hash || '')}</span> · aggregate-only payload.</div></div></div>

      <div class="sec-stack"><div class="lbl">Comparison</div><h2>Select descriptive split</h2></div>
      <div class="radio-row">
        ${supported.map(row => radio(row)).join('')}
      </div>

      <div class="sec-stack"><div class="lbl">Summary</div><h2>${esc(active.label || 'Descriptive split')} overview</h2></div>
      <div class="cols-3">
        ${activeGroups.map((g, i) => `<div class="stat ${i === 0 ? 'accent' : ''}"><div class="label">${esc(g.label)}</div><div class="val">${fmtInt(g.count)}</div><div style="font-size:11px;color:var(--ink-4);margin-top:4px;">${fmtPct(g.pct)}</div></div>`).join('')}
      </div>

      <div class="sec-stack"><div class="lbl">Descriptive profile</div><h2>Aggregate-only group characteristics</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Metric</th>${profileColumns.map(col => `<th class="num">${esc(col)}</th>`).join('')}<th>Status</th></tr></thead>
          <tbody>
            ${profileRows.map(row => `<tr>
              <td class="key">${esc(row.metric)}${row.unit ? ` <span class="mono" style="color:var(--ink-4);font-weight:500;">${esc(row.unit)}</span>` : ''}</td>
              ${(row.values || []).map(value => `<td class="num">${cohortProfileValue(row, value)}</td>`).join('')}
              <td><span class="pill ok" style="height:20px;">descriptive</span></td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>

      <div class="sec-stack"><div class="lbl">Fail-closed</div><h2>Blocked cohort functions</h2></div>
      <div class="cols-3">
        ${blocked.map(item => `<div class="stat"><div class="label">${esc(item.id)}</div><div class="val" style="font-size:13px;line-height:1.35;font-family:var(--font-body);font-weight:600;">${esc(item.status)}</div><div style="font-size:11px;color:var(--ink-4);margin-top:6px;">${esc(item.reason)}</div></div>`).join('')}
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">No row-level filters, generic Table One p-values, SMDs, matched cohort, or paired SOFA reclassification are exposed here. Use the Survival curves tab for audited KM/log-rank when timed outcomes exist.</p>`;
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
      <div class="sec-stack"><div class="lbl">Analysis table</div><h2>Local export group contrast</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">Total stays</div><div class="val">${fmtInt(s.stays)}</div></div>
        <div class="stat accent"><div class="label">Mean age</div><div class="val">${fmtNum(s.mean_age, 1)}</div></div>
        <div class="stat accent"><div class="label">Female %</div><div class="val">${fmtPct(s.female_pct)}</div></div>
        <div class="stat accent"><div class="label">Mortality</div><div class="val">${fmtPct(s.mortality)}</div></div>
      </div>
      <div class="sec-stack"><div class="lbl">Summary</div><h2>Outcome groups</h2></div>
      <div class="cols-3">
        <div class="stat"><div class="label">Survived</div><div class="val">${fmtInt(c.survived)}</div></div>
        <div class="stat"><div class="label">Deceased</div><div class="val">${fmtInt(c.deceased)}</div></div>
        <div class="stat accent"><div class="label">Rows reviewed</div><div class="val">${fmtInt((ws.tableRows || []).length)}</div></div>
      </div>
      <div class="sec-stack"><div class="lbl">Table one</div><h2>Baseline characteristics comparison</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Characteristic</th><th class="num">Overall</th><th class="num">Survived</th><th class="num">Deceased</th></tr></thead>
          <tbody>
            ${chars.map(r => `<tr><td class="key">${esc(r[0])}</td>${r.slice(1).map(c => `<td class="num">${fmtNum(c, 2)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Real local export summary. P-values and manuscript claims are intentionally withheld from this UI preview.</p>`;
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
    const radio = (key, label) => `<label class="radio ${cohortCompare === key ? 'on' : ''}" role="button" tabindex="0" data-cohort-comp="${key}"><span class="mk"></span> ${label}</label>`;
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
      <div class="sec-stack"><div class="lbl">Analysis table</div><h2>Group Contrast Table</h2></div>
      <div class="stat-grid">
        <div class="stat accent"><div class="label">Total patients</div><div class="val">10</div></div>
        <div class="stat accent"><div class="label">Mean age</div><div class="val">54.8</div></div>
        <div class="stat accent"><div class="label">Male %</div><div class="val">30.0%</div></div>
        <div class="stat accent"><div class="label">Mortality</div><div class="val">20.0%</div></div>
      </div>

      <div class="sec-stack"><div class="lbl">Comparison</div><h2>Select comparison mode</h2></div>
      <div class="radio-row">
        ${radio('outcome', 'Survived vs Deceased')}
        ${radio('age', 'Age Groups')}
        ${radio('sex', 'Male vs Female')}
        ${radio('los', 'Short vs Long Stay')}
        ${radio('sepsis', 'Sepsis vs Non-sepsis')}
        ${radio('custom', 'Custom Threshold')}
      </div>

      <div class="sec-stack"><div class="lbl">Features</div><h2>Select feature modules</h2></div>
      <div class="card pad" style="padding:14px 16px;">
        <div class="row wrap gap-6">
          <span class="chip solid">Demographics <span class="x">×</span></span>
          <span class="chip solid">Outcome <span class="x">×</span></span>
          <span class="chip solid">Vital Signs <span class="x">×</span></span>
          <span class="chip solid">Sepsis-3 (SOFA-2 based) <span class="x">×</span></span>
          <span class="grow"></span>
          <span style="color:var(--ink-4);">${icon('chevdown', 16)}</span>
        </div>
        <div class="row" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--hair);justify-content:space-between;">
          <span class="row gap-6" style="font-size:12px;color:var(--ink-3);">${icon('flask', 13)} Features to load: 9</span>
          <span style="color:var(--ink-4);">${icon('chevdown', 16)}</span>
        </div>
      </div>

      <div class="sec-stack"><div class="lbl">Summary</div><h2>${esc(comp.title)} overview</h2></div>
      <div class="cols-3">
        ${comp.groups.map((g, i) => `<div class="stat ${i === 2 ? 'accent' : ''}"><div class="label">${esc(g[0])}</div><div class="val">${esc(g[1])}</div></div>`).join('')}
      </div>
      ${comp.note ? `<div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="d" style="margin:0;">${esc(comp.note)}</div></div></div>` : ''}

      <div class="sec-stack"><div class="lbl">Table one</div><h2>Baseline characteristics comparison</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Characteristic</th><th class="num">Overall (n=10)</th><th class="num">Survived (n=8)</th><th class="num">Deceased (n=2)</th><th class="num">p-value</th></tr></thead>
          <tbody>
            ${comp.table.map(r => `<tr><td class="key">${esc(r[0])}</td>${r.slice(1).map(c => `<td class="num">${esc(c)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Demo / seeded example values for UI preview — not a real run output.</p>`;
  }

  S.cohort = {
    section: 'viz', nav: 'viz', sub: 'cohort',
    crumbs: ['Home', 'Data Visualization', 'Cohort Statistics'],
    get actionHtml() { return `<button class="btn primary" data-cohort-run ${cohortView === 'loading' ? 'aria-disabled="true"' : ''}>${icon('refresh', 13)} Re-run</button>`; },
    rail: () => vizRail('cohort'),
    afterRender(root) {
      bindSourceRegistry(root, 'cohort');
      root.querySelectorAll('[data-cohort-run]').forEach(b => b.addEventListener('click', () => {
        if (cohortView === 'loading') return;
        cohortView = 'loading'; repaintScreen('cohort');
        if (window.EU_DATA === 'real') {
          loadRealCohort(ok => { cohortView = 'loaded'; if (!ok) cohortView = 'loaded'; repaintScreen('cohort'); });
        } else {
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
      root.querySelectorAll('[data-cohort-surv-outcome]').forEach(b => b.addEventListener('click', () => {
        if (b.dataset.cohortSurvOutcome === cohortSurvivalOutcome) return;
        cohortSurvivalOutcome = b.dataset.cohortSurvOutcome || 'hospital_death';
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-surv-group]').forEach(b => b.addEventListener('click', () => {
        if (b.dataset.cohortSurvGroup === cohortSurvivalGroup) return;
        cohortSurvivalGroup = b.dataset.cohortSurvGroup || 'sepsis';
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
      if (cohortPanel === 'sofa' && window.EUSofa && window.EUSofa.bind) window.EUSofa.bind(root);
    },
    render() {
      if (window.__euCohortPanel) { cohortPanel = window.__euCohortPanel; window.__euCohortPanel = null; }
      const ws = window.EU_VIZ_WORKSPACE;
      const head = `
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>Workspace</span> ${icon('chevron', 11)} <span>${ws ? 'Local export' : 'Demo cohort'}</span> ${icon('chevron', 11)} <span style="color:var(--ink-2);">Cohort statistics</span>
      </div>
      <div class="page-head" style="margin-bottom:16px;">
        <h1 style="margin-top:0;">${ws ? 'Local export cohort' : 'Sepsis vs Non-sepsis'}</h1>
        <p class="lead">${ws ? 'Real exported module tables · local-only summary' : 'Group contrast · coverage audit · cohort profile · SOFA reclassification'}</p>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:9px;">${t('Key terms', '关键术语')}: ${window.gloss('cohort', t('cohort', '队列'))} · ${window.gloss('denominator', t('denominator', '分母'))} · ${window.gloss('SOFA')} · ${window.gloss('Sepsis-3')}</div>
      </div>`;
      if (window.EU_DATA === 'real' && !ws && cohortView !== 'loading') {
        return head + `<div class="card pad" style="max-width:720px;">
          <div class="panel-title" style="font-size:17px;">Load a local export first</div>
          <div class="panel-sub mt-4">Cohort Statistics uses the same export snapshot as Patient Review.</div>
          ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
          ${sourceRegistryBlock('single')}
          <button class="btn primary mt-16" data-cohort-run>${icon('folder', 14)} Load local export</button>
        </div>`;
      }
      if (cohortView === 'loading') {
        return head + `<div class="card pad">
          <div class="load-strip">
            <span class="spin accent"></span>
            <div class="grow"><div style="font-weight:600;font-size:12.75px;">Recomputing cohort statistics…</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">reproducible · no outbound calls</div></div>
          </div>
          <div class="indet mt-12"></div>
          <div class="st-stats mt-16">${[0,1,2,3].map(() => `<div class="sk-stat"><div class="sk sk-line sm" style="width:52%"></div><div class="sk" style="height:22px;width:64%;margin-top:10px;"></div></div>`).join('')}</div>
          <div class="sk-table mt-16">${[0,1,2,3,4].map(() => `<div class="sk-trow">${[60,40,40,40,30].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}</div>
        </div>`;
      }
      return head + `
      <div class="card" style="padding:0;overflow:hidden;">
        <div class="row" style="justify-content:space-between;padding:11px 16px;border-bottom:1px solid var(--hair);">
          <span style="font-weight:600;font-size:12.5px;">Agent preflight</span>
          <span class="mono" style="font-size:11px;color:var(--ink-4);">current session</span>
        </div>
        <div class="preflight">
          ${[
            ['Input package', ws ? `${fmtInt(ws.summary && ws.summary.stays)} stays · ${fmtInt(ws.summary && ws.summary.modules)} modules` : '10 stays · demo concept set', 'ok', null],
            ['Evidence checks', ws ? 'export parsed · denominators previewed' : 'coverage + denominators ready', 'ok', null],
            ['Draft gate', 'locked · needs reviewer sign-off', 'warn', 'agent'],
          ].map(([tt, d, s, nav]) => `
            <div class="pf-cell" ${nav ? `data-nav="${nav}" role="button" tabindex="0" style="cursor:pointer;"` : ''}>
              <div class="eyebrow" style="display:flex;align-items:center;gap:6px;">
                <span class="dot-${s}"></span>${tt}${nav ? `<span style="margin-left:auto;color:var(--ink-4);">${icon('arrow', 12)}</span>` : ''}
              </div>
              <div style="font-size:12.5px;color:var(--ink-2);margin-top:6px;">${d}</div>
              ${nav ? `<div style="font-size:11px;color:var(--ink-4);margin-top:4px;">${t('4 of 5 checks passed — sign off in the Agent to unlock', '已通过 5 项中的 4 项 —— 在代理中签署以解锁')}</div>` : ''}
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

  function crossHeader() {
    const ws = window.EU_VIZ_WORKSPACE;
    const xdb = window.EU_CROSSDB_WORKSPACE;
    const xdbDemo = xdb && xdb.source_type === 'legacy_simulated_multidb_feature_frames';
    return `
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>Workspace</span> ${icon('chevron', 11)} <span>${xdb ? (xdbDemo ? 'Demo simulated frames' : 'Local exports') : (ws ? 'Local export' : 'Demo cohort')}</span> ${icon('chevron', 11)} <span style="color:var(--ink-2);">Cross-DB benchmark</span>
      </div>
      <div class="page-head" style="margin-bottom:14px;">
        <h1 style="margin-top:0;">Cross-DB benchmark</h1>
        <p class="lead">${window.EU_DATA === 'real' ? 'Load real ICU database folders and compare feature density distributions by module.' : 'Same cohort definition compared across ≥2 ICU databases.'}</p>
      </div>`;
  }
  function crossFmt(key, value) {
    if (value == null) return '—';
    if (key === 'stays' || key === 'cohort_size' || key === 'modules' || key === 'total_rows' || key === 'total_records') return fmtInt(value);
    if (key === 'feature_rows' || key === 'concepts_present') return fmtInt(value);
    if (key === 'female_pct' || key === 'mortality' || key === 'mortality_pct' || key === 'sepsis_pct' || key === 'coverage_median_pct') return fmtPct(value);
    return fmtNum(value, 1);
  }
  function rawCrossdbSetup() {
    const sel = CROSS_DBS.filter(d => d[1]).length;
    return `
      <div class="note info">
        <div class="ico">${icon('benchmark', 16)}</div>
        <div class="body"><span class="t">Real raw database mode</span> <span class="d" style="display:inline;">— choose a local ICU data root containing database folders, then compare all catalog concepts with cross-database support. No rows leave this machine.</span></div>
      </div>
      ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
      <div class="card pad mt-16">
        <div class="row between gap-12" style="align-items:flex-start;">
          <div>
            <div class="panel-title">Raw ICU data root</div>
            <div class="panel-sub mt-4">Folder that contains database subfolders such as mimiciv, eicu, aumc, hirid, mimiciii, or sic.</div>
          </div>
          <span class="pill ok"><span class="dot"></span>local only</span>
        </div>
        <div class="path-field editable mt-14">
          <span class="pf-ico">${icon('folder', 14)}</span>
          <input class="pf-input" data-crossdb-root type="text" spellcheck="false" autocomplete="off" value="${esc(defaultRawCrossdbRoot())}" placeholder="${esc('Paste a local ICU database root folder')}" aria-label="ICU data root" />
        </div>
      </div>
      <div class="sec-stack"><div class="lbl">Databases · <span id="dbcount">${sel}</span> selected</div></div>
      <div class="db-grid" id="dbgrid">
        ${CROSS_DBS.map(([n, on]) => `
          <div class="db-card ${on ? 'sel' : ''}" data-db="${CROSS_DBS.findIndex(d => d[0] === n)}">
            <div class="row gap-8" style="min-width:0;">
              <span class="${on ? '' : 'ink-4'}" style="flex:none;color:${on ? 'var(--accent-ink)' : 'var(--ink-4)'};">${icon('db', 15)}</span>
              <div style="min-width:0;">
                <div style="font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${n}</div>
                <div class="mono" style="font-size:10.5px;color:var(--ink-4);">raw local database</div>
              </div>
            </div>
            <span class="db-mk pill ${on ? 'ok' : 'dashed'}" style="flex:none;height:20px;">${on ? '<span class="dot"></span>selected' : 'add'}</span>
          </div>`).join('')}
      </div>
      <div class="gate-strip mt-20">
        <span class="pill"><span style="color:var(--ink-3);">${icon('benchmark', 12)}</span> <span id="runhint">${sel} of 6 · need ≥ 2</span></span>
        <span class="pill">all supported catalog concepts</span>
        <div class="grow"></div>
        <button class="btn primary" data-run ${sel < 2 ? 'aria-disabled="true"' : ''}>${icon('play', 13)} Load real density benchmark</button>
      </div>`;
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
    return `
      <div class="loaded-bar">
        <span class="pill ok"><span class="dot"></span>Loaded</span>
        <div class="grow"><span style="font-weight:600;font-size:13px;">${demoMode ? 'Seeded simulated density benchmark ready' : (rawMode ? 'Real raw-database density benchmark ready' : 'Real cross-database benchmark ready')}</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${fmtInt(sources.length)} ${rawMode || demoMode ? 'databases' : 'exports'} · ${fmtInt(shared.length)} shared modules</span></div>
        <button class="btn sm" data-viz-reset>${icon('sliders', 13)} Change selection</button>
        <button class="btn sm" data-crossdb-export>${icon('download', 13)} Export JSON</button>
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('benchmark', 16)}</div>
        <div class="body"><span class="t">${demoMode ? 'Legacy seeded feature-frame distribution' : (rawMode ? 'Raw ICU database distribution' : 'Registered export comparison')}</span> <span class="d" style="display:inline;">— ${demoMode ? 'Feature density curves are computed from the old clinically-shaped demo generator, then aggregated by module; this is not a user database.' : (rawMode ? 'Feature density curves are computed from local ICU database folders through easyicu.load_concepts; no patient rows are returned.' : 'Cross-DB aggregate-only payload from the local source registry. Matched cohort definitions and formal claims still require the evidence-bound agent path.')}</span></div>
      </div>
      <div class="note ok mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">Compatibility gate: ${esc(gateStatus)}</span> <span class="d" style="display:inline;">— ${esc(mode)} · matched_cohort=false · inferential_statistics=false.</span></div>
      </div>
      <div class="sec-stack"><div class="lbl">Source provenance</div></div>
      <div class="src-grid">
        ${sources.map(source => `
          <div class="src-card">
            <div class="row gap-8" style="min-width:0;">
              <span style="color:var(--accent-ink);">${icon('db', 15)}</span>
              <div style="min-width:0;">
                <div style="font-weight:650;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(source.label || 'Local export')}</div>
                <div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc((source.database || 'local').toUpperCase())} · ${demoMode ? 'demo seed' : (rawMode ? 'root' : 'path')} hash ${esc(source.path_hash || '—')}</div>
              </div>
            </div>
          </div>`).join('')}
      </div>
      <div class="sec-stack"><div class="lbl">${demoMode ? 'Loaded seeded distribution summary' : (rawMode ? 'Loaded raw-database distribution summary' : 'Loaded cross-database export summary')}</div></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Metric</th>${labels.map(c => `<th class="num">${esc(c)}</th>`).join('')}<th class="num">Δ range</th></tr></thead>
          <tbody>
            ${(xdb.rows || []).map(row => `<tr><td class="key">${esc(row.label)}</td>${(row.values || []).map(v => `<td class="num">${crossFmt(row.key, v)}</td>`).join('')}<td class="num" style="color:var(--ink-3);">${crossFmt(row.key, row.delta)}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      ${crossRealFeatureDensityByModule(xdb.feature_distributions || [], labels)}
      <div class="sec-stack"><div class="lbl">Module availability matrix</div></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Module</th>${labels.map(c => `<th class="num">${esc(c)}</th>`).join('')}<th class="num">Shared</th></tr></thead>
          <tbody>
            ${availability.map(row => `<tr><td class="key">${esc(row.module)}</td>${(row.values || []).map(v => `<td class="num">${v.present ? fmtPct(v.coverage_pct) : 'Missing'}</td>`).join('')}<td class="num">${row.shared ? 'Yes' : 'No'}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="sec-stack"><div class="lbl">Shared exported modules</div></div>
      <div class="row wrap gap-6">
        ${shared.length ? shared.map(m => `<span class="chip solid">${esc(m)}</span>`).join('') : '<span class="pill warn">No shared modules detected</span>'}
      </div>
      <div class="note warn mt-16">
        <div class="ico">${icon('lock', 16)}</div>
        <div class="body"><span class="t">Fail-closed scope</span> <span class="d" style="display:inline;">— ${(blocked.map(item => item.id).join(', ') || 'unsupported analyses')} remain blocked. Raw rows returned=${privacy.raw_rows_returned === true ? 'true' : 'false'}; inference=${esc(provenance.inference || 'blocked_until_numeric_evidence_gate')}.</span></div>
      </div>`;
  }

  function crossRealFeatureDensityByModule(modules, labels) {
    return crossFeatureDensityPanel(
      'Multi-database feature density grid',
      'Old Cross-DB layout: one subplot per feature, grouped by module; each subplot overlays the selected database density curves. No patient rows are returned.',
      modules,
      labels,
    );
  }

  function crossFeatureDensityPanel(title, subtitle, modules, labels) {
    const cleaned = (modules || []).filter(module => module && (module.features || []).length);
    if (!cleaned.length) return '';
    let selectedModule = crossDensityModule || 'all';
    if (selectedModule !== 'all' && !cleaned.some(module => module.module === selectedModule)) selectedModule = 'all';
    const visible = selectedModule === 'all' ? cleaned : cleaned.filter(module => module.module === selectedModule);
    const totalFeatures = cleaned.reduce((acc, module) => acc + (module.features || []).length, 0);
    const sharedFeatures = cleaned.reduce((acc, module) => acc + Number(module.shared_feature_count || 0), 0);
    const visibleFeatures = visible.reduce((acc, module) => acc + (module.features || []).length, 0);
    const labelRow = (labels || []).map((label, i) => `<span><i style="background:${densityPalette(i)};"></i>${esc(label)}</span>`).join('');
    const detail = findCrossDensityFeature(visible, crossDensityFeature) || (visible[0] && visible[0].features && { module: visible[0], row: visible[0].features[0] });
    if (detail && !crossDensityFeature) crossDensityFeature = crossFeatureKey(detail.module, detail.row);
    return `
      <div class="sec-stack"><div class="lbl">${esc(title)}</div></div>
      <div class="xdb-density-panel" data-density-total="${totalFeatures}">
        <div class="xdb-density-top">
          <div>
            <div class="xdb-density-title">${esc(title)}</div>
            <div class="xdb-density-sub">${esc(subtitle)}</div>
            <div class="xdb-density-meta mono">${fmtInt(cleaned.length)} modules · ${fmtInt(totalFeatures)} features · ${fmtInt(sharedFeatures)} shared across selected databases · showing ${fmtInt(visibleFeatures)}</div>
          </div>
          <div class="xdb-density-legend">${labelRow}</div>
        </div>
        <div class="xdb-density-controls">
          <button class="chip ${selectedModule === 'all' ? 'solid' : ''}" data-density-module-filter="all">All modules</button>
          ${cleaned.map(module => `<button class="chip ${selectedModule === module.module ? 'solid' : ''}" data-density-module-filter="${esc(module.module)}">${esc(catalogModuleLabel(module.module))} <span class="mono">${fmtInt((module.features || []).length)}</span></button>`).join('')}
        </div>
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
          <div><h3>${esc(moduleLabel)}</h3><p>${fmtInt(features.length)} features · ${fmtInt(module.shared_feature_count || 0)} shared</p></div>
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
            <div class="xdb-density-detail-sub">${esc(moduleLabel)} · ${esc(row.feature)}${meta.unit ? ` · ${esc(meta.unit)}` : ''} · aggregate density only</div>
          </div>
          <span class="pill dashed">${fmtInt(values.filter(v => v.present).length)} curves</span>
        </div>
        <div class="xdb-density-detail-plot">${crossFeatureCurve(row, labels)}</div>
        <div class="table-wrap table-scroll xdb-density-detail-table">
          <table class="eu-table">
            <thead><tr><th>Database</th><th class="num">Values</th><th class="num">Range</th><th class="num">Density points</th></tr></thead>
            <tbody>
              ${values.map((v, i) => `<tr>
                <td class="key"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${densityPalette(i)};margin-right:6px;"></span>${esc(labels[i] || v.source || `database ${i + 1}`)}</td>
                <td class="num">${v.present ? fmtInt(v.non_null || v.n || 0) : 'Missing'}</td>
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
      .map((value, i) => ({ value, label: labels[i] || value.source || `export ${i + 1}`, color: densityPalette(i) }))
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
    const stats = `<span>${fmtInt(series.length)} database curves</span><span>x ${fmtDensity(minX)}-${fmtDensity(maxX)}</span><span>n=${fmtInt(totalN)}</span>`;
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
          const label = labels[i] || v.source || `export ${i + 1}`;
          const cats = (v.categories || []).slice(0, 4);
          return `
            <div class="xdb-density-cat ${v.present ? '' : 'missing'}">
              <span class="xdb-density-src">${esc(label)}</span>
              <span class="xdb-density-cat-bars">${cats.map(cat => `<i title="${esc(cat.label)} ${fmtPct(cat.pct)}" style="width:${Math.max(2, Number(cat.pct) || 0)}%;background:${densityPalette(i)};"></i>`).join('')}</span>
              <span class="xdb-density-num mono">${v.present ? `${fmtInt(v.non_null || 0)} values` : 'Missing'}</span>
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

  function catalogModuleLabel(key) {
    const hit = ((window.EU_CATALOG || {}).groups || []).find(row => row[0] === key);
    return hit ? hit[1] : key;
  }

  function catalogFeatureMeta(key) {
    const hit = ((window.EU_CATALOG || {}).dict || {})[key];
    return { name: hit ? hit[0] : key, unit: hit ? hit[2] : '' };
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
    const loadingTitle = window.EU_DATA === 'real'
      ? 'Loading real feature densities from local databases…'
      : 'Loading seeded frames for selected databases…';
    const progressText = p.message || (window.EU_DATA === 'real'
      ? 'Starting local raw Cross-DB density job…'
      : 'Building seeded density frames…');
    return `<div class="card pad">
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow"><div style="font-weight:600;font-size:12.75px;">${loadingTitle}</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">local-only · nothing uploaded${p.phase ? ` · ${esc(p.phase)}` : ''}</div></div>
        ${tot ? `<span class="mono" style="font-size:11px;color:var(--ink-3);">${cur}/${tot}</span>` : ''}
        <button class="btn sm" ${window.EU_DATA === 'real' ? 'data-crossdb-cancel' : 'data-viz-reset'} ${crossRawCancelRequested || (window.EU_DATA === 'real' && !crossRawJobId) ? 'disabled' : ''}>${icon('stop', 13)} ${crossRawCancelRequested ? 'Cancel requested' : 'Cancel'}</button>
      </div>
      ${tot ? `<div style="height:8px;border-radius:999px;background:var(--surface-2,#eef0f4);overflow:hidden;margin:12px 0 8px;"><div style="height:100%;width:${pct}%;background:var(--accent,#2f7d6b);transition:width .25s;"></div></div>` : '<div class="indet mt-12"></div>'}
      <div style="font-size:12px;color:var(--ink-3);min-height:18px;margin-top:8px;">${esc(progressText)}</div>
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
        ? `<button class="btn" data-viz-reset>${icon('sliders', 13)} Change selection</button><button class="btn" data-crossdb-export>${icon('download', 13)} Export JSON</button><button class="btn primary" data-run>${icon('refresh', 13)} Re-run</button>`
        : `<button class="btn primary" data-run ${crossView === 'loading' ? 'aria-disabled="true"' : ''}>${icon('play', 13)} Run</button>`;
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
          <div class="body"><span class="t">Demo benchmark was not loaded</span> <span class="d" style="display:inline;">— run the benchmark again so the backend can build the seeded distribution payload.</span></div>
        </div>`;
      }
      /* idle — select databases */
      const sel = CROSS_DBS.filter(d => d[1]).length;
      return crossHeader() + `
        <div class="note info">
          <div class="ico">${icon('benchmark', 16)}</div>
          <div class="body"><span class="t">Select databases to compare</span> <span class="d" style="display:inline;">— pick two or more standardized ICU sources, then run the benchmark. Each uses an independent seeded feature frame in Demo Mode.</span></div>
        </div>
        <div class="sec-stack"><div class="lbl">Available databases · <span id="dbcount">${sel}</span> selected</div></div>
        <div class="db-grid" id="dbgrid">
          ${CROSS_DBS.map(([n, on], i) => `
            <div class="db-card ${on ? 'sel' : ''}" data-db="${i}">
              <div class="row gap-8" style="min-width:0;">
                <span class="${on ? '' : 'ink-4'}" style="flex:none;color:${on ? 'var(--accent-ink)' : 'var(--ink-4)'};">${icon('db', 15)}</span>
                <div style="min-width:0;">
                  <div style="font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${n}</div>
                  <div class="mono" style="font-size:10.5px;color:var(--ink-4);">144 seeded rows</div>
                </div>
              </div>
              <span class="db-mk pill ${on ? 'ok' : 'dashed'}" style="flex:none;height:20px;">${on ? '<span class="dot"></span>selected' : 'add'}</span>
            </div>`).join('')}
        </div>
        <div class="gate-strip mt-20">
          <span class="pill"><span style="color:var(--ink-3);">${icon('benchmark', 12)}</span> <span id="runhint">${sel} of 6 · need ≥ 2</span></span>
          <div class="grow"></div>
          <button class="btn primary" data-run ${sel < 2 ? 'aria-disabled="true"' : ''}>${icon('play', 13)} Run benchmark</button>
        </div>`;
    },
    afterRender(root) {
      bindSourceRegistry(root, 'crossdb');
      root.querySelectorAll('[data-density-module-filter]').forEach(b => b.addEventListener('click', () => {
        crossDensityModule = b.dataset.densityModuleFilter || 'all';
        crossDensityFeature = null;
        repaintScreen('crossdb');
      }));
      root.querySelectorAll('[data-density-feature-key]').forEach(b => b.addEventListener('click', () => {
        crossDensityFeature = b.dataset.densityFeatureKey || null;
        repaintScreen('crossdb');
      }));
      const grid = root.querySelector('#dbgrid');
      if (grid) grid.addEventListener('click', e => {
        const card = e.target.closest('[data-db]'); if (!card) return;
        const i = +card.dataset.db;
        CROSS_DBS[i][1] = !CROSS_DBS[i][1];
        const on = CROSS_DBS[i][1];
        card.classList.toggle('sel', on);
        const mk = card.querySelector('.db-mk');
        mk.className = `db-mk pill ${on ? 'ok' : 'dashed'}`;
        mk.innerHTML = on ? '<span class="dot"></span>selected' : 'add';
        card.querySelector('span[style*="flex:none"]').style.color = on ? 'var(--accent-ink)' : 'var(--ink-4)';
        const sel = CROSS_DBS.filter(d => d[1]).length;
        const cnt = root.querySelector('#dbcount'); if (cnt) cnt.textContent = sel;
        const hint = root.querySelector('#runhint'); if (hint) hint.textContent = `${sel} of 6 · need ≥ 2`;
        root.querySelectorAll('[data-run]').forEach(b => { if (sel < 2) b.setAttribute('aria-disabled', 'true'); else b.removeAttribute('aria-disabled'); });
      });
      root.querySelectorAll('[data-run]').forEach(b => {
        if (b.dataset.crossdbRunBound === '1') return;
        b.dataset.crossdbRunBound = '1';
        b.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        if (b.getAttribute('aria-disabled') === 'true' || crossView === 'loading') return;
        b.setAttribute('aria-disabled', 'true');
        crossView = 'loading'; repaintScreen('crossdb');
        if (window.EU_DATA === 'real') {
          const rawRootInput = root.querySelector('[data-crossdb-root]');
          let rawRoot = rawRootInput && rawRootInput.value ? rawRootInput.value.trim() : '';
          if (!rawRoot) {
            try { rawRoot = localStorage.getItem('easyicu_crossdb_data_root') || localStorage.getItem('easyicu_raw_data_root') || ''; } catch (err) {}
          }
          loadRealCrossdb(() => { crossView = 'idle'; repaintScreen('crossdb'); }, { rawRoot });
        } else {
          loadDemoCrossdb(ok => { crossView = ok ? 'loaded' : 'idle'; repaintScreen('crossdb'); });
        }
        });
      });
      root.querySelectorAll('[data-crossdb-root]').forEach(input => {
        if (input.dataset.crossdbRootBound === '1') return;
        input.dataset.crossdbRootBound = '1';
        input.addEventListener('input', () => {
          try { localStorage.setItem('easyicu_crossdb_data_root', (input.value || '').trim()); } catch (err) {}
        });
        input.addEventListener('change', () => {
          try { localStorage.setItem('easyicu_crossdb_data_root', (input.value || '').trim()); } catch (err) {}
        });
      });
      root.querySelectorAll('[data-crossdb-cancel]').forEach(b => b.addEventListener('click', cancelCrossRawJob));
      root.querySelectorAll('[data-crossdb-export]').forEach(b => b.addEventListener('click', () => {
        const payload = window.EU_CROSSDB_WORKSPACE;
        if (!payload) {
          vizErr = 'No Cross-DB payload is loaded yet.';
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
        crossView = 'idle'; crossDensityModule = 'all'; crossDensityFeature = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; repaintScreen('crossdb');
      }));
    },
  };
})();
