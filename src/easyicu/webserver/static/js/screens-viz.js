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
    const dataset = xdb ? `${fmtInt(xdb.source_count)} exports` : (drill ? ((drill.source || {}).label || 'Local export') : (cohort ? ((cohort.source || {}).label || 'Local export') : (ws ? ((ws.path || '').split('/').filter(Boolean).slice(-2).join('/') || 'Local export') : (real ? 'No export loaded' : 'Demo · 10 patients'))));
    const cohortLine = xdb ? 'matched exports required' : (drill ? `${fmtInt(drill.summary && drill.summary.entities)} entities` : (cohort ? `${fmtInt(cohort.summary && cohort.summary.cohort_size)} entities` : (ws ? `${fmtInt(ws.summary && ws.summary.stays)} stays` : (real ? 'load exported tables' : 'demo defaults'))));
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
  let cohortView = 'loaded';  // loaded | loading
  let cohortPanel = 'groups'; // groups | coverage | snapshot | sofa
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
    return '/Users/haibo/easyicu/exports/miiv';
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
    if (sum.modules != null) parts.push(`${fmtInt(sum.modules)} modules`);
    if (sum.total_rows != null) parts.push(`${fmtInt(sum.total_rows)} rows`);
    return parts.join(' · ') || 'export folder';
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
          <input class="pf-input" data-src-path-input type="text" spellcheck="false" autocomplete="off" placeholder="${esc(defaultExportPath())}" aria-label="EasyICU export path" />
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
        window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_COHORT_REVIEW = null; window.EU_STALE = true;
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
        vizErr = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_COHORT_REVIEW = null; crossView = 'idle'; patientView = 'idle'; repaintScreen(screenId);
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
        vizErr = null; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_COHORT_REVIEW = null; crossView = 'idle'; patientView = 'idle'; repaintScreen(screenId);
      }).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-refresh]').forEach(b => b.addEventListener('click', () => {
      if (!(window.EU_API && window.EU_API.hydrateWorkspaceRegistry)) return;
      window.EU_API.hydrateWorkspaceRegistry().then(() => repaintScreen(screenId)).catch(err => { vizErr = String(err && err.message || err); repaintScreen(screenId); });
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
    const body = {};
    if (active) body.source_path = active;
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
  function loadRealCrossdb(done) {
    window.EU_CROSSDB_WORKSPACE = null;
    window.EU_COHORT_REVIEW = null;
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
    loadRealWorkspace(done);
  }

  /* allow the print harness to preset loaded states for a richer PDF */
  window.__euVizPreset = function (which) {
    if (!which || which === 'patient') patientView = 'loaded';
    if (!which || which === 'crossdb') crossView = 'loaded';
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
      const rows = [
        ['Entities', fmtInt(s.entities), 'cohort denominator from active export'],
        ['Mean age', fmtNum(s.mean_age, 1), 'demographics aggregate'],
        ['Female', fmtPct(s.female_pct), 'demographics aggregate'],
        ['Mortality', fmtPct(s.mortality), 'outcome aggregate'],
        ['Median SOFA-2', fmtNum(s.median_sofa2, 1), 'score aggregate'],
        ['Sepsis-3 positive', fmtPct(s.sepsis_pct), 'event aggregate'],
      ];
      return `
      <div class="st-stats mt-16">
        ${[
          ['Entities', fmtInt(s.entities), 'ok'],
          ['Mean age', fmtNum(s.mean_age, 1), 'accent'],
          ['Mortality', fmtPct(s.mortality), 'accent'],
          ['Median SOFA-2', fmtNum(s.median_sofa2, 1), 'accent'],
        ].map(([l, v, c]) => `<div class="stat ${c}"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('')}
      </div>
      <div class="table-wrap table-scroll mt-16">
        <table class="eu-table">
          <thead><tr><th>Aggregate</th><th class="num">Value</th><th>Basis</th></tr></thead>
          <tbody>
            ${rows.map(r => `<tr><td class="key">${esc(r[0])}</td><td class="num">${esc(r[1])}</td><td>${esc(r[2])}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">Row table blocked</span> <span class="d" style="display:inline;">— native Patient Review exposes cohort aggregates and one pseudonymous entity drilldown. Direct identifier tables stay out of the browser payload.</span></div>
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
    const signals = drill && drill.selected ? (drill.selected.signals || []) : [];
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
            <div style="height:36px;">${spark(s.values || [], 520, 36, palette[i % palette.length])}</div>
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
            <div style="height:36px;">${spark(s.values || [], 520, 36, palette[i % palette.length])}</div>
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
            <div style="height:36px;">${spark(vals, 520, 36, col)}</div>
          </div>`).join('')}
      </div>`;
  }

  function ptPatient() {
    const drill = patientDrilldown();
    if (drill && drill.selected) {
      const selected = drill.selected || {};
      const demo = selected.demographics || {};
      const scores = selected.scores || {};
      const outcomes = selected.outcomes || {};
      const signals = selected.signals || [];
      const entities = drill.entities || [];
      return `
      <div class="row wrap gap-6 mt-16">
        <span class="eyebrow" style="align-self:center;margin-right:4px;">Select entity</span>
        ${entities.map(item => `<button type="button" class="chip ${item.ref === selected.ref ? 'solid' : ''}" data-patient-entity="${esc(item.ref)}" style="${item.ref === selected.ref ? 'border-color:var(--ink);color:var(--ink);' : ''}">${esc(item.label || item.ref)}</button>`).join('')}
      </div>
      <div class="split-320 mt-16" style="grid-template-columns:300px 1fr;">
        <div class="card pad">
          <div class="eyebrow">Patient summary</div>
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
          <div class="mc-head"><div style="font-weight:600;font-size:13px;">Vitals · ${esc(selected.label || 'selected entity')}</div><span class="mono" style="font-size:10.5px;color:var(--ink-4);">local export · bounded</span></div>
          <div class="col gap-12 mt-8">
            ${signals.slice(0, 4).map((s, i) => `
              <div class="row gap-12" style="align-items:center;"><span class="mono" style="font-size:11px;color:var(--ink-3);width:42px;">${esc(s.key || s.name)}</span><div style="flex:1;height:30px;">${spark(s.values || [], 440, 30, ['var(--accent)', 'var(--accent)', 'var(--ok)', 'var(--warn)'][i % 4])}</div></div>`).join('') || '<div style="font-size:12px;color:var(--ink-4);">No vitals trend available in this export.</div>'}
          </div>
        </div>
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><span class="t">Pseudonymous drilldown</span> <span class="d" style="display:inline;">— entity refs are one-way browser tokens for the active local export; direct clinical identifiers are not returned.</span></div>
      </div>`;
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
              <div class="row gap-12" style="align-items:center;"><span class="mono" style="font-size:11px;color:var(--ink-3);width:42px;">${esc(s.key || s.name)}</span><div style="flex:1;height:30px;">${spark(s.values || [], 440, 30, ['var(--accent)', 'var(--accent)', 'var(--ok)', 'var(--warn)'][i % 4])}</div></div>`).join('') || '<div style="font-size:12px;color:var(--ink-4);">No vitals trend available in this export.</div>'}
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
              <div class="row gap-12" style="align-items:center;"><span class="mono" style="font-size:11px;color:var(--ink-3);width:42px;">${n}</span><div style="flex:1;height:30px;">${spark(vals, 440, 30, col)}</div></div>`).join('')}
          </div>
        </div>
      </div>`;
  }

  function ptQuality() {
    const drill = patientDrilldown();
    if (drill && Array.isArray(drill.quality)) {
      return `
      <div class="card pad mt-16">
        <div class="eyebrow" style="margin-bottom:6px;">Per-module entity coverage</div>
        ${drill.quality.map(q => `
          <div class="qrow"><span>${esc(q.module)}</span><div class="qbar ${q.quality_status === 'ok' ? '' : q.quality_status}"><span style="width:${q.coverage_pct == null ? 0 : Math.max(0, Math.min(100, q.coverage_pct))}%"></span></div><span class="qv">${q.coverage_pct == null ? fmtInt(q.rows) : fmtPct(q.coverage_pct)}</span></div>`).join('')}
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('shield', 16)}</div>
        <div class="body"><div class="t">Local export bounded review</div><div class="d">Coverage uses module entity presence over the active export denominator. Formal denominators and claims remain locked to the evidence-bound agent path.</div></div>
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
        return `
        <div class="loaded-bar">
          <span class="pill ok"><span class="dot"></span>Loaded</span>
          <div class="grow"><span style="font-weight:600;font-size:13px;">${readyTitle}</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${readyStats}</span></div>
          <button class="btn sm" data-viz-reset>${icon('sliders', 13)} Edit setup</button>
          <button class="btn sm">${icon('download', 13)} Export</button>
        </div>
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
            <label class="radio ${window.EU_DATA === 'real' ? 'on' : ''}"><span class="mk"></span> Previously exported data</label>
            <label class="radio ${window.EU_DATA !== 'real' ? 'on' : ''}"><span class="mk"></span> Demo data</label>
          </div>
        </div>

        <div class="card sunken pad mt-16">
          <div class="eyebrow" style="margin-bottom:4px;">${window.EU_DATA === 'real' ? 'Local export' : 'Demo review'}</div>
          <div style="font-weight:600;font-size:14px;">${window.EU_DATA === 'real' ? 'Load exported EasyICU tables' : 'Generate a lightweight demo review workspace'}</div>
          <div class="panel-sub" style="margin-top:2px;">${window.EU_DATA === 'real' ? 'Pick a registered local export, or add one by path.' : 'Loads a fast core ICU concept set for tables, trends, patient overview, and quality checks.'}</div>
          ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
          ${window.EU_DATA === 'real' ? `
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
      case 'coverage': return review ? cohortCoverageBody(review) : (window.EUAudit ? window.EUAudit.panel() : '');
      case 'sofa':     return review ? cohortSofaBody(review) : (window.EUSofa ? window.EUSofa.panel() : '');
      case 'snapshot': return cohortSnapshotBody();
      default:         return cohortGroupsBody();
    }
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
    const blocked = review.sofa_reclassification || {};
    const bins = sofa.bins || [];
    const maxBin = Math.max(1, ...bins.map(b => b.count || 0));
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
      <div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">Paired reclassification blocked</div><div class="d">${esc(blocked.reason || 'Paired SOFA-1/SOFA-2 reclassification is not available in Stage17.')}</div></div></div>`;
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
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">Real registered export aggregate. Row-level filters, p-values, matched cohorts, and paired SOFA reclassification remain blocked.</p>`;
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

      <div class="sec-stack"><div class="lbl">Summary</div><h2>Descriptive group splits</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Comparison</th><th>Group</th><th class="num">Count</th><th class="num">Percent</th><th>Status</th></tr></thead>
          <tbody>
            ${supported.flatMap(row => (row.groups || []).map(g => `<tr>
              <td class="key">${esc(row.label)}</td>
              <td>${esc(g.label)}</td>
              <td class="num">${fmtInt(g.count)}</td>
              <td class="num">${fmtPct(g.pct)}</td>
              <td><span class="pill ok" style="height:20px;">descriptive</span></td>
            </tr>`)).join('')}
          </tbody>
        </table>
      </div>

      <div class="sec-stack"><div class="lbl">Fail-closed</div><h2>Blocked cohort functions</h2></div>
      <div class="cols-3">
        ${blocked.map(item => `<div class="stat"><div class="label">${esc(item.id)}</div><div class="val" style="font-size:13px;line-height:1.35;font-family:var(--font-body);font-weight:600;">${esc(item.status)}</div><div style="font-size:11px;color:var(--ink-4);margin-top:6px;">${esc(item.reason)}</div></div>`).join('')}
      </div>
      <p style="font-size:11px;color:var(--ink-4);margin-top:8px;">No row-level filters, p-values, SMDs, matched cohort, or paired SOFA reclassification are exposed by Stage17.</p>`;
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
        <label class="radio on"><span class="mk"></span> Survived vs Deceased</label>
        <label class="radio"><span class="mk"></span> Age Groups</label>
        <label class="radio"><span class="mk"></span> Male vs Female</label>
        <label class="radio"><span class="mk"></span> Short vs Long Stay</label>
        <label class="radio"><span class="mk"></span> Sepsis vs Non-sepsis</label>
        <label class="radio"><span class="mk"></span> Custom Threshold</label>
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

      <div class="sec-stack"><div class="lbl">Summary</div><h2>Group overview</h2></div>
      <div class="cols-3">
        <div class="stat"><div class="label">Survived</div><div class="val">8</div></div>
        <div class="stat"><div class="label">Deceased</div><div class="val">2</div></div>
        <div class="stat accent"><div class="label">Ratio</div><div class="val">80.0% / 20.0%</div></div>
      </div>

      <div class="sec-stack"><div class="lbl">Table one</div><h2>Baseline characteristics comparison</h2></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Characteristic</th><th class="num">Overall (n=10)</th><th class="num">Survived (n=8)</th><th class="num">Deceased (n=2)</th><th class="num">p-value</th></tr></thead>
          <tbody>
            ${[
              ['Age, mean (SD)','54.8 (16.2)','52.1 (15.4)','65.5 (17.0)','0.31'],
              ['Male, n (%)','3 (30.0)','3 (37.5)','0 (0.0)','0.47'],
              ['SOFA, median','6','5','11','0.08'],
              ['Lactate, mmol/L','2.4','2.1','4.8','0.12'],
              ['ICU LOS, days','5.6','5.1','8.4','0.22'],
            ].map(r => `<tr><td class="key">${r[0]}</td>${r.slice(1).map(c => `<td class="num">${c}</td>`).join('')}</tr>`).join('')}
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
    ['MIMIC-IV', true], ['eICU-CRD', true], ['AmsterdamUMCdb', false],
    ['HiRID', false], ['MIMIC-III', false], ['SICdb', false],
  ];

  function crossHeader() {
    const ws = window.EU_VIZ_WORKSPACE;
    const xdb = window.EU_CROSSDB_WORKSPACE;
    return `
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>Workspace</span> ${icon('chevron', 11)} <span>${xdb ? 'Local exports' : (ws ? 'Local export' : 'Demo cohort')}</span> ${icon('chevron', 11)} <span style="color:var(--ink-2);">Cross-DB benchmark</span>
      </div>
      <div class="page-head" style="margin-bottom:14px;">
        <h1 style="margin-top:0;">Cross-DB benchmark</h1>
        <p class="lead">${window.EU_DATA === 'real' ? 'Load two or more local exports before comparing databases.' : 'Same cohort definition compared across ≥2 ICU databases.'}</p>
      </div>`;
  }
  function crossFmt(key, value) {
    if (value == null) return '—';
    if (key === 'stays' || key === 'cohort_size' || key === 'modules' || key === 'total_rows' || key === 'total_records') return fmtInt(value);
    if (key === 'female_pct' || key === 'mortality' || key === 'mortality_pct' || key === 'sepsis_pct' || key === 'coverage_median_pct') return fmtPct(value);
    return fmtNum(value, 1);
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
    return `
      <div class="loaded-bar">
        <span class="pill ok"><span class="dot"></span>Loaded</span>
        <div class="grow"><span style="font-weight:600;font-size:13px;">Real cross-database benchmark ready</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${fmtInt(sources.length)} exports · ${fmtInt(shared.length)} shared modules</span></div>
        <button class="btn sm" data-viz-reset>${icon('sliders', 13)} Change selection</button>
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('benchmark', 16)}</div>
        <div class="body"><span class="t">Registered export comparison</span> <span class="d" style="display:inline;">— Cross-DB aggregate-only payload from the local source registry. Matched cohort definitions and formal claims still require the evidence-bound agent path.</span></div>
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
                <div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc((source.database || 'local').toUpperCase())} · path hash ${esc(source.path_hash || '—')}</div>
              </div>
            </div>
          </div>`).join('')}
      </div>
      <div class="sec-stack"><div class="lbl">Loaded cross-database export summary</div></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Metric</th>${labels.map(c => `<th class="num">${esc(c)}</th>`).join('')}<th class="num">Δ range</th></tr></thead>
          <tbody>
            ${(xdb.rows || []).map(row => `<tr><td class="key">${esc(row.label)}</td>${(row.values || []).map(v => `<td class="num">${crossFmt(row.key, v)}</td>`).join('')}<td class="num" style="color:var(--ink-3);">${crossFmt(row.key, row.delta)}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
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

  function crossLoaded(selIdx) {
    const allCols = ['MIMIC-IV', 'eICU-CRD', 'AMSTERDAMUMCDB', 'HIRID', 'MIMIC-III', 'SICDB'];
    const allRows = [
      ['Feature rows', [144, 144, 144, 144, 144, 144], 0],
      ['Concepts present', [6, 6, 6, 6, 6, 6], 0],
      ['hr median', [76.55, 80.28, 74.09, 80.10, 80.22, 75.70], 2],
      ['sbp median', [125.36, 128.48, 119.92, 126.91, 129.89, 114.42], 2],
      ['map median', [85.28, 89.56, 83.15, 79.76, 84.39, 81.87], 2],
      ['temp median', [37.23, 37.10, 37.50, 37.39, 37.13, 37.22], 2],
      ['spo2 median', [96.95, 95.82, 96.69, 97.24, 94.80, 97.15], 2],
    ];
    const concepts = ['hr', 'sbp', 'map', 'temp', 'spo2', 'lact'];
    const cols = selIdx.map(i => allCols[i]);
    return `
      <div class="loaded-bar">
        <span class="pill ok"><span class="dot"></span>Loaded</span>
        <div class="grow"><span style="font-weight:600;font-size:13px;">Benchmark assembled</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${selIdx.length} databases · 6 shared concepts · 144 rows / db</span></div>
        <button class="btn sm" data-viz-reset>${icon('sliders', 13)} Change selection</button>
        <button class="btn sm">${icon('download', 13)} Export</button>
      </div>
      <div class="note info mt-16">
        <div class="ico">${icon('benchmark', 16)}</div>
        <div class="body"><span class="t">Demo simulated data</span> <span class="d" style="display:inline;">— the summary and matrix below use independent seeded feature frames for each database, not a user database.</span></div>
      </div>
      <div class="sec-stack"><div class="lbl">Loaded cross-database distribution summary</div></div>
      <div class="table-wrap table-scroll">
        <table class="eu-table">
          <thead><tr><th>Metric</th>${cols.map(c => `<th class="num">${c}</th>`).join('')}<th class="num">Δ range</th></tr></thead>
          <tbody>
            ${allRows.map(([name, vals, dp]) => {
              const v = selIdx.map(i => vals[i]);
              const delta = (Math.max(...v) - Math.min(...v));
              return `<tr><td class="key">${name}</td>${v.map(x => `<td class="num">${dp ? x.toFixed(dp) : x}</td>`).join('')}<td class="num" style="color:var(--ink-3);">${dp ? delta.toFixed(dp) : delta}</td></tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>
      <div class="sec-stack"><div class="lbl">Concept availability across databases</div></div>
      <div class="table-wrap table-scroll" style="padding:14px 16px;">
        <div class="avail-grid" style="grid-template-columns:80px repeat(${cols.length}, 1fr);min-width:${120 + cols.length * 70}px;">
          <div></div>
          ${cols.map(c => `<div class="mono avail-h">${c}</div>`).join('')}
          ${concepts.map(cp => `
            <div class="mono avail-row">${cp}</div>
            ${cols.map(() => `<div class="avail-cell ok">${icon('check', 12, 2.6)}</div>`).join('')}
          `).join('')}
        </div>
      </div>`;
  }

  S.crossdb = {
    section: 'viz', nav: 'viz', sub: 'crossdb', wide: true,
    crumbs: ['Home', 'Data Visualization', 'Cross-DB Benchmark'],
    get actionHtml() {
      return crossView === 'loaded' || (window.EU_DATA === 'real' && window.EU_CROSSDB_WORKSPACE)
        ? `<button class="btn" data-viz-reset>${icon('sliders', 13)} Change selection</button><button class="btn primary" data-run>${icon('refresh', 13)} Re-run</button>`
        : `<button class="btn primary" data-run ${crossView === 'loading' ? 'aria-disabled="true"' : ''}>${icon('play', 13)} Run</button>`;
    },
    rail: () => vizRail('crossdb'),
    render() {
      if (window.EU_DATA === 'real') {
        const xdb = window.EU_CROSSDB_WORKSPACE;
        const ws = window.EU_VIZ_WORKSPACE;
        if (xdb) {
          return crossHeader() + crossRealLoaded(xdb);
        }
        if (!ws) {
          return crossHeader() + `<div class="card pad" style="max-width:720px;">
            <div class="panel-title" style="font-size:17px;">Select local exports</div>
            <div class="panel-sub mt-4">Cross-DB Benchmark requires at least two registered EasyICU export folders.</div>
            ${vizErr ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(vizErr)}</div></div></div>` : ''}
            ${sourceRegistryBlock('multi')}
            <button class="btn primary mt-16" data-run>${icon('folder', 14)} Load selected exports</button>
          </div>`;
        }
        return crossHeader() + `
          <div class="loaded-bar">
            <span class="pill warn"><span class="dot"></span>One source loaded</span>
            <div class="grow"><span style="font-weight:600;font-size:13px;">${esc((ws.database || 'local').toUpperCase())} export ready</span> <span class="mono" style="font-size:11px;color:var(--ink-4);">${fmtInt(ws.summary && ws.summary.stays)} stays · ${fmtInt(ws.summary && ws.summary.modules)} modules</span></div>
            <button class="btn sm" data-viz-reset>${icon('sliders', 13)} Change selection</button>
          </div>
          <div class="note warn mt-16">
            <div class="ico">${icon('benchmark', 16)}</div>
            <div class="body"><span class="t">Cross-database comparison is not assembled yet.</span> <span class="d" style="display:inline;">A valid benchmark needs at least two database exports built with the same cohort definition. This screen will stay fail-closed rather than invent a second source.</span></div>
          </div>
          ${sourceRegistryBlock('multi')}
          <div class="sec-stack"><div class="lbl">Loaded source summary</div></div>
          <div class="table-wrap table-scroll">
            <table class="eu-table">
              <thead><tr><th>Metric</th><th class="num">${esc(ws.database || 'local')}</th></tr></thead>
              <tbody>
                ${[
                  ['Stays', fmtInt(ws.summary && ws.summary.stays)],
                  ['Modules', fmtInt(ws.summary && ws.summary.modules)],
                  ['Rows', fmtInt(ws.summary && ws.summary.total_rows)],
                  ['Mean age', fmtNum(ws.summary && ws.summary.mean_age, 1)],
                  ['Mortality', fmtPct(ws.summary && ws.summary.mortality)],
                ].map(r => `<tr><td class="key">${r[0]}</td><td class="num">${r[1]}</td></tr>`).join('')}
              </tbody>
            </table>
          </div>`;
      }
      if (crossView === 'loading') {
        return crossHeader() + `<div class="card pad">
          <div class="load-strip">
            <span class="spin accent"></span>
            <div class="grow"><div style="font-weight:600;font-size:12.75px;">Loading seeded frames for selected databases…</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">local-only · nothing uploaded</div></div>
            <button class="btn sm" data-viz-reset>${icon('stop', 13)} Cancel</button>
          </div>
          <div class="indet mt-12"></div>
          <div class="sk-table mt-16">
            <div class="sk-trow head">${[30,18,18,18,18].map(w => `<div class="sk sk-line sm" style="width:${w}%"></div>`).join('')}</div>
            ${[0,1,2,3,4,5].map(() => `<div class="sk-trow">${[55,40,40,40,40].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}
          </div>
        </div>`;
      }
      if (crossView === 'loaded') {
        const selIdx = CROSS_DBS.map((d, i) => d[1] ? i : -1).filter(i => i >= 0);
        return crossHeader() + crossLoaded(selIdx.length >= 2 ? selIdx : [0, 1]);
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
      root.querySelectorAll('[data-run]').forEach(b => b.addEventListener('click', () => {
        if (b.getAttribute('aria-disabled') === 'true' || crossView === 'loading') return;
        crossView = 'loading'; repaintScreen('crossdb');
        if (window.EU_DATA === 'real') {
          loadRealCrossdb(() => { crossView = 'idle'; repaintScreen('crossdb'); });
        } else {
          setTimeout(() => { crossView = 'loaded'; window.EU_HASWORK = true; repaintScreen('crossdb'); }, 1400);
        }
      }));
      root.querySelectorAll('[data-viz-reset]').forEach(b => b.addEventListener('click', () => {
        crossView = 'idle'; window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; repaintScreen('crossdb');
      }));
    },
  };
})();
