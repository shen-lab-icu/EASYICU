/* Screen: Workspace States
   A high-fidelity, clickable catalogue of the app's global states —
   loading / empty / no-data / error / blocked / success — shown in a real
   workspace frame, switchable across Demo + Real Data modes and three
   representative contexts (Patient Review, Cross-database comparison, Agent run).
   Pure design reference: every "result" is clearly a demo/seeded example. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  /* module-level UI state for the live stage */
  let stMode = 'demo';      // demo | real
  let stState = 'success';  // loading | empty | nodata | error | blocked | success
  let stCtx = 'patient';    // patient | crossdb | agent

  const STATES = [
    ['loading', 'Loading', 'refresh'],
    ['empty', 'Empty', 'layers'],
    ['nodata', 'No data', 'list'],
    ['error', 'Error', 'beaker'],
    ['blocked', 'Blocked', 'lock'],
    ['success', 'Success', 'check'],
  ];

  /* per-context copy + shapes */
  const CTX = {
    patient: {
      label: 'Patient Review', icon: 'patient', noun: 'review workspace', unit: 'stays',
      crumb: 'Data Visualization · Patient Review',
      load: { demo: 'Generating demo review data', real: 'Reading local export folder' },
      empty: {
        t: 'No review workspace loaded yet',
        d: { demo: 'Generate a compact demo set to populate tables, time series, patient overview, and quality checks.',
             real: 'Point EasyICU at a local export folder. Files are parsed on your machine — nothing is uploaded.' },
        cta: { demo: 'Generate demo data', real: 'Choose export folder' },
        chips: ['vitals', 'labs', 'sofa', 'sepsis-3', 'outcomes'],
      },
      nodata: {
        t: 'Cohort matched 0 stays',
        d: 'The current filters returned no patients. Loosen a constraint or widen the time window, then re-run.',
        filters: ['age ≥ 80', 'sepsis-3 = true', 'LOS > 14d', 'vasopressor = yes'],
      },
      error: {
        t: { demo: 'Demo generation failed', real: 'Couldn’t read the export folder' },
        d: { demo: 'The demo seed produced an inconsistent frame. Retrying re-seeds with a fresh value.',
             real: 'The selected folder doesn’t match a known ICU export layout. Pick a recognised database root.' },
        detail: { demo: ['$ easyicu demo --seed 42 --patients 10', 'ValueError: vitals frame length 0 != expected 240', 'hint: re-run regenerates the seed'],
                  real: ['$ scan /data/icu_export/', 'SchemaError: no recognised layout in folder', 'expected one of: MIMIC-IV · eICU · AUMC · HiRID'] },
      },
      success: {
        t: 'Review workspace loaded',
        stats: [['Stays', '10'], ['Time points', '240'], ['Modules', '19'], ['Coverage', '94%']],
        toast: ['Demo workspace ready', '10 stays · 19 modules · 0 errors'],
      },
    },
    crossdb: {
      label: 'Cross-database comparison', icon: 'benchmark', noun: 'comparison', unit: 'databases',
      crumb: 'Data Visualization · Cross-database comparison',
      load: { demo: 'Loading seeded frames for 6 databases', real: 'Connecting to selected ICU databases' },
      empty: {
        t: 'Select at least 2 databases',
        d: { demo: 'Cross-database comparison checks coverage and aggregate distributions across databases. Add a second source to begin.',
             real: 'Connect two or more local database roots to compare standardized concepts side by side.' },
        cta: { demo: 'Load demo databases', real: 'Connect databases' },
        chips: ['MIMIC-IV', 'eICU-CRD', 'AUMC', 'HiRID', 'SICdb'],
      },
      nodata: {
        t: 'No shared concepts across selection',
        d: 'The chosen databases have no overlapping standardized concepts for this cohort. Adjust the concept set or selection.',
        filters: ['MIMIC-IV', 'SICdb', 'concept = lactate', 'window = 6h'],
      },
      error: {
        t: { demo: 'Couldn’t assemble benchmark', real: 'Database connection failed' },
        d: { demo: 'One seeded frame failed to materialise. Retry rebuilds the comparison set.',
             real: 'A selected database root is unreadable or its concept map is missing. Re-check the path.' },
        detail: { demo: ['$ benchmark --dbs 6 --concepts 6', 'RuntimeError: frame "HiRID" returned empty', 'partial result discarded'],
                  real: ['$ connect /data/sicdb/', 'IOError: concepts/ not found at root', 'expected: <root>/concepts/*.parquet'] },
      },
      success: {
        t: 'Benchmark assembled',
        stats: [['Databases', '6'], ['Concepts', '6'], ['Δ range max', '15.5'], ['Rows / db', '144']],
        toast: ['Benchmark ready', '6 databases · 6 shared concepts'],
      },
    },
    agent: {
      label: 'Research Agent', icon: 'agent', noun: 'agent run', unit: 'steps',
      crumb: 'Research Agent · Run',
      load: { demo: 'Running demo pipeline (no tokens)', real: 'Executing plan · evidence-bound run' },
      empty: {
        t: 'No run yet',
        d: { demo: 'Demo Mode produces a static, reviewable gallery without calling a model. Confirm a plan to preview it.',
             real: 'Define a research question and confirm the preflight gate. No model call happens until you approve.' },
        cta: { demo: 'Preview demo run', real: 'Open plan setup' },
        chips: ['plan', 'build', 'analyze', 'gate', 'review'],
      },
      nodata: {
        t: 'Run produced no evidence artifacts',
        d: 'Every step completed but no artifact passed the evidence contract. Review step inputs before drafting.',
        filters: ['cohort = sepsis · 10', 'checks = coverage', 'gate = strict'],
      },
      error: {
        t: { demo: 'Demo pipeline halted', real: 'Run failed at analysis step' },
        d: { demo: 'A demo step contract could not be satisfied. Retry replays the deterministic pipeline.',
             real: 'Step 4 (LR + SOFA + lactate) raised an exception. The draft stays locked; the run is recoverable.' },
        detail: { demo: ['step 04 · model: LR + SOFA + lactate', 'AssertionError: design matrix has NaNs', 'auto-repair: off'],
                  real: ['step 04 · model: LR + SOFA + lactate', 'LinAlgError: singular matrix', 'evidence ledger: 3 of 6 steps logged'] },
      },
      success: {
        t: 'Run complete · awaiting review',
        stats: [['Steps', '6 / 6'], ['Figures', '6'], ['Tables', '3'], ['Duration', '2m 14s']],
        toast: ['Run 07 complete', '6 artifacts · draft gated on review'],
      },
    },
  };

  /* ----- shared stage chrome ----- */
  function modePill() {
    return stMode === 'demo'
      ? `<span class="pill demo"><span class="dot"></span>Demo</span>`
      : `<span class="pill real"><span class="dot"></span>Real · local</span>`;
  }
  function statePill(s) {
    const map = {
      loading: ['', 'Loading'], empty: ['dashed', 'Empty'], nodata: ['', 'No data'],
      error: ['bad', 'Error'], blocked: ['warn', 'Blocked'], success: ['ok', 'Success'],
    };
    const [cls, lab] = map[s];
    const dot = cls === 'dashed' ? '' : '<span class="dot"></span>';
    return `<span class="pill ${cls}">${dot}${lab}</span>`;
  }

  /* ----- state bodies ----- */
  function bLoading(c) {
    const msg = c.load[stMode];
    const skTable = `
      <div class="sk-table mt-16">
        <div class="sk-trow head">${[42,28,28,28,28].map(w => `<div class="sk sk-line sm" style="width:${w}%"></div>`).join('')}</div>
        ${[0,1,2,3,4].map(() => `<div class="sk-trow">${[70,55,48,52,40].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}
      </div>`;
    const skTasks = `
      <div class="sk-table mt-16">
        ${[0,1,2,3,4,5].map(() => `<div class="sk-trow" style="grid-template-columns:24px 1.6fr auto;"><div class="sk" style="width:18px;height:18px;border-radius:50%"></div><div class="sk sk-line" style="width:60%"></div><div class="sk sk-line sm" style="width:54px"></div></div>`).join('')}
      </div>`;
    return `
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow">
          <div style="font-weight:600;font-size:12.75px;">${msg}…</div>
          <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${stMode === 'demo' ? 'reproducible · no outbound calls' : 'local-only · nothing uploaded'}</div>
        </div>
        <button class="btn sm" aria-disabled="true" tabindex="-1">${icon('stop', 13)} Cancel</button>
      </div>
      <div class="indet mt-12"></div>
      <div class="st-stats mt-16">
        ${[0,1,2,3].map(() => `<div class="sk-stat"><div class="sk sk-line sm" style="width:52%"></div><div class="sk" style="height:22px;width:64%;margin-top:10px;"></div></div>`).join('')}
      </div>
      ${stCtx === 'agent' ? skTasks : skTable}`;
  }

  function bEmpty(c) {
    const e = c.empty;
    return `
      <div class="state-hero empty-state">
        <div class="glyph">${icon(c.icon, 26)}</div>
        <div class="st-t">${e.t}</div>
        <div class="st-d">${e.d[stMode]}</div>
        <div class="st-actions">
          <button class="btn primary lg" aria-disabled="true" tabindex="-1">${icon(stMode === 'demo' ? 'flask' : 'folder', 15)} ${e.cta[stMode]}</button>
          ${stMode === 'demo'
            ? `<button class="btn lg" aria-disabled="true" tabindex="-1">${icon('db', 15)} Switch to real data</button>`
            : `<button class="btn lg" aria-disabled="true" tabindex="-1">${icon('flask', 15)} Try demo instead</button>`}
        </div>
        <div class="st-meta">
          <div class="eyebrow" style="margin-bottom:8px;">${stMode === 'demo' ? 'Included in demo set' : 'Supported sources'}</div>
          <div class="row wrap gap-6" style="justify-content:center;">
            ${e.chips.map(x => `<span class="chip">${x}</span>`).join('')}
          </div>
        </div>
      </div>`;
  }

  function bNoData(c) {
    const n = c.nodata;
    return `
      <div class="state-hero nodata">
        <div class="glyph">${icon('eye', 24)}</div>
        <div class="st-t">${n.t}</div>
        <div class="st-d">${n.d}</div>
        <div class="filter-recap">
          <span class="eyebrow" style="margin-right:2px;">${stCtx === 'agent' ? 'run config' : 'active filters'}</span>
          ${n.filters.map(f => `<span class="chip solid">${f}</span>`).join('')}
        </div>
        <div class="st-actions">
          <button class="btn primary" aria-disabled="true" tabindex="-1">${icon('sliders', 14)} ${stCtx === 'agent' ? 'Review inputs' : 'Adjust filters'}</button>
          <button class="btn" aria-disabled="true" tabindex="-1">${icon('refresh', 14)} Reset to defaults</button>
        </div>
      </div>`;
  }

  function bError(c) {
    const er = c.error;
    const lines = er.detail[stMode];
    return `
      <div class="state-hero error solid">
        <div class="glyph">${icon('beaker', 24)}</div>
        <div class="st-t">${er.t[stMode]}</div>
        <div class="st-d">${er.d[stMode]}</div>
        <div class="detail-box">${lines.map((l, i) => i === 1 ? `<span class="ln-bad">${l}</span>` : i === 0 ? `<span class="ln-key">${l}</span>` : l).join('\n')}</div>
        <div class="st-actions">
          <button class="btn primary" aria-disabled="true" tabindex="-1">${icon('refresh', 14)} Retry</button>
          <button class="btn" aria-disabled="true" tabindex="-1">${icon('file', 14)} View log</button>
          ${stMode === 'real'
            ? `<button class="btn ghost" aria-disabled="true" tabindex="-1">${icon('flask', 14)} Switch to demo</button>`
            : `<button class="btn ghost" aria-disabled="true" tabindex="-1">${icon('help', 14)} Get help</button>`}
        </div>
      </div>`;
  }

  function bBlocked(c) {
    const checks = [
      ['Cohort denominators resolved', true],
      ['Per-concept coverage ≥ threshold', true],
      [stCtx === 'crossdb' ? 'Shared concept map verified' : 'Table 1 reproduces from manifest', true],
      ['Model card + metrics attached', stCtx !== 'agent'],
      ['Reviewer sign-off', false],
    ];
    const passed = checks.filter(c => c[1]).length;
    return `
      <div class="gate-block">
        <div class="note warn">
          <div class="ico">${icon('lock', 16)}</div>
          <div class="body">
            <div class="t">${stCtx === 'agent' ? 'Manuscript draft is locked until checks pass' : 'Export is locked until evidence checks pass'}</div>
            <div class="d">Drafting and export are intentionally second-stage actions. Every claim must trace to a logged artifact before the gate opens.</div>
          </div>
        </div>
        <div class="checks mt-16">
          ${checks.map(([t, ok]) => `
            <div class="check-row ${ok ? 'ok' : 'pending'}">
              <span class="check-mk">${ok ? icon('check', 12, 2.8) : icon('clock', 12)}</span>
              <span style="font-size:12.75px;color:${ok ? 'var(--ink)' : 'var(--ink-3)'};font-weight:${ok ? 500 : 400};">${t}</span>
              <span class="grow"></span>
              <span class="mono" style="font-size:10.5px;color:${ok ? 'var(--ok)' : 'var(--ink-4)'};">${ok ? 'passed' : 'pending'}</span>
            </div>`).join('')}
        </div>
        <div class="gate-strip mt-16" style="background:var(--surface-2);">
          <span class="pill warn"><span class="dot"></span>${passed} / ${checks.length} checks</span>
          <div class="grow">
            <div style="font-weight:600;font-size:13px;">One reviewer sign-off outstanding</div>
            <div style="font-size:11.5px;color:var(--ink-3);">The action unlocks once a reviewer confirms the findings.</div>
          </div>
          <button class="btn" aria-disabled="true" tabindex="-1">Request review</button>
          <button class="btn" aria-disabled="true">${icon('lock', 13)} ${stCtx === 'agent' ? 'Draft' : 'Export'}</button>
        </div>
      </div>`;
  }

  function bSuccess(c) {
    const s = c.success;
    const rows = stCtx === 'crossdb'
      ? [['hr median', '76.6', '80.3', '74.1'], ['sbp median', '125.4', '128.5', '119.9'], ['map median', '85.3', '89.6', '83.2']]
      : stCtx === 'agent'
      ? [['Cohort summary', 'n=10 · 20% mortality', 'done'], ['Table 1', '11 features', 'done'], ['ROC · LR + lactate', 'AUC 0.84', 'done']]
      : [['Age, mean (SD)', '54.8 (16.2)', '—'], ['SOFA, median', '6', '0.08'], ['Lactate, mmol/L', '2.4', '0.12']];
    const tableHead = stCtx === 'crossdb'
      ? ['Metric', 'MIMIC-IV', 'eICU', 'AUMC']
      : stCtx === 'agent'
      ? ['Artifact', 'Result', 'Status']
      : ['Characteristic', 'Value', 'p'];
    return `
      <div class="ok-banner">
        <span class="mk">${icon('check', 13, 2.8)}</span>
        <div class="grow"><strong style="font-weight:600;">${s.t}.</strong> <span style="color:var(--ink-3);">${stMode === 'demo' ? 'Seeded demo output — values are illustrative, not a real run.' : 'Local run — results stayed on your machine.'}</span></div>
      </div>
      <div class="st-stats mt-16">
        ${s.stats.map(([l, v], i) => `<div class="stat ${i === 0 ? 'ok' : 'accent'}"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('')}
      </div>
      <div class="table-wrap mt-16">
        <table class="eu-table">
          <thead><tr>${tableHead.map((h, i) => `<th class="${i === 0 ? '' : 'num'}">${h}</th>`).join('')}</tr></thead>
          <tbody>
            ${rows.map(r => `<tr><td class="key">${r[0]}</td>${r.slice(1).map((cell) => stCtx === 'agent' && cell === 'done' ? `<td class="num"><span class="pill ok" style="height:20px;"><span class="dot"></span>done</span></td>` : `<td class="num">${cell}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="row gap-8 mt-16">
        <button class="btn primary" aria-disabled="true" tabindex="-1">${icon(stCtx === 'agent' ? 'shield' : 'arrow', 14)} ${stCtx === 'agent' ? 'Open review checks' : 'Open full workspace'}</button>
        <button class="btn" aria-disabled="true" tabindex="-1">${icon('download', 14)} Export bundle</button>
        <span class="grow"></span>
        <span class="mono" style="font-size:10.5px;color:var(--ink-4);align-self:center;">${stMode === 'demo' ? 'demo · no tokens' : 'local · 0 uploads'}</span>
      </div>
      <div class="st-toast">
        <span class="mk">${icon('check', 13, 2.8)}</span>
        <div class="grow"><div class="tt">${s.toast[0]}</div><div class="td">${s.toast[1]}</div></div>
        <span class="x">${icon('stop', 12)}</span>
      </div>`;
  }

  function stageBody(c) {
    switch (stState) {
      case 'loading': return bLoading(c);
      case 'empty': return bEmpty(c);
      case 'nodata': return bNoData(c);
      case 'error': return bError(c);
      case 'blocked': return bBlocked(c);
      case 'success': return bSuccess(c);
    }
  }

  function stageHtml() {
    const c = CTX[stCtx];
    const tight = stState === 'success' || stState === 'loading';
    return `
      <div class="st-stage">
        <div class="st-head">
          <div class="st-mark">${icon(c.icon, 17)}</div>
          <div class="grow">
            <div class="st-title">${c.label}</div>
            <div class="st-sub mono">${c.crumb}</div>
          </div>
          ${modePill()}
          ${statePill(stState)}
        </div>
        <div class="st-body">${stageBody(c)}</div>
      </div>`;
  }

  function controls() {
    const ctxBtns = Object.entries(CTX).map(([k, c]) =>
      `<button class="${stCtx === k ? 'active' : ''}" data-ctx="${k}">${icon(c.icon, 14)} ${c.label}</button>`).join('');
    const modeBtns = `
      <button class="${stMode === 'demo' ? 'active' : ''}" data-mode="demo">${icon('flask', 14)} Demo</button>
      <button class="${stMode === 'real' ? 'active' : ''}" data-mode="real">${icon('db', 14)} Real Data</button>`;
    const stateBtns = STATES.map(([k, lab, ic]) =>
      `<button class="${stState === k ? 'active' : ''}" data-state="${k}">${icon(ic, 13)} ${lab}</button>`).join('');
    return `
      <div class="st-controls">
        <div class="ctl-group"><span class="ctl-lbl">Context</span><div class="seg" id="ctxSeg">${ctxBtns}</div></div>
        <div class="st-divider"></div>
        <div class="ctl-group"><span class="ctl-lbl">Mode</span><div class="seg" id="modeSeg">${modeBtns}</div></div>
        <span class="grow"></span>
        <div class="seg state" id="stateSeg">${stateBtns}</div>
      </div>`;
  }

  function primitives() {
    return `
      <div class="sec-stack"><div class="lbl">Status primitives</div><h2>Reusable building blocks</h2></div>
      <div class="prim-grid">
        <div class="prim">
          <div class="pk">Stat tile</div>
          <div class="pdemo" style="display:block;">
            <div class="stat accent" style="padding:10px 12px;"><div class="label">Mortality</div><div class="val" style="font-size:20px;margin-top:2px;">20.0%</div></div>
          </div>
        </div>
        <div class="prim">
          <div class="pk">Coverage bar</div>
          <div class="pdemo" style="display:block;">
            <div class="row" style="justify-content:space-between;font-size:11.5px;"><span>Labs</span><span class="mono" style="color:var(--ink-4);">94%</span></div>
            <div class="runbar" style="height:7px;margin-top:6px;"><div class="runbar-fill" style="width:94%;background:var(--ok);"></div></div>
          </div>
        </div>
        <div class="prim">
          <div class="pk">Data row</div>
          <div class="pdemo" style="display:block;font-family:var(--font-mono);font-size:11px;">
            <div class="row" style="justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--hair);"><span style="color:var(--ink-4);">stay_id</span><span>20001</span></div>
            <div class="row" style="justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--hair);"><span style="color:var(--ink-4);">lactate</span><span>2.1</span></div>
            <div class="row" style="justify-content:space-between;padding:4px 0;"><span style="color:var(--ink-4);">sofa</span><span>5</span></div>
          </div>
        </div>
        <div class="prim">
          <div class="pk">Status pills</div>
          <div class="pdemo">
            <span class="pill ok"><span class="dot"></span>passed</span>
            <span class="pill warn"><span class="dot"></span>blocked</span>
            <span class="pill bad"><span class="dot"></span>error</span>
            <span class="pill dashed">queued</span>
          </div>
        </div>
        <div class="prim">
          <div class="pk">Callouts</div>
          <div class="pdemo" style="display:block;">
            <div class="note ok" style="padding:8px 11px;"><div class="ico">${icon('check', 14)}</div><div class="body"><span class="t" style="font-size:12px;">Success</span></div></div>
            <div class="note bad" style="padding:8px 11px;margin-top:8px;background:var(--bad-soft);border-color:oklch(88% 0.05 25);"><div class="ico" style="color:var(--bad);">${icon('beaker', 14)}</div><div class="body"><span class="t" style="font-size:12px;">Error</span></div></div>
          </div>
        </div>
        <div class="prim">
          <div class="pk">Empty glyph</div>
          <div class="pdemo" style="justify-content:center;">
            <div class="glyph" style="width:40px;height:40px;border-radius:var(--r-2);background:var(--accent-soft);color:var(--accent-ink);display:grid;place-items:center;border:1px solid var(--accent-border);">${icon('layers', 18)}</div>
          </div>
        </div>
      </div>`;
  }

  S.states = {
    section: 'states', nav: 'states',
    crumbs: ['Home', 'Workspace States'],
    actionHtml: `<span class="pill">${icon('eye', 13)} Reference</span>`,
    rail() {
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">States</span><span class="pill" style="height:20px;">6</span></div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6">${icon('refresh', 13)} Loading · skeletons</div>
          <div class="row gap-6">${icon('layers', 13)} Empty · first run</div>
          <div class="row gap-6">${icon('list', 13)} No data · 0 results</div>
          <div class="row gap-6" style="color:var(--bad);">${icon('beaker', 13)} Error · recoverable</div>
          <div class="row gap-6" style="color:oklch(50% 0.11 70);">${icon('lock', 13)} Blocked · gated</div>
          <div class="row gap-6" style="color:var(--ok);">${icon('check', 13)} Success · loaded</div>
        </div>
        <div class="eyebrow mt-16" style="margin-bottom:8px;">Principles</div>
        <div class="col gap-6" style="font-size:11px;color:var(--ink-3);">
          <div class="row gap-6">${icon('shield', 13)} No fake results</div>
          <div class="row gap-6">${icon('arrow', 13)} Always one clear action</div>
        </div>
      </div>`;
    },
    render() {
      return `
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">Design system · 状态库</div>
        <h1 style="margin-top:6px;">Workspace states</h1>
        <p class="lead">Every data surface in EasyICU passes through the same six states. Switch context, mode, and state to preview the polished treatment for each — all reference-only, with no invented results.</p>
      </div>
      ${controls()}
      <div id="stStage">${stageHtml()}</div>`;
    },
    afterRender(root) {
      const stage = root.querySelector('#stStage');
      function repaint() {
        stage.innerHTML = stageHtml();
        // refresh active classes on control segs
        root.querySelectorAll('#ctxSeg button').forEach(b => b.classList.toggle('active', b.dataset.ctx === stCtx));
        root.querySelectorAll('#modeSeg button').forEach(b => b.classList.toggle('active', b.dataset.mode === stMode));
        root.querySelectorAll('#stateSeg button').forEach(b => b.classList.toggle('active', b.dataset.state === stState));
        wireToast();
      }
      function wireToast() {
        const x = stage.querySelector('.st-toast .x');
        if (x) x.addEventListener('click', () => { const t = stage.querySelector('.st-toast'); if (t) t.remove(); });
      }
      root.querySelector('#ctxSeg').addEventListener('click', (e) => {
        const b = e.target.closest('[data-ctx]'); if (!b) return; stCtx = b.dataset.ctx; repaint();
      });
      root.querySelector('#modeSeg').addEventListener('click', (e) => {
        const b = e.target.closest('[data-mode]'); if (!b) return; stMode = b.dataset.mode; repaint();
      });
      root.querySelector('#stateSeg').addEventListener('click', (e) => {
        const b = e.target.closest('[data-state]'); if (!b) return; stState = b.dataset.state; repaint();
      });
      wireToast();
    },
  };
})();
