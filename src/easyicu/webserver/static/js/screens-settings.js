/* Screen: Settings — local paths, data mode, privacy, model, language, about.
   Utility page reached from the sidebar gear. All controls are demo-interactive
   (toggles flip, segmented controls switch) with no real persistence. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  function sw(on, extra = '', settingKey = '') {
    const ds = settingKey ? ` data-setting="${settingKey}"` : '';
    return `<span class="switch ${on ? 'on' : ''} ${extra}" role="switch" aria-checked="${on}" tabindex="0"${ds}></span>`;
  }

  // Bound segmented control: options = [[value, label], ...]; current = persisted value.
  function segBound(settingKey, options, current) {
    const btns = options.map(([val, label, cls]) =>
      `<button class="${val === current ? 'active' : ''} ${cls || ''}" data-val="${val}">${label}</button>`).join('');
    return `<div class="seg" data-setting="${settingKey}">${btns}</div>`;
  }

  function S0() { return window.EU_SETTINGS || {}; }

  function row(t, d, ctl) {
    return `<div class="set-row"><div class="sr-main"><div class="sr-t">${t}</div><div class="sr-d">${d}</div></div><div class="sr-ctl">${ctl}</div></div>`;
  }

  S.settings = {
    section: 'settings', nav: 'settings',
    crumbs: ['Home', 'Settings'],
    actionHtml: `<button class="btn">${icon('refresh', 13)} Reset to defaults</button>`,
    rail() {
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">Settings</span></div>
        <div class="set-nav col gap-6" style="font-size:12.5px;">
          ${[['Workspace', 'folder'], ['Data mode', 'flask'], ['Privacy', 'shield'], ['Research Agent', 'agent'], ['Language', 'globe'], ['About', 'help']].map(([t, ic]) =>
            `<a class="nav-item" href="#set-${t.toLowerCase().replace(/\s/g, '-')}" style="height:30px;"><span class="ico">${icon(ic, 15)}</span>${t}</a>`).join('')}
        </div>
        <div class="note ok mt-16" style="padding:10px 12px;">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="t" style="font-size:12px;">Local-first</div><div class="d" style="font-size:11px;">All settings stay on this machine.</div></div>
        </div>
      </div>`;
    },
    render() {
      return `
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">Workspace · 设置</div>
        <h1 style="margin-top:6px;">Settings</h1>
        <p class="lead">Configure how EasyICU reads data, runs the agent, and presents the workspace. Everything is local and reversible.</p>
      </div>

      <div class="sec-stack" id="set-workspace"><div class="lbl">Workspace</div><h2>Local paths</h2></div>
      <div class="card pad">
        ${row('Working directory', 'Where demo runs, caches, and exports are written. Pick any local folder.',
          `<div class="path-field"><span class="pf-ico">${icon('folder', 14)}</span><span class="pf-path">${S0().working_dir || '~/easyicu/workspace'}</span></div><button class="btn">Change</button>`)}
        ${row('Default export folder', 'Destination for code, tables, figures, and the evidence ledger bundle.',
          `<div class="path-field"><span class="pf-ico">${icon('download', 14)}</span><span class="pf-path">${S0().export_dir || '~/easyicu/exports'}</span></div><button class="btn">Change</button>`)}
        ${row('Module-folder mode', 'Reuse a previously exported module folder instead of re-extracting.', sw(true))}
      </div>

      <div class="sec-stack" id="set-data-mode"><div class="lbl">Data mode</div><h2>Defaults for new sessions</h2></div>
      <div class="card pad">
        ${row('Start mode', 'Which mode a new workspace opens in. You can always switch later.',
          segBound('data_mode', [['demo', 'Demo'], ['real', 'Real Data']], S0().data_mode || 'demo'))}
        ${row('Demo patients', 'Default cohort size generated in Demo Mode (10–50).',
          `<div class="seg" data-seg="patients"><button>10</button><button class="active">20</button><button>50</button></div>`)}
        ${row('Demo duration', 'Default hours of hourly time points per stay.',
          `<div class="seg" data-seg="dur"><button class="active">24h</button><button>48h</button><button>168h</button></div>`)}
      </div>

      <div class="sec-stack" id="set-privacy"><div class="lbl">Privacy</div><h2>Local-first guarantees</h2></div>
      <div class="card pad">
        ${row('Local-only mode', 'Patient data never leaves your machine. This guarantee is enforced and cannot be disabled.',
          `<span class="mono" style="font-size:10.5px;color:var(--ink-4);margin-right:4px;">enforced</span>${sw(true, 'locked ok')}`)}
        ${row('Allow outbound model calls', 'Off by default. When on, only the Research Agent prompt and plan — never patient rows — may reach a configured model endpoint.', sw(!!S0().ai_enabled, '', 'ai_enabled'))}
        ${row('Anonymous usage telemetry', 'Share no data. EasyICU collects nothing unless you explicitly opt in.', sw(false))}
        ${row('Cache cohort frames', 'Keep extracted frames on disk to speed up repeat reviews.', sw(true))}
      </div>

      <div class="sec-stack" id="set-research-agent"><div class="lbl">Research Agent</div><h2>Run behavior</h2></div>
      <div class="card pad">
        ${row('Model', 'The local model that drafts plans and narrative. Demo Mode never calls a model.',
          `<div class="seg" data-seg="model"><button class="active">gpt-oss · local</button><button>External endpoint</button></div>`)}
        ${row('Token budget', 'Soft cap per run. Demo runs use zero tokens.',
          `<div class="path-field" style="min-width:120px;"><span class="pf-path" style="text-align:right;">120,000</span></div>`)}
        ${row('Auto-repair steps', 'Deterministically retry a failed analysis step before halting the run.', sw(true))}
        ${row('Evidence gate', 'Strict requires every contract to pass before drafting unlocks.',
          segBound('evidence_gate', [['strict', 'Strict'], ['standard', 'Standard']], S0().evidence_gate || 'strict'))}
      </div>

      <div class="sec-stack" id="set-language"><div class="lbl">Language</div><h2>Language & display</h2></div>
      <div class="card pad">
        ${row('Interface language', 'EasyICU is fully bilingual; labels fit both scripts.',
          segBound('language', [['en', 'English'], ['zh', '中文', 'cn']], S0().language || 'en'))}
        ${row('Density', 'Comfortable adds breathing room; compact maximises rows on screen.',
          `<div class="seg" data-seg="density"><button class="active">Comfortable</button><button>Compact</button></div>`)}
        ${row('Reduce motion', 'Disable shimmer and progress animations.', sw(false))}
      </div>

      <div class="sec-stack" id="set-about"><div class="lbl">About</div><h2>Environment</h2></div>
      <div class="card pad">
        <div class="setup-row"><span class="k">Version</span><span class="vv mono">EasyICU ${(S0().about || {}).version || '—'}</span></div>
        <div class="setup-row"><span class="k">Python</span><span class="vv mono">${(S0().about || {}).python || '—'}</span></div>
        <div class="setup-row"><span class="k">Databases detected</span><span class="vv mono">MIMIC-IV · eICU · AUMC · HiRID · MIMIC-III · SICdb</span></div>
        <div class="setup-row"><span class="k">Workspace</span><span class="vv mono">${S0().working_dir || '~/easyicu/workspace'}</span></div>
        <div class="row gap-8 mt-16">
          <button class="btn sm">${icon('file', 13)} Release notes</button>
          <button class="btn sm">${icon('help', 13)} Documentation</button>
          <button class="btn sm">${icon('download', 13)} Export diagnostics</button>
        </div>
      </div>`;
    },
    afterRender(root) {
      const persist = (key, value) => {
        if (key && window.EU_API && window.EU_API.saveSetting) {
          window.EU_API.saveSetting(key, value).catch(err =>
            console.error('[EasyICU] saveSetting failed', key, err));
        }
      };
      root.querySelectorAll('.switch:not(.locked)').forEach(s => {
        const key = s.getAttribute('data-setting');
        const flip = () => {
          s.classList.toggle('on');
          const on = s.classList.contains('on');
          s.setAttribute('aria-checked', on);
          if (key) persist(key, on);          // bound switch -> persist
        };
        s.addEventListener('click', flip);
        s.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); flip(); } });
      });
      // demo-only segmented controls (data-seg) keep their local-only behavior
      root.querySelectorAll('[data-seg]').forEach(seg => {
        seg.addEventListener('click', e => {
          const b = e.target.closest('button'); if (!b) return;
          seg.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
        });
      });
      // bound segmented controls (data-setting) persist the picked value
      root.querySelectorAll('.seg[data-setting]').forEach(seg => {
        const key = seg.getAttribute('data-setting');
        seg.addEventListener('click', e => {
          const b = e.target.closest('button'); if (!b) return;
          seg.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
          const val = b.getAttribute('data-val');
          persist(key, val);
          if (key === 'language' && val) {     // language flips the whole UI
            window.EU_LANG = val;
            if (typeof window.__euRender === 'function') window.__euRender();
          }
        });
      });
    },
  };
})();
