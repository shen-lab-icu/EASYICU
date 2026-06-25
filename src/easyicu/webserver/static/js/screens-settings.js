/* Screen: Settings — local paths, data mode, privacy, model, language, about.
   Utility page reached from the sidebar gear. Bound controls persist through
   /api/settings; path pickers list local folders through the FastAPI process. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  const DEFAULT_SETTINGS = {
    ai_enabled: false,
    language: 'en',
    data_mode: 'demo',
    evidence_gate: 'strict',
    demo_patients: 20,
    demo_duration: '24h',
    module_folder_mode: true,
    telemetry_enabled: false,
    cache_cohort_frames: true,
    agent_model_mode: 'local',
    token_budget: 120000,
    auto_repair: true,
    density: 'comfortable',
    reduce_motion: false,
  };
  let settingsPickerEl = null;
  let settingsNotice = '';

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
  function h(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  function row(t, d, ctl) {
    return `<div class="set-row"><div class="sr-main"><div class="sr-t">${t}</div><div class="sr-d">${d}</div></div><div class="sr-ctl">${ctl}</div></div>`;
  }
  function lockedCtl(label, note) {
    return `<span class="pill locked-setting" title="${h(note || '')}">${icon('lock', 12)} ${h(label)}</span>`;
  }
  function truthCtl(label) {
    return `<span class="pill ok-setting">${icon('check', 12)} ${h(label)}</span>`;
  }

  function pathCtl(key, fallback, iconName) {
    const value = S0()[key] || '';
    const label = value || fallback;
    const cls = value ? 'pf-path' : 'pf-path muted';
    return `<div class="path-field"><span class="pf-ico">${icon(iconName, 14)}</span><span class="${cls}">${h(label)}</span></div><button class="btn" data-setting-path="${key}">Change</button>`;
  }

  function ensureSettingPickerStyles() {
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
      .eu-pick-empty{padding:24px;text-align:center;color:var(--ink-4,#8a91a0);font-size:12px;}
      .setting-input{height:34px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink-1);padding:0 10px;text-align:right;max-width:140px;}`;
    document.head.appendChild(s);
  }

  function closeSettingsPicker() {
    if (settingsPickerEl) {
      settingsPickerEl.remove();
      settingsPickerEl = null;
    }
    document.removeEventListener('keydown', settingsPickerKey);
  }

  function settingsPickerKey(e) {
    if (e.key === 'Escape') closeSettingsPicker();
  }

  function openSettingsFolderPicker(startPath, title, onPick) {
    if (!window.EU_API || !window.EU_API.listDir) return;
    ensureSettingPickerStyles();
    closeSettingsPicker();
    let cur = startPath || '';
    const back = document.createElement('div');
    back.className = 'eu-pick-back';
    back.innerHTML = `
      <div class="eu-pick" role="dialog" aria-label="${h(title)}">
        <div class="eu-pick-h">
          <span style="color:var(--ink-3);">${icon('folder', 16)}</span>
          <span class="t">${h(title)}</span>
          <span class="grow" style="flex:1;"></span>
          <button class="btn sm ghost" data-pk-close>${icon('close', 13)}</button>
        </div>
        <div class="eu-pick-cur" data-pk-cur></div>
        <div class="eu-pick-sc" data-pk-sc></div>
        <div class="eu-pick-list" data-pk-list><div class="eu-pick-empty">Loading...</div></div>
        <div class="eu-pick-f">
          <button class="btn ghost sm" data-pk-up>${icon('back', 13)} Up</button>
          <span style="flex:1;"></span>
          <button class="btn primary" data-pk-use>${icon('check', 13)} Use this folder</button>
        </div>
      </div>`;
    document.body.appendChild(back);
    settingsPickerEl = back;
    const listEl = back.querySelector('[data-pk-list]');
    const curEl = back.querySelector('[data-pk-cur]');
    const scEl = back.querySelector('[data-pk-sc]');
    back.addEventListener('click', e => { if (e.target === back) closeSettingsPicker(); });
    back.querySelector('[data-pk-close]').addEventListener('click', closeSettingsPicker);
    back.querySelector('[data-pk-use]').addEventListener('click', () => {
      const chosen = cur;
      closeSettingsPicker();
      if (chosen) onPick(chosen);
    });
    document.addEventListener('keydown', settingsPickerKey);

    function load(path) {
      listEl.innerHTML = `<div class="eu-pick-empty">Loading...</div>`;
      window.EU_API.listDir(path).then(r => {
        cur = r.path || path || '';
        curEl.textContent = cur || '/';
        const up = back.querySelector('[data-pk-up]');
        up.disabled = !r.parent;
        up.onclick = () => r.parent && load(r.parent);
        scEl.innerHTML = '';
        (r.shortcuts || []).forEach(shortcut => {
          const b = document.createElement('button');
          b.textContent = shortcut.name;
          b.onclick = () => load(shortcut.path);
          scEl.appendChild(b);
        });
        if (!r.entries || !r.entries.length) {
          listEl.innerHTML = `<div class="eu-pick-empty">${r.ok === false ? 'Cannot read this folder.' : 'No sub-folders here.'}</div>`;
          return;
        }
        listEl.innerHTML = '';
        r.entries.forEach(entry => {
          const b = document.createElement('button');
          b.className = 'eu-pick-row';
          b.innerHTML = `<span style="color:var(--ink-3);flex:none;">${icon('folder', 15)}</span><span class="nm">${h(entry.name)}</span>${entry.hint ? `<span class="hint">${h(entry.hint)}</span>` : ''}`;
          b.onclick = () => load(entry.path);
          listEl.appendChild(b);
        });
      }).catch(err => {
        listEl.innerHTML = `<div class="eu-pick-empty">Failed to list folder: ${h(err && err.message || err)}</div>`;
      });
    }
    load(cur);
  }

  function downloadSettingsDiagnostics() {
    const payload = {
      generated_at: new Date().toISOString(),
      scope: 'settings_diagnostics_no_secrets',
      settings: Object.assign({}, S0()),
      registry: window.EU_WORKSPACE_REGISTRY || null,
    };
    if (payload.settings && payload.settings.about) {
      payload.settings.about = Object.assign({}, payload.settings.about);
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'easyicu_settings_diagnostics.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  S.settings = {
    section: 'settings', nav: 'settings',
    crumbs: ['Home', 'Settings'],
    actionHtml: `<button class="btn" data-settings-reset>${icon('refresh', 13)} Reset to defaults</button>`,
    rail() {
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">Settings</span></div>
        <div class="set-nav col gap-6" style="font-size:12.5px;">
          ${[
            ['Workspace', 'folder', 'set-workspace'],
            ['Data mode', 'flask', 'set-data-mode'],
            ['Privacy', 'shield', 'set-privacy'],
            ['Research Agent', 'agent', 'set-research-agent'],
            ['Language', 'globe', 'set-language'],
            ['About', 'help', 'set-about'],
          ].map(([t, ic, target]) =>
            `<button type="button" class="nav-item set-nav-btn" data-settings-jump="${target}" style="height:30px;"><span class="ico">${icon(ic, 15)}</span>${t}</button>`).join('')}
        </div>
        <div class="note ok mt-16" style="padding:10px 12px;">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="t" style="font-size:12px;">Local-first</div><div class="d" style="font-size:11px;">All settings stay on this machine.</div></div>
        </div>
      </div>`;
    },
    render() {
      return `
      <div class="settings-page">
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">Workspace · 设置</div>
        <h1 style="margin-top:6px;">Settings</h1>
        <p class="lead">Configure how EasyICU reads data, runs the agent, and presents the workspace. Everything is local and reversible.</p>
      </div>
      ${settingsNotice ? `<div class="note ok mt-12" data-settings-notice><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">Settings updated</div><div class="d">${h(settingsNotice)}</div></div></div>` : ''}

      <div class="sec-stack" id="set-workspace"><div class="lbl">Workspace</div><h2>Local paths</h2></div>
      <div class="card pad">
        ${row('Workspace root', 'Project and guided-run folders are selected in their own workflows; this page does not silently create user folders.',
          lockedCtl('per workflow', 'Current FastAPI workflows keep project roots explicit instead of relying on a hidden global workspace path.'))}
        ${row('Default export folder', 'Optional destination for code, tables, figures, and the evidence ledger bundle.',
          pathCtl('export_dir', 'Not set — EasyICU creates a local run folder when extracting', 'download'))}
        ${row('Module-folder mode', 'Use a registered export source from Data Extraction or Patient/Cohort/Cross-DB instead of a hidden global mode.',
          lockedCtl('chosen per source', 'This is intentionally controlled by the source registry, not a global toggle.'))}
      </div>

      <div class="sec-stack" id="set-data-mode"><div class="lbl">Data mode</div><h2>Defaults for new sessions</h2></div>
      <div class="card pad">
        ${row('Start mode', 'Which mode a new workspace opens in. You can always switch later.',
          segBound('data_mode', [['demo', 'Demo'], ['real', 'Real Data']], S0().data_mode || 'demo'))}
        ${row('Demo fixture size', 'Demo screens use bounded seeded fixtures. Real extraction/review reads your registered local export.',
          lockedCtl('seeded fixture', 'Demo fixture size is not a data-processing setting.'))}
        ${row('Demo duration', 'Demo time windows are illustrative only. Real patient and cohort review use the active export.',
          lockedCtl('fixture only', 'Real review duration comes from the exported data.'))}
      </div>

      <div class="sec-stack" id="set-privacy"><div class="lbl">Privacy</div><h2>Local-first guarantees</h2></div>
      <div class="card pad">
        ${row('Local-only mode', 'Patient data never leaves your machine. This guarantee is enforced and cannot be disabled.',
          `<span class="mono" style="font-size:10.5px;color:var(--ink-4);margin-right:4px;">enforced</span>${sw(true, 'locked ok')}`)}
        ${row('Allow outbound model calls', 'Off by default. When on, only the Research Agent prompt and plan — never patient rows — may reach a configured model endpoint.', sw(!!S0().ai_enabled, '', 'ai_enabled'))}
        ${row('Anonymous usage telemetry', 'There is no telemetry collector in this native app build.',
          truthCtl('not collected'))}
        ${row('Cache cohort frames', 'Review screens read bounded summaries from the registered local export; cache policy is owned by each job/export.',
          lockedCtl('per job', 'No global cohort-frame cache toggle is active in this native route.'))}
      </div>

      <div class="sec-stack" id="set-research-agent"><div class="lbl">Research Agent</div><h2>Run behavior</h2></div>
      <div class="card pad">
        ${row('Model provider', 'Provider is selected per run inside Agent Projects, after global opt-in and credential readiness checks.',
          lockedCtl('per run in Agent Projects', 'This avoids a hidden global model default causing accidental external calls.'))}
        ${row('Token budget', 'Current provider adapter enforces its own bounded max-output contract and records actual usage in the run ledger.',
          lockedCtl('run ledger', 'Global token budgeting is not an active control yet.'))}
        ${row('Auto-repair steps', 'The current native Agent path is deterministic and restartable; repair policy is recorded per run.',
          lockedCtl('restart/resume', 'No hidden repair loop is toggled from Settings.'))}
        ${row('Evidence gate', 'Strict evidence and numeric binding gates are enforced before any draft can become reportable.',
          truthCtl('strict enforced'))}
      </div>

      <div class="sec-stack" id="set-language"><div class="lbl">Language</div><h2>Language & display</h2></div>
      <div class="card pad">
        ${row('Interface language', 'EasyICU is fully bilingual; labels fit both scripts.',
          segBound('language', [['en', 'English'], ['zh', '中文', 'cn']], S0().language || 'en'))}
        ${row('Density', 'Comfortable adds breathing room; compact maximises rows on screen.',
          segBound('density', [['comfortable', 'Comfortable'], ['compact', 'Compact']], S0().density || 'comfortable'))}
        ${row('Reduce motion', 'Disable shimmer and progress animations.', sw(!!S0().reduce_motion, '', 'reduce_motion'))}
      </div>

      <div class="sec-stack" id="set-about"><div class="lbl">About</div><h2>Environment</h2></div>
      <div class="card pad">
        <div class="setup-row"><span class="k">Version</span><span class="vv mono">EasyICU ${(S0().about || {}).version || '—'}</span></div>
        <div class="setup-row"><span class="k">Python</span><span class="vv mono">${(S0().about || {}).python || '—'}</span></div>
        <div class="setup-row"><span class="k">Databases detected</span><span class="vv mono">MIMIC-IV · eICU · AUMC · HiRID · MIMIC-III · SICdb</span></div>
        <div class="setup-row"><span class="k">Workspace</span><span class="vv mono">${h(S0().working_dir || 'not set')}</span></div>
        <div class="set-about-actions row gap-8 mt-16">
          <button class="btn sm" data-settings-doc="release">${icon('file', 13)} Release notes</button>
          <button class="btn sm" data-settings-doc="docs">${icon('help', 13)} Documentation</button>
          <button class="btn sm" data-settings-diagnostics>${icon('download', 13)} Export diagnostics</button>
        </div>
      </div>
      </div>`;
    },
    afterRender(root) {
      ensureSettingPickerStyles();
      const rerender = () => { if (typeof window.__euRender === 'function') window.__euRender(); };
      const setNotice = msg => { settingsNotice = msg || ''; rerender(); };
      const persist = (key, value, msg) => {
        if (key && window.EU_API && window.EU_API.saveSetting) {
          window.EU_API.saveSetting(key, value).then(() => {
            if (msg) setNotice(msg);
          }).catch(err =>
            console.error('[EasyICU] saveSetting failed', key, err));
        }
      };
      root.querySelectorAll('[data-settings-jump]').forEach(btn => {
        btn.addEventListener('click', e => {
          e.preventDefault();
          const target = btn.getAttribute('data-settings-jump');
          const el = target ? document.getElementById(target) : null;
          if (!el) return;
          root.querySelectorAll('[data-settings-jump]').forEach(x => x.classList.toggle('active', x === btn));
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
      root.querySelectorAll('[data-setting-path]').forEach(btn => {
        btn.addEventListener('click', () => {
          const key = btn.getAttribute('data-setting-path');
          const title = key === 'export_dir' ? 'Choose default export folder' : 'Choose working directory';
          openSettingsFolderPicker(S0()[key], title, picked => persist(key, picked, `${title}: ${picked}`));
        });
      });
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
          if (key === 'data_mode' && val && window.setDataMode) {
            window.setDataMode(val);
            return;
          }
          persist(key, val);
          if (key === 'language' && val) {     // language flips the whole UI
            window.EU_LANG = val;
            if (typeof window.__euRender === 'function') window.__euRender();
          }
        });
      });
      root.querySelectorAll('[data-setting-input]').forEach(input => {
        input.addEventListener('change', () => {
          const key = input.getAttribute('data-setting-input');
          const value = input.type === 'number' ? Number(input.value) : input.value;
          persist(key, value, `${key.replace(/_/g, ' ')} saved`);
        });
      });
      root.querySelectorAll('[data-settings-reset]').forEach(btn => {
        btn.addEventListener('click', () => {
          if (!window.EU_API || !window.EU_API.resetSettings) return;
          window.EU_API.resetSettings().then(() => setNotice('Settings reset to backend defaults.')).catch(err =>
            console.error('[EasyICU] resetSettings failed', err));
        });
      });
      root.querySelectorAll('[data-settings-diagnostics]').forEach(btn => {
        btn.addEventListener('click', () => {
          downloadSettingsDiagnostics();
          setNotice('Downloaded local settings diagnostics JSON. No secrets are included.');
        });
      });
      root.querySelectorAll('[data-settings-doc]').forEach(btn => {
        btn.addEventListener('click', () => {
          const target = btn.getAttribute('data-settings-doc');
          if (target === 'docs') {
            location.hash = '#help';
          } else {
            setNotice('Release notes are tracked in the local docs/task logs for this build.');
          }
        });
      });
    },
  };
})();
