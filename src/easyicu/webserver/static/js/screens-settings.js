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
  function T(en, zh) { return t(en, zh); }

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
    return `<div class="path-field"><span class="pf-ico">${icon(iconName, 14)}</span><span class="${cls}">${h(label)}</span></div><button class="btn" data-setting-path="${key}">${T('Change', '更改')}</button>`;
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
        <div class="eu-pick-list" data-pk-list><div class="eu-pick-empty">${T('Loading...', '正在加载...')}</div></div>
        <div class="eu-pick-f">
          <button class="btn ghost sm" data-pk-up>${icon('back', 13)} ${T('Up', '上一级')}</button>
          <span style="flex:1;"></span>
          <button class="btn primary" data-pk-use>${icon('check', 13)} ${T('Use this folder', '使用此文件夹')}</button>
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
      listEl.innerHTML = `<div class="eu-pick-empty">${T('Loading...', '正在加载...')}</div>`;
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
          listEl.innerHTML = `<div class="eu-pick-empty">${r.ok === false ? T('Cannot read this folder.', '无法读取此文件夹。') : T('No sub-folders here.', '这里没有子文件夹。')}</div>`;
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
        listEl.innerHTML = `<div class="eu-pick-empty">${T('Failed to list folder:', '列出文件夹失败：')} ${h(err && err.message || err)}</div>`;
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
    get crumbs() { return [T('Home', '首页'), T('Settings', '设置')]; },
    get actionHtml() { return `<button class="btn" data-settings-reset>${icon('refresh', 13)} ${T('Reset to defaults', '恢复默认设置')}</button>`; },
    rail() {
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${T('Settings', '设置')}</span></div>
        <div class="set-nav col gap-6" style="font-size:12.5px;">
          ${[
            [T('Workspace', '工作区'), 'folder', 'set-workspace'],
            [T('Data mode', '数据模式'), 'flask', 'set-data-mode'],
            [T('Privacy', '隐私'), 'shield', 'set-privacy'],
            [T('Research Agent', '研究代理'), 'agent', 'set-research-agent'],
            [T('Language', '语言'), 'globe', 'set-language'],
            [T('About', '关于'), 'help', 'set-about'],
          ].map(([label, ic, target]) =>
            `<button type="button" class="nav-item set-nav-btn" data-settings-jump="${target}" style="height:30px;"><span class="ico">${icon(ic, 15)}</span>${label}</button>`).join('')}
        </div>
        <div class="note ok mt-16" style="padding:10px 12px;">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="t" style="font-size:12px;">${T('Local-first', '本地优先')}</div><div class="d" style="font-size:11px;">${T('All settings stay on this machine.', '所有设置都保存在本机。')}</div></div>
        </div>
      </div>`;
    },
    render() {
      return `
      <div class="settings-page">
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">${T('Workspace · Settings', '工作区 · 设置')}</div>
        <h1 style="margin-top:6px;">${T('Settings', '设置')}</h1>
        <p class="lead">${T('Configure how EasyICU reads data, runs the agent, and presents the workspace. Everything is local and reversible.', '配置 EasyICU 如何读取数据、运行研究代理并呈现工作区。所有设置都保存在本地，且可以撤销。')}</p>
      </div>
      ${settingsNotice ? `<div class="note ok mt-12" data-settings-notice><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${T('Settings updated', '设置已更新')}</div><div class="d">${h(settingsNotice)}</div></div></div>` : ''}

      <div class="sec-stack" id="set-workspace"><div class="lbl">${T('Workspace', '工作区')}</div><h2>${T('Local paths', '本地路径')}</h2></div>
      <div class="card pad">
        ${row(T('Workspace root', '工作区根目录'), T('Project and guided-run folders are selected in their own workflows; this page does not silently create user folders.', '项目和引导运行文件夹会在各自工作流中选择；此页面不会静默创建用户文件夹。'),
          lockedCtl(T('per workflow', '按工作流'), T('Current FastAPI workflows keep project roots explicit instead of relying on a hidden global workspace path.', '当前 FastAPI 工作流会显式选择项目根目录，而不是依赖隐藏的全局工作区路径。')))}
        ${row(T('Default export folder', '默认导出文件夹'), T('Optional destination for code, tables, figures, and the evidence ledger bundle.', '用于代码、表格、图件和证据账本包的可选导出位置。'),
          pathCtl('export_dir', T('Not set - EasyICU creates a local run folder when extracting', '未设置 - 抽取时 EasyICU 会创建本地运行文件夹'), 'download'))}
        ${row(T('Module-folder mode', '模块文件夹模式'), T('Use a registered export source from Data Extraction or Patient/Cohort/Cross-DB instead of a hidden global mode.', '使用数据抽取或患者/队列/跨库页面注册的导出来源，而不是隐藏的全局模式。'),
          lockedCtl(T('chosen per source', '按来源选择'), T('This is intentionally controlled by the source registry, not a global toggle.', '这里有意由来源注册表控制，而不是全局开关。')))}
      </div>

      <div class="sec-stack" id="set-data-mode"><div class="lbl">${T('Data mode', '数据模式')}</div><h2>${T('Defaults for new sessions', '新会话默认值')}</h2></div>
      <div class="card pad">
        ${row(T('Start mode', '启动模式'), T('Which mode a new workspace opens in. You can always switch later.', '新工作区默认打开的模式；之后随时可以切换。'),
          segBound('data_mode', [['demo', T('Demo', '演示')], ['real', T('Real Data', '真实数据')]], S0().data_mode || 'demo'))}
        ${row(T('Demo fixture size', '演示夹具大小'), T('Demo screens use bounded seeded fixtures. Real extraction/review reads your registered local export.', '演示页面使用有边界的种子夹具；真实抽取/审阅会读取已注册的本地导出。'),
          lockedCtl(T('seeded fixture', '种子夹具'), T('Demo fixture size is not a data-processing setting.', '演示夹具大小不是数据处理设置。')))}
        ${row(T('Demo duration', '演示时间范围'), T('Demo time windows are illustrative only. Real patient and cohort review use the active export.', '演示时间窗只用于示意；真实患者和队列审阅会使用当前导出。'),
          lockedCtl(T('fixture only', '仅夹具'), T('Real review duration comes from the exported data.', '真实审阅时间范围来自导出的数据。')))}
      </div>

      <div class="sec-stack" id="set-privacy"><div class="lbl">${T('Privacy', '隐私')}</div><h2>${T('Local-first guarantees', '本地优先保障')}</h2></div>
      <div class="card pad">
        ${row(T('Local-only mode', '仅本地模式'), T('Patient data never leaves your machine. This guarantee is enforced and cannot be disabled.', '患者数据永远不会离开本机。此保障已强制启用，不能关闭。'),
          `<span class="mono" style="font-size:10.5px;color:var(--ink-4);margin-right:4px;">${T('enforced', '已强制')}</span>${sw(true, 'locked ok')}`)}
        ${row(T('Allow outbound model calls', '允许外部模型调用'), T('Off by default. When on, only the Research Agent prompt and plan - never patient rows - may reach a configured model endpoint.', '默认关闭。开启后，只有研究代理的提示词和计划可以发送到已配置的模型端点，患者行数据永不发送。'), sw(!!S0().ai_enabled, '', 'ai_enabled'))}
        ${row(T('Anonymous usage telemetry', '匿名使用遥测'), T('There is no telemetry collector in this native app build.', '当前原生应用构建中没有遥测收集器。'),
          truthCtl(T('not collected', '未收集')))}
        ${row(T('Cache cohort frames', '缓存队列帧'), T('Review screens read bounded summaries from the registered local export; cache policy is owned by each job/export.', '审阅页面从已注册的本地导出读取有边界的摘要；缓存策略由每个任务/导出自行管理。'),
          lockedCtl(T('per job', '按任务'), T('No global cohort-frame cache toggle is active in this native route.', '此原生页面没有启用全局队列帧缓存开关。')))}
      </div>

      <div class="sec-stack" id="set-research-agent"><div class="lbl">${T('Research Agent', '研究代理')}</div><h2>${T('Run behavior', '运行行为')}</h2></div>
      <div class="card pad">
        ${row(T('Model provider', '模型提供方'), T('Provider is selected per run inside Agent Projects, after global opt-in and credential readiness checks.', '模型提供方在研究项目的每次运行中选择，并经过全局授权和凭证就绪检查。'),
          lockedCtl(T('per run in Agent Projects', '在研究项目中按运行选择'), T('This avoids a hidden global model default causing accidental external calls.', '这样可以避免隐藏的全局模型默认值造成意外外部调用。')))}
        ${row(T('Token budget', '令牌预算'), T('Current provider adapter enforces its own bounded max-output contract and records actual usage in the run ledger.', '当前提供方适配器会执行自己的有界最大输出契约，并在运行账本中记录实际用量。'),
          lockedCtl(T('run ledger', '运行账本'), T('Global token budgeting is not an active control yet.', '全局 token 预算尚不是启用中的控制项。')))}
        ${row(T('Auto-repair steps', '自动修复步骤'), T('The current native Agent path is deterministic and restartable; repair policy is recorded per run.', '当前原生代理路径是确定且可重启的；修复策略按运行记录。'),
          lockedCtl(T('restart/resume', '重启/继续'), T('No hidden repair loop is toggled from Settings.', '设置页不会切换隐藏的修复循环。')))}
        ${row(T('Evidence checks', '证据核验'), T('Strict evidence and numeric binding checks are enforced before any draft can become reportable.', '任何草稿进入可报告状态前，都会强制执行严格证据和数值绑定核验。'),
          truthCtl(T('strict enforced', '严格强制')))}
      </div>

      <div class="sec-stack" id="set-language"><div class="lbl">${T('Language', '语言')}</div><h2>${T('Language & display', '语言与显示')}</h2></div>
      <div class="card pad">
        ${row(T('Interface language', '界面语言'), T('EasyICU is fully bilingual; labels fit both scripts.', 'EasyICU 支持完整双语界面；标签会适配中英文显示。'),
          segBound('language', [['en', 'English'], ['zh', '中文', 'cn']], S0().language || 'en'))}
        ${row(T('Density', '界面密度'), T('Comfortable adds breathing room; compact maximises rows on screen.', '舒适模式增加留白；紧凑模式在屏幕中显示更多行。'),
          segBound('density', [['comfortable', T('Comfortable', '舒适')], ['compact', T('Compact', '紧凑')]], S0().density || 'comfortable'))}
        ${row(T('Reduce motion', '减少动画'), T('Disable shimmer and progress animations.', '关闭闪烁和进度动画。'), sw(!!S0().reduce_motion, '', 'reduce_motion'))}
      </div>

      <div class="sec-stack" id="set-about"><div class="lbl">${T('About', '关于')}</div><h2>${T('Environment', '环境')}</h2></div>
      <div class="card pad">
        <div class="setup-row"><span class="k">${T('Version', '版本')}</span><span class="vv mono">EasyICU ${(S0().about || {}).version || '—'}</span></div>
        <div class="setup-row"><span class="k">Python</span><span class="vv mono">${(S0().about || {}).python || '—'}</span></div>
        <div class="setup-row"><span class="k">${T('Databases detected', '已检测数据库')}</span><span class="vv mono">MIMIC-IV · eICU · AUMC · HiRID · MIMIC-III · SICdb</span></div>
        <div class="setup-row"><span class="k">${T('Workspace', '工作区')}</span><span class="vv mono">${h(S0().working_dir || T('not set', '未设置'))}</span></div>
        <div class="set-about-actions row gap-8 mt-16">
          <button class="btn sm" data-settings-doc="release">${icon('file', 13)} ${T('Release notes', '版本说明')}</button>
          <button class="btn sm" data-settings-doc="docs">${icon('help', 13)} ${T('Documentation', '文档')}</button>
          <button class="btn sm" data-settings-diagnostics>${icon('download', 13)} ${T('Export diagnostics', '导出诊断')}</button>
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
          const title = key === 'export_dir' ? T('Choose default export folder', '选择默认导出文件夹') : T('Choose working directory', '选择工作目录');
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
          if (key === 'language' && val && window.setLang) {
            window.setLang(val);
            return;
          }
          persist(key, val);
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
          window.EU_API.resetSettings().then(() => setNotice(T('Settings reset to backend defaults.', '设置已恢复为后端默认值。'))).catch(err =>
            console.error('[EasyICU] resetSettings failed', err));
        });
      });
      root.querySelectorAll('[data-settings-diagnostics]').forEach(btn => {
        btn.addEventListener('click', () => {
          downloadSettingsDiagnostics();
          setNotice(T('Downloaded local settings diagnostics JSON. No secrets are included.', '已下载本地设置诊断 JSON，不包含密钥。'));
        });
      });
      root.querySelectorAll('[data-settings-doc]').forEach(btn => {
        btn.addEventListener('click', () => {
          const target = btn.getAttribute('data-settings-doc');
          if (target === 'docs') {
            location.hash = '#help';
          } else {
            setNotice(T('Release notes are tracked in the local docs/task logs for this build.', '此构建的版本说明记录在本地文档和任务日志中。'));
          }
        });
      });
    },
  };
})();
