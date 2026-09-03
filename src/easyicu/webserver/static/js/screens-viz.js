/* Screens: Data Visualization — Patient Review, Cohort Statistics, Cross-database comparison */
(function () {
  const { esc } = window.EU_HTML;
  const S = (window.SCREENS = window.SCREENS || {});
  const patientReview = window.EU_VIZ_PATIENT;
  const cohortOwner = window.EU_VIZ_COHORT;
  const { catalogModuleLabel, catalogFeatureMeta } = window.VIZ_DEMO;

  function workspaceSamplingNote(summary) {
    const s = summary || {};
    if (!(Number(s.sampled_stays) < Number(s.total_stays))) return '';
    return `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${t('Bounded analysis snapshot', '有界分析快照')}</div><div class="d">${t('Displayed aggregate metrics use', '当前展示的聚合指标使用')} ${fmtInt(s.sampled_stays)} / ${fmtInt(s.total_stays)} ${t('stays; the full denominator remains visible.', '次住院；完整分母仍明确展示。')} <span class="mono">${esc(s.snapshot_basis || '')}</span></div></div></div>`;
  }

  function vizRail(active) {
    const dataMode = window.getDataMode
      ? window.getDataMode()
      : (window.EU_DATA === 'real' ? 'real' : 'demo');
    const real = dataMode === 'real';
    const officialDemo = !real && officialDemoContext();
    const xdb = active === 'crossdb' ? window.EU_CROSSDB_WORKSPACE : null;
    const drill = active === 'patient' ? patientReview.drilldown() : null;
    const cohort = active === 'cohort' ? cohortOwner.review() : null;
    const ws = window.EU_VIZ_WORKSPACE;
    const wsMatchesActive = ws && (
      ws.route === active
      || (!ws.route && active === 'patient' && ws.summary && ws.summary.stays != null)
    );
    const patientSource = active === 'patient' ? patientReview.activeSourceMeta() : null;
    const label = real
      ? t('Real', '真实')
      : (officialDemo ? t('Official demo', '官方演示') : t('Demo', '演示'));
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
        ? `${fmtInt(drill.summary && drill.summary.entities)} ${t('synthetic entities', '个合成实体')}`
        : `${fmtInt(drill.summary && drill.summary.entities)} ${t('entities', '个实体')}`;
      variables = drill.demo
        ? `${fmtInt(drill.summary && drill.summary.modules)} ${t('modules', '个模块')} · ${fmtInt(loaded.review_features)} ${t('features', '个特征')}`
        : `${fmtInt(drill.summary && drill.summary.modules)} ${t('modules', '个模块')} · ${fmtInt(loaded.review_features)} ${t('features', '个特征')}`;
    } else if (cohort) {
      const fsel = cohort.feature_selection || {};
      dataset = (cohort.source || {}).label || t('Local export', '本地导出');
      cohortLine = `${fmtInt(cohort.summary && cohort.summary.cohort_size)} ${t('entities', '个实体')}`;
      variables = `${fmtInt(cohort.summary && cohort.summary.modules)} ${t('modules', '个模块')} · ${fmtInt(fsel.selected_count)} / ${fmtInt(fsel.available_count)} ${t('features', '个特征')}`;
    } else if (real && patientSource) {
      const summary = patientSource.summary || {};
      dataset = patientSource.label || patientSource.database || t('Local export', '本地导出');
      cohortLine = `${fmtInt(summary.entities != null ? summary.entities : summary.stays)} ${t('entities', '个实体')}`;
      variables = `${fmtInt(summary.modules)} ${t('modules', '个模块')}`;
    } else if (wsMatchesActive) {
      dataset = (ws.path || '').split('/').filter(Boolean).slice(-2).join('/') || t('Local export', '本地导出');
      const summary = ws.summary || {};
      const sample = Number(summary.sampled_stays) < Number(summary.total_stays) ? ` · ${t('metrics n', '指标 n')}=${fmtInt(summary.sampled_stays)}` : '';
      cohortLine = `${fmtInt(summary.stays)} ${t('stays', '次住院')}${sample}`;
      variables = `${fmtInt(ws.summary && ws.summary.modules)} ${t('modules', '个模块')}`;
    } else {
      const cat = window.EU_CATALOG || {};
      const moduleCount = Array.isArray(cat.groups) ? cat.groups.length : 19;
      const featureCount = cat.totalConcepts || Object.values(cat.groupConcepts || {}).reduce((a, b) => a + (Array.isArray(b) ? b.length : 0), 0) || 247;
      if (!real && active === 'cohort' && cohortOwner.snapshot().view === 'loaded') {
        const scope = cohortOwner.demoCatalogScope();
        dataset = t('Demo · EasyICU catalog', '演示 · EasyICU 字典');
        cohortLine = `10 ${t('stays', '次住院')}`;
        variables = `${fmtInt(scope.selectedModuleCount)} / ${fmtInt(scope.totalModuleCount || moduleCount)} ${t('modules', '个模块')} · ${fmtInt(scope.selectedFeatureCount)} / ${fmtInt(scope.totalFeatureCount || featureCount)} ${t('features', '个特征')}`;
      } else {
        dataset = real ? t('No export loaded', '尚未加载导出') : t('Demo · EasyICU catalog', '演示 · EasyICU 字典');
        cohortLine = real ? t('load exported tables', '加载导出表') : t('official demos available', '官方演示数据可用');
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
      ${active === 'crossdb' && crossSetup.view() === 'loading' ? '' : `<button class="btn sm block" data-viz-reset style="margin-top:12px;">${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button>`}
    </div>`;
  }

  /* view state for the interactive viz screens */
  const crossSetup = window.EU_CROSSDB_SETUP;
  const crossResults = window.EU_CROSSDB_RESULTS;
  const crossRawProgress = window.EU_CROSSDB_PROGRESS;
  function resetCrossSetupForSourceChange() {
    crossSetup.onRegistryChanged();
  }
  let vizErr = null;

  function setRouteError(screenId, value) {
    if (screenId === 'patient') patientReview.setError(value);
    else if (screenId === 'cohort') cohortOwner.setError(value);
    else vizErr = value == null ? null : String(value);
  }

  function displayDataMode() {
    return window.getDataMode
      ? window.getDataMode()
      : (window.EU_DATA === 'real' ? 'real' : 'demo');
  }
  function officialDemoContext() {
    const context = window.EU_DATA_MODE_CONTEXT;
    return context && String(context.kind || '').startsWith('official_demo')
      ? context
      : null;
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
  function explicitRegistryCrossdbPaths() {
    const reg = window.EU_SOURCES && window.EU_SOURCES.registry
      ? window.EU_SOURCES.registry()
      : (window.EU_WORKSPACE_REGISTRY || {});
    return Array.isArray(reg.crossdb_paths)
      ? Array.from(new Set(reg.crossdb_paths.map(path => String(path || '').trim()).filter(Boolean)))
      : registryCrossdbPaths();
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
  function registeredSelectionIdentity(paths) {
    const selected = new Set((paths || []).map(path => String(path || '').trim()).filter(Boolean));
    return registrySources()
      .filter(source => selected.has(String(source.path || '').trim()))
      .map(source => String(source.id || '').trim())
      .filter(Boolean)
      .sort()
      .join(',');
  }
  function receiptSelectionIdentity(receipt) {
    return (Array.isArray(receipt && receipt.sources) ? receipt.sources : [])
      .map(source => String(source && source.source_id || '').trim())
      .filter(Boolean)
      .sort()
      .join(',');
  }
  function registeredSelectionAvailable(meta) {
    const expected = String(meta && meta.source_identity || '').split(',').filter(Boolean).sort();
    const available = new Set(registrySources().map(source => String(source.id || '').trim()).filter(Boolean));
    return expected.length >= 2 && expected.every(sourceId => available.has(sourceId));
  }
  function repaintCrossRawProgress() {
    const restoreCancelFocus = document.activeElement && document.activeElement.matches('[data-crossdb-cancel]');
    repaintScreen('crossdb');
    if (restoreCancelFocus && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => {
        const button = document.querySelector('[data-crossdb-cancel]');
        if (button) button.focus();
      });
    }
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
  function sourceRegistryBlock(mode) {
    const multi = mode === 'multi';
    const active = defaultExportPath();
    const selected = new Set(explicitRegistryCrossdbPaths());
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
          <div><div class="eyebrow">${title}</div><div class="src-sub">${multi ? t('Choose at least two exports for Cross-DB preview.', '请选择至少两个导出用于跨库预览。') : t('This active export is shared by Patient, Cohort, Agent, and Copilot.', '这个 active 导出会被患者审阅、队列统计、Agent 和 Copilot 共用。')}</div></div>
          <button class="btn sm ghost" data-src-refresh>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
        </div>
        <div class="src-list">
          ${sources.length ? (() => {
            const rowHtml = (s) => {
              const on = multi ? selected.has(s.path) : s.path === active;
              const attr = multi ? '' : `data-src-active="${esc(s.path)}"`;
              const label = s.label || s.database || t('local', '本地');
              return `
              <div class="src-row ${on ? 'on' : ''}" ${attr}>
                <span class="src-ico">${icon(multi && on ? 'check' : 'folder', 14, multi && on ? 2.6 : undefined)}</span>
                <span class="src-body"><span class="src-name">${esc(label)}</span><span class="src-meta">${esc(sourceLine(s))}</span><span class="src-path mono">${esc(s.path)}</span></span>
                ${multi
                  ? `<button class="btn sm ${on ? '' : 'ghost'}" type="button" data-src-cross="${esc(s.path)}" aria-pressed="${on ? 'true' : 'false'}">${on ? t('selected', '已选择') : t('add', '添加')}</button>`
                  : `<span class="pill ${on ? 'ok' : 'dashed'}" style="height:20px;">${on ? t('active', '当前') : t('use', '使用')}</span>`}
                <span class="src-actions">
                  <button class="btn icon sm ghost" data-src-action data-src-rename="${esc(s.path)}" data-src-label="${esc(label)}" title="${esc(t('Rename source', '重命名来源'))}">${icon('edit', 12)}</button>
                  <button class="btn icon sm ghost" data-src-action data-src-remove="${esc(s.path)}" title="${esc(t('Remove registration only; files stay on disk', '仅移除注册记录；磁盘文件保留'))}">${icon('close', 12)}</button>
                </span>
              </div>`;
            };
            /* Registered exports pile up fast and near-identical rows drown the
               chosen one — keep the selected/active + first few visible, fold the
               rest behind an explicit "older exports" toggle. */
            const FOLD_AFTER = 5;
            const head = sources.slice(0, FOLD_AFTER).map(rowHtml).join('');
            const rest = sources.slice(FOLD_AFTER);
            const folded = rest.length ? `
              <details class="src-fold">
                <summary>${t('Show', '显示其余')} ${fmtInt(rest.length)} ${t('older registered exports', '个较早注册的导出')}</summary>
                ${rest.map(rowHtml).join('')}
              </details>` : '';
            return head + folded;
          })() : `<div class="empty compact"><div class="glyph">${icon('folder', 20)}</div><div class="t">${empty}</div></div>`}
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
  function sourceModeSelector(realMode) {
    return `
      <div class="eyebrow" style="margin-bottom:10px;">${t('Data source', '数据源')}</div>
      <div class="radio-row">
        <label class="radio ${realMode ? 'on' : ''}" role="button" tabindex="0" data-datamode="real"><span class="mk"></span> ${t('Previously exported data', '此前导出的数据')}</label>
        <label class="radio ${realMode ? '' : 'on'}" role="button" tabindex="0" data-datamode="demo"><span class="mk"></span> ${t('Demo data', '演示数据')}</label>
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
      setRouteError(screenId, null);
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
      setRouteError(screenId, null); window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; cohortOwner.resetForSourceChange(); resetCrossSetupForSourceChange(); patientReview.resetForSourceChange(); repaintScreen(screenId);
    }).catch(err => {
      const msg = String(err && err.message || err);
      setRouteError(screenId, msg);
      if (button) button.removeAttribute('aria-disabled');
      setSourceAddFeedback(container, msg, 'warn');
    });
  }
  let sourcePickerEl = null;
  let sourcePickerReturnFocus = null;

  function closeSourcePicker(options) {
    const returnFocus = sourcePickerReturnFocus;
    if (sourcePickerEl) { sourcePickerEl.remove(); sourcePickerEl = null; }
    sourcePickerReturnFocus = null;
    document.removeEventListener('keydown', sourcePickerKey);
    if (!(options && options.restoreFocus === false) && returnFocus && returnFocus.isConnected && typeof returnFocus.focus === 'function') {
      returnFocus.focus();
    }
  }
  function sourcePickerKey(e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      closeSourcePicker();
      return;
    }
    if (e.key !== 'Tab' || !sourcePickerEl) return;
    const focusable = Array.from(sourcePickerEl.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])'));
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
  function openSourceFolderPicker(startPath, onPick, title) {
    if (!(window.EU_API && window.EU_API.listDir)) {
      if (onPick) onPick('');
      return;
    }
    closeSourcePicker({ restoreFocus: false });
    sourcePickerReturnFocus = document.activeElement;
    let cur = startPath || '';
    const pickerTitle = title || t('Choose EasyICU export folder', '选择 EasyICU 导出文件夹');
    const back = document.createElement('div'); back.className = 'eu-pick-back';
    back.innerHTML = `
      <div class="eu-pick" role="dialog" aria-modal="true" aria-labelledby="eu-source-picker-title">
        <div class="eu-pick-h">
          <span style="color:var(--ink-3);">${icon('folder', 16)}</span>
          <span class="t" id="eu-source-picker-title">${esc(pickerTitle)}</span>
          <span class="grow" style="flex:1;"></span>
          <button class="btn sm ghost" type="button" data-pk-close aria-label="${esc(t('Close folder picker', '关闭文件夹选择器'))}">${icon('close', 13)}</button>
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
    if (typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => {
        const closeButton = back.querySelector('[data-pk-close]');
        if (closeButton) closeButton.focus();
      });
    }

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
        window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; cohortOwner.resetForSourceChange(); window.EU_STALE = true;
        patientReview.resetForSourceChange(); resetCrossSetupForSourceChange();
        if (screenId === 'cohort' && window.EU_DATA === 'real') {
          cohortOwner.reloadAfterSourceChange();
          return;
        }
        repaintScreen(screenId);
      }).catch(err => { setRouteError(screenId, String(err && err.message || err)); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-cross]').forEach(b => b.addEventListener('click', e => {
      if (e.target.closest('[data-src-action]')) return;
      const path = b.dataset.srcCross;
      const cur = explicitRegistryCrossdbPaths();
      const next = cur.includes(path) ? cur.filter(p => p !== path) : cur.concat([path]);
      if (!(window.EU_API && window.EU_API.saveWorkspaceRegistry)) return;
      window.EU_API.saveWorkspaceRegistry({ crossdb_paths: next }).then(() => {
        const continuity = window.EU_CROSSDB_JOB_CONTINUITY;
        if (continuity && typeof continuity.onSelectionChanged === 'function') {
          continuity.onSelectionChanged(registeredSelectionIdentity(next), '');
        }
        window.EU_CROSSDB_WORKSPACE = null; cohortOwner.resetForSourceChange(); resetCrossSetupForSourceChange(); repaintScreen(screenId);
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
        setRouteError(screenId, null); repaintScreen(screenId);
      }).catch(err => { setRouteError(screenId, String(err && err.message || err)); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-remove]').forEach(b => b.addEventListener('click', e => {
      e.preventDefault(); e.stopPropagation();
      const path = b.dataset.srcRemove;
      if (!path || !(window.EU_API && window.EU_API.removeWorkspaceSource)) return;
      if (!window.confirm(t('Remove this source from the registry? Export files stay on disk.', '从注册表中移除此来源？导出文件仍会保留在磁盘上。'))) return;
      window.EU_API.removeWorkspaceSource(path).then(() => {
        setRouteError(screenId, null); window.EU_VIZ_WORKSPACE = null; window.EU_CROSSDB_WORKSPACE = null; window.EU_PATIENT_DRILLDOWN = null; window.EU_PATIENT_SOURCES = null; cohortOwner.resetForSourceChange(); resetCrossSetupForSourceChange(); patientReview.resetForSourceChange(); repaintScreen(screenId);
      }).catch(err => { setRouteError(screenId, String(err && err.message || err)); repaintScreen(screenId); });
    }));
    root.querySelectorAll('[data-src-refresh]').forEach(b => b.addEventListener('click', () => {
      if (!(window.EU_API && window.EU_API.hydrateWorkspaceRegistry)) return;
      window.EU_API.hydrateWorkspaceRegistry().then(() => { window.EU_PATIENT_SOURCES = null; repaintScreen(screenId); }).catch(err => { setRouteError(screenId, String(err && err.message || err)); repaintScreen(screenId); });
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
  function crossSetupConfig() {
    return {
      api: window.EU_API,
      catalogTotals: {
        modules: (window.EU_CATALOG && window.EU_CATALOG.groups || []).length,
        features: window.EU_CATALOG && window.EU_CATALOG.totalConcepts,
      },
      helpers: {
        esc,
        fmtInt,
        icon,
        progressMessage: crossProgressMessage,
        statusLabel: crossStatusLabel,
        t,
        term: crossTerm,
      },
      bindRegistry: bindSourceRegistry,
      getError() { return vizErr; },
      setError(value) { vizErr = value; },
      repaint() { repaintScreen('crossdb'); },
      registryHtml() { return sourceRegistryBlock('multi'); },
      openFolderPicker: openSourceFolderPicker,
      runRaw: loadRealCrossdb,
      runDemo: loadDemoCrossdb,
      header: crossHeader,
      renderLoaded(payload) {
        return crossResults.render(payload, crossResultsConfig());
      },
      resetResult() {
        crossResults.reset();
        window.EU_VIZ_WORKSPACE = null;
        window.EU_CROSSDB_WORKSPACE = null;
      },
    };
  }
  function crossResultsConfig() {
    return {
      catalogTotals: {
        modules: (window.EU_CATALOG && window.EU_CATALOG.groups || []).length,
        features: window.EU_CATALOG && window.EU_CATALOG.totalConcepts,
      },
      coreFeatures: window.EU_CROSSDB_RAW.coreFeatures(),
      helpers: {
        catalogFeatureMeta,
        catalogModuleLabel,
        esc,
        fmtInt,
        fmtNum,
        fmtPct,
        icon,
        metricLabel: crossMetricLabel,
        statusLabel: crossStatusLabel,
        t,
        term: crossTerm,
      },
      repaint() { repaintScreen('crossdb'); },
      expandScope() {
        crossSetup.setSourceMethod('raw');
        crossSetup.setFeatureScope('all');
        crossSetup.setView('idle');
        crossResults.reset();
        window.EU_VIZ_WORKSPACE = null;
        window.EU_CROSSDB_WORKSPACE = null;
        repaintScreen('crossdb');
      },
      exportPayload(payload) {
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
      },
    };
  }
  function loadRealCrossdb(done, opts) {
    const operationId = opts && opts.operationId;
    const operationActive = () => operationId == null || crossSetup.operationCurrent(operationId);
    if (!operationActive()) return;
    crossSetup.disconnectJob({ forget: true });
    crossRawProgress.clear();
    window.EU_CROSSDB_WORKSPACE = null;
    cohortOwner.resetForSourceChange();
    crossResults.reset();
    // An explicitly configured raw ICU root wins over the registered-export
    // fallback — otherwise the setup UI's root/database/sampling choices are
    // silently discarded whenever >=2 exports happen to be registered.
    const requestedRawRoot = opts && opts.rawRoot ? String(opts.rawRoot).trim() : '';
    const registeredPathOverride = !!(opts && Array.isArray(opts.registeredPaths));
    const paths = registeredPathOverride
      ? Array.from(new Set(opts.registeredPaths.map(path => String(path || '').trim()).filter(Boolean)))
      : defaultCrossdbPaths();
    const jobContinuity = window.EU_CROSSDB_JOB_CONTINUITY;
    if (!requestedRawRoot && paths.length >= 2 && window.EU_API && window.EU_API.startCrossdbReviewSummaryJob
        && jobContinuity && typeof jobContinuity.start === 'function' && window.EventSource) {
      if (crossRawProgress.snapshot().starting) return;
      crossRawProgress.beginStart();
      window.EU_API.startCrossdbReviewSummaryJob({ paths, deadline_seconds: 120 }).then(r => {
        if (!operationActive()) {
          if (r && r.job_id && window.EU_API.cancelJob) {
            window.EU_API.cancelJob(r.job_id, 'client_operation_invalidated').catch(() => {});
          }
          return;
        }
        const receipt = r && r.selection_receipt;
        const sourceIdentity = receiptSelectionIdentity(receipt);
        const expectedIdentity = registeredSelectionIdentity(paths);
        const selectionDigest = String(receipt && receipt.selection_digest || '').trim();
        const deadlineAt = Number(r && r.deadline_at);
        const lease = r && r.source_lease;
        const validSubmission = r && r.job_id && r.kind === 'crossdb-summary'
          && sourceIdentity && sourceIdentity === expectedIdentity
          && Number(receipt && receipt.source_count || 0) === paths.length
          && /^[a-f0-9]{64}$/.test(selectionDigest)
          && Number.isFinite(deadlineAt) && deadlineAt > 0
          && lease && lease.active === true && lease.selection_digest === selectionDigest;
        if (!validSubmission) {
          if (r && r.job_id && window.EU_API.cancelJob) {
            window.EU_API.cancelJob(r.job_id, 'invalid_submission_receipt').catch(() => {});
          }
          throw new Error(t('Registered Cross-DB job returned an invalid source lease receipt.', '已注册导出的跨库任务返回了无效来源租约回执。'));
        }
        const queuedProgress = {
          job_kind: 'crossdb-summary',
          phase: 'queued',
          current: 0,
          total: paths.length,
          source_count: paths.length,
          message: t('Queued aggregate summaries for the selected registered exports.', '所选已注册导出的聚合摘要任务已排队。'),
        };
        crossRawProgress.attach(r.job_id, queuedProgress);
        crossSetup.setRegisteredLoading(false);
        const watching = jobContinuity.start({
          job_id: r.job_id,
          kind: r.kind,
          source_identity: sourceIdentity,
          selection_digest: selectionDigest,
          deadline_at: deadlineAt,
        }, queuedProgress, done);
        if (!watching) {
          crossRawProgress.clear();
          vizErr = t('Could not keep the registered Cross-DB job attached to this browser session.', '无法将已注册导出的跨库任务绑定到当前浏览器会话。');
          if (window.EU_API.cancelJob) window.EU_API.cancelJob(r.job_id, 'client_attach_failed').catch(() => {});
          done && done(false);
        } else {
          crossRawProgress.flushCancel(window.EU_API, err => {
            vizErr = String(err && err.message || err);
            repaintCrossRawProgress();
          });
        }
        repaintScreen('crossdb');
      }).catch(err => {
        if (!operationActive()) return;
        crossRawProgress.clear();
        vizErr = String(err && err.message || err);
        done && done(false);
      });
      return;
    }
    if (!requestedRawRoot && (registeredPathOverride || crossSetup.sourceMethod() === 'registered')) {
      vizErr = paths.length < 2
        ? t('Select at least two registered EasyICU exports before starting the cross-database comparison.', '开始跨库对比前，请至少选择两个已注册的 EasyICU 导出。')
        : t('Registered export background-job API is unavailable in this browser session.', '当前浏览器会话无法使用已注册导出的后台任务 API。');
      done && done(false);
      return;
    }
    const setupSnapshot = opts && opts.setup ? opts.setup : crossSetup.snapshot(crossSetupConfig());
    const rawRoot = requestedRawRoot || setupSnapshot.rawRoot || '';
    const rawDatabases = Array.isArray(setupSnapshot.selectedKeys)
      ? setupSnapshot.selectedKeys.slice()
      : crossSetup.selectedKeys();
    const sampleProfile = setupSnapshot.sampleProfile || crossSetup.sampleProfile(crossSetupConfig());
    if (!rawRoot) {
      vizErr = t('Choose a local ICU data root before loading real Cross-DB densities.', '加载真实跨库密度前，请先选择本地 ICU 数据根目录。');
      done && done(false);
      return;
    }
    if (!crossSetup.scanReady(rawRoot)) {
      vizErr = t('Check the ICU data root first so EasyICU can confirm at least two selected database folders.', '请先检查 ICU 数据根目录，确认至少两个已选数据库文件夹可识别。');
      done && done(false);
      return;
    }
    if (rawRoot && rawDatabases.length >= 2 && window.EU_API && window.EU_API.startCrossdbRawDistributionJob && jobContinuity && typeof jobContinuity.start === 'function' && window.EventSource) {
      if (crossRawProgress.snapshot().starting) return;
      crossRawProgress.beginStart();
      crossSetup.setRawRoot(rawRoot);
      const rawRequest = window.EU_CROSSDB_RAW.buildRequest({
        dataRoot: rawRoot,
        databases: rawDatabases,
        featureScope: setupSnapshot.featureScope || crossSetup.featureScope(),
        maxPatients: sampleProfile.maxPatients,
        sampleSize: sampleProfile.sampleSize,
      });
      window.EU_API.startCrossdbRawDistributionJob(rawRequest).then(r => {
        if (!operationActive()) {
          if (r && r.job_id && window.EU_API.cancelJob) {
            window.EU_API.cancelJob(r.job_id, 'client_operation_invalidated').catch(() => {});
          }
          return;
        }
        const queuedProgress = {
          job_kind: 'crossdb-raw-distribution',
          phase: 'queued',
          max_patients: sampleProfile.maxPatients,
          sample_size: sampleProfile.sampleSize,
          message: `${t('Queued local raw Cross-DB density job.', '本地原始跨库密度任务已排队。')} ${crossSetup.sampleSummary(sampleProfile, crossSetupConfig())}`,
        };
        crossRawProgress.attach(r && r.job_id, queuedProgress);
        const watching = jobContinuity.start({
          job_id: r && r.job_id,
          kind: r && r.kind,
          raw_root: rawRoot,
          source_identity: crossSetup.sourceIdentity(rawDatabases),
          sample_mode: setupSnapshot.sampleMode || crossSetup.sampleMode(),
          feature_scope: rawRequest.feature_scope,
        }, queuedProgress, done);
        if (!watching) {
          crossRawProgress.clear();
          vizErr = t('Could not keep the raw Cross-DB job attached to this browser session.', '无法将原始跨库任务绑定到当前浏览器会话。');
          if (r && r.job_id && window.EU_API.cancelJob) window.EU_API.cancelJob(r.job_id, 'client_attach_failed').catch(() => {});
          done && done(false);
        } else {
          crossRawProgress.flushCancel(window.EU_API, err => {
            vizErr = String(err && err.message || err);
            repaintCrossRawProgress();
          });
        }
        repaintScreen('crossdb');
      }).catch(err => {
        if (!operationActive()) return;
        crossRawProgress.clear();
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
  window.EU_CROSSDB_JOB_HOST = {
    canRestore(meta) {
      if (displayDataMode() !== 'real' || crossSetup.view() !== 'idle' || window.EU_CROSSDB_WORKSPACE) return false;
      if (meta && meta.kind === 'crossdb-summary') return registeredSelectionAvailable(meta);
      return crossSetup.sourceMethod() === 'raw';
    },
    acceptResume(meta) {
      if (meta && meta.kind === 'crossdb-summary') {
        if (!registeredSelectionAvailable(meta)) return false;
        crossSetup.setSourceMethod('registered');
        return true;
      }
      return crossSetup.acceptResume(meta);
    },
    matchesSource(meta) {
      if (meta && meta.kind === 'crossdb-summary') return registeredSelectionAvailable(meta);
      return crossSetup.matchesSource(meta);
    },
    onProbe(meta) {
      const registered = meta && meta.kind === 'crossdb-summary';
      crossRawProgress.beginProbe(meta.job_id, {
        job_kind: meta && meta.kind,
        phase: 'reconnect',
        message: registered
          ? t('Checking the saved registered-export Cross-DB job…', '正在检查已保存的注册导出跨库任务…')
          : t('Checking the saved raw Cross-DB job…', '正在检查已保存的原始跨库任务…'),
      });
      crossSetup.setRegisteredLoading(false);
      crossSetup.setView('loading');
      vizErr = null;
      repaintScreen('crossdb');
    },
    onRunning(meta, progress, history) {
      const registered = meta && meta.kind === 'crossdb-summary';
      crossRawProgress.resume(meta.job_id, progress && (progress.type === 'progress' || progress.type === 'cancel_requested')
        ? { ...progress, job_kind: meta && meta.kind }
        : {
          job_kind: meta && meta.kind,
          phase: 'reconnect',
          message: registered
            ? t('Reconnected to the registered-export Cross-DB summary job.', '已重新连接注册导出的跨库摘要任务。')
            : t('Reconnected to the running raw Cross-DB density job.', '已重新连接正在运行的原始跨库密度任务。'),
        }, history);
      crossSetup.setRegisteredLoading(false);
      crossSetup.setView('loading');
      vizErr = null;
      repaintScreen('crossdb');
    },
    onProgress(meta, progress) {
      if (!crossRawProgress.appliesTo(meta.job_id)) return;
      if (crossRawProgress.applyProgress({ ...progress, job_kind: meta && meta.kind })) repaintCrossRawProgress();
    },
    onCancelRequested(meta, event) {
      if (!crossRawProgress.appliesTo(meta.job_id)) return;
      crossRawProgress.applyCancelRequested(event);
      repaintCrossRawProgress();
    },
    onTerminal(meta, snapshot) {
      if (!window.EU_CROSSDB_JOB_HOST.matchesSource(meta) || !crossRawProgress.appliesTo(meta.job_id)) return false;
      const registered = meta && meta.kind === 'crossdb-summary';
      let accepted = true;
      if (snapshot.status === 'done') {
        const xdb = snapshot.result;
        let validResult = false;
        if (registered) {
          const receipt = xdb && xdb.selection_receipt;
          const loadedIdentity = receiptSelectionIdentity(receipt);
          validResult = !!(xdb && typeof xdb === 'object' && xdb.ok === true && xdb.mode === 'real'
            && receipt && receipt.selection_digest === meta.selection_digest
            && loadedIdentity === meta.source_identity
            && Number(xdb.source_count || 0) === meta.source_identity.split(',').length);
        } else {
          const expectedDatabases = crossSetup.identityKeys(meta);
          const loadedDatabases = Array.isArray(xdb && xdb.sources)
            ? xdb.sources.map(row => String(row && row.database || '')).filter(Boolean).sort()
            : [];
          validResult = !!(xdb && typeof xdb === 'object' && xdb.ok === true && xdb.source_type === 'raw_database_root'
            && Number(xdb.source_count || 0) === expectedDatabases.length
            && loadedDatabases.join(',') === expectedDatabases.join(','));
        }
        if (!validResult) {
          accepted = false;
          crossSetup.setView('idle');
          vizErr = registered
            ? t('The restored registered-export job returned a mismatched selection receipt.', '恢复的注册导出任务返回了不匹配的选择回执。')
            : t('The restored raw Cross-DB job returned an invalid result.', '恢复的原始跨库任务返回了无效结果。');
        } else {
          window.EU_CROSSDB_WORKSPACE = xdb;
          const first = xdb.sources && xdb.sources[0];
          if (first) window.EU_VIZ_WORKSPACE = { route: 'crossdb', database: first.database, summary: first.summary };
          window.EU_HASWORK = true;
          crossSetup.setView('loaded');
          vizErr = null;
        }
      } else {
        window.EU_CROSSDB_WORKSPACE = null;
        crossSetup.setView('idle');
        if (snapshot.status === 'cancelled') {
          vizErr = registered
            ? t('Registered-export Cross-DB summary job cancelled before completion.', '注册导出的跨库摘要任务已在完成前取消。')
            : t('Raw Cross-DB density job cancelled before completion.', '原始跨库密度任务已在完成前取消。');
        } else {
          vizErr = snapshot.error || (registered
            ? t('Registered-export Cross-DB summary job failed.', '注册导出的跨库摘要任务失败。')
            : t('Raw Cross-DB density job failed.', '原始跨库密度任务失败。'));
        }
      }
      crossSetup.setRegisteredLoading(false);
      crossRawProgress.clear();
      repaintScreen('crossdb');
      return accepted;
    },
    onUnavailable(meta) {
      const registered = meta && meta.kind === 'crossdb-summary';
      crossRawProgress.clear();
      crossSetup.setRegisteredLoading(false);
      crossSetup.setView('idle');
      vizErr = registered
        ? t('This saved registered-export job is no longer available; the local server may have restarted. Start the comparison again.', '已保存的注册导出任务已不可用；本地服务可能已重启。请重新开始对比。')
        : t('This saved raw Cross-DB job is no longer available; the local server may have restarted. Start it again from this data root.', '已保存的原始跨库任务已不可用；本地服务可能已重启。请从当前数据根目录重新运行。');
      repaintScreen('crossdb');
    },
    onConnectionError(meta) {
      const registered = meta && meta.kind === 'crossdb-summary';
      crossRawProgress.clear();
      crossSetup.setRegisteredLoading(false);
      crossSetup.setView('idle');
      vizErr = registered
        ? t('Could not reconnect to the saved registered-export job. Refresh to retry; no completed result was assumed.', '无法重新连接已保存的注册导出任务。请刷新重试；系统没有假定任务已完成。')
        : t('Could not reconnect to the saved raw Cross-DB job. Refresh to try again; no completed result was assumed.', '无法重新连接已保存的原始跨库任务。请刷新后重试；系统没有假定任务已完成。');
      repaintScreen('crossdb');
    },
  };
  window.EU_CROSSDB_SOURCE_HOST = {
    registrySources() {
      return registrySources();
    },
    registeredPaths() {
      return explicitRegistryCrossdbPaths();
    },
    officialPaths() {
      const owner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
      return owner && typeof owner.registeredSources === 'function'
        ? owner.registeredSources(registrySources()).map(source => source.path)
        : [];
    },
    openOfficial(sourceId) {
      const owner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
      const source = owner && typeof owner.source === 'function' ? owner.source(sourceId) : null;
      if (!source || !source.status || source.status.state !== 'prepared' || !source.status.registered) {
        vizErr = t('Prepare this official demo before using it in the cross-database comparison.', '请先准备好这个官方 Demo，再用于跨库对比。');
        repaintScreen('crossdb');
        return;
      }
      window.EU_CROSSDB_SOURCE_HOST.runOfficial();
    },
    runOfficial() {
      if (crossSetup.view() === 'loading') return;
      const owner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
      const hydration = window.EU_API && window.EU_API.hydrateWorkspaceRegistry
        ? window.EU_API.hydrateWorkspaceRegistry()
        : Promise.resolve();
      Promise.resolve(hydration).then(() => {
        const pair = owner && typeof owner.rememberPair === 'function'
          ? owner.rememberPair(registrySources())
          : [];
        const paths = pair.map(source => source.path);
        if (paths.length < 2) {
          vizErr = t('Prepare both the MIMIC-IV and eICU official demos before starting the cross-database comparison.', '开始跨库对比前，请准备好 MIMIC-IV 与 eICU 两个官方 Demo。');
          repaintScreen('crossdb');
          return;
        }
        window.EU_DATA = 'real';
        vizErr = null;
        const operationId = crossSetup.beginOperation();
        crossSetup.setRegisteredLoading(true);
        crossSetup.setView('loading');
        repaintScreen('crossdb');
        loadRealCrossdb(ok => {
          if (!crossSetup.operationCurrent(operationId)) return;
          crossSetup.setRegisteredLoading(false);
          crossSetup.setView(ok ? 'loaded' : 'idle');
          repaintScreen('crossdb');
        }, { operationId, registeredPaths: paths });
      }).catch(error => {
        vizErr = String((error && error.message) || error);
        repaintScreen('crossdb');
      });
    },
    runRegistered() {
      if (crossSetup.view() === 'loading') return;
      const paths = window.EU_CROSSDB_SOURCE_HOST.registeredPaths();
      if (paths.length < 2) {
        vizErr = t('Select at least two registered EasyICU exports below.', '请在下方至少选择两个已注册的 EasyICU 导出。');
        repaintScreen('crossdb');
        return;
      }
      vizErr = null;
      const operationId = crossSetup.beginOperation();
      crossSetup.setRegisteredLoading(true);
      crossSetup.setView('loading');
      repaintScreen('crossdb');
      loadRealCrossdb(ok => {
        if (!crossSetup.operationCurrent(operationId)) return;
        crossSetup.setRegisteredLoading(false);
        crossSetup.setView(ok ? 'loaded' : 'idle');
        repaintScreen('crossdb');
      }, { operationId, registeredPaths: paths });
    },
    repaint() {
      repaintScreen('crossdb');
    },
  };
  function loadDemoCrossdb(done, opts) {
    const operationId = opts && opts.operationId;
    const operationActive = () => operationId == null || crossSetup.operationCurrent(operationId);
    if (!operationActive()) return;
    window.EU_CROSSDB_WORKSPACE = null;
    window.EU_VIZ_WORKSPACE = null;
    crossResults.reset();
    const databases = opts && Array.isArray(opts.selectedKeys)
      ? opts.selectedKeys.slice()
      : crossSetup.selectedKeys();
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
      if (!operationActive()) return;
      window.EU_CROSSDB_WORKSPACE = xdb;
      const first = xdb.sources && xdb.sources[0];
      if (first) window.EU_VIZ_WORKSPACE = { route: 'crossdb', database: first.database, summary: first.summary };
      vizErr = null;
      window.EU_HASWORK = true;
      done && done(true);
    }).catch(err => {
      if (!operationActive()) return;
      window.EU_CROSSDB_WORKSPACE = null;
      vizErr = String(err && err.message || err);
      done && done(false);
    });
  }

  /* allow the print harness to preset loaded states for a richer PDF */
  window.__euVizPreset = function (which) {
    if (!which || which === 'patient') patientReview.presetLoaded();
    if (!which || which === 'crossdb') crossSetup.presetLoaded();
  };
  window.__euVizResetForDataMode = function () {
    crossSetup.resetForDataMode(crossSetupConfig());
    patientReview.resetForDataMode();
    cohortOwner.resetForSourceChange();
    vizErr = null;
  };

  function activeVizRoute() {
    const raw = (location.hash || '#entry').slice(1).trim();
    if (raw === 'audit' || raw === 'sofareclass') return 'cohort';
    if (raw === 'icd') return 'extraction';
    return raw;
  }
  function repaintScreen(id) {
    // Background events (SSE progress, async loads resolving late) must never
    // repaint a screen the user is not on: the full-shell re-render wipes
    // focus, IME composition, and uncommitted input on unrelated routes.
    // Module state is already updated; the target screen re-renders from
    // state on the next visit.
    if (id && activeVizRoute() !== id) return;
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

  function skeletonWorkspace(mode) {
    const real = mode === 'real';
    const title = real
      ? t('Reading bounded Patient Review from local export…', '正在从本地导出读取有界患者审阅…')
      : t('Generating demo review workspace…', '正在生成演示审阅工作区…');
    const detail = real
      ? t('local-only · bounded browser payload · no outbound calls', '仅本地 · 有界浏览器载荷 · 无外部调用')
      : t('reproducible · no outbound calls', '可复现 · 无外部调用');
    return `
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow">
          <div style="font-weight:600;font-size:12.75px;">${title}</div>
          <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${detail}</div>
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

  /* ---------------- CROSS-DB BENCHMARK ---------------- */
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
      '12 curated core concepts': '12 个精选核心概念',
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
      resolving: '解析中',
      loading: '加载中',
      database: '逐库加载',
      chunk: '分块加载',
      finalizing: '汇总中',
      reconnect: '重新连接中',
      cancel: '取消中',
      running: '运行中',
      pending: '等待中',
      complete: '已完成',
      empty: '无可用值',
      stopping: '停止中',
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
    const xdbRaw = xdb && xdb.source_type === 'raw_database_root';
    const dataMode = displayDataMode();
    const officialDemo = officialDemoContext();
    const sourceLabel = xdb
      ? (xdbDemo ? crossTerm('Demo simulated frames') : (xdbRaw ? t('Local raw databases', '本地原始数据库') : (officialDemo ? t('Official demo pair', '官方 Demo 组合') : crossTerm('Local exports'))))
      : (ws ? crossTerm('Local export') : (dataMode === 'real' ? crossTerm('Not configured') : crossTerm('Demo cohort')));
    return `
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>${crossTerm('Workspace')}</span> ${icon('chevron', 11)} <span>${sourceLabel}</span> ${icon('chevron', 11)} <span style="color:var(--ink-2);">${t('Cross-database comparison', '跨库对比')}</span>
      </div>
      <div class="page-head" style="margin-bottom:14px;">
        <h1 style="margin-top:0;">${t('Cross-database comparison', '跨库对比')}</h1>
        <p class="lead">${dataMode === 'real' ? t('Check concept coverage and aggregate distributions across prepared exports or locally sampled ICU databases.', '检查已准备导出或本地抽样 ICU 数据库之间的概念覆盖和聚合分布。') : t('Compare the official MIMIC-IV and eICU demo exports through the same aggregate pipeline used for local data.', '通过与本地数据相同的聚合流程，对比官方 MIMIC-IV 与 eICU Demo 导出。')}</p>
      </div>`;
  }
  S.crossdb = {
    section: 'viz', nav: 'viz', sub: 'crossdb', wide: true,
    crumbs: ['Home', 'Data Workspace', 'Cross-database comparison'],
    get actionHtml() {
      return crossSetup.actionHtml(crossSetupConfig());
    },
    rail: () => vizRail('crossdb'),
    render() {
      return crossSetup.renderBody(crossSetupConfig());
    },
    afterRender(root) {
      crossSetup.bind(root, crossSetupConfig());
      crossResults.mount(root);
      crossResults.bind(root, window.EU_CROSSDB_WORKSPACE, crossResultsConfig());
    },
  };

  cohortOwner.init({
    t,
    icon,
    esc,
    fmtInt,
    fmtNum,
    fmtPct,
    fmtP,
    displayDataMode,
    officialDemoContext,
    registryActivePath,
    bindSourceRegistry,
    sourceModeSelector,
    sourceRegistryBlock,
    repaintScreen,
    vizRail,
    workspaceSamplingNote,
    cohortCharts: window.EU_COHORT_CHARTS,
  });

  patientReview.init({
    t,
    icon,
    esc,
    fmtInt,
    fmtNum,
    fmtPct,
    axisSpark,
    workspaceSamplingNote,
    vizRail,
    registrySources,
    registryActivePath,
    sourceLine,
    sourceRegistryBlock,
    bindSourceRegistry,
    repaintScreen,
    skeletonWorkspace,
    downloadJsonFile,
  });

  const vizContextOwner = window.EU_VIZ_CONTEXT_OWNER; if (vizContextOwner && typeof vizContextOwner.init === 'function') {
    vizContextOwner.init({
      activePath: registryActivePath, sources: registrySources, defaultExportPath, patient: patientReview.drilldown, cohort: cohortOwner.review,
      cohortComparison: cohortOwner.comparison, cohortOutcome: cohortOwner.outcome, crossdb: () => window.EU_CROSSDB_WORKSPACE || {},
      hydrate: {
        patient: patientReview.hydrate,
        cohort: cohortOwner.hydrate,
        crossdb(payload) { window.EU_DATA = 'real'; window.EU_CROSSDB_WORKSPACE = payload; crossSetup.presetLoaded(); const first = (payload.sources || [])[0]; window.EU_VIZ_WORKSPACE = first ? { route: 'crossdb', database: first.database, summary: first.summary } : null; vizErr = null; window.EU_HASWORK = true; },
      },
      cohortBegin: cohortOwner.beginCharts,
      cohortPanels: cohortOwner.panels,
      cohortMount: cohortOwner.mountCharts,
      patientSeriesHelpers: { t, esc, fmtInt, fmtNum, fmtPct, icon, axisSpark, signalLabel: patientReview.signalLabel, seriesLabel: patientReview.seriesLabel, signalKey: patientReview.signalKey, demoHours: () => null },
      crossdbResultsConfig(repaint) { const config = crossResultsConfig(); return { ...config, repaint: typeof repaint === 'function' ? repaint : config.repaint, expandScope: null, exportPayload: null }; },
    });
  }
})();
