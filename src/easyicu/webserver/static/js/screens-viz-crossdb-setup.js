/* Cross-DB setup owner: source selection, raw-root scan, sample budget, and route setup state. */
(function () {
  'use strict';

  const DATABASES = [
    { label: 'MIMIC-IV', key: 'miiv', selected: true },
    { label: 'eICU-CRD', key: 'eicu', selected: true },
    { label: 'AmsterdamUMCdb', key: 'aumc', selected: true },
    { label: 'HiRID', key: 'hirid', selected: true },
    { label: 'MIMIC-III', key: 'mimic', selected: true },
    { label: 'SICdb', key: 'sic', selected: true },
  ];
  const SAMPLE_SPECS = [
    {
      id: 'quick',
      label: ['Quick preview', '快速预览'],
      note: ['Fast first look for module-level density checks.', '优先快速看模块级分布。'],
      maxPatients: 200,
      sampleSize: 600,
    },
    {
      id: 'standard',
      label: ['Standard sample', '标准抽样'],
      note: ['Balanced default for smoother density curves.', '平衡速度和曲线稳定性。'],
      maxPatients: 300,
      sampleSize: 1500,
    },
    {
      id: 'deeper',
      label: ['Deeper sample', '较深抽样'],
      note: ['More stable, but can take longer on six raw databases.', '更稳定，但六库原始数据会更慢。'],
      maxPatients: 800,
      sampleSize: 3000,
    },
  ];
  const SAMPLE_MODES = new Set(SAMPLE_SPECS.map(row => row.id));
  const FEATURE_SCOPES = new Set(['core', 'all']);
  const state = {
    view: 'idle',
    sourceMethod: 'registered',
    rawRootDraft: '',
    rawRootScan: null,
    rawRootScanPath: '',
    rawRootScanning: false,
    scanRequestSeq: 0,
    operationSeq: 0,
    sampleMode: 'quick',
    featureScope: 'all',
    registeredLoading: false,
  };

  function helpersOf(config) {
    return Object.assign({
      t: english => english,
      esc: value => String(value == null ? '' : value),
      fmtInt: value => String(value == null ? '—' : value),
      icon: () => '',
      term: value => String(value == null ? '' : value),
      statusLabel: value => String(value == null ? '' : value),
      progressMessage: value => String(value == null ? '' : value),
    }, config && config.helpers || {});
  }

  function pathValue(value) {
    return String(value || '').trim();
  }

  function selectedKeys() {
    return DATABASES.filter(row => row.selected).map(row => row.key);
  }

  function databaseRows() {
    return DATABASES.map(row => ({ ...row }));
  }

  function setSelectedKeys(values) {
    const selected = new Set((values || []).map(value => String(value || '').trim()).filter(Boolean));
    DATABASES.forEach(row => { row.selected = selected.has(row.key); });
    return selectedKeys();
  }

  function toggleDatabase(index) {
    const row = DATABASES[Number(index)];
    if (!row) return null;
    row.selected = !row.selected;
    return { ...row };
  }

  function sourceIdentity(values) {
    return Array.from(new Set((values || selectedKeys()).map(key => String(key || '').trim()).filter(Boolean))).sort().join(',');
  }

  function identityKeys(meta) {
    const identity = String(meta && meta.source_identity || '').trim();
    const keys = Array.from(new Set(identity.split(',').map(key => key.trim()).filter(Boolean))).sort();
    const known = new Set(DATABASES.map(row => row.key));
    if (keys.length < 2 || keys.some(key => !known.has(key)) || keys.join(',') !== identity) return [];
    return keys;
  }

  function sampleProfiles(config) {
    const h = helpersOf(config);
    return SAMPLE_SPECS.map(row => ({
      id: row.id,
      label: h.t(row.label[0], row.label[1]),
      note: h.t(row.note[0], row.note[1]),
      maxPatients: row.maxPatients,
      sampleSize: row.sampleSize,
    }));
  }

  function sampleProfile(config) {
    const profiles = sampleProfiles(config);
    return profiles.find(row => row.id === state.sampleMode) || profiles[0];
  }

  function sampleSummary(profile, config) {
    const h = helpersOf(config);
    const current = profile || sampleProfile(config);
    return `${current.label} · ≤${h.fmtInt(current.maxPatients)} ${h.t('entities/db', '实体/库')} · ≤${h.fmtInt(current.sampleSize)} ${h.t('values/feature', '值/特征')}`;
  }

  function rawRoot() {
    return state.rawRootDraft;
  }

  function setRawRoot(value) {
    state.rawRootDraft = pathValue(value);
    return state.rawRootDraft;
  }

  function changeRawRoot(value) {
    const next = pathValue(value);
    if (next !== state.rawRootDraft) invalidateScan();
    state.rawRootDraft = next;
    return next;
  }

  function sampleMode() {
    return state.sampleMode;
  }

  function setSampleMode(value) {
    state.sampleMode = SAMPLE_MODES.has(value) ? value : 'quick';
    return state.sampleMode;
  }

  function featureScope() {
    return state.featureScope;
  }

  function setFeatureScope(value) {
    state.featureScope = FEATURE_SCOPES.has(value) ? value : 'core';
    return state.featureScope;
  }

  function view() {
    return state.view;
  }

  function setView(value) {
    state.view = ['idle', 'loading', 'loaded'].includes(value) ? value : 'idle';
    return state.view;
  }

  function registeredLoading() {
    return state.registeredLoading;
  }

  function setRegisteredLoading(value) {
    state.registeredLoading = value === true;
    return state.registeredLoading;
  }

  function sourceMethod() {
    return state.sourceMethod;
  }

  function setSourceMethod(value) {
    state.sourceMethod = value === 'raw' ? 'raw' : 'registered';
    return state.sourceMethod;
  }

  function beginOperation() {
    state.operationSeq += 1;
    return state.operationSeq;
  }

  function invalidateOperations() {
    state.operationSeq += 1;
    return state.operationSeq;
  }

  function operationCurrent(value) {
    return Number.isInteger(value) && value === state.operationSeq;
  }

  function invalidateScan() {
    state.scanRequestSeq += 1;
    state.rawRootScan = null;
    state.rawRootScanPath = '';
    state.rawRootScanning = false;
  }

  function selectionStatus(path) {
    const root = pathValue(path);
    const current = root && state.rawRootScan && state.rawRootScanPath === root
      ? state.rawRootScan
      : null;
    const selected = selectedKeys();
    const detectedKeys = new Set(
      current && current.ok
        ? (current.detected || []).map(row => row && row.key).filter(Boolean)
        : []
    );
    const detectedSelectedKeys = selected.filter(key => detectedKeys.has(key));
    const missingSelectedKeys = selected.filter(key => !detectedKeys.has(key));
    return {
      current,
      selectedKeys: selected,
      detectedKeys,
      detectedSelectedKeys,
      missingSelectedKeys,
      runnable: !!(root && !state.rawRootScanning && current && current.ok && detectedSelectedKeys.length >= 2 && missingSelectedKeys.length === 0),
    };
  }

  function scanReady(path) {
    return selectionStatus(path).runnable;
  }

  function scanCurrent(path) {
    const root = pathValue(path);
    return !!(root && state.rawRootScan && state.rawRootScanPath === root);
  }

  function setError(config, value) {
    if (config && typeof config.setError === 'function') config.setError(value == null ? null : String(value));
  }

  function getError(config) {
    return config && typeof config.getError === 'function' ? config.getError() : null;
  }

  function userError(value, config) {
    const raw = String(value || '').trim();
    if (!raw) return '';
    const h = helpersOf(config);
    const messages = {
      raw_database_concept_load_failed: h.t(
        'A selected database could not load one catalog feature. EasyICU stopped instead of reporting a false absence. Check the source files or use prepared exports.',
        '某个已选数据库无法读取一项目录特征。EasyICU 已停止，避免把读取失败误报为数据缺失；请检查源文件或改用已准备导出。'
      ),
      raw_distribution_load_failed: h.t(
        'The bounded raw-database comparison could not be completed. Check the source folders and retry.',
        '有界原始库对比未能完成。请检查源目录后重试。'
      ),
      loaded_fewer_than_requested_raw_databases: h.t(
        'Not every selected database produced readable aggregate data. Open Advanced settings to review the database selection.',
        '并非所有已选数据库都产生了可读聚合数据。请打开“高级设置”检查数据库选择。'
      ),
      requested_raw_databases_not_found: h.t(
        'One or more selected database folders were not recognized. Open Advanced settings to review detected databases.',
        '一个或多个已选数据库目录未被识别。请打开“高级设置”检查已识别数据库。'
      ),
    };
    return messages[raw] || raw;
  }

  function repaint(config, focusSelector) {
    if (config && typeof config.repaint === 'function') config.repaint();
    if (focusSelector && typeof document !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => {
        const target = document.querySelector(focusSelector);
        if (target && typeof target.focus === 'function') target.focus();
      });
    }
  }

  function scan(path, config, focusSelector) {
    const h = helpersOf(config);
    const root = pathValue(path || state.rawRootDraft);
    state.rawRootDraft = root;
    if (!root) {
      invalidateScan();
      setError(config, h.t('Choose a local ICU data root before checking Cross-DB folders.', '检查跨库文件夹前，请先选择本地 ICU 数据根目录。'));
      repaint(config, focusSelector);
      return Promise.resolve(false);
    }
    const api = config && config.api || window.EU_API;
    if (!api || typeof api.scanCrossdbRawRoot !== 'function') {
      invalidateScan();
      setError(config, h.t('Raw Cross-DB folder check API is unavailable in this browser session.', '当前浏览器会话无法使用原始跨库文件夹检查 API。'));
      repaint(config, focusSelector);
      return Promise.resolve(false);
    }
    if (state.rawRootScanning && state.rawRootScanPath === root) return Promise.resolve(false);
    state.rawRootScanning = true;
    state.rawRootScan = null;
    state.rawRootScanPath = root;
    const requestSeq = ++state.scanRequestSeq;
    setError(config, null);
    repaint(config, focusSelector);
    return api.scanCrossdbRawRoot({
      data_root: root,
      databases: selectedKeys(),
    }).then(result => {
      if (requestSeq !== state.scanRequestSeq || state.rawRootDraft !== root) return false;
      state.rawRootScanning = false;
      state.rawRootScan = result || null;
      state.rawRootScanPath = root;
      const detected = result && result.ok
        ? (result.detected || []).map(row => row && row.key).filter(Boolean)
        : [];
      // The default means "all databases found in this parent folder", not
      // "require six hard-coded folders". Preserve explicit advanced choices.
      if (detected.length >= 2 && selectedKeys().length === DATABASES.length) {
        setSelectedKeys(detected);
        notifySourceChanged(root, state.sampleMode, state.featureScope);
      }
      setError(config, result && result.ok === false
        ? result.hint || result.error || 'Could not check that raw ICU data root.'
        : null);
      repaint(config, focusSelector);
      return scanReady(root);
    }).catch(error => {
      if (requestSeq !== state.scanRequestSeq || state.rawRootDraft !== root) return false;
      state.rawRootScanning = false;
      state.rawRootScan = null;
      state.rawRootScanPath = root;
      setError(config, String(error && error.message || error));
      repaint(config, focusSelector);
      return false;
    });
  }

  function acceptResume(meta) {
    const root = pathValue(meta && meta.raw_root);
    const keys = identityKeys(meta);
    if (!root || !keys.length || (state.rawRootDraft && state.rawRootDraft !== root)) return false;
    state.rawRootDraft = root;
    setSampleMode(meta && meta.sample_mode);
    setFeatureScope(meta && meta.feature_scope === 'all_catalog' ? 'all' : 'core');
    setSelectedKeys(keys);
    return true;
  }

  function matchesSource(meta) {
    return pathValue(meta && meta.raw_root) === state.rawRootDraft
      && String(meta && meta.source_identity || '') === sourceIdentity();
  }

  function snapshot(config) {
    const profile = sampleProfile(config);
    return {
      view: state.view,
      rawRoot: state.rawRootDraft,
      selectedKeys: selectedKeys(),
      sourceIdentity: sourceIdentity(),
      sourceMethod: state.sourceMethod,
      sampleMode: state.sampleMode,
      sampleProfile: { ...profile },
      featureScope: state.featureScope,
      registeredLoading: state.registeredLoading,
      scanCurrent: scanCurrent(state.rawRootDraft),
      scanReady: scanReady(state.rawRootDraft),
    };
  }

  function disconnectJob(options) {
    const continuity = window.EU_CROSSDB_JOB_CONTINUITY;
    if (continuity && typeof continuity.disconnect === 'function') continuity.disconnect(options || {});
  }

  function notifySourceChanged(nextRoot, nextMode, nextFeatureScope) {
    const continuity = window.EU_CROSSDB_JOB_CONTINUITY;
    if (continuity && typeof continuity.onSourceChanged === 'function') {
      const apiScope = window.EU_CROSSDB_RAW && typeof window.EU_CROSSDB_RAW.apiFeatureScope === 'function'
        ? window.EU_CROSSDB_RAW.apiFeatureScope(nextFeatureScope || state.featureScope)
        : (nextFeatureScope === 'all' || state.featureScope === 'all' ? 'all_catalog' : 'curated_core');
      continuity.onSourceChanged(pathValue(nextRoot), sourceIdentity(), nextMode || state.sampleMode, apiScope);
    }
  }

  function repaintProgress(config) {
    const restoreCancelFocus = typeof document !== 'undefined'
      && document.activeElement
      && document.activeElement.matches('[data-crossdb-cancel]');
    repaint(config);
    if (restoreCancelFocus && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => {
        const button = document.querySelector('[data-crossdb-cancel]');
        if (button) button.focus();
      });
    }
  }

  function cancel(config) {
    const progress = window.EU_CROSSDB_PROGRESS;
    if (!progress || typeof progress.requestCancel !== 'function') return;
    progress.requestCancel({
      api: config && config.api || window.EU_API,
      onStateChange() {
        state.view = 'loading';
        setError(config, null);
        repaintProgress(config);
      },
      onError(error) {
        setError(config, String(error && error.message || error));
        repaintProgress(config);
      },
    });
  }

  function rawJobActive() {
    const progress = window.EU_CROSSDB_PROGRESS;
    const current = progress && typeof progress.snapshot === 'function' ? progress.snapshot() : null;
    return !!(current && (current.starting || current.jobId));
  }

  function aliasSummary(scanResult) {
    const aliases = scanResult && scanResult.aliases ? scanResult.aliases : {};
    return Object.keys(aliases).map(key => {
      const row = aliases[key] || {};
      const names = (row.aliases || []).slice(0, 4).join('/');
      return `${row.label || key}: ${names}`;
    }).join(' · ');
  }

  function databaseLabel(key) {
    const row = DATABASES.find(item => item.key === key);
    return row ? row.label : String(key || '');
  }

  function databaseStatus(key, selected, config) {
    const h = helpersOf(config);
    const root = state.rawRootDraft;
    if (state.rawRootScanning && state.rawRootScanPath === root) {
      return { cls: 'dashed', label: h.t('checking', '检查中'), sub: h.t('checking folder', '正在检查文件夹') };
    }
    const current = scanCurrent(root) ? state.rawRootScan : null;
    if (!current || current.ok === false) {
      return {
        cls: selected ? 'ok' : 'dashed',
        label: selected ? h.t('selected', '已选择') : h.t('add', '添加'),
        sub: h.t('not checked yet', '尚未检查'),
      };
    }
    const detected = (current.detected || []).find(row => row.key === key);
    if (detected) {
      return {
        cls: selected ? 'ok' : 'dashed',
        label: selected ? h.t('detected', '已识别') : h.t('not selected', '未选择'),
        sub: detected.folder_name ? `${h.t('folder', '文件夹')} ${detected.folder_name}` : h.t('recognized folder', '已识别文件夹'),
      };
    }
    return {
      cls: selected ? 'warn' : 'dashed',
      label: selected ? h.t('missing', '缺失') : h.t('not found', '未找到'),
      sub: selected ? h.t('not found in root', '根目录中未找到') : h.t('not detected', '未识别'),
    };
  }

  function renderScanPanel(config) {
    const h = helpersOf(config);
    const root = state.rawRootDraft;
    if (!root) {
      return `<div class="note info mt-12" role="status" aria-live="polite" aria-atomic="true">
        <div class="ico">${h.icon('folder', 14)}</div>
        <div class="body"><div class="t">${h.t('Choose a parent folder first', '先选择一个父文件夹')}</div><div class="d">${h.t('It should contain database subfolders. Accepted aliases include mimiciv, mimic-iv, miiv, eicu, eicu-crd, aumc, amsterdamumc, hirid, mimiciii, sicdb, and sic.', '它应包含数据库子文件夹。可识别别名包括 mimiciv、mimic-iv、miiv、eicu、eicu-crd、aumc、amsterdamumc、hirid、mimiciii、sicdb、sic。')}</div></div>
      </div>`;
    }
    if (state.rawRootScanning && state.rawRootScanPath === root) {
      return `<div class="note info mt-12" role="status" aria-live="polite" aria-atomic="true">
        <div class="ico"><span class="spin accent"></span></div>
        <div class="body"><div class="t">${h.t('Checking database folders', '正在检查数据库文件夹')}</div><div class="d">${h.t('EasyICU is matching top-level folders against supported database aliases. No patient rows are read.', 'EasyICU 正在用支持的数据库别名匹配顶层文件夹；不会读取患者行。')}</div></div>
      </div>`;
    }
    if (!scanCurrent(root)) {
      return `<div class="note warn mt-12" role="status" aria-live="polite" aria-atomic="true">
        <div class="ico">${h.icon('alert', 14)}</div>
        <div class="body"><div class="t">${h.t('Folder not checked', '文件夹尚未检查')}</div><div class="d">${h.t('Check this root before running so missing or custom-named database folders are visible.', '运行前先检查这个根目录，这样缺失或自定义命名的数据库文件夹会直接显示出来。')}</div></div>
      </div>`;
    }
    const result = state.rawRootScan || {};
    if (result.ok === false) {
      return `<div class="note warn mt-12" role="status" aria-live="polite" aria-atomic="true">
        <div class="ico">${h.icon('alert', 14)}</div>
        <div class="body"><div class="t">${h.t('Folder check failed', '文件夹检查失败')}</div><div class="d">${h.esc(result.hint || result.error || h.t('Could not check this folder.', '无法检查该文件夹。'))}</div></div>
      </div>`;
    }
    const status = selectionStatus(root);
    const detected = result.detected || [];
    const detectedKeys = detected.map(row => row && row.key).filter(Boolean);
    const detectedSelectionDiffers = detectedKeys.length >= 2
      && sourceIdentity(detectedKeys) !== sourceIdentity(status.selectedKeys);
    const selectedDetected = new Set(status.detectedSelectedKeys);
    const missing = status.missingSelectedKeys.map(key => ({ key, label: databaseLabel(key) }));
    const unknown = result.unrecognized_folders || [];
    const tone = status.runnable ? 'ok' : 'warn';
    const title = status.runnable ? h.t('Folder check ready', '文件夹检查通过') : h.t('Folder check needs attention', '文件夹检查需要处理');
    return `<div class="note ${tone} mt-12" role="status" aria-live="polite" aria-atomic="true">
      <div class="ico">${h.icon(status.runnable ? 'check' : 'alert', 14)}</div>
      <div class="body">
        <div class="t">${title}</div>
        <div class="d">${h.t('Detected database folders', '已识别数据库文件夹')}: ${h.fmtInt(detected.length)} · ${h.t('selected recognized', '已选且识别')}: ${h.fmtInt(status.detectedSelectedKeys.length)}/${h.fmtInt(status.selectedKeys.length)} · ${h.t('need at least 2', '至少需要 2 个')}.</div>
        <div class="row gap-8 mt-8" style="flex-wrap:wrap;">
          ${detected.length ? detected.map(row => `<span class="chip ${selectedDetected.has(row.key) ? 'solid' : ''}">${h.esc(row.label || row.key)} · ${h.esc(row.folder_name || row.key)}${selectedDetected.has(row.key) ? '' : ` · ${h.t('not selected', '未选择')}`}</span>`).join('') : `<span class="pill warn">${h.t('No supported database folders detected', '未识别到支持的数据库文件夹')}</span>`}
        </div>
        ${missing.length ? `<div class="d mt-8">${h.t('Missing selected database folders', '已选但缺失的数据库文件夹')}: ${missing.map(row => h.esc(row.label || row.key)).join(', ')}</div>` : ''}
        ${unknown.length ? `<div class="d mt-8">${h.t('Unrecognized folders', '未识别文件夹')}: ${unknown.map(h.esc).join(', ')}${result.unrecognized_count > unknown.length ? ` +${h.fmtInt(result.unrecognized_count - unknown.length)}` : ''}</div>` : ''}
        <div class="d mt-8">${h.t('Accepted aliases', '可识别别名')}: ${h.esc(aliasSummary(result))}</div>
        ${detectedSelectionDiffers ? `<div class="mt-10"><button class="btn sm" type="button" data-crossdb-select-detected>${h.icon('check', 12)} ${h.t('Use detected databases', '仅选择已识别数据库')}</button></div>` : ''}
      </div>
    </div>`;
  }

  function renderReal(config) {
    const h = helpersOf(config);
    const count = selectedKeys().length;
    const root = state.rawRootDraft;
    const canLaunch = Boolean(root && count >= 2 && !state.rawRootScanning);
    const currentProfile = sampleProfile(config);
    const profiles = sampleProfiles(config);
    const catalogTotals = config && config.catalogTotals || {};
    const catalog = window.EU_CATALOG || {};
    const catalogModuleCount = Number(catalogTotals.modules)
      || (Array.isArray(catalog.groups) ? catalog.groups.length : 0);
    const catalogFeatureCount = Number(catalogTotals.features)
      || Number(catalog.totalConcepts)
      || 0;
    const fullScope = state.featureScope === 'all';
    const rawSelection = selectionStatus(root);
    const detectedCount = rawSelection.current && rawSelection.current.ok
      ? rawSelection.detectedKeys.size
      : 0;
    const sourceChoice = window.EU_CROSSDB_SOURCE_CHOICE;
    const registryHtml = config && typeof config.registryHtml === 'function' ? config.registryHtml() : '';
    const sourceChoiceHtml = sourceChoice && typeof sourceChoice.render === 'function'
      ? sourceChoice.render({ registryHtml })
      : '';
    const error = getError(config);
    return `
      <div class="card pad">
        <div class="eyebrow">${h.t('Step 1 · Source method', '第 1 步 · 来源方式')}</div>
        <div class="panel-title">${h.t('How should EasyICU build this comparison?', '选择跨库对比的数据来源')}</div>
        <div class="panel-sub mt-4">${h.t('Choose one path. Controls for the other path stay hidden.', '一次只选择一种路径，未选择路径的控件不会展开。')}</div>
        <div class="crossdb-method-grid mt-14" role="group" aria-label="${h.esc(h.t('Cross-database source method', '跨库数据来源方式'))}">
          <button class="crossdb-method-card ${state.sourceMethod === 'registered' ? 'selected' : ''}" type="button" data-crossdb-source-method="registered" aria-pressed="${state.sourceMethod === 'registered' ? 'true' : 'false'}">
            <span class="crossdb-method-icon">${h.icon('table', 16)}</span>
            <span><b>${h.t('Prepared EasyICU exports', '已准备的 EasyICU 导出')}</b><small>${h.t('Compare two or more bounded aggregate exports.', '比较两个或更多有界聚合导出。')}</small></span>
            <span class="pill ${state.sourceMethod === 'registered' ? 'ok' : 'dashed'}">${state.sourceMethod === 'registered' ? h.t('Selected', '已选择') : h.t('Choose', '选择')}</span>
          </button>
          <button class="crossdb-method-card ${state.sourceMethod === 'raw' ? 'selected' : ''}" type="button" data-crossdb-source-method="raw" aria-pressed="${state.sourceMethod === 'raw' ? 'true' : 'false'}">
            <span class="crossdb-method-icon">${h.icon('folder', 16)}</span>
            <span><b>${h.t('Raw ICU database folder', '原始 ICU 数据库目录')}</b><small>${h.t('Check a local parent folder, then run bounded sampling.', '检查本地父目录后执行有界抽样。')}</small></span>
            <span class="pill ${state.sourceMethod === 'raw' ? 'ok' : 'dashed'}">${state.sourceMethod === 'raw' ? h.t('Selected', '已选择') : h.t('Choose', '选择')}</span>
          </button>
        </div>
      </div>
      ${error ? `<div class="note warn" role="alert" aria-live="assertive"><div class="ico">${h.icon('alert', 14)}</div><div class="body"><div class="t">${h.t('Comparison needs attention', '对比需要处理')}</div><div class="d">${h.esc(userError(error, config))}</div></div></div>` : ''}
      ${state.sourceMethod === 'registered' ? `
      <section class="crossdb-path-panel" data-crossdb-source-path="registered">
        <div class="crossdb-step-head">
          <span class="crossdb-step-number">2</span>
          <div><b>${h.t('Select at least two exports', '选择至少两个导出')}</b><small>${h.t('Only aggregate summaries are compared; raw tables are not scanned.', '仅比较聚合摘要，不扫描原始数据库表。')}</small></div>
        </div>
        ${sourceChoiceHtml || `<div class="note info"><div class="body"><div class="d">${h.t('No registered export source owner is available.', '当前没有可用的已注册导出来源。')}</div></div></div>`}
      </section>` : `
      <section class="crossdb-path-panel crossdb-raw-primary" data-crossdb-source-path="raw">
        <div class="crossdb-raw-head">
          <span class="crossdb-method-icon">${h.icon('benchmark', 16)}</span>
          <div>
            <div class="panel-title">${fullScope ? h.t('Complete raw-database comparison', '完整原始库对比') : h.t('Quick raw-database comparison', '快速原始库对比')}</div>
            <div class="panel-sub mt-4">${h.t('Choose one parent folder. EasyICU detects its databases, checks the folder, and starts the comparison in one action.', '只需选择一个父目录；EasyICU 会自动识别数据库、检查目录并直接开始对比。')}</div>
          </div>
          <span class="pill ok"><span class="dot"></span>${h.term('local only')}</span>
        </div>
        <div class="path-field editable mt-14">
          <span class="pf-ico">${h.icon('folder', 14)}</span>
          <input class="pf-input" data-crossdb-root type="text" spellcheck="false" autocomplete="off" value="${h.esc(root)}" placeholder="${h.esc(h.t('Choose the parent folder containing your ICU databases', '选择包含 ICU 数据库的父目录'))}" aria-label="${h.esc(h.t('ICU data root', 'ICU 数据根目录'))}" />
          <button class="btn sm" type="button" data-crossdb-root-browse>${h.icon('folder', 12)} ${h.t('Browse...', '浏览...')}</button>
        </div>
        <div class="crossdb-raw-summary mt-12">
          <span class="pill ${detectedCount >= 2 ? 'ok' : ''}">${state.rawRootScanning ? `<span class="spin accent"></span> ${h.t('Detecting databases…', '正在识别数据库…')}` : detectedCount >= 2 ? `${h.icon('check', 12)} ${h.fmtInt(detectedCount)} ${h.t('databases detected', '个数据库已识别')}` : `${h.icon('search', 12)} ${h.t('Databases are detected automatically at run time', '运行时自动识别数据库')}`}</span>
          <span class="pill ${fullScope ? 'ok' : ''}">${fullScope ? `${h.fmtInt(catalogModuleCount)} ${h.t('modules', '个模块')} · ${h.fmtInt(catalogFeatureCount)} ${h.t('catalog features', '个目录特征')}` : `4 ${h.t('modules', '个模块')} · 12 ${h.t('core features', '个核心特征')}`}</span>
          <span class="pill">${h.esc(sampleSummary(currentProfile, config))}</span>
        </div>
        <div class="gate-strip crossdb-run-strip mt-14">
          <span>${h.t('Only bounded aggregate distributions are produced; no patient rows leave this computer.', '只生成有界聚合分布，不会把患者行传出本机。')}</span>
          <div class="grow"></div>
          <button class="btn primary" data-crossdb-run-raw ${canLaunch ? '' : 'aria-disabled="true"'}>${h.icon('play', 13)} ${state.rawRootScanning ? h.t('Checking folder…', '正在检查目录…') : fullScope ? h.t('Start complete comparison', '开始完整对比') : h.t('Start quick comparison', '开始快速对比')}</button>
        </div>
        <details class="crossdb-advanced mt-14">
          <summary>
            <span>${h.icon('sliders', 13)} ${h.t('Advanced settings (optional)', '高级设置（可选）')}</span>
            <small>${h.t('Feature scope, sampling budget, and database selection', '特征范围、抽样预算和数据库选择')}</small>
          </summary>
          <div class="crossdb-advanced-body">
            <div class="row between gap-12" style="align-items:center;">
              <div><b>${h.t('Folder detection', '目录识别')}</b><small>${h.t('Review aliases or re-check after changing files.', '查看识别结果，或在目录文件变化后重新检查。')}</small></div>
              <button class="btn sm" type="button" data-crossdb-root-scan ${state.rawRootScanning ? 'aria-disabled="true" aria-busy="true"' : ''}>${state.rawRootScanning ? '<span class="spin accent"></span>' : h.icon('search', 12)} ${state.rawRootScanning ? h.t('Checking…', '检查中…') : h.t('Check again', '重新检查')}</button>
            </div>
            ${renderScanPanel(config)}
            <div class="crossdb-advanced-section">
              <b>${h.t('Feature scope', '特征范围')}</b>
              <div class="db-grid mt-10" style="grid-template-columns:repeat(2,minmax(0,1fr));">
                <button class="db-card ${fullScope ? '' : 'sel'}" type="button" data-crossdb-feature-scope="core" aria-pressed="${fullScope ? 'false' : 'true'}">
                  <div><b>${h.t('Quick core', '快速核心')}</b><small>4 / ${h.fmtInt(catalogModuleCount)} ${h.t('modules', '模块')} · 12 / ${h.fmtInt(catalogFeatureCount)} ${h.t('features', '特征')}</small></div>
                  <span class="db-mk pill ${fullScope ? 'dashed' : 'ok'}">${fullScope ? h.t('choose', '选择') : h.t('selected', '已选择')}</span>
                </button>
                <button class="db-card ${fullScope ? 'sel' : ''}" type="button" data-crossdb-feature-scope="all" aria-pressed="${fullScope ? 'true' : 'false'}">
                  <div><b>${h.t('Complete catalog', '完整目录')}</b><small>${h.fmtInt(catalogModuleCount)} ${h.t('modules', '模块')} · ${h.fmtInt(catalogFeatureCount)} ${h.t('features', '特征')}</small></div>
                  <span class="db-mk pill ${fullScope ? 'ok' : 'dashed'}">${fullScope ? h.t('selected', '已选择') : h.t('choose', '选择')}</span>
                </button>
              </div>
            </div>
            <div class="crossdb-advanced-section">
              <b>${h.t('Sampling budget', '抽样预算')}</b>
              <div class="db-grid mt-10" style="grid-template-columns:repeat(3,minmax(0,1fr));">
                ${profiles.map(profile => `<button class="db-card ${profile.id === currentProfile.id ? 'sel' : ''}" type="button" data-crossdb-sample-mode="${h.esc(profile.id)}" aria-pressed="${profile.id === currentProfile.id ? 'true' : 'false'}"><div><b>${h.esc(profile.label)}</b><small>≤${h.fmtInt(profile.maxPatients)} ${h.t('entities/db', '实体/库')} · ≤${h.fmtInt(profile.sampleSize)} ${h.t('values/feature', '值/特征')}</small></div><span class="db-mk pill ${profile.id === currentProfile.id ? 'ok' : 'dashed'}">${profile.id === currentProfile.id ? h.t('selected', '已选择') : h.t('choose', '选择')}</span></button>`).join('')}
              </div>
            </div>
            <div class="crossdb-advanced-section">
              <b>${h.term('Databases')} · <span id="dbcount">${count}</span> ${h.term('selected')}</b>
              <div class="db-grid mt-10" id="dbgrid">
                ${DATABASES.map((row, index) => {
                  const status = databaseStatus(row.key, row.selected, config);
                  return `<button class="db-card ${row.selected ? 'sel' : ''}" type="button" data-db="${index}" aria-pressed="${row.selected ? 'true' : 'false'}"><div><b>${h.esc(row.label)}</b><small>${h.esc(status.sub)}</small></div><span class="db-mk pill ${status.cls}">${h.esc(status.label)}</span></button>`;
                }).join('')}
              </div>
            </div>
          </div>
        </details>
      </section>`}`;
  }

  function renderSyntheticDemo(config) {
    const h = helpersOf(config);
    const count = selectedKeys().length;
    return `<div class="note info">
      <div class="ico">${h.icon('benchmark', 16)}</div>
      <div class="body"><span class="t">${h.term('Select databases to compare')}</span> <span class="d" style="display:inline;">— ${h.t('Pick two or more standardized ICU sources, then start the comparison. Each uses an independent seeded feature frame in Demo Mode.', '选择两个或更多标准化 ICU 来源后运行对比。演示模式下，每个数据库使用独立的种子特征帧。')}</span></div>
    </div>
    <div class="sec-stack"><div class="lbl">${h.term('Available databases')} · <span id="dbcount">${count}</span> ${h.term('selected')}</div></div>
    <div class="db-grid" id="dbgrid">
      ${DATABASES.map((row, index) => `<button class="db-card ${row.selected ? 'sel' : ''}" type="button" data-db="${index}" aria-pressed="${row.selected ? 'true' : 'false'}">
        <div class="row gap-8" style="min-width:0;">
          <span class="${row.selected ? '' : 'ink-4'}" style="flex:none;color:${row.selected ? 'var(--accent-ink)' : 'var(--ink-4)'};">${h.icon('db', 15)}</span>
          <div style="min-width:0;">
            <div style="font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${row.label}</div>
            <div class="mono" style="font-size:10.5px;color:var(--ink-4);">${h.term('all supported catalog concepts')}</div>
          </div>
        </div>
        <span class="db-mk pill ${row.selected ? 'ok' : 'dashed'}" style="flex:none;height:20px;">${row.selected ? `<span class="dot"></span>${h.term('selected')}` : h.term('add')}</span>
      </button>`).join('')}
    </div>
    <div class="gate-strip mt-20">
      <span class="pill"><span style="color:var(--ink-3);">${h.icon('benchmark', 12)}</span> <span id="runhint">${count} / 6 · ${h.t('need ≥ 2', '至少需要 2 个')}</span></span>
      <div class="grow"></div>
      <button class="btn primary" data-crossdb-run-demo ${count < 2 ? 'aria-disabled="true"' : ''}>${h.icon('play', 13)} ${h.t('Start comparison', '开始对比')}</button>
    </div>`;
  }

  function renderDemo(config) {
    const h = helpersOf(config);
    const sourceChoice = window.EU_CROSSDB_SOURCE_CHOICE;
    const syntheticHtml = renderSyntheticDemo(config);
    const error = getError(config);
    const body = sourceChoice && typeof sourceChoice.renderDemo === 'function'
      ? sourceChoice.renderDemo({ syntheticHtml })
      : `<div class="card pad">${syntheticHtml}</div>`;
    return `${error ? `<div class="note warn" role="alert" aria-live="assertive"><div class="ico">${h.icon('alert', 14)}</div><div class="body"><div class="t">${h.t('Comparison needs attention', '对比需要处理')}</div><div class="d">${h.esc(userError(error, config))}</div></div></div>` : ''}${body}`;
  }

  function renderLoading(config) {
    const h = helpersOf(config);
    const sourceChoice = window.EU_CROSSDB_SOURCE_CHOICE;
    if (state.registeredLoading && sourceChoice && typeof sourceChoice.renderLoading === 'function') {
      return sourceChoice.renderLoading();
    }
    if (window.EU_DATA !== 'real') {
      return `<div class="card pad">
        <div class="load-strip"><span class="spin accent"></span><div class="grow"><div style="font-weight:600;font-size:12.75px;">${h.progressMessage('Loading seeded frames for selected databases…')}</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${h.term('local-only · nothing uploaded')}</div></div><button class="btn sm" data-viz-reset>${h.icon('stop', 13)} ${h.t('Cancel', '取消')}</button></div>
        <div class="indet mt-12"></div>
        <div style="font-size:12px;color:var(--ink-3);min-height:18px;margin-top:8px;">${h.progressMessage('Building seeded density frames…')}</div>
      </div>`;
    }
    const progress = window.EU_CROSSDB_PROGRESS;
    return progress && typeof progress.render === 'function' ? progress.render({
      esc: h.esc,
      errorMessage: getError(config),
      fmtInt: h.fmtInt,
      icon: h.icon,
      progressMessage: h.progressMessage,
      sampleProfile: sampleProfile(config),
      statusLabel: h.statusLabel,
      t: h.t,
    }) : '';
  }

  function guidedNote() {
    if (window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.take) window.EU_GUIDED_HANDOFF.take('crossdb');
    return window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.noteHtml
      ? window.EU_GUIDED_HANDOFF.noteHtml('crossdb')
      : '';
  }

  function renderBody(config) {
    const h = helpersOf(config);
    const note = guidedNote();
    const header = config && typeof config.header === 'function' ? config.header() : '';
    const workspace = window.EU_CROSSDB_WORKSPACE;
    if (state.view === 'loading') return note + header + renderLoading(config);
    if (workspace && config && typeof config.renderLoaded === 'function') {
      return note + header + config.renderLoaded(workspace);
    }
    const displayMode = window.getDataMode
      ? window.getDataMode()
      : (window.EU_DATA === 'real' ? 'real' : 'demo');
    if (displayMode === 'real') return note + header + renderReal(config);
    if (state.view === 'loaded') {
      return note + header + `<div class="note warn">
        <div class="ico">${h.icon('alert', 14)}</div>
        <div class="body"><span class="t">${h.t('Demo comparison was not loaded', '演示对比未加载')}</span> <span class="d" style="display:inline;">— ${h.t('Run the comparison again so the backend can build the seeded distribution payload.', '请重新运行对比，让后端生成种子分布载荷。')}</span></div>
      </div>`;
    }
    return note + header + renderDemo(config);
  }

  function actionHtml(config) {
    const h = helpersOf(config);
    const loaded = state.view === 'loaded' || (window.EU_DATA === 'real' && window.EU_CROSSDB_WORKSPACE);
    if (!loaded) return '';
    const rawLoaded = window.EU_CROSSDB_WORKSPACE && window.EU_CROSSDB_WORKSPACE.source_type === 'raw_database_root';
    return `<button class="btn" data-viz-reset>${h.icon('sliders', 13)} ${h.term('Change selection')}</button><button class="btn" data-crossdb-export>${h.icon('download', 13)} ${h.term('Export JSON')}</button>${rawLoaded ? '' : `<button class="btn primary" data-crossdb-rerun>${h.icon('refresh', 13)} ${h.term('Re-run')}</button>`}`;
  }

  function startRawRun(rootValue, config) {
    if (!scanReady(rootValue) || state.view === 'loading') return false;
    const operationId = beginOperation();
    state.registeredLoading = false;
    state.view = 'loading';
    const runSnapshot = { ...snapshot(config), operationId };
    repaint(config);
    if (config && typeof config.runRaw === 'function') {
      config.runRaw(ok => {
        if (!operationCurrent(operationId)) return;
        state.view = ok ? 'loaded' : 'idle';
        repaint(config);
      }, { operationId, rawRoot: rootValue, setup: runSnapshot });
      return true;
    }
    state.view = 'idle';
    setError(config, helpersOf(config).t(
      'Raw Cross-DB comparison is unavailable in this browser session.',
      '当前浏览器会话无法启动原始跨库对比。'
    ));
    repaint(config);
    return false;
  }

  function bind(root, config) {
    if (!root) return;
    if (config && typeof config.bindRegistry === 'function') config.bindRegistry(root, 'crossdb');
    const sourceChoice = window.EU_CROSSDB_SOURCE_CHOICE;
    if (sourceChoice && typeof sourceChoice.wire === 'function') sourceChoice.wire(root);

    root.querySelectorAll('[data-crossdb-source-method]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      if (state.view === 'loading') return;
      const next = setSourceMethod(button.dataset.crossdbSourceMethod);
      setError(config, null);
      repaint(config, `[data-crossdb-source-method="${next}"]`);
    }));

    root.querySelectorAll('.radio[data-datamode]').forEach(button => button.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      if (window.setDataMode) window.setDataMode(button.dataset.datamode);
    }));

    root.querySelectorAll('[data-crossdb-sample-mode]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const nextMode = button.dataset.crossdbSampleMode || 'quick';
      notifySourceChanged(state.rawRootDraft, nextMode);
      setSampleMode(nextMode);
      repaint(config, `[data-crossdb-sample-mode="${nextMode}"]`);
    }));

    root.querySelectorAll('[data-crossdb-feature-scope]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const nextScope = button.dataset.crossdbFeatureScope === 'all' ? 'all' : 'core';
      notifySourceChanged(state.rawRootDraft, state.sampleMode, nextScope);
      setFeatureScope(nextScope);
      repaint(config, `[data-crossdb-feature-scope="${nextScope}"]`);
    }));

    const grid = root.querySelector('#dbgrid');
    if (grid) grid.addEventListener('click', event => {
      const card = event.target.closest('[data-db]');
      if (!card) return;
      const row = toggleDatabase(card.dataset.db);
      if (!row) return;
      if (window.EU_DATA === 'real') {
        notifySourceChanged(state.rawRootDraft, state.sampleMode);
        setError(config, null);
        repaint(config, `[data-db="${card.dataset.db}"]`);
        return;
      }
      card.classList.toggle('sel', row.selected);
      card.setAttribute('aria-pressed', row.selected ? 'true' : 'false');
      const marker = card.querySelector('.db-mk');
      if (marker) {
        marker.className = `db-mk pill ${row.selected ? 'ok' : 'dashed'}`;
        marker.innerHTML = row.selected ? `<span class="dot"></span>${helpersOf(config).term('selected')}` : helpersOf(config).term('add');
      }
      const count = selectedKeys().length;
      const counter = root.querySelector('#dbcount');
      if (counter) counter.textContent = count;
      const hint = root.querySelector('#runhint');
      if (hint) hint.textContent = `${count} / 6 · ${helpersOf(config).t('need ≥ 2', '至少需要 2 个')}`;
      root.querySelectorAll('[data-crossdb-run-demo]').forEach(button => {
        if (count < 2) button.setAttribute('aria-disabled', 'true');
        else button.removeAttribute('aria-disabled');
      });
    });

    root.querySelectorAll('[data-crossdb-select-detected]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const detected = Array.from(selectionStatus(state.rawRootDraft).detectedKeys);
      if (detected.length < 2) return;
      setSelectedKeys(detected);
      notifySourceChanged(state.rawRootDraft, state.sampleMode);
      setError(config, null);
      repaint(config, '[data-crossdb-run-raw]');
    }));

    root.querySelectorAll('[data-crossdb-run-raw]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      if (button.getAttribute('aria-disabled') === 'true' || state.view === 'loading') return;
      const input = root.querySelector('[data-crossdb-root]');
      const rootValue = pathValue(input && input.value || state.rawRootDraft);
      state.rawRootDraft = rootValue;
      if (!scanReady(rootValue)) {
        button.setAttribute('aria-disabled', 'true');
        scan(rootValue, config, '[data-crossdb-run-raw]').then(ready => {
          if (ready) {
            setError(config, null);
            startRawRun(rootValue, config);
            return;
          }
          if (!getError(config)) {
            setError(config, helpersOf(config).t(
              'The comparison did not start because fewer than two selected database folders were recognized. Open Advanced settings to review the detected databases.',
              '未能开始对比：识别到的已选数据库少于两个。请打开“高级设置”检查数据库识别结果。'
            ));
            repaint(config, '[data-crossdb-run-raw]');
          }
        });
        return;
      }
      button.setAttribute('aria-disabled', 'true');
      startRawRun(rootValue, config);
    }));

    root.querySelectorAll('[data-crossdb-run-demo]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      if (button.getAttribute('aria-disabled') === 'true' || state.view === 'loading') return;
      button.setAttribute('aria-disabled', 'true');
      const operationId = beginOperation();
      state.view = 'loading';
      const runSnapshot = { ...snapshot(config), operationId };
      repaint(config);
      if (config && typeof config.runDemo === 'function') {
        config.runDemo(ok => {
          if (!operationCurrent(operationId)) return;
          state.view = ok ? 'loaded' : 'idle';
          repaint(config);
        }, runSnapshot);
      }
    }));

    root.querySelectorAll('[data-crossdb-rerun]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      if (state.view === 'loading') return;
      button.setAttribute('aria-disabled', 'true');
      if (window.EU_DATA === 'real') {
        const sourceHost = window.EU_CROSSDB_SOURCE_HOST;
        const context = window.EU_DATA_MODE_CONTEXT;
        if (context && context.kind === 'official_demo_pair' && sourceHost && typeof sourceHost.runOfficial === 'function') {
          sourceHost.runOfficial();
        } else if (sourceHost && typeof sourceHost.runRegistered === 'function') {
          sourceHost.runRegistered();
        }
        return;
      }
      const operationId = beginOperation();
      state.view = 'loading';
      const runSnapshot = { ...snapshot(config), operationId };
      repaint(config);
      if (config && typeof config.runDemo === 'function') {
        config.runDemo(ok => {
          if (!operationCurrent(operationId)) return;
          state.view = ok ? 'loaded' : 'idle';
          repaint(config);
        }, runSnapshot);
      }
    }));

    root.querySelectorAll('[data-crossdb-root]').forEach(input => {
      input.addEventListener('input', () => {
        const next = pathValue(input.value);
        notifySourceChanged(next, state.sampleMode);
        changeRawRoot(next);
        root.querySelectorAll('[data-crossdb-run-raw]').forEach(button => {
          const enabled = Boolean(next)
            && selectedKeys().length >= 2
            && !state.rawRootScanning
            && state.view !== 'loading';
          if (enabled) button.removeAttribute('aria-disabled');
          else button.setAttribute('aria-disabled', 'true');
        });
      });
      input.addEventListener('change', () => {
        const next = pathValue(input.value);
        notifySourceChanged(next, state.sampleMode);
        changeRawRoot(next);
        repaint(config);
      });
    });

    root.querySelectorAll('[data-crossdb-root-browse]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const input = root.querySelector('[data-crossdb-root]');
      if (!config || typeof config.openFolderPicker !== 'function') {
        setError(config, helpersOf(config).t('Local folder picker API is not ready. Paste a raw ICU data root path instead.', '本地文件夹选择 API 尚未就绪。请改为粘贴原始 ICU 数据根目录路径。'));
        repaint(config);
        return;
      }
      config.openFolderPicker(
        pathValue(input && input.value || state.rawRootDraft),
        picked => {
          if (!picked || !input) {
            setError(config, helpersOf(config).t('Local folder picker API is not ready. Paste a raw ICU data root path instead.', '本地文件夹选择 API 尚未就绪。请改为粘贴原始 ICU 数据根目录路径。'));
            repaint(config);
            return;
          }
          notifySourceChanged(picked, state.sampleMode);
          changeRawRoot(picked);
          setError(config, null);
          input.value = picked;
          input.focus();
          scan(picked, config, '[data-crossdb-root]');
        },
        helpersOf(config).t('Choose local ICU data root', '选择本地 ICU 数据根目录')
      );
    }));

    root.querySelectorAll('[data-crossdb-root-scan]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      if (button.getAttribute('aria-disabled') === 'true') return;
      const input = root.querySelector('[data-crossdb-root]');
      scan(pathValue(input && input.value || state.rawRootDraft), config, '[data-crossdb-root-scan]');
    }));

    root.querySelectorAll('[data-crossdb-cancel]').forEach(button => button.addEventListener('click', () => cancel(config)));
    root.querySelectorAll('[data-viz-reset]').forEach(button => button.addEventListener('click', () => {
      if (rawJobActive()) {
        cancel(config);
        return;
      }
      const preserveScan = window.EU_CROSSDB_WORKSPACE && window.EU_CROSSDB_WORKSPACE.source_type === 'raw_database_root';
      invalidateOperations();
      disconnectJob({ forget: true });
      const progress = window.EU_CROSSDB_PROGRESS;
      if (progress && typeof progress.clear === 'function') progress.clear();
      state.registeredLoading = false;
      if (!preserveScan) invalidateScan();
      state.view = 'idle';
      if (config && typeof config.resetResult === 'function') config.resetResult();
      repaint(config);
    }));

    // Restore last: onProbe may repaint synchronously. Running it earlier lets
    // the outer bind continue against the freshly rendered nodes and double-bind them.
    const continuity = window.EU_CROSSDB_JOB_CONTINUITY;
    if (continuity && typeof continuity.restoreIfNeeded === 'function') continuity.restoreIfNeeded();
  }

  function presetLoaded() {
    invalidateOperations();
    state.view = 'loaded';
  }

  function resetAsyncState(config) {
    const progress = window.EU_CROSSDB_PROGRESS;
    if (rawJobActive() && progress && typeof progress.requestCancel === 'function') {
      progress.requestCancel({
        api: config && config.api || window.EU_API,
        onError(error) {
          if (window.console && typeof window.console.warn === 'function') {
            window.console.warn('[EasyICU] Cross-DB job cancellation during data-mode switch failed', error);
          }
        },
      });
    }
    invalidateOperations();
    disconnectJob({ forget: true });
    if (progress && typeof progress.clear === 'function') progress.clear();
    state.view = 'idle';
    state.registeredLoading = false;
  }

  function resetForDataMode(config) {
    resetAsyncState(config);
  }

  function onRegistryChanged() {
    if (rawJobActive()) return false;
    invalidateOperations();
    state.registeredLoading = false;
    state.view = 'idle';
    return true;
  }

  window.EU_CROSSDB_SETUP = {
    acceptResume,
    actionHtml,
    beginOperation,
    bind,
    changeRawRoot,
    databaseRows,
    disconnectJob,
    featureScope,
    identityKeys,
    invalidateOperations,
    invalidateScan,
    matchesSource,
    operationCurrent,
    onRegistryChanged,
    pathValue,
    presetLoaded,
    rawRoot,
    registeredLoading,
    renderBody,
    renderDemo,
    renderLoading,
    renderReal,
    resetForDataMode,
    sampleMode,
    sampleProfile,
    sampleProfiles,
    sampleSummary,
    scan,
    scanCurrent,
    scanReady,
    selectedKeys,
    selectionStatus,
    setRawRoot,
    setRegisteredLoading,
    setFeatureScope,
    setSampleMode,
    setSelectedKeys,
    setSourceMethod,
    setView,
    snapshot,
    sourceMethod,
    sourceIdentity,
    toggleDatabase,
    view,
  };
})();
