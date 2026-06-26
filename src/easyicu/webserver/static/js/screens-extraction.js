/* Screens: Entry + Data Extraction (redesigned, bilingual).
   Extraction is simplified from a 4-step wizard into:
     • an Express "recommended extraction" one-click path (the 80% case)
     • a single-page Custom panel with smart defaults + progressive disclosure
   Bilingual via window.t(en, zh). */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  /* ---------------- ENTRY / home ---------------- */
  function homeDataMode() { return window.EU_DATA || 'demo'; }
  function setHomeData(m) { window.EU_DATA = m; try { localStorage.setItem('easyicu_home_data', m); } catch (e) {} }

  function launchCopilot(text, branchHint) {
    try {
      window.__cpBridge = { ts: Date.now(), route: 'entry', lastUser: text || null, dataMode: homeDataMode(), branchHint: branchHint || null };
    } catch (e) {}
    location.hash = '#guided';
  }

  function homeHeader() {
    return `
      <header class="entry-top">
        <div class="row gap-12">
          <div class="mark">${icon('flask', 18)}</div>
          <div>
            <div class="name" style="font-size:16px;font-weight:600;letter-spacing:-0.01em;">EasyICU</div>
            <div class="tag" style="font-size:11px;color:var(--ink-4);">${t('ICU data research workspace', 'ICU 数据研究工作台')}</div>
          </div>
        </div>
        <div class="row gap-10">
          <div class="lang-seg" role="group" aria-label="Language">
            <button class="${window.EU_LANG !== 'zh' ? 'on' : ''}" data-lang="en">EN</button>
            <button class="${window.EU_LANG === 'zh' ? 'on' : ''}" data-lang="zh">中</button>
          </div>
          <span class="mono" style="font-size:11px;color:var(--ink-4);">v1.0 · py3.10+</span>
        </div>
      </header>`;
  }

  S.entry = {
    section: 'entry',
    full: true,
    render() {
      const dm = homeDataMode();
      const inner = `
        <div class="home-inner" style="max-width:1180px;">
          <h1 class="home-h1">${t('Welcome to EasyICU', '欢迎使用 EasyICU')}</h1>
          <p class="home-sub">${t('Your local-first ICU research workspace. Pick how you want to work — start a Guided study, or drive the panels yourself. Both go from question to a review-ready draft; nothing is uploaded.', '本地优先的 ICU 研究工作台。选一种你习惯的方式 —— 开始研究引导,或自己操作面板。两者都能从问题走到待核验草稿;不上传任何数据。')}</p>
          <div class="entry-firsttime" id="firstTimeNudge" hidden>
            <span class="ft-ico">${icon('play', 13)}</span>
            <span>${t('New here?', '第一次使用?')} <b>${t('Take the 2-minute demo', '先跑 2 分钟演示')}</b> ${t('— no setup, no data, fully explorable.', '—— 无需配置、无需数据,全程可探索。')}</span>
            <button class="ft-go" data-firsttime>${t('Start the tour', '开始引导')} ${icon('arrow', 13)}</button>
            <button class="ft-x" data-firsttime-dismiss aria-label="${t('Dismiss', '忽略')}">${icon('close', 13)}</button>
          </div>
          <div class="entry-journey">
            <div class="ej-cap">${t('The research journey · 4 steps', '研究旅程 · 四步')}</div>
            <div class="ej-track">
              ${[
                ['1', t('Frame', '框定'), t('the question', '研究问题')],
                ['2', t('Extract', '抽取'), t('the data', '数据')],
                ['3', t('Review', '审阅'), t('& explore', '与探索')],
                ['4', t('Analyze', '分析'), t('& draft', '与撰稿')],
              ].map((n, i) => `
                ${i > 0 ? '<div class="ej-conn"></div>' : ''}
                <div class="ej-node"><div class="ej-num">${n[0]}</div><div><div class="ej-lab">${n[1]}</div><div class="ej-sub">${n[2]}</div></div></div>`).join('')}
            </div>
            <div class="ej-foot">${t('Two ways through it:', '两种走法:')} <b>${t('① Guided study', '① 研究引导')}</b> ${t('or', '或')} <b>${t('② drive the panels yourself', '② 自己操作面板')}</b></div>
          </div>
          <div class="home-split">
            <div class="home-col col-copilot">
              <div class="col-head"><div class="col-mk">${icon('spark', 17)}</div><div><div class="col-t">${t('Guided study', '研究引导')}</div><div class="col-sub">${t('talk it through · easiest way to start', '对话引导 · 最推荐的入口')}</div></div><span class="col-badge">${t('Recommended', '推荐')}</span></div>
              <div class="col-body">
                <p class="col-lead">${t('Describe your study in a sentence — I frame it, pull the cohort, run the analysis, and prepare a review-ready draft.', '用一句话描述你的研究 —— 我来框定问题、筛取队列、运行分析,并准备一份待证据核验的草稿。')}</p>
                <div class="col-prompt">
                  <textarea class="hp-input" id="homeInput" rows="3" placeholder="${t('e.g. Among Sepsis-3 patients, does early lactate predict in-hospital mortality, and does adding it to SOFA improve the model?', '例如:在脓毒症(Sepsis-3)患者中,早期乳酸能否预测院内死亡?把它加入 SOFA 是否提升模型?')}" autocomplete="off" aria-label="Describe your study"></textarea>
                  <div class="hp-bar">
                    <span class="hp-hint">${icon('shield', 12)} ${t('local-only · nothing uploaded', '仅本地 · 不上传')}</span>
                    <button class="hp-send" id="homeSend" aria-label="Start study">${icon('arrow', 17)}</button>
                  </div>
                </div>
                <div class="col-chips">
                  <span class="cc-lead">${t('or start from a question type', '或选一种研究类型')}</span>
                  <button class="home-chip" data-hbranch="predict">${t('Model an outcome', '结局建模')}</button>
                  <button class="home-chip" data-hbranch="crossdb">${t('Compare databases', '跨库比较')}</button>
                  <button class="home-chip" data-hbranch="quality">${t('Audit data quality', '数据质量审计')}</button>
                </div>
              </div>
            </div>
            <div class="home-col col-classic">
              <div class="col-head"><div class="col-mk">${icon('grid', 17)}</div><div><div class="col-t">${t('Classic Workspace', '经典工作台')}</div><div class="col-sub">${t('drive it yourself · for when you know the steps', '自己操作 · 熟悉流程后使用')}</div></div></div>
              <div class="col-body">
                <p class="col-lead">${t('Open any section directly and work hands-on, at your own pace.', '直接打开任意模块,按自己的节奏动手操作。')}</p>
                <div class="col-entries">
                  ${[
                    ['extraction', 'extract', t('Data Extraction', '数据抽取'), t('One click to analysis-ready tables', '一键得到可分析的数据表')],
                    ['patient', 'viz', t('Data Visualization', '数据可视化'), t('Patient review · cohort stats · cross-DB', '患者审阅 · 队列统计 · 跨库')],
                    ['agent', 'agent', t('Agent Projects', '研究项目'), t('Auditable runs → review-ready manuscript', '可审计运行 → 待核验草稿')],
                  ].map(([nav, ic, ti, d]) => `
                    <button class="col-entry" data-nav="${nav}">
                      <span class="ce-ico">${icon(ic, 15)}</span>
                      <span><span class="ce-t">${ti}</span><span class="ce-d">${d}</span></span>
                      <span class="ce-go">${icon('arrow', 14)}</span>
                    </button>`).join('')}
                </div>
                <div class="col-foot">
                  <div class="col-dataline">${icon('shield', 12)} ${dm === 'demo' ? t('Demo data · reproducible', '演示数据 · 可复现') : t('Real data · local-only', '真实数据 · 仅本地')}</div>
                </div>
              </div>
            </div>
          </div>
          <div id="resumeSlot" class="home-resume"></div>
        </div>`;
      return `
      <div class="entry-shell">
        ${homeHeader()}
        <div class="home-wrap">${inner}</div>
      </div>`;
    },
    afterRender(root) {
      const dataEl = root.querySelector('#homeData');
      if (dataEl) dataEl.addEventListener('click', (e) => {
        const b = e.target.closest('[data-hd]'); if (!b) return;
        setHomeData(b.dataset.hd);
        dataEl.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
        const dl = root.querySelector('.col-dataline');
        if (dl) dl.innerHTML = `${icon('shield', 12)} ${b.dataset.hd === 'demo' ? t('Demo data · reproducible', '演示数据 · 可复现') : t('Real data · local-only', '真实数据 · 仅本地')}`;
      });
      const input = root.querySelector('#homeInput');
      const send = root.querySelector('#homeSend');
      const ft = root.querySelector('[data-firsttime]');
      if (ft) ft.addEventListener('click', () => { try { localStorage.setItem('easyicu_onboarded', '1'); } catch (e) {} if (window.setDataMode) window.setDataMode('demo', { force: true }); location.hash = '#tutorial'; });
      // First-run demo nudge: prominent for genuine newcomers, gone once seen.
      const nudge = root.querySelector('#firstTimeNudge');
      if (nudge) {
        let onboarded = false, hasStudy = false;
        try { onboarded = !!localStorage.getItem('easyicu_onboarded'); } catch (e) {}
        try { hasStudy = !!localStorage.getItem('easyicu_study'); } catch (e) {}
        if (!onboarded && !hasStudy && !window.EU_HASWORK) nudge.hidden = false;
        const dx = nudge.querySelector('[data-firsttime-dismiss]');
        if (dx) dx.addEventListener('click', () => { try { localStorage.setItem('easyicu_onboarded', '1'); } catch (e) {} nudge.hidden = true; });
      }
      function submit() { const v = (input.value || '').trim(); launchCopilot(v || null); }
      if (send) send.addEventListener('click', submit);
      if (input) input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } });
      root.querySelectorAll('[data-hbranch]').forEach(c => c.addEventListener('click', () => {
        launchCopilot(null, c.dataset.hbranch);
      }));
      // resume banner
      const slot = root.querySelector('#resumeSlot');
      if (slot) {
        let s = null;
        try { s = JSON.parse(localStorage.getItem('easyicu_study') || 'null'); } catch (e) {}
        if (s) {
          const names = { predict: t('Sepsis mortality prediction', '脓毒症死亡率预测'), crossdb: t('Cross-database comparison', '跨数据库对比'), quality: t('Data-quality audit', '数据质量审计') };
          const when = (function (ts) { const d = Math.round((Date.now() - ts) / 60000); return d < 1 ? t('just now', '刚刚') : d < 60 ? d + t('m ago', ' 分钟前') : Math.round(d / 60) + t('h ago', ' 小时前'); })(s.ts || Date.now());
          slot.innerHTML = `
            <div class="card flat" style="border-color:var(--accent-border);background:color-mix(in srgb, var(--accent-soft) 40%, var(--surface));">
              <div class="row gap-12">
                <div class="aux-ico" style="background:var(--accent-soft);color:var(--accent-ink);">${icon('history', 16)}</div>
                <div><div style="font-weight:600;font-size:13px;">${t('Resume your last study', '继续上次的研究')}</div><div style="font-size:12px;color:var(--ink-3);margin-top:1px;">${names[s.branch] || t('Study', '研究')} · ${s.patientN || 10} ${t('stays', '次住院')} · ${(s.mods || []).length} ${t('modules', '模块')} · ${when}</div></div>
              </div>
              <div class="row gap-8">
                <button class="btn sm" data-resume-open>${t('Open workspace', '打开工作台')} ${icon('arrow', 14)}</button>
                <button class="btn sm ghost" data-resume-clear>${t('Dismiss', '忽略')}</button>
              </div>
            </div>`;
          slot.querySelector('[data-resume-open]').addEventListener('click', () => {
            try { if (window.__euExtractApply) window.__euExtractApply(s.mods); } catch (e) {}
            try { if (window.__euVizPreset) window.__euVizPreset(); } catch (e) {}
            location.hash = '#patient';
          });
          slot.querySelector('[data-resume-clear]').addEventListener('click', () => {
            try { localStorage.removeItem('easyicu_study'); } catch (e) {}
            slot.innerHTML = '';
          });
        }
      }
      setTimeout(() => { if (input) input.focus(); }, 300);
    },
  };

  /* ================= DATA EXTRACTION (simplified) ================= */
  function dataMode() { return window.EU_DATA || 'demo'; }  // global Demo/Real (topbar)
  const DEFAULT_OBSERVATION_WINDOW_HOURS = 24 * 30;
  const MAX_OBSERVATION_WINDOW_HOURS = 24 * 30;
  let exView = 'home';          // home | running | done
  let exMaxPatients = 500;      // cohort sample cap for real extraction (full-cohort = 3c follow-up)
  let exportES = null;          // EventSource for the extract job
  let exportJobId = null;       // current extract job id for cooperative cancel
  let exportProg = null;        // {current,total,module} latest extract progress
  let exportResult = null;      // terminal export summary {out_dir,files,total_rows,...}
  let exportErr = null;         // terminal export error
  let exportCancelRequested = false;
  let exportRunMode = 'custom';  // custom | recommended
  let exportRunModules = null;   // module keys used by the current/last run
  let exCustomOpen = true;
  let exAdvCohort = false, exAdvExport = false, exShowAllMods = true;
  let exFormat = 'parquet';     // parquet | csv | excel
  let exMerge = 'separate';
  let exExportDir = null;
  let exReal = 'connect';   // connect | scanning | scanresult | converting | convfail | ready
  let exPath = '';   // the local folder the user points at; never prefilled
  let exManualSourceOpen = false;
  let exExpandedMod = 'demographics';
  let exSelectedConcepts = {};
  let convTimer = null;
  let convDone = 0;         // completed conversion steps (mock fallback only)
  let convJobId = null;     // live convert job id (SSE-driven)
  let convES = null;        // EventSource for the convert job
  let convProg = null;      // {current,total,file,counts} latest progress
  let convResult = null;    // terminal summary {converted,failed,skipped,nothing_to_do}
  let convErr = null;       // terminal error message
  let exSource = null;      // 'prepared' | 'module' | 'raw' — what the user pointed at
  let exScanResult = null;  // live /api/data/scan payload for exPath (null until scanned)
  let exScanError = null;   // scan failure message, if any
  let exFilterOptions = null;   // real-source advanced filter metadata
  let exFilterPreview = null;   // current metadata-filter preview
  let exFilterLoading = false;
  let exFilterError = null;
  let exMinCoveragePct = 0;
  let exQualityStatus = 'all';
  let exCohortPreset = 'adult_first';
  let exAgeMin = 18;
  let exAgeMax = 100;
  let exMinLosHours = 0;
  let exWindowHours = DEFAULT_OBSERVATION_WINDOW_HOURS;
  let exExcludeReadmissions = true;
  let convFail = false;     // demo: conversion hit a recoverable error

  const COHORT_PRESETS = [
    ['all_icu', 'All ICU stays', '全部 ICU 住院', 'Broad denominator; no diagnosis filter is applied.', '宽队列;不预设诊断筛选。'],
    ['adult_first', 'Adult first ICU stay', '成年首次 ICU', 'Default ICU denominator for most extraction workflows.', '多数抽取流程的默认 ICU 分母。'],
    ['sepsis3', 'Sepsis-3 / suspected infection', 'Sepsis-3 / 疑似感染', 'Uses Sepsis concepts when available; ICD is not prefilled.', '可用时使用 Sepsis 概念;不会预填 ICD。'],
    ['aki', 'AKI / renal dysfunction', 'AKI / 肾功能异常', 'Renal cohort starting point for AKI studies.', 'AKI 研究的肾功能队列起点。'],
    ['ventilation', 'Mechanical ventilation', '机械通气', 'Ventilator exposure cohort starting point.', '机械通气暴露队列起点。'],
    ['vasopressor', 'Vasopressor exposure', '血管活性药物暴露', 'Shock/pressor cohort starting point.', '休克/升压药队列起点。'],
    ['respiratory', 'Respiratory failure', '呼吸衰竭', 'Respiratory support and blood-gas focused cohort.', '呼吸支持与血气相关队列。'],
    ['icd', 'Diagnosis / ICD cohort', '诊断 / ICD 队列', 'Enter ICD prefixes or diagnosis terms manually.', '手动输入 ICD 前缀或诊断关键词。'],
  ];
  const REAL_EXPORT_COHORT_PRESETS = new Set(['all_icu', 'adult_first', 'adult_all', 'sepsis3', 'aki', 'ventilation', 'vasopressor', 'respiratory', 'icd']);
  const CONCEPT_DERIVED_COHORT_PRESETS = new Set(['sepsis3', 'aki', 'ventilation', 'vasopressor', 'respiratory']);

  // Feature modules — aligned to the real concept catalog (concept_catalog.py:
  // CONCEPT_GROUP_NAMES, 19 groups). Counts are a fallback; render-time values
  // come from window.EU_CATALOG.groupConcepts so the UI follows the backend.
  // [name_en, name_zh, fallbackConceptCount, selected, isCore]
  const MODS = [
    // —— recommended core ——
    ['Demographics', '人口统计', 6, true, true],
    ['Vital signs', '生命体征', 11, true, true],
    ['Lab — Chemistry', '实验室-生化', 30, true, true],
    ['SOFA-2 scores', 'SOFA-2 评分', 7, true, true],
    ['Sepsis-3 (SOFA-2)', 'Sepsis-3 (SOFA-2)', 1, true, true],
    ['Outcome', '结局', 10, true, true],
    // —— additional modules ——
    ['SOFA-1 scores', 'SOFA-1 评分', 7, true, false],
    ['Sepsis-3 (SOFA-1)', 'Sepsis-3 (SOFA-1)', 1, true, false],
    ['Sepsis shared', 'Sepsis 共享概念', 5, true, false],
    ['Respiratory', '呼吸系统', 15, true, false],
    ['Ventilator', '呼吸机参数', 12, true, false],
    ['Blood gas', '血气分析', 9, true, false],
    ['Lab — Hematology', '实验室-血液学', 22, true, false],
    ['Vasopressors', '血管活性药物', 17, true, false],
    ['Other medications', '其他药物', 49, true, false],
    ['Renal & urine output', '肾脏与尿量', 22, true, false],
    ['Neurological', '神经系统', 11, true, false],
    ['Circulatory', '循环系统', 3, true, false],
    ['Other scores', '其他评分', 9, true, false],
  ];
  const CORE = MODS.filter(m => m[4]).map(m => m[0]);
  const EX_KEYS = {
    'Demographics': 'demographics', 'Vital signs': 'vitals', 'Lab — Chemistry': 'chemistry',
    'SOFA-2 scores': 'sofa2_score', 'Sepsis-3 (SOFA-2)': 'sepsis3_sofa2', 'Outcome': 'outcome',
    'SOFA-1 scores': 'sofa1_score', 'Sepsis-3 (SOFA-1)': 'sepsis3_sofa1', 'Sepsis shared': 'sepsis_shared',
    'Respiratory': 'respiratory', 'Ventilator': 'ventilator', 'Blood gas': 'blood_gas',
    'Lab — Hematology': 'hematology', 'Vasopressors': 'vasopressors', 'Other medications': 'medications',
    'Renal & urine output': 'renal', 'Neurological': 'neurological', 'Circulatory': 'circulatory',
    'Other scores': 'other_scores',
  };
  const EX_EXT = { csv: 'csv', excel: 'xlsx', parquet: 'parquet' };

  function selMods() { return MODS.filter(m => m[3]); }
  function moduleKey(m) { return EX_KEYS[m[0]] || m[0].toLowerCase(); }
  function moduleConceptCount(m) {
    const groups = window.EU_CATALOG && window.EU_CATALOG.groupConcepts;
    const members = groups && groups[moduleKey(m)];
    return Array.isArray(members) ? members.length : (m[2] || 0);
  }
  function conceptIdsForModuleKey(key) {
    const groups = window.EU_CATALOG && window.EU_CATALOG.groupConcepts;
    const members = groups && groups[key];
    return Array.isArray(members) ? members.slice() : [];
  }
  function conceptIdsForModule(m) { return conceptIdsForModuleKey(moduleKey(m)); }
  function conceptMeta(id) {
    const dict = window.EU_CATALOG && window.EU_CATALOG.dict;
    const desc = window.EU_CATALOG && window.EU_CATALOG.desc;
    const row = dict && dict[id];
    const note = desc && desc[id];
    return {
      id,
      name: row ? t(row[0], row[1]) : id,
      unit: row && row[2] ? row[2] : '',
      desc: note ? t(note[0], note[1]) : '',
    };
  }
  function selectedConceptIdsForModule(m) {
    const key = moduleKey(m);
    const ids = conceptIdsForModule(m);
    if (!ids.length) return [];
    const saved = Array.isArray(exSelectedConcepts[key]) ? exSelectedConcepts[key] : ids;
    const savedSet = new Set(saved);
    return ids.filter(id => savedSet.has(id));
  }
  function selectedConceptCount(m) {
    const ids = conceptIdsForModule(m);
    if (!ids.length) return m[3] ? moduleConceptCount(m) : 0;
    return m[3] ? selectedConceptIdsForModule(m).length : 0;
  }
  function conceptTotal(modules) { return (modules || []).reduce((a, m) => a + selectedConceptCount(m), 0); }
  function conceptN() { return conceptTotal(selMods()); }
  function coreConceptN() { return conceptTotal(MODS.filter(m => m[4])); }
  function modKeys() { return selMods().map(moduleKey); }
  function coreModuleKeys() { return MODS.filter(m => m[4]).map(moduleKey); }
  function runModuleKeys(mode) { return mode === 'recommended' ? coreModuleKeys() : modKeys(); }
  function selectedConceptPayload(moduleKeys) {
    const wanted = new Set(moduleKeys || modKeys());
    const payload = {};
    MODS.forEach(m => {
      const key = moduleKey(m);
      if (!wanted.has(key)) return;
      const ids = conceptIdsForModule(m);
      const selected = ids.length ? selectedConceptIdsForModule(m) : [];
      if (selected.length) payload[key] = selected;
    });
    return payload;
  }
  function setModuleConceptSelection(m, ids) {
    const key = moduleKey(m);
    const all = conceptIdsForModule(m);
    const allowed = new Set(all);
    const clean = (ids || []).filter(id => allowed.has(id));
    if (!all.length || clean.length === all.length) delete exSelectedConcepts[key];
    else exSelectedConcepts[key] = clean;
    m[3] = !all.length ? !!m[3] : clean.length > 0;
  }
  function setAllConceptsInModule(m, on) {
    if (on) {
      delete exSelectedConcepts[moduleKey(m)];
      m[3] = true;
    } else {
      setModuleConceptSelection(m, []);
    }
    window.EU_STALE = true;
  }
  function toggleConceptInModule(m, concept) {
    const selected = new Set(selectedConceptIdsForModule(m));
    if (selected.has(concept)) selected.delete(concept);
    else selected.add(concept);
    setModuleConceptSelection(m, conceptIdsForModule(m).filter(id => selected.has(id)));
    window.EU_STALE = true;
  }
  function repaint() { if (window.__euRender) window.__euRender(); }
  function escHtml(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }
  function rememberExportPath(path, opts) {
    if (!path) return;
    try { localStorage.setItem('easyicu_last_export_dir', path); } catch (e) {}
    if (window.EU_API && window.EU_API.registerWorkspaceSource) {
      window.EU_API.registerWorkspaceSource(path, opts || { active: true, crossdb: true })
        .then(() => { if (window.__euRender) window.__euRender(); })
        .catch(err => { console.warn('[EasyICU] source registry update failed:', err); });
    }
  }
  function exportDestinationLabel() {
    return currentExportDir() || t('No export destination selected', '尚未选择导出目录');
  }
  function exportDestinationHint() {
    if (currentExportDir()) {
      return t(
        'Each extraction creates a timestamped folder inside the selected destination.',
        '每次抽取都会在所选目录下创建带时间戳的文件夹。'
      );
    }
    return t(
      'Choose an export destination before extraction starts.',
      '开始抽取前必须先选择导出目录。'
    );
  }
  function exportDestinationRequiredMessage() {
    return t('Choose an export destination before extracting.', '请先选择导出目录再开始抽取。');
  }
  function currentExportDir() {
    return exExportDir || (window.EU_SETTINGS && window.EU_SETTINGS.export_dir) || '';
  }
  function setExportDir(path) {
    exExportDir = path || null;
    if (window.EU_API && window.EU_API.saveSetting) {
      window.EU_API.saveSetting('export_dir', exExportDir).catch(err => {
        console.warn('[EasyICU] export_dir setting save failed:', err);
      });
    }
  }
  function preparedDestinationHint() {
    return t('local prepared-data folder', '本地预处理数据文件夹');
  }
  function pathDisplay(path) {
    return path ? path : t('No folder selected', '尚未选择文件夹');
  }

  /* continuity hook for Copilot / resume */
  window.__euExtractApply = function (modules) {
    if (Array.isArray(modules) && modules.length) {
      MODS.forEach(m => { m[3] = modules.includes(m[0]); });
    } else if (modules && typeof modules === 'object') {
      const keys = Array.isArray(modules.modules) ? new Set(modules.modules) : null;
      if (keys) MODS.forEach(m => { m[3] = keys.has(moduleKey(m)) || keys.has(m[0]); });
      if (modules.concepts && typeof modules.concepts === 'object') exSelectedConcepts = modules.concepts;
    }
    exView = 'done';
  };

  /* Switching the global data source invalidates any completed extraction:
     the previous run belonged to the other source. Reset to the start so the
     screen never shows a stale "Extraction complete" for the new source. */
  window.__euExtractReset = function () {
    exView = 'home';
    exReal = 'connect';
    if (convTimer) { clearInterval(convTimer); convTimer = null; }
  };

  function teardownExportES() { if (exportES) { try { exportES.close(); } catch (e) {} exportES = null; } }
  function runExtract(mode) {
    const runMode = mode === 'recommended' ? 'recommended' : 'custom';
    const modules = runModuleKeys(runMode);
    if (!modules.length) return;
    const outDir = currentExportDir();
    if (!outDir) {
      exAdvExport = true;
      repaint();
      return;
    }
    const real = dataMode() === 'real';
    const support = runMode === 'recommended' ? { ok: true, reason: 'recommended' } : cohortExportSupport();
    if (real && !support.ok) {
      teardownExportES();
      exportRunMode = runMode;
      exportRunModules = modules;
      exportProg = null;
      exportResult = null;
      exportJobId = null;
      exportCancelRequested = false;
      exportErr = support.message || t('This cohort cannot be exported yet.', '这个队列暂不能导出。');
      exView = 'running';
      repaint();
      return;
    }
    teardownExportES();
    exportRunMode = runMode;
    exportRunModules = modules;
    exportProg = null; exportResult = null; exportErr = null; exportJobId = null; exportCancelRequested = false;
    exView = 'running'; repaint();
    const database = (exScanResult && exScanResult.db_key) || 'miiv';
    const conceptSelection = selectedConceptPayload(modules);
    // Real path: only in real mode with a scanned/ready folder + live backend.
    if (real && exPath && window.EU_API && window.EU_API.postJSON && window.EventSource) {
      const payload = {
        path: exPath, database: database, modules: modules,
        format: exFormat, merge: exMerge === 'merged', max_patients: exMaxPatients,
        cohort: runMode === 'recommended' ? recommendedCohortContract() : cohortContract(),
      };
      payload.out_dir = outDir;
      if (Object.keys(conceptSelection).length) payload.concepts = conceptSelection;
      window.EU_API.postJSON('/api/jobs/extract', payload).then(r => {
        exportJobId = r.job_id;
        exportES = new EventSource('/api/jobs/' + r.job_id + '/events');
        exportES.onmessage = ev => {
          let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
          if (m.type === 'progress') { exportProg = m; }
          else if (m.type === 'cancel_requested') {
            exportCancelRequested = true;
            exportProg = { phase: 'cancel', message: t('Cancel requested. The current database read may finish before the job stops.', '已请求取消。当前数据库读取可能会先完成，然后任务停止。') };
          }
          else if (m.type === 'end') {
            if (m.status === 'done') {
              exportResult = m.result || {};
              window.EU_LAST_EXPORT = exportResult;
              if (exportResult.out_dir) {
                rememberExportPath(exportResult.out_dir);
              }
              exView = 'done'; window.EU_STALE = false; window.EU_HASWORK = true;
            }
            else if (m.status === 'cancelled') {
              exportErr = t('Extraction cancelled before completion.', '抽取已在完成前取消。');
            }
            else { exportErr = m.error || 'extraction failed'; }
            teardownExportES();
          }
          repaint();
        };
        exportES.onerror = () => {
          if (!exportResult && !exportErr) exportErr = t('Lost connection to the extraction job.', '与抽取任务的连接中断。');
          teardownExportES(); repaint();
        };
      }).catch(err => { exportErr = String(err && err.message || err); repaint(); });
    } else {
      // Demo / offline fallback: keep the lightweight mock completion.
      setTimeout(() => { exView = 'done'; window.EU_STALE = false; window.EU_HASWORK = true; repaint(); }, 1200);
    }
  }
  function resetToCore() { MODS.forEach(m => { m[3] = m[4]; }); exSelectedConcepts = {}; window.EU_STALE = true; }
  function setAllModules(on) { MODS.forEach(m => { m[3] = !!on; }); exSelectedConcepts = {}; window.EU_STALE = true; }
  function cancelExportJob() {
    if (!exportJobId || exportCancelRequested || !window.EU_API || !window.EU_API.postJSON) return;
    exportCancelRequested = true;
    exportProg = { phase: 'cancel', message: t('Cancel requested. The current database read may finish before the job stops.', '已请求取消。当前数据库读取可能会先完成，然后任务停止。') };
    repaint();
    window.EU_API.postJSON('/api/jobs/' + exportJobId + '/cancel', { reason: 'user_requested' })
      .catch(err => { exportErr = String(err && err.message || err); repaint(); });
  }

  /* ---- real-data connect + convert (the onboarding cliff) ---- */
  const CONV_STEPS = [
    ['Scan source tables', '扫描源数据表'],
    ['Convert to Parquet', '转换为 Parquet'],
    ['Verify concept mapping', '校验概念映射'],
    ['Index & freeze', '建立索引并冻结'],
  ];
  function connectState() {
    const opt = (src, ico, tEn, tZh, dEn, dZh) => `
      <button class="modcard" data-ex-src="${src}" style="align-items:flex-start;padding:14px;">
        <span style="flex:none;color:var(--ink-3);margin-top:1px;">${icon(ico, 17)}</span>
        <span style="min-width:0;"><span class="nm" style="display:block;">${t(tEn, tZh)}</span><span style="display:block;font-size:11px;color:var(--ink-4);margin-top:2px;font-weight:400;">${t(dEn, dZh)}</span></span>
      </button>`;
    return `
      <div class="cfg ex-connect-card">
        <div class="cfg-head">
          <div class="cfg-ico">${icon('folder', 17)}</div>
          <div class="grow"><div class="cfg-h">${t('Connect your data', '连接你的数据')}</div><div class="cfg-sub">${t('local-only · nothing is uploaded', '仅本地 · 不上传任何数据')}</div></div>
        </div>
        <div class="cfg-body">
          <label class="ex-connect-label" for="exPathInput">${t('Data folder on this machine', '本机上的数据文件夹')}</label>
          <div class="path-field editable ex-connect-path">
            <span class="pf-ico">${icon('folder', 14)}</span>
            <input class="pf-input" id="exPathInput" type="text" spellcheck="false" autocomplete="off" value="${escHtml(exPath)}" placeholder="${t('Paste or browse to a local ICU folder', '粘贴或浏览选择本机 ICU 文件夹')}" aria-label="${t('Data folder path', '数据文件夹路径')}" />
            <button class="btn sm ghost ex-connect-browse" data-ex-browse>${icon('folder', 13)} ${t('Browse…', '浏览…')}</button>
          </div>
          <div class="ex-connect-primary">
            <div class="ex-connect-copy">
              <div class="ex-connect-copy-ico">${icon('shield', 15)}</div>
              <div>
                <div class="ex-connect-copy-title">${t('Let EasyICU identify the folder', '让 EasyICU 自动识别文件夹')}</div>
                <div class="ex-connect-copy-desc">${t('We inspect filenames, manifests, and table layout only. No patient rows are uploaded or returned.', '只检查文件名、manifest 和表结构。不上传,也不返回患者行。')}</div>
              </div>
            </div>
            <button class="btn primary ex-connect-analyze" data-ex-analyze>${icon('search', 14)} ${t('Analyze folder', '识别数据目录')}</button>
          </div>
          <div class="ex-connect-actions">
            <button class="ex-linkbtn" data-ex-manual>${icon('sliders', 13)} ${t('Advanced: choose manually', '高级:手动选择')} <span class="chev">${icon('chevdown', 13)}</span></button>
            <button class="ex-linkbtn" data-ex-sample>${icon('play', 13)} ${t('Explore a sample study', '体验示例研究')}</button>
          </div>
          <div class="adv-body mt-12" ${exManualSourceOpen ? '' : 'hidden'} data-ex-manual-body>
            <div class="pf-hint" style="font-size:11px;color:var(--ink-4);margin-bottom:10px;">${t('Use this only if automatic detection is wrong:', '仅在自动识别不正确时使用:')}</div>
            <div class="col gap-8">
              ${opt('prepared', 'layers', 'Prepared data path', '已转换的路径', 'Converted ICU tables such as Parquet/FST — ready now.', '已转换的 ICU 表(如 Parquet/FST),可直接使用。')}
              ${opt('module', 'download', 'Module export folder', '模块导出文件夹', 'A prior EasyICU export with a manifest — ready now.', '带 manifest 的既有 EasyICU 导出,可直接使用。')}
              ${opt('raw', 'db', 'Raw ICU files (CSV / CSV.GZ)', '原始 ICU 文件', 'Original CSV/CSV.GZ — needs one-time conversion first.', '原始 CSV/CSV.GZ,需先做一次性转换。')}
            </div>
          </div>
        </div>
      </div>`;
  }
  /* what a folder scan reports for each source kind */
  const DETECTED = {
    prepared: { db: 'MIMIC-IV v3.1', layout: ['Prepared (Parquet)', '已转换 (Parquet)'], tables: 26, modules: 19, ready: true },
    module:   { db: 'MIMIC-IV v3.1', layout: ['EasyICU module export', 'EasyICU 模块导出'], tables: 8, modules: 8, ready: true },
    raw:      { db: 'MIMIC-IV v3.1', layout: ['Raw CSV.GZ', '原始 CSV.GZ'], tables: 26, modules: 19, ready: false, size: '~2.1 GB', est: t('~3 min', '约 3 分钟') },
  };
  function scanningState() {
    return `
      <div class="cfg" style="max-width:680px;">
        <div class="cfg-head">
          <div class="cfg-ico"><span class="spin sm" style="width:17px;height:17px;"></span></div>
          <div class="grow"><div class="cfg-h">${t('Scanning folder…', '正在扫描文件夹…')}</div><div class="cfg-sub mono">${escHtml(pathDisplay(exPath))}</div></div>
          <span class="pill warn" style="height:20px;"><span class="dot"></span>${t('reading', '读取中')}</span>
        </div>
        <div class="cfg-body">
          <div class="col gap-8">
            ${[['Locate source folder', '定位源文件夹'], ['Identify database layout', '识别数据库结构'], ['Count tables & columns', '统计表与字段']].map((s, i) =>
              `<div class="conv-step ${i === 0 ? 'done' : i === 1 ? 'run' : ''}"><div class="conv-node">${i === 0 ? icon('check', 12, 2.8) : i === 1 ? '<span class="spin sm" style="width:12px;height:12px;border-top-color:#fff;"></span>' : icon('clock', 11)}</div><div><div class="conv-t">${t(s[0], s[1])}</div></div><div></div></div>`).join('')}
          </div>
          <div class="note info mt-16" style="padding:10px 12px;"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="d" style="font-size:11px;margin:0;">${t('Read-only — EasyICU only inspects structure here, it doesn’t open patient rows yet.', '只读 —— 此处仅检查结构,尚未读取任何患者数据。')}</div></div></div>
        </div>
      </div>`;
  }
  // Map the live /api/data/scan payload onto the shape scanResultState renders.
  function scanFacts() {
    if (exScanResult) {
      const r = exScanResult;
      return {
        db: r.db || t('Unknown', '未知'),
        layout: r.layout || ['—', '—'],
        source: r.source || 'unknown',
        tables: r.tables, modules: r.modules,
        ready: !!r.ready,
        size: r.size_hint || '',
        est: r.source === 'raw' ? t('one-time', '一次性') : '',
        missing: r.missing_tables || [],
      };
    }
    const d = DETECTED[exSource] || DETECTED.prepared;  // offline fallback
    return { db: d.db, layout: d.layout, source: exSource || 'prepared', tables: d.tables, modules: d.modules,
             ready: d.ready, size: d.size || '', est: d.est || '', missing: [] };
  }
  function scanErrorState() {
    const msg = exScanError === 'no_path'
      ? t('Choose or paste a local folder path first.', '请先选择或粘贴本机文件夹路径。')
      : exScanError === 'not_a_directory'
      ? t('That path is not a folder on this machine.', '该路径不是本机上的文件夹。')
      : exScanError === 'permission_denied'
      ? t('Permission denied reading that folder.', '没有读取该文件夹的权限。')
      : exScanError === 'unrecognized_folder'
      ? t('EasyICU could not identify a supported ICU data layout in that folder.', 'EasyICU 未能在该文件夹中识别出支持的 ICU 数据结构。')
      : t('Could not scan that folder.', '无法扫描该文件夹。');
    return `
      <div class="cfg" style="max-width:680px;">
        <div class="cfg-head">
          <div class="cfg-ico" style="color:var(--bad,#c0392b);">${icon('alert', 17)}</div>
          <div class="grow"><div class="cfg-h">${t('Folder not recognized', '未识别该文件夹')}</div><div class="cfg-sub mono">${escHtml(pathDisplay(exPath))}</div></div>
        </div>
        <div class="cfg-body">
          <div class="note mt-4" style="padding:11px 13px;background:color-mix(in srgb,var(--bad,#c0392b) 7%,transparent);border-color:color-mix(in srgb,var(--bad,#c0392b) 22%,transparent);">
            <div class="ico" style="color:var(--bad,#c0392b);">${icon('alert', 15)}</div>
            <div class="body"><div class="d" style="font-size:11.5px;margin:0;">${msg}</div></div>
          </div>
          <div class="row gap-8 mt-16">
            <button class="btn primary" data-ex-browse>${icon('folder', 13)} ${t('Choose another folder', '换一个文件夹')}</button>
            <button class="btn ghost" data-ex-rescan>${t('Back', '返回')}</button>
          </div>
        </div>
      </div>`;
  }
  function scanResultState() {
    if (exScanError) return scanErrorState();
    const d = scanFacts();
    const okHead = `
        <div class="cfg-head">
          <div class="cfg-ico" style="color:var(--ok);">${icon('check', 17, 2.6)}</div>
          <div class="grow"><div class="cfg-h">${t('Folder recognized', '已识别该文件夹')}</div><div class="cfg-sub mono">${escHtml(pathDisplay(exPath))}</div></div>
          <span class="pill ok" style="height:20px;"><span class="dot"></span>${d.db}</span>
        </div>`;
    const facts = [
      [t('Database', '数据库'), d.db],
      [t('Layout', '结构'), t(d.layout[0], d.layout[1])],
      [t('Source tables', '源数据表'), String(d.tables)],
      [t('Mappable modules', '可映射模块'), String(d.modules)],
    ];
    const sourceLabel = d.source === 'module'
      ? t('EasyICU module export', 'EasyICU 模块导出')
      : d.source === 'raw'
      ? t('Raw ICU files', '原始 ICU 文件')
      : t('Prepared ICU data path', '已转换 ICU 数据路径');
    const cta = d.ready
      ? `<button class="btn primary" data-ex-usedata>${icon('arrow', 14)} ${d.source === 'module' ? t('Use this export', '使用这个导出') : t('Continue with prepared data', '继续使用已转换数据')}</button>`
      : `<button class="btn primary" data-ex-startconv>${icon('refresh', 14)} ${t('Convert raw files', '转换原始文件')}${d.size ? ` · ${d.size}` : ''}${d.est ? ` · ${d.est}` : ''}</button>`;
    const note = d.ready
      ? `<div class="note ok mt-16" style="padding:10px 12px;"><div class="ico">${icon('check', 14, 2.6)}</div><div class="body"><div class="t" style="font-size:12px;">${t('Detected', '已识别')}: ${sourceLabel}</div><div class="d" style="font-size:11px;margin:0;">${d.source === 'module' ? t('This looks like an existing EasyICU export. Register it as a reusable local source.', '这看起来是已有 EasyICU 导出。可以注册为可复用的本地来源。') : t('Already analysis-ready — no conversion needed. You can extract straight away.', '已是可分析格式 —— 无需转换,可直接抽取。')}</div></div></div>`
      : `<div class="note info mt-16" style="padding:10px 12px;"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="d" style="font-size:11px;margin:0;">${t('One-time conversion to Parquet. Runs on your machine; results are cached so a re-run is instant.', '一次性转换为 Parquet。在本机运行;结果会缓存,再次运行会很快。')}</div></div></div>`;
    return `
      <div class="cfg" style="max-width:680px;">
        ${okHead}
        <div class="cfg-body">
          <div class="cols-2" style="gap:10px 18px;">
            ${facts.map(([k, v]) => `<div class="setup-row"><span class="k">${k}</span><span class="vv">${v}</span></div>`).join('')}
          </div>
          ${note}
          <div class="row gap-8 mt-16">
            ${cta}
            <button class="btn ghost" data-ex-rescan>${icon('back', 13)} ${t('Pick another folder', '换一个文件夹')}</button>
          </div>
        </div>
      </div>`;
  }
  function convFailState() {
    const failAt = 2; // step index that failed
    return `
      <div class="cfg" style="max-width:680px;">
        <div class="cfg-head">
          <div class="cfg-ico" style="color:var(--bad,#c0392b);">${icon('alert', 17)}</div>
          <div class="grow"><div class="cfg-h">${t('Conversion paused', '转换已暂停')}</div><div class="cfg-sub mono">${escHtml(pathDisplay(exPath))} → ${preparedDestinationHint()}</div></div>
          <span class="pill" style="height:20px;background:color-mix(in srgb,var(--bad,#c0392b) 14%,transparent);color:var(--bad,#c0392b);"><span class="dot" style="background:var(--bad,#c0392b);"></span>${t('1 step failed', '1 步失败')}</span>
        </div>
        <div class="cfg-body">
          ${CONV_STEPS.map((s, i) => {
            const st = i < failAt ? 'done' : i === failAt ? 'fail' : '';
            const node = i < failAt ? icon('check', 12, 2.8) : i === failAt ? icon('close', 11, 3) : icon('clock', 11);
            return `<div class="conv-step ${st}"><div class="conv-node" ${i === failAt ? 'style="background:var(--bad,#c0392b);border-color:var(--bad,#c0392b);color:#fff;"' : ''}>${node}</div><div><div class="conv-t">${t(s[0], s[1])}</div></div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">${i < failAt ? 'ok' : i === failAt ? t('error', '错误') : ''}</div></div>`;
          }).join('')}
          <div class="note mt-16" style="padding:11px 13px;background:color-mix(in srgb,var(--bad,#c0392b) 7%,transparent);border-color:color-mix(in srgb,var(--bad,#c0392b) 22%,transparent);">
            <div class="ico" style="color:var(--bad,#c0392b);">${icon('alert', 15)}</div>
            <div class="body">
              <div class="t" style="font-size:12.5px;">${t('Verify concept mapping failed', '概念映射校验失败')}</div>
              <div class="d" style="font-size:11.5px;">${t('4 source tables had unmapped columns (e.g. labevents.valueuom). Steps 1–2 are cached, so resuming re-runs only this step.', '有 4 张源表存在未映射字段(如 labevents.valueuom)。步骤 1–2 已缓存,继续后仅重跑此步。')}</div>
            </div>
          </div>
          <div class="row gap-8 mt-16">
            <button class="btn primary" data-ex-resume>${icon('refresh', 14)} ${t('Resume from step 3', '从第 3 步继续')}</button>
            <button class="btn" data-nav="settings">${icon('sliders', 13)} ${t('Edit column mapping', '编辑字段映射')}</button>
            <button class="btn ghost" data-ex-rescan>${t('Start over', '重新开始')}</button>
          </div>
        </div>
      </div>`;
  }
  // Driven by the live convert job (/api/jobs/convert + SSE). Renders real
  // per-file progress instead of the four fake CONV_STEPS.
  function convertingState() {
    const done = !!convResult && !convErr;
    const err = !!convErr;
    const p = convProg || {};
    const cur = p.current || 0, tot = p.total || 0;
    const pct = tot ? Math.round((cur / tot) * 100) : (done ? 100 : 0);
    const c = (convResult && convResult.converted != null) ? convResult
            : (p.counts || { converted: 0, failed: 0, skipped: 0 });
    const headIco = err ? `<div class="cfg-ico" style="color:var(--bad,#c0392b);">${icon('alert', 17)}</div>`
                  : done ? `<div class="cfg-ico" style="color:var(--ok);">${icon('check', 17, 2.6)}</div>`
                  : `<div class="cfg-ico"><span class="spin sm" style="width:17px;height:17px;"></span></div>`;
    const headTitle = err ? t('Conversion failed', '转换失败')
                    : done ? (convResult.nothing_to_do ? t('Already converted', '已转换') : t('Conversion complete', '转换完成'))
                    : t('Converting raw files…', '正在转换原始文件…');
    const pill = err ? `<span class="pill" style="height:20px;background:color-mix(in srgb,var(--bad,#c0392b) 14%,transparent);color:var(--bad,#c0392b);"><span class="dot" style="background:var(--bad,#c0392b);"></span>${t('failed', '失败')}</span>`
               : done ? `<span class="pill ok" style="height:20px;"><span class="dot"></span>${t('done', '完成')}</span>`
               : `<span class="pill warn" style="height:20px;"><span class="dot"></span>${t('running', '进行中')}</span>`;
    let body;
    if (err) {
      body = `
        <div class="note mt-4" style="padding:11px 13px;background:color-mix(in srgb,var(--bad,#c0392b) 7%,transparent);border-color:color-mix(in srgb,var(--bad,#c0392b) 22%,transparent);">
          <div class="ico" style="color:var(--bad,#c0392b);">${icon('alert', 15)}</div>
          <div class="body"><div class="d mono" style="font-size:11.5px;margin:0;">${convErr}</div></div>
        </div>
        <div class="row gap-8 mt-16">
          <button class="btn primary" data-ex-startconv>${icon('refresh', 14)} ${t('Retry conversion', '重试转换')}</button>
          <button class="btn ghost" data-ex-rescan>${t('Start over', '重新开始')}</button>
        </div>`;
    } else {
      const bar = `
        <div style="height:8px;border-radius:999px;background:var(--surface-2,#eef0f4);overflow:hidden;margin:4px 0 10px;">
          <div style="height:100%;width:${pct}%;background:var(--accent,#2f7d6b);transition:width .25s;"></div>
        </div>`;
      const line = done
        ? (convResult.nothing_to_do
            ? t('All source tables were already converted — nothing to do.', '所有源表此前已转换 —— 无需重复处理。')
            : `${t('Converted', '已转换')} <b>${c.converted}</b> ${t('tables', '张表')}${c.skipped ? ` · ${c.skipped} ${t('cached', '缓存')}` : ''}${c.failed ? ` · ${c.failed} ${t('failed', '失败')}` : ''}.`)
        : `${tot ? `[${cur}/${tot}] ` : ''}${p.file ? `<span class="mono">${p.file}</span>` : t('preparing…', '准备中…')}${p.rows != null ? ` · ${Number(p.rows).toLocaleString()} ${t('rows', '行')}` : ''}`;
      body = `
        ${bar}
        <div style="font-size:12px;color:var(--ink-3);min-height:18px;">${line}</div>
        ${done
          ? `<div class="row gap-8 mt-16"><button class="btn primary" data-ex-convdone>${icon('arrow', 14)} ${t('Continue to extraction', '继续抽取')}</button></div>`
          : `<div class="note info mt-16" style="padding:10px 12px;"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="d" style="font-size:11px;margin:0;">${t('Runs entirely on your machine. Already-converted tables are skipped, so a re-run is fast.', '全程在本机运行。已转换的表会跳过,再次运行很快。')}</div></div></div>`}`;
    }
    return `
      <div class="cfg" style="max-width:680px;">
        <div class="cfg-head">
          ${headIco}
          <div class="grow"><div class="cfg-h">${headTitle}</div><div class="cfg-sub mono">${escHtml(pathDisplay(exPath))} → ${preparedDestinationHint()}</div></div>
          ${pill}
        </div>
        <div class="cfg-body">${body}</div>
      </div>`;
  }
  function teardownConvES() { if (convES) { try { convES.close(); } catch (e) {} convES = null; } }
  function startConvert() {
    teardownConvES();
    convProg = null; convResult = null; convErr = null; convJobId = null;
    exReal = 'converting'; convFail = false; repaint();
    const database = (exScanResult && exScanResult.db_key) || 'miiv';
    // Live path: submit a convert job, stream progress over SSE.
    if (window.EU_API && window.EU_API.postJSON && window.EventSource) {
      window.EU_API.postJSON('/api/jobs/convert', { path: exPath, database: database })
        .then(r => {
          convJobId = r.job_id;
          convES = new EventSource('/api/jobs/' + convJobId + '/events');
          convES.onmessage = ev => {
            let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
            if (m.type === 'progress') { convProg = m; }
            else if (m.type === 'end') {
              if (m.status === 'done') convResult = m.result || {};
              else convErr = m.error || 'conversion failed';
              teardownConvES();
            }
            if (exReal === 'converting') repaint();
          };
          convES.onerror = () => {
            // Stream dropped before 'end' — surface as a soft error unless done.
            if (!convResult && !convErr) { convErr = t('Lost connection to the conversion job.', '与转换任务的连接中断。'); }
            teardownConvES(); if (exReal === 'converting') repaint();
          };
        })
        .catch(err => { convErr = String(err && err.message || err); if (exReal === 'converting') repaint(); });
    } else {
      // Offline fallback: simple deterministic completion (no fake failure).
      convDone = 0; if (convTimer) clearInterval(convTimer);
      convTimer = setInterval(() => {
        convDone++;
        if (convDone >= CONV_STEPS.length) { clearInterval(convTimer); convTimer = null; convResult = { converted: CONV_STEPS.length, failed: 0, skipped: 0 }; }
        repaint();
      }, 600);
    }
  }
  function resumeConvert() { startConvert(); }  // re-run is idempotent: cached tables are skipped
  /* ---- server-side folder picker (net-new; the mock Browse was a no-op) ----
     A browser file input cannot enumerate the user's folders, so the local
     FastAPI process lists directories on demand via /api/fs/list. */
  let pickerEl = null;
  function ensurePickerStyles() {
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
  function closePicker() { if (pickerEl) { pickerEl.remove(); pickerEl = null; } document.removeEventListener('keydown', pickerKey); }
  function pickerKey(e) { if (e.key === 'Escape') closePicker(); }
  function cleanFolderName(raw) {
    return String(raw || '').trim().replace(/[\\/]+/g, '-');
  }
  function joinLocalPath(parent, name) {
    const base = String(parent || '').trim();
    if (!base) return name;
    return base + (base.endsWith('/') ? '' : '/') + name;
  }
  function openFolderPicker(startPath, onPick, title, options) {
    ensurePickerStyles();
    closePicker();
    const opts = options || {};
    let cur = startPath || '';
    const pickerTitle = title || t('Choose a data folder', '选择数据文件夹');
    const back = document.createElement('div'); back.className = 'eu-pick-back';
    back.innerHTML = `
      <div class="eu-pick" role="dialog" aria-label="${escHtml(pickerTitle)}">
        <div class="eu-pick-h">
          <span style="color:var(--ink-3);">${icon('folder', 16)}</span>
          <span class="t">${escHtml(pickerTitle)}</span>
          <span class="grow" style="flex:1;"></span>
          <button class="btn sm ghost" data-pk-close>${icon('close', 13)}</button>
        </div>
        <div class="eu-pick-cur" data-pk-cur></div>
        <div class="eu-pick-sc" data-pk-sc></div>
        <div class="eu-pick-list" data-pk-list><div class="eu-pick-empty">${t('Loading…', '加载中…')}</div></div>
        ${opts.allowCreate ? `
          <div class="eu-pick-create">
            <input data-pk-new-name placeholder="${escHtml(t('New folder name', '新文件夹名称'))}" />
            <button class="btn sm" data-pk-new>${icon('plus', 13)} ${t('Create folder', '创建文件夹')}</button>
            <div class="eu-pick-msg" data-pk-msg>${t('Create inside the folder shown above, then use it as the export destination.', '会在上方当前目录内创建，并把它作为导出目录。')}</div>
          </div>` : ''}
        <div class="eu-pick-f">
          <button class="btn ghost sm" data-pk-up>${icon('back', 13)} ${t('Up', '上一级')}</button>
          <span style="flex:1;"></span>
          <button class="btn primary" data-pk-use>${icon('check', 13)} ${t('Use this folder', '选择此文件夹')}</button>
        </div>
      </div>`;
    document.body.appendChild(back); pickerEl = back;
    const listEl = back.querySelector('[data-pk-list]');
    const curEl = back.querySelector('[data-pk-cur]');
    const scEl = back.querySelector('[data-pk-sc]');
    const newNameEl = back.querySelector('[data-pk-new-name]');
    const newBtn = back.querySelector('[data-pk-new]');
    const msgEl = back.querySelector('[data-pk-msg]');
    back.addEventListener('click', e => { if (e.target === back) closePicker(); });
    back.querySelector('[data-pk-close]').addEventListener('click', closePicker);
    back.querySelector('[data-pk-use]').addEventListener('click', () => { closePicker(); if (cur) onPick(cur); });
    if (newBtn && newNameEl && msgEl) {
      newBtn.addEventListener('click', () => {
        const name = cleanFolderName(newNameEl.value);
        msgEl.classList.remove('err');
        if (!cur) {
          msgEl.textContent = t('Choose a parent folder first.', '请先选择父目录。');
          msgEl.classList.add('err');
          return;
        }
        if (!name || name === '.' || name === '..') {
          msgEl.textContent = t('Enter a valid folder name.', '请输入有效的文件夹名称。');
          msgEl.classList.add('err');
          return;
        }
        if (!(window.EU_API && window.EU_API.createDir)) {
          msgEl.textContent = t('Folder creation endpoint is unavailable.', '文件夹创建接口不可用。');
          msgEl.classList.add('err');
          return;
        }
        const target = joinLocalPath(cur, name);
        newBtn.disabled = true;
        msgEl.textContent = t('Creating local folder…', '正在创建本地文件夹…');
        window.EU_API.createDir(target).then(r => {
          if (!r || !r.ok) throw new Error((r && (r.error || r.message)) || 'mkdir_failed');
          const createdPath = r.path || target;
          if (opts.pickCreated) {
            closePicker();
            onPick(createdPath);
          } else {
            newNameEl.value = '';
            msgEl.textContent = t('Folder created.', '文件夹已创建。');
            load(createdPath);
          }
        }).catch(err => {
          msgEl.textContent = String(err && err.message || err);
          msgEl.classList.add('err');
        }).finally(() => {
          if (pickerEl === back) newBtn.disabled = false;
        });
      });
      newNameEl.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          newBtn.click();
        }
      });
    }
    document.addEventListener('keydown', pickerKey);

    function load(path) {
      listEl.innerHTML = `<div class="eu-pick-empty">${t('Loading…', '加载中…')}</div>`;
      window.EU_API.listDir(path).then(r => {
        cur = r.path || path || '';
        curEl.textContent = cur || '/';
        const up = back.querySelector('[data-pk-up]'); up.disabled = !r.parent;
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
          b.innerHTML = `<span style="color:var(--ink-3);flex:none;">${icon('folder', 15)}</span><span class="nm">${en.name}</span>${en.hint ? `<span class="hint">${en.hint}</span>` : ''}`;
          b.onclick = () => load(en.path); listEl.appendChild(b);
        });
      }).catch(err => {
        listEl.innerHTML = `<div class="eu-pick-empty">${t('Failed to list folder', '列目录失败')}: ${String(err && err.message || err)}</div>`;
      });
    }
    load(cur);
  }

  function startScan(src) {
    if (!String(exPath || '').trim()) {
      exSource = src || null;
      exScanResult = null;
      exScanError = 'no_path';
      exReal = 'scanresult';
      repaint();
      return;
    }
    exSource = src || null; exScanResult = null; exScanError = null;
    exReal = 'scanning'; repaint();
    // Live scan via the FastAPI backend. Falls back to the mock DETECTED table
    // only if the API is unreachable (e.g. opened as a static file).
    if (window.EU_API && window.EU_API.scanPath) {
      window.EU_API.scanPath(exPath, src).then(r => {
        if (exReal !== 'scanning') return;          // user navigated away
        if (r && r.ok) { exScanResult = r; exSource = r.source || src || null; exReal = 'scanresult'; }
        else { exScanError = (r && r.error) || 'scan_failed'; exReal = 'scanresult'; }
        repaint();
      }).catch(err => {
        if (exReal !== 'scanning') return;
        exScanError = String(err && err.message || err); exReal = 'scanresult'; repaint();
      });
    } else {
      setTimeout(() => { exReal = 'scanresult'; repaint(); }, 950);
    }
  }

  /* cross-mode hand-off → Copilot (same pipeline, conversational) */
  function handoffBar() {
    return `
    <div class="handoff">
      <span class="ho-ico">${icon('spark', 17)}</span>
      <div class="ho-body"><b>${t('Prefer to talk it through?', '想用对话完成?')}</b> ${t('Guided study can pick the cohort, modules and export for you in plain conversation — same pipeline, same result.', '研究引导能用对话帮你选好队列、模块和导出 —— 同一条流水线,同样的结果。')}</div>
      <button class="btn" data-nav="guided">${icon('spark', 13)} ${t('Start Guided study', '开始研究引导')} ${icon('arrow', 13)}</button>
    </div>`;
  }

  /* ---- the express recommended card ---- */
  function expressCard() {
    const exportReady = !!currentExportDir();
    return `
    <div class="express">
      <div class="express-grid">
        <div>
          <div class="eyebrow">${icon('spark', 13)} ${t('Recommended', '推荐')}</div>
          <h2>${t('Recommended extraction', '推荐抽取')}</h2>
          <p class="lead">${t('Sensible defaults that work for most ICU studies — first ICU stay, the full available ICU window with a 30-day cap, and the six core feature modules. One click gives you analysis-ready tables.', '适用于大多数 ICU 研究的合理默认 —— 首次 ICU 住院、全可用 ICU 时间窗（30 天上限）,以及六个核心特征模块。一键得到可直接分析的数据表。')}</p>
          <div class="express-chips">
            ${CORE.map((n, i) => `<span class="chip solid">${t(n, MODS.find(m => m[0] === n)[1])}</span>`).join('')}
          </div>
          <div class="express-meta">
            <span>${t('Cohort', '队列')} · <b>${t('first ICU stay', '首次 ICU')} · ${t('full window', '全窗口')}</b></span>
            <span>${t('Modules', '模块')} · <b>${CORE.length}</b></span>
            <span>${t('Concepts', '概念')} · <b>~${coreConceptN()}</b></span>
            <span>${dataMode() === 'demo' ? t('Stays', '住院数') + ' · <b>10</b>' : t('Source', '来源') + ' · <b>' + t('local', '本地') + '</b>'}</span>
          </div>
        </div>
        <div class="express-cta">
          <button class="btn primary lg" data-ex-run="recommended" ${exportReady ? '' : 'disabled'}>${icon('play', 16)} ${t('Run recommended extraction', '运行推荐抽取')}</button>
          <div class="note-line" style="${exportReady ? '' : 'color:var(--warn,#a66a00);'}">${icon(exportReady ? 'shield' : 'alert', 12)} ${exportReady ? (dataMode() === 'demo' ? t('reproducible · no tokens', '可复现 · 不消耗 token') : t('local-only · nothing uploaded', '仅本地 · 不上传')) : exportDestinationRequiredMessage()}</div>
        </div>
      </div>
    </div>`;
  }

  function cohortPresetMeta() {
    return COHORT_PRESETS.find(p => p[0] === exCohortPreset) || COHORT_PRESETS[1];
  }
  function cohortPresetIsRealExportReady(id) {
    return REAL_EXPORT_COHORT_PRESETS.has(id);
  }
  function cohortPresetUsesConceptPrefilter(id) {
    return CONCEPT_DERIVED_COHORT_PRESETS.has(id);
  }
  function fmtHours(v) {
    const n = Math.max(0, Number(v || 0));
    if (!n) return t('Any', '不限');
    if (n % 24 === 0) return n + 'h · ' + (n / 24) + 'd';
    return n + 'h';
  }
  function fmtObservationWindow(v) {
    const n = Math.max(1, Number(v || DEFAULT_OBSERVATION_WINDOW_HOURS));
    if (n >= MAX_OBSERVATION_WINDOW_HOURS) return t('full available · 30d cap', '全可用 · 30天上限');
    return t('first ', '前 ') + fmtHours(n);
  }
  function fmtAgeRange() {
    if (exAgeMin <= 0 && exAgeMax >= 100) return t('all ages', '全部年龄');
    if (exAgeMin <= 0) return '≤ ' + exAgeMax;
    if (exAgeMax >= 100) return '≥ ' + exAgeMin;
    return exAgeMin + '–' + exAgeMax;
  }
  function sepsisDefinitionRelevant() {
    const sepsis = window.EUExtractionSepsis;
    return sepsis && sepsis.relevant ? sepsis.relevant(modKeys()) : false;
  }
  function sepsisDefinitionContract() {
    const sepsis = window.EUExtractionSepsis;
    return sepsis && sepsis.contract ? sepsis.contract() : {};
  }
  function cohortContract() {
    const icd = (window.EUIcd && window.EUIcd.contract) ? window.EUIcd.contract() : {};
    return {
      preset: exCohortPreset,
      age_min: exAgeMin,
      age_max: exAgeMax,
      min_icu_los_hours: exMinLosHours,
      observation_window_hours: exWindowHours,
      exclude_readmissions: exExcludeReadmissions,
      icd_enabled: exCohortPreset === 'icd',
      sepsis_definition: sepsisDefinitionContract(),
      ...icd,
    };
  }
  function recommendedCohortContract() {
    return {
      preset: 'adult_first',
      age_min: 18,
      age_max: 100,
      min_icu_los_hours: 0,
      observation_window_hours: DEFAULT_OBSERVATION_WINDOW_HOURS,
      exclude_readmissions: true,
      icd_enabled: false,
      sepsis_definition: sepsisDefinitionContract(),
    };
  }
  function cohortExportSupport() {
    if (dataMode() !== 'real') return { ok: true, reason: 'demo' };
    if (!cohortPresetIsRealExportReady(exCohortPreset)) {
      return {
        ok: false,
        reason: 'unsupported_preset',
        message: t(
          'This cohort preset is not wired to a real export denominator yet. Use All ICU, Adult first ICU stay, or Diagnosis / ICD cohort.',
          '这个队列预设还没有真实导出分母接线。请先使用全部 ICU、成年首次 ICU，或诊断 / ICD 队列。'
        ),
      };
    }
    if (exCohortPreset === 'icd') {
      const icd = (window.EUIcd && window.EUIcd.contract) ? window.EUIcd.contract() : {};
      const hasTokens = (icd.include_diagnoses || []).length || (icd.exclude_diagnoses || []).length;
      if (!hasTokens) {
        return {
          ok: false,
          reason: 'empty_icd_filter',
          message: t(
            'Add at least one ICD include or exclude token before exporting a diagnosis cohort.',
            '导出诊断队列前，请至少填写一个 ICD 包含或排除条件。'
          ),
        };
      }
    }
    return { ok: true, reason: 'ready' };
  }
  function cohortPresetCards() {
    return `
      <div class="cohort-preset-grid">
        ${COHORT_PRESETS.map(([id, en, zh, den, dzh]) => `
          <button class="cohort-preset ${exCohortPreset === id ? 'on' : ''} ${dataMode() === 'real' && !cohortPresetIsRealExportReady(id) ? 'pending' : ''}" data-ex-cohort-preset="${id}">
            <span class="cp-dot"></span>
            <span class="cp-body"><span class="cp-title">${t(en, zh)}</span><span class="cp-sub">${t(den, dzh)}${dataMode() === 'real' && !cohortPresetIsRealExportReady(id) ? ' · ' + t('planned, not export-ready', '计划中，暂不可真实导出') : ''}${dataMode() === 'real' && cohortPresetUsesConceptPrefilter(id) ? ' · ' + t('concept prefilter, slower', '概念预筛，较慢') : ''}</span></span>
          </button>`).join('')}
      </div>`;
  }
  function rangeCtl(key, value, min, max, step, display) {
    return `
      <div class="range-ctl">
        <input type="range" min="${min}" max="${max}" step="${step}" value="${value}" data-ex-range="${key}" />
        <span class="range-val mono" data-ex-range-value="${key}">${display}</span>
      </div>`;
  }
  function ageRangeCtl() {
    return `
      <div class="range-pair">
        <div class="range-ctl">
          <input type="range" min="0" max="100" step="1" value="${exAgeMin}" data-ex-range="age_min" />
          <span class="range-val mono" data-ex-range-value="age_min">min ${exAgeMin}</span>
        </div>
        <div class="range-ctl">
          <input type="range" min="0" max="100" step="1" value="${exAgeMax}" data-ex-range="age_max" />
          <span class="range-val mono" data-ex-range-value="age_max">max ${exAgeMax >= 100 ? '100+' : exAgeMax}</span>
        </div>
      </div>`;
  }
  function updateRangeLabel(root, key) {
    const el = root.querySelector(`[data-ex-range-value="${key}"]`);
    if (!el) return;
    if (key === 'age_min') el.textContent = 'min ' + exAgeMin;
    else if (key === 'age_max') el.textContent = 'max ' + (exAgeMax >= 100 ? '100+' : exAgeMax);
    else if (key === 'los_min') el.textContent = fmtHours(exMinLosHours);
    else if (key === 'window') el.textContent = fmtObservationWindow(exWindowHours);
  }
  function cohortChips() {
    const preset = cohortPresetMeta();
    const chips = [
      t(preset[1], preset[2]),
      t('age ', '年龄 ') + fmtAgeRange(),
      exMinLosHours ? t('ICU LOS ≥ ', 'ICU 时长 ≥ ') + fmtHours(exMinLosHours) : t('any ICU LOS', '不限 ICU 时长'),
      t('window: ', '观察窗:') + fmtObservationWindow(exWindowHours),
    ];
    if (exExcludeReadmissions) chips.push(t('exclude readmissions', '排除再入院'));
    return chips.map(c => `<span class="chip solid">${escHtml(c)}</span>`).join('');
  }

  /* ---- cohort cfg ---- */
  function cohortCfg() {
    const preset = cohortPresetMeta();
    const showICD = exCohortPreset === 'icd';
    const support = cohortExportSupport();
    return `
    <div class="cfg">
      <div class="cfg-head">
        <div class="cfg-ico">${icon('cohort', 17)}</div>
        <div class="grow"><div class="cfg-h">${t('Cohort', '队列')}</div><div class="cfg-sub">${t('who is included', '纳入哪些患者')}</div></div>
        <span class="pill"><span class="dot" style="background:var(--ok);"></span>${t('10 stays matched', '匹配 10 次住院')}</span>
      </div>
      <div class="cfg-body">
        <div class="cfg-chips">
          ${cohortChips()}
        </div>
        <button class="adv-toggle ${exAdvCohort ? 'open' : ''}" data-ex-advc>${t('Adjust inclusion criteria', '调整纳入标准')} <span class="chev">${icon('chevdown', 13)}</span></button>
        <div class="adv-body" ${exAdvCohort ? '' : 'hidden'}>
          <div class="col gap-12">
            <div>
              <div class="row" style="justify-content:space-between;gap:12px;align-items:flex-start;">
                <div><div style="font-size:12.5px;font-weight:600;color:var(--ink-2);">${t('Cohort preset', '队列预设')}</div><div style="font-size:11px;color:var(--ink-4);margin-top:2px;">${t('Pick the clinical starting point; use ICD only when you want a diagnosis-code cohort.', '选择临床起点;只有需要诊断编码队列时才使用 ICD。')}</div></div>
                <span class="pill">${escHtml(t(preset[1], preset[2]))}</span>
              </div>
              ${cohortPresetCards()}
              ${!support.ok ? `<div class="note warn mt-12" style="padding:10px 12px;"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t" style="font-size:12px;">${t('Real export is blocked for this cohort', '此队列暂不能真实导出')}</div><div class="d" style="font-size:11px;margin:0;">${support.message}</div></div></div>` : ''}
              ${support.ok && dataMode() === 'real' && cohortPresetUsesConceptPrefilter(exCohortPreset) ? `<div class="note info mt-12" style="padding:10px 12px;"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t" style="font-size:12px;">${t('Clinical cohort prefilter', '临床队列预筛')}</div><div class="d" style="font-size:11px;margin:0;">${t('This preset computes the defining concepts on the selected denominator before exporting matched stays, so it may take longer than demographic or ICD filters.', '该预设会先在所选分母上计算定义概念，再导出匹配住院，因此可能比人口学或 ICD 筛选更慢。')}</div></div></div>` : ''}
            </div>
            ${advRow(t('Age range at admission', '入院年龄范围'), ageRangeCtl())}
            ${advRow(t('Minimum ICU LOS', '最短 ICU 时长'), rangeCtl('los_min', exMinLosHours, 0, 168, 1, fmtHours(exMinLosHours)))}
            ${advRow(t('Observation window', '观察窗口'), rangeCtl('window', exWindowHours, 1, MAX_OBSERVATION_WINDOW_HOURS, 1, fmtObservationWindow(exWindowHours)))}
            ${advRow(t('Exclude readmissions', '排除再入院'), switchEl(exExcludeReadmissions, 'readmissions'))}
          </div>
          <div style="border-top:1px solid var(--hair);margin-top:14px;padding-top:14px;">
            ${showICD && window.EUIcd ? window.EUIcd.block() : `
              <div class="note info" style="padding:10px 12px;">
                <div class="ico">${icon('shield', 14)}</div>
                <div class="body"><div class="t" style="font-size:12px;">${t('ICD filter is off by default', 'ICD 默认关闭')}</div><div class="d" style="font-size:11px;margin:0;">${t('Choose “Diagnosis / ICD cohort” above to enter code prefixes or diagnosis terms. Nothing is prefilled.', '在上方选择“诊断 / ICD 队列”后再输入编码前缀或诊断关键词。不会预填任何编码。')}</div></div>
              </div>`}
          </div>
          <div style="border-top:1px solid var(--hair);margin-top:14px;padding-top:14px;">
            <div class="row" style="justify-content:space-between;gap:12px;align-items:flex-start;">
              <div><div style="font-size:12.5px;font-weight:600;color:var(--ink-2);">${t('Real-source filter audit', '真实来源筛选审计')}</div><div style="font-size:11px;color:var(--ink-4);margin-top:2px;">${t('Metadata-only checks from the active registered export; unsupported cohort filters fail closed.', '从当前注册导出的元数据计算;未支持的队列筛选保持 fail-closed。')}</div></div>
            </div>
            <div class="ex-filter-card">${filterSourceBody()}</div>
          </div>
        </div>
      </div>
    </div>`;
  }
  function advRow(label, ctl) {
    return `<div class="adv-row"><span class="adv-label">${label}</span><span class="adv-control">${ctl}</span></div>`;
  }
  function switchEl(on, key) { return `<span class="switch ${on ? 'on' : ''}" role="switch" aria-checked="${on}" tabindex="0" ${key ? `data-ex-switch="${key}"` : ''}></span>`; }

  function filterSourceBody() {
    if (dataMode() !== 'real') {
      return `
        <div class="note info mt-16" style="padding:10px 12px;">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="t" style="font-size:12px;">${t('Seeded demo filters', '种子演示筛选')}</div><div class="d" style="font-size:11px;margin:0;">${t('This is seeded demo metadata, not a real export source. Switch to Real and register an EasyICU export to compute filter provenance.', '这是种子演示元数据,不是真实导出源。切换到真实模式并注册 EasyICU 导出后才会计算筛选来源。')}</div></div>
        </div>`;
    }
    if (exFilterLoading) {
      return `
        <div class="note info mt-16" style="padding:10px 12px;">
          <div class="ico"><span class="spin sm" style="width:14px;height:14px;"></span></div>
          <div class="body"><div class="d" style="font-size:11px;margin:0;">${t('Reading registered export metadata…', '正在读取已注册导出的元数据…')}</div></div>
        </div>`;
    }
    if (exFilterError) {
      return `
        <div class="note mt-16" style="padding:10px 12px;background:color-mix(in srgb,var(--bad,#c0392b) 7%,transparent);border-color:color-mix(in srgb,var(--bad,#c0392b) 22%,transparent);">
          <div class="ico" style="color:var(--bad,#c0392b);">${icon('alert', 14)}</div>
          <div class="body"><div class="t" style="font-size:12px;">${t('Advanced filters failed closed', '高级筛选已 fail-closed')}</div><div class="d mono" style="font-size:11px;margin:0;">${escHtml(exFilterError)}</div></div>
        </div>
        <button class="btn sm ghost mt-12" data-ex-filter-load>${icon('refresh', 13)} ${t('Retry metadata check', '重试元数据检查')}</button>`;
    }
    if (!exFilterOptions) {
      return `
        <div class="note info mt-16" style="padding:10px 12px;">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="d" style="font-size:11px;margin:0;">${t('Advanced filters are computed from the active registered EasyICU export. Unsupported cohort-row filters stay blocked rather than being guessed.', '高级筛选从当前注册的 EasyICU 导出计算。未支持的队列行级筛选会保持阻断,不会猜测。')}</div></div>
        </div>
        <button class="btn sm mt-12" data-ex-filter-load>${icon('refresh', 13)} ${t('Load real filter options', '加载真实筛选选项')}</button>`;
    }
    const src = exFilterOptions.source || {};
    const mods = (exFilterPreview && exFilterPreview.matched_modules) || ((exFilterOptions.options || {}).modules || []);
    const unsupported = (((exFilterOptions.filters || {}).unsupported) || []).slice(0, 5);
    const qualitySeg = `<div class="seg" data-ex-filter-quality>
      ${['all', 'ok', 'warn', 'bad', 'neutral', 'unknown'].map(q => `<button class="${exQualityStatus === q ? 'active' : ''}" data-val="${q}">${q}</button>`).join('')}
    </div>`;
    const coverageSeg = `<div class="seg" data-ex-filter-coverage>
      ${[0, 50, 80].map(v => `<button class="${exMinCoveragePct === v ? 'active' : ''}" data-val="${v}">≥ ${v}%</button>`).join('')}
    </div>`;
    return `
      <div class="note ok mt-16" style="padding:10px 12px;">
        <div class="ico">${icon('check', 14, 2.6)}</div>
        <div class="body">
          <div class="t" style="font-size:12px;">${t('Real filter provenance', '真实筛选来源')}</div>
          <div class="d" style="font-size:11px;margin:0;">${escHtml(src.label || 'local')} · ${escHtml(src.database || 'unknown')} · <span class="mono">${escHtml(src.id || src.path_hash || '')}</span> · ${t('hash', '哈希')} <span class="mono">${escHtml(src.path_hash || '')}</span></div>
        </div>
      </div>
      <div class="col gap-10 mt-12">
        ${advRow(t('Minimum module coverage', '最低模块覆盖度'), coverageSeg)}
        ${advRow(t('Quality status', '质量状态'), qualitySeg)}
      </div>
      <div class="cols-2 mt-12" style="gap:8px;">
        ${mods.slice(0, 6).map(m => `
          <div class="ledger-row">
            <span class="ledger-ico">${icon(m.quality_status === 'ok' ? 'check' : 'shield', 14)}</span>
            <div><div class="mono" style="font-weight:600;font-size:12px;">${escHtml(m.module)}</div><div style="font-size:11px;color:var(--ink-4);">${Number(m.row_count || 0).toLocaleString()} ${t('rows', '行')} · ${m.coverage_pct == null ? 'coverage n/a' : m.coverage_pct + '%'} · ${escHtml(m.quality_status)}</div></div>
          </div>`).join('')}
      </div>
      <div class="row gap-8 mt-12">
        <button class="btn sm" data-ex-filter-apply>${icon('refresh', 13)} ${t('Preview supported filters', '预览支持的筛选')}</button>
        <button class="btn sm ghost" data-ex-filter-usemods ${mods.length ? '' : 'disabled'}>${icon('layers', 13)} ${t('Use matched modules', '使用匹配模块')}</button>
      </div>
      <div class="note info mt-12" style="padding:10px 12px;">
        <div class="ico">${icon('alert', 14)}</div>
        <div class="body"><div class="t" style="font-size:12px;">${t('Unsupported filters stay blocked', '未支持筛选保持阻断')}</div><div class="d" style="font-size:11px;margin:0;">${unsupported.map(u => escHtml(u.id)).join(', ')}</div></div>
      </div>`;
  }

  function loadExtractionFilters() {
    if (!(window.EU_API && window.EU_API.loadExtractionFilterOptions)) return;
    exFilterLoading = true; exFilterError = null; repaint();
    window.EU_API.loadExtractionFilterOptions({})
      .then(r => { exFilterOptions = r; exFilterPreview = null; exFilterError = null; })
      .catch(err => { exFilterError = String(err && err.message || err); exFilterOptions = null; exFilterPreview = null; })
      .finally(() => { exFilterLoading = false; repaint(); });
  }

  function previewExtractionFilters() {
    if (!(window.EU_API && window.EU_API.previewExtractionFilters)) return;
    exFilterLoading = true; exFilterError = null; repaint();
    const filters = { min_coverage_pct: exMinCoveragePct };
    if (exQualityStatus !== 'all') filters.quality_statuses = [exQualityStatus];
    window.EU_API.previewExtractionFilters({ filters })
      .then(r => { exFilterPreview = r; exFilterError = null; })
      .catch(err => { exFilterError = String(err && err.message || err); exFilterPreview = null; })
      .finally(() => { exFilterLoading = false; repaint(); });
  }

  function useMatchedFilterModules() {
    const mods = exFilterPreview && Array.isArray(exFilterPreview.matched_modules)
      ? exFilterPreview.matched_modules
      : (exFilterOptions && exFilterOptions.options && exFilterOptions.options.modules) || [];
    const keys = new Set(mods.map(m => m.module));
    if (!keys.size) return;
    MODS.forEach(m => { m[3] = keys.has(EX_KEYS[m[0]] || m[0].toLowerCase()); });
    window.EU_STALE = true;
    repaint();
  }

  /* ---- modules cfg ---- */
  function sepsisDefinitionPanel() {
    const sepsis = window.EUExtractionSepsis;
    return sepsis && sepsis.panel
      ? sepsis.panel({ moduleKeys: modKeys(), t, icon, escHtml })
      : '';
  }
  function conceptRows(m) {
    const ids = conceptIdsForModule(m);
    const selected = new Set(selectedConceptIdsForModule(m));
    if (!ids.length) {
      return `<div class="mod-empty">${t('Concept catalog is still loading for this module.', '这个模块的特征字典仍在加载。')}</div>`;
    }
    return ids.map(id => {
      const meta = conceptMeta(id);
      const on = selected.has(id);
      return `
        <button class="concept-toggle ${on ? 'on' : ''}" data-ex-concept="${escHtml(id)}" title="${escHtml(meta.desc || meta.name)}">
          <span class="mk">${on ? icon('check', 10, 3) : ''}</span>
          <span class="cx-main"><span class="cx-name">${escHtml(meta.name)}</span><span class="cx-id mono">${escHtml(id)}</span></span>
          ${meta.unit ? `<span class="cx-unit mono">${escHtml(meta.unit)}</span>` : ''}
        </button>`;
    }).join('');
  }
  function modCard(m, i) {
    const on = m[3];
    const key = moduleKey(m);
    const total = moduleConceptCount(m);
    const selected = selectedConceptCount(m);
    const open = exExpandedMod === key;
    return `
    <article class="modcard ${on ? 'on' : ''} ${open ? 'open' : ''} ${m[4] ? 'core' : ''}" data-ex-mod-card="${i}">
      <div class="modcard-head">
        <button class="mod-pick" data-ex-mod="${i}">
          <span class="mk">${on ? icon('check', 11, 3) : ''}</span>
          <span class="nm">${t(m[0], m[1])}</span>
          <span class="ct mono">${selected}/${total}</span>
        </button>
        <button class="mod-detail-btn" data-ex-mod-details="${i}" aria-expanded="${open ? 'true' : 'false'}">
          ${open ? t('Hide features', '收起特征') : t('Choose features', '选择特征')} ${icon('chevdown', 12)}
        </button>
      </div>
      ${open ? `
        <div class="mod-concepts">
          <div class="mod-concept-toolbar">
            <span class="hint">${selected} / ${total} ${t('features selected', '个特征已选')}</span>
            <span class="spacer"></span>
            <button class="linkbtn" data-ex-concepts-all="${i}">${t('All in module', '本模块全选')}</button>
            <button class="linkbtn" data-ex-concepts-clear="${i}">${t('Clear module', '清空本模块')}</button>
          </div>
          <div class="concept-list">${conceptRows(m)}</div>
        </div>` : ''}
    </article>`;
  }
  function modulesCfg() {
    const shown = exShowAllMods ? MODS : MODS.filter(m => m[4]);
    const selectedCount = selMods().length;
    return `
    <div class="cfg">
      <div class="cfg-head">
        <div class="cfg-ico">${icon('layers', 17)}</div>
        <div class="grow"><div class="cfg-h">${t('Feature modules', '特征模块')}</div><div class="cfg-sub"><span id="exModSub">${selMods().length} ${t('modules', '模块')} · ${conceptN()} ${t('concepts', '概念')}</span></div></div>
        <div class="row gap-6" style="flex-wrap:wrap;justify-content:flex-end;">
          <button class="btn sm ghost" data-ex-selectall ${selectedCount === MODS.length ? 'disabled' : ''}>${icon('check', 13)} ${t('Select all', '全选')}</button>
          <button class="btn sm ghost" data-ex-clearmods ${selectedCount === 0 ? 'disabled' : ''}>${icon('close', 13)} ${t('Clear all', '清空')}</button>
          <button class="btn sm ghost" data-ex-core>${icon('refresh', 13)} ${t('Core 6', '核心 6 项')}</button>
        </div>
      </div>
      <div class="cfg-body">
        <div class="modgrid" id="exModGrid">
          ${shown.map(m => modCard(m, MODS.indexOf(m))).join('')}
        </div>
        ${sepsisDefinitionPanel()}
        ${selectedCount === 0 ? `<div class="note mt-12" style="padding:10px 12px;background:color-mix(in srgb,var(--warn,#b45309) 8%,transparent);border-color:color-mix(in srgb,var(--warn,#b45309) 22%,transparent);"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d" style="font-size:11px;margin:0;">${t('Select at least one module before extracting.', '抽取前至少选择一个模块。')}</div></div></div>` : ''}
        <button class="adv-toggle ${exShowAllMods ? 'open' : ''}" data-ex-allmods>${exShowAllMods ? t('Show core modules only', '仅显示核心模块') : t('Show all ' + MODS.length + ' modules', '显示全部 ' + MODS.length + ' 个模块')} <span class="chev">${icon('chevdown', 13)}</span></button>
      </div>
    </div>`;
  }

  /* ---- export cfg ---- */
  function exportCfg() {
    const fmtSeg = `<div class="seg" data-ex-fmt><button class="${exFormat === 'parquet' ? 'active' : ''}" data-val="parquet">Parquet</button><button class="${exFormat === 'csv' ? 'active' : ''}" data-val="csv">CSV</button><button class="${exFormat === 'excel' ? 'active' : ''}" data-val="excel">Excel</button></div>`;
    const destination = currentExportDir();
    const hintTone = destination ? 'var(--ink-4)' : 'var(--warn,#a66a00)';
    return `
    <div class="cfg">
      <div class="cfg-head">
        <div class="cfg-ico">${icon('download', 17)}</div>
        <div class="grow"><div class="cfg-h">${t('Export', '导出')}</div><div class="cfg-sub">${t('package & destination', '打包与保存位置')}</div></div>
        <span class="pill"><span class="dot"></span>${exFormat.toUpperCase()}</span>
      </div>
      <div class="cfg-body">
        ${advRow(t('Format', '格式'), fmtSeg)}
        <div class="ex-export-destination">
          <div class="path-field ex-export-path"><span class="pf-ico">${icon('folder', 14)}</span><span class="pf-path ${destination ? '' : 'muted'}">${escHtml(exportDestinationLabel())}</span></div>
          <button class="btn sm ghost ex-export-browse" data-ex-export-browse>${icon('folder', 13)} ${t('Browse...', '浏览...')}</button>
          <button class="btn sm ghost ex-export-create" data-ex-export-create>${icon('plus', 13)} ${t('New folder', '新建目录')}</button>
        </div>
        <div class="note-line mt-8" style="font-size:11px;color:${hintTone};">${icon(destination ? 'shield' : 'alert', 11)} ${exportDestinationHint()}</div>
        <button class="adv-toggle ${exAdvExport ? 'open' : ''}" data-ex-adve>${t('Advanced export options', '高级导出选项')} <span class="chev">${icon('chevdown', 13)}</span></button>
        <div class="adv-body" ${exAdvExport ? '' : 'hidden'}>
          <div class="col gap-12">
            ${advRow(t('Merge mode', '合并方式'), `<div class="seg" data-ex-merge><button class="${exMerge === 'separate' ? 'active' : ''}" data-val="separate">${t('Separate', '分文件')}</button><button class="${exMerge === 'merged' ? 'active' : ''}" data-val="merged">${t('Merge one', '合并单文件')}</button></div>`)}
            ${advRow(t('Filter by patient list', '按患者列表筛选'), switchEl(false))}
            ${advRow(t('Include row index', '包含行索引'), switchEl(false))}
            ${advRow(t('Timestamp filenames', '文件名加时间戳'), switchEl(true))}
          </div>
        </div>
      </div>
    </div>`;
  }

  /* ---- summary rail ---- */
  function summaryRail() {
    const support = cohortExportSupport();
    const exportReady = !!currentExportDir();
    const extractDisabled = !selMods().length || !support.ok || !exportReady;
    const summaryMessage = !exportReady ? exportDestinationRequiredMessage() : (support.ok ? t('local-only · reproducible manifest', '仅本地 · 可复现清单') : support.message);
    return `
    <div class="ex2-summary">
      <div class="sumcard">
        <div class="eyebrow">${t('You will extract', '即将抽取')}</div>
        <div class="sum-row"><span class="k">${t('Source', '数据源')}</span><span class="v">${dataMode() === 'demo' ? t('Demo', '演示') : t('Real', '真实')}</span></div>
        <div class="sum-row"><span class="k">${t('Cohort', '队列')}</span><span class="v">10 ${t('stays', '住院')}</span></div>
        <div class="sum-row"><span class="k">${t('Modules', '模块')}</span><span class="v" id="exSumMods">${selMods().length}</span></div>
        <div class="sum-row"><span class="k">${t('Concepts', '概念')}</span><span class="v" id="exSumConc">${conceptN()}</span></div>
        <div class="sum-row"><span class="k">${t('Format', '格式')}</span><span class="v">${exFormat.toUpperCase()}</span></div>
        <button class="btn primary block mt-16" data-ex-run="custom" ${extractDisabled ? 'disabled' : ''}>${icon('download', 14)} ${t('Extract', '开始抽取')}</button>
        <div class="note-line mt-8" style="font-size:11px;color:${support.ok && exportReady ? 'var(--ink-4)' : 'var(--warn,#a66a00)'};text-align:center;">${icon(support.ok && exportReady ? 'shield' : 'alert', 11)} ${summaryMessage}</div>
      </div>
    </div>`;
  }

  /* ---- running / done states ---- */
  function runningState() {
    const p = exportProg || {};
    const cur = p.current || 0, tot = p.total || 0;
    const pct = tot ? Math.round((cur / tot) * 100) : 0;
    const err = !!exportErr;
    const progressText = p.message || (p.module
      ? `${p.module}${p.rows != null ? ` · ${Number(p.rows).toLocaleString()} ${t('rows', '行')}` : ''}`
      : t('selecting cohort…', '正在选择队列…'));
    return `
    <div class="card pad" style="max-width:680px;margin:0 auto;">
      <div class="load-strip">
        ${err ? `<span style="color:var(--bad,#c0392b);">${icon('alert', 18)}</span>` : `<span class="spin accent"></span>`}
        <div class="grow"><div style="font-weight:600;font-size:13px;">${err ? t('Extraction failed', '抽取失败') : t('Extracting feature modules…', '正在抽取特征模块…')}</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${t('local-only · writing to a timestamped export folder', '仅本地 · 写入带时间戳的导出文件夹')}</div></div>
        ${tot ? `<span class="mono" style="font-size:11px;color:var(--ink-3);">${cur}/${tot}</span>` : ''}
      </div>
      ${err
        ? `<div class="note mt-12" style="padding:11px 13px;background:color-mix(in srgb,var(--bad,#c0392b) 7%,transparent);border-color:color-mix(in srgb,var(--bad,#c0392b) 22%,transparent);"><div class="ico" style="color:var(--bad,#c0392b);">${icon('alert', 15)}</div><div class="body"><div class="d mono" style="font-size:11.5px;margin:0;">${exportErr}</div></div></div>
           <div class="row gap-8 mt-16"><button class="btn primary" data-ex-run="${exportRunMode}">${icon('refresh', 14)} ${t('Retry', '重试')}</button><button class="btn ghost" data-ex-reset>${t('Back', '返回')}</button></div>`
        : `<div style="height:8px;border-radius:999px;background:var(--surface-2,#eef0f4);overflow:hidden;margin:12px 0 8px;"><div style="height:100%;width:${pct}%;background:var(--accent,#2f7d6b);transition:width .25s;"></div></div>
           <div style="font-size:12px;color:var(--ink-3);min-height:18px;">${p.phase === 'cohort' || p.phase === 'cancel' ? `${escHtml(progressText)}` : `<span class="mono">${escHtml(progressText)}</span>`}</div>
           <div class="row mt-12" style="justify-content:flex-end;"><button class="btn sm ghost" data-ex-cancel ${exportCancelRequested || !exportJobId ? 'disabled' : ''}>${icon('alert', 13)} ${exportCancelRequested ? t('Cancel requested', '已请求取消') : t('Request cancel', '请求取消')}</button></div>`}
    </div>`;
  }
  function doneState() {
    const r = exportResult;
    // Live result from the export job; fall back to a descriptive summary offline.
    const files = (r && r.files) ? r.files
      : (exportRunModules || modKeys()).map(k => ({ file: k + '.' + EX_EXT[exFormat], module: k, rows: null }));
    const outDir = (r && r.out_dir) || t('timestamped export folder', '带时间戳导出文件夹');
    const totalRows = r ? r.total_rows : null;
    const fileList = files
      .concat([{ file: '_manifest.json', manifest: true }])
      .concat((r && r.readme) ? [{ file: r.readme, readme: true }] : []);
    return `
      <div class="state-hero success solid" style="max-width:720px;margin:0 auto;">
        <div class="glyph">${icon('check', 26, 2.6)}</div>
        <div class="st-t">${t('Extraction complete', '抽取完成')}</div>
        <div class="st-d">${(r ? r.file_count : files.length)} ${t('concept files', '个概念文件')}${totalRows != null ? ` · ${Number(totalRows).toLocaleString()} ${t('rows total', '行(合计)')}` : ''} + <span class="mono">_manifest.json</span> ${t('written to', '已写入')} <span class="mono">${outDir}</span>. ${t('Everything stayed on your machine.', '全部留在你的机器上。')}</div>
        <div class="st-actions">
          <button class="btn primary" data-nav="patient">${icon('patient', 14)} ${t('Open in Patient Review', '打开患者审阅')}</button>
          <button class="btn" data-nav="agent">${icon('agent', 14)} ${t('Hand off to Research Agent', '交给研究代理')}</button>
          <button class="btn ghost" data-ex-reset>${icon('refresh', 14)} ${t('Extract again', '重新抽取')}</button>
        </div>
      </div>
      <div class="cols-2 mt-20" style="max-width:720px;margin-left:auto;margin-right:auto;">
        ${fileList.map(f => `
          <div class="ledger-row"><span class="ledger-ico">${icon(f.manifest ? 'shield' : 'file', 14)}</span><div><div class="mono" style="font-weight:600;font-size:12px;">${f.file}</div><div style="font-size:11px;color:var(--ink-4);">${f.manifest ? t('reproducibility manifest', '可复现清单') : (f.readme ? t('human-readable extraction README', '可读抽取说明') : (f.rows != null ? Number(f.rows).toLocaleString() + ' ' + t('rows', '行') : (f.module || '')))}</div></div></div>`).join('')}
      </div>`;
  }

  S.extraction = {
    section: 'extraction',
    nav: 'extraction',
    get crumbs() { return [t('Home', '首页'), t('Data Extraction', '数据抽取')]; },
    get status() {
      if (exView === 'done') return `<span class="pill ok"><span class="dot"></span>${t('extracted', '已抽取')}</span>`;
      return '';
    },
    rail() {
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Extraction', '抽取')}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Source', '数据源')}</span><span class="vv">${dataMode() === 'demo' ? t('Demo · 10', '演示 · 10') : t('Real · local', '真实 · 本地')}</span></div>
          <div class="setup-row"><span class="k">${t('Modules', '模块')}</span><span class="vv">${selMods().length}</span></div>
          <div class="setup-row"><span class="k">${t('Format', '格式')}</span><span class="vv">${exFormat.toUpperCase()}</span></div>
        </div>
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Two ways', '两种方式')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6">${icon('spark', 13)} ${t('One-click recommended', '一键推荐')}</div>
          <div class="row gap-6">${icon('sliders', 13)} ${t('Customize if needed', '需要时再自定义')}</div>
        </div>
      </div>`;
    },
    render() {
      if (window.__euExtractFocusICD) { exAdvCohort = true; exCustomOpen = true; exCohortPreset = 'icd'; }
      let body;
      const real = dataMode() === 'real';
      if (exView === 'running') body = runningState();
      else if (exView === 'done') body = doneState();
      else if (real && exReal === 'connect') body = connectState();
      else if (real && exReal === 'scanning') body = scanningState();
      else if (real && exReal === 'scanresult') body = scanResultState();
      else if (real && exReal === 'convfail') body = convFailState();
      else if (real && exReal === 'converting') body = convertingState();
      else {
        body = `
          ${expressCard()}
          <div class="ex2-divider">
            <span class="ln"></span>
            <span class="lbl">${t('Need more control?', '需要更多控制?')}</span>
            <button class="ex2-disc ${exCustomOpen ? 'open' : ''}" data-ex-custom>${icon('sliders', 13)} ${t('Customize', '自定义')} <span class="chev">${icon('chevdown', 13)}</span></button>
            <span class="ln"></span>
          </div>
          <div class="ex2-custom" ${exCustomOpen ? '' : 'hidden'}>
            <div class="ex2-layout">
              <div>
                ${cohortCfg()}
                ${modulesCfg()}
                ${exportCfg()}
              </div>
              ${summaryRail()}
            </div>
          </div>
          ${handoffBar()}`;
      }
      return `
      <div class="page-head" style="margin-bottom:18px;">
        <h1>${t('Data Extraction', '数据抽取')}</h1>
        <p class="lead">${t('Turn ICU records into analysis-ready tables. Start with the recommended extraction, or customize every detail.', '把 ICU 记录变成可分析的数据表。可以直接用推荐配置,也可以自定义每个细节。')}</p>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:9px;">${t('Key terms', '关键术语')}: ${window.gloss('SOFA')} · ${window.gloss('Sepsis-3')} · ${window.gloss('cohort', t('cohort', '队列'))} · ${window.gloss('concept', t('concept', '概念'))} · <a class="dict-link" data-nav="dictionary" style="color:var(--accent-ink);cursor:pointer;">${t('Browse data dictionary', '浏览数据字典')} →</a></div>
      </div>
      ${body}`;
    },
    afterRender(root) {
      const c = root.querySelector('.content') || root;
      // real-data connect / scan / convert
      const pathInput = root.querySelector('#exPathInput');
      if (pathInput) pathInput.addEventListener('input', () => { exPath = pathInput.value; });
      const analyzeBtn = root.querySelector('[data-ex-analyze]');
      if (analyzeBtn) analyzeBtn.addEventListener('click', () => {
        if (pathInput) exPath = pathInput.value || exPath;
        startScan(null);
      });
      const manualBtn = root.querySelector('[data-ex-manual]');
      if (manualBtn) manualBtn.addEventListener('click', () => {
        exManualSourceOpen = !exManualSourceOpen;
        repaint();
      });
      root.querySelectorAll('[data-ex-browse]').forEach(browseBtn => browseBtn.addEventListener('click', () => {
        if (window.EU_API && window.EU_API.listDir) {
          openFolderPicker(exPath, picked => {
            exPath = picked;
            exReal = 'connect'; exScanError = null; exScanResult = null;
            repaint();
          });
        } else if (pathInput) { pathInput.focus(); pathInput.select(); }
      }));
      const openExportDestinationPicker = () => {
        if (window.EU_API && window.EU_API.listDir) {
          openFolderPicker(currentExportDir(), picked => {
            setExportDir(picked);
            repaint();
          }, t('Choose or create export destination', '选择或创建导出目录'), { allowCreate: true, pickCreated: true });
        }
      };
      root.querySelectorAll('[data-ex-export-browse], [data-ex-export-create]').forEach(browseBtn => browseBtn.addEventListener('click', () => {
        if (window.EU_API && window.EU_API.listDir) openExportDestinationPicker();
        else browseBtn.focus();
      }));
      root.querySelectorAll('[data-ex-src]').forEach(b => b.addEventListener('click', () => {
        if (pathInput) exPath = pathInput.value || exPath;
        startScan(b.dataset.exSrc);
      }));
      const useDataBtn = root.querySelector('[data-ex-usedata]'); if (useDataBtn) useDataBtn.addEventListener('click', () => {
        if (exSource === 'module') rememberExportPath(exPath);
        exReal = 'ready'; repaint();
      });
      const startConvBtn = root.querySelector('[data-ex-startconv]'); if (startConvBtn) startConvBtn.addEventListener('click', () => { startConvert(); });
      const resumeBtn = root.querySelector('[data-ex-resume]'); if (resumeBtn) resumeBtn.addEventListener('click', () => { resumeConvert(); });
      root.querySelectorAll('[data-ex-rescan]').forEach(b => b.addEventListener('click', () => { teardownConvES(); exReal = 'connect'; exSource = null; exScanResult = null; exScanError = null; convDone = 0; convProg = null; convResult = null; convErr = null; repaint(); }));
      const convDoneBtn = root.querySelector('[data-ex-convdone]'); if (convDoneBtn) convDoneBtn.addEventListener('click', () => { exReal = 'ready'; repaint(); });
      const sampleBtn = root.querySelector('[data-ex-sample]'); if (sampleBtn) sampleBtn.addEventListener('click', () => { if (window.setDataMode) window.setDataMode('demo'); });
      // run
      root.querySelectorAll('[data-ex-run]').forEach(b => b.addEventListener('click', () => runExtract(b.dataset.exRun || 'custom')));
      root.querySelectorAll('[data-ex-cancel]').forEach(b => b.addEventListener('click', cancelExportJob));
      root.querySelectorAll('[data-ex-reset]').forEach(b => b.addEventListener('click', () => { teardownExportES(); exView = 'home'; exportProg = null; exportResult = null; exportErr = null; exportJobId = null; exportCancelRequested = false; exportRunModules = null; repaint(); }));
      // custom disclosure
      const cust = root.querySelector('[data-ex-custom]');
      if (cust) cust.addEventListener('click', () => { exCustomOpen = !exCustomOpen; repaint(); setTimeout(() => { const el = root.querySelector('.ex2-custom'); if (el && exCustomOpen) el.scrollIntoView ? null : null; }, 0); });
      // advanced toggles
      const advc = root.querySelector('[data-ex-advc]'); if (advc) advc.addEventListener('click', () => { exAdvCohort = !exAdvCohort; repaint(); });
      const adve = root.querySelector('[data-ex-adve]'); if (adve) adve.addEventListener('click', () => { exAdvExport = !exAdvExport; repaint(); });
      const allm = root.querySelector('[data-ex-allmods]'); if (allm) allm.addEventListener('click', () => { exShowAllMods = !exShowAllMods; repaint(); });
      root.querySelectorAll('[data-ex-cohort-preset]').forEach(b => b.addEventListener('click', () => {
        exCohortPreset = b.dataset.exCohortPreset || 'adult_first';
        window.EU_STALE = true;
        repaint();
      }));
      root.querySelectorAll('[data-ex-range]').forEach(input => {
        const applyRange = () => {
          const key = input.dataset.exRange;
          const val = Number(input.value || 0);
          if (key === 'age_min') {
            exAgeMin = Math.min(Math.max(0, val), exAgeMax);
            if (val > exAgeMax) input.value = String(exAgeMin);
          } else if (key === 'age_max') {
            exAgeMax = Math.max(Math.min(100, val), exAgeMin);
            if (val < exAgeMin) input.value = String(exAgeMax);
          } else if (key === 'los_min') {
            exMinLosHours = Math.max(0, Math.min(168, val));
          } else if (key === 'window') {
            exWindowHours = Math.max(1, Math.min(MAX_OBSERVATION_WINDOW_HOURS, val));
          }
          updateRangeLabel(root, key);
          window.EU_STALE = true;
        };
        input.addEventListener('input', applyRange);
        input.addEventListener('change', () => { applyRange(); repaint(); });
      });
      root.querySelectorAll('[data-ex-switch="readmissions"]').forEach(s => s.addEventListener('click', () => {
        exExcludeReadmissions = !exExcludeReadmissions;
        window.EU_STALE = true;
        repaint();
      }));
      root.querySelectorAll('[data-ex-filter-load]').forEach(b => b.addEventListener('click', loadExtractionFilters));
      const coverage = root.querySelector('[data-ex-filter-coverage]'); if (coverage) coverage.addEventListener('click', e => {
        const b = e.target.closest('button'); if (!b) return;
        exMinCoveragePct = Number(b.dataset.val || 0);
        previewExtractionFilters();
      });
      const quality = root.querySelector('[data-ex-filter-quality]'); if (quality) quality.addEventListener('click', e => {
        const b = e.target.closest('button'); if (!b) return;
        exQualityStatus = b.dataset.val || 'all';
        previewExtractionFilters();
      });
      root.querySelectorAll('[data-ex-filter-apply]').forEach(b => b.addEventListener('click', previewExtractionFilters));
      root.querySelectorAll('[data-ex-filter-usemods]').forEach(b => b.addEventListener('click', useMatchedFilterModules));
      if (dataMode() === 'real' && exAdvCohort && !exFilterOptions && !exFilterLoading && !exFilterError) {
        setTimeout(loadExtractionFilters, 0);
      }
      if (window.EUExtractionSepsis && window.EUExtractionSepsis.bind) {
        window.EUExtractionSepsis.bind(root, {
          markStale: () => { window.EU_STALE = true; },
          repaint,
        });
      }
      // reset to core
      const core = root.querySelector('[data-ex-core]'); if (core) core.addEventListener('click', () => { resetToCore(); repaint(); });
      const selectAll = root.querySelector('[data-ex-selectall]'); if (selectAll) selectAll.addEventListener('click', () => { setAllModules(true); repaint(); });
      const clearMods = root.querySelector('[data-ex-clearmods]'); if (clearMods) clearMods.addEventListener('click', () => { setAllModules(false); repaint(); });
      // module toggle
      const grid = root.querySelector('#exModGrid');
      if (grid) grid.addEventListener('click', e => {
        const detail = e.target.closest('[data-ex-mod-details]');
        if (detail) {
          const i = +detail.dataset.exModDetails;
          const key = moduleKey(MODS[i]);
          exExpandedMod = exExpandedMod === key ? null : key;
          repaint();
          return;
        }
        const allConcepts = e.target.closest('[data-ex-concepts-all]');
        if (allConcepts) {
          setAllConceptsInModule(MODS[+allConcepts.dataset.exConceptsAll], true);
          repaint();
          return;
        }
        const clearConcepts = e.target.closest('[data-ex-concepts-clear]');
        if (clearConcepts) {
          setAllConceptsInModule(MODS[+clearConcepts.dataset.exConceptsClear], false);
          repaint();
          return;
        }
        const concept = e.target.closest('[data-ex-concept]');
        if (concept) {
          const card = concept.closest('.modcard');
          const i = card ? +card.dataset.exModCard : -1;
          if (i >= 0) toggleConceptInModule(MODS[i], concept.dataset.exConcept);
          repaint();
          return;
        }
        const card = e.target.closest('[data-ex-mod]'); if (!card) return;
        const i = +card.dataset.exMod;
        MODS[i][3] = !MODS[i][3];
        window.EU_STALE = true;  // selection changed → downstream out of date
        repaint();
      });
      // format
      const fmt = root.querySelector('[data-ex-fmt]'); if (fmt) fmt.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; exFormat = b.dataset.val; repaint(); });
      const merge = root.querySelector('[data-ex-merge]'); if (merge) merge.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; exMerge = b.dataset.val; repaint(); });
      // generic segs + switches (display only)
      root.querySelectorAll('.adv-body [data-seg]').forEach(seg => seg.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; seg.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b)); }));
      root.querySelectorAll('.adv-body .switch:not([data-ex-switch])').forEach(s => s.addEventListener('click', () => { s.classList.toggle('on'); s.setAttribute('aria-checked', s.classList.contains('on')); }));
      // ICD disease-cohort filter (folded in from the former standalone screen)
      if (window.EUIcd && window.EUIcd.bind) window.EUIcd.bind(root);
      if (window.__euExtractFocusICD) window.__euExtractFocusICD = false;
    },
  };
})();
