/* Extraction Sepsis-3 parameter panel.
   Owner: Data Extraction route. Keeps the code-aligned Sepsis controls out of
   the main extraction screen IIFE. */
(function () {
  const RUNTIME_PROFILE = 'easyicu_ricu_default_v1';
  const PROFILES = [
    ['sofa2_primary', 'SOFA-2 primary', 'SOFA-2 主口径', 'SOFA-2'],
    ['sofa1_sensitivity', 'SOFA-1 sensitivity', 'SOFA-1 敏感性', 'SOFA-1'],
    ['dual_audit', 'SOFA-2 + SOFA-1 audit', 'SOFA-2 + SOFA-1 审计', 'SOFA-2 + SOFA-1'],
  ];
  const state = {
    profile: 'sofa2_primary',
    siMode: 'auto',
    abxWinHours: 24,
    sampWinHours: 72,
    abxCountWinHours: 24,
    abxMinCount: 1,
    positiveCultures: false,
    siWindow: 'first',
    windowBeforeHours: 48,
    windowAfterHours: 24,
    deltaFunction: 'delta_cummin',
    threshold: 2,
    keepComponents: false,
  };

  function bi(t, en, zh) {
    return (typeof t === 'function' ? t : (window.t || ((a) => a)))(en, zh);
  }

  function esc(escHtml, value) {
    return typeof escHtml === 'function'
      ? escHtml(value)
      : String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  function ic(icon, name, size) {
    return typeof icon === 'function' ? icon(name, size) : '';
  }

  function activeProfile() {
    return PROFILES.find(p => p[0] === state.profile) || PROFILES[0];
  }

  function relevant(moduleKeys) {
    return (moduleKeys || []).some(k => k.startsWith('sepsis3_') || k.startsWith('sofa') || k === 'sepsis_shared');
  }

  function contract() {
    const profile = activeProfile();
    return {
      record_scope: 'metadata_current_runtime_defaults',
      runtime_profile: RUNTIME_PROFILE,
      implementation_profile: profile[0],
      score_family: profile[3],
      suspected_infection: {
        mode: state.siMode,
        abx_win_hours: state.abxWinHours,
        samp_win_hours: state.sampWinHours,
        abx_count_win_hours: state.abxCountWinHours,
        abx_min_count: state.abxMinCount,
        positive_cultures_required: state.positiveCultures,
      },
      sofa_increase: {
        si_window: state.siWindow,
        window_before_si_hours: state.windowBeforeHours,
        window_after_si_hours: state.windowAfterHours,
        delta_function: state.deltaFunction,
        threshold: state.threshold,
        keep_components: state.keepComponents,
      },
      review_options: {
        implementation_profile: PROFILES.map(p => p[0]),
        score_family: PROFILES.map(p => p[3]),
        si_mode: ['auto', 'and', 'icd_abx', 'abx', 'samp', 'or'],
        abx_win_hours: [12, 24, 48],
        samp_win_hours: [24, 48, 72],
        abx_count_win_hours: [12, 24, 48],
        abx_min_count: [1, 2, 3],
        positive_cultures_required: [false, true],
        si_window: ['first', 'last', 'any'],
        window_before_si_hours: [24, 48, 72],
        window_after_si_hours: [12, 24, 48],
        delta_function: ['delta_cummin', 'delta_start', 'delta_min'],
        threshold: [2, 3],
      },
    };
  }

  function profileLabel(t, profile) {
    const p = profile || activeProfile();
    return bi(t, p[1], p[2]);
  }

  function profileSeg(ctx) {
    return `
      <div class="sepsis-def-seg" data-ex-sepsis-profile>
        ${PROFILES.map(p => `<button class="${p[0] === state.profile ? 'active' : ''}" data-val="${esc(ctx.escHtml, p[0])}">${esc(ctx.escHtml, bi(ctx.t, p[1], p[2]))}</button>`).join('')}
      </div>`;
  }

  function optionSeg(ctx, key, options, value) {
    return `
      <div class="sepsis-def-seg" data-ex-sepsis="${key}">
        ${options.map(([val, en, zh]) => `<button class="${String(value) === String(val) ? 'active' : ''}" data-val="${esc(ctx.escHtml, val)}">${esc(ctx.escHtml, bi(ctx.t, en, zh))}</button>`).join('')}
      </div>`;
  }

  function boolSeg(ctx, key, value) {
    return optionSeg(ctx, key, [
      ['false', 'No', '否'],
      ['true', 'Yes', '是'],
    ], String(!!value));
  }

  function siModeLabel(t, value) {
    const map = {
      auto: bi(t, 'auto by database', '按数据库自动'),
      and: bi(t, 'ABX + sample', '抗菌药 + 采样'),
      icd_abx: bi(t, 'infection ICD + ABX', '感染 ICD + 抗菌药'),
      abx: bi(t, 'ABX only', '仅抗菌药'),
      samp: bi(t, 'sample only', '仅采样'),
      or: bi(t, 'ABX or sample', '抗菌药或采样'),
    };
    return map[value] || value;
  }

  function siWindowLabel(t, value) {
    const map = {
      first: bi(t, 'first SI event', '首个 SI 事件'),
      last: bi(t, 'last SI event', '末次 SI 事件'),
      any: bi(t, 'any SI event', '任一 SI 事件'),
    };
    return map[value] || value;
  }

  function deltaLabel(t, value) {
    const map = {
      delta_cummin: bi(t, 'cumulative minimum', '累积最小值'),
      delta_start: bi(t, 'first observed', '首个观测值'),
      delta_min: bi(t, 'sliding minimum', '滑动最小值'),
    };
    return map[value] || value;
  }

  function chip(ctx, label, tone) {
    return `<span class="sepsis-def-chip ${tone || ''}">${esc(ctx.escHtml, label)}</span>`;
  }

  function control(ctx, label, body, help) {
    return `
      <div class="sepsis-def-control">
        <div class="sepsis-def-label">${label}</div>
        ${body}
        ${help ? `<div class="sepsis-def-help">${help}</div>` : ''}
      </div>`;
  }

  function panel(ctx) {
    if (!relevant(ctx.moduleKeys || [])) return '';
    const profile = activeProfile();
    const current = [
      profileLabel(ctx.t, profile),
      siModeLabel(ctx.t, state.siMode),
      bi(ctx.t, 'sample→ABX ', '采样→抗菌药 ') + state.sampWinHours + 'h / ' + bi(ctx.t, 'ABX→sample ', '抗菌药→采样 ') + state.abxWinHours + 'h',
      bi(ctx.t, 'ABX count ', '抗菌药计数 ') + state.abxMinCount + bi(ctx.t, ' in ', ' 次 / ') + state.abxCountWinHours + 'h',
      state.positiveCultures ? bi(ctx.t, 'positive cultures required', '要求阳性培养') : bi(ctx.t, 'any sample accepted', '采样即可'),
      siWindowLabel(ctx.t, state.siWindow),
      bi(ctx.t, 'SOFA window ', 'SOFA 窗口 ') + '−' + state.windowBeforeHours + 'h/+' + state.windowAfterHours + 'h',
      deltaLabel(ctx.t, state.deltaFunction),
      'ΔSOFA ≥ ' + state.threshold,
    ];
    return `
      <div class="sepsis-def-panel">
        <div class="sepsis-def-head">
          <span class="sepsis-def-ico">${ic(ctx.icon, 'shield', 15)}</span>
          <div class="grow">
            <div class="sepsis-def-kicker">${bi(ctx.t, 'Definition checkpoint', '定义检查点')}</div>
            <div class="sepsis-def-title">${bi(ctx.t, 'Sepsis-3 implementation profile', 'Sepsis-3 实现口径')}</div>
            <div class="sepsis-def-copy">${bi(
              ctx.t,
              'These controls mirror easyicu.scores.sepsis.susp_inf() and sep3()/sep3_sofa2(). The default is the Sepsis-3 profile; non-default thresholds or modes are written as sensitivity/strategy choices and passed into the extraction callbacks.',
              '这里的控件对应 easyicu.scores.sepsis.susp_inf() 与 sep3()/sep3_sofa2() 的真实参数。默认是 Sepsis-3 主口径；非默认阈值或模式会作为敏感性/策略选择写入 manifest，并传入抽取 callback。'
            )}</div>
          </div>
          <span class="pill mono">${esc(ctx.escHtml, profile[3])}</span>
        </div>
        <div class="sepsis-def-summary">
          ${current.map(x => chip(ctx, x, 'current')).join('')}
        </div>
        <div class="sepsis-def-grid">
          ${control(ctx, bi(ctx.t, 'Implementation profile', '实现口径'), profileSeg(ctx), bi(ctx.t, 'Switches the SOFA score source used by sep3 or sep3_sofa2.', '切换 sep3 或 sep3_sofa2 使用的 SOFA 评分来源。'))}
          ${control(ctx, bi(ctx.t, 'Suspected infection mode', '疑似感染模式'), optionSeg(ctx, 'si_mode', [
            ['auto', 'Auto', '自动'],
            ['and', 'ABX + sample', '抗菌药 + 采样'],
            ['icd_abx', 'ICD + ABX', 'ICD + 抗菌药'],
            ['abx', 'ABX only', '仅抗菌药'],
            ['samp', 'Sample only', '仅采样'],
            ['or', 'ABX or sample', '抗菌药或采样'],
          ], state.siMode), bi(ctx.t, 'Matches susp_inf(si_mode=...). Auto uses database-specific defaults.', '对应 susp_inf(si_mode=...)；自动模式使用数据库特异默认值。'))}
          ${control(ctx, bi(ctx.t, 'ABX then sample window', '抗菌药后采样窗口'), optionSeg(ctx, 'abx_win_hours', [
            ['12', '12 h', '12 小时'],
            ['24', '24 h', '24 小时'],
            ['48', '48 h', '48 小时'],
          ], state.abxWinHours))}
          ${control(ctx, bi(ctx.t, 'Sample then ABX window', '采样后抗菌药窗口'), optionSeg(ctx, 'samp_win_hours', [
            ['24', '24 h', '24 小时'],
            ['48', '48 h', '48 小时'],
            ['72', '72 h', '72 小时'],
          ], state.sampWinHours))}
          ${control(ctx, bi(ctx.t, 'ABX counting rule', '抗菌药计数规则'), optionSeg(ctx, 'abx_min_count', [
            ['1', '≥1 dose', '≥1 次'],
            ['2', '≥2 doses', '≥2 次'],
            ['3', '≥3 doses', '≥3 次'],
          ], state.abxMinCount) + optionSeg(ctx, 'abx_count_win_hours', [
            ['12', '12 h window', '12 小时窗口'],
            ['24', '24 h window', '24 小时窗口'],
            ['48', '48 h window', '48 小时窗口'],
          ], state.abxCountWinHours), bi(ctx.t, 'Matches abx_min_count and abx_count_win in susp_inf().', '对应 susp_inf() 的 abx_min_count 与 abx_count_win。'))}
          ${control(ctx, bi(ctx.t, 'Culture requirement', '培养要求'), boolSeg(ctx, 'positive_cultures', state.positiveCultures), bi(ctx.t, 'Matches positive_cultures in susp_inf().', '对应 susp_inf() 的 positive_cultures。'))}
          ${control(ctx, bi(ctx.t, 'SI event for SOFA window', 'SOFA 窗口使用的 SI 事件'), optionSeg(ctx, 'si_window', [
            ['first', 'First', '首个'],
            ['last', 'Last', '末次'],
            ['any', 'Any', '任一'],
          ], state.siWindow), bi(ctx.t, 'Matches si_window in sep3()/sep3_sofa2().', '对应 sep3()/sep3_sofa2() 的 si_window。'))}
          ${control(ctx, bi(ctx.t, 'SOFA lookback', 'SOFA 回看窗口'), optionSeg(ctx, 'window_before_hours', [
            ['24', '−24 h', '−24 小时'],
            ['48', '−48 h', '−48 小时'],
            ['72', '−72 h', '−72 小时'],
          ], state.windowBeforeHours))}
          ${control(ctx, bi(ctx.t, 'SOFA follow-up', 'SOFA 后续窗口'), optionSeg(ctx, 'window_after_hours', [
            ['12', '+12 h', '+12 小时'],
            ['24', '+24 h', '+24 小时'],
            ['48', '+48 h', '+48 小时'],
          ], state.windowAfterHours))}
          ${control(ctx, bi(ctx.t, 'SOFA delta function', 'SOFA 增量函数'), optionSeg(ctx, 'delta_function', [
            ['delta_cummin', 'Cumulative min', '累积最小值'],
            ['delta_start', 'Start value', '起始值'],
            ['delta_min', 'Sliding min', '滑动最小值'],
          ], state.deltaFunction))}
          ${control(ctx, bi(ctx.t, 'SOFA threshold', 'SOFA 阈值'), optionSeg(ctx, 'threshold', [
            ['2', 'Δ ≥ 2', 'Δ ≥ 2'],
            ['3', 'Δ ≥ 3', 'Δ ≥ 3'],
          ], state.threshold), bi(ctx.t, 'Δ ≥ 2 is the default Sepsis-3 criterion; Δ ≥ 3 is a sensitivity setting supported by the code.', 'Δ ≥ 2 是默认 Sepsis-3 标准；Δ ≥ 3 是代码支持的敏感性设置。'))}
          ${control(ctx, bi(ctx.t, 'Keep diagnostic components', '保留诊断组件'), boolSeg(ctx, 'keep_components', state.keepComponents), bi(ctx.t, 'Keeps delta_sofa and component times when the callback emits them.', 'callback 输出时保留 delta_sofa 与组件时间。'))}
        </div>
        <div class="sepsis-def-foot">${ic(ctx.icon, 'file', 12)} ${bi(ctx.t, 'Recorded as cohort.sepsis_definition and passed as kwargs to sepsis callbacks during extraction.', '记录到 cohort.sepsis_definition，并在抽取时作为 kwargs 传给 Sepsis callback。')}</div>
      </div>`;
  }

  function bind(root, ctx) {
    root.querySelectorAll('[data-ex-sepsis-profile]').forEach(seg => seg.addEventListener('click', e => {
      const button = e.target.closest('button');
      if (!button) return;
      const val = button.dataset.val || '';
      if (!PROFILES.some(p => p[0] === val)) return;
      state.profile = val;
      if (ctx && typeof ctx.markStale === 'function') ctx.markStale();
      if (ctx && typeof ctx.repaint === 'function') ctx.repaint();
    }));
    root.querySelectorAll('[data-ex-sepsis]').forEach(seg => seg.addEventListener('click', e => {
      const button = e.target.closest('button');
      if (!button) return;
      const key = seg.dataset.exSepsis || '';
      const val = button.dataset.val || '';
      if (key === 'si_mode') state.siMode = val || 'auto';
      else if (key === 'abx_win_hours') state.abxWinHours = Number(val || 24);
      else if (key === 'samp_win_hours') state.sampWinHours = Number(val || 72);
      else if (key === 'abx_count_win_hours') state.abxCountWinHours = Number(val || 24);
      else if (key === 'abx_min_count') state.abxMinCount = Number(val || 1);
      else if (key === 'positive_cultures') state.positiveCultures = val === 'true';
      else if (key === 'si_window') state.siWindow = val || 'first';
      else if (key === 'window_before_hours') state.windowBeforeHours = Number(val || 48);
      else if (key === 'window_after_hours') state.windowAfterHours = Number(val || 24);
      else if (key === 'delta_function') state.deltaFunction = val || 'delta_cummin';
      else if (key === 'threshold') state.threshold = Number(val || 2);
      else if (key === 'keep_components') state.keepComponents = val === 'true';
      if (ctx && typeof ctx.markStale === 'function') ctx.markStale();
      if (ctx && typeof ctx.repaint === 'function') ctx.repaint();
    }));
  }

  window.EUExtractionSepsis = {
    state,
    profiles: PROFILES,
    relevant,
    contract,
    panel,
    bind,
  };
})();
