/* Extraction Sepsis-3 definition panel.
   Owner: Data Extraction route. The Web UI exposes only definition-safe
   choices; lower-level callback kwargs remain an implementation detail. */
(function () {
  const RUNTIME_PROFILE = 'easyicu_sepsis3_locked_v1';
  const IMPLEMENTATION_PROFILE = 'selected_module_defaults';
  const SCORE_FAMILY = 'module-specific SOFA source';
  const LOCKED = {
    abxWinHours: 24,
    sampWinHours: 72,
    abxCountWinHours: 24,
    abxMinCount: 1,
    positiveCultures: false,
    windowBeforeHours: 48,
    windowAfterHours: 24,
    deltaFunction: 'delta_cummin',
    threshold: 2,
    keepComponents: false,
  };
  const state = {
    siMode: 'auto',
    siWindow: 'first',
    detailsOpen: false,
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

  function relevant(moduleKeys) {
    return (moduleKeys || []).some(k => k.startsWith('sepsis3_') || k.startsWith('sofa') || k === 'sepsis_shared');
  }

  function contract() {
    return {
      record_scope: 'metadata_current_runtime_defaults',
      runtime_profile: RUNTIME_PROFILE,
      implementation_profile: IMPLEMENTATION_PROFILE,
      score_family: SCORE_FAMILY,
      definition_locked: true,
      suspected_infection: {
        mode: state.siMode,
        abx_win_hours: LOCKED.abxWinHours,
        samp_win_hours: LOCKED.sampWinHours,
        abx_count_win_hours: LOCKED.abxCountWinHours,
        abx_min_count: LOCKED.abxMinCount,
        positive_cultures_required: LOCKED.positiveCultures,
      },
      sofa_increase: {
        si_window: state.siWindow,
        window_before_si_hours: LOCKED.windowBeforeHours,
        window_after_si_hours: LOCKED.windowAfterHours,
        delta_function: LOCKED.deltaFunction,
        threshold: LOCKED.threshold,
        keep_components: LOCKED.keepComponents,
      },
      review_options: {
        si_window: ['first', 'any'],
      },
      locked_core: {
        suspected_infection_windows: 'ABX->sample 24h; sample->ABX 72h',
        sofa_window: '-48h/+24h',
        delta_rule: 'cumulative minimum within SI window',
        sofa_threshold: 'delta >= 2',
      },
    };
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
      auto: bi(t, 'database implementation', '当前数据库口径'),
      and: bi(t, 'ABX + sample', '抗菌药 + 采样'),
      icd_abx: bi(t, 'infection ICD + ABX', '感染 ICD + 抗菌药'),
    };
    return map[value] || value;
  }

  function databaseKey(ctx) {
    return String((ctx && ctx.database) || '').trim().toLowerCase();
  }

  function isEicu(ctx) {
    return databaseKey(ctx).includes('eicu');
  }

  function effectiveSiLabel(t, ctx) {
    return isEicu(ctx)
      ? bi(t, 'eICU fallback: infection ICD + ABX', 'eICU 兜底：感染 ICD + 抗菌药')
      : bi(t, 'ABX + sample', '抗菌药 + 采样');
  }

  function effectiveSiHelp(t, ctx) {
    return isEicu(ctx)
      ? bi(
        t,
        'eICU lacks a harmonized culture/sample timing chain, so EasyICU uses the documented infection-ICD plus antimicrobial fallback for eICU only.',
        'eICU 无法统一到同一套培养/采样时间链，因此 EasyICU 仅在 eICU 使用“感染 ICD + 抗菌药”兜底。'
      )
      : bi(
        t,
        'For MIMIC-style sources this is fixed to antimicrobial plus sample timing; there is no separate "auto" choice for users to tune.',
        '对 MIMIC 这类数据源，疑似感染锚点固定为“抗菌药 + 采样”；这里不再提供额外的“自动”选择。'
      );
  }

  function siWindowLabel(t, value) {
    const map = {
      first: bi(t, 'first SI event', '首个 SI 事件'),
      last: bi(t, 'last SI event', '末次 SI 事件'),
      any: bi(t, 'any SI event', '任一 SI 事件'),
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

  function refreshPanel(root, ctx) {
    root.querySelectorAll('[data-ex-sepsis]').forEach(seg => {
      const key = seg.dataset.exSepsis || '';
      const value = key === 'si_mode' ? state.siMode : state.siWindow;
      seg.querySelectorAll('button[data-val]').forEach(button => {
        button.classList.toggle('active', String(button.dataset.val || '') === String(value));
      });
    });
    const meta = root.querySelector('.sepsis-def-audit-meta');
    if (meta) {
      meta.innerHTML = [
        effectiveSiLabel(ctx && ctx.t, ctx),
        siWindowLabel(ctx && ctx.t, state.siWindow),
      ].map(x => chip(ctx || {}, x, 'current')).join('');
    }
  }

  function panel(ctx) {
    if (!relevant(ctx.moduleKeys || [])) return '';
    const lockedSummary = [
      bi(ctx.t, 'suspected infection', '疑似感染'),
      'ΔSOFA ≥ 2',
      bi(ctx.t, 'standard SI window', '标准 SI 窗口'),
    ];
    const current = [
      effectiveSiLabel(ctx.t, ctx),
      siWindowLabel(ctx.t, state.siWindow),
    ];
    const lockedCore = [
      bi(ctx.t, 'ABX then sample: 24h', '抗菌药后采样：24 小时'),
      bi(ctx.t, 'Sample then ABX: 72h', '采样后抗菌药：72 小时'),
      bi(ctx.t, 'ABX count: ≥1 in 24h', '抗菌药计数：24 小时内 ≥1 次'),
      bi(ctx.t, 'No positive-culture override', '不开放阳性培养覆盖'),
      bi(ctx.t, 'SOFA delta: cumulative minimum', 'SOFA 增量：窗口内累积最小值'),
      bi(ctx.t, 'Threshold fixed: ΔSOFA ≥ 2', '阈值固定：ΔSOFA ≥ 2'),
    ];
    if (isEicu(ctx)) {
      lockedCore.splice(3, 0, bi(ctx.t, 'eICU-only fallback: infection ICD + ABX', '仅 eICU 兜底：感染 ICD + 抗菌药'));
    }
    return `
      <div class="sepsis-def-panel">
        <div class="sepsis-def-head">
          <span class="sepsis-def-ico">${ic(ctx.icon, 'shield', 15)}</span>
          <div class="grow">
            <div class="sepsis-def-kicker">${bi(ctx.t, 'Definition checkpoint', '定义检查点')}</div>
            <div class="sepsis-def-title">${bi(ctx.t, 'Sepsis-3 definition locked', 'Sepsis-3 口径已锁定')}</div>
            <div class="sepsis-def-copy">${bi(
              ctx.t,
              'Used only to document the fixed implementation used for extraction; this is not part of the normal setup flow.',
              '这里只记录抽取时使用的固定实现口径；普通配置流程不需要展开。'
            )}</div>
          </div>
          <span class="pill mono">${bi(ctx.t, 'locked', '已锁定')}</span>
        </div>
        <div class="sepsis-def-audit-strip">
          <div>
            <div class="sepsis-def-audit-label">${bi(ctx.t, 'Core rule', '核心规则')}</div>
            <div class="sepsis-def-audit-rule">${lockedSummary.map(x => esc(ctx.escHtml, x)).join(' + ')}</div>
          </div>
          <div class="sepsis-def-audit-meta">
            ${current.map(x => chip(ctx, x, 'current')).join('')}
          </div>
        </div>
        <details class="sepsis-def-details sepsis-def-details-lite" ${state.detailsOpen ? 'open' : ''}>
            <summary>${ic(ctx.icon, 'sliders', 13)} ${bi(ctx.t, 'Advanced audit details', '高级审计细节')}</summary>
            <div class="sepsis-def-detail-body">
              <div class="sepsis-def-detail-title">${bi(ctx.t, 'Locked implementation constants', '锁定实现常量')}</div>
              <div class="sepsis-def-chips">${lockedCore.map(x => chip(ctx, x, 'review')).join('')}</div>
              <div class="sepsis-def-detail-title">${bi(ctx.t, 'Audit anchors', '审计锚点')}</div>
              <div class="sepsis-def-grid compact">
              ${control(ctx, bi(ctx.t, 'Suspected infection anchor', '疑似感染锚点'), `<div class="sepsis-def-static">${esc(ctx.escHtml, effectiveSiLabel(ctx.t, ctx))}</div>`, effectiveSiHelp(ctx.t, ctx))}
              ${control(ctx, bi(ctx.t, 'SI event', 'SI 事件'), optionSeg(ctx, 'si_window', [
                ['first', 'First', '首个'],
                ['any', 'Any', '任一'],
              ], state.siWindow), bi(ctx.t, 'The SOFA window and threshold stay fixed; this only chooses how repeated SI events are anchored.', 'SOFA 窗口和阈值保持固定；这里只选择多次 SI 事件时如何锚定。'))}
              </div>
              <div class="sepsis-def-help">${bi(ctx.t, 'Advanced callback kwargs are intentionally not exposed here because they would create non-standard sensitivity definitions.', '高级 callback 参数不在这里暴露，避免误生成非标准敏感性定义。')}</div>
            </div>
        </details>
        <div class="sepsis-def-foot">${ic(ctx.icon, 'file', 12)} ${bi(ctx.t, 'Recorded as cohort.sepsis_definition with locked core defaults and the small set of allowed implementation choices.', '记录到 cohort.sepsis_definition；核心定义固定，只保存少数允许的实现口径选择。')}</div>
      </div>`;
  }

  function bind(root, ctx) {
    root.querySelectorAll('.sepsis-def-details').forEach(details => details.addEventListener('toggle', () => {
      state.detailsOpen = !!details.open;
    }));
    root.querySelectorAll('[data-ex-sepsis]').forEach(seg => seg.addEventListener('click', e => {
      const button = e.target.closest('button');
      if (!button) return;
      const details = button.closest('.sepsis-def-details');
      if (details) state.detailsOpen = !!details.open;
      const key = seg.dataset.exSepsis || '';
      const val = button.dataset.val || '';
      if (key === 'si_window' && ['first', 'any'].includes(val)) state.siWindow = val || 'first';
      if (ctx && typeof ctx.markStale === 'function') ctx.markStale();
      refreshPanel(root, ctx);
    }));
  }

  window.EUExtractionSepsis = {
    state,
    relevant,
    contract,
    panel,
    bind,
  };
})();
