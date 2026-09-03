/* Cohort Statistics domain owner: state, effects, rendering orchestration, and route events. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  const state = {
    view: 'idle',
    panel: 'groups',
    compare: 'outcome',
    featureScope: 'recommended',
    featureModule: 'all',
    selectedFeatures: [],
    survivalOutcome: 'mort_28d',
    survivalGroup: 'sepsis',
    sofaMatrixMode: 'pct',
    sofaMatrixGranularity: 'exact',
    error: null,
  };
  let view;
  let cohortCharts;
  let t;
  let icon;
  let esc;
  let fmtInt;
  let fmtNum;
  let fmtPct;
  let fmtP;
  let displayDataMode;
  let officialDemoContext;
  let registryActivePath;
  let bindSourceRegistry;
  let sourceModeSelector;
  let sourceRegistryBlock;
  let repaintScreen;
  let vizRail;
  let workspaceSamplingNote;

  function cohortReview() {
    return window.EU_COHORT_REVIEW || null;
  }
  function cohortLoaded() {
    const review = cohortReview();
    return state.view === 'loaded' && (window.EU_DATA !== 'real' || !!(review && review.summary));
  }
  function reloadStaleRealCohortIfNeeded(review) {
    if (window.EU_DATA !== 'real' || state.view !== 'loaded' || (review && review.summary)) return false;
    if (!registryActivePath()) {
      state.view = 'idle';
      return false;
    }
    state.view = 'loading';
    setTimeout(() => loadRealCohort(ok => { state.view = ok ? 'loaded' : 'idle'; repaintScreen('cohort'); }), 0);
    return true;
  }
  function resetCohortFeatureSelection() {
    state.featureModule = 'all';
    state.selectedFeatures = [];
  }
  function cohortSelectedFeatureIds(review) {
    const selected = ((review || cohortReview() || {}).feature_selection || {}).selected || [];
    return selected.map(row => row && row.id).filter(Boolean);
  }
  function syncCohortFeatureSelection(payload) {
    const ids = cohortSelectedFeatureIds(payload);
    if (ids.length) state.selectedFeatures = ids;
  }
  function reloadCohortForFeatureSelection() {
    if (window.EU_DATA !== 'real') {
      repaintScreen('cohort');
      return;
    }
    state.view = 'loading';
    repaintScreen('cohort');
    loadRealCohort(ok => {
      state.view = ok ? 'loaded' : 'idle';
      repaintScreen('cohort');
    });
  }
  function cohortWorkspaceFromReview(payload) {
    const s = payload && payload.summary ? payload.summary : {};
    return {
      ok: true,
      route: 'cohort',
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
  function cohortMissingExportMessage() {
    return t('Choose or add a local EasyICU export before loading Cohort Statistics.', '请先选择或添加本地 EasyICU 导出，再加载队列统计。');
  }
  function loadRealCohort(done) {
    if (!(window.EU_API && window.EU_API.loadCohortReviewSummary)) {
      state.error = 'Cohort Review API is unavailable.';
      done && done(false);
      return;
    }
    const active = registryActivePath();
    if (!active) {
      window.EU_COHORT_REVIEW = null;
      window.EU_VIZ_WORKSPACE = null;
      state.error = cohortMissingExportMessage();
      done && done(false);
      return;
    }
    const body = { source_path: active };
    if (state.selectedFeatures.length) body.selected_features = state.selectedFeatures.slice();
    window.EU_API.loadCohortReviewSummary(body).then(payload => {
      window.EU_COHORT_REVIEW = payload;
      syncCohortFeatureSelection(payload);
      window.EU_VIZ_WORKSPACE = cohortWorkspaceFromReview(payload);
      state.error = null;
      window.EU_HASWORK = true;
      done && done(true);
    }).catch(err => {
      window.EU_COHORT_REVIEW = null;
      window.EU_VIZ_WORKSPACE = null;
      state.error = String(err && err.message || err);
      done && done(false);
    });
  }
  S.cohort = {
    section: 'viz', nav: 'viz', sub: 'cohort',
    crumbs: ['Home', 'Data Workspace','Cohort Statistics'],
    get actionHtml() {
      // Topbar actions only exist once the review is loaded — the setup card in
      // the body owns the single primary action before that.
      if (cohortLoaded()) {
        return `<button class="btn" data-viz-reset>${icon('sliders', 13)} ${t('Edit setup', '编辑设置')}</button><button class="btn primary" data-cohort-run>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>`;
      }
      return '';
    },
    rail: () => vizRail('cohort'),
    afterRender(root) {
      if (cohortCharts && typeof cohortCharts.mount === 'function') cohortCharts.mount(root);
      const realMode = displayDataMode() === 'real';
      bindSourceRegistry(root, 'cohort');
      const demoSourceOwner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
      if (!realMode && state.view === 'idle' && demoSourceOwner) {
        demoSourceOwner.ensureLoaded(() => {
          if (window.location.hash === '#cohort' && state.view === 'idle' && displayDataMode() !== 'real') {
            repaintScreen('cohort');
          }
        });
        demoSourceOwner.bind(root, {
          refresh: () => {
            if (window.location.hash === '#cohort' && state.view === 'idle') repaintScreen('cohort');
          },
          openPrepared: sourceId => {
            if (state.view === 'loading') return;
            const source = demoSourceOwner.rememberOpened && demoSourceOwner.rememberOpened(sourceId);
            if (!source || !source.status || !source.status.active) {
              state.error = t(
                'The selected official demo is not active yet. Activate it before opening.',
                '所选官方演示尚未激活，请先激活后再打开。',
              );
              repaintScreen('cohort');
              return;
            }
            state.view = 'loading';
            window.EU_DATA = 'real';
            window.EU_COHORT_REVIEW = null;
            window.EU_VIZ_WORKSPACE = null;
            resetCohortFeatureSelection();
            state.error = null;
            repaintScreen('cohort');
            const hydration = window.EU_API && window.EU_API.hydrateWorkspaceRegistry
              ? window.EU_API.hydrateWorkspaceRegistry()
              : Promise.resolve();
            Promise.resolve(hydration).then(() => {
              loadRealCohort(ok => {
                state.view = ok ? 'loaded' : 'idle';
                repaintScreen('cohort');
              });
            }).catch(error => {
              state.error = String((error && error.message) || error);
              state.view = 'idle';
              repaintScreen('cohort');
            });
          },
        });
      }
      root.querySelectorAll('[data-cohort-run]').forEach(b => b.addEventListener('click', () => {
        if (state.view === 'loading') return;
        if (window.EU_DATA === 'real') {
          if (!registryActivePath()) {
            state.view = 'idle';
            window.EU_COHORT_REVIEW = null;
            window.EU_VIZ_WORKSPACE = null;
            state.error = cohortMissingExportMessage();
            repaintScreen('cohort');
            return;
          }
          state.view = 'loading'; repaintScreen('cohort');
          loadRealCohort(ok => { state.view = ok ? 'loaded' : 'idle'; repaintScreen('cohort'); });
        } else {
          state.error = null;
          state.view = 'loading'; repaintScreen('cohort');
          setTimeout(() => { state.view = 'loaded'; window.EU_HASWORK = true; repaintScreen('cohort'); }, 1300);
        }
      }));
      root.querySelectorAll('[data-cohort-demo-fallback]').forEach(b => b.addEventListener('click', () => {
        if (state.view === 'loading') return;
        if (window.setDataModeContext) window.setDataModeContext(null);
        window.EU_DATA = 'demo';
        window.EU_COHORT_REVIEW = null;
        window.EU_VIZ_WORKSPACE = null;
        resetCohortFeatureSelection();
        state.error = null;
        state.view = 'loading';
        repaintScreen('cohort');
        setTimeout(() => {
          state.view = 'loaded';
          window.EU_HASWORK = true;
          repaintScreen('cohort');
        }, 450);
      }));
      const tabsEl = root.querySelector('#cohtabs');
      if (tabsEl) tabsEl.addEventListener('click', e => {
        const b = e.target.closest('[data-cohtab]'); if (!b) return;
        if (b.dataset.cohtab === state.panel) return;
        state.panel = b.dataset.cohtab;
        repaintScreen('cohort');
      });
      root.querySelectorAll('[data-cohgo]').forEach(b => b.addEventListener('click', () => {
        state.panel = b.dataset.cohgo;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-viz-reset]').forEach(b => b.addEventListener('click', () => {
        state.view = 'idle';
        state.featureScope = 'recommended';
        window.EU_COHORT_REVIEW = null;
        window.EU_VIZ_WORKSPACE = null;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-use-real]').forEach(b => b.addEventListener('click', () => {
        state.view = 'idle';
        if (window.setDataModeContext) window.setDataModeContext(null);
        window.EU_DATA = 'real';
        window.EU_COHORT_REVIEW = null;
        window.EU_VIZ_WORKSPACE = null;
        state.featureScope = 'recommended';
        try { localStorage.setItem('easyicu_home_data', 'real'); } catch (e) {}
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-scope]').forEach(b => b.addEventListener('click', () => {
        const next = b.dataset.state.featureScope === 'all' ? 'all' : 'recommended';
        if (next === state.featureScope) return;
        state.featureScope = next;
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-comp]').forEach(b => {
        const choose = () => {
          if (b.dataset.cohortComp === state.compare) return;
          state.compare = b.dataset.cohortComp || 'outcome';
          window.EU_STALE = true;
          repaintScreen('cohort');
        };
        b.addEventListener('click', choose);
        b.addEventListener('keydown', e => {
          if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            choose();
          }
        });
      });
      root.querySelectorAll('[data-cohort-surv-group]').forEach(b => b.addEventListener('click', () => {
        if (b.dataset.cohortSurvGroup === state.survivalGroup) return;
        state.survivalGroup = b.dataset.cohortSurvGroup || 'sepsis';
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-sofa-matrix-mode]').forEach(b => b.addEventListener('click', () => {
        const next = b.dataset.state.sofaMatrixMode === 'count' ? 'count' : 'pct';
        if (next === state.sofaMatrixMode) return;
        state.sofaMatrixMode = next;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-sofa-granularity]').forEach(b => b.addEventListener('click', () => {
        const next = b.dataset.cohortSofaGranularity || 'medium';
        if (!view.hasSofaGranularity(next) || next === state.sofaMatrixGranularity) return;
        state.sofaMatrixGranularity = next;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-module]').forEach(b => b.addEventListener('click', () => {
        state.featureModule = b.dataset.state.featureModule || 'all';
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-toggle]').forEach(b => b.addEventListener('click', () => {
        const id = b.dataset.cohortFeatureToggle;
        if (!id || b.getAttribute('aria-disabled') === 'true') return;
        const selected = new Set(state.selectedFeatures.length ? state.selectedFeatures : cohortSelectedFeatureIds(cohortReview()));
        if (selected.has(id)) selected.delete(id);
        else selected.add(id);
        state.selectedFeatures = Array.from(selected);
        window.EU_STALE = true;
        reloadCohortForFeatureSelection();
      }));
      root.querySelectorAll('[data-cohort-feature-default]').forEach(b => b.addEventListener('click', () => {
        resetCohortFeatureSelection();
        window.EU_STALE = true;
        reloadCohortForFeatureSelection();
      }));
      root.querySelectorAll('[data-cohort-feature-clear]').forEach(b => b.addEventListener('click', () => {
        state.selectedFeatures = [];
        const review = cohortReview();
        if (review && review.feature_selection) review.feature_selection.selected = [];
        if (review && review.groups && review.groups.supported) {
          review.groups.supported.forEach(row => {
            if (row.profile && Array.isArray(row.profile.rows)) {
              row.profile.rows = row.profile.rows.filter(metric => !metric.feature_id);
            }
          });
        }
        window.EU_STALE = true;
        repaintScreen('cohort');
      }));
    },
    render() {
      if (window.__euCohortPanel) { state.panel = window.__euCohortPanel; window.__euCohortPanel = null; }
      // Consume a Guided Copilot handoff so a study configured in conversation
      // does not silently vanish when Copilot lands the user here.
      if (window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.take) window.EU_GUIDED_HANDOFF.take('cohort');
      const guidedNote = window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.noteHtml ? window.EU_GUIDED_HANDOFF.noteHtml('cohort') : '';
      const ws = window.EU_VIZ_WORKSPACE;
      const dataMode = displayDataMode();
      const realMode = dataMode === 'real';
      const officialDemo = officialDemoContext();
      let review = cohortReview();
      if (reloadStaleRealCohortIfNeeded(review)) review = null;
      const loaded = cohortLoaded();
      const head = `${guidedNote}
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>${view.text('Workspace')}</span> ${icon('chevron', 11)} <span>${loaded ? (realMode ? view.text('Local export') : (officialDemo ? t('Official demo', '官方演示') : view.text('Demo cohort'))) : view.text('Not configured')}</span> ${icon('chevron', 11)} <span style="color:var(--ink-2);">${view.text('Cohort statistics')}</span>
      </div>
      <div class="page-head" style="margin-bottom:16px;">
        <h1 style="margin-top:0;">${t('Cohort Statistics', '队列统计')}</h1>
        <p class="lead">${loaded ? (review ? (realMode ? t('Local export cohort · real exported module tables · local-only summary', '本地导出队列 · 真实导出模块表 · 仅本地汇总') : t('Official deidentified demo cohort · source-backed aggregate review', '官方去标识化演示队列 · 有来源支撑的聚合审阅')) : t('Synthetic fallback contrast · UI rehearsal only', '合成兜底对照 · 仅用于界面演练')) : t('Choose one official demo or one registered export before viewing group contrasts, coverage, survival curves, and SOFA reclassification.', '选择一个官方演示或一个已注册导出，再查看组间对照、覆盖率、生存曲线和 SOFA 重分层。')}</p>
        <div style="font-size:11.5px;color:var(--ink-4);margin-top:9px;">${t('Key terms', '关键术语')}: ${window.gloss('cohort', t('cohort', '队列'))} · ${window.gloss('denominator', t('denominator', '分母'))} · ${window.gloss('SOFA')} · ${window.gloss('Sepsis-3')}</div>
      </div>`;
      if (state.view !== 'loading' && !loaded) {
        const demoSourceOwner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
        const officialDemoSources = demoSourceOwner && typeof demoSourceOwner.render === 'function'
          ? demoSourceOwner.render(
            { t, esc },
            { scope: 'cohort', fallbackAttribute: 'data-cohort-demo-fallback' },
          )
          : `<div class="note warn"><div class="body"><div class="d">${t('Official demo-source controls are unavailable.', '官方演示数据源控件暂不可用。')}</div></div></div>`;
        return head + `<div class="card pad" data-cohort-config-required="true">
            <div class="panel-head">
              <div>
                <div class="eyebrow">${t('Cohort Statistics', '队列统计')}</div>
                <div class="panel-title" style="font-size:17px;">${t('Choose one cohort data source', '选择一个队列数据源')}</div>
                <div class="panel-sub mt-4">${realMode ? t('Use one registered local EasyICU export.', '使用一个已注册的本地 EasyICU 导出。') : t('Use one official deidentified ICU demo; synthetic data remains an explicit offline fallback.', '使用一个官方去标识化 ICU 演示；合成数据仅作为明确的离线兜底。')}</div>
              </div>
              <span class="pill ${realMode ? '' : 'demo'}"><span class="dot"></span>${realMode ? t('1 source required', '需要 1 个来源') : t('Official demos', '官方演示')}</span>
            </div>
            <div style="border-top:1px solid var(--hair);padding-top:16px;">
              ${sourceModeSelector(realMode)}
            </div>
            ${state.error ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(state.error)}</div></div></div>` : ''}
            ${realMode ? `<div class="card sunken pad mt-16">
              <div class="eyebrow">${t('Local export', '本地导出')}</div>
              <div class="panel-title mt-4">${t('Load one exported cohort snapshot', '加载一个已导出的队列快照')}</div>
              <div class="panel-sub mt-4">${t('This is the same active export contract used by Patient Review.', '这里使用与患者审阅相同的 active 导出契约。')}</div>
              ${!registryActivePath() ? `<div class="note info mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('No active export selected', '尚未选择 active 导出')}</div><div class="d">${cohortMissingExportMessage()}</div></div></div>` : ''}
              ${sourceRegistryBlock('single')}
              <button class="btn primary mt-16" data-cohort-run ${!registryActivePath() ? 'aria-disabled="true"' : ''}>${icon('folder', 14)} ${t('Load local export', '加载本地导出')}</button>
            </div>` : `<div class="card sunken pad mt-16">${officialDemoSources}</div>`}
          </div>
          <div class="empty mt-16" data-cohort-empty-preview="true">
            <div class="glyph">${icon('cohort', 22)}</div>
            <div class="t">${t('Cohort review awaits setup', '队列审阅等待配置')}</div>
            <div class="d">${t('After setup, this page will show group contrast, KM/log-rank, coverage audit, cohort profile, and SOFA reclassification.', '配置后，这里会显示组间对照、KM/log-rank、生存风险表、覆盖审计、队列画像和 SOFA 重分层。')}</div>
          </div>`;
      }
      if (state.view === 'loading') {
        return head + `<div class="card pad">
          <div class="load-strip">
            <span class="spin accent"></span>
            <div class="grow"><div style="font-weight:600;font-size:12.75px;">${t('Recomputing cohort statistics…', '正在重新计算队列统计…')}</div><div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${t('reproducible · no outbound calls', '可复现 · 无外部调用')}</div></div>
          </div>
          <div class="indet mt-12"></div>
          <div class="st-stats mt-16">${[0,1,2,3].map(() => `<div class="sk-stat"><div class="sk sk-line sm" style="width:52%"></div><div class="sk" style="height:22px;width:64%;margin-top:10px;"></div></div>`).join('')}</div>
          <div class="sk-table mt-16">${[0,1,2,3,4].map(() => `<div class="sk-trow">${[60,40,40,40,30].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}</div>
        </div>`;
      }
      const demoScope = view.demoCatalogScope();
      const preflightItems = [
        [
          view.text('Input package'),
          ws ? `${fmtInt(ws.summary && ws.summary.stays)} ${t('entities', '个实体')} · ${fmtInt(ws.summary && ws.summary.modules)} ${t('modules', '个模块')}` : `10 ${t('stays', '次住院')} · ${fmtInt(demoScope.selectedModuleCount)} / ${fmtInt(demoScope.totalModuleCount)} ${t('modules', '个模块')} · ${fmtInt(demoScope.selectedFeatureCount)} / ${fmtInt(demoScope.totalFeatureCount)} ${t('features', '个特征')}`,
          'ok',
          null,
        ],
        [
          view.text('Backend evidence checks'),
          review ? view.text('manifest parsed · denominators previewed · aggregate payload returned') : view.text('coverage + denominators ready'),
          'ok',
          null,
        ],
        [
          view.text('Draft review'),
          view.text('locked · requires reviewer sign-off'),
          'warn',
          'agent',
        ],
      ];
      return head + `
      <div class="card" style="padding:0;overflow:hidden;">
        <div class="row" style="justify-content:space-between;padding:11px 16px;border-bottom:1px solid var(--hair);">
          <span style="font-weight:600;font-size:12.5px;">${view.text('Analysis readiness')}</span>
          <span class="mono" style="font-size:11px;color:var(--ink-4);">${view.text('current session')}</span>
        </div>
        <div class="preflight">
          ${preflightItems.map(([tt, d, s, nav]) => `
            <div class="pf-cell" ${nav ? `data-study-handoff data-study-source="cohort" data-study-target="${nav}" role="button" tabindex="0" style="cursor:pointer;"` : ''}>
              <div class="eyebrow" style="display:flex;align-items:center;gap:6px;">
                <span class="dot-${s}"></span>${tt}${nav ? `<span style="margin-left:auto;color:var(--ink-4);">${icon('arrow', 12)}</span>` : ''}
              </div>
              <div style="font-size:12.5px;color:var(--ink-2);margin-top:6px;">${d}</div>
              ${nav ? `<div style="font-size:11px;color:var(--ink-4);margin-top:4px;">${review ? t('Aggregate payload is ready; open Project Monitor for evidence-bound draft review.', '聚合载荷已就绪；打开项目监控做证据绑定草稿核验。') : t('Demo review is local-only; use Guided Copilot after choosing a real export.', '演示审阅仅限本地预览；选择真实导出后再使用研究引导。')}</div>` : ''}
            </div>`).join('')}
        </div>
      </div>

      ${view.tabs()}
      <div id="cohbody">${view.panelBody()}</div>
      <div class="nextbar accent mt-16">
        <div class="nb-ico">${icon('arrow', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Compared the groups — what’s next?', '对比完组间差异 —— 下一步？')}</div><div class="nb-d">${t('Ask Guided Copilot to assemble an auditable analysis and review-ready draft, or benchmark the cohort across databases.', '让「研究引导」组装可审计分析与待核验草稿，或跨数据库对比队列。')}</div></div>
        <button class="btn" data-nav="crossdb">${icon('benchmark', 13)} ${t('Cross-database comparison', '跨库对比')}</button>
        <button class="btn primary" data-study-handoff data-study-source="cohort" data-study-target="guided">${icon('agent', 13)} ${t('Continue in Guided Copilot','在研究引导中继续')}</button>
      </div>`;
    },
  };

  /* ---------------- CROSS-DB BENCHMARK ---------------- */
  function init(config) {
    t = config.t;
    icon = config.icon;
    esc = config.esc;
    fmtInt = config.fmtInt;
    fmtNum = config.fmtNum;
    fmtPct = config.fmtPct;
    fmtP = config.fmtP;
    displayDataMode = config.displayDataMode;
    officialDemoContext = config.officialDemoContext;
    registryActivePath = config.registryActivePath;
    bindSourceRegistry = config.bindSourceRegistry;
    sourceModeSelector = config.sourceModeSelector;
    sourceRegistryBlock = config.sourceRegistryBlock;
    repaintScreen = config.repaintScreen;
    vizRail = config.vizRail;
    workspaceSamplingNote = config.workspaceSamplingNote;
    cohortCharts = config.cohortCharts;
    view = window.EU_VIZ_COHORT_VIEW;
    view.init({
      state, t, icon, esc, fmtInt, fmtNum, fmtPct, fmtP,
      workspaceSamplingNote, cohortCharts, cohortReview,
    });
  }

  function resetForSourceChange() {
    window.EU_COHORT_REVIEW = null;
    resetCohortFeatureSelection();
    state.view = 'idle';
    state.error = null;
  }

  function reloadAfterSourceChange() {
    state.view = 'loading';
    repaintScreen('cohort');
    loadRealCohort(ok => {
      state.view = ok ? 'loaded' : 'idle';
      repaintScreen('cohort');
    });
  }

  function hydrate(payload) {
    window.EU_DATA = 'real';
    state.view = 'loaded';
    window.EU_COHORT_REVIEW = payload;
    syncCohortFeatureSelection(payload);
    window.EU_VIZ_WORKSPACE = cohortWorkspaceFromReview(payload);
    state.error = null;
    window.EU_HASWORK = true;
  }

  window.EU_VIZ_COHORT = {
    init,
    review: cohortReview,
    loaded: cohortLoaded,
    demoCatalogScope() { return view.demoCatalogScope(); },
    resetForSourceChange,
    reloadAfterSourceChange,
    setError(value) { state.error = value == null ? null : String(value); },
    error() { return state.error; },
    snapshot() { return { ...state, selectedFeatures: state.selectedFeatures.slice() }; },
    hydrate,
    comparison() { return state.compare; },
    outcome() { return state.survivalOutcome; },
    beginCharts() { return cohortCharts && cohortCharts.begin(); },
    panels: {
      groups: (...args) => view.groupsBody(...args),
      profile: (...args) => view.snapshotBody(...args),
      coverage: (...args) => view.coverageBody(...args),
      survival: (...args) => view.survivalBody(...args),
      sofa: (...args) => view.sofaBody(...args),
    },
    mountCharts(root) {
      return cohortCharts && typeof cohortCharts.mount === 'function'
        ? cohortCharts.mount(root)
        : 0;
    },
  };
})();

