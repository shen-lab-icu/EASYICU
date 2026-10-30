/* Owner: Cohort Statistics route. */
/* Cohort Statistics domain owner: state, effects, rendering orchestration, and route events. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  // A hosted instance names the administrator's prepared data, not a local export.
  const hostedServer = () => ((window.EU_CAPABILITIES || {}).deployment || {}).hosted === true;
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
    customGroup: null,
    customDraft: { kind: 'numeric_threshold' },
    customBusy: false,
    customError: null,
  };
  let requestSequence = 0;
  let view;
  let cohortCharts;
  let t;
  let icon;
  let esc;
  let fmtInt;
  let fmtNum;
  let fmtPct;
  let fmtP;
  let officialDemoContext;
  let registryActivePath;
  let bindSourceRegistry;
  let officialDemoOffers;
  let activeSourceKind;
  let sourceRegistryBlock;
  let repaintScreen;
  let vizRail;
  let workspaceSamplingNote;

  function cohortReview() {
    return window.EU_COHORT_REVIEW || null;
  }
  function cohortLoaded() {
    const review = cohortReview();
    return state.view === 'loaded' && !!(review && review.summary);
  }
  function reloadStaleRealCohortIfNeeded(review) {
    if (state.view !== 'loaded' || (review && review.summary)) return false;
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
  function customGroupReceiptMatches(payload, group) {
    if (!payload || !payload.custom_group_options || payload.custom_group_options.version !== '1') return false;
    const receipt = payload.custom_group_applied;
    const custom = (((payload.groups || {}).supported) || []).find(row => row.id === 'custom');
    if (!group) return receipt === null && !custom;
    if (!receipt || receipt.kind !== group.kind || receipt.feature_id !== group.feature_id) return false;
    if (!custom || custom.kind !== group.kind || custom.feature_id !== group.feature_id) return false;
    if (group.kind === 'numeric_threshold') return typeof receipt.threshold === 'number' && Number.isFinite(receipt.threshold) && receipt.threshold === group.threshold;
    if (group.kind === 'categorical_values') return Array.isArray(receipt.values) && receipt.values.length === 2 && receipt.values.every((value, i) => value === group.values[i]);
    return group.kind === 'sex';
  }
  function reloadCohortForFeatureSelection() {
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
    return hostedServer() ? t('Choose one of the prepared datasets before loading Cohort Statistics.', '请先选择一份已准备的数据，再加载队列统计。')
      : t('Choose prepared data, or prepare a public demo below, before loading Cohort Statistics.', '请先选择一份已准备的数据，或准备下方的公开 Demo，再加载队列统计。');
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
    const sequence = ++requestSequence;
    const body = { source_path: active };
    if (state.selectedFeatures.length) body.selected_features = state.selectedFeatures.slice();
    if (state.customGroup) body.custom_group = { ...state.customGroup };
    window.EU_API.loadCohortReviewSummary(body).then(payload => {
      if (sequence !== requestSequence || registryActivePath() !== active) return;
      if (state.customGroup && !customGroupReceiptMatches(payload, state.customGroup)) {
        throw new Error(t('The service did not confirm the requested grouping. Restart or update the service before using custom grouping.', '服务未确认请求的自定义分组，请重启或更新服务后重试。'));
      }
      window.EU_COHORT_REVIEW = payload;
      syncCohortFeatureSelection(payload);
      window.EU_VIZ_WORKSPACE = cohortWorkspaceFromReview(payload);
      state.error = null;
      window.EU_HASWORK = true;
      done && done(true);
    }).catch(err => {
      if (sequence !== requestSequence || registryActivePath() !== active) return;
      window.EU_COHORT_REVIEW = null;
      window.EU_VIZ_WORKSPACE = null;
      state.error = String(err && err.message || err);
      done && done(false);
    });
  }
  function applyCustomGroup(group) {
    if (state.customBusy || !registryActivePath()) return;
    const source = registryActivePath();
    const sequence = ++requestSequence;
    const body = { source_path: source };
    if (state.selectedFeatures.length) body.selected_features = state.selectedFeatures.slice();
    if (group) body.custom_group = group;
    state.customBusy = true;
    state.customError = null;
    repaintScreen('cohort');
    window.EU_API.loadCohortReviewSummary(body).then(payload => {
      if (sequence !== requestSequence || registryActivePath() !== source) return;
      if (!customGroupReceiptMatches(payload, group)) {
        state.customBusy = false;
        state.customError = t('The service did not confirm the requested grouping. Restart or update the service before using custom grouping.', '服务未确认请求的自定义分组，请重启或更新服务后重试。');
        repaintScreen('cohort');
        return;
      }
      window.EU_COHORT_REVIEW = payload;
      window.EU_VIZ_WORKSPACE = cohortWorkspaceFromReview(payload);
      syncCohortFeatureSelection(payload);
      state.customGroup = group;
      state.customBusy = false;
      state.compare = group ? 'custom' : 'outcome';
      const custom = (((payload || {}).survival_analysis || {}).group_options || []).find(row => row.id === 'custom');
      state.survivalGroup = group && custom && custom.status === 'ready' ? 'custom' : '';
      repaintScreen('cohort');
    }).catch(err => {
      if (sequence !== requestSequence || registryActivePath() !== source) return;
      state.customBusy = false;
      const code = err && (err.code || err.message);
      const messages = {
        custom_group_feature_unavailable: t('This field is not available in the current export.', '当前导出没有这个字段。'),
        custom_group_numeric_unavailable: t('This field has no valid numeric values for threshold grouping.', '这个字段没有可用于阈值分组的有效数值。'),
        custom_group_category_unavailable: t('The chosen categories are not available in the current export.', '所选类别不在当前导出中。'),
        invalid_custom_group_threshold: t('Enter a finite numeric threshold.', '请输入有限数值作为阈值。'),
        invalid_custom_group_categories: t('Choose two distinct recorded categories.', '请选择两个不同的已记录类别。'),
      };
      state.customError = messages[code] || String(err && err.message || err);
      repaintScreen('cohort');
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
      bindSourceRegistry(root, 'cohort');
      const demoSourceOwner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
      if (!hostedServer() && state.view === 'idle' && demoSourceOwner) {
        demoSourceOwner.ensureLoaded(() => {
          if (window.location.hash === '#cohort' && state.view === 'idle') {
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
        if (state.view === 'loading' || state.customBusy) return;
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
        resetForSourceChange();
        state.featureScope = 'recommended';
        window.EU_VIZ_WORKSPACE = null;
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-scope]').forEach(b => b.addEventListener('click', () => {
        const next = b.dataset.cohortFeatureScope === 'all' ? 'all' : 'recommended';
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
        const next = b.dataset.cohortSofaMatrixMode === 'count' ? 'count' : 'pct';
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
        state.featureModule = b.dataset.cohortFeatureModule || 'all';
        repaintScreen('cohort');
      }));
      root.querySelectorAll('[data-cohort-feature-toggle]').forEach(b => b.addEventListener('click', () => {
        if (state.customBusy) return;
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
        if (state.customBusy) return;
        resetCohortFeatureSelection();
        window.EU_STALE = true;
        reloadCohortForFeatureSelection();
      }));
      root.querySelectorAll('[data-cohort-feature-clear]').forEach(b => b.addEventListener('click', () => {
        if (state.customBusy) return;
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
      const groupKind = root.querySelector('[data-custom-kind]');
      const groupFeature = root.querySelector('[data-custom-feature]');
      const threshold = root.querySelector('[data-custom-threshold]');
      const valueA = root.querySelector('[data-custom-value-a]');
      const valueB = root.querySelector('[data-custom-value-b]');
      if (groupKind) groupKind.addEventListener('change', () => {
        state.customDraft = { kind: groupKind.value };
        state.customError = null;
        repaintScreen('cohort');
      });
      if (groupFeature) groupFeature.addEventListener('change', () => {
        state.customDraft = { ...state.customDraft, feature_id: groupFeature.value, value_a: '', value_b: '' };
        state.customError = null;
        repaintScreen('cohort');
      });
      if (threshold) threshold.addEventListener('input', () => { state.customDraft.threshold = threshold.value; });
      if (valueA) valueA.addEventListener('change', () => { state.customDraft.value_a = valueA.value; });
      if (valueB) valueB.addEventListener('change', () => { state.customDraft.value_b = valueB.value; });
      root.querySelectorAll('[data-custom-apply]').forEach(button => button.addEventListener('click', () => {
        if (state.customBusy) return;
        const kind = groupKind && groupKind.value;
        let group;
        if (kind === 'sex') {
          group = { kind, feature_id: 'demographics:sex' };
        } else if (kind === 'categorical_values') {
          if (!groupFeature || !groupFeature.value || !valueA || !valueB || !valueA.value || !valueB.value || valueA.value === valueB.value) {
            state.customError = t('Choose two distinct recorded categories.', '请选择两个不同的已记录类别。');
            repaintScreen('cohort');
            return;
          }
          group = { kind, feature_id: groupFeature.value, values: [valueA.value, valueB.value] };
        } else {
          const raw = threshold ? threshold.value.trim() : '';
          const number = Number(raw);
          if (!groupFeature || !groupFeature.value || !raw || !Number.isFinite(number)) {
            state.customError = t('Choose a field and enter a finite threshold.', '请选择字段并输入有效的有限阈值。');
            repaintScreen('cohort');
            return;
          }
          group = { kind: 'numeric_threshold', feature_id: groupFeature.value, threshold: number };
        }
        applyCustomGroup(group);
      }));
      root.querySelectorAll('[data-custom-clear]').forEach(button => button.addEventListener('click', () => applyCustomGroup(null)));
    },
    render() {
      if (window.__euCohortPanel) { state.panel = window.__euCohortPanel; window.__euCohortPanel = null; }
      // Consume a Guided Copilot handoff so a study configured in conversation
      // does not silently vanish when Copilot lands the user here.
      if (window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.take) window.EU_GUIDED_HANDOFF.take('cohort');
      const guidedNote = window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.noteHtml ? window.EU_GUIDED_HANDOFF.noteHtml('cohort') : '';
      const ws = window.EU_VIZ_WORKSPACE;
      const kind = activeSourceKind ? activeSourceKind() : null;
      const kindLabel = kind && kind.label ? kind.label : (officialDemoContext() ? t('Public demo', '公开 Demo') : '');
      let review = cohortReview();
      if (reloadStaleRealCohortIfNeeded(review)) review = null;
      const loaded = cohortLoaded();
      const head = `${guidedNote}
      <div class="row gap-8" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-4);margin-bottom:6px;white-space:nowrap;flex-wrap:wrap;row-gap:2px;">
        <span>${view.text('Workspace')}</span> ${icon('chevron', 11)} ${loaded ? `<span>${kindLabel || (hostedServer() ? t('Prepared data', '已准备的数据') : view.text('Local export'))}</span> ${icon('chevron', 11)} ` : ''}<span style="color:var(--ink-2);">${view.text('Cohort statistics')}</span>
      </div>
      <div class="page-head" style="margin-bottom:16px;">
        <h1 style="margin-top:0;">${t('Cohort Statistics', '队列统计')}</h1>
        ${loaded && !(kind && kind.kind === 'synthetic') ? '' : `<p class="lead">${loaded ? t('Synthetic test data · not a research result', '合成测试数据 · 不是研究结果') : (hostedServer() ? t('Choose one of the datasets the administrator prepared to view group contrasts, coverage, survival curves and SOFA reclassification.', '选择管理员准备的一份数据，查看组间对照、覆盖率、生存曲线和 SOFA 重分层。') : t('Choose prepared data to view group contrasts, coverage, survival curves and SOFA reclassification.', '选择一份已准备的数据，查看组间对照、覆盖率、生存曲线和 SOFA 重分层。'))}</p>`}
      </div>`;
      if (state.view !== 'loading' && !loaded) {
        return head + `<div class="card pad" data-cohort-config-required="true">
            ${hostedServer() ? '' : `<div class="panel-head">
              <div>
                <div class="panel-title" style="font-size:17px;">${t('Choose one cohort data source', '选择一个队列数据源')}</div>
                <div class="panel-sub mt-4">${t('Patient Review uses the same current data.', '患者审阅使用的也是这份当前数据。')}</div>
              </div>
            </div>`}
            ${state.error ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono" style="font-size:11px;margin:0;">${esc(state.error)}</div></div></div>` : ''}
            <div class="${hostedServer() ? '' : 'mt-16'}">
              ${!registryActivePath() ? `<div class="note info mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('No current data selected', '尚未选择当前数据')}</div><div class="d">${cohortMissingExportMessage()}</div></div></div>` : ''}
              ${sourceRegistryBlock('single')}
              <button class="btn primary mt-16" data-cohort-run ${!registryActivePath() ? 'aria-disabled="true"' : ''}>${icon('folder', 14)} ${t('Load this data', '加载这份数据')}</button>
            </div>
            ${officialDemoOffers ? officialDemoOffers('cohort') : ''}
          </div>
          ${hostedServer() ? '' : `<div class="empty mt-16" data-cohort-empty-preview="true">
            <div class="glyph">${icon('cohort', 22)}</div>
            <div class="t">${t('No cohort open yet', '还没有打开队列')}</div>
            <div class="d">${t('Open a dataset above to see group comparison, survival curves, coverage, cohort profile and SOFA reclassification.', '在上方打开一份数据后，这里会显示组间对照、生存曲线、覆盖审计、队列画像和 SOFA 重分层。')}</div>
          </div>`}`;
      }
      if (state.view === 'loading') {
        return head + `<div class="card pad">
          <div class="load-strip">
            <span class="spin accent"></span>
            <div class="grow"><div style="font-weight:600;font-size:12.75px;">${t('Recomputing cohort statistics…', '正在重新计算队列统计…')}</div>${hostedServer() ? '' : `<div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${t('reproducible · no outbound calls', '可复现 · 无外部调用')}</div>`}</div>
          </div>
          <div class="note-line mt-8" data-cohort-first-load>${t('The first load of a full database can take two to three minutes; opening it again is quick.', '第一次打开一份完整数据可能需要 2–3 分钟；之后再打开会很快。')}</div>
          <div class="indet mt-12"></div>
          <div class="st-stats mt-16">${[0,1,2,3].map(() => `<div class="sk-stat"><div class="sk sk-line sm" style="width:52%"></div><div class="sk" style="height:22px;width:64%;margin-top:10px;"></div></div>`).join('')}</div>
          <div class="sk-table mt-16">${[0,1,2,3,4].map(() => `<div class="sk-trow">${[60,40,40,40,30].map(w => `<div class="sk sk-line" style="width:${w}%"></div>`).join('')}</div>`).join('')}</div>
        </div>`;
      }
      return head + `
      ${review && review.summary && (!review.summary.mortality || review.summary.mortality.known_count == null) ? `<div class="note warn mt-12" role="status">${t('The current service is still running the older statistics logic; mortality and related summaries have not been updated. Restart the service before relying on them.', '当前服务仍在运行旧统计逻辑，死亡率等尚未更新；请重启服务后再使用这些结果。')}</div>` : ''}
      ${view.tabs()}
      ${['groups', 'survival'].includes(state.panel) && view.customGroupEditor ? view.customGroupEditor(review) : ''}
      <div id="cohbody">${view.panelBody()}</div>
      <div class="nextbar accent mt-16">
        <div class="nb-ico">${icon('arrow', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Next step', '下一步')}</div></div>
        <button class="btn" data-nav="crossdb">${icon('benchmark', 13)} ${t('Cross-database comparison', '跨库对比')}</button>
        <button class="btn primary" data-study-handoff data-study-source="cohort" data-study-target="guided">${icon('agent', 13)} ${t('Continue in Guided Copilot', '在研究对话中继续')}</button>
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
    officialDemoContext = config.officialDemoContext;
    registryActivePath = config.registryActivePath;
    bindSourceRegistry = config.bindSourceRegistry;
    officialDemoOffers = config.officialDemoOffers;
    activeSourceKind = config.activeSourceKind;
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
    ++requestSequence;
    window.EU_COHORT_REVIEW = null;
    resetCohortFeatureSelection();
    state.customGroup = null;
    state.customDraft = { kind: 'numeric_threshold' };
    state.customBusy = false;
    state.customError = null;
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
