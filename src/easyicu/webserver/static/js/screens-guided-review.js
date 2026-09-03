/* Guided Copilot active-export review owner.
   Owns review state, patient/cohort loading, KM presentation, and data-gr DOM
   transitions. The guided shell supplies conversation and rail callbacks but
   never mutates this module's state directly. */
(function () {
  'use strict';

  let host = {
    t: (en) => en,
    icon: () => '',
    esc: (value) => String(value == null ? '' : value),
    attr: (value) => String(value == null ? '' : value),
    fmtInt: (value, fallback) => (value == null ? fallback : String(value)),
    fmtNum: (value, fallback) => (value == null ? fallback : String(value)),
    fmtPct: (value) => String(value == null ? '' : value),
    fmtFixed: (value) => String(value == null ? '' : value),
    fmtP: (value) => String(value == null ? '' : value),
    bi: (en, zh) => ({ en, zh }),
    activeExportSource: () => null,
    activeExportLabel: () => 'local export',
    thread: () => [],
    clearChips: () => {},
    pushUser: () => {},
    setDataMode: () => {},
    setVal: () => {},
    markThrough: () => {},
    renderThread: () => {},
    renderChips: () => {},
    renderAside: () => {},
    scheduleGuidedSlotSave: () => {},
  };

  let reviewState = null;

  function createState() {
    return {
      loading: false,
      error: null,
      patient: null,
      cohort: null,
      selectedRef: null,
    };
  }

  function resetState() {
    reviewState = createState();
    return reviewState;
  }

  function clearState() { reviewState = null; }
  function state() { return reviewState; }

  function sourceLabel(payload) {
    const source = payload && payload.source ? payload.source : host.activeExportSource();
    if (!source) return host.t('No active export', '没有 active export');
    const label = source.label || source.database || 'active export';
    const summary = (payload && payload.summary) || source.summary || {};
    const parts = [];
    if (summary.entities != null || summary.stays != null) {
      parts.push(`${host.fmtInt(summary.entities != null ? summary.entities : summary.stays)} entities`);
    }
    if (summary.modules != null) parts.push(`${host.fmtInt(summary.modules)} modules`);
    return `${label}${parts.length ? ' · ' + parts.join(' · ') : ''}`;
  }

  function start(label) {
    if (label) host.pushUser(label);
    host.setDataMode('real');
    resetState();
    host.setVal({ data: host.activeExportLabel(), review: 'inline Copilot' });
    host.markThrough('review', 'active');
    host.thread().push({ bot: true, html: host.bi(
      `I’ll review the active local export here: patient drilldown, cohort summary, feature coverage, and KM/log-rank when the export has time-to-event fields.`,
      `我会直接在这里审阅 active 本地导出：患者概览、队列汇总、特征覆盖，以及在导出具备 time-to-event 字段时显示 KM/log-rank。`,
    ) });
    host.thread().push({ guidedReview: true });
    host.clearChips();
    host.renderThread();
    host.renderChips();
    host.scheduleGuidedSlotSave('start_review');
    load();
  }

  function load(entityRef) {
    if (!reviewState) resetState();
    const current = reviewState;
    if (!window.EU_API || !window.EU_API.loadPatientReviewDrilldown || !window.EU_API.loadCohortReviewSummary) {
      current.loading = false;
      current.error = 'Review APIs are unavailable.';
      host.renderThread();
      return;
    }
    current.loading = true;
    current.error = null;
    if (entityRef) current.selectedRef = entityRef;
    host.renderThread();
    const body = current.selectedRef ? { entity_ref: current.selectedRef } : {};
    Promise.allSettled([
      window.EU_API.loadPatientReviewDrilldown(body),
      window.EU_API.loadCohortReviewSummary({}),
    ]).then(([patientResult, cohortResult]) => {
      if (reviewState !== current) return;
      current.loading = false;
      const patientOk = patientResult.status === 'fulfilled' && patientResult.value && patientResult.value.ok;
      const cohortOk = cohortResult.status === 'fulfilled' && cohortResult.value && cohortResult.value.ok;
      current.patient = patientOk ? patientResult.value : null;
      current.cohort = cohortOk ? cohortResult.value : null;
      if (current.patient && current.patient.selected) current.selectedRef = current.patient.selected.ref;
      if (!patientOk && !cohortOk) {
        const patientError = patientResult.status === 'rejected'
          ? patientResult.reason
          : (patientResult.value && (patientResult.value.error || patientResult.value.reason));
        const cohortError = cohortResult.status === 'rejected'
          ? cohortResult.reason
          : (cohortResult.value && (cohortResult.value.error || cohortResult.value.reason));
        current.error = (patientError && (patientError.message || String(patientError)))
          || (cohortError && (cohortError.message || String(cohortError)))
          || 'No active registered export is available.';
      }
      if (current.cohort && current.cohort.summary) {
        host.setVal({
          cohort: `${host.fmtInt(current.cohort.summary.cohort_size || current.cohort.summary.entities)} entities`,
          review: 'cohort + KM audit',
        });
      }
      host.renderThread();
      host.renderAside();
      host.scheduleGuidedSlotSave('load_review_data');
    }).catch(error => {
      if (reviewState !== current) return;
      current.loading = false;
      current.error = error.message || String(error);
      host.renderThread();
      host.scheduleGuidedSlotSave('load_review_data_error');
    });
  }

  function metricCard(label, value, sub) {
    return `<div class="gdr-metric"><span>${host.esc(label)}</span><strong>${host.esc(value == null ? 'n/a' : value)}</strong>${sub ? `<small>${host.esc(sub)}</small>` : ''}</div>`;
  }

  function renderPatient(patient) {
    const t = host.t;
    if (!patient) {
      return `<div class="gdr-panel blocked"><strong>${t('Patient drilldown unavailable', '患者 drilldown 不可用')}</strong><span>${t('Select or register an active EasyICU export first.', '请先选择或注册一个 active EasyICU 导出。')}</span></div>`;
    }
    const summary = patient.summary || {};
    const selected = patient.selected || {};
    const demographics = selected.demographics || {};
    const scores = selected.scores || {};
    const outcomes = selected.outcomes || {};
    const entities = (patient.entities || []).slice(0, 5);
    const lanes = (patient.time_lanes || []).filter(row => row.status !== 'unavailable');
    return `
      <div class="gdr-panel">
        <div class="gdr-panel-head">
          <div><span class="gdx-label">${t('Patient drilldown', '患者 drilldown')}</span><strong>${host.esc(selected.label || 'Entity')}</strong></div>
          <div class="gdr-entity-pick">
            ${entities.map(row => `<button type="button" class="${row.ref === selected.ref ? 'on' : ''}" data-gr-entity="${host.attr(row.ref)}">${host.esc(row.label || row.ref)}</button>`).join('')}
          </div>
        </div>
        <div class="gdr-metrics">
          ${metricCard(t('Entities', '实体数'), host.fmtInt(summary.entities))}
          ${metricCard(t('Age', '年龄'), host.fmtNum(demographics.age, 'n/a'), demographics.sex || '')}
          ${metricCard(t('Outcome', '结局'), outcomes.status || 'Unknown', outcomes.icu_los_days != null ? `${host.fmtNum(outcomes.icu_los_days)} ICU days` : '')}
          ${metricCard('SOFA-2', host.fmtNum(scores.sofa2_max, 'n/a'), scores.sepsis3_sofa2 == null ? '' : `sepsis ${scores.sepsis3_sofa2 ? 'yes' : 'no'}`)}
        </div>
        <div class="gdr-mini-table">
          ${(patient.module_profiles || []).slice(0, 5).map(row => `
            <div><strong>${host.esc(row.label || row.module)}</strong><span>${host.fmtInt(row.rows)} rows · ${host.fmtPct(row.coverage_pct)} coverage · ${host.fmtInt(row.feature_count)} features</span></div>
          `).join('') || `<div><strong>${t('No modules found', '未找到模块')}</strong><span>${t('The export does not expose reviewable modules.', '该导出没有可审阅模块。')}</span></div>`}
        </div>
        ${lanes.length ? `<div class="gdr-note">${lanes.map(row => `${row.label}: ${host.fmtInt(row.signal_count)} signals`).join(' · ')}</div>` : `<div class="gdr-note">${t('No time-series lanes are available in this export. Add vitals/labs/scores modules to review trajectories.', '该导出暂无时间序列通道。请补充 vitals/labs/scores 模块后查看轨迹。')}</div>`}
      </div>`;
  }

  function survivalCurve(cohort) {
    const survival = cohort && cohort.survival_analysis ? cohort.survival_analysis : {};
    const outcomeId = survival.default_outcome || ((survival.outcomes || []).find(row => row.status === 'ready') || {}).id;
    const groupId = survival.default_group || ((survival.group_options || []).find(row => row.status === 'ready') || {}).id;
    return (survival.curves || []).find(row => row.outcome_id === outcomeId && row.group_id === groupId) || null;
  }

  function renderKm(cohort) {
    const t = host.t;
    const survival = cohort && cohort.survival_analysis ? cohort.survival_analysis : {};
    const curve = survivalCurve(cohort);
    const blocked = (survival.outcomes || []).filter(row => row.status !== 'ready').slice(0, 3);
    if (!curve) {
      return `
        <div class="gdr-panel blocked">
          <div class="gdr-panel-head"><div><span class="gdx-label">KM / log-rank</span><strong>${t('Blocked by export schema', '被导出结构阻断')}</strong></div></div>
          <p>${host.esc(survival.reason || t('This export does not expose event and time-to-event columns for KM/log-rank.', '该导出没有 KM/log-rank 所需的事件列和 time-to-event 列。'))}</p>
          ${blocked.length ? `<div class="gdr-mini-table">${blocked.map(row => `<div><strong>${host.esc(row.label || row.id)}</strong><span>${host.esc(row.reason || 'unavailable')}</span></div>`).join('')}</div>` : ''}
        </div>`;
    }
    const logrank = curve.logrank || {};
    const groups = curve.groups || [];
    const risk = curve.number_at_risk || {};
    const times = risk.times || [];
    const rows = risk.rows || [];
    return `
      <div class="gdr-panel">
        <div class="gdr-panel-head">
          <div><span class="gdx-label">KM / log-rank</span><strong>${host.esc(curve.label || 'Kaplan-Meier curve')}</strong></div>
          <div class="gdr-logrank"><span>log-rank</span><strong>${logrank.status === 'ready' ? `χ² ${host.fmtFixed(logrank.chi_square, 2)} · p ${host.fmtP(logrank.p_value)}` : 'blocked'}</strong></div>
        </div>
        <div class="gdr-km">
          ${groups.map((group, index) => `<div class="gdr-km-row"><span class="line c${index % 4}"></span><strong>${host.esc(group.label || `Group ${index + 1}`)}</strong><em>n ${host.fmtInt(group.n)} · events ${host.fmtInt(group.events)}</em></div>`).join('')}
        </div>
        ${times.length && rows.length ? `<div class="gdr-risk"><strong>${t('Number at risk', '风险人数表')}</strong><table><thead><tr><th>Group</th>${times.map(value => `<th>${host.fmtNum(value)}d</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr><td>${host.esc(row.label)}</td>${(row.values || []).map(value => `<td>${host.fmtInt(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : ''}
        <div class="gdr-note">${t('Exploratory aggregate only. Manuscript claims still need Agent evidence checks and human review.', '仅为探索性聚合结果。论文结论仍需 Agent 证据核验和人工审阅。')}</div>
      </div>`;
  }

  function renderCohort(cohort) {
    const t = host.t;
    if (!cohort) {
      return `<div class="gdr-panel blocked"><strong>${t('Cohort summary unavailable', '队列汇总不可用')}</strong><span>${t('Register an active export, then refresh this card.', '请先注册 active export，然后刷新这张卡。')}</span></div>`;
    }
    const summary = cohort.summary || {};
    const mortality = summary.mortality || {};
    const coverage = (cohort.coverage || []).slice(0, 6);
    return `
      <div class="gdr-panel">
        <div class="gdr-panel-head"><div><span class="gdx-label">${t('Cohort summary', '队列汇总')}</span><strong>${host.esc(sourceLabel(cohort))}</strong></div></div>
        <div class="gdr-metrics">
          ${metricCard(t('Cohort', '队列'), host.fmtInt(summary.cohort_size || summary.entities), 'entities')}
          ${metricCard(t('Mortality', '死亡率'), host.fmtPct(summary.mortality_pct), `${host.fmtInt(mortality.deceased_count, 'n/a')} events`)}
          ${metricCard(t('Median age', '年龄中位数'), host.fmtNum(summary.age && summary.age.median, 'n/a'), 'years')}
          ${metricCard(t('Modules', '模块'), host.fmtInt(summary.modules), `${host.fmtInt(summary.total_records)} records`)}
        </div>
        <div class="gdr-mini-table">
          ${coverage.map(row => `<div><strong>${host.esc(row.label || row.module)}</strong><span>${host.fmtPct(row.coverage_pct)} · ${host.fmtInt(row.rows)} rows · ${host.esc(row.quality_status || 'ok')}</span></div>`).join('') || `<div><strong>${t('No coverage rows', '暂无覆盖率')}</strong><span>${t('This export has no auditable feature modules yet.', '该导出还没有可审计特征模块。')}</span></div>`}
        </div>
      </div>`;
  }

  function renderCard() {
    if (!reviewState) resetState();
    const loading = reviewState.loading;
    const hasPayload = reviewState.patient || reviewState.cohort;
    return `
      <div class="gd-review-card">
        <div class="gdx-head">
          <span class="gdx-ico">${host.icon('eye', 15)}</span>
          <div>
            <strong>${host.t('Review active export inside Copilot', '在 Copilot 内审阅 active export')}</strong>
            <span>${host.t('Uses Patient Review and Cohort Statistics APIs; no seeded demo panels are substituted.', '复用 Patient Review 和 Cohort Statistics API；不会用 seeded demo 面板替代。')}</span>
          </div>
        </div>
        <div class="gdx-status ${reviewState.error ? 'bad' : hasPayload ? 'ok' : ''}">
          <span>${host.icon(reviewState.error ? 'x' : hasPayload ? 'check' : 'shield', 12)}</span>
          <div><strong>${loading ? host.t('Loading active export review...', '正在加载 active export 审阅...') : reviewState.error ? host.esc(reviewState.error) : hasPayload ? host.t('Loaded from active registered export.', '已从 active registered export 加载。') : host.t('No review loaded yet.', '尚未加载审阅。')}</strong><small>${host.esc(sourceLabel(reviewState.cohort || reviewState.patient))}</small></div>
        </div>
        ${hasPayload ? `
          <div class="gdr-grid">
            ${renderPatient(reviewState.patient)}
            ${renderCohort(reviewState.cohort)}
            ${renderKm(reviewState.cohort)}
          </div>
        ` : ''}
        <div class="gdx-actions">
          <button type="button" class="btn primary" data-gr-refresh ${loading ? 'disabled' : ''}>${host.icon('refresh', 13)} ${host.t('Refresh review', '刷新审阅')}</button>
          <button type="button" class="btn" data-guided-goal="data_extraction">${host.t('Prepare more modules here', '在这里补抽取模块')}</button>
          <button type="button" class="btn" data-guided-goal="run_agent" ${hasPayload ? '' : 'disabled'}>${host.t('Continue to Agent preflight', '继续 Agent 预检')}</button>
        </div>
      </div>`;
  }

  function handleClick(target) {
    if (!target || typeof target.closest !== 'function') return false;
    if (target.closest('[data-gr-refresh]')) { load(); return true; }
    const entity = target.closest('[data-gr-entity]');
    if (entity) { load(entity.dataset.grEntity); return true; }
    return false;
  }

  function slotSnapshot() {
    if (!reviewState) return null;
    return {
      selected_ref: reviewState.selectedRef || null,
      loaded: !!(reviewState.patient || reviewState.cohort),
      source_label: sourceLabel(reviewState.cohort || reviewState.patient),
      cohort_size: reviewState.cohort
        && reviewState.cohort.summary
        && (reviewState.cohort.summary.cohort_size || reviewState.cohort.summary.entities),
    };
  }

  function restoreSlot(slot) {
    if (!slot || typeof slot !== 'object') return;
    resetState();
    reviewState.selectedRef = slot.selected_ref || null;
  }

  window.EU_GUIDED_REVIEW = {
    init(bindings) { host = Object.assign({}, host, bindings || {}); },
    clearState,
    handleClick,
    load,
    renderCard,
    resetState,
    restoreSlot,
    slotSnapshot,
    sourceLabel,
    start,
    state,
  };
})();
