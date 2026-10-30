/* Owner: Cohort Statistics survival widget. */
/* Cohort Statistics survival rendering sub-owner. */
(function () {
  let state;
  let t;
  let icon;
  let esc;
  let fmtInt;
  let fmtNum;
  let fmtPct;
  let fmtP;
  let cohortCharts;
  let cohortReview;
  let cohortText;
  let cohortReason;

  // Name export columns by catalog concept, including SOFA evidence flags.
  function cohortFeatureName(column) {
    if (window.EU_LANG !== 'zh' || !column) return '';
    const dict = (window.EU_CATALOG || {}).dict || {};
    const id = String(column).toLowerCase();
    const zhOf = key => (dict[key] && dict[key][1]) || (dict[`sofa2_${key}`] && dict[`sofa2_${key}`][1]) || '';
    if (zhOf(id)) return zhOf(id);
    const flag = id.match(/^(.+)_(observed|available|time)$/);
    if (!flag || !zhOf(flag[1])) return '';
    return `${zhOf(flag[1])}${{ observed: ' 已观测', available: ' 可用', time: '时间' }[flag[2]]}`;
  }

  function customGroupFeatureLabel(group) {
    const id = String(group && group.feature_id || '');
    const column = id.split(':')[1] || id;
    const localized = cohortFeatureName(column);
    if (localized) return localized;
    if (column === 'age') return t('Age', '年龄');
    if (column === 'sex') return t('Sex', '性别');
    const review = cohortReview() || {};
    const modules = (review.feature_catalog || {}).modules || [];
    const feature = modules.flatMap(module => module.features || []).find(row => row.id === id);
    const category = ((review.custom_group_options || {}).categories || []).find(row => row.feature_id === id);
    return (feature && feature.label) || (category && category.label) || column;
  }

  function cohortSurvivalBody(review) {
    const survival = review.survival_analysis || {};
    const outcomes = survival.outcomes || [];
    const groups = survival.group_options || [];
    const readyOutcomes = outcomes.filter(row => row.status === 'ready');
    const readyGroups = groups.filter(row => row.status === 'ready');
    const selectedGroup = readyGroups.some(row => row.id === state.survivalGroup)
      ? state.survivalGroup
      : (survival.default_group || (readyGroups[0] && readyGroups[0].id));
    const landmarkHours = selectedGroup === 'custom' && state.customGroup && state.customGroup.landmark_hours;
    const selectedOutcome = landmarkHours ? 'icu_death' : (survival.default_outcome || (readyOutcomes[0] && readyOutcomes[0].id) || state.survivalOutcome);
    state.survivalOutcome = selectedOutcome || state.survivalOutcome;
    // Fail closed if an older or mixed service returns an unshifted or mismatched curve.
    const curve = (survival.curves || []).find(row => row.outcome_id === selectedOutcome && row.group_id === selectedGroup && (!landmarkHours || (
      row.landmark_hours === landmarkHours && row.event_time_basis === 'icu_discharge_status_and_exit_time_proxy'
      && row.measurement_aggregation === 'first' && row.measurement_window_end_inclusive === false
      && Array.isArray(row.measurement_window_hours) && row.measurement_window_hours[0] === 0 && row.measurement_window_hours[1] === landmarkHours
      && typeof row.measurement_unit === 'string' && !!row.measurement_unit.trim() && row.landmark_counts
    )));
    const outcomeCards = cohortSurvivalOutcomeCards(outcomes, selectedOutcome);
    const groupButtons = groups.map(row => {
      const ready = row.status === 'ready';
      const cls = `seg-btn ${selectedGroup === row.id ? 'active' : ''} ${ready ? '' : 'disabled'}`;
      const attr = ready ? `data-cohort-surv-group="${esc(row.id)}"` : `aria-disabled="true" title="${esc(cohortReason(row.reason || 'Unavailable'))}"`;
      const n = (row.groups || []).map(g => fmtInt(g.count)).join(' / ');
      return `<button class="${cls}" ${attr}><span>${esc(cohortText(row.label || row.id))}</span><b>${ready ? n : cohortText('blocked')}</b></button>`;
    }).join('');
    const blockedOutcomes = outcomes.filter(row => row.status !== 'ready');
    const blockedGroups = groups.filter(row => row.status !== 'ready');
    // Explain blocked choices next to their buttons, not after a long plot.
    // A blocked custom group must not look as if the fallback curve represents it.
    const customBlocked = blockedGroups.find(row => row.id === 'custom');
    const blockedCounts = customBlocked && (customBlocked.groups || []).length
      ? `<div class="d">${customBlocked.basis === 'icu_landmark_first_measurement' ? t('Landmark risk-set groups', '生存起点风险集分组') : t('Group counts', '分组人数')}：${customBlocked.groups.map(g => `${esc(g.label)} = ${fmtInt(g.count)}`).join(' · ')}${customBlocked.measurement_unit ? ` · ${t('declared unit (check source)', '声明单位（需核对来源）')} ${esc(customBlocked.measurement_unit)}` : ''}</div>` : '';
    const blockedNotice = customBlocked ? `<div class="note warn mt-12" role="status"><div class="body"><div class="t">${t('Custom survival grouping is unavailable', '自定义生存分组不可用')}</div><div class="d">${esc(cohortReason(customBlocked.reason || 'Unavailable for this export'))} ${t('The curve below uses the selected available grouping, not this custom grouping.', '下方曲线使用当前可用的分组，并非这个自定义分组。')}</div>${blockedCounts}</div></div>` : '';
    if (!curve) {
      const reason = landmarkHours
        ? t('The service did not return an ICU survival curve for the requested landmark. Review the reasons below; no other curve has been substituted.', '服务没有返回所选时间起点的 ICU 生存曲线。请查看下方原因；不会用其他曲线代替。')
        : cohortReason(survival.reason || 'This export does not expose an outcome with both event and time-to-event columns.');
      return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Kaplan-Meier module')}</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">${cohortText('Outcome overview')}${landmarkHours ? ` · ${t('whole cohort, not landmark risk set', '全队列，非 landmark 风险集')}` : ''}</div>${outcomeCards}</div>
        <div><div class="surv-label">${cohortText('Grouping')}</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      ${blockedNotice}
      ${cohortSurvivalBlockedList(blockedOutcomes)}
      ${cohortSurvivalBlockedList(blockedGroups)}
      <div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${cohortText('Survival analysis blocked')}</div><div class="d">${esc(reason)}</div></div></div>
      ${cohortSurvivalSourceHint(survival)}
      `;
    }
    const logrank = curve.logrank || {};
    const pValueLabel = logrank.p_value_label || fmtP(logrank.p_value);
    const windowNote = cohortSurvivalWindowNote(curve);
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Kaplan-Meier curves and log-rank')}</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">${cohortText('Outcome overview')}${landmarkHours ? ` · ${t('whole cohort, not landmark risk set', '全队列，非 landmark 风险集')}` : ''}</div>${outcomeCards}</div>
        <div><div class="surv-label">${cohortText('Grouping')}</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      ${blockedNotice}
      ${cohortSurvivalBlockedList(blockedOutcomes)}
      ${cohortSurvivalBlockedList(blockedGroups)}
      <div class="surv-card mt-14">
        <div class="surv-head">
          <div>
            <div class="eyebrow">${cohortText('Exploratory · unadjusted')}</div>
            <h3>${esc(cohortText(curve.label || 'Kaplan-Meier curve'))}</h3>
            <p>${esc(cohortText(curve.time_label || 'Time-to-event'))} · ${t('event', '事件')} <span class="mono">${esc(curve.event_column || '')}</span> · ${t('time', '时间')} <span class="mono">${esc(curve.time_column || '')}</span></p>
            ${landmarkHours ? cohortLandmarkNote(curve) : ''}
            ${windowNote ? `<p>${esc(windowNote)}</p>` : ''}
          </div>
          <div class="surv-logrank">
            <span>${cohortText('Log-rank')}</span>
            <strong>${logrank.status === 'ready' ? `χ² ${fmtNum(logrank.chi_square, 2)} · p = ${esc(pValueLabel)}` : cohortText('unavailable')}</strong>
            <small>${logrank.status === 'ready' ? cohortText('df 1 · exploratory only · point estimates, no CI') : esc(cohortReason(logrank.reason || 'not enough events'))}</small>
          </div>
        </div>
        ${cohortSurvivalChart(curve)}
        <div class="muted">${t('The event-free estimate is specific to the selected endpoint and censoring rule. A same-time group difference requires an audited calculation.', '无事件估计仅对应所选结局及其删失规则；组间同一时点差值需要经审计的计算。')}</div>
        ${curve.outcome_id === 'hospital_death' ? `<div class="muted">${t('Discharge alive is censored: this is an in-hospital death analysis, not all-cause 30-day survival.', '存活出院按删失处理：这是院内死亡分析，并非全因 30 天生存。')}</div>` : ''}
        ${cohortRiskTable(curve)}
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Not manuscript-ready by itself')}</div><div class="d">${t('KM/log-rank is computed from bounded cohort aggregates and marked exploratory. Any claim still needs the evidence-bound Agent check and human review.', 'KM 曲线和 log-rank 检验由队列汇总计算，属于探索性结果。要写进论文的结论，仍需在研究对话中核对证据并经人工审阅。')}</div></div></div>
      `;
  }

  function cohortLandmarkNote(curve) {
    const hours = curve.landmark_hours;
    const counts = curve.landmark_counts || {};
    const items = [
      ['included', t('included', '纳入')],
      ['missing_followup', t('missing follow-up', '随访时间缺失')],
      ['missing_event', t('event status unknown', '结局未知')],
      ['death_at_or_before_landmark', t('death at or before landmark', '起点前或起点时死亡')],
      ['discharged_at_or_before_landmark', t('left ICU at or before landmark', '起点前或起点时离开 ICU')],
      ['missing_measurement', t('no valid measurement in window', '窗口内无有效测量')],
    ];
    const countText = items.filter(([key]) => counts[key] != null).map(([key, label]) => `${label} ${fmtInt(counts[key])}`).join(' · ');
    const unit = curve.measurement_unit ? ` · ${t('declared unit (check source)', '声明单位（需核对来源）')} ${curve.measurement_unit}` : '';
    const eventBasis = curve.event_time_basis === 'icu_discharge_status_and_exit_time_proxy'
      ? `<p>${t('ICU death status is taken from ICU discharge; the ICU exit time is a proxy for event time, not a recorded time of death.', 'ICU 死亡状态来自 ICU 出科记录；事件时间使用 ICU 出科时点作为代理，并非记录的死亡时刻。')}</p>` : '';
    return `<p>${esc(t(`Landmark at ${hours} hours after ICU admission. First valid measurement from admission up to (excluding) the landmark; ties use the median. ICU survival is measured from the landmark.`, `生存起点为 ICU 入科后第 ${hours} 小时；按入科至该起点之前的首次有效测量分组，同一最早时点多次测量取中位数。生存时间从该起点开始计算。`) + unit)}</p>${eventBasis}${countText ? `<p>${esc(countText)}</p>` : ''}`;
  }

  function cohortSurvivalOutcomeCards(outcomes, selectedOutcome) {
    const rows = outcomes || [];
    if (!rows.length) {
      return `<div class="surv-outcome-grid"><div class="surv-outcome-card muted"><span>${cohortText('No outcome module')}</span><b>${cohortText('not available')}</b></div></div>`;
    }
    return `
      <div class="surv-outcome-grid">
        ${rows.map(row => {
          const summary = row.event_summary || {};
          const hasRate = summary.status === 'available' && summary.event_rate_pct != null;
          const selected = row.id === selectedOutcome;
          const cls = `surv-outcome-card ${selected ? 'active' : ''} ${hasRate ? '' : 'muted'}`;
          const rate = hasRate ? fmtPct(summary.event_rate_pct) : cohortText('not available');
          const events = hasRate
            ? `${fmtInt(summary.event_count)} / ${fmtInt(summary.denominator)} ${cohortText('events')}`
            : cohortReason(summary.reason || row.reason || 'No event column found');
          const meta = cohortSurvivalOutcomeMeta(row);
          return `
            <div class="${cls}">
              <span>${esc(cohortText(row.label || row.id))}</span>
              <strong>${esc(rate)}</strong>
              <b>${esc(events)}</b>
              ${meta ? `<em>${esc(meta)}</em>` : ''}
            </div>`;
        }).join('')}
      </div>`;
  }

  function cohortSurvivalOutcomeMeta(row) {
    if (!row) return '';
    const parts = [];
    const summary = row.event_summary || {};
    if (row.id === 'mort_28d' && row.status === 'ready') {
      parts.push(cohortText('KM curve endpoint'));
    } else if (summary.status === 'available') {
      parts.push(cohortText('Event rate summary'));
    }
    if (summary.basis === 'fixed_horizon_event_and_followup') {
      parts.push(cohortText('dedicated flag + follow-up'));
    }
    if (row.window_label && row.id === 'mort_28d') {
      parts.push(cohortText(row.window_label));
    }
    return parts.join(' · ');
  }

  function cohortSurvivalOutcomeUnavailable(row) {
    const reason = cohortReason(row && row.reason);
    if (reason.includes('ICU') || reason.includes('专用')) {
      return t('unavailable · missing ICU event/time columns', '不可用 · 缺少 ICU 事件/时间列');
    }
    return cohortText('unavailable');
  }

  function cohortSurvivalWindowNote(curve) {
    if (!curve || curve.display_horizon_days == null) return '';
    const days = fmtNum(curve.display_horizon_days, 0);
    const base = t(
      `Displayed on a ${days}-day window; later observations are censored at the window boundary.`,
      `默认显示 ${days} 天窗口；窗口之后的观测在边界处按删失处理。`
    );
    return base;
  }

  function cohortSurvivalSourceHint(survival) {
    const reason = cohortReason((survival && survival.reason) || 'This export does not expose an outcome with both event and time-to-event columns.');
    const review = cohortReview();
    const source = (review && review.source) || {};
    const summary = (review && review.summary) || {};
    return `
      <div class="note info mt-12" data-survival-source-hint>
        <div class="ico">${icon('db', 14)}</div>
        <div class="body">
          <div class="t">${t('Current export is already loaded', '当前导出已加载')}</div>
          <div class="d">${t('Cohort Review uses the active export. KM/log-rank also requires a supported grouping at the time origin as well as event and time columns. Review the reasons below or continue with an audited local analysis. No re-import is required.', '队列统计使用当前数据。KM 曲线和 log-rank 除事件及时间列外，还需要在时间起点有效的分组。请查看下方原因，或继续进行经审计的本地分析，不需要重新导入。')}</div>
          <div class="d mono" style="margin-top:4px;">${esc(reason)}</div>
        </div>
      </div>
      <div class="card pad mt-12" data-survival-current-export>
        <div class="sec-stack" style="margin-bottom:10px;"><div class="lbl">${t('Loaded source', '已加载来源')}</div><h2>${t('KM uses the current export snapshot', 'KM 曲线使用当前数据')}</h2></div>
        <div class="setup-row"><span class="k">${cohortText('Source')}</span><span class="vv">${esc(source.label || cohortText('Local export'))}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Cohort size')}</span><span class="vv">${fmtInt(summary.cohort_size)} ${t('entities', '例')}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Outcome')}</span><span class="vv">${t('Outcome module is registered in this export; blocked rows below explain any missing event/time pair.', '这份数据包含结局模块；如缺少事件列或时间列，会在下方说明。')}</span></div>
      </div>`;
  }

  function cohortSurvivalBlockedList(rows) {
    if (!rows || !rows.length) return '';
    return `
      <div class="surv-blocked mt-12">
        ${rows.map(row => `<div class="surv-blocked-row"><span>${esc(cohortText(row.label || row.id))}</span><em>${esc(cohortReason(row.reason || 'Unavailable for this export'))}</em></div>`).join('')}
      </div>`;
  }

  function cohortSurvivalChart(curve) {
    if (!cohortCharts || typeof cohortCharts.survivalSlot !== 'function') return '';
    return `<div class="km-chart-wrap">
      ${cohortCharts.survivalSlot({
        label: cohortText(curve.label || 'Kaplan-Meier curve'),
        description: t(
          'Kaplan-Meier survival probability by cohort group. Curves are unadjusted aggregate estimates.',
          '按队列分组的 Kaplan-Meier 生存概率；曲线为未校正的聚合估计。',
        ),
        xLabel: cohortText('Days'),
        yLabel: cohortText('Survival probability'),
        groupLabel: cohortText('Group'),
        eventsLabel: cohortText('events'),
        censoredLabel: t('censored', '删失'),
        finalLabel: t('Final survival', '末次生存率'),
        atRiskLabel: t('Number at risk', '风险人数'),
        /* The backend declares the display window; passing it through is what
           lets the number-at-risk columns line up with the axis ticks. */
        horizon: curve.display_horizon_days,
        /* Row labels go through cohortText like the group labels do, so the
           risk table and the group rows key off the same translated string. */
        atRisk: curve.number_at_risk
          ? {
            times: curve.number_at_risk.times || [],
            rows: (curve.number_at_risk.rows || []).map(row => ({
              label: cohortText(row.label),
              values: row.values || [],
            })),
          }
          : null,
        groups: (curve.groups || []).map(group => ({
          label: cohortText(group.label),
          n: fmtInt(group.n),
          events: fmtInt(group.events),
          censored: group.censored == null ? null : fmtInt(group.censored),
          points: group.points || [],
          censorMarks: group.censor_marks || [],
        })),
      })}
      <div class="viz-cap"><b>${t('How to read', '怎么读')}</b><span>${t('A downward step represents events at that time and can include more than one event. Separation shows different unadjusted estimates; it does not by itself establish a reliable or causal difference.', '曲线下降表示该时点发生了事件，同一时点可能有多例。曲线分离反映未经校正的估计不同，不能仅凭图形认定可靠差异或因果关系。')}</span></div>
    </div>`;
  }

  function cohortRiskTable(curve) {
    const risk = curve.number_at_risk || {};
    const times = risk.times || [];
    const rows = risk.rows || [];
    if (!times.length || !rows.length) return '';
    return `
      <div class="risk-table-wrap">
        <div class="surv-label">${cohortText('Number at risk')}</div>
        <table class="risk-table">
          <thead><tr><th>${cohortText('Group')}</th>${times.map(tick => `<th>${fmtNum(tick, 1)}d</th>`).join('')}</tr></thead>
          <tbody>
            ${rows.map(row => `<tr><td>${esc(cohortText(row.label))}</td>${(row.values || []).map(value => `<td>${fmtInt(value)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }

  function init(config) {
    state = config.state;
    t = config.t;
    icon = config.icon;
    esc = config.esc;
    fmtInt = config.fmtInt;
    fmtNum = config.fmtNum;
    fmtPct = config.fmtPct;
    fmtP = config.fmtP;
    cohortCharts = config.cohortCharts;
    cohortReview = config.cohortReview;
    cohortText = config.cohortText;
    cohortReason = config.cohortReason;
  }

  window.EU_VIZ_COHORT_SURVIVAL = {
    init,
    body: cohortSurvivalBody,
    featureName: cohortFeatureName,
    customGroupFeatureLabel,
  };
})();

