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

  function cohortSurvivalBody(review) {
    const survival = review.survival_analysis || {};
    const outcomes = survival.outcomes || [];
    const groups = survival.group_options || [];
    const readyOutcomes = outcomes.filter(row => row.status === 'ready');
    const readyGroups = groups.filter(row => row.status === 'ready');
    const selectedOutcome = survival.default_outcome || (readyOutcomes[0] && readyOutcomes[0].id) || state.survivalOutcome;
    state.survivalOutcome = selectedOutcome || state.survivalOutcome;
    const selectedGroup = readyGroups.some(row => row.id === state.survivalGroup)
      ? state.survivalGroup
      : (survival.default_group || (readyGroups[0] && readyGroups[0].id));
    const curve = (survival.curves || []).find(row => row.outcome_id === selectedOutcome && row.group_id === selectedGroup);
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
    if (!curve) {
      return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Kaplan-Meier module')}</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">${cohortText('Outcome overview')}</div>${outcomeCards}</div>
        <div><div class="surv-label">${cohortText('Grouping')}</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${cohortText('Survival analysis blocked')}</div><div class="d">${esc(cohortReason(survival.reason || 'This export does not expose an outcome with both event and time-to-event columns.'))}</div></div></div>
      ${cohortSurvivalSourceHint(survival)}
      ${cohortSurvivalBlockedList(blockedOutcomes)}
      ${cohortSurvivalBlockedList(blockedGroups)}`;
    }
    const logrank = curve.logrank || {};
    const pValueLabel = logrank.p_value_label || fmtP(logrank.p_value);
    const windowNote = cohortSurvivalWindowNote(curve);
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Kaplan-Meier curves and log-rank')}</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">${cohortText('Outcome overview')}</div>${outcomeCards}</div>
        <div><div class="surv-label">${cohortText('Grouping')}</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      <div class="surv-card mt-14">
        <div class="surv-head">
          <div>
            <div class="eyebrow">${cohortText('Exploratory · unadjusted')}</div>
            <h3>${esc(cohortText(curve.label || 'Kaplan-Meier curve'))}</h3>
            <p>${esc(cohortText(curve.time_label || 'Time-to-event'))} · ${t('event', '事件')} <span class="mono">${esc(curve.event_column || '')}</span> · ${t('time', '时间')} <span class="mono">${esc(curve.time_column || '')}</span></p>
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
      ${cohortSurvivalBlockedList(blockedOutcomes)}
      ${cohortSurvivalBlockedList(blockedGroups)}`;
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
  };
})();

