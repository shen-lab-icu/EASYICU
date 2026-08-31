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
    if (!curve) {
      return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Kaplan-Meier module')}</h2></div>
      <div class="surv-toolbar">
        <div><div class="surv-label">${cohortText('Outcome overview')}</div>${outcomeCards}</div>
        <div><div class="surv-label">${cohortText('Grouping')}</div><div class="surv-segments">${groupButtons}</div></div>
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${cohortText('Survival analysis blocked')}</div><div class="d">${esc(cohortReason(survival.reason || 'This export does not expose an outcome with both event and time-to-event columns.'))}</div></div></div>
      ${cohortSurvivalSourceHint(survival)}
      ${cohortSurvivalBlockedList(blockedOutcomes)}`;
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
        ${cohortSurvivalEffect(curve)}
        ${cohortRiskTable(curve)}
      </div>
      <div class="note warn mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Not manuscript-ready by itself')}</div><div class="d">${t('KM/log-rank is computed from bounded cohort aggregates and marked exploratory. Any claim still needs the evidence-bound Agent check and human review.', 'KM/log-rank 由有界队列聚合计算，标记为探索性。任何稿件声明仍需要证据绑定 Agent 检查和人工审阅。')}</div></div></div>
      ${cohortSurvivalBlockedList(blockedOutcomes)}`;
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
          <div class="d">${t('Cohort Review is using the active EasyICU export. KM/log-rank will run here when the Outcome module has both an event flag and a time-to-event or censoring-time column; otherwise continue from this same export in an audited local Agent analysis. No re-import is required.', '队列审阅正在使用当前 active 的 EasyICU 导出。只要 Outcome/结局模块同时有事件标志和事件时间或删失时间列，KM/log-rank 就会直接在这里运行；否则也应从同一个导出进入本地 Agent 审计分析，不需要重新导入。')}</div>
          <div class="d mono" style="margin-top:4px;">${esc(reason)}</div>
        </div>
      </div>
      <div class="card pad mt-12" data-survival-current-export>
        <div class="sec-stack" style="margin-bottom:10px;"><div class="lbl">${t('Loaded source', '已加载来源')}</div><h2>${t('KM uses the current export snapshot', 'KM 复用当前导出快照')}</h2></div>
        <div class="setup-row"><span class="k">${cohortText('Source')}</span><span class="vv">${esc(source.label || cohortText('Local export'))}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Path hash')}</span><span class="vv mono">${esc(source.path_hash || '')}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Cohort size')}</span><span class="vv">${fmtInt(summary.cohort_size)} ${t('entities', '个实体')}</span></div>
        <div class="setup-row"><span class="k">${cohortText('Outcome')}</span><span class="vv">${t('Outcome module is registered in this export; blocked rows below explain any missing event/time pair.', '此导出已注册 Outcome/结局模块；下方拦截行会说明缺少的事件/时间组合。')}</span></div>
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
      <div class="viz-cap"><b>${t('How to read', '怎么读')}</b><span>${t('Each step down is one event (e.g. a death); a gap between curves means the groups differ. Unadjusted — for an adjusted effect, continue with this cohort in Guided Copilot.', '曲线每下降一格代表一次事件（如一例死亡）；两条曲线分开表示组间有差异。未做校正 —— 想要校正后的效应，请把该队列带入「研究引导」继续。')}</span></div>
    </div>`;
  }

  // Coarse, always-available effect summary so the panel does not surface a lone
  // p-value: end-of-follow-up survival per group + absolute risk difference
  // (unadjusted, no CI). Complements the log-rank rather than replacing it.
  function cohortSurvivalEffect(curve) {
    const groups = (curve.groups || []).filter(g => Array.isArray(g.points) && g.points.length);
    if (groups.length < 2) return '';
    const finals = groups.map(g => {
      const last = g.points[g.points.length - 1];
      const surv = Number(last && last.survival);
      return { label: g.label, time: Number(last && last.time), surv, risk: 100 - surv };
    }).filter(f => Number.isFinite(f.surv));
    if (finals.length < 2) return '';
    const tMax = Math.max(...finals.map(f => f.time).filter(Number.isFinite));
    const survBits = finals.map(f => `${esc(cohortText(f.label))} ${fmtNum(f.surv, 1)}%`).join(' vs ');
    const risks = finals.map(f => f.risk);
    const ard = Math.max(...risks) - Math.min(...risks);
    const tLabel = Number.isFinite(tMax) ? `${fmtNum(tMax, 0)}${t('d survival', ' 天生存')}` : t('end-of-follow-up survival', '随访末生存');
    return `<div class="surv-effect">
      <span class="surv-effect-lab">${t('Effect (absolute · unadjusted)', '效应量（绝对 · 未校正）')}</span>
      <span>${tLabel}: ${survBits}</span>
      <span>${t('absolute risk difference', '绝对风险差')} <strong>${fmtNum(ard, 1)} ${t('pp', '个百分点')}</strong></span>
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

  function cohortSurvivalDemoBody() {
    const curve = {
      label: 'Demo hospital mortality by Sepsis vs Non-sepsis',
      time_label: 'Demo follow-up days',
      event_column: 'demo_hospital_death',
      time_column: 'demo_followup_days',
      logrank: { status: 'ready', chi_square: 5.42, p_value: 0.0198 },
      groups: [
        {
          label: 'Non-sepsis',
          n: 160,
          events: 18,
          points: [
            { time: 0, survival: 100 },
            { time: 1, survival: 98.8 },
            { time: 3, survival: 96.9 },
            { time: 7, survival: 93.8 },
            { time: 14, survival: 90.7 },
            { time: 28, survival: 88.8 },
          ],
        },
        {
          label: 'Sepsis',
          n: 140,
          events: 34,
          points: [
            { time: 0, survival: 100 },
            { time: 1, survival: 97.9 },
            { time: 3, survival: 93.6 },
            { time: 7, survival: 88.5 },
            { time: 14, survival: 81.8 },
            { time: 28, survival: 75.7 },
          ],
        },
      ],
      number_at_risk: {
        times: [0, 1, 3, 7, 14, 28],
        rows: [
          { label: 'Non-sepsis', values: [160, 158, 154, 148, 141, 132] },
          { label: 'Sepsis', values: [140, 137, 129, 119, 105, 92] },
        ],
      },
    };
    const logrank = curve.logrank;
    return `
      <div class="sec-stack"><div class="lbl">${cohortText('Survival analysis')}</div><h2>${cohortText('Demo simulated KM preview')}</h2></div>
      <div class="note warn mt-12" data-demo-survival-simulated><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Seeded demo only')}</div><div class="d">${t('This Kaplan-Meier curve is a fixed simulated preview for the demo workspace. It exercises the chart, log-rank, and number-at-risk UI only; it is not derived from a local export and must not be used for manuscript claims.', '这条 Kaplan-Meier 曲线是演示工作区的固定模拟预览，只用于展示图表、log-rank 和风险人数表交互；它不是来自本地导出，不能用于稿件结论。')}</div></div></div>
      <div class="surv-card mt-14">
        <div class="surv-head">
          <div>
            <div class="eyebrow">${cohortText('Seeded demo only')}</div>
            <h3>${esc(cohortText(curve.label))}</h3>
            <p>${esc(cohortText(curve.time_label))} · ${t('event', '事件')} <span class="mono">${esc(curve.event_column)}</span> · ${t('time', '时间')} <span class="mono">${esc(curve.time_column)}</span></p>
          </div>
          <div class="surv-logrank">
            <span>${cohortText('Log-rank')}</span>
            <strong>χ² ${fmtNum(logrank.chi_square, 2)} · p = ${fmtP(logrank.p_value)}</strong>
            <small>${cohortText('Seeded demo only')}</small>
          </div>
        </div>
        ${cohortSurvivalChart(curve)}
        ${cohortSurvivalEffect(curve)}
        ${cohortRiskTable(curve)}
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
    demoBody: cohortSurvivalDemoBody,
  };
})();

