/* Owner: how the selected database's records change a study, in plan review.
   Renders the host's database capability review (exact / substitution /
   refusal) as stated by databases/research_requirements.py. It never decides
   a substitution, and approval stays with the plan-review owner. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }

  const UNITS = {
    patient: ['patients', '按患者'], hospitalization: ['hospitalizations', '按住院'], icu_stay: ['ICU stays', '按 ICU 入住'],
  };
  const MORTALITY = {
    hospital_death: ['In-hospital death', '院内死亡'],
    icu_death: ['Death in the ICU', 'ICU 内死亡'],
    death_within_72h_after_icu_discharge: ['Death up to 72 h after ICU discharge', '出 ICU 后 72 小时内死亡'],
  };
  const DEATH_TIME = {
    recorded_death_time: ['Recorded time of death', '记录的死亡时间'],
    icu_discharge_proxy: ['ICU discharge time used as a proxy for death time', '以 ICU 出科时间近似死亡时间'],
    last_observation_proxy: ['Last observation time used as a proxy for death time', '以最后观察时间近似死亡时间'],
    date: ['Date of death only', '只有死亡日期'],
    date_only: ['Date of death only', '只有死亡日期'],
    proxy: ['Discharge time standing in for the time of death', '以出院时间代替死亡时间'],
    discharge_proxy: ['Discharge time standing in for the time of death', '以出院时间代替死亡时间'],
  };
  // Why the database cannot answer the question as asked, per stable code.
  const MISSING = {
    family_requirements_missing: ['this kind of study has no registered requirements', '这类研究没有登记所需条件'],
    analysis_unit_unavailable: ['it has no identifier for the analysis unit', '没有所需分析单位的标识'],
    mortality_status_unavailable: ['it does not record death', '没有记录死亡结局'],
    hospital_los_unavailable: ['it has no hospital length of stay', '没有住院时长'],
    icu_los_unavailable: ['it has no ICU length of stay', '没有 ICU 住院时长'],
    time_zero_unavailable: ['it has no ICU admission time to start follow-up', '没有可作为起点的 ICU 入科时间'],
    event_time_unavailable: ['it has no time of death', '没有死亡时间'],
    censor_time_unavailable: ['it has no discharge time to end follow-up', '没有出院时间，无法确定随访终点'],
    observation_timing_unavailable: ['its measurements carry no times', '观测记录没有时间'],
    treatment_timing_unavailable: ['its treatments carry no times', '治疗记录没有时间'],
  };

  function databaseName(key) {
    const text = String(key || '');
    return window.EU_DATABASE_NAMES.name(text) || text;
  }
  function pair(map, key) {
    const row = map[String(key || '')];
    return row ? tr(row[0], row[1]) : String(key || '');
  }
  function conceptName(concept) {
    const row = (window.EU_CATALOG && window.EU_CATALOG.dict || {})[concept];
    return row ? tr(row[0], row[1]) : String(concept || '');
  }
  function threshold(value) {
    if (!value || typeof value !== 'object') return '';
    const unit = { age: tr(' years', ' 岁'), weight: ' kg' }[value.concept] || '';
    const operator = { '>=': '≥', '<=': '≤', '!=': '≠' }[value.operator] || String(value.operator || '');
    return `${conceptName(value.concept)} ${operator} ${Number(value.value)}${unit}`;
  }
  function sourceField(field) {
    if (!field || typeof field !== 'object') return '';
    return [field.table, field.column].filter(part => typeof part === 'string' && part).join('.');
  }
  function substitutionRow(item, database) {
    const capability = String(item && item.capability || '');
    let original = '';
    let substitute = '';
    let reason = '';
    if (capability === 'analysis_unit') {
      original = pair(UNITS, item.original);
      substitute = pair(UNITS, item.substitute);
      reason = tr(`${database} has no hospitalization identifier, so each ICU stay is analysed on its own.`,
        `${database} 没有住院标识，只能把每次 ICU 入住作为一个分析单位。`);
    } else if (capability === 'outcome_meaning') {
      original = pair(MORTALITY, item.original);
      substitute = pair(MORTALITY, item.substitute);
      reason = tr(`${database} records death differently from the question.`, `${database} 记录的死亡口径与问题不同。`);
    } else if (capability === 'event_time') {
      original = pair(DEATH_TIME, item.original);
      substitute = pair(DEATH_TIME, item.substitute);
      reason = tr(`${database} has no exact time of death.`, `${database} 没有精确的死亡时间。`);
    } else if (capability.startsWith('threshold:')) {
      original = threshold(item.original);
      substitute = threshold(item.substitute);
      reason = tr(`${database} records ${conceptName(item.original && item.original.concept)} in bands, so the cut-off moves to the nearest band edge.`,
        `${database} 的${conceptName(item.original && item.original.concept)}按区间记录，切点改为最近的区间边界。`);
    } else {
      return '';
    }
    const field = sourceField(item.source_field);
    const label = tr(`Planned: ${original}; used: ${substitute}`, `原定：${original}；改为：${substitute}`);
    return `<li><div class="gpi-source-swap" role="group" aria-label="${esc(label)}"><s>${esc(original)}</s><span aria-hidden="true">→</span><strong>${esc(substitute)}</strong></div><p>${esc(reason)}${field ? ` <code>${esc(field)}</code>` : ''}</p></li>`;
  }
  function missingText(item) {
    const code = String(item && item.code || '');
    const capability = String(item && item.capability || '');
    if (code === 'required_concept_unmapped') {
      const concept = capability.replace(/^concept:/, '');
      return tr(`it has no ${conceptName(concept)} data`, `没有${conceptName(concept)}的数据`);
    }
    if (code === 'numeric_threshold_not_resolvable') {
      const concept = capability.replace(/^threshold:/, '');
      return tr(`it records ${conceptName(concept)} in bands that cannot express this cut-off`,
        `${conceptName(concept)}按区间记录，无法按这个数值筛选`);
    }
    return MISSING[code] ? tr(MISSING[code][0], MISSING[code][1]) : '';
  }
  // review: the host's database_capability_review ({decision, unapplied_substitutions}).
  function render(review) {
    const decision = review && review.decision && typeof review.decision === 'object' ? review.decision : null;
    if (!decision || decision.status === 'exact') return '';
    const database = databaseName(decision.database);
    if (decision.status === 'refusal') {
      const reasons = [...new Set((Array.isArray(decision.missing) ? decision.missing : []).map(missingText).filter(Boolean))];
      return `<section class="gpi-source-adjustments is-refusal"><h4>${esc(tr(`This study cannot run on ${database}`, `这项研究在 ${database} 上做不了`))}</h4>${reasons.length
        ? `<ul>${reasons.map(text => `<li>${esc(text)}</li>`).join('')}</ul>` : ''}</section>`;
    }
    if (decision.status !== 'substitution') return '';
    const rows = (Array.isArray(decision.substitutions) ? decision.substitutions : [])
      .map(item => substitutionRow(item, database)).filter(Boolean);
    if (!rows.length) return '';
    const unapplied = Array.isArray(review.unapplied_substitutions) && review.unapplied_substitutions.length;
    return `<section class="gpi-source-adjustments"><h4>${esc(tr(`How ${database} changes this study`, `${database} 上的口径调整`))}</h4><ul>${rows.join('')}</ul><p class="${unapplied ? 'is-blocked' : ''}">${esc(unapplied
      ? tr('The plan does not use these yet, so it cannot be approved. Generate the plan again.', '计划还没有按这些口径修改，暂不能批准。请重新生成计划。')
      : tr('Approving the plan accepts these changes.', '批准计划即同意按以上口径分析。'))}</p></section>`;
  }

  window.EasyICU.guidedPi.declare('sourceAdjustments', { render, missingText, databaseName });
})();
