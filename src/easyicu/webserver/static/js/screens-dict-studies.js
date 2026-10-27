/* Owner: the Data Dictionary's study-type view — which kinds of study each
   of the six databases records the basic data for.
   The host's data map (webserver/data_map.py, GET /api/data-map) owns every
   cell, from packaged metadata, never from patient rows. This view lays the
   study-family decisions out and explains one when it is opened; it decides
   nothing itself. screens-dict.js hosts it and owns the page.
   Interface: window.EU_DICT_STUDIES = { html(), bind(host), meaning(key) }. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function databases() { return window.EU_DATABASE_NAMES || null; }
  function dbName(key) { return (databases() && databases().name(key)) || key; }
  function dbAbbr(key) { return (databases() && databases().abbr(key)) || key; }
  function familyName(key) { return (window.EU_STUDY_FAMILY_NAMES && window.EU_STUDY_FAMILY_NAMES.name(key)) || key; }
  function familyOrder(key) { return window.EU_STUDY_FAMILY_NAMES ? window.EU_STUDY_FAMILY_NAMES.order(key) : 0; }

  const STUDY = {
    exact: ['Meets', '满足'], substitution: ['With substitutes', '需替换'], refused: ['Missing data', '不满足'],
  };
  const CAPABILITIES = {
    outcome_meaning: ['outcome definition', '结局定义'],
    event_time: ['event time', '事件发生时间'], censor_time: ['censoring time', '删失时间'],
    mortality: ['mortality', '死亡结局'], icu_admission: ['ICU admission time', 'ICU 入科时间'],
    icu_discharge: ['ICU discharge time', 'ICU 出科时间'], hospital_discharge: ['hospital discharge time', '出院时间'],
  };
  const SUBSTITUTES = {
    icu_death: ['death in the ICU', 'ICU 内死亡'],
    death_within_72h_after_icu_discharge: ['death up to 72 h after ICU discharge', '出 ICU 后 72 小时内死亡'],
    date: ['a date only', '只有日期'], last_observation_proxy: ['the last observation time as a proxy', '最后一次观测时间作为近似'],
    icu_discharge_proxy: ['ICU discharge time as a proxy for death time', 'ICU 出科时间作为死亡时间近似'],
    icu_discharge: ['ICU discharge, excluding later deaths', 'ICU 出科时间，不计之后的死亡'],
    hospital_discharge: ['hospital discharge, counting deaths after ICU discharge', '出院时间，计入出 ICU 后的死亡'],
  };
  for (const days of [28, 90, 365]) {
    SUBSTITUTES[`mort_${days}d`] = [`death within ${days} days of ICU admission`, `ICU 入科后 ${days} 天内死亡`];
    SUBSTITUTES[`followup_days_${days}d`] = [
      `days from ICU admission to death or censoring at ${days} days, with documented follow-up`,
      `有随访记录的从 ICU 入科到死亡或第 ${days} 天删失的天数`,
    ];
  }

  const state = { view: null, loading: false, failed: null, open: '' };

  function chip(status) {
    const label = STUDY[status];
    return `<span class="dst-chip is-${esc(label ? status : 'unknown')}">${esc(label ? tr(...label) : tr('Unknown', '未知'))}</span>`;
  }
  function phrase(table, key) { return table[key] ? tr(...table[key]) : key; }
  function rows() { return state.view && Array.isArray(state.view.databases) ? state.view.databases : []; }

  function matrix(list) {
    const families = [...new Set(list.flatMap(row => (row.studies || []).map(study => study.family)))]
      .sort((a, b) => familyOrder(a) - familyOrder(b) || a.localeCompare(b));
    const cell = (row, family) => (row.studies || []).find(study => study.family === family);
    const body = families.map(family => `<tr><th scope="row">${esc(familyName(family))}</th>${list.map(row => {
      const study = cell(row, family);
      if (!study) return '<td>—</td>';
      const key = `${row.database}|${family}`;
      return `<td>${study.status !== 'exact' ? `<button type="button" data-dst-study="${esc(key)}" aria-expanded="${state.open === key}">${chip(study.status)}</button>` : chip(study.status)}</td>`;
    }).join('')}</tr>`).join('');
    return `<div class="dst-scroll"><table class="dst-table"><thead><tr><th scope="col">${esc(tr('Study type', '研究类型'))}</th>${list.map(row =>
      `<th scope="col" title="${esc(dbName(row.database))}">${esc(dbAbbr(row.database))}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>
      <div data-dst-detail>${detail(list)}</div>`;
  }

  function detail(list) {
    if (!state.open) return '';
    const [database, family] = state.open.split('|');
    const row = list.find(item => item.database === database);
    const study = row && (row.studies || []).find(item => item.family === family);
    const decision = study && study.decision;
    if (!decision) return '';
    const lines = [];
    (decision.missing || []).forEach(item => lines.push(tr(`No ${phrase(CAPABILITIES, item.capability)} is recorded.`, `没有记录${phrase(CAPABILITIES, item.capability)}。`)));
    (decision.substitutions || []).forEach(item => lines.push(tr(
      `${phrase(CAPABILITIES, item.capability)} uses ${phrase(SUBSTITUTES, item.substitute)}${item.source_field && item.source_field.table ? ` (${item.source_field.table}.${item.source_field.column})` : ''}.`,
      `${phrase(CAPABILITIES, item.capability)}使用${phrase(SUBSTITUTES, item.substitute)}${item.source_field && item.source_field.table ? `（${item.source_field.table}.${item.source_field.column}）` : ''}。`)));
    return `<div class="dst-detail" role="region" aria-label="${esc(`${familyName(family)} · ${dbName(database)}`)}"><h3>${esc(familyName(family))} · ${esc(dbName(database))} ${chip(study.status)}</h3>
      ${lines.length ? `<ul>${lines.map(line => `<li>${esc(line)}</li>`).join('')}</ul>` : ''}
      ${study.status === 'substitution' ? `<p>${esc(tr('A plan on this database states the substitute and needs your confirmation before it runs.', '在这个数据库上做计划时，会写明替代内容，需要你确认后才运行。'))}</p>` : ''}</div>`;
  }

  function body() {
    if (state.failed) return `<p class="dst-note" role="alert">${esc(tr('The study types could not be read right now.', '暂时读不到研究类型。'))} <button type="button" class="btn sm" data-dst-retry>${esc(tr('Try again', '重试'))}</button></p>`;
    if (!state.view) return `<p class="dst-note" role="status">${esc(tr('Reading…', '正在读取…'))}</p>`;
    return `<p class="dst-note">${esc(tr('Whether the database records the basic data a study type needs, such as event and censoring times. Whether a particular study can run is settled in plan review.', '只看数据库是否记录了这类研究需要的基本数据，例如事件时间和删失时间；具体研究能否开展，在计划审查时确定。'))}</p>
      ${matrix(rows())}`;
  }

  let mounted = null;
  function paint() { if (mounted) mounted.innerHTML = body(); }
  async function load() {
    if (state.loading || typeof api().loadDataMap !== 'function') return;
    state.loading = true; state.failed = null;
    // The decisions arrive with every database row; one variable is the smallest page.
    try { state.view = await api().loadDataMap({ concept_offset: 0, concept_limit: 1 }); } catch (failure) { state.failed = failure; }
    state.loading = false;
    paint();
  }

  window.EU_DICT_STUDIES = {
    html() { return `<div class="dst" data-dst-root>${body()}</div>`; },
    bind(root) {
      mounted = root.querySelector('[data-dst-root]');
      if (!mounted) return;
      if (!state.view && !state.failed) void load();
      if (mounted.dataset.bound) return;
      mounted.dataset.bound = '1';
      mounted.addEventListener('click', event => {
        const study = event.target.closest('[data-dst-study]');
        if (study) {
          state.open = state.open === study.dataset.dstStudy ? '' : study.dataset.dstStudy;
          paint();
          // The explanation sits below the matrix; bring it into view on short windows.
          const shown = state.open && typeof mounted.querySelector === 'function' ? mounted.querySelector('.dst-detail') : null;
          if (shown && typeof shown.scrollIntoView === 'function') shown.scrollIntoView({ block: 'nearest' });
          return;
        }
        if (event.target.closest('[data-dst-retry]')) void load();
      });
    },
    // The words for a mortality meaning or timing the host names by key.
    meaning(key) { return SUBSTITUTES[key] ? tr(...SUBSTITUTES[key]) : ''; },
  };
})();
