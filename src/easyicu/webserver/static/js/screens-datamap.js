/* Owner: the Data map page — what each of the six databases holds and which
   kinds of study its data can support.
   The host's data map (webserver/data_map.py, GET /api/data-map) owns every
   cell: database profiles, study-family decisions and concept mappings, all
   from packaged metadata, never from patient rows. This page lays them out
   and explains a decision when it is opened; it decides nothing itself. */
(function () {
  'use strict';
  const S = (window.SCREENS = window.SCREENS || {});
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function databases() { return window.EU_DATABASE_NAMES || null; }
  function dbName(key) { return (databases() && databases().name(key)) || key; }
  function dbAbbr(key) { return (databases() && databases().abbr(key)) || key; }
  function familyName(key) { return (window.EU_STUDY_FAMILY_NAMES && window.EU_STUDY_FAMILY_NAMES.name(key)) || key; }
  function familyOrder(key) { return window.EU_STUDY_FAMILY_NAMES ? window.EU_STUDY_FAMILY_NAMES.order(key) : 0; }
  function conceptName(key) {
    const row = window.EU_CATALOG && window.EU_CATALOG.dict ? window.EU_CATALOG.dict[key] : null;
    return Array.isArray(row) ? (window.EU_LANG === 'zh' ? row[1] || row[0] : row[0]) : '';
  }

  const PAGE = 40;
  const STUDY = {
    exact: ['Meets', '满足'], substitution: ['With substitutes', '需替换'], refused: ['Missing data', '不满足'],
  };
  const CONCEPT = { available: ['Mapped', '可用'], substituted: ['Substitute', '替代'], missing: ['Missing', '缺失'] };
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
  };
  for (const days of [28, 90, 365]) {
    SUBSTITUTES[`mort_${days}d`] = [`death within ${days} days of ICU admission`, `ICU 入科后 ${days} 天内死亡`];
    SUBSTITUTES[`followup_days_${days}d`] = [
      `days from ICU admission to death or censoring at ${days} days, with documented follow-up`,
      `有随访记录的从 ICU 入科到死亡或第 ${days} 天删失的天数`,
    ];
  }
  const CONCEPT_SUBSTITUTES = {
    capped_or_shifted: ['Values above a limit are capped or shifted', '超过上限的值被截断或平移'],
    binned: ['Recorded in bands', '以分段区间记录'],
  };

  const state = { view: null, loading: false, failed: null, offset: 0, open: '' };

  function chip(table, status) {
    const label = table[status];
    return `<span class="dmap-chip is-${esc(label ? status : 'unknown')}">${esc(label ? tr(...label) : tr('Unknown', '未知'))}</span>`;
  }
  function phrase(table, key) { return table[key] ? tr(...table[key]) : key; }

  // The host names each registered source's kind (the official-demo catalogue
  // decides a public demo); the words are the data pages' own labels.
  const SOURCE_KINDS = {
    unrecorded: ['own', '自有'], public_demo: ['public demo', '公开 Demo'], synthetic: ['synthetic test data', '合成测试数据'],
  };
  function registeredSources(row) {
    const counts = {};
    (Array.isArray(row.sources) ? row.sources : []).forEach(source => {
      const kind = source && SOURCE_KINDS[source.kind] ? source.kind : 'unrecorded';
      counts[kind] = (counts[kind] || 0) + 1;
    });
    const parts = Object.keys(SOURCE_KINDS).filter(kind => counts[kind])
      .map(kind => tr(`${counts[kind]} ${SOURCE_KINDS[kind][0]}`, `${SOURCE_KINDS[kind][1]} ${counts[kind]}`));
    if (!parts.length) return tr('Not registered on this computer', '本机未登记');
    const total = row.source_next_offset != null ? tr(` (${row.source_total} in all)`, `（共 ${row.source_total} 个）`) : '';
    return tr(`Registered on this computer: ${parts.join(' · ')}${total}`, `本机已登记：${parts.join(' · ')}${total}`);
  }

  function databaseCards(rows) {
    return `<div class="dmap-dbs">${rows.map(row => {
      const release = row.profile && row.profile.reference_release;
      return `<article class="dmap-db"><h3>${esc(dbName(row.database))}</h3>
        ${release ? `<p>${esc(tr(`Reference release ${release}`, `参考版本 ${release}`))}</p>` : ''}
        <p>${esc(registeredSources(row))}</p>
        ${row.real_data_acceptance === 'not_accepted' ? `<span class="dmap-chip is-pending">${esc(tr('Not yet accepted on real data', '尚未在真实数据上验收'))}</span>` : ''}</article>`;
    }).join('')}</div>`;
  }

  function studyMatrix(rows) {
    const families = [...new Set(rows.flatMap(row => (row.studies || []).map(study => study.family)))]
      .sort((a, b) => familyOrder(a) - familyOrder(b) || a.localeCompare(b));
    const cell = (row, family) => (row.studies || []).find(study => study.family === family);
    const body = families.map(family => `<tr><th scope="row">${esc(familyName(family))}</th>${rows.map(row => {
      const study = cell(row, family);
      if (!study) return '<td>—</td>';
      const key = `${row.database}|${family}`;
      const detail = study.status !== 'exact';
      return `<td>${detail ? `<button type="button" data-dmap-study="${esc(key)}" aria-expanded="${state.open === key}">${chip(STUDY, study.status)}</button>` : chip(STUDY, study.status)}</td>`;
    }).join('')}</tr>`).join('');
    return `<div class="dmap-scroll"><table class="dmap-table dmap-matrix"><thead><tr><th scope="col">${esc(tr('Study type', '研究类型'))}</th>${rows.map(row =>
      `<th scope="col" title="${esc(dbName(row.database))}">${esc(dbAbbr(row.database))}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>
      <div data-dmap-study-detail>${studyDetail(rows)}</div>`;
  }

  function studyDetail(rows) {
    if (!state.open) return '';
    const [database, family] = state.open.split('|');
    const row = rows.find(item => item.database === database);
    const study = row && (row.studies || []).find(item => item.family === family);
    const decision = study && study.decision;
    if (!decision) return '';
    const lines = [];
    (decision.missing || []).forEach(item => lines.push(tr(`No ${phrase(CAPABILITIES, item.capability)} is recorded.`, `没有记录${phrase(CAPABILITIES, item.capability)}。`)));
    (decision.substitutions || []).forEach(item => lines.push(tr(
      `${phrase(CAPABILITIES, item.capability)} uses ${phrase(SUBSTITUTES, item.substitute)}${item.source_field && item.source_field.table ? ` (${item.source_field.table}.${item.source_field.column})` : ''}.`,
      `${phrase(CAPABILITIES, item.capability)}使用${phrase(SUBSTITUTES, item.substitute)}${item.source_field && item.source_field.table ? `（${item.source_field.table}.${item.source_field.column}）` : ''}。`)));
    return `<div class="dmap-detail" role="region" aria-label="${esc(`${familyName(family)} · ${dbName(database)}`)}"><h3>${esc(familyName(family))} · ${esc(dbName(database))} ${chip(STUDY, study.status)}</h3>
      ${lines.length ? `<ul>${lines.map(line => `<li>${esc(line)}</li>`).join('')}</ul>` : ''}
      ${study.status === 'substitution' ? `<p>${esc(tr('A plan on this database states the substitute and needs your confirmation before it runs.', '在这个数据库上做计划时，会写明替代内容，需要你确认后才运行。'))}</p>` : ''}</div>`;
  }

  function conceptTable(view, rows) {
    const concepts = (rows[0] && rows[0].concepts || []).map(item => item.concept);
    const lookup = (row, concept) => (row.concepts || []).find(item => item.concept === concept) || null;
    const body = concepts.map(concept => {
      const name = conceptName(concept);
      const unit = (lookup(rows[0], concept) || {}).unit;
      return `<tr><th scope="row"><code>${esc(concept)}</code>${name ? `<span>${esc(name)}</span>` : ''}</th><td class="dmap-unit">${esc(unit && unit !== 'boolean' ? unit : '')}</td>${rows.map(row => {
        const cell = lookup(row, concept);
        if (!cell) return '<td>—</td>';
        const sub = cell.substitute && CONCEPT_SUBSTITUTES[cell.substitute.kind];
        return `<td${sub ? ` title="${esc(tr(...sub))}"` : ''}>${chip(CONCEPT, cell.status)}</td>`;
      }).join('')}</tr>`;
    }).join('');
    const total = Number(view.concept_total || 0);
    const first = total ? state.offset + 1 : 0;
    const last = Math.min(total, state.offset + concepts.length);
    const prev = state.offset > 0;
    const next = view.concept_next_offset != null;
    return `<div class="dmap-scroll"><table class="dmap-table dmap-concepts"><thead><tr><th scope="col">${esc(tr('Variable', '变量'))}</th><th scope="col">${esc(tr('Unit', '单位'))}</th>${rows.map(row =>
      `<th scope="col" title="${esc(dbName(row.database))}">${esc(dbAbbr(row.database))}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>
      <div class="dmap-pager"><button type="button" class="btn sm" data-dmap-page="prev"${prev ? '' : ' disabled'}>${esc(tr('Previous', '上一页'))}</button>
        <span>${esc(tr(`${first}–${last} of ${total}`, `${first}–${last} / 共 ${total} 个`))}</span>
        <button type="button" class="btn sm" data-dmap-page="next"${next ? '' : ' disabled'}>${esc(tr('Next', '下一页'))}</button>
        <button type="button" class="btn sm ghost" data-nav="dictionary">${esc(tr('Open the data dictionary', '在数据字典中查看'))}</button></div>`;
  }

  function bodyHtml() {
    if (state.failed) return `<p class="dmap-note" role="alert">${esc(tr('The data map could not be read right now.', '暂时读不到数据地图。'))} <button type="button" class="btn sm" data-dmap-retry>${esc(tr('Try again', '重试'))}</button></p>`;
    if (!state.view) return `<p class="dmap-note" role="status">${esc(tr('Reading…', '正在读取…'))}</p>`;
    const rows = Array.isArray(state.view.databases) ? state.view.databases : [];
    return `${databaseCards(rows)}
      <section class="dmap-section"><h2>${esc(tr('Study types', '研究类型'))}</h2>
        <p class="dmap-note">${esc(tr('Whether the database records the basic data a study type needs, such as event and censoring times. Whether a particular study can run is settled in plan review.', '只看数据库是否记录了这类研究需要的基本数据，例如事件时间和删失时间；具体研究能否开展，在计划审查时确定。'))}</p>
        ${studyMatrix(rows)}</section>
      <section class="dmap-section"><h2>${esc(tr('Variables', '变量'))}</h2>
        <p class="dmap-note">${esc(tr('From EasyICU’s variable mappings, not from how complete the data are.', '按 EasyICU 的变量映射整理，不代表数据的实际完整程度。'))}</p>
        ${conceptTable(state.view, rows)}</section>`;
  }

  function paint() {
    const root = typeof document !== 'undefined' ? document.querySelector('[data-dmap-root]') : null;
    if (root) root.innerHTML = bodyHtml();
  }
  async function load() {
    if (state.loading || typeof api().loadDataMap !== 'function') return;
    state.loading = true; state.failed = null;
    try { state.view = await api().loadDataMap({ concept_offset: state.offset, concept_limit: PAGE }); } catch (failure) { state.failed = failure; }
    state.loading = false;
    paint();
  }

  S.datamap = {
    section: 'datamap', nav: 'datamap',
    get crumbs() { return [tr('Home', '首页'), tr('Data map', '数据地图')]; },
    actionHtml: '',
    render() {
      return `<div class="page-head dmap-head"><div class="eyebrow">${esc(tr('Data', '数据'))}</div><h1>${esc(tr('Data map', '数据地图'))}</h1>
        <p class="lead">${esc(tr('What each of the six ICU databases records and which kinds of study its data support.', '六个 ICU 数据库各自记录了什么、具备哪些研究的数据条件。'))}</p></div>
        <div class="dmap" data-dmap-root>${bodyHtml()}</div>`;
    },
    afterRender(root) {
      if (!state.view && !state.failed) void load();
      const host = root.querySelector('[data-dmap-root]');
      if (!host || host.dataset.bound) return;
      host.dataset.bound = '1';
      host.addEventListener('click', event => {
        const study = event.target.closest('[data-dmap-study]');
        if (study) {
          state.open = state.open === study.dataset.dmapStudy ? '' : study.dataset.dmapStudy;
          paint();
          // The explanation sits below the matrix; bring it into view on short windows.
          const detail = state.open && typeof host.querySelector === 'function' ? host.querySelector('.dmap-detail') : null;
          if (detail && typeof detail.scrollIntoView === 'function') detail.scrollIntoView({ block: 'nearest' });
          return;
        }
        const page = event.target.closest('[data-dmap-page]');
        if (page && !page.disabled) {
          state.offset = Math.max(0, state.offset + (page.dataset.dmapPage === 'next' ? PAGE : -PAGE));
          void load(); return;
        }
        if (event.target.closest('[data-dmap-retry]')) void load();
      });
    },
  };
})();
