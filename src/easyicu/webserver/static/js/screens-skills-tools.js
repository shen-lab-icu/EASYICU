/* Owner: the working tools on the data-definition and planning Skills' pages.
   The host owns every answer: concept definitions and the extraction metadata
   come from research_agent/concept_catalog.py, sample sizes from
   research_agent/skill_packages, all through routes/clinician_skills.py.
   This file lays them out and keeps what the clinician typed; it computes
   nothing itself. Export runs in Data Extraction, which the Skill page opens.
   screens-skills.js places a tool on its Skill's page. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function icon(name, size) { return typeof window.icon === 'function' ? window.icon(name, size, 1.6) : ''; }
  function names() { return window.EU_DATABASE_NAMES || null; }
  function databaseName(key) { return (names() && names().name(key)) || key; }

  const DEFINITIONS = 'clinical-concept-definitions';
  const EXTRACTION = 'six-database-cohort-extraction';
  const SAMPLE_SIZE = 'sample-size-and-power';
  const TOOLS = new Set([DEFINITIONS, EXTRACTION, SAMPLE_SIZE]);

  // Availability from packaged mappings, not from patient rows.
  const STATUS = {
    full: ['Available', '可用'], degraded: ['Partly available', '部分可用'],
    blocked: ['Not available', '不可用'], unavailable: ['Not available', '不可用'],
  };
  const CONTRACT_STATUS = {
    source_bound_golden: ['Checked against its published source', '已按出处核对'],
    experimental: ['Experimental', '实验性'],
  };
  const PLANNING_ERRORS = {
    'planning.null_effect': ['The two groups are assumed equal, so there is no difference to detect.', '两组假设相同，没有可检出的差异。'],
    'planning.power_not_above_alpha': ['Power must be larger than the significance level.', '把握度必须大于检验水准。'],
    'planning.incompatible_r2': ['This R² is not possible at this outcome rate.', '在这个结局发生率下，这个 R² 不可能达到。'],
    'planning.invalid_parameter_count': ['Enter a whole number of candidate parameters.', '候选参数个数需要是整数。'],
    'planning.numerical_failure': ['No sample size reaches this power under these assumptions.', '在这些假设下算不出达到该把握度的样本量。'],
  };
  const FAILURES = {
    'definition.database_unknown': ['This database has no definition record.', '这个数据库没有定义记录。'],
  };

  // One shape per calculation: [field, en, zh, unit, default, optional].
  // 'pct' fields are typed as percentages and sent as proportions.
  const CALCULATIONS = {
    two_proportions: {
      label: ['Two proportions', '两组率比较'],
      fields: [['p1', 'Rate in group 1', '组 1 发生率', 'pct', 30], ['p2', 'Rate in group 2', '组 2 发生率', 'pct', 20],
        ['alpha', 'Significance level α', '检验水准 α', 'num', 0.05], ['power', 'Power', '把握度', 'pct', 80],
        ['allocation_ratio', 'Group 2 : group 1 size', '组 2 : 组 1 人数比', 'num', 1]],
    },
    two_means: {
      label: ['Two means', '两组均值比较'],
      fields: [['difference', 'Difference in means', '均值差', 'num', 5], ['standard_deviation', 'Standard deviation', '标准差', 'num', 10],
        ['alpha', 'Significance level α', '检验水准 α', 'num', 0.05], ['power', 'Power', '把握度', 'pct', 80],
        ['allocation_ratio', 'Group 2 : group 1 size', '组 2 : 组 1 人数比', 'num', 1]],
    },
    survival_events: {
      label: ['Survival (events needed)', '生存分析所需事件数'],
      fields: [['hazard_ratio', 'Hazard ratio', '风险比 HR', 'num', 0.75], ['allocation_fraction', 'Share in the exposed group', '暴露组占比', 'pct', 50],
        ['event_fraction', 'Expected share with the event', '预期发生事件的比例', 'pct', '', true],
        ['alpha', 'Significance level α', '检验水准 α', 'num', 0.05], ['power', 'Power', '把握度', 'pct', 80]],
    },
    prediction_binary: {
      label: ['Prediction model (binary outcome)', '二分类预测模型'],
      fields: [['parameters', 'Candidate predictor parameters', '候选预测参数个数', 'int', 10], ['cox_snell_r2', 'Expected Cox–Snell R²', '预期 Cox–Snell R²', 'num', 0.1],
        ['prevalence', 'Outcome rate', '结局发生率', 'pct', 20], ['shrinkage', 'Target shrinkage', '目标收缩系数', 'num', 0.9],
        ['r2_optimism', 'Allowed R² optimism', '允许的 R² 乐观偏差', 'num', 0.05], ['prevalence_margin', 'Margin for the outcome rate', '结局发生率误差范围', 'num', 0.05],
        ['confidence', 'Confidence', '置信水平', 'pct', 95]],
    },
  };

  const state = {
    database: (names() && names().keys()[0]) || 'miiv',
    query: '', concept: '',
    definitions: new Map(), extraction: new Map(), loading: new Set(), failed: new Map(),
    calculation: 'two_proportions', values: {}, result: null, error: '', busy: false,
  };

  function plainName(name) { const text = String(name || '').replace(/^[^\p{L}\p{N}]+/u, ''); return text || String(name || ''); }
  function moduleName(key) {
    const groups = (window.EU_CATALOG && Array.isArray(window.EU_CATALOG.groups)) ? window.EU_CATALOG.groups : [];
    const row = groups.find(group => group[0] === key);
    return row ? plainName(window.EU_LANG === 'zh' ? row[2] : row[1]) : key;
  }
  function statusChip(status) {
    const label = STATUS[status] || ['Unknown', '未知'];
    return `<span class="eusk-tool-status is-${esc(STATUS[status] ? status : 'unknown')}">${esc(tr(...label))}</span>`;
  }
  function databaseSelect() {
    const keys = names() ? names().keys() : [state.database];
    return `<label class="eusk-tool-field"><span>${esc(tr('Database', '数据库'))}</span><select data-skt-database>${keys.map(key =>
      `<option value="${esc(key)}"${key === state.database ? ' selected' : ''}>${esc(databaseName(key))}</option>`).join('')}</select></label>`;
  }
  function failureHtml(failure) {
    const known = FAILURES[failure && failure.code];
    return `<p class="eusk-tool-note" role="alert">${esc(known ? tr(...known) : tr('This could not be read right now.', '暂时读不到。'))}</p>`;
  }
  function loadingHtml() { return `<p class="eusk-tool-note" role="status">${esc(tr('Reading…', '正在读取…'))}</p>`; }

  // ── Concept definitions ──────────────────────────────────────────────
  function definitionRows() {
    const view = state.definitions.get(state.database);
    const items = view && Array.isArray(view.items) ? view.items : [];
    const query = state.query.trim().toLocaleLowerCase();
    return query ? items.filter(item => [item.concept, item.description].some(value => String(value || '').toLocaleLowerCase().includes(query))) : items;
  }
  function availability(item) {
    const row = item && item.databases && item.databases[state.database];
    return row && row.availability ? row.availability : null;
  }
  function listHtml() {
    const rows = definitionRows();
    if (!rows.length) return `<p class="eusk-tool-note">${esc(tr('No definition matches.', '没有匹配的定义。'))}</p>`;
    return `<ul>${rows.map(item => {
      const status = (availability(item) || {}).status;
      return `<li><button type="button" data-skt-concept="${esc(item.concept)}"${item.concept === state.concept ? ' aria-current="true"' : ''}><strong>${esc(item.concept)}</strong><span>${esc(item.description || '')}</span>${statusChip(status)}</button></li>`;
    }).join('')}</ul>`;
  }
  function codeList(values) { return (Array.isArray(values) ? values : []).map(value => `<code>${esc(value)}</code>`).join(' '); }
  function conceptHtml() {
    const items = (state.definitions.get(state.database) || {}).items || [];
    const item = items.find(row => row.concept === state.concept);
    if (!item) return `<p class="eusk-tool-note">${esc(tr('Choose a definition to see its parts and where it can be built.', '选择一个定义，查看它的组成和能否在这个数据库中构建。'))}</p>`;
    const avail = availability(item) || {};
    const db = (item.databases || {})[state.database] || {};
    const components = db.component_availability && typeof db.component_availability === 'object' ? db.component_availability : {};
    const contract = item.clinical_contract || null;
    const window_ = item.time_window || {};
    const facts = [];
    if (Array.isArray(avail.degraded_dependencies) && avail.degraded_dependencies.length) facts.push([tr('Partly available inputs', '部分可用的输入'), codeList(avail.degraded_dependencies)]);
    if (Array.isArray(avail.missing_dependencies) && avail.missing_dependencies.length) facts.push([tr('Missing inputs', '缺少的输入'), codeList(avail.missing_dependencies)]);
    if (Array.isArray(avail.source_missing_tables) && avail.source_missing_tables.length) facts.push([tr('Missing source tables', '缺少的源表'), codeList(avail.source_missing_tables)]);
    if (window_.study_observation_window === 'requires_explicit_selection') facts.push([tr('Time window', '时间窗'), esc(tr('Set in the study plan', '在研究计划中指定'))]);
    if (Number.isFinite(window_.pre_admission_lookback_hours)) facts.push([tr('Look-back before admission', '入院前回溯'), esc(tr(`${window_.pre_admission_lookback_hours} h`, `${window_.pre_admission_lookback_hours} 小时`))]);
    const componentRows = Object.keys(components).map(key => `<tr><td><code>${esc(key)}</code></td><td>${statusChip(components[key] && components[key].status)}</td></tr>`).join('');
    const contractStatus = contract && CONTRACT_STATUS[contract.status];
    return `<header><h3>${esc(item.concept)}</h3>${statusChip(avail.status)}</header><p>${esc(item.description || '')}</p>
      ${facts.length ? `<dl>${facts.map(([term, value]) => `<dt>${esc(term)}</dt><dd>${value}</dd>`).join('')}</dl>` : ''}
      ${componentRows ? `<table class="eusk-tool-table"><thead><tr><th scope="col">${esc(tr('Component', '组成'))}</th><th scope="col">${esc(databaseName(state.database))}</th></tr></thead><tbody>${componentRows}</tbody></table>` : ''}
      ${contract ? `<section class="eusk-tool-contract"><h4>${esc(contract.definition || tr('Clinical definition', '临床定义'))}${contract.version ? ` <small>${esc(contract.version)}</small>` : ''}</h4>
        ${contract.source_id ? `<p>${esc(contract.source_id)}${contract.source_table ? ` · ${esc(contract.source_table)}` : ''}</p>` : ''}
        ${contractStatus ? `<p>${esc(tr(...contractStatus))}${contract.reference_implementation ? ` · ${esc(contract.reference_implementation)}` : ''}</p>` : ''}</section>` : ''}`;
  }
  function definitionsHtml() {
    const view = state.definitions.get(state.database);
    const body = state.failed.has(`definitions:${state.database}`) ? failureHtml(state.failed.get(`definitions:${state.database}`))
      : !view ? loadingHtml()
        : `<div class="eusk-tool-split"><div class="eusk-tool-list" data-skt-list>${listHtml()}</div><div class="eusk-tool-detail" data-skt-detail>${conceptHtml()}</div></div>`;
    return `<div class="eusk-tool-bar">${databaseSelect()}<label class="eusk-tool-search">${icon('search', 15)}<input type="search" data-skt-query value="${esc(state.query)}" placeholder="${esc(tr('Search definitions…', '搜索定义…'))}" aria-label="${esc(tr('Search definitions', '搜索定义'))}" autocomplete="off"></label></div>
      <p class="eusk-tool-note">${esc(tr('Availability follows EasyICU’s variable mappings, not how complete the data are.', '可用性按 EasyICU 的变量映射判断，不代表数据的实际完整程度。'))}</p>${body}`;
  }

  // ── Extraction ───────────────────────────────────────────────────────
  function extractionHtml() {
    const view = state.extraction.get(state.database);
    const failed = state.failed.get(`extraction:${state.database}`);
    let body;
    if (failed) body = failureHtml(failed);
    else if (!view) body = loadingHtml();
    else {
      const row = (Array.isArray(view.databases) ? view.databases : []).find(item => item.database === state.database) || {};
      const modules = row.modules && typeof row.modules === 'object' ? row.modules : {};
      const rows = Object.keys(modules).map(key => {
        const counts = { full: 0, degraded: 0, other: 0 };
        Object.values(modules[key].availability || {}).forEach(item => {
          const status = item && item.status;
          if (status === 'full') counts.full += 1; else if (status === 'degraded') counts.degraded += 1; else counts.other += 1;
        });
        return `<tr><th scope="row">${esc(moduleName(key))}</th><td>${counts.full}</td><td>${counts.degraded}</td><td>${counts.other}</td></tr>`;
      }).join('');
      body = rows ? `<table class="eusk-tool-table"><thead><tr><th scope="col">${esc(tr('Module', '模块'))}</th><th scope="col">${esc(tr(...STATUS.full))}</th><th scope="col">${esc(tr(...STATUS.degraded))}</th><th scope="col">${esc(tr(...STATUS.blocked))}</th></tr></thead><tbody>${rows}</tbody></table>`
        : `<p class="eusk-tool-note">${esc(tr('No modules are recorded for this database.', '这个数据库没有记录可提取的模块。'))}</p>`;
    }
    return `<div class="eusk-tool-bar">${databaseSelect()}</div>
      <p class="eusk-tool-note">${esc(tr('Variables per module, by EasyICU’s mappings for this database. Export runs in Data Extraction.', '按 EasyICU 对这个数据库的变量映射统计各模块的变量数；导出在「数据提取」中进行。'))}</p>${body}`;
  }

  // ── Sample size ──────────────────────────────────────────────────────
  function fieldValue(name, fallback) {
    const values = state.values[state.calculation] || {};
    return Object.prototype.hasOwnProperty.call(values, name) ? values[name] : fallback;
  }
  function percent(value) { return `${Math.round(value * 1000) / 10}%`; }
  function count(value) { return Number.isFinite(value) ? Number(value).toLocaleString(window.EU_LANG === 'zh' ? 'zh-CN' : 'en-US') : '—'; }
  function resultHtml() {
    const result = state.result;
    if (state.error) return `<p class="eusk-tool-note" role="alert">${esc(state.error)}</p>`;
    if (!result) return '';
    const stats = [];
    if (Array.isArray(result.group_sizes)) stats.push([tr('Per group', '每组人数'), result.group_sizes.map(count).join(' / ')]);
    if (Number.isFinite(result.total_n)) stats.push([tr('Total', '合计'), count(result.total_n)]);
    if (Number.isFinite(result.required_events)) stats.push([tr('Events needed', '所需事件数'), count(result.required_events)]);
    if (result.required_events != null && result.total_n == null) stats.push([tr('Total', '合计'), tr('Give the expected event share', '填写预期事件比例后给出')]);
    if (Number.isFinite(result.minimum_n)) stats.push([tr('Minimum sample', '最少样本量'), count(result.minimum_n)]);
    if (Number.isFinite(result.expected_events)) stats.push([tr('Expected events', '预期事件数'), count(Math.round(result.expected_events))]);
    if (Number.isFinite(result.events_per_parameter)) stats.push([tr('Events per parameter', '每参数事件数'), (Math.round(result.events_per_parameter * 10) / 10).toString()]);
    if (Number.isFinite(result.achieved_power)) stats.push([tr('Achieved power', '实际把握度'), percent(result.achieved_power)]);
    return `<div class="eusk-tool-result" role="status"><dl>${stats.map(([term, value]) => `<div><dt>${esc(term)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>
      <p class="eusk-tool-note">${esc(tr('Computed from the assumptions you entered, for planning; it is not a study result.', '按你填写的假设计算，用于研究规划，不是研究结果。'))}${result.method ? ` <code>${esc(result.method)}</code>` : ''}${result.implementation ? ` <code>${esc(result.implementation)}</code>` : ''}</p></div>`;
  }
  function sampleSizeHtml() {
    const spec = CALCULATIONS[state.calculation];
    const options = Object.keys(CALCULATIONS).map(key => `<option value="${key}"${key === state.calculation ? ' selected' : ''}>${esc(tr(...CALCULATIONS[key].label))}</option>`).join('');
    const fields = spec.fields.map(([name, en, zh, unit, fallback, optional]) => `<label class="eusk-tool-field"><span>${esc(tr(en, zh))}${unit === 'pct' ? ' (%)' : ''}${optional ? ` <small>${esc(tr('optional', '可不填'))}</small>` : ''}</span>
      <input type="number" inputmode="decimal" step="any" data-skt-value="${esc(name)}" value="${esc(String(fieldValue(name, fallback)))}"${optional ? '' : ' required'}></label>`).join('');
    return `<form class="eusk-tool-form" data-skt-calculate novalidate><label class="eusk-tool-field eusk-tool-wide"><span>${esc(tr('Comparison', '比较类型'))}</span><select data-skt-calculation>${options}</select></label>
      ${fields}<div class="eusk-tool-actions"><button type="submit" class="eusk-tool-primary"${state.busy ? ' disabled' : ''}>${esc(state.busy ? tr('Calculating…', '正在计算…') : tr('Calculate', '计算'))}</button></div></form>
      <div data-skt-result>${resultHtml()}</div>`;
  }
  function requestBody() {
    const spec = CALCULATIONS[state.calculation];
    const body = { calculation: state.calculation };
    for (const [name, , , unit, fallback, optional] of spec.fields) {
      const raw = String(fieldValue(name, fallback)).trim();
      if (!raw) { if (optional) continue; return null; }
      const value = Number(raw);
      if (!Number.isFinite(value)) return null;
      body[name] = unit === 'pct' ? value / 100 : unit === 'int' ? (Number.isInteger(value) ? value : NaN) : value;
      if (Number.isNaN(body[name])) return null;
    }
    return body;
  }
  async function calculate() {
    if (state.busy || typeof api().calculateSampleSize !== 'function') return;
    const body = requestBody();
    state.result = null;
    if (!body) { state.error = tr('Fill in every value with a number.', '请为每一项填写数字。'); paint(SAMPLE_SIZE); return; }
    state.error = ''; state.busy = true; paint(SAMPLE_SIZE);
    try { state.result = await api().calculateSampleSize(body); } catch (failure) {
      const known = PLANNING_ERRORS[failure && failure.code];
      state.error = known ? tr(...known) : failure && failure.status === 422
        ? tr('Check the values: one is outside what this calculation accepts.', '请检查填写的数值，有一项超出了这个计算允许的范围。')
        : tr('The calculation is unavailable right now.', '暂时无法计算。');
    } finally { state.busy = false; paint(SAMPLE_SIZE); }
  }

  // ── Loading and painting ─────────────────────────────────────────────
  async function load(kind) {
    const key = `${kind}:${state.database}`;
    const cache = kind === 'definitions' ? state.definitions : state.extraction;
    const loader = kind === 'definitions' ? api().loadClinicalDefinitions : api().loadExtractionProjection;
    // A failed read stays shown until the clinician picks a database again.
    if (cache.has(state.database) || state.loading.has(key) || state.failed.has(key) || typeof loader !== 'function') return;
    state.loading.add(key);
    const database = state.database;
    try { cache.set(database, await loader(database)); } catch (failure) { state.failed.set(key, failure); }
    state.loading.delete(key);
    paint(kind === 'definitions' ? DEFINITIONS : EXTRACTION);
  }
  function bodyHtml(skillName) {
    if (skillName === DEFINITIONS) { void load('definitions'); return definitionsHtml(); }
    if (skillName === EXTRACTION) { void load('extraction'); return extractionHtml(); }
    return sampleSizeHtml();
  }
  function html(skillName) {
    if (!TOOLS.has(skillName)) return '';
    return `<section class="eusk-tool" data-skt-tool="${esc(skillName)}" aria-label="${esc(tr('Tool', '工具'))}">${bodyHtml(skillName)}</section>`;
  }
  function paint(skillName) {
    if (typeof document === 'undefined') return;
    document.querySelectorAll(`[data-skt-tool="${skillName}"]`).forEach(node => {
      const focused = document.activeElement && node.contains(document.activeElement) ? document.activeElement.dataset.sktValue : '';
      node.innerHTML = bodyHtml(skillName);
      if (focused) { const input = node.querySelector(`[data-skt-value="${focused}"]`); if (input) input.focus(); }
    });
  }
  function repaintDefinitionParts() {
    if (typeof document === 'undefined') return;
    document.querySelectorAll(`[data-skt-tool="${DEFINITIONS}"]`).forEach(node => {
      const list = node.querySelector('[data-skt-list]'); if (list) list.innerHTML = listHtml();
      const detail = node.querySelector('[data-skt-detail]'); if (detail) detail.innerHTML = conceptHtml();
    });
  }

  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
    document.addEventListener('click', event => {
      const target = event.target && event.target.closest ? event.target.closest('[data-skt-concept]') : null;
      if (!target) return;
      state.concept = target.dataset.sktConcept; repaintDefinitionParts();
    });
    document.addEventListener('input', event => {
      const target = event.target;
      if (!target || !target.dataset) return;
      if (target.dataset.sktQuery !== undefined) { state.query = target.value; repaintDefinitionParts(); return; }
      if (target.dataset.sktValue !== undefined) {
        state.values[state.calculation] = { ...(state.values[state.calculation] || {}), [target.dataset.sktValue]: target.value };
      }
    });
    document.addEventListener('change', event => {
      const target = event.target;
      if (!target || !target.dataset) return;
      if (target.dataset.sktDatabase !== undefined) {
        state.database = target.value;
        state.failed.delete(`definitions:${state.database}`); state.failed.delete(`extraction:${state.database}`);
        paint(DEFINITIONS); paint(EXTRACTION);
      } else if (target.dataset.sktCalculation !== undefined) {
        state.calculation = target.value; state.result = null; state.error = '';
        paint(SAMPLE_SIZE);
      }
    });
    document.addEventListener('submit', event => {
      if (!event.target || !event.target.matches || !event.target.matches('[data-skt-calculate]')) return;
      event.preventDefault();
      void calculate();
    });
  }

  window.EU_SKILL_TOOLS = { has: skillName => TOOLS.has(skillName), html };
})();
