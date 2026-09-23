/* Owner: Guided Pi session starter widget. */
/* The signed-in entry canvas turns a reviewed Method Skill into an editable
   scientific question. Selecting a card never sends or grants data access. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;
  let shuffleOffset = 0;

  function fallbackCards(tr) {
    return [
      {
        id: 'cohort-characterization-table-one',
        title: tr('Cohort characterization and Table 1', '队列描述与 Table 1'),
        category: tr('Descriptive epidemiology', '描述性流行病学'),
        description: tr('Build a grouped baseline table from one typed, source-bound cohort.', '基于来源绑定的队列生成分组基线特征表。'),
        prompt: tr(
          'Use the Cohort characterization and Table 1 method. The research plan proposes the grouping levels, summarized variables, units, and summary rules; I review them together in the plan, and nothing runs before that review. My research question: ',
          '请使用“队列描述与 Table 1”方法。分组水平、汇总变量、单位与汇总规则由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：',
        ),
        capabilityId: 'descriptive_measurement_v1', actionIds: ['descriptive.table_one'], claimCeiling: 'analysis_only',
      },
      {
        id: 'exposure-outcome-distribution',
        title: tr('Exposure and outcome distribution', '研究因素与结局分布'),
        category: tr('Descriptive epidemiology', '描述性流行病学'),
        description: tr('Estimate typed absolute risks and denominators without causal language.', '计算类型明确的绝对风险与分母，并保持描述性解释。'),
        prompt: tr(
          'Use the Exposure and outcome distribution method. The research plan proposes the exposure levels, outcome definition, denominators, and interval method; I review them together in the plan, and nothing runs before that review. Interpretation stays descriptive. My research question: ',
          '请使用“研究因素与结局分布”方法。研究因素水平、结局定义、分母与区间方法由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。解释保持描述性，不作因果推断。我的研究问题是：',
        ),
        capabilityId: 'descriptive_exposure_outcome_distribution_v1', actionIds: [], claimCeiling: 'reportable',
      },
      {
        id: 'survival-time-to-event',
        title: tr('Survival and time-to-event analysis', '生存与时间结局分析'),
        category: tr('Survival analysis', '生存分析'),
        description: tr('Run a contract-bound Cox analysis with KM, log-rank, and PH diagnostics.', '运行契约绑定的 Cox 分析，并提供 KM、log-rank 与 PH 诊断。'),
        prompt: tr(
          'Use the Survival and time-to-event analysis method. The research plan proposes the time origin, event and censoring rules, reference level, covariates, follow-up horizon, and proportional-hazards check; I review them together in the plan, and nothing runs before that review. My research question: ',
          '请使用“生存与时间结局分析”方法。时间起点、事件与删失规则、参照水平、协变量、观察期限与比例风险检验由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：',
        ),
        capabilityId: 'survival_time_to_event_v1',
        actionIds: ['time_to_event.cox_hr', 'time_to_event.km_logrank', 'time_to_event.ph_check'],
        claimCeiling: 'reportable',
      },
    ];
  }

  function methodCards(tr) {
    const methodSkills = (((window.EU_CAPABILITIES || {}).capabilities || {}).method_skills || {});
    const items = Array.isArray(methodSkills.items) ? methodSkills.items : [];
    if (!items.length) return fallbackCards(tr);
    const cards = items.map(row => ({
      id: String(row.id || ''),
      title: tr(String(row.title || ''), String(row.title_zh || row.title || '')),
      category: tr(String(row.category || ''), String(row.category_zh || row.category || '')),
      description: tr(String(row.description || ''), String(row.description_zh || row.description || '')),
      prompt: tr(String(row.prompt || ''), String(row.prompt_zh || row.prompt || '')),
      capabilityId: String(row.capability_id || ''),
      actionIds: Array.isArray(row.action_ids) ? row.action_ids.map(String) : [],
      claimCeiling: String(row.claim_ceiling || 'analysis_only'),
    })).filter(row => row.id && row.title && row.prompt);
    if (!cards.length) return fallbackCards(tr);
    const preferred = [
      'cohort-characterization-table-one',
      'exposure-outcome-distribution',
      'survival-time-to-event',
    ];
    const byId = new Map(cards.map(row => [row.id, row]));
    return preferred.map(id => byId.get(id)).filter(Boolean)
      .concat(cards.filter(row => !preferred.includes(row.id)));
  }

  function visibleCards(tr) {
    const cards = methodCards(tr);
    if (cards.length <= 3) return cards;
    const start = shuffleOffset % cards.length;
    return [0, 1, 2].map(index => cards[(start + index) % cards.length]);
  }

  function shuffleIcon() {
    return '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M16 3h5v5M4 17.5h2.4c2.2 0 3.4-1.1 4.8-3.4l1.6-2.6c1.4-2.3 2.6-3.4 4.8-3.4H21M16 21h5v-5M4 6.5h2.4c1.7 0 2.8.7 3.8 2.1M13.8 15.4c1 1.4 2.1 2.1 3.8 2.1H21" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  }

  function card(row, disabled, tr) {
    const search = `${row.category} ${row.title} ${row.description}`.toLowerCase();
    return `<button type="button" class="gpi-starter-card" data-gpi-starter-card data-gpi-starter-search-text="${esc(search)}" data-gpi-starter-method="${esc(row.id)}" data-gpi-starter-intent="implement_scientific_question" data-gpi-starter-compose="${esc(row.prompt)}" ${disabled}>
      <span class="gpi-starter-category">${esc(row.category)}</span>
      <strong>${esc(row.title)}</strong>
      <small>${esc(row.description)}</small>
      <span class="gpi-starter-card-foot"><span>${esc(tr('Start', '开始'))}</span><span class="gpi-starter-badge">${esc(tr('Method', '方法'))}</span></span>
    </button>`;
  }

  // One runnable question on official demo data.  Naming the demo lets the
  // conversation offer that exact source, so a first look needs no setup; the
  // text only fills the composer and nothing is sent or run by the click.
  function demoCard(disabled, tr) {
    const prompt = tr(
      'Using the official eICU demo (eICU Collaborative Research Database Demo v2.0.1): how is ICU admission type (medical, surgical, other) associated with in-hospital mortality? Admission type is a baseline characteristic at ICU entry; the research plan decides and explains missing-value handling and the adjustment set.',
      '用 eICU 官方 Demo 数据（eICU Collaborative Research Database Demo v2.0.1）研究：ICU 入院类型（内科、外科、其他）与住院死亡有什么关联？入院类型是入 ICU 时的基线特征；缺失值怎么处理、调整哪些协变量由研究计划决定并说明。',
    );
    return `<div class="gpi-starter-demo"><button type="button" class="gpi-starter-card" data-gpi-starter-card data-gpi-starter-method="" data-gpi-starter-intent="implement_scientific_question" data-gpi-starter-compose="${esc(prompt)}" ${disabled}>
      <span class="gpi-starter-category">${esc(tr('Official demo · eICU', '官方 Demo · eICU'))}</span>
      <strong>${esc(tr('Try one question on official demo data', '用官方 Demo 数据试一个问题'))}</strong>
      <small>${esc(tr('Admission type and in-hospital mortality in 2,520 ICU stays. The data are prepared in one click; nothing runs until you approve the plan.', '2,520 次 ICU 住院中，入院类型与住院死亡的关联。数据一键准备；计划经你批准后才会运行。'))}</small>
      <span class="gpi-starter-card-foot"><span>${esc(tr('Start', '开始'))}</span><span class="gpi-starter-badge">Demo</span></span>
    </button></div>`;
  }

  function render(options) {
    const tr = options.tr;
    const disabled = options.disabled ? 'disabled' : '';
    const cards = visibleCards(tr);
    const canShuffle = methodCards(tr).length > 3;
    return `<section class="gpi-research-start gpi-entry-home" aria-label="${esc(tr('Start a research conversation', '开始研究对话'))}">
      <header class="gpi-entry-intro">
        <div class="gpi-entry-title"><h2>${esc(tr('Start an ICU research task', '开始一项 ICU 研究'))}</h2></div>
        <p>${esc(tr('Describe a scientific question, or choose a reviewed workflow below.', '描述一个科学问题，或从下方选择经过审阅的研究工作流。'))}</p>
      </header>
      ${options.composer || ''}
      ${demoCard(disabled, tr)}
      <div class="gpi-capability-head">
        <strong>${esc(tr('Explore EasyICU workflows', '探索 EasyICU 研究工作流'))}</strong>
        <div class="gpi-capability-tools"><label><span class="shell-sr-only">${esc(tr('Search workflows', '搜索研究工作流'))}</span><input type="search" data-gpi-starter-search placeholder="${esc(tr('Search…', '搜索…'))}" autocomplete="off"></label>
        <button type="button" data-gpi-starter-shuffle aria-label="${esc(tr('Show another set of workflows', '换一组研究工作流'))}" title="${esc(tr('Shuffle', '换一组'))}" ${disabled || (!canShuffle ? 'disabled' : '')}>${shuffleIcon()}</button></div>
      </div>
      <div class="gpi-starter-actions">${cards.map(row => card(row, disabled, tr)).join('')}</div>
      <p class="gpi-starter-empty" data-gpi-starter-empty hidden>${esc(tr('No matching workflows.', '没有匹配的研究工作流。'))}</p>
      <button class="gpi-starter-browse" type="button" data-gpi-starter-browse>${esc(tr('Browse all Skills and methods', '浏览全部技能与方法'))}</button>
    </section>`;
  }

  function actionFromEvent(event, tr) {
    const compose = event.target.closest('[data-gpi-starter-compose]');
    if (!compose) return null;
    const translate = typeof tr === 'function' ? tr : ((en) => en);
    const method = methodCards(translate).find(row => row.id === String(compose.dataset.gpiStarterMethod || '')) || null;
    return { kind: 'compose', text: String(compose.dataset.gpiStarterCompose || ''), intent: String(compose.dataset.gpiStarterIntent || ''), method };
  }

  const SEARCH_LIMIT = 6;
  function defaultTr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function cardSearchText(row) { return `${row.category} ${row.title} ${row.description}`.toLowerCase(); }

  // Search covers the whole reviewed catalogue, not only the three cards on
  // screen: a query re-renders the matching workflows (up to SEARCH_LIMIT) and
  // an empty query restores the current shuffle set.
  function filter(host, value, tr) {
    const translate = typeof tr === 'function' ? tr : defaultTr;
    const query = String(value || '').trim().toLowerCase();
    const actions = host && host.querySelector('.gpi-entry-home .gpi-starter-actions');
    if (!actions) return;
    const disabled = Boolean(actions.querySelector('[data-gpi-starter-card][disabled]')) ? 'disabled' : '';
    const rows = query
      ? methodCards(translate).filter(row => cardSearchText(row).includes(query)).slice(0, SEARCH_LIMIT)
      : visibleCards(translate);
    actions.innerHTML = rows.map(row => card(row, disabled, translate)).join('');
    const empty = host.querySelector('[data-gpi-starter-empty]');
    if (empty) empty.hidden = rows.length > 0;
  }

  function shuffle(host, tr) {
    const cards = methodCards(tr);
    if (!host || cards.length <= 3) return false;
    shuffleOffset = (shuffleOffset + 3) % cards.length;
    const actions = host.querySelector('.gpi-entry-home .gpi-starter-actions');
    if (!actions) return false;
    const disabled = Boolean(actions.querySelector('[data-gpi-starter-card][disabled]'));
    actions.innerHTML = visibleCards(tr).map(row => card(row, disabled ? 'disabled' : '', tr)).join('');
    const search = host.querySelector('[data-gpi-starter-search]');
    if (search) search.value = '';
    const empty = host.querySelector('[data-gpi-starter-empty]');
    if (empty) empty.hidden = true;
    return true;
  }

  function actionFromDiscoveryChoice(value, tr) {
    const choice = String(value || '').trim().replace(/[。. ]$/, '');
    if (choice === '从临床困惑开始' || choice === 'Start from a clinical uncertainty') return { kind: 'compose', text: tr('Use Idea Mining to discover research directions from this clinical uncertainty: ', '请用 Idea Mining 从这个临床困惑中发掘研究方向：'), intent: 'idea_mining_entry' };
    if (choice === '从已有文章或 PDF 开始' || choice === 'Start from an article or PDF') return { kind: 'compose', text: tr('Mine research directions from the article or PDF I attach: ', '请从我附加的文章或 PDF 中发掘研究方向：'), intent: 'idea_mining_entry' };
    if (choice === '从现有 ICU 数据开始' || choice === 'Start from existing ICU data') return { kind: 'compose', text: tr('I want to start from existing ICU data. First ask what data I have and what I want to accomplish. Do not read or analyse it until I confirm the source.', '我想从现有 ICU 数据开始。请先询问我有什么数据、希望完成什么任务；在我确认数据源前不要读取或分析。'), intent: 'data_first_entry' };
    return null;
  }

  window.EasyICU.guidedPi.declare('starters', { render, actionFromEvent, actionFromDiscoveryChoice, filter, shuffle });
})();
