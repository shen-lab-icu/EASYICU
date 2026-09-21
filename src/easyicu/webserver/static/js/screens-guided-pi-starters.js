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
          'In an adult ICU cohort, describe baseline characteristics by a clinically meaningful grouping variable and generate a complete Table 1. Please first ask me to confirm the cohort, grouping levels, variables, units, and summary rules; do not read data or run analysis yet.',
          '在成人 ICU 队列中，按一个具有临床意义的分组变量描述基线特征，并生成完整的 Table 1。请先让我确认队列、分组水平、变量、单位和汇总规则；暂不读取数据或运行分析。',
        ),
        capabilityId: 'descriptive_measurement_v1', actionIds: ['descriptive.table_one'], claimCeiling: 'analysis_only',
      },
      {
        id: 'exposure-outcome-distribution',
        title: tr('Exposure and outcome distribution', '研究因素与结局分布'),
        category: tr('Descriptive epidemiology', '描述性流行病学'),
        description: tr('Estimate typed absolute risks and denominators without causal language.', '计算类型明确的绝对风险与分母，并保持描述性解释。'),
        prompt: tr(
          'In an adult ICU cohort, describe the distribution of a clinically defined exposure and outcome and estimate absolute risks with transparent denominators. Please first ask me to confirm the cohort, exposure levels, outcome definition, interval method, and dependence structure; do not imply causality.',
          '在成人 ICU 队列中，描述一个临床定义明确的研究因素与结局分布，并以透明分母计算绝对风险。请先让我确认队列、研究因素水平、结局定义、区间方法和相关性结构；不要作因果推断。',
        ),
        capabilityId: 'descriptive_exposure_outcome_distribution_v1', actionIds: [], claimCeiling: 'reportable',
      },
      {
        id: 'survival-time-to-event',
        title: tr('Survival and time-to-event analysis', '生存与时间结局分析'),
        category: tr('Survival analysis', '生存分析'),
        description: tr('Run a contract-bound Cox analysis with KM, log-rank, and PH diagnostics.', '运行契约绑定的 Cox 分析，并提供 KM、log-rank 与 PH 诊断。'),
        prompt: tr(
          'In an adult ICU cohort, study the association between a clinically defined exposure and a time-to-event outcome. Please first ask me to confirm time origin and unit, event and censoring rules, exposure and reference, covariates, horizon, complete-case policy, and proportional-hazards policy; do not run analysis yet.',
          '在成人 ICU 队列中，研究一个临床定义明确的研究因素与时间结局之间的关系。请先让我确认时间起点与单位、事件与删失规则、研究因素及参考水平、协变量、观察期限、完整案例策略和比例风险策略；暂不运行分析。',
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

  function filter(host, value) {
    const query = String(value || '').trim().toLowerCase();
    let visible = 0;
    host.querySelectorAll('[data-gpi-starter-card]').forEach(row => {
      row.hidden = Boolean(query && !String(row.dataset.gpiStarterSearchText || '').includes(query));
      if (!row.hidden) visible += 1;
    });
    const empty = host.querySelector('[data-gpi-starter-empty]');
    if (empty) empty.hidden = visible > 0;
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
