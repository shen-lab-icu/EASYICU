/* Owner: Guided Pi session starter widget. */
/* The signed-in entry canvas offers example questions in a clinician's words
   (each names its database and what a run of it produces) and the reviewed
   Method Skills behind a quiet link.
   Selecting a card never sends or grants data access: it only fills the composer. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function fallbackCards(tr) {
    return [
      {
        id: 'cohort-characterization-table-one',
        title: tr('Cohort characterization and Table 1', '队列描述与 Table 1'),
        category: tr('Descriptive', '描述性研究'),
        description: tr('Describe the study population\'s baseline characteristics by group as a Table 1, with each variable\'s true denominator.', '按分组描述研究人群的基线特征，生成 Table 1，并保留每个变量的真实分母。'),
        prompt: tr(
          'Use the Cohort characterization and Table 1 method. The research plan proposes the grouping levels, summarized variables, units, and summary rules; I review them together in the plan, and nothing runs before that review. My research question: ',
          '请使用“队列描述与 Table 1”方法。分组水平、汇总变量、单位与汇总规则由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：',
        ),
        capabilityId: 'descriptive_measurement_v1', actionIds: ['descriptive.table_one'], claimCeiling: 'analysis_only',
      },
      {
        id: 'exposure-outcome-distribution',
        title: tr('Exposure and outcome distribution', '暴露与结局分布'),
        category: tr('Descriptive', '描述性研究'),
        description: tr('Report outcome risks and denominators for each exposure group; descriptive only, with no causal wording.', '按暴露分组计算结局的发生率和分母，只做描述，不作因果解释。'),
        prompt: tr(
          'Use the Exposure and outcome distribution method. The research plan proposes the exposure levels, outcome definition, denominators, and interval method; I review them together in the plan, and nothing runs before that review. Interpretation stays descriptive. My research question: ',
          '请使用“暴露与结局分布”方法。暴露水平、结局定义、分母与区间方法由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。解释保持描述性，不作因果推断。我的研究问题是：',
        ),
        capabilityId: 'descriptive_exposure_outcome_distribution_v1', actionIds: [], claimCeiling: 'reportable',
      },
      {
        id: 'survival-time-to-event',
        title: tr('Survival analysis (time to event)', '生存分析（时间到事件）'),
        category: tr('Survival', '生存分析'),
        description: tr('Compare survival across groups: Kaplan–Meier curves, log-rank test, adjusted Cox hazard ratios and a proportional-hazards check.', '比较不同分组的生存：Kaplan–Meier 曲线、log-rank 检验、调整后的 Cox 风险比，并检查比例风险假设。'),
        prompt: tr(
          'Use the Survival analysis (time to event) method. The research plan proposes the time origin, event and censoring rules, reference level, covariates, follow-up horizon, and proportional-hazards check; I review them together in the plan, and nothing runs before that review. My research question: ',
          '请使用“生存分析（时间到事件）”方法。时间起点、事件与删失规则、参照水平、协变量、观察期限与比例风险检验由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：',
        ),
        capabilityId: 'survival_time_to_event_v1',
        actionIds: ['time_to_event.cox_hr', 'time_to_event.km_logrank', 'time_to_event.ph_check'],
        claimCeiling: 'reportable',
      },
    ];
  }

  /* Example questions inside v1's scope, one per study kind a clinician asks
     first. "You get" names only what that study kind's reviewed method
     produces; no duration is shown until run records give a typical one. */
  function exampleQuestions(tr) {
    return [
      {
        id: 'example-glucose-groups-death', database: 'eICU',
        title: tr('Glucose groups and in-hospital death', '血糖分组与院内死亡'),
        gets: tr('Baseline table by group, death rate per group with CI, distribution figure', '分组基线表、各组死亡率与置信区间、分布图'),
        prompt: tr(
          'In eICU, group patients by their first-day ICU glucose into hypoglycaemia, hyperglycaemia only and normal glucose. How does in-hospital mortality differ across the three groups?',
          '在 eICU 中，按入 ICU 第一天的血糖分为低血糖、仅高血糖和血糖正常三组，各组的院内死亡率有什么不同？',
        ),
      },
      {
        id: 'example-lactate-death', database: 'eICU',
        title: tr('Lactate and in-hospital death', '乳酸与院内死亡'),
        gets: tr('Adjusted association estimates, association figure, baseline table', '调整后的关联估计、关联图、基线表'),
        prompt: tr(
          'In eICU, is the highest lactate on the first ICU day associated with in-hospital death? Does the association hold after adjusting for age, sex and illness severity?',
          '在 eICU 中，入 ICU 第一天的最高乳酸与院内死亡是否相关？调整年龄、性别和疾病严重程度后还成立吗？',
        ),
      },
      {
        id: 'example-predict-death', database: 'eICU',
        title: tr('Predicting in-hospital death', '预测院内死亡'),
        gets: tr('Discrimination (AUC), calibration curve, important variables', '区分度（AUC）、校准曲线、重要变量'),
        prompt: tr(
          'Using the first 24 hours of ICU data in eICU, build a model that predicts in-hospital death. How well does it discriminate and how well is it calibrated?',
          '用 eICU 中入 ICU 前 24 小时的数据建立一个预测院内死亡的模型，它的区分度和校准怎么样？',
        ),
      },
      {
        id: 'example-sofa2-survival', database: 'MIMIC-IV',
        title: tr('SOFA-2 and 28-day survival', 'SOFA-2 与 28 天生存'),
        gets: tr('Survival curves, Cox model results, proportional-hazards check', '生存曲线、Cox 模型结果、比例风险检验'),
        prompt: tr(
          'Among patients with sepsis in MIMIC-IV, how is the SOFA-2 score on the first ICU day related to 28-day survival?',
          '在 MIMIC-IV 的脓毒症患者中，入 ICU 第一天的 SOFA-2 评分与 28 天生存有什么关系？',
        ),
      },
      {
        id: 'example-early-norepinephrine', database: 'MIMIC-IV',
        title: tr('Early norepinephrine and death', '早期去甲肾上腺素与死亡'),
        gets: tr('Target-trial setup, weighted effect estimate, balance diagnostics', '目标试验设定、加权后的效应估计、平衡诊断'),
        prompt: tr(
          'Among patients with septic shock in MIMIC-IV, does starting norepinephrine within 6 hours of ICU admission lower 28-day mortality?',
          '在 MIMIC-IV 的脓毒症休克患者中，入 ICU 6 小时内开始使用去甲肾上腺素，是否降低 28 天死亡？',
        ),
      },
    ];
  }

  function exampleCard(row, disabled, tr) {
    return `<button type="button" class="gpi-starter-card gpi-starter-example" data-gpi-starter-card data-gpi-starter-method="" data-gpi-starter-example="${esc(row.id)}" data-gpi-starter-intent="implement_scientific_question" data-gpi-starter-compose="${esc(row.prompt)}" ${disabled}>
      <span class="gpi-starter-category">${esc(row.database)}</span>
      <strong>${esc(row.title)}</strong>
      <small>${esc(row.prompt)}</small>
      <span class="gpi-starter-gets"><b>${esc(tr('You get', '会得到'))}</b>${esc(row.gets)}</span>
      <span class="gpi-starter-card-foot"><span>${esc(tr('Use this question', '用这个问题'))}</span></span>
    </button>`;
  }

  /* What EasyICU does and does not do, before the first question. Open on a
     viewer's first visit; the viewer's own choice is remembered per browser. */
  const NOTES_KEY = 'easyicu.entryNotes.closed';
  function notesClosed() {
    try { return window.localStorage.getItem(NOTES_KEY) === '1'; } catch (error) { return false; }
  }
  function rememberNotes(closed) {
    try {
      if (closed) window.localStorage.setItem(NOTES_KEY, '1');
      else window.localStorage.removeItem(NOTES_KEY);
    } catch (error) { /* a blocked store keeps the default */ }
  }
  if (typeof document !== 'undefined' && document.addEventListener) {
    // toggle does not bubble; one capturing listener owns the block's state.
    document.addEventListener('toggle', event => {
      const target = event.target;
      if (target && target.matches && target.matches('[data-gpi-entry-notes]')) rememberNotes(!target.open);
    }, true);
  }
  function entryNotes(tr) {
    const workflow = window.EasyICU.guidedPi.optional('hosted')?.active()
      ? tr('Confirm the data and review the analysis plan. After approval, analysis runs on the server.', '先确认数据，再核对分析计划。批准后在服务器上运行分析。')
      : tr('Confirm the data and review the analysis plan. After approval, analysis runs on this computer.', '先确认数据，再核对分析计划。批准后在本机运行分析。');
    const lines = [
      [tr('Can do', '能做'), tr('descriptive, association, prediction, survival, subtyping and causal studies on MIMIC-III, MIMIC-IV, eICU, HiRID, AmsterdamUMCdb and SICdb, as far as each database records what the question needs.', 'MIMIC-III、MIMIC-IV、eICU、HiRID、AmsterdamUMCdb、SICdb 上的描述、关联、预测、生存、亚型与因果分析，以该库记录了问题所需的信息为限。')],
      [tr('Accepted', '已验收'), tr('every study type on MIMIC-IV, and descriptive, association and prediction studies on eICU. Other combinations run but are not yet accepted on real data.', 'MIMIC-IV 的全部研究类型，以及 eICU 的描述、关联与预测分析。其余组合可以运行，但尚未在真实数据上验收。')],
      [tr('When data differ', '口径不同'), tr('If a database lacks what the question asks for, such as in-hospital death or exact age, the plan states the substitute and runs only after you approve it; otherwise EasyICU says what is missing.', '某个库缺少问题要求的信息时（例如没有院内死亡或只有年龄段），计划会写明替代口径，你批准后才运行；无法替代时会说明缺少什么。')],
      [tr('How it works', '流程'), workflow],
      [tr('Results', '结果'), tr('analysis-level evidence; review them before any publication use.', '分析级证据。用于发表前需要人工复核。')],
    ];
    return `<details class="gpi-entry-notes" data-gpi-entry-notes${notesClosed() ? '' : ' open'}>
      <summary>${esc(tr('Before you start', '使用前须知'))}</summary>
      <dl>${lines.map(([term, text]) => `<div><dt>${esc(term)}</dt><dd>${esc(text)}</dd></div>`).join('')}</dl>
    </details>`;
  }

  // The official demo is the way in without data; an instance holding a
  // registered full source does not offer it. An older host that reports no
  // count keeps offering it.
  function offersDemo() {
    const registered = (((window.EU_CAPABILITIES || {}).registered_sources || {}).by_database);
    if (!registered || typeof registered !== 'object') return true;
    return !Object.values(registered).some(value => Number(value) > 0);
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
    return `<section class="gpi-research-start gpi-entry-home" aria-label="${esc(tr('Start a research conversation', '开始研究对话'))}">
      <header class="gpi-entry-intro">
        <div class="gpi-entry-title"><h2>${esc(tr('Start an ICU research task', '开始一项 ICU 研究'))}</h2></div>
        <p>${esc(tr('Describe your clinical question, choose the data, then review the analysis plan.', '写下你的临床问题，选择数据，再核对分析计划。'))}</p>
      </header>
      ${options.composer || ''}
      ${offersDemo() ? demoCard(disabled, tr) : ''}
      <div class="gpi-capability-head"><strong>${esc(tr('Try one of these questions', '试试这些问题'))}</strong></div>
      <div class="gpi-starter-actions gpi-starter-examples">${exampleQuestions(tr).map(row => exampleCard(row, disabled, tr)).join('')}</div>
      ${entryNotes(tr)}
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

  function actionFromDiscoveryChoice(value, tr) {
    const choice = String(value || '').trim().replace(/[。. ]$/, '');
    if (choice === '从临床困惑开始' || choice === 'Start from a clinical uncertainty') return { kind: 'compose', text: tr('Use Idea Mining to discover research directions from this clinical uncertainty: ', '请用 Idea Mining 从这个临床困惑中发掘研究方向：'), intent: 'idea_mining_entry' };
    if (choice === '从已有文章或 PDF 开始' || choice === 'Start from an article or PDF') return { kind: 'compose', text: tr('Mine research directions from the article or PDF I attach: ', '请从我附加的文章或 PDF 中发掘研究方向：'), intent: 'idea_mining_entry' };
    if (choice === '从现有 ICU 数据开始' || choice === 'Start from existing ICU data') return { kind: 'compose', text: tr('I want to start from existing ICU data. First ask what data I have and what I want to accomplish. Do not read or analyse it until I confirm the source.', '我想从现有 ICU 数据开始。请先询问我有什么数据、希望完成什么任务；在我确认数据源前不要读取或分析。'), intent: 'data_first_entry' };
    return null;
  }

  window.EasyICU.guidedPi.declare('starters', { render, actionFromEvent, actionFromDiscoveryChoice });
})();
