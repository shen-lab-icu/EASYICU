/* Owner: Guided Pi session starter widget. */
/* The signed-in entry canvas offers example questions in a clinician's words
   (each names its database, its study kind and what a run of it produces) and the reviewed
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

  /* Example questions, one per study kind, spread over the six databases.
     Each is chosen so that its database records what it asks for (the
     capability owner assesses it exact, with no substituted definition).
     "You get" names only what that study kind's reviewed method produces;
     no duration is shown until run records give a typical one. */
  function exampleQuestions(tr) {
    return [
      {
        id: 'example-lactate-death', database: 'eICU', kind: tr('Association', '关联分析'),
        title: tr('Lactate and in-hospital death', '乳酸与院内死亡'),
        gets: tr('Adjusted association estimates, association figure, baseline table', '调整后的关联估计、关联图、基线表'),
        prompt: tr(
          'In eICU, is the highest lactate on the first ICU day associated with in-hospital death? Does the association hold after adjusting for age, sex and illness severity?',
          '在 eICU 中，入 ICU 第一天的最高乳酸与院内死亡是否相关？调整年龄、性别和疾病严重程度后还成立吗？',
        ),
      },
      {
        id: 'example-aki-survival', database: 'MIMIC-IV', kind: tr('Survival', '生存分析'),
        title: tr('Acute kidney injury and 28-day survival', '急性肾损伤与 28 天生存'),
        gets: tr('Survival curves, Cox model results, proportional-hazards check', '生存曲线、Cox 模型结果、比例风险检验'),
        prompt: tr(
          'In MIMIC-IV, do patients with acute kidney injury (KDIGO criteria) on the first ICU day have worse 28-day survival?',
          '在 MIMIC-IV 中，入 ICU 第一天发生急性肾损伤（KDIGO 标准）的患者，28 天生存是否更差？',
        ),
      },
      {
        id: 'example-predict-death', database: 'SICdb', kind: tr('Prediction', '预测模型'),
        title: tr('Predicting in-hospital death', '预测院内死亡'),
        gets: tr('Discrimination (AUC), calibration curve, important variables', '区分度（AUC）、校准曲线、重要变量'),
        prompt: tr(
          'Using the first 24 hours of ICU data in SICdb, build a model that predicts in-hospital death. How well does it discriminate and how well is it calibrated?',
          '用 SICdb 中入 ICU 前 24 小时的数据建立一个预测院内死亡的模型，它的区分度和校准怎么样？',
        ),
      },
      {
        id: 'example-early-norepinephrine', database: 'MIMIC-IV', kind: tr('Causal inference', '因果推断'),
        title: tr('Early norepinephrine and death', '早期去甲肾上腺素与死亡'),
        gets: tr('Target-trial setup, weighted effect estimate, balance diagnostics', '目标试验设定、加权后的效应估计、平衡诊断'),
        prompt: tr(
          'In MIMIC-IV, among patients whose mean arterial pressure is below 65 mmHg at ICU admission, does starting norepinephrine within 6 hours lower 28-day mortality?',
          '在 MIMIC-IV 中，入 ICU 时平均动脉压低于 65 mmHg 的患者，6 小时内开始使用去甲肾上腺素是否降低 28 天死亡？',
        ),
      },
      {
        id: 'example-first-day-phenotypes', database: 'AmsterdamUMCdb', kind: tr('Phenotypes', '表型与亚型'),
        title: tr('Clinical phenotypes on the first ICU day', '入 ICU 第一天的临床表型'),
        gets: tr('Phenotypes and how many, stability check, each phenotype\'s profile and ICU death rate', '表型与分组数、稳定性检验、各表型特征与 ICU 死亡率'),
        prompt: tr(
          'In AmsterdamUMCdb, which phenotypes do first-day vital signs and laboratory results reveal, and how does ICU mortality differ between them?',
          '在 AmsterdamUMCdb 中，入 ICU 首日的生命体征和化验能分出哪几种表型？各型 ICU 死亡率有何不同？',
        ),
      },
      {
        id: 'example-ventilation-table-one', database: 'MIMIC-III', kind: tr('Descriptive', '描述性研究'),
        title: tr('Who is ventilated on the first day', '首日机械通气患者的特征'),
        gets: tr('Table 1 by group, each variable\'s denominator and missingness', '分组 Table 1、每个变量的分母与缺失'),
        prompt: tr(
          'In MIMIC-III, how do baseline characteristics differ between patients who did and did not receive invasive mechanical ventilation on the first ICU day?',
          '在 MIMIC-III 中，入 ICU 第一天接受与未接受有创机械通气的患者，基线特征有什么不同？',
        ),
      },
      {
        id: 'example-lactate-dose-response', database: 'SICdb', kind: tr('Dose–response', '剂量反应'),
        title: tr('How lactate level relates to death', '乳酸高低与院内死亡'),
        gets: tr('Spline dose–response curve, comparison with a linear model, adjusted estimates', '样条剂量反应曲线、与线性模型的比较、调整后的估计'),
        prompt: tr(
          'In SICdb, is the highest lactate in the first 24 ICU hours related to in-hospital death in a non-linear way after adjusting for age and sex?',
          '在 SICdb 中，入 ICU 24 小时内的最高乳酸与院内死亡之间，调整年龄和性别后是否呈非线性关系？',
        ),
      },
      {
        id: 'example-pressure-trajectories', database: 'eICU', kind: tr('Trajectories', '轨迹分型'),
        title: tr('Blood-pressure trajectories', '血压变化轨迹分型'),
        gets: tr('Trajectory groups and how many, stability check, early identification', '轨迹分组与分组数、稳定性检验、早期识别'),
        prompt: tr(
          'In eICU, into which groups do mean arterial pressure trajectories over the first 48 ICU hours fall, and can admission features tell which group a patient will follow?',
          '在 eICU 中，入 ICU 后 48 小时的平均动脉压轨迹可分为哪几类？能否在入 ICU 时就判断患者属于哪一类？',
        ),
      },
      {
        id: 'example-norepinephrine-use', database: 'HiRID', kind: tr('Treatment patterns', '治疗模式'),
        title: tr('How norepinephrine is used', '去甲肾上腺素的使用情况'),
        gets: tr('Share treated per group with denominators, time to first dose', '各组用药比例与分母、首次用药时间'),
        prompt: tr(
          'In HiRID, how do the share of patients given norepinephrine during the ICU stay and the time to the first dose differ between patients with and without invasive mechanical ventilation?',
          '在 HiRID 中，接受与未接受有创机械通气的患者，ICU 期间使用去甲肾上腺素的比例和首次用药时间有什么不同？',
        ),
      },
    ];
  }

  function exampleCard(row, disabled, tr) {
    return `<button type="button" class="gpi-starter-card gpi-starter-example" data-gpi-starter-card data-gpi-starter-method="" data-gpi-starter-example="${esc(row.id)}" data-gpi-starter-intent="implement_scientific_question" data-gpi-starter-compose="${esc(row.prompt)}" ${disabled}>
      <span class="gpi-starter-category">${esc(row.database)} · ${esc(row.kind)}</span>
      <strong>${esc(row.title)}</strong>
      <small>${esc(row.prompt)}</small>
      <span class="gpi-starter-gets"><b>${esc(tr('You get', '会得到'))}</b>${esc(row.gets)}</span>
      <span class="gpi-starter-card-foot"><span>${esc(tr('Use this question', '用这个问题'))}</span></span>
    </button>`;
  }

  // Where an approved analysis runs is read at render time: a hosted
  // instance runs it on the server, the desktop app on this computer.
  function introLine(tr) {
    return window.EasyICU.guidedPi.optional('hosted')?.active()
      ? tr('Describe your clinical question, choose the data, then review the analysis plan. After approval, analysis runs on the server.', '写下你的临床问题，选择数据，再核对分析计划。批准后在服务器上运行分析。')
      : tr('Describe your clinical question, choose the data, then review the analysis plan. After approval, analysis runs on this computer.', '写下你的临床问题，选择数据，再核对分析计划。批准后在本机运行分析。');
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
        <p>${esc(introLine(tr))}</p>
      </header>
      ${options.composer || ''}
      ${offersDemo() ? demoCard(disabled, tr) : ''}
      <div class="gpi-capability-head"><strong>${esc(tr('Try one of these questions', '试试这些问题'))}</strong></div>
      <div class="gpi-starter-actions gpi-starter-examples">${exampleQuestions(tr).map(row => exampleCard(row, disabled, tr)).join('')}</div>
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
