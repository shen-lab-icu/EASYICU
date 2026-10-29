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

  /* Example questions, one per method, spread over the six databases.
     Each is chosen so that its database records what it asks for (the
     capability owner assesses it exact, with no substituted definition, and
     every variable it names is fully derivable there).
     "You get" names only what that method's reviewed workflow produces;
     no duration is shown until run records give a typical one. */
  function exampleQuestions(tr) {
    return [
      {
        id: 'example-quick-lactate', database: 'HiRID', kind: tr('Quick answer', '快速回答'),
        title: tr('Highest first-day lactate', '首日最高乳酸'),
        gets: tr('One answer card: median and quartiles, patients and missing; it can grow into a full study', '一张回答卡：中位数与四分位数、人数与缺失，可再展开为完整研究'),
        prompt: tr(
          'In HiRID, what are the median and quartiles of the highest lactate on the first ICU day?',
          '在 HiRID 中，入 ICU 第一天最高乳酸的中位数和四分位数是多少？',
        ),
      },
      {
        id: 'example-mortality-by-sex', database: 'MIMIC-III', kind: tr('Descriptive', '描述性研究'),
        title: tr('In-hospital mortality by sex', '男女的院内死亡率'),
        gets: tr('Mortality per group with 95% CI and denominators, their difference', '各组院内死亡率与 95% 置信区间、分母，组间差值'),
        prompt: tr(
          'In MIMIC-III, for each patient\'s first ICU stay, what is the in-hospital mortality, and does it differ between men and women?',
          '在 MIMIC-III 中，每位患者首次 ICU 入住的院内死亡率是多少？男性和女性之间有差别吗？',
        ),
      },
      {
        id: 'example-lactate-death', database: 'eICU', kind: tr('Association', '关联分析'),
        title: tr('Raised lactate and in-hospital death', '乳酸升高与院内死亡'),
        gets: tr('Adjusted risk ratio (modified Poisson regression), absolute risk per group, association figure', '调整后的相对危险度（RR，修正泊松回归）、各组绝对风险、关联图'),
        prompt: tr(
          'In eICU, do patients whose highest lactate on the first ICU day is above 2 mmol/L have a higher risk of in-hospital death? Report it as a risk ratio after adjusting for age, sex and illness severity.',
          '在 eICU 中，入 ICU 第一天最高乳酸超过 2 mmol/L 的患者，院内死亡风险是否更高？调整年龄、性别和疾病严重程度后，用相对危险度（RR）表示。',
        ),
      },
      {
        id: 'example-sofa-trend', database: 'eICU', kind: tr('Ordinal trend', '有序趋势'),
        title: tr('First-day SOFA strata and death', '首日 SOFA 分层与院内死亡'),
        gets: tr('Risk and denominator per stratum, trend test, adjusted estimates', '各层死亡风险与分母、趋势检验、调整后的估计'),
        prompt: tr(
          'In eICU, grouping the first-day SOFA score as 0–5, 6–9 and 10 or more, does the risk of in-hospital death rise step by step across the groups after adjusting for age and sex?',
          '在 eICU 中，把入 ICU 第一天的 SOFA 评分分为 0–5、6–9、10 分及以上三层，调整年龄和性别后，院内死亡风险是否逐层升高？',
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
        id: 'example-landmark-lactate', database: 'MIMIC-IV', kind: tr('Landmark', '界标分析'),
        title: tr('Lactate at 48 hours and 28-day death', '48 小时界标：乳酸与 28 天死亡'),
        gets: tr('Landmark cohort and its size, adjusted association estimates, association figure', '界标队列与入组人数、调整后的关联估计、关联图'),
        prompt: tr(
          'In MIMIC-IV, among patients still in the ICU at 48 hours, is the highest lactate in those 48 hours associated with death within 28 days after adjusting for age, sex, blood pressure and creatinine?',
          '在 MIMIC-IV 中，入 ICU 48 小时仍在 ICU 的患者，48 小时内的最高乳酸与 28 天内死亡是否相关？调整年龄、性别、血压和肌酐后还成立吗？',
        ),
      },
      {
        id: 'example-aki-survival', database: 'MIMIC-IV', kind: tr('Survival', '生存分析'),
        title: tr('Acute kidney injury and 28-day survival', '急性肾损伤与 28 天生存'),
        gets: tr('Kaplan–Meier curves with log-rank test, Cox model results, proportional-hazards check', 'Kaplan–Meier 曲线与 log-rank 检验、Cox 模型结果、比例风险检验'),
        prompt: tr(
          'In MIMIC-IV, do patients with acute kidney injury (KDIGO criteria) on the first ICU day have worse 28-day survival?',
          '在 MIMIC-IV 中，入 ICU 第一天发生急性肾损伤（KDIGO 标准）的患者，28 天生存是否更差？',
        ),
      },
      {
        id: 'example-ventilation-rmst', database: 'SICdb', kind: tr('Survival with RMST', '生存分析与 RMST'),
        title: tr('Ventilation and 28-day survival', '机械通气与 28 天生存'),
        gets: tr('Kaplan–Meier curves, adjusted Cox hazard ratio, 28-day RMST difference (sensitivity)', 'Kaplan–Meier 曲线、调整后的 Cox 风险比、28 天 RMST 差值（敏感性分析）'),
        prompt: tr(
          'In SICdb, does 28-day survival differ between patients ventilated on the first ICU day and those who are not, after adjusting for age and sex? Report the 28-day restricted mean survival time difference as a sensitivity analysis.',
          '在 SICdb 中，入 ICU 第一天接受机械通气的患者与未通气的患者相比，调整年龄和性别后 28 天生存是否不同？并以 28 天限制性平均生存时间差作为敏感性分析。',
        ),
      },
      {
        id: 'example-ventilation-time-varying', database: 'MIMIC-III', kind: tr('Time-varying exposure', '时变暴露'),
        title: tr('Starting ventilation and ICU death', 'ICU 期间开始机械通气与 ICU 死亡'),
        gets: tr('Time-varying Cox model results, follow-up split at the start of exposure', '时变 Cox 模型结果、按开始时间切分的随访'),
        prompt: tr(
          'In MIMIC-III, taking the start of mechanical ventilation during the ICU stay as an exposure that changes over time, how is it related to death in the ICU after adjusting for age and sex?',
          '在 MIMIC-III 中，把 ICU 期间开始机械通气作为随时间变化的暴露，调整年龄和性别后，它与 ICU 内死亡的关系如何？',
        ),
      },
      {
        id: 'example-creatinine-competing', database: 'SICdb', kind: tr('Competing risks', '竞争风险'),
        title: tr('Creatinine and ICU death: competing risks', '肌酐与 ICU 死亡的竞争风险'),
        gets: tr('Cumulative incidence curves with Gray\'s test, cause-specific Cox and Fine–Gray models', '累积发生率曲线与 Gray 检验、原因别 Cox 与 Fine–Gray 模型'),
        prompt: tr(
          'In SICdb, how is the highest creatinine on the first ICU day related to death in the ICU, treating discharge alive from the ICU as a competing event and adjusting for age and sex?',
          '在 SICdb 中，入 ICU 第一天的最高肌酐与 ICU 内死亡有什么关系？把活着转出 ICU 作为竞争事件，调整年龄和性别。',
        ),
      },
      {
        id: 'example-predict-death', database: 'HiRID', kind: tr('Prediction', '预测模型'),
        title: tr('Predicting death in the ICU', '预测 ICU 内死亡'),
        gets: tr('Discrimination (AUC), calibration curve, important variables', '区分度（AUC）、校准曲线、重要变量'),
        prompt: tr(
          'Using the first 24 hours of ICU data in HiRID, build a model that predicts death in the ICU. How well does it discriminate and how well is it calibrated?',
          '用 HiRID 中入 ICU 前 24 小时的数据建立一个预测 ICU 内死亡的模型，它的区分度和校准怎么样？',
        ),
      },
      {
        id: 'example-dynamic-prediction', database: 'AmsterdamUMCdb', kind: tr('Dynamic prediction', '动态预测'),
        title: tr('Updating the risk of ICU death', '随时间更新的 ICU 死亡风险'),
        gets: tr('Discrimination and calibration at each landmark, patients at risk at each', '各时点的区分度与校准、各时点的在险人数'),
        prompt: tr(
          'In AmsterdamUMCdb, at 24, 48 and 72 hours after ICU admission, how well do the vital signs and laboratory results up to that moment predict death in the ICU over the next 7 days?',
          '在 AmsterdamUMCdb 中，在入 ICU 第 24、48、72 小时，用到当时为止的生命体征和化验，预测之后 7 天内 ICU 死亡的效果如何？',
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
        id: 'example-repeated-measure-change', database: 'MIMIC-IV', kind: tr('Repeated measures', '重复测量'),
        title: tr('Daily lactate change in the ICU', 'ICU 每日乳酸的变化'),
        gets: tr('Estimated change with confidence intervals and a death-dropout sensitivity', '变化估计、置信区间及死亡脱落敏感性分析'),
        prompt: tr(
          'In MIMIC-IV, how do daily lactate measurements change over time during the first six ICU days? State the limitation from informative dropout due to death and plan a sensitivity analysis.',
          '在 MIMIC-IV 中，入 ICU 前六天的每日乳酸如何随时间变化？请明确死亡所致信息性脱落的局限，并安排敏感性分析。',
        ),
      },
      {
        id: 'example-pressure-trajectories', database: 'eICU', kind: tr('Trajectories', '轨迹分型'),
        title: tr('Blood-pressure trajectories', '血压变化轨迹分型'),
        gets: tr('Trajectory groups and how many, stability check, in-hospital death rate per group', '轨迹分组与分组数、稳定性检验、各类院内死亡率'),
        prompt: tr(
          'In eICU, into which groups do mean arterial pressure trajectories over the first 48 ICU hours fall, and how does in-hospital mortality differ between them?',
          '在 eICU 中，入 ICU 后 48 小时的平均动脉压轨迹可分为哪几类？各类患者院内死亡率有何不同？',
        ),
      },
      {
        id: 'example-vasopressor-use', database: 'AmsterdamUMCdb', kind: tr('Treatment patterns', '治疗模式'),
        title: tr('When vasopressors are started', '血管活性药何时开始使用'),
        gets: tr('Patients treated, median and quartiles of the time to first use', '用药人数、首次使用时间的中位数与四分位数'),
        prompt: tr(
          'In AmsterdamUMCdb, among patients given a vasopressor on the first ICU day, how many hours after ICU admission is it first started?',
          '在 AmsterdamUMCdb 中，入 ICU 第一天用了血管活性药的患者，首次使用多在入 ICU 后几小时？',
        ),
      },
      {
        id: 'example-replicate-lactate', database: tr('MIMIC-IV, eICU, SICdb', 'MIMIC-IV、eICU、SICdb'),
        kind: tr('Multi-database replication', '多库验证'),
        title: tr('Does the lactate finding hold elsewhere?', '乳酸的结论在其他库也成立吗'),
        gets: tr('Each database\'s own estimate beside the original, whether the directions agree (not pooled)', '各库分别的估计与原研究并列、方向是否一致（不合并）'),
        prompt: tr(
          'In MIMIC-IV, study whether the highest lactate on the first ICU day is associated with in-hospital death after adjusting for age and sex; then repeat the same study in eICU and in SICdb.',
          '在 MIMIC-IV 中研究入 ICU 第一天最高乳酸与院内死亡的关联（调整年龄和性别），再在 eICU 和 SICdb 上分别重复同一研究。',
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
    const examples = exampleQuestions(tr);
    return `<section class="gpi-research-start gpi-entry-home" aria-label="${esc(tr('Start a research conversation', '开始研究对话'))}">
      <header class="gpi-entry-intro">
        <div class="gpi-entry-title"><h2>${esc(tr('Start an ICU research task', '开始一项 ICU 研究'))}</h2></div>
        <p>${esc(introLine(tr))}</p>
      </header>
      ${options.composer || ''}
      ${options.focused ? '' : `${offersDemo() ? demoCard(disabled, tr) : ''}
      <div class="gpi-capability-head"><strong>${esc(tr('Try one of these questions', '试试这些问题'))}</strong><div class="gpi-capability-tools"><label><span class="shell-sr-only">${esc(tr('Search example questions', '搜索示例问题'))}</span><input type="search" data-gpi-starter-search placeholder="${esc(tr('Database, question or method…', '搜索数据库、问题或方法…'))}" /></label></div></div>
      <div class="gpi-starter-actions gpi-starter-examples">${examples.slice(0, 3).map(row => exampleCard(row, disabled, tr)).join('')}</div>
      <details class="gpi-starter-more" data-gpi-more-examples><summary>${esc(tr(`View all ${examples.length} example questions`, `查看全部 ${examples.length} 个示例问题`))}</summary>
        <div class="gpi-starter-actions gpi-starter-examples">${examples.slice(3).map(row => exampleCard(row, disabled, tr)).join('')}</div>
      </details>
      <button class="gpi-starter-browse" type="button" data-gpi-starter-browse>${esc(tr('Browse all Skills and methods', '浏览全部技能与方法'))}</button>`}
    </section>`;
  }

  function filter(root, query) {
    const needle = String(query || '').trim().toLowerCase();
    root.querySelectorAll('[data-gpi-starter-example]').forEach(card => {
      card.hidden = Boolean(needle) && !card.textContent.toLowerCase().includes(needle);
    });
    const more = root.querySelector('[data-gpi-more-examples]');
    if (more) more.open = Boolean(needle);
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

  function exampleQuestion(id, tr) { return exampleQuestions(tr).find(row => row.id === id) || null; }
  window.EasyICU.guidedPi.declare('starters', { render, filter, actionFromEvent, actionFromDiscoveryChoice, exampleQuestion });
})();
