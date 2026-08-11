/* Guided Pi complete research-demo transcript.
   Owner: read-only product-demo fixtures and their safe structured preview.
   It never starts a provider job or mutates a real EasyICU project. */
(function () {
  'use strict';

  const SOURCE_RUN_ID = 'run_20260811T030843_4d45a8';
  const SOURCE_AUTHORITY = 'engineering_canary_demo_only';

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function demoArtifact(name, title, summary, metrics, sections, sourceKind) {
    return {
      schema_version: 'easyicu.pi-product-demo-artifact/1',
      artifact: name,
      title,
      summary,
      status: 'read_only_product_demo',
      source_run_id: SOURCE_RUN_ID,
      source_authority: sourceKind || SOURCE_AUTHORITY,
      reportable: false,
      claim_ceiling: 'analysis_only',
      metrics: metrics || [],
      sections: sections || [],
    };
  }

  function artifacts() {
    return {
      'idea_shortlist.json': demoArtifact(
        'idea_shortlist.json',
        tr('Research opportunity shortlist', '研究机会候选'),
        tr('Three feasible MIMIC-IV directions grounded in the Sepsis-3 definition and the prepared EasyICU concept coverage.', '结合 Sepsis-3 定义与 EasyICU 已准备概念覆盖形成的 3 个 MIMIC-IV 可行方向。'),
        [
          { label: tr('Candidates', '候选数'), value: '3' },
          { label: tr('Literature records screened', '筛选文献'), value: '9' },
          { label: tr('Selected direction', '推荐方向'), value: tr('Sepsis-3 and hospital death', 'Sepsis-3 与院内死亡') },
        ],
        [
          { heading: tr('1 · Descriptive', '1 · 描述性'), items: [tr('Estimate first-24-hour Sepsis-3 prevalence with a visible ICU-stay denominator.', '估计入 ICU 后 24 小时 Sepsis-3 比例，明确 ICU stay 分母。')] },
          { heading: tr('2 · Recommended association study', '2 · 推荐的关联研究'), items: [tr('Compare in-hospital mortality with versus without the early Sepsis-3 indicator.', '比较早期 Sepsis-3 指标有无两组的院内死亡。'), tr('Prespecify age and sex adjustment and avoid causal language.', '预先规定年龄与性别调整，不使用因果措辞。')] },
          { heading: tr('3 · Extension', '3 · 延伸方向'), items: [tr('Repeat the same transparent definition in another compatible ICU database.', '在另一兼容 ICU 数据库复现相同的透明定义。')] },
        ],
        'demo_orchestration_reconstructed_from_bound_literature',
      ),
      'extraction_quality.json': demoArtifact(
        'extraction_quality.json',
        tr('Data package and quality review', '数据包与质量审阅'),
        tr('Aggregate-only projection of the prepared MIMIC-IV engineering-canary cohort; no patient rows or identifiers are exposed.', '真实 MIMIC-IV 工程试跑队列的仅聚合投影；不展示患者行或标识符。'),
        [
          { label: tr('ICU stays', 'ICU stays'), value: '140' },
          { label: tr('Sepsis-3 indicator present', 'Sepsis-3 指标阳性'), value: '53 / 140 (37.9%)' },
          { label: tr('Primary complete cases', '主分析完整病例'), value: '140 / 140' },
        ],
        [
          { heading: tr('Registered fields', '已登记字段'), items: ['sep3_sofa2_max', 'death', 'age', 'sex'] },
          { heading: tr('Time and denominator', '时间与分母'), items: [tr('Exposure window: ICU admission through 24 hours.', '暴露窗：入 ICU 至 24 小时。'), tr('Analysis unit: one prepared row per ICU stay.', '分析单位：每个 ICU stay 一条准备后记录。')] },
          { heading: tr('Governance', '治理边界'), items: [tr('Engineering-canary aggregate only; not formal paper evidence.', '仅工程试跑聚合结果，不是正式论文证据。')] },
        ],
      ),
      'agent_plan.json': demoArtifact(
        'agent_plan.json',
        tr('Evidence-bound analysis plan', '证据绑定的分析计划'),
        tr('Retrospective association analysis using the complete prepared cohort, explicit timing, and a locked robustness replay.', '使用完整准备队列、明确时间锚点与锁定稳健性复跑的回顾性关联分析。'),
        [
          { label: tr('Planned steps', '计划步骤'), value: '11' },
          { label: tr('Primary model', '主模型'), value: tr('Logistic regression', 'Logistic 回归') },
          { label: tr('Adjustment', '调整变量'), value: tr('Age + sex', '年龄 + 性别') },
        ],
        [
          { heading: tr('Primary analysis', '主分析'), items: [tr('Estimate the odds ratio for in-hospital death: Sepsis-3 indicator present versus absent.', '估计 Sepsis-3 指标阳性对阴性的院内死亡 OR。'), tr('Report group counts, risks, adjusted OR, and 95% confidence interval.', '报告分组人数、风险、调整 OR 与 95% 置信区间。')] },
          { heading: tr('Quality and sensitivity', '质量与敏感性'), items: [tr('Audit variable availability and value missingness before modelling.', '建模前审计变量可得性与数值缺失。'), tr('Replay the same specification in complete cases without changing the estimand.', '在完整病例中按同一设定复跑，不改变 estimand。')] },
          { heading: tr('Literature anchors', '文献依据'), items: [tr('Sepsis-3 consensus definition; STROBE and RECORD reporting guidance.', 'Sepsis-3 共识定义；STROBE 与 RECORD 报告规范。')] },
        ],
      ),
      'result_summary.json': demoArtifact(
        'result_summary.json',
        tr('Analysis results', '分析结果'),
        tr('The point estimate was above one, but the confidence interval was wide and included the null.', '点估计高于 1，但置信区间较宽且包含无效值。'),
        [
          { label: tr('Sepsis-3 prevalence', 'Sepsis-3 比例'), value: '53 / 140 (37.9%)' },
          { label: tr('Mortality · indicator present', '死亡率 · 指标阳性'), value: '8 / 53 (15.1%)' },
          { label: tr('Mortality · indicator absent', '死亡率 · 指标阴性'), value: '7 / 87 (8.0%)' },
          { label: tr('Adjusted odds ratio', '调整后 OR'), value: '1.50 (95% CI 0.49–4.60)' },
        ],
        [
          { heading: tr('Interpretation', '结果解读'), items: [tr('The estimate is compatible with lower, similar, or higher odds of death; it is not conclusive evidence of an association.', '该区间同时兼容更低、相近或更高的死亡优势比，不能作为明确关联证据。'), tr('This observational result does not support a causal claim.', '该观察性结果不支持因果结论。')] },
          { heading: tr('Next evidence needed', '下一步证据'), items: [tr('Larger cohorts and external validation with the same concept and time-window contract.', '在更大队列中按相同概念与时间窗合同进行外部验证。')] },
        ],
      ),
      'quality_gate.json': demoArtifact(
        'quality_gate.json',
        tr('Evidence and reportability gate', '证据与可报告性闸门'),
        tr('All five engineering checks passed, but the manuscript remains locked for human scientific review.', '5 项工程检查均通过，但稿件仍锁定，等待人工科学审阅。'),
        [
          { label: tr('Execution', '执行'), value: tr('Passed', '通过') },
          { label: tr('Analysis validation', '分析验证'), value: tr('Passed', '通过') },
          { label: tr('Evidence completeness', '证据完整性'), value: tr('Passed', '通过') },
          { label: tr('Numeric verification', '数值核验'), value: tr('Passed', '通过') },
          { label: tr('Reportability', '可报告性'), value: tr('Analysis-only', '仅供分析') },
        ],
        [{ heading: tr('Current gate', '当前闸门'), items: [tr('reportable = false', 'reportable = false'), tr('Human interpretation and sign-off are still required.', '仍需要人工解读与签署。')] }],
      ),
      'manuscript_draft.json': demoArtifact(
        'manuscript_draft.json',
        tr('Evidence-bound manuscript draft', '证据绑定的论文初稿'),
        tr('A structured draft was generated from Agent-produced results and citations, then locked pending human review.', '初稿由 Agent 产出的结果与引用生成，随后锁定等待人工审阅。'),
        [
          { label: tr('Status', '状态'), value: 'locked_pending_human_review' },
          { label: tr('Cohort', '队列'), value: '140 ICU stays' },
          { label: tr('Primary estimate', '主要估计'), value: 'OR 1.50 (95% CI 0.49–4.60)' },
        ],
        [
          { heading: tr('Draft conclusion', '初稿结论'), items: [tr('The early Sepsis-3 indicator had an imprecise adjusted association with in-hospital mortality in this prepared cohort.', '在该准备队列中，早期 Sepsis-3 指标与院内死亡的调整关联估计不精确。')] },
          { heading: tr('Required author review', '作者必须审阅'), items: [tr('Confirm clinical interpretation, limitations, citation fit, and the analysis-only claim ceiling before any external use.', '对外使用前确认临床解读、局限性、引用匹配和仅供分析的结论上限。')] },
        ],
      ),
    };
  }

  function artifactResource(name, label) {
    const item = artifacts()[name];
    return {
      kind: 'demo_artifact', artifact: name, label: label || name,
      title: item ? item.title : name, run_id: SOURCE_RUN_ID,
      media_type: 'application/json',
    };
  }
  function literatureResource() {
    return {
      kind: 'literature_source',
      label: tr('Singer et al. · Sepsis-3 consensus', 'Singer 等 · Sepsis-3 共识'),
      title: 'The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).',
      venue: 'JAMA', year: '2016', pmid: '26903338',
      relevance: tr('Defines Sepsis-3 around infection-related organ dysfunction and the SOFA framework.', '以感染相关器官功能障碍与 SOFA 框架定义 Sepsis-3。'),
      url: 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
    };
  }
  function activity(id, startedAt, endedAt, steps) {
    return { id, role: 'activity', status: 'complete', startedAt, endedAt, steps };
  }
  function tool(id, name, text, resource, resources) {
    return { id, kind: 'tool', toolName: name, status: 'complete', text: text || '', resource: resource || null, resources: resources || [] };
  }
  function pipeline(id, label, text, resource) {
    return { id, kind: 'pipeline', status: 'complete', label, text: text || '', resource: resource || null };
  }
  function message(id, role, text) { return { id, role, text, complete: true }; }

  function messages() {
    const lit = literatureResource();
    return [
      message('demo-user-1', 'user', tr(
        'Help me find a feasible MIMIC-IV study about early sepsis and in-hospital death. Show me the literature and data feasibility before deciding.',
        '帮我从 MIMIC-IV 里找一个关于早期脓毒症和院内死亡、而且数据可做的研究方向。先把文献依据和数据可行性给我看，再决定。',
      )),
      activity('demo-activity-1', 1000, 7200, [
        tool('ideas', 'easyicu_mine_ideas', tr('Compared the scientific question with available ICU concepts.', '将科学问题与可用 ICU 概念做可行性匹配。'), artifactResource('idea_shortlist.json', tr('Open the three research ideas', '打开 3 个研究方向'))),
        tool('literature', 'easyicu_search_literature', tr('Screened Sepsis-3 and observational-study design sources.', '筛选 Sepsis-3 与观察性研究设计文献。'), lit),
        pipeline('feasibility', tr('Checked MIMIC-IV concept feasibility', '已检查 MIMIC-IV 概念可行性'), tr('The prepared source contains the required exposure, outcome, age, and sex fields.', '准备数据源包含所需暴露、结局、年龄和性别字段。')),
      ]),
      message('demo-assistant-1', 'assistant', tr(
        '**I found three feasible directions.** I recommend the second: estimate first-24-hour Sepsis-3 prevalence and compare in-hospital mortality with versus without the indicator. MIMIC-IV has the required fields, and the design can use an explicit denominator and time window.\n\nThe clinical definition is anchored to the Sepsis-3 consensus; the plan will use non-causal language and prespecified age/sex adjustment.',
        '**我找到了 3 个可行方向。** 我建议选第 2 个：估计入 ICU 后 24 小时 Sepsis-3 比例，并比较指标阳性与阴性患者的院内死亡。MIMIC-IV 具备所需字段，而且可以明确分母与时间窗。\n\n临床定义依据 Sepsis-3 共识；分析会使用非因果措辞，并预先规定年龄与性别调整。',
      )),
      message('demo-user-2', 'user', tr('Choose the second one. Use MIMIC-IV.', '选择第 2 个，使用 MIMIC-IV。')),
      activity('demo-activity-2', 9000, 13200, [
        tool('accept', 'easyicu_accept_idea_handoff', tr('Bound the selected idea to this Copilot project.', '将所选想法绑定到当前 Copilot 项目。')),
        tool('setup', 'easyicu_update_study_context', tr('Prepared the cohort, exposure, outcome, window, and adjustment slots.', '准备队列、暴露、结局、时间窗与调整变量配置。')),
      ]),
      message('demo-assistant-2', 'assistant', tr(
        'Study setup is ready: **MIMIC-IV ICU stays**, exposure = first-24-hour `sep3_sofa2` indicator, outcome = in-hospital death, adjustment = age and sex.\n\nData extraction is a real data operation. Shall I create the aggregate research package and quality report?',
        '研究配置已准备好：**MIMIC-IV ICU stay**，暴露为入 ICU 后 24 小时 `sep3_sofa2` 指标，结局为院内死亡，调整年龄和性别。\n\n数据提取属于真实数据操作。是否生成聚合研究数据包和质量报告？',
      )),
      message('demo-user-3', 'user', tr('Yes. Continue with data extraction.', '可以，继续提取数据。')),
      activity('demo-activity-3', 15000, 23600, [
        tool('extract', 'easyicu_start_extraction', tr('Materialized the registered MIMIC-IV fields with provenance receipts.', '按已登记字段生成带来源回执的 MIMIC-IV 数据包。'), artifactResource('extraction_quality.json', tr('Open data and quality review', '打开数据与质量审阅'))),
        pipeline('denominator', tr('Verified the cohort denominator', '已核验队列分母'), tr('140 ICU stays; no hidden exclusion was applied.', '140 个 ICU stay；没有隐藏排除。')),
        pipeline('missingness', tr('Audited analysis-variable missingness', '已审计分析变量缺失'), tr('All 140 stays were complete for exposure, outcome, age, and sex.', '暴露、结局、年龄和性别在 140 个 stay 中均完整。')),
      ]),
      message('demo-assistant-3', 'assistant', tr(
        'The data package contains **140 ICU stays**. The Sepsis-3 indicator is present in **53/140 (37.9%)**; all 140 stays are complete for the primary variables. No patient rows or identifiers are shown here.\n\nThe aggregate quality review is ready. I can now prepare the evidence-bound analysis plan.',
        '数据包包含 **140 个 ICU stay**。其中 **53/140（37.9%）** 为 Sepsis-3 指标阳性；主分析变量在 140 个 stay 中均完整。这里不展示患者行或标识符。\n\n聚合质量审阅已完成。接下来可以生成证据绑定的分析计划。',
      )),
      message('demo-user-4', 'user', tr('Prepare the analysis plan for this question.', '按这个问题制定分析计划。')),
      activity('demo-activity-4', 25000, 33600, [
        tool('inspect-lit', 'easyicu_inspect_literature', tr('Mapped the clinical definition and reporting guidance to the scientific steps.', '将临床定义与报告规范映射到科学步骤。'), lit),
        tool('plan', 'easyicu_run', tr('Generated an 11-step digest-bound plan and stopped before analysis.', '生成 11 步摘要绑定计划，并在分析前暂停。'), artifactResource('agent_plan.json', tr('Open the analysis plan', '打开分析计划'))),
        pipeline('plan-gate', tr('Stopped for plan review', '已停在计划审阅门'), tr('No model was executed before the user approval shown next.', '在下一条用户批准前没有执行模型。')),
      ]),
      message('demo-assistant-4', 'assistant', tr(
        'The plan uses one age/sex-adjusted logistic regression for in-hospital death, reports group counts and absolute risks, audits missingness, and repeats the same estimand in a locked complete-case sensitivity analysis.\n\nThe Sepsis-3 definition and reporting choices are tied to the literature. Approve this plan to continue?',
        '计划采用年龄、性别调整的 Logistic 回归分析院内死亡，同时报告分组人数和绝对风险、审计缺失，并用相同 estimand 做锁定的完整病例敏感性分析。\n\nSepsis-3 定义与报告选择均绑定文献依据。是否批准该计划继续？',
      )),
      message('demo-user-5', 'user', tr('The plan looks good. Continue the analysis.', '计划可以，继续分析。')),
      activity('demo-activity-5', 35000, 73100, [
        tool('resume', 'easyicu_resume', tr('Resumed the approved digest-bound plan.', '恢复已批准的摘要绑定计划。')),
        pipeline('step-1', tr('1/11 · Built the transparent cohort flow', '1/11 · 构建透明队列流程'), tr('Denominator remained 140 ICU stays.', '分母保持为 140 个 ICU stay。')),
        pipeline('step-2', tr('2/11 · Generated Table 1', '2/11 · 生成 Table 1'), tr('Summarized age and sex by exposure group.', '按暴露组汇总年龄和性别。')),
        pipeline('step-3', tr('3/11 · Estimated prevalence and mortality', '3/11 · 估计比例与死亡率'), tr('53 indicator-positive and 87 indicator-negative stays.', '指标阳性 53 个，阴性 87 个 stay。')),
        pipeline('step-4', tr('4/11 · Completed missingness audit', '4/11 · 完成缺失审计'), tr('Primary complete-case denominator: 140.', '主分析完整病例分母：140。')),
        pipeline('step-5', tr('5/11 · Fit the adjusted model', '5/11 · 拟合调整模型'), tr('Age/sex-adjusted logistic regression completed.', '年龄、性别调整的 Logistic 回归完成。')),
        pipeline('step-6', tr('6–10/11 · Rendered estimates and sensitivity figures', '6–10/11 · 生成估计与敏感性图表'), tr('Primary estimate, absolute risk, robustness, and data-quality panels.', '主要估计、绝对风险、稳健性和数据质量图。')),
        pipeline('step-11', tr('11/11 · Bound evidence and verified numbers', '11/11 · 绑定证据并核验数值'), tr('All five engineering checks passed.', '5 项工程检查全部通过。'), artifactResource('quality_gate.json', tr('Open the evidence gate', '打开证据闸门'))),
        pipeline('results', tr('Prepared aggregate results for review', '已准备聚合结果供审阅'), tr('The run remains analysis-only pending human interpretation.', '该运行仍为仅供分析，等待人工解读。'), artifactResource('result_summary.json', tr('Open result summary', '打开结果摘要'))),
      ]),
      message('demo-assistant-5', 'assistant', tr(
        '**Analysis complete.** Mortality was **15.1% (8/53)** with the indicator and **8.0% (7/87)** without it. The age/sex-adjusted odds ratio was **1.50 (95% CI 0.49–4.60)**.\n\nThe interval includes the null, so this is not conclusive evidence of an association. The run passed five engineering checks but remains **analysis-only** pending human scientific review.',
        '**分析完成。** 指标阳性组死亡率为 **15.1%（8/53）**，阴性组为 **8.0%（7/87）**；年龄、性别调整后的 OR 为 **1.50（95% CI 0.49–4.60）**。\n\n置信区间包含无效值，因此不能认为已经得到明确关联。运行通过 5 项工程检查，但在人工科学审阅前仍为**仅供分析**。',
      )),
      message('demo-user-6', 'user', tr('Interpret the result and organize it into a manuscript draft.', '帮我解读结果，并整理成论文初稿。')),
      activity('demo-activity-6', 75000, 91800, [
        tool('interpret', 'easyicu_inspect_interpretation', tr('Separated the observed estimate from causal and clinical claims.', '将观察到的估计与因果、临床结论分开。'), artifactResource('result_summary.json', tr('Open evidence-bound interpretation', '打开证据约束的解读'))),
        tool('manuscript', 'easyicu_inspect_manuscript', tr('Organized methods, results, interpretation, and limitations with evidence references.', '按证据引用组织方法、结果、解读与局限。'), artifactResource('manuscript_draft.json', tr('Open manuscript draft', '打开论文初稿'))),
        pipeline('lock', tr('Locked the draft for human review', '已锁定初稿等待人工审阅'), tr('No external or paper-ready claim is authorized by this demo.', '该演示未授权任何对外或论文级结论。')),
      ]),
      message('demo-assistant-6', 'assistant', tr(
        'The result suggests a higher point estimate for mortality in indicator-positive stays, but the data are too imprecise to establish a reliable association and cannot support causality. The main limitations are the small single-database cohort, residual confounding, and dependence on the registered 24-hour indicator definition.\n\nI organized the evidence-bound manuscript draft and kept it locked for author review. The next step is clinical/methods review, not automatic publication.',
        '结果的点估计提示指标阳性 stay 的死亡率可能更高，但当前数据不够精确，不能确认稳定关联，更不能支持因果结论。主要局限包括单库小样本、残余混杂，以及对已登记 24 小时指标定义的依赖。\n\n我已整理证据绑定的论文初稿，并保持锁定等待作者审阅。下一步是临床与方法学审阅，不是自动发表。',
      )),
    ];
  }

  function workflow() {
    return {
      current_stage: 'manuscript', completed_required_stages: 7, required_stage_count: 7,
      next_action_code: 'human_review_and_reporting',
      stages: [
        ['question', 'complete', 'question_bound'],
        ['idea', 'complete', 'idea_handoff_accepted'],
        ['setup', 'complete', 'study_setup_complete'],
        ['extraction', 'complete', 'active_export_ready'],
        ['plan', 'complete', 'operator_plan_approved'],
        ['analysis', 'complete', 'validated_analysis_complete'],
        ['interpretation', 'complete', 'interpretation_complete'],
        ['manuscript', 'review_required', 'human_review_required'],
      ].map(([id, status, reason_code]) => ({ id, status, reason_code })),
    };
  }
  function artifact(name) { return clone(artifacts()[String(name || '')] || null); }
  function hasArtifact(name) { return Object.prototype.hasOwnProperty.call(artifacts(), String(name || '')); }
  function artifactLabel(name) {
    const item = artifacts()[String(name || '')];
    return item ? item.title : String(name || '');
  }
  function renderArtifact(payload) {
    const item = payload && typeof payload === 'object' ? payload : {};
    const metrics = Array.isArray(item.metrics) ? item.metrics : [];
    const sections = Array.isArray(item.sections) ? item.sections : [];
    return `<div class="gpi-demo-artifact">
      <div class="gpi-demo-artifact-intro"><strong>${esc(item.title || item.artifact || tr('Demo artifact', '演示产物'))}</strong><p>${esc(item.summary || '')}</p></div>
      ${metrics.length ? `<dl>${metrics.map(metric => `<div><dt>${esc(metric.label || '')}</dt><dd>${esc(metric.value || '')}</dd></div>`).join('')}</dl>` : ''}
      ${sections.map(section => `<section><h4>${esc(section.heading || '')}</h4><ul>${(Array.isArray(section.items) ? section.items : []).map(value => `<li>${esc(value)}</li>`).join('')}</ul></section>`).join('')}
    </div>`;
  }

  window.EU_GUIDED_PI_DEMO = {
    messages, workflow, artifact, hasArtifact, artifactLabel, renderArtifact,
    sourceRunId: SOURCE_RUN_ID,
  };
})();
