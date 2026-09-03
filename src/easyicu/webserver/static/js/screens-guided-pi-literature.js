/* Guided Pi literature evidence renderer owner.
   Retrieval remains owned by Idea Mining / Research Agent. This module renders
   only host-projected metadata and never infers citations from prose. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function safeUrl(value) {
    try {
      const parsed = new URL(String(value || ''));
      if (parsed.protocol !== 'https:' || !parsed.hostname || parsed.username || parsed.password) return '';
      return parsed.href;
    } catch (_) { return ''; }
  }
  function sourceMeta(row) {
    return [row.venue, row.year, row.pmid ? `PMID ${row.pmid}` : '', row.doi ? `DOI ${row.doi}` : '']
      .filter(Boolean).map(esc).join(' · ');
  }
  function displayTitle(row) {
    const title = String(row.title || row.label || row.key || tr('Untitled source', '未命名文献'));
    if (window.EU_LANG !== 'zh') return title;
    if (/\bSTROBE\b/i.test(title)) return '观察性研究报告规范（STROBE）';
    if (/\bRECORD\b/i.test(title)) return '常规医疗数据研究报告规范（RECORD）';
    if (/\bSepsis-3\b/i.test(title)) return '脓毒症 Sepsis-3 共识定义';
    if (/\bSOFA\b/i.test(title)) return 'SOFA 器官功能评分定义';
    return title;
  }
  function semanticIntentKey(value) {
    const intent = String(value || '');
    const normalized = intent.toLowerCase();
    if (/cohort|eligib|population|account|人群|队列|纳入|排除|分母/.test(normalized)) return 'population';
    if (/nonlinear|spline|functional|dose|form|非线性|样条|函数形式|阈值/.test(normalized)) return 'functional_form';
    if (/exposure|lactate|measurement|time|window|暴露|乳酸|测量|时间窗|时间对齐/.test(normalized)) return 'exposure_timing';
    if (/outcome|mortality|death|结局|死亡/.test(normalized)) return 'outcome';
    if (/missing|imput|缺失|插补/.test(normalized)) return 'missingness';
    if (/confound|adjust|covariat|model|混杂|调整|协变量|模型/.test(normalized)) return 'adjustment_model';
    if (/sensitivity|robust|敏感|稳健/.test(normalized)) return 'robustness';
    if (/report|quality|summary|报告|质量|汇总/.test(normalized)) return 'reporting';
    return normalized || 'plan_decision';
  }
  function displayIntent(value) {
    const intent = String(value || '');
    if (window.EU_LANG !== 'zh') return intent || 'Plan decision';
    const functionalFormLabel = /lactate|乳酸/i.test(intent)
      ? '检验乳酸与死亡是否为非线性关系'
      : '检验连续研究因素与结局是否为非线性关系';
    const labels = {
      population: '确定研究人群与纳排标准',
      functional_form: functionalFormLabel,
      exposure_timing: '确定研究因素的测量方式与时间窗',
      outcome: '明确研究结局',
      missingness: '处理缺失数据',
      adjustment_model: '确定混杂因素与统计模型',
      robustness: '安排敏感性分析',
      reporting: '规范结果与研究流程的报告',
    };
    return labels[semanticIntentKey(intent)] || '计划中的科学设计决定';
  }
  function displayApplication(source) {
    const application = String((source && source.application) || '');
    if (window.EU_LANG !== 'zh') return application;
    const haystack = `${source && source.key || ''} ${source && source.title || ''} ${application}`.toLowerCase();
    if (/strobe/.test(haystack)) return '用于检查观察性研究是否完整报告研究对象、变量定义、偏倚、统计方法和结果。它规范报告方式，不证明当前研究因素与结局有关。';
    if (/record/.test(haystack)) return '用于补充说明常规医疗数据的来源、筛选和处理过程。它规范数据透明度，不证明当前研究因素与结局有关。';
    if (/missing|imput|sterne/.test(haystack)) return '用于预先规定研究因素和协变量缺失时的处理与敏感性分析，避免只分析完整病例造成偏倚。';
    if (/spline|nonlinear|durrleman/.test(haystack)) return '用于检验连续研究因素与结局是否存在弯曲或阈值关系，避免武断地假设每升高 1 单位风险都等比例变化。';
    if (/mimic[-_ ]?iv|johnson_mimic/.test(haystack)) return '用于说明本研究数据库的来源、覆盖范围和可复现引用，不作为当前研究因素与结局关联的直接证据。';
    if (/direct|compar|lactate|mortality|death|icu/.test(haystack)) return '用于对照相近 ICU 人群中研究因素、结局和分析策略的定义；本计划只借鉴设计，不复制文献中的效应值。';
    return application ? '这篇文献与该设计决定有关；下方保留原始绑定，仍需研究者核对其适用性。' : '';
  }
  function sourceKey(source) {
    const row = source && typeof source === 'object' ? source : {};
    return String(row.key || row.pmid || row.doi || row.source_url || row.url || row.title || '').trim();
  }
  function uniqueSources(rows) {
    const seen = new Set();
    return (Array.isArray(rows) ? rows : []).filter(row => {
      const key = sourceKey(row);
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }
  function boundUsage(payload) {
    const usage = new Map();
    (Array.isArray(payload.step_citation_map) ? payload.step_citation_map : []).forEach(step => {
      (Array.isArray(step.citation_bindings) ? step.citation_bindings : []).forEach(binding => {
        const key = String(binding.key || '');
        if (!key) return;
        if (!usage.has(key)) usage.set(key, []);
        const next = {
          key: semanticIntentKey(step.intent),
          intent: displayIntent(step.intent),
          application: displayApplication(binding),
          designElements: Array.from(bindingElements(binding)),
        };
        const duplicate = usage.get(key).some(row => row.intent === next.intent && row.application === next.application);
        if (!duplicate) usage.get(key).push(next);
      });
    });
    return usage;
  }
  function bindingElements(binding) {
    return new Set((Array.isArray(binding && binding.design_elements)
      ? binding.design_elements : []).map(String));
  }
  function isReportingOnly(binding) {
    const elements = bindingElements(binding);
    return elements.size === 1 && elements.has('reporting');
  }
  function evidenceRole(binding, directKeys) {
    const key = sourceKey(binding);
    if (directKeys && directKeys.has(key)) return 'direct';
    const haystack = `${key} ${binding && binding.title || ''}`.toLowerCase();
    const elements = bindingElements(binding);
    if (isReportingOnly(binding) || /\bstrobe\b|\brecord\b/.test(haystack)) return 'reporting';
    if (/imput|missing|spline|landmark|survival|regression|model/.test(haystack)
      || ['time_zero', 'missing_data', 'robustness'].some(value => elements.has(value))) return 'methods';
    if (['population', 'exposure', 'outcome', 'adjustment', 'dependence', 'measurement'].some(value => elements.has(value))) return 'variables';
    return 'methods';
  }
  function evidenceBuckets(payload) {
    const directKeys = new Set(Array.isArray(payload.direct_comparator_keys)
      ? payload.direct_comparator_keys.map(String) : []);
    const citationByKey = new Map((Array.isArray(payload.citations) ? payload.citations : [])
      .map(row => [sourceKey(row), row]).filter(row => row[0]));
    const entries = new Map();
    citationByKey.forEach((source, key) => {
      if (directKeys.has(key)) entries.set(key, { source, role: 'direct' });
    });
    (Array.isArray(payload.step_citation_map) ? payload.step_citation_map : []).forEach(step => {
      (Array.isArray(step.citation_bindings) ? step.citation_bindings : []).forEach(binding => {
        const key = sourceKey(binding);
        if (!key) return;
        const source = { ...(citationByKey.get(key) || {}), ...binding };
        const role = evidenceRole(binding, directKeys);
        const existing = entries.get(key);
        if (!existing || existing.role !== 'direct') entries.set(key, { source, role });
      });
    });
    const buckets = { direct: [], variables: [], methods: [], reporting: [] };
    entries.forEach(entry => buckets[entry.role].push(entry.source));
    Object.keys(buckets).forEach(key => { buckets[key] = uniqueSources(buckets[key]); });
    return buckets;
  }
  function evidenceCoverage(payload) {
    const buckets = evidenceBuckets(payload);
    const scientificBindings = buckets.variables.length + buckets.methods.length;
    const reportingBindings = buckets.reporting.length;
    const directCount = Number(payload.direct_comparator_count || 0);
    const incomplete = directCount === 0 || scientificBindings === 0;
    return `<section class="gpi-lit-coverage ${incomplete ? 'incomplete' : 'supported'}">
      <header><h3>${esc(tr('What the literature currently supports', '当前文献实际支撑了什么'))}</h3><span>${esc(incomplete ? tr('Incomplete', '依据不完整') : tr('Mapped', '已有科学依据'))}</span></header>
      <div class="gpi-lit-coverage-grid">
        <div><span>${esc(tr('Direct question studies', '研究问题直接相关'))}</span><strong>${esc(tr(`${buckets.direct.length || directCount} source(s)`, `${buckets.direct.length || directCount} 篇`))}</strong></div>
        <div><span>${esc(tr('Variables and covariates', '变量与协变量依据'))}</span><strong>${esc(tr(`${buckets.variables.length} source(s)`, `${buckets.variables.length} 篇`))}</strong></div>
        <div><span>${esc(tr('Statistical methods', '统计方法依据'))}</span><strong>${esc(tr(`${buckets.methods.length} source(s)`, `${buckets.methods.length} 篇`))}</strong></div>
        <div><span>${esc(tr('Reporting guidance', '报告规范'))}</span><strong>${esc(tr(`${reportingBindings} source(s)`, `${reportingBindings} 篇`))}</strong></div>
      </div>
      ${incomplete ? `<p>${esc(tr(
        'Reporting guidance alone cannot determine the exposure window, outcome definition, adjustment strategy, statistical model, or sensitivity analyses. This artifact is not yet a literature-grounded scientific plan.',
        '仅有报告规范不能决定研究因素时间窗、结局定义、混杂因素调整、统计模型或敏感性分析。这份产物目前还不能算作有完整文献依据的科学计划。'
      ))}</p>` : ''}
    </section>`;
  }
  function artifactStatusBanner(meta) {
    const context = meta && typeof meta === 'object' ? meta : {};
    const code = String(context.nextActionCode || '');
    const activeJob = context.activeJob && typeof context.activeJob === 'object' ? context.activeJob : {};
    const failedJob = context.failedJob && typeof context.failedJob === 'object' ? context.failedJob : {};
    const runId = String(context.runId || '');
    const currentRunId = String(context.currentRunId || '');
    const revisionRunning = activeJob.present && activeJob.kind === 'agent-run' && activeJob.status === 'running';
    const revisionFailed = code === 'failed_pipeline_requires_fresh_plan'
      && failedJob.kind === 'agent-run' && failedJob.status === 'failed';
    const superseded = revisionRunning || revisionFailed
      || ['plan_scientific_changes_required', 'plan_configuration_superseded', 'plan_review_not_resumable'].includes(code)
      || Boolean(runId && currentRunId && runId !== currentRunId);
    if (!superseded) return '';
    let title = tr('Historical plan snapshot', '历史计划快照');
    let detail = tr(
      'The counts below describe this saved run only. They are not the current conclusion of the project.',
      '下方数字只描述这次已保存的旧运行，不代表项目当前结论。'
    );
    if (revisionRunning) {
      title = tr('Previous plan snapshot · revision in progress', '上一版计划快照 · 修订版正在生成');
      detail = tr(
        'Keep this snapshot for audit only. A new literature search and candidate plan are being generated before analysis.',
        '这份快照仅供回看；系统正在分析前重新检索文献并生成新的候选计划。'
      );
    } else if (revisionFailed) {
      const drafts = Math.max(0, ...((Array.isArray(failedJob.progress) ? failedJob.progress : [])
        .filter(row => row && row.step === 'planning').map(row => Number(row.total || row.current || 0))));
      title = tr('Previous plan snapshot · revision did not complete', '上一版计划快照 · 修订版未生成成功');
      detail = drafts
        ? tr(
          `${drafts} draft attempt(s) failed the scientific contract. The old “0 direct matches” remains historical evidence, not a current finding. Return to the conversation to generate a fresh plan.`,
          `系统尝试了 ${drafts} 版草案，均未通过科学合同。旧页面的“直接匹配 0 篇”只是历史记录，不是当前结论；请返回对话重新生成计划。`
        )
        : tr(
          'The replacement plan did not complete. The old counts remain historical evidence, not a current finding. Return to the conversation to generate a fresh plan.',
          '替代计划未生成完成。旧页面数字只是历史记录，不是当前结论；请返回对话重新生成计划。'
        );
    } else if (code === 'plan_scientific_changes_required') {
      title = tr('Candidate plan awaiting revision', '待修订的上一版候选计划');
      detail = tr(
        'Scientific review rejected this version for analysis. Its literature counts remain visible only as immutable review evidence.',
        '科学审阅已要求修改，本版本不能进入分析；这里的文献数字仅作为不可变审阅记录保留。'
      );
    }
    return `<section class="gpi-lit-history" role="status"><strong>${esc(title)}</strong><span>${esc(detail)}</span></section>`;
  }
  function missingAxes(screening) {
    const missing = [];
    if (!screening.population_match) missing.push(tr('ICU population', 'ICU 人群'));
    if (!screening.exposure_match) missing.push(tr('the declared research factor', '用户指定的研究因素'));
    if (!screening.outcome_match) missing.push(tr('the declared outcome', '用户指定的结局'));
    if (!screening.design_excerpt_available) missing.push(tr('a source-backed study description', '可核对的研究摘要'));
    if (screening.publication_type_eligible === false) missing.push(tr('an eligible observational design', '合适的观察性研究类型'));
    return missing;
  }
  function reasonFor(row, kind, usages) {
    const screening = row.screening && typeof row.screening === 'object' ? row.screening : null;
    if (kind === 'excluded' && screening) {
      const missing = missingAxes(screening);
      return missing.length
        ? tr('Not accepted as direct evidence because it does not establish: ', '未作为直接依据，因为它没有同时证明：') + missing.join(tr(', ', '、')) + '。'
        : tr('Retrieved, but not accepted as direct evidence for this exact question.', '检索到了，但没有被接受为这个具体问题的直接依据。');
    }
    return tr(
      'This is a system reference and was not used as direct evidence for the current question.',
      '这是系统参考资料，没有被当作当前问题的直接依据。'
    );
  }
  function kindLabel(kind) {
    const labels = {
      direct: tr('Directly related candidate', '直接相关候选'),
      variables: tr('Variables / covariates', '变量与协变量'),
      methods: tr('Statistical method', '统计方法'),
      reporting: tr('Reporting guidance', '报告规范'),
      excluded: tr('Retrieved but not accepted', '检索到但未采用'),
      retrieval_direct: tr('Direct retrieval match · screening pending', '直接检索匹配 · 待筛选'),
      retrieval_adjacent: tr('Adjacent retrieval match · screening pending', '相邻检索匹配 · 待筛选'),
      retrieval_candidate: tr('Retrieval candidate · screening pending', '检索候选 · 待筛选'),
      reference: tr('Other system reference', '其他系统参考'),
    };
    return labels[kind] || labels.reference;
  }
  function articleKindInfo(kind) {
    const rows = {
      original_research: {
        label: tr('Original research', '原始研究'),
        supports: tr('Can provide empirical evidence for the studied population, factor, outcome, and observed findings when those definitions match the current Idea.', '如果人群、研究因素、结局和时间定义与当前 Idea 一致，它可以提供相应的实证依据。'),
        boundary: tr('Its result cannot be transferred to the current Idea until the exact definitions, time zero, analysis, and bias controls are appraised.', '在核对具体定义、时间零点、分析方法和偏倚控制前，不能把它的结果直接移植到当前 Idea。'),
      },
      systematic_review: {
        label: tr('Systematic review / meta-analysis', '系统综述 / Meta 分析'),
        supports: tr('Useful for judging whether the field is crowded, whether findings are consistent, and which gaps remain.', '适合判断该领域是否拥挤、既往结论是否一致，以及还剩下哪些证据空白。'),
        boundary: tr('It does not by itself prove that the proposed variables and timing can be reconstructed in EasyICU data.', '它本身不能证明当前研究定义能够在 EasyICU 数据中重建。'),
      },
      narrative_review: {
        label: tr('Narrative review', '叙述性综述'),
        supports: tr('Useful for the concept map, clinical background, and tracing primary studies.', '适合补充概念框架、临床背景并追踪关键原始研究。'),
        boundary: tr('It is not direct empirical support for the Idea and must not replace appraisal of the cited original studies.', '它不是当前 Idea 的直接实证依据，不能替代对其中原始研究的审阅。'),
      },
      guideline_consensus: {
        label: tr('Guideline / consensus', '指南 / 共识'),
        supports: tr('Useful for standard definitions, accepted clinical practice, and clinically meaningful outcomes.', '适合确定标准定义、现行临床实践和有意义的结局。'),
        boundary: tr('It does not establish novelty or an empirical association for the proposed Idea.', '它不能证明当前 Idea 具有创新性，也不能证明研究因素与结局存在实证关联。'),
      },
      editorial_commentary: {
        label: tr('Editorial / commentary', '社论 / 评论'),
        supports: tr('Useful for understanding why clinicians consider the problem important and which hypotheses are being debated.', '适合解释临床上为什么关注这个问题，以及目前有哪些值得讨论的假设。'),
        boundary: tr('It contains no eligible empirical result and cannot be labelled as evidence that the Idea works.', '它不提供合格的实证结果，不能标记为“该 Idea 已获证据支持”。'),
      },
      protocol: {
        label: tr('Study protocol', '研究方案 / Protocol'),
        supports: tr('Shows that a related question or design has been planned and helps compare methods.', '说明已有团队计划研究相近问题，可用于比较设计和方法。'),
        boundary: tr('A protocol has no results and cannot support an effect, feasibility, or novelty claim.', 'Protocol 没有研究结果，不能支持效应、可行性或创新性结论。'),
      },
      other: {
        label: tr('Article type pending classification', '文献类型待确认'),
        supports: tr('May provide background or a retrieval lead after appraisal.', '审阅后可能作为背景资料或进一步追踪的线索。'),
        boundary: tr('It is not treated as direct support before its study design and evidence role are confirmed.', '在确认研究设计与证据角色前，不把它视为直接支持。'),
      },
    };
    return rows[kind] || rows.other;
  }
  function whyRetrieved(row) {
    const fit = String(row.retrieval_fit || '');
    if (fit === 'direct_retrieval_fit') return tr(
      'The title or abstract matched the current ICU population and both main scientific concepts. It is a high-priority screening candidate, not yet accepted evidence.',
      '标题或摘要同时命中了当前 ICU 人群和两个主要科学概念，因此应优先审阅；它目前仍是检索候选，不是已经采纳的证据。'
    );
    if (fit === 'adjacent_retrieval_fit') return tr(
      'It covers part of the clinical setting or one scientific concept and can clarify definitions or neighboring evidence, but it does not answer the complete Idea.',
      '它覆盖了部分临床场景或其中一个科学概念，可用于补充定义或相邻证据，但不能完整回答当前 Idea。'
    );
    return tr(
      'It was returned by the bounded Idea Mining search and still requires relevance and design screening.',
      '它来自本轮有界 Idea Mining 检索，仍需核对相关性和研究设计。'
    );
  }
  function fullTextEvidence(row) {
    const fullText = row.full_text && typeof row.full_text === 'object' ? row.full_text : {};
    const spans = Array.isArray(fullText.evidence_spans) ? fullText.evidence_spans : [];
    if (fullText.status === 'reviewed' && spans.length) {
      return `<section class="gpi-lit-evidence-block"><header><h5>${esc(tr('Full-text supplement', '正文补充'))}</h5><span>${esc(tr('PMC reviewed on demand', '已按需审阅 PMC 正文'))}</span></header>
        <p class="gpi-lit-evidence-note">${esc(tr('These bounded passages supplement the abstract. They still require human appraisal in context.', '这些有界片段用于补充摘要，仍需结合上下文进行人工审阅。'))}</p>
        ${spans.map(span => `<div class="gpi-lit-evidence-span"><strong>${esc(span.label || span.section || tr('Article body', '正文'))}</strong><p>${esc(span.excerpt || '')}</p></div>`).join('')}
        ${safeUrl(fullText.url) ? `<a href="${esc(safeUrl(fullText.url))}" target="_blank" rel="noopener noreferrer">${esc(tr('Open PMC full text', '打开 PMC 正文'))}<span aria-hidden="true">↗</span></a>` : ''}</section>`;
    }
    const unavailable = row.source_review_status === 'reviewed'
      ? tr('No directly reviewable PMC full text was found. The current interpretation is limited to PubMed metadata and abstract.', '未找到可直接审阅的 PMC 正文；当前解释仅限 PubMed 元数据和摘要。')
      : row.source_review_status === 'unavailable'
        ? tr('The source could not be enriched at this time. The original PubMed page remains available for manual review.', '当前未能自动补充来源内容；仍可打开 PubMed 原页人工审阅。')
        : tr('Checking whether an openly reviewable full text is available…', '正在检查是否有可审阅的开放正文……');
    return `<section class="gpi-lit-evidence-block is-muted"><header><h5>${esc(tr('Full-text supplement', '正文补充'))}</h5><span>${esc(tr('Not available', '暂不可用'))}</span></header><p>${esc(unavailable)}</p></section>`;
  }
  function articleCard(row, indexByKey, options) {
    const config = options || {};
    const url = safeUrl(row.source_url || row.url);
    const title = displayTitle(row);
    const relevance = row.relevance || '';
    const key = row.key || '';
    const kind = config.kind || 'reference';
    const usages = Array.isArray(config.usages) ? config.usages : [];
    const genericIntent = tr('Plan decision', '计划中的科学设计决定');
    const useLabels = Array.from(new Set(usages.map(item => item.intent).filter(value => value && value !== genericIntent)));
    const excludedReason = kind === 'excluded' ? reasonFor(row, kind, usages) : '';
    if (key && indexByKey) indexByKey.set(String(key), row);
    return `<article class="gpi-lit-card ${esc(kind)}">
      <div class="gpi-lit-card-head"><span class="gpi-lit-kind">${esc(kindLabel(kind))}</span></div>
      <h4>${esc(title)}</h4>
      ${sourceMeta(row) ? `<div class="gpi-lit-meta">${sourceMeta(row)}</div>` : ''}
      ${useLabels.length ? `<div class="gpi-lit-use"><strong>${esc(tr('Used for: ', '用于计划：'))}</strong>${useLabels.map(label => `<span>${esc(label)}</span>`).join('')}</div>` : ''}
      ${excludedReason ? `<p class="gpi-lit-why"><strong>${esc(tr('Why it was not accepted: ', '未采用原因：'))}</strong>${esc(excludedReason)}</p>` : ''}
      ${relevance ? `<details class="gpi-lit-source-detail"><summary>${esc(tr('View retained source excerpt', '查看系统保留的摘要片段'))}</summary><p>${esc(relevance)}</p></details>` : ''}
      ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(tr('Open source record', '打开来源页面'))}<span aria-hidden="true">↗</span></a>` : `<span class="gpi-lit-no-link">${esc(tr('No verified source link in this artifact', '该产物没有已核验的来源链接'))}</span>`}
    </article>`;
  }
  function searchSummary(payload) {
    const search = payload.search && typeof payload.search === 'object' ? payload.search : {};
    const searched = !!search.search_conducted;
    const directCount = Number(payload.direct_comparator_count || 0);
    const title = directCount
      ? tr('Directly related studies were found, pending review', '找到了直接相关研究，仍需人工核对')
      : searched
        ? tr('No study directly supporting this question was found', '没有找到能直接支持这个问题的研究')
        : tr('No question-specific literature search was run', '尚未执行针对这个问题的文献检索');
    const sources = Array.isArray(search.sources_returning) ? search.sources_returning : [];
    const prisma = search.prisma && typeof search.prisma === 'object' ? search.prisma : null;
    const identified = Number((prisma || {}).identified || 0);
    const screened = Number((prisma || {}).screened || 0);
    const queries = search.queries && typeof search.queries === 'object' ? search.queries : {};
    const queryRows = Object.entries(queries).flatMap(([source, rows]) => (Array.isArray(rows) ? rows : []).map(query => ({ source, query })));
    return `<section class="gpi-lit-search ${searched ? 'searched' : 'curated'}">
      <div><span class="gpi-lit-status-dot" aria-hidden="true"></span><strong>${esc(title)}</strong></div>
      ${payload.research_question ? `<p class="gpi-lit-question"><strong>${esc(tr('Research question: ', '研究问题：'))}</strong>${esc(payload.research_question)}</p>` : ''}
      <p>${esc(directCount
        ? tr(`${directCount} candidate(s) matched the ICU population, exposure and outcome. They are not automatically authoritative.`, `${directCount} 篇候选同时匹配 ICU 人群、研究因素和结局，但还不能自动视为可靠依据。`)
        : searched
          ? tr(`${identified} record(s) were retrieved and ${screened} screened; none matched all parts of the question. The plan must not claim direct literature support.`, `共检索到 ${identified} 篇候选、完成 ${screened} 篇筛选；没有一篇同时匹配问题的全部要素。当前计划不能声称有同题文献直接支持。`)
          : tr('Only general definitions or reporting guidance may be present.', '当前最多只有通用定义或报告规范，不能当作同题研究证据。'))}</p>
      <div class="gpi-lit-counts"><span>${esc(tr('Retrieved', '检索到'))}<strong>${identified}</strong></span><span>${esc(tr('Screened', '已筛选'))}<strong>${screened}</strong></span><span>${esc(tr('Direct matches', '直接匹配'))}<strong>${directCount}</strong></span></div>
      ${sources.length ? `<small>${esc(tr('Literature database: ', '检索来源：'))}${sources.map(esc).join(', ')}</small>` : ''}
      ${queryRows.length ? `<details class="gpi-lit-queries"><summary>${esc(tr('Technical search query (for audit)', '专业检索式（供复核）'))}</summary>${queryRows.map(row => `<div><strong>${esc(row.source)}</strong><code>${esc(row.query)}</code></div>`).join('')}</details>` : ''}
    </section>`;
  }
  function planMap(payload) {
    const buckets = evidenceBuckets(payload);
    const usage = boundUsage(payload);
    const indexByKey = new Map();
    const searched = Boolean(payload.search && payload.search.search_conducted);
    const total = buckets.direct.length + buckets.variables.length + buckets.methods.length + buckets.reporting.length;
    if (!total) return `<section class="gpi-lit-map empty"><header><h3>${esc(tr('Literature grouped by purpose', '文献按用途分组'))}</h3><span>0</span></header><p>${esc(tr('No article is attached to this plan yet.', '目前还没有文献绑定到这份计划。'))}</p></section>`;
    return `<section class="gpi-lit-map">
      <header><h3>${esc(tr('Literature grouped by purpose', '文献按用途分组'))}</h3><span>${esc(tr(`${total} source(s)`, `${total} 篇`))}</span></header>
      ${articleGroup(tr('Studies directly related to the research question', '研究问题直接相关'), buckets.direct, 'direct', usage, indexByKey, { note: tr('This group is reserved for candidates matching the current study population, research factor, and outcome. Review the original article for its exact definitions, time window, and analysis design.', '此分组仅收录同时匹配当前研究人群、研究因素和结局的候选；具体定义、时间窗与分析方法仍需打开原文核对。'), emptyText: searched ? tr('No article passed direct-match screening.', '本次检索暂无文章通过筛选。') : tr('Question-specific search has not run yet.', '尚未执行针对这个问题的文献检索。') })}
      ${articleGroup(tr('Evidence for variables and covariates', '变量、特征与协变量依据'), buckets.variables, 'variables', usage, indexByKey, { collapsed: true, note: tr('Used to define the population, research factor, outcome, and adjustment variables; not to copy effect estimates.', '用于定义研究人群、研究因素、结局和调整变量，不复制文献的效应值。'), hideWhenEmpty: true })}
      ${articleGroup(tr('Evidence for statistical methods', '统计方法依据'), buckets.methods, 'methods', usage, indexByKey, { collapsed: true, note: tr('Used for time alignment, missing data, functional form, and sensitivity analyses.', '用于时间对齐、缺失数据、函数形式和敏感性分析。'), hideWhenEmpty: true })}
      ${articleGroup(tr('Reporting guidelines', '报告规范'), buckets.reporting, 'reporting', usage, indexByKey, { collapsed: true, note: tr('These sources govern transparent reporting only; they do not prove the study association.', '这些文献只规范透明报告，不证明当前研究因素与结局存在关联。'), hideWhenEmpty: true })}
    </section>`;
  }
  function articleGroup(title, rows, kind, usage, indexByKey, options) {
    if (!rows.length && options && options.hideWhenEmpty) return '';
    const cards = rows.map(row => articleCard(row, indexByKey, { kind, usages: usage.get(sourceKey(row)) || [] })).join('');
    if (options && options.collapsed) {
      return `<details class="gpi-lit-group collapsed"><summary><strong>${esc(title)}</strong><span>${rows.length}</span></summary><p class="gpi-lit-group-note">${esc(options.note || '')}</p>${cards}</details>`;
    }
    return `<section class="gpi-lit-group"><header><h3>${esc(title)}</h3><span>${rows.length}</span></header>${options && options.note ? `<p class="gpi-lit-group-note">${esc(options.note)}</p>` : ''}${cards || `<div class="gpi-lit-empty">${esc(options && options.emptyText || tr('No articles in this group.', '这一组没有文献。'))}</div>`}</section>`;
  }
  function renderArtifact(payload, meta) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const citations = Array.isArray(p.citations) ? p.citations : [];
    const indexByKey = new Map();
    const usage = boundUsage(p);
    const directKeys = new Set(Array.isArray(p.direct_comparator_keys) ? p.direct_comparator_keys.map(String) : []);
    const excluded = citations.filter(row => row.screening && !directKeys.has(String(row.key || '')) && !usage.has(String(row.key || '')));
    const reference = citations.filter(row => !row.screening && !directKeys.has(String(row.key || '')) && !usage.has(String(row.key || '')));
    return `<div class="gpi-lit-view">
      ${artifactStatusBanner({ ...(meta || {}), runId: String((meta && meta.runId) || p.run_id || '') })}
      ${searchSummary(p)}
      <div class="gpi-lit-boundary" role="note"><strong>${esc(tr('How to read this page', '这一页怎么看'))}</strong><span>${esc(tr('Only the direct-match group addresses the user question. Reporting standards and variable definitions can shape a plan, but they cannot prove an association between the declared exposure and outcome.', '只有“直接相关候选”能回答用户的问题。报告规范和变量定义可以帮助制定计划，但不能证明用户指定的研究因素与结局有关。'))}</span></div>
      ${evidenceCoverage(p)}
      ${planMap(p)}
      ${articleGroup(tr('Retrieved articles not accepted as direct evidence', '检索到但未作为直接依据的文章'), excluded, 'excluded', usage, indexByKey, { collapsed: true, note: tr('Open only when reviewing why candidates were rejected.', '需要核对筛选原因时再展开。') })}
      ${articleGroup(tr('Other system references not used as direct evidence', '系统参考库里的其他资料'), reference, 'reference', usage, indexByKey, { collapsed: true, note: tr('Older definition or method papers may appear here. They are not direct evidence for the current research question and are collapsed by default.', '这里可能包含较早的定义或方法文献；它们不是当前研究问题的直接证据，默认收起。') })}
    </div>`;
  }
  function renderSource(resource) {
    const row = {
      title: resource.title || resource.label,
      year: resource.year,
      venue: resource.venue,
      relevance: resource.relevance,
      doi: resource.doi,
      pmid: resource.pmid,
      source_url: resource.url,
      retrieval_fit: resource.retrieval_fit,
      retrieval_rationale: resource.retrieval_rationale,
      abstract_excerpt: resource.abstract_excerpt,
      publication_types: Array.isArray(resource.publication_types) ? resource.publication_types : [],
      article_kind: resource.article_kind,
      full_text: resource.full_text,
      source_review_status: resource.source_review_status,
    };
    const fit = String(row.retrieval_fit || '');
    const kind = fit === 'direct_retrieval_fit'
      ? 'retrieval_direct'
      : fit === 'adjacent_retrieval_fit'
        ? 'retrieval_adjacent'
        : 'retrieval_candidate';
    const typeInfo = articleKindInfo(String(row.article_kind || 'other'));
    const abstract = String(row.abstract_excerpt || '').trim();
    return `<div class="gpi-lit-view source-only">
      <section class="gpi-lit-search searched"><div><span class="gpi-lit-status-dot" aria-hidden="true"></span><strong>${esc(tr('PubMed search result', 'PubMed 检索结果'))}</strong></div><p>${esc(tr('This metadata came from the user-authorized Idea Mining search receipt.', '此元数据来自用户授权的 Idea Mining 检索回执。'))}</p></section>
      <article class="gpi-lit-card ${esc(kind)}">
        <div class="gpi-lit-card-head"><span class="gpi-lit-kind">${esc(kindLabel(kind))}</span><span class="gpi-lit-type">${esc(typeInfo.label)}</span></div>
        <h4>${esc(displayTitle(row))}</h4>
        ${sourceMeta(row) ? `<div class="gpi-lit-meta">${sourceMeta(row)}</div>` : ''}
        <section class="gpi-lit-interpretation"><h5>${esc(tr('Why this article is shown', '为什么收录这篇文献'))}</h5><p>${esc(whyRetrieved(row))}</p></section>
        <div class="gpi-lit-interpretation-grid">
          <section><h5>${esc(tr('What it can support', '它能支持什么'))}</h5><p>${esc(typeInfo.supports)}</p></section>
          <section><h5>${esc(tr('What it cannot support', '它不能支持什么'))}</h5><p>${esc(typeInfo.boundary)}</p></section>
        </div>
        <section class="gpi-lit-evidence-block"><header><h5>${esc(tr('Abstract evidence', '摘要证据'))}</h5><span>${esc(abstract ? tr('PubMed abstract', 'PubMed 摘要') : tr('No abstract', '暂无摘要'))}</span></header>${abstract ? `<p class="gpi-lit-abstract">${esc(abstract)}</p>` : `<p>${esc(tr('PubMed did not provide an abstract for this record.', 'PubMed 未为该记录提供摘要。'))}</p>`}</section>
        ${fullTextEvidence(row)}
        ${safeUrl(row.source_url) ? `<a href="${esc(safeUrl(row.source_url))}" target="_blank" rel="noopener noreferrer">${esc(tr('Open source record', '打开来源页面'))}<span aria-hidden="true">↗</span></a>` : ''}
      </article>
      <div class="gpi-lit-boundary" role="note"><strong>${esc(tr('Evidence boundary', '证据边界'))}</strong><span>${esc(tr('Abstract and bounded full-text passages help appraisal; they do not automatically establish novelty, causality, data feasibility, or publication authority.', '摘要和有界正文片段用于辅助审阅；它们不会自动证明创新性、因果关系、数据可行性或发表权限。'))}</span></div>
    </div>`;
  }

  window.EU_GUIDED_PI_LITERATURE = { renderArtifact, renderSource, safeUrl };
})();
