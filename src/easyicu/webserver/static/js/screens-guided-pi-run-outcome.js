/* Owner: Guided Pi run-outcome card widget. */
/* Guided Copilot durable completed-run card owner.
   A validated analysis remains visible after refresh even when publication
   review stays closed. It renders only host-projected artifact references. */
(function () {
  'use strict';

  function create(deps) {
    const APPEND = window.EasyICU.guidedPi.optional('resultAppend')?.create(deps);
    const {
      tr, esc, iconHtml, resourceButton, api, projectId, host, canPreview,
      preview, workflowContext, errorText, recordHostAction, onError, session, manuscriptNotPassedText,
    } = deps;
    const labels = {
      'result_tables.json': ['View result tables', '查看结果表'],
      'figure_gallery.json': ['View analysis figures', '查看分析图表'],
      'manuscript_provenance.json': ['Preview evidence-bound article', '预览证据绑定文章'],
      'manuscript_draft.json': ['View report draft and revision status', '查看报告草稿与修订状态'],
      'scientific_readiness.json': ['View scientific review', '查看科学审阅'],
    };
    let activeReviewTab = 0;
    const dismissedFollowUps = new Set();
    const answerOwner = window.EasyICU.guidedPi.optional('runAnswer');
    const ANSWER = answerOwner && typeof answerOwner.create === 'function'
      ? answerOwner.create({ tr, esc, api, projectId, resourceButton }) : null;
    // The host's itemised review (approach, materials, number check, limitations).
    const itemsOwner = window.EasyICU.guidedPi.optional('reviewItems');
    const REVIEW_ITEMS = itemsOwner && typeof itemsOwner.create === 'function'
      ? itemsOwner.create({ tr, esc, api, projectId, resourceButton }) : null;
    let scientificReview = { key: '', payload: null, loading: false, error: '' };
    // Display copy only: review codes, repair eligibility and approvals remain
    // owned by the saved scientific review and workflow.
    const reviewCopy = Object.freeze({
      IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED: ['Prior-art evidence is not established', '既往研究依据尚未建立',
        'Review the literature search and same-topic studies before judging novelty.', '人工核对文献检索与同主题研究，再判断创新性。'],
      SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN: ['Major scientific revision remains open', '重大科学修订尚未关闭',
        'Resolve or adjudicate each major finding and obtain an updated review.', '逐项处理或裁定重大问题，再取得更新的审阅记录。'],
      NOVELTY_POSITIONING_NOT_ESTABLISHED: ['Novelty positioning is not established', '创新性定位尚未建立',
        'Compare the population, exposure timing, outcome, methods, setting and contribution with retained studies; seek expert review.', '对照保留文献，说明人群、暴露时间、结局、方法、场景与贡献的实质差异，请专业人员审阅。'],
      REPORTING_CHECKLIST_ITEMS_OPEN: ['Reporting checklist items remain open', '报告规范项目尚未关闭',
        'Address each checklist item or document why it does not apply.', '逐项补充报告内容，或记录有依据的“不适用”判断。'],
      PAPER_AUTHORITY_NOT_GRANTED: ['Publication use is not authorized', '尚未取得投稿使用权限',
        'Complete the scientific, reporting, evidence and human reviews before external use.', '对外使用前，完成科学、报告、证据及人工审阅。'],
      MODEL_INCOMPLETE: ['Prespecified model is incomplete', '预设模型尚未完成',
        'Review the missing model, revise the analysis plan and seek approval before running it.', '核对缺少的模型，修订分析计划，经确认后再运行。'],
      COHORT_SOURCE_SCOPE_NOT_EXPLICIT: ['Cohort source scope is not explicit', '队列来源范围尚未写明',
        'Document the source population, selection flow, coverage and final denominator.', '补充来源人群、筛选流程、数据覆盖和最终分母。'],
      DIRECT_COMPARATOR_SCREENING_NOT_ESTABLISHED: ['Direct-comparator screening is incomplete', '直接对照文献筛选未完成',
        'Screen each study against the population, exposure and outcome and retain the decision.', '按人群、暴露和结局逐条筛选文献，保留直接对照的判断依据。'],
      POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED: ['Exposure is ascertained after time zero', '暴露判定晚于时间零点',
        'Review follow-up timing and early events; agree on a revised design before running it.', '人工核对随访起点及早期事件，确认修订方案后再分析。'],
      REPEATED_STAY_DEPENDENCE_UNRESOLVED: ['Repeated ICU stays are not handled', '同一患者多次入住未处理',
        'Check whether the source can supply a verified patient identifier; otherwise retain this limitation.', '核对数据源能否提供经核验的患者标识；无法提供时，明确保留这一局限。'],
      ROBUSTNESS_AXES_TOO_NARROW: ['Too few robustness axes', '稳健性分析维度不足',
        'Choose supported alternative definitions, windows or cohorts for a new plan, or retain the analysis-only scope.', '确认是否在新方案中增加数据支持的替代定义、时间窗或人群，或保留当前分析范围。'],
      MANUSCRIPT_CORE_SECTIONS_TOO_THIN: ['Core manuscript sections are too thin', '稿件核心章节过于单薄',
        'Expand the sections using this run’s evidence and verified references.', '依据本次结果和已核对文献，补充稿件中的方法、结果、解释与局限。'],
      RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED: ['Recent comparable studies need review', '近期同类研究尚未核对',
        'Review the search and screening for recent comparable work; retain valid older sources.', '人工复核检索与筛选，确认是否遗漏近期同类研究；保留仍有效的早期文献。'],
      ROBUSTNESS_REPLAY_DUPLICATES_PRIMARY: ['Robustness checks repeat the primary analysis', '稳健性分析重复了主要分析',
        'Record this as an equivalence check; agree on a distinct, supported specification for further analysis.', '将此次结果记录为等价核对，另行确认有实质差异且数据支持的分析方案。'],
      MANUSCRIPT_EXACT_LITERATURE_BINDING_INCOMPLETE: ['Manuscript citations need verification', '稿件引用尚未对应到文献',
        'Match each citation to the retained literature and revise unsupported statements.', '逐条将稿件引用对应到保留文献，修订缺乏依据的表述。'],
      MANUSCRIPT_REQUIRED_SECTIONS_INCOMPLETE: ['Required manuscript sections are missing', '稿件缺少必要章节',
        'Complete the missing sections using the reporting checklist and this run’s evidence.', '按报告清单补齐缺失章节，内容以本次研究证据为准。'],
      MANUSCRIPT_RENDER_QA_INCOMPLETE: ['Manuscript PDF checks are incomplete', '论文 PDF 尚未完成核对',
        'Generate the PDF for the current draft and verify its figures, numbers and layout.', '生成当前草稿对应的 PDF，核对图表、数字及排版并保留核对记录。'],
      FORMAL_PREREGISTRATION_NOT_ESTABLISHED: ['Study preregistration is unverified', '研究预注册尚未核实',
        'Manually verify external protocol registration, its date and its match to the approved plan; report writing is not preregistration.', '人工核对外部研究注册、注册时间及其与已批准方案的一致性；补写报告不能替代预注册。'],
    });
    function findingCopy(value) {
      const copy = reviewCopy[String(value.code || '')];
      return { title: copy ? tr(copy[0], copy[1])
        : tr('An analysis issue needs manual review', '有一项分析问题需人工核对'),
      action: copy ? tr(copy[2], copy[3]) : tr(
        'Open the original review and ask a qualified reviewer to assess this issue and the evidence it needs.',
        '请展开原始审阅意见，由专业人员人工核对具体问题及所需证据。') };
    }

    function scientificReviewReference(latestRun) {
      return (Array.isArray(latestRun && latestRun.artifact_refs) ? latestRun.artifact_refs : [])
        .find(row => row && row.run_id === latestRun.run_id
          && row.artifact === 'scientific_readiness.json'
          && /^[a-f0-9]{64}$/.test(String(row.sha256 || '')));
    }

    /* The card is first drawn with a reading placeholder; once the run's own
       numbers are read, the card is redrawn in place with its answer, file
       notes and follow-ups. */
    async function loadAnswer(latestRun, workflow) {
      if (!ANSWER || !resultsAvailable(latestRun, workflow)) return;
      if (!await ANSWER.load(latestRun)) return;
      const root = typeof host === 'function' ? host() : null;
      const card = root && Array.from(root.querySelectorAll('[data-gpi-run-outcome]'))
        .find(node => node.dataset.gpiRunOutcome === String(latestRun.run_id || ''));
      if (card) card.outerHTML = render(latestRun, workflow);
    }

    // The workflow owner calls this once per projected workflow; the run's
    // answer and its scientific review load side by side.
    async function loadScientificReview(latestRun, workflow) {
      void loadAnswer(latestRun, workflow);
      if (!resultsAvailable(latestRun, workflow) || typeof projectId !== 'function') return;
      void APPEND?.load(latestRun);
      if (REVIEW_ITEMS) void REVIEW_ITEMS.load(latestRun).then(changed => { if (changed) repaintReview(latestRun); });
      const ref = scientificReviewReference(latestRun);
      if (!ref) return;
      const expectedProjectId = projectId();
      const key = `${expectedProjectId}:${ref.run_id}:${ref.sha256}`;
      if (scientificReview.key === key) return;
      scientificReview = { key, payload: null, loading: true, error: '' };
      try {
        const client = typeof api === 'function' ? api() : null;
        if (!client || !client.loadPiCopilotResearchArtifact) throw new Error('scientific_review_api_unavailable');
        const response = await client.loadPiCopilotResearchArtifact(
          expectedProjectId, ref.run_id, ref.artifact, ref.sha256,
        );
        if (scientificReview.key !== key || projectId() !== expectedProjectId) return;
        const payload = response && response.payload;
        if (!response.ok || !payload || payload.run_id !== ref.run_id
          || !Array.isArray(payload.domains) || !Array.isArray(payload.findings)) {
          throw new Error('scientific_review_payload_invalid');
        }
        scientificReview = { key, payload, loading: false, error: '' };
      } catch (_error) {
        if (scientificReview.key !== key || projectId() !== expectedProjectId) return;
        scientificReview = { key, payload: null, loading: false, error: 'unavailable' };
      }
      repaintReview(latestRun);
    }
    function repaintReview(latestRun) {
      const root = typeof host === 'function' ? host() : null;
      const current = root && root.querySelector('.gpi-review-summary');
      if (current && current.dataset.gpiReviewRun === latestRun.run_id) {
        current.outerHTML = renderReviewSummary(latestRun);
      }
    }

    function selectReviewTab(index) {
      if (!Number.isInteger(index) || index < 0 || index > 3) return false;
      activeReviewTab = index;
      return true;
    }

    function resultsAvailable(latestRun, workflow) {
      if (!latestRun || latestRun.present !== true || latestRun.analysis_results_available !== true) return false;
      const stages = Array.isArray(workflow && workflow.stages) ? workflow.stages : [];
      const analysis = stages.find(stage => stage && stage.id === 'analysis');
      return Boolean(analysis && ['complete', 'review_required'].includes(String(analysis.status || '')));
    }

    // One host-derived collection powers both the persistent shelf and reader.
    // A new report revision must never fall back to the historical source PDF.
    function collection(latestRun, workflow) {
      if (!resultsAvailable(latestRun, workflow) || !latestRun.run_id) return [];
      const resources = (Array.isArray(latestRun.artifact_refs) ? latestRun.artifact_refs : [])
        .filter(row => row && row.run_id === latestRun.run_id);
      const ledger = resources.find(row => row.artifact === 'evidence_ledger.json');
      const rows = [];
      const add = (name, en, zh) => {
        const ref = resources.find(row => row.artifact === name);
        if (ref) rows.push({ ...ref, label: tr(en, zh) });
      };
      const addReport = (name, en, zh) => {
        if (ledger) rows.push({
          kind: 'research_report', run_id: latestRun.run_id, artifact: name,
          sha256: ledger.sha256, label: tr(en, zh), media_type: 'application/json',
        });
      };
      addReport('full_analysis_report.json', 'Study overview', '研究总览');
      add('result_tables.json', 'Result tables', '结果表');
      if (latestRun.figure_count !== 0) add('figure_gallery.json', 'Figures', '图表');
      if (latestRun.manuscript_ready === true || latestRun.report_revision_ready === true) {
        addReport('article_report.json', 'Article', '文章');
      }
      add('literature_evidence.json', 'References', '文献');
      if (latestRun.report_revision_pdf_ready === true) add('manuscript_revision.pdf', 'Current PDF', '当前 PDF');
      else if (latestRun.manuscript_ready === true && latestRun.report_revision_ready !== true) {
        add('manuscript_scaffold.pdf', 'Manuscript PDF', '稿件 PDF');
      }
      add('scientific_readiness.json', 'Scientific review', '科学审阅');
      return rows;
    }

    function renderShelf(latestRun, workflow, query = '', options = {}) {
      const rows = collection(latestRun, workflow);
      if (!rows.length) return '';
      const files = (Array.isArray(latestRun.artifact_refs) ? latestRun.artifact_refs : [])
        .filter(row => row && row.run_id === latestRun.run_id
          && (/^[A-Za-z0-9_.-]+\.json$/.test(String(row.artifact || ''))
            || ['manuscript_revision.pdf', 'manuscript_scaffold.pdf'].includes(row.artifact)))
        .filter(row => row.artifact !== 'manuscript_scaffold.pdf'
          || latestRun.report_revision_ready !== true)
        .filter(row => row.artifact !== 'manuscript_revision.pdf'
          || latestRun.report_revision_pdf_ready === true);
      const preferred = ['result_tables.json', 'figure_gallery.json', 'manuscript_revision.pdf',
        'scientific_readiness.json', 'manuscript_draft.json', 'literature_evidence.json'];
      files.sort((left, right) => {
        const leftRank = preferred.indexOf(left.artifact);
        const rightRank = preferred.indexOf(right.artifact);
        return (leftRank < 0 ? preferred.length : leftRank) - (rightRank < 0 ? preferred.length : rightRank)
          || String(left.artifact).localeCompare(String(right.artifact));
      });
      const sort = ['recommended', 'name', 'size'].includes(options.sort) ? options.sort : 'recommended';
      const view = options.view === 'details' ? 'details' : 'list';
      if (sort === 'name') files.sort((left, right) => String(left.artifact).localeCompare(String(right.artifact)));
      if (sort === 'size') files.sort((left, right) => Number(right.size || 0) - Number(left.size || 0)
        || String(left.artifact).localeCompare(String(right.artifact)));
      const needle = String(query || '').trim().toLocaleLowerCase();
      // A reader sees what each file is; the file name stays beside it for
      // the details view and search. Audit and provenance files keep their
      // place in a folded group instead of leading the list.
      const title = row => {
        const render = window.AGENT_RENDER;
        return render && typeof render.artifactTitle === 'function' ? String(render.artifactTitle(row.artifact) || row.artifact) : row.artifact;
      };
      const matches = row => `${row.artifact} ${title(row)}`.toLocaleLowerCase().includes(needle);
      const filtered = files.filter(matches);
      const size = value => Number.isFinite(value) && value >= 0
        ? (value < 1024 ? `${value} B` : value < 1024 * 1024
          ? `${(value / 1024).toFixed(1)} KB` : `${(value / (1024 * 1024)).toFixed(1)} MB`) : '';
      const download = row => {
        if (!/^[a-f0-9]{64}$/.test(String(row.sha256 || ''))) return '';
        const client = typeof api === 'function' ? api() : null;
        if (!client || typeof projectId !== 'function') return '';
        const url = row.artifact.endsWith('.pdf') && client.piCopilotResearchDocumentUrl
          ? client.piCopilotResearchDocumentUrl(projectId(), row.run_id, row.artifact, row.sha256)
          : row.artifact.endsWith('.json') && client.piCopilotResearchArtifactDownloadUrl
            ? client.piCopilotResearchArtifactDownloadUrl(projectId(), row.run_id, row.artifact, row.sha256) : '';
        const isPdf = row.artifact.endsWith('.pdf');
        const name = isPdf ? row.artifact : row.artifact.replace(/\.json$/, '.review.json');
        const title = isPdf ? tr('Download PDF', '下载 PDF')
          : tr('Download safe review copy', '下载脱敏审阅副本');
        return url ? `<a class="gpi-result-download" href="${esc(url)}" download="${esc(name)}" aria-label="${esc(title + ' ' + row.artifact)}" title="${title}">${iconHtml('download', 14)}</a>` : '';
      };
      const contracts = window.EU_GUIDED_CONTRACTS || {};
      const when = latestRun.updated_at && typeof contracts.fmtRunTime === 'function' ? String(contracts.fmtRunTime(latestRun.updated_at) || '') : '';
      const fileRow = row => `<div class="gpi-study-file" data-gpi-result-file="${esc(row.artifact)}" data-gpi-result-title="${esc(title(row))}"${needle && !matches(row) ? ' hidden' : ''}><span aria-hidden="true">${iconHtml(resourceIcon(row), 15)}</span>${resourceButton(row, title(row))}<small class="gpi-study-file-name">${esc(row.artifact)}</small>${size(row.size) ? `<small>${esc(size(row.size))}</small>` : ''}${download(row)}</div>`;
      const readerFiles = files.filter(row => READER_FILES.has(row.artifact));
      const auditFiles = files.filter(row => !READER_FILES.has(row.artifact));
      return `<section class="gpi-study-results" aria-label="${esc(tr('Current study results', '当前研究成果'))}">
        <div class="gpi-study-results-heading"><span>${tr('Current run', '本次运行')}${when ? ` · <time>${esc(when)}</time>` : ''} · ${files.length} ${tr('files', '个文件')}</span></div>
        <div class="gpi-study-results-primary">${rows.slice(0, 3).map(row => resourceButton(row, row.label)).join('')}</div>
        <div class="gpi-results-toolbar" role="group" aria-label="${tr('Result file display', '成果文件显示方式')}">
          <button type="button" data-gpi-results-view="list" aria-pressed="${view === 'list'}" title="${tr('Compact list', '简洁列表')}">${iconHtml('rows', 14)}<span>${tr('List', '列表')}</span></button>
          <button type="button" data-gpi-results-view="details" aria-pressed="${view === 'details'}" title="${tr('File details', '文件详情')}">${iconHtml('layers', 14)}<span>${tr('Details', '详情')}</span></button>
          <label><span class="shell-sr-only">${tr('Sort files', '文件排序')}</span><select data-gpi-results-sort><option value="recommended"${sort === 'recommended' ? ' selected' : ''}>${tr('Recommended', '推荐')}</option><option value="name"${sort === 'name' ? ' selected' : ''}>${tr('Name', '名称')}</option><option value="size"${sort === 'size' ? ' selected' : ''}>${tr('Size', '大小')}</option></select></label>
        </div>
        ${files.length > 5 ? `<label class="gpi-results-search"><span class="shell-sr-only">${tr('Filter result files', '筛选成果文件')}</span><input type="search" data-gpi-results-search placeholder="${tr('Filter files…', '搜索文件…')}" value="${esc(query)}"></label>` : ''}
        <div class="gpi-study-results-links" data-gpi-results-list data-view="${view}">${readerFiles.map(fileRow).join('')}${auditFiles.length ? `<details class="gpi-study-results-audit"${needle && auditFiles.some(matches) ? ' open' : ''}><summary>${tr('Audit and provenance files', '审计与溯源文件')} · ${auditFiles.length}</summary>${auditFiles.map(fileRow).join('')}</details>` : ''}</div>
        <p class="gpi-aside-empty" data-gpi-results-empty${filtered.length ? ' hidden' : ''}>${tr('No matching files', '没有匹配的文件')}</p>
        <small>${esc(tr('Files from the current run · publication review pending', '当前运行文件 · 投稿审阅尚未完成'))}</small>
        <details class="gpi-study-results-source"><summary>${tr('Result source', '成果来源')}</summary><code>${esc(latestRun.run_id)}</code></details>
      </section>`;
    }

    // Files a reader opens; everything else in the run is audit and provenance.
    const READER_FILES = new Set(['result_tables.json', 'figure_gallery.json', 'manuscript_revision.pdf',
      'manuscript_scaffold.pdf', 'manuscript_draft.json', 'literature_evidence.json',
      'scientific_readiness.json', 'agent_plan.json', 'cohort_summary.json']);

    function resourceIcon(row) {
      const names = { 'full_analysis_report.json': 'layers', 'result_tables.json': 'rows',
        'figure_gallery.json': 'viz', 'literature_evidence.json': 'link',
        'scientific_readiness.json': 'shield' };
      return names[row.artifact] || 'file';
    }

    function renderDeliverables(latestRun, workflow) {
      const descriptions = {
        'full_analysis_report.json': tr('Analysis summary and supporting evidence', '分析汇总与对应证据'),
        'result_tables.json': tr('Estimates and source values', '查看估计与原始数值'),
        'figure_gallery.json': tr('Browse analysis figures', '浏览本次分析图件'),
        'article_report.json': tr('Read the evidence-bound draft', '阅读证据绑定正文'),
        'literature_evidence.json': tr('Retrieved references and evidence', '检索文献与证据记录'),
        'manuscript_revision.pdf': tr('Current report for download', '当前修订版，可下载'),
        'manuscript_scaffold.pdf': tr('Manuscript draft for download', '稿件草稿，可下载'),
      };
      const rows = collection(latestRun, workflow).filter(row => descriptions[row.artifact]);
      const note = row => (ANSWER && ANSWER.fileNote(latestRun, row.artifact)) || descriptions[row.artifact];
      return rows.length ? `<table class="gpi-deliverables"><caption>${tr('Results from this run', '本次研究产物')}</caption><thead><tr><th>${tr('File', '文件')}</th><th>${tr('Contents', '内容')}</th></tr></thead><tbody>${rows.map(row => `<tr><td>${resourceButton(row, row.label)}</td><td>${esc(note(row))}</td></tr>`).join('')}</tbody></table>` : '';
    }

    // The review the workflow offers for this run now, or null.
    function reviewOffer(latestRun, workflow) {
      const stage = (workflow && workflow.stages || []).find(row => row.id === workflow.current_stage);
      if (!stage || !['ready', 'review_required'].includes(stage.status)) return null;
      const name = stage.id === 'interpretation' ? 'full_analysis_report.json'
        : stage.id === 'manuscript' ? 'article_report.json' : '';
      const resource = collection(latestRun, workflow).find(row => row.artifact === name);
      return resource ? { resource, stage: stage.id } : null;
    }
    function renderReviewAction(latestRun, workflow) {
      const offer = reviewOffer(latestRun, workflow);
      return offer ? resourceButton(offer.resource, offer.stage === 'interpretation'
        ? tr('Review results', '审阅研究结果') : tr('Review manuscript', '审阅稿件')) : '';
    }

    /* 过程说明: one paragraph from this run's receipts (screens-guided-pi-
       job-narration.js counts them). Every agent-run job of this run adds
       its progress: the planning job's corrections, the analysis job's
       repairs and continuations. */
    function processNotes(latestRun) {
      const narration = window.EasyICU.guidedPi.optional('jobNarration');
      const current = typeof session === 'function' ? session() : null;
      if (!narration || !latestRun || !latestRun.run_id) return '';
      const jobs = (Array.isArray(current && current.archived_child_jobs) ? current.archived_child_jobs : [])
        .filter(job => job && job.kind === 'agent-run' && job.run_id === latestRun.run_id);
      const text = narration.processNotes({ progress: jobs.flatMap(job => window.EasyICU.guidedPi.jobProgress(job)) }, latestRun, tr);
      return text ? `<section class="gpi-run-notes" aria-label="${esc(tr('Notes on the run', '过程说明'))}"><h4>${esc(tr('Notes on the run', '过程说明'))}</h4><p>${esc(text)}</p></section>` : '';
    }

    /* The answer's action row: copy its text, the review the workflow offers
       for this run (only then), and the run record's count, which opens the
       right rail's record. No rating: EasyICU keeps no feedback ledger. */
    function answerActions(latestRun, workflow) {
      const offer = reviewOffer(latestRun, workflow);
      const runs = Array.isArray(workflow && workflow.runs) ? workflow.runs.length : 0;
      return `<div class="gpi-answer-actions" role="group" aria-label="${esc(tr('Answer actions', '回答操作'))}">
        <button type="button" class="gpi-answer-action" data-gpi-run-answer-copy>${iconHtml('copy', 13)}<span>${esc(tr('Copy', '复制'))}</span></button>
        ${offer ? resourceButton(offer.resource, tr('Review', '审查'), { className: 'gpi-answer-action' }) : ''}
        ${runs ? `<button type="button" class="gpi-answer-action" data-gpi-open-run-record>${iconHtml('list', 13)}<span>${esc(tr(`Run record ${runs}`, `运行记录 ${runs}`))}</span></button>` : ''}
      </div>`;
    }

    function followUps(latestRun, workflow) {
      const specific = ANSWER ? ANSWER.followUps(latestRun) : null;
      if (specific && specific.length) return specific;
      const available = new Set(collection(latestRun, workflow).map(row => row.artifact));
      const suggestions = [];
      if (available.has('result_tables.json')) suggestions.push(tr(
        'Explain the most important estimates in the current result tables and their uncertainty, using only this run’s evidence.',
        '请依据本次运行的证据，解释结果表中最重要的估计及其不确定性。'));
      if (available.has('figure_gallery.json')) suggestions.push(tr(
        'Check whether each current figure agrees with the result tables and identify any presentation problems.',
        '请逐图核对本次图表与结果表是否一致，并指出展示上的问题。'));
      if (available.has('scientific_readiness.json')) suggestions.push(tr(
        'Use this run’s scientific-review record to list unresolved issues and the evidence needed to address them.',
        '请依据本次科学审阅记录，列出未解决的问题和所需证据。'));
      if (available.has('article_report.json')) suggestions.push(tr(
        'Review the current article against its evidence and suggest the next revision without changing the scientific scope.',
        '请对照现有证据审阅当前文章，提出下一轮修订建议，不改变科学问题范围。'));
      return suggestions;
    }

    function renderFollowUps(latestRun, workflow) {
      const suggestions = followUps(latestRun, workflow);
      const runId = String(latestRun && latestRun.run_id || 'current');
      const visible = suggestions.map((prompt, index) => ({ prompt, index }))
        .filter(row => !dismissedFollowUps.has(`${runId}:${row.prompt}`));
      if (!visible.length) return '';
      const disabled = typeof canPreview === 'function' && !canPreview() ? ' disabled' : '';
      return `<details class="gpi-followups" open><summary class="gpi-followups-head">${iconHtml('help', 14)}<span>${tr('Follow-up questions', '后续问题')}</span></summary><div class="gpi-followups-list">${visible.map(({ prompt, index }) => `<div class="gpi-followup-row"><button type="button" class="gpi-followup-prompt" data-gpi-followup="${index}"${disabled}><span aria-hidden="true">↳</span>${esc(prompt)}</button><div class="gpi-followup-actions"><button type="button" class="gpi-icon-action" data-gpi-followup-new="${index}"${disabled} title="${esc(tr('Ask in a new conversation', '在新对话中追问'))}" aria-label="${esc(tr('Ask in a new conversation', '在新对话中追问'))}">${iconHtml('arrow', 13)}</button><button type="button" class="gpi-icon-action" data-gpi-followup-dismiss="${index}" title="${esc(tr('Dismiss suggestion', '关闭建议'))}" aria-label="${esc(tr('Dismiss suggestion', '关闭建议'))}">${iconHtml('close', 13)}</button></div></div>`).join('')}</div></details>`;
    }

    function dismissFollowUp(index, latestRun, workflow) {
      const suggestions = followUps(latestRun, workflow);
      const prompt = suggestions[Number(index)];
      if (!prompt) return false;
      dismissedFollowUps.add(`${String(latestRun && latestRun.run_id || 'current')}:${prompt}`);
      return true;
    }

    function renderReviewSummary(latestRun) {
      const refs = (latestRun.artifact_refs || []).filter(row => row && row.run_id === latestRun.run_id
        && /^[a-f0-9]{64}$/.test(String(row.sha256 || '')));
      const evidence = name => refs.find(row => row.artifact === name);
      const reviewRef = scientificReviewReference(latestRun);
      const expectedKey = reviewRef && typeof projectId === 'function'
        ? `${projectId()}:${latestRun.run_id}:${reviewRef.sha256}` : '';
      const payload = scientificReview.key === expectedKey ? scientificReview.payload : null;
      const row = (title, description, value, resource, passed = tr('Recorded', '有记录'), refsHtml = '') => `<li><div><strong>${esc(title)}</strong>${description ? `<p>${esc(description)}</p>` : ''}</div><span class="gpi-review-state${value === true ? ' is-checked' : ''}">${value === true ? passed : value === false ? tr('Open', '待处理') : tr('Unreported', '未返回')}</span>${resource ? resourceButton(resource, tr('View evidence', '查看依据')) : ''}${refsHtml}</li>`;
      const steps = ANSWER ? ANSWER.approachSteps(latestRun) : [];
      const approachLead = steps.length
        ? `<li class="gpi-review-narrative"><div><strong>${tr('What this run did', '本次运行做了什么')}</strong><ol>${steps.map(step => `<li>${esc(step)}</li>`).join('')}</ol></div></li>` : '';
      const materialsText = ANSWER ? ANSWER.materials(latestRun) : '';
      const materialsLead = materialsText
        ? `<li class="gpi-review-narrative"><div><strong>${tr('Data analysed', '分析数据')}</strong><p>${esc(materialsText)}</p></div></li>` : '';
      const limitationText = ANSWER ? ANSWER.limitation(latestRun) : '';
      const limitationLead = limitationText
        ? `<li class="gpi-review-narrative"><div><strong>${tr('How to read the result', '结果的解读边界')}</strong><p>${esc(limitationText)}</p></div></li>` : '';
      const numericCopy = latestRun.numeric_verified === true
        ? tr('Numbers were checked against the result tables; conclusions still need expert review.', '已核对数字与结果表；研究结论仍需专业审阅。')
        : tr('Number checks are incomplete; check the draft against the result tables.', '数字核对尚未完成，请对照结果表检查稿件。');
      const findings = payload ? payload.findings.filter(value => value && typeof value === 'object'
        && !['passed', 'resolved', 'closed'].includes(value.status)) : [];
      const priority = { blocker: 0, major: 1, minor: 2, info: 3 };
      const first = findings.slice().sort((a, b) => (priority[a.severity] ?? 4) - (priority[b.severity] ?? 4))[0];
      const evidenceRefs = value => {
        const reviewEvidence = Array.isArray(latestRun.review_evidence_refs) ? latestRun.review_evidence_refs : [];
        const purposes = {
          preplan_literature_bundle: tr('Literature screening record', '文献筛选记录'),
          reviewer_report: tr('Scientific review', '科学审阅'), reviewer_report_post_readiness: tr('Scientific review', '科学审阅'),
          scientific_maturity_audit: tr('Open issues', '待处理问题'), critique_report: tr('Step check', '步骤检查'),
          run_status: tr('Run status', '运行状态'), robustness_panel: tr('Robustness checks', '稳健性分析记录'),
          novelty_positioning_audit: tr('Comparison with prior studies', '既往研究对照记录'),
        };
        const chips = (Array.isArray(value.evidence_refs) ? value.evidence_refs : []).map(rawRef => {
          const reference = String(rawRef || '').trim();
          if (!reference) return '';
          const label = purposes[reference.split(/\.(?:json|md)(?:\.|$)/)[0]] || tr('Supporting record', '关联记录');
          const matches = reviewEvidence.filter(ref => ref && ref.reference === reference);
          const registryRef = matches.length === 1
            && /^[A-Za-z0-9_.-]{1,160}$/.test(String(matches[0].evidence_id || ''))
            && /^[a-f0-9]{64}$/.test(String(matches[0].sha256 || '')) ? matches[0] : null;
          if (registryRef && reviewRef) return resourceButton({ ...reviewRef, label,
            evidence_id: registryRef.evidence_id, evidence_sha256: registryRef.sha256,
            evidence_kind: registryRef.kind, evidence_label: label, evidence_pointer: registryRef.pointer || '',
          }, label);
          const registered = refs.filter(ref => reference === ref.artifact || reference.startsWith(`${ref.artifact}.`));
          return matches.length <= 1 && registered.length === 1 ? resourceButton({ ...registered[0], label }, label)
            : `<span class="gpi-review-evidence-missing">${esc(label + tr(' (unavailable)', '（暂不可查看）'))}</span>`;
        }).filter(Boolean).join('');
        return chips ? `<div class="gpi-review-evidence">${chips}</div>` : '';
      };
      const issue = value => {
        const copy = findingCopy(value);
        const authorization = value.requires_user_authorization === true
          ? `<p>${tr('Your confirmation is required before changing the study plan.', '需要你确认后才能修改研究方案。')}</p>` : '';
        return `<li class="gpi-review-record" data-review-code="${esc(String(value.code || ''))}"><div><strong>${esc(copy.title)}</strong><p class="gpi-review-remedy">${tr('Next step: ', '下一步：')}${esc(copy.action)}</p>${authorization}${evidenceRefs(value)}</div><span class="gpi-review-state">${tr('Open', '待处理')}</span></li>`;
      };
      const domainNames = { analysis: tr('Analysis review', '分析审阅'), data: tr('Data review', '数据审阅'),
        literature: tr('Literature review', '文献审阅'), manuscript: tr('Manuscript review', '稿件审阅'), idea: tr('Research idea', '研究问题') };
      const forDomains = names => (payload ? payload.domains : []).filter(value => value && names.includes(value.domain))
        .map(value => row(domainNames[value.domain], '', value.status === 'passed' ? true
          : ['blocked', 'warning'].includes(value.status) ? false : null, null, tr('Recorded', '有记录'), evidenceRefs(value))).join('')
        + findings.filter(value => names.includes(String(value.domain || ''))).map(issue).join('');
      const recorded = (name, exclude) => (REVIEW_ITEMS ? REVIEW_ITEMS.panelHtml(latestRun, name, exclude) : '');
      // Readiness findings already appear as issues above; the itemised review
      // repeats only the plan review's own findings.
      const shownFinding = item => item.code === 'SR_FINDING' && item.source && item.source.artifact === 'scientific_readiness.json' && payload;
      const panels = [
        [tr('Approach', '方法'), (recorded('approach') || approachLead)
          + row(tr('Execution', '分析执行'), '', latestRun.execution_complete, evidence('quality_gate.json'), tr('Complete', '已完成'))
          + row(tr('Validation', '分析校验'), '', latestRun.analysis_validated, evidence('quality_report.json')) + forDomains(['analysis'])],
        [tr('Materials', '材料'), (recorded('materials') || materialsLead)
          + row(tr('Evidence record', '证据记录'), '', latestRun.evidence_complete, evidence('evidence_ledger.json')) + forDomains(['data', 'literature'])],
        [tr('Evidence checks', '证据核对'), row(tr('Numeric provenance', '数字溯源'), numericCopy, latestRun.numeric_verified, evidence('manuscript_provenance.json')) + recorded('numbers') + forDomains(['manuscript'])],
        [tr('Limitations', '局限'), limitationLead + recorded('limitations', shownFinding) + forDomains(['idea'])
          + findings.filter(value => !['analysis', 'data', 'literature', 'manuscript', 'idea'].includes(value.domain)).map(issue).join('')],
      ];
      const original = payload ? `<details class="gpi-review-original"><summary>${tr('Original review', '原始审阅意见')}</summary><pre>${esc(JSON.stringify(payload, null, 2))}</pre></details>` : '';
      return `<section class="gpi-review-summary" data-gpi-review-run="${esc(latestRun.run_id)}" aria-label="${tr('Scientific review', '科学审阅')}">
        ${first ? `<ol class="gpi-review-next-step">${issue(first)}</ol>` : ''}
        <details class="gpi-response-review"><summary>${tr('Scientific review', '科学审阅')}</summary>
        <div class="gpi-review-tabs" role="tablist" aria-label="${tr('Review dimensions', '审阅维度')}">${panels.map(([title], index) => `<button type="button" role="tab" data-gpi-review-tab="${index}" aria-selected="${index === activeReviewTab}">${esc(title)}</button>`).join('')}</div>
        ${panels.map(([, content], index) => `<div class="gpi-review-panel" role="tabpanel" data-gpi-review-panel="${index}"${index === activeReviewTab ? '' : ' hidden'}><ol>${content}</ol></div>`).join('')}
        ${original}</details>
      </section>`;
    }

    function render(latestRun, workflow) {
      if (!resultsAvailable(latestRun, workflow)) return '';
      const validated = latestRun.analysis_validated === true;
      const numericVerified = latestRun.numeric_verified === true;
      const manuscriptReady = latestRun.manuscript_ready === true || latestRun.report_revision_ready === true;
      const figureCount = Number.isInteger(latestRun.figure_count) ? latestRun.figure_count : null;
      const resources = Array.isArray(latestRun.artifact_refs) ? latestRun.artifact_refs : [];
      const ledger = resources.find(row => row && row.run_id === latestRun.run_id
        && row.artifact === 'evidence_ledger.json');
      const reportResource = name => {
        return ledger ? {
          kind: 'research_report', run_id: latestRun.run_id, artifact: name,
          sha256: ledger.sha256, media_type: 'application/json',
        } : null;
      };
      const analysisReport = reportResource('full_analysis_report.json');
      const articleReport = reportResource('article_report.json');
      const technicalReport = reportResource('technical_report.json');
      const detailActions = [];
      if (latestRun.run_id && ledger) {
        if (analysisReport) detailActions.push(resourceButton(
          analysisReport, tr('View complete analysis report', '查看完整分析报告'),
        ));
        if (manuscriptReady && articleReport) detailActions.push(resourceButton(
          articleReport, tr('View article report with figures', '查看含图文章报告'),
        ));
        if (technicalReport) detailActions.push(resourceButton(
          technicalReport, tr('View technical analysis report', '查看技术分析报告'),
        ));
      }
      if (latestRun.report_revision_pdf_ready === true) {
        const pdf = resources.find(row => row && row.artifact === 'manuscript_revision.pdf');
        if (pdf) detailActions.push(resourceButton(pdf, tr('View current report PDF', '查看当前报告 PDF')));
      }
      // Never present the historical source PDF as the current revision.
      if (latestRun.manuscript_ready === true && latestRun.report_revision_ready !== true) {
        const manuscriptPdf = resources.find(row => row && row.artifact === 'manuscript_scaffold.pdf');
        if (manuscriptPdf) {
          detailActions.push(resourceButton(
            { ...manuscriptPdf, label: tr('View LaTeX manuscript PDF', '查看 LaTeX 论文') },
            tr('View LaTeX manuscript PDF', '查看 LaTeX 论文'),
          ));
        }
      }
      if (ledger) {
        detailActions.push(resourceButton(
          { ...ledger, label: tr('View evidence ledger', '查看证据台账') },
          tr('View evidence ledger', '查看证据台账'),
        ));
      }
      const primaryActions = [
        latestRun.run_id && ledger && analysisReport
          ? resourceButton(analysisReport, tr('Open research results', '打开研究成果')) : '',
        `<button class="btn sm" type="button" data-gpi-run-outcome-data>${iconHtml('viz', 13)} ${esc(tr('Open data visualization', '打开数据可视化'))}</button>`,
      ];
      detailActions.push(...Object.keys(labels).map(name => {
          if (!manuscriptReady && name === 'manuscript_provenance.json') return '';
          if (name === 'figure_gallery.json' && latestRun.figure_count === 0) return '';
          const resource = resources.find(row => row && row.artifact === name);
          const label = tr(labels[name][0], labels[name][1]);
          return resource ? resourceButton({ ...resource, label }, label) : '';
        }).filter(Boolean));
      const retryAvailable = Boolean(
        workflow && workflow.analysis_validation_retry_available === true
      );
      if (retryAvailable && manuscriptReady && validated && numericVerified) {
        primaryActions.push(`<button class="btn sm" type="button" data-gpi-run-outcome-retry="report_only">${iconHtml('refresh', 13)} ${esc(tr('Recheck report without rerunning analysis', '重新校验报告（不重跑分析）'))}</button>`);
      }
      if (retryAvailable && (!validated || !manuscriptReady)) {
        const retryLabel = validated && numericVerified
          ? tr('Restore manuscript and evidence checks', '恢复稿件与证据校验')
          : tr('Repair and revalidate', '修复并重新校验');
        primaryActions.unshift(`<button class="btn sm primary" type="button" data-gpi-run-outcome-retry="${validated && numericVerified ? 'restore' : 'validation_repair'}">${iconHtml('refresh', 13)} ${esc(retryLabel)}</button>`);
      }
      if (!primaryActions.length && !detailActions.length) return '';
      const manuscript = latestRun.manuscript_outcome;
      const manuscriptNotice = validated && manuscript && manuscript.status === 'not_passed' && !manuscriptReady
        ? `<p data-manuscript-outcome="not_passed" data-manuscript-run="${esc(latestRun.run_id)}">${esc(manuscriptNotPassedText(manuscript.reason_code))}</p>` : '';
      const figureNote = figureCount === 0
        ? `<p class="gpi-run-outcome-figure-note">${esc(tr('No figure was registered for this analysis; review the result tables first.', '本次分析未登记图件；请先查看结果表。'))}</p>`
        : '';
      // The answer comes first, in the run's own numbers; the file table,
      // actions, follow-ups and review follow it. A run whose registered
      // results carry no summarizable answer keeps one short sentence.
      const answer = ANSWER ? ANSWER.render(latestRun) : '';
      const lead = answer || `<p>${esc(tr(
        validated
          ? 'Analysis complete. A result summary is unavailable; open the result table.'
          : 'The generated results are available below. Review the check records before interpreting them.',
        validated
          ? '分析已完成，暂未生成结果摘要。请查看结果表。'
          : '下方可以查看已生成的结果。解读前，请先核对各项检查记录。',
      ))}</p>`;
      const status = validated
        ? (numericVerified ? tr('Results for review · manuscript review incomplete', '结果待审阅 · 论文尚未完成审阅')
          : tr('Analysis review only · numeric provenance open', '仅供分析审阅 · 数字溯源未闭合'))
        : tr('Validation needs review', '校验待核对');
      return `<section class="gpi-run-outcome" data-gpi-run-outcome="${esc(latestRun.run_id || '')}" aria-label="${esc(tr('Completed analysis results', '已完成的分析结果'))}">
        <div class="gpi-run-outcome-icon" aria-hidden="true">${iconHtml(validated ? 'check' : 'shield', 17)}</div>
        <div class="gpi-run-outcome-copy">
          <strong>${esc(validated
            ? tr('Analysis complete', '分析已完成')
            : tr('Results generated — validation needs review', '结果已生成，请核对待处理事项'))}<span class="gpi-run-outcome-status">${esc(status)}</span></strong>
          ${lead}
          ${APPEND ? APPEND.render(latestRun) : ''}
          ${manuscriptNotice}
          ${renderDeliverables(latestRun, workflow)}
          ${figureNote}
          <div class="gpi-run-outcome-actions gpi-run-outcome-primary">${primaryActions.join('')}</div>
          ${processNotes(latestRun)}
          ${answerActions(latestRun, workflow)}
          ${renderReviewSummary(latestRun)}
          ${renderFollowUps(latestRun, workflow)}
        </div>
      </section>`;
    }

    async function openData(button) {
      if (!canPreview()) return;
      const client = api();
      if (!client.preparePiCopilotDataWorkbenchSnapshot || !preview()) {
        onError(tr('The source-data snapshot is temporarily unavailable. Refresh this project and try again.', '源数据快照暂时不可用，请刷新当前项目后重试。'));
        return;
      }
      const original = button ? button.innerHTML : '';
      if (button) {
        button.disabled = true;
        button.textContent = tr('Preparing source snapshot…', '正在准备源数据快照…');
      }
      const expectedProjectId = projectId();
      try {
        const payload = await client.preparePiCopilotDataWorkbenchSnapshot(expectedProjectId);
        if (projectId() !== expectedProjectId) return;
        const resource = payload && payload.resource;
        if (!resource) throw new Error(tr('EasyICU did not return a Data Workbench snapshot.', 'EasyICU 未返回数据工作台快照。'));
        resource.label = tr('EasyICU data visualization', 'EasyICU 数据可视化');
        preview().open(resource, expectedProjectId, workflowContext());
        const context = workflowContext();
        await recordHostAction(
          'review_prepared_data',
          `${String((context && context.currentRunId) || expectedProjectId)}:${String(resource.snapshot_sha256 || '')}`,
        );
      } catch (error) {
        if (projectId() !== expectedProjectId) return;
        onError(`${tr('Could not open the data visualization for this run', '无法打开本次运行的数据可视化')}: ${errorText(error)}`);
      } finally {
        if (button && button.isConnected) {
          button.disabled = false;
          button.innerHTML = original;
        }
      }
    }

    return Object.freeze({ render, renderShelf, renderReviewAction, collection, resultsAvailable,
      followUps, dismissFollowUp, selectReviewTab, loadScientificReview, openData,
      proposeAppend: kind => APPEND?.propose(kind) });
  }

  window.EasyICU.guidedPi.declare('runOutcome', { create });
})();
