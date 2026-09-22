/* Owner: Guided Pi run-outcome card widget. */
/* Guided Copilot durable completed-run card owner.
   A validated analysis remains visible after refresh even when publication
   review stays closed. It renders only host-projected artifact references. */
(function () {
  'use strict';

  function create(deps) {
    const {
      tr, esc, iconHtml, resourceButton, api, projectId, host, canPreview,
      preview, workflowContext, errorText, recordHostAction, onError,
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
    let scientificReview = { key: '', payload: null, loading: false, error: '' };
    const reviewCopyZh = Object.freeze({
      'Technical executability is not evidence of novelty or publication value.': '技术上可以执行，不等于已经证明研究具有创新性或投稿价值。',
      'A dated search returned sources and the plan mapping is complete.': '已完成带日期的文献检索，并记录了文献与研究计划的对应关系。',
      'The cohort definition and export authority are explicit.': '队列定义和数据导出权限已有明确记录。',
      'Execution completeness does not close the open scientific design defects.': '分析执行完成，不代表尚未解决的科学设计问题已经关闭。',
      'Draft generation is not equivalent to publication readiness.': '生成稿件不等于已经达到投稿条件。',
      'No accepted, digest-bound prior-art review proves that this question is a reliable or sufficiently differentiated research idea.': '尚无已接受且绑定版本的既往研究审阅，能够证明该问题可靠并具有足够差异化。',
      'Run Idea Mining prior-art retrieval, inspect same-topic hits, and accept the refreshed handoff before treating the idea as publishable.': '先完成研究想法的既往文献检索，核对同主题研究，并接受更新后的交接记录，再判断该问题能否用于投稿。',
      'The persisted independent reviewer package still contains an open major-revision or reject-level scientific finding.': '独立审阅记录中仍有尚未关闭的重大修订或拒稿级科学问题。',
      'Resolve or explicitly adjudicate every major finding and regenerate the reviewer receipt before human publication review.': '逐项解决或明确裁定所有重大问题，并重新生成审阅记录后，再进入人工投稿审阅。',
      'The run has no independent, source-bound comparison showing how its population/setting, exposure/time zero, outcome/estimand, analysis/robustness, data-source transportability, and clinical or methodological contribution differ from retained comparison sources.': '当前运行缺少独立且绑定来源的对照，尚未说明研究人群与场景、暴露与时间零点、结局与估计目标、分析与稳健性、数据来源可迁移性，以及临床或方法学贡献与保留文献有何差异。',
      'Create a comparator matrix from retained source excerpts, record substantive differences on all six dimensions, and obtain independent review. A new database/concept instantiation alone is not a novelty claim.': '依据保留的原文证据建立对照矩阵，记录六个维度的实质差异并完成独立审阅。仅更换数据库或实现同一概念，不能单独构成创新性。',
      'The reporting checklist contains unresolved items.': '报告规范检查表中仍有未解决项目。',
      'Address each item or record an evidence-backed not-applicable decision before calling the draft submission-ready.': '逐项处理，或记录有证据支持的“不适用”判断后，才能将稿件标记为可投稿。',
      'An evidence-bound draft exists at most; publication authority has not been granted by the Research Agent gates.': '当前最多形成了证据绑定草稿；研究闸门尚未授予投稿使用权限。',
      'Close scientific, display, reporting, provenance, and human-review gates on one exact run authority before external use.': '在同一精确运行版本上关闭科学、展示、报告、溯源和人工审阅闸门后，方可对外使用。',
      'Prespecified model is incomplete.': '预先设定的模型尚未完成。',
      'A required model is missing.': '缺少一项必需模型。',
      'Complete the model.': '完成该模型并重新审阅。',
    });
    const reviewCodeZh = Object.freeze({
      IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED: '既往研究依据尚未建立',
      SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN: '重大科学修订尚未关闭',
      NOVELTY_POSITIONING_NOT_ESTABLISHED: '创新性定位尚未建立',
      REPORTING_CHECKLIST_ITEMS_OPEN: '报告规范项目尚未关闭',
      PAPER_AUTHORITY_NOT_GRANTED: '尚未取得投稿使用权限',
      MODEL_INCOMPLETE: '预设模型尚未完成',
    });
    const reviewCodeEn = Object.freeze({
      IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED: 'Prior-art evidence is not established',
      SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN: 'Major scientific revision remains open',
      NOVELTY_POSITIONING_NOT_ESTABLISHED: 'Novelty positioning is not established',
      REPORTING_CHECKLIST_ITEMS_OPEN: 'Reporting checklist items remain open',
      PAPER_AUTHORITY_NOT_GRANTED: 'Publication use is not authorized',
      MODEL_INCOMPLETE: 'Prespecified model is incomplete',
    });
    function localizedReviewText(value) {
      const text = String(value || '');
      return reviewCopyZh[text] ? tr(text, reviewCopyZh[text]) : text;
    }
    function findingTitle(code, domainLabel) {
      const value = String(code || '');
      if (reviewCodeZh[value]) return tr(reviewCodeEn[value], reviewCodeZh[value]);
      return value ? tr('Scientific review finding', `${domainLabel || '科学审阅'}待处理项`) : tr('Review finding', '科学审阅待处理项');
    }

    function scientificReviewReference(latestRun) {
      return (Array.isArray(latestRun && latestRun.artifact_refs) ? latestRun.artifact_refs : [])
        .find(row => row && row.run_id === latestRun.run_id
          && row.artifact === 'scientific_readiness.json'
          && /^[a-f0-9]{64}$/.test(String(row.sha256 || '')));
    }

    async function loadScientificReview(latestRun, workflow) {
      if (!resultsAvailable(latestRun, workflow) || typeof projectId !== 'function') return;
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
      const root = typeof host === 'function' ? host() : null;
      const current = root && root.querySelector('.gpi-review-summary');
      if (current && current.dataset.gpiReviewRun === ref.run_id) {
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
      const filtered = files.filter(row => String(row.artifact).toLocaleLowerCase().includes(needle));
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
      return `<section class="gpi-study-results" aria-label="${esc(tr('Current study results', '当前研究成果'))}">
        <div class="gpi-study-results-heading"><span>${tr('Current run', '本次运行')} · ${files.length} ${tr('files', '个文件')}</span></div>
        <div class="gpi-study-results-primary">${rows.slice(0, 3).map(row => resourceButton(row, row.label)).join('')}</div>
        <div class="gpi-results-toolbar" role="group" aria-label="${tr('Result file display', '成果文件显示方式')}">
          <button type="button" data-gpi-results-view="list" aria-pressed="${view === 'list'}" title="${tr('Compact list', '简洁列表')}">${iconHtml('rows', 14)}<span>${tr('List', '列表')}</span></button>
          <button type="button" data-gpi-results-view="details" aria-pressed="${view === 'details'}" title="${tr('File details', '文件详情')}">${iconHtml('layers', 14)}<span>${tr('Details', '详情')}</span></button>
          <label><span class="shell-sr-only">${tr('Sort files', '文件排序')}</span><select data-gpi-results-sort><option value="recommended"${sort === 'recommended' ? ' selected' : ''}>${tr('Recommended', '推荐')}</option><option value="name"${sort === 'name' ? ' selected' : ''}>${tr('Name', '名称')}</option><option value="size"${sort === 'size' ? ' selected' : ''}>${tr('Size', '大小')}</option></select></label>
        </div>
        <label class="gpi-results-search"><span class="shell-sr-only">${tr('Filter result files', '筛选成果文件')}</span><input type="search" data-gpi-results-search placeholder="${tr('Filter files…', '搜索文件…')}" value="${esc(query)}"></label>
        <div class="gpi-study-results-links" data-gpi-results-list data-view="${view}">${files.map(row => `<div class="gpi-study-file" data-gpi-result-file="${esc(row.artifact)}"${needle && !String(row.artifact).toLocaleLowerCase().includes(needle) ? ' hidden' : ''}><span aria-hidden="true">${iconHtml(resourceIcon(row), 15)}</span>${resourceButton(row, row.artifact)}${size(row.size) ? `<small>${esc(size(row.size))}</small>` : ''}${download(row)}</div>`).join('')}</div>
        <p class="gpi-aside-empty" data-gpi-results-empty${filtered.length ? ' hidden' : ''}>${tr('No matching files', '没有匹配的文件')}</p>
        <small>${esc(tr('Files from the current run · publication review pending', '当前运行文件 · 投稿审阅尚未完成'))}</small>
        <details class="gpi-study-results-source"><summary>${tr('Result source', '成果来源')}</summary><code>${esc(latestRun.run_id)}</code></details>
      </section>`;
    }

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
      return rows.length ? `<table class="gpi-deliverables"><caption>${tr('Results from this run', '本次研究产物')}</caption><thead><tr><th>${tr('Result', '成果')}</th><th>${tr('Contents', '内容')}</th></tr></thead><tbody>${rows.map(row => `<tr><td>${resourceButton(row, row.label)}</td><td>${esc(descriptions[row.artifact])}</td></tr>`).join('')}</tbody></table>` : '';
    }

    function renderReviewAction(latestRun, workflow) {
      const stage = (workflow && workflow.stages || []).find(row => row.id === workflow.current_stage);
      if (!stage || !['ready', 'review_required'].includes(stage.status)) return '';
      const name = stage.id === 'interpretation' ? 'full_analysis_report.json'
        : stage.id === 'manuscript' ? 'article_report.json' : '';
      const resource = collection(latestRun, workflow).find(row => row.artifact === name);
      return resource ? resourceButton(resource, stage.id === 'interpretation'
        ? tr('Review results', '审阅研究结果') : tr('Review manuscript', '审阅稿件')) : '';
    }

    function followUps(latestRun, workflow) {
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
      return `<details class="gpi-followups" open><summary class="gpi-followups-head">${iconHtml('help', 14)}<span>${tr('Follow-up questions', '继续追问')}</span></summary><div class="gpi-followups-list">${visible.map(({ prompt, index }) => `<div class="gpi-followup-row"><button type="button" class="gpi-followup-prompt" data-gpi-followup="${index}"><span aria-hidden="true">↳</span>${esc(prompt)}</button><div class="gpi-followup-actions"><button type="button" class="gpi-icon-action" data-gpi-followup-new="${index}" title="${esc(tr('Ask in a new conversation', '在新对话中追问'))}" aria-label="${esc(tr('Ask in a new conversation', '在新对话中追问'))}">${iconHtml('arrow', 13)}</button><button type="button" class="gpi-icon-action" data-gpi-followup-dismiss="${index}" title="${esc(tr('Dismiss suggestion', '关闭建议'))}" aria-label="${esc(tr('Dismiss suggestion', '关闭建议'))}">${iconHtml('close', 13)}</button></div></div>`).join('')}</div></details>`;
    }

    function dismissFollowUp(index, latestRun, workflow) {
      const suggestions = followUps(latestRun, workflow);
      const prompt = suggestions[Number(index)];
      if (!prompt) return false;
      dismissedFollowUps.add(`${String(latestRun && latestRun.run_id || 'current')}:${prompt}`);
      return true;
    }

    function renderReviewSummary(latestRun) {
      const refs = (latestRun.artifact_refs || []).filter(row => row && row.run_id === latestRun.run_id);
      const evidence = name => refs.find(row => row.artifact === name);
      const row = (title, description, value, resource) => `<li><div><strong>${esc(title)}</strong><p>${esc(description)}</p></div><span class="gpi-review-state${value === true ? ' is-checked' : ''}">${value === true ? tr('Recorded', '有记录') : value === false ? tr('Open', '待处理') : tr('Unreported', '未返回')}</span>${resource ? resourceButton(resource, tr('View evidence', '查看依据')) : ''}</li>`;
      let panels = [
        [tr('Approach', '方法'), row(tr('Execution', '分析执行'), tr('Whether the approved analysis finished according to the run gate.', '依据运行闸门核对已批准分析是否完成。'), latestRun.execution_complete, evidence('quality_gate.json') || evidence('evidence_ledger.json'))
          + row(tr('Validation', '分析校验'), tr('Inspect validation findings before interpreting estimates.', '解释估计前先查看分析校验发现。'), latestRun.analysis_validated, evidence('quality_report.json') || evidence('quality_gate.json'))],
        [tr('Materials', '材料'), row(tr('Evidence record', '证据记录'), tr('Source and artifact registration can be inspected; this does not itself establish data suitability.', '可查看来源与产物登记；登记本身不等于数据适用性已获确认。'), latestRun.evidence_complete, evidence('evidence_ledger.json') || evidence('source_run_manifest.json'))],
        [tr('Evidence checks', '证据核对'), row(tr('Numeric provenance', '数字溯源'), tr('This gate compares manuscript numbers with run evidence. It is not an independent hallucination audit.', '此闸门核对稿件数字与运行证据，不等同于独立的幻觉审查。'), latestRun.numeric_verified, evidence('manuscript_provenance.json') || evidence('evidence_ledger.json'))],
        [tr('Limitations', '局限'), row(tr('Scientific and human review', '科学与人工审阅'), tr('Read unresolved scientific issues before any publication claim.', '形成投稿结论前应查看未解决的科学问题。'), latestRun.reportable === true, evidence('scientific_readiness.json'))],
      ];
      const reviewRef = evidence('scientific_readiness.json');
      const expectedKey = reviewRef && typeof projectId === 'function'
        ? `${projectId()}:${latestRun.run_id}:${reviewRef.sha256}` : '';
      const payload = scientificReview.key === expectedKey ? scientificReview.payload : null;
      if (payload) {
        const domainNames = { idea: tr('Research idea', '研究问题'), literature: tr('Literature', '文献'),
          data: tr('Data', '数据'), analysis: tr('Analysis', '分析'), manuscript: tr('Manuscript', '稿件') };
        const stateNames = { passed: tr('Passed', '通过'), blocked: tr('Blocked', '阻断'),
          not_assessed: tr('Not assessed', '未评估'), warning: tr('Attention', '需关注') };
        const safe = (value, limit = 700) => esc(String(value || '').slice(0, limit));
        const item = (title, summary, status, refsHtml, remedy, rawCode) => `<li class="gpi-review-record"${rawCode ? ` data-review-code="${safe(rawCode, 120)}"` : ''}><div><strong>${safe(title, 180)}</strong><p>${safe(localizedReviewText(summary))}</p>${remedy ? `<p class="gpi-review-remedy">${tr('Next step: ', '待处理：')}${safe(localizedReviewText(remedy))}</p>` : ''}${refsHtml || ''}</div><span class="gpi-review-state${status === 'passed' ? ' is-checked' : ''}">${safe(stateNames[status] || status, 40)}</span></li>`;
        const domains = payload.domains.slice(0, 12).filter(value => value && typeof value === 'object');
        const findings = payload.findings.slice(0, 16).filter(value => value && typeof value === 'object');
        const reviewEvidence = Array.isArray(latestRun.review_evidence_refs)
          ? latestRun.review_evidence_refs.filter(value => value && typeof value === 'object') : [];
        const evidenceHost = reviewRef || evidence('evidence_ledger.json');
        const evidenceRefs = value => {
          if (!Array.isArray(value.evidence_refs) || !value.evidence_refs.length) return '';
          const registered = refs.slice().sort((left, right) => String(right.artifact || '').length - String(left.artifact || '').length);
          const chips = value.evidence_refs.slice(0, 3).map(rawRef => {
            const label = String(rawRef || '').trim().slice(0, 120);
            if (!label) return '';
            const registryRef = reviewEvidence.find(ref => String(ref.reference || '') === label);
            if (registryRef && evidenceHost) {
              return resourceButton({
                ...evidenceHost,
                label,
                evidence_id: registryRef.evidence_id,
                evidence_sha256: registryRef.sha256,
                evidence_kind: registryRef.kind,
                evidence_label: registryRef.description || label,
                evidence_pointer: registryRef.pointer || '',
              }, label);
            }
            const matched = registered.find(ref => label === ref.artifact || label.startsWith(`${ref.artifact}.`));
            return matched
              ? resourceButton({ ...matched, label }, label)
              : `<span class="gpi-review-evidence-missing" title="${safe(tr('This reference is recorded but no governed preview is registered.', '该引用已记录，但尚未登记可受控预览。'), 180)}">${safe(label, 120)}</span>`;
          }).filter(Boolean).join('');
          return chips ? `<div class="gpi-review-evidence"><small>${tr('Evidence refs', '证据引用')}</small>${chips}</div>` : '';
        };
        const forDomains = names => {
          const found = domains.filter(value => names.includes(String(value.domain || '')));
          const issues = findings.filter(value => names.includes(String(value.domain || '')));
          const content = found.map(value => item(domainNames[value.domain] || value.domain,
            value.summary, value.status, evidenceRefs(value))).join('')
            + issues.map(value => item(findingTitle(value.code, domainNames[value.domain] || value.domain),
              value.message, value.severity === 'blocker' ? 'blocked' : 'warning',
              evidenceRefs(value), value.remediation, value.code)).join('');
          return content || `<li><small>${tr('No item was recorded in this area.', '本项没有返回记录。')}</small></li>`;
        };
        panels = [
          [tr('Approach', '方法'), forDomains(['analysis'])],
          [tr('Materials', '材料'), forDomains(['data', 'literature'])],
          [tr('Evidence checks', '证据核对'), forDomains(['manuscript'])
            + row(tr('Numeric provenance', '数字溯源'), tr('A numeric gate is separate from an independent hallucination audit.', '数字核验闸门不等于独立的幻觉审查。'), latestRun.numeric_verified, evidence('manuscript_provenance.json'))],
          [tr('Limitations', '局限'), forDomains(['idea'])],
        ];
      }
      return `<section class="gpi-review-summary" data-gpi-review-run="${esc(latestRun.run_id)}" aria-label="${tr('Scientific review', '科学审阅')}">
        <div class="gpi-review-tabs" role="tablist" aria-label="${tr('Review dimensions', '审阅维度')}">${panels.map(([title], index) => `<button type="button" role="tab" data-gpi-review-tab="${index}" aria-selected="${index === activeReviewTab}">${esc(title)}</button>`).join('')}</div>
        ${panels.map(([, content], index) => `<div class="gpi-review-panel" role="tabpanel" data-gpi-review-panel="${index}"${index === activeReviewTab ? '' : ' hidden'}><ol>${content}</ol></div>`).join('')}
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
      const figureNote = figureCount === 0
        ? `<p class="gpi-run-outcome-figure-note">${esc(tr('No figure was registered for this analysis; review the result tables first.', '本次分析未登记图件；请先查看结果表。'))}</p>`
        : '';
      return `<section class="gpi-run-outcome" aria-label="${esc(tr('Completed analysis results', '已完成的分析结果'))}">
        <div class="gpi-run-outcome-icon" aria-hidden="true">${iconHtml(validated ? 'check' : 'shield', 17)}</div>
        <div class="gpi-run-outcome-copy">
          <strong>${esc(validated
            ? tr('Analysis complete — results are ready for review', '分析已完成，可以审阅结果')
            : tr('Results generated — validation needs review', '结果已生成，请核对待处理事项'))}</strong>
          <p>${esc(tr(
            validated
              ? numericVerified
                ? 'The analysis and numeric checks are complete. Explore the results below; manuscript and publication review remain open.'
                : 'The approved analysis and automated validation completed. Manuscript-level numeric provenance and publication review remain open, so these results are for analysis review only.'
              : 'The generated results are available below. Review the check records before interpreting them.',
            validated
              ? numericVerified
                ? '分析与数值核验已完成。可以从下方浏览成果，稿件与投稿审阅仍待完成。'
                : '已批准的分析与自动校验已经完成。稿件级数字溯源和投稿审阅尚未闭合，因此当前结果仅供分析审阅。'
              : '下方可以查看已生成的结果。解读前，请先核对各项检查记录。',
          ))}</p>
          ${renderDeliverables(latestRun, workflow)}
          ${figureNote}
          <div class="gpi-run-outcome-actions gpi-run-outcome-primary">${primaryActions.join('')}</div>
          ${renderFollowUps(latestRun, workflow)}
          <details class="gpi-response-review" open><summary>${tr('Scientific review', '科学审阅')}</summary>${renderReviewSummary(latestRun)}</details>
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
      followUps, dismissFollowUp, selectReviewTab, loadScientificReview, openData });
  }

  window.EasyICU.guidedPi.declare('runOutcome', { create });
})();
