/* Owner: Guided Pi governed preview widget. */
/* Guided Pi governed-resource preview owner.
   It swaps the study-progress aside for one clicked project file, webpage, or
   path-free Research Agent artifact reference.
   D-P3-6: sha256 pins here use /^[a-f0-9]{64}$/ — keep in sync with the
   backend Sha256Text in src/easyicu/webserver/routes/pi_copilot.py (same
   pattern, no shared constant across languages; change both together). */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  const state = {
    host: null,
    projectId: '',
    resource: null,
    artifact: null,
    payload: null,
    studyContext: null,
    governance: null,
    mode: 'code',
    loading: false,
    error: '',
    request: 0,
    recentResources: [],
    evidenceTabs: [],
    activeEvidenceId: '',
    activeClaimId: '',
    workflowContext: {},
    focused: false,
    studyResources: [],
    studyProjectId: '',
    openStudyResource: null,
    studyTitle: '',
    referenceResource: null,
    previousAsideCollapsed: null,
  };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function icon(name, size) {
    return typeof window.icon === 'function' ? window.icon(name, size || 16, 1.55) : '';
  }
  function safeResource(value) {
    if (!value || typeof value !== 'object') return null;
    if (value.kind === 'demo_artifact') {
      const demo = window.EasyICU.guidedPi.optional('demo');
      const artifact = String(value.artifact || '').trim();
      if (!/^[A-Za-z0-9_.-]+\.json$/.test(artifact) || artifact.length > 160) return null;
      if (!demo || typeof demo.hasArtifact !== 'function' || !demo.hasArtifact(artifact)) return null;
      return {
        kind: 'demo_artifact', artifact,
        run_id: String(value.run_id || demo.sourceRunId || '').slice(0, 160),
        label: String(value.label || (demo.artifactLabel && demo.artifactLabel(artifact)) || artifact).slice(0, 160),
        media_type: 'application/json', authority_class: 'product_demo_projection',
      };
    }
    if (value.kind === 'demo_document') {
      const artifact = String(value.artifact || '').trim();
      if (!/^system-validation-report\.(html|pdf)$/.test(artifact)) return null;
      return {
        kind: 'demo_document', artifact,
        run_id: String(value.run_id || '').slice(0, 160),
        label: String(value.label || artifact).slice(0, 160),
        media_type: artifact.endsWith('.pdf') ? 'application/pdf' : 'text/html',
        authority_class: 'engineering_validation_only',
      };
    }
    if (value.kind === 'literature_source') {
      const literature = window.EasyICU.guidedPi.optional('literature');
      const authorityClass = value.authority_class === 'literature_method'
        ? 'literature_method' : 'literature_retrieval_candidate';
      const url = literature && typeof literature.safeUrl === 'function'
        ? literature.safeUrl(value.url) : '';
      const title = String(value.title || value.label || '').trim().slice(0, 500);
      if (!url || !title) return null;
      const retrievalFit = ['direct_retrieval_fit', 'adjacent_retrieval_fit', 'unclassified']
        .includes(value.retrieval_fit) ? value.retrieval_fit : '';
      return {
        kind: 'literature_source', url, title,
        label: String(value.label || title).slice(0, 160),
        year: String(value.year || '').slice(0, 16),
        venue: String(value.venue || '').slice(0, 240),
        relevance: String(value.relevance || '').slice(0, 1200),
        doi: String(value.doi || '').slice(0, 240),
        pmid: String(value.pmid || '').slice(0, 32),
        media_type: 'text/html', authority_class: authorityClass,
        ...(retrievalFit ? {
          retrieval_fit: retrievalFit,
          retrieval_rationale: String(value.retrieval_rationale || '').slice(0, 600),
        } : {}),
      };
    }
    if (value.kind === 'research_report') {
      const runId = String(value.run_id || '').trim();
      const sha256 = String(value.sha256 || '').trim().toLowerCase();
      const requestedArtifact = String(value.artifact || 'technical_report.json').trim();
      const supportedReports = new Set([
        'technical_report.json',
        'full_analysis_report.json',
        'article_report.json',
      ]);
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(runId)) return null;
      if (!supportedReports.has(requestedArtifact)) return null;
      if (!/^[a-f0-9]{64}$/.test(sha256)) return null;
      return {
        kind: 'research_report', run_id: runId, artifact: requestedArtifact,
        sha256,
        label: String(value.label || (requestedArtifact === 'full_analysis_report.json'
          ? tr('Complete analysis report', '完整分析报告')
          : requestedArtifact === 'article_report.json'
            ? tr('Article report with figures', '含图文章报告')
            : tr('Technical analysis report', '技术分析报告'))).slice(0, 160),
        media_type: 'application/json',
      };
    }
    if (value.kind === 'research_artifact') {
      const runId = String(value.run_id || '').trim();
      const artifact = String(value.artifact || '').trim();
      const sha256 = String(value.sha256 || '').trim().toLowerCase();
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(runId)) return null;
      if (!/^[A-Za-z0-9_.-]+\.json$/.test(artifact) || artifact.length > 160) return null;
      return {
        kind: 'research_artifact', run_id: runId, artifact,
        ...(/^[a-f0-9]{64}$/.test(sha256) ? { sha256 } : {}),
        label: String(value.label || artifact).slice(0, 160),
        media_type: 'application/json',
      };
    }
    if (value.kind === 'idea_plan') {
      const runId = String(value.run_id || '').trim();
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(runId)) return null;
      if (String(value.artifact || '') !== 'idea_plan.json') return null;
      return {
        kind: 'idea_plan', run_id: runId, artifact: 'idea_plan.json',
        label: String(value.label || tr('Idea Mining plan preview', 'Idea Mining 方案预览')).slice(0, 160),
        media_type: 'application/json', authority_class: 'idea_mining_planning_only',
      };
    }
    if (value.kind === 'research_document' || value.kind === 'system_validation_document') {
      const runId = String(value.run_id || '').trim();
      const artifact = String(value.artifact || '').trim();
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(runId)) return null;
      const validationDocument = value.kind === 'system_validation_document';
      const sha256 = String(value.sha256 || '').trim().toLowerCase();
      // A missing digest stays allowed because the inspect_manuscript tool
      // projection emits document references without one; previewUrl()
      // refuses to serve such a document unpinned instead of rejecting the
      // reference here.
      if (sha256 && !/^[a-f0-9]{64}$/.test(sha256)) return null;
      if (validationDocument
        ? !/^system_validation_report\.(html|pdf)$/.test(artifact)
        : !/^(manuscript_scaffold\.(pdf|tex|bib)|manuscript_revision\.pdf)$/.test(artifact)) return null;
      return {
        kind: validationDocument ? 'system_validation_document' : 'research_document', run_id: runId, artifact, sha256,
        label: String(value.label || artifact).slice(0, 160),
        media_type: String(value.media_type || (artifact.endsWith('.pdf') ? 'application/pdf' : (artifact.endsWith('.html') ? 'text/html' : 'text/plain'))).slice(0, 120),
      };
    }
    if (value.kind === 'data_package_review') {
      const studyContextId = String(value.study_context_id || '').trim();
      const reviewSha256 = String(value.review_sha256 || '').trim().toLowerCase();
      const studyRevision = Number(value.study_revision);
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(studyContextId)) return null;
      if (!/^[a-f0-9]{64}$/.test(reviewSha256)) return null;
      if (!Number.isInteger(studyRevision) || studyRevision < 0) return null;
      return {
        kind: 'data_package_review', study_context_id: studyContextId,
        study_revision: studyRevision, review_sha256: reviewSha256,
        label: String(value.label || tr('Data package review', '数据包审阅')).slice(0, 160),
        media_type: 'application/json',
      };
    }
    if (value.kind === 'data_workbench_snapshot') {
      const view = String(value.view || '').trim();
      const snapshotSha256 = String(value.snapshot_sha256 || '').trim().toLowerCase();
      if (!['cohort_summary', 'feature_distribution', 'icd_cohort_preview', 'patient_timeline', 'crossdb_comparison'].includes(view)) return null;
      if (!/^[a-f0-9]{64}$/.test(snapshotSha256)) return null;
      return {
        kind: 'data_workbench_snapshot', view, snapshot_sha256: snapshotSha256,
        label: String(value.label || tr('Data Workbench', '数据工作台')).slice(0, 160),
        media_type: 'application/json',
      };
    }
    if (value.kind === 'native_workspace') {
      const route = String(value.route || '').trim();
      const state = String(value.state || '').trim();
      const studyContextId = String(value.study_context_id || '').trim();
      const studyRevision = Number(value.study_revision);
      const jobId = String(value.job_id || '').trim();
      const sourceId = String(value.source_id || '').trim();
      const expectedDatabase = String(value.expected_database || '').trim();
      const entryMode = String(value.entry_mode || '').trim();
      const extractionScope = String(value.extraction_scope || '').trim();
      const supportedDatabases = new Set(['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic']);
      const supportedScopes = new Set(['study_required', 'all_supported', 'reuse_prepared_full']);
      if (route !== 'extraction' || !['setup', 'running', 'review'].includes(state)) return null;
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(studyContextId)) return null;
      if (!Number.isInteger(studyRevision) || studyRevision < 0) return null;
      if (jobId && !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/.test(jobId)) return null;
      if (sourceId && !/^src_[a-f0-9]{12}$/.test(sourceId)) return null;
      if (expectedDatabase && !supportedDatabases.has(expectedDatabase)) return null;
      if (entryMode && entryMode !== 'source_binding') return null;
      if (extractionScope && !supportedScopes.has(extractionScope)) return null;
      return {
        kind: 'native_workspace', route, state,
        study_context_id: studyContextId, study_revision: studyRevision,
        label: entryMode === 'source_binding'
          ? tr('Data source setup', '数据来源设置')
          : String(value.label || tr('Data Extraction', '数据提取')).slice(0, 160),
        media_type: 'application/vnd.easyicu.native-workspace',
        ...(jobId ? { job_id: jobId } : {}),
        ...(sourceId ? { source_id: sourceId.slice(0, 80) } : {}),
        ...(expectedDatabase ? { expected_database: expectedDatabase } : {}),
        ...(entryMode ? { entry_mode: entryMode } : {}),
        ...(extractionScope ? { extraction_scope: extractionScope } : {}),
      };
    }
    const file = String(value.file || '').trim().replace(/\\/g, '/');
    if (!file || file.startsWith('/') || file.includes('\0')) return null;
    if (file.split('/').some(part => !part || part === '.' || part === '..')) return null;
    const checkedSha256 = String(value.checked_sha256 || '').trim().toLowerCase();
    return {
      kind: value.kind === 'webpage' ? 'webpage' : 'file',
      file: file.slice(0, 240),
      label: String(value.label || file.split('/').pop() || file).slice(0, 160),
      media_type: String(value.media_type || 'text/plain').slice(0, 120),
      ...(value.kind === 'webpage' && /^[a-f0-9]{64}$/.test(checkedSha256)
        ? { checked_sha256: checkedSha256 } : {}),
      authority_class: 'workspace_artifact',
      scientific_evidence: false,
      validation_status: 'unvalidated',
      claim_ceiling: 'unsupported',
    };
  }
  function resourceKey(resource) {
    const owner = window.EasyICU.guidedPi.require('resources');
    if (owner && typeof owner.create === 'function') {
      const identity = owner.create({ esc });
      if (identity && typeof identity.key === 'function') return identity.key(resource) + ':' + String(resource.sha256 || resource.checked_sha256 || '');
    }
    return JSON.stringify(resource || {});
  }
  function syncLocation(resource, evidence) {
    if (!window.history || !window.location || window.location.protocol === 'file:') return;
    const url = new URL(window.location.href);
    ['pi_view', 'pi_view_kind', 'pi_view_run', 'pi_view_artifact', 'pi_view_sha',
      'pi_view_evidence', 'pi_view_evidence_sha', 'pi_view_evidence_kind',
      'pi_view_evidence_pointer'].forEach(key => url.searchParams.delete(key));
    if (resource && /^[a-f0-9]{64}$/.test(String(resource.sha256 || ''))
      && ['research_artifact', 'research_report', 'research_document', 'system_validation_document'].includes(resource.kind)) {
      const evidenceId = String(evidence && evidence.evidenceId || '').trim();
      const evidenceSha = String(evidence && evidence.sha256 || '').trim().toLowerCase();
      const hasEvidence = /^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) && /^[a-f0-9]{64}$/.test(evidenceSha);
      url.searchParams.set('pi_view', hasEvidence ? 'evidence' : 'artifact');
      url.searchParams.set('pi_view_kind', resource.kind);
      url.searchParams.set('pi_view_run', resource.run_id);
      url.searchParams.set('pi_view_artifact', resource.artifact);
      if (resource.sha256) url.searchParams.set('pi_view_sha', resource.sha256);
      if (hasEvidence) {
        url.searchParams.set('pi_view_evidence', evidenceId);
        url.searchParams.set('pi_view_evidence_sha', evidenceSha);
        url.searchParams.set('pi_view_evidence_kind', String(evidence.kind || 'artifact').slice(0, 80));
        if (evidence.pointer) url.searchParams.set('pi_view_evidence_pointer', String(evidence.pointer).slice(0, 500));
      }
    }
    window.history.replaceState(window.history.state, '', url.toString());
  }
  function restoreFromLocation(projectId, workflowContext) {
    if (!window.location || window.location.protocol === 'file:') return false;
    const params = new URLSearchParams(window.location.search || '');
    const view = params.get('pi_view');
    if (!['artifact', 'evidence'].includes(view)) return false;
    if (!/^[a-f0-9]{64}$/.test(params.get('pi_view_sha') || '')) return false;
    const resource = safeResource({
      kind: params.get('pi_view_kind') || 'research_artifact',
      run_id: params.get('pi_view_run') || '',
      artifact: params.get('pi_view_artifact') || '',
      sha256: params.get('pi_view_sha') || '',
      label: params.get('pi_view_artifact') || '',
    });
    if (!resource || !String(projectId || '').trim()) return false;
    const alreadyOpen = state.resource && state.projectId === String(projectId || '').trim()
      && resourceKey(state.resource) === resourceKey(resource);
    const opened = alreadyOpen || Boolean(open(resource, projectId, workflowContext));
    if (!opened || view !== 'evidence') return opened;
    const evidenceId = params.get('pi_view_evidence') || '';
    const evidenceSha = (params.get('pi_view_evidence_sha') || '').toLowerCase();
    if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(evidenceSha)) return false;
    void openEvidence({ dataset: {
      evidenceId,
      evidenceSha256: evidenceSha,
      evidenceKind: params.get('pi_view_evidence_kind') || 'artifact',
      evidenceLabel: evidenceId,
      evidencePointer: params.get('pi_view_evidence_pointer') || '',
    }, closest: () => null });
    return true;
  }
  function safeJobContext(value) {
    const job = value && typeof value === 'object' ? value : {};
    return {
      present: Boolean(job.present),
      kind: String(job.kind || '').slice(0, 80),
      status: String(job.status || '').slice(0, 40),
      error_code: String(job.error_code || '').slice(0, 120),
      progress: (Array.isArray(job.progress) ? job.progress : []).slice(-24).map(row => ({
        step: String((row && row.step) || '').slice(0, 80),
        current: Number((row && row.current) || 0),
        total: Number((row && row.total) || 0),
      })),
    };
  }
  function safeWorkflowContext(value) {
    const context = value && typeof value === 'object' ? value : {};
    return {
      nextActionCode: String(context.nextActionCode || '').slice(0, 120),
      currentRunId: String(context.currentRunId || '').slice(0, 160),
      activeJob: safeJobContext(context.activeJob),
      failedJob: safeJobContext(context.failedJob),
    };
  }
  function setWorkflowContext(value) {
    const next = safeWorkflowContext(value);
    if (JSON.stringify(next) === JSON.stringify(state.workflowContext)) return;
    state.workflowContext = next;
    if (state.resource) render();
  }
  function rememberResource(resource) {
    const key = resourceKey(resource);
    state.recentResources = [resource]
      .concat(state.recentResources.filter(item => resourceKey(item) !== key))
      .slice(0, 6);
  }
  function isResearchArtifact() { return !!state.resource && state.resource.kind === 'research_artifact'; }
  function isIdeaPlan() { return !!state.resource && state.resource.kind === 'idea_plan'; }
  function isResearchReport() { return !!state.resource && state.resource.kind === 'research_report'; }
  function researchReportOwner() {
    if (!isResearchReport()) return null;
    if (state.resource.artifact === 'full_analysis_report.json') return window.EasyICU.guidedPi.require('analysisReport');
    if (state.resource.artifact === 'article_report.json') return window.EasyICU.guidedPi.require('articleReport');
    return window.EasyICU.guidedPi.require('technicalReport');
  }
  function isEvidenceBoundResource() { return isResearchArtifact() || isResearchReport(); }
  function isResearchDocument() { return !!state.resource && (state.resource.kind === 'research_document' || state.resource.kind === 'system_validation_document'); }
  function isDemoDocument() { return !!state.resource && state.resource.kind === 'demo_document'; }
  function isDocument() { return isResearchDocument() || isDemoDocument(); }
  function isSystemValidationDocument() { return !!state.resource && state.resource.kind === 'system_validation_document'; }
  function isDemoArtifact() { return !!state.resource && state.resource.kind === 'demo_artifact'; }
  function isDataPackageReview() { return !!state.resource && state.resource.kind === 'data_package_review'; }
  function isDataWorkbenchSnapshot() { return !!state.resource && state.resource.kind === 'data_workbench_snapshot'; }
  function isNativeWorkspace() { return !!state.resource && state.resource.kind === 'native_workspace'; }
  function isStructuredArtifact() { return isResearchArtifact() || isResearchReport() || isIdeaPlan() || isDemoArtifact() || isDataPackageReview() || isDataWorkbenchSnapshot(); }
  function isLiteratureSource() { return !!state.resource && state.resource.kind === 'literature_source'; }
  function isHtml() {
    return !!state.resource && (
      state.resource.kind === 'webpage'
      || state.resource.media_type === 'text/html'
      || /\.html?$/i.test(state.resource.file)
    );
  }
  function previewUrl() {
    const api = window.EU_API || {};
    if (isDemoDocument()) {
      return `/assets/demo/${state.resource.artifact}?v=20260815-reviewer-demo1`;
    }
    if (isResearchDocument()) {
      // The click-time digest pins the served bytes to the projected run
      // ledger row; without a valid sha256 the document is never requested.
      // D-P3-6: /^[a-f0-9]{64}$/ mirrors backend Sha256Text
      // (src/easyicu/webserver/routes/pi_copilot.py); keep both in sync.
      const documentSha256 = String(state.resource.sha256 || '').trim().toLowerCase();
      if (!/^[a-f0-9]{64}$/.test(documentSha256)) return '';
      return api.piCopilotResearchDocumentUrl
        ? api.piCopilotResearchDocumentUrl(state.projectId, state.resource.run_id, state.resource.artifact, documentSha256)
        : '';
    }
    const checkedSha256 = String(state.resource && state.resource.checked_sha256 || '').trim().toLowerCase();
    if (!/^[a-f0-9]{64}$/.test(checkedSha256)) return '';
    return api.piCopilotWorkspacePreviewUrl
      ? api.piCopilotWorkspacePreviewUrl(
          state.projectId,
          state.resource.file,
          checkedSha256,
        )
      : '';
  }
  function setAsideOpen(open) {
    const study = document.getElementById('gdStudyAside');
    const aside = document.getElementById('gdContextAside');
    const main = aside && aside.closest('.gd-main');
    const log = main && main.querySelector('[data-gpi-log]');
    const layoutChanged = main && (main.classList.contains('gpi-preview-open') !== !!open
      || main.classList.contains('gpi-preview-focus') !== (!!open && state.focused));
    const followBottom = layoutChanged && log && log.clientHeight > 0
      && log.scrollHeight - log.scrollTop - log.clientHeight < 32;
    const panels = window.EU_GUIDED_PANELS;
    if (open && panels && panels.setContextAsideCollapsed) {
      if (state.previousAsideCollapsed === null) state.previousAsideCollapsed = panels.isContextAsideCollapsed();
      panels.setContextAsideCollapsed(false, main);
    } else if (!open && state.previousAsideCollapsed !== null && panels) {
      panels.setContextAsideCollapsed(state.previousAsideCollapsed, main);
      state.previousAsideCollapsed = null;
    }
    // CSS swaps the shelf for the preview on ordinary desktops and keeps both
    // visible on wide research workstations. Do not hide the shelf in the DOM.
    if (study) study.hidden = false;
    if (state.host) state.host.hidden = !open;
    if (aside) aside.classList.toggle('gpi-preview-open', !!open);
    if (main) main.classList.toggle('gpi-preview-open', !!open);
    if (main) main.classList.toggle('gpi-preview-focus', !!open && state.focused);
    // Width changes reflow messages. Only follow a reader already at the bottom.
    if (followBottom && log.clientHeight > 0) log.scrollTop = log.scrollHeight;
  }

  function studyResourcesHtml() {
    if (state.projectId !== state.studyProjectId || !state.resource || !state.resource.run_id) return '';
    return state.studyResources.map((resource, index) => resource.run_id !== state.resource.run_id ? '' :
      `<button type="button" data-gpi-study-resource="${index}" aria-current="${resourceKey(resource) === resourceKey(state.resource) ? 'true' : 'false'}">${esc(resource.label)}</button>`).join('');
  }

  function setStudyResources(resources, projectId, opener, context = {}) {
    state.studyProjectId = String(projectId || '');
    state.studyResources = (Array.isArray(resources) ? resources : []).map(safeResource).filter(Boolean);
    state.openStudyResource = typeof opener === 'function' ? opener : null;
    const studyTitle = String(context.title || '').trim();
    // D-P2-1: defensive label projection — a bundle without product-labels.js
    // must still render bounded raw text instead of throwing.
    state.studyTitle = studyTitle
      ? (window.EU_PRODUCT_LABELS?.projectTitle?.(studyTitle, '') ?? String(studyTitle)).slice(0, 200)
      : '';
    state.referenceResource = typeof context.reference === 'function' ? context.reference : null;
    // Workflow polling can update the shelf without replacing the open report.
    const nav = state.host && state.host.querySelector('[data-gpi-study-resources]');
    if (nav) { nav.innerHTML = studyResourcesHtml(); nav.hidden = !nav.innerHTML; }
  }

  function toggleFocus(button) {
    state.focused = !state.focused;
    setAsideOpen(true);
    button.setAttribute('aria-pressed', String(state.focused));
    button.textContent = state.focused ? tr('Show conversation', '边聊边看') : tr('Focus reading', '专注阅读');
  }
  function researchProvenance() {
    const governance = state.governance || {};
    if (isSystemValidationDocument() || governance.authority_class === 'easyicu_system_validation_report') {
      return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('System validation dossier · Engineering evidence only', '系统验证报告 · 仅限工程证据'))}</strong><span>${esc(tr('Not a clinical manuscript; cannot grant scientific or publication authority.', '不是临床论文；不能授予科学或发表权限。'))}</span></div>`;
    }
    if (!state.governance) {
      if (isResearchDocument() && /^[a-f0-9]{64}$/.test(state.resource.sha256 || '')) return `<div class="gpi-preview-provenance is-research" role="note"><strong>${tr('Research document · Version pinned', '研究文档 · 版本已绑定')}</strong><span>${tr('Scientific status remains in the review records.', '科学状态以审阅记录为准。')}</span></div>`;
      return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('EasyICU run artifact · Governance pending', 'EasyICU 运行产物 · 治理状态待确认'))}</strong><span>${esc(tr('Loading Host gate status…', '正在加载 Host 运行闸状态…'))}</span></div>`;
    }
    const ceiling = governance.claim_ceiling;
    const title = ceiling === 'analysis_only'
        ? tr('EasyICU run artifact · Analysis-only', 'EasyICU 运行产物 · 仅供分析')
        : tr('EasyICU run artifact · Governance pending', 'EasyICU 运行产物 · 治理状态待确认');
    const signoff = governance.human_signoff;
    const detail = signoff === 'required'
      ? tr('Human sign-off required', '需要人工签署')
      : signoff === 'signed'
        ? tr('Human sign-off recorded; claim ceiling remains Host-controlled.', '已记录人工签署；结论上限仍由 Host 控制。')
        : signoff === 'stale'
          ? tr('Sign-off is stale; do not use for claims.', '签署已失效；不得用于结论。')
          : tr('Current run gate does not permit sign-off.', '当前运行闸不允许签署。');
    return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(title)}</strong><span>${esc(detail)}</span></div>`;
  }
  function demoProvenance() {
    return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('Bounded reviewer projection · Standard Web renderer', '有界审稿投影 · 标准 Web 渲染器'))}</strong><span>${esc(tr('A read-only projection derived from the registered run and rendered with the live Web artifact views; it is not the live artifact transport or publication evidence.', '这是从登记运行派生并复用真实 Web 产物视图的只读投影；它不是 live artifact transport，也不是投稿证据。'))}</span></div>`;
  }
  function demoDocumentProvenance() {
    return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('Reviewer demonstration complete · Engineering evidence', '审稿人演示完整完成 · 工程证据'))}</strong><span>${esc(tr('The workflow demonstration is complete; clinical manuscript and publication authority remain separate and were not granted.', '流程演示已完整完成；临床稿件与发表权限属于独立边界，本报告未授予这些权限。'))}</span></div>`;
  }
  function dataPackageProvenance() {
    return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('Registered export · Pre-analysis review', '已登记数据源 · 分析前审阅'))}</strong><span>${esc(tr('Aggregate denominator and availability only; event rates, comparisons, and effect estimates are withheld until the governed analysis.', '仅展示聚合分母与可用性；事件率、组间比较和效应量留待受治理分析。'))}</span></div>`;
  }
  function dataWorkbenchProvenance() {
    return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('Conversational Data Workbench · Descriptive review', '对话式数据工作台 · 描述性审阅'))}</strong><span>${esc(tr('The browser opens an immutable local snapshot. Patient timelines remain pseudonymous and browser-only; reportable claims require the governed analysis path.', '浏览器打开不可变的本地快照。患者时间序列保持伪匿名且仅限浏览器；可报告结论仍需经过受治理分析流程。'))}</span></div>`;
  }
  function activeEvidenceTab() {
    return state.evidenceTabs.find(item => item.evidenceId === state.activeEvidenceId) || null;
  }
  function evidenceTabsView() {
    if (!isEvidenceBoundResource() || !state.evidenceTabs.length) return '';
    return state.evidenceTabs.map(item => `<span class="gpi-evidence-tab-wrap"><button type="button" role="tab" data-gpi-evidence-tab="${esc(item.evidenceId)}" aria-selected="${state.mode === 'evidence' && state.activeEvidenceId === item.evidenceId}">${esc(item.kindLabel)} · ${esc(item.label)}</button><button type="button" class="gpi-evidence-tab-close" data-gpi-evidence-tab-close="${esc(item.evidenceId)}" aria-label="${esc(tr('Close evidence tab', '关闭证据标签页'))}">×</button></span>`).join('');
  }
  function render() {
    if (!state.host || !state.resource) return;
    setAsideOpen(true);
    const selectedEvidence = activeEvidenceTab();
    const tabs = isLiteratureSource() ? '' : isStructuredArtifact() ? `
      <div class="gpi-preview-tabs" role="tablist" aria-label="${tr('Artifact views', '产物视图')}">
        ${isDataPackageReview() || isDataWorkbenchSnapshot() ? `<button type="button" role="tab" data-gpi-preview-mode="workbench" aria-selected="${state.mode === 'workbench'}">${icon('grid', 14)} ${tr('Workbench', '数据工作台')}</button>` : ''}
        <button type="button" role="tab" data-gpi-preview-mode="structured" aria-selected="${state.mode === 'structured'}">${icon('list', 14)} ${tr('Readable', '可读视图')}</button>
        <button type="button" role="tab" data-gpi-preview-mode="code" aria-selected="${state.mode === 'code'}">${icon('file', 14)} JSON</button>
        ${evidenceTabsView()}
      </div>` : state.resource.kind === 'webpage' && isHtml() ? `
      <div class="gpi-preview-tabs" role="tablist" aria-label="${tr('Artifact views', '产物视图')}">
        <button type="button" role="tab" data-gpi-preview-mode="code" aria-selected="${state.mode === 'code'}">${icon('file', 14)} ${tr('Code', '代码')}</button>
        <button type="button" role="tab" data-gpi-preview-mode="web" aria-selected="${state.mode === 'web'}">${icon('globe', 14)} ${tr('Web preview', '网页预览')}</button>
      </div>` : '';
    let body = '';
    if (state.loading) {
      body = `<div class="gpi-preview-state"><span class="gpi-preview-spinner"></span>${tr('Loading governed artifact…', '正在加载受治理产物…')}</div>`;
    } else if (state.error) {
      body = `<div class="gpi-preview-state error">${icon('alert', 16)}<strong>${tr('Preview unavailable', '无法预览')}</strong><span>${esc(state.error)}</span></div>`;
    } else if (isLiteratureSource()) {
      const renderer = window.EasyICU.guidedPi.require('literature');
      body = renderer && typeof renderer.renderSource === 'function'
        ? renderer.renderSource(state.resource)
        : `<div class="gpi-preview-state error">${esc(tr('Literature renderer unavailable', '文献渲染器不可用'))}</div>`;
    } else if (state.mode === 'document' && isDocument()) {
      const url = previewUrl();
      if (!url) body = `<div class="gpi-preview-state error">${icon('alert', 16)}<strong>${tr('Preview unavailable', '无法预览')}</strong><span>${tr('The registered document digest is missing, so this preview cannot be pinned to the run ledger.', '登记文档摘要缺失，预览无法钉定到运行台账。')}</span></div>`;
      else if (/\.pdf$/i.test(state.resource.artifact || '')) {
        // Chrome blocks its native PDF plugin in sandboxed frames. Keep the
        // sandbox contract; offer the exact bytes and the matching online article.
        const current = state.projectId === state.studyProjectId
          && state.studyResources.some(row => resourceKey(row) === resourceKey(state.resource));
        const article = current ? state.studyResources.findIndex(row => row.kind === 'research_report'
          && row.run_id === state.resource.run_id && row.artifact === 'article_report.json') : -1;
        body = `<section class="gpi-pdf-access"><span class="gpi-pdf-mark" aria-hidden="true">PDF</span><h2>${tr('Read the formatted PDF', '阅读完整排版的 PDF')}</h2>
          <p>${tr('Embedded PDF reading is unavailable in this preview. Download the file to read it in a PDF viewer.', '当前预览环境不支持内嵌 PDF 阅读，请下载后用 PDF 阅读器打开。')}</p>
          <small>${esc(state.resource.artifact)}</small><div><a class="btn primary" href="${esc(url)}" download="${esc(state.resource.artifact)}">${tr('Download this PDF', '下载此 PDF')}</a>
          ${article >= 0 ? `<button class="btn" type="button" data-gpi-study-resource="${article}">${tr('Read the matching article online', '在线阅读对应文章')}</button>` : ''}</div>
        </section>`;
      } else body = `<iframe class="gpi-preview-frame gpi-preview-document-frame" src="${esc(url)}" sandbox="allow-scripts" referrerpolicy="no-referrer" title="${esc(tr('Preview of ', '预览：') + state.resource.label)}"></iframe>`;
    } else if (state.mode === 'web' && isHtml()) {
      const url = previewUrl();
      body = url
        ? `<iframe class="gpi-preview-frame" src="${esc(url)}" sandbox="allow-scripts" referrerpolicy="no-referrer" title="${esc(tr('Preview of ', '预览：') + state.resource.label)}"></iframe>`
        : `<div class="gpi-preview-state error">${icon('alert', 16)}<strong>${tr('Preview unavailable', '无法预览')}</strong><span>${tr('The checked file digest is missing. Run the static check and prepare the preview again.', '文件检查摘要缺失。请重新执行静态检查并准备预览。')}</span></div>`;
    } else if (state.mode === 'native' && isNativeWorkspace()) {
      body = `<div data-gpi-native-workspace-mount></div>`;
    } else if (state.mode === 'workbench' && (isDataPackageReview() || isDataWorkbenchSnapshot())) {
      body = `<div data-gpi-workbench-mount></div>`;
    } else if (state.mode === 'structured' && isStructuredArtifact()) {
      const renderer = window.AGENT_RENDER;
      const literature = window.EasyICU.guidedPi.require('literature');
      const report = researchReportOwner();
      body = isResearchReport() && report && typeof report.render === 'function'
        ? report.render(state.payload || {})
        : isIdeaPlan() && window.EU_GUIDED_IDEA_PLAN && typeof window.EU_GUIDED_IDEA_PLAN.renderArtifact === 'function'
        ? window.EU_GUIDED_IDEA_PLAN.renderArtifact(state.payload || {}, { tr, esc, icon })
        : state.resource.artifact === 'literature_evidence.json'
        && literature && typeof literature.renderArtifact === 'function'
        ? literature.renderArtifact(state.payload || {}, {
          ...state.workflowContext,
          runId: state.resource.run_id,
        })
        : renderer && typeof renderer.artifactStructuredView === 'function'
          ? renderer.artifactStructuredView(state.resource.artifact, state.payload || {})
        : `<pre class="gpi-preview-code" tabindex="0"><code>${esc(JSON.stringify(state.payload || {}, null, 2))}</code></pre>`;
    } else if (state.mode === 'evidence' && activeEvidenceTab()) {
      const item = activeEvidenceTab();
      const renderer = window.EasyICU.guidedPi.require('evidencePreview');
      body = item.loading
        ? `<div class="gpi-preview-state"><span class="gpi-preview-spinner"></span>${tr('Loading digest-pinned evidence…', '正在加载摘要锁定的证据…')}</div>`
        : item.error
          ? `<div class="gpi-preview-state error">${icon('alert', 16)}<strong>${tr('Evidence preview unavailable', '证据预览不可用')}</strong><span>${esc(item.error)}</span></div>`
          : renderer && typeof renderer.render === 'function'
            ? renderer.render(item.payload || {}, item.locator || {})
            : `<div class="gpi-preview-state error">${esc(tr('Evidence renderer unavailable', '证据渲染器不可用'))}</div>`;
    } else {
      const text = isStructuredArtifact()
        ? JSON.stringify(state.payload || {}, null, 2)
        : (state.artifact && state.artifact.text != null ? state.artifact.text : '');
      body = `<pre class="gpi-preview-code" tabindex="0"><code>${esc(text)}</code></pre>`;
    }
    if (state.mode === 'evidence' && selectedEvidence && state.activeClaimId) {
      body = `<div class="gpi-evidence-stack">${body}<div class="gpi-evidence-actions"><span>${tr('Need the audit trail?', '需要核对审计链路？')}</span><button type="button" data-gpi-evidence-audit="${esc(state.activeClaimId)}">${tr('Open full evidence lineage', '打开完整证据链')}</button></div></div>`;
    }
    const reference = state.mode === 'evidence' && selectedEvidence
      ? `${state.resource.run_id} · ${selectedEvidence.evidenceId}`
      : isDataPackageReview()
      ? `${state.resource.study_context_id} · rev ${state.resource.study_revision}`
      : isDataWorkbenchSnapshot() ? `${state.resource.view} · ${state.resource.snapshot_sha256.slice(0, 12)}`
      : isNativeWorkspace() ? `${state.resource.entry_mode === 'source_binding' ? tr('data source', '数据来源') : state.resource.route} · rev ${state.resource.study_revision}`
      : isStructuredArtifact() || isDocument()
      ? `${state.resource.run_id} · ${state.resource.artifact}`
      : isLiteratureSource() ? state.resource.url : state.resource.file;
    const provenance = isDemoArtifact() ? demoProvenance() : isDemoDocument() ? demoDocumentProvenance() : isIdeaPlan() ? `
      <div class="gpi-preview-provenance is-research" role="note"><strong>${tr('Idea Mining plan · Planning only', 'Idea Mining 方案 · 仅限规划')}</strong><span>${tr('Built from the candidate ledger, literature receipt, and feasibility boundary. It is not an analysis result and cannot authorize execution.', '由候选 ledger、文献回执和可行性边界生成；不是分析结果，也不能授权执行。')}</span></div>` : isNativeWorkspace() ? `
      <div class="gpi-preview-provenance is-research" role="note"><strong>${tr('Native EasyICU owner · Local execution', 'EasyICU 原生 owner · 本地执行')}</strong><span>${tr('Folder paths and patient rows stay in the host UI; the model receives only governed receipts.', '目录路径和患者行只保留在本机界面；模型仅接收受治理回执。')}</span></div>` : isDataWorkbenchSnapshot() ? dataWorkbenchProvenance() : isDataPackageReview() ? dataPackageProvenance() : (isResearchArtifact() || isResearchReport() || isResearchDocument()) ? researchProvenance() : isLiteratureSource() ? `
      <div class="gpi-preview-provenance is-research" role="note">
        <strong>${tr('Literature metadata · Search receipt', '文献元数据 · 检索回执')}</strong>
        <span>${tr('Design evidence, separate from patient/result evidence.', '设计依据；与患者/结果证据分开治理。')}</span>
      </div>` : `
      <div class="gpi-preview-provenance" role="note">
        <strong>${tr('Workspace artifact · Unvalidated', '工作区产物 · 未验证')}</strong>
        <span>${tr('Not scientific evidence; unsupported for clinical or manuscript claims.', '不是科学证据；不支持临床或论文结论。')}</span>
      </div>`;
    const currentKey = resourceKey(state.resource);
    const recentPreviews = state.recentResources.length > 1 ? `
      <nav class="gpi-preview-recent" aria-label="${tr('Recent previews', '最近预览')}">
        <span>${tr('Recent', '回看')}</span>
        <div>${state.recentResources.map((resource, index) => `
          <button type="button" data-gpi-preview-recent="${index}" aria-current="${resourceKey(resource) === currentKey ? 'true' : 'false'}" title="${esc(resource.label)}">${esc(resource.label)}</button>`).join('')}</div>
      </nav>` : '';
    const previousInspector = state.host.querySelector && state.host.querySelector('.gpi-preview-inspector');
    const inspectorOpen = previousInspector && previousInspector.dataset.gpiPreviewResource === currentKey && previousInspector.open;
    const referenceOwner = window.EasyICU.guidedPi.optional('studyWorkspace');
    const canReference = state.projectId === state.studyProjectId && state.referenceResource && referenceOwner
      && referenceOwner.create({ tr, esc }).canReference(state.resource);
    const documentDownload = isResearchDocument() && /\.pdf$/i.test(state.resource.artifact || '') ? previewUrl() : '';
    state.host.innerHTML = `
      <div class="gpi-reader-context"><button type="button" data-gpi-preview-close>${icon('back', 14)} ${tr('Back to conversation', '返回对话')}</button><span title="${esc(state.studyTitle)}">${esc(state.projectId === state.studyProjectId ? state.studyTitle : '')}</span></div>
      <header class="gpi-preview-head">
        <div class="gpi-preview-file-icon" aria-hidden="true">${icon(state.mode === 'web' ? 'globe' : 'file', 16)}</div>
        <div class="gpi-preview-ident"><strong title="${esc(reference)}">${esc(state.resource.label)}</strong>${provenance}</div>
        ${canReference ? `<button class="gpi-preview-layout gpi-preview-reference-action" type="button" data-gpi-preview-reference>${tr('Reference in conversation', '引用到对话')}</button>` : ''}
        ${documentDownload ? `<a class="gpi-preview-layout gpi-preview-download" href="${esc(documentDownload)}" download="${esc(state.resource.artifact)}">${tr('Download PDF', '下载 PDF')}</a>` : ''}
        <button class="gpi-preview-layout" type="button" data-gpi-preview-focus aria-pressed="${state.focused}">${state.focused ? tr('Show conversation', '边聊边看') : tr('Focus reading', '专注阅读')}</button>
        <button class="gpi-preview-close" type="button" data-gpi-preview-close aria-label="${tr('Close preview', '关闭预览')}" title="${tr('Close preview', '关闭预览')}">${icon('close', 15)}</button>
      </header>
      <nav class="gpi-study-resource-tabs" data-gpi-study-resources aria-label="${tr('This run’s results', '本次运行成果')}"${studyResourcesHtml() ? '' : ' hidden'}>${studyResourcesHtml()}</nav>
      ${studyResourcesHtml() ? '' : recentPreviews}
      <details class="gpi-preview-inspector" data-gpi-preview-resource="${esc(currentKey)}"${inspectorOpen || state.mode === 'code' || state.mode === 'evidence' ? ' open' : ''}>
        <summary>${tr('Source & verification', '来源与核验')}</summary>
        <code class="gpi-preview-reference">${esc(reference)}</code>
        ${state.resource.sha256 ? `<code class="gpi-preview-reference">SHA-256 ${esc(state.resource.sha256)}</code>` : ''}
        ${provenance}
        ${tabs}
      </details>
      <div class="gpi-preview-body">${body}</div>`;
    if (state.mode === 'workbench' && (isDataPackageReview() || isDataWorkbenchSnapshot()) && !state.loading && !state.error) {
      const owner = isDataWorkbenchSnapshot()
        ? window.EasyICU.guidedPi.require('dataPreview')
        : window.EasyICU.guidedPi.require('workbenchPreview');
      const mount = state.host.querySelector('[data-gpi-workbench-mount]');
      if (owner && typeof owner.mount === 'function') owner.mount(mount, state.payload || {}, state.resource.view);
    }
    if (state.mode === 'native' && isNativeWorkspace() && !state.loading && !state.error) {
      const owner = window.EU_EXTRACTION_EMBEDDED_WORKSPACE;
      const mount = state.host.querySelector('[data-gpi-native-workspace-mount]');
      if (owner && typeof owner.mount === 'function') owner.mount(mount, {
        jobId: state.resource.job_id || '', jobSnapshot: state.payload || null,
        sourceId: state.resource.source_id || '',
        studyContext: state.studyContext,
        resource: state.resource,
      });
    }
  }
  async function openEvidence(button) {
    if (!isEvidenceBoundResource() || !button) return;
    const evidenceId = String(button.dataset.evidenceId || '').trim();
    const sha256 = String(button.dataset.evidenceSha256 || '').trim().toLowerCase();
    if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) return;
    const renderer = window.EasyICU.guidedPi.require('evidencePreview');
    const claimPanel = typeof button.closest === 'function' ? button.closest('[data-gpi-claim-panel]') : null;
    state.activeClaimId = String(button.dataset.gpiClaim || (claimPanel && claimPanel.dataset.gpiClaimPanel) || state.activeClaimId || '').trim();
    const label = String(button.dataset.evidenceLabel || evidenceId).slice(0, 160);
    const kind = String(button.dataset.evidenceKind || 'artifact').slice(0, 80);
    const locator = {
      pointer: String(button.dataset.evidencePointer || '').slice(0, 500),
      value: String(button.dataset.evidenceSourceValue || '').slice(0, 500),
      display: button.dataset.gpiClaim ? String(button.textContent || '').trim().slice(0, 120) : '',
    };
    let item = state.evidenceTabs.find(row => row.evidenceId === evidenceId);
    if (!item) {
      item = {
        evidenceId, sha256, label, kind,
        kindLabel: renderer && typeof renderer.kindLabel === 'function'
          ? renderer.kindLabel({ renderer: kind === 'code' ? 'code' : kind === 'table' ? 'table' : kind === 'statistic' ? 'json' : 'metadata' })
          : kind,
        locator, loading: true, error: '', payload: null,
      };
      state.evidenceTabs.push(item);
    } else {
      item.locator = locator;
    }
    state.activeEvidenceId = evidenceId;
    state.mode = 'evidence';
    if (typeof syncLocation === 'function') syncLocation(state.resource, {
      evidenceId, sha256, kind, pointer: locator.pointer,
    });
    render();
    if (item.payload || item.error || !item.loading) return;
    try {
      const api = window.EU_API || {};
      if (!api.loadPiCopilotResearchEvidence) throw new Error(tr('The evidence preview API is unavailable.', '证据预览接口不可用。'));
      const response = await api.loadPiCopilotResearchEvidence(
        state.projectId, state.resource.run_id, evidenceId, sha256,
      );
      item.payload = response && response.payload ? response.payload : {};
      item.kindLabel = renderer && typeof renderer.kindLabel === 'function'
        ? renderer.kindLabel(item.payload) : kind;
      item.error = '';
    } catch (error) {
      item.error = String(error && (error.message || error.code) || error);
    } finally {
      item.loading = false;
      if (state.activeEvidenceId === evidenceId) render();
    }
  }
  async function loadResource() {
    if (!state.resource || isDocument() || (!state.projectId && !isDemoArtifact() && !isLiteratureSource())) return;
    const ticket = ++state.request;
    state.loading = true; state.error = ''; render();
    try {
      const api = window.EU_API || {};
      let payload;
      let loadedStudyContext = null;
      if (isLiteratureSource()) {
        if (!state.resource.pmid || !api.loadPiCopilotLiteratureSource) {
          throw new Error(tr('The selected source cannot be enriched automatically.', '当前来源无法自动补充摘要或正文证据。'));
        }
        payload = await api.loadPiCopilotLiteratureSource(state.resource.pmid);
      } else if (isDemoArtifact()) {
        const demo = window.EasyICU.guidedPi.optional('demo');
        if (!demo || typeof demo.artifact !== 'function') throw new Error(tr('The product-demo artifact owner is unavailable.', '产品演示产物 owner 不可用。'));
        const item = typeof demo.previewArtifact === 'function'
          ? await demo.previewArtifact(state.resource.artifact)
          : demo.artifact(state.resource.artifact);
        if (!item) throw new Error(tr('The selected demo artifact does not exist.', '所选演示产物不存在。'));
        payload = {
          payload: item,
          governance: { claim_ceiling: 'analysis_only', reportable: false, human_signoff: 'required' },
        };
      } else if (isResearchReport()) {
        const owner = researchReportOwner();
        if (!owner || typeof owner.load !== 'function') throw new Error(tr('The report renderer is unavailable.', '报告渲染器不可用。'));
        payload = await owner.load(api, state.projectId, state.resource.run_id, state.resource);
      } else if (isResearchArtifact()) {
        if (!api.loadPiCopilotResearchArtifact) throw new Error(tr('The research artifact API is unavailable.', '研究产物接口不可用。'));
        payload = await api.loadPiCopilotResearchArtifact(
          state.projectId, state.resource.run_id, state.resource.artifact,
          state.resource.sha256,
        );
        if (ticket !== state.request) return;
        const reader = payload && payload.payload && (payload.payload.reader || (
          payload.payload.schema_version === 'easyicu.manuscript-provenance/1' ? payload.payload : null
        ));
        const galleryRef = reader && reader.figure_gallery_artifact;
        if (galleryRef) {
          if (galleryRef.name !== 'figure_gallery.json' || !/^[a-f0-9]{64}$/.test(String(galleryRef.sha256 || ''))) {
            throw new Error(tr('The reader figure source is invalid.', '文章图件来源绑定无效。'));
          }
          const gallery = await api.loadPiCopilotResearchArtifact(
            state.projectId, state.resource.run_id, galleryRef.name, galleryRef.sha256,
          );
          reader.figure_gallery = gallery.payload;
        }
      } else if (isIdeaPlan()) {
        if (!api.loadIdeaRun) throw new Error(tr('The Idea Mining run API is unavailable.', 'Idea Mining 运行接口不可用。'));
        const loaded = await api.loadIdeaRun({ run_id: state.resource.run_id });
        if (!loaded || !loaded.idea_plan) throw new Error(tr('The Idea Mining plan has not been generated.', 'Idea Mining 方案尚未生成。'));
        payload = {
          payload: loaded.idea_plan,
          governance: {
            authority_class: 'idea_mining_planning_only', claim_ceiling: 'planning_only',
            reportable: false, human_signoff: 'required',
          },
        };
      } else if (isDataPackageReview()) {
        if (!api.loadPiCopilotDataPackageReview) throw new Error(tr('The data-package review API is unavailable.', '数据包审阅接口不可用。'));
        payload = await api.loadPiCopilotDataPackageReview(
          state.projectId, state.resource.study_revision, state.resource.review_sha256,
        );
      } else if (isDataWorkbenchSnapshot()) {
        if (!api.loadPiCopilotDataWorkbenchSnapshot) throw new Error(tr('The conversational Data Workbench API is unavailable.', '对话式数据工作台接口不可用。'));
        payload = await api.loadPiCopilotDataWorkbenchSnapshot(
          state.projectId, state.resource.snapshot_sha256,
        );
      } else if (isNativeWorkspace()) {
        if (!api.loadStudyContext) throw new Error(tr('Study setup API is unavailable.', '研究配置 API 暂不可用。'));
        const [jobPayload, contextPayload] = await Promise.all([
          state.resource.job_id && api.loadJobSnapshot
            ? api.loadJobSnapshot(state.resource.job_id)
            : Promise.resolve(null),
          api.loadStudyContext(state.resource.study_context_id),
        ]);
        payload = jobPayload;
        loadedStudyContext = contextPayload && contextPayload.context ? contextPayload.context : null;
      } else {
        if (!api.loadPiCopilotWorkspaceFile) throw new Error(tr('The workspace file API is unavailable.', '工作区文件接口不可用。'));
        payload = await api.loadPiCopilotWorkspaceFile(state.projectId, state.resource.file);
      }
      if (ticket !== state.request) return;
      if (isLiteratureSource()) {
        // Merge only bounded enrichment fields; the response must not rewrite
        // safeResource-validated identity (kind, url, label, media_type,
        // authority_class) with unvalidated values.
        const enrichment = payload && typeof payload === 'object' ? payload : {};
        const enrichedTitle = String(enrichment.title || '').trim().slice(0, 500);
        state.resource = {
          ...state.resource,
          ...(enrichedTitle ? { title: enrichedTitle } : {}),
          year: String(enrichment.year || state.resource.year || '').slice(0, 16),
          doi: String(enrichment.doi || state.resource.doi || '').slice(0, 240),
          abstract_excerpt: String(enrichment.abstract_excerpt || '').slice(0, 1200),
          publication_types: (Array.isArray(enrichment.publication_types) ? enrichment.publication_types : [])
            .slice(0, 20).map(item => String(item || '').trim().slice(0, 160)).filter(Boolean),
          bibliographic_notices: (Array.isArray(enrichment.bibliographic_notices) ? enrichment.bibliographic_notices : [])
            .slice(0, 20).map(item => String(item || '').trim().slice(0, 600)).filter(Boolean),
          article_kind: String(enrichment.article_kind || '').slice(0, 80),
          full_text: enrichment.full_text && typeof enrichment.full_text === 'object' ? enrichment.full_text : null,
          source_review_status: 'reviewed',
        };
        state.payload = payload || null;
        return;
      }
      state.studyContext = isNativeWorkspace() ? loadedStudyContext : null;
      state.artifact = payload && payload.artifact ? payload.artifact : null;
      state.payload = isNativeWorkspace() ? payload : (isStructuredArtifact() && payload ? (payload.payload || {}) : null);
      state.governance = isStructuredArtifact() && payload ? (payload.governance || null) : null;
    } catch (error) {
      if (ticket !== state.request) return;
      if (isLiteratureSource()) {
        state.resource = {
          ...state.resource,
          source_review_status: 'unavailable',
          source_review_error: String(error && (error.message || error.code) || error),
        };
        state.error = '';
        return;
      }
      state.error = String(error && (error.message || error.code) || error);
    } finally {
      if (ticket === state.request) { state.loading = false; render(); }
    }
  }
  function open(resource, projectId, workflowContext) {
    const safe = safeResource(resource);
    const project = String(projectId || '').trim();
    if (!safe || (!project && safe.kind !== 'demo_artifact' && safe.kind !== 'demo_document' && safe.kind !== 'literature_source')) return;
    // An open workbench tab survives opening a file: the session owner closes
    // the workbench only when the conversation itself changes.
    state.request += 1;
    state.loading = false;
    if (!state.resource || state.projectId !== project) state.focused = false;
    if (state.projectId !== project) state.recentResources = [];
    const catalogResource = project === state.studyProjectId && state.studyResources.find(row =>
      resourceKey(row) === resourceKey(safe) && row.sha256 === safe.sha256);
    state.resource = catalogResource ? { ...safe, label: catalogResource.label } : safe;
    state.projectId = project;
    state.workflowContext = safeWorkflowContext(workflowContext);
    rememberResource(state.resource);
    state.artifact = null;
    state.payload = null;
    state.studyContext = null;
    state.governance = null;
    state.evidenceTabs = [];
    state.activeEvidenceId = '';
    state.error = '';
    state.mode = safe.kind === 'native_workspace' ? 'native' : safe.kind === 'research_document' || safe.kind === 'system_validation_document' || safe.kind === 'demo_document' ? 'document' : (safe.kind === 'data_package_review' || safe.kind === 'data_workbench_snapshot' ? 'workbench' : (safe.kind === 'research_artifact' || safe.kind === 'research_report' || safe.kind === 'idea_plan' || safe.kind === 'demo_artifact' ? 'structured' : (safe.kind === 'literature_source' ? 'source' : (safe.kind === 'webpage' ? 'web' : 'code'))));
    state.activeClaimId = '';
    if (typeof syncLocation === 'function') syncLocation(state.resource);
    render();
    if (state.mode !== 'web' && state.mode !== 'document') loadResource();
    return true;
  }
  function openRunEvidence(resource, projectId, button) {
    if (open(resource, projectId)) return openEvidence(button);
  }
  function close(options) {
    const preserveLocation = Boolean(options && options.preserveLocation);
    state.request += 1;
    state.focused = false;
    state.resource = null; state.artifact = null; state.payload = null; state.studyContext = null; state.governance = null; state.error = ''; state.loading = false;
    state.evidenceTabs = []; state.activeEvidenceId = '';
    state.activeClaimId = '';
    if (!preserveLocation && typeof syncLocation === 'function') syncLocation(null);
    setAsideOpen(false);
    if (state.host) state.host.replaceChildren();
  }
  function clearProject(options) { close(options); state.projectId = ''; state.recentResources = []; state.workflowContext = {}; state.studyResources = []; state.studyProjectId = ''; state.openStudyResource = null; state.studyTitle = ''; state.referenceResource = null; }
  function mount(host) {
    if (!host) return;
    state.host = host;
    host.addEventListener('click', event => {
      if (event.target.closest('[data-gpi-preview-close]')) { close(); return; }
      if (event.target.closest('[data-gpi-preview-reference]')) {
        if (state.projectId === state.studyProjectId && state.referenceResource) state.referenceResource(state.resource, state.projectId);
        return;
      }
      const focus = event.target.closest('[data-gpi-preview-focus]');
      if (focus) { toggleFocus(focus); return; }
      const studyResource = event.target.closest('[data-gpi-study-resource]');
      if (studyResource) {
        const resource = state.studyResources[Number(studyResource.dataset.gpiStudyResource)];
        if (resource && state.resource && state.projectId === state.studyProjectId && resource.run_id === state.resource.run_id) {
          if (state.openStudyResource) state.openStudyResource(resource);
          else open(resource, state.projectId, state.workflowContext);
        }
        return;
      }
      const recent = event.target.closest('[data-gpi-preview-recent]');
      if (recent) {
        const resource = state.recentResources[Number(recent.dataset.gpiPreviewRecent)];
        if (resource) open(resource, state.projectId, state.workflowContext);
        return;
      }
      const evidenceClose = event.target.closest('[data-gpi-evidence-tab-close]');
      if (evidenceClose) {
        const evidenceId = String(evidenceClose.dataset.gpiEvidenceTabClose || '');
        state.evidenceTabs = state.evidenceTabs.filter(item => item.evidenceId !== evidenceId);
        if (state.activeEvidenceId === evidenceId) {
          state.activeEvidenceId = '';
          state.mode = 'structured';
          if (typeof syncLocation === 'function') syncLocation(state.resource);
        }
        render();
        return;
      }
      const evidenceTab = event.target.closest('[data-gpi-evidence-tab]');
      if (evidenceTab) {
        state.activeEvidenceId = String(evidenceTab.dataset.gpiEvidenceTab || '');
        state.mode = 'evidence';
        const activeEvidence = state.evidenceTabs.find(item => item.evidenceId === state.activeEvidenceId);
        if (typeof syncLocation === 'function' && activeEvidence) syncLocation(state.resource, activeEvidence);
        render();
        return;
      }
      const evidenceAudit = event.target.closest('[data-gpi-evidence-audit]');
      if (evidenceAudit) {
        state.mode = 'structured';
        if (typeof syncLocation === 'function') syncLocation(state.resource);
        render();
        showClaimLineage(String(evidenceAudit.dataset.gpiEvidenceAudit || '').trim());
        return;
      }
      const evidenceButton = event.target.closest('[data-gpi-evidence-open]');
      if (evidenceButton) { openEvidence(evidenceButton); return; }
      const sourceButton = event.target.closest('[data-gpi-source-code]');
      if (sourceButton) {
        const sourceView = window.EasyICU.guidedPi.require('sourceView');
        if (sourceView && sourceView.open && state.resource && state.resource.run_id) {
          void sourceView.open(sourceButton, state.projectId, state.resource.run_id);
        }
        return;
      }
      const referenceLink = event.target.closest('[data-gpi-reference]');
      if (referenceLink) {
        // The app owns URL hashes for routing; keep article anchors local.
        event.preventDefault();
        const number = String(referenceLink.dataset.gpiReference || '');
        if (/^[1-9][0-9]*$/.test(number)) {
          const reference = host.querySelector('#gpi-reference-' + number);
          if (reference) reference.scrollIntoView({ block: 'start' });
        }
        return;
      }
      const reportArtifact = event.target.closest('[data-gpi-report-artifact]');
      if (reportArtifact && state.resource && state.resource.run_id) {
        const artifact = String(reportArtifact.dataset.gpiReportArtifact || '');
        open({
          kind: 'research_artifact', run_id: state.resource.run_id, artifact,
          label: String(reportArtifact.dataset.gpiReportLabel || artifact), media_type: 'application/json',
        }, state.projectId, state.workflowContext);
        return;
      }
      const displayLink = event.target.closest('[data-gpi-display]');
      if (displayLink) {
        event.preventDefault();
        const displayId = String(displayLink.dataset.gpiDisplay || '').trim();
        if (/^[A-Za-z0-9 _.:-]{1,160}$/.test(displayId)) {
          const anchor = Array.from(host.querySelectorAll('[data-gpi-display-anchor]'))
            .find(node => String(node.dataset.gpiDisplayAnchor || '') === displayId);
          if (anchor) {
            anchor.scrollIntoView({ block: 'start' });
            anchor.classList.add('is-focused');
            const clearFocus = () => anchor.classList.remove('is-focused');
            if (typeof window.setTimeout === 'function') window.setTimeout(clearFocus, 1600);
          }
        }
        return;
      }
      const claimButton = event.target.closest('[data-gpi-claim]');
      if (claimButton) {
        showClaimLineage(String(claimButton.dataset.gpiClaim || '').trim());
        return;
      }
      if (event.target.closest('[data-gpi-claim-close]')) {
        host.querySelectorAll('[data-gpi-claim]').forEach(button => button.setAttribute('aria-expanded', 'false'));
        host.querySelectorAll('[data-gpi-claim-panel]').forEach(panel => { panel.hidden = true; });
        const empty = host.querySelector('[data-gpi-claim-empty]');
        if (empty) empty.hidden = false;
        const drawer = host.querySelector('.gpi-claim-drawer');
        if (drawer) drawer.classList.remove('is-active');
        const layout = host.querySelector('[data-gpi-manuscript-layout]');
        if (layout) layout.classList.remove('has-claim-drawer');
        return;
      }
      const tab = event.target.closest('[data-gpi-preview-mode]');
      if (!tab || !state.resource) return;
      const requested = tab.dataset.gpiPreviewMode;
      const mode = requested === 'web' ? 'web' : (requested === 'workbench' ? 'workbench' : (requested === 'structured' ? 'structured' : 'code'));
      if (mode === state.mode) return;
      state.mode = mode;
      render();
      if (mode !== 'web' && !state.artifact && !state.loading) loadResource();
    });
    if (!state.resource) setAsideOpen(false);
    else render();
  }

  function showClaimLineage(claimId) {
    if (!state.host || !claimId) return;
    state.activeClaimId = claimId;
    state.host.querySelectorAll('[data-gpi-claim]').forEach(button => {
      button.setAttribute('aria-expanded', String(button.dataset.gpiClaim === claimId));
    });
    state.host.querySelectorAll('[data-gpi-claim-panel]').forEach(panel => {
      panel.hidden = panel.dataset.gpiClaimPanel !== claimId;
    });
    const empty = state.host.querySelector('[data-gpi-claim-empty]');
    if (empty) empty.hidden = true;
    const drawer = state.host.querySelector('.gpi-claim-drawer');
    if (drawer) drawer.classList.add('is-active');
    const layout = state.host.querySelector('[data-gpi-manuscript-layout]');
    if (layout) layout.classList.add('has-claim-drawer');
    const panel = Array.from(state.host.querySelectorAll('[data-gpi-claim-panel]'))
      .find(item => item.dataset.gpiClaimPanel === claimId);
    if (panel) panel.scrollIntoView({ block: 'nearest' });
  }

  window.EasyICU.guidedPi.declare('preview', { mount, open, openRunEvidence, close, clearProject, restoreFromLocation, setWorkflowContext, setStudyResources });
})();
