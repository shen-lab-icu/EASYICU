/* Guided Pi governed-resource preview owner.
   It swaps the study-progress aside for one clicked project file, webpage, or
   path-free Research Agent artifact reference. */
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
  };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function icon(name, size) {
    return typeof window.icon === 'function' ? window.icon(name, size || 16, 1.55) : '';
  }
  function safeResource(value) {
    if (!value || typeof value !== 'object') return null;
    if (value.kind === 'demo_artifact') {
      const demo = window.EU_GUIDED_PI_DEMO;
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
      const literature = window.EU_GUIDED_PI_LITERATURE;
      const authorityClass = value.authority_class === 'literature_method'
        ? 'literature_method' : 'literature_retrieval_candidate';
      const url = literature && typeof literature.safeUrl === 'function'
        ? literature.safeUrl(value.url) : '';
      const title = String(value.title || value.label || '').trim().slice(0, 500);
      if (!url || !title) return null;
      return {
        kind: 'literature_source', url, title,
        label: String(value.label || title).slice(0, 160),
        year: String(value.year || '').slice(0, 16),
        venue: String(value.venue || '').slice(0, 240),
        relevance: String(value.relevance || '').slice(0, 1200),
        doi: String(value.doi || '').slice(0, 240),
        pmid: String(value.pmid || '').slice(0, 32),
        media_type: 'text/html', authority_class: authorityClass,
      };
    }
    if (value.kind === 'research_artifact') {
      const runId = String(value.run_id || '').trim();
      const artifact = String(value.artifact || '').trim();
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(runId)) return null;
      if (!/^[A-Za-z0-9_.-]+\.json$/.test(artifact) || artifact.length > 160) return null;
      return {
        kind: 'research_artifact', run_id: runId, artifact,
        label: String(value.label || artifact).slice(0, 160),
        media_type: 'application/json',
      };
    }
    if (value.kind === 'research_document' || value.kind === 'system_validation_document') {
      const runId = String(value.run_id || '').trim();
      const artifact = String(value.artifact || '').trim();
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(runId)) return null;
      const validationDocument = value.kind === 'system_validation_document';
      if (validationDocument
        ? !/^system_validation_report\.(html|pdf)$/.test(artifact)
        : !/^manuscript_scaffold\.(pdf|tex|bib)$/.test(artifact)) return null;
      return {
        kind: validationDocument ? 'system_validation_document' : 'research_document', run_id: runId, artifact,
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
      const supportedDatabases = new Set(['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic']);
      if (route !== 'extraction' || !['setup', 'running', 'review'].includes(state)) return null;
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(studyContextId)) return null;
      if (!Number.isInteger(studyRevision) || studyRevision < 0) return null;
      if (jobId && !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/.test(jobId)) return null;
      if (sourceId && !/^src_[a-f0-9]{12}$/.test(sourceId)) return null;
      if (expectedDatabase && !supportedDatabases.has(expectedDatabase)) return null;
      if (entryMode && entryMode !== 'source_binding') return null;
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
    const owner = window.EU_GUIDED_PI_RESOURCES;
    if (owner && typeof owner.create === 'function') {
      const identity = owner.create({ esc });
      if (identity && typeof identity.key === 'function') return identity.key(resource);
    }
    return JSON.stringify(resource || {});
  }
  function rememberResource(resource) {
    const key = resourceKey(resource);
    state.recentResources = [resource]
      .concat(state.recentResources.filter(item => resourceKey(item) !== key))
      .slice(0, 6);
  }
  function isResearchArtifact() { return !!state.resource && state.resource.kind === 'research_artifact'; }
  function isResearchDocument() { return !!state.resource && (state.resource.kind === 'research_document' || state.resource.kind === 'system_validation_document'); }
  function isDemoDocument() { return !!state.resource && state.resource.kind === 'demo_document'; }
  function isDocument() { return isResearchDocument() || isDemoDocument(); }
  function isSystemValidationDocument() { return !!state.resource && state.resource.kind === 'system_validation_document'; }
  function isDemoArtifact() { return !!state.resource && state.resource.kind === 'demo_artifact'; }
  function isDataPackageReview() { return !!state.resource && state.resource.kind === 'data_package_review'; }
  function isDataWorkbenchSnapshot() { return !!state.resource && state.resource.kind === 'data_workbench_snapshot'; }
  function isNativeWorkspace() { return !!state.resource && state.resource.kind === 'native_workspace'; }
  function isStructuredArtifact() { return isResearchArtifact() || isDemoArtifact() || isDataPackageReview() || isDataWorkbenchSnapshot(); }
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
      return api.piCopilotResearchDocumentUrl
        ? api.piCopilotResearchDocumentUrl(state.projectId, state.resource.run_id, state.resource.artifact)
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
    if (study) study.hidden = !!open;
    if (state.host) state.host.hidden = !open;
    if (aside) aside.classList.toggle('gpi-preview-open', !!open);
    if (main) main.classList.toggle('gpi-preview-open', !!open);
  }
  function researchProvenance() {
    const governance = state.governance || {};
    if (isSystemValidationDocument() || governance.authority_class === 'easyicu_system_validation_report') {
      return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('System validation dossier · Engineering evidence only', '系统验证报告 · 仅限工程证据'))}</strong><span>${esc(tr('Not a clinical manuscript; cannot grant scientific or publication authority.', '不是临床论文；不能授予科学或发表权限。'))}</span></div>`;
    }
    if (!state.governance) {
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
  function render() {
    if (!state.host || !state.resource) return;
    setAsideOpen(true);
    const tabs = isLiteratureSource() ? '' : isStructuredArtifact() ? `
      <div class="gpi-preview-tabs" role="tablist" aria-label="${tr('Artifact views', '产物视图')}">
        ${isDataPackageReview() || isDataWorkbenchSnapshot() ? `<button type="button" role="tab" data-gpi-preview-mode="workbench" aria-selected="${state.mode === 'workbench'}">${icon('grid', 14)} ${tr('Workbench', '数据工作台')}</button>` : ''}
        <button type="button" role="tab" data-gpi-preview-mode="structured" aria-selected="${state.mode === 'structured'}">${icon('list', 14)} ${tr('Readable', '可读视图')}</button>
        <button type="button" role="tab" data-gpi-preview-mode="code" aria-selected="${state.mode === 'code'}">${icon('file', 14)} JSON</button>
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
      const renderer = window.EU_GUIDED_PI_LITERATURE;
      body = renderer && typeof renderer.renderSource === 'function'
        ? renderer.renderSource(state.resource)
        : `<div class="gpi-preview-state error">${esc(tr('Literature renderer unavailable', '文献渲染器不可用'))}</div>`;
    } else if (state.mode === 'document' && isDocument()) {
      body = `<iframe class="gpi-preview-frame gpi-preview-document-frame" src="${esc(previewUrl())}" referrerpolicy="no-referrer" title="${esc(tr('Preview of ', '预览：') + state.resource.label)}"></iframe>`;
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
      const literature = window.EU_GUIDED_PI_LITERATURE;
      body = state.resource.artifact === 'literature_evidence.json'
        && literature && typeof literature.renderArtifact === 'function'
        ? literature.renderArtifact(state.payload || {})
        : renderer && typeof renderer.artifactStructuredView === 'function'
          ? renderer.artifactStructuredView(state.resource.artifact, state.payload || {})
        : `<pre class="gpi-preview-code" tabindex="0"><code>${esc(JSON.stringify(state.payload || {}, null, 2))}</code></pre>`;
    } else {
      const text = isStructuredArtifact()
        ? JSON.stringify(state.payload || {}, null, 2)
        : (state.artifact && state.artifact.text != null ? state.artifact.text : '');
      body = `<pre class="gpi-preview-code" tabindex="0"><code>${esc(text)}</code></pre>`;
    }
    const reference = isDataPackageReview()
      ? `${state.resource.study_context_id} · rev ${state.resource.study_revision}`
      : isDataWorkbenchSnapshot() ? `${state.resource.view} · ${state.resource.snapshot_sha256.slice(0, 12)}`
      : isNativeWorkspace() ? `${state.resource.entry_mode === 'source_binding' ? tr('data source', '数据来源') : state.resource.route} · rev ${state.resource.study_revision}`
      : isStructuredArtifact() || isDocument()
      ? `${state.resource.run_id} · ${state.resource.artifact}`
      : isLiteratureSource() ? state.resource.url : state.resource.file;
    const provenance = isDemoArtifact() ? demoProvenance() : isDemoDocument() ? demoDocumentProvenance() : isNativeWorkspace() ? `
      <div class="gpi-preview-provenance is-research" role="note"><strong>${tr('Native EasyICU owner · Local execution', 'EasyICU 原生 owner · 本地执行')}</strong><span>${tr('Folder paths and patient rows stay in the host UI; the model receives only governed receipts.', '目录路径和患者行只保留在本机界面；模型仅接收受治理回执。')}</span></div>` : isDataWorkbenchSnapshot() ? dataWorkbenchProvenance() : isDataPackageReview() ? dataPackageProvenance() : (isResearchArtifact() || isResearchDocument()) ? researchProvenance() : isLiteratureSource() ? `
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
    state.host.innerHTML = `
      <header class="gpi-preview-head">
        <div class="gpi-preview-file-icon" aria-hidden="true">${icon(state.mode === 'web' ? 'globe' : 'file', 16)}</div>
        <div class="gpi-preview-ident"><strong>${esc(state.resource.label)}</strong><span>${esc(reference)}</span></div>
        <button class="gpi-preview-close" type="button" data-gpi-preview-close aria-label="${tr('Close preview', '关闭预览')}" title="${tr('Close preview', '关闭预览')}">${icon('close', 15)}</button>
      </header>
      ${recentPreviews}
      ${provenance}
      ${tabs}
      <div class="gpi-preview-body">${body}</div>`;
    if (state.mode === 'workbench' && (isDataPackageReview() || isDataWorkbenchSnapshot()) && !state.loading && !state.error) {
      const owner = isDataWorkbenchSnapshot()
        ? window.EU_GUIDED_PI_DATA_PREVIEW
        : window.EU_GUIDED_PI_WORKBENCH_PREVIEW;
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
  async function loadResource() {
    if (!state.resource || isLiteratureSource() || isDocument() || (!state.projectId && !isDemoArtifact())) return;
    const ticket = ++state.request;
    state.loading = true; state.error = ''; render();
    try {
      const api = window.EU_API || {};
      let payload;
      let loadedStudyContext = null;
      if (isDemoArtifact()) {
        const demo = window.EU_GUIDED_PI_DEMO;
        if (!demo || typeof demo.artifact !== 'function') throw new Error(tr('The product-demo artifact owner is unavailable.', '产品演示产物 owner 不可用。'));
        const item = typeof demo.previewArtifact === 'function'
          ? await demo.previewArtifact(state.resource.artifact)
          : demo.artifact(state.resource.artifact);
        if (!item) throw new Error(tr('The selected demo artifact does not exist.', '所选演示产物不存在。'));
        payload = {
          payload: item,
          governance: { claim_ceiling: 'analysis_only', reportable: false, human_signoff: 'required' },
        };
      } else if (isResearchArtifact()) {
        if (!api.loadPiCopilotResearchArtifact) throw new Error(tr('The research artifact API is unavailable.', '研究产物接口不可用。'));
        payload = await api.loadPiCopilotResearchArtifact(
          state.projectId, state.resource.run_id, state.resource.artifact,
        );
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
      state.studyContext = isNativeWorkspace() ? loadedStudyContext : null;
      state.artifact = payload && payload.artifact ? payload.artifact : null;
      state.payload = isNativeWorkspace() ? payload : (isStructuredArtifact() && payload ? (payload.payload || {}) : null);
      state.governance = isStructuredArtifact() && payload ? (payload.governance || null) : null;
    } catch (error) {
      if (ticket !== state.request) return;
      state.error = String(error && (error.message || error.code) || error);
    } finally {
      if (ticket === state.request) { state.loading = false; render(); }
    }
  }
  function open(resource, projectId) {
    const safe = safeResource(resource);
    const project = String(projectId || '').trim();
    if (!safe || (!project && safe.kind !== 'demo_artifact' && safe.kind !== 'demo_document' && safe.kind !== 'literature_source')) return;
    if (state.projectId && project && state.projectId !== project) state.recentResources = [];
    state.resource = safe;
    state.projectId = project;
    rememberResource(safe);
    state.artifact = null;
    state.payload = null;
    state.studyContext = null;
    state.governance = null;
    state.error = '';
    state.mode = safe.kind === 'native_workspace' ? 'native' : safe.kind === 'research_document' || safe.kind === 'system_validation_document' || safe.kind === 'demo_document' ? 'document' : (safe.kind === 'data_package_review' || safe.kind === 'data_workbench_snapshot' ? 'workbench' : (safe.kind === 'research_artifact' || safe.kind === 'demo_artifact' ? 'structured' : (safe.kind === 'literature_source' ? 'source' : (safe.kind === 'webpage' ? 'web' : 'code'))));
    render();
    if (state.mode !== 'web' && state.mode !== 'source' && state.mode !== 'document') loadResource();
  }
  function close() {
    state.request += 1;
    state.resource = null; state.artifact = null; state.payload = null; state.studyContext = null; state.governance = null; state.error = ''; state.loading = false;
    setAsideOpen(false);
    if (state.host) state.host.replaceChildren();
  }
  function clearProject() { close(); state.projectId = ''; state.recentResources = []; }
  function mount(host) {
    if (!host) return;
    state.host = host;
    host.addEventListener('click', event => {
      if (event.target.closest('[data-gpi-preview-close]')) { close(); return; }
      const recent = event.target.closest('[data-gpi-preview-recent]');
      if (recent) {
        const resource = state.recentResources[Number(recent.dataset.gpiPreviewRecent)];
        if (resource) open(resource, state.projectId);
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

  window.EU_GUIDED_PI_PREVIEW = { mount, open, close, clearProject };
})();
