/* Guided Pi governed-resource preview owner.
   It swaps the study-progress aside for one clicked project file, webpage, or
   path-free Research Agent artifact reference. */
(function () {
  'use strict';

  const state = {
    host: null,
    projectId: '',
    resource: null,
    artifact: null,
    payload: null,
    governance: null,
    mode: 'code',
    loading: false,
    error: '',
    request: 0,
  };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
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
      const authorityClass = value.authority_class === 'literature_method'
        ? 'literature_method' : 'literature_retrieval_candidate';
      return {
        kind: 'demo_artifact', artifact,
        run_id: String(value.run_id || demo.sourceRunId || '').slice(0, 160),
        label: String(value.label || (demo.artifactLabel && demo.artifactLabel(artifact)) || artifact).slice(0, 160),
        media_type: 'application/json', authority_class: 'product_demo_projection',
      };
    }
    if (value.kind === 'literature_source') {
      const literature = window.EU_GUIDED_PI_LITERATURE;
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
    if (value.kind === 'research_document') {
      const runId = String(value.run_id || '').trim();
      const artifact = String(value.artifact || '').trim();
      if (!/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(runId)) return null;
      if (!/^manuscript_scaffold\.(pdf|tex|bib)$/.test(artifact)) return null;
      return {
        kind: 'research_document', run_id: runId, artifact,
        label: String(value.label || artifact).slice(0, 160),
        media_type: String(value.media_type || (artifact.endsWith('.pdf') ? 'application/pdf' : 'text/plain')).slice(0, 120),
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
    const file = String(value.file || '').trim().replace(/\\/g, '/');
    if (!file || file.startsWith('/') || file.includes('\0')) return null;
    if (file.split('/').some(part => !part || part === '.' || part === '..')) return null;
    return {
      kind: value.kind === 'webpage' ? 'webpage' : 'file',
      file: file.slice(0, 240),
      label: String(value.label || file.split('/').pop() || file).slice(0, 160),
      media_type: String(value.media_type || 'text/plain').slice(0, 120),
      authority_class: 'workspace_artifact',
      scientific_evidence: false,
      validation_status: 'unvalidated',
      claim_ceiling: 'unsupported',
    };
  }
  function isResearchArtifact() { return !!state.resource && state.resource.kind === 'research_artifact'; }
  function isResearchDocument() { return !!state.resource && state.resource.kind === 'research_document'; }
  function isDemoArtifact() { return !!state.resource && state.resource.kind === 'demo_artifact'; }
  function isDataPackageReview() { return !!state.resource && state.resource.kind === 'data_package_review'; }
  function isStructuredArtifact() { return isResearchArtifact() || isDemoArtifact() || isDataPackageReview(); }
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
    if (isResearchDocument()) {
      return api.piCopilotResearchDocumentUrl
        ? api.piCopilotResearchDocumentUrl(state.projectId, state.resource.run_id, state.resource.artifact)
        : '';
    }
    return api.piCopilotWorkspacePreviewUrl
      ? api.piCopilotWorkspacePreviewUrl(state.projectId, state.resource.file)
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
    const authority = String((state.payload && state.payload.source_authority) || '');
    if (authority === 'independent_product_audit_20260811') {
      return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('Product demo · Historical run plus independent current audit', '产品演示 · 历史运行与独立当前审计'))}</strong><span>${esc(tr('The current sources adjudicate the historical canary; they were not retroactively inserted into that Agent run.', '当前来源用于裁决历史 canary；并未追溯性写入该 Agent 运行。'))}</span></div>`;
    }
    return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('Product demo · Real engineering-canary aggregate', '产品演示 · 真实工程试跑聚合产物'))}</strong><span>${esc(tr('Read-only and non-reportable; not formal paper evidence.', '只读且不可报告；不是正式论文证据。'))}</span></div>`;
  }
  function dataPackageProvenance() {
    return `<div class="gpi-preview-provenance is-research" role="note"><strong>${esc(tr('Registered export · Pre-analysis review', '已登记数据源 · 分析前审阅'))}</strong><span>${esc(tr('Aggregate denominator and availability only; event rates, comparisons, and effect estimates are withheld until the governed analysis.', '仅展示聚合分母与可用性；事件率、组间比较和效应量留待受治理分析。'))}</span></div>`;
  }
  function render() {
    if (!state.host || !state.resource) return;
    setAsideOpen(true);
    const tabs = isLiteratureSource() ? '' : isStructuredArtifact() ? `
      <div class="gpi-preview-tabs" role="tablist" aria-label="${tr('Artifact views', '产物视图')}">
        <button type="button" role="tab" data-gpi-preview-mode="structured" aria-selected="${state.mode === 'structured'}">${icon('list', 14)} ${tr('Readable', '可读视图')}</button>
        <button type="button" role="tab" data-gpi-preview-mode="code" aria-selected="${state.mode === 'code'}">${icon('file', 14)} JSON</button>
      </div>` : isHtml() ? `
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
    } else if (state.mode === 'document' && isResearchDocument()) {
      body = `<iframe class="gpi-preview-frame gpi-preview-document-frame" src="${esc(previewUrl())}" referrerpolicy="no-referrer" title="${esc(tr('Preview of ', '预览：') + state.resource.label)}"></iframe>`;
    } else if (state.mode === 'web' && isHtml()) {
      body = `<iframe class="gpi-preview-frame" src="${esc(previewUrl())}" sandbox="allow-scripts" referrerpolicy="no-referrer" title="${esc(tr('Preview of ', '预览：') + state.resource.label)}"></iframe>`;
    } else if (state.mode === 'structured' && isStructuredArtifact()) {
      const renderer = window.AGENT_RENDER;
      const literature = window.EU_GUIDED_PI_LITERATURE;
      const demo = window.EU_GUIDED_PI_DEMO;
      body = isDemoArtifact() && demo && typeof demo.renderArtifact === 'function'
        ? demo.renderArtifact(state.payload || {})
        : state.resource.artifact === 'literature_evidence.json'
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
      : isStructuredArtifact() || isResearchDocument()
      ? `${state.resource.run_id} · ${state.resource.artifact}`
      : isLiteratureSource() ? state.resource.url : state.resource.file;
    const provenance = isDemoArtifact() ? demoProvenance() : isDataPackageReview() ? dataPackageProvenance() : (isResearchArtifact() || isResearchDocument()) ? researchProvenance() : isLiteratureSource() ? `
      <div class="gpi-preview-provenance is-research" role="note">
        <strong>${tr('Literature metadata · Search receipt', '文献元数据 · 检索回执')}</strong>
        <span>${tr('Design evidence, separate from patient/result evidence.', '设计依据；与患者/结果证据分开治理。')}</span>
      </div>` : `
      <div class="gpi-preview-provenance" role="note">
        <strong>${tr('Workspace artifact · Unvalidated', '工作区产物 · 未验证')}</strong>
        <span>${tr('Not scientific evidence; unsupported for clinical or manuscript claims.', '不是科学证据；不支持临床或论文结论。')}</span>
      </div>`;
    state.host.innerHTML = `
      <header class="gpi-preview-head">
        <div class="gpi-preview-file-icon" aria-hidden="true">${icon(state.mode === 'web' ? 'globe' : 'file', 16)}</div>
        <div class="gpi-preview-ident"><strong>${esc(state.resource.label)}</strong><span>${esc(reference)}</span></div>
        <button class="gpi-preview-close" type="button" data-gpi-preview-close aria-label="${tr('Close preview', '关闭预览')}" title="${tr('Close preview', '关闭预览')}">${icon('close', 15)}</button>
      </header>
      ${provenance}
      ${tabs}
      <div class="gpi-preview-body">${body}</div>`;
  }
  async function loadResource() {
    if (!state.resource || isLiteratureSource() || isResearchDocument() || (!state.projectId && !isDemoArtifact())) return;
    const ticket = ++state.request;
    state.loading = true; state.error = ''; render();
    try {
      const api = window.EU_API || {};
      let payload;
      if (isDemoArtifact()) {
        const demo = window.EU_GUIDED_PI_DEMO;
        if (!demo || typeof demo.artifact !== 'function') throw new Error(tr('The product-demo artifact owner is unavailable.', '产品演示产物 owner 不可用。'));
        const item = demo.artifact(state.resource.artifact);
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
      } else {
        if (!api.loadPiCopilotWorkspaceFile) throw new Error(tr('The workspace file API is unavailable.', '工作区文件接口不可用。'));
        payload = await api.loadPiCopilotWorkspaceFile(state.projectId, state.resource.file);
      }
      if (ticket !== state.request) return;
      state.artifact = payload && payload.artifact ? payload.artifact : null;
      state.payload = isStructuredArtifact() && payload ? (payload.payload || {}) : null;
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
    if (!safe || (!project && safe.kind !== 'demo_artifact' && safe.kind !== 'literature_source')) return;
    state.resource = safe;
    state.projectId = project;
    state.artifact = null;
    state.payload = null;
    state.governance = null;
    state.error = '';
    state.mode = safe.kind === 'research_document' ? 'document' : (safe.kind === 'research_artifact' || safe.kind === 'demo_artifact' || safe.kind === 'data_package_review' ? 'structured' : (safe.kind === 'literature_source' ? 'source' : (safe.kind === 'webpage' ? 'web' : 'code')));
    render();
    if (state.mode !== 'web' && state.mode !== 'source' && state.mode !== 'document') loadResource();
  }
  function close() {
    state.request += 1;
    state.resource = null; state.artifact = null; state.payload = null; state.governance = null; state.error = ''; state.loading = false;
    setAsideOpen(false);
    if (state.host) state.host.replaceChildren();
  }
  function clearProject() { close(); state.projectId = ''; }
  function mount(host) {
    if (!host) return;
    state.host = host;
    host.addEventListener('click', event => {
      if (event.target.closest('[data-gpi-preview-close]')) { close(); return; }
      const tab = event.target.closest('[data-gpi-preview-mode]');
      if (!tab || !state.resource) return;
      const requested = tab.dataset.gpiPreviewMode;
      const mode = requested === 'web' ? 'web' : (requested === 'structured' ? 'structured' : 'code');
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
