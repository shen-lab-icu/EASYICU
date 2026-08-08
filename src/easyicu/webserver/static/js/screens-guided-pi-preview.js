/* Guided Pi project-artifact preview owner.
   It swaps the study-progress aside for one clicked file or webpage and never
   interprets paths outside the project-isolated Pi workspace. */
(function () {
  'use strict';

  const state = {
    host: null,
    projectId: '',
    resource: null,
    artifact: null,
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
    const file = String(value.file || '').trim().replace(/\\/g, '/');
    if (!file || file.startsWith('/') || file.includes('\0')) return null;
    if (file.split('/').some(part => !part || part === '.' || part === '..')) return null;
    return {
      kind: value.kind === 'webpage' ? 'webpage' : 'file',
      file: file.slice(0, 240),
      label: String(value.label || file.split('/').pop() || file).slice(0, 160),
      media_type: String(value.media_type || 'text/plain').slice(0, 120),
    };
  }
  function isHtml() {
    return !!state.resource && (
      state.resource.kind === 'webpage'
      || state.resource.media_type === 'text/html'
      || /\.html?$/i.test(state.resource.file)
    );
  }
  function previewUrl() {
    const api = window.EU_API || {};
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
  function render() {
    if (!state.host || !state.resource) return;
    setAsideOpen(true);
    const tabs = isHtml() ? `
      <div class="gpi-preview-tabs" role="tablist" aria-label="${tr('Artifact views', '产物视图')}">
        <button type="button" role="tab" data-gpi-preview-mode="code" aria-selected="${state.mode === 'code'}">${icon('file', 14)} ${tr('Code', '代码')}</button>
        <button type="button" role="tab" data-gpi-preview-mode="web" aria-selected="${state.mode === 'web'}">${icon('globe', 14)} ${tr('Web preview', '网页预览')}</button>
      </div>` : '';
    let body = '';
    if (state.loading) {
      body = `<div class="gpi-preview-state"><span class="gpi-preview-spinner"></span>${tr('Loading project artifact…', '正在加载项目产物…')}</div>`;
    } else if (state.error) {
      body = `<div class="gpi-preview-state error">${icon('alert', 16)}<strong>${tr('Preview unavailable', '无法预览')}</strong><span>${esc(state.error)}</span></div>`;
    } else if (state.mode === 'web' && isHtml()) {
      body = `<iframe class="gpi-preview-frame" src="${esc(previewUrl())}" sandbox="allow-scripts" referrerpolicy="no-referrer" title="${esc(tr('Preview of ', '预览：') + state.resource.label)}"></iframe>`;
    } else {
      const text = state.artifact && state.artifact.text != null ? state.artifact.text : '';
      body = `<pre class="gpi-preview-code" tabindex="0"><code>${esc(text)}</code></pre>`;
    }
    state.host.innerHTML = `
      <header class="gpi-preview-head">
        <div class="gpi-preview-file-icon" aria-hidden="true">${icon(state.mode === 'web' ? 'globe' : 'file', 16)}</div>
        <div class="gpi-preview-ident"><strong>${esc(state.resource.label)}</strong><span>${esc(state.resource.file)}</span></div>
        <button class="gpi-preview-close" type="button" data-gpi-preview-close aria-label="${tr('Close preview', '关闭预览')}" title="${tr('Close preview', '关闭预览')}">${icon('close', 15)}</button>
      </header>
      ${tabs}
      <div class="gpi-preview-body">${body}</div>`;
  }
  async function loadCode() {
    if (!state.resource || !state.projectId) return;
    const ticket = ++state.request;
    state.loading = true; state.error = ''; render();
    try {
      const api = window.EU_API || {};
      if (!api.loadPiCopilotWorkspaceFile) throw new Error(tr('The workspace file API is unavailable.', '工作区文件接口不可用。'));
      const payload = await api.loadPiCopilotWorkspaceFile(state.projectId, state.resource.file);
      if (ticket !== state.request) return;
      state.artifact = payload && payload.artifact ? payload.artifact : null;
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
    if (!safe || !project) return;
    state.resource = safe;
    state.projectId = project;
    state.artifact = null;
    state.error = '';
    state.mode = safe.kind === 'webpage' ? 'web' : 'code';
    render();
    if (state.mode === 'code') loadCode();
  }
  function close() {
    state.request += 1;
    state.resource = null; state.artifact = null; state.error = ''; state.loading = false;
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
      const mode = tab.dataset.gpiPreviewMode === 'web' ? 'web' : 'code';
      if (mode === state.mode) return;
      state.mode = mode;
      render();
      if (mode === 'code' && !state.artifact && !state.loading) loadCode();
    });
    if (!state.resource) setAsideOpen(false);
    else render();
  }

  window.EU_GUIDED_PI_PREVIEW = { mount, open, close, clearProject };
})();
