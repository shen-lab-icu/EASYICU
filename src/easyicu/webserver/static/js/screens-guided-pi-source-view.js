/* Owner: digest-pinned source and public operation workbench. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  const icon = window.icon;
  const state = {
    host: null, projectId: '', runId: '', tabs: [], activeId: '',
    request: 0, openResource: null,
  };
  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function clean(value, limit) { return String(value || '').trim().slice(0, limit || 240); }

  function validDescriptor(button) {
    if (!button || !button.dataset) return null;
    const evidenceId = clean(button.dataset.sourceEvidenceId, 160);
    const sha256 = clean(button.dataset.sourceSha256, 64).toLowerCase();
    if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) return null;
    const focusStart = Number(button.dataset.sourceFocusStart || 0);
    const focusEnd = Number(button.dataset.sourceFocusEnd || 0);
    return {
      evidenceId, sha256,
      title: clean(button.dataset.sourceTitle || tr('Figure source', '图件源代码')),
      output: clean(button.dataset.sourceOutput),
      step: clean(button.dataset.sourceStep, 160),
      producer: clean(button.dataset.sourceProducer, 160),
      generation: clean(button.dataset.sourceGeneration, 160),
      displayName: clean(button.dataset.sourceDisplayName || 'analysis.py', 160),
      language: clean(button.dataset.sourceLanguage || 'text', 40),
      focusStart: Number.isInteger(focusStart) && focusStart > 0 ? focusStart : 0,
      focusEnd: Number.isInteger(focusEnd) && focusEnd >= focusStart ? focusEnd : 0,
    };
  }

  function validOperation(button) {
    if (!button || !button.dataset || !button.dataset.gpiOperation) return null;
    try {
      const raw = JSON.parse(decodeURIComponent(button.dataset.gpiOperation));
      const resources = Array.isArray(raw.resources) ? raw.resources.slice(0, 12).map(resource => ({
        kind: clean(resource && resource.kind, 80),
        artifact: clean(resource && resource.artifact, 240),
        filename: clean(resource && resource.filename, 240),
        title: clean(resource && (resource.title || resource.label), 240),
        label: clean(resource && resource.label, 240),
        run_id: clean(resource && resource.run_id, 160),
        sha256: clean(resource && resource.sha256, 64),
      })) : [];
      return {
        id: clean(raw.id || `${raw.kind || 'step'}-${raw.title || ''}`, 240),
        title: clean(raw.title || tr('Execution step', '执行步骤')),
        kind: clean(raw.kind || 'operation', 80),
        icon: clean(raw.icon || 'play', 40),
        status: clean(raw.status || 'complete', 40),
        detail: clean(raw.detail, 800),
        duration: clean(raw.duration, 80),
        toolName: clean(raw.toolName, 160),
        resources,
      };
    } catch (_) { return null; }
  }

  function setOpen(open) {
    if (!state.host) return;
    state.host.hidden = !open;
    const main = state.host.closest('.gd-main');
    if (main) main.classList.toggle('gpi-source-open', Boolean(open));
  }

  function highlightedLine(value, language) {
    const line = String(value || '');
    const isCode = ['python', 'javascript', 'typescript', 'julia', 'r', 'sql', 'shell'].includes(language);
    if (!isCode) return esc(line) || '&nbsp;';
    const pattern = language === 'python' || language === 'r' || language === 'shell'
      ? /(#.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(def|class|return|if|else|elif|for|while|in|import|from|as|try|except|with|lambda|True|False|None|and|or|not|print)\b|\b(\d+(?:\.\d+)?)\b/g
      : /(\/\/.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\b(const|let|var|function|return|if|else|for|while|class|new|async|await|true|false|null|undefined|SELECT|FROM|WHERE|JOIN|GROUP|ORDER|BY|AS)\b|\b(\d+(?:\.\d+)?)\b/gi;
    let output = '';
    let cursor = 0;
    for (const match of line.matchAll(pattern)) {
      output += esc(line.slice(cursor, match.index));
      const token = match[0];
      const kind = match[1] ? 'comment' : match[2] ? 'string' : match[3] ? 'keyword' : 'number';
      output += `<span class="tok-${kind}">${esc(token)}</span>`;
      cursor = Number(match.index) + token.length;
    }
    output += esc(line.slice(cursor));
    return output || '&nbsp;';
  }

  function sourceWindow(tab) {
    const payload = tab.payload || {};
    const lines = String(payload.text || '').replace(/\r\n?/g, '\n').split('\n');
    if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
    const start = !tab.showFull && tab.descriptor.focusStart ? tab.descriptor.focusStart : 1;
    const end = !tab.showFull && tab.descriptor.focusEnd ? Math.min(lines.length, tab.descriptor.focusEnd) : lines.length;
    return { lines: lines.slice(start - 1, end), start, end, total: lines.length };
  }
  function codeRows(tab) {
    const payload = tab.payload || {};
    const windowed = sourceWindow(tab);
    return windowed.lines.map((line, index) => `<span class="gpi-source-line"><i>${windowed.start + index}</i><code>${highlightedLine(line, payload.language)}</code></span>`).join('');
  }
  function tabById(id) { return state.tabs.find(tab => tab.id === id) || null; }
  function activeTab() { return tabById(state.activeId) || state.tabs[0] || null; }

  function renderTabs() {
    return `<div class="gpi-source-tabbar"><div class="gpi-source-tabs" role="tablist">${state.tabs.map(tab => `
      <button type="button" class="gpi-source-tab ${tab.id === state.activeId ? 'active' : ''}" role="tab" aria-selected="${tab.id === state.activeId}" data-gpi-source-tab="${esc(tab.id)}">
        ${icon(tab.kind === 'operation' ? tab.operation.icon : 'file', 14)}<span title="${esc(tab.title)}">${esc(tab.title)}</span><i data-gpi-source-tab-close="${esc(tab.id)}" aria-label="${esc(tr('Close tab', '关闭标签'))}">${icon('close', 12)}</i>
      </button>`).join('')}</div><button type="button" class="gpi-source-panel-close" data-gpi-source-close aria-label="${esc(tr('Close workbench', '关闭工作台'))}" title="${esc(tr('Close workbench', '关闭工作台'))}">${icon('close', 15)}</button></div>`;
  }

  function renderCode(tab) {
    const descriptor = tab.descriptor;
    const payload = tab.payload || {};
    if (tab.loading) return `<div class="gpi-source-loading" role="status">${esc(tr('Loading registered source…', '正在读取已登记源代码…'))}</div>`;
    if (tab.error) return `<div class="gpi-source-error" role="alert"><strong>${esc(tr('Source unavailable', '源代码暂不可用'))}</strong><p>${esc(tab.error)}</p></div>`;
    const language = String(payload.language || descriptor.language || 'text').toUpperCase();
    const meta = [payload.producer || descriptor.producer, payload.generation_mode || descriptor.generation]
      .map(value => clean(value).replace(/_/g, ' ')).filter(Boolean).join(' · ');
    const windowed = sourceWindow(tab);
    const focused = windowed.total > windowed.lines.length;
    return `<div class="gpi-source-document">
      <header class="gpi-source-title"><h2>${esc(tab.title)}</h2><div><span>${esc(meta || 'Research Agent')}</span><b>${esc(language)}</b></div></header>
      <section class="gpi-source-block">
        <div class="gpi-source-block-head"><strong>Input</strong><button type="button" data-gpi-source-copy="input">${icon('copy', 14)} ${esc(tr('Copy', '复制'))}</button></div>
        <p class="gpi-source-input">${esc(descriptor.step ? `${descriptor.step} → ${descriptor.output || tab.title}` : (descriptor.output || tab.title))}</p>
        <div class="gpi-source-cell-meta"><span>${esc(payload.display_name || descriptor.displayName)}</span><div><b>${focused ? `${windowed.start}–${windowed.end} / ${windowed.total}` : language}</b>${focused || tab.showFull ? `<button type="button" data-gpi-source-full>${esc(tab.showFull ? tr('Focus', '定位') : tr('Full source', '完整源代码'))}</button>` : ''}<button type="button" data-gpi-source-copy="code">${icon('copy', 13)} ${esc(tr('Copy code', '复制代码'))}</button></div></div>
        <pre class="gpi-source-code" tabindex="0">${codeRows(tab)}</pre>
      </section>
      <section class="gpi-source-block is-output">
        <div class="gpi-source-block-head"><strong>Output</strong><button type="button" data-gpi-source-copy="output">${icon('copy', 14)} ${esc(tr('Copy', '复制'))}</button></div>
        <div class="gpi-source-cell-meta"><span>${esc(descriptor.output || tr('Registered figure', '已登记图件'))}</span><b>${esc(tr('COMPLETE', '已完成'))}</b></div>
        <pre class="gpi-source-output">${esc(descriptor.output || tab.title)}</pre>
      </section>
    </div>`;
  }

  function renderOperation(tab) {
    const operation = tab.operation;
    const status = operation.status === 'error' || operation.status === 'failed' || operation.status === 'cancelled'
      ? tr('Needs attention', '需要处理')
      : operation.status === 'running' ? tr('Running', '运行中') : tr('Complete', '已完成');
    const category = {
      tool: tr('EasyICU tool', 'EasyICU 工具'),
      pipeline: tr('Research workflow', '科研流程'),
      assistant: tr('Model response', '模型响应'),
      retry: tr('Retry', '自动重试'),
      compaction: tr('Context update', '上下文整理'),
      submitted: tr('Request', '研究请求'),
    }[operation.kind] || tr('Execution step', '执行步骤');
    const outputs = operation.resources.length
      ? `<div class="gpi-operation-results">${operation.resources.map((resource, index) => `<button type="button" data-gpi-operation-resource="${index}">${icon(resource.kind === 'figure' ? 'viz' : 'file', 14)}<span>${esc(resource.title || resource.label || resource.filename || resource.artifact || tr('Result', '结果'))}</span>${icon('arrow', 12)}</button>`).join('')}</div>`
      : `<div class="gpi-operation-empty">${icon(['error', 'failed', 'cancelled'].includes(operation.status) ? 'alert' : 'check', 17)}<div><strong>${esc(tr('Step recorded', '步骤已记录'))}</strong><span>${esc(tr('This step updated the run record without creating a separate file.', '这一步更新了运行记录，没有生成独立文件。'))}</span></div></div>`;
    return `<div class="gpi-source-document gpi-operation-document">
      <header class="gpi-source-title gpi-operation-title"><span class="gpi-operation-title-icon" aria-hidden="true">${icon(operation.icon, 18)}</span><div><h2>${esc(operation.title)}</h2><p><span>${esc(category)}</span><b class="status-${esc(operation.status)}">${esc(status)}</b>${operation.duration ? `<span>${icon('clock', 12)}${esc(operation.duration)}</span>` : ''}</p></div></header>
      <section class="gpi-source-block">
        <div class="gpi-source-block-head"><strong>Input</strong></div>
        <dl class="gpi-operation-facts"><div><dt>${esc(tr('Step type', '步骤类型'))}</dt><dd>${esc(category)}</dd></div><div><dt>${esc(tr('Action', '动作'))}</dt><dd>${esc(operation.title)}</dd></div></dl>
      </section>
      <section class="gpi-source-block is-output">
        <div class="gpi-source-block-head"><strong>Output</strong></div>
        ${operation.detail ? `<p class="gpi-operation-detail">${esc(operation.detail)}</p>` : ''}${outputs}
      </section>
    </div>`;
  }

  function render() {
    if (!state.host) return;
    const tab = activeTab();
    if (!tab) { state.host.replaceChildren(); setOpen(false); return; }
    state.host.innerHTML = `${renderTabs()}${tab.kind === 'operation' ? renderOperation(tab) : renderCode(tab)}`;
  }

  async function copyText(value, button) {
    const text = String(value || '');
    if (!text) return;
    if (navigator.clipboard && navigator.clipboard.writeText) await navigator.clipboard.writeText(text);
    else {
      const area = document.createElement('textarea');
      area.value = text; area.setAttribute('readonly', ''); area.style.position = 'fixed'; area.style.opacity = '0';
      document.body.appendChild(area); area.select(); document.execCommand('copy'); area.remove();
    }
    if (button) {
      const previous = button.innerHTML;
      button.textContent = tr('Copied', '已复制');
      window.setTimeout(() => { if (button.isConnected) button.innerHTML = previous; }, 1200);
    }
  }

  async function open(button, projectId, runId) {
    const descriptor = validDescriptor(button);
    const project = clean(projectId, 240);
    const run = clean(runId, 240);
    if (!descriptor || !project || !run) return false;
    // The reader stays open: a figure and its producing code are read side by
    // side. The desktop layout owner decides how the two panes share width.
    if (state.projectId && state.projectId !== project) state.tabs = [];
    state.projectId = project; state.runId = run;
    const id = `code:${descriptor.evidenceId}:${descriptor.sha256}`;
    let tab = tabById(id);
    if (!tab) {
      tab = { id, kind: 'code', title: descriptor.title, descriptor, payload: null, loading: true, error: '', showFull: false };
      state.tabs.push(tab);
    }
    state.activeId = id; setOpen(true); render();
    if (tab.payload && !tab.error) return true;
    const ticket = ++state.request;
    try {
      const api = window.EU_API || {};
      if (typeof api.loadPiCopilotResearchEvidence !== 'function') throw new Error(tr('The evidence API is unavailable.', '证据接口不可用。'));
      const response = await api.loadPiCopilotResearchEvidence(project, run, descriptor.evidenceId, descriptor.sha256);
      if (ticket !== state.request || !tabById(id)) return true;
      const payload = response && response.payload;
      if (!payload || payload.renderer !== 'code' || payload.previewable !== true || typeof payload.text !== 'string') throw new Error(tr('This registered source cannot be displayed safely.', '这份已登记源代码无法安全展示。'));
      tab.payload = payload;
    } catch (error) {
      if (ticket !== state.request || !tabById(id)) return true;
      tab.error = String(error && (error.message || error.code) || error);
    } finally {
      if (tabById(id)) { tab.loading = false; render(); }
    }
    return true;
  }

  function openOperation(button, projectId, runId, openResource) {
    const operation = validOperation(button);
    if (!operation) return false;
    const project = clean(projectId, 240);
    if (state.projectId && project && state.projectId !== project) state.tabs = [];
    state.projectId = project || state.projectId;
    state.runId = clean(runId, 240) || state.runId;
    state.openResource = typeof openResource === 'function' ? openResource : state.openResource;
    const id = `operation:${operation.id}`;
    const existing = tabById(id);
    if (existing) existing.operation = operation;
    else state.tabs.push({ id, kind: 'operation', title: operation.title, operation });
    state.activeId = id; setOpen(true); render();
    return true;
  }

  function closeTab(id) {
    const index = state.tabs.findIndex(tab => tab.id === id);
    if (index < 0) return;
    state.tabs.splice(index, 1);
    if (state.activeId === id) state.activeId = (state.tabs[Math.max(0, index - 1)] || state.tabs[0] || {}).id || '';
    render();
  }
  function close() {
    state.request += 1;
    state.projectId = ''; state.runId = ''; state.tabs = []; state.activeId = ''; state.openResource = null;
    setOpen(false);
    if (state.host) state.host.replaceChildren();
  }

  function mount(host) {
    if (!host || state.host === host) return;
    state.host = host;
    host.addEventListener('click', event => {
      const tabClose = event.target.closest('[data-gpi-source-tab-close]');
      if (tabClose) { event.stopPropagation(); closeTab(tabClose.dataset.gpiSourceTabClose); return; }
      const tabButton = event.target.closest('[data-gpi-source-tab]');
      if (tabButton) { state.activeId = tabButton.dataset.gpiSourceTab; render(); return; }
      if (event.target.closest('[data-gpi-source-close]')) { close(); return; }
      const tab = activeTab();
      if (!tab) return;
      if (event.target.closest('[data-gpi-source-full]') && tab.kind === 'code') { tab.showFull = !tab.showFull; render(); return; }
      const resource = event.target.closest('[data-gpi-operation-resource]');
      if (resource && tab.kind === 'operation' && state.openResource) {
        const descriptor = tab.operation.resources[Number(resource.dataset.gpiOperationResource)];
        if (descriptor) state.openResource(descriptor);
        return;
      }
      const copy = event.target.closest('[data-gpi-source-copy]');
      if (!copy || tab.kind !== 'code') return;
      const value = copy.dataset.gpiSourceCopy === 'input'
        ? `${tab.descriptor.step || ''} → ${tab.descriptor.output || tab.title}`
        : copy.dataset.gpiSourceCopy === 'output' ? (tab.descriptor.output || tab.title) : (tab.payload && tab.payload.text || '');
      void copyText(value, copy);
    });
    host.addEventListener('dblclick', event => {
      const tab = activeTab();
      if (tab && tab.kind === 'code' && event.target.closest('.gpi-source-code')) void copyText(tab.payload && tab.payload.text || '', null);
    });
  }

  window.EasyICU.guidedPi.declare('sourceView', { mount, open, openOperation, close });
})();
