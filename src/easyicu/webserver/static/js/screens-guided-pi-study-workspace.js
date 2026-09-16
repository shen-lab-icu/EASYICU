/* Presentation state for a study: retained conversation and an explicit,
   version-bound artifact reference in the composer. No execution authority. */
(function () {
  'use strict';
  function create({ tr, esc }) {
    const expanded = new Map();
    let reference = null;
    function capture(host) {
      const history = host && host.querySelector('[data-gpi-study-history]');
      if (history) expanded.set(history.dataset.gpiStudyHistory, history.open);
    }
    function history(messages, key, active) {
      if (!messages) return '';
      const open = active || expanded.get(key) === true;
      return `<details class="gpi-study-history" data-gpi-study-history="${esc(key)}"${open ? ' open' : ''}>
        <summary><strong>${tr('Conversation & research process', '对话与研究过程')}</strong><span>${tr('Includes earlier attempts and retries', '包含先前尝试与重试记录')}</span></summary>
        <div class="gpi-study-history-body">${messages}</div></details>`;
    }
    function canReference(resource) {
      return Boolean(resource && /^research_(artifact|report|document)$/.test(resource.kind)
        && /^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(resource.run_id || '')
        && /^[A-Za-z0-9_.-]{1,160}$/.test(resource.artifact || '')
        && /^[a-f0-9]{64}$/.test(resource.sha256 || ''));
    }
    function setReference(resource, projectId, sessionId) {
      if (!projectId || !sessionId || !canReference(resource)) return false;
      reference = { projectId, sessionId, resource: { kind: resource.kind,
        run_id: resource.run_id, artifact: resource.artifact, sha256: resource.sha256,
        label: String(resource.label || resource.artifact).slice(0, 160) } };
      return true;
    }
    function current(projectId, sessionId) {
      return reference && reference.projectId === projectId && reference.sessionId === sessionId ? reference : null;
    }
    function renderReference(projectId, sessionId) {
      const selected = current(projectId, sessionId);
      if (!selected) return '';
      return `<div class="gpi-composer-reference" role="status"><div><span>${tr('Referencing', '已引用')}</span><strong>${esc(selected.resource.label)}</strong><small>${tr('This version · add your question below', '当前版本 · 在下方填写你的问题')}</small></div>
        <button type="button" data-gpi-reference-remove aria-label="${tr('Remove reference', '移除引用')}">×</button></div>`;
    }
    function decorateMessage(text, projectId, sessionId) {
      const selected = current(projectId, sessionId);
      if (!selected || !String(text || '').trim()) return text;
      return `${text}\n\n${tr('Referenced project artifact (existing result):', '引用的项目资料（已有成果）：')}\n${JSON.stringify({ project_id: projectId, ...selected.resource })}`;
    }
    function messageView(row, projectId) {
      if (!row || row.role !== 'user') return row;
      const text = String(row.text || '');
      const match = /\n\n(?:Referenced project artifact \(existing result\):|引用的项目资料（已有成果）：)\n(\{[^\n]+\})$/.exec(text);
      if (!match) return row;
      try {
        const resource = JSON.parse(match[1]);
        if (resource.project_id !== projectId || !canReference(resource)) return row;
        return { ...row, text: text.slice(0, match.index), resources: [...(Array.isArray(row.resources) ? row.resources : []), resource] };
      } catch (_) { return row; }
    }
    function consume(projectId, sessionId) { if (current(projectId, sessionId)) reference = null; }
    function removeReference() { reference = null; }
    return { capture, history, messageView, hasReference: (projectId, sessionId) => Boolean(current(projectId, sessionId)), canReference, setReference, renderReference, decorateMessage, consume, removeReference };
  }
  window.EasyICU.guidedPi.declare('studyWorkspace', { create });
})();
