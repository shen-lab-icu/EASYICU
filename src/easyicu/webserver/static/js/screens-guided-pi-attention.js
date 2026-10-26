/* Owner: which conversations need the researcher's attention.
   The host's /api/attention feed owns the events (a plan to review, a run
   that finished, failed or stopped) and their acknowledgement per client; it
   carries only ids and codes. This file reads it, marks the project and
   conversation rows, counts open items in the tab title, and acknowledges a
   conversation's items once the researcher has it open in a visible window.
   A desktop shell that defines window.easyicuAttentionBridge receives each new
   item and the count; no other notification path is used. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }

  const POLL_MS = 20000;
  const CLIENT_KEY = 'easyicu.attention.client';
  const PUBLISHED_KEY = 'easyicu.attention.published';
  const MAX_ITEMS = 500;
  // A plan waiting for review outranks a result; a failure outranks a finish.
  const KINDS = {
    needs_review: [4, 'Needs your review', '需要你确认'],
    failed: [3, 'Run failed', '运行失败'],
    completed: [2, 'Finished', '已完成'],
    cancelled: [1, 'Stopped', '已停止'],
  };

  let clientId = '';
  let items = [];
  let after = 0;
  let count = 0;
  let timer = 0;
  let polling = false;
  let acking = false;
  let context = null;

  function storage() {
    try { return window.localStorage || null; } catch (_) { return null; }
  }
  // One persistent id per browser or desktop install: the host keeps each
  // client's acknowledgements apart.
  function client() {
    if (clientId) return clientId;
    const store = storage();
    let value = '';
    try { value = store ? String(store.getItem(CLIENT_KEY) || '') : ''; } catch (_) {}
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value)) {
      const random = window.crypto && typeof window.crypto.randomUUID === 'function'
        ? window.crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
      value = `web-${random}`.slice(0, 128);
      try { if (store) store.setItem(CLIENT_KEY, value); } catch (_) {}
    }
    clientId = value;
    return clientId;
  }

  function top(rows) {
    return rows.reduce((best, row) => (!best || (KINDS[row.kind] || [0])[0] > (KINDS[best.kind] || [0])[0] ? row : best), null);
  }
  function markHtml(rows) {
    const row = top(rows);
    if (!row || !KINDS[row.kind]) return '';
    const label = tr(KINDS[row.kind][1], KINDS[row.kind][2]);
    return `<i class="gpi-attention-mark is-${esc(row.kind)}" role="img" aria-label="${esc(label)}" title="${esc(label)}"></i>`;
  }
  function conversationMarkHtml(projectId, sessionId) {
    return markHtml(items.filter(row => row.project_id === projectId && row.conversation_id === sessionId));
  }
  function projectMarkHtml(projectId) {
    return markHtml(items.filter(row => row.project_id === projectId));
  }

  function syncTitle() {
    if (typeof document === 'undefined') return;
    const base = String(document.title || '').replace(/^\(\d+\) /, '');
    document.title = count > 0 ? `(${count}) ${base}` : base;
  }

  function published() {
    try { return Number((storage() || { getItem: () => 0 }).getItem(PUBLISHED_KEY)) || 0; } catch (_) { return 0; }
  }
  async function tellBridge(fresh) {
    const bridge = window.easyicuAttentionBridge;
    if (!bridge) return;
    let high = published();
    for (const item of fresh) {
      if (item.sequence <= high) continue;
      try { await bridge.publish({ schema_version: 'easyicu.attention/1', item }); } catch (_) { break; }
      high = item.sequence;
      try { storage() && storage().setItem(PUBLISHED_KEY, String(high)); } catch (_) {}
    }
    try { await bridge.badge({ unacknowledged_count: count }); } catch (_) {}
  }

  // The project rail renders each row's mark (screens-guided-projects.js);
  // between its renders this file keeps the marks current in place.
  function paintProjectRail() {
    if (typeof document === 'undefined' || typeof document.querySelectorAll !== 'function') return;
    document.querySelectorAll('.gd-sess[data-project-id]').forEach(row => {
      const old = row.querySelector('.gpi-attention-mark');
      const html = projectMarkHtml(row.dataset.projectId);
      if (old) old.remove();
      const time = row.querySelector('.ss-time');
      if (html && time) time.insertAdjacentHTML('beforebegin', html);
    });
  }
  function changed() {
    syncTitle();
    paintProjectRail();
    if (context && typeof context.onChange === 'function') context.onChange();
  }

  async function poll() {
    if (polling || typeof api().loadAttention !== 'function') return;
    polling = true;
    const fresh = [];
    try {
      for (let page = 0; page < 10; page += 1) {
        const payload = await api().loadAttention({ client_id: client(), after, limit: 100 });
        const rows = Array.isArray(payload && payload.items) ? payload.items : [];
        rows.forEach(row => {
          if (!row || !Number.isSafeInteger(row.sequence) || items.some(item => item.sequence === row.sequence)) return;
          items.push(row);
          fresh.push(row);
        });
        if (Number.isSafeInteger(payload && payload.next_after)) after = payload.next_after;
        if (Number.isSafeInteger(payload && payload.unacknowledged_count)) count = payload.unacknowledged_count;
        if (!(payload && payload.has_more)) break;
      }
      if (items.length > MAX_ITEMS) items = items.slice(-MAX_ITEMS);
      if (fresh.length) changed(); else syncTitle();
      await tellBridge(fresh);
      await seen();
    } catch (_) {
      // The feed is advisory; a failed read leaves the marks as they were.
    } finally {
      polling = false;
    }
  }

  // The open conversation in a visible window counts as read.
  async function seen() {
    if (acking || !context || typeof api().acknowledgeAttention !== 'function') return;
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;
    const projectId = context.projectId();
    const sessionId = context.selectedId();
    if (!projectId || !sessionId) return;
    const mine = items.filter(row => row.project_id === projectId && row.conversation_id === sessionId).slice(0, 100);
    if (!mine.length) return;
    acking = true;
    try {
      const payload = await api().acknowledgeAttention({ client_id: client(), sequences: mine.map(row => row.sequence) });
      const done = new Set(Array.isArray(payload && payload.acknowledged) ? payload.acknowledged : []);
      items = items.filter(row => !done.has(row.sequence));
      count = Math.max(0, count - done.size);
      changed();
      const bridge = window.easyicuAttentionBridge;
      if (bridge) { try { await bridge.badge({ unacknowledged_count: count }); } catch (_) {} }
    } catch (_) {
      // Left unacknowledged; the next poll tries again.
    } finally {
      acking = false;
    }
  }

  // ctx: { projectId(), selectedId(), openSession(id), onChange() }
  function bind(options) {
    context = options;
    if (!timer) {
      void poll();
      timer = setInterval(() => { void poll(); }, POLL_MS);
      if (typeof document !== 'undefined' && document.addEventListener) {
        document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') void poll(); });
      }
      const bridge = window.easyicuAttentionBridge;
      if (bridge && typeof bridge.onOpen === 'function') bridge.onOpen(item => { void open(item); });
    }
  }

  // A desktop notification opens its conversation the way search does.
  async function open(item) {
    if (!item || !context) return;
    const projectId = String(item.project_id || '');
    const sessionId = String(item.conversation_id || '');
    if (!projectId) return;
    if (projectId === context.projectId()) {
      if (sessionId) await context.openSession(sessionId);
      return;
    }
    const projectOwner = window.EasyICU.guidedPi.optional('project');
    if (projectOwner && projectOwner.syncLocation) projectOwner.syncLocation(projectId, sessionId);
    if (projectOwner && projectOwner.openListedProject) projectOwner.openListedProject(projectId);
  }

  window.EasyICU.guidedPi.declare('attention', {
    bind, poll, seen, open, conversationMarkHtml, projectMarkHtml,
    count: () => count,
  });
})();
