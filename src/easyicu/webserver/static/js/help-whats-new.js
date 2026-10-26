/* Owner: what is new in this version, shown once after an update and on request.
   The host's packaged release notes (webserver/release_notes.py) own the
   entries and which are newer than a given version. This file keeps, per
   browser, the version the researcher last acknowledged: on the first visit
   it records the current version without showing anything; after an update
   it shows the newer entries once. Settings' Release notes opens all of them. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function icon(name, size) { return typeof window.icon === 'function' ? window.icon(name, size, 1.6) : ''; }

  const SEEN_KEY = 'easyicu.whatsNew.seen';
  const VERSION = /^\d+\.\d+\.\d+$/;
  let dialog = null;

  function seen() {
    try {
      const value = String(window.localStorage.getItem(SEEN_KEY) || '');
      return VERSION.test(value) ? value : '';
    } catch (_) { return ''; }
  }
  function remember(version) {
    if (!VERSION.test(String(version || ''))) return;
    try { window.localStorage.setItem(SEEN_KEY, String(version)); } catch (_) {}
  }

  function entriesHtml(entries) {
    return entries.map(entry => {
      const changes = (Array.isArray(entry.changes) ? entry.changes : [])
        .map(change => `<li>${esc(tr(change.en || change.zh || '', change.zh || change.en || ''))}</li>`).join('');
      return `<section><h3>${esc(entry.version || '')}<small>${esc(entry.date || '')}</small></h3><ul>${changes}</ul></section>`;
    }).join('');
  }
  function show(title, body, latest) {
    if (!dialog) {
      dialog = document.createElement('dialog');
      dialog.className = 'eu-whats-new';
      dialog.addEventListener('click', event => {
        if (event.target === dialog || event.target.closest('[data-whats-new-close]')) {
          remember(dialog.dataset.latest);
          dialog.close();
        }
      });
      dialog.addEventListener('cancel', () => remember(dialog.dataset.latest));
      document.body.appendChild(dialog);
    }
    dialog.dataset.latest = latest || '';
    dialog.innerHTML = `<header><h2>${esc(title)}</h2><button type="button" data-whats-new-close aria-label="${esc(tr('Close', '关闭'))}">${icon('x', 15)}</button></header><div class="eu-whats-new-body">${body}</div><footer><button type="button" class="btn sm primary" data-whats-new-close>${esc(tr('Got it', '知道了'))}</button></footer>`;
    if (!dialog.open) dialog.showModal();
  }

  // After an update: the entries newer than the acknowledged version, once.
  async function check() {
    if (typeof api().loadWhatsNew !== 'function') return;
    const last = seen();
    let payload;
    try { payload = await api().loadWhatsNew(last); } catch (_) { return; }
    const latest = String(payload && payload.latest_version || '');
    if (!last) { remember(latest); return; }
    const entries = Array.isArray(payload && payload.entries) ? payload.entries : [];
    if (!payload.show_after_update || !entries.length) return;
    show(tr(`EasyICU is now ${latest}`, `EasyICU 已更新到 ${latest}`), entriesHtml(entries), latest);
  }
  // On request: every entry.
  async function open() {
    if (typeof api().loadWhatsNew !== 'function') return;
    let payload;
    try { payload = await api().loadWhatsNew(''); } catch (_) {
      show(tr('Release notes', '版本说明'), `<p role="alert">${esc(tr('Release notes are unavailable right now.', '暂时读不到版本说明。'))}</p>`, '');
      return;
    }
    const entries = Array.isArray(payload && payload.entries) ? payload.entries : [];
    show(tr('Release notes', '版本说明'), entriesHtml(entries), String(payload && payload.latest_version || ''));
  }

  window.EU_WHATS_NEW = { check, open, entriesHtml };
  if (typeof window.addEventListener === 'function') {
    window.addEventListener('load', () => { setTimeout(() => { void check(); }, 1200); });
  }
})();
