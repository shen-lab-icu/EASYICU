/* Owner: the desktop app's "new version available" notice.
   webserver/update_check.py owns the check, the release link and the
   per-version snooze; this file only presents its answer. Anything other than
   update_available (offline, hosted, current, snoozed, failed) shows nothing.
   The navigation rails own the place: each renders an empty
   `[data-update-slot]`, and this file fills every slot with an "Update" rail
   item whose popover menu holds the version, a note, Remind me later and
   Download. Nothing floats over the page until the item is opened. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function icon(name, size) { return typeof window.icon === 'function' ? window.icon(name, size, 1.6) : ''; }

  const SLOT = '[data-update-slot]';
  let shown = null;
  let unopened = false;
  let observer = null;

  function panelHtml() {
    const version = String(shown.latest_version || '');
    return `<strong>${esc(tr(`EasyICU ${version} is available`, `EasyICU ${version} 可以更新了`))}</strong>`
      + `<p>${esc(tr('Download the installer and install it over this version; projects and settings are kept.', '下载安装包覆盖安装即可，项目和设置会保留。'))}</p>`
      + (unopened ? `<p class="eu-update-link">${esc(tr('The browser did not open. Copy this address:', '没能打开浏览器，请复制这个地址：'))} <span>${esc(String(shown.release_url || ''))}</span></p>` : '')
      + `<footer><button type="button" class="btn sm" data-update-action="later">${esc(tr('Remind me later', '稍后提醒'))}</button>`
      + `<button type="button" class="btn sm primary" data-update-action="download">${esc(tr('Download', '下载'))}</button></footer>`;
  }

  function itemHtml() {
    const title = tr(`EasyICU ${shown.latest_version} is available`, `EasyICU ${shown.latest_version} 可以更新了`);
    return `<details class="eu-update" data-popover-menu><summary title="${esc(title)}" aria-label="${esc(title)}">${icon('download', 18)}<span>${esc(tr('Update', '更新'))}</span><i class="eu-update-dot" aria-hidden="true"></i></summary>`
      + `<div class="eu-update-panel" role="group" aria-label="${esc(title)}">${panelHtml()}</div></details>`;
  }

  // The panel is fixed to the window beside its item, so a rail that clips
  // its overflow (a collapsed project rail) cannot cut it off.
  function place(details) {
    const panel = details.querySelector('.eu-update-panel');
    const summary = details.querySelector('summary');
    if (!panel || !summary || typeof summary.getBoundingClientRect !== 'function') return;
    const rect = summary.getBoundingClientRect();
    const width = Math.min(300, window.innerWidth - 16);
    panel.style.left = `${Math.max(8, Math.min(rect.right + 8, window.innerWidth - width - 8))}px`;
    panel.style.bottom = `${Math.max(8, window.innerHeight - rect.bottom)}px`;
  }

  function fill(root) {
    if (!shown || !root || typeof root.querySelectorAll !== 'function') return;
    root.querySelectorAll(SLOT).forEach(slot => {
      if (slot.querySelector('.eu-update')) return;
      slot.innerHTML = itemHtml();
      const details = slot.querySelector('.eu-update');
      details.addEventListener('toggle', () => { if (details.open) place(details); });
      details.addEventListener('click', event => {
        const action = event.target.closest && event.target.closest('[data-update-action]');
        if (action) void act(action.dataset.updateAction, details);
      });
    });
  }

  function clear() {
    shown = null;
    if (observer) observer.disconnect();
    observer = null;
    document.querySelectorAll(SLOT).forEach(slot => { slot.innerHTML = ''; });
  }

  async function act(action, details) {
    if (!shown) return;
    if (action === 'later') {
      const version = shown.latest_version;
      clear();
      try { await api().snoozeUpdate(version); } catch (_) { /* hidden for this launch either way */ }
      return;
    }
    if (action === 'download') {
      let opened = false;
      try { opened = (await api().openUpdatePage()).opened === true; } catch (_) {}
      if (opened) { clear(); return; }
      unopened = true;
      document.querySelectorAll('.eu-update-panel').forEach(panel => { panel.innerHTML = panelHtml(); });
      if (details && details.open) place(details);
    }
  }

  async function check() {
    if (typeof api().loadUpdateCheck !== 'function') return;
    let payload;
    try { payload = await api().loadUpdateCheck(); } catch (_) { return; }
    if (!payload || payload.status !== 'update_available' || !payload.latest_version) return;
    shown = payload;
    fill(document);
    // Screens rebuild their rails on every route change; refill new slots.
    if (!observer && typeof window.MutationObserver === 'function') {
      observer = new window.MutationObserver(() => fill(document));
      observer.observe(document.body, { childList: true, subtree: true });
    }
  }

  window.EU_UPDATE_NOTICE = { check, fill };
  if (typeof window.addEventListener === 'function') {
    window.addEventListener('load', () => { setTimeout(() => { void check(); }, 2500); });
  }
})();
