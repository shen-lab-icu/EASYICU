/* Owner: Guided Pi deferred model connection. */
/* A researcher can enter the workspace before connecting a model: projects,
   data sources and skills need none. The step that first needs a model asks
   for the connection in a dialog and resumes once it is ready. The connection
   form, verification and account login stay with the provider owners; this
   module only decides when to ask and what to resume. Provider authority
   stays on the server (ai_optin / provider_gate): deferring grants nothing. */
(function () {
  'use strict';

  const STORAGE_KEY = 'easyicu.copilot.connectionDeferred';

  // A per-browser convenience only; private windows and blocked storage fall
  // back to showing the connection page first.
  function readDeferred() {
    try { return window.localStorage.getItem(STORAGE_KEY) === '1'; } catch (_) { return false; }
  }
  function writeDeferred() {
    try { window.localStorage.setItem(STORAGE_KEY, '1'); } catch (_) { /* storage unavailable */ }
  }

  const REASONS = Object.freeze({
    start_conversation: [
      'Research conversations need a model. Once connected, the conversation starts.',
      '研究对话需要模型。连接后会直接开始对话。',
    ],
    send_message: [
      'Sending this message needs a model. Once connected, it is sent.',
      '发送这条消息需要模型。连接后会接着发送。',
    ],
    manual: [
      'Projects, data sources and skills work without a model; conversations and research need one.',
      '浏览项目、数据和技能不需要模型；对话和研究需要。',
    ],
  });

  function create(options) {
    const { state, tr, esc, render, connectionReady, connectionConfigured, finishProviderSetup, body } = options;
    state.connectionDeferred = readDeferred();
    state.connectPrompt = null;

    // The full connection page shows on first entry, until the researcher
    // chooses to explore first, and whenever they open connection settings.
    function pageRequired() {
      return state.showSetup || (!connectionReady() && !state.connectionDeferred);
    }

    function deferConnection() {
      state.connectionDeferred = true;
      writeDeferred();
      state.showSetup = false;
      state.error = '';
      render();
    }

    // Returns true when the model is ready; otherwise opens the dialog and
    // keeps `resume` to run after the researcher connects.
    function ask(reason, resume) {
      if (connectionReady()) return true;
      state.connectPrompt = {
        reason: REASONS[reason] ? reason : 'manual',
        resume: typeof resume === 'function' ? resume : null,
      };
      state.error = '';
      render();
      return false;
    }

    function dismiss() {
      if (!state.connectPrompt || state.setupSaving || state.codexBusy) return;
      state.connectPrompt = null;
      state.error = '';
      render();
    }

    async function proceed() {
      const pending = state.connectPrompt;
      if (!pending || !connectionConfigured()) return;
      await finishProviderSetup();
      if (!connectionReady() || state.connectPrompt !== pending) return;
      state.connectPrompt = null;
      render();
      if (pending.resume) await pending.resume();
    }

    function dialogHtml() {
      const pending = state.connectPrompt;
      if (!pending) return '';
      const manual = pending.reason === 'manual';
      const [reasonEn, reasonZh] = REASONS[pending.reason];
      const ready = connectionConfigured();
      const busy = state.setupSaving || state.codexBusy;
      return `
        <div class="gpi-connect-backdrop" data-gpi-connect-backdrop>
          <section class="gpi-provider-shell gpi-connect-dialog" role="dialog" aria-modal="true" aria-labelledby="gpi-connect-title" data-gpi-connect-dialog tabindex="-1">
            <header class="gpi-connect-head">
              <div>
                <h2 id="gpi-connect-title">${manual ? tr('Connect a model', '连接模型') : tr('Connect a model to continue', '连接模型后继续')}</h2>
                <p>${esc(tr(reasonEn, reasonZh))}</p>
              </div>
              <button class="gpi-connect-close" type="button" data-gpi-connect-dismiss aria-label="${tr('Close', '关闭')}" ${busy ? 'disabled' : ''}>×</button>
            </header>
            ${body()}
            <div class="gpi-setup-actions">
              ${ready
                ? `<button class="btn primary" type="button" data-gpi-connect-continue ${busy ? 'disabled' : ''}>${manual ? tr('Done', '完成') : tr('Continue', '继续')}</button>`
                : `<button class="gpi-link" type="button" data-gpi-connect-dismiss ${busy ? 'disabled' : ''}>${tr('Not now', '暂不连接')}</button>`}
            </div>
          </section>
        </div>`;
    }

    // Called after each repaint: move focus into a newly opened dialog.
    function focusDialog(host) {
      const dialog = host && host.querySelector('[data-gpi-connect-dialog]');
      if (dialog && !dialog.contains(document.activeElement)) dialog.focus();
    }

    // Delegated events from the screen's host; returns true when handled.
    function handleClick(event) {
      if (event.target.closest('[data-gpi-connect-later]')) { deferConnection(); return true; }
      if (event.target.closest('[data-gpi-connect-open]')) { ask('manual'); return true; }
      if (event.target.closest('[data-gpi-connect-continue]')) { void proceed(); return true; }
      if (event.target.closest('[data-gpi-connect-dismiss]')) { dismiss(); return true; }
      if (event.target.matches('[data-gpi-connect-backdrop]')) { dismiss(); return true; }
      return false;
    }
    function handleKeydown(event) {
      if (event.key !== 'Escape' || !state.connectPrompt) return false;
      event.preventDefault();
      dismiss();
      return true;
    }

    return Object.freeze({ pageRequired, deferConnection, ask, dismiss, proceed, dialogHtml, focusDialog, handleClick, handleKeydown });
  }

  window.EasyICU.guidedPi.declare('connect', { create });
})();
