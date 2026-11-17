/* Owner: Guided startup cover widget. */
/* Guided Copilot startup transaction.
   Owns only the initial full-shell cover so project discovery, model status,
   saved conversations, and StudyContext can settle behind one stable frame. */
(function () {
  'use strict';

  let active = false;
  let watchdog = 0;
  // A cold server can be slow, but a startup that has not settled by then is
  // a product stop with a typed code, never an endless cover (pilot1 11-17).
  const STARTUP_LIMIT_MS = 120000;

  function begin() {
    active = true;
    clearTimeout(watchdog);
    watchdog = setTimeout(() => { if (active) fail('guided_startup_timeout'); }, STARTUP_LIMIT_MS);
  }

  function isActive() {
    return active;
  }

  function markup(t) {
    if (!active) return '';
    return `
      <div class="gd-startup-shield" data-guided-startup-shield role="status" aria-live="polite">
        <div class="gd-startup-card">
          <span>${t('Restoring your research workspace…', '正在恢复研究工作区…')}</span>
          <div class="gd-startup-progress" aria-hidden="true"><span></span></div>
        </div>
      </div>`;
  }

  function lang() {
    return (window.EU_I18N && window.EU_I18N.lang && window.EU_I18N.lang() === 'zh') ||
      String(document.documentElement.lang || '').startsWith('zh');
  }

  // Replace the cover with a readable, typed stop. The startup's own state is
  // not trusted after a failure, so the only offered action is a reload.
  function fail(code, error) {
    active = false;
    clearTimeout(watchdog);
    const name = error && error.name ? String(error.name).slice(0, 40) : '';
    console.error('[guided-startup]', code, error || '');
    window.EU_GUIDED_STARTUP_FAILURE = Object.freeze({ code, error: name });
    const zh = lang();
    document.querySelectorAll('[data-guided-startup-shield]').forEach((shield) => {
      shield.setAttribute('role', 'alert');
      shield.setAttribute('data-guided-startup-failed', code);
      const card = shield.querySelector('.gd-startup-card') || shield;
      card.textContent = '';
      const text = document.createElement('span');
      text.textContent = zh
        ? '研究工作区未能完成加载（' + code + (name ? ' · ' + name : '') + '）。请重新加载页面。'
        : 'The research workspace did not finish loading (' + code + (name ? ' · ' + name : '') + '). Reload the page.';
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'gd-startup-reload';
      button.textContent = zh ? '重新加载' : 'Reload';
      button.addEventListener('click', () => window.location.reload());
      card.append(text, button);
    });
  }

  // Run the first render's synchronous startup. A throw there used to leave
  // the cover in place with no request in flight; it now becomes a typed stop.
  // Outside the initial startup the error propagates as before.
  function guard(fn) {
    try {
      return fn();
    } catch (error) {
      if (!active) throw error;
      fail('guided_startup_render_failed', error);
      return undefined;
    }
  }

  function finish(root) {
    active = false;
    clearTimeout(watchdog);
    // The app shell may re-render while startup requests are in flight. The
    // callback's original root can therefore be detached; clear the currently
    // connected shell as well so a stale root cannot leave the cover behind.
    const scopes = [root, document].filter(Boolean);
    scopes.forEach((scope) => {
      scope.querySelectorAll('.gd-main.gd-startup-active').forEach((main) => {
        main.classList.remove('gd-startup-active');
        main.setAttribute('aria-busy', 'false');
      });
      scope.querySelectorAll('[data-guided-startup-shield]').forEach((shield) => shield.remove());
    });
  }

  window.EU_GUIDED_STARTUP = Object.freeze({ begin, fail, finish, guard, isActive, markup });
})();
