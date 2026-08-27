/* Guided Copilot startup transaction.
   Owns only the initial full-shell cover so project discovery, model status,
   saved conversations, and StudyContext can settle behind one stable frame. */
(function () {
  'use strict';

  let active = false;

  function begin() {
    active = true;
  }

  function isActive() {
    return active;
  }

  function markup(t) {
    if (!active) return '';
    return `
      <div class="gd-startup-shield" data-guided-startup-shield role="status" aria-live="polite">
        <div class="gpi-activate gpi-restoring gd-startup-card">
          <div class="gpi-kicker">EASYICU COPILOT · ${t('RESTORING PROJECT', '正在恢复项目')}</div>
          <h2>${t('Restoring your current research', '正在恢复当前研究')}</h2>
          <p>${t(
            'EasyICU is loading the saved project, model connection, and conversation together.',
            'EasyICU 正在一起读取已保存的项目、模型连接和对话。',
          )}</p>
          <div class="gd-startup-progress" aria-hidden="true"><span></span></div>
        </div>
      </div>`;
  }

  function finish(root) {
    active = false;
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

  window.EU_GUIDED_STARTUP = Object.freeze({ begin, finish, isActive, markup });
})();
