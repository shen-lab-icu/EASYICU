/* EasyICU Copilot shell entry.
   There is one user-visible conversation: the Pi AgentSession mounted at
   #guided. The historical page-guide dock intentionally is not constructed;
   EUPageGuide remains only as a compatibility alias for existing shell hooks. */
(function () {
  let fab = null;

  function tx(en, zh) {
    return window.t ? window.t(en, zh) : en;
  }

  function routeOf() {
    return (location.hash || '#entry').slice(1);
  }

  function shouldHideFab(route) {
    return route === 'guided' || route === 'agent';
  }

  function focusComposer() {
    const composer = document.querySelector('[data-gpi-input]');
    if (composer) composer.focus();
  }

  function open() {
    if (routeOf() !== 'guided') {
      location.hash = '#guided';
      window.setTimeout(focusComposer, 450);
      return;
    }
    focusComposer();
  }

  function close() {}

  function toggle() {
    open();
  }

  function refreshLanguage() {
    if (!fab) return;
    fab.setAttribute('aria-label', tx('Open EasyICU Copilot', '打开 EasyICU 研究助手'));
    fab.innerHTML = `<span class="fab-mk">${icon('spark', 14)}</span> ${tx('EasyICU Copilot', '研究助手')}`;
    fab.hidden = shouldHideFab(routeOf());
  }

  function build() {
    fab = document.createElement('button');
    fab.id = 'cpFab';
    fab.type = 'button';
    document.body.appendChild(fab);
    fab.addEventListener('click', open);
    window.addEventListener('hashchange', refreshLanguage);
    window.addEventListener('easyicu:languagechange', refreshLanguage);
  }

  function init() {
    build();
    window.EUPageGuide = { open, close, toggle, refreshLanguage };
    window.EUCopilot = window.EUPageGuide;
    refreshLanguage();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
