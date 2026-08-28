/* EasyICU Copilot shell entry.
   There is one user-visible conversation: the Pi AgentSession mounted at
   #guided. The historical page-guide dock intentionally is not constructed,
   and neither is a floating launcher: it overlapped the composer's own send
   control on the guided route. EUPageGuide remains the shell hook that app.js
   uses for [data-cpopen] affordances and the Cmd/Ctrl+K shortcut. */
(function () {
  function routeOf() {
    return (location.hash || '#entry').slice(1);
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

  // Kept so existing shell hooks can call it after a language change; there is
  // no longer a launcher label to relabel.
  function refreshLanguage() {}

  function init() {
    window.EUPageGuide = { open, close, toggle, refreshLanguage };
    window.EUCopilot = window.EUPageGuide;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
