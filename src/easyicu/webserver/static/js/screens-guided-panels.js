/* Guided Copilot shell side-panel state.
   Owns the right research-progress panel collapse state and controls. */
(function () {
  const CONTEXT_ASIDE_COLLAPSED_KEY = 'easyicu.guided.contextAsideCollapsed.v1';
  let contextAsideCollapsed = readContextAsideCollapsed();

  function readContextAsideCollapsed() {
    try {
      return !!(window.localStorage && window.localStorage.getItem(CONTEXT_ASIDE_COLLAPSED_KEY) === '1');
    } catch (_) {
      return false;
    }
  }

  function setContextAsideCollapsed(collapsed, main) {
    contextAsideCollapsed = !!collapsed;
    try {
      if (window.localStorage) window.localStorage.setItem(CONTEXT_ASIDE_COLLAPSED_KEY, contextAsideCollapsed ? '1' : '0');
    } catch (_) {}
    if (main && main.classList) main.classList.toggle('gd-context-aside-collapsed', contextAsideCollapsed);
  }

  function isContextAsideCollapsed() {
    return contextAsideCollapsed;
  }

  function contextAsideClass() {
    return contextAsideCollapsed ? 'gd-context-aside-collapsed' : '';
  }

  function renderContextAsideRestore(ctx) {
    return `<button class="gd-aside-restore" type="button" data-context-aside-toggle aria-label="${ctx.t('Show research progress', '显示研究进度栏')}" title="${ctx.t('Show research progress', '显示研究进度栏')}">${ctx.icon('chevron', 14)}</button>`;
  }

  function renderContextAsideCollapse(ctx) {
    return `<button class="gd-aside-collapse" type="button" data-context-aside-toggle aria-controls="gdContextAside" aria-label="${ctx.t('Hide research progress', '隐藏研究进度栏')}" title="${ctx.t('Hide research progress', '隐藏研究进度栏')}">${ctx.icon('chevron', 14)}</button>`;
  }

  window.EU_GUIDED_PANELS = {
    setContextAsideCollapsed,
    isContextAsideCollapsed,
    contextAsideClass,
    renderContextAsideRestore,
    renderContextAsideCollapse,
  };
})();
