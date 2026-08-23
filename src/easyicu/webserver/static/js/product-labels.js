/* User-visible product labels are projected here so internal runtime names do
   not leak differently across Copilot and Project Monitor. Internal API,
   storage, and diagnostics identifiers remain unchanged. */
(function () {
  const COPILOT_DEFAULT_TITLES = new Set(['Pi Copilot', 'EasyICU Copilot']);
  const PROJECT_DEFAULT_TITLES = new Set([
    ...COPILOT_DEFAULT_TITLES,
    'Untitled ICU study',
    'Untitled guided study',
  ]);

  function copilotTitle(value, fallback) {
    const title = String(value || '').trim();
    const alternative = String(fallback || '').trim();
    if (!title || COPILOT_DEFAULT_TITLES.has(title)) {
      return alternative || 'EasyICU Copilot';
    }
    return title;
  }

  function projectTitle(value, fallback) {
    const title = String(value || '').trim();
    const alternative = String(fallback || '').trim();
    if (!title || PROJECT_DEFAULT_TITLES.has(title)) {
      return alternative || (window.EU_LANG === 'zh' ? '研究项目' : 'Research project');
    }
    return title;
  }

  window.EU_PRODUCT_LABELS = Object.freeze({ copilotTitle, projectTitle });
})();
