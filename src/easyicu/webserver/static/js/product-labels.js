/* Owner: product label projection widget. */
/* User-visible product labels are projected here so internal runtime names do
   not leak differently across Copilot and Copilot conversation. Internal API,
   storage, and diagnostics identifiers remain unchanged.

   D-P2-1 default-title contract: when the stored title is empty or one of the
   legacy defaults below, every consumer projects `fallback || 研究项目 /
   Research project`. Call sites must pass the stored question (or id) as the
   fallback and must NOT invent new per-screen defaults — extend the sets here
   instead so the rail, the known-folder picker, the removal dialog and the
   run-files owner stay consistent. Consumers call this defensively
   (`window.EU_PRODUCT_LABELS?.projectTitle?.(...) ?? String(...).slice(0,200)`)
   so a bundle served without this owner still renders bounded raw text. */
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
