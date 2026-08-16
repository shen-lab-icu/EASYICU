/* Guided Pi assistant-message renderer owner.
   Keep this deliberately small: it supports only the trusted display subset
   needed by Copilot replies and never turns arbitrary HTML into DOM. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;


  function safeUrl(value) {
    const literature = window.EU_GUIDED_PI_LITERATURE;
    if (literature && typeof literature.safeUrl === 'function') {
      return literature.safeUrl(value);
    }
    try {
      const parsed = new URL(String(value || ''));
      if (parsed.protocol !== 'https:' || !parsed.hostname || parsed.username || parsed.password) return '';
      return parsed.href;
    } catch (_) { return ''; }
  }

  function inlineHtml(value) {
    return esc(value)
      .replace(/`([^`\n]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/\n/g, '<br>');
  }

  function render(value) {
    const source = String(value == null ? '' : value);
    const linkPattern = /\[([^\]\n]{1,240})\]\(([^)\s]+)\)/g;
    let cursor = 0;
    let output = '';
    let match;
    while ((match = linkPattern.exec(source)) !== null) {
      output += inlineHtml(source.slice(cursor, match.index));
      const url = safeUrl(match[2]);
      output += url
        ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(match[1])}<span aria-hidden="true">↗</span></a>`
        : inlineHtml(match[0]);
      cursor = match.index + match[0].length;
    }
    return output + inlineHtml(source.slice(cursor));
  }

  window.EU_GUIDED_PI_MARKDOWN = { render, safeUrl };
})();
