/* Guided Pi assistant-message renderer owner.
   Keep this deliberately small: it supports only the trusted display subset
   needed by Copilot replies and never turns arbitrary HTML into DOM. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  const HEADING = /^\s{0,3}(#{1,6})\s+(.*)$/;
  const BULLET = /^\s{0,3}[-*]\s+(.+)$/;
  const ORDERED = /^\s{0,3}\d{1,3}[.)]\s+(.+)$/;

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

  /* Inline formatting plus the one safe link form. Kept separate from the
     block pass so a link inside a list item or heading still renders. */
  function inlineWithLinks(value) {
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

  /* Block pass. Headings and lists are the only structures Copilot replies
     actually use; before this they leaked to the screen as literal "###" and
     "- " because only the newest reply is converted into host next-step
     buttons and every older turn kept its raw markers. Every heading level
     renders at one restrained size: a reply is a chat message, not a document,
     and an h1-sized "下一步" would out-shout the answer above it. */
  function render(value) {
    const source = String(value == null ? '' : value);
    if (!source.trim()) return '';
    const lines = source.split(/\r?\n/);
    const out = [];
    let listTag = '';
    let paragraph = [];

    function flushParagraph() {
      if (!paragraph.length) return;
      out.push(`<p>${inlineWithLinks(paragraph.join('\n'))}</p>`);
      paragraph = [];
    }
    function flushList() {
      if (!listTag) return;
      out.push(`</${listTag}>`);
      listTag = '';
    }
    function openList(tag) {
      if (listTag === tag) return;
      flushList();
      out.push(`<${tag} class="gpi-md-list">`);
      listTag = tag;
    }

    for (const line of lines) {
      const heading = line.match(HEADING);
      if (heading) {
        flushParagraph();
        flushList();
        out.push(`<p class="gpi-md-heading">${inlineWithLinks(heading[2])}</p>`);
        continue;
      }
      const bullet = line.match(BULLET);
      if (bullet) {
        flushParagraph();
        openList('ul');
        out.push(`<li>${inlineWithLinks(bullet[1])}</li>`);
        continue;
      }
      const ordered = line.match(ORDERED);
      if (ordered) {
        flushParagraph();
        openList('ol');
        out.push(`<li>${inlineWithLinks(ordered[1])}</li>`);
        continue;
      }
      if (!line.trim()) {
        flushParagraph();
        flushList();
        continue;
      }
      flushList();
      paragraph.push(line);
    }
    flushParagraph();
    flushList();
    return out.join('');
  }

  window.EU_GUIDED_PI_MARKDOWN = { render, safeUrl };
})();
