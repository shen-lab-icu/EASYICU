/* Guided Pi assistant-message renderer owner.
   Keep this deliberately small: it supports only the trusted display subset
   needed by Copilot replies and never turns arbitrary HTML into DOM. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  // Copilot occasionally omits the Markdown space before Chinese text. Treat
  // that bounded display variation as structure instead of leaking "###" or
  // a leading dash into the conversation.
  const HEADING = /^\s{0,3}(#{1,6})(?:\s+|(?=[\u3400-\u9fff]))(.*)$/;
  const BULLET = /^\s{0,3}[-*](?:\s+|(?=[\u3400-\u9fff]))(.+)$/;
  const ORDERED = /^\s{0,3}(\d{1,3})[.)]\s+(.+)$/;
  const IDEA_MINING_LEAD_INS = new Set([
    '我理解的想法', '已经明确', 'EasyICU 目前能做', '还需要文献回答',
    '我听到的现象', '可以探索的方向', '当前证据边界', '先选哪条路',
    '检索地图', '候选路线比较', '优先验证', '还不能证明',
    '我理解的起点', '值得继续验证的创新方向', '证据与数据边界', '建议先做什么',
  ]);
  const IDEA_MINING_INLINE = /^\s*(?:\*\*|__)?(我理解的想法|已经明确|EasyICU\s*目前能做|还需要文献回答|我听到的现象|可以探索的方向|当前证据边界|先选哪条路|检索地图|候选路线比较|优先验证|还不能证明|我理解的起点|值得继续验证的创新方向|证据与数据边界|建议先做什么)\s*[：:](?:\*\*|__)?\s*(.*)$/i;

  function ideaMiningLeadIn(value) {
    let text = String(value == null ? '' : value).trim();
    if ((text.startsWith('**') && text.endsWith('**'))
      || (text.startsWith('__') && text.endsWith('__'))) {
      text = text.slice(2, -2).trim();
    }
    text = text.replace(/[：:]$/, '').trim();
    return IDEA_MINING_LEAD_INS.has(text) ? text : '';
  }

  function ideaMiningBlock(value) {
    const leadIn = ideaMiningLeadIn(value);
    if (leadIn) return { leadIn, body: '' };
    const match = String(value == null ? '' : value).match(IDEA_MINING_INLINE);
    if (!match) return null;
    const compact = match[1].replace(/^EasyICU\s*/i, 'EasyICU ');
    const canonical = Array.from(IDEA_MINING_LEAD_INS).find(row => (
      row.toLowerCase() === compact.toLowerCase()
    ));
    return canonical ? { leadIn: canonical, body: String(match[2] || '').trim() } : null;
  }

  function safeUrl(value) {
    const literature = window.EasyICU.guidedPi.optional('literature');
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
    function openList(tag, start = 1) {
      if (listTag === tag) return;
      flushList();
      const startAttribute = tag === 'ol' && start > 1 ? ` start="${start}"` : '';
      out.push(`<${tag} class="gpi-md-list"${startAttribute}>`);
      listTag = tag;
    }

    for (const line of lines) {
      const heading = line.match(HEADING);
      if (heading) {
        flushParagraph();
        flushList();
        const ideaBlock = ideaMiningBlock(heading[2]);
        if (ideaBlock) {
          out.push(`<h3 class="gpi-md-heading gpi-idea-heading">${inlineWithLinks(ideaBlock.leadIn)}</h3>`);
          if (ideaBlock.body) out.push(`<p>${inlineWithLinks(ideaBlock.body)}</p>`);
        } else {
          out.push(`<p class="gpi-md-heading">${inlineWithLinks(heading[2])}</p>`);
        }
        continue;
      }
      const ideaBlock = ideaMiningBlock(line);
      if (ideaBlock) {
        flushParagraph();
        flushList();
        out.push(`<h3 class="gpi-md-heading gpi-idea-heading">${inlineWithLinks(ideaBlock.leadIn)}</h3>`);
        if (ideaBlock.body) out.push(`<p>${inlineWithLinks(ideaBlock.body)}</p>`);
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
        openList('ol', Number(ordered[1]));
        out.push(`<li>${inlineWithLinks(ordered[2])}</li>`);
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

  window.EasyICU.guidedPi.declare('markdown', { render, safeUrl });
})();
