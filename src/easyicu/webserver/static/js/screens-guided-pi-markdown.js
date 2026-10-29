/* Owner: Guided Pi markdown render widget. */
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
  function inlineWithLinks(value, resources) {
    const source = String(value == null ? '' : value);
    const linkPattern = /\[([^\]\n]{1,240})\]\(([^)\s]+)\)/g;
    let cursor = 0;
    let output = '';
    let match;
    while ((match = linkPattern.exec(source)) !== null) {
      output += inlineHtml(source.slice(cursor, match.index));
      const url = safeUrl(match[2]);
      const owner = !url && window.EasyICU.guidedPi.optional('resources');
      const resourceOwner = owner && owner.create({ esc });
      const attached = resourceOwner && typeof resourceOwner.relativeLink === 'function'
        ? resourceOwner.relativeLink(resources, match[2], match[1]) : '';
      output += url
        ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(match[1])}<span aria-hidden="true">↗</span></a>`
        : attached || inlineHtml(match[0]);
      cursor = match.index + match[0].length;
    }
    return output + inlineHtml(source.slice(cursor));
  }

  // A readable text fallback for common model-written TeX. This never runs a
  // TeX/HTML interpreter. Unsupported or malformed syntax stays escaped as
  // source instead of silently changing the meaning of an equation.
  // Preserve UTF-16 source offsets for each displayed character so callers
  // can retain an exact search highlight without guessing through a formula.
  // Generated operators and punctuation deliberately have no source offset.
  function mathProjection(value) {
    const source = String(value || '');
    if (source.length > 4096) return null;
    let cursor = 0;
    let valid = true;
    const symbols = {
      times: ' × ', cdot: ' · ', pm: ' ± ', mp: ' ∓ ',
      le: ' ≤ ', leq: ' ≤ ', ge: ' ≥ ', geq: ' ≥ ', ne: ' ≠ ', neq: ' ≠ ',
      approx: ' ≈ ', infty: '∞', to: ' → ', in: ' ∈ ',
      alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', Delta: 'Δ',
      mu: 'μ', sigma: 'σ', Sigma: 'Σ', lambda: 'λ', theta: 'θ',
      pi: 'π', epsilon: 'ε', rho: 'ρ', tau: 'τ', omega: 'ω', Omega: 'Ω',
      log: 'log', ln: 'ln', exp: 'exp', min: 'min', max: 'max', Pr: 'Pr',
      left: '', right: '', displaystyle: '', quad: ' ', qquad: ' ',
    };
    const generated = value => Array.from(String(value), ch => ({ ch, offset: null }));
    function group(depth) {
      while (/\s/.test(source[cursor] || '') && cursor < source.length) cursor += 1;
      if (source[cursor] !== '{') { valid = false; return []; }
      cursor += 1;
      return parse(true, depth + 1);
    }
    function parse(braced, depth) {
      if (depth > 24) { valid = false; return []; }
      let output = [];
      while (cursor < source.length && valid) {
        const offset = cursor;
        const ch = source[cursor++];
        if (ch === '}') {
          if (braced) return output;
          valid = false;
        } else if (ch === '{') {
          output.push(...generated('('), ...parse(true, depth + 1), ...generated(')'));
        } else if (ch === '\\') {
          const command = source.slice(cursor).match(/^[A-Za-z]+/);
          if (!command) {
            const literalOffset = cursor, literal = source[cursor++] || '';
            if ('%_{}$#&'.includes(literal) && literal) output.push({ ch: literal, offset: literalOffset });
            else if (',;:! '.includes(literal) && literal) output.push(...generated(' '));
            else valid = false;
            continue;
          }
          cursor += command[0].length;
          if (['frac', 'dfrac', 'tfrac'].includes(command[0])) {
            const numerator = group(depth), denominator = group(depth);
            output.push(...generated('('), ...numerator, ...generated(') / ('), ...denominator, ...generated(')'));
          } else if (['text', 'mathrm', 'textrm', 'mathbf', 'mathsf', 'operatorname'].includes(command[0])) {
            output.push(...group(depth));
          } else if (Object.prototype.hasOwnProperty.call(symbols, command[0])) {
            output.push(...generated(symbols[command[0]]));
          } else valid = false;
        } else output.push({ ch, offset });
      }
      if (braced) valid = false;
      return output;
    }
    const parsed = parse(false, 0);
    if (!valid) return null;
    // Match mathText's historic whitespace normalization while retaining one
    // source offset per rendered UTF-16 unit. Generated symbols have no offset.
    const result = [];
    let pendingSpace = null;
    for (const part of parsed) {
      if (/\s/.test(part.ch)) { if (result.length && pendingSpace === null) pendingSpace = part; continue; }
      if (pendingSpace) { result.push({ ch: ' ', offset: pendingSpace.offset }); pendingSpace = null; }
      result.push(part);
    }
    if (!result.length) return null;
    return { text: result.map(part => part.ch).join(''), sourceOffsets: result.map(part => part.offset) };
  }

  function mathText(value) {
    const projected = mathProjection(value);
    return projected ? projected.text : null;
  }

  function mathHtml(source) {
    const text = mathText(source);
    return text == null
      ? `<code class="gpi-math-source">${esc(source)}</code>`
      : `<span role="math">${esc(text)}</span>`;
  }

  function inlineWithMath(value, resources) {
    return String(value).split(/(`[^`\n]+`)/g).map(part => {
      if (part.startsWith('`')) return inlineWithLinks(part, resources);
      const pattern = /\\\(([^\n]*?)\\\)/g;
      let cursor = 0, output = '', match;
      while ((match = pattern.exec(part)) !== null) {
        output += inlineWithLinks(part.slice(cursor, match.index), resources) + mathHtml(match[1]);
        cursor = match.index + match[0].length;
      }
      return output + inlineWithLinks(part.slice(cursor), resources);
    }).join('');
  }

  /* Block pass. Headings and lists are the only structures Copilot replies
     actually use; before this they leaked to the screen as literal "###" and
     "- " because only the newest reply is converted into host next-step
     buttons and every older turn kept its raw markers. Every heading level
     renders at one restrained size: a reply is a chat message, not a document,
     and an h1-sized "下一步" would out-shout the answer above it. */
  function render(value, options) {
    const source = String(value == null ? '' : value);
    if (!source.trim()) return '';
    const resources = options && Array.isArray(options.resources) ? options.resources : [];
    const inline = text => inlineWithMath(text, resources);
    const lines = source.split(/\r?\n/);
    const out = [];
    let listTag = '';
    let paragraph = [];

    function flushParagraph() {
      if (!paragraph.length) return;
      out.push(`<p>${inline(paragraph.join('\n'))}</p>`);
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

    // GitHub-style tables: a header row, a |---| separator, then rows. The
    // reference UI renders these; a reply's result summary is often one.
    const TABLE_ROW = /^\s*\|.*\|\s*$/;
    const TABLE_RULE = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;
    const cells = row => row.trim().replace(/^\|/, '').replace(/\|$/, '')
      .split(/(?<!\\)\|/).map(cell => inline(cell.replace(/\\\|/g, '|').trim()));
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const fence = line.match(/^\s{0,3}(`{3,}|~{3,})[^\n]*$/);
      if (fence) {
        flushParagraph();
        flushList();
        const end = new RegExp(`^\\s{0,3}${fence[1][0]}{${fence[1].length},}\\s*$`);
        const code = [];
        while (++index < lines.length && !end.test(lines[index])) code.push(lines[index]);
        out.push(`<pre><code>${esc(code.join('\n'))}</code></pre>`);
        continue;
      }
      const trimmed = line.trim();
      const opening = trimmed.startsWith('\\[') ? '\\[' : trimmed.startsWith('$$') ? '$$' : '';
      if (opening) {
        const closing = opening === '\\[' ? '\\]' : '$$';
        let formula = trimmed.slice(opening.length), next = index;
        while (!formula.includes(closing) && formula.length <= 4096 && next + 1 < lines.length) {
          formula += '\n' + lines[++next];
        }
        const end = formula.indexOf(closing);
        if (end >= 0 && !formula.slice(end + closing.length).trim()) {
          flushParagraph();
          flushList();
          out.push(`<p class="gpi-md-math">${mathHtml(formula.slice(0, end))}</p>`);
          index = next;
          continue;
        }
      }
      if (TABLE_ROW.test(line) && index + 1 < lines.length && TABLE_RULE.test(lines[index + 1])) {
        flushParagraph();
        flushList();
        const head = cells(line);
        const body = [];
        index += 2;
        while (index < lines.length && TABLE_ROW.test(lines[index])) { body.push(cells(lines[index])); index += 1; }
        index -= 1;
        out.push(`<table class="gpi-md-table"><thead><tr>${head.map(cell => `<th>${cell}</th>`).join('')}</tr></thead>`
          + `<tbody>${body.map(row => `<tr>${head.map((_, column) => `<td>${row[column] || ''}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
        continue;
      }
      const heading = line.match(HEADING);
      if (heading) {
        flushParagraph();
        flushList();
        const ideaBlock = ideaMiningBlock(heading[2]);
        if (ideaBlock) {
          out.push(`<h3 class="gpi-md-heading gpi-idea-heading">${inline(ideaBlock.leadIn)}</h3>`);
          if (ideaBlock.body) out.push(`<p>${inline(ideaBlock.body)}</p>`);
        } else {
          out.push(`<p class="gpi-md-heading">${inline(heading[2])}</p>`);
        }
        continue;
      }
      const ideaBlock = ideaMiningBlock(line);
      if (ideaBlock) {
        flushParagraph();
        flushList();
        out.push(`<h3 class="gpi-md-heading gpi-idea-heading">${inline(ideaBlock.leadIn)}</h3>`);
        if (ideaBlock.body) out.push(`<p>${inline(ideaBlock.body)}</p>`);
        continue;
      }
      const bullet = line.match(BULLET);
      if (bullet) {
        flushParagraph();
        openList('ul');
        out.push(`<li>${inline(bullet[1])}</li>`);
        continue;
      }
      const ordered = line.match(ORDERED);
      if (ordered) {
        flushParagraph();
        openList('ol', Number(ordered[1]));
        out.push(`<li>${inline(ordered[2])}</li>`);
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

  window.EasyICU.guidedPi.declare('markdown', { render, safeUrl, mathProjection });
})();
