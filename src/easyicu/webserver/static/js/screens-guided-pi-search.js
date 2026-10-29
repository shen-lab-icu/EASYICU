/* Owner: search across this instance's projects, conversations, runs and files.
   The host's /api/search owns the index, its bounds and what is searchable;
   this file shows the hits as plain text and opens them through the owners
   that already open a project (the rail's row), a conversation (the rail's
   open, then the message it matched) and a Drive file (the rail's material
   button). Search is navigation only: it grants nothing and starts nothing. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function icon(name, size) { return typeof window.icon === 'function' ? window.icon(name, size, 1.6) : ''; }

  const LIMIT = 20;
  const SCOPES = [['', 'All', '全部'], ['tasks', 'Conversations', '对话'], ['files', 'Files', '文件'], ['projects', 'Projects', '项目']];
  const KINDS = {
    project: ['Project', '项目', 'folder'],
    conversation: ['Conversation', '对话', 'history'],
    message: ['Message', '消息', 'list'],
    run: ['Run', '运行', 'play'],
    file: ['File', '文件', 'file'],
  };

  let dialog = null;
  let ctx = null;
  let timer = 0;
  const state = { q: '', scope: '', onlyProject: false, hits: [], loading: false, error: '', hasMore: false, partial: false, revision: 0 };
  let titles = new Map();

  function kindOf(hit) { return KINDS[hit && hit.kind] || KINDS.message; }
  function projectLabel(id) {
    const key = String(id || '');
    return titles.get(key) || (ctx && key === ctx.projectId() ? ctx.projectTitle() : '') || '';
  }
  // Offsets are UTF-16 code units into the verbatim snippet, which is what a
  // JavaScript string indexes; anything else is shown without marks.
  function snippetHtml(hit) {
    const text = String(hit.snippet || '');
    const marks = hit.offset_unit === 'utf16' && Array.isArray(hit.matches) ? hit.matches
      .map(row => [Number(row && row.start), Number(row && row.end)])
      .filter(([start, end]) => Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start && end <= text.length)
      .sort((left, right) => left[0] - right[0]) : [];
    // Remove presentation markers by their original offsets, so cleaning a
    // Markdown snippet never moves the host's UTF-16 highlights onto other text.
    const hidden = new Set();
    const omit = (start, end) => { for (let i = start; i < end; i++) hidden.add(i); };
    const code = new Set();
    const codeRange = (start, end) => { for (let i = start; i < end; i++) code.add(i); };
    const containsCode = (start, end) => { for (let i = start; i < end; i++) if (code.has(i)) return true; return false; };
    for (const match of text.matchAll(/^[ \t]{0,3}(`{3,}|~{3,})[^\r\n]*\r?\n/gm)) {
      if (code.has(match.index) || hidden.has(match.index)) continue;
      const start = match.index + match[0].length;
      const fence = match[1];
      const closing = new RegExp(`^[ \\t]{0,3}${fence[0]}{${fence.length},}[ \\t]*(?:\\r?\\n|$)`, 'gm');
      closing.lastIndex = start;
      const close = closing.exec(text);
      codeRange(start, close ? close.index : text.length);
      omit(match.index, start);
      if (close) omit(close.index, close.index + close[0].replace(/\r?\n$/, '').length);
    }
    for (const match of text.matchAll(/(`+)([^`\n]+?)\1/g)) {
      const start = match.index, end = start + match[0].length;
      if (text[start - 1] === '`' || text[end] === '`' || code.has(start) || hidden.has(start)) continue;
      codeRange(start + match[1].length, end - match[1].length);
      omit(start, start + match[1].length);
      omit(end - match[1].length, end);
    }
    // Protect full math expressions from Markdown cleanup, even when TeX is
    // unsupported. Translate only a complete, understood expression outside
    // code and only if every highlighted original character survives verbatim.
    const math = window.EasyICU.guidedPi.optional('markdown');
    const protectedMath = new Set(), replacements = new Map();
    const hasProtectedMath = (start, end) => {
      for (let i = start; i < end; i++) if (protectedMath.has(i)) return true;
      return false;
    };
    const patterns = /\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|\$\$([\s\S]*?)\$\$/g;
    for (const match of text.matchAll(patterns)) {
      const start = match.index, end = start + match[0].length;
      if (containsCode(start, end)) continue;
      let precedingSlashes = 0;
      for (let i = start - 1; i >= 0 && text[i] === '\\'; i--) precedingSlashes++;
      if (precedingSlashes % 2) continue;
      for (let i = start; i < end; i++) protectedMath.add(i);
      const body = match[1] ?? match[2] ?? match[3];
      const projected = math && typeof math.mathProjection === 'function' ? math.mathProjection(body) : null;
      if (!projected || !Array.isArray(projected.sourceOffsets)
        || projected.sourceOffsets.length !== projected.text.length) continue;
      const bodyStart = start + 2;
      const exact = new Set();
      projected.sourceOffsets.forEach((offset, index) => {
        if (Number.isInteger(offset) && body[offset] === projected.text[index]) exact.add(bodyStart + offset);
      });
      const canKeepMarks = marks.every(([left, right]) => {
        for (let i = Math.max(left, start); i < Math.min(right, end); i++) if (!exact.has(i)) return false;
        return true;
      });
      if (canKeepMarks) replacements.set(start, { end, bodyStart, projected });
    }
    // Search excerpts can end mid-formula. Keep such a remainder as source,
    // including any Markdown-looking characters, instead of partly rewriting it.
    for (const opening of text.matchAll(/\\\[|\\\(|\$\$/g)) {
      const start = opening.index;
      if (code.has(start) || protectedMath.has(start)) continue;
      let precedingSlashes = 0;
      for (let i = start - 1; i >= 0 && text[i] === '\\'; i--) precedingSlashes++;
      if (precedingSlashes % 2) continue;
      const close = opening[0] === '\\[' ? '\\]' : opening[0] === '\\(' ? '\\)' : '$$';
      if (text.indexOf(close, start + 2) < 0) {
        for (let i = start; i < text.length; i++) protectedMath.add(i);
        break;
      }
    }
    for (const match of text.matchAll(/^\s{0,3}#{1,6}\s+/gm)) {
      if (!containsCode(match.index, match.index + match[0].length) && !hasProtectedMath(match.index, match.index + match[0].length)) omit(match.index, match.index + match[0].length);
    }
    for (const match of text.matchAll(/(^|[\s([{,.;:!?，。；：！？「『“‘])(\*\*|__|~~)(?=\S)([^\n]*?\S)\2(?=$|[\s)\]},.;:!?，。；：！？])/g)) {
      const start = match.index + match[1].length, end = match.index + match[0].length;
      if (containsCode(start, end) || hasProtectedMath(start, end)) continue;
      omit(start, start + match[2].length);
      omit(end - match[2].length, end);
    }
    for (const match of text.matchAll(/\[([^\]\n]+)\]\([^\)\n]+\)/g)) {
      if (containsCode(match.index, match.index + match[0].length) || hasProtectedMath(match.index, match.index + match[0].length)) continue;
      omit(match.index, match.index + 1);
      omit(match.index + match[1].length + 1, match.index + match[0].length);
    }
    let html = '', marked = false, space = false;
    function append(ch, sourceIndex) {
      const whitespace = /\s/.test(ch);
      if (whitespace && space) return;
      const next = sourceIndex !== null && marks.some(([start, end]) => start <= sourceIndex && sourceIndex < end);
      if (next !== marked) { html += next ? '<mark>' : '</mark>'; marked = next; }
      html += esc(whitespace ? ' ' : ch);
      space = whitespace;
    }
    for (let i = 0; i < text.length; i++) {
      if (hidden.has(i)) continue;
      const replacement = replacements.get(i);
      if (replacement) {
        const { projected, bodyStart } = replacement;
        for (let j = 0; j < projected.text.length; j++) {
          const offset = projected.sourceOffsets[j];
          append(projected.text[j], offset == null ? null : bodyStart + offset);
        }
        i = replacement.end - 1;
        continue;
      }
      append(text[i], i);
    }
    return html + (marked ? '</mark>' : '');
  }

  function hitHtml(hit, index, options = {}) {
    const kind = kindOf(hit);
    const title = String(hit.title || '').trim();
    const project = hit.kind === 'project' ? '' : projectLabel(hit.project_id);
    const snippet = String(hit.snippet || '').trim() && String(hit.snippet).trim() !== title
      ? `<span class="gpi-search-snippet">${snippetHtml(hit)}</span>` : '';
    const label = options.label || tr(kind[0], kind[1]);
    return `<button type="button" class="gpi-search-hit" data-gpi-search-hit="${index}"><span class="gpi-search-hit-icon" aria-hidden="true">${icon(kind[2], 15)}</span><span class="gpi-search-hit-copy">${title ? `<strong>${esc(title)}</strong>` : ''}${snippet}<small>${esc(label)}${project ? ` · ${esc(project)}` : ''}</small></span></button>`;
  }

  function resultGroups() {
    const groups = [], conversations = new Map();
    state.hits.forEach((hit, index) => {
      const target = hit.open || {};
      const id = String(target.conversation_id || hit.conversation_id || '');
      if (target.kind !== 'conversation' || !id) {
        groups.push({ hit, index });
        return;
      }
      const key = `${hit.project_id || target.project_id || ''}/${target.source || ''}/${id}`;
      let group = conversations.get(key);
      if (!group) {
        group = { hit, index, messages: [] };
        conversations.set(key, group);
        groups.push(group);
      }
      if (hit.kind === 'conversation') group.title = hit.title;
      if (hit.kind === 'message') group.messages.push({ hit, index });
    });
    return groups;
  }

  function groupHtml(group) {
    if (!group.messages) return hitHtml(group.hit, group.index);
    const first = group.messages[0] || group;
    const title = group.title || group.hit.title || tr('Research conversation', '研究对话');
    const label = group.messages.length
      ? tr(`Conversation · ${group.messages.length} matching messages`, `对话 · ${group.messages.length} 条匹配消息`)
      : tr('Conversation', '对话');
    const main = hitHtml({ ...first.hit, kind: 'conversation', title }, first.index, { label });
    if (group.messages.length < 2) return main;
    return `<section class="gpi-search-group">${main}<details class="gpi-search-matches"><summary>${esc(tr('Show matching messages', '展开匹配消息'))}</summary>${group.messages.map(row => hitHtml({ ...row.hit, title: '' }, row.index)).join('')}</details></section>`;
  }

  function resultsHtml() {
    if (state.error) return `<p class="gpi-search-state" role="alert">${esc(state.error)}</p>`;
    if (!state.q.trim()) return '';
    if (!state.hits.length) {
      return `<p class="gpi-search-state">${esc(state.loading ? tr('Searching…', '搜索中…') : tr(`Nothing matches “${state.q.trim()}”.`, `没有找到「${state.q.trim()}」。`))}</p>`;
    }
    const groups = resultGroups();
    const notes = [
      state.hasMore ? tr(`Showing the first ${state.hits.length}. Add a word to narrow it.`, `只显示前 ${state.hits.length} 条，多加一个词可以缩小范围。`) : '',
      state.partial ? tr('Some records were too large to search.', '有些记录过大，没有纳入搜索。') : '',
    ].filter(Boolean);
    return `<p class="gpi-search-count">${esc(tr(`${groups.length} results`, `${groups.length} 个结果`))}</p>${groups.map(groupHtml).join('')}${notes.length ? `<p class="gpi-search-note">${esc(notes.join(' '))}</p>` : ''}`;
  }

  function paintResults() {
    const box = dialog && dialog.querySelector('[data-gpi-search-results]');
    if (box) box.innerHTML = resultsHtml();
  }
  function paintScopes() {
    if (!dialog) return;
    dialog.querySelectorAll('[data-gpi-search-scope]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.gpiSearchScope === state.scope));
    });
    const current = dialog.querySelector('[data-gpi-search-project]');
    if (current) current.checked = state.onlyProject;
  }

  async function run() {
    const revision = ++state.revision;
    const q = state.q.trim();
    if (!q) {
      Object.assign(state, { hits: [], loading: false, error: '', hasMore: false, partial: false });
      paintResults();
      return;
    }
    state.loading = true; state.error = '';
    paintResults();
    try {
      const payload = await api().searchWorkspace({ q, scope: state.scope || undefined,
        project: state.onlyProject && ctx ? ctx.projectId() || undefined : undefined, limit: LIMIT });
      if (revision !== state.revision) return;
      Object.assign(state, {
        hits: Array.isArray(payload && payload.hits) ? payload.hits : [],
        hasMore: Boolean(payload && payload.has_more),
        partial: Boolean(payload && (payload.index_truncated || (Array.isArray(payload.issues) && payload.issues.length))),
        loading: false,
      });
    } catch (failure) {
      if (revision !== state.revision) return;
      Object.assign(state, { hits: [], loading: false, error: tr('Search is unavailable right now.', '暂时无法搜索。') });
    }
    paintResults();
  }
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(run, 180);
  }

  // Display names for hits in other projects come from the project catalogue.
  async function loadTitles() {
    if (!api().loadGuidedDrafts) return;
    try {
      const payload = await api().loadGuidedDrafts({ limit: 100 });
      const rows = payload && Array.isArray(payload.drafts) ? payload.drafts : [];
      const label = window.EU_PRODUCT_LABELS && window.EU_PRODUCT_LABELS.projectTitle;
      titles = new Map(rows.map(row => [String(row.id), label ? label(row.title, row.id) : String(row.title || row.id)]));
      paintResults();
    } catch (_) {}
  }

  // A Drive file opens once its project's rail lists it; the wait is bounded.
  function openDriveFile(target) {
    const key = `${target.run_id || ''}/${target.artifact || ''}`;
    let tries = 0;
    const look = () => {
      const button = Array.from(document.querySelectorAll('[data-gpi-rail-material-key]'))
        .find(node => node.dataset.gpiRailMaterialKey === key);
      if (button) { button.click(); return; }
      if (++tries < 24) setTimeout(look, 250);
    };
    look();
  }

  // A message hit is revealed once its conversation renders; the wait is bounded.
  function revealMessage(entryId) {
    let tries = 0;
    const look = () => {
      const node = Array.from(document.querySelectorAll('[data-gpi-entry-id]'))
        .find(row => row.dataset.gpiEntryId === entryId);
      if (node) {
        node.scrollIntoView({ block: 'center' });
        node.classList.add('is-search-hit');
        setTimeout(() => node.classList.remove('is-search-hit'), 2400);
        return;
      }
      if (++tries < 24) setTimeout(look, 250);
    };
    look();
  }

  async function openHit(hit) {
    const target = (hit && hit.open) || {};
    const projectId = String(target.project_id || (hit && hit.project_id) || '');
    if (!projectId || !ctx) return;
    const sessionId = target.kind === 'conversation' && target.source === 'pi' ? String(target.conversation_id || '') : '';
    if (projectId === ctx.projectId()) {
      if (sessionId && !ctx.canOpen()) {
        state.error = tr('Wait until the current reply finishes.', '请等当前回复结束后再打开。');
        paintResults();
        return;
      }
      close();
      if (sessionId) await ctx.openSession(sessionId);
    } else {
      close();
      // The project owner opens the conversation the address names once the
      // project's conversations load.
      const projectOwner = window.EasyICU.guidedPi.optional('project');
      if (projectOwner && projectOwner.syncLocation) projectOwner.syncLocation(projectId, sessionId);
      if (!projectOwner || !projectOwner.openListedProject || !projectOwner.openListedProject(projectId)) return;
    }
    if (target.kind === 'file') openDriveFile(target);
    else if (sessionId && target.message_id) revealMessage(String(target.message_id));
  }

  function close() {
    clearTimeout(timer);
    if (dialog && dialog.open) dialog.close();
  }

  function build() {
    const node = document.createElement('dialog');
    node.className = 'gpi-search';
    node.setAttribute('aria-label', tr('Search', '搜索'));
    node.innerHTML = `<div class="gpi-search-box"><span aria-hidden="true">${icon('search', 16)}</span><input type="search" data-gpi-search-input maxlength="200" autocomplete="off" placeholder="${esc(tr('Search conversations, files and projects', '搜索对话、文件和项目'))}"><button type="button" data-gpi-search-close aria-label="${esc(tr('Close', '关闭'))}">${icon('x', 15)}</button></div>`
      + `<div class="gpi-search-scopes">${SCOPES.map(([scope, en, zh]) => `<button type="button" data-gpi-search-scope="${scope}" aria-pressed="false">${esc(tr(en, zh))}</button>`).join('')}<label class="gpi-search-project"><input type="checkbox" data-gpi-search-project ${ctx && ctx.projectId() ? '' : 'disabled'}>${esc(tr('Only this project', '仅当前项目'))}</label></div>`
      + '<div class="gpi-search-results" data-gpi-search-results aria-live="polite"></div>';
    node.addEventListener('click', event => {
      if (event.target === node || event.target.closest('[data-gpi-search-close]')) { close(); return; }
      const scope = event.target.closest('[data-gpi-search-scope]');
      if (scope) {
        state.scope = scope.dataset.gpiSearchScope;
        paintScopes();
        run();
        node.querySelector('[data-gpi-search-input]').focus();
        return;
      }
      const hit = event.target.closest('[data-gpi-search-hit]');
      if (hit) void openHit(state.hits[Number(hit.dataset.gpiSearchHit)]);
    });
    node.addEventListener('input', event => {
      if (event.target.matches('[data-gpi-search-project]')) {
        state.onlyProject = Boolean(event.target.checked);
        run();
        return;
      }
      if (!event.target.matches('[data-gpi-search-input]')) return;
      state.q = event.target.value;
      schedule();
    });
    // Arrow keys move between the box and the hits; Enter opens the focused hit.
    node.addEventListener('keydown', event => {
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
      const stops = [node.querySelector('[data-gpi-search-input]'), ...Array.from(node.querySelectorAll('[data-gpi-search-hit]'))
        .filter(button => !button.closest('details:not([open])'))];
      const at = stops.indexOf(document.activeElement);
      if (at < 0 || stops.length < 2) return;
      event.preventDefault();
      const next = Math.max(0, Math.min(stops.length - 1, at + (event.key === 'ArrowDown' ? 1 : -1)));
      stops[next].focus();
    });
    document.body.appendChild(node);
    return node;
  }

  // ctx: { projectId(), projectTitle(), openSession(id), canOpen() }
  function open(context) {
    ctx = context;
    // Rebuilt on each open so the interface language is current.
    if (dialog) dialog.remove();
    dialog = build();
    Object.assign(state, { error: '', hits: [], hasMore: false, onlyProject: Boolean(state.onlyProject && ctx.projectId()) });
    const input = dialog.querySelector('[data-gpi-search-input]');
    input.value = state.q;
    paintScopes();
    paintResults();
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
    input.focus();
    input.select();
    void loadTitles();
    if (state.q.trim()) run();
  }

  window.EasyICU.guidedPi.declare('globalSearch', { open, close, snippetHtml });
})();
