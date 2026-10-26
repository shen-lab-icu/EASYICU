/* Owner: search across this instance's projects, conversations, runs and files.
   The host's /api/search owns the index, its bounds and what is searchable;
   this file shows the hits as plain text and opens them through the owners
   that already open a project (EU_GUIDED_CONTEXT.openProject), a conversation
   (the rail's open, then the message it matched) and a Drive file (the rail's
   material button). Search is navigation only: it grants nothing and starts
   nothing. */
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
  const state = { q: '', scope: '', hits: [], loading: false, error: '', hasMore: false, partial: false, revision: 0 };
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
    let html = '';
    let at = 0;
    marks.forEach(([start, end]) => {
      if (start < at) return;
      html += `${esc(text.slice(at, start))}<mark>${esc(text.slice(start, end))}</mark>`;
      at = end;
    });
    return html + esc(text.slice(at));
  }

  function hitHtml(hit, index) {
    const kind = kindOf(hit);
    const title = String(hit.title || '').trim();
    const project = hit.kind === 'project' ? '' : projectLabel(hit.project_id);
    const snippet = String(hit.snippet || '').trim() ? `<span class="gpi-search-snippet">${snippetHtml(hit)}</span>` : '';
    return `<button type="button" class="gpi-search-hit" data-gpi-search-hit="${index}"><span class="gpi-search-hit-icon" aria-hidden="true">${icon(kind[2], 15)}</span><span class="gpi-search-hit-copy">${title ? `<strong>${esc(title)}</strong>` : ''}${snippet}<small>${esc(tr(kind[0], kind[1]))}${project ? ` · ${esc(project)}` : ''}</small></span></button>`;
  }

  function resultsHtml() {
    if (state.error) return `<p class="gpi-search-state" role="alert">${esc(state.error)}</p>`;
    if (!state.q.trim()) return '';
    if (!state.hits.length) {
      return `<p class="gpi-search-state">${esc(state.loading ? tr('Searching…', '搜索中…') : tr(`Nothing matches “${state.q.trim()}”.`, `没有找到「${state.q.trim()}」。`))}</p>`;
    }
    const notes = [
      state.hasMore ? tr(`Showing the first ${state.hits.length}. Add a word to narrow it.`, `只显示前 ${state.hits.length} 条，多加一个词可以缩小范围。`) : '',
      state.partial ? tr('Some records were too large to search.', '有些记录过大，没有纳入搜索。') : '',
    ].filter(Boolean);
    return `${state.hits.map(hitHtml).join('')}${notes.length ? `<p class="gpi-search-note">${esc(notes.join(' '))}</p>` : ''}`;
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
      const payload = await api().searchWorkspace({ q, scope: state.scope || undefined, limit: LIMIT });
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
      const guided = window.EU_GUIDED_CONTEXT;
      if (!guided || !guided.openProject || !(await guided.openProject(projectId))) return;
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
      + `<div class="gpi-search-scopes">${SCOPES.map(([scope, en, zh]) => `<button type="button" data-gpi-search-scope="${scope}" aria-pressed="false">${esc(tr(en, zh))}</button>`).join('')}</div>`
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
      if (!event.target.matches('[data-gpi-search-input]')) return;
      state.q = event.target.value;
      schedule();
    });
    // Arrow keys move between the box and the hits; Enter opens the focused hit.
    node.addEventListener('keydown', event => {
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
      const stops = [node.querySelector('[data-gpi-search-input]'), ...node.querySelectorAll('[data-gpi-search-hit]')];
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
    Object.assign(state, { error: '' });
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
