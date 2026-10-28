/* Owner: the project's task table ("View all" from the conversation rail).
   It lists every conversation of the current project with its state and
   times, filters by text and state, sorts by activity or creation, and opens
   a conversation or starts a new one through the host's callbacks. The rail
   (screens-guided-pi-study-workspace.js) owns the state of each row; this
   file only reads it. */
(function () {
  'use strict';

  const FILTERS = [
    ['all', 'All states', '全部状态'],
    ['needs_user', 'Needs you', '待确认'],
    ['running', 'Running', '进行中'],
    ['stopped', 'Stopped', '已停止'],
    ['done', 'Finished', '已结束'],
    ['idle', 'Not started', '未开始'],
  ];
  const SORTS = { activity: row => row.last_activity_at || row.created_at || '', created: row => row.created_at || '' };

  // host: { tr, esc, iconHtml, state(), projectId(), render(), openSession,
  // startEntry, sessions(), title, status, railKey } from screens-guided-pi.js.
  function create(host) {
    const { tr, esc, iconHtml } = host;
    // The table belongs to the project it was opened in; switching project,
    // opening a conversation or starting one closes it.
    let openIn = '';
    let query = '';
    let filter = 'all';
    let sort = 'activity';
    let ctx = null;
    const bound = new WeakSet();

    const filterKey = key => (key === 'replied' || key === 'done' ? 'done' : key);
    function when(value) {
      const date = value ? new Date(value) : null;
      if (!date || Number.isNaN(date.getTime())) return '—';
      return date.toLocaleString(window.EU_LANG === 'zh' ? 'zh-CN' : 'en-US',
        { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    }
    function visible(row) {
      const key = filterKey(ctx.railKey(row));
      if (filter !== 'all' && key !== filter) return false;
      return !query || `${ctx.title(row)} ${ctx.status(row)}`.toLowerCase().includes(query);
    }
    function rowsHtml() {
      const order = SORTS[sort];
      const rows = ctx.sessions.slice().sort((a, b) => String(order(b)).localeCompare(String(order(a))));
      return rows.map(row => {
        const title = ctx.title(row);
        const key = ctx.railKey(row);
        return `<tr data-gpi-task-row="${esc(row.session_id)}" data-gpi-task-state="${esc(filterKey(key))}" ${visible(row) ? '' : 'hidden'}>`
          + `<td><button type="button" data-gpi-task-open="${esc(row.session_id)}">${esc(title)}</button></td>`
          + `<td><span class="gpi-task-state is-${esc(filterKey(key))}"><i aria-hidden="true"></i>${esc(ctx.status(row))}</span></td>`
          + `<td>${esc(when(row.last_activity_at || row.created_at))}</td><td>${esc(when(row.created_at))}</td></tr>`;
      }).join('');
    }
    function sortHeader(key, en, zh) {
      const active = sort === key;
      return `<th aria-sort="${active ? 'descending' : 'none'}"><button type="button" data-gpi-task-sort="${key}">${tr(en, zh)}${active ? ` ${iconHtml('chevron', 11)}` : ''}</button></th>`;
    }
    function tableHtml() {
      return `<table class="gpi-task-table"><thead><tr><th>${tr('Name', '名称')}</th><th>${tr('State', '状态')}</th>${sortHeader('activity', 'Last activity', '最近活动')}${sortHeader('created', 'Created', '创建时间')}</tr></thead><tbody>${rowsHtml()}</tbody></table>`;
    }
    function emptyNote() {
      const shown = ctx.sessions.some(visible);
      return `<p class="gpi-task-empty" data-gpi-task-empty ${shown ? 'hidden' : ''}>${ctx.sessions.length ? tr('No conversations match.', '没有符合条件的对话。') : tr('No conversations yet.', '暂无对话。')}</p>`;
    }

    function panel(next) {
      ctx = next;
      const options = FILTERS.map(([key, en, zh]) => `<option value="${key}" ${filter === key ? 'selected' : ''}>${tr(en, zh)}</option>`).join('');
      return `<section class="gpi-task-page" data-gpi-task-table aria-label="${tr('All conversations', '全部对话')}">
        <header><h2>${tr('Conversations', '对话')}<span>${ctx.sessions.length}</span></h2><button type="button" class="gpi-task-close" data-gpi-task-close aria-label="${tr('Close', '关闭')}" title="${tr('Close', '关闭')}">${iconHtml('x', 15)}</button></header>
        <div class="gpi-task-toolbar"><label class="gpi-task-search">${iconHtml('search', 14)}<span class="shell-sr-only">${tr('Search conversations', '搜索对话')}</span><input type="search" data-gpi-task-search value="${esc(query)}" placeholder="${tr('Search conversations', '搜索对话')}" autocomplete="off"></label>
          <label><span class="shell-sr-only">${tr('State', '状态')}</span><select data-gpi-task-filter>${options}</select></label>
          <button type="button" class="gpi-task-new" data-gpi-task-new ${ctx.locked ? 'disabled' : ''}>${iconHtml('plus', 14)}<span>${tr('Conversation', '对话')}</span></button></div>
        <div class="gpi-task-scroll" data-gpi-task-body>${tableHtml()}</div>${emptyNote()}</section>`;
    }

    // Filtering hides rows in place so typing keeps the caret in the box.
    function applyFilter(root) {
      root.querySelectorAll('[data-gpi-task-row]').forEach(tr => {
        const row = ctx.sessions.find(item => String(item.session_id) === tr.dataset.gpiTaskRow);
        tr.hidden = !row || !visible(row);
      });
      const empty = root.querySelector('[data-gpi-task-empty]');
      if (empty) empty.hidden = ctx.sessions.some(visible);
    }
    function bind(host) {
      if (!host || bound.has(host)) return;
      bound.add(host);
      host.addEventListener('click', event => {
        const root = event.target.closest('[data-gpi-task-table]');
        if (!root || !ctx) return;
        const open = event.target.closest('[data-gpi-task-open]');
        if (open) { if (!ctx.locked) ctx.open(open.dataset.gpiTaskOpen); return; }
        if (event.target.closest('[data-gpi-task-new]')) { if (!ctx.locked) ctx.create(); return; }
        if (event.target.closest('[data-gpi-task-close]')) { ctx.close(); return; }
        const sorter = event.target.closest('[data-gpi-task-sort]');
        if (sorter) {
          sort = sorter.dataset.gpiTaskSort in SORTS ? sorter.dataset.gpiTaskSort : 'activity';
          const body = root.querySelector('[data-gpi-task-body]');
          if (body) body.innerHTML = tableHtml();
        }
      });
      host.addEventListener('input', event => {
        if (!event.target.matches('[data-gpi-task-search]') || !ctx) return;
        query = event.target.value.trim().toLowerCase();
        applyFilter(event.target.closest('[data-gpi-task-table]'));
      });
      host.addEventListener('change', event => {
        if (!event.target.matches('[data-gpi-task-filter]') || !ctx) return;
        filter = FILTERS.some(([key]) => key === event.target.value) ? event.target.value : 'all';
        applyFilter(event.target.closest('[data-gpi-task-table]'));
      });
    }
    const shown = () => Boolean(openIn) && openIn === host.projectId();
    function show() { openIn = host.projectId(); host.render(); }
    function close() { openIn = ''; host.render(); }
    function openSession(sessionId) {
      const wasShown = shown();
      const state = host.state();
      openIn = '';
      // Reopening the conversation already loaded only needs the repaint.
      if (wasShown && state.session && state.session.session_id === sessionId) { host.render(); return undefined; }
      return host.openSession(sessionId);
    }
    function startEntry() { openIn = ''; return host.startEntry(); }
    function hostPanel() {
      const state = host.state();
      bind(state.host);
      return panel({ sessions: host.sessions(), title: host.title, status: host.status, railKey: host.railKey,
        locked: state.busy || Boolean(state.childJobId) || state.projectLoading,
        open: openSession, create: startEntry, close });
    }
    return Object.freeze({ panel: next => (next ? panel(next) : hostPanel()), bind, shown, show, openSession, startEntry });
  }
  window.EasyICU.guidedPi.declare('taskTable', { create });
})();
