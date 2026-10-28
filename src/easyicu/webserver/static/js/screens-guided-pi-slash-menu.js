/* Owner: inline /query selection in the conversation composer. The session's
   attachment owner supplies the catalogue, exact identities and limit, and
   persists the chosen Skill. This owner only edits the slash token after a
   successful attachment. Popover dismissal and fitting belong to popover-menus.js. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;
  const tr = (en, zh) => window.EU_LANG === 'zh' ? zh : en;
  const groups = () => [
    { key: 'workflow', title: tr('Workflows', '流程'), words: 'workflow research_workflow 流程' },
    { key: 'module', title: tr('Modules', '模块'), words: 'module analysis_module 模块' },
    { key: 'user', title: tr('Your Skills', '我的技能'), words: 'user personal skill 我的技能' },
    { key: 'other', title: tr('Other', '其他'), words: 'other data_definition planning_aid 其他 数据定义 规划' },
  ];
  const groupOf = row => row.source === 'user' ? 'user' : row.layer === 'research_workflow' ? 'workflow'
    : row.layer === 'analysis_module' ? 'module' : 'other';

  function tokenAt(input) {
    if (!input || input.selectionStart !== input.selectionEnd) return null;
    const caret = input.selectionStart;
    const match = input.value.slice(0, caret).match(/(?:^|\s)\/([^\s/]*)$/u);
    if (!match) return null;
    const tail = input.value.slice(caret).match(/^\S*/u)[0];
    if (tail.includes('/')) return null;
    const start = caret - match[1].length - 1;
    return { start, end: caret + tail.length, query: match[1] + tail };
  }

  function create({ state, projectId, render }) {
    const attach = window.EasyICU.guidedPi.optional('skillAttach');
    const inputOf = () => state.host && state.host.querySelector('[data-gpi-input]');
    const menuOf = () => state.host && state.host.querySelector('[data-gpi-slash-menu]');
    const ready = () => attach && state.session && !state.busy && !state.childJobId
      && !state.projectLoading && !state.session.stale?.stale && inputOf() && !inputOf().disabled;
    let active = null;
    let items = [];
    let rows = [];
    let selected = -1;
    let composing = false;
    let saving = false;
    let status = '';
    let sequence = 0;
    let selection = null;

    function close() {
      active = null; rows = []; selected = -1; sequence++;
      const menu = menuOf();
      if (menu) menu.remove();
      const input = inputOf();
      if (input) ['role', 'aria-autocomplete', 'aria-controls', 'aria-expanded', 'aria-activedescendant']
        .forEach(key => input.removeAttribute(key));
    }
    const sameContext = () => active && active.session === state.session && active.project === projectId();
    function reason(row) {
      if (attach.attached(state.session, row.source, row.id)) return tr('Already attached', '已添加');
      const max = attach.maxAttached(state.session);
      if ((state.session.draft_attached_skills || []).length >= max) return tr(`Limit of ${max} Skills reached`, `已达 ${max} 个技能上限`);
      return saving ? tr('Attaching…', '正在添加…') : '';
    }
    function paint() {
      if (!sameContext() || !ready() || composing) { close(); return; }
      const input = inputOf();
      let menu = menuOf();
      if (!menu) {
        const card = state.host.querySelector('.gpi-compose-card');
        if (!card) { close(); return; }
        window.EU_POPOVER_MENUS?.closeAll();
        menu = document.createElement('details');
        menu.className = 'gpi-slash-menu';
        menu.setAttribute('data-popover-menu', '');
        menu.setAttribute('data-gpi-slash-menu', '');
        card.appendChild(menu);
        menu.open = true;
      }
      const query = active.token.query.normalize('NFKC').toLowerCase();
      rows = groups().flatMap(group => items.filter(row => row.enabled === true && groupOf(row) === group.key
        && [row.id, row.title, row.title_zh, row.category, row.layer, group.words].filter(Boolean)
          .join(' ').normalize('NFKC').toLowerCase().includes(query)));
      if (selected < 0 || selected >= rows.length || reason(rows[selected])) selected = rows.findIndex(row => !reason(row));
      let previous = '';
      const list = rows.map((row, index) => {
        const group = groups().find(item => item.key === groupOf(row));
        const heading = previous !== group.key ? `<div class="gpi-slash-group" role="presentation">${esc(group.title)}</div>` : '';
        previous = group.key;
        const disabled = reason(row);
        const title = (window.EU_LANG === 'zh' && row.title_zh) || row.title || row.id;
        return `${heading}<button type="button" role="option" tabindex="-1" id="gpi-slash-row-${index}" class="gpi-slash-row${selected === index ? ' is-active' : ''}" data-gpi-slash-row="${index}" aria-selected="${selected === index}"${disabled ? ' aria-disabled="true"' : ''}><strong>${esc(title)}</strong><span>${esc(disabled || row.id)}</span></button>`;
      }).join('');
      const message = status === 'loading' ? tr('Loading Skills…', '正在加载技能…')
        : status === 'failed' ? tr('Could not load Skills right now. Type to retry.', '暂时无法加载技能，请继续输入重试。')
        : tr('No matching Skills.', '没有匹配的技能。');
      menu.innerHTML = `<summary class="gpi-slash-summary" tabindex="-1">${esc(tr('Skills', '技能'))}</summary><div class="gpi-slash-popover" id="gpi-slash-options" role="listbox" aria-label="${esc(tr('Attach a Skill', '添加技能'))}" aria-busy="${status === 'loading' || saving}">${list || `<p class="gpi-slash-empty" role="status">${esc(message)}</p>`}</div>`;
      input.setAttribute('role', 'combobox');
      input.setAttribute('aria-autocomplete', 'list');
      input.setAttribute('aria-controls', 'gpi-slash-options');
      input.setAttribute('aria-expanded', 'true');
      if (selected >= 0) input.setAttribute('aria-activedescendant', `gpi-slash-row-${selected}`);
      else input.removeAttribute('aria-activedescendant');
      window.EU_POPOVER_MENUS?.fit(menu);
    }

    function update(event) {
      const input = inputOf();
      if (!input || (event && event.target !== input)) return;
      if (!ready() || composing || event?.isComposing) { close(); return; }
      const token = tokenAt(input);
      if (!token) { close(); return; }
      const changed = !sameContext();
      if (changed) { items = []; selected = -1; }
      else if (active.token.query !== token.query) selected = -1;
      active = { token, session: state.session, project: projectId() };
      if (changed || status === 'failed') {
        status = 'loading';
        const mine = ++sequence;
        const session = state.session;
        attach.load(projectId(), session).then(loaded => {
          if (mine !== sequence || !sameContext() || state.session !== session) return;
          items = loaded; status = ''; paint();
        }, () => {
          if (mine !== sequence || !sameContext() || state.session !== session) return;
          items = []; status = 'failed'; paint();
        });
      }
      paint();
    }

    async function choose(index) {
      const row = rows[index];
      if (!ready() || !sameContext() || !row || reason(row)) return;
      const { session, project, token } = active;
      const input = inputOf();
      const currentToken = tokenAt(input);
      if (!currentToken || currentToken.start !== token.start || currentToken.end !== token.end
        || currentToken.query !== token.query) { update(); return; }
      const before = input.value;
      saving = true; paint();
      const added = await attach.ensure(project, session, row.source, row.id);
      saving = false;
      if (state.session !== session || projectId() !== project) return;
      const current = inputOf();
      const focused = current && document.activeElement === current;
      let caret = current && current.selectionStart;
      // A host write can finish after the researcher has edited the message.
      if (added && current && current.value === before) {
        current.value = before.slice(0, token.start) + before.slice(token.end);
        state.draft = current.value;
        caret = token.start;
        current.setSelectionRange(caret, caret);
      }
      if (added) close();
      render(true);
      if (focused) {
        const restored = inputOf();
        if (restored) { restored.focus(); restored.setSelectionRange(caret, caret); }
      }
    }

    function keydown(event) {
      if (composing || event.isComposing || event.keyCode === 229) return;
      const menu = menuOf();
      if (!sameContext() || !menu || !menu.open || !ready()) return;
      if (event.target !== inputOf() && !event.target.closest?.('[data-gpi-slash-row]')) return;
      if (event.key === 'Escape') { event.preventDefault(); close(); return; }
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        const enabled = rows.map((row, index) => reason(row) ? -1 : index).filter(index => index >= 0);
        if (!enabled.length) return;
        const at = enabled.indexOf(selected);
        selected = enabled[(at + (event.key === 'ArrowDown' ? 1 : enabled.length - 1)) % enabled.length];
        paint();
        menuOf()?.querySelector(`[data-gpi-slash-row="${selected}"]`)?.scrollIntoView({ block: 'nearest' });
      } else if ((event.key === 'Enter' && !event.shiftKey) || (event.key === 'Tab' && !event.shiftKey)) {
        event.preventDefault(); void choose(selected);
      }
    }
    function wire() {
      state.host.addEventListener('input', update);
      state.host.addEventListener('keydown', keydown);
      state.host.addEventListener('keyup', event => {
        if (active && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) update(event);
      });
      state.host.addEventListener('compositionstart', event => {
        if (event.target === inputOf()) { composing = true; close(); }
      });
      state.host.addEventListener('compositionend', event => {
        if (event.target === inputOf()) { composing = false; update(event); }
      });
      // Keep textarea focus for mouse selection; the shell still receives
      // the press first and closes any other popover.
      state.host.addEventListener('pointerdown', event => {
        if (event.target.closest?.('[data-gpi-slash-row]')) event.preventDefault();
      });
      state.host.addEventListener('click', event => {
        const row = event.target.closest?.('[data-gpi-slash-row]');
        if (row) void choose(Number(row.dataset.gpiSlashRow));
        else if (event.target === inputOf()) update(event);
      });
      state.host.addEventListener('toggle', event => {
        if (event.target === menuOf() && !event.target.open) close();
      }, true);
    }
    function capture() {
      // Native dismissal is synchronous; its toggle event is queued. Honor it
      // before a click action rebuilds the composer and loses the old details.
      const menu = menuOf();
      if (menu && !menu.open) close();
      const input = inputOf();
      selection = active && input ? { start: input.selectionStart, end: input.selectionEnd,
        focused: document.activeElement === input } : null;
    }
    function restore() {
      if (!sameContext() || !ready() || composing) { close(); selection = null; return; }
      const input = inputOf();
      if (selection) {
        input.setSelectionRange(selection.start, selection.end);
        if (selection.focused) input.focus();
        selection = null;
      }
      paint();
    }
    return { wire, capture, restore };
  }
  window.EasyICU.guidedPi.declare('slashMenu', { create });
})();
