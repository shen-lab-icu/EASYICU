/* Owner: Guided Pi conversation header widget. */
/* Guided Copilot conversation header owner. Keeps the primary session controls
   visible and groups infrequent actions without changing their permissions. */
(function () {
  'use strict';

  /* A new conversation inherits its project's name, so the kicker and the
     title were printing the same string twice, eight pixels apart. The session
     title also carries a mode suffix ("New local study · 研究"), so an exact
     match is not enough -- name the project above the title only when the
     title does not already contain it. */
  function kickerText(options) {
    const project = String(options.projectTitle || '').trim();
    return project || 'EasyICU';
  }

  function renderModelControl(options) {
    const { tr, esc, icon } = options;
    // D-P3-3: truncated model label keeps its full text in title (button title
    // stays the help copy; the inner span carries the connection label).
    return `<button class="gpi-model-binding" type="button" data-gpi-config title="${esc(tr('Change model connection; changes apply to a new conversation', '更改模型连接；变更将在新会话中生效'))}" aria-label="${esc(tr('Change model connection', '更改模型连接'))}"><span title="${esc(options.connectionLabel)}">${esc(options.connectionLabel)}</span>${icon('chevdown', 12)}</button>`;
  }

  function render(options) {
    const { tr, esc, icon } = options;
    return `<header class="gpi-head">
      <div class="gpi-head-title"><span class="gpi-kicker" title="${esc(kickerText(options))}">${esc(kickerText(options))}</span><span class="gpi-head-separator" aria-hidden="true">/</span><span class="gpi-title" title="${esc(options.sessionTitle)}"><span class="gpi-session-title-text">${esc(options.sessionTitle)}</span></span><span class="gpi-live" role="status" aria-live="polite">${options.busy ? tr('working', '工作中') : tr('ready', '就绪')}</span></div>
      <div class="gpi-head-meta">
        <button class="gpi-head-new" type="button" data-gpi-new>${icon('plus', 13)} ${tr('New conversation', '新会话')}</button>
        <details class="gpi-layout-control">
          <summary>${icon('grid', 13)} ${tr('Layout', '布局')}</summary>
          <div class="gpi-layout-popover" role="group" aria-label="${tr('Visible workspace panels', '工作区显示面板')}">
            <strong>${tr('Workspace panels', '工作区面板')}</strong>
            ${[
              ['progress', tr('To-dos', '待办')],
              ['results', tr('Results', '成果')],
              ['compute', tr('Compute', '计算')],
              ['notes', tr('Notes', '笔记')],
            ].map(([key, label]) => `<button type="button" data-gpi-layout-toggle="${key}" aria-label="${esc(label)}" aria-pressed="${Boolean(options.layout && options.layout[key])}"><span>${esc(label)}</span><span aria-hidden="true">${options.layout && options.layout[key] ? '✓' : ''}</span></button>`).join('')}
          </div>
        </details>
        <details class="gpi-head-overflow">
          <summary>${tr('More', '更多')}<span aria-hidden="true">⌄</span></summary>
          <div class="gpi-head-overflow-menu" role="menu">
            <div class="gpi-mode-switch" role="group" aria-label="${tr('Agent mode', 'Agent 模式')}">
              <button type="button" data-gpi-mode-switch="research" aria-pressed="${!options.workspace}">${tr('Research', '研究')}</button>
              <button type="button" data-gpi-mode-switch="workspace" aria-pressed="${options.workspace}">${tr('Workspace', '工作区')}</button>
            </div>
            <button type="button" role="menuitem" data-gpi-study-setup>${tr('Study setup', '研究配置')}</button>
            <button type="button" role="menuitem" data-gpi-presentation-pin aria-pressed="${options.pinned ? 'true' : 'false'}">${options.pinned ? tr('Remove from presentation', '取消保留演示') : tr('Save for presentation', '保留演示')}</button>
            <button type="button" role="menuitem" data-gpi-demo>${icon('play', 13)} ${tr('Reviewer demo', '审稿流程演示')}</button>
          </div>
        </details>
      </div>
    </header>`;
  }

  window.EasyICU.guidedPi.declare('header', { render, renderModelControl });
})();
