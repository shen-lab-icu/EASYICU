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
    const session = String(options.sessionTitle || '').trim();
    if (!project || (session && session.indexOf(project) >= 0)) return 'EASYICU COPILOT';
    return `EASYICU COPILOT · ${project}`;
  }

  function render(options) {
    const { tr, esc, icon } = options;
    return `<header class="gpi-head">
      <div class="gpi-head-title"><div class="gpi-kicker">${esc(kickerText(options))}</div><div class="gpi-title" title="${esc(options.sessionTitle)}">${esc(options.sessionTitle)} <span class="gpi-live" role="status" aria-live="polite">${options.busy ? tr('working', '工作中') : tr('ready', '就绪')}</span></div></div>
      <div class="gpi-head-meta">
        <div class="gpi-mode-switch" role="group" aria-label="${tr('Agent mode', 'Agent 模式')}">
          <button type="button" data-gpi-mode-switch="research" aria-pressed="${!options.workspace}">${tr('Research', '研究')}</button>
          <button type="button" data-gpi-mode-switch="workspace" aria-pressed="${options.workspace}">${tr('Workspace', '工作区')}</button>
        </div>
        <button class="gpi-model-binding" type="button" data-gpi-config title="${esc(tr('Change model connection; changes apply to a new conversation', '更改模型连接；变更将在新会话中生效'))}" aria-label="${esc(tr('Change model connection', '更改模型连接'))}"><span>${esc(options.connectionLabel)}</span>${icon('chevdown', 12)}</button>
        <button class="gpi-head-new" type="button" data-gpi-new>${icon('plus', 13)} ${tr('New conversation', '新会话')}</button>
        <details class="gpi-head-overflow">
          <summary>${tr('More', '更多')}<span aria-hidden="true">⌄</span></summary>
          <div class="gpi-head-overflow-menu" role="menu">
            <button type="button" role="menuitem" data-gpi-study-setup>${tr('Study setup', '研究配置')}</button>
            <button type="button" role="menuitem" data-gpi-presentation-pin aria-pressed="${options.pinned ? 'true' : 'false'}">${options.pinned ? tr('Remove from presentation', '取消保留演示') : tr('Save for presentation', '保留演示')}</button>
            <button type="button" role="menuitem" data-gpi-demo>${icon('play', 13)} ${tr('Reviewer demo', '审稿流程演示')}</button>
          </div>
        </details>
      </div>
    </header>`;
  }

  window.EU_GUIDED_PI_HEADER = { render };
})();
