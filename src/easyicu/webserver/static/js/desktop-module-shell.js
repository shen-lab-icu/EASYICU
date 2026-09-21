/* Owner: shared desktop shell for top-level EasyICU modules. */
(function () {
  'use strict';

  function tr(en, zh) { return window.t(en, zh); }
  const CONTEXT_KEY = 'easyicu.desktop.workspace-context.v1';
  function text(value, limit) { return String(value || '').trim().slice(0, limit || 200); }

  function rememberContext(value) {
    const row = value && typeof value === 'object' ? value : {};
    const context = {
      projectId: text(row.projectId, 200),
      projectTitle: text(row.projectTitle, 200),
      sessionId: text(row.sessionId, 200),
      sessionTitle: text(row.sessionTitle, 240),
    };
    if (!context.projectId) return;
    try { window.sessionStorage.setItem(CONTEXT_KEY, JSON.stringify(context)); } catch (_) {}
  }

  function workspaceContext() {
    try {
      const parsed = JSON.parse(window.sessionStorage.getItem(CONTEXT_KEY) || '{}');
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (_) { return {}; }
  }

  function navItem(active, id, iconName, en, zh) {
    const selected = active === id;
    const route = id === 'projects' ? 'guided' : id;
    return `<button type="button" class="euh-rail-item ${selected ? 'active' : ''}" data-nav="${route}" ${selected ? 'aria-current="page"' : ''} title="${tr(en, zh)}" aria-label="${tr(en, zh)}">${icon(iconName, 19)}<span>${tr(en, zh)}</span></button>`;
  }

  function rail(active) {
    const moduleActive = isDataRoute(active) ? 'extraction' : active;
    return `<nav class="euh-rail" aria-label="${tr('EasyICU navigation', 'EasyICU 导航')}">
      <button type="button" class="euh-brand" data-nav="entry" title="EasyICU" aria-label="EasyICU">${icon('spark', 19)}</button>
      ${navItem(moduleActive, 'projects', 'folder', 'Projects', '项目')}
      ${navItem(moduleActive, 'skills', 'layers', 'Skills', '技能')}
      ${navItem(moduleActive, 'extraction', 'grid', 'Data', '数据')}
      <div class="euh-rail-spacer"></div>
      ${navItem(moduleActive, 'settings', 'gear', 'Settings', '设置')}
      <button type="button" class="euh-rail-item" data-lang-toggle title="${tr('Switch language', '切换语言')}" aria-label="${tr('Switch language', '切换语言')}">${icon('globe', 18)}<span>${window.EU_LANG === 'zh' ? 'EN' : '中'}</span></button>
    </nav>`;
  }

  const DATA_ROUTES = Object.freeze([
    ['extraction', 'extract', 'Extraction', '数据抽取'],
    ['patient', 'patient', 'Patient review', '患者审阅'],
    ['cohort', 'cohort', 'Cohort statistics', '队列统计'],
    ['crossdb', 'benchmark', 'Cross-database', '跨库对比'],
  ]);

  function isDataRoute(route) {
    return DATA_ROUTES.some(([id]) => id === route);
  }

  function dataNav(active, actions) {
    return `<nav class="euh-context-nav" aria-label="${tr('Data workspace sections', '数据工作台分区')}">
      ${DATA_ROUTES.map(([id, iconName, en, zh]) => `<button type="button" data-nav="${id}" class="${active === id ? 'active' : ''}" ${active === id ? 'aria-current="page"' : ''}>${icon(iconName, 15)}<span>${tr(en, zh)}</span></button>`).join('')}
      ${actions ? `<div class="eudata-module-actions">${actions}</div>` : ''}
    </nav>`;
  }

  function moduleNavigation(active, options) {
    if (isDataRoute(active)) return dataNav(active, options.actions || '');
    if (active === 'skills') return `<nav class="euh-context-nav" aria-label="${tr('Skill sections', '技能分区')}">
      <button type="button" class="active" data-sk-mode="skills">${icon('layers', 15)}<span>${tr('Skills', '技能')}</span></button>
      <button type="button" data-sk-mode="methods">${icon('list', 15)}<span>${tr('Method library', '方法库')}</span></button>
    </nav>`;
    if (active === 'settings') return `<nav class="euh-context-nav" aria-label="${tr('Settings sections', '设置分区')}">
      ${[
        ['set-capabilities', 'layers', 'Capabilities', '能力'], ['set-workspace', 'folder', 'Workspace', '工作区'],
        ['set-data-mode', 'flask', 'Data mode', '数据模式'], ['set-privacy', 'shield', 'Privacy', '隐私'],
        ['set-research-agent', 'agent', 'Research Agent', '研究代理'], ['set-language', 'globe', 'Language', '语言'],
        ['set-about', 'help', 'About', '关于'],
      ].map(([id, iconName, en, zh], index) => `<button type="button" class="${index === 0 ? 'active' : ''}" data-settings-jump="${id}">${icon(iconName, 15)}<span>${tr(en, zh)}</span></button>`).join('')}
    </nav>`;
    return '';
  }

  function contextRail(active, options) {
    const context = workspaceContext();
    const projectTitle = text(context.projectTitle || context.projectId) || tr('Local research workspace', '本地研究工作区');
    const sessionTitle = text(context.sessionTitle) || tr('Open the research conversation', '打开研究对话');
    const sectionTitle = isDataRoute(active) ? tr('Data', '数据') : active === 'skills' ? tr('Skills', '技能') : tr('Settings', '设置');
    return `<aside class="euh-context" aria-label="${tr('Project context', '项目上下文')}">
      <header><span>${tr('PROJECT', '项目')}</span><button type="button" data-nav="guided" title="${tr('Back to research', '返回研究工作区')}">${icon('chevron', 13)}</button></header>
      <button type="button" class="euh-context-project" data-nav="guided"><strong>${projectTitle}</strong><small>${sessionTitle}</small></button>
      <div class="euh-context-section"><span>${sectionTitle}</span>${moduleNavigation(active, options)}</div>
      <div class="euh-context-spacer"></div>
      <button type="button" class="euh-context-return" data-nav="guided">${icon('arrow', 14)}<span>${tr('Back to research', '返回研究工作区')}</span></button>
    </aside>`;
  }

  function render(options) {
    const opts = options || {};
    return `<div class="euh-shell ${opts.shellClass || ''}">
      ${rail(opts.active || '')}
      ${contextRail(opts.active || '', opts)}
      <main class="${opts.mainClass || 'euh-main'}" aria-label="${opts.label || tr('Page content', '页面内容')}">
        <div class="${opts.innerClass || 'euh-main-inner'}">${opts.content || ''}</div>
      </main>
    </div>`;
  }

  function renderData(options) {
    const opts = options || {};
    const innerClass = `eudata-main-inner${opts.wide ? ' eudata-main-inner-wide' : ''}`;
    return render({
      active: isDataRoute(opts.active) ? opts.active : 'extraction',
      shellClass: `eudata-shell ${opts.shellClass || ''}`,
      mainClass: 'eudata-main',
      innerClass,
      label: opts.label || tr('Data workspace', '数据工作台'),
      actions: opts.actions || '',
      content: `<div class="eudata-module-content">${opts.content || ''}</div>`,
    });
  }

  window.EU_DESKTOP_MODULE_SHELL = Object.freeze({ rail, render, renderData, isDataRoute, rememberContext, workspaceContext });
})();
