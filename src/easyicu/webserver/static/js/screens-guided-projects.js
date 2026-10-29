/* Owner: Guided project picker widget. */
/* Guided Copilot project/folder picker rendering.
   Owns local study-folder rendering, server defaults and creation feedback;
   screens-guided.js wires conversation state and events through a context. */
(function () {
  const PROJECT_RAIL_COLLAPSED_KEY = 'easyicu.guided.projectRailCollapsed.v1';
  let projectManagementActive = false;
  let projectRailCollapsed = readProjectRailCollapsed();
  const selectedProjectIds = new Set();
  let folderCreation = { busy: false, error: null };
  // A new local project is named in place at the foot of the project list
  // (the folder takes this name); null while no name is being typed.
  let projectNaming = null;

  // The hosted-instance owner (screens-guided-pi-hosted.js), when it is
  // loaded and the server reports a hosted instance.
  function hostedOwner() {
    const owner = window.EasyICU?.guidedPi?.optional('hosted');
    return owner && owner.active() ? owner : null;
  }

  function defaultParentDir(ctx) {
    return String((!hostedOwner() && window.EU_SETTINGS?.project_dir) || ctx.guidedDrafts?.data?.default_parent_dir || '').trim();
  }

  function parentPathLabel(ctx) {
    return ctx.compactPath(defaultParentDir(ctx)) || ctx.t('Default study folder', '默认研究目录');
  }

  // A known code reads in the interface language alone; the server's own
  // reason (an OS error, an unknown code) is kept as a secondary detail line.
  function folderCreateError(ctx) {
    const error = folderCreation.error;
    if (!error) return null;
    const { t } = ctx;
    const message = error.error === 'guided_project_folder_create_failed'
      ? t('Could not create the study folder. Check that this folder can be written to.', '无法创建研究文件夹，请检查该目录能否写入。')
      : t('Could not create the study folder.', '无法创建研究文件夹。');
    return { message, detail: String(error.reason || error.error || '') };
  }

  function renderCreationState(ctx) {
    renderFolderDialog(ctx);
    renderProjectRail(ctx);
  }

  async function createDraft(payload, getContext, onCreated) {
    if (folderCreation.busy) return;
    folderCreation = { busy: true, error: null };
    renderCreationState(getContext());
    try {
      if (!window.EU_API?.createGuidedDraft) {
        throw new Error(getContext().t('Local folder creation API is unavailable.', '本地文件夹创建 API 不可用。'));
      }
      const result = await window.EU_API.createGuidedDraft(payload);
      if (!result || result.ok === false || !result.draft) {
        folderCreation.error = result || { error: 'guided_draft_create_failed' };
        return;
      }
      projectNaming = null;
      return await onCreated(result);
    } catch (err) {
      folderCreation.error = { reason: String(err.message || err) };
    } finally {
      folderCreation.busy = false;
      renderCreationState(getContext());
    }
  }

  // Opens the project list with a name field at its foot, focused and
  // selected so typing replaces the default title.
  function startProjectNaming(ctx, seedTitle) {
    projectNaming = { title: String(seedTitle || defaultStudyTitle(ctx)) };
    folderCreation.error = null;
    renderProjectRail(ctx);
    const picker = document.querySelector('#gdResearchProjectRail .gd-project-picker');
    if (picker) picker.open = true;
    const input = document.querySelector('[data-project-new-title]');
    if (input) { input.focus(); if (input.select) input.select(); }
  }

  function stopProjectNaming() {
    projectNaming = null;
    folderCreation.error = null;
  }

  function projectFooterHtml(ctx) {
    const { t, icon, esc, attr } = helpers(ctx);
    if (hostedOwner() || projectManagementActive) return '';
    const error = folderCreateError(ctx);
    const errorHtml = error ? `<p class="gd-project-new-error" role="alert">${esc(error.message)}${error.detail ? `<small>${esc(error.detail)}</small>` : ''}</p>` : '';
    if (projectNaming) {
      const busy = folderCreation.busy;
      return `<form class="gd-project-new" data-project-new-form>
        <input data-project-new-title value="${attr(projectNaming.title)}" maxlength="80" autocomplete="off" aria-label="${t('Project name', '项目名称')}" ${busy ? 'disabled' : ''}>
        <button class="btn primary sm" type="submit" ${busy ? 'disabled' : ''} aria-busy="${busy}">${busy ? t('Creating…', '正在新建…') : t('Create', '新建')}</button>
      </form>${errorHtml}`;
    }
    return `<div class="gd-project-foot">
      <button class="gd-project-foot-item" type="button" data-project-new>${icon('plus', 14)}<span>${t('New project', '新建项目')}</span></button>
      <button class="gd-project-foot-item" type="button" data-folder-choice="open">${icon('folder', 14)}<span>${t('Open existing project…', '打开已有项目…')}</span></button>
    </div>${errorHtml}`;
  }

  function readProjectRailCollapsed() {
    try {
      return !!(window.localStorage && window.localStorage.getItem(PROJECT_RAIL_COLLAPSED_KEY) === '1');
    } catch (_) {
      return false;
    }
  }

  function setProjectRailCollapsed(collapsed) {
    projectRailCollapsed = !!collapsed;
    try {
      if (window.localStorage) window.localStorage.setItem(PROJECT_RAIL_COLLAPSED_KEY, projectRailCollapsed ? '1' : '0');
    } catch (_) {}
  }

  function isProjectRailCollapsed() {
    return projectRailCollapsed;
  }

  // The global nav's Projects button: from the compact rail it first brings
  // the project panel back; otherwise it toggles the project switcher.
  function showProjects() {
    const collapsedMain = document.querySelector('.gd-main.gd-project-rail-collapsed');
    if (collapsedMain) {
      setProjectRailCollapsed(false);
      collapsedMain.classList.remove('gd-project-rail-collapsed');
      return;
    }
    const picker = document.querySelector('#gdResearchProjectRail .gd-project-picker');
    if (!picker) return;
    picker.open = !picker.open;
    const summary = picker.querySelector('summary');
    if (summary && picker.open) summary.focus();
  }

  function setProjectManagement(active) {
    projectManagementActive = !!active;
    if (!projectManagementActive) selectedProjectIds.clear();
  }

  function isProjectManagementActive() {
    return projectManagementActive;
  }

  function toggleProjectSelection(row, checked) {
    const id = row && row.id;
    if (!id) return;
    if (checked) selectedProjectIds.add(id);
    else selectedProjectIds.delete(id);
  }

  function selectAllProjects(rows, activeId, checked) {
    (rows || []).forEach(row => {
      if (!row || !row.id || row.id === activeId) return;
      if (checked) selectedProjectIds.add(row.id);
      else selectedProjectIds.delete(row.id);
    });
  }

  function selectedProjects(rows, activeId) {
    return (rows || []).filter(row => row && row.id !== activeId && selectedProjectIds.has(row.id));
  }

  function helpers(ctx) {
    return {
      t: ctx.t,
      icon: ctx.icon,
      esc: ctx.esc,
      attr: ctx.attr,
      compactPath: ctx.compactPath,
      fmtRunTime: ctx.fmtRunTime,
    };
  }

  // D-P2-1: defensive label projection. The owner loads first (see
  // index.html), but a truncated bundle or a static preview without it must
  // not throw here — fall back to bounded raw text instead.
  function projectTitle(value, fallback) {
    return window.EU_PRODUCT_LABELS?.projectTitle?.(value, fallback)
      ?? String(value ?? fallback ?? '').slice(0, 200);
  }

  // D-P2-1: one shared row-title fallback. 'Guided study' is kept (not
  // 'Guided project') so the rail, the known-folder picker and the removal
  // dialog project the same fallback shape the run-files owner uses
  // (stored question, else a single shared default). Do not invent new
  // per-callsite defaults — extend PROJECT_DEFAULT_TITLES instead.
  const GUIDED_ROW_FALLBACK = ['Guided study', '研究项目'];

  // Plans to review and finished runs (owner: screens-guided-pi-attention.js).
  function attentionMark(projectId) {
    const owner = window.EasyICU && window.EasyICU.guidedPi ? window.EasyICU.guidedPi.optional('attention') : null;
    return owner ? owner.projectMarkHtml(String(projectId || '')) : '';
  }
  function projectMeta(row, t) {
    // A Guided draft remains metadata-only storage even after Copilot binds a real
    // StudyContext and Research Agent run.  Do not expose that persistence
    // implementation as the project's scientific lifecycle status.
    const configurationMissing = row.configuration_health
      && row.configuration_health.status === 'configuration_missing';
    const status = configurationMissing
      ? t('Configuration expired', '配置已失效')
      : row.workflow_status === 'analysis_review'
      ? t('Results to review', '结果待审阅')
      : row.workflow_status === 'plan_review'
        ? t('Plan to review', '计划待审阅')
        : row.workflow_status === 'analysis_running'
          ? t('Analysis running', '分析中')
          : row.workflow_status === 'configured'
            ? t('Data selected', '已选择数据')
            : row.status === 'metadata_only'
              ? t('New project', '新项目')
              : (row.status === 'ready' ? t('Ready', '已就绪') : (row.status || t('Study', '研究')));
    const mode = row.data_mode === 'demo'
      ? t('Demo data', '演示数据')
      : (row.data_mode === 'real'
        // Every hosted project uses data its administrator prepared.
        ? (hostedOwner() ? '' : t('Local data', '本地数据'))
        : (row.data_mode === 'unbound' ? t('No data selected', '未选择数据') : (row.data_mode || '')));
    return [status, mode].filter(Boolean).join(' · ');
  }

  function renderShellRail(ctx) {
    const { t, icon } = helpers(ctx);
    return `
      <button class="gd-rail-restore" type="button" data-project-rail-toggle aria-label="${t('Show research projects', '显示研究项目栏')}" title="${t('Show research projects', '显示研究项目栏')}">${icon('chevron', 14)}</button>
      <aside class="gd-rail" id="gdResearchProjectRail">
        <nav class="gpi-global-nav" aria-label="${t('EasyICU navigation', 'EasyICU 导航')}">
          <button type="button" class="gpi-global-brand" data-open="entry" aria-label="${t('New conversation', '新建对话')}" title="${t('New conversation in this project', '在当前项目新建对话')}">${icon('spark', 19)}</button>
          <button type="button" class="gpi-global-item active" data-gpi-show-projects aria-current="page" title="${t('Research projects', '研究项目')}">${icon('folder', 19)}<span>${t('Projects', '项目')}</span></button>
          <button type="button" class="gpi-global-item" data-open="skills" title="${t('Research skills', '研究技能')}">${icon('layers', 19)}<span>${t('Skills', '技能')}</span></button>
          <button type="button" class="gpi-global-item" data-open="extraction" title="${t('Data workspace', '数据工作台')}">${icon('grid', 19)}<span>${t('Data', '数据')}</span></button>
          <div class="gpi-global-spacer"></div>
          <div data-update-slot></div>
          <button type="button" class="gpi-global-item" data-open="settings" title="${t('Settings', '设置')}">${icon('gear', 18)}<span>${t('Settings', '设置')}</span></button>
          <button type="button" class="gpi-global-item" data-lang-toggle title="${t('Switch language', '切换语言')}">${icon('globe', 18)}<span>${window.EU_LANG === 'zh' ? 'EN' : '中'}</span></button>
        </nav>
        <div class="gd-rail-top">
          <div class="gd-rail-heading">
            <div class="gd-rail-list" id="gdSessions"></div>
            <button class="gd-rail-collapse" type="button" data-project-rail-toggle aria-controls="gdResearchProjectRail" aria-label="${t('Hide research projects', '隐藏研究项目栏')}" title="${t('Hide research projects', '隐藏研究项目栏')}">${icon('chevron', 14)}</button>
          </div>
          <div class="gd-folder-controls" id="gdFolderControls"></div>
        </div>
        <section id="gdConversationRail" class="gpi-conversations" hidden></section>
        <div class="gd-rail-foot">
          <div class="gd-rail-utils" aria-label="${t('Guided Copilot utilities', '研究引导工具')}">
            <button class="gd-utilbtn" type="button" data-open="entry" title="${t('New conversation in this project', '在当前项目新建对话')}" aria-label="${t('New conversation', '新建对话')}">${icon('plus', 14)}</button>
            <button class="gd-utilbtn" type="button" data-open="settings" title="${t('Settings', '设置')}" aria-label="${t('Settings', '设置')}">${icon('gear', 14)}</button>
            <button class="gd-utilbtn lang" type="button" data-lang-toggle title="${t('Switch language', '切换语言')}" aria-label="${t('Switch language', '切换语言')}">
              ${icon('globe', 14)} <span>${window.EU_LANG === 'zh' ? 'EN' : '中'}</span>
            </button>
          </div>
          <button class="btn sm block gd-data-workspace" data-open="extraction">${icon('grid', 13)} ${t('Data workspace', '数据工作台')}</button>
          <button class="btn sm block gd-data-workspace" type="button" data-open="skills">${icon('layers', 13)} ${t('Research skills', '研究技能')}</button>
        </div>
      </aside>`;
  }

  function renderProjectRail(ctx) {
    const { t, icon, esc, fmtRunTime } = helpers(ctx);
    const host = document.getElementById('gdSessions');
    if (!host) return;
    const rows = ctx.localDraftRows();
    const activeId = ctx.selectedGuidedDraft && ctx.selectedGuidedDraft.id;
    const rowIds = new Set(rows.map(row => row && row.id).filter(Boolean));
    Array.from(selectedProjectIds).forEach(id => { if (!rowIds.has(id) || id === activeId) selectedProjectIds.delete(id); });
    const selectedRows = selectedProjects(rows, activeId);
    const eligibleCount = rows.filter(row => row && row.id && row.id !== activeId).length;
    const external = ctx.selectedGuidedDraft
      && activeId
      && !rows.some(row => row.id === activeId)
      ? ctx.selectedGuidedDraft
      : null;
    const externalHtml = external ? `
      <div class="gd-sessline">
        <div class="gd-sess draft active" aria-current="true">
          <span class="gd-sess-status" aria-hidden="true"></span>
          <span class="gd-sess-body"><span class="ss-t">${esc(projectTitle(external.title, external.question || external.id))}</span><span class="ss-m">${external.study_context_id ? `${t('Bound StudyContext', '已绑定 StudyContext')} · r${esc(external.study_context_revision == null ? '—' : external.study_context_revision)}` : t('Project selected · setup continues here', '项目已选择 · 在此继续配置')}</span></span>
          <span class="ss-time">${t('current', '当前')}</span>
        </div>
      </div>` : '';
    const draftHtml = ctx.guidedDrafts.loading
      ? `${externalHtml}<div class="gd-empty-local"><div class="ss-t">${t('Loading study folders', '正在加载研究文件夹')}</div><div class="ss-m">${t('Reading the project list.', '正在读取项目列表。')}</div></div>`
      : ctx.guidedDrafts.error
        ? `${externalHtml}<div class="gd-empty-local warn"><div class="ss-t">${t('Study folders unavailable', '研究文件夹不可用')}</div><div class="ss-m">${esc(ctx.guidedDrafts.error)}</div></div>`
        : rows.length
          ? externalHtml + rows.map((row, i) => {
            const active = activeId === row.id;
            const selected = selectedProjectIds.has(row.id);
            const selectable = !active;
            const configurationMissing = row.configuration_health
              && row.configuration_health.status === 'configuration_missing';
            const title = projectTitle(row.title, row.question || t(...GUIDED_ROW_FALLBACK));
            const meta = projectMeta(row, t);
            const time = fmtRunTime(row.updated_at || row.created_at);
            return `
            <div class="gd-sessline ${projectManagementActive ? 'is-managing' : ''} ${selected ? 'is-selected' : ''}" data-project-search-row="${esc(title.toLocaleLowerCase())}">
              ${projectManagementActive ? `<label class="gd-sess-select" title="${active ? t('The current project cannot be selected', '当前正在使用的项目不能勾选') : t('Select project', '选择项目')}"><input type="checkbox" data-select-localdraft="${i}" ${selected ? 'checked' : ''} ${selectable ? '' : 'disabled'} /><span aria-hidden="true"></span></label>` : ''}
              <button class="gd-sess draft ${active ? 'active' : ''} ${configurationMissing ? 'configuration-missing' : ''}" data-localdraft="${i}" data-project-id="${esc(row.id)}" title="${configurationMissing ? t('Configuration is missing; open recovery options', '研究配置已失效；打开恢复选项') : t('Open research project', '打开研究项目')}: ${esc(title)}" ${active ? 'aria-current="true"' : ''}>
                <span class="gd-sess-status" aria-hidden="true"></span>
                <span class="gd-sess-body"><span class="ss-t">${esc(title)}</span><span class="ss-m">${esc(meta)}</span></span>
                ${attentionMark(row.id)}<span class="ss-time">${esc(time)}</span>
              </button>
              ${projectManagementActive ? '' : `<button class="gd-sess-action danger" type="button" data-remove-localdraft="${i}" title="${t('Remove from project list', '从项目列表移除')}" aria-label="${t('Remove from project list', '从项目列表移除')}: ${esc(title)}">${icon('close', 12)}</button>`}
            </div>`;
          }).join('')
          : (externalHtml || `<div class="gd-empty-local"><div class="ss-t">${t('No projects yet', '还没有研究项目')}</div></div>`);
    const oldPicker = host.querySelector('.gd-project-picker');
    const search = oldPicker && typeof oldPicker.querySelector === 'function'
      && oldPicker.dataset.projectId === (activeId || '')
      ? String(oldPicker.querySelector('[data-project-search]')?.value || '') : '';
    // The project catalog is a selector, not the default reading surface. An
    // early loading render must not leave every project expanded after URL restore.
    const pickerOpen = projectManagementActive || (!activeId && !ctx.guidedDrafts.loading)
      || !!(oldPicker && oldPicker.dataset.projectId === activeId && oldPicker.open);
    const pickerTitle = projectTitle(ctx.selectedGuidedDraft && ctx.selectedGuidedDraft.title, t('Choose a project', '选择研究项目'));
    // One control names the project the rail is showing and switches it —
    // the reference's "PROJECT / name" combobox. A research project is the
    // study itself (question, data binding, plan, runs, evidence, results,
    // one local folder); the conversations listed under it are threads
    // about that one study, not a second kind of work item.
    // As a switcher it is a floating menu (closes on an outside press); with
    // no project chosen, or while managing projects, the list is the surface.
    const pickerDismissible = !!activeId && !projectManagementActive;
    host.innerHTML = `<details class="gd-project-picker${projectManagementActive ? ' is-managing' : ''}" data-project-id="${esc(activeId || '')}"${pickerOpen ? ' open' : ''}${pickerDismissible ? ' data-popover-menu' : ''}>
      <summary title="${esc(t('Switch research project', '切换研究项目'))}"><small>${t('Project', '项目')}</small><span title="${esc(pickerTitle)}">${esc(pickerTitle)}</span><i aria-hidden="true">⌄</i></summary>
      <div class="gd-project-picker-list">
      <div class="gd-project-heading"><span>${t('Research projects', '研究项目')}</span><span class="gd-project-heading-actions"><button class="gd-manage-mini ${projectManagementActive ? 'active' : ''}" type="button" data-project-manage>${projectManagementActive ? t('Done', '完成') : t('Manage', '管理')}</button><button class="gd-refresh-mini" type="button" data-refreshdrafts title="${t('Refresh research projects', '刷新研究项目')}" aria-label="${t('Refresh research projects', '刷新研究项目')}">${icon('refresh', 12)}</button></span></div>
      ${rows.length ? `<label class="gd-project-search"><span class="shell-sr-only">${t('Search research projects', '搜索研究项目')}</span><input type="search" data-project-search placeholder="${t('Search projects…', '搜索项目…')}" value="${esc(search)}" autocomplete="off"></label>` : ''}
      ${draftHtml}
      ${rows.length ? `<p class="gd-project-search-empty" data-project-search-empty hidden>${t('No matching projects', '没有匹配的项目')}</p>` : ''}
      ${projectFooterHtml(ctx)}
      ${projectManagementActive ? `<div class="gd-project-selection-bar"><label><input type="checkbox" data-select-all-projects ${eligibleCount && selectedRows.length === eligibleCount ? 'checked' : ''} ${eligibleCount ? '' : 'disabled'} /> ${t('Select all', '全选')}</label><span><strong>${selectedRows.length}</strong> ${t('selected; current project excluded', '项已选；当前项目不参与')}</span><button class="btn sm danger" type="button" data-remove-selected-projects ${selectedRows.length ? '' : 'disabled'}>${t('Remove selected', '移除所选')}</button></div>` : ''}</div></details>`;
    const filterRows = () => {
      const needle = String(host.querySelector('[data-project-search]')?.value || '').trim().toLocaleLowerCase();
      let visible = 0;
      host.querySelectorAll('[data-project-search-row]').forEach(row => {
        row.hidden = !!needle && !row.dataset.projectSearchRow.includes(needle);
        if (!row.hidden) visible++;
      });
      const empty = host.querySelector('[data-project-search-empty]');
      if (empty) empty.hidden = visible !== 0;
    };
    host.oninput = event => {
      if (event.target.matches('[data-project-search]')) filterRows();
      if (event.target.matches('[data-project-new-title]') && projectNaming) projectNaming.title = event.target.value;
    };
    host.onsubmit = event => {
      if (!event.target.matches('[data-project-new-form]')) return;
      event.preventDefault();
      const input = event.target.querySelector('[data-project-new-title]');
      const title = String(input && input.value || '').trim() || defaultStudyTitle(ctx);
      if (projectNaming) projectNaming.title = title;
      if (typeof ctx.createProject === 'function') return ctx.createProject(title);
    };
    // Escape leaves the name field without closing the project list.
    host.onkeydown = event => {
      if (event.key !== 'Escape' || !event.target.matches('[data-project-new-title]') || folderCreation.busy) return;
      event.preventDefault();
      event.stopPropagation();
      stopProjectNaming();
      renderProjectRail(ctx);
      host.querySelector('[data-project-new]')?.focus();
    };
    if (search) filterRows();
  }

  function renderFolderControls(ctx) {
    const { t, icon } = helpers(ctx);
    const host = document.getElementById('gdFolderControls');
    if (!host) return;
    const hosted = hostedOwner();
    if (hosted) {
      // Hosted: one action, a new project in the clinician's own study folder.
      host.innerHTML = `
        <div class="gd-folder-picker">
          <button class="gd-newbtn" type="button" data-folder-choice="new">${icon('plus', 14)} <span>${t('New research project', '新建研究项目')}</span></button>
        </div>`;
      return;
    }
    // Local projects are created and opened from the project list's foot.
    host.innerHTML = '';
  }

  function renderFolderBrowser(ctx) {
    const { t, icon, esc, compactPath } = helpers(ctx);
    const browser = ctx.guidedFolderBrowser || {};
    if (!browser.open) return '';
    const data = browser.data || {};
    const entries = Array.isArray(data.entries) ? data.entries : [];
    const shortcuts = Array.isArray(data.shortcuts) ? data.shortcuts : [];
    const currentPath = data.path || browser.path || '';
    const parent = data.parent || '';
    const failed = browser.error || (data && data.ok === false ? data.error : '');
    return `
      <div class="gds-browser" data-guided-folder-browser>
        <div class="gds-browser-head">
          <strong>${t('Folder picker', '文件夹选择器')}</strong>
          <button class="btn sm ghost" type="button" data-folder-browser-close>${icon('close', 12)}</button>
        </div>
        <div class="gds-browser-path"><span>${t('Current', '当前')}</span><code>${esc(compactPath(currentPath) || t('Home folder', '主目录'))}</code></div>
        ${shortcuts.length ? `<div class="gds-browser-shortcuts">${shortcuts.map((item, i) => `
          <button class="btn sm" type="button" data-folder-browser-shortcut="${i}">${esc(item.name || 'Folder')}</button>`).join('')}</div>` : ''}
        ${failed ? `<div class="gds-browser-message warn">${icon('info', 12)} <span>${esc(String(failed))}</span></div>` : ''}
        <div class="gds-browser-list">
          ${browser.loading ? `<div class="gds-browser-empty">${icon('refresh', 13)} ${t('Loading folders...', '正在加载文件夹...')}</div>` : ''}
          ${!browser.loading && !entries.length ? `<div class="gds-browser-empty">${t('No folders inside.', '这里没有下级文件夹。')}</div>` : ''}
          ${!browser.loading && entries.map((entry, i) => `
            <button class="gds-browser-row" type="button" data-folder-browser-entry="${i}">
              <span class="gds-ico">${icon('folder', 13)}</span>
              <span><strong>${esc(entry.name || 'Folder')}</strong><code>${esc(compactPath(entry.path || ''))}</code></span>
              ${entry.hint ? `<em>${esc(entry.hint)}</em>` : ''}
            </button>`).join('')}
        </div>
        <div class="gds-browser-actions">
          <button class="btn sm" type="button" data-folder-browser-up ${parent ? '' : 'disabled'}>${icon('back', 12)} ${t('Up', '上一级')}</button>
          <span class="grow"></span>
          <button class="btn primary sm" type="button" data-folder-browser-use ${currentPath ? '' : 'disabled'}>${icon('check', 12)} ${t('Use this folder', '选择此文件夹')}</button>
        </div>
      </div>`;
  }

  function renderFolderDialog(ctx) {
    const { t, icon } = helpers(ctx);
    const host = document.getElementById('gdFolderDialogHost');
    if (!host) return;
    if (!ctx.guidedFolderDialogMode) {
      host.innerHTML = '';
      return;
    }
    if (hostedOwner()) {
      const branchCfg = ctx.BRANCH && ctx.BRANCH[ctx.branch];
      const title = ctx.guidedFolderSeedTitle || (branchCfg && branchCfg.chip) || defaultStudyTitle(ctx);
      host.innerHTML = hostedProjectDialog(ctx, title, folderCreation.busy, folderCreateError(ctx));
      return;
    }
    // Local: open an existing EasyICU project, or an export folder to review.
    const openForReview = ctx.pendingGuidedGoal && ctx.pendingGuidedGoal.goal === 'review_data';
    const openActions = openForReview ? `
      <button class="btn primary sm" data-reviewexportfolder>${icon('eye', 13)} ${t('Review extracted data', '审阅已提取数据')}</button>
      <button class="btn sm" data-openprojectfolder>${icon('folder', 13)} ${t('Open project', '打开项目')}</button>` : `
      <button class="btn primary sm" data-openprojectfolder>${icon('folder', 13)} ${t('Open project', '打开项目')}</button>`;
    host.innerHTML = `
      <div class="gd-folder-backdrop" data-folder-dialog-close></div>
      <section class="gd-folder-dialog" data-folder-dialog data-draft-setup role="dialog" aria-modal="true" aria-label="${t('Open existing project', '打开已有项目')}">
        <div class="gd-folder-dialog-head">
          <span class="gds-ico">${icon('folder', 15)}</span>
          <div><strong>${openForReview ? t('Choose a folder', '选择文件夹') : t('Open existing project', '打开已有项目')}</strong></div>
          <button class="gd-folder-close" type="button" data-folder-dialog-close aria-label="${t('Close', '关闭')}">×</button>
        </div>
        <div class="gds-choice">
          <div class="gds-path-row">
            <label class="gds-field"><span>${t('Folder', '文件夹')}</span><input data-existing-project-dir autocomplete="off" /></label>
            <button class="btn sm" type="button" data-browseprojectfolder>${icon('folder', 13)} ${t('Browse…', '浏览…')}</button>
          </div>
          ${renderFolderBrowser(ctx)}
          <div class="row gap-8">${openActions}<button class="btn sm" data-folder-dialog-close>${t('Cancel', '取消')}</button></div>
          <div class="gds-status" data-project-open-status hidden></div>
        </div>
      </section>`;
  }

  // A hosted instance keeps every project in the clinician's own study
  // folder on the server (screens-guided-pi-hosted.js): the dialog asks for a
  // name only, with no server folder, path or existing-folder choice.
  function hostedProjectDialog(ctx, title, busy, error) {
    const { t, icon, esc, attr } = helpers(ctx);
    return `
      <div class="gd-folder-backdrop" data-folder-dialog-close></div>
      <section class="gd-folder-dialog gd-folder-dialog-hosted" data-folder-dialog data-draft-setup data-hosted-project-dialog role="dialog" aria-modal="true" aria-label="${t('New research project', '新建研究项目')}">
        <div class="gd-folder-dialog-head">
          <span class="gds-ico">${icon('folder', 15)}</span>
          <div>
            <strong>${t('New research project', '新建研究项目')}</strong>
            <span>${t('Each project keeps its own conversations, plans and results.', '每个项目单独保存自己的对话、计划和结果。')}</span>
          </div>
          <button class="gd-folder-close" type="button" data-folder-dialog-close aria-label="${t('Close', '关闭')}">×</button>
        </div>
        <div class="gds-choice">
          <label class="gds-field"><span>${t('Project name', '项目名称')}</span><input data-draft-title value="${attr(title)}" autocomplete="off" /></label>
          ${error ? `<div class="gds-status error" data-folder-create-error role="alert"><span>${esc(error.message)}</span></div>` : ''}
          <div class="row gap-8"><button class="btn primary sm" data-createdraft ${busy ? 'disabled' : ''} aria-busy="${busy}">${icon(busy ? 'refresh' : 'plus', 13)} ${busy ? t('Creating…', '正在新建…') : t('Create project', '新建项目')}</button><button class="btn sm" data-folder-dialog-close>${t('Cancel', '取消')}</button></div>
        </div>
      </section>`;
  }

  function renderDraftRemovalDialog(ctx) {
    const { t, icon, esc, compactPath } = helpers(ctx);
    const host = document.getElementById('gdRemoveDraftDialogHost');
    if (!host) return;
    const state = ctx.guidedDraftRemoval;
    const rows = state && Array.isArray(state.rows) && state.rows.length ? state.rows : (state && state.row ? [state.row] : []);
    const row = rows[0];
    if (!row) {
      host.innerHTML = '';
      return;
    }
    const title = projectTitle(row.title, row.question || t(...GUIDED_ROW_FALLBACK));
    const multiple = rows.length > 1;
    // A hosted project lives in the clinician's workspace on the shared server,
    // whose system trash is outside that workspace: removal only drops the entry.
    const hosted = !!hostedOwner();
    const trashProjectFolder = !hosted && !!state.trashProjectFolder;
    const busy = !!state.busy;
    host.innerHTML = `
      <div class="gd-folder-backdrop" data-remove-draft-close></div>
      <section class="gd-folder-dialog gd-remove-dialog" data-remove-draft-dialog role="dialog" aria-modal="true" aria-label="${t('Remove research project', '移除研究项目')}">
        <div class="gd-folder-dialog-head">
          <span class="gds-ico">${icon('folder', 15)}</span>
          <div><strong>${multiple ? t('Remove selected research projects', '移除所选研究项目') : t('Remove research project', '移除研究项目')}</strong><span>${hosted ? t('The project leaves your list. Its conversations, runs and results are not deleted and stay on the server; ask the administrator if you need one back.', '项目会从你的列表中移除；它的对话、运行和结果不会被删除，仍保存在服务器上。如需找回，请联系管理员。') : t('Choose whether to remove only the EasyICU list entry or also move the selected project folders to the system trash.', '请选择只从 EasyICU 列表移除，还是同时把所选项目文件夹移到系统废纸篓。')}</span></div>
          <button class="gd-folder-close" type="button" data-remove-draft-close aria-label="${t('Close', '关闭')}" ${busy ? 'disabled' : ''}>×</button>
        </div>
        <div class="gds-choice">
          <div class="gds-choice-head"><strong>${multiple ? `${rows.length} ${t('projects selected', '个项目已选择')}` : esc(title)}</strong>${hosted ? '' : `<span>${t('By default, project folders and all files on disk are preserved.', '默认会保留磁盘上的项目文件夹和全部文件。')}</span>`}</div>
          ${hosted ? '' : `<label class="gd-remove-option ${trashProjectFolder ? 'selected' : ''}">
            <input type="checkbox" data-remove-project-folder ${trashProjectFolder ? 'checked' : ''} ${busy || rows.some(item => !item.project_dir) ? 'disabled' : ''} />
            <span><strong>${t('Also move the local project folders to the system trash', '同时将本地项目文件夹移到系统废纸篓')}</strong><small>${t('This includes the selected projects’ conversations, configuration, runs, results, and evidence. They can normally be restored from the system trash.', '其中包括所选项目的对话、配置、运行、结果和证据；通常可以从系统废纸篓恢复。')}</small>${!multiple && row.project_dir ? `<code>${esc(compactPath(row.project_dir))}</code>` : ''}</span>
          </label>
          <div class="gd-remove-impact ${trashProjectFolder ? 'warn' : ''}">${trashProjectFolder
            ? t('The selected project folders will leave their current locations. Other EasyICU modules using them will no longer find them unless you restore them.', '所选项目文件夹将离开当前位置；其他正在使用这些目录的 EasyICU 模块将无法继续找到它们，除非你从废纸篓恢复。')
            : t('Only the list entries will be removed. Project folders remain unchanged on disk.', '只会移除列表记录，项目文件夹在磁盘上保持不变。')}</div>`}
          ${state.error ? `<div class="gd-remove-error" role="alert">${esc(state.error)}</div>` : ''}
          <div class="row gap-8">
            <button class="btn sm gd-remove-confirm ${trashProjectFolder ? 'danger' : ''}" type="button" data-confirm-remove-draft ${busy ? 'disabled' : ''}>${trashProjectFolder ? t('Move to trash and remove', '移到废纸篓并移除') : t('Remove from list', '从列表移除')}</button>
            <button class="btn sm" type="button" data-remove-draft-close ${busy ? 'disabled' : ''}>${t('Cancel', '取消')}</button>
          </div>
        </div>
      </section>`;
  }

  // The title a new study starts with, in the interface language.
  function defaultStudyTitle(ctx) {
    return ctx.t('New study', '新研究');
  }

  window.EU_GUIDED_PROJECTS = {
    defaultStudyTitle,
    defaultParentDir,
    parentPathLabel,
    createDraft,
    startProjectNaming,
    stopProjectNaming,
    renderShellRail,
    renderProjectRail,
    renderFolderControls,
    renderFolderBrowser,
    renderFolderDialog,
    renderDraftRemovalDialog,
    setProjectManagement,
    isProjectManagementActive,
    toggleProjectSelection,
    selectAllProjects,
    selectedProjects,
    setProjectRailCollapsed,
    isProjectRailCollapsed,
    showProjects,
  };
})();
