/* Guided Copilot project/folder picker rendering.
   Owns HTML for local study-folder selection; screens-guided.js owns state
   and event handlers, passing a small context object into this module. */
(function () {
  const PROJECT_RAIL_COLLAPSED_KEY = 'easyicu.guided.projectRailCollapsed.v1';
  let projectManagementActive = false;
  let projectRailCollapsed = readProjectRailCollapsed();
  const selectedProjectIds = new Set();

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
      slugifyDraftFolder: ctx.slugifyDraftFolder,
      fmtRunTime: ctx.fmtRunTime,
    };
  }

  function projectTitle(value, fallback) {
    return window.EU_PRODUCT_LABELS.projectTitle(value, fallback);
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
        ? t('Local data', '本地数据')
        : (row.data_mode === 'unbound' ? t('No data selected', '未选择数据') : (row.data_mode || '')));
    return [status, mode].filter(Boolean).join(' · ');
  }

  function renderShellRail(ctx) {
    const { t, icon } = helpers(ctx);
    return `
      <button class="gd-rail-restore" type="button" data-project-rail-toggle aria-label="${t('Show research projects', '显示研究项目栏')}" title="${t('Show research projects', '显示研究项目栏')}">${icon('chevron', 14)}</button>
      <aside class="gd-rail" id="gdResearchProjectRail">
        <div class="gd-rail-top">
          <div class="gd-rail-heading">
            <button class="gd-rail-brand" type="button" data-open="entry" aria-label="${t('Back to EasyICU home', '返回 EasyICU 首页')}" title="${t('Back to EasyICU home', '返回 EasyICU 首页')}"><span class="brand-mark">${icon('spark', 15)}</span><span class="gd-name">${t('EasyICU ICU Research Assistant', 'EasyICU 重症科研助手')}</span></button>
            <button class="gd-rail-collapse" type="button" data-project-rail-toggle aria-controls="gdResearchProjectRail" aria-label="${t('Hide research projects', '隐藏研究项目栏')}" title="${t('Hide research projects', '隐藏研究项目栏')}">${icon('chevron', 14)}</button>
          </div>
          <div class="gd-folder-controls" id="gdFolderControls"></div>
        </div>
        <div class="gd-rail-list" id="gdSessions"></div>
        <div class="gd-rail-foot">
          <div class="gd-rail-utils" aria-label="${t('Guided Copilot utilities', '研究引导工具')}">
            <button class="gd-utilbtn" type="button" data-open="entry" title="${t('Home', '主页')}" aria-label="${t('Home', '主页')}">${icon('back', 14)}</button>
            <button class="gd-utilbtn" type="button" data-open="settings" title="${t('Settings', '设置')}" aria-label="${t('Settings', '设置')}">${icon('gear', 14)}</button>
            <button class="gd-utilbtn lang" type="button" data-lang-toggle title="${t('Switch language', '切换语言')}" aria-label="${t('Switch language', '切换语言')}">
              ${icon('globe', 14)} <span>${window.EU_LANG === 'zh' ? 'EN' : '中'}</span>
            </button>
          </div>
          <button class="btn sm block gd-data-workspace" data-open="extraction">${icon('grid', 13)} ${t('Data workspace', '数据工作台')}</button>
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
      ? `${externalHtml}<div class="gd-empty-local"><div class="ss-t">${t('Loading study folders', '正在加载研究文件夹')}</div><div class="ss-m">${t('Reading the local project registry.', '正在读取本地项目列表。')}</div></div>`
      : ctx.guidedDrafts.error
        ? `${externalHtml}<div class="gd-empty-local warn"><div class="ss-t">${t('Study folders unavailable', '研究文件夹不可用')}</div><div class="ss-m">${esc(ctx.guidedDrafts.error)}</div></div>`
        : rows.length
          ? externalHtml + rows.map((row, i) => {
            const active = activeId === row.id;
            const selected = selectedProjectIds.has(row.id);
            const selectable = !active;
            const configurationMissing = row.configuration_health
              && row.configuration_health.status === 'configuration_missing';
            const title = projectTitle(row.title, row.question || t('Guided study', '研究项目'));
            const meta = projectMeta(row, t);
            const time = fmtRunTime(row.updated_at || row.created_at);
            return `
            <div class="gd-sessline ${projectManagementActive ? 'is-managing' : ''} ${selected ? 'is-selected' : ''}">
              ${projectManagementActive ? `<label class="gd-sess-select" title="${active ? t('The current project cannot be selected', '当前正在使用的项目不能勾选') : t('Select project', '选择项目')}"><input type="checkbox" data-select-localdraft="${i}" ${selected ? 'checked' : ''} ${selectable ? '' : 'disabled'} /><span aria-hidden="true"></span></label>` : ''}
              <button class="gd-sess draft ${active ? 'active' : ''} ${configurationMissing ? 'configuration-missing' : ''}" data-localdraft="${i}" title="${configurationMissing ? t('Configuration is missing; open recovery options', '研究配置已失效；打开恢复选项') : t('Open research project', '打开研究项目')}: ${esc(title)}" ${active ? 'aria-current="true"' : ''}>
                <span class="gd-sess-status" aria-hidden="true"></span>
                <span class="gd-sess-body"><span class="ss-t">${esc(title)}</span><span class="ss-m">${esc(meta)}</span></span>
                <span class="ss-time">${esc(time)}</span>
              </button>
              ${projectManagementActive ? '' : `<button class="gd-sess-action danger" type="button" data-remove-localdraft="${i}" title="${t('Remove from project list', '从项目列表移除')}" aria-label="${t('Remove from project list', '从项目列表移除')}: ${esc(title)}">${icon('close', 12)}</button>`}
            </div>`;
          }).join('')
          : (externalHtml || `<div class="gd-empty-local"><div class="ss-t">${t('No study folders yet', '还没有研究文件夹')}</div><div class="ss-m">${t('Create or open a project before starting a Copilot conversation.', '开始研究助手对话前，请先创建或打开一个项目。')}</div></div>`);
    host.innerHTML = `
      <div class="gd-project-heading"><span>${t('Research projects', '研究项目')}</span><span class="gd-project-heading-actions"><button class="gd-manage-mini ${projectManagementActive ? 'active' : ''}" type="button" data-project-manage>${projectManagementActive ? t('Done', '完成') : t('Manage', '管理')}</button><button class="gd-refresh-mini" type="button" data-refreshdrafts title="${t('Refresh research projects', '刷新研究项目')}" aria-label="${t('Refresh research projects', '刷新研究项目')}">${icon('refresh', 12)}</button></span></div>
      <div class="gd-project-summary">${icon('folder', 14)}<div><strong>${t('Local research workspace', '本地研究工作区')}</strong><span>${t('Study setup, runs, evidence, and conversation history stay here.', '研究配置、运行、证据和对话历史都保存在这里。')}</span></div></div>
      ${draftHtml}
      ${projectManagementActive ? `<div class="gd-project-selection-bar"><label><input type="checkbox" data-select-all-projects ${eligibleCount && selectedRows.length === eligibleCount ? 'checked' : ''} ${eligibleCount ? '' : 'disabled'} /> ${t('Select all', '全选')}</label><span><strong>${selectedRows.length}</strong> ${t('selected; current project excluded', '项已选；当前项目不参与')}</span><button class="btn sm danger" type="button" data-remove-selected-projects ${selectedRows.length ? '' : 'disabled'}>${t('Remove selected', '移除所选')}</button></div>` : ''}`;
  }

  function renderFolderControls(ctx) {
    const { t, icon } = helpers(ctx);
    const host = document.getElementById('gdFolderControls');
    if (!host) return;
    host.innerHTML = `
      <div class="gd-folder-picker ${ctx.guidedFolderMenuOpen ? 'open' : ''}">
        <button class="gd-newbtn" type="button" data-newstudy data-folder-menu-toggle aria-haspopup="menu" aria-expanded="${ctx.guidedFolderMenuOpen ? 'true' : 'false'}" title="${t('Choose or create a local study folder', '选择或创建本地研究文件夹')}">
          ${icon('plus', 14)} <span>${t('New / open research project', '新建 / 打开研究项目')}</span>
        </button>
        ${ctx.guidedFolderMenuOpen ? `
          <div class="gd-folder-menu" role="menu" aria-label="${t('Study folder actions', '研究文件夹操作')}">
            <button class="gd-folder-menu-item" type="button" role="menuitem" data-folder-choice="new">
              <span class="gds-ico">${icon('folder', 14)}</span>
              <span><strong>${t('New blank study folder', '新建空白项目')}</strong><small>${t('Choose a parent folder, then create a metadata-only Guided project subfolder.', '先选择父目录，再创建仅元数据的 Guided 项目子文件夹。')}</small></span>
            </button>
            <button class="gd-folder-menu-item" type="button" role="menuitem" data-folder-choice="open">
              <span class="gds-ico">${icon('folder', 14)}</span>
              <span><strong>${t('Use existing folder', '使用现有文件夹')}</strong><small>${t('Open a Guided, Idea Mining, or Agent project folder as the current research project.', '把已有 Guided、Idea Mining 或 Agent 项目文件夹作为当前研究项目打开。')}</small></span>
            </button>
          </div>` : ''}
      </div>`;
  }

  function renderKnownProjectPicker(ctx) {
    const { t, icon, esc, compactPath } = helpers(ctx);
    const loading = ctx.guidedDrafts && ctx.guidedDrafts.loading;
    const error = ctx.guidedDrafts && ctx.guidedDrafts.error;
    const rows = ctx.guidedKnownProjectRows();
    if (!ctx.guidedKnownProjectsOpen) {
      return `
      <div class="gds-known collapsed">
        <div class="gds-known-head">
          <div><strong>${t('Recent local projects', '最近的本地项目')}</strong><span>${t('Optional shortcut. The list stays collapsed until you ask; no project is opened automatically.', '可选快捷入口。列表默认折叠，只有你主动展开才显示；不会自动打开任何项目。')}</span></div>
          <button class="btn sm" type="button" data-toggle-known-projects>${icon('history', 12)} ${t('Show recent', '显示最近项目')}</button>
        </div>
      </div>`;
    }
    return `
      <div class="gds-known">
        <div class="gds-known-head">
          <div><strong>${t('Recent local projects', '最近的本地项目')}</strong><span>${t('Shown only after you ask. Pick one as a shortcut, or use Browse/manual path below.', '仅在你主动展开后显示。可以选一个作为快捷入口，也可以用下方浏览/手动路径。')}</span></div>
          <button class="btn sm" type="button" data-refreshfolderchoices>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
        </div>
        ${loading ? `<div class="gds-known-empty">${icon('refresh', 12)} ${t('Loading recent local projects...', '正在加载最近的本地项目...')}</div>` : ''}
        ${error ? `<div class="gds-known-empty warn">${icon('info', 12)} ${esc(error)}</div>` : ''}
        ${!loading && !rows.length && !error ? `<div class="gds-known-empty">${t('No recent Guided project shortcuts yet. Create a new folder, browse a folder, or paste a local path below.', '暂无最近 Guided 项目快捷入口。可以新建文件夹、浏览文件夹，或在下方粘贴本地路径。')}</div>` : ''}
        ${rows.length ? `<div class="gds-known-list">${rows.map((row, i) => `
          <button class="gds-known-row" type="button" data-known-project="${i}">
            <span class="gds-known-kind">${row.kind === 'run' ? icon('history', 13) : icon('file', 13)}</span>
            <span><strong>${esc(projectTitle(row.title, row.question))}</strong><small>${esc(row.subtitle)}</small><code>${esc(compactPath(row.project_dir))}</code></span>
            <span class="gds-known-open">${icon('arrow', 13)}</span>
          </button>`).join('')}</div>` : ''}
      </div>`;
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
          <div>
            <strong>${t('Folder picker', '文件夹选择器')}</strong>
            <span>${t('Browse folders through the local EasyICU server; nothing is uploaded.', '通过本机 EasyICU 服务浏览文件夹；不会上传任何内容。')}</span>
          </div>
          <button class="btn sm ghost" type="button" data-folder-browser-close>${icon('close', 12)}</button>
        </div>
        <div class="gds-browser-path"><span>${t('Current', '当前')}</span><code>${esc(compactPath(currentPath) || t('Home folder', '主目录'))}</code></div>
        ${shortcuts.length ? `<div class="gds-browser-shortcuts">${shortcuts.map((item, i) => `
          <button class="btn sm" type="button" data-folder-browser-shortcut="${i}">${esc(item.name || 'Folder')}</button>`).join('')}</div>` : ''}
        ${failed ? `<div class="gds-browser-message warn">${icon('info', 12)} <span>${esc(String(failed))}</span></div>` : ''}
        <div class="gds-browser-list">
          ${browser.loading ? `<div class="gds-browser-empty">${icon('refresh', 13)} ${t('Loading folders...', '正在加载文件夹...')}</div>` : ''}
          ${!browser.loading && !entries.length ? `<div class="gds-browser-empty">${t('No child folders here. You can still choose the current folder if it is the project folder.', '这里没有下级文件夹。如果当前目录就是项目文件夹，也可以直接选择当前文件夹。')}</div>` : ''}
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
    const { t, icon, esc, attr, compactPath, slugifyDraftFolder } = helpers(ctx);
    const host = document.getElementById('gdFolderDialogHost');
    if (!host) return;
    if (!ctx.guidedFolderDialogMode) {
      host.innerHTML = '';
      return;
    }
    const branchCfg = ctx.BRANCH && ctx.BRANCH[ctx.branch];
    const title = ctx.guidedFolderSeedTitle || (branchCfg && branchCfg.chip) || 'New local study';
    const slug = ctx.guidedDraftFolderSlug || slugifyDraftFolder(title);
    const parentDir = ctx.guidedDraftParentDir || '~/easyicu/projects';
    const parentDisplay = compactPath(parentDir).replace(/\/$/, '');
    const projectPreview = `${parentDisplay}/guided-${slug || 'study'}-...`;
    const mode = ctx.guidedFolderDialogMode === 'open' ? 'open' : 'new';
    const openForReview = ctx.pendingGuidedGoal && ctx.pendingGuidedGoal.goal === 'review_data';
    const openActions = openForReview ? `
      <button class="btn primary sm" data-reviewexportfolder>${icon('eye', 13)} ${t('Review extracted data', '审阅已提取数据')}</button>
      <button class="btn sm" data-openprojectfolder>${icon('folder', 13)} ${t('Open research project', '打开研究项目')}</button>` : `
      <button class="btn primary sm" data-openprojectfolder>${icon('folder', 13)} ${t('Open research project', '打开研究项目')}</button>
      <button class="btn sm" data-reviewexportfolder>${icon('eye', 13)} ${t('Review extracted data', '审阅已提取数据')}</button>`;
    host.innerHTML = `
      <div class="gd-folder-backdrop" data-folder-dialog-close></div>
      <section class="gd-folder-dialog" data-folder-dialog data-draft-setup role="dialog" aria-modal="true" aria-label="${t('Choose a local study folder', '选择本地研究文件夹')}">
        <div class="gd-folder-dialog-head">
          <span class="gds-ico">${icon('folder', 15)}</span>
          <div>
            <strong>${t('Choose a local folder', '选择本地文件夹')}</strong>
            <span>${t('Open a project folder for study setup, runs, evidence, and conversation history, or choose an EasyICU export folder to review extracted data.', '项目文件夹用于保存研究配置、运行、证据和对话历史；如果要审阅已提取数据，请选择 EasyICU export 文件夹。')}</span>
          </div>
          <button class="gd-folder-close" type="button" data-folder-dialog-close aria-label="${t('Close', '关闭')}">×</button>
        </div>
        <div class="gd-folder-tabs" role="tablist" aria-label="${t('Folder setup mode', '文件夹设置模式')}">
          <button class="${mode === 'new' ? 'active' : ''}" type="button" data-folder-choice="new">${t('New blank project', '新建空白项目')}</button>
          <button class="${mode === 'open' ? 'active' : ''}" type="button" data-folder-choice="open">${t('Use existing folder', '使用现有文件夹')}</button>
        </div>
        ${mode === 'open' ? `
          <div class="gds-choice">
            <div class="gds-choice-head"><strong>${t('Open project or extracted data folder', '打开项目或已提取数据文件夹')}</strong><span>${t('Required setup stays here instead of jumping to Classic Workspace. Use a project folder for study state, evidence, and conversations. Use an EasyICU export folder to review previously extracted data.', '必需配置都留在这里完成，不强制跳到其他页面。项目文件夹保存研究状态、证据和对话；EasyICU export 文件夹用于审阅之前提取的数据。')}</span></div>
            <div class="gds-path-row">
              <label class="gds-field"><span>${t('Local folder path', '本地文件夹路径')}</span><input data-existing-project-dir placeholder="${t('Paste a local project or EasyICU export folder path', '粘贴本地项目或 EasyICU 导出文件夹路径')}" autocomplete="off" /></label>
              <button class="btn sm" type="button" data-browseprojectfolder>${icon('folder', 13)} ${t('Browse...', '浏览...')}</button>
            </div>
            <div class="gds-path"><span>${t('Scope', '范围')}</span><code>local EasyICU project or export folder</code><small>${t('Use Browse to choose a folder. Path paste remains an advanced fallback for terminal workflows.', '请使用“浏览”选择文件夹。路径粘贴只作为终端工作流的高级 fallback。')}</small></div>
            ${renderFolderBrowser(ctx)}
            ${renderKnownProjectPicker(ctx)}
            <div class="row gap-8">${openActions}<button class="btn sm" data-folder-dialog-close>${t('Cancel', '取消')}</button></div>
            <div class="gds-status" data-project-open-status hidden></div>
          </div>` : `
          <div class="gds-choice">
            <div class="gds-choice-head"><strong>${t('Create new local study folder', '创建新的本地研究文件夹')}</strong><span>${t('Choose the parent folder first. EasyICU will create a new metadata-only project subfolder there; no patient rows, Agent run, or draft unlock is created.', '先选择父目录。EasyICU 会在其中创建新的仅元数据项目子文件夹；不会读取患者行、不会创建 Agent run、不会解锁草稿。')}</span></div>
            <label class="gds-field"><span>${t('Study title', '研究标题')}</span><input data-draft-title value="${attr(title)}" autocomplete="off" /></label>
            <label class="gds-field"><span>${t('Folder name', '文件夹名称')}</span><input data-draft-slug value="${attr(slug)}" autocomplete="off" /></label>
            <div class="gds-path-row">
              <label class="gds-field"><span>${t('Create inside folder', '创建到本地文件夹')}</span><input data-draft-parent-dir value="${attr(parentDir)}" placeholder="${t('Choose or paste a local parent folder', '选择或粘贴本地父目录')}" autocomplete="off" /></label>
              <button class="btn sm" type="button" data-browsedraftparent>${icon('folder', 13)} ${t('Browse...', '浏览...')}</button>
            </div>
            <div class="gds-path"><span>${t('Will create', '将创建')}</span><code>${esc(projectPreview)}</code><small>${t('The selected folder is the parent. The actual project folder gets a guided-* name so existing folders are not overwritten.', '所选文件夹是父目录。实际项目会使用 guided-* 子文件夹名称，避免覆盖已有文件夹。')}</small></div>
            ${renderFolderBrowser(ctx)}
            <div class="row gap-8"><button class="btn primary sm" data-createdraft>${icon('folder', 13)} ${t('Create study folder', '创建研究文件夹')}</button><button class="btn sm" data-folder-dialog-close>${t('Cancel', '取消')}</button></div>
          </div>`}
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
    const title = projectTitle(row.title, row.question || t('Guided project', '研究项目'));
    const multiple = rows.length > 1;
    const trashProjectFolder = !!state.trashProjectFolder;
    const busy = !!state.busy;
    host.innerHTML = `
      <div class="gd-folder-backdrop" data-remove-draft-close></div>
      <section class="gd-folder-dialog gd-remove-dialog" data-remove-draft-dialog role="dialog" aria-modal="true" aria-label="${t('Remove research project', '移除研究项目')}">
        <div class="gd-folder-dialog-head">
          <span class="gds-ico">${icon('folder', 15)}</span>
          <div><strong>${multiple ? t('Remove selected research projects', '移除所选研究项目') : t('Remove research project', '移除研究项目')}</strong><span>${t('Choose whether to remove only the EasyICU list entry or also move the selected project folders to the system trash.', '请选择只从 EasyICU 列表移除，还是同时把所选项目文件夹移到系统废纸篓。')}</span></div>
          <button class="gd-folder-close" type="button" data-remove-draft-close aria-label="${t('Close', '关闭')}" ${busy ? 'disabled' : ''}>×</button>
        </div>
        <div class="gds-choice">
          <div class="gds-choice-head"><strong>${multiple ? `${rows.length} ${t('projects selected', '个项目已选择')}` : esc(title)}</strong><span>${t('By default, project folders and all files on disk are preserved.', '默认会保留磁盘上的项目文件夹和全部文件。')}</span></div>
          <label class="gd-remove-option ${trashProjectFolder ? 'selected' : ''}">
            <input type="checkbox" data-remove-project-folder ${trashProjectFolder ? 'checked' : ''} ${busy || rows.some(item => !item.project_dir) ? 'disabled' : ''} />
            <span><strong>${t('Also move the local project folders to the system trash', '同时将本地项目文件夹移到系统废纸篓')}</strong><small>${t('This includes the selected projects’ conversations, configuration, runs, results, and evidence. They can normally be restored from the system trash.', '其中包括所选项目的对话、配置、运行、结果和证据；通常可以从系统废纸篓恢复。')}</small>${!multiple && row.project_dir ? `<code>${esc(compactPath(row.project_dir))}</code>` : ''}</span>
          </label>
          <div class="gd-remove-impact ${trashProjectFolder ? 'warn' : ''}">${trashProjectFolder
            ? t('The selected project folders will leave their current locations. Other EasyICU modules using them will no longer find them unless you restore them.', '所选项目文件夹将离开当前位置；其他正在使用这些目录的 EasyICU 模块将无法继续找到它们，除非你从废纸篓恢复。')
            : t('Only the list entries will be removed. Project folders remain unchanged on disk.', '只会移除列表记录，项目文件夹在磁盘上保持不变。')}</div>
          ${state.error ? `<div class="gd-remove-error" role="alert">${esc(state.error)}</div>` : ''}
          <div class="row gap-8">
            <button class="btn sm gd-remove-confirm ${trashProjectFolder ? 'danger' : ''}" type="button" data-confirm-remove-draft ${busy ? 'disabled' : ''}>${trashProjectFolder ? t('Move to trash and remove', '移到废纸篓并移除') : t('Remove from list', '从列表移除')}</button>
            <button class="btn sm" type="button" data-remove-draft-close ${busy ? 'disabled' : ''}>${t('Cancel', '取消')}</button>
          </div>
        </div>
      </section>`;
  }

  window.EU_GUIDED_PROJECTS = {
    renderShellRail,
    renderProjectRail,
    renderFolderControls,
    renderKnownProjectPicker,
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
  };
})();
