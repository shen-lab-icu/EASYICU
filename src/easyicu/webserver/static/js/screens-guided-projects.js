/* Guided Copilot project/folder picker rendering.
   Owns HTML for local study-folder selection; screens-guided.js owns state
   and event handlers, passing a small context object into this module. */
(function () {
  function helpers(ctx) {
    return {
      t: ctx.t,
      icon: ctx.icon,
      esc: ctx.esc,
      attr: ctx.attr,
      compactPath: ctx.compactPath,
      slugifyDraftFolder: ctx.slugifyDraftFolder,
    };
  }

  function renderFolderControls(ctx) {
    const { t, icon } = helpers(ctx);
    const host = document.getElementById('gdFolderControls');
    if (!host) return;
    host.innerHTML = `
      <div class="gd-folder-picker ${ctx.guidedFolderMenuOpen ? 'open' : ''}">
        <button class="gd-newbtn" type="button" data-newstudy data-folder-menu-toggle aria-haspopup="menu" aria-expanded="${ctx.guidedFolderMenuOpen ? 'true' : 'false'}" title="${t('Choose or create a local study folder', '选择或创建本地研究文件夹')}">
          ${icon('plus', 14)} ${t('New / open research project', '新建/打开研究项目')}
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
            <span><strong>${esc(row.title)}</strong><small>${esc(row.subtitle)}</small><code>${esc(compactPath(row.project_dir))}</code></span>
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
            <span>${t('Open a project folder for study setup, runs, and evidence, or choose an EasyICU export folder to review extracted data. Pi manages conversation history separately.', '项目文件夹用于保存研究配置、运行和证据；如果要审阅已提取数据，请选择 EasyICU export 文件夹。对话历史由 Pi 单独管理。')}</span>
          </div>
          <button class="gd-folder-close" type="button" data-folder-dialog-close aria-label="${t('Close', '关闭')}">×</button>
        </div>
        <div class="gd-folder-tabs" role="tablist" aria-label="${t('Folder setup mode', '文件夹设置模式')}">
          <button class="${mode === 'new' ? 'active' : ''}" type="button" data-folder-choice="new">${t('New blank project', '新建空白项目')}</button>
          <button class="${mode === 'open' ? 'active' : ''}" type="button" data-folder-choice="open">${t('Use existing folder', '使用现有文件夹')}</button>
        </div>
        ${mode === 'open' ? `
          <div class="gds-choice">
            <div class="gds-choice-head"><strong>${t('Open project or extracted data folder', '打开项目或已提取数据文件夹')}</strong><span>${t('Required setup stays here instead of jumping to Classic Workspace. Use a project folder for study state and evidence; Pi keeps its own conversations. Use an EasyICU export folder to review previously extracted data.', '必需配置都留在这里完成，不强制跳到其他页面。项目文件夹保存研究状态和证据，Pi 单独保存对话；EasyICU export 文件夹用于审阅之前提取的数据。')}</span></div>
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

  window.EU_GUIDED_PROJECTS = {
    renderFolderControls,
    renderKnownProjectPicker,
    renderFolderBrowser,
    renderFolderDialog,
  };
})();
