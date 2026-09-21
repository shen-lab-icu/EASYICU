/* Owner: desktop research Skill Hub. The catalog projects real host contracts. */
(function () {
  'use strict';
  const S = (window.SCREENS = window.SCREENS || {});
  const { esc } = window.EU_HTML;
  const ui = { catalogMode: 'skills', source: 'all', category: 'all', query: '', hideDisabled: false,
    selected: '', tab: 'overview', install: false, editing: '', draft: '', stages: ['conversation'],
    installEnabled: true, detail: null, loadingDetail: false, packageDetail: null,
    packageFile: 'SKILL.md', loadingPackage: false, busy: false, error: '', notice: '' };
  const tr = (en, zh) => window.t(en, zh);
  const api = () => window.EU_API || {};
  const registry = () => window.EU_EXTENSIONS || {};
  const capabilityRoot = () => (window.EU_CAPABILITIES || {}).capabilities || {};
  const publicationCapabilities = () => capabilityRoot().publication_skills || {};
  const methodCapabilities = () => capabilityRoot().method_skills || {};
  const builtinZh = {
    'nature-figure': {
      scope: '先确定图件要表达的核心论断，再使用已登记的源数据和代码生成承载结果的图件。',
      inputs: ['文章图件策略', '已登记的结果证据', '源数据与不确定性定义'],
      outputs: ['以论断为中心的 FigureContract', '可编辑的 SVG/PDF 及 PNG/TIFF 导出', '源数据与图件质检证据'],
      invariants: ['一句话核心论断及分图证据链', '数值图件由代码支撑；自由生成不能改变结果', '坐标轴、单位、不确定性、源数据和导出质检均可审计'],
    },
    'nature-writing': {
      scope: '按段落职责撰写面向广泛读者的科学论证，校准措辞，并精确绑定证据和文献。',
      inputs: ['研究背景', '机器生成的证据摘要', '绑定当前运行的文献摘要'],
      outputs: ['按章节组织的论文正文', '论断与证据绑定', '证据缺口、数值、文献及创新性审计'],
      invariants: ['不编造结果、引文、方法、创新性或统计量', '每段承担一项读者可理解的职责，且论断有证据支持', '术语、因果措辞与创新性表述保持审慎'],
    },
  };
  const methodSkills = () => (Array.isArray(methodCapabilities().items) ? methodCapabilities().items : []).map(row => ({
    id: `builtin:${row.id}`, name: row.id,
    title: window.EU_LANG === 'zh' ? row.title_zh : row.title,
    description: window.EU_LANG === 'zh' ? row.description_zh : row.description,
    category: window.EU_LANG === 'zh' ? row.category_zh : row.category,
    source: 'builtin', kind: 'method', skillLayer: row.layer || 'research_workflow', enabled: !!row.enabled, toggleable: false,
    version: row.version, capabilityId: row.capability_id, actionIds: row.action_ids || [],
    includedModuleIds: row.included_module_ids || [],
    executionMode: row.execution_mode, claimCeiling: row.claim_ceiling,
    scope: row.scope || '', inputs: row.inputs || [], outputs: row.outputs || [],
    diagnostics: row.diagnostics || [], prompt: window.EU_LANG === 'zh' ? row.prompt_zh : row.prompt,
  }));
  const methodComponents = () => (Array.isArray(methodCapabilities().components) ? methodCapabilities().components : []).map(row => ({
    id: `builtin:${row.id}`, name: row.id,
    title: window.EU_LANG === 'zh' ? row.title_zh : row.title,
    description: window.EU_LANG === 'zh' ? row.description_zh : row.description,
    category: window.EU_LANG === 'zh' ? row.category_zh : row.category,
    source: 'builtin', kind: 'method_component', enabled: !!row.enabled, toggleable: false,
    version: row.version, methodFamily: row.method_family, methodKey: row.method_key,
    tier: row.tier, implementation: row.implementation,
    executionMode: row.execution_mode, claimCeiling: row.claim_ceiling,
    outputs: row.outputs || [], reportingItems: row.reporting_items || [], kernelModules: row.kernel_modules || [],
    prompt: window.EU_LANG === 'zh' ? row.prompt_zh : row.prompt,
  }));
  const publicationSkills = () => (Array.isArray(publicationCapabilities().items) ? publicationCapabilities().items : []).map(row => ({
    id: `builtin:${row.id}`, name: row.id, title: row.id === 'nature-figure'
      ? tr('Evidence figures', '证据图件') : tr('Evidence writing', '证据写作'),
    description: row.id === 'nature-figure'
      ? tr('Plan a claim-first figure and render numerical results from registered data and code.', '先确定图件论断，再由已登记的数据和代码生成数值结果图。')
      : tr('Build manuscript claims with explicit evidence and literature bindings.', '围绕证据与文献组织论文论断。'),
    category: row.stage === 'Figure' ? tr('Figures', '图件') : tr('Writing', '写作'),
    source: 'builtin', kind: 'publication', skillLayer: 'analysis_module', enabled: !!row.enabled, settingKey: row.setting_key, toggleable: true,
    version: row.version,
    scope: window.EU_LANG === 'zh' && builtinZh[row.id] ? builtinZh[row.id].scope : row.scope,
    inputs: window.EU_LANG === 'zh' && builtinZh[row.id] ? builtinZh[row.id].inputs : row.inputs || [],
    outputs: window.EU_LANG === 'zh' && builtinZh[row.id] ? builtinZh[row.id].outputs : row.outputs || [],
    invariants: window.EU_LANG === 'zh' && builtinZh[row.id] ? builtinZh[row.id].invariants : row.invariants || [],
    executor: row.executor || '',
  }));
  const builtins = () => [...methodSkills(), ...methodComponents(), ...publicationSkills()];
  const userSkills = () => (Array.isArray(registry().skills) ? registry().skills : []).map(row => ({
    ...row, id: `user:${row.name}`, title: row.name, category: row.category || ((row.stages || []).includes('writing')
      ? tr('Writing', '写作') : tr('Conversation', '对话')), source: 'user', skillLayer: 'analysis_module',
  }));
  const allSkills = () => [...builtins(), ...userSkills()];
  const selectedSkill = () => allSkills().find(item => item.id === ui.selected);
  const date = value => value ? new Date(value).toLocaleDateString(window.EU_LANG === 'zh' ? 'zh-CN' : 'en-US') : '';
  const list = values => `<ul>${(values || []).map(value => `<li>${esc(value)}</li>`).join('')}</ul>`;
  const instructionHtml = text => {
    const owner = window.EasyICU && window.EasyICU.guidedPi
      && window.EasyICU.guidedPi.optional('markdown');
    return owner && typeof owner.render === 'function'
      ? owner.render(text) : `<pre>${esc(text)}</pre>`;
  };
  const sourceLabel = source => source === 'user' ? tr('Created by you', '我创建的') : tr('Verified by EasyICU', 'EasyICU 已验证');
  const executionLabel = skill => skill.executionMode === 'deterministic_host'
    ? tr('Host executed', '主机确定性执行') : tr('Agent coded + host gates', 'Agent 编码 + 主机门禁');
  const ceilingLabel = skill => skill.claimCeiling === 'reportable'
    ? tr('Reportable contract', '可报告契约') : tr('Analysis only', '仅分析');
  const tierLabel = skill => ({ primary: tr('Primary method', '主要方法'),
    standard_supporting: tr('Supporting method', '配套方法'), exploratory: tr('Exploratory method', '探索方法') }[skill.tier] || skill.tier || '');
  const methodFamilySpecs = () => [
    ['descriptive', tr('Descriptive and measurement', '描述性与测量')],
    ['association', tr('Association analysis', '关联分析')],
    ['time_to_event', tr('Survival and time-to-event', '生存与时间结局')],
    ['prediction', tr('Prediction and validation', '预测与验证')],
    ['causal_emulation', tr('Causal inference', '因果推断')],
    ['phenotyping', tr('Phenotyping and trajectories', '表型与轨迹')],
  ];
  const layerLabel = skill => skill.skillLayer === 'analysis_module'
    ? tr('Reusable module', '可复用模块') : tr('Research workflow', '研究工作流');

  function rail() {
    if (window.EU_DESKTOP_MODULE_SHELL) return window.EU_DESKTOP_MODULE_SHELL.rail('skills');
    return `<nav class="eusk-rail" aria-label="${tr('Main navigation', '主导航')}">
      <button type="button" class="eusk-brand" data-sk-route="guided" title="EasyICU" aria-label="EasyICU">${icon('spark', 19)}</button>
      <button type="button" data-sk-route="guided" title="${tr('Projects', '项目')}" aria-label="${tr('Projects', '项目')}">${icon('folder', 19)}<span>${tr('Projects', '项目')}</span></button>
      <button type="button" class="active" aria-current="page" title="${tr('Skills', '技能')}" aria-label="${tr('Skills', '技能')}">${icon('layers', 19)}<span>${tr('Skills', '技能')}</span></button>
      <button type="button" data-sk-route="extraction" title="${tr('Data', '数据')}" aria-label="${tr('Data', '数据')}">${icon('grid', 19)}<span>${tr('Data', '数据')}</span></button>
      <div class="eusk-rail-spacer"></div>
      <button type="button" data-sk-route="settings" title="${tr('Settings', '设置')}" aria-label="${tr('Settings', '设置')}">${icon('gear', 18)}<span>${tr('Settings', '设置')}</span></button>
    </nav>`;
  }
  function catalog() {
    const workflowEntries = methodSkills().filter(item => item.skillLayer === 'research_workflow');
    const analysisModuleEntries = methodSkills().filter(item => item.skillLayer === 'analysis_module');
    const publicationEntries = publicationSkills();
    const personalEntries = userSkills();
    const methodEntries = methodComponents();
    const skillEntries = [...workflowEntries, ...analysisModuleEntries, ...methodEntries, ...publicationEntries, ...personalEntries];
    const all = ui.catalogMode === 'methods' ? methodEntries : skillEntries;
    const categories = [...new Set(all.map(item => item.category))].sort((a, b) => a.localeCompare(b));
    const query = ui.query.trim().toLocaleLowerCase();
    const eligible = all.filter(item => !ui.hideDisabled || item.enabled);
    const filtered = eligible.filter(item => (ui.source === 'all' || item.source === ui.source)
      && (ui.category === 'all' || item.category === ui.category)
      && (!query || [item.name, item.title, item.description, item.category, item.scope].some(value => String(value || '').toLocaleLowerCase().includes(query))));
    const counts = { all: all.length, user: all.filter(item => item.source === 'user').length,
      builtin: all.filter(item => item.source === 'builtin').length };
    const workflowCount = workflowEntries.length;
    const moduleCount = analysisModuleEntries.length;
    const componentCount = methodEntries.length;
    const plannedCount = Number(methodCapabilities().planned_method_count || 0);
    const publicationCount = publicationEntries.length;
    const builtinGroups = [
      ['workflow', 'builtin', tr('Complete research workflows', '完整研究工作流')],
      ['module', 'builtin', tr('Reusable analysis modules', '可复用分析模块')],
      ...methodFamilySpecs().map(([family, heading]) => [`component:${family}`, 'builtin', heading]),
      ['publication', 'builtin', tr('Writing and figures', '写作与图件')],
    ];
    const groupSpecs = ui.catalogMode === 'methods'
      ? methodFamilySpecs().map(([family, heading]) => [`component:${family}`, 'builtin', heading])
      : ui.source === 'user' ? [['user', 'user', tr('Created by you', '我创建的')]]
        : ui.source === 'builtin' ? builtinGroups
          : [...builtinGroups, ['user', 'user', tr('Created by you', '我创建的')]];
    const cards = groupSpecs.map(([group, source, heading]) => {
      const rows = filtered.filter(item => item.source === source && (group === 'workflow' ? item.kind === 'method' && item.skillLayer === 'research_workflow'
        : group === 'module' ? item.kind === 'method' && item.skillLayer === 'analysis_module'
        : group.startsWith('component:') ? item.kind === 'method_component' && item.methodFamily === group.slice(10)
          : group === 'publication' ? item.kind === 'publication' : true));
      if (!rows.length) return '';
      return `<section class="eusk-group"><h2>${heading} <small>${rows.length}</small></h2>
        <div class="eusk-grid">${rows.map(item => `<article class="eusk-card ${item.enabled ? '' : 'disabled'}">
          <button type="button" class="eusk-card-open" data-sk-open="${esc(item.id)}" aria-label="${tr('Inspect skill', '查看技能')} ${esc(item.title)}">
            <strong>${esc(item.title)}</strong><small>${esc(item.category)}</small><p>${esc(item.description)}</p>
          </button><div class="eusk-card-foot"><span>${esc(source === 'user' ? date(item.updated_at) : item.kind === 'method' ? ceilingLabel(item) : item.kind === 'method_component' ? tierLabel(item) : (item.version || ''))}</span>
          <span class="eusk-source">${item.kind === 'method_component' ? tr('Method component', '方法组件') : item.kind === 'method' ? layerLabel(item) : sourceLabel(source)}</span><span class="eusk-state ${item.enabled ? 'on' : ''}">${item.enabled ? tr('Enabled', '已启用') : tr('Off', '未启用')}</span></div>
        </article>`).join('')}</div></section>`;
    }).join('');
    return `<div class="eusk-head"><div><span class="eusk-eyebrow">SKILL HUB</span><h1>${tr('Skills', '技能')}</h1></div>
      <details class="eusk-create-menu"><summary class="eusk-new">${icon('plus', 15)} ${tr('New skill', '新建技能')}</summary><div role="menu">
        <button type="button" role="menuitem" data-sk-upload aria-label="${tr('Upload SKILL.md', '上传 SKILL.md')}"><span>${icon('file', 16)}</span><span><strong>${tr('Upload SKILL.md', '上传 SKILL.md')}</strong><small>${tr('Review and install a local instruction file', '审阅并安装本地指令文件')}</small></span></button>
        <button type="button" role="menuitem" data-sk-create-with aria-label="${tr('Create with EasyICU', '使用 EasyICU 开发')}"><span>${icon('spark', 16)}</span><span><strong>${tr('Create with EasyICU', '使用 EasyICU 开发')}</strong><small>${tr('Open a new task with a Skill Builder draft', '在新任务中打开技能开发草稿')}</small></span></button>
      </div></details></div>
      <label class="eusk-search">${icon('search', 17)}<input data-sk-search type="search" placeholder="${ui.catalogMode === 'methods' ? tr('Search the method library…', '搜索方法库…') : tr('Search research skills…', '搜索研究技能…')}" value="${esc(ui.query)}" autocomplete="off"></label>
      <div class="eusk-mode-tabs" role="tablist" aria-label="${tr('Catalog level', '目录层级')}">
        <button type="button" role="tab" aria-selected="${ui.catalogMode === 'skills'}" data-sk-mode="skills">${tr('Skills', '技能')} <span>${skillEntries.length}</span><small>${tr('All loadable packages', '全部可加载包')}</small></button>
        <button type="button" role="tab" aria-selected="${ui.catalogMode === 'methods'}" data-sk-mode="methods">${tr('Method library', '方法库')} <span>${methodEntries.length}</span><small>${tr('Statistical components', '统计方法组件')}</small></button>
      </div>
      <div class="eusk-filters"><span class="eusk-filter-label">${tr('SOURCE', '来源')}</span>
      ${[['all', tr('All', '全部')], ['user', tr('Mine', '我的')], ['builtin', tr('EasyICU', 'EasyICU')]].map(([key, label]) =>
        `<button type="button" class="eusk-pill ${ui.source === key ? 'active' : ''}" data-sk-source="${key}" aria-pressed="${ui.source === key}">${label} <span>${counts[key]}</span></button>`).join('')}
      <label class="eusk-category"><span class="shell-sr-only">${tr('Category', '类别')}</span><select data-sk-category><option value="all">${tr('All categories', '全部类别')}</option>${categories.map(category => `<option value="${esc(category)}" ${ui.category === category ? 'selected' : ''}>${esc(category)}</option>`).join('')}</select></label>
      <label class="eusk-hide"><input type="checkbox" data-sk-hide ${ui.hideDisabled ? 'checked' : ''}> ${tr('Hide disabled', '隐藏未启用')}</label></div>
      <div class="eusk-catalog-summary" aria-label="${tr('Capability catalogue summary', '能力目录摘要')}">
        ${ui.catalogMode === 'methods'
          ? `<span>${tr(`${componentCount} available methods in six families.`, `${componentCount} 个可用方法，分为六个方法族。`)}</span>${plannedCount ? `<small>${tr(`${plannedCount} planned methods are not listed.`, `另有 ${plannedCount} 个规划中方法未上架。`)}</small>` : ''}`
          : `<span>${tr(`${workflowCount + moduleCount + componentCount + publicationCount} built-in skill packages: ${workflowCount} complete workflows, ${moduleCount} reusable analysis modules, ${componentCount} method packages, and ${publicationCount} writing or figure modules.`, `${workflowCount + moduleCount + componentCount + publicationCount} 个内置技能包：${workflowCount} 个完整项目流程、${moduleCount} 个可复用分析模块、${componentCount} 个方法包、${publicationCount} 个写作或图件模块。`)}</span>${personalEntries.length ? `<small>${tr(`${personalEntries.length} skills created by you.`, `另有 ${personalEntries.length} 个你创建的技能。`)}</small>` : ''}`}
      </div>
      ${cards || `<div class="eusk-empty"><strong>${tr('Nothing matches', '没有匹配的技能')}</strong><p>${tr('Try a different source, category, or search term.', '试试其他来源、类别或关键词。')}</p></div>`}`;
  }
  function detail() {
    const skill = selectedSkill();
    if (!skill) { ui.selected = ''; return catalog(); }
    const user = skill.source === 'user';
    const method = skill.kind === 'method' || skill.kind === 'method_component';
    const component = skill.kind === 'method_component';
    const workflow = skill.kind === 'method' && skill.skillLayer === 'research_workflow';
    const detail = user && ui.detail && ui.detail.name === skill.name ? ui.detail : null;
    const masterOff = !user && (method ? methodCapabilities().enabled === false : publicationCapabilities().enabled === false);
    return `<button type="button" class="eusk-back" data-sk-back>← ${tr('Back to skills', '返回技能目录')}</button>
      <div class="eusk-detail-head"><div><span class="eusk-eyebrow">${esc(skill.category)} · ${skill.kind === 'method' ? layerLabel(skill) : sourceLabel(skill.source)}</span>
      <h1>${esc(skill.title)}</h1><p>${esc(skill.description)}</p><small>${esc(user ? date(skill.updated_at) : skill.version)}</small></div>
      <div class="eusk-detail-actions">${method
        ? `<span class="eusk-state-button ${skill.enabled ? 'on' : ''}" aria-label="${skill.enabled ? tr('Available', '可用') : tr('Unavailable', '不可用')}">${skill.enabled ? '✓ ' + tr('Available', '可用') : tr('Unavailable', '不可用')}</span>`
        : `<button type="button" class="eusk-state-button ${skill.enabled ? 'on' : ''}" data-sk-toggle="${esc(skill.id)}" ${ui.busy || masterOff ? 'disabled' : ''}>${skill.enabled ? '✓ ' + tr('Enabled', '已启用') : tr('Enable', '启用')}</button>`}
      <button type="button" class="eusk-use" data-sk-use="${esc(skill.id)}" ${skill.enabled ? '' : 'disabled'}>${workflow ? tr('Start workflow', '启动工作流') : method || (user && skill.enabled && skill.stages.includes('conversation')) ? tr('Use in new task', '在新任务中使用') : tr('Ask in new task', '在新任务中询问')} ↗</button>
      ${user ? `<details class="eusk-actions-menu"><summary aria-label="${tr('Skill actions', '技能操作')}">···</summary><div><button type="button" data-sk-edit-with aria-label="${tr('Edit with EasyICU', '使用 EasyICU 修订')}">${tr('Edit with EasyICU', '使用 EasyICU 修订')}</button><button type="button" data-sk-edit aria-label="${tr('Edit SKILL.md directly', '直接编辑 SKILL.md')}">${tr('Edit SKILL.md', '直接编辑 SKILL.md')}</button><button type="button" data-sk-download aria-label="${tr('Download SKILL.md', '下载 SKILL.md')}">${tr('Download SKILL.md', '下载 SKILL.md')}</button><button type="button" data-sk-remove aria-label="${tr('Remove skill', '移除技能')}">${tr('Remove skill', '移除技能')}</button></div></details>` : ''}</div></div>
      <div class="eusk-tabs" role="tablist"><button type="button" role="tab" aria-selected="${ui.tab === 'overview'}" data-sk-tab="overview">${tr('Overview', '概览')}</button><button type="button" role="tab" aria-selected="${ui.tab === 'files'}" data-sk-tab="files">${tr('Files', '文件')}</button></div>
      ${ui.tab === 'files' ? `<div class="eusk-detail-body">${method ? methodPackage(skill, component) : user ? `<div class="eusk-file-row">${icon('file', 15)} SKILL.md <small>${skill.size_bytes || 0} B</small></div>
        ${ui.loadingDetail ? `<p>${tr('Loading the reviewed file…', '正在读取已审阅文件…')}</p>` : detail
          ? `<div class="eusk-file-preview"><h2>SKILL.md</h2><pre>${esc(detail.skill_md || '')}</pre></div>`
          : `<p>${tr('This file could not be read.', '无法读取这个文件。')}</p>`}`
        : `<div class="eusk-empty"><strong>${tr('Host contract', '内置契约')}</strong><p>${tr('This built-in capability is implemented in EasyICU code and has no installable SKILL.md package.', '此内置能力由 EasyICU 代码实现，没有可安装的 SKILL.md 文件包。')}</p></div>`}</div>`
      : method ? `<div class="eusk-detail-body">${methodOverview(skill, component)}</div>`
      : `<div class="eusk-detail-body"><h2>${tr('When to use', '适用场景')}</h2><p>${esc(user ? skill.description : skill.scope)}</p>
        ${user ? `<div class="eusk-contract"><div><h3>${tr('Activation stages', '启用阶段')}</h3>${list(skill.stages)}</div><div><h3>${tr('Version', '版本')}</h3><code>sha256:${esc(String(skill.digest || '').slice(0, 16))}</code></div></div>`
          : `<div class="eusk-contract"><div><h3>${tr('Inputs', '输入')}</h3>${list(skill.inputs)}</div><div><h3>${tr('Outputs', '输出')}</h3>${list(skill.outputs)}</div><div><h3>${tr('Evidence rules', '证据规则')}</h3>${list(skill.invariants)}</div></div>`}
        ${user ? ui.loadingDetail ? `<p>${tr('Loading reviewed instructions…', '正在读取已审阅指令…')}</p>`
          : detail ? `<section class="eusk-skill-instructions"><h2>${tr('Instructions', '技能说明')}</h2>${instructionHtml(detail.instructions)}</section>`
            : `<p>${tr('Instructions are unavailable.', '技能说明暂不可用。')}</p>` : ''}
        <p class="eusk-binding">${masterOff ? tr('The built-in Skill master switch is off in Settings.', '内置技能总开关已在设置中关闭。') + ' ' : ''}${tr('Changes apply to newly created sessions and runs; existing frozen sessions keep their original skill set.', '启用状态对新建会话和运行生效；已有会话保留原来固化的技能集合。')}</p></div>`}`;
  }
  function packageMarkdown(content) {
    return String(content || '').replace(/^---\s*\n[\s\S]*?\n---\s*(?:\n|$)/, '');
  }
  function methodOverview(skill, component) {
    if (ui.loadingPackage) return `<div class="eusk-document-loading"><span></span><p>${tr('Loading the reviewed SKILL.md…', '正在读取已审阅的 SKILL.md…')}</p></div>`;
    const pkg = ui.packageDetail && ui.packageDetail.skill_id === skill.name ? ui.packageDetail : null;
    const main = pkg && Array.isArray(pkg.files) ? pkg.files.find(file => file.path === 'SKILL.md') : null;
    if (!main) return `<div class="eusk-package-fallback">${methodContract(skill, component)}</div>`;
    return `<article class="eusk-rendered-markdown eusk-overview-document" data-sk-overview-source="SKILL.md">${instructionHtml(packageMarkdown(main.content))}</article>`;
  }
  function methodContract(skill, component) {
    return `<div class="eusk-method-summary"><div><span>${component ? tr('Method family', '方法族') : tr('Capability', '能力契约')}</span><code>${esc(component ? skill.methodFamily : skill.capabilityId)}</code></div><div><span>${tr('Execution', '执行方式')}</span><strong>${esc(executionLabel(skill))}</strong></div><div><span>${component ? tr('Method tier', '方法层级') : tr('Claim ceiling', '论断上限')}</span><strong>${esc(component ? tierLabel(skill) : ceilingLabel(skill))}</strong></div></div>
      <h2>${component ? tr('Reviewed implementation', '审阅实现') : tr('Registered method actions', '已登记的方法动作')}</h2>${component ? list([...new Set([skill.methodKey, ...skill.kernelModules])]) : skill.actionIds.length ? list(skill.actionIds) : `<p>${tr('This workflow binds directly to its capability owner.', '此工作流直接绑定能力所有者。')}</p>`}
      <h2>${component ? tr('Reporting bindings', '报告规范') : tr('Required diagnostics', '必要诊断')}</h2>${list(component ? skill.reportingItems : skill.diagnostics)}${skill.skillLayer === 'research_workflow' ? `<h2>${tr('Included reusable modules', '包含的可复用模块')}</h2>${skill.includedModuleIds.length ? list(skill.includedModuleIds) : `<p>${tr('No default modules.', '没有默认模块。')}</p>`}` : ''}<h2>${tr('Registered output', '登记产物')}</h2>${list(skill.outputs)}`;
  }
  function methodPackage(skill, component) {
    if (ui.loadingPackage) return `<p>${tr('Loading the reviewed package…', '正在读取已审阅技能包…')}</p>`;
    const pkg = ui.packageDetail && ui.packageDetail.skill_id === skill.name ? ui.packageDetail : null;
    if (!pkg) return methodContract(skill, component);
    const files = Array.isArray(pkg.files) ? pkg.files : [];
    const active = files.find(file => file.path === ui.packageFile) || files[0] || null;
    const groups = files.reduce((result, file) => {
      const parts = String(file.path || '').split('/');
      const folder = parts.length > 1 ? parts[0] : 'root';
      (result[folder] = result[folder] || []).push({ ...file, displayName: parts[parts.length - 1] });
      return result;
    }, {});
    const order = ['root', 'references', 'scripts', ...Object.keys(groups).filter(key => !['root', 'references', 'scripts'].includes(key))];
    const tree = order.filter(folder => groups[folder] && groups[folder].length).map(folder => `<section class="eusk-package-group"><h3>${folder === 'root' ? tr('Package file', '主文件') : `${icon('folder', 14)} ${esc(folder)}`}</h3>${groups[folder].map(file => `<button type="button" class="${active && active.path === file.path ? 'active' : ''}" data-sk-package-file="${esc(file.path)}">${icon('file', 15)}<span>${esc(file.displayName)}</span><small>${Number(file.size_bytes || 0)} B</small></button>`).join('')}</section>`).join('');
    const renderedMarkdown = active ? packageMarkdown(active.content) : '';
    const activeBody = active && String(active.language || '').toLocaleLowerCase() === 'markdown'
      ? `<div class="eusk-rendered-markdown">${instructionHtml(renderedMarkdown)}</div>`
      : active ? `<pre>${esc(active.content || '')}</pre>` : '';
    return `<div class="eusk-package-note"><strong>${tr('Read-only built-in package', '只读内置技能包')}</strong><span>sha256:${esc(String(pkg.package_sha256 || '').slice(0, 16))}</span><p>${tr('SKILL.md is the main guide. References hold the scientific contract, composition, and validation rules. A scripts folder appears only when this capability has a package-specific implementation; shared host execution is not copied into every package.', 'SKILL.md 是主说明；references 保存科学契约、组合与验证规则。只有该能力确有专用实现时才显示 scripts；共享宿主执行代码不会复制进每个包。')}</p></div>
      <div class="eusk-package-layout"><nav class="eusk-package-files" aria-label="${tr('Skill package files', '技能包文件')}"><div class="eusk-package-tree-head">${tr('Package files', '技能包文件')} <span>${files.length}</span></div>${tree}</nav>
      ${active ? `<div class="eusk-file-preview"><h2>${esc(active.path)} <small>${esc(active.language || '')}</small></h2>${activeBody}</div>` : `<p>${tr('The package has no readable files.', '技能包没有可读取的文件。')}</p>`}</div>`;
  }
  function installer() {
    return `<button type="button" class="eusk-back" data-sk-cancel>← ${tr('Back to skills', '返回技能目录')}</button>
      <div class="eusk-head"><div><span class="eusk-eyebrow">SKILL HUB</span><h1>${ui.editing ? tr('Edit SKILL.md', '编辑 SKILL.md') : tr('Upload a skill', '上传技能')}</h1><p>${tr('Review a local SKILL.md, choose where it applies, then install it for future sessions.', '审阅本地 SKILL.md，选择生效阶段，再安装到后续会话。')}</p></div></div>
      <div class="eusk-install"><div class="eusk-upload-note"><span>${icon('file', 18)}</span><div><strong>${tr('One reviewed SKILL.md file', '一个经过审阅的 SKILL.md 文件')}</strong><p>${tr('EasyICU currently installs a single instruction file up to 12 KB. Supporting-file packages are not yet accepted.', 'EasyICU 当前可安装一个不超过 12 KB 的指令文件，暂不接收包含辅助文件的技能包。')}</p></div></div>
      <label>${tr('Select SKILL.md', '选择 SKILL.md')}<input type="file" accept=".md,text/markdown,text/plain" data-sk-file></label>
      <p>${tr('Optional category: add category: Clinical Research to the SKILL.md frontmatter.', '可在 SKILL.md 的 frontmatter 中写 category: Clinical Research，供目录分类。')}</p>
      <label>${tr('Or paste its complete content', '或粘贴完整内容')}<textarea data-sk-draft rows="12" maxlength="12000" spellcheck="false" placeholder="---&#10;name: concise-writing&#10;description: Keep scientific prose concise.&#10;category: Clinical Research&#10;---&#10;Instructions…">${esc(ui.draft)}</textarea></label>
      <fieldset><legend>${tr('Workflow stages', '工作流阶段')}</legend><label><input type="checkbox" data-sk-stage="conversation" ${ui.stages.includes('conversation') ? 'checked' : ''}> ${tr('Conversation', '对话')}</label><label><input type="checkbox" data-sk-stage="writing" ${ui.stages.includes('writing') ? 'checked' : ''}> ${tr('Writing', '写作')}</label></fieldset>
      <label><input type="checkbox" data-sk-install-enabled ${ui.installEnabled ? 'checked' : ''}> ${tr('Enable after installation', '安装后启用')}</label>
      <div class="eusk-install-foot"><button type="button" data-sk-cancel>${tr('Cancel', '取消')}</button><button type="button" class="eusk-new" data-sk-install ${ui.busy ? 'disabled' : ''}>${ui.busy ? tr('Saving…', '保存中…') : ui.editing ? tr('Save skill', '保存技能') : tr('Install skill', '安装技能')}</button></div></div>`;
  }
  function renderContent() {
    const content = `
      ${ui.error ? `<div class="eusk-message error" role="alert">${esc(ui.error)}</div>` : ''}
      ${ui.notice ? `<div class="eusk-message" role="status">${esc(ui.notice)}</div>` : ''}
      ${ui.install ? installer() : ui.selected ? detail() : catalog()}
    `;
    if (window.EU_DESKTOP_MODULE_SHELL) return window.EU_DESKTOP_MODULE_SHELL.render({
      active: 'skills', shellClass: 'eusk-shell', mainClass: 'eusk-main', innerClass: 'eusk-main-inner',
      label: tr('Research skills', '研究技能'), content,
    });
    return `<div class="eusk-shell">${rail()}<main class="eusk-main"><div class="eusk-main-inner">${content}</div></main></div>`;
  }
  async function hydrate() {
    const requests = [];
    if (api().loadCapabilities) requests.push(api().loadCapabilities());
    if (api().loadExtensions) requests.push(api().loadExtensions());
    const settled = await Promise.allSettled(requests);
    const failed = settled.find(result => result.status === 'rejected');
    if (failed) ui.error = String(failed.reason && failed.reason.message || failed.reason);
    rerender();
  }
  function rerender(focusSearch) {
    if (window.__euRender) window.__euRender();
    if (focusSearch) requestAnimationFrame(() => {
      const input = document.querySelector('[data-sk-search]');
      if (input) { input.focus(); input.setSelectionRange(ui.query.length, ui.query.length); }
    });
  }
  async function openSkill(id) {
    ui.selected = id; ui.tab = 'overview'; ui.detail = null; ui.packageDetail = null;
    ui.packageFile = 'SKILL.md'; ui.loadingPackage = false; ui.error = ''; rerender();
    const skill = selectedSkill();
    if (skill && ['method', 'method_component'].includes(skill.kind)) {
      await loadMethodPackage(id);
      return;
    }
    if (!skill || skill.source !== 'user' || !api().loadExtensionSkill) return;
    ui.loadingDetail = true;
    try { ui.detail = await api().loadExtensionSkill(skill.name); }
    catch (error) { ui.error = String(error && error.message || error); }
    finally { ui.loadingDetail = false; if (ui.selected === id) rerender(); }
  }
  async function loadMethodPackage(expectedSelection = ui.selected) {
    const skill = selectedSkill();
    if (!skill || !['method', 'method_component'].includes(skill.kind)
      || !api().loadBuiltinSkillPackage || ui.loadingPackage) return;
    if (ui.packageDetail && ui.packageDetail.skill_id === skill.name) return;
    ui.loadingPackage = true; ui.error = ''; rerender();
    try {
      const loaded = await api().loadBuiltinSkillPackage(skill.name);
      if (ui.selected === expectedSelection) ui.packageDetail = loaded;
    }
    catch (error) { ui.error = String(error && error.message || error); }
    finally { if (ui.selected === expectedSelection) ui.loadingPackage = false; rerender(); }
  }
  async function toggle(id) {
    const skill = allSkills().find(item => item.id === id);
    if (!skill || ui.busy) return;
    if (skill.source === 'builtin' && !skill.toggleable) return;
    ui.busy = true; ui.error = ''; rerender();
    try {
      if (skill.source === 'builtin' && skill.toggleable) {
        await api().saveSetting(skill.settingKey, !skill.enabled);
        await api().loadCapabilities();
      } else {
        await api().setExtensionState({ kind: 'skill', name: skill.name, enabled: !skill.enabled,
          expected_sha256: registry().activation_sha256 });
      }
      ui.notice = tr('Skill state saved for new sessions and runs.', '技能状态已保存，将用于新会话和运行。');
    } catch (error) {
      ui.error = String(error && error.message || error);
      if (api().loadExtensions) await api().loadExtensions().catch(() => {});
    } finally { ui.busy = false; rerender(); }
  }
  function createPrompt() {
    return tr(
      'Help me create a reusable EasyICU Skill. First clarify its trigger, inputs, outputs, workflow stages, evidence boundaries, and failure behavior. Then return one complete SKILL.md with YAML frontmatter containing name, description, and category. Keep it under 12 KB and make every instruction operational and auditable. Do not install anything until I review the file.',
      '请帮我开发一个可复用的 EasyICU 技能。先和我确认触发条件、输入、输出、工作流阶段、证据边界和失败处理，再返回一个完整的 SKILL.md；YAML frontmatter 必须包含 name、description 和 category。文件控制在 12 KB 内，每条指令都应可执行、可审计。在我审阅文件前不要安装。',
    );
  }
  function editPrompt(file) {
    const intro = tr(
      `Help me revise the installed EasyICU Skill "${file.name}". Preserve its name, identify ambiguous triggers, missing inputs or outputs, evidence-boundary risks, and failure behavior, then return one complete replacement SKILL.md under 12 KB. Do not install it until I review the file.\n\nCurrent reviewed SKILL.md:\n\`\`\`markdown\n`,
      `请帮我修订已安装的 EasyICU 技能「${file.name}」。保持 name 不变，检查触发条件是否含糊、输入输出是否缺失、证据边界和失败处理是否明确，然后返回一个不超过 12 KB 的完整替换版 SKILL.md。在我审阅文件前不要安装。\n\n当前经过审阅的 SKILL.md：\n\`\`\`markdown\n`,
    );
    return `${intro}${file.skill_md}\n\`\`\``;
  }
  function openBuilder(mode, file) {
    const question = mode === 'edit' ? editPrompt(file) : createPrompt();
    if (question.length > 12000) {
      ui.error = tr('This SKILL.md is too long to open safely in the composer. Use the direct editor instead.', '这个 SKILL.md 太长，无法安全放入对话输入框，请改用直接编辑。');
      rerender(); return;
    }
    try {
      window.sessionStorage.setItem('easyicu.skillHub.builder', JSON.stringify({
        mode, name: file && file.name || '', title: mode === 'edit'
          ? tr('Skill revision', '技能修订') : tr('Skill Builder', '技能开发'),
      }));
      window.sessionStorage.setItem('easyicu.skillHub.question', question);
    } catch (_) {}
    location.hash = '#guided';
  }
  async function editWithEasyICU() {
    try {
      const file = await reviewedFile();
      if (file) openBuilder('edit', file);
    } catch (error) { ui.error = String(error && error.message || error); rerender(); }
  }
  function bind(root) {
    const hub = root.querySelector('.eusk-shell');
    if (!hub) return;
    hub.addEventListener('click', event => {
      const target = event.target.closest('button');
      if (!target) return;
      if (target.dataset.skRoute) { location.hash = '#' + target.dataset.skRoute; return; }
      if (target.dataset.skMode) {
        ui.catalogMode = target.dataset.skMode; ui.source = 'all'; ui.category = 'all'; ui.query = '';
        rerender(); return;
      }
      if (target.dataset.skSource) { ui.source = target.dataset.skSource; rerender(); return; }
      if (target.dataset.skOpen) { void openSkill(target.dataset.skOpen); return; }
      if (target.dataset.skBack !== undefined || target.dataset.skCancel !== undefined) {
        ui.selected = ''; ui.install = false; ui.editing = ''; ui.error = ''; rerender(); return;
      }
      if (target.dataset.skTab) {
        ui.tab = target.dataset.skTab; rerender();
        if (ui.tab === 'files') void loadMethodPackage();
        return;
      }
      if (target.dataset.skPackageFile) { ui.packageFile = target.dataset.skPackageFile; rerender(); return; }
      if (target.dataset.skToggle) { void toggle(target.dataset.skToggle); return; }
      if (target.dataset.skEditWith !== undefined) { void editWithEasyICU(); return; }
      if (target.dataset.skEdit !== undefined) { void editSkill(); return; }
      if (target.dataset.skDownload !== undefined) { void downloadSkill(); return; }
      if (target.dataset.skRemove !== undefined) { void removeSkill(); return; }
      if (target.dataset.skUpload !== undefined) { ui.install = true; ui.error = ''; rerender(); return; }
      if (target.dataset.skCreateWith !== undefined) { openBuilder('create'); return; }
      if (target.dataset.skUse) {
        const skill = allSkills().find(item => item.id === target.dataset.skUse);
        if (skill) try {
          if (skill.kind === 'method' || skill.kind === 'method_component') {
            window.sessionStorage.setItem('easyicu.skillHub.method', JSON.stringify({
              id: skill.name, title: skill.title, kind: skill.kind,
              layer: skill.skillLayer || 'method_component',
              capability_id: skill.capabilityId || '', action_ids: skill.actionIds || [],
              included_module_ids: skill.includedModuleIds || [],
              method_family: skill.methodFamily || '', method_key: skill.methodKey || '',
              claim_ceiling: skill.claimCeiling,
            }));
            window.sessionStorage.setItem('easyicu.skillHub.question', skill.prompt);
          } else if (skill.source === 'user' && skill.enabled && skill.stages.includes('conversation')) {
            window.sessionStorage.setItem('easyicu.skillHub.use', JSON.stringify({ name: skill.name, digest: skill.digest }));
          } else {
            window.sessionStorage.setItem('easyicu.skillHub.question',
              tr(`How should I use the ${skill.title} skill in this research project?`, `在当前研究项目中，如何使用「${skill.title}」技能？`));
          }
        } catch (_) {}
        location.hash = '#guided'; return;
      }
      if (target.dataset.skInstall !== undefined) { void install(); }
    });
    hub.addEventListener('input', event => {
      if (event.target.matches('[data-sk-search]')) { ui.query = event.target.value; rerender(true); }
      if (event.target.matches('[data-sk-draft]')) ui.draft = event.target.value;
    });
    hub.addEventListener('change', event => {
      const target = event.target;
      if (target.matches('[data-sk-category]')) { ui.category = target.value; rerender(); }
      if (target.matches('[data-sk-hide]')) { ui.hideDisabled = target.checked; rerender(); }
      if (target.matches('[data-sk-stage]')) {
        ui.stages = Array.from(hub.querySelectorAll('[data-sk-stage]:checked')).map(el => el.dataset.skStage);
      }
      if (target.matches('[data-sk-install-enabled]')) ui.installEnabled = target.checked;
      if (target.matches('[data-sk-file]')) {
        const file = target.files && target.files[0]; if (!file) return;
        if (file.size > 12000) { ui.error = tr('SKILL.md exceeds 12 KB.', 'SKILL.md 超过 12 KB。'); rerender(); return; }
        file.text().then(content => { ui.draft = content; rerender(); });
      }
    });
  }
  async function install() {
    if (!ui.draft.trim() || !ui.stages.length) {
      ui.error = tr('Provide a SKILL.md and select at least one stage.', '请填写 SKILL.md 并选择至少一个阶段。'); rerender(); return;
    }
    if (ui.editing) {
      const nameLine = /^name:\s*["']?([^\s"']+)["']?\s*$/m.exec(ui.draft);
      if (!nameLine || nameLine[1] !== ui.editing) {
        ui.error = tr('Keep the Skill name unchanged while editing.', '编辑技能时请保持名称不变。'); rerender(); return;
      }
    }
    ui.busy = true; ui.error = ''; rerender();
    try {
      const result = await api().installExtensionSkill({ skill_md: ui.draft,
        stages: ui.stages, enabled: ui.installEnabled,
        expected_sha256: registry().activation_sha256 });
      ui.draft = ''; ui.install = false; ui.editing = ''; ui.notice = tr('Skill saved for new sessions and runs.', '技能已保存，将用于新会话和运行。');
      ui.catalogMode = 'skills'; ui.source = 'user'; ui.category = 'all';
      ui.selected = `user:${result.skill.name}`; ui.tab = 'overview';
      await openSkill(ui.selected);
    } catch (error) {
      ui.error = String(error && error.message || error);
      if (api().loadExtensions) await api().loadExtensions().catch(() => {});
    } finally { ui.busy = false; rerender(); }
  }
  async function reviewedFile() {
    const skill = selectedSkill();
    if (!skill || skill.source !== 'user') return null;
    if (ui.detail && ui.detail.name === skill.name) return ui.detail;
    ui.detail = await api().loadExtensionSkill(skill.name);
    return ui.detail;
  }
  async function editSkill() {
    try {
      const file = await reviewedFile();
      if (!file) return;
      ui.editing = file.name; ui.draft = file.skill_md; ui.stages = file.stages.slice();
      ui.installEnabled = file.enabled; ui.install = true; ui.error = ''; rerender();
    } catch (error) { ui.error = String(error && error.message || error); rerender(); }
  }
  async function downloadSkill() {
    try {
      const file = await reviewedFile();
      if (!file) return;
      const url = URL.createObjectURL(new Blob([file.skill_md], { type: 'text/markdown' }));
      const link = document.createElement('a'); link.href = url; link.download = `${file.name}-SKILL.md`;
      document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { ui.error = String(error && error.message || error); rerender(); }
  }
  async function removeSkill() {
    const skill = selectedSkill();
    if (!skill || skill.source !== 'user' || !window.confirm(tr(`Remove ${skill.name} from future sessions?`, `从后续会话中移除「${skill.name}」？`))) return;
    ui.busy = true; ui.error = ''; rerender();
    try {
      await api().removeExtension({ kind: 'skill', name: skill.name,
        expected_sha256: registry().activation_sha256 });
      ui.selected = ''; ui.detail = null;
      ui.notice = tr('Skill removed from future activation.', '技能已从后续激活中移除。');
    } catch (error) { ui.error = String(error && error.message || error); }
    finally { ui.busy = false; rerender(); }
  }
  S.skills = { section: 'skills', full: true, get crumbs() { return [tr('Skills', '技能')]; },
    render: renderContent, afterRender(root) { bind(root); void hydrateIfStale(); } };
  let hydratedAt = 0;
  function hydrateIfStale() {
    if (Date.now() - hydratedAt < 10_000) return;
    hydratedAt = Date.now();
    return hydrate();
  }
})();
