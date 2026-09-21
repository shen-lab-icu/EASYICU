/* Owner: Guided Pi study-workspace widget. */
/* Presentation state for a study: retained conversation and an explicit,
   version-bound artifact reference in the composer. No execution authority. */
(function () {
  'use strict';
  function create({ tr, esc, iconHtml }) {
    let reference = null;
    let picker = { context: '', open: false, query: '', category: 'all', focused: false };
    let skill = null;
    let builder = null;
    let method = null;
    let skillPicker = { context: '', open: false, query: '', catalog: 'skills', category: 'all', focused: false };
    const contextKey = (projectId, sessionId) => JSON.stringify([projectId, sessionId]);
    const materialKey = resource => JSON.stringify([resource.kind, resource.run_id, resource.artifact, resource.sha256]);
    function capture(host) {
      const materials = host && host.querySelector('[data-gpi-material-picker]');
      const materialSearch = materials && materials.querySelector && materials.querySelector('[data-gpi-material-search]');
      if (materials) picker = { ...picker, context: materials.dataset.gpiMaterialPicker,
        open: true, query: materialSearch ? materialSearch.value : picker.query,
        focused: materialSearch === document.activeElement };
      const skills = host && host.querySelector('[data-gpi-skill-picker]');
      const skillSearch = skills && skills.querySelector && skills.querySelector('[data-gpi-skill-search]');
      if (skills) skillPicker = { ...skillPicker, context: skills.dataset.gpiSkillPicker,
        open: true, query: skillSearch ? skillSearch.value : skillPicker.query,
        focused: skillSearch === document.activeElement };
    }
    function availableSkills(session) {
      const frozen = session && session.extension_activation;
      return Array.isArray(frozen && frozen.skills)
        ? frozen.skills.filter(row => (row.stages || []).includes('conversation')) : [];
    }
    function currentSkill(projectId, session) {
      if (!skill || !session || skill.context !== contextKey(projectId, session.session_id)) return null;
      return availableSkills(session).find(row => row.name === skill.name && row.digest === skill.digest) || null;
    }
    function capabilityRoot() { return ((window.EU_CAPABILITIES || {}).capabilities || {}).method_skills || {}; }
    function methodSkills() {
      const rows = Array.isArray(capabilityRoot().items) ? capabilityRoot().items : [];
      return rows.map(row => ({ id: String(row.id || ''), kind: 'method',
        title: tr(String(row.title || ''), String(row.title_zh || row.title || '')),
        category: tr(String(row.category || ''), String(row.category_zh || row.category || '')),
        description: tr(String(row.description || ''), String(row.description_zh || row.description || '')),
        prompt: tr(String(row.prompt || ''), String(row.prompt_zh || row.prompt || '')),
        capabilityId: String(row.capability_id || ''), actionIds: Array.isArray(row.action_ids) ? row.action_ids.map(String) : [],
        claimCeiling: String(row.claim_ceiling || 'analysis_only') })).filter(row => row.id && row.title && row.prompt);
    }
    function methodComponents() {
      const rows = Array.isArray(capabilityRoot().components) ? capabilityRoot().components : [];
      return rows.map(row => ({ id: String(row.id || ''), kind: 'method_component',
        title: tr(String(row.title || ''), String(row.title_zh || row.title || '')),
        category: tr(String(row.category || ''), String(row.category_zh || row.category || '')),
        description: tr(String(row.description || ''), String(row.description_zh || row.description || '')),
        prompt: tr(String(row.prompt || ''), String(row.prompt_zh || row.prompt || '')),
        methodFamily: String(row.method_family || ''), methodKey: String(row.method_key || ''),
        claimCeiling: String(row.claim_ceiling || 'analysis_only') })).filter(row => row.id && row.title && row.methodFamily);
    }
    function openMaterials(projectId, sessionId) {
      picker = { context: contextKey(projectId, sessionId), open: true, query: '', category: 'all', focused: true };
      skillPicker.open = false;
    }
    function openSkills(projectId, session) {
      skillPicker = { context: contextKey(projectId, session && session.session_id), open: true,
        query: '', catalog: 'skills', category: 'all', focused: true };
      picker.open = false;
    }
    function renderSkillPicker(projectId, session, disabled) {
      const context = contextKey(projectId, session && session.session_id);
      if (skillPicker.context !== context || disabled) skillPicker = { context, open: false, query: '', catalog: 'skills', category: 'all', focused: false };
      if (disabled || !skillPicker.open) return '';
      const frozen = availableSkills(session).map(row => ({ ...row, id: `frozen:${row.name}`, title: row.name,
        category: tr('Your Skills', '会话技能'), prompt: row.description || '', kind: 'frozen_skill' }));
      const rows = skillPicker.catalog === 'methods' ? methodComponents() : [...methodSkills(), ...frozen];
      const selected = currentSkill(projectId, session);
      const categories = [...new Set(rows.map(row => row.category).filter(Boolean))];
      return `<section class="gpi-composer-drawer gpi-skill-picker" data-gpi-skill-picker="${esc(context)}" aria-label="${tr('Skills and methods', '技能与方法')}">
        <header class="gpi-composer-drawer-head"><strong>${tr('Skills and methods', '技能与方法')}</strong><button type="button" data-gpi-skill-close aria-label="${tr('Close Skills', '关闭技能')}">×</button></header>
        <label class="gpi-composer-drawer-search"><span aria-hidden="true">⌕</span><input type="search" data-gpi-skill-search aria-label="${tr('Search Skills', '搜索技能')}" placeholder="${tr('Search Skills and methods…', '搜索技能与方法……')}" value="${esc(skillPicker.query)}" autocomplete="off"></label>
        <div class="gpi-composer-drawer-tabs" role="tablist"><button type="button" role="tab" data-gpi-skill-catalog="skills" aria-selected="${skillPicker.catalog === 'skills'}">${tr('Skills', '技能')} <span>${methodSkills().length + frozen.length}</span></button><button type="button" role="tab" data-gpi-skill-catalog="methods" aria-selected="${skillPicker.catalog === 'methods'}">${tr('Method library', '方法库')} <span>${methodComponents().length}</span></button></div>
        <div class="gpi-composer-drawer-body"><nav aria-label="${tr('Skill categories', '技能分类')}"><button type="button" data-gpi-skill-category="all" aria-pressed="${skillPicker.category === 'all'}">${tr('All', '全部')}</button>${categories.map(category => `<button type="button" data-gpi-skill-category="${esc(category)}" aria-pressed="${skillPicker.category === category}">${esc(category)}</button>`).join('')}</nav>
          <div class="gpi-composer-drawer-list">${rows.map(row => `<button type="button" class="gpi-composer-drawer-row" data-gpi-skill-row data-gpi-skill-kind="${esc(row.kind)}" data-gpi-skill-id="${esc(row.id)}" data-gpi-skill-name="${esc(row.name || '')}" data-gpi-skill-digest="${esc(row.digest || '')}" data-gpi-skill-category-value="${esc(row.category || '')}" data-gpi-skill-search-text="${esc(`${row.title} ${row.description || ''} ${row.category || ''}`.toLowerCase())}" ${row.kind === 'frozen_skill' ? `data-gpi-skill-select="${esc(row.name)}" aria-pressed="${Boolean(selected && selected.name === row.name)}"` : 'data-gpi-catalog-select'}><strong>${esc(row.title)}</strong><span>${esc(row.description || row.category)}</span><small>${esc(row.category)}</small></button>`).join('')}<p data-gpi-skill-empty role="status" hidden>${tr('No matching Skills.', '没有匹配的技能。')}</p></div>
        </div><footer><button type="button" data-gpi-skill-hub>${tr('Open the full Skill Hub', '打开完整技能目录')} →</button></footer>
      </section>`;
    }
    function filterSkills(host, query) {
      skillPicker.query = query;
      const menu = host && host.querySelector('[data-gpi-skill-picker]');
      if (!menu) return;
      let count = 0;
      menu.querySelectorAll('[data-gpi-skill-row]').forEach(row => {
        const queryMatch = row.dataset.gpiSkillSearchText.includes(query.trim().toLowerCase());
        const categoryMatch = skillPicker.category === 'all' || row.dataset.gpiSkillCategoryValue === skillPicker.category;
        row.hidden = !(queryMatch && categoryMatch);
        if (!row.hidden) count += 1;
      });
      const empty = menu.querySelector('[data-gpi-skill-empty]');
      if (empty) empty.hidden = count > 0;
    }
    function setSkillCatalog(value) {
      if (!['skills', 'methods'].includes(value)) return false;
      skillPicker.catalog = value; skillPicker.category = 'all'; skillPicker.query = ''; return true;
    }
    function setSkillCategory(value) { skillPicker.category = String(value || 'all'); return true; }
    function selectCatalog(button) {
      const kind = String(button && button.dataset.gpiSkillKind || '');
      const id = String(button && button.dataset.gpiSkillId || '');
      const row = (kind === 'method_component' ? methodComponents() : methodSkills()).find(item => item.id === id);
      if (!row) return null;
      skillPicker.open = false;
      return { kind: 'compose', text: row.prompt || tr(`Use ${row.title} for this research question.`, `请使用“${row.title}”处理这个研究问题。`), intent: 'implement_scientific_question', method: row };
    }
    function selectSkill(button, projectId, session) {
      const row = button.closest('[data-gpi-skill-row]');
      if (!row || !session) return false;
      const chosen = availableSkills(session).find(item => item.name === row.dataset.gpiSkillName
        && item.digest === row.dataset.gpiSkillDigest);
      if (!chosen) return false;
      const context = contextKey(projectId, session.session_id);
      skill = currentSkill(projectId, session)?.name === chosen.name ? null
        : { context, name: chosen.name, digest: chosen.digest };
      skillPicker.open = false;
      return true;
    }
    function selectFrozenSkill(projectId, session, name, digest) {
      const chosen = availableSkills(session).find(item => item.name === name && item.digest === digest);
      if (!chosen) return false;
      skill = { context: contextKey(projectId, session.session_id), name, digest };
      return true;
    }
    function selectMethodTemplate(projectId, session, descriptor) {
      if (!session || !descriptor || !descriptor.id
          || !(descriptor.capabilityId || descriptor.methodFamily)) return false;
      method = {
        context: contextKey(projectId, session.session_id),
        id: String(descriptor.id),
        title: String(descriptor.title || descriptor.id),
        kind: String(descriptor.kind || 'method'),
        capabilityId: String(descriptor.capabilityId || ''),
        methodFamily: String(descriptor.methodFamily || ''),
        methodKey: String(descriptor.methodKey || ''),
        actionIds: Array.isArray(descriptor.actionIds) ? descriptor.actionIds.map(String) : [],
        claimCeiling: String(descriptor.claimCeiling || 'analysis_only'),
      };
      return true;
    }
    function applyHubIntent(projectId, session) {
      try {
        const builderIntent = JSON.parse(window.sessionStorage.getItem('easyicu.skillHub.builder') || 'null');
        if (builderIntent && session && (builderIntent.mode === 'create' || builderIntent.mode === 'edit')) {
          builder = { context: contextKey(projectId, session.session_id), mode: builderIntent.mode,
            name: String(builderIntent.name || ''), title: String(builderIntent.title || tr('Skill Builder', '技能开发')) };
          window.sessionStorage.removeItem('easyicu.skillHub.builder');
        }
        const intent = JSON.parse(window.sessionStorage.getItem('easyicu.skillHub.use') || 'null');
        if (intent && selectFrozenSkill(projectId, session, intent.name, intent.digest)) {
          window.sessionStorage.removeItem('easyicu.skillHub.use');
        }
        const methodIntent = JSON.parse(window.sessionStorage.getItem('easyicu.skillHub.method') || 'null');
        if (methodIntent && session && methodIntent.id && (methodIntent.capability_id || methodIntent.method_family)) {
          method = { context: contextKey(projectId, session.session_id),
            id: String(methodIntent.id), title: String(methodIntent.title || methodIntent.id),
            kind: String(methodIntent.kind || 'method'),
            capabilityId: String(methodIntent.capability_id || ''),
            methodFamily: String(methodIntent.method_family || ''),
            methodKey: String(methodIntent.method_key || ''),
            actionIds: Array.isArray(methodIntent.action_ids) ? methodIntent.action_ids.map(String) : [],
            claimCeiling: String(methodIntent.claim_ceiling || 'analysis_only') };
          window.sessionStorage.removeItem('easyicu.skillHub.method');
        }
      } catch (_) { /* A malformed or unavailable browser store cannot change the frozen session. */ }
    }
    function hasHubIntent() {
      try { return Boolean(window.sessionStorage.getItem('easyicu.skillHub.use')
        || window.sessionStorage.getItem('easyicu.skillHub.question')
        || window.sessionStorage.getItem('easyicu.skillHub.builder')
        || window.sessionStorage.getItem('easyicu.skillHub.method')); } catch (_) { return false; }
    }
    function closeSkills(host, focus) {
      skillPicker.open = false; skillPicker.focused = false;
      const drawer = host && host.querySelector('[data-gpi-skill-picker]');
      if (drawer && drawer.remove) drawer.remove();
      if (focus) {
        const trigger = host && host.querySelector('.gpi-idea-source-menu > summary');
        if (trigger) trigger.focus();
      }
    }
    function renderSkillReference(projectId, session) {
      const selected = currentSkill(projectId, session);
      const currentBuilder = builder && session && builder.context === contextKey(projectId, session.session_id) ? builder : null;
      const currentMethod = method && session && method.context === contextKey(projectId, session.session_id) ? method : null;
      const builderChip = currentBuilder ? `<div class="gpi-composer-skill gpi-builder-context" role="status">✧ <strong>${esc(currentBuilder.title)}</strong>${currentBuilder.name ? `<span>${esc(currentBuilder.name)}</span>` : ''}<button type="button" data-gpi-builder-hub>${tr('Skill Hub', '技能目录')} ↗</button><button type="button" data-gpi-builder-remove aria-label="${tr('Remove builder context', '移除技能开发上下文')}">×</button></div>` : '';
      const methodChip = currentMethod ? `<div class="gpi-composer-skill gpi-method-context" role="status"><span>${currentMethod.kind === 'method_component' ? tr('Component', '组件') : tr('Method', '方法')}</span><strong>${esc(currentMethod.title)}</strong><button type="button" data-gpi-method-remove aria-label="${tr('Remove method', '移除方法')}">×</button></div>` : '';
      const skillChip = selected ? `<div class="gpi-composer-skill" role="status">✧ <strong>${esc(selected.name)}</strong><button type="button" data-gpi-skill-remove aria-label="${tr('Remove Skill', '移除技能')}">×</button></div>` : '';
      return builderChip + methodChip + skillChip;
    }
    function decorateSkillMessage(text, projectId, session) {
      const selected = currentSkill(projectId, session);
      const currentMethod = method && session && method.context === contextKey(projectId, session.session_id) ? method : null;
      let decorated = text;
      if (selected && String(decorated || '').trim()) decorated = `${decorated}\n\n${tr('For this question, use the Skill frozen into this session:', '本次提问请使用此会话已固化的技能：')} ${selected.name} (sha256:${selected.digest}). ${tr('Load it with easyicu_load_skill before applying its instructions.', '请先用 easyicu_load_skill 加载精确版本，再按其指令处理。')}`;
      if (currentMethod && String(decorated || '').trim()) decorated = `${decorated}\n\nEasyICU method workflow request:\n${JSON.stringify({ method_skill_id: currentMethod.id,
        kind: currentMethod.kind, capability_id: currentMethod.capabilityId,
        method_family: currentMethod.methodFamily, method_key: currentMethod.methodKey,
        action_ids: currentMethod.actionIds,
        claim_ceiling: currentMethod.claimCeiling })}`;
      return decorated;
    }
    function consumeSkill(projectId, session) {
      if (currentSkill(projectId, session)) skill = null;
      if (method && session && method.context === contextKey(projectId, session.session_id)) method = null;
    }
    function removeSkill() { skill = null; }
    function removeBuilder() { builder = null; }
    function removeMethod() { method = null; }
    function renderAccessMode(mode, label, iconHtml) {
      const modes = [
        ['ask', tr('Ask before every tool action', '每次工具操作前都询问')],
        ['assist', tr('Auto-approve low-risk setup and inspection; ask before extraction and full analysis', '自动批准低风险配置与检查；提取和完整分析前仍询问')],
        ['full', tr('Allow all available tools; explicit scientific confirmation gates still apply', '允许所有可用工具；明确的科学确认门禁仍然有效')],
      ];
      const currentDescription = (modes.find(([key]) => key === mode) || modes[1])[1];
      return `<details class="gpi-access-menu">
        <summary aria-label="${esc(`${label(mode)}：${currentDescription}`)}" title="${esc(currentDescription)}">${iconHtml(mode === 'full' ? 'unlock' : 'shield', 15)}<span>${esc(label(mode))}</span><span class="gpi-access-chevron" aria-hidden="true">${iconHtml('chevron', 13)}</span></summary>
        <div class="gpi-access-popover" role="group" aria-label="${tr('Agent access level', 'Agent 访问级别')}">
          ${modes.map(([key, description]) => `<button type="button" data-gpi-access-mode="${key}" aria-pressed="${mode === key}"><span><strong>${esc(label(key))}</strong><small>${esc(description)}</small></span>${mode === key ? iconHtml('check', 15) : ''}</button>`).join('')}
          <p>${tr('Access levels never reveal credentials, patient rows, or arbitrary host files.', '任何访问级别都不会开放凭据、患者行级数据或任意本机文件。')}</p>
        </div>
      </details>`;
    }
    function renderMaterials(resources, projectId, sessionId, disabled) {
      const context = contextKey(projectId, sessionId);
      if (picker.context !== context || disabled) picker = { context, open: false, query: '', category: 'all', focused: false };
      if (disabled || !picker.open) return '';
      const rows = resources.filter(canReference);
      const classify = row => row.artifact.includes('figure') || /\.(png|svg|tiff?)$/i.test(row.artifact) ? tr('Figures', '图表')
        : row.artifact.includes('table') || /\.(csv|xlsx?)$/i.test(row.artifact) ? tr('Tables', '表格')
          : row.kind === 'research_document' || /\.(pdf|md)$/i.test(row.artifact) ? tr('Documents', '文稿') : tr('Reports', '报告');
      const categories = [...new Set(rows.map(classify))];
      return `<section class="gpi-composer-drawer gpi-material-picker" data-gpi-material-picker="${esc(context)}" aria-label="${tr('Project resources', '项目资料')}">
        <header class="gpi-composer-drawer-head"><strong>${tr('Project resources', '项目资料')}</strong><button type="button" data-gpi-material-close aria-label="${tr('Close materials', '关闭资料')}">×</button></header>
        <label class="gpi-composer-drawer-search"><span aria-hidden="true">⌕</span><input type="search" data-gpi-material-search aria-label="${tr('Search project materials', '搜索项目资料')}" placeholder="${tr('Search project resources…', '搜索项目资料……')}" value="${esc(picker.query)}" autocomplete="off"></label>
        <div class="gpi-composer-drawer-body"><nav aria-label="${tr('Resource categories', '资料分类')}"><button type="button" data-gpi-material-category="all" aria-pressed="${picker.category === 'all'}">${tr('All', '全部')}</button>${categories.map(category => `<button type="button" data-gpi-material-category="${esc(category)}" aria-pressed="${picker.category === category}">${esc(category)}</button>`).join('')}</nav>
          <div class="gpi-composer-drawer-list">${rows.length ? rows.map(row => { const category = classify(row); return `<button type="button" class="gpi-composer-drawer-row" data-gpi-material-row data-gpi-material-reference data-gpi-material-key="${esc(materialKey(row))}" data-gpi-material-category-value="${esc(category)}" data-gpi-material-search-text="${esc(`${row.label} ${row.artifact} ${category}`.toLowerCase())}"><strong>${esc(row.label)}</strong><span>${esc(row.artifact)}</span><small>${esc(category)}</small></button>`; }).join('') : `<p class="gpi-material-empty">${tr('No results yet. Completed results will appear here.', '当前还没有研究成果，完成研究后会显示在这里。')}</p>`}<p data-gpi-material-empty role="status" hidden>${tr('No matching results.', '没有匹配的资料。')}</p></div>
        </div>
      </section>`;
    }
    function filterMaterials(host, query) {
      const menu = host && host.querySelector('[data-gpi-material-picker]');
      if (!menu) return;
      picker.query = query;
      let count = 0;
      menu.querySelectorAll('[data-gpi-material-row]').forEach(row => {
        const queryMatch = row.dataset.gpiMaterialSearchText.includes(query.trim().toLowerCase());
        const categoryMatch = picker.category === 'all' || row.dataset.gpiMaterialCategoryValue === picker.category;
        row.hidden = !(queryMatch && categoryMatch);
        if (!row.hidden) count++;
      });
      const empty = menu.querySelector('[data-gpi-material-empty]');
      if (empty) empty.hidden = count > 0;
    }
    function restoreMaterials(host) {
      filterMaterials(host, picker.query);
      filterSkills(host, skillPicker.query);
      const search = host && host.querySelector('[data-gpi-material-search]');
      if (search && picker.open && picker.focused) {
        search.focus({ preventScroll: true });
      }
      const skillSearch = host && host.querySelector('[data-gpi-skill-search]');
      if (skillSearch && skillPicker.open && skillPicker.focused) skillSearch.focus({ preventScroll: true });
    }
    function selectedMaterial(button, resources, projectId, sessionId) {
      const menu = button.closest('[data-gpi-material-picker]');
      const row = button.closest('[data-gpi-material-row]');
      if (!menu || !row || menu.dataset.gpiMaterialPicker !== contextKey(projectId, sessionId)) return null;
      return resources.find(resource => canReference(resource) && materialKey(resource) === row.dataset.gpiMaterialKey) || null;
    }
    function closeMaterials(host, focus) {
      picker.open = false; picker.focused = false;
      const drawer = host && host.querySelector('[data-gpi-material-picker]');
      if (drawer && drawer.remove) drawer.remove();
      if (focus) {
        const trigger = host && host.querySelector('.gpi-idea-source-menu > summary');
        if (trigger) trigger.focus();
      }
    }
    function setMaterialCategory(value) { picker.category = String(value || 'all'); return true; }
    function syncNavigation(ctx) {
      const rail = document.getElementById('gdConversationRail');
      if (!rail) return;
      rail.hidden = !ctx.visible || !ctx.projectId;
      const locked = ctx.loading || ctx.disabled;
      const materials = (Array.isArray(ctx.resources) ? ctx.resources : []).filter(canReference).slice(0, 12);
      const materialsInteractive = ctx.materialsInteractive !== false;
      rail.innerHTML = `<div class="gpi-conversations-main"><div class="gpi-conversations-heading"><strong>${tr('Tasks', '任务')}</strong><span class="gpi-conversations-heading-actions"><button type="button" data-gpi-rail-task-search-toggle aria-label="${tr('Search tasks', '搜索任务')}" title="${tr('Search', '搜索')}">${iconHtml('search', 15)}</button><button type="button" data-gpi-rail-new ${locked ? 'disabled' : ''} aria-label="${tr('New conversation in this project', '在当前项目新建对话')}">${iconHtml('plus', 14)}<span>${tr('Task', '任务')}</span></button></span></div>
        <label class="gpi-conversation-search" hidden><span class="sr-only">${tr('Search tasks', '搜索任务')}</span><input type="search" data-gpi-rail-task-search placeholder="${tr('Search tasks…', '搜索任务…')}" autocomplete="off"></label>
        <nav aria-label="${tr('Conversations in this project', '当前项目中的对话')}">${ctx.loading
          ? `<p role="status">${tr('Loading conversations…', '正在读取对话…')}</p>`
          : ctx.sessions.length ? ctx.sessions.map(row => {
            const status = typeof ctx.status === 'function' ? ctx.status(row)
              : (row.agent_mode === 'workspace' ? tr('Workspace', '工作区') : tr('Research', '研究'));
            const time = typeof ctx.time === 'function' ? ctx.time(row) : '';
            const stateClass = row.active_message_job_id || row.last_turn_status === 'running' ? ' is-running'
              : ['failed', 'interrupted', 'cancelled'].includes(String(row.last_turn_status || '')) ? ' needs-attention'
              : row.last_turn_status === 'done' ? ' is-complete' : '';
            const title = ctx.title(row);
            const canRemove = !row.has_history && !row.active_message_job_id && !row.last_message_job_id && !row.last_turn_status;
            return `<div class="gpi-conversation-row" data-gpi-rail-session-row data-gpi-rail-session-text="${esc(`${title} ${status}`.toLowerCase())}"><button type="button" class="gpi-conversation-item${stateClass}" data-gpi-rail-session="${esc(row.session_id)}" ${row.session_id === ctx.selectedId ? 'aria-current="page"' : ''} ${locked ? 'disabled' : ''}><span class="gpi-conversation-copy"><strong>${esc(title)}</strong><small><i aria-hidden="true"></i>${esc(status)}</small></span>${time ? `<time datetime="${esc(row.last_activity_at || row.created_at || '')}">${esc(time)}</time>` : ''}</button><details class="gpi-conversation-menu"><summary aria-label="${esc(tr('Task actions', '任务操作'))}" title="${esc(tr('Task actions', '任务操作'))}">•••</summary><div><button type="button" data-gpi-rail-rename="${esc(row.session_id)}" ${locked ? 'disabled' : ''}>${tr('Rename', '重命名')}</button>${canRemove ? `<button type="button" data-gpi-rail-remove="${esc(row.session_id)}" ${locked ? 'disabled' : ''}>${tr('Remove empty task', '删除空任务')}</button>` : ''}</div></details></div>`;
          }).join('')
            : `<p>${tr('No conversations yet. Start one in this project.', '暂无对话，可在当前项目中开始。')}</p>`}</nav></div>
        <section class="gpi-project-materials" aria-label="${tr('Drive', '资料')}"><div class="gpi-conversations-heading"><strong>${tr('Drive', '资料')}</strong><button type="button" data-gpi-rail-material-search-toggle aria-label="${tr('Search Drive', '搜索资料')}" title="${tr('Search', '搜索')}">${iconHtml('search', 15)}</button></div>
          <label class="gpi-project-material-search" hidden><span class="sr-only">${tr('Search Drive', '搜索资料')}</span><input type="search" data-gpi-rail-material-search placeholder="${tr('Filter files…', '筛选文件…')}" autocomplete="off"></label>
          ${materials.length ? `<div class="gpi-project-material-list">${materials.map((row, i) => materialsInteractive
            ? `<button type="button" data-gpi-rail-material="${i}" data-gpi-rail-material-text="${esc(String(row.label || row.artifact).toLowerCase())}" title="${esc(row.label || row.artifact)}"><span aria-hidden="true">${iconHtml('file', 14)}</span>${esc(row.label || row.artifact)}</button>`
            : `<div class="gpi-project-material-row" data-gpi-rail-material-text="${esc(String(row.label || row.artifact).toLowerCase())}"><span aria-hidden="true">${iconHtml('file', 14)}</span>${esc(row.label || row.artifact)}</div>`).join('')}</div>` : `<p>${tr('No project results yet', '暂无可引用成果')}</p>`}
        </section>`;
      rail.onclick = event => {
        const taskSearchToggle = event.target.closest('[data-gpi-rail-task-search-toggle]');
        if (taskSearchToggle) {
          const label = rail.querySelector('.gpi-conversation-search');
          if (label) {
            label.hidden = !label.hidden;
            if (!label.hidden) label.querySelector('input').focus();
          }
          return;
        }
        const searchToggle = event.target.closest('[data-gpi-rail-material-search-toggle]');
        if (searchToggle) {
          const label = rail.querySelector('.gpi-project-material-search');
          if (label) {
            label.hidden = !label.hidden;
            if (!label.hidden) label.querySelector('input').focus();
          }
          return;
        }
        const material = event.target.closest('[data-gpi-rail-material]');
        if (material && typeof ctx.openResource === 'function') {
          const row = materials[Number(material.dataset.gpiRailMaterial)];
          if (row) ctx.openResource(row);
          return;
        }
        if (locked) return;
        const rename = event.target.closest('[data-gpi-rail-rename]');
        if (rename && typeof ctx.rename === 'function') {
          const row = ctx.sessions.find(item => item.session_id === rename.dataset.gpiRailRename);
          if (row) ctx.rename(row);
          return;
        }
        const remove = event.target.closest('[data-gpi-rail-remove]');
        if (remove && typeof ctx.remove === 'function') {
          const row = ctx.sessions.find(item => item.session_id === remove.dataset.gpiRailRemove);
          if (row) ctx.remove(row);
          return;
        }
        const session = event.target.closest('[data-gpi-rail-session]');
        if (session && ctx.sessions.some(row => row.session_id === session.dataset.gpiRailSession)) ctx.open(session.dataset.gpiRailSession);
        else if (event.target.closest('[data-gpi-rail-new]')) ctx.create();
      };
      rail.oninput = event => {
        if (event.target.matches('[data-gpi-rail-task-search]')) {
          const query = event.target.value.trim().toLowerCase();
          rail.querySelectorAll('[data-gpi-rail-session-row]').forEach(row => {
            row.hidden = Boolean(query && !row.dataset.gpiRailSessionText.includes(query));
          });
          return;
        }
        if (!event.target.matches('[data-gpi-rail-material-search]')) return;
        const query = event.target.value.trim().toLowerCase();
        rail.querySelectorAll('[data-gpi-rail-material-text]').forEach(row => {
          row.hidden = Boolean(query && !row.dataset.gpiRailMaterialText.includes(query));
        });
      };
    }
    function canReference(resource) {
      return Boolean(resource && /^research_(artifact|report|document)$/.test(resource.kind)
        && /^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(resource.run_id || '')
        && /^[A-Za-z0-9_.-]{1,160}$/.test(resource.artifact || '')
        && /^[a-f0-9]{64}$/.test(resource.sha256 || ''));
    }
    function setReference(resource, projectId, sessionId) {
      if (!projectId || !sessionId || !canReference(resource)) return false;
      reference = { projectId, sessionId, resource: { kind: resource.kind,
        run_id: resource.run_id, artifact: resource.artifact, sha256: resource.sha256,
        label: String(resource.label || resource.artifact).slice(0, 160) } };
      return true;
    }
    function current(projectId, sessionId) {
      return reference && reference.projectId === projectId && reference.sessionId === sessionId ? reference : null;
    }
    function renderReference(projectId, sessionId) {
      const selected = current(projectId, sessionId);
      if (!selected) return '';
      return `<div class="gpi-composer-reference" role="status"><div><span>${tr('Referencing', '已引用')}</span><strong>${esc(selected.resource.label)}</strong><small>${tr('This version · add your question below', '当前版本 · 在下方填写你的问题')}</small></div>
        <button type="button" data-gpi-reference-remove aria-label="${tr('Remove reference', '移除引用')}">×</button></div>`;
    }
    function decorateMessage(text, projectId, sessionId) {
      const selected = current(projectId, sessionId);
      if (!selected || !String(text || '').trim()) return text;
      return `${text}\n\n${tr('Referenced project artifact (existing result):', '引用的项目资料（已有成果）：')}\n${JSON.stringify({ project_id: projectId, ...selected.resource })}`;
    }
    function messageView(row, projectId) {
      if (!row || row.role !== 'user') return row;
      const text = String(row.text || '');
      const match = /\n\n(?:Referenced project artifact \(existing result\):|引用的项目资料（已有成果）：)\n(\{[^\n]+\})$/.exec(text);
      let view = row;
      if (match) {
        try {
          const resource = JSON.parse(match[1]);
          if (resource.project_id === projectId && canReference(resource)) {
            view = { ...row, text: text.slice(0, match.index), resources: [...(Array.isArray(row.resources) ? row.resources : []), resource] };
          }
        } catch (_) { /* Untrusted or stale reference remains plain text. */ }
      }
      const methodMatch = /\n\nEasyICU method workflow request:\n(\{[^\n]+\})$/.exec(String(view.text || ''));
      if (methodMatch) {
        try {
          const requestedMethod = JSON.parse(methodMatch[1]);
          if (requestedMethod.method_skill_id && (requestedMethod.capability_id || requestedMethod.method_family)) {
            view = { ...view, text: view.text.slice(0, methodMatch.index), requestedMethod };
          }
        } catch (_) { /* Invalid method metadata remains visible for audit. */ }
      }
      const skillMatch = /\n\n(?:For this question, use the Skill frozen into this session:|本次提问请使用此会话已固化的技能：) ([a-z0-9][a-z0-9-]*) \(sha256:[a-f0-9]{64}\)\. (?:Load it with easyicu_load_skill before applying its instructions\.|请先用 easyicu_load_skill 加载精确版本，再按其指令处理。)$/.exec(String(view.text || ''));
      if (skillMatch) view = { ...view, text: view.text.slice(0, skillMatch.index), requestedSkill: skillMatch[1] };
      return view;
    }
    function consume(projectId, sessionId) { if (current(projectId, sessionId)) reference = null; }
    function removeReference() { reference = null; }
    return { capture, syncNavigation, messageView, openMaterials, renderMaterials, filterMaterials, setMaterialCategory, restoreMaterials, selectedMaterial, closeMaterials,
      openSkills, renderSkillPicker, filterSkills, setSkillCatalog, setSkillCategory, selectCatalog, selectSkill, selectFrozenSkill, selectMethodTemplate, applyHubIntent, hasHubIntent, closeSkills, renderSkillReference, decorateSkillMessage, consumeSkill, removeSkill, removeBuilder, removeMethod, renderAccessMode,
      hasReference: (projectId, sessionId) => Boolean(current(projectId, sessionId)), canReference, setReference, renderReference, decorateMessage, consume, removeReference };
  }
  window.EasyICU.guidedPi.declare('studyWorkspace', { create });
})();
