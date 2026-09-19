/* Owner: Guided Pi study-workspace widget. */
/* Presentation state for a study: retained conversation and an explicit,
   version-bound artifact reference in the composer. No execution authority. */
(function () {
  'use strict';
  function create({ tr, esc }) {
    let reference = null;
    let picker = { context: '', open: false, query: '', focused: false };
    let skill = null;
    let builder = null;
    let method = null;
    let skillPicker = { context: '', open: false, query: '' };
    const contextKey = (projectId, sessionId) => JSON.stringify([projectId, sessionId]);
    const materialKey = resource => JSON.stringify([resource.kind, resource.run_id, resource.artifact, resource.sha256]);
    function capture(host) {
      const menu = host && host.querySelector('[data-gpi-material-picker]');
      const search = menu && menu.querySelector && menu.querySelector('[data-gpi-material-search]');
      if (search) picker = { context: menu.dataset.gpiMaterialPicker, open: menu.open,
        query: search.value, focused: search === document.activeElement };
      const skills = host && host.querySelector('[data-gpi-skill-picker]');
      const skillSearch = skills && skills.querySelector('[data-gpi-skill-search]');
      if (skills) skillPicker = { context: skills.dataset.gpiSkillPicker, open: skills.open,
        query: skillSearch ? skillSearch.value : '' };
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
    function renderSkillPicker(projectId, session, disabled) {
      const context = contextKey(projectId, session && session.session_id);
      if (skillPicker.context !== context || disabled) skillPicker = { context, open: false, query: '' };
      if (disabled) return `<button class="gpi-material-trigger" type="button" disabled>${tr('Skills', '技能')}</button>`;
      const rows = availableSkills(session);
      const selected = currentSkill(projectId, session);
      return `<details class="gpi-skill-picker" data-gpi-skill-picker="${esc(context)}"${skillPicker.open ? ' open' : ''}>
        <summary class="gpi-material-trigger" aria-label="${tr('Choose a Skill for this question', '为这次提问选择技能')}">✧ ${tr('Skills', '技能')} <span aria-hidden="true">⌃</span></summary>
        <section class="gpi-skill-panel" aria-label="${tr('Skills in this session', '当前会话的技能')}">
          <div class="gpi-material-heading"><strong>${tr('Skills in this session', '当前会话的技能')}</strong><button type="button" data-gpi-skill-close aria-label="${tr('Close Skills', '关闭技能')}">×</button></div>
          <p>${tr('Only Skills frozen into this session can be selected. New changes apply to a new task.', '只能选择已固化到此会话的技能。新启用的技能用于新任务。')}</p>
          ${rows.length ? `<input type="search" data-gpi-skill-search aria-label="${tr('Search Skills', '搜索技能')}" placeholder="${tr('Search Skills…', '搜索技能…')}" value="${esc(skillPicker.query)}" autocomplete="off">
            <ul>${rows.map(row => `<li data-gpi-skill-row data-gpi-skill-name="${esc(row.name)}" data-gpi-skill-digest="${esc(row.digest)}" data-gpi-skill-search-text="${esc(`${row.name} ${row.description}`.toLowerCase())}"><button type="button" data-gpi-skill-select="${esc(row.name)}" aria-pressed="${Boolean(selected && selected.name === row.name)}"><strong>${esc(row.name)}</strong><small>${esc(row.description)}</small></button></li>`).join('')}</ul>
            <p data-gpi-skill-empty role="status" hidden>${tr('No matching Skills.', '没有匹配的技能。')}</p>`
            : `<p class="gpi-material-empty">${tr('No conversation Skills are active in this session.', '此会话没有启用的对话技能。')}</p>`}
          <button type="button" class="gpi-skill-hub-link" data-gpi-skill-hub>${tr('View Skill Hub', '查看技能目录')} ↗</button>
        </section></details>`;
    }
    function filterSkills(host, query) {
      skillPicker.query = query;
      const menu = host && host.querySelector('[data-gpi-skill-picker]');
      if (!menu) return;
      let count = 0;
      menu.querySelectorAll('[data-gpi-skill-row]').forEach(row => {
        row.hidden = !row.dataset.gpiSkillSearchText.includes(query.trim().toLowerCase());
        if (!row.hidden) count += 1;
      });
      const empty = menu.querySelector('[data-gpi-skill-empty]');
      if (empty) empty.hidden = count > 0;
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
      skillPicker.open = false;
      const menu = host && host.querySelector('[data-gpi-skill-picker]');
      if (!menu) return;
      menu.open = false;
      if (focus) menu.querySelector('summary').focus();
    }
    function renderSkillReference(projectId, session) {
      const selected = currentSkill(projectId, session);
      const currentBuilder = builder && session && builder.context === contextKey(projectId, session.session_id) ? builder : null;
      const currentMethod = method && session && method.context === contextKey(projectId, session.session_id) ? method : null;
      const builderChip = currentBuilder ? `<div class="gpi-composer-skill gpi-builder-context" role="status">✧ <strong>${esc(currentBuilder.title)}</strong>${currentBuilder.name ? `<span>${esc(currentBuilder.name)}</span>` : ''}<button type="button" data-gpi-builder-hub>${tr('Skill Hub', '技能目录')} ↗</button><button type="button" data-gpi-builder-remove aria-label="${tr('Remove builder context', '移除技能开发上下文')}">×</button></div>` : '';
      const methodChip = currentMethod ? `<div class="gpi-composer-skill gpi-method-context" role="status">${currentMethod.kind === 'method_component' ? tr('Component', '组件') : tr('Method', '方法')} · <strong>${esc(currentMethod.title)}</strong><span>${currentMethod.claimCeiling === 'reportable' ? tr('reportable contract', '可报告契约') : tr('analysis only', '仅分析')}</span><button type="button" data-gpi-method-hub>${tr('Skill Hub', '技能目录')} ↗</button><button type="button" data-gpi-method-remove aria-label="${tr('Remove method', '移除方法')}">×</button></div>` : '';
      const skillChip = selected ? `<div class="gpi-composer-skill" role="status">✧ <strong>${esc(selected.name)}</strong><span>${tr('This question', '本次提问')}</span><button type="button" data-gpi-skill-remove aria-label="${tr('Remove Skill', '移除技能')}">×</button></div>` : '';
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
      return `<details class="gpi-access-menu">
        <summary>${iconHtml(mode === 'full' ? 'unlock' : 'shield', 15)}<span>${esc(label(mode))}</span><span class="gpi-access-chevron" aria-hidden="true">${iconHtml('chevron', 13)}</span></summary>
        <div class="gpi-access-popover" role="group" aria-label="${tr('Agent access level', 'Agent 访问级别')}">
          ${modes.map(([key, description]) => `<button type="button" data-gpi-access-mode="${key}" aria-pressed="${mode === key}"><span><strong>${esc(label(key))}</strong><small>${esc(description)}</small></span>${mode === key ? iconHtml('check', 15) : ''}</button>`).join('')}
          <p>${tr('Access levels never reveal credentials, patient rows, or arbitrary host files.', '任何访问级别都不会开放凭据、患者行级数据或任意本机文件。')}</p>
        </div>
      </details>`;
    }
    function renderMaterials(resources, projectId, sessionId, disabled) {
      const context = contextKey(projectId, sessionId);
      if (picker.context !== context || disabled) picker = { context, open: false, query: '', focused: false };
      if (disabled) return `<button class="gpi-material-trigger" type="button" disabled>${tr('Materials', '资料')}</button>`;
      const rows = resources.filter(canReference);
      return `<details class="gpi-material-picker" data-gpi-material-picker="${esc(context)}"${picker.open ? ' open' : ''}>
        <summary class="gpi-material-trigger" aria-label="${tr('Reference project materials', '引用项目资料')}">${tr('Materials', '资料')} <span aria-hidden="true">⌃</span></summary>
        <section class="gpi-material-panel" aria-label="${tr('Project materials', '项目资料')}">
          <div class="gpi-material-heading"><strong>${tr('Current research results', '当前研究成果')}</strong><button type="button" data-gpi-material-close aria-label="${tr('Close materials', '关闭资料')}">×</button></div>
          <p>${tr('Preview or reference a result. Your question stays as written.', '先预览，或引用后继续提问。保留你正在写的问题。')}</p>
          ${rows.length ? `<input type="search" data-gpi-material-search aria-label="${tr('Search project materials', '搜索项目资料')}" placeholder="${tr('Search name or file…', '搜索名称或文件……')}" value="${esc(picker.query)}" autocomplete="off">
          <ul>${rows.map(row => `<li data-gpi-material-row data-gpi-material-key="${esc(materialKey(row))}" data-gpi-material-search-text="${esc(`${row.label} ${row.artifact}`.toLowerCase())}"><div><strong>${esc(row.label)}</strong><small>${esc(row.artifact)}</small></div><button type="button" data-gpi-material-preview aria-label="${esc(tr('Preview ', '预览') + row.label)}">${tr('Preview', '预览')}</button><button type="button" data-gpi-material-reference aria-label="${esc(tr('Reference ', '引用') + row.label)}">${tr('Reference', '引用')}</button></li>`).join('')}</ul>
          <p data-gpi-material-empty role="status" hidden>${tr('No matching results. Try another name.', '没有匹配的资料，试试其他名称。')}</p>`
          : `<p class="gpi-material-empty">${tr('No results yet. Completed results will be available here for reference.', '当前还没有研究成果。生成后，可在这里选择并引用。')}</p>`}
        </section></details>`;
    }
    function filterMaterials(host, query) {
      const menu = host && host.querySelector('[data-gpi-material-picker]');
      if (!menu) return;
      picker.query = query;
      let count = 0;
      menu.querySelectorAll('[data-gpi-material-row]').forEach(row => {
        row.hidden = !row.dataset.gpiMaterialSearchText.includes(query.trim().toLowerCase());
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
    }
    function selectedMaterial(button, resources, projectId, sessionId) {
      const menu = button.closest('[data-gpi-material-picker]');
      const row = button.closest('[data-gpi-material-row]');
      if (!menu || !row || menu.dataset.gpiMaterialPicker !== contextKey(projectId, sessionId)) return null;
      return resources.find(resource => canReference(resource) && materialKey(resource) === row.dataset.gpiMaterialKey) || null;
    }
    function closeMaterials(host, focus) {
      picker.open = false; picker.focused = false;
      const menu = host && host.querySelector('[data-gpi-material-picker]');
      if (!menu) return;
      menu.open = false;
      if (focus) menu.querySelector('summary').focus();
    }
    function syncNavigation(ctx) {
      const rail = document.getElementById('gdConversationRail');
      if (!rail) return;
      rail.hidden = !ctx.visible || !ctx.projectId;
      const locked = ctx.loading || ctx.disabled;
      const materials = (Array.isArray(ctx.resources) ? ctx.resources : []).filter(canReference).slice(0, 12);
      rail.innerHTML = `<div class="gpi-conversations-main"><div class="gpi-conversations-heading"><strong>${tr('Tasks', '任务')}</strong><button type="button" data-gpi-rail-new ${locked ? 'disabled' : ''} aria-label="${tr('New conversation in this project', '在当前项目新建对话')}">+</button></div>
        <nav aria-label="${tr('Conversations in this project', '当前项目中的对话')}">${ctx.loading
          ? `<p role="status">${tr('Loading conversations…', '正在读取对话…')}</p>`
          : ctx.sessions.length ? ctx.sessions.map(row => {
            const status = typeof ctx.status === 'function' ? ctx.status(row)
              : (row.agent_mode === 'workspace' ? tr('Workspace', '工作区') : tr('Research', '研究'));
            const time = typeof ctx.time === 'function' ? ctx.time(row) : '';
            const stateClass = row.active_message_job_id || row.last_turn_status === 'running' ? ' is-running'
              : ['failed', 'interrupted', 'cancelled'].includes(String(row.last_turn_status || '')) ? ' needs-attention'
              : row.last_turn_status === 'done' ? ' is-complete' : '';
            return `<button type="button" class="gpi-conversation-item${stateClass}" data-gpi-rail-session="${esc(row.session_id)}" ${row.session_id === ctx.selectedId ? 'aria-current="page"' : ''} ${locked ? 'disabled' : ''}><span class="gpi-conversation-copy"><strong>${esc(ctx.title(row))}</strong><small><i aria-hidden="true"></i>${esc(status)}</small></span>${time ? `<time datetime="${esc(row.last_activity_at || row.created_at || '')}">${esc(time)}</time>` : ''}</button>`;
          }).join('')
            : `<p>${tr('No conversations yet. Start one in this project.', '暂无对话，可在当前项目中开始。')}</p>`}</nav></div>
        <section class="gpi-project-materials" aria-label="${tr('Project materials', '项目资料')}"><div class="gpi-conversations-heading"><strong>${tr('Drive', '资料')}</strong><small>${materials.length || ''}</small></div>
          ${materials.length ? `<div class="gpi-project-material-list">${materials.map((row, i) => `<button type="button" data-gpi-rail-material="${i}" title="${esc(row.label || row.artifact)}">${esc(row.label || row.artifact)}</button>`).join('')}</div>` : `<p>${tr('No project results yet', '暂无可引用成果')}</p>`}
        </section>`;
      rail.onclick = event => {
        const material = event.target.closest('[data-gpi-rail-material]');
        if (material && typeof ctx.openResource === 'function') {
          const row = materials[Number(material.dataset.gpiRailMaterial)];
          if (row) ctx.openResource(row);
          return;
        }
        if (locked) return;
        const session = event.target.closest('[data-gpi-rail-session]');
        if (session && ctx.sessions.some(row => row.session_id === session.dataset.gpiRailSession)) ctx.open(session.dataset.gpiRailSession);
        else if (event.target.closest('[data-gpi-rail-new]')) ctx.create();
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
    return { capture, syncNavigation, messageView, renderMaterials, filterMaterials, restoreMaterials, selectedMaterial, closeMaterials,
      renderSkillPicker, filterSkills, selectSkill, selectFrozenSkill, applyHubIntent, hasHubIntent, closeSkills, renderSkillReference, decorateSkillMessage, consumeSkill, removeSkill, removeBuilder, removeMethod, renderAccessMode,
      hasReference: (projectId, sessionId) => Boolean(current(projectId, sessionId)), canReference, setReference, renderReference, decorateMessage, consume, removeReference };
  }
  window.EasyICU.guidedPi.declare('studyWorkspace', { create });
})();
