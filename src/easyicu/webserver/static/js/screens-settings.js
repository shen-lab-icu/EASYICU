/* Owner: Settings route. */
/* Screen: Settings — local paths, data mode, privacy, model, language, about.
   Utility page reached from the sidebar gear. Bound controls persist through
   /api/settings; path pickers list local folders through the FastAPI process. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  // Mirrors easyicu.webserver.settings.DEFAULTS. Every key here is one the
  // backend actually reads; retired keys are rejected with a 400, so drawing a
  // control for one would be a control that cannot be saved.
  const DEFAULT_SETTINGS = {
    ai_enabled: false,
    language: 'en',
    science_skills_enabled: true,
    nature_figure_skill_enabled: true,
    nature_writing_skill_enabled: true,
    connector_pubmed_enabled: true,
    connector_zotero_enabled: false,
    mcp_tools_enabled: false,
    tool_audit_enabled: true,
    density: 'comfortable',
    reduce_motion: false,
  };
  let settingsPickerEl = null;
  let settingsNotice = '';
  let settingsCapabilityTab = 'overview';
  let settingsAuditEvents = null;
  let settingsAuditLoading = false;
  let settingsZoteroTesting = false;
  let settingsZoteroTest = null;
  let settingsEditingPath = '';
  let settingsKeys = [];
  let settingsKeysReadAt = 0;
  let settingsKeysLoading = false;
  let settingsStorageError = '';
  let settingsKeysError = '';
  const settingsPathDrafts = {};

  function storageSection() {
    let returnRoute = '';
    try { returnRoute = window.sessionStorage.getItem('easyicu.settings.storageReturn') || ''; } catch (_) {}
    return `<div class="sec-stack" id="set-storage"><h2>${T('Storage locations', '存储位置')}</h2></div>
      <div class="card pad settings-storage">
        ${[['data_dir', T('Data folder', '数据目录')], ['project_dir', T('Default project folder', '默认项目目录')]].map(([key, label]) => row(label,
          `<span class="settings-storage-path">${h(S0()[key] || '')}</span>`,
          settingsEditingPath === key ? `<form data-storage-form="${key}"><label><span class="sr-only">${label}</span><input name="path" aria-label="${label}" value="${h(settingsPathDrafts[key] ?? S0()[key] ?? '')}" required></label><button type="submit" class="btn">${T('Save', '保存')}</button><button type="button" class="btn" data-storage-cancel>${T('Cancel', '取消')}</button></form>` : `<button type="button" class="btn" data-storage-change="${key}">${T('Change', '更改')}</button>`)).join('')}
        ${settingsStorageError ? `<p class="settings-storage-error" role="alert">${h(settingsStorageError)}</p>` : ''}
        ${returnRoute ? `<button type="button" class="btn" data-storage-return>${T('Return to dataset', '返回数据卡片')}</button>` : ''}
      </div>`;
  }

  function rememberedKeysSection() {
    if ((S0().about || {}).deployment !== 'desktop') return '';
    return `<div class="sec-stack" id="set-keys"><h2>${T('Model credentials', '模型密钥')}</h2></div><div class="card pad">${settingsKeys.length ? settingsKeys.map(item => row(h(item.provider), h(item.base_url), `<span>${T('Key remembered', '已记住密钥')} · <button type="button" class="btn sm" data-forget-key="${h(item.id)}">${T('Delete', '删除')}</button></span>`)).join('') : settingsKeysError ? '' : `<p>${T('No remembered keys.', '尚未记住密钥。')}</p>`}${settingsKeysError ? `<p class="settings-key-error" role="alert">${h(settingsKeysError)}</p>` : ''}</div>`;
  }

  function sw(on, extra = '', settingKey = '') {
    const ds = settingKey ? ` data-setting="${settingKey}"` : '';
    return `<span class="switch ${on ? 'on' : ''} ${extra}" role="switch" aria-checked="${on}" tabindex="0"${ds}></span>`;
  }

  // Bound segmented control: options = [[value, label], ...]; current = persisted value.
  function segBound(settingKey, options, current) {
    const btns = options.map(([val, label, cls]) =>
      `<button class="${val === current ? 'active' : ''} ${cls || ''}" data-val="${val}">${label}</button>`).join('');
    return `<div class="seg" data-setting="${settingKey}">${btns}</div>`;
  }

  function S0() { return window.EU_SETTINGS || {}; }
  function C0() { return window.EU_CAPABILITIES || {}; }
  function cap(id) {
    const policy = C0();
    const caps = policy && policy.capabilities ? policy.capabilities : {};
    return caps[id] || {};
  }
  function h(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }
  function T(en, zh) { return t(en, zh); }
  function extensionUi() { return window.EU_SETTINGS_EXTENSIONS || {}; }
  function personalSection() { return window.EU_SETTINGS_PERSONALIZATION ? window.EU_SETTINGS_PERSONALIZATION.section() : ''; }

  function row(t, d, ctl) {
    return `<div class="set-row"><div class="sr-main"><div class="sr-t">${t}</div><div class="sr-d">${d}</div></div><div class="sr-ctl">${ctl}</div></div>`;
  }
  function lockedCtl(label, note) {
    return `<span class="pill locked-setting" title="${h(note || '')}">${icon('lock', 12)} ${h(label)}</span>`;
  }
  function truthCtl(label) {
    return `<span class="pill ok-setting">${icon('check', 12)} ${h(label)}</span>`;
  }

  // The remote-compute panel used to render `dual(status, reason)` — but those
  // are two different backend values, not a translation pair, so it printed
  // "disabled / remote_compute_enabled_false". Machine reason codes belong in
  // the API, not on a clinician's screen.
  function computeStateText(remote) {
    const state = String((remote || {}).status || 'disabled');
    return {
      disabled: T('off — local compute only', '已关闭 — 仅本地计算'),
      adapter_not_configured: T(
        'switched on, but no adapter is configured yet',
        '开关已开，但尚未配置适配器'),
    }[state] || state;
  }

  function statusText(status) {
    if (!status) return '';
    const statusMap = {
      disabled: dual('Off', '关闭'),
      available: dual('Reachable', '可连接'),
      unavailable: dual('Unavailable', '无法连接'),
      unsupported: dual('Desktop only', '仅桌面可用'),
      literature_source_ready: dual('literature source ready', '文献来源已就绪'),
      pasted_source_ready: dual('pasted source ready', '粘贴文献已就绪'),
    };
    const reasonMap = {
      connector_zotero_enabled_false: dual('off; pasted source import still works in Idea Mining', '已关闭；Idea Mining 仍可粘贴文献导入'),
      local_zotero_api_ready: dual('desktop link ready', '桌面连接已就绪'),
      local_zotero_api_http_error: dual('desktop link returned an error', '桌面连接返回错误'),
      local_zotero_api_unavailable: dual('desktop app not reachable', '暂时无法连接桌面应用'),
      local_zotero_api_invalid_response: dual('not a Zotero API response', '未返回有效的 Zotero 数据'),
      zotero_desktop_only: dual('not available on the hosted server', '托管版不连接服务器上的 Zotero'),
      local_zotero_search_failed: dual('desktop search failed', '桌面检索失败'),
      local_zotero_item_fetch_failed: dual('paper fetch failed', '文献读取失败'),
      pasted_source_ready: dual('pasted source ready', '粘贴文献已就绪'),
      literature_source_ready: dual('literature source ready', '文献来源已就绪'),
    };
    const state = statusMap[status.status] || status.status;
    const reason = reasonMap[status.reason] || status.reason;
    return [state, reason].filter(Boolean).join(' · ');
  }

  function zoteroTestBlock(zotero) {
    const status = settingsZoteroTest || zotero || {};
    const ready = !!status.available;
    const tone = ready ? 'ok' : status.enabled ? 'warn' : 'idle';
    const label = statusText(status) || dual('Not checked yet', '尚未检查');
    return `
      <div class="settings-zotero-test ${tone}">
        <div class="settings-inline-main">
          <b>${dual('Connection test', '连接测试')}</b>
          <span>${h(label)}</span>
        </div>
        <button type="button" class="btn sm" data-settings-zotero-test ${settingsZoteroTesting ? 'aria-disabled="true"' : ''}>${settingsZoteroTesting ? '<span class="spin"></span>' : icon('search', 12)} ${settingsZoteroTesting ? dual('Checking', '检查中') : dual('Test Zotero', '测试 Zotero')}</button>
      </div>`;
  }

  function auditEventsBlock(audit) {
    const payload = settingsAuditEvents || {};
    const rows = Array.isArray(payload.events) ? payload.events.slice().reverse() : [];
    const count = settingsAuditEvents ? Number(payload.count || rows.length || 0) : Number((audit || {}).event_count || 0);
    return `
      <div class="settings-audit-log">
        <div class="settings-inline-main">
          <b>${dual('Recent audit events', '最近审计事件')}</b>
          <span>${count} ${dual('events recorded locally', '条事件保存在本机')}</span>
        </div>
        <button type="button" class="btn sm" data-settings-audit-refresh ${settingsAuditLoading ? 'aria-disabled="true"' : ''}>${settingsAuditLoading ? '<span class="spin"></span>' : icon('refresh', 12)} ${settingsAuditLoading ? dual('Loading', '加载中') : dual('Refresh', '刷新')}</button>
        ${rows.length ? `<div class="settings-audit-list">
          ${rows.slice(0, 8).map(event => `<div class="settings-audit-event">
            <span class="settings-audit-type">${h(event.event_type || 'tool_event')}</span>
            <span class="settings-audit-meta">${h(event.ts || '')}</span>
          </div>`).join('')}
        </div>` : `<div class="settings-audit-empty">${dual('Refresh to inspect the local audit ledger.', '点击刷新查看本地审计账本。')}</div>`}
        <div class="settings-audit-path mono">${h((payload && payload.path) || (audit && audit.path) || '')}</div>
      </div>`;
  }

  // The Capabilities block was written before this screen was bilingual and
  // concatenated both scripts ("Overview / 总览"), so it ignored the language
  // setting the same page offers. Keep the call sites, honour the setting.
  function dual(en, zh) {
    return T(en, zh);
  }

  function settingOn(key, fallback) {
    const capSettings = (C0() && C0().settings) || {};
    if (Object.prototype.hasOwnProperty.call(capSettings, key)) return !!capSettings[key];
    const settings = S0();
    if (Object.prototype.hasOwnProperty.call(settings, key)) return !!settings[key];
    return !!fallback;
  }

  function savedSettingMessage(key) {
    const labels = {
      science_skills_enabled: T('Built-in research skills', '内置研究技能'),
      nature_figure_skill_enabled: T('Nature Figure', 'Nature 图件'),
      nature_writing_skill_enabled: T('Nature Writing', 'Nature 写作'),
    };
    const label = labels[key];
    return label
      ? T(`${label} saved.`, `${label}已保存。`)
      : T('Setting saved.', '设置已保存。');
  }

  function capStatus(on) {
    return `<span class="settings-cap-status ${on ? 'on' : 'off'}"><span class="dot"></span>${on ? dual('On', '开启') : dual('Off', '关闭')}</span>`;
  }

  function capRow(key, title, desc, fallback) {
    const on = settingOn(key, fallback);
    const active = key.startsWith('nature_') ? settingOn('science_skills_enabled', true) && on : on;
    return row(title, desc, `<div class="settings-cap-control">${capStatus(active)}${sw(on, '', key)}</div>`);
  }

  function capabilityTabs() {
    return [
      ['overview', dual('Overview', '总览')],
      ['skills', dual('Skills', '技能')],
      ['connectors', dual('Connectors', '连接器')],
      ['mcp', dual('MCP tools', 'MCP 工具')],
      ['prompts', dual('Prompt contracts', '提示词契约')],
      ['audit', dual('Audit & compute', '审计与计算')],
    ];
  }

  function capabilityKeys() {
    return [
      ['science_skills_enabled', true],
      ['connector_pubmed_enabled', true],
      ['connector_zotero_enabled', false],
      ['mcp_tools_enabled', false],
      ['tool_audit_enabled', true],
    ];
  }

  function capabilitySummaryTile(iconName, label, enabled, detail) {
    return `<div class="settings-cap-tile ${enabled ? 'on' : 'off'}">
      <span class="settings-cap-tile-icon">${icon(iconName, 14)}</span>
      <div><b>${label}</b><span>${detail}</span></div>
      ${capStatus(enabled)}
    </div>`;
  }

  function builtInSkillCards() {
    const records = cap('publication_skills');
    const skills = Array.isArray(records.items) ? records.items : [];
    const titles = {
      'nature-figure': dual('Evidence figures', '证据图件'),
      'nature-writing': dual('Evidence writing', '证据写作'),
    };
    const scopes = {
      'nature-figure': dual('', '先确定图件论断，再由登记的源数据与代码生成数值结果图。'),
      'nature-writing': dual('', '按段落职责组织论文论证，并逐项绑定证据和文献。'),
    };
    const list = values => `<ul>${(Array.isArray(values) ? values : []).map(value => `<li>${h(value)}</li>`).join('')}</ul>`;
    return `<div class="settings-skill-catalog" aria-label="${dual('Built-in skill packages', '内置技能包')}">
      <div class="settings-skill-catalog-head"><strong>${dual('Built-in skill packages', '内置技能包')}</strong><span>${dual('Select a package to inspect its contract. Enabled skills take effect on new runs.', '点击查看适用范围和契约；启用状态对新运行生效。')}</span></div>
      ${skills.map(skill => `<details class="settings-skill-card">
        <summary><span><small>${h(skill.stage || '')}</small><strong>${h(titles[skill.id] || skill.title || skill.id)}</strong><span>${h(window.EU_LANG === 'zh' && scopes[skill.id] || skill.scope || '')}</span></span><em class="${skill.enabled ? 'on' : ''}">${skill.enabled ? dual('Enabled', '已启用') : dual('Off', '未启用')}</em></summary>
        <div class="settings-skill-card-body"><div><b>${dual('Inputs', '输入')}</b>${list(skill.inputs)}</div><div><b>${dual('Outputs', '输出')}</b>${list(skill.outputs)}</div><div><b>${dual('Evidence rules', '证据规则')}</b>${list(skill.invariants)}</div><small>${h(skill.version || '')}</small></div>
      </details>`).join('')}
    </div>`;
  }

  function capabilityBody() {
    const pubmedOn = settingOn('connector_pubmed_enabled', true);
    const zoteroOn = settingOn('connector_zotero_enabled', false);
    const mcpOn = settingOn('mcp_tools_enabled', false);
    const zotero = cap('zotero_connector');
    const mcp = cap('mcp_tools');
    const audit = cap('tool_audit');
    const remote = cap('remote_compute');
    const allowedTools = Array.isArray(mcp.allowed_tools) ? mcp.allowed_tools.length : 0;
    const blockedTools = Array.isArray(mcp.blocked_tools) ? mcp.blocked_tools.length : 0;
    const enabledCount = capabilityKeys().filter(([key, fallback]) => settingOn(key, fallback)).length;
    if (settingsCapabilityTab === 'skills') {
      const skillsMasterOn = settingOn('science_skills_enabled', true);
      const natureFigureOn = skillsMasterOn && settingOn('nature_figure_skill_enabled', true);
      const natureWritingOn = skillsMasterOn && settingOn('nature_writing_skill_enabled', true);
      const activeSkillCount = [natureFigureOn, natureWritingOn].filter(Boolean).length;
      return `<div class="settings-cap-rows">
        ${capRow('science_skills_enabled', dual('Built-in research skills', '内置研究技能'), dual('Master switch for default Agent workflow skills and their Agent Science cards.', '控制默认 Agent 工作流技能及其 Agent Science 卡片的总开关。'), true)}
        ${capRow('nature_figure_skill_enabled', dual('Nature Figure', 'Nature 图件'), dual('Plans a claim-first figure and renders numeric results only from registered source data and code.', '先规划论断与证据链；数值结果图只能由已登记源数据和代码生成。'), true)}
        ${capRow('nature_writing_skill_enabled', dual('Nature Writing', 'Nature 写作'), dual('Adds paragraph-role, terminology, calibrated-claim, evidence, literature, and novelty rules to Writer.', '为 Writer 加入段落职责、术语一致性、措辞校准、证据、文献和创新性规则。'), true)}
        ${row(dual('Run binding', '运行绑定'), dual('The active switches are frozen into each new Agent run and written to an evidence receipt. Existing runs do not change retroactively.', '每次新 Agent 运行都会固化当前开关并写入证据回执；已有运行不会被追溯修改。'), truthCtl(`${activeSkillCount}/2 ${dual('active by default', '默认启用')}`))}
        ${row(dual('Skill scope', '技能范围'), dual('These skills coordinate local workflow contracts. They do not send patient rows outside this machine.', '这些技能编排本地工作流契约，不会把患者行发出本机。'), lockedCtl(dual('evidence-bound', '证据约束'), dual('Result figures and manuscript claims remain source-bound and audited.', '结果图与论文论断始终绑定来源并接受审计。')))}
        ${builtInSkillCards()}
        ${extensionUi().renderSkills ? extensionUi().renderSkills() : ''}
      </div>`;
    }
    if (settingsCapabilityTab === 'connectors') {
      return `<div class="settings-cap-rows">
        ${capRow('connector_pubmed_enabled', dual('Online literature · PubMed / OpenAlex', '在线文献 · PubMed / OpenAlex'), dual('Allows new literature searches in Idea Mining and before planning, with consent for each search or run. Existing bound literature is retained.', '允许 Idea Mining 和规划前发起在线文献检索，仍需按次检索或运行授权；已绑定文献保留。'), true)}
        ${capRow('connector_zotero_enabled', dual('Zotero desktop link', 'Zotero 桌面连接'), dual('Optional shortcut for searching Zotero Desktop. Idea Mining can still import pasted DOI, BibTeX, RIS, or title/abstract metadata without this switch.', '用于检索 Zotero Desktop 的可选快捷方式。即使不开启，Idea Mining 也能直接导入粘贴的 DOI、BibTeX、RIS 或标题摘要元数据。'), false)}
        ${row(dual('Search consent', '检索授权'), dual('The global switch permits access; each online search or planning run also needs consent.', '全局开关控制可用范围，在线检索或规划运行仍需单次授权。'), lockedCtl(dual('required per search', '按次授权')))}
        ${zoteroTestBlock(zotero)}
      </div>`;
    }
    if (settingsCapabilityTab === 'mcp') {
      return `<div class="settings-cap-rows">
        ${capRow('mcp_tools_enabled', dual('Conversation MCP tools', '对话 MCP 工具'), dual('Allows installed servers in new conversations. Each read-only call also requires a turn grant; turning this off blocks calls in existing sessions.', '允许新对话使用已安装服务。只读调用仍需按次授权；关闭后也会阻止已有会话调用。'), false)}
        ${row(dual('Tool allowlist', '工具白名单'), dual('External tools must be explicitly scoped before they can be used from research workflows.', '外部工具必须先明确作用域，之后才能在研究工作流中使用。'), lockedCtl(dual('required', '必须要求'), dual('Tool scope is an audit contract, not a cosmetic preference.', '工具作用域是审计契约，不是视觉选项。')))}
        ${row(dual('Installed tools', '已安装工具'), dual('Counts installed allowlist entries available to new conversations, before per-turn authorization.', '按已安装服务的白名单项统计新对话可用工具，调用前仍需按次授权。'), lockedCtl(`${allowedTools}/${allowedTools + blockedTools} ${dual('enabled', '可用')}`))}
        ${extensionUi().renderMcp ? extensionUi().renderMcp() : ''}
      </div>`;
    }
    if (settingsCapabilityTab === 'prompts') {
      return `<div class="settings-cap-rows">
        ${row(dual('Built-in prompts', '内置提示词'), dual('Host prompts are versioned with EasyICU. Study-specific configuration is reviewed in the conversation.', '内置提示词随 EasyICU 版本提供；研究专属配置在研究对话中审阅。'), lockedCtl(dual('host-owned', '内置')))}
        ${row(dual('Study configuration', '研究配置'), dual('Review the study plan and its constraints in the conversation.', '在研究对话中审阅计划与约束。'), `<button class="btn sm" data-settings-open="guided">${dual('Open conversation', '打开研究对话')}</button>`)}
      </div>`;
    }
    if (settingsCapabilityTab === 'audit') {
      return `<div class="settings-cap-rows">
        ${capRow('tool_audit_enabled', dual('Connector event log', '连接器事件日志'), dual('Records connector checks and access decisions locally. Run evidence receipts are recorded separately by the execution owner.', '在本机记录连接检查与访问决策；运行证据回执由执行流程单独记录。'), true)}
        ${row(dual('Audit events', '审计事件'), dual('Tool and connector decisions are written only when the audit ledger switch is on.', '只有工具审计账本开启时，工具和连接器决策才会写入。'), truthCtl(`${Number(audit.event_count || 0)} ${dual('recent events', '条近期事件')}`))}
        ${auditEventsBlock(audit)}
        ${row(dual('Compute', '计算'), dual('No remote adapter is installed. Non-local compute requests are rejected.', '尚无远程计算适配器，非本地计算请求会被拒绝。'), lockedCtl(computeStateText(remote)))}
      </div>`;
    }
    return `<div class="settings-cap-overview">
      <div class="settings-cap-meter">
        <b>${enabledCount}/${capabilityKeys().length}</b>
        <span>${dual('enabled options', '已开启选项')}</span>
      </div>
      <div class="settings-cap-grid">
        ${capabilitySummaryTile('layers', dual('Skills', '技能'), settingOn('science_skills_enabled', true), dual('local workflow templates', '本地工作流模板'))}
        ${capabilitySummaryTile('db', dual('Connectors', '连接器'), pubmedOn || zoteroOn, pubmedOn ? dual('PubMed / OpenAlex enabled; consent required', 'PubMed / OpenAlex 已开启；仍需授权') : zoteroOn ? statusText(zotero) : dual('off', '关闭'))}
        ${capabilitySummaryTile('globe', dual('MCP tools', 'MCP 工具'), mcpOn, `${allowedTools}/${allowedTools + blockedTools || 0} ${dual('tools allowed', '工具允许')}`)}
        <div class="settings-cap-tile"><div><b>${dual('Prompt contracts', '提示词契约')}</b><span>${dual('Built into this version', '随版本提供')}</span></div>${lockedCtl(dual('host-owned', '内置'))}</div>
        ${capabilitySummaryTile('shield', dual('Audit', '审计'), settingOn('tool_audit_enabled', true), `${Number(audit.event_count || 0)} ${dual('events', '事件')}`)}
        <div class="settings-cap-tile"><div><b>${dual('Compute', '计算')}</b><span>${dual('No remote adapter', '无远程适配器')}</span></div>${lockedCtl(dual('local only', '仅本地计算'))}</div>
      </div>
      <div class="settings-cap-actions">
        <button class="btn sm" data-settings-open="guided">${icon('agent', 12)} ${dual('Open conversation', '打开研究对话')}</button>
      </div>
    </div>`;
  }

  function capabilityManager() {
    const tabs = capabilityTabs();
    if (!tabs.some(([id]) => id === settingsCapabilityTab)) settingsCapabilityTab = 'overview';
    return `
      <div class="sec-stack" id="set-capabilities"><div class="lbl">${dual('Capabilities', '能力')}</div><h2>${dual('Research capability manager', '研究能力管理')}</h2></div>
      <section class="settings-cap-panel">
        <div class="settings-cap-head">
          <div>
            <div class="eyebrow">${dual('Research controls', '研究控制')}</div>
            <p>${dual('Manage research skills, connectors, MCP tools, prompt contracts, audit, and compute controls from one local settings surface.', '在一个本地设置界面管理研究技能、连接器、MCP 工具、提示词契约、审计和计算控制。')}</p>
          </div>
          <span class="pill info"><span class="dot"></span>${dual('local settings', '本地设置')}</span>
        </div>
        <div class="settings-cap-tabs" role="tablist" aria-label="${dual('Research capability tabs', '研究能力分栏')}">
          ${tabs.map(([id, label]) => `<button type="button" class="settings-cap-tab ${settingsCapabilityTab === id ? 'on' : ''}" role="tab" aria-selected="${settingsCapabilityTab === id ? 'true' : 'false'}" data-settings-cap-tab="${id}">${label}</button>`).join('')}
        </div>
        <div class="settings-cap-body" role="tabpanel">${capabilityBody()}</div>
      </section>`;
  }

  // Rendered from about().local_access, which host_security computes from the
  // live host policy. This row used to be the literal string "enforced" next
  // to a switch hardcoded to on, so it would have shown a green tick even with
  // EASYICU_WEB_ALLOW_ANY_HOST or EASYICU_WEB_TRUST_PROXY widening access.
  function localAccessRow() {
    const access = (S0().about || {}).local_access || {};
    const known = Object.prototype.hasOwnProperty.call(access, 'enforced');
    const enforced = !!access.enforced;
    const hosts = Array.isArray(access.allowed_hosts) ? access.allowed_hosts : [];
    const detail = !known
      ? T('Waiting for the backend to report its host policy.', '正在等待后端返回其主机策略。')
      : enforced
        ? T('The server accepts loopback clients only and rejects proxy-forwarded requests. This controls access to EasyICU; model requests follow the separate setting below.', '服务器只接受回环客户端，并拒绝经代理转发的请求。这限制了谁能访问 EasyICU；模型请求由下方的独立设置控制。')
        : T('Host access has been widened by an environment variable. Clients allowed by this host policy may access EasyICU through the configured proxy or network.', '主机访问范围已通过环境变量扩大。主机策略允许的客户端可以通过已配置的代理或网络访问 EasyICU。');
    const ctl = !known
      ? lockedCtl(T('unknown', '未知'), T('Backend has not reported local_access yet.', '后端尚未返回 local_access。'))
      : enforced
        ? truthCtl(T('loopback only', '仅回环'))
        : `<span class="pill warn">${icon('shield', 12)} ${T('widened', '已放宽')}</span>`;
    const hostNote = hosts.length
      ? `<div class="sr-d mono" style="margin-top:4px;">${T('allowed hosts', '允许的主机')}: ${h(hosts.join(', '))}${access.proxy_headers_trusted ? ` · ${T('proxy trusted', '信任代理')}` : ''}</div>`
      : '';
    return `<div class="set-row"><div class="sr-main"><div class="sr-t">${T('Local-only access', '仅本地访问')}</div><div class="sr-d">${detail}</div>${hostNote}</div><div class="sr-ctl">${ctl}</div></div>`;
  }

  function pathCtl(key, fallback, iconName) {
    const value = S0()[key] || '';
    const label = value || fallback;
    const cls = value ? 'pf-path' : 'pf-path muted';
    return `<div class="path-field"><span class="pf-ico">${icon(iconName, 14)}</span><span class="${cls}">${h(label)}</span></div><button class="btn" data-setting-path="${key}">${T('Change', '更改')}</button>`;
  }



  function closeSettingsPicker() {
    if (settingsPickerEl) {
      settingsPickerEl.remove();
      settingsPickerEl = null;
    }
    document.removeEventListener('keydown', settingsPickerKey);
  }

  function settingsPickerKey(e) {
    if (e.key === 'Escape') closeSettingsPicker();
  }

  function openSettingsFolderPicker(startPath, title, onPick) {
    if (!window.EU_API || !window.EU_API.listDir) return;
    closeSettingsPicker();
    let cur = startPath || '';
    const back = document.createElement('div');
    back.className = 'eu-pick-back';
    back.innerHTML = `
      <div class="eu-pick" role="dialog" aria-label="${h(title)}">
        <div class="eu-pick-h">
          <span style="color:var(--ink-3);">${icon('folder', 16)}</span>
          <span class="t">${h(title)}</span>
          <span class="grow" style="flex:1;"></span>
          <button class="btn sm ghost" data-pk-close>${icon('close', 13)}</button>
        </div>
        <div class="eu-pick-cur" data-pk-cur></div>
        <div class="eu-pick-sc" data-pk-sc></div>
        <div class="eu-pick-list" data-pk-list><div class="eu-pick-empty">${T('Loading...', '正在加载...')}</div></div>
        <div class="eu-pick-f">
          <button class="btn ghost sm" data-pk-up>${icon('back', 13)} ${T('Up', '上一级')}</button>
          <span style="flex:1;"></span>
          <button class="btn primary" data-pk-use>${icon('check', 13)} ${T('Use this folder', '使用此文件夹')}</button>
        </div>
      </div>`;
    document.body.appendChild(back);
    settingsPickerEl = back;
    const listEl = back.querySelector('[data-pk-list]');
    const curEl = back.querySelector('[data-pk-cur]');
    const scEl = back.querySelector('[data-pk-sc]');
    back.addEventListener('click', e => { if (e.target === back) closeSettingsPicker(); });
    back.querySelector('[data-pk-close]').addEventListener('click', closeSettingsPicker);
    back.querySelector('[data-pk-use]').addEventListener('click', () => {
      const chosen = cur;
      closeSettingsPicker();
      if (chosen) onPick(chosen);
    });
    document.addEventListener('keydown', settingsPickerKey);

    function load(path) {
      listEl.innerHTML = `<div class="eu-pick-empty">${T('Loading...', '正在加载...')}</div>`;
      window.EU_API.listDir(path).then(r => {
        cur = r.path || path || '';
        curEl.textContent = cur || '/';
        const up = back.querySelector('[data-pk-up]');
        up.disabled = !r.parent;
        up.onclick = () => r.parent && load(r.parent);
        scEl.innerHTML = '';
        (r.shortcuts || []).forEach(shortcut => {
          const b = document.createElement('button');
          b.textContent = shortcut.name;
          b.onclick = () => load(shortcut.path);
          scEl.appendChild(b);
        });
        if (!r.entries || !r.entries.length) {
          listEl.innerHTML = `<div class="eu-pick-empty">${r.ok === false ? T('Cannot read this folder.', '无法读取此文件夹。') : T('No sub-folders here.', '这里没有子文件夹。')}</div>`;
          return;
        }
        listEl.innerHTML = '';
        r.entries.forEach(entry => {
          const b = document.createElement('button');
          b.className = 'eu-pick-row';
          b.innerHTML = `<span style="color:var(--ink-3);flex:none;">${icon('folder', 15)}</span><span class="nm">${h(entry.name)}</span>${entry.hint ? `<span class="hint">${h(entry.hint)}</span>` : ''}`;
          b.onclick = () => load(entry.path);
          listEl.appendChild(b);
        });
      }).catch(err => {
        listEl.innerHTML = `<div class="eu-pick-empty">${T('Failed to list folder:', '列出文件夹失败：')} ${h(err && err.message || err)}</div>`;
      });
    }
    load(cur);
  }

  function downloadSettingsDiagnostics() {
    const payload = {
      generated_at: new Date().toISOString(),
      scope: 'settings_diagnostics_no_secrets',
      settings: Object.assign({}, S0()),
      capabilities: C0(),
      registry: window.EU_WORKSPACE_REGISTRY || null,
    };
    if (payload.settings && payload.settings.about) {
      payload.settings.about = Object.assign({}, payload.settings.about);
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'easyicu_settings_diagnostics.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // A hosted instance (screens-guided-pi-hosted.js) is one clinician's
  // EasyICU on the hospital server: the administrator owns the model, data
  // paths, capabilities and privacy switches, so the page keeps only the
  // clinician's own display settings and the way to reach the administrator.
  function hostedInstance() {
    const owner = window.EasyICU?.guidedPi?.optional('hosted');
    return Boolean(owner && owner.active());
  }
  function hostedContact() {
    const owner = window.EasyICU?.guidedPi?.optional('hosted');
    return owner ? owner.supportContact() : '';
  }

  function languageSection() {
    return `
      <div class="sec-stack" id="set-language"><div class="lbl">${T('Language', '语言')}</div><h2>${T('Language & display', '语言与显示')}</h2></div>
      <div class="card pad">
        ${row(T('Interface language', '界面语言'), T('EasyICU is fully bilingual; labels fit both scripts.', 'EasyICU 支持完整双语界面；标签会适配中英文显示。'),
          segBound('language', [['en', 'English'], ['zh', '中文', 'cn']], S0().language || 'en'))}
        ${row(T('Density', '界面密度'), T('Comfortable adds breathing room; compact maximises rows on screen.', '舒适模式增加留白；紧凑模式在屏幕中显示更多行。'),
          segBound('density', [['comfortable', T('Comfortable', '舒适')], ['compact', T('Compact', '紧凑')]], S0().density || 'comfortable'))}
        ${row(T('Reduce motion', '减少动画'), T('Disable shimmer and progress animations.', '关闭闪烁和进度动画。'), sw(!!S0().reduce_motion, '', 'reduce_motion'))}
      </div>

`;
  }

  // The gateway owns the sign-in; js/hosted-session.js owns the sign-out form.
  function accountSection() {
    const session = window.EU_HOSTED_SESSION;
    return `
      <div class="sec-stack" id="set-account"><div class="lbl">${T('Account', '账号')}</div><h2>${T('Sign-in', '登录')}</h2></div>
      <div class="card pad">
        ${row(T('Signed in', '已登录'), T('On a shared computer, sign out when you finish.', '在多人共用的电脑上，用完请退出登录。'), session ? session.logoutForm() : '')}
      </div>
`;
  }

  function aboutSection(hosted) {
    return `
      <div class="sec-stack" id="set-about"><div class="lbl">${T('About', '关于')}</div><h2>${T('Environment', '环境')}</h2></div>
      <div class="card pad">
        <div class="setup-row"><span class="k">${T('Version', '版本')}</span><span class="vv mono">EasyICU ${(S0().about || {}).version || '—'}</span></div>
        ${hosted ? `<div class="setup-row" data-settings-support-contact><span class="k">${T('Administrator', '管理员')}</span><span class="vv">${h(hostedContact() || T('Ask the person who gave you this link', '请联系给你这个链接的人'))}</span></div>` : `
        <div class="setup-row"><span class="k">Python</span><span class="vv mono">${(S0().about || {}).python || '—'}</span></div>
        <div class="setup-row"><span class="k">${T('Supported databases', '支持的数据库')}</span><span class="vv">${h(window.EU_DATABASE_NAMES.keys().map(window.EU_DATABASE_NAMES.name).join(' · '))}</span></div>
        <div class="setup-row"><span class="k">${T('Settings file', '设置文件')}</span><span class="vv mono">${h((S0().about || {}).config_path || '—')}</span></div>`}
        <div class="set-about-actions row gap-8 mt-16">
          ${hosted ? '' : `<button class="btn sm" data-settings-doc="release">${icon('file', 13)} ${T('Release notes', '版本说明')}</button>
          <button class="btn sm" data-settings-doc="docs">${icon('help', 13)} ${T('Documentation', '文档')}</button>`}
          <button class="btn sm" data-settings-diagnostics>${icon('download', 13)} ${T('Export diagnostics', '导出诊断')}</button>
        </div>
      </div>
`;
  }

  function hostedRail() {
    return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${T('Settings', '设置')}</span></div>
        <div class="set-nav col gap-6" style="font-size:12.5px;">
          ${[
            [T('Language', '语言'), 'globe', 'set-language'],
            [T('Account', '账号'), 'lock', 'set-account'],
            [T('About', '关于'), 'help', 'set-about'],
          ].map(([label, ic, target]) =>
            `<button type="button" class="nav-item set-nav-btn" data-settings-jump="${target}" style="height:30px;"><span class="ico">${icon(ic, 15)}</span>${label}</button>`).join('')}
        </div>
      </div>`;
  }

  function hostedRender() {
    return `
      <div class="settings-page" data-settings-hosted>
      <div class="page-head" style="margin-bottom:18px;">
        <h1 style="margin-top:6px;">${T('Settings', '设置')}</h1>
        <p class="lead">${T('The administrator sets up the model and the data. Here you adjust your own display.', '模型和数据由管理员统一配置。这里只调整你自己的界面。')}</p>
      </div>
      ${settingsNotice ? `<div class="note ok mt-12" data-settings-notice><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${T('Settings updated', '设置已更新')}</div><div class="d">${h(settingsNotice)}</div></div></div>` : ''}
      ${languageSection()}
      ${accountSection()}
      ${aboutSection(true)}
      </div>`;
  }

  S.settings = {
    section: 'settings', nav: 'settings',
    get crumbs() { return [T('Home', '首页'), T('Settings', '设置')]; },
    get actionHtml() { return hostedInstance() ? '' : `<button class="btn" data-settings-reset>${icon('refresh', 13)} ${T('Reset to defaults', '恢复默认设置')}</button>`; },
    rail() {
      if (hostedInstance()) return hostedRail();
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${T('Settings', '设置')}</span></div>
        <div class="set-nav col gap-6" style="font-size:12.5px;">
          ${[
            [dual('Capabilities', '能力'), 'layers', 'set-capabilities'],
            [T('Workspace', '工作区'), 'folder', 'set-workspace'],
            [T('Storage locations', '存储位置'), 'folder', 'set-storage'],
            [T('Privacy', '隐私'), 'shield', 'set-privacy'],
            [T('Research Agent', '研究代理'), 'agent', 'set-research-agent'],
            [T('Personal', '个人'), 'edit', 'set-personal'],
            [T('Language', '语言'), 'globe', 'set-language'],
            [T('About', '关于'), 'help', 'set-about'],
          ].map(([label, ic, target]) =>
            `<button type="button" class="nav-item set-nav-btn" data-settings-jump="${target}" style="height:30px;"><span class="ico">${icon(ic, 15)}</span>${label}</button>`).join('')}
        </div>
        <div class="note ok mt-16" style="padding:10px 12px;">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="t" style="font-size:12px;">${T('Local-first', '本地优先')}</div><div class="d" style="font-size:11px;">${T('All settings stay on this machine.', '所有设置都保存在本机。')}</div></div>
        </div>
      </div>`;
    },
    render() {
      if (hostedInstance()) return hostedRender();
      try {
        const requestedTab = window.sessionStorage.getItem('easyicu.settings.openCapabilityTab');
        if (requestedTab && capabilityTabs().some(([id]) => id === requestedTab)) {
          settingsCapabilityTab = requestedTab;
        }
        if (requestedTab) window.sessionStorage.removeItem('easyicu.settings.openCapabilityTab');
      } catch (_) {}
      return `
      <div class="settings-page">
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">${T('Workspace · Settings', '工作区 · 设置')}</div>
        <h1 style="margin-top:6px;">${T('Settings', '设置')}</h1>
        <p class="lead">${T('Configure how EasyICU reads data, runs the agent, and presents the workspace. Everything is local and reversible.', '配置 EasyICU 如何读取数据、运行研究代理并呈现工作区。所有设置都保存在本地，且可以撤销。')}</p>
      </div>
      ${settingsNotice ? `<div class="note ok mt-12" data-settings-notice><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${T('Settings updated', '设置已更新')}</div><div class="d">${h(settingsNotice)}</div></div></div>` : ''}

      ${capabilityManager()}
      ${storageSection()}
      ${rememberedKeysSection()}

      <div class="sec-stack" id="set-workspace"><div class="lbl">${T('Workspace', '工作区')}</div><h2>${T('Local paths', '本地路径')}</h2></div>
      <div class="card pad">
        ${row(T('Default export folder', '默认导出文件夹'), T('Default destination for Data Extraction tables and feature definitions. Research project folders are chosen in their own workflows.', '用于「数据提取」生成的表格和特征说明；研究项目文件夹在各自工作流中选择。'),
          pathCtl('export_dir', T('Not set - EasyICU creates a local run folder when extracting', '未设置 - 抽取时 EasyICU 会创建本地运行文件夹'), 'download'))}
      </div>

      <div class="sec-stack" id="set-privacy"><div class="lbl">${T('Privacy', '隐私')}</div><h2>${T('Local-first guarantees', '本地优先保障')}</h2></div>
      <div class="card pad">
        ${localAccessRow()}
        ${row(T('Allow outbound model calls', '允许外部模型调用'), T('Off by default. When enabled, your messages, analysis plans and code, variable names, error details, and summary statistics with very small counts hidden may reach the configured model. The data tools block patient-level rows; do not paste patient records into messages.', '默认关闭。开启后，你的消息、分析计划及代码、变量名称、错误详情，以及隐藏了极小计数的汇总统计可能发送给已配置的模型。数据工具会阻止患者逐行记录进入模型请求；请勿在消息中粘贴患者记录。'), sw(!!S0().ai_enabled, '', 'ai_enabled'))}
      </div>

      <div class="sec-stack" id="set-research-agent"><div class="lbl">${T('Research Agent', '研究代理')}</div><h2>${T('Run behavior', '运行行为')}</h2></div>
      <div class="card pad">
        ${row(T('Where run behavior is set', '运行行为在哪里设置'), T('Provider, model, output budget, and confirmations are chosen per run inside Guided Copilot and recorded in that run’s ledger — not as hidden global defaults that could make an unexpected external call.', '模型提供方、模型、输出预算与确认项都在「研究引导」中按次运行选择，并记录在该次运行的账本中 —— 不设隐藏的全局默认值，以免造成意外的外部调用。'),
          `<button class="btn sm" data-settings-open="guided">${icon('spark', 12)} ${T('Open Guided Copilot', '打开研究引导')}</button>`)}
      </div>

      ${personalSection()}
      ${languageSection()}

      ${aboutSection(false)}
      </div>`;
    },
    afterRender(root) {
      const rerender = () => { if (typeof window.__euRender === 'function') window.__euRender(); };
      const setNotice = msg => { settingsNotice = msg || ''; rerender(); };
      if (!hostedInstance() && (S0().about || {}).deployment === 'desktop' && !settingsKeysLoading && Date.now() - settingsKeysReadAt > 3000 && window.EU_API?.loadRememberedProviderKeys) {
        settingsKeysLoading = true;
        window.EU_API.loadRememberedProviderKeys().then(payload => { settingsKeys = payload.credentials || []; settingsKeysError = ''; })
          .catch(error => { settingsKeysError = error.message || T('Could not read remembered keys.', '无法读取已记住的密钥。'); })
          .finally(() => { settingsKeysLoading = false; settingsKeysReadAt = Date.now(); rerender(); });
      }
      root.querySelectorAll('[data-forget-key]').forEach(button => button.addEventListener('click', async () => {
        button.disabled = true;
        try {
          const payload = await window.EU_API.deleteRememberedProviderKey(button.getAttribute('data-forget-key'));
          settingsKeys = payload.credentials || [];
          settingsKeysError = '';
          setNotice(T('Remembered key deleted.', '已删除本机记住的密钥。'));
        } catch (error) { settingsKeysError = error.message; rerender(); }
        finally { button.disabled = false; }
      }));
      root.querySelectorAll('[data-storage-change]').forEach(button => button.addEventListener('click', () => {
        settingsEditingPath = button.getAttribute('data-storage-change'); settingsStorageError = ''; settingsPathDrafts[settingsEditingPath] = S0()[settingsEditingPath] || ''; rerender();
      }));
      root.querySelectorAll('[data-storage-cancel]').forEach(button => button.addEventListener('click', () => { settingsEditingPath = ''; settingsStorageError = ''; rerender(); }));
      root.querySelectorAll('[data-storage-form]').forEach(form => form.addEventListener('submit', async event => {
        event.preventDefault();
        const key = form.getAttribute('data-storage-form');
        const patch = { [key]: new FormData(form).get('path') };
        settingsPathDrafts[key] = patch[key]; settingsStorageError = '';
        if (key === 'data_dir') patch.data_dir_confirmed = true;
        const button = form.querySelector('[type="submit"]'); button.disabled = true;
        try {
          await window.EU_API.saveSettings(patch);
          settingsEditingPath = '';
          delete settingsPathDrafts[key];
          if (key === 'data_dir' && window.EU_OFFICIAL_DEMO_SOURCES) await window.EU_OFFICIAL_DEMO_SOURCES.ensureLoaded(null, true);
          setNotice(T('Storage location saved.', '存储位置已保存。'));
        } catch (error) { settingsStorageError = error.message; rerender(); }
        finally { button.disabled = false; }
      }));
      root.querySelectorAll('[data-storage-return]').forEach(button => button.addEventListener('click', () => {
        let route = '#data';
        try { route = window.sessionStorage.getItem('easyicu.settings.storageReturn') || route; window.sessionStorage.removeItem('easyicu.settings.storageReturn'); } catch (_) {}
        window.location.hash = /^#[a-z][a-z0-9-]*$/.test(route) ? route : '#data';
      }));
      try {
        if (window.sessionStorage.getItem('easyicu.settings.openStorage')) {
          root.querySelector('#set-storage')?.scrollIntoView({ block: 'start' });
          window.sessionStorage.removeItem('easyicu.settings.openStorage');
        }
      } catch (_) {}
      if (root.querySelector('[data-ext-manager]') && extensionUi().bind) {
        extensionUi().bind(root, { rerender, setNotice });
      }
      if (window.EU_SETTINGS_PERSONALIZATION) window.EU_SETTINGS_PERSONALIZATION.bind(root);
      const persist = (key, value, msg) => {
        if (key && window.EU_API && window.EU_API.saveSetting) {
          window.EU_API.saveSetting(key, value).then(() => {
            if (key === 'connector_zotero_enabled') settingsZoteroTest = null;
            setNotice(msg || savedSettingMessage(key));
          }).catch(err =>
          {
            console.error('[EasyICU] saveSetting failed', key, err);
            // The backend rejects an unknown/retired/invalid key with a 400 and
            // a human sentence in detail.reason; api.js puts it on err.message.
            // Reporting a generic failure here would hide which key was refused
            // and why — the whole point of failing loudly instead of no-op 200.
            const reason = (err && err.message) ? String(err.message) : '';
            // setNotice re-renders, which also snaps an optimistically flipped
            // switch back to the server's unchanged value.
            setNotice(reason
              ? T('Save failed: ', '保存失败：') + reason
              : T('Save failed. Check the browser console for details.', '保存失败，请查看浏览器 console。'));
          });
        }
      };
      root.querySelectorAll('[data-settings-cap-tab]').forEach(btn => {
        btn.addEventListener('click', e => {
          e.preventDefault();
          settingsCapabilityTab = btn.getAttribute('data-settings-cap-tab') || 'overview';
          rerender();
        });
      });
      root.querySelectorAll('[data-settings-open]').forEach(btn => {
        btn.addEventListener('click', e => {
          e.preventDefault();
          const target = btn.getAttribute('data-settings-open');
          if (target === 'guided') location.hash = '#guided';
        });
      });
      root.querySelectorAll('[data-settings-zotero-test]').forEach(btn => {
        btn.addEventListener('click', e => {
          e.preventDefault();
          if (settingsZoteroTesting || !(window.EU_API && window.EU_API.testZoteroConnection)) return;
          settingsZoteroTesting = true;
          settingsZoteroTest = null;
          settingsNotice = '';
          rerender();
          window.EU_API.testZoteroConnection({})
            .then(data => {
              settingsZoteroTest = data && data.status ? data.status : null;
              if (window.EU_API.loadCapabilities) return window.EU_API.loadCapabilities();
              return null;
            })
            .then(() => { settingsNotice = T('Zotero connection check finished.', 'Zotero 连接检查完成。'); })
            .catch(error => {
              console.error('[EasyICU] Zotero test failed', error);
              settingsNotice = T('Zotero connection check failed. See console for details.', 'Zotero 连接检查失败，请查看 console。');
            })
            .finally(() => {
              settingsZoteroTesting = false;
              rerender();
            });
        });
      });
      root.querySelectorAll('[data-settings-audit-refresh]').forEach(btn => {
        btn.addEventListener('click', e => {
          e.preventDefault();
          if (settingsAuditLoading || !(window.EU_API && window.EU_API.loadCapabilityAuditEvents)) return;
          settingsAuditLoading = true;
          settingsNotice = '';
          rerender();
          window.EU_API.loadCapabilityAuditEvents({ limit: 100 })
            .then(data => {
              settingsAuditEvents = data;
              settingsNotice = T('Audit ledger refreshed.', '审计账本已刷新。');
            })
            .catch(error => {
              console.error('[EasyICU] audit refresh failed', error);
              settingsNotice = T('Audit refresh failed. See console for details.', '审计刷新失败，请查看 console。');
            })
            .finally(() => {
              settingsAuditLoading = false;
              rerender();
            });
        });
      });
      root.querySelectorAll('[data-settings-jump]').forEach(btn => {
        btn.addEventListener('click', e => {
          e.preventDefault();
          const target = btn.getAttribute('data-settings-jump');
          const el = target ? document.getElementById(target) : null;
          if (!el) return;
          root.querySelectorAll('[data-settings-jump]').forEach(x => x.classList.toggle('active', x === btn));
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
      root.querySelectorAll('[data-setting-path]').forEach(btn => {
        btn.addEventListener('click', () => {
          const key = btn.getAttribute('data-setting-path');
          const title = key === 'export_dir' ? T('Choose default export folder', '选择默认导出文件夹') : T('Choose working directory', '选择工作目录');
          openSettingsFolderPicker(S0()[key], title, picked => persist(key, picked, `${title}: ${picked}`));
        });
      });
      root.querySelectorAll('.switch:not(.locked)').forEach(s => {
        const key = s.getAttribute('data-setting');
        const flip = () => {
          s.classList.toggle('on');
          const on = s.classList.contains('on');
          s.setAttribute('aria-checked', on);
          if (key) persist(key, on);          // bound switch -> persist
        };
        s.addEventListener('click', flip);
        s.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); flip(); } });
      });
      // demo-only segmented controls (data-seg) keep their local-only behavior
      root.querySelectorAll('[data-seg]').forEach(seg => {
        seg.addEventListener('click', e => {
          const b = e.target.closest('button'); if (!b) return;
          seg.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
        });
      });
      // bound segmented controls (data-setting) persist the picked value
      root.querySelectorAll('.seg[data-setting]').forEach(seg => {
        const key = seg.getAttribute('data-setting');
        seg.addEventListener('click', e => {
          const b = e.target.closest('button'); if (!b) return;
          seg.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
          const val = b.getAttribute('data-val');
          if (key === 'language' && val && window.setLang) {
            window.setLang(val);
            return;
          }
          persist(key, val);
        });
      });
      root.querySelectorAll('[data-setting-input]').forEach(input => {
        input.addEventListener('change', () => {
          const key = input.getAttribute('data-setting-input');
          const value = input.type === 'number' ? Number(input.value) : input.value;
          persist(key, value, savedSettingMessage(key));
        });
      });
      root.querySelectorAll('[data-settings-reset]').forEach(btn => {
        btn.addEventListener('click', () => {
          if (!window.EU_API || !window.EU_API.resetSettings) return;
          window.EU_API.resetSettings().then(() => {
            settingsZoteroTest = null; settingsAuditEvents = null;
            setNotice(T('Display and capability options reset; installed extensions and personal instructions are kept.', '显示与能力选项已恢复默认；已安装扩展和个人说明保留。'));
          }).catch(err => {
            console.error('[EasyICU] resetSettings failed', err);
            setNotice(T('Reset failed. Check the browser console for details.', '恢复失败，请查看浏览器 console。'));
          });
        });
      });
      root.querySelectorAll('[data-settings-diagnostics]').forEach(btn => {
        btn.addEventListener('click', () => {
          downloadSettingsDiagnostics();
          setNotice(T('Downloaded local settings diagnostics JSON. No secrets are included.', '已下载本地设置诊断 JSON，不包含密钥。'));
        });
      });
      root.querySelectorAll('[data-settings-doc]').forEach(btn => {
        btn.addEventListener('click', () => {
          const target = btn.getAttribute('data-settings-doc');
          if (target === 'docs') {
            location.hash = '#help';
          } else if (window.EU_WHATS_NEW) {
            window.EU_WHATS_NEW.open();
          }
        });
      });
    },
  };
})();
