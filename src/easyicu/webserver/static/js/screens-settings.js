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
    data_mode: 'demo',
    science_skills_enabled: true,
    connector_pubmed_enabled: true,
    connector_zotero_enabled: false,
    mcp_tools_enabled: false,
    prompt_contracts_enabled: true,
    tool_audit_enabled: true,
    remote_compute_enabled: false,
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
      literature_source_ready: dual('literature source ready', '文献来源已就绪'),
      pasted_source_ready: dual('pasted source ready', '粘贴文献已就绪'),
    };
    const reasonMap = {
      connector_zotero_enabled_false: dual('off; pasted source import still works in Idea Mining', '已关闭；Idea Mining 仍可粘贴文献导入'),
      local_zotero_api_ready: dual('desktop link ready', '桌面连接已就绪'),
      local_zotero_api_http_error: dual('desktop link returned an error', '桌面连接返回错误'),
      local_zotero_api_unavailable: dual('desktop app not reachable', '暂时无法连接桌面应用'),
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

  function capStatus(on) {
    return `<span class="settings-cap-status ${on ? 'on' : 'off'}"><span class="dot"></span>${on ? dual('On', '开启') : dual('Off', '关闭')}</span>`;
  }

  function capRow(key, title, desc, fallback) {
    const on = settingOn(key, fallback);
    return row(title, desc, `<div class="settings-cap-control">${capStatus(on)}${sw(on, '', key)}</div>`);
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
      ['prompt_contracts_enabled', true],
      ['tool_audit_enabled', true],
      ['remote_compute_enabled', false],
    ];
  }

  function capabilitySummaryTile(iconName, label, enabled, detail) {
    return `<div class="settings-cap-tile ${enabled ? 'on' : 'off'}">
      <span class="settings-cap-tile-icon">${icon(iconName, 14)}</span>
      <div><b>${label}</b><span>${detail}</span></div>
      ${capStatus(enabled)}
    </div>`;
  }

  function capabilityBody() {
    const pubmedOn = settingOn('connector_pubmed_enabled', true);
    const zoteroOn = settingOn('connector_zotero_enabled', false);
    const mcpOn = settingOn('mcp_tools_enabled', false);
    const zotero = cap('zotero_connector');
    const mcp = cap('mcp_tools');
    const prompts = cap('prompt_contracts');
    const audit = cap('tool_audit');
    const remote = cap('remote_compute');
    const allowedTools = Array.isArray(mcp.allowed_tools) ? mcp.allowed_tools.length : 0;
    const blockedTools = Array.isArray(mcp.blocked_tools) ? mcp.blocked_tools.length : 0;
    const promptRules = Array.isArray(prompts.rules) ? prompts.rules.length : 0;
    const enabledCount = capabilityKeys().filter(([key, fallback]) => settingOn(key, fallback)).length;
    if (settingsCapabilityTab === 'skills') {
      return `<div class="settings-cap-rows">
        ${capRow('science_skills_enabled', dual('Research skills', '研究技能'), dual('Reusable ICU workflows, figure-review protocols, and handoff templates stay available in Agent Science.', '在 Agent Science 中启用可复用 ICU 工作流、图件审阅 protocol 和交接模板。'), true)}
        ${row(dual('Skill scope', '技能范围'), dual('Skills are local workflow templates. They do not send patient rows outside this machine.', '技能是本地工作流模板，不会把患者行发出本机。'), lockedCtl(dual('local template', '本地模板'), dual('Skills reuse local prompts and audit checklists.', '技能复用本地提示词和审计清单。')))}
        ${row(dual('Runtime effect', '运行时效果'), dual('Turning this off removes reusable workflow cards from Agent Science and marks that coverage item unavailable.', '关闭后，Agent Science 会隐藏可复用工作流卡片，并把该覆盖项标记为不可用。'), truthCtl(settingOn('science_skills_enabled', true) ? dual('templates visible', '模板可见') : dual('templates hidden', '模板已隐藏')))}
      </div>`;
    }
    if (settingsCapabilityTab === 'connectors') {
      return `<div class="settings-cap-rows">
        ${capRow('connector_pubmed_enabled', dual('PubMed connector', 'PubMed 连接器'), dual('Controls whether Idea Mining may use PubMed metadata after a source-level opt-in.', '控制 Idea Mining 在来源级 opt-in 后是否可以查询 PubMed 元数据。'), true)}
        ${capRow('connector_zotero_enabled', dual('Zotero desktop link', 'Zotero 桌面连接'), dual('Optional shortcut for searching Zotero Desktop. Idea Mining can still import pasted DOI, BibTeX, RIS, or title/abstract metadata without this switch.', '用于检索 Zotero Desktop 的可选快捷方式。即使不开启，Idea Mining 也能直接导入粘贴的 DOI、BibTeX、RIS 或标题摘要元数据。'), false)}
        ${row(dual('Source opt-in', '来源 opt-in'), dual('Connectors are only a global availability switch. Each URL, paper, or topic still requires an explicit one-time network opt-in.', '连接器只是全局可用开关；每个 URL、文章或主题仍需单次网络 opt-in。'), truthCtl(dual('required per source', '按来源要求')))}
        ${row(dual('Zotero auto-connect', 'Zotero 自动连接'), dual('When enabled, EasyICU checks whether Zotero Desktop is reachable before library search. Pasted source import does not need this.', '启用后，EasyICU 会检查是否能连接 Zotero Desktop 再检索文献库。粘贴文献导入不需要这一步。'), lockedCtl(statusText(zotero) || dual('disabled', '未启用'), dual('Zotero Desktop connection check', 'Zotero Desktop 连接检查')))}
        ${zoteroTestBlock(zotero)}
      </div>`;
    }
    if (settingsCapabilityTab === 'mcp') {
      return `<div class="settings-cap-rows">
        ${capRow('mcp_tools_enabled', dual('MCP tools layer', 'MCP 工具层'), dual('Enables the standard tool boundary for future external systems. Current patient-data workflows remain local.', '为后续外部系统启用标准工具边界；当前患者数据工作流仍保持本地。'), false)}
        ${row(dual('Tool allowlist', '工具白名单'), dual('External tools must be explicitly scoped before they can be used from research workflows.', '外部工具必须先明确作用域，之后才能在研究工作流中使用。'), lockedCtl(dual('required', '必须要求'), dual('Tool scope is an audit contract, not a cosmetic preference.', '工具作用域是审计契约，不是视觉选项。')))}
        ${row(dual('Backend policy', '后端策略'), dual('The API now returns allow/block decisions for the registered tool boundary.', 'API 现在会返回注册工具边界的允许/阻止决策。'), truthCtl(`${allowedTools} ${dual('allowed', '允许')} · ${blockedTools} ${dual('blocked', '阻止')}`))}
      </div>`;
    }
    if (settingsCapabilityTab === 'prompts') {
      return `<div class="settings-cap-rows">
        ${capRow('prompt_contracts_enabled', dual('Prompt contracts', '提示词契约'), dual('Keeps global prompts case-neutral while storing project-specific rules in protocols and rubrics.', '保持全局提示词 case-neutral，并把项目特定规则写入 protocol 和 rubric。'), true)}
        ${row(dual('Case-specific rules', '个案规则'), dual('Study variables, figures, and benchmark cases belong in project protocols, not global prompts.', '研究变量、图件和 benchmark case 应写进项目 protocol，而不是全局提示词。'), truthCtl(dual('protocol-owned', '由 protocol 管理')))}
        ${row(dual('Backend contract rules', '后端契约规则'), dual('The workbench payload exposes the active prompt-contract rule set for review.', '工作台 payload 会暴露当前启用的提示词契约规则，便于审阅。'), truthCtl(`${promptRules} ${dual('active rules', '条有效规则')}`))}
      </div>`;
    }
    if (settingsCapabilityTab === 'audit') {
      return `<div class="settings-cap-rows">
        ${capRow('tool_audit_enabled', dual('Tool audit ledger', '工具审计账本'), dual('Records claims, tool use, citations, calculations, hashes, and reviewer-check state before draft release.', '草稿放行前记录论断、工具使用、引用、计算、哈希和审阅检查状态。'), true)}
        ${capRow('remote_compute_enabled', dual('Remote compute control', '远程计算控制'), dual('Keeps remote or HPC execution disabled until credentials, data boundary, and artifact return rules are configured.', '在凭证、数据边界和产物回传规则配置前，保持远程或 HPC 执行关闭。'), false)}
        ${row(dual('Audit events', '审计事件'), dual('Tool and connector decisions are written only when the audit ledger switch is on.', '只有工具审计账本开启时，工具和连接器决策才会写入。'), truthCtl(`${Number(audit.event_count || 0)} ${dual('recent events', '条近期事件')}`))}
        ${auditEventsBlock(audit)}
        ${row(dual('Compute adapter', '计算适配器'), dual('Non-local compute requests are rejected unless the remote compute switch and adapter are both ready.', '只有远程计算开关和适配器都就绪时，才允许非本地计算请求。'), lockedCtl(computeStateText(remote), dual('Remote compute backend status', '远程计算后端状态')))}
      </div>`;
    }
    return `<div class="settings-cap-overview">
      <div class="settings-cap-meter">
        <b>${enabledCount}/${capabilityKeys().length}</b>
        <span>${dual('enabled capabilities', '已启用能力')}</span>
      </div>
      <div class="settings-cap-grid">
        ${capabilitySummaryTile('layers', dual('Skills', '技能'), settingOn('science_skills_enabled', true), dual('local workflow templates', '本地工作流模板'))}
        ${capabilitySummaryTile('db', dual('Connectors', '连接器'), pubmedOn || zoteroOn, pubmedOn ? dual('PubMed ready; source opt-in still required', 'PubMed 可用；仍需来源 opt-in') : dual('off', '关闭'))}
        ${capabilitySummaryTile('globe', dual('MCP tools', 'MCP 工具'), mcpOn, `${allowedTools}/${allowedTools + blockedTools || 0} ${dual('tools allowed', '工具允许')}`)}
        ${capabilitySummaryTile('file', dual('Prompt contracts', '提示词契约'), settingOn('prompt_contracts_enabled', true), `${promptRules} ${dual('backend rules', '条后端规则')}`)}
        ${capabilitySummaryTile('shield', dual('Audit', '审计'), settingOn('tool_audit_enabled', true), `${Number(audit.event_count || 0)} ${dual('events', '事件')}`)}
        ${capabilitySummaryTile('gear', dual('Compute', '计算'), settingOn('remote_compute_enabled', false), computeStateText(remote))}
      </div>
      <div class="settings-cap-actions">
        <!-- Destination names must match the ones app.js puts in the rail
             ('Agent Projects' / '研究项目', 'Idea Mining' / '想法挖掘'). A button
             that opens #agent while calling it something the sidebar never
             says reads as a fourth feature, not a link to an existing one. -->
        <button class="btn sm" data-settings-open="agent">${icon('agent', 12)} ${dual('Open Agent Projects', '打开研究项目')}</button>
        <button class="btn sm" data-settings-open="ideas">${icon('spark', 12)} ${dual('Open Idea Mining', '打开想法挖掘')}</button>
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
        ? T('The server accepts loopback clients only and rejects proxy-forwarded requests, so patient data has no route off this machine.', '服务器只接受回环客户端并拒绝经代理转发的请求，患者数据没有离开本机的通道。')
        : T('Host access has been widened by an environment variable, so a loopback peer is no longer proof of a local user. Patient data can now reach whatever is in front of this server.', '主机访问已被环境变量放宽，回环 peer 不再等于本地用户。患者数据现在可以到达该服务器前面的任何东西。');
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

  S.settings = {
    section: 'settings', nav: 'settings',
    get crumbs() { return [T('Home', '首页'), T('Settings', '设置')]; },
    get actionHtml() { return `<button class="btn" data-settings-reset>${icon('refresh', 13)} ${T('Reset to defaults', '恢复默认设置')}</button>`; },
    rail() {
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${T('Settings', '设置')}</span></div>
        <div class="set-nav col gap-6" style="font-size:12.5px;">
          ${[
            [dual('Capabilities', '能力'), 'layers', 'set-capabilities'],
            [T('Workspace', '工作区'), 'folder', 'set-workspace'],
            [T('Data mode', '数据模式'), 'flask', 'set-data-mode'],
            [T('Privacy', '隐私'), 'shield', 'set-privacy'],
            [T('Research Agent', '研究代理'), 'agent', 'set-research-agent'],
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
      return `
      <div class="settings-page">
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">${T('Workspace · Settings', '工作区 · 设置')}</div>
        <h1 style="margin-top:6px;">${T('Settings', '设置')}</h1>
        <p class="lead">${T('Configure how EasyICU reads data, runs the agent, and presents the workspace. Everything is local and reversible.', '配置 EasyICU 如何读取数据、运行研究代理并呈现工作区。所有设置都保存在本地，且可以撤销。')}</p>
      </div>
      ${settingsNotice ? `<div class="note ok mt-12" data-settings-notice><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${T('Settings updated', '设置已更新')}</div><div class="d">${h(settingsNotice)}</div></div></div>` : ''}

      ${capabilityManager()}

      <div class="sec-stack" id="set-workspace"><div class="lbl">${T('Workspace', '工作区')}</div><h2>${T('Local paths', '本地路径')}</h2></div>
      <div class="card pad">
        ${row(T('Default export folder', '默认导出文件夹'), T('Optional destination for code, tables, figures, and the evidence ledger bundle. Project and guided-run folders are chosen inside their own workflows.', '用于代码、表格、图件和证据账本包的可选导出位置。项目和引导运行文件夹在各自工作流中选择。'),
          pathCtl('export_dir', T('Not set - EasyICU creates a local run folder when extracting', '未设置 - 抽取时 EasyICU 会创建本地运行文件夹'), 'download'))}
      </div>

      <div class="sec-stack" id="set-data-mode"><div class="lbl">${T('Data mode', '数据模式')}</div><h2>${T('Defaults for new sessions', '新会话默认值')}</h2></div>
      <div class="card pad">
        ${row(T('Start mode', '启动模式'), T('Which mode a new workspace opens in. Demo screens use bounded seeded fixtures; Real Data reads your registered local export. You can always switch later.', '新工作区默认打开的模式。演示页面使用有边界的种子夹具，真实数据读取已注册的本地导出；之后随时可以切换。'),
          segBound('data_mode', [['demo', T('Demo', '演示')], ['real', T('Real Data', '真实数据')]], S0().data_mode || 'demo'))}
      </div>

      <div class="sec-stack" id="set-privacy"><div class="lbl">${T('Privacy', '隐私')}</div><h2>${T('Local-first guarantees', '本地优先保障')}</h2></div>
      <div class="card pad">
        ${localAccessRow()}
        ${row(T('Allow outbound model calls', '允许外部模型调用'), T('Off by default. When on, only the Research Agent prompt and plan - never patient rows - may reach a configured model endpoint.', '默认关闭。开启后，只有研究代理的提示词和计划可以发送到已配置的模型端点，患者行数据永不发送。'), sw(!!S0().ai_enabled, '', 'ai_enabled'))}
      </div>

      <div class="sec-stack" id="set-research-agent"><div class="lbl">${T('Research Agent', '研究代理')}</div><h2>${T('Run behavior', '运行行为')}</h2></div>
      <div class="card pad">
        ${row(T('Where run behavior is set', '运行行为在哪里设置'), T('Provider, output budget, repair policy, and evidence binding are decided per run inside Agent Projects and recorded in that run’s ledger — not as hidden global defaults that could make an unexpected external call.', '模型提供方、输出预算、修复策略与证据绑定都在研究项目里按次运行决定，并记录在该次运行的账本中 —— 不设隐藏的全局默认值，以免造成意外的外部调用。'),
          `<button class="btn sm" data-settings-open="agent">${icon('agent', 12)} ${T('Open Agent Projects', '打开研究项目')}</button>`)}
      </div>

      <div class="sec-stack" id="set-language"><div class="lbl">${T('Language', '语言')}</div><h2>${T('Language & display', '语言与显示')}</h2></div>
      <div class="card pad">
        ${row(T('Interface language', '界面语言'), T('EasyICU is fully bilingual; labels fit both scripts.', 'EasyICU 支持完整双语界面；标签会适配中英文显示。'),
          segBound('language', [['en', 'English'], ['zh', '中文', 'cn']], S0().language || 'en'))}
        ${row(T('Density', '界面密度'), T('Comfortable adds breathing room; compact maximises rows on screen.', '舒适模式增加留白；紧凑模式在屏幕中显示更多行。'),
          segBound('density', [['comfortable', T('Comfortable', '舒适')], ['compact', T('Compact', '紧凑')]], S0().density || 'comfortable'))}
        ${row(T('Reduce motion', '减少动画'), T('Disable shimmer and progress animations.', '关闭闪烁和进度动画。'), sw(!!S0().reduce_motion, '', 'reduce_motion'))}
      </div>

      <div class="sec-stack" id="set-about"><div class="lbl">${T('About', '关于')}</div><h2>${T('Environment', '环境')}</h2></div>
      <div class="card pad">
        <div class="setup-row"><span class="k">${T('Version', '版本')}</span><span class="vv mono">EasyICU ${(S0().about || {}).version || '—'}</span></div>
        <div class="setup-row"><span class="k">Python</span><span class="vv mono">${(S0().about || {}).python || '—'}</span></div>
        <div class="setup-row"><span class="k">${T('Databases detected', '已检测数据库')}</span><span class="vv mono">MIMIC-IV · eICU · AUMC · HiRID · MIMIC-III · SICdb</span></div>
        <div class="setup-row"><span class="k">${T('Settings file', '设置文件')}</span><span class="vv mono">${h((S0().about || {}).config_path || '—')}</span></div>
        <div class="set-about-actions row gap-8 mt-16">
          <button class="btn sm" data-settings-doc="release">${icon('file', 13)} ${T('Release notes', '版本说明')}</button>
          <button class="btn sm" data-settings-doc="docs">${icon('help', 13)} ${T('Documentation', '文档')}</button>
          <button class="btn sm" data-settings-diagnostics>${icon('download', 13)} ${T('Export diagnostics', '导出诊断')}</button>
        </div>
      </div>
      </div>`;
    },
    afterRender(root) {
      const rerender = () => { if (typeof window.__euRender === 'function') window.__euRender(); };
      const setNotice = msg => { settingsNotice = msg || ''; rerender(); };
      const persist = (key, value, msg) => {
        if (key && window.EU_API && window.EU_API.saveSetting) {
          window.EU_API.saveSetting(key, value).then(() => {
            setNotice(msg || `${key.replace(/_/g, ' ')} saved`);
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
          if (target === 'agent') location.hash = '#agent';
          if (target === 'ideas') location.hash = '#ideas';
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
          window.EU_API.loadCapabilityAuditEvents({ limit: 20 })
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
          if (key === 'data_mode' && val && window.setDataMode) {
            window.setDataMode(val);
            return;
          }
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
          persist(key, value, `${key.replace(/_/g, ' ')} saved`);
        });
      });
      root.querySelectorAll('[data-settings-reset]').forEach(btn => {
        btn.addEventListener('click', () => {
          if (!window.EU_API || !window.EU_API.resetSettings) return;
          window.EU_API.resetSettings().then(() => setNotice(T('Settings reset to backend defaults.', '设置已恢复为后端默认值。'))).catch(err => {
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
          } else {
            setNotice(T('Release notes are tracked in the local docs/task logs for this build.', '此构建的版本说明记录在本地文档和任务日志中。'));
          }
        });
      });
    },
  };
})();
