/* Owner: Settings route user Skill/MCP installation and lifecycle UI. */
(function () {
  'use strict';

  const state = {
    loading: false,
    error: '',
    busy: '',
    mcpTest: null,
    skillDraft: '',
    mcpDraft: { name: '', url: '', tools: '' },
  };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
  }
  function api() { return window.EU_API || {}; }
  function registry() { return window.EU_EXTENSIONS || null; }
  function shortSha(value) { return String(value || '').slice(0, 12); }
  function errorText(error) {
    return String((error && error.message) || error || tr('Extension request failed.', '扩展请求失败。'));
  }
  function toggleButton(kind, row) {
    const on = !!row.enabled;
    return `<button type="button" class="btn sm ${on ? '' : 'ghost'}" data-ext-toggle="${kind}" data-ext-name="${esc(row.name)}" data-ext-enabled="${on ? 'true' : 'false'}">${on ? tr('Disable', '停用') : tr('Enable', '启用')}</button>`;
  }
  function emptyState(text) {
    return `<div class="settings-ext-empty">${esc(text)}</div>`;
  }
  function skillCards(rows) {
    if (!rows.length) return emptyState(tr('No user Skill is installed yet.', '尚未安装用户 Skill。'));
    return `<div class="settings-ext-list">${rows.map(row => `
      <article class="settings-ext-card ${row.enabled ? 'on' : ''}">
        <div class="settings-ext-card-main">
          <div class="settings-ext-title"><b>${esc(row.name)}</b><span class="settings-ext-kind">SKILL</span></div>
          <p>${esc(row.description)}</p>
          <div class="settings-ext-meta"><span>${(row.stages || []).map(stage => esc(stage)).join(' + ')}</span><span class="mono">sha256:${shortSha(row.digest)}</span></div>
        </div>
        <div class="settings-ext-actions">${toggleButton('skill', row)}<button type="button" class="btn sm danger" data-ext-remove="skill" data-ext-name="${esc(row.name)}">${tr('Remove', '移除')}</button></div>
      </article>`).join('')}</div>`;
  }
  function mcpCards(rows) {
    if (!rows.length) return emptyState(tr('No MCP server is installed yet.', '尚未安装 MCP 服务。'));
    return `<div class="settings-ext-list">${rows.map(row => `
      <article class="settings-ext-card ${row.enabled ? 'on' : ''}">
        <div class="settings-ext-card-main">
          <div class="settings-ext-title"><b>${esc(row.name)}</b><span class="settings-ext-kind">MCP</span></div>
          <p class="mono settings-ext-url">${esc(row.url)}</p>
          <div class="settings-ext-meta"><span>${(row.allowed_tools || []).length} ${tr('allowlisted tools', '个白名单工具')}</span><span>${esc(row.transport || 'streamable-http')}</span><span>${tr('no stored credentials', '不保存凭证')}</span></div>
        </div>
        <div class="settings-ext-actions">${toggleButton('mcp', row)}<button type="button" class="btn sm danger" data-ext-remove="mcp" data-ext-name="${esc(row.name)}">${tr('Remove', '移除')}</button></div>
      </article>`).join('')}</div>`;
  }
  function statusBlock() {
    if (state.loading) return `<div class="settings-ext-status"><span class="spin"></span>${tr('Loading installed extensions…', '正在读取已安装扩展…')}</div>`;
    if (state.error) return `<div class="settings-ext-status error">${esc(state.error)}</div>`;
    return '';
  }

  function renderSkills() {
    const data = registry();
    const rows = data && Array.isArray(data.skills) ? data.skills : [];
    return `<section class="settings-ext-manager" data-ext-manager="skills">
      <div class="settings-ext-head"><div><b>${tr('User-installed Skills', '用户安装的 Skill')}</b><span>${tr('Frozen into each new Pi session and Agent run. Writing Skills are advisory and cannot override evidence rules.', '在每个新 Pi 会话和 Agent 运行中固化版本。写作 Skill 仅作辅助，不能覆盖证据规则。')}</span></div><span class="pill info">${rows.filter(row => row.enabled).length}/${rows.length} ${tr('active', '启用')}</span></div>
      ${statusBlock()}${skillCards(rows)}
      <details class="settings-ext-add"><summary>${tr('Add or update a SKILL.md', '添加或更新 SKILL.md')}</summary>
        <div class="settings-ext-form">
          <label><span>${tr('Upload SKILL.md', '上传 SKILL.md')}</span><input type="file" accept=".md,text/markdown,text/plain" data-ext-skill-file></label>
          <label class="wide"><span>${tr('Or paste the complete file', '或粘贴完整文件')}</span><textarea rows="8" maxlength="12000" data-ext-skill-md placeholder="---&#10;name: concise-writing&#10;description: Keep scientific prose concise.&#10;---&#10;Use short paragraphs…">${esc(state.skillDraft)}</textarea></label>
          <fieldset class="settings-ext-stages"><legend>${tr('Workflow stages', '工作流阶段')}</legend><label><input type="checkbox" value="conversation" data-ext-skill-stage checked> ${tr('Conversation', '对话辅助')}</label><label><input type="checkbox" value="writing" data-ext-skill-stage> ${tr('Writing', '写作辅助')}</label></fieldset>
          <label class="settings-ext-check"><input type="checkbox" data-ext-install-enabled checked> ${tr('Enable after review', '审阅后立即启用')}</label>
          <button type="button" class="btn primary" data-ext-install-skill>${state.busy === 'skill-install' ? tr('Installing…', '安装中…') : tr('Install Skill', '安装 Skill')}</button>
        </div>
      </details>
    </section>`;
  }

  function renderMcp() {
    const data = registry();
    const rows = data && Array.isArray(data.mcp_servers) ? data.mcp_servers : [];
    const master = !!(window.EU_SETTINGS && window.EU_SETTINGS.mcp_tools_enabled);
    const tested = state.mcpTest && Array.isArray(state.mcpTest.tools) ? state.mcpTest.tools : [];
    return `<section class="settings-ext-manager" data-ext-manager="mcp">
      <div class="settings-ext-head"><div><b>${tr('Installed MCP servers', '已安装 MCP 服务')}</b><span>${tr('Streamable HTTP only; every callable tool must be allowlisted. The first release stores no tokens or custom headers.', '仅支持 Streamable HTTP；每个可调用工具必须进入白名单。第一版不保存令牌或自定义请求头。')}</span></div><span class="pill ${master ? 'ok' : 'warn'}">${master ? tr('master on', '总开关已开') : tr('master off', '总开关已关')}</span></div>
      ${statusBlock()}${mcpCards(rows)}
      <details class="settings-ext-add"><summary>${tr('Add or update an MCP server', '添加或更新 MCP 服务')}</summary>
        <div class="settings-ext-form">
          <label><span>${tr('Server name', '服务名称')}</span><input maxlength="64" data-ext-mcp-name value="${esc(state.mcpDraft.name)}" placeholder="literature-tools"></label>
          <label><span>${tr('Streamable HTTP URL', 'Streamable HTTP 地址')}</span><input maxlength="2048" data-ext-mcp-url value="${esc(state.mcpDraft.url)}" placeholder="https://example.org/mcp"></label>
          <label class="wide"><span>${tr('Allowed tool names (comma or newline separated)', '允许的工具名（逗号或换行分隔）')}</span><textarea rows="4" data-ext-mcp-tools placeholder="search&#10;fetch_metadata">${esc(state.mcpDraft.tools)}</textarea></label>
          <label class="settings-ext-check"><input type="checkbox" data-ext-install-enabled> ${tr('Enable after review', '审阅后立即启用')}</label>
          <div class="settings-ext-form-actions"><button type="button" class="btn" data-ext-test-mcp>${state.busy === 'mcp-test' ? tr('Testing…', '测试中…') : tr('Test & list tools', '测试并列出工具')}</button><button type="button" class="btn primary" data-ext-install-mcp>${state.busy === 'mcp-install' ? tr('Installing…', '安装中…') : tr('Install MCP', '安装 MCP')}</button></div>
          ${state.mcpTest ? `<div class="settings-ext-test ${state.mcpTest.ok ? 'ok' : 'error'}"><b>${state.mcpTest.ok ? tr('Handshake succeeded', '握手成功') : tr('Handshake failed', '握手失败')}</b><span>${tested.length ? tested.map(tool => esc(tool.name)).join(', ') : tr('No tools returned.', '未返回工具。')}</span></div>` : ''}
        </div>
      </details>
    </section>`;
  }

  function ensureLoaded(rerender) {
    if (registry() || state.loading || !api().loadExtensions) return;
    state.loading = true; state.error = '';
    api().loadExtensions().catch(error => { state.error = errorText(error); })
      .finally(() => { state.loading = false; rerender(); });
  }
  function captureMcpDraft(root) {
    state.mcpDraft = {
      name: String((root.querySelector('[data-ext-mcp-name]') || {}).value || '').trim(),
      url: String((root.querySelector('[data-ext-mcp-url]') || {}).value || '').trim(),
      tools: String((root.querySelector('[data-ext-mcp-tools]') || {}).value || '').trim(),
    };
    return state.mcpDraft;
  }
  function parseTools(text) {
    return Array.from(new Set(String(text || '').split(/[\n,]+/).map(value => value.trim()).filter(Boolean)));
  }
  function bind(root, options) {
    const rerender = options && options.rerender ? options.rerender : function () {};
    const notice = options && options.setNotice ? options.setNotice : function () {};
    ensureLoaded(rerender);
    root.querySelectorAll('[data-ext-toggle]').forEach(button => button.addEventListener('click', () => {
      const kind = button.getAttribute('data-ext-toggle');
      const name = button.getAttribute('data-ext-name');
      const enabled = button.getAttribute('data-ext-enabled') !== 'true';
      state.busy = `${kind}-toggle`; state.error = '';
      api().setExtensionState({ kind, name, enabled })
        .then(() => notice(enabled ? tr('Extension enabled for new sessions and runs.', '扩展已对新会话和运行启用。') : tr('Extension disabled for new sessions and runs.', '扩展已对新会话和运行停用。')))
        .catch(error => { state.error = errorText(error); })
        .finally(() => { state.busy = ''; rerender(); });
    }));
    root.querySelectorAll('[data-ext-remove]').forEach(button => button.addEventListener('click', () => {
      const kind = button.getAttribute('data-ext-remove');
      const name = button.getAttribute('data-ext-name');
      if (!window.confirm(tr(`Remove ${name}? Existing frozen sessions and runs keep their recorded version.`, `移除 ${name}？已有固化会话和运行仍保留原版本。`))) return;
      api().removeExtension({ kind, name }).then(() => notice(tr('Extension removed from future activation.', '扩展已从后续激活中移除。')))
        .catch(error => { state.error = errorText(error); }).finally(rerender);
    }));
    const file = root.querySelector('[data-ext-skill-file]');
    if (file) file.addEventListener('change', () => {
      const picked = file.files && file.files[0];
      if (!picked) return;
      if (picked.size > 12000) { state.error = tr('SKILL.md exceeds 12 KB.', 'SKILL.md 超过 12 KB。'); rerender(); return; }
      picked.text().then(text => { state.skillDraft = text; rerender(); });
    });
    const skillInstall = root.querySelector('[data-ext-install-skill]');
    if (skillInstall) skillInstall.addEventListener('click', () => {
      const area = root.querySelector('[data-ext-skill-md]');
      state.skillDraft = String((area && area.value) || '');
      const stages = Array.from(root.querySelectorAll('[data-ext-skill-stage]:checked')).map(input => input.value);
      const enabled = !!((root.querySelector('[data-ext-manager="skills"] [data-ext-install-enabled]') || {}).checked);
      state.busy = 'skill-install'; state.error = '';
      api().installExtensionSkill({ skill_md: state.skillDraft, stages, enabled })
        .then(() => { state.skillDraft = ''; notice(tr('Skill installed. It will enter newly created sessions and runs.', 'Skill 已安装，将进入之后新建的会话和运行。')); })
        .catch(error => { state.error = errorText(error); })
        .finally(() => { state.busy = ''; rerender(); });
    });
    const test = root.querySelector('[data-ext-test-mcp]');
    if (test) test.addEventListener('click', () => {
      const draft = captureMcpDraft(root); state.busy = 'mcp-test'; state.error = ''; state.mcpTest = null;
      api().testExtensionMcp({ url: draft.url }).then(result => {
        state.mcpTest = result;
        if (!state.mcpDraft.tools && result && Array.isArray(result.tools)) state.mcpDraft.tools = result.tools.map(tool => tool.name).join('\n');
      }).catch(error => { state.error = errorText(error); }).finally(() => { state.busy = ''; rerender(); });
    });
    const mcpInstall = root.querySelector('[data-ext-install-mcp]');
    if (mcpInstall) mcpInstall.addEventListener('click', () => {
      const draft = captureMcpDraft(root);
      const enabled = !!((root.querySelector('[data-ext-manager="mcp"] [data-ext-install-enabled]') || {}).checked);
      state.busy = 'mcp-install'; state.error = '';
      api().installExtensionMcp({ name: draft.name, url: draft.url, allowed_tools: parseTools(draft.tools), enabled })
        .then(() => { state.mcpDraft = { name: '', url: '', tools: '' }; state.mcpTest = null; notice(tr('MCP server installed with its explicit tool allowlist.', 'MCP 服务已按显式工具白名单安装。')); })
        .catch(error => { state.error = errorText(error); })
        .finally(() => { state.busy = ''; rerender(); });
    });
  }

  window.EU_SETTINGS_EXTENSIONS = { bind, ensureLoaded, renderMcp, renderSkills };
})();
