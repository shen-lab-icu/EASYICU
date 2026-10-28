/* Owner: Settings route user Skill/MCP installation and lifecycle UI. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  const state = {
    loading: false,
    error: '',
    busy: '',
    mcpTest: null,
    mcpDraft: { name: '', url: '', tools: '' },
  };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function registry() { return window.EU_EXTENSIONS || null; }
  // D-P2-7: every extension mutation carries the activation revision this
  // panel read, so an overwrite/removal cannot land on state the user never
  // reviewed. The backend rejects a missing value (422) or a stale one (409).
  function confirmRevision() { const data = registry(); return (data && data.activation_sha256) || ''; }
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
    const rows = (registry() || {}).skills || [];
    return `<div class="settings-ext-head"><div><b>${tr('My skills', '我的技能')}</b><span>${rows.filter(row => row.enabled).length}/${rows.length} ${tr('enabled for new sessions', '对新会话启用')}</span></div><a class="btn sm" href="#skills">${tr('Import, edit or export', '导入、编辑或导出')}</a></div>`;
  }

  function renderMcp() {
    const data = registry();
    const rows = data && Array.isArray(data.mcp_servers) ? data.mcp_servers : [];
    const master = !!(window.EU_SETTINGS && window.EU_SETTINGS.mcp_tools_enabled);
    const tested = state.mcpTest && Array.isArray(state.mcpTest.tools) ? state.mcpTest.tools : [];
    return `<section class="settings-ext-manager" data-ext-manager="mcp">
      <div class="settings-ext-head"><div><b>${tr('Installed MCP servers', '已安装 MCP 服务')}</b><span>${tr('Allowlisted read-only tools are callable in new conversations after a per-turn grant. Streamable HTTP; no stored credentials.', '新对话可按次授权调用白名单内的只读工具。支持 Streamable HTTP，不保存凭证。')}</span></div><span class="pill ${master ? 'ok' : 'warn'}">${master ? tr('master on', '总开关已开') : tr('master off', '总开关已关')}</span></div>
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
      api().setExtensionState({ kind, name, enabled, expected_sha256: confirmRevision() })
        .then(() => notice(enabled ? tr('Extension enabled for new sessions and runs.', '扩展已对新会话和运行启用。') : tr('Extension disabled for new sessions and runs.', '扩展已对新会话和运行停用。')))
        .catch(error => { state.error = errorText(error); })
        .finally(() => { state.busy = ''; rerender(); });
    }));
    root.querySelectorAll('[data-ext-remove]').forEach(button => button.addEventListener('click', () => {
      const kind = button.getAttribute('data-ext-remove');
      const name = button.getAttribute('data-ext-name');
      if (!window.confirm(tr(`Remove ${name}? Existing frozen sessions and runs keep their recorded version.`, `移除 ${name}？已有固化会话和运行仍保留原版本。`))) return;
      api().removeExtension({ kind, name, expected_sha256: confirmRevision() }).then(() => notice(tr('Extension removed from future activation.', '扩展已从后续激活中移除。')))
        .catch(error => { state.error = errorText(error); }).finally(rerender);
    }));
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
      api().installExtensionMcp({ name: draft.name, url: draft.url, allowed_tools: parseTools(draft.tools), enabled, expected_sha256: confirmRevision() })
        .then(() => { state.mcpDraft = { name: '', url: '', tools: '' }; state.mcpTest = null; notice(tr('MCP server installed with its explicit tool allowlist.', 'MCP 服务已按显式工具白名单安装。')); })
        .catch(error => { state.error = errorText(error); })
        .finally(() => { state.busy = ''; rerender(); });
    });
  }

  window.EU_SETTINGS_EXTENSIONS = { bind, ensureLoaded, renderMcp, renderSkills };
})();
