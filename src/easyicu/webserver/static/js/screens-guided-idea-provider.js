/* Guided Copilot idea-mining provider readiness/config panel.
   Owner file for API-provider setup UI used by screens-guided.js. */
(function () {
  'use strict';

  const { esc, escAttr: attr } = window.EU_HTML;

  function createState(overrides) {
    return Object.assign({
      provider: 'openai',
      loading: false,
      saving: false,
      error: null,
      saveError: null,
      saved: false,
      status: null,
      configOpen: false,
    }, overrides || {});
  }

  function state(ctx) {
    let current = ctx.getState && ctx.getState();
    if (!current) {
      current = createState();
      if (ctx.setState) ctx.setState(current);
    }
    return current;
  }

  function setState(ctx, patch) {
    const next = Object.assign({}, state(ctx), patch || {});
    if (ctx.setState) ctx.setState(next);
    return next;
  }

  function api(ctx) {
    return (ctx.api && ctx.api()) || window.EU_API || {};
  }

  function render(ctx) {
    if (ctx.renderThread) ctx.renderThread();
  }

  function translate(ctx, en, zh) {
    return ctx.t ? ctx.t(en, zh) : en;
  }

  function icon(ctx, name, size) {
    return ctx.icon ? ctx.icon(name, size) : '';
  }

  function setSettings(ctx, settings) {
    if (!settings) return;
    window.EU_SETTINGS = settings;
    if (window.applySettingsState) window.applySettingsState(settings);
  }

  function requestStatus(ctx, force) {
    const current = state(ctx);
    const eu = api(ctx);
    if (!eu.loadAgentProviderStatus) return;
    if (!force && (current.loading || current.status || current.error)) return;
    setState(ctx, { loading: true, error: null });
    eu.loadAgentProviderStatus(current.provider || 'openai').then(data => {
      setState(ctx, {
        loading: false,
        error: null,
        status: (data && data.provider_status) || data || null,
      });
      render(ctx);
    }).catch(err => {
      setState(ctx, {
        loading: false,
        error: err && err.message ? err.message : String(err || 'provider_status_error'),
        status: null,
      });
      render(ctx);
    });
  }

  function enableProvider(ctx) {
    const current = state(ctx);
    const eu = api(ctx);
    if (!eu.saveSetting) {
      setState(ctx, { saveError: 'settings_api_unavailable' });
      render(ctx);
      return;
    }
    setState(ctx, { saving: true, saveError: null, saved: false });
    render(ctx);
    eu.saveSetting('ai_enabled', true).then(settings => {
      setSettings(ctx, settings);
      setState(ctx, {
        provider: current.provider || 'openai',
        saving: false,
        saved: true,
        saveError: null,
      });
      requestStatus(ctx, true);
    }).catch(err => {
      setState(ctx, {
        saving: false,
        saved: false,
        saveError: err && err.message ? err.message : String(err || 'ai_opt_in_failed'),
      });
      render(ctx);
    });
  }

  function saveConfig(ctx, root) {
    state(ctx);
    const eu = api(ctx);
    if (!eu.saveAgentProviderConfig) {
      setState(ctx, { saveError: 'provider_config_api_unavailable' });
      render(ctx);
      return;
    }
    const host = root || document;
    const apiKey = ((host.querySelector('[data-gi-provider-key]') || {}).value || '').trim();
    const baseUrl = ((host.querySelector('[data-gi-provider-base]') || {}).value || '').trim();
    const model = ((host.querySelector('[data-gi-provider-model]') || {}).value || '').trim();
    const enable = !!((host.querySelector('[data-gi-provider-enable]') || {}).checked);
    if (!apiKey) {
      setState(ctx, {
        saveError: translate(ctx, 'Paste an API key first. It will be written only to the local private env file.', '请先粘贴 API key。它只会写入本机私有 env 文件。'),
        saved: false,
      });
      render(ctx);
      return;
    }
    if (!model) {
      setState(ctx, {
        saveError: translate(ctx, 'Enter the model name to use for this provider.', '请填写这个 provider 要使用的模型名。'),
        saved: false,
      });
      render(ctx);
      return;
    }
    const current = state(ctx);
    setState(ctx, {
      saving: true,
      saveError: null,
      saved: false,
    });
    render(ctx);
    eu.saveAgentProviderConfig({
      provider: current.provider || 'openai',
      api_key: apiKey,
      base_url: baseUrl,
      model,
      enable_ai: enable,
      json_format_style: 'responses',
    }).then(data => {
      if (data && data.settings) setSettings(ctx, data.settings);
      setState(ctx, {
        loading: false,
        saving: false,
        saved: true,
        saveError: null,
        status: (data && data.provider_status) || null,
        configOpen: !(data && data.provider_status && data.provider_status.ready),
      });
      render(ctx);
    }).catch(err => {
      setState(ctx, {
        saving: false,
        saved: false,
        saveError: err && err.message ? err.message : String(err || 'provider_config_failed'),
      });
      render(ctx);
    });
  }

  function renderCapabilityPanel(ctx) {
    const current = state(ctx);
    const guidedIdea = ctx.getIdea ? ctx.getIdea() : null;
    const st = current.status || {};
    const envFile = st.env_file || {};
    const missing = Array.isArray(st.missing) ? st.missing : [];
    const ready = !!st.ready;
    const aiOn = !!st.ai_enabled;
    const keyReady = !!st.credential_present;
    const modelReady = !!st.model_present;
    const envStatus = envFile.status || 'not_loaded';
    const blocked = missing.length ? missing.join(', ') : (ready ? '' : translate(ctx, 'provider not ready', 'provider 未就绪'));
    const providers = [
      ['openai', 'OpenAI'],
      ['openrouter', 'OpenRouter'],
      ['deepseek', 'DeepSeek'],
      ['custom', 'Custom/local'],
    ];
    const showConfig = current.configOpen || (!ready && (!keyReady || !modelReady));
    const canEnableSaved = !ready && keyReady && modelReady && !aiOn;
    return `
      <div class="gdx-status ${ready ? 'ok' : 'warn'}">
        <span>${icon(ctx, ready ? 'check' : 'shield', 12)}</span>
        <div>
          <strong>${ready ? translate(ctx, 'API provider ready', 'API provider 已就绪') : translate(ctx, 'API readiness setup', 'API 就绪配置')}</strong>
          <small>${translate(ctx, 'Local PDF/folder mining can run now. Frontier search, prior-art checks, AI synthesis, and full Agent provider calls need explicit opt-in and provider readiness first.', '本地 PDF/文献库挖掘可以直接运行。前沿检索、既有研究检查、AI 综合和 full Agent provider 调用必须先显式 opt-in 并通过 provider readiness。')}</small>
        </div>
      </div>
      <div class="gdx-actions">
        ${providers.map(([p, label]) => `<button type="button" class="btn sm ${current.provider === p ? 'primary' : ''}" data-gi-provider="${p}">${esc(label)}</button>`).join('')}
        <button type="button" class="btn sm" data-gi-provider-refresh>${icon(ctx, 'refresh', 12)} ${translate(ctx, 'Check API status', '检查 API 状态')}</button>
        ${canEnableSaved ? `<button type="button" class="btn sm primary" data-gi-enable-ai>${translate(ctx, 'Enable configured API', '启用已配置 API')}</button>` : ''}
        <button type="button" class="btn sm" data-gi-provider-config-toggle>${icon(ctx, 'gear', 12)} ${showConfig ? translate(ctx, 'Hide setup', '收起配置') : translate(ctx, 'Configure API here', '在这里配置 API')}</button>
      </div>
      ${showConfig ? `
        <div class="gdi-feature-list">
          <div class="gdi-feature-row">
            <div><strong>${translate(ctx, 'Provider config', 'Provider 配置')}</strong><small>${translate(ctx, 'Saved locally to a private 0600 env file. The UI never returns or stores the key value after submit.', '保存到本机私有 0600 env 文件；提交后 UI 不返回也不保存 key 值。')}</small></div>
            <span class="pill ${keyReady && modelReady ? 'ok' : 'warn'}">${keyReady && modelReady ? translate(ctx, 'existing config detected', '已检测到配置') : translate(ctx, 'needs key + model', '需要 key + model')}</span>
          </div>
          <label class="gdi-field wide">
            <span>${translate(ctx, 'API key', 'API key')}</span>
            <input type="password" autocomplete="off" data-gi-provider-key placeholder="${attr(translate(ctx, 'Paste provider key; it will not be echoed back', '粘贴 provider key；不会回显'))}">
          </label>
          <div class="gdi-field-grid">
            <label class="gdi-field">
              <span>${translate(ctx, 'Base URL / endpoint', 'Base URL / endpoint')}</span>
              <input data-gi-provider-base placeholder="http://127.0.0.1:8787/v1">
            </label>
            <label class="gdi-field">
              <span>${translate(ctx, 'Model', '模型')}</span>
              <input data-gi-provider-model placeholder="gpt5.4">
            </label>
          </div>
          <label class="gdi-check">
            <input type="checkbox" data-gi-provider-enable ${aiOn ? 'checked' : ''}>
            <span>${translate(ctx, 'Enable AI/provider opt-in after saving', '保存后启用 AI/provider opt-in')}</span>
          </label>
          <div class="gdx-actions">
            <button type="button" class="btn primary" data-gi-provider-save ${current.saving ? 'disabled' : ''}>${current.saving ? translate(ctx, 'Saving...', '保存中...') : translate(ctx, 'Save provider config locally', '本地保存 provider 配置')}</button>
            <span class="muted">${translate(ctx, 'Use this for local OpenAI-compatible endpoints, OpenAI, OpenRouter, DeepSeek, or Custom/local.', '可用于本地 OpenAI-compatible 端点、OpenAI、OpenRouter、DeepSeek 或 Custom/local。')}</span>
          </div>
          ${current.saved ? `<div class="gdx-status ok"><span>${icon(ctx, 'check', 12)}</span><div><strong>${translate(ctx, 'Provider config saved', 'Provider 配置已保存')}</strong><small>${translate(ctx, 'Secrets were not returned to the browser response.', '响应没有返回密钥。')}</small></div></div>` : ''}
          ${current.saveError ? `<div class="gdx-status bad"><span>${icon(ctx, 'x', 12)}</span><div><strong>${translate(ctx, 'Could not save provider config', 'Provider 配置保存失败')}</strong><small>${esc(current.saveError)}</small></div></div>` : ''}
        </div>` : ''}
      <div class="gdi-feature-list">
        <div class="gdi-feature-row">
          <div><strong>${translate(ctx, 'Local deterministic mining', '本地确定性挖掘')}</strong><small>${translate(ctx, 'PDF/folder bounded excerpt, dictionary feasibility assessment, Agent handoff seed', 'PDF/文件夹有界摘录、字典可行性评估、Agent 交接种子')}</small></div>
          <span class="pill ok">${translate(ctx, 'available now', '当前可用')}</span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${translate(ctx, 'Network prior-art check', '联网既有研究检查')}</strong><small>${translate(ctx, 'Runs only after the per-source network checkbox is selected; no request is made otherwise.', '只有勾选当前来源的网络 opt-in 后才会请求；否则不会联网。')}</small></div>
          <span class="pill ${guidedIdea && guidedIdea.allowNetwork ? 'warn' : 'dashed'}">${guidedIdea && guidedIdea.allowNetwork ? translate(ctx, 'armed for one request', '已允许一次请求') : translate(ctx, 'blocked until opt-in', '等待 opt-in')}</span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${translate(ctx, 'AI synthesis / full Agent provider', 'AI 综合 / full Agent provider')}</strong><small>${ready ? translate(ctx, 'Provider readiness passed. A later Agent run still needs per-run confirmation and evidence checks.', 'provider 已就绪。后续 Agent run 仍需要逐次确认和证据核验。') : translate(ctx, 'Blocked: configure AI opt-in, API key, model, and endpoint before any provider call.', '已阻断：需要配置 AI opt-in、API key、模型和端点后才允许 provider 调用。')}</small></div>
          <span class="pill ${ready ? 'ok' : 'warn'}">${ready ? translate(ctx, 'ready', '就绪') : esc(blocked)}</span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${translate(ctx, 'Provider status', 'Provider 状态')}</strong><small>${esc(current.provider || st.provider || 'openai')} · ${translate(ctx, 'sanitized flags only', '只显示脱敏标记')}</small></div>
          <span>${current.loading ? translate(ctx, 'checking...', '检查中...') : ready ? translate(ctx, 'ready', '就绪') : translate(ctx, 'blocked', '受阻')}</span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${translate(ctx, 'Readiness flags', '就绪标记')}</strong><small>${translate(ctx, 'Only variable names and booleans are shown; secret values and base URL values are never returned.', '只显示变量名和布尔值；密钥值和 base URL 值不会返回。')}</small></div>
          <span class="gdi-tags">
            <code>AI ${aiOn ? 'on' : 'off'}</code>
            <code>key ${keyReady ? 'present' : 'missing'}</code>
            <code>model ${modelReady ? 'present' : 'missing'}</code>
          </span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${translate(ctx, 'Env sources', '环境变量来源')}</strong><small>${translate(ctx, 'Sanitized provider status from Agent provider readiness.', '来自 Agent provider readiness 的脱敏状态。')}</small></div>
          <span class="gdi-tags">
            <code>${esc(st.credential_source || (st.credential_env_candidates || [])[0] || 'credential env')}</code>
            <code>${esc(st.model_source || (st.model_env_candidates || [])[0] || 'model env')}</code>
            <code>${esc(st.base_url_source || (st.base_url_env_candidates || [])[0] || 'base_url env')}</code>
            <code>${esc(envStatus)}</code>
          </span>
        </div>
        ${current.error ? `<div class="gdi-feature-row"><div><strong>${translate(ctx, 'Provider status unavailable', 'provider 状态不可用')}</strong><small>${esc(current.error)}</small></div><span class="pill warn">error</span></div>` : ''}
      </div>
      `;
  }

  function renderSetupPrompt(ctx) {
    const current = state(ctx);
    const st = current.status || {};
    const ready = !!st.ready;
    const aiOn = !!st.ai_enabled;
    const keyReady = !!st.credential_present;
    const modelReady = !!st.model_present;
    const providers = [
      ['openai', 'OpenAI'],
      ['openrouter', 'OpenRouter'],
      ['deepseek', 'DeepSeek'],
      ['custom', 'Custom/local'],
    ];
    const showConfig = current.configOpen !== false || !ready;
    const canEnableSaved = !ready && keyReady && modelReady && !aiOn;
    const continueLabel = ready
      ? translate(ctx, 'Continue to idea sources', '继续选择 idea 来源')
      : translate(ctx, 'Continue local-only for now', '暂时本地模式继续');
    return `
      <div class="gd-idea-api-card gd-idea-card">
        <div class="gdx-head">
          <span class="gdx-ico">${icon(ctx, 'shield', 15)}</span>
          <div>
            <strong>${translate(ctx, 'Step 1 · Configure API access', '第一步 · 配置 API')}</strong>
            <span>${translate(ctx, 'Idea mining can run local-only, but AI synthesis and Agent handoff need an explicit provider setup first.', 'Idea 挖掘可以先本地运行；AI 综合和 Agent 交接需要先显式配置 provider。')}</span>
          </div>
        </div>
        <div class="gdx-status ${ready ? 'ok' : 'warn'}">
          <span>${icon(ctx, ready ? 'check' : 'shield', 12)}</span>
          <div>
            <strong>${ready ? translate(ctx, 'API provider is ready', 'API provider 已就绪') : translate(ctx, 'API is not ready yet', 'API 还未就绪')}</strong>
            <small>${ready ? translate(ctx, 'Provider calls still stay opt-in per action.', '后续 provider 调用仍然逐步 opt-in。') : translate(ctx, 'Paste a key and model here, or continue in local-only mode and configure it later.', '可以在这里填写 key 和模型；也可以先用本地模式继续，稍后再配置。')}</small>
          </div>
        </div>
        <div class="gdx-actions">
          ${providers.map(([p, label]) => `<button type="button" class="btn sm ${current.provider === p ? 'primary' : ''}" data-gi-provider="${p}">${esc(label)}</button>`).join('')}
          <button type="button" class="btn sm" data-gi-provider-refresh>${icon(ctx, 'refresh', 12)} ${translate(ctx, 'Check status', '检查状态')}</button>
          ${canEnableSaved ? `<button type="button" class="btn sm primary" data-gi-enable-ai>${translate(ctx, 'Enable configured API', '启用已配置 API')}</button>` : ''}
          <button type="button" class="btn sm" data-gi-provider-config-toggle>${icon(ctx, 'gear', 12)} ${showConfig ? translate(ctx, 'Hide setup', '收起配置') : translate(ctx, 'Show setup', '展开配置')}</button>
        </div>
        ${showConfig ? `
          <div class="gdi-feature-list">
            <label class="gdi-field wide">
              <span>${translate(ctx, 'API key', 'API key')}</span>
              <input type="password" autocomplete="off" data-gi-provider-key placeholder="${attr(translate(ctx, 'Paste provider key; it will be saved only on this machine', '粘贴 provider key；只保存到本机'))}">
            </label>
            <div class="gdi-field-grid">
              <label class="gdi-field">
                <span>${translate(ctx, 'Base URL / endpoint', 'Base URL / endpoint')}</span>
                <input data-gi-provider-base placeholder="http://127.0.0.1:8787/v1">
              </label>
              <label class="gdi-field">
                <span>${translate(ctx, 'Model', '模型')}</span>
                <input data-gi-provider-model placeholder="gpt5.4">
              </label>
            </div>
            <label class="gdi-check">
              <input type="checkbox" data-gi-provider-enable ${aiOn ? 'checked' : ''}>
              <span>${translate(ctx, 'Enable AI/provider opt-in after saving', '保存后启用 AI/provider opt-in')}</span>
            </label>
            <div class="gdx-actions">
              <button type="button" class="btn primary" data-gi-provider-save ${current.saving ? 'disabled' : ''}>${current.saving ? translate(ctx, 'Saving...', '保存中...') : translate(ctx, 'Save API config locally', '本地保存 API 配置')}</button>
              <span class="muted">${translate(ctx, 'Secrets are written to a local private env file and are never echoed back.', '密钥写入本机私有 env 文件，不会回显。')}</span>
            </div>
            ${current.saved ? `<div class="gdx-status ok"><span>${icon(ctx, 'check', 12)}</span><div><strong>${translate(ctx, 'API config saved', 'API 配置已保存')}</strong><small>${translate(ctx, 'You can continue to source selection.', '现在可以继续选择来源。')}</small></div></div>` : ''}
            ${current.saveError ? `<div class="gdx-status bad"><span>${icon(ctx, 'x', 12)}</span><div><strong>${translate(ctx, 'Could not save API config', 'API 配置保存失败')}</strong><small>${esc(current.saveError)}</small></div></div>` : ''}
          </div>` : ''}
        <div class="gdx-actions">
          <button type="button" class="btn primary" data-gi-api-continue>${continueLabel}</button>
          <span class="muted">${translate(ctx, 'Next step: choose PDF, article clue, literature folder, or frontier topic.', '下一步：选择 PDF、文章线索、文献库文件夹或前沿主题。')}</span>
        </div>
      </div>`;
  }

  function renderMiniStatus(ctx) {
    const current = state(ctx);
    const st = current.status || {};
    const ready = !!st.ready;
    const missing = Array.isArray(st.missing) ? st.missing : [];
    const detail = ready
      ? translate(ctx, 'API-ready; provider calls still require explicit action opt-in.', 'API 已就绪；provider 调用仍需要逐步显式确认。')
      : `${translate(ctx, 'Local-only mode is active. Configure API before AI synthesis or full Agent provider calls.', '当前为本地模式。AI 综合或 full Agent provider 调用前再配置 API。')}${missing.length ? ' · ' + esc(missing.join(', ')) : ''}`;
    return `
      <div class="gdx-status ${ready ? 'ok' : 'warn'}">
        <span>${icon(ctx, ready ? 'check' : 'shield', 12)}</span>
        <div><strong>${ready ? translate(ctx, 'API ready', 'API 已就绪') : translate(ctx, 'API not configured', 'API 未配置')}</strong><small>${detail}</small></div>
        <button type="button" class="btn sm" data-gi-api-back>${icon(ctx, 'gear', 12)} ${translate(ctx, 'API setup', '配置 API')}</button>
      </div>`;
  }

  window.EU_GUIDED_IDEA_PROVIDER = {
    createState,
    requestStatus,
    enableProvider,
    saveConfig,
    renderCapabilityPanel,
    renderSetupPrompt,
    renderMiniStatus,
  };
})();
