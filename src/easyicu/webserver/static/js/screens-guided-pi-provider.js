/* Owner: Guided Pi model connection widget. */
/* Copilot-owned model connection selection.
   One immutable provider/model binding powers both conversation and governed
   analysis. Account state and scientific-run authority remain server-owned. */
(function () {
  'use strict';

  function connectionChoice(options) {
    const { state, tr, esc } = options;
    const auth = state.codexAuth || {};
    const login = state.codexLogin || {};
    const models = Array.isArray(state.codexModels) ? state.codexModels : [];
    const accountReady = auth.authentication_verified === true;
    const pending = auth.account_session_status === 'codex_auth_login_pending';
    const selected = state.researchProvider === 'codex' ? 'codex' : 'api';
    const model = state.researchModel || (models.find(row => row.is_default) || models[0] || {}).id || '';
    return `
      <section class="gpi-provider-section gpi-provider-choice" aria-labelledby="gpi-model-connection-title">
        <div class="gpi-provider-heading">
          <h3 id="gpi-model-connection-title">${tr('Choose a connection', '选择连接方式')}</h3>
        </div>
        <div class="gpi-provider-choices" role="radiogroup" aria-label="${tr('Model connection', '模型连接')}">
          <button type="button" role="radio" data-gpi-research-provider="codex" aria-checked="${selected === 'codex'}">
            <span class="gpi-provider-radio"></span><strong>${tr('ChatGPT / Codex account', 'ChatGPT / Codex 账户')}</strong>
          </button>
          <button type="button" role="radio" data-gpi-research-provider="api" aria-checked="${selected === 'api'}">
            <span class="gpi-provider-radio"></span><strong>${tr('API connection', 'API 连接')}</strong>
          </button>
        </div>
        ${selected === 'codex' ? `
          <div class="gpi-account-box ${accountReady ? 'ready' : ''}">
            ${accountReady ? `
              <div class="gpi-account-row"><span class="gpi-dot"></span><div><strong>${tr('Codex account connected', 'Codex 账户已连接')}</strong>${[auth.account_label, auth.plan_type].filter(Boolean).length ? `<span class="gpi-account-meta">${esc([auth.account_label, auth.plan_type].filter(Boolean).join(' · '))}</span>` : ''}</div><button class="gpi-link" type="button" data-gpi-codex-logout ${state.codexBusy ? 'disabled' : ''}>${tr('Sign out', '退出')}</button></div>
              <label class="gpi-model-field"><span>${tr('Account model', '账户模型')}</span><select data-gpi-codex-model ${state.codexBusy ? 'disabled' : ''}>${models.map(row => `<option value="${esc(row.id)}" ${row.id === model ? 'selected' : ''}>${esc(row.label || row.id)}${row.is_default ? tr(' · default', ' · 默认') : ''}</option>`).join('')}</select></label>
              ${models.length ? '' : `<button class="btn sm" type="button" data-gpi-codex-models>${tr('Load available models', '读取可用模型')}</button>`}
            ` : `
              <div class="gpi-account-row"><span class="gpi-dot waiting"></span><strong>${pending ? tr('Complete sign-in in the OpenAI window', '请在 OpenAI 窗口完成登录') : tr('Connect your ChatGPT account', '连接你的 ChatGPT 账户')}</strong></div>
              <div class="gpi-account-actions">
                <button class="btn primary" type="button" data-gpi-codex-login ${state.codexBusy ? 'disabled' : ''}>${tr('Continue with ChatGPT', '使用 ChatGPT 继续')}</button>
                <button class="gpi-link" type="button" data-gpi-codex-device ${state.codexBusy ? 'disabled' : ''}>${tr('Device-code fallback', '改用设备码')}</button>
                ${pending ? `<button class="gpi-link" type="button" data-gpi-codex-cancel ${state.codexBusy ? 'disabled' : ''}>${tr('Cancel', '取消')}</button>` : ''}
              </div>
              ${login.auth_url ? `<a class="gpi-auth-fallback" href="${esc(login.auth_url)}" target="_blank" rel="noopener noreferrer">${tr('Open the authorization page again', '重新打开授权页面')}</a>` : ''}
              ${login.user_code ? `<div class="gpi-device-code">${tr('Enter this code on the OpenAI page:', '请在 OpenAI 页面输入此代码：')} <strong>${esc(login.user_code)}</strong></div>` : ''}
            `}
          </div>
        ` : ''}
      </section>`;
  }

  function apiConnectionForm(options) {
    const { state, runtime, config, runtimeMissing, tr, esc, option, providerPreset } = options;
    const savedCredential = !!config.credential_present;
    const preset = providerPreset(config, runtime);
    const transport = config.api_transport || runtime.api_transport || 'openai-completions';
    const discovered = state.availableModels.map(model => `<option value="${esc(model)}"></option>`).join('');
    return `
      <form class="gpi-provider-section gpi-api-form" data-gpi-provider-form autocomplete="off">
        <div class="gpi-provider-heading"><h3>${tr('Connect an API service', '连接 API 服务')}</h3></div>
        ${options.staticPreview ? `<div class="gpi-preview-note"><span aria-hidden="true">i</span><span>${tr('Open EasyICU from the local service to configure this connection.', '请从 EasyICU 本地服务打开后配置连接。')}</span></div>` : ''}
        <div class="gpi-setup-grid">
          <label class="wide"><span>${tr('Service', '服务')}</span><select data-gpi-provider-preset>${option('cliproxyapi', preset, 'CLIProxyAPI / Local proxy')}${option('custom-openai', preset, 'OpenAI-compatible gateway')}${option('openai', preset, 'OpenAI API')}${option('openrouter', preset, 'OpenRouter API')}${option('deepseek', preset, 'DeepSeek API')}${option('anthropic', preset, 'Anthropic / Claude API')}${option('google', preset, 'Google Gemini API')}</select></label>
          <label><span>${tr('Model', '模型')}</span><input name="model" list="gpi-model-options" maxlength="256" value="${esc(config.model || runtime.model || 'gpt-5.6-luna')}" spellcheck="false" required><datalist id="gpi-model-options">${discovered}</datalist></label>
          <label><span>${tr('API credential', 'API 凭据')}</span><input name="api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="${savedCredential ? tr('Re-enter only to replace or re-verify', '仅在更换或重新验证时输入') : tr('Paste once; it will not be shown again', '仅粘贴一次，之后不再显示')}" ${options.runtimeReady ? '' : 'required'}></label>
        </div>
        <details class="gpi-api-advanced">
          <summary>${tr('Advanced settings', '高级设置')}</summary>
          <div class="gpi-setup-grid">
            <label><span>${tr('Provider ID', '提供方标识')}</span><input name="provider" maxlength="80" value="${esc(config.provider || runtime.provider || 'easyicu-local')}" required></label>
            <label><span>${tr('Compatibility protocol', '兼容协议')}</span><select name="api_transport">${option('openai-completions', transport, 'OpenAI Chat Completions')}${option('openai-responses', transport, 'OpenAI Responses')}${option('anthropic-messages', transport, 'Anthropic Messages')}${option('google-generative-ai', transport, 'Google Generative AI')}</select></label>
            <label class="wide"><span>${tr('Service address', '服务地址')}</span><input name="base_url" maxlength="2048" value="${esc(config.base_url || 'http://127.0.0.1:8317/v1')}" inputmode="url" spellcheck="false" required placeholder="https://llm-gateway.example/v1"></label>
          </div>
        </details>
        ${state.availableModels.length ? `<div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Models reported by this service:', '该服务返回的可用模型：')} ${esc(state.availableModels.slice(0, 12).join(', '))}</div>` : ''}
        ${!options.staticPreview && runtimeMissing.length ? `<div class="gpi-config-note warn gpi-blockers"><div class="gpi-blocker-lead">${tr('Before connecting:', '连接前请处理：')}</div><ol class="gpi-blocker-list">${runtimeMissing.map(b => `<li><span class="gpi-blocker-title">${esc(b.title)}</span>${b.fix ? `<span class="gpi-blocker-fix">${esc(b.fix)}</span>` : ''}<details class="gpi-blocker-diagnostic"><summary>${tr('Diagnostic details', '诊断详情')}</summary><code>${esc(b.code)}</code></details></li>`).join('')}</ol></div>` : ''}
        <div class="gpi-connection-notice">${tr('After saving, conversation text and files you choose may be sent to this service.', '保存后，对话内容和你选择的文件可能发送至该服务。')}</div>
        <button class="btn ${options.connectionConfigured ? '' : 'primary'} gpi-provider-submit" type="submit" ${state.setupSaving || options.staticPreview ? 'disabled' : ''}>${state.setupSaving ? tr('Verifying…', '正在验证…') : tr('Verify and save', '验证并保存')}</button>
        ${!options.staticPreview && !options.apiResearchReady ? `<div class="gpi-config-note warn"><span class="gpi-dot"></span>${tr('Use the OpenAI Chat Completions protocol for full research assistance.', '完整研究协助需要 OpenAI Chat Completions 兼容协议。')}</div>` : ''}
      </form>`;
  }

  function renderSetup(options) {
    const { state, runtime, config, tr, esc } = options;
    const canCancel = options.connectionConfigured;
    const languageLabel = window.EU_LANG === 'zh' ? 'EN' : '中';
    return `
      <div class="gpi-setup-wrap gpi-provider-setup" data-gpi-connection-page>
        <header class="gpi-connection-topbar">
          <div class="gpi-connection-brand" aria-label="EasyICU">
            <span class="gpi-connection-mark" aria-hidden="true">✦</span>
            <strong>EasyICU</strong>
          </div>
          <button class="gpi-connection-language" type="button" data-lang-toggle aria-label="${tr('Switch language', '切换语言')}">${languageLabel}</button>
        </header>
        <section class="gpi-connection-stage" aria-labelledby="gpi-connection-title">
          <div class="gpi-connection-intro">
            <h2 id="gpi-connection-title">${tr('Connect EasyICU', '连接 EasyICU')}</h2>
          </div>
          <div class="gpi-setup gpi-provider-shell">
            ${connectionChoice({ state, tr, esc })}
            ${state.researchProvider === 'api' ? apiConnectionForm(options) : ''}
            ${state.error ? `<div class="gpi-error inline">${esc(state.error)}</div>` : ''}
            <div class="gpi-setup-actions">
              ${canCancel ? `<button class="btn primary" type="button" data-gpi-provider-done>${tr('Enter research workspace', '进入研究工作区')}</button>` : `<button class="gpi-link" type="button" data-gpi-legacy>${tr('Continue with the local guided workflow', '暂不连接，使用本地研究引导')}</button>`}
            </div>
          </div>
        </section>
      </div>`;
  }

  function renderBindingSummary(options) {
    const { state, tr, esc } = options;
    const codex = state.researchProvider === 'codex';
    const auth = state.codexAuth || {};
    const label = codex
      ? [tr('ChatGPT / Codex account', 'ChatGPT / Codex 账户'), state.researchModel || tr('select a model', '请选择模型')].join(' · ')
      : tr('Verified API connection', '已验证的 API 连接');
    const ready = codex ? auth.authentication_verified === true && !!state.researchModel : options.apiResearchReady;
    return `<div class="gpi-provider-summary ${ready ? 'ready' : 'warn'}"><span class="gpi-dot"></span><div><strong>${tr('Copilot + Research Agent', 'Copilot + Research Agent')}</strong><small>${esc(label)}</small></div><button class="gpi-link" type="button" data-gpi-config>${tr('Change', '更改')}</button></div>`;
  }

  window.EasyICU.guidedPi.declare('provider', { renderSetup, renderBindingSummary });
})();
