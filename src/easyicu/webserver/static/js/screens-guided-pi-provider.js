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
      <section class="gpi-provider-section" aria-labelledby="gpi-model-connection-title">
        <div class="gpi-provider-heading">
          <div><div class="gpi-kicker">${tr('ONE MODEL CONNECTION', '单一模型连接')}</div><h3 id="gpi-model-connection-title">${tr('Account or API', '账户或 API')}</h3></div>
          <span class="gpi-provider-lock">${tr('Frozen when the conversation starts', '会话创建时冻结')}</span>
        </div>
        <p>${tr('The same selected provider and model powers Copilot conversation and the governed plan → execute → verify workflow. EasyICU still isolates their context, permissions, and evidence rules internally.', '同一个提供方与模型同时用于 Copilot 对话和受治理的“计划 → 执行 → 核验”流程；EasyICU 仍会在内部隔离上下文、权限与证据规则。')}</p>
        <div class="gpi-provider-choices" role="radiogroup" aria-label="${tr('Model connection', '模型连接')}">
          <button type="button" role="radio" data-gpi-research-provider="codex" aria-checked="${selected === 'codex'}">
            <span class="gpi-provider-radio"></span><span><strong>${tr('ChatGPT / Codex account', 'ChatGPT / Codex 账户')}</strong><small>${tr('Browser sign-in; no API key', '浏览器登录，无需 API Key')}</small></span>
          </button>
          <button type="button" role="radio" data-gpi-research-provider="api" aria-checked="${selected === 'api'}">
            <span class="gpi-provider-radio"></span><span><strong>${tr('API connection', 'API 连接')}</strong><small>${tr('One compatible API for conversation and analysis', '同一套兼容 API 用于对话与分析')}</small></span>
          </button>
        </div>
        ${selected === 'codex' ? `
          <div class="gpi-account-box ${accountReady ? 'ready' : ''}">
            ${accountReady ? `
              <div class="gpi-account-row"><span class="gpi-dot"></span><div><strong>${tr('Codex account connected', 'Codex 账户已连接')}</strong><small>${esc([auth.account_label, auth.plan_type].filter(Boolean).join(' · '))}</small></div><button class="gpi-link" type="button" data-gpi-codex-logout ${state.codexBusy ? 'disabled' : ''}>${tr('Sign out', '退出')}</button></div>
              <label class="gpi-model-field"><span>${tr('Account model', '账户模型')}</span><select data-gpi-codex-model ${state.codexBusy ? 'disabled' : ''}>${models.map(row => `<option value="${esc(row.id)}" ${row.id === model ? 'selected' : ''}>${esc(row.label || row.id)}${row.is_default ? tr(' · default', ' · 默认') : ''}</option>`).join('')}</select></label>
              ${models.length ? '' : `<button class="btn sm" type="button" data-gpi-codex-models>${tr('Load available models', '读取可用模型')}</button>`}
            ` : `
              <div class="gpi-account-row"><span class="gpi-dot waiting"></span><div><strong>${pending ? tr('Complete sign-in in the OpenAI window', '请在 OpenAI 窗口完成登录') : tr('Connect your ChatGPT account', '连接你的 ChatGPT 账户')}</strong><small>${tr('EasyICU never receives your password or copies Codex tokens into project files.', 'EasyICU 不接收你的密码，也不会把 Codex 令牌复制进项目文件。')}</small></div></div>
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
      <form class="gpi-provider-section" data-gpi-provider-form autocomplete="off">
        <div class="gpi-provider-heading"><div><div class="gpi-kicker">API CONNECTION</div><h3>${tr('Configure once for Copilot and Research Agent', '一次配置，同时用于 Copilot 与 Research Agent')}</h3></div><span class="gpi-provider-lock">${tr('Private local credential', '本机私密凭据')}</span></div>
        <p>${tr('The API credential is saved only in EasyICU’s private local credential file and is never returned to this page. Full analysis currently requires an OpenAI Chat Completions-compatible endpoint.', 'API 凭据只保存在 EasyICU 本机私密凭据文件中，不会回传到页面。完整分析目前需要兼容 OpenAI Chat Completions 的端点。')}</p>
        <div class="gpi-setup-grid">
          <label><span>${tr('Service type', '服务类型')}</span><select data-gpi-provider-preset>${option('cliproxyapi', preset, 'CLIProxyAPI / Local proxy')}${option('custom-openai', preset, 'OpenAI-compatible gateway')}${option('openai', preset, 'OpenAI API')}${option('openrouter', preset, 'OpenRouter API')}${option('deepseek', preset, 'DeepSeek API')}${option('anthropic', preset, 'Anthropic / Claude API')}${option('google', preset, 'Google Gemini API')}</select></label>
          <label><span>${tr('Provider ID', '提供方标识')}</span><input name="provider" maxlength="80" value="${esc(config.provider || runtime.provider || 'easyicu-local')}" required></label>
          <label class="wide"><span>${tr('Service address', '服务地址')}</span><input name="base_url" maxlength="2048" value="${esc(config.base_url || 'http://127.0.0.1:8317/v1')}" inputmode="url" spellcheck="false" required></label>
          <label><span>${tr('Compatibility protocol', '兼容协议')}</span><select name="api_transport">${option('openai-completions', transport, 'OpenAI Chat Completions')}${option('openai-responses', transport, 'OpenAI Responses')}${option('anthropic-messages', transport, 'Anthropic Messages')}${option('google-generative-ai', transport, 'Google Generative AI')}</select></label>
          <label><span>${tr('Model', '模型')}</span><input name="model" list="gpi-model-options" maxlength="256" value="${esc(config.model || runtime.model || 'gpt-5.6-luna')}" spellcheck="false" required><datalist id="gpi-model-options">${discovered}</datalist></label>
          <label><span>${tr('API credential', 'API 凭据')}</span><input name="api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="${savedCredential ? tr('Re-enter only to replace or re-verify', '仅在更换或重新验证时输入') : tr('Paste once; it will not be shown again', '仅粘贴一次，之后不再显示')}" ${options.runtimeReady ? '' : 'required'}></label>
        </div>
        ${state.availableModels.length ? `<div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Models reported by this service:', '该服务返回的可用模型：')} ${esc(state.availableModels.slice(0, 12).join(', '))}</div>` : ''}
        ${runtimeMissing.length ? `<div class="gpi-config-note warn gpi-blockers"><div class="gpi-blocker-lead">${tr('Fix these before the connection can open:', '连接开放前需要先解决：')}</div><ol class="gpi-blocker-list">${runtimeMissing.map(b => `<li><span class="gpi-blocker-title">${esc(b.title)}</span>${b.fix ? `<span class="gpi-blocker-fix">${esc(b.fix)}</span>` : ''}<span class="gpi-blocker-code mono" title="${esc(tr('Diagnostic code reported by the Copilot runtime', '研究助手运行环境上报的诊断码'))}">${esc(b.code)}</span></li>`).join('')}</ol></div>` : ''}
        <div class="gpi-consent">${tr('By verifying and saving, you authorize this connection check and allow EasyICU to use this service for later conversations. Conversation text, PHI-safe summaries, and workspace file contents may be sent to this service; scientific runs still require separate confirmation.', '点击“验证并保存连接”即授权本次连接检查，并允许 EasyICU 在后续对话中使用该服务。对话文字、经 PHI 安全投影的摘要和工作区文件内容可能发送到该服务；科研运行仍需另行确认。')}</div>
        <button class="btn" type="submit" ${state.setupSaving || options.staticPreview ? 'disabled' : ''}>${state.setupSaving ? tr('Verifying…', '正在验证…') : tr('Verify and save connection', '验证并保存连接')}</button>
        <div class="gpi-config-note ${options.apiResearchReady ? 'ok' : 'warn'}"><span class="gpi-dot"></span>${options.apiResearchReady ? tr('This verified connection is ready for both conversation and analysis.', '这套已验证连接已可同时用于对话和分析。') : tr('Choose the OpenAI Chat Completions protocol for one connection that supports both.', '请选择 OpenAI Chat Completions 协议，使这一套连接同时支持对话与分析。')}</div>
      </form>`;
  }

  function renderSetup(options) {
    const { state, runtime, config, tr, esc } = options;
    const canCancel = options.connectionConfigured;
    return `
      <div class="gpi-setup-wrap gpi-provider-setup">
        <div class="gpi-setup gpi-provider-shell">
          <div class="gpi-kicker">${tr('EASYICU COPILOT · ONE MODEL CONNECTION', 'EASYICU COPILOT · 单一模型连接')}</div>
          <h2>${tr('Choose one provider and model', '只选择一套提供方与模型')}</h2>
          <p>${tr('Use a ChatGPT/Codex account or one API connection. The choice powers both the conversation and governed scientific workflow; you do not configure a second analysis model elsewhere.', '可以使用 ChatGPT/Codex 账户，也可以配置一套 API。这一选择会同时用于对话与受治理的科研流程，无需再到其他页面配置第二个分析模型。')}</p>
          ${connectionChoice({ state, tr, esc })}
          ${state.researchProvider === 'api' ? apiConnectionForm(options) : ''}
          ${state.error ? `<div class="gpi-error inline">${esc(state.error)}</div>` : ''}
          <div class="gpi-setup-actions">
            ${canCancel ? `<button class="btn primary" type="button" data-gpi-provider-done>${tr('Finish connection setup', '完成连接设置')}</button>` : `<button class="gpi-link" type="button" data-gpi-legacy>${tr('Use local Guided workflow', '使用本地研究引导流程')}</button>`}
          </div>
          <div class="gpi-consent">${tr('Connecting an account or verifying an API authorizes that model connection for EasyICU. Credentials stay in private local storage; existing conversations never switch connection silently.', '连接账户或验证 API，即授权 EasyICU 使用该模型连接。凭据保存在本机私密空间中；已有会话绝不会静默切换连接。')}</div>
        </div>
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

  window.EU_GUIDED_PI_PROVIDER = Object.freeze({ renderSetup, renderBindingSummary });
})();
