/* Copilot-owned model and credential selection.
   This module renders configuration only; account state and scientific-run
   authority remain server-owned and are frozen into each Pi session. */
(function () {
  'use strict';

  function researchChoice(options) {
    const { state, tr, esc } = options;
    const auth = state.codexAuth || {};
    const login = state.codexLogin || {};
    const models = Array.isArray(state.codexModels) ? state.codexModels : [];
    const accountReady = auth.authentication_verified === true;
    const pending = auth.account_session_status === 'codex_auth_login_pending';
    const selected = state.researchProvider === 'codex' ? 'codex' : 'api';
    const model = state.researchModel || (models.find(row => row.is_default) || models[0] || {}).id || '';
    return `
      <section class="gpi-provider-section" aria-labelledby="gpi-research-provider-title">
        <div class="gpi-provider-heading">
          <div><div class="gpi-kicker">RESEARCH AGENT</div><h3 id="gpi-research-provider-title">${tr('Analysis model', '分析模型')}</h3></div>
          <span class="gpi-provider-lock">${tr('Frozen per conversation', '每段对话创建时冻结')}</span>
        </div>
        <p>${tr('This model runs the governed plan → execute → verify workflow. It is selected here in Copilot; Research Projects only shows the resulting run and evidence.', '这个模型负责受治理的“计划 → 执行 → 核验”流程。请在 Copilot 中选择；研究项目页只展示运行与证据。')}</p>
        <div class="gpi-provider-choices" role="radiogroup" aria-label="${tr('Research Agent model source', 'Research Agent 模型来源')}">
          <button type="button" role="radio" data-gpi-research-provider="codex" aria-checked="${selected === 'codex'}">
            <span class="gpi-provider-radio"></span><span><strong>${tr('ChatGPT / Codex account', 'ChatGPT / Codex 账户')}</strong><small>${tr('Browser sign-in; no API key', '浏览器登录，无需 API Key')}</small></span>
          </button>
          <button type="button" role="radio" data-gpi-research-provider="api" aria-checked="${selected === 'api'}">
            <span class="gpi-provider-radio"></span><span><strong>${tr('Verified API connection', '已验证的 API 连接')}</strong><small>${tr('Use the compatible API configured above', '使用上方配置的兼容 API')}</small></span>
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
        ` : `<div class="gpi-config-note ${options.apiResearchReady ? 'ok' : 'warn'}"><span class="gpi-dot"></span>${options.apiResearchReady ? tr('The verified OpenAI Chat Completions-compatible API will also power Research Agent runs.', '已验证的 OpenAI Chat Completions 兼容 API 也将用于 Research Agent 运行。') : tr('Research Agent currently requires a verified OpenAI Chat Completions-compatible API.', 'Research Agent 当前需要已验证的 OpenAI Chat Completions 兼容 API。')}</div>`}
      </section>`;
  }

  function renderSetup(options) {
    const { state, runtime, config, blockers, runtimeMissing, tr, esc, option, providerPreset } = options;
    const savedCredential = !!config.credential_present;
    const canCancel = options.runtimeReady;
    const preset = providerPreset(config, runtime);
    const transport = config.api_transport || runtime.api_transport || 'openai-completions';
    const discovered = state.availableModels.map(model => `<option value="${esc(model)}"></option>`).join('');
    return `
      <div class="gpi-setup-wrap gpi-provider-setup">
        <div class="gpi-setup gpi-provider-shell">
          <div class="gpi-kicker">PI COPILOT · MODELS & ACCOUNTS</div>
          <h2>${tr('Choose how Copilot and Research Agent use models', '选择 Copilot 与 Research Agent 如何调用模型')}</h2>
          <p>${tr('Copilot conversation and Research Agent analysis have different safety boundaries. Configure both here; no provider controls belong on the Research Projects page.', 'Copilot 对话与 Research Agent 分析具有不同安全边界。两者都在这里配置；研究项目页不再放提供方控件。')}</p>
          <form class="gpi-provider-section" data-gpi-provider-form autocomplete="off">
            <div class="gpi-provider-heading"><div><div class="gpi-kicker">COPILOT</div><h3>${tr('Conversation model API', '对话模型 API')}</h3></div><span class="gpi-provider-lock">${tr('Private local credential', '本机私密凭据')}</span></div>
            <p>${tr('The current Pi conversation shell uses a verified API connection. Your API credential is saved only in EasyICU’s private local credential file and is never returned to this page.', '当前 Pi 对话壳使用已验证的 API 连接。API 凭据只保存在 EasyICU 本机私密凭据文件中，不会回传到页面。')}</p>
            <div class="gpi-setup-grid">
              <label><span>${tr('Service type', '服务类型')}</span><select data-gpi-provider-preset>${option('cliproxyapi', preset, 'CLIProxyAPI / Local proxy')}${option('custom-openai', preset, 'OpenAI-compatible gateway')}${option('openai', preset, 'OpenAI API')}${option('openrouter', preset, 'OpenRouter API')}${option('deepseek', preset, 'DeepSeek API')}${option('anthropic', preset, 'Anthropic / Claude API')}${option('google', preset, 'Google Gemini API')}</select></label>
              <label><span>${tr('Provider ID', '提供方标识')}</span><input name="provider" maxlength="80" value="${esc(config.provider || runtime.provider || 'easyicu-local')}" required></label>
              <label class="wide"><span>${tr('Service address', '服务地址')}</span><input name="base_url" maxlength="2048" value="${esc(config.base_url || 'http://127.0.0.1:8317/v1')}" inputmode="url" spellcheck="false" required></label>
              <label><span>${tr('Compatibility protocol', '兼容协议')}</span><select name="api_transport">${option('openai-completions', transport, 'OpenAI Chat Completions')}${option('openai-responses', transport, 'OpenAI Responses')}${option('anthropic-messages', transport, 'Anthropic Messages')}${option('google-generative-ai', transport, 'Google Generative AI')}</select></label>
              <label><span>${tr('Model', '模型')}</span><input name="model" list="gpi-model-options" maxlength="256" value="${esc(config.model || runtime.model || 'gpt-5.6-luna')}" spellcheck="false" required><datalist id="gpi-model-options">${discovered}</datalist></label>
              <label><span>${tr('API credential', 'API 凭据')}</span><input name="api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="${savedCredential ? tr('Re-enter only to replace or re-verify', '仅在更换或重新验证时输入') : tr('Paste once; it will not be shown again', '仅粘贴一次，之后不再显示')}" ${options.runtimeReady ? '' : 'required'}></label>
            </div>
            ${state.availableModels.length ? `<div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Models reported by this service:', '该服务返回的可用模型：')} ${esc(state.availableModels.slice(0, 12).join(', '))}</div>` : ''}
            ${runtimeMissing.length ? `<div class="gpi-config-note warn gpi-blockers"><div class="gpi-blocker-lead">${tr('Fix these before the conversation can open:', '对话开放前需要先解决：')}</div><ol class="gpi-blocker-list">${runtimeMissing.map(b => `<li><span class="gpi-blocker-title">${esc(b.title)}</span>${b.fix ? `<span class="gpi-blocker-fix">${esc(b.fix)}</span>` : ''}<span class="gpi-blocker-code mono">${esc(b.code)}</span></li>`).join('')}</ol></div>` : ''}
            <label class="gpi-optin"><input name="enable_ai" type="checkbox" required> <span>${tr('I authorize this verification request and external AI use for Pi Copilot. Chat text, PHI-safe summaries, and workspace file contents may be sent to this configured service. Do not place PHI, patient rows, credentials, or private clinical data in the workspace.', '我授权本次连接验证，并允许 Pi Copilot 使用外部 AI；对话文字、经 PHI 安全投影的摘要和工作区文件内容可能发送到所配置的服务。请勿在工作区放置 PHI、患者行级数据、凭据或私密临床数据。')}</span></label>
            <button class="btn" type="submit" ${state.setupSaving || options.staticPreview ? 'disabled' : ''}>${state.setupSaving ? tr('Verifying…', '正在验证…') : tr('Verify API connection', '验证 API 连接')}</button>
          </form>
          ${researchChoice({ state, tr, esc, apiResearchReady: options.apiResearchReady })}
          ${state.error ? `<div class="gpi-error inline">${esc(state.error)}</div>` : ''}
          <div class="gpi-setup-actions">
            ${canCancel ? `<button class="btn" type="button" data-gpi-provider-done>${tr('Done', '完成')}</button>` : `<button class="gpi-link" type="button" data-gpi-legacy>${tr('Use local Guided workflow', '使用本地研究引导流程')}</button>`}
          </div>
          <div class="gpi-consent">${tr('A Codex account binding is isolated per browser and frozen into a new conversation. Existing conversations never switch provider silently.', 'Codex 账户绑定按浏览器隔离，并在新对话创建时冻结；已有对话绝不会静默切换提供方。')}</div>
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
    return `<div class="gpi-provider-summary ${ready ? 'ready' : 'warn'}"><span class="gpi-dot"></span><div><strong>${tr('Research Agent model', 'Research Agent 模型')}</strong><small>${esc(label)}</small></div><button class="gpi-link" type="button" data-gpi-config>${tr('Change', '更改')}</button></div>`;
  }

  window.EU_GUIDED_PI_PROVIDER = Object.freeze({ renderSetup, renderBindingSummary });
})();
