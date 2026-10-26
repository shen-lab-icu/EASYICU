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

  // Service presets. A proxy or gateway shows its address in the form; an
  // official service keeps it under Advanced. Models are never preset: the
  // service lists them after connecting, and the host recommends one.
  const PRESETS = Object.freeze({
    deepseek: { label: 'DeepSeek API', provider: 'deepseek', base_url: 'https://api.deepseek.com/v1', api_transport: 'openai-completions', group: 'cn' },
    qwen: { label: 'Qwen (Alibaba Cloud Model Studio)', zh: '通义千问（阿里云百炼）', provider: 'dashscope', base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', api_transport: 'openai-completions', group: 'cn' },
    zhipu: { label: 'Zhipu GLM', zh: '智谱 GLM', provider: 'zhipu', base_url: 'https://open.bigmodel.cn/api/paas/v4', api_transport: 'openai-completions', group: 'cn' },
    moonshot: { label: 'Kimi (Moonshot AI)', zh: 'Kimi（月之暗面）', provider: 'moonshot', base_url: 'https://api.moonshot.cn/v1', api_transport: 'openai-completions', group: 'cn' },
    doubao: { label: 'Doubao (Volcengine Ark)', zh: '豆包（火山方舟）', provider: 'volcengine-ark', base_url: 'https://ark.cn-beijing.volces.com/api/v3', api_transport: 'openai-completions', group: 'cn' },
    siliconflow: { label: 'SiliconFlow', zh: '硅基流动', provider: 'siliconflow', base_url: 'https://api.siliconflow.cn/v1', api_transport: 'openai-completions', group: 'cn' },
    minimax: { label: 'MiniMax', provider: 'minimax', base_url: 'https://api.minimaxi.com/v1', api_transport: 'openai-completions', group: 'cn' },
    openai: { label: 'OpenAI API', provider: 'openai', base_url: 'https://api.openai.com/v1', api_transport: 'openai-responses', group: 'intl' },
    anthropic: { label: 'Anthropic / Claude API', provider: 'anthropic', base_url: 'https://api.anthropic.com/v1', api_transport: 'anthropic-messages', group: 'intl' },
    google: { label: 'Google Gemini API', provider: 'google', base_url: 'https://generativelanguage.googleapis.com/v1beta', api_transport: 'google-generative-ai', group: 'intl' },
    // OpenRouter also signs in with its own account (OAuth PKCE); the key
    // stays on this computer (pi_copilot/openrouter_login.py).
    openrouter: { label: 'OpenRouter API', provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1', api_transport: 'openai-completions', group: 'intl', login: true },
    // Institutional clouds: the address names the researcher's own resource
    // or region, so it starts empty like the custom gateway.
    azure: { label: 'Azure OpenAI', provider: 'azure-openai', base_url: '', api_transport: 'openai-completions', group: 'cloud',
      hint: ['Your resource’s v1 address: https://<resource>.openai.azure.com/openai/v1. Choose your deployment name as the model.', '填写资源的 v1 地址：https://<资源名>.openai.azure.com/openai/v1。模型请选择你的部署名。'] },
    bedrock: { label: 'Amazon Bedrock', provider: 'amazon-bedrock', base_url: '', api_transport: 'openai-completions', group: 'cloud',
      hint: ['Bedrock’s OpenAI-compatible address: https://bedrock-mantle.<region>.api.aws/v1, with a Bedrock API key.', '填写 Bedrock 的 OpenAI 兼容地址：https://bedrock-mantle.<区域>.api.aws/v1，凭据用 Bedrock API 密钥。'] },
    // Models on this computer. These services do not check the credential.
    ollama: { label: 'Ollama (this computer)', zh: 'Ollama（本机）', provider: 'ollama', base_url: 'http://127.0.0.1:11434/v1', api_transport: 'openai-completions', group: 'local',
      hint: ['Ollama listens on http://127.0.0.1:11434/v1 by default. It does not check the credential; enter any text, such as ollama.', 'Ollama 默认地址为 http://127.0.0.1:11434/v1。它不校验凭据，填任意文字即可，例如 ollama。'] },
    lmstudio: { label: 'LM Studio (this computer)', zh: 'LM Studio（本机）', provider: 'lmstudio', base_url: 'http://127.0.0.1:1234/v1', api_transport: 'openai-completions', group: 'local',
      hint: ['LM Studio’s local server listens on http://127.0.0.1:1234/v1 by default. It does not check the credential; enter any text, such as lmstudio.', 'LM Studio 本地服务默认地址为 http://127.0.0.1:1234/v1。它不校验凭据，填任意文字即可，例如 lmstudio。'] },
    cliproxyapi: { label: 'CLIProxyAPI / Local proxy', provider: 'easyicu-local', base_url: 'http://127.0.0.1:8317/v1', api_transport: 'openai-completions', group: 'gateway',
      hint: ['CLIProxyAPI listens on http://127.0.0.1:8317/v1 by default. Change the port if yours differs.', 'CLIProxyAPI 默认地址为 http://127.0.0.1:8317/v1，端口不同请修改。'] },
    // D-P2-4: the custom gateway ships an EMPTY address on purpose. The
    // example domain is placeholder text only and must never be a
    // submittable value: the credential would reach a stand-in host. The
    // control refuses empty and example.* addresses, and so does the backend.
    'custom-openai': { label: 'OpenAI-compatible gateway', provider: 'custom-openai', base_url: '', api_transport: 'openai-completions', group: 'gateway',
      hint: ['The gateway’s OpenAI-compatible address, usually ending in /v1.', '填写网关的 OpenAI 兼容地址，通常以 /v1 结尾。'] },
  });

  const PRESET_GROUPS = Object.freeze([
    ['cn', 'Services in China', '国内服务'],
    ['intl', 'International services', '国际服务'],
    ['cloud', 'Institutional cloud', '机构云'],
    ['local', 'This computer', '本机'],
    ['gateway', 'Proxy and gateway', '代理与网关'],
  ]);

  // A first visit, with no saved connection, starts on the first service.
  const DEFAULT_PRESET = 'deepseek';

  function connectionSignature(values) {
    return ['provider', 'base_url', 'api_transport'].map(name => String((values && values[name]) || '').trim()).join('\n');
  }

  // What the form shows: the researcher's unsaved draft first, then the
  // saved connection. The credential is never part of this.
  function apiFormValues(options) {
    const { state, runtime, config, providerPreset } = options;
    const draft = state.apiDraft || {};
    // Without a saved connection the host reports only its built-in
    // defaults; those are not the researcher's choice, so the form starts
    // on DEFAULT_PRESET and takes every connection field from it.
    const saved = config.credential_present === true;
    const preset = PRESETS[draft.preset] ? draft.preset : saved ? providerPreset(config, runtime) : DEFAULT_PRESET;
    const pick = (name, fallback) => (Object.prototype.hasOwnProperty.call(draft, name) ? String(draft[name] || '') : fallback);
    const from = saved ? name => config[name] || (name === 'base_url' ? '' : runtime[name]) || PRESETS[preset][name] : name => PRESETS[preset][name];
    return {
      preset,
      provider: pick('provider', from('provider')),
      base_url: pick('base_url', from('base_url')),
      api_transport: pick('api_transport', from('api_transport')),
      model: pick('model', config.model || ''),
    };
  }

  function modelStep(options, values, catalog) {
    const { tr, option, esc } = options;
    if (catalog.manual) {
      return `
        <div class="gpi-setup-grid gpi-api-model-step">
          <label class="wide"><span>${tr('Model', '模型')}</span><input name="model" maxlength="256" value="${esc(values.model)}" spellcheck="false" autocomplete="off" required><small class="gpi-field-hint">${tr('This service does not list its models. Enter the model name; saving asks it for one short reply.', '该服务不提供模型列表。请填写模型名，保存时会让它试答一句。')}</small></label>
        </div>`;
    }
    if (!catalog.models.length) {
      return `<div class="gpi-config-note warn"><span class="gpi-dot"></span>${tr('This service reported no models. Check the address and credential, then connect again.', '该服务没有返回任何模型。请检查地址和凭据后重新连接。')}</div>`;
    }
    const selected = catalog.models.includes(values.model) ? values.model : catalog.recommended;
    return `
        <div class="gpi-setup-grid gpi-api-model-step">
          <label class="wide"><span>${tr('Model', '模型')}</span><select name="model" required>${selected ? '' : `<option value="" selected disabled>${tr('Choose a model', '选择模型')}</option>`}${catalog.models.map(model => option(model, selected, model === catalog.recommended ? model + tr(' · recommended', ' · 推荐') : model)).join('')}</select><small class="gpi-field-hint">${tr(`${catalog.models.length} models reported by this service`, `该服务返回了 ${catalog.models.length} 个模型`)}</small></label>
        </div>`;
  }

  function apiConnectionForm(options) {
    const { state, config, tr, esc, option } = options;
    const values = apiFormValues(options);
    // The saved connection's model and credential belong to it alone; another
    // service in the form neither shows that model nor can reuse that key.
    const runtime = options.runtime || {};
    const savedConnection = config.credential_present === true && connectionSignature(values) === connectionSignature({
      provider: config.provider || runtime.provider, base_url: config.base_url, api_transport: config.api_transport || runtime.api_transport,
    });
    const savedCredential = savedConnection;
    const preset = PRESETS[values.preset];
    const catalog = state.modelCatalog && state.modelCatalog.signature === connectionSignature(values) ? state.modelCatalog : null;
    const busy = state.setupSaving || state.modelDiscovering || state.openrouterStarting;
    // After an OpenRouter sign-in the host holds the key; the form asks for none.
    const signedIn = preset.login && state.openrouterLogin === 'signed_in';
    const addressField = `<label class="wide"><span>${tr('Service address', '服务地址')}</span><input name="base_url" maxlength="2048" value="${esc(values.base_url)}" inputmode="url" spellcheck="false" required placeholder="https://llm-gateway.example/v1">${preset.hint ? `<small class="gpi-field-hint">${tr(preset.hint[0], preset.hint[1])}</small>` : ''}</label>`;
    const savedModel = !catalog && savedConnection && config.connection_verified === true && config.model
      ? `<div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Current model:', '当前模型：')} ${esc(config.model)}</div>` : '';
    const submitLabel = state.modelDiscovering ? tr('Connecting…', '正在连接…')
      : state.setupSaving ? tr('Verifying…', '正在验证…')
      : catalog && (catalog.models.length || catalog.manual) ? tr('Verify and save', '验证并保存') : tr('Connect and list models', '连接并读取模型');
    return `
      <form class="gpi-provider-section gpi-api-form" data-gpi-provider-form autocomplete="off">
        <div class="gpi-provider-heading"><h3>${tr('Connect an API service', '连接 API 服务')}</h3></div>
        ${options.staticPreview ? `<div class="gpi-preview-note"><span aria-hidden="true">i</span><span>${tr('Open EasyICU from the local service to configure this connection.', '请从 EasyICU 本地服务打开后配置连接。')}</span></div>` : ''}
        <div class="gpi-setup-grid">
          <label class="wide"><span>${tr('Service', '服务')}</span><select data-gpi-provider-preset>${PRESET_GROUPS.map(([group, en, zh]) => `<optgroup label="${esc(tr(en, zh))}">${Object.keys(PRESETS).filter(key => PRESETS[key].group === group).map(key => option(key, values.preset, PRESETS[key].zh ? tr(PRESETS[key].label, PRESETS[key].zh) : PRESETS[key].label)).join('')}</optgroup>`).join('')}</select></label>
          ${preset.hint ? addressField : ''}
          ${signedIn ? `<div class="wide gpi-config-note ok"><span class="gpi-dot"></span>${tr('Signed in with OpenRouter. The key stays on this computer.', '已用 OpenRouter 账户登录，密钥只保存在本机。')}</div>` : `
          ${preset.login ? `<div class="wide gpi-provider-login"><button class="btn" type="button" data-gpi-openrouter-login ${busy || options.staticPreview ? 'disabled' : ''}>${tr('Sign in with OpenRouter', '用 OpenRouter 账户登录')}</button></div>` : ''}
          <label class="wide"><span>${tr('API credential', 'API 凭据')}</span><input name="api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="${savedCredential ? tr('Leave blank to use the saved credential', '留空则沿用已保存的凭据') : tr('Paste once; it will not be shown again', '仅粘贴一次，之后不再显示')}"></label>`}
        </div>
        <details class="gpi-api-advanced">
          <summary>${tr('Advanced settings', '高级设置')}</summary>
          <div class="gpi-setup-grid">
            <label><span>${tr('Provider ID', '提供方标识')}</span><input name="provider" maxlength="80" value="${esc(values.provider)}" required></label>
            <label><span>${tr('Compatibility protocol', '兼容协议')}</span><select name="api_transport">${option('openai-completions', values.api_transport, 'OpenAI Chat Completions')}${option('openai-responses', values.api_transport, 'OpenAI Responses')}${option('anthropic-messages', values.api_transport, 'Anthropic Messages')}${option('google-generative-ai', values.api_transport, 'Google Generative AI')}</select></label>
            ${preset.hint ? '' : addressField}
          </div>
        </details>
        ${catalog ? modelStep(options, values, catalog) : savedModel}
        <div class="gpi-connection-notice">${tr('Connecting uses this credential to read the service’s model list. After saving, conversation text and files you choose may be sent to this service.', '连接时会用这个凭据读取该服务的模型列表。保存后，对话内容和你选择的文件可能发送至该服务。')}</div>
        <button class="btn ${options.connectionConfigured ? '' : 'primary'} gpi-provider-submit" type="submit" ${busy || options.staticPreview ? 'disabled' : ''}>${submitLabel}</button>
        ${catalog && !busy ? `<button class="gpi-link gpi-provider-rediscover" type="button" data-gpi-provider-rediscover>${tr('List models again', '重新读取模型')}</button>` : ''}
        ${!options.staticPreview && config.connection_verified === true && values.api_transport !== 'openai-completions' ? `<div class="gpi-config-note warn"><span class="gpi-dot"></span>${tr('Use the OpenAI Chat Completions protocol for full research assistance.', '完整研究协助需要 OpenAI Chat Completions 兼容协议。')}</div>` : ''}
      </form>`;
  }

  // How the saved API connection handles credentialed data
  // (screens-guided-pi-provider-declaration.js).
  function declarationSection(options) {
    const owner = window.EasyICU.guidedPi.optional('providerDeclaration');
    return owner ? owner.renderSection(options) : '';
  }

  // The choice, form and error, shared by the connection page and the
  // connection dialog that a step needing a model opens. The credentialed-data
  // declaration is asked only when a run needed it (state.declarationAsked,
  // set by the stopped run's action), and then it leads the page.
  function renderConnectionBody(options) {
    const { state, tr, esc } = options;
    const asked = state.researchProvider === 'api' && state.declarationAsked === true;
    return `
      ${asked ? declarationSection(options) : ''}
      ${connectionChoice({ state, tr, esc })}
      ${state.researchProvider === 'api' ? apiConnectionForm(options) : ''}
      ${state.error ? `<div class="gpi-error inline">${esc(state.error)}</div>` : ''}`;
  }

  function renderSetup(options) {
    const { tr } = options;
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
            ${renderConnectionBody(options)}
            ${canCancel ? `<div class="gpi-setup-actions"><button class="btn primary" type="button" data-gpi-provider-done>${tr('Enter research workspace', '进入研究工作区')}</button></div>` : `
            <div class="gpi-setup-actions gpi-setup-actions-later">
              <button class="btn gpi-connect-later" type="button" data-gpi-connect-later>${tr('Explore first — connect when a step needs a model', '先进去看看，用到模型时再连接')}</button>
              <button class="gpi-link" type="button" data-gpi-demo>${tr('View the workflow demo first (no connection needed)', '先看流程演示（无需连接）')}</button>
            </div>`}
          </div>
        </section>
      </div>`;
  }

  function renderBindingSummary(options) {
    const { state, tr, esc } = options;
    const hosted = window.EasyICU.guidedPi.optional('hosted');
    if (hosted && hosted.active()) return hosted.summaryHtml({ tr, esc, connectionReady: options.connectionReady });
    const codex = state.researchProvider === 'codex';
    const auth = state.codexAuth || {};
    const label = codex
      ? [tr('ChatGPT / Codex account', 'ChatGPT / Codex 账户'), state.researchModel || tr('select a model', '请选择模型')].join(' · ')
      : tr('Verified API connection', '已验证的 API 连接');
    const ready = codex ? auth.authentication_verified === true && !!state.researchModel : options.apiResearchReady;
    if (!options.connectionReady) {
      return `<div class="gpi-provider-summary warn"><span class="gpi-dot"></span><div><strong>${tr('No model connected', '未连接模型')}</strong><small>${tr('Needed for conversations and research runs', '对话和研究运行需要模型')}</small></div><button class="gpi-link" type="button" data-gpi-connect-open>${tr('Connect', '连接')}</button></div>`;
    }
    return `<div class="gpi-provider-summary ${ready ? 'ready' : 'warn'}"><span class="gpi-dot"></span><div><strong>${tr('Copilot + Research Agent', 'Copilot + Research Agent')}</strong><small>${esc(label)}</small></div><button class="gpi-link" type="button" data-gpi-config>${tr('Change', '更改')}</button></div>`;
  }

  window.EasyICU.guidedPi.declare('provider', { renderSetup, renderConnectionBody, renderBindingSummary, PRESETS, PRESET_GROUPS, DEFAULT_PRESET, connectionSignature });
})();
