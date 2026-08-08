/* Pi AgentSession client for Guided Copilot.
   Owner: Pi chat/session/tool UX only. Study cards and scientific execution
   remain in their existing EasyICU owners. */
(function () {
  'use strict';

  const state = {
    host: null, conv: null, runtime: null, sessions: [], session: null,
    messages: [], loading: true, creating: false, busy: false, jobId: '',
    source: null, error: '', shell: 'pi', draft: '', setupSaving: false,
    showSetup: false,
  };

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function api() { return window.EU_API || {}; }
  function runtimeReady() { return !!(state.runtime && state.runtime.status === 'ready'); }
  function setShell(shell) {
    state.shell = shell === 'pi' ? 'pi' : 'legacy';
    if (state.conv) state.conv.classList.toggle('pi-active', state.shell === 'pi');
    render();
  }
  function rememberSession(id) {
    try {
      if (id) localStorage.setItem('easyicu_pi_copilot_session', id);
      else localStorage.removeItem('easyicu_pi_copilot_session');
    } catch (e) {}
  }
  function rememberedSession() {
    try { return localStorage.getItem('easyicu_pi_copilot_session') || ''; } catch (e) { return ''; }
  }
  function errorText(error) {
    if (!error) return '';
    if (error.code === 'pi_session_authority_stale') {
      return tr('The study binding changed after this conversation was saved. Rebind it before continuing.', '这段对话保存后研究绑定发生了变化，请先重新绑定再继续。');
    }
    if (error.code === 'pi_provider_auth_failed') {
      return tr('The model service rejected this API credential.', '模型服务拒绝了这个 API 凭据，请检查后重试。');
    }
    if (error.code === 'pi_provider_model_unavailable') {
      return tr('The selected model was not reported by this service.', '该服务没有返回所选模型，请检查模型名称。');
    }
    if (error.code === 'pi_provider_connection_failed') {
      return tr('EasyICU could not reach the model service.', 'EasyICU 无法连接到模型服务，请检查地址和服务状态。');
    }
    return String(error.message || error.code || error);
  }
  function sessionIsStale() {
    return !!(state.session && state.session.stale && state.session.stale.stale);
  }

  function transcriptMessages(session) {
    const rows = Array.isArray(session && session.transcript) ? session.transcript : [];
    return rows.map((row, index) => {
      const parts = Array.isArray(row.content) ? row.content : [];
      const text = parts.filter(p => p && p.type === 'text').map(p => p.text || '').join('');
      const tool = parts.find(p => p && p.type === 'tool_call');
      if (tool) return { id: 'history-tool-' + index, role: 'tool', toolName: tool.tool_name, status: 'complete', text: '' };
      return { id: 'history-' + index, role: row.role || 'assistant', text };
    }).filter(row => row.text || row.toolName);
  }

  function statusBanner() {
    if (state.loading) {
      return `<div class="gpi-inline"><span class="gpi-dot waiting"></span>${tr('Checking Pi Copilot…', '正在检查 Pi Copilot…')}</div>`;
    }
    if (!runtimeReady()) {
      const blockers = (state.runtime && state.runtime.blockers) || [];
      const reason = blockers.includes('api_key_configured')
        ? tr('Connect and verify your model service before entering Pi Copilot.', '请先连接并验证模型服务，再进入 Pi Copilot。')
        : blockers.includes('provider_connection_unverified')
          ? tr('Verify the saved model service before entering Pi Copilot.', '请先验证已保存的模型服务，再进入 Pi Copilot。')
        : blockers.includes('easyicu_ai_opt_in_disabled')
          ? tr('Confirm external AI use before entering Pi Copilot.', '请先确认允许使用外部 AI，再进入 Pi Copilot。')
          : tr('Pi is not ready on this machine. The local Guided workflow remains available.', '这台电脑上的 Pi 尚未就绪，仍可使用本地研究引导流程。');
      return `<div class="gpi-inline unavailable"><span class="gpi-dot"></span><span>${esc(reason)}</span><button class="gpi-link" type="button" data-gpi-setup>${tr('Set up', '开始配置')}</button></div>`;
    }
    if (state.shell === 'legacy') {
      return `<div class="gpi-inline ready"><span class="gpi-dot"></span><span>${tr('Pi AgentSession is ready with EasyICU-only tools.', 'Pi AgentSession 已就绪，仅开放 EasyICU 工具。')}</span><button class="gpi-link" type="button" data-gpi-open>${tr('Open Pi shell', '打开 Pi 交互壳')}</button></div>`;
    }
    return '';
  }

  function setupPanel() {
    const runtime = state.runtime || {};
    const config = runtime.configuration || {};
    const blockers = runtime.blockers || [];
    const runtimeMissing = blockers.filter(code => [
      'node_available', 'node_version_supported', 'entrypoint_available',
      'dependency_installed', 'lockfile_present', 'base_url_configured',
    ].includes(code));
    const savedCredential = !!config.credential_present;
    const canCancel = runtimeReady();
    return `
      <div class="gpi-setup-wrap">
        <form class="gpi-setup" data-gpi-provider-form autocomplete="off">
          <div class="gpi-kicker">PI COPILOT · FIRST-USE SETUP</div>
          <h2>${tr('Connect your model service', '连接你的模型服务')}</h2>
          <p>${tr('Like signing in to Codex or Claude Code, this one-time check must succeed before the conversation opens. The API credential is saved only in EasyICU’s private local credential file and is never returned to this page.', '就像登录 Codex 或 Claude Code 一样，只有这次连接验证成功后才会开放对话。API 凭据只保存在 EasyICU 本机私有凭据文件中，不会回传到页面。')}</p>
          <div class="gpi-setup-grid">
            <label><span>${tr('Service name', '服务名称')}</span><input name="provider" maxlength="80" value="${esc(config.provider || runtime.provider || 'easyicu-local')}" required></label>
            <label><span>${tr('API transport', 'API 协议')}</span><select name="api_transport"><option value="openai-completions" ${(config.api_transport || runtime.api_transport) === 'openai-responses' ? '' : 'selected'}>OpenAI Chat Completions</option><option value="openai-responses" ${(config.api_transport || runtime.api_transport) === 'openai-responses' ? 'selected' : ''}>OpenAI Responses</option></select></label>
            <label class="wide"><span>${tr('Service address', '服务地址')}</span><input name="base_url" maxlength="2048" value="${esc(config.base_url || 'http://127.0.0.1:8317/v1')}" inputmode="url" spellcheck="false" required></label>
            <label><span>${tr('Model', '模型')}</span><input name="model" maxlength="256" value="${esc(config.model || runtime.model || 'gpt5.6 luna')}" spellcheck="false" required></label>
            <label><span>${tr('API credential', 'API 凭据')}</span><input name="api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="${savedCredential ? tr('Re-enter to verify or replace', '重新输入以验证或更换') : tr('Paste once; it will not be shown again', '仅粘贴一次，之后不再显示')}" required></label>
          </div>
          ${savedCredential ? `<div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('A private credential is saved, but a newly entered credential is still required to verify or change this connection.', '本机已有私有凭据；为验证或更换连接，仍需重新输入一次凭据。')}</div>` : ''}
          ${runtimeMissing.length ? `<div class="gpi-config-note warn">${tr('The Pi runtime also needs attention before chat can open:', '聊天开放前还需要处理 Pi 运行环境：')} ${esc(runtimeMissing.join(', '))}</div>` : ''}
          <label class="gpi-optin"><input name="enable_ai" type="checkbox" required> <span>${tr('I authorize this verification request and external AI use for Pi Copilot. Chat text and PHI-safe summaries may be sent to this service.', '我授权本次连接验证，并允许 Pi Copilot 使用外部 AI；对话文字和经 PHI 安全投影的摘要可能发送到该服务。')}</span></label>
          ${state.error ? `<div class="gpi-error inline">${esc(state.error)}</div>` : ''}
          <div class="gpi-setup-actions">
            ${canCancel ? `<button class="btn" type="button" data-gpi-cancel-setup>${tr('Back to conversation', '返回对话')}</button>` : `<button class="gpi-link" type="button" data-gpi-legacy>${tr('Use local Guided workflow', '使用本地研究引导流程')}</button>`}
            <button class="btn primary" type="submit" ${state.setupSaving ? 'disabled' : ''}>${state.setupSaving ? tr('Verifying…', '正在验证…') : tr('Verify and enter Copilot', '验证并进入 Copilot')}</button>
          </div>
          <div class="gpi-consent">${tr('Verification calls only the service model-list endpoint. The credential is never written to project files, browser storage, logs, or Pi session history.', '验证仅调用服务的模型列表接口。凭据不会写入项目文件、浏览器存储、日志或 Pi 会话历史。')}</div>
        </form>
      </div>`;
  }

  function activatePanel() {
    const saved = state.sessions.map(row => `
      <button class="gpi-session-row" type="button" data-gpi-session="${esc(row.session_id)}">
        <span><strong>${esc(row.title || 'Pi Copilot')}</strong><small>${esc(row.updated_at || '')}</small></span>
        <span>${esc((row.binding && row.binding.study_context_id) || tr('Unbound', '未绑定'))}</span>
      </button>`).join('');
    return `
      <div class="gpi-activate">
        <div class="gpi-kicker">PI AGENTSESSION · EASYICU GATEWAY</div>
        <h2>${tr('Start a governed research conversation', '开始受治理的科研对话')}</h2>
        <p>${tr('Pi handles conversation and tool turns. EasyICU still owns study setup, runs, validation, and evidence. Patient rows and generic coding tools are blocked.', 'Pi 负责对话与工具循环；研究配置、运行、验证和证据仍由 EasyICU 管理。患者行级数据和通用编程工具均被阻止。')}</p>
        <button class="btn primary" type="button" data-gpi-create ${state.creating ? 'disabled' : ''}>
          ${state.creating ? tr('Starting…', '正在启动…') : tr('I agree — activate Pi Copilot', '我同意——启用 Pi Copilot')}
        </button>
        <div class="gpi-consent">${tr('This sends only your chat text and PHI-safe EasyICU summaries to the configured shell model. It does not authorize a scientific provider run.', '仅将你的对话文字和经 PHI 安全投影的 EasyICU 摘要发送给已配置的交互模型；这不代表授权科研模型运行。')}</div>
        ${saved ? `<div class="gpi-saved"><div class="gpi-section-title">${tr('Saved Pi sessions', '已保存的 Pi 会话')}</div>${saved}</div>` : ''}
        <button class="gpi-link" type="button" data-gpi-legacy>${tr('Use the local Guided workflow', '使用本地研究引导流程')}</button>
      </div>`;
  }

  function messageHtml(row) {
    if (row.role === 'tool') {
      const status = row.status || 'running';
      return `<details class="gpi-tool ${status === 'error' ? 'error' : ''}" ${status === 'running' ? 'open' : ''}>
        <summary><span class="gpi-tool-mark"></span><strong>${esc(row.toolName || 'EasyICU tool')}</strong><span>${status === 'running' ? tr('running', '运行中') : tr('complete', '已完成')}</span></summary>
        ${row.text ? `<pre>${esc(row.text)}</pre>` : ''}
      </details>`;
    }
    const cls = row.role === 'user' ? 'user' : 'assistant';
    return `<div class="gpi-message ${cls}">
      <div class="gpi-avatar">${cls === 'user' ? tr('You', '你') : 'Pi'}</div>
      <div class="gpi-message-body">
        ${row.text ? `<div class="gpi-text">${esc(row.text)}</div>` : `<div class="gpi-streaming"><i></i><i></i><i></i></div>`}
      </div>
    </div>`;
  }

  function sessionPanel() {
    const session = state.session || {};
    const model = session.model || {};
    const messages = state.messages.map(messageHtml).join('');
    const stale = sessionIsStale();
    return `
      <div class="gpi-panel">
        <header class="gpi-head">
          <div><div class="gpi-kicker">PI AGENTSESSION · UX STATE ONLY</div><div class="gpi-title">${esc(session.title || 'Pi Copilot')} <span class="gpi-live">${state.busy ? tr('streaming', '生成中') : tr('ready', '就绪')}</span></div></div>
          <div class="gpi-head-meta">
            <span>${esc(model.id || (state.runtime && state.runtime.model) || 'model')}</span>
            <button class="gpi-link" type="button" data-gpi-config>${tr('Model service', '模型服务')}</button>
            <button class="gpi-link" type="button" data-gpi-new>${tr('New', '新会话')}</button>
            <button class="gpi-link" type="button" data-gpi-legacy>${tr('Study setup', '研究配置')}</button>
          </div>
        </header>
        ${stale ? `<div class="gpi-stale"><strong>${tr('Authority changed', '权威状态已变化')}</strong><span>${tr('The EasyICU study binding, revision, or active run changed. Rebind before continuing.', 'EasyICU 研究绑定、版本或活动运行已变化，请先重新绑定。')}</span><button class="btn sm" type="button" data-gpi-rebind>${tr('Rebind current state', '重新绑定当前状态')}</button></div>` : ''}
        <div class="gpi-log" data-gpi-log>
          ${messages || `<div class="gpi-empty"><strong>${tr('Ask about the current study', '询问当前研究')}</strong><span>${tr('Pi can inspect context, plans, runs, validation, artefacts, evidence, and blockers through bounded EasyICU tools.', 'Pi 可通过受限的 EasyICU 工具检查上下文、计划、运行、验证、产物、证据和阻断原因。')}</span></div>`}
        </div>
        ${state.error ? `<div class="gpi-error">${esc(state.error)}</div>` : ''}
        <div class="gpi-compose">
          <textarea data-gpi-input rows="3" maxlength="12000" placeholder="${tr('Ask Pi about this EasyICU study — do not paste patient rows or identifiers.', '向 Pi 询问这个 EasyICU 研究——请勿粘贴患者行级数据或标识符。')}" ${state.busy || stale ? 'disabled' : ''}>${esc(state.draft)}</textarea>
          <div class="gpi-actions">
            <div class="gpi-grants" title="${tr('Actions are granted only for this message.', '操作权限仅对本条消息有效。')}">
              <label><input type="checkbox" data-gpi-grant="configure" ${state.busy ? 'disabled' : ''}> ${tr('Allow one setup update', '允许一次配置更新')}</label>
              <label><input type="checkbox" data-gpi-grant="run" ${state.busy ? 'disabled' : ''}> ${tr('Allow one local preflight', '允许一次本地预检')}</label>
              <label><input type="checkbox" data-gpi-grant="cancel" ${state.busy ? 'disabled' : ''}> ${tr('Allow job cancel', '允许取消任务')}</label>
            </div>
            ${state.busy ? `<button class="btn danger" type="button" data-gpi-stop>${tr('Stop', '停止')}</button>` : `<button class="btn primary" type="button" data-gpi-send ${stale ? 'disabled' : ''}>${tr('Send', '发送')}</button>`}
          </div>
          <div class="gpi-foot">${tr('Pi session = conversation history. EasyICU = scientific authority and evidence.', 'Pi 会话 = 对话历史；EasyICU = 科学权威与证据。')}</div>
        </div>
      </div>`;
  }

  function render() {
    if (!state.host) return;
    state.host.hidden = false;
    state.host.innerHTML = state.shell === 'legacy'
      ? statusBanner()
      : ((state.showSetup || !runtimeReady()) ? setupPanel() : (state.session ? sessionPanel() : activatePanel()));
    requestAnimationFrame(() => {
      const log = state.host && state.host.querySelector('[data-gpi-log]');
      if (log) log.scrollTop = log.scrollHeight;
    });
  }

  async function loadStatus() {
    state.loading = true; state.error = ''; render();
    try {
      const payload = await api().loadPiCopilotStatus();
      state.runtime = payload && payload.runtime;
      if (runtimeReady()) {
        const listed = await api().loadPiCopilotSessions(30);
        state.sessions = (listed && listed.sessions) || [];
        const remembered = rememberedSession();
        if (remembered && state.sessions.some(row => row.session_id === remembered)) {
          await openSession(remembered); return;
        }
      } else {
        state.showSetup = true;
      }
    } catch (error) {
      state.runtime = { status: 'unavailable', blockers: ['status_request_failed'] };
      state.error = errorText(error);
    } finally {
      state.loading = false; render();
    }
  }

  async function configureProvider(form) {
    if (state.setupSaving || !form) return;
    const data = new FormData(form);
    const apiKey = String(data.get('api_key') || '').trim();
    const keyInput = form.querySelector('[name="api_key"]');
    if (keyInput) keyInput.value = '';
    state.setupSaving = true; state.error = '';
    const submit = form.querySelector('[type="submit"]');
    if (submit) { submit.disabled = true; submit.textContent = tr('Verifying…', '正在验证…'); }
    try {
      const payload = await api().savePiCopilotProviderConfig({
        provider: String(data.get('provider') || '').trim(),
        api_key: apiKey,
        base_url: String(data.get('base_url') || '').trim(),
        model: String(data.get('model') || '').trim(),
        api_transport: String(data.get('api_transport') || 'openai-completions'),
        enable_ai: data.get('enable_ai') === 'on',
      });
      state.runtime = payload && payload.runtime;
      if (!runtimeReady()) {
        state.error = tr('The model connection was saved, but the Pi runtime is not ready yet.', '模型连接已保存，但 Pi 运行环境尚未就绪。');
        return;
      }
      state.showSetup = false;
      await createSession();
    } catch (error) {
      state.error = errorText(error);
    } finally {
      state.setupSaving = false; render();
    }
  }

  async function createSession() {
    if (state.creating) return;
    state.creating = true; state.error = ''; render();
    try {
      const payload = await api().createPiCopilotSession({
        title: tr('EasyICU research session', 'EasyICU 科研会话'),
        language: window.EU_LANG === 'zh' ? 'zh' : 'en',
        thinking_level: 'off', external_llm_opt_in: true,
      });
      state.session = payload.session; state.messages = transcriptMessages(state.session);
      rememberSession(state.session.session_id);
      state.sessions = [state.session].concat(state.sessions.filter(row => row.session_id !== state.session.session_id));
    } catch (error) { state.error = errorText(error); }
    finally { state.creating = false; render(); }
  }

  async function openSession(sessionId) {
    state.error = '';
    try {
      const payload = await api().loadPiCopilotSession(sessionId);
      state.session = payload.session; state.messages = transcriptMessages(state.session);
      rememberSession(sessionId); setShell('pi');
    } catch (error) { rememberSession(''); state.error = errorText(error); render(); }
  }

  function assistantRow() {
    let row = state.messages[state.messages.length - 1];
    if (!row || row.role !== 'assistant' || row.complete) {
      row = { id: 'live-' + Date.now(), role: 'assistant', text: '', complete: false };
      state.messages.push(row);
    }
    return row;
  }
  function handlePiEvent(event) {
    if (!event || typeof event !== 'object') return;
    if (event.type === 'text_delta') assistantRow().text += String(event.delta || '');
    else if (event.type === 'message_end') assistantRow().complete = true;
    else if (event.type === 'tool_start') state.messages.push({ id: event.tool_call_id, role: 'tool', toolName: event.tool_name, status: 'running', text: '' });
    else if (event.type === 'tool_end') {
      const row = state.messages.find(item => item.role === 'tool' && item.id === event.tool_call_id);
      if (row) { row.status = event.is_error ? 'error' : 'complete'; row.text = event.summary || ''; }
    }
    render();
  }
  function closeSource() { if (state.source) { state.source.close(); state.source = null; } }
  async function refreshSession() {
    if (!state.session) return;
    try {
      const payload = await api().loadPiCopilotSession(state.session.session_id);
      state.session = payload.session; state.messages = transcriptMessages(state.session);
    } catch (e) {}
  }
  function watchJob(jobId) {
    closeSource();
    state.source = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    state.source.onmessage = async event => {
      let row = null; try { row = JSON.parse(event.data); } catch (e) { return; }
      if (row.type === 'pi_event') handlePiEvent(row.event);
      if (row.type === 'end') {
        closeSource(); state.busy = false;
        if (row.status === 'failed') state.error = String(row.error || tr('Pi message failed.', 'Pi 消息失败。'));
        else if (row.status === 'cancelled') state.error = tr('Pi message stopped.', 'Pi 消息已停止。');
        await refreshSession(); render();
      }
    };
    state.source.onerror = () => { if (!state.busy) closeSource(); };
  }
  async function sendMessage() {
    if (!state.session || state.busy || sessionIsStale()) return;
    const input = state.host.querySelector('[data-gpi-input]');
    const text = String((input && input.value) || state.draft || '').trim();
    if (!text) return;
    const grants = Array.from(state.host.querySelectorAll('[data-gpi-grant]:checked')).map(node => node.dataset.gpiGrant);
    state.messages.push({ id: 'user-' + Date.now(), role: 'user', text, complete: true });
    state.draft = ''; state.busy = true; state.error = ''; render();
    try {
      const payload = await api().sendPiCopilotMessage(state.session.session_id, { message: text, allowed_actions: grants });
      state.jobId = payload.job_id; watchJob(payload.job_id);
    } catch (error) { state.busy = false; state.error = errorText(error); render(); }
  }
  async function stopMessage() {
    if (!state.session || !state.busy) return;
    try { await api().abortPiCopilotSession(state.session.session_id, { message_job_id: state.jobId || null }); }
    catch (error) { state.error = errorText(error); render(); }
  }
  async function rebind() {
    if (!state.session) return;
    try {
      const payload = await api().rebindPiCopilotSession(state.session.session_id);
      state.session = payload.session; state.error = ''; render();
    } catch (error) { state.error = errorText(error); render(); }
  }

  function wire() {
    if (!state.host) return;
    state.host.addEventListener('click', event => {
      const session = event.target.closest('[data-gpi-session]');
      if (session) { openSession(session.dataset.gpiSession); return; }
      if (event.target.closest('[data-gpi-retry]')) { loadStatus(); return; }
      if (event.target.closest('[data-gpi-setup]')) { state.showSetup = true; setShell('pi'); return; }
      if (event.target.closest('[data-gpi-open]')) { setShell('pi'); return; }
      if (event.target.closest('[data-gpi-legacy]')) { setShell('legacy'); return; }
      if (event.target.closest('[data-gpi-create]')) { createSession(); return; }
      if (event.target.closest('[data-gpi-send]')) { sendMessage(); return; }
      if (event.target.closest('[data-gpi-stop]')) { stopMessage(); return; }
      if (event.target.closest('[data-gpi-rebind]')) { rebind(); return; }
      if (event.target.closest('[data-gpi-config]')) { state.showSetup = true; state.error = ''; render(); return; }
      if (event.target.closest('[data-gpi-cancel-setup]')) { state.showSetup = false; state.error = ''; render(); return; }
      if (event.target.closest('[data-gpi-new]')) { state.session = null; state.messages = []; rememberSession(''); render(); }
    });
    state.host.addEventListener('input', event => {
      if (event.target.matches('[data-gpi-input]')) state.draft = event.target.value;
    });
    state.host.addEventListener('keydown', event => {
      if (event.target.matches('[data-gpi-input]') && event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault(); sendMessage();
      }
    });
    state.host.addEventListener('submit', event => {
      const form = event.target.closest('[data-gpi-provider-form]');
      if (!form) return;
      event.preventDefault(); configureProvider(form);
    });
  }
  function mount(host) {
    if (!host || state.host === host) return;
    closeSource(); state.host = host; state.conv = host.closest('.gd-conv'); state.shell = 'pi';
    if (state.conv) state.conv.classList.add('pi-active');
    wire(); loadStatus();
  }
  function unmount() {
    closeSource(); state.host = null; state.conv = null; state.busy = false; state.jobId = '';
  }
  window.EU_GUIDED_PI = { mount, unmount, setShell };
})();
