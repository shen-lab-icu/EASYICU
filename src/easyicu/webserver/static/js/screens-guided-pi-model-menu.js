/* Owner: the saved Guided Copilot model binding and its connection catalogue. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function clean(value, limit) { return String(value || '').trim().slice(0, limit || 256); }

  const state = { key: '', models: [], current: '', loading: false, error: '', request: 0 };
  let context = null;

  // Only the public, saved connection selects an owner. A printed label and
  // the legacy conversation/research pair cannot establish that identity.
  function connectionKind(connection) {
    if (!connection || !clean(connection.model)) return '';
    if (connection.provider === 'codex' && connection.authentication_mode === 'chatgpt_account'
      && connection.credential_source === 'codex_user_auth') return 'account';
    if (connection.provider === 'openai' && connection.authentication_mode === 'api_key'
      && connection.credential_source === 'pi_verified') return 'api';
    return '';
  }
  function bindingKey(session, project) {
    const binding = (session && session.model_connection) || {};
    return JSON.stringify([project, session && session.project_id, session && session.session_id,
      binding.schema_version, binding.provider, binding.authentication_mode, binding.credential_source, binding.model,
      session && session.model && session.model.id]);
  }
  function bind(options) {
    context = options;
    ++state.request;
    state.key = ''; state.models = []; state.loading = false; state.error = '';
  }
  function currentKey() { return context ? bindingKey(context.session(), context.projectId()) : ''; }
  function controlKey(control) {
    const popover = control.querySelector('[data-gpi-model-popover]');
    return popover && popover.dataset ? popover.dataset.gpiModelKey : '';
  }
  function live(control, key, request) {
    return Boolean(context && !context.locked() && control.isConnected !== false
      && key === currentKey() && controlKey(control) === key && request === state.request);
  }

  function rowsHtml() {
    // ResearchProviderBinding is frozen. The existing switch endpoint changes
    // the global API config, not this session, so no catalogue row is a switch.
    const hosted = context && context.hosted();
    const note = hosted
      ? tr('This conversation’s model connection cannot be changed here. To change models, contact the administrator and start a new conversation.', '本对话的模型连接不能在此切换。如需更换，请联系管理员并新建对话。')
      : tr('This conversation’s model connection cannot be changed here. To change models, choose one in connection settings and start a new conversation.', '本对话的模型连接不能在此切换。如需更换，请在连接设置中选择后新建对话。');
    const status = state.loading ? tr('Reading the model list…', '正在读取模型列表…') : state.error;
    const message = `<p class="gpi-model-note">${esc(note)}</p>`;
    if (status) return message + `<p class="gpi-model-note${state.error ? ' is-error' : ''}" role="status">${esc(status)}</p>`;
    if (!state.models.length) return message + `<p class="gpi-model-note">${esc(tr('No model list is available.', '暂无可用模型列表。'))}</p>`;
    // Catalogues describe the currently configured service/account. The
    // public session has no endpoint/account locator; do not claim these are
    // switchable models on that historical connection.
    return message + `<p class="gpi-model-note">${esc(tr('Models currently reported by this service (for new conversations)', '当前服务报告的模型（供新对话选择）'))}</p><ul>${state.models.map(model => `<li>${esc(model)}</li>`).join('')}</ul>`;
  }
  function paint(control) {
    const popover = control.querySelector('[data-gpi-model-popover]');
    if (!popover) return;
    popover.querySelectorAll('.gpi-model-rows').forEach(node => node.remove());
    const list = document.createElement('div');
    list.className = 'gpi-model-rows'; list.innerHTML = rowsHtml();
    const settings = popover.querySelector('.gpi-model-settings');
    if (settings) popover.insertBefore(list, settings); else popover.appendChild(list);
  }

  async function load(control) {
    const key = currentKey();
    if (!context || controlKey(control) !== key || control.isConnected === false) return;
    const session = context.session();
    const project = context.projectId();
    const kind = connectionKind(session && session.model_connection);
    const request = ++state.request;
    state.key = key; state.models = []; state.current = clean(session && session.model_connection && session.model_connection.model);
    state.loading = false; state.error = '';
    if (context.locked()) {
      state.error = tr('Wait for the current task or refresh this conversation before viewing models.', '请等待当前任务结束，或刷新对话后查看模型。');
      paint(control); return;
    }
    if (session && session.model_connection && session.model && session.model.id
      && session.model.id !== session.model_connection.model) {
      state.error = tr('The conversation and analysis models differ. Check the connection before starting a new conversation.', '对话和分析的模型不一致，请核对连接后新建对话。');
      paint(control); return;
    }
    if (!kind || !session.session_id || !project || session.project_id !== project) { paint(control); return; }
    state.loading = true; paint(control);
    try {
      if (typeof api().loadPiCopilotSession !== 'function') throw new Error(tr('The saved model connection is unavailable.', '无法读取已保存的模型连接。'));
      const acknowledged = await api().loadPiCopilotSession(session.session_id, project);
      if (!live(control, key, request)) return;
      if (!acknowledged || bindingKey(acknowledged.session, project) !== key) {
        throw new Error(tr('The connection changed. Reopen this conversation to view its model.', '连接状态已变化，请重新打开对话后查看模型。'));
      }
      state.current = clean(acknowledged.session.model_connection.model);
      const caller = kind === 'account' ? api().loadPiCopilotCodexModels : api().loadPiCopilotApiModels;
      if (typeof caller !== 'function') throw new Error(tr('Model listing is unavailable.', '模型列表不可用。'));
      const payload = await caller();
      if (!live(control, key, request)) return;
      const rows = Array.isArray(payload && payload.models) ? payload.models : [];
      state.models = [...new Set(rows.map(row => clean(typeof row === 'string' ? row : row && row.id)).filter(Boolean))];
    } catch (error) {
      if (!live(control, key, request)) return;
      state.error = friendlyError(error);
    } finally {
      if (live(control, key, request)) { state.loading = false; paint(control); }
      else if (request === state.request) { state.loading = false; state.models = []; }
    }
  }
  function friendlyError(error) {
    const status = Number(error && error.status);
    const message = clean(error && error.message, 200);
    const rawHttp = !message || /^(not found|internal server error|bad request|forbidden|unauthorized|service unavailable)$/i.test(message);
    if (Number.isFinite(status) && status > 0 && rawHttp) {
      return status === 404
        ? tr(`Model listing is not available on this server (HTTP ${status}). Restart EasyICU after updating.`, `当前服务未提供模型列表（HTTP ${status}），更新后请重启 EasyICU。`)
        : tr(`Could not read the model list (HTTP ${status}).`, `读取模型列表失败（HTTP ${status}）。`);
    }
    return clean(message || tr('Could not read the model list.', '读取模型列表失败。'), 240);
  }
  function onToggle(event) {
    const control = event.target.closest ? event.target.closest('.gpi-model-control') : null;
    if (control && control.open) void load(control);
  }
  function mount() {
    if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') document.addEventListener('toggle', onToggle, true);
  }
  mount();
  window.EasyICU.guidedPi.declare('modelMenu', { bind, bindingKey, load, state });
})();
