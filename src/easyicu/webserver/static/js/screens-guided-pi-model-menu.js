/* Owner: the saved Guided Copilot model binding and its connection catalogue. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function clean(value, limit) { return String(value || '').trim().slice(0, limit || 256); }

  const state = { key: '', models: [], current: '', loading: false, switching: false, error: '', request: 0 };
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
    state.key = ''; state.models = []; state.loading = false; state.switching = false; state.error = '';
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
    const status = state.loading ? tr('Reading models…', '正在读取模型…') : state.error;
    const message = status ? `<p class="gpi-model-note${state.error ? ' is-error' : ''}" role="status">${esc(status)}</p>` : '';
    if (!state.models.length) return message || `<p class="gpi-model-note">${esc(tr('No models available.', '暂无可用模型。'))}</p>`;
    return message + state.models.map(model => `<button type="button" data-gpi-pick-model="${esc(model)}" role="menuitemradio" aria-checked="${model === state.current}" ${state.switching ? 'disabled' : ''}><span>${esc(model)}</span><span aria-hidden="true">${model === state.current ? '✓' : ''}</span></button>`).join('');
  }
  function paint(control) {
    const popover = control.querySelector('[data-gpi-model-popover]');
    if (!popover) return;
    popover.querySelectorAll('.gpi-model-rows').forEach(node => node.remove());
    const list = document.createElement('div');
    list.className = 'gpi-model-rows'; list.role = 'menu'; list.innerHTML = rowsHtml();
    const settings = popover.querySelector('.gpi-model-settings');
    if (settings) popover.insertBefore(list, settings); else popover.appendChild(list);
  }

  async function load(control) {
    if (state.switching) return;
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
      const caller = api().loadPiCopilotConversationModels;
      if (typeof caller !== 'function') throw new Error(tr('Model listing is unavailable.', '模型列表不可用。'));
      const payload = await caller(session.session_id, project);
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
    const messages = {
      pi_session_busy: ['Wait for the current reply to finish.', '请等待当前回复结束。'],
      pi_session_model_unavailable: ['This model is not available on the saved connection.', '已保存的连接中没有此模型。'],
      pi_session_project_mismatch: ['This conversation belongs to another project.', '本对话属于另一个项目。'],
      provider_not_cleared_for_credentialed_data: ['This model’s declared data handling does not permit this data source.', '此模型声明的数据处理方式不允许使用当前数据源。'],
      pi_session_connection_changed: ['The saved connection changed. Reopen this conversation.', '已保存的连接已变化，请重新打开对话。'],
      pi_session_connection_unpinned: ['This conversation predates model switching. Start a new conversation to use another model.', '这条对话创建于旧版本，不能切换模型。新建对话即可选择其他模型。'],
    };
    const messagePair = messages[error && error.code];
    if (messagePair) return tr(...messagePair);
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
  async function select(control, model) {
    const key = currentKey();
    if (state.loading || state.switching || state.key !== key || !state.models.includes(model)
      || !live(control, key, state.request)) return;
    const session = context.session(), project = context.projectId();
    const request = ++state.request;
    state.switching = true; state.error = ''; paint(control);
    try {
      if (typeof api().setPiCopilotConversationModel !== 'function') throw new Error(tr('Model switching is unavailable.', '模型切换不可用。'));
      const payload = await api().setPiCopilotConversationModel(session.session_id, { project_id: project, model });
      if (!live(control, key, request)) return;
      const updated = payload && payload.session;
      if (!updated || updated.session_id !== session.session_id || updated.project_id !== project
        || updated.model_connection?.provider !== session.model_connection.provider
        || updated.model_connection?.authentication_mode !== session.model_connection.authentication_mode
        || updated.model_connection?.credential_source !== session.model_connection.credential_source
        || updated.model_connection?.model !== model || updated.model?.id !== model) {
        throw new Error(tr('The model change could not be confirmed. Reopen this conversation.', '无法确认模型切换，请重新打开对话。'));
      }
      context.applySession(updated);
      state.current = model; state.key = currentKey();
      control.querySelector('[data-gpi-model-popover]').dataset.gpiModelKey = state.key;
      const label = control.querySelector('.gpi-model-binding > span');
      if (label) label.textContent = label.title = [updated.model_connection.provider, model].join(' · ');
      control.open = false;
    } catch (error) {
      if (live(control, key, request)) { state.error = friendlyError(error); paint(control); }
    } finally {
      if (request === state.request) {
        state.switching = false;
        if (control.isConnected !== false && controlKey(control) === currentKey()) paint(control);
      }
    }
  }
  function onClick(event) {
    const choice = event.target.closest ? event.target.closest('[data-gpi-pick-model]') : null;
    if (!choice) return;
    const control = choice.closest('.gpi-model-control');
    if (control) { event.preventDefault(); void select(control, choice.dataset.gpiPickModel); }
  }
  function onToggle(event) {
    const control = event.target.closest ? event.target.closest('.gpi-model-control') : null;
    if (control && control.open) void load(control);
  }
  function mount() {
    if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') { document.addEventListener('toggle', onToggle, true); document.addEventListener('click', onClick); }
  }
  mount();
  window.EasyICU.guidedPi.declare('modelMenu', { bind, bindingKey, load, select, state });
})();
