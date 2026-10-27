/* Owner: what the Copilot may do without asking, and the question it asks
   when a step needs the researcher's go-ahead. The access menu picks a mode
   (remembered in this browser); each message sends that mode's one-turn
   grants and the host blocks any tool call outside them
   (pi_copilot/tools.py _consume_action). A blocked call that a turn grant
   can allow comes back with its action, and the composer asks: allowing it
   sends a follow-up carrying exactly that grant. Data extraction and full
   analyses are never allowed this way; the host accepts them only from the
   researcher's own words or a host control. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;
  const STORAGE_KEY = 'easyicu.pi.accessMode.v1';
  const DEFAULT_MODE = 'assist';
  const BLOCKED_CODE = 'pi_action_authorization_required';
  // One-turn grants per mode. Stopping a run is never routine: the assistant
  // asks first, and the researcher can always stop it from the run itself.
  const GRANTS = Object.freeze({
    ask: Object.freeze([]),
    assist: Object.freeze(['idea', 'literature', 'configure', 'run', 'workspace_write', 'mcp_read']),
  });
  const MODES = Object.keys(GRANTS);

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  const normalize = value => (MODES.includes(String(value || '')) ? String(value) : DEFAULT_MODE);
  // Writing project files belongs to the workspace agent only.
  const grants = (mode, agentMode) => GRANTS[normalize(mode)]
    .filter(action => action !== 'workspace_write' || agentMode === 'workspace');
  function label(mode) {
    return normalize(mode) === 'ask' ? tr('Ask me first', '每步先问我') : tr('Confirm routine actions automatically', '常规操作自动确认');
  }
  function description(mode) {
    return normalize(mode) === 'ask'
      ? tr('The assistant asks before it changes the setup, searches the literature or runs a check.', '助手修改配置、检索文献或运行预检前，先问你。')
      : tr('Setup, searches and the preflight check go ahead; data extraction and full analyses still need your explicit go-ahead.', '配置、检索和预检自动进行；数据提取和完整分析仍需你明确同意。');
  }
  function preferred() {
    try { return normalize(window.localStorage.getItem(STORAGE_KEY)); } catch (_) { return DEFAULT_MODE; }
  }
  function remember(mode) {
    try { window.localStorage.setItem(STORAGE_KEY, normalize(mode)); } catch (_) {}
  }

  function actionLabel(action) {
    return {
      configure: tr('change the study setup', '修改研究配置'),
      idea: tr('work on research ideas', '整理研究想法'),
      literature: tr('search the literature', '检索文献'),
      run: tr('run the preflight check', '运行预检'),
      cancel: tr('stop the running job', '停止正在进行的运行'),
      mcp_read: tr('read from an external tool', '读取外部工具'),
      workspace_write: tr('write project files', '写入项目文件'),
    }[action] || '';
  }

  // Actions the host blocked in the turn after the researcher's last message,
  // when a turn grant can allow them.
  function pending(messages) {
    const rows = Array.isArray(messages) ? messages : [];
    let start = 0;
    rows.forEach((row, index) => { if (row && row.role === 'user') start = index + 1; });
    const actions = [];
    rows.slice(start).forEach(row => {
      if (!row || row.role !== 'activity' || !Array.isArray(row.steps)) return;
      row.steps.forEach(step => {
        const action = String(step && step.requiredAction || '');
        if (step && step.kind === 'tool' && step.code === BLOCKED_CODE && step.grantable === true
          && actionLabel(action) && !actions.includes(action)) actions.push(action);
      });
    });
    return actions;
  }

  function render(actions, options) {
    if (!actions.length) return '';
    const disabled = options.disabled ? ' disabled' : '';
    const list = actions.map(actionLabel).join(tr(', ', '、'));
    const routine = grants('assist', options.agentMode);
    const autoAllows = normalize(options.mode) === 'ask' && actions.every(action => routine.includes(action));
    return `<div class="gpi-action-ask" role="group" aria-label="${esc(tr('Needs your go-ahead', '需要你同意'))}">
      <p>${esc(tr(`The assistant wants to ${list}.`, `助手要${list}。`))}</p>
      <div class="gpi-action-ask-buttons">
        <button type="button" class="btn sm primary" data-gpi-action-ask-allow="${esc(actions.join(' '))}"${disabled}>${esc(tr('Allow this once', '允许这一次'))}</button>
        ${autoAllows ? `<button type="button" class="btn sm" data-gpi-action-ask-routine="${esc(actions.join(' '))}"${disabled}>${esc(tr('Confirm routine actions automatically', '常规操作自动确认'))}</button>` : ''}
      </div>
    </div>`;
  }

  // Delegated click: returns true when the event belonged to this owner.
  function handleClick(event, callbacks) {
    const allow = event.target.closest && event.target.closest('[data-gpi-action-ask-allow], [data-gpi-action-ask-routine]');
    if (!allow) return false;
    event.preventDefault();
    if (callbacks.busy()) return true;
    const routine = allow.hasAttribute('data-gpi-action-ask-routine');
    const actions = String(allow.dataset.gpiActionAskAllow || allow.dataset.gpiActionAskRoutine || '')
      .split(' ').filter(actionLabel);
    if (!actions.length) return true;
    if (routine) { callbacks.setMode('assist'); remember('assist'); }
    // The follow-up keeps the mode's own grants and adds exactly these.
    callbacks.send(tr('Go ahead with that step.', '同意这一步，请继续。'), [...new Set(callbacks.grants().concat(actions))]);
    return true;
  }

  window.EasyICU.guidedPi.declare('access', {
    MODES, normalize, grants, label, description, preferred, remember, actionLabel, pending, render, handleClick,
  });
})();
