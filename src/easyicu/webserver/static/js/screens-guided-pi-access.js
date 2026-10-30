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
        if (step && step.kind === 'tool' && step.code === 'external_llm_opt_in_required'
          && !actions.includes('provider_plan')) actions.push('provider_plan');
        if (step && step.kind === 'tool' && step.code === BLOCKED_CODE && step.grantable === true
          && actionLabel(action) && !actions.includes(action)) actions.push(action);
      });
    });
    return actions;
  }

  function render(actions, options) {
    if (!actions.length) return '';
    const ordinary = actions.filter(action => action !== 'provider_plan');
    const provider = actions.includes('provider_plan');
    const disabled = options.disabled ? ' disabled' : '';
    const list = ordinary.map(actionLabel).join(tr(', ', '、'));
    const routine = grants('assist', options.agentMode);
    const autoAllows = ordinary.length && normalize(options.mode) === 'ask' && ordinary.every(action => routine.includes(action));
    return `<div class="gpi-action-ask" role="group" aria-label="${esc(tr('Needs your go-ahead', '需要你同意'))}">
      ${ordinary.length ? `<p>${esc(tr(`The assistant wants to ${list}.`, `助手要${list}。`))}</p>` : ''}
      ${provider ? `<p>${esc(tr('Review the current study configuration before authorizing one candidate plan. The plan still requires review.', '请先核对当前研究配置，再授权生成一次候选方案。方案仍需审阅。'))}</p>` : ''}
      <div class="gpi-action-ask-buttons">
        ${provider ? `<button type="button" class="btn sm primary" data-gpi-review-plan-authorization${disabled}>${esc(tr('Review and authorize one candidate plan', '核对并授权一次候选方案生成'))}</button>` : ''}
        ${ordinary.length ? `<button type="button" class="btn sm primary" data-gpi-action-ask-allow="${esc(ordinary.join(' '))}"${disabled}>${esc(tr('Allow this once', '允许这一次'))}</button>` : ''}
        ${autoAllows ? `<button type="button" class="btn sm" data-gpi-action-ask-routine="${esc(ordinary.join(' '))}"${disabled}>${esc(tr('Confirm routine actions automatically', '常规操作自动确认'))}</button>` : ''}
      </div>
    </div>`;
  }

  async function authorizePlan(callbacks) {
    const before = callbacks.planContext();
    const decision = before && before.decision;
    const supported = ['provider_ready_to_generate_plan', 'failed_pipeline_requires_fresh_plan'];
    if (!before || !before.projectId || !before.sessionId || !before.studyId || !decision
      || !supported.includes(before.code) || decision.next_action_code !== before.code) {
      callbacks.planError(tr('Complete the study configuration before authorizing a candidate plan.', '请先完成研究配置，再授权生成候选方案。'));
      return;
    }
    const frozen = JSON.stringify(decision);
    try {
      const response = await callbacks.loadStudy(before.studyId);
      const study = response && (response.context || response.study || response);
      const after = callbacks.planContext();
      if (!after || after.projectId !== before.projectId || after.sessionId !== before.sessionId
        || after.studyId !== before.studyId || after.userMessage !== before.userMessage
        || JSON.stringify(after.decision) !== frozen) {
        callbacks.planError(tr('The study configuration changed. Review it again.', '研究配置已变化，请重新核对。'));
        return;
      }
      const normalize = text => String(text || '').trim().replace(/\s+/g, ' ');
      const question = normalize(study && study.question);
      if (!question || question !== normalize(before.userMessage)) {
        callbacks.planError(tr('The current study question does not exactly match your latest request. Update the study configuration first.', '当前研究问题与本次请求不完全一致，请先更新研究配置。'));
        return;
      }
      const database = String((study.data_source || {}).database || (study.data_source || {}).db || '');
      if (!callbacks.confirmPlan(tr(
        `Authorize one candidate plan using the configured question below? The model may receive the study metadata and search literature. The plan still requires review; this does not approve analysis.\n\n${question}\n\nDatabase: ${database}`,
        `授权根据以下已配置问题生成一次候选方案？模型可能接收研究元数据并检索文献；方案仍需审阅，此操作不批准分析。\n\n${question}\n\n数据库：${database}`,
      ))) return;
      const current = callbacks.planContext();
      if (!current || current.projectId !== before.projectId || current.sessionId !== before.sessionId
        || current.studyId !== before.studyId || current.userMessage !== before.userMessage
        || JSON.stringify(current.decision) !== frozen) {
        callbacks.planError(tr('The study configuration changed. Review it again.', '研究配置已变化，请重新核对。'));
        return;
      }
      await callbacks.startPlan(before.code, { requireConfirmedDecision: true, confirmedDecision: frozen });
    } catch (_) {
      callbacks.planError(tr('Could not verify the study configuration. No plan was started.', '无法核对研究配置，未启动方案生成。'));
    }
  }

  // Delegated click: returns true when the event belonged to this owner.
  function handleClick(event, callbacks) {
    const plan = event.target.closest && event.target.closest('[data-gpi-review-plan-authorization]');
    if (plan) {
      event.preventDefault();
      if (!callbacks.busy()) void authorizePlan(callbacks);
      return true;
    }
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
    MODES, normalize, grants, label, description, preferred, remember, actionLabel, pending, render, authorizePlan, handleClick,
  });
})();
