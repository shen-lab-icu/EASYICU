/* Owner: Guided Copilot effort (thinking level) menu on the composer.
   One conversation carries one effort level: the Pi runtime's thinking
   level, clamped to what the selected model supports. New conversations
   start at the remembered choice (default medium); changing it mid-
   conversation goes through the host route so the session record and the
   bridge agree. Whatever the level, the trace shows only the bounded,
   sanitized reasoning summary the bridge projects. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  const STORAGE_KEY = 'easyicu.pi.thinkingLevel.v1';
  const DEFAULT_LEVEL = 'medium';
  // The three levels every OpenAI-style reasoning endpoint accepts as an
  // explicit `reasoning_effort`. "off" means the request carries no effort
  // parameter at all (the model service applies its own default) and
  // "minimal" is the least a model allows; the host reads conversations from
  // before the menu at the default, so either appears only when the model
  // clamped a request. Both are shown as the current state, not offered.
  const LEVELS = ['low', 'medium', 'high'];
  const ACCEPTED = ['off', 'minimal', 'low', 'medium', 'high'];

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }

  function normalize(value) {
    const text = String(value || '').trim().toLowerCase();
    return ACCEPTED.includes(text) ? text : DEFAULT_LEVEL;
  }

  function preferred() {
    try { return normalize(window.localStorage.getItem(STORAGE_KEY) || DEFAULT_LEVEL); } catch (_) { return DEFAULT_LEVEL; }
  }

  function remember(level) {
    try { window.localStorage.setItem(STORAGE_KEY, normalize(level)); } catch (_) {}
  }

  function label(level) {
    return {
      off: tr('Unspecified', '未指定'), minimal: tr('Minimal', '最低'), low: tr('Low', '低'),
      medium: tr('Medium', '中'), high: tr('High', '高'),
    }[normalize(level)];
  }

  function description(level) {
    return {
      off: tr('No effort level is sent; the model service applies its own default (where a model without an effort setting clamps to).', '不向模型指定推理档位，由模型服务按自身默认处理（所选模型不接受推理档位时会收敛到此状态）。'),
      minimal: tr('The least reasoning this model allows; reached only when the model clamped a request.', '该模型允许的最少推理；仅在模型收敛请求时出现。'),
      low: tr('Brief reasoning, faster replies; enough for status checks and rewording.', '简短推理，回复更快；查看状态、改措辞时够用。'),
      medium: tr('Default. Routine reasoning for planning research steps and explaining results.', '默认。规划研究步骤、解释结果的常规推理。'),
      high: tr('Deep reasoning for plan audits and hard questions; each reply is slower and uses more tokens.', '深入推理，审计方案、疑难问题时用；每轮更慢、消耗更多 token。'),
    }[normalize(level)];
  }

  function render(options) {
    const { iconHtml, disabled } = options;
    const current = normalize(options.level);
    const rows = LEVELS.includes(current) ? LEVELS : LEVELS.concat(current);
    return `<details class="gpi-effort-menu" data-gpi-effort-menu data-popover-menu>
      <summary aria-label="${esc(`${tr('Effort', '努力程度')}：${label(current)}`)}" title="${esc(description(current))}" ${disabled ? 'aria-disabled="true"' : ''}>${iconHtml('spark', 14)}<span>${esc(tr('Effort', '努力'))} · ${esc(label(current))}</span><span class="gpi-access-chevron" aria-hidden="true">${iconHtml('chevron', 13)}</span></summary>
      <div class="gpi-access-popover gpi-effort-popover" role="group" aria-label="${esc(tr('Effort level for this conversation', '本对话的努力程度'))}">
        ${rows.map(key => `<button type="button" data-gpi-effort-level="${key}" aria-pressed="${current === key}" ${disabled ? 'disabled' : ''}><span><strong>${esc(label(key))}</strong><small>${esc(description(key))}</small></span>${current === key ? iconHtml('check', 15) : ''}</button>`).join('')}
      </div>
    </details>`;
  }

  // Delegated click: returns true when the event belonged to this owner.
  function handleClick(event, callbacks) {
    const choice = event.target.closest && event.target.closest('[data-gpi-effort-level]');
    if (!choice) return false;
    event.preventDefault();
    const menu = choice.closest('[data-gpi-effort-menu]');
    if (menu) menu.removeAttribute('open');
    const level = normalize(choice.dataset.gpiEffortLevel);
    const session = callbacks.session();
    if (!session || callbacks.busy()) return true;
    if (normalize(session.thinking_level) === level) { remember(level); return true; }
    void apply(level, session, callbacks);
    return true;
  }

  async function apply(level, session, callbacks) {
    const caller = callbacks.api() && callbacks.api().setPiCopilotThinkingLevel;
    if (typeof caller !== 'function') return;
    try {
      const payload = await caller(session.session_id, {
        project_id: callbacks.projectId(), thinking_level: level,
      });
      const current = callbacks.session();
      if (!current || current.session_id !== session.session_id) return;
      const applied = normalize(payload && payload.thinking_level || level);
      // A clamped level (off/minimal) is this model's limit, not a preference.
      remember(LEVELS.includes(applied) ? applied : level);
      callbacks.setSession({ ...current, ...(payload && payload.session ? payload.session : {}), thinking_level: applied });
      callbacks.render(true);
    } catch (error) {
      callbacks.setError(String((error && error.message) || tr('The effort level could not be changed.', '努力程度未能修改。')).slice(0, 240));
      callbacks.render(true);
    }
  }

  window.EasyICU.guidedPi.declare('effortMenu', {
    render, handleClick, preferred, remember, normalize, label, description, LEVELS,
  });
})();
