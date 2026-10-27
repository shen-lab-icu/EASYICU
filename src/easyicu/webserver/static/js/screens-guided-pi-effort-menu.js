/* Owner: Guided Copilot effort (thinking level) menu on the composer.
   One conversation carries one effort level: the Pi runtime's thinking
   level, clamped to what the selected model supports. New conversations
   start at the remembered choice (default medium); changing it mid-
   conversation goes through the host route so the session record and the
   bridge agree. Whatever the level, the trace shows only the bounded,
   sanitized reasoning summary the bridge projects. The level is the
   conversation assistant's only: Research Agent planning and review calls
   use their own per-role effort profile. */
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
      minimal: tr('The model selected this level automatically.', '模型自动采用此档位。'),
      low: tr('The conversation assistant reasons briefly and replies faster.', '对话助手简短推理，回复更快。'),
      medium: tr('Default for the conversation assistant.', '对话助手的默认档位。'),
      high: tr('The conversation assistant thinks harder about difficult questions; replies are slower and use more tokens.', '对话助手对疑难问题想得更深；回复更慢，用量更多。'),
    }[normalize(level)];
  }

  function render(options) {
    const { iconHtml, disabled } = options;
    const current = normalize(options.level);
    const rows = LEVELS.includes(current) ? LEVELS : LEVELS.concat(current);
    return `<details class="gpi-effort-menu" data-gpi-effort-menu data-popover-menu>
      <summary aria-label="${esc(`${tr('Reasoning depth', '思考深度')}：${label(current)}`)}" title="${esc(description(current))}" ${disabled ? 'aria-disabled="true"' : ''}>${iconHtml('spark', 14)}<span>${esc(tr('Reasoning depth', '思考深度'))} · ${esc(label(current))}</span><span class="gpi-access-chevron" aria-hidden="true">${iconHtml('chevron', 13)}</span></summary>
      <div class="gpi-access-popover gpi-effort-popover" role="group" aria-label="${esc(tr('Reasoning depth for this conversation', '本对话的思考深度'))}">
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
      callbacks.setError(String((error && error.message) || tr('The reasoning depth could not be changed.', '思考深度未能修改。')).slice(0, 240));
      callbacks.render(true);
    }
  }

  window.EasyICU.guidedPi.declare('effortMenu', {
    render, handleClick, preferred, remember, normalize, label, description, LEVELS,
  });
})();
