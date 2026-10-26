/* Owner: Settings route personal instructions for the Copilot.
   The host's PersonalPreferencesStore owns the value (desktop: this computer;
   hosted: the signed-in user) and freezes it into each new conversation, where
   the Copilot reads it as advice on how to answer. It never reaches plan
   review, execution or claim checks, so this page says so once. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }

  const state = { loaded: false, loading: false, saved: '', draft: null, max: 4000, saving: false, error: '', notice: '' };
  let box = null;

  const ERRORS = {
    preferences_instructions_invalid: ['Remove control characters and keep it under the limit.', '请去掉特殊控制字符，并控制在字数上限内。'],
    pi_message_phi_risk: ['This text looks like it contains patient details. Remove them before saving.', '这段文字疑似含有患者信息，请删去后再保存。'],
    preferences_store_busy: ['Saving is busy. Try again in a moment.', '正在保存其他修改，请稍后再试。'],
    preferences_store_unavailable: ['Preferences cannot be saved right now.', '暂时无法保存偏好。'],
    preferences_store_invalid: ['The saved preferences file is damaged. Save again to replace it.', '已保存的偏好文件损坏，重新保存即可覆盖。'],
  };
  function errorText(failure) {
    const known = ERRORS[failure && failure.code];
    return known ? tr(...known) : tr('Could not save.', '保存失败。');
  }
  const length = text => Array.from(String(text || '')).length;

  async function ensureLoaded() {
    if (state.loaded || state.loading || typeof api().loadPersonalPreferences !== 'function') return;
    state.loading = true;
    try {
      const payload = await api().loadPersonalPreferences();
      state.saved = String((payload && payload.preferences && payload.preferences.instructions) || '');
      const max = payload && payload.options && payload.options.max_instructions_chars;
      if (Number.isInteger(max) && max > 0) state.max = max;
      state.loaded = true;
    } catch (failure) {
      state.error = tr('Preferences are unavailable right now.', '暂时读不到个人偏好。');
    } finally {
      state.loading = false;
      paint();
    }
  }

  function cardHtml() {
    const value = state.draft === null ? state.saved : state.draft;
    const count = length(value);
    const changed = state.loaded && value !== state.saved;
    return `
        <label for="setPersonalInstructions">${tr('Every new conversation starts with these, e.g. how to answer or your unit’s abbreviations. They change how the Copilot writes, not how studies are reviewed or run.', '每个新对话都会带上这段说明，例如回答方式或科室常用缩写。它只影响 Copilot 怎么回答，不改变研究的审查和运行。')}</label>
        <textarea id="setPersonalInstructions" rows="5" data-set-personal-text maxlength="${state.max}" ${state.loaded ? '' : 'disabled'}>${esc(value)}</textarea>
        <div class="set-personal-foot">
          <span class="${count > state.max ? 'is-over' : ''}" data-set-personal-count>${count} / ${state.max}</span>
          ${state.error ? `<span class="set-personal-error" role="alert">${esc(state.error)}</span>` : state.notice ? `<span role="status">${esc(state.notice)}</span>` : ''}
          <button type="button" class="btn sm primary" data-set-personal-save ${changed && !state.saving && count <= state.max ? '' : 'disabled'}>${tr('Save', '保存')}</button>
        </div>`;
  }
  function section() {
    return `
      <div class="sec-stack" id="set-personal"><div class="lbl">${tr('Personal', '个人')}</div><h2>${tr('Standing instructions for the Copilot', 'Copilot 长期说明')}</h2></div>
      <div class="card pad set-personal" data-set-personal>${cardHtml()}
      </div>
`;
  }
  // The card repaints in place, so the page keeps its scroll position.
  function paint() {
    if (box && box.isConnected !== false) box.innerHTML = cardHtml();
  }

  async function save() {
    if (state.saving || typeof api().updatePersonalPreferences !== 'function') return;
    const value = state.draft === null ? state.saved : state.draft;
    state.saving = true; state.error = ''; state.notice = '';
    paint();
    try {
      const payload = await api().updatePersonalPreferences({ instructions: value });
      state.saved = String((payload && payload.preferences && payload.preferences.instructions) || '');
      state.draft = null;
      state.notice = tr('Saved. New conversations will use it.', '已保存，新对话起生效。');
    } catch (failure) {
      state.error = errorText(failure);
    } finally {
      state.saving = false;
      paint();
    }
  }

  function bind(root) {
    box = root.querySelector('[data-set-personal]');
    if (!box) return;
    void ensureLoaded();
    box.addEventListener('input', event => {
      if (!event.target.matches('[data-set-personal-text]')) return;
      state.draft = event.target.value;
      state.notice = ''; state.error = '';
      const count = length(state.draft);
      const counter = box.querySelector('[data-set-personal-count]');
      if (counter) { counter.textContent = `${count} / ${state.max}`; counter.classList.toggle('is-over', count > state.max); }
      const button = box.querySelector('[data-set-personal-save]');
      if (button) button.disabled = state.draft === state.saved || count > state.max;
    });
    box.addEventListener('click', event => {
      if (event.target.closest('[data-set-personal-save]')) void save();
    });
  }

  window.EU_SETTINGS_PERSONALIZATION = { section, bind, ensureLoaded, state };
})();
