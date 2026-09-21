/* Owner: inline model menu on the Guided Copilot composer chip. */
/* Fills the chip popover when it opens and switches the model in place, so a
   model change never re-renders the composer and never asks the browser for a
   credential it does not hold. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function clean(value, limit) { return String(value || '').trim().slice(0, limit || 256); }

  const state = { models: [], current: '', loading: false, busy: false, error: '', request: 0 };

  function labelSpan(control) { return control.querySelector('.gpi-model-binding > span'); }

  // The chip already states the binding as "provider · model"; reading it back
  // keeps the checked row honest without a second status round trip.
  function currentFromControl(control) {
    const span = labelSpan(control);
    const parts = clean(span && span.textContent, 256).split('·');
    return clean(parts.length > 1 ? parts[parts.length - 1] : parts[0], 256);
  }
  function providerPrefix(control) {
    const span = labelSpan(control);
    const parts = clean(span && span.textContent, 256).split('·');
    return parts.length > 1 ? clean(parts[0], 80) : '';
  }

  function rowsHtml() {
    if (state.loading) {
      return `<p class="gpi-model-note" role="status">${esc(tr('Reading the model list…', '正在读取模型列表…'))}</p>`;
    }
    if (state.error) return `<p class="gpi-model-note is-error">${esc(state.error)}</p>`;
    if (!state.models.length) {
      return `<p class="gpi-model-note">${esc(tr('This service reported no models.', '该服务未报告任何模型。'))}</p>`;
    }
    return state.models.map(model => {
      const text = clean(model, 256);
      const active = text === state.current;
      return `<button type="button" role="menuitemradio" data-gpi-pick-model="${esc(text)}" aria-checked="${active}"><span>${esc(text)}</span><span aria-hidden="true">${active ? '✓' : ''}</span></button>`;
    }).join('');
  }

  function paint(control) {
    const popover = control.querySelector('[data-gpi-model-popover]');
    if (!popover) return;
    popover.querySelectorAll('.gpi-model-rows').forEach(node => node.remove());
    const list = document.createElement('div');
    list.className = 'gpi-model-rows';
    list.innerHTML = rowsHtml();
    const settings = popover.querySelector('.gpi-model-settings');
    if (settings) popover.insertBefore(list, settings); else popover.appendChild(list);
  }

  async function load(control) {
    state.current = currentFromControl(control);
    if (state.loading || state.models.length) { paint(control); return; }
    const request = ++state.request;
    state.loading = true;
    state.error = '';
    paint(control);
    try {
      const caller = api().loadPiCopilotApiModels;
      if (typeof caller !== 'function') {
        throw new Error(tr('Model listing is unavailable.', '模型列表不可用。'));
      }
      const payload = await caller();
      if (request !== state.request) return;
      const rows = Array.isArray(payload && payload.models) ? payload.models : [];
      state.models = rows.map(row => clean(row && row.id ? row.id : row, 256)).filter(Boolean);
    } catch (error) {
      if (request !== state.request) return;
      state.error = clean((error && error.message)
        || tr('Could not read the model list.', '读取模型列表失败。'), 240);
    } finally {
      if (request === state.request) state.loading = false;
      paint(control);
    }
  }

  async function pick(control, model) {
    if (!model || model === state.current || state.busy) return;
    const request = ++state.request;
    state.busy = true;
    state.error = '';
    paint(control);
    try {
      const caller = api().switchPiCopilotModel;
      if (typeof caller !== 'function') {
        throw new Error(tr('Model switching is unavailable.', '模型切换不可用。'));
      }
      await caller(model);
      if (request !== state.request) return;
      state.current = model;
      const span = labelSpan(control);
      if (span) {
        const prefix = providerPrefix(control);
        const label = prefix ? `${prefix} · ${model}` : model;
        // Rewritten in place: the composer does not re-render, so the chip
        // would otherwise keep naming the model that was just replaced.
        span.textContent = label;
        span.title = label;
      }
      const details = control.closest('.gpi-model-control');
      if (details) details.open = false;
    } catch (error) {
      if (request !== state.request) return;
      state.error = clean((error && error.message)
        || tr('The model service refused this change.', '模型服务拒绝了这次变更。'), 240);
    } finally {
      if (request === state.request) state.busy = false;
      paint(control);
    }
  }

  function controlOf(node) {
    return node && node.closest ? node.closest('.gpi-model-control') : null;
  }

  // toggle does not bubble, so both listeners are installed in the capture phase.
  function onToggle(event) {
    const control = controlOf(event.target);
    if (control && control.open) load(control);
  }

  function onClick(event) {
    const choice = event.target.closest ? event.target.closest('[data-gpi-pick-model]') : null;
    if (!choice) return;
    const control = controlOf(choice);
    if (!control) return;
    event.preventDefault();
    void pick(control, clean(choice.dataset.gpiPickModel, 256));
  }

  function mount() {
    if (!document || typeof document.addEventListener !== 'function') return;
    document.addEventListener('toggle', onToggle, true);
    document.addEventListener('click', onClick, true);
  }

  mount();
  window.EasyICU.guidedPi.declare('modelMenu', { mount, load, pick, state });
})();
