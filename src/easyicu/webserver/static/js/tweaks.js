/* EasyICU — Tweaks panel controller (vanilla).
   Implements the host edit-mode protocol so the toolbar "Tweaks" toggle
   shows/hides the panel. Applies tokens live + persists to localStorage so
   the chosen look survives reloads. Hidden entirely when Tweaks is off. */
(function () {
  const KEY = 'easyicu_tweaks_v1';
  // Appearance tokens only. `density` deliberately does NOT live here: it is a
  // persisted user setting owned by i18n.js::applyDisplayDom, which reads
  // /api/settings. Both files used to write body[data-density] — this one
  // synchronously at DOMContentLoaded, the other when the settings fetch
  // resolved — so the winner was decided by network latency, and the Tweaks
  // density control was a dead lever that reverted on every reload.
  const DEFAULTS = { accent: 'teal', tone: 'warm', radius: 'default', home: 'prompt' };

  const ACCENT_H = { teal: 205, blue: 245, green: 150, violet: 290 };
  const TONES = {
    warm:    ['#FAFAF7', '#F4F4F0', '#EEEEE8', '#FCFCFA'],
    neutral: ['#FAFAFA', '#F3F3F3', '#ECECEC', '#FCFCFC'],
    cool:    ['#F7F9FB', '#EFF2F5', '#E8ECF1', '#FAFBFD'],
  };
  const RADII = {
    sharp:   ['2px', '3px', '5px', '7px'],
    default: ['4px', '6px', '10px', '14px'],
    rounded: ['7px', '11px', '16px', '22px'],
  };

  let values = load();

  function load() {
    try { return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(KEY) || '{}')); }
    catch (e) { return Object.assign({}, DEFAULTS); }
  }
  function persist() { try { localStorage.setItem(KEY, JSON.stringify(values)); } catch (e) {} }

  function apply() {
    const r = document.documentElement.style;
    const h = ACCENT_H[values.accent] ?? 205;
    r.setProperty('--accent', `oklch(55% 0.075 ${h})`);
    r.setProperty('--accent-ink', `oklch(40% 0.07 ${h})`);
    r.setProperty('--accent-soft', `oklch(96% 0.02 ${h})`);
    r.setProperty('--accent-border', `oklch(86% 0.045 ${h})`);

    const tone = TONES[values.tone] || TONES.warm;
    r.setProperty('--bg', tone[0]);
    r.setProperty('--surface-2', tone[1]);
    r.setProperty('--surface-3', tone[2]);
    r.setProperty('--rail', tone[3]);

    const rad = RADII[values.radius] || RADII.default;
    r.setProperty('--r-1', rad[0]);
    r.setProperty('--r-2', rad[1]);
    r.setProperty('--r-3', rad[2]);
    r.setProperty('--r-4', rad[3]);

    // home layout is read by the entry screen from localStorage
    try { localStorage.setItem('easyicu_home', values.home); } catch (e) {}
  }

  function setTweak(key, val) {
    values[key] = val;
    apply();
    persist();
    refreshActive();
    // home layout changes the entry structure → re-render if we're on it
    if (key === 'home') { const r = (location.hash || '#entry').slice(1) || 'entry'; if (r === 'entry' && window.__euRender) window.__euRender(); }
    try { window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { [key]: val } }, '*'); } catch (e) {}
  }

  /* ---------- panel DOM ---------- */
  let panel;
  function swatch(group, val, color) {
    return `<button class="tw-sw ${values[group] === val ? 'on' : ''}" data-tw="${group}" data-val="${val}" title="${val}" style="background:${color};"></button>`;
  }
  function seg(group, opts) {
    return `<div class="tw-seg" data-tw="${group}">${opts.map(([v, lab]) =>
      `<button class="${values[group] === v ? 'on' : ''}" data-val="${v}">${lab}</button>`).join('')}</div>`;
  }
  function build() {
    panel = document.createElement('div');
    panel.className = 'tw-panel';
    panel.hidden = true;
    panel.innerHTML = `
      <div class="tw-head" id="twHead">
        <div class="tw-mark">${icon('sliders', 14)}</div>
        <div><div class="tw-title">Tweaks</div><div class="tw-sub">EasyICU appearance</div></div>
        <button class="tw-x" id="twClose" title="Close">${icon('stop', 13)}</button>
      </div>
      <div class="tw-body">
        <div class="tw-sec">
          <div class="tw-lbl">Accent</div>
          <div class="tw-swatches">
            ${swatch('accent', 'teal', 'oklch(55% 0.075 205)')}
            ${swatch('accent', 'blue', 'oklch(55% 0.075 245)')}
            ${swatch('accent', 'green', 'oklch(55% 0.075 150)')}
            ${swatch('accent', 'violet', 'oklch(55% 0.075 290)')}
          </div>
        </div>
        <div class="tw-sec">
          <div class="tw-lbl">Background tone</div>
          ${seg('tone', [['warm', 'Warm'], ['neutral', 'Neutral'], ['cool', 'Cool']])}
        </div>
        <div class="tw-sec">
          <div class="tw-lbl">Corner radius</div>
          ${seg('radius', [['sharp', 'Sharp'], ['default', 'Default'], ['rounded', 'Rounded']])}
        </div>
        <div class="tw-sec">
          <div class="tw-lbl">Home layout</div>
          ${seg('home', [['prompt', 'Prompt'], ['copilot', 'Copilot'], ['cards', 'Cards']])}
        </div>
      </div>
      <div class="tw-foot">
        <span class="tw-note">Saved locally · live preview</span>
        <button class="tw-reset" id="twReset">Reset</button>
      </div>`;
    document.body.appendChild(panel);

    panel.addEventListener('click', (e) => {
      const sw = e.target.closest('.tw-sw');
      if (sw) { setTweak(sw.dataset.tw, sw.dataset.val); return; }
      const sb = e.target.closest('.tw-seg button');
      if (sb) { setTweak(sb.parentElement.dataset.tw, sb.dataset.val); return; }
    });
    panel.querySelector('#twClose').addEventListener('click', dismiss);
    panel.querySelector('#twReset').addEventListener('click', () => {
      values = Object.assign({}, DEFAULTS); apply(); persist(); refreshActive();
      try { window.parent.postMessage({ type: '__edit_mode_set_keys', edits: values }, '*'); } catch (e) {}
    });
    enableDrag(panel.querySelector('#twHead'));
  }

  function refreshActive() {
    if (!panel) return;
    panel.querySelectorAll('.tw-sw').forEach(b => b.classList.toggle('on', values[b.dataset.tw] === b.dataset.val));
    panel.querySelectorAll('.tw-seg').forEach(seg => {
      const g = seg.dataset.tw;
      seg.querySelectorAll('button').forEach(b => b.classList.toggle('on', values[g] === b.dataset.val));
    });
  }

  function enableDrag(handle) {
    let sx, sy, ox, oy, dragging = false;
    handle.addEventListener('pointerdown', (e) => {
      if (e.target.closest('.tw-x')) return;
      dragging = true; handle.classList.add('grabbing');
      const r = panel.getBoundingClientRect();
      // switch to left/top positioning
      panel.style.left = r.left + 'px'; panel.style.top = r.top + 'px';
      panel.style.right = 'auto'; panel.style.bottom = 'auto';
      sx = e.clientX; sy = e.clientY; ox = r.left; oy = r.top;
      handle.setPointerCapture(e.pointerId);
    });
    handle.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const w = panel.offsetWidth, h = panel.offsetHeight;
      let nx = ox + (e.clientX - sx), ny = oy + (e.clientY - sy);
      nx = Math.max(6, Math.min(nx, innerWidth - w - 6));
      ny = Math.max(6, Math.min(ny, innerHeight - h - 6));
      panel.style.left = nx + 'px'; panel.style.top = ny + 'px';
    });
    const end = () => { dragging = false; handle.classList.remove('grabbing'); };
    handle.addEventListener('pointerup', end);
    handle.addEventListener('pointercancel', end);
  }

  /* ---------- host edit-mode protocol ---------- */
  function activate() { if (panel) panel.hidden = false; }
  function dismiss() {
    if (panel) panel.hidden = true;
    try { window.parent.postMessage({ type: '__edit_mode_dismissed' }, '*'); } catch (e) {}
  }

  function init() {
    apply();          // apply saved look immediately, even while hidden
    build();
    window.addEventListener('message', (e) => {
      const t = e && e.data && e.data.type;
      if (t === '__activate_edit_mode') activate();
      else if (t === '__deactivate_edit_mode') { if (panel) panel.hidden = true; }
    });
    try { window.parent.postMessage({ type: '__edit_mode_available' }, '*'); } catch (e) {}
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
