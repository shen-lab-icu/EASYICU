/* screens-one.js — owner: the single surface.
 *
 * There is one screen. The conversation already ships its own three columns —
 * a session rail, the thread, and a study-workspace aside — so this file does
 * not wrap it in a second shell. It mounts into the slots the conversation
 * already has:
 *
 *   .gd-rail    session / study folders   (left, owned by the conversation)
 *   .gd-main    the thread                (middle, owned by the conversation)
 *   .gd-aside   study workspace  → also the panel host  (right, shared)
 *   .gd-top     brand + exit     → also the source chip and panel openers
 *
 * Every other destination opens in the aside by rendering that screen's own
 * render(). No screen was rewritten to become a panel.
 *
 * The panel is mounted as a SIBLING of #gdAsideBody rather than replacing it,
 * because the conversation keeps writing its study workspace into that node on
 * its own schedule. Hiding it instead of overwriting it means the two owners
 * never fight over the same element.
 *
 * The summon contract is one attribute: [data-panel="<screenId>"].
 *
 * Shape follows the same split LangChain's agent-chat-ui uses — thread state
 * and side-panel state are separate owners — and OpenHands' rule that the UI
 * reads a log rather than calling the agent.
 */
(function () {
  const t = (en, zh) => (window.t ? window.t(en, zh) : en);
  const icon = (n, s) => (window.icon ? window.icon(n, s) : '');

  /* Routes that name a panel rather than a page. */
  const PANEL_ROUTES = [
    'patient', 'cohort', 'crossdb', 'extraction',
    'ideas', 'agent', 'dictionary', 'states', 'settings', 'tutorial',
  ];

  /* Panels a 460px aside would misrepresent. `states` is here because its
     table overflowed the narrow panel by 193px when measured — it scrolled
     rather than clipped, but a table you must scroll to read sideways is not
     a table you can read. */
  const WIDE_PANELS = ['crossdb', 'agent', 'ideas', 'dictionary', 'states'];

  const QUICK = [
    { id: 'patient', label: ['Patients', '患者'] },
    { id: 'cohort', label: ['Cohort', '队列'] },
    { id: 'crossdb', label: ['Cross-database', '跨库'] },
    { id: 'extraction', label: ['Extract', '抽取'] },
    { id: 'agent', label: ['Projects', '研究项目'] },
    { id: 'ideas', label: ['Ideas', '想法'] },
  ];

  const L = (v) => (Array.isArray(v) ? t(v[0], v[1]) : v);

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
    ));
  }

  function panelTitle(id) {
    const scr = window.SCREENS[id];
    if (scr && scr.crumbs && scr.crumbs.length && window.EU_CRUMB_LABEL) {
      return window.EU_CRUMB_LABEL(scr.crumbs[scr.crumbs.length - 1]);
    }
    const q = QUICK.find((x) => x.id === id);
    return q ? L(q.label) : id;
  }

  /* ---- source chip + panel openers, mounted into the conversation's own
     top bar. These controls lost their home when the navigation shell went;
     the data-mode one is not decorative — it is the only thing on screen that
     says whether these numbers are demo or real. ---- */

  function sourceChipHtml() {
    const mode = window.EU_DISPLAYED_DATA_MODE
      ? window.EU_DISPLAYED_DATA_MODE()
      : 'demo';
    const demo = mode === 'demo';
    const official = demo
      && window.EU_DATA_MODE_CONTEXT
      && window.EU_DATA_MODE_CONTEXT.kind === 'official_demo';
    const title = demo
      ? t('Demo mode uses official public deidentified demo datasets or clearly labelled seeded examples. It is never your local data; switch to Real to load a local export.', '演示模式使用官方公开去标识 Demo 数据集，或明确标注的种子示例；都不是你的本地数据。切换到真实模式可加载本地导出。')
      : t('Real mode: screens compute from your local EasyICU export. Nothing is uploaded.', '真实模式：各页面从你本地的 EasyICU 导出计算，不上传任何数据。');
    return `
    <div class="one-source mode-seg ${demo ? 'demo-active' : ''}" role="group"
      aria-label="Data mode" title="${esc(title)}">
      <button type="button" class="${demo ? 'on' : ''}" data-datamode="demo"
        aria-pressed="${demo}">${icon('flask', 12)} ${official ? t('Official demo', '官方演示') : t('Demo data', '演示数据')}</button>
      <button type="button" class="${!demo ? 'on' : ''}" data-datamode="real"
        aria-pressed="${!demo}">${icon('db', 12)} ${t('Real', '真实')}</button>
    </div>`;
  }

  function openersHtml(active) {
    return `
    <div class="one-quick" role="group" aria-label="${t('Open a panel', '打开面板')}">
      ${QUICK.map((q) => `
        <button type="button" class="one-chip" data-panel="${q.id}"
          aria-pressed="${active === q.id}">${L(q.label)}</button>`).join('')}
    </div>`;
  }

  function panelHtml(id) {
    const scr = window.SCREENS[id];
    if (!scr) return '';
    let body;
    try {
      body = scr.render();
    } catch (e) {
      body = `<div class="one-empty">${t('This panel failed to render.', '该面板渲染失败。')} ${esc(e && e.message)}</div>`;
    }
    return `
    <div class="one-panelbar">
      <span class="t">${esc(panelTitle(id))}</span>
      <span class="sp"></span>
      <button type="button" class="one-iconbtn" data-panel=""
        title="${t('Close panel', '关闭面板')}" aria-label="${t('Close panel', '关闭面板')}">${icon('x', 16)}</button>
    </div>
    <div class="one-panelbody"><div class="content">${body}</div></div>`;
  }

  /* ---- mount ---------------------------------------------------------- */

  function mount(root, panelId) {
    const top = root.querySelector('.gd-top');
    if (top && !top.querySelector('.one-source')) {
      const grow = top.querySelector('.grow');
      const frag = document.createElement('div');
      frag.className = 'one-topmount';
      frag.innerHTML = openersHtml(panelId) + sourceChipHtml();
      if (grow) top.insertBefore(frag, grow.nextSibling);
      else top.appendChild(frag);
    }

    const aside = root.querySelector('.gd-aside');
    if (!aside) return;
    const prior = aside.querySelector('.one-panelhost');
    if (prior) prior.remove();
    const wide = !!panelId && WIDE_PANELS.indexOf(panelId) >= 0;
    aside.classList.toggle('has-panel', !!panelId);
    aside.classList.toggle('panel-wide', wide);
    /* .gd-main is a grid, so the panel is sized by widening its third TRACK.
       Setting a width on the aside instead lets the item overflow its own
       track and run off the viewport. */
    const main = root.querySelector('.gd-main');
    if (main) {
      main.classList.toggle('has-panel', !!panelId);
      main.classList.toggle('panel-wide', wide);
    }
    if (!panelId) return;
    const host = document.createElement('div');
    host.className = 'one-panelhost';
    host.innerHTML = panelHtml(panelId);
    aside.appendChild(host);
  }

  window.EU_ONE = {
    PANEL_ROUTES,
    isPanel: (id) => PANEL_ROUTES.indexOf(id) >= 0,
    isWide: (id) => WIDE_PANELS.indexOf(id) >= 0,
    mount,
  };

  /* One delegated listener owns the summon contract. data-panel="" closes. */
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-panel]');
    if (!el) return;
    const id = el.getAttribute('data-panel');
    e.preventDefault();
    if (!id) { location.hash = '#guided'; return; }
    if (window.SCREENS[id]) location.hash = '#' + id;
  });
}());
