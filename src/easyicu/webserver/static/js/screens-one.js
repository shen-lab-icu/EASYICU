/* screens-one.js — owner: the single surface.
 *
 * There is one screen. The conversation already ships its own three columns —
 * a session rail, the thread, and a study-workspace aside — so this file does
 * not wrap it in a second shell. It mounts into the slots the conversation
 * already has:
 *
 *   .gd-rail    study folders  → also the destination nav      (left)
 *   .gd-conv    the thread                                     (middle)
 *   .gd-aside   study workspace → also the panel host          (right)
 *   .gd-top     brand + exit   → also the data-source chip
 *
 * Every other destination opens by rendering that screen's own render(). No
 * screen was rewritten to become a panel.
 *
 * PLACEMENT. A 460px aside is the wrong home for a screen built to be read: a
 * cross-database table or a concept dictionary arrives pre-squeezed. So a panel
 * declares where it belongs, and `center` swaps the two columns — the panel
 * takes the wide middle track and the conversation moves to a narrow column
 * beside it. The swap itself moves no node: it is grid-column assignment on a
 * grid that already has three tracks.
 *
 * That is NOT the same as the conversation surviving untouched. A route change
 * still runs the shell's render(), which rebuilds the whole subtree — measured:
 * the composer input is a different DOM node afterwards. What survives, and the
 * reason opening a panel mid-sentence does not cost you the sentence, is that
 * guided re-seeds its thread and composer draft from its own state on mount.
 * Lean on that, not on DOM persistence.
 *
 * The panel is mounted as a SIBLING of #gdAsideBody rather than replacing it,
 * because the conversation keeps writing its study workspace into that node on
 * its own schedule. Hiding it instead of overwriting it means the two owners
 * never fight over the same element.
 *
 * The summon contract is one attribute: [data-panel="<screenId>"].
 *
 * NAMES. This file holds no destination labels. It reads window.EU_DESTINATIONS
 * (shell-owned, built from crumb keys), because a nav that spells its own
 * labels is how a screen ends up called "Ideas" here and "Idea Mining"
 * everywhere else.
 *
 * Shape follows the same split LangChain's agent-chat-ui uses — thread state
 * and side-panel state are separate owners — and OpenHands' rule that the UI
 * reads a log rather than calling the agent.
 */
(function () {
  const t = (en, zh) => (window.t ? window.t(en, zh) : en);
  const icon = (n, s) => (window.icon ? window.icon(n, s) : '');
  const crumb = (c) => (window.EU_CRUMB_LABEL ? window.EU_CRUMB_LABEL(c) : c);
  const destinations = () => (window.EU_DESTINATIONS ? window.EU_DESTINATIONS() : []);

  /* Utility panels are not part of a study's reading order, so they open from
     the rail foot rather than the destination list. */
  const UTILITY = [
    { id: 'settings', crumb: 'Settings' },
    { id: 'tutorial', crumb: 'Get Started' },
  ];

  /* Where each panel opens. `center` gives it the wide middle column and moves
     the conversation to a narrow one beside it; `aside` leaves the conversation
     in the middle.

     Measured at 1440px, comparing the same screen in the 836px centre column
     against the 445px aside. Horizontal overflow turned out to be the wrong
     test — only `states` clips (194px). The screens are responsive, so the real
     cost is reflow: the content is wrung out into a taller and taller ribbon.
     How much taller, squeezed vs centred:

       dictionary 3.04×   ideas 2.85×   agent 1.95×   settings 1.43×
       crossdb 1.35×      tutorial 1.19×   patient 1.13×   cohort 1.11×

     The ratio under-detects two kinds of screen, so it is a floor rather than
     a verdict: a chart shrinks instead of rewrapping, and a fixed-height screen
     whose panes scroll internally reads the same either way (extraction and
     states are both 796px in either column, which is why states is placed on
     its 194px clip instead).

     Settings is here because the ratio was right and I was not: 1.43× looked
     mild enough to keep a dip-in-and-leave screen in the aside, and the aside
     then wrapped its capability cards to three characters per line. When the
     measurement and the story disagree, the measurement is the one that was
     looking at the screen. Anything still wrong is one button away — the
     override below is per panel and per reader. */
  const PLACEMENT = {
    patient: 'center',
    cohort: 'center',
    crossdb: 'center',
    extraction: 'center',
    ideas: 'center',
    agent: 'center',
    dictionary: 'center',
    states: 'center',
    settings: 'center',
    tutorial: 'aside',
  };
  const PANEL_ROUTES = Object.keys(PLACEMENT);
  /* Per-panel override: whatever the table guessed, the reader decides. */
  const chosen = {};
  const placementOf = (id) => chosen[id] || PLACEMENT[id] || 'aside';

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
    ));
  }

  function panelTitle(id) {
    const hit = destinations().find((d) => d.id === id);
    if (hit) return hit.label;
    const util = UTILITY.find((u) => u.id === id);
    if (util) return crumb(util.crumb);
    const scr = window.SCREENS[id];
    if (scr && scr.crumbs && scr.crumbs.length) return crumb(scr.crumbs[scr.crumbs.length - 1]);
    return id;
  }

  /* ---- the data-source chip, mounted into the conversation's own top bar.
     It lost its home when the navigation shell went; it is not decorative — it
     is the only thing on screen that says whether these numbers are demo or
     real. ---- */

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

  /* ---- destination nav, mounted into the conversation's own rail ---- */

  function railNavHtml(active) {
    const list = destinations();
    const items = list.map((d) => `
      <button type="button" class="one-navitem ${active === d.id ? 'on' : ''}"
        data-panel="${d.id}"${active === d.id ? ' aria-current="page"' : ''}>
        <span class="ico">${icon(d.ico, 15)}</span><span class="lb">${esc(d.label)}</span>
      </button>`).join('');
    /* The orientation hint the entry screen used to carry. It names two
       destinations, so each name IS the control that opens it — prose pointing
       at a menu that no longer exists would just be a dead end. */
    const named = (id) => {
      const hit = list.find((d) => d.id === id);
      return hit
        ? `<button type="button" class="one-inline" data-panel="${id}">${esc(hit.label)}</button>`
        : '';
    };
    return `
      <nav class="one-railnav" aria-label="${t('Destinations', '目的地')}">
        <button type="button" class="one-navitem chat ${active ? '' : 'on'}" data-panel=""${active ? '' : ' aria-current="page"'}>
          <span class="ico">${icon('spark', 15)}</span><span class="lb">${esc(crumb('Guided Copilot'))}</span>
        </button>
        <div class="one-navsec">${t('Open beside the conversation', '在对话旁打开')}</div>
        ${items}
        <p class="one-starthint">${t('Starting from a paper? Open', '有文章，从')}${named('ideas')}${t('. Have a clear question? Just say it here. Already have data? Open', '开始；有明确问题，直接在这里说；已经有数据，打开')}${named('extraction')}${t('.', '。')}</p>
      </nav>`;
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
    const wide = placementOf(id) === 'center';
    const swapTo = wide ? 'aside' : 'center';
    const swapHint = wide
      ? t('Move beside the conversation', '移到对话旁边')
      : t('Give this the wide column', '让它占用宽栏');
    return `
    <div class="one-panelbar">
      <span class="t">${esc(panelTitle(id))}</span>
      <span class="sp"></span>
      <button type="button" class="one-placebtn" data-oneplace="${swapTo}" data-oneplaceid="${id}"
        title="${esc(swapHint)}">${wide ? t('Side', '侧栏') : t('Wide', '宽栏')}</button>
      <button type="button" class="one-iconbtn" data-panel=""
        title="${t('Close panel', '关闭面板')}" aria-label="${t('Close panel', '关闭面板')}">${icon('close', 15)}</button>
    </div>
    <div class="one-panelbody"><div class="content">${body}</div></div>`;
  }

  /* ---- mount ---------------------------------------------------------- */

  function mountTop(root) {
    const top = root.querySelector('.gd-top');
    if (!top || top.querySelector('.one-source')) return;
    const frag = document.createElement('div');
    frag.className = 'one-topmount';
    frag.innerHTML = sourceChipHtml();
    const grow = top.querySelector('.grow');
    if (grow) top.insertBefore(frag, grow.nextSibling);
    else top.appendChild(frag);
  }

  function mountRail(root, panelId) {
    const rail = root.querySelector('.gd-rail');
    if (!rail) return;
    const prior = rail.querySelector('.one-railnav');
    if (prior) prior.remove();
    const holder = document.createElement('div');
    holder.innerHTML = railNavHtml(panelId);
    const nav = holder.firstElementChild;
    const folders = rail.querySelector('#gdSessions');
    if (folders) rail.insertBefore(nav, folders);
    else rail.appendChild(nav);

    const foot = rail.querySelector('.gd-rail-foot');
    if (foot && !foot.querySelector('[data-cpopen]') && window.EU_SHELL_CONTROLS) {
      const guide = document.createElement('div');
      guide.className = 'one-footmount';
      guide.innerHTML = window.EU_SHELL_CONTROLS();
      foot.insertBefore(guide, foot.firstChild);
    }
  }

  function mountPanel(root, panelId) {
    const aside = root.querySelector('.gd-aside');
    if (!aside) return;
    const prior = aside.querySelector('.one-panelhost');
    if (prior) prior.remove();
    const place = panelId ? placementOf(panelId) : '';
    aside.classList.toggle('has-panel', !!panelId);
    /* .gd-main is a grid, so a panel is sized by its TRACK. Setting a width on
       the aside instead let the item overflow its own column and run 138px past
       the right edge of the viewport — measured, before this line existed. */
    const main = root.querySelector('.gd-main');
    if (main) {
      main.classList.toggle('has-panel', !!panelId);
      main.classList.toggle('panel-center', place === 'center');
    }
    if (!panelId) return;
    const host = document.createElement('div');
    host.className = 'one-panelhost';
    host.innerHTML = panelHtml(panelId);
    aside.appendChild(host);
  }

  /* The conversation says what is open. Without this the two columns are two
     windows that happen to share a browser tab: you can be reading a cohort
     table while the thread talks about something else and nothing on screen
     connects them. It sits above the composer because that is where you act. */
  function mountConv(root, panelId) {
    const wrap = root.querySelector('.gd-composer-wrap');
    if (!wrap) return;
    const prior = root.querySelector('.one-viewing');
    if (prior) prior.remove();
    if (!panelId) return;
    const bar = document.createElement('div');
    bar.className = 'one-viewing';
    bar.innerHTML = `
      <span>${t('Viewing', '正在看')}</span>
      <span class="nm">${esc(panelTitle(panelId))}</span>
      <span class="sp"></span>
      <button type="button" data-panel="">${t('Back to the conversation', '回到对话')}</button>`;
    wrap.parentNode.insertBefore(bar, wrap);
  }

  function mount(root, panelId) {
    mountTop(root);
    mountRail(root, panelId);
    mountConv(root, panelId);
    mountPanel(root, panelId);
  }

  window.EU_ONE = {
    PANEL_ROUTES,
    isPanel: (id) => PANEL_ROUTES.indexOf(id) >= 0,
    placementOf,
    mount,
  };

  /* One delegated listener owns the summon contract. data-panel="" closes. */
  document.addEventListener('click', (e) => {
    const place = e.target.closest('[data-oneplace]');
    if (place) {
      e.preventDefault();
      chosen[place.getAttribute('data-oneplaceid')] = place.getAttribute('data-oneplace');
      if (window.__euRender) window.__euRender();
      return;
    }
    const el = e.target.closest('[data-panel]');
    if (!el) return;
    const id = el.getAttribute('data-panel');
    e.preventDefault();
    if (!id) { location.hash = '#guided'; return; }
    if (window.SCREENS[id]) location.hash = '#' + id;
  });
}());
