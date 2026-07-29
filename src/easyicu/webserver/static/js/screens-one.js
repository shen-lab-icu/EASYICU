/* screens-one.js — owner: the single surface.
 *
 * It is a conversation window. One column, one thread, and nothing else until
 * the conversation earns it. Depth arrives when it is reached, not up front.
 *
 * There is no destination menu, and removing it is the point. The conversation
 * already offers where to go — in its own words, at the moment it makes sense
 * ("找研究想法 · 准备/抽取数据 · 审阅已有数据 · 运行研究项目"). A permanent nav
 * beside it was that same list a second time, always on, always at full depth,
 * competing with the thread for the answer to "what do I do now".
 *
 * What is left mounts into the conversation's own slots:
 *
 *   .gd-rail    the study list — collapsed; only for switching studies
 *   .gd-conv    the thread — the window
 *   .gd-aside   the panel, when there is something to show
 *   .gd-top     the data-source chip and the rail toggle
 *
 * A panel still renders the destination screen's own render(); no screen was
 * rewritten. Deep links (#patient, #crossdb) still resolve, so nothing became
 * unreachable — it just stopped being advertised before it was relevant.
 *
 * PLACEMENT, and which column is the point. The middle is the conversation and
 * stays the conversation: it is where the project is controlled, and a reader
 * who watches only the middle should never miss what happened. The right column
 * is secondary — detail, a figure, a result, the thing a step produced. Nothing
 * displaces the middle on its own.
 *
 * `center` therefore exists as a deliberate override, not a default: it swaps
 * the two columns for one panel when you actually want to spread something out.
 * The swap itself moves no node — it is grid-column assignment on a grid that
 * already has three tracks.
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

  /* Every panel opens in the right column. That is the hierarchy, not a
     measurement result — the conversation is the control surface and does not
     get pushed aside by a page you opened to glance at.

     What IS measured is the cost of that choice, so it is not silent. At 1440px
     the same screen in the 560px panel vs the 836px middle is this much taller,
     wrung out into a narrower ribbon:

       dictionary 3.04×   ideas 2.85×   agent 1.95×   settings 1.43×
       crossdb 1.35×      tutorial 1.19×   patient 1.13×   cohort 1.11×

     (Measured against the older 445px panel; horizontal overflow turned out to
     be the wrong test — only `states` clips, by 194px. The ratio also
     under-detects two kinds of screen: a chart shrinks instead of rewrapping,
     and a fixed-height screen whose panes scroll internally reads the same
     either way — extraction and states are both 796px in either column.)

     So the top four have a real cost in the panel, and the answer is a control
     rather than a different default: the panel bar's 宽栏 / Wide button hands
     that one panel the middle for as long as you want it. Read the table as
     "these are the ones you will reach for it on", not as a placement. */
  const PANEL_ROUTES = [
    'patient', 'cohort', 'crossdb', 'extraction',
    'ideas', 'agent', 'dictionary', 'states', 'settings', 'tutorial',
  ];
  /* Per-panel override. The default is the hierarchy; this is the reader. */
  const chosen = {};
  const placementOf = (id) => chosen[id] || 'aside';

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
    /* Two tabs, because this column already had an owner. The study workspace
       is the conversation's own running output — which step you are on, what
       is left — and opening the dictionary to look one thing up used to make it
       vanish with nothing on screen saying it still existed. It is the first
       tab and the way back, so a panel borrows this column instead of taking
       it. This is also why there is no separate close button: the way out of a
       detail view is your study, not an X into nowhere. */
    return `
    <div class="one-panelbar" role="tablist">
      <button type="button" class="one-tab" data-panel="" role="tab" aria-selected="false"
        >${esc(t('Study workspace', '研究工作区'))}</button>
      <button type="button" class="one-tab on" role="tab" aria-selected="true"
        >${esc(panelTitle(id))}</button>
      <span class="sp"></span>
      <button type="button" class="one-placebtn" data-oneplace="${swapTo}" data-oneplaceid="${id}"
        title="${esc(swapHint)}">${wide ? t('Side', '侧栏') : t('Wide', '宽栏')}</button>
    </div>
    <div class="one-panelbody"><div class="content">${body}</div></div>`;
  }

  /* ---- mount ---------------------------------------------------------- */

  /* The rail is off by default and remembered, the way a chat app remembers
     whether you keep its sidebar open. Off is the shallow end. */
  const RAIL_KEY = 'easyicu.one.rail';
  const railOpen = () => {
    try { return localStorage.getItem(RAIL_KEY) === '1'; } catch (e) { return false; }
  };
  function setRailOpen(on) {
    try { localStorage.setItem(RAIL_KEY, on ? '1' : '0'); } catch (e) { /* private mode */ }
  }

  function mountTop(root) {
    const top = root.querySelector('.gd-top');
    if (!top || top.querySelector('.one-source')) return;
    const frag = document.createElement('div');
    frag.className = 'one-topmount';
    frag.innerHTML = sourceChipHtml();
    const grow = top.querySelector('.grow');
    if (grow) top.insertBefore(frag, grow.nextSibling);
    else top.appendChild(frag);
    const brand = top.querySelector('.gd-home-link');
    if (brand && !top.querySelector('[data-onerail]')) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'one-iconbtn one-railtoggle';
      b.setAttribute('data-onerail', '');
      b.title = t('Your studies', '你的研究');
      b.setAttribute('aria-label', t('Your studies', '你的研究'));
      b.innerHTML = icon('list', 16);
      top.insertBefore(b, brand);
    }
  }

  /* The rail is the study list and nothing else now — no destination menu.
     The conversation already offers where to go, in its own words, at the
     moment it makes sense; a permanent menu beside it was the same list a
     second time, always on, always at full depth. It stays collapsed until you
     ask for it, which is the only reason to open it: switching studies. */
  function mountRail(root, panelId) {
    const rail = root.querySelector('.gd-rail');
    if (!rail) return;
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
      /* One window by default: the rail appears only when asked for. */
      main.classList.toggle('show-rail', railOpen());
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

  /* Opening a main-line destination is a handoff, not a jump: it snapshots what
     the conversation has collected into the shared study and records the
     transition, so the destination arrives already knowing the cohort, the
     outcome, the export target. Review and reference destinations deliberately
     do NOT — looking a concept up in the dictionary is not a step in anyone's
     study, and writing one in would be a lie about what the study did.

     It lives on the mount path rather than on a click, because the click that
     used to trigger it was the nav's, and the nav is gone. A deep link, a
     conversation link and a restored hash are all openings and all deserve the
     same handoff. `handedOff` keeps re-renders of the same panel from writing
     the transition again.

     Through guided's own adapter, not EU_STUDY_CONTEXT.handoff directly: the
     adapter carries continueExisting (without it a handoff can mint a new study
     instead of advancing this one) and a real scientific guard that refuses to
     hand a Cross-DB plan to Agent Projects until it is reframed as a
     single-export question. Reaching past it would silently drop both. */
  let handedOff = '';
  function handOffStudy(panelId) {
    if (!panelId) { handedOff = ''; return; }
    if (handedOff === panelId) return;
    handedOff = panelId;
    const hit = destinations().find((d) => d.id === panelId);
    const guided = window.EU_GUIDED_STUDY_CONTEXT;
    if (!hit || hit.line !== 'main' || !guided || !guided.handoff) return;
    try {
      const r = guided.handoff(panelId);
      if (r && r.persisted && r.persisted.catch) {
        r.persisted.catch((err) => console.warn('[EasyICU] study handoff refused or stayed local:', err));
      }
    } catch (err) {
      /* A study that cannot be handed over must not strand the reader. */
      console.warn('[EasyICU] study handoff failed:', err);
    }
  }

  function mount(root, panelId) {
    mountTop(root);
    mountRail(root, panelId);
    mountConv(root, panelId);
    mountPanel(root, panelId);
    handOffStudy(panelId);
  }

  window.EU_ONE = {
    PANEL_ROUTES,
    isPanel: (id) => PANEL_ROUTES.indexOf(id) >= 0,
    placementOf,
    mount,
  };

  document.addEventListener('click', (e) => {
    const railBtn = e.target.closest('[data-onerail]');
    if (!railBtn) return;
    e.preventDefault();
    setRailOpen(!railOpen());
    if (window.__euRender) window.__euRender();
  });

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
    if (!window.SCREENS[id]) return;
    location.hash = '#' + id;
  });
}());
