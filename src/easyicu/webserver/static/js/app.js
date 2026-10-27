/* Owner: App shell route controller. */
/* App shell controller */
(function () {
  const app = document.getElementById('app');
  const routeAnnouncer = document.createElement('div');
  routeAnnouncer.className = 'shell-sr-only';
  routeAnnouncer.setAttribute('aria-live', 'polite');
  routeAnnouncer.setAttribute('aria-atomic', 'true');
  document.body.appendChild(routeAnnouncer);
  /* Route normalization: Copilot merge + folded Review destinations.
     Coverage Audit / SOFA Reclassification are now tabs inside Cohort Statistics;
     ICD Cohort Filter is folded into the extraction cohort filter. Old deep
     links still resolve to the right place. */
  const FALLBACK_ROUTE = 'guided';
  /* Retired pages: the classic home, the old Idea Mining workbench and the
     workspace-state catalogue. Old links and bookmarks land in the research
     conversation, which mines ideas from a topic, an article or a PDF. */
  const RETIRED_ROUTES = new Set(['entry', 'ideas', 'states']);

  function normRoute(r) {
    if (RETIRED_ROUTES.has(r)) { window.__euAlias = true; return 'guided'; }
    if (r === 'help') return 'tutorial';
    if (r === 'assistant') return 'guided';
    if (r === 'agent') { window.__euAlias = true; return 'guided'; }
    if (r === 'audit')       { window.__euCohortPanel = 'coverage'; window.__euAlias = true; return 'cohort'; }
    if (r === 'sofareclass') { window.__euCohortPanel = 'sofa';     window.__euAlias = true; return 'cohort'; }
    if (r === 'icd')         { window.__euExtractFocusICD = true;   window.__euAlias = true; return 'extraction'; }
    // The Data map is the data dictionary's study-type view.
    if (r === 'datamap')     { window.__euDictView = 'studies';     window.__euAlias = true; return 'dictionary'; }
    return r;
  }
  function rawRouteFromHash() {
    const raw = (location.hash || `#${FALLBACK_ROUTE}`).slice(1).trim();
    // A screen may address one of its records after '/', e.g. #skills/<id>;
    // the screen owner reads that part itself.
    const slash = raw.indexOf('/');
    return slash > 0 && window.SCREENS && window.SCREENS[raw.slice(0, slash)] ? raw.slice(0, slash) : raw;
  }
  function replaceHash(id) {
    const next = `${location.pathname}${location.search}#${id}`;
    if (location.hash !== `#${id}`) history.replaceState(null, '', next);
  }
  function resolveRoute(raw, opts = {}) {
    const id = normRoute(raw || FALLBACK_ROUTE);
    if (window.SCREENS[id]) {
      if (opts.rewrite && raw === 'datamap') replaceHash('dictionary/studies');
      else if (opts.rewrite && (raw === 'agent' || RETIRED_ROUTES.has(raw))) replaceHash(id);
      return { id, fallback: false };
    }
    const fallback = window.SCREENS[FALLBACK_ROUTE] ? FALLBACK_ROUTE : Object.keys(window.SCREENS)[0];
    if (opts.rewrite && fallback) replaceHash(fallback);
    return { id: fallback, fallback: true };
  }
  let route = resolveRoute(rawRouteFromHash(), { rewrite: true }).id;
  let activeRouteTransition = null;

  /* Guided Copilot -> module handoff payload. The guided screen set()s the
     backend handoff object before navigating; the target screen take()s it
     once (applying prefill in its own owner file) and note() keeps the
     "handed off from Copilot" banner alive for the visit until dismissed. */
  let guidedHandoff = null;
  let guidedHandoffNote = null;
  window.EU_GUIDED_HANDOFF = {
    set(h) { guidedHandoff = (h && typeof h === 'object') ? h : null; },
    take(routeId) {
      if (!guidedHandoff) return null;
      const target = String(guidedHandoff.target_route || '');
      if (routeId && target && target !== routeId) return null;
      const h = guidedHandoff;
      guidedHandoff = null;
      guidedHandoffNote = { route: routeId || target || '', handoff: h };
      return h;
    },
    note(routeId) {
      return guidedHandoffNote && guidedHandoffNote.route === routeId ? guidedHandoffNote.handoff : null;
    },
    noteHtml(routeId) {
      const h = this.note(routeId);
      if (!h) return '';
      const esc = (v) => String(v == null ? '' : v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
      const p = h.prefill || {};
      const T = window.t || ((en) => en);
      const shortPath = (v) => { const s = String(v || ''); return s.length > 34 ? '…' + s.slice(-33) : s; };
      const lbl = (en, zh, v) => v ? `${T(en, zh)} ${esc(v)}` : '';
      // Surface the full study design the user configured in Copilot — not just the
      // question — so the collected outcome / window / comparator / export destination
      // are visible on the target page instead of being silently dropped on handoff.
      const bits = [
        p.question_hint ? esc(p.question_hint) : '',
        lbl('Cohort', '队列', p.cohort_hint),
        lbl('Outcome', '结局', p.outcome_hint),
        lbl('Window', '时窗', p.time_window_hint),
        lbl('Compare', '对比', p.comparator_hint),
        lbl('Modules', '模块', p.module_hint),
        lbl('Save to', '保存到', p.export_destination_hint ? shortPath(p.export_destination_hint) : ''),
      ].filter(Boolean).join(' · ');
      const ic = window.icon ? window.icon('spark', 14) : '';
      return `<div class="note info" style="margin-bottom:14px;"><div class="ico">${ic}</div><div class="body"><span class="t">${T('Handed off from Guided Copilot.', '来自研究引导的交接。')}</span> <span class="d" style="display:inline;">${bits || T('Continue the study you configured in Copilot.', '继续你在研究引导里配置的研究。')}</span></div><button class="btn sm ghost" data-guided-prefill-dismiss type="button" style="margin-left:auto;align-self:center;">${T('Dismiss', '关闭')}</button></div>`;
    },
    dismiss() { guidedHandoffNote = null; },
  };
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-guided-prefill-dismiss]');
    if (!b) return;
    window.EU_GUIDED_HANDOFF.dismiss();
    if (window.__euRender) window.__euRender();
  });

  const L = (v) => Array.isArray(v) ? t(v[0], v[1]) : v;
  const CRUMB_LABELS = {
    Home: ['Home', '首页'],
    'Get Started': ['Get Started', '快速上手'],
    'Data Workspace': ['Data Workspace', '数据工作台'],
    /* One destination, one name: these zh labels must match each screen's own
       page title. Retired aliases include 患者明细 and 跨库基准. */
    'Patient Review': ['Patient Review', '患者审阅'],
    'Cohort Statistics': ['Cohort Statistics', '队列统计'],
    'Cross-database comparison': ['Cross-database comparison', '跨库对比'],
    Settings: ['Settings', '设置'],
  };
  const crumbLabel = (c) => L(CRUMB_LABELS[c] || c);
  function routeTitleOf(scr) {
    const heading = app.querySelector('h1');
    const headingText = heading && String(heading.textContent || '').trim();
    if (headingText) return headingText;
    const crumbs = scr && Array.isArray(scr.crumbs) ? scr.crumbs : [];
    return crumbs.length ? crumbLabel(crumbs[crumbs.length - 1]) : 'EasyICU';
  }
  function routeDocumentTitle(title) {
    return title === 'EasyICU' ? title : `${title} — EasyICU`;
  }
  function focusRouteContent() {
    const main = app.querySelector('main') || app;
    main.setAttribute('tabindex', '-1');
    const heading = main.querySelector && main.querySelector('h1');
    const target = heading || main;
    if (heading) heading.setAttribute('tabindex', '-1');
    if (target && typeof target.focus === 'function') {
      target.dataset.euRouteFocus = 'true';
      target.addEventListener('blur', () => { delete target.dataset.euRouteFocus; }, { once: true });
      try { target.focus({ preventScroll: true }); }
      catch (_) { target.focus(); }
    }
  }
  function announceRoute(title) {
    routeAnnouncer.textContent = '';
    const publish = () => { routeAnnouncer.textContent = title; };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(publish);
    else setTimeout(publish, 0);
  }
  const actionHtmlOf = (scr) => {
    if (!scr) return '';
    return typeof scr.actionHtml === 'function' ? scr.actionHtml() : (scr.actionHtml || '');
  };

  function syncShellAccessibility(root, fullScreen) {
    if (fullScreen && root.firstElementChild) {
      root.firstElementChild.setAttribute('role', 'main');
      root.firstElementChild.setAttribute('aria-label', t('Page content', '页面内容'));
    }
    const language = window.EU_LANG === 'zh' ? 'zh' : 'en';
    root.querySelectorAll('[data-lang]').forEach((control) => {
      control.setAttribute('aria-pressed', String(control.dataset.lang === language));
    });
  }

  function screenOf(id) { return window.SCREENS[id]; }

  function hostedInstance() {
    return ((window.EU_CAPABILITIES || {}).deployment || {}).hosted === true;
  }
  function render(opts = {}) {
    document.documentElement.toggleAttribute('data-eu-hosted', hostedInstance());
    const resetScroll = !!opts.resetScroll;
    const priorContent = app.querySelector('.content');
    const priorScrollOwner = priorContent || app.querySelector('.eudata-main') || app.querySelector('.eusk-main');
    const priorMain = app.querySelector('main');
    const priorActive = document.activeElement;
    const preserveRouteFocus = !!(
      priorMain && priorActive && priorMain.contains(priorActive)
      && (priorActive === priorMain || priorActive.tagName === 'H1')
    );
    const scrollState = {
      x: window.scrollX || 0,
      y: window.scrollY || 0,
      contentTop: priorScrollOwner ? priorScrollOwner.scrollTop : 0,
    };
    const scr = screenOf(route);
    const moduleShell = window.EU_DESKTOP_MODULE_SHELL;
    const sharedDataRoute = moduleShell && moduleShell.isDataRoute(route) && route !== 'extraction';
    const sharedSettingsRoute = moduleShell && route === 'settings';
    // Get Started sits in the module shell under Help; on a hosted instance the
    // aside carries the administrator's contact.
    const sharedHelpRoute = moduleShell && route === 'tutorial';
    if (sharedDataRoute) {
      app.innerHTML = moduleShell.renderData({
        active: route,
        wide: !!scr.wide,
        label: t('Data workspace', '数据工作台'),
        actions: actionHtmlOf(scr),
        content: scr.render(),
      });
    } else if (sharedSettingsRoute) {
      app.innerHTML = moduleShell.render({
        active: 'settings',
        shellClass: 'euh-settings-shell',
        mainClass: 'euh-settings-main',
        innerClass: 'euh-settings-main-inner',
        label: t('Settings', '设置'),
        content: `<div class="euh-settings-actions">${actionHtmlOf(scr)}</div>${scr.render()}`,
      });
    } else if (sharedHelpRoute) {
      app.innerHTML = moduleShell.render({
        active: 'tutorial',
        shellClass: 'euh-help-shell',
        mainClass: 'euh-settings-main',
        innerClass: 'euh-settings-main-inner',
        label: t('Get started', '快速上手'),
        aside: scr.contextAside ? scr.contextAside() : '',
        content: scr.render(),
      });
    } else if (scr.full || !moduleShell) {
      app.innerHTML = scr.render();
    } else {
      // Pages that draw no shell of their own sit in the module shell.
      app.innerHTML = moduleShell.render({ active: route, content: scr.render() });
    }
    if (scr.afterRender) scr.afterRender(app);
    syncShellAccessibility(app, !!scr.full);
    const title = routeTitleOf(scr);
    document.title = routeDocumentTitle(title);
    const c = app.querySelector('.content') || app.querySelector('.eudata-main') || app.querySelector('.eusk-main');
    if (resetScroll) {
      if (c) c.scrollTop = 0;
      window.scrollTo(0, 0);
      focusRouteContent();
      announceRoute(title);
    } else {
      if (c) c.scrollTop = scrollState.contentTop;
      window.scrollTo(scrollState.x, scrollState.y);
      if (preserveRouteFocus) focusRouteContent();
    }
  }

  function routeMotionAllowed() {
    if (typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(min-width: 1181px)').matches
      && !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function renderRoute(opts = {}) {
    if (!routeMotionAllowed()) {
      render(opts);
      return;
    }
    if (activeRouteTransition && typeof activeRouteTransition.skipTransition === 'function') {
      activeRouteTransition.skipTransition();
    }
    if (typeof document.startViewTransition !== 'function') {
      render(opts);
      const surface = app.querySelector(
        '.gd-main.gpi-workspace .gd-conv, .eusk-main, .eudata-main, .euh-settings-main, .euh-main'
      );
      if (surface) {
        surface.classList.add('eu-route-enter');
        surface.addEventListener('animationend', () => surface.classList.remove('eu-route-enter'), { once: true });
      }
      return;
    }
    document.documentElement.classList.add('eu-route-transitioning');
    const transition = document.startViewTransition(() => render(opts));
    activeRouteTransition = transition;
    const finish = () => {
      if (activeRouteTransition !== transition) return;
      activeRouteTransition = null;
      document.documentElement.classList.remove('eu-route-transitioning');
    };
    transition.finished.then(finish, finish);
  }

  window.__euRender = function (opts) { render(opts || {}); };

  app.addEventListener('click', (e) => {
    const langEl = e.target.closest('[data-lang]');
    if (langEl) { if (window.setLang) window.setLang(langEl.dataset.lang); return; }
    const langToggleEl = e.target.closest('[data-lang-toggle]');
    if (langToggleEl) { if (window.setLang) window.setLang(window.EU_LANG === 'zh' ? 'en' : 'zh'); return; }
    const goalEl = e.target.closest('[data-goal-cycle]');
    if (goalEl) { if (window.cycleGoal) window.cycleGoal(); return; }
    const navEl = e.target.closest('[data-nav]');
    if (navEl && navEl.dataset.nav) {
      let id = normRoute(navEl.dataset.nav);
      if (window.SCREENS[id]) {
        route = id;
        location.hash = '#' + id;
        renderRoute({ resetScroll: true });
        return;
      }
    }
    // expand/collapse handled by re-render via nav to first child already
  });

  window.addEventListener('hashchange', () => {
    const resolved = resolveRoute(rawRouteFromHash(), { rewrite: true });
    let r = resolved.id;
    const alias = window.__euAlias; window.__euAlias = false;
    if (window.SCREENS[r] && (r !== route || alias || resolved.fallback)) { route = r; renderRoute({ resetScroll: true }); }
  });

  /* ---- global keyboard shortcuts ---- */
  const SHORTCUT_SECTIONS = ['guided', 'extraction', 'patient', 'crossdb'];
  function goto(id) { if (window.SCREENS[id]) { route = id; location.hash = '#' + id; renderRoute({ resetScroll: true }); } }
  document.addEventListener('keydown', (e) => {
    const tgt = e.target;
    const typing = tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.tagName === 'SELECT' || tgt.isContentEditable);
    // ⌘K / Ctrl+K → open the single EasyICU Copilot conversation.
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      const guide = window.EUPageGuide || window.EUCopilot;
      if (guide) guide.toggle();
      return;
    }
    if (typing || e.metaKey || e.ctrlKey || e.altKey || hostedInstance()) return;
    // 1–4 → switch section
    if (e.key >= '1' && e.key <= '4') { goto(SHORTCUT_SECTIONS[+e.key - 1]); return; }
    // L → toggle language
    if (e.key === 'l' || e.key === 'L') { if (window.setLang) window.setLang(window.EU_LANG === 'zh' ? 'en' : 'zh'); }
  });

  render({ resetScroll: true });
})();
