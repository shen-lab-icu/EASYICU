/* App shell controller */
(function () {
  const app = document.getElementById('app');
  /* Route normalization: Copilot merge + folded Review destinations.
     Coverage Audit / SOFA Reclassification are now tabs inside Cohort Statistics;
     ICD Cohort Filter is folded into the extraction cohort filter. Old deep
     links still resolve to the right place. */
  const FALLBACK_ROUTE = 'entry';

  function normRoute(r) {
    if (r === 'help') return 'tutorial';
    if (r === 'assistant') return 'guided';
    if (r === 'audit')       { window.__euCohortPanel = 'coverage'; window.__euAlias = true; return 'cohort'; }
    if (r === 'sofareclass') { window.__euCohortPanel = 'sofa';     window.__euAlias = true; return 'cohort'; }
    if (r === 'icd')         { window.__euExtractFocusICD = true;   window.__euAlias = true; return 'extraction'; }
    return r;
  }
  function rawRouteFromHash() {
    return (location.hash || `#${FALLBACK_ROUTE}`).slice(1).trim();
  }
  function replaceHash(id) {
    const next = `${location.pathname}${location.search}#${id}`;
    if (location.hash !== `#${id}`) history.replaceState(null, '', next);
  }
  function resolveRoute(raw, opts = {}) {
    const id = normRoute(raw || FALLBACK_ROUTE);
    if (window.SCREENS[id]) return { id, fallback: false };
    const fallback = window.SCREENS[FALLBACK_ROUTE] ? FALLBACK_ROUTE : Object.keys(window.SCREENS)[0];
    if (opts.rewrite && fallback) replaceHash(fallback);
    return { id: fallback, fallback: true };
  }
  let route = resolveRoute(rawRouteFromHash(), { rewrite: true }).id;

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

  /* User-facing data workspace. Labels match each screen's own page title and
     breadcrumb so a destination has ONE name across the sidebar, header, and
     home entries (no sidebar-vs-title drift). Copilot is a separate, parallel
     system that completes the same flow conversationally. */
  const CLASSIC = [
    { id: 'extraction', label: ['Data Extraction', '数据抽取'], sub: ['choose cohort + modules', '选择队列 + 模块'], ico: 'extract' },
    { id: 'patient', label: ['Patient Review', '患者审阅'], sub: ['tables · trends · patients', '表格 · 趋势 · 患者'], ico: 'patient' },
    { id: 'cohort', label: ['Cohort Statistics', '队列统计'], sub: ['groups + coverage', '分组 + 覆盖率'], ico: 'cohort' },
    { id: 'crossdb', label: ['Cross-database comparison', '跨库对比'], sub: ['coverage + distributions', '覆盖率 + 分布'], ico: 'benchmark' },
  ];
  let classicOpen = true;

  const MOBILE_NAV = [
    { id: 'guided', label: ['Guide', '引导'], ico: 'spark' },
    { id: 'ideas', label: ['Ideas', '想法'], ico: 'target' },
    { id: 'extraction', label: ['Extract', '抽取'], ico: 'extract' },
    { id: 'patient', label: ['Review', '审阅'], ico: 'patient' },
    { id: 'agent', label: ['Analyze', '分析'], ico: 'agent' },
  ];
  const L = (v) => Array.isArray(v) ? t(v[0], v[1]) : v;
  const CRUMB_LABELS = {
    Home: ['Home', '首页'],
    'Get Started': ['Get Started', '快速上手'],
    'Idea Mining': ['Idea Mining', '想法挖掘'],
    'Data Workspace': ['Data Workspace', '数据工作台'],
    /* One destination, one name: these zh labels must match the sidebar CLASSIC
       labels and each screen's own page title. Retired aliases include 患者明细
       and 跨库基准. */
    'Patient Review': ['Patient Review', '患者审阅'],
    'Cohort Statistics': ['Cohort Statistics', '队列统计'],
    'Cross-database comparison': ['Cross-database comparison', '跨库对比'],
    'Data Extraction': ['Data Extraction', '数据抽取'],
    'Agent Projects': ['Agent Projects', '研究项目'],
    'Data Dictionary': ['Data Dictionary', '数据字典'],
    'Guided Copilot': ['Guided Copilot', '研究引导'],
    Settings: ['Settings', '设置'],
    'Workspace States': ['Workspace States', '工作区状态'],
  };
  const crumbLabel = (c) => L(CRUMB_LABELS[c] || c);
  window.EU_CRUMB_LABEL = crumbLabel;

  /* The one catalogue of destinations the surface can summon. Entries carry a
     crumb KEY, never a label, so a rail cannot mint a third alias for a screen
     that already names itself — inventing "Ideas" next to "Idea Mining" is the
     exact drift "one destination, one name" exists to stop. Order is the
     reading order of a study, not alphabetical. */
  const DESTINATIONS = [
    { id: 'ideas', crumb: 'Idea Mining', ico: 'target' },
    { id: 'extraction', crumb: 'Data Extraction', ico: 'extract' },
    { id: 'patient', crumb: 'Patient Review', ico: 'patient' },
    { id: 'cohort', crumb: 'Cohort Statistics', ico: 'cohort' },
    { id: 'crossdb', crumb: 'Cross-database comparison', ico: 'benchmark' },
    { id: 'agent', crumb: 'Agent Projects', ico: 'agent' },
    { id: 'dictionary', crumb: 'Data Dictionary', ico: 'list' },
    { id: 'states', crumb: 'Workspace States', ico: 'layers' },
  ];
  window.EU_DESTINATIONS = () => DESTINATIONS.map(
    (d) => ({ id: d.id, ico: d.ico, label: crumbLabel(d.crumb) }),
  );

  /* Page guide is shell-global — it explains whichever screen you are looking
     at — so the shell keeps owning it even though the surface decides where to
     hang it. It is spelled out against Guided Copilot because the two were read
     as the same thing: one is a help overlay, the other is the conversation. */
  window.EU_SHELL_CONTROLS = () => `
    <button type="button" class="one-ghost" data-cpopen
      title="${t('Page guide: what this screen does and its safe shortcuts. Not Guided Copilot, which is the conversation that plans the study.', '页面指南：说明当前界面能做什么、有哪些安全快捷操作。它不是研究引导——研究引导是规划研究的那段对话。')}"
      aria-label="${t('Page guide', '页面指南')}">${icon('spark', 13)}<span>${t('Page guide', '页面指南')}</span></button>`;
  const actionHtmlOf = (scr) => {
    if (!scr) return '';
    return typeof scr.actionHtml === 'function' ? scr.actionHtml() : (scr.actionHtml || '');
  };
  const displayedDataMode = () => (
    window.getDataMode
      ? window.getDataMode()
      : (window.EU_DATA === 'real' ? 'real' : 'demo')
  );
  /* The surface renders the data-mode control but must not re-derive the mode:
     a second copy of this expression is how "demo" and "real" drift apart. */
  window.EU_DISPLAYED_DATA_MODE = displayedDataMode;

  function syncShellAccessibility(root, fullScreen) {
    if (fullScreen && root.firstElementChild) {
      root.firstElementChild.setAttribute('role', 'main');
      root.firstElementChild.setAttribute('aria-label', t('Page content', '页面内容'));
    }
    root.querySelectorAll('.mbottomnav [data-nav]').forEach((control) => {
      /* A group link is current for any of its children, which this exact-id
         comparison cannot see; topnav() already marked it. */
      if (control.dataset.navgroup) return;
      const current = control.dataset.nav === route;
      if (current) control.setAttribute('aria-current', 'page');
      else control.removeAttribute('aria-current');
    });
    const dataMode = displayedDataMode();
    root.querySelectorAll('[data-datamode], [data-hd]').forEach((control) => {
      const value = control.dataset.datamode || control.dataset.hd;
      control.setAttribute('aria-pressed', String(value === dataMode));
    });
    const language = window.EU_LANG === 'zh' ? 'zh' : 'en';
    root.querySelectorAll('[data-lang]').forEach((control) => {
      control.setAttribute('aria-pressed', String(control.dataset.lang === language));
    });
  }

  function screenOf(id) { return window.SCREENS[id]; }
  function sectionOf(id) { return screenOf(id).section; }

  function mobileChrome() {
    const scr = screenOf(route);
    const title = scr.crumbs && scr.crumbs.length ? crumbLabel(scr.crumbs[scr.crumbs.length - 1]) : 'EasyICU';
    const actionHtml = actionHtmlOf(scr);
    const top = `
      <header class="mtopbar" aria-label="${t('Mobile page controls', '移动端页面控制')}">
        <button type="button" class="mark" data-nav="entry" aria-label="${t('Back to home', '返回首页')}">${icon('flask', 16)}</button>
        <div class="name">${title}</div>
        <div class="spacer"></div>
        <button type="button" class="btn sm icon" data-cpopen title="${t('Page guide', '页面指南')}" aria-label="${t('Page guide', '页面指南')}">${icon('spark', 16)}</button>
        ${actionHtml}
      </header>`;
    const bottom = `
      <nav class="mbottomnav" aria-label="${t('Mobile navigation', '移动端导航')}">
        ${MOBILE_NAV.map(n => `
          <button type="button" data-nav="${n.id}" class="${route === n.id ? 'active' : ''}">
            <span class="ico">${icon(n.ico, 19)}</span>${L(n.label)}
          </button>`).join('')}
      </nav>`;
    return { top, bottom };
  }

  /* One surface. The conversation ships its own three columns, so it IS the
     surface — nothing wraps it. `route` names which panel is mounted into its
     aside, so existing deep links (#patient, #crossdb) keep resolving without
     a navigation menu existing. Panels render each screen's own render(). */
  function render(opts = {}) {
    const resetScroll = !!opts.resetScroll;
    const one = window.EU_ONE;
    const chat = window.SCREENS.guided || screenOf(route);
    const panelId = one && one.isPanel(route) ? route : '';
    const priorBody = app.querySelector('.one-panelbody');
    const priorTop = priorBody ? priorBody.scrollTop : 0;
    app.innerHTML = chat.render();
    if (chat.afterRender) chat.afterRender(app);
    if (one) one.mount(app, panelId);
    syncShellAccessibility(app, true);
    const body = app.querySelector('.one-panelbody');
    if (body) body.scrollTop = resetScroll ? 0 : priorTop;
  }

  window.__euRender = function (opts) { render(opts || {}); };

  app.addEventListener('click', (e) => {
    const langEl = e.target.closest('[data-lang]');
    if (langEl) { if (window.setLang) window.setLang(langEl.dataset.lang); return; }
    const langToggleEl = e.target.closest('[data-lang-toggle]');
    if (langToggleEl) { if (window.setLang) window.setLang(window.EU_LANG === 'zh' ? 'en' : 'zh'); return; }
    const dmEl = e.target.closest('[data-datamode]');
    if (dmEl) { if (window.setDataMode) window.setDataMode(dmEl.dataset.datamode); return; }
    const goalEl = e.target.closest('[data-goal-cycle]');
    if (goalEl) { if (window.cycleGoal) window.cycleGoal(); return; }
    const wsEl = e.target.closest('[data-ws-toggle]');
    if (wsEl) { classicOpen = !classicOpen; render(); return; }
    const cpEl = e.target.closest('[data-cpopen]');
    if (cpEl) {
      const guide = window.EUPageGuide || window.EUCopilot;
      if (guide) guide.toggle();
      return;
    }
    const navEl = e.target.closest('[data-nav]');
    if (navEl && navEl.dataset.nav) {
      let id = normRoute(navEl.dataset.nav);
      if (window.SCREENS[id]) {
        route = id;
        location.hash = '#' + id;
        render({ resetScroll: true });
        return;
      }
    }
    // expand/collapse handled by re-render via nav to first child already
  });

  window.addEventListener('hashchange', () => {
    const resolved = resolveRoute(rawRouteFromHash(), { rewrite: true });
    let r = resolved.id;
    const alias = window.__euAlias; window.__euAlias = false;
    if (window.SCREENS[r] && (r !== route || alias || resolved.fallback)) { route = r; render({ resetScroll: true }); }
  });

  /* ---- global keyboard shortcuts (advertised on Get Started) ---- */
  const SHORTCUT_SECTIONS = ['ideas', 'extraction', 'patient', 'crossdb', 'agent'];
  function goto(id) { if (window.SCREENS[id]) { route = id; location.hash = '#' + id; render({ resetScroll: true }); } }
  document.addEventListener('keydown', (e) => {
    const tgt = e.target;
    const typing = tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable);
    // ⌘K / Ctrl+K → open the Page guide command surface (works even while typing)
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      const guide = window.EUPageGuide || window.EUCopilot;
      if (guide) guide.toggle();
      return;
    }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    // 1–5 → switch section
    if (e.key >= '1' && e.key <= '5') { goto(SHORTCUT_SECTIONS[+e.key - 1]); return; }
    // L → toggle language
    if (e.key === 'l' || e.key === 'L') { if (window.setLang) window.setLang(window.EU_LANG === 'zh' ? 'en' : 'zh'); }
  });

  render({ resetScroll: true });
})();
