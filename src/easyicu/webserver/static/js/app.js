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
    Settings: ['Settings', '设置'],
    'Workspace States': ['Workspace States', '工作区状态'],
  };
  const crumbLabel = (c) => L(CRUMB_LABELS[c] || c);
  const actionHtmlOf = (scr) => {
    if (!scr) return '';
    return typeof scr.actionHtml === 'function' ? scr.actionHtml() : (scr.actionHtml || '');
  };
  const displayedDataMode = () => (
    window.getDataMode
      ? window.getDataMode()
      : (window.EU_DATA === 'real' ? 'real' : 'demo')
  );

  function syncShellAccessibility(root, fullScreen) {
    if (fullScreen && root.firstElementChild) {
      root.firstElementChild.setAttribute('role', 'main');
      root.firstElementChild.setAttribute('aria-label', t('Page content', '页面内容'));
    }
    root.querySelectorAll('.cs-nav [data-nav], .cs-sub [data-nav], .mbottomnav [data-nav]').forEach((control) => {
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

  /* ---- shell -------------------------------------------------------
     Top nav + centred column. Every screen still renders through
     scr.render(); the swap changed where the shell puts things, not what a
     screen produces. The data-nav / data-datamode / data-lang / data-cpopen
     hooks below are the compatibility contract with the click delegation and
     syncShellAccessibility further down — renaming one silently breaks
     navigation, so they are kept verbatim from the sidebar era. */

  function topnav() {
    const scr = screenOf(route);
    const dataMode = displayedDataMode();
    const demoMode = dataMode === 'demo';
    const officialDemo = demoMode
      && window.EU_DATA_MODE_CONTEXT
      && window.EU_DATA_MODE_CONTEXT.kind === 'official_demo';
    const modeTitle = demoMode
      ? t('Demo mode uses official public deidentified demo datasets or clearly labelled seeded examples. It is never your local data; switch to Real to load a local export.', '演示模式使用官方公开去标识 Demo 数据集，或明确标注的种子示例；都不是你的本地数据。切换到真实模式可加载本地导出。')
      : t('Real mode: screens compute from your local EasyICU export. Nothing is uploaded.', '真实模式：各页面从你本地的 EasyICU 导出计算，不上传任何数据。');
    /* Destinations are spelled out rather than mapped from NAV so that
       `data-nav="ideas"` and friends stay greppable. Guard tests assert the
       literal wiring in this source, and a template hole would let a
       destination silently disappear while the tests still passed. */
    const cur = (id) => (route === id ? ' aria-current="page"' : '');
    const dataCur = CLASSIC.some(c => c.id === route) ? ' aria-current="page"' : '';
    const links = `
      <button type="button" class="cs-link" data-nav="guided"${cur('guided')}>${t('Guided Copilot', '研究引导')}</button>
      <button type="button" class="cs-link" data-nav="extraction" data-navgroup="1"${dataCur}>${t('Data Workspace', '数据工作台')}</button>
      <button type="button" class="cs-link" data-nav="agent"${cur('agent')}>${t('Agent Projects', '研究项目')}</button>
      <button type="button" class="cs-link" data-nav="ideas"${cur('ideas')}>${t('Idea Mining', '想法挖掘')}</button>`;
    return `
    <nav class="cs-nav" aria-label="${t('Primary navigation', '主导航')}">
      <button type="button" class="cs-brand" data-nav="entry">
        ${icon('flask', 17)}<span>EasyICU</span>
      </button>
      <div class="cs-links">${links}</div>
      <div class="cs-spacer"></div>
      <div class="cs-right">
        ${scr.status || ''}
        <div class="mode-seg ${demoMode ? 'demo-active' : ''}" role="group"
          aria-label="Data mode" title="${modeTitle}">
          <button type="button" class="${demoMode ? 'on' : ''}" data-datamode="demo"
            aria-pressed="${demoMode}">${icon('flask', 12)} ${officialDemo ? t('Official demo', '官方演示') : (demoMode ? t('Demo data', '演示数据') : t('Demo', '演示'))}</button>
          <button type="button" class="${!demoMode ? 'on' : ''}" data-datamode="real"
            aria-pressed="${!demoMode}">${icon('db', 12)} ${t('Real', '真实')}</button>
        </div>
        <div class="lang-seg" role="group" aria-label="Language">
          <button type="button" class="${window.EU_LANG !== 'zh' ? 'on' : ''}" data-lang="en"
            aria-pressed="${window.EU_LANG !== 'zh'}">EN</button>
          <button type="button" class="${window.EU_LANG === 'zh' ? 'on' : ''}" data-lang="zh"
            aria-pressed="${window.EU_LANG === 'zh'}">中</button>
        </div>
        <button type="button" class="cs-guide" data-cpopen
          title="${t('Open page guide for this screen', '打开当前页面指南')}">${icon('spark', 13)} ${t('Page guide', '页面指南')}</button>
        <button type="button" class="cs-link" data-nav="dictionary"
          ${route === 'dictionary' ? 'aria-current="page"' : ''}>${t('Dictionary', '字典')}</button>
        <button type="button" class="cs-link" data-nav="settings"
          ${route === 'settings' ? 'aria-current="page"' : ''}>${t('Settings', '设置')}</button>
      </div>
    </nav>`;
  }

  /* The four data destinations used to be a collapsible sidebar group. They
     become a sub-tab row so no destination lost its entry point in the swap. */
  function subnav() {
    if (CLASSIC.every(c => c.id !== route)) return '';
    return `
    <div class="cs-sub" role="group" aria-label="${t('Data Workspace', '数据工作台')}">
      ${CLASSIC.map(c => `
        <button type="button" class="cs-subitem ${route === c.id ? 'on' : ''}"
          data-nav="${c.id}" ${route === c.id ? 'aria-current="page"' : ''}>${L(c.label)}</button>`).join('')}
    </div>`;
  }

  function pagehead() {
    const scr = screenOf(route);
    const crumbs = (scr.crumbs || []).map((c, i, arr) => {
      const label = crumbLabel(c);
      if (i === arr.length - 1) return `<span class="cur" aria-current="page">${label}</span>`;
      if (i === 0) return `<button type="button" class="crumb-link" data-nav="entry">${label}</button><span class="sep">/</span>`;
      return `<span class="mid">${label}</span><span class="sep">/</span>`;
    }).join(' ');
    const actionHtml = actionHtmlOf(scr);
    const title = scr.crumbs && scr.crumbs.length
      ? crumbLabel(scr.crumbs[scr.crumbs.length - 1])
      : '';
    return `
    <div class="cs-head">
      ${crumbs ? `<div class="cs-crumbs">${crumbs}</div>` : ''}
      ${title || actionHtml ? `
      <div class="cs-title">
        ${title ? `<h1>${title}</h1>` : ''}
        ${actionHtml ? `<div class="cs-actions">${actionHtml}</div>` : ''}
      </div>` : ''}
      ${subnav()}
    </div>`;
  }

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

  function render(opts = {}) {
    const resetScroll = !!opts.resetScroll;
    const priorContent = app.querySelector('.content');
    const scrollState = {
      x: window.scrollX || 0,
      y: window.scrollY || 0,
      contentTop: priorContent ? priorContent.scrollTop : 0,
    };
    const scr = screenOf(route);
    if (scr.full) {
      app.innerHTML = scr.render();
    } else {
      const m = mobileChrome();
      /* The rail moved from the sidebar into a horizontal strip. Screens still
         supply it through the same scr.rail() contract. */
      const rail = scr.rail ? scr.rail() : '';
      app.innerHTML = `
        <div class="app console">
          ${m.top}
          ${topnav()}
          <main class="main" aria-label="${t('Page content', '页面内容')}">
            <div class="cs-col ${scr.wide ? 'wide' : ''}">
              ${pagehead()}
              ${rail ? `<div class="cs-strip">${rail}</div>` : ''}
              <div class="content ${scr.wide ? 'wide' : ''}">${scr.render()}</div>
            </div>
          </main>
          ${m.bottom}
        </div>`;
    }
    if (scr.afterRender) scr.afterRender(app);
    syncShellAccessibility(app, !!scr.full);
    const c = app.querySelector('.content');
    if (resetScroll) {
      if (c) c.scrollTop = 0;
      window.scrollTo(0, 0);
    } else {
      if (c) c.scrollTop = scrollState.contentTop;
      window.scrollTo(scrollState.x, scrollState.y);
    }
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
