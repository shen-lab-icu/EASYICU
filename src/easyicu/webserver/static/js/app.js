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

  /* The classic workspace pipeline. Copilot is a SEPARATE, parallel system that
     can complete the same flow conversationally — it is not a step in here. */
  const PHASES = [
    { n: 1, id: 'extraction', label: ['Extract the data', '抽取数据'], sub: ['cohort → analysis-ready', '队列 → 可分析数据'], ico: 'extract' },
    {
      n: 2, id: 'patient', label: ['Review & explore', '审阅与探索'], sub: ['patient · cohort · cross-DB', '患者 · 队列 · 跨库'], ico: 'viz',
      children: [
        { id: 'patient', label: ['Patient Review', '患者审阅'], ico: 'patient' },
        { id: 'cohort', label: ['Cohort Statistics', '队列统计'], ico: 'cohort' },
        { id: 'crossdb', label: ['Cross-DB Benchmark', '跨库基准'], ico: 'benchmark' },
      ],
    },
    { n: 3, id: 'agent', label: ['Analyze & draft', '分析与撰稿'], sub: ['runs → gated manuscript', '运行 → 受闸草稿'], ico: 'agent' },
  ];
  const VIZ_IDS = ['patient', 'cohort', 'crossdb'];
  /* User-facing data workspace: task-oriented labels, no migration-era "classic"
     terminology in the navigation. */
  const CLASSIC = [
    { id: 'extraction', label: ['Extract Data', '抽取数据'], sub: ['choose cohort + modules', '选择队列 + 模块'], ico: 'extract', stale: false },
    { id: 'patient', label: ['Patient Drilldown', '患者明细'], sub: ['single-patient review', '单患者审阅'], ico: 'patient', stale: true },
    { id: 'cohort', label: ['Cohort Review', '队列审阅'], sub: ['groups + coverage', '分组 + 覆盖率'], ico: 'cohort', stale: true },
    { id: 'crossdb', label: ['Cross-DB Compare', '跨库对比'], sub: ['multi-database checks', '多数据库检查'], ico: 'benchmark', stale: true },
  ];
  let classicOpen = true;
  function phaseOf(r) {
    if (r === 'extraction') return 1;
    if (VIZ_IDS.includes(r)) return 2;
    if (r === 'agent') return 3;
    return 0;
  }

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
    'Data Visualization': ['Data Visualization', '数据可视化'],
    'Patient Review': ['Patient Review', '患者明细'],
    'Cohort Statistics': ['Cohort Statistics', '队列统计'],
    'Cross-DB Benchmark': ['Cross-DB Benchmark', '跨库对比'],
    Settings: ['Settings', '设置'],
    'Workspace States': ['Workspace States', '工作区状态'],
  };
  const crumbLabel = (c) => L(CRUMB_LABELS[c] || c);
  const actionHtmlOf = (scr) => {
    if (!scr) return '';
    return typeof scr.actionHtml === 'function' ? scr.actionHtml() : (scr.actionHtml || '');
  };

  function screenOf(id) { return window.SCREENS[id]; }
  function sectionOf(id) { return screenOf(id).section; }

  function sidebar() {
    const scr = screenOf(route);
    const workspaceIndex = CLASSIC.findIndex(c => c.id === route);
    const classicActive = CLASSIC.some(c => c.id === route);
    const wsOpen = classicOpen || classicActive;

    const rail = scr.rail ? scr.rail() : '';
    const progLabel = workspaceIndex >= 0 ? `${workspaceIndex + 1} / ${CLASSIC.length}` : `· / ${CLASSIC.length}`;

    return `
    <aside class="sidebar">
      <div class="brand" data-nav="entry">
        <div class="mark">${icon('flask', 18)}</div>
        <div><div class="name">EasyICU</div><div class="tag">${t('ICU Research Workspace', 'ICU 研究工作台')}</div></div>
      </div>
      <button class="cp-entry ${route === 'guided' ? 'on' : ''}" data-nav="guided">
        <span class="cp-ico">${icon('spark', 16)}</span>
        <span class="cp-body"><span class="cp-t">${t('Guided study', '研究引导')}</span><span class="cp-d">${window.EU_HASWORK ? t('continue the current workflow by chat', '用对话继续当前流程') : t('plan a study by conversation', '用对话规划研究')}</span></span>
        <span class="cp-go">${icon('arrow', 14)}</span>
      </button>
      <div class="shared-note"><span class="ico">${icon('target', 11)}</span><span>${t('Have a paper or question? Start with Idea Mining. Already have data? Start with Extract Data.', '有文章或问题，从 Idea 挖掘开始；已有数据，从抽取数据开始。')}</span></div>
      <div class="sec-label nav-sec">${t('Discovery & Plan', '发现与计划')}</div>
      <button class="cp-entry ideas-entry ${route === 'ideas' ? 'on' : ''}" data-nav="ideas">
        <span class="cp-ico">${icon('target', 16)}</span>
        <span class="cp-body"><span class="cp-t">${t('Find a Study Idea', '找研究想法')}</span><span class="cp-d">${t('paper, PDF, or topic → feasible plan', '文章、PDF 或主题 → 可行计划')}</span></span>
        <span class="cp-go">${icon('arrow', 14)}</span>
      </button>
      <button class="cp-entry agent-entry ${route === 'agent' ? 'on' : ''}" data-nav="agent">
        <span class="cp-ico">${icon('agent', 16)}</span>
        <span class="cp-body"><span class="cp-t">${t('Run a Research Project', '运行研究项目')}</span><span class="cp-d">${t('confirmed plan → evidence-gated draft', '确认计划 → 证据闸草稿')}</span></span>
        <span class="cp-go">${icon('arrow', 14)}</span>
      </button>
      <div class="sec-label nav-sec">${t('Data & Review', '数据与审阅')}</div>
      <div class="wsnav">
        <button class="wsgroup-head ${wsOpen ? 'open' : ''} ${classicActive ? 'active' : ''}" data-ws-toggle>
          <span class="wsg-ico">${icon('grid', 15)}</span>
          <span class="wsg-t">${t('Data Workspace', '数据工作台')}</span>
          <span class="wsg-prog">${progLabel}</span>
          <span class="wsg-chev">${icon(wsOpen ? 'chevdown' : 'chevron', 13)}</span>
        </button>
        ${wsOpen ? `
        <div class="wsg-children">
          ${CLASSIC.map(c => `
            <button class="wsitem ${route === c.id ? 'active' : ''}" data-nav="${c.id}">
              <span class="ico">${icon(c.ico, 15)}</span>
              <span class="wsi-copy"><span class="wsi-t">${L(c.label)}${(window.EU_STALE && c.stale) ? ` <span class="jp-stale">${icon('refresh', 9)} ${t('stale', '过期')}</span>` : ''}</span><span class="wsi-sub">${L(c.sub)}</span></span>
            </button>`).join('')}
        </div>` : ''}
      </div>
      <div class="sec-label" style="margin:16px 0 6px;">${t('Reference', '参考')}</div>
      <div class="nav" style="padding-top:0;">
        <div class="nav-item ${route === 'tutorial' ? 'active' : ''}" data-nav="tutorial"><span class="ico">${icon('help', 17)}</span>${t('Get Started', '快速上手')}</div>
        <div class="nav-item ${route === 'dictionary' ? 'active' : ''}" data-nav="dictionary"><span class="ico">${icon('list', 17)}</span>${t('Data Dictionary', '数据字典')}</div>
        <div class="nav-item ${route === 'settings' ? 'active' : ''}" data-nav="settings"><span class="ico">${icon('gear', 17)}</span>${t('Settings', '设置')}</div>
      </div>
      ${rail}
      <div class="rail-spacer"></div>
      <div class="rail-foot">
        <div class="icobtn" title="Back" data-nav="entry">${icon('back', 16)}</div>
        <div class="icobtn ${route === 'tutorial' ? 'on' : ''}" title="Help" data-nav="tutorial">${icon('help', 16)}</div>
        <div class="icobtn ${route === 'settings' ? 'on' : ''}" title="Settings" data-nav="settings">${icon('gear', 16)}</div>
        <div class="icobtn" title="Language" data-lang-toggle>${icon('globe', 16)}</div>
        <div class="avatar">LK</div>
      </div>
    </aside>`;
  }

  // middle breadcrumbs that map to a real destination become live links again
  const CRUMB_NAV = {
    'Data Visualization': 'patient', '数据可视化': 'patient',
  };
  function topbar() {
    const scr = screenOf(route);
    const crumbs = (scr.crumbs || []).map((c, i, arr) => {
      const label = crumbLabel(c);
      if (i === arr.length - 1) return `<span class="cur">${label}</span>`;
      let node;
      if (i === 0) node = `<a data-nav="entry">${label}</a>`;
      else if (CRUMB_NAV[c]) node = `<a data-nav="${CRUMB_NAV[c]}">${label}</a>`;
      else node = `<span class="mid">${label}</span>`;
      return `${node}<span class="sep">/</span>`;
    }).join(' ');
    const actionHtml = actionHtmlOf(scr);
    return `
    <div class="topbar">
      <div class="crumbs">${crumbs}</div>
      <div class="spacer"></div>
      ${scr.status || ''}
      <div class="mode-seg" role="group" aria-label="Data mode" title="${t('Data source for the whole workspace', '整个工作台的数据源')}">
        <button class="${window.EU_DATA !== 'real' ? 'on' : ''}" data-datamode="demo">${icon('flask', 12)} ${t('Demo', '演示')}</button>
        <button class="${window.EU_DATA === 'real' ? 'on' : ''}" data-datamode="real">${icon('db', 12)} ${t('Real', '真实')}</button>
      </div>
      <div class="lang-seg" role="group" aria-label="Language">
        <button class="${window.EU_LANG !== 'zh' ? 'on' : ''}" data-lang="en">EN</button>
        <button class="${window.EU_LANG === 'zh' ? 'on' : ''}" data-lang="zh">中</button>
      </div>
      <button class="btn sm" data-cpopen title="${t('Open page guide for this screen', '打开当前页面指南')}">${icon('spark', 13)} ${t('Page guide','页面指南')}</button>
      ${actionHtml}
    </div>`;
  }

  function mobileChrome() {
    const scr = screenOf(route);
    const title = scr.crumbs && scr.crumbs.length ? crumbLabel(scr.crumbs[scr.crumbs.length - 1]) : 'EasyICU';
    const actionHtml = actionHtmlOf(scr);
    const top = `
      <div class="mtopbar">
        <div class="mark" data-nav="entry">${icon('flask', 16)}</div>
        <div class="name">${title}</div>
        <div class="spacer"></div>
        <button class="btn sm icon" data-cpopen title="${t('Page guide', '页面指南')}">${icon('spark', 16)}</button>
        ${actionHtml}
      </div>`;
    const bottom = `
      <nav class="mbottomnav">
        ${MOBILE_NAV.map(n => `
          <button data-nav="${n.id}" class="${route === n.id ? 'active' : ''}">
            <span class="ico">${icon(n.ico, 19)}</span>${L(n.label)}
          </button>`).join('')}
      </nav>`;
    return { top, bottom };
  }

  function render() {
    window.__euRender = render;  // expose so api.js can re-render after live data loads
    const scr = screenOf(route);
    if (scr.full) {
      app.innerHTML = scr.render();
    } else {
      const m = mobileChrome();
      app.innerHTML = `
        <div class="app">
          ${sidebar()}
          <div class="main">
            ${m.top}
            ${topbar()}
            <div class="content ${scr.wide ? 'wide' : ''}">${scr.render()}</div>
          </div>
          ${m.bottom}
        </div>`;
    }
    if (scr.afterRender) scr.afterRender(app);
    // restore scroll top
    const c = app.querySelector('.content');
    if (c) c.scrollTop = 0;
    window.scrollTo(0, 0);
  }

  window.__euRender = render;

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
        render();
        return;
      }
    }
    // expand/collapse handled by re-render via nav to first child already
  });

  window.addEventListener('hashchange', () => {
    const resolved = resolveRoute(rawRouteFromHash(), { rewrite: true });
    let r = resolved.id;
    const alias = window.__euAlias; window.__euAlias = false;
    if (window.SCREENS[r] && (r !== route || alias || resolved.fallback)) { route = r; render(); }
  });

  /* ---- global keyboard shortcuts (advertised on Get Started) ---- */
  const SHORTCUT_SECTIONS = ['ideas', 'extraction', 'patient', 'crossdb', 'agent'];
  function goto(id) { if (window.SCREENS[id]) { route = id; location.hash = '#' + id; render(); } }
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

  render();
})();
