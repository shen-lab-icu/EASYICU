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
  /* Extraction produces the export the other three read. Listed as four peers,
     a new user clicks Patient Review first and meets an empty state; the order
     of operations was only discoverable by tripping over it. `role` renders
     that dependency in the nav instead. */
  const CLASSIC = [
    { id: 'extraction', label: ['Data Extraction', '数据抽取'], sub: ['choose cohort + modules', '选择队列 + 模块'], ico: 'extract', role: 'produces' },
    { id: 'patient', label: ['Patient Review', '患者审阅'], sub: ['tables · trends · patients', '表格 · 趋势 · 患者'], ico: 'patient', role: 'reads' },
    { id: 'cohort', label: ['Cohort Statistics', '队列统计'], sub: ['groups + coverage', '分组 + 覆盖率'], ico: 'cohort', role: 'reads' },
    { id: 'crossdb', label: ['Cross-database comparison', '跨库对比'], sub: ['coverage + distributions', '覆盖率 + 分布'], ico: 'benchmark', role: 'reads' },
  ];
  let classicOpen = true;

  const MOBILE_NAV = [
    { id: 'guided', label: ['Guide', '引导'], ico: 'spark' },
    { id: 'ideas', label: ['Ideas', '想法'], ico: 'target' },
    { id: 'extraction', label: ['Extract', '抽取'], ico: 'extract' },
    { id: 'patient', label: ['Review', '审阅'], ico: 'patient' },
    { id: 'agent', label: ['Monitor', '监控'], ico: 'agent' },
  ];
  const L = (v) => Array.isArray(v) ? t(v[0], v[1]) : v;
  /* A button's accessible name is its concatenated text content, so a title
     span followed by a sublabel span was announced as one run-together string
     ("Patient Reviewtables · trends · patients"). Rejoin them with a real
     separator instead of hiding the sublabel — it carries information the
     sighted user gets. The visible title stays first so voice control still
     matches on it (WCAG 2.5.3 Label in Name). */
  const navLabel = (label, sub) => `${L(label)} — ${L(sub)}`;
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
    root.querySelectorAll('.sidebar [data-nav], .mbottomnav [data-nav]').forEach((control) => {
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

  /* The three nav sections are the product's main line (plan -> data -> draft),
     but they rendered as static categories, so nothing told the user how far
     they had got or what was still missing. study-progress.js derives the
     state from the active study context and exposes a small snapshot; the
     shell renders that and never reads the context store itself (a layering
     rule locked by test_route_handoffs_..._has_its_own_owner). */
  function stageChip(stageId, progress) {
    const state = (progress.byId || {})[stageId] || 'todo';
    const label = {
      done: t('done', '已完成'),
      active: t('in progress', '进行中'),
      todo: t('not started', '未开始'),
    }[state];
    const stale = state === 'done' && progress.stale;
    const shown = stale ? t('out of date', '已过期') : label;
    return `<span class="stage-chip ${stale ? 'stale' : state}" title="${shown}">${state === 'done' && !stale ? icon('check', 10) : ''}${shown}</span>`;
  }

  function navSection(stageId, labelEn, labelZh, progress) {
    return `<div class="sec-label nav-sec stage-sec"><span>${t(labelEn, labelZh)}</span>${progress.started ? stageChip(stageId, progress) : ''}</div>`;
  }

  function sidebar() {
    const scr = screenOf(route);
    const classicActive = CLASSIC.some(c => c.id === route);
    const wsOpen = classicOpen || classicActive;
    const progress = (window.EU_STUDY_PROGRESS && window.EU_STUDY_PROGRESS.snapshot())
      || { started: false, stale: false, planOnly: false, stages: [], byId: {} };

    const rail = scr.rail ? scr.rail() : '';
    const guidedSub = window.EU_HASWORK
      ? t('continue the current workflow by chat', '用对话继续当前流程')
      : t('plan a study by conversation', '用对话规划研究');

    return `
    <aside class="sidebar" aria-label="${t('Application sidebar', '应用侧边栏')}">
      <button type="button" class="brand" data-nav="entry" aria-label="${navLabel('EasyICU', t('ICU Research Workspace', 'ICU 研究工作台'))}">
        <span class="mark">${icon('flask', 18)}</span>
        <span><span class="name">EasyICU</span><span class="tag">${t('ICU Research Workspace', 'ICU 研究工作台')}</span></span>
      </button>
      <nav class="shell-nav" aria-label="${t('Primary navigation', '主导航')}">
      ${navSection('discovery', 'Discovery & Plan', '发现与计划', progress)}
      <button type="button" class="cp-entry ${route === 'guided' ? 'on' : ''}" data-nav="guided" aria-label="${navLabel(t('EasyICU Copilot', 'EasyICU 研究助手'), guidedSub)}">
        <span class="cp-ico">${icon('spark', 16)}</span>
        <span class="cp-body"><span class="cp-t">${t('EasyICU Copilot', 'EasyICU 研究助手')}</span><span class="cp-d">${guidedSub}</span></span>
        <span class="cp-go">${icon('arrow', 14)}</span>
      </button>
      <button type="button" class="cp-entry ideas-entry ${route === 'ideas' ? 'on' : ''}" data-nav="ideas" aria-label="${navLabel(t('Idea Mining', '想法挖掘'), t('paper, PDF, or topic → feasible plan', '文章、PDF 或主题 → 可行计划'))}">
        <span class="cp-ico">${icon('target', 16)}</span>
        <span class="cp-body"><span class="cp-t">${t('Idea Mining', '想法挖掘')}</span><span class="cp-d">${t('paper, PDF, or topic → feasible plan', '文章、PDF 或主题 → 可行计划')}</span></span>
        <span class="cp-go">${icon('arrow', 14)}</span>
      </button>
      <!-- The two entries above and the Data Workspace below are the SAME
           pipeline with different skins; first-time users read them as three
           separate products and do not know a half-finished conversation can
           be continued on the classic pages (it can — study-context.js carries
           the handoff both ways). Say so where the choice is made. -->
      <div class="shared-note"><span class="ico">${icon('target', 11)}</span><span>${t('Paper or topic? Start with Idea Mining. Clear question? Start Guided Copilot. Already have data? Start with Extract Data. All three feed one pipeline — you can switch between conversation and the classic pages at any point without losing the study.', '有文章或主题，从想法挖掘开始；有明确问题，从研究引导开始；已有数据，从数据抽取开始。三者进入同一条流水线 —— 对话与经典页面之间随时可以互相切换，研究不会丢。')}</span></div>
      ${navSection('data', 'Data & Review', '数据与审阅', progress)}
      <div class="wsnav">
        <button type="button" class="wsgroup-head ${wsOpen ? 'open' : ''} ${classicActive ? 'active' : ''}" data-ws-toggle aria-expanded="${wsOpen}" aria-controls="data-workspace-links">
          <span class="wsg-ico">${icon('grid', 15)}</span>
          <span class="wsg-t">${t('Data Workspace', '数据工作台')}</span>
          <span class="wsg-chev">${icon(wsOpen ? 'chevdown' : 'chevron', 13)}</span>
        </button>
        ${wsOpen ? `
        <div class="wsg-children" id="data-workspace-links">
          ${CLASSIC.map((c, i) => `
            ${c.role === 'reads' && CLASSIC[i - 1] && CLASSIC[i - 1].role === 'produces'
              ? `<div class="wsg-step">${t('then, from that export', '然后，基于该导出')}</div>` : ''}
            <button type="button" class="wsitem ${route === c.id ? 'active' : ''} ws-${c.role}" data-nav="${c.id}" aria-label="${navLabel(c.label, c.sub)}">
              <span class="ico">${icon(c.ico, 15)}</span>
              <span class="wsi-copy"><span class="wsi-t">${L(c.label)}</span><span class="wsi-sub">${L(c.sub)}</span></span>
            </button>`).join('')}
        </div>` : ''}
      </div>
      ${navSection('analysis', 'Analysis & Evidence', '分析与证据', progress)}
      <button type="button" class="cp-entry agent-entry ${route === 'agent' ? 'on' : ''}" data-nav="agent" aria-label="${navLabel(t('Project Monitor', '项目监控'), t('runs · outputs · evidence · review', '运行 · 产出 · 证据 · 审阅'))}">
        <span class="cp-ico">${icon('agent', 16)}</span>
        <span class="cp-body"><span class="cp-t">${t('Project Monitor', '项目监控')}</span><span class="cp-d">${t('runs · outputs · evidence · review', '运行 · 产出 · 证据 · 审阅')}</span></span>
        <span class="cp-go">${icon('arrow', 14)}</span>
      </button>
      ${progress.planOnly ? `<div class="shared-note plan-only"><span class="ico">${icon('shield', 11)}</span><span>${t('Cross-DB comparison is plan-only: it can shape an analysis plan, but a reviewed cohort is still required before a draft.', '跨库对比仅用于制定计划：它可以塑造分析方案，但出草稿前仍需要一个已审阅的队列。')}</span></div>` : ''}
      <div class="sec-label" style="margin:16px 0 6px;">${t('Reference', '参考')}</div>
      <div class="nav" style="padding-top:0;">
        <button type="button" class="nav-item ${route === 'tutorial' ? 'active' : ''}" data-nav="tutorial"><span class="ico">${icon('help', 17)}</span>${t('Get Started', '快速上手')}</button>
        <button type="button" class="nav-item ${route === 'dictionary' ? 'active' : ''}" data-nav="dictionary"><span class="ico">${icon('list', 17)}</span>${t('Data Dictionary', '数据字典')}</button>
        <button type="button" class="nav-item ${route === 'settings' ? 'active' : ''}" data-nav="settings"><span class="ico">${icon('gear', 17)}</span>${t('Settings', '设置')}</button>
      </div>
      </nav>
      ${rail}
      <div class="rail-spacer"></div>
      <div class="rail-foot">
        <button type="button" class="icobtn" title="${t('Back to home', '返回首页')}" aria-label="${t('Back to home', '返回首页')}" data-nav="entry">${icon('back', 16)}</button>
        <button type="button" class="icobtn ${route === 'tutorial' ? 'on' : ''}" title="${t('Help', '帮助')}" aria-label="${t('Help', '帮助')}" data-nav="tutorial">${icon('help', 16)}</button>
        <button type="button" class="icobtn ${route === 'settings' ? 'on' : ''}" title="${t('Settings', '设置')}" aria-label="${t('Settings', '设置')}" data-nav="settings">${icon('gear', 16)}</button>
        <button type="button" class="icobtn" title="${t('Switch language', '切换语言')}" aria-label="${t('Switch language', '切换语言')}" data-lang-toggle>${icon('globe', 16)}</button>
        <div class="avatar">LK</div>
      </div>
    </aside>`;
  }

  function topbar() {
    const scr = screenOf(route);
    const dataMode = displayedDataMode();
    const demoMode = dataMode === 'demo';
    const officialDemo = demoMode
      && window.EU_DATA_MODE_CONTEXT
      && window.EU_DATA_MODE_CONTEXT.kind === 'official_demo';
    const consequential = !!window.EU_HASWORK;
    const crumbs = (scr.crumbs || []).map((c, i, arr) => {
      const label = crumbLabel(c);
      if (i === arr.length - 1) return `<span class="cur" aria-current="page">${label}</span>`;
      let node;
      if (i === 0) node = `<button type="button" class="crumb-link" data-nav="entry">${label}</button>`;
      else node = `<span class="mid">${label}</span>`;
      return `${node}<span class="sep">/</span>`;
    }).join(' ');
    const actionHtml = actionHtmlOf(scr);
    return `
    <header class="topbar" aria-label="${t('Page controls', '页面控制')}">
      <div class="crumbs">${crumbs}</div>
      <div class="spacer"></div>
      ${scr.status || ''}
      <!-- This control sits between "EasyICU Copilot" and the EN/中 toggle, but it
           is not a display preference: flipping it swaps the data source,
           marks every downstream cohort/extraction/review stale and cancels a
           running Cross-DB scan. Once there IS downstream work, give it the
           weight of a destructive action so it stops reading like EN/中. -->
      <div class="mode-seg ${demoMode ? 'demo-active' : ''} ${consequential ? 'consequential' : ''}" role="group" aria-label="Data mode" title="${consequential ? t('Switching the data source will mark your current cohort, extraction and review as out of date.', '切换数据源会把当前的队列、抽取与审阅标记为过期。') : (demoMode ? t('Demo mode uses official public deidentified demo datasets or clearly labelled seeded examples. It is never your local data; switch to Real to load a local export.', '演示模式使用官方公开去标识 Demo 数据集，或明确标注的种子示例；都不是你的本地数据。切换到真实模式可加载本地导出。') : t('Real mode: screens compute from your local EasyICU export. Nothing is uploaded.', '真实模式：各页面从你本地的 EasyICU 导出计算，不上传任何数据。'))}">
        <button type="button" class="${demoMode ? 'on' : ''}" data-datamode="demo" aria-pressed="${demoMode}">${icon('flask', 12)} ${officialDemo ? t('Official demo', '官方演示') : (demoMode ? t('Demo data', '演示数据') : t('Demo', '演示'))}</button>
        <button type="button" class="${!demoMode ? 'on' : ''}" data-datamode="real" aria-pressed="${!demoMode}">${icon('db', 12)} ${t('Real', '真实')}</button>
      </div>
      <div class="lang-seg" role="group" aria-label="Language">
        <button type="button" class="${window.EU_LANG !== 'zh' ? 'on' : ''}" data-lang="en" aria-pressed="${window.EU_LANG !== 'zh'}">EN</button>
        <button type="button" class="${window.EU_LANG === 'zh' ? 'on' : ''}" data-lang="zh" aria-pressed="${window.EU_LANG === 'zh'}">中</button>
      </div>
      <button type="button" class="btn sm" data-cpopen title="${t('Open the one EasyICU Copilot conversation', '打开唯一的 EasyICU 研究助手对话')}">${icon('spark', 13)} ${t('EasyICU Copilot','研究助手')}</button>
      ${actionHtml}
    </header>`;
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
        <button type="button" class="btn sm icon" data-cpopen title="${t('EasyICU Copilot', 'EasyICU 研究助手')}" aria-label="${t('Open EasyICU Copilot', '打开 EasyICU 研究助手')}">${icon('spark', 16)}</button>
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
    const priorMain = app.querySelector('main');
    const priorActive = document.activeElement;
    const preserveRouteFocus = !!(
      priorMain && priorActive && priorMain.contains(priorActive)
      && (priorActive === priorMain || priorActive.tagName === 'H1')
    );
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
      app.innerHTML = `
        <div class="app">
          ${sidebar()}
          <main class="main" aria-label="${t('Page content', '页面内容')}">
            ${m.top}
            ${topbar()}
            <div class="content ${scr.wide ? 'wide' : ''}">${scr.render()}</div>
          </main>
          ${m.bottom}
        </div>`;
    }
    if (scr.afterRender) scr.afterRender(app);
    syncShellAccessibility(app, !!scr.full);
    const title = routeTitleOf(scr);
    document.title = routeDocumentTitle(title);
    const c = app.querySelector('.content');
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
    // ⌘K / Ctrl+K → open the single EasyICU Copilot conversation.
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
