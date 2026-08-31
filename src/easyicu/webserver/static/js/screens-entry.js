/* Entry/home route owner. Data Extraction is loaded next but no longer owns
   this route's study launcher, resume banner, or first-run demo nudge. */
(function () {
  'use strict';
  const S = (window.SCREENS = window.SCREENS || {});
  const escHtml = window.EU_HTML.esc;

  function homeDataMode() { return window.EU_DATA || 'demo'; }
  function setHomeData(mode) {
    if (window.setDataMode) { window.setDataMode(mode); return; }
    window.EU_DATA = mode;
    try { localStorage.setItem('easyicu_home_data', mode); } catch (_) {}
  }

  const HOME_BRANCH_GOALS = Object.freeze({
    predict: 'Model an outcome', crossdb: 'Compare databases', quality: 'Audit data quality',
  });
  const RESUME_ROUTE_ALLOWLIST = new Set(['guided', 'ideas', 'extraction', 'patient', 'cohort', 'crossdb', 'agent']);
  const LEGACY_RESUME_ROUTES = Object.freeze({ predict: 'patient', crossdb: 'crossdb', quality: 'cohort' });
  let homeQuestionDraft = '';

  function startHomeStudy(route, patch) {
    const store = window.EU_STUDY_CONTEXT;
    if (!store || typeof store.startNew !== 'function') return;
    store.startNew(Object.assign({ last_route: route, current_stage: 'study_setup' }, patch || {}), { reason: 'home-new-study' });
  }

  function launchCopilot(text, branchHint) {
    const question = String(text || '').trim();
    const analysisGoal = question || HOME_BRANCH_GOALS[branchHint] || '';
    startHomeStudy('guided', { question, analysis_goal: analysisGoal });
    try {
      window.__cpBridge = { ts: Date.now(), route: 'entry', lastUser: question || null, dataMode: homeDataMode(), branchHint: branchHint || null };
    } catch (_) {}
    location.hash = '#guided';
  }

  function homeHeader() {
    return `
      <header class="entry-top">
        <div class="row gap-12">
          <div class="mark">${icon('flask', 18)}</div>
          <div>
            <div class="name" style="font-size:16px;font-weight:600;letter-spacing:-0.01em;">EasyICU</div>
            <div class="tag" style="font-size:11px;color:var(--ink-4);">${t('ICU data research workspace', 'ICU 数据研究工作台')}</div>
          </div>
        </div>
        <div class="row gap-10">
          <div class="lang-seg" role="group" aria-label="Language">
            <button type="button" class="${window.EU_LANG !== 'zh' ? 'on' : ''}" data-lang="en" aria-pressed="${window.EU_LANG !== 'zh'}">EN</button>
            <button type="button" class="${window.EU_LANG === 'zh' ? 'on' : ''}" data-lang="zh" aria-pressed="${window.EU_LANG === 'zh'}">中</button>
          </div>
          <span class="mono" style="font-size:11px;color:var(--ink-4);">v1.0 · py3.10+</span>
        </div>
      </header>`;
  }

  S.entry = {
    section: 'entry',
    full: true,
    render() {
      const dm = homeDataMode();
      const inner = `
        <div class="home-inner" style="max-width:1180px;">
          <h1 class="home-h1">${t('Welcome to EasyICU', '欢迎使用 EasyICU')}</h1>
          <p class="home-sub">${t('Start from what you already have: a paper or topic, a clear research question, or local ICU data. EasyICU keeps the study local and carries its context between modules.', '从你已有的内容开始：文章或主题、明确的研究问题，或本地 ICU 数据。EasyICU 全程在本地运行，并在模块之间延续同一研究上下文。')}</p>
          <div class="home-split">
            <div class="home-col col-copilot">
              <div class="col-head"><div class="col-mk">${icon('spark', 17)}</div><div><div class="col-t">${t('I have a clear research question', '我有明确的研究问题')}</div><div class="col-sub">${t('Guided Copilot · describe it in one sentence', '研究引导 · 用一句话描述')}</div></div><span class="col-badge">${t('Recommended', '推荐')}</span></div>
              <div class="col-body">
                <p class="col-lead">${t('Guided Copilot collects the cohort, outcome, time window, modules, and export settings in the conversation before anything runs.', '研究引导会先在对话中收集队列、结局、时间窗、模块和导出设置，再开始运行。')}</p>
                <div class="col-prompt">
                  <textarea class="hp-input" id="homeInput" rows="3" placeholder="${t('e.g. Among Sepsis-3 patients, does early lactate predict in-hospital mortality, and does adding it to SOFA improve the model?', '例如:在脓毒症(Sepsis-3)患者中,早期乳酸能否预测院内死亡?把它加入 SOFA 是否提升模型?')}" autocomplete="off" aria-label="${t('Describe your study', '描述你的研究')}">${escHtml(homeQuestionDraft)}</textarea>
                  <div class="hp-bar">
                    <span class="hp-hint">${icon('shield', 12)} ${t('local-only · nothing uploaded', '仅本地 · 不上传')}</span>
                    <button type="button" class="hp-send" id="homeSend" aria-label="${t('Start Guided Copilot', '开始研究引导')}">${icon('arrow', 17)}</button>
                  </div>
                </div>
                <div class="col-chips">
                  <span class="cc-lead">${t('or start from a question type', '或选一种研究类型')}</span>
                  <button type="button" class="home-chip" data-hbranch="predict">${t('Model an outcome', '结局建模')}</button>
                  <button type="button" class="home-chip" data-hbranch="crossdb">${t('Compare databases', '跨库比较')}</button>
                  <button type="button" class="home-chip" data-hbranch="quality">${t('Audit data quality', '数据质量审计')}</button>
                </div>
              </div>
            </div>
            <div class="home-col col-classic">
              <div class="col-head"><div class="col-mk">${icon('grid', 17)}</div><div><div class="col-t">${t('Classic Workspace', '经典工作台')}</div><div class="col-sub">${t('drive it yourself · for when you know the steps', '自己操作 · 熟悉流程后使用')}</div></div></div>
              <div class="col-body">
                <p class="col-lead">${t('Choose the entry that matches what you already have.', '按照你已经拥有的材料选择入口。')}</p>
                <div class="col-entries">
                  ${[
                    ['ideas', 'target', t('I have a paper or topic', '我有文章或研究主题'), t('Mine a feasible question in Idea Mining', '在想法挖掘中形成可行问题'), 'ideas'],
                    ['extraction', 'extract', t('I have local ICU data', '我有本地 ICU 数据'), t('Validate and extract analysis-ready tables', '校验并抽取可分析数据表'), 'extraction'],
                    ['patient', 'viz', t('Patient Review', '患者审阅'), t('Review patients, tables, and trends from an export', '审阅导出中的患者、表格与趋势'), ''],
                    ['agent', 'agent', t('Project Monitor', '项目监控'), t('Review runs, outputs, evidence, and sign-off', '查看运行、产出、证据与签署'), ''],
                  ].map(([nav, ic, title, detail, newStudy]) => `
                    <button type="button" class="col-entry" data-nav="${nav}" ${newStudy ? `data-home-new-study="${newStudy}"` : ''}>
                      <span class="ce-ico">${icon(ic, 15)}</span>
                      <span><span class="ce-t">${title}</span><span class="ce-d">${detail}</span></span>
                      <span class="ce-go">${icon('arrow', 14)}</span>
                    </button>`).join('')}
                </div>
                <div class="col-foot">
                  <div class="home-datamode">
                    <span class="hdm-lab">${t('Data', '数据')}</span>
                    <div class="seg home-data-seg" id="homeData" role="group" aria-label="${t('Data mode', '数据模式')}">
                      <button type="button" class="${dm === 'demo' ? 'on' : ''}" data-hd="demo" aria-pressed="${dm === 'demo'}">${t('Demo', '演示')}</button>
                      <button type="button" class="${dm === 'real' ? 'on' : ''}" data-hd="real" aria-pressed="${dm === 'real'}">${t('Real', '真实')}</button>
                    </div>
                    <span class="col-dataline">${icon('shield', 12)} ${dm === 'demo' ? t('Demo data · reproducible', '演示数据 · 可复现') : t('Real data · local-only', '真实数据 · 仅本地')}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div id="resumeSlot" class="home-resume"></div>
          <div class="entry-journey">
            <div class="ej-cap">${t('The research journey · 4 steps', '研究旅程 · 四步')}</div>
            <div class="ej-track">
              ${[
                ['1', t('Frame', '框定'), t('the question', '研究问题'), 'guided'],
                ['2', t('Extract', '抽取'), t('the data', '数据'), 'extraction'],
                ['3', t('Review', '审阅'), t('& explore', '与探索'), 'patient'],
                ['4', t('Analyze', '分析'), t('& draft', '与撰稿'), 'guided'],
              ].map((item, index) => `
                ${index > 0 ? '<div class="ej-conn"></div>' : ''}
                <button type="button" class="ej-node" data-nav="${item[3]}" title="${t('Go to this step', '前往这一步')}"><div class="ej-num">${item[0]}</div><div><div class="ej-lab">${item[1]}</div><div class="ej-sub">${item[2]}</div></div></button>`).join('')}
            </div>
            <div class="ej-foot">${t('Plan → extract → review → analyze, with the same study context carried forward.', '计划 → 抽取 → 审阅 → 分析，全程延续同一研究上下文。')}</div>
          </div>
          <div class="entry-firsttime" id="firstTimeNudge" hidden>
            <span class="ft-ico">${icon('play', 13)}</span>
            <span>${t('Want to explore first?', '想先体验一下？')} <b>${t('Try the 2-minute demo', '试用 2 分钟演示')}</b> ${t('— no setup or data required.', '—— 无需配置或自备数据。')}</span>
            <button type="button" class="ft-go" data-firsttime>${t('Start the tour', '开始引导')} ${icon('arrow', 13)}</button>
            <button type="button" class="ft-x" data-firsttime-dismiss aria-label="${t('Dismiss', '忽略')}">${icon('close', 13)}</button>
          </div>
        </div>`;
      return `<div class="entry-shell">${homeHeader()}<div class="home-wrap">${inner}</div></div>`;
    },
    afterRender(root) {
      const dataEl = root.querySelector('#homeData');
      if (dataEl) dataEl.addEventListener('click', event => {
        const button = event.target.closest('[data-hd]');
        if (button) setHomeData(button.dataset.hd);
      });
      const input = root.querySelector('#homeInput');
      const send = root.querySelector('#homeSend');
      const firstTime = root.querySelector('[data-firsttime]');
      if (firstTime) firstTime.addEventListener('click', () => {
        try { localStorage.setItem('easyicu_onboarded', '1'); } catch (_) {}
        if (window.setDataMode) window.setDataMode('demo', { force: true });
        location.hash = '#tutorial';
      });
      const nudge = root.querySelector('#firstTimeNudge');
      if (nudge) {
        let onboarded = false;
        let hasStudy = false;
        try { onboarded = !!localStorage.getItem('easyicu_onboarded'); } catch (_) {}
        try { hasStudy = !!localStorage.getItem('easyicu_study'); } catch (_) {}
        try { hasStudy = hasStudy || !!(window.EU_STUDY_CONTEXT && window.EU_STUDY_CONTEXT.active && window.EU_STUDY_CONTEXT.active()); } catch (_) {}
        if (!onboarded && !hasStudy && !window.EU_HASWORK) nudge.hidden = false;
        const dismiss = nudge.querySelector('[data-firsttime-dismiss]');
        if (dismiss) dismiss.addEventListener('click', () => {
          try { localStorage.setItem('easyicu_onboarded', '1'); } catch (_) {}
          nudge.hidden = true;
        });
      }
      root.querySelectorAll('[data-home-new-study]').forEach(button => button.addEventListener('click', () => {
        const target = button.dataset.homeNewStudy;
        startHomeStudy(target, target === 'ideas' ? { purpose: 'idea_mining' } : {});
        if (target === 'extraction' && window.setDataMode) window.setDataMode('real', { force: true });
      }));
      function submit() { launchCopilot(((input && input.value) || '').trim() || null); }
      if (send) send.addEventListener('click', submit);
      if (input) {
        input.addEventListener('input', () => { homeQuestionDraft = input.value; });
        input.addEventListener('keydown', event => {
          if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit(); }
        });
      }
      root.querySelectorAll('[data-hbranch]').forEach(button => button.addEventListener('click', () => launchCopilot(null, button.dataset.hbranch)));

      // Resume banner: current StudyContext wins; legacy branch state is fallback-only.
      const slot = root.querySelector('#resumeSlot');
      if (slot) {
        let activeContext = null;
        let legacyStudy = null;
        try {
          const store = window.EU_STUDY_CONTEXT;
          activeContext = store && typeof store.active === 'function' ? store.active() : null;
        } catch (_) {}
        try { legacyStudy = JSON.parse(localStorage.getItem('easyicu_study') || 'null'); } catch (_) {}
        const contextRoute = activeContext && RESUME_ROUTE_ALLOWLIST.has(activeContext.last_route) ? activeContext.last_route : null;
        const legacyRoute = legacyStudy ? LEGACY_RESUME_ROUTES[legacyStudy.branch] : null;
        const resumeRoute = contextRoute || legacyRoute || null;
        const usingContext = !!contextRoute;
        let contextDismissed = false;
        try {
          contextDismissed = usingContext && !!activeContext.id
            && sessionStorage.getItem('easyicu.studyContext.resumeDismissed.v1') === activeContext.id;
        } catch (_) {}
        if (resumeRoute && !contextDismissed) {
          const branchNames = { predict: t('Sepsis mortality prediction', '脓毒症死亡率预测'), crossdb: t('Cross-database comparison', '跨数据库对比'), quality: t('Data-quality audit', '数据质量审计') };
          const routeNames = {
            guided: t('Guided Copilot', '研究引导'), ideas: t('Idea Mining', '想法挖掘'),
            extraction: t('Data Extraction', '数据抽取'), patient: t('Patient Review', '患者审阅'),
            cohort: t('Cohort Statistics', '队列统计'), crossdb: t('Cross-database comparison', '跨库对比'),
            agent: t('Project Monitor', '项目监控'),
          };
          const rawTime = usingContext ? Date.parse(activeContext.updated_at || '') : Number(legacyStudy.ts || 0);
          const when = (() => {
            const minutes = Math.max(0, Math.round((Date.now() - (Number.isFinite(rawTime) && rawTime > 0 ? rawTime : Date.now())) / 60000));
            return minutes < 1 ? t('just now', '刚刚') : minutes < 60 ? minutes + t('m ago', ' 分钟前') : Math.round(minutes / 60) + t('h ago', ' 小时前');
          })();
          const contextTitle = usingContext ? String(activeContext.question || activeContext.title || '').trim() : '';
          const contextSummary = contextTitle && contextTitle !== 'Untitled ICU study'
            ? `${escHtml(contextTitle)} · ${routeNames[resumeRoute]} · ${when}`
            : `${routeNames[resumeRoute]} · ${when}`;
          // patientN is a stay count written by screens-guided.js as a clamped
          // integer, but it round-trips through localStorage, so coerce it back
          // to a number: legacySummary is interpolated into innerHTML below and
          // a string value would otherwise reach the DOM unescaped.
          const legacyStays = Number.parseInt((legacyStudy && legacyStudy.patientN), 10);
          const legacySummary = `${branchNames[legacyStudy && legacyStudy.branch] || t('Study', '研究')} · ${Number.isFinite(legacyStays) ? legacyStays : 10} ${t('stays', '次住院')} · ${((legacyStudy && legacyStudy.mods) || []).length} ${t('modules', '模块')} · ${when}`;
          slot.innerHTML = `
            <div class="card flat" style="border-color:var(--accent-border);background:color-mix(in srgb, var(--accent-soft) 40%, var(--surface));">
              <div class="row gap-12">
                <div class="aux-ico" style="background:var(--accent-soft);color:var(--accent-ink);">${icon('history', 16)}</div>
                <div><div style="font-weight:600;font-size:13px;">${t('Resume your last study', '继续上次的研究')}</div><div style="font-size:12px;color:var(--ink-3);margin-top:1px;">${usingContext ? contextSummary : legacySummary}</div></div>
              </div>
              <div class="row gap-8">
                <button type="button" class="btn sm" data-resume-open>${t('Open', '打开')} ${routeNames[resumeRoute]} ${icon('arrow', 14)}</button>
                <button type="button" class="btn sm ghost" data-resume-clear>${t('Dismiss', '忽略')}</button>
              </div>
            </div>`;
          slot.querySelector('[data-resume-open]').addEventListener('click', () => {
            if (!usingContext && legacyStudy) {
              try { if (window.__euExtractApply) window.__euExtractApply(legacyStudy.mods); } catch (_) {}
              try { if (window.__euVizPreset) window.__euVizPreset(); } catch (_) {}
            }
            location.hash = '#' + resumeRoute;
          });
          slot.querySelector('[data-resume-clear]').addEventListener('click', () => {
            try {
              if (usingContext && activeContext.id) sessionStorage.setItem('easyicu.studyContext.resumeDismissed.v1', activeContext.id);
              else localStorage.removeItem('easyicu_study');
            } catch (_) {}
            slot.innerHTML = '';
          });
        }
      }
      setTimeout(() => { if (input) input.focus(); }, 300);
    },
  };
})();
