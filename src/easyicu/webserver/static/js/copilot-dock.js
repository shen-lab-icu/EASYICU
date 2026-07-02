/* EasyICU — Page guide dock controller.
   A persistent command panel for the active workspace page. It explains the
   current screen and offers bounded local shortcuts. It is deliberately not
   the full conversational Copilot; longer study planning belongs in Guided
   Copilot. Body-level so it survives route re-renders. */
(function () {
  let dock, fab, backdrop, scroll, suggestEl, input;
  let thread = [];
  let busy = false;
  let lastRoute = '';
  let sessionId = null;
  let sessionReady = null;

  function bi(en, zh) { return { en, zh }; }
  function htmlOf(value) {
    if (value && typeof value === 'object' && Object.prototype.hasOwnProperty.call(value, 'en')) {
      return window.t ? window.t(value.en, value.zh) : value.en;
    }
    return value == null ? '' : String(value);
  }
  function currentContext() {
    const src = window.EU_SOURCES && window.EU_SOURCES.activeSource ? window.EU_SOURCES.activeSource() : null;
    const selected = src ? {
      label: src.label || src.name || src.database || '',
      database: src.database || src.source || '',
      path: src.path || '',
    } : null;
    return {
      route: routeOf(),
      language: window.EU_LANG || 'en',
      data_mode: window.EU_DATA || 'demo',
      selected_source: selected,
    };
  }
  function localized(value) {
    if (value && typeof value === 'object') {
      return window.t ? window.t(value.en || '', value.zh || value.en || '') : (value.en || value.zh || '');
    }
    return value == null ? '' : String(value);
  }
  function tx(en, zh) { return window.t ? window.t(en, zh) : en; }
  function labelEn(ctx) {
    const label = ctx && ctx.label;
    return label && typeof label === 'object' ? (label.en || label.zh || '') : (label || '');
  }
  function labelZh(ctx) {
    const label = ctx && ctx.label;
    return label && typeof label === 'object' ? (label.zh || label.en || '') : (label || '');
  }
  function labelOf(ctx) { return tx(labelEn(ctx), labelZh(ctx)); }
  function contextText(route) { return tx('on ', '当前页面：') + labelOf(ctxFor(route)); }
  function chipText(chip) {
    if (Array.isArray(chip)) return htmlOf(chip[0]);
    return localized({ en: chip.label_en || chip.label || '', zh: chip.label_zh || chip.label || chip.label_en || '' });
  }
  function hasPageGuideBackend() {
    return !!(window.EU_API && window.EU_API.createPageGuideSession && window.EU_API.sendPageGuideMessage && window.EU_API.runPageGuideAction);
  }
  async function ensureSession() {
    if (!hasPageGuideBackend()) return null;
    if (sessionId) return { session: { id: sessionId } };
    if (!sessionReady) {
      sessionReady = window.EU_API.createPageGuideSession({
        scope: 'page_guide',
        context: currentContext(),
      }).then(payload => {
        sessionId = payload && payload.session && payload.session.id;
        return payload;
      }).catch(err => {
        sessionReady = null;
        throw err;
      });
    }
    return sessionReady;
  }
  function applyBackendResult(result) {
    if (!result || !result.type) return false;
    if (result.type === 'navigate' && result.target) {
      nav('#' + result.target);
      return true;
    }
    if (result.type === 'reply') {
      botSay(localized(result.reply), result.chips || ctxFor(routeOf()).chips);
      return true;
    }
    return false;
  }
  async function sendBackendMessage(value) {
    try {
      const session = await ensureSession();
      const payload = await window.EU_API.sendPageGuideMessage({
        session_id: session && session.session && session.session.id,
        scope: 'page_guide',
        message: value,
        context: currentContext(),
      });
      botSay(localized(payload.reply), payload.chips);
      const action = payload.actions && payload.actions[0];
      if (action && action.type === 'navigate' && action.target) {
        setTimeout(() => nav('#' + action.target), 720);
      }
      return true;
    } catch (err) {
      console.warn('[EasyICU] Page guide backend unavailable, using local fallback', err);
      return false;
    }
  }
  async function runBackendAction(action, label) {
    if (!action || !hasPageGuideBackend()) return false;
    if (action.indexOf('explain_') === 0) {
      return sendBackendMessage(label || action.replace(/^explain_/, ''));
    }
    try {
      const payload = await window.EU_API.runPageGuideAction({
        session_id: sessionId,
        action: action,
        context: currentContext(),
      });
      if (payload.blocked) {
        botSay(localized(payload.reason || payload.error || 'Blocked.'), ctxFor(routeOf()).chips);
        return true;
      }
      if (applyBackendResult(payload.result)) return true;
      botSay(bi('Done.', '完成。'), ctxFor(routeOf()).chips);
      return true;
    } catch (err) {
      console.warn('[EasyICU] Page guide action failed', err);
      botSay(bi('That action is not available yet.', '这个动作现在还不可用。'), ctxFor(routeOf()).chips);
      return true;
    }
  }
  function backendExplainActionForRoute(route) {
    return {
      extraction: 'explain_extraction',
      patient: 'explain_patient',
      crossdb: 'explain_crossdb',
      agent: 'explain_gate',
      settings: 'explain_settings',
    }[route] || null;
  }
  function appendRouteGreeting(route) {
    const c = ctxFor(route);
    thread.push({ ctxline: true, html: bi(`now on ${labelEn(c)}`, `现在在：${labelZh(c)}`) });
    render();
    const action = backendExplainActionForRoute(route);
    if (action && hasPageGuideBackend()) {
      window.EU_API.runPageGuideAction({
        session_id: sessionId,
        action: action,
        context: currentContext(),
      }).then(payload => {
        const result = payload && payload.result;
        if (result && result.type === 'reply') {
          thread.push({ bot: true, html: localized(result.reply) });
          render(); renderChips(result.chips || c.chips);
          return;
        }
        thread.push({ bot: true, html: c.hi });
        render(); renderChips(c.chips);
      }).catch(err => {
        console.warn('[EasyICU] Page guide route context backend failed', err);
        thread.push({ bot: true, html: c.hi });
        render(); renderChips(c.chips);
      });
      return;
    }
    thread.push({ bot: true, html: c.hi });
    render(); renderChips(c.chips);
  }

  /* ---- per-route context: greeting + chips + answers ---- */
  const CTX = {
    entry: {
      label: bi('Home', '首页'),
      hi: bi(`This Page guide explains the current screen and offers safe shortcuts. Open Guided Copilot when you want a full conversational study plan.`, `页面指南会解释当前页面并提供安全快捷操作。需要完整对话式研究规划时，请打开 Guided Copilot。`),
      chips: [[bi('Open Guided Copilot', '打开 Guided Copilot'), '@guided', 'act'], [bi('How does EasyICU work?', 'EasyICU 怎么工作？'), '@say:how'], [bi('Load a demo workspace', '加载演示工作区'), '@load', 'act']],
    },
    extraction: {
      label: bi('Data Extraction', '数据抽取'),
      hi: bi(`You’re in Data Extraction. Use the page controls to configure cohort, modules, and export. This guide can explain the page or open Guided Copilot.`, `这里是数据抽取。请用页面控件配置队列、模块和导出。页面指南可以解释页面，或打开 Guided Copilot。`),
      chips: [[bi('How does extraction work?', '抽取怎么工作？'), '@say:extract'], [bi('What gets exported?', '会导出什么？'), '@say:export'], [bi('Open Guided Copilot', '打开 Guided Copilot'), '@guided', 'act']],
    },
    patient: {
      label: bi('Patient Review', '患者审阅'),
      hi: bi(`This is Patient Review — tables, time series, a patient overview, and data-quality flags. I can load a demo workspace so it’s populated.`, `这里是患者审阅：包含表格、时间序列、患者概览和数据质量标记。我可以先加载一个演示工作区。`),
      chips: [[bi('Load demo workspace', '加载演示工作区'), '@load', 'act'], [bi('What’s in each tab?', '每个标签页是什么？'), '@say:tabs'], [bi('Explain data-quality flags', '解释数据质量标记'), '@say:quality']],
    },
    cohort: {
      label: bi('Cohort Statistics', '队列统计'),
      hi: bi(`Cohort Statistics compares groups, audits coverage, and reclassifies SOFA. I can re-run it or explain what you’re seeing.`, `队列统计用于分组比较、覆盖率审计和 SOFA 重分类。我可以重新运行，或解释当前结果。`),
      chips: [[bi('Re-run statistics', '重新运行统计'), '@cohortrun', 'act'], [bi('Explain SOFA reclassification', '解释 SOFA 重分类'), '@say:sofa'], [bi('What’s the comparison?', '当前比较是什么？'), '@say:contrast']],
    },
    crossdb: {
      label: bi('Cross-DB Benchmark', '跨库对比'),
      hi: bi(`Cross-DB Benchmark applies one cohort definition across ICU databases. I can load it or explain where databases diverge.`, `跨库比较会把同一个队列定义应用到多个 ICU 数据库。我可以加载比较，或解释哪些数据库差异最大。`),
      chips: [[bi('Load the benchmark', '加载跨库比较'), '@loadcross', 'act'], [bi('Which databases overlap?', '哪些数据库有重叠？'), '@say:overlap'], [bi('Open Guided Copilot', '打开 Guided Copilot'), '@guided', 'act']],
    },
    agent: {
      label: bi('Agent Projects', '研究项目'),
      hi: bi(`The Research Agent runs an auditable pipeline and drafts findings — but the draft stays locked until checks pass. Want the run, or the reasoning?`, `Research Agent 会运行可审计 pipeline 并生成 findings 草稿，但检查通过前草稿保持锁定。你想看运行，还是想看核验逻辑？`),
      chips: [[bi('Why is the draft locked?', '为什么草稿锁定？'), '@say:gate'], [bi('Show a completed run', '查看已完成 run'), '@agentrun', 'act'], [bi('Open Guided Copilot', '打开 Guided Copilot'), '@guided', 'act']],
    },
    states: { label: bi('Workspace States', '工作区状态'), hi: bi(`This is the states reference — loading, empty, no-data, error, blocked, success. Use the shortcuts below or open Guided Copilot.`, `这里是工作区状态参考：加载、空、无数据、错误、阻断、成功。可用下方快捷操作，或打开 Guided Copilot。`), chips: [[bi('When do states show?', '状态什么时候出现？'), '@say:states'], [bi('Open Guided Copilot', '打开 Guided Copilot'), '@guided', 'act']] },
    settings: { label: bi('Settings', '设置'), hi: bi(`Settings are local-first and reversible. Ask me what any option does.`, `设置是本地优先且可回退的。你可以问任意选项的含义。`), chips: [[bi('Is my data uploaded?', '我的数据会上传吗？'), '@say:privacy'], [bi('Explain evidence checks', '解释证据核验'), '@say:gate']] },
    tutorial: { label: bi('Get Started', '快速上手'), hi: bi(`Get Started orients you to the workflow. Open Guided Copilot for the full conversational path.`, `快速上手会介绍整个流程；完整对话式路径请打开 Guided Copilot。`), chips: [[bi('Open Guided Copilot', '打开 Guided Copilot'), '@guided', 'act'], [bi('How does EasyICU work?', 'EasyICU 怎么工作？'), '@say:how']] },
  };

  const ANSWERS = {
    how: bi(`EasyICU runs locally and moves a study through four stages: <strong>frame → extract → review → analyze & draft</strong>. This Page guide explains and links; Guided Copilot handles full conversational planning. Nothing is uploaded.`, `EasyICU 在本机运行，把研究推进为四步：<strong>定义问题 → 抽取 → 审阅 → 分析与草稿</strong>。页面指南负责解释和跳转；完整对话式规划由 Guided Copilot 承担。不会上传数据。`),
    extract: bi(`Extraction has a one-click <strong>recommended</strong> path — first ICU stay, full available ICU window with a 30-day cap, six core feature modules — that writes analysis-ready frames plus a reproducible manifest. Need more control? The <strong>Customize</strong> panel opens cohort criteria, feature modules, and export format. Either way, nothing runs on incomplete data.`, `抽取有一键<strong>推荐路径</strong>：首个 ICU stay、完整可用 ICU 窗口并设 30 天上限、核心特征模块，输出可分析数据表和可复现清单。需要更多控制时，<strong>自定义</strong>面板可以配置队列、模块和导出格式。`),
    export: bi(`An export bundle is the concept data plus a <strong>manifest.json</strong> — not the figures. Code, tables, and figures come from a Research Agent run, with an evidence ledger.`, `导出包包含概念数据和 <strong>manifest.json</strong> 清单，不是图件。代码、表格和图件来自 Research Agent run，并附证据台账。`),
    tabs: bi(`<ul class="mb-list"><li><strong>Data Tables</strong> — raw stay-level rows</li><li><strong>Time Series</strong> — hourly trajectories</li><li><strong>Patient Overview</strong> — one stay at a glance</li><li><strong>Data Quality</strong> — coverage, ranges, missingness</li></ul>`, `<ul class="mb-list"><li><strong>数据表</strong>：stay 级数据</li><li><strong>时间序列</strong>：小时级轨迹</li><li><strong>患者概览</strong>：单个 stay 一屏查看</li><li><strong>数据质量</strong>：覆盖率、范围和缺失</li></ul>`),
    quality: bi(`Flags mark concepts that are sparse or out-of-range. They protect your denominators — a flagged module is surfaced before it can bias a model. Want me to load the workspace so you can see them?`, `标记用于提示稀疏或越界的概念，保护分母不被悄悄污染。某个模块影响模型前，会先暴露出来。要我加载工作区给你看吗？`),
    sofa: bi(`SOFA reclassification shows how patients move between severity bands when you recompute the score — it’s a sensitivity check, not a new finding.`, `SOFA 重分类展示重新计算评分后患者如何在严重度分层之间移动。它是敏感性检查，不是新的临床发现。`),
    contrast: bi(`The default contrast is Survived vs Deceased; you can switch to age groups, sex, length-of-stay, or Sepsis vs non-sepsis. All values shown are seeded demo numbers.`, `默认比较是存活 vs 死亡；也可以切换到年龄、性别、住院时长或 Sepsis vs 非 Sepsis。演示模式下显示的是种子演示数字。`),
    overlap: bi(`Only concepts present in every selected database can be compared fairly — the availability matrix shows where one is missing. Lactate and MAP usually overlap; some scores are partial.`, `只有在所选数据库中都存在的概念才适合公平比较；可用性矩阵会显示缺失在哪里。乳酸和 MAP 通常有重叠，部分评分可能不完整。`),
    gate: bi(`Drafting is a deliberate second stage. A claim may only be written once it traces to a logged artifact — denominators resolved, coverage above threshold, tables reproducing from the manifest — and a human signs off. That’s why the draft is locked until then.`, `草稿是有意延后的阶段。只有当声明能追溯到已记录产物，并且分母、覆盖率、表格复现和人工签署都通过后，才允许写入稿件。所以在此之前草稿保持锁定。`),
    states: bi(`Every data surface passes through the same six: loading (skeletons), empty (first run), no-data (0 results), error (recoverable), blocked (gated), success. The reference page lets you preview each.`, `每个数据界面都经过六种状态：加载、空、无数据、错误、阻断和成功。参考页可以预览这些状态。`),
    privacy: bi(`No. EasyICU is local-first and the guarantee is enforced — extraction, review, and analysis run on your machine. Only the agent’s plan text can ever leave, and only if you explicitly enable it. Never patient rows.`, `不会。EasyICU 是本地优先：抽取、审阅和分析都在你的机器上运行。只有你明确启用时，Agent 的计划文本才可能离开本机，患者行永远不会发送。`),
  };

  function routeOf() { const r = (location.hash || '#entry').slice(1); return CTX[r] ? r : (window.SCREENS && window.SCREENS[r] ? r : 'entry'); }
  function ctxFor(r) { return CTX[r] || CTX.entry; }

  /* ---- rendering ---- */
  function render() {
    scroll.innerHTML = thread.map(t => {
      if (t.typing) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble"><div class="typing"><span></span><span></span><span></span></div></div></div></div>`;
      if (t.ctxline) return `<div class="cp-ctxline"><span class="dotline"></span>${htmlOf(t.html)}<span class="dotline"></span></div>`;
      if (t.bot) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble">${htmlOf(t.html)}</div></div></div>`;
      return `<div class="msg user"><div class="m-ava">LK</div><div class="m-body"><div class="m-bubble">${t.html}</div></div></div>`;
    }).join('');
    scroll.scrollTop = scroll.scrollHeight + 400;
  }
  function renderChips(chips) {
    suggestEl.innerHTML = (chips || []).map(chip => {
      if (Array.isArray(chip)) {
        const [label, tok, cls] = chip;
        return `<button class="suggest-chip ${cls || ''}" data-cp="${tok}">${htmlOf(label)}</button>`;
      }
      const action = esc(chip.action || '');
      const cls = chip.class || (action.indexOf('open_') === 0 ? 'act' : '');
      return `<button class="suggest-chip ${cls}" data-cp-action="${action}">${esc(chipText(chip))}</button>`;
    }).join('');
  }
  function esc(s) { return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

  function botSay(html, chips) {
    busy = true; thread.push({ typing: true }); render();
    setTimeout(() => {
      thread = thread.filter(t => !t.typing); busy = false;
      thread.push({ bot: true, html });
      render();
      if (chips !== undefined) renderChips(chips);
    }, 480);
  }

  /* ---- GUI driving ---- */
  function flashContent() {
    const c = document.querySelector('#app .content') || document.querySelector('#app .app');
    if (c) { c.classList.remove('eu-flash'); void c.offsetWidth; c.classList.add('eu-flash'); }
  }
  function nav(hash) { location.hash = hash; setTimeout(flashContent, 80); }
  function drive(tok) {
    switch (tok) {
      case '@load': try { window.__euVizPreset && window.__euVizPreset(); } catch (e) {} nav('#patient'); return bi('Loaded a demo workspace — Patient Review is populated. Tabs are live on the left.', '已加载演示工作区，Patient Review 已填充。左侧标签页可以直接查看。');
      case '@loadcross': try { window.__euVizPreset && window.__euVizPreset(); } catch (e) {} nav('#crossdb'); return bi('Loaded the benchmark — Cross-DB is on the left with the availability matrix.', '已加载跨库比较，跨库页面会显示可用性矩阵。');
      case '@cohortrun': nav('#cohort'); setTimeout(() => { const b = document.querySelector('[data-cohort-run]'); if (b) b.click(); }, 240); return bi('Re-running cohort statistics — watch the panel recompute on the left.', '正在重新运行队列统计，请看左侧面板重新计算。');
      case '@agentrun': try { window.__euAgentPreset && window.__euAgentPreset(); } catch (e) {} nav('#agent'); return bi('Opened a completed Research Agent run — the Summary gate is on the left.', '已打开一个完成的 Research Agent run，左侧是 Summary gate。');
    }
    return null;
  }

  /* ---- intent on free text ---- */
  function handleText(v) {
    if (!v || busy) return;
    thread.push({ user: true, html: esc(v) }); render();
    if (hasPageGuideBackend()) {
      sendBackendMessage(v).then(ok => { if (!ok) fallbackText(v); });
      return;
    }
    fallbackText(v);
  }

  function fallbackText(v) {
    const t = v.toLowerCase();
    if (/\b(guided|whole|run it|do it for me|walk me)\b/.test(t)) { botSay(bi(`Opening Guided Copilot for the full conversational workflow.`, `正在打开 Guided Copilot，用于完整对话式流程。`), []); setTimeout(openGuided, 700); return; }
    if (/\b(load|demo|populate|show me data)\b/.test(t)) { const m = drive('@load'); botSay(m, ctxFor(routeOf()).chips); return; }
    if (/\b(privacy|upload|local|phi)\b/.test(t)) { botSay(ANSWERS.privacy, ctxFor(routeOf()).chips); return; }
    if (/\b(gate|lock|draft|sign)\b/.test(t)) { botSay(ANSWERS.gate, ctxFor(routeOf()).chips); return; }
    if (/\b(quality|missing|coverage|flag)\b/.test(t)) { botSay(ANSWERS.quality, ctxFor(routeOf()).chips); return; }
    if (/\b(how|what is|explain|work)\b/.test(t)) { botSay(ANSWERS.how, ctxFor(routeOf()).chips); return; }
    // restate fallback — never a dead end
    botSay(bi(
      `I’ll treat that as “<em>${esc(v)}</em>”. Page guide supports fixed shortcuts only. Pick an action below, or open Guided Copilot for free-form planning.`,
      `我会把它理解为“<em>${esc(v)}</em>”。页面指南只支持固定快捷操作。请选择下方动作，或打开 Guided Copilot 做自由对话规划。`,
    ), ctxFor(routeOf()).chips);
  }

  function openGuided() {
    // hand off the page context so Guided Copilot continues, not restarts
    try {
      const lastUser = [...thread].reverse().find(t => t.user);
      window.__cpBridge = { ts: Date.now(), route: routeOf(), lastUser: lastUser ? lastUser.html : null };
    } catch (e) {}
    close(); location.hash = '#guided';
  }

  async function greet(force) {
    const r = routeOf();
    const c = ctxFor(r);
    if (force || !thread.length) {
      thread = [{ ctxline: true, html: bi(`on ${labelEn(c)}`, `当前页面：${labelZh(c)}`) }];
      render(); renderChips([]);
      if (hasPageGuideBackend()) {
        try {
          const payload = await ensureSession();
          thread.push({ bot: true, html: localized(payload.reply) });
          render(); renderChips(payload.chips || c.chips);
          return;
        } catch (err) {
          console.warn('[EasyICU] Page guide greeting backend failed', err);
        }
      }
      thread.push({ bot: true, html: c.hi });
      render(); renderChips(c.chips);
    }
  }

  /* ---- open / close ---- */
  function open() {
    dock.classList.add('open'); backdrop.classList.add('open'); fab.hidden = true;
    document.getElementById('cpCtx').textContent = contextText(routeOf());
    greet();
    setTimeout(() => input && input.focus(), 320);
    try { localStorage.setItem('easyicu_dock', '1'); } catch (e) {}
  }
  function close() {
    dock.classList.remove('open'); backdrop.classList.remove('open');
    if (routeOf() !== 'guided') fab.hidden = false;
    try { localStorage.setItem('easyicu_dock', '0'); } catch (e) {}
  }
  function toggle() { dock.classList.contains('open') ? close() : open(); }

  function refreshLanguage() {
    if (!dock || !fab) return;
    fab.setAttribute('aria-label', tx('Open EasyICU page guide', '打开 EasyICU 页面指南'));
    fab.innerHTML = `<span class="fab-mk">${icon('spark', 14)}</span> ${tx('Page guide', '页面指南')}`;
    dock.setAttribute('aria-label', tx('EasyICU page guide', 'EasyICU 页面指南'));
    const name = dock.querySelector('.cp-name');
    const ctx = dock.querySelector('#cpCtx');
    const expand = dock.querySelector('#cpExpand');
    const closeBtn = dock.querySelector('#cpClose');
    const send = dock.querySelector('#cpSend');
    const foot = dock.querySelector('.cp-foot');
    if (name) name.textContent = tx('Page guide', '页面指南');
    if (ctx) ctx.textContent = contextText(routeOf());
    if (expand) {
      expand.setAttribute('title', tx('Open Guided Copilot', '打开 Guided Copilot'));
      expand.setAttribute('aria-label', tx('Open Guided Copilot', '打开 Guided Copilot'));
    }
    if (closeBtn) {
      closeBtn.setAttribute('title', tx('Close', '关闭'));
      closeBtn.setAttribute('aria-label', tx('Close page guide', '关闭页面指南'));
    }
    if (input) {
      input.setAttribute('placeholder', tx('Optional shortcut, e.g. "open Patient Review"...', '可选快捷指令，例如“打开患者审阅”…'));
      input.setAttribute('aria-label', tx('Page guide shortcut', '页面指南快捷指令'));
    }
    if (send) send.setAttribute('aria-label', tx('Run shortcut', '运行快捷指令'));
    if (foot) foot.textContent = tx('Page-specific · safe shortcuts · local only', '当前页面 · 安全快捷操作 · 仅本地');
    if (dock.classList.contains('open')) greet(true);
    else {
      render();
      renderChips(ctxFor(routeOf()).chips);
    }
  }

  /* ---- build ---- */
  function build() {
    backdrop = document.createElement('div'); backdrop.id = 'cpBackdrop';
    fab = document.createElement('button'); fab.id = 'cpFab'; fab.setAttribute('aria-label', tx('Open EasyICU page guide', '打开 EasyICU 页面指南'));
    fab.innerHTML = `<span class="fab-mk">${icon('spark', 14)}</span> ${tx('Page guide', '页面指南')}`;
    dock = document.createElement('aside'); dock.id = 'cpDock'; dock.setAttribute('aria-label', tx('EasyICU page guide', 'EasyICU 页面指南'));
    dock.innerHTML = `
      <div class="cp-head">
        <div class="cp-mk">${icon('spark', 16)}</div>
        <div class="grow"><div class="cp-name">${tx('Page guide', '页面指南')}</div><div class="cp-ctx" id="cpCtx">${contextText(routeOf())}</div></div>
        <button class="cp-iconbtn" id="cpExpand" title="${tx('Open Guided Copilot', '打开 Guided Copilot')}" aria-label="${tx('Open Guided Copilot', '打开 Guided Copilot')}">${icon('grid', 16)}</button>
        <button class="cp-iconbtn" id="cpClose" title="${tx('Close', '关闭')}" aria-label="${tx('Close page guide', '关闭页面指南')}">${icon('stop', 15)}</button>
      </div>
      <div class="cp-scroll" id="cpScroll" role="log" aria-live="polite"></div>
      <div class="cp-suggest" id="cpSuggest"></div>
      <div class="cp-composer">
        <input class="c-in" id="cpInput" placeholder="${tx('Optional shortcut, e.g. "open Patient Review"...', '可选快捷指令，例如“打开患者审阅”…')}" autocomplete="off" aria-label="${tx('Page guide shortcut', '页面指南快捷指令')}" />
        <button class="c-go" id="cpSend" aria-label="${tx('Run shortcut', '运行快捷指令')}">${icon('arrow', 16)}</button>
      </div>
      <div class="cp-foot">${tx('Page-specific · safe shortcuts · local only', '当前页面 · 安全快捷操作 · 仅本地')}</div>`;
    document.body.appendChild(backdrop); document.body.appendChild(fab); document.body.appendChild(dock);

    scroll = dock.querySelector('#cpScroll');
    suggestEl = dock.querySelector('#cpSuggest');
    input = dock.querySelector('#cpInput');

    fab.addEventListener('click', open);
    backdrop.addEventListener('click', close);
    dock.querySelector('#cpClose').addEventListener('click', close);
    dock.querySelector('#cpExpand').addEventListener('click', openGuided);

    suggestEl.addEventListener('click', (e) => {
      const backendBtn = e.target.closest('[data-cp-action]');
      if (backendBtn) {
        const action = backendBtn.dataset.cpAction;
        const label = backendBtn.textContent.trim();
        thread.push({ user: true, html: esc(label) }); render();
        runBackendAction(action, label);
        return;
      }
      const b = e.target.closest('[data-cp]'); if (!b) return;
      const tok = b.dataset.cp; const label = b.textContent.trim();
      thread.push({ user: true, html: esc(label) }); render();
      if (tok === '@guided') { botSay(bi(`Opening Guided Copilot…`, `正在打开 Guided Copilot…`), []); setTimeout(openGuided, 650); return; }
      if (tok.startsWith('@say:')) { botSay(ANSWERS[tok.split(':')[1]] || '…', ctxFor(routeOf()).chips); return; }
      const msg = drive(tok);
      if (msg) { botSay(msg, ctxFor(routeOf()).chips); return; }
      botSay(bi(`Done.`, `完成。`), ctxFor(routeOf()).chips);
    });
    dock.querySelector('#cpSend').addEventListener('click', () => { const v = input.value.trim(); input.value = ''; handleText(v); });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); const v = input.value.trim(); input.value = ''; handleText(v); } });

    // react to navigation: hide FAB on the full guided screen; update context if open
    window.addEventListener('hashchange', () => {
      const r = routeOf();
      if (r === 'guided') { close(); fab.hidden = true; }
      else { if (!dock.classList.contains('open')) fab.hidden = false; }
      if (r !== lastRoute && r !== 'guided') {
        // Keep the context label in sync with the route even while the dock is
        // closed — it lives in the DOM and otherwise keeps its build-time value.
        const ctxEl = document.getElementById('cpCtx');
        if (ctxEl) ctxEl.textContent = contextText(r);
        if (dock.classList.contains('open')) appendRouteGreeting(r);
      }
      lastRoute = r;
    });
    window.addEventListener('easyicu:languagechange', refreshLanguage);

    lastRoute = routeOf();
    if (routeOf() === 'guided') fab.hidden = true;
  }

  function init() {
    build();
    window.EUPageGuide = { open, close, toggle, refreshLanguage };
    window.EUCopilot = window.EUPageGuide; // compatibility for existing shell shortcuts
    refreshLanguage();
    setTimeout(refreshLanguage, 0);
    setTimeout(refreshLanguage, 250);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
