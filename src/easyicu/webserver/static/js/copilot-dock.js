/* EasyICU — Copilot dock controller.
   A persistent companion for the classic workspace. Body-level (survives
   route re-renders). It reads the current screen for context, answers
   page-specific questions, and DRIVES the GUI (navigate, load, run) so the
   user sees the workspace change while chatting. Can escalate to the full
   guided Copilot. Evidence-bound, demo-safe. */
(function () {
  let dock, fab, backdrop, scroll, suggestEl, input;
  let thread = [];
  let busy = false;
  let lastRoute = '';

  /* ---- per-route context: greeting + chips + answers ---- */
  const CTX = {
    entry: {
      label: 'Home',
      hi: `I’m your EasyICU companion. I can explain any screen, drive it for you, or run a whole study by chat.`,
      chips: [['Start a guided study', '@guided', 'act'], ['How does EasyICU work?', '@say:how'], ['Load a demo workspace', '@load', 'act']],
    },
    extraction: {
      label: 'Data Extraction',
      hi: `You’re in Data Extraction — one click runs the recommended tables, or you can customize the cohort, modules, and export yourself. Want me to walk it, or just run a guided extraction?`,
      chips: [['How does extraction work?', '@say:extract'], ['What gets exported?', '@say:export'], ['Run a guided study', '@guided', 'act']],
    },
    patient: {
      label: 'Patient Review',
      hi: `This is Patient Review — tables, time series, a patient overview, and data-quality flags. I can load a demo workspace so it’s populated.`,
      chips: [['Load demo workspace', '@load', 'act'], ['What’s in each tab?', '@say:tabs'], ['Explain data-quality flags', '@say:quality']],
    },
    cohort: {
      label: 'Cohort Statistics',
      hi: `Cohort Statistics compares groups, audits coverage, and reclassifies SOFA. I can re-run it or explain what you’re seeing.`,
      chips: [['Re-run statistics', '@cohortrun', 'act'], ['Explain SOFA reclassification', '@say:sofa'], ['What’s the comparison?', '@say:contrast']],
    },
    crossdb: {
      label: 'Cross-DB Benchmark',
      hi: `Cross-DB Benchmark applies one cohort definition across ICU databases. I can load it or explain where databases diverge.`,
      chips: [['Load the benchmark', '@loadcross', 'act'], ['Which databases overlap?', '@say:overlap'], ['Run a guided comparison', '@guided', 'act']],
    },
    agent: {
      label: 'Agent Projects',
      hi: `The Research Agent runs an auditable pipeline and drafts findings — but the draft stays gated until checks pass. Want the run, or the reasoning?`,
      chips: [['Why is the draft locked?', '@say:gate'], ['Show a completed run', '@agentrun', 'act'], ['Run a guided study', '@guided', 'act']],
    },
    states: { label: 'Workspace States', hi: `This is the states reference — loading, empty, no-data, error, blocked, success. Ask me when each one shows.`, chips: [['When do states show?', '@say:states'], ['Start a guided study', '@guided', 'act']] },
    settings: { label: 'Settings', hi: `Settings are local-first and reversible. Ask me what any option does.`, chips: [['Is my data uploaded?', '@say:privacy'], ['Explain the evidence gate', '@say:gate']] },
    tutorial: { label: 'Get Started', hi: `Get Started orients you to the workflow. I can run the whole thing by chat instead.`, chips: [['Run a guided study', '@guided', 'act'], ['How does EasyICU work?', '@say:how']] },
  };

  const ANSWERS = {
    how: `EasyICU runs locally and moves a study through four stages: <strong>frame → extract → review → analyze & draft</strong>. You can drive each panel yourself, or let me run the whole thing by chat. Nothing is uploaded.`,
    extract: `Extraction has a one-click <strong>recommended</strong> path — first ICU stay, first 24h, six core feature modules — that writes analysis-ready frames plus a reproducible manifest. Need more control? The <strong>Customize</strong> panel opens cohort criteria, feature modules, and export format. Either way, nothing runs on incomplete data.`,
    export: `An export bundle is the concept data plus a <strong>manifest.json</strong> — not the figures. Code, tables, and figures come from a Research Agent run, with an evidence ledger.`,
    tabs: `<ul class="mb-list"><li><strong>Data Tables</strong> — raw stay-level rows</li><li><strong>Time Series</strong> — hourly trajectories</li><li><strong>Patient Overview</strong> — one stay at a glance</li><li><strong>Data Quality</strong> — coverage, ranges, missingness</li></ul>`,
    quality: `Flags mark concepts that are sparse or out-of-range. They protect your denominators — a flagged module is surfaced before it can bias a model. Want me to load the workspace so you can see them?`,
    sofa: `SOFA reclassification shows how patients move between severity bands when you recompute the score — it’s a sensitivity check, not a new finding.`,
    contrast: `The default contrast is Survived vs Deceased; you can switch to age groups, sex, length-of-stay, or Sepsis vs non-sepsis. All values shown are seeded demo numbers.`,
    overlap: `Only concepts present in every selected database can be compared fairly — the availability matrix shows where one is missing. Lactate and MAP usually overlap; some scores are partial.`,
    gate: `Drafting is a deliberate second stage. A claim may only be written once it traces to a logged artifact — denominators resolved, coverage above threshold, tables reproducing from the manifest — and a human signs off. That’s why the draft is locked until then.`,
    states: `Every data surface passes through the same six: loading (skeletons), empty (first run), no-data (0 results), error (recoverable), blocked (gated), success. The reference page lets you preview each.`,
    privacy: `No. EasyICU is local-first and the guarantee is enforced — extraction, review, and analysis run on your machine. Only the agent’s plan text can ever leave, and only if you explicitly enable it. Never patient rows.`,
  };

  function routeOf() { const r = (location.hash || '#entry').slice(1); return CTX[r] ? r : (window.SCREENS && window.SCREENS[r] ? r : 'entry'); }
  function ctxFor(r) { return CTX[r] || CTX.entry; }

  /* ---- rendering ---- */
  function render() {
    scroll.innerHTML = thread.map(t => {
      if (t.typing) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble"><div class="typing"><span></span><span></span><span></span></div></div></div></div>`;
      if (t.ctxline) return `<div class="cp-ctxline"><span class="dotline"></span>${t.html}<span class="dotline"></span></div>`;
      if (t.bot) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble">${t.html}</div></div></div>`;
      return `<div class="msg user"><div class="m-ava">LK</div><div class="m-body"><div class="m-bubble">${t.html}</div></div></div>`;
    }).join('');
    scroll.scrollTop = scroll.scrollHeight + 400;
  }
  function renderChips(chips) {
    suggestEl.innerHTML = (chips || []).map(([label, tok, cls]) => `<button class="suggest-chip ${cls || ''}" data-cp="${tok}">${label}</button>`).join('');
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
      case '@load': try { window.__euVizPreset && window.__euVizPreset(); } catch (e) {} nav('#patient'); return 'Loaded a demo workspace — Patient Review is populated. Tabs are live on the left.';
      case '@loadcross': try { window.__euVizPreset && window.__euVizPreset(); } catch (e) {} nav('#crossdb'); return 'Loaded the benchmark — Cross-DB is on the left with the availability matrix.';
      case '@cohortrun': nav('#cohort'); setTimeout(() => { const b = document.querySelector('[data-cohort-run]'); if (b) b.click(); }, 240); return 'Re-running cohort statistics — watch the panel recompute on the left.';
      case '@agentrun': try { window.__euAgentPreset && window.__euAgentPreset(); } catch (e) {} nav('#agent'); return 'Opened a completed Research Agent run — the Summary gate is on the left.';
    }
    return null;
  }

  /* ---- intent on free text ---- */
  function handleText(v) {
    if (!v || busy) return;
    thread.push({ user: true, html: esc(v) }); render();
    const t = v.toLowerCase();
    if (/\b(guided|whole|run it|do it for me|walk me)\b/.test(t)) { botSay(`Opening the guided Copilot — it’ll run the full study by chat.`, []); setTimeout(openGuided, 700); return; }
    if (/\b(load|demo|populate|show me data)\b/.test(t)) { const m = drive('@load'); botSay(m, ctxFor(routeOf()).chips); return; }
    if (/\b(privacy|upload|local|phi)\b/.test(t)) { botSay(ANSWERS.privacy, ctxFor(routeOf()).chips); return; }
    if (/\b(gate|lock|draft|sign)\b/.test(t)) { botSay(ANSWERS.gate, ctxFor(routeOf()).chips); return; }
    if (/\b(quality|missing|coverage|flag)\b/.test(t)) { botSay(ANSWERS.quality, ctxFor(routeOf()).chips); return; }
    if (/\b(how|what is|explain|work)\b/.test(t)) { botSay(ANSWERS.how, ctxFor(routeOf()).chips); return; }
    // restate fallback — never a dead end
    botSay(`I’ll treat that as “<em>${esc(v)}</em>”. In this demo I can explain the current screen, load/run it for you, or open the full guided study — tap one below, or rephrase.`, ctxFor(routeOf()).chips);
  }

  function openGuided() {
    // hand off the dock conversation so the full Copilot continues, not restarts
    try {
      const lastUser = [...thread].reverse().find(t => t.user);
      window.__cpBridge = { ts: Date.now(), route: routeOf(), lastUser: lastUser ? lastUser.html : null };
    } catch (e) {}
    close(); location.hash = '#guided';
  }

  function greet(force) {
    const r = routeOf();
    const c = ctxFor(r);
    if (force || !thread.length) {
      thread = [{ ctxline: true, html: `on ${c.label}` }, { bot: true, html: c.hi }];
      render(); renderChips(c.chips);
    }
  }

  /* ---- open / close ---- */
  function open() {
    dock.classList.add('open'); backdrop.classList.add('open'); fab.hidden = true;
    document.getElementById('cpCtx').textContent = 'on ' + ctxFor(routeOf()).label;
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

  /* ---- build ---- */
  function build() {
    backdrop = document.createElement('div'); backdrop.id = 'cpBackdrop';
    fab = document.createElement('button'); fab.id = 'cpFab'; fab.setAttribute('aria-label', 'Open EasyICU Copilot');
    fab.innerHTML = `<span class="fab-mk">${icon('spark', 14)}</span> Copilot`;
    dock = document.createElement('aside'); dock.id = 'cpDock'; dock.setAttribute('aria-label', 'EasyICU Copilot');
    dock.innerHTML = `
      <div class="cp-head">
        <div class="cp-mk">${icon('spark', 16)}</div>
        <div class="grow"><div class="cp-name">Copilot</div><div class="cp-ctx" id="cpCtx">companion</div></div>
        <button class="cp-iconbtn" id="cpExpand" title="Open full guided study" aria-label="Open full guided study">${icon('grid', 16)}</button>
        <button class="cp-iconbtn" id="cpClose" title="Close" aria-label="Close Copilot">${icon('stop', 15)}</button>
      </div>
      <div class="cp-scroll" id="cpScroll" role="log" aria-live="polite"></div>
      <div class="cp-suggest" id="cpSuggest"></div>
      <div class="cp-composer">
        <input class="c-in" id="cpInput" placeholder="Ask about this screen, or say “run a guided study”…" autocomplete="off" aria-label="Message Copilot" />
        <button class="c-go" id="cpSend" aria-label="Send">${icon('arrow', 16)}</button>
      </div>
      <div class="cp-foot">Context-aware · drives the workspace · evidence-bound</div>`;
    document.body.appendChild(backdrop); document.body.appendChild(fab); document.body.appendChild(dock);

    scroll = dock.querySelector('#cpScroll');
    suggestEl = dock.querySelector('#cpSuggest');
    input = dock.querySelector('#cpInput');

    fab.addEventListener('click', open);
    backdrop.addEventListener('click', close);
    dock.querySelector('#cpClose').addEventListener('click', close);
    dock.querySelector('#cpExpand').addEventListener('click', openGuided);

    suggestEl.addEventListener('click', (e) => {
      const b = e.target.closest('[data-cp]'); if (!b) return;
      const tok = b.dataset.cp; const label = b.textContent.trim();
      thread.push({ user: true, html: esc(label) }); render();
      if (tok === '@guided') { botSay(`Opening the guided Copilot…`, []); setTimeout(openGuided, 650); return; }
      if (tok.startsWith('@say:')) { botSay(ANSWERS[tok.split(':')[1]] || '…', ctxFor(routeOf()).chips); return; }
      const msg = drive(tok);
      if (msg) { botSay(msg, ctxFor(routeOf()).chips); return; }
      botSay(`Done.`, ctxFor(routeOf()).chips);
    });
    dock.querySelector('#cpSend').addEventListener('click', () => { const v = input.value.trim(); input.value = ''; handleText(v); });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); const v = input.value.trim(); input.value = ''; handleText(v); } });

    // react to navigation: hide FAB on the full guided screen; update context if open
    window.addEventListener('hashchange', () => {
      const r = routeOf();
      if (r === 'guided') { close(); fab.hidden = true; }
      else { if (!dock.classList.contains('open')) fab.hidden = false; }
      if (dock.classList.contains('open') && r !== lastRoute && r !== 'guided') {
        document.getElementById('cpCtx').textContent = 'on ' + ctxFor(r).label;
        thread.push({ ctxline: true, html: `now on ${ctxFor(r).label}` });
        thread.push({ bot: true, html: ctxFor(r).hi });
        render(); renderChips(ctxFor(r).chips);
      }
      lastRoute = r;
    });

    lastRoute = routeOf();
    if (routeOf() === 'guided') fab.hidden = true;
  }

  function init() { build(); window.EUCopilot = { open, close, toggle }; }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
