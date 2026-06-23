/* Screen: Research Copilot — conversational mode (v2).
   A branching, forgiving conversation that drives the whole EasyICU workflow.
   Highlights over v1:
     • Three real branches (predict / cross-DB / quality) with distinct cards
     • One card per step; completed steps collapse to editable one-line summaries
     • Edit / rewind any earlier decision; downstream resets cleanly
     • Inline no-data recovery (reuses the workspace state library)
     • Composer understands shallow intents (counts, why, back, run-it-all)
     • "Why this step?" rationale on every card
     • Express lane that autopilots to the gate — then stops for a human
     • State continuity: choices carry into the classic workspace
   Evidence-bound throughout: outputs are clearly seeded demos; the draft is gated. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  /* ============== study panel model ============== */
  const STUDY = [
    ['question', 'Research question', 'spark'],
    ['data', 'Data source', 'flask'],
    ['cohort', 'Cohort', 'cohort'],
    ['concepts', 'Feature modules', 'layers'],
    ['extract', 'Extraction', 'extract'],
    ['review', 'Review', 'eye'],
    ['analysis', 'Analysis run', 'agent'],
    ['draft', 'Manuscript draft', 'shield'],
  ];
  const STEP_INDEX = {}; STUDY.forEach(([id], i) => STEP_INDEX[id] = i);

  /* one genuine clarifying question per branch (reciprocal turn) */
  const CLARIFY = {
    predict: { q: `Quick check before I build the plan — which mortality endpoint do you mean?`, opts: [['In-hospital mortality', 'in-hospital'], ['28-day mortality', '28-day'], ['ICU mortality', 'ICU']] },
    crossdb: { q: `How many databases should we compare?`, opts: [['All six', 'all 6 databases'], ['A focused three', '3 databases'], ['Let me pick', 'a custom set']] },
    quality: { q: `Should I audit everything, or focus on the modelling features?`, opts: [['Everything (19 modules)', 'all 19 modules'], ['Modelling features only', 'the modelling features']] },
  };

  /* ============== branch configs ============== */
  const BRANCH = {
    predict: {
      chip: 'Model an ICU outcome',
      frame: '“Among Sepsis-3 patients, do first-24h bedside features predict in-hospital mortality, and does adding lactate improve it?”',
      plan: [['Question', 'Sepsis-3 · 24h features → mortality'], ['Outcome', 'In-hospital mortality'], ['Comparator', 'Model ± lactate'], ['Design', 'Retrospective · single-center demo']],
      cohortKind: 'cohort',
      runTasks: [['Cohort summary', '0:06'], ['Table 1', '0:11'], ['Missingness audit', '0:09'], ['LR + SOFA + lactate', '0:48'], ['ROC · Calibration', '0:22']],
      reviewTitle: 'Quick review · Table 1',
      findings: 'First-24h lactate, SOFA max, and age were the strongest predictors; adding lactate raised AUC by ~0.03. <span style="color:var(--ink-4);">Seeded demo outputs — confirm before any claim.</span>',
      openTarget: 'patient',
      why: {
        question: 'A vague aim (“does lactate matter”) isn’t testable. I bind it to an outcome, a window, and a comparator so every later step has a clear target.',
        data: 'Demo data lets you learn the flow with zero risk; real data stays on your machine. The choice only changes the source, not the steps.',
        cohort: 'The cohort defines your denominator. Getting inclusion right here is what keeps every downstream rate and p-value honest.',
        concepts: 'I load only the modules your question needs, then audit their coverage — sparse features get flagged before they can bias a model.',
        extract: 'Normalizing once, up front, means every panel and the agent read the same frozen frames — that’s what makes the run reproducible.',
        review: 'A quick human look at Table 1 catches obvious data problems before we spend a run on them.',
        analysis: 'Five deterministic steps, each with an evidence contract. I draft findings only if all of them pass.',
        draft: 'Drafting is gated on purpose: a claim may only be written once it traces to a logged artifact and a human signs off.',
      },
    },
    crossdb: {
      chip: 'Compare databases',
      frame: '“Does the sepsis mortality signal replicate across ICU databases, and where do feature distributions diverge?”',
      plan: [['Question', 'Sepsis cohort across databases'], ['Comparison', '6 standardized concepts'], ['Databases', 'MIMIC-IV · eICU · AUMC …'], ['Design', 'Cross-database benchmark · demo']],
      cohortKind: 'databases',
      runTasks: [['Align concepts', '0:08'], ['Per-database summaries', '0:14'], ['Distribution deltas', '0:19'], ['Availability matrix', '0:11'], ['Concordance check', '0:16']],
      reviewTitle: 'Quick review · Availability matrix',
      findings: 'The mortality signal direction held across 6 databases; lactate and MAP distributions diverged most between MIMIC-IV and AUMC. <span style="color:var(--ink-4);">Seeded demo outputs — confirm before any claim.</span>',
      openTarget: 'crossdb',
      why: {
        question: 'Replication is the real test. I frame it as one cohort definition applied identically across databases so differences are about data, not method.',
        data: 'Cross-DB needs ≥2 databases. Demo loads seeded frames for six; real mode connects local roots — nothing is uploaded either way.',
        cohort: 'Here the “cohort” is your set of databases and the shared concept definition applied to each.',
        concepts: 'Only concepts that exist in every selected database can be compared fairly — I keep the shared set and flag the rest.',
        extract: 'Each database is normalized to the same standardized concepts so a “lactate” column means the same thing everywhere.',
        review: 'The availability matrix shows where a concept is missing in a database before you read too much into a comparison.',
        analysis: 'Per-database summaries plus distribution deltas, with a concordance check so we don’t over-claim agreement.',
        draft: 'Same gate: cross-database claims must trace to the matrix and the logged deltas, and a human signs off.',
      },
    },
    quality: {
      chip: 'Audit data quality',
      frame: '“Before any modelling, where is this cohort sparse or out-of-range, and which concepts are trustworthy?”',
      plan: [['Goal', 'Coverage + range audit'], ['Scope', '19 feature modules'], ['Output', 'Trust map + flags'], ['Design', 'Pre-analysis QC · demo']],
      cohortKind: 'cohort',
      runTasks: [['Per-concept coverage', '0:07'], ['Range / outlier scan', '0:12'], ['Missingness pattern', '0:10'], ['Density by hour', '0:09'], ['Trust scoring', '0:08']],
      reviewTitle: 'Quick review · Coverage audit',
      findings: 'Vitals and chemistry cleared coverage thresholds; ventilator and renal were sparse and flagged before any modelling. <span style="color:var(--ink-4);">Seeded demo outputs — confirm before any claim.</span>',
      openTarget: 'cohort',
      why: {
        question: 'Modelling on untrusted data wastes a run. I make the first deliverable a coverage and range audit, not a result.',
        data: 'Same sources as any study — the audit just reads them first.',
        cohort: 'The audit scope is the same cohort you’d model later, so the trust map actually applies.',
        concepts: 'I scan every module you might use so nothing sparse slips silently into a later analysis.',
        extract: 'Frames are normalized so coverage and ranges are measured against consistent definitions.',
        review: 'The coverage table is the deliverable here — a quick read tells you what’s safe to use.',
        analysis: 'Coverage, outliers, missingness pattern, density, and a trust score — no effect estimates, by design.',
        draft: 'Even a QC summary is gated: every flag must trace to the scan and a human confirms it.',
      },
    },
  };

  /* ============== depth axis: HOW FAR the user wants to go ==============
     Orthogonal to the research-question branch above. The branch decides WHAT
     we study; depth decides where the study STOPS. Every depth can be extended
     later, and the study rail draws a finish line at the chosen goal. */
  const DEPTH_ORDER = ['extract', 'review', 'full'];
  const DEPTH = {
    extract: {
      label: 'Extract only', goal: 'extract',
      chip: 'Just a cohort & data',
      hi: `Got it — an <strong>extract-only</strong> run. I’ll stop once your cohort is resolved and packaged, and you leave with analysis-ready frames plus a reproducible manifest.`,
    },
    review: {
      label: 'Extract + review', goal: 'review',
      chip: 'Data, then a visual review',
      hi: `Good — <strong>extract &amp; review</strong>. I’ll pull the data and prepare a quick visual review, then hand you a populated workspace. No agent run unless you ask.`,
    },
    full: {
      label: 'Full study', goal: 'draft',
      chip: 'All the way to a gated draft',
      hi: `The full ride — <strong>extract → review → analyse → gated draft</strong>. Everything runs locally and the draft stays locked until checks pass.`,
    },
  };

  /* ============== runtime state ============== */
  let branch, depth, dataMode, mods, cohortPhase, extractPhase, runPhase, draftPhase;
  let thread, chips, busy, expandedStep, whyOpen, autop, patientN, clarified, outputsReady, diffExpanded, liveAgentRun, workspaceSnapshot, workspaceSnapshotPath;
  let studyParams;   // dynamic params extracted from clarify answers + free text

  const DEFAULT_MODS = ['Demographics', 'Vital signs', 'Lab — Chemistry', 'SOFA-2 scores', 'Sepsis-3 (SOFA-2)', 'Outcome'];
  function reset() {
    branch = 'predict'; depth = 'full'; dataMode = 'demo'; mods = DEFAULT_MODS.slice();
    cohortPhase = 'normal'; extractPhase = 'run'; runPhase = 'run'; draftPhase = 'gate';
    thread = []; chips = []; busy = false; expandedStep = 'question'; whyOpen = {}; autop = false; patientN = 10; clarified = null; outputsReady = false; diffExpanded = false; liveAgentRun = null; workspaceSnapshot = null; workspaceSnapshotPath = null;
    studyParams = { outcome: 'In-hospital mortality', window: 'first 24h', exposure: 'lactate', scope: 'all 19 modules', caught: null };
    studyStatus = {}; studyVal = {};
    gen++;
    STUDY.forEach(([id]) => { studyStatus[id] = 'pending'; });
  }
  let studyStatus = {}, studyVal = {};
  let gen = 0;   // bumped on every (re)entry; stale timers check against it

  function activeExportSource() {
    if (window.EU_SOURCES && window.EU_SOURCES.activeSource) return window.EU_SOURCES.activeSource();
    const reg = window.EU_WORKSPACE_REGISTRY || {};
    return (reg.sources || []).find(s => s.path === reg.active_path) || (reg.sources || []).find(s => s.ok) || null;
  }
  function activeExportLabel() {
    const src = activeExportSource();
    if (!src) return 'local export';
    const sum = src.summary || {};
    const parts = [];
    if (sum.stays != null) parts.push(Number(sum.stays).toLocaleString() + ' stays');
    if (sum.modules != null) parts.push(Number(sum.modules).toLocaleString() + ' modules');
    return `${src.label || src.database || 'local export'}${parts.length ? ' · ' + parts.join(' · ') : ''}`;
  }
  function realMode() { return dataMode !== 'demo'; }
  function snapshotSummary() {
    const live = liveAgentRun && liveAgentRun.result;
    if (live && live.summary) return live.summary;
    if (workspaceSnapshot && workspaceSnapshot.summary) return workspaceSnapshot.summary;
    const src = activeExportSource();
    return (src && src.summary) || {};
  }
  function snapshotCohort() {
    const live = liveAgentRun && liveAgentRun.result;
    if (live && live.cohort) return live.cohort;
    return (workspaceSnapshot && workspaceSnapshot.cohort) || {};
  }
  function fmtInt(v, fallback) {
    const n = Number(v);
    return Number.isFinite(n) ? n.toLocaleString() : (fallback || 'n/a');
  }
  function fmtPct(v, fallback) {
    const n = Number(v);
    return Number.isFinite(n) ? `${n.toFixed(Math.abs(n) < 10 && n % 1 ? 1 : 1).replace(/\.0$/, '')}%` : (fallback || 'n/a');
  }
  function fmtNum(v, fallback) {
    const n = Number(v);
    return Number.isFinite(n) ? String(Math.round(n * 10) / 10) : (fallback || 'n/a');
  }
  function cohortLine() {
    if (!realMode()) return `${patientN} stays · 20% mort.`;
    const s = snapshotSummary();
    const mort = Number.isFinite(Number(s.mortality)) ? ` · ${fmtPct(s.mortality)} mort.` : '';
    return `${fmtInt(s.stays, 'registered')} stays${mort}`;
  }
  function extractLine() {
    if (!realMode()) return `${patientN * 24} time points · 94%`;
    const s = snapshotSummary();
    const rows = Number.isFinite(Number(s.total_rows)) ? `${fmtInt(s.total_rows)} rows` : `${fmtInt(s.stays, 'registered')} stays`;
    const modules = Number.isFinite(Number(s.modules)) ? ` · ${fmtInt(s.modules)} modules` : '';
    return `${rows}${modules}`;
  }
  function analysisLine() {
    const live = liveAgentRun && liveAgentRun.result;
    if (live) return `preflight · ${fmtInt((live.artifacts || []).length)} artifacts`;
    return realMode() ? 'local preflight' : '5 steps · 6 artifacts';
  }
  function reviewLine() {
    if (!realMode()) return BRANCH[branch].reviewTitle.replace('Quick review · ', '');
    return thread && thread.some(t => t.card && t.step === 'review') ? 'export snapshot' : 'skipped';
  }
  function analysisTasks() {
    if (realMode()) {
      return [
        ['Resolve export source', 'done'],
        ['Summarise cohort snapshot', 'done'],
        ['Evaluate evidence gate', 'done'],
        ['Write local artifacts', 'done'],
      ];
    }
    return BRANCH[branch].runTasks;
  }
  function loadWorkspaceSnapshot(src) {
    if (!src || !src.path || !window.EU_API || !window.EU_API.loadWorkspaceSummary) return Promise.resolve(null);
    if (workspaceSnapshot && workspaceSnapshotPath === src.path) return Promise.resolve(workspaceSnapshot);
    workspaceSnapshotPath = src.path;
    return window.EU_API.loadWorkspaceSummary(src.path).then(snapshot => {
      workspaceSnapshot = snapshot;
      renderThread();
      renderAside();
      return snapshot;
    }).catch(() => null);
  }

  /* ---- dynamic plan/frame from extracted params ---- */
  function planFor(b) {
    if (b === 'predict') return [
      ['Question', `Sepsis-3 · ${studyParams.window} features → mortality`],
      ['Outcome', studyParams.outcome],
      ['Comparator', `Model ± ${studyParams.exposure}`],
      ['Design', realMode() ? 'Retrospective · local export preflight' : 'Retrospective · single-center demo'],
    ];
    if (b === 'crossdb') return [
      ['Question', 'Sepsis cohort across databases'],
      ['Comparison', '6 standardized concepts'],
      ['Databases', `${dbCount()} selected`],
      ['Design', 'Cross-database benchmark · demo'],
    ];
    return [
      ['Goal', 'Coverage + range audit'],
      ['Scope', studyParams.scope],
      ['Output', 'Trust map + flags'],
      ['Design', 'Pre-analysis QC · demo'],
    ];
  }
  function frameFor(b) {
    if (b === 'predict') return `“Among Sepsis-3 patients, do ${studyParams.window} bedside features predict ${studyParams.outcome.toLowerCase()}, and does adding ${studyParams.exposure} improve it?”`;
    return BRANCH[b].frame;
  }
  /* turn a clarify answer into a real param change (not just a label) */
  function applyClarify(b, detail) {
    const d = detail.toLowerCase();
    if (b === 'predict') {
      if (/28/.test(d)) { studyParams.outcome = '28-day mortality'; return '28-day mortality'; }
      if (/icu/.test(d)) { studyParams.outcome = 'ICU mortality'; return 'ICU mortality'; }
      studyParams.outcome = 'In-hospital mortality'; return 'in-hospital mortality';
    }
    if (b === 'crossdb') {
      if (/6|six/.test(d)) { _dbCount = 6; return 'all six databases'; }
      if (/3|three/.test(d)) { _dbCount = 3; return 'three databases'; }
      return 'a custom set';
    }
    studyParams.scope = /19|every/.test(d) ? 'all 19 modules' : 'modelling features only';
    return studyParams.scope;
  }
  /* pull entities out of a free-text research question */
  function extractEntities(text) {
    const t = text.toLowerCase();
    const found = [];
    const exposures = [['lactate', 'lactate'], ['sofa', 'SOFA'], ['\\bmap\\b', 'MAP'], ['creatinine', 'creatinine'], ['heart rate', 'heart rate'], ['wbc|white cell', 'WBC']];
    for (const [re, name] of exposures) { if (new RegExp(re).test(t)) { studyParams.exposure = name; found.push(name); break; } }
    const wm = t.match(/(?:first\s*)?(\d{1,3})\s*(?:h\b|hr|hour)/);
    if (wm) { studyParams.window = 'first ' + wm[1] + 'h'; found.push(studyParams.window); }
    else if (/48/.test(t)) { studyParams.window = 'first 48h'; found.push('first 48h'); }
    if (/28[\s-]*day/.test(t)) { studyParams.outcome = '28-day mortality'; found.push('28-day'); }
    else if (/icu\s*mortalit|icu\s*death/.test(t)) { studyParams.outcome = 'ICU mortality'; found.push('ICU mortality'); }
    const sm = t.match(/\b(\d{1,3})\s*(?:patient|case|stay|subject)/);
    if (sm) { patientN = Math.max(5, Math.min(50, parseInt(sm[1], 10))); found.push(patientN + ' stays'); }
    studyParams.caught = found.length ? found.join(' · ') : null;
    return found;
  }
  function pickBranch(text) {
    const t = text.toLowerCase();
    if (/database|cross|compare|replicat|multi-?center|across/.test(t)) return 'crossdb';
    if (/quality|missing|coverage|audit|qc|trust|sparse/.test(t)) return 'quality';
    return 'predict';
  }
  /* did the user already pin the endpoint, so we can skip the clarify? */
  function endpointPinned(text) { return /in-?hospital|28[\s-]*day|icu\s*mortalit/.test(text.toLowerCase()); }

  /* ============== study panel ============== */
  function setStudy(map) { Object.assign(studyStatus, map); renderAside(); }
  function setVal(map) { Object.assign(studyVal, map); renderAside(); }
  function markThrough(step, activeStatus) {
    // steps before `step` done, `step` -> activeStatus, after -> pending (unless locked)
    const si = STEP_INDEX[step];
    STUDY.forEach(([id], i) => {
      if (i < si) studyStatus[id] = 'done';
      else if (i === si) studyStatus[id] = activeStatus || 'active';
      else studyStatus[id] = (studyStatus[id] === 'locked') ? 'locked' : 'pending';
    });
    renderAside();
  }

  /* ============== conversation engine ============== */
  function scrollEnd() { const sc = document.getElementById('gdScroll'); if (sc) sc.scrollTop = sc.scrollHeight + 600; }
  function esc(s) { return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
  function stripTags(s) { return String(s).replace(/<[^>]*>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').trim(); }

  function pushUser(text) { thread.push({ user: true, html: esc(text) }); renderThread(); }

  function go(id, userText) {
    if (busy) return;
    const st = STATES[id];
    if (!st) return;
    currentId = id;
    if (userText != null) pushUser(userText);
    chips = []; renderChips();
    busy = true; thread.push({ typing: true }); renderThread();
    const delay = st.delay != null ? st.delay : 560;
    const myGen = gen;
    setTimeout(() => {
     if (myGen !== gen) return;
     try {
      thread = thread.filter(t => !t.typing); busy = false;
      const botMsgs = typeof st.bot === 'function' ? st.bot() : (st.bot || []);
      botMsgs.forEach(html => thread.push({ bot: true, html }));
      if (st.step && st.card) {
        if (!thread.some(t => t.card && t.step === st.step)) thread.push({ card: true, step: st.step });
        expandedStep = st.step;
      }
      if (st.once) thread.push({ once: st.once });
      chips = (typeof st.chips === 'function' ? st.chips() : st.chips) || [];
      if (st.markStep) markThrough(st.markStep, st.markStatus || 'active');
      if (st.val) setVal(st.val);
      renderThread(); renderChips();
      if (st.onShown) st.onShown();
     } catch (err) { busy = false; }
    }, delay);
  }
  let currentId = 'welcome';

  /* edit / rewind to an earlier step */
  const STEP_STATE = { question: 'frame', data: 'toData', cohort: 'toCohort', concepts: 'toConcepts' };
  /* clicking a Study-panel step: editable steps rewind; later/process steps just scroll into view */
  function jumpToStep(step) {
    if (busy) return;
    const editable = STEP_STATE[step] && STEP_INDEX[step] < STEP_INDEX[expandedStep];
    if (editable) { editStep(step); return; }
    const host = document.getElementById('gdThread');
    const cards = host ? host.querySelectorAll('.gd-card, .gd-collapsed') : [];
    // find the card/row matching this step by walking thread order
    let n = -1; thread.forEach((t, i) => { if (t.card && t.step === step) n = i; });
    if (n < 0) return;
    // count rendered card/collapsed elements up to that thread index
    let visIdx = 0, target = null;
    thread.forEach((t, i) => { if (t.card) { if (i === n) target = cards[visIdx]; visIdx++; } });
    if (target) { target.classList.remove('eu-flash'); void target.offsetWidth; target.classList.add('eu-flash'); const sc = document.getElementById('gdScroll'); if (sc) sc.scrollTop = target.offsetTop - 16; }
  }
  function editStep(step) {
    if (busy) return;
    const idx = thread.findIndex(t => t.card && t.step === step);
    if (idx < 0) return;
    thread = thread.slice(0, idx + 1);          // keep that card, drop everything after
    expandedStep = step;
    // reset downstream phases
    if (STEP_INDEX[step] <= STEP_INDEX.cohort) cohortPhase = 'normal';
    extractPhase = 'run'; runPhase = 'run'; draftPhase = 'gate';
    markThrough(step, 'active');
    thread.push({ bot: true, html: `Sure — let’s adjust this. Anything downstream will re-run from here.` });
    const st = STATES[STEP_STATE[step]];
    chips = (st && (typeof st.chips === 'function' ? st.chips() : st.chips)) || [];
    renderThread(); renderChips();
  }

  /* ============== timed sub-flows ============== */
  function streamTasks(sel, durs, done, opts) {
    let i = 0;
    let repaired = false;
    const failAt = opts && opts.failAt != null ? opts.failAt : -1;
    const myGen = gen;
    (function step() {
      if (myGen !== gen) return;
      const rows = document.querySelectorAll(sel + ' .gd-task');
      if (!rows.length || i >= rows.length) { done(); return; }
      const r = rows[i];
      r.className = 'gd-task running';
      const tk = r.querySelector('.tk'); if (tk) tk.innerHTML = '<span class="spin sm accent" style="width:11px;height:11px;"></span>';
      setTimeout(() => {
        if (myGen !== gen) return;
        const cur = document.querySelectorAll(sel + ' .gd-task');
        const rr = cur[i];
        // failure → self-repair on the designated step (once)
        if (i === failAt && !repaired) {
          repaired = true;
          if (rr) {
            rr.className = 'gd-task fail';
            const t2 = rr.querySelector('.tk'); if (t2) t2.innerHTML = icon('beaker', 9);
            const d2 = rr.querySelector('.tdur'); if (d2) { d2.textContent = 'singular matrix'; d2.style.color = 'var(--bad)'; }
            const note = rr.querySelector('.tk-note');
            if (!note) { const n = document.createElement('div'); n.className = 'tk-repair'; n.innerHTML = `${icon('refresh', 10)} auto-repair: dropped 1 collinear feature · retrying`; rr.insertAdjacentElement('afterend', n); }
          }
          scrollEnd();
          setTimeout(step, 900);   // retry the same index
          return;
        }
        if (rr) {
          rr.className = 'gd-task done';
          const t2 = rr.querySelector('.tk'); if (t2) t2.innerHTML = icon('check', 10, 3);
          const d2 = rr.querySelector('.tdur'); if (d2) { d2.textContent = durs[i] || ''; d2.style.color = ''; }
          const rep = rr.nextElementSibling; if (rep && rep.classList.contains('tk-repair')) rep.classList.add('done');
          const bar = document.querySelector(sel + '-bar'); if (bar) bar.style.width = Math.round(((i + 1) / cur.length) * 100) + '%';
        }
        i++; scrollEnd(); step();
      }, 380 + Math.random() * 220);
    })();
  }
  function runExtract() {
    streamTasks('#gdExProg', ['0:03', '0:02', '0:05', '0:04', '0:02'], () => {
      extractPhase = 'done'; renderThread();
      if (depth === 'extract') {
        markThrough('extract', 'done');
        thread.push({ bot: true, html: `Done — frames packaged and frozen locally. This is your finish line for an extract-only run.` });
        chips = [['Finish &amp; export', '@finish', 'express'], ['Open in workspace', '@open'], ['Take it further → review', '@extendNext']];
        renderThread(); renderChips();
        if (autop) schedule(() => finishHere());
        return;
      }
      thread.push({ bot: true, html: `Done — the workspace is loaded and frozen for analysis. Want a quick look before we run?` });
      chips = [['Review the data', 'toReview'], (depth === 'full' ? ['Skip to analysis', 'toRun'] : null), ['Open in workspace', '@open']].filter(Boolean);
      renderThread(); renderChips();
      if (autop) schedule(() => go('toReview'));
    });
  }
  /* generated artifacts (Codex/Claude-Code style file diff) */
  const OUTPUTS = [
    ['src/easyicu/pipeline.py', '+148', 'file'],
    ['analysis/table_one.csv', '+22', 'rows'],
    ['analysis/cohort_summary.json', '+18', 'file'],
    ['analysis/roc_curve.png', 'bin', 'viz'],
    ['analysis/calibration.png', 'bin', 'viz'],
    ['manifest.json', '+124', 'shield'],
  ];

  /* ---- inline artifact preview (open a generated file) ---- */
  const ART = {
    'src/easyicu/pipeline.py': { kind: 'code', meta: 'python · 148 lines', body: () =>
      `<span class="ln-c"># EasyICU — generated analysis pipeline (demo)</span>
<span class="ln-k">import</span> pandas <span class="ln-k">as</span> pd
<span class="ln-k">from</span> easyicu <span class="ln-k">import</span> cohort, models

df = cohort.load(<span class="ln-s">"sepsis_demo"</span>, window=<span class="ln-s">"${'${win}'}"</span>)
X = df[[<span class="ln-s">"sofa"</span>, <span class="ln-s">"lactate"</span>, <span class="ln-s">"age"</span>, <span class="ln-s">"map"</span>]]
y = df[<span class="ln-s">"${'${out}'}"</span>]

<span class="ln-k">def</span> <span class="ln-f">fit_and_eval</span>(X, y):
    m = models.logistic(X, y, repair=<span class="ln-k">True</span>)
    <span class="ln-k">return</span> m.auc(), m.calibration()

auc, cal = fit_and_eval(X, y)   <span class="ln-c"># AUC ≈ 0.84 (seeded)</span>
models.export(auc, cal, ledger=<span class="ln-s">"manifest.json"</span>)` },
    'analysis/table_one.csv': { kind: 'table', meta: '22 rows · csv', body: () => [
      ['characteristic', 'survived', 'deceased', 'p'],
      ['n', '8', '2', '—'],
      ['age, mean', '52.1', '64.8', '0.04'],
      ['sofa, median', '5', '9', '0.01'],
      ['lactate mmol/L', '2.1', '3.6', '0.03'],
      ['map mmHg', '78', '69', '0.06'],
    ] },
    'analysis/cohort_summary.json': { kind: 'json', meta: '18 lines · json', body: () => ({
      cohort: 'sepsis_demo', n_stays: patientN, mortality: 0.20, window: studyParams.window,
      outcome: studyParams.outcome, modules: mods.length, coverage: 0.94, seed: 42,
    }) },
    'analysis/roc_curve.png': { kind: 'roc', meta: 'figure · png', auc: 0.84 },
    'analysis/calibration.png': { kind: 'calib', meta: 'figure · png' },
    'manifest.json': { kind: 'json', meta: '124 lines · json', body: () => ({
      run: 'demo-07', steps: 5, artifacts: 6, evidence_contract: 'strict',
      checks: { denominators: true, coverage: true, reproduces: true, model_card: true, signoff: false },
      uploads: 0, tokens: 0,
    }) },
  };
  function artBody(path) {
    const a = ART[path]; if (!a) return '<div class="json-block">No preview.</div>';
    if (a.kind === 'code') {
      let s = a.body().replace('${win}', studyParams.window).replace('${out}', studyParams.outcome === 'In-hospital mortality' ? 'hospital_death' : studyParams.outcome === '28-day mortality' ? 'death_28d' : 'icu_death');
      return `<div class="code-block">${s}</div>`;
    }
    if (a.kind === 'table') {
      const rows = a.body();
      return `<table class="eu-table" style="font-size:11.5px;"><thead><tr>${rows[0].map((h, i) => `<th class="${i ? 'num' : ''}">${h}</th>`).join('')}</tr></thead><tbody>${rows.slice(1).map(r => `<tr>${r.map((c, i) => `<td class="${i ? 'num' : 'key'}">${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }
    if (a.kind === 'json') {
      const j = a.body();
      const s = JSON.stringify(j, null, 2).replace(/"([^"]+)":/g, '<span class="jk">"$1"</span>:').replace(/: (\d+\.?\d*)/g, ': <span class="jn">$1</span>');
      return `<div class="json-block">${s}</div>`;
    }
    if (a.kind === 'roc') return rocSvg(a.auc);
    if (a.kind === 'calib') return calibSvg();
    return '';
  }
  function rocSvg(auc) {
    const W = 320, H = 240, p = 34;
    const pts = [[0, 0], [0.06, 0.42], [0.14, 0.63], [0.28, 0.78], [0.46, 0.88], [0.7, 0.95], [1, 1]];
    const X = v => p + v * (W - 2 * p), Y = v => H - p - v * (H - 2 * p);
    const path = pts.map((pt, i) => `${i ? 'L' : 'M'}${X(pt[0]).toFixed(1)},${Y(pt[1]).toFixed(1)}`).join(' ');
    return `<svg class="roc-svg" viewBox="0 0 ${W} ${H}">
      <line x1="${p}" y1="${H - p}" x2="${W - p}" y2="${p}" stroke="var(--hair-2)" stroke-dasharray="4 4"/>
      <line x1="${p}" y1="${p}" x2="${p}" y2="${H - p}" stroke="var(--hair-2)"/>
      <line x1="${p}" y1="${H - p}" x2="${W - p}" y2="${H - p}" stroke="var(--hair-2)"/>
      <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>
      <text x="${W / 2}" y="${H - 8}" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="var(--ink-4)">false positive rate</text>
      <text x="${W - p - 6}" y="${p + 14}" text-anchor="end" font-family="var(--font-mono)" font-size="11" fill="var(--accent-ink)">AUC ${auc}</text>
    </svg>`;
  }
  function calibSvg() {
    const W = 320, H = 240, p = 34;
    const pts = [[0, 0.02], [0.2, 0.18], [0.4, 0.43], [0.6, 0.58], [0.8, 0.79], [1, 0.96]];
    const X = v => p + v * (W - 2 * p), Y = v => H - p - v * (H - 2 * p);
    const path = pts.map((pt, i) => `${i ? 'L' : 'M'}${X(pt[0]).toFixed(1)},${Y(pt[1]).toFixed(1)}`).join(' ');
    return `<svg class="roc-svg" viewBox="0 0 ${W} ${H}">
      <line x1="${p}" y1="${H - p}" x2="${W - p}" y2="${p}" stroke="var(--hair-2)" stroke-dasharray="4 4"/>
      <line x1="${p}" y1="${p}" x2="${p}" y2="${H - p}" stroke="var(--hair-2)"/>
      <line x1="${p}" y1="${H - p}" x2="${W - p}" y2="${H - p}" stroke="var(--hair-2)"/>
      <path d="${path}" fill="none" stroke="var(--ok)" stroke-width="2.2"/>
      <text x="${W / 2}" y="${H - 8}" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="var(--ink-4)">predicted probability</text>
    </svg>`;
  }
  function openArtifact(path) {
    const a = ART[path]; if (!a) return;
    let back = document.getElementById('gdArt');
    if (!back) {
      back = document.createElement('div'); back.id = 'gdArt'; back.className = 'art-back';
      (document.querySelector('.gd-shell') || document.body).appendChild(back);
    }
    const name = path.split('/').pop();
    back.innerHTML = `
      <div class="art-modal" role="dialog" aria-label="${name}">
        <div class="art-head">
          <span class="ah-ico">${icon(a.kind === 'roc' || a.kind === 'calib' ? 'viz' : a.kind === 'table' ? 'list' : 'file', 14)}</span>
          <div><div class="ah-p">${path}</div><div class="ah-m">${a.meta} · demo · local</div></div>
          <button class="ah-x" data-artclose aria-label="Close">${icon('stop', 14)}</button>
        </div>
        <div class="art-body">${artBody(path)}</div>
        <div class="art-foot"><span class="mono">${icon('shield', 11)} seeded demo artifact — not a real result</span><span class="grow" style="flex:1;"></span><button class="btn sm" data-artclose>Close</button></div>
      </div>`;
    requestAnimationFrame(() => back.classList.add('open'));
    setTimeout(() => back.classList.add('open'), 20);   // fallback if rAF is throttled
  }
  function closeArtifact() { const b = document.getElementById('gdArt'); if (b) b.classList.remove('open'); }

  function diffCard() {
    const live = liveAgentRun && liveAgentRun.result;
    const outputs = live ? (live.artifacts || []).map(a => [a.relative_path || a.name, String(a.bytes || 0) + 'b', 'shield']) : OUTPUTS;
    const shown = diffExpanded ? outputs : outputs.slice(0, 4);
    const rows = shown.map(([p, d, ic]) => `<div class="gd-file" ${live ? '' : `data-artopen="${p}" role="button" tabindex="0"`}><span class="gf-ico">${icon(ic === 'rows' ? 'list' : ic === 'viz' ? 'viz' : ic === 'shield' ? 'shield' : 'file', 13)}</span><span class="gf-path">${esc(p)}</span><span class="gf-delta">${esc(d)}</span></div>`).join('');
    const more = (!diffExpanded && outputs.length > 4) ? `<button class="df-more" data-diffmore>Show ${outputs.length - 4} more files</button>` : '';
    const headMeta = live
      ? `<span class="df-sum"><span class="df-add">${esc((live.gate && live.gate.status) || 'analysis_only')}</span></span>`
      : `<span class="df-sum"><span class="df-add">+312</span><span class="df-del">−0</span></span>`;
    return `
    <div class="gd-diff">
      <div class="df-head">
        <span class="df-ico">${icon('extract', 14)}</span>
        <span class="df-t">${live ? 'Wrote' : 'Generated'} ${outputs.length} files</span>
        ${headMeta}
      </div>
      ${rows}
      ${more}
      <div class="df-foot">
        <button class="btn sm" data-act="open">${icon('eye', 13)} Review artifacts</button>
        <button class="btn sm ghost" data-act="draft">Open in agent</button>
        <span class="grow"></span>
        <span class="mono" style="font-size:10px;color:var(--ink-4);align-self:center;">${live ? 'real preflight · local · evidence-ledgered' : 'demo · local · evidence-ledgered'}</span>
      </div>
    </div>`;
  }
  function renderOutputs(host) {
    if (!outputsReady) return '';
    const live = liveAgentRun && liveAgentRun.result;
    const outputs = live ? (live.artifacts || []).map(a => [a.name || a.relative_path, String(a.bytes || 0) + 'b', 'shield']) : OUTPUTS;
    return `
      <div class="gd-out-sec">
        <div class="os-head">${icon('extract', 11)} Outputs <span style="color:var(--ink-3);">· ${outputs.length} files</span></div>
        ${outputs.map(([p, d, ic]) => `<div class="gd-out" ${live ? '' : `data-artopen="${p}" role="button" tabindex="0"`}><span class="o-ico">${icon(ic === 'rows' ? 'list' : ic === 'viz' ? 'viz' : ic === 'shield' ? 'shield' : 'file', 12)}</span><span class="o-p">${esc(String(p).split('/').pop())}</span><span class="o-d">${esc(d)}</span></div>`).join('')}
      </div>`;
  }

  /* one-off flavour cards (real-data connect / detect) — not part of the step-collapse system */
  const ONCE = {
    folder() {
      const src = activeExportSource();
      const path = src && src.path ? src.path : '~/easyicu/exports/';
      return `
      <div class="gd-card" style="max-width:600px;margin-left:39px;">
        <div class="gc-head"><div class="gc-ico">${icon('folder', 15)}</div><div class="grow"><div class="gc-t">Connect a local export folder</div><div class="gc-sub">read locally · nothing uploaded</div></div></div>
        <div class="gc-body">
          <div class="gd-folder"><span class="fo-ico">${icon('folder', 16)}</span><div><div style="font-weight:600;font-size:12px;">${src ? 'Use active EasyICU export' : 'Choose an ICU export root'}</div><div class="fo-p">${esc(path)}</div></div></div>
          <div class="row wrap gap-6 mt-12"><span class="chip">MIMIC-IV</span><span class="chip">eICU-CRD</span><span class="chip">AUMCdb</span><span class="chip">HiRID</span><span class="chip">SICdb</span></div>
        </div>
        <div class="gd-cardfoot"><button class="btn primary sm" data-go="detect">${icon('folder', 13)} ${src ? 'Use active export' : 'Choose this folder'}</button></div>
      </div>`;
    },
    detect() {
      const src = activeExportSource();
      const path = src && src.path ? src.path : '~/easyicu/exports/';
      const tasks = ['Read folder tree', 'Match known layout', 'Verify concept map', 'Index tables'];
      return `
      <div class="gd-card" style="max-width:600px;margin-left:39px;">
        <div class="gc-head"><div class="gc-ico">${icon('db', 15)}</div><div class="grow"><div class="gc-t">Detecting schema</div><div class="gc-sub mono">${esc(path)}</div></div></div>
        <div class="gc-body">
          <div class="gd-prog" id="gdDetect">${tasks.map(t => `<div class="gd-task queued"><span class="tk">${icon('clock', 9)}</span><span class="grow">${t}</span><span class="tdur"></span></div>`).join('')}</div>
          <div class="indet mt-12"></div>
        </div>
      </div>`;
    },
    detected() {
      const src = activeExportSource();
      return `
      <div class="gd-card" style="max-width:600px;margin-left:39px;">
        <div class="gc-head"><div class="gc-ico" style="background:var(--ok-soft);color:var(--ok);border-color:oklch(88% 0.05 150);">${icon('check', 14, 3)}</div><div class="grow"><div class="gc-t">Detected ${esc(activeExportLabel())}</div><div class="gc-sub">local · verified</div></div></div>
        <div class="gc-body">
          <div class="col gap-6" style="font-size:12px;">
            <div class="setup-row"><span class="k">Export</span><span class="vv">${esc(src && src.path ? src.path : 'local path')}</span></div>
            <div class="setup-row"><span class="k">Modules</span><span class="vv">${esc(src && src.summary && src.summary.modules != null ? String(src.summary.modules) : 'registered')}</span></div>
            <div class="setup-row"><span class="k">Upload</span><span class="vv">none — read locally</span></div>
          </div>
        </div>
        <div class="gd-cardfoot"><button class="btn primary sm" data-go="toCohort">Map concepts &amp; continue ${icon('arrow', 13)}</button></div>
      </div>`;
    },
  };
  function runDetect() {
    streamTasks('#gdDetect', ['0:01', '0:02', '0:02', '0:03'], () => {
      const finish = () => {
        thread = thread.filter(t => !(t.once === 'detect'));   // drop the transient scan card
        thread.push({ bot: true, html: `Recognized <strong>${esc(activeExportLabel())}</strong> — concept map verified. Files stay on your machine.` });
        go('detected');
      };
      loadWorkspaceSnapshot(activeExportSource()).then(finish);
    });
  }

  function runPipeline() {
    const src = dataMode !== 'demo' ? activeExportSource() : null;
    if (src && window.EU_API && window.EU_API.startAgentRun && window.EventSource) {
      runLivePipeline(src);
      return;
    }
    streamTasks('#gdRunProg', BRANCH[branch].runTasks.map(t => t[1]), () => {
      runPhase = 'done';
      const p = document.getElementById('gdRunPill'); if (p) p.outerHTML = '<span class="pill ok" id="gdRunPill"><span class="dot"></span>Complete</span>';
      outputsReady = true;
      renderThread();
      thread.push({ bot: true, html: `Run complete — six artifacts written locally and logged to the evidence ledger. <span style="color:var(--ink-4);">(Step 4 hit a singular matrix; auto-repair dropped one collinear feature and re-fit — logged in the ledger.)</span>` });
      thread.push({ diff: true });
      renderThread(); renderAside();
      thread.push({ bot: true, html: `I’ve drafted findings from these — but the manuscript draft stays <strong>locked</strong> until you sign off.` });
      chips = []; renderThread();
      go('toFindings');
    }, { failAt: 3 });
  }

  function runLivePipeline(src) {
    liveAgentRun = { active: true, result: null, error: null, step: 'Submitting local run' };
    const rows = () => [...document.querySelectorAll('#gdRunProg .gd-task')];
    const setProgress = (current, total, label) => {
      const all = rows();
      const idx = Math.max(0, Math.min(all.length, Number(current || 0)));
      all.forEach((r, i) => {
        if (i < idx) {
          r.className = 'gd-task done';
          const tk = r.querySelector('.tk'); if (tk) tk.innerHTML = icon('check', 10, 3);
          const d = r.querySelector('.tdur'); if (d && !d.textContent) d.textContent = 'done';
        } else if (i === idx && idx < all.length) {
          r.className = 'gd-task running';
          const tk = r.querySelector('.tk'); if (tk) tk.innerHTML = '<span class="spin sm accent" style="width:11px;height:11px;"></span>';
          const d = r.querySelector('.tdur'); if (d && label) d.textContent = label;
        }
      });
      const bar = document.getElementById('gdRunProg-bar');
      if (bar && total) bar.style.width = Math.round(Math.min(1, Number(current || 0) / Number(total || 1)) * 100) + '%';
    };
    window.EU_API.startAgentRun({
      path: src.path,
      study_id: branch || 'guided',
      mode: 'analysis',
      run_type: 'preflight',
      question: BRANCH[branch].chip,
    }).then(r => {
      const es = new EventSource('/api/jobs/' + r.job_id + '/events');
      liveAgentRun.jobId = r.job_id;
      es.onmessage = msg => {
        const ev = JSON.parse(msg.data);
        if (ev.label) liveAgentRun.step = ev.label;
        if (ev.total) setProgress(ev.current, ev.total, ev.label);
        const pill = document.getElementById('gdRunPill');
        if (pill && ev.label) pill.innerHTML = `<span class="dot"></span>${esc(ev.label)}`;
        if (ev.type === 'end') {
          es.close();
          if (ev.status === 'done') completeLivePipeline(ev.result);
          else failLivePipeline(ev.error || 'Agent run failed.');
        }
      };
      es.onerror = () => { es.close(); failLivePipeline('Lost connection to the agent run.'); };
    }).catch(err => failLivePipeline(err.message || String(err)));
  }

  function completeLivePipeline(result) {
    liveAgentRun = { active: false, result: result, error: null };
    runPhase = 'done';
    outputsReady = true;
    if (result && result.summary) workspaceSnapshot = { summary: result.summary, cohort: result.cohort || {}, quality: result.quality || [] };
    setVal({ analysis: () => analysisLine(), draft: 'locked · analysis_only' });
    const p = document.getElementById('gdRunPill'); if (p) p.outerHTML = '<span class="pill ok" id="gdRunPill"><span class="dot"></span>Preflight complete</span>';
    renderThread();
    thread.push({ bot: true, html: `Run complete — registry-backed preflight artifacts were written locally and logged to the evidence ledger. <span style="color:var(--ink-4);">No patient rows were persisted and no external model call was made.</span>` });
    thread.push({ diff: true });
    renderThread(); renderAside();
    thread.push({ bot: true, html: `I can open this in Agent Projects now. Manuscript claims remain <strong>locked</strong> until human sign-off.` });
    chips = []; renderThread();
    go('toFindings');
  }

  function failLivePipeline(error) {
    liveAgentRun = { active: false, result: null, error: error };
    runPhase = 'run';
    const p = document.getElementById('gdRunPill'); if (p) p.outerHTML = '<span class="pill bad" id="gdRunPill"><span class="dot"></span>Failed closed</span>';
    thread.push({ bot: true, html: `The run failed closed: <span class="mono">${esc(error)}</span>` });
    chips = [['Retry analysis', 'toRun'], ['Open Agent Projects', '@draft']];
    renderThread(); renderChips();
  }
  function schedule(fn) { const myGen = gen; const t = () => { if (myGen !== gen) return; if (busy) return setTimeout(t, 160); fn(); }; setTimeout(t, 520); }

  /* ============== card renderers (one per step) ============== */
  function cardShell(step, ico, title, sub, bodyHtml, footHtml) {
    const w = BRANCH[branch].why[step];
    const on = whyOpen[step];
    return `
    <div class="gd-card" data-card-step="${step}">
      <div class="gc-head">
        <div class="gc-ico">${icon(ico, 15)}</div>
        <div class="grow"><div class="gc-t">${title}</div><div class="gc-sub">${sub}</div></div>
        ${w ? `<button class="gc-why ${on ? 'on' : ''}" data-why="${step}">${icon('help', 11)} Why${on ? '' : ' this step'}</button>` : ''}
      </div>
      ${w ? `<div class="gd-why" ${on ? '' : 'hidden'}>${w}</div>` : ''}
      <div class="gc-body">${bodyHtml}</div>
      ${footHtml ? `<div class="gd-cardfoot">${footHtml}</div>` : ''}
    </div>`;
  }

  const CARD = {
    question() {
      const b = BRANCH[branch];
      return cardShell('question', 'spark', 'Study plan', 'forming', `
        <p style="font-size:12.5px;color:var(--ink-2);font-style:italic;margin:0 0 12px;line-height:1.5;">${frameFor(branch)}</p>
        <div class="col gap-6" style="font-size:12.25px;">
          ${planFor(branch).map(([k, v]) => `<div class="setup-row"><span class="k">${k}</span><span class="vv">${v}</span></div>`).join('')}
        </div>
        <div class="m-cite" style="margin-top:11px;">${icon('shield', 11)} evidence-bound · I won’t assert effect sizes</div>`,
        `<button class="btn primary sm" data-go="toData">Looks right — continue ${icon('arrow', 13)}</button>
         <button class="btn sm" data-go="welcome">Reframe</button>`);
    },
    data() {
      return cardShell('data', 'folder', 'Connect your data', 'a local folder', `
        <div class="gd-opts">
          <button class="gd-opt" data-datasrc="prepared">
            <span class="o-mk">${icon('layers', 16)}</span>
            <span><span class="o-t">Prepared data path</span><span class="o-d">A converted EasyICU folder (Parquet) — analysis-ready</span></span>
            <span class="o-go">${icon('arrow', 15)}</span>
          </button>
          <button class="gd-opt" data-datasrc="module">
            <span class="o-mk">${icon('download', 16)}</span>
            <span><span class="o-t">Module export folder</span><span class="o-d">A prior EasyICU export (per-concept Parquet + manifest)</span></span>
            <span class="o-go">${icon('arrow', 15)}</span>
          </button>
          <button class="gd-opt" data-datasrc="raw">
            <span class="o-mk">${icon('db', 16)}</span>
            <span><span class="o-t">Raw ICU files</span><span class="o-d">Original CSV/CSV.GZ — needs conversion first</span></span>
            <span class="o-go">${icon('arrow', 15)}</span>
          </button>
        </div>`);
    },
    cohort() {
      const b = BRANCH[branch];
      if (b.cohortKind === 'databases') return CARD._databases();
      if (cohortPhase === 'empty') return CARD._cohortEmpty();
      const s = snapshotSummary();
      const matched = realMode() ? fmtInt(s.stays, 'registered') : patientN;
      const mort = realMode() ? fmtPct(s.mortality) : '20%';
      const ageLabel = realMode() ? 'Mean age' : 'Median age';
      const age = realMode() ? fmtNum(s.mean_age) : '55';
      const sofa = realMode() && s.median_sofa2 != null ? `<span class="chip solid">median SOFA-2 = ${esc(fmtNum(s.median_sofa2))}</span>` : '';
      return cardShell('cohort', 'cohort', 'Proposed cohort', realMode() ? `local · ${activeExportLabel()}` : `demo · ${patientN} stays`, `
        <div class="gd-stats" style="grid-template-columns:repeat(3,1fr);">
          <div class="gd-stat"><div class="l">Matched</div><div class="v">${matched}</div></div>
          <div class="gd-stat"><div class="l">Mortality</div><div class="v">${mort}</div></div>
          <div class="gd-stat"><div class="l">${ageLabel}</div><div class="v">${age}</div></div>
        </div>
        <div class="eyebrow" style="margin:13px 0 7px;">Inclusion criteria</div>
        <div class="row wrap gap-6">
          <span class="chip solid">first ICU stay</span>
          <span class="chip solid">age ≥ 18</span>
          <span class="chip solid">window = first 24h</span>
          ${sofa}
        </div>`,
        `<button class="btn primary sm" data-go="toConcepts">Use this cohort ${icon('arrow', 13)}</button>
         <button class="btn sm" data-act="strict">Restrict: Sepsis-3 + age ≥ 80</button>`);
    },
    _cohortEmpty() {
      return cardShell('cohort', 'cohort', 'Cohort matched 0 stays', 'too strict', `
        <div class="state-hero nodata">
          <div class="glyph">${icon('eye', 22)}</div>
          <div class="st-t">No patients match those filters</div>
          <div class="st-d">“Sepsis-3 + age ≥ 80” is empty in this ${dataMode === 'demo' ? 'demo set' : 'export'}. Loosen a constraint and I’ll re-match.</div>
          <div class="filter-recap"><span class="chip solid">sepsis-3</span><span class="chip solid">age ≥ 80</span><span class="chip solid">first 24h</span></div>
        </div>`,
        `<button class="btn primary sm" data-act="loosen">${icon('refresh', 13)} Loosen filters</button>
         <button class="btn sm" data-act="loosen">Back to defaults</button>`);
    },
    _databases() {
      const dbs = [['MIMIC-IV', true], ['eICU-CRD', true], ['AUMCdb', true], ['HiRID', false], ['MIMIC-III', false], ['SICdb', false]];
      return cardShell('cohort', 'benchmark', 'Select databases', 'compare ≥ 2 sources', `
        <div class="gd-mods" id="gdDbs">
          ${dbs.map(([n, on]) => `<button class="gd-mod ${on ? 'on' : ''}" data-db="${n}"><span class="mk">${on ? icon('check', 10, 3) : ''}</span>${n}</button>`).join('')}
        </div>
        <div class="note info mt-12" style="padding:9px 11px;"><div class="ico">${icon('shield', 13)}</div><div class="body"><div class="d" style="font-size:10.5px;margin:0;">One sepsis cohort definition is applied identically to each database.</div></div></div>`,
        `<button class="btn primary sm" data-go="toConcepts">Use <span id="gdDbN">3</span> databases ${icon('arrow', 13)}</button>`);
    },
    concepts() {
      const ALL = [['Demographics', 6], ['Vital signs', 8], ['Lab — Chemistry', 22], ['SOFA-2 scores', 7], ['Sepsis-3 (SOFA-2)', 1], ['Outcome', 3], ['Respiratory', 14], ['Renal & urine output', 20]];
      return cardShell('concepts', 'layers', 'Feature modules', 'pre-selected from cohort', `
        <div class="gd-mods" id="gdMods">
          ${ALL.map(([n, c]) => { const on = mods.includes(n); return `<button class="gd-mod ${on ? 'on' : ''}" data-mod="${n}"><span class="mk">${on ? icon('check', 10, 3) : ''}</span>${n}<span class="mc">${c}</span></button>`; }).join('')}
        </div>`,
        `<button class="btn primary sm" data-go="toExtract">Confirm <span id="gdModN">${mods.length}</span> modules ${icon('arrow', 13)}</button>`);
    },
    extract() {
      if (extractPhase === 'done') {
        const s = snapshotSummary();
        return cardShell('extract', 'extract', realMode() ? 'Snapshot ready' : 'Extraction complete', realMode() ? 'active export frozen' : 'demo workspace ready', `
          <div class="ok-banner" style="margin-bottom:12px;"><span class="mk">${icon('check', 12, 3)}</span><div class="grow"><strong style="font-weight:600;">Packaged locally.</strong> <span style="color:var(--ink-3);">${realMode() ? 'Queue-level bounded snapshot; no patient rows persisted.' : 'Seeded demo values — illustrative, not a real run.'}</span></div></div>
          <div class="gd-stats">
            <div class="gd-stat"><div class="l">Stays</div><div class="v">${realMode() ? fmtInt(s.stays) : patientN}</div></div>
            <div class="gd-stat"><div class="l">${realMode() ? 'Rows' : 'Time pts'}</div><div class="v">${realMode() ? fmtInt(s.total_rows) : patientN * 24}</div></div>
            <div class="gd-stat"><div class="l">Modules</div><div class="v">${realMode() ? fmtInt(s.modules, String(mods.length)) : mods.length}</div></div>
            <div class="gd-stat"><div class="l">${realMode() ? 'Sepsis' : 'Coverage'}</div><div class="v">${realMode() ? fmtPct(s.sepsis_pct) : '94%'}</div></div>
          </div>`, '');
      }
      const tasks = ['Normalize source', 'Resolve cohort', 'Map concepts', 'Coverage audit', 'Package frames'];
      return cardShell('extract', 'extract', 'Extracting', dataMode === 'demo' ? 'demo · local-only' : 'local · no uploads', `
        <div class="gd-prog" id="gdExProg">
          ${tasks.map(t => `<div class="gd-task queued"><span class="tk">${icon('clock', 9)}</span><span class="grow">${t}</span><span class="tdur"></span></div>`).join('')}
        </div>
        <div class="indet mt-12"></div>`, '');
    },
    review() {
      const b = BRANCH[branch];
      let body;
      if (realMode()) {
        const s = snapshotSummary();
        const c = snapshotCohort();
        body = `<div class="gd-stats" style="grid-template-columns:repeat(3,1fr);">
          <div class="gd-stat"><div class="l">Survived</div><div class="v">${fmtInt(c.survived)}</div></div>
          <div class="gd-stat"><div class="l">Deceased</div><div class="v">${fmtInt(c.deceased)}</div></div>
          <div class="gd-stat"><div class="l">Mortality</div><div class="v">${fmtPct(s.mortality)}</div></div>
          <div class="gd-stat"><div class="l">Mean age</div><div class="v">${fmtNum(s.mean_age)}</div></div>
          <div class="gd-stat"><div class="l">Median SOFA-2</div><div class="v">${fmtNum(s.median_sofa2)}</div></div>
          <div class="gd-stat"><div class="l">Rows</div><div class="v">${fmtInt(s.total_rows)}</div></div>
        </div>`;
      } else if (branch === 'crossdb') {
        body = `<div class="table-wrap" style="border:0;"><table class="eu-table" style="font-size:11px;"><thead><tr><th>Concept</th><th class="num">MIMIC-IV</th><th class="num">eICU</th><th class="num">AUMC</th></tr></thead><tbody>
          <tr><td class="key">lactate</td><td class="num">✓</td><td class="num">✓</td><td class="num">✓</td></tr>
          <tr><td class="key">map</td><td class="num">✓</td><td class="num">✓</td><td class="num">✓</td></tr>
          <tr><td class="key">sofa</td><td class="num">✓</td><td class="num">partial</td><td class="num">✓</td></tr></tbody></table></div>`;
      } else if (branch === 'quality') {
        body = `<div class="col gap-8">
          ${[['Vitals', 97, 'ok'], ['Chemistry', 91, 'ok'], ['SOFA-2', 88, 'warn'], ['Ventilator', 62, 'bad']].map(([n, p, c]) => `<div><div class="row" style="justify-content:space-between;font-size:11.5px;"><span>${n}</span><span class="mono" style="color:var(--ink-4);">${p}%</span></div><div class="qbar ${c} mt-4" style="height:7px;"><span style="width:${p}%"></span></div></div>`).join('')}
        </div>`;
      } else {
        body = `<div class="table-wrap" style="border:0;"><table class="eu-table" style="font-size:11.5px;"><thead><tr><th>Characteristic</th><th class="num">Survived</th><th class="num">Deceased</th><th class="num">p</th></tr></thead><tbody>
          <tr><td class="key">Age, mean</td><td class="num">52.1</td><td class="num">64.8</td><td class="num">0.04</td></tr>
          <tr><td class="key">SOFA, median</td><td class="num">5</td><td class="num">9</td><td class="num">0.01</td></tr>
          <tr><td class="key">Lactate</td><td class="num">2.1</td><td class="num">3.6</td><td class="num">0.03</td></tr></tbody></table></div>`;
      }
      return cardShell('review', 'eye', realMode() ? 'Quick review · export snapshot' : b.reviewTitle, 'before analysis', body +
        `<div class="note ${realMode() ? 'info' : 'demo'} mt-12" style="padding:9px 11px;"><div class="ico">${icon(realMode() ? 'shield' : 'flask', 13)}</div><div class="body"><div class="d" style="font-size:10.5px;margin:0;">${realMode() ? 'Registry-backed aggregate snapshot; preview payload stays bounded and local.' : 'Seeded demo numbers for layout — not a finding.'}</div></div></div>`,
        (depth === 'full'
          ? `<button class="btn primary sm" data-go="toRun">Run the analysis ${icon('arrow', 13)}</button>
             <button class="btn sm" data-act="open">${icon('grid', 13)} See it in the workspace</button>`
          : `<button class="btn primary sm" data-go="@finish">${icon('check', 13)} Finish here</button>
             <button class="btn sm" data-go="@extendNext">Take it further → analyse ${icon('arrow', 13)}</button>`));
    },
    analysis() {
      const tasks = analysisTasks();
      if (runPhase === 'done') {
        const live = liveAgentRun && liveAgentRun.result;
        return cardShell('analysis', 'agent', 'Research Agent · run', live ? `preflight complete · ${(live.artifacts || []).length} artifacts` : 'complete · 6 artifacts', `
          <div class="gd-prog">${tasks.map(([t, d]) => `<div class="gd-task done"><span class="tk">${icon('check', 10, 3)}</span><span class="grow">${t}</span><span class="tdur">${d}</span></div>`).join('')}</div>
          <div class="run-strip mt-12" style="padding:8px 10px;"><span class="pill ok"><span class="dot"></span>${live ? 'Preflight complete' : 'Complete'}</span><div class="grow runbar"><div class="runbar-fill" style="width:100%"></div></div></div>`, '');
      }
      return cardShell('analysis', 'agent', 'Research Agent · run', dataMode !== 'demo' ? 'registry-backed · local preflight' : 'demo pipeline · no tokens', `
        <div class="gd-prog" id="gdRunProg">${tasks.map(([t]) => `<div class="gd-task queued"><span class="tk">${icon('clock', 9)}</span><span class="tt-cmd">py</span><span class="grow">${t}</span><span class="tdur"></span></div>`).join('')}</div>
        <div class="run-strip mt-12" style="padding:8px 10px;"><span class="pill warn" id="gdRunPill"><span class="dot"></span>Running</span><div class="grow runbar"><div class="runbar-fill" id="gdRunProg-bar" style="width:0%;transition:width .12s linear;"></div></div></div>`, '');
    },
    draft() {
      const b = BRANCH[branch];
      const live = liveAgentRun && liveAgentRun.result;
      if (live) {
        const gate = live.gate || {};
        const checks = Array.isArray(gate.checks) ? gate.checks : [];
        return cardShell('draft', 'shield', 'Preflight complete · draft locked', gate.status || 'analysis_only', `
          <div class="m-bubble" style="background:var(--surface-2);border:1px solid var(--hair);font-size:12.25px;margin-bottom:12px;">Local preflight finished for <span class="mono">${esc(live.run_id || 'run')}</span>. <span style="color:var(--ink-4);">No external model call, no uploads, and no patient rows persisted. Manuscript claims remain locked.</span></div>
          <div class="eyebrow" style="margin:0 0 8px;">Evidence gate</div>
          <div class="checks">
            ${checks.map(c => `<div class="check-row ${c.passed ? 'ok' : 'pending'}"><span class="check-mk">${c.passed ? icon('check', 11, 2.8) : icon('clock', 11)}</span><span style="font-size:11.75px;color:${c.passed ? 'var(--ink)' : 'var(--ink-3)'};">${esc(c.label || c.id)}</span><span class="grow"></span><span class="mono" style="font-size:10px;color:${c.passed ? 'var(--ok)' : 'var(--ink-4)'};">${c.passed ? 'passed' : 'pending'}</span></div>`).join('')}
          </div>`,
          `<button class="btn primary sm" data-act="draft">Open in Research Agent</button>
           <button class="btn sm" data-act="open">${icon('grid', 13)} Open workspace</button>`);
      }
      if (draftPhase === 'signed') {
        return cardShell('draft', 'check', 'Study assembled', 'gated draft unlocked', `
          <div class="col gap-6" style="font-size:12px;">
            <div class="setup-row"><span class="k">Data</span><span class="vv">${patientN} stays · ${mods.length} modules</span></div>
            <div class="setup-row"><span class="k">Analysis</span><span class="vv">5 steps · 6 artifacts</span></div>
            <div class="setup-row"><span class="k">Draft</span><span class="vv">unlocked after sign-off</span></div>
            <div class="setup-row"><span class="k">Bundle</span><span class="vv">~/easyicu/exports</span></div>
          </div>`,
          `<button class="btn primary sm" data-act="open">${icon('arrow', 13)} Open full workspace</button>
           <button class="btn sm" data-act="draft">Open draft</button>
           <button class="btn sm ghost" data-go="welcome">New study</button>`);
      }
      return cardShell('draft', 'shield', 'Findings drafted · review required', 'evidence-bound', `
        <div class="m-bubble" style="background:var(--surface-2);border:1px solid var(--hair);font-size:12.25px;margin-bottom:12px;">${b.findings}</div>
        <div class="eyebrow" style="margin:0 0 8px;">Evidence gate</div>
        <div class="checks">
          ${[['Denominators resolved', true], ['Coverage ≥ threshold', true], ['Reproduces from manifest', true], ['Model card attached', true], ['Reviewer sign-off', false]].map(([t, ok]) => `<div class="check-row ${ok ? 'ok' : 'pending'}"><span class="check-mk">${ok ? icon('check', 11, 2.8) : icon('clock', 11)}</span><span style="font-size:11.75px;color:${ok ? 'var(--ink)' : 'var(--ink-3)'};">${t}</span><span class="grow"></span><span class="mono" style="font-size:10px;color:${ok ? 'var(--ok)' : 'var(--ink-4)'};">${ok ? 'passed' : 'pending'}</span></div>`).join('')}
        </div>`,
        `<button class="btn primary sm" data-act="signoff">${icon('check', 13)} Review &amp; sign off</button>
         <button class="btn sm" data-act="draft">Open in Research Agent</button>`);
    },
  };

  /* collapsed one-line summary per step */
  function summaryOf(step) {
    const b = BRANCH[branch];
    switch (step) {
      case 'question': return { t: 'Question', v: b.chip, edit: true };
      case 'data': return { t: 'Data', v: dataMode === 'demo' ? 'Demo · local' : 'Local export', edit: true };
      case 'cohort': return { t: b.cohortKind === 'databases' ? 'Databases' : 'Cohort', v: b.cohortKind === 'databases' ? `${dbCount()} databases` : cohortLine(), edit: true };
      case 'concepts': return { t: 'Modules', v: `${mods.length} feature modules`, edit: true };
      case 'extract': return { t: 'Extraction', v: extractLine(), edit: false };
      case 'review': return { t: 'Review', v: realMode() ? 'export snapshot' : b.reviewTitle.replace('Quick review · ', ''), edit: false };
      case 'analysis': return { t: 'Analysis', v: analysisLine(), edit: false };
      case 'draft': return { t: 'Draft', v: (liveAgentRun && liveAgentRun.result) ? 'locked · analysis_only' : (draftPhase === 'signed' ? 'unlocked' : 'gated'), edit: false };
    }
    return { t: step, v: '', edit: false };
  }
  let _dbCount = 3;
  function dbCount() { return _dbCount; }

  /* ---- depth helpers: where does this study stop? ---- */
  function goalStep() { return (DEPTH[depth] || DEPTH.full).goal; }
  function goalIdx() { return STEP_INDEX[goalStep()]; }
  function isBeyondGoal(step) { return STEP_INDEX[step] > goalIdx(); }
  function bumpDepth() { const i = DEPTH_ORDER.indexOf(depth); depth = DEPTH_ORDER[Math.min(DEPTH_ORDER.length - 1, i + 1)]; return depth; }
  function finishHere() {
    markThrough(goalStep(), 'done');
    const msg = depth === 'extract'
      ? `That’s your finish line for an <strong>extract-only</strong> run — cohort frames and a reproducible <code>manifest.json</code> are written locally. Nothing left this machine.`
      : `That’s your finish line for <strong>${DEPTH[depth].label}</strong> — the populated workspace is ready to explore. Nothing left this machine.`;
    thread.push({ bot: true, html: msg });
    chips = [['Open in workspace', '@open', 'express'], (depth !== 'full' ? ['Actually, take it further', '@extendNext'] : null)].filter(Boolean);
    renderThread(); renderChips();
  }

  /* ============== DOM render ============== */
  function renderThread() {
    const host = document.getElementById('gdThread');
    if (!host) return;
    host.innerHTML = thread.map(t => {
      if (t.typing) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble"><div class="typing"><span></span><span></span><span></span></div></div></div></div>`;
      if (t.diff) return diffCard();
      if (t.once) return ONCE[t.once] ? ONCE[t.once]() : '';
      if (t.card) {
        if (t.step === expandedStep) return CARD[t.step] ? CARD[t.step]() : '';
        const s = summaryOf(t.step);
        return `<div class="gd-collapsed"><span class="cc-mk">${icon('check', 10, 3)}</span><span class="cc-t">${s.t}</span><span class="cc-v">${s.v}</span>${s.edit ? `<button class="cc-edit" data-edit="${t.step}">${icon('sliders', 11)} Edit</button>` : ''}</div>`;
      }
      if (t.bot) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble">${t.html}</div></div></div>`;
      return `<div class="msg user"><div class="m-ava">LK</div><div class="m-body"><div class="m-bubble">${t.html}</div></div></div>`;
    }).join('');
    scrollEnd();
  }
  function renderChips() {
    const host = document.getElementById('gdSuggest');
    if (!host) return;
    host.innerHTML = chips.map(([label, next, cls]) => `<button class="suggest-chip ${cls || ''}" data-go="${next}">${label}</button>`).join('');
    host.style.display = chips.length ? 'flex' : 'none';
  }
  function renderAside() {
    const host = document.getElementById('gdAsideBody');
    if (!host) return;
    const gi = goalIdx();
    host.innerHTML = STUDY.map(([id, label, ico], idx) => {
      let stt = studyStatus[id] || 'pending';
      // steps past the chosen finish line are optional — dim them unless already reached
      if (idx > gi && (stt === 'pending')) stt = 'beyond';
      let v = studyVal[id]; if (typeof v === 'function') v = v();
      const dot = stt === 'done' ? icon('check', 11, 3) : stt === 'locked' ? icon('lock', 10) : icon(ico, 12);
      const badge = stt === 'active' ? '<span class="si-state"><span class="spin sm" style="width:11px;height:11px;"></span></span>'
        : stt === 'locked' ? `<span class="si-state pill warn" style="height:18px;"><span class="dot"></span></span>`
        : stt === 'beyond' ? `<span class="si-state si-opt">optional</span>` : '';
      const clickable = thread.some(t => t.card && t.step === id);
      const row = `<div class="study-item ${stt}${clickable ? ' nav' : ''}" ${clickable ? `data-study="${id}" role="button" tabindex="0"` : ''}><span class="si-dot">${dot}</span><div class="si-txt"><div class="si-t">${label}</div>${v ? `<div class="si-v">${v}</div>` : ''}</div>${badge}</div>`;
      // draw the finish line right after the goal step (only when stopping short of the full study)
      const fin = (idx === gi && depth !== 'full')
        ? `<div class="study-finishline"><span class="fl-flag">${icon('check', 10, 3)}</span><span class="fl-t">Finish line · ${DEPTH[depth].label}</span></div>`
        : '';
      return row + fin;
    }).join('') + renderOutputs(host);
  }

  /* ============== composer intent parsing ============== */
  function parseIntent(text) {
    const t = text.toLowerCase();
    // depth selection at the goal gate
    if (currentId === 'goal') {
      depth = /\b(only|just)?\s*(extract|filter|pull|cohort|data)\b/.test(t) && !/review|visual|analy|agent|full|draft/.test(t) ? 'extract'
        : /review|visual|visualis|visualiz|inspect|explore|look at|chart/.test(t) && !/agent|full|draft|model|predict/.test(t) ? 'review'
        : 'full';
      return () => { renderAside(); go('welcome', text); };
    }
    // branch selection at welcome
    if (currentId === 'welcome') {
      if (/\b(run|whole|everything|just do|autopilot|for me)\b/.test(t)) return () => autopilot(text);
      branch = pickBranch(text);
      extractEntities(text);
      // if they already pinned the endpoint (or it's not the predict branch), skip the clarify
      if (branch !== 'predict' || endpointPinned(text)) return () => go('frame', text);
      return () => go('clarify', text);
    }
    // global intents
    if (/\b(run|skip|whole|everything|autopilot|just do it|for me)\b/.test(t)) return () => autopilot(text);
    if (/\b(back|undo|change|edit|previous|redo)\b/.test(t)) { const ed = lastEditable(); if (ed) return () => { pushUser(text); editStep(ed); }; }
    if (/\b(why|explain|reason)\b/.test(t)) return () => { pushUser(text); toggleWhy(expandedStep, true); };
    // patient count while at/around cohort
    const m = t.match(/\b(\d{1,3})\b/);
    if (m && /(patient|case|stay|stays|cohort|n=|sample)/.test(t)) {
      const n = Math.max(5, Math.min(50, parseInt(m[1], 10)));
      patientN = n;
      return () => { pushUser(text); if (currentId !== 'toCohort' && STEP_INDEX[expandedStep] > STEP_INDEX.cohort) editStep('cohort'); else { renderThread(); renderAside(); thread.push({ bot: true, html: `Set to <strong>${n}</strong> demo stays. ${currentId === 'toCohort' ? 'Use this cohort when ready.' : ''}` }); renderThread(); } };
    }
    return null;
  }
  function lastEditable() {
    const editable = ['concepts', 'cohort', 'data', 'question'];
    for (const s of editable) if (thread.some(t => t.card && t.step === s) && STEP_INDEX[s] < STEP_INDEX[expandedStep]) return s;
    return null;
  }
  function toggleWhy(step, force) {
    whyOpen[step] = force != null ? force : !whyOpen[step];
    renderThread();
  }

  /* ============== express lane (autopilot to the gate) ============== */
  function autopilot(userText) {
    if (autop) return;
    autop = true; branch = branch || 'predict'; dataMode = 'demo';
    pushUser(userText || 'Run the whole demo for me');
    const acts = [
      () => go('frame'),
      () => go('toData'),
      () => go('toCohort'),
      () => go('toConcepts'),
      () => go('toExtract'),  // chains review → run → findings, then stops at the gate
    ];
    let i = 0;
    const myGen = gen;
    (function tick() {
      if (myGen !== gen) return;
      if (i >= acts.length) return;
      if (busy) { setTimeout(tick, 160); return; }
      acts[i++](); setTimeout(tick, 600);
    })();
  }

  /* ============== continuity into the classic workspace ============== */
  function openWorkspace(target) {
    try { if (window.__euExtractApply) window.__euExtractApply(mods); } catch (e) {}
    try { if (window.__euVizPreset) window.__euVizPreset(); } catch (e) {}
    location.hash = '#' + (target || BRANCH[branch].openTarget || 'patient');
  }
  function openDraft() {
    try { if (window.__euAgentPreset) window.__euAgentPreset(); } catch (e) {}
    location.hash = '#agent';
  }

  /* ============== conversation script ============== */
  const STATES = {
    goal: {
      delay: 340,
      bot: [
        `Hi — I’m the EasyICU <strong>Research Copilot</strong>. I’ll drive the workspace by chat, and you can stop at any point.`,
        `First, <strong>how far do you want to go today?</strong> This just sets where I stop — you can always extend later.`,
      ],
      chips: () => [
        [DEPTH.extract.chip, '@depth:extract'],
        [DEPTH.review.chip, '@depth:review'],
        [DEPTH.full.chip, '@depth:full'],
      ],
    },
    welcome: {
      delay: 340,
      step: 'question',
      bot: () => [
        DEPTH[depth].hi,
        `Now — what would you like to study? Pick a direction below, or describe your own.`,
      ],
      chips: () => {
        const base = [
          [BRANCH.predict.chip, '@branch:predict'],
          [BRANCH.crossdb.chip, '@branch:crossdb'],
          [BRANCH.quality.chip, '@branch:quality'],
          ['Type my own question', '@typemine'],
        ];
        base.push([`Change goal (now: ${DEPTH[depth].label})`, '@regoal']);
        return base;
      },
      markStep: 'question',
    },
    clarify: {
      delay: 420,
      step: 'question',
      bot: () => [CLARIFY[branch].q],
      chips: () => CLARIFY[branch].opts.map(([label, detail]) => [label, '@clarify:' + detail]),
      markStep: 'question',
    },
    frame: {
      step: 'question', card: true,
      bot: () => [studyParams.caught ? `From your description I picked up <strong>${studyParams.caught}</strong>. Here’s a tighter, researchable framing — tweak anything:` : clarified ? `Got it — <strong>${clarified}</strong>. Here’s a tighter, researchable framing:` : `Good — here’s a tighter, researchable framing:`],
      chips: () => [['Why frame it this way?', '@why'], ['Use my own wording', '@noop']],
      markStep: 'question', markStatus: 'active',
      val: { question: () => BRANCH[branch].chip },
    },
    toData: {
      step: 'data', card: true,
      bot: [`How should data enter the workspace?`],
      chips: () => [['What’s the difference?', '@why']],
      markStep: 'data',
      val: { question: () => BRANCH[branch].chip },
    },
    realConfirm: {
      step: 'data',
      bot: [`Before we read local data, two things: this reads files on your machine and this first Agent run is a <strong>local preflight only</strong>: no external model call, no uploads, and never patient rows. Continue?`],
      chips: () => [['Continue with local data', 'connect'], ['Use demo instead', '@usedemo']],
      markStep: 'data',
      val: { question: () => BRANCH[branch].chip },
    },
    connect: {
      step: 'data',
      bot: [`Point me at a local ICU export root — I’ll detect the layout. Nothing leaves your machine.`],
      once: 'folder',
      chips: [],
      markStep: 'data',
    },
    detect: {
      delay: 360,
      step: 'data',
      bot: [`Scanning the folder…`],
      once: 'detect',
      chips: [],
      markStep: 'data',
      onShown() { runDetect(); },
    },
    detected: {
      delay: 300,
      step: 'data',
      bot: [],
      once: 'detected',
      chips: [],
      markStep: 'data',
    },
    toCohort: {
      step: 'cohort', card: true,
      bot: () => [BRANCH[branch].cohortKind === 'databases' ? `Pick the databases to compare — the same cohort definition applies to each.` : (realMode() ? `Here’s the active export cohort summary. Full-cohort aggregates are used; row previews stay bounded.` : `Here’s a starting cohort. The demo set is small on purpose so every screen stays explorable.`)],
      chips: () => [['Adjust patient count', '@hintN'], ['Why this matters', '@why']],
      markStep: 'cohort',
      val: { data: () => dataMode === 'demo' ? 'Demo · local' : 'Local export' },
    },
    toConcepts: {
      step: 'concepts', card: true,
      bot: [`I’ve pre-selected the feature modules your question needs. Toggle any — coverage gets audited before modelling.`],
      chips: () => [['Why these modules?', '@why']],
      markStep: 'concepts',
      val: { cohort: () => BRANCH[branch].cohortKind === 'databases' ? `${dbCount()} databases` : cohortLine() },
    },
    toExtract: {
      delay: 420,
      step: 'extract', card: true,
      bot: [`Extracting now — normalizing, resolving the cohort, and packaging frames locally.`],
      chips: [],
      markStep: 'extract',
      val: { concepts: () => `${mods.length} modules` },
      onShown() { extractPhase = 'run'; runExtract(); },
    },
    toReview: {
      delay: 300,
      step: 'review', card: true,
      bot: () => [`Here’s a quick look — the full ${BRANCH[branch].openTarget === 'crossdb' ? 'benchmark' : 'review'} is one click away in the workspace.`],
      chips: () => depth === 'full'
        ? [['Looks fine — analyze', 'toRun'], ['Open the workspace', '@open']]
        : [['Finish here', '@finish', 'express'], ['Open the workspace', '@open'], ['Take it further → analyse', '@extendNext']],
      markStep: 'review',
      val: { extract: () => extractLine() },
      onShown() { if (autop) { if (depth === 'full') schedule(() => go('toRun')); else schedule(() => finishHere()); } },
    },
    toRun: {
      delay: 420,
      step: 'analysis', card: true,
      bot: () => [realMode()
        ? `Running a registry-backed local preflight — source resolution, bounded snapshot, evidence gate, and local artifact write. No external model call.`
        : `Running the analysis — deterministic steps, no tokens. I’ll only draft findings after every step’s evidence contract passes.`],
      chips: [],
      markStep: 'analysis',
      val: { extract: () => extractLine(), review: () => reviewLine() },
      onShown() { runPhase = 'run'; runPipeline(); },
    },
    toFindings: {
      delay: 300,
      step: 'draft', card: true,
      bot: [],
      chips: [],
      markStep: 'draft', markStatus: 'locked',
      val: { analysis: () => analysisLine() },
      onShown() { autop = false; },  // express lane deliberately stops at the human gate
    },
  };

  function renderSessions() {
    const host = document.getElementById('gdSessions');
    if (!host) return;
    const items = [
      ['live', 'Sepsis mortality prediction', '~/easyicu/projects/sepsis_mortality', true],
      ['s1', 'Lactate trajectory · 48h', '~/easyicu/projects/lactate_48h', false],
      ['s2', 'AKI onset · MIMIC-IV / eICU', '~/easyicu/projects/aki_onset', false],
      ['s3', 'Vasopressor exposure audit', '~/easyicu/projects/vaso_audit', false],
    ];
    host.innerHTML = items.map(([id, tt, folder, live]) =>
      `<button class="gd-sess ${live ? 'live active' : ''}" data-sess="${id}"><span class="ss-fold">${icon('folder', 15)}</span><span><span class="ss-t">${tt}</span><span class="ss-m mono">${folder}</span></span></button>`
    ).join('');
  }

  /* ============== screen ============== */
  S.guided = {
    section: 'guided', full: true,
    render() {
      reset();
      return `
      <div class="gd-shell">
        <div class="gd-top">
          <div class="brand-mark">${icon('spark', 16)}</div>
          <div><div class="gd-name">Research Copilot</div><div class="gd-mode">EasyICU · research copilot</div></div>
          <span class="grow"></span>
          <button class="btn sm" data-open="entry">${icon('back', 13)} Exit</button>
          <button class="btn sm" data-open="extraction">${icon('grid', 13)} Classic workspace</button>
        </div>
        <div class="gd-main threecol">
          <aside class="gd-rail">
            <div class="gd-rail-top"><button class="gd-newbtn" data-newstudy title="Creates a new local project folder">${icon('plus', 14)} New study</button></div>
            <div class="gd-rail-sec">Studies · local folders</div>
            <div class="gd-rail-list" id="gdSessions"></div>
            <div class="gd-rail-foot"><button class="btn sm block" data-open="extraction">${icon('grid', 13)} Classic workspace</button></div>
          </aside>
          <div class="gd-conv">
            <div class="gd-scroll" id="gdScroll"><div class="gd-thread" id="gdThread" role="log" aria-live="polite" aria-label="Copilot conversation"></div></div>
            <div class="gd-suggest" id="gdSuggest"></div>
            <div class="gd-composer-wrap">
              <div class="gd-composer">
                <input class="gd-input" id="gdInput" placeholder="Reply, or tap an option above to continue…" autocomplete="off" />
                <button class="gd-send" id="gdSend">${icon('arrow', 16)}</button>
              </div>
              <div class="gd-foot-note">Research Copilot · reproducible · nothing leaves your machine</div>
            </div>
          </div>
          <aside class="gd-aside">
            <div class="gd-aside-head"><div class="eyebrow">Building your study</div><div class="at">Study workspace</div><div class="asub">Assembles as we talk · edit any step</div></div>
            <div class="gd-aside-body" id="gdAsideBody"></div>
            <div class="gd-aside-foot"><div class="note ok" style="padding:9px 11px;"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t" style="font-size:11.5px;">Evidence-bound</div><div class="d" style="font-size:10.5px;">Draft stays gated until checks pass.</div></div></div></div>
          </aside>
        </div>
      </div>`;
    },
    afterRender(root) {
      renderAside();
      renderSessions();
      // continue from the dock if we just expanded it
      let bridged = false;
      try {
        const b = window.__cpBridge;
        if (b && Date.now() - b.ts < 60000) {
          bridged = true; window.__cpBridge = null;
          if (b.dataMode) dataMode = b.dataMode;
          go('welcome');
          setTimeout(() => {
            if (b.branchHint && BRANCH[b.branchHint]) {
              branch = b.branchHint;
              extractEntities(b.lastUser || '');
              if (branch === 'predict' && !endpointPinned(b.lastUser || '')) { go('clarify', b.lastUser || BRANCH[branch].chip); }
              else { go('frame', b.lastUser || BRANCH[branch].chip); }
            } else if (b.lastUser) {
              handleText(stripTags(b.lastUser));
            } else {
              thread.push({ bot: true, html: `Continuing from the dock${b.route && b.route !== 'entry' ? ` — you were on <strong>${({extraction:'Data Extraction',patient:'Patient Review',cohort:'Cohort Statistics',crossdb:'Cross-DB Benchmark',agent:'Research Agent'}[b.route]) || 'the workspace'}</strong>` : ''}. Want to turn that into a full study?` });
              renderThread();
            }
          }, 700);
        }
      } catch (e) {}
      if (!bridged) go('goal');
      setTimeout(() => { const inp = root.querySelector('#gdInput'); if (inp) inp.focus(); }, 400);

      const shell = root.querySelector('.gd-shell');
      shell.addEventListener('click', (e) => {
        // open / close an artifact preview
        const artOpen = e.target.closest('[data-artopen]');
        if (artOpen) { openArtifact(artOpen.dataset.artopen); return; }
        if (e.target.closest('[data-artclose]') || e.target.id === 'gdArt') { closeArtifact(); return; }
        // why toggle
        const whyBtn = e.target.closest('[data-why]');
        if (whyBtn) { toggleWhy(whyBtn.dataset.why); return; }
        // expand the file-diff card
        const moreBtn = e.target.closest('[data-diffmore]');
        if (moreBtn) { diffExpanded = true; renderThread(); return; }
        // edit collapsed step
        const editBtn = e.target.closest('[data-edit]');
        if (editBtn) { editStep(editBtn.dataset.edit); return; }
        // hint chips → fill + submit
        const hint = e.target.closest('[data-hint]');
        if (hint) { handleText(hint.dataset.hint); return; }
        // data-go (chips + card buttons), with special @tokens
        const goEl = e.target.closest('[data-go]');
        if (goEl) {
          const tok = goEl.dataset.go;
          const label = stripText(goEl.textContent);
          if (tok.startsWith('@depth:')) { depth = tok.split(':')[1]; renderAside(); go('welcome', label); return; }
          if (tok === '@regoal') { pushUser(label); go('goal'); return; }
          if (tok === '@finish') { pushUser(label); finishHere(); return; }
          if (tok === '@extendNext') {
            const prev = depth; bumpDepth(); renderAside(); pushUser(label);
            thread.push({ bot: true, html: `Extending from <strong>${DEPTH[prev].label}</strong> to <strong>${DEPTH[depth].label}</strong> — picking up where we left off.` });
            renderThread();
            if (prev === 'extract') { schedule(() => go('toReview')); }
            else if (prev === 'review') { schedule(() => go('toRun')); }
            return;
          }
          if (tok.startsWith('@branch:')) { branch = tok.split(':')[1]; go('clarify', label); return; }
          if (tok.startsWith('@clarify:')) { clarified = applyClarify(branch, tok.slice(9)); go('frame', goEl.classList.contains('suggest-chip') ? label : null); return; }
          if (tok === '@usedemo') { dataMode = 'demo'; go('toCohort', 'Use demo instead'); return; }
          if (tok === '@autopilot') { autopilot(label); return; }
          if (tok === '@why') { toggleWhy(expandedStep, true); return; }
          if (tok === '@open') { openWorkspace(); return; }
          if (tok === '@noop') { pushUser(label); thread.push({ bot: true, html: `Go ahead — type your own wording in the box and I’ll work from it.` }); renderThread(); return; }
          if (tok === '@typemine') { pushUser(label); thread.push({ bot: true, html: `Of course — type your research question in the box below and I’ll frame it with you.` }); renderThread(); const inp = document.getElementById('gdInput'); if (inp) inp.focus(); return; }
          if (tok === '@folderpick') { pushUser(label); thread.push({ bot: true, html: `Opened <span class="mono">~/easyicu/projects/</span> — choose an existing study folder to resume, or start fresh below.` }); renderThread(); if (window.__euRender) setTimeout(window.__euRender, 700); return; }
          if (tok === '@foldernew') { pushUser(label); thread.push({ bot: true, html: `Created a new project folder. Let’s set up the study — intermediate files will be written there.` }); renderThread(); if (window.__euRender) setTimeout(window.__euRender, 700); return; }
          if (tok === '@hintN') { handleText('use 30 patients'); return; }
          go(tok, goEl.classList.contains('suggest-chip') ? label : null);
          return;
        }
        // mode picker
        const modeEl = e.target.closest('[data-mode]');
        if (modeEl) { dataMode = modeEl.dataset.mode; if (dataMode === 'real') { go('realConfirm', 'Use my local data'); } else { go('toCohort', 'Use demo data'); } return; }
        // data-source picker (folder-based — no demo mode in Copilot)
        const dsEl = e.target.closest('[data-datasrc]');
        if (dsEl) { dataMode = 'real'; const lab = dsEl.querySelector('.o-t'); go('realConfirm', lab ? lab.textContent : 'Connect a folder'); return; }
        // module toggles
        const modBtn = e.target.closest('[data-mod]');
        if (modBtn) {
          const n = modBtn.dataset.mod;
          if (mods.includes(n)) mods = mods.filter(m => m !== n); else mods.push(n);
          const on = mods.includes(n);
          modBtn.classList.toggle('on', on);
          modBtn.querySelector('.mk').innerHTML = on ? icon('check', 10, 3) : '';
          const c = document.getElementById('gdModN'); if (c) c.textContent = mods.length;
          return;
        }
        // database toggles
        const dbBtn = e.target.closest('[data-db]');
        if (dbBtn) {
          dbBtn.classList.toggle('on');
          const on = dbBtn.classList.contains('on');
          dbBtn.querySelector('.mk').innerHTML = on ? icon('check', 10, 3) : '';
          _dbCount = root.querySelectorAll('#gdDbs .gd-mod.on').length;
          const c = document.getElementById('gdDbN'); if (c) c.textContent = _dbCount;
          return;
        }
        // card actions
        const actEl = e.target.closest('[data-act]');
        if (actEl) {
          const a = actEl.dataset.act;
          if (a === 'strict') { cohortPhase = 'empty'; renderThread(); thread.push({ bot: true, html: `Trying “Sepsis-3 + age ≥ 80”…` }); renderThread(); return; }
          if (a === 'loosen') { cohortPhase = 'normal'; renderThread(); thread.push({ bot: true, html: `Loosened back to the working cohort — ${patientN} stays match again.` }); renderThread(); return; }
          if (a === 'open') { openWorkspace(); return; }
          if (a === 'draft') { openDraft(); return; }
          if (a === 'signoff') {
            draftPhase = 'signed'; markThrough('draft', 'done'); setVal({ draft: 'unlocked' }); renderThread();
            try { localStorage.setItem('easyicu_study', JSON.stringify({ branch, mods, patientN, ts: Date.now() })); } catch (e) {}
            thread.push({ bot: true, html: `Signed off — the draft is unlocked and the full study is assembled. Open the workspace or start another.` });
            chips = []; renderThread(); renderChips();
            return;
          }
          return;
        }
        // exit / classic
        const openEl = e.target.closest('[data-open]');
        if (openEl) { location.hash = '#' + openEl.dataset.open; return; }
        // clickable Study panel step → jump to / edit that step
        const stEl = e.target.closest('[data-study]');
        if (stEl) { jumpToStep(stEl.dataset.study); return; }
        // sessions rail
        if (e.target.closest('[data-newstudy]')) {
          pushUser('New study');
          thread.push({ bot: true, html: `Starting a new study. Each study lives in its own local folder — what would you like to do?` });
          chips = [['Open an existing folder', '@folderpick'], ['Create a new folder', '@foldernew']];
          renderThread(); renderChips();
          return;
        }
        const sessEl = e.target.closest('[data-sess]');
        if (sessEl) {
          root.querySelectorAll('.gd-sess').forEach(s => s.classList.toggle('active', s === sessEl));
          if (!sessEl.classList.contains('live')) {
            thread.push({ bot: true, html: `That’s a saved demo session — in this prototype I’ll start a fresh study instead. Tell me what to explore, or pick a direction below.` });
            chips = STATES.welcome.chips(); renderThread(); renderChips();
          }
          return;
        }
      });

      // composer
      const input = root.querySelector('#gdInput');
      const send = root.querySelector('#gdSend');
      function handleTextLocal() { const v = input.value.trim(); if (!v) return; input.value = ''; handleText(v); }
      send.addEventListener('click', handleTextLocal);
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); handleTextLocal(); } });
    },
  };

  /* handle free text (from composer or hint chips) */
  function handleText(v) {
    if (busy) return;
    if (autop && /\b(stop|pause|halt|cancel)\b/i.test(v)) { autop = false; pushUser(v); thread.push({ bot: true, html: `Autopilot paused — tap a suggestion to continue manually.` }); renderThread(); return; }
    const fn = parseIntent(v);
    if (fn) { fn(); return; }
    // fallback: advance the primary path of the current state, echoing the text
    const map = { frame: 'toData', toData: null, toCohort: 'toConcepts', toConcepts: 'toExtract', toReview: 'toRun', toFindings: null };
    const next = map[currentId];
    if (next) { go(next, v); }
    else { pushUser(v); thread.push({ bot: true, html: `I’ll treat that as “<em>${esc(v)}</em>”. In this guided demo I move step by step — tap a suggestion to continue, or say <strong>“why?”</strong>, <strong>“go back”</strong>, <strong>“use 30 patients”</strong>, or <strong>“run the whole demo”</strong>.` }); renderThread(); }
  }

  function stripText(s) { return s.replace(/\s+/g, ' ').trim(); }
})();
