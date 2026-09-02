/* Screen: Guided Copilot — conversational front door (v2).
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
  const { esc, escAttr: attr } = window.EU_HTML;
  const S = (window.SCREENS = window.SCREENS || {});
  const IDEA = window.EU_GUIDED_IDEA;
  const EXTRACT = window.EU_GUIDED_EXTRACT;
  const REVIEW = window.EU_GUIDED_REVIEW;
  const STARTUP = window.EU_GUIDED_STARTUP;
  const projectTitle = (value, fallback) => window.EU_PRODUCT_LABELS.projectTitle(value, fallback);
  const {
    BRANCH, CLARIFY, DEPTH, DEPTH_ORDER, STEP_INDEX, STUDY,
    compactHash, compactPath, fmtFixed, fmtInt, fmtNum, fmtP, fmtPct, fmtRunTime,
    guidedGateCheckRows, guidedGateFailedNames, guidedGateState, guidedJobEndError,
  } = window.EU_GUIDED_CONTRACTS;

  /* The idea sub-flow owns its own state and reaches the shell only through
     these. Thread and chips are handed over as accessors because the shell
     reassigns the arrays on reset — passing the arrays themselves would leave
     the sub-flow writing into a detached conversation. */
  IDEA.init({
    thread: () => thread,
    chips: () => chips,
    clearChips: () => { chips = []; },
    activeExportSource,
    bi,
    compactHash,
    compactPath,
    fmtInt,
    fmtNum,
    fmtPct,
    guidedMetricCard,
    markThrough,
    pushUser,
    renderAside,
    renderChips,
    renderThread,
    scheduleGuidedSlotSave,
    setVal,
  });

  /* ============== runtime state ============== */
  let branch, depth, dataMode, mods, cohortPhase, extractPhase, runPhase, draftPhase;
  let thread, chips, busy, expandedStep, whyOpen, autop, patientN, clarified, outputsReady, diffExpanded, liveAgentRun, workspaceSnapshot, workspaceSnapshotPath, guidedAgent;
  let guidedRunStream = null;
  const guidedRunChannel = window.EU_AGENT_STUDY_CONTEXT.createRunChannel();

  function disconnectGuidedRunUi() {
    if (guidedRunStream) {
      try { guidedRunStream.close(); } catch (_) {}
      guidedRunStream = null;
    }
    guidedRunChannel.clear();
  }

  let guidedDrafts = { loading: false, error: null, data: null };
  let guidedCopilot = { loading: false, error: null, session: null, last: null };
  let selectedGuidedRun = null;
  let selectedGuidedDraft = null;
  let pendingGuidedGoal = null;
  let guidedFrontdoorSeedText = null;
  let guidedFolderMenuOpen = false;
  let guidedDraftRemoval = null;
  let guidedFolderDialogMode = null;
  let guidedFolderSeedTitle = 'New local study';
  let guidedDraftFolderSlug = '';
  let guidedDraftParentDir = '~/easyicu/projects';
  let guidedFolderBrowser = { open: false, loading: false, error: null, data: null, path: '' };
  let guidedKnownProjectsOpen = false;
  let guidedPipelineOpen = false;
  let guidedSlotSaveTimer = null;
  let guidedMounted = false;
  let guidedInitialRender = false;
  let guidedComposerDraft = '';
  let studyParams;   // dynamic params extracted from clarify answers + free text
  // The user's own words, kept verbatim. `frameFor()` only ever proposes a
  // rewording — it must never silently become the question we submit, persist
  // or bind evidence to. `acceptedFrame` records an explicit user acceptance.
  let userQuestion = '';
  let acceptedFrame = false;
  // Typed study-contract proposal read from `userQuestion` by the intent owner
  // (screens-guided-intent.js). Null until it answers; null also means "we do
  // not know", never "nothing to configure".
  let studyContract = null;

  const DEFAULT_MODS = ['Demographics', 'Vital signs', 'Lab — Chemistry', 'SOFA-2 scores', 'Sepsis-3 (SOFA-2)', 'Outcome'];
  const GUIDED_EXTRACT_WINDOW_HOURS = 24 * 30;
  /* [id, English label, Chinese label, is-core]. Concept counts used to be a
     fourth column, kept as a fallback for guidedModuleConceptCount(). Two of
     the nineteen had gone stale — renal said 22 against the catalog's 35, and
     neurological 12 against 14 — so the fallback could only ever have shown a
     number that under-promises what the module actually extracts. The catalog
     is a static table loaded before this screen and is held to the backend
     concept-by-concept by test_concept_catalog_consistency.py, so it is the
     only source now; if it were ever missing, an empty count is honest where
     a stale one is not. */
  const GUIDED_EXTRACT_MODULES = [
    ['demographics', 'Demographics', '人口统计', true],
    ['vitals', 'Vital signs', '生命体征', true],
    ['chemistry', 'Lab — Chemistry', '实验室-生化', true],
    ['sofa2_score', 'SOFA-2 scores', 'SOFA-2 评分', true],
    ['sepsis3_sofa2', 'Sepsis-3 (SOFA-2)', 'Sepsis-3 (SOFA-2)', true],
    ['outcome', 'Outcome', '结局', true],
    ['sofa1_score', 'SOFA-1 scores', 'SOFA-1 评分', false],
    ['sepsis3_sofa1', 'Sepsis-3 (SOFA-1)', 'Sepsis-3 (SOFA-1)', false],
    ['sepsis_shared', 'Sepsis shared', 'Sepsis 共享概念', false],
    ['respiratory', 'Respiratory', '呼吸系统', false],
    ['ventilator', 'Ventilator', '呼吸机参数', false],
    ['blood_gas', 'Blood gas', '血气分析', false],
    ['hematology', 'Lab — Hematology', '实验室-血液学', false],
    ['vasopressors', 'Vasopressors', '血管活性药物', false],
    ['medications', 'Other medications', '其他药物', false],
    ['renal', 'Renal & urine output', '肾脏与尿量', false],
    ['neurological', 'Neurological', '神经系统', false],
    ['circulatory', 'Circulatory', '循环系统', false],
    ['other_scores', 'Other scores', '其他评分', false],
  ];
  const GUIDED_CORE_MODULES = GUIDED_EXTRACT_MODULES.filter(m => m[3]).map(m => m[0]);
  const GUIDED_COHORT_PRESETS = [
    ['all_icu', 'All ICU stays', '全部 ICU 住院', 'Broad denominator, no diagnosis filter.', '宽队列，不预设诊断筛选。'],
    ['adult_first', 'Adult first ICU stay', '成年首次 ICU', 'Default denominator for most extraction workflows.', '多数抽取流程的默认分母。'],
    ['sepsis3', 'Sepsis-3 / suspected infection', 'Sepsis-3 / 疑似感染', 'Uses Sepsis concepts when available; ICD is not prefilled.', '可用时使用 Sepsis 概念；不会预填 ICD。'],
    ['aki', 'AKI / renal dysfunction', 'AKI / 肾功能异常', 'Renal cohort starting point.', 'AKI 研究的肾功能队列起点。'],
    ['ventilation', 'Mechanical ventilation', '机械通气', 'Ventilator exposure cohort starting point.', '机械通气暴露队列起点。'],
    ['vasopressor', 'Vasopressor exposure', '血管活性药物暴露', 'Shock or pressor cohort starting point.', '休克/升压药队列起点。'],
    ['respiratory', 'Respiratory failure', '呼吸衰竭', 'Respiratory support and blood-gas focused cohort.', '呼吸支持与血气相关队列。'],
  ];
  EXTRACT.init({
    t,
    icon,
    esc,
    attr,
    fmtInt,
    compactPath,
    bi,
    modules: GUIDED_EXTRACT_MODULES,
    coreModules: GUIDED_CORE_MODULES,
    cohortPresets: GUIDED_COHORT_PRESETS,
    thread: () => thread,
    clearChips: () => { chips = []; },
    pushUser,
    setDataMode: value => { dataMode = value; },
    setVal,
    markThrough,
    applyStudyDesign({ outcome, comparator, comparatorKind, windowLabel }) {
      if (outcome) studyParams.outcome = outcome;
      if (comparator && comparatorKind !== 'none') studyParams.exposure = comparator;
      if (windowLabel) studyParams.window = windowLabel;
      const railBits = [outcome || '', windowLabel].filter(Boolean).join(' · ');
      if (railBits) setVal({ question: railBits });
    },
    renderThread,
    renderChips,
    renderAside,
    scheduleGuidedSlotSave,
    guidedJobEndError,
  });
  REVIEW.init({
    t,
    icon,
    esc,
    attr,
    fmtInt,
    fmtNum,
    fmtPct,
    fmtFixed,
    fmtP,
    bi,
    activeExportSource,
    activeExportLabel,
    thread: () => thread,
    clearChips: () => { chips = []; },
    pushUser,
    setDataMode: value => { dataMode = value; },
    setVal,
    markThrough,
    renderThread,
    renderChips,
    renderAside,
    scheduleGuidedSlotSave,
  });
  function reset() {
    disconnectGuidedRunUi();
    branch = 'predict'; depth = 'full'; dataMode = 'demo'; mods = DEFAULT_MODS.slice();
    cohortPhase = 'normal'; extractPhase = 'run'; runPhase = 'run'; draftPhase = 'gate';
    thread = []; chips = []; busy = false; expandedStep = 'question'; whyOpen = {}; autop = false; patientN = 10; clarified = null; outputsReady = false; diffExpanded = false; liveAgentRun = null; workspaceSnapshot = null; workspaceSnapshotPath = null; guidedAgent = null; IDEA.clearIdeaState(); EXTRACT.clearState(); REVIEW.clearState();
    pendingGuidedGoal = null;
    guidedFrontdoorSeedText = null;
    guidedFolderMenuOpen = false;
    guidedDraftRemoval = null;
    if (window.EU_GUIDED_PROJECTS && window.EU_GUIDED_PROJECTS.setProjectManagement) {
      window.EU_GUIDED_PROJECTS.setProjectManagement(false);
    }
    guidedFolderDialogMode = null;
    guidedFolderSeedTitle = 'New local study';
    guidedDraftFolderSlug = '';
    guidedDraftParentDir = '~/easyicu/projects';
    guidedFolderBrowser = { open: false, loading: false, error: null, data: null, path: '' };
    guidedKnownProjectsOpen = false;
    guidedPipelineOpen = false;
    studyParams = { outcome: 'In-hospital mortality', window: 'full available window', exposure: 'lactate', scope: 'all 19 modules', caught: null };
    userQuestion = ''; acceptedFrame = false; studyContract = null;
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
        ['Evaluate evidence checks', 'done'],
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
  /* Remember the user's own words. Called wherever free text first arrives. */
  function rememberUserQuestion(text) {
    const raw = String(text == null ? '' : text).trim();
    // Chip labels and control tokens are UI affordances, not the research
    // question — they must not overwrite what the user actually typed.
    if (!raw || raw.startsWith('@')) return;
    if (!userQuestion) { userQuestion = raw; refreshStudyContract(); }
  }
  /* Ask the intent owner to read the question into a typed contract. Stale
     answers are dropped via `gen`; a failed read leaves `studyContract` null,
     which the card renders as "unknown", never as an empty study. */
  function refreshStudyContract() {
    studyContract = null;
    if (!userQuestion || !window.EU_STUDY_INTENT || !window.EU_STUDY_INTENT.extract) return;
    const myGen = gen;
    const asked = userQuestion;
    window.EU_STUDY_INTENT.extract(asked).then(contract => {
      if (myGen !== gen || asked !== userQuestion) return;
      studyContract = contract;
      renderThread(); renderAside();
    }).catch(() => {});
  }
  /* A later correction AMENDS the question of record. It appends rather than
     replaces: "my outcome is AKI, not death" on its own loses the exposure the
     user gave in their first sentence, so both are kept and re-read together.
     Returns true when the question of record actually changed. */
  function replaceUserQuestion(text) {
    const raw = String(text == null ? '' : text).trim();
    if (!raw || raw.startsWith('@') || raw.length < 8) return false;
    if (raw === userQuestion || (userQuestion && userQuestion.endsWith(raw))) return false;
    userQuestion = userQuestion ? `${userQuestion} ${raw}` : raw;
    acceptedFrame = false;
    extractEntities(raw);
    refreshStudyContract();
    return true;
  }
  /* The question we are entitled to submit / persist / bind evidence to.
     Defaults to the user's own wording; a template framing is only used when
     the user explicitly accepted it, or when they never gave us any words. */
  function submittedQuestion() {
    if (userQuestion && !acceptedFrame) return userQuestion;
    if (acceptedFrame) return stripQuotes(frameFor(branch)) || userQuestion;
    return userQuestion || stripQuotes(frameFor(branch)) || (BRANCH[branch] ? BRANCH[branch].chip : '');
  }
  function stripQuotes(value) {
    return String(value == null ? '' : value).replace(/^[“"']+|[”"']+$/g, '').trim();
  }
  function tg(en, zh) { return window.t ? window.t(en, zh) : en; }
  /* Slots the template framing filled from its own defaults because nothing in
     the user's words matched. Naming them is the difference between "here is a
     tighter framing of your question" and an undisclosed substitution. */
  function unreadSlots() {
    const caught = String(studyParams.caught || '');
    const slots = [];
    if (branch !== 'predict') return slots;
    if (!/lactate|SOFA|MAP|creatinine|heart rate|WBC/i.test(caught)) {
      slots.push([tg('exposure', '暴露'), studyParams.exposure]);
    }
    if (!/28-day|ICU mortality/i.test(caught)) {
      slots.push([tg('outcome', '结局'), studyParams.outcome]);
    }
    if (!/first \d+h/i.test(caught)) {
      slots.push([tg('time window', '时间窗'), studyParams.window]);
    }
    slots.push([tg('population', '人群'), 'Sepsis-3']);
    return slots;
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
  function stripTags(s) { return String(s).replace(/<[^>]*>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').trim(); }
  function bi(en, zh) { return { en, zh }; }
  function htmlOf(value) {
    if (value && typeof value === 'object' && Object.prototype.hasOwnProperty.call(value, 'en')) {
      return window.t ? window.t(value.en, value.zh) : value.en;
    }
    return value == null ? '' : String(value);
  }
  function pushBot(en, zh) { thread.push({ bot: true, html: bi(en, zh) }); }

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
    pushBot(
      `Sure — let’s adjust this. Anything downstream will re-run from here.`,
      `可以，我们从这里调整。后续步骤会基于这个改动重新运行。`,
    );
    const st = STATES[STEP_STATE[step]];
    chips = (st && (typeof st.chips === 'function' ? st.chips() : st.chips)) || [];
    renderThread(); renderChips();
  }

  /* ============== timed sub-flows ============== */
  /* Seeded demo animation ONLY. The row states and the `durs` timings here are
     invented: rows tick on a 380-600ms timer while `durs` claims seconds. In
     real mode that would assert work the app has not done yet (the caller's
     real work runs inside `done`), so real mode gets an honest indeterminate
     state and the work starts immediately instead. */
  function markTasksIndeterminate(sel) {
    document.querySelectorAll(sel + ' .gd-task').forEach(r => {
      r.className = 'gd-task running';
      r.setAttribute('data-progress-source', 'live-indeterminate');
      const tk = r.querySelector('.tk');
      if (tk) tk.innerHTML = '<span class="spin sm accent" style="width:11px;height:11px;"></span>';
      // No per-row duration: we do not know it, and inventing one is the bug.
      const d = r.querySelector('.tdur'); if (d) { d.textContent = ''; d.style.color = ''; }
    });
  }
  function streamTasks(sel, durs, done, opts) {
    if (realMode()) { markTasksIndeterminate(sel); done(); return; }
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
        pushBot(
          `Done — frames packaged and frozen locally. This is your finish line for an extract-only run.`,
          `完成：数据表已在本地打包并冻结。这里就是仅抽取流程的终点。`,
        );
        chips = [['Finish &amp; export', '@finish', 'express'], ['Open in workspace', '@open'], ['Take it further → review', '@extendNext']];
        renderThread(); renderChips();
        if (autop) schedule(() => finishHere());
        return;
      }
      pushBot(
        `Done — the workspace is loaded and frozen for analysis. Want a quick look before we run?`,
        `完成：工作区已加载并冻结，可用于分析。运行前要先快速看一眼吗？`,
      );
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
      const path = src && src.path ? src.path : 'No export selected yet';
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
      const path = src && src.path ? src.path : 'No export selected yet';
      const tasks = ['Read folder tree', 'Match known layout', 'Read export manifest', 'Index tables'];
      return `
      <div class="gd-card" style="max-width:600px;margin-left:39px;">
        <div class="gc-head"><div class="gc-ico">${icon('db', 15)}</div><div class="grow"><div class="gc-t">Detecting schema</div><div class="gc-sub mono">${esc(path)}</div></div></div>
        <div class="gc-body">
          <div class="gd-prog" id="gdDetect">${tasks.map(t => `<div class="gd-task queued" data-progress-source="${realMode() ? 'live' : 'scripted'}"><span class="tk">${icon('clock', 9)}</span><span class="grow">${t}</span><span class="tdur"></span></div>`).join('')}</div>
          <div class="indet mt-12"></div>
        </div>
      </div>`;
    },
    detected() {
      const src = activeExportSource();
      return `
      <div class="gd-card" style="max-width:600px;margin-left:39px;">
        <div class="gc-head"><div class="gc-ico" style="background:var(--ok-soft);color:var(--ok);border-color:oklch(88% 0.05 150);">${icon('check', 14, 3)}</div><div class="grow"><div class="gc-t">Detected ${esc(activeExportLabel())}</div><div class="gc-sub">local · read from manifest</div></div></div>
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
        pushBot(
          `Recognized <strong>${esc(activeExportLabel())}</strong> — read its module manifest. Files stay on your machine.`,
          `已识别 <strong>${esc(activeExportLabel())}</strong>，已读取其模块清单。文件仍留在你的机器上。`,
        );
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
    if (dataMode !== 'demo') {
      const unavailableToken = guidedRunChannel.start({
        surface: 'guided-legacy',
        study_id: branch || 'guided',
        question: submittedQuestion(),
        source_path: src && src.path,
      });
      failLivePipeline(
        unavailableToken,
        src
          ? 'The Agent backend or browser event stream is unavailable; no real run was submitted.'
          : 'No active registered export is selected; no real run was submitted.',
      );
      return;
    }
    streamTasks('#gdRunProg', BRANCH[branch].runTasks.map(t => t[1]), () => {
      runPhase = 'done';
      const p = document.getElementById('gdRunPill'); if (p) p.outerHTML = '<span class="pill ok" id="gdRunPill"><span class="dot"></span>Complete</span>';
      outputsReady = true;
      renderThread();
      pushBot(
        `Run complete — six artifacts written locally and logged to the evidence ledger. <span style="color:var(--ink-4);">(Step 4 hit a singular matrix; auto-repair dropped one collinear feature and re-fit — logged in the ledger.)</span>`,
        `运行完成：6 个 artifact 已写入本地，并记录到 evidence ledger。<span style="color:var(--ink-4);">(第 4 步遇到奇异矩阵；自动修复删除了一个共线特征并重新拟合，已写入 ledger。)</span>`,
      );
      thread.push({ diff: true });
      renderThread(); renderAside();
      pushBot(
        `I’ve drafted findings from these — but the manuscript draft stays <strong>locked</strong> until you sign off.`,
        `我已经基于这些结果生成 findings 草稿，但在你签署前，稿件草稿会保持<strong>锁定</strong>。`,
      );
      chips = []; renderThread();
      go('toFindings');
    }, { failAt: 3 });
  }

  function runLivePipeline(src) {
    const capturedBranch = branch;
    // The run binds evidence to this string, so it must be the user's own
    // question unless they explicitly accepted a proposed rewording.
    const capturedQuestion = submittedQuestion();
    let runToken = guidedRunChannel.start({
      surface: 'guided-legacy',
      study_id: capturedBranch || 'guided',
      context_id: window.EU_GUIDED_STUDY_CONTEXT && window.EU_GUIDED_STUDY_CONTEXT.activeId ? window.EU_GUIDED_STUDY_CONTEXT.activeId() : '',
      question: capturedQuestion,
      source_path: src.path,
      study_mode: 'analysis',
      run_type: 'preflight',
      provider: 'mock',
    });
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
    const contextReady = window.EU_GUIDED_STUDY_CONTEXT && window.EU_GUIDED_STUDY_CONTEXT.persistForRun
      ? window.EU_GUIDED_STUDY_CONTEXT.persistForRun('agent_preflight')
      : Promise.reject(new Error('StudyContext persistence is unavailable; the real Guided run was not submitted.'));
    contextReady.then(studyContext => {
      if (!studyContext || !studyContext.id) throw new Error('StudyContext persistence did not return a project id.');
      runToken = guidedRunChannel.bind(runToken, { context_id: studyContext.id, study_id: studyContext.id });
      return window.EU_API.startAgentRun({
        path: runToken.source_path,
        study_id: runToken.study_id,
        study_context_id: runToken.context_id,
        mode: runToken.study_mode,
        run_type: runToken.run_type,
        llm_provider: runToken.provider,
        external_llm_opt_in: false,
        question: runToken.question,
      });
    }).then(r => {
      runToken = guidedRunChannel.bind(runToken, { job_id: r.job_id, context_revision: r.study_context_revision });
      if (window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.markContextRunning) {
        window.EU_AGENT_STUDY_CONTEXT.markContextRunning(runToken.context_id, runToken.job_id, runToken.context_revision);
      }
      const es = new EventSource('/api/jobs/' + encodeURIComponent(runToken.job_id) + '/events');
      if (guidedRunChannel.isCurrent(runToken)) {
        guidedRunStream = es;
        liveAgentRun.jobId = runToken.job_id;
      }
      let ended = false;
      es.onmessage = msg => {
        const ev = JSON.parse(msg.data);
        if (ev.type === 'end') {
          ended = true;
          try { es.close(); } catch (_) {}
          if (window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.markContextFinished) {
            window.EU_AGENT_STUDY_CONTEXT.markContextFinished(
              runToken.context_id,
              ev.status,
              ev.result || null,
              runToken.job_id,
              ev.study_context_revision || (ev.result && ev.result.study_context_revision),
            );
          }
          if (ev.status === 'done') completeLivePipeline(runToken, ev.result);
          else failLivePipeline(runToken, guidedJobEndError(ev) || 'Agent run failed.');
          return;
        }
        if (!guidedRunChannel.isCurrent(runToken)) return;
        if (ev.label) liveAgentRun.step = ev.label;
        if (ev.total) setProgress(ev.current, ev.total, ev.label);
        const pill = document.getElementById('gdRunPill');
        if (pill && ev.label) pill.innerHTML = `<span class="dot"></span>${esc(ev.label)}`;
      };
      es.onerror = () => {
        if (ended) return;
        try { es.close(); } catch (_) {}
        failLivePipeline(runToken, 'Lost connection to the agent run. The server job may still be running.');
      };
    }).catch(err => failLivePipeline(runToken, err.message || String(err)));
  }

  function completeLivePipeline(runToken, result) {
    if (!guidedRunChannel.isCurrent(runToken)) return;
    if (guidedRunStream) { try { guidedRunStream.close(); } catch (_) {} guidedRunStream = null; }
    liveAgentRun = { active: false, result: result, error: null };
    runPhase = 'done';
    const gateState = guidedGateState(result);
    outputsReady = !gateState.blocked;
    if (result && result.summary) workspaceSnapshot = { summary: result.summary, cohort: result.cohort || {}, quality: result.quality || [] };
    if (gateState.blocked) {
      setVal({ analysis: 'verification blocked', draft: 'locked · review_blocked' });
      markThrough('analysis', 'locked');
      setStudy({ draft: 'locked' });
      const p = document.getElementById('gdRunPill');
      if (p) p.outerHTML = '<span class="pill bad" id="gdRunPill"><span class="dot"></span>Verification blocked</span>';
      renderThread();
      const reason = gateState.gate.reason ? ` <span class="mono">${esc(gateState.gate.reason)}</span>` : '';
      const failedNames = guidedGateFailedNames(gateState);
      const failedEn = failedNames.length ? ` Failed checks: <strong>${failedNames.map(esc).join(' · ')}</strong>.` : '';
      const failedZh = failedNames.length ? ` 未通过的检查：<strong>${failedNames.map(esc).join(' · ')}</strong>。` : '';
      pushBot(
        `The run finished, but evidence verification blocked the Findings step.${reason}${failedEn} Artifacts were retained for review; the manuscript draft remains <strong>locked</strong>.`,
        `运行已结束，但证据核验未通过，因此没有进入 Findings。${reason}${failedZh} Artifacts 已保留供复核；稿件草稿仍保持<strong>锁定</strong>。`,
      );
      chips = [['Review blocked checks', '@reviewBlocked'], ['Retry analysis', 'toRun'], ['Open Project Monitor', '@openAgent']];
      guidedRunChannel.clear(runToken);
      renderThread(); renderAside(); renderChips();
      return;
    }
    setVal({ analysis: () => analysisLine(), draft: 'locked · analysis_only' });
    const p = document.getElementById('gdRunPill'); if (p) p.outerHTML = '<span class="pill ok" id="gdRunPill"><span class="dot"></span>Preflight complete</span>';
    renderThread();
    pushBot(
      `Run complete — registry-backed preflight artifacts were written locally and logged to the evidence ledger. <span style="color:var(--ink-4);">No patient rows were persisted and no external model call was made.</span>`,
      `运行完成：registry-backed 预检 artifacts 已写入本地，并记录到 evidence ledger。<span style="color:var(--ink-4);">没有持久化患者行，也没有外部模型调用。</span>`,
    );
    thread.push({ diff: true });
    renderThread(); renderAside();
    pushBot(
      `I can open this in Project Monitor now. Manuscript claims remain <strong>locked</strong> until human sign-off.`,
      `现在可以在项目监控中打开它。人工签署前，稿件 claims 仍保持<strong>锁定</strong>。`,
    );
    chips = []; renderThread();
    guidedRunChannel.clear(runToken);
    go('toFindings');
  }

  function failLivePipeline(runToken, error) {
    if (!guidedRunChannel.isCurrent(runToken)) return;
    if (guidedRunStream) { try { guidedRunStream.close(); } catch (_) {} guidedRunStream = null; }
    liveAgentRun = { active: false, result: null, error: error };
    runPhase = 'run';
    const p = document.getElementById('gdRunPill'); if (p) p.outerHTML = '<span class="pill bad" id="gdRunPill"><span class="dot"></span>Failed closed</span>';
    pushBot(
      `The run failed closed: <span class="mono">${esc(error)}</span>`,
      `这次 run 已 fail-closed：<span class="mono">${esc(error)}</span>`,
    );
    chips = [['Retry analysis', 'toRun'], ['Open Project Monitor', '@openAgent']];
    guidedRunChannel.clear(runToken);
    renderThread(); renderChips();
  }
  function schedule(fn) { const myGen = gen; const t = () => { if (myGen !== gen) return; if (busy) return setTimeout(t, 160); fn(); }; setTimeout(t, 520); }

  /* ============== card renderers (one per step) ============== */
  /* `title` and `sub` are plain text from every caller, and one of them carries
     the active export's label — which comes from the prepared export's own
     manifest (`database`), not from anything this app authored. Prepared
     exports are the artifact researchers hand to each other, so a crafted
     manifest field would have run in the recipient's page, on an origin that
     also serves the local filesystem API. Escape at the boundary that renders,
     so a future caller cannot reintroduce it. `bodyHtml`/`footHtml` are markup
     by contract and stay raw. */
  function cardShell(step, ico, title, sub, bodyHtml, footHtml) {
    const w = BRANCH[branch].why[step];
    const on = whyOpen[step];
    return `
    <div class="gd-card" data-card-step="${step}">
      <div class="gc-head">
        <div class="gc-ico">${icon(ico, 15)}</div>
        <div class="grow"><div class="gc-t">${esc(title)}</div><div class="gc-sub">${esc(sub)}</div></div>
        ${w ? `<button class="gc-why ${on ? 'on' : ''}" data-why="${step}">${icon('help', 11)} Why${on ? '' : ' this step'}</button>` : ''}
      </div>
      ${w ? `<div class="gd-why" ${on ? '' : 'hidden'}>${w}</div>` : ''}
      <div class="gc-body">${bodyHtml}</div>
      ${footHtml ? `<div class="gd-cardfoot">${footHtml}</div>` : ''}
    </div>`;
  }

  const CARD = {
    question() {
      const unread = unreadSlots();
      // The user's own words are the question. The template below is a
      // proposal; it only becomes the submitted question via @useFrame.
      const mine = userQuestion
        ? `<div class="eyebrow" style="margin:0 0 6px;">${tg('Your question', '你的问题')}</div>
           <p style="font-size:12.5px;color:var(--ink);margin:0 0 12px;line-height:1.5;">${esc(userQuestion)}</p>`
        : '';
      const proposalLabel = acceptedFrame
        ? tg('Wording you accepted', '你已采用的措辞')
        : tg('Suggested wording (template — not yet applied)', '建议措辞(模板 · 尚未采用)');
      // Preferred: the typed contract read from the user's own words. It names
      // what it could not read instead of defaulting, so it replaces the
      // template-gap warning entirely when present.
      const contractHtml = (studyContract && window.EU_STUDY_INTENT && window.EU_STUDY_INTENT.cardHtml)
        ? window.EU_STUDY_INTENT.cardHtml(studyContract, { icon })
        : '';
      const gap = contractHtml ? '' : (!acceptedFrame && userQuestion && unread.length)
        ? `<div class="note warn mt-12" style="padding:9px 11px;"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="d" style="font-size:11px;margin:0;">
             ${tg('I could not read these from your words, so the suggestion below uses defaults:', '下面这些我没能从你的话里读出来,建议措辞用的是默认值:')}
             <strong>${unread.map(([k, v]) => `${esc(k)} = ${esc(v)}`).join(' · ')}</strong>.
             ${tg('Continuing keeps your own wording.', '继续将保留你自己的表述。')}
           </div></div></div>`
        : '';
      return cardShell('question', 'spark', 'Study plan', 'forming', `
        ${mine}
        ${contractHtml}
        <div class="eyebrow" style="margin:${contractHtml ? '14px' : '0'} 0 6px;">${esc(proposalLabel)}</div>
        <p style="font-size:12.5px;color:var(--ink-2);font-style:italic;margin:0 0 12px;line-height:1.5;">${frameFor(branch)}</p>
        <div class="col gap-6" style="font-size:12.25px;">
          ${planFor(branch).map(([k, v]) => `<div class="setup-row"><span class="k">${k}</span><span class="vv">${v}</span></div>`).join('')}
        </div>
        ${gap}
        <div class="m-cite" style="margin-top:11px;">${icon('shield', 11)} evidence-bound · I won’t assert effect sizes</div>`,
        `<button class="btn primary sm" data-go="toData">${userQuestion && !acceptedFrame ? tg('Continue with my wording', '用我的表述继续') : tg('Looks right — continue', '没问题,继续')} ${icon('arrow', 13)}</button>
         ${userQuestion && !acceptedFrame ? `<button class="btn sm" data-go="@useFrame">${tg('Use the suggested wording', '改用建议措辞')}</button>` : ''}
         <button class="btn sm" data-go="welcome">${tg('Reframe', '重新表述')}</button>`);
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
      const ALL = [['Demographics', 6], ['Vital signs', 8], ['Lab — Chemistry', 22], ['SOFA-2 scores', 10], ['Sepsis-3 (SOFA-2)', 1], ['Outcome', 3], ['Respiratory', 14], ['Renal & urine output', 20]];
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
          ${tasks.map(t => `<div class="gd-task queued" data-progress-source="${realMode() ? 'live' : 'scripted'}"><span class="tk">${icon('clock', 9)}</span><span class="grow">${t}</span><span class="tdur"></span></div>`).join('')}
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
        // For a live run, drive the pill + progress bar from the real evidence
        // gate — never hard-pin 100%/green when the gate came back blocked.
        const gateState = guidedGateState(live);
        const gchecks = gateState.checks;
        const gpassed = gateState.passed;
        const gateBlocked = gateState.blocked;
        const barPct = live ? (gchecks.length ? Math.round((gpassed / gchecks.length) * 100) : 100) : 100;
        const pillCls = gateBlocked ? 'warn' : 'ok';
        const pillTxt = live ? (gateBlocked ? `Verification: ${gpassed}/${gchecks.length} checks` : 'Preflight complete') : 'Complete';
        const subTxt = live ? `preflight ${gateBlocked ? 'blocked' : 'passed'} · ${(live.artifacts || []).length} artifacts` : 'complete · 6 artifacts';
        // Blocked runs list the actual gate checks (named, with pass/fail per
        // row) — not the generic all-green task list that hid which check failed.
        const progRows = live && gateBlocked
          ? guidedGateCheckRows(gateState)
          : tasks.map(([tk, d]) => `<div class="gd-task done" data-progress-source="${realMode() ? 'live' : 'scripted'}"><span class="tk">${icon('check', 10, 3)}</span><span class="grow">${tk}</span><span class="tdur">${d}</span></div>`).join('');
        return cardShell('analysis', 'agent', 'Research Agent · run', subTxt, `
          <div class="gd-prog">${progRows}</div>
          <div class="run-strip mt-12" style="padding:8px 10px;"><span class="pill ${pillCls}"><span class="dot"></span>${pillTxt}</span><div class="grow runbar"><div class="runbar-fill" style="width:${barPct}%"></div></div></div>`, '');
      }
      return cardShell('analysis', 'agent', 'Research Agent · run', dataMode !== 'demo' ? 'registry-backed · local preflight' : 'demo pipeline · no tokens', `
        <div class="gd-prog" id="gdRunProg">${tasks.map(([t]) => `<div class="gd-task queued" data-progress-source="${realMode() ? 'live' : 'scripted'}"><span class="tk">${icon('clock', 9)}</span><span class="tt-cmd">py</span><span class="grow">${t}</span><span class="tdur"></span></div>`).join('')}</div>
        <div class="run-strip mt-12" style="padding:8px 10px;"><span class="pill warn" id="gdRunPill"><span class="dot"></span>Running</span><div class="grow runbar"><div class="runbar-fill" id="gdRunProg-bar" style="width:0%;transition:width .12s linear;"></div></div></div>`, '');
    },
    draft() {
      const b = BRANCH[branch];
      const live = liveAgentRun && liveAgentRun.result;
      if (live) {
        const gateState = guidedGateState(live);
        const gate = gateState.gate;
        const checks = gateState.checks;
        const title = gateState.blocked ? 'Evidence verification blocked · draft locked' : 'Preflight complete · draft locked';
        const intro = gateState.blocked
          ? 'Preflight execution finished, but evidence verification blocked the Findings step.'
          : 'Local preflight finished.';
        return cardShell('draft', 'shield', title, gate.status || 'analysis_only', `
          <div class="m-bubble" style="background:var(--surface-2);border:1px solid var(--hair);font-size:12.25px;margin-bottom:12px;">${intro} Run <span class="mono">${esc(live.run_id || 'run')}</span>. <span style="color:var(--ink-4);">No external model call, no uploads, and no patient rows persisted. Manuscript claims remain locked.</span></div>
          <div class="eyebrow" style="margin:0 0 8px;">Evidence checks</div>
          <div class="checks">
            ${checks.map(c => {
              const pending = !c.passed && c.id === 'human_signoff';
              const state = c.passed ? 'ok' : pending ? 'pending' : 'bad';
              const markerStyle = c.passed || pending ? '' : 'background:var(--bad-soft);color:var(--bad);';
              const marker = c.passed ? icon('check', 11, 2.8) : pending ? icon('clock', 11) : icon('x', 11);
              const color = c.passed ? 'var(--ink)' : pending ? 'var(--ink-3)' : 'var(--bad)';
              const status = c.passed ? 'passed' : pending ? 'pending' : 'failed';
              return `<div class="check-row ${state}"><span class="check-mk" style="${markerStyle}">${marker}</span><span style="font-size:11.75px;color:${color};">${esc(c.label || c.id)}</span><span class="grow"></span><span class="mono" style="font-size:10px;color:${c.passed ? 'var(--ok)' : color};">${status}</span></div>`;
            }).join('')}
          </div>`,
          `<button class="btn primary sm" data-act="draft">Open in Research Agent</button>
           <button class="btn sm" data-act="open">${icon('grid', 13)} Open workspace</button>`);
      }
      if (draftPhase === 'signed') {
        return cardShell('draft', 'check', 'Study assembled', 'draft unlocked after checks', `
          <div class="col gap-6" style="font-size:12px;">
            <div class="setup-row"><span class="k">Data</span><span class="vv">${patientN} stays · ${mods.length} modules</span></div>
            <div class="setup-row"><span class="k">Analysis</span><span class="vv">5 steps · 6 artifacts</span></div>
            <div class="setup-row"><span class="k">Draft</span><span class="vv">unlocked after sign-off</span></div>
            <div class="setup-row"><span class="k">Bundle</span><span class="vv">local run folder selected at runtime</span></div>
          </div>`,
          `<button class="btn primary sm" data-act="open">${icon('arrow', 13)} Open full workspace</button>
           <button class="btn sm" data-act="draft">Open draft</button>
           <button class="btn sm ghost" data-go="welcome">New study</button>`);
      }
      return cardShell('draft', 'shield', 'Findings drafted · review required', 'evidence-bound', `
        <div class="m-bubble" style="background:var(--surface-2);border:1px solid var(--hair);font-size:12.25px;margin-bottom:12px;">${b.findings}</div>
        <div class="eyebrow" style="margin:0 0 8px;">Evidence checks</div>
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
      ? bi(
          `That’s your finish line for an <strong>extract-only</strong> run — cohort frames and a reproducible <code>manifest.json</code> are written locally. Nothing left this machine.`,
          `这里就是<strong>仅抽取</strong>的终点：队列数据表和可复现 <code>manifest.json</code> 已写入本机。没有任何数据离开这台机器。`,
        )
      : bi(
          `That’s your finish line for <strong>${DEPTH[depth].label}</strong> — the populated workspace is ready to explore. Nothing left this machine.`,
          `这里就是<strong>${esc(DEPTH[depth].label)}</strong>的终点：工作区已经填充好，可以继续查看。没有任何数据离开这台机器。`,
        );
    thread.push({ bot: true, html: msg });
    chips = [['Open in workspace', '@open', 'express'], (depth !== 'full' ? ['Actually, take it further', '@extendNext'] : null)].filter(Boolean);
    renderThread(); renderChips();
  }

  /* ============== inline native data extraction ============== */
  /* Extraction state, effects, rendering, and DOM transitions live in
     screens-guided-extract.js. These accessors are the small read contract
     used by project-memory restore and the downstream Agent handoff. */
  function resetGuidedExtractionState() { return EXTRACT.resetState(); }
  function resetGuidedDesignState() { return EXTRACT.resetDesignState(); }
  function guidedDesignOutcome() { return EXTRACT.resolveOutcome(EXTRACT.design(), t); }
  function guidedDesignComparator() { return EXTRACT.resolveComparator(EXTRACT.design(), t); }
  function guidedDesignWindowHours() {
    return EXTRACT.windowHours(EXTRACT.design()) || GUIDED_EXTRACT_WINDOW_HOURS;
  }
  /* ============== inline native review: patient + cohort + KM ============== */
  /* Review state, effects, presentation, and DOM transitions live in
     screens-guided-review.js. Agent keeps a tiny local metric primitive
     because its evidence card is not part of the active-export review. */
  function guidedMetricCard(label, value, sub) {
    return `<div class="gdr-metric"><span>${esc(label)}</span><strong>${esc(value == null ? 'n/a' : value)}</strong>${sub ? `<small>${esc(sub)}</small>` : ''}</div>`;
  }
  /* ============== inline Agent preflight ============== */
  function resetGuidedAgentState() {
    const guidedDesign = EXTRACT.design();
    disconnectGuidedRunUi();
    // Derive the objective from the study design the user actually collected in the
    // conversation — never a hard-coded exposure/outcome framing. If they never set
    // one, keep an honest generic sentence they can edit, not a fictional example.
    const collected = guidedDesign && guidedDesign.collected;
    const outcome = collected ? guidedDesignOutcome() : '';
    const comparator = collected ? guidedDesignComparator() : '';
    let question;
    if (outcome) {
      question = (comparator && guidedDesign.comparator !== 'none')
        ? `Evaluate ${outcome} in the active ICU cohort, comparing ${comparator}.`
        : `Evaluate ${outcome} in the active ICU cohort.`;
    } else {
      question = 'Evaluate the active ICU cohort with an evidence-bound local preflight.';
    }
    guidedAgent = {
      question,
      running: false,
      jobId: null,
      contextId: null,
      progress: null,
      result: null,
      error: null,
      warning: null,
    };
  }
  function startGuidedAgentFlow(label) {
    if (label) pushUser(label);
    dataMode = 'real';
    resetGuidedAgentState();
    setVal({ analysis: 'local preflight', draft: 'locked' });
    markThrough('analysis', 'active');
    thread.push({ bot: true, html: bi(
      `We can run the local Agent preflight here. It consumes the active export, writes local artifacts, and keeps the manuscript draft locked until evidence checks and human sign-off pass.`,
      `可以直接在这里启动本地 Agent 预检。它读取 active export、写入本地 artifacts，并在证据检查与人工签署前保持论文草稿锁定。`,
    ) });
    thread.push({ guidedAgent: true });
    chips = [];
    renderThread();
    renderChips();
    scheduleGuidedSlotSave('start_agent');
  }
  function guidedAgentStatusText() {
    if (!guidedAgent) return '';
    if (guidedAgent.running && guidedAgent.progress) {
      return esc(guidedAgent.progress.message || guidedAgent.progress.phase || guidedAgent.progress.step || 'running');
    }
    if (guidedAgent.running) return t('Starting local Agent preflight...', '正在启动本地 Agent 预检...');
    if (guidedAgent.error) return esc(guidedAgent.error);
    if (guidedAgent.result) {
      const gateState = guidedGateState(guidedAgent.result);
      return gateState.blocked
        ? `${t('Agent preflight blocked', 'Agent 预检未通过')} · ${esc(gateState.gate.status || 'blocked')}`
        : `${t('Agent preflight complete', 'Agent 预检完成')} · ${esc(gateState.gate.status || 'analysis_only')}`;
    }
    const src = activeExportSource();
    return src ? t('Ready to run against the active export. No external provider is used.', '可以基于 active export 运行。不会使用外部 provider。') : t('No active export. Prepare/register data first.', '没有 active export。请先准备或注册数据。');
  }
  // Carry the study configured in Copilot into the governed Research Agent
  // handoff so a run never starts from an empty form. Mirrors the backend
  // _prefill_for shape so app.js's handoff banner surfaces the same design.
  // Machine-readable config for the Copilot -> classic extraction exit. The
  // banner shows the human hints; this is what the classic form actually
  // consumes so the user does not re-enter what the conversation collected.
  function guidedExtractionClassicConfig() {
    const guidedExtract = EXTRACT.state();
    return {
      cohort_preset: (guidedExtract && guidedExtract.cohort) || '',
      modules: guidedExtract && Array.isArray(guidedExtract.modules) ? guidedExtract.modules.slice() : [],
      format: (guidedExtract && guidedExtract.format) || '',
      export_dir: (guidedExtract && guidedExtract.exportDir) || '',
      max_patients: guidedExtract && Number.isFinite(guidedExtract.maxPatients) ? guidedExtract.maxPatients : null,
      source_path: (guidedExtract && guidedExtract.path) || '',
      observation_window_hours: guidedDesignWindowHours(),
    };
  }
  function guidedAgentHandoffPrefill() {
    const guidedExtract = EXTRACT.state();
    const guidedDesign = EXTRACT.design();
    const collected = !!(guidedDesign && guidedDesign.collected);
    const windowLabel = window.EU_GUIDED_EXTRACT ? window.EU_GUIDED_EXTRACT.windowLabel(guidedDesign, t) : '';
    const src = activeExportSource();
    const mods = guidedExtract && Array.isArray(guidedExtract.modules) ? guidedExtract.modules : [];
    return {
      source: 'guided_copilot',
      goal: 'run_agent',
      question_hint: (guidedAgent && guidedAgent.question) || '',
      cohort_hint: (guidedExtract && guidedExtract.cohort) || '',
      module_hint: mods.length ? `${mods.length} ${t('modules', '模块')}` : '',
      outcome_hint: collected ? guidedDesignOutcome() : '',
      time_window_hint: collected ? windowLabel : '',
      comparator_hint: (collected && guidedDesign.comparator !== 'none') ? guidedDesignComparator() : '',
      export_destination_hint: (guidedExtract && guidedExtract.exportDir) || (src && src.path) || '',
    };
  }
  window.EU_GUIDED_CONTEXT = {
    snapshot() {
    const guidedExtract = EXTRACT.state();
    const guidedDesign = EXTRACT.design();
    const src = activeExportSource() || {};
    const windowLabel = window.EU_GUIDED_EXTRACT ? window.EU_GUIDED_EXTRACT.windowLabel(guidedDesign, t) : '';
    return {
      question: (guidedAgent && guidedAgent.question) || submittedQuestion(),
      source: {
        path: (guidedExtract && guidedExtract.result && (guidedExtract.result.out_dir || guidedExtract.result.path)) || src.path || '',
        label: src.label || src.database || (dataMode === 'demo' ? 'Demo data' : 'Local EasyICU export'),
        database: src.database || (dataMode === 'demo' ? 'demo' : ''),
      },
      cohort_preset: (guidedExtract && guidedExtract.cohort) || 'adult_first',
      max_patients: guidedExtract && guidedExtract.maxPatients,
      modules: guidedExtract && Array.isArray(guidedExtract.modules) ? guidedExtract.modules.slice() : [],
      outcome: guidedDesign && guidedDesign.collected ? guidedDesignOutcome() : '',
      window_preset: (guidedDesign && guidedDesign.window) || 'whole_stay',
      window_label: windowLabel,
      comparator: guidedDesign && guidedDesign.collected && guidedDesign.comparator !== 'none' ? guidedDesignComparator() : '',
      export_format: (guidedExtract && guidedExtract.format) || '',
      configured: !!(guidedDesign && guidedDesign.collected),
    };
    },
  };
  function openGuidedAgentHandoff() {
    if (window.EU_GUIDED_STUDY_CONTEXT && window.EU_GUIDED_STUDY_CONTEXT.handoff) {
      const sync = window.EU_GUIDED_STUDY_CONTEXT.handoff('agent');
      sync.persisted.catch(error => console.warn('[EasyICU] Guided StudyContext handoff stayed local:', error));
    }
    if (window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.set) {
      window.EU_GUIDED_HANDOFF.set({
        type: 'module_handoff', status: 'ready', goal: 'run_agent',
        target_route: 'agent', prefill: guidedAgentHandoffPrefill(), requires_user_confirm: true,
      });
    }
    location.hash = '#agent';
  }
  function renderGuidedAgentCard() {
    if (!guidedAgent) resetGuidedAgentState();
    const src = activeExportSource();
    const result = guidedAgent.result || {};
    const gate = result.gate || {};
    const gateBlocked = !!guidedAgent.result && guidedGateState(guidedAgent.result).blocked;
    const artifacts = result.artifacts || result.artifact_manifest || [];
    const artCount = Array.isArray(artifacts) ? artifacts.length : (result.artifact_count || 0);
    return `
      <div class="gd-agent-card">
        <div class="gdx-head">
          <span class="gdx-ico">${icon('agent', 15)}</span>
          <div>
            <strong>${t('Run Agent preflight inside Copilot', '在 Copilot 内运行 Agent 预检')}</strong>
            <span>${t('Uses the governed Research Agent backend, defaulting to local mock/preflight and evidence checks.', '使用受治理的 Research Agent 后端，默认本地 mock/preflight 与证据核验。')}</span>
          </div>
        </div>
        <label class="gda-question">
          <span>${t('Research question / run objective', '研究问题 / 运行目标')}</span>
          <textarea data-ga-question rows="3">${esc(guidedAgent.question || '')}</textarea>
        </label>
        <div class="gdx-source ${src ? '' : 'blocked'}">
          <span>${icon(src ? 'check' : 'shield', 12)}</span>
          <div><strong>${src ? t('Active local export', 'active 本地导出') : t('No active export', '没有 active export')}</strong><small>${src ? esc(activeExportLabel()) : t('Use Prepare Data first, or register an existing EasyICU export.', '先使用准备数据，或注册已有 EasyICU 导出。')}</small></div>
        </div>
        <div class="gdx-status ${guidedAgent.error || gateBlocked ? 'bad' : guidedAgent.result ? 'ok' : ''}">
          <span>${icon(guidedAgent.error ? 'x' : gateBlocked ? 'alert' : guidedAgent.result ? 'check' : 'shield', 12)}</span>
          <div><strong>${guidedAgentStatusText()}</strong>${guidedAgent.jobId ? `<small>job ${esc(guidedAgent.jobId)}</small>` : ''}</div>
        </div>
        ${window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.warningNote ? window.EU_AGENT_STUDY_CONTEXT.warningNote(guidedAgent.warning) : ''}
        ${guidedAgent.result ? `<div class="gda-result">
          ${guidedMetricCard(t('Run type', '运行类型'), result.run_type || 'preflight')}
          ${guidedMetricCard(t('Reportable', '可报告'), result.reportable ? 'true' : 'false', t('Draft remains locked until sign-off.', '签署前草稿保持锁定。'))}
          ${guidedMetricCard(t('Evidence check', '证据核验'), gate.status || 'analysis_only', gate.reason || '')}
          ${guidedMetricCard(t('Artifacts', 'Artifacts'), fmtInt(artCount), result.project_dir ? compactPath(result.project_dir) : '')}
        </div>
        <div class="note info" style="margin-top:10px;padding:9px 11px;"><div class="ico">${icon('shield', 13)}</div><div class="body"><div class="d" style="font-size:10.5px;margin:0;">${t('This was a local, no-cost preflight (mock provider — no external model call): it checks coverage and the evidence contract, but is not a reportable run. Provider and model selection stay in Guided Copilot; Project Monitor only reviews the resulting run, artifacts, and evidence. External-provider use is always explicit opt-in.', '这是一次本地零成本预检（mock provider —— 不调用外部模型）：它检查覆盖率与证据合约，但不是可报告运行。provider 和模型选择仍在研究引导中完成；项目监控只审阅生成的运行、artifact 和证据。外部 provider 始终需要显式授权。')}</div></div></div>` : ''}
        <div class="gdx-actions">
          <button type="button" class="btn ${guidedAgent.result ? '' : 'primary'}" data-ga-run ${!src || guidedAgent.running ? 'disabled' : ''}>${icon('play', 13)} ${guidedAgent.result ? t('Re-run preflight', '重跑预检') : t('Start local preflight', '启动本地预检')}</button>
          <button type="button" class="btn" data-guided-goal="data_extraction">${t('Prepare/register data', '准备/注册数据')}</button>
          ${guidedAgent.result ? `<button type="button" class="btn primary" data-ga-open-agent>${icon('arrow', 13)} ${t('Open Project Monitor', '打开项目监控')}</button>` : ''}
        </div>
      </div>`;
  }
  function runGuidedAgentPreflight() {
    if (!guidedAgent || guidedAgent.running) return;
    const src = activeExportSource();
    if (!src) {
      guidedAgent.error = 'No active registered export is selected.';
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.startAgentRun || !window.EventSource) {
      guidedAgent.error = 'Agent run backend or event stream is unavailable.';
      renderThread();
      return;
    }
    guidedAgent.running = true;
    guidedAgent.error = null;
    guidedAgent.result = null;
    guidedAgent.progress = null;
    guidedAgent.warning = null;
    renderThread();
    const runState = guidedAgent;
    const capturedQuestion = runState.question || 'Evaluate the active ICU cohort with an evidence-bound local preflight.';
    const studyId = slugifyDraftFolder(capturedQuestion.slice(0, 80)) || 'guided-agent-preflight';
    let runToken = guidedRunChannel.start({
      surface: 'guided-agent-card',
      study_id: studyId,
      context_id: window.EU_GUIDED_STUDY_CONTEXT && window.EU_GUIDED_STUDY_CONTEXT.activeId ? window.EU_GUIDED_STUDY_CONTEXT.activeId() : '',
      question: capturedQuestion,
      source_path: src.path,
      study_mode: 'analysis',
      run_type: 'preflight',
      provider: 'mock',
    });
    const contextReady = window.EU_GUIDED_STUDY_CONTEXT && window.EU_GUIDED_STUDY_CONTEXT.persistForRun
      ? window.EU_GUIDED_STUDY_CONTEXT.persistForRun('agent_preflight')
      : Promise.resolve(null);
    contextReady.then(studyContext => {
      runToken = guidedRunChannel.bind(runToken, {
        context_id: studyContext && studyContext.id,
        study_id: (studyContext && studyContext.id) || studyId,
      });
      if (guidedRunChannel.isCurrent(runToken) && guidedAgent === runState) guidedAgent.contextId = runToken.context_id || null;
      return window.EU_API.startAgentRun({
        path: runToken.source_path,
        study_id: runToken.study_id,
        study_context_id: runToken.context_id || undefined,
        mode: runToken.study_mode,
        run_type: runToken.run_type,
        llm_provider: runToken.provider,
        external_llm_opt_in: false,
        question: runToken.question,
      });
    }).then(r => {
      runToken = guidedRunChannel.bind(runToken, { job_id: r.job_id, context_revision: r.study_context_revision });
      const isCurrent = guidedRunChannel.isCurrent(runToken) && guidedAgent === runState;
      if (isCurrent) {
        guidedAgent.jobId = runToken.job_id;
        guidedAgent.warning = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.submissionWarning
          ? window.EU_AGENT_STUDY_CONTEXT.submissionWarning(r)
          : null;
      }
      if (runToken.context_id && window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.markContextRunning) {
        window.EU_AGENT_STUDY_CONTEXT.markContextRunning(runToken.context_id, runToken.job_id, runToken.context_revision);
      }
      if (isCurrent) {
        renderThread();
        scheduleGuidedSlotSave('start_agent_preflight');
      }
      const es = new EventSource('/api/jobs/' + encodeURIComponent(runToken.job_id) + '/events');
      if (isCurrent) guidedRunStream = es;
      let ended = false;
      es.onmessage = ev => {
        let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
        if (m.type === 'progress') {
          if (!guidedRunChannel.isCurrent(runToken) || guidedAgent !== runState) return;
          guidedAgent.progress = m;
          renderThread();
          return;
        }
        if (m.type === 'end') {
          ended = true;
          try { es.close(); } catch (e) {}
          if (runToken.context_id && window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.markContextFinished) {
            window.EU_AGENT_STUDY_CONTEXT.markContextFinished(
              runToken.context_id,
              m.status,
              m.result || null,
              runToken.job_id,
              m.study_context_revision || (m.result && m.result.study_context_revision),
            );
          }
          if (!guidedRunChannel.isCurrent(runToken) || guidedAgent !== runState) return;
          if (guidedRunStream === es) guidedRunStream = null;
          guidedAgent.running = false;
          if (m.status === 'done') {
            guidedAgent.result = m.result || {};
            liveAgentRun = { result: guidedAgent.result };
            if (guidedGateState(guidedAgent.result).blocked) {
              setVal({ analysis: 'verification blocked', draft: 'locked · review_blocked' });
              markThrough('analysis', 'locked');
              setStudy({ draft: 'locked' });
            } else {
              setVal({ analysis: 'preflight complete', draft: 'locked' });
              markThrough('draft', 'locked');
            }
          } else {
            guidedAgent.error = guidedJobEndError(m) || 'Agent preflight failed.';
          }
          guidedRunChannel.clear(runToken);
          renderThread();
          renderAside();
          scheduleGuidedSlotSave('finish_agent_preflight');
        }
      };
      es.onerror = () => {
        if (ended) return;
        try { es.close(); } catch (e) {}
        if (!guidedRunChannel.isCurrent(runToken) || guidedAgent !== runState) return;
        if (guidedRunStream === es) guidedRunStream = null;
        guidedAgent.running = false;
        guidedAgent.error = 'Agent event stream stopped before completion.';
        guidedRunChannel.clear(runToken);
        renderThread();
        scheduleGuidedSlotSave('agent_event_stream_error');
      };
    }).catch(err => {
      if (!guidedRunChannel.isCurrent(runToken) || guidedAgent !== runState) return;
      guidedAgent.running = false;
      guidedAgent.error = err.message || String(err);
      guidedRunChannel.clear(runToken);
      renderThread();
      scheduleGuidedSlotSave('start_agent_preflight_error');
    });
  }

  /* ============== inline Idea Mining ============== */
  /* ============== local concept definition answers ============== */
  const CONCEPT_ALIASES = [
    ['sofa2', ['sofa2', 'sofa-2', 'sofa 2', 'SOFA-2', 'SOFA2']],
    ['sofa', ['sofa1', 'sofa-1', 'sofa 1', 'traditional sofa']],
    ['sep3_sofa2', ['sepsis3 sofa2', 'sepsis-3 sofa-2', 'sepsis 3 sofa 2', 'sepsis3']],
    ['lact', ['lactate', '乳酸']],
  ];
  function findLocalConceptQuery(text) {
    const raw = String(text || '');
    const lower = raw.toLowerCase();
    const asks = /定义|怎么定义|是什么|解释|怎么算|如何计算|definition|define|what is|how.*defined/.test(raw + ' ' + lower);
    if (!asks) return null;
    for (const [code, aliases] of CONCEPT_ALIASES) {
      if (aliases.some(a => lower.includes(String(a).toLowerCase()))) return code;
    }
    const dict = (window.EU_CATALOG && window.EU_CATALOG.dict) || {};
    for (const [code, row] of Object.entries(dict)) {
      const fields = [code].concat(Array.isArray(row) ? row : Object.values(row || {})).join(' ').toLowerCase();
      if (code.length > 2 && lower.includes(code.toLowerCase())) return code;
      const name = Array.isArray(row) ? row[0] : (row && (row.name || row.label));
      if (name && lower.includes(String(name).toLowerCase())) return code;
    }
    return null;
  }
  function conceptRowsForAnswer(code) {
    const cat = window.EU_CATALOG || {};
    const row = (cat.dict || {})[code] || [];
    const desc = (cat.desc || {})[code] || [];
    const meta = (cat.cov || {})[code] || {};
    const active = cat.activeExportCoverage && cat.activeExportCoverage.concepts && cat.activeExportCoverage.concepts[code];
    const groups = [];
    Object.entries(cat.groupConcepts || {}).forEach(([group, members]) => {
      if (Array.isArray(members) && members.includes(code)) groups.push(group);
    });
    return {
      code,
      name: Array.isArray(row) ? row[0] : (row.name || code),
      nameZh: Array.isArray(row) ? row[1] : (row.name_zh || row.zh || ''),
      unit: Array.isArray(row) ? row[2] : (row.unit || ''),
      desc: Array.isArray(desc) ? desc[0] : (desc.en || desc.description || ''),
      descZh: Array.isArray(desc) ? desc[1] : (desc.zh || ''),
      basis: meta.basis || meta.kind || 'EasyICU concept catalog',
      databases: meta.databases,
      group: groups.join(', ') || 'catalog',
      active,
    };
  }
  function answerConceptQuestion(text, code) {
    pushUser(text);
    const info = conceptRowsForAnswer(code);
    const dbLine = info.databases != null
      ? `${info.databases}/${((window.EU_CATALOG && window.EU_CATALOG.supportedDbs) || []).length || 6} databases`
      : info.basis;
    const activeLine = info.active
      ? `${fmtPct(info.active.coverage_pct)} active-export coverage · ${fmtInt(info.active.observed_entities, 'n/a')} entities`
      : t('not present in the active export coverage summary', '当前 active export 覆盖统计里没有这个字段');
    thread.push({ bot: true, html: `
      <div class="gd-concept-answer">
        <div class="gca-head">
          <span>${icon('book', 14)}</span>
          <div><strong>${esc(info.name)}${info.nameZh ? ` · ${esc(info.nameZh)}` : ''}</strong><small>${esc(info.code)} · ${esc(info.group)}</small></div>
        </div>
        <div class="gca-grid">
          <span>${t('Unit', '单位')}</span><strong>${esc(info.unit || 'n/a')}</strong>
          <span>${t('Definition', '定义')}</span><p>${esc(t(info.desc || 'No dictionary definition is available.', info.descZh || info.desc || '字典中暂无定义。'))}</p>
          <span>${t('Dictionary coverage', '字典覆盖')}</span><strong>${esc(dbLine)}</strong>
          <span>${t('Active export', '当前导出')}</span><strong>${esc(activeLine)}</strong>
        </div>
        <div class="gca-note">${t('This answer is local and code-backed: it reads EasyICU concept_catalog through /api/catalog. It does not call an external model or literature search.', '这个回答来自本地代码字典：通过 /api/catalog 读取 EasyICU concept_catalog。没有调用外部模型或文献搜索。')}</div>
        <div class="gca-actions"><button class="btn sm" data-open="dictionary">${t('Open Data Dictionary', '打开数据字典')}</button></div>
      </div>` });
    renderThread();
  }
  function isGuidedExtractionIntent(text) {
    const s = String(text || '').toLowerCase();
    return /extract|export|prepare data|data extraction|抽取|提取|导出|准备数据|生成数据/.test(s);
  }
  function isGuidedReviewIntent(text) {
    const s = String(text || '').toLowerCase();
    return /review|patient|cohort|km|kaplan|logrank|visual|visualiz|可视化|审阅|查看|队列|患者|生存|曲线/.test(s);
  }
  function isGuidedAgentIntent(text) {
    const s = String(text || '').toLowerCase();
    return /agent|analysis|run project|research project|preflight|manuscript|draft|跑研究|运行研究|分析|预检|草稿/.test(s);
  }

  /* ============== DOM render ============== */
  function renderThread() {
    const host = document.getElementById('gdThread');
    if (!host) return;
    host.innerHTML = thread.map(t => {
      if (t.typing) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble"><div class="typing"><span></span><span></span><span></span></div></div></div></div>`;
      if (t.guidedExtraction) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${EXTRACT.renderCard()}</div></div>`;
      if (t.guidedReview) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${REVIEW.renderCard()}</div></div>`;
      if (t.guidedAgent) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${renderGuidedAgentCard()}</div></div>`;
      if (t.guidedIdeaApiSetup) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${IDEA.renderGuidedIdeaApiSetupCard()}</div></div>`;
      if (t.guidedIdea) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${IDEA.renderGuidedIdeaCard()}</div></div>`;
      if (t.diff) return diffCard();
      if (t.once) return ONCE[t.once] ? ONCE[t.once]() : '';
      if (t.card) {
        if (t.step === expandedStep) return CARD[t.step] ? CARD[t.step]() : '';
        const s = summaryOf(t.step);
        return `<div class="gd-collapsed"><span class="cc-mk">${icon('check', 10, 3)}</span><span class="cc-t">${esc(s.t)}</span><span class="cc-v">${esc(s.v)}</span>${s.edit ? `<button class="cc-edit" data-edit="${t.step}">${icon('sliders', 11)} Edit</button>` : ''}</div>`;
      }
      if (t.bot) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble">${htmlOf(t.html)}</div></div></div>`;
      return `<div class="msg user"><div class="m-ava">LK</div><div class="m-body"><div class="m-bubble">${t.html}</div></div></div>`;
    }).join('');
    scrollEnd();
  }
  function renderChips() {
    const host = document.getElementById('gdSuggest');
    if (!host) return;
    host.innerHTML = chips.map(([label, next, cls]) => `<button class="suggest-chip ${cls || ''}" data-go="${next}">${htmlOf(label)}</button>`).join('');
    host.style.display = chips.length ? 'flex' : 'none';
  }
  function renderAside() {
    const host = document.getElementById('gdAsideBody');
    if (!host || piProjectShellActive()) return; // Copilot owns this panel while mounted
    host.innerHTML = renderStudyPipelineSummary() + renderStudyItemList() + renderOutputs(host);
  }
  function normalizedStudyRows() {
    const gi = goalIdx();
    return STUDY.map(([id, label, ico, labelZh], idx) => {
      let stt = studyStatus[id] || 'pending';
      // steps past the chosen finish line are optional — dim them unless already reached
      if (idx > gi && (stt === 'pending')) stt = 'beyond';
      let v = studyVal[id]; if (typeof v === 'function') v = v();
      return { id, label, ico, labelZh, idx, stt, v };
    });
  }
  function renderStudyPipelineSummary() {
    const rows = normalizedStudyRows();
    let activeIdx = rows.findIndex(r => r.stt === 'active');
    if (activeIdx < 0) activeIdx = rows.findIndex(r => r.stt !== 'done' && r.stt !== 'beyond');
    if (activeIdx < 0) activeIdx = 0;
    const active = rows[activeIdx] || rows[0];
    const next = rows.slice(activeIdx + 1).find(r => r.stt !== 'beyond');
    const done = rows.filter(r => r.stt === 'done').length;
    const total = Math.max(1, Math.min(goalIdx() + 1, STUDY.length));
    const pct = Math.max(0, Math.min(100, Math.round(done / total * 100)));
    const currentValue = active && active.v ? `<div class="gd-pipeline-value">${esc(active.v)}</div>` : '';
    const nextLine = next
      ? `<span>${t('Next', '下一步')}</span><strong>${t(next.label, next.labelZh || next.label)}</strong>`
      : `<span>${t('Next', '下一步')}</span><strong>${t('Ready for sign-off', '等待核验')}</strong>`;
    return `
      <div class="gd-pipeline-summary" data-gd-pipeline-summary>
        <div class="gd-pipeline-summary-head">
          <div>
            <div class="eyebrow">${t('Step overview', '步骤总览')}</div>
            <strong>${t(active.label, active.labelZh || active.label)}</strong>
            ${currentValue}
          </div>
          <button class="gd-pipeline-toggle" type="button" data-gd-pipeline-toggle aria-controls="gdPipelineList" aria-expanded="${guidedPipelineOpen ? 'true' : 'false'}">
            ${guidedPipelineOpen ? t('Hide steps', '收起步骤') : t('Show all steps', '展开步骤')}
          </button>
        </div>
        <div class="gd-pipeline-bar" aria-label="${t('Guided Copilot progress', '研究引导进度')}"><span style="width:${pct}%;"></span></div>
        <div class="gd-pipeline-meta">
          <span><strong>${done}/${total}</strong> ${t('required steps done', '个必需步骤完成')}</span>
          <span>${t('Goal', '目标')} · ${DEPTH[depth].label}</span>
        </div>
        <div class="gd-pipeline-next">${nextLine}</div>
      </div>`;
  }
  function renderStudyItemList() {
    const gi = goalIdx();
    return `<div class="gd-pipeline-list ${guidedPipelineOpen ? 'open' : 'collapsed'}" id="gdPipelineList" ${guidedPipelineOpen ? '' : 'hidden'} data-gd-pipeline-list>` + normalizedStudyRows().map(({ id, label, ico, labelZh, idx, stt, v }) => {
      const dot = stt === 'done' ? icon('check', 11, 3) : stt === 'locked' ? icon('lock', 10) : icon(ico, 12);
      const badge = stt === 'active' ? '<span class="si-state"><span class="spin sm" style="width:11px;height:11px;"></span></span>'
        : stt === 'locked' ? `<span class="si-state pill warn" style="height:18px;"><span class="dot"></span></span>`
        : stt === 'beyond' ? `<span class="si-state si-opt">${t('optional', '可选')}</span>` : '';
      const clickable = thread.some(t => t.card && t.step === id);
      const row = `<div class="study-item ${stt}${clickable ? ' nav' : ''}" ${clickable ? `data-study="${id}" role="button" tabindex="0"` : ''}><span class="si-dot">${dot}</span><div class="si-txt"><div class="si-t">${t(label, labelZh || label)}</div>${v ? `<div class="si-v">${esc(v)}</div>` : ''}</div>${badge}</div>`;
      // draw the finish line right after the goal step (only when stopping short of the full study)
      const fin = (idx === gi && depth !== 'full')
        ? `<div class="study-finishline"><span class="fl-flag">${icon('check', 10, 3)}</span><span class="fl-t">${t('Finish line', '终点线')} · ${DEPTH[depth].label}</span></div>`
        : '';
      return row + fin;
    }).join('') + '</div>';
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
      rememberUserQuestion(text);
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
      return () => { pushUser(text); if (currentId !== 'toCohort' && STEP_INDEX[expandedStep] > STEP_INDEX.cohort) editStep('cohort'); else { renderThread(); renderAside(); pushBot(
        `Set to <strong>${n}</strong> demo stays. ${currentId === 'toCohort' ? 'Use this cohort when ready.' : ''}`,
        `已设为 <strong>${n}</strong> 个 demo stays。${currentId === 'toCohort' ? '准备好后就使用这个队列。' : ''}`,
      ); renderThread(); } };
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

  function localDraftRows() {
    const data = guidedDrafts && guidedDrafts.data;
    return data && Array.isArray(data.drafts) ? data.drafts : [];
  }
  function loadGuidedDrafts(force) {
    if (!window.EU_API || !window.EU_API.loadGuidedDrafts) return Promise.resolve(null);
    if (!force && (guidedDrafts.loading || guidedDrafts.data || guidedDrafts.error)) {
      return Promise.resolve(guidedDrafts.data);
    }
    guidedDrafts = { loading: true, error: null, data: guidedDrafts.data || null };
    renderSessions();
    return window.EU_API.loadGuidedDrafts({ limit: 100 }).then(async data => {
      guidedDrafts = { loading: false, error: null, data: data };
      renderSessions();
      if (guidedFolderDialogMode) renderGuidedFolderDialog();
      const continuity = window.EU_GUIDED_PROJECT_CONTINUITY;
      const piProjectOwner = window.EU_GUIDED_PI_PROJECT;
      const requestedId = !selectedGuidedDraft && piProjectOwner
        && typeof piProjectOwner.requestedProjectId === 'function'
        ? piProjectOwner.requestedProjectId()
        : '';
      const rememberedId = requestedId || (
        !selectedGuidedDraft && continuity && continuity.remembered
          ? continuity.remembered()
          : ''
      );
      if (rememberedId) {
        const rememberedRow = localDraftRows().find(row => row && row.id === rememberedId);
        if (rememberedRow) {
          selectedGuidedDraft = rememberedRow;
          selectedGuidedRun = null;
          await openGuidedProjectMemory(rememberedRow, null, 'draft');
        } else if (continuity.forget) {
          continuity.forget(rememberedId);
        }
      }
    }).catch(err => {
      guidedDrafts = { loading: false, error: err.message || String(err), data: null };
      renderSessions();
      if (guidedFolderDialogMode) renderGuidedFolderDialog();
      return null;
    });
  }
  document.addEventListener('easyicu:guided-projects-refresh', () => {
    loadGuidedDrafts(true);
  });
  function renderGuidedDraftRemovalDialog() {
    guidedProjectRenderer('renderDraftRemovalDialog');
  }
  function rerenderProjectRailKeepingScroll() {
    const before = document.getElementById('gdSessions');
    const top = before ? before.scrollTop : 0;
    renderSessions();
    const after = document.getElementById('gdSessions');
    if (after) after.scrollTop = top;
  }
  function closeGuidedDraftRemovalDialog() {
    if (guidedDraftRemoval && guidedDraftRemoval.busy) return;
    guidedDraftRemoval = null;
    renderGuidedDraftRemovalDialog();
  }
  function removeLocalGuidedDraft(rowOrRows) {
    const rows = (Array.isArray(rowOrRows) ? rowOrRows : [rowOrRows]).filter(row => row && row.id);
    if (!rows.length || !window.EU_API || !window.EU_API.removeGuidedDraft) return;
    guidedDraftRemoval = { row: rows[0], rows, trashProjectFolder: false, busy: false, error: null };
    renderGuidedDraftRemovalDialog();
  }
  async function confirmLocalGuidedDraftRemoval() {
    if (!guidedDraftRemoval || guidedDraftRemoval.busy) return;
    const rows = Array.isArray(guidedDraftRemoval.rows) && guidedDraftRemoval.rows.length
      ? guidedDraftRemoval.rows
      : [guidedDraftRemoval.row].filter(Boolean);
    if (!rows.length || !window.EU_API || !window.EU_API.removeGuidedDraft) return;
    const trashProjectFolder = !!guidedDraftRemoval.trashProjectFolder;
    guidedDraftRemoval.busy = true;
    guidedDraftRemoval.error = null;
    renderGuidedDraftRemovalDialog();
    let completed = 0;
    try {
      for (const row of rows) {
        const result = await window.EU_API.removeGuidedDraft({
          draft_id: row.id,
          project_dir: row.project_dir,
          delete_project_folder: false,
          trash_project_folder: trashProjectFolder,
          trash_confirmation: trashProjectFolder ? row.id : null,
        });
        if (!result || result.ok === false) {
          throw new Error((result && (result.reason || result.error)) || 'remove_failed');
        }
        if (trashProjectFolder && !result.project_folder_trashed) {
          throw new Error('project_folder_trash_not_confirmed');
        }
        completed += 1;
        if (selectedGuidedDraft && selectedGuidedDraft.id === row.id) {
          if (window.EU_GUIDED_PROJECT_CONTINUITY) window.EU_GUIDED_PROJECT_CONTINUITY.forget(row.id);
          selectedGuidedDraft = null;
          if (window.EU_GUIDED_PI && window.EU_GUIDED_PI.bindProject) window.EU_GUIDED_PI.bindProject(null);
        }
      }
      if (window.EU_GUIDED_PROJECTS && window.EU_GUIDED_PROJECTS.setProjectManagement) {
        window.EU_GUIDED_PROJECTS.setProjectManagement(false);
      }
      guidedDraftRemoval = null;
      renderGuidedDraftRemovalDialog();
      loadGuidedDrafts(true);
      pushBot(
        trashProjectFolder
          ? `Removed <strong>${completed}</strong> project${completed === 1 ? '' : 's'} from EasyICU and moved the local project folder${completed === 1 ? '' : 's'} to the system trash.`
          : `Removed <strong>${completed}</strong> project${completed === 1 ? '' : 's'} from the Guided project list. Local project folders were left untouched.`,
        trashProjectFolder
          ? `已从 EasyICU 移除 <strong>${completed}</strong> 个项目，并将其本地项目文件夹移到系统废纸篓。`
          : `已从研究项目列表移除 <strong>${completed}</strong> 个项目。磁盘上的项目文件夹没有改动。`,
      );
      renderSessions();
      renderThread();
    } catch (err) {
      if (!guidedDraftRemoval) return;
      guidedDraftRemoval.busy = false;
      guidedDraftRemoval.error = completed
        ? t(
          `${completed} of ${rows.length} projects were processed before the operation stopped. ${err.message || String(err)}`,
          `操作停止前已处理 ${rows.length} 个项目中的 ${completed} 个。${err.message || String(err)}`,
        )
        : (err.message || String(err));
      renderGuidedDraftRemovalDialog();
      loadGuidedDrafts(true);
    }
  }
  function guidedBackendContext() {
    const src = activeExportSource();
    const sum = (src && src.summary) || {};
    return {
      route: 'guided',
      data_mode: dataMode || 'demo',
      language: window.EU_LANG || 'en',
      selected_source: src ? {
        id: src.id,
        label: src.label || src.database || 'active export',
        database: src.database,
        path: src.path,
      } : null,
      summary: {
        stays: sum.stays,
        modules: sum.modules,
        database: src && src.database,
        label: src && src.label,
      },
    };
  }
  function guidedSessionId() {
    return guidedCopilot && guidedCopilot.session && guidedCopilot.session.id;
  }
  function guidedActiveFlow() {
    if (IDEA.state()) return 'idea_mining';
    if (EXTRACT.state()) return 'data_extraction';
    if (REVIEW.state()) return 'review_data';
    if (guidedAgent) return 'run_agent';
    return null;
  }
  function boundedScan(scan) {
    if (!scan) return null;
    return {
      ok: !!scan.ok,
      ready: !!scan.ready,
      source: scan.source || null,
      db: scan.db || null,
      db_key: scan.db_key || null,
      path: scan.path || null,
      tables: scan.tables == null ? null : Number(scan.tables),
      modules: scan.modules == null ? null : Number(scan.modules),
      error: scan.error || null,
      reason: scan.reason || null,
    };
  }
  function guidedSlotSnapshot() {
    const guidedExtract = EXTRACT.state();
    const guidedDesign = EXTRACT.design();
    const src = activeExportSource();
    const idea = IDEA.state() && IDEA.guidedIdeaSelected ? IDEA.guidedIdeaSelected() : null;
    return {
      active_flow: guidedActiveFlow(),
      data_mode: dataMode || 'demo',
      branch,
      depth,
      study_params: {
        outcome: studyParams && studyParams.outcome,
        window: studyParams && studyParams.window,
        exposure: studyParams && studyParams.exposure,
        scope: studyParams && studyParams.scope,
      },
      study_design: guidedDesign ? {
        outcome: guidedDesign.outcome || '',
        outcome_custom: guidedDesign.outcomeCustom || '',
        window: guidedDesign.window || 'whole_stay',
        comparator: guidedDesign.comparator || 'none',
        comparator_custom: guidedDesign.comparatorCustom || '',
        collected: !!guidedDesign.collected,
        outcome_label: guidedDesignOutcome(),
        comparator_label: guidedDesignComparator(),
      } : null,
      active_export: src ? {
        label: src.label || src.database || 'active export',
        database: src.database || null,
        path: src.path || null,
        modules: src.summary && src.summary.modules,
        stays: src.summary && src.summary.stays,
      } : null,
      extraction: guidedExtract ? {
        path: guidedExtract.path || '',
        step: guidedExtract.step || 'source',
        cohort: guidedExtract.cohort || 'adult_first',
        modules: guidedExtract.modules || [],
        format: guidedExtract.format || 'parquet',
        export_dir: guidedExtract.exportDir || '',
        merge: !!guidedExtract.merge,
        max_patients: guidedExtract.maxPatients == null ? null : Number(guidedExtract.maxPatients),
        scan: boundedScan(guidedExtract.scan),
        registered: !!guidedExtract.registered,
        result: guidedExtract.result ? {
          out_dir: guidedExtract.result.out_dir || guidedExtract.result.path || '',
          total_rows: guidedExtract.result.total_rows == null ? null : Number(guidedExtract.result.total_rows),
          files_written: guidedExtract.result.files_written == null ? guidedExtract.result.files : guidedExtract.result.files_written,
        } : null,
      } : null,
      review: REVIEW.slotSnapshot(),
      agent: guidedAgent ? {
        question: guidedAgent.question || '',
        job_id: guidedAgent.jobId || null,
        result: guidedAgent.result ? {
          project_dir: guidedAgent.result.project_dir || '',
          run_type: guidedAgent.result.run_type || '',
          reportable: !!guidedAgent.result.reportable,
          gate_status: guidedAgent.result.gate && guidedAgent.result.gate.status,
        } : null,
      } : null,
      idea: IDEA.slotSnapshot(),
    };
  }
  function saveGuidedSlotsNow(reason) {
    if (!window.EU_API || !window.EU_API.saveGuidedSlots) return Promise.resolve(null);
    const sessionPromise = guidedSessionId() && guidedCopilot.session && guidedCopilot.session.memory_scope === 'project_folder'
      ? Promise.resolve(guidedCopilot.session)
      : ensureGuidedSession();
    return sessionPromise.then(session => {
      if (!session || session.memory_scope !== 'project_folder') return null;
      const flow = guidedActiveFlow();
      return window.EU_API.saveGuidedSlots({
        session_id: session.id,
        goal: flow || undefined,
        step: flow ? `${flow}_configuration` : 'choose_goal',
        context: guidedBackendContext(),
        slots: Object.assign({ save_reason: reason || 'state_change' }, guidedSlotSnapshot()),
      }).then(result => {
        if (result && result.session) guidedCopilot.session = result.session;
        return result;
      });
    }).catch(err => {
      console.warn('[EasyICU] Guided slot save failed:', err);
      return null;
    });
  }
  function scheduleGuidedSlotSave(reason) {
    clearTimeout(guidedSlotSaveTimer);
    guidedSlotSaveTimer = setTimeout(() => { saveGuidedSlotsNow(reason); }, 350);
  }
  function captureGuidedComposerDraft() {
    const input = document.getElementById('gdInput');
    if (input) guidedComposerDraft = input.value || '';
  }
  function flushGuidedSlotSave(reason) {
    clearTimeout(guidedSlotSaveTimer);
    guidedSlotSaveTimer = null;
    return saveGuidedSlotsNow(reason);
  }
  window.__euGuidedBeforeLanguageRerender = function () {
    if (!guidedMounted) return Promise.resolve(null);
    captureGuidedComposerDraft();
    return flushGuidedSlotSave('language_change');
  };
  function restoreGuidedSlotsFromSession(session) {
    const slots = session && session.slots && typeof session.slots === 'object' ? session.slots : {};
    if (slots.study_params && typeof slots.study_params === 'object') {
      studyParams = Object.assign({}, studyParams || {}, slots.study_params);
    }
    if (slots.study_design && typeof slots.study_design === 'object') {
      const guidedDesign = resetGuidedDesignState();
      guidedDesign.outcome = slots.study_design.outcome || '';
      guidedDesign.outcomeCustom = slots.study_design.outcome_custom || '';
      guidedDesign.window = slots.study_design.window || 'whole_stay';
      guidedDesign.comparator = slots.study_design.comparator || 'none';
      guidedDesign.comparatorCustom = slots.study_design.comparator_custom || '';
      guidedDesign.collected = !!slots.study_design.collected;
    }
    const active = slots.active_flow || session && session.goal;
    if (slots.extraction && typeof slots.extraction === 'object') {
      const guidedExtract = resetGuidedExtractionState();
      guidedExtract.step = slots.extraction.step || 'source';
      guidedExtract.path = slots.extraction.path || '';
      guidedExtract.cohort = slots.extraction.cohort || guidedExtract.cohort;
      guidedExtract.modules = Array.isArray(slots.extraction.modules) ? slots.extraction.modules.slice() : guidedExtract.modules;
      guidedExtract.format = slots.extraction.format || guidedExtract.format;
      guidedExtract.exportDir = slots.extraction.export_dir || '';
      guidedExtract.merge = !!slots.extraction.merge;
      guidedExtract.maxPatients = slots.extraction.max_patients == null ? null : Number(slots.extraction.max_patients);
      guidedExtract.scan = slots.extraction.scan || null;
      guidedExtract.registered = !!slots.extraction.registered;
      guidedExtract.result = slots.extraction.result || null;
    }
    IDEA.restoreSlot(slots.idea);
    if (slots.agent && typeof slots.agent === 'object') {
      resetGuidedAgentState();
      guidedAgent.question = slots.agent.question || guidedAgent.question;
      guidedAgent.jobId = slots.agent.job_id || null;
      guidedAgent.result = slots.agent.result || null;
    }
    REVIEW.restoreSlot(slots.review);
    return active || null;
  }
  function hasGuidedProjectMemory() {
    const session = guidedCopilot && guidedCopilot.session;
    if (session && session.project_dir && session.memory_scope === 'project_folder') return true;
    if (selectedGuidedDraft && selectedGuidedDraft.project_dir) return true;
    return false;
  }
  function rememberPendingGoal(goal, label) {
    pendingGuidedGoal = goal ? { goal, label: label || (guidedGoalMeta(goal) && guidedGoalMeta(goal).label_en) || goal } : null;
  }
  function requireGuidedProjectMemory(goal, label) {
    if (goal === 'review_data' && activeExportSource()) return false;
    if (hasGuidedProjectMemory()) return false;
    rememberPendingGoal(goal, label);
    if (label) pushUser(label);
    if (goal === 'review_data') {
      thread.push({ bot: true, html: bi(
        `Choose a <strong>local EasyICU export folder</strong> first. I’ll register it as the active export and review it inside Copilot; project memory is optional for this read-only review.`,
        `请先选择一个<strong>本地 EasyICU export 文件夹</strong>。我会把它注册为 active export，并直接在 Copilot 内审阅；这个只读审阅不强制要求项目记忆。`,
      ) });
      showGuidedDraftSetup(label || 'Review extracted data', 'open');
      chips = [['Use active export', '@activeExport'], ['Choose export folder', '@folderopen']];
      renderThread();
      renderChips();
      return true;
    }
    thread.push({ bot: true, html: bi(
      `One quick setup step: this goal saves into a <strong>local study folder</strong> (its own conversation + memory). I can create a starter folder for you in one click, or you can pick a custom path.`,
      `只差一步快速设置：这个目标会保存到一个<strong>本地研究文件夹</strong>（独立的对话 + 记忆）。我可以一键为你创建入门文件夹，你也可以选择自定义路径。`,
    ) });
    chips = [
      [t('Create a starter folder & continue', '一键创建入门文件夹并继续'), '@folderquick'],
      [t('Pick a custom folder', '选择自定义文件夹'), '@foldernew'],
    ];
    renderThread();
    renderChips();
    return true;
  }
  function bindGuidedDraftMemory(draft, blankProject) {
    if (!draft || !draft.project_dir || !window.EU_API || !window.EU_API.openGuidedProject) {
      return Promise.resolve(null);
    }
    return window.EU_API.openGuidedProject({
      project_dir: draft.project_dir,
      draft_id: draft.id || null,
      title: draft.title || 'local study',
      mode: 'local',
      context: blankProject ? blankGuidedBackendContext() : guidedBackendContext(),
    }).then(result => {
      if (result && result.ok) {
        guidedCopilot = { loading: false, error: null, session: result.session || null, last: result };
      }
      return result;
    });
  }
  function piProjectShellActive() {
    return !!(
      window.EU_GUIDED_PI &&
      window.EU_GUIDED_PI.isActive &&
      window.EU_GUIDED_PI.isActive()
    );
  }
  function bindProjectToPi(result, row) {
    const session = result && result.session ? result.session : null;
    guidedCopilot = { loading: false, error: null, session, last: result || guidedCopilot.last };
    restoreGuidedSlotsFromSession(session);
    const projectId = String((session && session.draft_id) || (row && row.id) || '').trim();
    const title = projectTitle(
      (session && session.project_title) || (row && row.title),
      (row && (row.question || row.study_id || row.run_label)) || projectId,
    );
    let binding = null;
    if (projectId && window.EU_GUIDED_PI && window.EU_GUIDED_PI.bindProject) {
      if (window.EU_GUIDED_PROJECT_CONTINUITY) {
        window.EU_GUIDED_PROJECT_CONTINUITY.remember(projectId);
      }
      binding = window.EU_GUIDED_PI.bindProject({ id: projectId, title });
    }
    renderAside();
    renderSessions();
    return binding;
  }
  function continuePendingGuidedGoal() {
    if (!pendingGuidedGoal) return false;
    const pending = pendingGuidedGoal;
    pendingGuidedGoal = null;
    const meta = guidedGoalMeta(pending.goal);
    thread.push({ bot: true, html: bi(
      `Folder memory is ready. Continuing with <strong>${esc(meta.label_en || pending.label || pending.goal)}</strong> inside this project context.`,
      `文件夹记忆已就绪。现在会在这个项目上下文里继续<strong>${esc(meta.label_zh || pending.label || pending.goal)}</strong>。`,
    ) });
    chooseGuidedGoal(pending.goal, null);
    return true;
  }
  function ensureGuidedSession(force) {
    if (!window.EU_API || !window.EU_API.createGuidedSession) return Promise.resolve(null);
    if (
      !force &&
      guidedCopilot.session &&
      guidedCopilot.session.memory_scope === 'project_folder' &&
      (!selectedGuidedDraft || guidedCopilot.session.project_dir === selectedGuidedDraft.project_dir)
    ) return Promise.resolve(guidedCopilot.session);
    if (selectedGuidedDraft && selectedGuidedDraft.project_dir && window.EU_API.openGuidedProject) {
      guidedCopilot = { loading: true, error: null, session: guidedCopilot.session, last: guidedCopilot.last };
      return window.EU_API.openGuidedProject({
        project_dir: selectedGuidedDraft.project_dir,
        draft_id: selectedGuidedDraft.id,
        title: selectedGuidedDraft.title,
        mode: 'local',
        context: guidedBackendContext(),
      }).then(data => {
        guidedCopilot = { loading: false, error: null, session: data.session || null, last: data };
        return guidedCopilot.session;
      }).catch(err => {
        guidedCopilot = { loading: false, error: err.message || String(err), session: null, last: null };
        renderThread();
        return null;
      });
    }
    if (!force) return Promise.resolve(null);
    guidedCopilot = { loading: true, error: null, session: guidedCopilot.session, last: guidedCopilot.last };
    return window.EU_API.createGuidedSession({
      mode: 'local',
      context: guidedBackendContext(),
    }).then(data => {
      guidedCopilot = { loading: false, error: null, session: data.session || null, last: data };
      return guidedCopilot.session;
    }).catch(err => {
      guidedCopilot = { loading: false, error: err.message || String(err), session: null, last: null };
      renderThread();
      return null;
    });
  }
  function threadFromSessionMessage(msg) {
    if (!msg || typeof msg !== 'object') return null;
    if (msg.role === 'user') {
      const text = msg.text || msg.goal || msg.action;
      return text ? { user: true, html: esc(text) } : null;
    }
    const reply = msg.reply || {};
    const text = reply.en || reply.zh || msg.text || msg.intent || '';
    if (!text) return null;
    return { bot: true, html: bi(reply.en || esc(text), reply.zh || reply.en || esc(text)) };
  }
  function restoreGuidedProjectThread(result, row, kind) {
    disconnectGuidedRunUi();
    const session = result && result.session ? result.session : null;
    guidedCopilot = { loading: false, error: null, session, last: result || guidedCopilot.last };
    const restoredFlow = restoreGuidedSlotsFromSession(session);
    currentId = 'frontdoor';
    thread = [];
    const title = projectTitle(
      (session && session.project_title) || (row && row.title),
      (row && (row.question || row.study_id || row.run_label)) || t('Local project', '本地项目'),
    );
    const path = row && row.project_dir ? compactPath(row.project_dir) : (session && session.project_dir ? compactPath(session.project_dir) : '~/easyicu/projects');
    const nounEn = kind === 'run' ? 'Agent run project' : 'guided draft';
    const nounZh = kind === 'run' ? 'Agent run 项目' : '引导草稿';
    thread.push({ bot: true, html: bi(
      `Opened <strong>${esc(title)}</strong> as this ${nounEn} context. Memory is scoped to <span class="mono">${esc(path)}</span>; Idea Mining and the Research Agent backend still own their artifacts.`,
      `已切换到 <strong>${esc(title)}</strong> 这个${nounZh}上下文。记忆范围限定在 <span class="mono">${esc(path)}</span>；Idea Mining 和 Research Agent 后端仍各自管理其 artifacts。`,
    ) });
    const restored = session && Array.isArray(session.messages) ? session.messages.map(threadFromSessionMessage).filter(Boolean) : [];
    if (restored.length) {
      restored.forEach(item => thread.push(item));
      if (session && session.handoff) thread.push({ bot: true, html: renderGuidedHandoffCard(session.handoff) });
    }
    if (restoredFlow && (EXTRACT.state() || IDEA.state() || REVIEW.state() || guidedAgent)) {
      thread.push({ bot: true, html: bi(
        `Restored the saved setup for this folder. Continue editing here; required configuration stays inside Guided Copilot.`,
        `已恢复这个文件夹里保存的配置。你可以继续在这里编辑；必需配置仍留在研究引导内完成。`,
      ) });
      if (restoredFlow === 'data_extraction' && EXTRACT.state()) thread.push({ guidedExtraction: true });
      else if (restoredFlow === 'idea_mining' && IDEA.state()) thread.push({ guidedIdea: true });
      else if (restoredFlow === 'review_data' && REVIEW.state()) thread.push({ guidedReview: true });
      else if (restoredFlow === 'run_agent' && guidedAgent) thread.push({ guidedAgent: true });
    } else if (!restored.length && kind === 'run') {
      thread.push({ bot: true, html: bi(
        `This context is attached to an existing Agent run folder. Review artifacts here or open Project Monitor; Guided will not rewrite the run outputs.`,
        `这个上下文关联到已有 Agent run 文件夹。你可以在这里审阅 artifacts 或打开项目监控；Guided 不会改写 run 输出。`,
      ) });
    } else if (!restored.length) {
      thread.push({ bot: true, html: bi(renderGuidedGoalCards(), renderGuidedGoalCards()) });
    }
    chips = kind === 'run'
      ? [['Review local artifacts', '@reviewLocalRun'], ['Open Project Monitor', '@openAgent'], ['Use active export for a new run', '@activeExport']]
      : restoredFlow === 'idea_mining'
        ? []
        : [['Use active export', '@activeExport'], ['Continue conversation', '@noop'], ['Open Project Monitor', '@openAgent']];
    renderThread(); renderChips();
  }
  function startFreshGuidedProjectThread(title, path) {
    disconnectGuidedRunUi();
    currentId = 'frontdoor';
    pendingGuidedGoal = null;
    busy = false;
    outputsReady = false;
    diffExpanded = false;
    liveAgentRun = null;
    workspaceSnapshot = null;
    workspaceSnapshotPath = null;
    EXTRACT.clearState();
    REVIEW.clearState();
    guidedAgent = null;
    IDEA.clearIdeaState();
    thread = [];
    thread.push({ bot: true, html: bi(
      `Created <strong>${esc(title)}</strong> at <span class="mono">${esc(path)}</span>. A new Guided conversation has started for this project; previous chat content stays with the previous project.`,
      `已创建 <strong>${esc(title)}</strong> 到 <span class="mono">${esc(path)}</span>。这个项目的新 Guided 对话已开始；之前的聊天内容仍归属之前的项目。`,
    ) });
    thread.push({ bot: true, html: bi(renderGuidedGoalCards(), renderGuidedGoalCards()) });
    chips = [
      [t('Find a Study Idea', '找研究想法'), '@guidedGoal:idea_mining'],
      [t('Prepare Data', '准备/抽取数据'), '@guidedGoal:data_extraction'],
      [t('Review Data', '审阅已有数据'), '@guidedGoal:review_data'],
      [t('Run a Research Project', '运行研究项目'), '@guidedGoal:run_agent'],
    ];
    renderThread();
    renderChips();
  }
  function openGuidedProjectMemory(row, el, kind) {
    if (!row || !row.project_dir || !window.EU_API || !window.EU_API.openGuidedProject) {
      pushBot(
        `This local project cannot be opened as scoped Guided memory yet.`,
        `这个本地项目暂时不能作为有范围的 Guided 记忆打开。`,
      );
      renderThread();
      return Promise.resolve(null);
    }
    document.querySelectorAll('.gd-sess').forEach(s => s.classList.toggle('active', s === el));
    const usePiSession = piProjectShellActive();
    guidedCopilot = { loading: true, error: null, session: null, last: guidedCopilot.last };
    if (!usePiSession) {
      thread = [{ typing: true }];
      chips = [];
      renderThread(); renderChips();
    }
    return window.EU_API.openGuidedProject({
      project_dir: row.project_dir,
      draft_id: row.id || null,
      title: projectTitle(
        row.title,
        row.question || row.study_id || row.run_label || t('Local project', '本地项目'),
      ),
      mode: 'local',
      context: guidedBackendContext(),
    }).then(result => {
      if (!usePiSession) thread = thread.filter(item => !item.typing);
      if (!result || !result.ok) {
        const reason = result && (result.reason || result.error) ? (result.reason || result.error) : 'unknown error';
        pushBot(`Could not open project memory: <span class="mono">${esc(reason)}</span>`, `无法打开项目记忆：<span class="mono">${esc(reason)}</span>`);
        renderThread();
        return;
      }
      if (usePiSession) return bindProjectToPi(result, row);
      restoreGuidedProjectThread(result, row, kind);
      return null;
    }).catch(err => {
      if (!usePiSession) thread = thread.filter(item => !item.typing);
      pushBot(`Could not open project memory: <span class="mono">${esc(err.message || String(err))}</span>`, `无法打开项目记忆：<span class="mono">${esc(err.message || String(err))}</span>`);
      renderThread();
    });
  }
  function guidedGoalMeta(goal) {
    const cards = guidedCopilot.last && Array.isArray(guidedCopilot.last.goal_cards) ? guidedCopilot.last.goal_cards : [];
    return cards.find(c => c.goal === goal) || {
      goal,
      label_en: goal === 'idea_mining' ? 'Find a Study Idea' : goal === 'data_extraction' ? 'Prepare Data' : goal === 'review_data' ? 'Review Data' : 'Run a Research Project',
      label_zh: goal === 'idea_mining' ? '找研究想法' : goal === 'data_extraction' ? '准备/抽取数据' : goal === 'review_data' ? '审阅已有数据' : '运行研究项目',
      target_route: goal === 'idea_mining' ? 'ideas' : goal === 'data_extraction' ? 'extraction' : goal === 'review_data' ? 'patient' : 'agent',
    };
  }
  function renderGuidedGoalCards() {
    const cards = [
      ['idea_mining', 'spark', t('Find a Study Idea', '找研究想法'), t('Paper, PDF, review topic, or hunch → idea ledger.', '文章、PDF、综述主题或想法 → idea ledger。')],
      ['data_extraction', 'extract', t('Prepare Data', '准备/抽取数据'), t('Choose a local data folder, cohort, modules, and export format.', '选择本地数据文件夹、队列、模块和导出格式。')],
      ['review_data', 'eye', t('Review Data', '审阅已有数据'), t('Open patient, cohort, or Cross-DB review for an active export.', '打开 active export 的患者、队列或跨库审阅。')],
      ['run_agent', 'agent', t('Run a Research Project', '运行研究项目'), t('Confirm the plan, provider, and run here; review outputs later in Project Monitor.', '在这里确认计划、provider 与运行；之后到项目监控审阅产出。')],
    ];
    return `
      <div class="gd-frontdoor" data-guided-frontdoor>
        <div class="gdf-memory ${hasGuidedProjectMemory() ? 'ready' : ''}">
          <span>${icon(hasGuidedProjectMemory() ? 'check' : 'folder', 13)}</span>
          <div><strong>${hasGuidedProjectMemory() ? t('Project memory bound', '已绑定项目记忆') : t('Pick a goal to start — your local folder is set up for you', '选一个目标就能开始 —— 本地文件夹我来建')}</strong>
          <small>${hasGuidedProjectMemory()
            ? esc(compactPath((guidedCopilot.session && guidedCopilot.session.project_dir) || (selectedGuidedDraft && selectedGuidedDraft.project_dir) || ''))
            : t('Each study saves into its own local folder (conversation + memory). No path setup up front — I create one in one click, or you can choose a custom folder.', '每个研究都保存到独立的本地文件夹（对话 + 记忆）。无需预先设置路径 —— 我可以一键创建，你也可以选择自定义文件夹。')}</small></div>
          ${hasGuidedProjectMemory() ? '' : `<button class="btn sm" type="button" data-go="@folderquick">${t('Create a starter folder', '创建入门文件夹')}</button>`}
        </div>
        <div class="gdf-head">
          <span class="gdf-kicker">${t('Choose a goal', '选择目标')}</span>
          <strong>${t('What should this local study folder do next?', '这个本地研究文件夹下一步要做什么？')}</strong>
          <span>${t('Pick a goal. If no folder is bound yet, I set up a starter folder in one click and continue the selected workflow inside this conversation.', '选择目标。如果还没有绑定文件夹，我会一键创建入门文件夹，然后在本对话内继续刚才选择的流程。')}</span>
        </div>
        <div class="gdf-grid">
          ${cards.map(([goal, ico, title, body]) => `
            <button class="gdf-card" type="button" data-guided-goal="${goal}">
              <span class="gdf-ico">${icon(ico, 16)}</span>
              <span><strong>${title}</strong><small>${body}</small></span>
              <span class="gdf-go">${icon('arrow', 14)}</span>
            </button>
          `).join('')}
        </div>
        <div class="gdf-ai">
          <span>${icon('shield', 12)} ${t('AI-assisted mode stays opt-in. Local mode never calls a model or reads patient rows in this front door.', 'AI 辅助模式保持显式 opt-in。本地模式在这个前门不会调用模型，也不会读取患者行。')}</span>
        </div>
      </div>`;
  }
  function applyGuidedBackendReply(result, userLabel) {
    if (userLabel) pushUser(userLabel);
    guidedCopilot = {
      loading: false,
      error: null,
      session: result.session || guidedCopilot.session,
      last: result || guidedCopilot.last,
    };
    const reply = result.reply || {};
    if (reply.en || reply.zh) thread.push({ bot: true, html: bi(reply.en || '', reply.zh || reply.en || '') });
    if (result.handoff || (result.result && result.result.handoff)) {
      const handoff = result.handoff || result.result.handoff;
      thread.push({ bot: true, html: renderGuidedHandoffCard(handoff) });
    } else if (result.goal_cards) {
      thread.push({ bot: true, html: renderGuidedGoalCards() });
    }
    renderThread();
    renderChips();
  }
  function renderGuidedHandoffCard(handoff) {
    const meta = guidedGoalMeta(handoff && handoff.goal);
    const target = (handoff && handoff.target_route) || meta.target_route || 'entry';
    return `
      <div class="gd-handoff-ready">
        <span class="gdf-ico">${icon('arrow', 15)}</span>
        <div><strong>${esc(t('Ready to hand off', '可以交接了'))}: ${esc(t(meta.label_en, meta.label_zh || meta.label_en))}</strong>
        <small>${esc(t('The target module will own the detailed settings. You can still edit there before anything runs.', '目标模块会继续负责详细设置。真正运行前你仍可在那里修改。'))}</small></div>
        <button class="btn primary sm" data-guided-handoff="${attr((handoff && handoff.goal) || meta.goal)}" data-target="${attr(target)}">${t('Open module', '打开模块')}</button>
      </div>`;
  }
  function chooseGuidedGoal(goal, label) {
    const seed = guidedFrontdoorSeedText;
    if (requireGuidedProjectMemory(goal, label)) return;
    guidedFrontdoorSeedText = null;
    if (goal === 'idea_mining') {
      IDEA.startGuidedIdeaFlow(label || guidedGoalMeta(goal).label_en);
      if (seed && IDEA.state()) { IDEA.state().topic = seed; renderThread(); }
      return;
    }
    if (goal === 'data_extraction') {
      EXTRACT.start(label || guidedGoalMeta(goal).label_en);
      return;
    }
    if (goal === 'review_data') {
      REVIEW.start(label || guidedGoalMeta(goal).label_en);
      return;
    }
    if (goal === 'run_agent') {
      startGuidedAgentFlow(label || guidedGoalMeta(goal).label_en);
      return;
    }
    if (!window.EU_API || !window.EU_API.runGuidedAction) {
      pushUser(label || goal);
      pushBot(
        `Guided Copilot backend is unavailable, so I cannot create a reliable handoff yet.`,
        `研究引导后端不可用，所以暂时不能创建可靠交接。`,
      );
      renderThread();
      return;
    }
    ensureGuidedSession().then(session => {
      window.EU_API.runGuidedAction({
        session_id: session && session.id,
        action: 'choose_goal',
        goal,
        context: guidedBackendContext(),
      }).then(result => applyGuidedBackendReply(result, label || (guidedGoalMeta(goal).label_en)))
        .catch(err => {
          pushBot(`Guided handoff failed: <span class="mono">${esc(err.message || String(err))}</span>`, `引导交接失败：<span class="mono">${esc(err.message || String(err))}</span>`);
          renderThread();
        });
    });
  }
  function runGuidedHandoff(goal, target, label) {
    if (requireGuidedProjectMemory(goal, label)) return;
    ensureGuidedSession().then(session => {
      window.EU_API.runGuidedAction({
        session_id: session && session.id,
        action: 'handoff_to_module',
        goal,
        context: guidedBackendContext(),
      }).then(result => {
        const handoff = (result.result && result.result.handoff) || result.handoff || {};
        try {
          if (window.EU_GUIDED_HANDOFF) window.EU_GUIDED_HANDOFF.set(handoff);
        } catch (e) {}
        pushUser(label || 'Open module');
        location.hash = '#' + (target || handoff.target_route || 'entry');
      }).catch(err => {
        pushBot(`Could not open the module: <span class="mono">${esc(err.message || String(err))}</span>`, `无法打开模块：<span class="mono">${esc(err.message || String(err))}</span>`);
        renderThread();
      });
    });
  }
  function sendGuidedShortcut(text) {
    if (!window.EU_API || !window.EU_API.sendGuidedMessage) return false;
    if (!hasGuidedProjectMemory()) {
      const goal = IDEA.isGuidedIdeaIntent(text) ? 'idea_mining'
        : isGuidedExtractionIntent(text) ? 'data_extraction'
          : isGuidedReviewIntent(text) ? 'review_data'
            : isGuidedAgentIntent(text) ? 'run_agent'
              : null;
      requireGuidedProjectMemory(goal, text);
      return true;
    }
    pushUser(text);
    ensureGuidedSession().then(session => {
      window.EU_API.sendGuidedMessage({
        session_id: session && session.id,
        message: text,
        context: guidedBackendContext(),
      }).then(result => applyGuidedBackendReply(result, null))
        .catch(err => {
          pushBot(`Guided Copilot could not classify that request: <span class="mono">${esc(err.message || String(err))}</span>`, `研究引导无法识别这个请求：<span class="mono">${esc(err.message || String(err))}</span>`);
          renderThread();
        });
    });
    return true;
  }
  function guidedDraftPayload(label) {
    const src = activeExportSource();
    return {
      title: label || (BRANCH[branch] && BRANCH[branch].chip) || 'Guided Copilot draft',
      folder_slug: slugifyDraftFolder(label || (BRANCH[branch] && BRANCH[branch].chip) || 'guided-study'),
      branch: branch || 'predict',
      depth: depth || 'full',
      data_mode: dataMode || 'demo',
      question: frameFor(branch || 'predict'),
      cohort_hint: BRANCH[branch] && BRANCH[branch].cohortKind === 'databases' ? `${dbCount()} databases` : cohortLine(),
      module_hint: `${mods.length} modules`,
      source: src ? {
        id: src.id,
        label: src.label || src.database || 'active export',
        database: src.database,
        path: src.path,
      } : null,
    };
  }
  function blankGuidedDraftPayload(label) {
    const title = label || 'New local study';
    return {
      title,
      folder_slug: slugifyDraftFolder(title),
      branch: branch || 'predict',
      depth: depth || 'full',
      data_mode: 'unbound',
      question: '',
      cohort_hint: '',
      module_hint: '',
      source: null,
    };
  }
  function blankGuidedBackendContext() {
    return {
      route: 'guided',
      data_mode: 'unbound',
      language: window.EU_LANG || 'en',
      selected_source: null,
      summary: null,
    };
  }
  function slugifyDraftFolder(text) {
    return String(text || 'guided-study').normalize('NFKC').trim().toLowerCase()
      .replace(/[^\p{L}\p{N}._-]+/gu, '-')
      .replace(/-{2,}/g, '-')
      .replace(/^[-._]+|[-._]+$/g, '')
      .slice(0, 64) || 'guided-study';
  }
  function guidedProjectContext() {
    return {
      t,
      icon,
      esc,
      attr,
      compactPath,
      fmtRunTime,
      BRANCH,
      branch,
      guidedDrafts,
      guidedFolderMenuOpen,
      guidedDraftRemoval,
      guidedFolderDialogMode,
      guidedFolderSeedTitle,
      guidedDraftFolderSlug,
      guidedDraftParentDir,
      guidedFolderBrowser,
      guidedKnownProjectsOpen,
      selectedGuidedDraft,
      pendingGuidedGoal,
      localDraftRows,
      guidedKnownProjectRows,
      slugifyDraftFolder,
    };
  }
  function guidedProjectRenderer(name, fallback) {
    const mod = window.EU_GUIDED_PROJECTS;
    if (!mod || typeof mod[name] !== 'function') return fallback || '';
    return mod[name](guidedProjectContext());
  }
  function guidedProjectRailClass() {
    const mod = window.EU_GUIDED_PROJECTS;
    return mod && mod.isProjectRailCollapsed && mod.isProjectRailCollapsed()
      ? 'gd-project-rail-collapsed'
      : '';
  }
  function guidedPanelRenderer(name, fallback) {
    const mod = window.EU_GUIDED_PANELS;
    if (!mod || typeof mod[name] !== 'function') return fallback || '';
    return mod[name]({ t, icon });
  }
  function guidedContextAsideClass() {
    const mod = window.EU_GUIDED_PANELS;
    return mod && typeof mod.contextAsideClass === 'function' ? mod.contextAsideClass() : '';
  }
  function renderGuidedFolderControls() {
    guidedProjectRenderer('renderFolderControls');
  }
  function guidedKnownProjectRows() {
    const seen = new Set();
    const rows = [];
    function add(row, kind) {
      if (!row || !row.project_dir || seen.has(row.project_dir)) return;
      seen.add(row.project_dir);
      rows.push({
        kind,
        project_dir: row.project_dir,
        title: projectTitle(
          row.title,
          row.question || row.study_id || row.run_label || (kind === 'run'
            ? t('Agent run folder', 'Agent 运行文件夹')
            : t('Guided Copilot folder', '研究引导文件夹')),
        ),
        subtitle: kind === 'run'
          ? `${row.readiness_status || row.gate_status || 'analysis_only'} · ${row.artifact_count || 0} artifacts · ${fmtRunTime(row.updated_at)}`
          : `${row.status || 'metadata_only'} · ${row.depth || 'full'} · ${row.data_mode || 'local'} · ${fmtRunTime(row.updated_at || row.created_at)}`,
      });
    }
    localDraftRows().forEach(row => add(row, 'draft'));
    return rows.slice(0, 12);
  }
  function renderGuidedKnownProjectPicker() {
    return guidedProjectRenderer('renderKnownProjectPicker');
  }
  function renderGuidedFolderBrowser() {
    return guidedProjectRenderer('renderFolderBrowser');
  }
  function captureGuidedDraftDialogState(box) {
    if (!box) return;
    const titleEl = box.querySelector('[data-draft-title]');
    const slugEl = box.querySelector('[data-draft-slug]');
    const parentEl = box.querySelector('[data-draft-parent-dir]');
    if (titleEl && titleEl.value.trim()) guidedFolderSeedTitle = titleEl.value.trim();
    if (slugEl && slugEl.value.trim()) guidedDraftFolderSlug = slugifyDraftFolder(slugEl.value);
    if (parentEl) guidedDraftParentDir = parentEl.value.trim() || guidedDraftParentDir || '~/easyicu/projects';
  }
  function loadGuidedFolderBrowser(path) {
    if (guidedFolderDialogMode === 'new') {
      captureGuidedDraftDialogState(document.querySelector('[data-folder-dialog][data-draft-setup]'));
    }
    guidedFolderBrowser.open = true;
    guidedFolderBrowser.loading = true;
    guidedFolderBrowser.error = null;
    guidedFolderBrowser.path = String(path || guidedFolderBrowser.path || '');
    renderGuidedFolderDialog();
    if (!window.EU_API || !window.EU_API.listDir) {
      guidedFolderBrowser.loading = false;
      guidedFolderBrowser.error = t('Local folder picker API is unavailable.', '本地文件夹选择 API 不可用。');
      renderGuidedFolderDialog();
      return;
    }
    window.EU_API.listDir(guidedFolderBrowser.path)
      .then(result => {
        guidedFolderBrowser.loading = false;
        guidedFolderBrowser.data = result || {};
        guidedFolderBrowser.path = (result && result.path) || guidedFolderBrowser.path || '';
        guidedFolderBrowser.error = result && result.ok === false ? (result.error || 'folder_error') : null;
        renderGuidedFolderDialog();
      })
      .catch(err => {
        guidedFolderBrowser.loading = false;
        guidedFolderBrowser.error = String(err && err.message || err || 'folder_error');
        renderGuidedFolderDialog();
      });
  }
  function renderGuidedFolderDialog() {
    guidedProjectRenderer('renderFolderDialog');
  }
  function openGuidedFolderDialog(mode, seedTitle) {
    guidedFolderMenuOpen = false;
    guidedFolderDialogMode = mode === 'open' ? 'open' : 'new';
    guidedFolderSeedTitle = seedTitle || guidedFolderSeedTitle || 'New local study';
    if (guidedFolderDialogMode === 'new') {
      guidedDraftFolderSlug = slugifyDraftFolder(guidedFolderSeedTitle);
      guidedDraftParentDir = guidedDraftParentDir || '~/easyicu/projects';
    }
    guidedFolderBrowser = { open: false, loading: false, error: null, data: null, path: '' };
    guidedKnownProjectsOpen = false;
    renderGuidedFolderControls();
    renderGuidedFolderDialog();
    setTimeout(() => {
      const selector = guidedFolderDialogMode === 'open' ? '[data-existing-project-dir]' : '[data-draft-title]';
      const inp = document.querySelector(`[data-folder-dialog] ${selector}`);
      if (inp) { inp.focus(); if (inp.select) inp.select(); }
    }, 80);
  }
  function closeGuidedFolderDialog() {
    guidedFolderMenuOpen = false;
    guidedFolderDialogMode = null;
    guidedFolderBrowser = { open: false, loading: false, error: null, data: null, path: '' };
    guidedKnownProjectsOpen = false;
    renderGuidedFolderControls();
    renderGuidedFolderDialog();
  }
  function showGuidedDraftSetup(seedTitle, mode) {
    openGuidedFolderDialog(mode || 'new', seedTitle || (BRANCH[branch] && BRANCH[branch].chip) || 'New local study');
  }
  function latestGuidedDraftSetupBox(fallback) {
    if (fallback && fallback.matches && fallback.matches('[data-draft-setup]') && fallback.isConnected) return fallback;
    const boxes = Array.from(document.querySelectorAll('[data-draft-setup]'));
    return boxes.length ? boxes[boxes.length - 1] : null;
  }
  function setGuidedProjectOpenStatus(box, state, html) {
    const target = latestGuidedDraftSetupBox(box);
    if (!target) return;
    const status = target.querySelector('[data-project-open-status]');
    const buttons = target.querySelectorAll('[data-openprojectfolder], [data-reviewexportfolder]');
    buttons.forEach(button => {
      button.disabled = state === 'loading';
      button.setAttribute('aria-busy', state === 'loading' ? 'true' : 'false');
    });
    if (!status) return;
    if (!html) {
      status.hidden = true;
      status.className = 'gds-status';
      status.innerHTML = '';
      return;
    }
    status.hidden = false;
    status.className = `gds-status ${state || 'info'}`;
    status.innerHTML = html;
  }
  function registerExistingExportForReview(exportDir, setupBox, opts) {
    const raw = String(exportDir || '').trim();
    const options = opts || {};
    if (!raw) {
      setGuidedProjectOpenStatus(
        setupBox,
        'error',
        `${icon('info', 12)} <span>${t('Choose or paste a local EasyICU export folder first, then I can review the extracted data.', '请先选择或粘贴一个本地 EasyICU export 文件夹，然后我才能审阅已提取数据。')}</span>`,
      );
      return Promise.resolve(false);
    }
    if (!window.EU_API || !window.EU_API.registerWorkspaceSource) {
      setGuidedProjectOpenStatus(
        setupBox,
        'error',
        `${icon('info', 12)} <span>${t('The export registration API is unavailable, so I cannot safely review that folder yet.', '导出注册 API 不可用，所以暂时不能安全审阅该文件夹。')}</span>`,
      );
      return Promise.resolve(false);
    }
    if (options.pushUser !== false) pushUser(`Review extracted data folder: ${raw}`);
    setGuidedProjectOpenStatus(
      setupBox,
      'loading',
      `${icon('refresh', 12)} <span>${t('Checking and registering this EasyICU export for review...', '正在检查并注册这个 EasyICU export 以便审阅...')}</span>`,
    );
    thread.push({ typing: true });
    renderThread();
    return window.EU_API.registerWorkspaceSource(raw, {
      active: true,
      crossdb: true,
      label: 'Guided review export',
    }).then(registry => {
      thread = thread.filter(item => !item.typing);
      const source = (registry.sources || []).find(s => s.path === registry.active_path) || (registry.sources || []).find(s => s.path === raw);
      setGuidedProjectOpenStatus(
        setupBox,
        'ok',
        `${icon('check', 12)} <span>${t('Export registered as the active data source. Loading review panels...', '导出已注册为 active 数据源，正在加载审阅面板...')}</span>`,
      );
      pendingGuidedGoal = null;
      closeGuidedFolderDialog();
      pushBot(
        `Registered <span class="mono">${esc(compactPath(raw))}</span> as the active EasyICU export. I’ll review the extracted data here; no project memory or Agent run was created.`,
        `已将 <span class="mono">${esc(compactPath(raw))}</span> 注册为 active EasyICU export。接下来会在这里审阅已提取数据；不会创建项目记忆或 Agent run。`,
      );
      setVal({ data: (source && (source.label || source.database)) || 'active export' });
          REVIEW.start(null);
      return true;
    }).catch(err => {
      thread = thread.filter(item => !item.typing);
      const msg = err && (err.message || String(err)) || 'export_register_failed';
      const fallback = options.projectReason
        ? t('This folder is neither an openable Guided project nor a valid EasyICU export.', '这个文件夹既不是可打开的 Guided 项目，也不是有效的 EasyICU export。')
        : t('This does not look like a valid EasyICU export folder.', '这看起来不是有效的 EasyICU export 文件夹。');
      pushBot(
        `I could not review that folder as extracted data: <span class="mono">${esc(msg)}</span>`,
        `无法把该文件夹作为已提取数据审阅：<span class="mono">${esc(msg)}</span>`,
      );
      renderThread();
      setGuidedProjectOpenStatus(
        setupBox,
        'error',
        `${icon('info', 12)} <span>${esc(fallback)} <span class="mono">${esc(msg)}</span></span>`,
      );
      return false;
    });
  }
  function openExistingGuidedProject(projectDir, setupBox) {
    const raw = String(projectDir || '').trim();
    if (!raw) {
      setGuidedProjectOpenStatus(
        setupBox,
        'error',
        `${icon('info', 12)} <span>${t('Paste a local EasyICU project folder path first, then I can open the memory scoped to that folder.', '请先粘贴一个本地 EasyICU 项目文件夹路径，然后我才能打开绑定到该文件夹的记忆。')}</span>`,
      );
      const focus = latestGuidedDraftSetupBox(setupBox);
      const input = focus ? focus.querySelector('[data-existing-project-dir]') : null;
      if (input) input.focus();
      return;
    }
    pushUser(`Open local project folder: ${raw}`);
    if (!window.EU_API || !window.EU_API.openGuidedProject) {
      renderThread();
      setGuidedProjectOpenStatus(
        setupBox,
        'error',
        `${icon('info', 12)} <span>${t('The local project memory endpoint is unavailable, so I cannot open that folder reliably yet.', '本地项目记忆端点不可用，所以暂时不能可靠打开这个文件夹。')}</span>`,
      );
      return;
    }
    setGuidedProjectOpenStatus(
      setupBox,
      'loading',
      `${icon('refresh', 12)} <span>${t('Opening folder memory and restoring this project context...', '正在打开文件夹记忆并恢复这个项目上下文...')}</span>`,
    );
    thread.push({ typing: true });
    renderThread();
    setGuidedProjectOpenStatus(setupBox, 'loading', `${icon('refresh', 12)} <span>${t('Opening folder memory and restoring this project context...', '正在打开文件夹记忆并恢复这个项目上下文...')}</span>`);
    window.EU_API.openGuidedProject({
      project_dir: raw,
      mode: 'local',
      context: guidedBackendContext(),
    }).then(result => {
      thread = thread.filter(item => !item.typing);
      if (!result || !result.ok) {
        const reason = result && (result.reason || result.error) ? (result.reason || result.error) : 'unknown error';
        return registerExistingExportForReview(raw, setupBox, {
          pushUser: false,
          projectReason: reason,
        });
      }
      setGuidedProjectOpenStatus(
        setupBox,
        'ok',
        `${icon('check', 12)} <span>${t('Research project opened.', '研究项目已打开。')}</span>`,
      );
      selectedGuidedDraft = {
        id: result.session && result.session.draft_id,
        title: result.session && result.session.project_title,
        project_dir: result.session && result.session.project_dir,
      };
      selectedGuidedRun = null;
      closeGuidedFolderDialog();
      if (piProjectShellActive()) bindProjectToPi(result, selectedGuidedDraft);
      else restoreGuidedProjectThread(result, selectedGuidedDraft, 'draft');
      continuePendingGuidedGoal();
      loadGuidedDrafts(true);
    }).catch(err => {
      thread = thread.filter(item => !item.typing);
      return registerExistingExportForReview(raw, setupBox, {
        pushUser: false,
        projectReason: err && (err.message || String(err)) || 'open_project_failed',
      });
    });
  }
  function createLocalGuidedDraft(label, folderSlug, parentDir, opts) {
    const options = opts || {};
    const text = label || 'New local study';
    const parent = String(parentDir || '').trim();
    pushUser(parent ? `Create local study folder: ${text} in ${parent}` : `Create local study folder: ${text}`);
    if (!window.EU_API || !window.EU_API.createGuidedDraft) {
      pushBot(
        `This browser can draft the conversation, but the local draft registry endpoint is not available yet.`,
        `这个浏览器可以先记录对话，但本地草稿 registry 端点暂时不可用。`,
      );
      renderThread();
      return;
    }
    pushBot(
      `Creating a <strong>metadata-only local study folder</strong>. This does not create an Agent run, does not read patient rows, and does not unlock a manuscript draft.`,
      `正在创建<strong>仅元数据的本地研究文件夹</strong>。这不会创建 Agent run、不会读取患者行，也不会解锁稿件草稿。`,
    );
    renderThread();
    const payload = blankGuidedDraftPayload(text);
    payload.folder_slug = folderSlug || payload.folder_slug;
    if (parent) payload.parent_dir = parent;
    window.EU_API.createGuidedDraft(payload).then(result => {
      if (!result || result.ok === false || !result.draft) {
        throw new Error((result && (result.reason || result.error)) || 'guided_draft_create_failed');
      }
      selectedGuidedDraft = result.draft || null;
      loadGuidedDrafts(true);
      return bindGuidedDraftMemory(selectedGuidedDraft, true).then(opened => ({ result, opened }));
    }).then(({ result, opened }) => {
      const draft = (opened && opened.session) ? {
        id: opened.session.draft_id,
        title: opened.session.project_title,
        project_dir: opened.session.project_dir,
      } : (result.draft || selectedGuidedDraft);
      selectedGuidedDraft = draft || selectedGuidedDraft;
      const title = selectedGuidedDraft && selectedGuidedDraft.title ? selectedGuidedDraft.title : text;
      const path = selectedGuidedDraft && selectedGuidedDraft.project_dir ? compactPath(selectedGuidedDraft.project_dir) : '~/easyicu/projects';
      closeGuidedFolderDialog();
      if (piProjectShellActive() && opened && opened.ok) {
        bindProjectToPi(opened, selectedGuidedDraft);
        return;
      }
      // One-click starter path: keep the goal the user already picked and resume
      // it in the fresh folder, instead of resetting them back to goal cards.
      if (options.continueGoal && pendingGuidedGoal) {
        pushBot(
          `Created <strong>${esc(title)}</strong> at <span class="mono">${esc(path)}</span> — metadata-only, no run yet. Continuing where you left off.`,
          `已创建 <strong>${esc(title)}</strong> 到 <span class="mono">${esc(path)}</span> —— 仅元数据、暂无运行。这就接着刚才的步骤继续。`,
        );
        renderThread();
        continuePendingGuidedGoal();
      } else {
        startFreshGuidedProjectThread(title, path);
      }
    }).catch(err => {
      pushBot(
        `Could not save the guided draft: <span class="mono">${esc(err.message || String(err))}</span>`,
        `无法保存引导草稿：<span class="mono">${esc(err.message || String(err))}</span>`,
      );
      renderThread();
    });
  }
  function quickCreateGuidedStarterFolder() {
    // First-run friction reducer: create a metadata-only folder at the default
    // location in one click (no path dialog, no typing) and resume the goal the
    // user just picked. Still explicit (a button/chip), still local-only.
    // Name the folder after the chosen goal (not the button text) so it reads
    // sensibly on disk; fall back to a neutral title when no goal is pending yet.
    const pend = pendingGuidedGoal;
    const meta = pend ? guidedGoalMeta(pend.goal) : null;
    const seed = meta
      ? t(meta.label_en || 'New study', meta.label_zh || meta.label_en || 'New study')
      : t('My first study', '我的第一个研究');
    createLocalGuidedDraft(seed, slugifyDraftFolder(seed), '~/easyicu/projects', { continueGoal: !!pend });
  }
  function openGuidedRunReview(row, label) {
    if (!row || !row.project_dir || !window.EU_API || !window.EU_API.loadAgentRunReview) {
      pushBot(
        `This run does not expose a readable local artifact folder yet, so I cannot open it as a reviewable run.`,
        `这个 run 还没有可读取的本地 artifact 文件夹，所以暂时不能作为可审阅运行打开。`,
      );
      renderThread();
      return;
    }
    selectedGuidedRun = row;
    selectedGuidedDraft = null;
    pushUser(label || 'Review local run');
    pushBot(
      `Reading local run artifacts from <span class="mono">${esc(compactPath(row.project_dir))}</span>. Only whitelisted JSON files are opened.`,
      `正在从 <span class="mono">${esc(compactPath(row.project_dir))}</span> 读取本地 run artifacts。只会打开白名单 JSON 文件。`,
    );
    renderThread();
    window.EU_API.loadAgentRunReview(row.project_dir).then(review => {
      liveAgentRun = {
        active: false,
        result: {
          run_id: review.run_id,
          run_label: row.run_label || review.run_id,
          study_id: review.study_id,
          project_dir: review.project_dir,
          run_type: review.run_type,
          artifacts: review.artifacts || [],
          gate: review.gate || {},
        },
        error: null,
      };
      outputsReady = true;
      const readiness = (review.readiness && review.readiness.status) || row.readiness_status || 'analysis_only';
      pushBot(
        `Opened <strong>${esc(review.study_id || row.study_id || 'local study')}</strong> / <span class="mono">${esc(review.run_id || row.run_id || 'run')}</span>: ${esc(readiness)} · ${(review.artifacts || []).length} artifacts. Draft/reportable remains locked unless Agent evidence checks say otherwise.`,
        `已打开 <strong>${esc(review.study_id || row.study_id || '本地研究')}</strong> / <span class="mono">${esc(review.run_id || row.run_id || 'run')}</span>：${esc(readiness)} · ${(review.artifacts || []).length} 个 artifact。除非 Agent 证据核验明确允许，草稿/reportable 仍保持锁定。`,
      );
      thread.push({ diff: true });
      chips = [['Open in Project Monitor', '@openAgent'], ['Use active export for a new run', '@activeExport']];
      renderThread(); renderChips();
    }).catch(err => {
      pushBot(
        `Could not open that run: <span class="mono">${esc(err.message || String(err))}</span>`,
        `无法打开这个 run：<span class="mono">${esc(err.message || String(err))}</span>`,
      );
      renderThread();
    });
  }

  /* ============== conversation script ============== */
  const STATES = {
    frontdoor: {
      delay: 220,
      step: 'question',
      bot: () => [
        bi(
          `Hi — I’m the EasyICU <strong>Guided Copilot</strong>. I can help you finish the common workflows here, but first every conversation is scoped to a local study folder.`,
          `你好，我是 EasyICU <strong>研究引导</strong>。常用流程可以在这里完成，但每条对话都必须先绑定到本地研究文件夹。`,
        ),
        bi(renderGuidedGoalCards(), renderGuidedGoalCards()),
      ],
      chips: () => [
        [t('New / open study folder', '新建/打开研究文件夹'), '@foldernew'],
        [t('Find a Study Idea', '找研究想法'), '@guidedGoal:idea_mining'],
        [t('Prepare Data', '准备/抽取数据'), '@guidedGoal:data_extraction'],
        [t('Review Data', '审阅已有数据'), '@guidedGoal:review_data'],
        [t('Run a Research Project', '运行研究项目'), '@guidedGoal:run_agent'],
      ],
      markStep: 'question',
    },
    goal: {
      delay: 340,
      bot: [
        bi(
          `Hi — I’m <strong>Guided Copilot</strong>, running a <strong>scripted demo walkthrough</strong>. Every number and artifact here is a seeded example, not a real result — switch to <strong>Real</strong> data any time to run your own study.`,
          `你好，我是<strong>研究引导</strong>，当前是<strong>脚本化演示流程</strong>。这里的每个数字和产物都是示例种子数据，不是真实结果 —— 随时可切换到<strong>真实</strong>数据来运行你自己的研究。`,
        ),
        bi(
          `First, <strong>how far do you want to go today?</strong> This just sets where I stop — you can always extend later.`,
          `先确认一下：<strong>今天你想做到哪一步？</strong> 这只是设置我在哪里停下，后面随时可以继续扩展。`,
        ),
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
        bi(
          `Now — what would you like to study? Pick a direction below, or describe your own.`,
          `接下来，你想研究什么？可以选下面的方向，也可以直接描述自己的问题。`,
        ),
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
      bot: () => [studyParams.caught
        ? bi(
            `From your description I picked up <strong>${studyParams.caught}</strong>. Here’s a tighter, researchable framing — tweak anything:`,
            `我从你的描述里识别到 <strong>${studyParams.caught}</strong>。下面是一个更紧凑、可执行的研究表述，你可以继续改：`,
          )
        : clarified
          ? bi(
              `Got it — <strong>${clarified}</strong>. Here’s a tighter, researchable framing:`,
              `明白：<strong>${clarified}</strong>。下面是一个更紧凑、可执行的研究表述：`,
            )
          : bi(
              `Good — here’s a tighter, researchable framing:`,
              `好的，下面是一个更紧凑、可执行的研究表述：`,
            )],
      chips: () => [['Why frame it this way?', '@why'], ['Use my own wording', '@noop']],
      markStep: 'question', markStatus: 'active',
      val: { question: () => userQuestion || BRANCH[branch].chip },
    },
    toData: {
      step: 'data', card: true,
      bot: [bi(`How should data enter the workspace?`, `数据要怎样进入工作区？`)],
      chips: () => [['What’s the difference?', '@why']],
      markStep: 'data',
      val: { question: () => userQuestion || BRANCH[branch].chip },
    },
    realConfirm: {
      step: 'data',
      bot: [bi(
        `Before we read local data, two things: this reads files on your machine and this first Agent run is a <strong>local preflight only</strong>: no external model call, no uploads, and never patient rows. Continue?`,
        `读取本地数据前先确认两点：这会读取你机器上的文件；第一次 Agent run 只是<strong>本地预检</strong>，不会外部模型调用、不会上传、也不会持久化患者行。继续吗？`,
      )],
      chips: () => [['Continue with local data', 'connect'], ['Use demo instead', '@usedemo']],
      markStep: 'data',
      val: { question: () => userQuestion || BRANCH[branch].chip },
    },
    connect: {
      step: 'data',
      bot: [bi(
        `Point me at a local ICU export root — I’ll detect the layout. Nothing leaves your machine.`,
        `请选择一个本地 ICU export 根目录，我会自动识别布局。任何数据都不会离开本机。`,
      )],
      once: 'folder',
      chips: [],
      markStep: 'data',
    },
    detect: {
      delay: 360,
      step: 'data',
      bot: [bi(`Scanning the folder…`, `正在扫描文件夹…`)],
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
      bot: () => [BRANCH[branch].cohortKind === 'databases'
        ? bi(
            `Pick the databases to compare — the same cohort definition applies to each.`,
            `选择要比较的数据库，同一个队列定义会应用到每个数据库。`,
          )
        : (realMode()
            ? bi(
                `Here’s the active export cohort summary. Full-cohort aggregates are used; row previews stay bounded.`,
                `这是当前 active export 的队列摘要。这里使用全队列聚合，行级预览保持有界。`,
              )
            : bi(
                `Here’s a starting cohort. The demo set is small on purpose so every screen stays explorable.`,
                `这是一个起始队列。演示集故意保持较小，方便每个页面都能快速探索。`,
              ))],
      chips: () => [['Adjust patient count', '@hintN'], ['Why this matters', '@why']],
      markStep: 'cohort',
      val: { data: () => dataMode === 'demo' ? 'Demo · local' : 'Local export' },
    },
    toConcepts: {
      step: 'concepts', card: true,
      bot: [bi(
        `I’ve pre-selected the feature modules your question needs. Toggle any — coverage gets audited before modelling.`,
        `我已预选这个问题需要的特征模块。你可以增删模块；建模前会先审计覆盖率。`,
      )],
      chips: () => [['Why these modules?', '@why']],
      markStep: 'concepts',
      val: { cohort: () => BRANCH[branch].cohortKind === 'databases' ? `${dbCount()} databases` : cohortLine() },
    },
    toExtract: {
      delay: 420,
      step: 'extract', card: true,
      bot: [bi(
        `Extracting now — normalizing, resolving the cohort, and packaging frames locally.`,
        `正在抽取：标准化概念、解析队列，并在本机打包数据表。`,
      )],
      chips: [],
      markStep: 'extract',
      val: { concepts: () => `${mods.length} modules` },
      onShown() { extractPhase = 'run'; runExtract(); },
    },
    toReview: {
      delay: 300,
      step: 'review', card: true,
      bot: () => [bi(
        `Here’s a quick look — the full ${BRANCH[branch].openTarget === 'crossdb' ? 'benchmark' : 'review'} is one click away in the workspace.`,
        `这里先快速看一眼；完整${BRANCH[branch].openTarget === 'crossdb' ? '跨库比较' : '审阅'}可以一键在工作区打开。`,
      )],
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
        ? bi(
            `Running a registry-backed local preflight — source resolution, bounded snapshot, evidence checks, and local artifact write. No external model call.`,
            `正在运行 registry-backed 本地预检：解析数据源、生成有界快照、执行证据核验，并写入本地 artifact。不会调用外部模型。`,
          )
        : bi(
            `Running the analysis — deterministic steps, no tokens. I’ll only draft findings after every step’s evidence contract passes.`,
            `正在运行分析：确定性步骤，不消耗 token。只有每一步 evidence contract 通过后，才会进入 findings 草稿。`,
          )],
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
    guidedProjectRenderer('renderProjectRail');
  }

  /* ============== screen ============== */
  function initializeGuidedState() {
    if (guidedMounted) {
      captureGuidedComposerDraft();
      return false;
    }
    reset();
    currentId = 'frontdoor';
    guidedMounted = true;
    if (STARTUP && STARTUP.begin) STARTUP.begin();
    return true;
  }

  S.guided = {
    section: 'guided', full: true,
    render() {
      guidedInitialRender = initializeGuidedState();
      return `
      <div class="gd-shell">
        <h1 class="shell-sr-only" tabindex="-1">${t('Guided Copilot', '研究引导')}</h1>
        <div class="gd-main threecol ${guidedProjectRailClass()} ${guidedContextAsideClass()} ${STARTUP && STARTUP.isActive && STARTUP.isActive() ? 'gd-startup-active' : ''}" ${STARTUP && STARTUP.isActive && STARTUP.isActive() ? 'aria-busy="true"' : ''}>
          ${guidedProjectRenderer('renderShellRail')}
          <div class="gd-conv">
            <div class="gd-pi-shell" id="gdPiShell" aria-label="${t('EasyICU Copilot conversation', 'EasyICU 研究助手对话')}">
              <div class="gpi-activate gpi-restoring" role="status" aria-live="polite">
                <div class="gpi-kicker">EASYICU COPILOT · ${t('RESTORING PROJECT', '正在恢复项目')}</div>
                <h2>${t('Restoring your current research', '正在恢复当前研究')}</h2>
                <p>${t('EasyICU is loading the saved project, model connection, and conversation together.', 'EasyICU 正在一起读取已保存的项目、模型连接和对话。')}</p>
              </div>
            </div>
            <div class="gd-legacy-shell" id="gdLegacyShell">
              <div class="gd-scroll" id="gdScroll"><div class="gd-thread" id="gdThread" role="log" aria-live="polite" aria-label="Copilot conversation"></div></div>
              <div class="gd-suggest" id="gdSuggest"></div>
              <div class="gd-composer-wrap">
                <div class="gd-composer">
                  <input class="gd-input" id="gdInput" value="${attr(guidedComposerDraft)}" placeholder="${t('Reply, or tap an option above to continue…', '回复，或点击上方选项继续…')}" autocomplete="off" aria-label="${t('Message Guided Copilot', '给研究引导发送消息')}" />
                  <button type="button" class="gd-send" id="gdSend" aria-label="${t('Send message', '发送消息')}">${icon('arrow', 16)}</button>
                </div>
                <div class="gd-foot-note">${t('Guided Copilot · local first · nothing leaves your machine', '研究引导 · 本地优先 · 数据不离开你的电脑')}</div>
              </div>
            </div>
          </div>
          ${guidedPanelRenderer('renderContextAsideRestore')}
          <aside class="gd-aside" id="gdContextAside">
            ${guidedPanelRenderer('renderContextAsideCollapse')}
            <div class="gd-study-aside" id="gdStudyAside">
              <div class="gd-aside-head"><div class="eyebrow">${t('Building your study', '正在搭建你的研究')}</div><div class="at">${t('Study workspace', '研究工作区')}</div><div class="asub">${t('Assembles as we talk · edit any step', '随对话逐步组装 · 任意步骤可编辑')}</div></div>
              <div class="gd-aside-body" id="gdAsideBody"></div>
            </div>
            <div class="gpi-preview-aside" id="gdPreviewAside" hidden></div>
          </aside>
          ${STARTUP && STARTUP.markup ? STARTUP.markup(t) : ''}
        </div>
        <div id="gdFolderDialogHost"></div>
        <div id="gdRemoveDraftDialogHost"></div>
      </div>`;
    },
    afterRender(root) {
      const initialRender = guidedInitialRender;
      guidedInitialRender = false;
      const guidedBinding = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.takeGuidedHandoff
        ? window.EU_AGENT_STUDY_CONTEXT.takeGuidedHandoff()
        : null;
      if (guidedBinding) {
        selectedGuidedDraft = {
          id: guidedBinding.project_id,
          title: guidedBinding.project_title,
          binding_receipt: guidedBinding.binding_receipt || null,
          study_context_id: guidedBinding.binding_receipt && guidedBinding.binding_receipt.study_context_id,
          study_context_revision: guidedBinding.binding_receipt && guidedBinding.binding_receipt.study_context_revision,
        };
      }
      renderGuidedFolderControls();
      renderGuidedFolderDialog();
      renderGuidedDraftRemovalDialog();
      renderAside();
      if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.mount) {
        window.EU_GUIDED_PI_PREVIEW.mount(root.querySelector('#gdPreviewAside'));
      }
      renderSessions();
      const piOwner = window.EU_GUIDED_PI;
      let piReady = Promise.resolve();
      if (piOwner && piOwner.mount) {
        if (piOwner.setProjectDiscoveryLoading) piOwner.setProjectDiscoveryLoading(true);
        piReady = Promise.resolve(piOwner.mount(root.querySelector('#gdPiShell')));
      }
      const draftsReady = loadGuidedDrafts();
      Promise.allSettled([Promise.resolve(draftsReady), piReady]).finally(() => {
        let projectReady = Promise.resolve();
        if (selectedGuidedDraft && piOwner && piOwner.bindProject) {
          const bindingReceipt = selectedGuidedDraft.binding_receipt || null;
          projectReady = Promise.resolve(piOwner.bindProject({
            id: selectedGuidedDraft.id,
            title: selectedGuidedDraft.title || selectedGuidedDraft.id,
            binding_receipt: bindingReceipt,
          }));
          if (bindingReceipt) delete selectedGuidedDraft.binding_receipt;
        }
        projectReady.finally(() => {
          if (piOwner && piOwner.setProjectDiscoveryLoading) piOwner.setProjectDiscoveryLoading(false);
          renderSessions();
          renderAside();
          if (STARTUP && STARTUP.finish) STARTUP.finish(root);
        });
      });
      // The global topbar Demo/Real toggle is the source of truth on entry:
      // sync UP only (demo → real), so a Real-mode user never sees the aside
      // claim "Demo · local". The conversation may still opt into demo
      // explicitly (@usedemo), which stays untouched.
      if (window.EU_DATA === 'real' && dataMode !== 'real') dataMode = 'real';
      // continue from the dock if we just expanded it
      let bridged = false;
      try {
        const b = window.__cpBridge;
        if (b && Date.now() - b.ts < 60000) {
          bridged = true; window.__cpBridge = null;
          if (b.dataMode) dataMode = b.dataMode;
          // Real mode must land in the real project-memory flow (frontdoor), never
          // the seeded-demo pipeline (welcome/BRANCH) — that pipeline is an
          // illustrative demo and only appropriate when the user is in Demo mode.
          if (dataMode === 'real') {
            go('frontdoor');
            if (b.lastUser) setTimeout(() => { handleText(stripTags(b.lastUser)); }, 300);
          } else {
          go('welcome');
          setTimeout(() => {
            rememberUserQuestion(b.lastUser || '');
            if (b.branchHint && BRANCH[b.branchHint]) {
              branch = b.branchHint;
              extractEntities(b.lastUser || '');
              if (branch === 'predict' && !endpointPinned(b.lastUser || '')) { go('clarify', b.lastUser || BRANCH[branch].chip); }
              else { go('frame', b.lastUser || BRANCH[branch].chip); }
            } else if (b.lastUser) {
              handleText(stripTags(b.lastUser));
            } else {
              const routeLabel = b.route && b.route !== 'entry'
                ? (({extraction:'Data Extraction',patient:'Patient Review',cohort:'Cohort Statistics',crossdb:'Cross-database comparison',agent:'Project Monitor'}[b.route]) || 'the workspace')
                : '';
              pushBot(
                `Continuing from the dock${routeLabel ? ` — you were on <strong>${routeLabel}</strong>` : ''}. Want to turn that into a full study?`,
                `我会接着右下角 quick help 的上下文继续${routeLabel ? `：刚才你在 <strong>${routeLabel}</strong>` : ''}。要把它扩展成完整研究吗？`,
              );
              renderThread();
            }
          }, 700);
          }
        }
      } catch (e) {}
      if (!bridged) {
        if (initialRender) go('frontdoor');
        else {
          renderThread();
          renderChips();
        }
      }
      setTimeout(() => { const inp = root.querySelector('#gdInput'); if (inp) inp.focus(); }, 400);

      const shell = root.querySelector('.gd-shell');
      shell.addEventListener('click', (e) => {
        const folderToggle = e.target.closest('[data-folder-menu-toggle]');
        if (folderToggle) {
          guidedFolderMenuOpen = !guidedFolderMenuOpen;
          renderGuidedFolderControls();
          return;
        }
        const folderChoice = e.target.closest('[data-folder-choice]');
        if (folderChoice) {
          openGuidedFolderDialog(folderChoice.dataset.folderChoice, guidedFolderSeedTitle || 'New local study');
          return;
        }
        if (guidedFolderMenuOpen && !e.target.closest('.gd-folder-picker')) {
          guidedFolderMenuOpen = false;
          renderGuidedFolderControls();
        }
        if (e.target.closest('[data-folder-dialog-close]')) {
          closeGuidedFolderDialog();
          return;
        }
        if (e.target.closest('[data-remove-draft-close]')) {
          closeGuidedDraftRemovalDialog();
          return;
        }
        if (e.target.closest('[data-confirm-remove-draft]')) {
          confirmLocalGuidedDraftRemoval();
          return;
        }
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
          if (tok.startsWith('@guidedGoal:')) { chooseGuidedGoal(tok.split(':')[1], label); return; }
          if (tok.startsWith('@depth:')) { depth = tok.split(':')[1]; renderAside(); go('welcome', label); return; }
          if (tok === '@regoal') { pushUser(label); go('goal'); return; }
          if (tok === '@finish') { pushUser(label); finishHere(); return; }
          if (tok === '@extendNext') {
            const prev = depth; bumpDepth(); renderAside(); pushUser(label);
            pushBot(
              `Extending from <strong>${DEPTH[prev].label}</strong> to <strong>${DEPTH[depth].label}</strong> — picking up where we left off.`,
              `正在从 <strong>${DEPTH[prev].label}</strong> 扩展到 <strong>${DEPTH[depth].label}</strong>，我会从刚才停下的地方继续。`,
            );
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
          if (tok === '@useFrame') {
            pushUser(label);
            acceptedFrame = true;
            pushBot(
              `Switched to the suggested wording. It is now the question I submit and bind evidence to — say “reframe” to go back to your own words.`,
              `已改用建议措辞。它现在就是我提交并用于证据绑定的问题 —— 说“重新表述”可以换回你自己的说法。`,
            );
            renderThread(); renderAside(); renderChips();
            return;
          }
          // "Use my own wording" must actually restore the user's own words,
          // not just say it will. Anything already accepted is released here.
          if (tok === '@noop') {
            pushUser(label);
            acceptedFrame = false;
            pushBot(
              userQuestion
                ? `Back to your own wording — I’ll submit and bind evidence to: “${esc(userQuestion)}”. Type a new sentence any time to replace it.`
                : `Go ahead — type your own wording in the box and I’ll work from it.`,
              userQuestion
                ? `已换回你自己的表述 —— 我会提交并用于证据绑定的是:“${esc(userQuestion)}”。随时可以再输入一句替换它。`
                : `可以，直接在输入框里写你的表述，我会基于你的文字继续。`,
            );
            renderThread(); renderAside(); renderChips();
            return;
          }
          if (tok === '@typemine') { pushUser(label); pushBot(`Of course — type your research question in the box below and I’ll frame it with you.`, `当然可以。请在下面输入你的研究问题，我会帮你整理成可执行框架。`); renderThread(); const inp = document.getElementById('gdInput'); if (inp) inp.focus(); return; }
          if (tok === '@openAgent') { pushUser(label); location.hash = '#agent'; return; }
          if (tok === '@reviewBlocked') { expandedStep = 'analysis'; renderThread(); jumpToStep('analysis'); return; }
          if (tok === '@reviewLocalRun') { openGuidedRunReview(selectedGuidedRun, label); return; }
          if (tok === '@activeExport') { pushUser(label); dataMode = 'real'; go('realConfirm', label); return; }
          if (tok === '@folderquick') { quickCreateGuidedStarterFolder(); return; }
          if (tok === '@foldernew') { pushUser(label || 'New / open study folder'); showGuidedDraftSetup('Guided Copilot draft'); return; }
          if (tok === '@hintN') { handleText('use 30 patients'); return; }
          go(tok, goEl.classList.contains('suggest-chip') ? label : null);
          return;
        }
        const guidedGoalEl = e.target.closest('[data-guided-goal]');
        if (guidedGoalEl) {
          chooseGuidedGoal(guidedGoalEl.dataset.guidedGoal, stripText(guidedGoalEl.textContent));
          return;
        }
        if (EXTRACT.handleClick(e.target)) return;
        if (REVIEW.handleClick(e.target)) return;
        if (e.target.closest('[data-ga-run]')) {
          runGuidedAgentPreflight();
          return;
        }
        if (e.target.closest('[data-ga-open-agent]')) {
          openGuidedAgentHandoff();
          return;
        }
        if (e.target.closest('[data-gi-api-continue]')) {
          IDEA.showGuidedIdeaSourceForm();
          return;
        }
        if (e.target.closest('[data-gi-api-back]')) {
          IDEA.showGuidedIdeaApiSetup();
          return;
        }
        const giSource = e.target.closest('[data-gi-source]');
        if (giSource && IDEA.state()) {
          IDEA.state().sourceType = giSource.dataset.giSource || 'manual';
          IDEA.state().error = null;
          IDEA.state().sourceEditorOpen = true;
          IDEA.state().allowNetwork = false;
          IDEA.clearGuidedIdeaOutputs(true);
          renderThread();
          scheduleGuidedSlotSave('set_idea_source_type');
          return;
        }
        if (e.target.closest('[data-gi-pdf-pick]')) {
          const fileInput = root.querySelector('[data-gi-pdf-file]');
          if (fileInput) fileInput.click();
          return;
        }
        if (e.target.closest('[data-gi-lit-browse]')) {
          IDEA.loadGuidedLiteratureBrowser(IDEA.state() && IDEA.state().literatureFolder);
          return;
        }
        if (e.target.closest('[data-gi-lit-scan]')) {
          IDEA.scanGuidedLiteratureFolder();
          return;
        }
        if (e.target.closest('[data-gi-discover]')) {
          IDEA.runGuidedIdeaDiscover();
          return;
        }
        const discoveryUse = e.target.closest('[data-gi-discovery-use]');
        if (discoveryUse) {
          IDEA.useGuidedIdeaDiscoveryCandidate(discoveryUse.dataset.giDiscoveryUse || 0);
          return;
        }
        const giProvider = e.target.closest('[data-gi-provider]');
        if (giProvider) {
          IDEA.selectProvider(giProvider.dataset.giProvider);
          IDEA.requestGuidedIdeaProviderStatus(true);
          renderThread();
          scheduleGuidedSlotSave('set_idea_provider');
          return;
        }
        if (e.target.closest('[data-gi-provider-config-toggle]')) {
          IDEA.toggleProviderConfig();
          renderThread();
          return;
        }
        if (e.target.closest('[data-gi-enable-ai]')) {
          IDEA.enableGuidedIdeaProvider();
          return;
        }
        if (e.target.closest('[data-gi-provider-save]')) {
          const card = e.target.closest('.gd-idea-card') || document;
          IDEA.saveGuidedIdeaProviderConfig(card);
          return;
        }
        if (e.target.closest('[data-gi-provider-refresh]')) {
          IDEA.requestGuidedIdeaProviderStatus(true);
          return;
        }
        if (e.target.closest('[data-lit-browser-close]')) {
          if (IDEA.literatureBrowser()) IDEA.literatureBrowser().open = false;
          renderThread();
          return;
        }
        const litShortcut = e.target.closest('[data-lit-browser-shortcut]');
        if (litShortcut) {
          const shortcuts = IDEA.literatureBrowser() && IDEA.literatureBrowser().data && Array.isArray(IDEA.literatureBrowser().data.shortcuts)
            ? IDEA.literatureBrowser().data.shortcuts
            : [];
          const row = shortcuts[Number(litShortcut.dataset.litBrowserShortcut || -1)];
          if (row && row.path) IDEA.loadGuidedLiteratureBrowser(row.path);
          return;
        }
        const litEntry = e.target.closest('[data-lit-browser-entry]');
        if (litEntry) {
          const entries = IDEA.literatureBrowser() && IDEA.literatureBrowser().data && Array.isArray(IDEA.literatureBrowser().data.entries)
            ? IDEA.literatureBrowser().data.entries
            : [];
          const row = entries[Number(litEntry.dataset.litBrowserEntry || -1)];
          if (row && row.path) IDEA.loadGuidedLiteratureBrowser(row.path);
          return;
        }
        if (e.target.closest('[data-lit-browser-up]')) {
          const parent = IDEA.literatureBrowser() && IDEA.literatureBrowser().data && IDEA.literatureBrowser().data.parent;
          if (parent) IDEA.loadGuidedLiteratureBrowser(parent);
          return;
        }
        if (e.target.closest('[data-lit-browser-use]')) {
          const path = (IDEA.literatureBrowser() && IDEA.literatureBrowser().data && IDEA.literatureBrowser().data.path) || (IDEA.literatureBrowser() && IDEA.literatureBrowser().path) || '';
          if (IDEA.state() && path) {
            IDEA.state().literatureFolder = path;
            IDEA.state().sourceType = 'literature_folder';
          }
          if (IDEA.literatureBrowser()) IDEA.literatureBrowser().open = false;
          renderThread();
          scheduleGuidedSlotSave('set_literature_folder');
          return;
        }
        if (e.target.closest('[data-gi-resolve]')) {
          IDEA.runGuidedIdeaResolve();
          return;
        }
        if (e.target.closest('[data-gi-mine]')) {
          IDEA.runGuidedIdeaMine();
          return;
        }
        if (e.target.closest('[data-gi-edit-source]')) {
          if (IDEA.state()) {
            IDEA.state().sourceEditorOpen = true;
            IDEA.state().error = null;
          }
          renderThread();
          scheduleGuidedSlotSave('edit_idea_source_again');
          return;
        }
        if (e.target.closest('[data-gi-confirm-data]')) {
          IDEA.confirmGuidedIdeaDataContext();
          return;
        }
        if (e.target.closest('[data-gi-plan]')) {
          IDEA.runGuidedIdeaPlan('plan');
          return;
        }
        if (e.target.closest('[data-gi-replan]')) {
          IDEA.runGuidedIdeaPlan('replan');
          return;
        }
        if (e.target.closest('[data-gi-prior]')) {
          IDEA.runGuidedIdeaPriorArt();
          return;
        }
        if (e.target.closest('[data-gi-handoff]')) {
          IDEA.runGuidedIdeaHandoff();
          return;
        }
        if (e.target.closest('[data-gi-project]')) {
          IDEA.runGuidedIdeaCreateProject();
          return;
        }
        const guidedHandoffEl = e.target.closest('[data-guided-handoff]');
        if (guidedHandoffEl) {
          runGuidedHandoff(guidedHandoffEl.dataset.guidedHandoff, guidedHandoffEl.dataset.target, stripText(guidedHandoffEl.textContent));
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
          if (a === 'strict') { cohortPhase = 'empty'; renderThread(); pushBot(`Trying “Sepsis-3 + age ≥ 80”…`, `正在尝试 “Sepsis-3 + age ≥ 80”…`); renderThread(); return; }
          if (a === 'loosen') { cohortPhase = 'normal'; renderThread(); pushBot(`Loosened back to the working cohort — ${patientN} stays match again.`, `已放宽回可用队列：现在匹配 ${patientN} 个 stay。`); renderThread(); return; }
          if (a === 'open') { openWorkspace(); return; }
          if (a === 'draft') { openDraft(); return; }
          if (a === 'signoff') {
            draftPhase = 'signed'; markThrough('draft', 'done'); setVal({ draft: 'unlocked' }); renderThread();
            try { localStorage.setItem('easyicu_study', JSON.stringify({ branch, mods, patientN, ts: Date.now() })); } catch (e) {}
            pushBot(
              `Signed off — the draft is unlocked and the full study is assembled. Open the workspace or start another.`,
              `已签署：草稿已解锁，完整研究已组装。你可以打开工作区，或开始另一个研究。`,
            );
            chips = []; renderThread(); renderChips();
            return;
          }
          return;
        }
        const pipelineToggle = e.target.closest('[data-gd-pipeline-toggle]');
        if (pipelineToggle) {
          guidedPipelineOpen = !guidedPipelineOpen;
          renderAside();
          return;
        }
        // exit / classic
        const openEl = e.target.closest('[data-open]');
        if (openEl) {
          const target = openEl.dataset.open;
          // Copilot -> classic exits must carry the collected study config as a
          // real prefill instead of dumping the user on a blank expert form.
          if (target === 'extraction' && guidedExtract && window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.set) {
            window.EU_GUIDED_HANDOFF.set({
              type: 'module_handoff', status: 'ready', goal: 'configure_extraction',
              target_route: 'extraction',
              prefill: guidedAgentHandoffPrefill(),
              config: guidedExtractionClassicConfig(),
              requires_user_confirm: true,
            });
          }
          location.hash = '#' + target;
          return;
        }
        // clickable Study panel step → jump to / edit that step
        const stEl = e.target.closest('[data-study]');
        if (stEl) { jumpToStep(stEl.dataset.study); return; }
        // sessions rail
        const projectOwner = window.EU_GUIDED_PROJECTS;
        const projectRailToggle = e.target.closest('[data-project-rail-toggle]');
        if (projectRailToggle && projectOwner && projectOwner.setProjectRailCollapsed) {
          const collapsed = !(projectOwner.isProjectRailCollapsed && projectOwner.isProjectRailCollapsed());
          projectOwner.setProjectRailCollapsed(collapsed);
          const main = document.querySelector('.gd-main');
          if (main) main.classList.toggle('gd-project-rail-collapsed', collapsed);
          return;
        }
        const contextAsideToggle = e.target.closest('[data-context-aside-toggle]');
        const panelOwner = window.EU_GUIDED_PANELS;
        if (contextAsideToggle && panelOwner && panelOwner.setContextAsideCollapsed) {
          const collapsed = !(panelOwner.isContextAsideCollapsed && panelOwner.isContextAsideCollapsed());
          panelOwner.setContextAsideCollapsed(collapsed, document.querySelector('.gd-main'));
          return;
        }
        const manageProjects = e.target.closest('[data-project-manage]');
        if (manageProjects && projectOwner && projectOwner.setProjectManagement) {
          projectOwner.setProjectManagement(!projectOwner.isProjectManagementActive());
          rerenderProjectRailKeepingScroll();
          return;
        }
        const removeSelectedProjects = e.target.closest('[data-remove-selected-projects]');
        if (removeSelectedProjects && projectOwner && projectOwner.selectedProjects) {
          const activeId = selectedGuidedDraft && selectedGuidedDraft.id;
          const rows = projectOwner.selectedProjects(localDraftRows(), activeId);
          if (rows.length) removeLocalGuidedDraft(rows);
          return;
        }
        const refreshDrafts = e.target.closest('[data-refreshdrafts]');
        if (refreshDrafts) { loadGuidedDrafts(true); return; }
        const removeDraftEl = e.target.closest('[data-remove-localdraft]');
        if (removeDraftEl) {
          const row = localDraftRows()[Number(removeDraftEl.dataset.removeLocaldraft || -1)];
          if (row) removeLocalGuidedDraft(row);
          return;
        }
        const localDraftEl = e.target.closest('[data-localdraft]');
        if (localDraftEl) {
          const row = localDraftRows()[Number(localDraftEl.dataset.localdraft || -1)];
          if (!row) return;
          if (projectOwner && projectOwner.isProjectManagementActive && projectOwner.isProjectManagementActive()) {
            if (!selectedGuidedDraft || row.id !== selectedGuidedDraft.id) {
              const activeId = selectedGuidedDraft && selectedGuidedDraft.id;
              const alreadySelected = projectOwner.selectedProjects(localDraftRows(), activeId).some(item => item.id === row.id);
              projectOwner.toggleProjectSelection(row, !alreadySelected);
              rerenderProjectRailKeepingScroll();
            }
            return;
          }
          selectedGuidedDraft = row;
          selectedGuidedRun = null;
          openGuidedProjectMemory(row, localDraftEl, 'draft');
          return;
        }
        if (e.target.closest('[data-newstudy]')) {
          guidedFolderMenuOpen = !guidedFolderMenuOpen;
          renderGuidedFolderControls();
          return;
        }
        if (e.target.closest('[data-toggle-known-projects]')) {
          guidedKnownProjectsOpen = !guidedKnownProjectsOpen;
          if (guidedKnownProjectsOpen) {
            loadGuidedDrafts(true);
          }
          renderGuidedFolderDialog();
          return;
        }
        if (e.target.closest('[data-refreshfolderchoices]')) {
          guidedKnownProjectsOpen = true;
          loadGuidedDrafts(true);
          return;
        }
        const knownProjectEl = e.target.closest('[data-known-project]');
        if (knownProjectEl) {
          const row = guidedKnownProjectRows()[Number(knownProjectEl.dataset.knownProject || -1)];
          if (!row) return;
          const box = knownProjectEl.closest('[data-draft-setup]');
          openExistingGuidedProject(row.project_dir, box);
          return;
        }
        if (e.target.closest('[data-browseprojectfolder]')) {
          const box = e.target.closest('[data-draft-setup]');
          const pathEl = box ? box.querySelector('[data-existing-project-dir]') : null;
          loadGuidedFolderBrowser(pathEl && pathEl.value);
          return;
        }
        if (e.target.closest('[data-browsedraftparent]')) {
          const box = e.target.closest('[data-draft-setup]');
          captureGuidedDraftDialogState(box);
          const parentEl = box ? box.querySelector('[data-draft-parent-dir]') : null;
          loadGuidedFolderBrowser((parentEl && parentEl.value) || guidedDraftParentDir || '~/easyicu/projects');
          return;
        }
        if (e.target.closest('[data-folder-browser-close]')) {
          guidedFolderBrowser.open = false;
          renderGuidedFolderDialog();
          return;
        }
        const browserShortcut = e.target.closest('[data-folder-browser-shortcut]');
        if (browserShortcut) {
          const shortcuts = guidedFolderBrowser.data && Array.isArray(guidedFolderBrowser.data.shortcuts)
            ? guidedFolderBrowser.data.shortcuts
            : [];
          const row = shortcuts[Number(browserShortcut.dataset.folderBrowserShortcut || -1)];
          if (row && row.path) loadGuidedFolderBrowser(row.path);
          return;
        }
        const browserEntry = e.target.closest('[data-folder-browser-entry]');
        if (browserEntry) {
          const entries = guidedFolderBrowser.data && Array.isArray(guidedFolderBrowser.data.entries)
            ? guidedFolderBrowser.data.entries
            : [];
          const row = entries[Number(browserEntry.dataset.folderBrowserEntry || -1)];
          if (row && row.path) loadGuidedFolderBrowser(row.path);
          return;
        }
        if (e.target.closest('[data-folder-browser-up]')) {
          const parent = guidedFolderBrowser.data && guidedFolderBrowser.data.parent;
          if (parent) loadGuidedFolderBrowser(parent);
          return;
        }
        if (e.target.closest('[data-folder-browser-use]')) {
          const box = e.target.closest('[data-draft-setup]');
          const path = (guidedFolderBrowser.data && guidedFolderBrowser.data.path) || guidedFolderBrowser.path;
          const parentEl = box ? box.querySelector('[data-draft-parent-dir]') : null;
          if (guidedFolderDialogMode === 'new' && parentEl) {
            captureGuidedDraftDialogState(box);
            if (path) {
              guidedDraftParentDir = path;
              parentEl.value = path;
            }
            guidedFolderBrowser.open = false;
            renderGuidedFolderDialog();
            return;
          }
          const pathEl = box ? box.querySelector('[data-existing-project-dir]') : null;
          if (pathEl && path) pathEl.value = path;
          if (pendingGuidedGoal && pendingGuidedGoal.goal === 'review_data') registerExistingExportForReview(path, box);
          else openExistingGuidedProject(path, box);
          return;
        }
        const reviewExportFolderEl = e.target.closest('[data-reviewexportfolder]');
        if (reviewExportFolderEl) {
          const box = reviewExportFolderEl.closest('[data-draft-setup]');
          const pathEl = box ? box.querySelector('[data-existing-project-dir]') : null;
          registerExistingExportForReview(pathEl && pathEl.value, box);
          return;
        }
        const openProjectFolderEl = e.target.closest('[data-openprojectfolder]');
        if (openProjectFolderEl) {
          const box = openProjectFolderEl.closest('[data-draft-setup]');
          const pathEl = box ? box.querySelector('[data-existing-project-dir]') : null;
          openExistingGuidedProject(pathEl && pathEl.value, box);
          return;
        }
        const createDraftEl = e.target.closest('[data-createdraft]');
        if (createDraftEl) {
          const box = createDraftEl.closest('[data-draft-setup]');
          const titleEl = box ? box.querySelector('[data-draft-title]') : null;
          const slugEl = box ? box.querySelector('[data-draft-slug]') : null;
          const parentEl = box ? box.querySelector('[data-draft-parent-dir]') : null;
          const title = (titleEl && titleEl.value || '').trim() || 'New local study';
          const slug = slugifyDraftFolder((slugEl && slugEl.value) || title);
          const parent = (parentEl && parentEl.value || '').trim();
          guidedFolderSeedTitle = title;
          guidedDraftFolderSlug = slug;
          guidedDraftParentDir = parent || guidedDraftParentDir || '~/easyicu/projects';
          createLocalGuidedDraft(title, slug, guidedDraftParentDir);
          return;
        }
        if (e.target.closest('[data-canceldraft]')) {
          pushBot(
            `No folder created. Use <strong>New / open study folder</strong> when you want to bind the conversation to a local project folder.`,
            `没有创建文件夹。需要把对话绑定到本地项目文件夹时，再使用 <strong>New / open study folder</strong>。`,
          );
          renderThread();
          return;
        }
      });

      shell.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (guidedDraftRemoval && !guidedDraftRemoval.busy) {
          closeGuidedDraftRemovalDialog();
          e.preventDefault();
          return;
        }
        if (guidedFolderMenuOpen) {
          guidedFolderMenuOpen = false;
          renderGuidedFolderControls();
          e.preventDefault();
        }
      });

      shell.addEventListener('input', (e) => {
        if (EXTRACT.handleInput(e.target)) return;
        const gaQuestion = e.target.closest('[data-ga-question]');
        if (gaQuestion && guidedAgent) {
          guidedAgent.question = gaQuestion.value;
          guidedAgent.error = null;
          scheduleGuidedSlotSave('edit_agent_question');
          return;
        }
        const giField = e.target.closest('[data-gi-field]');
        if (giField && IDEA.state()) {
          const key = giField.dataset.giField;
          const hadOutputs = !!(IDEA.state().resolved || IDEA.state().result || IDEA.state().prior || IDEA.state().handoff || IDEA.state().project);
          if (key) IDEA.state()[key] = giField.value;
          IDEA.state().error = null;
          if (key !== 'planEdits') {
            IDEA.state().sourceEditorOpen = true;
            IDEA.clearGuidedIdeaOutputs(true);
          } else {
            IDEA.state().handoff = null;
            IDEA.state().project = null;
          }
          if (key !== 'planEdits' && hadOutputs) {
            // Editing the source invalidates the stale result/feasibility/
            // handoff cards, so the thread must re-render. renderThread()
            // rebuilds the textarea/input being typed in, which drops focus
            // and caret mid-keystroke — restore both onto the same field.
            const selStart = giField.selectionStart;
            const selEnd = giField.selectionEnd;
            renderThread();
            const next = shell.querySelector(`[data-gi-field="${key}"]`);
            if (next) {
              next.focus();
              if (selStart != null && typeof next.setSelectionRange === 'function') {
                try { next.setSelectionRange(selStart, selEnd); } catch (_) { /* unsupported input type */ }
              }
            }
          }
          scheduleGuidedSlotSave('edit_idea_field');
          return;
        }
        const draftParent = e.target.closest('[data-draft-parent-dir]');
        if (draftParent) {
          guidedDraftParentDir = draftParent.value.trim();
          return;
        }
        const title = e.target.closest('[data-draft-title]');
        if (!title) return;
        guidedFolderSeedTitle = title.value.trim() || guidedFolderSeedTitle || 'New local study';
        const box = title.closest('[data-draft-setup]');
        const slug = box ? box.querySelector('[data-draft-slug]') : null;
        if (slug && !slug.dataset.edited) slug.value = slugifyDraftFolder(title.value);
      });
      shell.addEventListener('change', (e) => {
        const projectOwner = window.EU_GUIDED_PROJECTS;
        const selectedProject = e.target.closest('[data-select-localdraft]');
        if (selectedProject && projectOwner && projectOwner.toggleProjectSelection) {
          const row = localDraftRows()[Number(selectedProject.dataset.selectLocaldraft || -1)];
          if (row) projectOwner.toggleProjectSelection(row, !!selectedProject.checked);
          rerenderProjectRailKeepingScroll();
          return;
        }
        const selectAllProjects = e.target.closest('[data-select-all-projects]');
        if (selectAllProjects && projectOwner && projectOwner.selectAllProjects) {
          const activeId = selectedGuidedDraft && selectedGuidedDraft.id;
          projectOwner.selectAllProjects(localDraftRows(), activeId, !!selectAllProjects.checked);
          rerenderProjectRailKeepingScroll();
          return;
        }
        const removeProjectFolder = e.target.closest('[data-remove-project-folder]');
        if (removeProjectFolder && guidedDraftRemoval && !guidedDraftRemoval.busy) {
          guidedDraftRemoval.trashProjectFolder = !!removeProjectFolder.checked;
          guidedDraftRemoval.error = null;
          renderGuidedDraftRemovalDialog();
          return;
        }
        const giPdfFile = e.target.closest('[data-gi-pdf-file]');
        if (giPdfFile && IDEA.state()) {
          const file = giPdfFile.files && giPdfFile.files[0];
          IDEA.ingestGuidedIdeaPdfFile(file);
          giPdfFile.value = '';
          return;
        }
        const giNetwork = e.target.closest('[data-gi-network]');
        if (giNetwork && IDEA.state()) {
          IDEA.state().allowNetwork = !!giNetwork.checked;
          IDEA.state().error = null;
          IDEA.state().prior = null;
          renderThread();
          scheduleGuidedSlotSave('toggle_idea_network');
          return;
        }
        const slug = e.target.closest('[data-draft-slug]');
        if (!slug) return;
        slug.dataset.edited = 'true';
        slug.value = slugifyDraftFolder(slug.value);
        guidedDraftFolderSlug = slug.value;
      });

      // composer
      const input = root.querySelector('#gdInput');
      const send = root.querySelector('#gdSend');
      function handleTextLocal() {
        const v = input.value.trim();
        if (!v || busy) return;
        input.value = '';
        guidedComposerDraft = '';
        handleText(v);
      }
      send.addEventListener('click', handleTextLocal);
      input.addEventListener('input', () => { guidedComposerDraft = input.value; });
      input.addEventListener('keydown', (e) => { if (window.EU_COMPOSER_KEYBOARD.enterShouldSend(e)) { e.preventDefault(); handleTextLocal(); } });
    },
  };

  /* handle free text (from composer or hint chips) */
  function handleText(v) {
    if (busy) return;
    rememberUserQuestion(v);
    const conceptCode = findLocalConceptQuery(v);
    if (conceptCode) {
      answerConceptQuestion(v, conceptCode);
      return;
    }
    if (currentId === 'frontdoor' && IDEA.isGuidedIdeaIntent(v)) {
      if (requireGuidedProjectMemory('idea_mining', v)) return;
      IDEA.startGuidedIdeaFlow(v);
      IDEA.state().topic = v;
      renderThread();
      return;
    }
    if (currentId === 'frontdoor' && isGuidedExtractionIntent(v)) {
      if (requireGuidedProjectMemory('data_extraction', v)) return;
      EXTRACT.start(v);
      return;
    }
    if (currentId === 'frontdoor' && isGuidedReviewIntent(v)) {
      if (requireGuidedProjectMemory('review_data', v)) return;
      REVIEW.start(v);
      return;
    }
    if (currentId === 'frontdoor' && isGuidedAgentIntent(v)) {
      if (requireGuidedProjectMemory('run_agent', v)) return;
      startGuidedAgentFlow(v);
      return;
    }
    if (currentId === 'frontdoor') {
      // Unmatched free text at the front door: reflect it back and offer the four
      // real goals as a disambiguation, instead of a flat "pick a goal card" bounce.
      reflectGuidedFrontdoor(v);
      return;
    }
    if (autop && /\b(stop|pause|halt|cancel)\b/i.test(v)) { autop = false; pushUser(v); pushBot(`Autopilot paused — tap a suggestion to continue manually.`, `自动流程已暂停。你可以点一个建议继续手动推进。`); renderThread(); return; }
    const fn = parseIntent(v);
    if (fn) { fn(); return; }
    // fallback: advance the primary path of the current state, echoing the text
    const map = { frame: 'toData', toData: null, toCohort: 'toConcepts', toConcepts: 'toExtract', toReview: 'toRun', toFindings: null };
    const next = map[currentId];
    if (next) { go(next, v); return; }
    pushUser(v);
    // Free text we could not route is still the user correcting their own
    // study. Before the run is bound, it REPLACES the question of record —
    // echoing "I'll treat that as X" and then discarding it is how an AKI
    // question silently stayed a mortality question.
    const bindable = !liveAgentRun && runPhase !== 'done' && STEP_INDEX[expandedStep] <= STEP_INDEX.concepts;
    if (bindable && replaceUserQuestion(v)) {
      pushBot(
        `Updated the question of record to “<em>${esc(userQuestion)}</em>”. That is what I will submit and bind evidence to. I could not map it onto a preset branch, so the suggested wording below may still be off — tap <strong>Reframe</strong> if it is.`,
        `已把记录在案的研究问题更新为“<em>${esc(userQuestion)}</em>”。我提交并用于证据绑定的就是这一句。我没能把它映射到预设分支，所以下面的建议措辞可能仍不准确 —— 不对就点 <strong>重新表述</strong>。`,
      );
      renderThread(); renderAside(); renderChips();
      return;
    }
    if (dataMode === 'demo') {
      // Demo pipeline: the shortcut coaching only makes sense inside the seeded demo.
      pushBot(
        `I’ll treat that as “<em>${esc(v)}</em>”. In this guided demo I move step by step — tap a suggestion to continue, or say <strong>“why?”</strong>, <strong>“go back”</strong>, <strong>“use 30 patients”</strong>, or <strong>“run the whole demo”</strong>.`,
        `我会把它理解为“<em>${esc(v)}</em>”。在引导模式里我会一步一步推进；你可以点建议继续，或说 <strong>“为什么”</strong>、<strong>“返回”</strong>、<strong>“用 30 个患者”</strong>、<strong>“跑完整演示”</strong>。`,
      );
    } else {
      // Real, folder-bound flow: no demo shortcuts — just point back to this step's controls.
      pushBot(
        `Noted: “<em>${esc(v)}</em>”. Use the controls in the current card to adjust this step, or say <strong>“go back”</strong> to change an earlier decision.`,
        `已记录：“<em>${esc(v)}</em>”。可用当前卡片里的控件调整这一步，或说 <strong>“返回”</strong> 修改前面的决定。`,
      );
    }
    renderThread();
  }
  function reflectGuidedFrontdoor(v) {
    pushUser(v);
    guidedFrontdoorSeedText = v;
    thread.push({ bot: true, html: bi(
      `It sounds like you want to study “<em>${esc(v)}</em>”. I can take that four ways — pick the one that fits and I’ll carry your wording into it.`,
      `听起来你想研究“<em>${esc(v)}</em>”。我可以从四个方向推进 —— 选一个最合适的，我会把你的描述带进去。`,
    ) });
    thread.push({ bot: true, html: renderGuidedGoalCards() });
    chips = [];
    renderThread();
    renderChips();
  }

  function stripText(s) { return s.replace(/\s+/g, ' ').trim(); }
})();
