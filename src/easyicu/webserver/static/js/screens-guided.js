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
    predict: { q: bi(`Quick check before I build the plan — which mortality endpoint do you mean?`, `建计划前先确认一下：你说的死亡结局是哪一种？`), opts: [['In-hospital mortality', 'in-hospital'], ['28-day mortality', '28-day'], ['ICU mortality', 'ICU']] },
    crossdb: { q: bi(`How many databases should we compare?`, `这次要比较多少个数据库？`), opts: [['All six', 'all 6 databases'], ['A focused three', '3 databases'], ['Let me pick', 'a custom set']] },
    quality: { q: bi(`Should I audit everything, or focus on the modelling features?`, `要审计全部模块，还是只关注建模特征？`), opts: [['Everything (19 modules)', 'all 19 modules'], ['Modelling features only', 'the modelling features']] },
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
      hi: bi(
        `Got it — an <strong>extract-only</strong> run. I’ll stop once your cohort is resolved and packaged, and you leave with analysis-ready frames plus a reproducible manifest.`,
        `明白，这次走<strong>仅抽取</strong>。我会在队列解析并打包完成后停下，给你留下可分析的数据表和可复现 manifest。`,
      ),
    },
    review: {
      label: 'Extract + review', goal: 'review',
      chip: 'Data, then a visual review',
      hi: bi(
        `Good — <strong>extract &amp; review</strong>. I’ll pull the data and prepare a quick visual review, then hand you a populated workspace. No agent run unless you ask.`,
        `好的，走<strong>抽取 + 审阅</strong>。我会读取数据、生成快速可视化审阅，再把已填充的工作区交给你；除非你确认，不会启动 Agent run。`,
      ),
    },
    full: {
      label: 'Full study', goal: 'draft',
      chip: 'All the way to a review-ready draft',
      hi: bi(
        `The full ride — <strong>extract → review → analyse → review-ready draft</strong>. Everything runs locally and the draft stays locked until checks pass.`,
        `完整流程：<strong>抽取 → 审阅 → 分析 → 待核验草稿</strong>。所有步骤都在本机运行，检查通过前草稿保持锁定。`,
      ),
    },
  };

  /* ============== runtime state ============== */
  let branch, depth, dataMode, mods, cohortPhase, extractPhase, runPhase, draftPhase;
  let thread, chips, busy, expandedStep, whyOpen, autop, patientN, clarified, outputsReady, diffExpanded, liveAgentRun, workspaceSnapshot, workspaceSnapshotPath, guidedExtract, guidedReview, guidedAgent, guidedIdea, guidedIdeaProvider, guidedLiteratureBrowser;
  let guidedDrafts = { loading: false, error: null, data: null };
  let guidedCopilot = { loading: false, error: null, session: null, last: null };
  let selectedGuidedRun = null;
  let selectedGuidedDraft = null;
  let pendingGuidedGoal = null;
  let guidedFolderMenuOpen = false;
  let guidedFolderDialogMode = null;
  let guidedFolderSeedTitle = 'New local study';
  let guidedDraftFolderSlug = '';
  let guidedDraftParentDir = '~/easyicu/projects';
  let guidedFolderBrowser = { open: false, loading: false, error: null, data: null, path: '' };
  let guidedKnownProjectsOpen = false;
  let guidedSlotSaveTimer = null;
  let studyParams;   // dynamic params extracted from clarify answers + free text

  const DEFAULT_MODS = ['Demographics', 'Vital signs', 'Lab — Chemistry', 'SOFA-2 scores', 'Sepsis-3 (SOFA-2)', 'Outcome'];
  const GUIDED_EXTRACT_WINDOW_HOURS = 24 * 30;
  const GUIDED_EXTRACT_MODULES = [
    ['demographics', 'Demographics', '人口统计', 6, true],
    ['vitals', 'Vital signs', '生命体征', 11, true],
    ['chemistry', 'Lab — Chemistry', '实验室-生化', 30, true],
    ['sofa2_score', 'SOFA-2 scores', 'SOFA-2 评分', 7, true],
    ['sepsis3_sofa2', 'Sepsis-3 (SOFA-2)', 'Sepsis-3 (SOFA-2)', 1, true],
    ['outcome', 'Outcome', '结局', 10, true],
    ['sofa1_score', 'SOFA-1 scores', 'SOFA-1 评分', 7, false],
    ['sepsis3_sofa1', 'Sepsis-3 (SOFA-1)', 'Sepsis-3 (SOFA-1)', 1, false],
    ['sepsis_shared', 'Sepsis shared', 'Sepsis 共享概念', 5, false],
    ['respiratory', 'Respiratory', '呼吸系统', 15, false],
    ['ventilator', 'Ventilator', '呼吸机参数', 12, false],
    ['blood_gas', 'Blood gas', '血气分析', 9, false],
    ['hematology', 'Lab — Hematology', '实验室-血液学', 22, false],
    ['vasopressors', 'Vasopressors', '血管活性药物', 17, false],
    ['medications', 'Other medications', '其他药物', 49, false],
    ['renal', 'Renal & urine output', '肾脏与尿量', 22, false],
    ['neurological', 'Neurological', '神经系统', 11, false],
    ['circulatory', 'Circulatory', '循环系统', 3, false],
    ['other_scores', 'Other scores', '其他评分', 9, false],
  ];
  const GUIDED_CORE_MODULES = GUIDED_EXTRACT_MODULES.filter(m => m[4]).map(m => m[0]);
  const GUIDED_COHORT_PRESETS = [
    ['all_icu', 'All ICU stays', '全部 ICU 住院', 'Broad denominator, no diagnosis filter.', '宽队列，不预设诊断筛选。'],
    ['adult_first', 'Adult first ICU stay', '成年首次 ICU', 'Default denominator for most extraction workflows.', '多数抽取流程的默认分母。'],
    ['sepsis3', 'Sepsis-3 / suspected infection', 'Sepsis-3 / 疑似感染', 'Uses Sepsis concepts when available; ICD is not prefilled.', '可用时使用 Sepsis 概念；不会预填 ICD。'],
    ['aki', 'AKI / renal dysfunction', 'AKI / 肾功能异常', 'Renal cohort starting point.', 'AKI 研究的肾功能队列起点。'],
    ['ventilation', 'Mechanical ventilation', '机械通气', 'Ventilator exposure cohort starting point.', '机械通气暴露队列起点。'],
    ['vasopressor', 'Vasopressor exposure', '血管活性药物暴露', 'Shock or pressor cohort starting point.', '休克/升压药队列起点。'],
    ['respiratory', 'Respiratory failure', '呼吸衰竭', 'Respiratory support and blood-gas focused cohort.', '呼吸支持与血气相关队列。'],
  ];
  function reset() {
    branch = 'predict'; depth = 'full'; dataMode = 'demo'; mods = DEFAULT_MODS.slice();
    cohortPhase = 'normal'; extractPhase = 'run'; runPhase = 'run'; draftPhase = 'gate';
    thread = []; chips = []; busy = false; expandedStep = 'question'; whyOpen = {}; autop = false; patientN = 10; clarified = null; outputsReady = false; diffExpanded = false; liveAgentRun = null; workspaceSnapshot = null; workspaceSnapshotPath = null; guidedExtract = null; guidedReview = null; guidedAgent = null; guidedIdea = null; guidedLiteratureBrowser = null;
    pendingGuidedGoal = null;
    guidedFolderMenuOpen = false;
    guidedFolderDialogMode = null;
    guidedFolderSeedTitle = 'New local study';
    guidedDraftFolderSlug = '';
    guidedDraftParentDir = '~/easyicu/projects';
    guidedFolderBrowser = { open: false, loading: false, error: null, data: null, path: '' };
    guidedKnownProjectsOpen = false;
    studyParams = { outcome: 'In-hospital mortality', window: 'full available window', exposure: 'lactate', scope: 'all 19 modules', caught: null };
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
  function fmtFixed(v, digits, fallback) {
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(digits == null ? 1 : digits).replace(/\.0+$/, '') : (fallback || 'n/a');
  }
  function fmtP(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 'n/a';
    if (n < 0.001) return '<0.001';
    return n.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }
  function compactPath(value) {
    const text = String(value || '');
    if (!text) return '';
    const home = (window.EU_SETTINGS && window.EU_SETTINGS.about && window.EU_SETTINGS.about.home) || '';
    if (home && text.startsWith(home + '/')) return '~/' + text.slice(home.length + 1);
    const match = text.match(/^\/Users\/[^/]+\/(.+)$/);
    return match ? '~/' + match[1] : text;
  }
  function compactHash(value) {
    const text = String(value || '');
    if (!text) return '';
    return text.length > 14 ? `${text.slice(0, 8)}...${text.slice(-6)}` : text;
  }
  function fmtRunTime(value) {
    if (!value) return '';
    const d = new Date(String(value));
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
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
  function attr(s) { return esc(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }
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
        pushBot(
          `Recognized <strong>${esc(activeExportLabel())}</strong> — concept map verified. Files stay on your machine.`,
          `已识别 <strong>${esc(activeExportLabel())}</strong>，概念映射已验证。文件仍留在你的机器上。`,
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
    pushBot(
      `Run complete — registry-backed preflight artifacts were written locally and logged to the evidence ledger. <span style="color:var(--ink-4);">No patient rows were persisted and no external model call was made.</span>`,
      `运行完成：registry-backed 预检 artifacts 已写入本地，并记录到 evidence ledger。<span style="color:var(--ink-4);">没有持久化患者行，也没有外部模型调用。</span>`,
    );
    thread.push({ diff: true });
    renderThread(); renderAside();
    pushBot(
      `I can open this in Agent Projects now. Manuscript claims remain <strong>locked</strong> until human sign-off.`,
      `现在可以在 Agent Projects 中打开它。人工签署前，稿件 claims 仍保持<strong>锁定</strong>。`,
    );
    chips = []; renderThread();
    go('toFindings');
  }

  function failLivePipeline(error) {
    liveAgentRun = { active: false, result: null, error: error };
    runPhase = 'run';
    const p = document.getElementById('gdRunPill'); if (p) p.outerHTML = '<span class="pill bad" id="gdRunPill"><span class="dot"></span>Failed closed</span>';
    pushBot(
      `The run failed closed: <span class="mono">${esc(error)}</span>`,
      `这次 run 已 fail-closed：<span class="mono">${esc(error)}</span>`,
    );
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
          <div class="eyebrow" style="margin:0 0 8px;">Evidence checks</div>
          <div class="checks">
            ${checks.map(c => `<div class="check-row ${c.passed ? 'ok' : 'pending'}"><span class="check-mk">${c.passed ? icon('check', 11, 2.8) : icon('clock', 11)}</span><span style="font-size:11.75px;color:${c.passed ? 'var(--ink)' : 'var(--ink-3)'};">${esc(c.label || c.id)}</span><span class="grow"></span><span class="mono" style="font-size:10px;color:${c.passed ? 'var(--ok)' : 'var(--ink-4)'};">${c.passed ? 'passed' : 'pending'}</span></div>`).join('')}
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
  function guidedModuleConceptCount(key, fallback) {
    const groups = window.EU_CATALOG && window.EU_CATALOG.groupConcepts;
    const members = groups && groups[key];
    return Array.isArray(members) ? members.length : (fallback || 0);
  }
  function guidedSelectedConceptCount() {
    if (!guidedExtract) return 0;
    return GUIDED_EXTRACT_MODULES
      .filter(m => guidedExtract.modules.includes(m[0]))
      .reduce((sum, m) => sum + guidedModuleConceptCount(m[0], m[3]), 0);
  }
  function guidedExtractionCohortContract() {
    const preset = guidedExtract && guidedExtract.cohort ? guidedExtract.cohort : 'adult_first';
    return {
      preset,
      age_min: preset === 'adult_first' ? 18 : 0,
      age_max: 100,
      min_icu_los_hours: 0,
      observation_window_hours: GUIDED_EXTRACT_WINDOW_HOURS,
      exclude_readmissions: preset === 'adult_first',
      icd_enabled: false,
      icd_include: [],
      icd_exclude: [],
    };
  }
  function resetGuidedExtractionState() {
    guidedExtract = {
      path: '',
      scan: null,
      scanError: null,
      scanning: false,
      cohort: 'adult_first',
      modules: GUIDED_EXTRACT_MODULES.map(m => m[0]),
      format: 'parquet',
      merge: false,
      maxPatients: 500,
      running: false,
      jobId: null,
      progress: null,
      result: null,
      error: null,
      registered: false,
    };
  }
  function sourceReadyForGuidedExtraction() {
    const scan = guidedExtract && guidedExtract.scan;
    return !!(guidedExtract && guidedExtract.path && scan && scan.ok && scan.ready && scan.source !== 'module');
  }
  function guidedExtractionStatusText() {
    if (!guidedExtract) return '';
    if (guidedExtract.scanning) return t('Analyzing folder structure...', '正在识别文件夹结构...');
    if (guidedExtract.scanError) return esc(guidedExtract.scanError);
    if (guidedExtract.scan && guidedExtract.scan.source === 'module') {
      return t('This is already an EasyICU module export. Register it for review instead of extracting again.', '这是已有 EasyICU 模块导出。应注册后审阅，不需要再次抽取。');
    }
    if (guidedExtract.scan && !guidedExtract.scan.ready) {
      return t('Folder was recognized but is not extraction-ready yet. Use Advanced classic settings for the one-time conversion path.', '已识别该文件夹，但尚未达到可直接抽取状态。请用高级经典设置走一次性转换。');
    }
    if (guidedExtract.running && guidedExtract.progress) {
      const p = guidedExtract.progress;
      const msg = p.message || p.phase || 'running';
      const cur = p.current != null && p.total ? ` · ${p.current}/${p.total}` : '';
      return esc(msg + cur);
    }
    if (guidedExtract.running) return t('Starting extraction job...', '正在启动抽取任务...');
    if (guidedExtract.error) return esc(guidedExtract.error);
    if (guidedExtract.result) return t('Extraction complete. Output registered as the active local export.', '抽取完成。输出已注册为 active 本地 export。');
    return sourceReadyForGuidedExtraction()
      ? t('Ready to run locally. Nothing is uploaded.', '可以在本机运行。不会上传数据。')
      : t('Paste or choose a local ICU data folder, then analyze it before running.', '先粘贴或选择本机 ICU 数据文件夹，然后识别目录再运行。');
  }
  function renderGuidedExtractionCard() {
    if (!guidedExtract) resetGuidedExtractionState();
    const ready = sourceReadyForGuidedExtraction();
    const selected = guidedExtract.modules.length;
    const concepts = guidedSelectedConceptCount();
    const scan = guidedExtract.scan || {};
    const sourceMeta = scan.ok
      ? `${esc(scan.db || 'Unknown')} · ${esc(scan.source || 'source')} · ${fmtInt(scan.tables, 'n/a')} tables · ${fmtInt(scan.modules, 'n/a')} modules`
      : t('No path is prefilled because every user machine is different. Paste or choose a local ICU folder.', '不会预填路径，因为每台用户电脑都不同。请粘贴或选择本机 ICU 文件夹。');
    const progressPct = guidedExtract.progress && guidedExtract.progress.total
      ? Math.max(0, Math.min(100, Math.round((Number(guidedExtract.progress.current || 0) / Number(guidedExtract.progress.total || 1)) * 100)))
      : (guidedExtract.result ? 100 : 0);
    return `
      <div class="gd-x-card" data-guided-extraction-card>
        <div class="gdx-head">
          <span class="gdx-ico">${icon('extract', 15)}</span>
          <div>
            <strong>${t('Prepare data inside Copilot', '在 Copilot 内准备/抽取数据')}</strong>
            <span>${t('Same backend as Classic Data Extraction: cohort, modules, Parquet export, and local job progress.', '复用经典数据抽取同一个后端：队列、模块、Parquet 导出和本地 job 进度。')}</span>
          </div>
        </div>
        <div class="gdx-source ${ready ? '' : 'blocked'}">
          <span>${icon(ready ? 'check' : 'shield', 12)}</span>
          <div><strong>${t('Local source', '本地数据源')}</strong><small>${sourceMeta}</small></div>
        </div>
        <div class="gdx-pathrow">
          <label>
            <span>${t('Data folder path', '数据文件夹路径')}</span>
            <input data-gx-path value="${attr(guidedExtract.path || '')}" placeholder="${attr(t('Paste or browse to a local ICU folder', '粘贴或选择本机 ICU 文件夹'))}" />
          </label>
          <button type="button" class="btn primary" data-gx-analyze ${guidedExtract.scanning ? 'disabled' : ''}>${icon('search', 13)} ${t('Analyze folder', '识别目录')}</button>
        </div>
        <div class="gdx-section">
          <div class="gdx-label">${t('Cohort preset', '队列预设')}</div>
          <div class="gdx-presets">
            ${GUIDED_COHORT_PRESETS.map(([key, en, zh, den, dzh]) => `
              <button type="button" class="gdx-preset ${guidedExtract.cohort === key ? 'on' : ''}" data-gx-cohort="${attr(key)}">
                <strong>${t(en, zh)}</strong><span>${t(den, dzh)}</span>
              </button>
            `).join('')}
          </div>
          <div class="gdx-note">${t('Observation window defaults to full available data with a 30-day cap, not first 24h.', '观察窗默认使用全可用数据（30 天上限），不是前 24 小时。')}</div>
        </div>
        <div class="gdx-section">
          <div class="gdx-row">
            <div><div class="gdx-label">${t('Feature modules', '特征模块')}</div><small>${selected} modules · ${concepts} concepts</small></div>
            <div class="gdx-tools">
              <button type="button" class="btn sm" data-gx-module-set="all">${icon('check', 12)} ${t('Select all', '全选')}</button>
              <button type="button" class="btn sm" data-gx-module-set="none">${icon('x', 12)} ${t('Clear', '清空')}</button>
              <button type="button" class="btn sm" data-gx-module-set="core">${icon('refresh', 12)} ${t('Core 6', '核心 6')}</button>
            </div>
          </div>
          <div class="gdx-modgrid">
            ${GUIDED_EXTRACT_MODULES.map(([key, en, zh, fallback]) => {
              const on = guidedExtract.modules.includes(key);
              return `<button type="button" class="gdx-module ${on ? 'on' : ''}" data-gx-module="${attr(key)}">
                <span class="mk">${on ? icon('check', 10, 3) : ''}</span><strong>${t(en, zh)}</strong><span>${guidedModuleConceptCount(key, fallback)}</span>
              </button>`;
            }).join('')}
          </div>
        </div>
        <div class="gdx-section compact">
          <div class="gdx-row">
            <div><div class="gdx-label">${t('Export', '导出')}</div><small>${t('Parquet is the default. Each run creates a timestamped folder with README.md and _manifest.json.', '默认 Parquet。每次运行创建带时间戳的文件夹，并写入 README.md 和 _manifest.json。')}</small></div>
            <div class="gdx-seg" role="group" aria-label="Export format">
              ${['parquet', 'csv', 'excel'].map(fmt => `<button type="button" class="${guidedExtract.format === fmt ? 'on' : ''}" data-gx-format="${fmt}">${fmt === 'parquet' ? 'Parquet' : fmt.toUpperCase()}</button>`).join('')}
            </div>
          </div>
          <div class="gdx-row slim">
            <span>${t('Cohort size', '队列规模')}</span>
            <div class="gdx-seg" role="group" aria-label="Cohort size">
              <button type="button" class="${guidedExtract.maxPatients === 500 ? 'on' : ''}" data-gx-max="500">500 safety cap</button>
              <button type="button" class="${guidedExtract.maxPatients === null ? 'on' : ''}" data-gx-max="all">${t('All stays', '全量 stays')}</button>
            </div>
          </div>
        </div>
        <div class="gdx-status ${guidedExtract.error ? 'bad' : guidedExtract.result ? 'ok' : ''}">
          <span>${icon(guidedExtract.error ? 'x' : guidedExtract.result ? 'check' : 'shield', 12)}</span>
          <div><strong>${guidedExtractionStatusText()}</strong>${guidedExtract.jobId ? `<small>job ${esc(guidedExtract.jobId)}</small>` : ''}</div>
        </div>
        ${(guidedExtract.running || guidedExtract.result) ? `<div class="gdx-bar"><span style="width:${progressPct}%"></span></div>` : ''}
        ${guidedExtract.result ? `<div class="gdx-result">
          <span>${t('Output folder', '输出文件夹')}</span>
          <code>${esc(compactPath(guidedExtract.result.out_dir || guidedExtract.result.path || ''))}</code>
          <span>${t('Rows', '行数')}</span><strong>${fmtInt(guidedExtract.result.total_rows, 'n/a')}</strong>
          <span>${t('Files', '文件')}</span><strong>${fmtInt(guidedExtract.result.files_written || guidedExtract.result.files, 'n/a')}</strong>
        </div>` : ''}
        <div class="gdx-actions">
          <button type="button" class="btn primary" data-gx-run ${!ready || !selected || guidedExtract.running ? 'disabled' : ''}>${icon('play', 13)} ${t('Run extraction here', '在这里开始抽取')}</button>
          ${scan.ok && scan.source === 'module' ? `<button type="button" class="btn primary" data-gx-use-export>${icon('check', 13)} ${t('Register this export', '注册这个导出')}</button>` : ''}
          <button type="button" class="btn" data-open="extraction">${t('Advanced classic settings', '打开高级经典设置')}</button>
          ${guidedExtract.result ? `<button type="button" class="btn" data-open="patient">${t('Review export', '审阅导出结果')}</button>` : ''}
        </div>
      </div>`;
  }
  function startGuidedExtractionFlow(label) {
    if (label) pushUser(label);
    resetGuidedExtractionState();
    dataMode = 'real';
    setVal({ data: 'choose local folder', concepts: 'all modules', extract: 'inline Copilot' });
    markThrough('extract', 'active');
    thread.push({ bot: true, html: bi(
      `We can do the core extraction flow here in Copilot. Choose a local ICU data folder, I’ll scan it first, then start the same local extraction job Classic uses.`,
      `核心数据抽取可以直接在 Copilot 里完成。先选择本机 ICU 数据文件夹，我会先识别目录，再启动和经典视图相同的本地抽取任务。`,
    ) });
    thread.push({ guidedExtraction: true });
    chips = [];
    renderThread();
    renderChips();
    scheduleGuidedSlotSave('start_extraction');
  }
  function updateGuidedExtractionModules(mode) {
    if (!guidedExtract) return;
    if (mode === 'all') guidedExtract.modules = GUIDED_EXTRACT_MODULES.map(m => m[0]);
    else if (mode === 'core') guidedExtract.modules = GUIDED_CORE_MODULES.slice();
    else if (mode === 'none') guidedExtract.modules = [];
  }
  function scanGuidedExtractionPath() {
    if (!guidedExtract || guidedExtract.scanning) return;
    const path = String(guidedExtract.path || '').trim();
    if (!path) {
      guidedExtract.scan = null;
      guidedExtract.scanError = 'Choose or paste a local folder path first.';
      guidedExtract.error = null;
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.scanPath) {
      guidedExtract.scan = null;
      guidedExtract.scanError = 'Folder scan API is unavailable.';
      renderThread();
      return;
    }
    guidedExtract.scanning = true;
    guidedExtract.scan = null;
    guidedExtract.scanError = null;
    guidedExtract.error = null;
    renderThread();
    window.EU_API.scanPath(path, null).then(r => {
      guidedExtract.scanning = false;
      if (r && r.ok) {
        guidedExtract.path = r.path || path;
        guidedExtract.scan = r;
        guidedExtract.scanError = null;
        setVal({ data: (r.db || 'local ICU') + ' · ' + (r.source || 'source') });
      } else {
        guidedExtract.scan = r || null;
        guidedExtract.scanError = (r && (r.error || r.reason)) || 'Could not recognize this folder.';
      }
      renderThread();
      scheduleGuidedSlotSave('scan_extraction_folder');
    }).catch(err => {
      guidedExtract.scanning = false;
      guidedExtract.scan = null;
      guidedExtract.scanError = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('scan_extraction_folder_error');
    });
  }
  function registerGuidedModuleExport() {
    if (!guidedExtract || !guidedExtract.path || !window.EU_API || !window.EU_API.registerWorkspaceSource) return;
    guidedExtract.error = null;
    window.EU_API.registerWorkspaceSource(guidedExtract.path, { active: true, crossdb: true, label: 'Guided selected export' })
      .then(() => {
        guidedExtract.result = { out_dir: guidedExtract.path, total_rows: null, files_written: null };
        guidedExtract.registered = true;
        setVal({ data: 'registered export', extract: 'already exported' });
        markThrough('review', 'active');
        renderThread();
        scheduleGuidedSlotSave('register_module_export');
      })
      .catch(err => {
        guidedExtract.error = err.message || String(err);
        renderThread();
        scheduleGuidedSlotSave('register_module_export_error');
      });
  }
  function runGuidedExtractionJob() {
    if (!guidedExtract || guidedExtract.running) return;
    if (!sourceReadyForGuidedExtraction()) {
      guidedExtract.error = 'Analyze a prepared local ICU data folder before running extraction.';
      renderThread();
      return;
    }
    if (!guidedExtract.modules.length) {
      guidedExtract.error = 'Select at least one feature module before running.';
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.startExtractionJob || !window.EventSource) {
      guidedExtract.error = 'Extraction backend or browser event stream is unavailable.';
      renderThread();
      return;
    }
    guidedExtract.running = true;
    guidedExtract.error = null;
    guidedExtract.result = null;
    guidedExtract.progress = null;
    renderThread();
    const scan = guidedExtract.scan || {};
    window.EU_API.startExtractionJob({
      path: guidedExtract.path,
      database: scan.db_key || 'miiv',
      modules: guidedExtract.modules.slice(),
      format: guidedExtract.format,
      merge: guidedExtract.merge,
      max_patients: guidedExtract.maxPatients,
      cohort: guidedExtractionCohortContract(),
    }).then(r => {
      guidedExtract.jobId = r.job_id;
      renderThread();
      scheduleGuidedSlotSave('start_extraction_job');
      const es = new EventSource('/api/jobs/' + encodeURIComponent(r.job_id) + '/events');
      es.onmessage = ev => {
        let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
        if (m.type === 'progress') {
          guidedExtract.progress = m;
          renderThread();
          return;
        }
        if (m.type === 'end') {
          try { es.close(); } catch (e) {}
          guidedExtract.running = false;
          if (m.status === 'done') {
            guidedExtract.result = m.result || {};
            window.EU_LAST_EXPORT = guidedExtract.result;
            const out = guidedExtract.result.out_dir;
            if (out && window.EU_API && window.EU_API.registerWorkspaceSource) {
              window.EU_API.registerWorkspaceSource(out, { active: true, crossdb: true, label: 'Guided export' })
                .then(() => { guidedExtract.registered = true; renderAside(); })
                .catch(err => { console.warn('[EasyICU] guided export registry update failed:', err); });
            }
            setVal({ extract: 'done', data: 'Guided export' });
            markThrough('review', 'active');
          } else {
            guidedExtract.error = (m.error && (m.error.message || m.error.code)) || m.status || 'Extraction failed.';
          }
          renderThread();
          scheduleGuidedSlotSave('finish_extraction_job');
        }
      };
      es.onerror = () => {
        try { es.close(); } catch (e) {}
        guidedExtract.running = false;
        guidedExtract.error = 'Extraction event stream stopped before completion.';
        renderThread();
        scheduleGuidedSlotSave('extraction_event_stream_error');
      };
    }).catch(err => {
      guidedExtract.running = false;
      guidedExtract.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('start_extraction_job_error');
    });
  }

  /* ============== inline native review: patient + cohort + KM ============== */
  function resetGuidedReviewState() {
    guidedReview = {
      loading: false,
      error: null,
      patient: null,
      cohort: null,
      selectedRef: null,
    };
  }
  function guidedSourceLabel(payload) {
    const source = payload && payload.source ? payload.source : activeExportSource();
    if (!source) return t('No active export', '没有 active export');
    const label = source.label || source.database || 'active export';
    const summary = (payload && payload.summary) || source.summary || {};
    const parts = [];
    if (summary.entities != null || summary.stays != null) parts.push(`${fmtInt(summary.entities != null ? summary.entities : summary.stays)} entities`);
    if (summary.modules != null) parts.push(`${fmtInt(summary.modules)} modules`);
    return `${label}${parts.length ? ' · ' + parts.join(' · ') : ''}`;
  }
  function startGuidedReviewFlow(label) {
    if (label) pushUser(label);
    dataMode = 'real';
    resetGuidedReviewState();
    setVal({ data: activeExportLabel(), review: 'inline Copilot' });
    markThrough('review', 'active');
    thread.push({ bot: true, html: bi(
      `I’ll review the active local export here: patient drilldown, cohort summary, feature coverage, and KM/log-rank when the export has time-to-event fields.`,
      `我会直接在这里审阅 active 本地导出：患者概览、队列汇总、特征覆盖，以及在导出具备 time-to-event 字段时显示 KM/log-rank。`,
    ) });
    thread.push({ guidedReview: true });
    chips = [];
    renderThread();
    renderChips();
    scheduleGuidedSlotSave('start_review');
    loadGuidedReviewData();
  }
  function loadGuidedReviewData(entityRef) {
    if (!guidedReview) resetGuidedReviewState();
    if (!window.EU_API || !window.EU_API.loadPatientReviewDrilldown || !window.EU_API.loadCohortReviewSummary) {
      guidedReview.loading = false;
      guidedReview.error = 'Review APIs are unavailable.';
      renderThread();
      return;
    }
    guidedReview.loading = true;
    guidedReview.error = null;
    if (entityRef) guidedReview.selectedRef = entityRef;
    renderThread();
    const body = guidedReview.selectedRef ? { entity_ref: guidedReview.selectedRef } : {};
    Promise.allSettled([
      window.EU_API.loadPatientReviewDrilldown(body),
      window.EU_API.loadCohortReviewSummary({}),
    ]).then(([patientResult, cohortResult]) => {
      guidedReview.loading = false;
      const patientOk = patientResult.status === 'fulfilled' && patientResult.value && patientResult.value.ok;
      const cohortOk = cohortResult.status === 'fulfilled' && cohortResult.value && cohortResult.value.ok;
      guidedReview.patient = patientOk ? patientResult.value : null;
      guidedReview.cohort = cohortOk ? cohortResult.value : null;
      if (guidedReview.patient && guidedReview.patient.selected) guidedReview.selectedRef = guidedReview.patient.selected.ref;
      if (!patientOk && !cohortOk) {
        const pErr = patientResult.status === 'rejected' ? patientResult.reason : (patientResult.value && (patientResult.value.error || patientResult.value.reason));
        const cErr = cohortResult.status === 'rejected' ? cohortResult.reason : (cohortResult.value && (cohortResult.value.error || cohortResult.value.reason));
        guidedReview.error = (pErr && (pErr.message || String(pErr))) || (cErr && (cErr.message || String(cErr))) || 'No active registered export is available.';
      }
      if (guidedReview.cohort && guidedReview.cohort.summary) {
        setVal({
          cohort: `${fmtInt(guidedReview.cohort.summary.cohort_size || guidedReview.cohort.summary.entities)} entities`,
          review: 'cohort + KM audit',
        });
      }
      renderThread();
      renderAside();
      scheduleGuidedSlotSave('load_review_data');
    }).catch(err => {
      guidedReview.loading = false;
      guidedReview.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('load_review_data_error');
    });
  }
  function guidedMetricCard(label, value, sub) {
    return `<div class="gdr-metric"><span>${esc(label)}</span><strong>${esc(value == null ? 'n/a' : value)}</strong>${sub ? `<small>${esc(sub)}</small>` : ''}</div>`;
  }
  function renderGuidedPatientPanel(patient) {
    if (!patient) {
      return `<div class="gdr-panel blocked"><strong>${t('Patient drilldown unavailable', '患者 drilldown 不可用')}</strong><span>${t('Select or register an active EasyICU export first.', '请先选择或注册一个 active EasyICU 导出。')}</span></div>`;
    }
    const summary = patient.summary || {};
    const selected = patient.selected || {};
    const demo = selected.demographics || {};
    const scores = selected.scores || {};
    const outcomes = selected.outcomes || {};
    const entities = (patient.entities || []).slice(0, 5);
    const lanes = (patient.time_lanes || []).filter(row => row.status !== 'unavailable');
    return `
      <div class="gdr-panel">
        <div class="gdr-panel-head">
          <div><span class="gdx-label">${t('Patient drilldown', '患者 drilldown')}</span><strong>${esc(selected.label || 'Entity')}</strong></div>
          <div class="gdr-entity-pick">
            ${entities.map(row => `<button type="button" class="${row.ref === selected.ref ? 'on' : ''}" data-gr-entity="${attr(row.ref)}">${esc(row.label || row.ref)}</button>`).join('')}
          </div>
        </div>
        <div class="gdr-metrics">
          ${guidedMetricCard(t('Entities', '实体数'), fmtInt(summary.entities))}
          ${guidedMetricCard(t('Age', '年龄'), fmtNum(demo.age, 'n/a'), demo.sex || '')}
          ${guidedMetricCard(t('Outcome', '结局'), outcomes.status || 'Unknown', outcomes.icu_los_days != null ? `${fmtNum(outcomes.icu_los_days)} ICU days` : '')}
          ${guidedMetricCard('SOFA-2', fmtNum(scores.sofa2_max, 'n/a'), scores.sepsis3_sofa2 == null ? '' : `sepsis ${scores.sepsis3_sofa2 ? 'yes' : 'no'}`)}
        </div>
        <div class="gdr-mini-table">
          ${(patient.module_profiles || []).slice(0, 5).map(row => `
            <div><strong>${esc(row.label || row.module)}</strong><span>${fmtInt(row.rows)} rows · ${fmtPct(row.coverage_pct)} coverage · ${fmtInt(row.feature_count)} features</span></div>
          `).join('') || `<div><strong>${t('No modules found', '未找到模块')}</strong><span>${t('The export does not expose reviewable modules.', '该导出没有可审阅模块。')}</span></div>`}
        </div>
        ${lanes.length ? `<div class="gdr-note">${lanes.map(row => `${row.label}: ${fmtInt(row.signal_count)} signals`).join(' · ')}</div>` : `<div class="gdr-note">${t('No time-series lanes are available in this export. Add vitals/labs/scores modules to review trajectories.', '该导出暂无时间序列通道。请补充 vitals/labs/scores 模块后查看轨迹。')}</div>`}
      </div>`;
  }
  function guidedSurvivalCurve(cohort) {
    const survival = cohort && cohort.survival_analysis ? cohort.survival_analysis : {};
    const outcomeId = survival.default_outcome || ((survival.outcomes || []).find(row => row.status === 'ready') || {}).id;
    const groupId = survival.default_group || ((survival.group_options || []).find(row => row.status === 'ready') || {}).id;
    return (survival.curves || []).find(row => row.outcome_id === outcomeId && row.group_id === groupId) || null;
  }
  function renderGuidedKmPanel(cohort) {
    const survival = cohort && cohort.survival_analysis ? cohort.survival_analysis : {};
    const curve = guidedSurvivalCurve(cohort);
    const blocked = (survival.outcomes || []).filter(row => row.status !== 'ready').slice(0, 3);
    if (!curve) {
      return `
        <div class="gdr-panel blocked">
          <div class="gdr-panel-head"><div><span class="gdx-label">KM / log-rank</span><strong>${t('Blocked by export schema', '被导出结构阻断')}</strong></div></div>
          <p>${esc(survival.reason || t('This export does not expose event and time-to-event columns for KM/log-rank.', '该导出没有 KM/log-rank 所需的事件列和 time-to-event 列。'))}</p>
          ${blocked.length ? `<div class="gdr-mini-table">${blocked.map(row => `<div><strong>${esc(row.label || row.id)}</strong><span>${esc(row.reason || 'unavailable')}</span></div>`).join('')}</div>` : ''}
        </div>`;
    }
    const logrank = curve.logrank || {};
    const groups = curve.groups || [];
    const risk = curve.number_at_risk || {};
    const times = risk.times || [];
    const rows = risk.rows || [];
    return `
      <div class="gdr-panel">
        <div class="gdr-panel-head">
          <div><span class="gdx-label">KM / log-rank</span><strong>${esc(curve.label || 'Kaplan-Meier curve')}</strong></div>
          <div class="gdr-logrank"><span>log-rank</span><strong>${logrank.status === 'ready' ? `χ² ${fmtFixed(logrank.chi_square, 2)} · p ${fmtP(logrank.p_value)}` : 'blocked'}</strong></div>
        </div>
        <div class="gdr-km">
          ${groups.map((g, i) => `<div class="gdr-km-row"><span class="line c${i % 4}"></span><strong>${esc(g.label || `Group ${i + 1}`)}</strong><em>n ${fmtInt(g.n)} · events ${fmtInt(g.events)}</em></div>`).join('')}
        </div>
        ${times.length && rows.length ? `<div class="gdr-risk"><strong>${t('Number at risk', '风险人数表')}</strong><table><thead><tr><th>Group</th>${times.map(x => `<th>${fmtNum(x)}d</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr><td>${esc(row.label)}</td>${(row.values || []).map(v => `<td>${fmtInt(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : ''}
        <div class="gdr-note">${t('Exploratory aggregate only. Manuscript claims still need Agent evidence checks and human review.', '仅为探索性聚合结果。论文结论仍需 Agent 证据核验和人工审阅。')}</div>
      </div>`;
  }
  function renderGuidedCohortPanel(cohort) {
    if (!cohort) {
      return `<div class="gdr-panel blocked"><strong>${t('Cohort summary unavailable', '队列汇总不可用')}</strong><span>${t('Register an active export, then refresh this card.', '请先注册 active export，然后刷新这张卡。')}</span></div>`;
    }
    const summary = cohort.summary || {};
    const mortality = summary.mortality || {};
    const coverage = (cohort.coverage || []).slice(0, 6);
    return `
      <div class="gdr-panel">
        <div class="gdr-panel-head"><div><span class="gdx-label">${t('Cohort summary', '队列汇总')}</span><strong>${esc(guidedSourceLabel(cohort))}</strong></div></div>
        <div class="gdr-metrics">
          ${guidedMetricCard(t('Cohort', '队列'), fmtInt(summary.cohort_size || summary.entities), 'entities')}
          ${guidedMetricCard(t('Mortality', '死亡率'), fmtPct(summary.mortality_pct), `${fmtInt(mortality.deceased_count, 'n/a')} events`)}
          ${guidedMetricCard(t('Median age', '年龄中位数'), fmtNum(summary.age && summary.age.median, 'n/a'), 'years')}
          ${guidedMetricCard(t('Modules', '模块'), fmtInt(summary.modules), `${fmtInt(summary.total_records)} records`)}
        </div>
        <div class="gdr-mini-table">
          ${coverage.map(row => `<div><strong>${esc(row.label || row.module)}</strong><span>${fmtPct(row.coverage_pct)} · ${fmtInt(row.rows)} rows · ${esc(row.quality_status || 'ok')}</span></div>`).join('') || `<div><strong>${t('No coverage rows', '暂无覆盖率')}</strong><span>${t('This export has no auditable feature modules yet.', '该导出还没有可审计特征模块。')}</span></div>`}
        </div>
      </div>`;
  }
  function renderGuidedReviewCard() {
    if (!guidedReview) resetGuidedReviewState();
    const loading = guidedReview.loading;
    const hasPayload = guidedReview.patient || guidedReview.cohort;
    return `
      <div class="gd-review-card">
        <div class="gdx-head">
          <span class="gdx-ico">${icon('eye', 15)}</span>
          <div>
            <strong>${t('Review active export inside Copilot', '在 Copilot 内审阅 active export')}</strong>
            <span>${t('Uses Patient Review and Cohort Statistics APIs; no seeded demo panels are substituted.', '复用 Patient Review 和 Cohort Statistics API；不会用 seeded demo 面板替代。')}</span>
          </div>
        </div>
        <div class="gdx-status ${guidedReview.error ? 'bad' : hasPayload ? 'ok' : ''}">
          <span>${icon(guidedReview.error ? 'x' : hasPayload ? 'check' : 'shield', 12)}</span>
          <div><strong>${loading ? t('Loading active export review...', '正在加载 active export 审阅...') : guidedReview.error ? esc(guidedReview.error) : hasPayload ? t('Loaded from active registered export.', '已从 active registered export 加载。') : t('No review loaded yet.', '尚未加载审阅。')}</strong><small>${esc(guidedSourceLabel(guidedReview.cohort || guidedReview.patient))}</small></div>
        </div>
        ${hasPayload ? `
          <div class="gdr-grid">
            ${renderGuidedPatientPanel(guidedReview.patient)}
            ${renderGuidedCohortPanel(guidedReview.cohort)}
            ${renderGuidedKmPanel(guidedReview.cohort)}
          </div>
        ` : ''}
        <div class="gdx-actions">
          <button type="button" class="btn primary" data-gr-refresh ${loading ? 'disabled' : ''}>${icon('refresh', 13)} ${t('Refresh review', '刷新审阅')}</button>
          <button type="button" class="btn" data-guided-goal="data_extraction">${t('Prepare more modules here', '在这里补抽取模块')}</button>
          <button type="button" class="btn" data-guided-goal="run_agent" ${hasPayload ? '' : 'disabled'}>${t('Continue to Agent preflight', '继续 Agent 预检')}</button>
        </div>
      </div>`;
  }

  /* ============== inline Agent preflight ============== */
  function resetGuidedAgentState() {
    guidedAgent = {
      question: (studyParams && studyParams.exposure)
        ? `Evaluate whether ${studyParams.exposure} is associated with ${studyParams.outcome || 'mortality'} in the active ICU cohort.`
        : 'Evaluate the active ICU cohort with an evidence-bound local preflight.',
      running: false,
      jobId: null,
      progress: null,
      result: null,
      error: null,
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
      const gate = guidedAgent.result.gate || {};
      return `${t('Agent preflight complete', 'Agent 预检完成')} · ${esc(gate.status || 'analysis_only')}`;
    }
    const src = activeExportSource();
    return src ? t('Ready to run against the active export. No external provider is used.', '可以基于 active export 运行。不会使用外部 provider。') : t('No active export. Prepare/register data first.', '没有 active export。请先准备或注册数据。');
  }
  function renderGuidedAgentCard() {
    if (!guidedAgent) resetGuidedAgentState();
    const src = activeExportSource();
    const result = guidedAgent.result || {};
    const gate = result.gate || {};
    const artifacts = result.artifacts || result.artifact_manifest || [];
    const artCount = Array.isArray(artifacts) ? artifacts.length : (result.artifact_count || 0);
    return `
      <div class="gd-agent-card">
        <div class="gdx-head">
          <span class="gdx-ico">${icon('agent', 15)}</span>
          <div>
            <strong>${t('Run Agent preflight inside Copilot', '在 Copilot 内运行 Agent 预检')}</strong>
            <span>${t('Same /api/jobs/agent-run path as Agent Projects, defaulting to local mock/preflight and evidence checks.', '复用 Agent Projects 相同的 /api/jobs/agent-run，默认本地 mock/preflight 与证据核验。')}</span>
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
        <div class="gdx-status ${guidedAgent.error ? 'bad' : guidedAgent.result ? 'ok' : ''}">
          <span>${icon(guidedAgent.error ? 'x' : guidedAgent.result ? 'check' : 'shield', 12)}</span>
          <div><strong>${guidedAgentStatusText()}</strong>${guidedAgent.jobId ? `<small>job ${esc(guidedAgent.jobId)}</small>` : ''}</div>
        </div>
        ${guidedAgent.result ? `<div class="gda-result">
          ${guidedMetricCard(t('Run type', '运行类型'), result.run_type || 'preflight')}
          ${guidedMetricCard(t('Reportable', '可报告'), result.reportable ? 'true' : 'false', t('Draft remains locked until sign-off.', '签署前草稿保持锁定。'))}
          ${guidedMetricCard(t('Evidence check', '证据核验'), gate.status || 'analysis_only', gate.reason || '')}
          ${guidedMetricCard(t('Artifacts', 'Artifacts'), fmtInt(artCount), result.project_dir ? compactPath(result.project_dir) : '')}
        </div>` : ''}
        <div class="gdx-actions">
          <button type="button" class="btn primary" data-ga-run ${!src || guidedAgent.running ? 'disabled' : ''}>${icon('play', 13)} ${t('Start local preflight', '启动本地预检')}</button>
          <button type="button" class="btn" data-guided-goal="data_extraction">${t('Prepare/register data', '准备/注册数据')}</button>
          ${guidedAgent.result && guidedAgent.result.project_dir ? `<button type="button" class="btn" data-open="agent">${t('Open Agent Projects', '打开 Agent Projects')}</button>` : ''}
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
    renderThread();
    const studyId = slugifyDraftFolder((guidedAgent.question || 'guided-agent-preflight').slice(0, 80)) || 'guided-agent-preflight';
    window.EU_API.startAgentRun({
      path: src.path,
      study_id: studyId,
      mode: 'analysis',
      run_type: 'preflight',
      llm_provider: 'mock',
      external_llm_opt_in: false,
      question: guidedAgent.question,
    }).then(r => {
      guidedAgent.jobId = r.job_id;
      renderThread();
      scheduleGuidedSlotSave('start_agent_preflight');
      const es = new EventSource('/api/jobs/' + encodeURIComponent(r.job_id) + '/events');
      es.onmessage = ev => {
        let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
        if (m.type === 'progress') {
          guidedAgent.progress = m;
          renderThread();
          return;
        }
        if (m.type === 'end') {
          try { es.close(); } catch (e) {}
          guidedAgent.running = false;
          if (m.status === 'done') {
            guidedAgent.result = m.result || {};
            liveAgentRun = { result: guidedAgent.result };
            setVal({ analysis: 'preflight complete', draft: 'locked' });
            markThrough('draft', 'locked');
          } else {
            guidedAgent.error = (m.error && (m.error.message || m.error.code)) || m.status || 'Agent preflight failed.';
          }
          renderThread();
          renderAside();
          scheduleGuidedSlotSave('finish_agent_preflight');
        }
      };
      es.onerror = () => {
        try { es.close(); } catch (e) {}
        guidedAgent.running = false;
        guidedAgent.error = 'Agent event stream stopped before completion.';
        renderThread();
        scheduleGuidedSlotSave('agent_event_stream_error');
      };
    }).catch(err => {
      guidedAgent.running = false;
      guidedAgent.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('start_agent_preflight_error');
    });
  }

  /* ============== inline Idea Mining ============== */
  function resetGuidedIdeaState() {
    guidedIdea = {
      sourceType: 'manual',
      topic: '',
      excerpt: '',
      title: '',
      journal: '',
      year: '',
      doi: '',
      pmid: '',
      url: '',
      sourceFileName: '',
      sourceFileSha256: '',
      sourceFileBytes: 0,
      sourceFilePages: 0,
      literatureFolder: '',
      literaturePdfCount: 0,
      allowNetwork: false,
      planEdits: '',
      resolving: false,
      pdfIngesting: false,
      literatureScanning: false,
      discovering: false,
      mining: false,
      priorArting: false,
      handoffing: false,
      projectCreating: false,
      resolved: null,
      result: null,
      prior: null,
      handoff: null,
      project: null,
      pdf: null,
      literatureScan: null,
      discovery: null,
      error: null,
    };
    guidedIdeaProvider = {
      provider: 'openai',
      loading: false,
      saving: false,
      error: null,
      saveError: null,
      saved: false,
      status: null,
      configOpen: false,
    };
    guidedLiteratureBrowser = { open: false, loading: false, error: null, data: null, path: '' };
  }
  function startGuidedIdeaFlow(label) {
    if (label) pushUser(label);
    resetGuidedIdeaState();
    setVal({ question: 'idea mining', analysis: 'not started', draft: 'locked' });
    markThrough('question', 'active');
    thread.push({ bot: true, html: bi(
      `We can mine a research idea here, without leaving Copilot. Choose a local PDF, scan a local literature folder, paste a paper clue, or describe the topic; I’ll create a local evidence-bound idea ledger, check the active export, and prepare a handoff for Agent Projects.`,
      `可以直接在 Copilot 里挖掘研究想法。选择本地 PDF、扫描本地文献库文件夹、粘贴文章线索，或描述主题后，我会生成本地 evidence-bound idea ledger，检查 active export，并准备交接给 Agent Projects。`,
    ) });
    thread.push({ guidedIdea: true });
    chips = [];
    renderThread();
    renderChips();
    scheduleGuidedSlotSave('start_idea');
  }
  function guidedIdeaPayload() {
    if (!guidedIdea) resetGuidedIdeaState();
    return {
      source_type: guidedIdea.sourceType || 'manual',
      topic: String(guidedIdea.topic || '').trim(),
      excerpt: String(guidedIdea.excerpt || '').trim(),
      title: String(guidedIdea.title || '').trim(),
      journal: String(guidedIdea.journal || '').trim(),
      year: String(guidedIdea.year || '').trim(),
      doi: String(guidedIdea.doi || '').trim(),
      pmid: String(guidedIdea.pmid || '').trim(),
      url: String(guidedIdea.url || '').trim(),
      source_file_name: String(guidedIdea.sourceFileName || '').trim(),
      source_file_sha256: String(guidedIdea.sourceFileSha256 || '').trim(),
      literature_folder: String(guidedIdea.literatureFolder || '').trim(),
      literature_pdf_count: Number(guidedIdea.literaturePdfCount || 0),
      allow_network: !!guidedIdea.allowNetwork,
    };
  }
  function guidedIdeaHasInput() {
    const p = guidedIdeaPayload();
    return !!(p.topic || p.excerpt || p.title || p.url || p.doi || p.pmid || p.source_file_sha256 || p.literature_folder);
  }
  function applyGuidedIdeaSuggestion(suggested) {
    if (!guidedIdea || !suggested) return;
    const scalarMap = {
      topic: 'topic',
      excerpt: 'excerpt',
      title: 'title',
      journal: 'journal',
      year: 'year',
      doi: 'doi',
      pmid: 'pmid',
      url: 'url',
      source_file_name: 'sourceFileName',
      source_file_sha256: 'sourceFileSha256',
      literature_folder: 'literatureFolder',
    };
    Object.entries(scalarMap).forEach(([from, to]) => {
      if (!guidedIdea[to] && suggested[from]) guidedIdea[to] = String(suggested[from]);
    });
    if (!guidedIdea.literaturePdfCount && suggested.literature_pdf_count != null) {
      guidedIdea.literaturePdfCount = Number(suggested.literature_pdf_count || 0);
    }
  }
  function guidedIdeaSelected() {
    const result = guidedIdea && guidedIdea.result;
    const ideas = result && Array.isArray(result.idea_ledger) ? result.idea_ledger : [];
    const id = result && result.selected_idea_id;
    return ideas.find(row => row.idea_id === id) || ideas[0] || null;
  }
  function guidedIdeaStatusText() {
    if (!guidedIdea) return '';
    if (guidedIdea.resolving) return t('Resolving bounded source metadata...', '正在解析有界来源元数据...');
    if (guidedIdea.discovering) return t('Searching opt-in PubMed metadata and mapping candidate ideas...', '正在按 opt-in 检索 PubMed 元数据并映射候选 idea...');
    if (guidedIdea.mining) return t('Mining local idea ledger and active-export feasibility...', '正在生成本地 idea ledger 并检查 active export 可行性...');
    if (guidedIdea.priorArting) return t('Checking prior art under explicit opt-in rules...', '正在按显式 opt-in 规则检查 prior art...');
    if (guidedIdea.handoffing) return t('Writing local handoff plan...', '正在写入本地 handoff plan...');
    if (guidedIdea.projectCreating) return t('Creating metadata-only Agent project seed...', '正在创建 metadata-only Agent project seed...');
    if (guidedIdea.error) return esc(guidedIdea.error);
    if (guidedIdea.project) return t('Agent project seed created from the idea handoff.', '已从 idea handoff 创建 Agent project seed。');
    if (guidedIdea.handoff) return t('Handoff written. You can create an Agent project seed next.', 'handoff 已写入。下一步可创建 Agent project seed。');
    if (guidedIdea.result) return t('Local idea ledger and pre-experiment are ready; API-backed checks remain gated until configured.', '本地 idea ledger 与预实验已生成；需要 API 的检查仍会在配置前保持阻断。');
    if (guidedIdea.resolved) return t('Source metadata resolved. Run local mining next.', '来源元数据已解析。下一步运行本地挖掘。');
    return t('Add a source clue or topic, then run local mining.', '先添加来源线索或主题，然后运行本地挖掘。');
  }
  function requestGuidedIdeaProviderStatus(force) {
    if (!guidedIdeaProvider) {
      guidedIdeaProvider = {
        provider: 'openai',
        loading: false,
        saving: false,
        error: null,
        saveError: null,
        saved: false,
        status: null,
        configOpen: false,
      };
    }
    if (!window.EU_API || !window.EU_API.loadAgentProviderStatus) return;
    if (!force && (guidedIdeaProvider.loading || guidedIdeaProvider.status || guidedIdeaProvider.error)) return;
    guidedIdeaProvider = Object.assign({}, guidedIdeaProvider, { loading: true, error: null });
    window.EU_API.loadAgentProviderStatus(guidedIdeaProvider.provider || 'openai').then(data => {
      guidedIdeaProvider = Object.assign({}, guidedIdeaProvider, {
        loading: false,
        error: null,
        status: (data && data.provider_status) || data || null,
      });
      renderThread();
    }).catch(err => {
      guidedIdeaProvider = Object.assign({}, guidedIdeaProvider, {
        loading: false,
        error: err && err.message ? err.message : String(err || 'provider_status_error'),
        status: null,
      });
      renderThread();
    });
  }
  function renderGuidedIdeaCapabilityPanel() {
    if (!guidedIdeaProvider) guidedIdeaProvider = { provider: 'openai', loading: false, error: null, status: null };
    const st = guidedIdeaProvider.status || {};
    const envFile = st.env_file || {};
    const missing = Array.isArray(st.missing) ? st.missing : [];
    const ready = !!st.ready;
    const aiOn = !!st.ai_enabled;
    const keyReady = !!st.credential_present;
    const modelReady = !!st.model_present;
    const envStatus = envFile.status || 'not_loaded';
    const blocked = missing.length ? missing.join(', ') : (ready ? '' : t('provider not ready', 'provider 未就绪'));
    const providers = [
      ['openai', 'OpenAI'],
      ['openrouter', 'OpenRouter'],
      ['deepseek', 'DeepSeek'],
      ['custom', 'Custom/local'],
    ];
    return `
      <div class="gdx-status ${ready ? 'ok' : 'warn'}">
        <span>${icon(ready ? 'check' : 'shield', 12)}</span>
        <div>
          <strong>${t('Capability boundary', '能力边界')}</strong>
          <small>${t('Local source parsing, dictionary matching, active-export feasibility, and handoff can run without an API. Prior-art search and AI synthesis require explicit opt-in and provider readiness.', '本地来源解析、字典匹配、active export 可行性和交接不需要 API。既有研究检索和 AI 综合必须显式 opt-in 且 provider 就绪。')}</small>
        </div>
      </div>
      <div class="gdi-feature-list">
        <div class="gdi-feature-row">
          <div><strong>${t('Local deterministic mining', '本地确定性挖掘')}</strong><small>${t('PDF/folder bounded excerpt, dictionary feasibility, pre-experiment, Agent handoff seed', 'PDF/文件夹有界摘录、字典可行性、预实验、Agent 交接种子')}</small></div>
          <span class="pill ok">${t('available now', '当前可用')}</span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${t('Network prior-art check', '联网既有研究检查')}</strong><small>${t('Runs only after the per-source network checkbox is selected; no request is made otherwise.', '只有勾选当前来源的网络 opt-in 后才会请求；否则不会联网。')}</small></div>
          <span class="pill ${guidedIdea && guidedIdea.allowNetwork ? 'warn' : 'dashed'}">${guidedIdea && guidedIdea.allowNetwork ? t('armed for one request', '已允许一次请求') : t('blocked until opt-in', '等待 opt-in')}</span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${t('AI synthesis / full Agent provider', 'AI 综合 / full Agent provider')}</strong><small>${ready ? t('Provider readiness passed. A later Agent run still needs per-run confirmation and evidence checks.', 'provider 已就绪。后续 Agent run 仍需要逐次确认和证据核验。') : t('Blocked: configure AI opt-in, API key, model, and endpoint before any provider call.', '已阻断：需要配置 AI opt-in、API key、模型和端点后才允许 provider 调用。')}</small></div>
          <span class="pill ${ready ? 'ok' : 'warn'}">${ready ? t('ready', '就绪') : esc(blocked)}</span>
        </div>
      </div>
      <div class="gdi-feature-list">
        <div class="gdi-feature-row">
          <div><strong>${t('Provider', 'Provider')}</strong><small>${esc(guidedIdeaProvider.provider || st.provider || 'openai')}</small></div>
          <span>${guidedIdeaProvider.loading ? t('checking...', '检查中...') : ready ? t('ready', '就绪') : t('blocked', '受阻')}</span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${t('Readiness flags', '就绪标记')}</strong><small>${t('Only variable names and booleans are shown; secret values and base URL values are never returned.', '只显示变量名和布尔值；密钥值和 base URL 值不会返回。')}</small></div>
          <span class="gdi-tags">
            <code>AI ${aiOn ? 'on' : 'off'}</code>
            <code>key ${keyReady ? 'present' : 'missing'}</code>
            <code>model ${modelReady ? 'present' : 'missing'}</code>
          </span>
        </div>
        <div class="gdi-feature-row">
          <div><strong>${t('Env sources', '环境变量来源')}</strong><small>${t('Sanitized provider status from Agent provider readiness.', '来自 Agent provider readiness 的脱敏状态。')}</small></div>
          <span class="gdi-tags">
            <code>${esc(st.credential_source || (st.credential_env_candidates || [])[0] || 'credential env')}</code>
            <code>${esc(st.model_source || (st.model_env_candidates || [])[0] || 'model env')}</code>
            <code>${esc(st.base_url_source || (st.base_url_env_candidates || [])[0] || 'base_url env')}</code>
            <code>${esc(envStatus)}</code>
          </span>
        </div>
        ${guidedIdeaProvider.error ? `<div class="gdi-feature-row"><div><strong>${t('Provider status unavailable', 'provider 状态不可用')}</strong><small>${esc(guidedIdeaProvider.error)}</small></div><span class="pill warn">error</span></div>` : ''}
      </div>
      <div class="gdx-actions">
        ${providers.map(([p, label]) => `<button type="button" class="btn sm ${guidedIdeaProvider.provider === p ? 'primary' : ''}" data-gi-provider="${p}">${esc(label)}</button>`).join('')}
        <button type="button" class="btn sm" data-gi-provider-refresh>${icon('refresh', 12)} ${t('Check API status', '检查 API 状态')}</button>
        <button type="button" class="btn sm" data-open="settings">${icon('gear', 12)} ${t('Configure in Settings', '去设置中配置')}</button>
      </div>`;
  }
  function renderGuidedIdeaPdfPicker() {
    const pdf = guidedIdea && guidedIdea.pdf;
    const status = guidedIdea && guidedIdea.pdfIngesting
      ? t('Reading selected PDF...', '正在读取选中的 PDF...')
      : pdf
        ? `${pdf.filename || guidedIdea.sourceFileName || 'PDF'} · ${fmtInt(pdf.page_count, '?')} ${t('pages', '页')} · ${compactHash(pdf.sha256 || guidedIdea.sourceFileSha256)}`
        : t('Choose a local PDF. Only a bounded excerpt and SHA-256 hash are kept; full text is not stored.', '选择本地 PDF。只保留有界摘录和 SHA-256 哈希，不保存全文。');
    return `
      <div class="gdx-status ${pdf ? 'ok' : ''}">
        <input type="file" accept="application/pdf,.pdf" data-gi-pdf-file hidden />
        <span>${icon(pdf ? 'check' : 'file', 12)}</span>
        <div><strong>${esc(status)}</strong><small>${t('The browser reads the file locally and sends bounded metadata to the local EasyICU server.', '浏览器在本机读取文件，并只把有界元数据交给本机 EasyICU 服务。')}</small></div>
        <button type="button" class="btn sm" data-gi-pdf-pick ${guidedIdea.pdfIngesting ? 'disabled' : ''}>${icon('folder', 12)} ${pdf ? t('Choose another PDF', '换一个 PDF') : t('Choose PDF', '选择 PDF')}</button>
      </div>
      ${pdf && pdf.excerpt ? `<div class="gdi-muted">${t('Extracted excerpt', '已提取摘录')} · ${fmtInt(pdf.excerpt_char_count, '0')} chars</div>` : ''}`;
  }
  function renderGuidedLiteratureBrowser() {
    const browser = guidedLiteratureBrowser || {};
    if (!browser.open) return '';
    const data = browser.data || {};
    const entries = Array.isArray(data.entries) ? data.entries : [];
    const shortcuts = Array.isArray(data.shortcuts) ? data.shortcuts : [];
    const currentPath = data.path || browser.path || '';
    const parent = data.parent || '';
    const failed = browser.error || (data && data.ok === false ? data.error : '');
    return `
      <div class="gds-browser" data-guided-literature-browser>
        <div class="gds-browser-head">
          <div>
            <strong>${t('Literature folder picker', '文献库文件夹选择器')}</strong>
            <span>${t('Browse local folders through EasyICU. Select a folder that contains downloaded PDFs.', '通过 EasyICU 浏览本地文件夹。请选择包含已下载 PDF 的文件夹。')}</span>
          </div>
          <button class="btn sm ghost" type="button" data-lit-browser-close>${icon('close', 12)}</button>
        </div>
        <div class="gds-browser-path"><span>${t('Current', '当前')}</span><code>${esc(compactPath(currentPath) || t('Home folder', '主目录'))}</code></div>
        ${shortcuts.length ? `<div class="gds-browser-shortcuts">${shortcuts.map((item, i) => `
          <button class="btn sm" type="button" data-lit-browser-shortcut="${i}">${esc(item.name || 'Folder')}</button>`).join('')}</div>` : ''}
        ${failed ? `<div class="gds-browser-message warn">${icon('info', 12)} <span>${esc(String(failed))}</span></div>` : ''}
        <div class="gds-browser-list">
          ${browser.loading ? `<div class="gds-browser-empty">${icon('refresh', 13)} ${t('Loading folders...', '正在加载文件夹...')}</div>` : ''}
          ${!browser.loading && !entries.length ? `<div class="gds-browser-empty">${t('No child folders here. You can still choose the current folder.', '这里没有下级文件夹。也可以直接选择当前文件夹。')}</div>` : ''}
          ${!browser.loading && entries.map((entry, i) => `
            <button class="gds-browser-row" type="button" data-lit-browser-entry="${i}">
              <span class="gds-ico">${icon('folder', 13)}</span>
              <span><strong>${esc(entry.name || 'Folder')}</strong><code>${esc(compactPath(entry.path || ''))}</code></span>
              ${entry.hint ? `<em>${esc(entry.hint)}</em>` : ''}
            </button>`).join('')}
        </div>
        <div class="gds-browser-actions">
          <button class="btn sm" type="button" data-lit-browser-up ${parent ? '' : 'disabled'}>${icon('back', 12)} ${t('Up', '上一级')}</button>
          <span class="grow"></span>
          <button class="btn primary sm" type="button" data-lit-browser-use ${currentPath ? '' : 'disabled'}>${icon('check', 12)} ${t('Use this folder', '选择此文件夹')}</button>
        </div>
      </div>`;
  }
  function renderGuidedIdeaLiteraturePicker() {
    const scan = guidedIdea && guidedIdea.literatureScan;
    const docs = scan && Array.isArray(scan.documents) ? scan.documents.slice(0, 4) : [];
    return `
      <div class="gdi-field wide">
        <span>${t('Local literature folder', '本地文献库文件夹')}</span>
        <div class="path-field">
          <span class="pf-path">${esc(guidedIdea.literatureFolder || t('Choose a folder containing local PDFs', '选择一个包含本地 PDF 的文件夹'))}</span>
          <button type="button" class="btn sm" data-gi-lit-browse>${icon('folder', 12)} ${t('Browse...', '浏览...')}</button>
        </div>
        <input data-gi-field="literatureFolder" value="${attr(guidedIdea.literatureFolder || '')}" placeholder="${attr(t('Optional: paste a folder path if browser selection is unavailable', '可选：如果浏览器选择不可用，可以粘贴文件夹路径'))}" />
      </div>
      <div class="gdx-actions">
        <button type="button" class="btn primary" data-gi-lit-scan ${guidedIdea.literatureScanning ? 'disabled' : ''}>${icon('search', 13)} ${guidedIdea.literatureScanning ? t('Scanning PDFs...', '正在扫描 PDF...') : t('Scan literature folder', '扫描文献库文件夹')}</button>
      </div>
      ${renderGuidedLiteratureBrowser()}
      ${scan ? `<div class="gdx-status ok">
        <span>${icon('check', 12)}</span>
        <div><strong>${fmtInt((scan.folder || {}).pdf_count, 0)} ${t('PDFs found', '个 PDF 已发现')}</strong><small>${esc(compactPath((scan.folder || {}).path || guidedIdea.literatureFolder || ''))}</small></div>
      </div>` : ''}
      ${docs.length ? `<div class="gdi-feature-list">${docs.map(doc => `<div class="gdi-feature-row"><div><strong>${esc(doc.title || doc.filename || 'PDF')}</strong><small>${esc(compactPath(doc.path || doc.filename || ''))}</small></div><span>${compactHash(doc.sha256 || '')}</span></div>`).join('')}</div>` : ''}`;
  }
  function renderGuidedIdeaDiscovery() {
    const discovery = guidedIdea && guidedIdea.discovery;
    if (!discovery && guidedIdea && guidedIdea.sourceType !== 'frontier') return '';
    const candidates = discovery && Array.isArray(discovery.idea_candidates) ? discovery.idea_candidates.slice(0, 6) : [];
    const queries = discovery && Array.isArray(discovery.queries_to_run) ? discovery.queries_to_run.slice(0, 4) : [];
    return `
      <div class="gdi-prior">
        <div class="gdi-ledger-title">
          <div>
            <span class="gdx-label">${t('Literature discovery', '文献发现')}</span>
            <strong>${esc((discovery && discovery.status) || t('not searched yet', '尚未检索'))}</strong>
          </div>
          <button type="button" class="btn sm ${guidedIdea && guidedIdea.allowNetwork ? 'primary' : ''}" data-gi-discover ${guidedIdea && guidedIdea.discovering ? 'disabled' : ''}>${guidedIdea && guidedIdea.discovering ? '<span class="spin"></span>' : icon('search', 12)} ${t('Discover papers', '检索文章')}</button>
        </div>
        <p>${esc((discovery && discovery.reason) || t('Use this for frontier/review topics. It searches PubMed metadata only after network opt-in, then maps candidate ideas to the EasyICU dictionary and active export.', '用于前沿/review 主题。只有勾选网络 opt-in 后才检索 PubMed 元数据，然后把候选 idea 映射到 EasyICU 字典和当前导出。'))}</p>
        ${queries.length ? `<div class="gdi-query-list">${queries.map(q => `<code>${esc(q)}</code>`).join('')}</div>` : ''}
        ${candidates.length ? `<div class="gdi-feature-list">
          ${candidates.map((row, i) => {
            const idea = row.idea || {};
            const src = row.source || {};
            const feas = idea.feasibility || {};
            return `<div class="gdi-feature-row">
              <div>
                <strong>${esc(idea.idea_title || src.title || 'Candidate idea')}</strong>
                <small>${esc([src.year, src.journal, src.pmid ? 'PMID ' + src.pmid : ''].filter(Boolean).join(' · '))}</small>
              </div>
              <span class="pill ${feas.tier === 'executable' ? 'ok' : 'warn'}">${esc(feas.tier || idea.go_no_go || 'review')}</span>
              <button type="button" class="btn sm" data-gi-discovery-use="${i}">${t('Use', '使用')}</button>
            </div>`;
          }).join('')}
        </div>` : ''}
      </div>`;
  }
  function renderGuidedIdeaSourceFields() {
    if (!guidedIdea) resetGuidedIdeaState();
    const tab = guidedIdea.sourceType || 'manual';
    const tabs = [
      ['manual', t('Manual idea', '手动想法')],
      ['url', t('Article URL', '文章链接')],
      ['pdf', t('PDF file', 'PDF 文件')],
      ['literature_folder', t('Literature folder', '文献库文件夹')],
      ['frontier', t('Frontier topic', '前沿主题')],
    ];
    return `
      <div class="gdi-tabs" role="group" aria-label="Idea source type">
        ${tabs.map(([key, label]) => `<button type="button" class="${tab === key ? 'on' : ''}" data-gi-source="${key}">${label}</button>`).join('')}
      </div>
      <div class="gdi-form">
        <label class="gdi-field wide">
          <span>${t('Research idea / topic', '研究想法 / 主题')}</span>
          <textarea rows="3" data-gi-field="topic" placeholder="${attr(t('e.g. Does early vasopressor strategy change mortality in septic ICU patients?', '例如：早期升压药策略是否影响脓毒症 ICU 患者死亡率？'))}">${esc(guidedIdea.topic || '')}</textarea>
        </label>
        <label class="gdi-field wide">
          <span>${t('Source quote or PDF excerpt', '来源句子或 PDF 摘录')}</span>
          <textarea rows="3" data-gi-field="excerpt" placeholder="${attr(t('Paste only the sentence(s) that motivated the idea; do not paste a full paper.', '只粘贴触发想法的句子；不要粘贴全文。'))}">${esc(guidedIdea.excerpt || '')}</textarea>
        </label>
        ${tab === 'pdf' ? renderGuidedIdeaPdfPicker() : ''}
        ${tab === 'literature_folder' ? renderGuidedIdeaLiteraturePicker() : ''}
        ${tab === 'frontier' ? renderGuidedIdeaDiscovery() : ''}
        <div class="gdi-meta-grid">
          <label class="gdi-field"><span>Title</span><input data-gi-field="title" value="${attr(guidedIdea.title || '')}" placeholder="${attr(t('Article or review title', '文章或综述标题'))}" /></label>
          <label class="gdi-field"><span>Journal</span><input data-gi-field="journal" value="${attr(guidedIdea.journal || '')}" placeholder="e.g. Intensive Care Medicine" /></label>
          <label class="gdi-field"><span>Year</span><input data-gi-field="year" value="${attr(guidedIdea.year || '')}" placeholder="2026" /></label>
          <label class="gdi-field"><span>DOI / PMID</span><input data-gi-field="doi" value="${attr(guidedIdea.doi || '')}" placeholder="10.xxxx or PMID" /></label>
        </div>
        <label class="gdi-field wide">
          <span>URL</span>
          <input data-gi-field="url" value="${attr(guidedIdea.url || '')}" placeholder="https://..." />
        </label>
        <label class="gdi-check">
          <input type="checkbox" data-gi-network ${guidedIdea.allowNetwork ? 'checked' : ''} />
          <span>${t('Allow one bounded network metadata/prior-art request for this source', '允许针对该来源进行一次有界网络元数据/prior-art 请求')}</span>
          <em>opt-in</em>
        </label>
      </div>`;
  }
  function renderGuidedIdeaEvidence(result) {
    const src = ((result && result.source_evidence) || [])[0] || {};
    if (!src.source_id && !src.title) return '';
    return `
      <div class="gdi-source-card">
        <div class="gdx-label">${t('Source evidence', '来源证据')}</div>
        <strong>${esc(src.title || 'Untitled source')}</strong>
        <small>${[src.year, src.journal, src.doi || src.pmid].filter(Boolean).map(esc).join(' · ') || esc(src.source_type || 'manual')}</small>
        ${src.evidence_quote ? `<blockquote>${esc(src.evidence_quote)}</blockquote>` : ''}
        <div class="gdi-muted">${t('Only metadata, a bounded quote, and hashes are persisted.', '仅持久化元数据、有界摘录和哈希。')}</div>
      </div>`;
  }
  function renderGuidedIdeaLedger(idea) {
    if (!idea) return '';
    const concepts = (idea.mapped_concepts || []).slice(0, 10);
    const feasibility = idea.feasibility || {};
    const prior = idea.prior_art || {};
    return `
      <div class="gdi-ledger">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">Idea ledger</span><strong>${esc(idea.idea_title || 'Candidate idea')}</strong></div>
          <span class="pill ${idea.go_no_go === 'recommend' ? 'ok' : 'warn'}">${esc(idea.go_no_go || 'hold')}</span>
        </div>
        <p>${esc(idea.rationale || '')}</p>
        <div class="gdi-ledger-grid">
          <div>
            <span>${t('Mapped concepts', '映射概念')}</span>
            <div class="gdi-tags">${concepts.map(row => `<code>${esc(row.concept_id || row.label)} · ${esc(row.tier || '')}</code>`).join('') || `<em>${t('No dictionary mapping yet', '暂无字典映射')}</em>`}</div>
          </div>
          <div><span>${t('Feasibility', '可行性')}</span><strong>${esc(feasibility.label || feasibility.tier || 'unknown')}</strong><small>${esc(feasibility.reason || '')}</small></div>
          <div><span>${t('Prior art', '既有研究')}</span><strong>${esc(prior.status || 'not checked')}</strong><small>${esc(prior.reason || '')}</small></div>
          <div><span>${t('Next action', '下一步')}</span><strong>${esc(idea.next_action || idea.go_no_go_reason || 'review')}</strong></div>
        </div>
      </div>`;
  }
  function renderGuidedIdeaPreExperiment(result) {
    const pre = result && result.pre_experiment;
    if (!pre) return '';
    const cohort = pre.cohort || {};
    const stats = (pre.feature_statistics || []).slice(0, 8);
    return `
      <div class="gdi-pre">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">${t('Pre-experiment on active export', 'active export 预实验')}</span><strong>${esc(pre.status || 'blocked')}</strong></div>
          <span class="pill">${esc(pre.payload_scope || 'aggregate')}</span>
        </div>
        ${pre.reason ? `<p>${esc(pre.reason)}</p>` : ''}
        <div class="gdi-stats">
          ${guidedMetricCard(t('Entities', '实体数'), fmtInt(cohort.entities, 'n/a'))}
          ${guidedMetricCard(t('Modules', '模块'), fmtInt(cohort.modules, 'n/a'))}
          ${guidedMetricCard(t('Feature checks', '特征检查'), fmtInt(stats.length, '0'))}
          ${guidedMetricCard(t('Rows', '行数'), fmtInt(cohort.total_rows, 'n/a'))}
        </div>
        ${stats.length ? `<div class="gdi-feature-list">
          ${stats.map(row => {
            const coverage = Number(row.coverage_pct || 0);
            return `<div class="gdi-feature-row">
              <div><strong>${esc(row.label || row.concept_id)}</strong><small>${esc(row.concept_id || '')} · ${esc(row.module || '')}</small></div>
              <div class="gdi-feature-bar"><span style="width:${Math.max(0, Math.min(100, coverage))}%"></span></div>
              <span>${fmtPct(row.coverage_pct)}</span>
              <small>${esc(row.numeric_summary || row.status || '')}</small>
            </div>`;
          }).join('')}
        </div>` : ''}
        ${(pre.interpretation || []).length ? `<div class="gdr-note">${pre.interpretation.map(row => esc(row)).join('<br>')}</div>` : ''}
      </div>`;
  }
  function renderGuidedIdeaPrior() {
    if (!guidedIdea || (!guidedIdea.prior && !guidedIdea.result)) return '';
    const prior = (guidedIdea.prior && guidedIdea.prior.prior_art) || (guidedIdea.result && guidedIdea.result.prior_art) || {};
    const queries = prior.queries_to_run || [];
    const results = prior.results || [];
    return `
      <div class="gdi-prior">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">Prior-art check</span><strong>${esc(prior.status || 'not checked')}</strong></div>
          <button type="button" class="btn sm" data-gi-prior ${guidedIdea.priorArting ? 'disabled' : ''}>${icon('search', 12)} ${t('Check prior art', '检查既有研究')}</button>
        </div>
        <p>${esc(prior.reason || t('Optional bounded network metadata search. It does not use an LLM provider, and it stays blocked until you explicitly opt in for this source.', '可选有界网络元数据搜索。它不使用 LLM provider，且在当前来源显式 opt-in 前保持阻断。'))}</p>
        ${queries.length ? `<div class="gdi-query-list">${queries.slice(0, 4).map(q => `<code>${esc(q)}</code>`).join('')}</div>` : ''}
        ${results.length ? `<div class="gdi-feature-list">${results.slice(0, 5).map(row => `<div class="gdi-feature-row"><div><strong>${esc(row.title || 'result')}</strong><small>${esc([row.year, row.journal, row.pmid].filter(Boolean).join(' · '))}</small></div><span>${esc(row.database || '')}</span></div>`).join('')}</div>` : ''}
      </div>`;
  }
  function renderGuidedIdeaHandoff() {
    if (!guidedIdea || !guidedIdea.result) return '';
    const handoff = guidedIdea.handoff || {};
    const plan = (handoff.handoff_plan || guidedIdea.result.handoff_plan || {});
    const steps = plan.analysis_plan || [];
    return `
      <div class="gdi-plan">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">${t('Agent handoff', 'Agent 交接')}</span><strong>${esc(plan.research_question || 'Confirm the plan before Agent run')}</strong></div>
          <span class="pill warn">${t('draft locked', '草稿锁定')}</span>
        </div>
        ${steps.length ? `<ol>${steps.map(row => `<li>${esc(row)}</li>`).join('')}</ol>` : ''}
        <label class="gdi-field wide">
          <span>${t('Natural-language plan edits', '用自然语言微调计划')}</span>
          <textarea rows="3" data-gi-field="planEdits" placeholder="${attr(t('e.g. restrict to first ICU stay; compare norepinephrine-equivalent dose groups.', '例如：限制首次 ICU；比较去甲肾上腺素等效剂量分组。'))}">${esc(guidedIdea.planEdits || '')}</textarea>
        </label>
        <div class="gdx-actions">
          <button type="button" class="btn primary" data-gi-handoff ${guidedIdea.handoffing ? 'disabled' : ''}>${icon('lock', 13)} ${t('Freeze handoff for Agent', '冻结交接给 Agent')}</button>
          <button type="button" class="btn" data-gi-project ${!guidedIdea.handoff || guidedIdea.projectCreating ? 'disabled' : ''}>${icon('agent', 13)} ${t('Create Agent project', '创建 Agent 项目')}</button>
          ${guidedIdea.project ? `<button type="button" class="btn" data-open="agent">${t('Open Agent Projects', '打开 Agent Projects')}</button>` : ''}
        </div>
      </div>`;
  }
  function renderGuidedIdeaCard() {
    if (!guidedIdea) resetGuidedIdeaState();
    requestGuidedIdeaProviderStatus(false);
    const idea = guidedIdeaSelected();
    const result = guidedIdea.result;
    const miningBlocked = guidedIdea.mining || guidedIdea.resolving;
    return `
      <div class="gd-idea-card">
        <div class="gdx-head">
          <span class="gdx-ico">${icon('spark', 15)}</span>
          <div>
            <strong>${t('Mine a study idea inside Copilot', '在 Copilot 内挖掘研究想法')}</strong>
            <span>${t('Source evidence, dictionary feasibility, pre-experiment, and Agent handoff all stay metadata-only and local-first.', '来源证据、字典可行性、预实验和 Agent 交接均保持 metadata-only 与 local-first。')}</span>
          </div>
        </div>
        ${renderGuidedIdeaSourceFields()}
        ${renderGuidedIdeaCapabilityPanel()}
        <div class="gdx-status ${guidedIdea.error ? 'bad' : result ? 'ok' : ''}">
          <span>${icon(guidedIdea.error ? 'x' : result ? 'check' : 'shield', 12)}</span>
          <div><strong>${guidedIdeaStatusText()}</strong><small>${t('No patient rows, full papers, external calls, or provider clients unless you explicitly opt in and provider readiness passes.', '不会返回患者行、全文、外部调用或 provider client，除非你显式 opt-in 且 provider readiness 通过。')}</small></div>
        </div>
        <div class="gdx-actions">
          <button type="button" class="btn" data-gi-resolve ${guidedIdea.resolving ? 'disabled' : ''}>${icon('search', 13)} ${t('Resolve source', '解析来源')}</button>
          <button type="button" class="btn primary" data-gi-mine ${miningBlocked ? 'disabled' : ''}>${icon('play', 13)} ${t('Mine locally', '本地挖掘 idea')}</button>
          <button type="button" class="btn" data-guided-goal="data_extraction">${t('Prepare data first', '先准备数据')}</button>
        </div>
        ${renderGuidedIdeaEvidence(result)}
        ${renderGuidedIdeaLedger(idea)}
        ${renderGuidedIdeaPreExperiment(result)}
        ${renderGuidedIdeaPrior()}
        ${renderGuidedIdeaHandoff()}
        ${guidedIdea.project ? `<div class="gdx-status ok"><span>${icon('check', 12)}</span><div><strong>${t('Agent project seed created', 'Agent project seed 已创建')}</strong><small>${esc(compactPath((guidedIdea.project.project || {}).project_dir || (guidedIdea.project.project || {}).study_id || ''))}</small></div></div>` : ''}
      </div>`;
  }
  function loadGuidedLiteratureBrowser(path) {
    if (!guidedLiteratureBrowser) guidedLiteratureBrowser = { open: false, loading: false, error: null, data: null, path: '' };
    guidedLiteratureBrowser.open = true;
    guidedLiteratureBrowser.loading = true;
    guidedLiteratureBrowser.error = null;
    guidedLiteratureBrowser.path = String(path || guidedLiteratureBrowser.path || guidedIdea.literatureFolder || '');
    renderThread();
    if (!window.EU_API || !window.EU_API.listDir) {
      guidedLiteratureBrowser.loading = false;
      guidedLiteratureBrowser.error = t('Local folder picker API is unavailable.', '本地文件夹选择 API 不可用。');
      renderThread();
      return;
    }
    window.EU_API.listDir(guidedLiteratureBrowser.path)
      .then(result => {
        guidedLiteratureBrowser.loading = false;
        guidedLiteratureBrowser.data = result || {};
        guidedLiteratureBrowser.path = (result && result.path) || guidedLiteratureBrowser.path || '';
        guidedLiteratureBrowser.error = result && result.ok === false ? (result.error || 'folder_error') : null;
        renderThread();
      })
      .catch(err => {
        guidedLiteratureBrowser.loading = false;
        guidedLiteratureBrowser.error = String(err && err.message || err || 'folder_error');
        renderThread();
      });
  }
  function ingestGuidedIdeaPdfFile(file) {
    if (!guidedIdea || !file) return;
    const name = String(file.name || '');
    if (!/\.pdf$/i.test(name) && file.type && file.type !== 'application/pdf') {
      guidedIdea.error = t('Choose a PDF file.', '请选择 PDF 文件。');
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.ingestIdeaPdf) {
      guidedIdea.error = t('PDF ingestion API is unavailable.', 'PDF 解析 API 不可用。');
      renderThread();
      return;
    }
    guidedIdea.sourceType = 'pdf';
    guidedIdea.pdfIngesting = true;
    guidedIdea.error = null;
    renderThread();
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      const contentBase64 = text.includes(',') ? text.split(',').pop() : text;
      window.EU_API.ingestIdeaPdf({ filename: name, content_base64: contentBase64 })
        .then(result => {
          guidedIdea.pdfIngesting = false;
          guidedIdea.pdf = result && result.pdf ? result.pdf : null;
          if (guidedIdea.pdf) {
            guidedIdea.sourceFileName = guidedIdea.pdf.filename || name;
            guidedIdea.sourceFileSha256 = guidedIdea.pdf.sha256 || '';
            guidedIdea.sourceFileBytes = Number(guidedIdea.pdf.bytes || file.size || 0);
            guidedIdea.sourceFilePages = Number(guidedIdea.pdf.page_count || 0);
          }
          applyGuidedIdeaSuggestion(result && result.suggested_payload);
          guidedIdea.resolved = result;
          renderThread();
          scheduleGuidedSlotSave('ingest_idea_pdf');
        })
        .catch(err => {
          guidedIdea.pdfIngesting = false;
          guidedIdea.error = err.message || String(err);
          renderThread();
          scheduleGuidedSlotSave('ingest_idea_pdf_error');
        });
    };
    reader.onerror = () => {
      guidedIdea.pdfIngesting = false;
      guidedIdea.error = t('Could not read the selected PDF file.', '无法读取选中的 PDF 文件。');
      renderThread();
    };
    reader.readAsDataURL(file);
  }
  function scanGuidedLiteratureFolder() {
    if (!guidedIdea || guidedIdea.literatureScanning) return;
    const path = String(guidedIdea.literatureFolder || '').trim();
    if (!path) {
      guidedIdea.error = t('Choose or paste a local literature folder first.', '请先选择或粘贴本地文献库文件夹。');
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.scanIdeaLiteratureFolder) {
      guidedIdea.error = t('Literature-folder scan API is unavailable.', '文献库文件夹扫描 API 不可用。');
      renderThread();
      return;
    }
    guidedIdea.sourceType = 'literature_folder';
    guidedIdea.literatureScanning = true;
    guidedIdea.error = null;
    renderThread();
    window.EU_API.scanIdeaLiteratureFolder({ path })
      .then(result => {
        guidedIdea.literatureScanning = false;
        guidedIdea.literatureScan = result;
        const folder = (result && result.folder) || {};
        guidedIdea.literatureFolder = folder.path || path;
        guidedIdea.literaturePdfCount = Number(folder.pdf_count || 0);
        applyGuidedIdeaSuggestion(result && result.suggested_payload);
        guidedIdea.resolved = result;
        renderThread();
        scheduleGuidedSlotSave('scan_idea_literature_folder');
      })
      .catch(err => {
        guidedIdea.literatureScanning = false;
        guidedIdea.error = err.message || String(err);
        renderThread();
        scheduleGuidedSlotSave('scan_idea_literature_folder_error');
      });
  }
  function runGuidedIdeaResolve() {
    if (!guidedIdea || guidedIdea.resolving) return;
    if (!window.EU_API || !window.EU_API.resolveIdeaSource) {
      guidedIdea.error = 'Idea source backend is unavailable.';
      renderThread();
      return;
    }
    guidedIdea.resolving = true;
    guidedIdea.error = null;
    renderThread();
    window.EU_API.resolveIdeaSource(guidedIdeaPayload()).then(result => {
      guidedIdea.resolving = false;
      guidedIdea.resolved = result;
      applyGuidedIdeaSuggestion(result && result.suggested_payload);
      renderThread();
      scheduleGuidedSlotSave('resolve_idea_source');
    }).catch(err => {
      guidedIdea.resolving = false;
      guidedIdea.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('resolve_idea_source_error');
    });
  }
  function runGuidedIdeaDiscover() {
    if (!guidedIdea || guidedIdea.discovering) return;
    const topic = String(guidedIdea.topic || guidedIdea.title || '').trim();
    if (!topic) {
      guidedIdea.error = t('Describe a frontier topic or review scope before literature discovery.', '请先描述前沿主题或 review 范围，再做文献发现。');
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.discoverIdeas) {
      guidedIdea.error = t('Literature discovery backend is unavailable.', '文献发现后端不可用。');
      renderThread();
      return;
    }
    guidedIdea.sourceType = 'frontier';
    guidedIdea.discovering = true;
    guidedIdea.error = null;
    renderThread();
    window.EU_API.discoverIdeas(Object.assign({}, guidedIdeaPayload(), {
      topic,
      journal: guidedIdea.journal || '',
      limit: 8,
      allow_network: !!guidedIdea.allowNetwork,
    })).then(result => {
      guidedIdea.discovering = false;
      guidedIdea.discovery = result;
      if (result && result.suggested_payload && result.status !== 'blocked_network_opt_in_required') {
        applyGuidedIdeaSuggestion(result.suggested_payload);
      }
      renderThread();
      scheduleGuidedSlotSave('discover_idea_literature');
    }).catch(err => {
      guidedIdea.discovering = false;
      guidedIdea.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('discover_idea_literature_error');
    });
  }
  function useGuidedIdeaDiscoveryCandidate(index) {
    if (!guidedIdea || !guidedIdea.discovery) return;
    const rows = Array.isArray(guidedIdea.discovery.idea_candidates) ? guidedIdea.discovery.idea_candidates : [];
    const row = rows[Number(index || 0)];
    if (!row) return;
    applyGuidedIdeaSuggestion(row.suggested_payload || {});
    guidedIdea.sourceType = 'frontier';
    guidedIdea.resolved = {
      ok: true,
      mode: 'frontier_discovery_candidate',
      resolved_source: row.source || null,
      suggested_payload: row.suggested_payload || {},
      source_adapter: {
        status: 'pubmed_candidate_selected',
        network_calls: guidedIdea.discovery.network_calls || 0,
        external_llm_calls: 0,
      },
    };
    guidedIdea.error = null;
    renderThread();
    scheduleGuidedSlotSave('use_discovered_idea_candidate');
  }
  function runGuidedIdeaMine() {
    if (!guidedIdea || guidedIdea.mining) return;
    if (!guidedIdeaHasInput()) {
      guidedIdea.error = 'Add a topic, source quote, title, DOI, or URL first.';
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.mineIdeas) {
      guidedIdea.error = 'Idea mining backend is unavailable.';
      renderThread();
      return;
    }
    guidedIdea.mining = true;
    guidedIdea.error = null;
    renderThread();
    window.EU_API.mineIdeas(guidedIdeaPayload()).then(result => {
      guidedIdea.mining = false;
      guidedIdea.result = result;
      guidedIdea.handoff = null;
      guidedIdea.project = null;
      const idea = guidedIdeaSelected();
      setVal({ question: idea ? (idea.idea_title || 'idea ledger') : 'idea ledger', analysis: 'pre-experiment' });
      markThrough('analysis', 'active');
      renderThread();
      renderAside();
      scheduleGuidedSlotSave('mine_idea');
    }).catch(err => {
      guidedIdea.mining = false;
      guidedIdea.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('mine_idea_error');
    });
  }
  function runGuidedIdeaPriorArt() {
    if (!guidedIdea || guidedIdea.priorArting) return;
    const idea = guidedIdeaSelected();
    if (!idea || !guidedIdea.result) {
      guidedIdea.error = 'Run local idea mining before prior-art check.';
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.checkIdeaPriorArt) {
      guidedIdea.error = 'Prior-art backend is unavailable.';
      renderThread();
      return;
    }
    guidedIdea.priorArting = true;
    guidedIdea.error = null;
    renderThread();
    window.EU_API.checkIdeaPriorArt({
      run_id: guidedIdea.result.run_id,
      idea_id: idea.idea_id,
      allow_network: !!guidedIdea.allowNetwork,
    }).then(result => {
      guidedIdea.priorArting = false;
      guidedIdea.prior = result;
      renderThread();
      scheduleGuidedSlotSave('check_idea_prior_art');
    }).catch(err => {
      guidedIdea.priorArting = false;
      guidedIdea.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('check_idea_prior_art_error');
    });
  }
  function runGuidedIdeaHandoff() {
    if (!guidedIdea || guidedIdea.handoffing) return;
    const idea = guidedIdeaSelected();
    if (!idea || !guidedIdea.result) {
      guidedIdea.error = 'Run local idea mining before creating a handoff.';
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.handoffIdea) {
      guidedIdea.error = 'Idea handoff backend is unavailable.';
      renderThread();
      return;
    }
    guidedIdea.handoffing = true;
    guidedIdea.error = null;
    renderThread();
    window.EU_API.handoffIdea({
      run_id: guidedIdea.result.run_id,
      idea_id: idea.idea_id,
      plan_edits: guidedIdea.planEdits || '',
    }).then(result => {
      guidedIdea.handoffing = false;
      guidedIdea.handoff = result;
      setVal({ analysis: 'handoff ready', draft: 'locked' });
      renderThread();
      renderAside();
      scheduleGuidedSlotSave('handoff_idea');
    }).catch(err => {
      guidedIdea.handoffing = false;
      guidedIdea.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('handoff_idea_error');
    });
  }
  function runGuidedIdeaCreateProject() {
    if (!guidedIdea || guidedIdea.projectCreating) return;
    const idea = guidedIdeaSelected();
    if (!idea || !guidedIdea.result) {
      guidedIdea.error = 'Run local idea mining before creating an Agent project.';
      renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.createIdeaAgentProject) {
      guidedIdea.error = 'Agent project seed backend is unavailable.';
      renderThread();
      return;
    }
    guidedIdea.projectCreating = true;
    guidedIdea.error = null;
    renderThread();
    window.EU_API.createIdeaAgentProject({
      run_id: guidedIdea.result.run_id,
      idea_id: idea.idea_id,
      plan_edits: guidedIdea.planEdits || '',
    }).then(result => {
      guidedIdea.projectCreating = false;
      guidedIdea.project = result;
      setVal({ analysis: 'Agent seed ready', draft: 'locked' });
      renderThread();
      renderAside();
      scheduleGuidedSlotSave('create_idea_agent_project');
    }).catch(err => {
      guidedIdea.projectCreating = false;
      guidedIdea.error = err.message || String(err);
      renderThread();
      scheduleGuidedSlotSave('create_idea_agent_project_error');
    });
  }

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
  function isGuidedIdeaIntent(text) {
    const s = String(text || '').toLowerCase();
    return /idea|study idea|research idea|paper|article|pdf|literature|frontier|review topic|研究想法|研究问题|挖掘|论文|文章|文献|综述|前沿|选题/.test(s);
  }

  /* ============== DOM render ============== */
  function renderThread() {
    const host = document.getElementById('gdThread');
    if (!host) return;
    host.innerHTML = thread.map(t => {
      if (t.typing) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body"><div class="m-bubble"><div class="typing"><span></span><span></span><span></span></div></div></div></div>`;
      if (t.guidedExtraction) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${renderGuidedExtractionCard()}</div></div>`;
      if (t.guidedReview) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${renderGuidedReviewCard()}</div></div>`;
      if (t.guidedAgent) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${renderGuidedAgentCard()}</div></div>`;
      if (t.guidedIdea) return `<div class="msg bot"><div class="m-ava">${icon('spark', 14)}</div><div class="m-body">${renderGuidedIdeaCard()}</div></div>`;
      if (t.diff) return diffCard();
      if (t.once) return ONCE[t.once] ? ONCE[t.once]() : '';
      if (t.card) {
        if (t.step === expandedStep) return CARD[t.step] ? CARD[t.step]() : '';
        const s = summaryOf(t.step);
        return `<div class="gd-collapsed"><span class="cc-mk">${icon('check', 10, 3)}</span><span class="cc-t">${s.t}</span><span class="cc-v">${s.v}</span>${s.edit ? `<button class="cc-edit" data-edit="${t.step}">${icon('sliders', 11)} Edit</button>` : ''}</div>`;
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
    if (!window.EU_API || !window.EU_API.loadGuidedDrafts) return;
    if (!force && (guidedDrafts.loading || guidedDrafts.data || guidedDrafts.error)) return;
    guidedDrafts = { loading: true, error: null, data: guidedDrafts.data || null };
    renderSessions();
    window.EU_API.loadGuidedDrafts({ limit: 20 }).then(data => {
      guidedDrafts = { loading: false, error: null, data: data };
      renderSessions();
      if (guidedFolderDialogMode) renderGuidedFolderDialog();
    }).catch(err => {
      guidedDrafts = { loading: false, error: err.message || String(err), data: null };
      renderSessions();
      if (guidedFolderDialogMode) renderGuidedFolderDialog();
    });
  }
  function removeLocalGuidedDraft(row) {
    if (!row || !row.id || !window.EU_API || !window.EU_API.removeGuidedDraft) return;
    const title = row.title || 'Guided draft';
    const ok = window.confirm(t(
      `Remove "${title}" from the Guided draft list? The local project folder will not be deleted.`,
      `从研究引导草稿列表移除“${title}”？本地项目文件夹不会被删除。`,
    ));
    if (!ok) return;
    window.EU_API.removeGuidedDraft({
      draft_id: row.id,
      project_dir: row.project_dir,
      delete_project_folder: false,
    }).then(result => {
      if (!result || result.ok === false) {
        throw new Error((result && (result.reason || result.error)) || 'remove_failed');
      }
      if (selectedGuidedDraft && selectedGuidedDraft.id === row.id) selectedGuidedDraft = null;
      guidedDrafts = { loading: false, error: null, data: result.drafts ? { drafts: result.drafts } : null };
      loadGuidedDrafts(true);
      pushBot(
        `Removed <strong>${esc(title)}</strong> from the Guided draft list. The project folder on disk was left untouched.`,
        `已从研究引导草稿列表移除 <strong>${esc(title)}</strong>。磁盘上的项目文件夹没有被删除。`,
      );
      renderSessions();
      renderThread();
    }).catch(err => {
      pushBot(
        `I could not remove that draft: <span class="mono">${esc(err.message || String(err))}</span>`,
        `无法移除这个草稿：<span class="mono">${esc(err.message || String(err))}</span>`,
      );
      renderThread();
    });
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
    if (guidedIdea) return 'idea_mining';
    if (guidedExtract) return 'data_extraction';
    if (guidedReview) return 'review_data';
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
    const src = activeExportSource();
    const idea = guidedIdea && guidedIdeaSelected ? guidedIdeaSelected() : null;
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
      active_export: src ? {
        label: src.label || src.database || 'active export',
        database: src.database || null,
        path: src.path || null,
        modules: src.summary && src.summary.modules,
        stays: src.summary && src.summary.stays,
      } : null,
      extraction: guidedExtract ? {
        path: guidedExtract.path || '',
        cohort: guidedExtract.cohort || 'adult_first',
        modules: guidedExtract.modules || [],
        format: guidedExtract.format || 'parquet',
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
      review: guidedReview ? {
        selected_ref: guidedReview.selectedRef || null,
        loaded: !!(guidedReview.patient || guidedReview.cohort),
        source_label: guidedSourceLabel(guidedReview.cohort || guidedReview.patient),
        cohort_size: guidedReview.cohort && guidedReview.cohort.summary && (guidedReview.cohort.summary.cohort_size || guidedReview.cohort.summary.entities),
      } : null,
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
      idea: guidedIdea ? {
        source_type: guidedIdea.sourceType || 'manual',
        topic: guidedIdea.topic || '',
        excerpt: guidedIdea.excerpt || '',
        title: guidedIdea.title || '',
        journal: guidedIdea.journal || '',
        year: guidedIdea.year || '',
        doi: guidedIdea.doi || '',
        pmid: guidedIdea.pmid || '',
        url: guidedIdea.url || '',
        allow_network: !!guidedIdea.allowNetwork,
        plan_edits: guidedIdea.planEdits || '',
        run_id: guidedIdea.result && guidedIdea.result.run_id,
        selected_idea_id: idea && idea.idea_id,
        handoff_id: guidedIdea.handoff && (guidedIdea.handoff.handoff_id || guidedIdea.handoff.run_id),
        agent_project_dir: guidedIdea.project && guidedIdea.project.project && guidedIdea.project.project.project_dir,
      } : null,
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
  function restoreGuidedSlotsFromSession(session) {
    const slots = session && session.slots && typeof session.slots === 'object' ? session.slots : {};
    if (slots.study_params && typeof slots.study_params === 'object') {
      studyParams = Object.assign({}, studyParams || {}, slots.study_params);
    }
    const active = slots.active_flow || session && session.goal;
    if (slots.extraction && typeof slots.extraction === 'object') {
      resetGuidedExtractionState();
      guidedExtract.path = slots.extraction.path || '';
      guidedExtract.cohort = slots.extraction.cohort || guidedExtract.cohort;
      guidedExtract.modules = Array.isArray(slots.extraction.modules) ? slots.extraction.modules.slice() : guidedExtract.modules;
      guidedExtract.format = slots.extraction.format || guidedExtract.format;
      guidedExtract.merge = !!slots.extraction.merge;
      guidedExtract.maxPatients = slots.extraction.max_patients == null ? null : Number(slots.extraction.max_patients);
      guidedExtract.scan = slots.extraction.scan || null;
      guidedExtract.registered = !!slots.extraction.registered;
      guidedExtract.result = slots.extraction.result || null;
    }
    if (slots.idea && typeof slots.idea === 'object') {
      resetGuidedIdeaState();
      guidedIdea.sourceType = slots.idea.source_type || guidedIdea.sourceType;
      guidedIdea.topic = slots.idea.topic || '';
      guidedIdea.excerpt = slots.idea.excerpt || '';
      guidedIdea.title = slots.idea.title || '';
      guidedIdea.journal = slots.idea.journal || '';
      guidedIdea.year = slots.idea.year || '';
      guidedIdea.doi = slots.idea.doi || '';
      guidedIdea.pmid = slots.idea.pmid || '';
      guidedIdea.url = slots.idea.url || '';
      guidedIdea.allowNetwork = !!slots.idea.allow_network;
      guidedIdea.planEdits = slots.idea.plan_edits || '';
      if (slots.idea.run_id) guidedIdea.result = { ok: true, run_id: slots.idea.run_id, ideas: [] };
    }
    if (slots.agent && typeof slots.agent === 'object') {
      resetGuidedAgentState();
      guidedAgent.question = slots.agent.question || guidedAgent.question;
      guidedAgent.jobId = slots.agent.job_id || null;
      guidedAgent.result = slots.agent.result || null;
    }
    if (slots.review && typeof slots.review === 'object') {
      resetGuidedReviewState();
      guidedReview.selectedRef = slots.review.selected_ref || null;
    }
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
      `First choose or create a <strong>local study folder</strong>. Each folder owns a separate Guided conversation and memory, like a Codex/Claude project context.`,
      `请先选择或创建一个<strong>本地研究文件夹</strong>。每个文件夹都有独立的 Guided 对话和记忆，类似 Codex/Claude 的项目上下文。`,
    ) });
    showGuidedDraftSetup(label || 'New local study');
    chips = [['Open/create study folder', '@foldernew']];
    renderThread();
    renderChips();
    return true;
  }
  function bindGuidedDraftMemory(draft) {
    if (!draft || !draft.project_dir || !window.EU_API || !window.EU_API.openGuidedProject) {
      return Promise.resolve(null);
    }
    return window.EU_API.openGuidedProject({
      project_dir: draft.project_dir,
      draft_id: draft.id || null,
      title: draft.title || 'local study',
      mode: 'local',
      context: guidedBackendContext(),
    }).then(result => {
      if (result && result.ok) {
        guidedCopilot = { loading: false, error: null, session: result.session || null, last: result };
      }
      return result;
    });
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
    const session = result && result.session ? result.session : null;
    guidedCopilot = { loading: false, error: null, session, last: result || guidedCopilot.last };
    const restoredFlow = restoreGuidedSlotsFromSession(session);
    currentId = 'frontdoor';
    thread = [];
    const title = (session && session.project_title) || (row && (row.title || row.study_id || row.run_label)) || 'local project';
    const path = row && row.project_dir ? compactPath(row.project_dir) : (session && session.project_dir ? compactPath(session.project_dir) : '~/easyicu/projects');
    const nounEn = kind === 'run' ? 'Agent run project' : 'guided draft';
    const nounZh = kind === 'run' ? 'Agent run 项目' : '引导草稿';
    thread.push({ bot: true, html: bi(
      `Opened <strong>${esc(title)}</strong> as this ${nounEn} context. Memory is scoped to <span class="mono">${esc(path)}</span>; Idea Mining and Agent Projects still own their own artifacts.`,
      `已切换到 <strong>${esc(title)}</strong> 这个${nounZh}上下文。记忆范围限定在 <span class="mono">${esc(path)}</span>；Idea Mining 和 Agent Projects 仍各自管理自己的 artifacts。`,
    ) });
    const restored = session && Array.isArray(session.messages) ? session.messages.map(threadFromSessionMessage).filter(Boolean) : [];
    if (restored.length) {
      restored.forEach(item => thread.push(item));
      if (session && session.handoff) thread.push({ bot: true, html: renderGuidedHandoffCard(session.handoff) });
    }
    if (restoredFlow && (guidedExtract || guidedIdea || guidedReview || guidedAgent)) {
      thread.push({ bot: true, html: bi(
        `Restored the saved setup for this folder. Continue editing here; required configuration stays inside Guided Copilot.`,
        `已恢复这个文件夹里保存的配置。你可以继续在这里编辑；必需配置仍留在 Guided Copilot 内完成。`,
      ) });
      if (restoredFlow === 'data_extraction' && guidedExtract) thread.push({ guidedExtraction: true });
      else if (restoredFlow === 'idea_mining' && guidedIdea) thread.push({ guidedIdea: true });
      else if (restoredFlow === 'review_data' && guidedReview) thread.push({ guidedReview: true });
      else if (restoredFlow === 'run_agent' && guidedAgent) thread.push({ guidedAgent: true });
    } else if (!restored.length && kind === 'run') {
      thread.push({ bot: true, html: bi(
        `This context is attached to an existing Agent run folder. Review artifacts or open Agent Projects; Guided will not rewrite the run outputs.`,
        `这个上下文关联到已有 Agent run 文件夹。你可以审阅 artifacts 或打开 Agent Projects；Guided 不会改写 run 输出。`,
      ) });
    } else if (!restored.length) {
      thread.push({ bot: true, html: bi(renderGuidedGoalCards(), renderGuidedGoalCards()) });
    }
    chips = kind === 'run'
      ? [['Review local artifacts', '@reviewLocalRun'], ['Open Agent Projects', '@openAgent'], ['Use active export for a new run', '@activeExport']]
      : [['Use active export', '@activeExport'], ['Continue conversation', '@noop'], ['Open Agent Projects', '@openAgent']];
    renderThread(); renderChips();
  }
  function openGuidedProjectMemory(row, el, kind) {
    if (!row || !row.project_dir || !window.EU_API || !window.EU_API.openGuidedProject) {
      pushBot(
        `This local project cannot be opened as scoped Guided memory yet.`,
        `这个本地项目暂时不能作为有范围的 Guided 记忆打开。`,
      );
      renderThread();
      return;
    }
    document.querySelectorAll('.gd-sess').forEach(s => s.classList.toggle('active', s === el));
    guidedCopilot = { loading: true, error: null, session: null, last: guidedCopilot.last };
    thread = [{ typing: true }];
    chips = [];
    renderThread(); renderChips();
    window.EU_API.openGuidedProject({
      project_dir: row.project_dir,
      draft_id: row.id || null,
      title: row.title || row.study_id || row.run_label || 'local project',
      mode: 'local',
      context: guidedBackendContext(),
    }).then(result => {
      thread = thread.filter(item => !item.typing);
      if (!result || !result.ok) {
        const reason = result && (result.reason || result.error) ? (result.reason || result.error) : 'unknown error';
        pushBot(`Could not open project memory: <span class="mono">${esc(reason)}</span>`, `无法打开项目记忆：<span class="mono">${esc(reason)}</span>`);
        renderThread();
        return;
      }
      restoreGuidedProjectThread(result, row, kind);
    }).catch(err => {
      thread = thread.filter(item => !item.typing);
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
      ['run_agent', 'agent', t('Run a Research Project', '运行研究项目'), t('Confirm a plan, then hand it to Agent Projects.', '确认计划后交接到研究项目。')],
    ];
    return `
      <div class="gd-frontdoor" data-guided-frontdoor>
        <div class="gdf-memory ${hasGuidedProjectMemory() ? 'ready' : ''}">
          <span>${icon(hasGuidedProjectMemory() ? 'check' : 'folder', 13)}</span>
          <div><strong>${hasGuidedProjectMemory() ? t('Project memory bound', '已绑定项目记忆') : t('Start by binding a local study folder', '先绑定本地研究文件夹')}</strong>
          <small>${hasGuidedProjectMemory()
            ? esc(compactPath((guidedCopilot.session && guidedCopilot.session.project_dir) || (selectedGuidedDraft && selectedGuidedDraft.project_dir) || ''))
            : t('Required setup stays inside Guided Copilot; every folder has its own conversation, settings, and handoff memory.', '必需配置都在 Guided Copilot 内完成；每个文件夹都有自己的对话、设置和交接记忆。')}</small></div>
          ${hasGuidedProjectMemory() ? '' : `<button class="btn sm" type="button" data-go="@foldernew">${t('New / open folder', '新建/打开文件夹')}</button>`}
        </div>
        <div class="gdf-head">
          <span class="gdf-kicker">${t('Choose a goal', '选择目标')}</span>
          <strong>${t('What should this local study folder do next?', '这个本地研究文件夹下一步要做什么？')}</strong>
          <span>${t('Pick a goal. If no folder is bound yet, I will ask you to create or open one first, then continue the selected workflow inside this conversation.', '选择目标。如果还没有绑定文件夹，我会先让你创建或打开一个文件夹，然后在本对话内继续刚才选择的流程。')}</span>
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
    if (requireGuidedProjectMemory(goal, label)) return;
    if (goal === 'idea_mining') {
      startGuidedIdeaFlow(label || guidedGoalMeta(goal).label_en);
      return;
    }
    if (goal === 'data_extraction') {
      startGuidedExtractionFlow(label || guidedGoalMeta(goal).label_en);
      return;
    }
    if (goal === 'review_data') {
      startGuidedReviewFlow(label || guidedGoalMeta(goal).label_en);
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
        `Guided Copilot 后端不可用，所以暂时不能创建可靠交接。`,
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
        try { window.__euGuidedHandoff = handoff.prefill || null; } catch (e) {}
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
      const goal = isGuidedIdeaIntent(text) ? 'idea_mining'
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
          pushBot(`Guided Copilot could not classify that request: <span class="mono">${esc(err.message || String(err))}</span>`, `Guided Copilot 无法识别这个请求：<span class="mono">${esc(err.message || String(err))}</span>`);
          renderThread();
        });
    });
    return true;
  }
  function guidedDraftPayload(label) {
    const src = activeExportSource();
    return {
      title: label || (BRANCH[branch] && BRANCH[branch].chip) || 'Guided study draft',
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
  function slugifyDraftFolder(text) {
    return String(text || 'guided-study').trim().toLowerCase()
      .replace(/[^a-z0-9._-]+/g, '-')
      .replace(/-{2,}/g, '-')
      .replace(/^[-._]+|[-._]+$/g, '')
      .slice(0, 64) || 'guided-study';
  }
  function renderGuidedFolderControls() {
    const host = document.getElementById('gdFolderControls');
    if (!host) return;
    host.innerHTML = `
      <div class="gd-folder-picker ${guidedFolderMenuOpen ? 'open' : ''}">
        <button class="gd-newbtn" type="button" data-newstudy data-folder-menu-toggle aria-haspopup="menu" aria-expanded="${guidedFolderMenuOpen ? 'true' : 'false'}" title="${t('Choose or create a local study folder', '选择或创建本地研究文件夹')}">
          ${icon('plus', 14)} ${t('New / open study folder', '新建/打开研究文件夹')}
        </button>
        ${guidedFolderMenuOpen ? `
          <div class="gd-folder-menu" role="menu" aria-label="${t('Study folder actions', '研究文件夹操作')}">
            <button class="gd-folder-menu-item" type="button" role="menuitem" data-folder-choice="new">
              <span class="gds-ico">${icon('folder', 14)}</span>
              <span><strong>${t('New blank study folder', '新建空白项目')}</strong><small>${t('Choose a parent folder, then create a metadata-only Guided project subfolder.', '先选择父目录，再创建仅元数据的 Guided 项目子文件夹。')}</small></span>
            </button>
            <button class="gd-folder-menu-item" type="button" role="menuitem" data-folder-choice="open">
              <span class="gds-ico">${icon('folder', 14)}</span>
              <span><strong>${t('Use existing folder', '使用现有文件夹')}</strong><small>${t('Open a Guided, Idea Mining, or Agent project folder as this conversation context.', '把已有 Guided、Idea Mining 或 Agent 项目文件夹作为当前对话上下文打开。')}</small></span>
            </button>
          </div>` : ''}
      </div>`;
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
        title: row.title || row.study_id || row.run_label || (kind === 'run' ? 'Agent run folder' : 'Guided study folder'),
        subtitle: kind === 'run'
          ? `${row.readiness_status || row.gate_status || 'analysis_only'} · ${row.artifact_count || 0} artifacts · ${fmtRunTime(row.updated_at)}`
          : `${row.status || 'metadata_only'} · ${row.depth || 'full'} · ${row.data_mode || 'local'} · ${fmtRunTime(row.updated_at || row.created_at)}`,
      });
    }
    localDraftRows().forEach(row => add(row, 'draft'));
    return rows.slice(0, 12);
  }
  function renderGuidedKnownProjectPicker() {
    const loading = guidedDrafts.loading;
    const error = guidedDrafts.error;
    const rows = guidedKnownProjectRows();
    if (!guidedKnownProjectsOpen) {
      return `
      <div class="gds-known collapsed">
        <div class="gds-known-head">
          <div><strong>${t('Recent local projects', '最近的本地项目')}</strong><span>${t('Optional shortcut. The list stays collapsed until you ask; no project is opened automatically.', '可选快捷入口。列表默认折叠，只有你主动展开才显示；不会自动打开任何项目。')}</span></div>
          <button class="btn sm" type="button" data-toggle-known-projects>${icon('history', 12)} ${t('Show recent', '显示最近项目')}</button>
        </div>
      </div>`;
    }
    return `
      <div class="gds-known">
        <div class="gds-known-head">
          <div><strong>${t('Recent local projects', '最近的本地项目')}</strong><span>${t('Shown only after you ask. Pick one as a shortcut, or use Browse/manual path below.', '仅在你主动展开后显示。可以选一个作为快捷入口，也可以用下方浏览/手动路径。')}</span></div>
          <button class="btn sm" type="button" data-refreshfolderchoices>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
        </div>
        ${loading ? `<div class="gds-known-empty">${icon('refresh', 12)} ${t('Loading recent local projects...', '正在加载最近的本地项目...')}</div>` : ''}
        ${error ? `<div class="gds-known-empty warn">${icon('info', 12)} ${esc(error)}</div>` : ''}
        ${!loading && !rows.length && !error ? `<div class="gds-known-empty">${t('No recent Guided project shortcuts yet. Create a new folder, browse a folder, or paste a local path below.', '暂无最近 Guided 项目快捷入口。可以新建文件夹、浏览文件夹，或在下方粘贴本地路径。')}</div>` : ''}
        ${rows.length ? `<div class="gds-known-list">${rows.map((row, i) => `
          <button class="gds-known-row" type="button" data-known-project="${i}">
            <span class="gds-known-kind">${row.kind === 'run' ? icon('history', 13) : icon('file', 13)}</span>
            <span><strong>${esc(row.title)}</strong><small>${esc(row.subtitle)}</small><code>${esc(compactPath(row.project_dir))}</code></span>
            <span class="gds-known-open">${icon('arrow', 13)}</span>
          </button>`).join('')}</div>` : ''}
      </div>`;
  }
  function renderGuidedFolderBrowser() {
    if (!guidedFolderBrowser.open) return '';
    const data = guidedFolderBrowser.data || {};
    const entries = Array.isArray(data.entries) ? data.entries : [];
    const shortcuts = Array.isArray(data.shortcuts) ? data.shortcuts : [];
    const currentPath = data.path || guidedFolderBrowser.path || '';
    const parent = data.parent || '';
    const failed = guidedFolderBrowser.error || (data && data.ok === false ? data.error : '');
    return `
      <div class="gds-browser" data-guided-folder-browser>
        <div class="gds-browser-head">
          <div>
            <strong>${t('Folder picker', '文件夹选择器')}</strong>
            <span>${t('Browse folders through the local EasyICU server; nothing is uploaded.', '通过本机 EasyICU 服务浏览文件夹；不会上传任何内容。')}</span>
          </div>
          <button class="btn sm ghost" type="button" data-folder-browser-close>${icon('close', 12)}</button>
        </div>
        <div class="gds-browser-path"><span>${t('Current', '当前')}</span><code>${esc(compactPath(currentPath) || t('Home folder', '主目录'))}</code></div>
        ${shortcuts.length ? `<div class="gds-browser-shortcuts">${shortcuts.map((item, i) => `
          <button class="btn sm" type="button" data-folder-browser-shortcut="${i}">${esc(item.name || 'Folder')}</button>`).join('')}</div>` : ''}
        ${failed ? `<div class="gds-browser-message warn">${icon('info', 12)} <span>${esc(String(failed))}</span></div>` : ''}
        <div class="gds-browser-list">
          ${guidedFolderBrowser.loading ? `<div class="gds-browser-empty">${icon('refresh', 13)} ${t('Loading folders...', '正在加载文件夹...')}</div>` : ''}
          ${!guidedFolderBrowser.loading && !entries.length ? `<div class="gds-browser-empty">${t('No child folders here. You can still choose the current folder if it is the project folder.', '这里没有下级文件夹。如果当前目录就是项目文件夹，也可以直接选择当前文件夹。')}</div>` : ''}
          ${!guidedFolderBrowser.loading && entries.map((entry, i) => `
            <button class="gds-browser-row" type="button" data-folder-browser-entry="${i}">
              <span class="gds-ico">${icon('folder', 13)}</span>
              <span><strong>${esc(entry.name || 'Folder')}</strong><code>${esc(compactPath(entry.path || ''))}</code></span>
              ${entry.hint ? `<em>${esc(entry.hint)}</em>` : ''}
            </button>`).join('')}
        </div>
        <div class="gds-browser-actions">
          <button class="btn sm" type="button" data-folder-browser-up ${parent ? '' : 'disabled'}>${icon('back', 12)} ${t('Up', '上一级')}</button>
          <span class="grow"></span>
          <button class="btn primary sm" type="button" data-folder-browser-use ${currentPath ? '' : 'disabled'}>${icon('check', 12)} ${t('Use this folder', '选择此文件夹')}</button>
        </div>
      </div>`;
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
    const host = document.getElementById('gdFolderDialogHost');
    if (!host) return;
    if (!guidedFolderDialogMode) {
      host.innerHTML = '';
      return;
    }
    const title = guidedFolderSeedTitle || (BRANCH[branch] && BRANCH[branch].chip) || 'New local study';
    const slug = guidedDraftFolderSlug || slugifyDraftFolder(title);
    const parentDir = guidedDraftParentDir || '~/easyicu/projects';
    const parentDisplay = compactPath(parentDir).replace(/\/$/, '');
    const projectPreview = `${parentDisplay}/guided-${slug || 'study'}-...`;
    const mode = guidedFolderDialogMode === 'open' ? 'open' : 'new';
    const openForReview = pendingGuidedGoal && pendingGuidedGoal.goal === 'review_data';
    const openActions = openForReview ? `
      <button class="btn primary sm" data-reviewexportfolder>${icon('eye', 13)} ${t('Review extracted data', '审阅已提取数据')}</button>
      <button class="btn sm" data-openprojectfolder>${icon('folder', 13)} ${t('Open project memory', '打开项目记忆')}</button>` : `
      <button class="btn primary sm" data-openprojectfolder>${icon('folder', 13)} ${t('Open project memory', '打开项目记忆')}</button>
      <button class="btn sm" data-reviewexportfolder>${icon('eye', 13)} ${t('Review extracted data', '审阅已提取数据')}</button>`;
    host.innerHTML = `
      <div class="gd-folder-backdrop" data-folder-dialog-close></div>
      <section class="gd-folder-dialog" data-folder-dialog data-draft-setup role="dialog" aria-modal="true" aria-label="${t('Choose a local study folder', '选择本地研究文件夹')}">
        <div class="gd-folder-dialog-head">
          <span class="gds-ico">${icon('folder', 15)}</span>
          <div>
            <strong>${t('Choose a local folder', '选择本地文件夹')}</strong>
            <span>${t('Open a project folder for Guided memory, or choose an EasyICU export folder when you want to review extracted data.', '打开项目文件夹用于 Guided 记忆；如果要审阅已提取数据，请选择 EasyICU export 文件夹。')}</span>
          </div>
          <button class="gd-folder-close" type="button" data-folder-dialog-close aria-label="${t('Close', '关闭')}">×</button>
        </div>
        <div class="gd-folder-tabs" role="tablist" aria-label="${t('Folder setup mode', '文件夹设置模式')}">
          <button class="${mode === 'new' ? 'active' : ''}" type="button" data-folder-choice="new">${t('New blank project', '新建空白项目')}</button>
          <button class="${mode === 'open' ? 'active' : ''}" type="button" data-folder-choice="open">${t('Use existing folder', '使用现有文件夹')}</button>
        </div>
        ${mode === 'open' ? `
          <div class="gds-choice">
            <div class="gds-choice-head"><strong>${t('Open project or extracted data folder', '打开项目或已提取数据文件夹')}</strong><span>${t('Required setup stays here instead of jumping to Classic Workspace. Use a project folder for conversation memory, or an EasyICU export folder to review previously extracted data.', '必需配置都留在这里完成，不强制跳到其他页面。项目文件夹用于对话记忆；EasyICU export 文件夹用于审阅之前提取的数据。')}</span></div>
            <div class="gds-path-row">
              <label class="gds-field"><span>${t('Local folder path', '本地文件夹路径')}</span><input data-existing-project-dir placeholder="${t('Paste a local project or EasyICU export folder path', '粘贴本地项目或 EasyICU 导出文件夹路径')}" autocomplete="off" /></label>
              <button class="btn sm" type="button" data-browseprojectfolder>${icon('folder', 13)} ${t('Browse...', '浏览...')}</button>
            </div>
            <div class="gds-path"><span>${t('Scope', '范围')}</span><code>local EasyICU project or export folder</code><small>${t('Use Browse to choose a folder. Path paste remains an advanced fallback for terminal workflows.', '请使用“浏览”选择文件夹。路径粘贴只作为终端工作流的高级 fallback。')}</small></div>
            ${renderGuidedFolderBrowser()}
            ${renderGuidedKnownProjectPicker()}
            <div class="row gap-8">${openActions}<button class="btn sm" data-folder-dialog-close>${t('Cancel', '取消')}</button></div>
            <div class="gds-status" data-project-open-status hidden></div>
          </div>` : `
          <div class="gds-choice">
            <div class="gds-choice-head"><strong>${t('Create new local study folder', '创建新的本地研究文件夹')}</strong><span>${t('Choose the parent folder first. EasyICU will create a new metadata-only project subfolder there; no patient rows, Agent run, or draft unlock is created.', '先选择父目录。EasyICU 会在其中创建新的仅元数据项目子文件夹；不会读取患者行、不会创建 Agent run、不会解锁草稿。')}</span></div>
            <label class="gds-field"><span>${t('Study title', '研究标题')}</span><input data-draft-title value="${attr(title)}" autocomplete="off" /></label>
            <label class="gds-field"><span>${t('Folder name', '文件夹名称')}</span><input data-draft-slug value="${attr(slug)}" autocomplete="off" /></label>
            <div class="gds-path-row">
              <label class="gds-field"><span>${t('Create inside folder', '创建到本地文件夹')}</span><input data-draft-parent-dir value="${attr(parentDir)}" placeholder="${t('Choose or paste a local parent folder', '选择或粘贴本地父目录')}" autocomplete="off" /></label>
              <button class="btn sm" type="button" data-browsedraftparent>${icon('folder', 13)} ${t('Browse...', '浏览...')}</button>
            </div>
            <div class="gds-path"><span>${t('Will create', '将创建')}</span><code>${esc(projectPreview)}</code><small>${t('The selected folder is the parent. The actual project folder gets a guided-* name so existing folders are not overwritten.', '所选文件夹是父目录。实际项目会使用 guided-* 子文件夹名称，避免覆盖已有文件夹。')}</small></div>
            ${renderGuidedFolderBrowser()}
            <div class="row gap-8"><button class="btn primary sm" data-createdraft>${icon('folder', 13)} ${t('Create study folder', '创建研究文件夹')}</button><button class="btn sm" data-folder-dialog-close>${t('Cancel', '取消')}</button></div>
          </div>`}
      </section>`;
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
      startGuidedReviewFlow(null);
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
        `${icon('check', 12)} <span>${t('Project memory opened. Restoring the conversation...', '项目记忆已打开，正在恢复对话...')}</span>`,
      );
      selectedGuidedDraft = {
        id: result.session && result.session.draft_id,
        title: result.session && result.session.project_title,
        project_dir: result.session && result.session.project_dir,
      };
      selectedGuidedRun = null;
      closeGuidedFolderDialog();
      restoreGuidedProjectThread(result, selectedGuidedDraft, 'draft');
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
  function createLocalGuidedDraft(label, folderSlug, parentDir) {
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
    const payload = guidedDraftPayload(text);
    payload.folder_slug = folderSlug || payload.folder_slug;
    if (parent) payload.parent_dir = parent;
    window.EU_API.createGuidedDraft(payload).then(result => {
      if (!result || result.ok === false || !result.draft) {
        throw new Error((result && (result.reason || result.error)) || 'guided_draft_create_failed');
      }
      selectedGuidedDraft = result.draft || null;
      loadGuidedDrafts(true);
      return bindGuidedDraftMemory(selectedGuidedDraft).then(opened => ({ result, opened }));
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
      pushBot(
        `Saved local guided draft <strong>${esc(title)}</strong> at <span class="mono">${esc(path)}</span>. This folder now owns the Guided conversation memory; Agent artifacts are created only when you start an auditable run.`,
        `已保存本地引导草稿 <strong>${esc(title)}</strong> 到 <span class="mono">${esc(path)}</span>。现在这个文件夹拥有本次 Guided 对话记忆；只有启动可审计 run 时才会生成 Agent artifacts。`,
      );
      chips = [['Use active export', '@activeExport'], ['Open Agent Projects', '@openAgent'], ['Continue conversation', '@noop']];
      renderThread(); renderChips();
      if (!continuePendingGuidedGoal()) {
        thread.push({ bot: true, html: bi(renderGuidedGoalCards(), renderGuidedGoalCards()) });
        renderThread();
      }
    }).catch(err => {
      pushBot(
        `Could not save the guided draft: <span class="mono">${esc(err.message || String(err))}</span>`,
        `无法保存引导草稿：<span class="mono">${esc(err.message || String(err))}</span>`,
      );
      renderThread();
    });
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
      chips = [['Open in Agent Projects', '@openAgent'], ['Use active export for a new run', '@activeExport']];
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
          `你好，我是 EasyICU <strong>引导式 Copilot</strong>。常用流程可以在这里完成，但每条对话都必须先绑定到本地研究文件夹。`,
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
          `Hi — I’m the EasyICU <strong>Research Copilot</strong>. I’ll drive the workspace by chat, and you can stop at any point.`,
          `你好，我是 EasyICU <strong>研究 Copilot</strong>。我会用对话驱动工作区，你可以随时停下。`,
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
      val: { question: () => BRANCH[branch].chip },
    },
    toData: {
      step: 'data', card: true,
      bot: [bi(`How should data enter the workspace?`, `数据要怎样进入工作区？`)],
      chips: () => [['What’s the difference?', '@why']],
      markStep: 'data',
      val: { question: () => BRANCH[branch].chip },
    },
    realConfirm: {
      step: 'data',
      bot: [bi(
        `Before we read local data, two things: this reads files on your machine and this first Agent run is a <strong>local preflight only</strong>: no external model call, no uploads, and never patient rows. Continue?`,
        `读取本地数据前先确认两点：这会读取你机器上的文件；第一次 Agent run 只是<strong>本地预检</strong>，不会外部模型调用、不会上传、也不会持久化患者行。继续吗？`,
      )],
      chips: () => [['Continue with local data', 'connect'], ['Use demo instead', '@usedemo']],
      markStep: 'data',
      val: { question: () => BRANCH[branch].chip },
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
    const host = document.getElementById('gdSessions');
    if (!host) return;
    const drafts = localDraftRows();
    const draftHtml = guidedDrafts.loading
      ? `<div class="gd-empty-local"><div class="ss-t">${t('Loading study folders', '正在加载研究文件夹')}</div><div class="ss-m">${t('Reading metadata-only Guided folder registry.', '正在读取仅元数据的 Guided 文件夹 registry。')}</div></div>`
      : guidedDrafts.error
        ? `<div class="gd-empty-local warn"><div class="ss-t">${t('Study folders unavailable', '研究文件夹不可用')}</div><div class="ss-m">${esc(guidedDrafts.error)}</div></div>`
        : drafts.length
          ? drafts.slice(0, 8).map((row, i) => `
            <div class="gd-sessline">
              <button class="gd-sess draft ${selectedGuidedDraft && selectedGuidedDraft.id === row.id ? 'active' : ''}" data-localdraft="${i}" title="${t('Open this folder conversation memory', '打开这个文件夹的对话记忆')}">
                <span class="ss-fold">${icon('file', 15)}</span>
                <span>
                  <span class="ss-t">${esc(row.title || 'Guided draft')}</span>
                  <span class="ss-m">${esc(row.status || 'metadata_only')} · ${esc(row.depth || 'full')} · ${esc(row.data_mode || 'demo')}</span>
                  <span class="ss-m mono">${row.project_dir ? esc(compactPath(row.project_dir)) : 'legacy registry-only draft'}</span>
                  <span class="ss-m mono">${esc(fmtRunTime(row.updated_at || row.created_at))}</span>
                </span>
              </button>
              <button class="gd-sess-action danger" type="button" data-remove-localdraft="${i}" title="${t('Remove from Guided draft list', '从草稿列表移除')}" aria-label="${t('Remove from Guided draft list', '从草稿列表移除')}">${icon('close', 12)}</button>
            </div>`).join('')
          : `<div class="gd-empty-local">
              <div class="ss-t">${t('No study folders yet', '还没有研究文件夹')}</div>
              <div class="ss-m">${t('Use New / open study folder to bind this conversation to a local project folder first.', '先使用“新建/打开研究文件夹”把这条对话绑定到本地项目文件夹。')}</div>
            </div>`;
    host.innerHTML = `
      <div class="gd-rail-sec in-list">${t('Study folders', '研究文件夹')} <button class="gd-refresh-mini" data-refreshdrafts title="${t('Refresh study folders', '刷新研究文件夹')}">${icon('refresh', 10)}</button></div>
      <div class="gd-rail-note"><strong>${t('Conversation memory', '对话记忆')}</strong><span>${t('Open these to continue setup inside Guided Copilot.', '打开这里可继续 Guided Copilot 内的配置。')}</span></div>
      ${draftHtml}`;
  }

  /* ============== screen ============== */
  S.guided = {
    section: 'guided', full: true,
    render() {
      reset();
      currentId = 'frontdoor';
      return `
      <div class="gd-shell">
        <div class="gd-top">
          <button class="gd-home-link" type="button" data-open="entry" aria-label="Back to EasyICU home" title="Back to EasyICU home">
            <span class="brand-mark">${icon('spark', 16)}</span>
            <span><span class="gd-name">Guided Copilot</span><span class="gd-mode">EasyICU · guided study</span></span>
          </button>
          <span class="grow"></span>
          <button class="btn sm" data-open="entry">${icon('back', 13)} ${t('Exit', '退出')}</button>
          <button class="btn sm" data-open="extraction">${icon('grid', 13)} ${t('Data workspace', '数据工作台')}</button>
        </div>
        <div class="gd-main threecol">
          <aside class="gd-rail">
            <div class="gd-rail-top" id="gdFolderControls"></div>
            <div class="gd-rail-sec">Workspace</div>
            <div class="gd-rail-list" id="gdSessions"></div>
            <div class="gd-rail-foot">
              <div class="gd-rail-utils" aria-label="${t('Guided study utilities', '研究引导工具')}">
                <button class="gd-utilbtn" type="button" data-open="entry" title="${t('Home', '主页')}" aria-label="${t('Home', '主页')}">${icon('back', 14)}</button>
                <button class="gd-utilbtn" type="button" data-open="settings" title="${t('Settings', '设置')}" aria-label="${t('Settings', '设置')}">${icon('gear', 14)}</button>
                <button class="gd-utilbtn lang" type="button" data-lang-toggle title="${t('Switch language', '切换语言')}" aria-label="${t('Switch language', '切换语言')}">
                  ${icon('globe', 14)} <span>${window.EU_LANG === 'zh' ? 'EN' : '中'}</span>
                </button>
              </div>
              <button class="btn sm block gd-data-workspace" data-open="extraction">${icon('grid', 13)} ${t('Data workspace', '数据工作台')}</button>
            </div>
          </aside>
          <div class="gd-conv">
            <div class="gd-scroll" id="gdScroll"><div class="gd-thread" id="gdThread" role="log" aria-live="polite" aria-label="Copilot conversation"></div></div>
            <div class="gd-suggest" id="gdSuggest"></div>
            <div class="gd-composer-wrap">
              <div class="gd-composer">
                <input class="gd-input" id="gdInput" placeholder="Reply, or tap an option above to continue…" autocomplete="off" />
                <button class="gd-send" id="gdSend">${icon('arrow', 16)}</button>
              </div>
              <div class="gd-foot-note">Guided Copilot · local first · nothing leaves your machine</div>
            </div>
          </div>
          <aside class="gd-aside">
            <div class="gd-aside-head"><div class="eyebrow">Building your study</div><div class="at">Study workspace</div><div class="asub">Assembles as we talk · edit any step</div></div>
            <div class="gd-aside-body" id="gdAsideBody"></div>
            <div class="gd-aside-foot"><div class="note ok" style="padding:9px 11px;"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t" style="font-size:11.5px;">Evidence-bound</div><div class="d" style="font-size:10.5px;">Draft stays gated until checks pass.</div></div></div></div>
          </aside>
        </div>
        <div id="gdFolderDialogHost"></div>
      </div>`;
    },
    afterRender(root) {
      renderGuidedFolderControls();
      renderGuidedFolderDialog();
      renderAside();
      renderSessions();
      loadGuidedDrafts();
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
              const routeLabel = b.route && b.route !== 'entry'
                ? (({extraction:'Data Extraction',patient:'Patient Review',cohort:'Cohort Statistics',crossdb:'Cross-DB Benchmark',agent:'Research Agent'}[b.route]) || 'the workspace')
                : '';
              pushBot(
                `Continuing from the dock${routeLabel ? ` — you were on <strong>${routeLabel}</strong>` : ''}. Want to turn that into a full study?`,
                `我会接着右下角 quick help 的上下文继续${routeLabel ? `：刚才你在 <strong>${routeLabel}</strong>` : ''}。要把它扩展成完整研究吗？`,
              );
              renderThread();
            }
          }, 700);
        }
      } catch (e) {}
      if (!bridged) go('frontdoor');
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
        if (e.target.closest('[data-folder-dialog-close]')) {
          closeGuidedFolderDialog();
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
          if (tok === '@noop') { pushUser(label); pushBot(`Go ahead — type your own wording in the box and I’ll work from it.`, `可以，直接在输入框里写你的表述，我会基于你的文字继续。`); renderThread(); return; }
          if (tok === '@typemine') { pushUser(label); pushBot(`Of course — type your research question in the box below and I’ll frame it with you.`, `当然可以。请在下面输入你的研究问题，我会帮你整理成可执行框架。`); renderThread(); const inp = document.getElementById('gdInput'); if (inp) inp.focus(); return; }
          if (tok === '@openAgent') { pushUser(label); location.hash = '#agent'; return; }
          if (tok === '@reviewLocalRun') { openGuidedRunReview(selectedGuidedRun, label); return; }
          if (tok === '@activeExport') { pushUser(label); dataMode = 'real'; go('realConfirm', label); return; }
          if (tok === '@foldernew') { pushUser(label || 'New / open study folder'); showGuidedDraftSetup('Guided study draft'); return; }
          if (tok === '@hintN') { handleText('use 30 patients'); return; }
          go(tok, goEl.classList.contains('suggest-chip') ? label : null);
          return;
        }
        const guidedGoalEl = e.target.closest('[data-guided-goal]');
        if (guidedGoalEl) {
          chooseGuidedGoal(guidedGoalEl.dataset.guidedGoal, stripText(guidedGoalEl.textContent));
          return;
        }
        const gxCohort = e.target.closest('[data-gx-cohort]');
        if (gxCohort && guidedExtract) {
          guidedExtract.cohort = gxCohort.dataset.gxCohort || 'adult_first';
          guidedExtract.error = null;
          renderThread();
          scheduleGuidedSlotSave('set_extraction_cohort');
          return;
        }
        const gxModule = e.target.closest('[data-gx-module]');
        if (gxModule && guidedExtract) {
          const key = gxModule.dataset.gxModule;
          guidedExtract.error = null;
          if (guidedExtract.modules.includes(key)) guidedExtract.modules = guidedExtract.modules.filter(m => m !== key);
          else guidedExtract.modules.push(key);
          renderThread();
          scheduleGuidedSlotSave('toggle_extraction_module');
          return;
        }
        const gxSet = e.target.closest('[data-gx-module-set]');
        if (gxSet && guidedExtract) {
          guidedExtract.error = null;
          updateGuidedExtractionModules(gxSet.dataset.gxModuleSet);
          renderThread();
          scheduleGuidedSlotSave('set_extraction_modules');
          return;
        }
        const gxFormat = e.target.closest('[data-gx-format]');
        if (gxFormat && guidedExtract) {
          guidedExtract.format = gxFormat.dataset.gxFormat || 'parquet';
          guidedExtract.error = null;
          renderThread();
          scheduleGuidedSlotSave('set_extraction_format');
          return;
        }
        const gxMax = e.target.closest('[data-gx-max]');
        if (gxMax && guidedExtract) {
          guidedExtract.maxPatients = gxMax.dataset.gxMax === 'all' ? null : Number(gxMax.dataset.gxMax || 500);
          guidedExtract.error = null;
          renderThread();
          scheduleGuidedSlotSave('set_extraction_max_patients');
          return;
        }
        if (e.target.closest('[data-gx-analyze]')) {
          scanGuidedExtractionPath();
          return;
        }
        if (e.target.closest('[data-gx-use-export]')) {
          registerGuidedModuleExport();
          return;
        }
        if (e.target.closest('[data-gx-run]')) {
          runGuidedExtractionJob();
          return;
        }
        if (e.target.closest('[data-gr-refresh]')) {
          loadGuidedReviewData();
          return;
        }
        const grEntity = e.target.closest('[data-gr-entity]');
        if (grEntity) {
          loadGuidedReviewData(grEntity.dataset.grEntity);
          return;
        }
        if (e.target.closest('[data-ga-run]')) {
          runGuidedAgentPreflight();
          return;
        }
        const giSource = e.target.closest('[data-gi-source]');
        if (giSource && guidedIdea) {
          guidedIdea.sourceType = giSource.dataset.giSource || 'manual';
          guidedIdea.error = null;
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
          loadGuidedLiteratureBrowser(guidedIdea && guidedIdea.literatureFolder);
          return;
        }
        if (e.target.closest('[data-gi-lit-scan]')) {
          scanGuidedLiteratureFolder();
          return;
        }
        if (e.target.closest('[data-gi-discover]')) {
          runGuidedIdeaDiscover();
          return;
        }
        const discoveryUse = e.target.closest('[data-gi-discovery-use]');
        if (discoveryUse) {
          useGuidedIdeaDiscoveryCandidate(discoveryUse.dataset.giDiscoveryUse || 0);
          return;
        }
        const giProvider = e.target.closest('[data-gi-provider]');
        if (giProvider) {
          guidedIdeaProvider = {
            provider: giProvider.dataset.giProvider || 'openai',
            loading: false,
            error: null,
            status: null,
          };
          requestGuidedIdeaProviderStatus(true);
          renderThread();
          scheduleGuidedSlotSave('set_idea_provider');
          return;
        }
        if (e.target.closest('[data-gi-provider-refresh]')) {
          requestGuidedIdeaProviderStatus(true);
          return;
        }
        if (e.target.closest('[data-lit-browser-close]')) {
          if (guidedLiteratureBrowser) guidedLiteratureBrowser.open = false;
          renderThread();
          return;
        }
        const litShortcut = e.target.closest('[data-lit-browser-shortcut]');
        if (litShortcut) {
          const shortcuts = guidedLiteratureBrowser && guidedLiteratureBrowser.data && Array.isArray(guidedLiteratureBrowser.data.shortcuts)
            ? guidedLiteratureBrowser.data.shortcuts
            : [];
          const row = shortcuts[Number(litShortcut.dataset.litBrowserShortcut || -1)];
          if (row && row.path) loadGuidedLiteratureBrowser(row.path);
          return;
        }
        const litEntry = e.target.closest('[data-lit-browser-entry]');
        if (litEntry) {
          const entries = guidedLiteratureBrowser && guidedLiteratureBrowser.data && Array.isArray(guidedLiteratureBrowser.data.entries)
            ? guidedLiteratureBrowser.data.entries
            : [];
          const row = entries[Number(litEntry.dataset.litBrowserEntry || -1)];
          if (row && row.path) loadGuidedLiteratureBrowser(row.path);
          return;
        }
        if (e.target.closest('[data-lit-browser-up]')) {
          const parent = guidedLiteratureBrowser && guidedLiteratureBrowser.data && guidedLiteratureBrowser.data.parent;
          if (parent) loadGuidedLiteratureBrowser(parent);
          return;
        }
        if (e.target.closest('[data-lit-browser-use]')) {
          const path = (guidedLiteratureBrowser && guidedLiteratureBrowser.data && guidedLiteratureBrowser.data.path) || (guidedLiteratureBrowser && guidedLiteratureBrowser.path) || '';
          if (guidedIdea && path) {
            guidedIdea.literatureFolder = path;
            guidedIdea.sourceType = 'literature_folder';
          }
          if (guidedLiteratureBrowser) guidedLiteratureBrowser.open = false;
          renderThread();
          scheduleGuidedSlotSave('set_literature_folder');
          return;
        }
        if (e.target.closest('[data-gi-resolve]')) {
          runGuidedIdeaResolve();
          return;
        }
        if (e.target.closest('[data-gi-mine]')) {
          runGuidedIdeaMine();
          return;
        }
        if (e.target.closest('[data-gi-prior]')) {
          runGuidedIdeaPriorArt();
          return;
        }
        if (e.target.closest('[data-gi-handoff]')) {
          runGuidedIdeaHandoff();
          return;
        }
        if (e.target.closest('[data-gi-project]')) {
          runGuidedIdeaCreateProject();
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
        // exit / classic
        const openEl = e.target.closest('[data-open]');
        if (openEl) { location.hash = '#' + openEl.dataset.open; return; }
        // clickable Study panel step → jump to / edit that step
        const stEl = e.target.closest('[data-study]');
        if (stEl) { jumpToStep(stEl.dataset.study); return; }
        // sessions rail
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

      shell.addEventListener('input', (e) => {
        const gxPath = e.target.closest('[data-gx-path]');
        if (gxPath && guidedExtract) {
          guidedExtract.path = gxPath.value;
          guidedExtract.scan = null;
          guidedExtract.scanError = null;
          guidedExtract.error = null;
          scheduleGuidedSlotSave('edit_extraction_path');
          return;
        }
        const gaQuestion = e.target.closest('[data-ga-question]');
        if (gaQuestion && guidedAgent) {
          guidedAgent.question = gaQuestion.value;
          guidedAgent.error = null;
          scheduleGuidedSlotSave('edit_agent_question');
          return;
        }
        const giField = e.target.closest('[data-gi-field]');
        if (giField && guidedIdea) {
          const key = giField.dataset.giField;
          if (key) guidedIdea[key] = giField.value;
          guidedIdea.error = null;
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
        const giPdfFile = e.target.closest('[data-gi-pdf-file]');
        if (giPdfFile && guidedIdea) {
          const file = giPdfFile.files && giPdfFile.files[0];
          ingestGuidedIdeaPdfFile(file);
          giPdfFile.value = '';
          return;
        }
        const giNetwork = e.target.closest('[data-gi-network]');
        if (giNetwork && guidedIdea) {
          guidedIdea.allowNetwork = !!giNetwork.checked;
          guidedIdea.error = null;
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
        handleText(v);
      }
      send.addEventListener('click', handleTextLocal);
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); handleTextLocal(); } });
    },
  };

  /* handle free text (from composer or hint chips) */
  function handleText(v) {
    if (busy) return;
    const conceptCode = findLocalConceptQuery(v);
    if (conceptCode) {
      answerConceptQuestion(v, conceptCode);
      return;
    }
    if (currentId === 'frontdoor' && isGuidedIdeaIntent(v)) {
      if (requireGuidedProjectMemory('idea_mining', v)) return;
      startGuidedIdeaFlow(v);
      guidedIdea.topic = v;
      renderThread();
      return;
    }
    if (currentId === 'frontdoor' && isGuidedExtractionIntent(v)) {
      if (requireGuidedProjectMemory('data_extraction', v)) return;
      startGuidedExtractionFlow(v);
      return;
    }
    if (currentId === 'frontdoor' && isGuidedReviewIntent(v)) {
      if (requireGuidedProjectMemory('review_data', v)) return;
      startGuidedReviewFlow(v);
      return;
    }
    if (currentId === 'frontdoor' && isGuidedAgentIntent(v)) {
      if (requireGuidedProjectMemory('run_agent', v)) return;
      startGuidedAgentFlow(v);
      return;
    }
    if (currentId === 'frontdoor') {
      if (sendGuidedShortcut(v)) return;
    }
    if (autop && /\b(stop|pause|halt|cancel)\b/i.test(v)) { autop = false; pushUser(v); pushBot(`Autopilot paused — tap a suggestion to continue manually.`, `自动流程已暂停。你可以点一个建议继续手动推进。`); renderThread(); return; }
    const fn = parseIntent(v);
    if (fn) { fn(); return; }
    // fallback: advance the primary path of the current state, echoing the text
    const map = { frame: 'toData', toData: null, toCohort: 'toConcepts', toConcepts: 'toExtract', toReview: 'toRun', toFindings: null };
    const next = map[currentId];
    if (next) { go(next, v); }
    else { pushUser(v); pushBot(
      `I’ll treat that as “<em>${esc(v)}</em>”. In this guided demo I move step by step — tap a suggestion to continue, or say <strong>“why?”</strong>, <strong>“go back”</strong>, <strong>“use 30 patients”</strong>, or <strong>“run the whole demo”</strong>.`,
      `我会把它理解为“<em>${esc(v)}</em>”。在引导模式里我会一步一步推进；你可以点建议继续，或说 <strong>“为什么”</strong>、<strong>“返回”</strong>、<strong>“用 30 个患者”</strong>、<strong>“跑完整演示”</strong>。`,
    ); renderThread(); }
  }

  function stripText(s) { return s.replace(/\s+/g, ' ').trim(); }
})();
