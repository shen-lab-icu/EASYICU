/* Guided Copilot's case-neutral display contracts and formatters.
   Owner: seeded classic Guided flow configuration and preflight gate display.
   The route shell keeps mutable conversation state; this module is immutable. */
(function () {
  'use strict';

  function bi(en, zh) { return { en, zh }; }

  function guidedJobEndError(message) {
    if (!message) return '';
    if (typeof message.error === 'string' && message.error.trim()) return message.error.trim();
    if (message.error && typeof message.error === 'object') return message.error.message || message.error.code || '';
    return message.status && message.status !== 'failed' ? message.status : '';
  }

  function guidedGateState(result) {
    const resultValid = !!result && typeof result === 'object' && !Array.isArray(result);
    const gate = resultValid && result.gate && typeof result.gate === 'object' && !Array.isArray(result.gate)
      ? result.gate : {};
    const checks = Array.isArray(gate.checks) ? gate.checks : [];
    const passed = checks.filter(check => check && check.passed === true).length;
    const ids = checks.map(check => check && check.id);
    const required = ['source_valid', 'denominator_resolved', 'quality_audited', 'no_bad_non_event_coverage', 'no_patient_rows_persisted', 'human_signoff'];
    const checksValid = checks.length >= required.length
      && checks.every(check => check && typeof check.id === 'string' && typeof check.passed === 'boolean')
      && new Set(ids).size === ids.length
      && required.every(id => ids.includes(id));
    const hardChecksPassed = checksValid && checks.every(check => check.id === 'human_signoff' || check.passed === true);
    const humanSignoff = checksValid ? checks.find(check => check.id === 'human_signoff') : null;
    const validPreflight = resultValid
      && result.run_type === 'preflight'
      && gate.status === 'analysis_only'
      && gate.reportable === false
      && gate.draft_unlocked === false
      && gate.reason === 'preflight_complete_human_signoff_required'
      && hardChecksPassed
      && humanSignoff.passed === false;
    return { gate, checks, passed, blocked: !validPreflight, contractValid: validPreflight };
  }

  function guidedGateCheckLabel(id) {
    const labels = {
      source_valid: t('Data source resolves to a registered export', '数据源解析为已注册导出'),
      denominator_resolved: t('Cohort denominator resolved', '队列分母已解析'),
      quality_audited: t('Feature quality audited', '特征质量已审计'),
      no_bad_non_event_coverage: t('No disqualifying non-event coverage gap', '非事件覆盖无致命缺口'),
      no_patient_rows_persisted: t('No patient rows persisted', '不落任何患者行级数据'),
      human_signoff: t('Human sign-off', '人工签署'),
    };
    return labels[String(id || '')] || String(id || '').replace(/_/g, ' ');
  }

  function guidedGateCheckRows(gateState) {
    const checks = (gateState && gateState.checks) || [];
    if (!checks.length) {
      return `<div class="gd-task failed"><span class="tk">${icon('alert', 10)}</span><span class="grow">${t('The gate returned no readable checks — fail closed, run treated as blocked.', '核验没有返回可读的检查项 —— 按失败关闭处理，运行视为受阻。')}</span></div>`;
    }
    return checks.map(check => {
      const id = check && check.id;
      const passed = check && check.passed === true;
      const pending = id === 'human_signoff' && !passed;
      const cls = passed ? 'done' : (pending ? 'queued' : 'failed');
      const mark = passed ? icon('check', 10, 3) : (pending ? icon('clock', 9) : icon('alert', 10));
      const suffix = passed ? '' : (pending
        ? ` <span style="color:var(--ink-4);">· ${t('pending review', '待审阅')}</span>`
        : ` <span style="color:var(--bad,#c0392b);font-weight:600;">· ${t('failed', '未通过')}</span>`);
      return `<div class="gd-task ${cls}"><span class="tk">${mark}</span><span class="grow">${guidedGateCheckLabel(id)}${suffix}</span></div>`;
    }).join('');
  }

  function guidedGateFailedNames(gateState) {
    return ((gateState && gateState.checks) || [])
      .filter(check => check && check.passed !== true && check.id !== 'human_signoff')
      .map(check => guidedGateCheckLabel(check && check.id));
  }

  const STUDY = Object.freeze([
    ['question', 'Research question', 'spark', '研究问题'],
    ['data', 'Data source', 'flask', '数据源'],
    ['cohort', 'Cohort', 'cohort', '队列'],
    ['concepts', 'Feature modules', 'layers', '特征模块'],
    ['extract', 'Extraction', 'extract', '数据抽取'],
    ['review', 'Review', 'eye', '审阅'],
    ['analysis', 'Analysis run', 'agent', '分析运行'],
    ['draft', 'Manuscript draft', 'shield', '稿件草稿'],
  ]);
  const STEP_INDEX = Object.freeze(Object.fromEntries(STUDY.map(([id], index) => [id, index])));

  const CLARIFY = Object.freeze({
    predict: { q: bi('Quick check before I build the plan — which mortality endpoint do you mean?', '建计划前先确认一下：你说的死亡结局是哪一种？'), opts: [['In-hospital mortality', 'in-hospital'], ['28-day mortality', '28-day'], ['ICU mortality', 'ICU']] },
    crossdb: { q: bi('How many databases should we compare?', '这次要比较多少个数据库？'), opts: [['All six', 'all 6 databases'], ['A focused three', '3 databases'], ['Let me pick', 'a custom set']] },
    quality: { q: bi('Should I audit everything, or focus on the modelling features?', '要审计全部模块，还是只关注建模特征？'), opts: [['Everything (19 modules)', 'all 19 modules'], ['Modelling features only', 'the modelling features']] },
  });

  const BRANCH = Object.freeze({
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
  });

  const DEPTH_ORDER = Object.freeze(['extract', 'review', 'full']);
  const DEPTH = Object.freeze({
    extract: {
      label: 'Extract only', goal: 'extract', chip: 'Just a cohort & data',
      hi: bi('Got it — an <strong>extract-only</strong> run. I’ll stop once your cohort is resolved and packaged, and you leave with analysis-ready frames plus a reproducible manifest.', '明白，这次走<strong>仅抽取</strong>。我会在队列解析并打包完成后停下，给你留下可分析的数据表和可复现 manifest。'),
    },
    review: {
      label: 'Extract + review', goal: 'review', chip: 'Data, then a visual review',
      hi: bi('Good — <strong>extract &amp; review</strong>. I’ll pull the data and prepare a quick visual review, then hand you a populated workspace. No agent run unless you ask.', '好的，走<strong>抽取 + 审阅</strong>。我会读取数据、生成快速可视化审阅，再把已填充的工作区交给你；除非你确认，不会启动 Agent run。'),
    },
    full: {
      label: 'Full study', goal: 'draft', chip: 'All the way to a review-ready draft',
      hi: bi('The full ride — <strong>extract → review → analyse → review-ready draft</strong>. Everything runs locally and the draft stays locked until checks pass.', '完整流程：<strong>抽取 → 审阅 → 分析 → 待核验草稿</strong>。所有步骤都在本机运行，检查通过前草稿保持锁定。'),
    },
  });

  function fmtInt(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString() : (fallback || 'n/a');
  }
  function fmtPct(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(1).replace(/\.0$/, '')}%` : (fallback || 'n/a');
  }
  function fmtNum(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? String(Math.round(number * 10) / 10) : (fallback || 'n/a');
  }
  function fmtFixed(value, digits, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits == null ? 1 : digits).replace(/\.0+$/, '') : (fallback || 'n/a');
  }
  function fmtP(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 'n/a';
    if (number < 0.001) return '<0.001';
    return number.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
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
    const date = new Date(String(value));
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  window.EU_GUIDED_CONTRACTS = Object.freeze({
    BRANCH, CLARIFY, DEPTH, DEPTH_ORDER, STEP_INDEX, STUDY,
    compactHash, compactPath, fmtFixed, fmtInt, fmtNum, fmtP, fmtPct, fmtRunTime,
    guidedGateCheckLabel, guidedGateCheckRows, guidedGateFailedNames,
    guidedGateState, guidedJobEndError,
  });
})();
