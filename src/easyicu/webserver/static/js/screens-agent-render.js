/* ============================================================
   screens-agent-render.js — fixture data + pure renderers for
   the Research Agent screen (screens-agent.js).

   First owner-file carve-out of the screens-agent.js monolith
   (see the file-size budget rule in CLAUDE.md / AGENTS.md).
   Everything here is PURE: demo/fixture study + block catalogs,
   artifact classifiers/label maps, and the artifact table/JSON
   renderers. The only external dependencies are the globals
   window.t (i18n) and window.icon (icon registry) — no closure
   state from screens-agent.js is referenced.

   Exposed via window.AGENT_RENDER; screens-agent.js rebinds the
   names at the top of its IIFE so call sites stay unchanged.
   This file MUST load before screens-agent.js in index.html.
   ============================================================ */
(function () {
  const t = window.t;
  const icon = window.icon;

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  }
  function escAttr(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }
  function boundedPngDataUrl(value) {
    const source = String(value || '');
    return /^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(source) ? source : '';
  }
  function bi(value) {
    return Array.isArray(value) ? t(value[0], value[1]) : esc(value);
  }

  /* ---------------- project + block fixture data ---------------- */
  const DEMO_STUDIES = [
    {
      id: 'sepsis', name: ['Sepsis mortality prediction', '脓毒症死亡率预测'],
      mode: 'analysis', status: 'gate', stage: 3, // 0 plan,1 build,2 analyze,3 gate,4 draft
      cohort: 'sepsis_mortality_demo', source: ['demo · 10 stays · 6 modules', '演示 · 10 次住院 · 6 模块'],
      question: [
        'Which first-24h bedside features best predict in-hospital mortality among Sepsis-3 patients, and how does adding lactate change calibration?',
        '在 Sepsis-3 患者中,入院前 24 小时的哪些床旁特征最能预测院内死亡?加入乳酸如何改变校准度?',
      ],
      runs: [
        ['run 07', ['ROC + calibration', 'ROC + 校准'], 'complete', '2m 14s', ['today 14:22', '今天 14:22']],
        ['run 06', ['Table 1 + missingness', 'Table 1 + 缺失审计'], 'complete', '1m 02s', ['today 11:08', '今天 11:08']],
        ['run 05', ['Cohort summary only', '仅队列摘要'], 'complete', '0:36', ['yesterday', '昨天']],
        ['run 04', ['Full plan (review-ready draft)', '完整计划(草稿待核验)'], 'blocked', '2m 41s', ['2 days ago', '2 天前']],
      ],
      signed: false,
    },
    {
      id: 'crossdb', name: ['Cross-DB sepsis replication', '跨库脓毒症复现'],
      mode: 'analysis', status: 'ready', stage: 2,
      cohort: 'sepsis_crossdb', source: ['demo · 3 databases · 6 concepts', '演示 · 3 个数据库 · 6 概念'],
      question: [
        'Does the sepsis mortality signal replicate across MIMIC-IV, eICU and AUMCdb, and where do feature distributions diverge?',
        '脓毒症死亡信号能否在 MIMIC-IV、eICU 和 AUMCdb 间复现?特征分布在哪里出现分歧?',
      ],
      runs: [
        ['run 02', ['Distribution deltas', '分布差异'], 'complete', '1m 48s', ['today 09:30', '今天 09:30']],
        ['run 01', ['Availability matrix', '可用性矩阵'], 'complete', '0:52', ['yesterday', '昨天']],
      ],
      signed: false,
    },
    {
      id: 'lactate', name: ['Early lactate research idea', '早期乳酸研究想法'],
      mode: 'analysis', status: 'idle', stage: 0,
      cohort: 'sepsis_mortality_demo', source: ['research idea · not yet run', '研究想法 · 尚未运行'],
      question: [
        'Test whether early lactate trajectory adds prognostic information after the idea has been confirmed in Idea Mining.',
        '在 Idea 挖掘确认后,检验早期乳酸轨迹是否提供额外预后信息。',
      ],
      runs: [
        ['idea 02', ['Feasibility handoff', '可行性交接'], 'complete', '0:41', ['today 13:05', '今天 13:05']],
        ['idea 01', ['Source check', '来源核查'], 'complete', '0:19', ['today 12:40', '今天 12:40']],
      ],
      signed: false,
    },
    {
      id: 'aki', name: ['AKI in CKD patients', 'CKD 患者的急性肾损伤'],
      mode: 'analysis', status: 'idle', stage: 0,
      cohort: 'aki_ckd_demo', source: ['demo · not yet run', '演示 · 尚未运行'],
      question: [
        'Among CKD patients, which interventions in the first 48h are associated with progression to KDIGO stage 3 AKI?',
        '在 CKD 患者中,前 48 小时的哪些干预与进展至 KDIGO 3 期 AKI 相关?',
      ],
      runs: [],
      signed: false,
    },
  ];

  const BLOCK_FAMILIES = [
    ['all', ['All blocks', '全部块']],
    ['discovery', ['Discovery', '发现']],
    ['analysis', ['Analysis', '分析']],
    ['nature', ['Nature package', 'Nature 套件']],
    ['submission', ['Submission', '投稿']],
  ];
  const BLOCK_LIBRARY = [
    {
      id: 'discovery_literature',
      family: 'discovery',
      title: ['Literature source intake', '文献来源录入'],
      stage: ['Idea intake', '想法录入'],
      icon: 'target',
      desc: ['Capture a paper, PDF, review, or clinical question as the external source for an idea.', '把文章、PDF、综述或临床问题记录为研究想法的外部来源。'],
      inputs: ['paper/PDF/topic', 'source metadata', 'uncertainty statement'],
      outputs: ['source card', 'bounded excerpt', 'research idea'],
      evidence: ['source id', 'metadata', 'excerpt hash'],
      route: 'ideas',
    },
    {
      id: 'discovery_feasibility',
      family: 'discovery',
      title: ['Outcome-blind feasibility', '结局盲可行性'],
      stage: ['Feasibility', '可行性'],
      icon: 'shield',
      desc: ['Check denominator, concept availability, windows, and portability before looking at target outcome differences.', '在查看目标结局差异前，先核查分母、概念可得性、时间窗和跨库可迁移性。'],
      inputs: ['research idea', 'candidate concepts', 'database scope'],
      outputs: ['feasibility ledger', 'risk plan', 'go/hold call'],
      evidence: ['coverage table', 'denominator summary', 'audit notes'],
      route: 'ideas',
    },
    {
      id: 'agent_handoff',
      family: 'analysis',
      title: ['Agent handoff object', 'Agent 交接对象'],
      stage: ['Handoff', '交接'],
      icon: 'agent',
      desc: ['Freeze the confirmed study question, cohort recipe, concepts, and analysis boundaries for an Agent run.', '冻结已确认的问题、队列配方、概念和分析边界，供 Agent 运行使用。'],
      inputs: ['confirmed idea', 'cohort recipe', 'allowed claims'],
      outputs: ['Agent project', 'analysis plan', 'run contract'],
      evidence: ['handoff package', 'handoff manifest'],
      route: 'agent',
    },
    {
      id: 'analysis_agent_run',
      family: 'analysis',
      title: ['Evidence-bound analysis run', '证据绑定分析运行'],
      stage: ['Run', '运行'],
      icon: 'play',
      desc: ['Run the local Agent project against an active export and write bounded artifacts for review.', '基于 active export 运行本地 Agent 项目，并写入有界产物供审阅。'],
      inputs: ['active export', 'study id', 'provider choice'],
      outputs: ['tables', 'figures', 'run manifest'],
      evidence: ['artifact hashes', 'run_context.json', 'quality_gate.json'],
      route: 'agent',
    },
    {
      id: 'evidence_review',
      family: 'analysis',
      title: ['Evidence review and sign-off', '证据审阅与签署'],
      stage: ['Review', '审阅'],
      icon: 'check',
      desc: ['Review local artifacts before any manuscript claim is unlocked or exported.', '在任何稿件论断解锁或导出前，审阅本地产物。'],
      inputs: ['run artifacts', 'claim ledger', 'reviewer confirmations'],
      outputs: ['human_signoff.json', 'readiness status'],
      evidence: ['sha256 registry', 'review checklist'],
      route: 'agent',
    },
    {
      id: 'nature_figure',
      family: 'nature',
      title: ['Nature figure block', 'Nature 图件块'],
      stage: ['Figure', '图件'],
      icon: 'viz',
      desc: ['Separate storyboard/reference figures from code-backed numeric result figures and preserve source-data provenance.', '区分 storyboard/reference 图和代码驱动的数值结果图，并保留 source-data provenance。'],
      inputs: ['caption', 'source data or storyboard prompt', 'figure type'],
      outputs: ['svg/png', 'source data', 'figure audit'],
      evidence: ['plotting code', 'axis/unit checks', 'storyboard prompt when applicable'],
      route: 'agent',
    },
    {
      id: 'nature_writing',
      family: 'nature',
      title: ['Nature writing block', 'Nature 写作块'],
      stage: ['Writing', '写作'],
      icon: 'file',
      desc: ['Draft or revise manuscript sections from evidence-bound claims, without promoting unsupported results.', '基于证据绑定论断起草或修订稿件章节，不放大无证据支撑的结果。'],
      inputs: ['claim ledger', 'section target', 'tone constraints'],
      outputs: ['section draft', 'revision notes', 'claim map'],
      evidence: ['evidence ids', 'blocked unsupported claims'],
      route: 'agent',
    },
    {
      id: 'nature_citation',
      family: 'nature',
      title: ['Nature citation block', 'Nature 引用块'],
      stage: ['Citation', '引用'],
      icon: 'list',
      desc: ['Attach citations to claims and keep literature provenance separate from result provenance.', '为论断绑定引用，并把文献 provenance 与结果 provenance 分开。'],
      inputs: ['claim text', 'candidate references', 'citation style'],
      outputs: ['citation map', 'reference list', 'unresolved claims'],
      evidence: ['DOI/PMID', 'source title', 'retrieval notes'],
      route: 'agent',
    },
    {
      id: 'nature_data',
      family: 'submission',
      title: ['Data availability block', '数据可用性块'],
      stage: ['Submission', '投稿'],
      icon: 'db',
      desc: ['Prepare Data Availability, source-data inventory, and FAIR metadata checks for submission.', '准备 Data Availability、source-data 清单和 FAIR metadata 核查。'],
      inputs: ['artifact list', 'repository plan', 'data restrictions'],
      outputs: ['data availability text', 'source-data checklist'],
      evidence: ['artifact manifest', 'repository/access notes'],
      route: 'agent',
    },
    {
      id: 'reviewer_response',
      family: 'submission',
      title: ['Reviewer response block', '审稿回复块'],
      stage: ['Revision', '返修'],
      icon: 'history',
      desc: ['Turn reviewer comments into response items, manuscript edits, and evidence-backed rebuttal notes.', '把审稿意见拆成回复项、稿件修改和证据支持的 rebuttal notes。'],
      inputs: ['reviewer comments', 'changed sections', 'new artifacts'],
      outputs: ['response letter', 'change map', 'open blockers'],
      evidence: ['diff references', 'artifact ids'],
      route: 'agent',
    },
  ];
  const NATURE_PACK = ['nature_figure', 'nature_writing', 'nature_citation', 'nature_data'];

  /* ---------------- pure status / text helpers ---------------- */
  function runStatusLabel(status) {
    const key = String(status || '').toLowerCase();
    const labels = {
      gate_reportable: t('verification passed', '核验通过'),
      reportable: t('verification passed', '核验通过'),
      // Backend readiness_status success outcomes (reporting/readiness.py) — the two
      // most common results of a completed real run; must not leak as raw snake_case.
      publication_ready: t('publication-ready', '可发表'),
      manuscript_ready: t('manuscript-ready', '手稿就绪'),
      analysis_only: t('analysis-only', '仅分析'),
      signed_analysis_only: t('signed · analysis-only', '已签署 · 仅分析'),
      diagnostic_only: t('diagnostic only', '诊断性'),
      awaiting_human_signoff: t('awaiting review', '待审阅'),
      signoff_stale: t('sign-off stale', '签署已失效'),
      blocked: t('blocked', '已阻断'),
      cancelled: t('cancelled', '已取消'),
      preflight: t('preflight', '预检'),
      ready: t('ready', '就绪'),
      imported: t('imported package', '已导入包'),
    };
    return labels[key] || status || t('analysis-only', '仅分析');
  }
  // Plain-language explanation of a run status, for pill tooltips so a newcomer
  // can learn the vocabulary without a separate legend.
  function runStatusHint(status) {
    const key = String(status || '').toLowerCase();
    const hints = {
      gate_reportable: t('All evidence checks passed — findings may be reported.', '所有证据检查已通过 —— 结论可报告。'),
      reportable: t('All evidence checks passed — findings may be reported.', '所有证据检查已通过 —— 结论可报告。'),
      publication_ready: t('All evidence checks passed and the manuscript is publication-ready.', '所有证据检查通过，手稿达到可发表状态。'),
      manuscript_ready: t('Evidence checks passed and a manuscript draft was written.', '证据检查通过，已写出手稿草稿。'),
      analysis_only: t('The run finished but claims stay locked until STRICT evidence + human sign-off pass.', '运行已完成，但在 STRICT 证据与人工签署通过前结论保持锁定。'),
      signed_analysis_only: t('A human signed off, but evidence verification still keeps claims non-reportable.', '已有人工签署，但证据核验仍使结论不可报告。'),
      diagnostic_only: t('Run produced diagnostics only — not enough to support a reportable claim.', '运行仅产出诊断信息 —— 不足以支撑可报告结论。'),
      awaiting_human_signoff: t('Evidence checks passed; a human reviewer still needs to sign off.', '证据检查已通过；仍需人工审阅者签署。'),
      signoff_stale: t('Artifacts changed after sign-off, so the sign-off no longer matches the files.', '签署后产物已变更，签署与文件不再一致。'),
      blocked: t('Evidence verification blocked this run; see the failing checks.', '证据核验已阻断本次运行；见未通过的检查。'),
      cancelled: t('The run was cancelled before it finished.', '运行在完成前被取消。'),
      preflight: t('A deterministic, local evidence preflight — no external model call.', '确定性的本地证据预检 —— 不调用外部模型。'),
      imported: t('A read-only completed analysis imported from a prior run.', '从既往运行导入的只读完成分析。'),
    };
    return hints[key] || '';
  }
  // Bilingual labels for the evidence-gate check ids (agent_runs.py). The
  // sign-off checklist is the most consequential reading moment of the agent
  // journey and must not fall back to English-only backend labels for zh users.
  function gateCheckLabel(check) {
    const id = String((check && (check.id || check.name)) || '').toLowerCase();
    const labels = {
      no_patient_rows_persisted: t('No patient rows persisted in agent artifacts', 'Agent 产物中不落任何患者行级数据'),
      provider_opt_in: t('LLM provider path resolved before invocation', '外部模型调用前已确认授权路径'),
      strict_evidence_bound_claims: t('All manuscript claims bind to known evidence', '所有稿件声明都绑定到已知证据'),
      strict_evidence_bound_sentences: t('All manuscript sentences bind to known evidence', '所有稿件句子都绑定到已知证据'),
      numeric_evidence_value_binding: t('All numeric manuscript claims match artifact values', '所有数值声明与产物数值一致'),
      human_signoff: t('Human sign-off before manuscript claims', '稿件声明前需人工签署'),
    };
    if (labels[id]) return labels[id];
    const raw = check && (check.label || check.title);
    return raw ? String(raw) : id.replace(/_/g, ' ');
  }
  function readableArtifactText(value) {
    return String(value || '')
      .replace(/\bgate_reportable\b/g, runStatusLabel('gate_reportable'))
      .replace(/\bawaiting_human_signoff\b/g, runStatusLabel('awaiting_human_signoff'))
      .replace(/\banalysis_only\b/g, runStatusLabel('analysis_only'))
      .replace(/\bdiagnostic_only\b/g, runStatusLabel('diagnostic_only'));
  }
  function firstValue() {
    for (let i = 0; i < arguments.length; i += 1) {
      const value = arguments[i];
      if (value !== null && value !== undefined && value !== '') return value;
    }
    return null;
  }
  function fmtCount(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toLocaleString() : '—';
  }

  /* ---------------- artifact classifiers / label maps ---------------- */
  function artifactKind(name) {
    const n = String(name || '').toLowerCase();
    if (n.includes('figure')) return 'figure';
    if (n.includes('scorecard')) return 'score';
    if (n.includes('workflow')) return 'workflow';
    if (n.includes('cohort')) return 'num';
    if (n.includes('ledger') || n.includes('gate') || n.includes('plan') || n.includes('draft')) return 'table';
    if (n.includes('missing')) return 'heat';
    if (n.includes('roc')) return 'roc';
    if (n.includes('calib')) return 'calib';
    return 'file';
  }
  function artifactTitle(name) {
    const n = String(name || '');
    const labels = {
      'run_context.json': t('Run context', '运行上下文'),
      'cohort_summary.json': t('Cohort summary', '队列摘要'),
      'table1_summary.json': t('Table 1 summary', 'Table 1 摘要'),
      'missingness_audit.json': t('Missingness audit', '缺失审计'),
      'roc_curve.json': t('ROC curve', 'ROC 曲线'),
      'calibration_curve.json': t('Calibration curve', '校准曲线'),
      'quality_gate.json': t('Evidence check', '证据核验'),
      'evidence_ledger.json': t('Evidence ledger', '证据账本'),
      'agent_plan.json': t('Agent plan', 'Agent 计划'),
      'manuscript_draft.json': t('Locked manuscript draft', '锁定论文草稿'),
      'benchmark_scorecard.json': t('Benchmark scorecard', 'Benchmark 记分卡'),
      'workflow_graph.json': t('Workflow graph', '工作流图谱'),
      'figure_gallery.json': t('Figure gallery', '图件画廊'),
      'source_run_manifest.json': t('Source run manifest', '原始运行清单'),
      'human_signoff.json': t('Human sign-off', '人工签署'),
    };
    if (labels[n]) return labels[n];
    return n.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
  function artifactCategory(name) {
    const n = String(name || '').toLowerCase();
    if (n === 'figure_gallery.json') return t('Figures', '图件');
    if (n.includes('scorecard')) return t('Scorecard', '记分卡');
    if (n.includes('workflow')) return t('Workflow', '流程');
    if (n.includes('ledger')) return t('Evidence', '证据');
    if (n.includes('gate')) return t('Quality check', '质量核验');
    if (n.includes('plan')) return t('Plan', '计划');
    if (n.includes('draft')) return t('Claims', '论断');
    if (n.includes('cohort')) return t('Cohort', '队列');
    if (n.includes('context')) return t('Context', '上下文');
    if (n.includes('manifest')) return t('Provenance', '溯源');
    return t('Artifact', '产物');
  }
  function artifactSummary(name) {
    const n = String(name || '').toLowerCase();
    const labels = {
      'figure_gallery.json': t('Task-specific figures rendered from this completed run.', '这道问题已渲染出的任务特异图件。'),
      'benchmark_scorecard.json': t('Plan, code, evidence binding, and safety scores for the question.', '这道问题的计划、代码、证据绑定与安全评分。'),
      'workflow_graph.json': t('Agent steps and handoffs from question to evidence review.', '从研究问题到证据审阅的 Agent 步骤与交接。'),
      'evidence_ledger.json': t('Artifact hashes, evidence ids, and privacy-audit status.', '产物哈希、证据 ID 与隐私审计状态。'),
      'quality_gate.json': t('Automated checks explaining why the run remains analysis-only.', '自动核验结果，说明为何仍保持 analysis-only。'),
      'agent_plan.json': t('The step-by-step analysis plan used by the Agent.', 'Agent 执行时使用的分步分析计划。'),
      'manuscript_draft.json': t('Locked claims and evidence ids; not a reportable manuscript.', '锁定论断及其证据 ID；不是可报告论文草稿。'),
      'run_context.json': t('Question, cohort, source run, and local project metadata.', '研究问题、队列、原始运行与本地项目元数据。'),
      'cohort_summary.json': t('Denominator, cohort basis, and outcome availability.', '分母、队列依据与结局可用性。'),
      'source_run_manifest.json': t('Original completed run provenance and import manifest.', '原始完成运行的溯源与导入清单。'),
    };
    return labels[n] || t('Whitelisted local artifact opened from the run folder.', '从运行文件夹读取的白名单本地产物。');
  }
  function artifactRank(name) {
    const order = [
      'figure_gallery.json',
      'benchmark_scorecard.json',
      'workflow_graph.json',
      'quality_gate.json',
      'evidence_ledger.json',
      'agent_plan.json',
      'manuscript_draft.json',
      'run_context.json',
      'cohort_summary.json',
      'source_run_manifest.json',
    ];
    const idx = order.indexOf(String(name || ''));
    return idx === -1 ? order.length : idx;
  }
  function defaultArtifactName(artifacts) {
    const rows = Array.isArray(artifacts) ? artifacts : [];
    const names = rows.map(a => a && (a.name || a.relative_path || '')).filter(Boolean);
    return names.find(n => n === 'figure_gallery.json')
      || names.find(n => n === 'benchmark_scorecard.json')
      || names[0]
      || null;
  }

  /* ---------------- artifact thumbnails + table renderers ---------------- */
  function thumb(kind) {
    if (kind === 'num') return `<div class="mono" style="font-size:30px;font-weight:500;color:var(--ink);">10</div>`;
    if (kind === 'file') return `<div style="color:var(--ink-3);">${icon('file', 34, 1.8)}</div>`;
    if (kind === 'figure') return `<div style="color:var(--accent);">${icon('viz', 34, 1.8)}</div>`;
    if (kind === 'score') return `<svg width="120" height="64" viewBox="0 0 120 64">${[0, 1, 2, 3, 4].map((r) => `<text x="14" y="${14 + r * 10}" font-size="7" fill="var(--ink-4)">D${r + 1}</text><rect x="34" y="${8 + r * 10}" width="62" height="5" fill="var(--hair-2)" rx="2"/><rect x="34" y="${8 + r * 10}" width="${38 + (r % 2) * 18}" height="5" fill="var(--ok)" opacity=".75" rx="2"/>`).join('')}</svg>`;
    if (kind === 'workflow') return `<svg width="120" height="64" viewBox="0 0 120 64"><circle cx="22" cy="32" r="7" fill="var(--accent)" opacity=".75"/><circle cx="60" cy="18" r="7" fill="var(--ok)" opacity=".75"/><circle cx="60" cy="46" r="7" fill="var(--warn)" opacity=".75"/><circle cx="98" cy="32" r="7" fill="var(--ink-3)" opacity=".55"/><path d="M29 31 L52 20 M29 33 L52 44 M68 20 L91 31 M68 44 L91 33" stroke="var(--hair-3)" fill="none"/></svg>`;
    if (kind === 'table') return `<svg width="120" height="64" viewBox="0 0 120 64">${[0, 1, 2, 3, 4].map(r => `<rect x="12" y="${8 + r * 11}" width="38" height="5" fill="${r === 0 ? 'var(--ink-3)' : 'var(--hair-3)'}" rx="1"/><rect x="56" y="${8 + r * 11}" width="22" height="5" fill="var(--hair-2)" rx="1"/><rect x="84" y="${8 + r * 11}" width="22" height="5" fill="var(--hair-2)" rx="1"/>`).join('')}</svg>`;
    if (kind === 'roc') return `<svg width="120" height="64" viewBox="0 0 120 64"><line x1="14" y1="54" x2="106" y2="8" stroke="var(--hair-2)" stroke-dasharray="2 3"/><line x1="14" y1="54" x2="14" y2="8" stroke="var(--hair-3)"/><line x1="14" y1="54" x2="106" y2="54" stroke="var(--hair-3)"/><path d="M14 54 Q 30 24 60 16 Q 90 11 106 9" stroke="var(--accent)" stroke-width="1.8" fill="none"/></svg>`;
    if (kind === 'calib') return `<svg width="120" height="64" viewBox="0 0 120 64"><line x1="14" y1="54" x2="106" y2="8" stroke="var(--hair-2)" stroke-dasharray="2 3"/><line x1="14" y1="54" x2="14" y2="8" stroke="var(--hair-3)"/><line x1="14" y1="54" x2="106" y2="54" stroke="var(--hair-3)"/><path d="M14 52 Q 40 40 62 30 Q 86 18 104 10" stroke="var(--ok)" stroke-width="1.8" fill="none"/></svg>`;
    return `<svg width="120" height="64" viewBox="0 0 120 64">${Array.from({ length: 6 }, (_, r) => Array.from({ length: 11 }, (_, c) => { const m = ((r * 7 + c * 3) % 10) > 7; return `<rect x="${10 + c * 9}" y="${8 + r * 7.5}" width="6.5" height="5.5" fill="${m ? 'var(--bad)' : 'var(--hair-3)'}" opacity="${m ? 0.65 : 1}" rx="0.5"/>`; }).join('')).join('')}</svg>`;
  }
  function scrubDataUrls(value) {
    if (Array.isArray(value)) return value.map(scrubDataUrls);
    if (!value || typeof value !== 'object') return value;
    const out = {};
    Object.keys(value).forEach(key => {
      if (key === 'data_url' || key === 'image_data_url') out[key] = '[embedded image hidden in JSON preview]';
      else out[key] = scrubDataUrls(value[key]);
    });
    return out;
  }
  function figureGallery(payload) {
    const figs = payload && Array.isArray(payload.figures) ? payload.figures : [];
    const visible = figs
      .map(row => ({ row, source: boundedPngDataUrl(row && (row.data_url || row.image_data_url)) }))
      .filter(item => item.row && item.source);
    if (!visible.length) return '';
    return `
      <div class="ag-figure-gallery">
        ${visible.map(({ row, source }) => `
          <figure>
            <img src="${escAttr(source)}" alt="${escAttr(row.label || row.relative_path || 'figure')}" />
            <figcaption><strong>${esc(row.label || 'figure')}</strong><span class="mono">${esc(row.relative_path || row.name || '')}</span></figcaption>
          </figure>`).join('')}
      </div>`;
  }
  function artifactScalar(value) {
    if (value === null || value === undefined || value === '') return '—';
    if (typeof value === 'boolean') return value ? t('yes', '是') : t('no', '否');
    if (typeof value === 'number') return Number.isFinite(value) ? value.toLocaleString() : '—';
    if (Array.isArray(value)) return `${value.length.toLocaleString()} ${t('items', '项')}`;
    if (typeof value === 'object') return `${Object.keys(value).length.toLocaleString()} ${t('fields', '字段')}`;
    return readableArtifactText(String(value));
  }
  function artifactKeyLabel(key) {
    const labels = {
      run_id: t('Run ID', '运行 ID'),
      run_type: t('Run type', '运行类型'),
      study_id: t('Study ID', '研究 ID'),
      status: t('Status', '状态'),
      mode: t('Mode', '模式'),
      question: t('Question', '问题'),
      local_first: t('Local-first', '本地优先'),
      cohort_size: t('Cohort size', '队列规模'),
      evidence_count: t('Evidence items', '证据项'),
      missing_evidence: t('Missing evidence', '缺失证据'),
      signed: t('Signed', '已签署'),
      provider: t('Provider', 'Provider'),
      database_scope: t('Database scope', '数据库范围'),
    };
    return labels[key] || String(key || '').replace(/_/g, ' ');
  }
  function artifactSummaryRows(payload, preferred) {
    const source = payload && typeof payload === 'object' ? payload : {};
    const keys = (preferred || []).concat(Object.keys(source)).filter((key, idx, arr) => arr.indexOf(key) === idx);
    return keys
      .filter(key => Object.prototype.hasOwnProperty.call(source, key))
      .filter(key => !/data_url|image_data_url/i.test(key))
      .map(key => [artifactKeyLabel(key), artifactScalar(source[key])])
      .filter(row => row[1] !== '—')
      .slice(0, 10);
  }
  function artifactTable(title, headers, rows, emptyText) {
    const safeRows = Array.isArray(rows) ? rows.filter(Boolean) : [];
    if (!safeRows.length) {
      return `<div class="ag-artifact-section"><div class="ag-artifact-section-title">${title}</div><div class="ag-artifact-empty">${emptyText || t('No table rows in this artifact.', '这个产物没有可展示的表格行。')}</div></div>`;
    }
    return `
      <div class="ag-artifact-section">
        <div class="ag-artifact-section-title">${title}</div>
        <div class="ag-artifact-table-wrap">
          <table class="ag-artifact-table">
            <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
            <tbody>
              ${safeRows.map(row => `<tr>${row.map(cell => `<td>${esc(artifactScalar(cell))}</td>`).join('')}</tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  }
  function objectArrayRows(rows, headers) {
    const arr = Array.isArray(rows) ? rows : [];
    return arr.slice(0, 12).map(row => headers.map(key => {
      if (!row || typeof row !== 'object') return row;
      return row[key];
    }));
  }
  function firstObjectArray(payload) {
    if (!payload || typeof payload !== 'object') return null;
    const found = Object.entries(payload).find(([, value]) => Array.isArray(value) && value.some(row => row && typeof row === 'object'));
    if (!found) return null;
    const keys = [];
    found[1].forEach(row => {
      if (!row || typeof row !== 'object') return;
      Object.keys(row).forEach(key => {
        if (keys.length < 5 && !keys.includes(key) && !/data_url|image_data_url/i.test(key)) keys.push(key);
      });
    });
    return keys.length ? { name: found[0], rows: found[1], keys } : null;
  }
  function stepRowsFrom(payload) {
    const rows =
      (payload && Array.isArray(payload.steps) && payload.steps)
      || (payload && Array.isArray(payload.nodes) && payload.nodes)
      || (payload && payload.workflow && Array.isArray(payload.workflow.steps) && payload.workflow.steps)
      || (payload && payload.agent_plan && Array.isArray(payload.agent_plan.steps) && payload.agent_plan.steps)
      || [];
    return rows.slice(0, 12).map((row, i) => [
      row.id || row.step || String(i + 1),
      row.title || row.name || row.label || row.stage || 'step',
      row.status || row.state || row.kind || 'planned',
      Array.isArray(row.evidence_ids) ? row.evidence_ids.join(', ') : (row.evidence || row.output || row.outputs || ''),
    ]);
  }
  function artifactStructuredView(name, payload) {
    const n = String(name || '').toLowerCase();
    const p = payload && typeof payload === 'object' ? payload : {};
    const gate = p.gate && typeof p.gate === 'object' ? p.gate : p;
    const sections = [];
    const summary = artifactSummaryRows(
      p,
      ['run_id', 'study_id', 'run_type', 'status', 'mode', 'database_scope', 'cohort_size', 'evidence_count', 'missing_evidence', 'local_first']
    );
    if (summary.length) {
      sections.push(artifactTable(t('Readable artifact summary', '可读产物摘要'), [t('Field', '字段'), t('Value', '值')], summary));
    }
    if (n.includes('figure_gallery')) {
      const figs = Array.isArray(p.figures) ? p.figures : [];
      sections.push(artifactTable(
        t('Figure table', '图件表'),
        [t('Label', '标签'), t('File', '文件'), t('Status', '状态')],
        figs.map(row => [row.label || row.name || 'figure', row.relative_path || row.path || row.name || '', row.status || t('available', '可用')]),
        t('No figures were embedded in this artifact.', '这个产物没有嵌入图件。')
      ));
    }
    if (n.includes('scorecard')) {
      const dims =
        (Array.isArray(p.dimensions) && p.dimensions)
        || (p.scorecard && Array.isArray(p.scorecard.dimensions) && p.scorecard.dimensions)
        || (Array.isArray(p.scores) && p.scores)
        || [];
      sections.push(artifactTable(
        t('Scorecard dimensions', '记分卡维度'),
        [t('Dimension', '维度'), t('Score', '评分'), t('Status', '状态'), t('Evidence', '证据')],
        dims.map(row => [
          row.dimension || row.name || row.id || row.metric || '',
          firstValue(row.score, row.value, row.points, row.grade),
          row.status || row.level || row.rating || '',
          row.evidence || row.note || row.reason || row.summary || '',
        ]),
        t('No per-dimension scores are present.', '没有逐维度评分。')
      ));
    }
    if (n.includes('workflow') || n.includes('plan')) {
      sections.push(artifactTable(
        t('Workflow steps', '工作流步骤'),
        [t('ID', 'ID'), t('Step', '步骤'), t('Status', '状态'), t('Evidence / output', '证据 / 产物')],
        stepRowsFrom(p),
        t('No workflow steps are present.', '没有工作流步骤。')
      ));
    }
    if (n.includes('gate')) {
      const checks = Array.isArray(gate.checks) ? gate.checks : (Array.isArray(p.checks) ? p.checks : []);
      sections.push(artifactTable(
        t('Quality gate checks', '质量核验项'),
        [t('Check', '检查项'), t('Status', '状态'), t('Evidence', '证据'), t('Reason', '原因')],
        checks.map(row => [row.id || row.name || row.check || '', row.status || row.result || '', firstValue(row.evidence_count, row.evidence, row.evidence_id), row.reason || row.message || row.note || '']),
        t('No quality checks are present.', '没有质量核验项。')
      ));
    }
    if (n.includes('ledger')) {
      const artifacts = Array.isArray(p.artifacts) ? p.artifacts : [];
      sections.push(artifactTable(
        t('Evidence artifact registry', '证据产物登记表'),
        [t('Artifact', '产物'), t('Category', '类别'), t('SHA-256', 'SHA-256'), t('Size', '大小')],
        artifacts.map(row => [row.name || row.relative_path || '', artifactCategory(row.name || row.relative_path || ''), row.sha256 || '', row.bytes == null ? '' : `${Number(row.bytes || 0).toLocaleString()} B`]),
        t('No artifact registry is present.', '没有产物登记表。')
      ));
    }
    if (n.includes('draft')) {
      const claims = Array.isArray(p.claims) ? p.claims : (Array.isArray(p.sentences) ? p.sentences : []);
      sections.push(artifactTable(
        t('Locked claims', '锁定论断'),
        [t('Claim', '论断'), t('Evidence IDs', '证据 ID'), t('Status', '状态')],
        claims.map(row => [row.text || row.claim || row.sentence || '', Array.isArray(row.evidence_ids) ? row.evidence_ids.join(', ') : '', row.status || p.status || 'locked']),
        t('No claims are present.', '没有论断。')
      ));
    }
    if (sections.length <= (summary.length ? 1 : 0)) {
      const firstTable = firstObjectArray(p);
      if (firstTable) {
        sections.push(artifactTable(
          t('Structured rows', '结构化行'),
          firstTable.keys.map(artifactKeyLabel),
          objectArrayRows(firstTable.rows, firstTable.keys),
          t('No structured rows are present.', '没有结构化行。')
        ));
      }
    }
    if (!sections.length) {
      sections.push(artifactTable(t('Readable artifact summary', '可读产物摘要'), [t('Field', '字段'), t('Value', '值')], [[t('Artifact', '产物'), name || 'artifact'], [t('Payload', '内容'), artifactScalar(p)]]));
    }
    return `
      <div class="ag-artifact-readable">
        <div class="ag-artifact-readable-head">
          <div>
            <div class="eyebrow">${t('Table view', '表格视图')}</div>
            <div class="ag-artifact-readable-title">${t('Raw JSON is kept for audit, but the default view is table-based.', '原始 JSON 保留用于审计；默认展示为表格。')}</div>
          </div>
          <span class="pill ok" style="height:22px;"><span class="dot"></span>${t('readable', '可读')}</span>
        </div>
        ${figureGallery(p)}
        ${sections.join('')}
      </div>`;
  }

  window.AGENT_RENDER = {
    DEMO_STUDIES, BLOCK_FAMILIES, BLOCK_LIBRARY, NATURE_PACK,
    runStatusLabel, runStatusHint, gateCheckLabel, readableArtifactText, firstValue, fmtCount,
    artifactKind, artifactTitle, artifactCategory, artifactSummary, artifactRank, defaultArtifactName,
    thumb, scrubDataUrls, figureGallery, artifactScalar, artifactKeyLabel,
    artifactSummaryRows, artifactTable, objectArrayRows, firstObjectArray, stepRowsFrom, artifactStructuredView,
  };
})();
