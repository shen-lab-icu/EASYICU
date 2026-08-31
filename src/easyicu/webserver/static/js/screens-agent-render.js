/* ============================================================
   screens-agent-render.js — fixture data + pure renderers for
   the Project Monitor screen (legacy route id: #agent).

   First owner-file carve-out of the screens-agent.js monolith
   (see the file-size budget rule in CLAUDE.md / AGENTS.md).
   Everything here is PURE: demo/fixture studies, artifact
   classifiers/label maps, and the artifact table/JSON
   renderers. The only external dependencies are the globals
   window.t (i18n) and window.icon (icon registry) — no closure
   state from screens-agent.js is referenced.

   Exposed via window.AGENT_RENDER; screens-agent.js rebinds the
   names at the top of its IIFE so call sites stay unchanged.
   This file MUST load before screens-agent.js in index.html.
   ============================================================ */
(function () {
  const { esc, escAttr } = window.EU_HTML;
  const t = window.t;
  const icon = window.icon;

  function boundedPngDataUrl(value) {
    const source = String(value || '');
    return /^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(source) ? source : '';
  }
  /* ---------------- project fixture data ---------------- */
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
    if (value === null || value === undefined || value === '') return '—';
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
      'literature_evidence.json': t('Literature evidence', '文献证据'),
      'scientific_plan_review.json': t('Scientific plan review', '科学计划审阅'),
      'manuscript_draft.json': t('Locked manuscript draft', '锁定论文草稿'),
      'manuscript_provenance.json': t('Evidence-bound manuscript reader', '证据绑定论文阅读器'),
      'benchmark_scorecard.json': t('Evaluation scorecard', '评估记分卡'),
      'workflow_graph.json': t('Workflow graph', '工作流图谱'),
      'figure_gallery.json': t('Figure gallery', '图件画廊'),
      'result_tables.json': t('Research result tables', '科研结果表'),
      'system_validation_report.json': t('System validation dossier', '系统验证报告'),
      'system_validation_report_receipt.json': t('System validation receipt', '系统验证回执'),
      'source_run_manifest.json': t('Source run manifest', '原始运行清单'),
      'human_signoff.json': t('Human sign-off', '人工签署'),
    };
    if (labels[n]) return labels[n];
    return n.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
  function artifactCategory(name) {
    const n = String(name || '').toLowerCase();
    if (n === 'figure_gallery.json') return t('Figures', '图件');
    if (n === 'result_tables.json') return t('Result tables', '结果表');
    if (n === 'system_validation_report.json') return t('System validation', '系统验证');
    if (n.includes('scorecard')) return t('Scorecard', '记分卡');
    if (n.includes('workflow')) return t('Workflow', '流程');
    if (n.includes('ledger')) return t('Evidence', '证据');
    if (n.includes('literature')) return t('Literature', '文献');
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
      'result_tables.json': t('Bounded aggregate table previews from registered Research Agent evidence.', '来自 Research Agent 已登记证据的有界聚合表格预览。'),
      'system_validation_report.json': t('Source-bound engineering validation; explicitly not a clinical manuscript.', '源绑定的工程验证报告；明确不是临床论文。'),
      'system_validation_report_receipt.json': t('Digest receipt for the engineering-only report and rendered document.', '仅限工程用途的报告及渲染文档摘要回执。'),
      'benchmark_scorecard.json': t('Plan, code, evidence binding, and safety scores for this research run.', '本次研究运行的计划、代码、证据绑定与安全评分。'),
      'workflow_graph.json': t('Agent steps and handoffs from question to evidence review.', '从研究问题到证据审阅的 Agent 步骤与交接。'),
      'evidence_ledger.json': t('Artifact hashes, evidence ids, and privacy-audit status.', '产物哈希、证据 ID 与隐私审计状态。'),
      'quality_gate.json': t('Automated checks explaining why the run remains analysis-only.', '自动核验结果，说明为何仍保持 analysis-only。'),
      'agent_plan.json': t('The step-by-step analysis plan used by the Agent.', 'Agent 执行时使用的分步分析计划。'),
      'literature_evidence.json': t('Search provenance, article metadata, and exact plan-step citation bindings.', '检索溯源、文章元数据以及计划步骤的精确文献绑定。'),
      'scientific_plan_review.json': t('Digest-bound multi-dimensional review before the plan can be approved.', '计划批准前的摘要绑定多维科学审阅。'),
      'manuscript_draft.json': t('Locked claims and evidence ids; not a reportable manuscript.', '锁定论断及其证据 ID；不是可报告论文草稿。'),
      'manuscript_provenance.json': t('Click any bound number to inspect its exact JSON field, step, and registered code/data lineage.', '点击正文中的任一绑定数字，可查看准确 JSON 字段、分析步骤及已登记的代码/数据链路。'),
      'run_context.json': t('Question, cohort, source run, and local project metadata.', '研究问题、队列、原始运行与本地项目元数据。'),
      'cohort_summary.json': t('Denominator, cohort basis, and outcome availability.', '分母、队列依据与结局可用性。'),
      'source_run_manifest.json': t('Original completed run provenance and import manifest.', '原始完成运行的溯源与导入清单。'),
    };
    return labels[n] || t('Whitelisted local artifact opened from the run folder.', '从运行文件夹读取的白名单本地产物。');
  }
  function artifactRank(name) {
    const order = [
      'system_validation_report.json',
      'figure_gallery.json',
      'result_tables.json',
      'benchmark_scorecard.json',
      'workflow_graph.json',
      'quality_gate.json',
      'evidence_ledger.json',
      'agent_plan.json',
      'scientific_plan_review.json',
      'literature_evidence.json',
      'manuscript_provenance.json',
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
    return names.find(n => n === 'system_validation_report.json')
      || names.find(n => n === 'figure_gallery.json')
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
      return `<div class="ag-artifact-section"><div class="ag-artifact-section-title">${esc(title)}</div><div class="ag-artifact-empty">${esc(emptyText || t('No table rows in this artifact.', '这个产物没有可展示的表格行。'))}</div></div>`;
    }
    return `
      <div class="ag-artifact-section">
        <div class="ag-artifact-section-title">${esc(title)}</div>
        <div class="ag-artifact-table-wrap">
          <table class="ag-artifact-table">
            <thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead>
            <tbody>
              ${safeRows.map(row => `<tr>${row.map(cell => `<td>${esc(artifactScalar(cell))}</td>`).join('')}</tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  }
  /* A long readable artifact is a stack of ag-artifact-section blocks. Once
     there are several, the reader cannot see what the artifact contains
     without scrolling it end to end, so lead with the section index. The
     titles are read back from the already-escaped section markup, which
     keeps the strip and the body from drifting apart. */
  function artifactContentsStrip(sections) {
    const titles = (Array.isArray(sections) ? sections : []).map(html => {
      const match = /<div class="ag-artifact-section-title">([^<]*)<\/div>/.exec(String(html || ''));
      return match ? match[1] : '';
    }).filter(Boolean);
    if (titles.length < 3) return '';
    return `<nav class="ag-artifact-contents" aria-label="${escAttr(t('Artifact contents', '产物内容'))}">
      <small>${esc(t(`${titles.length} sections in this artifact`, `本产物共 ${titles.length} 个区块`))}</small>
      <div>${titles.map((title, index) => `<span><b>${index + 1}</b>${title}</span>`).join('')}</div>
    </nav>`;
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
    return rows.slice(0, 12).map((row, i) => {
      const expectedOutputs = Array.isArray(row.expected_outputs)
        ? row.expected_outputs.join(', ')
        : '';
      const outputs = Array.isArray(row.outputs) ? row.outputs.join(', ') : row.outputs;
      const literature = Array.isArray(row.literature_citation_keys)
        ? row.literature_citation_keys.map(key => `literature:${key}`).join(', ')
        : '';
      const evidenceIds = Array.isArray(row.evidence_ids) ? row.evidence_ids.join(', ') : '';
      const evidence = evidenceIds || row.evidence || literature;
      return [
        row.step_id || row.id || row.step || String(i + 1),
        row.intent || row.title || row.name || row.label || row.stage || row.method || 'step',
        row.status || row.state || row.kind || (row.planned_analysis_role ? `planned · ${row.planned_analysis_role}` : 'planned'),
        expectedOutputs || evidence || row.output || outputs || '',
      ];
    });
  }
  function manuscriptProvenanceView(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const claims = Array.isArray(p.claims) ? p.claims.slice(0, 240) : [];
    const blocks = Array.isArray(p.article_blocks) ? p.article_blocks.slice(0, 240) : [];
    const claimMap = new Map(claims.map(row => [String(row && row.claim_id || ''), row || {}]));
    const readableText = value => {
      const source = String(value || '')
        .replace(/\s*\[(?!@)[A-Za-z_][A-Za-z0-9_.-]*\]/g, '')
        .replace(/\s+([,.;:)])/g, '$1');
      const tokens = source.split(/(\*\*[^*]+\*\*|\[@[^\]]+\])/g).filter(Boolean);
      return tokens.map(token => {
        if (/^\*\*[^*]+\*\*$/.test(token)) return `<strong>${esc(token.slice(2, -2))}</strong>`;
        if (/^\[@[^\]]+\]$/.test(token)) {
          const key = token.slice(2, -1);
          return `<span class="gpi-reader-citation" title="${esc(key)}">[ref]</span>`;
        }
        return esc(token);
      }).join('');
    };
    const claimEvidenceAttrs = claim => {
      const evidence = claim && claim.evidence && typeof claim.evidence === 'object' ? claim.evidence : {};
      const evidenceId = String(evidence.evidence_id || '').trim();
      const sha256 = String(evidence.sha256 || '').trim().toLowerCase();
      if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) return '';
      return ` data-gpi-evidence-open data-evidence-id="${escAttr(evidenceId)}" data-evidence-sha256="${escAttr(sha256)}" data-evidence-kind="${escAttr(String(evidence.kind || 'statistic'))}" data-evidence-label="${escAttr(t('Exact result source', '准确结果来源'))}" data-evidence-pointer="${escAttr(String(claim.source_json_pointer || ''))}" data-evidence-source-value="${escAttr(String(claim.source_value == null ? '' : claim.source_value))}"`;
    };
    const renderSegments = value => (Array.isArray(value) ? value : []).map(segment => {
      const text = readableText(segment && segment.text || '');
      const claimId = String(segment && segment.claim_id || '');
      if (!segment || segment.kind !== 'claim' || !claimMap.has(claimId)) return text;
      const claim = claimMap.get(claimId) || {};
      const evidenceAttrs = claimEvidenceAttrs(claim);
      return `<button type="button" class="gpi-bound-number" id="claim-${escAttr(claimId)}" data-gpi-claim="${escAttr(claimId)}"${evidenceAttrs} aria-controls="gpi-claim-detail-${escAttr(claimId)}" aria-expanded="false" title="${escAttr(evidenceAttrs ? t('Open result evidence preview', '打开结果证据预览') : t('Open evidence lineage', '查看证据链路'))}">${text}</button>`;
    }).join('');
    const reportFigures = figureGallery(p.figure_gallery || {});
    const article = blocks.map(block => {
      const content = renderSegments(block && block.segments);
      const headingText = (Array.isArray(block && block.segments) ? block.segments : [])
        .map(segment => String(segment && segment.text || '')).join('').trim();
      const figureInsert = reportFigures && block && block.kind === 'heading'
        && Number(block.level || 2) === 2 && /^Discussion$/i.test(headingText)
        ? `<section class="gpi-article-figure-insert"><div class="gpi-article-figure-head"><span>${esc(t('Registered result figures', '已登记结果图'))}</span><h2>${esc(t('Main visual results', '主要可视化结果'))}</h2><p>${esc(p.figure_gallery && p.figure_gallery.presentation_variant ? t('Re-rendered from digest-verified source tables. Original run figures remain unchanged.', '根据摘要核验后的源数据表重新排版；原始运行图件保持不变。') : t('Figures registered by this run.', '本次运行登记的图件。'))}</p></div>${reportFigures}</section>`
        : '';
      if (block && block.kind === 'heading') {
        const level = Math.max(2, Math.min(4, Number(block.level || 2)));
        return `${figureInsert}<h${level}>${content}</h${level}>`;
      }
      return `${figureInsert}<p>${content}</p>`;
    }).join('');
    const evidenceButton = (row, label, pointer, sourceValue) => {
      const evidenceId = String(row && row.evidence_id || '').trim();
      const sha256 = String(row && row.sha256 || '').trim().toLowerCase();
      if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) {
        return esc(label || evidenceId || t('Unavailable', '不可用'));
      }
      return `<button type="button" class="gpi-evidence-open" data-gpi-evidence-open data-evidence-id="${escAttr(evidenceId)}" data-evidence-sha256="${escAttr(sha256)}" data-evidence-kind="${escAttr(String(row.kind || 'artifact'))}" data-evidence-role="${escAttr(String(row.role || ''))}" data-evidence-label="${escAttr(String(label || evidenceId))}" data-evidence-pointer="${escAttr(String(pointer || ''))}" data-evidence-source-value="${escAttr(String(sourceValue == null ? '' : sourceValue))}">${esc(label || evidenceId)}</button>`;
    };
    const lineageTable = (source, rows, pointer, sourceValue) => {
      const entries = [];
      if (source && source.evidence_id) {
        entries.push({ ...source, role: t('Exact result source', '准确结果来源'), kind: source.kind || 'statistic' });
      }
      (Array.isArray(rows) ? rows : []).forEach(row => entries.push(row || {}));
      if (!entries.length) return `<p class="gpi-claim-boundary">${esc(t('No registered evidence artifacts.', '没有登记证据产物。'))}</p>`;
      return `<div class="ag-artifact-section"><div class="ag-artifact-section-title">${esc(t('Open registered evidence', '打开已登记证据'))}</div><div class="ag-artifact-table-wrap"><table class="ag-artifact-table"><thead><tr><th>${esc(t('Role', '角色'))}</th><th>${esc(t('Type', '类型'))}</th><th>${esc(t('Preview', '预览'))}</th><th>SHA-256</th></tr></thead><tbody>${entries.map(row => `<tr><td>${esc(row.role || '')}</td><td>${esc(row.kind || '')}</td><td>${evidenceButton(row, row.evidence_id || t('Open', '打开'), pointer, sourceValue)}</td><td>${esc(row.sha256 || '')}</td></tr>`).join('')}</tbody></table></div></div>`;
    };
    const panels = claims.map(claim => {
      const claimId = String(claim && claim.claim_id || '');
      const evidence = claim && claim.evidence && typeof claim.evidence === 'object' ? claim.evidence : {};
      const artifacts = Array.isArray(claim && claim.related_artifacts) ? claim.related_artifacts : [];
      return `<section class="gpi-claim-panel" id="gpi-claim-detail-${escAttr(claimId)}" data-gpi-claim-panel="${escAttr(claimId)}" hidden>
        <div class="gpi-claim-panel-head"><div><span>${esc(t('Bound number', '绑定数字'))}</span><strong>${esc(claim.display_value || '')}</strong></div><button type="button" data-gpi-claim-close aria-label="${escAttr(t('Close evidence detail', '关闭证据详情'))}">${esc(t('Close', '关闭'))}</button></div>
        ${artifactTable(t('Exact result source', '准确结果来源'), [t('Item', '项目'), t('Value', '值')], [
          [t('JSON field', 'JSON 字段'), claim.source_field || ''],
          [t('JSON pointer', 'JSON 指针'), claim.source_json_pointer || ''],
          [t('Source value', '源数值'), claim.source_value || ''],
          [t('Analysis step', '分析步骤'), claim.step_id || ''],
          [t('Evidence ID', '证据 ID'), evidence.evidence_id || ''],
          ['SHA-256', evidence.sha256 || ''],
        ])}
        ${lineageTable(evidence, artifacts, claim.source_json_pointer, claim.source_value)}
        <p class="gpi-claim-boundary">${esc(t('This view exposes immutable IDs and digests, not patient rows or host file paths. Scientific authority remains analysis-only until Host gates and human review permit more.', '此视图只显示不可变 ID 与摘要，不暴露患者行或主机文件路径。除非 Host 闸门与人工审阅另行许可，科学权限仍为 analysis-only。'))}</p>
      </section>`;
    }).join('');
    return `<div class="ag-artifact-readable ag-manuscript-reader">
      <div class="ag-artifact-readable-head"><div><div class="eyebrow">${esc(t('Evidence-bound article', '证据绑定文章'))}</div><div class="ag-artifact-readable-title">${esc(t('Click a highlighted number to open its exact result evidence preview. Full lineage remains available when needed.', '点击高亮数字，直接打开对应结果证据的可视化；需要时仍可查看完整证据链。'))}</div></div><span class="pill warn">analysis-only</span></div>
      <div class="gpi-manuscript-layout" data-gpi-manuscript-layout><article class="gpi-manuscript-article">${article || `<p>${esc(t('No reader blocks are available.', '没有可用的文章阅读内容。'))}</p>`}</article><aside class="gpi-claim-drawer" aria-live="polite"><div class="gpi-claim-empty" data-gpi-claim-empty>${esc(t('Claims without a previewable result source can still open their exact audit lineage here.', '没有可直接预览结果来源的论断，仍可在这里打开准确审计链路。'))}</div>${panels}</aside></div>
    </div>`;
  }
  function scientificFindingCopy(row) {
    const code = String(row && row.code || '');
    const known = {
      SCIENTIFIC_CAPABILITY_NOT_REPORTABLE: [
        'Upgrade the analysis method', '升级分析方法',
        'The current diagnostic-only capability will be replaced by a formally validated analysis capability.',
        '当前计划仅支持诊断性描述，EasyICU 会改用经过正式验证的分析能力。',
      ],
      DESIGN_ANALOGUE_NOT_ESTABLISHED: [
        'Find a comparable ICU study design', '补充可参照的 ICU 研究设计',
        'EasyICU will search for a clinically and methodologically comparable study and retain the source-backed screening decision.',
        'EasyICU 将继续检索临床主题与方法均可参照的研究，并保留有来源依据的筛选记录。',
      ],
      OUTCOME_DEFINITION_UNRESOLVED: [
        'Planner will define the primary outcome', 'Planner 将补全主要结局定义',
        'EasyICU must propose one clinically meaningful endpoint and observation horizon in the revised candidate plan.',
        'EasyICU 需要在修订版候选计划中提出临床含义明确的结局及观察时间范围。',
      ],
      ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED: [
        'Planner will propose sensitivity analyses', 'Planner 将提出敏感性分析',
        'EasyICU must propose study-appropriate executable checks in the revised candidate plan for one complete review.',
        'EasyICU 需要在修订版候选计划中提出适合本研究的可执行检查，供你一次性完整审阅。',
      ],
      FIGURE_ROLE_COVERAGE_INCOMPLETE: [
        'Add data-quality and distribution figures', '补齐数据质量与分布图',
        'The revised plan will include source-data-bound figures for the missing article roles.',
        '修订版计划会为缺失的文章角色补充与源数据绑定的图件。',
      ],
      FIGURE_CHART_TYPES_TOO_NARROW: [
        'Broaden the figure strategy', '扩展图表表达方式',
        'The revised plan will use complementary chart families instead of repeating one generic overview.',
        '修订版计划会采用互补的图表类型，而不是重复单一概览图。',
      ],
      NOVELTY_NOT_ESTABLISHED: [
        'Verify the novelty position', '核对研究创新性',
        'EasyICU will compare the proposed study with direct comparators before making any novelty claim.',
        'EasyICU 会先与直接可比研究进行来源可追溯的比较，再判断能否提出创新性主张。',
      ],
    };
    const copy = known[code];
    return {
      title: copy ? t(copy[0], copy[1]) : String(row && (row.message || row.code) || t('Unresolved review item', '待处理审阅项')),
      detail: copy ? t(copy[2], copy[3]) : String(row && (row.remediation || '') || ''),
    };
  }
  function scientificDecisionQuestion(row) {
    const code = String(row && row.code || '');
    const known = {
      OUTCOME_DEFINITION_UNRESOLVED: t(
        'Which available clinical endpoint and time horizon should this study use?',
        '这项研究应采用哪个临床结局，以及多长的观察时间范围？',
      ),
      ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED: t(
        'Which executable sensitivity analyses should be prespecified for this study?',
        '这项研究需要预先设定哪些可执行的敏感性分析？',
      ),
      POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED: t(
        'Should the revised study use a landmark or time-varying design, or remain descriptive?',
        '修订后的研究应采用 landmark／时变设计，还是仅保留描述性分析？',
      ),
      ADJUSTMENT_SET_NOT_USER_CONFIRMED: t(
        'Do you approve the proposed baseline adjustment set?',
        '你是否批准建议的基线调整变量？',
      ),
    };
    return known[code] || String(row && (row.authorization_question || row.message) || '');
  }
  function plannerOwnedScientificFinding(row) {
    return new Set([
      'OUTCOME_DEFINITION_UNRESOLVED',
      'ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED',
    ]).has(String(row && row.code || ''));
  }
  function scientificPlanReviewView(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const findings = Array.isArray(p.findings) ? p.findings : [];
    const decisions = findings.filter(row => row && !plannerOwnedScientificFinding(row) && (row.requires_user_authorization || row.remediation_route === 'study_authority_change'));
    const automatic = findings.filter(row => row && (row.remediation_route === 'agent_plan_revision' || plannerOwnedScientificFinding(row)));
    const evidence = findings.filter(row => row && (row.remediation_route === 'external_evidence' || row.remediation_route === 'independent_review'));
    const firstDecision = decisions[0] || null;
    const firstDecisionCopy = firstDecision ? scientificFindingCopy(firstDecision) : null;
    const laterDecisions = decisions.slice(1);
    const findingList = rows => rows.map(row => {
      const copy = scientificFindingCopy(row);
      return `<li><strong>${esc(copy.title)}</strong><span>${esc(copy.detail)}</span></li>`;
    }).join('');
    const bindingSteps = p.facts && p.facts.literature_design_bindings
      && Array.isArray(p.facts.literature_design_bindings.steps)
      ? p.facts.literature_design_bindings.steps : [];
    const citationMap = new Map();
    bindingSteps.forEach(step => {
      (Array.isArray(step.citations) ? step.citations : []).forEach(row => {
        const key = String(row.citation_key || row.title || '');
        if (key && !citationMap.has(key)) citationMap.set(key, row);
      });
    });
    const citations = Array.from(citationMap.values()).slice(0, 6);
    const citationCards = citations.map(row => `<article><strong>${esc(row.title || row.citation_key || '')}</strong><span>${esc(row.year || '')}</span><p>${esc(row.application || '')}</p></article>`).join('');
    const approvalAllowed = p.approval_allowed === true;
    const waiting = !approvalAllowed && decisions.length > 0;
    return `<div class="ag-artifact-readable ag-science-review">
      <header class="ag-science-review-hero">
        <div><div class="eyebrow">${esc(t('Plan review', '计划审阅'))}</div><h2>${esc(waiting
          ? t(`${decisions.length} decisions remain before analysis`, `计划还差 ${decisions.length} 个决定`)
          : approvalAllowed ? t('The plan is ready for approval', '计划已可进入批准') : t('EasyICU needs to revise this candidate plan', 'EasyICU 需要修订这份候选计划'))}</h2><p>${esc(waiting
          ? t('Answer one question at a time. EasyICU will handle the plan repair and evidence work.', '一次只回答一个问题。计划修订和补充证据由 EasyICU 处理。')
          : t('Endpoint, sensitivity design, plan structure, and evidence follow-up are system-owned proposal work. Review the revised complete plan instead of designing them here.', '结局定义、敏感性分析、计划结构和补证都属于系统的方案工作；你应审阅修订后的完整计划，不必在这里替系统设计。'))}</p></div>
        <span class="ag-science-review-state ${approvalAllowed ? 'is-ready' : 'is-waiting'}">${esc(approvalAllowed ? t('Ready', '可批准') : t('Analysis paused', '分析已暂停'))}</span>
      </header>
      ${firstDecision ? `<section class="ag-science-review-section is-current"><div class="ag-science-review-heading"><div><span>${esc(t('Do this now', '现在只做这一步'))}</span><strong>${esc(firstDecisionCopy.title)}</strong></div><em>1</em></div><div class="ag-science-current-question"><p>${esc(scientificDecisionQuestion(firstDecision))}</p><span>${esc(t('Use “Answer decision 1” in the conversation to reply.', '在左侧对话中点击「回答第 1 项」。'))}</span></div>${laterDecisions.length ? `<div class="ag-science-later"><span>${esc(t('Later', '稍后'))}</span><strong>${esc(scientificFindingCopy(laterDecisions[0]).title)}</strong><small>${esc(t('EasyICU will ask after the first answer is saved.', '第 1 项保存后，EasyICU 再询问这一项。'))}</small></div>` : ''}</section>` : ''}
      <details class="ag-science-review-details"><summary><span>${esc(t('EasyICU will handle', 'EasyICU 会自动处理'))}</span><strong>${esc(t(`${automatic.length + evidence.length} plan and evidence items`, `${automatic.length + evidence.length} 项计划修订与补证`))}</strong><em>${esc(t('No action needed now', '现在不需你处理'))}</em></summary><div class="ag-science-lanes">
        <article><div><strong>${esc(t('Plan revision', '计划修订'))}</strong><span>${esc(t(`${automatic.length} items`, `${automatic.length} 项`))}</span></div><ul>${findingList(automatic)}</ul></article>
        <article><div><strong>${esc(t('Evidence follow-up', '证据补充'))}</strong><span>${esc(t(`${evidence.length} items`, `${evidence.length} 项`))}</span></div><ul>${findingList(evidence)}</ul></article>
      </div></details>
      ${citationCards ? `<details class="ag-science-review-details"><summary><span>${esc(t('Methods references', '方法学依据'))}</span><strong>${esc(t(`${citations.length} references already used`, `已使用 ${citations.length} 篇方法学文献`))}</strong><em>${esc(t('Optional', '可选查看'))}</em></summary><div class="ag-science-citations">${citationCards}</div></details>` : ''}
      <p class="ag-science-review-audit">${esc(t('Raw scores, finding codes, and digest-bound details remain available in the JSON audit view.', '原始评分、问题代码和摘要绑定细节仍保留在 JSON 审计视图中。'))}</p>
    </div>`;
  }
  function agentPlanAnalysisLabel(value) {
    const labels = {
      data_quality_audit: t('Data-quality audit', '数据质量审计'),
      descriptive: t('Descriptive study', '描述性研究'),
      descriptive_epidemiology: t('Descriptive epidemiology', '描述性流行病学'),
      association: t('Association analysis', '关联分析'),
      regression: t('Regression analysis', '回归分析'),
      prediction: t('Prediction study', '预测研究'),
      survival: t('Time-to-event analysis', '生存／时间结局分析'),
    };
    return labels[String(value || '').toLowerCase()] || t('Study analysis', '研究分析');
  }
  function agentPlanOutputLabel(value) {
    const raw = String(value || '');
    const labels = {
      'artifact:analysis_cohort': t('Analysis cohort', '分析队列'),
      'table:cohort_flow': t('Cohort flow', '队列流程'),
      'table:baseline_table': t('Baseline summary', '基线特征汇总'),
      'table:descriptive_quality_summary': t('Descriptive summary', '描述性汇总'),
      'table:measurement_process_audit': t('Measurement audit', '测量过程审计'),
      'table:measurement_audit': t('Measurement availability', '变量可用性与缺失情况'),
      'table:measurement_missingness': t('Measurement completeness', '测量完整性'),
      'table:measurement_process': t('Measurement process', '测量过程'),
      'table:adjusted_association_estimates': t('Adjusted association estimates', '校正后关联估计'),
      'table:absolute_risk_context': t('Absolute-risk context', '绝对风险概览'),
      'statistic:primary_or': t('Primary odds ratio', '主要比值比'),
      'statistic:complete_case_n': t('Complete-case sample size', '完整病例数'),
      'table:robustness_summary': t('Robustness summary', '稳健性汇总'),
      'log:missingness_strategy_notes': t('Missing-data strategy', '缺失数据处理说明'),
      'table:robustness_matrix': t('Robustness matrix', '稳健性分析矩阵'),
      'figure:overview': t('Study overview figure', '研究概览图'),
      'figure:cohort_flow': t('Cohort flow figure', '队列流程图'),
      'figure:robustness_plot': t('Robustness figure', '稳健性分析图'),
      'figure:data_quality': t('Data-quality figure', '数据质量图'),
    };
    if (labels[raw]) return labels[raw];
    const humanized = raw.replace(/^(?:artifact|table|figure|statistic|log):/, '').replace(/[_-]+/g, ' ');
    const kind = /^figure:/i.test(raw) ? t('Figure', '图件')
      : /^table:/i.test(raw) ? t('Table', '结果表')
      : /^statistic:/i.test(raw) ? t('Statistic', '统计量')
      : /^log:/i.test(raw) ? t('Log', '日志')
      : /^artifact:/i.test(raw) ? t('Artifact', '中间产物') : '';
    return kind ? `${kind} · ${humanized}` : humanized;
  }
  function agentPlanVariableLabel(value, labels) {
    const raw = String(value || '');
    const canonical = {
      lact: t('Lactate', '乳酸'), death: t('In-hospital death', '院内死亡'),
      age: t('Age', '年龄'), sex: t('Sex', '性别'), adm: t('Admission type', '入院类型'),
      sofa: t('SOFA score', 'SOFA 评分'), los_icu: t('ICU length of stay', 'ICU 住院时长'),
      hr: t('Heart rate', '心率'), map: t('Mean arterial pressure', '平均动脉压'),
      ph: t('Blood pH', '血液酸碱度（pH）'), crea: t('Creatinine', '肌酐'),
      bili: t('Bilirubin', '胆红素'), plt: t('Platelet count', '血小板计数'),
      gcs: t('Glasgow Coma Scale', '格拉斯哥昏迷评分'),
      wbc: t('White blood cell count', '白细胞计数'),
    };
    return labels[raw] || canonical[raw.toLowerCase()] || raw.replace(/[_-]+/g, ' ');
  }
  /* Design-level fields carry the plan's own answers, so they must never be
     replaced by a canned sentence keyed on the field name - the retired copy
     asserted an ICU-stay time zero and a logistic primary method, which is
     false for a landmark or survival design. When the plan wrote in another
     language we prepend a gloss ONLY where it is derivable from structured
     payload (`analysis_type`) or from an unambiguous method token in the
     plan's own wording, and always keep that wording verbatim beside it.
     Fields with no derivable gloss (rationale, time zero, observation
     window) show the plan's wording alone rather than an invented summary. */
  function agentPlanDesignGloss(kind, raw, analysisType, analysisFamily) {
    const text = String(raw || '').toLowerCase();
    // `analysis_family` is compiled by research_agent/planning/study_design.py
    // and stamped by the artifact projection; prefer it over the raw type text.
    const type = `${String(analysisFamily || '')} ${String(analysisType || '')}`.toLowerCase();
    if (kind === 'primary_method') {
      const families = [];
      if (/cox|proportional[ _-]?hazard/.test(text)) families.push(t('a Cox proportional-hazards model', 'Cox 比例风险模型'));
      else if (/logistic/.test(text)) families.push(t('multivariable logistic regression', '多变量 Logistic 回归'));
      else if (/linear regression|ordinary least squares/.test(text)) families.push(t('linear regression', '线性回归'));
      else if (/poisson|negative binomial/.test(text)) families.push(t('count regression', '计数回归'));
      if (/restricted[ _-]?cubic[ _-]?spline|\brcs\b|spline/.test(text)) families.push(t('restricted cubic splines', '限制性立方样条'));
      if (/landmark/.test(text)) families.push(t('a landmark start', 'landmark 起点'));
      if (/propensity|matching|weighting|\biptw\b/.test(text)) families.push(t('propensity-score adjustment', '倾向评分调整'));
      if (/mixed[ _-]?effect|random[ _-]?effect|hierarchical/.test(text)) families.push(t('mixed effects', '混合效应'));
      if (!families.length && /descriptive|counts only|proportions?\b/.test(text)) families.push(t('descriptive statistics', '描述性统计'));
      if (!families.length) return '';
      return t(`Method named by the plan: ${families.join(' + ')}.`, `计划写明的方法：${families.join(' + ')}。`);
    }
    if (kind === 'estimand') {
      if (/surviv|time[ _-]?to[ _-]?event|hazard/.test(type)) return t('This design estimates an association on the time-to-event scale.', '这套设计要估计的是时间结局尺度上的关联。');
      if (/predict/.test(type)) return t('This design estimates how well the model discriminates and calibrates.', '这套设计要估计的是模型的区分度与校准。');
      if (/descriptive/.test(type)) return t('This design estimates distributions and proportions; it does not make between-group inferences.', '这套设计要估计的是分布与比例，不做组间推断。');
      if (/association|regression/.test(type)) return t('This design estimates the association between the exposure and the outcome, adjusted for the prespecified covariates.', '这套设计要估计的是暴露与结局之间、经预先指定协变量校正后的关联。');
      return '';
    }
    if (!/association|regression|descriptive|predict|surviv/.test(type)) return '';
    if (kind === 'supports') {
      return t('This design can report the direction, size, and uncertainty of the target quantity in the data at hand.', '这套设计能报告目标量在当前数据中的方向、大小和不确定性。');
    }
    if (kind === 'cannot_prove') {
      return t('Being observational, it cannot establish causation, and cannot rule out unmeasured confounding, selection bias, or measurement error.', '这是观察性设计，不能证明因果关系，也不能排除未测量混杂、选择偏倚或测量误差。');
    }
    return '';
  }
  function agentPlanFieldBody(kind, raw, analysisType, leadClass, analysisFamily) {
    const value = String(raw || '').trim();
    const open = leadClass ? `<p class="${leadClass}">` : '<p>';
    if (!value) return `${open}—</p>`;
    if (!agentPlanIntentNeedsTranslation(value)) return `${open}${esc(value)}</p>`;
    const gloss = agentPlanDesignGloss(kind, value, analysisType, analysisFamily);
    if (!gloss) return `${open}${esc(value)}</p>`;
    return `${open}${esc(gloss)}</p><p class="ag-plan-field-source"><small>${esc(t('Plan wording', '计划原文'))}</small>${esc(value)}</p>`;
  }
  function agentPlanStepIntent(step) {
    if (String(step && step.method || '') === 'visualization') {
      const outputs = Array.isArray(step && step.expected_outputs) ? step.expected_outputs : [];
      if (outputs.includes('figure:cohort_flow')) {
        return t(
          'Draw the cohort inclusion and exclusion flow from the registered cohort-flow table.',
          '根据已登记的队列流程表绘制纳入与排除流程图。',
        );
      }
      return t(
        'Create a source-data-bound figure from the planned results, with an auditable data table and vector export.',
        '根据计划产物生成与源数据绑定的图件，并保留可审计数据表和矢量版本。',
      );
    }
    const stated = String(step && (step.intent || step.title || step.name) || '').trim();
    return stated && !agentPlanIntentNeedsTranslation(stated) ? stated : agentPlanMethodIntent(step);
  }
  /* The plan's own wording is authoritative, but it arrives in whatever
     language the provider produced. When it does not match the reader's
     language we lead with a description of the METHOD - always true, and
     naming no variable, outcome, score, or database - and keep the plan's
     own wording underneath as `agentPlanStepStatedSource`. Never substitute
     a canned sentence that asserts what the study is about. */
  function agentPlanIntentNeedsTranslation(text) {
    const hasCjk = /[\u3400-\u9fff]/.test(String(text || ''));
    return window.EU_LANG === 'zh' ? !hasCjk : hasCjk;
  }
  function agentPlanStepStatedSource(step) {
    if (String(step && step.method || '') === 'visualization') return '';
    const stated = String(step && (step.intent || step.title || step.name) || '').trim();
    return stated && agentPlanIntentNeedsTranslation(stated) ? stated : '';
  }
  function agentPlanMethodIntent(step) {
    const blob = `${String(step && step.method || '')} ${String(step && step.step_id || '')}`.toLowerCase();
    if (/cohort|attrition|eligib/.test(blob)) {
      return t('Count the records that enter and leave the analysis, and fix the final denominator.', '统计纳入与排除的记录数，确定最终分析分母。');
    }
    if (/table_one|baseline/.test(blob)) {
      return t('Summarize the baseline characteristics of the study population.', '汇总研究人群的基线特征分布。');
    }
    if (/missing|measurement|quality|audit|applicab|profile/.test(blob)) {
      return t('Check measurement coverage and missingness for the variables this plan needs.', '检查本计划所需变量的测量覆盖与缺失情况。');
    }
    if (/sensitivit|robust/.test(blob)) {
      return t('Replay the prespecified sensitivity settings and check whether the main result holds.', '按预先规定的敏感性设定复核主要结论是否稳健。');
    }
    if (/absolute_risk/.test(blob)) {
      return t('Report the absolute risk of the outcome alongside the relative effect.', '在相对效应之外，补充结局的绝对风险背景。');
    }
    if (/surviv|hazard|time_to_event/.test(blob)) {
      return t('Estimate the time-to-event association using the prespecified adjustment set.', '按预先设定的调整变量估计时间结局关联。');
    }
    if (/predict|discriminat|calibrat/.test(blob)) {
      return t('Fit and evaluate the prediction model with the prespecified discrimination and calibration measures.', '按预先设定的区分度与校准指标拟合并评价预测模型。');
    }
    if (/descriptive|counts|distribution|prevalence|incidence|proportion/.test(blob)) {
      return t('Report the planned distributions and proportions with exact denominators.', '按精确分母报告计划中的分布与比例。');
    }
    if (/associat|regress|model|effect|estimand|contrast|spline|landmark/.test(blob)) {
      return t('Estimate the prespecified adjusted association and report its magnitude and uncertainty.', '按预先设定的调整变量估计校正后关联，并报告效应大小与不确定性。');
    }
    return t('Complete this planned analysis step.', '完成这一项计划分析。');
  }
  /* ---- plan flow map ----------------------------------------------
     A generated plan is a flat list of typed steps (often 8-12). Read as a
     numbered list it is a wall of prose. These helpers fold the steps into
     the few research stages every study shares, so the reader sees the
     shape of the run first and the per-step prose only on demand.
     Classification is method-driven and case-neutral: no benchmark case,
     variable, score, or database is named here. ------------------------ */
  const AGENT_PLAN_STAGES = [
    {
      key: 'population',
      label: () => t('Build the study population', '建立研究人群'),
      hint: () => t('Who enters the analysis, and what the denominator is', '谁进入分析、分母是多少'),
    },
    {
      key: 'quality',
      label: () => t('Check the data', '核查数据'),
      hint: () => t('Baseline picture, measurement coverage, and missingness', '基线画像、测量覆盖与缺失'),
    },
    {
      key: 'primary',
      label: () => t('Run the main analysis', '做主分析'),
      hint: () => t('The estimate that answers the research question', '回答研究问题的主要估计'),
    },
    {
      key: 'robustness',
      label: () => t('Stress-test the result', '检验稳健性'),
      hint: () => t('Replay the prespecified sensitivity settings', '按预先规定的设定复核结论'),
    },
    {
      key: 'figure',
      label: () => t('Draw the figures', '生成图件'),
      hint: () => t('Source-data-bound figures with auditable tables', '与源数据绑定、可审计的图件'),
    },
    {
      key: 'support',
      label: () => t('Supporting steps', '支持性步骤'),
      hint: () => t('The plan declares these auxiliary: they carry no result claim', '计划声明为辅助步骤，不承载结论'),
    },
  ];
  /* What SHAPE of work a step is, read from its method. Used for the step's
     own title. This is a heuristic and must never decide whether a step is a
     result - see agentPlanStepStage. */
  function agentPlanStepMethodKind(step) {
    const outputs = (Array.isArray(step && step.expected_outputs) ? step.expected_outputs : [])
      .map(value => String(value || '').toLowerCase());
    const blob = `${String(step && step.method || '')} ${String(step && step.step_id || '')}`.toLowerCase();
    if (/visuali|render|figure|plot|chart/.test(blob)) return 'figure';
    if (outputs.length && outputs.every(value => value.startsWith('figure:'))) return 'figure';
    if (/sensitivit|robust/.test(blob)) return 'robustness';
    if (/cohort|attrition|eligib/.test(blob)) return 'population';
    // audit words win over estimate words, so `descriptive_quality_summary`
    // is a data check while `descriptive counts` is the study's own result
    if (/table_one|baseline|missing|measurement|quality|audit|applicab|profile|readiness|coverage/.test(blob)) return 'quality';
    return 'primary';
  }
  /* WHERE the step sits in the run. This is a study-semantics question, so it
     is answered by the layer that owns plan semantics, not here:
     research_agent/planning/step_phase.py compiles `planned_phase` from the
     Planner-declared role plus the plan's own method contracts, and the
     Copilot artifact projection stamps it onto the preview payload. Reading
     the compiled value is the whole point - re-deriving study semantics from
     free-text method names is what produced this reader's canned-prose bugs.
     The phase is a projection, never persisted: plan_sha256 covers the whole
     plan dump, so a new schema field would invalidate the stored digest of
     every run already on disk.
     A plan payload with no compiled phase (a demo fixture, a hand-built
     preview, an artifact served by an older host) falls back to the local
     heuristic below, which still refuses to promote a declared-auxiliary step
     into a result. */
  const AGENT_PLAN_PHASE_STAGE = {
    cohort: 'population',
    data_check: 'quality',
    analysis: 'primary',
    robustness: 'robustness',
    reporting: 'figure',
    support: 'support',
  };
  function agentPlanStepStage(step) {
    const compiled = AGENT_PLAN_PHASE_STAGE[String(step && step.planned_phase || '').toLowerCase()];
    if (compiled) return compiled;
    const role = String(step && step.planned_analysis_role || '').toLowerCase();
    if (role === 'primary' || role === 'secondary') return 'primary';
    if (role === 'sensitivity') return 'robustness';
    const kind = agentPlanStepMethodKind(step);
    if (role === 'auxiliary' && (kind === 'primary' || kind === 'robustness')) return 'support';
    return kind;
  }
  function agentPlanStepTitle(step) {
    const blob = `${String(step && step.method || '')} ${String(step && step.step_id || '')}`.toLowerCase();
    const stage = agentPlanStepMethodKind(step);
    if (stage === 'figure') {
      const figure = (Array.isArray(step && step.expected_outputs) ? step.expected_outputs : [])
        .find(value => /^figure:/i.test(String(value || '')));
      if (figure) return agentPlanOutputLabel(figure);
      if (/robust|sensitivit/.test(blob)) return t('Robustness figure', '稳健性分析图');
      if (/quality|missing|measurement/.test(blob)) return t('Data-quality figure', '数据质量图');
      if (/cohort|attrition/.test(blob)) return t('Cohort flow figure', '队列流程图');
      return t('Result figure', '结果图件');
    }
    if (stage === 'robustness') {
      if (/spline|functional_form/.test(blob)) return t('Sensitivity · exposure functional form', '敏感性分析 · 暴露形式设定');
      if (/missing|complete_case|imputation/.test(blob)) return t('Sensitivity · missing-data handling', '敏感性分析 · 缺失处理');
      if (/landmark|immortal|time/.test(blob)) return t('Sensitivity · time definition', '敏感性分析 · 时间定义');
      return t('Robustness replay', '稳健性复核');
    }
    if (stage === 'population') return t('Cohort and denominator accounting', '队列与分母账本');
    if (stage === 'quality') {
      if (/table_one|baseline/.test(blob)) return t('Baseline characteristics table', '基线特征表');
      if (/missing|measurement|quality|audit/.test(blob)) return t('Measurement coverage and missingness audit', '测量覆盖与缺失审计');
      return t('Data readiness check', '数据可用性核查');
    }
    if (/absolute_risk/.test(blob)) return t('Absolute-risk context', '绝对风险背景');
    if (/landmark/.test(blob)) return t('Primary association at the landmark time', '主关联分析（landmark 起点）');
    if (/spline/.test(blob)) return t('Primary association, spline-based exposure', '主关联分析（样条暴露形式）');
    if (/surviv|time_to_event|hazard/.test(blob)) return t('Time-to-event analysis', '时间结局分析');
    if (/predict|discriminat|calibrat/.test(blob)) return t('Prediction model', '预测模型');
    if (/descriptive|counts|distribution|prevalence/.test(blob)) return t('Descriptive distribution', '描述性分布');
    if (/associat|regress|model|effect|estimand|contrast/.test(blob)) return t('Primary adjusted association', '主要校正后关联');
    return t('Planned analysis step', '计划分析步骤');
  }
  function agentPlanFlowStages(steps) {
    return AGENT_PLAN_STAGES.filter(stage => steps.some(step => agentPlanStepStage(step) === stage.key));
  }
  function agentPlanFlowMap(steps) {
    const stages = agentPlanFlowStages(steps);
    if (!stages.length) return '';
    return `<ol class="ag-plan-flow">${stages.map((stage, position) => {
      const rows = steps
        .map((step, index) => ({ step, index }))
        .filter(row => agentPlanStepStage(row.step) === stage.key);
      return `<li class="ag-plan-flow-stage is-${stage.key}">
        <div class="ag-plan-flow-head"><span class="ag-plan-flow-mark">${position + 1}</span><div><strong>${esc(stage.label())}</strong><small>${esc(stage.hint())}</small></div></div>
        <ul class="ag-plan-flow-steps">${rows.map(row => `<li><b>${row.index + 1}</b><span>${esc(agentPlanStepTitle(row.step))}</span></li>`).join('')}</ul>
      </li>`;
    }).join('')}</ol>`;
  }
  function agentPlanGlance(steps, stageCount, citationCount) {
    const outputs = new Set();
    steps.forEach(step => (Array.isArray(step && step.expected_outputs) ? step.expected_outputs : [])
      .forEach(value => outputs.add(String(value || ''))));
    const list = Array.from(outputs);
    const tables = list.filter(value => /^table:/i.test(value)).length;
    const figures = list.filter(value => /^figure:/i.test(value)).length;
    const chips = [
      [steps.length, t('planned steps', '个计划步骤')],
      [stageCount, t('stages', '个阶段')],
      tables ? [tables, t('result tables', '张结果表')] : null,
      figures ? [figures, t('figures', '张图件')] : null,
      citationCount ? [citationCount, t('bound sources', '篇计划依据')] : null,
    ].filter(Boolean);
    if (!chips.length) return '';
    return `<div class="ag-plan-glance">${chips
      .map(([value, label]) => `<span><b>${esc(String(value))}</b>${esc(label)}</span>`).join('')}</div>`;
  }
  function agentPlanView(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const labels = p.display_labels && typeof p.display_labels === 'object' ? p.display_labels : {};
    const designSelection = p.design_selection && typeof p.design_selection === 'object' ? p.design_selection : {};
    const candidates = Array.isArray(designSelection.candidates) ? designSelection.candidates : [];
    const selected = candidates.find(row => row && row.disposition === 'selected') || candidates[0] || {};
    const recommendation = Array.isArray(selected.reviewable_plan) && selected.reviewable_plan.length === 6
      ? selected.reviewable_plan : null;
    const steps = Array.isArray(p.steps) ? p.steps : [];
    const robustness = Array.isArray(p.robustness_specs) ? p.robustness_specs : [];
    const endpoint = p.endpoint && typeof p.endpoint === 'object' ? p.endpoint : null;
    const required = Array.isArray(selected.required_variables) ? selected.required_variables : [];
    const visibleVariables = required.filter(value => !/(?:^|_)id$/i.test(String(value || ''))).slice(0, 10);
    const citations = Array.from(new Set([
      ...(Array.isArray(selected.literature_citation_keys) ? selected.literature_citation_keys : []),
      ...steps.flatMap(step => Array.isArray(step && step.literature_citation_keys) ? step.literature_citation_keys : []),
    ].filter(Boolean)));
    const gaps = [];
    if (!recommendation) gaps.push(t('The Planner has not yet produced a complete reviewable recommendation.', 'Planner 尚未给出完整、可审阅的推荐方案。'));
    if (!endpoint) gaps.push(t('The primary outcome still lacks an executable definition and observation horizon.', '主要结局尚缺可执行定义和观察时间范围。'));
    if (!robustness.length) gaps.push(t('The sensitivity-analysis proposal has not yet been formed.', '敏感性分析方案尚未形成。'));
    if (String(p.analysis_type || '') === 'data_quality_audit') gaps.push(t('This version answers data readiness only; it does not yet answer the stated association question.', '这一版只能回答数据是否可用，还不能回答原研究问题中的关联。'));
    const planReady = gaps.length === 0;
    const flowStages = agentPlanFlowStages(steps);
    const stepCards = steps.map((step, index) => {
      const outputs = Array.isArray(step && step.expected_outputs) ? step.expected_outputs : [];
      const shown = outputs.slice(0, 4);
      const hidden = outputs.length - shown.length;
      const note = agentPlanStepIntent(step);
      const title = agentPlanStepTitle(step);
      const source = agentPlanStepStatedSource(step);
      return `<li><span>${index + 1}</span><div><strong>${esc(title)}</strong>${note && note !== title ? `<p>${esc(note)}</p>` : ''}${source ? `<p class="ag-plan-step-source"><small>${esc(t('Plan wording', '计划原文'))}</small>${esc(source)}</p>` : ''}${outputs.length ? `<div class="ag-plan-step-outputs"><small>${esc(t('Planned output', '计划产物'))}</small>${shown.map(value => `<span>${esc(agentPlanOutputLabel(value))}</span>`).join('')}${hidden > 0 ? `<span class="is-more">+${hidden}</span>` : ''}</div>` : ''}</div></li>`;
    }).join('');
    const variableChips = visibleVariables.map(value => `<span>${esc(agentPlanVariableLabel(value, labels))}</span>`).join('');
    const literatureSummary = citations.length
      ? `<span>${esc(t(`${citations.length} bound sources`, `已绑定 ${citations.length} 篇计划依据`))}</span>`
      : '';
    const designType = String(selected.analysis_type || p.analysis_type || '');
    const designFamily = String(selected.analysis_family || '');
    const planField = (kind, value, leadClass) => agentPlanFieldBody(kind, value, designType, leadClass, designFamily);
    const recommendationCards = recommendation ? [
      [t('Population and analysis unit', '研究人群与分析单位'), recommendation[0]],
      [t('Exposure definition and timing', '暴露定义、时间窗与汇总方式'), recommendation[1]],
      [t('Outcome and follow-up', '结局定义与随访范围'), recommendation[2]],
      [t('Adjustment and model', '调整变量与主要模型'), recommendation[3]],
      [t('Missing-data strategy', '缺失数据处理'), recommendation[4]],
      [t('Sensitivity and feasibility checks', '敏感性分析与数据可行性检查'), recommendation[5]],
    ].map(([label, value]) => `<article><small>${esc(label)}</small><p>${esc(String(value || '').trim() || '—')}</p></article>`).join('') : '';
    return `<div class="ag-artifact-readable ag-plan-reader">
      <header class="ag-plan-hero"><div><div class="eyebrow">${esc(t('Candidate research plan', '候选研究计划'))}</div><h2>${esc(p.research_question || t('Research question not recorded', '尚未记录研究问题'))}</h2><p>${esc(t('This is a proposal for review. No patient-data analysis has started.', '这是供审阅的候选方案，尚未开始患者数据分析。'))}</p></div><span class="ag-plan-state ${planReady ? 'is-ready' : 'is-revision'}">${esc(planReady ? t('Ready to review', '待你审阅') : t('Needs EasyICU revision', '待 EasyICU 修订'))}</span></header>
      ${agentPlanGlance(steps, flowStages.length, citations.length)}
      ${gaps.length ? `<section class="ag-plan-section is-gap"><div class="ag-plan-section-head"><span>!</span><div><small>${esc(t('EasyICU must revise', 'EasyICU 需要修订'))}</small><h3>${esc(t('Why this version is not ready for approval', '为什么这一版还不能批准'))}</h3></div></div><ul>${gaps.map(value => `<li>${esc(value)}</li>`).join('')}</ul><p>${esc(t('These are Planner responsibilities. The researcher reviews the revised complete plan instead of filling these implementation details one by one.', '这些属于 Planner 的职责。研究者应审阅修订后的完整计划，而不是逐项替系统填写实现细节。'))}</p></section>` : ''}
      <section class="ag-plan-section"><div class="ag-plan-section-head"><span>01</span><div><small>${esc(t('Chosen design · plan at a glance', '设计选择 · 先看核心设定'))}</small><h3>${esc(agentPlanAnalysisLabel(selected.analysis_type || p.analysis_type))}</h3></div></div><p class="ag-plan-lead">${esc(t('Start with the target quantity, study start, follow-up, and primary method. The full rationale remains available below.', '先看要估计什么、研究从哪里开始、随访到哪里以及主要方法；完整设计理由保留在下方。'))}</p><div class="ag-plan-design-grid"><article><small>${esc(t('Target quantity', '要估计什么'))}</small>${planField('estimand', selected.estimand)}</article><article><small>${esc(t('Study start', '研究起点'))}</small>${planField('time_zero', selected.time_zero)}</article><article><small>${esc(t('Observation window', '观察范围'))}</small>${planField('observation_window', selected.observation_window)}</article><article><small>${esc(t('Primary method', '主要方法'))}</small>${planField('primary_method', selected.primary_method)}</article></div><div class="ag-plan-boundaries"><article><strong>${esc(t('What this design can answer', '这套设计能回答'))}</strong>${planField('supports', selected.supports)}</article><article><strong>${esc(t('What it cannot prove', '这套设计不能证明'))}</strong>${planField('cannot_prove', selected.cannot_prove)}</article></div></section>
      <section class="ag-plan-section"><div class="ag-plan-section-head"><span>02</span><div><small>${esc(t('Analysis path · workflow', '分析路径 · 分析流程'))}</small><h3>${esc(t(`${steps.length} planned steps in ${flowStages.length} stages`, `共 ${steps.length} 个步骤 · ${flowStages.length} 个阶段`))}</h3></div></div><p class="ag-plan-lead">${esc(t('Read the map first: each stage says what the run finishes before it moves on. Open the detail list only when you need the exact wording of a step.', '先看流程图：每个阶段说明这一段要做完什么，再进入下一段；需要逐条核对时再展开详细说明。'))}</p>${agentPlanFlowMap(steps)}${stepCards ? `<details class="ag-plan-step-detail"><summary>${esc(t(`Step-by-step detail · ${steps.length} steps`, `逐步说明 · 共 ${steps.length} 步`))}</summary><ol class="ag-plan-steps">${stepCards}</ol></details>` : `<ol class="ag-plan-steps"><li><span>—</span><div><strong>${esc(t('No analysis steps are present.', '尚未形成分析步骤。'))}</strong></div></li></ol>`}</section>
      <section class="ag-plan-section"><div class="ag-plan-section-head"><span>03</span><div><small>${esc(t('Study ingredients', '研究要素'))}</small><h3>${esc(t('Variables named in the candidate plan', '候选计划涉及的变量'))}</h3></div></div><div class="ag-plan-chips">${variableChips || `<span>${esc(t('Not yet specified', '尚未明确'))}</span>`}</div>${endpoint ? `<p class="ag-plan-note"><strong>${esc(t('Primary outcome', '主要结局'))}：</strong>${esc(agentPlanVariableLabel(endpoint.name, labels))}</p>` : ''}</section>
      ${recommendation ? `<details class="ag-plan-recommendations"><summary><span>${esc(t('Planner recommendation for review · 6 exact settings', 'Planner 推荐方案（待审阅）· 6 项具体设定'))}</span><small>${esc(t('Open when you need to inspect or change the exact definitions.', '需要逐项核对或修改时再展开。'))}</small></summary><p class="ag-plan-lead">${esc(t('EasyICU proposes these choices first; modify or approve them after review. They are not yet treated as researcher-confirmed.', '先给方案，再由你修改或批准。以下内容由 EasyICU 先行推荐，尚未视为研究者确认。'))}</p><div class="ag-plan-design-grid">${recommendationCards}</div></details>` : ''}
      <details class="ag-plan-details"><summary>${esc(t('Why this design was chosen', '查看完整设计理由'))}</summary>${planField('decision_reason', p.rationale || selected.decision_reason)}</details>
      ${literatureSummary ? `<section class="ag-plan-literature"><div><strong>${esc(t('Literature used by this plan', '本计划使用的文献依据'))}</strong><small>${esc(t('Open “Literature evidence” for source, screening, and exact step bindings.', '具体来源、筛选理由及步骤绑定请打开「文献依据」。'))}</small></div><div>${literatureSummary}</div></section>` : ''}
      <p class="ag-plan-audit">${esc(t('Internal ids, methods, inputs, outputs, and the full immutable payload remain in the JSON audit view.', '内部标识、方法、输入输出及完整不可变内容仍保留在 JSON 审计视图中。'))}</p>
    </div>`;
  }
  function artifactStructuredView(name, payload) {
    const n = String(name || '').toLowerCase();
    const p = payload && typeof payload === 'object' ? payload : {};
    const gate = p.gate && typeof p.gate === 'object' ? p.gate : p;
    if (String(p.schema_version || '') === 'easyicu.manuscript-provenance/1') {
      return manuscriptProvenanceView(p);
    }
    if (n === 'agent_plan.json') return agentPlanView(p);
    if (n === 'scientific_plan_review.json') return scientificPlanReviewView(p);
    const sections = [];
    const summary = artifactSummaryRows(
      p,
      ['run_id', 'study_id', 'run_type', 'status', 'mode', 'database_scope', 'cohort_size', 'evidence_count', 'missing_evidence', 'local_first']
    );
    if (summary.length) {
      sections.push(artifactTable(t('Readable artifact summary', '可读产物摘要'), [t('Field', '字段'), t('Value', '值')], summary));
    }
    if (String(p.schema_version || '') === 'easyicu.data-package-review/1') {
      const denominator = p.denominator && typeof p.denominator === 'object' ? p.denominator : {};
      sections.push(artifactTable(
        t('Data package checkpoint', '数据包检查点'),
        [t('Item', '项目'), t('Value', '值')],
        [
          [t('Review status', '审阅状态'), p.status || ''],
          [t('Analysis unit', '分析单位'), denominator.analysis_unit || ''],
          [t('Aggregate denominator', '聚合分母'), denominator.count == null ? '' : Number(denominator.count).toLocaleString()],
          [t('Review digest', '审阅摘要'), p.review_sha256 || ''],
        ]
      ));
      const concepts = Array.isArray(p.concepts) ? p.concepts : [];
      sections.push(artifactTable(
        t('Configured concept availability', '已配置概念可用性'),
        [t('Study role', '研究角色'), t('Concept', '概念'), t('Status', '状态'), t('Evaluable / denominator', '可评估 / 分母'), t('Missingness semantics', '缺失语义')],
        concepts.map(row => [
          row.study_role || '', row.concept_id || '', row.availability_status || '',
          row.evaluable_count == null ? '' : `${Number(row.evaluable_count).toLocaleString()} / ${Number(row.denominator_count || 0).toLocaleString()}`,
          row.interpretation || row.reason_code || '',
        ]),
        t('No execution concepts were configured.', '尚未配置执行概念。')
      ));
    }
    if (String(p.schema_version || '') === 'easyicu.system-validation-report/1') {
      sections.push(artifactTable(
        t('Engineering validation boundary', '工程验证边界'),
        [t('Item', '项目'), t('Value', '值')],
        [
          [t('Status', '状态'), p.status || ''],
          [t('Authority', '权限'), p.authority_class || ''],
          [t('Claim ceiling', '结论上限'), p.claim_ceiling || ''],
          [t('Publication authorized', '发表授权'), p.publication_authorized ? t('yes', '是') : t('no', '否')],
        ]
      ));
      sections.push(artifactTable(
        t('Measured run facts', '运行实测事实'),
        [t('Metric', '指标'), t('Value', '值'), t('Interpretation', '解释')],
        (Array.isArray(p.metrics) ? p.metrics : []).map(row => [row.label || '', row.value || '', row.detail || ''])
      ));
      sections.push(artifactTable(
        t('Authority-aware lifecycle', '权限感知生命周期'),
        [t('Stage', '阶段'), t('Status', '状态'), t('Meaning', '含义')],
        (Array.isArray(p.lifecycle) ? p.lifecycle : []).map(row => [row.label || row.stage || '', row.status || '', row.summary || ''])
      ));
      sections.push(artifactTable(
        t('Scientific blockers retained', '保留的科学阻断项'),
        [t('Severity', '级别'), t('Finding', '问题'), t('Why it matters', '影响')],
        (Array.isArray(p.scientific_findings) ? p.scientific_findings : []).slice(0, 12).map(row => [row.severity || '', row.code || '', row.message || ''])
      ));
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
    if (n.includes('result_tables')) {
      const tables = Array.isArray(p.tables) ? p.tables.slice(0, 8) : [];
      tables.forEach((table, index) => {
        const headers = Array.isArray(table.headers) ? table.headers.slice(0, 12) : [];
        const rows = Array.isArray(table.rows) ? table.rows.slice(0, 30) : [];
        sections.push(artifactTable(
          table.label || `${t('Result table', '结果表')} ${index + 1}`,
          headers,
          rows,
          t('This evidence table has no previewable aggregate rows.', '这个证据表没有可预览的聚合行。')
        ));
      });
      if (!tables.length) {
        sections.push(artifactTable(
          t('Research result tables', '科研结果表'),
          [t('Status', '状态')],
          [],
          t('No aggregate result tables passed the bounded preview policy.', '没有聚合结果表通过有界预览策略。')
        ));
      }
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
    if ((n.includes('workflow') || n.includes('plan')) && n !== 'scientific_plan_review.json') {
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
        ${artifactContentsStrip(sections)}
        ${figureGallery(p)}
        ${sections.join('')}
      </div>`;
  }

  window.AGENT_RENDER = {
    DEMO_STUDIES,
    runStatusLabel, runStatusHint, gateCheckLabel, readableArtifactText, firstValue, fmtCount,
    artifactKind, artifactTitle, artifactCategory, artifactSummary, artifactRank, defaultArtifactName,
    thumb, scrubDataUrls, figureGallery, artifactScalar, artifactKeyLabel,
    artifactSummaryRows, artifactTable, objectArrayRows, firstObjectArray, stepRowsFrom, manuscriptProvenanceView, artifactStructuredView,
  };
})();
