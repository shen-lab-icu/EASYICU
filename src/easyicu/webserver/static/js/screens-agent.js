/* Screen: Research Agent — project workspace (redesigned, bilingual).
   Goal: make the WORKFLOW and PROJECT MANAGEMENT obvious.
     • Left rail = a persistent list of studies (projects), like chat sessions.
     • Each study carries a linked cohort, its own run history, outputs, and
       draft versions. Idea mining lives in the separate #ideas workspace.
     • The pipeline + evidence gate are drawn explicitly per study.
   Outputs fail closed: Real mode lists only whitelisted local artifacts. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  /* ---------------- project data ---------------- */
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
        ['run 04', ['Full plan (gated draft)', '完整计划(草稿受闸)'], 'blocked', '2m 41s', ['2 days ago', '2 天前']],
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
      id: 'lactate', name: ['Early lactate analysis seed', '早期乳酸分析种子'],
      mode: 'analysis', status: 'idle', stage: 0,
      cohort: 'sepsis_mortality_demo', source: ['idea seed · not yet run', '想法种子 · 尚未运行'],
      question: [
        'Test whether early lactate trajectory adds prognostic information after the idea has been confirmed in Idea Mining.',
        '在 Idea 挖掘确认后,检验早期乳酸轨迹是否提供额外预后信息。',
      ],
      runs: [
        ['seed 02', ['Feasibility handoff', '可行性交接'], 'complete', '0:41', ['today 13:05', '今天 13:05']],
        ['seed 01', ['Source check', '来源核查'], 'complete', '0:19', ['today 12:40', '今天 12:40']],
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

  let agSel = null;
  let agTab = 'overview';
  let agEvOpen = -1;   // expanded evidence-gate check index
  let agRun = { active: false, prog: 0, timer: null, es: null, jobId: null, step: null, error: null, result: null };
  let agReview = { projectDir: null, loading: false, error: null, data: null, signing: false };
  let agHistory = { studyId: null, loading: false, error: null, data: null };
  let agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
  let agProvider = { provider: 'openai', consent: false, loading: false, error: null, status: null };
  let agIdeaProjects = { loading: false, error: null, data: null };
  const AG_JOB_KEY = 'easyicu.agent.activeJob.v1';
  let agResumeProbe = { loading: false, checkedJobId: null };

  /* continuity: Copilot can land a completed run */
  window.__euAgentPreset = function () { agSel = 'sepsis'; agTab = 'outputs'; };

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  }
  function seedStudy(row) {
    const title = row.title || row.question || 'Idea-derived study';
    const q = row.question || title;
    const pre = row.pre_experiment_summary || {};
    const source = row.source || {};
    return {
      id: row.study_id || row.id || title,
      name: [title, title],
      mode: 'analysis',
      status: row.status === 'seeded_from_idea' ? 'idle' : (row.status || 'idle'),
      stage: Number(row.stage || 0),
      cohort: row.study_id || 'idea_seed',
      source: [
        `idea seed · ${pre.status || 'pre-experiment'} · ${pre.feature_count || 0} features`,
        `idea 种子 · ${pre.status || '预实验'} · ${pre.feature_count || 0} 特征`,
      ],
      question: [q, q],
      runs: (row.runs || []).map((r, i) => [
        r.label || ('seed ' + String(i + 1).padStart(2, '0')),
        [r.scope || 'metadata seed', r.scope || '元数据种子'],
        r.status || 'complete',
        '—',
        [r.created_at || 'local', r.created_at || '本地'],
      ]),
      signed: false,
      ideaSeed: row,
      sourceArticle: [source.title, source.journal, source.year].filter(Boolean).join(' · '),
    };
  }
  function realMode() {
    return window.EU_DATA === 'real';
  }
  function emptyStudy() {
    return {
      id: '__none__',
      name: [t('No local research project selected', '尚未选择本地研究项目'), t('No local research project selected', '尚未选择本地研究项目')],
      mode: 'analysis',
      status: 'idle',
      stage: 0,
      cohort: '—',
      source: [t('No local project seed', '无本地项目种子'), t('No local project seed', '无本地项目种子')],
      question: [
        t('Create a project from Idea Mining or start a local Agent run from an active export. Demo studies are hidden in Real mode.', '请从 Idea Mining 创建项目，或基于 active export 启动本地 Agent run。真实模式下不会显示演示研究。'),
        t('Create a project from Idea Mining or start a local Agent run from an active export. Demo studies are hidden in Real mode.', '请从 Idea Mining 创建项目，或基于 active export 启动本地 Agent run。真实模式下不会显示演示研究。'),
      ],
      runs: [],
      signed: false,
      empty: true,
    };
  }
  function allStudies() {
    const seeds = (agIdeaProjects.data && Array.isArray(agIdeaProjects.data.projects) ? agIdeaProjects.data.projects : []).map(seedStudy);
    const seedIds = new Set(seeds.map(s => s.id));
    const base = realMode() ? [] : DEMO_STUDIES;
    return seeds.concat(base.filter(s => !seedIds.has(s.id)));
  }
  function study() { return allStudies().find(s => s.id === agSel) || allStudies()[0] || emptyStudy(); }
  function displayPath(path) {
    const raw = String(path || '');
    const home = String(window.EU_HOME || '');
    return home && raw.startsWith(home) ? raw.replace(home, '~') : raw;
  }
  function projectFolderLabel(s) {
    if (s && s.empty) return t('No local project folder yet', '还没有本地项目文件夹');
    return s && s.ideaSeed && s.ideaSeed.project_dir
      ? displayPath(s.ideaSeed.project_dir)
      : t('Created when this study is first run', '首次运行时创建');
  }
  function requestIdeaAgentProjects(force) {
    if (!window.EU_API || !window.EU_API.loadIdeaAgentProjects) return;
    if (!force && (agIdeaProjects.loading || agIdeaProjects.data || agIdeaProjects.error)) return;
    agIdeaProjects = { loading: true, error: null, data: null };
    window.EU_API.loadIdeaAgentProjects({ limit: 50 }).then(data => {
      agIdeaProjects = { loading: false, error: null, data: data };
      const studies = allStudies();
      let preferred = null;
      try {
        const raw = localStorage.getItem('easyicu_last_idea_agent_project');
        const parsed = raw ? JSON.parse(raw) : null;
        preferred = parsed && parsed.study_id;
      } catch (_) {}
      const before = agSel;
      if (preferred && studies.some(row => row.id === preferred)) agSel = preferred;
      else if (!studies.some(row => row.id === agSel) && studies.length) agSel = studies[0].id;
      maybeRestoreAgentJob();
      if (agSel !== before && window.__euRender) window.__euRender();
      else repaintBody();
    }).catch(err => {
      agIdeaProjects = { loading: false, error: err.message || String(err), data: null };
      repaintBody();
    });
  }
  function activeExportSource() {
    if (window.EU_DATA !== 'real') return null;
    if (window.EU_SOURCES && window.EU_SOURCES.activeSource) return window.EU_SOURCES.activeSource();
    const reg = window.EU_WORKSPACE_REGISTRY || {};
    return (reg.sources || []).find(s => s.path === reg.active_path) || null;
  }
  function activeSourceLabel() {
    const src = activeExportSource();
    if (!src) return null;
    const sum = src.summary || {};
    const parts = [];
    if (sum.stays != null) parts.push(Number(sum.stays).toLocaleString() + ' stays');
    if (sum.modules != null) parts.push(Number(sum.modules).toLocaleString() + ' modules');
    return `${src.label || src.database || 'local'}${parts.length ? ' · ' + parts.join(' · ') : ''}`;
  }
  function liveRunForStudy() {
    const s = study();
    return window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.study_id === s.id ? window.EU_AGENT_LAST_RUN : null;
  }
  function artifactsForLive(live) {
    if (!live) return [];
    const review = currentLiveReview(live);
    if (review && Array.isArray(review.artifacts)) return review.artifacts;
    return Array.isArray(live.artifacts) ? live.artifacts : [];
  }
  function outputCountForStudy() {
    const live = liveRunForStudy();
    if (live) return artifactsForLive(live).length;
    return 0;
  }
  function artifactKind(name) {
    const n = String(name || '').toLowerCase();
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
      'quality_gate.json': t('Evidence gate', '证据闸'),
      'evidence_ledger.json': t('Evidence ledger', '证据账本'),
      'agent_plan.json': t('Agent plan', 'Agent 计划'),
      'manuscript_draft.json': t('Locked manuscript draft', '锁定论文草稿'),
      'human_signoff.json': t('Human sign-off', '人工签署'),
    };
    if (labels[n]) return labels[n];
    return n.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
  function requestLiveReview(live) {
    if (!live || !live.project_dir || !window.EU_API || !window.EU_API.loadAgentRunReview) return;
    if (agReview.projectDir === live.project_dir && (agReview.loading || agReview.data || agReview.error)) return;
    agReview = { projectDir: live.project_dir, loading: true, error: null, data: null, signing: false };
    window.EU_API.loadAgentRunReview(live.project_dir).then(data => {
      agReview = { projectDir: live.project_dir, loading: false, error: null, data: data, signing: false };
      window.EU_AGENT_RUN_REVIEW = data;
      repaintBody();
    }).catch(err => {
      agReview = { projectDir: live.project_dir, loading: false, error: err.message || String(err), data: null, signing: false };
      window.EU_AGENT_RUN_REVIEW = null;
      repaintBody();
    });
  }
  function currentLiveReview(live) {
    if (!live || agReview.projectDir !== live.project_dir) return null;
    return agReview.data || null;
  }
  function requestRunHistory(force) {
    if (window.EU_DATA !== 'real' || !window.EU_API || !window.EU_API.loadAgentRunHistory) return;
    const s = study();
    if (s.empty) return;
    if (!force && agHistory.studyId === s.id && (agHistory.loading || agHistory.data || agHistory.error)) return;
    agHistory = { studyId: s.id, loading: true, error: null, data: null };
    window.EU_API.loadAgentRunHistory({ study_id: s.id, limit: 50 }).then(data => {
      agHistory = { studyId: s.id, loading: false, error: null, data: data };
      window.EU_AGENT_RUN_HISTORY = data;
      repaintBody();
    }).catch(err => {
      agHistory = { studyId: s.id, loading: false, error: err.message || String(err), data: null };
      repaintBody();
    });
  }
  function requestProviderStatus(force) {
    if (window.EU_DATA !== 'real' || !window.EU_API || !window.EU_API.loadAgentProviderStatus) return;
    if (!force && (agProvider.loading || agProvider.status || agProvider.error)) return;
    agProvider = Object.assign({}, agProvider, { loading: true, error: null });
    window.EU_API.loadAgentProviderStatus(agProvider.provider).then(data => {
      agProvider = Object.assign({}, agProvider, { loading: false, error: null, status: (data && data.provider_status) || data || null });
      repaintBody();
    }).catch(err => {
      agProvider = Object.assign({}, agProvider, { loading: false, error: err.message || String(err), status: null });
      repaintBody();
    });
  }
  function liveRunFromReview(review) {
    const payloads = review && review.artifact_payloads ? review.artifact_payloads : {};
    const context = payloads['run_context.json'] || {};
    return {
      run_id: review.run_id,
      run_label: String(review.run_id || 'run').replace(/_/g, ' '),
      study_id: review.study_id || study().id,
      mode: review.mode || study().mode,
      run_type: review.run_type || 'preflight',
      project_dir: review.project_dir,
      source: context.source || {},
      summary: context.summary || {},
      gate: review.gate || {},
      artifacts: review.artifacts || [],
      uploads: 0,
      tokens: 0,
    };
  }
  function openReview(review) {
    if (!review || !review.ok) return;
    window.EU_AGENT_LAST_RUN = liveRunFromReview(review);
    window.EU_AGENT_RUN_REVIEW = review;
    agReview = { projectDir: review.project_dir, loading: false, error: null, data: review, signing: false };
    agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
    agTab = 'draft';
  }
  function requestArtifact(live, name) {
    if (!live || !live.project_dir || !name || !window.EU_API || !window.EU_API.loadAgentRunArtifact) return;
    agArtifact = { projectDir: live.project_dir, name: name, loading: true, error: null, data: null };
    window.EU_API.loadAgentRunArtifact(live.project_dir, name).then(data => {
      agArtifact = { projectDir: live.project_dir, name: name, loading: false, error: null, data: data };
      repaintBody();
    }).catch(err => {
      agArtifact = { projectDir: live.project_dir, name: name, loading: false, error: err.message || String(err), data: null };
      repaintBody();
    });
  }
  function repaintBody() {
    const app = document.getElementById('app');
    const host = app && app.querySelector('#agHost');
    if (!host) { if (window.__euRender) window.__euRender(); return; }
    host.innerHTML = agShell();
    wire(app);
  }
  function closeRunStream() {
    if (agRun.timer) { clearInterval(agRun.timer); agRun.timer = null; }
    if (agRun.es) { agRun.es.close(); agRun.es = null; }
  }
  function rememberAgentJob(meta) {
    try {
      localStorage.setItem(AG_JOB_KEY, JSON.stringify(Object.assign({ created_at: Date.now() }, meta || {})));
    } catch (_) {}
  }
  function readRememberedAgentJob() {
    try {
      const raw = localStorage.getItem(AG_JOB_KEY);
      const meta = raw ? JSON.parse(raw) : null;
      if (!meta || !meta.job_id) return null;
      const age = Date.now() - Number(meta.created_at || 0);
      if (age > 24 * 60 * 60 * 1000) {
        localStorage.removeItem(AG_JOB_KEY);
        return null;
      }
      return meta;
    } catch (_) {
      return null;
    }
  }
  function clearRememberedAgentJob(jobId) {
    try {
      const meta = readRememberedAgentJob();
      if (!jobId || !meta || meta.job_id === jobId) localStorage.removeItem(AG_JOB_KEY);
    } catch (_) {}
  }
  function lastRunEvent(snapshot) {
    const events = snapshot && Array.isArray(snapshot.events) ? snapshot.events : [];
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i] && events[i].label) return events[i];
    }
    return null;
  }
  function applyRunEventProgress(ev) {
    if (ev && ev.total) agRun.prog = Math.max(agRun.prog || 0, Math.min(1, Number(ev.current || 0) / Number(ev.total || 1)));
    if (ev && ev.label) agRun.step = ev.label;
    const bar = document.querySelector('#agHost .runbar-fill');
    if (bar) bar.style.width = Math.round((agRun.prog || 0) * 100) + '%';
    const detail = document.querySelector('#agHost .nb-d');
    if (detail && agRun.step) detail.innerHTML = `${esc(agRun.step)}${agRun.jobId ? ` · <span class="mono">${esc(agRun.jobId)}</span>` : ''}`;
  }
  function attachAgentJobStream(jobId, s) {
    if (!jobId || !window.EventSource) return;
    closeRunStream();
    const es = new EventSource('/api/jobs/' + jobId + '/events');
    agRun.es = es;
    es.onmessage = msg => {
      const ev = JSON.parse(msg.data);
      applyRunEventProgress(ev);
      if (ev.type === 'end') finishRealRun(s, ev.status, ev.result, ev.error);
    };
    es.onerror = () => {
      if (!agRun.active) return;
      closeRunStream();
      agRun.active = false;
      agRun.error = t('Connection interrupted. If the server job is still running, resume the stream; otherwise retry from the active export.', '连接中断。如果服务端任务仍在运行,可恢复任务流；否则从 active export 重跑。');
      agRun.reconnectable = true;
      repaintBody();
    };
  }
  function restoreAgentJobFromSnapshot(meta, snapshot) {
    const s = study();
    if (meta && meta.study_id && allStudies().some(row => row.id === meta.study_id)) agSel = meta.study_id;
    const target = study();
    if (snapshot.status === 'running') {
      const ev = lastRunEvent(snapshot);
      agRun = {
        active: true,
        prog: 0,
        timer: null,
        es: null,
        jobId: snapshot.id || (meta && meta.job_id),
        step: ev ? ev.label : t('Reconnected to running Agent job', '已重新连接正在运行的 Agent 任务'),
        error: null,
        result: null,
        reconnectable: false,
      };
      applyRunEventProgress(ev || {});
      agTab = 'overview';
      attachAgentJobStream(agRun.jobId, target);
      repaintBody();
      return;
    }
    clearRememberedAgentJob(snapshot.id || (meta && meta.job_id));
    finishRealRun(target, snapshot.status, snapshot.result, snapshot.error);
  }
  function maybeRestoreAgentJob() {
    if (agRun.active || agRun.result || agRun.error || agResumeProbe.loading) return;
    if (!window.EU_API || !window.EU_API.loadJobSnapshot) return;
    if (realMode() && agIdeaProjects.loading) return;
    if (realMode() && !agIdeaProjects.data && !agIdeaProjects.error) {
      requestIdeaAgentProjects();
      return;
    }
    const meta = readRememberedAgentJob();
    if (!meta || !meta.job_id || agResumeProbe.checkedJobId === meta.job_id) return;
    agResumeProbe = { loading: true, checkedJobId: meta.job_id };
    window.EU_API.loadJobSnapshot(meta.job_id).then(snapshot => {
      agResumeProbe.loading = false;
      restoreAgentJobFromSnapshot(meta, snapshot);
    }).catch(() => {
      agResumeProbe.loading = false;
      clearRememberedAgentJob(meta.job_id);
    });
  }

  /* ---------------- pipeline ---------------- */
  function pipeStages(mode) {
    if (mode === 'idea') return [
      ['Frame', '想法框定', t('source + exposure/outcome', '来源 + 暴露/结局'), 'spark'],
      ['Recipe', '数据配方', t('data recipe', '数据配方'), 'layers'],
      ['Dry-run', '试运行', t('feasibility · no claims', '可行性 · 不下结论'), 'play'],
      ['Recommend', '建议', t('suggested workflow', '建议的研究流程'), 'target'],
    ];
    return [
      ['Plan', '计划', t('question → recipe', '问题 → 配方'), 'play'],
      ['Build', '构建', t('exports → 1 row / stay', '导出 → 每次住院一行'), 'layers'],
      ['Analyze', '分析', t('tables · figures · checks', '表 · 图 · 校验'), 'viz'],
      ['Gate', '证据闸', t('evidence before drafting', '撰稿前的证据校验'), 'shield'],
      ['Draft', '草稿', t('approve · export', '确认 · 导出'), 'file'],
    ];
  }
  function pipeline() {
    const s = study();
    const stages = pipeStages(s.mode);
    const now = s.signed ? stages.length - 1 : s.stage;
    let html = '';
    stages.forEach((st, i) => {
      const en = st[0], zh = st[1], d = st[2], ic = st[3];
      const isGate = s.mode === 'analysis' && i === 3 && !s.signed;
      const cls = i < now ? 'done' : i === now ? (isGate ? 'gate' : 'now') : '';
      const node = i < now ? icon('check', 14, 2.8) : (isGate ? icon('lock', 13, 2) : icon(ic, 13));
      if (i > 0) html += `<div class="pline ${i <= now ? 'done' : ''}"></div>`;
      html += `<div class="pstep ${cls}"><div class="pnode">${node}</div><div class="pmeta"><div class="pt">${t(en, zh)}</div><div class="pd">${d}</div></div></div>`;
    });
    return `<div class="ag-pipe">${html}</div>`;
  }

  /* ---------------- studies list ---------------- */
  function studyList() {
    const studies = allStudies();
    const dotCls = { ready: 'ready', running: 'running', gate: 'running', draft: 'draft', idle: 'idle' };
    return `
    <div class="ag-list">
      <div class="ag-list-head">
        <div><span class="ttl">${t('Studies', '研究项目')} · ${studies.length}</span><div class="ag-list-cap">${t('each study = a local project folder', '每个研究 = 一个本地项目文件夹')}</div></div>
        <button class="ag-newbtn" data-ag-new>${icon('plus', 13)} ${t('New', '新建')}</button>
      </div>
      <div class="ag-studies">
        ${agIdeaProjects.loading ? `<div class="empty-mini" style="margin:10px;min-height:80px;">${t('Loading idea project seeds…', '正在加载 idea 项目种子…')}</div>` : ''}
        ${agIdeaProjects.error ? `<div class="note warn" style="margin:10px;"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="t">${t('Idea project seeds unavailable', 'Idea 项目种子不可用')}</div><div class="d">${esc(agIdeaProjects.error)}</div></div></div>` : ''}
        ${!studies.length && !agIdeaProjects.loading ? `<div class="empty-mini ideas-empty-list" style="margin:10px;min-height:210px;">
          <div>${icon('folder', 22)}</div>
          <h3>${t('No local projects yet', '还没有本地项目')}</h3>
          <p>${t('Real mode only lists project seeds and runs written on this machine. Create one from Idea Mining, then refresh.', '真实模式只列出写在本机的 project seed 和 run。从 Idea Mining 创建后再刷新。')}</p>
          <div class="row gap-8 mt-12" style="justify-content:center;">
            <button class="btn primary sm" data-nav="ideas">${icon('target', 12)} ${t('Open Idea Mining', '打开 Idea Mining')}</button>
            <button class="btn sm" data-ag-refresh-projects>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
          </div>
        </div>` : ''}
        ${studies.map(s => {
          const zh = window.EU_LANG === 'zh';
          const folder = projectFolderLabel(s);
          return `
          <button class="studycard ${s.id === agSel ? 'on' : ''}" data-ag-sel="${s.id}">
            <div class="sc-top">
              <span class="sc-dot ${dotCls[s.status] || 'idle'}"></span>
              <span class="sc-name">${t(s.name[0], s.name[1])}</span>
              <span class="sc-mode analysis">${s.ideaSeed ? t('Idea seed', '想法种子') : t('Analysis', '分析')}</span>
            </div>
            <div class="sc-meta"><span class="sc-folder">${icon('folder', 11)} ${folder}</span></div>
            <div class="sc-meta" style="margin-top:3px;">${s.runs.length ? `${s.runs[0][0]}<span class="mid"></span>${s.runs[0][4][zh ? 1 : 0]}` : t('not run yet', '尚未运行')}</div>
          </button>`;
        }).join('')}
      </div>
    </div>`;
  }

  /* ---------------- detail header ---------------- */
  function detailHead() {
    const s = study();
    const live = liveRunForStudy();
    const review = currentLiveReview(live);
    const liveSigned = !!(review && review.signed);
    const statusPill = {
      ready: `<span class="pill ok"><span class="dot"></span>${t('Ready to run', '可运行')}</span>`,
      reviewed: `<span class="pill ok"><span class="dot"></span>${t('Signed analysis-only', '已签署 analysis-only')}</span>`,
      gate: `<span class="pill warn"><span class="dot"></span>${t('Awaiting sign-off', '待签署')}</span>`,
      running: `<span class="pill warn"><span class="dot"></span>${t('Running', '运行中')}</span>`,
      draft: `<span class="pill demo"><span class="dot"></span>${t('Exploring', '探索中')}</span>`,
      idle: `<span class="pill"><span class="dot"></span>${t('Not run yet', '尚未运行')}</span>`,
    }[liveSigned ? 'reviewed' : (s.signed ? 'ready' : s.status)] || '';
    return `
    <div class="ag-dhead">
      <div class="ag-dtop">
        <div style="min-width:0;">
          <div class="ag-title">${t(s.name[0], s.name[1])} <span class="editmk">${icon('edit', 14)}</span></div>
          <div class="ag-src">
            <span class="lk" title="${t('Local project folder — intermediate files are written here', '本地项目文件夹 — 中间文件写在这里')}">${icon('folder', 12)} ${esc(projectFolderLabel(s))}</span>
            <span class="mid"></span>
            <span class="lk">${icon('cohort', 12)} ${s.cohort}</span>
            ${activeSourceLabel() ? `<span class="mid"></span><span class="lk">${icon('db', 12)} ${activeSourceLabel()}</span>` : ''}
            <span class="mid"></span>
            ${statusPill}
          </div>
        </div>
        <div class="row gap-8">
          <button class="btn sm" data-nav="ideas">${icon('target', 13)} ${t('Open Idea Mining', '打开 Idea 挖掘')}</button>
          <span class="pill ok"><span class="dot"></span>${t('Analysis workspace', '分析运行工作台')}</span>
        </div>
      </div>
      ${pipeline()}
    </div>`;
  }

  /* ---------------- tabs ---------------- */
  function tabsFor(mode) {
    const s = study();
    if (s.empty) return [
      ['overview', t('Overview', '概览'), null],
    ];
    if (mode === 'idea') return [
      ['overview', t('Overview', '概览'), null],
      ['runs', t('Dry-runs', '试运行'), s.runs.length],
      ['notes', t('Notes', '笔记'), null],
    ];
    return [
      ['overview', t('Overview', '概览'), null],
      ['runs', t('Runs', '运行历史'), s.runs.length],
      ['outputs', t('Outputs', '产出'), outputCountForStudy()],
      ['draft', t('Draft', '草稿'), null],
    ];
  }
  function tabsRow() {
    const s = study();
    const tabs = tabsFor(s.mode);
    if (!tabs.some(x => x[0] === agTab)) agTab = 'overview';
    return `<div class="ag-tabs" data-ag-tabs>
      ${tabs.map(([id, lab, cnt]) => `<button class="ag-tab ${agTab === id ? 'on' : ''}" data-ag-tab="${id}">${lab}${cnt != null ? `<span class="cnt">${cnt}</span>` : ''}</button>`).join('')}
    </div>`;
  }

  /* ---------------- tab bodies ---------------- */
  function planList() {
    const s = study();
    const seedPlan = s.ideaSeed && Array.isArray(s.ideaSeed.analysis_plan) ? s.ideaSeed.analysis_plan : null;
    const plan = seedPlan
      ? seedPlan.map(x => [x, t('from Idea Mining handoff', '来自 Idea Mining 交接'), 'ready'])
      : s.mode === 'idea'
      ? [
        [t('Frame the idea', '框定想法'), t('exposure / outcome / source', '暴露 / 结局 / 来源'), 'ready'],
        [t('Data recipe', '数据配方'), t('concepts the dry-run needs', '试运行所需概念'), 'ready'],
        [t('Feasibility dry-run', '可行性试运行'), t('counts, coverage — no effect sizes', '计数、覆盖率 —— 不给效应量'), 'ready'],
        [t('Recommendation', '建议'), t('is a full study worth it?', '是否值得开展完整研究?'), 'ready'],
      ]
      : [
        [t('Cohort summary', '队列摘要'), t('n, demographics, outcome rates', 'n、人口学、结局率'), 'ready'],
        [t('Table 1', 'Table 1'), t('baseline by group', '分组基线特征'), 'ready'],
        [t('Missingness audit', '缺失审计'), t('coverage + denominators', '覆盖率 + 分母'), 'ready'],
        [t('Model: LR + SOFA + lactate', '模型:LR + SOFA + 乳酸'), t('first-24h predictors', '前 24h 预测因子'), 'ready'],
        [t('ROC · Calibration', 'ROC · 校准'), t('discrimination + calibration', '区分度 + 校准度'), 'ready'],
        [t('Manuscript draft', '论文草稿'), t('methods + results', '方法 + 结果'), 'gated'],
      ];
    return `
      <div class="card pad">
        <div class="row" style="justify-content:space-between;align-items:baseline;">
          <div class="eyebrow">${t('Plan', '计划')} · ${plan.length} ${t('steps', '步')}</div>
          ${s.mode === 'analysis' ? `<span class="pill ok" style="height:20px;"><span class="dot"></span>5 ${t('ready', '就绪')} · 1 ${t('gated', '受闸')}</span>` : `<span class="pill ok" style="height:20px;"><span class="dot"></span>${t('feasibility only', '仅可行性')}</span>`}
        </div>
        <div class="planlist mt-12">
          ${plan.map(([ti, d, st], i) => `
            <div class="plan-item ${st}">
              <div class="pi-n mono">${String(i + 1).padStart(2, '0')}</div>
              <div class="pi-node">${st === 'gated' ? icon('lock', 11, 2) : icon('check', 12, 2.6)}</div>
              <div class="pi-body"><div class="pi-t">${ti}</div><div class="pi-d">${d}</div></div>
              <div class="pi-tag">${st === 'gated' ? `<span class="pill dashed">${t('requires review', '需审阅')}</span>` : `<span class="pill ok" style="height:20px;"><span class="dot"></span>${t('planned', '已计划')}</span>`}</div>
            </div>`).join('')}
        </div>
      </div>`;
  }

  function providerRunPanel() {
    if (window.EU_DATA !== 'real') return '';
    if (study().empty || study().mode !== 'analysis') return '';
    const src = activeExportSource();
    requestProviderStatus();
    const st = agProvider.status || {};
    const limits = st.limits || {};
    const envFile = st.env_file || {};
    const missing = Array.isArray(st.missing) ? st.missing : [];
    const ready = !!st.ready;
    const canRun = !!(src && ready && agProvider.consent && !agRun.active);
    const disabledReason = !src
      ? t('No active export source', '没有 active export 源')
      : !ready
      ? (missing.length ? missing.join(', ') : t('Provider not ready', 'provider 未就绪'))
      : !agProvider.consent
      ? t('Per-run confirmation required', '需要逐次确认')
      : agRun.active
      ? t('Run already in progress', '已有运行进行中')
      : '';
    const providers = [
      ['openai', 'OpenAI'],
      ['openrouter', 'OpenRouter'],
      ['deepseek', 'DeepSeek-compatible'],
      ['custom', 'Custom / local OpenAI-compatible'],
    ];
    return `
      <div class="card pad">
        <div class="row" style="justify-content:space-between;align-items:baseline;">
          <div>
            <div class="eyebrow">${t('External provider control', '外部 provider 控制')}</div>
            <div class="panel-sub" style="margin-top:4px;">${t('Optional full-agent scaffold. Uses env vars only; secrets are never shown or written to artifacts.', '可选 full-agent 骨架。只读取环境变量；密钥不会显示,也不会写入产物。')}</div>
          </div>
          <button class="btn sm ghost" data-ag-provider-refresh>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
        </div>
        <div class="row wrap gap-6 mt-12">
          ${providers.map(([p, label]) => `<button class="btn sm ${agProvider.provider === p ? 'primary' : 'ghost'}" data-ag-provider="${p}">${esc(label)}</button>`).join('')}
        </div>
        <div class="note-line mt-8" style="font-size:11px;color:var(--ink-4);">${icon('shield', 11)} ${t('Custom/local endpoints must be OpenAI-compatible and configured by environment variables; values are never shown here.', 'Custom/本地端点必须兼容 OpenAI Chat Completions,并通过环境变量配置;这里不会显示具体值。')}</div>
        ${agProvider.loading ? `<div class="note info mt-12"><div class="ico">${icon('shield', 16)}</div><div class="body"><span class="t">${t('Checking provider readiness', '正在检查 provider 就绪状态')}</span><span class="d">${t('No client is constructed and no network call is made.', '不会构造 client,也不会发起网络调用。')}</span></div></div>` : ''}
        ${agProvider.error ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('Provider status unavailable', 'provider 状态不可用')}</span><span class="d">${esc(agProvider.error)}</span></div></div>` : ''}
        <div class="row wrap gap-6 mt-12">
          <span class="pill ${st.ai_enabled ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>AI ${st.ai_enabled ? t('enabled', '已开启') : t('off', '关闭')}</span>
          <span class="pill ${st.credential_present ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>${st.credential_present ? t('key env present', 'key env 已配置') : t('key env missing', 'key env 缺失')}</span>
          <span class="pill ${st.model_present ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>${st.model_present ? t('model env present', 'model env 已配置') : t('model env missing', 'model env 缺失')}</span>
          <span class="pill ${st.ready ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>${st.ready ? t('ready', '就绪') : t('blocked', '受阻')}</span>
        </div>
        <div class="cols-2 mt-12" style="gap:8px;">
          ${[
            [t('Credential source', '凭据来源'), st.credential_source || (st.credential_env_candidates || []).join(' / ') || '—'],
            [t('Model source', '模型来源'), st.model_source || (st.model_env_candidates || []).join(' / ') || '—'],
            [t('Base URL source', 'Base URL 来源'), st.base_url_source || (st.base_url_env_candidates || []).join(' / ') || '—'],
            [t('Private env file', '私有 env 文件'), envFile.status ? `${envFile.status}${Array.isArray(envFile.loaded_keys) && envFile.loaded_keys.length ? ' · ' + envFile.loaded_keys.join(' / ') : ''}` : '—'],
            [t('Budget', '预算'), `${Number(limits.max_external_calls_per_run || 1)} call · ${Number(limits.max_output_tokens || 1200)} max tokens`],
          ].map(([k, v]) => `
            <div style="padding:8px 10px;background:var(--surface-2);border-radius:var(--r-2);min-width:0;">
              <div class="eyebrow" style="font-size:9.5px;">${k}</div>
              <div class="mono" style="font-size:11.5px;color:var(--ink);margin-top:3px;overflow:hidden;text-overflow:ellipsis;">${esc(v)}</div>
            </div>`).join('')}
        </div>
        <label class="rtodo-row mt-12" style="background:var(--surface-2);">
          <input type="checkbox" data-ag-external-consent ${agProvider.consent ? 'checked' : ''} />
          <span class="rtodo-t">${t('I authorize this single external provider call for this run only', '我授权本次运行进行一次外部 provider 调用')}</span>
          <span class="rtodo-ref mono">per_run_opt_in</span>
        </label>
        <div class="row gap-8 mt-12">
          <button class="btn primary sm" data-ag-external-run aria-disabled="${canRun ? 'false' : 'true'}">${icon('play', 12)} ${t('Run full with provider', '使用 provider 运行 full')}</button>
          <span style="font-size:11px;color:var(--ink-4);align-self:center;">${canRun ? t('Will remain analysis_only unless STRICT evidence and human review pass.', '除非 STRICT evidence 与人工审阅通过,否则仍保持 analysis_only。') : esc(disabledReason)}</span>
        </div>
      </div>`;
  }

  function nextBar() {
    const s = study();
    if (agRun.active) {
      const detail = agRun.step || t('deterministic · no tokens · nothing uploaded', '确定性 · 不消耗 token · 不上传');
      return `
      <div class="nextbar">
        <div class="nb-ico"><span class="spin sm" style="width:16px;height:16px;border-top-color:#fff;"></span></div>
        <div class="grow">
          <div class="nb-t">${t('Running the analysis…', '正在运行分析…')}</div>
          <div class="nb-d">${esc(detail)}${agRun.jobId ? ` · <span class="mono">${esc(agRun.jobId)}</span>` : ''}</div>
          <div class="runbar mt-8" style="height:6px;"><div class="runbar-fill" style="width:${Math.round(agRun.prog * 100)}%;transition:width .12s linear;"></div></div>
        </div>
        ${agRun.jobId ? `<button class="btn ghost" data-ag-cancel-job>${icon('stop', 13)} ${t('Request cancel', '请求取消')}</button>` : ''}
      </div>`;
    }
    if (agRun.error) {
      const canReconnect = !!agRun.jobId && !!agRun.reconnectable;
      const retryLabel = agRun.result && agRun.result.cancelled
        ? t('Restart from active export', '从 active export 重跑')
        : t('Retry from active export', '从 active export 重试');
      return `
      <div class="nextbar gate">
        <div class="nb-ico">${icon('shield', 16)}</div>
        <div class="grow"><div class="nb-t">${agRun.result && agRun.result.cancelled ? t('Run cancelled safely', '运行已安全取消') : t('Run failed closed', '运行已 fail-closed')}</div><div class="nb-d">${esc(agRun.error)}</div></div>
        ${canReconnect ? `<button class="btn" data-ag-reconnect>${icon('history', 13)} ${t('Resume stream', '恢复任务流')}</button>` : ''}
        <button class="btn primary" data-ag-runbtn>${icon('refresh', 13)} ${retryLabel}</button>
      </div>`;
    }
    if (s.mode === 'idea') {
      return `
      <div class="nextbar accent">
        <div class="nb-ico" style="background:oklch(52% 0.10 280);">${icon('play', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Run a feasibility dry-run', '运行可行性试运行')}</div><div class="nb-d">${t('Counts and coverage only — no effect sizes, no manuscript.', '仅计数与覆盖率 —— 不给效应量,不生成论文。')}</div></div>
        <button class="btn primary" data-ag-runbtn>${icon('play', 13)} ${t('Run dry-run', '运行试运行')}</button>
      </div>`;
    }
    if (s.signed) {
      return `
      <div class="nextbar">
        <div class="nb-ico" style="background:var(--ok);">${icon('check', 16, 2.8)}</div>
        <div class="grow"><div class="nb-t">${t('All checks passed · draft unlocked', '全部校验通过 · 草稿已解锁')}</div><div class="nb-d">${t('The manuscript draft is ready to open and export.', '论文草稿可打开并导出。')}</div></div>
        <button class="btn primary" data-ag-tab="draft">${icon('file', 13)} ${t('Open draft', '打开草稿')}</button>
      </div>`;
    }
    if (s.status === 'gate') {
      const checks = [
        [t('Denominators resolved', '\u5206\u6bcd\u5df2\u786e\u5b9a'), true],
        [t('Coverage \u2265 threshold', '\u8986\u76d6\u7387 \u2265 \u9608\u503c'), true],
        [t('Reproduces from manifest', '\u53ef\u7531\u6e05\u5355\u590d\u73b0'), true],
        [t('Model card attached', '\u5df2\u9644\u6a21\u578b\u8bf4\u660e\u5361'), true],
        [t('Reviewer sign-off', '\u5ba1\u9605\u8005\u7b7e\u7f72'), false],
      ];
      const passed = checks.filter(c => c[1]).length;
      return `
      <div class=\"nextbar gate\">
        <div class=\"nb-ico\">${icon('shield', 16)}</div>
        <div class=\"grow\"><div class=\"nb-t\">${t('Evidence gate \u2014 1 check pending', '\u8bc1\u636e\u95f8 \u2014\u2014 \u8fd8\u5dee 1 \u9879\u6821\u9a8c')}</div><div class=\"nb-d\">${t('The draft stays locked until a reviewer signs off the findings.', '\u5728\u5ba1\u9605\u8005\u7b7e\u7f72\u7ed3\u8bba\u524d,\u8349\u7a3f\u4fdd\u6301\u9501\u5b9a\u3002')}</div></div>\n        <button class=\"btn primary\" data-ag-tab=\"draft\">${icon('check', 13)} ${t('Review & sign off', '\u5ba1\u9605\u5e76\u7b7e\u7f72')}</button>\n      </div>\n      <div class=\"card pad\" style=\"margin-top:10px;\">\n        <div class=\"eyebrow\" style=\"display:flex;align-items:center;gap:8px;\">${t('What unlocks the draft', '\u89e3\u9501\u8349\u7a3f\u7684\u6761\u4ef6')}<span class=\"mono\" style=\"margin-left:auto;color:var(--ink-4);font-size:10.5px;\">${passed}/${checks.length}</span></div>\n        <div class=\"gate-checklist\">\n          ${checks.map(([label, ok]) => `\n            <div class=\"gc-row ${ok ? 'ok' : 'pending'}\">\n              <span class=\"gc-mk\">${ok ? icon('check', 11, 2.8) : icon('clock', 11)}</span>\n              <span>${label}</span>\n              <span class=\"gc-tag\">${ok ? t('passed', '\u901a\u8fc7') : t('pending \u00b7 your turn', '\u5f85\u529e \u00b7 \u8f6e\u5230\u4f60')}</span>\n            </div>`).join('')}\n        </div>\n      </div>`;
    }
    if (s.status === 'idle') {
      return `
      <div class="nextbar accent">
        <div class="nb-ico">${icon('play', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Ready to run the plan', '准备运行计划')}</div><div class="nb-d">${t('Steps execute deterministically and stream into the run. You confirm before any model call.', '步骤确定性执行并流入运行记录。任何模型调用前都需你确认。')}</div></div>
        <button class="btn primary" data-ag-runbtn>${icon('play', 13)} ${t('Run analysis', '运行分析')}</button>
      </div>`;
    }
    return `
      <div class="nextbar accent">
        <div class="nb-ico">${icon('refresh', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Re-run or extend the analysis', '重新运行或扩展分析')}</div><div class="nb-d">${t('Outputs are current. Run again to refresh, or move to the evidence gate.', '产出为最新。可重新运行刷新,或前往证据闸。')}</div></div>
        <button class="btn primary" data-ag-runbtn>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>
      </div>`;
  }

  function contextStats() {
    const s = study();
    const src = activeExportSource();
    const sum = (src && src.summary) || {};
    const stats = src
      ? [['Stays', '住院数', sum.stays == null ? '—' : Number(sum.stays).toLocaleString()], ['Modules', '模块', sum.modules == null ? '—' : String(sum.modules)], ['Rows', '行数', sum.total_rows == null ? '—' : Number(sum.total_rows).toLocaleString()], ['Gate', '证据闸', 'strict']]
      : s.id === 'crossdb'
      ? [['Databases', '数据库', '3'], ['Shared concepts', '共享概念', '6'], ['Mortality', '死亡率', '20.0%'], ['Concordance', '一致性', 'high']]
      : [['Mean age', '平均年龄', '54.8 y'], ['Mortality', '死亡率', '20.0%'], ['Sepsis-3', 'Sepsis-3', '45.3%'], ['Mech vent', '机械通气', '52.1%']];
    const linked = src ? (src.label || src.database || 'local export') : t(s.source[0], s.source[1]);
    const linkedPath = src ? src.path : null;
    return `
    <div class="card pad">
      <div class="eyebrow" style="margin-bottom:12px;">${t('Project folder', '项目文件夹')}</div>
      <div class="row gap-8" style="align-items:center;"><span style="color:var(--ink-3);flex:none;">${icon('folder', 14)}</span><div class="mono" style="font-size:11.5px;color:var(--ink-2);min-width:0;overflow:hidden;text-overflow:ellipsis;">${esc(projectFolderLabel(s))}</div></div>
      <div class="eyebrow" style="margin:14px 0 8px;">${src ? t('Linked export source', '关联导出源') : t('Linked cohort', '关联队列')}</div>
      <div style="font-weight:600;font-size:13px;">${s.cohort}</div>
      <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${linkedPath || linked}</div>
      ${s.ideaSeed ? `<div class="note ok mt-12"><div class="ico">${icon('target', 13)}</div><div class="body"><div class="t">${t('Seeded from Idea Mining', '来自 Idea Mining 种子')}</div><div class="d">${esc(s.sourceArticle || '')}</div></div></div>` : ''}
      <div class="cols-2 mt-12" style="gap:8px;">
        ${stats.map(([en, zh, v]) => `
          <div style="padding:8px 10px;background:var(--surface-2);border-radius:var(--r-2);">
            <div class="eyebrow" style="font-size:9.5px;">${t(en, zh)}</div>
            <div class="mono" style="font-size:13px;font-weight:500;color:var(--ink);margin-top:3px;">${v}</div>
          </div>`).join('')}
      </div>
      <button class="btn sm block mt-16" data-nav="extraction">${icon('layers', 13)} ${t('Open in Data Extraction', '在数据抽取中打开')}</button>
    </div>`;
  }

  function tabOverview() {
    const s = study();
    if (s.empty) {
      return `
      <div class="state-hero empty-state" style="min-height:360px;">
        <div class="glyph">${icon('folder', 28)}</div>
        <div class="st-t">${t('No local Agent project selected', '尚未选择本地 Agent 项目')}</div>
        <div class="st-d">${t('Agent Projects no longer shows fabricated studies in Real mode. Create a project seed from Idea Mining, or switch to Demo to inspect examples.', '真实模式下 Agent Projects 不再显示编造研究。请从 Idea Mining 创建 project seed，或切到 Demo 查看示例。')}</div>
        <div class="st-actions">
          <button class="btn primary" data-nav="ideas">${icon('target', 14)} ${t('Create from Idea Mining', '从 Idea Mining 创建')}</button>
          <button class="btn" data-ag-refresh-projects>${icon('refresh', 13)} ${t('Refresh local projects', '刷新本地项目')}</button>
        </div>
      </div>`;
    }
    const staleBanner = (window.EU_STALE && !agRun.active) ? `
      <div class="stale-banner">
        <span class="sb-ico">${icon('refresh', 16)}</span>
        <div class="grow"><div class="sb-t">${t('Extraction changed since the last run', '自上次运行后抽取已变更')}</div><div class="sb-d">${t('The cohort or modules were edited — runs, outputs and the draft are out of date until you re-run.', '队列或模块被修改 — 运行、产出和草稿在重跑前都已过期。')}</div></div>
        <button class="btn primary" data-ag-runbtn>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>
      </div>` : '';
    return `
      ${staleBanner}
      ${nextBar()}
      ${s.mode === 'idea' ? `<div class="idea-band mt-16"><span class="ico">${icon('spark', 16)}</span><div><div style="font-weight:600;font-size:13px;">${t('Legacy feasibility seed', '旧可行性种子')}</div><div style="font-size:12px;color:var(--ink-3);margin-top:2px;">${t('New discovery work starts in Idea Mining. Agent Projects only executes confirmed analysis runs.', '新的发现流程从 Idea 挖掘开始。研究项目只执行已确认的分析运行。')}</div></div></div>` : ''}
      <div class="split-320 mt-16" style="grid-template-columns:1fr 300px;">
        <div class="col gap-16">
          <div class="card pad">
            <div class="eyebrow">${t('Research question', '研究问题')}</div>
            <div class="qbox mt-12" style="font-size:13.5px;line-height:1.5;">${t(s.question[0], s.question[1])}</div>
            <div class="row wrap gap-6 mt-12">
              <span class="chip">@${s.cohort}</span>
              ${s.id === 'crossdb' ? '<span class="chip">@cross_db</span>' : '<span class="chip">@first_24h</span>'}
              <span class="chip">@${s.id === 'aki' ? 'kdigo' : 'lactate'}</span>
            </div>
          </div>
          ${planList()}
          ${providerRunPanel()}
        </div>
        ${contextStats()}
      </div>
      <div class="handoff">
        <span class="ho-ico">${icon('spark', 17)}</span>
        <div class="ho-body"><b>${t('Rather drive this study by chat?', '想用对话来推进这项研究?')}</b> ${t('Guided study walks the same plan → run → review → gated-draft workflow conversationally, then hands the study back here.', '研究引导用对话走同一套 计划 → 运行 → 审阅 → 受闸草稿 的流程,完成后把研究交回这里。')}</div>
        <button class="btn" data-nav="guided">${icon('spark', 13)} ${t('Continue in Guided study', '在研究引导中继续')} ${icon('arrow', 13)}</button>
      </div>`;
  }

  function tabRuns() {
    const s = study();
    if (window.EU_DATA === 'real') {
      requestRunHistory();
      const rows = agHistory.studyId === s.id && agHistory.data && Array.isArray(agHistory.data.runs) ? agHistory.data.runs : [];
      return `
      <div class="card pad" style="padding:16px 18px 8px;">
        <div class="panel-head" style="margin-bottom:6px;">
          <div><div class="panel-title" style="font-size:15px;">${t('Run history', '运行历史')}</div><div class="panel-sub">${t('Whitelisted local artifacts only · hashes checked on every review.', '仅白名单本地产物 · 每次回看都校验哈希。')}</div></div>
          <button class="btn sm" data-ag-history-refresh>${icon('refresh', 13)} ${t('Refresh', '刷新')}</button>
        </div>
        ${agHistory.studyId === s.id && agHistory.loading ? `<div class="note info mt-12"><div class="ico">${icon('history', 16)}</div><div class="body"><span class="t">${t('Loading local run history', '正在加载本地运行历史')}</span><span class="d">${t('Scanning configured local Agent project folders without reading export rows.', '扫描已配置的本地 Agent 项目文件夹,不读取 export 行。')}</span></div></div>` : ''}
        ${agHistory.studyId === s.id && agHistory.error ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('History load failed', '历史加载失败')}</span><span class="d">${esc(agHistory.error)}</span></div></div>` : ''}
        ${!rows.length && !(agHistory.studyId === s.id && agHistory.loading) ? `<div class="state-hero empty-state"><div class="glyph">${icon('history', 26)}</div><div class="st-t">${t('No local runs found', '未找到本地运行')}</div><div class="st-d">${t('Run the plan from the Overview tab. History is rebuilt from local artifacts, not browser memory.', '从概览页运行计划。历史由本地产物重建,不依赖浏览器内存。')}</div></div>` : ''}
        <div style="margin-top:6px;">
          ${rows.map((r, ri) => {
            const stale = !!r.signoff_stale;
            const status = r.readiness_status || r.gate_status || 'analysis_only';
            const ok = !stale && status !== 'blocked';
            return `
            <div class="runrow">
              <div class="rn-node">${ok ? `<span style="color:var(--ok);">${icon('check', 14, 2.8)}</span>` : `<span style="color:var(--bad);">${icon('lock', 12, 2)}</span>`}</div>
              <div><div class="run-name mono">${esc(r.run_label || r.run_id || ('run ' + (ri + 1)))}${stale ? ` <span class="jp-stale">${icon('alert', 9)} ${t('tampered', '已篡改')}</span>` : ''}</div><div class="run-scope">${esc(status)} · ${Number(r.artifact_count || 0)} ${t('artifacts', '产物')}</div></div>
              <div class="row gap-10" style="flex:none;">
                <span class="pill ${stale ? 'bad' : (r.signed ? 'ok' : 'warn')}" style="height:20px;"><span class="dot"></span>${esc(stale ? 'signoff_stale' : status)}</span>
                <span class="mono" style="font-size:11px;color:var(--ink-4);width:54px;text-align:right;">${esc(r.run_type || '')}</span>
              </div>
              <div class="mono" style="font-size:11px;color:var(--ink-4);text-align:right;white-space:nowrap;">${esc((r.updated_at || '').slice(0, 19).replace('T', ' '))}</div>
              <button class="btn sm ghost" data-ag-history-open="${ri}" title="${t('Open this run review', '打开这次运行审阅')}">${icon('eye', 12)} ${t('Review', '审阅')}</button>
            </div>`;
          }).join('')}
        </div>
      </div>`;
    }
    if (!s.runs.length) {
      return `<div class="state-hero empty-state"><div class="glyph">${icon('history', 26)}</div><div class="st-t">${t('No runs yet', '尚无运行记录')}</div><div class="st-d">${t('Run the plan from the Overview tab to populate this history. Every run writes a local manifest.', '在概览页运行计划即可填充历史。每次运行都会写入本地清单。')}</div><div class="st-actions"><button class="btn primary" data-ag-tab="overview">${icon('play', 14)} ${t('Go to Overview', '前往概览')}</button></div></div>`;
    }
    return `
      <div class="card pad" style="padding:16px 18px 8px;">
        <div class="panel-head" style="margin-bottom:6px;">
          <div><div class="panel-title" style="font-size:15px;">${t('Run history', '运行历史')}</div><div class="panel-sub">${t('Local manifests · resumable — nothing leaves your machine.', '本地清单 · 可继续运行 —— 不离开你的机器。')}</div></div>
          <button class="btn sm">${icon('download', 13)} ${t('Export ledger', '导出账本')}</button>
        </div>
        <div style="margin-top:6px;">
          ${s.runs.map((r, ri) => `
            <div class="runrow">
              <div class="rn-node">${r[2] === 'complete' ? `<span style="color:var(--ok);">${icon('check', 14, 2.8)}</span>` : `<span style="color:var(--bad);">${icon('lock', 12, 2)}</span>`}</div>
              <div><div class="run-name mono">${r[0]}${ri === 0 && window.EU_STALE ? ` <span class="jp-stale">${icon('refresh', 9)} ${t('stale', '过期')}</span>` : ''}</div><div class="run-scope">${t(r[1][0], r[1][1])}</div></div>
              <div class="row gap-10" style="flex:none;">
                ${r[2] === 'complete' ? `<span class="pill ok" style="height:20px;"><span class="dot"></span>${t('complete', '完成')}</span>` : `<span class="pill bad" style="height:20px;"><span class="dot"></span>${t('blocked', '受阻')}</span>`}
                <span class="mono" style="font-size:11px;color:var(--ink-4);width:54px;text-align:right;">${r[3]}</span>
              </div>
              <div class="mono" style="font-size:11px;color:var(--ink-4);text-align:right;white-space:nowrap;">${t(r[4][0], r[4][1])}</div>
              <button class="btn sm ghost" data-ag-runbtn title="${t('Resume this run', '继续这次运行')}">${icon('refresh', 12)} ${t('Resume', '继续')}</button>
            </div>`).join('')}
        </div>
      </div>`;
  }

  function thumb(kind) {
    if (kind === 'num') return `<div class="mono" style="font-size:30px;font-weight:500;color:var(--ink);">10</div>`;
    if (kind === 'file') return `<div style="color:var(--ink-3);">${icon('file', 34, 1.8)}</div>`;
    if (kind === 'table') return `<svg width="120" height="64" viewBox="0 0 120 64">${[0, 1, 2, 3, 4].map(r => `<rect x="12" y="${8 + r * 11}" width="38" height="5" fill="${r === 0 ? 'var(--ink-3)' : 'var(--hair-3)'}" rx="1"/><rect x="56" y="${8 + r * 11}" width="22" height="5" fill="var(--hair-2)" rx="1"/><rect x="84" y="${8 + r * 11}" width="22" height="5" fill="var(--hair-2)" rx="1"/>`).join('')}</svg>`;
    if (kind === 'roc') return `<svg width="120" height="64" viewBox="0 0 120 64"><line x1="14" y1="54" x2="106" y2="8" stroke="var(--hair-2)" stroke-dasharray="2 3"/><line x1="14" y1="54" x2="14" y2="8" stroke="var(--hair-3)"/><line x1="14" y1="54" x2="106" y2="54" stroke="var(--hair-3)"/><path d="M14 54 Q 30 24 60 16 Q 90 11 106 9" stroke="var(--accent)" stroke-width="1.8" fill="none"/></svg>`;
    if (kind === 'calib') return `<svg width="120" height="64" viewBox="0 0 120 64"><line x1="14" y1="54" x2="106" y2="8" stroke="var(--hair-2)" stroke-dasharray="2 3"/><line x1="14" y1="54" x2="14" y2="8" stroke="var(--hair-3)"/><line x1="14" y1="54" x2="106" y2="54" stroke="var(--hair-3)"/><path d="M14 52 Q 40 40 62 30 Q 86 18 104 10" stroke="var(--ok)" stroke-width="1.8" fill="none"/></svg>`;
    return `<svg width="120" height="64" viewBox="0 0 120 64">${Array.from({ length: 6 }, (_, r) => Array.from({ length: 11 }, (_, c) => { const m = ((r * 7 + c * 3) % 10) > 7; return `<rect x="${10 + c * 9}" y="${8 + r * 7.5}" width="6.5" height="5.5" fill="${m ? 'var(--bad)' : 'var(--hair-3)'}" opacity="${m ? 0.65 : 1}" rx="0.5"/>`; }).join('')).join('')}</svg>`;
  }
  function artifactViewer(live) {
    if (!live || agArtifact.projectDir !== live.project_dir) return '';
    if (agArtifact.loading) {
      return `<div class="card pad mt-16"><div class="eyebrow">${t('Artifact viewer', '产物查看器')}</div><div class="note info mt-12"><div class="ico">${icon('file', 16)}</div><div class="body"><span class="t">${t('Loading artifact', '正在加载产物')}</span><span class="d">${esc(agArtifact.name || '')}</span></div></div></div>`;
    }
    if (agArtifact.error) {
      return `<div class="card pad mt-16"><div class="eyebrow">${t('Artifact viewer', '产物查看器')}</div><div class="note warn mt-12"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('Artifact load failed', '产物加载失败')}</span><span class="d">${esc(agArtifact.error)}</span></div></div></div>`;
    }
    if (!agArtifact.data) return '';
    const data = agArtifact.data || {};
    const artifact = data.artifact || {};
    const payload = JSON.stringify(data.payload || {}, null, 2);
    const scan = data.privacy_scan || {};
    return `
    <div class="card pad mt-16">
      <div class="row" style="justify-content:space-between;align-items:baseline;margin-bottom:10px;">
        <div><div class="eyebrow">${t('Artifact viewer', '产物查看器')}</div><div class="panel-title" style="font-size:14px;margin-top:4px;">${esc(artifact.name || agArtifact.name || 'artifact')}</div></div>
        <button class="btn sm" data-ag-artifact-download="${esc(artifact.name || agArtifact.name || '')}">${icon('download', 13)} ${t('Download JSON', '下载 JSON')}</button>
      </div>
      <div class="row wrap gap-8">
        <span class="pill ${scan.passed ? 'ok' : 'bad'}" style="height:22px;"><span class="dot"></span>${scan.passed ? t('privacy scan clean', '隐私扫描干净') : t('privacy markers found', '发现隐私标记')}</span>
        <span class="pill" style="height:22px;"><span class="dot"></span>${esc((artifact.sha256 || '').slice(0, 12))}</span>
        <span class="pill" style="height:22px;"><span class="dot"></span>${Number(artifact.bytes || 0).toLocaleString()} B</span>
      </div>
      <pre class="mono" style="margin-top:12px;max-height:360px;overflow:auto;background:var(--surface-2);border:1px solid var(--hair);border-radius:8px;padding:12px;font-size:11px;line-height:1.5;white-space:pre-wrap;">${esc(payload.slice(0, 12000))}${payload.length > 12000 ? '\n...' : ''}</pre>
    </div>`;
  }
  function tabOutputs() {
    const s = study();
    const live = liveRunForStudy();
    if (live) {
      requestLiveReview(live);
      const review = currentLiveReview(live);
      const artifacts = artifactsForLive(live);
      const integrity = review && review.signoff_integrity ? review.signoff_integrity : null;
      const loadingReview = agReview.projectDir === live.project_dir && agReview.loading;
      return `
      <div class="row" style="justify-content:space-between;align-items:baseline;margin-bottom:14px;">
        <div><div class="panel-title" style="font-size:15px;">${t('Outputs', '产出物')}</div><div class="panel-sub">${t('Real local artifacts read from', '真实本地产物读取自')} <span class="mono">${esc(live.project_dir || '')}</span></div></div>
        <div class="row gap-8">
          <span class="pill ${review && review.signoff_stale ? 'bad' : 'warn'}" style="height:22px;"><span class="dot"></span>${esc(review && review.signoff_stale ? 'signoff_stale' : (live.gate && live.gate.status ? live.gate.status : 'analysis_only'))}</span>
          ${artifacts.length ? `<button class="btn sm" data-ag-bundle-download>${icon('download', 13)} ${t('Download bundle', '下载打包')}</button>` : ''}
        </div>
      </div>
      ${loadingReview && !artifacts.length ? `<div class="note info mt-12"><div class="ico">${icon('file', 16)}</div><div class="body"><span class="t">${t('Loading local artifacts', '正在加载本地产物')}</span><span class="d">${t('Reading the whitelisted run folder before showing any output cards.', '先读取白名单运行文件夹,再展示产物卡片。')}</span></div></div>` : ''}
      ${artifacts.length ? `
        <div class="outgrid">
          ${artifacts.map((a, i) => {
            const name = a.name || a.relative_path || '';
            const kind = artifactKind(name);
            return `
            <button class="outcard" data-ag-artifact-view="${esc(name)}" type="button">
              <div class="outthumb">${thumb(kind)}</div>
              <div class="outmeta"><div class="od" style="font-size:10px;">${String(i + 1).padStart(2, '0')} · ${esc(a.kind || 'json')}</div><div class="ot">${esc(artifactTitle(name))}</div><div class="od mono">${esc(name)}</div><div class="od mono">${esc((a.sha256 || '').slice(0, 12))}${a.bytes != null ? ' · ' + Number(a.bytes || 0).toLocaleString() + ' B' : ''}</div></div>
            </button>`;
          }).join('')}
        </div>` : (!loadingReview ? `
        <div class="state-hero empty-state">
          <div class="glyph">${icon('file', 28)}</div>
          <div class="st-t">${t('No real output artifacts yet', '还没有真实产物')}</div>
          <div class="st-d">${t('This project has not produced Table 1, missingness, ROC, calibration, or evidence files yet. Run the analysis or open a reviewed local run; placeholders are not shown in Real mode.', '这个项目还没有生成 Table 1、缺失审计、ROC、校准或证据文件。请先运行分析,或打开已有本地运行；真实模式不会显示占位产物。')}</div>
          <div class="st-actions">
            <button class="btn primary" data-ag-tab="overview">${icon('play', 14)} ${t('Run analysis', '运行分析')}</button>
            <button class="btn" data-ag-tab="runs">${icon('history', 14)} ${t('Open Runs', '打开运行历史')}</button>
          </div>
        </div>` : '')}
      ${integrity && integrity.status !== 'unsigned' ? `<div class="note ${integrity.signoff_stale ? 'warn' : 'ok'} mt-16"><div class="ico">${icon(integrity.signoff_stale ? 'alert' : 'shield', 16)}</div><div class="body"><span class="t">${integrity.signoff_stale ? t('Sign-off is stale', '签署已失效') : t('Sign-off hashes verified', '签署哈希已验证')}</span><span class="d" style="display:block;margin-top:2px;">${integrity.signoff_stale ? esc((integrity.tampered_artifacts || []).map(x => x.name).concat(integrity.missing_artifacts || []).join(', ') || integrity.reason || 'artifact hash mismatch') : t('Signed artifact hashes match current files.', '已签署产物哈希与当前文件一致。')}</span></div></div>` : ''}
      <div class="note info mt-16"><div class="ico">${icon('shield', 16)}</div><div class="body"><span class="t">${t('Evidence-bound preflight.', '证据绑定预检。')}</span> <span class="d" style="display:inline;">${t('The run used the active export snapshot, wrote bounded local artifacts, and kept the manuscript draft locked until human sign-off.', '本次运行使用 active export snapshot,写入有界本地产物,并在人工签署前保持论文草稿锁定。')}</span></div></div>
      ${artifactViewer(live)}`;
    }
    return `
      <div class="row" style="justify-content:space-between;align-items:baseline;margin-bottom:14px;">
        <div><div class="panel-title" style="font-size:15px;">${t('Outputs', '产出物')}</div><div class="panel-sub">${t('No local run has been opened for this project.', '这个项目尚未打开本地运行。')}</div></div>
      </div>
      <div class="state-hero empty-state">
        <div class="glyph">${icon('file', 28)}</div>
        <div class="st-t">${t('No real output artifacts yet', '还没有真实产物')}</div>
        <div class="st-d">${t('Outputs are generated only by a local Agent run. This panel will list real JSON/CSV/PNG/HTML files from the run folder and let you open or download them. It will not show seeded Table 1, missingness, ROC, or calibration placeholders.', '产物只来自本地 Agent 运行。这里会列出运行文件夹里的真实 JSON/CSV/PNG/HTML 文件,并允许打开或下载；不会显示种子 Table 1、缺失审计、ROC 或校准占位卡片。')}</div>
        <div class="st-actions">
          <button class="btn primary" data-ag-tab="overview">${icon('play', 14)} ${t('Run analysis', '运行分析')}</button>
          <button class="btn" data-ag-tab="runs">${icon('history', 14)} ${t('Open Runs', '打开运行历史')}</button>
        </div>
      </div>`;
  }

  function tabNotes() {
    const s = study();
    return `
      <div class="card pad">
        <div class="eyebrow">${t('Feasibility verdict', '可行性结论')}</div>
        <div class="panel-title" style="margin-top:4px;">${t('Worth a full study — with caveats', '值得开展完整研究 —— 但有前提')}</div>
        <div class="m-bubble mt-12" style="background:var(--surface-2);border:1px solid var(--hair);font-size:12.75px;line-height:1.6;padding:13px 15px;border-radius:var(--r-3);">
          ${t('First-6h lactate clearance is recorded for 78% of the demo cohort and shows a visible spread by outcome. Coverage is adequate to power a real study, but AUMCdb lacks early lactate timestamps — scope to MIMIC-IV + eICU first.', '前 6 小时乳酸清除率在演示队列中有 78% 有记录,且按结局呈现可见的分布差异。覆盖率足以支撑一项真实研究,但 AUMCdb 缺少早期乳酸时间戳 —— 建议先限定 MIMIC-IV + eICU。')}
        </div>
        <div class="row gap-8 mt-16">
          <button class="btn primary" data-ag-promote>${icon('agent', 13)} ${t('Promote to Analysis run', '升级为分析运行')}</button>
          <button class="btn">${icon('download', 13)} ${t('Export notes', '导出笔记')}</button>
        </div>
      </div>`;
  }

  function tabDraft() {
    const s = study();
    const live = liveRunForStudy();
    if (live) {
      requestLiveReview(live);
      const gate = live.gate || {};
      const checks = Array.isArray(gate.checks) ? gate.checks : [];
      const passed = checks.filter(c => c.passed).length;
      const review = currentLiveReview(live);
      const readiness = review && review.readiness ? review.readiness : null;
      const signed = !!(review && review.signed);
      const artifacts = review && Array.isArray(review.artifacts) ? review.artifacts : (live.artifacts || []);
      const reviewStatus = readiness ? readiness.status : (gate.status || 'analysis_only');
      const draft = review && review.artifact_payloads ? review.artifact_payloads['manuscript_draft.json'] : null;
      const draftClaims = draft && Array.isArray(draft.claims) ? draft.claims : [];
      const failures = readiness && Array.isArray(readiness.non_human_failures) ? readiness.non_human_failures : [];
      const integrity = review && review.signoff_integrity ? review.signoff_integrity : null;
      const required = [
        ['evidence_reviewed', t('I reviewed the evidence artifacts', '我已审阅证据产物')],
        ['claims_remain_locked', t('I confirm claims remain locked / not reportable', '我确认论断仍保持锁定 / 不可报告')],
        ['no_patient_rows_persisted', t('I confirm no patient rows are persisted', '我确认未持久化患者行')],
      ];
      return `
      <div class="split-320" style="grid-template-columns:1fr 300px;">
        <div class="card pad">
          <div class="eyebrow">${t('Evidence gate', '证据闸')}</div>
          <div class="panel-title" style="margin-top:4px;">${signed ? t('Local sign-off recorded · draft locked', '本地签署已记录 · 草稿保持锁定') : t('Preflight complete · draft locked', '预检完成 · 草稿保持锁定')}</div>
          <div class="panel-sub">${t('This real run is analysis_only. It wrote bounded local evidence artifacts; human sign-off records review but does not make the draft reportable.', '这次真实运行是 analysis_only。它写入有界本地证据产物；人工签署只记录审阅,不会让草稿可报告。')}</div>
          <div class="checks2 mt-16">
            ${checks.map((c, i) => `
              <div class="chk ${c.passed ? 'ok' : 'pending'}">
                <span class="cmk">${c.passed ? icon('check', 12, 2.8) : icon('clock', 12)}</span>
                <span style="color:${c.passed ? 'var(--ink)' : 'var(--ink-3)'};font-weight:${c.passed ? 500 : 400};">${esc(c.label || c.id || ('check ' + (i + 1)))}</span>
                <span class="cstate">${c.passed ? t('passed', '通过') : t('pending', '待定')}</span>
              </div>`).join('')}
          </div>
          ${agReview.projectDir === live.project_dir && agReview.error ? `<div class="note warn mt-16"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('Review load failed', '审阅状态加载失败')}</span><span class="d">${esc(agReview.error)}</span></div></div>` : ''}
          ${agReview.projectDir === live.project_dir && agReview.loading ? `<div class="note info mt-16"><div class="ico">${icon('shield', 16)}</div><div class="body"><span class="t">${t('Loading local review bundle', '正在加载本地审阅包')}</span><span class="d">${t('Reading whitelisted JSON artifacts only.', '仅读取白名单 JSON 产物。')}</span></div></div>` : ''}
          ${readiness ? `
          <div class="nextbar mt-16 gate" style="background:var(--surface-2);">
            <span class="pill ${signed ? 'ok' : (failures.length ? 'warn' : 'info')}"><span class="dot"></span>${esc(reviewStatus)}</span>
            <div class="grow"><div class="nb-t">${signed ? t('Human review artifact written', '人工审阅产物已写入') : (readiness.signable ? t('Ready for local human sign-off', '可进行本地人工签署') : t('Blocked before sign-off', '签署前已阻断'))}</div><div class="nb-d">${t('Reportable remains false and draft_unlocked remains false in this stage.', '当前阶段 reportable 仍为 false,draft_unlocked 仍为 false。')}</div></div>
          </div>` : `
          <div class="nextbar mt-16 gate" style="background:var(--surface-2);">
            <span class="pill warn"><span class="dot"></span>${esc(gate.status || 'analysis_only')}</span>
            <div class="grow"><div class="nb-t">${t('Manuscript claims remain locked', '论文论断保持锁定')}</div><div class="nb-d">${t('A full agent run must pass its own evidence gate before any draft can unlock.', '只有完整 agent run 通过自己的证据闸后,草稿才可解锁。')}</div></div>
          </div>`}
          ${readiness && failures.length ? `<div class="note warn mt-16"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('Non-human gate failures', '非人工闸失败项')}</span><span class="d">${esc(failures.join(', '))}</span></div></div>` : ''}
          ${readiness && readiness.signable ? `<div class="ev-detail mt-16">
            <div style="font-weight:600;font-size:12.25px;color:var(--ink);margin-bottom:3px;">${t('Local sign-off confirmations', '本地签署确认项')}</div>
            <div style="font-size:11.5px;color:var(--ink-3);margin-bottom:10px;">${t('This writes human_signoff.json only; it does not unlock a manuscript draft.', '这只会写入 human_signoff.json；不会解锁论文草稿。')}</div>
            <div class="review-todo">
              ${required.map(([id, label]) => `<label class="rtodo-row"><input type="checkbox" data-ag-live-confirm="${id}" /><span class="rtodo-t">${label}</span><span class="rtodo-ref mono">${esc(id)}</span></label>`).join('')}
            </div>
            <div class="row gap-8 mt-12">
              <button class="btn primary sm" data-ag-live-signoff aria-disabled="true">${icon('check', 12)} ${t('Write local sign-off', '写入本地签署')}</button>
              <span class="rtodo-hint" style="font-size:11px;color:var(--ink-4);align-self:center;">${t('Check all 3 to enable', '勾选全部 3 项后可签署')}</span>
            </div>
          </div>` : ''}
          ${signed && integrity && integrity.signoff_stale ? `<div class="note warn mt-16"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('Sign-off is stale', '签署已失效')}</span><span class="d" style="display:block;margin-top:2px;">${esc((integrity.tampered_artifacts || []).map(x => x.name).concat(integrity.missing_artifacts || []).join(', ') || integrity.reason || 'artifact hash mismatch')}</span></div></div>` : ''}
          ${signed && !(integrity && integrity.signoff_stale) ? `<div class="note ok mt-16"><div class="ico">${icon('check', 16)}</div><div class="body"><span class="t">human_signoff.json</span><span class="d" style="display:block;margin-top:2px;">${t('Signed analysis-only review recorded locally; manuscript claims remain locked.', 'analysis-only 审阅签署已在本地记录；论文论断仍保持锁定。')}</span></div></div>` : ''}
          ${draftClaims.length ? `<div class="ev-detail mt-16">
            <div style="font-weight:600;font-size:12.25px;color:var(--ink);margin-bottom:8px;">${t('Evidence-bound draft claims', '证据绑定草稿论断')}</div>
            ${draftClaims.map(row => `<div style="font-size:12px;line-height:1.55;margin-bottom:8px;"><span class="mono" style="color:var(--ink-4);">${esc(row.claim_id || 'claim')}</span> ${esc(row.text || '')}<div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc((row.evidence_ids || []).join(', '))}</div></div>`).join('')}
          </div>` : ''}
        </div>
        <div class="card pad" style="align-self:start;">
          <div class="eyebrow">${t('Local artifacts', '本地产物')}</div>
          <div class="col gap-8 mt-12">
            ${artifacts.map(a => `
              <div class="ledger-row"><span class="ledger-ico">${icon((a.name || '').includes('signoff') ? 'check' : 'shield', 14)}</span><div><div style="font-weight:600;font-size:12.5px;">${esc(a.name || a.relative_path || 'artifact')}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc((a.sha256 || '').slice(0, 12))}${a.bytes != null ? ' · ' + Number(a.bytes || 0).toLocaleString() + ' B' : ''}</div></div></div>`).join('')}
          </div>
          <div class="mt-16 mono" style="font-size:11px;color:var(--ink-4);">${passed}/${checks.length} ${t('checks passed', '项校验通过')}</div>
          <div class="mt-8 mono" style="font-size:10.5px;color:var(--ink-4);">${esc(live.project_dir || '')}</div>
        </div>
      </div>`;
    }
    const checks = [
      [t('Cohort denominators resolved', '队列分母已确定'), true, '01 · ' + t('Cohort summary', '队列摘要'), t('n and outcome rates computed from the frozen cohort frame.', 'n 与结局率由冻结的队列帧计算得出。')],
      [t('Per-concept coverage ≥ threshold', '各概念覆盖率 ≥ 阈值'), true, '03 · ' + t('Missingness', '缺失审计'), t('Per-concept coverage table — every module clears the threshold.', '各概念覆盖率表 —— 每个模块均达阈值。')],
      [t('Table 1 reproduces from manifest', 'Table 1 可从清单复现'), true, '02 · Table 1', t('Re-generated row-for-row from the run manifest seed.', '依据运行清单种子逐行重新生成。')],
      [t('Model card + metrics attached', '模型卡 + 指标已附'), true, '04 · ROC', t('AUC, calibration and a model card are bound to the run.', 'AUC、校准与模型卡已绑定到本次运行。')],
      [t('Reviewer sign-off', '审阅者签署'), s.signed, null, t('Awaiting a human reviewer signature.', '等待人工审阅者签署。')],
    ];
    const passed = checks.filter(c => c[1]).length;
    return `
    <div class="split-320" style="grid-template-columns:1fr 300px;">
      <div class="col gap-16">
        <div class="card pad">
          <div class="eyebrow">${t('Evidence gate', '证据闸')}</div>
          <div class="panel-title" style="margin-top:4px;">${s.signed ? t('Manuscript draft unlocked', '论文草稿已解锁') : t('Manuscript draft is locked until checks pass', '在校验通过前论文草稿保持锁定')}</div>
          <div class="panel-sub">${t('Every claim traces to a logged artifact — tap a check to see its evidence.', '每个论断都可追溯到已记录的产物 —— 点一项校验即可查看其证据。')}</div>
          <div class="checks2 mt-16">
            ${checks.map(([ti, ok, art, ev], i) => {
              const isReview = !ok && i === checks.length - 1;
              const clickable = art || isReview;
              return `
              <div class="chk ${ok ? 'ok' : 'pending'} ${clickable ? 'linked' : ''}" ${clickable ? `data-ev="${i}"` : ''}>
                <span class="cmk">${ok ? icon('check', 12, 2.8) : icon('clock', 12)}</span>
                <span style="color:${ok ? 'var(--ink)' : 'var(--ink-3)'};font-weight:${ok ? 500 : 400};">${ti}</span>
                ${art ? `<span class="cev">${icon('eye', 11)} ${agEvOpen === i ? t('hide', '收起') : t('evidence', '证据')}</span>` : ''}
                ${isReview ? `<span class="cev">${icon('arrow', 11)} ${agEvOpen === i ? t('hide', '收起') : t('review now', '立即审阅')}</span>` : ''}
                <span class="cstate">${ok ? t('passed', '通过') : t('pending', '待定')}</span>
              </div>
              ${art && agEvOpen === i ? `<div class="ev-detail"><span class="ev-art">${icon('file', 12)} ${art}</span><div style="margin-top:5px;">${ev}</div><button class="btn sm mt-8" data-ag-tab="outputs">${icon('arrow', 12)} ${t('Open in Outputs', '在产出中打开')}</button></div>` : ''}
              ${isReview && agEvOpen === i ? `<div class="ev-detail">
                <div style="font-weight:600;font-size:12.25px;color:var(--ink);margin-bottom:3px;">${t('What you are confirming', '你将确认以下事项')}</div>
                <div style="font-size:11.5px;color:var(--ink-3);margin-bottom:10px;">${t('Sign-off is the last gate. Confirm each point, then the draft unlocks.', '签署是最后一道闸。逐项确认后，草稿即解锁。')}</div>
                <div class="review-todo">
                  ${[
                    [t('Findings match the logged tables & figures', '结论与已记录的表格和图一致'), '02 · Table 1'],
                    [t('No claim exceeds what the evidence supports', '没有超出证据支撑的论断'), '04 · ROC'],
                    [t('Limitations & cohort caveats are stated', '已说明局限性与队列注意事项'), '03 · ' + t('Missingness', '缺失审计')],
                  ].map(([rt, ref]) => `<label class="rtodo-row"><input type="checkbox" data-ag-rtodo /><span class="rtodo-t">${rt}</span><span class="rtodo-ref mono">${ref}</span></label>`).join('')}
                </div>
                <div class="row gap-8 mt-12">
                  <button class="btn primary sm" data-ag-signoff aria-disabled="true">${icon('check', 12)} ${t('Confirm all & sign off', '全部确认并签署')}</button>
                  <span class="rtodo-hint" style="font-size:11px;color:var(--ink-4);align-self:center;">${t('Check all 3 to enable', '勾选全部 3 项后可签署')}</span>
                </div>
              </div>` : ''}`;
            }).join('')}
          </div>
          <div class="nextbar mt-16 ${s.signed ? '' : 'gate'}" style="background:var(--surface-2);">
            <span class="pill ${passed === checks.length ? 'ok' : 'warn'}"><span class="dot"></span>${passed} / ${checks.length} ${t('checks', '校验')}</span>
            <div class="grow"><div class="nb-t">${s.signed ? t('Draft ready', '草稿就绪') : t('One reviewer sign-off outstanding', '还差一位审阅者签署')}</div><div class="nb-d">${s.signed ? t('Methods + results drafted from logged evidence.', '方法 + 结果基于已记录证据撰写。') : t('The draft button unlocks once a reviewer confirms the findings.', '审阅者确认结论后,撰稿按钮即解锁。')}</div></div>
            ${s.signed
              ? `<button class="btn primary">${icon('wand', 13)} ${t('Open manuscript', '打开论文')}</button>`
              : `<button class="btn" data-ag-decline>${t('Decline', '退回')}</button><button class="btn primary" data-ag-signoff>${icon('check', 13)} ${t('Sign off & draft', '签署并撰稿')}</button>`}
          </div>
        </div>
        ${s.signed ? `
        <div class="card pad">
          <div class="eyebrow">${t('Draft versions', '草稿版本')}</div>
          <div class="mt-12">
            <div class="draftver"><span class="dv-badge">v0.2</span><div class="grow"><div style="font-weight:600;font-size:12.75px;">${t('Methods + Results + Limitations', '方法 + 结果 + 局限')}</div><div style="font-size:11px;color:var(--ink-4);">${t('signed off · today 14:40', '已签署 · 今天 14:40')}</div></div><button class="btn sm">${icon('eye', 13)} ${t('Open', '打开')}</button></div>
            <div class="draftver"><span class="dv-badge">v0.1</span><div class="grow"><div style="font-weight:600;font-size:12.75px;">${t('Methods + Results', '方法 + 结果')}</div><div style="font-size:11px;color:var(--ink-4);">${t('auto-draft · today 14:38', '自动草稿 · 今天 14:38')}</div></div><button class="btn sm ghost">${icon('history', 13)} ${t('Diff', '对比')}</button></div>
          </div>
        </div>` : ''}
      </div>
      <div class="card pad" style="align-self:start;">
        <div class="eyebrow">${t('Output bundle', '产出打包')}</div>
        <div class="col gap-8 mt-12">
          ${[['6 ' + t('figures', '张图'), 'png + svg', 'viz'], ['3 ' + t('tables', '个表'), 'csv + tex', 'list'], [t('Evidence ledger', '证据账本'), 'json manifest', 'shield'], [t('Repro code', '复现代码'), 'py + notebook', 'file']].map(([ti, d, ic]) => `
            <div class="ledger-row"><span class="ledger-ico">${icon(ic, 14)}</span><div><div style="font-weight:600;font-size:12.5px;">${ti}</div><div style="font-size:11px;color:var(--ink-4);">${d}</div></div></div>`).join('')}
        </div>
        <button class="btn sm block mt-16">${icon('download', 13)} ${t('Export bundle', '导出打包')}</button>
      </div>
    </div>`;
  }

  function tabBody() {
    const s = study();
    if (agTab === 'runs') return tabRuns();
    if (agTab === 'outputs') return tabOutputs();
    if (agTab === 'notes') return tabNotes();
    if (agTab === 'draft') return tabDraft();
    return tabOverview();
  }

  function agShell() {
    return `
    <div class="ag-wrap">
      ${studyList()}
      <div class="ag-detail">
        ${detailHead()}
        ${tabsRow()}
        <div class="ag-body">${tabBody()}</div>
      </div>
    </div>`;
  }

  /* ---------------- run animation ---------------- */
  function startRun() {
    const s = study();
    if (s.empty) {
      agRun.error = t('No local research project is selected. Create an Agent project seed from Idea Mining first.', '尚未选择本地研究项目。请先从 Idea Mining 创建 Agent project seed。');
      repaintBody();
      return;
    }
    const src = activeExportSource();
    agRun.error = null;
    if (src && window.EU_API && window.EU_API.startAgentRun && window.EventSource) {
      startRealRun(src);
      return;
    }
    if (realMode()) {
      agRun.error = t('No active registered export is selected. Register/select a local export before running this project.', '尚未选择 active registered export。请先注册/选择本地导出后再运行该项目。');
      repaintBody();
      return;
    }
    startDemoRun();
  }

  function startRealRun(src) {
    const opts = arguments.length > 1 && arguments[1] ? arguments[1] : {};
    const s = study();
    if (s.empty) {
      finishRealRun(s, 'failed', null, t('No local research project is selected. Create an Agent project seed from Idea Mining first.', '尚未选择本地研究项目。请先从 Idea Mining 创建 Agent project seed。'));
      return;
    }
    closeRunStream();
    agRun = { active: true, prog: 0, timer: null, es: null, jobId: null, step: t('Submitting local run', '提交本地运行'), error: null, result: null };
    agReview = { projectDir: null, loading: false, error: null, data: null, signing: false };
    agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
    agHistory = { studyId: null, loading: false, error: null, data: null };
    window.EU_AGENT_RUN_REVIEW = null;
    agTab = 'overview';
    window.EU_STALE = false;  // a fresh run consumes the current inputs
    repaintBody();
    window.EU_API.startAgentRun({
      path: src.path,
      study_id: s.id,
      mode: s.mode,
      run_type: opts.runType || 'preflight',
      llm_provider: opts.provider || 'mock',
      external_llm_opt_in: !!opts.externalOptIn,
      question: s.question && s.question[0],
    }).then(r => {
      agRun.jobId = r.job_id;
      agRun.step = t('Connected to job stream', '已连接任务流');
      rememberAgentJob({
        job_id: r.job_id,
        study_id: s.id,
        source_path: src.path,
        run_type: opts.runType || 'preflight',
        provider: opts.provider || 'mock',
        external_llm_opt_in: !!opts.externalOptIn,
      });
      attachAgentJobStream(r.job_id, s);
    }).catch(err => finishRealRun(s, 'failed', null, err.message || String(err)));
  }

  function finishRealRun(s, status, result, error) {
    closeRunStream();
    agRun.active = false;
    agRun.prog = 1;
    if (status !== 'running') clearRememberedAgentJob(agRun.jobId);
    if (status === 'done' && result) {
      agRun.result = result;
      agRun.error = null;
      agRun.reconnectable = false;
      window.EU_AGENT_LAST_RUN = result;
      agReview = { projectDir: null, loading: false, error: null, data: null, signing: false };
      agHistory = { studyId: null, loading: false, error: null, data: null };
      window.EU_AGENT_RUN_REVIEW = null;
      if (s.mode === 'analysis') { s.status = 'gate'; s.stage = 3; }
      else { s.status = 'draft'; s.stage = 2; }
      s.runs.unshift([
        result.run_label || ('run ' + (result.run_id || '').slice(-6)),
        result.run_type === 'full'
          ? [t('Provider-gated full scaffold', 'provider 受闸 full 骨架'), t('Provider-gated full scaffold', 'provider 受闸 full 骨架')]
          : [t('Registry-backed preflight', '注册表支持的预检'), t('Registry-backed preflight', '注册表支持的预检')],
        result.gate && result.gate.status === 'blocked' ? 'blocked' : 'complete',
        result.duration_sec != null ? `${result.duration_sec}s` : '—',
        ['just now', '刚刚'],
      ]);
      agTab = 'outputs';
    } else if (status === 'cancelled') {
      agRun.result = result || null;
      agRun.reconnectable = false;
      agRun.error = result && result.cancelled
        ? `${t('Cancelled at', '取消阶段')}: ${result.cancelled_at || 'agent_run'} · ${t('safe continuation is to restart from the active export.', '安全继续方式是从 active export 重跑。')}`
        : (error || t('Agent run cancelled.', 'Agent 运行已取消。'));
      s.status = 'idle';
    } else {
      agRun.error = error || t('Agent run failed.', 'Agent 运行失败。');
      agRun.result = result || null;
      agRun.reconnectable = false;
      s.status = 'idle';
    }
    repaintBody();
  }

  function startDemoRun() {
    const s = study();
    closeRunStream();
    agRun.active = true; agRun.prog = 0; agRun.step = null; agRun.error = null; agRun.result = null; agRun.jobId = null;
    agTab = 'overview';
    window.EU_STALE = false;  // a fresh run consumes the current inputs
    repaintBody();
    agRun.timer = setInterval(() => {
      agRun.prog = Math.min(1, agRun.prog + 0.012 + Math.random() * 0.01);
      if (agRun.prog >= 1) {
        agRun.prog = 1; agRun.active = false; clearInterval(agRun.timer); agRun.timer = null;
        // landing state
        if (s.mode === 'analysis') { s.status = 'gate'; s.stage = 3; }
        else { s.status = 'draft'; s.stage = 2; }
        repaintBody();
        return;
      }
      const bar = document.querySelector('#agHost .runbar-fill');
      if (bar) bar.style.width = Math.round(agRun.prog * 100) + '%';
    }, 110);
  }

  /* ---------------- wiring ---------------- */
  function wire(root) {
    const host = root.querySelector('#agHost'); if (!host) return;
    host.querySelectorAll('[data-ag-sel]').forEach(b => b.addEventListener('click', () => {
      closeRunStream(); agRun.active = false;
      agSel = b.dataset.agSel; agTab = 'overview'; repaintBody();
    }));
    host.querySelectorAll('[data-ag-tab]').forEach(b => b.addEventListener('click', () => { agTab = b.dataset.agTab; repaintBody(); }));
    host.querySelectorAll('[data-ev]').forEach(b => b.addEventListener('click', () => { const i = +b.dataset.ev; agEvOpen = (agEvOpen === i ? -1 : i); repaintBody(); }));
    host.querySelectorAll('[data-ag-runbtn]').forEach(b => b.addEventListener('click', startRun));
    host.querySelectorAll('[data-ag-cancel-job]').forEach(b => b.addEventListener('click', () => {
      if (!agRun.jobId || !window.EU_API || !window.EU_API.cancelJob) return;
      b.setAttribute('disabled', 'true');
      agRun.step = t('Cancel requested. The current local phase may finish before the job stops.', '已请求取消。当前本地阶段可能会先完成,然后任务停止。');
      repaintBody();
      window.EU_API.cancelJob(agRun.jobId, 'user_requested').catch(err => {
        agRun.error = err.message || String(err);
        agRun.active = false;
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-reconnect]').forEach(b => b.addEventListener('click', () => {
      if (!agRun.jobId || !window.EU_API || !window.EU_API.loadJobSnapshot) return;
      const jobId = agRun.jobId;
      agRun.error = null;
      agRun.active = true;
      agRun.step = t('Checking server job state', '正在检查服务端任务状态');
      repaintBody();
      window.EU_API.loadJobSnapshot(jobId).then(snapshot => {
        restoreAgentJobFromSnapshot({ job_id: jobId, study_id: study().id }, snapshot);
      }).catch(err => {
        agRun.active = false;
        agRun.error = err.message || String(err);
        clearRememberedAgentJob(jobId);
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-provider-refresh]').forEach(b => b.addEventListener('click', () => requestProviderStatus(true)));
    host.querySelectorAll('[data-ag-provider]').forEach(b => b.addEventListener('click', () => {
      agProvider = { provider: b.dataset.agProvider || 'openai', consent: false, loading: false, error: null, status: null };
      requestProviderStatus(true);
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-external-consent]').forEach(c => c.addEventListener('change', () => {
      agProvider = Object.assign({}, agProvider, { consent: !!c.checked });
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-external-run]').forEach(b => b.addEventListener('click', () => {
      if (b.getAttribute('aria-disabled') === 'true') return;
      const src = activeExportSource();
      if (!src) return;
      startRealRun(src, { runType: 'full', provider: agProvider.provider, externalOptIn: true });
    }));
    host.querySelectorAll('[data-ag-history-refresh]').forEach(b => b.addEventListener('click', () => requestRunHistory(true)));
    host.querySelectorAll('[data-ag-refresh-projects]').forEach(b => b.addEventListener('click', () => requestIdeaAgentProjects(true)));
    host.querySelectorAll('[data-ag-history-open]').forEach(b => b.addEventListener('click', () => {
      const rows = agHistory.data && Array.isArray(agHistory.data.runs) ? agHistory.data.runs : [];
      const row = rows[Number(b.dataset.agHistoryOpen || -1)];
      if (!row || !row.project_dir || !window.EU_API || !window.EU_API.loadAgentRunReview) return;
      agReview = { projectDir: row.project_dir, loading: true, error: null, data: null, signing: false };
      agTab = 'draft';
      repaintBody();
      window.EU_API.loadAgentRunReview(row.project_dir).then(data => {
        openReview(data);
        repaintBody();
      }).catch(err => {
        agReview = { projectDir: row.project_dir, loading: false, error: err.message || String(err), data: null, signing: false };
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-artifact-view]').forEach(card => card.addEventListener('click', () => {
      const live = liveRunForStudy();
      requestArtifact(live, card.dataset.agArtifactView);
    }));
    host.querySelectorAll('[data-ag-artifact-download]').forEach(b => b.addEventListener('click', e => {
      e.stopPropagation();
      const live = liveRunForStudy();
      if (!live || !window.EU_API || !window.EU_API.downloadAgentRunArtifact) return;
      window.EU_API.downloadAgentRunArtifact(live.project_dir, b.dataset.agArtifactDownload).catch(err => {
        agArtifact = { projectDir: live.project_dir, name: b.dataset.agArtifactDownload, loading: false, error: err.message || String(err), data: agArtifact.data };
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-bundle-download]').forEach(b => b.addEventListener('click', () => {
      const live = liveRunForStudy();
      if (!live || !window.EU_API || !window.EU_API.downloadAgentRunBundle) return;
      window.EU_API.downloadAgentRunBundle(live.project_dir).catch(err => {
        agArtifact = { projectDir: live.project_dir, name: 'bundle', loading: false, error: err.message || String(err), data: null };
        repaintBody();
      });
    }));
    const liveConfirms = [...host.querySelectorAll('[data-ag-live-confirm]')];
    if (liveConfirms.length) {
      const liveBtn = host.querySelector('[data-ag-live-signoff]');
      const hint = host.querySelector('.rtodo-hint');
      const syncLive = () => {
        const all = liveConfirms.every(c => c.checked);
        if (liveBtn) liveBtn.setAttribute('aria-disabled', all ? 'false' : 'true');
        if (hint) hint.style.display = all ? 'none' : '';
      };
      liveConfirms.forEach(c => c.addEventListener('change', syncLive));
      syncLive();
    }
    host.querySelectorAll('[data-ag-live-signoff]').forEach(b => b.addEventListener('click', () => {
      if (b.getAttribute('aria-disabled') === 'true') return;
      const live = liveRunForStudy();
      if (!live || !live.project_dir || !window.EU_API || !window.EU_API.signoffAgentRun) return;
      const confirmations = [...host.querySelectorAll('[data-ag-live-confirm]')]
        .filter(c => c.checked)
        .map(c => c.dataset.agLiveConfirm);
      b.setAttribute('aria-disabled', 'true');
      window.EU_API.signoffAgentRun(live.project_dir, {
        reviewer: 'local_reviewer',
        confirmations: confirmations,
        note: 'Reviewed from EasyICU local WebApp',
      }).then(data => {
        agReview = { projectDir: live.project_dir, loading: false, error: null, data: data, signing: false };
        agHistory = { studyId: null, loading: false, error: null, data: null };
        window.EU_AGENT_RUN_REVIEW = data;
        repaintBody();
      }).catch(err => {
        agReview = { projectDir: live.project_dir, loading: false, error: err.message || String(err), data: currentLiveReview(live), signing: false };
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-signoff]').forEach(b => b.addEventListener('click', () => {
      if (b.getAttribute('aria-disabled') === 'true') return;
      const live = liveRunForStudy();
      if (live && !(live.gate && live.gate.draft_unlocked)) {
        agTab = 'draft';
        repaintBody();
        return;
      }
      study().signed = true; study().status = 'ready'; study().stage = 4; repaintBody();
    }));
    const rtodos = [...host.querySelectorAll('[data-ag-rtodo]')];
    if (rtodos.length) {
      const gated = host.querySelector('.ev-detail [data-ag-signoff]');
      const hint = host.querySelector('.rtodo-hint');
      const sync = () => {
        const all = rtodos.every(c => c.checked);
        if (gated) gated.setAttribute('aria-disabled', all ? 'false' : 'true');
        if (hint) hint.style.display = all ? 'none' : '';
      };
      rtodos.forEach(c => c.addEventListener('change', sync));
      sync();
    }
    host.querySelectorAll('[data-ag-promote]').forEach(b => b.addEventListener('click', () => { study().mode = 'analysis'; study().status = 'idle'; study().stage = 0; agTab = 'overview'; repaintBody(); }));
    const newBtn = host.querySelector('[data-ag-new]');
    if (newBtn) newBtn.addEventListener('click', () => { location.hash = '#ideas'; });
  }

  S.agent = {
    section: 'agent', nav: 'agent',
    wide: true,
    get crumbs() { return [t('Home', '首页'), t('Agent Projects', '研究项目')]; },
    get actionHtml() { return `<button class="btn">${icon('help', 13)} ${t('Agent guide', '代理指南')}</button>`; },
    rail() {
      const s = study();
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Projects', '项目')}</span><span class="pill ok" style="height:20px;"><span class="dot"></span>${allStudies().length}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Active', '当前')}</span><span class="vv" style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${t(s.name[0], s.name[1])}</span></div>
          <div class="setup-row"><span class="k">${t('Mode', '模式')}</span><span class="vv">${t('Analysis', '分析')}</span></div>
          <div class="setup-row"><span class="k">${t('Runs', '运行')}</span><span class="vv">${s.runs.length}</span></div>
        </div>
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Guarantees', '保证')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6">${icon('shield', 13)} ${t('Local-first · no upload', '本地优先 · 不上传')}</div>
          <div class="row gap-6">${icon('lock', 13)} ${t('Draft gated on evidence', '草稿受证据约束')}</div>
          <div class="row gap-6">${icon('check', 13)} ${t('Human confirms each run', '每次运行需人工确认')}</div>
        </div>
      </div>`;
    },
    render() {
      return `
      <div class="page-head" style="margin-bottom:16px;">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:16px;">
          <div>
            <div class="eyebrow">${t('Agent Projects · 研究项目', '研究项目 · Agent Projects')}</div>
            <h1 style="margin-top:6px;">${t('Agent Projects', '研究项目')}</h1>
            <p class="lead">${t('A workspace of research projects. Each study has a workflow, its own runs, outputs, and a gated draft — all auditable, all local.', '一个研究项目工作台。每个研究都有自己的工作流、运行记录、产出和受闸草稿 —— 全程可审计、全程本地。')}</p>
            <div style="font-size:11.5px;color:var(--ink-4);margin-top:9px;">${t('Key terms', '关键术语')}: ${window.gloss('denominator', t('denominator', '分母'))} · ${window.gloss('concept', t('concept', '概念'))} · ${window.gloss('SOFA')}</div>
          </div>
        </div>
      </div>
      <div id="agHost">${agShell()}</div>`;
    },
    afterRender(root) { wire(root); requestIdeaAgentProjects(); maybeRestoreAgentJob(); },
  };
})();
