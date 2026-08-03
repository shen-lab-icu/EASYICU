/* Screen: Research Agent — project workspace (redesigned, bilingual).
   Goal: make the WORKFLOW and PROJECT MANAGEMENT obvious.
     • Left rail = a persistent list of studies (projects), like chat sessions.
     • Each study carries a linked cohort, its own run history, outputs, and
       draft versions. Idea mining lives in the separate #ideas workspace.
     • The pipeline + evidence checks are drawn explicitly per study.
   Outputs fail closed: Real mode lists only whitelisted local artifacts. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  /* Fixture data + pure renderers live in screens-agent-render.js
     (owner-file carve-out; loads before this file). Rebind the names
     so call sites in this IIFE stay unchanged. */
  const R = window.AGENT_RENDER || {};
  const {
    DEMO_STUDIES, BLOCK_FAMILIES, BLOCK_LIBRARY, NATURE_PACK,
    runStatusLabel, runStatusHint, gateCheckLabel, readableArtifactText, firstValue, fmtCount,
    artifactKind, artifactTitle, artifactCategory, artifactSummary, artifactRank, defaultArtifactName,
    thumb, scrubDataUrls, figureGallery, artifactStructuredView,
  } = R;

  let agSel = null;
  let agTab = 'overview';
  let agEvOpen = -1;   // expanded evidence-gate check index
  let agRun = { active: false, prog: 0, timer: null, es: null, jobId: null, step: null, error: null, result: null, warning: null };
  let agReview = { projectDir: null, loading: false, error: null, data: null, signing: false };
  let agHistory = { studyId: null, loading: false, error: null, data: null };
  let agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
  let agProvider = { provider: 'openai', consent: false, loading: false, error: null, status: null };
  let agIdeaProjects = { loading: false, error: null, data: null };
  let agBlockFamily = 'all';
  let agBlockSelected = 'nature_writing';
  let agListMode = 'auto'; // auto, open, or focus
  const AG_JOB_KEY = 'easyicu.agent.activeJob.v1';
  const AG_BLOCKS_VERSION = 'v1';
  let agResumeProbe = { loading: false, checkedJobId: null };
  const AG_FOCUS_TABS = new Set(['science', 'runs', 'outputs', 'notes', 'draft']);
  const agRunChannel = window.EU_AGENT_STUDY_CONTEXT.createRunChannel();
  const agJobMemory = window.EU_AGENT_STUDY_CONTEXT.createJobMemory(localStorage, AG_JOB_KEY);

  /* continuity: Copilot can land a completed run */
  window.__euAgentPreset = function () { agSel = 'sepsis'; agTab = 'outputs'; };

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  }
  function bi(value) {
    return Array.isArray(value) ? t(value[0], value[1]) : esc(value);
  }
  function blockById(id) {
    return BLOCK_LIBRARY.find(b => b.id === id) || null;
  }
  function defaultWorkflowIds(s) {
    if (!s || s.empty) return [];
    if (s.ideaSeed) return ['discovery_literature', 'discovery_feasibility', 'agent_handoff', 'analysis_agent_run', 'evidence_review', 'nature_writing'];
    if (s.id === 'lactate') return ['discovery_feasibility', 'agent_handoff', 'analysis_agent_run', 'evidence_review', 'nature_figure', 'nature_writing'];
    return ['analysis_agent_run', 'evidence_review', 'nature_figure', 'nature_writing'];
  }
  function blockStoreKey(s) {
    return `easyicu.agent.workflowBlocks.${AG_BLOCKS_VERSION}.${s && s.id ? s.id : 'none'}`;
  }
  function workflowIds(s) {
    if (!s || s.empty) return [];
    try {
      const raw = localStorage.getItem(blockStoreKey(s));
      const parsed = raw ? JSON.parse(raw) : null;
      if (Array.isArray(parsed)) {
        const clean = parsed.filter(id => !!blockById(id));
        if (clean.length) return clean;
      }
    } catch (_) {}
    return defaultWorkflowIds(s);
  }
  function writeWorkflowIds(s, ids) {
    if (!s || s.empty) return;
    const clean = (ids || []).filter(id => !!blockById(id));
    try { localStorage.setItem(blockStoreKey(s), JSON.stringify(clean)); } catch (_) {}
  }
  function workflowBlocks(s) {
    return workflowIds(s).map(blockById).filter(Boolean);
  }
  function addWorkflowBlock(id) {
    const s = study();
    const ids = workflowIds(s);
    if (!blockById(id) || ids.includes(id)) return;
    ids.push(id);
    writeWorkflowIds(s, ids);
    agBlockSelected = id;
  }
  function addWorkflowPack(ids) {
    const s = study();
    const current = workflowIds(s);
    ids.forEach(id => {
      if (blockById(id) && !current.includes(id)) current.push(id);
    });
    writeWorkflowIds(s, current);
    if (ids.length) agBlockSelected = ids[0];
  }
  function moveWorkflowBlock(index, delta) {
    const s = study();
    const ids = workflowIds(s);
    const next = index + delta;
    if (next < 0 || next >= ids.length) return;
    const [row] = ids.splice(index, 1);
    ids.splice(next, 0, row);
    writeWorkflowIds(s, ids);
    agBlockSelected = row;
  }
  function removeWorkflowBlock(index) {
    const s = study();
    const ids = workflowIds(s);
    const removed = ids.splice(index, 1)[0];
    writeWorkflowIds(s, ids);
    if (agBlockSelected === removed) agBlockSelected = ids[index] || ids[index - 1] || 'nature_writing';
  }
  function resetWorkflowBlocks() {
    const s = study();
    writeWorkflowIds(s, defaultWorkflowIds(s));
    agBlockSelected = workflowIds(s)[0] || 'nature_writing';
  }
  function familyLabel(key) {
    const row = BLOCK_FAMILIES.find(f => f[0] === key);
    return row ? bi(row[1]) : esc(key);
  }
  function blockListItems(items) {
    return (items || []).map(x => `<span>${esc(x)}</span>`).join('');
  }
  function seedStudy(row) {
    const title = row.title || row.question || 'Idea-derived study';
    const q = row.question || title;
    const pre = row.pre_experiment_summary || {};
    const source = row.source || {};
    const seedRuns = Array.isArray(row.runs) ? row.runs : [];
    const reviewRun = seedRuns.find(r => r && r.project_dir) || null;
    const imported = row.seed_kind === 'canonical9_import' || !!row.benchmark;
    return {
      id: row.study_id || row.id || title,
      name: [title, title],
      mode: 'analysis',
      status: row.status === 'seeded_from_idea' ? 'idle' : (row.status || 'idle'),
      stage: Number(row.stage || 0),
      cohort: row.cohort || row.study_id || 'research_idea',
      source: imported
        ? [
          `Fig 2 question · ${pre.status || 'imported'} · ${pre.feature_count || 0} evidence`,
          `Fig 2 问题 · ${pre.status || '已导入'} · ${pre.feature_count || 0} 条证据`,
        ]
        : [
          `research idea · ${pre.status || 'feasibility'} · ${pre.feature_count || 0} features`,
          `研究想法 · ${pre.status || '可行性'} · ${pre.feature_count || 0} 特征`,
        ],
      question: [q, q],
      runs: seedRuns.map((r, i) => [
        r.label || ('package ' + String(i + 1).padStart(2, '0')),
        [r.scope || 'review package', r.scope || '审阅包'],
        r.status || 'complete',
        '—',
        [r.created_at || 'local', r.created_at || '本地'],
      ]),
      signed: false,
      ideaSeed: row,
      projectKind: imported ? 'canonical9' : 'idea',
      seedRuns: seedRuns,
      reviewProjectDir: reviewRun && reviewRun.project_dir,
      benchmark: row.benchmark || null,
      sourceArticle: [source.title, source.journal, source.year].filter(Boolean).join(' · '),
    };
  }
  function seedExecutionGate(s) {
    return s && s.ideaSeed && s.ideaSeed.execution_gate ? s.ideaSeed.execution_gate : null;
  }
  function seedGateBlocksRun(s) {
    const gate = seedExecutionGate(s);
    if (s && s.ideaSeed && !gate) return true;
    return !!(gate && gate.agent_run_ready_after_human_confirmation === false);
  }
  function gateBlockerLabel(raw) {
    const text = String(raw || '');
    if (/refresh Agent project/i.test(text)) return t('Refresh this project from Idea Mining', '回到 Idea Mining 刷新这个项目');
    if (/prior-art/i.test(text)) return t('Run prior-art review before Agent execution', '在 Agent 执行前完成 prior-art 审阅');
    if (/missing required concepts|re-extract/i.test(text)) return t('Re-extract or confirm missing required concepts', '重新抽取或确认缺失的必要概念');
    if (/real EasyICU export|real export|select a real/i.test(text)) return t('Select a real EasyICU export', '选择真实 EasyICU 导出');
    if (/idea feasibility/i.test(text)) return t('Resolve idea feasibility before execution', '先解决 idea 可行性问题');
    if (/same active export/i.test(text)) return t('Select the same active export used by Idea Mining', '选择 Idea Mining 使用的同一个 active export');
    // Plain text — callers escape at their own HTML insertion point.
    // Returning esc() here double-escaped agRun.error, which is escaped again
    // by the run-error nextbar.
    return text;
  }
  function seedGateBlockerText(s) {
    const gate = seedExecutionGate(s);
    if (s && s.ideaSeed && !gate) return t('Refresh this project from Idea Mining so the preflight checks are available.', '请回到 Idea Mining 刷新这个项目，让预检条件可用。');
    const blockers = gate && Array.isArray(gate.blockers) ? gate.blockers : [];
    return blockers.length
      ? blockers.map(gateBlockerLabel).join(' · ')
      : t('Confirm the Idea Mining preflight checks before running Agent preflight.', '运行 Agent 预检前先确认 Idea Mining 预检条件。');
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
      source: [t('No local project', '无本地项目'), t('No local project', '无本地项目')],
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
    const contextStudies = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.projects
      ? window.EU_AGENT_STUDY_CONTEXT.projects()
      : [];
    const seeds = (agIdeaProjects.data && Array.isArray(agIdeaProjects.data.projects) ? agIdeaProjects.data.projects : []).map(seedStudy);
    const contextIds = new Set(contextStudies.map(row => row.id));
    const occupiedIds = new Set(seeds.map(s => s.id));
    contextStudies.forEach(row => occupiedIds.add(row.id));
    const base = realMode() ? [] : DEMO_STUDIES;
    return contextStudies.concat(seeds.filter(s => !contextIds.has(s.id)), base.filter(s => !occupiedIds.has(s.id)));
  }
  function study() { return allStudies().find(s => s.id === agSel) || allStudies()[0] || emptyStudy(); }
  function displayPath(path) {
    const raw = String(path || '');
    const home = String(window.EU_HOME || '');
    return home && raw.startsWith(home) ? raw.replace(home, '~') : raw;
  }
  // Long project paths (seed folders routinely exceed 80 chars) must not dominate
  // the rail cards or the detail header — compact to '…/basename' and put the
  // full path in the title tooltip instead.
  function compactMiddlePath(path, max) {
    const raw = String(path || '');
    const cap = max || 42;
    if (raw.length <= cap || raw.indexOf('/') === -1) return raw;
    const parts = raw.split('/');
    const base = parts[parts.length - 1] || raw;
    const tail = base.length > cap - 2 ? '…' + base.slice(-(cap - 2)) : base;
    return '…/' + tail;
  }
  function projectFolderLabel(s) {
    if (s && s.empty) return t('No local project folder yet', '还没有本地项目文件夹');
    return s && s.ideaSeed && s.ideaSeed.project_dir
      ? displayPath(s.ideaSeed.project_dir)
      : t('Created when this study is first run', '首次运行时创建');
  }
  function studyBadgeLabel(s) {
    if (s && s.projectKind === 'canonical9') return t('Completed analysis', '已完成分析');
    if (s && s.studyContext) return t('StudyContext', '研究上下文');
    if (s && s.ideaSeed) return t('Research idea', '研究想法');
    return t('Analysis', '分析');
  }
  function studyListContext(studies) {
    const count = Array.isArray(studies) ? studies.length : 0;
    if (agIdeaProjects.loading || !count) return '';
    const localCount = studies.filter(s => s && (s.ideaSeed || s.studyContext)).length;
    const demoCount = studies.filter(s => s && !s.ideaSeed && !s.studyContext).length;
    const line = realMode()
      ? t('This list only shows local Agent projects and completed runs written on this machine.', '这里仅显示本机写入的 Agent 项目和已完成运行。')
      : t('Demo mode includes example projects for exploration. Your own projects appear here after Idea Mining or an Agent run creates a local folder.', '演示模式会放入可探索的示例项目。你自己的项目会在 Idea Mining 或 Agent run 创建本地文件夹后出现在这里。');
    return `
      <div class="ag-list-context" role="note">
        <div class="ag-list-context-k">${realMode() ? t('Local projects', '本地项目') : t('Example projects', '示例项目')}</div>
        <div class="ag-list-context-d">${line}</div>
        <div class="ag-list-context-m">
          <span>${t('local', '本地')} ${localCount}</span>
          ${demoCount ? `<span>${t('examples', '示例')} ${demoCount}</span>` : ''}
        </div>
      </div>`;
  }
  function agentTermStrip(s) {
    if (!s || s.empty) return '';
    return `
      <div class="ag-term-strip" role="note">
        <span class="ag-term"><b>${t('Read-only review', '只读审阅')}</b><em>${t('review an imported/demo package; it does not create or unlock a manuscript draft', '审阅导入或演示包；不会创建或解锁论文草稿')}</em></span>
        <span class="ag-term"><b>${t('Evidence items', '证据项')}</b><em>${t('traceable local artifacts used to support visible claims', '用于支撑可见论断的可追溯本地产物')}</em></span>
        <span class="ag-term"><b>${t('Verification passed', '核验通过')}</b><em>${t('automated checks passed; human review is still required before claims move forward', '自动检查通过；论断推进前仍需要人工审阅')}</em></span>
      </div>`;
  }
  function requestIdeaAgentProjects(force) {
    if (!window.EU_API || !window.EU_API.loadIdeaAgentProjects) return;
    if (!force && (agIdeaProjects.loading || agIdeaProjects.data || agIdeaProjects.error)) return;
    agIdeaProjects = { loading: true, error: null, data: null };
    window.EU_API.loadIdeaAgentProjects({ limit: 50 }).then(data => {
      agIdeaProjects = { loading: false, error: null, data: data };
      const studies = allStudies();
      let preferred = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.activeId
        ? window.EU_AGENT_STUDY_CONTEXT.activeId()
        : null;
      try {
        if (!preferred) {
          const raw = localStorage.getItem('easyicu_last_idea_agent_project');
          const parsed = raw ? JSON.parse(raw) : null;
          preferred = parsed && parsed.study_id;
        }
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
  function exportSourceForStudy(s) {
    const fallback = activeExportSource();
    return window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.sourceFor
      ? window.EU_AGENT_STUDY_CONTEXT.sourceFor(s, fallback)
      : fallback;
  }
  function activeSourceLabel(s) {
    const src = exportSourceForStudy(s);
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
  function importedRunForStudy(s) {
    if (!s || !s.reviewProjectDir) return null;
    const row = Array.isArray(s.seedRuns) ? s.seedRuns.find(r => r && r.project_dir === s.reviewProjectDir) : null;
    const b = s.benchmark || {};
    return {
      run_id: (row && row.label) || b.task_id || s.id,
      run_label: (row && row.label) || b.task_id || s.id,
      study_id: s.id,
      mode: s.mode || 'analysis',
      run_type: 'canonical9_import',
      project_dir: s.reviewProjectDir,
      source: {},
      summary: {},
      gate: {
        status: b.readiness_status || b.tristate || 'analysis_only',
        reportable: false,
        draft_unlocked: false,
        checks: [],
      },
      artifacts: [],
      imported: true,
    };
  }
  function reviewableRunForStudy() {
    const live = liveRunForStudy();
    return live || importedRunForStudy(study());
  }
  function isImportedRun(live, s) {
    return !!(live && (live.imported || live.run_type === 'canonical9_import' || (s && s.projectKind === 'canonical9')));
  }
  function questionText(s) {
    if (!s || !s.question) return '';
    return Array.isArray(s.question) ? t(s.question[0], s.question[1]) : String(s.question || '');
  }
  function questionParts(text) {
    const raw = String(text || '').replace(/\r\n?/g, '\n').trim();
    if (!raw) return { lead: '', context: [], requirements: [] };
    const marked = raw
      .replace(/[ \t]+/g, ' ')
      .replace(/([.;:!?])\s+(\d{1,2})[.)、]\s+/g, (_, punct, n) => `${punct}\n${n}. `);
    const lines = marked.split(/\n+/).map(line => line.trim()).filter(Boolean);
    const intro = [];
    const requirements = [];
    let current = null;
    lines.forEach(line => {
      const match = line.match(/^(\d{1,2})[.)、]\s*(.+)$/);
      if (match) {
        if (current) requirements.push(current);
        current = { n: match[1], text: match[2].trim() };
        return;
      }
      if (current) {
        current.text = `${current.text} ${line}`.trim();
        return;
      }
      intro.push(line);
    });
    if (current) requirements.push(current);
    return {
      lead: intro.shift() || '',
      context: intro,
      requirements,
    };
  }
  function questionTags(s, raw) {
    const text = String(raw || '').toLowerCase();
    const tags = [s && s.cohort ? s.cohort : 'analysis_cohort'];
    if (s && s.id === 'crossdb' || /\bcross[- ]?db\b|multi[- ]database|eicu|aumc|hirid|amsterdamumc/i.test(text)) tags.push('cross_db');
    else if (/first[- ]?24|24\s*h|24-hour|admission-window|admission window/i.test(text)) tags.push('first_24h');
    else tags.push('analysis_window');
    if (/sofa/i.test(text)) tags.push('sofa');
    else if (/ventilat|mechanical vent/i.test(text)) tags.push('ventilation');
    else if (/aki|kdigo/i.test(text)) tags.push('kdigo');
    else if (/lactate/i.test(text)) tags.push('lactate');
    else if (/sepsis|suspected infection/i.test(text)) tags.push('sepsis');
    else tags.push('evidence_bound');
    return tags;
  }
  function renderStructuredQuestion(s) {
    const raw = questionText(s);
    const parts = questionParts(raw);
    const paragraphs = parts.context.map(p => `<p>${esc(p)}</p>`).join('');
    const reqs = parts.requirements.map(row => `<li><span>${esc(row.text)}</span></li>`).join('');
    const fallback = !parts.lead && !parts.context.length && !parts.requirements.length
      ? `<div class="ag-q-lead">${esc(raw)}</div>`
      : '';
    // The data context + numbered tasks are the raw run brief (often verbose,
    // English benchmark prose). Keep the human-readable core question visible
    // and fold the technical brief so the overview does not open as a wall of
    // text. No-lead parses keep the fallback inline so nothing is hidden.
    const contextSection = paragraphs
      ? `<section class="ag-q-section"><div class="ag-q-kicker">${t('Data context', '数据上下文')}</div><div class="ag-q-copy">${paragraphs}</div></section>`
      : '';
    const reqsSection = reqs
      ? `<section class="ag-q-section"><div class="ag-q-kicker">${t('Analysis requirements', '任务要求')}</div><ol class="ag-req-list">${reqs}</ol></section>`
      : '';
    const taskCount = parts.requirements.length;
    const brief = (parts.lead && (contextSection || reqsSection))
      ? `<details class="ag-q-more">
          <summary class="ag-q-more-sum">
            <span class="ag-q-more-ico">${icon('chevron', 12)}</span>
            <span class="ag-q-more-lab">${t('Data context & analysis tasks', '数据上下文与分析任务')}${taskCount ? ` · ${taskCount} ${t('tasks', '项任务')}` : ''}</span>
            <span class="ag-q-more-hint">${t('technical brief', '技术细节')}</span>
          </summary>
          <div class="ag-q-more-body">${contextSection}${reqsSection}</div>
        </details>`
      : `${contextSection}${reqsSection}`;
    return `
      <div class="card pad ag-question-brief">
        <div class="eyebrow">${t('Research question', '研究问题')}</div>
        ${parts.lead ? `<section class="ag-q-section"><div class="ag-q-kicker">${t('Core question', '核心问题')}</div><div class="ag-q-lead">${esc(parts.lead)}</div></section>` : fallback}
        ${brief}
        <div class="row wrap gap-6 mt-12">
          ${questionTags(s, raw).map(tag => `<span class="chip">@${esc(tag)}</span>`).join('')}
        </div>
      </div>`;
  }
  function reviewPayload(live, name) {
    const review = currentLiveReview(live);
    const payloads = review && review.artifact_payloads ? review.artifact_payloads : {};
    return payloads[name] || null;
  }
  function evidenceLinkPanel(live, s) {
    if (!live && !(s && s.benchmark)) return '';
    const b = (s && s.benchmark) || {};
    const ledger = reviewPayload(live, 'evidence_ledger.json') || {};
    const source = reviewPayload(live, 'source_run_manifest.json') || {};
    const gatePayload = reviewPayload(live, 'quality_gate.json') || {};
    const draft = reviewPayload(live, 'manuscript_draft.json') || {};
    const artifacts = Array.isArray(ledger.artifacts) ? ledger.artifacts : artifactsForLive(live);
    const checks = gatePayload.gate && Array.isArray(gatePayload.gate.checks) ? gatePayload.gate.checks : (live && live.gate && Array.isArray(live.gate.checks) ? live.gate.checks : []);
    const evidenceCheck = checks.find(c => c && c.id === 'evidence_binding') || {};
    const readiness = source.readiness || {};
    const strictAudit = ledger.strict_evidence_audit || {};
    const privacy = ledger.privacy || source.privacy || {};
    const evidenceCount = firstValue(b.evidence_count, source.evidence_count, evidenceCheck.evidence_count, strictAudit.evidence_count);
    const missingEvidence = firstValue(b.missing_evidence, readiness.missing_evidence_count, evidenceCheck.missing_evidence, strictAudit.missing_evidence_count, 0);
    const claimCount = Array.isArray(draft.claims) ? draft.claims.length : (Array.isArray(draft.sentences) ? draft.sentences.length : 0);
    const hashCount = artifacts.filter(a => a && a.sha256).length;
    const privacyClean = privacy.patient_rows_returned === false || privacy.direct_identifiers_returned === false;
    const linked = Number(missingEvidence || 0) === 0;
    return `
      <div class="ag-cap-card evidence">
        <div class="ag-cap-head">
          <div>
            <div class="eyebrow">${t('Evidence Link', '证据链接')}</div>
            <div class="ag-cap-title">${t('Claim-to-artifact trace is explicit', '论断到产物的追踪是显式的')}</div>
          </div>
          <span class="pill ${linked ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>${linked ? t('linked', '已绑定') : t('needs review', '需审阅')}</span>
        </div>
        <div class="ag-link-chain" aria-label="${esc(t('Claim to evidence chain', '论断到证据链'))}">
          <span>${t('Claim', '论断')}</span>
          <span>${t('Evidence ID', '证据 ID')}</span>
          <span>${t('SHA-256', 'SHA-256')}</span>
          <span>${t('Gate', '核验')}</span>
        </div>
        <div class="ag-cap-text">${t('Each visible claim can be traced to local artifacts, evidence IDs, hashes, and the quality gate. This is the part to emphasize as the trust layer of the Agent module.', '每条可见论断都能追到本地产物、证据 ID、哈希和质量核验。这就是 Agent 模块的可信层，汇报时应该重点讲。')}</div>
        <div class="ag-cap-metrics">
          <div><span>${t('Evidence items', '证据项')}</span><strong>${fmtCount(evidenceCount)}</strong></div>
          <div><span>${t('Hashed artifacts', '哈希产物')}</span><strong>${fmtCount(hashCount)}</strong></div>
          <div><span>${t('Locked claims', '锁定论断')}</span><strong>${fmtCount(claimCount)}</strong></div>
          <div><span>${t('Missing evidence', '缺失证据')}</span><strong>${fmtCount(missingEvidence)}</strong></div>
        </div>
        <div class="ag-cap-actions">
          <button class="btn sm primary" data-ag-artifact-jump="evidence_ledger.json">${icon('shield', 12)} ${t('Open evidence ledger', '打开证据账本')}</button>
          <button class="btn sm" data-ag-artifact-jump="source_run_manifest.json">${icon('history', 12)} ${t('Open provenance', '打开溯源')}</button>
          <span class="ag-cap-note">${privacyClean ? t('No patient rows persisted in review artifacts.', '审阅产物不持久化患者行。') : t('Privacy scan is shown in the ledger.', '隐私扫描见证据账本。')}</span>
        </div>
      </div>`;
  }
  function crossDataPanel(live, s) {
    if (!s || s.empty) return '';
    const context = reviewPayload(live, 'run_context.json') || {};
    const cohort = reviewPayload(live, 'cohort_summary.json') || {};
    const score = reviewPayload(live, 'benchmark_scorecard.json') || {};
    const questionText = [context.question, s.question && s.question[0], s.cohort, s.sourceArticle].filter(Boolean).join(' ');
    const explicitScope = firstValue(s.ideaSeed && s.ideaSeed.cohort, context.source && context.source.database, score.database_scope);
    const inferredScope = /mimic-iv/i.test(questionText) ? 'MIMIC-IV canonical benchmark universe' : (s.id === 'crossdb' ? 'Cross-DB comparison workspace' : explicitScope || t('Active export scope', '当前导出范围'));
    const cohortSize = firstValue(score.cohort_size, context.summary && context.summary.stays, cohort.summary && cohort.summary.stays, cohort.cohort && cohort.cohort.entities, s.benchmark && s.benchmark.cohort_size);
    const modules = firstValue(context.summary && context.summary.modules, cohort.summary && cohort.summary.modules);
    let crossCount = null;
    try {
      crossCount = window.EU_SOURCES && window.EU_SOURCES.crossdbPaths ? window.EU_SOURCES.crossdbPaths().length : null;
    } catch (_) {
      crossCount = null;
    }
    const isCross = /cross|multi|six|database/i.test(String(inferredScope || '')) && !/mimic-iv canonical/i.test(String(inferredScope || ''));
    return `
      <div class="ag-cap-card cross">
        <div class="ag-cap-head">
          <div>
            <div class="eyebrow">${t('Cross-data scope', '跨数据范围')}</div>
            <div class="ag-cap-title">${isCross ? t('Multi-database analysis context', '多数据库分析上下文') : t('Data scope is declared before claims', '下结论前先声明数据范围')}</div>
          </div>
          <span class="pill ${isCross ? 'ok' : 'info'}" style="height:22px;"><span class="dot"></span>${isCross ? t('cross-db', '跨库') : t('scoped', '已限定')}</span>
        </div>
        <div class="ag-cap-text">${t('The Agent module should show which data context a run consumed. Cross-DB comparisons are prepared in the Cross-DB workspace, then passed into the same evidence-bound Agent review path.', 'Agent 模块应该显示一次运行消费了哪个数据上下文。跨库比较先在 Cross-DB 工作台准备，再进入同一套证据绑定 Agent 审阅路径。')}</div>
        <div class="ag-cap-metrics">
          <div><span>${t('Current scope', '当前范围')}</span><strong>${esc(inferredScope)}</strong></div>
          <div><span>${t('Denominator', '分母')}</span><strong>${fmtCount(cohortSize)}</strong></div>
          <div><span>${t('Modules', '模块')}</span><strong>${modules == null ? '—' : fmtCount(modules)}</strong></div>
          <div><span>${t('Cross-DB exports', '跨库导出')}</span><strong>${crossCount == null ? '—' : fmtCount(crossCount)}</strong></div>
        </div>
        <div class="ag-cap-actions">
          <button class="btn sm" data-nav="crossdb">${icon('benchmark', 12)} ${t('Open Cross-DB workspace', '打开跨库工作台')}</button>
          <span class="ag-cap-note">${isCross ? t('This project is already using a multi-database context.', '这个项目已经使用多数据库上下文。') : t('Current canonical9 package is scoped; use Cross-DB for six-database comparison.', '当前 canonical9 包是限定范围；六库比较请进入 Cross-DB。')}</span>
        </div>
      </div>`;
  }
  function capabilityHighlights(live, s) {
    const evidence = evidenceLinkPanel(live, s);
    const cross = crossDataPanel(live, s);
    if (!evidence && !cross) return '';
    return `<div class="ag-cap-grid">${evidence}${cross}</div>`;
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
    const s = study();
    const imported = s.seedRuns && s.seedRuns.find(r => r && r.project_dir && r.artifact_count != null);
    if (imported) return Number(imported.artifact_count || 0);
    return 0;
  }
  function requestLiveReview(live) {
    if (!live || !live.project_dir || !window.EU_API || !window.EU_API.loadAgentRunReview) return;
    if (agReview.projectDir === live.project_dir && (agReview.loading || agReview.data || agReview.error)) return;
    agReview = { projectDir: live.project_dir, loading: true, error: null, data: null, signing: false };
    window.EU_API.loadAgentRunReview(live.project_dir).then(data => {
      agReview = { projectDir: live.project_dir, loading: false, error: null, data: data, signing: false };
      window.EU_AGENT_RUN_REVIEW = data;
      const firstArtifact = defaultArtifactName(data && data.artifacts);
      if (agTab === 'outputs' && firstArtifact && (!agArtifact.name || agArtifact.projectDir !== live.project_dir)) {
        requestArtifact(live, firstArtifact);
        return;
      }
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
    const seedDir = (s.ideaSeed && s.ideaSeed.project_dir) || undefined;
    window.EU_API.loadAgentRunHistory({ study_id: s.id, limit: 50, project_seed_dir: seedDir }).then(data => {
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
  function openReview(review, targetTab) {
    if (!review || !review.ok) return;
    window.EU_AGENT_LAST_RUN = liveRunFromReview(review);
    window.EU_AGENT_RUN_REVIEW = review;
    agReview = { projectDir: review.project_dir, loading: false, error: null, data: review, signing: false };
    agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
    agTab = targetTab || 'draft';
  }
  function requestArtifact(live, name) {
    if (!live || !live.project_dir || !name || !window.EU_API || !window.EU_API.loadAgentRunArtifact) return;
    agArtifact = { projectDir: live.project_dir, name: name, loading: true, error: null, data: null };
    repaintBody();
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
    agJobMemory.remember(meta);
  }
  function readRememberedAgentJob(studyId) {
    return agJobMemory.get(studyId);
  }
  function clearRememberedAgentJob(jobId, studyId) {
    agJobMemory.clear(jobId, studyId);
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
  function attachAgentJobStream(runToken) {
    const jobId = runToken && runToken.job_id;
    if (!jobId || !window.EventSource) return;
    if (agRunChannel.isCurrent(runToken)) closeRunStream();
    const es = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    if (agRunChannel.isCurrent(runToken)) agRun.es = es;
    let ended = false;
    es.onmessage = msg => {
      const ev = JSON.parse(msg.data);
      if (agRunChannel.isCurrent(runToken)) applyRunEventProgress(ev);
      if (ev.type === 'end') {
        ended = true;
        try { es.close(); } catch (_) {}
        finishRealRun(runToken, ev.status, ev.result, ev.error);
      }
    };
    es.onerror = () => {
      if (ended) return;
      try { es.close(); } catch (_) {}
      if (!agRunChannel.isCurrent(runToken) || !agRun.active) return;
      if (agRun.es === es) agRun.es = null;
      agRun.active = false;
      agRun.error = t('Connection interrupted. If the server job is still running, resume the stream; otherwise retry from the active export.', '连接中断。如果服务端任务仍在运行,可恢复任务流；否则从 active export 重跑。');
      agRun.reconnectable = true;
      repaintBody();
    };
  }
  function restoreAgentJobFromSnapshot(meta, snapshot) {
    const target = study();
    if (!meta || !meta.study_id || target.id !== meta.study_id) return;
    const runToken = agRunChannel.start({
      surface: 'agent',
      study_id: target.id,
      context_id: (target.studyContext && target.studyContext.id) || meta.study_context_id || '',
      context_revision: Number.isInteger(meta.study_context_revision) ? meta.study_context_revision : null,
      job_id: snapshot.id || meta.job_id,
      question: target.question && target.question[0],
      source_path: meta.source_path || '',
      study_mode: target.mode,
      run_type: meta.run_type || 'preflight',
      provider: meta.provider || 'mock',
      project_seed_dir: target.ideaSeed && target.ideaSeed.project_dir,
    });
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
        warning: null,
        reconnectable: false,
      };
      applyRunEventProgress(ev || {});
      agTab = 'overview';
      attachAgentJobStream(runToken);
      repaintBody();
      return;
    }
    finishRealRun(runToken, snapshot.status, snapshot.result, snapshot.error);
  }
  function maybeRestoreAgentJob() {
    if (agRun.active || agRun.result || agRun.error || agResumeProbe.loading) return;
    if (!window.EU_API || !window.EU_API.loadJobSnapshot) return;
    if (realMode() && agIdeaProjects.loading) return;
    if (realMode() && !agIdeaProjects.data && !agIdeaProjects.error) {
      requestIdeaAgentProjects();
      return;
    }
    const selected = study();
    const contextJobId = selected.studyContext && selected.studyContext.active_job_id;
    const remembered = readRememberedAgentJob(selected.id);
    const meta = contextJobId
      ? { job_id: contextJobId, study_id: selected.id, source_path: selected.studyContext.data_source && selected.studyContext.data_source.path }
      : (remembered && remembered.study_id === selected.id ? remembered : null);
    if (!meta || !meta.job_id || agResumeProbe.checkedJobId === meta.job_id) return;
    const selectedId = selected.id;
    agResumeProbe = { loading: true, checkedJobId: meta.job_id };
    window.EU_API.loadJobSnapshot(meta.job_id).then(snapshot => {
      agResumeProbe.loading = false;
      if (study().id !== selectedId) return;
      restoreAgentJobFromSnapshot(meta, snapshot);
    }).catch(() => {
      agResumeProbe.loading = false;
      if (!contextJobId) clearRememberedAgentJob(meta.job_id, selectedId);
    });
  }

  /* ---------------- pipeline ---------------- */
  function pipeStages(mode, projectKind) {
    if (mode === 'idea') return [
      ['Frame', '想法框定', t('source + exposure/outcome', '来源 + 暴露/结局'), 'spark'],
      ['Recipe', '数据配方', t('data recipe', '数据配方'), 'layers'],
      ['Dry-run', '试运行', t('feasibility · no claims', '可行性 · 不下结论'), 'play'],
      ['Recommend', '建议', t('suggested workflow', '建议的研究流程'), 'target'],
    ];
    if (projectKind === 'canonical9') return [
      ['Question', '问题', t('clinical benchmark task', '临床 benchmark 问题'), 'play'],
      ['Build', '构建', t('cohort + variables', '队列 + 变量'), 'layers'],
      ['Analyze', '分析', t('figures + checks', '图件 + 校验'), 'viz'],
      ['Evidence check', '证据核验', t('claims tied to artifacts', '论断绑定产物'), 'shield'],
      ['Review', '审阅', t('read-only package', '只读审阅包'), 'file'],
    ];
    return [
      ['Plan', '计划', t('question → recipe', '问题 → 配方'), 'play'],
      ['Build', '构建', t('exports → 1 row / stay', '导出 → 每次住院一行'), 'layers'],
      ['Analyze', '分析', t('tables · figures · checks', '表 · 图 · 校验'), 'viz'],
      ['Evidence check', '证据核验', t('evidence before drafting', '撰稿前的证据校验'), 'shield'],
      ['Draft', '草稿', t('approve · export', '确认 · 导出'), 'file'],
    ];
  }
  function pipeline() {
    const s = study();
    const stages = pipeStages(s.mode, s.projectKind);
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
  function agentListCollapsed() {
    if (agListMode === 'focus') return true;
    if (agListMode === 'open') return false;
    return AG_FOCUS_TABS.has(agTab);
  }
  function studyList() {
    const studies = allStudies();
    const dotCls = { ready: 'ready', running: 'running', gate: 'running', review_blocked: 'running', draft: 'draft', idle: 'idle' };
    return `
    <div class="ag-list" id="agStudyList" aria-label="${t('Research project list', '研究项目列表')}">
      <div class="ag-list-head">
        <div><span class="ttl">${t('Studies', '研究项目')} · ${studies.length}</span><div class="ag-list-cap">${t('each study = a local project folder', '每个研究 = 一个本地项目文件夹')}</div></div>
        <button class="ag-newbtn" data-ag-new>${icon('plus', 13)} ${t('New', '新建')}</button>
      </div>
      ${studyListContext(studies)}
      <div class="ag-studies">
        ${agIdeaProjects.loading ? `<div class="empty-mini" style="margin:10px;min-height:80px;">${t('Loading local research projects…', '正在加载本地研究项目…')}</div>` : ''}
        ${agIdeaProjects.error ? `<div class="note warn" style="margin:10px;"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="t">${t('Local research projects unavailable', '本地研究项目不可用')}</div><div class="d">${esc(agIdeaProjects.error)}</div></div></div>` : ''}
        ${!studies.length && !agIdeaProjects.loading ? `<div class="empty-mini ideas-empty-list" style="margin:10px;min-height:210px;">
          <div>${icon('folder', 22)}</div>
          <h3>${t('No local projects yet', '还没有本地项目')}</h3>
          <p>${t('An Agent project is a local study folder — a question plus its runs, outputs, and an evidence-checked draft. Two ways to start one: turn a question into a plan in Idea Mining, or extract a cohort and hand it off from Data Extraction. Then it appears here.', 'Agent 项目就是一个本地研究文件夹 —— 一个问题加上它的运行、产出和经过证据核验的草稿。两种创建方式：在 Idea 挖掘里把问题变成计划，或先抽取队列再从数据抽取交接过来。之后它会出现在这里。')}</p>
          <div class="row gap-8 mt-12" style="justify-content:center;flex-wrap:wrap;">
            <button class="btn primary sm" data-nav="ideas">${icon('target', 12)} ${t('Open Idea Mining', '打开想法挖掘')}</button>
            <button class="btn sm" data-nav="extraction">${icon('extract', 12)} ${t('Extract data', '抽取数据')}</button>
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
              <span class="sc-name">${esc(t(s.name[0], s.name[1]))}</span>
              <span class="sc-mode analysis">${!s.ideaSeed && !s.studyContext && !s.empty && !realMode() ? `${t('Example', '示例')} · ` : ''}${studyBadgeLabel(s)}</span>
            </div>
            <div class="sc-meta"><span class="sc-folder" title="${esc(folder)}">${icon('folder', 11)} ${esc(compactMiddlePath(folder))}</span></div>
            <div class="sc-meta" style="margin-top:3px;">${s.runs.length ? `${s.runs[0][0]}<span class="mid"></span>${s.runs[0][4][zh ? 1 : 0]}` : t('not run yet', '尚未运行')}</div>
          </button>`;
        }).join('')}
      </div>
    </div>`;
  }

  /* ---------------- detail header ---------------- */
  function detailHead() {
    const s = study();
    const live = reviewableRunForStudy();
    const review = currentLiveReview(live);
    const liveSigned = !!(review && review.signed);
    const compactHeader = agTab !== 'overview' && agTab !== 'workflow';
    const listCollapsed = agentListCollapsed();
    const statusKey = s.projectKind === 'canonical9' ? 'imported' : (liveSigned ? 'reviewed' : (s.signed ? 'ready' : s.status));
    const statusPill = {
      imported: `<span class="pill info"><span class="dot"></span>${t('Read-only review', '只读审阅')}</span>`,
      ready: `<span class="pill ok"><span class="dot"></span>${t('Ready to run', '可运行')}</span>`,
      reviewed: `<span class="pill ok"><span class="dot"></span>${t('Signed analysis-only', '已签署 analysis-only')}</span>`,
      gate: `<span class="pill warn"><span class="dot"></span>${t('Awaiting sign-off', '待签署')}</span>`,
      review_blocked: `<span class="pill bad"><span class="dot"></span>${t('Evidence verification blocked', '证据核验受阻')}</span>`,
      running: `<span class="pill warn"><span class="dot"></span>${t('Running', '运行中')}</span>`,
      draft: `<span class="pill demo"><span class="dot"></span>${t('Exploring', '探索中')}</span>`,
      idle: `<span class="pill"><span class="dot"></span>${t('Not run yet', '尚未运行')}</span>`,
    }[statusKey] || '';
    return `
    <div class="ag-dhead ${compactHeader ? 'compact' : ''}">
      <div class="ag-dtop">
        <div style="min-width:0;">
          <div class="ag-title">${esc(t(s.name[0], s.name[1]))} <span class="editmk">${icon('edit', 14)}</span></div>
          <div class="ag-src">
            <span class="lk" title="${t('Local project folder — intermediate files are written here', '本地项目文件夹 — 中间文件写在这里')}: ${esc(projectFolderLabel(s))}">${icon('folder', 12)} ${esc(compactMiddlePath(projectFolderLabel(s)))}</span>
            <span class="mid"></span>
            <span class="lk">${icon('cohort', 12)} ${esc(s.cohort)}</span>
            ${activeSourceLabel(s) ? `<span class="mid"></span><span class="lk">${icon('db', 12)} ${esc(activeSourceLabel(s))}</span>` : ''}
            <span class="mid"></span>
            ${statusPill}
          </div>
        </div>
        <div class="row gap-8">
          <button class="btn sm" data-ag-toggle-list aria-controls="agStudyList" aria-expanded="${listCollapsed ? 'false' : 'true'}">
            ${icon(listCollapsed ? 'list' : 'close', 13)} ${listCollapsed ? t('Show projects', '显示项目') : t('Focus view', '专注视图')}
          </button>
          <button class="btn sm" data-ag-tab="workflow">${icon('layers', 13)} ${t('Planning Blocks', '规划块')}</button>
          ${s.projectKind === 'canonical9' ? '' : `<button class="btn sm" data-nav="ideas">${icon('target', 13)} ${t('Open Idea Mining', '打开想法挖掘')}</button>`}
          <span class="pill ok"><span class="dot"></span>${t('Analysis workspace', '分析运行工作台')}</span>
        </div>
      </div>
      ${compactHeader ? '' : pipeline()}
      ${compactHeader ? '' : agentTermStrip(s)}
    </div>`;
  }

  /* ---------------- tabs ---------------- */
  function tabsFor(mode) {
    const s = study();
    if (s.empty) return [
      ['overview', t('Overview', '概览'), null],
    ];
    // Tab order follows the actual workflow: lead with what the user consumes
    // (Runs -> Outputs -> Draft), then the same run's deeper Evidence view and the
    // Planning Blocks. The Evidence tab (id 'science') is the provenance deep-dive
    // of THIS study's run, not a separate app — see screens-agent-science.js.
    if (mode === 'idea') return [
      ['overview', t('Overview', '概览'), null],
      ['runs', t('Dry-runs', '试运行'), s.runs.length],
      ['notes', t('Notes', '笔记'), null],
      ['science', t('Evidence', '证据'), null],
      ['workflow', t('Planning Blocks', '规划块'), workflowBlocks(s).length],
    ];
    return [
      ['overview', t('Overview', '概览'), null],
      ['runs', t('Runs', '运行历史'), s.runs.length],
      ['outputs', t('Outputs', '产出'), outputCountForStudy()],
      ['draft', s.projectKind === 'canonical9' ? t('Review', '审阅') : t('Draft', '草稿'), null],
      ['science', t('Evidence', '证据'), null],
      ['workflow', t('Planning Blocks', '规划块'), workflowBlocks(s).length],
    ];
  }
  // Tabs that only fill in after a run exists — flagged so a first-time user
  // isn't invited to click empty heroes one by one.
  const AG_RUN_GATED_TABS = new Set(['runs', 'outputs', 'draft', 'science']);
  function tabsRow() {
    const s = study();
    const tabs = tabsFor(s.mode);
    if (!tabs.some(x => x[0] === agTab)) agTab = 'overview';
    const noRun = !s.runs || s.runs.length === 0;
    return `<div class="ag-tabs" data-ag-tabs role="tablist" aria-label="${t('Agent project views', 'Agent 项目视图')}">
      ${tabs.map(([id, lab, cnt]) => {
        const gated = noRun && AG_RUN_GATED_TABS.has(id);
        const title = gated ? ` title="${t('Available after a run', '运行后可用')}"` : '';
        const selected = agTab === id;
        return `<button id="agTab-${id}" class="ag-tab ${selected ? 'on' : ''}${gated ? ' gated' : ''}" data-ag-tab="${id}" role="tab" aria-selected="${selected}" aria-controls="agTabPanel" tabindex="${selected ? '0' : '-1'}"${title}>${lab}${cnt != null ? `<span class="cnt">${cnt}</span>` : ''}${gated ? `<span class="ag-tab-lock">${icon('lock', 9)}</span>` : ''}</button>`;
      }).join('')}
    </div>`;
  }

  /* ---------------- tab bodies ---------------- */
  function seedPlanStepDisplay(row) {
    if (row && typeof row === 'object') {
      const title = row.title || row.action || row.phase || t('Plan step', '计划步骤');
      const detail = [
        row.phase,
        row.output || row.guardrail || t('from Idea Mining handoff', '来自 Idea Mining 交接'),
      ].filter(Boolean).join(' · ');
      return { title, detail };
    }
    return {
      title: String(row || t('Plan step', '计划步骤')),
      detail: t('from Idea Mining handoff', '来自 Idea Mining 交接'),
    };
  }
  function planList() {
    const s = study();
    const seedPlan = s.ideaSeed && Array.isArray(s.ideaSeed.analysis_plan) ? s.ideaSeed.analysis_plan : null;
    const real = window.EU_DATA === 'real';
    // Plan provenance rule: a seed plan (Idea Mining handoff) is the study's
    // real plan; otherwise Real mode shows the actual preflight step list a
    // run performs — never the seeded sepsis demo fixture presented as if it
    // were this study's confirmed plan.
    const demoFixture = !seedPlan && s.mode !== 'idea' && !real;
    const plan = seedPlan
      ? seedPlan.map(x => {
        const step = seedPlanStepDisplay(x);
        return [step.title, step.detail, 'ready'];
      })
      : s.mode === 'idea'
      ? [
        [t('Frame the idea', '框定想法'), t('exposure / outcome / source', '暴露 / 结局 / 来源'), 'ready'],
        [t('Data recipe', '数据配方'), t('concepts the dry-run needs', '试运行所需概念'), 'ready'],
        [t('Feasibility dry-run', '可行性试运行'), t('counts, coverage — no effect sizes', '计数、覆盖率 —— 不给效应量'), 'ready'],
        [t('Recommendation', '建议'), t('is a full study worth it?', '是否值得开展完整研究?'), 'ready'],
      ]
      : real
      ? [
        [t('Export snapshot', '导出快照'), t('bind the active export · hash inputs', '绑定 active export · 记录输入哈希'), 'ready'],
        [t('Cohort & quality summary', '队列与质量摘要'), t('denominators, coverage, quality flags', '分母、覆盖率与质量标记'), 'ready'],
        [t('Bounded output artifacts', '有界输出产物'), t('tables · figures · run manifest', '表格 · 图件 · 运行清单'), 'ready'],
        [t('Evidence verification & sign-off', '证据核验与签署'), t('claims stay locked until checks pass', '检查通过前结论保持锁定'), 'gated'],
      ]
      : [
        [t('Cohort summary', '队列摘要'), t('n, demographics, outcome rates', 'n、人口学、结局率'), 'ready'],
        [t('Table 1', 'Table 1'), t('baseline by group', '分组基线特征'), 'ready'],
        [t('Missingness audit', '缺失审计'), t('coverage + denominators', '覆盖率 + 分母'), 'ready'],
        [t('Model: LR + SOFA + lactate', '模型:LR + SOFA + 乳酸'), t('first-24h predictors', '前 24h 预测因子'), 'ready'],
        [t('ROC · Calibration', 'ROC · 校准'), t('discrimination + calibration', '区分度 + 校准度'), 'ready'],
        [t('Manuscript draft', '论文草稿'), t('methods + results', '方法 + 结果'), 'gated'],
      ];
    const gatedN = plan.filter(row => row[2] === 'gated').length;
    const readyN = plan.length - gatedN;
    const planPill = s.mode === 'analysis'
      ? `<span class="pill ok" style="height:20px;"><span class="dot"></span>${readyN} ${t('ready', '就绪')}${gatedN ? ` · ${gatedN} ${t('needs review', '待核验')}` : ''}</span>`
      : `<span class="pill ok" style="height:20px;"><span class="dot"></span>${t('feasibility only', '仅可行性')}</span>`;
    return `
      <div class="card pad">
        <div class="row" style="justify-content:space-between;align-items:baseline;">
          <div class="eyebrow">${t('Plan', '计划')} · ${plan.length} ${t('steps', '步')}${demoFixture ? ` · <span style="color:var(--warn,#a66a00);">${t('demo example plan', '演示示例计划')}</span>` : (!seedPlan && s.mode !== 'idea' ? ` · ${t('preflight steps', '预检步骤')}` : '')}</div>
          ${planPill}
        </div>
        <div class="planlist mt-12">
          ${plan.map(([ti, d, st], i) => `
            <div class="plan-item ${st}">
              <div class="pi-n mono">${String(i + 1).padStart(2, '0')}</div>
              <div class="pi-node">${st === 'gated' ? icon('lock', 11, 2) : icon('check', 12, 2.6)}</div>
              <div class="pi-body"><div class="pi-t">${esc(ti)}</div><div class="pi-d">${esc(d)}</div></div>
              <div class="pi-tag">${st === 'gated' ? `<span class="pill dashed">${t('requires review', '需审阅')}</span>` : `<span class="pill ok" style="height:20px;"><span class="dot"></span>${t('planned', '已计划')}</span>`}</div>
            </div>`).join('')}
        </div>
      </div>`;
  }

  function providerRunPanel() {
    if (window.EU_DATA !== 'real') return '';
    if (study().empty || study().mode !== 'analysis') return '';
    const src = exportSourceForStudy(study());
    const contextBlocker = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.runBlocker
      ? window.EU_AGENT_STUDY_CONTEXT.runBlocker(study())
      : '';
    requestProviderStatus();
    const st = agProvider.status || {};
    const limits = st.limits || {};
    const envFile = st.env_file || {};
    const missing = Array.isArray(st.missing) ? st.missing : [];
    const ready = !!st.ready;
    const canRun = !!(src && ready && agProvider.consent && !agRun.active && !contextBlocker);
    const disabledReason = contextBlocker || (!src
      ? t('No active export source', '没有 active export 源')
      : !ready
      ? (missing.length ? missing.join(', ') : t('Provider not ready', 'provider 未就绪'))
      : !agProvider.consent
      ? t('Per-run confirmation required', '需要逐次确认')
      : agRun.active
      ? t('Run already in progress', '已有运行进行中')
      : '');
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
            <div class="eyebrow">${t('External provider scaffold', '外部 provider 骨架')}</div>
            <div class="panel-sub" style="margin-top:4px;">${t('Generates a provider-backed plan and draft scaffold; it does not run a complete research analysis. Uses env vars only, and secrets are never shown or written to artifacts.', '生成 provider-backed 计划与草稿骨架；这不是完整的研究分析。只读取环境变量，密钥不会显示或写入产物。')}</div>
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
          <button class="btn primary sm" data-ag-external-run aria-disabled="${canRun ? 'false' : 'true'}">${icon('file', 12)} ${t('Generate provider scaffold', '生成 provider 骨架')}</button>
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
    if (seedGateBlocksRun(s)) {
      return `
      <div class="nextbar gate">
        <div class="nb-ico">${icon('shield', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Agent preflight checks are not ready', 'Agent 预检条件尚未就绪')}</div><div class="nb-d">${esc(seedGateBlockerText(s))}</div></div>
        <button class="btn" data-nav="ideas">${icon('spark', 13)} ${t('Review in Idea Mining', '回到 Idea Mining 审阅')}</button>
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
    if (s.reviewProjectDir) {
      const b = s.benchmark || {};
      const status = b.tristate || b.readiness_status || 'imported';
      return `
      <div class="nextbar gate">
        <div class="nb-ico">${icon('history', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Completed analysis package is ready to review', '已完成分析包可审阅')}</div><div class="nb-d">${esc(runStatusLabel(status))} · ${Number(b.evidence_count || 0).toLocaleString()} ${t('evidence artifacts', '条证据产物')} · <span class="mono">${esc(displayPath(s.reviewProjectDir))}</span></div></div>
        <button class="btn primary" data-ag-open-seed-run="${esc(s.reviewProjectDir)}">${icon('eye', 13)} ${t('Open results', '查看结果')}</button>
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
    if (s.status === 'gate' || s.status === 'review_blocked') {
      const gate = (agRun.result && agRun.result.gate) || s.gate || null;
      const checks = gate && Array.isArray(gate.checks) ? gate.checks.filter(row => row && typeof row === 'object') : [];
      const rows = checks.map(check => {
        const id = String(check.id || check.name || 'evidence_check');
        const human = id === 'human_signoff';
        const passed = check.passed === true;
        const state = passed ? 'passed' : (human ? 'pending' : 'failed');
        const label = gateCheckLabel(check);
        return { label, passed, state };
      });
      const passed = rows.filter(row => row.passed).length;
      const failed = rows.filter(row => row.state === 'failed').length;
      const pending = rows.filter(row => row.state === 'pending').length;
      const blocked = s.status === 'review_blocked' || !!(gate && gate.status === 'blocked') || failed > 0;
      const title = blocked
        ? t('Evidence verification blocked', '\u8bc1\u636e\u6838\u9a8c\u53d7\u963b')
        : (pending ? t('Evidence checks need human sign-off', '\u8bc1\u636e\u6838\u9a8c\u9700\u8981\u4eba\u5de5\u7b7e\u7f72') : t('Evidence check results', '\u8bc1\u636e\u6838\u9a8c\u7ed3\u679c'));
      const detail = gate && gate.reason
        ? String(gate.reason).replace(/_/g, ' ')
        : t('Waiting for verification results. No check is shown as passed until the backend reports it.', '\u7b49\u5f85\u6838\u9a8c\u7ed3\u679c\u3002\u540e\u7aef\u672a\u62a5\u544a\u524d\uff0c\u4e0d\u4f1a\u628a\u4efb\u4f55\u68c0\u67e5\u6807\u4e3a\u5df2\u901a\u8fc7\u3002');
      return `
      <div class="nextbar gate">
        <div class="nb-ico">${icon('shield', 16)}</div>
        <div class="grow"><div class="nb-t">${title}</div><div class="nb-d">${esc(detail)}</div></div>
        <button class="btn primary" data-ag-tab="${blocked ? 'science' : 'draft'}">${icon(blocked ? 'eye' : 'check', 13)} ${blocked ? t('Review failed evidence', '\u5ba1\u9605\u5931\u8d25\u8bc1\u636e') : t('Review & sign off', '\u5ba1\u9605\u5e76\u7b7e\u7f72')}</button>
      </div>
      <div class="card pad" style="margin-top:10px;">
        <div class="eyebrow" style="display:flex;align-items:center;gap:8px;">${t('Reported verification checks', '\u540e\u7aef\u62a5\u544a\u7684\u6838\u9a8c\u9879')}<span class="mono" style="margin-left:auto;color:var(--ink-4);font-size:10.5px;">${rows.length ? `${passed}/${rows.length}` : '\u2014'}</span></div>
        <div class="gate-checklist">
          ${rows.length ? rows.map(row => `
            <div class="gc-row ${row.passed ? 'ok' : 'pending'}">
              <span class="gc-mk">${row.passed ? icon('check', 11, 2.8) : icon(row.state === 'pending' ? 'clock' : 'alert', 11)}</span>
              <span>${esc(row.label)}</span>
              <span class="gc-tag">${row.state === 'passed' ? t('passed', '\u901a\u8fc7') : (row.state === 'pending' ? t('pending human sign-off', '\u7b49\u5f85\u4eba\u5de5\u7b7e\u7f72') : t('failed', '\u5931\u8d25'))}</span>
            </div>`).join('') : `<div class="gc-row pending"><span class="gc-mk">${icon('clock', 11)}</span><span>${t('Waiting for verification results', '\u7b49\u5f85\u6838\u9a8c\u7ed3\u679c')}</span><span class="gc-tag">${t('not reported', '\u672a\u62a5\u544a')}</span></div>`}
        </div>
      </div>`;
    }
    if (s.status === 'idle') {
      return `
      <div class="nextbar accent">
        <div class="nb-ico">${icon('play', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Ready to run the preflight', '准备运行预检')}</div><div class="nb-d">${t('This runs a deterministic, local evidence preflight (cohort, coverage, Table 1 — no external model call). The provider panel below can separately generate a plan and draft scaffold, not a complete analysis.', '这会运行确定性的本地证据预检（队列、覆盖率、Table 1 —— 不调用外部模型）。下方 provider 面板可单独生成计划与草稿骨架，但不会运行完整分析。')}</div></div>
        <button class="btn primary" data-ag-runbtn>${icon('play', 13)} ${t('Run preflight', '运行预检')}</button>
      </div>`;
    }
    return `
      <div class="nextbar accent">
        <div class="nb-ico">${icon('refresh', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Re-run or extend the analysis', '重新运行或扩展分析')}</div><div class="nb-d">${t('Outputs are current. Run again to refresh, or move to evidence verification.', '产出为最新。可重新运行刷新,或前往证据核验。')}</div></div>
        <button class="btn primary" data-ag-runbtn>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>
      </div>`;
  }

  function contextStats() {
    const s = study();
    const src = exportSourceForStudy(s);
    const sum = (src && src.summary) || {};
    const b = s.benchmark || null;
    // Real mode must never show invented clinical numbers: if no benchmark and
    // no attached export, fall to em-dashes + an "attach an export" hint rather
    // than the seeded demo figures (which are only honest as a demo preview).
    const noData = !b && !src;
    const stats = b
      ? [['Cohort', '队列', b.cohort_size == null ? '—' : Number(b.cohort_size).toLocaleString()], ['Evidence', '证据', b.evidence_count == null ? '—' : Number(b.evidence_count).toLocaleString()], ['Warnings', '警告', b.warnings == null ? '—' : String(b.warnings)], ['Status', '状态', runStatusLabel(b.tristate || b.readiness_status || 'analysis_only')]]
      : src
      ? [['Stays', '住院数', sum.stays == null ? '—' : Number(sum.stays).toLocaleString()], ['Modules', '模块', sum.modules == null ? '—' : String(sum.modules)], ['Rows', '行数', sum.total_rows == null ? '—' : Number(sum.total_rows).toLocaleString()], ['Evidence check', '证据核验', 'strict']]
      : (noData && realMode())
      ? (s.id === 'crossdb'
        ? [['Databases', '数据库', '—'], ['Shared concepts', '共享概念', '—'], ['Mortality', '死亡率', '—'], ['Concordance', '一致性', '—']]
        : [['Mean age', '平均年龄', '—'], ['Mortality', '死亡率', '—'], ['Sepsis-3', 'Sepsis-3', '—'], ['Mech vent', '机械通气', '—']])
      : s.id === 'crossdb'
      ? [['Databases', '数据库', '3'], ['Shared concepts', '共享概念', '6'], ['Mortality', '死亡率', '20.0%'], ['Concordance', '一致性', 'high']]
      : [['Mean age', '平均年龄', '54.8 y'], ['Mortality', '死亡率', '20.0%'], ['Sepsis-3', 'Sepsis-3', '45.3%'], ['Mech vent', '机械通气', '52.1%']];
    const noDataHint = noData && realMode();
    const linked = src ? (src.label || src.database || 'local export') : t(s.source[0], s.source[1]);
    const linkedPath = src ? src.path : null;
    return `
    <div class="card pad">
      <div class="eyebrow" style="margin-bottom:12px;">${t('Project folder', '项目文件夹')}</div>
      <div class="row gap-8" style="align-items:center;"><span style="color:var(--ink-3);flex:none;">${icon('folder', 14)}</span><div class="mono" style="font-size:11.5px;color:var(--ink-2);min-width:0;overflow:hidden;text-overflow:ellipsis;">${esc(projectFolderLabel(s))}</div></div>
      <div class="eyebrow" style="margin:14px 0 8px;">${src ? t('Linked export source', '关联导出源') : t('Linked cohort', '关联队列')}</div>
      <div style="font-weight:600;font-size:13px;">${esc(s.cohort)}</div>
      <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${esc(linkedPath || linked)}</div>
      ${window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.bindingNote ? window.EU_AGENT_STUDY_CONTEXT.bindingNote(s) : ''}
      ${window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.warningNote ? window.EU_AGENT_STUDY_CONTEXT.warningNote(agRun.warning) : ''}
      ${s.projectKind === 'canonical9' ? `<div class="note ok mt-12"><div class="ico">${icon('shield', 13)}</div><div class="body"><div class="t">${t('Imported completed analysis', '已导入完成分析')}</div><div class="d">${esc(s.sourceArticle || t('Figure 2 benchmark question package', 'Figure 2 基准问题审阅包'))}</div></div></div>` : (s.ideaSeed ? `<div class="note ok mt-12"><div class="ico">${icon('target', 13)}</div><div class="body"><div class="t">${t('Created from Idea Mining', '来自 Idea Mining 的研究想法')}</div><div class="d">${esc(s.sourceArticle || '')}</div></div></div>` : '')}
      <div class="cols-2 mt-12" style="gap:8px;">
        ${stats.map(([en, zh, v]) => `
          <div style="padding:8px 10px;background:var(--surface-2);border-radius:var(--r-2);">
            <div class="eyebrow" style="font-size:9.5px;">${t(en, zh)}</div>
            <div class="mono" style="font-size:13px;font-weight:500;color:var(--ink);margin-top:3px;">${v}</div>
          </div>`).join('')}
      </div>
      ${noDataHint
        ? `<div class="note info mt-12"><div class="ico">${icon('folder', 13)}</div><div class="body"><div class="t">${t('No export attached', '未关联导出')}</div><div class="d">${t('Attach a local EasyICU export to this project to populate real cohort figures.', '为此项目关联本地 EasyICU 导出后，这里会显示真实队列数据。')}</div></div></div>`
        : (noData ? `<div class="note warn mt-12" style="padding:8px 11px;"><div class="ico">${icon('beaker', 13)}</div><div class="body"><div class="d" style="margin:0;">${t('Illustrative demo figures — not a computed result.', '示例演示数据 —— 非计算结果。')}</div></div></div>` : '')}
      <button class="btn sm block mt-16" data-nav="extraction">${icon('layers', 13)} ${t('Open in Data Extraction', '在数据抽取中打开')}</button>
    </div>`;
  }

  function benchmarkPanel(s) {
    const b = s && s.benchmark;
    if (!b) return '';
    const dims = Array.isArray(b.dimensions) ? b.dimensions : [];
    return `
      <div class="card pad ag-bench-card">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:12px;">
          <div>
            <div class="eyebrow">${t('Figure 2 question package', 'Figure 2 问题包')}</div>
            <div class="panel-title" style="font-size:15px;margin-top:4px;">${esc(b.task_id || s.id)} · ${esc(runStatusLabel(b.tristate || b.readiness_status || 'analysis_only'))}</div>
            <div class="panel-sub">${t('Imported from the completed EasyICU aware workflow. The run stays read-only and analysis-only until the Fig 2 freeze decision.', '来自已完成的 EasyICU aware workflow。该运行保持只读和 analysis-only，直到 Fig 2 冻结策略拍板。')}</div>
          </div>
          <button class="btn sm primary" data-ag-open-seed-run="${esc(s.reviewProjectDir || '')}">${icon('eye', 12)} ${t('Open results', '查看结果')}</button>
        </div>
        <div class="ag-bench-metrics mt-12">
          ${[
            ['cohort', t('Cohort', '队列'), b.cohort_size == null ? '—' : Number(b.cohort_size).toLocaleString()],
            ['evidence', t('Evidence', '证据'), b.evidence_count == null ? '—' : Number(b.evidence_count).toLocaleString()],
            ['missing', t('Missing evidence', '缺失证据'), b.missing_evidence == null ? '—' : String(b.missing_evidence)],
            ['errors', t('Errors', '错误'), b.errors == null ? '—' : String(b.errors)],
          ].map(([key, label, value]) => `<div class="ag-bench-metric ${key}"><span>${label}</span><strong>${esc(value)}</strong></div>`).join('')}
        </div>
        <div class="ag-score-grid mt-12">
          ${dims.map(row => {
            const score = row.subscore == null ? null : Number(row.subscore);
            const width = score == null || Number.isNaN(score) ? 0 : Math.max(0, Math.min(100, score * 100));
            const level = row.level || (score == null ? 'Unscored' : score >= 0.8 ? 'Full' : score >= 0.5 ? 'Partial' : 'Fail');
            return `<div class="ag-score-row">
              <div class="ag-score-top"><span>${esc(row.label || row.id)}</span><span class="mono">${score == null || Number.isNaN(score) ? '—' : score.toFixed(2)} · ${esc(level)}</span></div>
              <div class="ag-score-bar"><span style="width:${width}%"></span></div>
            </div>`;
          }).join('')}
        </div>
      </div>`;
  }

  function presentationSummary(s) {
    if (!s || s.projectKind !== 'canonical9') return '';
    const b = s.benchmark || {};
    const artifactCount = outputCountForStudy();
    const status = runStatusLabel(b.tristate || b.readiness_status || 'analysis_only');
    return `
      <div class="ag-present-brief">
        <div class="ag-present-main">
          <div class="eyebrow">${t('Study brief', '汇报摘要')}</div>
          <div class="ag-present-title">${esc(t(s.name[0], s.name[1]))}</div>
          <div class="ag-present-text">${t('A completed evidence-bound Agent analysis for Figure 2. One clinical question is organized into a plan, figures, scores, and an auditable evidence ledger.', '这是 Figure 2 的一个已完成证据绑定 Agent 分析。一个临床问题在这里被组织为计划、图件、评分和可审计证据账本。')}</div>
        </div>
        <div class="ag-present-grid">
          <div><span>${t('Status', '状态')}</span><strong>${esc(status)}</strong></div>
          <div><span>${t('Evidence', '证据')}</span><strong>${b.evidence_count == null ? '—' : Number(b.evidence_count).toLocaleString()}</strong></div>
          <div><span>${t('Outputs', '产出')}</span><strong>${artifactCount || 0}</strong></div>
          <div><span>${t('Boundary', '边界')}</span><strong>${t('read-only', '只读审阅')}</strong></div>
        </div>
      </div>`;
  }

  function tabOverview() {
    const s = study();
    if (s.empty) {
      return `
      <div class="state-hero empty-state" style="min-height:360px;">
        <div class="glyph">${icon('folder', 28)}</div>
        <div class="st-t">${t('Turn a question into a review-ready draft', '把研究问题变成待审阅草稿')}</div>
        <div class="st-d">${t('An Agent project takes a confirmed question and your extracted cohort and runs an auditable, local analysis — every claim stays locked until evidence checks pass and you sign off. Real mode lists only your own local projects; it never shows fabricated example studies.', 'Agent 项目会把一个已确认的问题与你抽取的队列，运行成一份可审计的本地分析 —— 每条结论在证据检查通过并经你签署前都保持锁定。真实模式只列出你自己的本地项目；绝不显示编造的示例研究。')}</div>
        <div class="ag-empty-steps">
          <span>${icon('extract', 12)} ${t('1 · Extract data', '1 · 抽取数据')}</span>
          <span class="sep">${icon('arrow', 11)}</span>
          <span>${icon('eye', 12)} ${t('2 · Review', '2 · 审阅')}</span>
          <span class="sep">${icon('arrow', 11)}</span>
          <span>${icon('agent', 12)} ${t('3 · Run analysis', '3 · 运行分析')}</span>
        </div>
        <div class="st-actions">
          <button class="btn primary" data-nav="ideas">${icon('target', 14)} ${t('Create from Idea Mining', '从 Idea Mining 创建')}</button>
          <button class="btn" data-ag-see-demo>${icon('flask', 13)} ${t('See a completed example (Demo)', '查看完整示例（演示）')}</button>
          <button class="btn" data-ag-refresh-projects>${icon('refresh', 13)} ${t('Refresh', '刷新')}</button>
        </div>
      </div>`;
    }
    const live = reviewableRunForStudy();
    if (live) requestLiveReview(live);
    const staleBanner = (window.EU_STALE && !agRun.active) ? `
      <div class="stale-banner">
        <span class="sb-ico">${icon('refresh', 16)}</span>
        <div class="grow"><div class="sb-t">${t('Extraction changed since the last run', '自上次运行后抽取已变更')}</div><div class="sb-d">${t('The cohort or modules were edited — runs, outputs and the draft are out of date until you re-run.', '队列或模块被修改 — 运行、产出和草稿在重跑前都已过期。')}</div></div>
        <button class="btn primary" data-ag-runbtn>${icon('refresh', 13)} ${t('Re-run', '重新运行')}</button>
      </div>` : '';
    return `
      ${staleBanner}
      ${renderStructuredQuestion(s)}
      ${presentationSummary(s)}
      ${capabilityHighlights(live, s)}
      ${nextBar()}
      ${s.mode === 'idea' ? `<div class="idea-band mt-16"><span class="ico">${icon('spark', 16)}</span><div><div style="font-weight:600;font-size:13px;">${t('Legacy feasibility idea', '旧可行性想法')}</div><div style="font-size:12px;color:var(--ink-3);margin-top:2px;">${t('New discovery work starts in Idea Mining. Agent Projects only executes confirmed analysis runs.', '新的发现流程从 Idea 挖掘开始。研究项目只执行已确认的分析运行。')}</div></div></div>` : ''}
      <div class="split-320 mt-16" style="grid-template-columns:1fr 300px;">
        <div class="col gap-16">
          ${benchmarkPanel(s)}
          ${planList()}
          ${providerRunPanel()}
        </div>
        ${contextStats()}
      </div>
      <div class="handoff">
        <span class="ho-ico">${icon('spark', 17)}</span>
        <div class="ho-body"><b>${t('Rather drive this study by chat?', '想用对话来推进这项研究?')}</b> ${t('Guided Copilot walks the same plan → run → review → review-ready draft workflow conversationally, then hands the study back here.', '研究引导用对话走同一套 计划 → 运行 → 审阅 → 待核验草稿 的流程,完成后把研究交回这里。')}</div>
        <button class="btn" data-nav="guided">${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')} ${icon('arrow', 13)}</button>
      </div>`;
  }

  function blockContract(block) {
    return `
      <div class="ag-block-contract">
        <div>
          <div class="ag-block-k">${t('What the user confirms', '用户确认')}</div>
          <div class="ag-block-tags">${blockListItems(block.inputs)}</div>
        </div>
        <div>
          <div class="ag-block-k">${t('Planned outputs', '计划产出')}</div>
          <div class="ag-block-tags">${blockListItems(block.outputs)}</div>
        </div>
        <div>
          <div class="ag-block-k">${t('Evidence check', '证据检查')}</div>
          <div class="ag-block-tags evidence">${blockListItems(block.evidence)}</div>
        </div>
      </div>`;
  }
  function compactBlockList(items, limit) {
    const rows = Array.isArray(items) ? items.filter(Boolean) : [];
    const kept = rows.slice(0, limit || 3).map(x => `<span>${esc(x)}</span>`);
    if (rows.length > kept.length) kept.push(`<span>${t('+' + (rows.length - kept.length) + ' more', '另 ' + (rows.length - kept.length) + ' 项')}</span>`);
    return kept.join('');
  }
  function workflowCell(label, items) {
    return `
      <div class="ag-wf-cell">
        <div class="ag-wf-label">${label}</div>
        <div class="ag-wf-listline">${compactBlockList(items, 3)}</div>
      </div>`;
  }

  function workflowRow(block, index, total) {
    return `
      <div class="ag-wf-row ${agBlockSelected === block.id ? 'selected' : ''}" data-ag-block-select="${block.id}">
        <div class="ag-wf-step">
          <div class="ag-wf-n mono">${String(index + 1).padStart(2, '0')}</div>
          <div class="ag-wf-ico">${icon(block.icon || 'layers', 14)}</div>
        </div>
        <div class="ag-wf-main">
          <div class="ag-wf-title">${bi(block.title)}</div>
          <div class="ag-wf-desc">${bi(block.stage)} · ${bi(block.desc)}</div>
        </div>
        ${workflowCell(t('What you confirm', '你确认什么'), block.inputs)}
        ${workflowCell(t('Planned outputs', '计划产出'), block.outputs)}
        ${workflowCell(t('Evidence check', '证据检查'), block.evidence)}
        <div class="ag-wf-actions">
          <button class="icobtn xs" data-ag-block-up="${index}" title="${t('Move up', '上移')}" ${index === 0 ? 'aria-disabled="true"' : ''}>${icon('chevdown', 12)}</button>
          <button class="icobtn xs" data-ag-block-down="${index}" title="${t('Move down', '下移')}" ${index >= total - 1 ? 'aria-disabled="true"' : ''}>${icon('chevdown', 12)}</button>
          <button class="icobtn xs danger" data-ag-block-remove="${index}" title="${t('Remove workflow step', '移除步骤')}">${icon('stop', 12)}</button>
        </div>
      </div>`;
  }

  function tabWorkflow() {
    const s = study();
    if (s.empty) {
      return `
      <div class="state-hero empty-state" style="min-height:320px;">
        <div class="glyph">${icon('layers', 28)}</div>
        <div class="st-t">${t('No planning blocks yet', '还没有规划块')}</div>
        <div class="st-d">${t('Create or select a local Agent project first. Planning blocks are stored per project and stay local to this browser.', '请先创建或选择一个本地 Agent 项目。规划块按项目存储，并只保留在本机浏览器。')}</div>
        <div class="st-actions"><button class="btn primary" data-nav="ideas">${icon('target', 14)} ${t('Create from Idea Mining', '从 Idea Mining 创建')}</button></div>
      </div>`;
    }
    const ids = workflowIds(s);
    const rows = ids.map(blockById).filter(Boolean);
    const selected = blockById(agBlockSelected) || rows[0] || BLOCK_LIBRARY[0];
    const filtered = BLOCK_LIBRARY.filter(block => agBlockFamily === 'all' || block.family === agBlockFamily);
    const inWorkflow = new Set(ids);
    return `
      <div class="ag-block-hero">
        <div>
          <div class="eyebrow">${t('Planning Blocks', '规划块')}</div>
          <div class="ag-block-title">${t('Turn the project into reviewable research steps', '把项目拆成可审阅的研究步骤')}</div>
          <div class="ag-block-sub">${t('Review and design only. These blocks do not change the current /api/jobs/agent-run execution; execution is determined by run type and project configuration.', '仅用于审阅和设计。这些规划块不会改变当前 /api/jobs/agent-run 的执行；实际执行由 run type 与项目配置决定。')}</div>
        </div>
        <div class="row wrap gap-8">
          <button class="btn sm" data-ag-block-pack="nature">${icon('plus', 12)} ${t('Add Nature pack', '加入 Nature 套件')}</button>
          <button class="btn sm ghost" data-ag-block-reset>${icon('refresh', 12)} ${t('Reset default', '恢复默认')}</button>
        </div>
      </div>
      <div class="ag-block-grid">
        <section class="ag-block-panel">
          <div class="ag-block-panel-head">
            <div><div class="ag-block-panel-title">${t('Planning step table', '规划步骤表')}</div><div class="ag-block-panel-sub">${rows.length} ${t('steps planned for this project', '个步骤已规划到此项目')}</div></div>
            <span class="pill ok"><span class="dot"></span>${t('local config', '本地配置')}</span>
          </div>
          <div class="ag-wf-guide">
            <span>${t('Step', '步骤')}</span>
            <span>${t('Research step', '研究步骤')}</span>
            <span>${t('What you confirm', '你确认什么')}</span>
            <span>${t('Planned outputs', '计划产出')}</span>
            <span>${t('Evidence check', '证据检查')}</span>
            <span>${t('Edit', '编辑')}</span>
          </div>
          <div class="ag-wf-list">
            ${rows.length ? rows.map((block, i) => workflowRow(block, i, rows.length)).join('') : `
              <div class="empty-mini" style="min-height:160px;">
                <div>${icon('layers', 22)}</div>
                <h3>${t('No planning steps selected', '尚未选择规划步骤')}</h3>
                <p>${t('Add steps from the library to make the research path explicit.', '从右侧库中加入步骤，让研究路径变得明确。')}</p>
              </div>`}
          </div>
          ${selected ? `
          <div class="ag-block-detail">
            <div class="ag-block-chip">${familyLabel(selected.family)} · ${bi(selected.stage)}</div>
            <div class="ag-block-detail-title">${bi(selected.title)}</div>
            <div class="ag-block-detail-desc">${bi(selected.desc)}</div>
            ${blockContract(selected)}
            <div class="row wrap gap-8 mt-12">
              <button class="btn sm" data-nav="${selected.route || 'agent'}">${icon('arrow', 12)} ${selected.route === 'ideas' ? t('Open Idea Mining', '打开想法挖掘') : t('Open Agent Projects', '打开研究项目')}</button>
              <span class="ag-block-note">${t('Selected step details. Execution remains gated by active export, provider consent, and evidence review.', '当前选中步骤详情。执行仍由 active export、provider 授权和证据审阅约束。')}</span>
            </div>
          </div>` : ''}
        </section>
        <section class="ag-block-panel">
          <div class="ag-block-panel-head">
            <div><div class="ag-block-panel-title">${t('Planning block library', '规划块库')}</div><div class="ag-block-panel-sub">${t('Insert exactly the capability this project needs.', '只插入当前项目需要的能力。')}</div></div>
          </div>
          <div class="ag-block-filters">
            ${BLOCK_FAMILIES.map(([key, label]) => `<button class="chip ${agBlockFamily === key ? 'on' : ''}" data-ag-block-filter="${key}">${bi(label)}</button>`).join('')}
          </div>
          <div class="ag-lib-list">
            ${filtered.map(block => `
              <button class="ag-lib-card ${agBlockSelected === block.id ? 'selected' : ''}" data-ag-block-select="${block.id}" type="button">
                <div class="ag-lib-top">
                  <span class="ag-lib-ico">${icon(block.icon || 'layers', 13)}</span>
                  <span class="ag-lib-title">${bi(block.title)}</span>
                  <span class="ag-lib-family">${familyLabel(block.family)}</span>
                </div>
                <div class="ag-lib-desc">${bi(block.desc)}</div>
                <div class="ag-block-mini">${block.inputs.slice(0, 3).map(x => `<span>${esc(x)}</span>`).join('')}</div>
                <div class="ag-lib-actions">
                  <span class="mono">${esc(block.id)}</span>
                  <span class="btn sm ${inWorkflow.has(block.id) ? 'ghost' : 'primary'}" data-ag-block-add="${block.id}">${inWorkflow.has(block.id) ? t('Added', '已加入') : t('Add planning block', '加入规划块')}</span>
                </div>
              </button>`).join('')}
          </div>
        </section>
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
              <div><div class="run-name mono">${esc(r.run_label || r.run_id || ('run ' + (ri + 1)))}${stale ? ` <span class="jp-stale" title="${esc(runStatusHint('signoff_stale'))}">${icon('alert', 9)} ${t('changed since sign-off', '签署后已改动')}</span>` : ''}</div><div class="run-scope">${esc(runStatusLabel(status))} · ${Number(r.artifact_count || 0)} ${t('artifacts', '产物')}</div></div>
              <div class="row gap-10" style="flex:none;">
                <span class="pill ${stale ? 'bad' : (r.signed ? 'ok' : 'warn')}" style="height:20px;" title="${esc(runStatusHint(stale ? 'signoff_stale' : status))}"><span class="dot"></span>${esc(stale ? runStatusLabel('signoff_stale') : runStatusLabel(status))}</span>
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
      return `<div class="state-hero empty-state"><div class="glyph">${icon('history', 26)}</div><div class="st-t">${t('No runs yet', '尚无运行记录')}</div><div class="st-d">${t('Run the plan from the Overview tab to populate this history. Every run writes a local manifest. Not sure the plan is right yet? Refine it in Idea Mining first.', '在概览页运行计划即可填充历史。每次运行都会写入本地清单。还不确定计划是否合适？可以先回「想法挖掘」细化。')}</div><div class="st-actions"><button class="btn primary" data-ag-tab="overview">${icon('play', 14)} ${t('Go to Overview', '前往概览')}</button><button class="btn" data-nav="ideas">${icon('target', 14)} ${t('Refine in Idea Mining', '回想法挖掘细化')}</button></div></div>`;
    }
    return `
      <div class="card pad" style="padding:16px 18px 8px;">
        <div class="panel-head" style="margin-bottom:6px;">
          <div><div class="panel-title" style="font-size:15px;">${t('Run history', '运行历史')}</div><div class="panel-sub">${t('Local manifests · resumable — nothing leaves your machine.', '本地清单 · 可继续运行 —— 不离开你的机器。')}</div></div>
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
    const publicPayload = scrubDataUrls(data.payload || {});
    const payload = JSON.stringify(publicPayload, null, 2);
    const scan = data.privacy_scan || {};
    return `
    <div class="card pad mt-16">
      <div class="row" style="justify-content:space-between;align-items:baseline;margin-bottom:10px;">
        <div><div class="eyebrow">${t('Artifact viewer', '产物查看器')}</div><div class="panel-title" style="font-size:14px;margin-top:4px;">${esc(artifact.name || agArtifact.name || 'artifact')}</div></div>
        <button class="btn sm" data-ag-artifact-download="${esc(artifact.name || agArtifact.name || '')}">${icon('download', 13)} ${t('Download artifact', '下载产物')}</button>
      </div>
      <div class="row wrap gap-8">
        <span class="pill ${scan.passed ? 'ok' : 'bad'}" style="height:22px;"><span class="dot"></span>${scan.passed ? t('privacy scan clean', '隐私扫描干净') : t('privacy markers found', '发现隐私标记')}</span>
        <span class="pill" style="height:22px;"><span class="dot"></span>${esc((artifact.sha256 || '').slice(0, 12))}</span>
        <span class="pill" style="height:22px;"><span class="dot"></span>${Number(artifact.bytes || 0).toLocaleString()} B</span>
      </div>
      ${artifactStructuredView(artifact.name || agArtifact.name || '', data.payload || {})}
      <details class="ag-raw-json">
        <summary>${t('View raw JSON', '查看原始 JSON')}</summary>
        <pre class="mono">${esc(payload.slice(0, 12000))}${payload.length > 12000 ? '\n...' : ''}</pre>
      </details>
    </div>`;
  }
  function outputBrief(artifacts) {
    const names = (artifacts || []).map(a => a && (a.name || a.relative_path || '')).filter(Boolean);
    const quick = [
      ['figure_gallery.json', t('Open figures', '查看图件'), 'viz'],
      ['benchmark_scorecard.json', t('Open scorecard', '查看记分卡'), 'list'],
      ['evidence_ledger.json', t('Open evidence', '查看证据链'), 'shield'],
    ].filter(([name]) => names.includes(name));
    return `
      <div class="ag-output-brief">
        <div class="ag-output-brief-copy">
          <div class="ag-output-brief-title">${t('Primary review outputs', '主要审阅产出')}</div>
          <div class="ag-output-brief-text">${t('Figures, the benchmark scorecard, and the evidence ledger are surfaced first. File names and hashes stay visible as provenance, not as the main story.', '图件、Benchmark 记分卡和证据账本被放在前面。文件名与哈希保留为溯源信息，不再作为主展示内容。')}</div>
        </div>
        <div class="row gap-8 wrap">
          ${quick.map(([name, label, ic]) => `<button class="btn sm" data-ag-artifact-view="${esc(name)}">${icon(ic, 12)} ${label}</button>`).join('')}
        </div>
      </div>`;
  }
  function featuredFigurePreview(live) {
    if (!live || agArtifact.projectDir !== live.project_dir || agArtifact.name !== 'figure_gallery.json' || agArtifact.loading || agArtifact.error || !agArtifact.data) return '';
    const gallery = figureGallery(agArtifact.data.payload || {});
    if (!gallery) return '';
    return `
      <div class="ag-featured-results">
        <div class="ag-featured-head">
          <div>
            <div class="eyebrow">${t('Result figures', '结果图件')}</div>
            <div class="panel-title" style="font-size:14px;margin-top:3px;">${t('Task-specific figure gallery', '本题图件画廊')}</div>
          </div>
          <span class="pill ok" style="height:22px;"><span class="dot"></span>${t('loaded from local artifact', '来自本地产物')}</span>
        </div>
        ${gallery}
      </div>`;
  }
  function tabOutputs() {
    const s = study();
    const live = reviewableRunForStudy();
    if (live) {
      requestLiveReview(live);
      const review = currentLiveReview(live);
      const artifacts = artifactsForLive(live).slice().sort((a, b) => artifactRank(a.name || a.relative_path || '') - artifactRank(b.name || b.relative_path || ''));
      const integrity = review && review.signoff_integrity ? review.signoff_integrity : null;
      const loadingReview = agReview.projectDir === live.project_dir && agReview.loading;
      return `
      <div class="row" style="justify-content:space-between;align-items:baseline;margin-bottom:14px;">
        <div><div class="panel-title" style="font-size:15px;">${isImportedRun(live, s) ? t('Completed analysis outputs', '已完成分析产出') : t('Outputs', '产出物')}</div><div class="panel-sub">${t('Real local artifacts read from', '真实本地产物读取自')} <span class="mono">${esc(live.project_dir || '')}</span></div></div>
        <div class="row gap-8">
          <span class="pill ${review && review.signoff_stale ? 'bad' : 'warn'}" style="height:22px;" title="${esc(runStatusHint(review && review.signoff_stale ? 'signoff_stale' : (live.gate && live.gate.status ? live.gate.status : 'analysis_only')))}"><span class="dot"></span>${esc(review && review.signoff_stale ? t('stale sign-off', '签署失效') : runStatusLabel(live.gate && live.gate.status ? live.gate.status : 'analysis_only'))}</span>
          <button class="btn sm" data-ag-tab="science" title="${t('Open the same run at the evidence + provenance level', '在证据与溯源层打开同一次运行')}">${icon('shield', 13)} ${t('Evidence & provenance', '证据与溯源')}</button>
          ${artifacts.length ? `<button class="btn sm" data-ag-bundle-download>${icon('download', 13)} ${t('Download bundle', '下载打包')}</button>` : ''}
        </div>
      </div>
      ${loadingReview && !artifacts.length ? `<div class="note info mt-12"><div class="ico">${icon('file', 16)}</div><div class="body"><span class="t">${t('Loading local artifacts', '正在加载本地产物')}</span><span class="d">${t('Reading the whitelisted run folder before showing any output cards.', '先读取白名单运行文件夹,再展示产物卡片。')}</span></div></div>` : ''}
      ${artifacts.length ? `
        ${outputBrief(artifacts)}
        ${capabilityHighlights(live, s)}
        ${featuredFigurePreview(live)}
        <div class="outgrid">
          ${artifacts.map((a, i) => {
            const name = a.name || a.relative_path || '';
            const kind = artifactKind(name);
            const metaBits = [];
            if (a.sha256) metaBits.push((a.sha256 || '').slice(0, 12));
            if (a.bytes != null) metaBits.push(Number(a.bytes || 0).toLocaleString() + ' B');
            const meta = metaBits.join(' · ');
            return `
            <button class="outcard ${agArtifact.name === name ? 'on' : ''}" data-ag-artifact-view="${esc(name)}" type="button">
              <div class="outthumb">${thumb(kind)}</div>
              <div class="outmeta">
                <div class="od ag-out-kicker">${String(i + 1).padStart(2, '0')} · ${esc(artifactCategory(name))}</div>
                <div class="ot">${esc(artifactTitle(name))}</div>
                <div class="outdesc">${esc(artifactSummary(name))}</div>
                <div class="od mono ag-fileline">${esc(name)}${meta ? ' · ' + esc(meta) : ''}</div>
              </div>
            </button>`;
          }).join('')}
        </div>` : (!loadingReview ? `
        <div class="state-hero empty-state">
          <div class="glyph">${icon('file', 28)}</div>
          <div class="st-t">${t('No real output artifacts yet', '还没有真实产物')}</div>
          <div class="st-d">${isImportedRun(live, s) ? t('The imported review package is registered, but the local artifact scan did not return whitelisted files. Open Run history to inspect the source folder.', '已注册导入审阅包，但本地产物扫描没有返回白名单文件。请打开运行历史检查来源文件夹。') : t('This project has not produced Table 1, missingness, ROC, calibration, or evidence files yet. Run the analysis or open a reviewed local run; placeholders are not shown in Real mode.', '这个项目还没有生成 Table 1、缺失审计、ROC、校准或证据文件。请先运行分析,或打开已有本地运行；真实模式不会显示占位产物。')}</div>
          <div class="st-actions">
            <button class="btn primary" data-ag-tab="overview">${icon('play', 14)} ${t('Run preflight', '运行预检')}</button>
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
        <div class="st-d">${t('Outputs are generated only by a local Agent run. This panel will list real JSON/CSV/PNG/HTML files from the run folder and let you open or download them. It will not show demo Table 1, missingness, ROC, or calibration placeholders.', '产物只来自本地 Agent 运行。这里会列出运行文件夹里的真实 JSON/CSV/PNG/HTML 文件,并允许打开或下载；不会显示演示 Table 1、缺失审计、ROC 或校准占位卡片。')}</div>
        <div class="st-actions">
          <button class="btn primary" data-ag-tab="overview">${icon('play', 14)} ${t('Run preflight', '运行预检')}</button>
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
        </div>
      </div>`;
  }

  function tabDraft() {
    const s = study();
    const live = reviewableRunForStudy();
    if (live) {
      requestLiveReview(live);
      const imported = isImportedRun(live, s);
      const gate = live.gate || {};
      const checks = Array.isArray(gate.checks) ? gate.checks : [];
      const passed = checks.filter(c => c.passed).length;
      const review = currentLiveReview(live);
      const readiness = review && review.readiness ? review.readiness : null;
      const signed = !!(review && review.signed);
      const artifacts = review && Array.isArray(review.artifacts) ? review.artifacts : (live.artifacts || []);
      const reviewStatus = readiness ? readiness.status : (gate.status || 'analysis_only');
      const reviewStatusLabel = runStatusLabel(reviewStatus);
      const draft = review && review.artifact_payloads ? review.artifact_payloads['manuscript_draft.json'] : null;
      const draftClaims = draft && Array.isArray(draft.claims) ? draft.claims : [];
      const failures = readiness && Array.isArray(readiness.non_human_failures) ? readiness.non_human_failures : [];
      const integrity = review && review.signoff_integrity ? review.signoff_integrity : null;
      const required = [
        ['evidence_reviewed', t('I reviewed the evidence artifacts', '我已审阅证据产物')],
        ['claims_remain_locked', t('I confirm claims remain locked / not reportable', '我确认论断仍保持锁定 / 不可报告')],
        ['no_patient_rows_persisted', t('I confirm no patient rows are persisted', '我确认未持久化患者行')],
      ];
      const visibleArtifacts = artifacts.slice(0, 6);
      const hiddenArtifacts = artifacts.slice(6);
      const artifactRow = (a) => `
        <div class="ledger-row"><span class="ledger-ico">${icon((a.name || '').includes('signoff') ? 'check' : 'shield', 14)}</span><div><div style="font-weight:600;font-size:12.5px;">${esc(artifactTitle(a.name || a.relative_path || 'artifact'))}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc(a.name || a.relative_path || 'artifact')}${a.sha256 ? ' · ' + esc((a.sha256 || '').slice(0, 12)) : ''}${a.bytes != null ? ' · ' + Number(a.bytes || 0).toLocaleString() + ' B' : ''}</div></div></div>`;
      return `
      <div class="split-320 ag-review-layout" style="grid-template-columns:1fr 300px;">
        <div class="card pad ag-review-main">
          <div class="eyebrow">${t('Evidence check', '证据核验')}</div>
          <div class="panel-title" style="margin-top:4px;">${imported ? t('Read-only review · manuscript not unlocked', '只读审阅 · 不解锁论文草稿') : (signed ? t('Local sign-off recorded · draft locked', '本地签署已记录 · 草稿保持锁定') : t('Preflight complete · draft locked', '预检完成 · 草稿保持锁定'))}</div>
          <div class="panel-sub">${imported ? t('This is a completed benchmark analysis package imported for presentation and evidence review. It is not a new Agent run, and it will not unlock a reportable manuscript draft.', '这是为展示和证据审阅导入的已完成 benchmark 分析包。它不是新的 Agent 运行，也不会解锁可报告论文草稿。') : t('This real run is analysis_only. It wrote bounded local evidence artifacts; human sign-off records review but does not make the draft reportable.', '这次真实运行是 analysis_only。它写入有界本地证据产物；人工签署只记录审阅,不会让草稿可报告。')}</div>
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
            <span class="pill ${signed ? 'ok' : (failures.length ? 'warn' : 'info')}"><span class="dot"></span>${esc(reviewStatusLabel)}</span>
            <div class="grow"><div class="nb-t">${imported ? t('Read-only package loaded', '只读审阅包已加载') : (signed ? t('Human review artifact written', '人工审阅产物已写入') : (readiness.signable ? t('Ready for local human sign-off', '可进行本地人工签署') : t('Blocked before sign-off', '签署前已阻断')))}</div><div class="nb-d">${imported ? t('Use Outputs for figures, scorecard, workflow graph, and evidence ledger.', '请在“产出”页查看图件、记分卡、工作流图谱和证据账本。') : t('Reportable remains false and draft_unlocked remains false in this stage.', '当前阶段 reportable 仍为 false,draft_unlocked 仍为 false。')}</div></div>
          </div>` : `
          <div class="nextbar mt-16 gate" style="background:var(--surface-2);">
            <span class="pill warn" title="${esc(runStatusHint(gate.status || 'analysis_only'))}"><span class="dot"></span>${esc(runStatusLabel(gate.status || 'analysis_only'))}</span>
            <div class="grow"><div class="nb-t">${imported ? t('Review package remains non-reportable', '审阅包保持不可报告') : t('Manuscript claims remain locked', '论文论断保持锁定')}</div><div class="nb-d">${imported ? t('This imported package is for evidence-chain review; it does not create a new manuscript draft.', '导入包用于审阅证据链；不会创建新的论文草稿。') : t("This run's artifacts must pass evidence verification before any draft can unlock.", '只有本次运行的产物通过证据核验后，草稿才可解锁。')}</div></div>
          </div>`}
          ${readiness && failures.length ? `<div class="note warn mt-16"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('Automated check failures', '自动核验失败项')}</span><span class="d">${esc(failures.join(', '))}</span></div></div>` : ''}
          ${imported ? `<div class="ev-detail mt-16">
            <div style="font-weight:600;font-size:12.25px;color:var(--ink);margin-bottom:3px;">${t('Review route', '审阅路径')}</div>
            <div style="font-size:11.5px;color:var(--ink-3);margin-bottom:10px;">${t('The review package centers on output cards, the figure gallery, the benchmark scorecard, and provenance.', '审阅包围绕产出卡片、图件画廊、Benchmark 记分卡和溯源记录组织。')}</div>
            <div class="row gap-8 mt-12">
              <button class="btn primary sm" data-ag-tab="outputs">${icon('viz', 12)} ${t('Open outputs', '打开产出')}</button>
              <button class="btn sm" data-ag-tab="runs">${icon('history', 12)} ${t('Open provenance', '打开溯源')}</button>
            </div>
          </div>` : ''}
          ${!imported && readiness && readiness.signable ? `<div class="ev-detail mt-16">
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
          ${draftClaims.length ? `<details class="ag-review-claims mt-16">
            <summary>${icon('file', 13)} ${t('Evidence-bound draft claims', '证据绑定草稿论断')} <span>${draftClaims.length}</span></summary>
            <div class="ag-review-claim-list">
              ${draftClaims.map(row => `<div class="ag-review-claim"><span class="mono">${esc(row.claim_id || 'claim')}</span><p>${esc(readableArtifactText(row.text || ''))}</p><code>${esc((row.evidence_ids || []).join(', '))}</code></div>`).join('')}
            </div>
          </details>` : ''}
        </div>
        <div class="card pad ag-review-side">
          <div class="eyebrow">${t('Local artifacts', '本地产物')}</div>
          <div class="col gap-8 mt-12">
            ${visibleArtifacts.map(artifactRow).join('')}
          </div>
          ${hiddenArtifacts.length ? `<details class="ag-review-more mt-10">
            <summary>${t('Show all artifacts', '查看全部产物')} <span>${artifacts.length}</span></summary>
            <div class="col gap-8 mt-10">${hiddenArtifacts.map(artifactRow).join('')}</div>
          </details>` : ''}
          <div class="mt-16 mono" style="font-size:11px;color:var(--ink-4);">${passed}/${checks.length} ${t('checks passed', '项校验通过')}</div>
          <div class="mt-8 mono" style="font-size:10.5px;color:var(--ink-4);">${esc(live.project_dir || '')}</div>
        </div>
      </div>`;
    }
    if (realMode()) {
      return `
      <div class="state-hero empty-state">
        <div class="glyph">${icon('shield', 28)}</div>
        <div class="st-t">${t('No reviewable evidence yet', '还没有可审阅的证据')}</div>
        <div class="st-d">${t('Draft review works on a real local run: run the analysis (or open a reviewed run), then confirm its evidence checks here. Demo placeholder checks are not shown in Real mode.', '草稿审阅基于真实本地运行：请先运行分析（或打开已审阅的运行），再在这里确认其证据检查。真实模式不会显示演示占位校验。')}</div>
        <div class="st-actions">
          <button class="btn primary" data-ag-tab="overview">${icon('play', 14)} ${t('Run preflight', '运行预检')}</button>
          <button class="btn" data-ag-tab="runs">${icon('history', 14)} ${t('Open Runs', '打开运行历史')}</button>
        </div>
      </div>`;
    }
    const checks = [
      [t('Cohort denominators resolved', '队列分母已确定'), true, '01 · ' + t('Cohort summary', '队列摘要'), t('n and outcome rates computed from the frozen cohort frame.', 'n 与结局率由冻结的队列帧计算得出。')],
      [t('Per-concept coverage ≥ threshold', '各概念覆盖率 ≥ 阈值'), true, '03 · ' + t('Missingness', '缺失审计'), t('Per-concept coverage table — every module clears the threshold.', '各概念覆盖率表 —— 每个模块均达阈值。')],
      [t('Table 1 reproduces from manifest', 'Table 1 可从清单复现'), true, '02 · Table 1', t('Re-generated row-for-row from the run manifest.', '依据运行清单逐行重新生成。')],
      [t('Model card + metrics attached', '模型卡 + 指标已附'), true, '04 · ROC', t('AUC, calibration and a model card are bound to the run.', 'AUC、校准与模型卡已绑定到本次运行。')],
      [t('Reviewer sign-off', '审阅者签署'), s.signed, null, t('Awaiting a human reviewer signature.', '等待人工审阅者签署。')],
    ];
    const passed = checks.filter(c => c[1]).length;
    return `
    <div class="split-320" style="grid-template-columns:1fr 300px;">
      <div class="col gap-16">
        <div class="card pad">
          <div class="eyebrow">${t('Evidence check', '证据核验')} · ${t('demo preview', '演示预览')}</div>
          <div class="panel-title" style="margin-top:4px;">${s.signed ? t('Manuscript draft unlocked (demo)', '论文草稿已解锁（演示）') : t('Manuscript draft is locked until checks pass', '在校验通过前论文草稿保持锁定')}</div>
          <div class="panel-sub">${t('Illustrative demo of evidence verification. In Real mode, sign-off records a human review but never auto-unlocks a reportable manuscript — claims stay locked by STRICT evidence.', '这是证据核验环节的演示。真实模式下，签署只记录人工审阅，绝不会自动解锁可报告的稿件 —— 结论始终受 STRICT 证据约束。')}</div>
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
                <div style="font-size:11.5px;color:var(--ink-3);margin-bottom:10px;">${t('Sign-off is the last release check. Confirm each point, then the draft unlocks.', '签署是最后一道放行检查。逐项确认后，草稿即解锁。')}</div>
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
              ? `<button class="btn primary" aria-disabled="true" title="${t('Demo preview — no manuscript file in demo mode', '演示预览 —— 演示模式没有稿件文件')}">${icon('wand', 13)} ${t('Open manuscript (demo)', '打开论文（演示）')}</button>`
              : `<button class="btn primary" data-ag-signoff aria-disabled="true">${icon('check', 13)} ${t('Sign off & draft', '签署并撰稿')}</button>`}
          </div>
        </div>
        ${s.signed ? `
        <div class="card pad">
          <div class="eyebrow">${t('Draft versions', '草稿版本')}</div>
          <div class="mt-12">
            <div class="draftver"><span class="dv-badge">v0.2</span><div class="grow"><div style="font-weight:600;font-size:12.75px;">${t('Methods + Results + Limitations', '方法 + 结果 + 局限')}</div><div style="font-size:11px;color:var(--ink-4);">${t('example version · demo', '示例版本 · 演示')}</div></div><button class="btn sm" aria-disabled="true" title="${t('Demo preview', '演示预览')}">${icon('eye', 13)} ${t('Open', '打开')}</button></div>
            <div class="draftver"><span class="dv-badge">v0.1</span><div class="grow"><div style="font-weight:600;font-size:12.75px;">${t('Methods + Results', '方法 + 结果')}</div><div style="font-size:11px;color:var(--ink-4);">${t('example version · demo', '示例版本 · 演示')}</div></div><button class="btn sm ghost" aria-disabled="true" title="${t('Demo preview', '演示预览')}">${icon('history', 13)} ${t('Diff', '对比')}</button></div>
          </div>
        </div>` : ''}
      </div>
      <div class="card pad" style="align-self:start;">
        <div class="eyebrow">${t('Output bundle', '产出打包')} · ${t('demo', '演示')}</div>
        <div class="col gap-8 mt-12">
          ${[[t('figures', '图'), 'png + svg', 'viz'], [t('tables', '表'), 'csv + tex', 'list'], [t('Evidence ledger', '证据账本'), 'json manifest', 'shield'], [t('Repro code', '复现代码'), 'py + notebook', 'file']].map(([ti, d, ic]) => `
            <div class="ledger-row"><span class="ledger-ico">${icon(ic, 14)}</span><div><div style="font-weight:600;font-size:12.5px;">${ti}</div><div style="font-size:11px;color:var(--ink-4);">${d}</div></div></div>`).join('')}
        </div>
        <button class="btn sm block mt-16" aria-disabled="true" title="${t('Demo preview — run a real analysis to export a bundle', '演示预览 —— 运行真实分析后才能导出打包')}">${icon('download', 13)} ${t('Export bundle', '导出打包')}</button>
      </div>
    </div>`;
  }

  function tabBody() {
    const s = study();
    if (agTab === 'workflow') return tabWorkflow();
    if (agTab === 'science') {
      const live = reviewableRunForStudy();
      return window.EU_AGENT_SCIENCE
        ? window.EU_AGENT_SCIENCE.render({ live: live, study: s, repaint: repaintBody })
        : `<div class="card pad"><div class="panel-title">${t('Evidence view unavailable', '证据视图不可用')}</div></div>`;
    }
    if (agTab === 'runs') return tabRuns();
    if (agTab === 'outputs') return tabOutputs();
    if (agTab === 'notes') return tabNotes();
    if (agTab === 'draft') return tabDraft();
    return tabOverview();
  }
  function focusAgentBody() {
    window.requestAnimationFrame(() => {
      const detail = document.querySelector('#agHost .ag-detail');
      if (detail && detail.scrollIntoView) detail.scrollIntoView({ block: 'start', behavior: 'auto' });
    });
  }

  function agShell() {
    const listCollapsed = agentListCollapsed();
    return `
    <div class="ag-wrap ${listCollapsed ? 'list-collapsed' : 'list-open'}" data-ag-list-state="${listCollapsed ? 'collapsed' : 'open'}">
      ${studyList()}
      <div class="ag-detail">
        ${detailHead()}
        ${tabsRow()}
        <div class="ag-body" id="agTabPanel" role="tabpanel" aria-labelledby="agTab-${agTab}" tabindex="0">${tabBody()}</div>
      </div>
    </div>`;
  }

  /* ---------------- run animation ---------------- */
  function startRun() {
    const s = study();
    if (s.empty) {
      agRun.error = t('No local research project is selected. Create an Agent project from Idea Mining first.', '尚未选择本地研究项目。请先从 Idea Mining 创建 Agent 项目。');
      repaintBody();
      return;
    }
    const contextBlocker = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.runBlocker
      ? window.EU_AGENT_STUDY_CONTEXT.runBlocker(s)
      : '';
    if (contextBlocker) {
      agRun.error = contextBlocker;
      repaintBody();
      return;
    }
    if (seedGateBlocksRun(s)) {
      agRun.error = seedGateBlockerText(s);
      repaintBody();
      return;
    }
    const src = exportSourceForStudy(s);
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
    const contextBlocker = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.runBlocker
      ? window.EU_AGENT_STUDY_CONTEXT.runBlocker(s)
      : '';
    if (contextBlocker) {
      agRun.error = contextBlocker;
      repaintBody();
      return;
    }
    if (s.empty) {
      agRun.error = t('No local research project is selected. Create an Agent project from Idea Mining first.', '尚未选择本地研究项目。请先从 Idea Mining 创建 Agent 项目。');
      repaintBody();
      return;
    }
    let runToken = agRunChannel.start({
      surface: 'agent',
      study_id: s.id,
      context_id: s.studyContext && s.studyContext.id,
      question: s.question && s.question[0],
      source_path: src.path,
      study_mode: s.mode,
      run_type: opts.runType || 'preflight',
      provider: opts.provider || 'mock',
      project_seed_dir: s.ideaSeed && s.ideaSeed.project_dir,
    });
    closeRunStream();
    agRun = { active: true, prog: 0, timer: null, es: null, jobId: null, step: t('Submitting local run', '提交本地运行'), error: null, result: null, warning: null };
    agReview = { projectDir: null, loading: false, error: null, data: null, signing: false };
    agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
    agHistory = { studyId: null, loading: false, error: null, data: null };
    window.EU_AGENT_RUN_REVIEW = null;
    agTab = 'overview';
    window.EU_STALE = false;  // a fresh run consumes the current inputs
    repaintBody();
    const contextReady = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.persistForRun
      ? window.EU_AGENT_STUDY_CONTEXT.persistForRun(s)
      : Promise.resolve(s.studyContext || null);
    contextReady.then(boundContext => {
      runToken = agRunChannel.bind(runToken, { context_id: boundContext && boundContext.id });
      return window.EU_API.startAgentRun({
        path: runToken.source_path,
        study_id: runToken.study_id,
        mode: runToken.study_mode,
        project_seed_dir: runToken.project_seed_dir || undefined,
        project_root: runToken.project_seed_dir ? `${runToken.project_seed_dir}/runs` : undefined,
        run_type: runToken.run_type,
        llm_provider: runToken.provider,
        external_llm_opt_in: !!opts.externalOptIn,
        question: runToken.question,
        study_context_id: runToken.context_id || undefined,
      });
    }).then(r => {
      runToken = agRunChannel.bind(runToken, { job_id: r.job_id, context_revision: r.study_context_revision });
      const isCurrent = agRunChannel.isCurrent(runToken);
      if (isCurrent) agRun.jobId = runToken.job_id;
      if (isCurrent) agRun.warning = window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.submissionWarning
        ? window.EU_AGENT_STUDY_CONTEXT.submissionWarning(r)
        : null;
      if (runToken.context_id && window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.markContextRunning) {
        window.EU_AGENT_STUDY_CONTEXT.markContextRunning(runToken.context_id, runToken.job_id, runToken.context_revision);
      }
      if (isCurrent) agRun.step = t('Connected to job stream', '已连接任务流');
      rememberAgentJob({
        job_id: runToken.job_id,
        study_id: runToken.study_id,
        source_path: runToken.source_path,
        run_type: runToken.run_type,
        provider: runToken.provider,
        external_llm_opt_in: !!opts.externalOptIn,
        study_context_id: runToken.context_id || undefined,
        study_context_revision: runToken.context_revision,
      });
      attachAgentJobStream(runToken);
    }).catch(err => finishRealRun(runToken, 'failed', null, err.message || String(err)));
  }

  function finishRealRun(runToken, status, result, error) {
    const isCurrent = agRunChannel.isCurrent(runToken);
    if (runToken && runToken.context_id && runToken.job_id && window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.markContextFinished) {
      window.EU_AGENT_STUDY_CONTEXT.markContextFinished(
        runToken.context_id,
        status,
        result,
        runToken.job_id,
        result && result.study_context_revision,
      );
    }
    if (status !== 'running' && runToken && runToken.job_id) clearRememberedAgentJob(runToken.job_id, runToken.study_id);
    if (!isCurrent) return;
    closeRunStream();
    const s = allStudies().find(row => row.id === runToken.study_id) || study();
    agRun.active = false;
    agRun.prog = 1;
    if (status === 'done' && result) {
      agRun.result = result;
      agRun.error = null;
      agRun.reconnectable = false;
      if ((!s.studyContext || !window.EU_AGENT_STUDY_CONTEXT) && study().id === runToken.study_id) window.EU_AGENT_LAST_RUN = result;
      agReview = { projectDir: null, loading: false, error: null, data: null, signing: false };
      agHistory = { studyId: null, loading: false, error: null, data: null };
      window.EU_AGENT_RUN_REVIEW = null;
      if (s.mode === 'analysis') {
        s.status = result.gate && result.gate.status === 'blocked' ? 'review_blocked' : 'gate';
        s.stage = 3;
      }
      else { s.status = 'draft'; s.stage = 2; }
      s.runs.unshift([
        result.run_label || ('run ' + (result.run_id || '').slice(-6)),
        result.run_type === 'full'
          ? [t('Provider plan and draft scaffold', 'provider 计划与草稿骨架'), t('Provider plan and draft scaffold', 'provider 计划与草稿骨架')]
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
    agRunChannel.clear(runToken);
    repaintBody();
  }

  function startDemoRun() {
    const s = study();
    closeRunStream();
    agRun.active = true; agRun.prog = 0; agRun.step = null; agRun.error = null; agRun.result = null; agRun.jobId = null; agRun.warning = null;
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
      closeRunStream();
      agRunChannel.clear();
      agRun = { active: false, prog: 0, timer: null, es: null, jobId: null, step: null, error: null, result: null, warning: null };
      agResumeProbe = { loading: false, checkedJobId: null };
      window.EU_AGENT_RUN_REVIEW = null;
      agSel = b.dataset.agSel;
      if (window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.has(agSel)) {
        window.EU_AGENT_STUDY_CONTEXT.activate(agSel).catch(error => console.warn('[EasyICU] StudyContext activation failed:', error));
      }
      agTab = 'overview'; repaintBody(); maybeRestoreAgentJob(); focusAgentBody();
    }));
    host.querySelectorAll('[data-ag-tab]').forEach(b => b.addEventListener('click', () => { agTab = b.dataset.agTab; repaintBody(); focusAgentBody(); }));
    const tabList = host.querySelector('[data-ag-tabs]');
    if (tabList) tabList.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      const tabs = Array.from(tabList.querySelectorAll('[role="tab"]'));
      const current = tabs.indexOf(document.activeElement);
      if (current < 0 || !tabs.length) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0
        : (event.key === 'End' ? tabs.length - 1
          : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length);
      agTab = tabs[next].dataset.agTab;
      repaintBody();
      const refreshed = document.getElementById(`agTab-${agTab}`);
      if (refreshed) refreshed.focus();
      focusAgentBody();
    });
    host.querySelectorAll('[data-ag-block-filter]').forEach(b => b.addEventListener('click', () => { agBlockFamily = b.dataset.agBlockFilter || 'all'; repaintBody(); }));
    host.querySelectorAll('[data-ag-block-select]').forEach(b => b.addEventListener('click', () => { agBlockSelected = b.dataset.agBlockSelect || agBlockSelected; repaintBody(); }));
    host.querySelectorAll('[data-ag-block-add]').forEach(b => b.addEventListener('click', e => {
      e.stopPropagation();
      addWorkflowBlock(b.dataset.agBlockAdd);
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-block-pack]').forEach(b => b.addEventListener('click', () => {
      if ((b.dataset.agBlockPack || '') === 'nature') addWorkflowPack(NATURE_PACK);
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-block-reset]').forEach(b => b.addEventListener('click', () => { resetWorkflowBlocks(); repaintBody(); }));
    host.querySelectorAll('[data-ag-block-up]').forEach(b => b.addEventListener('click', e => {
      e.stopPropagation();
      if (b.getAttribute('aria-disabled') === 'true') return;
      moveWorkflowBlock(Number(b.dataset.agBlockUp || 0), -1);
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-block-down]').forEach(b => b.addEventListener('click', e => {
      e.stopPropagation();
      if (b.getAttribute('aria-disabled') === 'true') return;
      moveWorkflowBlock(Number(b.dataset.agBlockDown || 0), 1);
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-block-remove]').forEach(b => b.addEventListener('click', e => {
      e.stopPropagation();
      removeWorkflowBlock(Number(b.dataset.agBlockRemove || 0));
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-toggle-list]').forEach(b => b.addEventListener('click', () => {
      agListMode = agentListCollapsed() ? 'open' : 'focus';
      repaintBody();
      focusAgentBody();
    }));
    if (window.EU_AGENT_SCIENCE) {
      window.EU_AGENT_SCIENCE.wire(root, { live: reviewableRunForStudy(), study: study(), repaint: repaintBody });
    }
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
      const selectedId = study().id;
      agRun.error = null;
      agRun.active = true;
      agRun.step = t('Checking server job state', '正在检查服务端任务状态');
      repaintBody();
      window.EU_API.loadJobSnapshot(jobId).then(snapshot => {
        if (study().id !== selectedId) return;
        restoreAgentJobFromSnapshot({ job_id: jobId, study_id: selectedId }, snapshot);
      }).catch(err => {
        if (study().id !== selectedId) return;
        agRun.active = false;
        agRun.error = err.message || String(err);
        clearRememberedAgentJob(jobId, selectedId);
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
      const src = exportSourceForStudy(study());
      if (!src) return;
      startRealRun(src, { runType: 'full', provider: agProvider.provider, externalOptIn: true });
      // "for this run only": consume the consent so the next full run must be
      // re-authorized instead of silently reusing stale approval.
      agProvider = Object.assign({}, agProvider, { consent: false });
      repaintBody();
    }));
    host.querySelectorAll('[data-ag-history-refresh]').forEach(b => b.addEventListener('click', () => requestRunHistory(true)));
    host.querySelectorAll('[data-ag-refresh-projects]').forEach(b => b.addEventListener('click', () => requestIdeaAgentProjects(true)));
    host.querySelectorAll('[data-ag-see-demo]').forEach(b => b.addEventListener('click', () => { if (window.setDataMode) window.setDataMode('demo'); }));
    host.querySelectorAll('[data-ag-open-seed-run]').forEach(b => b.addEventListener('click', () => {
      const projectDir = b.dataset.agOpenSeedRun || '';
      if (!projectDir || !window.EU_API || !window.EU_API.loadAgentRunReview) return;
      agReview = { projectDir: projectDir, loading: true, error: null, data: null, signing: false };
      agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
      agTab = 'outputs';
      repaintBody();
      window.EU_API.loadAgentRunReview(projectDir).then(data => {
        openReview(data, 'outputs');
        repaintBody();
      }).catch(err => {
        agReview = { projectDir: projectDir, loading: false, error: err.message || String(err), data: null, signing: false };
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-history-open]').forEach(b => b.addEventListener('click', () => {
      const rows = agHistory.data && Array.isArray(agHistory.data.runs) ? agHistory.data.runs : [];
      const row = rows[Number(b.dataset.agHistoryOpen || -1)];
      if (!row || !row.project_dir || !window.EU_API || !window.EU_API.loadAgentRunReview) return;
      agReview = { projectDir: row.project_dir, loading: true, error: null, data: null, signing: false };
      agTab = 'draft';
      repaintBody();
      window.EU_API.loadAgentRunReview(row.project_dir).then(data => {
        openReview(data, data && data.run_type === 'canonical9_import' ? 'outputs' : 'draft');
        repaintBody();
      }).catch(err => {
        agReview = { projectDir: row.project_dir, loading: false, error: err.message || String(err), data: null, signing: false };
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-artifact-view]').forEach(card => card.addEventListener('click', () => {
      const live = reviewableRunForStudy();
      requestArtifact(live, card.dataset.agArtifactView);
    }));
    host.querySelectorAll('[data-ag-artifact-jump]').forEach(b => b.addEventListener('click', () => {
      const live = reviewableRunForStudy();
      agTab = 'outputs';
      requestArtifact(live, b.dataset.agArtifactJump);
    }));
    host.querySelectorAll('[data-ag-artifact-download]').forEach(b => b.addEventListener('click', e => {
      e.stopPropagation();
      const live = reviewableRunForStudy();
      if (!live || !window.EU_API || !window.EU_API.downloadAgentRunArtifact) return;
      window.EU_API.downloadAgentRunArtifact(live.project_dir, b.dataset.agArtifactDownload).catch(err => {
        agArtifact = { projectDir: live.project_dir, name: b.dataset.agArtifactDownload, loading: false, error: err.message || String(err), data: agArtifact.data };
        repaintBody();
      });
    }));
    host.querySelectorAll('[data-ag-bundle-download]').forEach(b => b.addEventListener('click', () => {
      const live = reviewableRunForStudy();
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
      // Gate every sign-off trigger in the demo draft panel (detail + bottom
      // bar) on the confirmation checkboxes, not just the inline one.
      const gatedButtons = [...host.querySelectorAll('[data-ag-signoff]')];
      const hint = host.querySelector('.rtodo-hint');
      const sync = () => {
        const all = rtodos.every(c => c.checked);
        gatedButtons.forEach(b => b.setAttribute('aria-disabled', all ? 'false' : 'true'));
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
    /* No per-screen guide action: the removed button opened the SAME dock as
       the topbar 'Page guide' button beside it, reading as a second
       agent-specific help system that didn't exist. The dock already greets
       agent-specifically via its CTX.agent entry. */
    rail() {
      const s = study();
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Projects', '项目')}</span><span class="pill ok" style="height:20px;"><span class="dot"></span>${allStudies().length}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Active', '当前')}</span><span class="vv" style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(t(s.name[0], s.name[1]))}</span></div>
          <div class="setup-row"><span class="k">${t('Mode', '模式')}</span><span class="vv">${t('Analysis', '分析')}</span></div>
          <div class="setup-row"><span class="k">${t('Blocks', '块')}</span><span class="vv">${workflowBlocks(s).length}</span></div>
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
      if (window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.take) window.EU_GUIDED_HANDOFF.take('agent');
      const guidedNote = window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.noteHtml ? window.EU_GUIDED_HANDOFF.noteHtml('agent') : '';
      return `
      ${guidedNote}
      <div class="page-head" style="margin-bottom:16px;">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:16px;">
          <div>
            <div class="eyebrow">${t('Agent Projects · 研究项目', '研究项目 · Agent Projects')}</div>
            <h1 style="margin-top:6px;">${t('Agent Projects', '研究项目')}</h1>
            <p class="lead">${t('A workspace of research projects. Each study has a workflow, its own runs, outputs, and a review-ready draft — all auditable, all local.', '一个研究项目工作台。每个研究都有自己的工作流、运行记录、产出和待核验草稿 —— 全程可审计、全程本地。')}</p>
            <div style="font-size:11.5px;color:var(--ink-4);margin-top:9px;">${t('Key terms', '关键术语')}: ${window.gloss('denominator', t('denominator', '分母'))} · ${window.gloss('concept', t('concept', '概念'))} · ${window.gloss('SOFA')}</div>
          </div>
        </div>
      </div>
      <div id="agHost">${agShell()}</div>`;
    },
    afterRender(root) {
      wire(root);
      if (window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.hydrate) window.EU_AGENT_STUDY_CONTEXT.hydrate();
      requestIdeaAgentProjects();
      maybeRestoreAgentJob();
    },
  };

  if (window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.subscribe) {
    window.EU_AGENT_STUDY_CONTEXT.subscribe(context => {
      if (!context || !context.id) return;
      if (agSel && agSel !== context.id) {
        closeRunStream();
        agRunChannel.clear();
        agRun = { active: false, prog: 0, timer: null, es: null, jobId: null, step: null, error: null, result: null, warning: null };
        agResumeProbe = { loading: false, checkedJobId: null };
      }
      agSel = context.id;
      if (location.hash === '#agent') repaintBody();
    });
  }
})();
