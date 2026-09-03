/* Screen: Research project monitor (legacy route id: #agent).
   Goal: keep project state, runs, evidence, artifacts, and review visible.
   Research requirements, model setup, planning, and run initiation belong to
   Guided Copilot; this route must not become a second agent conversation UI.
     • Left rail = a persistent list of studies (projects), like chat sessions.
     • Each study carries a linked cohort, its own run history, outputs, and
       draft versions. Idea mining lives in the separate #ideas workspace.
     • The pipeline + evidence checks are drawn explicitly per study.
   Outputs fail closed: Real mode lists only whitelisted local artifacts. */
(function () {
  const { esc } = window.EU_HTML;
  const S = (window.SCREENS = window.SCREENS || {});
  const RUN_HISTORY_VIEW = window.EU_AGENT_RUN_HISTORY_VIEW;

  /* Fixture data + pure renderers live in screens-agent-render.js
     (owner-file carve-out; loads before this file). Rebind the names
     so call sites in this IIFE stay unchanged. */
  const R = window.AGENT_RENDER || {};
  const {
    DEMO_STUDIES,
    runStatusLabel, runStatusHint, gateCheckLabel, readableArtifactText, firstValue, fmtCount,
    artifactKind, artifactTitle, artifactCategory, artifactSummary, artifactRank, defaultArtifactName,
    thumb, scrubDataUrls, figureGallery, artifactStructuredView,
  } = R;

  let agSel = null;
  let agTab = 'overview';
  let agEvOpen = -1;   // expanded evidence-gate check index
  let agRun = { active: false, prog: 0, timer: null, es: null, jobId: null, step: null, error: null, errorRemedies: '', result: null, warning: null };
  let agReview = { projectDir: null, loading: false, error: null, data: null, signing: false };
  let agHistory = { studyId: null, loading: false, error: null, data: null };
  let agArtifact = { projectDir: null, name: null, loading: false, error: null, data: null };
  let agIdeaProjects = { loading: false, error: null, data: null };
  let agListMode = 'auto'; // auto, open, or focus
  let agGuidedHandoffError = '';
  const AG_JOB_KEY = 'easyicu.agent.activeJob.v1';
  let agResumeProbe = { loading: false, checkedJobId: null };
  const AG_FOCUS_TABS = new Set(['science', 'runs', 'outputs', 'notes', 'draft']);
  const agRunChannel = window.EU_AGENT_STUDY_CONTEXT.createRunChannel();
  const agJobMemory = window.EU_AGENT_STUDY_CONTEXT.createJobMemory(localStorage, AG_JOB_KEY);

  /* continuity: Copilot can land a completed run */
  window.__euAgentPreset = function () { agSel = 'sepsis'; agTab = 'outputs'; };

  function seedStudy(row) {
    const title = window.EU_PRODUCT_LABELS.projectTitle(
      row.title,
      row.question || t('Research idea', '研究想法'),
    );
    const q = row.question || title;
    const pre = row.pre_experiment_summary || {};
    const source = row.source || {};
    const seedRuns = Array.isArray(row.runs) ? row.runs : [];
    const reviewRun = seedRuns.find(r => r && r.project_dir) || null;
    // Historical evaluation imports remain readable, but they are not a
    // product-level project type.  Users see the same study/run/evidence
    // workflow as every other completed research project; benchmark scoring
    // stays in its artifact bundle and external experiment harness.
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
          `completed run · ${pre.status || 'imported'} · ${pre.feature_count || 0} evidence`,
          `已完成运行 · ${pre.status || '已导入'} · ${pre.feature_count || 0} 条证据`,
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
      projectKind: imported ? 'analysis' : 'idea',
      readOnlyImport: imported,
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
  /* The gate decides from typed conditions and now ships stable codes; this
     used to run regular expressions over the backend's English prose to work
     out what to tell the user, which meant a reworded sentence silently lost
     its remedy. Codes only — an unknown one yields no remedy rather than a
     guessed one. */
  function remedyFor(code) {
    const api = window.EU_GATE_REMEDY;
    return api ? api.forCode(code) : null;
  }
  function gateRemedyHtml(remedies) {
    const api = window.EU_GATE_REMEDY;
    return api ? api.render(remedies) : '';
  }
  function seedGateCodes(s) {
    const gate = seedExecutionGate(s);
    if (s && s.ideaSeed && !gate) return ['seed_gate_missing'];
    const codes = gate && Array.isArray(gate.blocker_codes) ? gate.blocker_codes : [];
    return codes.filter(Boolean).map(String);
  }
  function seedGateRemedies(s) {
    const api = window.EU_GATE_REMEDY;
    if (!api) return [];
    return seedGateCodes(s).map(api.forCode).filter(Boolean);
  }
  function seedGateBlockerText(s) {
    const gate = seedExecutionGate(s);
    if (s && s.ideaSeed && !gate) return t('Continue in Guided Copilot so it can refresh the project readiness checks.', '请回到研究引导，让 Copilot 刷新项目就绪检查。');
    // Plain text — callers escape at their own HTML insertion point.
    // Returning esc() here double-escaped agRun.error, which is escaped again
    // by the run-error nextbar.
    const blockers = gate && Array.isArray(gate.blockers) ? gate.blockers : [];
    return blockers.length
      ? blockers.join(' · ')
      : t('Confirm the outstanding study setup in Guided Copilot before continuing.', '继续前请在研究引导中确认尚未完成的研究配置。');
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
        t('Create and configure a study in Guided Copilot. This page displays project status after the project exists.', '请在研究引导中创建并配置研究；项目建立后，本页负责展示项目状态。'),
        t('Create and configure a study in Guided Copilot. This page displays project status after the project exists.', '请在研究引导中创建并配置研究；项目建立后，本页负责展示项目状态。'),
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
  function monitorViewState(studies) {
    if (studies.length) return 'ready';
    if (agIdeaProjects.error) return 'error';
    if (agIdeaProjects.loading || (realMode() && !agIdeaProjects.data)) return 'loading';
    return 'empty';
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
    return RUN_HISTORY_VIEW.projectFolder(agHistory, s, displayPath, { empty: t('No local project folder yet', '还没有本地项目文件夹'), pending: t('Created when this study is first run', '首次运行时创建') });
  }
  function studyBadgeLabel(s) {
    if (s && s.readOnlyImport) return t('Read-only result', '只读结果');
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
      ? t('This list only shows local monitored projects and completed runs written on this machine.', '这里仅显示本机写入的受监控项目和已完成运行。')
      : t('Demo mode includes example projects for exploration. Your own projects appear here after Guided Copilot creates a governed local project.', '演示模式会放入可探索的示例项目。你自己的项目会在研究引导创建受治理的本地项目后出现在这里。');
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
      if (preferred && studies.some(row => row.id === preferred)) agSel = preferred;
      else if (!studies.some(row => row.id === agSel) && studies.length) agSel = studies[0].id;
      maybeRestoreAgentJob();
      requestRunHistory();
      if (window.__euRender) window.__euRender();
      else repaintBody();
    }).catch(err => {
      agIdeaProjects = { loading: false, error: err.message || String(err), data: null };
      if (window.__euRender) window.__euRender();
      else repaintBody();
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
  const historyRowsForStudy = s => RUN_HISTORY_VIEW.rows(agHistory, s);
  const monitorRunCount = s => RUN_HISTORY_VIEW.count(agHistory, s, realMode());
  const historyRunForStudy = s => RUN_HISTORY_VIEW.run(agHistory, s);
  function importedRunForStudy(s) {
    if (!s || !s.reviewProjectDir || !s.readOnlyImport) return null;
    const row = Array.isArray(s.seedRuns) ? s.seedRuns.find(r => r && r.project_dir === s.reviewProjectDir) : null;
    const b = s.benchmark || {};
    return {
      run_id: (row && row.label) || b.task_id || s.id,
      run_label: (row && row.label) || b.task_id || s.id,
      study_id: s.id,
      mode: s.mode || 'analysis',
      run_type: 'imported_review',
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
    const selected = study();
    return live || historyRunForStudy(selected) || importedRunForStudy(selected);
  }
  function isImportedRun(live, s) {
    return !!(live && (live.imported || (s && s.readOnlyImport)));
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
    /* This card exists to declare which data a run consumed — the one place
       that must not guess. It used to test the user's own question text for
       "mimic-iv" and, on a hit, announce the MIMIC-IV canonical universe: ask
       "does this replicate the MIMIC-IV finding?" while your active export is
       eICU and the provenance card would name the wrong database. The run
       artifacts declare the scope; when they do not, say so rather than
       infer it from prose. */
    const crossScope = s.id === 'crossdb' || !!s.planOnly;
    const crossdbSelection = s.studyContext && s.studyContext.crossdb_selection || {};
    const selectedSources = Array.isArray(crossdbSelection.sources) ? crossdbSelection.sources : [];
    const selectedScope = selectedSources.map(row => row && (row.label || row.database)).filter(Boolean).join(' + ');
    const declaredScope = firstValue(
      selectedScope,
      s.ideaSeed && s.ideaSeed.cohort,
      context.source && context.source.database,
      score.database_scope,
    );
    const scopeDeclared = !!declaredScope || crossScope;
    const inferredScope = declaredScope
      || (crossScope ? t('Cross-DB comparison workspace', '跨库对比工作台') : t('Not declared by this run', '本次运行未声明'));
    const cohortSize = firstValue(score.cohort_size, context.summary && context.summary.stays, cohort.summary && cohort.summary.stays, cohort.cohort && cohort.cohort.entities, s.benchmark && s.benchmark.cohort_size);
    const modules = firstValue(
      context.summary && context.summary.modules,
      cohort.summary && cohort.summary.modules,
      crossScope && s.studyContext && Array.isArray(s.studyContext.modules) ? s.studyContext.modules.length : null,
    );
    const crossCount = crossScope && Number.isInteger(crossdbSelection.source_count)
      ? crossdbSelection.source_count
      : null;
    // Was: regex the scope LABEL for "cross|multi|six|database", which made any
    // single-database scope whose name contained the word "database" render as
    // a multi-database context. The study itself knows.
    const isCross = crossScope;
    return `
      <div class="ag-cap-card cross">
        <div class="ag-cap-head">
          <div>
            <div class="eyebrow">${t('Cross-data scope', '跨数据范围')}</div>
            <div class="ag-cap-title">${isCross ? t('Multi-database analysis context', '多数据库分析上下文') : t('Data scope is declared before claims', '下结论前先声明数据范围')}</div>
          </div>
          <span class="pill ${isCross ? 'ok' : (scopeDeclared ? 'info' : 'warn')}" style="height:22px;"><span class="dot"></span>${isCross ? t('cross-db', '跨库') : (scopeDeclared ? t('scoped', '已限定') : t('undeclared', '未声明'))}</span>
        </div>
        <div class="ag-cap-text">${t('The project monitor shows which data context a run consumed. Cross-DB comparisons are prepared in the Cross-DB workspace, then passed into the same evidence-bound review path.', '项目监控页会显示一次运行消费了哪个数据上下文。跨库比较先在 Cross-DB 工作台准备，再进入同一套证据绑定审阅路径。')}</div>
        <div class="ag-cap-metrics">
          <div><span>${t('Current scope', '当前范围')}</span><strong>${esc(inferredScope)}</strong></div>
          <div><span>${t('Denominator', '分母')}</span><strong>${fmtCount(cohortSize)}</strong></div>
          <div><span>${t('Modules', '模块')}</span><strong>${modules == null ? '—' : fmtCount(modules)}</strong></div>
          <div><span>${t('Cross-DB exports', '跨库导出')}</span><strong>${crossCount == null ? '—' : fmtCount(crossCount)}</strong></div>
        </div>
        <div class="ag-cap-actions">
          <button class="btn sm" data-nav="crossdb">${icon('benchmark', 12)} ${t('Open Cross-DB workspace', '打开跨库工作台')}</button>
          <span class="ag-cap-note">${isCross ? t('This project is already using a multi-database context.', '这个项目已经使用多数据库上下文。') : t('The current run has a declared data scope; use Cross-DB when the scientific question requires a multi-database comparison.', '当前运行已有明确的数据范围；科学问题需要多数据库比较时可进入跨库工作台。')}</span>
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
    const selectedId = s.id;
    agHistory = { studyId: selectedId, loading: true, error: null, data: null };
    const seedDir = (s.ideaSeed && s.ideaSeed.project_dir) || undefined;
    window.EU_API.loadAgentRunHistory({ study_id: selectedId, limit: 50, project_seed_dir: seedDir }).then(data => {
      if (study().id !== selectedId) return;
      agHistory = { studyId: selectedId, loading: false, error: null, data: data };
      window.EU_AGENT_RUN_HISTORY = data;
      if (window.__euRender) window.__euRender();
      else repaintBody();
    }).catch(err => {
      if (study().id !== selectedId) return;
      agHistory = { studyId: selectedId, loading: false, error: err.message || String(err), data: null };
      if (window.__euRender) window.__euRender();
      else repaintBody();
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
      agRun.error = t('Connection interrupted. If the server job is still running, resume the stream; otherwise continue from Guided Copilot.', '连接中断。如果服务端任务仍在运行，可恢复任务流；否则请回到研究引导继续。');
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
        errorRemedies: '',
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
    const host = document.getElementById('agHost');
    if (host && host.clientWidth > 0 && host.clientWidth < 1040) return true;
    return AG_FOCUS_TABS.has(agTab);
  }
  function studyList(studies, monitorState) {
    const dotCls = { ready: 'ready', running: 'running', gate: 'running', review_blocked: 'running', draft: 'draft', idle: 'idle' };
    const railState = {
      loading: [t('Checking local records…', '正在检查本地记录…'), t('Projects and runs will appear when the index is ready.', '索引就绪后会显示项目与运行。'), 'refresh'],
      error: [t('Project index unavailable', '项目索引暂不可用'), t('Retry from the main panel.', '请在主面板重试。'), 'alert'],
      empty: [t('Nothing to monitor yet', '暂时没有可监控内容'), t('Start a study in Guided Copilot.', '请先在研究引导中发起研究。'), 'folder'],
    }[monitorState];
    return `
    <div class="ag-list" id="agStudyList" aria-label="${t('Research project list', '研究项目列表')}">
      <div class="ag-list-head">
        <div><span class="ttl">${t('Projects', '项目')} · ${studies.length}</span><div class="ag-list-cap">${t('read-only status index', '只读状态索引')}</div></div>
      </div>
      ${studyListContext(studies)}
      <div class="ag-studies">
        ${railState ? `<div class="ag-list-state ${monitorState}" role="status"><span>${icon(railState[2], 16)}</span><div><strong>${railState[0]}</strong><p>${railState[1]}</p></div></div>` : ''}
        ${monitorState === 'ready' && agIdeaProjects.error ? `<div class="ag-list-sync-warning" role="status"><span>${icon('alert', 13)}</span><span>${t('Some project records may be missing.', '部分项目记录可能尚未载入。')}</span><button type="button" data-ag-refresh-projects>${t('Retry', '重试')}</button></div>` : ''}
        ${studies.map(s => {
          const zh = window.EU_LANG === 'zh';
          const folder = projectFolderLabel(s);
          const historyRows = historyRowsForStudy(s);
          const historyCount = monitorRunCount(s);
          const latest = historyRows[0];
          const runMeta = latest
            ? `${esc(latest.run_label || latest.run_id || t('local run', '本地运行'))}<span class="mid"></span>${esc((latest.updated_at || '').slice(0, 10) || t('local', '本地'))}`
            : historyCount === 0
              ? t('not run yet', '尚未运行')
              : s.id === agSel && agHistory.loading
                ? t('checking run history…', '正在检查运行历史…')
                : t('select to check run history', '选择后检查运行历史');
          const cardStatus = latest && (latest.gate_status === 'blocked' || latest.run_status === 'failed')
            ? 'review_blocked'
            : latest ? 'gate' : s.status;
          return `
          <button class="studycard ${s.id === agSel ? 'on' : ''}" data-ag-sel="${s.id}">
            <div class="sc-top">
              <span class="sc-dot ${dotCls[cardStatus] || 'idle'}"></span>
              <span class="sc-name">${esc(t(s.name[0], s.name[1]))}</span>
              <span class="sc-mode analysis">${!s.ideaSeed && !s.studyContext && !s.empty && !realMode() ? `${t('Example', '示例')} · ` : ''}${studyBadgeLabel(s)}</span>
            </div>
            <div class="sc-meta"><span class="sc-folder" title="${esc(folder)}">${icon('folder', 11)} ${esc(compactMiddlePath(folder))}</span></div>
            <div class="sc-meta" style="margin-top:3px;">${realMode() ? runMeta : (s.runs.length ? `${s.runs[0][0]}<span class="mid"></span>${s.runs[0][4][zh ? 1 : 0]}` : t('not run yet', '尚未运行'))}</div>
          </button>`;
        }).join('')}
      </div>
    </div>`;
  }

  function monitorBlankDetail(monitorState) {
    if (monitorState === 'loading') return `<div class="state-hero solid ag-monitor-state loading" role="status" aria-live="polite"><div class="glyph">${icon('refresh', 26)}</div><div class="st-t">${t('Loading projects', '正在加载项目')}</div><div class="st-d">${t('Checking this machine for governed project and run records.', '正在检查本机的受治理项目与运行记录。')}</div></div>`;
    if (monitorState === 'error') return `<div class="state-hero solid error ag-monitor-state" role="alert"><div class="glyph">${icon('alert', 26)}</div><div class="st-t">${t('Could not load local projects', '无法加载本地项目')}</div><div class="st-d">${t('The project index did not respond. Retry before relying on this monitor; no project state has been inferred.', '项目索引没有响应。请重试后再使用本监控页；当前不会推断任何项目状态。')}</div><div class="st-actions"><button class="btn primary" data-ag-refresh-projects>${icon('refresh', 13)} ${t('Retry', '重试')}</button><button class="btn" data-nav="guided">${icon('spark', 13)} ${t('Open Guided Copilot', '打开研究引导')}</button></div></div>`;
    return `<div class="state-hero solid empty-state ag-monitor-state"><div class="glyph">${icon('folder', 26)}</div><div class="st-t">${t('No projects to monitor yet', '还没有可监控的项目')}</div><div class="st-d">${t('Start a study in Guided Copilot. Its runs, outputs, evidence, and review status will appear here automatically.', '请先在研究引导中发起研究；运行、产出、证据和审阅状态会自动显示在这里。')}</div><div class="st-actions"><button class="btn primary" data-nav="guided">${icon('spark', 13)} ${t('Open Guided Copilot', '打开研究引导')}</button><button class="btn" data-ag-see-demo>${icon('flask', 13)} ${t('View completed example', '查看完整示例')}</button></div></div>`;
  }

  /* ---------------- detail header ---------------- */
  function detailHead() {
    const s = study();
    const live = reviewableRunForStudy();
    const review = currentLiveReview(live);
    const liveSigned = !!(review && review.signed);
    const compactHeader = agTab !== 'overview';
    const listCollapsed = agentListCollapsed();
    const persistedRows = historyRowsForStudy(s);
    const persistedCount = monitorRunCount(s);
    const persistedLatest = persistedRows[0];
    const statusKey = s.readOnlyImport ? 'imported'
      : liveSigned ? 'reviewed'
        : s.signed ? 'ready'
          : realMode() && persistedCount == null ? (agHistory.error ? 'history_error' : 'history_loading')
            : realMode() && persistedLatest ? ((persistedLatest.gate_status === 'blocked' || persistedLatest.run_status === 'failed') ? 'review_blocked' : 'gate')
              : s.status;
    const statusPill = {
      imported: `<span class="pill info"><span class="dot"></span>${t('Read-only review', '只读审阅')}</span>`,
      ready: `<span class="pill ok"><span class="dot"></span>${t('Ready in Copilot', '可在 Copilot 中继续')}</span>`,
      reviewed: `<span class="pill ok"><span class="dot"></span>${t('Signed analysis-only', '已签署 analysis-only')}</span>`,
      gate: `<span class="pill warn"><span class="dot"></span>${t('Awaiting sign-off', '待签署')}</span>`,
      review_blocked: `<span class="pill bad"><span class="dot"></span>${t('Evidence verification blocked', '证据核验受阻')}</span>`,
      running: `<span class="pill warn"><span class="dot"></span>${t('Running', '运行中')}</span>`,
      draft: `<span class="pill demo"><span class="dot"></span>${t('Exploring', '探索中')}</span>`,
      idle: `<span class="pill"><span class="dot"></span>${t('Not run yet', '尚未运行')}</span>`,
      history_loading: `<span class="pill info"><span class="dot"></span>${t('Checking run history', '正在检查运行历史')}</span>`,
      history_error: `<span class="pill bad"><span class="dot"></span>${t('Run history unavailable', '运行历史暂不可用')}</span>`,
    }[statusKey] || '';
    return `
    <div class="ag-dhead ${compactHeader ? 'compact' : ''}">
      <div class="ag-dtop">
        <div style="min-width:0;">
          <div class="ag-title">${esc(t(s.name[0], s.name[1]))}</div>
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
          <button class="btn sm" data-ag-guided>${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
          <span class="pill info"><span class="dot"></span>${t('Project monitor', '项目监控')}</span>
        </div>
      </div>
      ${compactHeader ? '' : pipeline()}
      ${compactHeader ? '' : agentTermStrip(s)}
    </div>`;
  }

  /* ---------------- tabs ---------------- */
  function tabsFor(mode) {
    const s = study();
    const runCount = monitorRunCount(s);
    if (s.empty) return [
      ['overview', t('Overview', '概览'), null],
    ];
    // Tab order follows the actual workflow: lead with what the user consumes
    // (Runs -> Outputs -> Draft), then the same run's deeper Evidence view.
    // The Evidence tab (id 'science') is the provenance deep-dive
    // of THIS study's run, not a separate app — see screens-agent-science.js.
    if (mode === 'idea') return [
      ['overview', t('Overview', '概览'), null],
      ['runs', t('Dry-runs', '试运行'), runCount],
      ['notes', t('Notes', '笔记'), null],
      ['science', t('Evidence', '证据'), null],
    ];
    return [
      ['overview', t('Overview', '概览'), null],
      ['runs', t('Runs', '运行历史'), runCount],
      ['outputs', t('Outputs', '产出'), outputCountForStudy()],
      ['draft', s.readOnlyImport ? t('Review', '审阅') : t('Draft', '草稿'), null],
      ['science', t('Evidence', '证据'), null],
    ];
  }
  // Tabs that only fill in after a run exists — flagged so a first-time user
  // isn't invited to click empty heroes one by one.
  const AG_RUN_GATED_TABS = new Set(['runs', 'outputs', 'draft', 'science']);
  function tabsRow() {
    const s = study();
    const tabs = tabsFor(s.mode);
    if (!tabs.some(x => x[0] === agTab)) agTab = 'overview';
    const noRun = monitorRunCount(s) === 0;
    return `<div class="ag-tabs" data-ag-tabs role="tablist" aria-label="${t('Project monitor views', '项目监控视图')}">
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
      return `
      <div class="nextbar gate">
        <div class="nb-ico">${icon('shield', 16)}</div>
        <div class="grow"><div class="nb-t">${agRun.result && agRun.result.cancelled ? t('Run cancelled safely', '运行已安全取消') : t('Run failed closed', '运行已 fail-closed')}</div><div class="nb-d">${esc(agRun.error)}</div>${agRun.errorRemedies || ''}</div>
        ${canReconnect ? `<button class="btn" data-ag-reconnect>${icon('history', 13)} ${t('Resume stream', '恢复任务流')}</button>` : ''}
        <button class="btn primary" data-ag-guided>${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
      </div>`;
    }
    if (seedGateBlocksRun(s)) {
      return `
      <div class="nextbar gate">
        <div class="nb-ico">${icon('shield', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Project readiness checks are not complete', '项目就绪检查尚未完成')}</div><div class="nb-d">${esc(seedGateBlockerText(s))}</div>${gateRemedyHtml(seedGateRemedies(s))}</div>
        <button class="btn" data-ag-guided>${icon('spark', 13)} ${t('Resolve in Copilot', '在 Copilot 中处理')}</button>
      </div>`;
    }
    if (s.mode === 'idea') {
      return `
      <div class="nextbar accent">
        <div class="nb-ico" style="background:oklch(52% 0.10 280);">${icon('play', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Feasibility work continues in Copilot', '可行性工作在 Copilot 中继续')}</div><div class="nb-d">${t('Requirements, data scope, and execution choices are collected conversationally; this page only monitors the resulting project.', '研究需求、数据范围与执行选择由 Copilot 对话收集；本页只监控形成的项目。')}</div></div>
        <button class="btn primary" data-ag-guided>${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
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
        <div class="grow"><div class="nb-t">${t('Ready to continue in Copilot', '可在 Copilot 中继续')}</div><div class="nb-d">${t('Confirm requirements, model connection, and the next governed action in the conversation. Progress and evidence return to this monitor.', '请在对话中确认研究需求、模型连接与下一项受治理操作；进度和证据会回到此监控页。')}</div></div>
        <button class="btn primary" data-ag-guided>${icon('spark', 13)} ${t('Open Copilot', '打开 Copilot')}</button>
      </div>`;
    }
    return `
      <div class="nextbar accent">
        <div class="nb-ico">${icon('refresh', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Project outputs are available to review', '项目产出可供审阅')}</div><div class="nb-d">${t('Use Copilot to extend or rerun the study; use this monitor for runs, outputs, evidence, and review.', '扩展或重跑研究请进入 Copilot；本页用于查看运行、产出、证据与审阅状态。')}</div></div>
        <button class="btn primary" data-ag-guided>${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
      </div>`;
  }

  function contextStats() {
    const s = study();
    const src = exportSourceForStudy(s);
    const sum = (src && src.summary) || {};
    const b = s.benchmark || null;
    const crossdbSelection = s.studyContext && s.studyContext.crossdb_selection || {};
    const selectedSources = Array.isArray(crossdbSelection.sources) ? crossdbSelection.sources : [];
    const crossScope = s.id === 'crossdb' || !!s.planOnly || selectedSources.length > 1;
    const selectedCount = Number.isInteger(crossdbSelection.source_count)
      ? crossdbSelection.source_count
      : selectedSources.length;
    const selectionDigest = typeof crossdbSelection.selection_digest === 'string'
      ? crossdbSelection.selection_digest.slice(0, 12)
      : '';
    // Real mode must never show invented clinical numbers: if no benchmark and
    // no attached export, fall to em-dashes + an "attach an export" hint rather
    // than the seeded demo figures (which are only honest as a demo preview).
    const noData = !b && !src;
    const stats = b
      ? [['Cohort', '队列', b.cohort_size == null ? '—' : Number(b.cohort_size).toLocaleString()], ['Evidence', '证据', b.evidence_count == null ? '—' : Number(b.evidence_count).toLocaleString()], ['Warnings', '警告', b.warnings == null ? '—' : String(b.warnings)], ['Status', '状态', runStatusLabel(b.tristate || b.readiness_status || 'analysis_only')]]
      : src
      ? [['Stays', '住院数', sum.stays == null ? '—' : Number(sum.stays).toLocaleString()], ['Modules', '模块', sum.modules == null ? '—' : String(sum.modules)], ['Rows', '行数', sum.total_rows == null ? '—' : Number(sum.total_rows).toLocaleString()], ['Evidence check', '证据核验', 'strict']]
      : (noData && realMode())
      ? (crossScope
        ? [['Selected exports', '选中导出', selectedCount || '—'], ['Modules', '模块', s.studyContext && Array.isArray(s.studyContext.modules) ? s.studyContext.modules.length : '—'], ['Selection digest', '选择摘要', selectionDigest || '—'], ['Execution', '执行', t('plan-only', '仅计划')]]
        : [['Mean age', '平均年龄', '—'], ['Mortality', '死亡率', '—'], ['Sepsis-3', 'Sepsis-3', '—'], ['Mech vent', '机械通气', '—']])
      : s.id === 'crossdb'
      ? [['Databases', '数据库', '3'], ['Shared concepts', '共享概念', '6'], ['Mortality', '死亡率', '20.0%'], ['Concordance', '一致性', 'high']]
      : [['Mean age', '平均年龄', '54.8 y'], ['Mortality', '死亡率', '20.0%'], ['Sepsis-3', 'Sepsis-3', '45.3%'], ['Mech vent', '机械通气', '52.1%']];
    const noDataHint = noData && realMode() && !crossScope;
    const receiptOnly = noData && realMode() && crossScope && selectedSources.length > 1;
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
      ${s.readOnlyImport ? `<div class="note ok mt-12"><div class="ico">${icon('shield', 13)}</div><div class="body"><div class="t">${t('Completed local result', '已完成的本地结果')}</div><div class="d">${esc(s.sourceArticle || t('Read-only research result with its original evidence bundle', '带原始证据包的只读研究结果'))}</div></div></div>` : (s.ideaSeed ? `<div class="note ok mt-12"><div class="ico">${icon('target', 13)}</div><div class="body"><div class="t">${t('Created from Idea Mining', '来自 Idea Mining 的研究想法')}</div><div class="d">${esc(s.sourceArticle || '')}</div></div></div>` : '')}
      <div class="cols-2 mt-12" style="gap:8px;">
        ${stats.map(([en, zh, v]) => `
          <div style="padding:8px 10px;background:var(--surface-2);border-radius:var(--r-2);">
            <div class="eyebrow" style="font-size:9.5px;">${t(en, zh)}</div>
            <div class="mono" style="font-size:13px;font-weight:500;color:var(--ink);margin-top:3px;">${v}</div>
          </div>`).join('')}
      </div>
      ${receiptOnly
        ? `<div class="note info mt-12"><div class="ico">${icon('shield', 13)}</div><div class="body"><div class="t">${t('Cross-DB selection receipt bound', '已绑定 Cross-DB 选择收据')}</div><div class="d">${t('No single export path or stay count is substituted for this multi-source plan.', '这个多来源计划不会再用单一导出路径或住院数代替研究范围。')}</div></div></div>`
        : noDataHint
        ? `<div class="note info mt-12"><div class="ico">${icon('folder', 13)}</div><div class="body"><div class="t">${t('No export attached', '未关联导出')}</div><div class="d">${t('Attach a local EasyICU export to this project to populate real cohort figures.', '为此项目关联本地 EasyICU 导出后，这里会显示真实队列数据。')}</div></div></div>`
        : (noData ? `<div class="note warn mt-12" style="padding:8px 11px;"><div class="ico">${icon('beaker', 13)}</div><div class="body"><div class="d" style="margin:0;">${t('Illustrative demo figures — not a computed result.', '示例演示数据 —— 非计算结果。')}</div></div></div>` : '')}
      <button class="btn sm block mt-16" data-nav="${crossScope ? 'crossdb' : 'extraction'}">${icon('layers', 13)} ${crossScope ? t('Open Cross-DB workspace', '打开跨库工作台') : t('Open in Data Extraction', '在数据抽取中打开')}</button>
    </div>`;
  }

  function importedResultSummary(s) {
    if (!s || !s.readOnlyImport) return '';
    const b = s.benchmark || {};
    const artifactCount = outputCountForStudy();
    const status = runStatusLabel(b.tristate || b.readiness_status || 'analysis_only');
    return `
      <div class="ag-present-brief">
        <div class="ag-present-main">
          <div class="eyebrow">${t('Study brief', '汇报摘要')}</div>
          <div class="ag-present-title">${esc(t(s.name[0], s.name[1]))}</div>
          <div class="ag-present-text">${t('A completed evidence-bound research result. The scientific question, plan, outputs, and auditable evidence ledger remain together in this read-only project.', '这是一项已完成的证据绑定研究结果。科学问题、计划、产物和可审计证据账本在同一个只读项目中保留。')}</div>
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
    const live = reviewableRunForStudy();
    if (live) requestLiveReview(live);
    const staleBanner = (window.EU_STALE && !agRun.active) ? `
      <div class="stale-banner">
        <span class="sb-ico">${icon('refresh', 16)}</span>
        <div class="grow"><div class="sb-t">${t('Extraction changed since the last run', '自上次运行后抽取已变更')}</div><div class="sb-d">${t('The cohort or modules were edited — runs, outputs and the draft are out of date until you re-run.', '队列或模块被修改 — 运行、产出和草稿在重跑前都已过期。')}</div></div>
        <button class="btn primary" data-ag-guided>${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
      </div>` : '';
    return `
      ${staleBanner}
      ${renderStructuredQuestion(s)}
      ${importedResultSummary(s)}
      ${capabilityHighlights(live, s)}
      ${nextBar()}
      ${s.mode === 'idea' ? `<div class="idea-band mt-16"><span class="ico">${icon('spark', 16)}</span><div><div style="font-weight:600;font-size:13px;">${t('Legacy feasibility idea', '旧可行性想法')}</div><div style="font-size:12px;color:var(--ink-3);margin-top:2px;">${t('Continue discovery and configuration in Copilot; this monitor preserves the resulting status and evidence.', '请在 Copilot 中继续发现与配置；本监控页保留形成的状态与证据。')}</div></div></div>` : ''}
      <div class="split-320 mt-16" style="grid-template-columns:1fr 300px;">
        <div class="col gap-16">
          ${planList()}
        </div>
        ${contextStats()}
      </div>
      <div class="handoff">
        <span class="ho-ico">${icon('spark', 17)}</span>
        <div class="ho-body"><b>${t('Requirements and execution live in Guided Copilot.', '研究需求与执行入口统一在研究引导中。')}</b> ${t('Use this page for project status, runs, outputs, evidence, and review; return to Copilot for any next action.', '本页只查看项目状态、运行、产出、证据与审阅；任何下一步操作请回到 Copilot。')}</div>
        <button class="btn" data-ag-guided>${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')} ${icon('arrow', 13)}</button>
      </div>
      ${agGuidedHandoffError ? `<div class="note warn mt-8" role="alert"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="t">${t('Guided Copilot handoff failed', '研究引导交接失败')}</div><div class="d">${esc(agGuidedHandoffError)}</div></div></div>` : ''}`;
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
        ${!rows.length && !(agHistory.studyId === s.id && agHistory.loading) ? `<div class="state-hero empty-state"><div class="glyph">${icon('history', 26)}</div><div class="st-t">${t('No local runs found', '未找到本地运行')}</div><div class="st-d">${t('Start or resume the governed task in Guided Copilot. History is rebuilt here from local artifacts, not browser memory.', '请在研究引导中发起或恢复受治理任务；本页运行历史由本地产物重建，不依赖浏览器内存。')}</div><div class="st-actions"><button class="btn primary" data-ag-guided>${icon('spark', 14)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button></div></div>` : ''}
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
      return `<div class="state-hero empty-state"><div class="glyph">${icon('history', 26)}</div><div class="st-t">${t('No runs yet', '尚无运行记录')}</div><div class="st-d">${t('Use Guided Copilot to confirm the study and start the governed run. Every returned run writes a local manifest for this monitor.', '请在研究引导中确认研究并发起受治理运行；返回的每次运行都会为本监控页写入本地清单。')}</div><div class="st-actions"><button class="btn primary" data-ag-guided>${icon('spark', 14)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button></div></div>`;
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
              <button class="btn sm ghost" data-ag-guided title="${t('Continue this study in Copilot', '在 Copilot 中继续此研究')}">${icon('spark', 12)} ${t('Copilot', 'Copilot')}</button>
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
          <div class="ag-output-brief-text">${t('Figures, the evaluation scorecard, and the evidence ledger are surfaced first. File names and hashes stay visible as provenance, not as the main story.', '图件、评估记分卡和证据账本被放在前面。文件名与哈希保留为溯源信息，不再作为主展示内容。')}</div>
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
          <div class="st-d">${isImportedRun(live, s) ? t('The imported review package is registered, but the local artifact scan did not return whitelisted files. Open Run history to inspect the source folder.', '已注册导入审阅包，但本地产物扫描没有返回白名单文件。请打开运行历史检查来源文件夹。') : t('This project has not produced Table 1, missingness, ROC, calibration, or evidence files yet. Continue the study in Copilot or open a reviewed local run; placeholders are not shown in Real mode.', '这个项目还没有生成 Table 1、缺失审计、ROC、校准或证据文件。请在 Copilot 中继续研究，或打开已有本地运行；真实模式不会显示占位产物。')}</div>
          <div class="st-actions">
            <button class="btn primary" data-ag-guided>${icon('spark', 14)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
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
          <button class="btn primary" data-ag-guided>${icon('spark', 14)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
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
          <button class="btn primary" data-ag-guided>${icon('spark', 13)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
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
          <div class="panel-sub">${imported ? t('This is a completed historical analysis opened for presentation and evidence review. It is not a new Agent run, and it will not unlock a reportable manuscript draft.', '这是为展示和证据审阅打开的历史分析结果。它不是新的 Agent 运行，也不会解锁可报告论文草稿。') : t('This real run is analysis_only. It wrote bounded local evidence artifacts; human sign-off records review but does not make the draft reportable.', '这次真实运行是 analysis_only。它写入有界本地证据产物；人工签署只记录审阅,不会让草稿可报告。')}</div>
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
            <div style="font-size:11.5px;color:var(--ink-3);margin-bottom:10px;">${t('The review package centers on output cards, the figure gallery, the evaluation record, and provenance.', '审阅包围绕产出卡片、图件画廊、评估记录和溯源记录组织。')}</div>
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
        <div class="st-d">${t('Draft review works on a real local run. Start or resume the analysis in Guided Copilot, then inspect its evidence checks here. Demo placeholder checks are not shown in Real mode.', '草稿审阅基于真实本地运行。请在研究引导中发起或恢复分析，再到这里检查证据；真实模式不会显示演示占位校验。')}</div>
        <div class="st-actions">
          <button class="btn primary" data-ag-guided>${icon('spark', 14)} ${t('Continue in Guided Copilot', '在研究引导中继续')}</button>
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
    const studies = allStudies();
    const monitorState = monitorViewState(studies);
    const listCollapsed = agentListCollapsed();
    if (monitorState !== 'ready') return `
    <div class="ag-wrap ag-wrap-blank ${listCollapsed ? 'list-collapsed' : 'list-open'}" data-ag-monitor-state="${monitorState}" data-ag-list-state="${listCollapsed ? 'collapsed' : 'open'}">
      ${studyList(studies, monitorState)}
      <div class="ag-detail ag-detail-blank">${monitorBlankDetail(monitorState)}</div>
    </div>`;
    return `
    <div class="ag-wrap ${listCollapsed ? 'list-collapsed' : 'list-open'}" data-ag-list-state="${listCollapsed ? 'collapsed' : 'open'}">
      ${studyList(studies, monitorState)}
      <div class="ag-detail">
        ${detailHead()}
        ${tabsRow()}
        <div class="ag-body" id="agTabPanel" role="tabpanel" aria-labelledby="agTab-${agTab}" tabindex="0">${tabBody()}</div>
      </div>
    </div>`;
  }

  /* ---------------- run monitoring ---------------- */
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
        ? `${t('Cancelled at', '取消阶段')}: ${result.cancelled_at || 'agent_run'} · ${t('continue safely from Guided Copilot.', '请从研究引导安全继续。')}`
        : (error || t('Agent run cancelled.', 'Agent 运行已取消。'));
      s.status = 'idle';
    } else {
      agRun.error = error || t('Agent run failed.', 'Agent 运行失败。');
      agRun.result = result || null;
      agRun.reconnectable = false;
      s.status = 'idle';
      if (study().id === runToken.study_id) window.EU_AGENT_LAST_RUN = null;
    }
    agRunChannel.clear(runToken);
    agHistory = { studyId: null, loading: false, error: null, data: null };
    requestRunHistory(true);
    repaintBody();
  }

  /* ---------------- wiring ---------------- */
  function wire(root) {
    const host = root.querySelector('#agHost'); if (!host) return;
    host.querySelectorAll('[data-ag-sel]').forEach(b => b.addEventListener('click', () => {
      closeRunStream();
      agRunChannel.clear();
      agRun = { active: false, prog: 0, timer: null, es: null, jobId: null, step: null, error: null, errorRemedies: '', result: null, warning: null };
      agResumeProbe = { loading: false, checkedJobId: null };
      window.EU_AGENT_RUN_REVIEW = null;
      agSel = b.dataset.agSel;
      agGuidedHandoffError = '';
      if (window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.has(agSel)) {
        window.EU_AGENT_STUDY_CONTEXT.activate(agSel).catch(error => console.warn('[EasyICU] StudyContext activation failed:', error));
      }
      agTab = 'overview';
      requestRunHistory();
      repaintBody(); maybeRestoreAgentJob(); focusAgentBody();
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
    host.querySelectorAll('[data-ag-toggle-list]').forEach(b => b.addEventListener('click', () => {
      agListMode = agentListCollapsed() ? 'open' : 'focus';
      repaintBody();
      focusAgentBody();
    }));
    if (window.EU_AGENT_SCIENCE) {
      window.EU_AGENT_SCIENCE.wire(root, { live: reviewableRunForStudy(), study: study(), repaint: repaintBody });
    }
    host.querySelectorAll('[data-ev]').forEach(b => b.addEventListener('click', () => { const i = +b.dataset.ev; agEvOpen = (agEvOpen === i ? -1 : i); repaintBody(); }));
    host.querySelectorAll('[data-ag-guided]').forEach(b => b.addEventListener('click', () => {
      const adapter = window.EU_AGENT_STUDY_CONTEXT;
      if (!adapter || !adapter.prepareGuidedHandoff) return;
      const selected = study();
      const selectedId = selected.id;
      agGuidedHandoffError = '';
      b.disabled = true;
      adapter.prepareGuidedHandoff(selected).then(() => {
        if (study().id === selectedId) location.hash = '#guided';
      }).catch(error => {
        if (study().id !== selectedId) return;
        agGuidedHandoffError = t(
          `Could not bind this project to Guided Copilot: ${error.message || error}`,
          `无法将当前项目绑定到研究引导：${error.message || error}`,
        );
        repaintBody();
      });
    }));
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
  }

  S.agent = {
    section: 'agent', nav: 'agent',
    wide: true,
    get crumbs() { return [t('Home', '首页'), t('Project Monitor', '项目监控')]; },
    /* No per-screen guide action: the removed button opened the SAME dock as
       the topbar 'Page guide' button beside it, reading as a second
       agent-specific help system that didn't exist. The dock already greets
       agent-specifically via its CTX.agent entry. */
    rail() {
      const s = study();
      const studies = allStudies();
      const monitorState = monitorViewState(studies);
      const count = monitorState === 'loading' || monitorState === 'error' ? '—' : studies.length;
      const selectedRunCount = monitorRunCount(s);
      const summary = monitorState === 'ready'
        ? `<div class="col gap-6" style="font-size:12px;"><div class="setup-row"><span class="k">${t('Active', '当前')}</span><span class="vv" style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(t(s.name[0], s.name[1]))}</span></div><div class="setup-row"><span class="k">${t('Mode', '模式')}</span><span class="vv">${t('Analysis', '分析')}</span></div><div class="setup-row"><span class="k">${t('Evidence', '证据')}</span><span class="vv">${s.gate ? t('available', '可查看') : '—'}</span></div><div class="setup-row"><span class="k">${t('Runs', '运行')}</span><span class="vv">${selectedRunCount == null ? '—' : selectedRunCount}</span></div></div>`
        : `<div class="note ${monitorState === 'error' ? 'warn' : 'info'}" style="padding:8px 10px;"><div class="body"><div class="t">${monitorState === 'loading' ? t('Checking project index', '正在检查项目索引') : monitorState === 'error' ? t('Project index unavailable', '项目索引暂不可用') : t('No monitored project yet', '还没有可监控项目')}</div></div></div>`;
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Projects', '项目')}</span><span class="pill ${monitorState === 'error' ? 'warn' : 'ok'}" style="height:20px;"><span class="dot"></span>${count}</span></div>
        ${summary}
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
            <div class="eyebrow">${t('Project Monitor · 项目监控', '项目监控 · Project Monitor')}</div>
            <h1 style="margin-top:6px;">${t('Research Project Monitor', '研究项目监控')}</h1>
            <p class="lead">${t('Read-only project state plus governed review actions: plans, runs, outputs, evidence, and gates. Requirements, model setup, and run initiation stay in Guided Copilot.', '以只读方式查看项目状态，并执行受治理的审阅操作：计划、运行、产出、证据与闸门。研究需求、模型配置和任务发起统一留在研究引导中。')}</p>
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
      requestRunHistory();
      maybeRestoreAgentJob();
    },
  };

  if (window.EU_AGENT_STUDY_CONTEXT && window.EU_AGENT_STUDY_CONTEXT.subscribe) {
    window.EU_AGENT_STUDY_CONTEXT.subscribe(context => {
      if (!context || !context.id) return;
      if (agSel && agSel !== context.id) {
        closeRunStream();
        agRunChannel.clear();
        agRun = { active: false, prog: 0, timer: null, es: null, jobId: null, step: null, error: null, errorRemedies: '', result: null, warning: null };
        agResumeProbe = { loading: false, checkedJobId: null };
      }
      agSel = context.id;
      if (location.hash === '#agent') {
        requestRunHistory();
        repaintBody();
      }
    });
  }
})();
